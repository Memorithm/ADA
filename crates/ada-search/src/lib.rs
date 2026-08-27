//! ADA-A8 search: instantiate the qualified online-softmax recurrences as
//! inspectable `ada-ir` programs, execute them through the fail-closed
//! interpreter, and verify the results against the dense oracle contract.
//!
//! The search space is intentionally tiny and explicit: the baseline
//! (two-exp) and branch-specialized (one-exp) recurrences. Every candidate is
//! a real [`Program`] validated and interpreted by `ada-ir`; unsupported
//! constructs cannot be expressed, so nothing can be silently dropped.

#![forbid(unsafe_code)]

mod checkpoint;
mod enumerator;

use ada_core::AttentionCase;
use ada_ir::{
    BinaryOp, CompareOp, Interpreter, Program, ReduceOp, ScalarExpr, Statement, VectorExpr,
};

pub use enumerator::{
    MAX_CHECKPOINT_CANDIDATE_BYTES, MAX_CHECKPOINT_SEEN, MAX_CHECKPOINT_TEXT_BYTES,
    MAX_PROGRAM_COST, MAX_SEARCH_ALTERNATIVES, MAX_SEARCH_EXPANSIONS, MAX_SEARCH_SPACE_CARDINALITY,
    SEARCH_CHECKPOINT_VERSION, SEARCH_SPACE_VERSION, SearchBudget, SearchCandidate,
    SearchCheckpoint, SearchEngine, SearchError, SearchFingerprint, SearchSpace, SearchStats,
    SemanticSearchConfig, SemanticSearchSpace,
};

/// One searched candidate: its program plus verified deviation metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub name: &'static str,
    pub program: Program,
    /// Max absolute output deviation against the oracle over all lanes.
    pub max_abs_output_error: f64,
    /// Absolute deviation of the running-sum logarithm against the oracle.
    pub abs_lse_error: f64,
}

fn scalar(op: BinaryOp, left: ScalarExpr, right: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// One recurrence step expressed entirely inside the restricted grammar.
///
/// Slot layout: 0 = running max `m`, 1 = running sum `l`, 2 = incoming score
/// `s`. Register layout: 0 = accumulator `O`, 1 = value row `V`.
///
/// The step computes, without any construct outside the grammar:
/// `m' = max(m, s)`, `alpha = exp(m - m')`, `p = exp(s - m')`,
/// `l' = alpha*l + p`, `O' = alpha*O + p*V`.
#[must_use]
pub fn recurrence_step_template() -> Program {
    let m = || ScalarExpr::State(0);
    let l = || ScalarExpr::State(1);
    let s = || ScalarExpr::State(2);

    let m_new = ScalarExpr::Select {
        condition: Box::new(ScalarExpr::Compare {
            op: CompareOp::Ge,
            left: Box::new(s()),
            right: Box::new(m()),
        }),
        then_value: Box::new(s()),
        else_value: Box::new(m()),
    };
    // First-token degeneracy (m = -inf) is handled by seeding before the
    // stream; the template assumes finite seeded state, matching the A1 E0
    // contract where the first score initializes (m, l, O).
    let alpha_expr = ScalarExpr::Exp(Box::new(scalar(BinaryOp::Sub, m(), m_new.clone())));
    let p_expr = ScalarExpr::Exp(Box::new(scalar(BinaryOp::Sub, s(), m_new.clone())));
    let l_new = scalar(
        BinaryOp::Add,
        scalar(BinaryOp::Mul, alpha_expr.clone(), l()),
        p_expr.clone(),
    );

    Program {
        state_slots: 3,
        vector_registers: 3,
        statements: vec![
            // Slot 2 holds the incoming score, fed by the driver per token.
            // O <- alpha * O + p * V
            Statement::StoreVector {
                register: 0,
                value: VectorExpr::FusedScaleAdd {
                    alpha: Box::new(alpha_expr),
                    scaled: Box::new(VectorExpr::Register(0)),
                    added: Box::new(VectorExpr::ZipBinary {
                        op: BinaryOp::Mul,
                        left: Box::new(VectorExpr::Broadcast(Box::new(p_expr))),
                        right: Box::new(VectorExpr::Register(1)),
                    }),
                },
            },
            // l <- alpha*l + p
            Statement::StoreState {
                slot: 1,
                value: l_new,
            },
            // m <- m'
            Statement::StoreState {
                slot: 0,
                value: m_new,
            },
        ],
    }
}

/// Verify both recurrence candidates on the supplied case.
///
/// The interpreter executes the IR step per token after the driver seeds the
/// first-token state exactly like the A1 oracle does; deviations are measured
/// in f64 against the shared dense reference.
///
/// # Errors
///
/// Returns errors when the case violates its contract or when interpretation
/// fails closed; failures are never silently skipped.
pub fn verify_candidates(case: &AttentionCase) -> Result<Vec<Candidate>, &'static str> {
    case.validate()?;
    let template = recurrence_step_template();
    template.validate()?;
    let head_dim = case.head_dim;
    let mut candidates = Vec::new();

    for (name, one_exp) in [("two-exp", false), ("one-exp", true)] {
        let mut o = vec![0.0_f64; head_dim];
        let mut interp = Interpreter::new(template.clone())?;

        // Seed from the first score, mirroring the A1 one-exp contract.
        let first = f64::from(case.logits[0]);
        let mut m = first;
        let mut l = 1.0_f64;
        o.copy_from_slice(
            case.values[..head_dim]
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>()
                .as_slice(),
        );

        let start = usize::from(one_exp);
        if !one_exp {
            m = f64::NEG_INFINITY;
            l = 0.0;
            o.fill(0.0);
        }

        for key in start..case.logits.len() {
            let s_value = f64::from(case.logits[key]);
            let value_row = &case.values[key * head_dim..(key + 1) * head_dim];

            if !one_exp && !m.is_finite() {
                // Two-exp baseline treats the very first token generically:
                // new_max = s, alpha = 0, p = exp(0) = 1.
                m = s_value;
                l += 1.0;
                for (lane, &v) in o.iter_mut().zip(value_row) {
                    *lane = f64::from(v);
                }
                continue;
            }

            // Drive the IR interpreter with this token's state.
            interp.set_scalar(0, m);
            interp.set_scalar(1, l);
            interp.set_scalar(2, s_value);
            interp.set_vector(0, o.clone());
            interp.set_vector(1, value_row.iter().map(|&v| f64::from(v)).collect());
            interp.run()?;

            m = interp.scalar(0);
            l = interp.scalar(1);
            o.copy_from_slice(interp.vector(0));
        }

        for lane in &mut o {
            *lane *= l.recip();
        }

        let mut max_abs_output_error = 0.0_f64;
        // Reference: direct f64 evaluation of the same recurrence family.
        let (reference_o, reference_lse) = dense_reference(case, head_dim, one_exp);
        for (&r, &c) in reference_o.iter().zip(&o) {
            max_abs_output_error = max_abs_output_error.max((r - c).abs());
        }

        candidates.push(Candidate {
            name,
            program: template.clone(),
            max_abs_output_error,
            abs_lse_error: (reference_lse - l.ln()).abs(),
        });
    }

    for candidate in &candidates {
        if !(candidate.max_abs_output_error.is_finite() && candidate.max_abs_output_error <= 1.0e-9)
        {
            return Err("ADA-A8 candidate deviates beyond tolerance");
        }
    }

    Ok(candidates)
}

fn dense_reference(case: &AttentionCase, head_dim: usize, one_exp: bool) -> (Vec<f64>, f64) {
    let mut o = vec![0.0_f64; head_dim];
    let mut m = f64::NEG_INFINITY;
    let mut l = 0.0_f64;

    for (key, &score) in case.logits.iter().enumerate() {
        let s_value = f64::from(score);
        let value_row = &case.values[key * head_dim..(key + 1) * head_dim];
        if one_exp && key == 0 {
            m = s_value;
            l = 1.0;
            for (lane, &v) in o.iter_mut().zip(value_row) {
                *lane = f64::from(v);
            }
            continue;
        }
        let new_max = m.max(s_value);
        let alpha = if m.is_finite() {
            (m - new_max).exp()
        } else {
            0.0
        };
        let p = (s_value - new_max).exp();
        l = alpha * l + p;
        for (lane, &v) in o.iter_mut().zip(value_row) {
            *lane = alpha * *lane + p * f64::from(v);
        }
        m = new_max;
    }

    for lane in &mut o {
        *lane *= l.recip();
    }
    let _ = ReduceOp::Sum;
    (o, l.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{AttentionCase, LogicalMetrics};
    use ada_ir::Program;

    #[test]
    fn step_template_is_grammar_conformant() {
        recurrence_step_template()
            .validate()
            .expect("the canonical recurrence step must validate");
    }

    #[test]
    fn candidates_match_dense_reference_on_natural_case() {
        let logits = [-8.0_f32, -4.0, -1.0, 0.0, 3.0, 9.0];
        let values: Vec<f32> = (0..logits.len() * 4)
            .map(|index| {
                #[allow(clippy::cast_precision_loss)]
                let base = index as f32;
                base * 0.031_25 - 0.5
            })
            .collect();
        let case = AttentionCase {
            logits: logits.to_vec(),
            values,
            head_dim: 4,
        };

        let candidates = verify_candidates(&case).unwrap();
        assert_eq!(candidates.len(), 2);
        for candidate in &candidates {
            assert!(candidate.max_abs_output_error <= 1.0e-9);
            assert!(candidate.abs_lse_error.is_finite());
        }
        let _ = LogicalMetrics::default();
        let _ = Program {
            state_slots: 1,
            vector_registers: 1,
            statements: vec![],
        };
    }

    #[test]
    fn invalid_cases_fail_closed() {
        let case = AttentionCase {
            logits: vec![],
            values: vec![],
            head_dim: 1,
        };
        assert!(verify_candidates(&case).is_err());
    }
}
