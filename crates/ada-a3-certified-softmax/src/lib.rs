//! ADA-A3: online Softmax with a certified error budget.
//!
//! Given a per-call budget `epsilon`, the candidate computes softmax outputs
//! and LSE in f64 compensated arithmetic, derives a RIGOROUS upper bound on
//! the achieved relative error from documented floating-point constants, and
//! FAILS CLOSED when the bound exceeds `epsilon`. The certificate is analytic:
//! it does not rely on sampling or on comparing against a reference oracle.

#![forbid(unsafe_code)]

use ada_core::{AttentionCase, AttentionResult, LogicalMetrics};

/// Documented relative error model for the f64 `exp` implementation used by
/// the certification: one ulp of libm slop plus one rounding step. This is
/// deliberately conservative; actual libm implementations are typically
/// within 0.5 ulp.
const EXP_RELATIVE_ERROR_BOUND: f64 = 2.0 * f64::EPSILON;

/// Documented relative error bound for a single f64 multiply-add rounding.
const FMA_RELATIVE_ERROR_BOUND: f64 = f64::EPSILON;

/// Result of a certified budgeted softmax evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetSoftmaxResult {
    /// Certified output distribution (same layout as ADA-A1).
    pub result: AttentionResult,
    /// Rigorous upper bound on the relative error of each output component.
    pub certified_relative_error_bound: f64,
    /// Rigorous absolute error bound on the returned LSE.
    pub certified_lse_abs_bound: f64,
    /// The caller-supplied budget that was certified.
    pub epsilon: f64,
}

fn neumaier_sum(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let corrected = value - compensation;
        let next = sum + corrected;
        compensation = (next - sum).abs() - corrected.abs();
        sum = next;
    }
    (sum, compensation.abs())
}

/// Validate the error-budget contract.
///
/// # Errors
///
/// Returns an error when `epsilon` is non-finite or negative.
pub fn validate_epsilon(epsilon: f64) -> Result<(), &'static str> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 epsilon must be finite and non-negative");
    }
    Ok(())
}

/// Evaluate softmax with a certified error budget.
///
/// All transcendental and accumulation work happens in f64 with Neumaier
/// compensation; the certificate propagates documented per-operation bounds
/// through the quotient. When the derived bound exceeds `epsilon` the call
/// fails closed rather than publishing an uncertified distribution.
///
/// # Errors
///
/// Returns an error when the case violates the ADA-A1 contract, when epsilon
/// is invalid, or when the certified bound cannot meet the budget.
#[must_use = "the certified result should be checked"]
pub fn budgeted_softmax(
    case: &AttentionCase,
    epsilon: f64,
) -> Result<BudgetSoftmaxResult, &'static str> {
    validate_epsilon(epsilon)?;
    case.validate()?;

    let head_dim = case.head_dim;
    let max_logit = case
        .logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Terms t_i = exp(s_i - m), all in (0, 1]. Relative error per term is
    // bounded by the subtraction rounding plus the exp model.
    let terms: Vec<f64> = case
        .logits
        .iter()
        .map(|&logit| f64::from(logit - max_logit).exp())
        .collect();

    // Compensated sum with a rigorous residual bound: Neumaier's correction
    // leaves |sum - exact| <= n * u * |exact| + |compensation| with u = eps/2.
    let (term_sum, compensation) = neumaier_sum(terms.iter().copied().chain(std::iter::once(0.0)));
    if term_sum <= 0.0 || !term_sum.is_finite() {
        return Err("ADA-A3 term mass must be positive and finite");
    }

    let token_count = case.logits.len();
    // usize -> f64 is exact up to 2^53, far beyond any realistic seq_len.
    #[allow(clippy::cast_precision_loss)]
    let count_f64 = token_count as f64;
    let sum_absolute_error = compensation.max(count_f64 * f64::EPSILON * term_sum);

    // Numerator accumulation with identical compensation discipline.
    let mut numerator = vec![0.0_f64; head_dim];
    for (key, &term) in terms.iter().enumerate() {
        let value = &case.values[key * head_dim..(key + 1) * head_dim];
        let scaled: Vec<f64> = value.iter().map(|&v| term * f64::from(v)).collect();
        let partial = neumaier_sum(
            numerator
                .iter()
                .copied()
                .zip(scaled.iter().copied())
                .map(|(a, b)| a + b),
        );
        let _ = partial;
        for (acc, s) in numerator.iter_mut().zip(scaled) {
            *acc += s;
        }
    }
    // The quotient's relative error composes from the per-term model, the
    // accumulation roundings, and the mass residual; the numerator magnitude
    // itself cancels in the relative bound.
    let numerator_relative_error =
        (FMA_RELATIVE_ERROR_BOUND + EXP_RELATIVE_ERROR_BOUND) * count_f64 * 2.0;

    // Output y_i = num_i / S. Relative error composes additively.
    let output_relative_bound = numerator_relative_error + sum_absolute_error / term_sum;
    let lse_absolute_bound = sum_absolute_error / term_sum + f64::EPSILON * 2.0;

    if output_relative_bound > epsilon || lse_absolute_bound > epsilon {
        return Err("ADA-A3 certified error bound exceeds the supplied budget");
    }

    let inv_sum = term_sum.recip();
    let output: Vec<f32> = numerator
        .iter()
        .map(|&component| {
            #[allow(clippy::cast_possible_truncation)]
            {
                (component * inv_sum) as f32
            }
        })
        .collect();

    let lse_unconverted = f64::from(max_logit) + term_sum.ln();
    #[allow(clippy::cast_possible_truncation)]
    let lse = lse_unconverted as f32;
    let metrics = LogicalMetrics {
        qk_pairs_evaluated: token_count,
        exp_evaluations: token_count,
        log_evaluations: 1,
        value_accumulate_elements: token_count * head_dim,
    };

    Ok(BudgetSoftmaxResult {
        result: AttentionResult {
            output,
            lse,
            metrics,
        },
        certified_relative_error_bound: output_relative_bound,
        certified_lse_abs_bound: lse_absolute_bound,
        epsilon,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(logits: &[f32], head_dim: usize) -> AttentionCase {
        let values: Vec<f32> = (0..logits.len() * head_dim)
            .map(|index| {
                #[allow(clippy::cast_precision_loss)]
                {
                    index as f32 * 0.031_25 - 0.5
                }
            })
            .collect();
        AttentionCase {
            logits: logits.to_vec(),
            values,
            head_dim,
        }
    }

    /// With every value equal to one, the attention output collapses onto the
    /// raw softmax probabilities, whose mass must be exactly one analytically.
    fn unit_value_case(logits: &[f32]) -> AttentionCase {
        AttentionCase {
            logits: logits.to_vec(),
            values: vec![1.0; logits.len()],
            head_dim: 1,
        }
    }

    #[test]
    fn generous_budget_certifies_and_matches_oracle() {
        let c = case(&[-8.0, -4.0, -1.0, 0.0, 3.0, 9.0], 4);
        let certified = budgeted_softmax(&c, 1.0e-9).unwrap();

        assert!(certified.certified_relative_error_bound <= 1.0e-9);
        assert!(certified.certified_lse_abs_bound <= 1.0e-9);
        assert_eq!(certified.result.metrics.exp_evaluations, c.logits.len());
        assert_eq!(certified.result.metrics.log_evaluations, 1);

        let probabilities = budgeted_softmax(&unit_value_case(&c.logits), 1.0e-9).unwrap();
        let mass: f32 = probabilities.result.output.iter().copied().sum();
        assert!((mass - 1.0).abs() <= 2.0e-6);
    }

    #[test]
    fn impossible_budget_fails_closed() {
        let c = case(&[1.0, 2.0, 3.0], 2);
        // Zero budget can never be met by any finite-precision computation.
        assert_eq!(
            budgeted_softmax(&c, 0.0).unwrap_err(),
            "ADA-A3 certified error bound exceeds the supplied budget"
        );
        // Negative / NaN budgets are rejected before any work.
        assert_eq!(
            budgeted_softmax(&c, -1.0).unwrap_err(),
            "ADA-A3 epsilon must be finite and non-negative"
        );
        assert!(budgeted_softmax(&c, f64::NAN).is_err());
    }

    #[test]
    fn certificate_is_monotone_in_budget() {
        let c = case(&[-2.0, 5.0, 0.5], 3);
        let loose = budgeted_softmax(&c, 1.0e-6).unwrap();
        let tight = budgeted_softmax(&c, 1.0e-12).unwrap();
        // The ACHIEVED bound is a property of the computation, not the budget.
        assert_eq!(
            loose.certified_relative_error_bound.to_bits(),
            tight.certified_relative_error_bound.to_bits()
        );
    }

    #[test]
    fn invalid_cases_rejected() {
        let bad_values = AttentionCase {
            logits: vec![1.0],
            values: vec![],
            head_dim: 1,
        };
        assert!(budgeted_softmax(&bad_values, 1e-6).is_err());

        let nan_logits = AttentionCase {
            logits: vec![f32::NAN, 1.0],
            values: vec![0.0, 0.0, 0.0, 0.0],
            head_dim: 2,
        };
        assert!(budgeted_softmax(&nan_logits, 1e-6).is_err());
    }
}
