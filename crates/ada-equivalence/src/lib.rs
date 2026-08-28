//! Conservative semantic rewrite/equivalence proofs for ADA.
//!
//! This crate deliberately separates two arithmetic domains:
//!
//! - [`ArithmeticDomain::RealArithmetic`] for proofs that may rely on exact-real
//!   algebra in future extensions;
//! - [`ArithmeticDomain::IeeeF64`] for rewrites that must preserve the current
//!   deterministic f64 execution order and observable bits.
//!
//! V1 implements only arithmetic-neutral selection rewrites whose selected key
//! sequence is unchanged for the supplied workload. It does **not** apply
//! associativity, distributivity, reassociation, fast-math identities, or
//! transcendental approximations. Consequently every V1 proof is valid in both
//! domains. Keeping the domain in the witness prevents a future real-only rule
//! from silently being reused as an IEEE-754 proof.
//!
//! Equivalence never mutates or merges [`ada_core::SemanticId`]. Two distinct
//! research identities may receive a workload-bounded execution-equivalence
//! witness while remaining distinct semantics for provenance and novelty
//! review.

#![forbid(unsafe_code)]

use ada_core::SemanticId;
use ada_semantic::{
    SemanticIrError, SemanticProgram, SemanticProgramSpec, SelectionRule,
};
use ada_workload::WorkloadContract;
use std::fmt::{Display, Formatter};

/// Arithmetic contract under which an equivalence witness is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmeticDomain {
    /// Mathematical real arithmetic. Future rules may be valid here while
    /// being unsafe for finite-precision evaluation.
    RealArithmetic,
    /// The current deterministic IEEE-754 f64 reference execution contract.
    IeeeF64,
}

/// One explicitly justified semantic rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RewriteRule {
    /// `TopK { k }` selects every possible key because `k >= kv_length`.
    ExhaustiveTopK,
    /// `Window { radius }` covers every possible query/key index pair.
    ExhaustiveWindow,
}

impl RewriteRule {
    /// Whether this rule is valid in the requested arithmetic domain.
    ///
    /// V1 rules only change a redundant selector while preserving the selected
    /// key sequence, so they are arithmetic-neutral and valid in both domains.
    #[must_use]
    pub const fn valid_in(self, _domain: ArithmeticDomain) -> bool {
        match self {
            Self::ExhaustiveTopK | Self::ExhaustiveWindow => true,
        }
    }
}

/// Fail-closed normalization/equivalence errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquivalenceError {
    /// A semantic or workload lies outside the executable reference contract.
    Semantic(SemanticIrError),
    /// The workload geometry did not provide a usable bounded sequence shape.
    InvalidWorkloadGeometry,
}

impl Display for EquivalenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Semantic(error) => write!(formatter, "semantic equivalence error: {error}"),
            Self::InvalidWorkloadGeometry => {
                formatter.write_str("semantic equivalence requires non-empty workload geometry")
            }
        }
    }
}

impl std::error::Error for EquivalenceError {}

impl From<SemanticIrError> for EquivalenceError {
    fn from(value: SemanticIrError) -> Self {
        Self::Semantic(value)
    }
}

/// A program normalized by only those rewrite rules proven valid for one
/// explicit workload and arithmetic domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationResult {
    domain: ArithmeticDomain,
    program: SemanticProgram,
    applied_rules: Vec<RewriteRule>,
}

impl NormalizationResult {
    /// Arithmetic contract used during normalization.
    #[must_use]
    pub const fn domain(&self) -> ArithmeticDomain {
        self.domain
    }

    /// Normalized executable program. Semantic identity is retained unchanged.
    #[must_use]
    pub const fn program(&self) -> &SemanticProgram {
        &self.program
    }

    /// Ordered rewrite log.
    #[must_use]
    pub fn applied_rules(&self) -> &[RewriteRule] {
        &self.applied_rules
    }
}

/// Proof artifact for two workload-bounded execution-equivalent programs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquivalenceWitness {
    domain: ArithmeticDomain,
    left_semantic: SemanticId,
    right_semantic: SemanticId,
    left_rules: Vec<RewriteRule>,
    right_rules: Vec<RewriteRule>,
    normalized_canonical_text: String,
}

impl EquivalenceWitness {
    /// Arithmetic contract under which the proof was established.
    #[must_use]
    pub const fn domain(&self) -> ArithmeticDomain {
        self.domain
    }

    /// Original semantic identity on the left side.
    #[must_use]
    pub const fn left_semantic(&self) -> &SemanticId {
        &self.left_semantic
    }

    /// Original semantic identity on the right side.
    #[must_use]
    pub const fn right_semantic(&self) -> &SemanticId {
        &self.right_semantic
    }

    /// Rewrites applied to the left program.
    #[must_use]
    pub fn left_rules(&self) -> &[RewriteRule] {
        &self.left_rules
    }

    /// Rewrites applied to the right program.
    #[must_use]
    pub fn right_rules(&self) -> &[RewriteRule] {
        &self.right_rules
    }

    /// Canonical executable structure used by the proof, with semantic identity
    /// intentionally excluded from equality comparison but retained separately
    /// in this witness.
    #[must_use]
    pub fn normalized_canonical_text(&self) -> &str {
        &self.normalized_canonical_text
    }
}

/// Result of conservative proof search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquivalenceResult {
    /// Equivalence was established by the declared finite rewrite system.
    Proven(EquivalenceWitness),
    /// The current rules could not establish equivalence. This is not a proof
    /// of inequivalence.
    NotProven,
}

/// Normalize one semantic program for a concrete workload.
///
/// # Errors
///
/// Returns an error when the program/workload is outside the executable
/// reference domain or the workload geometry is empty.
pub fn normalize_for_workload(
    program: &SemanticProgram,
    workload: &WorkloadContract,
    domain: ArithmeticDomain,
) -> Result<NormalizationResult, EquivalenceError> {
    program.validate_for_workload(workload)?;

    let lengths = workload.geometry().sequence_lengths();
    let query_length = lengths.query_length();
    let kv_length = lengths.kv_length();
    if query_length == 0 || kv_length == 0 {
        return Err(EquivalenceError::InvalidWorkloadGeometry);
    }

    let mut selection = program.selection();
    let mut applied_rules = Vec::new();
    match selection {
        SelectionRule::TopK { k } if k >= kv_length => {
            let rule = RewriteRule::ExhaustiveTopK;
            debug_assert!(rule.valid_in(domain));
            selection = SelectionRule::All;
            applied_rules.push(rule);
        }
        SelectionRule::Window { radius }
            if radius >= maximum_index_distance(query_length, kv_length) =>
        {
            let rule = RewriteRule::ExhaustiveWindow;
            debug_assert!(rule.valid_in(domain));
            selection = SelectionRule::All;
            applied_rules.push(rule);
        }
        SelectionRule::All | SelectionRule::Window { .. } | SelectionRule::TopK { .. } => {}
    }

    let normalized = rebuild_with_selection(program, selection)?;
    normalized.validate_for_workload(workload)?;
    Ok(NormalizationResult {
        domain,
        program: normalized,
        applied_rules,
    })
}

/// Attempt to prove execution equivalence for one explicit workload.
///
/// Semantic IDs are deliberately ignored by the executable-structure
/// comparison and retained in the witness. A successful proof therefore does
/// not collapse research identity, prior-art status, or provenance.
///
/// # Errors
///
/// Returns an error when either program/workload is outside the executable
/// reference domain.
pub fn prove_equivalent_for_workload(
    left: &SemanticProgram,
    right: &SemanticProgram,
    workload: &WorkloadContract,
    domain: ArithmeticDomain,
) -> Result<EquivalenceResult, EquivalenceError> {
    let left_normalized = normalize_for_workload(left, workload, domain)?;
    let right_normalized = normalize_for_workload(right, workload, domain)?;

    if !same_execution_structure(left_normalized.program(), right_normalized.program()) {
        return Ok(EquivalenceResult::NotProven);
    }

    Ok(EquivalenceResult::Proven(EquivalenceWitness {
        domain,
        left_semantic: left.descriptor().id().clone(),
        right_semantic: right.descriptor().id().clone(),
        left_rules: left_normalized.applied_rules,
        right_rules: right_normalized.applied_rules,
        normalized_canonical_text: execution_structure_text(left_normalized.program()),
    }))
}

const fn maximum_index_distance(query_length: usize, kv_length: usize) -> usize {
    let query_max = query_length.saturating_sub(1);
    let kv_max = kv_length.saturating_sub(1);
    if query_max > kv_max {
        query_max
    } else {
        kv_max
    }
}

fn rebuild_with_selection(
    program: &SemanticProgram,
    selection: SelectionRule,
) -> Result<SemanticProgram, SemanticIrError> {
    SemanticProgram::new(SemanticProgramSpec {
        descriptor: program.descriptor().clone(),
        input_transform: program.input_transform(),
        affinity: program.affinity(),
        mask: program.mask().clone(),
        selection,
        weight: program.weight(),
        value_mix: program.value_mix(),
        output: program.output(),
    })
}

fn same_execution_structure(left: &SemanticProgram, right: &SemanticProgram) -> bool {
    left.descriptor().mask() == right.descriptor().mask()
        && left.descriptor().state() == right.descriptor().state()
        && left.descriptor().weights() == right.descriptor().weights()
        && left.input_transform() == right.input_transform()
        && left.affinity() == right.affinity()
        && left.mask() == right.mask()
        && left.selection() == right.selection()
        && left.weight() == right.weight()
        && left.value_mix() == right.value_mix()
        && left.output() == right.output()
}

fn execution_structure_text(program: &SemanticProgram) -> String {
    // The semantic codec contains identity fields. Keep a deterministic proof
    // summary that records executable components only, so distinct SemanticIds
    // can be proven workload-equivalent without being merged.
    format!(
        "input={:?}\naffinity={:?}\nmask={:?}\nselection={:?}\nweight={:?}\nvalue_mix={:?}\noutput={:?}\nmask_contract={:?}\nstate_contract={:?}\nweight_contract={:?}\n",
        program.input_transform(),
        program.affinity(),
        program.mask(),
        program.selection(),
        program.weight(),
        program.value_mix(),
        program.output(),
        program.descriptor().mask(),
        program.descriptor().state(),
        program.descriptor().weights(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{SemanticFamily, SemanticId};
    use ada_semantic::MaskRule;
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, PrecisionPolicy,
        ScalarPrecision, SequenceLengths, WorkloadOptions,
    };

    fn workload(query_length: usize, kv_length: usize) -> WorkloadContract {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, query_length, kv_length).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(8),
            value_dimension: 8,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    fn softmax(name: &str, selection: SelectionRule, scale: f64) -> SemanticProgram {
        SemanticProgram::standard_softmax(
            SemanticId::new(SemanticFamily::Experimental, name, 1).unwrap(),
            MaskRule::Unmasked,
            selection,
            scale,
        )
        .unwrap()
    }

    #[test]
    fn exhaustive_topk_is_ieee_safe_and_preserves_distinct_semantic_ids() {
        let workload = workload(3, 4);
        let left = softmax("topk-all-equivalent", SelectionRule::TopK { k: 4 }, 1.0);
        let right = softmax("explicit-all", SelectionRule::All, 1.0);

        let result = prove_equivalent_for_workload(
            &left,
            &right,
            &workload,
            ArithmeticDomain::IeeeF64,
        )
        .unwrap();
        let EquivalenceResult::Proven(witness) = result else {
            panic!("expected proof");
        };
        assert_ne!(witness.left_semantic(), witness.right_semantic());
        assert_eq!(witness.left_rules(), &[RewriteRule::ExhaustiveTopK]);
        assert!(witness.right_rules().is_empty());
        assert_eq!(witness.domain(), ArithmeticDomain::IeeeF64);
    }

    #[test]
    fn exhaustive_window_is_proven_in_both_domains() {
        let workload = workload(3, 5);
        let windowed = softmax("wide-window", SelectionRule::Window { radius: 4 }, 1.0);
        let all = softmax("all-visible", SelectionRule::All, 1.0);

        for domain in [
            ArithmeticDomain::RealArithmetic,
            ArithmeticDomain::IeeeF64,
        ] {
            let result = prove_equivalent_for_workload(&windowed, &all, &workload, domain).unwrap();
            let EquivalenceResult::Proven(witness) = result else {
                panic!("expected proof");
            };
            assert_eq!(witness.left_rules(), &[RewriteRule::ExhaustiveWindow]);
            assert_eq!(witness.domain(), domain);
        }
    }

    #[test]
    fn insufficient_topk_is_not_silently_equated_with_all() {
        let workload = workload(2, 4);
        let topk = softmax("topk-two", SelectionRule::TopK { k: 2 }, 1.0);
        let all = softmax("all-four", SelectionRule::All, 1.0);

        let result = prove_equivalent_for_workload(
            &topk,
            &all,
            &workload,
            ArithmeticDomain::IeeeF64,
        )
        .unwrap();
        assert_eq!(result, EquivalenceResult::NotProven);
    }

    #[test]
    fn real_domain_does_not_enable_unimplemented_float_algebra() {
        let workload = workload(2, 2);
        let left = softmax("scale-one", SelectionRule::All, 1.0);
        let right = softmax("scale-two", SelectionRule::All, 2.0);

        let result = prove_equivalent_for_workload(
            &left,
            &right,
            &workload,
            ArithmeticDomain::RealArithmetic,
        )
        .unwrap();
        assert_eq!(result, EquivalenceResult::NotProven);
    }

    #[test]
    fn normalization_is_workload_bound() {
        let short = workload(2, 3);
        let long = workload(2, 5);
        let candidate = softmax("topk-three", SelectionRule::TopK { k: 3 }, 1.0);

        let short_normalized = normalize_for_workload(
            &candidate,
            &short,
            ArithmeticDomain::IeeeF64,
        )
        .unwrap();
        assert_eq!(short_normalized.program().selection(), SelectionRule::All);
        assert_eq!(
            short_normalized.applied_rules(),
            &[RewriteRule::ExhaustiveTopK]
        );

        let long_normalized = normalize_for_workload(
            &candidate,
            &long,
            ArithmeticDomain::IeeeF64,
        )
        .unwrap();
        assert_eq!(
            long_normalized.program().selection(),
            SelectionRule::TopK { k: 3 }
        );
        assert!(long_normalized.applied_rules().is_empty());
    }
}
