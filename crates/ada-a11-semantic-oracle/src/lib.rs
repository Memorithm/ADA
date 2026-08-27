//! ADA-A11-E1 deterministic reference semantic.
//!
//! This crate introduces one deliberately tiny non-softmax sequence-interaction
//! semantic whose behavior can be derived independently by hand. It is a
//! scientific plumbing fixture, not a production attention mechanism and not a
//! claim of model usefulness.

#![forbid(unsafe_code)]

use ada_core::{
    ImplementationCandidateId, MaskContract, SemanticContractError, SemanticDescriptor,
    SemanticFamily, SemanticId, StateContract, WeightContract,
};
use ada_workload::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, InputRepresentation,
    MaskKind, MaskSpec, PrecisionPolicy, ScalarPrecision, SequenceLengths, StateSpec,
    WorkloadContract, WorkloadContractError, WorkloadMode, WorkloadOptions,
};

/// Stable semantic identity used by the E1 fixture.
pub const SEMANTIC_NAME: &str = "balanced-three-token-mixer";
/// Stable implementation identity for the scalar reference evaluator.
pub const IMPLEMENTATION_NAME: &str = "scalar-reference";
/// Named precomputed interaction artifact declared by the workload contract.
pub const WORKLOAD_INPUT_IDENTITY: &str = "ada-a11-e1-fixed-mixer";
/// Number of scalar token positions in the deterministic fixture.
pub const TOKEN_COUNT: usize = 3;

/// Frozen row-stochastic interaction matrix.
///
/// ```text
///       [ 1/2  1/2   0  ]
/// M  =  [ 1/4  1/2  1/4 ]
///       [  0   1/2  1/2 ]
/// ```
///
/// The fixture is intentionally not represented as Q/K + softmax. The workload
/// contract records it as a named precomputed interaction rule.
pub const MIXER: [[f64; TOKEN_COUNT]; TOKEN_COUNT] =
    [[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]];

/// Finite scalar sequence state for the deterministic E1 evaluator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarSequenceState {
    values: [f64; TOKEN_COUNT],
}

/// State construction failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateError {
    NonFinite { index: usize },
}

impl ScalarSequenceState {
    /// Construct a finite three-token state.
    ///
    /// # Errors
    ///
    /// Returns the index of the first non-finite component.
    pub fn new(values: [f64; TOKEN_COUNT]) -> Result<Self, StateError> {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(StateError::NonFinite { index });
        }
        Ok(Self { values })
    }

    /// All-zero reference state.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            values: [0.0; TOKEN_COUNT],
        }
    }

    /// Frozen scalar values in token order.
    #[must_use]
    pub const fn values(&self) -> &[f64; TOKEN_COUNT] {
        &self.values
    }
}

/// Build the semantic identity and high-level reference contract.
///
/// # Errors
///
/// Fails closed if the hard-coded semantic identifier ever violates ADA's
/// identity rules.
pub fn semantic_descriptor() -> Result<SemanticDescriptor, SemanticContractError> {
    let id = SemanticId::new(SemanticFamily::Experimental, SEMANTIC_NAME, 1)?;
    Ok(SemanticDescriptor::new(
        id,
        MaskContract::Bidirectional,
        StateContract::Stateless,
        WeightContract::ProbabilitySimplex,
    ))
}

/// Identity of the scalar reference implementation used by this crate.
///
/// # Errors
///
/// Propagates ADA semantic/implementation identity validation failures.
pub fn reference_implementation_id() -> Result<ImplementationCandidateId, SemanticContractError> {
    let descriptor = semantic_descriptor()?;
    ImplementationCandidateId::new(descriptor.id().clone(), IMPLEMENTATION_NAME, 1)
}

/// Build the explicit A11-E1 research workload.
///
/// The workload declares a single batch item, three query/KV token positions,
/// one head, one scalar value lane, bidirectional visibility, f64 numerical
/// policy, no recurrent state, and a named precomputed interaction rule. It
/// deliberately has no Q/K dimension because this fixture is not reinterpreted
/// as dot-product attention.
///
/// # Errors
///
/// Propagates workload construction/validation failures.
pub fn workload_contract() -> Result<WorkloadContract, WorkloadContractError> {
    let geometry = AttentionGeometry::new(GeometrySpec {
        sequence_lengths: SequenceLengths::uniform(1, TOKEN_COUNT, TOKEN_COUNT)?,
        query_heads: 1,
        kv_heads: 1,
        qk_dimension: None,
        value_dimension: 1,
        topology: AttentionTopology::SelfAttention,
        head_grouping: HeadGrouping::MultiHead,
    })?;

    let mask = MaskSpec::new(MaskKind::Bidirectional)?;
    let options = WorkloadOptions {
        mode: WorkloadMode::Prefill,
        mask,
        precision: PrecisionPolicy::new(
            ScalarPrecision::F64,
            ScalarPrecision::F64,
            ScalarPrecision::F64,
            ScalarPrecision::F64,
        ),
        inputs: InputRepresentation::PrecomputedScores {
            identity: WORKLOAD_INPUT_IDENTITY.into(),
        },
        state: StateSpec::Stateless,
        ..WorkloadOptions::default()
    };

    WorkloadContract::new(geometry, options)
}

/// Apply one deterministic mixer step.
///
/// Because the input state is finite and every coefficient in [`MIXER`] is
/// finite, non-finite output can arise only from floating-point overflow. Such
/// an output fails closed.
///
/// # Errors
///
/// Returns the first non-finite output component.
pub fn advance(state: &ScalarSequenceState) -> Result<ScalarSequenceState, StateError> {
    let mut output = [0.0_f64; TOKEN_COUNT];
    for (row_index, row) in MIXER.iter().enumerate() {
        output[row_index] = row
            .iter()
            .zip(state.values())
            .map(|(weight, value)| *weight * *value)
            .sum();
    }
    ScalarSequenceState::new(output)
}

/// Apply the deterministic semantic for `horizon` successive steps.
///
/// # Errors
///
/// Propagates non-finite output failures from [`advance`].
pub fn advance_horizon(
    initial: &ScalarSequenceState,
    horizon: usize,
) -> Result<ScalarSequenceState, StateError> {
    let mut current = *initial;
    for _ in 0..horizon {
        current = advance(&current)?;
    }
    Ok(current)
}

/// Balanced antisymmetric mode used for the independent E1 oracle.
///
/// For the frozen matrix, `v = [1, 0, -1]^T` satisfies `M v = (1/2) v`.
#[must_use]
pub const fn antisymmetric_seed() -> ScalarSequenceState {
    ScalarSequenceState {
        values: [1.0, 0.0, -1.0],
    }
}

/// Hand-derived antisymmetric state after `horizon` mixer steps.
///
/// This function intentionally does **not** call [`advance`]. It computes the
/// independent closed-form oracle `2^-h [1, 0, -1]` by repeated exact dyadic
/// scaling, so tests can compare the evaluator with a separately defined
/// mathematical expectation.
#[must_use]
pub fn antisymmetric_oracle(horizon: usize) -> ScalarSequenceState {
    let mut factor = 1.0_f64;
    for _ in 0..horizon {
        factor *= 0.5;
    }
    ScalarSequenceState {
        values: [factor, 0.0, -factor],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_workload::{InputRepresentation, MaskKind, ScalarPrecision};

    fn assert_state_bits(actual: &ScalarSequenceState, expected: &ScalarSequenceState) {
        for (actual_value, expected_value) in actual.values().iter().zip(expected.values()) {
            assert_eq!(actual_value.to_bits(), expected_value.to_bits());
        }
    }

    fn assert_values_bits(actual: &[f64; TOKEN_COUNT], expected: [f64; TOKEN_COUNT]) {
        for (actual_value, expected_value) in actual.iter().zip(expected) {
            assert_eq!(actual_value.to_bits(), expected_value.to_bits());
        }
    }

    #[test]
    fn semantic_and_implementation_identities_are_separate_but_bound() {
        let semantic = semantic_descriptor().expect("hard-coded semantic identity is valid");
        let implementation =
            reference_implementation_id().expect("hard-coded implementation identity is valid");

        assert_eq!(semantic.id().family(), SemanticFamily::Experimental);
        assert_eq!(semantic.id().name(), SEMANTIC_NAME);
        assert_eq!(semantic.id().revision(), 1);
        assert_eq!(implementation.semantic(), semantic.id());
        assert_eq!(implementation.name(), IMPLEMENTATION_NAME);
    }

    #[test]
    fn workload_does_not_masquerade_as_explicit_qk_attention() {
        let workload = workload_contract().expect("hard-coded workload is valid");
        assert_eq!(workload.mode(), WorkloadMode::Prefill);
        assert_eq!(workload.geometry().qk_dimension(), None);
        assert_eq!(workload.geometry().query_heads(), 1);
        assert_eq!(workload.geometry().kv_heads(), 1);
        assert_eq!(workload.geometry().value_dimension(), 1);
        assert_eq!(
            workload.geometry().sequence_lengths().query_length(),
            TOKEN_COUNT
        );
        assert_eq!(
            workload.geometry().sequence_lengths().kv_length(),
            TOKEN_COUNT
        );
        assert!(matches!(workload.mask().kind(), MaskKind::Bidirectional));
        assert!(matches!(workload.state(), StateSpec::Stateless));
        assert_eq!(workload.precision().input(), ScalarPrecision::F64);
        assert_eq!(workload.precision().accumulation(), ScalarPrecision::F64);
        assert_eq!(
            workload.inputs(),
            &InputRepresentation::PrecomputedScores {
                identity: WORKLOAD_INPUT_IDENTITY.into(),
            }
        );
    }

    #[test]
    fn mixer_rows_are_probability_simplex_rows() {
        for row in MIXER {
            assert!(row.iter().all(|value| value.is_finite() && *value >= 0.0));
            assert_eq!(row.iter().sum::<f64>().to_bits(), 1.0_f64.to_bits());
        }
    }

    #[test]
    fn constant_mode_is_invariant_under_row_stochastic_mixing() {
        let constant = ScalarSequenceState::new([2.0, 2.0, 2.0]).expect("finite fixture");
        let advanced = advance(&constant).expect("finite output");
        assert_state_bits(&advanced, &constant);
    }

    #[test]
    fn evaluator_matches_independent_antisymmetric_oracle() {
        let seed = antisymmetric_seed();
        for horizon in 0..=16 {
            let evaluated = advance_horizon(&seed, horizon).expect("dyadic fixture remains finite");
            let expected = antisymmetric_oracle(horizon);
            assert_state_bits(&evaluated, &expected);
        }
    }

    #[test]
    fn first_three_downstream_states_match_gate_b_fixture() {
        let seed = antisymmetric_seed();
        let first = advance_horizon(&seed, 1).expect("finite output");
        let second = advance_horizon(&seed, 2).expect("finite output");
        let third = advance_horizon(&seed, 3).expect("finite output");

        assert_values_bits(first.values(), [0.5, 0.0, -0.5]);
        assert_values_bits(second.values(), [0.25, 0.0, -0.25]);
        assert_values_bits(third.values(), [0.125, 0.0, -0.125]);
    }

    #[test]
    fn non_finite_state_fails_closed() {
        assert!(matches!(
            ScalarSequenceState::new([0.0, f64::INFINITY, 0.0]),
            Err(StateError::NonFinite { index: 1 })
        ));
    }
}
