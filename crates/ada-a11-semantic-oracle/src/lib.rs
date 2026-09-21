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

/// Integer numerator matrix for the frozen mixer with common denominator 4.
///
/// ```text
/// M = (1/4) * [[2, 2, 0], [1, 2, 1], [0, 2, 2]]
/// ```
pub const MIXER_NUMERATORS: [[i64; TOKEN_COUNT]; TOKEN_COUNT] = [[2, 2, 0], [1, 2, 1], [0, 2, 2]];
/// Shared positive denominator for [`MIXER_NUMERATORS`].
pub const MIXER_DENOMINATOR: i64 = 4;
/// Bit shift equivalent of [`MIXER_DENOMINATOR`] (`4 = 2^2`).
pub const MIXER_DENOMINATOR_SHIFT: u32 = 2;

/// Exact dyadic-rational state for the frozen E1 mixer.
///
/// Each coordinate is `numerators[i] / 2^denominator_shift`. The evaluator never
/// accumulates through general `f64` multiply-add for the mixer step itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExactScalarSequenceState {
    numerators: [i128; TOKEN_COUNT],
    denominator_shift: u32,
}

/// Fail-closed errors from the exact/integer-scaled mixer path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactStateError {
    /// A coordinate was non-finite when converting from [`ScalarSequenceState`].
    NonFinite {
        /// Index of the first non-finite component.
        index: usize,
    },
    /// An exact step overflowed signed 128-bit intermediate arithmetic.
    Overflow,
    /// The denominator shift exceeded the supported bound.
    DenominatorShiftOverflow,
    /// Exact value could not be represented as a finite `f64`.
    NonFiniteF64 {
        /// Index of the first non-finite conversion result.
        index: usize,
    },
}

impl ExactScalarSequenceState {
    /// Construct an exact integer state (`denominator_shift = 0`).
    #[must_use]
    pub const fn from_integers(values: [i64; TOKEN_COUNT]) -> Self {
        Self {
            numerators: [values[0] as i128, values[1] as i128, values[2] as i128],
            denominator_shift: 0,
        }
    }

    /// All-zero exact state.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            numerators: [0; TOKEN_COUNT],
            denominator_shift: 0,
        }
    }

    /// Borrow exact numerators.
    #[must_use]
    pub const fn numerators(&self) -> &[i128; TOKEN_COUNT] {
        &self.numerators
    }

    /// Power-of-two denominator shift (`value = numerator / 2^shift`).
    #[must_use]
    pub const fn denominator_shift(&self) -> u32 {
        self.denominator_shift
    }

    /// Convert a finite f64 state into an exact integer state when every
    /// coordinate is an integer representable in `i64`.
    ///
    /// Non-integer finite inputs are rejected; callers that need a general
    /// rational lift must construct [`ExactScalarSequenceState`] explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`ExactStateError::NonFinite`] for non-finite inputs and
    /// [`ExactStateError::Overflow`] when a finite value is not an `i64`
    /// integer.
    pub fn try_from_integer_f64(state: &ScalarSequenceState) -> Result<Self, ExactStateError> {
        let mut numerators = [0_i128; TOKEN_COUNT];
        for (index, value) in state.values().iter().enumerate() {
            if !value.is_finite() {
                return Err(ExactStateError::NonFinite { index });
            }
            numerators[index] = f64_exact_i64(*value).map(i128::from)?;
        }
        Ok(Self {
            numerators,
            denominator_shift: 0,
        })
    }

    /// Convert to a finite [`ScalarSequenceState`].
    ///
    /// # Errors
    ///
    /// Fails closed on non-finite `f64` conversion.
    pub fn to_f64_state(&self) -> Result<ScalarSequenceState, ExactStateError> {
        let mut values = [0.0_f64; TOKEN_COUNT];
        for (index, numerator) in self.numerators.iter().enumerate() {
            let value = dyadic_i128_to_f64(*numerator, self.denominator_shift)?;
            if !value.is_finite() {
                return Err(ExactStateError::NonFiniteF64 { index });
            }
            values[index] = value;
        }
        ScalarSequenceState::new(values).map_err(|error| match error {
            StateError::NonFinite { index } => ExactStateError::NonFiniteF64 { index },
        })
    }
}

/// Apply one exact mixer step using integer-scaled dyadic arithmetic.
///
/// The step computes `numerators' = MIXER_NUMERATORS * numerators` and increases
/// the denominator shift by [`MIXER_DENOMINATOR_SHIFT`], without a general `f64`
/// multiply accumulation.
///
/// # Errors
///
/// Returns [`ExactStateError::Overflow`] or
/// [`ExactStateError::DenominatorShiftOverflow`] on range failure.
pub fn advance_exact(
    state: &ExactScalarSequenceState,
) -> Result<ExactScalarSequenceState, ExactStateError> {
    let next_shift = state
        .denominator_shift
        .checked_add(MIXER_DENOMINATOR_SHIFT)
        .ok_or(ExactStateError::DenominatorShiftOverflow)?;
    let mut numerators = [0_i128; TOKEN_COUNT];
    for (row_index, row) in MIXER_NUMERATORS.iter().enumerate() {
        let mut acc = 0_i128;
        for (col_index, weight) in row.iter().enumerate() {
            let term = i128::from(*weight)
                .checked_mul(state.numerators[col_index])
                .ok_or(ExactStateError::Overflow)?;
            acc = acc.checked_add(term).ok_or(ExactStateError::Overflow)?;
        }
        numerators[row_index] = acc;
    }
    Ok(ExactScalarSequenceState {
        numerators,
        denominator_shift: next_shift,
    })
}

/// Apply `horizon` exact mixer steps.
///
/// # Errors
///
/// Propagates overflow / denominator failures from [`advance_exact`].
pub fn advance_exact_horizon(
    initial: &ExactScalarSequenceState,
    horizon: usize,
) -> Result<ExactScalarSequenceState, ExactStateError> {
    let mut current = *initial;
    for _ in 0..horizon {
        current = advance_exact(&current)?;
    }
    Ok(current)
}

/// Exact closed-form antisymmetric oracle `2^-h [1, 0, -1]` as dyadic integers.
///
/// Represented in the same unreduced form produced by [`advance_exact_horizon`]:
/// numerators `[2^h, 0, -2^h]` with `denominator_shift = 2h`.
///
/// This function does not call [`advance_exact`]; it constructs the independent
/// mathematical expectation directly.
#[must_use]
pub const fn antisymmetric_oracle_exact(horizon: usize) -> ExactScalarSequenceState {
    let mut factor = 1_i128;
    let mut shift = 0_u32;
    let mut step = 0_usize;
    while step < horizon {
        factor *= 2;
        let Some(next_shift) = shift.checked_add(MIXER_DENOMINATOR_SHIFT) else {
            return ExactScalarSequenceState {
                numerators: [factor, 0, -factor],
                denominator_shift: u32::MAX,
            };
        };
        shift = next_shift;
        step += 1;
    }
    ExactScalarSequenceState {
        numerators: [factor, 0, -factor],
        denominator_shift: shift,
    }
}

fn dyadic_i128_to_f64(numerator: i128, denominator_shift: u32) -> Result<f64, ExactStateError> {
    if numerator == 0 {
        return Ok(0.0);
    }
    let num = i128_to_exact_f64(numerator)?;
    if denominator_shift == 0 {
        return Ok(num);
    }
    // Scale by repeated division by 2 to preserve dyadic exactness for small shifts.
    let mut value = num;
    let mut remaining = denominator_shift;
    while remaining > 0 {
        value *= 0.5;
        if !value.is_finite() {
            return Err(ExactStateError::Overflow);
        }
        remaining -= 1;
    }
    Ok(value)
}

fn i128_to_exact_f64(value: i128) -> Result<f64, ExactStateError> {
    // Only accept integers inside the exact f64 integer range.
    const EXACT_F64_INT_MAX: i128 = 1_i128 << 53;
    if !(-EXACT_F64_INT_MAX..=EXACT_F64_INT_MAX).contains(&value) {
        return Err(ExactStateError::Overflow);
    }
    let as_i64 = i64::try_from(value).map_err(|_| ExactStateError::Overflow)?;
    Ok(i64_to_f64_exact(as_i64))
}

fn i64_to_f64_exact(value: i64) -> f64 {
    // Safe: callers restrict to |value| <= 2^53.
    #[allow(clippy::cast_precision_loss)]
    {
        value as f64
    }
}

fn f64_exact_i64(value: f64) -> Result<i64, ExactStateError> {
    // 2^53: largest magnitude with exact integer representation in f64.
    const EXACT_F64_INT_MAX: f64 = 9_007_199_254_740_992.0;
    if !value.is_finite() {
        return Err(ExactStateError::Overflow);
    }
    if value.fract() != 0.0 {
        return Err(ExactStateError::Overflow);
    }
    if !(-EXACT_F64_INT_MAX..=EXACT_F64_INT_MAX).contains(&value) {
        return Err(ExactStateError::Overflow);
    }
    #[allow(clippy::cast_possible_truncation)]
    let as_i64 = value as i64;
    if i64_to_f64_exact(as_i64).to_bits() != value.to_bits() {
        return Err(ExactStateError::Overflow);
    }
    Ok(as_i64)
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

    #[test]
    fn exact_mixer_matches_hand_derived_basis_and_integer_cases() {
        let zero = advance_exact(&ExactScalarSequenceState::zeros()).expect("exact");
        assert_eq!(zero.numerators(), &[0, 0, 0]);
        assert_eq!(zero.denominator_shift(), MIXER_DENOMINATOR_SHIFT);

        let e0 = advance_exact(&ExactScalarSequenceState::from_integers([1, 0, 0])).expect("exact");
        assert_eq!(e0.numerators(), &[2, 1, 0]);
        assert_eq!(e0.denominator_shift(), 2);

        let e1 = advance_exact(&ExactScalarSequenceState::from_integers([0, 1, 0])).expect("exact");
        assert_eq!(e1.numerators(), &[2, 2, 2]);
        assert_eq!(e1.denominator_shift(), 2);

        let e2 = advance_exact(&ExactScalarSequenceState::from_integers([0, 0, 1])).expect("exact");
        assert_eq!(e2.numerators(), &[0, 1, 2]);
        assert_eq!(e2.denominator_shift(), 2);

        let simple =
            advance_exact(&ExactScalarSequenceState::from_integers([4, -2, 2])).expect("exact");
        assert_eq!(simple.numerators(), &[4, 2, 0]);
        assert_eq!(simple.denominator_shift(), 2);
    }

    #[test]
    fn exact_path_parity_checks_f64_path_on_dyadic_fixtures() {
        let seeds = [
            ExactScalarSequenceState::zeros(),
            ExactScalarSequenceState::from_integers([1, 0, 0]),
            ExactScalarSequenceState::from_integers([0, 1, 0]),
            ExactScalarSequenceState::from_integers([0, 0, 1]),
            ExactScalarSequenceState::from_integers([1, 0, -1]),
            ExactScalarSequenceState::from_integers([2, 2, 2]),
            ExactScalarSequenceState::from_integers([4, -2, 2]),
        ];
        for seed in seeds {
            let exact = advance_exact(&seed).expect("exact step");
            let f64_seed = seed.to_f64_state().expect("finite seed");
            let f64_out = advance(&f64_seed).expect("f64 step");
            let exact_as_f64 = exact.to_f64_state().expect("finite exact");
            assert_state_bits(&exact_as_f64, &f64_out);
        }
    }

    #[test]
    fn exact_antisymmetric_oracle_matches_evaluator_and_f64_oracle() {
        let seed = ExactScalarSequenceState::from_integers([1, 0, -1]);
        for horizon in 0..=16 {
            let evaluated = advance_exact_horizon(&seed, horizon).expect("exact");
            let expected = antisymmetric_oracle_exact(horizon);
            assert_eq!(evaluated.numerators(), expected.numerators());
            assert_eq!(evaluated.denominator_shift(), expected.denominator_shift());
            let as_f64 = evaluated.to_f64_state().expect("finite");
            assert_state_bits(&as_f64, &antisymmetric_oracle(horizon));
            let f64_eval =
                advance_horizon(&antisymmetric_seed(), horizon).expect("f64 remains finite");
            assert_state_bits(&as_f64, &f64_eval);
        }
    }

    #[test]
    fn exact_path_fails_closed_on_overflow_and_non_integer_inputs() {
        let non_integer = ScalarSequenceState::new([0.5, 0.0, 0.0]).expect("finite");
        assert!(matches!(
            ExactScalarSequenceState::try_from_integer_f64(&non_integer),
            Err(ExactStateError::Overflow)
        ));
        let huge = ExactScalarSequenceState {
            numerators: [i128::MAX, i128::MAX, i128::MAX],
            denominator_shift: 0,
        };
        assert!(matches!(
            advance_exact(&huge),
            Err(ExactStateError::Overflow)
        ));
    }
}
