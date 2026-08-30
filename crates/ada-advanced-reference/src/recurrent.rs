//! Stateful delta-rule and causal linear-attention references.

use ada_core::SemanticId;
use ada_workload::{
    HeadGrouping, InputRepresentation, KvCacheSpec, KvIndexing, KvRepresentation, MaskKind,
    MatrixLayout, PositionInfo, ScalarPrecision, ScoreBiasSpec, StateSpec, WorkloadContract,
    WorkloadMode,
};

use crate::{AdvancedReferenceError, check_len, checked_product, ensure_finite};

/// Explicit recurrent delta-memory update rule.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaRuleSpec {
    semantic: SemanticId,
    state_identity: String,
    decay: f64,
    learning_rate: f64,
}

impl DeltaRuleSpec {
    /// Construct a delta-memory rule `S <- decay*S + lr*(v outer k)`.
    ///
    /// # Errors
    ///
    /// Rejects empty state identity, decay outside `[0, 1]`, or non-finite /
    /// negative learning rate.
    pub fn new(
        semantic: SemanticId,
        state_identity: impl Into<String>,
        decay: f64,
        learning_rate: f64,
    ) -> Result<Self, AdvancedReferenceError> {
        let state_identity = state_identity.into();
        if state_identity.is_empty() || state_identity.chars().any(char::is_whitespace) {
            return Err(AdvancedReferenceError::InvalidField("delta state identity"));
        }
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(AdvancedReferenceError::InvalidField("delta decay"));
        }
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(AdvancedReferenceError::InvalidField("delta learning rate"));
        }
        Ok(Self {
            semantic,
            state_identity,
            decay,
            learning_rate,
        })
    }

    /// Semantic identity.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }
}

/// Explicit recurrent Q/K/V stream and initial state.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentInput {
    /// Row-major token queries `[token][qk_dimension]`.
    pub queries: Vec<f64>,
    /// Row-major token keys `[token][qk_dimension]`.
    pub keys: Vec<f64>,
    /// Row-major token values `[token][value_dimension]`.
    pub values: Vec<f64>,
    /// Exact initial state matrix.
    pub initial_state: Vec<f64>,
}

/// Delta-rule output and final state.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaRuleOutput {
    output: Vec<f64>,
    final_state: Vec<f64>,
}

impl DeltaRuleOutput {
    /// Row-major per-token output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Final state matrix `[value_dimension][qk_dimension]`.
    #[must_use]
    pub fn final_state(&self) -> &[f64] {
        &self.final_state
    }
}

/// Evaluate the explicit recurrent delta-memory rule.
///
/// The workload must declare `StateSpec::Recurrent` with rows equal to the
/// value dimension and columns equal to the Q/K dimension, using the exact
/// state identity from the rule.
///
/// # Errors
///
/// Rejects mismatched workload state/geometry, malformed input, or non-finite
/// intermediate state/output.
pub fn evaluate_delta_rule(
    spec: &DeltaRuleSpec,
    workload: &WorkloadContract,
    input: &RecurrentInput,
) -> Result<DeltaRuleOutput, AdvancedReferenceError> {
    let geometry = validate_recurrent_workload(workload, &spec.state_identity, StateShape::Delta)?;
    validate_stream_input(
        input,
        geometry,
        checked_product(
            geometry.value_dimension,
            geometry.qk_dimension,
            "delta state",
        )?,
    )?;
    let mut state = input.initial_state.clone();
    let mut output =
        vec![0.0; checked_product(geometry.tokens, geometry.value_dimension, "delta output")?];
    for token in 0..geometry.tokens {
        let key = &input.keys[token * geometry.qk_dimension..(token + 1) * geometry.qk_dimension];
        let query =
            &input.queries[token * geometry.qk_dimension..(token + 1) * geometry.qk_dimension];
        let value =
            &input.values[token * geometry.value_dimension..(token + 1) * geometry.value_dimension];
        for (value_index, &value_component) in value.iter().enumerate() {
            for (key_index, &key_component) in key.iter().enumerate() {
                let index = value_index * geometry.qk_dimension + key_index;
                state[index] = spec.decay * state[index]
                    + spec.learning_rate * value_component * key_component;
            }
        }
        let out_start = token * geometry.value_dimension;
        for value_index in 0..geometry.value_dimension {
            output[out_start + value_index] = (0..geometry.qk_dimension)
                .map(|dimension| {
                    state[value_index * geometry.qk_dimension + dimension] * query[dimension]
                })
                .sum();
        }
    }
    ensure_finite(&state, "delta final state")?;
    ensure_finite(&output, "delta output")?;
    Ok(DeltaRuleOutput {
        output,
        final_state: state,
    })
}

/// Explicit positive-feature linear-attention rule.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAttentionSpec {
    semantic: SemanticId,
    state_identity: String,
    epsilon: f64,
}

impl LinearAttentionSpec {
    /// Construct a causal linear-attention reference using `phi(x)=ELU(x)+1`.
    ///
    /// # Errors
    ///
    /// Rejects invalid state identity or non-positive/non-finite epsilon.
    pub fn new(
        semantic: SemanticId,
        state_identity: impl Into<String>,
        epsilon: f64,
    ) -> Result<Self, AdvancedReferenceError> {
        let state_identity = state_identity.into();
        if state_identity.is_empty() || state_identity.chars().any(char::is_whitespace) {
            return Err(AdvancedReferenceError::InvalidField(
                "linear state identity",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(AdvancedReferenceError::InvalidField("linear epsilon"));
        }
        Ok(Self {
            semantic,
            state_identity,
            epsilon,
        })
    }

    /// Semantic identity.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }
}

/// Linear-attention output and final sufficient-statistic state.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAttentionOutput {
    output: Vec<f64>,
    final_state: Vec<f64>,
}

impl LinearAttentionOutput {
    /// Row-major causal output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Final state `[qk_dimension][value_dimension + 1]`; last column is normalizer state.
    #[must_use]
    pub fn final_state(&self) -> &[f64] {
        &self.final_state
    }
}

/// Execute causal feature-map linear attention from recurrent sufficient statistics.
///
/// # Errors
///
/// The workload state must have `qk_dimension` rows and
/// `value_dimension + 1` columns with the exact declared identity. Malformed or
/// non-finite inputs and non-positive normalizers fail closed.
pub fn evaluate_linear_attention(
    spec: &LinearAttentionSpec,
    workload: &WorkloadContract,
    input: &RecurrentInput,
) -> Result<LinearAttentionOutput, AdvancedReferenceError> {
    let geometry = validate_recurrent_workload(workload, &spec.state_identity, StateShape::Linear)?;
    let columns = geometry
        .value_dimension
        .checked_add(1)
        .ok_or(AdvancedReferenceError::ExceedsLimit("linear state columns"))?;
    validate_stream_input(
        input,
        geometry,
        checked_product(geometry.qk_dimension, columns, "linear state")?,
    )?;
    let mut state = input.initial_state.clone();
    let mut output =
        vec![0.0; checked_product(geometry.tokens, geometry.value_dimension, "linear output")?];
    for token in 0..geometry.tokens {
        let key = &input.keys[token * geometry.qk_dimension..(token + 1) * geometry.qk_dimension];
        let query =
            &input.queries[token * geometry.qk_dimension..(token + 1) * geometry.qk_dimension];
        let value =
            &input.values[token * geometry.value_dimension..(token + 1) * geometry.value_dimension];
        let key_features = key.iter().copied().map(phi).collect::<Vec<_>>();
        let query_features = query.iter().copied().map(phi).collect::<Vec<_>>();
        ensure_finite(&key_features, "linear key features")?;
        ensure_finite(&query_features, "linear query features")?;
        for (dimension, &key_feature) in key_features.iter().enumerate() {
            let state_start = dimension * columns;
            for (value_index, &value_component) in value.iter().enumerate() {
                state[state_start + value_index] += key_feature * value_component;
            }
            state[state_start + geometry.value_dimension] += key_feature;
        }
        let denominator = spec.epsilon
            + (0..geometry.qk_dimension)
                .map(|dimension| {
                    query_features[dimension]
                        * state[dimension * columns + geometry.value_dimension]
                })
                .sum::<f64>();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(AdvancedReferenceError::NonFinite("linear denominator"));
        }
        let out_start = token * geometry.value_dimension;
        for value_index in 0..geometry.value_dimension {
            let numerator = (0..geometry.qk_dimension)
                .map(|dimension| {
                    query_features[dimension] * state[dimension * columns + value_index]
                })
                .sum::<f64>();
            output[out_start + value_index] = numerator / denominator;
        }
    }
    ensure_finite(&state, "linear final state")?;
    ensure_finite(&output, "linear output")?;
    Ok(LinearAttentionOutput {
        output,
        final_state: state,
    })
}

fn phi(value: f64) -> f64 {
    if value >= 0.0 {
        value + 1.0
    } else {
        value.exp()
    }
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    tokens: usize,
    qk_dimension: usize,
    value_dimension: usize,
}

#[derive(Debug, Clone, Copy)]
enum StateShape {
    Delta,
    Linear,
}

fn validate_recurrent_workload(
    workload: &WorkloadContract,
    state_identity: &str,
    shape: StateShape,
) -> Result<Geometry, AdvancedReferenceError> {
    validate_recurrent_domain(workload)?;
    let geometry = workload.geometry();
    let qk_dimension = geometry
        .qk_dimension()
        .ok_or(AdvancedReferenceError::InvalidField(
            "recurrent qk dimension",
        ))?;
    let value_dimension = geometry.value_dimension();
    let query_tokens = geometry.sequence_lengths().query_length_for(0).ok_or(
        AdvancedReferenceError::InvalidField("recurrent query length"),
    )?;
    let kv_tokens = geometry
        .sequence_lengths()
        .kv_length_for(0)
        .ok_or(AdvancedReferenceError::InvalidField("recurrent kv length"))?;
    if query_tokens != kv_tokens {
        return Err(AdvancedReferenceError::InvalidField(
            "recurrent stream requires equal query/KV token counts",
        ));
    }
    validate_state_shape(
        workload,
        state_identity,
        shape,
        qk_dimension,
        value_dimension,
    )?;
    Ok(Geometry {
        tokens: query_tokens,
        qk_dimension,
        value_dimension,
    })
}

fn validate_recurrent_domain(workload: &WorkloadContract) -> Result<(), AdvancedReferenceError> {
    workload
        .validate()
        .map_err(|_| AdvancedReferenceError::Unsupported("invalid recurrent workload"))?;
    let geometry = workload.geometry();
    if geometry.sequence_lengths().batch_count() != 1
        || geometry.query_heads() != 1
        || geometry.kv_heads() != 1
        || geometry.head_grouping() != HeadGrouping::MultiHead
    {
        return Err(AdvancedReferenceError::Unsupported(
            "recurrent v1 requires one example and one head",
        ));
    }
    if !matches!(
        workload.mode(),
        WorkloadMode::Prefill | WorkloadMode::TrainingForward
    ) || !matches!(workload.inputs(), InputRepresentation::ExplicitQkv)
        || !matches!(workload.kv_representation(), KvRepresentation::Full)
        || !matches!(workload.kv_cache(), KvCacheSpec::None)
        || !matches!(workload.kv_indexing(), KvIndexing::Identity)
        || !matches!(workload.positions(), PositionInfo::None)
        || !matches!(workload.score_bias(), ScoreBiasSpec::None)
        || !matches!(workload.mask().kind(), MaskKind::Causal)
    {
        return Err(AdvancedReferenceError::Unsupported(
            "recurrent v1 requires causal explicit-QKV prefill/training-forward",
        ));
    }
    validate_recurrent_precision_and_layout(workload)
}

fn validate_recurrent_precision_and_layout(
    workload: &WorkloadContract,
) -> Result<(), AdvancedReferenceError> {
    let precision = workload.precision();
    if [
        precision.input(),
        precision.accumulation(),
        precision.output(),
        precision.storage(),
    ]
    .into_iter()
    .any(|value| value != ScalarPrecision::F64)
    {
        return Err(AdvancedReferenceError::Unsupported(
            "recurrent v1 is explicitly f64",
        ));
    }
    let layout = workload.layout();
    if [
        layout.query(),
        layout.key(),
        layout.value(),
        layout.output(),
    ]
    .into_iter()
    .any(|value| value != MatrixLayout::RowMajor)
    {
        return Err(AdvancedReferenceError::Unsupported(
            "recurrent v1 requires row-major tensors",
        ));
    }
    Ok(())
}

fn validate_state_shape(
    workload: &WorkloadContract,
    state_identity: &str,
    shape: StateShape,
    qk_dimension: usize,
    value_dimension: usize,
) -> Result<(), AdvancedReferenceError> {
    let StateSpec::Recurrent {
        rows,
        columns,
        identity,
    } = workload.state()
    else {
        return Err(AdvancedReferenceError::Unsupported(
            "workload does not declare recurrent state",
        ));
    };
    if identity != state_identity {
        return Err(AdvancedReferenceError::IdentityMismatch("recurrent state"));
    }
    let expected_shape = match shape {
        StateShape::Delta => (value_dimension, qk_dimension),
        StateShape::Linear => (
            qk_dimension,
            value_dimension
                .checked_add(1)
                .ok_or(AdvancedReferenceError::ExceedsLimit("linear columns"))?,
        ),
    };
    if (*rows, *columns) != expected_shape {
        return Err(AdvancedReferenceError::InvalidField(
            "recurrent state shape",
        ));
    }
    Ok(())
}

fn validate_stream_input(
    input: &RecurrentInput,
    geometry: Geometry,
    state_elements: usize,
) -> Result<(), AdvancedReferenceError> {
    check_len(
        "recurrent queries",
        checked_product(geometry.tokens, geometry.qk_dimension, "recurrent queries")?,
        input.queries.len(),
    )?;
    check_len(
        "recurrent keys",
        checked_product(geometry.tokens, geometry.qk_dimension, "recurrent keys")?,
        input.keys.len(),
    )?;
    check_len(
        "recurrent values",
        checked_product(
            geometry.tokens,
            geometry.value_dimension,
            "recurrent values",
        )?,
        input.values.len(),
    )?;
    check_len("initial state", state_elements, input.initial_state.len())?;
    ensure_finite(&input.queries, "recurrent queries")?;
    ensure_finite(&input.keys, "recurrent keys")?;
    ensure_finite(&input.values, "recurrent values")?;
    ensure_finite(&input.initial_state, "recurrent initial state")
}

#[cfg(test)]
mod tests {
    use ada_core::{SemanticFamily, SemanticId};
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, MaskSpec, PrecisionPolicy,
        SequenceLengths, TensorLayout, WorkloadOptions,
    };

    use super::*;

    fn workload(rows: usize, columns: usize, identity: &str) -> WorkloadContract {
        WorkloadContract::new(
            AttentionGeometry::new(GeometrySpec {
                sequence_lengths: SequenceLengths::uniform(1, 2, 2).unwrap(),
                query_heads: 1,
                kv_heads: 1,
                qk_dimension: Some(1),
                value_dimension: 1,
                topology: AttentionTopology::SelfAttention,
                head_grouping: HeadGrouping::MultiHead,
            })
            .unwrap(),
            WorkloadOptions {
                mask: MaskSpec::new(MaskKind::Causal).unwrap(),
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                layout: TensorLayout::row_major(),
                state: StateSpec::Recurrent {
                    rows,
                    columns,
                    identity: identity.into(),
                },
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    fn id(name: &str) -> SemanticId {
        SemanticId::new(SemanticFamily::Experimental, name, 1).unwrap()
    }

    #[test]
    fn delta_rule_updates_state_and_emits_causal_outputs() {
        let spec = DeltaRuleSpec::new(id("delta"), "delta-state", 1.0, 1.0).unwrap();
        let result = evaluate_delta_rule(
            &spec,
            &workload(1, 1, "delta-state"),
            &RecurrentInput {
                queries: vec![1.0, 1.0],
                keys: vec![2.0, 3.0],
                values: vec![4.0, 5.0],
                initial_state: vec![0.0],
            },
        )
        .unwrap();
        assert_eq!(result.output(), &[8.0, 23.0]);
        assert_eq!(result.final_state(), &[23.0]);
    }

    #[test]
    fn linear_attention_accumulates_positive_feature_statistics() {
        let spec = LinearAttentionSpec::new(id("linear"), "linear-state", 1.0e-12).unwrap();
        let result = evaluate_linear_attention(
            &spec,
            &workload(1, 2, "linear-state"),
            &RecurrentInput {
                queries: vec![0.0, 0.0],
                keys: vec![0.0, 0.0],
                values: vec![2.0, 4.0],
                initial_state: vec![0.0, 0.0],
            },
        )
        .unwrap();
        assert!((result.output()[0] - 2.0).abs() < 1.0e-9);
        assert!((result.output()[1] - 3.0).abs() < 1.0e-9);
        assert_eq!(result.final_state(), &[6.0, 2.0]);
    }

    #[test]
    fn state_identity_mismatch_fails_closed() {
        let spec = DeltaRuleSpec::new(id("delta-id"), "wrong", 1.0, 1.0).unwrap();
        assert_eq!(
            evaluate_delta_rule(
                &spec,
                &workload(1, 1, "expected"),
                &RecurrentInput {
                    queries: vec![1.0, 1.0],
                    keys: vec![1.0, 1.0],
                    values: vec![1.0, 1.0],
                    initial_state: vec![0.0],
                },
            ),
            Err(AdvancedReferenceError::IdentityMismatch("recurrent state"))
        );
    }
}
