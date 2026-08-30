//! Executable linear reconstruction for declared latent/compressed KV storage.

use ada_core::SemanticId;
use ada_workload::{
    HeadGrouping, InputRepresentation, KvCacheSpec, KvIndexing, KvRepresentation, MaskKind,
    MatrixLayout, PositionInfo, ScalarPrecision, ScoreBiasSpec, StateSpec, WorkloadContract,
    WorkloadMode,
};

use crate::{AdvancedReferenceError, check_len, checked_product, ensure_finite, stable_softmax};

/// Explicit latent-KV attention rule.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentAttentionSpec {
    semantic: SemanticId,
    scale: f64,
}

impl LatentAttentionSpec {
    /// Construct a latent-KV reference rule.
    ///
    /// # Errors
    ///
    /// Rejects non-finite or non-positive affinity scale.
    pub fn new(semantic: SemanticId, scale: f64) -> Result<Self, AdvancedReferenceError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(AdvancedReferenceError::InvalidField("latent scale"));
        }
        Ok(Self { semantic, scale })
    }

    /// Semantic identity of the reconstructed attention rule.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }
}

/// Caller-owned exact latent storage and reconstruction matrices.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentReferenceInput {
    /// Row-major query matrix `[query][qk_dimension]`.
    pub queries: Vec<f64>,
    /// Row-major latent key rows `[kv][latent_dimension]`.
    pub latent_keys: Vec<f64>,
    /// Row-major latent value rows `[kv][latent_dimension]`.
    pub latent_values: Vec<f64>,
    /// Row-major linear map `[latent_dimension][qk_dimension]`.
    pub key_reconstruction: Vec<f64>,
    /// Row-major linear map `[latent_dimension][value_dimension]`.
    pub value_reconstruction: Vec<f64>,
    /// Identity of the exact key reconstruction artifact.
    pub key_reconstruction_identity: String,
    /// Identity of the exact value reconstruction artifact.
    pub value_reconstruction_identity: String,
}

/// Exact reconstructed K/V plus reference attention output.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentReferenceOutput {
    reconstructed_keys: Vec<f64>,
    reconstructed_values: Vec<f64>,
    output: Vec<f64>,
}

impl LatentReferenceOutput {
    /// Row-major reconstructed keys.
    #[must_use]
    pub fn reconstructed_keys(&self) -> &[f64] {
        &self.reconstructed_keys
    }

    /// Row-major reconstructed values.
    #[must_use]
    pub fn reconstructed_values(&self) -> &[f64] {
        &self.reconstructed_values
    }

    /// Row-major attention output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }
}

/// Reconstruct declared latent K/V linearly and execute dense softmax attention.
///
/// V1 is deliberately single-example/single-head, unmasked, row-major f64,
/// stateless prefill with no cache or position/bias rule. The reconstruction
/// identities must exactly match [`ada_workload::LatentKvSpec`].
///
/// # Errors
///
/// Rejects workloads outside that bounded domain, reconstruction identity or
/// shape mismatches, and non-finite data/intermediates.
pub fn evaluate_latent(
    spec: &LatentAttentionSpec,
    workload: &WorkloadContract,
    input: &LatentReferenceInput,
) -> Result<LatentReferenceOutput, AdvancedReferenceError> {
    let geometry = validate_workload(workload, input)?;
    let keys = reconstruct(
        &input.latent_keys,
        geometry.kv_count,
        geometry.latent_dimension,
        &input.key_reconstruction,
        geometry.qk_dimension,
        "key reconstruction",
    )?;
    let values = reconstruct(
        &input.latent_values,
        geometry.kv_count,
        geometry.latent_dimension,
        &input.value_reconstruction,
        geometry.value_dimension,
        "value reconstruction",
    )?;
    let mut output = vec![
        0.0;
        checked_product(
            geometry.query_count,
            geometry.value_dimension,
            "latent output"
        )?
    ];
    for query in 0..geometry.query_count {
        let query_start = query * geometry.qk_dimension;
        let query_row = &input.queries[query_start..query_start + geometry.qk_dimension];
        let mut scores = Vec::with_capacity(geometry.kv_count);
        for key in 0..geometry.kv_count {
            let key_start = key * geometry.qk_dimension;
            let key_row = &keys[key_start..key_start + geometry.qk_dimension];
            let dot = query_row
                .iter()
                .zip(key_row)
                .map(|(&left, &right)| left * right)
                .sum::<f64>();
            let score = dot * spec.scale;
            if !score.is_finite() {
                return Err(AdvancedReferenceError::NonFinite("latent affinity"));
            }
            scores.push(score);
        }
        let weights = stable_softmax(&scores)?;
        let output_start = query * geometry.value_dimension;
        for (key, &weight) in weights.iter().enumerate() {
            let value_start = key * geometry.value_dimension;
            for dimension in 0..geometry.value_dimension {
                output[output_start + dimension] += weight * values[value_start + dimension];
            }
        }
    }
    ensure_finite(&output, "latent value mixing")?;
    Ok(LatentReferenceOutput {
        reconstructed_keys: keys,
        reconstructed_values: values,
        output,
    })
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    query_count: usize,
    kv_count: usize,
    latent_dimension: usize,
    qk_dimension: usize,
    value_dimension: usize,
}

fn validate_workload(
    workload: &WorkloadContract,
    input: &LatentReferenceInput,
) -> Result<Geometry, AdvancedReferenceError> {
    validate_latent_domain(workload)?;
    let geometry = workload.geometry();
    let KvRepresentation::LatentCompressed(latent) = workload.kv_representation() else {
        return Err(AdvancedReferenceError::Unsupported(
            "workload does not declare latent-compressed KV",
        ));
    };
    if latent.key_reconstruction() != input.key_reconstruction_identity {
        return Err(AdvancedReferenceError::IdentityMismatch(
            "key reconstruction",
        ));
    }
    if latent.value_reconstruction() != input.value_reconstruction_identity {
        return Err(AdvancedReferenceError::IdentityMismatch(
            "value reconstruction",
        ));
    }
    let result = Geometry {
        query_count: geometry
            .sequence_lengths()
            .query_length_for(0)
            .ok_or(AdvancedReferenceError::InvalidField("query length"))?,
        kv_count: geometry
            .sequence_lengths()
            .kv_length_for(0)
            .ok_or(AdvancedReferenceError::InvalidField("KV length"))?,
        latent_dimension: latent.latent_dimension(),
        qk_dimension: geometry
            .qk_dimension()
            .ok_or(AdvancedReferenceError::InvalidField("Q/K dimension"))?,
        value_dimension: geometry.value_dimension(),
    };
    validate_latent_shapes(input, result)?;
    Ok(result)
}

fn validate_latent_domain(workload: &WorkloadContract) -> Result<(), AdvancedReferenceError> {
    workload
        .validate()
        .map_err(|_| AdvancedReferenceError::Unsupported("invalid latent workload"))?;
    let geometry = workload.geometry();
    if geometry.sequence_lengths().batch_count() != 1
        || geometry.query_heads() != 1
        || geometry.kv_heads() != 1
        || geometry.head_grouping() != HeadGrouping::MultiHead
    {
        return Err(AdvancedReferenceError::Unsupported(
            "latent v1 requires one example and one head",
        ));
    }
    if workload.mode() != WorkloadMode::Prefill
        || !matches!(workload.inputs(), InputRepresentation::ExplicitQkv)
        || !matches!(workload.kv_cache(), KvCacheSpec::None)
        || !matches!(workload.kv_indexing(), KvIndexing::Identity)
        || !matches!(workload.state(), StateSpec::Stateless)
        || !matches!(workload.positions(), PositionInfo::None)
        || !matches!(workload.score_bias(), ScoreBiasSpec::None)
        || !matches!(
            workload.mask().kind(),
            MaskKind::None | MaskKind::Bidirectional
        )
    {
        return Err(AdvancedReferenceError::Unsupported(
            "latent v1 requires stateless unmasked prefill without cache/position/bias",
        ));
    }
    validate_latent_precision_and_layout(workload)
}

fn validate_latent_precision_and_layout(
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
            "latent v1 is explicitly f64",
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
            "latent v1 requires row-major layout",
        ));
    }
    Ok(())
}

fn validate_latent_shapes(
    input: &LatentReferenceInput,
    geometry: Geometry,
) -> Result<(), AdvancedReferenceError> {
    check_len(
        "latent queries",
        checked_product(
            geometry.query_count,
            geometry.qk_dimension,
            "latent queries",
        )?,
        input.queries.len(),
    )?;
    check_len(
        "latent keys",
        checked_product(geometry.kv_count, geometry.latent_dimension, "latent keys")?,
        input.latent_keys.len(),
    )?;
    check_len(
        "latent values",
        checked_product(
            geometry.kv_count,
            geometry.latent_dimension,
            "latent values",
        )?,
        input.latent_values.len(),
    )?;
    check_len(
        "key reconstruction matrix",
        checked_product(
            geometry.latent_dimension,
            geometry.qk_dimension,
            "key reconstruction matrix",
        )?,
        input.key_reconstruction.len(),
    )?;
    check_len(
        "value reconstruction matrix",
        checked_product(
            geometry.latent_dimension,
            geometry.value_dimension,
            "value reconstruction matrix",
        )?,
        input.value_reconstruction.len(),
    )?;
    ensure_finite(&input.queries, "latent queries")?;
    ensure_finite(&input.latent_keys, "latent keys")?;
    ensure_finite(&input.latent_values, "latent values")?;
    ensure_finite(&input.key_reconstruction, "key reconstruction matrix")?;
    ensure_finite(&input.value_reconstruction, "value reconstruction matrix")
}

fn reconstruct(
    latent: &[f64],
    row_count: usize,
    latent_dimension: usize,
    matrix: &[f64],
    output_dimension: usize,
    stage: &'static str,
) -> Result<Vec<f64>, AdvancedReferenceError> {
    let mut output = vec![0.0; checked_product(row_count, output_dimension, stage)?];
    for row in 0..row_count {
        for output_dimension_index in 0..output_dimension {
            let mut value = 0.0;
            for latent_index in 0..latent_dimension {
                value += latent[row * latent_dimension + latent_index]
                    * matrix[latent_index * output_dimension + output_dimension_index];
            }
            output[row * output_dimension + output_dimension_index] = value;
        }
    }
    ensure_finite(&output, stage)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use ada_core::{SemanticFamily, SemanticId};
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, LatentKvSpec, LatentPositionHandling,
        PrecisionPolicy, SequenceLengths, TensorLayout, WorkloadOptions,
    };

    use super::*;

    fn workload() -> WorkloadContract {
        WorkloadContract::new(
            AttentionGeometry::new(GeometrySpec {
                sequence_lengths: SequenceLengths::uniform(1, 1, 2).unwrap(),
                query_heads: 1,
                kv_heads: 1,
                qk_dimension: Some(2),
                value_dimension: 1,
                topology: AttentionTopology::SelfAttention,
                head_grouping: HeadGrouping::MultiHead,
            })
            .unwrap(),
            WorkloadOptions {
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                layout: TensorLayout::row_major(),
                kv_representation: KvRepresentation::LatentCompressed(
                    LatentKvSpec::new(
                        1,
                        "key-map",
                        "value-map",
                        LatentPositionHandling::BeforeCompression,
                    )
                    .unwrap(),
                ),
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn latent_linear_maps_reconstruct_exact_kv_then_attend() {
        let spec = LatentAttentionSpec::new(
            SemanticId::new(SemanticFamily::Experimental, "latent", 1).unwrap(),
            1.0,
        )
        .unwrap();
        let result = evaluate_latent(
            &spec,
            &workload(),
            &LatentReferenceInput {
                queries: vec![1.0, 0.0],
                latent_keys: vec![1.0, 2.0],
                latent_values: vec![3.0, 5.0],
                key_reconstruction: vec![1.0, 0.0],
                value_reconstruction: vec![2.0],
                key_reconstruction_identity: "key-map".into(),
                value_reconstruction_identity: "value-map".into(),
            },
        )
        .unwrap();
        assert_eq!(result.reconstructed_keys(), &[1.0, 0.0, 2.0, 0.0]);
        assert_eq!(result.reconstructed_values(), &[6.0, 10.0]);
        assert!(result.output()[0] > 8.9 && result.output()[0] < 9.0);
    }

    #[test]
    fn reconstruction_identity_mismatch_fails_closed() {
        let spec = LatentAttentionSpec::new(
            SemanticId::new(SemanticFamily::Experimental, "latent-id", 1).unwrap(),
            1.0,
        )
        .unwrap();
        let result = evaluate_latent(
            &spec,
            &workload(),
            &LatentReferenceInput {
                queries: vec![1.0, 0.0],
                latent_keys: vec![1.0, 2.0],
                latent_values: vec![3.0, 5.0],
                key_reconstruction: vec![1.0, 0.0],
                value_reconstruction: vec![2.0],
                key_reconstruction_identity: "wrong".into(),
                value_reconstruction_identity: "value-map".into(),
            },
        );
        assert_eq!(
            result,
            Err(AdvancedReferenceError::IdentityMismatch(
                "key reconstruction"
            ))
        );
    }
}
