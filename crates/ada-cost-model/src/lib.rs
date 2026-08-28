//! Explicit, backend-neutral estimated cost model for ADA implementation plans.
//!
//! The model consumes a validated workload plus an implementation schedule and
//! caller-declared assumptions. It reports logical work and logical payload
//! traffic. It does not report physical DRAM transactions, cache hits, latency,
//! energy, occupancy, or instructions.

#![forbid(unsafe_code)]

use ada_implementation::{ImplementationPlan, WorkPartition};
use ada_objective::EstimatedCost;
use ada_workload::{
    InputRepresentation, KvCacheSpec, KvRepresentation, MaskKind, ScalarPrecision, StateSpec,
    WorkloadContract, WorkloadMode,
};
use std::fmt::{Display, Formatter};

/// Cost-model failures. Unsupported domains fail closed rather than receiving
/// an invented estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostModelError {
    /// The workload contains a feature not yet modeled by this estimator.
    Unsupported(&'static str),
    /// A caller-declared model assumption is invalid.
    InvalidAssumption(&'static str),
    /// Exact integer accounting overflowed `u64`.
    ArithmeticOverflow(&'static str),
}

impl Display for CostModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(feature) => {
                write!(formatter, "cost model does not support {feature}")
            }
            Self::InvalidAssumption(field) => write!(formatter, "invalid cost assumption: {field}"),
            Self::ArithmeticOverflow(field) => {
                write!(formatter, "cost accounting overflow: {field}")
            }
        }
    }
}

impl std::error::Error for CostModelError {}

/// Semantic-operation costs supplied by the experiment.
///
/// ADA deliberately does not infer these values from a semantic family name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationProfile {
    /// Logical FLOPs charged for one score/affinity evaluation.
    pub score_flops_per_pair: u64,
    /// Transcendental evaluations charged for one score pair.
    pub transcendentals_per_pair: u64,
    /// Logical FLOPs charged for one value element mixed into an output.
    pub value_flops_per_element: u64,
    /// Logical FLOPs charged for finalizing one output element.
    pub finalize_flops_per_output: u64,
}

impl OperationProfile {
    /// Conventional dense scaled-dot-product/softmax accounting profile.
    ///
    /// This helper counts a length-`d` dot product as `d` multiplies plus
    /// `d-1` additions and adds one scale multiply, for `2d` score FLOPs. It
    /// charges one transcendental and two value-mixing FLOPs per value element.
    /// The caller must still decide whether this convention is appropriate to
    /// the semantic being studied.
    ///
    /// # Errors
    ///
    /// Returns an error when `qk_dimension` is zero or conversion/multiplication
    /// overflows `u64`.
    pub fn scaled_dot_softmax(qk_dimension: usize) -> Result<Self, CostModelError> {
        if qk_dimension == 0 {
            return Err(CostModelError::InvalidAssumption("qk_dimension"));
        }
        let dimension = u64::try_from(qk_dimension)
            .map_err(|_| CostModelError::ArithmeticOverflow("qk_dimension"))?;
        let score_flops_per_pair = checked_mul(dimension, 2, "score_flops_per_pair")?;
        Ok(Self {
            score_flops_per_pair,
            transcendentals_per_pair: 1,
            value_flops_per_element: 2,
            finalize_flops_per_output: 1,
        })
    }
}

/// Explicit implementation-specific assumptions used by the traffic model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CostAssumptions {
    /// Number of complete score/K passes.
    pub score_passes: u16,
    /// Number of complete value-mixing/V passes.
    pub value_passes: u16,
    /// Whether Q payload is reloaded for each KV tile within each score pass.
    pub reload_query_per_kv_tile: bool,
    /// Whether a shared KV head is reused across its MQA/GQA query heads.
    pub reuse_shared_kv_across_query_heads: bool,
}

impl Default for CostAssumptions {
    fn default() -> Self {
        Self {
            score_passes: 1,
            value_passes: 1,
            reload_query_per_kv_tile: false,
            reuse_shared_kv_across_query_heads: true,
        }
    }
}

impl CostAssumptions {
    fn validate(self) -> Result<(), CostModelError> {
        if self.score_passes == 0 {
            return Err(CostModelError::InvalidAssumption("score_passes"));
        }
        if self.value_passes == 0 {
            return Err(CostModelError::InvalidAssumption("value_passes"));
        }
        Ok(())
    }
}

/// Deterministic estimated cost report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EstimatedCostReport {
    score_pairs: u64,
    value_elements: u64,
    output_elements: u64,
    logical_flops: u64,
    transcendental_operations: u64,
    reduction_operations: u64,
    logical_payload_read_bits: u64,
    logical_payload_write_bits: u64,
    logical_kv_cache_payload_bits: Option<u64>,
    workspace_bytes: u64,
}

impl EstimatedCostReport {
    /// Score/affinity evaluations under the declared pass count.
    #[must_use]
    pub const fn score_pairs(self) -> u64 {
        self.score_pairs
    }

    /// Value elements mixed under the declared pass count.
    #[must_use]
    pub const fn value_elements(self) -> u64 {
        self.value_elements
    }

    /// Final output elements.
    #[must_use]
    pub const fn output_elements(self) -> u64 {
        self.output_elements
    }

    /// Logical FLOPs under the supplied operation profile.
    #[must_use]
    pub const fn logical_flops(self) -> u64 {
        self.logical_flops
    }

    /// Logical transcendental count.
    #[must_use]
    pub const fn transcendental_operations(self) -> u64 {
        self.transcendental_operations
    }

    /// Explicit split-KV reduction operations modeled from the schedule.
    #[must_use]
    pub const fn reduction_operations(self) -> u64 {
        self.reduction_operations
    }

    /// Logical payload bits read from source representations.
    ///
    /// This is not a DRAM/cache transaction counter.
    #[must_use]
    pub const fn logical_payload_read_bits(self) -> u64 {
        self.logical_payload_read_bits
    }

    /// Logical output payload bits written.
    ///
    /// This is not a physical store-transaction counter.
    #[must_use]
    pub const fn logical_payload_write_bits(self) -> u64 {
        self.logical_payload_write_bits
    }

    /// Logical K/V cache payload bits when a cache is declared.
    #[must_use]
    pub const fn logical_kv_cache_payload_bits(self) -> Option<u64> {
        self.logical_kv_cache_payload_bits
    }

    /// Workspace bytes declared by the implementation IR.
    #[must_use]
    pub const fn workspace_bytes(self) -> u64 {
        self.workspace_bytes
    }

    /// Convert the report into ADA's typed **estimated** objective section.
    ///
    /// Physical latency/energy remain absent. Payload bits are rounded up to
    /// whole bytes because `ada-objective` currently stores byte estimates.
    ///
    /// # Errors
    ///
    /// Returns an error if total payload-bit accounting overflows.
    pub fn to_objective_estimate(self) -> Result<EstimatedCost, CostModelError> {
        let total_bits = checked_add(
            self.logical_payload_read_bits,
            self.logical_payload_write_bits,
            "payload_bits",
        )?;
        Ok(EstimatedCost {
            bytes_moved: Some(bits_to_bytes_ceil(total_bits)),
            workspace_bytes: Some(self.workspace_bytes),
            kv_cache_bytes: self.logical_kv_cache_payload_bits.map(bits_to_bytes_ceil),
            index_construction: None,
            communication_bytes: None,
            reduction_operations: Some(self.reduction_operations),
        })
    }
}

/// Estimate one implementation/workload pair under explicit assumptions.
///
/// The v1 model supports explicit-Q/K/V, full-KV, stateless forward workloads
/// with unmasked or bidirectional visibility. External/causal masks, latent
/// reconstruction, recurrent state, historical/precomputed-score inputs, and
/// backward execution fail closed because their exact traffic/work depends on
/// semantics not represented by this estimator yet.
///
/// # Errors
///
/// Returns an error for unsupported domains, invalid assumptions, or exact
/// integer-accounting overflow.
pub fn estimate_cost(
    workload: &WorkloadContract,
    implementation: &ImplementationPlan,
    operations: OperationProfile,
    assumptions: CostAssumptions,
) -> Result<EstimatedCostReport, CostModelError> {
    assumptions.validate()?;
    validate_supported_workload(workload)?;

    let geometry = workload.geometry();
    let qk_dimension = geometry
        .qk_dimension()
        .ok_or(CostModelError::Unsupported("missing Q/K dimension"))?;
    let query_heads = to_u64(geometry.query_heads(), "query_heads")?;
    let kv_heads = to_u64(geometry.kv_heads(), "kv_heads")?;
    let value_dimension = to_u64(geometry.value_dimension(), "value_dimension")?;
    let qk_dimension = to_u64(qk_dimension, "qk_dimension")?;
    let q_tile = u64::from(implementation.schedule().tile.queries);
    let kv_tile = u64::from(implementation.schedule().tile.keys);
    let score_passes = u64::from(assumptions.score_passes);
    let value_passes = u64::from(assumptions.value_passes);
    let kv_head_reads = if assumptions.reuse_shared_kv_across_query_heads {
        kv_heads
    } else {
        query_heads
    };

    let input_bits = precision_bits(workload.precision().input());
    let storage_bits = precision_bits(workload.precision().storage());
    let output_bits = precision_bits(workload.precision().output());

    let mut totals = RunningTotals::default();
    let lengths = geometry.sequence_lengths();
    for batch in 0..lengths.batch_count() {
        let query_length = to_u64(
            lengths
                .query_length_for(batch)
                .ok_or(CostModelError::ArithmeticOverflow("query_length"))?,
            "query_length",
        )?;
        let kv_length = to_u64(
            lengths
                .kv_length_for(batch)
                .ok_or(CostModelError::ArithmeticOverflow("kv_length"))?,
            "kv_length",
        )?;
        let query_tiles = query_length.div_ceil(q_tile);
        let kv_tiles = kv_length.div_ceil(kv_tile);

        let base_pairs = product(&[query_length, query_heads, kv_length], "score_pairs")?;
        totals.score_pairs = checked_add(
            totals.score_pairs,
            checked_mul(base_pairs, score_passes, "score_pairs")?,
            "score_pairs",
        )?;

        let base_value_elements = checked_mul(base_pairs, value_dimension, "value_elements")?;
        totals.value_elements = checked_add(
            totals.value_elements,
            checked_mul(base_value_elements, value_passes, "value_elements")?,
            "value_elements",
        )?;

        let output_elements = product(
            &[query_length, query_heads, value_dimension],
            "output_elements",
        )?;
        totals.output_elements =
            checked_add(totals.output_elements, output_elements, "output_elements")?;

        let q_elements = product(&[query_length, query_heads, qk_dimension], "query_elements")?;
        let q_reloads = if assumptions.reload_query_per_kv_tile {
            checked_mul(score_passes, kv_tiles, "query_reloads")?
        } else {
            score_passes
        };
        let q_read_bits = product(&[q_elements, q_reloads, input_bits], "query_read_bits")?;

        let k_elements = product(&[kv_length, kv_head_reads, qk_dimension], "key_elements")?;
        let k_read_bits = product(
            &[k_elements, query_tiles, score_passes, storage_bits],
            "key_read_bits",
        )?;

        let v_elements = product(
            &[kv_length, kv_head_reads, value_dimension],
            "value_source_elements",
        )?;
        let v_read_bits = product(
            &[v_elements, query_tiles, value_passes, storage_bits],
            "value_read_bits",
        )?;
        totals.read_bits = checked_add(
            totals.read_bits,
            checked_add(
                q_read_bits,
                checked_add(k_read_bits, v_read_bits, "kv_read_bits")?,
                "input_read_bits",
            )?,
            "payload_read_bits",
        )?;

        let write_bits = checked_mul(output_elements, output_bits, "output_write_bits")?;
        totals.write_bits = checked_add(totals.write_bits, write_bits, "payload_write_bits")?;

        let logical_kv_elements = product(
            &[
                kv_length,
                kv_heads,
                checked_add(qk_dimension, value_dimension, "kv_dimensions")?,
            ],
            "kv_cache_elements",
        )?;
        let logical_kv_bits = checked_mul(logical_kv_elements, storage_bits, "kv_cache_bits")?;
        totals.kv_payload_bits = checked_add(
            totals.kv_payload_bits,
            logical_kv_bits,
            "kv_cache_payload_bits",
        )?;
    }

    let score_flops = checked_mul(
        totals.score_pairs,
        operations.score_flops_per_pair,
        "score_flops",
    )?;
    let value_flops = checked_mul(
        totals.value_elements,
        operations.value_flops_per_element,
        "value_flops",
    )?;
    let finalize_flops = checked_mul(
        totals.output_elements,
        operations.finalize_flops_per_output,
        "finalize_flops",
    )?;
    let logical_flops = checked_add(
        score_flops,
        checked_add(value_flops, finalize_flops, "non_score_flops")?,
        "logical_flops",
    )?;
    let transcendental_operations = checked_mul(
        totals.score_pairs,
        operations.transcendentals_per_pair,
        "transcendentals",
    )?;
    let reduction_operations = reduction_operations(implementation, totals.output_elements)?;
    let logical_kv_cache_payload_bits = if matches!(workload.kv_cache(), KvCacheSpec::None) {
        None
    } else {
        Some(totals.kv_payload_bits)
    };

    Ok(EstimatedCostReport {
        score_pairs: totals.score_pairs,
        value_elements: totals.value_elements,
        output_elements: totals.output_elements,
        logical_flops,
        transcendental_operations,
        reduction_operations,
        logical_payload_read_bits: totals.read_bits,
        logical_payload_write_bits: totals.write_bits,
        logical_kv_cache_payload_bits,
        workspace_bytes: implementation.memory().workspace_bytes,
    })
}

#[derive(Debug, Default, Clone, Copy)]
struct RunningTotals {
    score_pairs: u64,
    value_elements: u64,
    output_elements: u64,
    read_bits: u64,
    write_bits: u64,
    kv_payload_bits: u64,
}

fn validate_supported_workload(workload: &WorkloadContract) -> Result<(), CostModelError> {
    workload
        .validate()
        .map_err(|_| CostModelError::Unsupported("invalid workload contract"))?;
    match workload.mask().kind() {
        MaskKind::None | MaskKind::Bidirectional => {}
        MaskKind::Causal => {
            return Err(CostModelError::Unsupported("causal visibility accounting"));
        }
        MaskKind::External { .. } => {
            return Err(CostModelError::Unsupported(
                "external mask visibility accounting",
            ));
        }
    }
    if !matches!(workload.inputs(), InputRepresentation::ExplicitQkv) {
        return Err(CostModelError::Unsupported(
            "non-Q/K/V input representation",
        ));
    }
    if !matches!(workload.kv_representation(), KvRepresentation::Full) {
        return Err(CostModelError::Unsupported(
            "latent/compressed KV representation",
        ));
    }
    if !matches!(workload.state(), StateSpec::Stateless) {
        return Err(CostModelError::Unsupported("recurrent state traffic"));
    }
    if matches!(workload.mode(), WorkloadMode::TrainingBackward) {
        return Err(CostModelError::Unsupported("backward execution"));
    }
    Ok(())
}

fn reduction_operations(
    implementation: &ImplementationPlan,
    output_elements: u64,
) -> Result<u64, CostModelError> {
    match implementation.schedule().partition {
        WorkPartition::SplitKv { partitions } => checked_mul(
            output_elements,
            u64::from(partitions.saturating_sub(1)),
            "reduction_operations",
        ),
        WorkPartition::Serial | WorkPartition::QueryTiles => Ok(0),
    }
}

const fn precision_bits(precision: ScalarPrecision) -> u64 {
    match precision {
        ScalarPrecision::F64 => 64,
        ScalarPrecision::F32 => 32,
        ScalarPrecision::BF16 | ScalarPrecision::F16 => 16,
        ScalarPrecision::F8 | ScalarPrecision::I8 => 8,
        ScalarPrecision::F4 => 4,
    }
}

const fn bits_to_bytes_ceil(bits: u64) -> u64 {
    bits.div_ceil(8)
}

fn to_u64(value: usize, field: &'static str) -> Result<u64, CostModelError> {
    u64::try_from(value).map_err(|_| CostModelError::ArithmeticOverflow(field))
}

fn product(values: &[u64], field: &'static str) -> Result<u64, CostModelError> {
    values
        .iter()
        .try_fold(1_u64, |acc, &value| checked_mul(acc, value, field))
}

fn checked_add(left: u64, right: u64, field: &'static str) -> Result<u64, CostModelError> {
    left.checked_add(right)
        .ok_or(CostModelError::ArithmeticOverflow(field))
}

fn checked_mul(left: u64, right: u64, field: &'static str) -> Result<u64, CostModelError> {
    left.checked_mul(right)
        .ok_or(CostModelError::ArithmeticOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_implementation::{
        AlgorithmPlan, Buffering, ExpStrategy, MemoryLevel, MemoryPlan, ReductionTopology,
        SchedulePlan, TileShape,
    };
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskSpec,
        PrecisionPolicy, SequenceLengths, TensorLayout, WorkloadOptions,
    };

    fn implementation(partition: WorkPartition) -> ImplementationPlan {
        let semantic = ada_core::SemanticId::new(
            ada_core::SemanticFamily::StandardSoftmax,
            "cost-test-softmax",
            1,
        )
        .unwrap();
        let id = ada_core::ImplementationCandidateId::new(semantic, "blocked", 1).unwrap();
        ImplementationPlan::new(
            id,
            AlgorithmPlan::DenseBlocked,
            SchedulePlan {
                tile: TileShape {
                    queries: 2,
                    keys: 4,
                    values: 8,
                },
                partition,
                reduction: ReductionTopology::Tree,
                exp_strategy: ExpStrategy::Standard,
                pipeline_stages: 2,
                vector_width: 4,
                buffering: Buffering::Double,
            },
            MemoryPlan {
                query: MemoryLevel::Shared,
                key: MemoryLevel::Shared,
                value: MemoryLevel::Shared,
                output: MemoryLevel::Global,
                accumulator: MemoryLevel::Register,
                workspace_bytes: 4_096,
                alignment_bytes: 128,
                kv_page_rows: None,
            },
        )
        .unwrap()
    }

    fn workload(query_heads: usize, kv_heads: usize) -> WorkloadContract {
        let grouping = HeadGrouping::from_head_counts(query_heads, kv_heads).unwrap();
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 4, 8).unwrap(),
            query_heads,
            kv_heads,
            qk_dimension: Some(8),
            value_dimension: 8,
            topology: AttentionTopology::SelfAttention,
            head_grouping: grouping,
        })
        .unwrap();
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mask: MaskSpec::none(),
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F32,
                    ScalarPrecision::F32,
                    ScalarPrecision::F32,
                    ScalarPrecision::F32,
                ),
                layout: TensorLayout::row_major(),
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn dense_mha_counts_are_exact_under_declared_model() {
        let report = estimate_cost(
            &workload(2, 2),
            &implementation(WorkPartition::QueryTiles),
            OperationProfile::scaled_dot_softmax(8).unwrap(),
            CostAssumptions::default(),
        )
        .unwrap();
        assert_eq!(report.score_pairs(), 64);
        assert_eq!(report.value_elements(), 512);
        assert_eq!(report.output_elements(), 64);
        assert_eq!(report.transcendental_operations(), 64);
        assert_eq!(report.reduction_operations(), 0);
        assert_eq!(report.workspace_bytes(), 4_096);
    }

    #[test]
    fn gqa_reuse_changes_payload_not_semantic_pair_count() {
        let contract = workload(4, 2);
        let plan = implementation(WorkPartition::QueryTiles);
        let profile = OperationProfile::scaled_dot_softmax(8).unwrap();
        let reused = estimate_cost(&contract, &plan, profile, CostAssumptions::default()).unwrap();
        let no_reuse = estimate_cost(
            &contract,
            &plan,
            profile,
            CostAssumptions {
                reuse_shared_kv_across_query_heads: false,
                ..CostAssumptions::default()
            },
        )
        .unwrap();
        assert_eq!(reused.score_pairs(), no_reuse.score_pairs());
        assert!(reused.logical_payload_read_bits() < no_reuse.logical_payload_read_bits());
    }

    #[test]
    fn split_kv_reductions_are_explicit_estimates() {
        let report = estimate_cost(
            &workload(2, 2),
            &implementation(WorkPartition::SplitKv { partitions: 4 }),
            OperationProfile::scaled_dot_softmax(8).unwrap(),
            CostAssumptions::default(),
        )
        .unwrap();
        assert_eq!(report.reduction_operations(), report.output_elements() * 3);
        let objective = report.to_objective_estimate().unwrap();
        assert_eq!(objective.workspace_bytes, Some(4_096));
        assert_eq!(
            objective.reduction_operations,
            Some(report.output_elements() * 3)
        );
        assert!(objective.bytes_moved.is_some());
        assert_eq!(objective.communication_bytes, None);
    }

    #[test]
    fn query_reload_assumption_is_auditable() {
        let contract = workload(2, 2);
        let plan = implementation(WorkPartition::QueryTiles);
        let profile = OperationProfile::scaled_dot_softmax(8).unwrap();
        let resident =
            estimate_cost(&contract, &plan, profile, CostAssumptions::default()).unwrap();
        let reloaded = estimate_cost(
            &contract,
            &plan,
            profile,
            CostAssumptions {
                reload_query_per_kv_tile: true,
                ..CostAssumptions::default()
            },
        )
        .unwrap();
        assert!(resident.logical_payload_read_bits() < reloaded.logical_payload_read_bits());
    }

    #[test]
    fn unsupported_semantic_dependencies_fail_closed() {
        let mut options = WorkloadOptions::default();
        options.mask = MaskSpec::new(MaskKind::Causal).unwrap();
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 4, 4).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(8),
            value_dimension: 8,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        let contract = WorkloadContract::new(geometry, options).unwrap();
        let result = estimate_cost(
            &contract,
            &implementation(WorkPartition::Serial),
            OperationProfile::scaled_dot_softmax(8).unwrap(),
            CostAssumptions::default(),
        );
        assert_eq!(
            result,
            Err(CostModelError::Unsupported("causal visibility accounting"))
        );
    }

    #[test]
    fn pathological_operation_profile_overflow_is_rejected() {
        let result = estimate_cost(
            &workload(2, 2),
            &implementation(WorkPartition::Serial),
            OperationProfile {
                score_flops_per_pair: u64::MAX,
                transcendentals_per_pair: 1,
                value_flops_per_element: 2,
                finalize_flops_per_output: 1,
            },
            CostAssumptions::default(),
        );
        assert!(matches!(
            result,
            Err(CostModelError::ArithmeticOverflow(_))
        ));
    }
}
