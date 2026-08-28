//! Bounded cached-attention reference execution for ADA.
//!
//! This crate executes already-defined [`ada_semantic::SemanticProgram`] rules
//! against explicit KV-cache storage. It is deliberately separate from the
//! semantic IR so cache layout, GQA/MQA head sharing, and logical-to-physical
//! indexing do not become part of semantic identity.
//!
//! The implementation is an offline f64 reference. It makes no latency,
//! bandwidth, cache-hit, occupancy, or hardware-efficiency claim.

#![forbid(unsafe_code)]

use ada_semantic::{AffinityRule, InputTransform, MaskRule, SelectionRule, SemanticProgram, WeightRule};
use ada_workload::{
    AttentionTopology, HeadGrouping, InputRepresentation, KvCacheSpec, KvIndexing,
    KvRepresentation, MaskKind, MatrixLayout, PositionInfo, ScalarPrecision, StateSpec,
    WorkloadContract, WorkloadMode,
};
use std::fmt::{Display, Formatter};

/// Maximum scalar count accepted by one cached reference tensor.
pub const MAX_CACHE_REFERENCE_ELEMENTS: usize = 1 << 25;
/// Maximum logical score evaluations accepted by one cached reference run.
pub const MAX_CACHE_REFERENCE_SCORES: usize = 1 << 22;

/// Cached-reference construction or execution failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheReferenceError {
    /// The workload is outside this bounded executor's supported domain.
    Unsupported(&'static str),
    /// A tensor or metadata vector has an unexpected length.
    ShapeMismatch {
        /// Field whose shape was rejected.
        field: &'static str,
        /// Expected scalar/item count.
        expected: usize,
        /// Actual scalar/item count.
        actual: usize,
    },
    /// An index or cross-field invariant is invalid.
    InvalidField(&'static str),
    /// A checked integer product overflowed or exceeded the executor bound.
    ExceedsLimit(&'static str),
    /// A non-finite input or intermediate was encountered.
    NonFinite(&'static str),
    /// A query has no visible selected keys.
    EmptySelection {
        /// Query-token index within the chunk.
        query: usize,
        /// Query-head index.
        head: usize,
    },
}

impl Display for CacheReferenceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(formatter, "unsupported cached reference: {reason}"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(formatter, "{field} has {actual} items; expected {expected}"),
            Self::InvalidField(field) => write!(formatter, "invalid cached reference field: {field}"),
            Self::ExceedsLimit(field) => write!(formatter, "cached reference limit exceeded: {field}"),
            Self::NonFinite(stage) => write!(formatter, "non-finite cached reference value at {stage}"),
            Self::EmptySelection { query, head } => {
                write!(formatter, "query {query}, head {head} has no visible selected KV rows")
            }
        }
    }
}

impl std::error::Error for CacheReferenceError {}

/// Explicit storage and query data for one cached reference execution.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheReferenceInput {
    /// Row-major query data: `[query_token][query_head][qk_dimension]`.
    pub queries: Vec<f64>,
    /// Physical row-major key cache: `[kv_head][physical_row][qk_dimension]`.
    pub physical_keys: Vec<f64>,
    /// Physical row-major value cache: `[kv_head][physical_row][value_dimension]`.
    pub physical_values: Vec<f64>,
    /// Logical KV row -> physical row mapping for paged caches.
    /// Must be empty for contiguous caches.
    pub logical_to_physical: Vec<usize>,
    /// Absolute logical position for each query token in the decode/chunk.
    pub query_positions: Vec<usize>,
    /// Optional shared logical visibility mask `[query_token][logical_kv_row]`.
    pub external_visibility: Option<Vec<bool>>,
}

/// Deterministic cached-reference result.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheReferenceOutput {
    output: Vec<f64>,
    weights: Vec<f64>,
    selected_keys: Vec<Vec<usize>>,
    query_count: usize,
    query_heads: usize,
    kv_length: usize,
    value_dimension: usize,
}

impl CacheReferenceOutput {
    /// Row-major output `[query_token][query_head][value_dimension]`.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        &self.output
    }

    /// Dense logical weights `[query_token][query_head][logical_kv_row]`.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Selected logical KV indices for each `(query_token, query_head)` row.
    #[must_use]
    pub fn selected_keys(&self) -> &[Vec<usize>] {
        &self.selected_keys
    }

    /// Number of query tokens evaluated.
    #[must_use]
    pub const fn query_count(&self) -> usize {
        self.query_count
    }

    /// Number of query heads evaluated.
    #[must_use]
    pub const fn query_heads(&self) -> usize {
        self.query_heads
    }

    /// Logical KV length.
    #[must_use]
    pub const fn kv_length(&self) -> usize {
        self.kv_length
    }

    /// Value/output head dimension.
    #[must_use]
    pub const fn value_dimension(&self) -> usize {
        self.value_dimension
    }
}

/// Execute an existing semantic against a contiguous or paged full-KV cache.
///
/// Supported reference features are:
///
/// - single-example `Decode` and `ChunkedDecode` workloads;
/// - MHA, MQA, and GQA head mapping;
/// - contiguous or paged full-KV caches;
/// - identity or explicit logical-to-physical indexing;
/// - unmasked, causal, or fixed external visibility;
/// - `All`, `Window`, and deterministic `TopK` selection;
/// - softmax and signed-difference weighting;
/// - identity or row-centering input transform;
/// - row-major f64 input/storage/accumulation/output.
///
/// # Errors
///
/// Returns [`CacheReferenceError`] for unsupported workload features, invalid
/// storage/indexing metadata, shape mismatches, empty visible selections, or
/// non-finite inputs/intermediates.
pub fn evaluate_cached(
    program: &SemanticProgram,
    workload: &WorkloadContract,
    input: &CacheReferenceInput,
) -> Result<CacheReferenceOutput, CacheReferenceError> {
    let context = validate_domain(program, workload, input)?;
    let output_elements = bounded_product(
        bounded_product(context.query_count, context.query_heads, "output rows")?,
        context.value_dimension,
        "output elements",
    )?;
    let weight_elements = bounded_product(
        bounded_product(context.query_count, context.query_heads, "weight rows")?,
        context.kv_length,
        "weight elements",
    )?;
    let mut output = vec![0.0_f64; output_elements];
    let mut weights = vec![0.0_f64; weight_elements];
    let mut selected_keys = Vec::with_capacity(context.query_count * context.query_heads);

    for query in 0..context.query_count {
        for query_head in 0..context.query_heads {
            let kv_head = mapped_kv_head(workload.geometry().head_grouping(), query_head)?;
            let scores = score_row(program, input, &context, query, query_head, kv_head)?;
            let selected = select_logical_keys(program, workload, input, &context, query, query_head, &scores)?;
            let selected_scores = selected.iter().map(|&key| scores[key]).collect::<Vec<_>>();
            let selected_weights = weights_for(program.weight(), &selected_scores)?;
            let row_index = query * context.query_heads + query_head;
            let output_start = row_index * context.value_dimension;
            let weight_start = row_index * context.kv_length;
            for (&logical_key, &weight) in selected.iter().zip(&selected_weights) {
                weights[weight_start + logical_key] = weight;
                let physical = context.physical_row(input, logical_key);
                let value_start = ((kv_head * context.physical_capacity + physical)
                    * context.value_dimension);
                for dimension in 0..context.value_dimension {
                    output[output_start + dimension] +=
                        weight * input.physical_values[value_start + dimension];
                }
            }
            if output[output_start..output_start + context.value_dimension]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(CacheReferenceError::NonFinite("value mixing"));
            }
            selected_keys.push(selected);
        }
    }

    Ok(CacheReferenceOutput {
        output,
        weights,
        selected_keys,
        query_count: context.query_count,
        query_heads: context.query_heads,
        kv_length: context.kv_length,
        value_dimension: context.value_dimension,
    })
}

#[derive(Debug, Clone, Copy)]
struct Context {
    query_count: usize,
    query_heads: usize,
    kv_heads: usize,
    kv_length: usize,
    physical_capacity: usize,
    qk_dimension: usize,
    value_dimension: usize,
    paged: bool,
}

impl Context {
    fn physical_row(self, input: &CacheReferenceInput, logical: usize) -> usize {
        if self.paged {
            input.logical_to_physical[logical]
        } else {
            logical
        }
    }
}

fn validate_domain(
    program: &SemanticProgram,
    workload: &WorkloadContract,
    input: &CacheReferenceInput,
) -> Result<Context, CacheReferenceError> {
    workload
        .validate()
        .map_err(|_| CacheReferenceError::Unsupported("invalid workload contract"))?;
    let geometry = workload.geometry();
    if geometry.sequence_lengths().batch_count() != 1 {
        return Err(CacheReferenceError::Unsupported("reference cache executor is single-example"));
    }
    if !matches!(
        geometry.topology(),
        AttentionTopology::SelfAttention | AttentionTopology::CrossAttention
    ) {
        return Err(CacheReferenceError::Unsupported("historical topology"));
    }
    if !matches!(workload.mode(), WorkloadMode::Decode | WorkloadMode::ChunkedDecode) {
        return Err(CacheReferenceError::Unsupported("workload mode is not decode/chunked-decode"));
    }
    if !matches!(workload.inputs(), InputRepresentation::ExplicitQkv)
        || !matches!(workload.kv_representation(), KvRepresentation::Full)
        || !matches!(workload.state(), StateSpec::Stateless)
    {
        return Err(CacheReferenceError::Unsupported(
            "executor requires explicit QKV, full KV, and stateless semantics",
        ));
    }
    if !matches!(workload.positions(), PositionInfo::None)
        || !matches!(workload.score_bias(), ada_workload::ScoreBiasSpec::None)
    {
        return Err(CacheReferenceError::Unsupported("position or score-bias rules are not executable here"));
    }
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
        return Err(CacheReferenceError::Unsupported("cached reference is explicitly f64"));
    }
    let layout = workload.layout();
    if [layout.query(), layout.key(), layout.value(), layout.output()]
        .into_iter()
        .any(|value| value != MatrixLayout::RowMajor)
    {
        return Err(CacheReferenceError::Unsupported("cached reference requires row-major tensors"));
    }
    validate_mask_binding(program, workload)?;

    let query_count = geometry
        .sequence_lengths()
        .query_length_for(0)
        .ok_or(CacheReferenceError::InvalidField("query length"))?;
    let kv_length = geometry
        .sequence_lengths()
        .kv_length_for(0)
        .ok_or(CacheReferenceError::InvalidField("KV length"))?;
    let qk_dimension = geometry
        .qk_dimension()
        .ok_or(CacheReferenceError::Unsupported("missing Q/K dimension"))?;
    let query_heads = geometry.query_heads();
    let kv_heads = geometry.kv_heads();
    let value_dimension = geometry.value_dimension();

    let (physical_capacity, paged) = match workload.kv_cache() {
        KvCacheSpec::None => return Err(CacheReferenceError::Unsupported("decode requires a KV cache")),
        KvCacheSpec::Contiguous => {
            if !matches!(workload.kv_indexing(), KvIndexing::Identity) {
                return Err(CacheReferenceError::InvalidField("contiguous cache indexing"));
            }
            if !input.logical_to_physical.is_empty() {
                return Err(CacheReferenceError::InvalidField(
                    "contiguous cache must not supply logical_to_physical",
                ));
            }
            (kv_length, false)
        }
        KvCacheSpec::Paged {
            page_size,
            physical_pages,
            ..
        } => {
            if !matches!(workload.kv_indexing(), KvIndexing::LogicalToPhysical { .. }) {
                return Err(CacheReferenceError::InvalidField("paged cache indexing"));
            }
            let capacity = bounded_product(*page_size, *physical_pages, "paged cache capacity")?;
            if input.logical_to_physical.len() != kv_length {
                return Err(CacheReferenceError::ShapeMismatch {
                    field: "logical_to_physical",
                    expected: kv_length,
                    actual: input.logical_to_physical.len(),
                });
            }
            let mut seen = vec![false; capacity];
            for &physical in &input.logical_to_physical {
                if physical >= capacity {
                    return Err(CacheReferenceError::InvalidField("logical_to_physical range"));
                }
                if seen[physical] {
                    return Err(CacheReferenceError::InvalidField("logical_to_physical alias"));
                }
                seen[physical] = true;
            }
            (capacity, true)
        }
    };

    let expected_queries = bounded_product(
        bounded_product(query_count, query_heads, "query rows")?,
        qk_dimension,
        "queries",
    )?;
    let expected_keys = bounded_product(
        bounded_product(kv_heads, physical_capacity, "physical key rows")?,
        qk_dimension,
        "physical_keys",
    )?;
    let expected_values = bounded_product(
        bounded_product(kv_heads, physical_capacity, "physical value rows")?,
        value_dimension,
        "physical_values",
    )?;
    check_len("queries", expected_queries, input.queries.len())?;
    check_len("physical_keys", expected_keys, input.physical_keys.len())?;
    check_len("physical_values", expected_values, input.physical_values.len())?;
    check_len("query_positions", query_count, input.query_positions.len())?;
    if input.queries.iter().chain(&input.physical_keys).chain(&input.physical_values).any(|value| !value.is_finite()) {
        return Err(CacheReferenceError::NonFinite("input"));
    }
    for window in input.query_positions.windows(2) {
        if window[0] >= window[1] {
            return Err(CacheReferenceError::InvalidField("query_positions must be strictly increasing"));
        }
    }
    if input.query_positions.iter().any(|&position| position >= kv_length) {
        return Err(CacheReferenceError::InvalidField("query_positions range"));
    }
    if matches!(workload.mode(), WorkloadMode::Decode)
        && input.query_positions.first().copied() != Some(kv_length - 1)
    {
        return Err(CacheReferenceError::InvalidField(
            "decode query position must be the final logical KV row",
        ));
    }
    if let MaskRule::External { .. } = program.mask() {
        let expected = bounded_product(query_count, kv_length, "external visibility")?;
        let Some(mask) = &input.external_visibility else {
            return Err(CacheReferenceError::InvalidField("missing external_visibility"));
        };
        check_len("external_visibility", expected, mask.len())?;
    } else if input.external_visibility.is_some() {
        return Err(CacheReferenceError::InvalidField(
            "external_visibility supplied for non-external semantic",
        ));
    }
    let score_rows = bounded_product(query_count, query_heads, "score rows")?;
    let score_count = score_rows
        .checked_mul(kv_length)
        .ok_or(CacheReferenceError::ExceedsLimit("score count"))?;
    if score_count > MAX_CACHE_REFERENCE_SCORES {
        return Err(CacheReferenceError::ExceedsLimit("score count"));
    }

    Ok(Context {
        query_count,
        query_heads,
        kv_heads,
        kv_length,
        physical_capacity,
        qk_dimension,
        value_dimension,
        paged,
    })
}

fn validate_mask_binding(
    program: &SemanticProgram,
    workload: &WorkloadContract,
) -> Result<(), CacheReferenceError> {
    let matches = match (program.mask(), workload.mask().kind()) {
        (MaskRule::Unmasked, MaskKind::None | MaskKind::Bidirectional)
        | (MaskRule::Causal, MaskKind::Causal) => true,
        (
            MaskRule::External {
                identity: program_identity,
            },
            MaskKind::External {
                identity: workload_identity,
            },
        ) => program_identity == workload_identity,
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(CacheReferenceError::InvalidField("semantic/workload mask binding"))
    }
}

fn bounded_product(left: usize, right: usize, field: &'static str) -> Result<usize, CacheReferenceError> {
    let value = left
        .checked_mul(right)
        .ok_or(CacheReferenceError::ExceedsLimit(field))?;
    if value > MAX_CACHE_REFERENCE_ELEMENTS {
        return Err(CacheReferenceError::ExceedsLimit(field));
    }
    Ok(value)
}

fn check_len(field: &'static str, expected: usize, actual: usize) -> Result<(), CacheReferenceError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CacheReferenceError::ShapeMismatch {
            field,
            expected,
            actual,
        })
    }
}

fn mapped_kv_head(grouping: HeadGrouping, query_head: usize) -> Result<usize, CacheReferenceError> {
    match grouping {
        HeadGrouping::MultiHead => Ok(query_head),
        HeadGrouping::MultiQuery => Ok(0),
        HeadGrouping::GroupedQuery { queries_per_kv } => {
            if queries_per_kv == 0 {
                return Err(CacheReferenceError::InvalidField("queries_per_kv"));
            }
            Ok(query_head / queries_per_kv)
        }
    }
}

fn score_row(
    program: &SemanticProgram,
    input: &CacheReferenceInput,
    context: &Context,
    query: usize,
    query_head: usize,
    kv_head: usize,
) -> Result<Vec<f64>, CacheReferenceError> {
    if kv_head >= context.kv_heads {
        return Err(CacheReferenceError::InvalidField("mapped KV head"));
    }
    let query_start = ((query * context.query_heads + query_head) * context.qk_dimension);
    let query_row = &input.queries[query_start..query_start + context.qk_dimension];
    let mut scores = Vec::with_capacity(context.kv_length);
    let scale = match program.affinity() {
        AffinityRule::ScaledDotProduct { scale } => scale,
    };
    for logical_key in 0..context.kv_length {
        let physical = context.physical_row(input, logical_key);
        let key_start = ((kv_head * context.physical_capacity + physical) * context.qk_dimension);
        let key_row = &input.physical_keys[key_start..key_start + context.qk_dimension];
        let dot = dot_with_transform(program.input_transform(), query_row, key_row)?;
        let score = dot * scale;
        if !score.is_finite() {
            return Err(CacheReferenceError::NonFinite("affinity"));
        }
        scores.push(score);
    }
    Ok(scores)
}

fn dot_with_transform(
    transform: InputTransform,
    query: &[f64],
    key: &[f64],
) -> Result<f64, CacheReferenceError> {
    match transform {
        InputTransform::Identity => Ok(query.iter().zip(key).map(|(&left, &right)| left * right).sum()),
        InputTransform::CenterRows => {
            let dimension = u32::try_from(query.len())
                .map_err(|_| CacheReferenceError::ExceedsLimit("qk dimension"))?;
            let denominator = f64::from(dimension);
            let query_mean = query.iter().sum::<f64>() / denominator;
            let key_mean = key.iter().sum::<f64>() / denominator;
            let dot = query
                .iter()
                .zip(key)
                .map(|(&left, &right)| (left - query_mean) * (right - key_mean))
                .sum::<f64>();
            if dot.is_finite() {
                Ok(dot)
            } else {
                Err(CacheReferenceError::NonFinite("row centering"))
            }
        }
    }
}

fn select_logical_keys(
    program: &SemanticProgram,
    _workload: &WorkloadContract,
    input: &CacheReferenceInput,
    context: &Context,
    query: usize,
    query_head: usize,
    scores: &[f64],
) -> Result<Vec<usize>, CacheReferenceError> {
    let query_position = input.query_positions[query];
    let mut visible = Vec::new();
    for logical_key in 0..context.kv_length {
        let mask_visible = match program.mask() {
            MaskRule::Unmasked => true,
            MaskRule::Causal => logical_key <= query_position,
            MaskRule::External { .. } => input
                .external_visibility
                .as_ref()
                .is_some_and(|mask| mask[query * context.kv_length + logical_key]),
        };
        let selection_visible = match program.selection() {
            SelectionRule::Window { radius } => query_position.abs_diff(logical_key) <= radius,
            SelectionRule::All | SelectionRule::TopK { .. } => true,
        };
        if mask_visible && selection_visible {
            visible.push(logical_key);
        }
    }
    if visible.is_empty() {
        return Err(CacheReferenceError::EmptySelection {
            query,
            head: query_head,
        });
    }
    if let SelectionRule::TopK { k } = program.selection() {
        if k > visible.len() {
            return Err(CacheReferenceError::InvalidField("TopK exceeds visible logical keys"));
        }
        visible.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        visible.truncate(k);
        visible.sort_unstable();
    }
    Ok(visible)
}

fn weights_for(rule: WeightRule, scores: &[f64]) -> Result<Vec<f64>, CacheReferenceError> {
    match rule {
        WeightRule::Softmax => stable_softmax(scores, 1.0),
        WeightRule::SignedDifference {
            positive_scale,
            negative_scale,
        } => {
            let positive = stable_softmax(scores, positive_scale)?;
            let negative = stable_softmax(scores, negative_scale)?;
            let weights = positive
                .into_iter()
                .zip(negative)
                .map(|(left, right)| left - right)
                .collect::<Vec<_>>();
            if weights.iter().all(|value| value.is_finite()) {
                Ok(weights)
            } else {
                Err(CacheReferenceError::NonFinite("signed-difference weights"))
            }
        }
    }
}

fn stable_softmax(scores: &[f64], scale: f64) -> Result<Vec<f64>, CacheReferenceError> {
    let maximum = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(CacheReferenceError::InvalidField("empty score row"))?;
    if !maximum.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(CacheReferenceError::NonFinite("softmax scale/maximum"));
    }
    let scaled_maximum = maximum * scale;
    let mut weights = Vec::with_capacity(scores.len());
    let mut sum = 0.0_f64;
    for &score in scores {
        let exponent = (score * scale - scaled_maximum).exp();
        if !exponent.is_finite() {
            return Err(CacheReferenceError::NonFinite("softmax exponential"));
        }
        weights.push(exponent);
        sum += exponent;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(CacheReferenceError::NonFinite("softmax normalizer"));
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{SemanticFamily, SemanticId};
    use ada_workload::{
        AttentionGeometry, GeometrySpec, HeadGrouping, MaskSpec, PrecisionPolicy, SequenceLengths,
        TensorLayout, WorkloadOptions,
    };

    fn semantic(name: &str, mask: MaskRule, selection: SelectionRule) -> SemanticProgram {
        SemanticProgram::standard_softmax(
            SemanticId::new(SemanticFamily::StandardSoftmax, name, 1).unwrap(),
            mask,
            selection,
            1.0,
        )
        .unwrap()
    }

    fn workload(
        mode: WorkloadMode,
        query_count: usize,
        kv_length: usize,
        query_heads: usize,
        kv_heads: usize,
        mask: MaskSpec,
        cache: KvCacheSpec,
        indexing: KvIndexing,
    ) -> WorkloadContract {
        let grouping = HeadGrouping::from_head_counts(query_heads, kv_heads).unwrap();
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, query_count, kv_length).unwrap(),
            query_heads,
            kv_heads,
            qk_dimension: Some(1),
            value_dimension: 1,
            topology: AttentionTopology::SelfAttention,
            head_grouping: grouping,
        })
        .unwrap();
        WorkloadContract::new(
            geometry,
            WorkloadOptions {
                mode,
                mask,
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                layout: TensorLayout::row_major(),
                kv_cache: cache,
                kv_indexing: indexing,
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn mqa_decode_shares_one_kv_head_across_query_heads() {
        let workload = workload(
            WorkloadMode::Decode,
            1,
            3,
            2,
            1,
            MaskSpec::none(),
            KvCacheSpec::Contiguous,
            KvIndexing::Identity,
        );
        let output = evaluate_cached(
            &semantic("mqa", MaskRule::Unmasked, SelectionRule::All),
            &workload,
            &CacheReferenceInput {
                queries: vec![0.0, 0.0],
                physical_keys: vec![1.0, 2.0, 3.0],
                physical_values: vec![3.0, 6.0, 9.0],
                logical_to_physical: Vec::new(),
                query_positions: vec![2],
                external_visibility: None,
            },
        )
        .unwrap();
        assert_eq!(output.output(), &[6.0, 6.0]);
        assert_eq!(output.selected_keys(), &[vec![0, 1, 2], vec![0, 1, 2]]);
    }

    #[test]
    fn gqa_maps_query_head_groups_to_distinct_kv_heads() {
        let workload = workload(
            WorkloadMode::Decode,
            1,
            2,
            4,
            2,
            MaskSpec::none(),
            KvCacheSpec::Contiguous,
            KvIndexing::Identity,
        );
        let output = evaluate_cached(
            &semantic("gqa", MaskRule::Unmasked, SelectionRule::All),
            &workload,
            &CacheReferenceInput {
                queries: vec![0.0; 4],
                physical_keys: vec![0.0; 4],
                physical_values: vec![2.0, 4.0, 10.0, 14.0],
                logical_to_physical: Vec::new(),
                query_positions: vec![1],
                external_visibility: None,
            },
        )
        .unwrap();
        assert_eq!(output.output(), &[3.0, 3.0, 12.0, 12.0]);
    }

    #[test]
    fn paged_mapping_preserves_logical_attention_order() {
        let workload = workload(
            WorkloadMode::Decode,
            1,
            3,
            1,
            1,
            MaskSpec::none(),
            KvCacheSpec::Paged {
                page_size: 2,
                physical_pages: 2,
                block_table_identity: "table".into(),
            },
            KvIndexing::LogicalToPhysical {
                identity: "logical-map".into(),
            },
        );
        let output = evaluate_cached(
            &semantic("paged", MaskRule::Unmasked, SelectionRule::All),
            &workload,
            &CacheReferenceInput {
                queries: vec![0.0],
                physical_keys: vec![30.0, 10.0, 20.0, 99.0],
                physical_values: vec![9.0, 3.0, 6.0, 99.0],
                logical_to_physical: vec![1, 2, 0],
                query_positions: vec![2],
                external_visibility: None,
            },
        )
        .unwrap();
        assert_eq!(output.output(), &[6.0]);
        assert_eq!(output.selected_keys(), &[vec![0, 1, 2]]);
    }

    #[test]
    fn causal_chunked_decode_uses_absolute_query_positions() {
        let workload = workload(
            WorkloadMode::ChunkedDecode,
            2,
            4,
            1,
            1,
            MaskSpec::new(MaskKind::Causal).unwrap(),
            KvCacheSpec::Contiguous,
            KvIndexing::Identity,
        );
        let output = evaluate_cached(
            &semantic("causal-chunk", MaskRule::Causal, SelectionRule::All),
            &workload,
            &CacheReferenceInput {
                queries: vec![0.0, 0.0],
                physical_keys: vec![0.0; 4],
                physical_values: vec![1.0, 3.0, 5.0, 7.0],
                logical_to_physical: Vec::new(),
                query_positions: vec![2, 3],
                external_visibility: None,
            },
        )
        .unwrap();
        assert_eq!(output.output(), &[3.0, 4.0]);
        assert_eq!(output.selected_keys(), &[vec![0, 1, 2], vec![0, 1, 2, 3]]);
    }

    #[test]
    fn paged_mapping_rejects_physical_aliases() {
        let workload = workload(
            WorkloadMode::Decode,
            1,
            2,
            1,
            1,
            MaskSpec::none(),
            KvCacheSpec::Paged {
                page_size: 2,
                physical_pages: 1,
                block_table_identity: "table".into(),
            },
            KvIndexing::LogicalToPhysical {
                identity: "logical-map".into(),
            },
        );
        let result = evaluate_cached(
            &semantic("alias", MaskRule::Unmasked, SelectionRule::All),
            &workload,
            &CacheReferenceInput {
                queries: vec![0.0],
                physical_keys: vec![0.0; 2],
                physical_values: vec![1.0, 2.0],
                logical_to_physical: vec![0, 0],
                query_positions: vec![1],
                external_visibility: None,
            },
        );
        assert_eq!(
            result,
            Err(CacheReferenceError::InvalidField("logical_to_physical alias"))
        );
    }

    #[test]
    fn topk_fails_closed_when_causal_visibility_is_too_small() {
        let workload = workload(
            WorkloadMode::ChunkedDecode,
            1,
            3,
            1,
            1,
            MaskSpec::new(MaskKind::Causal).unwrap(),
            KvCacheSpec::Contiguous,
            KvIndexing::Identity,
        );
        let result = evaluate_cached(
            &semantic("topk", MaskRule::Causal, SelectionRule::TopK { k: 2 }),
            &workload,
            &CacheReferenceInput {
                queries: vec![1.0],
                physical_keys: vec![1.0, 2.0, 3.0],
                physical_values: vec![1.0, 2.0, 3.0],
                logical_to_physical: Vec::new(),
                query_positions: vec![0],
                external_visibility: None,
            },
        );
        assert_eq!(
            result,
            Err(CacheReferenceError::InvalidField(
                "TopK exceeds visible logical keys"
            ))
        );
    }
}
