use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::io;

use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::dense_qk_scores;
use ada_a5_hierarchical_bounds::{
    branch_and_bound_entmax_hierarchical_priority_lazy, build_hierarchical_key_index,
};
use ada_a5_real_qk_trace::{TraceRecord, read_trace_file};

const PAGE_SIZE: usize = 16;
const LEAF_DIVISOR: usize = 8;
const ALPHAS: [f64; 2] = [1.5, 2.0];

const EXPECTED_RECORDS: usize = 3_072;
const EXPECTED_GROUPS: usize = 1_536;

const EXPECTED_MODEL: &str = "Qwen/Qwen3-0.6B";
const EXPECTED_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";
const EXPECTED_CAPTURE_ID: &str = "qwen3-0.6b-a2-e3-allheads-wikitext2raw-val16";

const PROBABILITY_TOLERANCE: f64 = 2.0e-10;
const TAU_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    sample_id: String,
    layer: u32,
    kv_head: u32,
    query_position: u64,
    key_start_position: u64,
    head_dim: usize,
    key_count: usize,
}

#[derive(Debug)]
struct HeadMetrics {
    q_head: u32,
    loaded: Vec<bool>,
    support: Vec<bool>,
    probability_error: f64,
    tau_error: f64,
}

#[derive(Debug)]
struct GroupMetrics {
    q0: u32,
    q1: u32,

    visible_rows: usize,

    q_k_sum: usize,
    q_support_sum: usize,

    k_union: usize,
    support_union: usize,

    k_intersection: usize,
    support_intersection: usize,

    probability_error_max: f64,
    tau_error_max: f64,
}

#[derive(Debug, Default)]
struct Aggregate {
    groups: usize,

    visible_rows: u64,

    q_k_sum: u64,
    q_support_sum: u64,

    k_union: u64,
    support_union: u64,

    k_intersection: u64,
    support_intersection: u64,

    groups_with_residual_a2: usize,
    groups_without_residual_a2: usize,

    probability_error_max: f64,
    tau_error_max: f64,
}

impl Aggregate {
    fn add_usize(target: &mut u64, value: usize) {
        *target += u64::try_from(value).expect("A2-E3a counter must fit u64");
    }

    fn record(&mut self, metrics: &GroupMetrics) {
        self.groups += 1;

        Self::add_usize(&mut self.visible_rows, metrics.visible_rows);

        Self::add_usize(&mut self.q_k_sum, metrics.q_k_sum);

        Self::add_usize(&mut self.q_support_sum, metrics.q_support_sum);

        Self::add_usize(&mut self.k_union, metrics.k_union);

        Self::add_usize(&mut self.support_union, metrics.support_union);

        Self::add_usize(&mut self.k_intersection, metrics.k_intersection);

        Self::add_usize(&mut self.support_intersection, metrics.support_intersection);

        if metrics.k_union > metrics.support_union {
            self.groups_with_residual_a2 += 1;
        } else {
            self.groups_without_residual_a2 += 1;
        }

        self.probability_error_max = self
            .probability_error_max
            .max(metrics.probability_error_max);

        self.tau_error_max = self.tau_error_max.max(metrics.tau_error_max);
    }

    #[allow(clippy::cast_precision_loss)]
    fn ratio(numerator: u64, denominator: u64) -> f64 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f64 / denominator as f64
        }
    }

    fn print(&self, scope: &str, dimension: &str, alpha: f64) {
        let k_union_fraction = Self::ratio(self.k_union, self.visible_rows);

        let support_union_fraction = Self::ratio(self.support_union, self.visible_rows);

        let a2_avoidance_after_k = if self.k_union == 0 {
            0.0
        } else {
            1.0 - Self::ratio(self.support_union, self.k_union)
        };

        let total_v_avoidance = 1.0 - support_union_fraction;

        let additional_v_avoidance_after_k = Self::ratio(
            self.k_union.saturating_sub(self.support_union),
            self.visible_rows,
        );

        let k_dedup_factor = Self::ratio(self.q_k_sum, self.k_union);

        let support_dedup_factor = Self::ratio(self.q_support_sum, self.support_union);

        let k_overlap_jaccard = Self::ratio(self.k_intersection, self.k_union);

        let support_overlap_jaccard = Self::ratio(self.support_intersection, self.support_union);

        println!(
            "aggregate,scope={scope},dimension={dimension},\
alpha={alpha:.1},groups={},\
visible_rows={},\
q_k_sum={},q_support_sum={},\
k_union={},support_union={},\
k_intersection={},support_intersection={},\
weighted_k_union_fraction={k_union_fraction:.6},\
weighted_support_union_fraction={support_union_fraction:.6},\
weighted_a2_v_avoidance_after_k={a2_avoidance_after_k:.6},\
weighted_additional_v_avoidance_after_k={additional_v_avoidance_after_k:.6},\
weighted_total_v_avoidance={total_v_avoidance:.6},\
k_gqa_dedup_factor={k_dedup_factor:.6},\
support_gqa_dedup_factor={support_dedup_factor:.6},\
k_overlap_jaccard={k_overlap_jaccard:.6},\
support_overlap_jaccard={support_overlap_jaccard:.6},\
groups_with_residual_a2={},\
groups_without_residual_a2={},\
max_probability_difference={:.3e},\
max_tau_difference={:.3e}",
            self.groups,
            self.visible_rows,
            self.q_k_sum,
            self.q_support_sum,
            self.k_union,
            self.support_union,
            self.k_intersection,
            self.support_intersection,
            self.groups_with_residual_a2,
            self.groups_without_residual_a2,
            self.probability_error_max,
            self.tau_error_max,
        );
    }
}

fn distribution_error(
    reference: &EntmaxDistribution,
    candidate: &EntmaxDistribution,
) -> (f64, f64) {
    let probability_error = reference
        .probabilities
        .iter()
        .zip(candidate.probabilities.iter())
        .map(|(&expected, &actual)| (expected - actual).abs())
        .fold(0.0_f64, f64::max);

    let tau_error = (reference.tau - candidate.tau).abs();

    (probability_error, tau_error)
}

fn support_masks_equal(left: &EntmaxDistribution, right: &EntmaxDistribution) -> bool {
    left.probabilities.len() == right.probabilities.len()
        && left
            .probabilities
            .iter()
            .zip(right.probabilities.iter())
            .all(|(&left_probability, &right_probability)| {
                (left_probability > 0.0) == (right_probability > 0.0)
            })
}

fn support_mask(distribution: &EntmaxDistribution) -> Vec<bool> {
    distribution
        .probabilities
        .iter()
        .map(|&probability| probability > 0.0)
        .collect()
}

fn count_true(mask: &[bool]) -> usize {
    mask.iter().filter(|&&value| value).count()
}

fn count_union(left: &[bool], right: &[bool]) -> usize {
    left.iter()
        .zip(right)
        .filter(|&(left_value, right_value)| *left_value || *right_value)
        .count()
}

fn count_intersection(left: &[bool], right: &[bool]) -> usize {
    left.iter()
        .zip(right)
        .filter(|&(left_value, right_value)| *left_value && *right_value)
        .count()
}

fn usize_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }

    let numerator = u32::try_from(numerator).expect("A2-E3a row count must fit u32");

    let denominator = u32::try_from(denominator).expect("A2-E3a row count must fit u32");

    f64::from(numerator) / f64::from(denominator)
}

fn validate_head_pair(key: &GroupKey, records: &[&TraceRecord]) -> Result<(), String> {
    if records.len() != 2 {
        return Err(format!(
            "GQA group must contain exactly two Q heads; \
sample={:?},layer={},kv_head={},position={},count={}",
            key.sample_id,
            key.layer,
            key.kv_head,
            key.query_position,
            records.len(),
        ));
    }

    let expected_q0 = key
        .kv_head
        .checked_mul(2)
        .ok_or_else(|| "Q-head mapping overflow".to_owned())?;

    let expected_q1 = expected_q0 + 1;

    let mut heads = records
        .iter()
        .map(|record| record.query_head_index)
        .collect::<Vec<_>>();

    heads.sort_unstable();

    if heads != [expected_q0, expected_q1] {
        return Err(format!(
            "unexpected Q heads for KV head {}: {:?}",
            key.kv_head, heads,
        ));
    }

    let first = records[0];
    let second = records[1];

    if first.keys != second.keys {
        return Err(format!(
            "paired Q heads do not carry identical K matrix; \
sample={:?},layer={},kv_head={},position={}",
            key.sample_id, key.layer, key.kv_head, key.query_position,
        ));
    }

    if first.score_scale.to_bits() != second.score_scale.to_bits() {
        return Err("paired Q heads have different score scales".to_owned());
    }

    Ok(())
}

fn measure_head(
    record: &TraceRecord,
    alpha: f64,
    index: &ada_a5_hierarchical_bounds::HierarchicalKeyIndex,
) -> Result<HeadMetrics, String> {
    let case = record
        .to_query_key_case(PAGE_SIZE, alpha)
        .map_err(|error| error.to_string())?;

    let dense_scores = dense_qk_scores(&case).map_err(str::to_owned)?;

    let dense = dense_entmax(&dense_scores, alpha).map_err(str::to_owned)?;

    let priority =
        branch_and_bound_entmax_hierarchical_priority_lazy(&case, index).map_err(str::to_owned)?;

    let (probability_error, tau_error) = distribution_error(&dense, &priority.distribution);

    if probability_error > PROBABILITY_TOLERANCE || tau_error > TAU_TOLERANCE {
        return Err(format!(
            "dense parity exceeded tolerance: \
probability={probability_error:e},\
tau={tau_error:e}"
        ));
    }

    if !support_masks_equal(&dense, &priority.distribution) {
        return Err("dense and priority support masks differ".to_owned());
    }

    let support = support_mask(&priority.distribution);

    if support.len() != priority.loaded_tokens.len() {
        return Err("support and K-loaded masks differ in length".to_owned());
    }

    if support
        .iter()
        .zip(priority.loaded_tokens.iter())
        .any(|(&supported, &loaded)| supported && !loaded)
    {
        return Err("exact support is not a subset of K-loaded".to_owned());
    }

    Ok(HeadMetrics {
        q_head: record.query_head_index,
        loaded: priority.loaded_tokens,
        support,
        probability_error,
        tau_error,
    })
}

fn measure_group(
    key: &GroupKey,
    records: &[&TraceRecord],
    alpha: f64,
) -> Result<GroupMetrics, String> {
    validate_head_pair(key, records)?;

    let leaf_size = PAGE_SIZE.div_ceil(LEAF_DIVISOR);

    let first_record = records[0];

    let index = build_hierarchical_key_index(
        &first_record.keys,
        first_record.head_dim,
        PAGE_SIZE,
        leaf_size,
    )
    .map_err(str::to_owned)?;

    let mut heads = Vec::with_capacity(2);

    for record in records {
        heads.push(
            measure_head(record, alpha, &index)
                .map_err(|message| format!("q_head={}: {message}", record.query_head_index,))?,
        );
    }

    heads.sort_by_key(|metrics| metrics.q_head);

    let q0 = &heads[0];
    let q1 = &heads[1];

    if q0.loaded.len() != key.key_count
        || q1.loaded.len() != key.key_count
        || q0.support.len() != key.key_count
        || q1.support.len() != key.key_count
    {
        return Err("GQA masks do not match visible key count".to_owned());
    }

    let q0_k = count_true(&q0.loaded);
    let q1_k = count_true(&q1.loaded);

    let q0_support = count_true(&q0.support);
    let q1_support = count_true(&q1.support);

    let k_union = count_union(&q0.loaded, &q1.loaded);

    let support_union = count_union(&q0.support, &q1.support);

    let k_intersection = count_intersection(&q0.loaded, &q1.loaded);

    let support_intersection = count_intersection(&q0.support, &q1.support);

    if support_union > k_union {
        return Err("GQA support union exceeds K-loaded union".to_owned());
    }

    if q0_k + q1_k != k_union + k_intersection {
        return Err("K union/intersection identity failed".to_owned());
    }

    if q0_support + q1_support != support_union + support_intersection {
        return Err("support union/intersection identity failed".to_owned());
    }

    Ok(GroupMetrics {
        q0: q0.q_head,
        q1: q1.q_head,
        visible_rows: key.key_count,
        q_k_sum: q0_k + q1_k,
        q_support_sum: q0_support + q1_support,
        k_union,
        support_union,
        k_intersection,
        support_intersection,
        probability_error_max: q0.probability_error.max(q1.probability_error),
        tau_error_max: q0.tau_error.max(q1.tau_error),
    })
}

// This research example intentionally keeps the experiment protocol
// visible in one auditable entry point.
#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: a2_e3a_natural_gqa_union <trace.adaqk>",
        )
    })?;

    let corpus = read_trace_file(path)?;
    let metadata = corpus.metadata();

    if corpus.len() != EXPECTED_RECORDS {
        return Err(io::Error::other(format!(
            "expected {EXPECTED_RECORDS} records, found {}",
            corpus.len(),
        ))
        .into());
    }

    if metadata.model_id != EXPECTED_MODEL {
        return Err(io::Error::other("unexpected model id").into());
    }

    if metadata.model_revision != EXPECTED_REVISION {
        return Err(io::Error::other("unexpected model revision").into());
    }

    if metadata.capture_id != EXPECTED_CAPTURE_ID {
        return Err(io::Error::other("unexpected capture id").into());
    }

    if metadata.source_dtype != "bfloat16" {
        return Err(io::Error::other("unexpected source dtype").into());
    }

    if metadata.tensor_stage != "attention_score_input" {
        return Err(io::Error::other("unexpected tensor stage").into());
    }

    let mut groups = BTreeMap::<GroupKey, Vec<&TraceRecord>>::new();

    let mut samples = BTreeSet::new();
    let mut layers = BTreeSet::new();
    let mut q_heads = BTreeSet::new();
    let mut kv_heads = BTreeSet::new();
    let mut positions = BTreeSet::new();

    for record in corpus.records() {
        samples.insert(record.sample_id.clone());
        layers.insert(record.layer_index);
        q_heads.insert(record.query_head_index);
        kv_heads.insert(record.kv_head_index);
        positions.insert(record.query_position);

        let key = GroupKey {
            sample_id: record.sample_id.clone(),
            layer: record.layer_index,
            kv_head: record.kv_head_index,
            query_position: record.query_position,
            key_start_position: record.key_start_position,
            head_dim: record.head_dim,
            key_count: record.key_count,
        };

        groups.entry(key).or_default().push(record);
    }

    if samples.len() != 16 {
        return Err(io::Error::other("expected 16 samples").into());
    }

    if layers != BTreeSet::from([0, 13, 27]) {
        return Err(io::Error::other("unexpected layer coverage").into());
    }

    if q_heads != (0_u32..16_u32).collect::<BTreeSet<_>>() {
        return Err(io::Error::other("unexpected Q-head coverage").into());
    }

    if kv_heads != (0_u32..8_u32).collect::<BTreeSet<_>>() {
        return Err(io::Error::other("unexpected KV-head coverage").into());
    }

    if positions != BTreeSet::from([63_u64, 127_u64, 255_u64, 511_u64]) {
        return Err(io::Error::other("unexpected position coverage").into());
    }

    if groups.len() != EXPECTED_GROUPS {
        return Err(io::Error::other(format!(
            "expected {EXPECTED_GROUPS} GQA groups, found {}",
            groups.len(),
        ))
        .into());
    }

    println!("survey=ada_a2_e3a_natural_gqa_union");
    println!("synthetic_only=false");
    println!("wall_clock_benchmark=false");
    println!("physical_v_traffic_measured=false");
    println!("gqa_unique_row_accounting=true");
    println!("model_id={:?}", metadata.model_id);
    println!("model_revision={:?}", metadata.model_revision);
    println!("capture_id={:?}", metadata.capture_id);
    println!("source_dtype={:?}", metadata.source_dtype);
    println!("record_count={}", corpus.len());
    println!("sample_count={}", samples.len());
    println!("gqa_group_count={}", groups.len());
    println!("q_heads={}", q_heads.len());
    println!("kv_heads={}", kv_heads.len());
    println!("q_per_kv=2");
    println!("page_size={PAGE_SIZE}");
    println!("leaf_divisor={LEAF_DIVISOR}");
    println!("leaf_size={}", PAGE_SIZE.div_ceil(LEAF_DIVISOR));
    println!("group_alpha_case_count={}", groups.len() * ALPHAS.len());
    println!("head_alpha_case_count={}", corpus.len() * ALPHAS.len());

    let mut global = BTreeMap::<u64, Aggregate>::new();

    let mut by_layer = BTreeMap::<(u32, u64), Aggregate>::new();

    let mut by_position = BTreeMap::<(u64, u64), Aggregate>::new();

    let mut by_kv_head = BTreeMap::<(u32, u64), Aggregate>::new();

    for (group_index, (key, records)) in groups.iter().enumerate() {
        for alpha in ALPHAS {
            let metrics = measure_group(key, records, alpha).map_err(|message| {
                io::Error::other(format!(
                    "group={group_index},\
sample={:?},layer={},kv_head={},\
position={},alpha={alpha:.1}: {message}",
                    key.sample_id, key.layer, key.kv_head, key.query_position,
                ))
            })?;

            let k_avoidance = if metrics.k_union == 0 {
                0.0
            } else {
                1.0 - usize_ratio(metrics.support_union, metrics.k_union)
            };

            let total_avoidance = 1.0 - usize_ratio(metrics.support_union, metrics.visible_rows);

            println!(
                "group,group_index={group_index},\
sample_fingerprint={:016x},\
layer={},kv_head={},q0={},q1={},\
query_position={},key_count={},\
alpha={alpha:.1},\
q_k_sum={},q_support_sum={},\
k_union={},support_union={},\
k_intersection={},support_intersection={},\
a2_v_avoidance_after_k={k_avoidance:.6},\
total_v_avoidance={total_avoidance:.6},\
probability_difference={:.3e},\
tau_difference={:.3e}",
                records[0].sample_fingerprint(),
                key.layer,
                key.kv_head,
                metrics.q0,
                metrics.q1,
                key.query_position,
                key.key_count,
                metrics.q_k_sum,
                metrics.q_support_sum,
                metrics.k_union,
                metrics.support_union,
                metrics.k_intersection,
                metrics.support_intersection,
                metrics.probability_error_max,
                metrics.tau_error_max,
            );

            let alpha_bits = alpha.to_bits();

            global.entry(alpha_bits).or_default().record(&metrics);

            by_layer
                .entry((key.layer, alpha_bits))
                .or_default()
                .record(&metrics);

            by_position
                .entry((key.query_position, alpha_bits))
                .or_default()
                .record(&metrics);

            by_kv_head
                .entry((key.kv_head, alpha_bits))
                .or_default()
                .record(&metrics);
        }
    }

    for (alpha_bits, aggregate) in global {
        aggregate.print("global", "all", f64::from_bits(alpha_bits));
    }

    for ((layer, alpha_bits), aggregate) in by_layer {
        aggregate.print("layer", &layer.to_string(), f64::from_bits(alpha_bits));
    }

    for ((position, alpha_bits), aggregate) in by_position {
        aggregate.print(
            "query_position",
            &position.to_string(),
            f64::from_bits(alpha_bits),
        );
    }

    for ((kv_head, alpha_bits), aggregate) in by_kv_head {
        aggregate.print("kv_head", &kv_head.to_string(), f64::from_bits(alpha_bits));
    }

    println!("survey_status=complete");

    Ok(())
}
