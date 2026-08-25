use std::collections::BTreeMap;
use std::env;
use std::io;

use ada_a2_k_first_v_late::{LogicalKFirstVLateMetrics, logical_k_first_v_late_accounting};
use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::dense_qk_scores;
use ada_a5_hierarchical_bounds::{
    branch_and_bound_entmax_hierarchical_priority_lazy, build_hierarchical_key_index,
};
use ada_a5_real_qk_trace::{TraceRecord, read_trace_file};

const PAGE_SIZE: usize = 16;
const LEAF_DIVISOR: usize = 8;
const ALPHAS: [f64; 2] = [1.5, 2.0];

const PROBABILITY_TOLERANCE: f64 = 2.0e-10;
const TAU_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug)]
struct CaseMetrics {
    logical: LogicalKFirstVLateMetrics,

    bound_evaluations: usize,
    frontier_insertions: usize,
    frontier_min_checks: usize,
    frontier_max_pops: usize,
    threshold_solves: usize,

    probability_error: f64,
    tau_error: f64,
}

#[derive(Debug, Default)]
struct Aggregate {
    cases: usize,

    tokens_total: u64,
    k_loaded: u64,
    k_pruned: u64,
    v_loaded: u64,
    v_skipped_total: u64,
    v_skipped_after_k: u64,

    bound_evaluations: u64,
    frontier_insertions: u64,
    frontier_min_checks: u64,
    frontier_max_pops: u64,
    threshold_solves: u64,

    probability_error_max: f64,
    tau_error_max: f64,
}

impl Aggregate {
    fn add_usize(target: &mut u64, value: usize) {
        *target += u64::try_from(value).expect("A2-E1 counter must fit u64");
    }

    fn record(&mut self, metrics: &CaseMetrics) {
        self.cases += 1;

        Self::add_usize(&mut self.tokens_total, metrics.logical.tokens_total);

        Self::add_usize(&mut self.k_loaded, metrics.logical.k_loaded);

        Self::add_usize(&mut self.k_pruned, metrics.logical.k_pruned);

        Self::add_usize(&mut self.v_loaded, metrics.logical.v_loaded);

        Self::add_usize(&mut self.v_skipped_total, metrics.logical.v_skipped_total);

        Self::add_usize(
            &mut self.v_skipped_after_k,
            metrics.logical.v_skipped_after_k,
        );

        Self::add_usize(&mut self.bound_evaluations, metrics.bound_evaluations);

        Self::add_usize(&mut self.frontier_insertions, metrics.frontier_insertions);

        Self::add_usize(&mut self.frontier_min_checks, metrics.frontier_min_checks);

        Self::add_usize(&mut self.frontier_max_pops, metrics.frontier_max_pops);

        Self::add_usize(&mut self.threshold_solves, metrics.threshold_solves);

        self.probability_error_max = self.probability_error_max.max(metrics.probability_error);

        self.tau_error_max = self.tau_error_max.max(metrics.tau_error);
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
        let frontier_operations =
            self.frontier_insertions + self.frontier_min_checks + self.frontier_max_pops;

        println!(
            "aggregate,scope={scope},dimension={dimension},\
alpha={alpha:.1},page_size={PAGE_SIZE},\
leaf_divisor={LEAF_DIVISOR},cases={},\
tokens_total={},\
k_loaded={},\
k_pruned={},\
v_loaded={},\
v_skipped_total={},\
v_skipped_after_k={},\
weighted_k_load_fraction={:.6},\
weighted_k_pruning_fraction={:.6},\
weighted_v_load_fraction={:.6},\
weighted_v_total_avoidance={:.6},\
weighted_additional_v_avoidance_after_k={:.6},\
weighted_v_avoidance_within_loaded_k={:.6},\
bound_evaluations_per_token={:.6},\
frontier_operations_per_token={:.6},\
mean_threshold_solves={:.6},\
max_probability_difference={:.3e},\
max_tau_difference={:.3e}",
            self.cases,
            self.tokens_total,
            self.k_loaded,
            self.k_pruned,
            self.v_loaded,
            self.v_skipped_total,
            self.v_skipped_after_k,
            Self::ratio(self.k_loaded, self.tokens_total),
            Self::ratio(self.k_pruned, self.tokens_total),
            Self::ratio(self.v_loaded, self.tokens_total),
            Self::ratio(self.v_skipped_total, self.tokens_total,),
            Self::ratio(self.v_skipped_after_k, self.tokens_total,),
            Self::ratio(self.v_skipped_after_k, self.k_loaded,),
            Self::ratio(self.bound_evaluations, self.tokens_total,),
            Self::ratio(frontier_operations, self.tokens_total,),
            Self::ratio(
                self.threshold_solves,
                u64::try_from(self.cases).expect("A2-E1 case count fits u64"),
            ),
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

fn measure_case(record: &TraceRecord, alpha: f64) -> Result<CaseMetrics, String> {
    let case = record
        .to_query_key_case(PAGE_SIZE, alpha)
        .map_err(|error| error.to_string())?;

    let leaf_size = PAGE_SIZE.div_ceil(LEAF_DIVISOR);

    let index = build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)
        .map_err(str::to_owned)?;

    let dense_scores = dense_qk_scores(&case).map_err(str::to_owned)?;

    let dense = dense_entmax(&dense_scores, alpha).map_err(str::to_owned)?;

    let priority =
        branch_and_bound_entmax_hierarchical_priority_lazy(&case, &index).map_err(str::to_owned)?;

    let (probability_error, tau_error) = distribution_error(&dense, &priority.distribution);

    if probability_error > PROBABILITY_TOLERANCE || tau_error > TAU_TOLERANCE {
        return Err(format!(
            "dense parity exceeded tolerance: \
probability={probability_error:e},\
tau={tau_error:e}"
        ));
    }

    if !support_masks_equal(&dense, &priority.distribution) {
        return Err("dense and priority exact support masks differ".to_owned());
    }

    let logical =
        logical_k_first_v_late_accounting(&priority.distribution, &priority.loaded_tokens)
            .map_err(str::to_owned)?;

    if logical.tokens_total != case.key_count() {
        return Err("A2 logical total differs from visible key count".to_owned());
    }

    if logical.k_loaded != priority.metrics.tokens_loaded {
        return Err("A2 K-loaded count differs from A5 metrics".to_owned());
    }

    if logical.k_pruned != priority.metrics.tokens_pruned {
        return Err("A2 K-pruned count differs from A5 metrics".to_owned());
    }

    if logical.tokens_total != logical.k_pruned + logical.v_skipped_after_k + logical.v_loaded {
        return Err("A2 K/V decomposition identity failed".to_owned());
    }

    Ok(CaseMetrics {
        logical,

        bound_evaluations: priority.metrics.bound_evaluations,

        frontier_insertions: priority.metrics.frontier_insertions,

        frontier_min_checks: priority.metrics.frontier_min_checks,

        frontier_max_pops: priority.metrics.frontier_max_pops,

        threshold_solves: priority.metrics.threshold_solves,

        probability_error,
        tau_error,
    })
}

fn print_case(record_index: usize, record: &TraceRecord, alpha: f64, metrics: &CaseMetrics) {
    println!(
        "case,record_index={record_index},\
sample_fingerprint={:016x},\
layer={},query_head={},kv_head={},\
query_position={},key_count={},\
alpha={alpha:.1},page_size={PAGE_SIZE},\
leaf_divisor={LEAF_DIVISOR},\
k_loaded={},k_pruned={},\
v_loaded={},v_skipped_total={},\
v_skipped_after_k={},\
k_pruning_fraction={:.6},\
v_total_avoidance={:.6},\
additional_v_avoidance_after_k={:.6},\
v_avoidance_within_loaded_k={:.6},\
bound_evaluations={},\
frontier_insertions={},\
frontier_min_checks={},\
frontier_max_pops={},\
threshold_solves={},\
probability_difference={:.3e},\
tau_difference={:.3e}",
        record.sample_fingerprint(),
        record.layer_index,
        record.query_head_index,
        record.kv_head_index,
        record.query_position,
        record.key_count,
        metrics.logical.k_loaded,
        metrics.logical.k_pruned,
        metrics.logical.v_loaded,
        metrics.logical.v_skipped_total,
        metrics.logical.v_skipped_after_k,
        metrics.logical.k_pruning_fraction(),
        metrics.logical.v_total_avoidance(),
        metrics.logical.additional_v_avoidance_fraction(),
        metrics.logical.v_avoidance_within_loaded_k(),
        metrics.bound_evaluations,
        metrics.frontier_insertions,
        metrics.frontier_min_checks,
        metrics.frontier_max_pops,
        metrics.threshold_solves,
        metrics.probability_error,
        metrics.tau_error,
    );
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: a2_e1_natural_kv_accounting <trace.adaqk>",
        )
    })?;

    let corpus = read_trace_file(path)?;
    let metadata = corpus.metadata();

    println!("survey=ada_a2_e1_natural_kv_accounting");
    println!("synthetic_only=false");
    println!("wall_clock_benchmark=false");
    println!("physical_v_traffic_measured=false");
    println!("candidate_work_is_logical=true");
    println!("model_id={:?}", metadata.model_id);
    println!("model_revision={:?}", metadata.model_revision);
    println!("capture_id={:?}", metadata.capture_id);
    println!("source_dtype={:?}", metadata.source_dtype);
    println!("tensor_stage={:?}", metadata.tensor_stage);
    println!("record_count={}", corpus.len());
    println!("page_size={PAGE_SIZE}");
    println!("leaf_divisor={LEAF_DIVISOR}");
    println!("leaf_size={}", PAGE_SIZE.div_ceil(LEAF_DIVISOR));
    println!("comparison_case_count={}", corpus.len() * ALPHAS.len());

    let mut global = BTreeMap::<u64, Aggregate>::new();

    let mut by_layer = BTreeMap::<(u32, u64), Aggregate>::new();

    let mut by_head = BTreeMap::<(u32, u64), Aggregate>::new();

    let mut by_position = BTreeMap::<(u64, u64), Aggregate>::new();

    for (record_index, record) in corpus.records().iter().enumerate() {
        for alpha in ALPHAS {
            let metrics = measure_case(record, alpha).map_err(|message| {
                io::Error::other(format!(
                    "record_index={record_index},\
layer={},query_head={},\
query_position={},alpha={alpha:.1}: \
{message}",
                    record.layer_index, record.query_head_index, record.query_position,
                ))
            })?;

            print_case(record_index, record, alpha, &metrics);

            let alpha_bits = alpha.to_bits();

            global.entry(alpha_bits).or_default().record(&metrics);

            by_layer
                .entry((record.layer_index, alpha_bits))
                .or_default()
                .record(&metrics);

            by_head
                .entry((record.query_head_index, alpha_bits))
                .or_default()
                .record(&metrics);

            by_position
                .entry((record.query_position, alpha_bits))
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

    for ((head, alpha_bits), aggregate) in by_head {
        aggregate.print("query_head", &head.to_string(), f64::from_bits(alpha_bits));
    }

    for ((position, alpha_bits), aggregate) in by_position {
        aggregate.print(
            "query_position",
            &position.to_string(),
            f64::from_bits(alpha_bits),
        );
    }

    println!("survey_status=complete");

    Ok(())
}
