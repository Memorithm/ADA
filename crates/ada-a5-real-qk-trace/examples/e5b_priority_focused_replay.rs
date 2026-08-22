use std::collections::BTreeMap;
use std::env;
use std::io;

use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::{QueryKeyPagedCase, dense_qk_scores};
use ada_a5_hierarchical_bounds::{
    LazyHierarchicalResult, PriorityLazyHierarchicalResult,
    branch_and_bound_entmax_hierarchical_lazy, branch_and_bound_entmax_hierarchical_priority_lazy,
    build_hierarchical_key_index,
};
use ada_a5_real_qk_trace::{TraceRecord, read_trace_file};

const SUPPORT_EPSILON: f64 = 1.0e-12;
const PROBABILITY_TOLERANCE: f64 = 2.0e-10;
const TAU_TOLERANCE: f64 = 1.0e-10;

const PAGE_SIZE: usize = 16;
const LEAF_DIVISOR: usize = 8;
const ALPHAS: [f64; 2] = [1.5, 2.0];

#[derive(Debug)]
struct CaseMetrics {
    token_count: usize,
    support_tokens: usize,

    historical_loaded: usize,
    historical_pruned: usize,
    priority_loaded: usize,
    priority_pruned: usize,

    loaded_set_match: bool,
    loaded_set_symmetric_difference: usize,
    distribution_bitwise_match: bool,

    nodes_total: usize,

    historical_bound_evaluations: usize,
    historical_bound_cache_hits: usize,

    priority_bound_evaluations: usize,
    priority_nodes_never_evaluated: usize,
    priority_frontier_insertions: usize,
    priority_frontier_min_checks: usize,
    priority_frontier_max_pops: usize,

    historical_nodes_expanded: usize,
    priority_nodes_expanded: usize,
    historical_threshold_solves: usize,
    priority_threshold_solves: usize,

    historical_probability_error: f64,
    priority_probability_error: f64,
    historical_tau_error: f64,
    priority_tau_error: f64,

    historical_priority_probability_difference: f64,
    historical_priority_tau_difference: f64,
}

#[derive(Debug, Default)]
struct Aggregate {
    cases: usize,
    loaded_set_matches: usize,
    distribution_bitwise_matches: usize,
    loaded_set_symmetric_difference_sum: f64,

    token_count_sum: f64,
    support_tokens_sum: f64,

    historical_loaded_sum: f64,
    historical_pruned_sum: f64,
    priority_loaded_sum: f64,
    priority_pruned_sum: f64,

    nodes_total_sum: f64,

    historical_bound_evaluations_sum: f64,
    historical_bound_cache_hits_sum: f64,

    priority_bound_evaluations_sum: f64,
    priority_nodes_never_evaluated_sum: f64,
    priority_frontier_insertions_sum: f64,
    priority_frontier_min_checks_sum: f64,
    priority_frontier_max_pops_sum: f64,

    historical_nodes_expanded_sum: f64,
    priority_nodes_expanded_sum: f64,
    historical_threshold_solves_sum: f64,
    priority_threshold_solves_sum: f64,

    historical_probability_error_max: f64,
    priority_probability_error_max: f64,
    historical_tau_error_max: f64,
    priority_tau_error_max: f64,

    historical_priority_probability_difference_max: f64,
    historical_priority_tau_difference_max: f64,
}

impl Aggregate {
    fn record(&mut self, metrics: &CaseMetrics) {
        self.cases += 1;

        if metrics.loaded_set_match {
            self.loaded_set_matches += 1;
        }

        if metrics.distribution_bitwise_match {
            self.distribution_bitwise_matches += 1;
        }

        self.loaded_set_symmetric_difference_sum +=
            usize_as_f64(metrics.loaded_set_symmetric_difference);

        self.token_count_sum += usize_as_f64(metrics.token_count);
        self.support_tokens_sum += usize_as_f64(metrics.support_tokens);

        self.historical_loaded_sum += usize_as_f64(metrics.historical_loaded);
        self.historical_pruned_sum += usize_as_f64(metrics.historical_pruned);
        self.priority_loaded_sum += usize_as_f64(metrics.priority_loaded);
        self.priority_pruned_sum += usize_as_f64(metrics.priority_pruned);

        self.nodes_total_sum += usize_as_f64(metrics.nodes_total);

        self.historical_bound_evaluations_sum += usize_as_f64(metrics.historical_bound_evaluations);

        self.historical_bound_cache_hits_sum += usize_as_f64(metrics.historical_bound_cache_hits);

        self.priority_bound_evaluations_sum += usize_as_f64(metrics.priority_bound_evaluations);

        self.priority_nodes_never_evaluated_sum +=
            usize_as_f64(metrics.priority_nodes_never_evaluated);

        self.priority_frontier_insertions_sum += usize_as_f64(metrics.priority_frontier_insertions);

        self.priority_frontier_min_checks_sum += usize_as_f64(metrics.priority_frontier_min_checks);

        self.priority_frontier_max_pops_sum += usize_as_f64(metrics.priority_frontier_max_pops);

        self.historical_nodes_expanded_sum += usize_as_f64(metrics.historical_nodes_expanded);

        self.priority_nodes_expanded_sum += usize_as_f64(metrics.priority_nodes_expanded);

        self.historical_threshold_solves_sum += usize_as_f64(metrics.historical_threshold_solves);

        self.priority_threshold_solves_sum += usize_as_f64(metrics.priority_threshold_solves);

        self.historical_probability_error_max = self
            .historical_probability_error_max
            .max(metrics.historical_probability_error);

        self.priority_probability_error_max = self
            .priority_probability_error_max
            .max(metrics.priority_probability_error);

        self.historical_tau_error_max = self
            .historical_tau_error_max
            .max(metrics.historical_tau_error);

        self.priority_tau_error_max = self.priority_tau_error_max.max(metrics.priority_tau_error);

        self.historical_priority_probability_difference_max = self
            .historical_priority_probability_difference_max
            .max(metrics.historical_priority_probability_difference);

        self.historical_priority_tau_difference_max = self
            .historical_priority_tau_difference_max
            .max(metrics.historical_priority_tau_difference);
    }

    fn ratio_or_zero(numerator: f64, denominator: f64) -> f64 {
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn print(&self, scope: &str, dimension: &str, alpha: f64) {
        let cases = usize_as_f64(self.cases);

        let historical_score_avoidance = 1.0 - self.historical_loaded_sum / self.token_count_sum;

        let priority_score_avoidance = 1.0 - self.priority_loaded_sum / self.token_count_sum;

        let historical_bound_avoidance =
            1.0 - self.historical_bound_evaluations_sum / self.nodes_total_sum;

        let priority_bound_avoidance =
            1.0 - self.priority_bound_evaluations_sum / self.nodes_total_sum;

        let historical_bound_requests =
            self.historical_bound_evaluations_sum + self.historical_bound_cache_hits_sum;

        let priority_frontier_operations = self.priority_frontier_insertions_sum
            + self.priority_frontier_min_checks_sum
            + self.priority_frontier_max_pops_sum;

        let priority_combined_logical_actions =
            self.priority_bound_evaluations_sum + priority_frontier_operations;

        println!(
            "aggregate,scope={scope},dimension={dimension},\
alpha={alpha:.1},page_size={PAGE_SIZE},leaf_divisor={LEAF_DIVISOR},\
cases={},\
loaded_set_match_fraction={:.6},\
distribution_bitwise_match_fraction={:.6},\
mean_loaded_set_symmetric_difference={:.6},\
weighted_support_fraction={:.6},\
historical_weighted_score_avoidance={:.6},\
priority_weighted_score_avoidance={:.6},\
priority_score_avoidance_delta={:.6},\
historical_weighted_bound_avoidance={:.6},\
priority_weighted_bound_avoidance={:.6},\
historical_bound_evaluations_per_pruned_token={:.6},\
priority_bound_evaluations_per_pruned_token={:.6},\
historical_bound_requests_per_pruned_token={:.6},\
priority_frontier_operations_per_pruned_token={:.6},\
priority_combined_logical_actions_per_pruned_token={:.6},\
historical_mean_bound_cache_hits={:.6},\
priority_mean_nodes_never_evaluated={:.6},\
priority_mean_frontier_insertions={:.6},\
priority_mean_frontier_min_checks={:.6},\
priority_mean_frontier_max_pops={:.6},\
historical_mean_nodes_expanded={:.6},\
priority_mean_nodes_expanded={:.6},\
historical_mean_threshold_solves={:.6},\
priority_mean_threshold_solves={:.6},\
max_historical_probability_difference={:.3e},\
max_priority_probability_difference={:.3e},\
max_historical_tau_difference={:.3e},\
max_priority_tau_difference={:.3e},\
max_historical_priority_probability_difference={:.3e},\
max_historical_priority_tau_difference={:.3e}",
            self.cases,
            usize_as_f64(self.loaded_set_matches) / cases,
            usize_as_f64(self.distribution_bitwise_matches) / cases,
            self.loaded_set_symmetric_difference_sum / cases,
            self.support_tokens_sum / self.token_count_sum,
            historical_score_avoidance,
            priority_score_avoidance,
            priority_score_avoidance - historical_score_avoidance,
            historical_bound_avoidance,
            priority_bound_avoidance,
            Self::ratio_or_zero(
                self.historical_bound_evaluations_sum,
                self.historical_pruned_sum,
            ),
            Self::ratio_or_zero(
                self.priority_bound_evaluations_sum,
                self.priority_pruned_sum,
            ),
            Self::ratio_or_zero(historical_bound_requests, self.historical_pruned_sum,),
            Self::ratio_or_zero(priority_frontier_operations, self.priority_pruned_sum,),
            Self::ratio_or_zero(priority_combined_logical_actions, self.priority_pruned_sum,),
            self.historical_bound_cache_hits_sum / cases,
            self.priority_nodes_never_evaluated_sum / cases,
            self.priority_frontier_insertions_sum / cases,
            self.priority_frontier_min_checks_sum / cases,
            self.priority_frontier_max_pops_sum / cases,
            self.historical_nodes_expanded_sum / cases,
            self.priority_nodes_expanded_sum / cases,
            self.historical_threshold_solves_sum / cases,
            self.priority_threshold_solves_sum / cases,
            self.historical_probability_error_max,
            self.priority_probability_error_max,
            self.historical_tau_error_max,
            self.priority_tau_error_max,
            self.historical_priority_probability_difference_max,
            self.historical_priority_tau_difference_max,
        );
    }
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("E5b replay dimensions fit in u32"))
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

fn distributions_bitwise_equal(left: &EntmaxDistribution, right: &EntmaxDistribution) -> bool {
    left.tau.to_bits() == right.tau.to_bits()
        && left.probabilities.len() == right.probabilities.len()
        && left
            .probabilities
            .iter()
            .zip(right.probabilities.iter())
            .all(|(&left_value, &right_value)| left_value.to_bits() == right_value.to_bits())
}

fn check_dense_tolerance(
    probability_error: f64,
    tau_error: f64,
    label: &str,
) -> Result<(), String> {
    if probability_error > PROBABILITY_TOLERANCE || tau_error > TAU_TOLERANCE {
        Err(format!(
            "ADA-A5 E5b {label} dense tolerance exceeded: \
probability={probability_error:e}, tau={tau_error:e}"
        ))
    } else {
        Ok(())
    }
}

fn verify_dense_support(
    dense: &EntmaxDistribution,
    historical: &LazyHierarchicalResult,
    priority: &PriorityLazyHierarchicalResult,
) -> Result<usize, String> {
    let mut support_tokens = 0;

    for (token, &probability) in dense.probabilities.iter().enumerate() {
        if probability <= SUPPORT_EPSILON {
            continue;
        }

        support_tokens += 1;

        if !historical.loaded_tokens[token] {
            return Err("ADA-A5 E5b historical lazy pruned dense support".to_owned());
        }

        if !priority.loaded_tokens[token] {
            return Err("ADA-A5 E5b priority lazy pruned dense support".to_owned());
        }
    }

    Ok(support_tokens)
}

fn symmetric_difference_count(left: &[bool], right: &[bool]) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(left_value, right_value)| left_value != right_value)
        .count()
}

fn measure_case(case: &QueryKeyPagedCase) -> Result<CaseMetrics, String> {
    let leaf_size = PAGE_SIZE.div_ceil(LEAF_DIVISOR);

    let index = build_hierarchical_key_index(&case.keys, case.head_dim, PAGE_SIZE, leaf_size)
        .map_err(str::to_owned)?;

    let dense_scores = dense_qk_scores(case).map_err(str::to_owned)?;

    let dense = dense_entmax(&dense_scores, case.alpha).map_err(str::to_owned)?;

    let historical =
        branch_and_bound_entmax_hierarchical_lazy(case, &index).map_err(str::to_owned)?;

    let priority =
        branch_and_bound_entmax_hierarchical_priority_lazy(case, &index).map_err(str::to_owned)?;

    let (historical_probability_error, historical_tau_error) =
        distribution_error(&dense, &historical.distribution);

    let (priority_probability_error, priority_tau_error) =
        distribution_error(&dense, &priority.distribution);

    check_dense_tolerance(
        historical_probability_error,
        historical_tau_error,
        "historical",
    )?;

    check_dense_tolerance(priority_probability_error, priority_tau_error, "priority")?;

    let support_tokens = verify_dense_support(&dense, &historical, &priority)?;

    let (historical_priority_probability_difference, historical_priority_tau_difference) =
        distribution_error(&historical.distribution, &priority.distribution);

    let loaded_set_symmetric_difference =
        symmetric_difference_count(&historical.loaded_tokens, &priority.loaded_tokens);

    Ok(CaseMetrics {
        token_count: case.key_count(),
        support_tokens,

        historical_loaded: historical.metrics.tokens_loaded,
        historical_pruned: historical.metrics.tokens_pruned,

        priority_loaded: priority.metrics.tokens_loaded,
        priority_pruned: priority.metrics.tokens_pruned,

        loaded_set_match: loaded_set_symmetric_difference == 0,

        loaded_set_symmetric_difference,

        distribution_bitwise_match: distributions_bitwise_equal(
            &historical.distribution,
            &priority.distribution,
        ),

        nodes_total: historical.metrics.nodes_total,

        historical_bound_evaluations: historical.metrics.bound_evaluations,

        historical_bound_cache_hits: historical.metrics.bound_cache_hits,

        priority_bound_evaluations: priority.metrics.bound_evaluations,

        priority_nodes_never_evaluated: priority.metrics.nodes_never_evaluated,

        priority_frontier_insertions: priority.metrics.frontier_insertions,

        priority_frontier_min_checks: priority.metrics.frontier_min_checks,

        priority_frontier_max_pops: priority.metrics.frontier_max_pops,

        historical_nodes_expanded: historical.metrics.nodes_expanded,

        priority_nodes_expanded: priority.metrics.nodes_expanded,

        historical_threshold_solves: historical.metrics.threshold_solves,

        priority_threshold_solves: priority.metrics.threshold_solves,

        historical_probability_error,
        priority_probability_error,
        historical_tau_error,
        priority_tau_error,

        historical_priority_probability_difference,
        historical_priority_tau_difference,
    })
}

fn print_case(record_index: usize, record: &TraceRecord, alpha: f64, metrics: &CaseMetrics) {
    println!(
        "case,record_index={record_index},\
sample_fingerprint={:016x},\
layer={},query_head={},kv_head={},query_position={},\
key_count={},alpha={alpha:.1},\
page_size={PAGE_SIZE},leaf_divisor={LEAF_DIVISOR},\
loaded_set_match={},\
loaded_set_symmetric_difference={},\
distribution_bitwise_match={},\
historical_tokens_loaded={},\
priority_tokens_loaded={},\
historical_tokens_pruned={},\
priority_tokens_pruned={},\
historical_bound_evaluations={},\
historical_bound_cache_hits={},\
priority_bound_evaluations={},\
priority_nodes_never_evaluated={},\
priority_frontier_insertions={},\
priority_frontier_min_checks={},\
priority_frontier_max_pops={},\
historical_nodes_expanded={},\
priority_nodes_expanded={},\
historical_threshold_solves={},\
priority_threshold_solves={},\
historical_probability_difference={:.3e},\
priority_probability_difference={:.3e},\
historical_tau_difference={:.3e},\
priority_tau_difference={:.3e},\
historical_priority_probability_difference={:.3e},\
historical_priority_tau_difference={:.3e}",
        record.sample_fingerprint(),
        record.layer_index,
        record.query_head_index,
        record.kv_head_index,
        record.query_position,
        record.key_count,
        metrics.loaded_set_match,
        metrics.loaded_set_symmetric_difference,
        metrics.distribution_bitwise_match,
        metrics.historical_loaded,
        metrics.priority_loaded,
        metrics.historical_pruned,
        metrics.priority_pruned,
        metrics.historical_bound_evaluations,
        metrics.historical_bound_cache_hits,
        metrics.priority_bound_evaluations,
        metrics.priority_nodes_never_evaluated,
        metrics.priority_frontier_insertions,
        metrics.priority_frontier_min_checks,
        metrics.priority_frontier_max_pops,
        metrics.historical_nodes_expanded,
        metrics.priority_nodes_expanded,
        metrics.historical_threshold_solves,
        metrics.priority_threshold_solves,
        metrics.historical_probability_error,
        metrics.priority_probability_error,
        metrics.historical_tau_error,
        metrics.priority_tau_error,
        metrics.historical_priority_probability_difference,
        metrics.historical_priority_tau_difference,
    );
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: e5b_priority_focused_replay <trace.adaqk>",
        )
    })?;

    let corpus = read_trace_file(path)?;
    let metadata = corpus.metadata();

    println!("survey=ada_a5_e5b_priority_focused_replay");
    println!("synthetic_only=false");
    println!("wall_clock_benchmark=false");
    println!("candidate_work_is_logical=true");
    println!("priority_structure=BTreeSet");
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

    for (record_index, record) in corpus.records().iter().enumerate() {
        for alpha in ALPHAS {
            let case = record
                .to_query_key_case(PAGE_SIZE, alpha)
                .map_err(|error| io::Error::other(error.to_string()))?;

            let metrics = measure_case(&case).map_err(|message| {
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
        }
    }

    for (alpha_bits, aggregate) in global {
        aggregate.print("global", "all", f64::from_bits(alpha_bits));
    }

    for ((layer, alpha_bits), aggregate) in by_layer {
        aggregate.print("layer", &layer.to_string(), f64::from_bits(alpha_bits));
    }

    println!("survey_status=complete");

    Ok(())
}
