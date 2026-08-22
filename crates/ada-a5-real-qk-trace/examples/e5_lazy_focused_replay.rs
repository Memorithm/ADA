use std::collections::BTreeMap;
use std::env;
use std::io;

use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::{QueryKeyPagedCase, dense_qk_scores};
use ada_a5_hierarchical_bounds::{
    HierarchicalResult, LazyHierarchicalResult, branch_and_bound_entmax_hierarchical,
    branch_and_bound_entmax_hierarchical_lazy, build_hierarchical_key_index,
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
    support_fraction: f64,
    score_avoidance: f64,
    eager_bound_evaluation_fraction: f64,
    lazy_bound_evaluation_fraction: f64,
    lazy_bound_avoidance: f64,
    nodes_total: usize,
    lazy_bound_evaluations: usize,
    nodes_never_evaluated: usize,
    bound_cache_hits: usize,
    bound_evaluations_per_loaded_token: f64,
    bound_evaluations_per_pruned_token: f64,
    nodes_expanded: usize,
    subtrees_pruned: usize,
    threshold_solves: usize,
    eager_probability_error: f64,
    lazy_probability_error: f64,
    eager_tau_error: f64,
    lazy_tau_error: f64,
}

#[derive(Debug, Default)]
struct Aggregate {
    cases: usize,
    support_fraction_sum: f64,
    score_avoidance_sum: f64,
    eager_bound_evaluation_fraction_sum: f64,
    lazy_bound_evaluation_fraction_sum: f64,
    lazy_bound_avoidance_sum: f64,
    nodes_total_sum: f64,
    lazy_bound_evaluations_sum: f64,
    nodes_never_evaluated_sum: f64,
    bound_cache_hits_sum: f64,
    bound_evaluations_per_loaded_token_sum: f64,
    bound_evaluations_per_pruned_token_sum: f64,
    nodes_expanded_sum: f64,
    subtrees_pruned_sum: f64,
    threshold_solves_sum: f64,
    eager_probability_error_max: f64,
    lazy_probability_error_max: f64,
    eager_tau_error_max: f64,
    lazy_tau_error_max: f64,
}

impl Aggregate {
    fn record(&mut self, metrics: &CaseMetrics) {
        self.cases += 1;
        self.support_fraction_sum += metrics.support_fraction;
        self.score_avoidance_sum += metrics.score_avoidance;
        self.eager_bound_evaluation_fraction_sum += metrics.eager_bound_evaluation_fraction;
        self.lazy_bound_evaluation_fraction_sum += metrics.lazy_bound_evaluation_fraction;
        self.lazy_bound_avoidance_sum += metrics.lazy_bound_avoidance;
        self.nodes_total_sum += usize_as_f64(metrics.nodes_total);
        self.lazy_bound_evaluations_sum += usize_as_f64(metrics.lazy_bound_evaluations);
        self.nodes_never_evaluated_sum += usize_as_f64(metrics.nodes_never_evaluated);
        self.bound_cache_hits_sum += usize_as_f64(metrics.bound_cache_hits);
        self.bound_evaluations_per_loaded_token_sum += metrics.bound_evaluations_per_loaded_token;
        self.bound_evaluations_per_pruned_token_sum += metrics.bound_evaluations_per_pruned_token;
        self.nodes_expanded_sum += usize_as_f64(metrics.nodes_expanded);
        self.subtrees_pruned_sum += usize_as_f64(metrics.subtrees_pruned);
        self.threshold_solves_sum += usize_as_f64(metrics.threshold_solves);

        self.eager_probability_error_max = self
            .eager_probability_error_max
            .max(metrics.eager_probability_error);
        self.lazy_probability_error_max = self
            .lazy_probability_error_max
            .max(metrics.lazy_probability_error);
        self.eager_tau_error_max = self.eager_tau_error_max.max(metrics.eager_tau_error);
        self.lazy_tau_error_max = self.lazy_tau_error_max.max(metrics.lazy_tau_error);
    }

    fn print(&self, scope: &str, dimension: &str, alpha: f64) {
        let denominator = usize_as_f64(self.cases);

        println!(
            "aggregate,scope={scope},dimension={dimension},alpha={alpha:.1},\
page_size={PAGE_SIZE},leaf_divisor={LEAF_DIVISOR},cases={},\
mean_support_fraction={:.6},mean_score_avoidance={:.6},\
mean_eager_bound_evaluation_fraction={:.6},\
mean_lazy_bound_evaluation_fraction={:.6},\
mean_lazy_bound_avoidance={:.6},\
mean_nodes_total={:.6},mean_lazy_bound_evaluations={:.6},\
mean_nodes_never_evaluated={:.6},mean_bound_cache_hits={:.6},\
mean_bound_evaluations_per_loaded_token={:.6},\
mean_bound_evaluations_per_pruned_token={:.6},\
mean_nodes_expanded={:.6},mean_subtrees_pruned={:.6},\
mean_threshold_solves={:.6},\
max_eager_probability_difference={:.3e},\
max_lazy_probability_difference={:.3e},\
max_eager_tau_difference={:.3e},max_lazy_tau_difference={:.3e}",
            self.cases,
            self.support_fraction_sum / denominator,
            self.score_avoidance_sum / denominator,
            self.eager_bound_evaluation_fraction_sum / denominator,
            self.lazy_bound_evaluation_fraction_sum / denominator,
            self.lazy_bound_avoidance_sum / denominator,
            self.nodes_total_sum / denominator,
            self.lazy_bound_evaluations_sum / denominator,
            self.nodes_never_evaluated_sum / denominator,
            self.bound_cache_hits_sum / denominator,
            self.bound_evaluations_per_loaded_token_sum / denominator,
            self.bound_evaluations_per_pruned_token_sum / denominator,
            self.nodes_expanded_sum / denominator,
            self.subtrees_pruned_sum / denominator,
            self.threshold_solves_sum / denominator,
            self.eager_probability_error_max,
            self.lazy_probability_error_max,
            self.eager_tau_error_max,
            self.lazy_tau_error_max,
        );
    }
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("E5 replay dimensions fit in u32"))
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
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
}

fn check_dense_tolerance(
    probability_error: f64,
    tau_error: f64,
    label: &str,
) -> Result<(), String> {
    if probability_error > PROBABILITY_TOLERANCE || tau_error > TAU_TOLERANCE {
        Err(format!(
            "ADA-A5 E5 {label} dense tolerance exceeded: \
probability={probability_error:e}, tau={tau_error:e}"
        ))
    } else {
        Ok(())
    }
}

fn verify_dense_support(
    dense: &EntmaxDistribution,
    eager: &HierarchicalResult,
    lazy: &LazyHierarchicalResult,
) -> Result<usize, String> {
    let mut support_tokens = 0_usize;

    for (token, &probability) in dense.probabilities.iter().enumerate() {
        if probability <= SUPPORT_EPSILON {
            continue;
        }

        support_tokens += 1;

        if !eager.loaded_tokens[token] {
            return Err("ADA-A5 E5 eager candidate pruned dense support".to_owned());
        }

        if !lazy.loaded_tokens[token] {
            return Err("ADA-A5 E5 lazy candidate pruned dense support".to_owned());
        }
    }

    Ok(support_tokens)
}

fn verify_eager_lazy_identity(
    eager: &HierarchicalResult,
    lazy: &LazyHierarchicalResult,
) -> Result<(), String> {
    if eager.loaded_tokens != lazy.loaded_tokens {
        return Err("ADA-A5 E5 eager/lazy loaded-token sets differ".to_owned());
    }

    if !distributions_bitwise_equal(&eager.distribution, &lazy.distribution) {
        return Err("ADA-A5 E5 eager/lazy Entmax distributions are not bitwise equal".to_owned());
    }

    if eager.metrics.nodes_expanded != lazy.metrics.nodes_expanded {
        return Err("ADA-A5 E5 eager/lazy node-expansion counts differ".to_owned());
    }

    if eager.metrics.subtrees_pruned != lazy.metrics.subtrees_pruned {
        return Err("ADA-A5 E5 eager/lazy subtree-prune counts differ".to_owned());
    }

    if eager.metrics.leaves_loaded != lazy.metrics.leaves_loaded {
        return Err("ADA-A5 E5 eager/lazy loaded-leaf counts differ".to_owned());
    }

    if eager.metrics.tokens_loaded != lazy.metrics.tokens_loaded {
        return Err("ADA-A5 E5 eager/lazy loaded-token counts differ".to_owned());
    }

    if eager.metrics.tokens_pruned != lazy.metrics.tokens_pruned {
        return Err("ADA-A5 E5 eager/lazy pruned-token counts differ".to_owned());
    }

    if eager.metrics.threshold_solves != lazy.metrics.threshold_solves {
        return Err("ADA-A5 E5 eager/lazy threshold-solve counts differ".to_owned());
    }

    Ok(())
}

fn measure_case(case: &QueryKeyPagedCase) -> Result<CaseMetrics, String> {
    let leaf_size = PAGE_SIZE.div_ceil(LEAF_DIVISOR);

    let index = build_hierarchical_key_index(&case.keys, case.head_dim, PAGE_SIZE, leaf_size)
        .map_err(str::to_owned)?;

    let dense_scores = dense_qk_scores(case).map_err(str::to_owned)?;
    let dense = dense_entmax(&dense_scores, case.alpha).map_err(str::to_owned)?;

    let eager = branch_and_bound_entmax_hierarchical(case, &index).map_err(str::to_owned)?;

    let lazy = branch_and_bound_entmax_hierarchical_lazy(case, &index).map_err(str::to_owned)?;

    verify_eager_lazy_identity(&eager, &lazy)?;

    if eager.metrics.bound_evaluations != index.node_count() {
        return Err("ADA-A5 E5 historical eager controller did not evaluate every node".to_owned());
    }

    if lazy.metrics.nodes_total != index.node_count() {
        return Err("ADA-A5 E5 lazy nodes_total differs from index node count".to_owned());
    }

    if lazy.metrics.nodes_never_evaluated
        != lazy.metrics.nodes_total - lazy.metrics.bound_evaluations
    {
        return Err("ADA-A5 E5 lazy unevaluated-node accounting is inconsistent".to_owned());
    }

    let (eager_probability_error, eager_tau_error) =
        distribution_error(&dense, &eager.distribution);

    let (lazy_probability_error, lazy_tau_error) = distribution_error(&dense, &lazy.distribution);

    check_dense_tolerance(eager_probability_error, eager_tau_error, "eager")?;
    check_dense_tolerance(lazy_probability_error, lazy_tau_error, "lazy")?;

    let support_tokens = verify_dense_support(&dense, &eager, &lazy)?;

    let token_count = usize_as_f64(case.key_count());
    let nodes_total = usize_as_f64(index.node_count());

    Ok(CaseMetrics {
        support_fraction: usize_as_f64(support_tokens) / token_count,
        score_avoidance: 1.0 - usize_as_f64(lazy.metrics.tokens_loaded) / token_count,
        eager_bound_evaluation_fraction: usize_as_f64(eager.metrics.bound_evaluations)
            / nodes_total,
        lazy_bound_evaluation_fraction: lazy.metrics.bound_evaluation_fraction(),
        lazy_bound_avoidance: lazy.metrics.bound_avoidance(),
        nodes_total: index.node_count(),
        lazy_bound_evaluations: lazy.metrics.bound_evaluations,
        nodes_never_evaluated: lazy.metrics.nodes_never_evaluated,
        bound_cache_hits: lazy.metrics.bound_cache_hits,
        bound_evaluations_per_loaded_token: lazy.metrics.bound_evaluations_per_loaded_token(),
        bound_evaluations_per_pruned_token: lazy.metrics.bound_evaluations_per_pruned_token(),
        nodes_expanded: lazy.metrics.nodes_expanded,
        subtrees_pruned: lazy.metrics.subtrees_pruned,
        threshold_solves: lazy.metrics.threshold_solves,
        eager_probability_error,
        lazy_probability_error,
        eager_tau_error,
        lazy_tau_error,
    })
}

fn print_case(record_index: usize, record: &TraceRecord, alpha: f64, metrics: &CaseMetrics) {
    println!(
        "case,record_index={record_index},sample_fingerprint={:016x},\
layer={},query_head={},kv_head={},query_position={},\
key_start_position={},key_count={},head_dim={},alpha={alpha:.1},\
page_size={PAGE_SIZE},leaf_divisor={LEAF_DIVISOR},\
support_fraction={:.6},score_avoidance={:.6},\
eager_bound_evaluation_fraction={:.6},\
lazy_bound_evaluation_fraction={:.6},lazy_bound_avoidance={:.6},\
nodes_total={},lazy_bound_evaluations={},nodes_never_evaluated={},\
bound_cache_hits={},bound_evaluations_per_loaded_token={:.6},\
bound_evaluations_per_pruned_token={:.6},nodes_expanded={},\
subtrees_pruned={},threshold_solves={},\
eager_probability_difference={:.3e},lazy_probability_difference={:.3e},\
eager_tau_difference={:.3e},lazy_tau_difference={:.3e}",
        record.sample_fingerprint(),
        record.layer_index,
        record.query_head_index,
        record.kv_head_index,
        record.query_position,
        record.key_start_position,
        record.key_count,
        record.head_dim,
        metrics.support_fraction,
        metrics.score_avoidance,
        metrics.eager_bound_evaluation_fraction,
        metrics.lazy_bound_evaluation_fraction,
        metrics.lazy_bound_avoidance,
        metrics.nodes_total,
        metrics.lazy_bound_evaluations,
        metrics.nodes_never_evaluated,
        metrics.bound_cache_hits,
        metrics.bound_evaluations_per_loaded_token,
        metrics.bound_evaluations_per_pruned_token,
        metrics.nodes_expanded,
        metrics.subtrees_pruned,
        metrics.threshold_solves,
        metrics.eager_probability_error,
        metrics.lazy_probability_error,
        metrics.eager_tau_error,
        metrics.lazy_tau_error,
    );
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args_os().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: e5_lazy_focused_replay <trace.adaqk>",
        )
    })?;

    let corpus = read_trace_file(path)?;
    let metadata = corpus.metadata();

    println!("survey=ada_a5_e5_lazy_focused_replay");
    println!("synthetic_only=false");
    println!("wall_clock_benchmark=false");
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
    let mut by_query_head = BTreeMap::<(u32, u64), Aggregate>::new();
    let mut by_query_position = BTreeMap::<(u64, u64), Aggregate>::new();

    for (record_index, record) in corpus.records().iter().enumerate() {
        for alpha in ALPHAS {
            let case = record
                .to_query_key_case(PAGE_SIZE, alpha)
                .map_err(|error| io::Error::other(error.to_string()))?;

            let metrics = measure_case(&case).map_err(|message| {
                io::Error::other(format!(
                    "record_index={record_index}, layer={}, query_head={}, \
query_position={}, alpha={alpha:.1}: {message}",
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

            by_query_head
                .entry((record.query_head_index, alpha_bits))
                .or_default()
                .record(&metrics);

            by_query_position
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

    for ((query_head, alpha_bits), aggregate) in by_query_head {
        aggregate.print(
            "query_head",
            &query_head.to_string(),
            f64::from_bits(alpha_bits),
        );
    }

    for ((query_position, alpha_bits), aggregate) in by_query_position {
        aggregate.print(
            "query_position",
            &query_position.to_string(),
            f64::from_bits(alpha_bits),
        );
    }

    println!("survey_status=complete");

    Ok(())
}
