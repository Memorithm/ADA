use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use ada_a4_entmax_bnb::{BranchAndBoundResult, EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::{QueryKeyPagedCase, branch_and_bound_entmax_qk_box, dense_qk_scores};
use ada_a5_content_aware_bounds::{
    ContentAwareResult, branch_and_bound_entmax_content_aware, build_content_aware_key_index,
};
use ada_a5_hierarchical_bounds::{
    HierarchicalResult, branch_and_bound_entmax_hierarchical, build_hierarchical_key_index,
};
use ada_a5_real_qk_trace::{TraceRecord, read_trace_file};

const SUPPORT_EPSILON: f64 = 1.0e-12;
const PROBABILITY_TOLERANCE: f64 = 2.0e-10;
const TAU_TOLERANCE: f64 = 1.0e-10;
const ALPHAS: [f64; 2] = [1.5, 2.0];
const PAGE_SIZES: [usize; 4] = [16, 32, 64, 128];
const LEAF_DIVISORS: [usize; 3] = [2, 4, 8];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ConfigKey {
    alpha_bits: u64,
    page_size: usize,
    leaf_divisor: usize,
}

impl ConfigKey {
    const fn new(alpha: f64, page_size: usize, leaf_divisor: usize) -> Self {
        Self {
            alpha_bits: alpha.to_bits(),
            page_size,
            leaf_divisor,
        }
    }

    const fn alpha(self) -> f64 {
        f64::from_bits(self.alpha_bits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerConfigKey {
    layer_index: u32,
    config: ConfigKey,
}

#[derive(Debug)]
struct CaseMetrics {
    support_fraction: f64,
    flat_avoidance: f64,
    contiguous_avoidance: f64,
    content_avoidance: f64,
    content_gain_over_flat: f64,
    content_gain_over_contiguous: f64,
    contiguous_bound_density: f64,
    content_bound_density: f64,
    ball_win_fraction: f64,
    contiguous_nodes_expanded: usize,
    content_nodes_expanded: usize,
    contiguous_threshold_solves: usize,
    content_threshold_solves: usize,
    flat_probability_error: f64,
    contiguous_probability_error: f64,
    content_probability_error: f64,
    flat_tau_error: f64,
    contiguous_tau_error: f64,
    content_tau_error: f64,
}

#[derive(Debug, Default)]
struct Aggregate {
    cases: usize,
    support_fraction_sum: f64,
    flat_avoidance_sum: f64,
    contiguous_avoidance_sum: f64,
    content_avoidance_sum: f64,
    content_gain_over_flat_sum: f64,
    content_gain_over_contiguous_sum: f64,
    contiguous_bound_density_sum: f64,
    content_bound_density_sum: f64,
    ball_win_fraction_sum: f64,
    contiguous_nodes_expanded_sum: f64,
    content_nodes_expanded_sum: f64,
    contiguous_threshold_solves_sum: f64,
    content_threshold_solves_sum: f64,
    flat_probability_error_max: f64,
    contiguous_probability_error_max: f64,
    content_probability_error_max: f64,
    flat_tau_error_max: f64,
    contiguous_tau_error_max: f64,
    content_tau_error_max: f64,
}

impl Aggregate {
    fn record(&mut self, metrics: &CaseMetrics) {
        self.cases += 1;
        self.support_fraction_sum += metrics.support_fraction;
        self.flat_avoidance_sum += metrics.flat_avoidance;
        self.contiguous_avoidance_sum += metrics.contiguous_avoidance;
        self.content_avoidance_sum += metrics.content_avoidance;
        self.content_gain_over_flat_sum += metrics.content_gain_over_flat;
        self.content_gain_over_contiguous_sum += metrics.content_gain_over_contiguous;
        self.contiguous_bound_density_sum += metrics.contiguous_bound_density;
        self.content_bound_density_sum += metrics.content_bound_density;
        self.ball_win_fraction_sum += metrics.ball_win_fraction;
        self.contiguous_nodes_expanded_sum += usize_as_f64(metrics.contiguous_nodes_expanded);
        self.content_nodes_expanded_sum += usize_as_f64(metrics.content_nodes_expanded);
        self.contiguous_threshold_solves_sum += usize_as_f64(metrics.contiguous_threshold_solves);
        self.content_threshold_solves_sum += usize_as_f64(metrics.content_threshold_solves);
        self.flat_probability_error_max = self
            .flat_probability_error_max
            .max(metrics.flat_probability_error);
        self.contiguous_probability_error_max = self
            .contiguous_probability_error_max
            .max(metrics.contiguous_probability_error);
        self.content_probability_error_max = self
            .content_probability_error_max
            .max(metrics.content_probability_error);
        self.flat_tau_error_max = self.flat_tau_error_max.max(metrics.flat_tau_error);
        self.contiguous_tau_error_max = self
            .contiguous_tau_error_max
            .max(metrics.contiguous_tau_error);
        self.content_tau_error_max = self.content_tau_error_max.max(metrics.content_tau_error);
    }

    fn print(&self, scope: &str, layer_index: Option<u32>, config: ConfigKey) {
        let denominator = usize_as_f64(self.cases);
        let layer = layer_index.map_or_else(|| "all".to_owned(), |value| value.to_string());
        println!(
            "aggregate,scope={scope},layer={layer},alpha={:.1},page_size={},leaf_divisor={},cases={},mean_support_fraction={:.6},mean_flat_score_avoidance={:.6},mean_contiguous_score_avoidance={:.6},mean_content_score_avoidance={:.6},mean_content_gain_over_flat={:.6},mean_content_gain_over_contiguous={:.6},mean_contiguous_bound_evaluations_per_token={:.6},mean_content_bound_evaluations_per_token={:.6},mean_ball_bound_win_fraction={:.6},mean_contiguous_nodes_expanded={:.6},mean_content_nodes_expanded={:.6},mean_contiguous_threshold_solves={:.6},mean_content_threshold_solves={:.6},max_flat_probability_difference={:.3e},max_contiguous_probability_difference={:.3e},max_content_probability_difference={:.3e},max_flat_tau_difference={:.3e},max_contiguous_tau_difference={:.3e},max_content_tau_difference={:.3e}",
            config.alpha(),
            config.page_size,
            config.leaf_divisor,
            self.cases,
            self.support_fraction_sum / denominator,
            self.flat_avoidance_sum / denominator,
            self.contiguous_avoidance_sum / denominator,
            self.content_avoidance_sum / denominator,
            self.content_gain_over_flat_sum / denominator,
            self.content_gain_over_contiguous_sum / denominator,
            self.contiguous_bound_density_sum / denominator,
            self.content_bound_density_sum / denominator,
            self.ball_win_fraction_sum / denominator,
            self.contiguous_nodes_expanded_sum / denominator,
            self.content_nodes_expanded_sum / denominator,
            self.contiguous_threshold_solves_sum / denominator,
            self.content_threshold_solves_sum / denominator,
            self.flat_probability_error_max,
            self.contiguous_probability_error_max,
            self.content_probability_error_max,
            self.flat_tau_error_max,
            self.contiguous_tau_error_max,
            self.content_tau_error_max,
        );
    }
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("E4 replay dimensions fit in u32"))
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

fn check_tolerance(probability_error: f64, tau_error: f64) -> Result<(), String> {
    if probability_error > PROBABILITY_TOLERANCE || tau_error > TAU_TOLERANCE {
        Err(format!(
            "ADA-A5 E4 dense/candidate tolerance exceeded: probability={probability_error:e}, tau={tau_error:e}"
        ))
    } else {
        Ok(())
    }
}

fn verify_support(
    case: &QueryKeyPagedCase,
    dense: &EntmaxDistribution,
    flat: &BranchAndBoundResult,
    contiguous: &HierarchicalResult,
    content: &ContentAwareResult,
) -> Result<usize, String> {
    let mut support_tokens = 0_usize;
    for (token, &probability) in dense.probabilities.iter().enumerate() {
        if probability > SUPPORT_EPSILON {
            support_tokens += 1;
            let page = token / case.page_size;
            if !flat.loaded_pages[page] {
                return Err("ADA-A5 E4 flat candidate pruned dense support".to_owned());
            }
            if !contiguous.loaded_tokens[token] {
                return Err("ADA-A5 E4 contiguous candidate pruned dense support".to_owned());
            }
            if !content.loaded_tokens[token] {
                return Err("ADA-A5 E4 content-aware candidate pruned dense support".to_owned());
            }
        }
    }
    Ok(support_tokens)
}

fn measure_divisor(
    case: &QueryKeyPagedCase,
    dense: &EntmaxDistribution,
    flat: &BranchAndBoundResult,
    leaf_divisor: usize,
) -> Result<CaseMetrics, String> {
    let leaf_size = case.page_size.div_ceil(leaf_divisor);
    let contiguous_index = build_hierarchical_key_index(
        &case.keys,
        case.head_dim,
        case.page_size,
        leaf_size,
    )
    .map_err(str::to_owned)?;
    let contiguous = branch_and_bound_entmax_hierarchical(case, &contiguous_index)
        .map_err(str::to_owned)?;

    let content_index = build_content_aware_key_index(
        &case.keys,
        case.head_dim,
        case.page_size,
        leaf_size,
    )
    .map_err(str::to_owned)?;
    let content =
        branch_and_bound_entmax_content_aware(case, &content_index).map_err(str::to_owned)?;

    let (flat_probability_error, flat_tau_error) = distribution_error(dense, &flat.distribution);
    let (contiguous_probability_error, contiguous_tau_error) =
        distribution_error(dense, &contiguous.distribution);
    let (content_probability_error, content_tau_error) =
        distribution_error(dense, &content.distribution);

    check_tolerance(flat_probability_error, flat_tau_error)?;
    check_tolerance(contiguous_probability_error, contiguous_tau_error)?;
    check_tolerance(content_probability_error, content_tau_error)?;

    let support_tokens = verify_support(case, dense, flat, &contiguous, &content)?;
    let total = usize_as_f64(case.key_count());
    let support_fraction = usize_as_f64(support_tokens) / total;
    let flat_avoidance = 1.0 - usize_as_f64(flat.metrics.scores_loaded) / total;
    let contiguous_avoidance = 1.0 - usize_as_f64(contiguous.metrics.tokens_loaded) / total;
    let content_avoidance = 1.0 - usize_as_f64(content.metrics.tokens_loaded) / total;
    let contiguous_bound_density = usize_as_f64(contiguous.metrics.bound_evaluations) / total;
    let content_bound_density = usize_as_f64(content.metrics.hybrid_bound_evaluations) / total;
    let ball_win_fraction = usize_as_f64(content.metrics.ball_bound_wins)
        / usize_as_f64(content.metrics.hybrid_bound_evaluations);

    Ok(CaseMetrics {
        support_fraction,
        flat_avoidance,
        contiguous_avoidance,
        content_avoidance,
        content_gain_over_flat: content_avoidance - flat_avoidance,
        content_gain_over_contiguous: content_avoidance - contiguous_avoidance,
        contiguous_bound_density,
        content_bound_density,
        ball_win_fraction,
        contiguous_nodes_expanded: contiguous.metrics.nodes_expanded,
        content_nodes_expanded: content.metrics.nodes_expanded,
        contiguous_threshold_solves: contiguous.metrics.threshold_solves,
        content_threshold_solves: content.metrics.threshold_solves,
        flat_probability_error,
        contiguous_probability_error,
        content_probability_error,
        flat_tau_error,
        contiguous_tau_error,
        content_tau_error,
    })
}

fn print_case(
    record_index: usize,
    record: &TraceRecord,
    config: ConfigKey,
    metrics: &CaseMetrics,
) {
    println!(
        "case,record_index={record_index},sample_fingerprint={:016x},layer={},query_head={},kv_head={},query_position={},key_start_position={},key_count={},head_dim={},alpha={:.1},page_size={},leaf_divisor={},leaf_size={},support_fraction={:.6},flat_score_avoidance={:.6},contiguous_score_avoidance={:.6},content_score_avoidance={:.6},content_gain_over_flat={:.6},content_gain_over_contiguous={:.6},contiguous_bound_evaluations_per_token={:.6},content_bound_evaluations_per_token={:.6},ball_bound_win_fraction={:.6},contiguous_nodes_expanded={},content_nodes_expanded={},contiguous_threshold_solves={},content_threshold_solves={},flat_probability_difference={:.3e},contiguous_probability_difference={:.3e},content_probability_difference={:.3e},flat_tau_difference={:.3e},contiguous_tau_difference={:.3e},content_tau_difference={:.3e}",
        record.sample_fingerprint(),
        record.layer_index,
        record.query_head_index,
        record.kv_head_index,
        record.query_position,
        record.key_start_position,
        record.key_count,
        record.head_dim,
        config.alpha(),
        config.page_size,
        config.leaf_divisor,
        config.page_size.div_ceil(config.leaf_divisor),
        metrics.support_fraction,
        metrics.flat_avoidance,
        metrics.contiguous_avoidance,
        metrics.content_avoidance,
        metrics.content_gain_over_flat,
        metrics.content_gain_over_contiguous,
        metrics.contiguous_bound_density,
        metrics.content_bound_density,
        metrics.ball_win_fraction,
        metrics.contiguous_nodes_expanded,
        metrics.content_nodes_expanded,
        metrics.contiguous_threshold_solves,
        metrics.content_threshold_solves,
        metrics.flat_probability_error,
        metrics.contiguous_probability_error,
        metrics.content_probability_error,
        metrics.flat_tau_error,
        metrics.contiguous_tau_error,
        metrics.content_tau_error,
    );
}

fn main() -> Result<(), String> {
    let trace_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: e4_trace_replay <trace.adaqk>".to_owned())?;
    let corpus = read_trace_file(&trace_path).map_err(|error| error.to_string())?;
    let metadata = corpus.metadata();

    println!("survey=ada_a5_e4_real_qk_trace_replay");
    println!("synthetic_only=false");
    println!("wall_clock_benchmark=false");
    println!("model_id={:?}", metadata.model_id);
    println!("model_revision={:?}", metadata.model_revision);
    println!("tokenizer_id={:?}", metadata.tokenizer_id);
    println!("tokenizer_revision={:?}", metadata.tokenizer_revision);
    println!("capture_id={:?}", metadata.capture_id);
    println!("source_dtype={:?}", metadata.source_dtype);
    println!("tensor_stage={:?}", metadata.tensor_stage);
    println!("record_count={}", corpus.len());
    println!("support_epsilon={SUPPORT_EPSILON:.3e}");
    println!("probability_tolerance={PROBABILITY_TOLERANCE:.3e}");
    println!("tau_tolerance={TAU_TOLERANCE:.3e}");

    let comparison_count = corpus
        .len()
        .checked_mul(ALPHAS.len())
        .and_then(|value| value.checked_mul(PAGE_SIZES.len()))
        .and_then(|value| value.checked_mul(LEAF_DIVISORS.len()))
        .ok_or_else(|| "ADA-A5 E4 comparison count overflow".to_owned())?;
    println!("comparison_case_count={comparison_count}");

    let mut global_aggregates: BTreeMap<ConfigKey, Aggregate> = BTreeMap::new();
    let mut layer_aggregates: BTreeMap<LayerConfigKey, Aggregate> = BTreeMap::new();

    for (record_index, record) in corpus.records().iter().enumerate() {
        for alpha in ALPHAS {
            for page_size in PAGE_SIZES {
                let case = record
                    .to_query_key_case(page_size, alpha)
                    .map_err(|error| error.to_string())?;
                let dense_scores = dense_qk_scores(&case).map_err(str::to_owned)?;
                let dense = dense_entmax(&dense_scores, alpha).map_err(str::to_owned)?;
                let flat = branch_and_bound_entmax_qk_box(&case).map_err(str::to_owned)?;

                for leaf_divisor in LEAF_DIVISORS {
                    let config = ConfigKey::new(alpha, page_size, leaf_divisor);
                    let metrics = measure_divisor(&case, &dense, &flat, leaf_divisor)?;
                    print_case(record_index, record, config, &metrics);
                    global_aggregates.entry(config).or_default().record(&metrics);
                    layer_aggregates
                        .entry(LayerConfigKey {
                            layer_index: record.layer_index,
                            config,
                        })
                        .or_default()
                        .record(&metrics);
                }
            }
        }
    }

    for (config, aggregate) in &global_aggregates {
        aggregate.print("global", None, *config);
    }
    for (key, aggregate) in &layer_aggregates {
        aggregate.print("layer", Some(key.layer_index), key.config);
    }

    println!("survey_status=complete");
    Ok(())
}
