use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::{QueryKeyPagedCase, branch_and_bound_entmax_qk_box, dense_qk_scores};
use ada_a5_content_aware_bounds::{
    ContentAwareResult, branch_and_bound_entmax_content_aware, build_content_aware_key_index,
};
use ada_a5_hierarchical_bounds::{
    HierarchicalResult, branch_and_bound_entmax_hierarchical, build_hierarchical_key_index,
};

const SUPPORT_EPSILON: f64 = 1.0e-12;
const PROBABILITY_TOLERANCE: f64 = 2.0e-10;
const TAU_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy)]
struct SurveyShape {
    sequence_length: usize,
    head_dim: usize,
    page_size: usize,
}

const SHAPES: [SurveyShape; 7] = [
    SurveyShape {
        sequence_length: 128,
        head_dim: 32,
        page_size: 16,
    },
    SurveyShape {
        sequence_length: 128,
        head_dim: 64,
        page_size: 32,
    },
    SurveyShape {
        sequence_length: 512,
        head_dim: 64,
        page_size: 32,
    },
    SurveyShape {
        sequence_length: 512,
        head_dim: 128,
        page_size: 64,
    },
    SurveyShape {
        sequence_length: 1024,
        head_dim: 64,
        page_size: 64,
    },
    SurveyShape {
        sequence_length: 2048,
        head_dim: 128,
        page_size: 64,
    },
    SurveyShape {
        sequence_length: 2048,
        head_dim: 128,
        page_size: 128,
    },
];

const ALPHAS: [f64; 2] = [1.5, 2.0];
const LEAF_DIVISORS: [usize; 3] = [2, 4, 8];
const SEEDS: [u64; 3] = [
    0xA4E2_0000_0000_0001,
    0xA4E2_0000_0000_1001,
    0xA4E2_0000_0001_0001,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Regime {
    IidUniform,
    PageClustered,
    DominantPage,
}

const REGIMES: [Regime; 3] = [
    Regime::IidUniform,
    Regime::PageClustered,
    Regime::DominantPage,
];

impl Regime {
    const fn name(self) -> &'static str {
        match self {
            Self::IidUniform => "iid_uniform",
            Self::PageClustered => "page_clustered",
            Self::DominantPage => "dominant_page",
        }
    }
}

#[derive(Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.state >> 32).expect("upper 32 bits fit in u32")
    }

    fn centered(&mut self) -> f64 {
        let unit = f64::from(self.next_u32()) / f64::from(u32::MAX);
        unit * 2.0 - 1.0
    }
}

#[derive(Debug)]
struct CaseMetrics {
    support_tokens: usize,
    flat_scores: usize,
    ordered_tokens: usize,
    geometry_tokens: usize,
    flat_avoidance: f64,
    ordered_avoidance: f64,
    geometry_avoidance: f64,
    gain_over_flat: f64,
    gain_over_ordered: f64,
    tree_bound_density: f64,
    geometry_bound_density: f64,
    ball_win_fraction: f64,
    tree_nodes: usize,
    geometry_nodes: usize,
    tree_solves: usize,
    geometry_solves: usize,
    flat_prob_error: f64,
    tree_prob_error: f64,
    geometry_prob_error: f64,
    flat_tau_error: f64,
    tree_tau_error: f64,
    geometry_tau_error: f64,
}

#[derive(Debug)]
struct Aggregate {
    regime: Regime,
    alpha: f64,
    leaf_divisor: usize,
    cases: usize,
    flat_sum: f64,
    ordered_sum: f64,
    geometry_sum: f64,
    gain_over_flat_sum: f64,
    gain_over_ordered_sum: f64,
    tree_work_sum: f64,
    geometry_work_sum: f64,
    ball_win_sum: f64,
    tree_nodes_sum: usize,
    geometry_nodes_sum: usize,
    tree_solver_sum: usize,
    geometry_solver_sum: usize,
    flat_prob_max: f64,
    tree_prob_max: f64,
    geometry_prob_max: f64,
    flat_tau_max: f64,
    tree_tau_max: f64,
    geometry_tau_max: f64,
}

impl Aggregate {
    const fn new(regime: Regime, alpha: f64, leaf_divisor: usize) -> Self {
        Self {
            regime,
            alpha,
            leaf_divisor,
            cases: 0,
            flat_sum: 0.0,
            ordered_sum: 0.0,
            geometry_sum: 0.0,
            gain_over_flat_sum: 0.0,
            gain_over_ordered_sum: 0.0,
            tree_work_sum: 0.0,
            geometry_work_sum: 0.0,
            ball_win_sum: 0.0,
            tree_nodes_sum: 0,
            geometry_nodes_sum: 0,
            tree_solver_sum: 0,
            geometry_solver_sum: 0,
            flat_prob_max: 0.0,
            tree_prob_max: 0.0,
            geometry_prob_max: 0.0,
            flat_tau_max: 0.0,
            tree_tau_max: 0.0,
            geometry_tau_max: 0.0,
        }
    }

    fn record(&mut self, metrics: &CaseMetrics) {
        self.cases += 1;
        self.flat_sum += metrics.flat_avoidance;
        self.ordered_sum += metrics.ordered_avoidance;
        self.geometry_sum += metrics.geometry_avoidance;
        self.gain_over_flat_sum += metrics.gain_over_flat;
        self.gain_over_ordered_sum += metrics.gain_over_ordered;
        self.tree_work_sum += metrics.tree_bound_density;
        self.geometry_work_sum += metrics.geometry_bound_density;
        self.ball_win_sum += metrics.ball_win_fraction;
        self.tree_nodes_sum += metrics.tree_nodes;
        self.geometry_nodes_sum += metrics.geometry_nodes;
        self.tree_solver_sum += metrics.tree_solves;
        self.geometry_solver_sum += metrics.geometry_solves;
        self.flat_prob_max = self.flat_prob_max.max(metrics.flat_prob_error);
        self.tree_prob_max = self.tree_prob_max.max(metrics.tree_prob_error);
        self.geometry_prob_max = self.geometry_prob_max.max(metrics.geometry_prob_error);
        self.flat_tau_max = self.flat_tau_max.max(metrics.flat_tau_error);
        self.tree_tau_max = self.tree_tau_max.max(metrics.tree_tau_error);
        self.geometry_tau_max = self.geometry_tau_max.max(metrics.geometry_tau_error);
    }

    fn print(&self) {
        let denominator = usize_as_f64(self.cases);
        println!(
            "aggregate,regime={},alpha={:.1},leaf_divisor={},cases={},mean_flat_score_avoidance={:.6},mean_contiguous_score_avoidance={:.6},mean_content_score_avoidance={:.6},mean_content_gain_over_flat={:.6},mean_content_gain_over_contiguous={:.6},mean_contiguous_bound_evaluations_per_token={:.6},mean_content_bound_evaluations_per_token={:.6},mean_ball_bound_win_fraction={:.6},mean_contiguous_nodes_expanded={:.6},mean_content_nodes_expanded={:.6},mean_contiguous_threshold_solves={:.6},mean_content_threshold_solves={:.6},max_flat_probability_difference={:.3e},max_contiguous_probability_difference={:.3e},max_content_probability_difference={:.3e},max_flat_tau_difference={:.3e},max_contiguous_tau_difference={:.3e},max_content_tau_difference={:.3e}",
            self.regime.name(),
            self.alpha,
            self.leaf_divisor,
            self.cases,
            self.flat_sum / denominator,
            self.ordered_sum / denominator,
            self.geometry_sum / denominator,
            self.gain_over_flat_sum / denominator,
            self.gain_over_ordered_sum / denominator,
            self.tree_work_sum / denominator,
            self.geometry_work_sum / denominator,
            self.ball_win_sum / denominator,
            usize_as_f64(self.tree_nodes_sum) / denominator,
            usize_as_f64(self.geometry_nodes_sum) / denominator,
            usize_as_f64(self.tree_solver_sum) / denominator,
            usize_as_f64(self.geometry_solver_sum) / denominator,
            self.flat_prob_max,
            self.tree_prob_max,
            self.geometry_prob_max,
            self.flat_tau_max,
            self.tree_tau_max,
            self.geometry_tau_max,
        );
    }
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("survey dimensions fit in u32"))
}

fn generate_query(rng: &mut DeterministicRng, head_dim: usize) -> Vec<f64> {
    (0..head_dim).map(|_| rng.centered()).collect()
}

fn generate_iid_keys(
    rng: &mut DeterministicRng,
    sequence_length: usize,
    head_dim: usize,
) -> Vec<f64> {
    let element_count = sequence_length * head_dim;
    (0..element_count).map(|_| rng.centered()).collect()
}

fn generate_clustered_keys(rng: &mut DeterministicRng, shape: SurveyShape) -> Vec<f64> {
    let page_count = shape.sequence_length.div_ceil(shape.page_size);
    let mut centroids = Vec::with_capacity(page_count * shape.head_dim);
    for _ in 0..page_count * shape.head_dim {
        centroids.push(rng.centered());
    }

    let mut keys = Vec::with_capacity(shape.sequence_length * shape.head_dim);
    for token in 0..shape.sequence_length {
        let page = token / shape.page_size;
        let centroid_start = page * shape.head_dim;
        let centroid_end = centroid_start + shape.head_dim;
        for &centroid in &centroids[centroid_start..centroid_end] {
            keys.push(centroid + 0.08 * rng.centered());
        }
    }
    keys
}

fn generate_dominant_page_keys(
    rng: &mut DeterministicRng,
    query: &[f64],
    shape: SurveyShape,
) -> Vec<f64> {
    let mut keys = Vec::with_capacity(shape.sequence_length * shape.head_dim);
    for token in 0..shape.sequence_length {
        let page = token / shape.page_size;
        for &query_value in query {
            let key_value = if page == 0 {
                1.75 * query_value + 0.05 * rng.centered()
            } else {
                -0.45 * query_value + 0.12 * rng.centered()
            };
            keys.push(key_value);
        }
    }
    keys
}

fn generate_case(shape: SurveyShape, regime: Regime, alpha: f64, seed: u64) -> QueryKeyPagedCase {
    let mut rng = DeterministicRng::new(seed);
    let query = generate_query(&mut rng, shape.head_dim);
    let keys = match regime {
        Regime::IidUniform => generate_iid_keys(&mut rng, shape.sequence_length, shape.head_dim),
        Regime::PageClustered => generate_clustered_keys(&mut rng, shape),
        Regime::DominantPage => generate_dominant_page_keys(&mut rng, &query, shape),
    };
    let score_scale = usize_as_f64(shape.head_dim).sqrt().recip();
    QueryKeyPagedCase {
        query,
        keys,
        head_dim: shape.head_dim,
        page_size: shape.page_size,
        alpha,
        score_scale,
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

fn check_tolerance(probability_error: f64, tau_error: f64) -> Result<(), &'static str> {
    if probability_error > PROBABILITY_TOLERANCE || tau_error > TAU_TOLERANCE {
        return Err("ADA-A5 E3 dense/candidate tolerance exceeded");
    }
    Ok(())
}

fn verify_support(
    case: &QueryKeyPagedCase,
    dense: &EntmaxDistribution,
    flat_pages: &[bool],
    ordered: &HierarchicalResult,
    geometry: &ContentAwareResult,
) -> Result<usize, &'static str> {
    let mut support_tokens = 0_usize;
    for (token, &probability) in dense.probabilities.iter().enumerate() {
        if probability > SUPPORT_EPSILON {
            support_tokens += 1;
            let page = token / case.page_size;
            if !flat_pages[page] {
                return Err("ADA-A5 E3 flat candidate pruned a dense-support page");
            }
            if !ordered.loaded_tokens[token] {
                return Err("ADA-A5 E3 contiguous candidate pruned a dense-support token");
            }
            if !geometry.loaded_tokens[token] {
                return Err("ADA-A5 E3 content-aware candidate pruned a dense-support token");
            }
        }
    }
    Ok(support_tokens)
}

fn measure_case(
    case: &QueryKeyPagedCase,
    leaf_divisor: usize,
) -> Result<CaseMetrics, &'static str> {
    let dense_scores = dense_qk_scores(case)?;
    let dense = dense_entmax(&dense_scores, case.alpha)?;
    let flat = branch_and_bound_entmax_qk_box(case)?;
    let leaf_size = case.page_size.div_ceil(leaf_divisor);

    let ordered_index =
        build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)?;
    let ordered = branch_and_bound_entmax_hierarchical(case, &ordered_index)?;

    let geometry_index =
        build_content_aware_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)?;
    let geometry = branch_and_bound_entmax_content_aware(case, &geometry_index)?;

    let (flat_prob_error, flat_tau_error) = distribution_error(&dense, &flat.distribution);
    let (tree_prob_error, tree_tau_error) = distribution_error(&dense, &ordered.distribution);
    let (geometry_prob_error, geometry_tau_error) =
        distribution_error(&dense, &geometry.distribution);
    check_tolerance(flat_prob_error, flat_tau_error)?;
    check_tolerance(tree_prob_error, tree_tau_error)?;
    check_tolerance(geometry_prob_error, geometry_tau_error)?;

    let support_tokens = verify_support(case, &dense, &flat.loaded_pages, &ordered, &geometry)?;
    let total = usize_as_f64(case.key_count());
    let flat_scores = flat.metrics.scores_loaded;
    let ordered_tokens = ordered.metrics.tokens_loaded;
    let geometry_tokens = geometry.metrics.tokens_loaded;
    let flat_avoidance = 1.0 - usize_as_f64(flat_scores) / total;
    let ordered_avoidance = 1.0 - usize_as_f64(ordered_tokens) / total;
    let geometry_avoidance = 1.0 - usize_as_f64(geometry_tokens) / total;
    let tree_bound_density = usize_as_f64(ordered.metrics.bound_evaluations) / total;
    let geometry_bound_density = usize_as_f64(geometry.metrics.hybrid_bound_evaluations) / total;
    let ball_win_fraction = usize_as_f64(geometry.metrics.ball_bound_wins)
        / usize_as_f64(geometry.metrics.hybrid_bound_evaluations);

    Ok(CaseMetrics {
        support_tokens,
        flat_scores,
        ordered_tokens,
        geometry_tokens,
        flat_avoidance,
        ordered_avoidance,
        geometry_avoidance,
        gain_over_flat: geometry_avoidance - flat_avoidance,
        gain_over_ordered: geometry_avoidance - ordered_avoidance,
        tree_bound_density,
        geometry_bound_density,
        ball_win_fraction,
        tree_nodes: ordered.metrics.nodes_expanded,
        geometry_nodes: geometry.metrics.nodes_expanded,
        tree_solves: ordered.metrics.threshold_solves,
        geometry_solves: geometry.metrics.threshold_solves,
        flat_prob_error,
        tree_prob_error,
        geometry_prob_error,
        flat_tau_error,
        tree_tau_error,
        geometry_tau_error,
    })
}

#[derive(Debug, Clone, Copy)]
struct CaseDescriptor {
    shape: SurveyShape,
    regime: Regime,
    alpha: f64,
    seed: u64,
    leaf_divisor: usize,
}

fn print_case(descriptor: CaseDescriptor, metrics: &CaseMetrics) {
    let leaf_size = descriptor.shape.page_size.div_ceil(descriptor.leaf_divisor);
    println!(
        "case,regime={},seed={:#018x},n={},d={},page_size={},alpha={:.1},leaf_divisor={},leaf_size={},support_tokens={},flat_scores_loaded={},contiguous_tokens_loaded={},content_tokens_loaded={},flat_score_avoidance={:.6},contiguous_score_avoidance={:.6},content_score_avoidance={:.6},content_gain_over_flat={:.6},content_gain_over_contiguous={:.6},contiguous_bound_evaluations_per_token={:.6},content_bound_evaluations_per_token={:.6},ball_bound_win_fraction={:.6},contiguous_nodes_expanded={},content_nodes_expanded={},contiguous_threshold_solves={},content_threshold_solves={},max_flat_probability_difference={:.3e},max_contiguous_probability_difference={:.3e},max_content_probability_difference={:.3e},flat_tau_difference={:.3e},contiguous_tau_difference={:.3e},content_tau_difference={:.3e}",
        descriptor.regime.name(),
        descriptor.seed,
        descriptor.shape.sequence_length,
        descriptor.shape.head_dim,
        descriptor.shape.page_size,
        descriptor.alpha,
        descriptor.leaf_divisor,
        leaf_size,
        metrics.support_tokens,
        metrics.flat_scores,
        metrics.ordered_tokens,
        metrics.geometry_tokens,
        metrics.flat_avoidance,
        metrics.ordered_avoidance,
        metrics.geometry_avoidance,
        metrics.gain_over_flat,
        metrics.gain_over_ordered,
        metrics.tree_bound_density,
        metrics.geometry_bound_density,
        metrics.ball_win_fraction,
        metrics.tree_nodes,
        metrics.geometry_nodes,
        metrics.tree_solves,
        metrics.geometry_solves,
        metrics.flat_prob_error,
        metrics.tree_prob_error,
        metrics.geometry_prob_error,
        metrics.flat_tau_error,
        metrics.tree_tau_error,
        metrics.geometry_tau_error,
    );
}

fn build_aggregates() -> Vec<Aggregate> {
    let mut aggregates = Vec::with_capacity(REGIMES.len() * ALPHAS.len() * LEAF_DIVISORS.len());
    for regime in REGIMES {
        for alpha in ALPHAS {
            for leaf_divisor in LEAF_DIVISORS {
                aggregates.push(Aggregate::new(regime, alpha, leaf_divisor));
            }
        }
    }
    aggregates
}

fn aggregate_for(
    aggregates: &mut [Aggregate],
    regime: Regime,
    alpha: f64,
    leaf_divisor: usize,
) -> &mut Aggregate {
    aggregates
        .iter_mut()
        .find(|aggregate| {
            aggregate.regime == regime
                && aggregate.alpha.to_bits() == alpha.to_bits()
                && aggregate.leaf_divisor == leaf_divisor
        })
        .expect("aggregate exists for every E3 regime/alpha/divisor tuple")
}

fn main() -> Result<(), &'static str> {
    println!("survey=ada_a5_e3_three_way");
    println!("synthetic_only=true");
    println!("wall_clock_benchmark=false");
    println!("score_scale=1/sqrt(head_dim)");
    println!("support_epsilon={SUPPORT_EPSILON:.3e}");
    println!("probability_tolerance={PROBABILITY_TOLERANCE:.3e}");
    println!("tau_tolerance={TAU_TOLERANCE:.3e}");
    println!(
        "comparison_case_count={}",
        SHAPES.len() * REGIMES.len() * ALPHAS.len() * SEEDS.len() * LEAF_DIVISORS.len()
    );

    let mut aggregates = build_aggregates();
    for shape in SHAPES {
        for regime in REGIMES {
            for alpha in ALPHAS {
                for seed in SEEDS {
                    let case = generate_case(shape, regime, alpha, seed);
                    for leaf_divisor in LEAF_DIVISORS {
                        let metrics = measure_case(&case, leaf_divisor)?;
                        print_case(
                            CaseDescriptor {
                                shape,
                                regime,
                                alpha,
                                seed,
                                leaf_divisor,
                            },
                            &metrics,
                        );
                        aggregate_for(&mut aggregates, regime, alpha, leaf_divisor)
                            .record(&metrics);
                    }
                }
            }
        }
    }

    for aggregate in &aggregates {
        aggregate.print();
    }
    println!("survey_status=complete");
    Ok(())
}
