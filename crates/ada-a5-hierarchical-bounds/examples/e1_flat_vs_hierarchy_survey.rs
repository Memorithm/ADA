#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::dense_entmax;
use ada_a4_qk_box::{QueryKeyPagedCase, branch_and_bound_entmax_qk_box, qk_box_entmax_case};
use ada_a5_hierarchical_bounds::{
    branch_and_bound_entmax_hierarchical, build_hierarchical_key_index,
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
const SEEDS: [u64; 3] = [
    0xA4E2_0000_0000_0001,
    0xA4E2_0000_0000_1001,
    0xA4E2_0000_0001_0001,
];
const LEAF_DIVISORS: [usize; 3] = [2, 4, 8];

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
struct ComparisonMetrics {
    support_tokens: usize,
    flat_scores_loaded: usize,
    hierarchical_tokens_loaded: usize,
    flat_score_avoidance: f64,
    hierarchical_score_avoidance: f64,
    additional_score_avoidance: f64,
    hierarchical_load_vs_flat: f64,
    hierarchy_bound_evaluations: usize,
    hierarchy_bound_evaluations_per_token: f64,
    hierarchy_nodes_expanded: usize,
    hierarchy_subtrees_pruned: usize,
    hierarchy_threshold_solves: usize,
    max_flat_probability_difference: f64,
    max_hierarchical_probability_difference: f64,
    flat_tau_difference: f64,
    hierarchical_tau_difference: f64,
}

#[derive(Debug)]
struct Aggregate {
    regime: Regime,
    alpha: f64,
    leaf_divisor: usize,
    cases: usize,
    flat_score_avoidance_sum: f64,
    hierarchical_score_avoidance_sum: f64,
    additional_score_avoidance_sum: f64,
    hierarchical_load_vs_flat_sum: f64,
    bound_evaluations_per_token_sum: f64,
    nodes_expanded_sum: usize,
    subtrees_pruned_sum: usize,
    threshold_solves_sum: usize,
    max_flat_probability_difference: f64,
    max_hierarchical_probability_difference: f64,
    max_flat_tau_difference: f64,
    max_hierarchical_tau_difference: f64,
}

impl Aggregate {
    const fn new(regime: Regime, alpha: f64, leaf_divisor: usize) -> Self {
        Self {
            regime,
            alpha,
            leaf_divisor,
            cases: 0,
            flat_score_avoidance_sum: 0.0,
            hierarchical_score_avoidance_sum: 0.0,
            additional_score_avoidance_sum: 0.0,
            hierarchical_load_vs_flat_sum: 0.0,
            bound_evaluations_per_token_sum: 0.0,
            nodes_expanded_sum: 0,
            subtrees_pruned_sum: 0,
            threshold_solves_sum: 0,
            max_flat_probability_difference: 0.0,
            max_hierarchical_probability_difference: 0.0,
            max_flat_tau_difference: 0.0,
            max_hierarchical_tau_difference: 0.0,
        }
    }

    fn record(&mut self, metrics: &ComparisonMetrics) {
        self.cases += 1;
        self.flat_score_avoidance_sum += metrics.flat_score_avoidance;
        self.hierarchical_score_avoidance_sum += metrics.hierarchical_score_avoidance;
        self.additional_score_avoidance_sum += metrics.additional_score_avoidance;
        self.hierarchical_load_vs_flat_sum += metrics.hierarchical_load_vs_flat;
        self.bound_evaluations_per_token_sum += metrics.hierarchy_bound_evaluations_per_token;
        self.nodes_expanded_sum += metrics.hierarchy_nodes_expanded;
        self.subtrees_pruned_sum += metrics.hierarchy_subtrees_pruned;
        self.threshold_solves_sum += metrics.hierarchy_threshold_solves;
        self.max_flat_probability_difference = self
            .max_flat_probability_difference
            .max(metrics.max_flat_probability_difference);
        self.max_hierarchical_probability_difference = self
            .max_hierarchical_probability_difference
            .max(metrics.max_hierarchical_probability_difference);
        self.max_flat_tau_difference = self
            .max_flat_tau_difference
            .max(metrics.flat_tau_difference);
        self.max_hierarchical_tau_difference = self
            .max_hierarchical_tau_difference
            .max(metrics.hierarchical_tau_difference);
    }

    fn print(&self) {
        let denominator = usize_as_f64(self.cases);
        println!(
            "aggregate,regime={},alpha={:.1},leaf_divisor={},cases={},mean_flat_score_avoidance={:.6},mean_hierarchical_score_avoidance={:.6},mean_additional_score_avoidance={:.6},mean_hierarchical_load_vs_flat={:.6},mean_bound_evaluations_per_token={:.6},mean_nodes_expanded={:.6},mean_subtrees_pruned={:.6},mean_threshold_solves={:.6},max_flat_probability_difference={:.3e},max_hierarchical_probability_difference={:.3e},max_flat_tau_difference={:.3e},max_hierarchical_tau_difference={:.3e}",
            self.regime.name(),
            self.alpha,
            self.leaf_divisor,
            self.cases,
            self.flat_score_avoidance_sum / denominator,
            self.hierarchical_score_avoidance_sum / denominator,
            self.additional_score_avoidance_sum / denominator,
            self.hierarchical_load_vs_flat_sum / denominator,
            self.bound_evaluations_per_token_sum / denominator,
            usize_as_f64(self.nodes_expanded_sum) / denominator,
            usize_as_f64(self.subtrees_pruned_sum) / denominator,
            usize_as_f64(self.threshold_solves_sum) / denominator,
            self.max_flat_probability_difference,
            self.max_hierarchical_probability_difference,
            self.max_flat_tau_difference,
            self.max_hierarchical_tau_difference,
        );
    }
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("A5-E1 survey dimensions fit in u32"))
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

fn max_probability_difference(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn validate_support(
    dense_probabilities: &[f64],
    case: &QueryKeyPagedCase,
    flat_loaded_pages: &[bool],
    hierarchical_loaded_tokens: &[bool],
) -> Result<usize, &'static str> {
    let mut support_tokens = 0_usize;
    for (token, &probability) in dense_probabilities.iter().enumerate() {
        if probability > SUPPORT_EPSILON {
            support_tokens += 1;
            let page = token / case.page_size;
            if !flat_loaded_pages[page] {
                return Err("ADA-A5 E1 flat candidate pruned a dense-support page");
            }
            if !hierarchical_loaded_tokens[token] {
                return Err("ADA-A5 E1 hierarchy pruned a dense-support token");
            }
        }
    }
    Ok(support_tokens)
}

fn measure_case(
    case: &QueryKeyPagedCase,
    leaf_divisor: usize,
) -> Result<ComparisonMetrics, &'static str> {
    let paged = qk_box_entmax_case(case)?;
    let dense = dense_entmax(&paged.scores, case.alpha)?;
    let flat = branch_and_bound_entmax_qk_box(case)?;
    let leaf_size = case.page_size.div_ceil(leaf_divisor);
    let index = build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)?;
    let hierarchical = branch_and_bound_entmax_hierarchical(case, &index)?;

    let max_flat_probability_difference =
        max_probability_difference(&dense.probabilities, &flat.distribution.probabilities);
    let max_hierarchical_probability_difference = max_probability_difference(
        &dense.probabilities,
        &hierarchical.distribution.probabilities,
    );
    let flat_tau_difference = (dense.tau - flat.distribution.tau).abs();
    let hierarchical_tau_difference = (dense.tau - hierarchical.distribution.tau).abs();

    if max_flat_probability_difference > PROBABILITY_TOLERANCE
        || max_hierarchical_probability_difference > PROBABILITY_TOLERANCE
        || flat_tau_difference > TAU_TOLERANCE
        || hierarchical_tau_difference > TAU_TOLERANCE
    {
        return Err("ADA-A5 E1 dense/candidate tolerance exceeded");
    }

    let support_tokens = validate_support(
        &dense.probabilities,
        case,
        &flat.loaded_pages,
        &hierarchical.loaded_tokens,
    )?;

    let score_total = case.key_count();
    let flat_scores_loaded = flat.metrics.scores_loaded;
    let hierarchical_tokens_loaded = hierarchical.metrics.tokens_loaded;
    let flat_score_avoidance = 1.0 - usize_as_f64(flat_scores_loaded) / usize_as_f64(score_total);
    let hierarchical_score_avoidance =
        1.0 - usize_as_f64(hierarchical_tokens_loaded) / usize_as_f64(score_total);
    let additional_score_avoidance = (usize_as_f64(flat_scores_loaded)
        - usize_as_f64(hierarchical_tokens_loaded))
        / usize_as_f64(score_total);
    let hierarchical_load_vs_flat =
        usize_as_f64(hierarchical_tokens_loaded) / usize_as_f64(flat_scores_loaded);
    let hierarchy_bound_evaluations_per_token =
        usize_as_f64(hierarchical.metrics.bound_evaluations) / usize_as_f64(score_total);

    Ok(ComparisonMetrics {
        support_tokens,
        flat_scores_loaded,
        hierarchical_tokens_loaded,
        flat_score_avoidance,
        hierarchical_score_avoidance,
        additional_score_avoidance,
        hierarchical_load_vs_flat,
        hierarchy_bound_evaluations: hierarchical.metrics.bound_evaluations,
        hierarchy_bound_evaluations_per_token,
        hierarchy_nodes_expanded: hierarchical.metrics.nodes_expanded,
        hierarchy_subtrees_pruned: hierarchical.metrics.subtrees_pruned,
        hierarchy_threshold_solves: hierarchical.metrics.threshold_solves,
        max_flat_probability_difference,
        max_hierarchical_probability_difference,
        flat_tau_difference,
        hierarchical_tau_difference,
    })
}

fn print_case(
    shape: SurveyShape,
    regime: Regime,
    alpha: f64,
    seed: u64,
    leaf_divisor: usize,
    metrics: &ComparisonMetrics,
) {
    let leaf_size = shape.page_size.div_ceil(leaf_divisor);
    println!(
        "case,regime={},seed={seed:#018x},n={},d={},page_size={},alpha={alpha:.1},leaf_divisor={},leaf_size={},support_tokens={},flat_scores_loaded={},hierarchical_tokens_loaded={},flat_score_avoidance={:.6},hierarchical_score_avoidance={:.6},additional_score_avoidance={:.6},hierarchical_load_vs_flat={:.6},hierarchy_bound_evaluations={},hierarchy_bound_evaluations_per_token={:.6},hierarchy_nodes_expanded={},hierarchy_subtrees_pruned={},hierarchy_threshold_solves={},max_flat_probability_difference={:.3e},max_hierarchical_probability_difference={:.3e},flat_tau_difference={:.3e},hierarchical_tau_difference={:.3e}",
        regime.name(),
        shape.sequence_length,
        shape.head_dim,
        shape.page_size,
        leaf_divisor,
        leaf_size,
        metrics.support_tokens,
        metrics.flat_scores_loaded,
        metrics.hierarchical_tokens_loaded,
        metrics.flat_score_avoidance,
        metrics.hierarchical_score_avoidance,
        metrics.additional_score_avoidance,
        metrics.hierarchical_load_vs_flat,
        metrics.hierarchy_bound_evaluations,
        metrics.hierarchy_bound_evaluations_per_token,
        metrics.hierarchy_nodes_expanded,
        metrics.hierarchy_subtrees_pruned,
        metrics.hierarchy_threshold_solves,
        metrics.max_flat_probability_difference,
        metrics.max_hierarchical_probability_difference,
        metrics.flat_tau_difference,
        metrics.hierarchical_tau_difference,
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
        .expect("aggregate exists for every A5-E1 regime/alpha/leaf pair")
}

fn main() -> Result<(), &'static str> {
    println!("survey=ada_a5_e1_flat_vs_hierarchy");
    println!("fixture_family=ada_a4_e2_exact_synthetic_family");
    println!("synthetic_only=true");
    println!("wall_clock_benchmark=false");
    println!("hierarchy_bound_evaluation=eager_all_nodes");
    println!("score_scale=1/sqrt(head_dim)");
    println!("support_epsilon={SUPPORT_EPSILON:.3e}");
    println!("probability_tolerance={PROBABILITY_TOLERANCE:.3e}");
    println!("tau_tolerance={TAU_TOLERANCE:.3e}");
    println!(
        "base_fixture_count={}",
        SHAPES.len() * REGIMES.len() * ALPHAS.len() * SEEDS.len()
    );
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
                        print_case(shape, regime, alpha, seed, leaf_divisor, &metrics);
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
