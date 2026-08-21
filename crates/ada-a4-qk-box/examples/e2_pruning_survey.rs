#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::dense_entmax;
use ada_a4_qk_box::{
    QueryKeyPagedCase, branch_and_bound_entmax_qk_box, qk_box_entmax_case,
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
    support_token_fraction: f64,
    support_page_fraction: f64,
    page_load_ratio: f64,
    score_avoidance: f64,
    slack_mean: f64,
    slack_p95: f64,
    slack_max: f64,
    max_probability_difference: f64,
    tau_difference: f64,
    pages_loaded: usize,
    pages_total: usize,
    scores_loaded: usize,
    support_tokens: usize,
    support_pages: usize,
}

#[derive(Debug)]
struct Aggregate {
    regime: Regime,
    alpha: f64,
    cases: usize,
    page_load_ratio_sum: f64,
    score_avoidance_sum: f64,
    support_token_fraction_sum: f64,
    support_page_fraction_sum: f64,
    slack_mean_sum: f64,
    slack_maximum: f64,
    max_probability_difference: f64,
    max_tau_difference: f64,
}

impl Aggregate {
    const fn new(regime: Regime, alpha: f64) -> Self {
        Self {
            regime,
            alpha,
            cases: 0,
            page_load_ratio_sum: 0.0,
            score_avoidance_sum: 0.0,
            support_token_fraction_sum: 0.0,
            support_page_fraction_sum: 0.0,
            slack_mean_sum: 0.0,
            slack_maximum: 0.0,
            max_probability_difference: 0.0,
            max_tau_difference: 0.0,
        }
    }

    fn record(&mut self, metrics: &CaseMetrics) {
        self.cases += 1;
        self.page_load_ratio_sum += metrics.page_load_ratio;
        self.score_avoidance_sum += metrics.score_avoidance;
        self.support_token_fraction_sum += metrics.support_token_fraction;
        self.support_page_fraction_sum += metrics.support_page_fraction;
        self.slack_mean_sum += metrics.slack_mean;
        self.slack_maximum = self.slack_maximum.max(metrics.slack_max);
        self.max_probability_difference = self
            .max_probability_difference
            .max(metrics.max_probability_difference);
        self.max_tau_difference = self.max_tau_difference.max(metrics.tau_difference);
    }

    fn print(&self) {
        let denominator = usize_as_f64(self.cases);
        println!(
            "aggregate,regime={},alpha={:.1},cases={},mean_page_load_ratio={:.6},mean_score_avoidance={:.6},mean_support_token_fraction={:.6},mean_support_page_fraction={:.6},mean_bound_slack={:.9},max_bound_slack={:.9},max_probability_difference={:.3e},max_tau_difference={:.3e}",
            self.regime.name(),
            self.alpha,
            self.cases,
            self.page_load_ratio_sum / denominator,
            self.score_avoidance_sum / denominator,
            self.support_token_fraction_sum / denominator,
            self.support_page_fraction_sum / denominator,
            self.slack_mean_sum / denominator,
            self.slack_maximum,
            self.max_probability_difference,
            self.max_tau_difference,
        );
    }
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("E2 survey dimensions fit in u32"))
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

fn generate_clustered_keys(
    rng: &mut DeterministicRng,
    shape: SurveyShape,
) -> Vec<f64> {
    let page_count = shape.sequence_length.div_ceil(shape.page_size);
    let mut centroids = Vec::with_capacity(page_count * shape.head_dim);
    for _ in 0..page_count * shape.head_dim {
        centroids.push(rng.centered());
    }

    let mut keys = Vec::with_capacity(shape.sequence_length * shape.head_dim);
    for token in 0..shape.sequence_length {
        let page = token / shape.page_size;
        let centroid_start = page * shape.head_dim;
        for lane in 0..shape.head_dim {
            let centroid = centroids[centroid_start + lane];
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

fn generate_case(
    shape: SurveyShape,
    regime: Regime,
    alpha: f64,
    seed: u64,
) -> QueryKeyPagedCase {
    let mut rng = DeterministicRng::new(seed);
    let query = generate_query(&mut rng, shape.head_dim);
    let keys = match regime {
        Regime::IidUniform => {
            generate_iid_keys(&mut rng, shape.sequence_length, shape.head_dim)
        }
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

fn slack_statistics(mut slacks: Vec<f64>) -> (f64, f64, f64) {
    let mean = slacks.iter().sum::<f64>() / usize_as_f64(slacks.len());
    slacks.sort_unstable_by(f64::total_cmp);
    let p95_rank = (slacks.len() * 95).div_ceil(100).saturating_sub(1);
    let p95 = slacks[p95_rank.min(slacks.len() - 1)];
    let maximum = slacks[slacks.len() - 1];
    (mean, p95, maximum)
}

fn measure_case(case: &QueryKeyPagedCase) -> Result<CaseMetrics, &'static str> {
    let paged = qk_box_entmax_case(case)?;
    let dense = dense_entmax(&paged.scores, case.alpha)?;
    let candidate = branch_and_bound_entmax_qk_box(case)?;

    let mut slacks = Vec::with_capacity(paged.page_upper_bounds.len());
    for (page, &bound) in paged.page_upper_bounds.iter().enumerate() {
        let start = page * case.page_size;
        let end = (start + case.page_size).min(paged.scores.len());
        let actual_maximum = paged.scores[start..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if bound < actual_maximum {
            return Err("ADA-A4 E2 observed a non-conservative page bound");
        }
        slacks.push(bound - actual_maximum);
    }
    let (slack_mean, slack_p95, slack_max) = slack_statistics(slacks);

    let max_probability_difference = dense
        .probabilities
        .iter()
        .zip(candidate.distribution.probabilities.iter())
        .map(|(&expected, &actual)| (expected - actual).abs())
        .fold(0.0_f64, f64::max);
    let tau_difference = (dense.tau - candidate.distribution.tau).abs();
    if max_probability_difference > PROBABILITY_TOLERANCE || tau_difference > TAU_TOLERANCE {
        return Err("ADA-A4 E2 dense/candidate tolerance exceeded");
    }

    let mut support_tokens = 0_usize;
    let mut support_page_mask = vec![false; case.page_count()];
    for (index, &probability) in dense.probabilities.iter().enumerate() {
        if probability > SUPPORT_EPSILON {
            support_tokens += 1;
            let page = index / case.page_size;
            support_page_mask[page] = true;
            if !candidate.loaded_pages[page] {
                return Err("ADA-A4 E2 pruned a dense-support page");
            }
        }
    }
    let support_pages = support_page_mask.iter().filter(|&&active| active).count();

    let page_total = candidate.metrics.pages_total;
    let score_total = case.key_count();
    let page_load_ratio = usize_as_f64(candidate.metrics.pages_loaded) / usize_as_f64(page_total);
    let score_avoidance =
        1.0 - usize_as_f64(candidate.metrics.scores_loaded) / usize_as_f64(score_total);
    let support_token_fraction = usize_as_f64(support_tokens) / usize_as_f64(score_total);
    let support_page_fraction = usize_as_f64(support_pages) / usize_as_f64(page_total);

    Ok(CaseMetrics {
        support_token_fraction,
        support_page_fraction,
        page_load_ratio,
        score_avoidance,
        slack_mean,
        slack_p95,
        slack_max,
        max_probability_difference,
        tau_difference,
        pages_loaded: candidate.metrics.pages_loaded,
        pages_total: page_total,
        scores_loaded: candidate.metrics.scores_loaded,
        support_tokens,
        support_pages,
    })
}

fn print_case(
    shape: SurveyShape,
    regime: Regime,
    alpha: f64,
    seed: u64,
    metrics: &CaseMetrics,
) {
    println!(
        "case,regime={},seed={seed:#018x},n={},d={},page_size={},alpha={alpha:.1},support_tokens={},support_pages={},pages_loaded={},pages_total={},page_load_ratio={:.6},scores_loaded={},score_avoidance={:.6},slack_mean={:.9},slack_p95={:.9},slack_max={:.9},max_probability_difference={:.3e},tau_difference={:.3e}",
        regime.name(),
        shape.sequence_length,
        shape.head_dim,
        shape.page_size,
        metrics.support_tokens,
        metrics.support_pages,
        metrics.pages_loaded,
        metrics.pages_total,
        metrics.page_load_ratio,
        metrics.scores_loaded,
        metrics.score_avoidance,
        metrics.slack_mean,
        metrics.slack_p95,
        metrics.slack_max,
        metrics.max_probability_difference,
        metrics.tau_difference,
    );
}

fn build_aggregates() -> Vec<Aggregate> {
    let mut aggregates = Vec::with_capacity(REGIMES.len() * ALPHAS.len());
    for regime in REGIMES {
        for alpha in ALPHAS {
            aggregates.push(Aggregate::new(regime, alpha));
        }
    }
    aggregates
}

fn aggregate_for(
    aggregates: &mut [Aggregate],
    regime: Regime,
    alpha: f64,
) -> &mut Aggregate {
    aggregates
        .iter_mut()
        .find(|aggregate| {
            aggregate.regime == regime && aggregate.alpha.to_bits() == alpha.to_bits()
        })
        .expect("aggregate exists for every E2 regime/alpha pair")
}

fn main() -> Result<(), &'static str> {
    println!("survey=ada_a4_e2_qk_box_pruning");
    println!("synthetic_only=true");
    println!("wall_clock_benchmark=false");
    println!("score_scale=1/sqrt(head_dim)");
    println!("support_epsilon={SUPPORT_EPSILON:.3e}");
    println!("probability_tolerance={PROBABILITY_TOLERANCE:.3e}");
    println!("tau_tolerance={TAU_TOLERANCE:.3e}");
    println!(
        "case_count={}",
        SHAPES.len() * REGIMES.len() * ALPHAS.len() * SEEDS.len()
    );

    let mut aggregates = build_aggregates();
    for shape in SHAPES {
        for regime in REGIMES {
            for alpha in ALPHAS {
                for seed in SEEDS {
                    let case = generate_case(shape, regime, alpha, seed);
                    let metrics = measure_case(&case)?;
                    print_case(shape, regime, alpha, seed, &metrics);
                    aggregate_for(&mut aggregates, regime, alpha).record(&metrics);
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
