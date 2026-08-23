#![forbid(unsafe_code)]

use ada_core::AttentionCase;
use ada_oracle::{online_softmax_baseline, online_softmax_one_exp};
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 0xADA0_0000_0000_0001;
const ROUNDS: usize = 21;
const WARMUP_CALLS: usize = 20;

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn f32_signed(&mut self) -> f32 {
        // Construct a uniform value in [0, 1) directly from the 23 explicit
        // mantissa bits of an IEEE-754 f32. This avoids integer-to-float casts
        // whose precision semantics would otherwise be ambiguous to Clippy.
        let mantissa_bits = u32::try_from((self.next_u64() >> 41) & 0x007f_ffff)
            .expect("23 mantissa bits always fit in u32");
        let unit = f32::from_bits(0x3f80_0000 | mantissa_bits) - 1.0;
        2.0 * unit - 1.0
    }
}

#[derive(Clone, Copy)]
struct BenchShape {
    seq_len: usize,
    head_dim: usize,
    iterations: usize,
}

fn make_case(seq_len: usize, head_dim: usize, seed: u64) -> AttentionCase {
    let mut rng = Rng::new(seed);
    let logits = (0..seq_len)
        .map(|_| 12.0 * rng.f32_signed())
        .collect::<Vec<_>>();
    let values = (0..seq_len * head_dim)
        .map(|_| rng.f32_signed())
        .collect::<Vec<_>>();
    AttentionCase {
        logits,
        values,
        head_dim,
    }
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn bench_ns_per_iteration(case: &AttentionCase, iterations: usize, candidate: bool) -> u128 {
    let start = Instant::now();
    for _ in 0..iterations {
        let result = if candidate {
            online_softmax_one_exp(black_box(case))
        } else {
            online_softmax_baseline(black_box(case))
        }
        .expect("validated deterministic case");
        black_box(result);
    }
    let iterations_u128 = u128::try_from(iterations).expect("iteration count fits in u128");
    start.elapsed().as_nanos() / iterations_u128
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn p95(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    let rank = samples.len().saturating_mul(95).div_ceil(100);
    let index = rank.saturating_sub(1).min(samples.len() - 1);
    samples[index]
}

fn mad(samples: &[u128], center: u128) -> u128 {
    median(
        samples
            .iter()
            .map(|sample| sample.abs_diff(center))
            .collect(),
    )
}

fn main() {
    let shapes = [
        BenchShape {
            seq_len: 128,
            head_dim: 64,
            iterations: 512,
        },
        BenchShape {
            seq_len: 512,
            head_dim: 64,
            iterations: 128,
        },
        BenchShape {
            seq_len: 2048,
            head_dim: 128,
            iterations: 16,
        },
        BenchShape {
            seq_len: 4096,
            head_dim: 128,
            iterations: 8,
        },
    ];

    println!("ADA-A1 deterministic CPU evidence v1");
    println!("seed={SEED:#018x} rounds={ROUNDS} warmup_calls={WARMUP_CALLS}");
    println!(
        "seq_len,head_dim,iterations,max_abs_O,max_abs_LSE,baseline_exp,candidate_exp,baseline_median_ns,candidate_median_ns,baseline_p95_ns,candidate_p95_ns,baseline_mad_ns,candidate_mad_ns,speedup_ppm"
    );

    for (index, shape) in shapes.iter().copied().enumerate() {
        let index_u64 = u64::try_from(index).expect("shape index fits in u64");
        let case = make_case(shape.seq_len, shape.head_dim, SEED.wrapping_add(index_u64));
        let baseline = online_softmax_baseline(&case).expect("generated case must validate");
        let candidate = online_softmax_one_exp(&case).expect("generated case must validate");
        let max_o = max_abs_diff(&baseline.output, &candidate.output);
        let max_lse = (baseline.lse - candidate.lse).abs();

        for _ in 0..WARMUP_CALLS {
            black_box(online_softmax_baseline(black_box(&case)).unwrap());
            black_box(online_softmax_one_exp(black_box(&case)).unwrap());
        }

        let mut baseline_samples = Vec::with_capacity(ROUNDS);
        let mut candidate_samples = Vec::with_capacity(ROUNDS);

        for round in 0..ROUNDS {
            if round % 2 == 0 {
                baseline_samples.push(bench_ns_per_iteration(&case, shape.iterations, false));
                candidate_samples.push(bench_ns_per_iteration(&case, shape.iterations, true));
            } else {
                candidate_samples.push(bench_ns_per_iteration(&case, shape.iterations, true));
                baseline_samples.push(bench_ns_per_iteration(&case, shape.iterations, false));
            }
        }

        let baseline_median = median(baseline_samples.clone());
        let candidate_median = median(candidate_samples.clone());
        let baseline_p95 = p95(baseline_samples.clone());
        let candidate_p95 = p95(candidate_samples.clone());
        let baseline_mad = mad(&baseline_samples, baseline_median);
        let candidate_mad = mad(&candidate_samples, candidate_median);
        let speedup_ppm = baseline_median.saturating_mul(1_000_000) / candidate_median.max(1);

        println!(
            "{},{},{},{max_o:.9e},{max_lse:.9e},{},{},{baseline_median},{candidate_median},{baseline_p95},{candidate_p95},{baseline_mad},{candidate_mad},{speedup_ppm}",
            shape.seq_len,
            shape.head_dim,
            shape.iterations,
            baseline.metrics.exp_evaluations,
            candidate.metrics.exp_evaluations,
        );
    }

    println!(
        "NOTE: CPU timings are local Thor evidence only; speedup_ppm is baseline_median/candidate_median * 1e6. No GPU/FLAT performance claim is implied."
    );
}
