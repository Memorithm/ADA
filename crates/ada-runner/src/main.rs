#![forbid(unsafe_code)]

use ada_core::AttentionCase;
use ada_oracle::{online_softmax_baseline, online_softmax_one_exp};
use std::hint::black_box;
use std::time::{Duration, Instant};

const SEED: u64 = 0xADA0_0000_0000_0001;

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

fn bench(case: &AttentionCase, iterations: usize, candidate: bool) -> Duration {
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
    start.elapsed()
}

fn main() {
    let shapes = [(128, 64), (512, 64), (2048, 128), (4096, 128)];
    let iterations = 200_usize;
    let iterations_u128 = u128::try_from(iterations).expect("iteration count fits in u128");

    println!("ADA-A1 deterministic CPU evidence");
    println!("seed={SEED:#018x} iterations={iterations}");
    println!(
        "seq_len,head_dim,max_abs_O,max_abs_LSE,baseline_exp,candidate_exp,baseline_ns,candidate_ns,speedup"
    );

    for (index, &(seq_len, head_dim)) in shapes.iter().enumerate() {
        let index_u64 = u64::try_from(index).expect("shape index fits in u64");
        let case = make_case(seq_len, head_dim, SEED.wrapping_add(index_u64));
        let baseline = online_softmax_baseline(&case).expect("generated case must validate");
        let candidate = online_softmax_one_exp(&case).expect("generated case must validate");
        let max_o = max_abs_diff(&baseline.output, &candidate.output);
        let max_lse = (baseline.lse - candidate.lse).abs();

        // Warm both code paths before collecting wall-clock evidence.
        for _ in 0..20 {
            black_box(online_softmax_baseline(black_box(&case)).unwrap());
            black_box(online_softmax_one_exp(black_box(&case)).unwrap());
        }

        let baseline_time = bench(&case, iterations, false);
        let candidate_time = bench(&case, iterations, true);
        let baseline_ns = baseline_time.as_nanos() / iterations_u128;
        let candidate_ns = candidate_time.as_nanos() / iterations_u128;
        let speedup = baseline_time.as_secs_f64()
            / candidate_time.as_secs_f64().max(f64::MIN_POSITIVE);

        println!(
            "{seq_len},{head_dim},{max_o:.9e},{max_lse:.9e},{},{},{baseline_ns},{candidate_ns},{speedup:.6}",
            baseline.metrics.exp_evaluations, candidate.metrics.exp_evaluations,
        );
    }

    println!(
        "NOTE: CPU timings are local evidence only; they are not GPU/FLAT performance claims."
    );
}
