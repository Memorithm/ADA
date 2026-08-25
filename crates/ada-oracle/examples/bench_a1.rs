//! Dependency-free benchmark harness for the A1 online-softmax family,
//! including the explicit NEON kernels on aarch64 hosts.
//!
//! Run with: `cargo run --release -p ada-oracle --example bench_a1 -- [seq:hd] ...`
//! Timing uses `std::time::Instant` and `std::hint::black_box` only; there is
//! deliberately no third-party benchmarking dependency.

#![forbid(unsafe_code)]

use ada_a1_neon::{online_softmax_baseline_neon, online_softmax_one_exp_neon};
use ada_core::AttentionCase;
use ada_oracle::{online_softmax_baseline, online_softmax_one_exp};
use std::hint::black_box;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn f32_signed(&mut self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let mantissa = ((self.next() >> 41) & 0x007f_ffff) as u32;
        let unit = f32::from_bits(0x3f80_0000 | mantissa) - 1.0;
        2.0 * unit - 1.0
    }
}

fn make_case(seq_len: usize, head_dim: usize) -> AttentionCase {
    let mut rng = Rng(0xADA0_BEEF);
    let logits = (0..seq_len).map(|_| 12.0 * rng.f32_signed()).collect();
    let values = (0..seq_len * head_dim).map(|_| rng.f32_signed()).collect();
    AttentionCase {
        logits,
        values,
        head_dim,
    }
}

fn median_ns(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn time<F: FnMut()>(mut body: F, iterations: usize, rounds: usize) -> u128 {
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(&mut body)();
        }
        #[allow(clippy::cast_precision_loss)]
        let per_iter = start.elapsed().as_nanos() / iterations.max(1) as u128;
        samples.push(per_iter);
    }
    median_ns(samples)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shapes: Vec<(usize, usize)> = if args.is_empty() {
        vec![(128, 64), (512, 64), (2048, 128), (4096, 128)]
    } else {
        args.iter()
            .filter_map(|arg| {
                let parts: Vec<&str> = arg.split(':').collect();
                match parts.as_slice() {
                    [seq, hd] => Some(((*seq).parse().ok()?, (*hd).parse().ok()?)),
                    _ => None,
                }
            })
            .collect()
    };

    println!("ADA-A1 dependency-free bench (median ns/iter, 11 rounds)");
    println!("seq_len,head_dim,iterations,baseline_ns,one_exp_ns,baseline_neon_ns,one_exp_neon_ns");

    for (seq_len, head_dim) in shapes {
        let case = make_case(seq_len, head_dim);
        let iterations = (4096 * 1024 / (seq_len * head_dim)).max(4);

        // Warmup.
        let _ = online_softmax_baseline(black_box(&case));
        let _ = online_softmax_one_exp(black_box(&case));
        let _ = online_softmax_baseline_neon(black_box(&case));
        let _ = online_softmax_one_exp_neon(black_box(&case));

        let baseline = time(
            || {
                black_box(online_softmax_baseline(black_box(&case)).ok());
            },
            iterations,
            11,
        );
        let one_exp = time(
            || {
                black_box(online_softmax_one_exp(black_box(&case)).ok());
            },
            iterations,
            11,
        );
        let baseline_neon = time(
            || {
                black_box(online_softmax_baseline_neon(black_box(&case)).ok());
            },
            iterations,
            11,
        );
        let one_exp_neon = time(
            || {
                black_box(online_softmax_one_exp_neon(black_box(&case)).ok());
            },
            iterations,
            11,
        );

        println!(
            "{seq_len},{head_dim},{iterations},{baseline},{one_exp},{baseline_neon},{one_exp_neon}"
        );
    }
}
