#![forbid(unsafe_code)]

use ada_core::AttentionCase;
use ada_oracle::{online_softmax_baseline, online_softmax_one_exp};
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 0xADA0_0000_0000_0001;
const ROUNDS: usize = 21;
const WARMUP_CALLS: usize = 20;

/// Runtime configuration for the evidence runner.
///
/// Every field defaults to the historical compile-time constant so the
/// default output stays byte-compatible with prior evidence. Overrides are
/// read once from the environment and fail closed on any malformed value.
struct RunnerConfig {
    seed: u64,
    rounds: usize,
    warmup_calls: usize,
    shapes: Vec<BenchShape>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            seed: SEED,
            rounds: ROUNDS,
            warmup_calls: WARMUP_CALLS,
            shapes: vec![
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
            ],
        }
    }
}

fn parse_env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn parse_usize_env(name: &str, minimum: usize) -> Result<Option<usize>, String> {
    parse_usize_value(name, parse_env_var(name).as_deref(), minimum)
}

fn parse_seed_env() -> Result<Option<u64>, String> {
    match parse_env_var("ADA_RUNNER_SEED") {
        None => Ok(None),
        Some(raw) => parse_seed_value(&raw),
    }
}

fn parse_shapes_env() -> Result<Option<Vec<BenchShape>>, String> {
    match parse_env_var("ADA_RUNNER_SHAPES") {
        None => Ok(None),
        Some(raw) => parse_shapes_value(&raw),
    }
}

fn parse_shapes_value(raw: &str) -> Result<Option<Vec<BenchShape>>, String> {
    let tokens: Vec<&str> = raw.split(',').filter(|t| !t.is_empty()).collect();
    if tokens.is_empty() {
        return Err(format!("ADA_RUNNER_SHAPES={raw} contains no shapes"));
    }
    tokens
        .iter()
        .map(|token| parse_shape(token))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn parse_usize_value(
    name: &str,
    raw: Option<&str>,
    minimum: usize,
) -> Result<Option<usize>, String> {
    match raw {
        None => Ok(None),
        Some(raw) => {
            let value = raw
                .parse::<usize>()
                .map_err(|_| format!("{name}={raw} is not a valid usize"))?;
            if value < minimum {
                return Err(format!("{name}={value} is below the minimum {minimum}"));
            }
            Ok(Some(value))
        }
    }
}

fn parse_seed_value(raw: &str) -> Result<Option<u64>, String> {
    let compact: String = raw.chars().filter(|&c| c != '_').collect();
    let parsed = compact
        .strip_prefix("0x")
        .or_else(|| compact.strip_prefix("0X"))
        .map_or_else(
            || compact.parse::<u64>(),
            |hex| u64::from_str_radix(hex, 16),
        );
    parsed
        .map(Some)
        .map_err(|_| format!("ADA_RUNNER_SEED={raw} is not a valid u64"))
}

fn parse_shape(token: &str) -> Result<BenchShape, String> {
    let parts: Vec<&str> = token.split(':').collect();
    let [seq, hd, iters] = parts.as_slice() else {
        return Err(format!("shape '{token}' must be seq:head_dim:iterations"));
    };
    let parse_field = |name: &str, raw: &str| {
        raw.parse::<usize>()
            .map_err(|_| format!("shape field {name} in '{token}' is not a valid usize"))
    };
    let shape = BenchShape {
        seq_len: parse_field("seq", seq)?,
        head_dim: parse_field("head_dim", hd)?,
        iterations: parse_field("iterations", iters)?,
    };
    if shape.seq_len == 0 || shape.head_dim == 0 || shape.iterations == 0 {
        return Err(format!("shape '{token}' fields must all be non-zero"));
    }
    Ok(shape)
}

fn load_runner_config() -> Result<RunnerConfig, Box<dyn Error>> {
    let mut config = RunnerConfig::default();

    if let Some(seed) = parse_seed_env()? {
        config.seed = seed;
    }
    if let Some(rounds) = parse_usize_env("ADA_RUNNER_ROUNDS", 1)? {
        config.rounds = rounds;
    }
    if let Some(warmup) = parse_usize_env("ADA_RUNNER_WARMUP_CALLS", 0)? {
        config.warmup_calls = warmup;
    }
    if let Some(shapes) = parse_shapes_env()? {
        config.shapes = shapes;
    }

    Ok(config)
}

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
        #[allow(clippy::cast_possible_truncation)]
        let mantissa_bits = ((self.next_u64() >> 41) & 0x007f_ffff) as u32;
        let unit = f32::from_bits(0x3f80_0000 | mantissa_bits) - 1.0;
        2.0 * unit - 1.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

fn bench_ns_per_iteration(
    case: &AttentionCase,
    iterations: usize,
    candidate: bool,
) -> Result<u128, &'static str> {
    let start = Instant::now();
    for _ in 0..iterations {
        let result = if candidate {
            online_softmax_one_exp(black_box(case))
        } else {
            online_softmax_baseline(black_box(case))
        }?;
        black_box(result);
    }
    let iterations_u128 =
        u128::try_from(iterations.max(1)).map_err(|_| "iteration count must fit in u128")?;
    Ok(start.elapsed().as_nanos() / iterations_u128)
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

fn warmup(case: &AttentionCase, warmup_calls: usize) -> Result<(), &'static str> {
    for _ in 0..warmup_calls {
        black_box(online_softmax_baseline(black_box(case))?);
        black_box(online_softmax_one_exp(black_box(case))?);
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = load_runner_config()?;
    let shapes: &[BenchShape] = &config.shapes;

    println!("ADA-A1 deterministic CPU evidence v1");
    println!(
        "seed={:#018x} rounds={} warmup_calls={}",
        config.seed, config.rounds, config.warmup_calls
    );
    println!(
        "seq_len,head_dim,iterations,max_abs_O,max_abs_LSE,baseline_exp,candidate_exp,baseline_median_ns,candidate_median_ns,baseline_p95_ns,candidate_p95_ns,baseline_mad_ns,candidate_mad_ns,speedup_ppm"
    );

    let mut shape_seed = config.seed;
    for shape in shapes {
        let case = make_case(shape.seq_len, shape.head_dim, shape_seed);
        shape_seed = shape_seed.wrapping_add(1);
        let baseline = online_softmax_baseline(&case)?;
        let candidate = online_softmax_one_exp(&case)?;
        let max_o = max_abs_diff(&baseline.output, &candidate.output);
        let max_lse = (baseline.lse - candidate.lse).abs();

        warmup(&case, config.warmup_calls)?;

        let mut baseline_samples = Vec::with_capacity(config.rounds);
        let mut candidate_samples = Vec::with_capacity(config.rounds);

        for round in 0..config.rounds {
            if round % 2 == 0 {
                baseline_samples.push(bench_ns_per_iteration(&case, shape.iterations, false)?);
                candidate_samples.push(bench_ns_per_iteration(&case, shape.iterations, true)?);
            } else {
                candidate_samples.push(bench_ns_per_iteration(&case, shape.iterations, true)?);
                baseline_samples.push(bench_ns_per_iteration(&case, shape.iterations, false)?);
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
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_values_fail_closed() {
        assert!(parse_usize_value("ROUNDS", Some("zero"), 1).is_err());
        assert!(parse_usize_value("ROUNDS", Some("0"), 1).is_err());
        assert!(parse_seed_value("not-a-number").is_err());
        assert!(parse_shape("128:64:512:oops").is_err());
        assert!(parse_shape("128:64").is_err());
        assert!(parse_shape("128:0:512").is_err());
        assert!(parse_shapes_value("128:64:512,,,").is_ok());
        assert!(parse_shapes_value(",").is_err());
    }

    #[test]
    fn well_formed_values_parse() {
        assert_eq!(parse_usize_value("ROUNDS", Some("7"), 1).unwrap(), Some(7));
        assert_eq!(parse_usize_value("WARMUP", Some("0"), 0).unwrap(), Some(0));
        assert_eq!(parse_seed_value("0xdead_beef").unwrap(), Some(0xdead_beef));
        assert_eq!(parse_seed_value("42").unwrap(), Some(42));

        let shapes = parse_shapes_value("128:64:8,256:32:4").unwrap().unwrap();
        assert_eq!(
            shapes,
            vec![
                BenchShape {
                    seq_len: 128,
                    head_dim: 64,
                    iterations: 8
                },
                BenchShape {
                    seq_len: 256,
                    head_dim: 32,
                    iterations: 4
                },
            ]
        );
    }

    #[test]
    fn unset_values_map_to_none_and_defaults_stay_historical() {
        assert_eq!(parse_usize_value("ROUNDS", None, 1).unwrap(), None);

        let config = RunnerConfig::default();
        assert_eq!(config.seed, SEED);
        assert_eq!(config.rounds, ROUNDS);
        assert_eq!(config.warmup_calls, WARMUP_CALLS);
        assert_eq!(config.shapes.len(), 4);
        assert_eq!(config.shapes[0].seq_len, 128);
        assert_eq!(config.shapes[3].iterations, 8);
    }
}
