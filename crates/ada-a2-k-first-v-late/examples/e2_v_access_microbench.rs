use std::env;
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 0xADA2_E200_0000_0001;

const DEFAULT_ROUNDS: usize = 9;
const DEFAULT_TARGET_SCALARS: usize = 1_000_000;
const WARMUP_CALLS: usize = 8;

const VALUE_DIM: usize = 128;

const L2_BYTES: usize = 1024 * 1024;
const L3_BYTES: usize = 16 * 1024 * 1024;
const EVICTION_BYTES: usize = 32 * 1024 * 1024;

const TOKEN_COUNTS: [usize; 8] = [64, 128, 256, 512, 2048, 8192, 32768, 65536];

const DENSITY_PPM: [usize; 6] = [5_000, 10_000, 20_000, 50_000, 250_000, 1_000_000];

#[derive(Debug, Clone, Copy)]
enum SupportPattern {
    Prefix,
    Spread,
}

impl SupportPattern {
    const fn name(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Spread => "spread",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Kernel {
    Dense,
    Gather,
}

#[derive(Debug)]
struct BenchCase<'a> {
    probabilities: &'a [f32],
    values: &'a [f32],
    support_indices: &'a [usize],
    support_weights: &'a [f32],
    value_dim: usize,
}

#[derive(Debug, Clone)]
struct Samples {
    values: Vec<u128>,
}

impl Samples {
    fn median(&self) -> u128 {
        median(self.values.clone())
    }

    fn p95(&self) -> u128 {
        p95(self.values.clone())
    }

    fn mad(&self) -> u128 {
        let center = self.median();

        median(
            self.values
                .iter()
                .map(|sample| sample.abs_diff(center))
                .collect(),
        )
    }
}

#[derive(Debug)]
struct PairStats {
    dense: Samples,
    gather: Samples,
    dense_iterations: usize,
    gather_iterations: usize,
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
        let mantissa_bits = u32::try_from((self.next_u64() >> 41) & 0x007f_ffff)
            .expect("23 mantissa bits fit in u32");

        let unit = f32::from_bits(0x3f80_0000 | mantissa_bits) - 1.0;

        2.0 * unit - 1.0
    }
}

fn env_positive_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
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

fn make_values(token_count: usize, value_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);

    (0..token_count * value_dim)
        .map(|_| rng.f32_signed())
        .collect()
}

fn support_count(token_count: usize, density_ppm: usize) -> usize {
    token_count
        .checked_mul(density_ppm)
        .expect("benchmark dimensions fit usize")
        .div_ceil(1_000_000)
        .max(1)
        .min(token_count)
}

fn make_indices(token_count: usize, count: usize, pattern: SupportPattern) -> Vec<usize> {
    match pattern {
        SupportPattern::Prefix => (0..count).collect(),

        SupportPattern::Spread => {
            if count == 1 {
                vec![token_count / 2]
            } else {
                (0..count)
                    .map(|index| {
                        index
                            .checked_mul(token_count - 1)
                            .expect("benchmark index product fits usize")
                            / (count - 1)
                    })
                    .collect()
            }
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn make_probabilities(token_count: usize, indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let count_u32 = u32::try_from(indices.len()).expect("benchmark support count fits u32");

    let weight = 1.0_f32 / count_u32 as f32;

    let mut probabilities = vec![0.0_f32; token_count];

    for &index in indices {
        probabilities[index] = weight;
    }

    let support_weights = vec![weight; indices.len()];

    (probabilities, support_weights)
}

#[inline(never)]
fn dense_value_scan(case: &BenchCase<'_>, output: &mut [f32]) {
    output.fill(0.0);

    for (row_index, &probability) in case.probabilities.iter().enumerate() {
        let start = row_index * case.value_dim;

        let row = &case.values[start..start + case.value_dim];

        for (accumulator, &value) in output.iter_mut().zip(row.iter()) {
            *accumulator = probability.mul_add(value, *accumulator);
        }
    }
}

#[inline(never)]
fn support_value_gather(case: &BenchCase<'_>, output: &mut [f32]) {
    output.fill(0.0);

    for (&row_index, &probability) in case.support_indices.iter().zip(case.support_weights.iter()) {
        let start = row_index * case.value_dim;

        let row = &case.values[start..start + case.value_dim];

        for (accumulator, &value) in output.iter_mut().zip(row.iter()) {
            *accumulator = probability.mul_add(value, *accumulator);
        }
    }
}

fn run_kernel(kind: Kernel, case: &BenchCase<'_>, output: &mut [f32]) {
    match kind {
        Kernel::Dense => dense_value_scan(case, output),
        Kernel::Gather => support_value_gather(case, output),
    }

    black_box(&output[..]);
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(&left_value, &right_value)| (left_value - right_value).abs())
        .fold(0.0_f32, f32::max)
}

fn warm_iterations(scalars_per_call: usize, target_scalars: usize) -> usize {
    target_scalars
        .div_ceil(scalars_per_call.max(1))
        .clamp(1, 100_000)
}

fn timed_batch(kind: Kernel, case: &BenchCase<'_>, iterations: usize, output: &mut [f32]) -> u128 {
    let start = Instant::now();

    for _ in 0..iterations {
        run_kernel(kind, black_box(case), black_box(output));
    }

    let elapsed = start.elapsed().as_nanos();

    elapsed / u128::try_from(iterations).expect("iteration count fits u128")
}

fn bench_warm(case: &BenchCase<'_>, rounds: usize, target_scalars: usize) -> PairStats {
    let dense_scalars = case.probabilities.len() * case.value_dim;

    let gather_scalars = case.support_indices.len() * case.value_dim;

    let dense_iterations = warm_iterations(dense_scalars, target_scalars);

    let gather_iterations = warm_iterations(gather_scalars, target_scalars);

    let mut dense_output = vec![0.0_f32; case.value_dim];

    let mut gather_output = vec![0.0_f32; case.value_dim];

    for _ in 0..WARMUP_CALLS {
        run_kernel(Kernel::Dense, case, &mut dense_output);

        run_kernel(Kernel::Gather, case, &mut gather_output);
    }

    let mut dense_samples = Vec::with_capacity(rounds);

    let mut gather_samples = Vec::with_capacity(rounds);

    for round in 0..rounds {
        if round % 2 == 0 {
            dense_samples.push(timed_batch(
                Kernel::Dense,
                case,
                dense_iterations,
                &mut dense_output,
            ));

            gather_samples.push(timed_batch(
                Kernel::Gather,
                case,
                gather_iterations,
                &mut gather_output,
            ));
        } else {
            gather_samples.push(timed_batch(
                Kernel::Gather,
                case,
                gather_iterations,
                &mut gather_output,
            ));

            dense_samples.push(timed_batch(
                Kernel::Dense,
                case,
                dense_iterations,
                &mut dense_output,
            ));
        }
    }

    PairStats {
        dense: Samples {
            values: dense_samples,
        },
        gather: Samples {
            values: gather_samples,
        },
        dense_iterations,
        gather_iterations,
    }
}

fn evict_cache(buffer: &mut [u8]) {
    let mut checksum = 0_u8;

    for index in (0..buffer.len()).step_by(64) {
        let value = buffer[index].wrapping_add(1);
        buffer[index] = value;
        checksum ^= value;
    }

    black_box(checksum);
}

fn timed_evicted_once(
    kind: Kernel,
    case: &BenchCase<'_>,
    output: &mut [f32],
    eviction: &mut [u8],
) -> u128 {
    evict_cache(eviction);

    let start = Instant::now();

    run_kernel(kind, black_box(case), black_box(output));

    start.elapsed().as_nanos()
}

fn bench_evicted(case: &BenchCase<'_>, rounds: usize, eviction: &mut [u8]) -> PairStats {
    let mut dense_output = vec![0.0_f32; case.value_dim];

    let mut gather_output = vec![0.0_f32; case.value_dim];

    let mut dense_samples = Vec::with_capacity(rounds);

    let mut gather_samples = Vec::with_capacity(rounds);

    for round in 0..rounds {
        if round % 2 == 0 {
            dense_samples.push(timed_evicted_once(
                Kernel::Dense,
                case,
                &mut dense_output,
                eviction,
            ));

            gather_samples.push(timed_evicted_once(
                Kernel::Gather,
                case,
                &mut gather_output,
                eviction,
            ));
        } else {
            gather_samples.push(timed_evicted_once(
                Kernel::Gather,
                case,
                &mut gather_output,
                eviction,
            ));

            dense_samples.push(timed_evicted_once(
                Kernel::Dense,
                case,
                &mut dense_output,
                eviction,
            ));
        }
    }

    PairStats {
        dense: Samples {
            values: dense_samples,
        },
        gather: Samples {
            values: gather_samples,
        },
        dense_iterations: 1,
        gather_iterations: 1,
    }
}

fn cache_region(value_bytes: usize) -> &'static str {
    if value_bytes <= L2_BYTES {
        "l2_capacity"
    } else if value_bytes <= L3_BYTES {
        "l3_capacity"
    } else {
        "beyond_l3"
    }
}

fn validate_case(case: &BenchCase<'_>) -> f32 {
    let mut dense = vec![0.0_f32; case.value_dim];

    let mut gather = vec![0.0_f32; case.value_dim];

    dense_value_scan(case, &mut dense);
    support_value_gather(case, &mut gather);

    let difference = max_abs_diff(&dense, &gather);

    assert!(
        difference <= 2.0e-5,
        "dense/gather output mismatch: {difference:e}"
    );

    difference
}

// Benchmark serialization keeps the emitted record fields explicit.
#[allow(clippy::too_many_arguments)]
fn print_result(
    mode: &str,
    token_count: usize,
    value_dim: usize,
    requested_density_ppm: usize,
    support_count: usize,
    pattern: SupportPattern,
    stats: &PairStats,
    difference: f32,
) {
    let value_bytes = token_count
        .checked_mul(value_dim)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .expect("benchmark footprint fits usize");

    let realized_density_ppm = support_count
        .checked_mul(1_000_000)
        .expect("support density fits usize")
        / token_count;

    let dense_median = stats.dense.median();
    let gather_median = stats.gather.median();

    let speedup_ppm = dense_median.saturating_mul(1_000_000) / gather_median.max(1);

    println!(
        "result,mode={mode},tokens={token_count},\
value_dim={value_dim},value_bytes={value_bytes},\
cache_region={},requested_density_ppm={requested_density_ppm},\
realized_density_ppm={realized_density_ppm},\
support_count={support_count},pattern={},\
dense_iterations={},gather_iterations={},\
dense_median_ns={dense_median},\
gather_median_ns={gather_median},\
dense_p95_ns={},gather_p95_ns={},\
dense_mad_ns={},gather_mad_ns={},\
speedup_ppm={speedup_ppm},\
max_abs_difference={difference:.3e}",
        cache_region(value_bytes),
        pattern.name(),
        stats.dense_iterations,
        stats.gather_iterations,
        stats.dense.p95(),
        stats.gather.p95(),
        stats.dense.mad(),
        stats.gather.mad(),
    );
}

fn run_configuration(
    token_count: usize,
    density_ppm: usize,
    pattern: SupportPattern,
    values: &[f32],
    rounds: usize,
    target_scalars: usize,
    eviction: &mut [u8],
) {
    let count = support_count(token_count, density_ppm);

    let indices = make_indices(token_count, count, pattern);

    let (probabilities, support_weights) = make_probabilities(token_count, &indices);

    let case = BenchCase {
        probabilities: &probabilities,
        values,
        support_indices: &indices,
        support_weights: &support_weights,
        value_dim: VALUE_DIM,
    };

    let difference = validate_case(&case);

    let warm = bench_warm(&case, rounds, target_scalars);

    print_result(
        "warm",
        token_count,
        VALUE_DIM,
        density_ppm,
        count,
        pattern,
        &warm,
        difference,
    );

    if matches!(token_count, 512 | 8192 | 65536) {
        let evicted = bench_evicted(&case, rounds, eviction);

        print_result(
            "evicted",
            token_count,
            VALUE_DIM,
            density_ppm,
            count,
            pattern,
            &evicted,
            difference,
        );
    }
}

fn main() {
    let rounds = env_positive_usize("ADA_E2_ROUNDS", DEFAULT_ROUNDS);

    let target_scalars = env_positive_usize("ADA_E2_TARGET_SCALARS", DEFAULT_TARGET_SCALARS);

    println!("survey=ada_a2_e2_v_access_microbench");
    println!("physical_wall_clock=true");
    println!("pmu_counters=false");
    println!("value_dtype=f32");
    println!("value_dim={VALUE_DIM}");
    println!("rounds={rounds}");
    println!("warmup_calls={WARMUP_CALLS}");
    println!("target_scalars={target_scalars}");
    println!("l2_bytes={L2_BYTES}");
    println!("l3_shared_bytes={L3_BYTES}");
    println!("eviction_bytes={EVICTION_BYTES}");
    println!("seed={SEED:#018x}");

    println!(
        "NOTE: evicted means a 32 MiB cache-eviction buffer \
is touched before each timed kernel call; it is not a \
hardware cache-state proof."
    );

    println!(
        "NOTE: speedup_ppm is dense_median/gather_median * \
1e6. Values above 1000000 favor support gather."
    );

    let mut eviction = vec![0_u8; EVICTION_BYTES];

    for (token_index, token_count) in TOKEN_COUNTS.iter().copied().enumerate() {
        let token_index_u64 = u64::try_from(token_index).expect("token index fits u64");

        let values = make_values(token_count, VALUE_DIM, SEED.wrapping_add(token_index_u64));

        for density_ppm in DENSITY_PPM {
            for pattern in [SupportPattern::Prefix, SupportPattern::Spread] {
                if density_ppm == 1_000_000 && matches!(pattern, SupportPattern::Spread) {
                    continue;
                }

                run_configuration(
                    token_count,
                    density_ppm,
                    pattern,
                    &values,
                    rounds,
                    target_scalars,
                    &mut eviction,
                );
            }
        }
    }

    println!("survey_status=complete");
}
