use std::env;
use std::hint::black_box;
use std::time::Instant;

const SEED: u64 = 0xADA2_E200_0000_0002;

const VALUE_DIM: usize = 128;
const DEFAULT_ROUNDS: usize = 7;
const DEFAULT_EVICTED_ROUNDS: usize = 15;
const DEFAULT_TARGET_SCALARS: usize = 1_000_000;
const WARMUP_CALLS: usize = 6;

const L2_BYTES: usize = 1_048_576;
const L3_BYTES: usize = 16_777_216;
const EVICTION_BYTES: usize = 33_554_432;

const TOKEN_COUNTS: [usize; 8] = [64, 128, 256, 512, 2048, 8192, 32768, 65536];

// Natural E1 global K-load anchors:
// alpha=2.0 -> 22.5564%
// alpha=1.5 -> 29.1461%
// 100% -> no-A5-pruning control.
const K_DENSITIES_PPM: [usize; 3] = [225_564, 291_461, 1_000_000];

// Natural E1 support anchors include:
// alpha=2.0 -> 0.8344%
// alpha=1.5 -> 1.5370%.
const SUPPORT_DENSITIES_PPM: [usize; 5] = [5_000, 8_344, 15_370, 20_000, 50_000];

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Prefix,
    Spread,
}

impl Pattern {
    const fn name(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Spread => "spread",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Full,
    KLoaded,
    Support,
}

impl Kernel {
    const fn index(self) -> usize {
        match self {
            Self::Full => 0,
            Self::KLoaded => 1,
            Self::Support => 2,
        }
    }
}

#[derive(Debug)]
struct Case<'a> {
    probabilities: &'a [f32],
    values: &'a [f32],
    k_indices: &'a [usize],
    k_weights: &'a [f32],
    support_indices: &'a [usize],
    support_weights: &'a [f32],
}

#[derive(Debug, Clone)]
struct Samples(Vec<u128>);

impl Samples {
    fn median(&self) -> u128 {
        median(self.0.clone())
    }

    fn p95(&self) -> u128 {
        let mut values = self.0.clone();
        values.sort_unstable();

        let rank = values.len().saturating_mul(95).div_ceil(100);
        values[rank.saturating_sub(1).min(values.len() - 1)]
    }

    fn mad(&self) -> u128 {
        let center = self.median();

        median(self.0.iter().map(|value| value.abs_diff(center)).collect())
    }
}

#[derive(Debug)]
struct Timings {
    samples: [Samples; 3],
    iterations: [usize; 3],
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
        let mantissa = u32::try_from((self.next_u64() >> 41) & 0x007f_ffff)
            .expect("23 mantissa bits fit in u32");

        let unit = f32::from_bits(0x3f80_0000 | mantissa) - 1.0;

        2.0 * unit - 1.0
    }
}

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn env_positive_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn count_from_density(token_count: usize, density_ppm: usize) -> usize {
    token_count
        .checked_mul(density_ppm)
        .expect("density product fits usize")
        .div_ceil(1_000_000)
        .clamp(1, token_count)
}

fn make_values(token_count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);

    (0..token_count * VALUE_DIM)
        .map(|_| rng.f32_signed())
        .collect()
}

fn spread_indices(total: usize, count: usize) -> Vec<usize> {
    if count == 1 {
        return vec![total / 2];
    }

    (0..count)
        .map(|index| {
            index
                .checked_mul(total - 1)
                .expect("index product fits usize")
                / (count - 1)
        })
        .collect()
}

fn make_k_indices(token_count: usize, count: usize, pattern: Pattern) -> Vec<usize> {
    match pattern {
        Pattern::Prefix => (0..count).collect(),
        Pattern::Spread => spread_indices(token_count, count),
    }
}

fn make_support_indices(k_indices: &[usize], count: usize, pattern: Pattern) -> Vec<usize> {
    match pattern {
        Pattern::Prefix => k_indices[..count].to_vec(),

        Pattern::Spread => {
            if count == 1 {
                return vec![k_indices[k_indices.len() / 2]];
            }

            (0..count)
                .map(|index| {
                    let rank = index
                        .checked_mul(k_indices.len() - 1)
                        .expect("support rank fits usize")
                        / (count - 1);

                    k_indices[rank]
                })
                .collect()
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn make_weights(
    token_count: usize,
    k_indices: &[usize],
    support_indices: &[usize],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let count = u32::try_from(support_indices.len()).expect("support count fits u32");

    let weight = 1.0_f32 / count as f32;

    let mut probabilities = vec![0.0_f32; token_count];

    for &index in support_indices {
        probabilities[index] = weight;
    }

    let k_weights = k_indices
        .iter()
        .map(|&index| probabilities[index])
        .collect();

    let support_weights = vec![weight; support_indices.len()];

    (probabilities, k_weights, support_weights)
}

#[inline(never)]
fn full_scan(case: &Case<'_>, output: &mut [f32]) {
    output.fill(0.0);

    for (row_index, &probability) in case.probabilities.iter().enumerate() {
        let start = row_index * VALUE_DIM;
        let row = &case.values[start..start + VALUE_DIM];

        for (accumulator, &value) in output.iter_mut().zip(row) {
            *accumulator = probability.mul_add(value, *accumulator);
        }
    }
}

#[inline(never)]
fn indexed_scan(indices: &[usize], weights: &[f32], values: &[f32], output: &mut [f32]) {
    output.fill(0.0);

    for (&row_index, &probability) in indices.iter().zip(weights) {
        let start = row_index * VALUE_DIM;
        let row = &values[start..start + VALUE_DIM];

        for (accumulator, &value) in output.iter_mut().zip(row) {
            *accumulator = probability.mul_add(value, *accumulator);
        }
    }
}

fn run_kernel(kernel: Kernel, case: &Case<'_>, output: &mut [f32]) {
    match kernel {
        Kernel::Full => full_scan(case, output),

        Kernel::KLoaded => indexed_scan(case.k_indices, case.k_weights, case.values, output),

        Kernel::Support => indexed_scan(
            case.support_indices,
            case.support_weights,
            case.values,
            output,
        ),
    }

    black_box(&output[..]);
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&left_value, &right_value)| (left_value - right_value).abs())
        .fold(0.0_f32, f32::max)
}

fn validate(case: &Case<'_>) -> (f32, f32) {
    let mut full = vec![0.0_f32; VALUE_DIM];
    let mut k_loaded = vec![0.0_f32; VALUE_DIM];
    let mut support = vec![0.0_f32; VALUE_DIM];

    run_kernel(Kernel::Full, case, &mut full);
    run_kernel(Kernel::KLoaded, case, &mut k_loaded);
    run_kernel(Kernel::Support, case, &mut support);

    let full_k = max_abs_diff(&full, &k_loaded);
    let k_support = max_abs_diff(&k_loaded, &support);

    assert!(full_k <= 2.0e-5);
    assert!(k_support <= 2.0e-5);

    (full_k, k_support)
}

fn iterations_for(scalar_count: usize, target_scalars: usize) -> usize {
    target_scalars
        .div_ceil(scalar_count.max(1))
        .clamp(1, 100_000)
}

fn timed_batch(kernel: Kernel, case: &Case<'_>, iterations: usize, output: &mut [f32]) -> u128 {
    let start = Instant::now();

    for _ in 0..iterations {
        run_kernel(kernel, black_box(case), black_box(output));
    }

    start.elapsed().as_nanos() / u128::try_from(iterations).expect("iteration count fits u128")
}

fn kernel_order(round: usize) -> [Kernel; 3] {
    const ORDERS: [[Kernel; 3]; 6] = [
        [Kernel::Full, Kernel::KLoaded, Kernel::Support],
        [Kernel::KLoaded, Kernel::Support, Kernel::Full],
        [Kernel::Support, Kernel::Full, Kernel::KLoaded],
        [Kernel::Full, Kernel::Support, Kernel::KLoaded],
        [Kernel::Support, Kernel::KLoaded, Kernel::Full],
        [Kernel::KLoaded, Kernel::Full, Kernel::Support],
    ];

    ORDERS[round % ORDERS.len()]
}

fn bench_warm(case: &Case<'_>, rounds: usize, target_scalars: usize) -> Timings {
    let iterations = [
        iterations_for(case.probabilities.len() * VALUE_DIM, target_scalars),
        iterations_for(case.k_indices.len() * VALUE_DIM, target_scalars),
        iterations_for(case.support_indices.len() * VALUE_DIM, target_scalars),
    ];

    let mut outputs = [
        vec![0.0_f32; VALUE_DIM],
        vec![0.0_f32; VALUE_DIM],
        vec![0.0_f32; VALUE_DIM],
    ];

    for _ in 0..WARMUP_CALLS {
        for kernel in [Kernel::Full, Kernel::KLoaded, Kernel::Support] {
            run_kernel(kernel, case, &mut outputs[kernel.index()]);
        }
    }

    let mut samples: [Vec<u128>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for round in 0..rounds {
        for kernel in kernel_order(round) {
            let index = kernel.index();

            samples[index].push(timed_batch(
                kernel,
                case,
                iterations[index],
                &mut outputs[index],
            ));
        }
    }

    Timings {
        samples: samples.map(Samples),
        iterations,
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

fn bench_evicted(case: &Case<'_>, rounds: usize, eviction: &mut [u8]) -> Timings {
    let mut outputs = [
        vec![0.0_f32; VALUE_DIM],
        vec![0.0_f32; VALUE_DIM],
        vec![0.0_f32; VALUE_DIM],
    ];

    let mut samples: [Vec<u128>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for round in 0..rounds {
        for kernel in kernel_order(round) {
            evict_cache(eviction);

            let index = kernel.index();
            let start = Instant::now();

            run_kernel(kernel, black_box(case), black_box(&mut outputs[index]));

            samples[index].push(start.elapsed().as_nanos());
        }
    }

    Timings {
        samples: samples.map(Samples),
        iterations: [1, 1, 1],
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

fn ppm(count: usize, total: usize) -> usize {
    count
        .checked_mul(1_000_000)
        .expect("ppm product fits usize")
        / total
}

#[allow(clippy::too_many_arguments)]
fn print_result(
    mode: &str,
    token_count: usize,
    k_density_ppm: usize,
    support_density_ppm: usize,
    pattern: Pattern,
    k_count: usize,
    support_count: usize,
    timings: &Timings,
    full_k_diff: f32,
    k_support_diff: f32,
) {
    let value_bytes = token_count
        .checked_mul(VALUE_DIM)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .expect("V footprint fits usize");

    let full = &timings.samples[Kernel::Full.index()];
    let k = &timings.samples[Kernel::KLoaded.index()];
    let support = &timings.samples[Kernel::Support.index()];

    let full_median = full.median();
    let k_median = k.median();
    let support_median = support.median();

    let full_to_k = full_median.saturating_mul(1_000_000) / k_median.max(1);

    let k_to_support = k_median.saturating_mul(1_000_000) / support_median.max(1);

    let full_to_support = full_median.saturating_mul(1_000_000) / support_median.max(1);

    println!(
        "result,mode={mode},tokens={token_count},\
value_dim={VALUE_DIM},value_bytes={value_bytes},\
cache_region={},pattern={},\
requested_k_density_ppm={k_density_ppm},\
realized_k_density_ppm={},k_count={k_count},\
requested_support_density_ppm={support_density_ppm},\
realized_support_density_ppm={},support_count={support_count},\
full_iterations={},k_iterations={},support_iterations={},\
full_median_ns={full_median},k_median_ns={k_median},\
support_median_ns={support_median},\
full_p95_ns={},k_p95_ns={},support_p95_ns={},\
full_mad_ns={},k_mad_ns={},support_mad_ns={},\
full_to_k_speedup_ppm={full_to_k},\
k_to_support_speedup_ppm={k_to_support},\
full_to_support_speedup_ppm={full_to_support},\
max_abs_full_k_difference={full_k_diff:.3e},\
max_abs_k_support_difference={k_support_diff:.3e}",
        cache_region(value_bytes),
        pattern.name(),
        ppm(k_count, token_count),
        ppm(support_count, token_count),
        timings.iterations[0],
        timings.iterations[1],
        timings.iterations[2],
        full.p95(),
        k.p95(),
        support.p95(),
        full.mad(),
        k.mad(),
        support.mad(),
    );
}

fn evicted_probe(token_count: usize, k_density_ppm: usize, support_density_ppm: usize) -> bool {
    let representative = matches!(token_count, 512 | 8192 | 65536);

    let alpha2_like = k_density_ppm == 225_564 && support_density_ppm == 8_344;

    let alpha15_like = k_density_ppm == 291_461 && support_density_ppm == 15_370;

    let stress = k_density_ppm == 291_461 && support_density_ppm == 50_000;

    representative && (alpha2_like || alpha15_like || stress)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    token_count: usize,
    k_density_ppm: usize,
    support_density_ppm: usize,
    pattern: Pattern,
    values: &[f32],
    rounds: usize,
    evicted_rounds: usize,
    target_scalars: usize,
    eviction: &mut [u8],
) {
    let k_count = count_from_density(token_count, k_density_ppm);

    let support_count = count_from_density(token_count, support_density_ppm).min(k_count);

    let k_indices = make_k_indices(token_count, k_count, pattern);

    let support_indices = make_support_indices(&k_indices, support_count, pattern);

    assert!(
        support_indices
            .iter()
            .all(|index| { k_indices.binary_search(index).is_ok() })
    );

    let (probabilities, k_weights, support_weights) =
        make_weights(token_count, &k_indices, &support_indices);

    let case = Case {
        probabilities: &probabilities,
        values,
        k_indices: &k_indices,
        k_weights: &k_weights,
        support_indices: &support_indices,
        support_weights: &support_weights,
    };

    let (full_k_diff, k_support_diff) = validate(&case);

    let warm = bench_warm(&case, rounds, target_scalars);

    print_result(
        "warm",
        token_count,
        k_density_ppm,
        support_density_ppm,
        pattern,
        k_count,
        support_count,
        &warm,
        full_k_diff,
        k_support_diff,
    );

    if evicted_probe(token_count, k_density_ppm, support_density_ppm) {
        let evicted = bench_evicted(&case, evicted_rounds, eviction);

        print_result(
            "evicted",
            token_count,
            k_density_ppm,
            support_density_ppm,
            pattern,
            k_count,
            support_count,
            &evicted,
            full_k_diff,
            k_support_diff,
        );
    }
}

fn main() {
    let rounds = env_positive_usize("ADA_E2_ROUNDS", DEFAULT_ROUNDS);

    let evicted_rounds =
        env_positive_usize("ADA_E2_EVICTED_ROUNDS", rounds.max(DEFAULT_EVICTED_ROUNDS));

    let target_scalars = env_positive_usize("ADA_E2_TARGET_SCALARS", DEFAULT_TARGET_SCALARS);

    println!("survey=ada_a2_e2_three_level_v_access");
    println!("physical_wall_clock=true");
    println!("pmu_counters=false");
    println!("comparison=full_dense_vs_k_loaded_vs_support");
    println!("value_dtype=f32");
    println!("value_dim={VALUE_DIM}");
    println!("rounds={rounds}");
    println!("evicted_rounds={evicted_rounds}");
    println!("warmup_calls={WARMUP_CALLS}");
    println!("target_scalars={target_scalars}");
    println!("l2_bytes={L2_BYTES}");
    println!("l3_shared_bytes={L3_BYTES}");
    println!("eviction_bytes={EVICTION_BYTES}");
    println!("seed={SEED:#018x}");
    println!("kernel_rotation=balanced_6_permutation");
    println!(
        "NOTE: full_to_k_speedup_ppm/1e6 = G_A5 isolates A5 K pruning; \
k_to_support_speedup_ppm/1e6 = G_A2_after_A5 isolates A2 V-late after A5; \
full_to_support_speedup_ppm/1e6 = G_total must not be attributed to A2 alone."
    );
    println!(
        "NOTE: evicted touches a dedicated eviction buffer before each \
individually timed kernel call, outside the timed interval; it is not a \
hardware cache-state proof."
    );
    println!(
        "NOTE: cache_region labels are descriptive V-footprint \
classifications, not measured cache residency."
    );

    let mut eviction = vec![0_u8; EVICTION_BYTES];

    for (shape_index, token_count) in TOKEN_COUNTS.iter().copied().enumerate() {
        let shape_index = u64::try_from(shape_index).expect("shape index fits u64");

        let values = make_values(token_count, SEED.wrapping_add(shape_index));

        for k_density_ppm in K_DENSITIES_PPM {
            let mut supports = SUPPORT_DENSITIES_PPM.to_vec();

            // S=K control.
            supports.push(k_density_ppm);

            for support_density_ppm in supports {
                for pattern in [Pattern::Prefix, Pattern::Spread] {
                    run_case(
                        token_count,
                        k_density_ppm,
                        support_density_ppm,
                        pattern,
                        &values,
                        rounds,
                        evicted_rounds,
                        target_scalars,
                        &mut eviction,
                    );
                }
            }
        }
    }

    println!("survey_status=complete");
}
