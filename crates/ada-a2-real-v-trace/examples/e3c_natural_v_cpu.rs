use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
use std::io;
use std::time::Instant;

use ada_a2_real_v_trace::{ValueTraceRecord, read_value_trace_file};
use ada_a4_qk_box::QueryKeyPagedCase;
use ada_a5_hierarchical_bounds::{
    HierarchicalKeyIndex, branch_and_bound_entmax_hierarchical_priority_lazy,
    build_hierarchical_key_index,
};
use ada_a5_real_qk_trace::{TraceRecord, read_trace_file};

const HEAD_DIM: usize = 128;

const PAGE_SIZE: usize = 16;
const LEAF_SIZE: usize = 2;

const DEFAULT_ROUNDS: usize = 12;
const DEFAULT_CASE_LIMIT: usize = 24;
const DEFAULT_TARGET_SCALARS: usize = 262_144;
const DEFAULT_EVICTION_BYTES: usize = 8 * 1024 * 1024;

const WARMUP_CALLS: usize = 6;

const EXPECTED_QK_RECORDS: usize = 3072;
const EXPECTED_V_RECORDS: usize = 384;
const EXPECTED_GROUPS: usize = 1536;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    sample_id: String,
    layer_index: u32,
    kv_head_index: u32,
    query_position: u64,
    key_start_position: u64,
    head_dim: usize,
    key_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Full,
    KUnion,
    SupportUnion,
}

impl Kernel {
    const fn index(self) -> usize {
        match self {
            Self::Full => 0,
            Self::KUnion => 1,
            Self::SupportUnion => 2,
        }
    }
}

#[derive(Debug)]
struct NaturalCase {
    sample_fingerprint: u64,
    layer_index: u32,
    kv_head_index: u32,
    query_position: u64,
    key_count: usize,
    alpha: f64,

    values: Vec<f32>,

    full_weights: Vec<[f32; 2]>,

    k_indices: Vec<usize>,
    k_weights: Vec<[f32; 2]>,

    support_indices: Vec<usize>,
    support_weights: Vec<[f32; 2]>,
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

fn laboratory_error(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
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

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|value| match value.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),

            "0" | "false" | "FALSE" | "no" | "NO" => Some(false),

            _ => None,
        })
        .unwrap_or(default)
}

fn validate_rounds(rounds: usize) -> Result<(), io::Error> {
    if rounds < 6 {
        return Err(laboratory_error("E3c rounds must be at least 6"));
    }

    if rounds % 6 != 0 {
        return Err(laboratory_error("E3c rounds must be divisible by 6"));
    }

    Ok(())
}

fn kernel_order(round: usize) -> [Kernel; 3] {
    match round % 6 {
        0 => [Kernel::Full, Kernel::KUnion, Kernel::SupportUnion],

        1 => [Kernel::Full, Kernel::SupportUnion, Kernel::KUnion],

        2 => [Kernel::KUnion, Kernel::Full, Kernel::SupportUnion],

        3 => [Kernel::KUnion, Kernel::SupportUnion, Kernel::Full],

        4 => [Kernel::SupportUnion, Kernel::Full, Kernel::KUnion],

        _ => [Kernel::SupportUnion, Kernel::KUnion, Kernel::Full],
    }
}

fn selected_group_indices(total: usize, requested: usize) -> Vec<usize> {
    let count = requested.min(total);

    if count == total {
        return (0..total).collect();
    }

    if count == 1 {
        return vec![total / 2];
    }

    (0..count)
        .map(|index| {
            index
                .checked_mul(total - 1)
                .expect("E3c selection product fits usize")
                / (count - 1)
        })
        .collect()
}

fn validate_pair(
    first: &TraceRecord,
    second: &TraceRecord,
    key: &GroupKey,
) -> Result<(), io::Error> {
    let expected_q0 = key
        .kv_head_index
        .checked_mul(2)
        .ok_or_else(|| laboratory_error("Q-head index overflow"))?;

    let expected_q1 = expected_q0
        .checked_add(1)
        .ok_or_else(|| laboratory_error("Q-head index overflow"))?;

    if first.query_head_index != expected_q0 || second.query_head_index != expected_q1 {
        return Err(laboratory_error("natural GQA Q-head mapping mismatch"));
    }

    if first.keys != second.keys {
        return Err(laboratory_error(
            "natural GQA pair does not share identical K rows",
        ));
    }

    if first.score_scale.to_bits() != second.score_scale.to_bits() {
        return Err(laboratory_error(
            "natural GQA pair has different score scale",
        ));
    }

    Ok(())
}

fn priority_distribution(
    record: &TraceRecord,
    index: &HierarchicalKeyIndex,
    alpha: f64,
) -> Result<(Vec<f64>, Vec<bool>), io::Error> {
    let case: QueryKeyPagedCase = record
        .to_query_key_case(PAGE_SIZE, alpha)
        .map_err(|error| laboratory_error(error.to_string()))?;

    let result = branch_and_bound_entmax_hierarchical_priority_lazy(&case, index)
        .map_err(laboratory_error)?;

    if result.distribution.probabilities.len() != record.key_count
        || result.loaded_tokens.len() != record.key_count
    {
        return Err(laboratory_error("A5 result length mismatch"));
    }

    let support: Vec<bool> = result
        .distribution
        .probabilities
        .iter()
        .map(|&probability| probability > 0.0)
        .collect();

    if support
        .iter()
        .zip(&result.loaded_tokens)
        .any(|(&in_support, &loaded)| in_support && !loaded)
    {
        return Err(laboratory_error("support escapes A5 loaded set"));
    }

    Ok((result.distribution.probabilities, result.loaded_tokens))
}

fn exact_f32_values(record: &ValueTraceRecord, key_count: usize) -> Result<Vec<f32>, io::Error> {
    let values = record
        .prefix_values(key_count)
        .map_err(|error| laboratory_error(error.to_string()))?;

    let mut converted = Vec::with_capacity(values.len());

    for &value in values {
        #[allow(clippy::cast_possible_truncation)]
        let as_f32 = value as f32;

        if f64::from(as_f32).to_bits() != value.to_bits() {
            return Err(laboratory_error(
                "ADAV01 stored V value is not exactly representable as f32",
            ));
        }

        converted.push(as_f32);
    }

    Ok(converted)
}

fn build_natural_case(
    first: &TraceRecord,
    second: &TraceRecord,
    value_record: &ValueTraceRecord,
    index: &HierarchicalKeyIndex,
    alpha: f64,
) -> Result<NaturalCase, io::Error> {
    let (probabilities0, loaded0) = priority_distribution(first, index, alpha)?;

    let (probabilities1, loaded1) = priority_distribution(second, index, alpha)?;

    if probabilities0.len() != probabilities1.len() || probabilities0.len() != first.key_count {
        return Err(laboratory_error("GQA probability lengths differ"));
    }

    if value_record.head_dim != HEAD_DIM || first.head_dim != HEAD_DIM {
        return Err(laboratory_error("E3c requires head_dim=128"));
    }

    if value_record.value_start_position != first.key_start_position {
        return Err(laboratory_error("Q/K and V starts differ"));
    }

    if value_record.value_count < first.key_count {
        return Err(laboratory_error("V trace does not cover visible prefix"));
    }

    let support0: Vec<bool> = probabilities0
        .iter()
        .map(|&probability| probability > 0.0)
        .collect();

    let support1: Vec<bool> = probabilities1
        .iter()
        .map(|&probability| probability > 0.0)
        .collect();

    let mut full_weights = Vec::with_capacity(first.key_count);

    let mut k_indices = Vec::new();

    let mut k_weights = Vec::new();

    let mut support_indices = Vec::new();

    let mut support_weights = Vec::new();

    for index_value in 0..first.key_count {
        #[allow(clippy::cast_possible_truncation)]
        let weight = [
            probabilities0[index_value] as f32,
            probabilities1[index_value] as f32,
        ];

        full_weights.push(weight);

        let in_k = loaded0[index_value] || loaded1[index_value];

        let in_support = support0[index_value] || support1[index_value];

        if in_support && !in_k {
            return Err(laboratory_error("support union escapes K union"));
        }

        if in_k {
            k_indices.push(index_value);

            k_weights.push(weight);
        }

        if in_support {
            support_indices.push(index_value);

            support_weights.push(weight);
        }
    }

    if k_indices.is_empty() || support_indices.is_empty() {
        return Err(laboratory_error(
            "E3c natural case has empty K/support union",
        ));
    }

    let values = exact_f32_values(value_record, first.key_count)?;

    Ok(NaturalCase {
        sample_fingerprint: first.sample_fingerprint(),
        layer_index: first.layer_index,
        kv_head_index: first.kv_head_index,
        query_position: first.query_position,
        key_count: first.key_count,
        alpha,
        values,
        full_weights,
        k_indices,
        k_weights,
        support_indices,
        support_weights,
    })
}

#[inline]
fn accumulate_row(row: &[f32], weights: [f32; 2], output: &mut [f32]) {
    let (output0, output1) = output.split_at_mut(HEAD_DIM);

    for (dimension, &value) in row.iter().enumerate() {
        output0[dimension] = weights[0].mul_add(value, output0[dimension]);

        output1[dimension] = weights[1].mul_add(value, output1[dimension]);
    }
}

#[inline(never)]
fn full_scan(case: &NaturalCase, output: &mut [f32]) {
    output.fill(0.0);

    for (row, &weights) in case.values.chunks_exact(HEAD_DIM).zip(&case.full_weights) {
        accumulate_row(row, weights, output);
    }

    black_box(&output[..]);
}

#[inline(never)]
fn indexed_scan(indices: &[usize], weights: &[[f32; 2]], values: &[f32], output: &mut [f32]) {
    output.fill(0.0);

    for (&row_index, &weight) in indices.iter().zip(weights) {
        let start = row_index
            .checked_mul(HEAD_DIM)
            .expect("validated row offset fits usize");

        let row = &values[start..start + HEAD_DIM];

        accumulate_row(row, weight, output);
    }

    black_box(&output[..]);
}

fn run_kernel(kernel: Kernel, case: &NaturalCase, support_equals_k: bool, output: &mut [f32]) {
    match kernel {
        Kernel::Full => {
            full_scan(case, output);
        }

        Kernel::KUnion => {
            indexed_scan(&case.k_indices, &case.k_weights, &case.values, output);
        }

        Kernel::SupportUnion => {
            if support_equals_k {
                indexed_scan(&case.k_indices, &case.k_weights, &case.values, output);
            } else {
                indexed_scan(
                    &case.support_indices,
                    &case.support_weights,
                    &case.values,
                    output,
                );
            }
        }
    }
}

fn kernel_rows(kernel: Kernel, case: &NaturalCase, support_equals_k: bool) -> usize {
    match kernel {
        Kernel::Full => case.key_count,

        Kernel::KUnion => case.k_indices.len(),

        Kernel::SupportUnion => {
            if support_equals_k {
                case.k_indices.len()
            } else {
                case.support_indices.len()
            }
        }
    }
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&left_value, &right_value)| (left_value - right_value).abs())
        .fold(0.0_f32, f32::max)
}

fn validate_case(case: &NaturalCase, support_equals_k: bool) -> Result<(f32, f32), io::Error> {
    let mut full = vec![0.0_f32; HEAD_DIM * 2];

    let mut k = vec![0.0_f32; HEAD_DIM * 2];

    let mut support = vec![0.0_f32; HEAD_DIM * 2];

    run_kernel(Kernel::Full, case, support_equals_k, &mut full);

    run_kernel(Kernel::KUnion, case, support_equals_k, &mut k);

    run_kernel(Kernel::SupportUnion, case, support_equals_k, &mut support);

    let full_k = max_abs_difference(&full, &k);

    let k_support = max_abs_difference(&k, &support);

    if full_k != 0.0 || k_support != 0.0 {
        return Err(laboratory_error(format!(
            "E3c f32 output mismatch: full_k={full_k:e}, k_support={k_support:e}"
        )));
    }

    Ok((full_k, k_support))
}

fn iterations_for(rows: usize, target_scalars: usize) -> usize {
    let scalars = rows
        .checked_mul(HEAD_DIM)
        .and_then(|value| value.checked_mul(2))
        .expect("E3c timed scalar count fits usize");

    target_scalars.div_ceil(scalars.max(1)).clamp(1, 100_000)
}

fn timed_batch(
    kernel: Kernel,
    case: &NaturalCase,
    support_equals_k: bool,
    iterations: usize,
    output: &mut [f32],
) -> u128 {
    let start = Instant::now();

    for _ in 0..iterations {
        run_kernel(kernel, black_box(case), support_equals_k, black_box(output));
    }

    start.elapsed().as_nanos() / u128::try_from(iterations).expect("iteration count fits u128")
}

fn bench_warm(
    case: &NaturalCase,
    support_equals_k: bool,
    rounds: usize,
    target_scalars: usize,
) -> Timings {
    let iterations = [
        iterations_for(
            kernel_rows(Kernel::Full, case, support_equals_k),
            target_scalars,
        ),
        iterations_for(
            kernel_rows(Kernel::KUnion, case, support_equals_k),
            target_scalars,
        ),
        iterations_for(
            kernel_rows(Kernel::SupportUnion, case, support_equals_k),
            target_scalars,
        ),
    ];

    let mut outputs = [
        vec![0.0_f32; HEAD_DIM * 2],
        vec![0.0_f32; HEAD_DIM * 2],
        vec![0.0_f32; HEAD_DIM * 2],
    ];

    for _ in 0..WARMUP_CALLS {
        for kernel in [Kernel::Full, Kernel::KUnion, Kernel::SupportUnion] {
            run_kernel(kernel, case, support_equals_k, &mut outputs[kernel.index()]);
        }
    }

    let mut samples: [Vec<u128>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for round in 0..rounds {
        for kernel in kernel_order(round) {
            let index = kernel.index();

            samples[index].push(timed_batch(
                kernel,
                case,
                support_equals_k,
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

fn evict_l2(buffer: &mut [u8]) {
    let mut checksum = 0_u8;

    for index in (0..buffer.len()).step_by(64) {
        let value = buffer[index].wrapping_add(1);

        buffer[index] = value;

        checksum ^= value;
    }

    black_box(checksum);
}

fn bench_evicted(
    case: &NaturalCase,
    support_equals_k: bool,
    rounds: usize,
    eviction: &mut [u8],
) -> Timings {
    let mut outputs = [
        vec![0.0_f32; HEAD_DIM * 2],
        vec![0.0_f32; HEAD_DIM * 2],
        vec![0.0_f32; HEAD_DIM * 2],
    ];

    let mut samples: [Vec<u128>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for round in 0..rounds {
        for kernel in kernel_order(round) {
            evict_l2(eviction);

            let index = kernel.index();

            let start = Instant::now();

            run_kernel(
                kernel,
                black_box(case),
                support_equals_k,
                black_box(&mut outputs[index]),
            );

            samples[index].push(start.elapsed().as_nanos());
        }
    }

    Timings {
        samples: samples.map(Samples),
        iterations: [1, 1, 1],
    }
}

fn ratio_ppm(numerator: u128, denominator: u128) -> u128 {
    numerator.saturating_mul(1_000_000) / denominator.max(1)
}

#[allow(clippy::too_many_arguments)]
fn print_result(
    mode: &str,
    kind: &str,
    case: &NaturalCase,
    support_equals_k: bool,
    timings: &Timings,
    full_k_diff: f32,
    k_support_diff: f32,
) {
    let full = &timings.samples[Kernel::Full.index()];

    let k = &timings.samples[Kernel::KUnion.index()];

    let support = &timings.samples[Kernel::SupportUnion.index()];

    let full_median = full.median();

    let k_median = k.median();

    let support_median = support.median();

    let full_rows = kernel_rows(Kernel::Full, case, support_equals_k);

    let k_rows = kernel_rows(Kernel::KUnion, case, support_equals_k);

    let support_rows = kernel_rows(Kernel::SupportUnion, case, support_equals_k);

    let value_bytes = case
        .key_count
        .checked_mul(HEAD_DIM)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .expect("E3c V footprint fits usize");

    println!(
        "result,mode={mode},kind={kind},sample_fingerprint={:016x},layer={},kv_head={},query_position={},alpha={:.1},key_count={},value_bytes_f32={value_bytes},full_rows={full_rows},k_union_rows={k_rows},support_union_rows={support_rows},full_iterations={},k_iterations={},support_iterations={},full_median_ns={full_median},k_median_ns={k_median},support_median_ns={support_median},full_p95_ns={},k_p95_ns={},support_p95_ns={},full_mad_ns={},k_mad_ns={},support_mad_ns={},full_to_k_speedup_ppm={},k_to_support_speedup_ppm={},full_to_support_speedup_ppm={},max_abs_full_k_difference={full_k_diff:.9e},max_abs_k_support_difference={k_support_diff:.9e}",
        case.sample_fingerprint,
        case.layer_index,
        case.kv_head_index,
        case.query_position,
        case.alpha,
        case.key_count,
        timings.iterations[0],
        timings.iterations[1],
        timings.iterations[2],
        full.p95(),
        k.p95(),
        support.p95(),
        full.mad(),
        k.mad(),
        support.mad(),
        ratio_ppm(full_median, k_median,),
        ratio_ppm(k_median, support_median,),
        ratio_ppm(full_median, support_median,),
    );
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();

    let _program = args.next();

    let qk_path = args
        .next()
        .ok_or_else(|| laboratory_error("usage: e3c_natural_v_cpu <trace.adaqk> <trace.adav>"))?;

    let v_path = args
        .next()
        .ok_or_else(|| laboratory_error("usage: e3c_natural_v_cpu <trace.adaqk> <trace.adav>"))?;

    if args.next().is_some() {
        return Err(laboratory_error("unexpected extra command-line argument").into());
    }

    let rounds = env_positive_usize("ADA_E3C_ROUNDS", DEFAULT_ROUNDS);

    validate_rounds(rounds)?;

    let case_limit = env_positive_usize("ADA_E3C_CASE_LIMIT", DEFAULT_CASE_LIMIT);

    let target_scalars = env_positive_usize("ADA_E3C_TARGET_SCALARS", DEFAULT_TARGET_SCALARS);

    let eviction_bytes = env_positive_usize("ADA_E3C_EVICTION_BYTES", DEFAULT_EVICTION_BYTES);

    let run_evicted = env_bool("ADA_E3C_RUN_EVICTED", true);

    if eviction_bytes <= 1_048_576 {
        return Err(
            laboratory_error("E3c eviction buffer must exceed the 1 MiB private L2").into(),
        );
    }

    let qk = read_trace_file(qk_path)?;

    let values = read_value_trace_file(v_path)?;

    if qk.len() != EXPECTED_QK_RECORDS || values.len() != EXPECTED_V_RECORDS {
        return Err(laboratory_error("frozen E3c trace cardinality mismatch").into());
    }

    let mut value_index: BTreeMap<(String, u32, u32), usize> = BTreeMap::new();

    for (index, record) in values.records().iter().enumerate() {
        let identity = (
            record.sample_id.clone(),
            record.layer_index,
            record.kv_head_index,
        );

        if value_index.insert(identity, index).is_some() {
            return Err(laboratory_error("duplicate ADAV01 identity").into());
        }
    }

    let mut groups: BTreeMap<GroupKey, Vec<usize>> = BTreeMap::new();

    for (index, record) in qk.records().iter().enumerate() {
        let key = GroupKey {
            sample_id: record.sample_id.clone(),
            layer_index: record.layer_index,
            kv_head_index: record.kv_head_index,
            query_position: record.query_position,
            key_start_position: record.key_start_position,
            head_dim: record.head_dim,
            key_count: record.key_count,
        };

        groups.entry(key).or_default().push(index);
    }

    if groups.len() != EXPECTED_GROUPS {
        return Err(laboratory_error("frozen E3c GQA group count mismatch").into());
    }

    let selected = selected_group_indices(groups.len(), case_limit);

    let group_entries: Vec<_> = groups.iter().collect();

    let mut eviction = vec![0_u8; eviction_bytes];

    println!("survey=ada_a2_e3c_natural_v_cpu");

    println!("physical_wall_clock=true");

    println!("pmu_counters=false");

    println!("dram_bytes_measured=false");

    println!("cache_target=private_l2_plus_evicted_refill");

    println!("value_dtype=f32");

    println!("value_dim={HEAD_DIM}");

    println!("q_heads_per_kv=2");

    println!("comparison=full_dense_v_vs_k_union_v_vs_support_union_v");

    println!("timed_scope=v_weighted_sum_only");

    println!("a5_search_inside_timing=false");

    println!("distribution_construction_inside_timing=false");

    println!("order_schedule=all_6_kernel_permutations");

    println!("rounds={rounds}");

    println!("rounds_divisible_by_6={}", rounds % 6 == 0);

    println!("warmup_calls={WARMUP_CALLS}");

    println!("target_scalars={target_scalars}");

    println!("eviction_bytes={eviction_bytes}");

    println!("run_evicted={run_evicted}");

    println!("available_group_count={}", groups.len());

    println!("selected_group_count={}", selected.len());

    println!("selection=deterministic_evenly_spaced_over_sorted_natural_groups");

    let alphas = [1.5_f64, 2.0_f64];

    let mut natural_result_count = 0_usize;

    let mut control_result_count = 0_usize;

    for group_index in selected {
        let (key, members) = group_entries[group_index];

        if members.len() != 2 {
            return Err(
                laboratory_error("natural GQA group does not contain exactly two Q heads").into(),
            );
        }

        let mut ordered = members.clone();

        ordered.sort_by_key(|&index| qk.records()[index].query_head_index);

        let first = &qk.records()[ordered[0]];

        let second = &qk.records()[ordered[1]];

        validate_pair(first, second, key)?;

        let value_identity = (key.sample_id.clone(), key.layer_index, key.kv_head_index);

        let value_record = &values.records()[*value_index
            .get(&value_identity)
            .ok_or_else(|| laboratory_error("missing natural ADAV01 record"))?];

        let index = build_hierarchical_key_index(&first.keys, first.head_dim, PAGE_SIZE, LEAF_SIZE)
            .map_err(laboratory_error)?;

        for &alpha in &alphas {
            let case = build_natural_case(first, second, value_record, &index, alpha)?;

            let (full_k_diff, k_support_diff) = validate_case(&case, false)?;

            let warm = bench_warm(&case, false, rounds, target_scalars);

            print_result(
                "warm",
                "natural",
                &case,
                false,
                &warm,
                full_k_diff,
                k_support_diff,
            );

            natural_result_count += 1;

            let (control_full_k_diff, control_k_support_diff) = validate_case(&case, true)?;

            let control_warm = bench_warm(&case, true, rounds, target_scalars);

            print_result(
                "warm",
                "support_equals_k_control",
                &case,
                true,
                &control_warm,
                control_full_k_diff,
                control_k_support_diff,
            );

            control_result_count += 1;

            if run_evicted {
                let evicted = bench_evicted(&case, false, rounds, &mut eviction);

                print_result(
                    "evicted",
                    "natural",
                    &case,
                    false,
                    &evicted,
                    full_k_diff,
                    k_support_diff,
                );

                natural_result_count += 1;

                let control_evicted = bench_evicted(&case, true, rounds, &mut eviction);

                print_result(
                    "evicted",
                    "support_equals_k_control",
                    &case,
                    true,
                    &control_evicted,
                    control_full_k_diff,
                    control_k_support_diff,
                );

                control_result_count += 1;
            }
        }
    }

    println!("natural_result_count={natural_result_count}");

    println!("control_result_count={control_result_count}");

    println!("survey_status=complete");

    Ok(())
}
