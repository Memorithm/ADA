use std::collections::BTreeMap;
use std::env;
use std::io;

use ada_a2_real_v_trace::{ValueTraceRecord, read_value_trace_file};
use ada_a4_entmax_bnb::{EntmaxDistribution, dense_entmax};
use ada_a4_qk_box::dense_qk_scores;
use ada_a5_hierarchical_bounds::{
    HierarchicalKeyIndex, branch_and_bound_entmax_hierarchical_priority_lazy,
    build_hierarchical_key_index,
};
use ada_a5_real_qk_trace::{TraceRecord, read_trace_file};

const PAGE_SIZE: usize = 16;
const LEAF_SIZE: usize = 2;

const PROBABILITY_TOLERANCE: f64 = 2.0e-10;
const TAU_TOLERANCE: f64 = 1.0e-10;
const OUTPUT_TOLERANCE: f64 = 2.0e-10;

const EXPECTED_QK_RECORDS: usize = 3072;
const EXPECTED_V_RECORDS: usize = 384;
const EXPECTED_GROUPS: usize = 1536;

const EXPECTED_MODEL: &str = "Qwen/Qwen3-0.6B";

const EXPECTED_REVISION: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

const EXPECTED_QK_CAPTURE: &str = "qwen3-0.6b-a2-e3-allheads-wikitext2raw-val16";

const EXPECTED_V_CAPTURE: &str = "qwen3-0.6b-a2-e3b-v-wikitext2raw-val16";

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

#[derive(Debug)]
struct HeadEvaluation {
    loaded_tokens: Vec<bool>,
    support: Vec<bool>,
    probability_difference: f64,
    tau_difference: f64,
    full_vs_k_linf: f64,
    full_vs_support_linf: f64,
    k_vs_support_linf: f64,
}

#[derive(Debug, Default)]
struct Aggregate {
    groups: usize,
    heads: usize,
    visible_rows: usize,
    k_union: usize,
    support_union: usize,
    groups_without_residual_a2: usize,
    max_probability_difference: f64,
    max_tau_difference: f64,
    max_full_vs_k_linf: f64,
    max_full_vs_support_linf: f64,
    max_k_vs_support_linf: f64,
}

fn laboratory_error(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

fn usize_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }

    let numerator = u32::try_from(numerator).expect("E3b aggregate numerator must fit u32");

    let denominator = u32::try_from(denominator).expect("E3b aggregate denominator must fit u32");

    f64::from(numerator) / f64::from(denominator)
}

fn max_probability_difference(
    left: &EntmaxDistribution,
    right: &EntmaxDistribution,
) -> Result<f64, io::Error> {
    if left.probabilities.len() != right.probabilities.len() {
        return Err(laboratory_error("Entmax probability lengths differ"));
    }

    Ok(left
        .probabilities
        .iter()
        .zip(&right.probabilities)
        .map(|(&left_value, &right_value)| (left_value - right_value).abs())
        .fold(0.0_f64, f64::max))
}

fn linf_difference(left: &[f64], right: &[f64]) -> Result<f64, io::Error> {
    if left.len() != right.len() {
        return Err(laboratory_error("output vector lengths differ"));
    }

    Ok(left
        .iter()
        .zip(right)
        .map(|(&left_value, &right_value)| (left_value - right_value).abs())
        .fold(0.0_f64, f64::max))
}

fn weighted_output(
    distribution: &EntmaxDistribution,
    values: &[f64],
    head_dim: usize,
    active_rows: Option<&[bool]>,
) -> Result<Vec<f64>, io::Error> {
    if head_dim == 0 {
        return Err(laboratory_error("head_dim must be non-zero"));
    }

    let row_count = distribution.probabilities.len();

    let expected_values = row_count
        .checked_mul(head_dim)
        .ok_or_else(|| laboratory_error("V scalar count overflow"))?;

    if values.len() != expected_values {
        return Err(laboratory_error(
            "V prefix shape does not match probabilities",
        ));
    }

    if let Some(active) = active_rows {
        if active.len() != row_count {
            return Err(laboratory_error("active V-row mask length mismatch"));
        }
    }

    let mut output = vec![0.0_f64; head_dim];

    for (row_index, (&probability, row)) in distribution
        .probabilities
        .iter()
        .zip(values.chunks_exact(head_dim))
        .enumerate()
    {
        if let Some(active) = active_rows {
            if !active[row_index] {
                continue;
            }
        }

        for (output_value, &value) in output.iter_mut().zip(row) {
            *output_value += probability * value;
        }
    }

    Ok(output)
}

fn evaluate_head(
    record: &TraceRecord,
    value_record: &ValueTraceRecord,
    index: &HierarchicalKeyIndex,
    alpha: f64,
) -> Result<HeadEvaluation, io::Error> {
    let case = record
        .to_query_key_case(PAGE_SIZE, alpha)
        .map_err(|error| laboratory_error(error.to_string()))?;

    let dense_scores = dense_qk_scores(&case).map_err(laboratory_error)?;

    let dense_distribution = dense_entmax(&dense_scores, alpha).map_err(laboratory_error)?;

    let priority = branch_and_bound_entmax_hierarchical_priority_lazy(&case, index)
        .map_err(laboratory_error)?;

    if priority.loaded_tokens.len() != record.key_count {
        return Err(laboratory_error("A5 loaded-token mask length mismatch"));
    }

    let probability_difference =
        max_probability_difference(&dense_distribution, &priority.distribution)?;

    let tau_difference = (dense_distribution.tau - priority.distribution.tau).abs();

    let dense_support: Vec<bool> = dense_distribution
        .probabilities
        .iter()
        .map(|&probability| probability > 0.0)
        .collect();

    let support: Vec<bool> = priority
        .distribution
        .probabilities
        .iter()
        .map(|&probability| probability > 0.0)
        .collect();

    if dense_support != support {
        return Err(laboratory_error(
            "dense and priority Entmax supports differ",
        ));
    }

    if support
        .iter()
        .zip(&priority.loaded_tokens)
        .any(|(&in_support, &loaded)| in_support && !loaded)
    {
        return Err(laboratory_error(
            "final Entmax support escapes A5 K-loaded set",
        ));
    }

    if value_record.head_dim != record.head_dim {
        return Err(laboratory_error("Q/K and V head dimensions differ"));
    }

    if value_record.value_start_position != record.key_start_position {
        return Err(laboratory_error(
            "Q/K and V visible intervals start differently",
        ));
    }

    if value_record.value_count < record.key_count {
        return Err(laboratory_error(
            "ADAV01 record does not cover Q/K visible prefix",
        ));
    }

    let values = value_record
        .prefix_values(record.key_count)
        .map_err(|error| laboratory_error(error.to_string()))?;

    let full_output = weighted_output(&dense_distribution, values, record.head_dim, None)?;

    let k_loaded_output = weighted_output(
        &priority.distribution,
        values,
        record.head_dim,
        Some(&priority.loaded_tokens),
    )?;

    let support_output = weighted_output(
        &priority.distribution,
        values,
        record.head_dim,
        Some(&support),
    )?;

    let full_vs_k_linf = linf_difference(&full_output, &k_loaded_output)?;

    let full_vs_support_linf = linf_difference(&full_output, &support_output)?;

    let k_vs_support_linf = linf_difference(&k_loaded_output, &support_output)?;

    if probability_difference > PROBABILITY_TOLERANCE {
        return Err(laboratory_error(format!(
            "probability parity exceeded tolerance: {probability_difference:e}"
        )));
    }

    if tau_difference > TAU_TOLERANCE {
        return Err(laboratory_error(format!(
            "tau parity exceeded tolerance: {tau_difference:e}"
        )));
    }

    if full_vs_k_linf > OUTPUT_TOLERANCE
        || full_vs_support_linf > OUTPUT_TOLERANCE
        || k_vs_support_linf > OUTPUT_TOLERANCE
    {
        return Err(laboratory_error(format!(
            "natural-V output parity exceeded tolerance: full_k={full_vs_k_linf:e}, full_support={full_vs_support_linf:e}, k_support={k_vs_support_linf:e}"
        )));
    }

    Ok(HeadEvaluation {
        loaded_tokens: priority.loaded_tokens,
        support,
        probability_difference,
        tau_difference,
        full_vs_k_linf,
        full_vs_support_linf,
        k_vs_support_linf,
    })
}

fn union_count(left: &[bool], right: &[bool]) -> Result<usize, io::Error> {
    if left.len() != right.len() {
        return Err(laboratory_error("GQA union masks have different lengths"));
    }

    Ok(left
        .iter()
        .zip(right)
        .filter(|&(left_value, right_value)| *left_value || *right_value)
        .count())
}

fn union_subset(
    support_left: &[bool],
    support_right: &[bool],
    loaded_left: &[bool],
    loaded_right: &[bool],
) -> Result<bool, io::Error> {
    let length = support_left.len();

    if support_right.len() != length || loaded_left.len() != length || loaded_right.len() != length
    {
        return Err(laboratory_error("GQA masks have inconsistent lengths"));
    }

    Ok((0..length).all(|index| {
        let support = support_left[index] || support_right[index];

        let loaded = loaded_left[index] || loaded_right[index];

        !support || loaded
    }))
}

fn update_maximum(target: &mut f64, value: f64) {
    *target = target.max(value);
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
        return Err(laboratory_error(format!(
            "unexpected GQA pair for kv_head={}: got q{} and q{}",
            key.kv_head_index, first.query_head_index, second.query_head_index,
        )));
    }

    if first.keys != second.keys {
        return Err(laboratory_error(
            "two Q heads sharing a KV head do not share identical K rows",
        ));
    }

    if first.score_scale.to_bits() != second.score_scale.to_bits() {
        return Err(laboratory_error(
            "two Q heads sharing a KV head have different score scale",
        ));
    }

    Ok(())
}

fn expected_e3a_totals(alpha: f64) -> (usize, usize) {
    if alpha.to_bits() == 1.5_f64.to_bits() {
        (137_098, 8_555)
    } else if alpha.to_bits() == 2.0_f64.to_bits() {
        (110_536, 4_466)
    } else {
        unreachable!("E3b only evaluates frozen alpha values");
    }
}

// This executable intentionally keeps the complete qualification
// traversal visible in one auditable research entry point.
#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os();

    let _program = arguments.next();

    let qk_path = arguments
        .next()
        .ok_or_else(|| laboratory_error("usage: e3b_natural_v_gqa <trace.adaqk> <trace.adav>"))?;

    let v_path = arguments
        .next()
        .ok_or_else(|| laboratory_error("usage: e3b_natural_v_gqa <trace.adaqk> <trace.adav>"))?;

    if arguments.next().is_some() {
        return Err(laboratory_error("unexpected extra command-line argument").into());
    }

    let qk = read_trace_file(qk_path)?;

    let values = read_value_trace_file(v_path)?;

    if qk.len() != EXPECTED_QK_RECORDS {
        return Err(laboratory_error(format!(
            "expected {EXPECTED_QK_RECORDS} Q/K records, found {}",
            qk.len(),
        ))
        .into());
    }

    if values.len() != EXPECTED_V_RECORDS {
        return Err(laboratory_error(format!(
            "expected {EXPECTED_V_RECORDS} V records, found {}",
            values.len(),
        ))
        .into());
    }

    let qk_metadata = qk.metadata();

    let v_metadata = values.metadata();

    if qk_metadata.model_id != EXPECTED_MODEL
        || v_metadata.model_id != EXPECTED_MODEL
        || qk_metadata.model_revision != EXPECTED_REVISION
        || v_metadata.model_revision != EXPECTED_REVISION
    {
        return Err(laboratory_error(
            "Q/K and V model provenance does not match frozen E3b contract",
        )
        .into());
    }

    if qk_metadata.capture_id != EXPECTED_QK_CAPTURE {
        return Err(laboratory_error(format!(
            "unexpected Q/K capture id: {}",
            qk_metadata.capture_id,
        ))
        .into());
    }

    if v_metadata.capture_id != EXPECTED_V_CAPTURE {
        return Err(laboratory_error(format!(
            "unexpected V capture id: {}",
            v_metadata.capture_id,
        ))
        .into());
    }

    if qk_metadata.source_dtype != "bfloat16" || v_metadata.source_dtype != "bfloat16" {
        return Err(laboratory_error("Q/K/V source dtype must be bfloat16").into());
    }

    let mut v_index: BTreeMap<(String, u32, u32), usize> = BTreeMap::new();

    for (index, record) in values.records().iter().enumerate() {
        let key = (
            record.sample_id.clone(),
            record.layer_index,
            record.kv_head_index,
        );

        if v_index.insert(key, index).is_some() {
            return Err(laboratory_error("duplicate ADAV01 natural identity").into());
        }
    }

    if v_index.len() != EXPECTED_V_RECORDS {
        return Err(laboratory_error("ADAV01 identity count mismatch").into());
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
        return Err(laboratory_error(format!(
            "expected {EXPECTED_GROUPS} GQA groups, found {}",
            groups.len(),
        ))
        .into());
    }

    let alphas = [1.5_f64, 2.0_f64];

    let mut aggregates: BTreeMap<u64, Aggregate> = BTreeMap::new();

    println!("experiment=ADA-A2-E3B-NATURAL-V-GQA");

    println!("qk_record_count={}", qk.len());

    println!("v_record_count={}", values.len());

    println!("gqa_group_count={}", groups.len());

    println!("page_size={PAGE_SIZE}");

    println!("leaf_size={LEAF_SIZE}");

    println!("physical_v_traffic_measured=false");

    println!("natural_v_values_used=true");

    println!("gqa_unique_row_accounting=true");

    println!("attention_semantics=exact_sparse_entmax_lab");

    for (key, member_indices) in &groups {
        if member_indices.len() != 2 {
            return Err(laboratory_error(format!(
                "GQA group has {} Q heads instead of 2",
                member_indices.len(),
            ))
            .into());
        }

        let mut ordered = member_indices.clone();

        ordered.sort_by_key(|&index| qk.records()[index].query_head_index);

        let first = &qk.records()[ordered[0]];

        let second = &qk.records()[ordered[1]];

        validate_pair(first, second, key)?;

        let v_key = (key.sample_id.clone(), key.layer_index, key.kv_head_index);

        let v_index_value = v_index.get(&v_key).ok_or_else(|| {
            laboratory_error(format!(
                "missing ADAV01 record for sample={},layer={},kv={}",
                key.sample_id, key.layer_index, key.kv_head_index,
            ))
        })?;

        let value_record = &values.records()[*v_index_value];

        if value_record.value_start_position != key.key_start_position {
            return Err(laboratory_error("ADAV01 and ADAQK01 interval starts differ").into());
        }

        if value_record.value_count < key.key_count {
            return Err(laboratory_error("ADAV01 does not contain full visible Q/K prefix").into());
        }

        if value_record.head_dim != key.head_dim {
            return Err(laboratory_error("ADAV01 and ADAQK01 head dimensions differ").into());
        }

        let index = build_hierarchical_key_index(&first.keys, first.head_dim, PAGE_SIZE, LEAF_SIZE)
            .map_err(laboratory_error)?;

        for &alpha in &alphas {
            let first_eval = evaluate_head(first, value_record, &index, alpha)?;

            let second_eval = evaluate_head(second, value_record, &index, alpha)?;

            let k_union = union_count(&first_eval.loaded_tokens, &second_eval.loaded_tokens)?;

            let support_union = union_count(&first_eval.support, &second_eval.support)?;

            if !union_subset(
                &first_eval.support,
                &second_eval.support,
                &first_eval.loaded_tokens,
                &second_eval.loaded_tokens,
            )? {
                return Err(laboratory_error("GQA support union escapes K-loaded union").into());
            }

            let a2_avoidance = if k_union == 0 {
                0.0
            } else {
                1.0 - usize_ratio(support_union, k_union)
            };

            let probability_difference = first_eval
                .probability_difference
                .max(second_eval.probability_difference);

            let tau_difference = first_eval.tau_difference.max(second_eval.tau_difference);

            let full_vs_k_linf = first_eval.full_vs_k_linf.max(second_eval.full_vs_k_linf);

            let full_vs_support_linf = first_eval
                .full_vs_support_linf
                .max(second_eval.full_vs_support_linf);

            let k_vs_support_linf = first_eval
                .k_vs_support_linf
                .max(second_eval.k_vs_support_linf);

            println!(
                "group,alpha={alpha:.1},sample_fingerprint={:016x},layer={},kv_head={},q0={},q1={},query_position={},key_count={},k_union={},support_union={},a2_v_avoidance_after_k={a2_avoidance:.9},max_probability_difference={probability_difference:.17e},max_tau_difference={tau_difference:.17e},max_full_vs_k_linf={full_vs_k_linf:.17e},max_full_vs_support_linf={full_vs_support_linf:.17e},max_k_vs_support_linf={k_vs_support_linf:.17e}",
                first.sample_fingerprint(),
                key.layer_index,
                key.kv_head_index,
                first.query_head_index,
                second.query_head_index,
                key.query_position,
                key.key_count,
                k_union,
                support_union,
            );

            let aggregate = aggregates.entry(alpha.to_bits()).or_default();

            aggregate.groups += 1;
            aggregate.heads += 2;
            aggregate.visible_rows += key.key_count;

            aggregate.k_union += k_union;

            aggregate.support_union += support_union;

            if k_union == support_union {
                aggregate.groups_without_residual_a2 += 1;
            }

            update_maximum(
                &mut aggregate.max_probability_difference,
                probability_difference,
            );

            update_maximum(&mut aggregate.max_tau_difference, tau_difference);

            update_maximum(&mut aggregate.max_full_vs_k_linf, full_vs_k_linf);

            update_maximum(
                &mut aggregate.max_full_vs_support_linf,
                full_vs_support_linf,
            );

            update_maximum(&mut aggregate.max_k_vs_support_linf, k_vs_support_linf);
        }
    }

    let mut e3a_accounting_reproduced = true;

    let mut numerical_output_parity_ok = true;

    for &alpha in &alphas {
        let aggregate = aggregates
            .get(&alpha.to_bits())
            .ok_or_else(|| laboratory_error("missing alpha aggregate"))?;

        if aggregate.groups != EXPECTED_GROUPS || aggregate.heads != EXPECTED_QK_RECORDS {
            return Err(laboratory_error("aggregate case count mismatch").into());
        }

        let (expected_k_union, expected_support_union) = expected_e3a_totals(alpha);

        let e3a_match = aggregate.k_union == expected_k_union
            && aggregate.support_union == expected_support_union;

        e3a_accounting_reproduced &= e3a_match;

        let residual_a2 = 1.0 - usize_ratio(aggregate.support_union, aggregate.k_union);

        let total_v_avoidance = 1.0 - usize_ratio(aggregate.support_union, aggregate.visible_rows);

        let output_ok = aggregate.max_full_vs_k_linf <= OUTPUT_TOLERANCE
            && aggregate.max_full_vs_support_linf <= OUTPUT_TOLERANCE
            && aggregate.max_k_vs_support_linf <= OUTPUT_TOLERANCE
            && aggregate.max_probability_difference <= PROBABILITY_TOLERANCE
            && aggregate.max_tau_difference <= TAU_TOLERANCE;

        numerical_output_parity_ok &= output_ok;

        println!(
            "aggregate,scope=global,alpha={alpha:.1},groups={},head_cases={},visible_rows={},k_union={},support_union={},weighted_k_union_fraction={:.9},weighted_support_union_fraction={:.9},weighted_a2_v_avoidance_after_k={residual_a2:.9},weighted_total_v_avoidance={total_v_avoidance:.9},groups_without_residual_a2={},max_probability_difference={:.17e},max_tau_difference={:.17e},max_full_vs_k_linf={:.17e},max_full_vs_support_linf={:.17e},max_k_vs_support_linf={:.17e},e3a_accounting_match={e3a_match},output_parity_ok={output_ok}",
            aggregate.groups,
            aggregate.heads,
            aggregate.visible_rows,
            aggregate.k_union,
            aggregate.support_union,
            usize_ratio(aggregate.k_union, aggregate.visible_rows,),
            usize_ratio(aggregate.support_union, aggregate.visible_rows,),
            aggregate.groups_without_residual_a2,
            aggregate.max_probability_difference,
            aggregate.max_tau_difference,
            aggregate.max_full_vs_k_linf,
            aggregate.max_full_vs_support_linf,
            aggregate.max_k_vs_support_linf,
        );
    }

    println!("group_alpha_case_count={}", EXPECTED_GROUPS * 2);

    println!("head_alpha_case_count={}", EXPECTED_QK_RECORDS * 2);

    println!("e3a_accounting_reproduced={e3a_accounting_reproduced}");

    println!("numerical_output_parity_ok={numerical_output_parity_ok}");

    let join_contract_ok = e3a_accounting_reproduced && numerical_output_parity_ok;

    println!("join_contract_ok={join_contract_ok}");

    println!("survey_status=complete");

    if !join_contract_ok {
        return Err(laboratory_error("A2-E3b natural Q/K/V qualification contract failed").into());
    }

    Ok(())
}
