//! Two-statistic QK-box certified block-sparse softmax.
//!
//! Cycles #67 and #68 certify an L1 omission bound from *score-level*
//! block statistics. Forming those statistics requires reading every
//! score, so if the scores themselves are scaled dots `scale * <q, k_i>`
//! the expensive inner products have already been paid.
//!
//! This module composes the A3 greedy skip rule with A4 coordinate boxes
//! and a depth-1 A5 split of each contiguous key block:
//!
//! - each block stores the axis-aligned box of each half (min/max per
//!   coordinate), built once from the keys without forming scores;
//! - at query time, the A4 inequalities
//!   `hi = scale * Σ_j max(q_j kmin_j, q_j kmax_j)` and
//!   `lo = scale * Σ_j min(q_j kmin_j, q_j kmax_j)` bound every score
//!   in a half without reading the half's keys;
//! - one witness key per half (its first row) supplies a lower bound
//!   `M_lo` on the true maximum score `M` and a tighter mass lower bound;
//! - `M_hi = max half-box hi` upper-bounds `M`.
//!
//! Then, writing `n_h` for a half's length,
//!
//! - `U_b = Σ_{h ∈ b} n_h * exp(hi_h - M_lo)` upper-bounds the true
//!   `M`-shifted block mass, because `M >= M_lo` and every score is
//!   at most its half's `hi`;
//! - `L_b = Σ_{h ∈ b} [exp(w_h - M_hi) + (n_h - 1) * exp(lo_h - M_hi)]`
//!   lower-bounds it, because `M <= M_hi`, the witness score is exact,
//!   and every other score is at least `lo_h`;
//! - `L = Σ_b L_b` lower-bounds the total mass `S`;
//! - skipping a set `K` of blocks whose box-max is strictly below `M_lo`
//!   (so they cannot contain a top-plateau token) and renormalizing the
//!   complement yields `||p - q||_1 = 2 S_K / S <= 2 (Σ_{b ∈ K} U_b) / L`.
//!
//! A single full-block box cannot separate the max from the second max:
//! the same corner bound applies to every key. Splitting the block is
//! what makes the two-half bound never looser than the first-order box
//! bound `n_b * exp(hi_b - M_lo)`. First-order is exposed for comparison.
//!
//! Query-time work materializes a key row only as a witness or as a member
//! of a kept block. Skipped blocks contribute at most their witnesses.
//! Logical counters are not hardware measurements. No usefulness, novelty,
//! or FLAT-adoption claim is made.

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QkBoxCost {
    /// Contiguous blocks in the partition (the last one may be shorter).
    pub blocks_total: usize,
    /// Blocks assigned probability zero by the skip rule.
    pub blocks_skipped: usize,
    /// Half-boxes evaluated against the query (`hi` and `lo` share one pass).
    pub box_bound_evaluations: usize,
    /// Key rows dotted to form per-half witnesses (at most two per block).
    pub witness_dots: usize,
    /// Key rows dotted in the second pass (every token of every kept block).
    pub key_dots_materialized: usize,
    /// Distinct key rows read at query time (witnesses plus non-witness kept rows).
    pub query_time_key_rows_read: usize,
    /// Distinct `exp` evaluations used for `(U_b, L_b)` plus kept tokens.
    pub exp_evaluations: usize,
    /// Greedy certificate checks (one per block that is not possibly top).
    pub bound_checks: usize,
    /// Tokens that received a positive unnormalized `exp` in the second pass.
    pub tokens_exponentiated: usize,
}

/// Certified QK-box block-sparse softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct QkBoxSoftmax {
    /// Renormalized probabilities in input order. Skipped tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate `2 * sum_{skipped} U_b / L`.
    pub certified_l1_bound: f64,
    /// Caller budget that the certificate was required to respect.
    pub epsilon: f64,
    /// Whether each contiguous block was skipped, in block order.
    pub block_skipped: Vec<bool>,
    /// Requested block size (the last block may be shorter).
    pub block_size: usize,
    /// Positive scale applied to every `q · k` score.
    pub score_scale: f64,
}

/// How a block's mass is bounded from key boxes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QkBoxBoundMode {
    /// One box for the whole block: `U_b = n_b * exp(hi - M_lo)`.
    FirstOrder,
    /// A4/A5 two-half boxes: `U_b = Σ_h n_h * exp(hi_h - M_lo)`.
    TwoStat,
}

#[derive(Clone, Debug)]
struct CoordBox {
    minimum: Vec<f64>,
    maximum: Vec<f64>,
}

#[derive(Clone, Debug)]
struct HalfStat {
    start: usize,
    end: usize,
    hi: f64,
    lo: f64,
    witness: f64,
}

#[derive(Clone, Debug)]
struct BlockStat {
    start: usize,
    end: usize,
    halves: Vec<HalfStat>,
    upper: f64,
    lower: f64,
    box_max: f64,
}

fn validate_args(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    epsilon: f64,
) -> Result<usize, &'static str> {
    if head_dim == 0 {
        return Err("ADA-A3 QK-box softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 QK-box softmax query must contain head_dim elements");
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A3 QK-box softmax query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 QK-box softmax requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 QK-box softmax keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A3 QK-box softmax keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 QK-box softmax score_scale must be finite and positive");
    }
    if block_size == 0 {
        return Err("ADA-A3 QK-box softmax block_size must be positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 QK-box softmax epsilon must be finite and non-negative");
    }
    Ok(keys.len() / head_dim)
}

fn build_coord_box(keys: &[f64], head_dim: usize, start: usize, end: usize) -> CoordBox {
    let first = &keys[start * head_dim..(start + 1) * head_dim];
    let mut minimum = first.to_vec();
    let mut maximum = first.to_vec();
    for row in start + 1..end {
        let values = &keys[row * head_dim..(row + 1) * head_dim];
        for ((min_value, max_value), &value) in minimum
            .iter_mut()
            .zip(maximum.iter_mut())
            .zip(values.iter())
        {
            *min_value = min_value.min(value);
            *max_value = max_value.max(value);
        }
    }
    CoordBox { minimum, maximum }
}

fn box_bounds(
    query: &[f64],
    page: &CoordBox,
    score_scale: f64,
) -> Result<(f64, f64), &'static str> {
    let mut hi_sum = 0.0_f64;
    let mut lo_sum = 0.0_f64;
    for ((&q, &minimum), &maximum) in query
        .iter()
        .zip(page.minimum.iter())
        .zip(page.maximum.iter())
    {
        if minimum > maximum {
            return Err("ADA-A3 QK-box softmax box coordinates must be ordered");
        }
        let q_min = q * minimum;
        let q_max = q * maximum;
        hi_sum += q_min.max(q_max);
        lo_sum += q_min.min(q_max);
    }
    let hi = hi_sum * score_scale;
    let lo = lo_sum * score_scale;
    if !hi.is_finite() || !lo.is_finite() {
        return Err("ADA-A3 QK-box softmax box bound overflowed");
    }
    Ok((hi, lo))
}

fn dot_score(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    row: usize,
    score_scale: f64,
) -> Result<f64, &'static str> {
    let key = &keys[row * head_dim..(row + 1) * head_dim];
    let mut dot = 0.0_f64;
    for (&q, &k) in query.iter().zip(key.iter()) {
        dot += q * k;
    }
    let score = dot * score_scale;
    if !score.is_finite() {
        return Err("ADA-A3 QK-box softmax witness or kept score overflowed");
    }
    Ok(score)
}

fn evaluate_half(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    start: usize,
    end: usize,
) -> Result<HalfStat, &'static str> {
    let page = build_coord_box(keys, head_dim, start, end);
    let (hi, lo) = box_bounds(query, &page, score_scale)?;
    let witness = dot_score(query, keys, head_dim, start, score_scale)?;
    // Absorb rounding so a witness cannot sit outside its own box.
    let hi = hi.max(witness);
    let lo = lo.min(witness);
    Ok(HalfStat {
        start,
        end,
        hi,
        lo,
        witness,
    })
}

fn split_block(start: usize, end: usize, mode: QkBoxBoundMode) -> Vec<(usize, usize)> {
    match mode {
        QkBoxBoundMode::FirstOrder => vec![(start, end)],
        QkBoxBoundMode::TwoStat => {
            let length = end - start;
            if length <= 1 {
                vec![(start, end)]
            } else {
                let mid = start + length / 2;
                vec![(start, mid), (mid, end)]
            }
        }
    }
}

fn partition_blocks(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    key_count: usize,
    score_scale: f64,
    block_size: usize,
    mode: QkBoxBoundMode,
) -> Result<(Vec<BlockStat>, usize, usize), &'static str> {
    let mut blocks = Vec::new();
    let mut box_bound_evaluations = 0_usize;
    let mut witness_dots = 0_usize;
    let mut offset = 0;
    while offset < key_count {
        let end = (offset + block_size).min(key_count);
        let mut halves = Vec::new();
        for (half_start, half_end) in split_block(offset, end, mode) {
            halves.push(evaluate_half(
                query,
                keys,
                head_dim,
                score_scale,
                half_start,
                half_end,
            )?);
            box_bound_evaluations = box_bound_evaluations.saturating_add(1);
            witness_dots = witness_dots.saturating_add(1);
        }
        let box_max = halves
            .iter()
            .map(|half| half.hi)
            .fold(f64::NEG_INFINITY, f64::max);
        blocks.push(BlockStat {
            start: offset,
            end,
            halves,
            upper: 0.0,
            lower: 0.0,
            box_max,
        });
        offset = end;
    }
    Ok((blocks, box_bound_evaluations, witness_dots))
}

fn shifted_exp(score: f64, shift: f64) -> Result<f64, &'static str> {
    let term = (score - shift).exp();
    if !term.is_finite() || term < 0.0 {
        return Err("ADA-A3 QK-box softmax bound term must be finite");
    }
    Ok(term)
}

fn annotate_block_bounds(blocks: &mut [BlockStat]) -> Result<(f64, f64, f64, usize), &'static str> {
    let max_lo = blocks
        .iter()
        .flat_map(|block| block.halves.iter())
        .map(|half| half.witness)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_hi = blocks
        .iter()
        .map(|block| block.box_max)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lo.is_finite() || !max_hi.is_finite() {
        return Err("ADA-A3 QK-box softmax global score bounds must be finite");
    }
    let mut mass_lower = 0.0_f64;
    let mut exp_evaluations = 0_usize;
    for block in blocks.iter_mut() {
        let mut upper = 0.0_f64;
        let mut lower = 0.0_f64;
        for half in &block.halves {
            let exp_hi = shifted_exp(half.hi, max_lo)?;
            exp_evaluations = exp_evaluations.saturating_add(1);
            // L uses the M_hi shift; never reuse exp_hi (shifted by M_lo).
            let exp_lo = shifted_exp(half.lo, max_hi)?;
            exp_evaluations = exp_evaluations.saturating_add(1);
            let exp_witness = if half.witness.total_cmp(&half.lo).is_eq() {
                exp_lo
            } else {
                exp_evaluations = exp_evaluations.saturating_add(1);
                shifted_exp(half.witness, max_hi)?
            };
            let length = half.end - half.start;
            #[allow(clippy::cast_precision_loss)]
            let rest = length.saturating_sub(1) as f64;
            #[allow(clippy::cast_precision_loss)]
            let length_f = length as f64;
            upper += length_f * exp_hi;
            lower += exp_witness + rest * exp_lo;
        }
        if !upper.is_finite() || upper < 0.0 {
            return Err("ADA-A3 QK-box softmax block upper bound must be finite");
        }
        block.upper = upper;
        block.lower = lower;
        mass_lower += lower;
    }
    Ok((max_lo, max_hi, mass_lower, exp_evaluations))
}

struct SkipPlan {
    omitted: Vec<bool>,
    omitted_count: usize,
    omitted_upper: f64,
    bound_checks: usize,
}

fn plan_skips(blocks: &[BlockStat], max_lo: f64, mass_lower: f64, epsilon: f64) -> SkipPlan {
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        blocks[left]
            .upper
            .total_cmp(&blocks[right].upper)
            .then_with(|| left.cmp(&right))
    });

    let mut omitted = vec![false; blocks.len()];
    let mut omitted_upper = 0.0_f64;
    let mut bound_checks = 0_usize;
    let mut omitted_count = 0_usize;
    if !mass_lower.is_finite() || mass_lower <= 0.0 {
        return SkipPlan {
            omitted,
            omitted_count,
            omitted_upper,
            bound_checks,
        };
    }
    for index in order {
        // A block whose box-max reaches M_lo might hold the true maximum.
        if blocks[index].box_max >= max_lo {
            continue;
        }
        bound_checks = bound_checks.saturating_add(1);
        let next = omitted_upper + blocks[index].upper;
        if 2.0 * next / mass_lower <= epsilon {
            omitted[index] = true;
            omitted_upper = next;
            omitted_count = omitted_count.saturating_add(1);
        }
    }
    SkipPlan {
        omitted,
        omitted_count,
        omitted_upper,
        bound_checks,
    }
}

fn renormalize_kept(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    key_count: usize,
    blocks: &[BlockStat],
    omitted: &[bool],
) -> Result<(Vec<f64>, usize), &'static str> {
    let mut scores = vec![0.0_f64; key_count];
    let mut tokens_exponentiated = 0_usize;
    let mut kept_shift = f64::NEG_INFINITY;
    for (block, skipped) in blocks.iter().zip(omitted.iter().copied()) {
        if skipped {
            continue;
        }
        for (row, slot) in scores
            .iter_mut()
            .enumerate()
            .take(block.end)
            .skip(block.start)
        {
            *slot = dot_score(query, keys, head_dim, row, score_scale)?;
            if *slot > kept_shift {
                kept_shift = *slot;
            }
        }
    }
    if !kept_shift.is_finite() {
        return Err("ADA-A3 QK-box softmax kept scores must be finite");
    }
    let mut kept_mass = 0.0_f64;
    let mut probabilities = vec![0.0_f64; key_count];
    for (block, skipped) in blocks.iter().zip(omitted.iter().copied()) {
        if skipped {
            continue;
        }
        for (score, slot) in scores[block.start..block.end]
            .iter()
            .zip(probabilities[block.start..block.end].iter_mut())
        {
            let term = (*score - kept_shift).exp();
            tokens_exponentiated = tokens_exponentiated.saturating_add(1);
            if !term.is_finite() || term < 0.0 {
                return Err("ADA-A3 QK-box softmax kept term must be finite");
            }
            *slot = term;
            kept_mass += term;
        }
    }
    if !kept_mass.is_finite() || kept_mass <= 0.0 {
        return Err("ADA-A3 QK-box softmax kept mass must be positive");
    }
    for probability in &mut probabilities {
        *probability /= kept_mass;
    }
    Ok((probabilities, tokens_exponentiated))
}

fn query_time_rows_read(blocks: &[BlockStat], omitted: &[bool]) -> (usize, usize) {
    let mut read = 0_usize;
    let mut materialized = 0_usize;
    for (block, skipped) in blocks.iter().zip(omitted.iter().copied()) {
        if skipped {
            read = read.saturating_add(block.halves.len());
            continue;
        }
        materialized = materialized.saturating_add(block.end - block.start);
        read = read.saturating_add(block.end - block.start);
    }
    (read, materialized)
}

fn certified_qk_box_softmax_accounted_with_mode(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    epsilon: f64,
    mode: QkBoxBoundMode,
) -> Result<(QkBoxSoftmax, QkBoxCost), &'static str> {
    let key_count = validate_args(query, keys, head_dim, score_scale, block_size, epsilon)?;
    let (mut blocks, box_bound_evaluations, witness_dots) = partition_blocks(
        query,
        keys,
        head_dim,
        key_count,
        score_scale,
        block_size,
        mode,
    )?;
    let (max_lo, _max_hi, mass_lower, bound_exps) = annotate_block_bounds(&mut blocks)?;
    let plan = plan_skips(&blocks, max_lo, mass_lower, epsilon);
    let certified_l1_bound = if plan.omitted_count == 0 {
        0.0
    } else {
        2.0 * plan.omitted_upper / mass_lower
    };
    if !certified_l1_bound.is_finite() || certified_l1_bound > epsilon {
        return Err("ADA-A3 QK-box softmax certificate exceeds the supplied budget");
    }
    let (probabilities, tokens_exponentiated) = renormalize_kept(
        query,
        keys,
        head_dim,
        score_scale,
        key_count,
        &blocks,
        &plan.omitted,
    )?;
    let (query_time_key_rows_read, key_dots_materialized) =
        query_time_rows_read(&blocks, &plan.omitted);
    let cost = QkBoxCost {
        blocks_total: blocks.len(),
        blocks_skipped: plan.omitted_count,
        box_bound_evaluations,
        witness_dots,
        key_dots_materialized,
        query_time_key_rows_read,
        exp_evaluations: bound_exps.saturating_add(tokens_exponentiated),
        bound_checks: plan.bound_checks,
        tokens_exponentiated,
    };
    let result = QkBoxSoftmax {
        probabilities,
        certified_l1_bound,
        epsilon,
        block_skipped: plan.omitted,
        block_size,
        score_scale,
    };
    Ok((result, cost))
}

/// Evaluate two-statistic QK-box certified block-sparse softmax and record counters.
///
/// # Errors
///
/// Returns an error when the Q/K layout is empty or inconsistent, when any
/// input is non-finite, when `score_scale` is not finite and positive, when
/// `block_size` is zero, when `epsilon` is non-finite or negative, or when a
/// mass residual is non-finite.
#[must_use = "the certified distribution should be checked"]
pub fn certified_qk_box_two_stat_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    epsilon: f64,
) -> Result<(QkBoxSoftmax, QkBoxCost), &'static str> {
    certified_qk_box_softmax_accounted_with_mode(
        query,
        keys,
        head_dim,
        score_scale,
        block_size,
        epsilon,
        QkBoxBoundMode::TwoStat,
    )
}

/// Evaluate two-statistic QK-box certified block-sparse softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_qk_box_two_stat_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_qk_box_two_stat_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    epsilon: f64,
) -> Result<QkBoxSoftmax, &'static str> {
    certified_qk_box_two_stat_softmax_accounted(
        query,
        keys,
        head_dim,
        score_scale,
        block_size,
        epsilon,
    )
    .map(|(result, _)| result)
}

/// Evaluate first-order (single-box) QK-box certified block-sparse softmax.
///
/// Exposed so tests can compare the two-half bound against the looser
/// full-block box on the same partition and budget.
///
/// # Errors
///
/// Same failure conditions as [`certified_qk_box_two_stat_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_qk_box_first_order_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    epsilon: f64,
) -> Result<(QkBoxSoftmax, QkBoxCost), &'static str> {
    certified_qk_box_softmax_accounted_with_mode(
        query,
        keys,
        head_dim,
        score_scale,
        block_size,
        epsilon,
        QkBoxBoundMode::FirstOrder,
    )
}

/// Dense scaled QK scores. Laboratory helper, not used by the operator.
///
/// # Errors
///
/// Returns an error for an inconsistent layout or a non-finite score.
#[must_use = "dense scores feed the independent softmax oracle"]
pub fn dense_qk_scores(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
) -> Result<Vec<f64>, &'static str> {
    if head_dim == 0 || query.len() != head_dim {
        return Err("ADA-A3 QK-box dense scores require a matching head_dim");
    }
    if keys.is_empty() || !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 QK-box dense scores require row-major keys");
    }
    let mut scores = Vec::with_capacity(keys.len() / head_dim);
    for row in 0..keys.len() / head_dim {
        scores.push(dot_score(query, keys, head_dim, row, score_scale)?);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::{
        QkBoxSoftmax, certified_qk_box_first_order_softmax_accounted,
        certified_qk_box_two_stat_softmax, certified_qk_box_two_stat_softmax_accounted,
        dense_qk_scores,
    };

    fn oracle_softmax(scores: &[f64]) -> Vec<f64> {
        let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let terms: Vec<f64> = scores.iter().map(|&score| (score - shift).exp()).collect();
        let mass: f64 = terms.iter().sum();
        terms.into_iter().map(|term| term / mass).collect()
    }

    fn l1_distance(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum()
    }

    fn assert_certificate_holds(scores: &[f64], result: &QkBoxSoftmax) {
        let oracle = oracle_softmax(scores);
        let observed = l1_distance(&result.probabilities, &oracle);
        assert!(
            observed <= result.certified_l1_bound + 1.0e-12,
            "L1 {observed} exceeded certificate {} + 1e-12 on {scores:?}",
            result.certified_l1_bound
        );
        assert!(result.certified_l1_bound <= result.epsilon + 1.0e-15);
        let mass: f64 = result.probabilities.iter().sum();
        assert!((mass - 1.0).abs() <= 1.0e-12, "mass {mass}");
        assert!(
            result
                .probabilities
                .iter()
                .all(|p| p.is_finite() && *p >= 0.0)
        );
        let true_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for (block_index, skipped) in result.block_skipped.iter().copied().enumerate() {
            let start = block_index * result.block_size;
            let end = (start + result.block_size).min(scores.len());
            if skipped {
                assert!(
                    result.probabilities[start..end]
                        .iter()
                        .all(|p| p.abs() <= 0.0)
                );
                assert!(
                    scores[start..end].iter().all(|&score| score < true_max),
                    "skipped a top-plateau token in block {block_index}"
                );
            }
        }
    }

    fn run_two(
        query: &[f64],
        keys: &[f64],
        head_dim: usize,
        score_scale: f64,
        block_size: usize,
        epsilon: f64,
    ) -> (QkBoxSoftmax, super::QkBoxCost, Vec<f64>) {
        let (result, cost) = certified_qk_box_two_stat_softmax_accounted(
            query,
            keys,
            head_dim,
            score_scale,
            block_size,
            epsilon,
        )
        .unwrap();
        let scores = dense_qk_scores(query, keys, head_dim, score_scale).unwrap();
        assert_certificate_holds(&scores, &result);
        (result, cost, scores)
    }

    #[test]
    fn rejects_invalid_inputs() {
        let q = [1.0_f64];
        let k = [1.0_f64, 0.0];
        assert!(certified_qk_box_two_stat_softmax(&[], &k, 1, 1.0, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_two_stat_softmax(&q, &[], 1, 1.0, 1, 1.0e-3).is_err());
        assert!(
            certified_qk_box_two_stat_softmax(&q, &[1.0, 0.0, 1.0], 2, 1.0, 1, 1.0e-3).is_err()
        );
        assert!(certified_qk_box_two_stat_softmax(&[f64::NAN], &k, 1, 1.0, 1, 1.0e-3).is_err());
        assert!(
            certified_qk_box_two_stat_softmax(&q, &[1.0, f64::INFINITY], 1, 1.0, 1, 1.0e-3)
                .is_err()
        );
        assert!(certified_qk_box_two_stat_softmax(&q, &k, 0, 1.0, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_two_stat_softmax(&q, &k, 1, 0.0, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_two_stat_softmax(&q, &k, 1, -1.0, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_two_stat_softmax(&q, &k, 1, 1.0, 0, 1.0e-3).is_err());
        assert!(certified_qk_box_two_stat_softmax(&q, &k, 1, 1.0, 1, -0.1).is_err());
        assert!(certified_qk_box_two_stat_softmax(&q, &k, 1, 1.0, 1, f64::NAN).is_err());
    }

    #[test]
    fn singleton_and_all_equal_never_skip() {
        let query = [1.0_f64, 0.0];
        let one_key = [2.0_f64, -3.0];
        let (one, cost, _) = run_two(&query, &one_key, 2, 1.0, 4, 1.0e-6);
        assert_eq!(one.probabilities, vec![1.0]);
        assert_eq!(one.block_skipped, vec![false]);
        assert!(one.certified_l1_bound.abs() <= f64::EPSILON);
        assert_eq!(cost.witness_dots, 1);
        assert_eq!(cost.key_dots_materialized, 1);

        let keys = [1.0_f64, 0.0].repeat(8);
        let (equal, cost, _) = run_two(&query, &keys, 2, 1.0, 2, 1.0e-4);
        assert!(equal.block_skipped.iter().all(|skipped| !skipped));
        assert!(equal.certified_l1_bound.abs() <= f64::EPSILON);
        for probability in &equal.probabilities {
            assert!((probability - 0.125).abs() <= 1.0e-12);
        }
        assert_eq!(cost.key_dots_materialized, 8);
        assert_eq!(cost.query_time_key_rows_read, 8);
    }

    #[test]
    fn never_skips_the_global_maximum_block() {
        let query = [1.0_f64, 0.0];
        let mut keys = vec![0.0_f64; 16];
        // Block 1 (rows 4..8) holds the unique aligned key at row 5.
        keys[10] = 3.0;
        keys[11] = 0.0;
        let (result, cost, scores) = run_two(&query, &keys, 2, 1.0, 4, 1.0);
        assert!(!result.block_skipped[1], "max block must be kept");
        let peak = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
            .0;
        assert_eq!(peak, 5);
        let prob_peak = result
            .probabilities
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
            .0;
        assert_eq!(prob_peak, 5);
        assert!(cost.blocks_skipped < cost.blocks_total);
    }

    #[test]
    fn anti_aligned_blocks_are_skipped_without_reading_every_key() {
        let query = [1.0_f64, 0.0];
        let mut keys = vec![0.0_f64; 24];
        // Three blocks of 4 keys in dim 2. Block 2 holds the needle.
        keys[16] = 1.0;
        for row in 0..12 {
            if row == 8 {
                continue;
            }
            keys[2 * row] = -20.0;
        }
        let (result, cost, _) = run_two(&query, &keys, 2, 1.0, 4, 1.0e-6);
        assert!(result.block_skipped[0]);
        assert!(result.block_skipped[1]);
        assert!(!result.block_skipped[2]);
        assert_eq!(cost.blocks_skipped, 2);
        assert_eq!(cost.key_dots_materialized, 4);
        // Two skipped blocks contribute their witnesses only (two halves each).
        assert!(cost.query_time_key_rows_read < 12);
        assert!(result.probabilities[8] > 0.999_999);
        assert!(result.certified_l1_bound <= 1.0e-6);
    }

    #[test]
    fn two_stat_skips_a_mixed_block_that_first_order_keeps() {
        // Block 0: four copies of q, unique top plateau.
        // Block 1: two moderately aligned keys and two strongly anti-aligned
        // keys. The full-block box inherits the warm half's upper bound and
        // pretends all four keys are that warm; the two-half bound does not.
        let query = [1.0_f64, 0.0];
        let mut keys = Vec::new();
        for _ in 0..4 {
            keys.extend_from_slice(&[4.0, 0.0]);
        }
        keys.extend_from_slice(&[2.0, 0.0, 2.0, 0.0, -20.0, 0.0, -20.0, 0.0]);
        let epsilon = 0.2_f64;
        let (first, first_cost) =
            certified_qk_box_first_order_softmax_accounted(&query, &keys, 2, 1.0, 4, epsilon)
                .unwrap();
        let (two, two_cost, scores) = run_two(&query, &keys, 2, 1.0, 4, epsilon);
        assert_eq!(first_cost.blocks_skipped, 0, "first-order should keep both");
        assert!(!first.block_skipped[1]);
        assert_eq!(two_cost.blocks_skipped, 1);
        assert!(two.block_skipped[1]);
        assert!(!two.block_skipped[0]);
        assert!(two_cost.query_time_key_rows_read < scores.len());
        assert!(two.certified_l1_bound <= epsilon);
        assert_certificate_holds(&scores, &first);
    }

    #[test]
    fn unique_needle_hit_at_small_and_large_margins() {
        let query = [1.0_f64, 0.0];
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut keys = vec![0.0_f64; 16];
            keys[6] = margin;
            let (result, _, _) = run_two(&query, &keys, 2, 1.0, 2, 1.0e-4);
            let peak = result
                .probabilities
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            assert_eq!(peak, 3, "missed needle at margin {margin}");
        }
    }

    #[test]
    fn equal_pair_interpolation_is_preserved() {
        let query = [1.0_f64, 0.0];
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut keys = vec![0.0_f64; 16];
            keys[4] = margin;
            keys[10] = margin;
            let (result, _, _) = run_two(&query, &keys, 2, 1.0, 2, 1.0e-4);
            assert!(!result.block_skipped[1]);
            assert!(!result.block_skipped[2]);
            assert!((result.probabilities[2] - result.probabilities[5]).abs() <= 1.0e-12);
            assert!(result.probabilities[2] > result.probabilities[0]);
        }
    }

    #[test]
    fn zero_budget_matches_dense_softmax() {
        let query = [0.5_f64, -0.25];
        let keys = [
            1.0, 0.0, 0.0, 1.0, -1.0, 0.5, 0.25, -0.5, 2.0, 1.0, -0.5, 0.0,
        ];
        let (result, cost, scores) = run_two(&query, &keys, 2, 1.0, 2, 0.0);
        assert_eq!(cost.blocks_skipped, 0);
        assert!(result.certified_l1_bound.abs() <= f64::EPSILON);
        assert_eq!(cost.key_dots_materialized, scores.len());
        let observed = l1_distance(&result.probabilities, &oracle_softmax(&scores));
        assert!(observed <= 1.0e-14, "dense path L1 {observed}");
    }

    #[test]
    fn last_block_may_be_short() {
        let query = [1.0_f64];
        let keys = [0.0_f64, -4.0, -4.0, -4.0, 2.0];
        let (result, _, _) = run_two(&query, &keys, 1, 1.0, 3, 1.0e-3);
        assert_eq!(result.block_skipped.len(), 2);
        assert!(!result.block_skipped[1]);
    }

    fn grid_domain() -> (Vec<[f64; 2]>, usize, usize) {
        if cfg!(miri) {
            (vec![[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]], 3, 2)
        } else {
            (vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]], 4, 2)
        }
    }

    fn enumerate_key_layouts(length: usize, domain: &[[f64; 2]]) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        let mut current = vec![0_usize; length];
        let radix = domain.len();
        loop {
            let mut keys = Vec::with_capacity(length * 2);
            for &index in &current {
                keys.extend_from_slice(&domain[index]);
            }
            out.push(keys);
            let mut pos = 0;
            while pos < length {
                current[pos] += 1;
                if current[pos] < radix {
                    break;
                }
                current[pos] = 0;
                pos += 1;
            }
            if pos == length {
                break;
            }
        }
        out
    }

    #[test]
    fn exhaustive_grid_respects_certificate() {
        let query = [1.0_f64, 0.25];
        let (domain, max_len, _) = grid_domain();
        let mut vectors = 0_usize;
        let mut tight_skips = 0_usize;
        let mut loose_skips = 0_usize;
        let mut first_loose_skips = 0_usize;
        let mut two_extra_loose = 0_usize;
        let mut saved_row_cases = 0_usize;
        for length in 1..=max_len {
            for keys in enumerate_key_layouts(length, &domain) {
                for block_size in 1..=length {
                    let (tight, _, scores) = run_two(&query, &keys, 2, 1.0, block_size, 1.0e-3);
                    let (loose, loose_cost, _) = run_two(&query, &keys, 2, 1.0, block_size, 5.0e-2);
                    vectors += 1;
                    if tight.block_skipped.iter().any(|skipped| *skipped) {
                        tight_skips += 1;
                    }
                    if loose.block_skipped.iter().any(|skipped| *skipped) {
                        loose_skips += 1;
                    }
                    if loose_cost.query_time_key_rows_read < scores.len() {
                        saved_row_cases += 1;
                    }
                    let (first_loose, _) = certified_qk_box_first_order_softmax_accounted(
                        &query, &keys, 2, 1.0, block_size, 5.0e-2,
                    )
                    .unwrap();
                    assert_certificate_holds(&scores, &first_loose);
                    if first_loose.block_skipped.iter().any(|skipped| *skipped) {
                        first_loose_skips += 1;
                    }
                    let two_count = loose.block_skipped.iter().filter(|s| **s).count();
                    let first_count = first_loose.block_skipped.iter().filter(|s| **s).count();
                    if two_count > first_count {
                        two_extra_loose += 1;
                    }
                }
            }
        }
        assert!(vectors > 0);
        eprintln!(
            "qk_box exhaustive_grid vectors={vectors} tight_eps=1e-3 skips={tight_skips} loose_eps=5e-2 skips={loose_skips} first_order_loose_skips={first_loose_skips} two_stat_extra_loose={two_extra_loose} row_saving_loose_cases={saved_row_cases}"
        );
    }

    fn lcg_step(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        #[allow(clippy::cast_precision_loss)]
        {
            ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
        }
    }

    fn box_muller(state: &mut u64) -> (f64, f64) {
        let mut u1 = lcg_step(state);
        while u1 <= 1.0e-12 {
            u1 = lcg_step(state);
        }
        let u2 = lcg_step(state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        (radius * angle.cos(), radius * angle.sin())
    }

    fn fill_unit_gaussians(values: &mut [f64], state: &mut u64) {
        let mut index = 0;
        while index < values.len() {
            let (first, second) = box_muller(state);
            values[index] = first;
            if index + 1 < values.len() {
                values[index + 1] = second;
            }
            index += 2;
        }
    }

    fn survey_regime(
        label: &str,
        samples: usize,
        key_count: usize,
        head_dim: usize,
        clustered: bool,
        epsilon: f64,
    ) {
        let mut state = 0xA3B5_C0DE_2026_1110;
        let mut violations = 0_usize;
        let mut skip_sum = 0.0_f64;
        let mut first_skip_sum = 0.0_f64;
        let mut row_frac_sum = 0.0_f64;
        let mut max_observed = 0.0_f64;
        let mut max_bound = 0.0_f64;
        let block_size = if key_count >= 8 { 8 } else { 2 };
        let mut query = vec![0.0; head_dim];
        query[0] = 1.0;
        for _ in 0..samples {
            let mut keys = vec![0.0; key_count * head_dim];
            fill_unit_gaussians(&mut keys, &mut state);
            if clustered {
                let mut offset = 0;
                let mut block_index = 0_usize;
                while offset < key_count {
                    let end = (offset + block_size).min(key_count);
                    let polarity = if block_index == 0 { 6.0_f64 } else { -6.0_f64 };
                    for row in offset..end {
                        keys[row * head_dim] = polarity + 0.05 * keys[row * head_dim];
                        for dim in 1..head_dim {
                            keys[row * head_dim + dim] *= 0.05;
                        }
                    }
                    block_index += 1;
                    offset = end;
                }
            }
            let (result, cost) = certified_qk_box_two_stat_softmax_accounted(
                &query, &keys, head_dim, 1.0, block_size, epsilon,
            )
            .unwrap();
            let (first, first_cost) = certified_qk_box_first_order_softmax_accounted(
                &query, &keys, head_dim, 1.0, block_size, epsilon,
            )
            .unwrap();
            let scores = dense_qk_scores(&query, &keys, head_dim, 1.0).unwrap();
            let observed = l1_distance(&result.probabilities, &oracle_softmax(&scores));
            if observed > result.certified_l1_bound + 1.0e-12 {
                violations += 1;
            }
            max_observed = max_observed.max(observed);
            max_bound = max_bound.max(result.certified_l1_bound);
            #[allow(clippy::cast_precision_loss)]
            {
                skip_sum += cost.blocks_skipped as f64 / cost.blocks_total as f64;
                first_skip_sum += first_cost.blocks_skipped as f64 / first_cost.blocks_total as f64;
                row_frac_sum += cost.query_time_key_rows_read as f64 / key_count as f64;
            }
            assert_certificate_holds(&scores, &result);
            assert_certificate_holds(&scores, &first);
        }
        assert_eq!(violations, 0, "{label} had certificate violations");
        #[allow(clippy::cast_precision_loss)]
        let samples_f = f64::from(u32::try_from(samples).expect("sample count fits u32"));
        let mean_skip = skip_sum / samples_f;
        let mean_first = first_skip_sum / samples_f;
        let mean_rows = row_frac_sum / samples_f;
        eprintln!(
            "{label} n={key_count} d={head_dim} samples={samples} block={block_size} eps={epsilon}: mean_skip_frac={mean_skip:.6} first_order_mean_skip={mean_first:.6} mean_row_frac={mean_rows:.6} max_L1={max_observed:.6e} max_bound={max_bound:.6e}"
        );
    }

    #[test]
    fn gaussian_and_clustered_survey_never_exceeds_certificate() {
        let samples = if cfg!(miri) { 3 } else { 80 };
        let key_count = if cfg!(miri) { 8 } else { 32 };
        let head_dim = if cfg!(miri) { 2 } else { 4 };
        if cfg!(miri) {
            survey_regime("iid_keys", samples, key_count, head_dim, false, 1.0e-2);
            survey_regime("clustered_keys", samples, key_count, head_dim, true, 1.0e-2);
            return;
        }
        for epsilon in [1.0e-4_f64, 1.0e-2, 1.0e-1] {
            survey_regime("iid_keys", samples, key_count, head_dim, false, epsilon);
            survey_regime(
                "clustered_keys",
                samples,
                key_count,
                head_dim,
                true,
                epsilon,
            );
        }
    }
}
