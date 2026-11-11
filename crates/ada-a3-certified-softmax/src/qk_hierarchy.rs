//! Hierarchical QK-box certified block-sparse softmax.
//!
//! Cycle #69 certifies an L1 omission bound from depth-1 half-boxes and
//! then skips *whole blocks*. A mixed block whose warm half reaches `M_lo`
//! can never be omitted, so every key of that block is materialized even
//! when the other half is far colder than the budget would allow.
//!
//! This module keeps the A4 coordinate-box arithmetic and the same
//! greedy L1 certificate, but:
//!
//! - recursively splits each contiguous block `hierarchy_depth` times
//!   (depth 0 = one leaf per block; depth 1 = the two halves of #69;
//!   depth `D` yields at most `2^D` leaves);
//! - treats each leaf as an independent skip unit:
//!   `U_ℓ = n_ℓ exp(hi_ℓ − M_lo)` and
//!   `L_ℓ = exp(w_ℓ − M_hi) + (n_ℓ − 1) exp(lo_ℓ − M_hi)`;
//! - never skips a leaf whose box-max reaches `M_lo` (it might hold a
//!   top-plateau token).
//!
//! `M_lo` is the maximum of one witness score per leaf (the first key of
//! the leaf) and `M_hi` is the maximum leaf-box `hi`, so
//! `M_lo ≤ M ≤ M_hi`. Skipping a set `K` of leaves and renormalizing the
//! complement yields `||p − q||_1 = 2 S_K / S ≤ 2 (Σ_{ℓ ∈ K} U_ℓ) / L`
//! with `L = Σ_ℓ L_ℓ`. Finer leaves have boxes contained in their parent,
//! so their `hi` cannot exceed the parent's; more witnesses can only
//! raise `M_lo`. The leaf-level bound is therefore never looser than the
//! parent block bound of #69 on the same omitted region.
//!
//! A skipped leaf contributes only its witness at query time. Logical
//! counters are not hardware measurements. No usefulness, novelty, or
//! FLAT-adoption claim is made.

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QkHierarchyCost {
    /// Contiguous top-level blocks in the partition.
    pub blocks_total: usize,
    /// Leaves produced by the requested hierarchy depth.
    pub leaves_total: usize,
    /// Leaves assigned probability zero by the skip rule.
    pub leaves_skipped: usize,
    /// Top-level blocks whose every leaf was skipped.
    pub blocks_fully_skipped: usize,
    /// Leaf-boxes evaluated against the query (`hi` and `lo` share one pass).
    pub box_bound_evaluations: usize,
    /// Key rows dotted to form per-leaf witnesses (one per leaf).
    pub witness_dots: usize,
    /// Key rows dotted in the second pass (every token of every kept leaf).
    pub key_dots_materialized: usize,
    /// Distinct key rows read at query time (skipped-leaf witnesses plus kept rows).
    pub query_time_key_rows_read: usize,
    /// Distinct `exp` evaluations used for `(U_ℓ, L_ℓ)` plus kept tokens.
    pub exp_evaluations: usize,
    /// Greedy certificate checks (one per leaf that is not possibly top).
    pub bound_checks: usize,
    /// Tokens that received a positive unnormalized `exp` in the second pass.
    pub tokens_exponentiated: usize,
    /// Requested number of binary splits inside each block.
    pub hierarchy_depth: usize,
}

/// Certified hierarchical QK-box softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct QkHierarchySoftmax {
    /// Renormalized probabilities in input order. Skipped tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate `2 * sum_{skipped} U_ℓ / L`.
    pub certified_l1_bound: f64,
    /// Caller budget that the certificate was required to respect.
    pub epsilon: f64,
    /// Requested top-level block size (the last block may be shorter).
    pub block_size: usize,
    /// Requested number of binary splits inside each block.
    pub hierarchy_depth: usize,
    /// Whether each leaf was skipped, in left-to-right leaf order.
    pub leaf_skipped: Vec<bool>,
    /// Half-open `[start, end)` token range of each leaf, matching `leaf_skipped`.
    pub leaf_ranges: Vec<(usize, usize)>,
    /// Positive scale applied to every `q · k` score.
    pub score_scale: f64,
}

#[derive(Clone, Debug)]
struct CoordBox {
    minimum: Vec<f64>,
    maximum: Vec<f64>,
}

#[derive(Clone, Debug)]
struct LeafStat {
    start: usize,
    end: usize,
    hi: f64,
    lo: f64,
    witness: f64,
    upper: f64,
    lower: f64,
    block_index: usize,
}

fn validate_args(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    hierarchy_depth: usize,
    epsilon: f64,
) -> Result<usize, &'static str> {
    if head_dim == 0 {
        return Err("ADA-A3 hierarchical QK-box softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 hierarchical QK-box softmax query must contain head_dim elements");
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A3 hierarchical QK-box softmax query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 hierarchical QK-box softmax requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err(
            "ADA-A3 hierarchical QK-box softmax keys must be row-major [key_count, head_dim]",
        );
    }
    if keys.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A3 hierarchical QK-box softmax keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 hierarchical QK-box softmax score_scale must be finite and positive");
    }
    if block_size == 0 {
        return Err("ADA-A3 hierarchical QK-box softmax block_size must be positive");
    }
    if hierarchy_depth > 8 {
        return Err("ADA-A3 hierarchical QK-box softmax hierarchy_depth must be at most 8");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 hierarchical QK-box softmax epsilon must be finite and non-negative");
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
            return Err("ADA-A3 hierarchical QK-box softmax box coordinates must be ordered");
        }
        let q_min = q * minimum;
        let q_max = q * maximum;
        hi_sum += q_min.max(q_max);
        lo_sum += q_min.min(q_max);
    }
    let hi = hi_sum * score_scale;
    let lo = lo_sum * score_scale;
    if !hi.is_finite() || !lo.is_finite() {
        return Err("ADA-A3 hierarchical QK-box softmax box bound overflowed");
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
        return Err("ADA-A3 hierarchical QK-box softmax witness or kept score overflowed");
    }
    Ok(score)
}

fn split_range(start: usize, end: usize, remaining_depth: usize) -> Vec<(usize, usize)> {
    if remaining_depth == 0 || end - start <= 1 {
        return vec![(start, end)];
    }
    let mid = start + (end - start) / 2;
    let mut ranges = split_range(start, mid, remaining_depth - 1);
    ranges.extend(split_range(mid, end, remaining_depth - 1));
    ranges
}

fn evaluate_leaf(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    start: usize,
    end: usize,
    block_index: usize,
) -> Result<LeafStat, &'static str> {
    let page = build_coord_box(keys, head_dim, start, end);
    let (hi, lo) = box_bounds(query, &page, score_scale)?;
    let witness = dot_score(query, keys, head_dim, start, score_scale)?;
    // Absorb rounding so a witness cannot sit outside its own box.
    let hi = hi.max(witness);
    let lo = lo.min(witness);
    Ok(LeafStat {
        start,
        end,
        hi,
        lo,
        witness,
        upper: 0.0,
        lower: 0.0,
        block_index,
    })
}

fn partition_leaves(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    key_count: usize,
    score_scale: f64,
    block_size: usize,
    hierarchy_depth: usize,
) -> Result<(Vec<LeafStat>, usize), &'static str> {
    let mut leaves = Vec::new();
    let mut block_index = 0_usize;
    let mut offset = 0;
    while offset < key_count {
        let end = (offset + block_size).min(key_count);
        for (leaf_start, leaf_end) in split_range(offset, end, hierarchy_depth) {
            leaves.push(evaluate_leaf(
                query,
                keys,
                head_dim,
                score_scale,
                leaf_start,
                leaf_end,
                block_index,
            )?);
        }
        block_index = block_index.saturating_add(1);
        offset = end;
    }
    Ok((leaves, block_index))
}

fn shifted_exp(score: f64, shift: f64) -> Result<f64, &'static str> {
    let term = (score - shift).exp();
    if !term.is_finite() || term < 0.0 {
        return Err("ADA-A3 hierarchical QK-box softmax bound term must be finite");
    }
    Ok(term)
}

fn annotate_leaf_bounds(leaves: &mut [LeafStat]) -> Result<(f64, f64, f64, usize), &'static str> {
    let max_lo = leaves
        .iter()
        .map(|leaf| leaf.witness)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_hi = leaves
        .iter()
        .map(|leaf| leaf.hi)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lo.is_finite() || !max_hi.is_finite() {
        return Err("ADA-A3 hierarchical QK-box softmax global score bounds must be finite");
    }
    let mut mass_lower = 0.0_f64;
    let mut exp_evaluations = 0_usize;
    for leaf in leaves.iter_mut() {
        let exp_hi = shifted_exp(leaf.hi, max_lo)?;
        exp_evaluations = exp_evaluations.saturating_add(1);
        let exp_lo = shifted_exp(leaf.lo, max_hi)?;
        exp_evaluations = exp_evaluations.saturating_add(1);
        let exp_witness = if leaf.witness.total_cmp(&leaf.lo).is_eq() {
            exp_lo
        } else {
            exp_evaluations = exp_evaluations.saturating_add(1);
            shifted_exp(leaf.witness, max_hi)?
        };
        let length = leaf.end - leaf.start;
        #[allow(clippy::cast_precision_loss)]
        let rest = length.saturating_sub(1) as f64;
        #[allow(clippy::cast_precision_loss)]
        let length_f = length as f64;
        let upper = length_f * exp_hi;
        let lower = exp_witness + rest * exp_lo;
        if !upper.is_finite() || upper < 0.0 {
            return Err("ADA-A3 hierarchical QK-box softmax leaf upper bound must be finite");
        }
        leaf.upper = upper;
        leaf.lower = lower;
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

fn plan_skips(leaves: &[LeafStat], max_lo: f64, mass_lower: f64, epsilon: f64) -> SkipPlan {
    let mut order: Vec<usize> = (0..leaves.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        leaves[left]
            .upper
            .total_cmp(&leaves[right].upper)
            .then_with(|| left.cmp(&right))
    });

    let mut omitted = vec![false; leaves.len()];
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
        if leaves[index].hi >= max_lo {
            continue;
        }
        bound_checks = bound_checks.saturating_add(1);
        let next = omitted_upper + leaves[index].upper;
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
    leaves: &[LeafStat],
    omitted: &[bool],
) -> Result<(Vec<f64>, usize), &'static str> {
    let mut scores = vec![0.0_f64; key_count];
    let mut tokens_exponentiated = 0_usize;
    let mut kept_shift = f64::NEG_INFINITY;
    for (leaf, skipped) in leaves.iter().zip(omitted.iter().copied()) {
        if skipped {
            continue;
        }
        for (row, slot) in scores
            .iter_mut()
            .enumerate()
            .take(leaf.end)
            .skip(leaf.start)
        {
            *slot = dot_score(query, keys, head_dim, row, score_scale)?;
            if *slot > kept_shift {
                kept_shift = *slot;
            }
        }
    }
    if !kept_shift.is_finite() {
        return Err("ADA-A3 hierarchical QK-box softmax kept scores must be finite");
    }
    let mut kept_mass = 0.0_f64;
    let mut probabilities = vec![0.0_f64; key_count];
    for (leaf, skipped) in leaves.iter().zip(omitted.iter().copied()) {
        if skipped {
            continue;
        }
        for (score, slot) in scores[leaf.start..leaf.end]
            .iter()
            .zip(probabilities[leaf.start..leaf.end].iter_mut())
        {
            let term = (*score - kept_shift).exp();
            tokens_exponentiated = tokens_exponentiated.saturating_add(1);
            if !term.is_finite() || term < 0.0 {
                return Err("ADA-A3 hierarchical QK-box softmax kept term must be finite");
            }
            *slot = term;
            kept_mass += term;
        }
    }
    if !kept_mass.is_finite() || kept_mass <= 0.0 {
        return Err("ADA-A3 hierarchical QK-box softmax kept mass must be positive");
    }
    for probability in &mut probabilities {
        *probability /= kept_mass;
    }
    Ok((probabilities, tokens_exponentiated))
}

fn query_time_rows_read(leaves: &[LeafStat], omitted: &[bool]) -> (usize, usize) {
    let mut read = 0_usize;
    let mut materialized = 0_usize;
    for (leaf, skipped) in leaves.iter().zip(omitted.iter().copied()) {
        if skipped {
            read = read.saturating_add(1);
            continue;
        }
        let length = leaf.end - leaf.start;
        materialized = materialized.saturating_add(length);
        read = read.saturating_add(length);
    }
    (read, materialized)
}

fn blocks_fully_skipped(leaves: &[LeafStat], omitted: &[bool], blocks_total: usize) -> usize {
    let mut skipped_in_block = vec![0_usize; blocks_total];
    let mut total_in_block = vec![0_usize; blocks_total];
    for (leaf, skipped) in leaves.iter().zip(omitted.iter().copied()) {
        total_in_block[leaf.block_index] = total_in_block[leaf.block_index].saturating_add(1);
        if skipped {
            skipped_in_block[leaf.block_index] =
                skipped_in_block[leaf.block_index].saturating_add(1);
        }
    }
    skipped_in_block
        .iter()
        .zip(total_in_block.iter())
        .filter(|&(&skipped, &total)| total > 0 && skipped == total)
        .count()
}

/// Evaluate hierarchical QK-box certified block-sparse softmax and record counters.
///
/// `hierarchy_depth` is the number of binary splits inside each top-level
/// block of length at most `block_size` (0 = one leaf per block).
///
/// # Errors
///
/// Returns an error when the Q/K layout is empty or inconsistent, when any
/// input is non-finite, when `score_scale` is not finite and positive, when
/// `block_size` is zero, when `hierarchy_depth` exceeds 8, when `epsilon`
/// is non-finite or negative, or when a mass residual is non-finite.
#[must_use = "the certified distribution should be checked"]
pub fn certified_qk_box_hierarchical_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    hierarchy_depth: usize,
    epsilon: f64,
) -> Result<(QkHierarchySoftmax, QkHierarchyCost), &'static str> {
    let key_count = validate_args(
        query,
        keys,
        head_dim,
        score_scale,
        block_size,
        hierarchy_depth,
        epsilon,
    )?;
    let (mut leaves, blocks_total) = partition_leaves(
        query,
        keys,
        head_dim,
        key_count,
        score_scale,
        block_size,
        hierarchy_depth,
    )?;
    let leaf_count = leaves.len();
    let (max_lo, _max_hi, mass_lower, bound_exps) = annotate_leaf_bounds(&mut leaves)?;
    let plan = plan_skips(&leaves, max_lo, mass_lower, epsilon);
    let certified_l1_bound = if plan.omitted_count == 0 {
        0.0
    } else {
        2.0 * plan.omitted_upper / mass_lower
    };
    if !certified_l1_bound.is_finite() || certified_l1_bound > epsilon {
        return Err("ADA-A3 hierarchical QK-box softmax certificate exceeds the supplied budget");
    }
    let (probabilities, tokens_exponentiated) = renormalize_kept(
        query,
        keys,
        head_dim,
        score_scale,
        key_count,
        &leaves,
        &plan.omitted,
    )?;
    let (query_time_key_rows_read, key_dots_materialized) =
        query_time_rows_read(&leaves, &plan.omitted);
    let cost = QkHierarchyCost {
        blocks_total,
        leaves_total: leaf_count,
        leaves_skipped: plan.omitted_count,
        blocks_fully_skipped: blocks_fully_skipped(&leaves, &plan.omitted, blocks_total),
        box_bound_evaluations: leaf_count,
        witness_dots: leaf_count,
        key_dots_materialized,
        query_time_key_rows_read,
        exp_evaluations: bound_exps.saturating_add(tokens_exponentiated),
        bound_checks: plan.bound_checks,
        tokens_exponentiated,
        hierarchy_depth,
    };
    let leaf_ranges = leaves.iter().map(|leaf| (leaf.start, leaf.end)).collect();
    let result = QkHierarchySoftmax {
        probabilities,
        certified_l1_bound,
        epsilon,
        block_size,
        hierarchy_depth,
        leaf_skipped: plan.omitted,
        leaf_ranges,
        score_scale,
    };
    Ok((result, cost))
}

/// Evaluate hierarchical QK-box certified block-sparse softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_qk_box_hierarchical_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_qk_box_hierarchical_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    block_size: usize,
    hierarchy_depth: usize,
    epsilon: f64,
) -> Result<QkHierarchySoftmax, &'static str> {
    certified_qk_box_hierarchical_softmax_accounted(
        query,
        keys,
        head_dim,
        score_scale,
        block_size,
        hierarchy_depth,
        epsilon,
    )
    .map(|(result, _)| result)
}

#[cfg(test)]
mod tests {
    use super::{
        QkHierarchySoftmax, certified_qk_box_hierarchical_softmax,
        certified_qk_box_hierarchical_softmax_accounted,
    };
    use crate::{
        certified_qk_box_first_order_softmax_accounted,
        certified_qk_box_two_stat_softmax_accounted, dense_qk_scores,
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

    fn assert_certificate_holds(scores: &[f64], result: &QkHierarchySoftmax) {
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
        for (skipped, &(start, end)) in result.leaf_skipped.iter().zip(result.leaf_ranges.iter()) {
            if *skipped {
                assert!(
                    result.probabilities[start..end]
                        .iter()
                        .all(|p| p.abs() <= 0.0)
                );
                assert!(
                    scores[start..end].iter().all(|&score| score < true_max),
                    "skipped a top-plateau token in leaf {start}..{end}"
                );
            }
        }
    }

    fn run_h(
        query: &[f64],
        keys: &[f64],
        head_dim: usize,
        score_scale: f64,
        block_size: usize,
        depth: usize,
        epsilon: f64,
    ) -> (QkHierarchySoftmax, super::QkHierarchyCost, Vec<f64>) {
        let (result, cost) = certified_qk_box_hierarchical_softmax_accounted(
            query,
            keys,
            head_dim,
            score_scale,
            block_size,
            depth,
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
        assert!(certified_qk_box_hierarchical_softmax(&[], &k, 1, 1.0, 1, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &[], 1, 1.0, 1, 1, 1.0e-3).is_err());
        assert!(
            certified_qk_box_hierarchical_softmax(&q, &[1.0, 0.0, 1.0], 2, 1.0, 1, 1, 1.0e-3)
                .is_err()
        );
        assert!(
            certified_qk_box_hierarchical_softmax(&[f64::NAN], &k, 1, 1.0, 1, 1, 1.0e-3).is_err()
        );
        assert!(
            certified_qk_box_hierarchical_softmax(&q, &[1.0, f64::INFINITY], 1, 1.0, 1, 1, 1.0e-3)
                .is_err()
        );
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 0, 1.0, 1, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 1, 0.0, 1, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 1, -1.0, 1, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 1, 1.0, 0, 1, 1.0e-3).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 1, 1.0, 1, 9, 1.0e-3).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 1, 1.0, 1, 1, -0.1).is_err());
        assert!(certified_qk_box_hierarchical_softmax(&q, &k, 1, 1.0, 1, 1, f64::NAN).is_err());
    }

    #[test]
    fn singleton_and_all_equal_never_skip() {
        let query = [1.0_f64, 0.0];
        let one_key = [2.0_f64, -3.0];
        let (one, cost, _) = run_h(&query, &one_key, 2, 1.0, 4, 2, 1.0e-6);
        assert_eq!(one.probabilities, vec![1.0]);
        assert_eq!(one.leaf_skipped, vec![false]);
        assert!(one.certified_l1_bound.abs() <= f64::EPSILON);
        assert_eq!(cost.witness_dots, 1);
        assert_eq!(cost.key_dots_materialized, 1);

        let keys = [1.0_f64, 0.0].repeat(8);
        let (equal, cost, _) = run_h(&query, &keys, 2, 1.0, 4, 2, 1.0e-4);
        assert!(equal.leaf_skipped.iter().all(|skipped| !skipped));
        assert!(equal.certified_l1_bound.abs() <= f64::EPSILON);
        for probability in &equal.probabilities {
            assert!((probability - 0.125).abs() <= 1.0e-12);
        }
        assert_eq!(cost.key_dots_materialized, 8);
        assert_eq!(cost.query_time_key_rows_read, 8);
    }

    #[test]
    fn leaf_skip_omits_the_cold_half_of_a_mixed_block() {
        // Block 0: four copies of the query direction (unique top plateau).
        // Block 1: two equally aligned keys, then two strongly anti-aligned
        // keys. Cycle #69 cannot skip block 1 because its warm half reaches
        // M_lo. Depth-1 leaf skip can omit only the cold half.
        let query = [1.0_f64, 0.0];
        let mut keys = Vec::new();
        for _ in 0..4 {
            keys.extend_from_slice(&[3.0, 0.0]);
        }
        keys.extend_from_slice(&[3.0, 0.0, 3.0, 0.0, -20.0, 0.0, -20.0, 0.0]);
        let epsilon = 0.2_f64;
        let (two, two_cost) =
            certified_qk_box_two_stat_softmax_accounted(&query, &keys, 2, 1.0, 4, epsilon).unwrap();
        let (hier, hier_cost, scores) = run_h(&query, &keys, 2, 1.0, 4, 1, epsilon);
        assert_eq!(two_cost.blocks_skipped, 0, "mixed warm block must be kept");
        assert!(!two.block_skipped[1]);
        assert_eq!(hier_cost.leaves_total, 4);
        assert_eq!(hier_cost.leaves_skipped, 1);
        assert!(!hier.leaf_skipped[0]);
        assert!(!hier.leaf_skipped[1]);
        assert!(!hier.leaf_skipped[2]);
        assert!(hier.leaf_skipped[3]);
        assert_eq!(hier.leaf_ranges[3], (6, 8));
        assert!(hier_cost.query_time_key_rows_read < scores.len());
        assert!(hier_cost.query_time_key_rows_read < two_cost.query_time_key_rows_read);
        assert!(hier.probabilities[6].abs() <= 0.0);
        assert!(hier.probabilities[7].abs() <= 0.0);
        assert!(hier.certified_l1_bound <= epsilon);
    }

    #[test]
    fn pair_interleaved_cold_leaves_are_skipped_at_depth_two() {
        // Eight keys alternating aligned pairs and anti-aligned pairs.
        // Depth 1 produces two mixed halves and cannot skip; depth 2
        // isolates the cold pairs.
        let query = [1.0_f64, 0.0];
        let pattern = [
            4.0, 0.0, 4.0, 0.0, -20.0, 0.0, -20.0, 0.0, 4.0, 0.0, 4.0, 0.0, -20.0, 0.0, -20.0, 0.0,
        ];
        let (shallow, shallow_cost, _) = run_h(&query, &pattern, 2, 1.0, 8, 1, 0.2);
        let (deep, deep_cost, scores) = run_h(&query, &pattern, 2, 1.0, 8, 2, 0.2);
        assert_eq!(shallow_cost.leaves_skipped, 0);
        assert!(shallow.leaf_skipped.iter().all(|skipped| !skipped));
        assert_eq!(deep_cost.leaves_skipped, 2);
        assert!(!deep.leaf_skipped[0]);
        assert!(deep.leaf_skipped[1]);
        assert!(!deep.leaf_skipped[2]);
        assert!(deep.leaf_skipped[3]);
        assert!(deep_cost.query_time_key_rows_read < scores.len());
        assert!(deep_cost.query_time_key_rows_read < shallow_cost.query_time_key_rows_read);
        let (first, first_cost) =
            certified_qk_box_first_order_softmax_accounted(&query, &pattern, 2, 1.0, 8, 0.2)
                .unwrap();
        assert_eq!(first_cost.blocks_skipped, 0);
        let _ = first;
    }

    #[test]
    fn unique_needle_hit_at_small_and_large_margins() {
        let query = [1.0_f64, 0.0];
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut keys = vec![0.0_f64; 16];
            keys[6] = margin;
            let (result, _, _) = run_h(&query, &keys, 2, 1.0, 4, 2, 1.0e-4);
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
            let (result, _, _) = run_h(&query, &keys, 2, 1.0, 4, 2, 1.0e-4);
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
        let (result, cost, scores) = run_h(&query, &keys, 2, 1.0, 2, 2, 0.0);
        assert_eq!(cost.leaves_skipped, 0);
        assert!(result.certified_l1_bound.abs() <= f64::EPSILON);
        assert_eq!(cost.key_dots_materialized, scores.len());
        let observed = l1_distance(&result.probabilities, &oracle_softmax(&scores));
        assert!(observed <= 1.0e-14, "dense path L1 {observed}");
    }

    #[test]
    fn last_block_may_be_short() {
        let query = [1.0_f64];
        let keys = [0.0_f64, -4.0, -4.0, -4.0, 2.0];
        let (result, _, _) = run_h(&query, &keys, 1, 1.0, 3, 2, 1.0e-3);
        assert!(result.leaf_ranges.len() >= 2);
        let last = result.leaf_ranges.len() - 1;
        assert_eq!(result.leaf_ranges[last].1, 5);
        assert!(!result.leaf_skipped[last]);
    }

    fn grid_domain() -> (Vec<[f64; 2]>, usize) {
        if cfg!(miri) {
            (vec![[0.0, 0.0], [1.0, 0.0]], 2)
        } else {
            (vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]], 4)
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
        let (domain, max_len) = grid_domain();
        let mut vectors = 0_usize;
        let mut loose_skips = 0_usize;
        let mut two_stat_kept_but_hier_saved = 0_usize;
        let mut saved_row_cases = 0_usize;
        for length in 1..=max_len {
            for keys in enumerate_key_layouts(length, &domain) {
                for block_size in 1..=length {
                    let depth = if cfg!(miri) { 1 } else { 2 };
                    let (tight, _, scores) =
                        run_h(&query, &keys, 2, 1.0, block_size, depth, 1.0e-3);
                    let (loose, loose_cost, _) =
                        run_h(&query, &keys, 2, 1.0, block_size, depth, 5.0e-2);
                    let _ = tight;
                    vectors += 1;
                    if loose.leaf_skipped.iter().any(|skipped| *skipped) {
                        loose_skips += 1;
                    }
                    if loose_cost.query_time_key_rows_read < scores.len() {
                        saved_row_cases += 1;
                    }
                    let (two_loose, two_cost) = certified_qk_box_two_stat_softmax_accounted(
                        &query, &keys, 2, 1.0, block_size, 5.0e-2,
                    )
                    .unwrap();
                    let _ = two_loose;
                    if two_cost.query_time_key_rows_read == scores.len()
                        && loose_cost.query_time_key_rows_read < scores.len()
                    {
                        two_stat_kept_but_hier_saved += 1;
                    }
                }
            }
        }
        assert!(vectors > 0);
        eprintln!(
            "qk_hierarchy exhaustive_grid vectors={vectors} loose_eps=5e-2 leaf_skip_cases={loose_skips} row_saving_loose_cases={saved_row_cases} two_stat_no_save_but_hier_saved={two_stat_kept_but_hier_saved}"
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

    #[derive(Clone, Copy)]
    enum Regime {
        Iid,
        Clustered,
        Moderate,
        PairInterleaved,
    }

    fn rewrite_keys(keys: &mut [f64], head_dim: usize, block_size: usize, regime: Regime) {
        let key_count = keys.len() / head_dim;
        match regime {
            Regime::Iid => {}
            Regime::Clustered | Regime::Moderate => {
                let polarity_hot = if matches!(regime, Regime::Clustered) {
                    6.0_f64
                } else {
                    2.5_f64
                };
                let jitter = if matches!(regime, Regime::Clustered) {
                    0.05_f64
                } else {
                    0.3_f64
                };
                let mut offset = 0;
                let mut block_index = 0_usize;
                while offset < key_count {
                    let end = (offset + block_size).min(key_count);
                    let polarity = if block_index == 0 {
                        polarity_hot
                    } else {
                        -polarity_hot
                    };
                    for row in offset..end {
                        keys[row * head_dim] = polarity + jitter * keys[row * head_dim];
                        for dim in 1..head_dim {
                            keys[row * head_dim + dim] *= jitter;
                        }
                    }
                    block_index += 1;
                    offset = end;
                }
            }
            Regime::PairInterleaved => {
                for row in 0..key_count {
                    let pair = row / 2;
                    let polarity = if pair % 2 == 0 { 4.0_f64 } else { -8.0_f64 };
                    keys[row * head_dim] = polarity + 0.05 * keys[row * head_dim];
                    for dim in 1..head_dim {
                        keys[row * head_dim + dim] *= 0.05;
                    }
                }
            }
        }
    }

    fn survey_regime(
        label: &str,
        samples: usize,
        key_count: usize,
        head_dim: usize,
        regime: Regime,
        depth: usize,
        epsilon: f64,
    ) {
        let mut state = 0xA3B5_C0DE_2026_1110;
        let mut violations = 0_usize;
        let mut leaf_skip_sum = 0.0_f64;
        let mut two_skip_sum = 0.0_f64;
        let mut row_frac_sum = 0.0_f64;
        let mut two_row_frac_sum = 0.0_f64;
        let mut max_observed = 0.0_f64;
        let mut max_bound = 0.0_f64;
        let block_size = if key_count >= 8 { 8 } else { 2 };
        let mut query = vec![0.0; head_dim];
        query[0] = 1.0;
        for _ in 0..samples {
            let mut keys = vec![0.0; key_count * head_dim];
            fill_unit_gaussians(&mut keys, &mut state);
            rewrite_keys(&mut keys, head_dim, block_size, regime);
            let (result, cost) = certified_qk_box_hierarchical_softmax_accounted(
                &query, &keys, head_dim, 1.0, block_size, depth, epsilon,
            )
            .unwrap();
            let (two, two_cost) = certified_qk_box_two_stat_softmax_accounted(
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
                leaf_skip_sum += cost.leaves_skipped as f64 / cost.leaves_total as f64;
                two_skip_sum += two_cost.blocks_skipped as f64 / two_cost.blocks_total as f64;
                row_frac_sum += cost.query_time_key_rows_read as f64 / key_count as f64;
                two_row_frac_sum += two_cost.query_time_key_rows_read as f64 / key_count as f64;
            }
            assert_certificate_holds(&scores, &result);
            let _ = two;
        }
        assert_eq!(violations, 0, "{label} had certificate violations");
        #[allow(clippy::cast_precision_loss)]
        let samples_f = f64::from(u32::try_from(samples).expect("sample count fits u32"));
        let mean_leaf = leaf_skip_sum / samples_f;
        let mean_two = two_skip_sum / samples_f;
        let mean_rows = row_frac_sum / samples_f;
        let mean_two_rows = two_row_frac_sum / samples_f;
        eprintln!(
            "{label} n={key_count} d={head_dim} samples={samples} block={block_size} depth={depth} eps={epsilon}: mean_leaf_skip={mean_leaf:.6} two_stat_mean_block_skip={mean_two:.6} mean_row_frac={mean_rows:.6} two_stat_mean_row_frac={mean_two_rows:.6} max_L1={max_observed:.6e} max_bound={max_bound:.6e}"
        );
    }

    #[test]
    fn gaussian_regimes_never_exceed_certificate() {
        let samples = if cfg!(miri) { 2 } else { 80 };
        let key_count = if cfg!(miri) { 8 } else { 32 };
        let head_dim = if cfg!(miri) { 2 } else { 4 };
        if cfg!(miri) {
            survey_regime(
                "iid_keys",
                samples,
                key_count,
                head_dim,
                Regime::Iid,
                2,
                1.0e-2,
            );
            survey_regime(
                "pair_interleaved",
                samples,
                key_count,
                head_dim,
                Regime::PairInterleaved,
                2,
                1.0e-2,
            );
            return;
        }
        for epsilon in [1.0e-4_f64, 1.0e-2, 1.0e-1] {
            for depth in [1_usize, 2] {
                survey_regime(
                    "iid_keys",
                    samples,
                    key_count,
                    head_dim,
                    Regime::Iid,
                    depth,
                    epsilon,
                );
                survey_regime(
                    "clustered_keys",
                    samples,
                    key_count,
                    head_dim,
                    Regime::Clustered,
                    depth,
                    epsilon,
                );
                survey_regime(
                    "moderate_cluster",
                    samples,
                    key_count,
                    head_dim,
                    Regime::Moderate,
                    depth,
                    epsilon,
                );
                survey_regime(
                    "pair_interleaved",
                    samples,
                    key_count,
                    head_dim,
                    Regime::PairInterleaved,
                    depth,
                    epsilon,
                );
            }
        }
    }
}
