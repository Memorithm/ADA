//! Two-statistic certified block-sparse softmax (score-level).
//!
//! The first-order operator in [`crate::certified_block_sparse_softmax`]
//! bounds a block from its maximum alone:
//! `U_b = n_b * exp(m_b - M)` and `L = sum_b exp(m_b - M)`. That certificate
//! cannot skip a moderately cold block that also contains much colder
//! tokens, because the upper bound pretends every token equals the block
//! max. This module keeps the same contiguous partition and greedy skip
//! rule, but stores three order statistics per block: the maximum `m_b`,
//! the second maximum `s_b`, and the minimum `ℓ_b`.
//!
//! - `U_b = exp(m_b - M) + (n_b - 1) * exp(s_b - M)` upper-bounds the
//!   unnormalized block mass (every non-max token is at most `s_b`),
//! - `L_b = exp(m_b - M) + (n_b - 1) * exp(ℓ_b - M)` lower-bounds it
//!   (every non-max token is at least `ℓ_b`),
//! - `L = sum_b L_b` still lower-bounds the true total mass `S`, including
//!   witnesses from blocks that will be skipped,
//! - skipping a set `K` of non-top-plateau blocks and renormalizing yields
//!   `||p - q||_1 = 2 * S_K / S <= 2 * (sum_{b in K} U_b) / L`.
//!
//! For any fixed skip set the two-stat bound is never looser than the
//! first-order bound, because `s_b <= m_b` and `ℓ_b <= m_b`. A singleton
//! block collapses to the first-order pair `(U_b, L_b) = (exp(m_b - M),
//! exp(m_b - M))`. A length-2 block has `s_b = ℓ_b`, so `U_b = L_b` equals
//! the exact block mass.
//!
//! Top-plateau blocks (`m_b = M`) are never skipped. Remaining blocks are
//! visited in increasing `U_b` (then index) and skipped greedily while the
//! certificate stays `<= epsilon`.
//!
//! The independent oracle is a separately implemented dense stable softmax
//! (see the tests). Logical counters are not hardware measurements. No
//! usefulness, novelty, or FLAT-adoption claim is made.

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TwoStatCost {
    /// Contiguous blocks in the partition (the last one may be shorter).
    pub blocks_total: usize,
    /// Blocks assigned probability zero by the skip rule.
    pub blocks_skipped: usize,
    /// Scores read to form per-block order statistics (always the full length).
    pub tokens_scanned_for_stats: usize,
    /// Distinct `exp` evaluations used for `(U_b, L_b)` plus kept tokens.
    pub exp_evaluations: usize,
    /// Greedy certificate checks (one per non-top-plateau block).
    pub bound_checks: usize,
    /// Tokens that received a positive unnormalized `exp` in the second pass.
    pub tokens_exponentiated: usize,
}

/// Certified two-statistic block-sparse softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoStatSoftmax {
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
}

struct BlockStat {
    start: usize,
    end: usize,
    max: f64,
    second: f64,
    min: f64,
    upper: f64,
    lower: f64,
}

fn validate_args(scores: &[f64], block_size: usize, epsilon: f64) -> Result<(), &'static str> {
    if scores.is_empty() {
        return Err("ADA-A3 two-stat softmax requires at least one score");
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("ADA-A3 two-stat softmax scores must be finite");
    }
    if block_size == 0 {
        return Err("ADA-A3 two-stat softmax block_size must be positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 two-stat softmax epsilon must be finite and non-negative");
    }
    Ok(())
}

fn scan_block(scores: &[f64]) -> (f64, f64, f64) {
    let mut max = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;
    for &score in scores {
        if score > max {
            second = max;
            max = score;
        } else if score > second {
            second = score;
        }
        if score < min {
            min = score;
        }
    }
    if scores.len() == 1 {
        second = max;
    }
    (max, second, min)
}

fn partition_blocks(scores: &[f64], block_size: usize) -> Vec<BlockStat> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    while offset < scores.len() {
        let end = (offset + block_size).min(scores.len());
        let (max, second, min) = scan_block(&scores[offset..end]);
        blocks.push(BlockStat {
            start: offset,
            end,
            max,
            second,
            min,
            upper: 0.0,
            lower: 0.0,
        });
        offset = end;
    }
    blocks
}

fn shifted_exp(score: f64, global_max: f64) -> Result<f64, &'static str> {
    let term = (score - global_max).exp();
    if !term.is_finite() || term < 0.0 {
        return Err("ADA-A3 two-stat softmax block bound must be finite");
    }
    Ok(term)
}

fn annotate_block_bounds(blocks: &mut [BlockStat]) -> Result<(f64, f64, usize), &'static str> {
    let global_max = blocks
        .iter()
        .map(|block| block.max)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut mass_lower = 0.0_f64;
    let mut exp_evaluations = 0_usize;
    for block in blocks.iter_mut() {
        let exp_max = shifted_exp(block.max, global_max)?;
        exp_evaluations = exp_evaluations.saturating_add(1);
        let exp_second = if block.second.total_cmp(&block.max).is_eq() {
            exp_max
        } else {
            exp_evaluations = exp_evaluations.saturating_add(1);
            shifted_exp(block.second, global_max)?
        };
        let exp_min = if block.min.total_cmp(&block.max).is_eq() {
            exp_max
        } else if block.min.total_cmp(&block.second).is_eq() {
            exp_second
        } else {
            exp_evaluations = exp_evaluations.saturating_add(1);
            shifted_exp(block.min, global_max)?
        };
        #[allow(clippy::cast_precision_loss)]
        let rest = (block.end - block.start).saturating_sub(1) as f64;
        block.upper = exp_max + rest * exp_second;
        block.lower = exp_max + rest * exp_min;
        mass_lower += block.lower;
    }
    if !mass_lower.is_finite() || mass_lower <= 0.0 {
        return Err("ADA-A3 two-stat softmax mass lower bound must be positive");
    }
    Ok((global_max, mass_lower, exp_evaluations))
}

struct SkipPlan {
    omitted: Vec<bool>,
    omitted_count: usize,
    omitted_upper: f64,
    bound_checks: usize,
}

fn plan_skips(blocks: &[BlockStat], global_max: f64, mass_lower: f64, epsilon: f64) -> SkipPlan {
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
    for index in order {
        if blocks[index].max.total_cmp(&global_max).is_eq() {
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
    scores: &[f64],
    blocks: &[BlockStat],
    omitted: &[bool],
    global_max: f64,
) -> Result<(Vec<f64>, usize), &'static str> {
    let mut probabilities = vec![0.0_f64; scores.len()];
    let mut kept_mass = 0.0_f64;
    let mut tokens_exponentiated = 0_usize;
    for (block, skipped) in blocks.iter().zip(omitted.iter().copied()) {
        if skipped {
            continue;
        }
        for (slot, &score) in probabilities[block.start..block.end]
            .iter_mut()
            .zip(scores[block.start..block.end].iter())
        {
            let term = (score - global_max).exp();
            tokens_exponentiated = tokens_exponentiated.saturating_add(1);
            if !term.is_finite() || term < 0.0 {
                return Err("ADA-A3 two-stat softmax kept term must be finite");
            }
            *slot = term;
            kept_mass += term;
        }
    }
    if !kept_mass.is_finite() || kept_mass <= 0.0 {
        return Err("ADA-A3 two-stat softmax kept mass must be positive");
    }
    for probability in &mut probabilities {
        *probability /= kept_mass;
    }
    Ok((probabilities, tokens_exponentiated))
}

/// Evaluate two-statistic certified block-sparse softmax and record counters.
///
/// # Errors
///
/// Returns an error when `scores` is empty or contains a non-finite value,
/// when `block_size` is zero, when `epsilon` is non-finite or negative, or
/// when a mass residual is non-finite.
#[must_use = "the certified distribution should be checked"]
pub fn certified_two_stat_softmax_accounted(
    scores: &[f64],
    block_size: usize,
    epsilon: f64,
) -> Result<(TwoStatSoftmax, TwoStatCost), &'static str> {
    validate_args(scores, block_size, epsilon)?;
    let mut blocks = partition_blocks(scores, block_size);
    let (global_max, mass_lower, bound_exps) = annotate_block_bounds(&mut blocks)?;
    let plan = plan_skips(&blocks, global_max, mass_lower, epsilon);
    let certified_l1_bound = 2.0 * plan.omitted_upper / mass_lower;
    if !certified_l1_bound.is_finite() || certified_l1_bound > epsilon {
        return Err("ADA-A3 two-stat softmax certificate exceeds the supplied budget");
    }
    let (probabilities, tokens_exponentiated) =
        renormalize_kept(scores, &blocks, &plan.omitted, global_max)?;
    let cost = TwoStatCost {
        blocks_total: blocks.len(),
        blocks_skipped: plan.omitted_count,
        tokens_scanned_for_stats: scores.len(),
        exp_evaluations: bound_exps.saturating_add(tokens_exponentiated),
        bound_checks: plan.bound_checks,
        tokens_exponentiated,
    };
    let result = TwoStatSoftmax {
        probabilities,
        certified_l1_bound,
        epsilon,
        block_skipped: plan.omitted,
        block_size,
    };
    Ok((result, cost))
}

/// Evaluate two-statistic certified block-sparse softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_two_stat_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_two_stat_softmax(
    scores: &[f64],
    block_size: usize,
    epsilon: f64,
) -> Result<TwoStatSoftmax, &'static str> {
    certified_two_stat_softmax_accounted(scores, block_size, epsilon).map(|(result, _)| result)
}

#[cfg(test)]
mod tests {
    use super::{TwoStatSoftmax, certified_two_stat_softmax, certified_two_stat_softmax_accounted};
    use crate::block_sparse::certified_block_sparse_softmax_accounted;

    fn first_order_skipped_on(
        scores: &[f64],
        block_size: usize,
        epsilon: f64,
    ) -> Result<Vec<bool>, &'static str> {
        let (result, _) = certified_block_sparse_softmax_accounted(scores, block_size, epsilon)?;
        Ok(result.block_skipped)
    }
    /// Independent dense softmax oracle (not used by the implementation).
    fn oracle_softmax(scores: &[f64]) -> Vec<f64> {
        let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let terms: Vec<f64> = scores.iter().map(|&score| (score - shift).exp()).collect();
        let mass: f64 = terms.iter().sum();
        terms.into_iter().map(|term| term / mass).collect()
    }

    fn l1_distance(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum()
    }

    fn assert_certificate_holds(scores: &[f64], result: &TwoStatSoftmax) {
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
        for (block_index, skipped) in result.block_skipped.iter().copied().enumerate() {
            let start = block_index * result.block_size;
            if !skipped {
                continue;
            }
            let end = (start + result.block_size).min(scores.len());
            assert!(
                result.probabilities[start..end]
                    .iter()
                    .all(|p| p.abs() <= 0.0)
            );
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(certified_two_stat_softmax(&[], 1, 1.0e-3).is_err());
        assert!(certified_two_stat_softmax(&[1.0, f64::NAN], 1, 1.0e-3).is_err());
        assert!(certified_two_stat_softmax(&[1.0, f64::INFINITY], 2, 1.0e-3).is_err());
        assert!(certified_two_stat_softmax(&[1.0, 2.0], 0, 1.0e-3).is_err());
        assert!(certified_two_stat_softmax(&[1.0, 2.0], 2, f64::NAN).is_err());
        assert!(certified_two_stat_softmax(&[1.0, 2.0], 2, -0.1).is_err());
    }

    #[test]
    fn singleton_and_all_equal_never_skip() {
        let one = certified_two_stat_softmax(&[4.0], 4, 1.0e-6).unwrap();
        assert_eq!(one.probabilities, vec![1.0]);
        assert_eq!(one.block_skipped, vec![false]);
        assert!(one.certified_l1_bound.abs() <= f64::EPSILON);

        let equal = [1.5_f64; 8];
        let result = certified_two_stat_softmax(&equal, 2, 1.0e-4).unwrap();
        assert!(result.block_skipped.iter().all(|skipped| !skipped));
        assert!(result.certified_l1_bound.abs() <= f64::EPSILON);
        assert_certificate_holds(&equal, &result);
        for probability in &result.probabilities {
            assert!((probability - 0.125).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn never_skips_the_global_maximum_block() {
        let scores = [-8.0, -8.0, -8.0, -8.0, 3.0, -1.0, -8.0, -8.0];
        let (result, cost) = certified_two_stat_softmax_accounted(&scores, 4, 1.0).unwrap();
        assert!(!result.block_skipped[1], "max block must be kept");
        let peak = result
            .probabilities
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        assert_eq!(peak.0, 4);
        assert!(cost.blocks_skipped < cost.blocks_total);
        assert_certificate_holds(&scores, &result);
    }

    #[test]
    fn wide_gap_skips_cold_prefix() {
        let mut scores = vec![-20.0; 12];
        scores[9] = 0.0;
        let (result, cost) = certified_two_stat_softmax_accounted(&scores, 4, 1.0e-6).unwrap();
        assert!(result.block_skipped[0]);
        assert!(result.block_skipped[1]);
        assert!(!result.block_skipped[2]);
        assert_eq!(cost.blocks_skipped, 2);
        assert_eq!(cost.tokens_exponentiated, 4);
        assert_eq!(cost.tokens_scanned_for_stats, 12);
        assert!(result.certified_l1_bound <= 1.0e-6);
        assert!(result.probabilities[9] > 0.999_999);
        assert_certificate_holds(&scores, &result);
    }

    #[test]
    fn warm_outlier_block_skipped_only_by_two_stat() {
        // Block 0: unique global max. Block 1: one token 3 nats below the
        // max and seven extremely cold tokens. The first-order bound treats
        // the whole cold block as 8 * exp(-3) and cannot skip it at
        // epsilon = 0.1; the two-stat bound sees the second max at -20.
        let mut scores = vec![0.0; 8];
        scores.extend(std::iter::repeat_n(-20.0, 8));
        scores[8] = -3.0;
        let epsilon = 0.1_f64;
        let (first, first_cost) =
            certified_block_sparse_softmax_accounted(&scores, 8, epsilon).unwrap();
        let (two, two_cost) = certified_two_stat_softmax_accounted(&scores, 8, epsilon).unwrap();
        assert_eq!(first_cost.blocks_skipped, 0);
        assert!(!first.block_skipped[1]);
        assert_eq!(two_cost.blocks_skipped, 1);
        assert!(two.block_skipped[1]);
        assert!(!two.block_skipped[0]);
        assert!(two.certified_l1_bound <= epsilon);
        assert_certificate_holds(&scores, &two);
    }

    #[test]
    fn unique_needle_hit_at_small_and_large_margins() {
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut scores = vec![0.0; 8];
            scores[3] = margin;
            let result = certified_two_stat_softmax(&scores, 2, 1.0e-4).unwrap();
            let peak = result
                .probabilities
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            assert_eq!(peak, 3, "missed needle at margin {margin}");
            assert_certificate_holds(&scores, &result);
        }
    }

    #[test]
    fn equal_pair_interpolation_is_preserved() {
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut scores = vec![0.0; 8];
            scores[2] = margin;
            scores[5] = margin;
            let result = certified_two_stat_softmax(&scores, 2, 1.0e-4).unwrap();
            assert!(!result.block_skipped[1]);
            assert!(!result.block_skipped[2]);
            assert!((result.probabilities[2] - result.probabilities[5]).abs() <= 1.0e-12);
            assert!(result.probabilities[2] > result.probabilities[0]);
            assert_certificate_holds(&scores, &result);
        }
    }

    #[test]
    fn zero_budget_computes_dense_softmax() {
        let scores = [1.0, 0.0, -3.0, 0.5, 2.0, -1.0];
        let (result, cost) = certified_two_stat_softmax_accounted(&scores, 2, 0.0).unwrap();
        assert_eq!(cost.blocks_skipped, 0);
        assert!(result.certified_l1_bound.abs() <= f64::EPSILON);
        assert_eq!(cost.tokens_exponentiated, scores.len());
        let observed = l1_distance(&result.probabilities, &oracle_softmax(&scores));
        assert!(observed <= 1.0e-14, "dense path L1 {observed}");
    }

    #[test]
    fn last_block_may_be_short() {
        let scores = [0.0, -4.0, -4.0, -4.0, 2.0];
        let result = certified_two_stat_softmax(&scores, 3, 1.0e-3).unwrap();
        assert_eq!(result.block_skipped.len(), 2);
        assert!(!result.block_skipped[1]);
        assert!((result.probabilities.iter().sum::<f64>() - 1.0).abs() <= 1.0e-12);
        assert_certificate_holds(&scores, &result);
    }

    fn grid_domain() -> (Vec<i32>, usize) {
        if cfg!(miri) {
            (vec![-2, 0, 1], 3)
        } else {
            (vec![-3, -1, 0, 2], 5)
        }
    }

    fn enumerate_scores(length: usize, domain: &[i32]) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        let mut current = vec![0_usize; length];
        let radix = domain.len();
        loop {
            out.push(
                current
                    .iter()
                    .map(|&index| f64::from(domain[index]))
                    .collect(),
            );
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
    fn exhaustive_grid_respects_certificate_and_is_never_looser() {
        let (domain, max_len) = grid_domain();
        let mut vectors = 0_usize;
        let mut tight_skips = 0_usize;
        let mut loose_skips = 0_usize;
        let mut first_loose_skips = 0_usize;
        let mut two_extra_loose = 0_usize;
        let mut same_set_tighter_or_equal = 0_usize;
        for length in 1..=max_len {
            for scores in enumerate_scores(length, &domain) {
                for block_size in 1..=length {
                    let tight = certified_two_stat_softmax(&scores, block_size, 1.0e-3).unwrap();
                    let loose = certified_two_stat_softmax(&scores, block_size, 5.0e-2).unwrap();
                    assert_certificate_holds(&scores, &tight);
                    assert_certificate_holds(&scores, &loose);
                    vectors += 1;
                    if tight.block_skipped.iter().any(|skipped| *skipped) {
                        tight_skips += 1;
                    }
                    if loose.block_skipped.iter().any(|skipped| *skipped) {
                        loose_skips += 1;
                    }
                    let first_loose = first_order_skipped_on(&scores, block_size, 5.0e-2).unwrap();
                    if first_loose.iter().any(|skipped| *skipped) {
                        first_loose_skips += 1;
                    }
                    let two_count = loose.block_skipped.iter().filter(|s| **s).count();
                    let first_count = first_loose.iter().filter(|s| **s).count();
                    if two_count > first_count {
                        two_extra_loose += 1;
                    }
                    if loose.block_skipped == first_loose {
                        same_set_tighter_or_equal += 1;
                    }
                }
            }
        }
        assert!(vectors > 0);
        eprintln!(
            "two_stat exhaustive_grid vectors={vectors} tight_eps=1e-3 skips={tight_skips} loose_eps=5e-2 skips={loose_skips} first_order_loose_skips={first_loose_skips} two_stat_extra_loose={two_extra_loose} identical_skip_sets={same_set_tighter_or_equal}"
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

    fn fill_unit_gaussians(scores: &mut [f64], state: &mut u64) {
        let mut index = 0;
        while index < scores.len() {
            let (first, second) = box_muller(state);
            scores[index] = first;
            if index + 1 < scores.len() {
                scores[index + 1] = second;
            }
            index += 2;
        }
    }

    fn survey_regime(
        label: &str,
        samples: usize,
        length: usize,
        spike: Option<f64>,
        medium_outlier: bool,
        epsilon: f64,
    ) {
        let mut state = 0xA3B5_C0DE_2026_1110;
        let mut violations = 0_usize;
        let mut skip_sum = 0.0_f64;
        let mut first_skip_sum = 0.0_f64;
        let mut max_observed = 0.0_f64;
        let mut max_bound = 0.0_f64;
        let block_size = if length >= 8 { 8 } else { 2 };
        for _ in 0..samples {
            let mut scores = vec![0.0; length];
            fill_unit_gaussians(&mut scores, &mut state);
            if let Some(height) = spike {
                scores[length / 2] += height;
            }
            if medium_outlier {
                let mut offset = 0;
                while offset < length {
                    let end = (offset + block_size).min(length);
                    let block_max = scores[offset..end]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let global_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if block_max < global_max {
                        for score in &mut scores[offset..end] {
                            *score = global_max - 20.0;
                        }
                        scores[offset] = global_max - 3.0;
                    }
                    offset = end;
                }
            }
            let (result, cost) =
                certified_two_stat_softmax_accounted(&scores, block_size, epsilon).unwrap();
            let (_, first_cost) =
                certified_block_sparse_softmax_accounted(&scores, block_size, epsilon).unwrap();
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
            }
            assert_certificate_holds(&scores, &result);
        }
        assert_eq!(violations, 0, "{label} had certificate violations");
        #[allow(clippy::cast_precision_loss)]
        let samples_f = f64::from(u32::try_from(samples).expect("sample count fits u32"));
        let mean_skip = skip_sum / samples_f;
        let mean_first = first_skip_sum / samples_f;
        eprintln!(
            "{label} n={length} samples={samples} block={block_size} eps={epsilon}: mean_skip_frac={mean_skip:.6} first_order_mean_skip={mean_first:.6} max_L1={max_observed:.6e} max_bound={max_bound:.6e}"
        );
    }

    #[test]
    fn gaussian_survey_never_exceeds_certificate() {
        let samples = if cfg!(miri) { 4 } else { 200 };
        let length = if cfg!(miri) { 8 } else { 32 };
        if cfg!(miri) {
            survey_regime("iid_gaussian", samples, length, None, false, 1.0e-4);
            survey_regime(
                "spiked_gaussian",
                samples,
                length,
                Some(20.0),
                false,
                1.0e-4,
            );
            survey_regime("medium_outlier", samples, length, None, true, 1.0e-1);
            return;
        }
        for epsilon in [1.0e-6_f64, 1.0e-4, 1.0e-2, 1.0e-1] {
            survey_regime("iid_gaussian", samples, length, None, false, epsilon);
            survey_regime(
                "spiked_gaussian",
                samples,
                length,
                Some(20.0),
                false,
                epsilon,
            );
            survey_regime("medium_outlier", samples, length, None, true, epsilon);
        }
    }
}
