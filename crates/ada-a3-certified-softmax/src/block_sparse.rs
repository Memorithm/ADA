//! Certified block-sparse softmax (score-level).
//!
//! Softmax never has exact zeros, so the A5/A4 support-pruning certificates
//! do not apply. This module tests a different mechanism: partition the
//! scores into contiguous blocks, bound each block's probability mass from
//! its maximum alone, and skip a prefix of the coldest blocks whenever a
//! conservative L1 certificate stays inside a caller budget `epsilon`.
//!
//! Let `M` be the global maximum and `m_b` the maximum of block `b` of
//! length `n_b`. The shifted terms satisfy `0 < exp(z_i - M) <= 1`. Then
//!
//! - `U_b = n_b * exp(m_b - M)` upper-bounds the block's unnormalized mass,
//! - `L = sum_b exp(m_b - M)` lower-bounds the total mass `S`
//!   (one witness token per block),
//! - skipping a set `K` of blocks that do not contain a global maximum
//!   and renormalizing the rest yields
//!   `||p - q||_1 = 2 * S_K / S <= 2 * (sum_{b in K} U_b) / L`.
//!
//! The operator never skips a top-plateau block (any block with `m_b = M`),
//! so unique-argmax and equal-max interpolation cannot be erased by the
//! skip rule itself. Blocks are considered in increasing `U_b` (then index)
//! and skipped greedily while the certificate stays `<= epsilon`.
//!
//! The independent oracle is a separately implemented dense stable softmax
//! over all scores (see the tests). Logical counters are not hardware
//! measurements. No usefulness, novelty, or FLAT-adoption claim is made.

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockSparseCost {
    /// Contiguous blocks in the partition (the last one may be shorter).
    pub blocks_total: usize,
    /// Blocks assigned probability zero by the skip rule.
    pub blocks_skipped: usize,
    /// Scores read to form block maxima (always the full length).
    pub tokens_scanned_for_max: usize,
    /// `exp` evaluations used for block bounds (`blocks_total`) plus kept tokens.
    pub exp_evaluations: usize,
    /// Greedy certificate checks (one per non-top-plateau block).
    pub bound_checks: usize,
    /// Tokens that received a positive unnormalized `exp` in the second pass.
    pub tokens_exponentiated: usize,
}

/// Certified block-sparse softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockSparseSoftmax {
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
    upper: f64,
}

fn validate_args(scores: &[f64], block_size: usize, epsilon: f64) -> Result<(), &'static str> {
    if scores.is_empty() {
        return Err("ADA-A3 block-sparse softmax requires at least one score");
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("ADA-A3 block-sparse softmax scores must be finite");
    }
    if block_size == 0 {
        return Err("ADA-A3 block-sparse softmax block_size must be positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 block-sparse softmax epsilon must be finite and non-negative");
    }
    Ok(())
}

fn partition_blocks(scores: &[f64], block_size: usize) -> Vec<BlockStat> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    while offset < scores.len() {
        let end = (offset + block_size).min(scores.len());
        let max = scores[offset..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        blocks.push(BlockStat {
            start: offset,
            end,
            max,
            upper: 0.0,
        });
        offset = end;
    }
    blocks
}

fn annotate_block_bounds(blocks: &mut [BlockStat]) -> Result<(f64, f64, usize), &'static str> {
    let global_max = blocks
        .iter()
        .map(|block| block.max)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut mass_lower = 0.0_f64;
    let mut exp_evaluations = 0_usize;
    for block in blocks.iter_mut() {
        let shifted = (block.max - global_max).exp();
        exp_evaluations = exp_evaluations.saturating_add(1);
        if !shifted.is_finite() || shifted < 0.0 {
            return Err("ADA-A3 block-sparse softmax block bound must be finite");
        }
        #[allow(clippy::cast_precision_loss)]
        let width = (block.end - block.start) as f64;
        block.upper = width * shifted;
        mass_lower += shifted;
    }
    if !mass_lower.is_finite() || mass_lower <= 0.0 {
        return Err("ADA-A3 block-sparse softmax mass lower bound must be positive");
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
                return Err("ADA-A3 block-sparse softmax kept term must be finite");
            }
            *slot = term;
            kept_mass += term;
        }
    }
    if !kept_mass.is_finite() || kept_mass <= 0.0 {
        return Err("ADA-A3 block-sparse softmax kept mass must be positive");
    }
    for probability in &mut probabilities {
        *probability /= kept_mass;
    }
    Ok((probabilities, tokens_exponentiated))
}

/// Evaluate certified block-sparse softmax and record logical counters.
///
/// # Errors
///
/// Returns an error when `scores` is empty or contains a non-finite value,
/// when `block_size` is zero, when `epsilon` is non-finite or negative, or
/// when a mass residual is non-finite.
#[must_use = "the certified distribution should be checked"]
pub fn certified_block_sparse_softmax_accounted(
    scores: &[f64],
    block_size: usize,
    epsilon: f64,
) -> Result<(BlockSparseSoftmax, BlockSparseCost), &'static str> {
    validate_args(scores, block_size, epsilon)?;
    let mut blocks = partition_blocks(scores, block_size);
    let (global_max, mass_lower, bound_exps) = annotate_block_bounds(&mut blocks)?;
    let plan = plan_skips(&blocks, global_max, mass_lower, epsilon);
    let certified_l1_bound = 2.0 * plan.omitted_upper / mass_lower;
    if !certified_l1_bound.is_finite() || certified_l1_bound > epsilon {
        return Err("ADA-A3 block-sparse softmax certificate exceeds the supplied budget");
    }
    let (probabilities, tokens_exponentiated) =
        renormalize_kept(scores, &blocks, &plan.omitted, global_max)?;
    let cost = BlockSparseCost {
        blocks_total: blocks.len(),
        blocks_skipped: plan.omitted_count,
        tokens_scanned_for_max: scores.len(),
        exp_evaluations: bound_exps.saturating_add(tokens_exponentiated),
        bound_checks: plan.bound_checks,
        tokens_exponentiated,
    };
    let result = BlockSparseSoftmax {
        probabilities,
        certified_l1_bound,
        epsilon,
        block_skipped: plan.omitted,
        block_size,
    };
    Ok((result, cost))
}

/// Evaluate certified block-sparse softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_block_sparse_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_block_sparse_softmax(
    scores: &[f64],
    block_size: usize,
    epsilon: f64,
) -> Result<BlockSparseSoftmax, &'static str> {
    certified_block_sparse_softmax_accounted(scores, block_size, epsilon).map(|(result, _)| result)
}

#[cfg(test)]
mod tests {
    use super::{
        BlockSparseSoftmax, certified_block_sparse_softmax,
        certified_block_sparse_softmax_accounted,
    };

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

    fn assert_certificate_holds(scores: &[f64], result: &BlockSparseSoftmax) {
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
        assert!(certified_block_sparse_softmax(&[], 1, 1.0e-3).is_err());
        assert!(certified_block_sparse_softmax(&[1.0, f64::NAN], 1, 1.0e-3).is_err());
        assert!(certified_block_sparse_softmax(&[1.0, f64::INFINITY], 2, 1.0e-3).is_err());
        assert!(certified_block_sparse_softmax(&[1.0, 2.0], 0, 1.0e-3).is_err());
        assert!(certified_block_sparse_softmax(&[1.0, 2.0], 2, f64::NAN).is_err());
        assert!(certified_block_sparse_softmax(&[1.0, 2.0], 2, -0.1).is_err());
    }

    #[test]
    fn singleton_and_all_equal_never_skip() {
        let one = certified_block_sparse_softmax(&[4.0], 4, 1.0e-6).unwrap();
        assert_eq!(one.probabilities, vec![1.0]);
        assert_eq!(one.block_skipped, vec![false]);
        assert!(one.certified_l1_bound.abs() <= f64::EPSILON);

        let equal = [1.5_f64; 8];
        let result = certified_block_sparse_softmax(&equal, 2, 1.0e-4).unwrap();
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
        let (result, cost) = certified_block_sparse_softmax_accounted(&scores, 4, 1.0).unwrap();
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
        let (result, cost) = certified_block_sparse_softmax_accounted(&scores, 4, 1.0e-6).unwrap();
        assert!(result.block_skipped[0]);
        assert!(result.block_skipped[1]);
        assert!(!result.block_skipped[2]);
        assert_eq!(cost.blocks_skipped, 2);
        assert_eq!(cost.tokens_exponentiated, 4);
        assert_eq!(cost.tokens_scanned_for_max, 12);
        assert!(result.certified_l1_bound <= 1.0e-6);
        assert!(result.probabilities[9] > 0.999_999);
        let peak = result
            .probabilities
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
            .0;
        assert_eq!(peak, 9);
        assert_certificate_holds(&scores, &result);
    }

    #[test]
    fn unique_needle_hit_at_small_and_large_margins() {
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut scores = vec![0.0; 8];
            scores[3] = margin;
            let result = certified_block_sparse_softmax(&scores, 2, 1.0e-4).unwrap();
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
            let oracle = oracle_softmax(&scores);
            let oracle_peak = oracle
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;
            assert_eq!(oracle_peak, 3);
        }
    }

    #[test]
    fn equal_pair_interpolation_is_preserved() {
        for margin in [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0] {
            let mut scores = vec![0.0; 8];
            scores[2] = margin;
            scores[5] = margin;
            let result = certified_block_sparse_softmax(&scores, 2, 1.0e-4).unwrap();
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
        let (result, cost) = certified_block_sparse_softmax_accounted(&scores, 2, 0.0).unwrap();
        assert_eq!(cost.blocks_skipped, 0);
        assert!(result.certified_l1_bound.abs() <= f64::EPSILON);
        assert_eq!(cost.tokens_exponentiated, scores.len());
        let observed = l1_distance(&result.probabilities, &oracle_softmax(&scores));
        assert!(observed <= 1.0e-14, "dense path L1 {observed}");
    }

    #[test]
    fn last_block_may_be_short() {
        let scores = [0.0, -4.0, -4.0, -4.0, 2.0];
        let result = certified_block_sparse_softmax(&scores, 3, 1.0e-3).unwrap();
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
    fn exhaustive_grid_respects_certificate() {
        let (domain, max_len) = grid_domain();
        let mut vectors = 0_usize;
        let mut tight_skips = 0_usize;
        let mut loose_skips = 0_usize;
        for length in 1..=max_len {
            for scores in enumerate_scores(length, &domain) {
                for block_size in 1..=length {
                    let tight =
                        certified_block_sparse_softmax(&scores, block_size, 1.0e-3).unwrap();
                    let loose =
                        certified_block_sparse_softmax(&scores, block_size, 5.0e-2).unwrap();
                    assert_certificate_holds(&scores, &tight);
                    assert_certificate_holds(&scores, &loose);
                    vectors += 1;
                    if tight.block_skipped.iter().any(|skipped| *skipped) {
                        tight_skips += 1;
                    }
                    if loose.block_skipped.iter().any(|skipped| *skipped) {
                        loose_skips += 1;
                    }
                }
            }
        }
        assert!(vectors > 0);
        eprintln!(
            "exhaustive_grid vectors={vectors} tight_eps=1e-3 skips={tight_skips} loose_eps=5e-2 skips={loose_skips}"
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

    fn survey_regime(label: &str, samples: usize, length: usize, spike: Option<f64>, epsilon: f64) {
        let mut state = 0xA3B5_C0DE_2026_1110;
        let mut violations = 0_usize;
        let mut skip_sum = 0.0_f64;
        let mut max_observed = 0.0_f64;
        let mut max_bound = 0.0_f64;
        let block_size = if length >= 8 { 8 } else { 2 };
        for _ in 0..samples {
            let mut scores = vec![0.0; length];
            fill_unit_gaussians(&mut scores, &mut state);
            if let Some(height) = spike {
                scores[length / 2] += height;
            }
            let (result, cost) =
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
            }
            assert_certificate_holds(&scores, &result);
        }
        assert_eq!(violations, 0, "{label} had certificate violations");
        #[allow(clippy::cast_precision_loss)]
        let mean_skip =
            skip_sum / f64::from(u32::try_from(samples).expect("sample count fits u32"));
        eprintln!(
            "{label} n={length} samples={samples} block={block_size} eps={epsilon}: mean_skip_frac={mean_skip:.6} max_L1={max_observed:.6e} max_bound={max_bound:.6e}"
        );
    }

    #[test]
    fn gaussian_survey_never_exceeds_certificate() {
        let samples = if cfg!(miri) { 4 } else { 200 };
        let length = if cfg!(miri) { 8 } else { 32 };
        if cfg!(miri) {
            survey_regime("iid_gaussian", samples, length, None, 1.0e-4);
            survey_regime("spiked_gaussian", samples, length, Some(20.0), 1.0e-4);
            return;
        }
        for epsilon in [1.0e-6_f64, 1.0e-4, 1.0e-2, 1.0e-1] {
            survey_regime("iid_gaussian", samples, length, None, epsilon);
            survey_regime("spiked_gaussian", samples, length, Some(20.0), epsilon);
        }
    }
}
