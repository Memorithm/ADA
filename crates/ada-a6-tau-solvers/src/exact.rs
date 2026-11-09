//! Exact rational tau domains for integer scores.
//!
//! `sparsemax_exact_i64` evaluates alpha = 2 (sparsemax) in checked `i128`
//! arithmetic: the threshold and every probability are reduced rationals, and
//! the probabilities sum to one with no f64 rounding and no mass renormalization.
//!
//! `entmax15_exact_i64` returns an exact rational alpha = 1.5 distribution
//! when some support prefix of the descending scores has a perfect-square
//! discriminant in the integer threshold quadratic and satisfies the
//! complementary-support inequalities. That domain strictly contains the
//! historical subset (equal scores of perfect-square length, or a unique
//! maximum with gap at least 2). Every other 1.5-entmax input fails closed,
//! including the symmetric length-2 case whose probabilities are `1/2` but
//! whose threshold is irrational. This module does not replace
//! `ada_a4_entmax_bnb::dense_entmax` and does not claim usefulness, novelty,
//! or FLAT adoption.

/// Reduced rational `numerator / denominator` with a positive denominator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRational {
    numerator: i128,
    denominator: u128,
}

impl ExactRational {
    /// Numerator of the reduced fraction. The sign lives here.
    #[must_use]
    pub const fn numerator(self) -> i128 {
        self.numerator
    }

    /// Positive reduced denominator.
    #[must_use]
    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    /// Build a reduced rational. The sign is normalized onto the numerator.
    ///
    /// # Errors
    ///
    /// Returns an error when the denominator is zero or reduction overflows.
    pub fn reduced(numerator: i128, denominator: i128) -> Result<Self, &'static str> {
        Self::from_parts(numerator, denominator)
    }

    /// Narrow to binary64 when both parts are exact binary64 integers.
    ///
    /// The division itself may still round. Callers that need a bit-exact
    /// rational must keep the numerator and denominator.
    ///
    /// # Errors
    ///
    /// Returns an error when either part exceeds `2^53` or the quotient is
    /// non-finite.
    pub fn to_f64(self) -> Result<f64, &'static str> {
        const EXACT_F64_INT_MAX: u128 = 1_u128 << 53;
        let magnitude = abs_i128(self.numerator)?;
        if magnitude > EXACT_F64_INT_MAX || self.denominator > EXACT_F64_INT_MAX {
            return Err("ADA-A6 exact rational is outside the exact binary64 integer range");
        }
        #[allow(clippy::cast_precision_loss)]
        let numerator = self.numerator as f64;
        #[allow(clippy::cast_precision_loss)]
        let denominator = self.denominator as f64;
        let value = numerator / denominator;
        if value.is_finite() {
            Ok(value)
        } else {
            Err("ADA-A6 exact rational quotient is non-finite")
        }
    }

    fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    fn from_parts(numerator: i128, denominator: i128) -> Result<Self, &'static str> {
        if denominator == 0 {
            return Err("ADA-A6 exact rational denominator is zero");
        }
        let (numerator, denominator) = if denominator < 0 {
            (
                numerator
                    .checked_neg()
                    .ok_or("ADA-A6 exact rational sign overflow")?,
                denominator
                    .checked_neg()
                    .ok_or("ADA-A6 exact rational sign overflow")?,
            )
        } else {
            (numerator, denominator)
        };
        let denominator = u128::try_from(denominator)
            .map_err(|_| "ADA-A6 exact rational denominator is not positive")?;
        let magnitude = abs_i128(numerator)?;
        let divisor = gcd_u128(magnitude, denominator);
        let numerator = numerator
            / i128::try_from(divisor).map_err(|_| "ADA-A6 exact rational reduction overflow")?;
        Ok(Self {
            numerator,
            denominator: denominator / divisor,
        })
    }
}

/// Exact rational sparsemax or exact rational 1.5-entmax distribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactDistribution {
    /// One reduced probability per input score, in input order.
    pub probabilities: Vec<ExactRational>,
    /// Normalization threshold in the alpha-entmax convention
    /// `p_i = [(alpha - 1) z_i - tau]_+ ^{1/(alpha-1)}`.
    pub tau: ExactRational,
    /// Number of strictly positive probabilities.
    pub support_size: usize,
}

/// Exact sparsemax (`alpha = 2`) for integer scores.
///
/// # Errors
///
/// Returns an error on an empty input or when checked arithmetic would
/// overflow. A non-empty `i64` vector whose prefix sums fit in `i128` has a
/// defined exact result.
#[must_use = "the exact sparsemax distribution should be checked"]
pub fn sparsemax_exact_i64(scores: &[i64]) -> Result<ExactDistribution, &'static str> {
    if scores.is_empty() {
        return Err("ADA-A6 exact sparsemax requires at least one score");
    }

    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| scores[right].cmp(&scores[left]).then(left.cmp(&right)));

    let mut cumulative: i128 = 0;
    let mut support = 0_usize;
    for (position, &index) in order.iter().enumerate() {
        let score = i128::from(scores[index]);
        cumulative = cumulative
            .checked_add(score)
            .ok_or("ADA-A6 exact sparsemax cumulative overflow")?;
        let j = i128::try_from(position + 1)
            .map_err(|_| "ADA-A6 exact sparsemax support index overflow")?;
        let left = j
            .checked_mul(score)
            .and_then(|product| product.checked_add(1))
            .ok_or("ADA-A6 exact sparsemax support test overflow")?;
        if left > cumulative {
            support = position + 1;
        }
    }
    if support == 0 {
        return Err("ADA-A6 exact sparsemax found an empty support");
    }

    let mut top_sum: i128 = 0;
    for &index in &order[..support] {
        top_sum = top_sum
            .checked_add(i128::from(scores[index]))
            .ok_or("ADA-A6 exact sparsemax support sum overflow")?;
    }
    let k = i128::try_from(support).map_err(|_| "ADA-A6 exact sparsemax support index overflow")?;
    let tau_numerator = top_sum
        .checked_sub(1)
        .ok_or("ADA-A6 exact sparsemax threshold overflow")?;

    let mut probabilities = vec![ExactRational::zero(); scores.len()];
    let mut numerator_sum: i128 = 0;
    for &index in &order[..support] {
        let numerator = k
            .checked_mul(i128::from(scores[index]))
            .and_then(|product| product.checked_sub(tau_numerator))
            .ok_or("ADA-A6 exact sparsemax probability overflow")?;
        if numerator <= 0 {
            return Err("ADA-A6 exact sparsemax support member is not strictly positive");
        }
        numerator_sum = numerator_sum
            .checked_add(numerator)
            .ok_or("ADA-A6 exact sparsemax probability sum overflow")?;
        probabilities[index] = ExactRational::from_parts(numerator, k)?;
    }
    if numerator_sum != k {
        return Err("ADA-A6 exact sparsemax probabilities do not sum to one");
    }
    for &index in &order[support..] {
        let numerator = k
            .checked_mul(i128::from(scores[index]))
            .and_then(|product| product.checked_sub(tau_numerator))
            .ok_or("ADA-A6 exact sparsemax inactive-score overflow")?;
        if numerator > 0 {
            return Err("ADA-A6 exact sparsemax inactive score is strictly positive");
        }
    }

    Ok(ExactDistribution {
        probabilities,
        tau: ExactRational::from_parts(tau_numerator, k)?,
        support_size: support,
    })
}

/// Logical counters for one exact 1.5-entmax discriminant scan.
///
/// These count algorithmic steps. They are not hardware instruction, cycle,
/// bandwidth, or wall-clock claims.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactEntmax15Cost {
    /// Input scores placed into descending, index-stable order.
    pub scores_sorted: usize,
    /// Support prefixes whose first two moments and discriminant were formed.
    pub candidate_supports_tested: usize,
    /// Prefixes whose discriminant was a non-negative perfect square.
    pub perfect_square_discriminants: usize,
    /// Individual complementary score-versus-threshold inequality tests.
    pub support_inequality_checks: usize,
}

/// Exact alpha = 1.5 entmax on the rational discriminant domain of integer scores.
///
/// A descending prefix of size `ρ` is accepted only when the threshold
/// quadratic `ρ t² - 2 S₁ t + S₂ - 4 = 0` has a perfect-square discriminant
/// `D = S₁² - ρ S₂ + 4ρ`, the smaller root `t = (S₁ - √D) / ρ` is rational,
/// every prefix score is strictly above `t`, and every excluded score is at
/// most `t`. Probabilities are then `pᵢ = (zᵢ - t)₊² / 4`.
///
/// # Errors
///
/// Returns an error on an empty input, on arithmetic overflow, when two
/// prefixes both look admissible, or when no prefix is in the rational
/// domain (fail closed; no bisection fallback).
#[must_use = "the exact 1.5-entmax distribution should be checked"]
pub fn entmax15_exact_i64(scores: &[i64]) -> Result<ExactDistribution, &'static str> {
    let (distribution, _cost) = entmax15_exact_i64_accounted(scores)?;
    Ok(distribution)
}

/// Exact alpha = 1.5 entmax together with logical scan counters.
///
/// # Errors
///
/// Same fail-closed conditions as [`entmax15_exact_i64`].
#[must_use = "the exact 1.5-entmax distribution should be checked"]
pub fn entmax15_exact_i64_accounted(
    scores: &[i64],
) -> Result<(ExactDistribution, ExactEntmax15Cost), &'static str> {
    if scores.is_empty() {
        return Err("ADA-A6 exact entmax15 requires at least one score");
    }

    let mut cost = ExactEntmax15Cost {
        scores_sorted: scores.len(),
        ..ExactEntmax15Cost::default()
    };
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&left, &right| scores[right].cmp(&scores[left]).then(left.cmp(&right)));

    let mut accepted: Option<(usize, i128, i128)> = None;
    let mut support_sum: i128 = 0;
    let mut support_sq: i128 = 0;
    for prefix in 1..=scores.len() {
        let score = i128::from(scores[order[prefix - 1]]);
        support_sum = support_sum
            .checked_add(score)
            .ok_or("ADA-A6 exact entmax15 support sum overflow")?;
        let square = score
            .checked_mul(score)
            .ok_or("ADA-A6 exact entmax15 score square overflow")?;
        support_sq = support_sq
            .checked_add(square)
            .ok_or("ADA-A6 exact entmax15 support square overflow")?;
        cost.candidate_supports_tested = cost
            .candidate_supports_tested
            .checked_add(1)
            .ok_or("ADA-A6 exact entmax15 cost overflow")?;

        let Some((t_numerator, rho)) = perfect_square_threshold(support_sum, support_sq, prefix)?
        else {
            continue;
        };
        cost.perfect_square_discriminants = cost
            .perfect_square_discriminants
            .checked_add(1)
            .ok_or("ADA-A6 exact entmax15 cost overflow")?;
        if !prefix_respects_threshold(scores, &order, prefix, t_numerator, rho, &mut cost)? {
            continue;
        }
        if accepted.is_some() {
            return Err("ADA-A6 exact entmax15 found multiple rational supports");
        }
        accepted = Some((prefix, t_numerator, rho));
    }

    let Some((support, t_numerator, rho)) = accepted else {
        return Err("ADA-A6 exact entmax15 is outside the rational domain");
    };
    let distribution = assemble_entmax15(scores, &order, support, t_numerator, rho)?;
    Ok((distribution, cost))
}

fn perfect_square_threshold(
    support_sum: i128,
    support_sq: i128,
    prefix: usize,
) -> Result<Option<(i128, i128)>, &'static str> {
    let rho = i128::try_from(prefix).map_err(|_| "ADA-A6 exact entmax15 support index overflow")?;
    let four_rho = rho
        .checked_mul(4)
        .ok_or("ADA-A6 exact entmax15 discriminant overflow")?;
    let rho_sq = rho
        .checked_mul(support_sq)
        .ok_or("ADA-A6 exact entmax15 discriminant overflow")?;
    let sum_sq = support_sum
        .checked_mul(support_sum)
        .ok_or("ADA-A6 exact entmax15 discriminant overflow")?;
    let Some(discriminant) = sum_sq
        .checked_sub(rho_sq)
        .and_then(|value| value.checked_add(four_rho))
    else {
        return Err("ADA-A6 exact entmax15 discriminant overflow");
    };
    if discriminant < 0 {
        return Ok(None);
    }
    let discriminant =
        u128::try_from(discriminant).map_err(|_| "ADA-A6 exact entmax15 discriminant overflow")?;
    let Some(root) = perfect_square_root_u128(discriminant) else {
        return Ok(None);
    };
    let root =
        i128::try_from(root).map_err(|_| "ADA-A6 exact entmax15 discriminant root overflow")?;
    let t_numerator = support_sum
        .checked_sub(root)
        .ok_or("ADA-A6 exact entmax15 threshold overflow")?;
    Ok(Some((t_numerator, rho)))
}

fn assemble_entmax15(
    scores: &[i64],
    order: &[usize],
    support: usize,
    t_numerator: i128,
    rho: i128,
) -> Result<ExactDistribution, &'static str> {
    let tau_denominator = rho
        .checked_mul(2)
        .ok_or("ADA-A6 exact entmax15 threshold overflow")?;
    let probability_denominator = rho
        .checked_mul(rho)
        .and_then(|value| value.checked_mul(4))
        .ok_or("ADA-A6 exact entmax15 probability overflow")?;

    let mut probabilities = vec![ExactRational::zero(); scores.len()];
    let mut numerator_sum: i128 = 0;
    for &index in &order[..support] {
        let shifted = rho
            .checked_mul(i128::from(scores[index]))
            .and_then(|product| product.checked_sub(t_numerator))
            .ok_or("ADA-A6 exact entmax15 probability overflow")?;
        if shifted <= 0 {
            return Err("ADA-A6 exact entmax15 support member is not strictly positive");
        }
        let numerator = shifted
            .checked_mul(shifted)
            .ok_or("ADA-A6 exact entmax15 probability overflow")?;
        numerator_sum = numerator_sum
            .checked_add(numerator)
            .ok_or("ADA-A6 exact entmax15 probability sum overflow")?;
        probabilities[index] = ExactRational::from_parts(numerator, probability_denominator)?;
    }
    if numerator_sum != probability_denominator {
        return Err("ADA-A6 exact entmax15 probabilities do not sum to one");
    }
    Ok(ExactDistribution {
        probabilities,
        tau: ExactRational::from_parts(t_numerator, tau_denominator)?,
        support_size: support,
    })
}

fn prefix_respects_threshold(
    scores: &[i64],
    order: &[usize],
    prefix: usize,
    t_numerator: i128,
    rho: i128,
    cost: &mut ExactEntmax15Cost,
) -> Result<bool, &'static str> {
    for &index in &order[..prefix] {
        cost.support_inequality_checks = cost
            .support_inequality_checks
            .checked_add(1)
            .ok_or("ADA-A6 exact entmax15 cost overflow")?;
        let left = rho
            .checked_mul(i128::from(scores[index]))
            .ok_or("ADA-A6 exact entmax15 threshold comparison overflow")?;
        if left <= t_numerator {
            return Ok(false);
        }
    }
    for &index in &order[prefix..] {
        cost.support_inequality_checks = cost
            .support_inequality_checks
            .checked_add(1)
            .ok_or("ADA-A6 exact entmax15 cost overflow")?;
        let left = rho
            .checked_mul(i128::from(scores[index]))
            .ok_or("ADA-A6 exact entmax15 threshold comparison overflow")?;
        if left > t_numerator {
            return Ok(false);
        }
    }
    Ok(true)
}

fn perfect_square_root_u128(value: u128) -> Option<u128> {
    let root = integer_sqrt_u128(value);
    match root.checked_mul(root) {
        Some(square) if square == value => Some(root),
        _ => None,
    }
}

fn integer_sqrt_u128(value: u128) -> u128 {
    if value <= 1 {
        return value;
    }
    let mut x = value;
    let mut y = value / 2;
    while y < x {
        x = y;
        y = u128::midpoint(x, value / x);
    }
    x
}

fn abs_i128(value: i128) -> Result<u128, &'static str> {
    if value == i128::MIN {
        return Err("ADA-A6 exact rational magnitude overflow");
    }
    let magnitude = if value < 0 { -value } else { value };
    u128::try_from(magnitude).map_err(|_| "ADA-A6 exact rational magnitude overflow")
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparsemax_sorted;
    use ada_a4_entmax_bnb::dense_entmax;

    fn i64_to_f64(score: i64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let value = score as f64;
        value
    }

    fn rat(numerator: i128, denominator: i128) -> ExactRational {
        ExactRational::from_parts(numerator, denominator).unwrap()
    }

    fn assert_distribution(
        actual: &ExactDistribution,
        probabilities: &[ExactRational],
        tau: ExactRational,
        support: usize,
    ) {
        assert_eq!(actual.probabilities, probabilities);
        assert_eq!(actual.tau, tau);
        assert_eq!(actual.support_size, support);
        assert_eq!(
            actual
                .probabilities
                .iter()
                .filter(|probability| probability.numerator() > 0)
                .count(),
            support
        );
    }

    fn is_legacy_entmax15_domain(scores: &[i64]) -> bool {
        if scores.is_empty() {
            return false;
        }
        if scores.iter().all(|&score| score == scores[0]) {
            let length = u64::try_from(scores.len()).ok();
            return length
                .is_some_and(|value| perfect_square_root_u128(u128::from(value)).is_some());
        }
        let maximum = scores.iter().copied().max().unwrap();
        let max_count = scores.iter().filter(|&&score| score == maximum).count();
        max_count == 1
            && scores
                .iter()
                .filter(|&&score| score != maximum)
                .all(|&score| i128::from(score) <= i128::from(maximum) - 2)
    }

    fn assert_matches_dense_oracle(scores: &[i64], exact: &ExactDistribution) {
        let f64_scores: Vec<f64> = scores.iter().copied().map(i64_to_f64).collect();
        let oracle = dense_entmax(&f64_scores, 1.5).unwrap();
        for (rational, &approx) in exact.probabilities.iter().zip(&oracle.probabilities) {
            let rendered = rational.to_f64().unwrap();
            assert!(
                (rendered - approx).abs() <= 1.0e-9,
                "{rendered} != {approx} for {scores:?}"
            );
        }
        let tau = exact.tau.to_f64().unwrap();
        let mut objective = -1.0_f64;
        for score in &f64_scores {
            let shifted = 0.5 * score - tau;
            if shifted > 0.0 {
                objective += shifted * shifted;
            }
        }
        assert!(objective.abs() <= 1.0e-9, "objective {objective} at {tau}");
    }

    #[test]
    fn sparsemax_hand_derived_fixtures() {
        let cases = [
            (vec![5], vec![rat(1, 1)], rat(4, 1), 1),
            (vec![0, 0], vec![rat(1, 2), rat(1, 2)], rat(-1, 2), 2),
            (vec![2, 0], vec![rat(1, 1), rat(0, 1)], rat(1, 1), 1),
            (
                vec![1, 1, 0],
                vec![rat(1, 2), rat(1, 2), rat(0, 1)],
                rat(1, 2),
                2,
            ),
            (
                vec![1, 1, 1],
                vec![rat(1, 3), rat(1, 3), rat(1, 3)],
                rat(2, 3),
                3,
            ),
            (
                vec![2, 2, 0],
                vec![rat(1, 2), rat(1, 2), rat(0, 1)],
                rat(3, 2),
                2,
            ),
            (vec![-5, -4], vec![rat(0, 1), rat(1, 1)], rat(-5, 1), 1),
            (
                vec![3, 1, 0],
                vec![rat(1, 1), rat(0, 1), rat(0, 1)],
                rat(2, 1),
                1,
            ),
        ];
        for (scores, probabilities, tau, support) in cases {
            let actual = sparsemax_exact_i64(&scores).unwrap();
            assert_distribution(&actual, &probabilities, tau, support);
        }
    }

    #[test]
    fn sparsemax_permutation_and_ties_are_index_stable() {
        let forward = sparsemax_exact_i64(&[4, 4, 1, 0]).unwrap();
        let backward = sparsemax_exact_i64(&[0, 1, 4, 4]).unwrap();
        assert_eq!(forward.probabilities[0], backward.probabilities[3]);
        assert_eq!(forward.probabilities[1], backward.probabilities[2]);
        assert_eq!(forward.tau, backward.tau);
        assert_eq!(forward.support_size, 2);
        assert_eq!(forward.probabilities[0], rat(1, 2));
        assert_eq!(forward.probabilities[2].numerator(), 0);
    }

    #[test]
    fn sparsemax_matches_f64_sorted_projection_on_small_integers() {
        for scores in [
            vec![0_i64],
            vec![3, -1, 3, 0],
            vec![-8, -1, 12, 12, 11],
            vec![1, 1, 1, 1, 1, 1],
            vec![9, 1, 1, 1, -4],
        ] {
            let exact = sparsemax_exact_i64(&scores).unwrap();
            let f64_scores: Vec<f64> = scores.iter().copied().map(i64_to_f64).collect();
            let approximate = sparsemax_sorted(&f64_scores).unwrap();
            assert_eq!(exact.probabilities.len(), approximate.probabilities.len());
            for (rational, &approx) in exact.probabilities.iter().zip(&approximate.probabilities) {
                let rendered = rational.to_f64().unwrap();
                let scale = rendered.abs().max(approx.abs()).max(1.0);
                assert!(
                    (rendered - approx).abs() <= 1.0e-12 * scale,
                    "{rendered} != {approx} for {scores:?}"
                );
            }
        }
    }

    #[test]
    fn sparsemax_extreme_singleton_stays_exact() {
        let actual = sparsemax_exact_i64(&[i64::MAX]).unwrap();
        assert_distribution(&actual, &[rat(1, 1)], rat(i128::from(i64::MAX) - 1, 1), 1);
        let low = sparsemax_exact_i64(&[i64::MIN, i64::MIN]).unwrap();
        assert_eq!(low.support_size, 2);
        assert_eq!(low.probabilities[0], rat(1, 2));
    }

    #[test]
    fn sparsemax_rejects_empty_input() {
        assert!(sparsemax_exact_i64(&[]).is_err());
    }

    #[test]
    fn entmax15_hand_derived_one_hot_and_square_supports() {
        let singleton = entmax15_exact_i64(&[7]).unwrap();
        assert_distribution(&singleton, &[rat(1, 1)], rat(5, 2), 1);

        let gap = entmax15_exact_i64(&[4, 1, 0]).unwrap();
        assert_distribution(&gap, &[rat(1, 1), rat(0, 1), rat(0, 1)], rat(1, 1), 1);

        let boundary = entmax15_exact_i64(&[-1, -3]).unwrap();
        assert_distribution(&boundary, &[rat(1, 1), rat(0, 1)], rat(-3, 2), 1);

        let square = entmax15_exact_i64(&[0, 0, 0, 0]).unwrap();
        assert_distribution(
            &square,
            &[rat(1, 4), rat(1, 4), rat(1, 4), rat(1, 4)],
            rat(-1, 2),
            4,
        );

        let shifted = entmax15_exact_i64(&[5, 5, 5, 5]).unwrap();
        assert_eq!(shifted.tau, rat(2, 1));
        assert_eq!(shifted.probabilities[0], rat(1, 4));

        let nine = entmax15_exact_i64(&[1; 9]).unwrap();
        assert_distribution(&nine, &[rat(1, 9); 9], rat(1, 6), 9);
    }

    #[test]
    fn entmax15_new_unit_spike_family_is_exact() {
        let spike5 = entmax15_exact_i64(&[1, 0, 0, 0, 0]).unwrap();
        assert_distribution(
            &spike5,
            &[
                rat(16, 25),
                rat(9, 100),
                rat(9, 100),
                rat(9, 100),
                rat(9, 100),
            ],
            rat(-3, 10),
            5,
        );

        let permuted = entmax15_exact_i64(&[0, 0, 1, 0, 0]).unwrap();
        assert_eq!(permuted.probabilities[2], rat(16, 25));
        assert_eq!(permuted.probabilities[0], rat(9, 100));
        assert_eq!(permuted.tau, rat(-3, 10));
        assert_eq!(permuted.support_size, 5);

        let translated = entmax15_exact_i64(&[4, 3, 3, 3, 3]).unwrap();
        assert_eq!(translated.probabilities, spike5.probabilities);
        assert_eq!(translated.tau, rat(6, 5));

        let spike8 = entmax15_exact_i64(&[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(spike8.probabilities[0], rat(9, 16));
        assert_eq!(spike8.probabilities[7], rat(1, 16));
        assert_eq!(spike8.tau, rat(-1, 4));
        assert_eq!(spike8.support_size, 8);

        let partial = entmax15_exact_i64(&[1, 0, 0, 0, 0, -1]).unwrap();
        assert_eq!(partial.support_size, 5);
        assert_eq!(partial.probabilities[0], rat(16, 25));
        assert_eq!(partial.probabilities[5], rat(0, 1));
        assert_eq!(partial.tau, rat(-3, 10));
    }

    #[test]
    fn entmax15_matches_dense_oracle_inside_the_domain() {
        for scores in [
            vec![7_i64],
            vec![4, 2, -3],
            vec![0, 0, 0, 0],
            vec![1; 9],
            vec![1, 0, 0, 0, 0],
            vec![0, 0, 1, 0, 0],
            vec![4, 3, 3, 3, 3],
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            vec![1, 0, 0, 0, 0, -1],
        ] {
            let exact = entmax15_exact_i64(&scores).unwrap();
            assert_matches_dense_oracle(&scores, &exact);
        }
    }

    #[test]
    fn entmax15_fails_closed_outside_the_rational_domain() {
        assert!(entmax15_exact_i64(&[]).is_err());
        assert!(entmax15_exact_i64(&[0, 0]).is_err());
        assert!(entmax15_exact_i64(&[3, 2]).is_err());
        assert!(entmax15_exact_i64(&[1, 1, 0]).is_err());
        assert!(entmax15_exact_i64(&[4, 4, 0]).is_err());
        assert!(entmax15_exact_i64(&[2, 1, 0]).is_err());
        assert!(entmax15_exact_i64(&[5, 5, 3]).is_err());
    }

    #[test]
    fn entmax15_reports_logical_cost_without_hardware_claims() {
        let (_distribution, cost) = entmax15_exact_i64_accounted(&[1, 0, 0, 0, 0]).unwrap();
        assert_eq!(cost.scores_sorted, 5);
        assert_eq!(cost.candidate_supports_tested, 5);
        assert_eq!(cost.perfect_square_discriminants, 2);
        assert_eq!(cost.support_inequality_checks, 7);
    }

    #[test]
    fn entmax15_exhaustive_small_grid_matches_oracle_and_expands_legacy_domain() {
        let mut accepted = 0_usize;
        let mut legacy = 0_usize;
        let mut novel = 0_usize;
        let mut rejected = 0_usize;
        let mut dual = 0_usize;
        for length in 1..=5_usize {
            let bound = 4_i64;
            let states = bound.pow(u32::try_from(length).unwrap());
            for state in 0..states {
                let mut scores = vec![0_i64; length];
                let mut cursor = state;
                for slot in &mut scores {
                    *slot = cursor % bound;
                    cursor /= bound;
                }
                match entmax15_exact_i64(&scores) {
                    Ok(exact) => {
                        accepted += 1;
                        if is_legacy_entmax15_domain(&scores) {
                            legacy += 1;
                        } else {
                            novel += 1;
                        }
                        assert_matches_dense_oracle(&scores, &exact);
                    }
                    Err(message) => {
                        if message == "ADA-A6 exact entmax15 found multiple rational supports" {
                            dual += 1;
                        }
                        assert!(
                            !is_legacy_entmax15_domain(&scores),
                            "legacy domain rejected for {scores:?}: {message}"
                        );
                        rejected += 1;
                    }
                }
            }
        }
        assert_eq!(dual, 0);
        assert_eq!(accepted, legacy + novel);
        // Lengths 1..=5 over scores in {0,1,2,3}: 1364 vectors. The partition
        // is an algorithmic observation on that finite grid, not a hardware claim.
        assert_eq!((accepted, legacy, novel, rejected), (195, 150, 45, 1169));
    }
}
