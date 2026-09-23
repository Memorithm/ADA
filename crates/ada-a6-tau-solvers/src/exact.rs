//! Exact rational tau domains for integer scores.
//!
//! `sparsemax_exact_i64` evaluates alpha = 2 (sparsemax) in checked `i128`
//! arithmetic: the threshold and every probability are reduced rationals, and
//! the probabilities sum to one with no f64 rounding and no mass renormalization.
//!
//! `entmax15_exact_i64` is intentionally partial. It returns an exact rational
//! alpha = 1.5 distribution only when:
//! - every score is equal and the length is a perfect square, or
//! - there is a unique maximum at least 2 above every other score
//!   (one-hot support; the boundary gap of exactly 2 is included).
//!
//! Every other 1.5-entmax input fails closed. That includes the symmetric
//! length-2 case, whose probabilities are `1/2` but whose threshold is
//! irrational. This module does not replace `ada_a4_entmax_bnb::dense_entmax`
//! and does not claim usefulness, novelty, or FLAT adoption.

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

/// Exact rational sparsemax or restricted 1.5-entmax distribution.
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

/// Exact alpha = 1.5 entmax on the documented rational subset of integer scores.
///
/// # Errors
///
/// Returns an error on an empty input, on arithmetic overflow, or when the
/// scores are outside the exact domain (fail closed; no bisection fallback).
#[must_use = "the exact 1.5-entmax distribution should be checked"]
pub fn entmax15_exact_i64(scores: &[i64]) -> Result<ExactDistribution, &'static str> {
    if scores.is_empty() {
        return Err("ADA-A6 exact entmax15 requires at least one score");
    }
    if scores.iter().all(|&score| score == scores[0]) {
        return entmax15_equal(scores);
    }
    entmax15_unique_max(scores)
}

fn entmax15_equal(scores: &[i64]) -> Result<ExactDistribution, &'static str> {
    let Some(root) = perfect_square_root(scores.len()) else {
        return Err("ADA-A6 exact entmax15 is outside the rational domain");
    };
    let score = i128::from(scores[0]);
    let root_i = i128::from(root);
    let tau_numerator = score
        .checked_mul(root_i)
        .and_then(|product| product.checked_sub(2))
        .ok_or("ADA-A6 exact entmax15 threshold overflow")?;
    let tau_denominator = root_i
        .checked_mul(2)
        .ok_or("ADA-A6 exact entmax15 threshold overflow")?;
    let length = i128::try_from(scores.len())
        .map_err(|_| "ADA-A6 exact entmax15 length does not fit i128")?;
    let probability = ExactRational::from_parts(1, length)?;
    Ok(ExactDistribution {
        probabilities: vec![probability; scores.len()],
        tau: ExactRational::from_parts(tau_numerator, tau_denominator)?,
        support_size: scores.len(),
    })
}

fn entmax15_unique_max(scores: &[i64]) -> Result<ExactDistribution, &'static str> {
    let mut maximum = i64::MIN;
    let mut max_index = 0_usize;
    let mut max_count = 0_usize;
    for (index, &score) in scores.iter().enumerate() {
        if score > maximum {
            maximum = score;
            max_index = index;
            max_count = 1;
        } else if score == maximum {
            max_count += 1;
        }
    }
    if max_count != 1 {
        return Err("ADA-A6 exact entmax15 is outside the rational domain");
    }
    let limit = i128::from(maximum) - 2;
    if scores
        .iter()
        .any(|&score| score != maximum && i128::from(score) > limit)
    {
        return Err("ADA-A6 exact entmax15 is outside the rational domain");
    }

    let mut probabilities = vec![ExactRational::zero(); scores.len()];
    probabilities[max_index] = ExactRational::from_parts(1, 1)?;
    let tau_numerator = i128::from(maximum) - 2;
    Ok(ExactDistribution {
        probabilities,
        tau: ExactRational::from_parts(tau_numerator, 2)?,
        support_size: 1,
    })
}

fn perfect_square_root(length: usize) -> Option<u32> {
    let length = u64::try_from(length).ok()?;
    let root_u = integer_sqrt_u64(length);
    if root_u.checked_mul(root_u) == Some(length) {
        u32::try_from(root_u).ok()
    } else {
        None
    }
}

fn integer_sqrt_u64(value: u64) -> u64 {
    if value == 0 {
        return 0;
    }
    let target = u128::from(value);
    let mut x = target;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = u128::midpoint(x, target / x);
    }
    u64::try_from(x).unwrap_or(u64::MAX)
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
    fn entmax15_matches_dense_oracle_inside_the_domain() {
        for scores in [vec![7_i64], vec![4, 2, -3], vec![0, 0, 0, 0], vec![1; 9]] {
            let exact = entmax15_exact_i64(&scores).unwrap();
            let f64_scores: Vec<f64> = scores.iter().copied().map(i64_to_f64).collect();
            let oracle = dense_entmax(&f64_scores, 1.5).unwrap();
            for (rational, &approx) in exact.probabilities.iter().zip(&oracle.probabilities) {
                let rendered = rational.to_f64().unwrap();
                assert!(
                    (rendered - approx).abs() <= 1.0e-9,
                    "{rendered} != {approx}"
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
    }

    #[test]
    fn entmax15_fails_closed_outside_the_rational_domain() {
        assert!(entmax15_exact_i64(&[]).is_err());
        assert!(entmax15_exact_i64(&[0, 0]).is_err());
        assert!(entmax15_exact_i64(&[3, 2]).is_err());
        assert!(entmax15_exact_i64(&[1, 1, 0]).is_err());
        assert!(entmax15_exact_i64(&[4, 4, 0]).is_err());
    }
}
