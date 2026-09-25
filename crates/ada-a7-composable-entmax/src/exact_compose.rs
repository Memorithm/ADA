//! Exact integer composition of sparsemax sufficient statistics.
//!
//! For alpha = 2 and integer scores, a part is the multiset of its scores.
//! Composition is multiset union. Solving the merged multiset in checked
//! `i128` arithmetic yields the same reduced-rational distribution as solving
//! the concatenated scores with `ada_a6_tau_solvers::sparsemax_exact_i64`.
//!
//! An equal-score level is atomic: the support test does not depend on how
//! many copies sit inside that level, so the level is entirely inside the
//! support or entirely outside it. This is a reference-arithmetic identity,
//! not a usefulness, novelty, or FLAT-adoption claim. The historical f64
//! summary solver is unchanged.

use std::collections::BTreeMap;

use ada_a6_tau_solvers::ExactRational;

/// Multiset summary of integer sparsemax scores.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegerSparsemaxSummary {
    levels: BTreeMap<i64, u64>,
    total_count: u64,
}

/// One distinct integer score after an exact solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactLevel {
    /// Shared score of this plateau.
    pub score: i64,
    /// Number of scores on this plateau.
    pub count: u64,
    /// Probability of each score on this plateau.
    pub probability: ExactRational,
}

/// Exact rational sparsemax solved from a multiset, independent of token order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactLevelSolution {
    /// Descending plateaus, including inactive ones.
    pub levels: Vec<ExactLevel>,
    /// Normalization threshold in the alpha = 2 convention `p_i = [z_i - tau]_+`.
    pub tau: ExactRational,
    /// Number of strictly positive scores.
    pub support_count: u64,
}

impl IntegerSparsemaxSummary {
    /// Summarize integer scores.
    ///
    /// # Errors
    ///
    /// Returns an error on an empty part or a count that overflows `u64`.
    pub fn from_scores(scores: &[i64]) -> Result<Self, &'static str> {
        if scores.is_empty() {
            return Err("ADA-A7 exact composition requires at least one score");
        }
        let mut levels = BTreeMap::new();
        for &score in scores {
            let count = levels.entry(score).or_insert(0_u64);
            *count = count
                .checked_add(1)
                .ok_or("ADA-A7 exact composition count overflow")?;
        }
        Ok(Self {
            levels,
            total_count: u64::try_from(scores.len())
                .map_err(|_| "ADA-A7 exact composition count overflow")?,
        })
    }

    /// Multiset union of two non-empty parts.
    ///
    /// # Errors
    ///
    /// Returns an error when either part is empty or a merged count overflows.
    pub fn compose(left: &Self, right: &Self) -> Result<Self, &'static str> {
        if left.total_count == 0 || right.total_count == 0 {
            return Err("ADA-A7 exact composition rejects an empty part");
        }
        let mut levels = left.levels.clone();
        for (&score, &count) in &right.levels {
            let entry = levels.entry(score).or_insert(0);
            *entry = entry
                .checked_add(count)
                .ok_or("ADA-A7 exact composition count overflow")?;
        }
        Ok(Self {
            levels,
            total_count: left
                .total_count
                .checked_add(right.total_count)
                .ok_or("ADA-A7 exact composition count overflow")?,
        })
    }

    /// Solve sparsemax exactly from the multiset.
    ///
    /// # Errors
    ///
    /// Returns an error on overflow or if the prefix support identity fails.
    pub fn solve_exact(&self) -> Result<ExactLevelSolution, &'static str> {
        if self.levels.is_empty() || self.total_count == 0 {
            return Err("ADA-A7 exact composition cannot solve an empty summary");
        }

        let mut descending: Vec<(i64, u64)> = self
            .levels
            .iter()
            .map(|(&score, &count)| (score, count))
            .collect();
        descending.sort_by(|left, right| right.0.cmp(&left.0).then(left.0.cmp(&right.0)));

        let mut cumulative: i128 = 0;
        let mut rank: i128 = 0;
        let mut support_levels = 0_usize;
        for (index, &(score, count)) in descending.iter().enumerate() {
            let score_i = i128::from(score);
            let count_i = i128::from(count);
            let boundary = rank
                .checked_mul(score_i)
                .and_then(|product| product.checked_add(1))
                .ok_or("ADA-A7 exact composition support test overflow")?;
            if boundary > cumulative {
                let added = count_i
                    .checked_mul(score_i)
                    .ok_or("ADA-A7 exact composition support sum overflow")?;
                cumulative = cumulative
                    .checked_add(added)
                    .ok_or("ADA-A7 exact composition support sum overflow")?;
                rank = rank
                    .checked_add(count_i)
                    .ok_or("ADA-A7 exact composition support count overflow")?;
                support_levels = index + 1;
            } else {
                break;
            }
        }
        if support_levels == 0 || rank <= 0 {
            return Err("ADA-A7 exact composition found an empty support");
        }

        let tau_numerator = cumulative
            .checked_sub(1)
            .ok_or("ADA-A7 exact composition threshold overflow")?;
        let mut weighted_numerator: i128 = 0;
        let mut levels = Vec::with_capacity(descending.len());
        for (index, &(score, count)) in descending.iter().enumerate() {
            let unit_numerator = rank
                .checked_mul(i128::from(score))
                .and_then(|product| product.checked_sub(tau_numerator))
                .ok_or("ADA-A7 exact composition probability overflow")?;
            if index < support_levels {
                if unit_numerator <= 0 {
                    return Err("ADA-A7 exact composition support member is not strictly positive");
                }
                let weighted = unit_numerator
                    .checked_mul(i128::from(count))
                    .ok_or("ADA-A7 exact composition probability sum overflow")?;
                weighted_numerator = weighted_numerator
                    .checked_add(weighted)
                    .ok_or("ADA-A7 exact composition probability sum overflow")?;
                levels.push(ExactLevel {
                    score,
                    count,
                    probability: ExactRational::reduced(unit_numerator, rank)?,
                });
            } else if unit_numerator > 0 {
                return Err("ADA-A7 exact composition inactive level is strictly positive");
            } else {
                levels.push(ExactLevel {
                    score,
                    count,
                    probability: ExactRational::reduced(0, 1)?,
                });
            }
        }
        if weighted_numerator != rank {
            return Err("ADA-A7 exact composition probabilities do not sum to one");
        }

        Ok(ExactLevelSolution {
            levels,
            tau: ExactRational::reduced(tau_numerator, rank)?,
            support_count: u64::try_from(rank)
                .map_err(|_| "ADA-A7 exact composition support count overflow")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use ada_a6_tau_solvers::sparsemax_exact_i64;

    fn assert_matches_expanded(scores: &[i64]) {
        let summary = IntegerSparsemaxSummary::from_scores(scores).unwrap();
        let solved = summary.solve_exact().unwrap();
        let expanded = sparsemax_exact_i64(scores).unwrap();
        assert_eq!(solved.tau, expanded.tau);
        assert_eq!(
            solved.support_count,
            u64::try_from(expanded.support_size).unwrap()
        );
        for (&score, probability) in scores.iter().zip(&expanded.probabilities) {
            let level = solved
                .levels
                .iter()
                .find(|level| level.score == score)
                .unwrap();
            assert_eq!(level.probability, *probability);
        }
    }

    #[test]
    fn hand_derived_plateaus_match_the_expanded_oracle() {
        let solved = IntegerSparsemaxSummary::from_scores(&[5, 1, 5])
            .unwrap()
            .solve_exact()
            .unwrap();
        assert_eq!(solved.tau, ExactRational::reduced(9, 2).unwrap());
        assert_eq!(solved.support_count, 2);
        assert_eq!(solved.levels[0].score, 5);
        assert_eq!(solved.levels[0].count, 2);
        assert_eq!(
            solved.levels[0].probability,
            ExactRational::reduced(1, 2).unwrap()
        );
        assert_eq!(solved.levels[1].probability.numerator(), 0);
        assert_matches_expanded(&[5, 1, 5]);
        assert_matches_expanded(&[1, 1, 1]);
        assert_matches_expanded(&[2, 0]);
        assert_matches_expanded(&[-5, -4, -4]);
        assert_matches_expanded(&[i64::MAX]);
        assert_matches_expanded(&[i64::MIN, i64::MIN]);
    }

    #[test]
    fn composition_is_commutative_and_matches_concatenation() {
        let left = IntegerSparsemaxSummary::from_scores(&[4, 0, 4]).unwrap();
        let right = IntegerSparsemaxSummary::from_scores(&[1, 4, -3]).unwrap();
        let forward = IntegerSparsemaxSummary::compose(&left, &right)
            .unwrap()
            .solve_exact()
            .unwrap();
        let backward = IntegerSparsemaxSummary::compose(&right, &left)
            .unwrap()
            .solve_exact()
            .unwrap();
        assert_eq!(forward, backward);

        let mut combined = vec![4, 0, 4, 1, 4, -3];
        let direct = IntegerSparsemaxSummary::from_scores(&combined)
            .unwrap()
            .solve_exact()
            .unwrap();
        assert_eq!(forward, direct);
        assert_matches_expanded(&combined);
        combined.reverse();
        assert_matches_expanded(&combined);
    }

    #[test]
    fn equal_levels_are_atomic() {
        let solved = IntegerSparsemaxSummary::from_scores(&[3, 3, 3, 0, 0])
            .unwrap()
            .solve_exact()
            .unwrap();
        assert!(solved.levels.iter().all(|level| {
            level.probability.numerator() == 0 || level.count == 3 || level.count == 2
        }));
        let positive = solved
            .levels
            .iter()
            .filter(|level| level.probability.numerator() > 0)
            .count();
        assert_eq!(positive, 1);
        assert_eq!(solved.levels[0].count, 3);
        assert_matches_expanded(&[3, 0, 3, 0, 3]);
    }

    #[test]
    fn empty_parts_fail_closed() {
        assert!(IntegerSparsemaxSummary::from_scores(&[]).is_err());
        let part = IntegerSparsemaxSummary::from_scores(&[1]).unwrap();
        let empty = IntegerSparsemaxSummary {
            levels: BTreeMap::new(),
            total_count: 0,
        };
        assert!(IntegerSparsemaxSummary::compose(&part, &empty).is_err());
        assert!(empty.solve_exact().is_err());
    }
}
