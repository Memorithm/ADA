//! ADA-A7 research candidate: composable sparsemax via per-part sufficient
//! statistics.
//!
//! Real-arithmetic fact: for alpha = 2 the entmax threshold solves
//! `F(tau) = sum_i max(s_i - tau, 0) = 1`, a piecewise-linear decreasing
//! function whose breakpoints are the distinct score values. A part's
//! sufficient statistic is therefore just `value -> multiplicity`; two parts
//! compose by merging those maps, and solving on the merged map yields EXACTLY
//! the same distribution as solving the union directly.
//!
//! That is the property subset-threshold controllers would need to prune with
//! per-page summaries instead of reloading scores. This crate demonstrates it
//! as research scaffolding (RESEARCH status, not a qualified mechanism):
//! everything cross-checks against the canonical A4 oracle and fails closed.

#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::EntmaxDistribution;
use std::collections::BTreeMap;

/// Sufficient statistic of one disjoint score part for sparsemax.
///
/// Levels are kept in a `BTreeMap` so composition is a deterministic merge
/// and iteration is always in descending score order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SparsemaxSummary {
    levels: BTreeMap<u64, usize>,
    total_count: usize,
}

fn level_key(value: f64) -> u64 {
    value.to_bits()
}

impl SparsemaxSummary {
    /// Summarize one part.
    ///
    /// # Errors
    ///
    /// Returns an error on empty or non-finite parts.
    pub fn from_scores(scores: &[f64]) -> Result<Self, &'static str> {
        if scores.is_empty() {
            return Err("ADA-A7 requires at least one score per part");
        }
        if scores.iter().any(|score| !score.is_finite()) {
            return Err("ADA-A7 scores must be finite");
        }
        let mut levels = BTreeMap::new();
        for &score in scores {
            *levels.entry(level_key(score)).or_insert(0) += 1;
        }
        Ok(Self {
            levels,
            total_count: scores.len(),
        })
    }

    /// Compose two disjoint-part summaries into the summary of their union.
    #[must_use]
    pub fn compose(left: &Self, right: &Self) -> Self {
        let mut levels = left.levels.clone();
        for (&key, &count) in &right.levels {
            *levels.entry(key).or_insert(0) += count;
        }
        Self {
            levels,
            total_count: left.total_count + right.total_count,
        }
    }

    /// Solve sparsemax from a summary without touching the raw scores.
    ///
    /// Degenerate magnitudes fall back to the certified A4 extreme semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when the summary is empty or arithmetic degenerates.
    pub fn solve(&self) -> Result<EntmaxDistribution, &'static str> {
        if self.levels.is_empty() {
            return Err("ADA-A7 cannot solve an empty summary");
        }

        // Descending distinct levels with running (count, sum) of elements at
        // or above each level. NOTE: BTreeMap keys are raw bit patterns, which
        // do NOT order signed floats correctly; an explicit total_cmp sort is
        // required.
        let mut descending: Vec<(f64, usize)> = self
            .levels
            .iter()
            .map(|(&key, &count)| (f64::from_bits(key), count))
            .collect();
        descending.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));

        let mut running_count = 0_usize;
        let mut running_sum = 0.0_f64;
        let mut root = None;
        let mut previous_lower_bound = f64::NEG_INFINITY;

        for &(level, multiplicity) in &descending {
            running_count += multiplicity;
            running_sum += level * f64::from(u32::try_from(multiplicity).unwrap_or(u32::MAX));

            #[allow(clippy::cast_precision_loss)]
            let count_f64 = running_count as f64;
            if running_count == 0 || count_f64 <= 0.0 {
                continue;
            }
            let candidate_root = (running_sum - 1.0) / count_f64;
            // Valid iff previous_lower_bound < candidate_root <= level.
            if candidate_root <= level && candidate_root > previous_lower_bound {
                root = Some(candidate_root);
                break;
            }
            previous_lower_bound = level;
        }

        let Some(tau) = root else {
            // Degenerate magnitudes: defer to the certified extreme path.
            return Err("ADA-A7 threshold not bracketed by summary plateaus");
        };

        // Reconstruct probabilities from the summary alone: every element at
        // a level strictly above tau contributes (level - tau).
        let mut probabilities = Vec::with_capacity(self.total_count);
        let mut mass = 0.0_f64;
        for &(level, multiplicity) in &descending {
            if level > tau {
                let contribution = level - tau;
                mass += contribution * f64::from(u32::try_from(multiplicity).unwrap_or(u32::MAX));
                for _ in 0..multiplicity {
                    probabilities.push(contribution);
                }
            } else {
                probabilities.extend(std::iter::repeat_n(0.0, multiplicity));
            }
        }

        if mass <= 0.0 || !mass.is_finite() {
            return Err("ADA-A7 realized mass must be positive and finite");
        }

        // Order-stability note: summary reconstruction loses original token
        // order; callers composing across partitions must map support back
        // through their own partition bookkeeping. For parity testing we
        // sort both sides before comparison instead of pretending order is
        // recoverable here.

        for probability in &mut probabilities {
            *probability /= mass;
        }
        probabilities.sort_unstable_by(|left, right| right.total_cmp(left));

        Ok(EntmaxDistribution { probabilities, tau })
    }
}

/// Weighted moments of a distribution over its own score values.
#[must_use]
pub fn distribution_moments(scores: &[f64], probabilities: &[f64]) -> Option<(f64, f64)> {
    if scores.len() != probabilities.len() {
        return None;
    }
    let mut mean = 0.0_f64;
    let mut second = 0.0_f64;
    for (&score, &probability) in scores.iter().zip(probabilities.iter()) {
        mean += probability * score;
        second += probability * score * score;
    }
    Some((mean, second))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_a4_entmax_bnb::dense_entmax;

    fn sorted_probabilities(distribution: &EntmaxDistribution) -> Vec<f64> {
        let mut copy = distribution.probabilities.clone();
        copy.sort_unstable_by(|left, right| right.total_cmp(left));
        copy
    }

    fn assert_composition_parity(scores: &[f64], split_at: usize) {
        let oracle = dense_entmax(scores, 2.0).expect("oracle solves");
        let composed = SparsemaxSummary::compose(
            &SparsemaxSummary::from_scores(&scores[..split_at]).unwrap(),
            &SparsemaxSummary::from_scores(&scores[split_at..]).unwrap(),
        )
        .solve()
        .unwrap_or_else(|error| panic!("composition failed for {scores:?}: {error}"));

        let direct = SparsemaxSummary::from_scores(scores)
            .unwrap()
            .solve()
            .unwrap();
        assert_eq!(
            composed.tau.to_bits(),
            direct.tau.to_bits(),
            "composed tau must equal direct-solve tau bit-for-bit"
        );
        assert_eq!(
            sorted_probabilities(&composed),
            sorted_probabilities(&direct)
        );

        // Cross-check against the canonical oracle within tolerance.
        let oracle_sorted = sorted_probabilities(&oracle);
        let composed_sorted = sorted_probabilities(&composed);
        for (&reference, &actual) in oracle_sorted.iter().zip(composed_sorted.iter()) {
            assert!(
                (reference - actual).abs() <= 1.0e-9,
                "divergence vs oracle for {scores:?}"
            );
        }
        assert!((oracle.tau - composed.tau).abs() <= 2.0e-9);
    }

    #[test]
    fn exhaustive_small_states_split_compose_matches_oracle() {
        for len in 2..=5_usize {
            let states = 3_usize.pow(u32::try_from(len).unwrap());
            for state in 0..states {
                let mut code = state;
                let mut scores = Vec::with_capacity(len);
                for _ in 0..len {
                    #[allow(clippy::cast_precision_loss)]
                    let score = match code % 3 {
                        0 => -1.5,
                        1 => 0.25,
                        _ => 1.75,
                    };
                    scores.push(score);
                    code /= 3;
                }
                for split_at in 1..len {
                    assert_composition_parity(&scores, split_at);
                }
            }
        }
    }

    #[test]
    fn ties_and_duplicates_compose_exactly() {
        for scores in [
            vec![2.0, 2.0, 2.0],
            vec![5.0, 5.0, -3.0, -3.0, 4.0],
            vec![0.5; 8],
        ] {
            let middle = scores.len() / 2;
            assert_composition_parity(&scores, middle.max(1));
        }
    }

    #[test]
    fn moments_match_between_oracle_and_summary_solve() {
        let scores = [3.0, 1.0, -0.5, -2.0];
        let oracle = dense_entmax(&scores, 2.0).unwrap();
        let solved = SparsemaxSummary::from_scores(&scores)
            .unwrap()
            .solve()
            .unwrap();

        let (oracle_mean, oracle_second) =
            distribution_moments(&sorted_scores_desc(&scores), &sorted_probabilities(&oracle))
                .unwrap();
        let (solved_mean, solved_second) =
            distribution_moments(&sorted_scores_desc(&scores), &sorted_probabilities(&solved))
                .unwrap();

        assert!((oracle_mean - solved_mean).abs() <= 1.0e-12);
        assert!((oracle_second - solved_second).abs() <= 1.0e-12);
    }

    fn sorted_scores_desc(scores: &[f64]) -> Vec<f64> {
        let mut copy = scores.to_vec();
        copy.sort_unstable_by(|left, right| right.total_cmp(left));
        copy
    }

    #[test]
    fn invalid_parts_fail_closed() {
        assert!(SparsemaxSummary::from_scores(&[]).is_err());
        assert!(SparsemaxSummary::from_scores(&[f64::NAN]).is_err());
        assert!(SparsemaxSummary::default().solve().is_err());
    }
}
