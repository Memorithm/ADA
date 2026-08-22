#![forbid(unsafe_code)]

const BISECTION_STEPS: usize = 96;

/// A conservative numerical bracket around the unique alpha-entmax threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdBracket {
    pub lower: f64,
    pub upper: f64,
}

impl ThresholdBracket {
    #[must_use]
    pub fn midpoint(self) -> f64 {
        self.lower + (self.upper - self.lower) * 0.5
    }
}

/// Dense alpha-entmax probabilities and their normalization threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct EntmaxDistribution {
    pub probabilities: Vec<f64>,
    pub tau: f64,
}

/// Deterministic paged input for the ADA-A4 E0 laboratory.
///
/// `page_upper_bounds[p]` must upper-bound every score in page `p`. E0 validates
/// that condition against the stored scores because the full scores are present
/// as oracle data; a production implementation would obtain the bounds from
/// metadata without first loading the scores.
#[derive(Debug, Clone, PartialEq)]
pub struct EntmaxPagedCase {
    pub scores: Vec<f64>,
    pub page_size: usize,
    pub alpha: f64,
    pub page_upper_bounds: Vec<f64>,
}

impl EntmaxPagedCase {
    #[must_use]
    pub fn page_count(&self) -> usize {
        if self.page_size == 0 {
            0
        } else {
            self.scores.len().div_ceil(self.page_size)
        }
    }

    /// Validate the A4-E0 scalar research contract.
    ///
    /// # Errors
    ///
    /// Returns an error for empty/non-finite inputs, alpha outside `(1, 2]`, an
    /// invalid page size, a wrong number of page bounds, or a bound that is not
    /// conservative relative to the oracle scores.
    pub fn validate(&self) -> Result<(), &'static str> {
        validate_scores_alpha(&self.scores, self.alpha)?;
        if self.page_size == 0 {
            return Err("ADA-A4 page_size must be non-zero");
        }
        if self.page_upper_bounds.len() != self.page_count() {
            return Err("ADA-A4 requires exactly one upper bound per page");
        }
        if self
            .page_upper_bounds
            .iter()
            .any(|bound| !bound.is_finite())
        {
            return Err("ADA-A4 page upper bounds must be finite");
        }

        for (page, &bound) in self.page_upper_bounds.iter().enumerate() {
            let start = page * self.page_size;
            let end = (start + self.page_size).min(self.scores.len());
            let actual_max = self.scores[start..end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if bound < actual_max {
                return Err("ADA-A4 page upper bound is not conservative");
            }
        }
        Ok(())
    }
}

/// Logical work performed by the E0 branch-and-bound candidate.
///
/// These are algorithmic counters, not hardware traffic or instruction counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BranchAndBoundMetrics {
    pub pages_total: usize,
    pub pages_loaded: usize,
    pub pages_pruned: usize,
    pub scores_loaded: usize,
    pub threshold_solves: usize,
    pub rounds: usize,
}

/// Output of the A4-E0 branch-and-bound candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchAndBoundResult {
    pub distribution: EntmaxDistribution,
    pub loaded_pages: Vec<bool>,
    pub metrics: BranchAndBoundMetrics,
}

fn validate_scores_alpha(scores: &[f64], alpha: f64) -> Result<(), &'static str> {
    if scores.is_empty() {
        return Err("ADA-A4 requires at least one score");
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("ADA-A4 scores must be finite");
    }
    if !alpha.is_finite() || alpha <= 1.0 || alpha > 2.0 {
        return Err("ADA-A4 E0 requires finite alpha in (1, 2]");
    }
    Ok(())
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }

    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn objective(scores: &[f64], alpha: f64, tau: f64) -> f64 {
    let scale = alpha - 1.0;
    let exponent = scale.recip();
    scores.iter().fold(-1.0, |sum, &score| {
        let shifted = scale * score - tau;
        if shifted > 0.0 {
            sum + shifted.powf(exponent)
        } else {
            sum
        }
    })
}

/// Return a deterministic bisection bracket around the alpha-entmax threshold.
///
/// The initial interval `[m-1, m]`, where `m=(alpha-1) max(scores)`, is valid:
/// at the lower endpoint the maximum score contributes exactly one and at the
/// upper endpoint every contribution is zero. Bisection keeps `lower` on the
/// non-negative-objective side and `upper` on the non-positive side.
///
/// # Errors
///
/// Returns an error when scores/alpha violate the A4-E0 scalar contract.
#[must_use = "the threshold bracket is the certification state"]
pub fn entmax_threshold_bracket(
    scores: &[f64],
    alpha: f64,
) -> Result<ThresholdBracket, &'static str> {
    validate_scores_alpha(scores, alpha)?;
    let scale = alpha - 1.0;
    let max_scaled = scores
        .iter()
        .copied()
        .map(|score| scale * score)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut lower = max_scaled - 1.0;
    let mut upper = max_scaled;

    let mut lower_value = objective(scores, alpha, lower);

    // In exact arithmetic, max_scaled - 1 is always on the
    // non-negative-objective side because the maximum score contributes
    // exactly one. Binary64 subtraction can round the difference slightly
    // below one, making the computed objective spuriously negative by a few
    // ulps. Move the endpoint one representable value toward -infinity only
    // when that numerical invariant is violated.
    if lower_value < 0.0 {
        lower = next_down(lower);
        lower_value = objective(scores, alpha, lower);
    }

    if lower_value == 0.0 {
        return Ok(ThresholdBracket {
            lower,
            upper: lower,
        });
    }
    if lower_value < 0.0 {
        return Err("ADA-A4 numerical bracket invariant failed at lower endpoint");
    }

    for _ in 0..BISECTION_STEPS {
        let midpoint = lower + (upper - lower) * 0.5;
        if midpoint.to_bits() == lower.to_bits() || midpoint.to_bits() == upper.to_bits() {
            break;
        }
        let value = objective(scores, alpha, midpoint);
        if value >= 0.0 {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    Ok(ThresholdBracket { lower, upper })
}

fn probabilities_at_tau(scores: &[f64], alpha: f64, tau: f64) -> Vec<f64> {
    let scale = alpha - 1.0;
    let exponent = scale.recip();
    scores
        .iter()
        .map(|&score| {
            let shifted = scale * score - tau;
            if shifted > 0.0 {
                shifted.powf(exponent)
            } else {
                0.0
            }
        })
        .collect()
}

/// Independent dense scalar alpha-entmax oracle for A4-E0.
///
/// # Errors
///
/// Returns an error when scores/alpha violate the A4-E0 scalar contract.
#[must_use = "the dense oracle result should be checked"]
pub fn dense_entmax(scores: &[f64], alpha: f64) -> Result<EntmaxDistribution, &'static str> {
    let bracket = entmax_threshold_bracket(scores, alpha)?;
    let tau = bracket.midpoint();
    Ok(EntmaxDistribution {
        probabilities: probabilities_at_tau(scores, alpha, tau),
        tau,
    })
}

fn load_page(
    case: &EntmaxPagedCase,
    page: usize,
    loaded_pages: &mut [bool],
    loaded_indices: &mut Vec<usize>,
    metrics: &mut BranchAndBoundMetrics,
) {
    debug_assert!(!loaded_pages[page]);
    loaded_pages[page] = true;
    metrics.pages_loaded += 1;
    let start = page * case.page_size;
    let end = (start + case.page_size).min(case.scores.len());
    loaded_indices.extend(start..end);
    metrics.scores_loaded += end - start;
}

/// ADA-A4 E0 subset-threshold exact branch-and-bound candidate.
///
/// The candidate loads one page, solves entmax on the loaded subset, then uses
/// the lower endpoint of that subset threshold bracket to prune pages whose
/// conservative score upper bound cannot cross the entmax support threshold.
/// Unresolved pages are loaded in descending upper-bound order. Loose bounds
/// therefore degrade safely to the dense algorithm.
///
/// # Errors
///
/// Returns an error when the paged case violates the A4-E0 research contract or
/// when the scalar threshold solver cannot maintain its numerical bracket.
#[must_use = "the candidate result and logical work should be checked"]
pub fn branch_and_bound_entmax(
    case: &EntmaxPagedCase,
) -> Result<BranchAndBoundResult, &'static str> {
    case.validate()?;
    let page_count = case.page_count();
    let mut loaded_pages = vec![false; page_count];
    let mut pruned_pages = vec![false; page_count];
    let mut loaded_indices = Vec::with_capacity(case.scores.len());
    let mut metrics = BranchAndBoundMetrics {
        pages_total: page_count,
        ..BranchAndBoundMetrics::default()
    };

    let mut seed_page = 0;
    for (page, &bound) in case.page_upper_bounds.iter().enumerate().skip(1) {
        if bound > case.page_upper_bounds[seed_page] {
            seed_page = page;
        }
    }
    load_page(
        case,
        seed_page,
        &mut loaded_pages,
        &mut loaded_indices,
        &mut metrics,
    );

    loop {
        metrics.rounds += 1;
        let subset_scores: Vec<f64> = loaded_indices
            .iter()
            .map(|&index| case.scores[index])
            .collect();
        let bracket = entmax_threshold_bracket(&subset_scores, case.alpha)?;
        metrics.threshold_solves += 1;
        let tau_lower = bracket.lower;
        let scale = case.alpha - 1.0;

        let mut next_page = None;
        for (page, &bound) in case.page_upper_bounds.iter().enumerate() {
            if loaded_pages[page] || pruned_pages[page] {
                continue;
            }
            if scale * bound <= tau_lower {
                pruned_pages[page] = true;
                continue;
            }
            if next_page.is_none_or(|best| bound > case.page_upper_bounds[best]) {
                next_page = Some(page);
            }
        }

        if let Some(page) = next_page {
            load_page(
                case,
                page,
                &mut loaded_pages,
                &mut loaded_indices,
                &mut metrics,
            );
            continue;
        }

        metrics.pages_pruned = pruned_pages.iter().filter(|&&pruned| pruned).count();
        let subset_distribution = dense_entmax(&subset_scores, case.alpha)?;
        let mut probabilities = vec![0.0; case.scores.len()];
        for (&index, &probability) in loaded_indices
            .iter()
            .zip(subset_distribution.probabilities.iter())
        {
            probabilities[index] = probability;
        }
        return Ok(BranchAndBoundResult {
            distribution: EntmaxDistribution {
                probabilities,
                tau: subset_distribution.tau,
            },
            loaded_pages,
            metrics,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_page_bounds(scores: &[f64], page_size: usize) -> Vec<f64> {
        scores
            .chunks(page_size)
            .map(|page| page.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect()
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right}"
        );
    }

    fn assert_candidate_matches_dense(case: &EntmaxPagedCase) -> BranchAndBoundResult {
        let dense = dense_entmax(&case.scores, case.alpha).unwrap();
        let candidate = branch_and_bound_entmax(case).unwrap();
        assert_close(dense.tau, candidate.distribution.tau, 2.0e-12);
        for (&expected, &actual) in dense
            .probabilities
            .iter()
            .zip(candidate.distribution.probabilities.iter())
        {
            assert_close(expected, actual, 4.0e-12);
        }
        for (index, &probability) in dense.probabilities.iter().enumerate() {
            if probability > 1.0e-12 {
                assert!(
                    candidate.loaded_pages[index / case.page_size],
                    "dense support token {index} was placed in a pruned page"
                );
            }
        }
        candidate
    }

    #[test]
    fn repairs_sparsemax_lower_endpoint_rounding_conservatively() {
        // This value is derived from the Qwen3 E4 real-Q/K smoke trace.
        // For alpha=2, exact arithmetic gives score - (score - 1) == 1,
        // but binary64 subtraction evaluates the nominal endpoint slightly
        // below one and therefore produces a spuriously negative objective.
        let score = f64::from_bits(0xbfe6_5fb9_f9e6_0d95);
        let nominal_lower = score - 1.0;
        let nominal_value = objective(&[score], 2.0, nominal_lower);

        assert!(nominal_value < 0.0);

        let repaired = next_down(nominal_lower);
        assert!(repaired < nominal_lower);
        assert!(objective(&[score], 2.0, repaired) >= 0.0);

        let bracket = entmax_threshold_bracket(&[score], 2.0).unwrap();
        assert!(objective(&[score], 2.0, bracket.lower) >= 0.0);
        assert!(objective(&[score], 2.0, bracket.upper) <= 0.0);

        let dense = dense_entmax(&[score], 2.0).unwrap();
        assert_close(dense.probabilities[0], 1.0, 2.0e-15);
    }

    #[test]
    fn obvious_pages_are_pruned_for_sparsemax() {
        let scores = vec![10.0, 9.0, -20.0, -21.0, -30.0, -31.0];
        let case = EntmaxPagedCase {
            page_upper_bounds: exact_page_bounds(&scores, 2),
            scores,
            page_size: 2,
            alpha: 2.0,
        };
        let result = assert_candidate_matches_dense(&case);
        assert_eq!(result.metrics.pages_total, 3);
        assert_eq!(result.metrics.pages_loaded, 1);
        assert_eq!(result.metrics.pages_pruned, 2);
        assert_eq!(result.metrics.scores_loaded, 2);
    }

    #[test]
    fn obvious_pages_are_pruned_for_entmax15() {
        let scores = vec![8.0, 7.0, -20.0, -21.0, -30.0, -31.0];
        let case = EntmaxPagedCase {
            page_upper_bounds: exact_page_bounds(&scores, 2),
            scores,
            page_size: 2,
            alpha: 1.5,
        };
        let result = assert_candidate_matches_dense(&case);
        assert!(result.metrics.pages_loaded < result.metrics.pages_total);
    }

    #[test]
    fn near_threshold_page_is_not_pruned() {
        let scores = vec![4.0, 3.0, 3.1, -10.0];
        let case = EntmaxPagedCase {
            page_upper_bounds: exact_page_bounds(&scores, 2),
            scores,
            page_size: 2,
            alpha: 2.0,
        };
        let result = assert_candidate_matches_dense(&case);
        assert_eq!(result.metrics.pages_loaded, 2);
        assert!(result.distribution.probabilities[2] > 0.0);
    }

    #[test]
    fn loose_bounds_fall_back_to_dense_pages() {
        let scores = vec![3.0, 2.0, 1.0, 0.0, -1.0, -2.0];
        let page_count = scores.len().div_ceil(2);
        let case = EntmaxPagedCase {
            scores,
            page_size: 2,
            alpha: 1.5,
            page_upper_bounds: vec![100.0; page_count],
        };
        let result = assert_candidate_matches_dense(&case);
        assert_eq!(result.metrics.pages_loaded, result.metrics.pages_total);
        assert_eq!(result.metrics.pages_pruned, 0);
    }

    #[test]
    fn subset_thresholds_are_monotone_lower_bounds() {
        let scores = [5.0, 3.0, 2.0, 1.0, -4.0];
        for alpha in [1.5, 2.0] {
            let full_tau = entmax_threshold_bracket(&scores, alpha).unwrap().midpoint();
            let mut previous = f64::NEG_INFINITY;
            for prefix_len in 1..=scores.len() {
                let tau = entmax_threshold_bracket(&scores[..prefix_len], alpha)
                    .unwrap()
                    .midpoint();
                assert!(tau + 2.0e-12 >= previous);
                assert!(tau <= full_tau + 2.0e-12);
                previous = tau;
            }
        }
    }

    #[test]
    fn conservative_slack_bounds_preserve_exactness() {
        let scores = vec![6.0, 5.0, 0.5, 0.25, -4.0, -7.0, 1.0, -3.0];
        let mut bounds = exact_page_bounds(&scores, 2);
        for (page, bound) in bounds.iter_mut().enumerate() {
            if page % 2 == 0 {
                *bound += 0.375;
            } else {
                *bound += 1.25;
            }
        }
        for alpha in [1.5, 2.0] {
            let case = EntmaxPagedCase {
                scores: scores.clone(),
                page_size: 2,
                alpha,
                page_upper_bounds: bounds.clone(),
            };
            assert_candidate_matches_dense(&case);
        }
    }

    #[test]
    fn exhaustive_small_states_match_dense_oracle() {
        for len in 1..=5 {
            let mut state_count = 1_usize;
            for _ in 0..len {
                state_count *= 3;
            }
            for state in 0..state_count {
                let mut code = state;
                let mut scores = Vec::with_capacity(len);
                for _ in 0..len {
                    let score = match code % 3 {
                        0 => -2.0,
                        1 => 0.0,
                        _ => 2.0,
                    };
                    scores.push(score);
                    code /= 3;
                }
                let mut bounds = exact_page_bounds(&scores, 2);
                for (page, bound) in bounds.iter_mut().enumerate() {
                    if page % 2 == 1 {
                        *bound += 0.25;
                    }
                }
                for alpha in [1.5, 2.0] {
                    let case = EntmaxPagedCase {
                        scores: scores.clone(),
                        page_size: 2,
                        alpha,
                        page_upper_bounds: bounds.clone(),
                    };
                    assert_candidate_matches_dense(&case);
                }
            }
        }
    }
}
