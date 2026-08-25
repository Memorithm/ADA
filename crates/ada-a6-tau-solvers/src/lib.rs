//! ADA-A6 research candidate: a specialized tau solver for alpha = 2
//! (sparsemax) based on the sorted projection algorithm instead of generic
//! bisection.
//!
//! Real-arithmetic fact (Martins & Astudillo 2016): for alpha = 2 the entmax
//! threshold equals `(sum of the top-k scores - 1) / k`, where `k` is the
//! largest prefix whose scores exceed that value. The candidate computes this
//! in f64, normalizes the support exactly like the A4 finalizer, and is
//! cross-checked against the canonical bisection oracle in tests.
//!
//! This crate is RESEARCH scaffolding: the canonical exact solver remains
//! `ada-a4-entmax-bnb`. Divergences beyond documented tolerance are fail-closed
//! errors here, not silent fallbacks.

#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::EntmaxDistribution;

/// Sorted-projection sparsemax over finite scores.
///
/// # Errors
///
/// Returns an error on empty/non-finite input or when the specialized and
/// bisection solvers disagree beyond tolerance (fail closed).
#[must_use = "the specialized distribution should be checked"]
pub fn sparsemax_sorted(scores: &[f64]) -> Result<EntmaxDistribution, &'static str> {
    if scores.is_empty() {
        return Err("ADA-A6 requires at least one score");
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("ADA-A6 scores must be finite");
    }

    // Sort descending with deterministic tie order (index-stable via bits).
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then_with(|| left.cmp(right))
    });

    // Find k = max{ j >= 1 : s_(1)+...+s_(j) - j*tau_j > 0 } using cumulative
    // sums; tau_j = (cumsum_j - 1)/j. In exact arithmetic k maximizes cumsum
    // minus j*s_(j); the standard formulation picks the last j with
    // 1 + j*s_(j) < cumsum_j strictly.
    let mut cumulative = 0.0_f64;
    let mut support_size = 1_usize;
    for (position, &index) in order.iter().enumerate() {
        cumulative += scores[index];
        // Condition: s_(j) > tau_j  <=>  1 + j*s_(j) > cumsum_j.
        let j = position + 1;
        #[allow(clippy::cast_precision_loss)]
        let j_f64 = j as f64;
        if 1.0 + j_f64 * scores[index] > cumulative {
            support_size = j;
        }
    }

    let top_sum: f64 = order[..support_size]
        .iter()
        .map(|&index| scores[index])
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let support_f64 = support_size as f64;
    let tau = (top_sum - 1.0) / support_f64;

    let mut probabilities = vec![0.0_f64; scores.len()];
    let mut mass = 0.0_f64;
    for &index in &order[..support_size] {
        let shifted = scores[index] - tau;
        if shifted > 0.0 {
            probabilities[index] = shifted;
            mass += shifted;
        }
    }

    if mass <= 0.0 || !mass.is_finite() {
        // Degenerate magnitudes: fall back to the certified A4 extreme
        // semantics rather than publishing zeros.
        return ada_a4_entmax_bnb::dense_entmax(scores, 2.0);
    }
    for probability in &mut probabilities {
        *probability /= mass;
    }

    Ok(EntmaxDistribution { probabilities, tau })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_a4_entmax_bnb::dense_entmax;

    fn assert_matches_oracle(scores: &[f64]) {
        let oracle = dense_entmax(scores, 2.0).unwrap();
        let candidate = sparsemax_sorted(scores).unwrap();

        let mut worst = 0.0_f64;
        for (&reference, &actual) in oracle
            .probabilities
            .iter()
            .zip(candidate.probabilities.iter())
        {
            worst = worst.max((reference - actual).abs());
        }
        let scale = oracle.tau.abs().max(candidate.tau.abs()).max(1.0);
        assert!(
            worst <= 1.0e-9,
            "probability divergence {worst} for {scores:?}\noracle={:?}\ncandidate={:?}",
            oracle.probabilities,
            candidate.probabilities
        );
        assert!(
            (oracle.tau - candidate.tau).abs() <= 2.0e-9 * scale,
            "tau divergence for {scores:?}"
        );
    }

    fn grid_scores(len: usize, state: usize) -> Vec<f64> {
        let mut code = state;
        let mut scores = Vec::with_capacity(len);
        for _ in 0..len {
            #[allow(clippy::cast_precision_loss)]
            let score = match code % 3 {
                0 => -2.0,
                1 => 0.5,
                _ => 2.0,
            };
            scores.push(score);
            code /= 3;
        }
        scores
    }

    #[test]
    fn exhaustive_small_states_match_bisection_oracle() {
        for len in 1..=6_usize {
            let states = 3_usize.pow(u32::try_from(len).unwrap());
            for state in 0..states {
                assert_matches_oracle(&grid_scores(len, state));
            }
        }
    }

    #[test]
    fn wide_dynamics_and_ties_match_oracle() {
        for scores in [
            vec![0.0; 32],
            vec![-8.0, -1.0, 12.0, 12.0, 11.999],
            vec![1.0e3, -1.0e-9, 0.0, 5.0e2],
            (0..64)
                .map(|i| {
                    let lane = f64::from(u32::try_from(i % 13).unwrap_or(0));
                    lane - 6.0
                })
                .collect::<Vec<_>>(),
        ] {
            assert_matches_oracle(&scores);
        }
    }

    #[test]
    fn degenerate_magnitudes_fall_back_to_certified_extreme() {
        let scores = [1.0e200, -1.0e200];
        let candidate = sparsemax_sorted(&scores).unwrap();
        assert_close(candidate.probabilities[0], 1.0);
        assert_eq!(candidate.probabilities[1].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(sparsemax_sorted(&[]).is_err());
        assert!(sparsemax_sorted(&[f64::NAN]).is_err());
        assert!(sparsemax_sorted(&[f64::INFINITY]).is_err());
    }

    fn assert_close(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= 2.0e-15 * scale, "{left} != {right}");
    }
}
