//! Quantized lattice attention: a total attention map built from score
//! quantisation plus exact `alpha = 1 + 1/k` lattice entmax.
//!
//! Cycle #63 showed that the rational domain of [`crate::lattice_entmax_i64`]
//! collapses quickly as `k` grows, so the solver alone is not a usable
//! attention operator. This module tests the follow-up mechanism: round
//! real scores onto the integer lattice (after a positive scale and a
//! max-shift), run the exact solver, and if the rounded vector is outside
//! the rational domain apply a deterministic *gap completion* that forces
//! a unique maximum with gap at least `k`. The completion is always
//! defined after max-shift quantisation (the peak becomes `0`) and yields
//! one-hot exact-rational weights.
//!
//! The `scale` parameter amplifies scores before rounding. The integer
//! vector is then fed to lattice entmax as-is, so `scale > 1` is a
//! sharpness parameter rather than a finer reconstruction of the original
//! real-valued operator (a reconstructed `q / scale` solver is left open).
//!
//! Independent oracles: [`ada_a4_entmax_bnb::dense_entmax`] on the original
//! scores and on the quantized integers, plus a stable softmax used only
//! as a task baseline. Logical counters are not hardware measurements.
//! No usefulness, novelty, or FLAT adoption claim is made.

use crate::exact::ExactDistribution;
use crate::lattice::{LatticeEntmaxCost, lattice_entmax_i64_accounted};

/// How the operator produced a total distribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QlaPath {
    /// [`crate::lattice_entmax_i64`] accepted the quantized scores.
    LatticeHit,
    /// The quantized scores were outside the rational domain; gap completion
    /// produced a unique maximum of gap at least `k` and the solver then hit.
    GapCompleted,
}

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QlaCost {
    /// Scores passed through the quantizer.
    pub scores_quantized: usize,
    /// Calls to the inner lattice solver (1 on a hit, 2 after completion).
    pub lattice_solves: usize,
    /// Times gap completion ran (`0` or `1` per successful call).
    pub gap_completions: usize,
    /// Inner solver counters from the last successful solve.
    pub inner: LatticeEntmaxCost,
}

/// Total quantized-lattice-attention result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QlaDistribution {
    /// Exact rational weights, in input order.
    pub distribution: ExactDistribution,
    /// Integer scores after max-shift and rounding.
    pub quantized_scores: Vec<i64>,
    /// Whether the solver hit before or after gap completion.
    pub path: QlaPath,
    /// Logical counters.
    pub cost: QlaCost,
}

const MSG_EMPTY: &str = "ADA-A6 quantized lattice attention needs a non-empty score vector";
const MSG_K: &str = "ADA-A6 quantized lattice attention needs 1 <= k <= 16";
const MSG_SCALE: &str = "ADA-A6 quantized lattice attention needs a finite positive scale";
const MSG_NON_FINITE: &str = "ADA-A6 quantized lattice attention scores must be finite";
const MSG_OVERFLOW: &str = "ADA-A6 quantized lattice attention integer overflow";
const MSG_INNER: &str =
    "ADA-A6 quantized lattice attention gap completion did not enter the rational domain";

/// Quantized lattice attention with logical counters.
///
/// # Errors
///
/// Fails closed on an empty input, a `k` outside `1..=16`, a non-finite or
/// non-positive scale, a non-finite score, or an integer overflow. Gap
/// completion is part of the operator: being outside the raw rational
/// domain is not an error.
pub fn quantized_lattice_attention_accounted(
    scores: &[f64],
    scale: f64,
    k: u32,
) -> Result<QlaDistribution, &'static str> {
    if scores.is_empty() {
        return Err(MSG_EMPTY);
    }
    if !(1..=16).contains(&k) {
        return Err(MSG_K);
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(MSG_SCALE);
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(MSG_NON_FINITE);
    }

    let peak = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(MSG_EMPTY)?;
    let mut quantized = Vec::with_capacity(scores.len());
    for &score in scores {
        quantized.push(round_shifted(score, peak, scale)?);
    }
    let mut cost = QlaCost {
        scores_quantized: quantized.len(),
        ..QlaCost::default()
    };

    cost.lattice_solves += 1;
    if let Ok((distribution, inner)) = lattice_entmax_i64_accounted(&quantized, k) {
        cost.inner = inner;
        Ok(QlaDistribution {
            distribution,
            quantized_scores: quantized,
            path: QlaPath::LatticeHit,
            cost,
        })
    } else {
        let completed = gap_complete(&quantized, k)?;
        cost.gap_completions += 1;
        cost.lattice_solves += 1;
        let (distribution, inner) =
            lattice_entmax_i64_accounted(&completed, k).map_err(|_| MSG_INNER)?;
        cost.inner = inner;
        Ok(QlaDistribution {
            distribution,
            quantized_scores: completed,
            path: QlaPath::GapCompleted,
            cost,
        })
    }
}

/// As [`quantized_lattice_attention_accounted`], without counters.
///
/// # Errors
///
/// Same as [`quantized_lattice_attention_accounted`].
pub fn quantized_lattice_attention(
    scores: &[f64],
    scale: f64,
    k: u32,
) -> Result<QlaDistribution, &'static str> {
    quantized_lattice_attention_accounted(scores, scale, k)
}

/// Stable softmax baseline used as an independent task comparator.
///
/// # Errors
///
/// Fails closed on empty or non-finite input, or if the exponential mass
/// underflows to zero.
pub fn stable_softmax(scores: &[f64]) -> Result<Vec<f64>, &'static str> {
    if scores.is_empty() {
        return Err("ADA-A6 softmax baseline needs a non-empty score vector");
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("ADA-A6 softmax baseline scores must be finite");
    }
    let peak = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or("ADA-A6 softmax baseline needs a non-empty score vector")?;
    let mut weights: Vec<f64> = scores.iter().map(|score| (score - peak).exp()).collect();
    let mass: f64 = weights.iter().sum();
    if !(mass.is_finite() && mass > 0.0) {
        return Err("ADA-A6 softmax baseline mass underflowed");
    }
    for weight in &mut weights {
        *weight /= mass;
    }
    Ok(weights)
}

fn round_shifted(score: f64, peak: f64, scale: f64) -> Result<i64, &'static str> {
    let shifted = (score - peak) * scale;
    if !shifted.is_finite() {
        return Err(MSG_OVERFLOW);
    }
    let rounded = shifted.round();
    // `i64::MAX` is not an exact `f64`; reject anything outside `(-2^63, 2^63)`.
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&rounded) {
        return Err(MSG_OVERFLOW);
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(rounded as i64)
}

fn gap_complete(quantized: &[i64], k: u32) -> Result<Vec<i64>, &'static str> {
    let gap = i64::from(k);
    let mut peak_index = 0_usize;
    let mut peak_value = quantized[0];
    for (index, &value) in quantized.iter().enumerate() {
        if value > peak_value {
            peak_value = value;
            peak_index = index;
        }
    }
    let ceiling = peak_value.checked_sub(gap).ok_or(MSG_OVERFLOW)?;
    let mut completed = quantized.to_vec();
    for (index, slot) in completed.iter_mut().enumerate() {
        if index != peak_index && *slot > ceiling {
            *slot = ceiling;
        }
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::ExactRational;
    use crate::lattice_entmax_i64;
    use ada_a4_entmax_bnb::dense_entmax;

    fn to_f(rational: ExactRational) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let numerator = rational.numerator() as f64;
        #[allow(clippy::cast_precision_loss)]
        let denominator = rational.denominator() as f64;
        numerator / denominator
    }

    fn weights(distribution: &ExactDistribution) -> Vec<f64> {
        distribution
            .probabilities
            .iter()
            .copied()
            .map(to_f)
            .collect()
    }

    fn l1(left: &[f64], right: &[f64]) -> f64 {
        left.iter().zip(right).map(|(a, b)| (a - b).abs()).sum()
    }

    fn argmax(weights: &[f64]) -> usize {
        let mut best = 0_usize;
        let mut best_value = f64::NEG_INFINITY;
        for (index, &value) in weights.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        best
    }

    fn weighted_sum(masses: &[f64], values: &[f64]) -> f64 {
        masses
            .iter()
            .zip(values)
            .map(|(mass, value)| mass * value)
            .sum()
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.0
        }

        fn uniform(&mut self) -> f64 {
            let bits = self.next_u64() >> 11;
            #[allow(clippy::cast_precision_loss)]
            let mantissa = bits as f64;
            mantissa * (1.0 / 9_007_199_254_740_992.0)
        }

        fn gaussian(&mut self) -> f64 {
            let first = self.uniform().max(1.0e-16);
            let second = self.uniform();
            (-2.0 * first.ln()).sqrt() * (2.0 * std::f64::consts::PI * second).cos()
        }
    }

    fn assert_probability_vector(masses: &[f64]) {
        let total: f64 = masses.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1.0e-12,
            "mass {total} for {masses:?}"
        );
        assert!(masses.iter().all(|mass| *mass >= -1.0e-15));
    }

    fn i64_to_f64(score: i64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            score as f64
        }
    }

    fn usize_to_f64(value: usize) -> f64 {
        f64::from(u32::try_from(value).unwrap())
    }

    #[test]
    fn identity_on_in_domain_integers_matches_lattice_and_oracle() {
        let scores = [1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let outcome = quantized_lattice_attention(&scores, 1.0, 2).unwrap();
        assert_eq!(outcome.path, QlaPath::LatticeHit);
        assert_eq!(outcome.quantized_scores, vec![0, -1, -1, -1, -1]);
        let integers = vec![1_i64, 0, 0, 0, 0];
        let lattice = lattice_entmax_i64(&integers, 2).unwrap();
        assert_eq!(outcome.distribution.probabilities, lattice.probabilities);
        let oracle = dense_entmax(&scores, 1.5).unwrap();
        let l1_oracle = l1(&weights(&outcome.distribution), &oracle.probabilities);
        assert!(l1_oracle <= 1.0e-9, "l1 {l1_oracle}");
    }

    #[test]
    fn k1_never_needs_gap_completion_on_integer_grids() {
        for length in 1..=5_usize {
            let mut vector = vec![0_i64; length];
            loop {
                let scores: Vec<f64> = vector.iter().copied().map(i64_to_f64).collect();
                let outcome = quantized_lattice_attention(&scores, 1.0, 1).unwrap();
                assert_eq!(outcome.path, QlaPath::LatticeHit, "{vector:?}");
                assert_probability_vector(&weights(&outcome.distribution));
                let mut index = 0;
                while index < length {
                    if vector[index] < 3 {
                        vector[index] += 1;
                        break;
                    }
                    vector[index] = 0;
                    index += 1;
                }
                if index == length {
                    break;
                }
            }
        }
    }

    #[test]
    fn gap_completion_makes_the_operator_total() {
        // `[0, 0]` is outside the k = 2 rational domain; completion must one-hot
        // the first index (stable unique-max rule).
        let outcome = quantized_lattice_attention(&[0.0, 0.0], 1.0, 2).unwrap();
        assert_eq!(outcome.path, QlaPath::GapCompleted);
        assert_eq!(outcome.quantized_scores, vec![0, -2]);
        assert!((to_f(outcome.distribution.probabilities[0]) - 1.0).abs() <= 1.0e-15);
        assert!(to_f(outcome.distribution.probabilities[1]).abs() <= 1.0e-15);
        assert_eq!(outcome.cost.gap_completions, 1);
        assert_eq!(outcome.cost.lattice_solves, 2);
    }

    #[test]
    fn pair_interpolation_is_kept_by_sparsemax_and_destroyed_by_k2() {
        // Two equal peaks, two lower distractors. Values +1 and -1 at the peaks
        // should average to 0 when mass is shared, and jump to ±1 when the
        // operator collapses to a one-hot after gap completion.
        let scores = [2.0_f64, 2.0, 0.0, 0.0];
        let values = [1.0_f64, -1.0, 4.0, -4.0];
        let softmax = stable_softmax(&scores).unwrap();
        let sparsemax = dense_entmax(&scores, 2.0).unwrap();
        let entmax15 = dense_entmax(&scores, 1.5).unwrap();
        let qla1 = quantized_lattice_attention(&scores, 1.0, 1).unwrap();
        let qla2 = quantized_lattice_attention(&scores, 1.0, 2).unwrap();
        let qla3 = quantized_lattice_attention(&scores, 1.0, 3).unwrap();

        let softmax_out = weighted_sum(&softmax, &values);
        let sparsemax_out = weighted_sum(&sparsemax.probabilities, &values);
        let entmax15_out = weighted_sum(&entmax15.probabilities, &values);
        let qla1_out = weighted_sum(&weights(&qla1.distribution), &values);
        let qla2_out = weighted_sum(&weights(&qla2.distribution), &values);
        let qla3_out = weighted_sum(&weights(&qla3.distribution), &values);

        assert!(softmax_out.abs() <= 1.0e-12, "softmax {softmax_out}");
        assert!(sparsemax_out.abs() <= 1.0e-12, "sparsemax {sparsemax_out}");
        assert!(entmax15_out.abs() <= 1.0e-9, "entmax15 {entmax15_out}");
        assert_eq!(qla1.path, QlaPath::LatticeHit);
        assert!(qla1_out.abs() <= 1.0e-12, "qla k=1 {qla1_out}");
        assert_eq!(qla2.path, QlaPath::GapCompleted);
        assert!((qla2_out - 1.0).abs() <= 1.0e-12, "qla k=2 {qla2_out}");
        assert_eq!(qla3.path, QlaPath::GapCompleted);
        assert!((qla3_out - 1.0).abs() <= 1.0e-12, "qla k=3 {qla3_out}");
    }

    #[test]
    fn large_margin_needle_is_recovered_by_every_compared_operator() {
        let scores = [0.0_f64, 5.0, 0.2, -1.0, 0.1];
        let values = [0.0_f64, 1.0, 0.0, 0.0, 0.0];
        let softmax = stable_softmax(&scores).unwrap();
        let sparsemax = dense_entmax(&scores, 2.0).unwrap();
        let qla1 = quantized_lattice_attention(&scores, 1.0, 1).unwrap();
        let qla2 = quantized_lattice_attention(&scores, 1.0, 2).unwrap();
        assert!(weighted_sum(&softmax, &values) > 0.5);
        assert!(weighted_sum(&sparsemax.probabilities, &values) > 0.5);
        assert!(weighted_sum(&weights(&qla1.distribution), &values) > 0.5);
        assert!(weighted_sum(&weights(&qla2.distribution), &values) > 0.5);
        assert_eq!(argmax(&softmax), 1);
        assert_eq!(argmax(&weights(&qla2.distribution)), 1);
    }

    #[test]
    fn sub_bin_margin_erases_the_needle_for_qla_but_not_softmax_hit() {
        // Margin 0.2 at scale 1 rounds every shifted score to 0, so QLA sees a
        // four-way tie and (for k = 1) returns the uniform vector. Softmax
        // still ranks the unique maximum first.
        let scores = [0.0_f64, 0.2, 0.0, 0.0];
        let softmax = stable_softmax(&scores).unwrap();
        let qla1 = quantized_lattice_attention(&scores, 1.0, 1).unwrap();
        assert_eq!(argmax(&softmax), 1);
        assert_eq!(qla1.path, QlaPath::LatticeHit);
        assert!(outcome_support_uniform(&qla1));
        assert_eq!(argmax(&weights(&qla1.distribution)), 0);
    }

    fn outcome_support_uniform(outcome: &QlaDistribution) -> bool {
        let masses = weights(&outcome.distribution);
        let expected = 1.0 / f64::from(u32::try_from(masses.len()).unwrap());
        masses.iter().all(|mass| (mass - expected).abs() <= 1.0e-12)
    }

    #[test]
    fn translation_invariance_of_weights() {
        let base = [1.25_f64, -0.5, 0.75, 0.75];
        let shifted: Vec<f64> = base.iter().map(|score| score + 3.5).collect();
        for k in [1_u32, 2, 3] {
            let left = quantized_lattice_attention(&base, 2.0, k).unwrap();
            let right = quantized_lattice_attention(&shifted, 2.0, k).unwrap();
            assert_eq!(left.path, right.path, "k={k}");
            assert_eq!(
                left.distribution.probabilities, right.distribution.probabilities,
                "k={k}"
            );
        }
    }

    #[test]
    fn adversarial_inputs_fail_closed() {
        assert_eq!(
            quantized_lattice_attention(&[], 1.0, 2).unwrap_err(),
            MSG_EMPTY
        );
        assert_eq!(
            quantized_lattice_attention(&[1.0], 1.0, 0).unwrap_err(),
            MSG_K
        );
        assert_eq!(
            quantized_lattice_attention(&[1.0], 1.0, 17).unwrap_err(),
            MSG_K
        );
        assert_eq!(
            quantized_lattice_attention(&[1.0], 0.0, 2).unwrap_err(),
            MSG_SCALE
        );
        assert_eq!(
            quantized_lattice_attention(&[1.0], f64::NAN, 2).unwrap_err(),
            MSG_SCALE
        );
        assert_eq!(
            quantized_lattice_attention(&[f64::INFINITY], 1.0, 2).unwrap_err(),
            MSG_NON_FINITE
        );
        assert_eq!(
            quantized_lattice_attention(&[0.0, -1.0], 1.0e300, 2).unwrap_err(),
            MSG_OVERFLOW
        );
    }

    #[test]
    fn gaussian_survey_against_independent_oracles() {
        let (samples, length) = if cfg!(miri) {
            (8_usize, 4_usize)
        } else {
            (200, 8)
        };
        let scales = if cfg!(miri) {
            [1.0_f64, 4.0].as_slice()
        } else {
            [1.0, 2.0, 4.0, 8.0].as_slice()
        };
        let ks = if cfg!(miri) {
            [1_u32, 2].as_slice()
        } else {
            [1, 2, 3].as_slice()
        };

        for &scale in scales {
            for &k in ks {
                report_gaussian_cell(samples, length, scale, k);
            }
        }
    }

    fn report_gaussian_cell(samples: usize, length: usize, scale: f64, k: u32) {
        let mut hits = 0_usize;
        let mut completions = 0_usize;
        let mut oracle_disagree = 0_usize;
        let mut mean_l1_original = 0.0_f64;
        let mut mean_l1_quantized_on_hit = 0.0_f64;
        let mut hit_l1_count = 0_usize;
        let alpha = 1.0 + 1.0 / f64::from(k);
        let mut rng = Lcg(0xA6A6_0000_0000_0000 ^ (u64::from(k) << 32) ^ scale.to_bits());
        for _ in 0..samples {
            let scores: Vec<f64> = (0..length).map(|_| rng.gaussian()).collect();
            let outcome = quantized_lattice_attention(&scores, scale, k).unwrap();
            let masses = weights(&outcome.distribution);
            assert_probability_vector(&masses);
            match outcome.path {
                QlaPath::LatticeHit => hits += 1,
                QlaPath::GapCompleted => completions += 1,
            }
            let original_oracle = dense_entmax(&scores, alpha).unwrap();
            mean_l1_original += l1(&masses, &original_oracle.probabilities);
            if outcome.path == QlaPath::LatticeHit {
                let quantized_as_f: Vec<f64> = outcome
                    .quantized_scores
                    .iter()
                    .copied()
                    .map(i64_to_f64)
                    .collect();
                let quantized_oracle = dense_entmax(&quantized_as_f, alpha).unwrap();
                let dist = l1(&masses, &quantized_oracle.probabilities);
                if dist > 1.0e-9 {
                    oracle_disagree += 1;
                }
                mean_l1_quantized_on_hit += dist;
                hit_l1_count += 1;
            }
        }
        mean_l1_original /= usize_to_f64(samples);
        if hit_l1_count > 0 {
            mean_l1_quantized_on_hit /= usize_to_f64(hit_l1_count);
        }
        assert_eq!(hits + completions, samples);
        assert_eq!(oracle_disagree, 0, "k={k} scale={scale}");
        if !cfg!(miri) {
            println!(
                "qla-gaussian n={length} samples={samples} k={k} scale={scale} \
                 hits={hits} completions={completions} \
                 mean_l1_vs_unquantized_oracle={mean_l1_original:.6} \
                 mean_l1_vs_quantized_oracle_on_hit={mean_l1_quantized_on_hit:.6e}"
            );
        }
    }

    #[test]
    fn mechanistic_task_sweep_against_softmax_and_sparsemax() {
        if cfg!(miri) {
            return;
        }
        let margins = [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0];
        let task_scale = 1.0_f64;
        let length = 8_usize;
        for &margin in &margins {
            report_task_margin(margin, task_scale, length);
        }
    }

    fn report_task_margin(margin: f64, task_scale: f64, length: usize) {
        let mut needle = vec![0.0_f64; length];
        needle[3] = margin;
        let needle_values: Vec<f64> = (0..length)
            .map(|index| if index == 3 { 1.0 } else { 0.0 })
            .collect();
        let mut pair = vec![0.0_f64; length];
        pair[2] = margin;
        pair[5] = margin;
        let mut pair_values = vec![0.0_f64; length];
        pair_values[2] = 1.0;
        pair_values[5] = -1.0;

        let softmax_n = stable_softmax(&needle).unwrap();
        let sparse_n = dense_entmax(&needle, 2.0).unwrap();
        let softmax_p = stable_softmax(&pair).unwrap();
        let sparse_p = dense_entmax(&pair, 2.0).unwrap();
        let ent_p = dense_entmax(&pair, 1.5).unwrap();

        print_task(
            "needle",
            margin,
            &softmax_n,
            &needle_values,
            argmax(&softmax_n) == 3,
        );
        print_task(
            "needle-sparsemax",
            margin,
            &sparse_n.probabilities,
            &needle_values,
            argmax(&sparse_n.probabilities) == 3,
        );
        print_task(
            "pair-softmax",
            margin,
            &softmax_p,
            &pair_values,
            pair_interpolation_ok(&softmax_p),
        );
        print_task(
            "pair-sparsemax",
            margin,
            &sparse_p.probabilities,
            &pair_values,
            pair_interpolation_ok(&sparse_p.probabilities),
        );
        print_task(
            "pair-entmax15",
            margin,
            &ent_p.probabilities,
            &pair_values,
            pair_interpolation_ok(&ent_p.probabilities),
        );
        for k in 1_u32..=3 {
            let q_n = quantized_lattice_attention(&needle, task_scale, k).unwrap();
            let q_p = quantized_lattice_attention(&pair, task_scale, k).unwrap();
            let n_w = weights(&q_n.distribution);
            let p_w = weights(&q_p.distribution);
            print_task(
                &format!("needle-qla-k{k}"),
                margin,
                &n_w,
                &needle_values,
                argmax(&n_w) == 3,
            );
            print_task(
                &format!("pair-qla-k{k}-path={:?}", q_p.path),
                margin,
                &p_w,
                &pair_values,
                pair_interpolation_ok(&p_w),
            );
        }
    }

    fn pair_interpolation_ok(masses: &[f64]) -> bool {
        let decode = masses[2] - masses[5];
        decode.abs() <= 0.05 && masses[2] > 0.05 && masses[5] > 0.05
    }

    fn print_task(label: &str, margin: f64, masses: &[f64], values: &[f64], success: bool) {
        let decode = weighted_sum(masses, values);
        println!(
            "qla-task {label} margin={margin} decode={decode:.6} hit={success} path_mass_on_positive={:.6}",
            masses
                .iter()
                .zip(values)
                .filter(|(_, value)| **value > 0.0)
                .map(|(mass, _)| *mass)
                .sum::<f64>()
        );
    }
}
