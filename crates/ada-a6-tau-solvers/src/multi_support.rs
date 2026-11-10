//! Multi-support quantized lattice attention (MS-QLA).
//!
//! Cycle #65 showed that quantized lattice attention is total, but its
//! one-hot gap completion destroys tied-pair interpolation for `k >= 2`,
//! and fixed-scale rounding erases sub-bin unique needles. This module
//! tests the follow-up mechanism:
//!
//! 1. Optionally raise the scale so the gap between the top plateau and
//!    the next distinct score is at least half a rounding bin
//!    (`effective_scale = max(scale, 1/2 / gap)`). After max-shift,
//!    background scores then round to at most `-1` instead of collapsing
//!    onto the plateau.
//! 2. Quantize as in QLA and apply [`crate::lattice_entmax_i64`].
//! 3. On a miss, *plateau-complete*: keep every quantized argmax and push
//!    every other score down to at most `max - k`, then retry the solver.
//!    When the argmax set is a singleton this is QLA's one-hot completion.
//! 4. On a second miss, emit the exact uniform distribution on the
//!    quantized argmax set (exact sparsemax of a `{0, -1}` indicator).
//!    This path is required: for `k = 2` no integer vector has a
//!    two-point lattice-entmax support, so no in-domain score completion
//!    can preserve a tied pair.
//!
//! Independent oracles: A4 `dense_entmax` on original and quantized
//! scores (parity is required on lattice hits), A4 sparsemax on the
//! argmax indicator (parity on the uniform path), and a stable softmax
//! used only as a task baseline. ADA2 `CausalArgmaxRetrievalTask` and
//! `CopyTokenAttentionTask` grade hard retrieval of a unique needle;
//! tied-pair interpolation remains a separate soft-mix check because
//! those fixtures pick the first index on ties.
//! Logical counters are not hardware measurements. No usefulness,
//! novelty, or FLAT adoption claim is made.

use crate::exact::ExactDistribution;
use crate::lattice::{LatticeEntmaxCost, lattice_entmax_i64_accounted};
use crate::sparsemax_exact_i64;

/// How the operator produced a total distribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MsqlaPath {
    /// [`crate::lattice_entmax_i64`] accepted the quantized scores.
    LatticeHit,
    /// Plateau completion (all quantized maxima kept) entered the domain.
    PlateauCompleted,
    /// The completed plateau was still outside the rational domain; the
    /// operator returned the exact uniform distribution on the quantized
    /// argmax set.
    ArgmaxUniform,
}

/// Whether the scale is raised to protect the top-plateau gap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MsqlaScaleMode {
    /// Use the caller-supplied scale unchanged.
    Fixed,
    /// `effective_scale = max(scale, 1/2 / gap)` when a score sits
    /// strictly below the maximum.
    GapAdaptive,
}

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MsqlaCost {
    /// Scores passed through the quantizer.
    pub scores_quantized: usize,
    /// Times the scale was raised above the caller-supplied floor (`0` or `1`).
    pub scale_adaptations: usize,
    /// Calls to the inner lattice solver (`1` on a hit, `2` after a plateau try).
    pub lattice_solves: usize,
    /// Times plateau completion ran (`0` or `1` per successful call).
    pub plateau_completions: usize,
    /// Times the argmax-uniform fallback ran (`0` or `1` per successful call).
    pub argmax_uniforms: usize,
    /// Inner solver counters from the last successful lattice solve, if any.
    pub inner: LatticeEntmaxCost,
}

/// Total multi-support quantized-lattice-attention result.
#[derive(Clone, Debug, PartialEq)]
pub struct MsqlaDistribution {
    /// Exact rational weights, in input order.
    pub distribution: ExactDistribution,
    /// Integer scores after max-shift and rounding (before plateau completion).
    pub quantized_scores: Vec<i64>,
    /// Scale actually used for rounding.
    pub effective_scale: f64,
    /// Whether the solver hit, plateau-completed, or fell back to uniform.
    pub path: MsqlaPath,
    /// Logical counters.
    pub cost: MsqlaCost,
}

const MSG_EMPTY: &str = "ADA-A6 multi-support QLA needs a non-empty score vector";
const MSG_K: &str = "ADA-A6 multi-support QLA needs 1 <= k <= 16";
const MSG_SCALE: &str = "ADA-A6 multi-support QLA needs a finite positive scale";
const MSG_NON_FINITE: &str = "ADA-A6 multi-support QLA scores must be finite";
const MSG_OVERFLOW: &str = "ADA-A6 multi-support QLA integer overflow";

/// Multi-support quantized lattice attention with logical counters.
///
/// # Errors
///
/// Fails closed on an empty input, a `k` outside `1..=16`, a non-finite or
/// non-positive scale, a non-finite score, or an integer overflow. Being
/// outside the raw rational domain is not an error: plateau completion or
/// the argmax-uniform fallback makes the map total on finite inputs.
pub fn multi_support_quantized_lattice_attention_accounted(
    scores: &[f64],
    scale: f64,
    k: u32,
    scale_mode: MsqlaScaleMode,
) -> Result<MsqlaDistribution, &'static str> {
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

    let (effective_scale, adapted) = effective_scale(scores, scale, scale_mode)?;
    let peak = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(MSG_EMPTY)?;
    let mut quantized = Vec::with_capacity(scores.len());
    for &score in scores {
        quantized.push(round_shifted(score, peak, effective_scale)?);
    }
    let mut cost = MsqlaCost {
        scores_quantized: quantized.len(),
        scale_adaptations: usize::from(adapted),
        ..MsqlaCost::default()
    };

    cost.lattice_solves += 1;
    if let Ok((distribution, inner)) = lattice_entmax_i64_accounted(&quantized, k) {
        cost.inner = inner;
        return Ok(MsqlaDistribution {
            distribution,
            quantized_scores: quantized,
            effective_scale,
            path: MsqlaPath::LatticeHit,
            cost,
        });
    }

    let completed = plateau_complete(&quantized, k)?;
    cost.plateau_completions += 1;
    cost.lattice_solves += 1;
    if let Ok((distribution, inner)) = lattice_entmax_i64_accounted(&completed, k) {
        cost.inner = inner;
        return Ok(MsqlaDistribution {
            distribution,
            quantized_scores: quantized,
            effective_scale,
            path: MsqlaPath::PlateauCompleted,
            cost,
        });
    }

    let distribution = argmax_uniform(&quantized)?;
    cost.argmax_uniforms += 1;
    Ok(MsqlaDistribution {
        distribution,
        quantized_scores: quantized,
        effective_scale,
        path: MsqlaPath::ArgmaxUniform,
        cost,
    })
}

/// As [`multi_support_quantized_lattice_attention_accounted`], without
/// exposing a distinct entry point for counters.
///
/// # Errors
///
/// Same as [`multi_support_quantized_lattice_attention_accounted`].
pub fn multi_support_quantized_lattice_attention(
    scores: &[f64],
    scale: f64,
    k: u32,
    scale_mode: MsqlaScaleMode,
) -> Result<MsqlaDistribution, &'static str> {
    multi_support_quantized_lattice_attention_accounted(scores, scale, k, scale_mode)
}

fn effective_scale(
    scores: &[f64],
    scale: f64,
    mode: MsqlaScaleMode,
) -> Result<(f64, bool), &'static str> {
    if mode == MsqlaScaleMode::Fixed {
        return Ok((scale, false));
    }
    let peak = scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or(MSG_EMPTY)?;
    let mut next = f64::NEG_INFINITY;
    let mut has_below = false;
    for &score in scores {
        if score < peak && score > next {
            next = score;
            has_below = true;
        }
    }
    if !has_below {
        return Ok((scale, false));
    }
    let gap = peak - next;
    if !(gap.is_finite() && gap > 0.0) {
        return Err(MSG_OVERFLOW);
    }
    let needed = 0.5 / gap;
    if !needed.is_finite() {
        return Err(MSG_OVERFLOW);
    }
    let raised = scale.max(needed);
    Ok((raised, raised > scale))
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

fn plateau_complete(quantized: &[i64], k: u32) -> Result<Vec<i64>, &'static str> {
    let peak = *quantized.iter().max().ok_or(MSG_EMPTY)?;
    let ceiling = peak.checked_sub(i64::from(k)).ok_or(MSG_OVERFLOW)?;
    let mut completed = quantized.to_vec();
    for slot in &mut completed {
        if *slot < peak && *slot > ceiling {
            *slot = ceiling;
        }
    }
    Ok(completed)
}

fn argmax_uniform(quantized: &[i64]) -> Result<ExactDistribution, &'static str> {
    let peak = *quantized.iter().max().ok_or(MSG_EMPTY)?;
    let mut indicator = Vec::with_capacity(quantized.len());
    for &value in quantized {
        indicator.push(if value == peak { 0_i64 } else { -1 });
    }
    sparsemax_exact_i64(&indicator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::ExactRational;
    use crate::lattice_entmax_i64;
    use crate::quantized::{QlaPath, quantized_lattice_attention, stable_softmax};
    use ada_a4_entmax_bnb::dense_entmax;
    use ada_objective::{
        CausalArgmaxCandidate, CausalArgmaxRetrievalTask, CopyTokenAttentionTask,
        CopyTokenCandidate, CorrectnessStatus, LogicalCost,
    };

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

    fn argmax(masses: &[f64]) -> usize {
        let mut best = 0_usize;
        let mut best_value = f64::NEG_INFINITY;
        for (index, &value) in masses.iter().enumerate() {
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

    fn pair_interpolation_ok(masses: &[f64]) -> bool {
        let decode = masses[2] - masses[5];
        decode.abs() <= 0.05 && masses[2] > 0.05 && masses[5] > 0.05
    }

    fn logical_from(cost: &MsqlaCost) -> LogicalCost {
        LogicalCost {
            flops: None,
            qk_evaluations: Some(u64::try_from(cost.scores_quantized).unwrap_or(u64::MAX)),
            transcendental_operations: None,
            value_operations: Some(u64::try_from(cost.lattice_solves).unwrap_or(u64::MAX)),
        }
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

    #[test]
    fn identity_on_in_domain_integers_matches_lattice_and_oracle() {
        let scores = [1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let outcome =
            multi_support_quantized_lattice_attention(&scores, 1.0, 2, MsqlaScaleMode::Fixed)
                .unwrap();
        assert_eq!(outcome.path, MsqlaPath::LatticeHit);
        assert_eq!(outcome.quantized_scores, vec![0, -1, -1, -1, -1]);
        let integers = vec![1_i64, 0, 0, 0, 0];
        let lattice = lattice_entmax_i64(&integers, 2).unwrap();
        assert_eq!(outcome.distribution.probabilities, lattice.probabilities);
        let oracle = dense_entmax(&scores, 1.5).unwrap();
        let l1_oracle = l1(&weights(&outcome.distribution), &oracle.probabilities);
        assert!(l1_oracle <= 1.0e-9, "l1 {l1_oracle}");
    }

    #[test]
    fn k2_has_no_integer_two_point_lattice_support_on_the_tested_grid() {
        // Analytic companion: for two integer scores, D = 8 - (z1 - z2)^2,
        // which is a perfect square only at |z1 - z2| = 2 (D = 4), and then
        // the complementary inequalities exclude the lower score. So a
        // support of size exactly 2 never occurs for k = 2.
        let max_len = if cfg!(miri) { 3_usize } else { 5 };
        let max_score = if cfg!(miri) { 2_i64 } else { 4 };
        let mut two_point = 0_usize;
        let mut accepted = 0_usize;
        for length in 1..=max_len {
            let mut vector = vec![0_i64; length];
            loop {
                if let Ok(distribution) = lattice_entmax_i64(&vector, 2) {
                    accepted += 1;
                    if distribution.support_size == 2 {
                        two_point += 1;
                    }
                }
                let mut index = 0;
                while index < length {
                    if vector[index] < max_score {
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
        assert!(accepted > 0, "grid produced no k = 2 hits");
        assert_eq!(two_point, 0, "unexpected two-point k = 2 support");
    }

    #[test]
    fn two_way_tie_uses_argmax_uniform_instead_of_one_hot() {
        let scores = [0.0_f64, 0.0];
        let qla = quantized_lattice_attention(&scores, 1.0, 2).unwrap();
        assert_eq!(qla.path, QlaPath::GapCompleted);
        assert!((to_f(qla.distribution.probabilities[0]) - 1.0).abs() <= 1.0e-15);

        let outcome =
            multi_support_quantized_lattice_attention(&scores, 1.0, 2, MsqlaScaleMode::Fixed)
                .unwrap();
        assert_eq!(outcome.path, MsqlaPath::ArgmaxUniform);
        assert!((to_f(outcome.distribution.probabilities[0]) - 0.5).abs() <= 1.0e-15);
        assert!((to_f(outcome.distribution.probabilities[1]) - 0.5).abs() <= 1.0e-15);
        assert_eq!(outcome.cost.argmax_uniforms, 1);
        let indicator = [0.0_f64, 0.0];
        let oracle = dense_entmax(&indicator, 2.0).unwrap();
        assert!(l1(&weights(&outcome.distribution), &oracle.probabilities) <= 1.0e-9);
    }

    #[test]
    fn pair_interpolation_is_kept_by_msqla_and_destroyed_by_qla_k2() {
        let scores = [2.0_f64, 2.0, 0.0, 0.0];
        let values = [1.0_f64, -1.0, 4.0, -4.0];
        let softmax = stable_softmax(&scores).unwrap();
        let sparsemax = dense_entmax(&scores, 2.0).unwrap();
        let entmax15 = dense_entmax(&scores, 1.5).unwrap();
        let qla2 = quantized_lattice_attention(&scores, 1.0, 2).unwrap();
        let ms2 =
            multi_support_quantized_lattice_attention(&scores, 1.0, 2, MsqlaScaleMode::GapAdaptive)
                .unwrap();
        let ms3 =
            multi_support_quantized_lattice_attention(&scores, 1.0, 3, MsqlaScaleMode::GapAdaptive)
                .unwrap();

        let softmax_out = weighted_sum(&softmax, &values);
        let sparsemax_out = weighted_sum(&sparsemax.probabilities, &values);
        let entmax15_out = weighted_sum(&entmax15.probabilities, &values);
        let qla2_out = weighted_sum(&weights(&qla2.distribution), &values);
        let ms2_out = weighted_sum(&weights(&ms2.distribution), &values);
        let ms3_out = weighted_sum(&weights(&ms3.distribution), &values);

        assert!(softmax_out.abs() <= 1.0e-12, "softmax {softmax_out}");
        assert!(sparsemax_out.abs() <= 1.0e-12, "sparsemax {sparsemax_out}");
        assert!(entmax15_out.abs() <= 1.0e-9, "entmax15 {entmax15_out}");
        assert_eq!(qla2.path, QlaPath::GapCompleted);
        assert!((qla2_out - 1.0).abs() <= 1.0e-12, "qla k=2 {qla2_out}");
        assert_eq!(ms2.path, MsqlaPath::ArgmaxUniform);
        assert!(ms2_out.abs() <= 1.0e-12, "msqla k=2 {ms2_out}");
        assert_eq!(ms3.path, MsqlaPath::ArgmaxUniform);
        assert!(ms3_out.abs() <= 1.0e-12, "msqla k=3 {ms3_out}");
        assert_eq!(ms2.distribution.support_size, 2);
    }

    #[test]
    fn gap_adaptive_scale_recovers_a_sub_bin_needle() {
        let scores = [0.0_f64, 0.2, 0.0, 0.0];
        let fixed =
            multi_support_quantized_lattice_attention(&scores, 1.0, 1, MsqlaScaleMode::Fixed)
                .unwrap();
        let adaptive =
            multi_support_quantized_lattice_attention(&scores, 1.0, 1, MsqlaScaleMode::GapAdaptive)
                .unwrap();
        let softmax = stable_softmax(&scores).unwrap();
        assert_eq!(argmax(&softmax), 1);
        assert_eq!(argmax(&weights(&fixed.distribution)), 0);
        assert!(adaptive.effective_scale >= 2.5);
        assert_eq!(argmax(&weights(&adaptive.distribution)), 1);
        assert_eq!(adaptive.cost.scale_adaptations, 1);
    }

    #[test]
    fn unique_max_of_rounding_gap_one_plateau_completes_for_k2() {
        // After adaptive scale a unique needle typically becomes `[0, -1, ...]`.
        // For k = 2 that vector is outside the rational domain (gap 1 < 2), so
        // plateau completion must push the background to `-2` and one-hot.
        let scores = [0.0_f64, 0.4, 0.0, 0.0];
        let outcome =
            multi_support_quantized_lattice_attention(&scores, 1.0, 2, MsqlaScaleMode::GapAdaptive)
                .unwrap();
        assert_eq!(outcome.path, MsqlaPath::PlateauCompleted);
        assert_eq!(argmax(&weights(&outcome.distribution)), 1);
        assert!((to_f(outcome.distribution.probabilities[1]) - 1.0).abs() <= 1.0e-15);
    }

    #[test]
    fn translation_invariance_of_weights() {
        let base = [1.25_f64, -0.5, 0.75, 0.75];
        let shifted: Vec<f64> = base.iter().map(|score| score + 3.5).collect();
        for k in [1_u32, 2, 3] {
            for mode in [MsqlaScaleMode::Fixed, MsqlaScaleMode::GapAdaptive] {
                let left = multi_support_quantized_lattice_attention(&base, 2.0, k, mode).unwrap();
                let right =
                    multi_support_quantized_lattice_attention(&shifted, 2.0, k, mode).unwrap();
                assert_eq!(left.path, right.path, "k={k} mode={mode:?}");
                assert_eq!(
                    left.distribution.probabilities, right.distribution.probabilities,
                    "k={k} mode={mode:?}"
                );
            }
        }
    }

    #[test]
    fn adversarial_inputs_fail_closed() {
        assert_eq!(
            multi_support_quantized_lattice_attention(&[], 1.0, 2, MsqlaScaleMode::Fixed)
                .unwrap_err(),
            MSG_EMPTY
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(&[1.0], 1.0, 0, MsqlaScaleMode::Fixed)
                .unwrap_err(),
            MSG_K
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(&[1.0], 1.0, 17, MsqlaScaleMode::Fixed)
                .unwrap_err(),
            MSG_K
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(&[1.0], 0.0, 2, MsqlaScaleMode::Fixed)
                .unwrap_err(),
            MSG_SCALE
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(&[1.0], f64::NAN, 2, MsqlaScaleMode::Fixed)
                .unwrap_err(),
            MSG_SCALE
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(
                &[f64::INFINITY],
                1.0,
                2,
                MsqlaScaleMode::Fixed
            )
            .unwrap_err(),
            MSG_NON_FINITE
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(
                &[0.0, -1.0],
                1.0e300,
                2,
                MsqlaScaleMode::Fixed
            )
            .unwrap_err(),
            MSG_OVERFLOW
        );
        assert_eq!(
            multi_support_quantized_lattice_attention(
                &[0.0, -1.0e-320],
                1.0,
                2,
                MsqlaScaleMode::GapAdaptive
            )
            .unwrap_err(),
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
                for mode in [MsqlaScaleMode::Fixed, MsqlaScaleMode::GapAdaptive] {
                    report_gaussian_cell(samples, length, scale, k, mode);
                }
            }
        }
    }

    fn report_gaussian_cell(
        samples: usize,
        length: usize,
        scale: f64,
        k: u32,
        mode: MsqlaScaleMode,
    ) {
        let mut hits = 0_usize;
        let mut plateaus = 0_usize;
        let mut uniforms = 0_usize;
        let mut oracle_disagree = 0_usize;
        let mut uniform_disagree = 0_usize;
        let mut mean_l1_original = 0.0_f64;
        let mut adaptations = 0_usize;
        let alpha = 1.0 + 1.0 / f64::from(k);
        let mut rng = Lcg(0xA6A6_4D53_0000_0000_u64
            ^ (u64::from(k) << 32)
            ^ scale.to_bits()
            ^ u64::from(mode == MsqlaScaleMode::GapAdaptive));
        for _ in 0..samples {
            let scores: Vec<f64> = (0..length).map(|_| rng.gaussian()).collect();
            let outcome =
                multi_support_quantized_lattice_attention(&scores, scale, k, mode).unwrap();
            let masses = weights(&outcome.distribution);
            assert_probability_vector(&masses);
            adaptations += outcome.cost.scale_adaptations;
            match outcome.path {
                MsqlaPath::LatticeHit => hits += 1,
                MsqlaPath::PlateauCompleted => plateaus += 1,
                MsqlaPath::ArgmaxUniform => uniforms += 1,
            }
            let original_oracle = dense_entmax(&scores, alpha).unwrap();
            mean_l1_original += l1(&masses, &original_oracle.probabilities);
            if outcome.path == MsqlaPath::LatticeHit {
                let quantized_as_f: Vec<f64> = outcome
                    .quantized_scores
                    .iter()
                    .copied()
                    .map(i64_to_f64)
                    .collect();
                let quantized_oracle = dense_entmax(&quantized_as_f, alpha).unwrap();
                if l1(&masses, &quantized_oracle.probabilities) > 1.0e-9 {
                    oracle_disagree += 1;
                }
            }
            if outcome.path == MsqlaPath::ArgmaxUniform {
                let peak = *outcome.quantized_scores.iter().max().unwrap();
                let indicator: Vec<f64> = outcome
                    .quantized_scores
                    .iter()
                    .map(|&value| if value == peak { 0.0 } else { -1.0 })
                    .collect();
                let uniform_oracle = dense_entmax(&indicator, 2.0).unwrap();
                if l1(&masses, &uniform_oracle.probabilities) > 1.0e-9 {
                    uniform_disagree += 1;
                }
            }
        }
        mean_l1_original /= usize_to_f64(samples);
        assert_eq!(hits + plateaus + uniforms, samples);
        assert_eq!(oracle_disagree, 0, "k={k} scale={scale} mode={mode:?}");
        assert_eq!(uniform_disagree, 0, "k={k} scale={scale} mode={mode:?}");
        if !cfg!(miri) {
            println!(
                "msqla-gaussian n={length} samples={samples} k={k} scale={scale} \
                 mode={mode:?} hits={hits} plateaus={plateaus} uniforms={uniforms} \
                 adaptations={adaptations} mean_l1_vs_unquantized_oracle={mean_l1_original:.6}"
            );
        }
    }

    #[test]
    fn mechanistic_and_ada2_task_sweep() {
        if cfg!(miri) {
            return;
        }
        let margins = [0.1_f64, 0.4, 0.6, 1.0, 2.0, 4.0];
        let length = 8_usize;
        for &margin in &margins {
            report_task_margin(margin, length);
        }
    }

    fn report_task_margin(margin: f64, length: usize) {
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
        report_baseline_tasks(margin, &needle, &needle_values, &pair, &pair_values);
        report_operator_tasks(margin, &needle, &needle_values, &pair, &pair_values);
    }

    fn report_baseline_tasks(
        margin: f64,
        needle: &[f64],
        needle_values: &[f64],
        pair: &[f64],
        pair_values: &[f64],
    ) {
        let softmax_n = stable_softmax(needle).unwrap();
        let sparse_n = dense_entmax(needle, 2.0).unwrap();
        let softmax_p = stable_softmax(pair).unwrap();
        let sparse_p = dense_entmax(pair, 2.0).unwrap();
        let ent_p = dense_entmax(pair, 1.5).unwrap();
        print_task(
            "needle-softmax",
            margin,
            &softmax_n,
            needle_values,
            argmax(&softmax_n) == 3,
        );
        print_task(
            "needle-sparsemax",
            margin,
            &sparse_n.probabilities,
            needle_values,
            argmax(&sparse_n.probabilities) == 3,
        );
        print_task(
            "pair-softmax",
            margin,
            &softmax_p,
            pair_values,
            pair_interpolation_ok(&softmax_p),
        );
        print_task(
            "pair-sparsemax",
            margin,
            &sparse_p.probabilities,
            pair_values,
            pair_interpolation_ok(&sparse_p.probabilities),
        );
        print_task(
            "pair-entmax15",
            margin,
            &ent_p.probabilities,
            pair_values,
            pair_interpolation_ok(&ent_p.probabilities),
        );
        grade_ada2_needle("softmax", margin, needle, needle_values, &softmax_n, 0);
        grade_ada2_needle(
            "sparsemax",
            margin,
            needle,
            needle_values,
            &sparse_n.probabilities,
            0,
        );
    }

    fn report_operator_tasks(
        margin: f64,
        needle: &[f64],
        needle_values: &[f64],
        pair: &[f64],
        pair_values: &[f64],
    ) {
        for k in 1_u32..=3 {
            let q_n = quantized_lattice_attention(needle, 1.0, k).unwrap();
            let q_p = quantized_lattice_attention(pair, 1.0, k).unwrap();
            print_task(
                &format!("needle-qla-k{k}"),
                margin,
                &weights(&q_n.distribution),
                needle_values,
                argmax(&weights(&q_n.distribution)) == 3,
            );
            print_task(
                &format!("pair-qla-k{k}-path={:?}", q_p.path),
                margin,
                &weights(&q_p.distribution),
                pair_values,
                pair_interpolation_ok(&weights(&q_p.distribution)),
            );
            grade_ada2_needle(
                &format!("qla-k{k}"),
                margin,
                needle,
                needle_values,
                &weights(&q_n.distribution),
                q_n.cost.lattice_solves,
            );

            for mode in [MsqlaScaleMode::Fixed, MsqlaScaleMode::GapAdaptive] {
                let m_n = multi_support_quantized_lattice_attention(needle, 1.0, k, mode).unwrap();
                let m_p = multi_support_quantized_lattice_attention(pair, 1.0, k, mode).unwrap();
                let n_w = weights(&m_n.distribution);
                let p_w = weights(&m_p.distribution);
                print_task(
                    &format!("needle-msqla-k{k}-mode={mode:?}-path={:?}", m_n.path),
                    margin,
                    &n_w,
                    needle_values,
                    argmax(&n_w) == 3,
                );
                print_task(
                    &format!("pair-msqla-k{k}-mode={mode:?}-path={:?}", m_p.path),
                    margin,
                    &p_w,
                    pair_values,
                    pair_interpolation_ok(&p_w),
                );
                grade_ada2_needle(
                    &format!("msqla-k{k}-mode={mode:?}"),
                    margin,
                    needle,
                    needle_values,
                    &n_w,
                    m_n.cost.lattice_solves,
                );
            }
        }
    }

    fn grade_ada2_needle(
        label: &str,
        margin: f64,
        scores: &[f64],
        values: &[f64],
        masses: &[f64],
        lattice_solves: usize,
    ) {
        let selected_index = argmax(masses);
        let causal =
            CausalArgmaxRetrievalTask::new(scores.to_vec(), values.to_vec(), scores.len() - 1)
                .unwrap();
        let causal_grade = causal
            .grade(
                CausalArgmaxCandidate {
                    selected_index,
                    selected_value: values[selected_index],
                },
                logical_from(&MsqlaCost {
                    scores_quantized: scores.len(),
                    lattice_solves,
                    ..MsqlaCost::default()
                }),
            )
            .unwrap();
        let copy = CopyTokenAttentionTask::new(scores.to_vec(), values.to_vec()).unwrap();
        let copy_grade = copy
            .grade(
                CopyTokenCandidate {
                    selected_index,
                    selected_value: values[selected_index],
                },
                logical_from(&MsqlaCost {
                    scores_quantized: scores.len(),
                    lattice_solves,
                    ..MsqlaCost::default()
                }),
            )
            .unwrap();

        let hidden = CausalArgmaxRetrievalTask::new(scores.to_vec(), values.to_vec(), 2).unwrap();
        let hidden_visible = &scores[..=2];
        let hidden_softmax = stable_softmax(hidden_visible).unwrap();
        let hidden_choice = argmax(&hidden_softmax);
        let hidden_grade = hidden
            .grade(
                CausalArgmaxCandidate {
                    selected_index: hidden_choice,
                    selected_value: values[hidden_choice],
                },
                LogicalCost::default(),
            )
            .unwrap();

        println!(
            "ada2-needle {label} margin={margin} causal={:?} copy={:?} \
             causal_cut_query2={:?}",
            causal_grade.correctness(),
            copy_grade.correctness(),
            hidden_grade.correctness()
        );
        assert_eq!(hidden_grade.correctness(), CorrectnessStatus::Provisional);
    }

    fn print_task(label: &str, margin: f64, masses: &[f64], values: &[f64], success: bool) {
        let decode = weighted_sum(masses, values);
        println!(
            "msqla-task {label} margin={margin} decode={decode:.6} hit={success} \
             path_mass_on_positive={:.6}",
            masses
                .iter()
                .zip(values)
                .filter(|(_, value)| **value > 0.0)
                .map(|(mass, _)| *mass)
                .sum::<f64>()
        );
    }
}
