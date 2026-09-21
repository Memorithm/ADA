//! Strengthened / exact reference paths for streaming online Softmax.
//!
//! Complements the naive f32 [`crate::online_softmax_baseline`] with:
//! 1. Exact singleton path: output equals the sole value row; LSE equals the
//!    sole logit (bit-preserving through f64 then finite f32 narrowing).
//! 2. Exact equal-logit path: uniform weights `1/n` when every logit is
//!    bit-identical, Neumaier-compensated value mean, LSE = m + ln(n).
//! 3. Otherwise: f64 online Softmax recurrence with Neumaier-compensated mass
//!    and value accumulation, fail-closed on non-finite intermediates.
//!
//! Results are narrowed to [`AttentionResult`]'s f32 fields for API parity.
//! These paths are higher-assurance reference helpers for differential checks.
//! They do not claim usefulness, novelty, or FLAT adoption.

use ada_core::{AttentionCase, AttentionResult, LogicalMetrics};

/// Largest magnitude with exact integer representation in binary64.
const EXACT_F64_INT_MAX: i128 = 1_i128 << 53;

/// Strengthened streaming online Softmax reference.
///
/// # Errors
///
/// Returns an error when the case fails [`AttentionCase::validate`], when an
/// intermediate is non-finite, when exact uniform weight `1/n` cannot be formed,
/// or when the final f32 narrowing is non-finite.
pub fn online_softmax_strengthened(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    case.validate()?;

    if case.logits.len() == 1 {
        return exact_singleton(case);
    }

    if all_logits_bit_equal(&case.logits) {
        return exact_equal_logits(case);
    }

    streaming_neumaier(case)
}

fn exact_singleton(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    let score = f64::from(case.logits[0]);
    if !score.is_finite() {
        return Err("strengthened online Softmax: non-finite logit");
    }
    let mut output = Vec::with_capacity(case.head_dim);
    for &v in &case.values[..case.head_dim] {
        let value = f64::from(v);
        if !value.is_finite() {
            return Err("strengthened online Softmax: non-finite value");
        }
        output.push(finite_f32(value, "singleton output")?);
    }
    Ok(AttentionResult {
        output,
        lse: finite_f32(score, "singleton LSE")?,
        metrics: LogicalMetrics {
            qk_pairs_evaluated: 1,
            exp_evaluations: 0,
            log_evaluations: 0,
            value_accumulate_elements: case.head_dim,
        },
    })
}

fn exact_equal_logits(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    let n = case.logits.len();
    let weight = exact_uniform_weight(n)?;
    let maximum = f64::from(case.logits[0]);
    if !maximum.is_finite() {
        return Err("strengthened online Softmax: non-finite logit");
    }
    let count_f64 = usize_to_exact_f64(n)?;
    let log_sum_exp = maximum + count_f64.ln();
    if !log_sum_exp.is_finite() {
        return Err("strengthened online Softmax: non-finite LSE");
    }

    let mut output = Vec::with_capacity(case.head_dim);
    for lane in 0..case.head_dim {
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;
        for key in 0..n {
            let value = f64::from(case.values[key * case.head_dim + lane]);
            if !value.is_finite() {
                return Err("strengthened online Softmax: non-finite value");
            }
            let term = weight * value;
            if !term.is_finite() {
                return Err("strengthened online Softmax: non-finite value mix");
            }
            let (next, next_comp) = neumaier_add(sum, compensation, term);
            sum = next;
            compensation = next_comp;
        }
        let mixed = sum + compensation;
        if !mixed.is_finite() {
            return Err("strengthened online Softmax: non-finite value mix");
        }
        output.push(finite_f32(mixed, "equal-logit output")?);
    }

    Ok(AttentionResult {
        output,
        lse: finite_f32(log_sum_exp, "equal-logit LSE")?,
        metrics: LogicalMetrics {
            qk_pairs_evaluated: n,
            exp_evaluations: 0,
            log_evaluations: 1,
            value_accumulate_elements: n * case.head_dim,
        },
    })
}

fn streaming_neumaier(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    let head_dim = case.head_dim;
    let mut running_max = f64::NEG_INFINITY;
    let mut running_sum = 0.0_f64;
    let mut running_sum_comp = 0.0_f64;
    let mut numerator = vec![0.0_f64; head_dim];
    let mut numerator_comp = vec![0.0_f64; head_dim];
    let mut metrics = LogicalMetrics::default();

    for (key, &score_f32) in case.logits.iter().enumerate() {
        metrics.qk_pairs_evaluated += 1;
        let score = f64::from(score_f32);
        if !score.is_finite() {
            return Err("strengthened online Softmax: non-finite logit");
        }
        let new_max = running_max.max(score);
        let alpha = if running_max.is_finite() {
            metrics.exp_evaluations += 1;
            let a = (running_max - new_max).exp();
            if !a.is_finite() {
                return Err("strengthened online Softmax: non-finite rescale");
            }
            a
        } else {
            0.0
        };
        metrics.exp_evaluations += 1;
        let probability_numerator = (score - new_max).exp();
        if !probability_numerator.is_finite() {
            return Err("strengthened online Softmax: non-finite exp");
        }

        // Rescale compensated mass, then Neumaier-add the new contribution.
        running_sum *= alpha;
        running_sum_comp *= alpha;
        let (next_sum, next_sum_comp) =
            neumaier_add(running_sum, running_sum_comp, probability_numerator);
        running_sum = next_sum;
        running_sum_comp = next_sum_comp;

        let value = &case.values[key * head_dim..(key + 1) * head_dim];
        for lane in 0..head_dim {
            let v = f64::from(value[lane]);
            if !v.is_finite() {
                return Err("strengthened online Softmax: non-finite value");
            }
            let term = probability_numerator * v;
            if !term.is_finite() {
                return Err("strengthened online Softmax: non-finite value mix");
            }
            numerator[lane] *= alpha;
            numerator_comp[lane] *= alpha;
            let (next, next_comp) = neumaier_add(numerator[lane], numerator_comp[lane], term);
            numerator[lane] = next;
            numerator_comp[lane] = next_comp;
            metrics.value_accumulate_elements += 1;
        }
        running_max = new_max;
    }

    let mass = running_sum + running_sum_comp;
    if !mass.is_finite() || mass <= 0.0 {
        return Err("strengthened online Softmax: non-finite normalizer");
    }
    let inv_sum = mass.recip();
    if !inv_sum.is_finite() {
        return Err("strengthened online Softmax: non-finite normalizer");
    }

    let mut output = Vec::with_capacity(head_dim);
    for lane in 0..head_dim {
        let mixed = (numerator[lane] + numerator_comp[lane]) * inv_sum;
        if !mixed.is_finite() {
            return Err("strengthened online Softmax: non-finite output");
        }
        output.push(finite_f32(mixed, "streaming output")?);
    }

    metrics.log_evaluations += 1;
    let lse = running_max + mass.ln();
    if !lse.is_finite() {
        return Err("strengthened online Softmax: non-finite LSE");
    }

    Ok(AttentionResult {
        output,
        lse: finite_f32(lse, "streaming LSE")?,
        metrics,
    })
}

/// Exact uniform weight `1/n` when representable as a finite f64.
///
/// # Errors
///
/// Fails closed when `count` is zero, exceeds the exact binary64 integer range,
/// or `1/count` is not finite.
pub fn exact_uniform_weight(count: usize) -> Result<f64, &'static str> {
    if count == 0 {
        return Err("strengthened online Softmax: empty logit sequence");
    }
    let count_f64 = usize_to_exact_f64(count)?;
    let weight = 1.0_f64 / count_f64;
    if !weight.is_finite() {
        return Err("strengthened online Softmax: non-finite uniform weight");
    }
    Ok(weight)
}

fn all_logits_bit_equal(logits: &[f32]) -> bool {
    let Some(first) = logits.first() else {
        return true;
    };
    let bits = first.to_bits();
    logits.iter().all(|logit| logit.to_bits() == bits)
}

fn usize_to_exact_f64(count: usize) -> Result<f64, &'static str> {
    let count_bits =
        u64::try_from(count).map_err(|_| "strengthened online Softmax: count overflow")?;
    if i128::from(count_bits) > EXACT_F64_INT_MAX {
        return Err("strengthened online Softmax: count exceeds exact f64 integer range");
    }
    let as_i64 =
        i64::try_from(count_bits).map_err(|_| "strengthened online Softmax: count overflow")?;
    // Safe: |as_i64| <= 2^53.
    #[allow(clippy::cast_precision_loss)]
    Ok(as_i64 as f64)
}

fn finite_f32(value: f64, _context: &'static str) -> Result<f32, &'static str> {
    // AttentionResult stores f32; callers already require finite f64.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err("strengthened online Softmax: non-finite f32 narrow");
    }
    Ok(narrowed)
}

fn neumaier_add(sum: f64, compensation: f64, value: f64) -> (f64, f64) {
    let corrected = value - compensation;
    let next = sum + corrected;
    let next_compensation = (next - sum) - corrected;
    (next, next_compensation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::online_softmax_baseline;

    fn case(logits: &[f32], values: &[f32], head_dim: usize) -> AttentionCase {
        AttentionCase {
            logits: logits.to_vec(),
            values: values.to_vec(),
            head_dim,
        }
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tol * scale,
            "{a} != {b} (tol={tol}, scale={scale})"
        );
    }

    fn assert_parity(c: &AttentionCase) {
        let baseline = online_softmax_baseline(c).expect("baseline");
        let strengthened = online_softmax_strengthened(c).expect("strengthened");
        assert_eq!(baseline.output.len(), strengthened.output.len());
        for (&a, &b) in baseline.output.iter().zip(&strengthened.output) {
            assert_close(a, b, 2.0e-5);
        }
        assert_close(baseline.lse, strengthened.lse, 2.0e-5);
    }

    #[test]
    fn singleton_bit_exact_hand_fixture() {
        let c = case(&[2.5], &[1.0, -2.0, 4.0], 3);
        let result = online_softmax_strengthened(&c).expect("singleton");
        assert_eq!(result.output, vec![1.0, -2.0, 4.0]);
        assert_eq!(result.lse.to_bits(), 2.5_f32.to_bits());
        assert_eq!(result.metrics.exp_evaluations, 0);
        assert_eq!(result.metrics.log_evaluations, 0);
    }

    #[test]
    fn equal_logits_power_of_two_uniform_hand_fixture() {
        // n = 4 => weight = 0.25 exact. Values per key: [0, 4], [4, 0], [8, 8], [4, 4]
        // lane0 mean = (0+4+8+4)/4 = 4; lane1 mean = (4+0+8+4)/4 = 4.
        let c = case(
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 4.0, 4.0, 0.0, 8.0, 8.0, 4.0, 4.0],
            2,
        );
        let result = online_softmax_strengthened(&c).expect("equal");
        assert_eq!(result.output, vec![4.0, 4.0]);
        assert_eq!(
            exact_uniform_weight(4).unwrap().to_bits(),
            0.25_f64.to_bits()
        );
        // LSE = 1 + ln(4) = 1 + 2*ln(2); check via f64 then f32 narrow.
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let expected_lse = (1.0_f64 + 4.0_f64.ln()) as f32;
        assert_eq!(result.lse.to_bits(), expected_lse.to_bits());
        assert_eq!(result.metrics.exp_evaluations, 0);
        assert_eq!(result.metrics.log_evaluations, 1);
    }

    #[test]
    fn equal_logits_n2_mean_hand_fixture() {
        let c = case(&[0.0, 0.0], &[1.0, 3.0], 1);
        let result = online_softmax_strengthened(&c).expect("n2");
        assert_eq!(result.output, vec![2.0]);
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let expected_lse = (0.0_f64 + 2.0_f64.ln()) as f32;
        assert_eq!(result.lse.to_bits(), expected_lse.to_bits());
    }

    #[test]
    fn parity_monotone_increasing() {
        assert_parity(&case(
            &[-8.0, -4.0, -1.0, 0.0, 3.0, 9.0],
            &(0..48)
                .map(|i| f32::from(u16::try_from(i).unwrap()) * 0.03125 - 0.5)
                .collect::<Vec<_>>(),
            8,
        ));
    }

    #[test]
    fn parity_monotone_decreasing() {
        assert_parity(&case(
            &[9.0, 3.0, 0.0, -1.0, -4.0, -8.0],
            &(0..48)
                .map(|i| f32::from(u16::try_from(i).unwrap()) * 0.03125 - 0.5)
                .collect::<Vec<_>>(),
            8,
        ));
    }

    #[test]
    fn parity_equal_logits() {
        let logits = vec![2.0_f32; 32];
        let values: Vec<f32> = (0..32 * 16)
            .map(|i| f32::from(u16::try_from(i).unwrap()) * 0.03125 - 0.5)
            .collect();
        assert_parity(&case(&logits, &values, 16));
    }

    #[test]
    fn parity_alternating_new_maxima() {
        assert_parity(&case(
            &[0.0, -20.0, 1.0, -30.0, 2.0, -40.0, 3.0],
            &(0..28)
                .map(|i| f32::from(u16::try_from(i).unwrap()) * 0.03125 - 0.5)
                .collect::<Vec<_>>(),
            4,
        ));
    }

    #[test]
    fn parity_singleton() {
        assert_parity(&case(&[7.0], &[0.25, -0.5, 1.0], 3));
    }

    #[test]
    fn streaming_exp_count_matches_baseline_shape() {
        let c = case(&[0.0, 1.0, -1.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        let strengthened = online_softmax_strengthened(&c).expect("stream");
        // Non-equal path uses the same 2n-1 exp schedule as the scalar baseline.
        assert_eq!(strengthened.metrics.exp_evaluations, 2 * c.logits.len() - 1);
        assert_eq!(strengthened.metrics.log_evaluations, 1);
    }

    #[test]
    fn rejects_empty_logits() {
        let c = AttentionCase {
            logits: vec![],
            values: vec![],
            head_dim: 1,
        };
        assert!(online_softmax_strengthened(&c).is_err());
    }
}
