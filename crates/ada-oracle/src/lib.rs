#![forbid(unsafe_code)]

use ada_core::{AttentionCase, AttentionResult, LogicalMetrics};

/// Baseline streaming online Softmax recurrence.
///
/// For `n` logits this evaluates `2n-1` non-trivial/explicit `exp` calls in
/// this scalar formulation and one final `ln` for LSE.
pub fn online_softmax_baseline(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    case.validate()?;

    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;
    let mut numerator = vec![0.0_f32; case.head_dim];
    let mut metrics = LogicalMetrics::default();

    for (key, &score) in case.logits.iter().enumerate() {
        metrics.qk_pairs_evaluated += 1;
        let new_max = running_max.max(score);
        let alpha = if running_max.is_finite() {
            metrics.exp_evaluations += 1;
            (running_max - new_max).exp()
        } else {
            0.0
        };
        metrics.exp_evaluations += 1;
        let probability_numerator = (score - new_max).exp();
        running_sum = alpha * running_sum + probability_numerator;

        let value = &case.values[key * case.head_dim..(key + 1) * case.head_dim];
        for (acc, &v) in numerator.iter_mut().zip(value) {
            *acc = alpha * *acc + probability_numerator * v;
            metrics.value_accumulate_elements += 1;
        }
        running_max = new_max;
    }

    let inv_sum = running_sum.recip();
    for x in &mut numerator {
        *x *= inv_sum;
    }
    metrics.log_evaluations += 1;
    let lse = running_max + running_sum.ln();

    Ok(AttentionResult {
        output: numerator,
        lse,
        metrics,
    })
}

/// Hand-seeded ADA-A1 candidate: branch-specialized exact real-arithmetic
/// online Softmax recurrence.
///
/// The first score initializes `(m, l, O) = (s0, 1, V0)`. Each later score
/// evaluates exactly one `exp`: either the new score is below the running max,
/// or it becomes the new max and rescales the previous state.
pub fn online_softmax_one_exp(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    case.validate()?;

    let first_score = case.logits[0];
    let mut running_max = first_score;
    let mut running_sum = 1.0_f32;
    let mut numerator = case.values[..case.head_dim].to_vec();
    let mut metrics = LogicalMetrics {
        qk_pairs_evaluated: 1,
        exp_evaluations: 0,
        log_evaluations: 0,
        value_accumulate_elements: case.head_dim,
    };

    for (key, &score) in case.logits.iter().enumerate().skip(1) {
        metrics.qk_pairs_evaluated += 1;
        let value = &case.values[key * case.head_dim..(key + 1) * case.head_dim];

        if score <= running_max {
            metrics.exp_evaluations += 1;
            let p = (score - running_max).exp();
            running_sum += p;
            for (acc, &v) in numerator.iter_mut().zip(value) {
                *acc += p * v;
                metrics.value_accumulate_elements += 1;
            }
        } else {
            metrics.exp_evaluations += 1;
            let alpha = (running_max - score).exp();
            running_sum = alpha * running_sum + 1.0;
            for (acc, &v) in numerator.iter_mut().zip(value) {
                *acc = alpha * *acc + v;
                metrics.value_accumulate_elements += 1;
            }
            running_max = score;
        }
    }

    let inv_sum = running_sum.recip();
    for x in &mut numerator {
        *x *= inv_sum;
    }
    metrics.log_evaluations += 1;
    let lse = running_max + running_sum.ln();

    Ok(AttentionResult {
        output: numerator,
        lse,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(logits: &[f32], head_dim: usize) -> AttentionCase {
        let mut values = Vec::with_capacity(logits.len() * head_dim);
        for key in 0..logits.len() {
            for lane in 0..head_dim {
                values.push((key * head_dim + lane) as f32 * 0.03125 - 0.5);
            }
        }
        AttentionCase {
            logits: logits.to_vec(),
            values,
            head_dim,
        }
    }

    fn assert_close(a: f32, b: f32, tol: f32) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!((a - b).abs() <= tol * scale, "{a} != {b}");
    }

    fn assert_parity(c: &AttentionCase) {
        let baseline = online_softmax_baseline(c).unwrap();
        let candidate = online_softmax_one_exp(c).unwrap();
        assert_eq!(baseline.output.len(), candidate.output.len());
        for (&a, &b) in baseline.output.iter().zip(&candidate.output) {
            assert_close(a, b, 2.0e-6);
        }
        assert_close(baseline.lse, candidate.lse, 2.0e-6);
        assert_eq!(baseline.metrics.exp_evaluations, 2 * c.logits.len() - 1);
        assert_eq!(candidate.metrics.exp_evaluations, c.logits.len() - 1);
        assert_eq!(baseline.metrics.log_evaluations, 1);
        assert_eq!(candidate.metrics.log_evaluations, 1);
    }

    #[test]
    fn one_exp_matches_monotone_increasing() {
        assert_parity(&case(&[-8.0, -4.0, -1.0, 0.0, 3.0, 9.0], 8));
    }

    #[test]
    fn one_exp_matches_monotone_decreasing() {
        assert_parity(&case(&[9.0, 3.0, 0.0, -1.0, -4.0, -8.0], 8));
    }

    #[test]
    fn one_exp_matches_equal_logits() {
        assert_parity(&case(&[2.0; 32], 16));
    }

    #[test]
    fn one_exp_matches_alternating_new_maxima() {
        assert_parity(&case(&[0.0, -20.0, 1.0, -30.0, 2.0, -40.0, 3.0], 4));
    }

    #[test]
    fn one_exp_matches_singleton() {
        assert_parity(&case(&[7.0], 3));
    }
}
