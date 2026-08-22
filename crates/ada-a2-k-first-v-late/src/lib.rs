#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::EntmaxDistribution;
use ada_a4_qk_box::QueryKeyPagedCase;
use ada_a5_hierarchical_bounds::{
    HierarchicalKeyIndex, PriorityLazyHierarchicalMetrics,
    branch_and_bound_entmax_hierarchical_priority_lazy,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VLateMetrics {
    pub rows_total: usize,
    pub rows_loaded: usize,
    pub rows_skipped: usize,
    pub scalar_values_loaded: usize,
    pub scalar_values_skipped: usize,
}

#[allow(clippy::cast_precision_loss)]
fn usize_ratio(numerator: usize, denominator: usize) -> f64 {
    debug_assert!(denominator != 0);
    numerator as f64 / denominator as f64
}

impl VLateMetrics {
    #[must_use]
    pub fn row_load_fraction(self) -> f64 {
        if self.rows_total == 0 {
            0.0
        } else {
            usize_ratio(self.rows_loaded, self.rows_total)
        }
    }

    #[must_use]
    pub fn row_avoidance(self) -> f64 {
        1.0 - self.row_load_fraction()
    }

    #[must_use]
    pub fn scalar_load_fraction(self) -> f64 {
        let total = self.scalar_values_loaded + self.scalar_values_skipped;

        if total == 0 {
            0.0
        } else {
            usize_ratio(self.scalar_values_loaded, total)
        }
    }

    #[must_use]
    pub fn scalar_avoidance(self) -> f64 {
        1.0 - self.scalar_load_fraction()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VLateResult {
    pub output: Vec<f64>,
    pub metrics: VLateMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KFirstVLateResult {
    pub distribution: EntmaxDistribution,
    pub output: Vec<f64>,
    pub loaded_k_tokens: Vec<bool>,
    pub k_metrics: PriorityLazyHierarchicalMetrics,
    pub v_metrics: VLateMetrics,
}

fn validate_layout(
    probabilities: &[f64],
    values: &[f64],
    value_dim: usize,
) -> Result<(), &'static str> {
    if probabilities.is_empty() {
        return Err("ADA-A2 requires at least one probability");
    }

    if value_dim == 0 {
        return Err("ADA-A2 value_dim must be non-zero");
    }

    let expected_values = probabilities
        .len()
        .checked_mul(value_dim)
        .ok_or("ADA-A2 value layout overflowed")?;

    if values.len() != expected_values {
        return Err("ADA-A2 values must be row-major [token_count, value_dim]");
    }

    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err("ADA-A2 probabilities must be finite and non-negative");
    }

    Ok(())
}

/// Dense eager-V research oracle.
///
/// This path deliberately reads every V row, including rows whose probability
/// is exactly zero. It is a semantic oracle and logical eager-load baseline,
/// not a hardware implementation.
///
/// # Errors
///
/// Returns an error for malformed probability/value layouts, invalid
/// probabilities, or any non-finite V scalar.
#[must_use = "the dense eager-V oracle output should be compared with V-late"]
pub fn dense_eager_value_oracle(
    probabilities: &[f64],
    values: &[f64],
    value_dim: usize,
) -> Result<Vec<f64>, &'static str> {
    validate_layout(probabilities, values, value_dim)?;

    let mut output = vec![0.0_f64; value_dim];

    for (row_index, &probability) in probabilities.iter().enumerate() {
        let start = row_index * value_dim;
        let row = &values[start..start + value_dim];

        if row.iter().any(|value| !value.is_finite()) {
            return Err("ADA-A2 dense eager oracle requires finite V values");
        }

        for (accumulator, &value) in output.iter_mut().zip(row.iter()) {
            *accumulator += probability * value;
        }
    }

    Ok(output)
}

/// Exact logical V-late weighted sum.
///
/// V rows are inspected only when their final probability is strictly
/// positive. A zero-probability row is counted as skipped before any scalar
/// from that row is read.
///
/// This establishes a source-level logical access contract only. Compilers,
/// caches, memory transactions, vector loads, prefetching, GQA reuse, and
/// physical bandwidth are outside this E0 qualification.
///
/// # Errors
///
/// Returns an error for malformed layouts, invalid probabilities, or a
/// non-finite scalar in a V row that is actually loaded.
#[must_use = "the V-late output and logical load metrics should be checked"]
pub fn exact_v_late_weighted_sum(
    probabilities: &[f64],
    values: &[f64],
    value_dim: usize,
) -> Result<VLateResult, &'static str> {
    validate_layout(probabilities, values, value_dim)?;

    let mut output = vec![0.0_f64; value_dim];

    let mut metrics = VLateMetrics {
        rows_total: probabilities.len(),
        ..VLateMetrics::default()
    };

    for (row_index, &probability) in probabilities.iter().enumerate() {
        if probability == 0.0 {
            metrics.rows_skipped += 1;
            metrics.scalar_values_skipped += value_dim;
            continue;
        }

        let start = row_index * value_dim;
        let row = &values[start..start + value_dim];

        if row.iter().any(|value| !value.is_finite()) {
            return Err("ADA-A2 loaded V row contains a non-finite value");
        }

        metrics.rows_loaded += 1;
        metrics.scalar_values_loaded += value_dim;

        for (accumulator, &value) in output.iter_mut().zip(row.iter()) {
            *accumulator += probability * value;
        }
    }

    debug_assert_eq!(
        metrics.rows_loaded + metrics.rows_skipped,
        metrics.rows_total
    );

    debug_assert_eq!(
        metrics.scalar_values_loaded + metrics.scalar_values_skipped,
        metrics.rows_total * value_dim
    );

    Ok(VLateResult { output, metrics })
}

/// Run the A5 exact priority-bound K-first controller, then materialize only
/// final-support V rows.
///
/// The A5 implementation still constructs dense Q/K scores internally as a
/// research oracle. Its candidate work counters, rather than that oracle work,
/// define the logical K-side accounting in this laboratory.
///
/// # Errors
///
/// Propagates A5 Q/K/index/Entmax errors and A2 V-layout/value errors.
#[must_use = "the K-first/V-late result and work metrics should be checked"]
pub fn priority_k_first_v_late(
    case: &QueryKeyPagedCase,
    index: &HierarchicalKeyIndex,
    values: &[f64],
    value_dim: usize,
) -> Result<KFirstVLateResult, &'static str> {
    let priority = branch_and_bound_entmax_hierarchical_priority_lazy(case, index)?;

    let v_late =
        exact_v_late_weighted_sum(&priority.distribution.probabilities, values, value_dim)?;

    Ok(KFirstVLateResult {
        distribution: priority.distribution,
        output: v_late.output,
        loaded_k_tokens: priority.loaded_tokens,
        k_metrics: priority.metrics,
        v_metrics: v_late.metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_a4_entmax_bnb::dense_entmax;
    use ada_a4_qk_box::dense_qk_scores;
    use ada_a5_hierarchical_bounds::build_hierarchical_key_index;

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);

        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right}"
        );
    }

    fn assert_outputs_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());

        for (&left_value, &right_value) in left.iter().zip(right.iter()) {
            assert_close(left_value, right_value, 4.0e-12);
        }
    }

    #[test]
    fn v_late_skips_zero_probability_rows_before_reading_values() {
        let probabilities = [1.0, 0.0, 0.0];

        let values = [2.0, -3.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN];

        let result = exact_v_late_weighted_sum(&probabilities, &values, 2).unwrap();

        assert_eq!(result.output, vec![2.0, -3.0]);
        assert_eq!(result.metrics.rows_total, 3);
        assert_eq!(result.metrics.rows_loaded, 1);
        assert_eq!(result.metrics.rows_skipped, 2);
        assert_eq!(result.metrics.scalar_values_loaded, 2);
        assert_eq!(result.metrics.scalar_values_skipped, 4);
        assert_close(result.metrics.row_avoidance(), 2.0 / 3.0, 1.0e-15);
    }

    #[test]
    fn v_late_rejects_non_finite_loaded_rows() {
        let probabilities = [0.5, 0.5];
        let values = [1.0, 2.0, f64::NAN, 4.0];

        assert!(exact_v_late_weighted_sum(&probabilities, &values, 2).is_err());
    }

    #[test]
    fn all_small_support_masks_match_dense_eager_oracle() {
        const ROWS: usize = 4;
        const VALUE_DIM: usize = 3;

        let values = [
            1.0, -2.0, 0.5, 3.0, 4.0, -1.0, -5.0, 2.0, 7.0, 0.25, -0.5, 8.0,
        ];

        for mask in 1_usize..(1_usize << ROWS) {
            let support_count = usize::try_from(mask.count_ones()).unwrap();

            let probability = 1.0 / f64::from(u32::try_from(support_count).unwrap());

            let probabilities = (0..ROWS)
                .map(|row| {
                    if mask & (1_usize << row) == 0 {
                        0.0
                    } else {
                        probability
                    }
                })
                .collect::<Vec<_>>();

            let dense = dense_eager_value_oracle(&probabilities, &values, VALUE_DIM).unwrap();

            let late = exact_v_late_weighted_sum(&probabilities, &values, VALUE_DIM).unwrap();

            assert_outputs_close(&dense, &late.output);
            assert_eq!(late.metrics.rows_loaded, support_count);
            assert_eq!(late.metrics.rows_skipped, ROWS - support_count);
        }
    }

    #[test]
    fn priority_k_first_v_late_matches_dense_for_sparse_entmax() {
        let keys = vec![
            10.0, 0.0, 9.5, 0.0, -10.0, 4.0, -11.0, -4.0, -20.0, 7.0, -21.0, -7.0, -30.0, 9.0,
            -31.0, -9.0,
        ];

        let values = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];

        for alpha in [1.5, 2.0] {
            let case = QueryKeyPagedCase {
                query: vec![1.0, 0.0],
                keys: keys.clone(),
                head_dim: 2,
                page_size: 8,
                alpha,
                score_scale: 1.0,
            };

            let index =
                build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, 2).unwrap();

            let dense_scores = dense_qk_scores(&case).unwrap();
            let dense_distribution = dense_entmax(&dense_scores, alpha).unwrap();

            let dense_output =
                dense_eager_value_oracle(&dense_distribution.probabilities, &values, 3).unwrap();

            let candidate = priority_k_first_v_late(&case, &index, &values, 3).unwrap();

            assert_eq!(dense_distribution, candidate.distribution);

            assert_outputs_close(&dense_output, &candidate.output);

            let support_count = dense_distribution
                .probabilities
                .iter()
                .filter(|&&probability| probability > 0.0)
                .count();

            assert_eq!(candidate.v_metrics.rows_loaded, support_count);

            assert_eq!(
                candidate.v_metrics.rows_skipped,
                case.key_count() - support_count
            );

            assert!(candidate.k_metrics.tokens_loaded >= candidate.v_metrics.rows_loaded);

            assert!(candidate.v_metrics.rows_skipped > 0);
        }
    }

    #[test]
    fn dense_support_requires_all_v_rows() {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 0.0],
            keys: vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            ],
            head_dim: 2,
            page_size: 8,
            alpha: 2.0,
            score_scale: 1.0,
        };

        let values = vec![
            1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
        ];

        let index =
            build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, 2).unwrap();

        let candidate = priority_k_first_v_late(&case, &index, &values, 2).unwrap();

        assert_eq!(candidate.v_metrics.rows_loaded, 8);
        assert_eq!(candidate.v_metrics.rows_skipped, 0);
        assert_close(candidate.v_metrics.row_avoidance(), 0.0, 1.0e-15);
    }
}
