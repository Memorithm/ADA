#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::{BranchAndBoundResult, EntmaxPagedCase, branch_and_bound_entmax};

/// Coordinate-wise min/max metadata for one key page.
///
/// The metadata is intended to be built while the page keys are available
/// (for example during prefill). Query-time bound evaluation then needs only
/// this compact metadata, not the underlying key rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PageKeyBox {
    pub minimum: Vec<f64>,
    pub maximum: Vec<f64>,
    pub token_count: usize,
}

/// Query/key fixture for the ADA-A4 E1 coordinate-box experiment.
///
/// `keys` is row-major `[key_count, head_dim]`. E1 requires a positive score
/// scale so the coordinate-box upper-bound inequality preserves direction.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryKeyPagedCase {
    pub query: Vec<f64>,
    pub keys: Vec<f64>,
    pub head_dim: usize,
    pub page_size: usize,
    pub alpha: f64,
    pub score_scale: f64,
}

impl QueryKeyPagedCase {
    /// Number of keys implied by `keys.len() / head_dim`.
    ///
    /// For a case whose `keys` length is not an exact multiple of `head_dim`
    /// this floors; call [`QueryKeyPagedCase::validate`] first, which rejects
    /// inconsistent shapes, before relying on the value.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len().checked_div(self.head_dim).unwrap_or(0)
    }

    /// Number of pages implied by `key_count() / page_size`, rounding up.
    ///
    /// Call [`QueryKeyPagedCase::validate`] first; see [`Self::key_count`].
    #[must_use]
    pub fn page_count(&self) -> usize {
        if self.page_size == 0 {
            0
        } else {
            self.key_count().div_ceil(self.page_size)
        }
    }

    /// Validate the A4-E1 scalar Q/K research contract.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or non-finite inputs, inconsistent tensor
    /// shapes, non-positive page/head dimensions, alpha outside `(1, 2]`, or a
    /// non-positive/non-finite score scale.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.head_dim == 0 {
            return Err("ADA-A4 E1 head_dim must be non-zero");
        }
        if self.page_size == 0 {
            return Err("ADA-A4 E1 page_size must be non-zero");
        }
        if self.query.len() != self.head_dim {
            return Err("ADA-A4 E1 query must contain head_dim elements");
        }
        if self.keys.is_empty() {
            return Err("ADA-A4 E1 requires at least one key");
        }
        if !self.keys.chunks_exact(self.head_dim).remainder().is_empty() {
            return Err("ADA-A4 E1 keys must be row-major [key_count, head_dim]");
        }
        if self.query.iter().any(|value| !value.is_finite())
            || self.keys.iter().any(|value| !value.is_finite())
        {
            return Err("ADA-A4 E1 Q/K values must be finite");
        }
        if !self.alpha.is_finite() || self.alpha <= 1.0 || self.alpha > 2.0 {
            return Err("ADA-A4 E1 requires finite alpha in (1, 2]");
        }
        if !self.score_scale.is_finite() || self.score_scale <= 0.0 {
            return Err("ADA-A4 E1 score_scale must be finite and positive");
        }
        Ok(())
    }
}

/// Build coordinate-wise key minima/maxima for every fixed-size page.
///
/// This is a metadata-construction operation. It intentionally scans the keys
/// once; the resulting [`PageKeyBox`] values can then be reused across decode
/// queries without rereading the page keys.
///
/// # Errors
///
/// Returns an error for invalid dimensions, empty/non-finite keys, or a key
/// buffer whose length is not a multiple of `head_dim`.
#[must_use = "the page metadata is required for query-time bounds"]
pub fn build_page_key_boxes(
    keys: &[f64],
    head_dim: usize,
    page_size: usize,
) -> Result<Vec<PageKeyBox>, &'static str> {
    if head_dim == 0 {
        return Err("ADA-A4 E1 head_dim must be non-zero");
    }
    if page_size == 0 {
        return Err("ADA-A4 E1 page_size must be non-zero");
    }
    if keys.is_empty() {
        return Err("ADA-A4 E1 requires at least one key");
    }
    let rows = keys.chunks_exact(head_dim);
    if !rows.remainder().is_empty() {
        return Err("ADA-A4 E1 keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A4 E1 keys must be finite");
    }

    let key_count = keys.len() / head_dim;
    let mut boxes = Vec::with_capacity(key_count.div_ceil(page_size));
    for page_start in (0..key_count).step_by(page_size) {
        let page_end = (page_start + page_size).min(key_count);
        let first_start = page_start * head_dim;
        let first = &keys[first_start..first_start + head_dim];
        let mut minimum = first.to_vec();
        let mut maximum = first.to_vec();

        let page_values = &keys[first_start..page_end * head_dim];
        for row in page_values.chunks_exact(head_dim).skip(1) {
            for ((min_value, max_value), &value) in
                minimum.iter_mut().zip(maximum.iter_mut()).zip(row.iter())
            {
                *min_value = min_value.min(value);
                *max_value = max_value.max(value);
            }
        }

        boxes.push(PageKeyBox {
            minimum,
            maximum,
            token_count: page_end - page_start,
        });
    }
    Ok(boxes)
}

/// Evaluate the conservative query-specific score upper bound for each page.
///
/// For positive `score_scale`, page `p` uses
///
/// `score_scale * sum_j max(q_j * k_min[p,j], q_j * k_max[p,j])`.
///
/// In exact arithmetic this upper-bounds every scaled dot product in the page.
/// E1 evaluates the same expression in deterministic `f64` and separately
/// validates the resulting bounds against dense scores in its oracle tests.
///
/// # Errors
///
/// Returns an error for malformed/non-finite page metadata, a dimension
/// mismatch, a non-finite query, or a non-positive/non-finite score scale.
#[must_use = "the bounds drive certified page pruning"]
pub fn query_box_upper_bounds(
    query: &[f64],
    boxes: &[PageKeyBox],
    score_scale: f64,
) -> Result<Vec<f64>, &'static str> {
    if query.is_empty() {
        return Err("ADA-A4 E1 query must be non-empty");
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err("ADA-A4 E1 query must be finite");
    }
    if boxes.is_empty() {
        return Err("ADA-A4 E1 requires at least one page box");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A4 E1 score_scale must be finite and positive");
    }

    let mut bounds = Vec::with_capacity(boxes.len());
    for page_box in boxes {
        if page_box.minimum.len() != query.len() || page_box.maximum.len() != query.len() {
            return Err("ADA-A4 E1 page-box dimension mismatch");
        }
        if page_box.token_count == 0 {
            return Err("ADA-A4 E1 page boxes must contain at least one token");
        }

        let mut sum = 0.0_f64;
        for ((&q, &minimum), &maximum) in query
            .iter()
            .zip(page_box.minimum.iter())
            .zip(page_box.maximum.iter())
        {
            if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                return Err("ADA-A4 E1 page-box coordinates must be finite and ordered");
            }
            sum += (q * minimum).max(q * maximum);
        }
        let bound = sum * score_scale;
        if !bound.is_finite() {
            return Err("ADA-A4 E1 page upper bound overflowed");
        }
        bounds.push(bound);
    }
    Ok(bounds)
}

/// Compute deterministic dense scaled QK scores for the E1 oracle.
///
/// # Errors
///
/// Returns an error when the supplied [`QueryKeyPagedCase`] violates the E1
/// scalar contract or when a dot product overflows to a non-finite value.
#[must_use = "dense scores are the independent E1 oracle input"]
pub fn dense_qk_scores(case: &QueryKeyPagedCase) -> Result<Vec<f64>, &'static str> {
    case.validate()?;
    let mut scores = Vec::with_capacity(case.key_count());
    for key in case.keys.chunks_exact(case.head_dim) {
        let dot = case
            .query
            .iter()
            .zip(key.iter())
            .fold(0.0_f64, |sum, (&q, &k)| sum + q * k);
        let score = dot * case.score_scale;
        if !score.is_finite() {
            return Err("ADA-A4 E1 dense QK score overflowed");
        }
        scores.push(score);
    }
    Ok(scores)
}

/// Materialize the E1 score-level case from Q/K and page-box metadata.
///
/// This helper is deliberately an oracle/laboratory bridge: it computes all
/// dense scores so E1 can verify exactness, while its page bounds are computed
/// only from precomputed min/max metadata. A production sparse decoder would
/// compute scores only for pages selected by the branch-and-bound controller.
///
/// # Errors
///
/// Returns an error for an invalid Q/K case, malformed metadata, or a bound that
/// fails the existing score-level conservative-bound validator.
#[must_use = "the resulting case should be compared against the dense oracle"]
pub fn qk_box_entmax_case(case: &QueryKeyPagedCase) -> Result<EntmaxPagedCase, &'static str> {
    case.validate()?;
    let boxes = build_page_key_boxes(&case.keys, case.head_dim, case.page_size)?;
    let page_upper_bounds = query_box_upper_bounds(&case.query, &boxes, case.score_scale)?;
    let scores = dense_qk_scores(case)?;
    let paged = EntmaxPagedCase {
        scores,
        page_size: case.page_size,
        alpha: case.alpha,
        page_upper_bounds,
    };
    paged.validate()?;
    Ok(paged)
}

/// Run the existing exact score-level branch-and-bound using Q/K box bounds.
///
/// # Errors
///
/// Returns an error from Q/K validation, page-box construction, or the exact
/// score-level branch-and-bound threshold solver.
#[must_use = "the E1 result and logical page work should be checked"]
pub fn branch_and_bound_entmax_qk_box(
    case: &QueryKeyPagedCase,
) -> Result<BranchAndBoundResult, &'static str> {
    let paged = qk_box_entmax_case(case)?;
    branch_and_bound_entmax(&paged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_a4_entmax_bnb::dense_entmax;

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right}"
        );
    }

    fn assert_bound_dominates_page_scores(case: &QueryKeyPagedCase) {
        let paged = qk_box_entmax_case(case).unwrap();
        for (page, &bound) in paged.page_upper_bounds.iter().enumerate() {
            let start = page * case.page_size;
            let end = (start + case.page_size).min(paged.scores.len());
            for &score in &paged.scores[start..end] {
                assert!(bound >= score, "page {page}: bound {bound} < score {score}");
            }
        }
    }

    fn assert_qk_candidate_matches_dense(case: &QueryKeyPagedCase) -> BranchAndBoundResult {
        let paged = qk_box_entmax_case(case).unwrap();
        let dense = dense_entmax(&paged.scores, case.alpha).unwrap();
        let candidate = branch_and_bound_entmax_qk_box(case).unwrap();
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
                assert!(candidate.loaded_pages[index / case.page_size]);
            }
        }
        candidate
    }

    #[test]
    fn mixed_sign_query_box_bound_dominates_every_page_score() {
        let case = QueryKeyPagedCase {
            query: vec![2.0, -3.0, 0.5],
            keys: vec![
                1.0, 4.0, -2.0, 3.0, -1.0, 5.0, -4.0, 2.0, 1.0, 0.5, -3.0, 7.0, 2.0, 2.0, 2.0,
            ],
            head_dim: 3,
            page_size: 2,
            alpha: 1.5,
            score_scale: 0.5,
        };
        assert_bound_dominates_page_scores(&case);
    }

    #[test]
    fn qk_box_branch_and_bound_prunes_obvious_pages_exactly() {
        let keys = vec![
            8.0, 0.0, 7.0, 0.0, -10.0, 0.0, -12.0, 1.0, 0.0, 9.0, 1.0, 10.0,
        ];
        for alpha in [1.5, 2.0] {
            let case = QueryKeyPagedCase {
                query: vec![1.0, -1.0],
                keys: keys.clone(),
                head_dim: 2,
                page_size: 2,
                alpha,
                score_scale: 1.0,
            };
            let result = assert_qk_candidate_matches_dense(&case);
            assert_eq!(result.metrics.pages_total, 3);
            assert_eq!(result.metrics.pages_loaded, 1);
            assert_eq!(result.metrics.pages_pruned, 2);
        }
    }

    #[test]
    fn loose_coordinate_box_degrades_safely() {
        let case = QueryKeyPagedCase {
            query: vec![1.0, 1.0],
            keys: vec![10.0, -10.0, -10.0, 10.0, 2.0, 2.0, 1.5, 1.5],
            head_dim: 2,
            page_size: 2,
            alpha: 1.5,
            score_scale: 1.0,
        };
        assert_bound_dominates_page_scores(&case);
        let result = assert_qk_candidate_matches_dense(&case);
        assert_eq!(result.metrics.pages_loaded, result.metrics.pages_total);
    }

    #[test]
    fn default_attention_scale_is_supported() {
        let case = QueryKeyPagedCase {
            query: vec![1.0, -2.0, 0.5, 3.0],
            keys: vec![
                2.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0, -2.0, -3.0, 2.0, 1.0, 0.0, 0.0, -1.0, -2.0, 1.0,
            ],
            head_dim: 4,
            page_size: 2,
            alpha: 1.5,
            score_scale: 0.5,
        };
        assert_bound_dominates_page_scores(&case);
        assert_qk_candidate_matches_dense(&case);
    }

    #[test]
    fn exhaustive_small_qk_boxes_are_conservative() {
        let values = [-1.0_f64, 0.0, 1.0];
        for &q0 in &values {
            for &q1 in &values {
                for state in 0..729_usize {
                    let mut code = state;
                    let mut keys = Vec::with_capacity(6);
                    for _ in 0..6 {
                        keys.push(values[code % 3]);
                        code /= 3;
                    }
                    let case = QueryKeyPagedCase {
                        query: vec![q0, q1],
                        keys,
                        head_dim: 2,
                        page_size: 2,
                        alpha: 1.5,
                        score_scale: 0.75,
                    };
                    assert_bound_dominates_page_scores(&case);
                }
            }
        }
    }

    #[test]
    fn invalid_scale_is_rejected() {
        let base = QueryKeyPagedCase {
            query: vec![1.0],
            keys: vec![2.0],
            head_dim: 1,
            page_size: 1,
            alpha: 1.5,
            score_scale: 0.0,
        };
        assert!(base.validate().is_err());

        let mut negative = base;
        negative.score_scale = -1.0;
        assert!(negative.validate().is_err());
    }
}
