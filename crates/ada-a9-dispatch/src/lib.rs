//! ADA-A9 end-to-end dispatcher: select an exact execution plan from
//! measurable workload signals and execute it, returning the chosen plan next
//! to its certified distribution.
//!
//! The dispatcher is the integration seam between the rule-based selector and
//! the qualified controllers. Every arm computes the SAME dense reference the
//! controllers use for bound validation, so a dispatch result is always
//! comparable against `dense_entmax` within documented tolerance.

#![forbid(unsafe_code)]

use ada_a4_entmax_bnb::{
    EntmaxDistribution, EntmaxPagedCase, branch_and_bound_entmax, dense_entmax,
};
use ada_a4_qk_box::{QueryKeyPagedCase, dense_qk_scores};
use ada_a5_content_aware_bounds::{
    ContentAwareGeometry, branch_and_bound_entmax_content_aware,
    build_content_aware_key_index_with_geometry,
};
use ada_a5_hierarchical_bounds::{
    branch_and_bound_entmax_hierarchical, build_hierarchical_key_index,
};
use ada_a9_plan_selector::{ExecutionPlan, PlanSignals, Rationale, select_plan};

/// Result of one dispatched execution.
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchOutcome {
    pub plan: ExecutionPlan,
    pub rationale: Rationale,
    pub distribution: EntmaxDistribution,
}

fn exact_page_bounds(scores: &[f64], page_size: usize) -> Vec<f64> {
    scores
        .chunks(page_size)
        .map(|page| page.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect()
}

/// Select and execute the plan for this case.
///
/// # Errors
///
/// Returns errors from validation, score computation, index construction, or
/// the underlying controller — never silently substitutes another plan.
#[must_use = "the dispatched outcome should be checked"]
pub fn execute_selected_plan(
    case: &QueryKeyPagedCase,
    leaf_size: usize,
) -> Result<DispatchOutcome, &'static str> {
    let dense_scores = dense_qk_scores(case)?;
    let max_abs_score = dense_scores
        .iter()
        .fold(0.0_f64, |acc, &s| acc.max(s.abs()));

    let signals = PlanSignals {
        key_count: case.key_count(),
        head_dim: case.head_dim,
        page_size: case.page_size,
        max_abs_logit: max_abs_score,
        alpha: case.alpha,
    };
    let (plan, rationale) = select_plan(&signals)?;

    let distribution = match plan {
        ExecutionPlan::Dense => dense_entmax(&dense_scores, case.alpha)?,
        // Not produced by the current rule table; supported for forward
        // compatibility so callers can force paged BnB via selector evolution.
        ExecutionPlan::PagedBranchAndBound => {
            let paged = EntmaxPagedCase {
                page_upper_bounds: exact_page_bounds(&dense_scores, case.page_size),
                scores: dense_scores.clone(),
                page_size: case.page_size,
                alpha: case.alpha,
            };
            branch_and_bound_entmax(&paged)?.distribution
        }
        ExecutionPlan::Hierarchical => {
            let index =
                build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)?;
            branch_and_bound_entmax_hierarchical(case, &index)?.distribution
        }
        ExecutionPlan::ContentAware => {
            let index = build_content_aware_key_index_with_geometry(
                &case.keys,
                case.head_dim,
                case.page_size,
                leaf_size,
                ContentAwareGeometry::PcaShrunkBall,
            )?;
            branch_and_bound_entmax_content_aware(case, &index)?.distribution
        }
    };

    Ok(DispatchOutcome {
        plan,
        rationale,
        distribution,
    })
}

/// Execute EVERY controller on this case (diagnostic helper for parity
/// audits); used by tests and the E4 replay example to prove that whatever
/// plan the selector picks matches all alternatives.
///
/// # Errors
///
/// Propagates any controller failure.
pub fn execute_all_plans(
    case: &QueryKeyPagedCase,
    leaf_size: usize,
) -> Result<Vec<(ExecutionPlan, EntmaxDistribution)>, &'static str> {
    let dense_scores = dense_qk_scores(case)?;

    let mut results = vec![(
        ExecutionPlan::Dense,
        dense_entmax(&dense_scores, case.alpha)?,
    )];

    let paged = EntmaxPagedCase {
        page_upper_bounds: exact_page_bounds(&dense_scores, case.page_size),
        scores: dense_scores.clone(),
        page_size: case.page_size,
        alpha: case.alpha,
    };
    results.push((
        ExecutionPlan::PagedBranchAndBound,
        branch_and_bound_entmax(&paged)?.distribution,
    ));

    let hierarchical =
        build_hierarchical_key_index(&case.keys, case.head_dim, case.page_size, leaf_size)?;
    results.push((
        ExecutionPlan::Hierarchical,
        branch_and_bound_entmax_hierarchical(case, &hierarchical)?.distribution,
    ));

    let content_aware = build_content_aware_key_index_with_geometry(
        &case.keys,
        case.head_dim,
        case.page_size,
        leaf_size,
        ContentAwareGeometry::PcaShrunkBall,
    )?;
    results.push((
        ExecutionPlan::ContentAware,
        branch_and_bound_entmax_content_aware(case, &content_aware)?.distribution,
    ));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qk_case(
        query: [f64; 2],
        keys: &[f64],
        head_dim: usize,
        page_size: usize,
        alpha: f64,
    ) -> QueryKeyPagedCase {
        QueryKeyPagedCase {
            query: query.to_vec(),
            keys: keys.to_vec(),
            head_dim,
            page_size,
            alpha,
            score_scale: 1.0,
        }
    }

    fn assert_parity(case: &QueryKeyPagedCase, leaf_size: usize) -> DispatchOutcome {
        let outcome = execute_selected_plan(case, leaf_size).unwrap();
        let all = execute_all_plans(case, leaf_size).unwrap();

        for (plan, distribution) in &all {
            assert_close(outcome.distribution.tau, distribution.tau, 2.0e-12);
            assert_eq!(
                outcome.distribution.probabilities.len(),
                distribution.probabilities.len()
            );
            for (&selected, &alternative) in outcome
                .distribution
                .probabilities
                .iter()
                .zip(distribution.probabilities.iter())
            {
                assert_close(selected, alternative, 4.0e-12);
            }
            let _ = plan;
        }
        outcome
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= tolerance * scale,
            "{left} != {right}"
        );
    }

    #[test]
    fn small_workload_dispatches_dense_with_exact_distribution() {
        let case = qk_case(
            [1.0, -0.5],
            &[1.0, 0.0, -1.0, 0.5, 0.5, 0.7, -0.2, 0.9],
            2,
            2,
            1.5,
        );
        let outcome = assert_parity(&case, 1);
        assert_eq!(outcome.plan, ExecutionPlan::Dense);
    }

    #[test]
    fn large_workloads_dispatch_pruning_plans_and_stay_exact() {
        // 512 keys, page 64: 8 pages per window >= 8 -> ContentAware.
        let mut keys = Vec::new();
        for i in 0..512_usize {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            #[allow(clippy::cast_precision_loss)]
            let magnitude = i as f64 * 0.015_625;
            #[allow(clippy::cast_precision_loss)]
            let residue = (i % 7) as f64;
            keys.extend_from_slice(&[sign * magnitude, residue * 0.25 - 0.75]);
        }
        let case = qk_case([0.3, -0.6], &keys, 2, 64, 2.0);
        let outcome = assert_parity(&case, 8);
        assert_eq!(outcome.plan, ExecutionPlan::ContentAware);

        // Same keys with one giant page: 1 window < 8 -> Hierarchical.
        let coarse = qk_case([0.3, -0.6], &keys, 2, 512, 2.0);
        let outcome = assert_parity(&coarse, 16);
        assert_eq!(outcome.plan, ExecutionPlan::Hierarchical);
    }

    #[test]
    fn degenerate_magnitudes_force_dense_everywhere() {
        let keys = vec![1.0e200, 0.0, -1.0e200, 0.0];
        let case = qk_case([1.0, 1.0], &keys, 2, 2, 2.0);

        // The pruning controllers legitimately refuse to build indexes over
        // magnitudes whose geometry overflows, so here we compare the
        // dispatched plan against the dense oracle directly instead of
        // running every controller.
        let outcome = execute_selected_plan(&case, 1).unwrap();
        assert_eq!(outcome.plan, ExecutionPlan::Dense);
        assert!(outcome.rationale.0.contains("degenerate"));

        let dense_scores = dense_qk_scores(&case).unwrap();
        let dense = dense_entmax(&dense_scores, case.alpha).unwrap();
        assert_close(dense.tau, outcome.distribution.tau, 2.0e-12);
        for (&expected, &actual) in dense
            .probabilities
            .iter()
            .zip(outcome.distribution.probabilities.iter())
        {
            assert_close(expected, actual, 4.0e-12);
        }

        let mass: f64 = outcome.distribution.probabilities.iter().copied().sum();
        assert_close(mass, 1.0, 4.0e-15);
        assert_close(outcome.distribution.probabilities[0], 1.0, 2.0e-15);
    }
}
