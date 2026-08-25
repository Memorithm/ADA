//! ADA-A9: deterministic execution-plan selection for exact attention.
//!
//! The selector consumes measurable workload signals and returns one of the
//! qualified exact plans (dense, A4 paged branch-and-bound, A5 hierarchical,
//! A5 content-aware) together with the rationale that drove the choice. All
//! thresholds trace back to the Thor L1/L2 evidence: pruning gains were
//! measured at 1.08x-1.24x on large natural workloads, while small workloads
//! paid more in bookkeeping than they saved.

#![forbid(unsafe_code)]

use ada_core::AttentionCase;

/// The exact execution plans available to the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPlan {
    /// Straight dense evaluation (ADA-A1 oracle family).
    Dense,
    /// ADA-A4 paged subset-threshold branch-and-bound.
    PagedBranchAndBound,
    /// ADA-A5 E0/E5 hierarchical key-box controller.
    Hierarchical,
    /// ADA-A5 E2 content-aware hybrid-bound controller.
    ContentAware,
}

/// Why a plan was chosen; carried next to the plan for evidence trails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rationale(pub &'static str);

/// Measurable signals extracted from a concrete workload.
#[derive(Debug, Clone, Copy)]
pub struct PlanSignals {
    /// Number of keys (sequence length).
    pub key_count: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// KV page size available for paged controllers; 0 means "no paging".
    pub page_size: usize,
    /// Largest absolute logit magnitude, used for the certified degenerate-
    /// magnitude check (`ulp((alpha-1) * max |logit|) >= 0.5`).
    pub max_abs_logit: f64,
    /// Softmax/entmax alpha in (1, 2].
    pub alpha: f64,
}

/// Below this key count every measured candidate was slower than dense.
const DENSE_CROSSOVER_KEY_COUNT: usize = 256;
/// Above this ratio of pages-to-load bookkeeping wins dominate hierarchical
/// traversal and content-aware partitioning is preferred.
const HIERARCHICAL_PAGE_RATIO_THRESHOLD: usize = 8;

impl PlanSignals {
    fn validate(&self) -> Result<(), &'static str> {
        if self.key_count == 0 || self.head_dim == 0 {
            return Err("ADA-A9 key_count and head_dim must be non-zero");
        }
        if !self.max_abs_logit.is_finite() {
            return Err("ADA-A9 max_abs_logit must be finite");
        }
        if !(self.alpha > 1.0 && self.alpha <= 2.0) {
            return Err("ADA-A9 alpha must lie in (1, 2]");
        }
        Ok(())
    }

    fn extreme_magnitude(&self) -> bool {
        let scale = self.alpha - 1.0;
        let max_scaled = scale * self.max_abs_logit.abs();
        // ulp(max_scaled) >= 0.5 mirrors the A4 collapse threshold exactly:
        // one ulp above 0.5 makes the nominal initial bracket unresolvable.
        if !max_scaled.is_finite() || max_scaled <= 0.0 {
            return false;
        }
        let next = f64::from_bits(max_scaled.to_bits() + 1);
        next - max_scaled >= 0.5
    }
}

/// Select the execution plan deterministically with its rationale.
///
/// # Errors
///
/// Returns an error when the signals violate the structural contract.
#[must_use = "the selected plan should drive the actual dispatch"]
pub fn select_plan(signals: &PlanSignals) -> Result<(ExecutionPlan, Rationale), &'static str> {
    signals.validate()?;

    if signals.extreme_magnitude() {
        return Ok((
            ExecutionPlan::Dense,
            Rationale(
                "degenerate magnitude regime: bracket collapse makes subset thresholds \
                 unusable, dense path finalizes through the certified normalization",
            ),
        ));
    }

    if signals.key_count < DENSE_CROSSOVER_KEY_COUNT {
        return Ok((
            ExecutionPlan::Dense,
            Rationale(
                "key count below the measured crossover where pruning bookkeeping \
                 costs more than it saves",
            ),
        ));
    }

    if signals.page_size == 0 {
        return Ok((
            ExecutionPlan::Dense,
            Rationale("no page geometry supplied, paged controllers are unavailable"),
        ));
    }

    let pages_per_leaf_window = signals.key_count / signals.page_size.max(1);
    if pages_per_leaf_window >= HIERARCHICAL_PAGE_RATIO_THRESHOLD {
        return Ok((
            ExecutionPlan::ContentAware,
            Rationale(
                "large natural workload: content-aware hybrid bounds pruned most \
                 subtrees in the qualified E4 evidence",
            ),
        ));
    }

    Ok((
        ExecutionPlan::Hierarchical,
        Rationale(
            "moderate paging depth: hierarchical key-box traversal captured the \
             measured gains without partition overhead",
        ),
    ))
}

/// Convenience bridge from an [`AttentionCase`] to plan signals.
///
/// # Errors
///
/// Returns an error when the case fails validation or alpha is out of range.
pub fn signals_from_case(
    case: &AttentionCase,
    alpha: f64,
    page_size: usize,
) -> Result<PlanSignals, &'static str> {
    case.validate()?;
    let max_abs_logit = f64::from(case.logits.iter().fold(0.0_f32, |acc, &l| acc.max(l.abs())));
    #[allow(clippy::cast_precision_loss)]
    Ok(PlanSignals {
        key_count: case.logits.len(),
        head_dim: case.head_dim,
        page_size,
        max_abs_logit,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signals(key_count: usize, page_size: usize) -> PlanSignals {
        PlanSignals {
            key_count,
            head_dim: 64,
            page_size,
            max_abs_logit: 12.0,
            alpha: 2.0,
        }
    }

    #[test]
    fn every_plan_branch_is_reachable_and_deterministic() {
        assert_eq!(
            select_plan(&signals(64, 32)).unwrap(),
            (
                ExecutionPlan::Dense,
                Rationale(
                    "key count below the measured crossover where pruning bookkeeping \
                     costs more than it saves"
                )
            )
        );
        assert_eq!(
            select_plan(&signals(1024, 0)).unwrap().0,
            ExecutionPlan::Dense
        );
        assert_eq!(
            select_plan(&signals(1024, 512)).unwrap().0,
            ExecutionPlan::Hierarchical
        );
        assert_eq!(
            select_plan(&signals(4096, 128)).unwrap().0,
            ExecutionPlan::ContentAware
        );
        // Determinism: identical signals select identical plans.
        assert_eq!(
            select_plan(&signals(4096, 128)),
            select_plan(&signals(4096, 128))
        );
    }

    #[test]
    fn extreme_magnitudes_force_dense_with_certified_finalization() {
        let mut s = signals(4096, 128);
        s.max_abs_logit = 1.0e200;
        let (plan, _) = select_plan(&s).unwrap();
        assert_eq!(plan, ExecutionPlan::Dense);

        // Normal magnitudes stay eligible for pruning plans.
        let mut s = signals(4096, 128);
        s.max_abs_logit = 12.0;
        assert_ne!(select_plan(&s).unwrap().0, ExecutionPlan::Dense);
    }

    #[test]
    fn invalid_signals_fail_closed() {
        let mut s = signals(10, 4);
        s.alpha = 2.5;
        assert!(select_plan(&s).is_err());

        let mut s = signals(0, 4);
        s.alpha = 2.0;
        assert!(select_plan(&s).is_err());

        let mut s = signals(100, 4);
        s.max_abs_logit = f64::INFINITY;
        assert!(select_plan(&s).is_err());
    }
}
