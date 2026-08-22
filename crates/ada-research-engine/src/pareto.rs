//! Deterministic Pareto ranking.
//!
//! Candidates are ranked by non-dominated fronts over the objective vector
//! `(train_loss, total_operators, exp_count, max_count, mul_count,
//! add_sub_count, depth, state_outputs, temporary_count)`. Within a front the order is a strict total order:
//! loss (`f64::total_cmp`), then every cost dimension (derived `Ord` on
//! [`CostVector`]), then the canonical expression string — never insertion
//! order and never hash iteration.
//!
//! `train_loss` is always finite (execution failures receive a large finite
//! penalty), so `total_cmp` yields a total order with no NaN hazard.

use std::cmp::Ordering;

use crate::cost::CostVector;

/// The ranking objectives of one candidate.
#[derive(Debug, Clone)]
pub struct ObjectiveView<'a> {
    /// Discovery-corpus loss (finite).
    pub loss: f64,
    /// Structural cost vector.
    pub cost: CostVector,
    /// Canonical expression string (identity tie-breaker).
    pub canonical: &'a str,
}

impl ObjectiveView<'_> {
    fn objectives(&self) -> [f64; 9] {
        [
            self.loss,
            f64::from(self.cost.total_operators),
            f64::from(self.cost.exp_count),
            f64::from(self.cost.max_count),
            f64::from(self.cost.mul_count),
            f64::from(self.cost.add_sub_count),
            f64::from(self.cost.depth),
            f64::from(self.cost.state_outputs),
            f64::from(self.cost.temporary_count),
        ]
    }
}

/// Strict total order over two objective views.
#[must_use]
pub fn total_order(left: &ObjectiveView<'_>, right: &ObjectiveView<'_>) -> Ordering {
    left.loss
        .total_cmp(&right.loss)
        .then_with(|| left.cost.cmp(&right.cost))
        .then_with(|| left.canonical.cmp(right.canonical))
}

/// Component-wise domination across `(loss, cost dimensions)`:
/// `left <= right` everywhere and strictly better somewhere.
#[must_use]
pub fn dominates(left: &ObjectiveView<'_>, right: &ObjectiveView<'_>) -> bool {
    let left_obj = left.objectives();
    let right_obj = right.objectives();
    let mut all_le = true;
    let mut any_lt = false;
    for index in 0..left_obj.len() {
        match left_obj[index].total_cmp(&right_obj[index]) {
            Ordering::Less => any_lt = true,
            Ordering::Greater => return false,
            Ordering::Equal => {}
        }
        all_le &= left_obj[index] <= right_obj[index];
    }
    all_le && any_lt
}

/// Partition indices into non-dominated fronts, each front internally sorted
/// by [`total_order`].
///
/// O(n^2) pairwise domination, which is sufficient for archive-scale
/// survivor sets and fully deterministic.
#[must_use]
pub fn pareto_fronts(views: &[ObjectiveView<'_>]) -> Vec<Vec<usize>> {
    let n = views.len();
    let mut victims: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dominance_count = vec![0usize; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&views[i], &views[j]) {
                victims[i].push(j);
                dominance_count[j] += 1;
            } else if dominates(&views[j], &views[i]) {
                victims[j].push(i);
                dominance_count[i] += 1;
            }
        }
    }

    let mut assigned = vec![false; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    loop {
        let current: Vec<usize> = (0..n)
            .filter(|&index| !assigned[index] && dominance_count[index] == 0)
            .collect();
        if current.is_empty() {
            break;
        }
        for &index in &current {
            assigned[index] = true;
        }
        for &index in &current {
            for &victim in &victims[index] {
                dominance_count[victim] -= 1;
            }
        }
        let mut front = current;
        front.sort_by(|&a, &b| total_order(&views[a], &views[b]).then_with(|| a.cmp(&b)));
        fronts.push(front);
    }
    fronts
}

/// Indices of the first (best) non-dominated front, sorted by total order.
#[must_use]
pub fn pareto_front(views: &[ObjectiveView<'_>]) -> Vec<usize> {
    pareto_fronts(views).into_iter().next().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(total: u32) -> CostVector {
        CostVector {
            total_operators: total,
            ..CostVector::default()
        }
    }

    #[test]
    fn domination_is_componentwise() {
        let a = ObjectiveView {
            loss: 1.0,
            cost: cost(3),
            canonical: "a",
        };
        let b = ObjectiveView {
            loss: 2.0,
            cost: cost(5),
            canonical: "b",
        };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
        assert!(!dominates(&a, &a));
    }

    #[test]
    fn tradeoffs_are_mutually_non_dominating() {
        let cheap_wrong = ObjectiveView {
            loss: 10.0,
            cost: cost(1),
            canonical: "cheap",
        };
        let dear_right = ObjectiveView {
            loss: 0.0,
            cost: cost(9),
            canonical: "dear",
        };
        assert!(!dominates(&cheap_wrong, &dear_right));
        assert!(!dominates(&dear_right, &cheap_wrong));
    }

    #[test]
    fn fronts_are_layered_and_deterministic() {
        let make = |loss: f64, ops: u32, name: &'static str| ObjectiveView {
            loss,
            cost: cost(ops),
            canonical: name,
        };
        let views = vec![
            make(3.0, 9, "c"),
            make(1.0, 5, "a"),
            make(2.0, 7, "b"),
            make(1.0, 4, "a2"),
            make(9.9, 1, "tiny_loss_huge_cost_is_not_dominating"),
        ];
        let views_clone_views: Vec<ObjectiveView<'_>> = views.clone();
        let fronts_a = pareto_fronts(&views);
        let fronts_b = pareto_fronts(&views_clone_views);
        assert_eq!(fronts_a, fronts_b);

        // Front 0: index 3 dominates every other low-loss point; index 4 has
        // minimal operator count and is dominated by nobody (its cost is
        // lower than every dominator candidate could offer).
        assert_eq!(fronts_a[0], vec![3, 4]);
        assert_eq!(fronts_a[1], vec![1]);
        assert_eq!(fronts_a[2], vec![2]);
        assert_eq!(fronts_a[3], vec![0]);

        // The best front contains both non-dominated points in total order.
        let front = pareto_front(&views);
        assert_eq!(front, vec![3, 4]);
    }

    #[test]
    fn equal_objectives_break_ties_on_canonical_string() {
        let x = ObjectiveView {
            loss: 1.0,
            cost: cost(3),
            canonical: "(v 1)",
        };
        let y = ObjectiveView {
            loss: 1.0,
            cost: cost(3),
            canonical: "(v 0)",
        };
        assert_eq!(total_order(&y, &x), std::cmp::Ordering::Less);
    }

    #[test]
    fn negative_zero_and_nan_free_losses_sort_total() {
        let zero = ObjectiveView {
            loss: 0.0,
            cost: cost(1),
            canonical: "z",
        };
        let neg_zero = ObjectiveView {
            loss: -0.0,
            cost: cost(1),
            canonical: "nz",
        };
        // total_cmp treats -0.0 < 0.0; both are finite so ordering is stable.
        assert_eq!(total_order(&neg_zero, &zero), std::cmp::Ordering::Less);
    }
}
