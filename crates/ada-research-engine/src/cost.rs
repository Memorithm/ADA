//! Deterministic structural cost model for candidate expressions.
//!
//! Every metric is a pure function of the normalized tree — never of
//! wall-clock time or hardware counters. Structural cost participates in
//! Pareto ranking so a simpler exact/sufficient candidate can outrank a more
//! expensive equivalent one without collapsing everything into one magic
//! score.

use serde::{Deserialize, Serialize};

use crate::candidate::Candidate;
use crate::expr::Expr;

/// Structural operator counts and depth of a normalized expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct CostVector {
    /// All non-leaf nodes.
    pub total_operators: u32,
    /// Number of `Exp` nodes.
    pub exp_count: u32,
    /// Number of `Max` nodes.
    pub max_count: u32,
    /// Number of `Mul` nodes.
    pub mul_count: u32,
    /// Combined `Add`/`Sub` node count.
    pub add_sub_count: u32,
    /// Maximum root-to-leaf depth.
    pub depth: u32,
    /// Number of state values emitted by the recurrence.
    pub state_outputs: u32,
    /// Deterministic upper bound on scalar temporaries (one per operator).
    pub temporary_count: u32,
}

impl CostVector {
    /// Compute the cost vector of an expression (normalize first for stable
    /// identity; this function itself is purely structural).
    #[must_use]
    pub fn of(candidate: &Candidate) -> Self {
        let mut costs = Self::default();
        for expression in candidate.outputs() {
            costs.measure(expression);
            costs.depth = costs
                .depth
                .max(u32::try_from(expression.depth()).unwrap_or(u32::MAX));
        }
        costs.state_outputs = u32::try_from(candidate.output_arity()).unwrap_or(u32::MAX);
        costs.temporary_count = costs.total_operators;
        costs
    }

    fn measure(&mut self, expr: &Expr) {
        match expr {
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
                self.total_operators += 1;
                self.add_sub_count += 1;
                self.measure(lhs);
                self.measure(rhs);
            }
            Expr::Mul(lhs, rhs) => {
                self.total_operators += 1;
                self.mul_count += 1;
                self.measure(lhs);
                self.measure(rhs);
            }
            Expr::Max(lhs, rhs) => {
                self.total_operators += 1;
                self.max_count += 1;
                self.measure(lhs);
                self.measure(rhs);
            }
            Expr::Exp(inner) => {
                self.total_operators += 1;
                self.exp_count += 1;
                self.measure(inner);
            }
        }
    }

    /// Component-wise domination: `self <= other` everywhere and strictly
    /// smaller somewhere. Loss is not part of this predicate; see
    /// [`crate::pareto`].
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        let le = self.total_operators <= other.total_operators
            && self.exp_count <= other.exp_count
            && self.max_count <= other.max_count
            && self.mul_count <= other.mul_count
            && self.add_sub_count <= other.add_sub_count
            && self.depth <= other.depth
            && self.state_outputs <= other.state_outputs
            && self.temporary_count <= other.temporary_count;
        let lt = self.total_operators < other.total_operators
            || self.exp_count < other.exp_count
            || self.max_count < other.max_count
            || self.mul_count < other.mul_count
            || self.add_sub_count < other.add_sub_count
            || self.depth < other.depth
            || self.state_outputs < other.state_outputs
            || self.temporary_count < other.temporary_count;
        le && lt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(index: usize) -> Expr {
        Expr::Var(index)
    }

    #[test]
    fn counts_operators_by_kind_and_depth() {
        // l_old * exp(m_old - m_new) + exp(score - m_new)
        let target = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(v(1)),
                Box::new(Expr::Exp(Box::new(Expr::Sub(
                    Box::new(v(0)),
                    Box::new(v(3)),
                )))),
            )),
            Box::new(Expr::Exp(Box::new(Expr::Sub(
                Box::new(v(2)),
                Box::new(v(3)),
            )))),
        );
        let candidate = Candidate::new(vec![Expr::Max(Box::new(v(0)), Box::new(v(2))), target]);
        let cost = CostVector::of(&candidate);
        assert_eq!(cost.total_operators, 7);
        assert_eq!(cost.exp_count, 2);
        assert_eq!(cost.mul_count, 1);
        assert_eq!(cost.add_sub_count, 3);
        assert_eq!(cost.max_count, 1);
        assert_eq!(cost.depth, 5);
        assert_eq!(cost.state_outputs, 2);
        assert_eq!(cost.temporary_count, 7);
    }

    #[test]
    fn domination_is_componentwise_strict_somewhere() {
        let cheap = CostVector {
            total_operators: 3,
            exp_count: 1,
            max_count: 0,
            mul_count: 1,
            add_sub_count: 1,
            depth: 3,
            ..CostVector::default()
        };
        let mut dearer = cheap;
        dearer.total_operators += 1;
        assert!(cheap.dominates(&dearer));
        assert!(!dearer.dominates(&cheap));
        assert!(!cheap.dominates(&cheap));
        let mixed = CostVector {
            total_operators: 2,
            depth: 4,
            ..dearer
        };
        assert!(!mixed.dominates(&cheap));
        assert!(!cheap.dominates(&mixed));
    }
}
