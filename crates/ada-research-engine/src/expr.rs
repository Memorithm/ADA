//! Restricted expression/recurrence IR for ADA attention discovery.
//!
//! The IR is intentionally tiny: scalar variables, finite constants and the
//! operators `Add`, `Sub`, `Mul`, `Max`, `Exp`. There is no loop construct, no
//! memory access, no FFI, no `unsafe` and no dynamic source execution; a
//! candidate is a pure function from a finite variable vector to one finite
//! `f64`.
//!
//! Division is deliberately absent from the E0 grammar: the stable online
//! softmax recurrence family requires no division and a pole at a zero
//! denominator would complicate the finite-value contract without enlarging
//! the reachable algorithm space.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::grammar::GrammarSpec;

/// A pure scalar expression over grammar variables.
///
/// Variable indices refer to positions in the problem's [`crate::GrammarSpec`]
/// variable list. Constants must be finite; non-finite constants are rejected
/// by [`Expr::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Grammar variable by index.
    Var(usize),
    /// Finite constant.
    Const(f64),
    /// `lhs + rhs`
    Add(Box<Expr>, Box<Expr>),
    /// `lhs - rhs`
    Sub(Box<Expr>, Box<Expr>),
    /// `lhs * rhs`
    Mul(Box<Expr>, Box<Expr>),
    /// `max(lhs, rhs)` with tie-break `if rhs > lhs { rhs } else { lhs }`
    Max(Box<Expr>, Box<Expr>),
    /// `exp(src)`
    Exp(Box<Expr>),
}

/// Stable operator tags used by static-validation errors and archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorKind {
    Add,
    Sub,
    Mul,
    Max,
    Exp,
}

/// Static validation failure for an expression against a grammar budget.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExprError {
    /// A variable index is outside the grammar's variable list.
    UnknownVariable { index: usize, available: usize },
    /// A constant is not finite (NaN or infinite).
    NonFiniteConstant,
    /// A finite constant was not declared by the grammar.
    UndeclaredConstant { bits: u64 },
    /// An expression used an operator disabled by the grammar.
    UnsupportedOperator { operator: OperatorKind },
    /// The expression uses more nodes than the budget allows.
    NodeBudgetExceeded { nodes: usize, maximum: usize },
    /// The expression is deeper than the budget allows.
    DepthBudgetExceeded { depth: usize, maximum: usize },
}

impl fmt::Display for ExprError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnknownVariable { index, available } => {
                write!(
                    formatter,
                    "variable index {index} out of range ({available} grammar variables)"
                )
            }
            Self::NonFiniteConstant => write!(formatter, "constant must be finite"),
            Self::UndeclaredConstant { bits } => {
                write!(
                    formatter,
                    "constant 0x{bits:016x} is not declared by the grammar"
                )
            }
            Self::UnsupportedOperator { operator } => {
                write!(
                    formatter,
                    "operator {operator:?} is not enabled by the grammar"
                )
            }
            Self::NodeBudgetExceeded { nodes, maximum } => {
                write!(
                    formatter,
                    "expression uses {nodes} nodes, exceeding budget {maximum}"
                )
            }
            Self::DepthBudgetExceeded { depth, maximum } => {
                write!(
                    formatter,
                    "expression depth {depth}, exceeding budget {maximum}"
                )
            }
        }
    }
}

impl std::error::Error for ExprError {}

impl Expr {
    /// Number of nodes in the expression tree (leaves included).
    #[must_use]
    pub fn node_count(&self) -> usize {
        let mut count = 0usize;
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            count = count.saturating_add(1);
            match expression {
                Self::Var(_) | Self::Const(_) => {}
                Self::Exp(inner) => pending.push(inner),
                Self::Add(lhs, rhs)
                | Self::Sub(lhs, rhs)
                | Self::Mul(lhs, rhs)
                | Self::Max(lhs, rhs) => {
                    pending.push(rhs);
                    pending.push(lhs);
                }
            }
        }
        count
    }

    /// Maximum root-to-leaf depth (a bare leaf has depth 1).
    #[must_use]
    pub fn depth(&self) -> usize {
        let mut maximum = 0usize;
        let mut pending = vec![(self, 1usize)];
        while let Some((expression, depth)) = pending.pop() {
            maximum = maximum.max(depth);
            match expression {
                Self::Var(_) | Self::Const(_) => {}
                Self::Exp(inner) => pending.push((inner, depth.saturating_add(1))),
                Self::Add(lhs, rhs)
                | Self::Sub(lhs, rhs)
                | Self::Mul(lhs, rhs)
                | Self::Max(lhs, rhs) => {
                    let child_depth = depth.saturating_add(1);
                    pending.push((rhs, child_depth));
                    pending.push((lhs, child_depth));
                }
            }
        }
        maximum
    }

    /// Validate against a variable count and node/depth budgets.
    ///
    /// # Errors
    ///
    /// Returns [`ExprError`] when a variable index is unknown, a constant is
    /// non-finite or a budget is exceeded.
    pub fn validate(
        &self,
        variable_count: usize,
        max_nodes: usize,
        max_depth: usize,
    ) -> Result<(), ExprError> {
        if self.node_count() > max_nodes {
            return Err(ExprError::NodeBudgetExceeded {
                nodes: self.node_count(),
                maximum: max_nodes,
            });
        }
        if self.depth() > max_depth {
            return Err(ExprError::DepthBudgetExceeded {
                depth: self.depth(),
                maximum: max_depth,
            });
        }
        self.validate_walk(variable_count, None)
    }

    /// Validate variables, constants, and enabled operators against a grammar.
    /// Aggregate candidate resource budgets are checked by
    /// [`crate::Candidate::validate`].
    ///
    /// # Errors
    ///
    /// Returns [`ExprError`] for the first malformed node in deterministic
    /// pre-order.
    pub fn validate_against(&self, grammar: &GrammarSpec) -> Result<(), ExprError> {
        self.validate_walk(grammar.input_count(), Some(grammar))
    }

    fn validate_walk(
        &self,
        variable_count: usize,
        grammar: Option<&GrammarSpec>,
    ) -> Result<(), ExprError> {
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            match expression {
                Self::Var(index) => {
                    if *index >= variable_count {
                        return Err(ExprError::UnknownVariable {
                            index: *index,
                            available: variable_count,
                        });
                    }
                }
                Self::Const(value) => {
                    if !value.is_finite() {
                        return Err(ExprError::NonFiniteConstant);
                    }
                    if let Some(grammar) = grammar {
                        let declared = grammar
                            .constants
                            .iter()
                            .any(|constant| constant.to_bits() == value.to_bits());
                        if !declared {
                            return Err(ExprError::UndeclaredConstant {
                                bits: value.to_bits(),
                            });
                        }
                    }
                }
                Self::Add(lhs, rhs) => {
                    require_operator(grammar, OperatorKind::Add)?;
                    pending.push(rhs);
                    pending.push(lhs);
                }
                Self::Sub(lhs, rhs) => {
                    require_operator(grammar, OperatorKind::Sub)?;
                    pending.push(rhs);
                    pending.push(lhs);
                }
                Self::Mul(lhs, rhs) => {
                    require_operator(grammar, OperatorKind::Mul)?;
                    pending.push(rhs);
                    pending.push(lhs);
                }
                Self::Max(lhs, rhs) => {
                    require_operator(grammar, OperatorKind::Max)?;
                    pending.push(rhs);
                    pending.push(lhs);
                }
                Self::Exp(inner) => {
                    require_operator(grammar, OperatorKind::Exp)?;
                    pending.push(inner);
                }
            }
        }
        Ok(())
    }

    /// Deterministic scalar evaluation.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::NonFiniteResult`] when any intermediate or final
    /// value is not finite, and [`ExecError::VariableCount`] when `vars` does
    /// not supply exactly the expected number of variables.
    pub fn eval(&self, vars: &[f64]) -> Result<f64, ExecError> {
        match self {
            Self::Var(index) => {
                let value = vars.get(*index).copied().ok_or(ExecError::VariableCount {
                    expected: *index + 1,
                    found: vars.len(),
                })?;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(ExecError::NonFiniteResult)
                }
            }
            Self::Const(value) => {
                if value.is_finite() {
                    Ok(*value)
                } else {
                    Err(ExecError::NonFiniteResult)
                }
            }
            Self::Add(lhs, rhs) => {
                let sum = lhs.eval(vars)? + rhs.eval(vars)?;
                finish(sum)
            }
            Self::Sub(lhs, rhs) => {
                let difference = lhs.eval(vars)? - rhs.eval(vars)?;
                finish(difference)
            }
            Self::Mul(lhs, rhs) => {
                let product = lhs.eval(vars)? * rhs.eval(vars)?;
                finish(product)
            }
            Self::Max(lhs, rhs) => {
                let left = lhs.eval(vars)?;
                let right = rhs.eval(vars)?;
                Ok(symmetric_max(left, right))
            }
            Self::Exp(inner) => finish(inner.eval(vars)?.exp()),
        }
    }
}

fn require_operator(
    grammar: Option<&GrammarSpec>,
    operator: OperatorKind,
) -> Result<(), ExprError> {
    let Some(grammar) = grammar else {
        return Ok(());
    };
    let enabled = match operator {
        OperatorKind::Add => grammar.operators.add,
        OperatorKind::Sub => grammar.operators.sub,
        OperatorKind::Mul => grammar.operators.mul,
        OperatorKind::Max => grammar.operators.max,
        OperatorKind::Exp => grammar.operators.exp,
    };
    if enabled {
        Ok(())
    } else {
        Err(ExprError::UnsupportedOperator { operator })
    }
}

fn finish(value: f64) -> Result<f64, ExecError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ExecError::NonFiniteResult)
    }
}

/// Commutative maximum with a deterministic tie rule.
///
/// Returns the strictly greater operand; on numeric ties (which can only
/// differ bitwise for the `(+0.0, -0.0)` pair, NaN being rejected elsewhere)
/// it returns `+0.0` so `Max(a, b)` and `Max(b, a)` are always bitwise
/// identical.
#[must_use]
pub fn symmetric_max(left: f64, right: f64) -> f64 {
    if right > left {
        right
    } else if left > right {
        left
    } else if left.to_bits() == 0 || right.to_bits() == 0 {
        0.0
    } else {
        left
    }
}

/// Runtime evaluation failure. Candidates that produce non-finite
/// intermediates are rejected; they are never silently coerced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecError {
    /// An intermediate or final value was NaN or infinite.
    NonFiniteResult,
    /// The variable vector did not contain the requested index.
    VariableCount { expected: usize, found: usize },
}

impl fmt::Display for ExecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NonFiniteResult => write!(formatter, "non-finite intermediate result"),
            Self::VariableCount { expected, found } => {
                write!(formatter, "expected {expected} variables, found {found}")
            }
        }
    }
}

impl std::error::Error for ExecError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(index: usize) -> Expr {
        Expr::Var(index)
    }

    #[test]
    fn node_count_and_depth_are_structural() {
        // l_old * exp(m_old - m_new) + exp(score - m_new)
        let target = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(var(1)),
                Box::new(Expr::Exp(Box::new(Expr::Sub(
                    Box::new(var(0)),
                    Box::new(var(3)),
                )))),
            )),
            Box::new(Expr::Exp(Box::new(Expr::Sub(
                Box::new(var(2)),
                Box::new(var(3)),
            )))),
        );
        // l_old * exp(m_old - m_new) + exp(score - m_new): 5 variable leaves,
        // 6 operators => 11 nodes.
        assert_eq!(target.node_count(), 11);
        assert_eq!(target.depth(), 5);
    }

    #[test]
    fn validates_unknown_variable() {
        let bad = Expr::Add(Box::new(var(0)), Box::new(var(7)));
        assert_eq!(
            bad.validate(4, 16, 8),
            Err(ExprError::UnknownVariable {
                index: 7,
                available: 4
            })
        );
    }

    #[test]
    fn rejects_non_finite_constant() {
        let bad = Expr::Const(f64::NAN);
        assert_eq!(bad.validate(4, 16, 8), Err(ExprError::NonFiniteConstant));
        let infinite = Expr::Const(f64::INFINITY);
        assert_eq!(
            infinite.validate(4, 16, 8),
            Err(ExprError::NonFiniteConstant)
        );
    }

    #[test]
    fn rejects_budget_violations() {
        let big = Expr::Add(
            Box::new(Expr::Add(Box::new(var(0)), Box::new(var(1)))),
            Box::new(Expr::Add(Box::new(var(2)), Box::new(var(3)))),
        );
        assert_eq!(
            big.validate(4, 6, 8),
            Err(ExprError::NodeBudgetExceeded {
                nodes: 7,
                maximum: 6
            })
        );
        let deep = Expr::Exp(Box::new(Expr::Exp(Box::new(Expr::Exp(Box::new(var(0)))))));
        assert_eq!(
            deep.validate(4, 16, 3),
            Err(ExprError::DepthBudgetExceeded {
                depth: 4,
                maximum: 3
            })
        );
    }

    #[test]
    fn eval_matches_direct_arithmetic() {
        let expr = Expr::Add(
            Box::new(Expr::Mul(Box::new(var(0)), Box::new(Expr::Const(2.0)))),
            Box::new(var(1)),
        );
        let value = expr.eval(&[3.0, 0.5]).unwrap();
        assert_eq!(value.to_bits(), 6.5_f64.to_bits());
    }

    #[test]
    fn max_is_symmetric_on_signed_zero_ties() {
        let left = Expr::Max(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(-0.0)));
        let right = Expr::Max(Box::new(Expr::Const(-0.0)), Box::new(Expr::Const(0.0)));
        assert_eq!(
            left.eval(&[]).unwrap().to_bits(),
            right.eval(&[]).unwrap().to_bits()
        );
        assert_eq!(left.eval(&[]).unwrap().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn overflow_is_rejected_not_silent() {
        let huge = Expr::Exp(Box::new(Expr::Const(1_000.0)));
        assert_eq!(huge.eval(&[]), Err(ExecError::NonFiniteResult));
        let product = Expr::Mul(Box::new(Expr::Const(f64::MAX)), Box::new(Expr::Const(2.0)));
        assert_eq!(product.eval(&[]), Err(ExecError::NonFiniteResult));
    }

    #[test]
    fn missing_variables_are_reported() {
        let expr = Expr::Var(2);
        assert_eq!(
            expr.eval(&[1.0]),
            Err(ExecError::VariableCount {
                expected: 3,
                found: 1
            })
        );
    }
}
