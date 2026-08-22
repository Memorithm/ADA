//! Candidate recurrence programs.
//!
//! A candidate is an ordered, fixed-arity tuple of scalar expressions.  The
//! order is defined by [`crate::GrammarSpec::outputs`].  E0 therefore models
//! the online-softmax state transition directly as two outputs (`m_new`,
//! `l_new`) instead of leaking `m_new` back into the proposal inputs.

use serde::{Deserialize, Serialize};

use crate::expr::{ExecError, Expr, ExprError};
use crate::grammar::GrammarSpec;

/// A safe, inspectable state-transition candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    outputs: Vec<Expr>,
}

impl Candidate {
    /// Construct an ordered output tuple.
    #[must_use]
    pub fn new(outputs: Vec<Expr>) -> Self {
        Self { outputs }
    }

    /// Construct a one-output candidate (mainly useful for generic tests).
    #[must_use]
    pub fn scalar(output: Expr) -> Self {
        Self::new(vec![output])
    }

    /// Ordered output expressions.
    #[must_use]
    pub fn outputs(&self) -> &[Expr] {
        &self.outputs
    }

    /// Number of state outputs.
    #[must_use]
    pub fn output_arity(&self) -> usize {
        self.outputs.len()
    }

    /// Total nodes across every output tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.outputs.iter().fold(0usize, |total, expression| {
            total.saturating_add(expression.node_count())
        })
    }

    /// Maximum depth of any output tree (zero for an empty candidate).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.outputs.iter().map(Expr::depth).max().unwrap_or(0)
    }

    /// Statically validate arity, aggregate resources, variables, constants,
    /// and operators against the declared grammar.
    ///
    /// # Errors
    ///
    /// Returns a precise [`CandidateError`] before the candidate is
    /// normalized or interpreted.
    pub fn validate(&self, grammar: &GrammarSpec) -> Result<(), CandidateError> {
        if self.outputs.len() != grammar.outputs.len() {
            return Err(CandidateError::OutputArity {
                found: self.outputs.len(),
                expected: grammar.outputs.len(),
            });
        }
        let nodes = self.node_count();
        if nodes > grammar.max_nodes {
            return Err(CandidateError::NodeBudgetExceeded {
                nodes,
                maximum: grammar.max_nodes,
            });
        }
        let depth = self.depth();
        if depth > grammar.max_depth {
            return Err(CandidateError::DepthBudgetExceeded {
                depth,
                maximum: grammar.max_depth,
            });
        }
        for (output, expression) in self.outputs.iter().enumerate() {
            expression
                .validate_against(grammar)
                .map_err(|error| CandidateError::MalformedOutput { output, error })?;
        }
        Ok(())
    }

    /// Interpret every output in its declared order.
    ///
    /// # Errors
    ///
    /// Returns the first deterministic expression execution failure.
    pub fn eval(&self, variables: &[f64]) -> Result<Vec<f64>, ExecError> {
        self.outputs
            .iter()
            .map(|expression| expression.eval(variables))
            .collect()
    }

    pub(crate) fn outputs_mut(&mut self) -> &mut [Expr] {
        &mut self.outputs
    }
}

/// Candidate-level static validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateError {
    /// Candidate state-output arity disagrees with the grammar contract.
    OutputArity { found: usize, expected: usize },
    /// Aggregate candidate node budget exceeded.
    NodeBudgetExceeded { nodes: usize, maximum: usize },
    /// Per-output tree-depth budget exceeded.
    DepthBudgetExceeded { depth: usize, maximum: usize },
    /// One output expression is malformed.
    MalformedOutput { output: usize, error: ExprError },
}

impl std::fmt::Display for CandidateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputArity { found, expected } => {
                write!(
                    formatter,
                    "candidate has {found} outputs, expected {expected}"
                )
            }
            Self::NodeBudgetExceeded { nodes, maximum } => {
                write!(
                    formatter,
                    "candidate uses {nodes} nodes, exceeding budget {maximum}"
                )
            }
            Self::DepthBudgetExceeded { depth, maximum } => {
                write!(
                    formatter,
                    "candidate depth {depth} exceeds budget {maximum}"
                )
            }
            Self::MalformedOutput { output, error } => {
                write!(formatter, "output {output} is malformed: {error}")
            }
        }
    }
}

impl std::error::Error for CandidateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::OperatorSet;

    fn grammar() -> GrammarSpec {
        GrammarSpec {
            inputs: vec!["x".into(), "y".into()],
            outputs: vec!["a".into(), "b".into()],
            constants: vec![],
            operators: OperatorSet::arithmetic_only(),
            max_nodes: 5,
            max_depth: 3,
            version: 1,
        }
    }

    #[test]
    fn validates_output_arity_and_aggregate_budget() {
        let wrong_arity = Candidate::scalar(Expr::Var(0));
        assert!(matches!(
            wrong_arity.validate(&grammar()),
            Err(CandidateError::OutputArity {
                found: 1,
                expected: 2
            })
        ));

        let too_large = Candidate::new(vec![
            Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
        ]);
        assert!(matches!(
            too_large.validate(&grammar()),
            Err(CandidateError::NodeBudgetExceeded {
                nodes: 6,
                maximum: 5
            })
        ));
    }

    #[test]
    fn evaluates_outputs_in_declared_order() {
        let candidate = Candidate::new(vec![
            Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            Expr::Sub(Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
        ]);
        assert_eq!(candidate.eval(&[4.0, 1.5]).unwrap(), vec![5.5, 2.5]);
    }
}
