//! Candidate grammar specification.
//!
//! The grammar tells a proposer which inputs, outputs, constants and operators may
//! appear. It deliberately does **not** encode any target syntax tree: the E0
//! search grammar knows the variable *names* of the online-softmax problem,
//! nothing about the reference recurrence.

use serde::{Deserialize, Serialize};

use crate::canon::hex;
use crate::digest_writer::DigestWriter;
use crate::expr::Expr;

/// Which operators a grammar exposes to proposers.
// A capability bitset is clearer at this trust boundary than a state machine;
// every flag is independently validated and canonically digested.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSet {
    pub add: bool,
    pub sub: bool,
    pub mul: bool,
    pub max: bool,
    pub exp: bool,
}

impl OperatorSet {
    /// All E0 operators.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            add: true,
            sub: true,
            mul: true,
            max: true,
            exp: true,
        }
    }

    /// Arithmetic without `Max`/`Exp`.
    #[must_use]
    pub const fn arithmetic_only() -> Self {
        Self {
            add: true,
            sub: true,
            mul: true,
            max: false,
            exp: false,
        }
    }
}

/// The candidate search space offered to every proposer.
///
/// `constants` must contain finite values only. An empty constant list is
/// normal: the E0 reference recurrence is constant-free, so the default E0
/// grammar keeps the search space minimal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrammarSpec {
    /// Ordered input names; index `i` in [`Expr::Var`] refers to `inputs[i]`.
    pub inputs: Vec<String>,
    /// Ordered state-output names. Candidate arity must match exactly.
    pub outputs: Vec<String>,
    /// Finite constants proposers may use as leaves (may be empty).
    pub constants: Vec<f64>,
    /// Available operators.
    pub operators: OperatorSet,
    /// Maximum node count per candidate expression.
    pub max_nodes: usize,
    /// Maximum tree depth per candidate expression.
    pub max_depth: usize,
    /// Version tag recorded in manifests and archives.
    pub version: u32,
}

/// A grammar-level defect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    /// No inputs were declared; no expression could ever be built.
    NoInputs,
    /// No outputs were declared; no recurrence could be observed.
    NoOutputs,
    /// Input/output names must be non-empty and unique within each list.
    InvalidNames,
    /// A declared constant is not finite.
    NonFiniteConstant,
    /// Node or depth budget is zero.
    EmptyBudget,
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::NoInputs => write!(formatter, "grammar declares no inputs"),
            Self::NoOutputs => write!(formatter, "grammar declares no outputs"),
            Self::InvalidNames => write!(
                formatter,
                "grammar input/output names must be non-empty and unique"
            ),
            Self::NonFiniteConstant => write!(formatter, "grammar constants must be finite"),
            Self::EmptyBudget => write!(formatter, "grammar node/depth budgets must be non-zero"),
        }
    }
}

impl std::error::Error for GrammarError {}

impl GrammarSpec {
    /// Validate the grammar's own preconditions.
    ///
    /// # Errors
    ///
    /// See [`GrammarError`].
    pub fn validate(&self) -> Result<(), GrammarError> {
        if self.inputs.is_empty() {
            return Err(GrammarError::NoInputs);
        }
        if self.outputs.is_empty() {
            return Err(GrammarError::NoOutputs);
        }
        if !names_are_valid(&self.inputs) || !names_are_valid(&self.outputs) {
            return Err(GrammarError::InvalidNames);
        }
        if self.constants.iter().any(|value| !value.is_finite()) {
            return Err(GrammarError::NonFiniteConstant);
        }
        if self.max_nodes == 0 || self.max_depth == 0 {
            return Err(GrammarError::EmptyBudget);
        }
        Ok(())
    }

    /// Number of declared variables.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of recurrence outputs required from each candidate.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Total leaf count (variables plus constants).
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.inputs.len() + self.constants.len()
    }

    /// Leaf `index` as an expression (variables first, then constants).
    #[must_use]
    pub fn leaf(&self, index: usize) -> Option<Expr> {
        if index < self.inputs.len() {
            return Some(Expr::Var(index));
        }
        self.constants
            .get(index - self.inputs.len())
            .map(|value| Expr::Const(*value))
    }

    /// Stable digest over the grammar definition (names, constants by bit
    /// pattern, operator flags, budgets, version).
    #[must_use]
    pub fn digest(&self) -> String {
        let mut writer = DigestWriter::new(b"ADA-GRAMMAR-v1");
        let _ = writer.usize(self.inputs.len());
        for name in &self.inputs {
            let _ = writer.str(name);
        }
        let _ = writer.usize(self.outputs.len());
        for name in &self.outputs {
            let _ = writer.str(name);
        }
        let _ = writer.usize(self.constants.len());
        for value in &self.constants {
            writer.f64(*value);
        }
        writer.u8(u8::from(self.operators.add));
        writer.u8(u8::from(self.operators.sub));
        writer.u8(u8::from(self.operators.mul));
        writer.u8(u8::from(self.operators.max));
        writer.u8(u8::from(self.operators.exp));
        let _ = writer.usize(self.max_nodes);
        let _ = writer.usize(self.max_depth);
        writer.u32(self.version);
        hex(&writer.finish())
    }
}

fn names_are_valid(names: &[String]) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    names
        .iter()
        .all(|name| !name.is_empty() && seen.insert(name.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e0_grammar() -> GrammarSpec {
        GrammarSpec {
            inputs: vec!["m_old".into(), "l_old".into(), "score".into()],
            outputs: vec!["m_new".into(), "l_new".into()],
            constants: vec![],
            operators: OperatorSet::all(),
            max_nodes: 24,
            max_depth: 10,
            version: 1,
        }
    }

    #[test]
    fn validates_and_indexes_leaves() {
        let grammar = e0_grammar();
        assert!(grammar.validate().is_ok());
        assert_eq!(grammar.input_count(), 3);
        assert_eq!(grammar.output_count(), 2);
        assert_eq!(grammar.leaf_count(), 3);
        assert_eq!(grammar.leaf(0), Some(Expr::Var(0)));
        assert_eq!(grammar.leaf(3), None);
    }

    #[test]
    fn rejects_bad_grammars() {
        let mut grammar = e0_grammar();
        grammar.inputs.clear();
        assert_eq!(grammar.validate(), Err(GrammarError::NoInputs));

        let mut grammar = e0_grammar();
        grammar.outputs.clear();
        assert_eq!(grammar.validate(), Err(GrammarError::NoOutputs));

        let mut grammar = e0_grammar();
        grammar.constants = vec![f64::NAN];
        assert_eq!(grammar.validate(), Err(GrammarError::NonFiniteConstant));

        let mut grammar = e0_grammar();
        grammar.max_nodes = 0;
        assert_eq!(grammar.validate(), Err(GrammarError::EmptyBudget));
    }

    #[test]
    fn digest_is_stable_and_content_sensitive() {
        let grammar = e0_grammar();
        assert_eq!(grammar.digest(), grammar.digest());

        let mut other = e0_grammar();
        other.max_nodes += 1;
        assert_ne!(grammar.digest(), other.digest());

        let mut renamed = e0_grammar();
        renamed.outputs[0] = "m_new_prime".into();
        assert_ne!(grammar.digest(), renamed.digest());
    }
}
