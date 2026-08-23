//! Research problem specification: corpora, grammar, budgets, tolerances.
//!
//! A [`ResearchProblem`] fully determines a run: the engine consumes it plus
//! a proposer set and produces an [`crate::ExperimentArchive`] deterministically.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::corpus::ProblemCorpus;
use crate::grammar::GrammarSpec;

/// Numerical survival thresholds per gate (maximum tolerated relative error
/// under the corpus protocol's scale convention).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerances {
    /// Probe-corpus threshold.
    #[serde(with = "crate::float_serde::scalar")]
    pub probe_max_rel_error: f64,
    /// Oracle-corpus threshold.
    #[serde(with = "crate::float_serde::scalar")]
    pub oracle_max_rel_error: f64,
    /// Adversarial-holdout threshold.
    #[serde(with = "crate::float_serde::scalar")]
    pub holdout_max_rel_error: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            probe_max_rel_error: 1.0e-9,
            oracle_max_rel_error: 1.0e-9,
            holdout_max_rel_error: 1.0e-9,
        }
    }
}

/// Deterministic search budget. Every field is a pure counter; none involve
/// wall-clock time, so stopping criteria are replay-stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchBudget {
    /// Maximum raw candidates pulled across all proposers, including
    /// malformed and duplicate candidates.
    pub max_generated_candidates: u64,
    /// Maximum canonical-unique candidates evaluated on the discovery corpus.
    pub max_candidate_evaluations: u64,
    /// Maximum candidates processed through the oracle + holdout gates in
    /// the finalize phase (cheapest-first by ranking).
    pub max_gate_evaluations: usize,
    /// Maximum oracle cases declared by this experiment.
    pub max_oracle_cases: usize,
    /// Maximum adversarial/holdout cases declared by this experiment.
    pub max_adversarial_cases: usize,
    /// Maximum generations any iterative proposer may execute.
    pub max_generations: usize,
    /// Maximum retry attempts in one proposal mutation event.
    pub max_mutation_attempts: usize,
    /// Early-stop the search once this many candidates achieve bitwise-zero
    /// discovery loss (they still pass every gate afterwards).
    pub stop_after_train_exact: usize,
    /// Hall-of-fame capacity in the archive.
    pub hall_of_fame_capacity: usize,
    /// Counterexamples retained independently for each gate.
    pub max_counterexamples_per_gate: usize,
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            max_generated_candidates: 500_000,
            max_candidate_evaluations: 200_000,
            max_gate_evaluations: 64,
            max_oracle_cases: 1_024,
            max_adversarial_cases: 256,
            max_generations: 128,
            max_mutation_attempts: 8,
            stop_after_train_exact: 4,
            hall_of_fame_capacity: 16,
            max_counterexamples_per_gate: 16,
        }
    }
}

impl SearchBudget {
    /// Budget for small deterministic tests.
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            max_generated_candidates: 1_024,
            max_candidate_evaluations: 512,
            max_gate_evaluations: 8,
            max_oracle_cases: 1_024,
            max_adversarial_cases: 256,
            max_generations: 8,
            max_mutation_attempts: 4,
            stop_after_train_exact: 1,
            hall_of_fame_capacity: 8,
            max_counterexamples_per_gate: 4,
        }
    }
}

/// A complete, deterministic research problem.
#[derive(Clone)]
pub struct ResearchProblem {
    /// Stable problem name (recorded in manifests).
    pub name: String,
    /// Problem version (recorded in manifests).
    pub problem_version: u32,
    /// Experiment-level seed recorded in the manifest. Individual proposers
    /// derive their own seeds; this is the identity of the run.
    pub seed: u64,
    /// Candidate grammar. Knows variable names and operators only.
    pub grammar: GrammarSpec,
    /// Gate tolerances.
    pub tolerances: Tolerances,
    /// Search budget.
    pub budget: SearchBudget,
    /// Training corpus visible to the ranking loop (its *inputs* drive
    /// proposals only indirectly through engine-assigned losses).
    discovery_corpus: Arc<dyn ProblemCorpus>,
    /// Tiny cheap falsification corpus.
    probe_corpus: Arc<dyn ProblemCorpus>,
    /// Thorough normal-regime oracle corpus.
    oracle_corpus: Arc<dyn ProblemCorpus>,
    /// Adversarial holdout. Never touched during search; structurally
    /// inaccessible to proposers.
    adversarial_holdout: Arc<dyn ProblemCorpus>,
}

impl std::fmt::Debug for ResearchProblem {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResearchProblem")
            .field("name", &self.name)
            .field("problem_version", &self.problem_version)
            .field("seed", &self.seed)
            .field("grammar_version", &self.grammar.version)
            .field("budget", &self.budget)
            .field("tolerances", &self.tolerances)
            .field("discovery_cases", &self.discovery_corpus.len())
            .field("probe_cases", &self.probe_corpus.len())
            .field("oracle_cases", &self.oracle_corpus.len())
            .field("holdout_cases", &self.adversarial_holdout.len())
            .finish()
    }
}

impl ResearchProblem {
    /// Construct a problem while keeping all evaluation corpora behind the
    /// engine boundary. Proposers receive only the grammar and feedback.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        name: String,
        problem_version: u32,
        seed: u64,
        grammar: GrammarSpec,
        tolerances: Tolerances,
        budget: SearchBudget,
        discovery_corpus: Arc<dyn ProblemCorpus>,
        probe_corpus: Arc<dyn ProblemCorpus>,
        oracle_corpus: Arc<dyn ProblemCorpus>,
        adversarial_holdout: Arc<dyn ProblemCorpus>,
    ) -> Self {
        Self {
            name,
            problem_version,
            seed,
            grammar,
            tolerances,
            budget,
            discovery_corpus,
            probe_corpus,
            oracle_corpus,
            adversarial_holdout,
        }
    }

    pub(crate) fn discovery_corpus(&self) -> &dyn ProblemCorpus {
        self.discovery_corpus.as_ref()
    }

    pub(crate) fn probe_corpus(&self) -> &dyn ProblemCorpus {
        self.probe_corpus.as_ref()
    }

    pub(crate) fn oracle_corpus(&self) -> &dyn ProblemCorpus {
        self.oracle_corpus.as_ref()
    }

    pub(crate) fn adversarial_holdout(&self) -> &dyn ProblemCorpus {
        self.adversarial_holdout.as_ref()
    }

    /// Case counts by evidence role, without exposing inputs or oracle values.
    #[must_use]
    pub fn case_counts(&self) -> [(&'static str, usize); 4] {
        [
            (self.discovery_corpus.role(), self.discovery_corpus.len()),
            (self.probe_corpus.role(), self.probe_corpus.len()),
            (self.oracle_corpus.role(), self.oracle_corpus.len()),
            (
                self.adversarial_holdout.role(),
                self.adversarial_holdout.len(),
            ),
        ]
    }

    /// Validate the problem's own preconditions.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when the grammar is invalid or any
    /// corpus is empty.
    pub fn validate(&self) -> Result<(), String> {
        self.grammar
            .validate()
            .map_err(|error| format!("grammar invalid: {error}"))?;
        if self.budget.max_generated_candidates == 0
            || self.budget.max_candidate_evaluations == 0
            || self.budget.max_gate_evaluations == 0
            || self.budget.max_oracle_cases == 0
            || self.budget.max_adversarial_cases == 0
            || self.budget.max_generations == 0
            || self.budget.max_mutation_attempts == 0
            || self.budget.stop_after_train_exact == 0
        {
            return Err("search budgets must be non-zero".into());
        }
        for (name, tolerance) in [
            ("probe", self.tolerances.probe_max_rel_error),
            ("oracle", self.tolerances.oracle_max_rel_error),
            ("holdout", self.tolerances.holdout_max_rel_error),
        ] {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(format!("{name} tolerance must be finite and non-negative"));
            }
        }
        for corpus in [
            &self.discovery_corpus,
            &self.probe_corpus,
            &self.oracle_corpus,
            &self.adversarial_holdout,
        ] {
            if corpus.is_empty() {
                return Err(format!("corpus '{}' is empty", corpus.role()));
            }
        }
        if self.oracle_corpus.len() > self.budget.max_oracle_cases {
            return Err(format!(
                "oracle corpus has {} cases, exceeding budget {}",
                self.oracle_corpus.len(),
                self.budget.max_oracle_cases
            ));
        }
        if self.adversarial_holdout.len() > self.budget.max_adversarial_cases {
            return Err(format!(
                "adversarial holdout has {} cases, exceeding budget {}",
                self.adversarial_holdout.len(),
                self.budget.max_adversarial_cases
            ));
        }
        let mut roles = std::collections::BTreeSet::new();
        for (role, _) in self.case_counts() {
            if !roles.insert(role) {
                return Err(format!("duplicate corpus role '{role}'"));
            }
        }
        Ok(())
    }
}
