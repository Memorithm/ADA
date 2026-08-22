//! Gate pipeline vocabulary: gates, rejection reasons, counterexamples.
//!
//! Gates are ordered cheapest-first. A candidate must survive every gate;
//! surviving a finite set of cases is *empirical survival*, never a
//! mathematical proof. The engine's vocabulary deliberately contains no
//! state named "proven".

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateError;

/// Stages of the falsification-first pipeline, cheapest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResearchGate {
    /// Static validation against the grammar budget.
    StaticValidation,
    /// Canonical deduplication of already-evaluated candidates.
    CanonicalDedup,
    /// Tiny counterexample corpus: a handful of cheap, regime-spanning cases.
    ProbeCorpus,
    /// Normal oracle corpus: thorough single-step validation.
    OracleCorpus,
    /// Adversarial holdout: multi-step streams and unseen extremes, never
    /// visible to any proposer during search.
    AdversarialHoldout,
    /// Structural cost qualification.
    StructuralCost,
    /// Final Pareto/ranking archive.
    ParetoArchive,
}

/// Why a candidate was rejected at a gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Failed static validation.
    MalformedCandidate(CandidateError),
    /// A canonically equivalent candidate was already evaluated.
    DuplicateCanonicalForm,
    /// Relative error exceeded tolerance on the probe corpus.
    ProbeCorpusMismatch,
    /// Relative error exceeded tolerance on the oracle corpus.
    OracleCorpusMismatch,
    /// Relative error exceeded tolerance on the adversarial holdout.
    AdversarialHoldoutMismatch,
    /// Execution produced a non-finite intermediate at this gate.
    NonFiniteExecution {
        gate: ResearchGate,
        case_index: usize,
    },
    /// Structural cost exceeds the configured budget.
    StructuralCostOverBudget,
}

impl RejectionReason {
    /// Stable snake-case tag used in archives (independent of `Display` text).
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MalformedCandidate(_) => "malformed_candidate",
            Self::DuplicateCanonicalForm => "duplicate_canonical_form",
            Self::ProbeCorpusMismatch => "probe_corpus_mismatch",
            Self::OracleCorpusMismatch => "oracle_corpus_mismatch",
            Self::AdversarialHoldoutMismatch => "adversarial_holdout_mismatch",
            Self::NonFiniteExecution { .. } => "non_finite_execution",
            Self::StructuralCostOverBudget => "structural_cost_over_budget",
        }
    }
}

impl std::fmt::Display for RejectionReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedCandidate(error) => write!(formatter, "malformed candidate: {error}"),
            Self::DuplicateCanonicalForm => write!(formatter, "duplicate canonical form"),
            Self::ProbeCorpusMismatch => write!(formatter, "probe corpus mismatch"),
            Self::OracleCorpusMismatch => write!(formatter, "oracle corpus mismatch"),
            Self::AdversarialHoldoutMismatch => write!(formatter, "adversarial holdout mismatch"),
            Self::NonFiniteExecution { gate, case_index } => {
                write!(
                    formatter,
                    "non-finite execution at {gate:?} case {case_index}"
                )
            }
            Self::StructuralCostOverBudget => write!(formatter, "structural cost over budget"),
        }
    }
}

/// Terminal survival classification of an evaluated candidate.
///
/// Terminology discipline: these names describe empirical outcomes against
/// finite case sets. Nothing here claims proof, novelty or production
/// readiness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurvivalClass {
    /// Structurally or procedurally rejected without a numerical
    /// counterexample (for example malformed, duplicate, or over cost).
    Rejected {
        at_gate: ResearchGate,
        reason: Box<RejectionReason>,
    },
    /// Died at some gate; see [`RejectionReason`] and counterexample records.
    Falsified {
        at_gate: ResearchGate,
        reason: Box<RejectionReason>,
    },
    /// Survived search-phase gates only (static, dedup, probe). Transient:
    /// present when finalize never processed the candidate.
    SurvivedProbeOnly,
    /// Survived every oracle-corpus case within tolerance.
    SurvivedOracleCases,
    /// Also survived every adversarial holdout case within tolerance.
    SurvivedAdversarialCases,
    /// Numerically qualified **and** within the structural cost budget.
    CostQualified,
    /// Reserved for externally supplied physical benchmark artifacts. The E0
    /// engine cannot construct this value from its own evaluation paths; only
    /// an external [`crate::benchmark`] provider could attach it explicitly.
    BenchmarkEvidenceAvailable { artifact_digest: String },
}

/// One deterministic minimal counterexample preserved when a candidate dies
/// numerically. Counterexamples are how ADA learns *why* candidates die.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterexampleRecord {
    /// Candidate that produced it.
    pub candidate_id: String,
    /// Canonical expression of the failing candidate.
    pub canonical_expression: String,
    /// Gate that surfaced the failure.
    pub gate: ResearchGate,
    /// Index of the failing case inside that gate's corpus.
    pub case_index: usize,
    /// Labeled input snapshot of the case.
    #[serde(with = "crate::float_serde::named_vector")]
    pub inputs: Vec<(String, f64)>,
    /// Candidate output (`None` on execution failure).
    #[serde(with = "crate::float_serde::optional_vector")]
    pub candidate_output: Option<Vec<f64>>,
    /// Oracle output for the same case.
    #[serde(with = "crate::float_serde::vector")]
    pub oracle_output: Vec<f64>,
    /// Absolute error (`None` on execution failure).
    #[serde(with = "crate::float_serde::optional")]
    pub absolute_error: Option<f64>,
    /// Relative error (`None` on execution failure).
    #[serde(with = "crate::float_serde::optional")]
    pub relative_error: Option<f64>,
}
