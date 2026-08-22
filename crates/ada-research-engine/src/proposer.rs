//! Candidate proposal boundary.
//!
//! A [`CandidateProposer`] is any source of raw candidate expressions:
//! deterministic enumeration, evolutionary search, symbolic search, a future
//! `SciRust` algogen adapter, an external LLM/`SciAgent`, or a human-supplied
//! list. The API is deliberately narrow:
//!
//! * proposers see the [`GrammarSpec`] and feedback about candidates **they
//!   already emitted** (their normalized forms and discovery-corpus losses,
//!   assigned by the engine's evaluator);
//! * proposers never see any corpus, never see oracle outputs, and can never
//!   reach the adversarial holdout — those types do not appear anywhere in
//!   this module;
//! * an externally proposed candidate is just another proposal source: it
//!   passes through exactly the same gates as everything else.
//!
//! There is no bypass: `propose` returns data, never verdicts.

use serde::{Deserialize, Serialize};

use crate::candidate::Candidate;
use crate::canon::hex;
use crate::digest_writer::DigestWriter;
use crate::gates::ResearchGate;
use crate::grammar::GrammarSpec;
use crate::problem::SearchBudget;

/// Coarse provenance label for manifests (per-proposer).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalSourceKind {
    /// Deterministic grammar enumeration.
    DeterministicEnumeration,
    /// Seeded evolutionary search.
    EvolutionarySearch,
    /// Fixed human-supplied list (used by tests and benchmarks).
    ManualList,
    /// Any future external proposer (`SciAgent`, LLM, symbolic search...).
    External { name: String },
}

/// Immutable, replay-relevant proposer configuration recorded in manifests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalDescriptor {
    pub kind: ProposalSourceKind,
    /// Versioned canonical configuration text. Numeric values use exact bit
    /// encodings where floating-point values are involved.
    pub configuration: String,
    /// SHA-256 of `(kind, configuration)` under a versioned domain prefix.
    pub digest: String,
}

impl ProposalDescriptor {
    #[must_use]
    pub fn new(kind: ProposalSourceKind, configuration: String) -> Self {
        let mut writer = DigestWriter::new(b"ADA-PROPOSER-DESCRIPTOR-v1\0");
        let _ = writer.str(&kind.to_string());
        let _ = writer.str(&configuration);
        let digest = hex(&writer.finish());
        Self {
            kind,
            configuration,
            digest,
        }
    }
}

impl std::fmt::Display for ProposalSourceKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeterministicEnumeration => write!(formatter, "deterministic_enumeration"),
            Self::EvolutionarySearch => write!(formatter, "evolutionary_search"),
            Self::ManualList => write!(formatter, "manual_list"),
            Self::External { name } => write!(formatter, "external:{name}"),
        }
    }
}

/// Fine-grained provenance of one proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProposalSource {
    /// Emitted by the enumerative proposer (`emission_index`).
    Enumerative { emission_index: usize },
    /// Emitted by the evolutionary proposer.
    Evolutionary {
        seed: u64,
        generation: usize,
        individual: usize,
    },
    /// Human-supplied, labeled.
    Manual { label: String },
    /// Emitted by a built-in engine strategy (still fully gated).
    Composer { strategy: String },
    /// Supplied by an external proposer.
    External { proposer_name: String, note: String },
}

impl std::fmt::Display for ProposalSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Enumerative { emission_index } => {
                write!(formatter, "enumerative#{emission_index}")
            }
            Self::Evolutionary {
                seed,
                generation,
                individual,
            } => write!(
                formatter,
                "evolutionary(seed={seed},gen={generation},{individual})"
            ),
            Self::Manual { label } => write!(formatter, "manual:{label}"),
            Self::Composer { strategy } => write!(formatter, "composer:{strategy}"),
            Self::External {
                proposer_name,
                note,
            } => write!(formatter, "external({proposer_name}):{note}"),
        }
    }
}

/// What the engine reports back to the proposer that emitted a candidate.
///
/// Feedback contains only engine-assigned facts about the proposer's own
/// prior emissions: the normalized expression, its discovery-corpus loss and
/// whether a gate has already rejected it. No corpus contents, no oracle
/// outputs, no holdout data.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// Normalized form of the emitted candidate.
    pub normalized_candidate: Candidate,
    /// Discovery-corpus mean squared relative error assigned by the engine
    /// (finite; execution failures receive [`crate::engine::EXECUTION_FAILURE_LOSS`]).
    pub train_loss: f64,
    /// Gate that rejected this candidate, if any, so far.
    pub rejected_at: Option<ResearchGate>,
}

/// Context handed to a proposer on each pull.
#[derive(Debug)]
pub struct ProposalContext<'a> {
    /// The search grammar. This is the entire problem view a proposer gets.
    pub grammar: &'a GrammarSpec,
    /// Deterministic counter budgets. This exposes no case or oracle data.
    pub budget: &'a SearchBudget,
    /// Engine-scored results for this proposer's earlier emissions, in pull
    /// order.
    pub feedback: &'a [ScoredCandidate],
}

/// One raw proposal from any source. The engine normalizes and validates it;
/// proposers cannot pre-clear gates.
#[derive(Debug, Clone)]
pub struct RawProposal {
    pub candidate: Candidate,
    pub source: ProposalSource,
}

/// A pluggable candidate source.
pub trait CandidateProposer {
    /// Provenance label recorded in manifests.
    fn descriptor(&self) -> ProposalDescriptor;

    /// Return the next raw proposal, or `None` when this source is exhausted.
    ///
    /// Implementations must be deterministic given their construction
    /// parameters and the sequence of feedback they receive.
    fn propose(&mut self, context: &ProposalContext<'_>) -> Option<RawProposal>;
}
