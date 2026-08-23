//! ARE-E0: the ADA Research Engine.
//!
//! A deterministic research engine for automated algorithm discovery:
//!
//! ```text
//! PROPOSE -> STATIC VALIDATE -> FALSIFY (probe) -> ORACLE CHECK
//!         -> ADVERSARIAL CHECK -> COST -> RANK / PARETO -> ARCHIVE
//! ```
//!
//! # Trust boundaries
//!
//! * Passing finite oracle/adversarial cases is **empirical survival**, never
//!   mathematical proof; the vocabulary (`SurvivedOracleCases`,
//!   `SurvivedAdversarialCases`, `CostQualified`, `PromotionPending`, ...)
//!   contains no "proven" state.
//! * The engine never promotes anything into production ADA. Promotion is a
//!   human/manual gate outside this crate.
//! * Candidate proposal is pluggable ([`CandidateProposer`]); an external
//!   LLM/SciAgent/human candidate is just another proposal source and can
//!   never bypass validation gates.
//! * Proposers see only the [`GrammarSpec`] and engine-assigned losses for
//!   their own emissions. Corpora — including the adversarial holdout — are
//!   structurally inaccessible to them.
//! * Physical benchmark evidence has a reserved, unimplemented boundary
//!   ([`benchmark`]); discovery fitness never uses wall-clock time.
//!
//! # Determinism contract
//!
//! same problem + same engine version + same grammar + same seed
//! + same corpora + same budgets => bit/replay-stable archive.
//!
//! # E0 seed challenge
//!
//! [`online_softmax`] defines the first automated rediscovery experiment:
//! stable online-softmax normalizer recurrence over generated normal and
//! adversarial regimes.

#![forbid(unsafe_code)]

pub mod archive;
pub mod benchmark;
pub mod candidate;
pub mod canon;
pub mod corpus;
pub mod cost;
pub mod digest_writer;
pub mod engine;
pub mod expr;
mod float_serde;
pub mod gates;
pub mod grammar;
pub mod online_softmax;
pub mod pareto;
pub mod problem;
pub mod proposer;
pub mod proposers;
pub mod rng;

pub use archive::{
    ARCHIVE_SCHEMA_VERSION, ArchiveError, ArchiveStats, BestCandidateRecord, CandidateEvaluation,
    CorpusIdentity, ExperimentArchive, ExperimentManifest, ExperimentOutcome, GateDisposition,
    GateResult, ParetoEntry, PromotionState, ReplayReport, SearchTermination, compare_archives,
};
pub use candidate::{Candidate, CandidateError};
pub use canon::{
    candidate_canon_string, candidate_id, canon_string, normalize, normalize_candidate,
};
pub use corpus::{CaseRun, CorpusFailure, ProblemCorpus};
pub use cost::CostVector;
pub use engine::{EngineError, EngineOptions, discovery_loss, run_experiment};
pub use expr::{ExecError, Expr, ExprError};
pub use gates::{CounterexampleRecord, RejectionReason, ResearchGate, SurvivalClass};
pub use grammar::{GrammarError, GrammarSpec, OperatorSet};
pub use problem::{ResearchProblem, SearchBudget, Tolerances};
pub use proposer::{
    CandidateProposer, ProposalContext, ProposalDescriptor, ProposalSource, ProposalSourceKind,
    RawProposal, ScoredCandidate,
};
pub use proposers::{
    EnumerativeConfig, EnumerativeProposer, EvolutionaryConfig, EvolutionaryProposer,
    ManualCandidate, ManualProposer,
};
pub use rng::SearchRng;
