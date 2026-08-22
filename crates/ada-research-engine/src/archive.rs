//! Deterministic ARE-E0 experiment manifests, evaluation ledgers, and replay
//! verification.
//!
//! Archives contain no timestamps. Their integrity digest is computed over an
//! explicit, versioned byte encoding of every replay-relevant field; JSON is
//! only the inspectable transport representation.

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateError;
use crate::canon::hex;
use crate::cost::CostVector;
use crate::digest_writer::DigestWriter;
use crate::expr::{ExprError, OperatorKind};
use crate::gates::{CounterexampleRecord, RejectionReason, ResearchGate, SurvivalClass};
use crate::problem::{SearchBudget, Tolerances};
use crate::proposer::{ProposalDescriptor, ProposalSource, ProposalSourceKind};

/// Archive schema version.
pub const ARCHIVE_SCHEMA_VERSION: u32 = 2;

pub(crate) const SURVIVAL_CONTRACT: &str = "Finite deterministic cases can falsify a candidate; survival is not mathematical proof, and promotion is external.";

/// Why proposal generation stopped. Every option is counter-based and replay
/// stable; wall-clock time is deliberately absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchTermination {
    ProposersExhausted,
    GeneratedCandidateBudgetExhausted,
    CanonicalCandidateBudgetExhausted,
    DiscoveryExactSurvivorTargetReached,
}

/// Scientific outcome of the bounded experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentOutcome {
    /// No candidate survived all declared evidence gates.
    BoundedSearchNoSurvivor,
    /// At least one finalized candidate existed, but every one was falsified.
    AllFinalizedCandidatesFalsified,
    /// At least one candidate survived every declared finite case within the
    /// fixed tolerance.
    SurvivedDeclaredGatesWithinTolerance,
    /// At least one candidate matched every declared finite case exactly.
    SurvivedDeclaredGatesExactly,
}

impl std::fmt::Display for ExperimentOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::BoundedSearchNoSurvivor => "bounded_search_no_survivor",
            Self::AllFinalizedCandidatesFalsified => "all_finalized_candidates_falsified",
            Self::SurvivedDeclaredGatesWithinTolerance => {
                "survived_declared_gates_within_tolerance"
            }
            Self::SurvivedDeclaredGatesExactly => "survived_declared_gates_exactly",
        };
        formatter.write_str(value)
    }
}

/// Human promotion is intentionally outside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionState {
    NotApplicable,
    PromotionPending,
}

/// Stable corpus identity included in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusIdentity {
    pub role: String,
    pub case_count: usize,
    pub digest: String,
}

/// Immutable description of the logical inputs to a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub schema_version: u32,
    pub engine_version: String,
    /// Exact committed source revision supplied by the runner.
    pub source_revision: String,
    /// Declared floating-point/interpreter environment. Cross-platform `exp`
    /// bit identity is not assumed silently.
    pub numeric_semantics: String,
    pub experiment_id: String,
    pub problem_name: String,
    pub problem_version: u32,
    pub grammar_version: u32,
    pub grammar_digest: String,
    pub seed: u64,
    pub search_budget: SearchBudget,
    pub tolerances: Tolerances,
    /// Sorted by role.
    pub corpora: Vec<CorpusIdentity>,
    /// Pipeline order; full configurations, not just coarse source kinds.
    pub proposers: Vec<ProposalDescriptor>,
    pub survival_contract: String,
}

/// Aggregate accounting for every generated proposal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveStats {
    pub generated: u64,
    pub canonical_unique: u64,
    pub rejected_static: u64,
    pub rejected_duplicate: u64,
    pub rejected_probe: u64,
    pub rejected_oracle: u64,
    pub rejected_adversarial: u64,
    pub rejected_cost: u64,
    /// Numerically killed at probe, oracle, or adversarial gates (including
    /// non-finite execution). Static rejection and dedup are not falsification.
    pub falsified: u64,
    pub survived_probe: u64,
    pub survived_oracle: u64,
    pub survived_adversarial: u64,
    pub cost_qualified: u64,
    pub discovery_evaluations: u64,
    pub finalized_candidates: u64,
    pub oracle_cases_evaluated: u64,
    pub adversarial_cases_evaluated: u64,
}

/// Per-gate disposition. `NotRun` is evidence too: it makes budget-limited
/// qualification explicit rather than implying a candidate passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateDisposition {
    NotRun,
    Survived,
    Falsified,
    Rejected,
}

/// One gate result in a candidate's ordered evaluation history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateResult {
    pub gate: ResearchGate,
    pub disposition: GateDisposition,
    pub cases_evaluated: usize,
    pub all_exact: bool,
    #[serde(with = "crate::float_serde::optional")]
    pub max_abs_error: Option<f64>,
    #[serde(with = "crate::float_serde::optional")]
    pub max_rel_error: Option<f64>,
    pub rejection_reason: Option<RejectionReason>,
}

impl GateResult {
    #[must_use]
    pub fn not_run(gate: ResearchGate) -> Self {
        Self {
            gate,
            disposition: GateDisposition::NotRun,
            cases_evaluated: 0,
            all_exact: false,
            max_abs_error: None,
            max_rel_error: None,
            rejection_reason: None,
        }
    }
}

/// Complete disposition of one generated proposal. Digest collisions never
/// establish equality: `canonical_candidate` remains the structural key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub generated_index: u64,
    pub candidate_id: String,
    pub canonical_candidate: String,
    pub source: ProposalSource,
    #[serde(with = "crate::float_serde::optional")]
    pub train_loss: Option<f64>,
    pub structural_cost: Option<CostVector>,
    pub gate_results: Vec<GateResult>,
    pub final_class: SurvivalClass,
    pub pareto_member: bool,
}

/// One member of the nondominated set of fully qualified candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoEntry {
    pub candidate_id: String,
    pub canonical_candidate: String,
    #[serde(with = "crate::float_serde::scalar")]
    pub train_loss: f64,
    pub cost: CostVector,
    pub class: SurvivalClass,
    pub source: ProposalSource,
}

/// Compact best-evidence view; the full record remains in `evaluations`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestCandidateRecord {
    pub candidate_id: String,
    pub canonical_candidate: String,
    #[serde(with = "crate::float_serde::scalar")]
    pub train_loss: f64,
    pub cost: CostVector,
    pub class: SurvivalClass,
    pub promotion_state: PromotionState,
    pub source: ProposalSource,
}

/// Complete deterministic experiment artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentArchive {
    pub manifest: ExperimentManifest,
    pub termination: SearchTermination,
    pub stats: ArchiveStats,
    pub outcome: ExperimentOutcome,
    /// Digest of every proposal in emission order, including duplicates and
    /// malformed proposals. This independently commits to stream order.
    pub proposal_trajectory_digest: String,
    pub evaluations: Vec<CandidateEvaluation>,
    pub pareto_front: Vec<ParetoEntry>,
    pub hall_of_fame: Vec<BestCandidateRecord>,
    pub counterexamples: Vec<CounterexampleRecord>,
    pub rejection_reason_counts: Vec<(String, u64)>,
    pub best: Option<BestCandidateRecord>,
    /// Unkeyed SHA-256 content-integrity digest; not authentication.
    pub archive_digest: String,
}

/// Archive parsing or integrity failure.
#[derive(Debug)]
pub enum ArchiveError {
    Json(serde_json::Error),
    UnsupportedSchema { found: u32, expected: u32 },
    DigestMismatch { stored: String, recomputed: String },
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid archive JSON: {error}"),
            Self::UnsupportedSchema { found, expected } => {
                write!(
                    formatter,
                    "archive schema {found} is not supported (expected {expected})"
                )
            }
            Self::DigestMismatch { stored, recomputed } => {
                write!(
                    formatter,
                    "archive digest mismatch: stored {stored}, recomputed {recomputed}"
                )
            }
        }
    }
}

impl std::error::Error for ArchiveError {}

impl ExperimentArchive {
    /// Serialize as stable pretty JSON (struct field and vector order only;
    /// archives intentionally contain no map values).
    ///
    /// # Errors
    ///
    /// Returns a serialization error if an archive field cannot be encoded.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse and verify schema and content integrity.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed JSON, an unsupported schema, or a
    /// content-digest mismatch.
    pub fn from_json(json: &str) -> Result<Self, ArchiveError> {
        let archive: Self = serde_json::from_str(json).map_err(ArchiveError::Json)?;
        archive.verify()?;
        Ok(archive)
    }

    /// Verify schema and content digest.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported schema or digest mismatch.
    pub fn verify(&self) -> Result<(), ArchiveError> {
        if self.manifest.schema_version != ARCHIVE_SCHEMA_VERSION {
            return Err(ArchiveError::UnsupportedSchema {
                found: self.manifest.schema_version,
                expected: ARCHIVE_SCHEMA_VERSION,
            });
        }
        let recomputed = self.recompute_digest();
        if recomputed != self.archive_digest {
            return Err(ArchiveError::DigestMismatch {
                stored: self.archive_digest.clone(),
                recomputed,
            });
        }
        Ok(())
    }

    /// Recompute the versioned canonical content digest.
    #[must_use]
    pub fn recompute_digest(&self) -> String {
        let mut writer = DigestWriter::new(b"ADA-RESEARCH-ARCHIVE-v2\0");
        write_manifest(&self.manifest, &mut writer);
        write_termination(self.termination, &mut writer);
        write_stats(&self.stats, &mut writer);
        write_outcome(self.outcome, &mut writer);
        write_str(&mut writer, &self.proposal_trajectory_digest);
        write_len(&mut writer, self.evaluations.len());
        for evaluation in &self.evaluations {
            write_evaluation(evaluation, &mut writer);
        }
        write_len(&mut writer, self.pareto_front.len());
        for entry in &self.pareto_front {
            write_str(&mut writer, &entry.candidate_id);
            write_str(&mut writer, &entry.canonical_candidate);
            writer.f64(entry.train_loss);
            write_cost(&entry.cost, &mut writer);
            write_class(&entry.class, &mut writer);
            write_source(&entry.source, &mut writer);
        }
        write_len(&mut writer, self.hall_of_fame.len());
        for entry in &self.hall_of_fame {
            write_str(&mut writer, &entry.candidate_id);
            write_str(&mut writer, &entry.canonical_candidate);
            writer.f64(entry.train_loss);
            write_cost(&entry.cost, &mut writer);
            write_class(&entry.class, &mut writer);
            writer.u8(match entry.promotion_state {
                PromotionState::NotApplicable => 0,
                PromotionState::PromotionPending => 1,
            });
            write_source(&entry.source, &mut writer);
        }
        write_len(&mut writer, self.counterexamples.len());
        for record in &self.counterexamples {
            write_counterexample(record, &mut writer);
        }
        write_len(&mut writer, self.rejection_reason_counts.len());
        for (reason, count) in &self.rejection_reason_counts {
            write_str(&mut writer, reason);
            writer.u64(*count);
        }
        match &self.best {
            None => writer.bool(false),
            Some(best) => {
                writer.bool(true);
                write_str(&mut writer, &best.candidate_id);
                write_str(&mut writer, &best.canonical_candidate);
                writer.f64(best.train_loss);
                write_cost(&best.cost, &mut writer);
                write_class(&best.class, &mut writer);
                writer.u8(match best.promotion_state {
                    PromotionState::NotApplicable => 0,
                    PromotionState::PromotionPending => 1,
                });
                write_source(&best.source, &mut writer);
            }
        }
        hex(&writer.finish())
    }
}

fn write_manifest(manifest: &ExperimentManifest, writer: &mut DigestWriter) {
    writer.u32(manifest.schema_version);
    write_str(writer, &manifest.engine_version);
    write_str(writer, &manifest.source_revision);
    write_str(writer, &manifest.numeric_semantics);
    write_str(writer, &manifest.experiment_id);
    write_str(writer, &manifest.problem_name);
    writer.u32(manifest.problem_version);
    writer.u32(manifest.grammar_version);
    write_str(writer, &manifest.grammar_digest);
    writer.u64(manifest.seed);
    write_budget(&manifest.search_budget, writer);
    writer.f64(manifest.tolerances.probe_max_rel_error);
    writer.f64(manifest.tolerances.oracle_max_rel_error);
    writer.f64(manifest.tolerances.holdout_max_rel_error);
    write_len(writer, manifest.corpora.len());
    for corpus in &manifest.corpora {
        write_str(writer, &corpus.role);
        write_len(writer, corpus.case_count);
        write_str(writer, &corpus.digest);
    }
    write_len(writer, manifest.proposers.len());
    for proposer in &manifest.proposers {
        write_source_kind(&proposer.kind, writer);
        write_str(writer, &proposer.configuration);
        write_str(writer, &proposer.digest);
    }
    write_str(writer, &manifest.survival_contract);
}

fn write_budget(budget: &SearchBudget, writer: &mut DigestWriter) {
    writer.u64(budget.max_generated_candidates);
    writer.u64(budget.max_candidate_evaluations);
    write_len(writer, budget.max_gate_evaluations);
    write_len(writer, budget.max_oracle_cases);
    write_len(writer, budget.max_adversarial_cases);
    write_len(writer, budget.max_generations);
    write_len(writer, budget.max_mutation_attempts);
    write_len(writer, budget.stop_after_train_exact);
    write_len(writer, budget.hall_of_fame_capacity);
    write_len(writer, budget.max_counterexamples_per_gate);
}

fn write_stats(stats: &ArchiveStats, writer: &mut DigestWriter) {
    for value in [
        stats.generated,
        stats.canonical_unique,
        stats.rejected_static,
        stats.rejected_duplicate,
        stats.rejected_probe,
        stats.rejected_oracle,
        stats.rejected_adversarial,
        stats.rejected_cost,
        stats.falsified,
        stats.survived_probe,
        stats.survived_oracle,
        stats.survived_adversarial,
        stats.cost_qualified,
        stats.discovery_evaluations,
        stats.finalized_candidates,
        stats.oracle_cases_evaluated,
        stats.adversarial_cases_evaluated,
    ] {
        writer.u64(value);
    }
}

fn write_evaluation(evaluation: &CandidateEvaluation, writer: &mut DigestWriter) {
    writer.u64(evaluation.generated_index);
    write_str(writer, &evaluation.candidate_id);
    write_str(writer, &evaluation.canonical_candidate);
    write_source(&evaluation.source, writer);
    write_optional_f64(evaluation.train_loss, writer);
    match evaluation.structural_cost {
        None => writer.bool(false),
        Some(cost) => {
            writer.bool(true);
            write_cost(&cost, writer);
        }
    }
    write_len(writer, evaluation.gate_results.len());
    for result in &evaluation.gate_results {
        write_gate(result.gate, writer);
        writer.u8(match result.disposition {
            GateDisposition::NotRun => 0,
            GateDisposition::Survived => 1,
            GateDisposition::Falsified => 2,
            GateDisposition::Rejected => 3,
        });
        write_len(writer, result.cases_evaluated);
        writer.bool(result.all_exact);
        write_optional_f64(result.max_abs_error, writer);
        write_optional_f64(result.max_rel_error, writer);
        match &result.rejection_reason {
            None => writer.bool(false),
            Some(reason) => {
                writer.bool(true);
                write_reason(reason, writer);
            }
        }
    }
    write_class(&evaluation.final_class, writer);
    writer.bool(evaluation.pareto_member);
}

fn write_counterexample(record: &CounterexampleRecord, writer: &mut DigestWriter) {
    write_str(writer, &record.candidate_id);
    write_str(writer, &record.canonical_expression);
    write_gate(record.gate, writer);
    write_len(writer, record.case_index);
    write_len(writer, record.inputs.len());
    for (name, value) in &record.inputs {
        write_str(writer, name);
        writer.f64(*value);
    }
    match &record.candidate_output {
        None => writer.bool(false),
        Some(values) => {
            writer.bool(true);
            let _ = writer.f64_slice(values);
        }
    }
    let _ = writer.f64_slice(&record.oracle_output);
    write_optional_f64(record.absolute_error, writer);
    write_optional_f64(record.relative_error, writer);
}

fn write_cost(cost: &CostVector, writer: &mut DigestWriter) {
    writer.u32(cost.total_operators);
    writer.u32(cost.exp_count);
    writer.u32(cost.max_count);
    writer.u32(cost.mul_count);
    writer.u32(cost.add_sub_count);
    writer.u32(cost.depth);
    writer.u32(cost.state_outputs);
    writer.u32(cost.temporary_count);
}

fn write_class(class: &SurvivalClass, writer: &mut DigestWriter) {
    match class {
        SurvivalClass::Rejected { at_gate, reason } => {
            writer.u8(6);
            write_gate(*at_gate, writer);
            write_reason(reason, writer);
        }
        SurvivalClass::Falsified { at_gate, reason } => {
            writer.u8(0);
            write_gate(*at_gate, writer);
            write_reason(reason, writer);
        }
        SurvivalClass::SurvivedProbeOnly => writer.u8(1),
        SurvivalClass::SurvivedOracleCases => writer.u8(2),
        SurvivalClass::SurvivedAdversarialCases => writer.u8(3),
        SurvivalClass::CostQualified => writer.u8(4),
        SurvivalClass::BenchmarkEvidenceAvailable { artifact_digest } => {
            writer.u8(5);
            write_str(writer, artifact_digest);
        }
    }
}

fn write_reason(reason: &RejectionReason, writer: &mut DigestWriter) {
    match reason {
        RejectionReason::MalformedCandidate(error) => {
            writer.u8(0);
            write_candidate_error(error, writer);
        }
        RejectionReason::DuplicateCanonicalForm => writer.u8(1),
        RejectionReason::ProbeCorpusMismatch => writer.u8(2),
        RejectionReason::OracleCorpusMismatch => writer.u8(3),
        RejectionReason::AdversarialHoldoutMismatch => writer.u8(4),
        RejectionReason::NonFiniteExecution { gate, case_index } => {
            writer.u8(5);
            write_gate(*gate, writer);
            write_len(writer, *case_index);
        }
        RejectionReason::StructuralCostOverBudget => writer.u8(6),
    }
}

fn write_candidate_error(error: &CandidateError, writer: &mut DigestWriter) {
    match error {
        CandidateError::OutputArity { found, expected } => {
            writer.u8(0);
            write_len(writer, *found);
            write_len(writer, *expected);
        }
        CandidateError::NodeBudgetExceeded { nodes, maximum } => {
            writer.u8(1);
            write_len(writer, *nodes);
            write_len(writer, *maximum);
        }
        CandidateError::DepthBudgetExceeded { depth, maximum } => {
            writer.u8(2);
            write_len(writer, *depth);
            write_len(writer, *maximum);
        }
        CandidateError::MalformedOutput { output, error } => {
            writer.u8(3);
            write_len(writer, *output);
            write_expr_error(error, writer);
        }
    }
}

fn write_expr_error(error: &ExprError, writer: &mut DigestWriter) {
    match error {
        ExprError::UnknownVariable { index, available } => {
            writer.u8(0);
            write_len(writer, *index);
            write_len(writer, *available);
        }
        ExprError::NonFiniteConstant => writer.u8(1),
        ExprError::UndeclaredConstant { bits } => {
            writer.u8(2);
            writer.u64(*bits);
        }
        ExprError::UnsupportedOperator { operator } => {
            writer.u8(3);
            writer.u8(match operator {
                OperatorKind::Add => 0,
                OperatorKind::Sub => 1,
                OperatorKind::Mul => 2,
                OperatorKind::Max => 3,
                OperatorKind::Exp => 4,
            });
        }
        ExprError::NodeBudgetExceeded { nodes, maximum } => {
            writer.u8(4);
            write_len(writer, *nodes);
            write_len(writer, *maximum);
        }
        ExprError::DepthBudgetExceeded { depth, maximum } => {
            writer.u8(5);
            write_len(writer, *depth);
            write_len(writer, *maximum);
        }
    }
}

fn write_source(source: &ProposalSource, writer: &mut DigestWriter) {
    match source {
        ProposalSource::Enumerative { emission_index } => {
            writer.u8(0);
            write_len(writer, *emission_index);
        }
        ProposalSource::Evolutionary {
            seed,
            generation,
            individual,
        } => {
            writer.u8(1);
            writer.u64(*seed);
            write_len(writer, *generation);
            write_len(writer, *individual);
        }
        ProposalSource::Manual { label } => {
            writer.u8(2);
            write_str(writer, label);
        }
        ProposalSource::Composer { strategy } => {
            writer.u8(3);
            write_str(writer, strategy);
        }
        ProposalSource::External {
            proposer_name,
            note,
        } => {
            writer.u8(4);
            write_str(writer, proposer_name);
            write_str(writer, note);
        }
    }
}

fn write_source_kind(kind: &ProposalSourceKind, writer: &mut DigestWriter) {
    match kind {
        ProposalSourceKind::DeterministicEnumeration => writer.u8(0),
        ProposalSourceKind::EvolutionarySearch => writer.u8(1),
        ProposalSourceKind::ManualList => writer.u8(2),
        ProposalSourceKind::External { name } => {
            writer.u8(3);
            write_str(writer, name);
        }
    }
}

fn write_gate(gate: ResearchGate, writer: &mut DigestWriter) {
    writer.u8(match gate {
        ResearchGate::StaticValidation => 0,
        ResearchGate::CanonicalDedup => 1,
        ResearchGate::ProbeCorpus => 2,
        ResearchGate::OracleCorpus => 3,
        ResearchGate::AdversarialHoldout => 4,
        ResearchGate::StructuralCost => 5,
        ResearchGate::ParetoArchive => 6,
    });
}

fn write_termination(termination: SearchTermination, writer: &mut DigestWriter) {
    writer.u8(match termination {
        SearchTermination::ProposersExhausted => 0,
        SearchTermination::GeneratedCandidateBudgetExhausted => 1,
        SearchTermination::CanonicalCandidateBudgetExhausted => 2,
        SearchTermination::DiscoveryExactSurvivorTargetReached => 3,
    });
}

fn write_outcome(outcome: ExperimentOutcome, writer: &mut DigestWriter) {
    writer.u8(match outcome {
        ExperimentOutcome::BoundedSearchNoSurvivor => 0,
        ExperimentOutcome::AllFinalizedCandidatesFalsified => 1,
        ExperimentOutcome::SurvivedDeclaredGatesWithinTolerance => 2,
        ExperimentOutcome::SurvivedDeclaredGatesExactly => 3,
    });
}

fn write_optional_f64(value: Option<f64>, writer: &mut DigestWriter) {
    match value {
        None => writer.bool(false),
        Some(value) => {
            writer.bool(true);
            writer.f64(value);
        }
    }
}

fn write_len(writer: &mut DigestWriter, value: usize) {
    let _ = writer.usize(value);
}

fn write_str(writer: &mut DigestWriter, value: &str) {
    let _ = writer.str(value);
}

/// Field-by-field deterministic replay comparison.
#[must_use]
pub fn compare_archives(expected: &ExperimentArchive, actual: &ExperimentArchive) -> ReplayReport {
    let mut mismatches = Vec::new();
    macro_rules! compare {
        ($field:ident) => {
            if expected.$field != actual.$field {
                mismatches.push(stringify!($field).to_string());
            }
        };
    }
    compare!(manifest);
    compare!(termination);
    compare!(stats);
    compare!(outcome);
    compare!(proposal_trajectory_digest);
    compare!(evaluations);
    compare!(pareto_front);
    compare!(hall_of_fame);
    compare!(counterexamples);
    compare!(rejection_reason_counts);
    compare!(best);
    compare!(archive_digest);
    ReplayReport {
        identical: mismatches.is_empty(),
        mismatches,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub identical: bool,
    pub mismatches: Vec<String>,
}
