//! Deterministic falsification-first research engine.

use std::collections::BTreeMap;

use crate::archive::{
    ARCHIVE_SCHEMA_VERSION, ArchiveStats, BestCandidateRecord, CandidateEvaluation, CorpusIdentity,
    ExperimentArchive, ExperimentManifest, ExperimentOutcome, GateDisposition, GateResult,
    ParetoEntry, PromotionState, SURVIVAL_CONTRACT, SearchTermination,
};
use crate::candidate::Candidate;
use crate::canon::{candidate_canon_string, candidate_id, hex, normalize_candidate};
use crate::corpus::ProblemCorpus;
use crate::cost::CostVector;
use crate::digest_writer::DigestWriter;
use crate::gates::{CounterexampleRecord, RejectionReason, ResearchGate, SurvivalClass};
use crate::pareto::{ObjectiveView, pareto_front, total_order};
use crate::problem::ResearchProblem;
use crate::proposer::{
    CandidateProposer, ProposalContext, ProposalDescriptor, ProposalSource, ScoredCandidate,
};

/// Finite search loss assigned after any discovery-corpus execution failure.
pub const EXECUTION_FAILURE_LOSS: f64 = 1.0e6;

/// Signed log compression used only for proposal feedback and ranking.
#[must_use]
pub fn soft_log(value: f64) -> f64 {
    if value.is_sign_positive() {
        value.abs().ln_1p()
    } else {
        -value.abs().ln_1p()
    }
}

/// Engine inputs not already owned by [`ResearchProblem`].
pub struct EngineOptions {
    pub proposers: Vec<Box<dyn CandidateProposer>>,
    /// Exact committed revision of the engine code used for the run. Test
    /// harnesses may use another non-empty stable identifier.
    pub source_revision: String,
    pub structural_cost_budget: Option<CostVector>,
    pub stop_finalize_on_first_qualified: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            proposers: Vec::new(),
            source_revision: "uncommitted-or-unspecified".into(),
            structural_cost_budget: None,
            stop_finalize_on_first_qualified: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidProblem(String),
    NoProposers,
    MissingSourceRevision,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProblem(reason) => write!(formatter, "invalid problem: {reason}"),
            Self::NoProposers => formatter.write_str("no candidate proposers configured"),
            Self::MissingSourceRevision => formatter.write_str("source revision must not be empty"),
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Debug, Clone)]
struct CandidateRecord {
    candidate: Candidate,
    evaluation_index: usize,
    train_loss: f64,
    cost: CostVector,
}

/// Mean squared soft-log error over every case and output component.
/// Execution failure yields [`EXECUTION_FAILURE_LOSS`].
#[must_use]
pub fn discovery_loss(corpus: &dyn ProblemCorpus, candidate: &Candidate) -> f64 {
    let mut squared_error = 0.0_f64;
    let mut components = 0usize;
    for index in 0..corpus.len() {
        let Ok(run) = corpus.run_case(candidate, index) else {
            return EXECUTION_FAILURE_LOSS;
        };
        for (&actual, &expected) in run.candidate_output.iter().zip(&run.oracle_output) {
            let difference = soft_log(actual) - soft_log(expected);
            squared_error += difference * difference;
            components += 1;
        }
    }
    if !squared_error.is_finite() || components == 0 {
        return EXECUTION_FAILURE_LOSS;
    }
    #[allow(clippy::cast_precision_loss)]
    let divisor = components as f64;
    squared_error / divisor
}

struct ToleranceOutcome {
    result: GateResult,
    counterexample: Option<CounterexampleRecord>,
}

#[allow(clippy::too_many_arguments)]
fn tolerance_gate(
    corpus: &dyn ProblemCorpus,
    candidate: &Candidate,
    tolerance: f64,
    gate: ResearchGate,
    candidate_id: &str,
    canonical_candidate: &str,
) -> ToleranceOutcome {
    let mut all_exact = true;
    let mut maximum_absolute = 0.0_f64;
    let mut maximum_relative = 0.0_f64;
    for index in 0..corpus.len() {
        match corpus.run_case(candidate, index) {
            Ok(run) => {
                maximum_absolute = maximum_absolute.max(run.max_abs_error);
                maximum_relative = maximum_relative.max(run.max_rel_error);
                all_exact &= run.max_abs_error == 0.0;
                if run.max_rel_error > tolerance {
                    let reason = mismatch_reason(gate);
                    return ToleranceOutcome {
                        result: GateResult {
                            gate,
                            disposition: GateDisposition::Falsified,
                            cases_evaluated: index + 1,
                            all_exact: false,
                            max_abs_error: Some(maximum_absolute),
                            max_rel_error: Some(maximum_relative),
                            rejection_reason: Some(reason.clone()),
                        },
                        counterexample: Some(CounterexampleRecord {
                            candidate_id: candidate_id.into(),
                            canonical_expression: canonical_candidate.into(),
                            gate,
                            case_index: index,
                            inputs: corpus.describe_case(index),
                            candidate_output: Some(run.candidate_output),
                            oracle_output: run.oracle_output,
                            absolute_error: Some(run.max_abs_error),
                            relative_error: Some(run.max_rel_error),
                        }),
                    };
                }
            }
            Err(failure) => {
                let reason = RejectionReason::NonFiniteExecution {
                    gate,
                    case_index: failure.case_index,
                };
                return ToleranceOutcome {
                    result: GateResult {
                        gate,
                        disposition: GateDisposition::Falsified,
                        cases_evaluated: index + 1,
                        all_exact: false,
                        max_abs_error: if index == 0 {
                            None
                        } else {
                            Some(maximum_absolute)
                        },
                        max_rel_error: if index == 0 {
                            None
                        } else {
                            Some(maximum_relative)
                        },
                        rejection_reason: Some(reason),
                    },
                    counterexample: Some(CounterexampleRecord {
                        candidate_id: candidate_id.into(),
                        canonical_expression: canonical_candidate.into(),
                        gate,
                        case_index: failure.case_index,
                        inputs: corpus.describe_case(failure.case_index),
                        candidate_output: None,
                        oracle_output: corpus.oracle_output(failure.case_index),
                        absolute_error: None,
                        relative_error: None,
                    }),
                };
            }
        }
    }
    ToleranceOutcome {
        result: GateResult {
            gate,
            disposition: GateDisposition::Survived,
            cases_evaluated: corpus.len(),
            all_exact,
            max_abs_error: Some(maximum_absolute),
            max_rel_error: Some(maximum_relative),
            rejection_reason: None,
        },
        counterexample: None,
    }
}

const fn mismatch_reason(gate: ResearchGate) -> RejectionReason {
    match gate {
        ResearchGate::ProbeCorpus => RejectionReason::ProbeCorpusMismatch,
        ResearchGate::AdversarialHoldout => RejectionReason::AdversarialHoldoutMismatch,
        _ => RejectionReason::OracleCorpusMismatch,
    }
}

fn empty_gate_history() -> Vec<GateResult> {
    [
        ResearchGate::StaticValidation,
        ResearchGate::CanonicalDedup,
        ResearchGate::ProbeCorpus,
        ResearchGate::OracleCorpus,
        ResearchGate::AdversarialHoldout,
        ResearchGate::StructuralCost,
        ResearchGate::ParetoArchive,
    ]
    .into_iter()
    .map(GateResult::not_run)
    .collect()
}

fn set_gate(evaluation: &mut CandidateEvaluation, result: GateResult) {
    if let Some(slot) = evaluation
        .gate_results
        .iter_mut()
        .find(|slot| slot.gate == result.gate)
    {
        *slot = result;
    }
}

fn survived_without_cases(gate: ResearchGate) -> GateResult {
    GateResult {
        gate,
        disposition: GateDisposition::Survived,
        cases_evaluated: 0,
        all_exact: true,
        max_abs_error: None,
        max_rel_error: None,
        rejection_reason: None,
    }
}

struct SearchState<'a> {
    problem: &'a ResearchProblem,
    stats: ArchiveStats,
    rejection_counts: BTreeMap<&'static str, u64>,
    counterexamples: Vec<CounterexampleRecord>,
    counterexamples_by_gate: BTreeMap<ResearchGate, usize>,
    /// Digest -> canonical form -> known discovery loss. Nested structural
    /// equality prevents a hypothetical digest collision from deduplicating.
    seen: SeenCandidates,
    evaluations: Vec<CandidateEvaluation>,
    candidates: Vec<CandidateRecord>,
    probe_survivors: Vec<usize>,
    trajectory: DigestWriter,
}

#[derive(Debug, Default)]
struct SeenCandidates(BTreeMap<String, BTreeMap<String, f64>>);

impl SeenCandidates {
    fn loss(&self, digest: &str, canonical: &str) -> Option<f64> {
        self.0
            .get(digest)
            .and_then(|bucket| bucket.get(canonical))
            .copied()
    }

    fn insert(&mut self, digest: String, canonical: String, loss: f64) {
        self.0.entry(digest).or_default().insert(canonical, loss);
    }
}

enum Admission {
    SurvivedProbe { exact_discovery: bool },
    Rejected,
}

impl SearchState<'_> {
    fn bump_reason(&mut self, reason: &RejectionReason) {
        *self.rejection_counts.entry(reason.code()).or_default() += 1;
    }

    fn store_counterexample(&mut self, record: CounterexampleRecord) {
        let retained = self.counterexamples_by_gate.entry(record.gate).or_default();
        if *retained < self.problem.budget.max_counterexamples_per_gate {
            *retained += 1;
            self.counterexamples.push(record);
        }
    }

    fn commit_trajectory(
        &mut self,
        generated_index: u64,
        representation: &str,
        source: &ProposalSource,
    ) {
        self.trajectory.u64(generated_index);
        let _ = self.trajectory.str(representation);
        let _ = self.trajectory.str(&source.to_string());
    }

    // Kept linear intentionally: this is the auditable, ordered cheap-gate
    // transaction for one proposal.
    #[allow(clippy::too_many_lines)]
    fn admit(
        &mut self,
        raw_candidate: Candidate,
        source: ProposalSource,
        feedback: impl FnOnce(ScoredCandidate),
    ) -> Admission {
        let generated_index = self.stats.generated;
        self.stats.generated += 1;

        // Validate raw structure before normalization so simplification cannot
        // erase an illegal node, operator, constant, arity, or resource excess.
        if let Err(error) = raw_candidate.validate(&self.problem.grammar) {
            let raw = candidate_canon_string(&raw_candidate);
            let representation = format!(
                "(invalid-candidate raw={raw} outputs={} nodes={} depth={} error={error})",
                raw_candidate.output_arity(),
                raw_candidate.node_count(),
                raw_candidate.depth(),
            );
            let id = text_digest(b"ADA-INVALID-CANDIDATE-v1\0", &representation);
            self.commit_trajectory(generated_index, &representation, &source);
            let reason = RejectionReason::MalformedCandidate(error);
            self.bump_reason(&reason);
            self.stats.rejected_static += 1;
            let mut evaluation = CandidateEvaluation {
                generated_index,
                candidate_id: id,
                canonical_candidate: representation,
                source,
                train_loss: None,
                structural_cost: None,
                gate_results: empty_gate_history(),
                final_class: SurvivalClass::Rejected {
                    at_gate: ResearchGate::StaticValidation,
                    reason: Box::new(reason.clone()),
                },
                pareto_member: false,
            };
            set_gate(
                &mut evaluation,
                GateResult {
                    gate: ResearchGate::StaticValidation,
                    disposition: GateDisposition::Rejected,
                    cases_evaluated: 0,
                    all_exact: false,
                    max_abs_error: None,
                    max_rel_error: None,
                    rejection_reason: Some(reason),
                },
            );
            self.evaluations.push(evaluation);
            feedback(ScoredCandidate {
                normalized_candidate: raw_candidate,
                train_loss: EXECUTION_FAILURE_LOSS,
                rejected_at: Some(ResearchGate::StaticValidation),
            });
            return Admission::Rejected;
        }

        let candidate = normalize_candidate(&raw_candidate);
        let canonical = candidate_canon_string(&candidate);
        let id = candidate_id(&candidate);
        self.commit_trajectory(
            generated_index,
            &candidate_canon_string(&raw_candidate),
            &source,
        );

        let mut evaluation = CandidateEvaluation {
            generated_index,
            candidate_id: id.clone(),
            canonical_candidate: canonical.clone(),
            source: source.clone(),
            train_loss: None,
            structural_cost: None,
            gate_results: empty_gate_history(),
            final_class: SurvivalClass::SurvivedProbeOnly,
            pareto_member: false,
        };
        set_gate(
            &mut evaluation,
            survived_without_cases(ResearchGate::StaticValidation),
        );

        if let Some(loss) = self.seen.loss(&id, &canonical) {
            let reason = RejectionReason::DuplicateCanonicalForm;
            self.stats.rejected_duplicate += 1;
            self.bump_reason(&reason);
            set_gate(
                &mut evaluation,
                GateResult {
                    gate: ResearchGate::CanonicalDedup,
                    disposition: GateDisposition::Rejected,
                    cases_evaluated: 0,
                    all_exact: false,
                    max_abs_error: None,
                    max_rel_error: None,
                    rejection_reason: Some(reason.clone()),
                },
            );
            evaluation.final_class = SurvivalClass::Rejected {
                at_gate: ResearchGate::CanonicalDedup,
                reason: Box::new(reason),
            };
            self.evaluations.push(evaluation);
            feedback(ScoredCandidate {
                normalized_candidate: candidate,
                train_loss: loss,
                rejected_at: Some(ResearchGate::CanonicalDedup),
            });
            return Admission::Rejected;
        }

        self.stats.canonical_unique += 1;
        self.stats.discovery_evaluations += 1;
        set_gate(
            &mut evaluation,
            survived_without_cases(ResearchGate::CanonicalDedup),
        );
        let loss = discovery_loss(self.problem.discovery_corpus(), &candidate);
        let cost = CostVector::of(&candidate);
        evaluation.train_loss = Some(loss);
        evaluation.structural_cost = Some(cost);
        self.seen.insert(id.clone(), canonical.clone(), loss);

        let probe = tolerance_gate(
            self.problem.probe_corpus(),
            &candidate,
            self.problem.tolerances.probe_max_rel_error,
            ResearchGate::ProbeCorpus,
            &id,
            &canonical,
        );
        set_gate(&mut evaluation, probe.result.clone());
        if probe.result.disposition != GateDisposition::Survived {
            let reason = probe
                .result
                .rejection_reason
                .clone()
                .unwrap_or(RejectionReason::ProbeCorpusMismatch);
            self.stats.rejected_probe += 1;
            self.stats.falsified += 1;
            self.bump_reason(&reason);
            if let Some(counterexample) = probe.counterexample {
                self.store_counterexample(counterexample);
            }
            evaluation.final_class = SurvivalClass::Falsified {
                at_gate: ResearchGate::ProbeCorpus,
                reason: Box::new(reason),
            };
            let evaluation_index = self.evaluations.len();
            self.evaluations.push(evaluation);
            self.candidates.push(CandidateRecord {
                candidate: candidate.clone(),
                evaluation_index,
                train_loss: loss,
                cost,
            });
            feedback(ScoredCandidate {
                normalized_candidate: candidate,
                train_loss: loss,
                rejected_at: Some(ResearchGate::ProbeCorpus),
            });
            return Admission::Rejected;
        }

        self.stats.survived_probe += 1;
        let evaluation_index = self.evaluations.len();
        self.evaluations.push(evaluation);
        let candidate_index = self.candidates.len();
        self.candidates.push(CandidateRecord {
            candidate: candidate.clone(),
            evaluation_index,
            train_loss: loss,
            cost,
        });
        self.probe_survivors.push(candidate_index);
        feedback(ScoredCandidate {
            normalized_candidate: candidate,
            train_loss: loss,
            rejected_at: None,
        });
        Admission::SurvivedProbe {
            exact_discovery: loss == 0.0,
        }
    }
}

/// Run a complete bounded experiment.
///
/// # Errors
///
/// Returns an error if the problem contract is invalid, no proposer exists,
/// or the source revision is empty.
#[allow(clippy::too_many_lines)]
pub fn run_experiment(
    problem: &ResearchProblem,
    mut options: EngineOptions,
) -> Result<ExperimentArchive, EngineError> {
    problem.validate().map_err(EngineError::InvalidProblem)?;
    if options.proposers.is_empty() {
        return Err(EngineError::NoProposers);
    }
    if options.source_revision.trim().is_empty() {
        return Err(EngineError::MissingSourceRevision);
    }

    let descriptors: Vec<ProposalDescriptor> = options
        .proposers
        .iter()
        .map(|proposer| proposer.descriptor())
        .collect();
    let manifest = build_manifest(problem, &options.source_revision, descriptors);
    let mut state = SearchState {
        problem,
        stats: ArchiveStats::default(),
        rejection_counts: BTreeMap::new(),
        counterexamples: Vec::new(),
        counterexamples_by_gate: BTreeMap::new(),
        seen: SeenCandidates::default(),
        evaluations: Vec::new(),
        candidates: Vec::new(),
        probe_survivors: Vec::new(),
        trajectory: DigestWriter::new(b"ADA-PROPOSAL-TRAJECTORY-v1\0"),
    };
    let mut feedback: Vec<Vec<ScoredCandidate>> =
        (0..options.proposers.len()).map(|_| Vec::new()).collect();
    let mut retired = vec![false; options.proposers.len()];
    let mut exact_discovery_survivors = 0usize;

    let termination = 'search: loop {
        if state.stats.generated >= problem.budget.max_generated_candidates {
            break SearchTermination::GeneratedCandidateBudgetExhausted;
        }
        if state.stats.canonical_unique >= problem.budget.max_candidate_evaluations {
            break SearchTermination::CanonicalCandidateBudgetExhausted;
        }
        for (index, proposer) in options.proposers.iter_mut().enumerate() {
            if retired[index] {
                continue;
            }
            if state.stats.generated >= problem.budget.max_generated_candidates {
                break 'search SearchTermination::GeneratedCandidateBudgetExhausted;
            }
            if state.stats.canonical_unique >= problem.budget.max_candidate_evaluations {
                break 'search SearchTermination::CanonicalCandidateBudgetExhausted;
            }
            let context = ProposalContext {
                grammar: &problem.grammar,
                budget: &problem.budget,
                feedback: &feedback[index],
            };
            let Some(raw) = proposer.propose(&context) else {
                retired[index] = true;
                continue;
            };
            let sink = |scored| feedback[index].push(scored);
            if let Admission::SurvivedProbe {
                exact_discovery: true,
            } = state.admit(raw.candidate, raw.source, sink)
            {
                exact_discovery_survivors += 1;
                if exact_discovery_survivors >= problem.budget.stop_after_train_exact {
                    break 'search SearchTermination::DiscoveryExactSurvivorTargetReached;
                }
            }
        }
        if retired.iter().all(|retired| *retired) {
            break SearchTermination::ProposersExhausted;
        }
    };

    let mut all_ranked: Vec<usize> = (0..state.candidates.len()).collect();
    sort_candidate_indices(&state, &mut all_ranked);
    let mut finalize_ranked = state.probe_survivors.clone();
    sort_candidate_indices(&state, &mut finalize_ranked);

    let mut qualified = Vec::new();
    let mut exact_qualified = false;
    for &survivor_index in finalize_ranked
        .iter()
        .take(problem.budget.max_gate_evaluations)
    {
        state.stats.finalized_candidates += 1;
        let record = &state.candidates[survivor_index];
        let evaluation_index = record.evaluation_index;
        let candidate_id = state.evaluations[evaluation_index].candidate_id.clone();
        let canonical = state.evaluations[evaluation_index]
            .canonical_candidate
            .clone();

        let oracle = tolerance_gate(
            problem.oracle_corpus(),
            &record.candidate,
            problem.tolerances.oracle_max_rel_error,
            ResearchGate::OracleCorpus,
            &candidate_id,
            &canonical,
        );
        state.stats.oracle_cases_evaluated += oracle.result.cases_evaluated as u64;
        set_gate(
            &mut state.evaluations[evaluation_index],
            oracle.result.clone(),
        );
        if oracle.result.disposition != GateDisposition::Survived {
            reject_finalized(
                &mut state,
                evaluation_index,
                ResearchGate::OracleCorpus,
                &oracle,
            );
            state.stats.rejected_oracle += 1;
            continue;
        }
        state.stats.survived_oracle += 1;
        state.evaluations[evaluation_index].final_class = SurvivalClass::SurvivedOracleCases;

        let holdout = tolerance_gate(
            problem.adversarial_holdout(),
            &record.candidate,
            problem.tolerances.holdout_max_rel_error,
            ResearchGate::AdversarialHoldout,
            &candidate_id,
            &canonical,
        );
        state.stats.adversarial_cases_evaluated += holdout.result.cases_evaluated as u64;
        set_gate(
            &mut state.evaluations[evaluation_index],
            holdout.result.clone(),
        );
        if holdout.result.disposition != GateDisposition::Survived {
            reject_finalized(
                &mut state,
                evaluation_index,
                ResearchGate::AdversarialHoldout,
                &holdout,
            );
            state.stats.rejected_adversarial += 1;
            continue;
        }
        state.stats.survived_adversarial += 1;

        let within_cost = options
            .structural_cost_budget
            .is_none_or(|budget| !exceeds(&record.cost, &budget));
        if !within_cost {
            let reason = RejectionReason::StructuralCostOverBudget;
            set_gate(
                &mut state.evaluations[evaluation_index],
                GateResult {
                    gate: ResearchGate::StructuralCost,
                    disposition: GateDisposition::Rejected,
                    cases_evaluated: 0,
                    all_exact: false,
                    max_abs_error: None,
                    max_rel_error: None,
                    rejection_reason: Some(reason.clone()),
                },
            );
            state.evaluations[evaluation_index].final_class = SurvivalClass::Rejected {
                at_gate: ResearchGate::StructuralCost,
                reason: Box::new(reason.clone()),
            };
            state.stats.rejected_cost += 1;
            state.bump_reason(&reason);
            continue;
        }
        set_gate(
            &mut state.evaluations[evaluation_index],
            survived_without_cases(ResearchGate::StructuralCost),
        );
        state.evaluations[evaluation_index].final_class =
            if options.structural_cost_budget.is_some() {
                state.stats.cost_qualified += 1;
                SurvivalClass::CostQualified
            } else {
                SurvivalClass::SurvivedAdversarialCases
            };
        exact_qualified |= oracle.result.all_exact && holdout.result.all_exact;
        qualified.push(survivor_index);
        if options.stop_finalize_on_first_qualified {
            break;
        }
    }

    let qualified_views: Vec<_> = qualified
        .iter()
        .map(|&index| {
            let record = &state.candidates[index];
            ObjectiveView {
                loss: record.train_loss,
                cost: record.cost,
                canonical: &state.evaluations[record.evaluation_index].canonical_candidate,
            }
        })
        .collect();
    let pareto_positions = pareto_front(&qualified_views);
    let mut pareto_entries = Vec::new();
    for position in pareto_positions {
        let survivor_index = qualified[position];
        let record = &state.candidates[survivor_index];
        let evaluation = &mut state.evaluations[record.evaluation_index];
        evaluation.pareto_member = true;
        set_gate(
            evaluation,
            survived_without_cases(ResearchGate::ParetoArchive),
        );
        pareto_entries.push(ParetoEntry {
            candidate_id: evaluation.candidate_id.clone(),
            canonical_candidate: evaluation.canonical_candidate.clone(),
            train_loss: record.train_loss,
            cost: record.cost,
            class: evaluation.final_class.clone(),
            source: evaluation.source.clone(),
        });
    }

    let hall_of_fame: Vec<_> = all_ranked
        .iter()
        .take(problem.budget.hall_of_fame_capacity)
        .map(|&index| best_from(&state, index, qualified.contains(&index)))
        .collect();
    let best_index = qualified
        .first()
        .copied()
        .or_else(|| all_ranked.first().copied());
    let best = best_index.map(|index| best_from(&state, index, qualified.contains(&index)));
    let outcome = if qualified.is_empty() {
        if state.stats.finalized_candidates == 0 {
            ExperimentOutcome::BoundedSearchNoSurvivor
        } else {
            ExperimentOutcome::AllFinalizedCandidatesFalsified
        }
    } else if exact_qualified {
        ExperimentOutcome::SurvivedDeclaredGatesExactly
    } else {
        ExperimentOutcome::SurvivedDeclaredGatesWithinTolerance
    };
    let rejection_reason_counts = state
        .rejection_counts
        .into_iter()
        .map(|(reason, count)| (reason.into(), count))
        .collect();
    let proposal_trajectory_digest = hex(&state.trajectory.finish());
    let mut archive = ExperimentArchive {
        manifest,
        termination,
        stats: state.stats,
        outcome,
        proposal_trajectory_digest,
        evaluations: state.evaluations,
        pareto_front: pareto_entries,
        hall_of_fame,
        counterexamples: state.counterexamples,
        rejection_reason_counts,
        best,
        archive_digest: String::new(),
    };
    archive.archive_digest = archive.recompute_digest();
    Ok(archive)
}

fn reject_finalized(
    state: &mut SearchState<'_>,
    evaluation_index: usize,
    gate: ResearchGate,
    outcome: &ToleranceOutcome,
) {
    let reason = outcome
        .result
        .rejection_reason
        .clone()
        .unwrap_or_else(|| mismatch_reason(gate));
    state.evaluations[evaluation_index].final_class = SurvivalClass::Falsified {
        at_gate: gate,
        reason: Box::new(reason.clone()),
    };
    state.stats.falsified += 1;
    state.bump_reason(&reason);
    if let Some(counterexample) = outcome.counterexample.clone() {
        state.store_counterexample(counterexample);
    }
}

fn sort_candidate_indices(state: &SearchState<'_>, indices: &mut [usize]) {
    indices.sort_by(|&left, &right| {
        let left_record = &state.candidates[left];
        let right_record = &state.candidates[right];
        let left_evaluation = &state.evaluations[left_record.evaluation_index];
        let right_evaluation = &state.evaluations[right_record.evaluation_index];
        total_order(
            &ObjectiveView {
                loss: left_record.train_loss,
                cost: left_record.cost,
                canonical: &left_evaluation.canonical_candidate,
            },
            &ObjectiveView {
                loss: right_record.train_loss,
                cost: right_record.cost,
                canonical: &right_evaluation.canonical_candidate,
            },
        )
        .then_with(|| left.cmp(&right))
    });
}

fn exceeds(cost: &CostVector, budget: &CostVector) -> bool {
    cost.total_operators > budget.total_operators
        || cost.exp_count > budget.exp_count
        || cost.max_count > budget.max_count
        || cost.mul_count > budget.mul_count
        || cost.add_sub_count > budget.add_sub_count
        || cost.depth > budget.depth
        || cost.state_outputs > budget.state_outputs
        || cost.temporary_count > budget.temporary_count
}

fn best_from(
    state: &SearchState<'_>,
    survivor_index: usize,
    qualified: bool,
) -> BestCandidateRecord {
    let record = &state.candidates[survivor_index];
    let evaluation = &state.evaluations[record.evaluation_index];
    BestCandidateRecord {
        candidate_id: evaluation.candidate_id.clone(),
        canonical_candidate: evaluation.canonical_candidate.clone(),
        train_loss: record.train_loss,
        cost: record.cost,
        class: evaluation.final_class.clone(),
        promotion_state: if qualified {
            PromotionState::PromotionPending
        } else {
            PromotionState::NotApplicable
        },
        source: evaluation.source.clone(),
    }
}

fn build_manifest(
    problem: &ResearchProblem,
    source_revision: &str,
    proposers: Vec<ProposalDescriptor>,
) -> ExperimentManifest {
    let mut corpora = vec![
        corpus_identity(problem.discovery_corpus()),
        corpus_identity(problem.probe_corpus()),
        corpus_identity(problem.oracle_corpus()),
        corpus_identity(problem.adversarial_holdout()),
    ];
    corpora.sort_by(|left, right| left.role.cmp(&right.role));
    let experiment_id = experiment_id(problem, source_revision, &corpora, &proposers);
    ExperimentManifest {
        schema_version: ARCHIVE_SCHEMA_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").into(),
        source_revision: source_revision.into(),
        numeric_semantics: numeric_semantics(),
        experiment_id,
        problem_name: problem.name.clone(),
        problem_version: problem.problem_version,
        grammar_version: problem.grammar.version,
        grammar_digest: problem.grammar.digest(),
        seed: problem.seed,
        search_budget: problem.budget,
        tolerances: problem.tolerances,
        corpora,
        proposers,
        survival_contract: SURVIVAL_CONTRACT.into(),
    }
}

fn corpus_identity(corpus: &dyn ProblemCorpus) -> CorpusIdentity {
    CorpusIdentity {
        role: corpus.role().into(),
        case_count: corpus.len(),
        digest: corpus.digest(),
    }
}

fn experiment_id(
    problem: &ResearchProblem,
    source_revision: &str,
    corpora: &[CorpusIdentity],
    proposers: &[ProposalDescriptor],
) -> String {
    let mut writer = DigestWriter::new(b"ADA-EXPERIMENT-ID-v2\0");
    let _ = writer.str(env!("CARGO_PKG_VERSION"));
    let _ = writer.str(source_revision);
    let _ = writer.str(&numeric_semantics());
    let _ = writer.str(&problem.name);
    writer.u32(problem.problem_version);
    let _ = writer.str(&problem.grammar.digest());
    writer.u64(problem.seed);
    write_budget_identity(problem, &mut writer);
    let _ = writer.usize(corpora.len());
    for corpus in corpora {
        let _ = writer.str(&corpus.role);
        let _ = writer.usize(corpus.case_count);
        let _ = writer.str(&corpus.digest);
    }
    let _ = writer.usize(proposers.len());
    for proposer in proposers {
        let _ = writer.str(&proposer.digest);
    }
    hex(&writer.finish())
}

fn write_budget_identity(problem: &ResearchProblem, writer: &mut DigestWriter) {
    let budget = problem.budget;
    writer.u64(budget.max_generated_candidates);
    writer.u64(budget.max_candidate_evaluations);
    let _ = writer.usize(budget.max_gate_evaluations);
    let _ = writer.usize(budget.max_oracle_cases);
    let _ = writer.usize(budget.max_adversarial_cases);
    let _ = writer.usize(budget.max_generations);
    let _ = writer.usize(budget.max_mutation_attempts);
    let _ = writer.usize(budget.stop_after_train_exact);
    let _ = writer.usize(budget.hall_of_fame_capacity);
    let _ = writer.usize(budget.max_counterexamples_per_gate);
    writer.f64(problem.tolerances.probe_max_rel_error);
    writer.f64(problem.tolerances.oracle_max_rel_error);
    writer.f64(problem.tolerances.holdout_max_rel_error);
}

fn text_digest(domain: &[u8], text: &str) -> String {
    let mut writer = DigestWriter::new(domain);
    let _ = writer.str(text);
    hex(&writer.finish())
}

fn numeric_semantics() -> String {
    format!(
        "ieee754-f64;ordered-tree-evaluation;nonfinite-reject;std-exp;target={}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

#[cfg(test)]
mod tests {
    use super::SeenCandidates;

    #[test]
    fn digest_collision_does_not_imply_structural_equality() {
        let mut seen = SeenCandidates::default();
        seen.insert("synthetic-collision".into(), "candidate-a".into(), 1.0);
        assert_eq!(seen.loss("synthetic-collision", "candidate-a"), Some(1.0));
        assert_eq!(seen.loss("synthetic-collision", "candidate-b"), None);
        seen.insert("synthetic-collision".into(), "candidate-b".into(), 2.0);
        assert_eq!(seen.loss("synthetic-collision", "candidate-a"), Some(1.0));
        assert_eq!(seen.loss("synthetic-collision", "candidate-b"), Some(2.0));
    }
}
