use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ada_research_engine::digest_writer::DigestWriter;
use ada_research_engine::{
    ArchiveError, Candidate, CandidateProposer, CaseRun, CorpusFailure, CostVector, EngineOptions,
    EnumerativeConfig, EnumerativeProposer, ExperimentArchive, ExperimentOutcome, Expr,
    GateDisposition, ManualCandidate, ManualProposer, OperatorSet, ProblemCorpus, ProposalContext,
    ProposalDescriptor, ProposalSource, ProposalSourceKind, RawProposal, ResearchGate,
    ResearchProblem, SearchBudget, SearchTermination, SurvivalClass, Tolerances,
    candidate_canon_string, compare_archives, run_experiment,
};

#[derive(Debug, Clone)]
struct IdentityCorpus {
    role: &'static str,
    inputs: Vec<f64>,
    calls: Option<Arc<AtomicUsize>>,
}

impl IdentityCorpus {
    fn new(role: &'static str, inputs: Vec<f64>) -> Self {
        Self {
            role,
            inputs,
            calls: None,
        }
    }

    fn tracked(role: &'static str, inputs: Vec<f64>, calls: Arc<AtomicUsize>) -> Self {
        Self {
            role,
            inputs,
            calls: Some(calls),
        }
    }
}

impl ProblemCorpus for IdentityCorpus {
    fn role(&self) -> &'static str {
        self.role
    }

    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn run_case(&self, candidate: &Candidate, index: usize) -> Result<CaseRun, CorpusFailure> {
        if let Some(calls) = &self.calls {
            calls.fetch_add(1, Ordering::SeqCst);
        }
        let expected = self.inputs[index];
        let candidate_output = candidate
            .eval(&[expected])
            .map_err(|reason| CorpusFailure {
                case_index: index,
                reason,
            })?;
        let actual = candidate_output[0];
        let absolute = (actual - expected).abs();
        Ok(CaseRun {
            candidate_output,
            oracle_output: vec![expected],
            max_abs_error: absolute,
            max_rel_error: ada_research_engine::corpus::relative_error(actual, expected),
        })
    }

    fn describe_case(&self, index: usize) -> Vec<(String, f64)> {
        vec![("x".into(), self.inputs[index])]
    }

    fn oracle_output(&self, index: usize) -> Vec<f64> {
        vec![self.inputs[index]]
    }

    fn digest_cases(&self, writer: &mut DigestWriter) {
        let _ = writer.f64_slice(&self.inputs);
    }
}

fn grammar() -> ada_research_engine::GrammarSpec {
    ada_research_engine::GrammarSpec {
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        constants: vec![0.0, f64::MAX],
        operators: OperatorSet::all(),
        max_nodes: 9,
        max_depth: 6,
        version: 7,
    }
}

fn problem_with_holdout(
    seed: u64,
    budget: SearchBudget,
    holdout: Arc<dyn ProblemCorpus>,
) -> ResearchProblem {
    ResearchProblem::new(
        "identity-contract-test".into(),
        1,
        seed,
        grammar(),
        Tolerances::default(),
        budget,
        Arc::new(IdentityCorpus::new("discovery", vec![-2.0, 0.5, 3.0])),
        Arc::new(IdentityCorpus::new("probe", vec![1.0, -1.0])),
        Arc::new(IdentityCorpus::new("oracle", vec![-5.0, -0.25, 2.0, 8.0])),
        holdout,
    )
}

fn problem(seed: u64, budget: SearchBudget) -> ResearchProblem {
    problem_with_holdout(
        seed,
        budget,
        Arc::new(IdentityCorpus::new(
            "adversarial_holdout",
            vec![-11.0, 0.125, 17.0],
        )),
    )
}

fn options(proposer: Box<dyn CandidateProposer>) -> EngineOptions {
    EngineOptions {
        proposers: vec![proposer],
        source_revision: "engine-contract-test-revision".into(),
        ..EngineOptions::default()
    }
}

fn identity() -> Candidate {
    Candidate::scalar(Expr::Var(0))
}

#[test]
fn raw_resource_excess_is_rejected_before_normalization() {
    // max(max(x,x),max(x,x)) normalizes to x, but its raw seven-node tree is
    // over a three-node budget and must never reach canonicalization.
    let mut constrained = grammar();
    constrained.max_nodes = 3;
    let oversized = Candidate::scalar(Expr::Max(
        Box::new(Expr::Max(Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
        Box::new(Expr::Max(Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
    ));
    let mut problem = problem(1, SearchBudget::tiny());
    problem.grammar = constrained;
    let archive = run_experiment(
        &problem,
        options(Box::new(ManualProposer::new(vec![
            ManualCandidate::recurrence("oversized", oversized),
        ]))),
    )
    .unwrap();
    assert_eq!(archive.stats.rejected_static, 1);
    assert!(matches!(
        archive.evaluations[0].final_class,
        SurvivalClass::Rejected {
            at_gate: ResearchGate::StaticValidation,
            ..
        }
    ));
}

#[test]
fn unsupported_variable_and_operator_are_static_rejections() {
    let mut restricted = grammar();
    restricted.operators.add = false;
    let mut problem = problem(2, SearchBudget::tiny());
    problem.grammar = restricted;
    let proposals = vec![
        ManualCandidate::recurrence("unknown-variable", Candidate::scalar(Expr::Var(9))),
        ManualCandidate::recurrence(
            "unsupported-add",
            Candidate::scalar(Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
        ),
    ];
    let archive =
        run_experiment(&problem, options(Box::new(ManualProposer::new(proposals)))).unwrap();
    assert_eq!(archive.stats.rejected_static, 2);
    assert!(
        archive.evaluations.iter().all(|evaluation| {
            evaluation.gate_results[0].disposition == GateDisposition::Rejected
        })
    );
}

#[test]
fn non_finite_candidate_is_falsified_with_counterexample() {
    let explosive = Candidate::scalar(Expr::Mul(
        Box::new(Expr::Const(f64::MAX)),
        Box::new(Expr::Const(f64::MAX)),
    ));
    let archive = run_experiment(
        &problem(3, SearchBudget::tiny()),
        options(Box::new(ManualProposer::new(vec![
            ManualCandidate::recurrence("overflow", explosive),
        ]))),
    )
    .unwrap();
    assert_eq!(archive.stats.falsified, 1);
    assert!(matches!(
        archive.evaluations[0].final_class,
        SurvivalClass::Falsified {
            at_gate: ResearchGate::ProbeCorpus,
            ..
        }
    ));
    assert_eq!(archive.counterexamples.len(), 1);
    assert!(archive.counterexamples[0].candidate_output.is_none());
}

#[derive(Debug)]
struct RepeatingProposer;

impl CandidateProposer for RepeatingProposer {
    fn descriptor(&self) -> ProposalDescriptor {
        ProposalDescriptor::new(
            ProposalSourceKind::External {
                name: "repeat-test".into(),
            },
            "v1".into(),
        )
    }

    fn propose(&mut self, _context: &ProposalContext<'_>) -> Option<RawProposal> {
        Some(RawProposal {
            candidate: identity(),
            source: ProposalSource::External {
                proposer_name: "repeat-test".into(),
                note: "same candidate forever".into(),
            },
        })
    }
}

#[test]
fn generated_budget_stops_an_infinite_duplicate_source() {
    let mut budget = SearchBudget::tiny();
    budget.max_generated_candidates = 5;
    budget.max_candidate_evaluations = 50;
    budget.stop_after_train_exact = usize::MAX;
    let archive =
        run_experiment(&problem(4, budget), options(Box::new(RepeatingProposer))).unwrap();
    assert_eq!(
        archive.termination,
        SearchTermination::GeneratedCandidateBudgetExhausted
    );
    assert_eq!(archive.stats.generated, 5);
    assert_eq!(archive.stats.canonical_unique, 1);
    assert_eq!(archive.stats.rejected_duplicate, 4);
    assert_eq!(archive.evaluations.len(), 5);
}

#[derive(Debug)]
struct HoldoutObservingProposer {
    holdout_calls: Arc<AtomicUsize>,
    next: usize,
}

impl CandidateProposer for HoldoutObservingProposer {
    fn descriptor(&self) -> ProposalDescriptor {
        ProposalDescriptor::new(
            ProposalSourceKind::External {
                name: "holdout-observer-test".into(),
            },
            "v1".into(),
        )
    }

    fn propose(&mut self, _context: &ProposalContext<'_>) -> Option<RawProposal> {
        assert_eq!(
            self.holdout_calls.load(Ordering::SeqCst),
            0,
            "holdout was evaluated while proposals were still being requested"
        );
        let candidate = match self.next {
            0 => identity(),
            1 => Candidate::scalar(Expr::Const(0.0)),
            _ => return None,
        };
        let position = self.next;
        self.next += 1;
        Some(RawProposal {
            candidate,
            source: ProposalSource::External {
                proposer_name: "holdout-observer-test".into(),
                note: format!("proposal-{position}"),
            },
        })
    }
}

#[test]
fn holdout_is_not_touched_until_all_proposal_calls_finish() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut budget = SearchBudget::tiny();
    budget.stop_after_train_exact = usize::MAX;
    let problem = problem_with_holdout(
        5,
        budget,
        Arc::new(IdentityCorpus::tracked(
            "adversarial_holdout",
            vec![-11.0, 17.0],
            Arc::clone(&calls),
        )),
    );
    let proposer = HoldoutObservingProposer {
        holdout_calls: Arc::clone(&calls),
        next: 0,
    };
    let archive = run_experiment(&problem, options(Box::new(proposer))).unwrap();
    assert!(calls.load(Ordering::SeqCst) > 0);
    assert_eq!(archive.stats.survived_adversarial, 1);
}

#[test]
fn same_seed_and_manifest_produce_byte_identical_archive() {
    let mut budget = SearchBudget::tiny();
    budget.max_generated_candidates = 80;
    budget.max_candidate_evaluations = 40;
    budget.stop_after_train_exact = usize::MAX;
    let run = || {
        run_experiment(
            &problem(9, budget),
            options(Box::new(EnumerativeProposer::new(EnumerativeConfig {
                max_nodes: 3,
                max_emissions: 80,
            }))),
        )
        .unwrap()
    };
    let first = run();
    let second = run();
    assert!(compare_archives(&first, &second).identical);
    assert_eq!(first.to_json().unwrap(), second.to_json().unwrap());
    assert_eq!(
        first.proposal_trajectory_digest,
        second.proposal_trajectory_digest
    );
}

#[test]
fn archive_round_trip_verifies_and_tampering_is_detected() {
    let archive = run_experiment(
        &problem(10, SearchBudget::tiny()),
        options(Box::new(ManualProposer::new(vec![
            ManualCandidate::recurrence("identity", identity()),
        ]))),
    )
    .unwrap();
    let parsed = ExperimentArchive::from_json(&archive.to_json().unwrap()).unwrap();
    assert!(compare_archives(&archive, &parsed).identical);
    let mut tampered = parsed;
    tampered.stats.generated += 1;
    assert!(matches!(
        tampered.verify(),
        Err(ArchiveError::DigestMismatch { .. })
    ));
}

#[test]
fn wrong_candidate_is_falsified_and_valid_fixture_survives_declared_gates() {
    let wrong = run_experiment(
        &problem(11, SearchBudget::tiny()),
        options(Box::new(ManualProposer::new(vec![
            ManualCandidate::recurrence("zero", Candidate::scalar(Expr::Const(0.0))),
        ]))),
    )
    .unwrap();
    assert_eq!(wrong.stats.falsified, 1);
    assert_eq!(wrong.outcome, ExperimentOutcome::BoundedSearchNoSurvivor);

    let valid = run_experiment(
        &problem(12, SearchBudget::tiny()),
        options(Box::new(ManualProposer::new(vec![
            ManualCandidate::recurrence("identity-test-fixture", identity()),
        ]))),
    )
    .unwrap();
    assert_eq!(
        valid.outcome,
        ExperimentOutcome::SurvivedDeclaredGatesExactly
    );
    assert_eq!(valid.stats.survived_oracle, 1);
    assert_eq!(valid.stats.survived_adversarial, 1);
    assert_eq!(valid.pareto_front.len(), 1);
}

#[test]
fn exact_outcome_requires_exact_probe_as_well_as_final_gates() {
    let tolerances = Tolerances {
        probe_max_rel_error: 1.0,
        oracle_max_rel_error: 1.0,
        holdout_max_rel_error: 1.0,
    };
    let problem = ResearchProblem::new(
        "exact-outcome-contract-test".into(),
        1,
        13,
        grammar(),
        tolerances,
        SearchBudget::tiny(),
        Arc::new(IdentityCorpus::new("discovery", vec![1.0e-12])),
        Arc::new(IdentityCorpus::new("probe", vec![1.0e-12])),
        Arc::new(IdentityCorpus::new("oracle", vec![0.0])),
        Arc::new(IdentityCorpus::new("adversarial_holdout", vec![0.0])),
    );
    let archive = run_experiment(
        &problem,
        options(Box::new(ManualProposer::new(vec![
            ManualCandidate::recurrence("zero", Candidate::scalar(Expr::Const(0.0))),
        ]))),
    )
    .unwrap();
    assert_eq!(
        archive.outcome,
        ExperimentOutcome::SurvivedDeclaredGatesWithinTolerance
    );
}

#[test]
fn cost_rejection_is_not_mislabeled_as_numerical_falsification() {
    let mut engine_options = options(Box::new(ManualProposer::new(vec![
        ManualCandidate::recurrence("identity", identity()),
    ])));
    engine_options.structural_cost_budget = Some(CostVector::default());
    let archive = run_experiment(&problem(14, SearchBudget::tiny()), engine_options).unwrap();
    assert_eq!(archive.stats.finalized_candidates, 1);
    assert_eq!(archive.stats.falsified, 0);
    assert_eq!(archive.stats.rejected_cost, 1);
    assert_eq!(archive.outcome, ExperimentOutcome::BoundedSearchNoSurvivor);
}

#[test]
fn built_in_proposer_sources_have_no_oracle_or_holdout_dependency() {
    let boundary = include_str!("../src/proposer.rs");
    let enumeration = include_str!("../src/proposers/enumerative.rs");
    let evolution = include_str!("../src/proposers/evolutionary.rs");
    for source in [boundary, enumeration, evolution] {
        assert!(!source.contains("use crate::online_softmax"));
        assert!(!source.contains("use crate::corpus"));
        assert!(!source.contains("oracle_step("));
        assert!(!source.contains("build_holdout("));
    }
    assert!(!enumeration.contains("m_old"));
    assert!(!evolution.contains("m_old"));
}

#[test]
fn candidate_output_order_remains_structural_identity() {
    let left = Candidate::new(vec![Expr::Var(0), Expr::Const(0.0)]);
    let right = Candidate::new(vec![Expr::Const(0.0), Expr::Var(0)]);
    assert_ne!(
        candidate_canon_string(&left),
        candidate_canon_string(&right)
    );
}

#[test]
fn real_e0_archive_survives_json_round_trip() {
    let mut problem = ada_research_engine::online_softmax::build_e0_problem(20_260_822);
    problem.budget.max_generated_candidates = 200;
    problem.budget.max_candidate_evaluations = 100;
    problem.budget.stop_after_train_exact = usize::MAX;
    let archive = run_experiment(
        &problem,
        options(Box::new(EnumerativeProposer::new(EnumerativeConfig {
            max_nodes: 3,
            max_emissions: 200,
        }))),
    )
    .unwrap();
    let json = archive.to_json().unwrap();
    let parsed: ExperimentArchive = serde_json::from_str(&json).unwrap();
    let report = compare_archives(&archive, &parsed);
    if !report.identical {
        if let Some((index, (left, right))) = archive
            .evaluations
            .iter()
            .zip(&parsed.evaluations)
            .enumerate()
            .find(|(_, (left, right))| left != right)
        {
            eprintln!("first evaluation mismatch at {index}:\nleft={left:#?}\nright={right:#?}");
        }
        if let Some((index, (left, right))) = archive
            .counterexamples
            .iter()
            .zip(&parsed.counterexamples)
            .enumerate()
            .find(|(_, (left, right))| left != right)
        {
            eprintln!(
                "first counterexample mismatch at {index}:\nleft={left:#?}\nright={right:#?}"
            );
        }
    }
    assert!(
        report.identical,
        "round-trip mismatches: {:?}",
        report.mismatches
    );
    parsed.verify().unwrap();
}
