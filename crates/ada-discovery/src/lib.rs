//! End-to-end bounded discovery orchestration for ADA.
//!
//! This crate composes existing ADA layers without weakening their scientific
//! boundaries. It runs deterministic search and CEGIS first, then evaluates
//! only the final CEGIS survivors with a caller-owned objective evaluator and
//! places those independently supplied objective vectors into the existing
//! deterministic Pareto archive.
//!
//! The orchestrator does not invent oracle truth, adversarial fixtures,
//! objective weights, novelty, hardware measurements, or promotion verdicts.
//! In particular, a bounded CEGIS survivor may enter this layer only with
//! [`CorrectnessStatus::Provisional`]; finite-corpus survival cannot silently
//! self-promote to qualified correctness.

#![forbid(unsafe_code)]

use ada_cegis::{
    AdversarialGenerator, CegisConfig, CegisEngine, CegisError, CegisResult, DifferentialOracle,
    Fixture,
};
use ada_objective::{
    CandidateKey, CorrectnessStatus, ObjectiveError, ObjectiveVector, ParetoArchive, ParetoEntry,
};
use ada_search::{SearchCandidate, SearchEngine, SearchSpace};
use std::fmt::{Display, Formatter};

/// Fail-closed errors from the composed discovery lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryError {
    /// Search/CEGIS setup or execution failed.
    Cegis(CegisError),
    /// Objective construction or Pareto insertion failed.
    Objective(ObjectiveError),
    /// The caller-owned objective evaluator could not produce evidence.
    EvaluatorFailure(String),
    /// A finite-corpus survivor attempted to claim a stronger or weaker
    /// correctness status than the orchestrator can justify.
    InvalidSurvivorCorrectness(CorrectnessStatus),
}

impl Display for DiscoveryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cegis(error) => write!(formatter, "CEGIS discovery stage failed: {error}"),
            Self::Objective(error) => write!(formatter, "objective/Pareto stage failed: {error}"),
            Self::EvaluatorFailure(reason) => {
                write!(formatter, "objective evaluator failed: {reason}")
            }
            Self::InvalidSurvivorCorrectness(status) => write!(
                formatter,
                "bounded CEGIS survivor must be provisional, got {status}"
            ),
        }
    }
}

impl std::error::Error for DiscoveryError {}

impl From<CegisError> for DiscoveryError {
    fn from(error: CegisError) -> Self {
        Self::Cegis(error)
    }
}

impl From<ObjectiveError> for DiscoveryError {
    fn from(error: ObjectiveError) -> Self {
        Self::Objective(error)
    }
}

/// Caller-owned conversion from one final CEGIS survivor into independent
/// objective evidence.
///
/// The evaluator receives the final active fixture corpus so cost, numerical,
/// task, or externally measured evidence can explicitly bind to the corpus
/// that survived falsification. The orchestrator does not infer any objective
/// value itself.
pub trait ObjectiveEvaluator<C, I> {
    /// Error type for an evaluator infrastructure failure.
    type Error: Display;

    /// Evaluate one final survivor against the declared objective protocol.
    ///
    /// The returned vector must use [`CorrectnessStatus::Provisional`]. A
    /// stronger status requires a separate correctness protocol; a falsified or
    /// unknown status is inconsistent with entry into this survivor stage.
    ///
    /// # Errors
    ///
    /// Returns an error when the evaluator cannot produce its declared
    /// objective evidence. Such an error stops the discovery run rather than
    /// silently dropping an objective dimension or candidate.
    fn evaluate(
        &mut self,
        candidate: &SearchCandidate<C>,
        active_fixtures: &[Fixture<I>],
    ) -> Result<ObjectiveVector, Self::Error>;
}

/// One final CEGIS survivor and the exact objective vector supplied for it.
#[derive(Debug, Clone)]
pub struct EvaluatedSurvivor<C> {
    candidate: SearchCandidate<C>,
    objectives: ObjectiveVector,
}

impl<C> EvaluatedSurvivor<C> {
    /// Search candidate that survived the completed CEGIS run.
    #[must_use]
    pub const fn candidate(&self) -> &SearchCandidate<C> {
        &self.candidate
    }

    /// Independently supplied objective vector used for Pareto comparison.
    #[must_use]
    pub const fn objectives(&self) -> &ObjectiveVector {
        &self.objectives
    }
}

/// Complete evidence-preserving result of one bounded discovery run.
///
/// `evaluated` retains every final CEGIS survivor, including candidates later
/// found Pareto-dominated. Pareto entry payloads are indices into that vector,
/// so the frontier never becomes the only record of what was evaluated.
pub struct DiscoveryResult<C, I> {
    cegis: CegisResult<C, I>,
    evaluated: Vec<EvaluatedSurvivor<C>>,
    pareto: ParetoArchive<usize>,
}

impl<C, I> DiscoveryResult<C, I> {
    /// Full CEGIS result, including rejections and retained counterexamples.
    #[must_use]
    pub const fn cegis(&self) -> &CegisResult<C, I> {
        &self.cegis
    }

    /// Every final survivor and its exact evaluated objectives, in deterministic
    /// CEGIS survivor order.
    #[must_use]
    pub fn evaluated(&self) -> &[EvaluatedSurvivor<C>] {
        &self.evaluated
    }

    /// Strict non-weighted Pareto frontier over the evaluated survivors.
    ///
    /// Each entry payload is an index into [`Self::evaluated`].
    #[must_use]
    pub const fn pareto(&self) -> &ParetoArchive<usize> {
        &self.pareto
    }
}

/// Compose deterministic search, bounded CEGIS, objective evaluation, and
/// Pareto selection into one auditable run.
pub struct DiscoveryEngine<S, I, O, G, E>
where
    S: SearchSpace,
    I: Clone,
    O: DifferentialOracle<S::Candidate, I>,
    G: AdversarialGenerator<S::Candidate, I>,
    E: ObjectiveEvaluator<S::Candidate, I>,
{
    search: SearchEngine<S>,
    oracle: O,
    generator: G,
    evaluator: E,
    cegis_config: CegisConfig,
    initial_fixtures: Vec<Fixture<I>>,
}

impl<S, I, O, G, E> DiscoveryEngine<S, I, O, G, E>
where
    S: SearchSpace,
    I: Clone,
    O: DifferentialOracle<S::Candidate, I>,
    G: AdversarialGenerator<S::Candidate, I>,
    E: ObjectiveEvaluator<S::Candidate, I>,
{
    /// Construct a discovery engine without executing candidates.
    #[must_use]
    pub fn new(
        search: SearchEngine<S>,
        oracle: O,
        generator: G,
        evaluator: E,
        cegis_config: CegisConfig,
        initial_fixtures: Vec<Fixture<I>>,
    ) -> Self {
        Self {
            search,
            oracle,
            generator,
            evaluator,
            cegis_config,
            initial_fixtures,
        }
    }

    /// Execute the complete bounded lifecycle.
    ///
    /// The ordering is strict:
    ///
    /// 1. deterministic search generation;
    /// 2. CEGIS oracle/adversarial falsification and survivor revalidation;
    /// 3. objective evaluation of final survivors only;
    /// 4. strict Pareto archiving without scalar weights.
    ///
    /// A candidate rejected by CEGIS is never passed to the objective evaluator.
    /// A later objective cannot resurrect it. Likewise, bounded CEGIS survival
    /// is forced to remain provisional and cannot self-promote to qualified
    /// correctness.
    ///
    /// # Errors
    ///
    /// Returns an error for any search/CEGIS failure, evaluator infrastructure
    /// failure, invalid objective vector, non-provisional survivor correctness,
    /// or Pareto archive failure. The caller receives no partial success result
    /// when a required stage fails.
    pub fn run_to_end(self) -> Result<DiscoveryResult<S::Candidate, I>, DiscoveryError> {
        let Self {
            search,
            oracle,
            generator,
            mut evaluator,
            cegis_config,
            initial_fixtures,
        } = self;

        let cegis = CegisEngine::new(search, oracle, generator, cegis_config, initial_fixtures)?
            .run_to_end()?;

        let mut evaluated = Vec::with_capacity(cegis.survivors().len());
        let mut pareto = ParetoArchive::new();
        for survivor in cegis.survivors() {
            let objectives = evaluator
                .evaluate(survivor, cegis.active_fixtures())
                .map_err(|error| DiscoveryError::EvaluatorFailure(error.to_string()))?;
            objectives.validate()?;
            if objectives.correctness() != CorrectnessStatus::Provisional {
                return Err(DiscoveryError::InvalidSurvivorCorrectness(
                    objectives.correctness(),
                ));
            }

            let evaluated_index = evaluated.len();
            evaluated.push(EvaluatedSurvivor {
                candidate: survivor.clone(),
                objectives: objectives.clone(),
            });
            let key = CandidateKey::new(survivor.canonical_text().to_owned())?;
            let entry = ParetoEntry::new(key, objectives, evaluated_index)?;
            let _ = pareto.insert(entry)?;
        }

        Ok(DiscoveryResult {
            cegis,
            evaluated,
            pareto,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_cegis::{MAX_ACTIVE_FIXTURES, MAX_ADVERSARIAL_OUTPUTS, OracleOutcome};
    use ada_objective::{LogicalCost, ParetoDisposition};
    use ada_search::{SearchBudget, SearchError, SearchFingerprint};
    use std::convert::Infallible;

    #[derive(Clone)]
    struct TinySpace {
        candidates: Vec<u64>,
    }

    impl TinySpace {
        fn new() -> Self {
            Self {
                candidates: vec![0, 1, 2],
            }
        }
    }

    impl SearchSpace for TinySpace {
        type Candidate = u64;

        fn cardinality(&self) -> u64 {
            u64::try_from(self.candidates.len()).unwrap_or(u64::MAX)
        }

        fn fingerprint(&self) -> SearchFingerprint {
            SearchFingerprint::of_canonical_text("tiny-space-v1\n")
        }

        fn candidate_at(&self, ordinal: u64) -> Result<Self::Candidate, SearchError> {
            let index = usize::try_from(ordinal)
                .map_err(|_| SearchError::InvalidConfiguration("tiny ordinal"))?;
            self.candidates
                .get(index)
                .copied()
                .ok_or(SearchError::InvalidConfiguration("tiny ordinal"))
        }

        fn candidate_canonical_text(&self, candidate: &Self::Candidate) -> String {
            format!("tiny-candidate={candidate}\n")
        }

        fn candidate_cost(&self, candidate: &Self::Candidate) -> u32 {
            u32::try_from(*candidate).unwrap_or(u32::MAX)
        }
    }

    struct RejectMatchingFixture;

    impl DifferentialOracle<u64, u64> for RejectMatchingFixture {
        type Error = Infallible;

        fn compare(
            &mut self,
            candidate: &u64,
            fixture: &Fixture<u64>,
        ) -> Result<OracleOutcome, Self::Error> {
            if candidate == fixture.input() {
                Ok(OracleOutcome::Falsified {
                    reason: "candidate matches forbidden fixture".into(),
                })
            } else {
                Ok(OracleOutcome::Pass)
            }
        }
    }

    struct NoAdversary;

    impl AdversarialGenerator<u64, u64> for NoAdversary {
        type Error = Infallible;

        fn generate(
            &mut self,
            _seed: u64,
            _candidate: &u64,
            _active: &[Fixture<u64>],
        ) -> Result<Vec<Fixture<u64>>, Self::Error> {
            Ok(Vec::new())
        }
    }

    struct ProvisionalCost;

    impl ObjectiveEvaluator<u64, u64> for ProvisionalCost {
        type Error = Infallible;

        fn evaluate(
            &mut self,
            candidate: &SearchCandidate<u64>,
            _active_fixtures: &[Fixture<u64>],
        ) -> Result<ObjectiveVector, Self::Error> {
            let flops = 10 + *candidate.candidate() * 10;
            Ok(ObjectiveVector::new(CorrectnessStatus::Provisional)
                .with_logical(LogicalCost {
                    flops: Some(flops),
                    qk_evaluations: None,
                    transcendental_operations: None,
                    value_operations: None,
                })
                .expect("test logical objective is valid"))
        }
    }

    struct IllegalPromotion;

    impl ObjectiveEvaluator<u64, u64> for IllegalPromotion {
        type Error = Infallible;

        fn evaluate(
            &mut self,
            _candidate: &SearchCandidate<u64>,
            _active_fixtures: &[Fixture<u64>],
        ) -> Result<ObjectiveVector, Self::Error> {
            Ok(ObjectiveVector::new(CorrectnessStatus::Qualified))
        }
    }

    fn search() -> SearchEngine<TinySpace> {
        SearchEngine::new(
            TinySpace::new(),
            SearchBudget::new(3, 3, 64).expect("test search budget is valid"),
        )
        .expect("test search engine is valid")
    }

    fn cegis_config() -> CegisConfig {
        CegisConfig::new(7, MAX_ACTIVE_FIXTURES, 16, MAX_ADVERSARIAL_OUTPUTS)
            .expect("test CEGIS config is valid")
    }

    fn fixture() -> Fixture<u64> {
        Fixture::new("reject-one", "value=1\n", 1).expect("test fixture is valid")
    }

    #[test]
    fn only_final_cegis_survivors_enter_objective_and_pareto_stages() {
        let result = DiscoveryEngine::new(
            search(),
            RejectMatchingFixture,
            NoAdversary,
            ProvisionalCost,
            cegis_config(),
            vec![fixture()],
        )
        .run_to_end()
        .unwrap();

        assert_eq!(result.cegis().rejected().len(), 1);
        assert_eq!(result.cegis().survivors().len(), 2);
        assert_eq!(result.evaluated().len(), 2);
        assert!(
            result
                .evaluated()
                .iter()
                .all(|item| *item.candidate().candidate() != 1)
        );

        assert_eq!(result.pareto().entries().len(), 1);
        let frontier_index = *result.pareto().entries()[0].payload();
        assert_eq!(
            *result.evaluated()[frontier_index].candidate().candidate(),
            0
        );
        assert_eq!(result.pareto().decisions().len(), 2);
        assert_eq!(
            result.pareto().decisions()[1].disposition(),
            ParetoDisposition::Dominated
        );
    }

    #[test]
    fn bounded_survival_cannot_self_promote_to_qualified_correctness() {
        let error = DiscoveryEngine::new(
            search(),
            RejectMatchingFixture,
            NoAdversary,
            IllegalPromotion,
            cegis_config(),
            vec![fixture()],
        )
        .run_to_end()
        .err()
        .expect("illegal promotion must fail");

        assert_eq!(
            error,
            DiscoveryError::InvalidSurvivorCorrectness(CorrectnessStatus::Qualified)
        );
    }

    #[test]
    fn repeated_runs_preserve_evaluation_and_pareto_decision_order() {
        let run = || {
            DiscoveryEngine::new(
                search(),
                RejectMatchingFixture,
                NoAdversary,
                ProvisionalCost,
                cegis_config(),
                vec![fixture()],
            )
            .run_to_end()
            .unwrap()
        };
        let left = run();
        let right = run();

        let evaluated_keys = |result: &DiscoveryResult<u64, u64>| {
            result
                .evaluated()
                .iter()
                .map(|item| item.candidate().canonical_text().to_owned())
                .collect::<Vec<_>>()
        };
        let decisions = |result: &DiscoveryResult<u64, u64>| {
            result
                .pareto()
                .decisions()
                .iter()
                .map(|decision| {
                    (
                        decision.candidate_key().to_owned(),
                        decision.disposition(),
                        decision.dominator().map(str::to_owned),
                        decision.removed().to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(evaluated_keys(&left), evaluated_keys(&right));
        assert_eq!(decisions(&left), decisions(&right));
    }
}
