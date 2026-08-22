//! E0 challenge: automated rediscovery of a stable online-softmax state
//! transition.
//!
//! Candidates receive only `[m_old, l_old, score]` and emit the ordered state
//! tuple `[m_new, l_new]`.  The trusted recurrence is confined to this corpus
//! and oracle module; no proposer imports or receives it.

use std::sync::Arc;

use crate::candidate::Candidate;
use crate::corpus::{CaseRun, CorpusFailure, ProblemCorpus, relative_error};
use crate::digest_writer::DigestWriter;
use crate::grammar::{GrammarSpec, OperatorSet};
use crate::problem::{ResearchProblem, SearchBudget, Tolerances};
use crate::rng::SearchRng;

/// Problem name recorded in experiment manifests.
pub const PROBLEM_NAME: &str = "online_softmax_recurrence_e0";
/// Problem contract version.
pub const PROBLEM_VERSION: u32 = 2;
/// Grammar contract version.
pub const GRAMMAR_VERSION: u32 = 2;

/// Independent reference maximum.  This deliberately does not call the IR
/// interpreter's maximum helper, so candidate and oracle do not share an
/// implementation path.  The explicit zero rule documents the agreed E0
/// floating-point tie semantics.
fn oracle_m_new(m_old: f64, score: f64) -> f64 {
    if score > m_old {
        score
    } else if m_old > score {
        m_old
    } else if m_old == 0.0 {
        0.0
    } else {
        m_old
    }
}

/// Trusted reference state transition, confined to the evaluation layer.
fn oracle_step(m_old: f64, l_old: f64, score: f64) -> [f64; 2] {
    let m_new = oracle_m_new(m_old, score);
    let l_new = l_old * (m_old - m_new).exp() + (score - m_new).exp();
    [m_new, l_new]
}

/// One single-step case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCase {
    pub m_old: f64,
    pub l_old: f64,
    pub score: f64,
}

impl StepCase {
    #[must_use]
    pub const fn new(m_old: f64, l_old: f64, score: f64) -> Self {
        Self {
            m_old,
            l_old,
            score,
        }
    }

    fn inputs(self) -> [f64; 3] {
        [self.m_old, self.l_old, self.score]
    }

    fn describe(self) -> Vec<(String, f64)> {
        vec![
            ("m_old".into(), self.m_old),
            ("l_old".into(), self.l_old),
            ("score".into(), self.score),
        ]
    }
}

/// One multi-step adversarial rollout.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamCase {
    pub m_init: f64,
    pub l_init: f64,
    pub scores: Vec<f64>,
}

fn compare_outputs(candidate: &[f64], oracle: &[f64]) -> (f64, f64) {
    candidate.iter().zip(oracle).fold(
        (0.0_f64, 0.0_f64),
        |(maximum_absolute, maximum_relative), (&actual, &expected)| {
            (
                maximum_absolute.max((actual - expected).abs()),
                maximum_relative.max(relative_error(actual, expected)),
            )
        },
    )
}

/// Single-step corpus over [`StepCase`] values.
#[derive(Debug, Clone)]
pub struct StepCorpus {
    role: &'static str,
    cases: Vec<StepCase>,
}

impl StepCorpus {
    #[must_use]
    pub fn new(role: &'static str, cases: Vec<StepCase>) -> Self {
        Self { role, cases }
    }

    #[cfg(test)]
    fn cases(&self) -> &[StepCase] {
        &self.cases
    }
}

impl ProblemCorpus for StepCorpus {
    fn role(&self) -> &'static str {
        self.role
    }

    fn len(&self) -> usize {
        self.cases.len()
    }

    fn run_case(&self, candidate: &Candidate, index: usize) -> Result<CaseRun, CorpusFailure> {
        let case = self.cases[index];
        let candidate_output = candidate
            .eval(&case.inputs())
            .map_err(|reason| CorpusFailure {
                case_index: index,
                reason,
            })?;
        let oracle_output = oracle_step(case.m_old, case.l_old, case.score).to_vec();
        let (max_abs_error, max_rel_error) = compare_outputs(&candidate_output, &oracle_output);
        Ok(CaseRun {
            candidate_output,
            oracle_output,
            max_abs_error,
            max_rel_error,
        })
    }

    fn describe_case(&self, index: usize) -> Vec<(String, f64)> {
        self.cases[index].describe()
    }

    fn oracle_output(&self, index: usize) -> Vec<f64> {
        let case = self.cases[index];
        oracle_step(case.m_old, case.l_old, case.score).to_vec()
    }

    fn digest_cases(&self, writer: &mut DigestWriter) {
        writer.tag(0x50);
        writer.u32(PROBLEM_VERSION);
        let _ = writer.usize(self.cases.len());
        for case in &self.cases {
            writer.f64(case.m_old);
            writer.f64(case.l_old);
            writer.f64(case.score);
            for output in oracle_step(case.m_old, case.l_old, case.score) {
                writer.f64(output);
            }
        }
    }
}

/// Multi-step corpus. Candidate and oracle states advance independently, so
/// a locally plausible but unstable recurrence is exposed by rollout drift.
#[derive(Debug, Clone)]
pub struct StreamCorpus {
    role: &'static str,
    cases: Vec<StreamCase>,
}

impl StreamCorpus {
    #[must_use]
    pub fn new(role: &'static str, cases: Vec<StreamCase>) -> Self {
        Self { role, cases }
    }

    #[cfg(test)]
    fn cases(&self) -> &[StreamCase] {
        &self.cases
    }

    fn oracle_rollout(case: &StreamCase) -> [f64; 2] {
        let mut state = [case.m_init, case.l_init];
        for &score in &case.scores {
            state = oracle_step(state[0], state[1], score);
        }
        state
    }
}

impl ProblemCorpus for StreamCorpus {
    fn role(&self) -> &'static str {
        self.role
    }

    fn len(&self) -> usize {
        self.cases.len()
    }

    fn run_case(&self, candidate: &Candidate, index: usize) -> Result<CaseRun, CorpusFailure> {
        let case = &self.cases[index];
        let mut candidate_state = vec![case.m_init, case.l_init];
        let mut oracle_state = [case.m_init, case.l_init];
        let mut maximum_absolute = 0.0_f64;
        let mut maximum_relative = 0.0_f64;

        for &score in &case.scores {
            candidate_state = candidate
                .eval(&[candidate_state[0], candidate_state[1], score])
                .map_err(|reason| CorpusFailure {
                    case_index: index,
                    reason,
                })?;
            oracle_state = oracle_step(oracle_state[0], oracle_state[1], score);
            let (absolute, relative) = compare_outputs(&candidate_state, &oracle_state);
            maximum_absolute = maximum_absolute.max(absolute);
            maximum_relative = maximum_relative.max(relative);
        }

        Ok(CaseRun {
            candidate_output: candidate_state,
            oracle_output: oracle_state.to_vec(),
            max_abs_error: maximum_absolute,
            max_rel_error: maximum_relative,
        })
    }

    fn describe_case(&self, index: usize) -> Vec<(String, f64)> {
        let case = &self.cases[index];
        let mut description = vec![
            ("m_init".into(), case.m_init),
            ("l_init".into(), case.l_init),
            (
                "steps".into(),
                u32::try_from(case.scores.len()).map_or(f64::from(u32::MAX), f64::from),
            ),
        ];
        description.extend(
            case.scores
                .iter()
                .enumerate()
                .map(|(position, &score)| (format!("score_{position}"), score)),
        );
        description
    }

    fn oracle_output(&self, index: usize) -> Vec<f64> {
        Self::oracle_rollout(&self.cases[index]).to_vec()
    }

    fn digest_cases(&self, writer: &mut DigestWriter) {
        writer.tag(0x51);
        writer.u32(PROBLEM_VERSION);
        let _ = writer.usize(self.cases.len());
        for case in &self.cases {
            writer.f64(case.m_init);
            writer.f64(case.l_init);
            let _ = writer.usize(case.scores.len());
            for &score in &case.scores {
                writer.f64(score);
            }
            for output in Self::oracle_rollout(case) {
                writer.f64(output);
            }
        }
    }
}

/// E0 grammar: general variables and mathematical ingredients only.  It
/// contains no target AST, target fragment, supplied future state, or rewrite.
#[must_use]
pub fn e0_grammar() -> GrammarSpec {
    GrammarSpec {
        inputs: vec!["m_old".into(), "l_old".into(), "score".into()],
        outputs: vec!["m_new".into(), "l_new".into()],
        constants: vec![],
        operators: OperatorSet::all(),
        max_nodes: 30,
        max_depth: 10,
        version: GRAMMAR_VERSION,
    }
}

#[allow(clippy::cast_precision_loss)]
fn jitter(rng: &mut SearchRng, low: f64, high: f64) -> f64 {
    low + rng.unit() * (high - low)
}

#[allow(clippy::cast_precision_loss)]
fn log_uniform(rng: &mut SearchRng, low_exp: i32, high_exp: i32) -> f64 {
    let exponent = f64::from(low_exp) + rng.unit() * f64::from(high_exp - low_exp);
    10f64.powf(exponent)
}

fn build_discovery_corpus(seed: u64) -> StepCorpus {
    let mut cases = Vec::new();
    for delta in [-40.0, -2.0, 0.0, 0.5, 8.0] {
        for m_old in [-9.0, 0.0, 9.0] {
            for l_old in [1.0e-20, 1.0, 1.0e20] {
                cases.push(StepCase::new(m_old, l_old, m_old + delta));
            }
        }
    }
    let mut rng = SearchRng::new(seed);
    for _ in 0..51 {
        let m_old = jitter(&mut rng, -15.0, 15.0);
        let l_old = log_uniform(&mut rng, -25, 25);
        let delta = jitter(&mut rng, -45.0, 45.0);
        cases.push(StepCase::new(m_old, l_old, m_old + delta));
    }
    StepCorpus::new("discovery", cases)
}

fn build_probe_corpus() -> StepCorpus {
    StepCorpus::new(
        "probe",
        vec![
            StepCase::new(0.0, 1.0, -40.0),
            StepCase::new(-3.0, 1.0, -3.01),
            StepCase::new(5.0, 1.0, 5.0),
            StepCase::new(0.0, 1.0e-10, 0.5),
            StepCase::new(3.0, 1.0e-30, 11.0),
            StepCase::new(-3.0, 1.0e30, -11.0),
        ],
    )
}

fn build_oracle_corpus() -> StepCorpus {
    let mut cases = Vec::new();
    for delta in [
        -120.0, -60.0, -20.0, -4.0, -1.0, -0.25, -0.01, 0.0, 0.01, 0.25, 1.0, 4.0, 20.0, 60.0,
    ] {
        // 0.125 (rather than the discovery grid's 0.0) keeps the explicit
        // oracle partition free of exact discovery-case duplicates.
        for m_old in [-30.0, -10.0, -3.0, 0.125, 3.0, 10.0, 30.0] {
            for l_old in [1.0e-38, 1.0e-20, 1.0e-6, 1.0, 1.0e6, 1.0e20, 1.0e38] {
                cases.push(StepCase::new(m_old, l_old, m_old + delta));
            }
        }
    }
    StepCorpus::new("oracle", cases)
}

fn build_holdout(seed: u64) -> (StepCorpus, StreamCorpus) {
    let mut steps = Vec::new();
    for m_old in [-45.0, -17.0, 17.0, 45.0] {
        for delta in [-300.0, -120.0, -65.0, 0.0, 65.0, 120.0] {
            for l_old in [1.0e-300, 1.0e-150, 1.0e150, 1.0e300] {
                steps.push(StepCase::new(m_old, l_old, m_old + delta));
            }
        }
    }

    let mut streams = vec![
        StreamCase {
            m_init: -5.0,
            l_init: 1.0,
            scores: vec![10.0; 64],
        },
        StreamCase {
            m_init: 0.0,
            l_init: 1.0e-10,
            scores: vec![25.0; 128],
        },
    ];
    for (low, high, length, m_init, l_init) in [
        (-15.0, 15.0, 128_usize, 0.0, 1.0),
        (-40.0, 40.0, 64, -2.0, 1.0e5),
        (5.0, 6.0, 256, 4.0, 1.0e-18),
    ] {
        streams.push(StreamCase {
            m_init,
            l_init,
            scores: (0..length)
                .map(|position| if position % 2 == 0 { high } else { low })
                .collect(),
        });
    }
    let ramp: Vec<f64> = (0..128)
        .map(|position| -20.0 + 40.0 * f64::from(position) / 127.0)
        .collect();
    streams.push(StreamCase {
        m_init: -19.0,
        l_init: 1.0e12,
        scores: ramp.clone(),
    });
    streams.push(StreamCase {
        m_init: 19.0,
        l_init: 1.0e-12,
        scores: ramp.into_iter().rev().collect(),
    });
    let mut rng = SearchRng::new(seed);
    for (l_init, span) in [(1.0e-8, 12.0), (1.0e8, 25.0), (1.0, 3.0)] {
        streams.push(StreamCase {
            m_init: 0.0,
            l_init,
            scores: (0..256).map(|_| jitter(&mut rng, -span, span)).collect(),
        });
    }
    (
        StepCorpus::new("adversarial_holdout_steps", steps),
        StreamCorpus::new("adversarial_holdout_streams", streams),
    )
}

/// One opaque holdout slot combining unseen single steps and trajectories.
#[derive(Debug, Clone)]
struct CombinedHoldout {
    steps: StepCorpus,
    streams: StreamCorpus,
}

impl CombinedHoldout {
    fn resolve(&self, index: usize) -> (bool, usize) {
        if index < self.steps.len() {
            (true, index)
        } else {
            (false, index - self.steps.len())
        }
    }
}

impl ProblemCorpus for CombinedHoldout {
    fn role(&self) -> &'static str {
        "adversarial_holdout"
    }

    fn len(&self) -> usize {
        self.steps.len() + self.streams.len()
    }

    fn run_case(&self, candidate: &Candidate, index: usize) -> Result<CaseRun, CorpusFailure> {
        match self.resolve(index) {
            (true, inner) => self.steps.run_case(candidate, inner),
            (false, inner) => self.streams.run_case(candidate, inner),
        }
    }

    fn describe_case(&self, index: usize) -> Vec<(String, f64)> {
        match self.resolve(index) {
            (true, inner) => self.steps.describe_case(inner),
            (false, inner) => self.streams.describe_case(inner),
        }
    }

    fn oracle_output(&self, index: usize) -> Vec<f64> {
        match self.resolve(index) {
            (true, inner) => self.steps.oracle_output(inner),
            (false, inner) => self.streams.oracle_output(inner),
        }
    }

    fn digest_cases(&self, writer: &mut DigestWriter) {
        writer.tag(0x52);
        self.steps.digest_cases(writer);
        self.streams.digest_cases(writer);
    }
}

/// Assemble the default bounded E0 research problem.
#[must_use]
pub fn build_e0_problem(seed: u64) -> ResearchProblem {
    let (steps, streams) = build_holdout(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    ResearchProblem::new(
        PROBLEM_NAME.into(),
        PROBLEM_VERSION,
        seed,
        e0_grammar(),
        Tolerances::default(),
        SearchBudget::default(),
        Arc::new(build_discovery_corpus(seed ^ 0xD15C_0001)),
        Arc::new(build_probe_corpus()),
        Arc::new(build_oracle_corpus()),
        Arc::new(CombinedHoldout { steps, streams }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    fn variable(index: usize) -> Expr {
        Expr::Var(index)
    }

    /// This fixture is deliberately test-only. No production proposer can
    /// obtain it, and the generated candidate streams are tested separately.
    fn reference_candidate_fixture() -> Candidate {
        let maximum = || Expr::Max(Box::new(variable(0)), Box::new(variable(2)));
        Candidate::new(vec![
            maximum(),
            Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(variable(1)),
                    Box::new(Expr::Exp(Box::new(Expr::Sub(
                        Box::new(variable(0)),
                        Box::new(maximum()),
                    )))),
                )),
                Box::new(Expr::Exp(Box::new(Expr::Sub(
                    Box::new(variable(2)),
                    Box::new(maximum()),
                )))),
            ),
        ])
    }

    #[test]
    fn oracle_is_stable_across_declared_regimes() {
        for delta in [-300.0, -1.0, 0.0, 0.5, 120.0] {
            for (m_old, l_old) in [(-45.0, 1.0e-300), (17.0, 1.0e150), (0.0, 1.0)] {
                let output = oracle_step(m_old, l_old, m_old + delta);
                assert!(output.iter().all(|value| value.is_finite()));
                assert!(output[1] > 0.0);
            }
        }
    }

    #[test]
    fn corpus_partitions_are_distinct_and_stable() {
        let discovery = build_discovery_corpus(7);
        let probe = build_probe_corpus();
        let oracle = build_oracle_corpus();
        let (holdout_steps, holdout_streams) = build_holdout(7);
        assert_eq!(discovery.len(), 96);
        assert_eq!(probe.len(), 6);
        assert_eq!(oracle.len(), 686);
        assert_eq!(holdout_steps.len(), 96);
        assert_eq!(holdout_streams.len(), 10);
        assert_eq!(discovery.digest(), build_discovery_corpus(7).digest());
        assert_ne!(discovery.digest(), build_discovery_corpus(8).digest());
        assert_ne!(discovery.digest(), oracle.digest());
        assert!(discovery.cases().iter().all(|case| case.l_old > 0.0));
    }

    #[test]
    fn holdout_contains_repeated_and_alternating_maxima() {
        let (_, streams) = build_holdout(3);
        assert!(streams.cases().iter().any(|stream| {
            stream.scores.len() >= 2
                && stream
                    .scores
                    .iter()
                    .all(|score| score.to_bits() == stream.scores[0].to_bits())
        }));
        assert!(streams.cases().iter().any(|stream| {
            stream.scores.len() >= 4
                && stream.scores[0].to_bits() != stream.scores[1].to_bits()
                && stream.scores[0].to_bits() == stream.scores[2].to_bits()
                && stream.scores[1].to_bits() == stream.scores[3].to_bits()
        }));
    }

    #[test]
    fn known_valid_fixture_survives_every_case() {
        let candidate = reference_candidate_fixture();
        candidate.validate(&e0_grammar()).unwrap();
        let probe = build_probe_corpus();
        let oracle = build_oracle_corpus();
        let (steps, streams) = build_holdout(1);
        for corpus in [&probe as &dyn ProblemCorpus, &oracle, &steps, &streams] {
            for index in 0..corpus.len() {
                let run = corpus.run_case(&candidate, index).unwrap();
                assert_eq!(run.max_abs_error.to_bits(), 0.0_f64.to_bits());
            }
        }
    }
}
