//! Bounded Counterexample-Guided Inductive Synthesis infrastructure.
//!
//! This crate orchestrates the search layer without making a correctness or
//! novelty claim for any generated candidate. Static rejection remains owned
//! by [`ada_search`]. This layer adds deterministic fixture evaluation,
//! adversarial fixture generation, survivor re-evaluation, and retained
//! counterexample artifacts.

#![forbid(unsafe_code)]

use ada_search::{SearchCandidate, SearchEngine, SearchError, SearchFingerprint, SearchSpace};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Write as _};

/// Version of the CEGIS orchestration and counterexample artifact contract.
pub const CEGIS_VERSION: u16 = 1;
/// Maximum number of active fixtures in one bounded run.
pub const MAX_ACTIVE_FIXTURES: u64 = 1 << 12;
/// Maximum candidate-specific counterexample artifacts retained in one run.
pub const MAX_COUNTEREXAMPLE_ARTIFACTS: u64 = 1 << 16;
/// Maximum adversarial fixtures emitted for one candidate.
pub const MAX_ADVERSARIAL_OUTPUTS: u64 = 1 << 8;
/// Maximum fixture identifier length in bytes.
pub const MAX_FIXTURE_ID_BYTES: usize = 1 << 12;
/// Maximum fixture canonical text length in bytes.
pub const MAX_FIXTURE_TEXT_BYTES: usize = 1 << 20;
/// Maximum oracle failure reason length in bytes.
pub const MAX_REASON_BYTES: usize = 1 << 12;
/// Maximum serialized counterexample artifact length in bytes.
pub const MAX_ARTIFACT_TEXT_BYTES: usize = 16 << 20;

/// Fail-closed errors from CEGIS setup, execution, or artifact handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CegisError {
    /// A configuration or fixture field violated its structural contract.
    InvalidConfiguration(&'static str),
    /// A bounded value exceeded its explicit maximum.
    ExceedsLimit {
        /// Field whose value was rejected.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Inclusive maximum.
        maximum: u64,
    },
    /// A fixture was not a valid deterministic evidence input.
    InvalidFixture(String),
    /// The oracle itself could not produce a decision.
    OracleFailure(String),
    /// The adversarial generator itself could not produce a bounded result.
    GeneratorFailure(String),
    /// A persisted counterexample artifact was malformed.
    InvalidArtifact(String),
    /// The underlying deterministic search layer rejected the operation.
    Search(SearchError),
}

impl Display for CegisError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(field) => write!(formatter, "invalid CEGIS field: {field}"),
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::InvalidFixture(reason) => write!(formatter, "invalid CEGIS fixture: {reason}"),
            Self::OracleFailure(reason) => write!(formatter, "CEGIS oracle failure: {reason}"),
            Self::GeneratorFailure(reason) => {
                write!(formatter, "CEGIS adversarial generator failure: {reason}")
            }
            Self::InvalidArtifact(reason) => {
                write!(formatter, "invalid CEGIS counterexample artifact: {reason}")
            }
            Self::Search(error) => write!(formatter, "search layer failure: {error}"),
        }
    }
}

impl std::error::Error for CegisError {}

impl From<SearchError> for CegisError {
    fn from(error: SearchError) -> Self {
        Self::Search(error)
    }
}

/// Stable dual-lane identity for an active fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixtureFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl FixtureFingerprint {
    fn of_bytes(bytes: &[u8]) -> Self {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        const MIX_MULT: u64 = 0xff51_afd7_ed55_8ccd;
        let mut primary = FNV_OFFSET;
        let mut secondary = FNV_OFFSET;
        for &byte in bytes {
            primary ^= u64::from(byte);
            primary = primary.wrapping_mul(FNV_PRIME);
            secondary ^= u64::from(byte);
            secondary = secondary.rotate_left(27).wrapping_mul(MIX_MULT);
        }
        let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        primary = primary.wrapping_mul(FNV_PRIME) ^ length;
        secondary = secondary.rotate_left(31) ^ length;
        Self {
            primary,
            secondary,
            length,
        }
    }

    /// First fingerprint lane.
    #[must_use]
    pub const fn primary(self) -> u64 {
        self.primary
    }

    /// Second fingerprint lane.
    #[must_use]
    pub const fn secondary(self) -> u64 {
        self.secondary
    }

    /// Number of canonical identity bytes included in the fingerprint.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for FixtureFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

/// A deterministic input fixture with an explicit evidence identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Fixture<I> {
    id: String,
    canonical_text: String,
    fingerprint: FixtureFingerprint,
    input: I,
}

impl<I> Fixture<I> {
    /// Construct a bounded fixture. The input is never serialized implicitly;
    /// callers provide the canonical text that an artifact can retain.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/oversized identifier or canonical text,
    /// control separators in the identifier, or carriage returns.
    pub fn new(
        id: impl Into<String>,
        canonical_text: impl Into<String>,
        input: I,
    ) -> Result<Self, CegisError> {
        let id = id.into();
        let canonical_text = canonical_text.into();
        validate_fixture_id(&id)?;
        validate_fixture_text(&canonical_text)?;
        let identity = fixture_identity_key(&id, &canonical_text);
        Ok(Self {
            id,
            canonical_text,
            fingerprint: FixtureFingerprint::of_bytes(identity.as_bytes()),
            input,
        })
    }

    /// Stable fixture/sample identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Canonical representation retained in evidence artifacts.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }

    /// Stable identity of the fixture identifier and canonical input text.
    #[must_use]
    pub const fn fingerprint(&self) -> FixtureFingerprint {
        self.fingerprint
    }

    /// Borrow the typed fixture input used by the oracle.
    #[must_use]
    pub const fn input(&self) -> &I {
        &self.input
    }

    /// Consume the fixture and return its typed input.
    #[must_use]
    pub fn into_input(self) -> I {
        self.input
    }

    fn identity_key(&self) -> String {
        fixture_identity_key(&self.id, &self.canonical_text)
    }
}

/// Configuration bounding one deterministic CEGIS run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CegisConfig {
    /// Seed passed to the adversarial generator and recorded by the caller.
    pub seed: u64,
    /// Maximum unique fixtures retained in the active corpus.
    pub max_active_fixtures: u64,
    /// Maximum candidate-specific rejection artifacts retained.
    pub max_counterexample_artifacts: u64,
    /// Maximum adversarial fixtures emitted for one candidate.
    pub max_adversarial_outputs: u64,
}

impl CegisConfig {
    /// Construct and validate bounded CEGIS configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when any configured bound exceeds the crate limit.
    pub fn new(
        seed: u64,
        max_active_fixtures: u64,
        max_counterexample_artifacts: u64,
        max_adversarial_outputs: u64,
    ) -> Result<Self, CegisError> {
        if max_active_fixtures > MAX_ACTIVE_FIXTURES {
            return Err(CegisError::ExceedsLimit {
                field: "max_active_fixtures",
                value: max_active_fixtures,
                maximum: MAX_ACTIVE_FIXTURES,
            });
        }
        if max_counterexample_artifacts > MAX_COUNTEREXAMPLE_ARTIFACTS {
            return Err(CegisError::ExceedsLimit {
                field: "max_counterexample_artifacts",
                value: max_counterexample_artifacts,
                maximum: MAX_COUNTEREXAMPLE_ARTIFACTS,
            });
        }
        if max_adversarial_outputs > MAX_ADVERSARIAL_OUTPUTS {
            return Err(CegisError::ExceedsLimit {
                field: "max_adversarial_outputs",
                value: max_adversarial_outputs,
                maximum: MAX_ADVERSARIAL_OUTPUTS,
            });
        }
        Ok(Self {
            seed,
            max_active_fixtures,
            max_counterexample_artifacts,
            max_adversarial_outputs,
        })
    }
}

impl Default for CegisConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            max_active_fixtures: MAX_ACTIVE_FIXTURES,
            max_counterexample_artifacts: MAX_COUNTEREXAMPLE_ARTIFACTS,
            max_adversarial_outputs: MAX_ADVERSARIAL_OUTPUTS,
        }
    }
}

/// Counters from the CEGIS orchestration stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CegisStats {
    candidates_considered: u64,
    active_fixture_checks: u64,
    adversarial_fixture_generated: u64,
    adversarial_fixture_checks: u64,
    survivors_retested: u64,
    survivors_falsified: u64,
    oracle_falsified: u64,
    adversarial_falsified: u64,
    active_fixtures_added: u64,
    counterexamples_recorded: u64,
    survivors_admitted: u64,
}

impl CegisStats {
    /// Number of candidates received from the bounded search engine.
    #[must_use]
    pub const fn candidates_considered(self) -> u64 {
        self.candidates_considered
    }

    /// Number of checks against the active corpus.
    #[must_use]
    pub const fn active_fixture_checks(self) -> u64 {
        self.active_fixture_checks
    }

    /// Number of adversarial fixtures returned by generators.
    #[must_use]
    pub const fn adversarial_fixture_generated(self) -> u64 {
        self.adversarial_fixture_generated
    }

    /// Number of generated adversarial fixtures evaluated by the oracle.
    #[must_use]
    pub const fn adversarial_fixture_checks(self) -> u64 {
        self.adversarial_fixture_checks
    }

    /// Number of previously surviving candidates re-evaluated after a new
    /// counterexample entered the active corpus.
    #[must_use]
    pub const fn survivors_retested(self) -> u64 {
        self.survivors_retested
    }

    /// Number of previously surviving candidates killed during re-evaluation.
    #[must_use]
    pub const fn survivors_falsified(self) -> u64 {
        self.survivors_falsified
    }

    /// Number of candidates falsified by the initial active corpus.
    #[must_use]
    pub const fn oracle_falsified(self) -> u64 {
        self.oracle_falsified
    }

    /// Number of candidates falsified by adversarial fixtures or rechecks.
    #[must_use]
    pub const fn adversarial_falsified(self) -> u64 {
        self.adversarial_falsified
    }

    /// Number of new unique fixtures inserted into the active corpus.
    #[must_use]
    pub const fn active_fixtures_added(self) -> u64 {
        self.active_fixtures_added
    }

    /// Number of candidate-specific counterexample artifacts retained.
    #[must_use]
    pub const fn counterexamples_recorded(self) -> u64 {
        self.counterexamples_recorded
    }

    /// Number of candidates admitted to the provisional survivor archive.
    ///
    /// A later adversarial fixture may remove one of these candidates; the
    /// final archive is exposed by [`CegisResult::survivors`].
    #[must_use]
    pub const fn survivors_admitted(self) -> u64 {
        self.survivors_admitted
    }
}

/// Result of an oracle comparison for one candidate and one fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleOutcome {
    /// Candidate matched the oracle policy for this fixture.
    Pass,
    /// Candidate was falsified; the reason becomes evidence.
    Falsified {
        /// Short deterministic explanation of the mismatch.
        reason: String,
    },
}

/// Oracle interface used by the CEGIS loop.
pub trait DifferentialOracle<C, I> {
    /// Error type for an oracle implementation failure, distinct from a
    /// candidate-specific [`OracleOutcome::Falsified`] result.
    type Error: Display;

    /// Compare one candidate with one explicit fixture.
    ///
    /// An implementation should return `Falsified` for a candidate failure and
    /// reserve `Err` for a broken/unavailable oracle. The latter stops the run
    /// rather than silently rejecting a candidate.
    ///
    /// # Errors
    ///
    /// Returns an error when the oracle cannot complete its comparison.
    fn compare(
        &mut self,
        candidate: &C,
        fixture: &Fixture<I>,
    ) -> Result<OracleOutcome, Self::Error>;
}

/// Deterministic bounded adversarial fixture generator.
pub trait AdversarialGenerator<C, I> {
    /// Error type for a generator failure.
    type Error: Display;

    /// Generate possible counterexamples for one candidate.
    ///
    /// The active corpus is passed in stable insertion order. The runner sorts
    /// returned fixtures by their explicit identity before oracle checks and
    /// retains the first failing fixture as the deterministic minimal witness.
    /// The generator must use the supplied seed when it uses randomness.
    ///
    /// # Errors
    ///
    /// Returns an error when generation cannot complete.
    fn generate(
        &mut self,
        seed: u64,
        candidate: &C,
        active: &[Fixture<I>],
    ) -> Result<Vec<Fixture<I>>, Self::Error>;
}

/// Origin of a retained candidate-specific counterexample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CounterexampleSource {
    /// The candidate failed an initial deterministic corpus fixture.
    InitialCorpus,
    /// The candidate failed a newly generated adversarial fixture.
    Adversarial,
    /// A candidate that previously survived was killed by a new fixture.
    Revalidation,
}

impl Display for CounterexampleSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::InitialCorpus => "initial",
            Self::Adversarial => "adversarial",
            Self::Revalidation => "revalidation",
        };
        formatter.write_str(value)
    }
}

/// Persistable, type-erased counterexample evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterexampleArtifact {
    candidate_ordinal: u64,
    candidate_fingerprint: String,
    candidate_canonical_text: String,
    fixture_fingerprint: String,
    fixture_id: String,
    fixture_canonical_text: String,
    source: CounterexampleSource,
    reason: String,
}

impl CounterexampleArtifact {
    /// Candidate ordinal retained in the artifact.
    #[must_use]
    pub const fn candidate_ordinal(&self) -> u64 {
        self.candidate_ordinal
    }

    /// Candidate fingerprint as emitted by the search layer.
    #[must_use]
    pub fn candidate_fingerprint(&self) -> &str {
        &self.candidate_fingerprint
    }

    /// Candidate canonical text retained for inspection/reconstruction.
    #[must_use]
    pub fn candidate_canonical_text(&self) -> &str {
        &self.candidate_canonical_text
    }

    /// Fixture fingerprint retained in the artifact.
    #[must_use]
    pub fn fixture_fingerprint(&self) -> &str {
        &self.fixture_fingerprint
    }

    /// Fixture/sample identifier.
    #[must_use]
    pub fn fixture_id(&self) -> &str {
        &self.fixture_id
    }

    /// Fixture canonical text.
    #[must_use]
    pub fn fixture_canonical_text(&self) -> &str {
        &self.fixture_canonical_text
    }

    /// Counterexample origin.
    #[must_use]
    pub const fn source(&self) -> CounterexampleSource {
        self.source
    }

    /// Deterministic oracle failure reason.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Encode the artifact as strict line-oriented text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("ADA-CEGIS-COUNTEREXAMPLE-V{CEGIS_VERSION}\n");
        append_field(&mut text, "candidate_ordinal", self.candidate_ordinal);
        append_field(
            &mut text,
            "candidate_fingerprint",
            &self.candidate_fingerprint,
        );
        append_field(
            &mut text,
            "candidate_text",
            hex_encode(&self.candidate_canonical_text),
        );
        append_field(&mut text, "fixture_fingerprint", &self.fixture_fingerprint);
        append_field(&mut text, "fixture_id", hex_encode(&self.fixture_id));
        append_field(
            &mut text,
            "fixture_text",
            hex_encode(&self.fixture_canonical_text),
        );
        append_field(&mut text, "source", self.source);
        append_field(&mut text, "reason", hex_encode(&self.reason));
        text
    }

    /// Decode and validate a canonical counterexample artifact.
    ///
    /// The typed fixture input is intentionally not reconstructed here. The
    /// decoded canonical fixture fields are the handoff to a caller-owned,
    /// versioned fixture factory.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, unknown, duplicate, non-canonical, or
    /// oversized fields.
    pub fn from_canonical_text(text: &str) -> Result<Self, CegisError> {
        let fields = parse_artifact_fields(text)?;
        let candidate_ordinal = parse_u64_field(&fields, "candidate_ordinal")?;
        let candidate_fingerprint = field(&fields, "candidate_fingerprint")?.to_owned();
        validate_fingerprint_text(&candidate_fingerprint, "candidate_fingerprint")?;
        let candidate_canonical_text = decode_text_field(&fields, "candidate_text")?;
        validate_fixture_text(&candidate_canonical_text)?;
        let fixture_fingerprint = field(&fields, "fixture_fingerprint")?.to_owned();
        validate_fingerprint_text(&fixture_fingerprint, "fixture_fingerprint")?;
        let fixture_id = hex_decode(field(&fields, "fixture_id")?)?;
        validate_fixture_id(&fixture_id)?;
        let fixture_canonical_text = decode_text_field(&fields, "fixture_text")?;
        validate_fixture_text(&fixture_canonical_text)?;
        let source = parse_source(field(&fields, "source")?)?;
        let reason = hex_decode(field(&fields, "reason")?)?;
        validate_reason(&reason)?;
        Ok(Self {
            candidate_ordinal,
            candidate_fingerprint,
            candidate_canonical_text,
            fixture_fingerprint,
            fixture_id,
            fixture_canonical_text,
            source,
            reason,
        })
    }
}

/// Candidate-specific counterexample retaining its typed fixture input.
#[derive(Debug, Clone, PartialEq)]
pub struct Counterexample<I> {
    candidate_ordinal: u64,
    candidate_fingerprint: SearchFingerprint,
    candidate_canonical_text: String,
    fixture: Fixture<I>,
    source: CounterexampleSource,
    reason: String,
}

impl<I: Clone> Counterexample<I> {
    fn new(
        candidate: &SearchCandidate<impl Clone>,
        fixture: &Fixture<I>,
        source: CounterexampleSource,
        reason: String,
    ) -> Result<Self, CegisError> {
        validate_reason(&reason)?;
        Ok(Self {
            candidate_ordinal: candidate.ordinal(),
            candidate_fingerprint: candidate.fingerprint(),
            candidate_canonical_text: candidate.canonical_text().to_owned(),
            fixture: fixture.clone(),
            source,
            reason,
        })
    }

    /// Candidate ordinal that was falsified.
    #[must_use]
    pub const fn candidate_ordinal(&self) -> u64 {
        self.candidate_ordinal
    }

    /// Candidate fingerprint that was falsified.
    #[must_use]
    pub const fn candidate_fingerprint(&self) -> SearchFingerprint {
        self.candidate_fingerprint
    }

    /// Candidate canonical text retained for diagnosis.
    #[must_use]
    pub fn candidate_canonical_text(&self) -> &str {
        &self.candidate_canonical_text
    }

    /// Typed minimal fixture selected by deterministic corpus/generator order.
    #[must_use]
    pub const fn fixture(&self) -> &Fixture<I> {
        &self.fixture
    }

    /// Counterexample origin.
    #[must_use]
    pub const fn source(&self) -> CounterexampleSource {
        self.source
    }

    /// Failure reason retained verbatim in the artifact.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Convert to a type-erased persistence artifact.
    #[must_use]
    pub fn artifact(&self) -> CounterexampleArtifact {
        CounterexampleArtifact {
            candidate_ordinal: self.candidate_ordinal,
            candidate_fingerprint: self.candidate_fingerprint.to_string(),
            candidate_canonical_text: self.candidate_canonical_text.clone(),
            fixture_fingerprint: self.fixture.fingerprint().to_string(),
            fixture_id: self.fixture.id().to_owned(),
            fixture_canonical_text: self.fixture.canonical_text().to_owned(),
            source: self.source,
            reason: self.reason.clone(),
        }
    }

    /// Encode the counterexample as a canonical evidence artifact.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        self.artifact().to_canonical_text()
    }
}

/// A candidate rejected with its minimal retained counterexample.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectedCandidate<C, I> {
    candidate: SearchCandidate<C>,
    counterexample: Counterexample<I>,
}

impl<C, I> RejectedCandidate<C, I> {
    /// Reconstructible candidate that was rejected.
    #[must_use]
    pub const fn candidate(&self) -> &SearchCandidate<C> {
        &self.candidate
    }

    /// Minimal counterexample and reason for rejection.
    #[must_use]
    pub const fn counterexample(&self) -> &Counterexample<I> {
        &self.counterexample
    }
}

/// Completed CEGIS result with survivors, rejections, and retained artifacts.
#[derive(Clone)]
pub struct CegisResult<C, I> {
    survivors: Vec<SearchCandidate<C>>,
    rejected: Vec<RejectedCandidate<C, I>>,
    active_fixtures: Vec<Fixture<I>>,
    counterexamples: Vec<Counterexample<I>>,
    stats: CegisStats,
    search_stats: ada_search::SearchStats,
}

impl<C, I> CegisResult<C, I> {
    /// Candidates surviving all active-corpus and adversarial checks.
    #[must_use]
    pub fn survivors(&self) -> &[SearchCandidate<C>] {
        &self.survivors
    }

    /// Rejected candidates, each paired with a retained counterexample.
    #[must_use]
    pub fn rejected(&self) -> &[RejectedCandidate<C, I>] {
        &self.rejected
    }

    /// Initial and subsequently inserted active fixtures in insertion order.
    #[must_use]
    pub fn active_fixtures(&self) -> &[Fixture<I>] {
        &self.active_fixtures
    }

    /// All candidate-specific counterexamples retained by the run.
    #[must_use]
    pub fn counterexamples(&self) -> &[Counterexample<I>] {
        &self.counterexamples
    }

    /// CEGIS-specific counters.
    #[must_use]
    pub const fn stats(&self) -> CegisStats {
        self.stats
    }

    /// Underlying search-generation counters, including later evidence marks.
    #[must_use]
    pub const fn search_stats(&self) -> ada_search::SearchStats {
        self.search_stats
    }
}

/// Bounded CEGIS runner over any [`ada_search::SearchSpace`].
pub struct CegisEngine<S, I, O, G>
where
    S: SearchSpace,
    I: Clone,
    O: DifferentialOracle<S::Candidate, I>,
    G: AdversarialGenerator<S::Candidate, I>,
{
    search: SearchEngine<S>,
    oracle: O,
    generator: G,
    config: CegisConfig,
    active_fixtures: Vec<Fixture<I>>,
    active_keys: BTreeSet<String>,
    survivors: Vec<SearchCandidate<S::Candidate>>,
    rejected: Vec<RejectedCandidate<S::Candidate, I>>,
    counterexamples: Vec<Counterexample<I>>,
    stats: CegisStats,
}

impl<S, I, O, G> CegisEngine<S, I, O, G>
where
    S: SearchSpace,
    I: Clone,
    O: DifferentialOracle<S::Candidate, I>,
    G: AdversarialGenerator<S::Candidate, I>,
{
    /// Create a CEGIS runner with an explicit deterministic seed corpus.
    ///
    /// Duplicate initial fixtures are removed by their explicit identity. No
    /// oracle call occurs during construction.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration or the initial corpus exceeds a
    /// declared bound.
    pub fn new(
        search: SearchEngine<S>,
        oracle: O,
        generator: G,
        config: CegisConfig,
        initial_fixtures: Vec<Fixture<I>>,
    ) -> Result<Self, CegisError> {
        let config = CegisConfig::new(
            config.seed,
            config.max_active_fixtures,
            config.max_counterexample_artifacts,
            config.max_adversarial_outputs,
        )?;
        let mut active_fixtures = Vec::new();
        let mut active_keys = BTreeSet::new();
        for fixture in initial_fixtures {
            let key = fixture.identity_key();
            if active_keys.insert(key) {
                active_fixtures.push(fixture);
            }
        }
        active_fixtures.sort_by_key(Fixture::identity_key);
        let active_count = u64::try_from(active_fixtures.len()).unwrap_or(u64::MAX);
        if active_count > config.max_active_fixtures {
            return Err(CegisError::ExceedsLimit {
                field: "initial_fixtures",
                value: active_count,
                maximum: config.max_active_fixtures,
            });
        }
        Ok(Self {
            search,
            oracle,
            generator,
            config,
            active_fixtures,
            active_keys,
            survivors: Vec::new(),
            rejected: Vec::new(),
            counterexamples: Vec::new(),
            stats: CegisStats::default(),
        })
    }

    /// Borrow the underlying deterministic search state.
    #[must_use]
    pub const fn search(&self) -> &SearchEngine<S> {
        &self.search
    }

    /// Borrow the current active corpus.
    #[must_use]
    pub fn active_fixtures(&self) -> &[Fixture<I>] {
        &self.active_fixtures
    }

    /// Current CEGIS counters before completion.
    #[must_use]
    pub const fn stats(&self) -> CegisStats {
        self.stats
    }

    /// Run the bounded search to exhaustion and return all retained evidence.
    ///
    /// Every candidate is checked against the active corpus. A candidate that
    /// passes may cause a bounded adversarial search; a newly failing fixture
    /// is inserted and all previous survivors are re-evaluated against it.
    /// Oracle/generator infrastructure failures stop the run.
    ///
    /// # Errors
    ///
    /// Returns an error for search failure, oracle/generator failure, an
    /// invalid oracle reason, or an exceeded active/artifact bound.
    pub fn run_to_end(mut self) -> Result<CegisResult<S::Candidate, I>, CegisError> {
        while let Some(candidate) = self.search.next_candidate()? {
            self.stats.candidates_considered += 1;
            if let Some((fixture, reason)) = self.check_active_candidate(&candidate)? {
                self.reject_candidate(
                    candidate,
                    &fixture,
                    CounterexampleSource::InitialCorpus,
                    reason,
                )?;
                continue;
            }

            let active_snapshot = self.active_fixtures.clone();
            let generated = self
                .generator
                .generate(self.config.seed, candidate.candidate(), &active_snapshot)
                .map_err(|error| CegisError::GeneratorFailure(error.to_string()))?;
            let generated_count = u64::try_from(generated.len()).unwrap_or(u64::MAX);
            if generated_count > self.config.max_adversarial_outputs {
                return Err(CegisError::ExceedsLimit {
                    field: "adversarial_outputs",
                    value: generated_count,
                    maximum: self.config.max_adversarial_outputs,
                });
            }
            self.stats.adversarial_fixture_generated += generated_count;
            if let Some((fixture, reason)) =
                self.check_adversarial_candidate(&candidate, generated)?
            {
                self.insert_active_fixture(fixture.clone())?;
                self.reject_candidate(
                    candidate,
                    &fixture,
                    CounterexampleSource::Adversarial,
                    reason,
                )?;
                self.revalidate_survivors(&fixture)?;
            } else {
                self.stats.survivors_admitted += 1;
                self.survivors.push(candidate);
            }
        }

        Ok(CegisResult {
            survivors: self.survivors,
            rejected: self.rejected,
            active_fixtures: self.active_fixtures,
            counterexamples: self.counterexamples,
            stats: self.stats,
            search_stats: self.search.stats(),
        })
    }

    fn check_active_candidate(
        &mut self,
        candidate: &SearchCandidate<S::Candidate>,
    ) -> Result<Option<(Fixture<I>, String)>, CegisError> {
        for index in 0..self.active_fixtures.len() {
            let fixture = self.active_fixtures[index].clone();
            self.stats.active_fixture_checks += 1;
            if let Some(reason) = self.compare(candidate.candidate(), &fixture)? {
                return Ok(Some((fixture, reason)));
            }
        }
        Ok(None)
    }

    fn check_adversarial_candidate(
        &mut self,
        candidate: &SearchCandidate<S::Candidate>,
        mut generated: Vec<Fixture<I>>,
    ) -> Result<Option<(Fixture<I>, String)>, CegisError> {
        generated.sort_by_key(Fixture::identity_key);
        let mut generated_keys = BTreeSet::new();
        for fixture in generated {
            let key = fixture.identity_key();
            if !generated_keys.insert(key.clone()) || self.active_keys.contains(&key) {
                continue;
            }
            self.stats.adversarial_fixture_checks += 1;
            if let Some(reason) = self.compare(candidate.candidate(), &fixture)? {
                return Ok(Some((fixture, reason)));
            }
        }
        Ok(None)
    }

    fn compare(
        &mut self,
        candidate: &S::Candidate,
        fixture: &Fixture<I>,
    ) -> Result<Option<String>, CegisError> {
        let outcome = self
            .oracle
            .compare(candidate, fixture)
            .map_err(|error| CegisError::OracleFailure(error.to_string()))?;
        match outcome {
            OracleOutcome::Pass => Ok(None),
            OracleOutcome::Falsified { reason } => {
                validate_reason(&reason)?;
                Ok(Some(reason))
            }
        }
    }

    fn insert_active_fixture(&mut self, fixture: Fixture<I>) -> Result<(), CegisError> {
        let key = fixture.identity_key();
        if self.active_keys.contains(&key) {
            return Ok(());
        }
        let active_count = u64::try_from(self.active_fixtures.len()).unwrap_or(u64::MAX);
        if active_count >= self.config.max_active_fixtures {
            return Err(CegisError::ExceedsLimit {
                field: "active_fixtures",
                value: active_count + 1,
                maximum: self.config.max_active_fixtures,
            });
        }
        self.active_keys.insert(key);
        self.active_fixtures.push(fixture);
        self.stats.active_fixtures_added += 1;
        Ok(())
    }

    fn record_counterexample(
        &mut self,
        counterexample: Counterexample<I>,
    ) -> Result<(), CegisError> {
        let count = u64::try_from(self.counterexamples.len()).unwrap_or(u64::MAX);
        if count >= self.config.max_counterexample_artifacts {
            return Err(CegisError::ExceedsLimit {
                field: "counterexample_artifacts",
                value: count + 1,
                maximum: self.config.max_counterexample_artifacts,
            });
        }
        self.counterexamples.push(counterexample);
        self.stats.counterexamples_recorded += 1;
        Ok(())
    }

    fn reject_candidate(
        &mut self,
        candidate: SearchCandidate<S::Candidate>,
        fixture: &Fixture<I>,
        source: CounterexampleSource,
        reason: String,
    ) -> Result<(), CegisError> {
        let counterexample = Counterexample::new(&candidate, fixture, source, reason)?;
        self.record_counterexample(counterexample.clone())?;
        self.rejected.push(RejectedCandidate {
            candidate,
            counterexample,
        });
        match source {
            CounterexampleSource::InitialCorpus => {
                self.stats.oracle_falsified += 1;
                self.search.record_oracle_falsified();
            }
            CounterexampleSource::Adversarial | CounterexampleSource::Revalidation => {
                self.stats.adversarial_falsified += 1;
                self.search.record_adversarial_falsified();
            }
        }
        Ok(())
    }

    fn revalidate_survivors(&mut self, fixture: &Fixture<I>) -> Result<(), CegisError> {
        let previous = std::mem::take(&mut self.survivors);
        let mut retained = Vec::with_capacity(previous.len());
        for candidate in previous {
            self.stats.survivors_retested += 1;
            if let Some(reason) = self.compare(candidate.candidate(), fixture)? {
                self.stats.survivors_falsified += 1;
                self.reject_candidate(
                    candidate,
                    fixture,
                    CounterexampleSource::Revalidation,
                    reason,
                )?;
            } else {
                retained.push(candidate);
            }
        }
        self.survivors = retained;
        Ok(())
    }
}

const ARTIFACT_FIELDS: &[&str] = &[
    "candidate_ordinal",
    "candidate_fingerprint",
    "candidate_text",
    "fixture_fingerprint",
    "fixture_id",
    "fixture_text",
    "source",
    "reason",
];

fn validate_fixture_id(value: &str) -> Result<(), CegisError> {
    if value.is_empty() || value.len() > MAX_FIXTURE_ID_BYTES {
        return Err(CegisError::InvalidFixture(
            "fixture identifier is empty or oversized".into(),
        ));
    }
    if value.bytes().any(|byte| matches!(byte, b'\r' | b'\n')) {
        return Err(CegisError::InvalidFixture(
            "fixture identifier contains a line separator".into(),
        ));
    }
    Ok(())
}

fn validate_fixture_text(value: &str) -> Result<(), CegisError> {
    if value.is_empty() || value.len() > MAX_FIXTURE_TEXT_BYTES {
        return Err(CegisError::InvalidFixture(
            "fixture canonical text is empty or oversized".into(),
        ));
    }
    if value.contains('\r') {
        return Err(CegisError::InvalidFixture(
            "fixture canonical text contains CR".into(),
        ));
    }
    Ok(())
}

fn validate_reason(value: &str) -> Result<(), CegisError> {
    if value.is_empty() || value.len() > MAX_REASON_BYTES || value.contains('\r') {
        return Err(CegisError::InvalidArtifact(
            "oracle reason is empty, oversized, or contains CR".into(),
        ));
    }
    Ok(())
}

fn fixture_identity_key(id: &str, canonical_text: &str) -> String {
    let mut key = String::with_capacity(id.len() + 1 + canonical_text.len());
    key.push_str(id);
    key.push('\n');
    key.push_str(canonical_text);
    key
}

fn append_field(text: &mut String, key: &str, value: impl Display) {
    let _ = writeln!(text, "{key}={value}");
}

fn parse_artifact_fields(text: &str) -> Result<BTreeMap<&str, &str>, CegisError> {
    if text.len() > MAX_ARTIFACT_TEXT_BYTES || text.contains('\r') {
        return Err(CegisError::InvalidArtifact(
            "artifact exceeds its limit or contains CR".into(),
        ));
    }
    if !text.ends_with('\n') {
        return Err(CegisError::InvalidArtifact(
            "artifact must end with a newline".into(),
        ));
    }
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err(CegisError::InvalidArtifact("missing version header".into()));
    };
    let expected_header = format!("ADA-CEGIS-COUNTEREXAMPLE-V{CEGIS_VERSION}");
    if header != expected_header {
        return Err(CegisError::InvalidArtifact(
            "unsupported or non-canonical version header".into(),
        ));
    }
    let mut fields = BTreeMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Err(CegisError::InvalidArtifact("field is missing '='".into()));
        };
        if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
            return Err(CegisError::InvalidArtifact(
                "empty, duplicate, or ambiguous field".into(),
            ));
        }
    }
    if fields.len() != ARTIFACT_FIELDS.len()
        || ARTIFACT_FIELDS.iter().any(|key| !fields.contains_key(key))
    {
        return Err(CegisError::InvalidArtifact(
            "field set is incomplete or has unknown keys".into(),
        ));
    }
    Ok(fields)
}

fn field<'a>(fields: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str, CegisError> {
    fields
        .get(key)
        .copied()
        .ok_or_else(|| CegisError::InvalidArtifact(format!("missing field {key}")))
}

fn parse_u64_field(fields: &BTreeMap<&str, &str>, key: &str) -> Result<u64, CegisError> {
    field(fields, key)?.parse::<u64>().map_err(|_| {
        CegisError::InvalidArtifact(format!("{key} is not an unsigned decimal integer"))
    })
}

fn decode_text_field(fields: &BTreeMap<&str, &str>, key: &str) -> Result<String, CegisError> {
    hex_decode(field(fields, key)?)
}

fn validate_fingerprint_text(value: &str, field_name: &'static str) -> Result<(), CegisError> {
    let mut parts = value.split('-');
    let valid = parts.by_ref().count() == 3
        && value.split('-').all(|part| {
            part.len() == 16
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        });
    if valid {
        Ok(())
    } else {
        Err(CegisError::InvalidArtifact(format!(
            "{field_name} is not a canonical fingerprint"
        )))
    }
}

fn parse_source(value: &str) -> Result<CounterexampleSource, CegisError> {
    match value {
        "initial" => Ok(CounterexampleSource::InitialCorpus),
        "adversarial" => Ok(CounterexampleSource::Adversarial),
        "revalidation" => Ok(CounterexampleSource::Revalidation),
        _ => Err(CegisError::InvalidArtifact(
            "unknown counterexample source".into(),
        )),
    }
}

fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(value: &str) -> Result<String, CegisError> {
    if value.is_empty() || value.len() % 2 != 0 {
        return Err(CegisError::InvalidArtifact(
            "hex field is empty or has odd length".into(),
        ));
    }
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = hex_value(pair[0]).ok_or_else(|| {
            CegisError::InvalidArtifact("hex field contains non-canonical digits".into())
        })?;
        let low = hex_value(pair[1]).ok_or_else(|| {
            CegisError::InvalidArtifact("hex field contains non-canonical digits".into())
        })?;
        decoded.push((high << 4) | low);
    }
    String::from_utf8(decoded)
        .map_err(|_| CegisError::InvalidArtifact("hex field is not UTF-8".into()))
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::SemanticFamily;
    use ada_search::{SearchBudget, SemanticSearchConfig, SemanticSearchSpace};
    use ada_semantic::{ReferenceInput, ReferenceInputSpec, SemanticProgram};
    use std::convert::Infallible;

    #[derive(Debug, Clone, Copy)]
    struct ToyOracle;

    impl DifferentialOracle<i32, i32> for ToyOracle {
        type Error = Infallible;

        fn compare(
            &mut self,
            candidate: &i32,
            fixture: &Fixture<i32>,
        ) -> Result<OracleOutcome, Self::Error> {
            if *candidate <= *fixture.input() {
                Ok(OracleOutcome::Falsified {
                    reason: format!("candidate {candidate} <= fixture {}", fixture.input()),
                })
            } else {
                Ok(OracleOutcome::Pass)
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct ToyGenerator;

    impl AdversarialGenerator<i32, i32> for ToyGenerator {
        type Error = Infallible;

        fn generate(
            &mut self,
            _seed: u64,
            candidate: &i32,
            _active: &[Fixture<i32>],
        ) -> Result<Vec<Fixture<i32>>, Self::Error> {
            if *candidate == 1 {
                Ok(vec![Fixture::new("adv-1", "value=1", 1).unwrap()])
            } else {
                Ok(Vec::new())
            }
        }
    }

    #[derive(Debug, Clone)]
    struct ToySpace {
        values: Vec<i32>,
    }

    impl SearchSpace for ToySpace {
        type Candidate = i32;

        fn cardinality(&self) -> u64 {
            u64::try_from(self.values.len()).unwrap()
        }

        fn fingerprint(&self) -> SearchFingerprint {
            let text = self
                .values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            SearchFingerprint::of_canonical_text(&text)
        }

        fn candidate_at(&self, ordinal: u64) -> Result<Self::Candidate, SearchError> {
            self.values
                .get(usize::try_from(ordinal).unwrap())
                .copied()
                .ok_or(SearchError::InvalidConfiguration("toy ordinal"))
        }

        fn candidate_canonical_text(&self, candidate: &Self::Candidate) -> String {
            format!("toy-candidate={candidate}")
        }

        fn candidate_cost(&self, _candidate: &Self::Candidate) -> u32 {
            1
        }
    }

    fn toy_search(values: Vec<i32>) -> SearchEngine<ToySpace> {
        SearchEngine::new(ToySpace { values }, SearchBudget::new(16, 16, 1).unwrap()).unwrap()
    }

    fn fixture(id: &str, value: i32) -> Fixture<i32> {
        Fixture::new(id, format!("value={value}"), value).unwrap()
    }

    #[test]
    fn cegis_inserts_counterexamples_and_retests_survivors() {
        let runner = CegisEngine::new(
            toy_search(vec![0, 1, 2]),
            ToyOracle,
            ToyGenerator,
            CegisConfig::new(7, 8, 8, 4).unwrap(),
            vec![fixture("seed-low", -1)],
        )
        .unwrap();
        let result = runner.run_to_end().unwrap();
        assert_eq!(result.survivors().len(), 1);
        assert_eq!(*result.survivors()[0].candidate(), 2);
        assert_eq!(result.rejected().len(), 2);
        assert_eq!(result.counterexamples().len(), 2);
        assert_eq!(result.active_fixtures().len(), 2);
        assert_eq!(result.stats().survivors_retested(), 1);
        assert_eq!(result.stats().survivors_falsified(), 1);
        assert_eq!(result.stats().active_fixtures_added(), 1);
        assert_eq!(result.stats().oracle_falsified(), 0);
        assert_eq!(result.stats().adversarial_falsified(), 2);
        assert_eq!(result.search_stats().adversarial_falsified(), 2);
    }

    #[test]
    fn identical_seed_corpus_and_search_reproduce_cegis_artifacts() {
        let run = || {
            CegisEngine::new(
                toy_search(vec![0, 1, 2]),
                ToyOracle,
                ToyGenerator,
                CegisConfig::new(7, 8, 8, 4).unwrap(),
                vec![fixture("seed-low", -1)],
            )
            .unwrap()
            .run_to_end()
            .unwrap()
        };
        let left = run();
        let right = run();
        let left_candidates = left
            .survivors()
            .iter()
            .chain(left.rejected().iter().map(RejectedCandidate::candidate))
            .map(SearchCandidate::canonical_text)
            .collect::<Vec<_>>();
        let right_candidates = right
            .survivors()
            .iter()
            .chain(right.rejected().iter().map(RejectedCandidate::candidate))
            .map(SearchCandidate::canonical_text)
            .collect::<Vec<_>>();
        let left_artifacts = left
            .counterexamples()
            .iter()
            .map(Counterexample::to_canonical_text)
            .collect::<Vec<_>>();
        let right_artifacts = right
            .counterexamples()
            .iter()
            .map(Counterexample::to_canonical_text)
            .collect::<Vec<_>>();
        assert_eq!(left_candidates, right_candidates);
        assert_eq!(left_artifacts, right_artifacts);
        assert_eq!(left.stats(), right.stats());
        assert_eq!(left.search_stats(), right.search_stats());
    }

    #[test]
    fn counterexample_artifact_round_trips_and_rejects_malformed_text() {
        let runner = CegisEngine::new(
            toy_search(vec![0]),
            ToyOracle,
            ToyGenerator,
            CegisConfig::default(),
            vec![fixture("seed", 0)],
        )
        .unwrap();
        let result = runner.run_to_end().unwrap();
        let artifact = result.counterexamples()[0].artifact();
        let text = artifact.to_canonical_text();
        assert_eq!(
            CounterexampleArtifact::from_canonical_text(&text).unwrap(),
            artifact
        );
        assert!(
            CounterexampleArtifact::from_canonical_text(&(text.clone() + "unknown=x\n")).is_err()
        );
        assert!(CounterexampleArtifact::from_canonical_text(&text[..text.len() - 1]).is_err());
        assert!(
            CounterexampleArtifact::from_canonical_text(
                &text.replace("ADA-CEGIS-COUNTEREXAMPLE-V1", "ADA-CEGIS-COUNTEREXAMPLE-V2")
            )
            .is_err()
        );
    }

    #[test]
    fn fixture_and_configuration_contracts_fail_closed() {
        assert!(Fixture::<u8>::new("", "input", 1).is_err());
        assert!(Fixture::<u8>::new("id\n", "input", 1).is_err());
        assert!(CegisConfig::new(0, MAX_ACTIVE_FIXTURES + 1, 1, 1).is_err());
        assert!(
            CegisConfig::new(
                0,
                MAX_ACTIVE_FIXTURES,
                MAX_COUNTEREXAMPLE_ARTIFACTS,
                MAX_ADVERSARIAL_OUTPUTS + 1
            )
            .is_err()
        );

        let runner = CegisEngine::new(
            toy_search(vec![0]),
            ToyOracle,
            ToyGenerator,
            CegisConfig::new(0, 8, 0, 4).unwrap(),
            vec![fixture("seed", 0)],
        )
        .unwrap();
        assert!(runner.run_to_end().is_err());
    }

    struct NoopGenerator;

    impl<C, I> AdversarialGenerator<C, I> for NoopGenerator {
        type Error = Infallible;

        fn generate(
            &mut self,
            _seed: u64,
            _candidate: &C,
            _active: &[Fixture<I>],
        ) -> Result<Vec<Fixture<I>>, Self::Error> {
            Ok(Vec::new())
        }
    }

    struct SemanticOracle {
        expected: Vec<f64>,
    }

    impl DifferentialOracle<SemanticProgram, ReferenceInput> for SemanticOracle {
        type Error = String;

        fn compare(
            &mut self,
            candidate: &SemanticProgram,
            fixture: &Fixture<ReferenceInput>,
        ) -> Result<OracleOutcome, Self::Error> {
            let actual = match candidate.evaluate(fixture.input()) {
                Ok(value) => value,
                Err(error) => {
                    return Ok(OracleOutcome::Falsified {
                        reason: format!("candidate evaluation rejected: {error}"),
                    });
                }
            };
            let max_error = self
                .expected
                .iter()
                .zip(actual.output())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            if max_error <= 1.0e-12 {
                Ok(OracleOutcome::Pass)
            } else {
                Ok(OracleOutcome::Falsified {
                    reason: format!("max output error {max_error:.17e}"),
                })
            }
        }
    }

    #[test]
    fn semantic_candidates_are_differentially_checked_without_novelty_claims() {
        let input = ReferenceInput::new(ReferenceInputSpec {
            query_count: 2,
            key_count: 2,
            q_dimension: 1,
            value_dimension: 1,
            queries: vec![1.0, 0.5],
            keys: vec![1.0, -1.0],
            values: vec![2.0, 4.0],
            external_mask: None,
        })
        .unwrap();
        let fixture = Fixture::new("semantic-seed", "reference-input-v1", input).unwrap();
        let positive_one = 1.0_f64.exp();
        let negative_one = (-1.0_f64).exp();
        let positive_half = 0.5_f64.exp();
        let negative_half = (-0.5_f64).exp();
        let expected = vec![
            (2.0 * positive_one + 4.0 * negative_one) / (positive_one + negative_one),
            (2.0 * positive_half + 4.0 * negative_half) / (positive_half + negative_half),
        ];
        let runner = CegisEngine::new(
            SearchEngine::new(
                SemanticSearchSpace::new(SemanticSearchConfig::default()).unwrap(),
                SearchBudget::new(16, 16, ada_search::MAX_PROGRAM_COST).unwrap(),
            )
            .unwrap(),
            SemanticOracle { expected },
            NoopGenerator,
            CegisConfig::default(),
            vec![fixture],
        )
        .unwrap();
        let result = runner.run_to_end().unwrap();
        assert!(result.stats().candidates_considered() > 0);
        assert!(result.stats().oracle_falsified() > 0);
        assert!(!result.counterexamples().is_empty());
        assert!(
            result.counterexamples().iter().all(
                |counterexample| counterexample.source() == CounterexampleSource::InitialCorpus
            )
        );
        assert!(
            result.survivors().iter().all(|candidate| candidate
                .candidate()
                .descriptor()
                .id()
                .family()
                == SemanticFamily::StandardSoftmax)
        );
    }
}
