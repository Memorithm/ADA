//! Persistable, fail-closed archive for a completed CEGIS falsification run.
//!
//! A [`CegisRunArchive`] records the bounded evidence produced by one finished
//! [`crate::CegisResult`]: configuration, active fixtures, retained rejection
//! counterexamples, provisional survivors, and counters.
//!
//! # Non-claims
//!
//! - Survival in this archive is **not** qualification, adoption, novelty, or
//!   FLAT readiness. Survivors remain subject to later independent oracle,
//!   adversarial, cost, and (when relevant) hardware gates.
//! - The archive is evidence that a **bounded** falsification pass completed
//!   under the recorded config and fixtures. It is not a proof of correctness
//!   over an unbounded input domain.
//! - Mid-run CEGIS checkpoint/resume is intentionally out of scope for this
//!   format version; use [`ada_search::SearchCheckpoint`] for search-prefix
//!   resume and treat full CEGIS engine checkpointing as a follow-up.

use crate::{
    CEGIS_VERSION, CegisConfig, CegisError, CegisResult, CegisStats, CounterexampleArtifact,
    FixtureFingerprint, MAX_ACTIVE_FIXTURES, MAX_ARTIFACT_TEXT_BYTES, MAX_COUNTEREXAMPLE_ARTIFACTS,
    append_field, field, fixture_identity_key, hex_decode, hex_encode, parse_u64_field,
    validate_fingerprint_text, validate_fixture_id, validate_fixture_text,
};
use std::collections::BTreeMap;

/// Format version of the completed-run archive codec.
pub const CEGIS_RUN_ARCHIVE_VERSION: u16 = 1;
/// Maximum serialized run-archive length in bytes.
pub const MAX_RUN_ARCHIVE_TEXT_BYTES: usize = 64 << 20;

const BASE_FIELDS: &[&str] = &[
    "archive_format_version",
    "cegis_version",
    "space_primary",
    "space_secondary",
    "space_length",
    "config_seed",
    "config_max_active_fixtures",
    "config_max_counterexample_artifacts",
    "config_max_adversarial_outputs",
    "active_fixture_count",
    "rejection_count",
    "survivor_count",
    "cegis_candidates_considered",
    "cegis_active_fixture_checks",
    "cegis_adversarial_fixture_generated",
    "cegis_adversarial_fixture_checks",
    "cegis_survivors_retested",
    "cegis_survivors_falsified",
    "cegis_oracle_falsified",
    "cegis_adversarial_falsified",
    "cegis_active_fixtures_added",
    "cegis_counterexamples_recorded",
    "cegis_survivors_admitted",
    "search_generated",
    "search_statically_rejected",
    "search_duplicate",
    "search_oracle_falsified",
    "search_adversarial_falsified",
    "search_cost_dominated",
    "search_surviving",
];

/// Type-erased fixture identity retained in a run archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchivedFixtureIdentity {
    id: String,
    fingerprint: String,
    canonical_text: String,
}

impl ArchivedFixtureIdentity {
    /// Fixture/sample identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Canonical fixture fingerprint text.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Canonical fixture text retained for caller-owned reconstruction.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }
}

/// Provisional survivor identity retained in a run archive.
///
/// Presence here means only that the candidate survived the recorded bounded
/// falsification pass. It does **not** mark the candidate adopted, qualified,
/// novel, or FLAT-ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchivedSurvivor {
    ordinal: u64,
    fingerprint: String,
    canonical_text: String,
}

impl ArchivedSurvivor {
    /// Search ordinal of the survivor.
    #[must_use]
    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Candidate fingerprint text.
    #[must_use]
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Candidate canonical text.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }
}

/// Deterministic, versioned archive of one completed CEGIS falsification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CegisRunArchive {
    archive_format_version: u16,
    cegis_version: u16,
    space_primary: u64,
    space_secondary: u64,
    space_length: u64,
    config: CegisConfig,
    active_fixtures: Vec<ArchivedFixtureIdentity>,
    rejections: Vec<CounterexampleArtifact>,
    survivors: Vec<ArchivedSurvivor>,
    cegis_stats: CegisStats,
    search_generated: u64,
    search_statically_rejected: u64,
    search_duplicate: u64,
    search_oracle_falsified: u64,
    search_adversarial_falsified: u64,
    search_cost_dominated: u64,
    search_surviving: u64,
}

impl CegisRunArchive {
    /// Archive codec format version.
    #[must_use]
    pub const fn archive_format_version(&self) -> u16 {
        self.archive_format_version
    }

    /// CEGIS contract version recorded in the archive.
    #[must_use]
    pub const fn cegis_version(&self) -> u16 {
        self.cegis_version
    }

    /// Search-space fingerprint primary lane.
    #[must_use]
    pub const fn space_primary(&self) -> u64 {
        self.space_primary
    }

    /// Search-space fingerprint secondary lane.
    #[must_use]
    pub const fn space_secondary(&self) -> u64 {
        self.space_secondary
    }

    /// Search-space fingerprint length lane.
    #[must_use]
    pub const fn space_length(&self) -> u64 {
        self.space_length
    }

    /// CEGIS configuration bounds and seed recorded for the run.
    #[must_use]
    pub const fn config(&self) -> CegisConfig {
        self.config
    }

    /// Active fixtures in archive order (insertion order from the run).
    #[must_use]
    pub fn active_fixtures(&self) -> &[ArchivedFixtureIdentity] {
        &self.active_fixtures
    }

    /// Rejected candidates' minimal counterexample artifacts, in run order.
    #[must_use]
    pub fn rejections(&self) -> &[CounterexampleArtifact] {
        &self.rejections
    }

    /// Provisional survivors of the bounded falsification pass.
    #[must_use]
    pub fn survivors(&self) -> &[ArchivedSurvivor] {
        &self.survivors
    }

    /// CEGIS orchestration counters.
    #[must_use]
    pub const fn cegis_stats(&self) -> CegisStats {
        self.cegis_stats
    }

    /// Search-layer `generated` counter.
    #[must_use]
    pub const fn search_generated(&self) -> u64 {
        self.search_generated
    }

    /// Search-layer `statically_rejected` counter.
    #[must_use]
    pub const fn search_statically_rejected(&self) -> u64 {
        self.search_statically_rejected
    }

    /// Search-layer `duplicate` counter.
    #[must_use]
    pub const fn search_duplicate(&self) -> u64 {
        self.search_duplicate
    }

    /// Search-layer `oracle_falsified` counter.
    #[must_use]
    pub const fn search_oracle_falsified(&self) -> u64 {
        self.search_oracle_falsified
    }

    /// Search-layer `adversarial_falsified` counter.
    #[must_use]
    pub const fn search_adversarial_falsified(&self) -> u64 {
        self.search_adversarial_falsified
    }

    /// Search-layer `cost_dominated` counter.
    #[must_use]
    pub const fn search_cost_dominated(&self) -> u64 {
        self.search_cost_dominated
    }

    /// Search-layer `surviving` counter.
    #[must_use]
    pub const fn search_surviving(&self) -> u64 {
        self.search_surviving
    }

    /// Build a fail-closed archive from a completed [`CegisResult`].
    ///
    /// # Errors
    ///
    /// Returns an error when rejection/counterexample inventories disagree, a
    /// retained counterexample would be dropped, a bound is exceeded, or a
    /// fixture/survivor field violates the structural contract.
    pub fn from_cegis_result<C, I: Clone>(result: &CegisResult<C, I>) -> Result<Self, CegisError> {
        let rejections = collect_rejections(result)?;
        let active_fixtures = collect_active_fixtures(result)?;
        let survivors = collect_survivors(result)?;
        let space = result.space_fingerprint();
        let search = result.search_stats();
        let archive = Self {
            archive_format_version: CEGIS_RUN_ARCHIVE_VERSION,
            cegis_version: CEGIS_VERSION,
            space_primary: space.primary(),
            space_secondary: space.secondary(),
            space_length: space.length(),
            config: result.config(),
            active_fixtures,
            rejections,
            survivors,
            cegis_stats: result.stats(),
            search_generated: search.generated(),
            search_statically_rejected: search.statically_rejected(),
            search_duplicate: search.duplicate(),
            search_oracle_falsified: search.oracle_falsified(),
            search_adversarial_falsified: search.adversarial_falsified(),
            search_cost_dominated: search.cost_dominated(),
            search_surviving: search.surviving(),
        };
        archive.validate_invariants()?;
        Ok(archive)
    }

    /// Encode the archive as strict line-oriented canonical text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("ADA-CEGIS-RUN-ARCHIVE-V{CEGIS_RUN_ARCHIVE_VERSION}\n");
        self.append_identity_and_config_fields(&mut text);
        self.append_counter_fields(&mut text);
        self.append_inventory_fields(&mut text);
        text
    }

    /// Decode and validate a canonical run archive.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, unknown, duplicate, truncated,
    /// oversized, or inconsistent fields.
    pub fn from_canonical_text(text: &str) -> Result<Self, CegisError> {
        let fields = parse_archive_fields(text)?;
        let (archive_format_version, cegis_version) = parse_versions(&fields)?;
        let (space_primary, space_secondary, space_length) = parse_space(&fields)?;
        let config = parse_config(&fields)?;
        let (active_count, rejection_count, survivor_count) = parse_counts(&fields, &config)?;
        expect_field_cardinality(&fields, active_count, rejection_count, survivor_count)?;
        let search = parse_search_stats(&fields)?;
        let archive = Self {
            archive_format_version,
            cegis_version,
            space_primary,
            space_secondary,
            space_length,
            config,
            active_fixtures: parse_active_fixtures(&fields, active_count)?,
            rejections: parse_rejections(&fields, rejection_count)?,
            survivors: parse_survivors(&fields, survivor_count)?,
            cegis_stats: parse_cegis_stats(&fields)?,
            search_generated: search[0],
            search_statically_rejected: search[1],
            search_duplicate: search[2],
            search_oracle_falsified: search[3],
            search_adversarial_falsified: search[4],
            search_cost_dominated: search[5],
            search_surviving: search[6],
        };
        archive.validate_invariants()?;
        Ok(archive)
    }

    fn append_identity_and_config_fields(&self, text: &mut String) {
        append_field(text, "archive_format_version", self.archive_format_version);
        append_field(text, "cegis_version", self.cegis_version);
        append_field(
            text,
            "space_primary",
            format_args!("{:016x}", self.space_primary),
        );
        append_field(
            text,
            "space_secondary",
            format_args!("{:016x}", self.space_secondary),
        );
        append_field(text, "space_length", self.space_length);
        append_field(text, "config_seed", self.config.seed);
        append_field(
            text,
            "config_max_active_fixtures",
            self.config.max_active_fixtures,
        );
        append_field(
            text,
            "config_max_counterexample_artifacts",
            self.config.max_counterexample_artifacts,
        );
        append_field(
            text,
            "config_max_adversarial_outputs",
            self.config.max_adversarial_outputs,
        );
        append_field(text, "active_fixture_count", self.active_fixtures.len());
        append_field(text, "rejection_count", self.rejections.len());
        append_field(text, "survivor_count", self.survivors.len());
    }

    fn append_counter_fields(&self, text: &mut String) {
        append_field(
            text,
            "cegis_candidates_considered",
            self.cegis_stats.candidates_considered(),
        );
        append_field(
            text,
            "cegis_active_fixture_checks",
            self.cegis_stats.active_fixture_checks(),
        );
        append_field(
            text,
            "cegis_adversarial_fixture_generated",
            self.cegis_stats.adversarial_fixture_generated(),
        );
        append_field(
            text,
            "cegis_adversarial_fixture_checks",
            self.cegis_stats.adversarial_fixture_checks(),
        );
        append_field(
            text,
            "cegis_survivors_retested",
            self.cegis_stats.survivors_retested(),
        );
        append_field(
            text,
            "cegis_survivors_falsified",
            self.cegis_stats.survivors_falsified(),
        );
        append_field(
            text,
            "cegis_oracle_falsified",
            self.cegis_stats.oracle_falsified(),
        );
        append_field(
            text,
            "cegis_adversarial_falsified",
            self.cegis_stats.adversarial_falsified(),
        );
        append_field(
            text,
            "cegis_active_fixtures_added",
            self.cegis_stats.active_fixtures_added(),
        );
        append_field(
            text,
            "cegis_counterexamples_recorded",
            self.cegis_stats.counterexamples_recorded(),
        );
        append_field(
            text,
            "cegis_survivors_admitted",
            self.cegis_stats.survivors_admitted(),
        );
        append_field(text, "search_generated", self.search_generated);
        append_field(
            text,
            "search_statically_rejected",
            self.search_statically_rejected,
        );
        append_field(text, "search_duplicate", self.search_duplicate);
        append_field(
            text,
            "search_oracle_falsified",
            self.search_oracle_falsified,
        );
        append_field(
            text,
            "search_adversarial_falsified",
            self.search_adversarial_falsified,
        );
        append_field(text, "search_cost_dominated", self.search_cost_dominated);
        append_field(text, "search_surviving", self.search_surviving);
    }

    fn append_inventory_fields(&self, text: &mut String) {
        for (index, fixture) in self.active_fixtures.iter().enumerate() {
            append_field(text, &format!("active_{index}_id"), hex_encode(&fixture.id));
            append_field(
                text,
                &format!("active_{index}_fingerprint"),
                &fixture.fingerprint,
            );
            append_field(
                text,
                &format!("active_{index}_text"),
                hex_encode(&fixture.canonical_text),
            );
        }
        for (index, rejection) in self.rejections.iter().enumerate() {
            append_field(
                text,
                &format!("rejection_{index}"),
                hex_encode(&rejection.to_canonical_text()),
            );
        }
        for (index, survivor) in self.survivors.iter().enumerate() {
            append_field(text, &format!("survivor_{index}_ordinal"), survivor.ordinal);
            append_field(
                text,
                &format!("survivor_{index}_fingerprint"),
                &survivor.fingerprint,
            );
            append_field(
                text,
                &format!("survivor_{index}_text"),
                hex_encode(&survivor.canonical_text),
            );
        }
    }

    fn validate_invariants(&self) -> Result<(), CegisError> {
        if self.archive_format_version != CEGIS_RUN_ARCHIVE_VERSION {
            return Err(CegisError::InvalidArtifact(
                "archive_format_version mismatch".into(),
            ));
        }
        if self.cegis_version != CEGIS_VERSION {
            return Err(CegisError::InvalidArtifact("cegis_version mismatch".into()));
        }
        if u64::try_from(self.rejections.len()).unwrap_or(u64::MAX)
            != self.cegis_stats.counterexamples_recorded()
        {
            return Err(CegisError::InvalidArtifact(
                "rejection count does not match cegis_counterexamples_recorded".into(),
            ));
        }
        // Survivors are provisional only; the archive must never imply adoption.
        // There is intentionally no adopted/qualified field in this format.
        Ok(())
    }
}

fn collect_rejections<C, I: Clone>(
    result: &CegisResult<C, I>,
) -> Result<Vec<CounterexampleArtifact>, CegisError> {
    if result.rejected().len() != result.counterexamples().len() {
        return Err(CegisError::InvalidArtifact(
            "rejected inventory and counterexample inventory disagree".into(),
        ));
    }
    let rejection_count = u64::try_from(result.rejected().len()).unwrap_or(u64::MAX);
    if rejection_count > result.config().max_counterexample_artifacts
        || rejection_count > MAX_COUNTEREXAMPLE_ARTIFACTS
    {
        return Err(CegisError::ExceedsLimit {
            field: "rejection_count",
            value: rejection_count,
            maximum: result
                .config()
                .max_counterexample_artifacts
                .min(MAX_COUNTEREXAMPLE_ARTIFACTS),
        });
    }
    let mut rejections = Vec::with_capacity(result.rejected().len());
    for (index, rejected) in result.rejected().iter().enumerate() {
        let artifact = rejected.counterexample().artifact();
        let retained = result.counterexamples()[index].artifact();
        if artifact != retained {
            return Err(CegisError::InvalidArtifact(
                "rejection archive would silently drop or reorder a retained counterexample".into(),
            ));
        }
        rejections.push(artifact);
    }
    if rejections.len() != result.counterexamples().len() {
        return Err(CegisError::InvalidArtifact(
            "rejection archive would silently drop a retained counterexample".into(),
        ));
    }
    Ok(rejections)
}

fn collect_active_fixtures<C, I>(
    result: &CegisResult<C, I>,
) -> Result<Vec<ArchivedFixtureIdentity>, CegisError> {
    let active_count = u64::try_from(result.active_fixtures().len()).unwrap_or(u64::MAX);
    if active_count > result.config().max_active_fixtures || active_count > MAX_ACTIVE_FIXTURES {
        return Err(CegisError::ExceedsLimit {
            field: "active_fixture_count",
            value: active_count,
            maximum: result.config().max_active_fixtures.min(MAX_ACTIVE_FIXTURES),
        });
    }
    let mut active_fixtures = Vec::with_capacity(result.active_fixtures().len());
    for fixture in result.active_fixtures() {
        validate_fixture_id(fixture.id())?;
        validate_fixture_text(fixture.canonical_text())?;
        active_fixtures.push(ArchivedFixtureIdentity {
            id: fixture.id().to_owned(),
            fingerprint: fixture.fingerprint().to_string(),
            canonical_text: fixture.canonical_text().to_owned(),
        });
    }
    Ok(active_fixtures)
}

fn collect_survivors<C, I>(
    result: &CegisResult<C, I>,
) -> Result<Vec<ArchivedSurvivor>, CegisError> {
    let survivor_count = u64::try_from(result.survivors().len()).unwrap_or(u64::MAX);
    if survivor_count > MAX_COUNTEREXAMPLE_ARTIFACTS {
        return Err(CegisError::ExceedsLimit {
            field: "survivor_count",
            value: survivor_count,
            maximum: MAX_COUNTEREXAMPLE_ARTIFACTS,
        });
    }
    let mut survivors = Vec::with_capacity(result.survivors().len());
    for survivor in result.survivors() {
        validate_fixture_text(survivor.canonical_text())?;
        survivors.push(ArchivedSurvivor {
            ordinal: survivor.ordinal(),
            fingerprint: survivor.fingerprint().to_string(),
            canonical_text: survivor.canonical_text().to_owned(),
        });
    }
    Ok(survivors)
}

fn parse_archive_fields(text: &str) -> Result<BTreeMap<&str, &str>, CegisError> {
    if text.len() > MAX_RUN_ARCHIVE_TEXT_BYTES || text.contains('\r') {
        return Err(CegisError::InvalidArtifact(
            "run archive exceeds its limit or contains CR".into(),
        ));
    }
    if !text.ends_with('\n') {
        return Err(CegisError::InvalidArtifact(
            "run archive must end with a newline".into(),
        ));
    }
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err(CegisError::InvalidArtifact("missing version header".into()));
    };
    let expected_header = format!("ADA-CEGIS-RUN-ARCHIVE-V{CEGIS_RUN_ARCHIVE_VERSION}");
    if header != expected_header {
        return Err(CegisError::InvalidArtifact(
            "unsupported or non-canonical run-archive version header".into(),
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
    if BASE_FIELDS.iter().any(|key| !fields.contains_key(key)) {
        return Err(CegisError::InvalidArtifact(
            "base field set is incomplete".into(),
        ));
    }
    Ok(fields)
}

fn parse_versions(fields: &BTreeMap<&str, &str>) -> Result<(u16, u16), CegisError> {
    let archive_format_version = parse_u16_field(fields, "archive_format_version")?;
    if archive_format_version != CEGIS_RUN_ARCHIVE_VERSION {
        return Err(CegisError::InvalidArtifact(format!(
            "unsupported archive_format_version {archive_format_version}"
        )));
    }
    let cegis_version = parse_u16_field(fields, "cegis_version")?;
    if cegis_version != CEGIS_VERSION {
        return Err(CegisError::InvalidArtifact(format!(
            "unsupported cegis_version {cegis_version}"
        )));
    }
    Ok((archive_format_version, cegis_version))
}

fn parse_space(fields: &BTreeMap<&str, &str>) -> Result<(u64, u64, u64), CegisError> {
    Ok((
        parse_hex_u64_field(fields, "space_primary")?,
        parse_hex_u64_field(fields, "space_secondary")?,
        parse_u64_field(fields, "space_length")?,
    ))
}

fn parse_config(fields: &BTreeMap<&str, &str>) -> Result<CegisConfig, CegisError> {
    CegisConfig::new(
        parse_u64_field(fields, "config_seed")?,
        parse_u64_field(fields, "config_max_active_fixtures")?,
        parse_u64_field(fields, "config_max_counterexample_artifacts")?,
        parse_u64_field(fields, "config_max_adversarial_outputs")?,
    )
}

fn parse_counts(
    fields: &BTreeMap<&str, &str>,
    config: &CegisConfig,
) -> Result<(u64, u64, u64), CegisError> {
    let active_fixture_count = parse_u64_field(fields, "active_fixture_count")?;
    let rejection_count = parse_u64_field(fields, "rejection_count")?;
    let survivor_count = parse_u64_field(fields, "survivor_count")?;
    if active_fixture_count > config.max_active_fixtures
        || active_fixture_count > MAX_ACTIVE_FIXTURES
    {
        return Err(CegisError::ExceedsLimit {
            field: "active_fixture_count",
            value: active_fixture_count,
            maximum: config.max_active_fixtures.min(MAX_ACTIVE_FIXTURES),
        });
    }
    if rejection_count > config.max_counterexample_artifacts
        || rejection_count > MAX_COUNTEREXAMPLE_ARTIFACTS
    {
        return Err(CegisError::ExceedsLimit {
            field: "rejection_count",
            value: rejection_count,
            maximum: config
                .max_counterexample_artifacts
                .min(MAX_COUNTEREXAMPLE_ARTIFACTS),
        });
    }
    if survivor_count > MAX_COUNTEREXAMPLE_ARTIFACTS {
        return Err(CegisError::ExceedsLimit {
            field: "survivor_count",
            value: survivor_count,
            maximum: MAX_COUNTEREXAMPLE_ARTIFACTS,
        });
    }
    Ok((active_fixture_count, rejection_count, survivor_count))
}

fn expect_field_cardinality(
    fields: &BTreeMap<&str, &str>,
    active_fixture_count: u64,
    rejection_count: u64,
    survivor_count: u64,
) -> Result<(), CegisError> {
    let expected_fields = BASE_FIELDS.len()
        + usize::try_from(active_fixture_count)
            .unwrap_or(usize::MAX)
            .saturating_mul(3)
        + usize::try_from(rejection_count).unwrap_or(usize::MAX)
        + usize::try_from(survivor_count)
            .unwrap_or(usize::MAX)
            .saturating_mul(3);
    if fields.len() != expected_fields {
        return Err(CegisError::InvalidArtifact(
            "field set is incomplete or has unknown keys".into(),
        ));
    }
    Ok(())
}

fn parse_active_fixtures(
    fields: &BTreeMap<&str, &str>,
    active_fixture_count: u64,
) -> Result<Vec<ArchivedFixtureIdentity>, CegisError> {
    let mut active_fixtures =
        Vec::with_capacity(usize::try_from(active_fixture_count).unwrap_or(0));
    for index in 0..active_fixture_count {
        let id = hex_decode(field(fields, &format!("active_{index}_id"))?)?;
        validate_fixture_id(&id)?;
        let fingerprint = field(fields, &format!("active_{index}_fingerprint"))?.to_owned();
        validate_fingerprint_text(&fingerprint, "active fixture fingerprint")?;
        let canonical_text = hex_decode(field(fields, &format!("active_{index}_text"))?)?;
        validate_fixture_text(&canonical_text)?;
        let expected =
            FixtureFingerprint::of_bytes(fixture_identity_key(&id, &canonical_text).as_bytes())
                .to_string();
        if fingerprint != expected {
            return Err(CegisError::InvalidArtifact(
                "active fixture fingerprint does not match identity".into(),
            ));
        }
        active_fixtures.push(ArchivedFixtureIdentity {
            id,
            fingerprint,
            canonical_text,
        });
    }
    Ok(active_fixtures)
}

fn parse_rejections(
    fields: &BTreeMap<&str, &str>,
    rejection_count: u64,
) -> Result<Vec<CounterexampleArtifact>, CegisError> {
    let mut rejections = Vec::with_capacity(usize::try_from(rejection_count).unwrap_or(0));
    for index in 0..rejection_count {
        let artifact_text = hex_decode(field(fields, &format!("rejection_{index}"))?)?;
        if artifact_text.len() > MAX_ARTIFACT_TEXT_BYTES {
            return Err(CegisError::InvalidArtifact(
                "nested rejection artifact is oversized".into(),
            ));
        }
        rejections.push(CounterexampleArtifact::from_canonical_text(&artifact_text)?);
    }
    Ok(rejections)
}

fn parse_survivors(
    fields: &BTreeMap<&str, &str>,
    survivor_count: u64,
) -> Result<Vec<ArchivedSurvivor>, CegisError> {
    let mut survivors = Vec::with_capacity(usize::try_from(survivor_count).unwrap_or(0));
    for index in 0..survivor_count {
        let ordinal = field(fields, &format!("survivor_{index}_ordinal"))?
            .parse::<u64>()
            .map_err(|_| {
                CegisError::InvalidArtifact("survivor ordinal is not an unsigned decimal".into())
            })?;
        let fingerprint = field(fields, &format!("survivor_{index}_fingerprint"))?.to_owned();
        validate_fingerprint_text(&fingerprint, "survivor fingerprint")?;
        let canonical_text = hex_decode(field(fields, &format!("survivor_{index}_text"))?)?;
        validate_fixture_text(&canonical_text)?;
        survivors.push(ArchivedSurvivor {
            ordinal,
            fingerprint,
            canonical_text,
        });
    }
    Ok(survivors)
}

fn parse_cegis_stats(fields: &BTreeMap<&str, &str>) -> Result<CegisStats, CegisError> {
    Ok(CegisStats::from_parts([
        parse_u64_field(fields, "cegis_candidates_considered")?,
        parse_u64_field(fields, "cegis_active_fixture_checks")?,
        parse_u64_field(fields, "cegis_adversarial_fixture_generated")?,
        parse_u64_field(fields, "cegis_adversarial_fixture_checks")?,
        parse_u64_field(fields, "cegis_survivors_retested")?,
        parse_u64_field(fields, "cegis_survivors_falsified")?,
        parse_u64_field(fields, "cegis_oracle_falsified")?,
        parse_u64_field(fields, "cegis_adversarial_falsified")?,
        parse_u64_field(fields, "cegis_active_fixtures_added")?,
        parse_u64_field(fields, "cegis_counterexamples_recorded")?,
        parse_u64_field(fields, "cegis_survivors_admitted")?,
    ]))
}

fn parse_search_stats(fields: &BTreeMap<&str, &str>) -> Result<[u64; 7], CegisError> {
    Ok([
        parse_u64_field(fields, "search_generated")?,
        parse_u64_field(fields, "search_statically_rejected")?,
        parse_u64_field(fields, "search_duplicate")?,
        parse_u64_field(fields, "search_oracle_falsified")?,
        parse_u64_field(fields, "search_adversarial_falsified")?,
        parse_u64_field(fields, "search_cost_dominated")?,
        parse_u64_field(fields, "search_surviving")?,
    ])
}

fn parse_u16_field(fields: &BTreeMap<&str, &str>, key: &str) -> Result<u16, CegisError> {
    field(fields, key)?.parse::<u16>().map_err(|_| {
        CegisError::InvalidArtifact(format!("{key} is not an unsigned 16-bit decimal integer"))
    })
}

fn parse_hex_u64_field(fields: &BTreeMap<&str, &str>, key: &str) -> Result<u64, CegisError> {
    let value = field(fields, key)?;
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(CegisError::InvalidArtifact(format!(
            "{key} is not a canonical lowercase hex u64"
        )));
    }
    u64::from_str_radix(value, 16).map_err(|_| {
        CegisError::InvalidArtifact(format!("{key} is not a canonical lowercase hex u64"))
    })
}
