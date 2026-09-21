//! Mid-run CEGIS engine checkpoint/resume.
//!
//! A [`CegisCheckpoint`] captures enough deterministic engine state to resume a
//! bounded CEGIS run at the same logical point: configuration seed/bounds,
//! nested [`ada_search::SearchCheckpoint`] enumerator cursor, active fixture
//! corpus identities and payloads, provisional survivors, rejected-so-far
//! counterexamples, and CEGIS counters.
//!
//! # Non-claims
//!
//! - A checkpoint is **not** qualification, adoption, novelty, or FLAT
//!   readiness. Resume only continues a bounded falsification pass.
//! - Resume does **not** create adoption or novelty claims for provisional
//!   survivors retained in the checkpoint.
//! - This format is distinct from a completed-run archive: it is intended for
//!   exact mid-run continuation, not as a graduation/evidence handoff record.

use crate::{
    CEGIS_VERSION, CegisConfig, CegisError, CegisStats, CounterexampleArtifact, Fixture,
    MAX_ACTIVE_FIXTURES, MAX_COUNTEREXAMPLE_ARTIFACTS, RejectedCandidate, append_field, field,
    hex_decode, hex_encode, parse_u64_field, validate_fingerprint_text, validate_fixture_id,
    validate_fixture_text,
};
use ada_search::{SearchCandidate, SearchCheckpoint, SearchFingerprint};
use std::collections::BTreeMap;

/// Format version of the mid-run CEGIS checkpoint codec.
pub const CEGIS_CHECKPOINT_VERSION: u16 = 1;
/// Maximum serialized mid-run checkpoint length in bytes.
pub const MAX_CEGIS_CHECKPOINT_TEXT_BYTES: usize = 64 << 20;

const BASE_FIELDS: &[&str] = &[
    "checkpoint_format_version",
    "cegis_version",
    "space_primary",
    "space_secondary",
    "space_length",
    "config_seed",
    "config_max_active_fixtures",
    "config_max_counterexample_artifacts",
    "config_max_adversarial_outputs",
    "config_fingerprint",
    "search_checkpoint",
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
];

/// Type-erased active fixture identity + payload retained for resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointFixture {
    id: String,
    fingerprint: String,
    canonical_text: String,
}

impl CheckpointFixture {
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

    /// Canonical fixture text payload required for deterministic resume.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }
}

/// Provisional survivor identity retained mid-run.
///
/// Presence here means only that the candidate has not yet been falsified by
/// the recorded corpus. It is **not** adoption, qualification, or novelty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSurvivor {
    ordinal: u64,
    fingerprint: String,
    canonical_text: String,
}

impl CheckpointSurvivor {
    /// Search ordinal of the provisional survivor.
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

/// Deterministic, versioned mid-run CEGIS engine checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CegisCheckpoint {
    checkpoint_format_version: u16,
    cegis_version: u16,
    space_primary: u64,
    space_secondary: u64,
    space_length: u64,
    config: CegisConfig,
    config_fingerprint: String,
    search_checkpoint_text: String,
    active_fixtures: Vec<CheckpointFixture>,
    rejections: Vec<CounterexampleArtifact>,
    survivors: Vec<CheckpointSurvivor>,
    cegis_stats: CegisStats,
}

impl CegisCheckpoint {
    /// Checkpoint codec format version.
    #[must_use]
    pub const fn checkpoint_format_version(&self) -> u16 {
        self.checkpoint_format_version
    }

    /// CEGIS contract version recorded in the checkpoint.
    #[must_use]
    pub const fn cegis_version(&self) -> u16 {
        self.cegis_version
    }

    /// Search-space fingerprint bound into this checkpoint.
    #[must_use]
    pub const fn space_fingerprint(&self) -> SearchFingerprint {
        SearchFingerprint::from_parts(self.space_primary, self.space_secondary, self.space_length)
    }

    /// Raw space fingerprint primary lane.
    #[must_use]
    pub const fn space_primary(&self) -> u64 {
        self.space_primary
    }

    /// Raw space fingerprint secondary lane.
    #[must_use]
    pub const fn space_secondary(&self) -> u64 {
        self.space_secondary
    }

    /// Raw space fingerprint length lane.
    #[must_use]
    pub const fn space_length(&self) -> u64 {
        self.space_length
    }

    /// Configuration seed and bounds bound into this checkpoint.
    #[must_use]
    pub const fn config(&self) -> CegisConfig {
        self.config
    }

    /// Canonical fingerprint of the recorded configuration fields.
    #[must_use]
    pub fn config_fingerprint(&self) -> &str {
        &self.config_fingerprint
    }

    /// Nested search-layer checkpoint canonical text.
    #[must_use]
    pub fn search_checkpoint_text(&self) -> &str {
        &self.search_checkpoint_text
    }

    /// Active fixtures in insertion order at snapshot time.
    #[must_use]
    pub fn active_fixtures(&self) -> &[CheckpointFixture] {
        &self.active_fixtures
    }

    /// Rejected-so-far counterexample artifacts in run order.
    #[must_use]
    pub fn rejections(&self) -> &[CounterexampleArtifact] {
        &self.rejections
    }

    /// Provisional survivors at snapshot time.
    #[must_use]
    pub fn survivors(&self) -> &[CheckpointSurvivor] {
        &self.survivors
    }

    /// CEGIS orchestration counters at snapshot time.
    #[must_use]
    pub const fn cegis_stats(&self) -> CegisStats {
        self.cegis_stats
    }

    /// Build a fail-closed checkpoint from live engine inventories.
    ///
    /// # Errors
    ///
    /// Returns an error when inventories disagree with counters, a bound is
    /// exceeded, or a retained field violates its structural contract.
    pub(crate) fn from_parts(
        space: SearchFingerprint,
        config: CegisConfig,
        search_checkpoint: &SearchCheckpoint,
        active_fixtures: &[Fixture<impl Clone>],
        survivors: &[SearchCandidate<impl Clone>],
        rejected: &[RejectedCandidate<impl Clone, impl Clone>],
        stats: CegisStats,
    ) -> Result<Self, CegisError> {
        let config = CegisConfig::new(
            config.seed,
            config.max_active_fixtures,
            config.max_counterexample_artifacts,
            config.max_adversarial_outputs,
        )?;
        let active = collect_active_fixtures(active_fixtures, config.max_active_fixtures)?;
        let rejections = collect_rejections(rejected, config.max_counterexample_artifacts)?;
        let survivors = collect_survivors(survivors)?;
        if u64::try_from(rejections.len()).unwrap_or(u64::MAX) != stats.counterexamples_recorded() {
            return Err(CegisError::InvalidCheckpoint(
                "rejection inventory disagrees with counterexamples_recorded".into(),
            ));
        }
        if u64::try_from(survivors.len()).unwrap_or(u64::MAX) > stats.survivors_admitted() {
            return Err(CegisError::InvalidCheckpoint(
                "survivor inventory exceeds survivors_admitted".into(),
            ));
        }
        if search_checkpoint.space_fingerprint() != space {
            return Err(CegisError::InvalidCheckpoint(
                "nested search checkpoint space fingerprint mismatch".into(),
            ));
        }
        let checkpoint = Self {
            checkpoint_format_version: CEGIS_CHECKPOINT_VERSION,
            cegis_version: CEGIS_VERSION,
            space_primary: space.primary(),
            space_secondary: space.secondary(),
            space_length: space.length(),
            config_fingerprint: config_fingerprint_text(&config),
            config,
            search_checkpoint_text: search_checkpoint.to_canonical_text(),
            active_fixtures: active,
            rejections,
            survivors,
            cegis_stats: stats,
        };
        checkpoint.validate_invariants()?;
        Ok(checkpoint)
    }

    /// Encode as strict line-oriented canonical text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("ADA-CEGIS-CHECKPOINT-V{CEGIS_CHECKPOINT_VERSION}\n");
        self.append_identity_and_config_fields(&mut text);
        self.append_counter_fields(&mut text);
        self.append_inventory_fields(&mut text);
        text
    }

    fn append_identity_and_config_fields(&self, text: &mut String) {
        append_field(
            text,
            "checkpoint_format_version",
            self.checkpoint_format_version,
        );
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
        append_field(text, "config_fingerprint", &self.config_fingerprint);
        append_field(
            text,
            "search_checkpoint",
            hex_encode(&self.search_checkpoint_text),
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

    /// Decode and validate a canonical mid-run checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for version mismatch, malformed/truncated/unknown
    /// fields, fingerprint mismatch, or bound violations.
    pub fn from_canonical_text(text: &str) -> Result<Self, CegisError> {
        let fields = parse_checkpoint_fields(text)?;
        let checkpoint_format_version = parse_u16_field(&fields, "checkpoint_format_version")?;
        let cegis_version = parse_u16_field(&fields, "cegis_version")?;
        if checkpoint_format_version != CEGIS_CHECKPOINT_VERSION {
            return Err(CegisError::InvalidCheckpoint(format!(
                "unsupported checkpoint_format_version {checkpoint_format_version}"
            )));
        }
        if cegis_version != CEGIS_VERSION {
            return Err(CegisError::InvalidCheckpoint(format!(
                "cegis_version mismatch: got {cegis_version}, expected {CEGIS_VERSION}"
            )));
        }
        let space_primary = parse_hex_u64_field(&fields, "space_primary")?;
        let space_secondary = parse_hex_u64_field(&fields, "space_secondary")?;
        let space_length = parse_u64_field(&fields, "space_length")?;
        let config = CegisConfig::new(
            parse_u64_field(&fields, "config_seed")?,
            parse_u64_field(&fields, "config_max_active_fixtures")?,
            parse_u64_field(&fields, "config_max_counterexample_artifacts")?,
            parse_u64_field(&fields, "config_max_adversarial_outputs")?,
        )?;
        let config_fingerprint = field(&fields, "config_fingerprint")?.to_owned();
        let expected_fp = config_fingerprint_text(&config);
        if config_fingerprint != expected_fp {
            return Err(CegisError::InvalidCheckpoint(
                "config_fingerprint does not match recorded bounds".into(),
            ));
        }
        let search_checkpoint_text = hex_decode(field(&fields, "search_checkpoint")?)?;
        // Fail closed early on nested decode / version problems.
        let _ =
            SearchCheckpoint::from_canonical_text(&search_checkpoint_text).map_err(|error| {
                CegisError::InvalidCheckpoint(format!("nested search checkpoint: {error}"))
            })?;
        let active_count = parse_u64_field(&fields, "active_fixture_count")?;
        let rejection_count = parse_u64_field(&fields, "rejection_count")?;
        let survivor_count = parse_u64_field(&fields, "survivor_count")?;
        if active_count > config.max_active_fixtures || active_count > MAX_ACTIVE_FIXTURES {
            return Err(CegisError::ExceedsLimit {
                field: "active_fixture_count",
                value: active_count,
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
        expect_field_cardinality(&fields, active_count, rejection_count, survivor_count)?;
        let checkpoint = Self {
            checkpoint_format_version,
            cegis_version,
            space_primary,
            space_secondary,
            space_length,
            config,
            config_fingerprint,
            search_checkpoint_text,
            active_fixtures: parse_active_fixtures(&fields, active_count)?,
            rejections: parse_rejections(&fields, rejection_count)?,
            survivors: parse_survivors(&fields, survivor_count)?,
            cegis_stats: CegisStats::from_parts([
                parse_u64_field(&fields, "cegis_candidates_considered")?,
                parse_u64_field(&fields, "cegis_active_fixture_checks")?,
                parse_u64_field(&fields, "cegis_adversarial_fixture_generated")?,
                parse_u64_field(&fields, "cegis_adversarial_fixture_checks")?,
                parse_u64_field(&fields, "cegis_survivors_retested")?,
                parse_u64_field(&fields, "cegis_survivors_falsified")?,
                parse_u64_field(&fields, "cegis_oracle_falsified")?,
                parse_u64_field(&fields, "cegis_adversarial_falsified")?,
                parse_u64_field(&fields, "cegis_active_fixtures_added")?,
                parse_u64_field(&fields, "cegis_counterexamples_recorded")?,
                parse_u64_field(&fields, "cegis_survivors_admitted")?,
            ]),
        };
        checkpoint.validate_invariants()?;
        Ok(checkpoint)
    }

    fn validate_invariants(&self) -> Result<(), CegisError> {
        if self.checkpoint_format_version != CEGIS_CHECKPOINT_VERSION {
            return Err(CegisError::InvalidCheckpoint(
                "checkpoint_format_version mismatch".into(),
            ));
        }
        if self.cegis_version != CEGIS_VERSION {
            return Err(CegisError::InvalidCheckpoint(
                "cegis_version mismatch".into(),
            ));
        }
        if self.config_fingerprint != config_fingerprint_text(&self.config) {
            return Err(CegisError::InvalidCheckpoint(
                "config_fingerprint mismatch".into(),
            ));
        }
        if u64::try_from(self.rejections.len()).unwrap_or(u64::MAX)
            != self.cegis_stats.counterexamples_recorded()
        {
            return Err(CegisError::InvalidCheckpoint(
                "rejection count does not match cegis_counterexamples_recorded".into(),
            ));
        }
        if u64::try_from(self.survivors.len()).unwrap_or(u64::MAX)
            > self.cegis_stats.survivors_admitted()
        {
            return Err(CegisError::InvalidCheckpoint(
                "survivor count exceeds cegis_survivors_admitted".into(),
            ));
        }
        if self.cegis_stats.survivors_falsified() > self.cegis_stats.survivors_retested() {
            return Err(CegisError::InvalidCheckpoint(
                "survivors_falsified exceeds survivors_retested".into(),
            ));
        }
        if u64::try_from(self.active_fixtures.len()).unwrap_or(u64::MAX)
            > self.config.max_active_fixtures
        {
            return Err(CegisError::ExceedsLimit {
                field: "active_fixtures",
                value: u64::try_from(self.active_fixtures.len()).unwrap_or(u64::MAX),
                maximum: self.config.max_active_fixtures,
            });
        }
        let search = SearchCheckpoint::from_canonical_text(&self.search_checkpoint_text).map_err(
            |error| CegisError::InvalidCheckpoint(format!("nested search checkpoint: {error}")),
        )?;
        if search.space_fingerprint().primary() != self.space_primary
            || search.space_fingerprint().secondary() != self.space_secondary
            || search.space_fingerprint().length() != self.space_length
        {
            return Err(CegisError::InvalidCheckpoint(
                "space fingerprint does not match nested search checkpoint".into(),
            ));
        }
        // Intentionally no adopted/qualified marker: checkpoint ≠ qualification.
        Ok(())
    }
}

fn config_fingerprint_text(config: &CegisConfig) -> String {
    let body = format!(
        "seed={}\nmax_active_fixtures={}\nmax_counterexample_artifacts={}\nmax_adversarial_outputs={}\n",
        config.seed,
        config.max_active_fixtures,
        config.max_counterexample_artifacts,
        config.max_adversarial_outputs
    );
    SearchFingerprint::of_canonical_text(&body).to_string()
}

fn collect_active_fixtures(
    fixtures: &[Fixture<impl Clone>],
    maximum: u64,
) -> Result<Vec<CheckpointFixture>, CegisError> {
    let count = u64::try_from(fixtures.len()).unwrap_or(u64::MAX);
    if count > maximum || count > MAX_ACTIVE_FIXTURES {
        return Err(CegisError::ExceedsLimit {
            field: "active_fixtures",
            value: count,
            maximum: maximum.min(MAX_ACTIVE_FIXTURES),
        });
    }
    let mut out = Vec::with_capacity(fixtures.len());
    for fixture in fixtures {
        validate_fixture_id(fixture.id())?;
        validate_fixture_text(fixture.canonical_text())?;
        out.push(CheckpointFixture {
            id: fixture.id().to_owned(),
            fingerprint: fixture.fingerprint().to_string(),
            canonical_text: fixture.canonical_text().to_owned(),
        });
    }
    Ok(out)
}

fn collect_rejections(
    rejected: &[RejectedCandidate<impl Clone, impl Clone>],
    maximum: u64,
) -> Result<Vec<CounterexampleArtifact>, CegisError> {
    let count = u64::try_from(rejected.len()).unwrap_or(u64::MAX);
    if count > maximum || count > MAX_COUNTEREXAMPLE_ARTIFACTS {
        return Err(CegisError::ExceedsLimit {
            field: "rejections",
            value: count,
            maximum: maximum.min(MAX_COUNTEREXAMPLE_ARTIFACTS),
        });
    }
    Ok(rejected
        .iter()
        .map(|item| item.counterexample().artifact())
        .collect())
}

fn collect_survivors(
    survivors: &[SearchCandidate<impl Clone>],
) -> Result<Vec<CheckpointSurvivor>, CegisError> {
    let mut out = Vec::with_capacity(survivors.len());
    for survivor in survivors {
        validate_fixture_text(survivor.canonical_text()).map_err(|_| {
            CegisError::InvalidCheckpoint("survivor canonical text is empty or oversized".into())
        })?;
        out.push(CheckpointSurvivor {
            ordinal: survivor.ordinal(),
            fingerprint: survivor.fingerprint().to_string(),
            canonical_text: survivor.canonical_text().to_owned(),
        });
    }
    Ok(out)
}

fn parse_checkpoint_fields(text: &str) -> Result<BTreeMap<&str, &str>, CegisError> {
    if text.len() > MAX_CEGIS_CHECKPOINT_TEXT_BYTES || text.contains('\r') {
        return Err(CegisError::InvalidCheckpoint(
            "checkpoint exceeds its limit or contains CR".into(),
        ));
    }
    if !text.ends_with('\n') {
        return Err(CegisError::InvalidCheckpoint(
            "checkpoint must end with a newline".into(),
        ));
    }
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err(CegisError::InvalidCheckpoint(
            "missing version header".into(),
        ));
    };
    let expected = format!("ADA-CEGIS-CHECKPOINT-V{CEGIS_CHECKPOINT_VERSION}");
    if header != expected {
        return Err(CegisError::InvalidCheckpoint(
            "unsupported or non-canonical version header".into(),
        ));
    }
    let mut fields = BTreeMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Err(CegisError::InvalidCheckpoint("field is missing '='".into()));
        };
        if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
            return Err(CegisError::InvalidCheckpoint(
                "empty, duplicate, or ambiguous field".into(),
            ));
        }
    }
    if BASE_FIELDS.iter().any(|key| !fields.contains_key(key)) {
        return Err(CegisError::InvalidCheckpoint(
            "base field set is incomplete".into(),
        ));
    }
    Ok(fields)
}

fn expect_field_cardinality(
    fields: &BTreeMap<&str, &str>,
    active_count: u64,
    rejection_count: u64,
    survivor_count: u64,
) -> Result<(), CegisError> {
    let active_fields = usize::try_from(active_count)
        .unwrap_or(usize::MAX)
        .checked_mul(3)
        .ok_or_else(|| CegisError::InvalidCheckpoint("active field count overflow".into()))?;
    let rejection_fields = usize::try_from(rejection_count).unwrap_or(usize::MAX);
    let survivor_fields = usize::try_from(survivor_count)
        .unwrap_or(usize::MAX)
        .checked_mul(3)
        .ok_or_else(|| CegisError::InvalidCheckpoint("survivor field count overflow".into()))?;
    let expected = BASE_FIELDS
        .len()
        .checked_add(active_fields)
        .and_then(|n| n.checked_add(rejection_fields))
        .and_then(|n| n.checked_add(survivor_fields))
        .ok_or_else(|| CegisError::InvalidCheckpoint("field cardinality overflow".into()))?;
    if fields.len() != expected {
        return Err(CegisError::InvalidCheckpoint(
            "field set is incomplete or has unknown keys".into(),
        ));
    }
    for index in 0..active_count {
        for suffix in ["id", "fingerprint", "text"] {
            let key = format!("active_{index}_{suffix}");
            if !fields.contains_key(key.as_str()) {
                return Err(CegisError::InvalidCheckpoint(
                    "active fixture fields are not contiguous".into(),
                ));
            }
        }
    }
    for index in 0..rejection_count {
        let key = format!("rejection_{index}");
        if !fields.contains_key(key.as_str()) {
            return Err(CegisError::InvalidCheckpoint(
                "rejection fields are not contiguous".into(),
            ));
        }
    }
    for index in 0..survivor_count {
        for suffix in ["ordinal", "fingerprint", "text"] {
            let key = format!("survivor_{index}_{suffix}");
            if !fields.contains_key(key.as_str()) {
                return Err(CegisError::InvalidCheckpoint(
                    "survivor fields are not contiguous".into(),
                ));
            }
        }
    }
    Ok(())
}

fn parse_active_fixtures(
    fields: &BTreeMap<&str, &str>,
    count: u64,
) -> Result<Vec<CheckpointFixture>, CegisError> {
    let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for index in 0..count {
        let id = hex_decode(field(fields, &format!("active_{index}_id"))?)?;
        validate_fixture_id(&id)?;
        let fingerprint = field(fields, &format!("active_{index}_fingerprint"))?.to_owned();
        validate_fingerprint_text(&fingerprint, "active fixture fingerprint")?;
        let canonical_text = hex_decode(field(fields, &format!("active_{index}_text"))?)?;
        validate_fixture_text(&canonical_text)?;
        let rebuilt = Fixture::new(id.clone(), canonical_text.clone(), ())?;
        if rebuilt.fingerprint().to_string() != fingerprint {
            return Err(CegisError::InvalidCheckpoint(
                "active fixture fingerprint does not match payload".into(),
            ));
        }
        out.push(CheckpointFixture {
            id,
            fingerprint,
            canonical_text,
        });
    }
    Ok(out)
}

fn parse_rejections(
    fields: &BTreeMap<&str, &str>,
    count: u64,
) -> Result<Vec<CounterexampleArtifact>, CegisError> {
    let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for index in 0..count {
        let text = hex_decode(field(fields, &format!("rejection_{index}"))?)?;
        out.push(CounterexampleArtifact::from_canonical_text(&text)?);
    }
    Ok(out)
}

fn parse_survivors(
    fields: &BTreeMap<&str, &str>,
    count: u64,
) -> Result<Vec<CheckpointSurvivor>, CegisError> {
    let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for index in 0..count {
        let ordinal = parse_u64_field(fields, &format!("survivor_{index}_ordinal"))?;
        let fingerprint = field(fields, &format!("survivor_{index}_fingerprint"))?.to_owned();
        validate_fingerprint_text(&fingerprint, "survivor fingerprint")?;
        let canonical_text = hex_decode(field(fields, &format!("survivor_{index}_text"))?)?;
        if canonical_text.is_empty() {
            return Err(CegisError::InvalidCheckpoint(
                "survivor canonical text is empty".into(),
            ));
        }
        out.push(CheckpointSurvivor {
            ordinal,
            fingerprint,
            canonical_text,
        });
    }
    Ok(out)
}

fn parse_u16_field(fields: &BTreeMap<&str, &str>, key: &str) -> Result<u16, CegisError> {
    field(fields, key)?
        .parse::<u16>()
        .map_err(|_| CegisError::InvalidCheckpoint(format!("{key} is not u16")))
}

fn parse_hex_u64_field(fields: &BTreeMap<&str, &str>, key: &str) -> Result<u64, CegisError> {
    let value = field(fields, key)?;
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(CegisError::InvalidCheckpoint(format!(
            "{key} is not a 16-digit hexadecimal lane"
        )));
    }
    u64::from_str_radix(value, 16)
        .map_err(|_| CegisError::InvalidCheckpoint(format!("{key} has invalid hex")))
}
