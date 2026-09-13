//! ADA-A11 machine-readable `SemanticQualificationRecord` (`schema_version` 1).
//!
//! This module implements typed construction/validation and a deterministic
//! canonical JSON encode/decode for
//! `docs/ada-a11-flat-graduation-record.schema.json`.
//!
//! # Boundary with `FlatGraduationRecord` / `FlatGraduationBundle`
//!
//! `ada_core::FlatGraduationRecord` and `ada_graduation::FlatGraduationBundle`
//! are related research handoff artifacts, but they do **not** contain every
//! required field of this JSON schema (for example `candidate_id`,
//! `invariants`, `mask_state_contract`, `numeric_policy`,
//! `prior_art_status`, and separately provenance-bound ITD/TDI lanes with
//! exact `commit` + `sha256`). There is therefore **no automatic conversion**
//! from those types into [`SemanticQualificationRecord`]. Callers must supply
//! each required evidence lane explicitly; missing evidence must fail closed
//! rather than be invented.
//!
//! This codec does not implement FLAT kernels, GPU lowering, semantic search
//! families, ITD/TDI runtime dependencies, or SciRust/Riemann interpretation
//! transfer.

mod json;

#[cfg(test)]
mod tests;

/// Schema version fixed by `docs/ada-a11-flat-graduation-record.schema.json`.
pub const SEMANTIC_QUALIFICATION_SCHEMA_VERSION: u16 = 1;

/// One provenance-bound evidence artifact reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    /// Evidence kind label (producer-defined; non-empty).
    pub kind: String,
    /// Producing repository identity string (non-empty).
    pub repository: String,
    /// Exact lowercase 40-hex Git commit.
    pub commit: String,
    /// Artifact identity / path (non-empty).
    pub artifact: String,
    /// Lowercase 64-hex SHA-256 of the referenced artifact bytes.
    pub sha256: String,
}

/// Exact reference definition binding for the candidate semantic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceDefinition {
    /// Repository identity string (non-empty).
    pub repository: String,
    /// Exact lowercase 40-hex Git commit.
    pub commit: String,
    /// Artifact identity / path (non-empty).
    pub artifact: String,
    /// Lowercase 64-hex SHA-256 of the referenced artifact bytes.
    pub sha256: String,
}

/// Mask and state contracts kept separate from semantic identity hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskStateContract {
    /// Mask contract label.
    pub mask: String,
    /// State contract label.
    pub state: String,
}

/// Numeric reference policy kept separate from semantic identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericPolicy {
    /// Declared reference precision policy.
    pub reference_precision: String,
    /// Declared non-finite handling policy.
    pub non_finite_policy: String,
}

/// External mechanistic diagnostics held outside semantic identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalDiagnostics {
    /// ITD structural evidence references (may be empty).
    pub itd: Vec<EvidenceRef>,
    /// TDI recovery evidence references (may be empty).
    pub tdi: Vec<EvidenceRef>,
}

/// Prior-art review status, never inferred from usefulness evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorArtStatusKind {
    /// No dedicated prior-art review has been performed.
    NotAssessed,
    /// A prior-art search is in progress.
    SearchInProgress,
    /// Known prior art covers the candidate at the stated scope.
    KnownPriorArt,
    /// Distinctness remains unresolved after review work.
    DistinctnessUnresolved,
    /// Review completed with an explicit no-novelty-claim outcome.
    ReviewedNoClaim,
}

impl PriorArtStatusKind {
    /// Canonical JSON enum string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotAssessed => "not_assessed",
            Self::SearchInProgress => "search_in_progress",
            Self::KnownPriorArt => "known_prior_art",
            Self::DistinctnessUnresolved => "distinctness_unresolved",
            Self::ReviewedNoClaim => "reviewed_no_claim",
        }
    }

    fn parse(value: &str) -> Result<Self, SemanticQualificationError> {
        match value {
            "not_assessed" => Ok(Self::NotAssessed),
            "search_in_progress" => Ok(Self::SearchInProgress),
            "known_prior_art" => Ok(Self::KnownPriorArt),
            "distinctness_unresolved" => Ok(Self::DistinctnessUnresolved),
            "reviewed_no_claim" => Ok(Self::ReviewedNoClaim),
            _ => Err(SemanticQualificationError::InvalidEnum {
                field: "prior_art_status.status",
                value: value.to_owned(),
            }),
        }
    }
}

/// Prior-art status plus optional supporting references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorArtStatus {
    /// Explicit prior-art status enum.
    pub status: PriorArtStatusKind,
    /// Supporting prior-art references (may be empty).
    pub references: Vec<EvidenceRef>,
}

/// Research verdict lane, kept separate from evidence payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualificationVerdictJson {
    /// Promote/adopt only after stronger external gates; JSON may carry this
    /// enum value but ADA research codecs must not invent adoption evidence.
    Adopt,
    /// Adapt the candidate under further research constraints.
    Adapt,
    /// Reject the candidate at the stated research scope.
    Reject,
    /// Continue research without adoption.
    ContinueResearch,
}

impl QualificationVerdictJson {
    /// Canonical JSON enum string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adopt => "ADOPT",
            Self::Adapt => "ADAPT",
            Self::Reject => "REJECT",
            Self::ContinueResearch => "CONTINUE_RESEARCH",
        }
    }

    fn parse(value: &str) -> Result<Self, SemanticQualificationError> {
        match value {
            "ADOPT" => Ok(Self::Adopt),
            "ADAPT" => Ok(Self::Adapt),
            "REJECT" => Ok(Self::Reject),
            "CONTINUE_RESEARCH" => Ok(Self::ContinueResearch),
            _ => Err(SemanticQualificationError::InvalidEnum {
                field: "verdict",
                value: value.to_owned(),
            }),
        }
    }
}

/// Construction input for one schema_version-1 qualification record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticQualificationSpec {
    /// Stable candidate identifier (non-empty).
    pub candidate_id: String,
    /// Semantic identity string (non-empty); not an implementation id.
    pub semantic_id: String,
    /// Positive semantic version.
    pub semantic_version: u32,
    /// Exact reference definition binding.
    pub reference_definition: ReferenceDefinition,
    /// Non-empty unique invariant statements.
    pub invariants: Vec<String>,
    /// Mask/state contract labels.
    pub mask_state_contract: MaskStateContract,
    /// Numeric policy labels.
    pub numeric_policy: NumericPolicy,
    /// Oracle fixture evidence references.
    pub oracle_fixtures: Vec<EvidenceRef>,
    /// Adversarial case evidence references.
    pub adversarial_cases: Vec<EvidenceRef>,
    /// Task-behavior evidence references.
    pub task_evidence: Vec<EvidenceRef>,
    /// External ITD/TDI diagnostics.
    pub external_diagnostics: ExternalDiagnostics,
    /// Logical cost evidence references.
    pub logical_cost_evidence: Vec<EvidenceRef>,
    /// Prior-art status and references.
    pub prior_art_status: PriorArtStatus,
    /// Explicit research verdict.
    pub verdict: QualificationVerdictJson,
}

/// Fail-closed errors for construction, validation, or JSON codec use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticQualificationError {
    /// Unsupported schema version.
    UnsupportedSchemaVersion(u16),
    /// A required non-empty text field is empty.
    EmptyField(&'static str),
    /// `semantic_version` is zero.
    ZeroSemanticVersion,
    /// `invariants` is empty.
    EmptyInvariants,
    /// `invariants` contains a duplicate entry.
    DuplicateInvariant,
    /// A commit field is not exactly 40 lowercase hex characters.
    InvalidCommit(&'static str),
    /// A SHA-256 field is not exactly 64 lowercase hex characters.
    InvalidSha256(&'static str),
    /// An enum field has an unrecognized value.
    InvalidEnum {
        /// Field path.
        field: &'static str,
        /// Rejected value.
        value: String,
    },
    /// Canonical JSON text is malformed or non-canonical for this schema.
    MalformedJson(String),
    /// An unexpected JSON property was present (`additionalProperties: false`).
    UnknownProperty {
        /// Object path that contained the property.
        path: &'static str,
        /// Rejected property name.
        key: String,
    },
    /// A required JSON property was missing.
    MissingProperty {
        /// Object path that should contain the property.
        path: &'static str,
        /// Missing property name.
        key: &'static str,
    },
    /// A JSON value had the wrong type for the schema field.
    TypeMismatch {
        /// Field path.
        field: &'static str,
        /// Expected JSON type description.
        expected: &'static str,
    },
}

impl std::fmt::Display for SemanticQualificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported schema_version {version}")
            }
            Self::EmptyField(field) => write!(formatter, "field `{field}` must be non-empty"),
            Self::ZeroSemanticVersion => formatter.write_str("semantic_version must be >= 1"),
            Self::EmptyInvariants => {
                formatter.write_str("invariants must contain at least one item")
            }
            Self::DuplicateInvariant => formatter.write_str("invariants must be unique"),
            Self::InvalidCommit(field) => {
                write!(
                    formatter,
                    "`{field}` must be exactly 40 lowercase hex characters"
                )
            }
            Self::InvalidSha256(field) => {
                write!(
                    formatter,
                    "`{field}` must be exactly 64 lowercase hex characters"
                )
            }
            Self::InvalidEnum { field, value } => {
                write!(formatter, "invalid enum for `{field}`: {value}")
            }
            Self::MalformedJson(reason) => write!(formatter, "malformed JSON: {reason}"),
            Self::UnknownProperty { path, key } => {
                write!(formatter, "unknown property `{key}` under `{path}`")
            }
            Self::MissingProperty { path, key } => {
                write!(formatter, "missing property `{key}` under `{path}`")
            }
            Self::TypeMismatch { field, expected } => {
                write!(
                    formatter,
                    "type mismatch for `{field}`: expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for SemanticQualificationError {}

/// Validated schema_version-1 semantic qualification record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticQualificationRecord {
    schema_version: u16,
    candidate_id: String,
    semantic_id: String,
    semantic_version: u32,
    reference_definition: ReferenceDefinition,
    invariants: Vec<String>,
    mask_state_contract: MaskStateContract,
    numeric_policy: NumericPolicy,
    oracle_fixtures: Vec<EvidenceRef>,
    adversarial_cases: Vec<EvidenceRef>,
    task_evidence: Vec<EvidenceRef>,
    external_diagnostics: ExternalDiagnostics,
    logical_cost_evidence: Vec<EvidenceRef>,
    prior_art_status: PriorArtStatus,
    verdict: QualificationVerdictJson,
}

impl SemanticQualificationRecord {
    /// Construct and validate one record.
    ///
    /// # Errors
    ///
    /// Fails closed on empty required strings, zero semantic version, empty or
    /// duplicate invariants, malformed commit/sha256 digests, or schema version
    /// mismatch.
    pub fn new(spec: SemanticQualificationSpec) -> Result<Self, SemanticQualificationError> {
        let record = Self {
            schema_version: SEMANTIC_QUALIFICATION_SCHEMA_VERSION,
            candidate_id: spec.candidate_id,
            semantic_id: spec.semantic_id,
            semantic_version: spec.semantic_version,
            reference_definition: spec.reference_definition,
            invariants: spec.invariants,
            mask_state_contract: spec.mask_state_contract,
            numeric_policy: spec.numeric_policy,
            oracle_fixtures: spec.oracle_fixtures,
            adversarial_cases: spec.adversarial_cases,
            task_evidence: spec.task_evidence,
            external_diagnostics: spec.external_diagnostics,
            logical_cost_evidence: spec.logical_cost_evidence,
            prior_art_status: spec.prior_art_status,
            verdict: spec.verdict,
        };
        record.validate()?;
        Ok(record)
    }

    /// Validate schema constraints without re-allocating.
    ///
    /// # Errors
    ///
    /// Returns the first violated schema rule.
    pub fn validate(&self) -> Result<(), SemanticQualificationError> {
        if self.schema_version != SEMANTIC_QUALIFICATION_SCHEMA_VERSION {
            return Err(SemanticQualificationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        require_non_empty(&self.candidate_id, "candidate_id")?;
        require_non_empty(&self.semantic_id, "semantic_id")?;
        if self.semantic_version == 0 {
            return Err(SemanticQualificationError::ZeroSemanticVersion);
        }
        validate_reference_definition(&self.reference_definition)?;
        if self.invariants.is_empty() {
            return Err(SemanticQualificationError::EmptyInvariants);
        }
        let mut seen = std::collections::BTreeSet::new();
        for invariant in &self.invariants {
            require_non_empty(invariant, "invariants")?;
            if !seen.insert(invariant.as_str()) {
                return Err(SemanticQualificationError::DuplicateInvariant);
            }
        }
        require_non_empty(&self.mask_state_contract.mask, "mask_state_contract.mask")?;
        require_non_empty(&self.mask_state_contract.state, "mask_state_contract.state")?;
        require_non_empty(
            &self.numeric_policy.reference_precision,
            "numeric_policy.reference_precision",
        )?;
        require_non_empty(
            &self.numeric_policy.non_finite_policy,
            "numeric_policy.non_finite_policy",
        )?;
        validate_evidence_list(&self.oracle_fixtures)?;
        validate_evidence_list(&self.adversarial_cases)?;
        validate_evidence_list(&self.task_evidence)?;
        validate_evidence_list(&self.external_diagnostics.itd)?;
        validate_evidence_list(&self.external_diagnostics.tdi)?;
        validate_evidence_list(&self.logical_cost_evidence)?;
        validate_evidence_list(&self.prior_art_status.references)?;
        Ok(())
    }

    /// Schema version (always 1 for this codec).
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Candidate identifier.
    #[must_use]
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    /// Semantic identity string.
    #[must_use]
    pub fn semantic_id(&self) -> &str {
        &self.semantic_id
    }

    /// Semantic version.
    #[must_use]
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }

    /// Reference definition binding.
    #[must_use]
    pub const fn reference_definition(&self) -> &ReferenceDefinition {
        &self.reference_definition
    }

    /// Invariant statements.
    #[must_use]
    pub fn invariants(&self) -> &[String] {
        &self.invariants
    }

    /// Mask/state contract.
    #[must_use]
    pub const fn mask_state_contract(&self) -> &MaskStateContract {
        &self.mask_state_contract
    }

    /// Numeric policy.
    #[must_use]
    pub const fn numeric_policy(&self) -> &NumericPolicy {
        &self.numeric_policy
    }

    /// Oracle fixture evidence.
    #[must_use]
    pub fn oracle_fixtures(&self) -> &[EvidenceRef] {
        &self.oracle_fixtures
    }

    /// Adversarial case evidence.
    #[must_use]
    pub fn adversarial_cases(&self) -> &[EvidenceRef] {
        &self.adversarial_cases
    }

    /// Task evidence.
    #[must_use]
    pub fn task_evidence(&self) -> &[EvidenceRef] {
        &self.task_evidence
    }

    /// External diagnostics.
    #[must_use]
    pub const fn external_diagnostics(&self) -> &ExternalDiagnostics {
        &self.external_diagnostics
    }

    /// Logical cost evidence.
    #[must_use]
    pub fn logical_cost_evidence(&self) -> &[EvidenceRef] {
        &self.logical_cost_evidence
    }

    /// Prior-art status.
    #[must_use]
    pub const fn prior_art_status(&self) -> &PriorArtStatus {
        &self.prior_art_status
    }

    /// Research verdict.
    #[must_use]
    pub const fn verdict(&self) -> QualificationVerdictJson {
        self.verdict
    }

    /// Encode deterministic canonical JSON matching the schema key order.
    #[must_use]
    pub fn to_canonical_json(&self) -> String {
        json::encode_record(self)
    }

    /// Decode and validate canonical JSON for this fixed schema.
    ///
    /// # Errors
    ///
    /// Rejects malformed JSON, unknown properties, missing fields, type
    /// mismatches, and all constructor-level schema violations.
    pub fn from_canonical_json(text: &str) -> Result<Self, SemanticQualificationError> {
        json::decode_record(text)
    }
}

fn require_non_empty(value: &str, field: &'static str) -> Result<(), SemanticQualificationError> {
    if value.is_empty() {
        Err(SemanticQualificationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn is_lower_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_commit(value: &str, field: &'static str) -> Result<(), SemanticQualificationError> {
    if is_lower_hex_exact(value, 40) {
        Ok(())
    } else {
        Err(SemanticQualificationError::InvalidCommit(field))
    }
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), SemanticQualificationError> {
    if is_lower_hex_exact(value, 64) {
        Ok(())
    } else {
        Err(SemanticQualificationError::InvalidSha256(field))
    }
}

fn validate_reference_definition(
    value: &ReferenceDefinition,
) -> Result<(), SemanticQualificationError> {
    require_non_empty(&value.repository, "reference_definition.repository")?;
    validate_commit(&value.commit, "reference_definition.commit")?;
    require_non_empty(&value.artifact, "reference_definition.artifact")?;
    validate_sha256(&value.sha256, "reference_definition.sha256")?;
    Ok(())
}

fn validate_evidence_ref(value: &EvidenceRef) -> Result<(), SemanticQualificationError> {
    require_non_empty(&value.kind, "evidence.kind")?;
    require_non_empty(&value.repository, "evidence.repository")?;
    validate_commit(&value.commit, "evidence.commit")?;
    require_non_empty(&value.artifact, "evidence.artifact")?;
    validate_sha256(&value.sha256, "evidence.sha256")?;
    Ok(())
}

fn validate_evidence_list(values: &[EvidenceRef]) -> Result<(), SemanticQualificationError> {
    for value in values {
        validate_evidence_ref(value)?;
    }
    Ok(())
}
