//! ADA-A11-E2 semantic/mechanistic evidence interchange.
//!
//! The historical A10 [`crate::EvidenceRecord`] remains the schema for
//! hardware-oriented ADA evidence. This module adds a separate versioned
//! artifact for semantic research so ITD, TDI, task benches, adversarial
//! harnesses and later producers can bind evidence to the same ADA semantic and
//! workload without becoming dependencies of `ada-core`.

use ada_core::{DiagnosticEvidenceKind, DiagnosticEvidenceRef, SemanticContractError, SemanticId};
use ada_workload::{WorkloadContract, WorkloadFingerprint};

mod canonical;

#[cfg(test)]
mod tests;

/// Canonical semantic evidence schema version.
pub const SEMANTIC_EVIDENCE_VERSION: u16 = 1;
/// Canonical artifact header.
pub const SEMANTIC_EVIDENCE_HEADER: &str = "ADA-SEMANTIC-EVIDENCE-V1";
/// Upper bound for one canonical semantic-evidence artifact.
pub const MAX_SEMANTIC_EVIDENCE_BYTES: usize = 1 << 20;
/// Upper bound for names and external identifiers stored in this schema.
pub const MAX_EVIDENCE_IDENTIFIER_BYTES: usize = 256;
/// Upper bound for scalar summary metrics in one record.
pub const MAX_SUMMARY_METRICS: usize = 1024;

/// Serializable copy of the stable three-lane workload fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvidenceWorkloadFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl EvidenceWorkloadFingerprint {
    /// Bind to an already-validated workload fingerprint.
    #[must_use]
    pub const fn from_workload_fingerprint(fingerprint: WorkloadFingerprint) -> Self {
        Self {
            primary: fingerprint.primary(),
            secondary: fingerprint.secondary(),
            length: fingerprint.length(),
        }
    }

    /// Bind directly to a workload's canonical fingerprint.
    #[must_use]
    pub fn from_workload(workload: &WorkloadContract) -> Self {
        Self::from_workload_fingerprint(workload.fingerprint())
    }

    /// Primary fingerprint lane.
    #[must_use]
    pub const fn primary(self) -> u64 {
        self.primary
    }

    /// Secondary fingerprint lane.
    #[must_use]
    pub const fn secondary(self) -> u64 {
        self.secondary
    }

    /// Canonical workload text byte length included in the fingerprint.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }

    pub(super) const fn from_parts(primary: u64, secondary: u64, length: u64) -> Self {
        Self {
            primary,
            secondary,
            length,
        }
    }
}

impl From<WorkloadFingerprint> for EvidenceWorkloadFingerprint {
    fn from(value: WorkloadFingerprint) -> Self {
        Self::from_workload_fingerprint(value)
    }
}

/// Construction input for one semantic evidence record.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticEvidenceSpec {
    pub semantic: SemanticId,
    pub workload: EvidenceWorkloadFingerprint,
    pub kind: DiagnosticEvidenceKind,
    pub producer_repository: String,
    pub producer_revision: String,
    pub artifact_identity: String,
    pub intervention_identity: Option<String>,
    pub observation_horizon: Option<u32>,
    pub metric_identity: String,
    pub sha256_evidence: String,
    pub metrics: Vec<(String, f64)>,
}

/// Fail-closed errors for semantic evidence construction or decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticEvidenceError {
    UnsupportedVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidRepository,
    InvalidProducerRevision,
    InvalidEvidenceDigest,
    TooManyMetrics,
    DuplicateMetric,
    NonFiniteMetric,
    MissingIntervention,
    MissingObservationHorizon,
    SemanticIdentity(SemanticContractError),
    MalformedCanonicalText(String),
}

impl From<SemanticContractError> for SemanticEvidenceError {
    fn from(value: SemanticContractError) -> Self {
        Self::SemanticIdentity(value)
    }
}

/// Versioned evidence binding for semantic/mechanistic attention research.
///
/// Evidence is attached to a semantic and workload; it never participates in
/// semantic identity. The producer is bound by repository, exact Git revision
/// and SHA-256 of the preserved evidence artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticEvidenceRecord {
    pub(super) version: u16,
    pub(super) semantic: SemanticId,
    pub(super) workload: EvidenceWorkloadFingerprint,
    pub(super) kind: DiagnosticEvidenceKind,
    pub(super) producer_repository: String,
    pub(super) producer_revision: String,
    pub(super) artifact_identity: String,
    pub(super) intervention_identity: Option<String>,
    pub(super) observation_horizon: Option<u32>,
    pub(super) metric_identity: String,
    pub(super) sha256_evidence: String,
    pub(super) metrics: Vec<(String, f64)>,
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), SemanticEvidenceError> {
    if value.is_empty()
        || value.len() > MAX_EVIDENCE_IDENTIFIER_BYTES
        || value.chars().any(char::is_control)
        || value.chars().any(char::is_whitespace)
    {
        return Err(SemanticEvidenceError::InvalidIdentifier(field));
    }
    Ok(())
}

fn valid_repository(value: &str) -> bool {
    let mut parts = value.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(repository) = parts.next() else {
        return false;
    };
    if parts.next().is_some() || owner.is_empty() || repository.is_empty() {
        return false;
    }
    owner
        .bytes()
        .chain(repository.bytes())
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

pub(super) fn is_lower_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl SemanticEvidenceRecord {
    /// Construct and validate one semantic evidence binding.
    ///
    /// Summary metrics are sorted by name so canonical identity is independent
    /// of producer insertion order. Duplicate names fail closed.
    ///
    /// # Errors
    ///
    /// Returns a precise error for malformed provenance, non-finite metrics,
    /// duplicate metrics, or missing TDI intervention/recovery metadata.
    pub fn new(mut spec: SemanticEvidenceSpec) -> Result<Self, SemanticEvidenceError> {
        validate_identifier("semantic_name", spec.semantic.name())?;
        if !valid_repository(&spec.producer_repository) {
            return Err(SemanticEvidenceError::InvalidRepository);
        }
        if !is_lower_hex_exact(&spec.producer_revision, 40) {
            return Err(SemanticEvidenceError::InvalidProducerRevision);
        }
        validate_identifier("artifact_identity", &spec.artifact_identity)?;
        if let Some(intervention) = &spec.intervention_identity {
            validate_identifier("intervention_identity", intervention)?;
        }
        validate_identifier("metric_identity", &spec.metric_identity)?;
        if !is_lower_hex_exact(&spec.sha256_evidence, 64) {
            return Err(SemanticEvidenceError::InvalidEvidenceDigest);
        }
        if spec.metrics.len() > MAX_SUMMARY_METRICS {
            return Err(SemanticEvidenceError::TooManyMetrics);
        }
        for (name, value) in &spec.metrics {
            validate_identifier("metric_name", name)?;
            if !value.is_finite() {
                return Err(SemanticEvidenceError::NonFiniteMetric);
            }
        }
        spec.metrics.sort_by(|left, right| left.0.cmp(&right.0));
        if spec.metrics.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(SemanticEvidenceError::DuplicateMetric);
        }

        if matches!(spec.kind, DiagnosticEvidenceKind::TdiRecovery) {
            if spec.intervention_identity.is_none() {
                return Err(SemanticEvidenceError::MissingIntervention);
            }
            if spec.observation_horizon.is_none() {
                return Err(SemanticEvidenceError::MissingObservationHorizon);
            }
        }

        Ok(Self {
            version: SEMANTIC_EVIDENCE_VERSION,
            semantic: spec.semantic,
            workload: spec.workload,
            kind: spec.kind,
            producer_repository: spec.producer_repository,
            producer_revision: spec.producer_revision,
            artifact_identity: spec.artifact_identity,
            intervention_identity: spec.intervention_identity,
            observation_horizon: spec.observation_horizon,
            metric_identity: spec.metric_identity,
            sha256_evidence: spec.sha256_evidence,
            metrics: spec.metrics,
        })
    }

    /// Schema version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Semantic hypothesis to which this evidence applies.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }

    /// Workload binding used to produce the evidence.
    #[must_use]
    pub const fn workload(&self) -> EvidenceWorkloadFingerprint {
        self.workload
    }

    /// Kind of evidence produced.
    #[must_use]
    pub const fn kind(&self) -> DiagnosticEvidenceKind {
        self.kind
    }

    /// Producing GitHub repository in `owner/name` form.
    #[must_use]
    pub fn producer_repository(&self) -> &str {
        &self.producer_repository
    }

    /// Exact lowercase 40-hex producer Git revision.
    #[must_use]
    pub fn producer_revision(&self) -> &str {
        &self.producer_revision
    }

    /// Producer-defined artifact identity.
    #[must_use]
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }

    /// Intervention identity, when the evidence is intervention-based.
    #[must_use]
    pub fn intervention_identity(&self) -> Option<&str> {
        self.intervention_identity.as_deref()
    }

    /// Observation horizon, when defined by the evidence producer.
    #[must_use]
    pub const fn observation_horizon(&self) -> Option<u32> {
        self.observation_horizon
    }

    /// Primary metric/protocol identity.
    #[must_use]
    pub fn metric_identity(&self) -> &str {
        &self.metric_identity
    }

    /// SHA-256 digest of the preserved raw evidence artifact.
    #[must_use]
    pub fn sha256_evidence(&self) -> &str {
        &self.sha256_evidence
    }

    /// Sorted finite scalar summaries. These are metadata, not replacements for
    /// the raw evidence artifact bound by SHA-256.
    #[must_use]
    pub fn metrics(&self) -> &[(String, f64)] {
        &self.metrics
    }

    /// Convert this fully bound artifact to the lighter A11 evidence reference
    /// stored by semantic qualification/graduation records.
    ///
    /// # Errors
    ///
    /// Propagates the core evidence-reference validation contract.
    pub fn diagnostic_reference(&self) -> Result<DiagnosticEvidenceRef, SemanticContractError> {
        DiagnosticEvidenceRef::new(
            self.kind,
            self.producer_repository.clone(),
            self.artifact_identity.clone(),
            format!(
                "git:{};sha256:{}",
                self.producer_revision, self.sha256_evidence
            ),
        )
    }
}
