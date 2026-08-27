//! ADA-A11-E2 semantic/mechanistic evidence interchange.
//!
//! The historical A10 [`crate::EvidenceRecord`] remains the schema for
//! hardware-oriented ADA evidence. This module adds a separate versioned
//! artifact for semantic research so ITD, TDI, task benches, adversarial
//! harnesses and later producers can bind evidence to the same ADA semantic and
//! workload without becoming dependencies of `ada-core`.

use std::collections::{BTreeMap, BTreeSet};

use ada_core::{
    DiagnosticEvidenceKind, DiagnosticEvidenceRef, SemanticContractError, SemanticFamily,
    SemanticId,
};
use ada_workload::{WorkloadContract, WorkloadFingerprint};

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

    const fn from_parts(primary: u64, secondary: u64, length: u64) -> Self {
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
    version: u16,
    semantic: SemanticId,
    workload: EvidenceWorkloadFingerprint,
    kind: DiagnosticEvidenceKind,
    producer_repository: String,
    producer_revision: String,
    artifact_identity: String,
    intervention_identity: Option<String>,
    observation_horizon: Option<u32>,
    metric_identity: String,
    sha256_evidence: String,
    metrics: Vec<(String, f64)>,
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

fn is_lower_hex_exact(value: &str, length: usize) -> bool {
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
        if spec
            .metrics
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0)
        {
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

    /// Canonical deterministic interchange text.
    ///
    /// Floating-point summaries are encoded by exact IEEE-754 bit pattern so
    /// re-serialization does not depend on decimal formatting.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(SEMANTIC_EVIDENCE_HEADER);
        text.push('\n');
        append_field(&mut text, "version", &self.version.to_string());
        append_field(
            &mut text,
            "semantic_family",
            semantic_family_text(self.semantic.family()),
        );
        append_field(
            &mut text,
            "semantic_name",
            &hex_encode(self.semantic.name()),
        );
        append_field(
            &mut text,
            "semantic_revision",
            &self.semantic.revision().to_string(),
        );
        append_field(
            &mut text,
            "workload_primary",
            &format!("{:016x}", self.workload.primary()),
        );
        append_field(
            &mut text,
            "workload_secondary",
            &format!("{:016x}", self.workload.secondary()),
        );
        append_field(
            &mut text,
            "workload_length",
            &format!("{:016x}", self.workload.length()),
        );
        append_field(&mut text, "evidence_kind", evidence_kind_text(self.kind));
        append_field(
            &mut text,
            "producer_repository",
            &hex_encode(&self.producer_repository),
        );
        append_field(&mut text, "producer_revision", &self.producer_revision);
        append_field(
            &mut text,
            "artifact_identity",
            &hex_encode(&self.artifact_identity),
        );
        append_field(
            &mut text,
            "intervention_identity",
            &self
                .intervention_identity
                .as_deref()
                .map_or_else(|| "-".into(), hex_encode),
        );
        append_field(
            &mut text,
            "observation_horizon",
            &self
                .observation_horizon
                .map_or_else(|| "-".into(), |value| value.to_string()),
        );
        append_field(
            &mut text,
            "metric_identity",
            &hex_encode(&self.metric_identity),
        );
        append_field(&mut text, "sha256_evidence", &self.sha256_evidence);
        append_field(&mut text, "metrics_count", &self.metrics.len().to_string());
        for (index, (name, value)) in self.metrics.iter().enumerate() {
            append_field(
                &mut text,
                &format!("metric_{index}_name"),
                &hex_encode(name),
            );
            append_field(
                &mut text,
                &format!("metric_{index}_bits"),
                &format!("{:016x}", value.to_bits()),
            );
        }
        text
    }

    /// Decode strict canonical interchange text.
    ///
    /// # Errors
    ///
    /// Rejects unknown/duplicate fields, unsupported versions, malformed
    /// fingerprints or float bits, invalid semantic identity, incomplete metric
    /// sets and all constructor-level provenance violations.
    pub fn from_canonical_text(text: &str) -> Result<Self, SemanticEvidenceError> {
        if text.len() > MAX_SEMANTIC_EVIDENCE_BYTES || text.contains('\r') || !text.ends_with('\n')
        {
            return Err(SemanticEvidenceError::MalformedCanonicalText(
                "artifact exceeds its limit, contains CR, or lacks final newline".into(),
            ));
        }
        let mut lines = text.lines();
        if lines.next() != Some(SEMANTIC_EVIDENCE_HEADER) {
            return Err(SemanticEvidenceError::MalformedCanonicalText(
                "missing ADA-SEMANTIC-EVIDENCE-V1 header".into(),
            ));
        }
        let mut fields = BTreeMap::new();
        for line in lines {
            let Some((key, value)) = line.split_once('=') else {
                return Err(SemanticEvidenceError::MalformedCanonicalText(
                    "field is missing '='".into(),
                ));
            };
            if key.is_empty() || value.contains('=') || fields.insert(key.to_string(), value).is_some()
            {
                return Err(SemanticEvidenceError::MalformedCanonicalText(
                    "empty, duplicate, or ambiguous field".into(),
                ));
            }
        }

        let field = |key: &str| {
            fields.get(key).copied().ok_or_else(|| {
                SemanticEvidenceError::MalformedCanonicalText(format!("missing field {key}"))
            })
        };
        let version = parse_u16("version", field("version")?)?;
        if version != SEMANTIC_EVIDENCE_VERSION {
            return Err(SemanticEvidenceError::UnsupportedVersion(version));
        }
        let metrics_count = parse_usize("metrics_count", field("metrics_count")?)?;
        if metrics_count > MAX_SUMMARY_METRICS {
            return Err(SemanticEvidenceError::TooManyMetrics);
        }

        let fixed_fields = [
            "version",
            "semantic_family",
            "semantic_name",
            "semantic_revision",
            "workload_primary",
            "workload_secondary",
            "workload_length",
            "evidence_kind",
            "producer_repository",
            "producer_revision",
            "artifact_identity",
            "intervention_identity",
            "observation_horizon",
            "metric_identity",
            "sha256_evidence",
            "metrics_count",
        ];
        let mut expected: BTreeSet<String> =
            fixed_fields.into_iter().map(str::to_string).collect();
        for index in 0..metrics_count {
            expected.insert(format!("metric_{index}_name"));
            expected.insert(format!("metric_{index}_bits"));
        }
        let actual: BTreeSet<String> = fields.keys().cloned().collect();
        if actual != expected {
            return Err(SemanticEvidenceError::MalformedCanonicalText(
                "canonical field set is incomplete or contains unknown keys".into(),
            ));
        }

        let family = parse_semantic_family(field("semantic_family")?)?;
        let semantic_name = hex_decode("semantic_name", field("semantic_name")?)?;
        let semantic_revision = parse_u32("semantic_revision", field("semantic_revision")?)?;
        let semantic = SemanticId::new(family, semantic_name, semantic_revision)?;
        let workload = EvidenceWorkloadFingerprint::from_parts(
            parse_fixed_u64_hex("workload_primary", field("workload_primary")?)?,
            parse_fixed_u64_hex("workload_secondary", field("workload_secondary")?)?,
            parse_fixed_u64_hex("workload_length", field("workload_length")?)?,
        );
        let kind = parse_evidence_kind(field("evidence_kind")?)?;
        let producer_repository =
            hex_decode("producer_repository", field("producer_repository")?)?;
        let producer_revision = field("producer_revision")?.to_string();
        let artifact_identity = hex_decode("artifact_identity", field("artifact_identity")?)?;
        let intervention_identity = parse_optional_hex_identifier(
            "intervention_identity",
            field("intervention_identity")?,
        )?;
        let observation_horizon = parse_optional_u32(
            "observation_horizon",
            field("observation_horizon")?,
        )?;
        let metric_identity = hex_decode("metric_identity", field("metric_identity")?)?;
        let sha256_evidence = field("sha256_evidence")?.to_string();
        let mut metrics = Vec::with_capacity(metrics_count);
        for index in 0..metrics_count {
            let name = hex_decode(
                "metric_name",
                field(&format!("metric_{index}_name"))?,
            )?;
            let bits = parse_fixed_u64_hex(
                "metric_bits",
                field(&format!("metric_{index}_bits"))?,
            )?;
            metrics.push((name, f64::from_bits(bits)));
        }

        Self::new(SemanticEvidenceSpec {
            semantic,
            workload,
            kind,
            producer_repository,
            producer_revision,
            artifact_identity,
            intervention_identity,
            observation_horizon,
            metric_identity,
            sha256_evidence,
            metrics,
        })
    }
}

fn append_field(text: &mut String, key: &str, value: &str) {
    text.push_str(key);
    text.push('=');
    text.push_str(value);
    text.push('\n');
}

fn semantic_family_text(family: SemanticFamily) -> &'static str {
    match family {
        SemanticFamily::StandardSoftmax => "standard-softmax",
        SemanticFamily::DifferentialSigned => "differential-signed",
        SemanticFamily::ToeplitzStructured => "toeplitz-structured",
        SemanticFamily::ProlateConcentration => "prolate-concentration",
        SemanticFamily::GroundStateGreen => "ground-state-green",
        SemanticFamily::SpectralFlow => "spectral-flow",
        SemanticFamily::RecurrentMemory => "recurrent-memory",
        SemanticFamily::Hybrid => "hybrid",
        SemanticFamily::Experimental => "experimental",
    }
}

fn parse_semantic_family(value: &str) -> Result<SemanticFamily, SemanticEvidenceError> {
    match value {
        "standard-softmax" => Ok(SemanticFamily::StandardSoftmax),
        "differential-signed" => Ok(SemanticFamily::DifferentialSigned),
        "toeplitz-structured" => Ok(SemanticFamily::ToeplitzStructured),
        "prolate-concentration" => Ok(SemanticFamily::ProlateConcentration),
        "ground-state-green" => Ok(SemanticFamily::GroundStateGreen),
        "spectral-flow" => Ok(SemanticFamily::SpectralFlow),
        "recurrent-memory" => Ok(SemanticFamily::RecurrentMemory),
        "hybrid" => Ok(SemanticFamily::Hybrid),
        "experimental" => Ok(SemanticFamily::Experimental),
        _ => Err(SemanticEvidenceError::MalformedCanonicalText(
            "unknown semantic family".into(),
        )),
    }
}

fn evidence_kind_text(kind: DiagnosticEvidenceKind) -> &'static str {
    match kind {
        DiagnosticEvidenceKind::TaskBehavior => "task-behavior",
        DiagnosticEvidenceKind::StaticOperator => "static-operator",
        DiagnosticEvidenceKind::ItdStructural => "itd-structural",
        DiagnosticEvidenceKind::TdiRecovery => "tdi-recovery",
        DiagnosticEvidenceKind::Adversarial => "adversarial",
        DiagnosticEvidenceKind::LogicalCost => "logical-cost",
        DiagnosticEvidenceKind::HardwareCost => "hardware-cost",
        DiagnosticEvidenceKind::Generalization => "generalization",
        DiagnosticEvidenceKind::PriorArt => "prior-art",
    }
}

fn parse_evidence_kind(value: &str) -> Result<DiagnosticEvidenceKind, SemanticEvidenceError> {
    match value {
        "task-behavior" => Ok(DiagnosticEvidenceKind::TaskBehavior),
        "static-operator" => Ok(DiagnosticEvidenceKind::StaticOperator),
        "itd-structural" => Ok(DiagnosticEvidenceKind::ItdStructural),
        "tdi-recovery" => Ok(DiagnosticEvidenceKind::TdiRecovery),
        "adversarial" => Ok(DiagnosticEvidenceKind::Adversarial),
        "logical-cost" => Ok(DiagnosticEvidenceKind::LogicalCost),
        "hardware-cost" => Ok(DiagnosticEvidenceKind::HardwareCost),
        "generalization" => Ok(DiagnosticEvidenceKind::Generalization),
        "prior-art" => Ok(DiagnosticEvidenceKind::PriorArt),
        _ => Err(SemanticEvidenceError::MalformedCanonicalText(
            "unknown evidence kind".into(),
        )),
    }
}

fn parse_u16(field: &str, value: &str) -> Result<u16, SemanticEvidenceError> {
    value.parse::<u16>().map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not an unsigned 16-bit integer"
        ))
    })
}

fn parse_u32(field: &str, value: &str) -> Result<u32, SemanticEvidenceError> {
    value.parse::<u32>().map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not an unsigned 32-bit integer"
        ))
    })
}

fn parse_usize(field: &str, value: &str) -> Result<usize, SemanticEvidenceError> {
    value.parse::<usize>().map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not an unsigned integer"
        ))
    })
}

fn parse_optional_u32(field: &str, value: &str) -> Result<Option<u32>, SemanticEvidenceError> {
    if value == "-" {
        Ok(None)
    } else {
        parse_u32(field, value).map(Some)
    }
}

fn parse_fixed_u64_hex(field: &str, value: &str) -> Result<u64, SemanticEvidenceError> {
    if !is_lower_hex_exact(value, 16) {
        return Err(SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} must be exactly 16 lowercase hex characters"
        )));
    }
    u64::from_str_radix(value, 16).map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!("{field} is not valid hex"))
    })
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

fn hex_decode(field: &str, value: &str) -> Result<String, SemanticEvidenceError> {
    if value.len() % 2 != 0 {
        return Err(SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} has an odd-length hex value"
        )));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut chars = value.bytes();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = hex_digit(high).ok_or_else(|| {
            SemanticEvidenceError::MalformedCanonicalText(format!(
                "{field} contains a non-hex digit"
            ))
        })?;
        let low = hex_digit(low).ok_or_else(|| {
            SemanticEvidenceError::MalformedCanonicalText(format!(
                "{field} contains a non-hex digit"
            ))
        })?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not UTF-8 after hex decoding"
        ))
    })
}

fn parse_optional_hex_identifier(
    field: &str,
    value: &str,
) -> Result<Option<String>, SemanticEvidenceError> {
    if value == "-" {
        Ok(None)
    } else {
        hex_decode(field, value).map(Some)
    }
}

const fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, InputRepresentation,
        MaskKind, MaskSpec, PrecisionPolicy, ScalarPrecision, SequenceLengths, StateSpec,
        WorkloadMode, WorkloadOptions,
    };

    fn workload(mask: MaskKind) -> WorkloadContract {
        WorkloadContract::new(
            AttentionGeometry::new(GeometrySpec {
                sequence_lengths: SequenceLengths::uniform(1, 3, 3).unwrap(),
                query_heads: 1,
                kv_heads: 1,
                qk_dimension: None,
                value_dimension: 1,
                topology: AttentionTopology::SelfAttention,
                head_grouping: HeadGrouping::MultiHead,
            })
            .unwrap(),
            WorkloadOptions {
                mode: WorkloadMode::Prefill,
                mask: MaskSpec::new(mask).unwrap(),
                precision: PrecisionPolicy::new(
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                    ScalarPrecision::F64,
                ),
                inputs: InputRepresentation::PrecomputedScores {
                    identity: "ada-a11-e1-fixed-mixer".into(),
                },
                state: StateSpec::Stateless,
                ..WorkloadOptions::default()
            },
        )
        .unwrap()
    }

    fn semantic() -> SemanticId {
        SemanticId::new(
            SemanticFamily::Experimental,
            "balanced-three-token-mixer",
            1,
        )
        .unwrap()
    }

    fn golden_tdi() -> SemanticEvidenceRecord {
        SemanticEvidenceRecord::new(SemanticEvidenceSpec {
            semantic: semantic(),
            workload: EvidenceWorkloadFingerprint::from_workload(&workload(
                MaskKind::Bidirectional,
            )),
            kind: DiagnosticEvidenceKind::TdiRecovery,
            producer_repository: "Memorithm/TDI".into(),
            producer_revision: "a".repeat(40),
            artifact_identity: "tdi-ai-gate-b".into(),
            intervention_identity: Some("balanced-antisymmetric-mode".into()),
            observation_horizon: Some(3),
            metric_identity: "reciprocal-linf-recovery".into(),
            sha256_evidence: "b".repeat(64),
            metrics: vec![
                ("recovery_h3".into(), 8.0 / 9.0),
                ("linf_h3".into(), 0.125),
            ],
        })
        .unwrap()
    }

    fn assert_metric_bits(actual: &[(String, f64)], expected: &[(String, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for ((actual_name, actual_value), (expected_name, expected_value)) in
            actual.iter().zip(expected)
        {
            assert_eq!(actual_name, expected_name);
            assert_eq!(actual_value.to_bits(), expected_value.to_bits());
        }
    }

    #[test]
    fn tdi_record_round_trips_canonically_and_preserves_exact_metric_bits() {
        let record = golden_tdi();
        let text = record.to_canonical_text();
        let decoded = SemanticEvidenceRecord::from_canonical_text(&text).unwrap();

        assert_eq!(decoded.to_canonical_text(), text);
        assert_eq!(decoded.semantic(), record.semantic());
        assert_eq!(decoded.workload(), record.workload());
        assert_eq!(decoded.kind(), DiagnosticEvidenceKind::TdiRecovery);
        assert_eq!(decoded.producer_repository(), "Memorithm/TDI");
        assert_eq!(decoded.intervention_identity(), Some("balanced-antisymmetric-mode"));
        assert_eq!(decoded.observation_horizon(), Some(3));
        assert_metric_bits(decoded.metrics(), record.metrics());
    }

    #[test]
    fn workload_binding_changes_when_experimental_visibility_changes() {
        let bidirectional = EvidenceWorkloadFingerprint::from_workload(&workload(
            MaskKind::Bidirectional,
        ));
        let unmasked = EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::None));
        assert_ne!(bidirectional, unmasked);
    }

    #[test]
    fn tdi_recovery_requires_intervention_and_horizon() {
        let mut spec = SemanticEvidenceSpec {
            semantic: semantic(),
            workload: EvidenceWorkloadFingerprint::from_workload(&workload(
                MaskKind::Bidirectional,
            )),
            kind: DiagnosticEvidenceKind::TdiRecovery,
            producer_repository: "Memorithm/TDI".into(),
            producer_revision: "a".repeat(40),
            artifact_identity: "gate-b".into(),
            intervention_identity: None,
            observation_horizon: Some(3),
            metric_identity: "recovery".into(),
            sha256_evidence: "b".repeat(64),
            metrics: vec![],
        };
        assert_eq!(
            SemanticEvidenceRecord::new(spec.clone()),
            Err(SemanticEvidenceError::MissingIntervention)
        );
        spec.intervention_identity = Some("balanced-mode".into());
        spec.observation_horizon = None;
        assert_eq!(
            SemanticEvidenceRecord::new(spec),
            Err(SemanticEvidenceError::MissingObservationHorizon)
        );
    }

    #[test]
    fn structural_itd_evidence_can_be_non_interventional() {
        let record = SemanticEvidenceRecord::new(SemanticEvidenceSpec {
            semantic: semantic(),
            workload: EvidenceWorkloadFingerprint::from_workload(&workload(
                MaskKind::Bidirectional,
            )),
            kind: DiagnosticEvidenceKind::ItdStructural,
            producer_repository: "Memorithm/itd-simulator".into(),
            producer_revision: "c".repeat(40),
            artifact_identity: "attention-structure-v1".into(),
            intervention_identity: None,
            observation_horizon: None,
            metric_identity: "structural-concentration".into(),
            sha256_evidence: "d".repeat(64),
            metrics: vec![("descriptor".into(), 0.25)],
        })
        .unwrap();
        assert_eq!(record.kind(), DiagnosticEvidenceKind::ItdStructural);
        assert_eq!(record.intervention_identity(), None);
        assert_eq!(record.observation_horizon(), None);
    }

    #[test]
    fn metric_order_is_canonical_and_duplicate_names_fail_closed() {
        let record = golden_tdi();
        assert_eq!(record.metrics()[0].0, "linf_h3");
        assert_eq!(record.metrics()[1].0, "recovery_h3");

        let mut duplicate = record.metrics().to_vec();
        duplicate.push(("linf_h3".into(), 1.0));
        let result = SemanticEvidenceRecord::new(SemanticEvidenceSpec {
            semantic: record.semantic().clone(),
            workload: record.workload(),
            kind: record.kind(),
            producer_repository: record.producer_repository().into(),
            producer_revision: record.producer_revision().into(),
            artifact_identity: record.artifact_identity().into(),
            intervention_identity: record.intervention_identity().map(str::to_string),
            observation_horizon: record.observation_horizon(),
            metric_identity: record.metric_identity().into(),
            sha256_evidence: record.sha256_evidence().into(),
            metrics: duplicate,
        });
        assert!(matches!(result, Err(SemanticEvidenceError::DuplicateMetric)));
    }

    #[test]
    fn non_finite_summary_and_invalid_digests_are_rejected() {
        let record = golden_tdi();
        let bad_metric = SemanticEvidenceRecord::new(SemanticEvidenceSpec {
            semantic: record.semantic().clone(),
            workload: record.workload(),
            kind: record.kind(),
            producer_repository: record.producer_repository().into(),
            producer_revision: record.producer_revision().into(),
            artifact_identity: record.artifact_identity().into(),
            intervention_identity: record.intervention_identity().map(str::to_string),
            observation_horizon: record.observation_horizon(),
            metric_identity: record.metric_identity().into(),
            sha256_evidence: record.sha256_evidence().into(),
            metrics: vec![("bad".into(), f64::NAN)],
        });
        assert!(matches!(bad_metric, Err(SemanticEvidenceError::NonFiniteMetric)));

        let mut text = record.to_canonical_text();
        text = text.replace(&"b".repeat(64), "deadbeef");
        assert!(SemanticEvidenceRecord::from_canonical_text(&text).is_err());
    }

    #[test]
    fn unknown_or_duplicate_canonical_fields_fail_closed() {
        let record = golden_tdi();
        let mut unknown = record.to_canonical_text();
        unknown.push_str("unknown=field\n");
        assert!(SemanticEvidenceRecord::from_canonical_text(&unknown).is_err());

        let mut duplicate = record.to_canonical_text();
        duplicate.push_str("version=1\n");
        assert!(SemanticEvidenceRecord::from_canonical_text(&duplicate).is_err());
    }

    #[test]
    fn diagnostic_reference_keeps_evidence_out_of_semantic_identity() {
        let record = golden_tdi();
        let reference = record.diagnostic_reference().unwrap();
        assert_eq!(reference.kind(), DiagnosticEvidenceKind::TdiRecovery);
        assert_eq!(reference.repository(), "Memorithm/TDI");
        assert_eq!(reference.artifact(), "tdi-ai-gate-b");
        assert_eq!(record.semantic().name(), "balanced-three-token-mixer");
        assert!(reference.revision_binding().contains("git:"));
        assert!(reference.revision_binding().contains(";sha256:"));
    }
}
