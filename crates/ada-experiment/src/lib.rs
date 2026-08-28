//! Versioned, reproducible experiment records and deterministic indexing for ADA.
//!
//! `ADA-EXPERIMENT-V2` binds an exact semantic program, workload, implementation
//! plan, objective vector, producer provenance, and explicit evidence references.
//! Estimated objectives remain distinct from measured objectives. Measured
//! latency/energy are rejected unless hardware evidence is attached.
//!
//! `ADA-EXPERIMENT-INDEX-V1` is a bounded, deterministic archive of complete
//! experiment records. It is an interchange/index layer, not a database and not
//! a source of scientific truth beyond the evidence it binds.

#![forbid(unsafe_code)]

use ada_core::{
    DiagnosticEvidenceKind, DiagnosticEvidenceRef, ImplementationCandidateId, SemanticId,
};
use ada_implementation::ImplementationPlan;
use ada_objective::ObjectiveVector;
use ada_semantic::SemanticProgram;
use ada_workload::{WorkloadContract, WorkloadFingerprint};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

pub const EXPERIMENT_VERSION: u16 = 2;
pub const EXPERIMENT_HEADER: &str = "ADA-EXPERIMENT-V2";
pub const EXPERIMENT_INDEX_VERSION: u16 = 1;
pub const EXPERIMENT_INDEX_HEADER: &str = "ADA-EXPERIMENT-INDEX-V1";
pub const MAX_EXPERIMENT_BYTES: usize = 16 << 20;
pub const MAX_INDEX_BYTES: usize = 64 << 20;
pub const MAX_EVIDENCE_BINDINGS: usize = 1_024;
pub const MAX_INDEX_ENTRIES: usize = 65_536;
pub const MAX_IDENTIFIER_BYTES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    InvalidField(&'static str),
    InvalidProducerRevision,
    InvalidArtifactDigest,
    Semantic(String),
    Workload(String),
    Implementation(String),
    Objective(String),
    SemanticImplementationMismatch,
    MissingHardwareEvidence,
    MissingQualityEvidence,
    TooManyEvidenceBindings,
    DuplicateEvidenceBinding,
    UnsupportedVersion(u16),
    MalformedCanonical(String),
    IndexFull,
    DuplicateExperiment,
    FingerprintCollision,
}

impl Display for ExperimentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField(field) => write!(f, "invalid experiment field: {field}"),
            Self::InvalidProducerRevision => f.write_str("producer revision must be 40 lowercase hex"),
            Self::InvalidArtifactDigest => f.write_str("artifact digest must be 64 lowercase hex"),
            Self::Semantic(reason) => write!(f, "semantic artifact error: {reason}"),
            Self::Workload(reason) => write!(f, "workload artifact error: {reason}"),
            Self::Implementation(reason) => write!(f, "implementation artifact error: {reason}"),
            Self::Objective(reason) => write!(f, "objective artifact error: {reason}"),
            Self::SemanticImplementationMismatch => {
                f.write_str("implementation identity is bound to another semantic")
            }
            Self::MissingHardwareEvidence => {
                f.write_str("measured latency/energy requires HardwareCost evidence")
            }
            Self::MissingQualityEvidence => {
                f.write_str("observed quality requires TaskBehavior or Generalization evidence")
            }
            Self::TooManyEvidenceBindings => f.write_str("too many experiment evidence bindings"),
            Self::DuplicateEvidenceBinding => f.write_str("duplicate experiment evidence binding"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported experiment version {version}"),
            Self::MalformedCanonical(reason) => write!(f, "malformed experiment artifact: {reason}"),
            Self::IndexFull => f.write_str("experiment index capacity exhausted"),
            Self::DuplicateExperiment => f.write_str("experiment already exists in index"),
            Self::FingerprintCollision => f.write_str("experiment fingerprint collision"),
        }
    }
}

impl std::error::Error for ExperimentError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerProvenance {
    repository: String,
    git_revision: String,
    artifact_identity: String,
    artifact_sha256: String,
}

impl ProducerProvenance {
    pub fn new(
        repository: impl Into<String>,
        git_revision: impl Into<String>,
        artifact_identity: impl Into<String>,
        artifact_sha256: impl Into<String>,
    ) -> Result<Self, ExperimentError> {
        let value = Self {
            repository: repository.into(),
            git_revision: git_revision.into(),
            artifact_identity: artifact_identity.into(),
            artifact_sha256: artifact_sha256.into(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), ExperimentError> {
        if !valid_repository(&self.repository) {
            return Err(ExperimentError::InvalidField("provenance.repository"));
        }
        if !is_lower_hex_exact(&self.git_revision, 40) {
            return Err(ExperimentError::InvalidProducerRevision);
        }
        validate_identifier("provenance.artifact_identity", &self.artifact_identity)?;
        if !is_lower_hex_exact(&self.artifact_sha256, 64) {
            return Err(ExperimentError::InvalidArtifactDigest);
        }
        Ok(())
    }

    #[must_use]
    pub fn repository(&self) -> &str { &self.repository }
    #[must_use]
    pub fn git_revision(&self) -> &str { &self.git_revision }
    #[must_use]
    pub fn artifact_identity(&self) -> &str { &self.artifact_identity }
    #[must_use]
    pub fn artifact_sha256(&self) -> &str { &self.artifact_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceBinding {
    kind: EvidenceKind,
    repository: String,
    artifact: String,
    revision_binding: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EvidenceKind {
    TaskBehavior,
    StaticOperator,
    ItdStructural,
    TdiRecovery,
    Adversarial,
    LogicalCost,
    HardwareCost,
    Generalization,
    PriorArt,
}

impl EvidenceBinding {
    pub fn from_reference(reference: &DiagnosticEvidenceRef) -> Result<Self, ExperimentError> {
        let value = Self {
            kind: EvidenceKind::from_diagnostic(reference.kind()),
            repository: reference.repository().to_string(),
            artifact: reference.artifact().to_string(),
            revision_binding: reference.revision_binding().to_string(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), ExperimentError> {
        if !valid_repository(&self.repository) {
            return Err(ExperimentError::InvalidField("evidence.repository"));
        }
        validate_identifier("evidence.artifact", &self.artifact)?;
        validate_identifier("evidence.revision_binding", &self.revision_binding)
    }

    #[must_use]
    pub fn kind(&self) -> DiagnosticEvidenceKind { self.kind.to_diagnostic() }
    #[must_use]
    pub fn repository(&self) -> &str { &self.repository }
    #[must_use]
    pub fn artifact(&self) -> &str { &self.artifact }
    #[must_use]
    pub fn revision_binding(&self) -> &str { &self.revision_binding }
}

impl EvidenceKind {
    const fn from_diagnostic(value: DiagnosticEvidenceKind) -> Self {
        match value {
            DiagnosticEvidenceKind::TaskBehavior => Self::TaskBehavior,
            DiagnosticEvidenceKind::StaticOperator => Self::StaticOperator,
            DiagnosticEvidenceKind::ItdStructural => Self::ItdStructural,
            DiagnosticEvidenceKind::TdiRecovery => Self::TdiRecovery,
            DiagnosticEvidenceKind::Adversarial => Self::Adversarial,
            DiagnosticEvidenceKind::LogicalCost => Self::LogicalCost,
            DiagnosticEvidenceKind::HardwareCost => Self::HardwareCost,
            DiagnosticEvidenceKind::Generalization => Self::Generalization,
            DiagnosticEvidenceKind::PriorArt => Self::PriorArt,
        }
    }

    const fn to_diagnostic(self) -> DiagnosticEvidenceKind {
        match self {
            Self::TaskBehavior => DiagnosticEvidenceKind::TaskBehavior,
            Self::StaticOperator => DiagnosticEvidenceKind::StaticOperator,
            Self::ItdStructural => DiagnosticEvidenceKind::ItdStructural,
            Self::TdiRecovery => DiagnosticEvidenceKind::TdiRecovery,
            Self::Adversarial => DiagnosticEvidenceKind::Adversarial,
            Self::LogicalCost => DiagnosticEvidenceKind::LogicalCost,
            Self::HardwareCost => DiagnosticEvidenceKind::HardwareCost,
            Self::Generalization => DiagnosticEvidenceKind::Generalization,
            Self::PriorArt => DiagnosticEvidenceKind::PriorArt,
        }
    }

    const fn as_text(self) -> &'static str {
        match self {
            Self::TaskBehavior => "task-behavior",
            Self::StaticOperator => "static-operator",
            Self::ItdStructural => "itd-structural",
            Self::TdiRecovery => "tdi-recovery",
            Self::Adversarial => "adversarial",
            Self::LogicalCost => "logical-cost",
            Self::HardwareCost => "hardware-cost",
            Self::Generalization => "generalization",
            Self::PriorArt => "prior-art",
        }
    }

    fn parse(value: &str) -> Result<Self, ExperimentError> {
        match value {
            "task-behavior" => Ok(Self::TaskBehavior),
            "static-operator" => Ok(Self::StaticOperator),
            "itd-structural" => Ok(Self::ItdStructural),
            "tdi-recovery" => Ok(Self::TdiRecovery),
            "adversarial" => Ok(Self::Adversarial),
            "logical-cost" => Ok(Self::LogicalCost),
            "hardware-cost" => Ok(Self::HardwareCost),
            "generalization" => Ok(Self::Generalization),
            "prior-art" => Ok(Self::PriorArt),
            _ => malformed("unknown evidence kind"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSpec {
    pub semantic: SemanticProgram,
    pub workload: WorkloadContract,
    pub implementation: ImplementationPlan,
    pub objective: ObjectiveVector,
    pub provenance: ProducerProvenance,
    pub evidence: Vec<EvidenceBinding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentRecord {
    semantic: SemanticProgram,
    workload: WorkloadContract,
    implementation: ImplementationPlan,
    objective: ObjectiveVector,
    provenance: ProducerProvenance,
    evidence: Vec<EvidenceBinding>,
}

impl ExperimentRecord {
    pub fn new(mut spec: ExperimentSpec) -> Result<Self, ExperimentError> {
        spec.workload.validate().map_err(|e| ExperimentError::Workload(e.to_string()))?;
        spec.objective.validate().map_err(|e| ExperimentError::Objective(e.to_string()))?;
        spec.provenance.validate()?;
        if spec.implementation.id().semantic() != spec.semantic.descriptor().id() {
            return Err(ExperimentError::SemanticImplementationMismatch);
        }
        if spec.evidence.len() > MAX_EVIDENCE_BINDINGS {
            return Err(ExperimentError::TooManyEvidenceBindings);
        }
        for binding in &spec.evidence { binding.validate()?; }
        spec.evidence.sort();
        if spec.evidence.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ExperimentError::DuplicateEvidenceBinding);
        }
        let measured = spec.objective.measured();
        if (measured.latency_ns.is_some() || measured.energy_nj.is_some())
            && !spec.evidence.iter().any(|e| e.kind == EvidenceKind::HardwareCost)
        {
            return Err(ExperimentError::MissingHardwareEvidence);
        }
        let observed_quality = spec.objective.quality().iter().any(|metric| metric.value().is_some());
        if observed_quality
            && !spec.evidence.iter().any(|e| {
                matches!(e.kind, EvidenceKind::TaskBehavior | EvidenceKind::Generalization)
            })
        {
            return Err(ExperimentError::MissingQualityEvidence);
        }
        Ok(Self {
            semantic: spec.semantic,
            workload: spec.workload,
            implementation: spec.implementation,
            objective: spec.objective,
            provenance: spec.provenance,
            evidence: spec.evidence,
        })
    }

    #[must_use]
    pub const fn semantic(&self) -> &SemanticProgram { &self.semantic }
    #[must_use]
    pub const fn workload(&self) -> &WorkloadContract { &self.workload }
    #[must_use]
    pub const fn implementation(&self) -> &ImplementationPlan { &self.implementation }
    #[must_use]
    pub const fn objective(&self) -> &ObjectiveVector { &self.objective }
    #[must_use]
    pub const fn provenance(&self) -> &ProducerProvenance { &self.provenance }
    #[must_use]
    pub fn evidence(&self) -> &[EvidenceBinding] { &self.evidence }

    #[must_use]
    pub fn fingerprint(&self) -> ExperimentFingerprint {
        ExperimentFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }

    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(EXPERIMENT_HEADER);
        text.push('\n');
        field(&mut text, "version", EXPERIMENT_VERSION);
        field(&mut text, "producer_repository", hex_encode(self.provenance.repository()));
        field(&mut text, "producer_revision", self.provenance.git_revision());
        field(&mut text, "artifact_identity", hex_encode(self.provenance.artifact_identity()));
        field(&mut text, "artifact_sha256", self.provenance.artifact_sha256());
        field(&mut text, "semantic_text", hex_encode(&self.semantic.to_canonical_text()));
        field(&mut text, "workload_text", hex_encode(&self.workload.to_canonical_text()));
        field(&mut text, "implementation_text", hex_encode(&self.implementation.to_canonical_text()));
        field(&mut text, "objective_text", hex_encode(&self.objective.to_canonical_text()));
        field(&mut text, "evidence_count", self.evidence.len());
        for (index, evidence) in self.evidence.iter().enumerate() {
            field(&mut text, &format!("evidence_{index}_kind"), evidence.kind.as_text());
            field(&mut text, &format!("evidence_{index}_repository"), hex_encode(evidence.repository()));
            field(&mut text, &format!("evidence_{index}_artifact"), hex_encode(evidence.artifact()));
            field(&mut text, &format!("evidence_{index}_revision"), hex_encode(evidence.revision_binding()));
        }
        text
    }

    pub fn from_canonical_text(text: &str) -> Result<Self, ExperimentError> {
        if text.len() > MAX_EXPERIMENT_BYTES || text.contains('\r') || !text.ends_with('\n') {
            return malformed("artifact exceeds limit, contains CR, or lacks final newline");
        }
        let mut lines = text.lines();
        if lines.next() != Some(EXPERIMENT_HEADER) { return malformed("invalid experiment header"); }
        let version = parse_u16(next(&mut lines, "version")?)?;
        if version != EXPERIMENT_VERSION { return Err(ExperimentError::UnsupportedVersion(version)); }
        let repository = hex_decode(next(&mut lines, "producer_repository")?)?;
        let revision = next(&mut lines, "producer_revision")?.to_string();
        let artifact_identity = hex_decode(next(&mut lines, "artifact_identity")?)?;
        let artifact_sha256 = next(&mut lines, "artifact_sha256")?.to_string();
        let semantic_text = hex_decode(next(&mut lines, "semantic_text")?)?;
        let workload_text = hex_decode(next(&mut lines, "workload_text")?)?;
        let implementation_text = hex_decode(next(&mut lines, "implementation_text")?)?;
        let objective_text = hex_decode(next(&mut lines, "objective_text")?)?;
        let evidence_count = parse_usize(next(&mut lines, "evidence_count")?)?;
        if evidence_count > MAX_EVIDENCE_BINDINGS { return Err(ExperimentError::TooManyEvidenceBindings); }
        let mut evidence = Vec::with_capacity(evidence_count);
        for index in 0..evidence_count {
            let kind = EvidenceKind::parse(next(&mut lines, &format!("evidence_{index}_kind"))?)?;
            let repository = hex_decode(next(&mut lines, &format!("evidence_{index}_repository"))?)?;
            let artifact = hex_decode(next(&mut lines, &format!("evidence_{index}_artifact"))?)?;
            let revision_binding = hex_decode(next(&mut lines, &format!("evidence_{index}_revision"))?)?;
            evidence.push(EvidenceBinding { kind, repository, artifact, revision_binding });
        }
        if lines.next().is_some() { return malformed("unexpected trailing experiment field"); }
        let record = Self::new(ExperimentSpec {
            semantic: SemanticProgram::from_canonical_text(&semantic_text)
                .map_err(|e| ExperimentError::Semantic(e.to_string()))?,
            workload: WorkloadContract::from_canonical_text(&workload_text)
                .map_err(|e| ExperimentError::Workload(e.to_string()))?,
            implementation: ImplementationPlan::from_canonical_text(&implementation_text)
                .map_err(|e| ExperimentError::Implementation(e.to_string()))?,
            objective: ObjectiveVector::from_canonical_text(&objective_text)
                .map_err(|e| ExperimentError::Objective(e.to_string()))?,
            provenance: ProducerProvenance::new(repository, revision, artifact_identity, artifact_sha256)?,
            evidence,
        })?;
        if record.to_canonical_text() != text { return malformed("experiment artifact is not canonical"); }
        Ok(record)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExperimentFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl ExperimentFingerprint {
    fn of_bytes(bytes: &[u8]) -> Self {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        const MIX: u64 = 0xff51_afd7_ed55_8ccd;
        let mut primary = OFFSET;
        let mut secondary = OFFSET;
        for &byte in bytes {
            primary ^= u64::from(byte);
            primary = primary.wrapping_mul(PRIME);
            secondary ^= u64::from(byte);
            secondary = secondary.rotate_left(27).wrapping_mul(MIX);
        }
        let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        Self { primary: primary ^ length, secondary: secondary.rotate_left(31) ^ length, length }
    }
    #[must_use] pub const fn primary(self) -> u64 { self.primary }
    #[must_use] pub const fn secondary(self) -> u64 { self.secondary }
    #[must_use] pub const fn length(self) -> u64 { self.length }
}

impl Display for ExperimentFingerprint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}-{:016x}-{:016x}", self.primary, self.secondary, self.length)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExperimentIndex {
    records: BTreeMap<ExperimentFingerprint, ExperimentRecord>,
}

impl ExperimentIndex {
    #[must_use]
    pub fn new() -> Self { Self::default() }

    pub fn insert(&mut self, record: ExperimentRecord) -> Result<ExperimentFingerprint, ExperimentError> {
        if self.records.len() >= MAX_INDEX_ENTRIES { return Err(ExperimentError::IndexFull); }
        let fingerprint = record.fingerprint();
        if let Some(existing) = self.records.get(&fingerprint) {
            if existing.to_canonical_text() == record.to_canonical_text() {
                return Err(ExperimentError::DuplicateExperiment);
            }
            return Err(ExperimentError::FingerprintCollision);
        }
        self.records.insert(fingerprint, record);
        Ok(fingerprint)
    }

    #[must_use] pub fn len(&self) -> usize { self.records.len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.records.is_empty() }
    #[must_use] pub fn get(&self, fingerprint: ExperimentFingerprint) -> Option<&ExperimentRecord> { self.records.get(&fingerprint) }

    #[must_use]
    pub fn records_for_semantic(&self, semantic: &SemanticId) -> Vec<&ExperimentRecord> {
        self.records.values().filter(|record| record.semantic.descriptor().id() == semantic).collect()
    }

    #[must_use]
    pub fn records_for_workload(&self, workload: WorkloadFingerprint) -> Vec<&ExperimentRecord> {
        self.records.values().filter(|record| record.workload.fingerprint() == workload).collect()
    }

    #[must_use]
    pub fn records_for_implementation(&self, implementation: &ImplementationCandidateId) -> Vec<&ExperimentRecord> {
        self.records.values().filter(|record| record.implementation.id() == implementation).collect()
    }

    #[must_use]
    pub fn records_with_measured_cost(&self) -> Vec<&ExperimentRecord> {
        self.records.values().filter(|record| {
            let measured = record.objective.measured();
            measured.latency_ns.is_some() || measured.energy_nj.is_some()
        }).collect()
    }

    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(EXPERIMENT_INDEX_HEADER);
        text.push('\n');
        field(&mut text, "version", EXPERIMENT_INDEX_VERSION);
        field(&mut text, "count", self.records.len());
        for (index, record) in self.records.values().enumerate() {
            field(&mut text, &format!("entry_{index}"), hex_encode(&record.to_canonical_text()));
        }
        text
    }

    pub fn from_canonical_text(text: &str) -> Result<Self, ExperimentError> {
        if text.len() > MAX_INDEX_BYTES || text.contains('\r') || !text.ends_with('\n') {
            return malformed("index exceeds limit, contains CR, or lacks final newline");
        }
        let mut lines = text.lines();
        if lines.next() != Some(EXPERIMENT_INDEX_HEADER) { return malformed("invalid index header"); }
        let version = parse_u16(next(&mut lines, "version")?)?;
        if version != EXPERIMENT_INDEX_VERSION { return Err(ExperimentError::UnsupportedVersion(version)); }
        let count = parse_usize(next(&mut lines, "count")?)?;
        if count > MAX_INDEX_ENTRIES { return Err(ExperimentError::IndexFull); }
        let mut index = Self::new();
        for position in 0..count {
            let encoded = next(&mut lines, &format!("entry_{position}"))?;
            let record = ExperimentRecord::from_canonical_text(&hex_decode(encoded)?)?;
            index.insert(record)?;
        }
        if lines.next().is_some() { return malformed("unexpected trailing index field"); }
        if index.to_canonical_text() != text { return malformed("index is not canonical"); }
        Ok(index)
    }
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), ExperimentError> {
    if value.is_empty() || value.len() > MAX_IDENTIFIER_BYTES || value.chars().any(char::is_control) || value.chars().any(char::is_whitespace) {
        return Err(ExperimentError::InvalidField(field));
    }
    Ok(())
}

fn valid_repository(value: &str) -> bool {
    let mut parts = value.split('/');
    let (Some(owner), Some(repo)) = (parts.next(), parts.next()) else { return false; };
    !owner.is_empty() && !repo.is_empty() && parts.next().is_none()
        && owner.bytes().chain(repo.bytes()).all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn is_lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn field(text: &mut String, key: &str, value: impl Display) {
    text.push_str(key);
    text.push('=');
    text.push_str(&value.to_string());
    text.push('\n');
}

fn next<'a>(lines: &mut std::str::Lines<'a>, key: &str) -> Result<&'a str, ExperimentError> {
    let line = lines.next().ok_or_else(|| ExperimentError::MalformedCanonical(format!("missing field {key}")))?;
    let prefix = format!("{key}=");
    line.strip_prefix(&prefix).ok_or_else(|| ExperimentError::MalformedCanonical(format!("expected field {key}")))
}

fn parse_u16(value: &str) -> Result<u16, ExperimentError> { value.parse().map_err(|_| ExperimentError::MalformedCanonical("invalid u16".into())) }
fn parse_usize(value: &str) -> Result<usize, ExperimentError> { value.parse().map_err(|_| ExperimentError::MalformedCanonical("invalid usize".into())) }
fn malformed<T>(reason: &str) -> Result<T, ExperimentError> { Err(ExperimentError::MalformedCanonical(reason.into())) }

fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value.bytes() {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn hex_decode(value: &str) -> Result<String, ExperimentError> {
    if value.len() % 2 != 0 { return malformed("odd-length hex field"); }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = nibble(pair[0])?;
        let low = nibble(pair[1])?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).map_err(|_| ExperimentError::MalformedCanonical("hex field is not UTF-8".into()))
}

fn nibble(byte: u8) -> Result<u8, ExperimentError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(ExperimentError::MalformedCanonical("non-canonical hex".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_core::{DiagnosticEvidenceKind, ImplementationCandidateId, SemanticFamily, SemanticId};
    use ada_implementation::{
        AlgorithmPlan, Buffering, ExpStrategy, MemoryLevel, MemoryPlan, ReductionTopology,
        SchedulePlan, TileShape, WorkPartition,
    };
    use ada_objective::{
        CorrectnessStatus, EstimatedCost, LogicalCost, MeasuredCost, NumericalObjectives,
        ObjectiveDirection, QualityMetric,
    };
    use ada_semantic::{MaskRule, SelectionRule};
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, PrecisionPolicy,
        ScalarPrecision, SequenceLengths, WorkloadOptions,
    };

    fn components() -> (SemanticProgram, WorkloadContract, ImplementationPlan) {
        let semantic_id = SemanticId::new(SemanticFamily::StandardSoftmax, "experiment-softmax", 1).unwrap();
        let semantic = SemanticProgram::standard_softmax(
            semantic_id.clone(), MaskRule::Unmasked, SelectionRule::All, 1.0,
        ).unwrap();
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 2, 2).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(4),
            value_dimension: 4,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        }).unwrap();
        let workload = WorkloadContract::new(geometry, WorkloadOptions {
            precision: PrecisionPolicy::new(ScalarPrecision::F64, ScalarPrecision::F64, ScalarPrecision::F64, ScalarPrecision::F64),
            ..WorkloadOptions::default()
        }).unwrap();
        let implementation = ImplementationPlan::new(
            ImplementationCandidateId::new(semantic_id, "blocked", 1).unwrap(),
            AlgorithmPlan::DenseBlocked,
            SchedulePlan {
                tile: TileShape { queries: 2, keys: 2, values: 4 },
                partition: WorkPartition::QueryTiles,
                reduction: ReductionTopology::Tree,
                exp_strategy: ExpStrategy::Standard,
                pipeline_stages: 2,
                vector_width: 4,
                buffering: Buffering::Double,
            },
            MemoryPlan {
                query: MemoryLevel::Shared,
                key: MemoryLevel::Shared,
                value: MemoryLevel::Shared,
                output: MemoryLevel::Global,
                accumulator: MemoryLevel::Register,
                workspace_bytes: 0,
                alignment_bytes: 16,
                kv_page_rows: None,
            },
        ).unwrap();
        (semantic, workload, implementation)
    }

    fn provenance() -> ProducerProvenance {
        ProducerProvenance::new(
            "Memorithm/ADA",
            "0123456789abcdef0123456789abcdef01234567",
            "unit-fixture",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        ).unwrap()
    }

    fn evidence(kind: DiagnosticEvidenceKind, artifact: &str) -> EvidenceBinding {
        let reference = DiagnosticEvidenceRef::new(kind, "Memorithm/ADA", artifact, "git:0123456789abcdef").unwrap();
        EvidenceBinding::from_reference(&reference).unwrap()
    }

    #[test]
    fn experiment_round_trip_binds_all_identity_layers() {
        let (semantic, workload, implementation) = components();
        let objective = ObjectiveVector::from_parts(
            CorrectnessStatus::Provisional,
            NumericalObjectives { max_abs_error: Some(0.0), ..NumericalObjectives::default() },
            LogicalCost { flops: Some(128), ..LogicalCost::default() },
            EstimatedCost { bytes_moved: Some(1024), ..EstimatedCost::default() },
            MeasuredCost::default(),
            Vec::new(),
        ).unwrap();
        let record = ExperimentRecord::new(ExperimentSpec {
            semantic,
            workload,
            implementation,
            objective,
            provenance: provenance(),
            evidence: vec![evidence(DiagnosticEvidenceKind::LogicalCost, "logical-cost")],
        }).unwrap();
        let text = record.to_canonical_text();
        let decoded = ExperimentRecord::from_canonical_text(&text).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(decoded.fingerprint(), record.fingerprint());
    }

    #[test]
    fn measured_cost_fails_closed_without_hardware_evidence() {
        let (semantic, workload, implementation) = components();
        let objective = ObjectiveVector::new(CorrectnessStatus::Provisional)
            .with_measured(MeasuredCost { latency_ns: Some(42), energy_nj: None }).unwrap();
        let result = ExperimentRecord::new(ExperimentSpec {
            semantic,
            workload,
            implementation,
            objective,
            provenance: provenance(),
            evidence: Vec::new(),
        });
        assert_eq!(result, Err(ExperimentError::MissingHardwareEvidence));
    }

    #[test]
    fn observed_quality_requires_behavioral_provenance() {
        let (semantic, workload, implementation) = components();
        let quality = QualityMetric::new("accuracy", Some(0.9), ObjectiveDirection::Maximize).unwrap();
        let objective = ObjectiveVector::new(CorrectnessStatus::Provisional).with_quality(vec![quality]).unwrap();
        let result = ExperimentRecord::new(ExperimentSpec {
            semantic,
            workload,
            implementation,
            objective,
            provenance: provenance(),
            evidence: vec![evidence(DiagnosticEvidenceKind::LogicalCost, "logical")],
        });
        assert_eq!(result, Err(ExperimentError::MissingQualityEvidence));
    }

    #[test]
    fn index_is_deterministic_queryable_and_round_trips() {
        let (semantic, workload, implementation) = components();
        let semantic_id = semantic.descriptor().id().clone();
        let workload_fp = workload.fingerprint();
        let implementation_id = implementation.id().clone();
        let objective = ObjectiveVector::new(CorrectnessStatus::Provisional)
            .with_measured(MeasuredCost { latency_ns: Some(42), energy_nj: Some(7) }).unwrap();
        let record = ExperimentRecord::new(ExperimentSpec {
            semantic,
            workload,
            implementation,
            objective,
            provenance: provenance(),
            evidence: vec![evidence(DiagnosticEvidenceKind::HardwareCost, "hardware")],
        }).unwrap();
        let mut index = ExperimentIndex::new();
        let fingerprint = index.insert(record).unwrap();
        assert_eq!(index.records_for_semantic(&semantic_id).len(), 1);
        assert_eq!(index.records_for_workload(workload_fp).len(), 1);
        assert_eq!(index.records_for_implementation(&implementation_id).len(), 1);
        assert_eq!(index.records_with_measured_cost().len(), 1);
        assert!(index.get(fingerprint).is_some());
        let text = index.to_canonical_text();
        let decoded = ExperimentIndex::from_canonical_text(&text).unwrap();
        assert_eq!(decoded.to_canonical_text(), text);
    }
}
