use ada_core::{DiagnosticEvidenceKind, DiagnosticEvidenceRef};
use ada_implementation::ImplementationPlan;
use ada_objective::ObjectiveVector;
use ada_semantic::SemanticProgram;
use ada_workload::WorkloadContract;
use std::fmt::{Display, Formatter};

pub const EXPERIMENT_VERSION: u16 = 2;
pub const EXPERIMENT_HEADER: &str = "ADA-EXPERIMENT-V2";
pub const MAX_EXPERIMENT_BYTES: usize = 16 << 20;
pub const MAX_EVIDENCE_BINDINGS: usize = 1_024;
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
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidField(field) => write!(formatter, "invalid experiment field: {field}"),
            Self::InvalidProducerRevision => {
                formatter.write_str("producer revision must be 40 lowercase hex")
            }
            Self::InvalidArtifactDigest => {
                formatter.write_str("artifact digest must be 64 lowercase hex")
            }
            Self::Semantic(reason) => write!(formatter, "semantic artifact error: {reason}"),
            Self::Workload(reason) => write!(formatter, "workload artifact error: {reason}"),
            Self::Implementation(reason) => {
                write!(formatter, "implementation artifact error: {reason}")
            }
            Self::Objective(reason) => write!(formatter, "objective artifact error: {reason}"),
            Self::SemanticImplementationMismatch => {
                formatter.write_str("implementation identity is bound to another semantic")
            }
            Self::MissingHardwareEvidence => {
                formatter.write_str("measured latency/energy requires HardwareCost evidence")
            }
            Self::MissingQualityEvidence => formatter
                .write_str("observed quality requires TaskBehavior or Generalization evidence"),
            Self::TooManyEvidenceBindings => {
                formatter.write_str("too many experiment evidence bindings")
            }
            Self::DuplicateEvidenceBinding => {
                formatter.write_str("duplicate experiment evidence binding")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported experiment version {version}")
            }
            Self::MalformedCanonical(reason) => {
                write!(formatter, "malformed experiment artifact: {reason}")
            }
            Self::IndexFull => formatter.write_str("experiment index capacity exhausted"),
            Self::DuplicateExperiment => formatter.write_str("experiment already exists in index"),
            Self::FingerprintCollision => formatter.write_str("experiment fingerprint collision"),
        }
    }
}

impl std::error::Error for ExperimentError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerProvenance {
    pub(crate) repository: String,
    pub(crate) git_revision: String,
    pub(crate) artifact_identity: String,
    pub(crate) artifact_sha256: String,
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

    pub(crate) fn validate(&self) -> Result<(), ExperimentError> {
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
    pub fn repository(&self) -> &str {
        &self.repository
    }

    #[must_use]
    pub fn git_revision(&self) -> &str {
        &self.git_revision
    }

    #[must_use]
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }

    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceBinding {
    pub(crate) kind: EvidenceKind,
    pub(crate) repository: String,
    pub(crate) artifact: String,
    pub(crate) revision_binding: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EvidenceKind {
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

    pub(crate) fn validate(&self) -> Result<(), ExperimentError> {
        if !valid_repository(&self.repository) {
            return Err(ExperimentError::InvalidField("evidence.repository"));
        }
        validate_identifier("evidence.artifact", &self.artifact)?;
        validate_identifier("evidence.revision_binding", &self.revision_binding)
    }

    #[must_use]
    pub fn kind(&self) -> DiagnosticEvidenceKind {
        self.kind.to_diagnostic()
    }

    #[must_use]
    pub fn repository(&self) -> &str {
        &self.repository
    }

    #[must_use]
    pub fn artifact(&self) -> &str {
        &self.artifact
    }

    #[must_use]
    pub fn revision_binding(&self) -> &str {
        &self.revision_binding
    }
}

impl EvidenceKind {
    pub(crate) const fn from_diagnostic(value: DiagnosticEvidenceKind) -> Self {
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

    pub(crate) const fn as_text(self) -> &'static str {
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

    pub(crate) fn parse(value: &str) -> Result<Self, ExperimentError> {
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
    pub(crate) semantic: SemanticProgram,
    pub(crate) workload: WorkloadContract,
    pub(crate) implementation: ImplementationPlan,
    pub(crate) objective: ObjectiveVector,
    pub(crate) provenance: ProducerProvenance,
    pub(crate) evidence: Vec<EvidenceBinding>,
}

impl ExperimentRecord {
    pub fn new(mut spec: ExperimentSpec) -> Result<Self, ExperimentError> {
        spec.workload
            .validate()
            .map_err(|error| ExperimentError::Workload(error.to_string()))?;
        spec.objective
            .validate()
            .map_err(|error| ExperimentError::Objective(error.to_string()))?;
        spec.provenance.validate()?;
        if spec.implementation.id().semantic() != spec.semantic.descriptor().id() {
            return Err(ExperimentError::SemanticImplementationMismatch);
        }
        if spec.evidence.len() > MAX_EVIDENCE_BINDINGS {
            return Err(ExperimentError::TooManyEvidenceBindings);
        }
        for binding in &spec.evidence {
            binding.validate()?;
        }
        spec.evidence.sort();
        if spec.evidence.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ExperimentError::DuplicateEvidenceBinding);
        }
        let measured = spec.objective.measured();
        if (measured.latency_ns.is_some() || measured.energy_nj.is_some())
            && !spec
                .evidence
                .iter()
                .any(|binding| binding.kind == EvidenceKind::HardwareCost)
        {
            return Err(ExperimentError::MissingHardwareEvidence);
        }
        let observed_quality = spec
            .objective
            .quality()
            .iter()
            .any(|metric| metric.value().is_some());
        if observed_quality
            && !spec.evidence.iter().any(|binding| {
                matches!(
                    binding.kind,
                    EvidenceKind::TaskBehavior | EvidenceKind::Generalization
                )
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
    pub const fn semantic(&self) -> &SemanticProgram {
        &self.semantic
    }

    #[must_use]
    pub const fn workload(&self) -> &WorkloadContract {
        &self.workload
    }

    #[must_use]
    pub const fn implementation(&self) -> &ImplementationPlan {
        &self.implementation
    }

    #[must_use]
    pub const fn objective(&self) -> &ObjectiveVector {
        &self.objective
    }

    #[must_use]
    pub const fn provenance(&self) -> &ProducerProvenance {
        &self.provenance
    }

    #[must_use]
    pub fn evidence(&self) -> &[EvidenceBinding] {
        &self.evidence
    }

    #[must_use]
    pub fn fingerprint(&self) -> ExperimentFingerprint {
        ExperimentFingerprint::of_bytes(self.to_canonical_text().as_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExperimentFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl ExperimentFingerprint {
    pub(crate) fn of_bytes(bytes: &[u8]) -> Self {
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
        Self {
            primary: primary ^ length,
            secondary: secondary.rotate_left(31) ^ length,
            length,
        }
    }

    #[must_use]
    pub const fn primary(self) -> u64 {
        self.primary
    }

    #[must_use]
    pub const fn secondary(self) -> u64 {
        self.secondary
    }

    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for ExperimentFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

pub(crate) fn validate_identifier(field: &'static str, value: &str) -> Result<(), ExperimentError> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || value.chars().any(char::is_control)
        || value.chars().any(char::is_whitespace)
    {
        return Err(ExperimentError::InvalidField(field));
    }
    Ok(())
}

pub(crate) fn valid_repository(value: &str) -> bool {
    let mut parts = value.split('/');
    let (Some(owner), Some(repository)) = (parts.next(), parts.next()) else {
        return false;
    };
    !owner.is_empty()
        && !repository.is_empty()
        && parts.next().is_none()
        && owner
            .bytes()
            .chain(repository.bytes())
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

pub(crate) fn is_lower_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn malformed<T>(reason: &str) -> Result<T, ExperimentError> {
    Err(ExperimentError::MalformedCanonical(reason.into()))
}
