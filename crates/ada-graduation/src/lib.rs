//! Fail-closed A11-E4 graduation bundle for exporting qualified semantics toward FLAT.
//!
//! This crate does not lower kernels or mutate FLAT-ATTENTION. It assembles the
//! exact semantic definition, workload, oracle fixtures, implementation plan,
//! reproducible A12 cost estimate, typed objectives, E2 evidence, and explicit
//! research verdict into one deterministic artifact.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

use ada_a10_evidence_schema::{EvidenceWorkloadFingerprint, SemanticEvidenceRecord};
use ada_cegis::{CegisResult, Fixture};
use ada_core::{DiagnosticEvidenceKind, QualificationVerdict};
use ada_cost_model::{
    CostAssumptions, CostModelError, EstimatedCostReport, OperationProfile, estimate_cost,
};
use ada_implementation::ImplementationPlan;
use ada_objective::{
    CandidateKey, CorrectnessStatus, LogicalCost, MeasuredCost, NumericalObjectives,
    ObjectiveError, ObjectiveVector, QualityMetric,
};
use ada_qualification::{
    EvidenceBoundQualification, QUALIFICATION_CASE_VERSION, SemanticWorkloadCase,
};
use ada_semantic::SemanticProgram;
use ada_workload::WorkloadContract;

/// Canonical A11-E4 graduation artifact version.
pub const FLAT_GRADUATION_BUNDLE_VERSION: u16 = 1;
/// Canonical artifact header.
pub const FLAT_GRADUATION_BUNDLE_HEADER: &str = "ADA-FLAT-GRADUATION-V1";
/// Upper bound for one graduation artifact.
pub const MAX_GRADUATION_BUNDLE_BYTES: usize = 16 << 20;
/// Upper bound for retained oracle fixtures.
pub const MAX_GRADUATION_FIXTURES: u64 = 1 << 12;
/// Upper bound for retained E2 evidence records.
pub const MAX_GRADUATION_EVIDENCE: u64 = 1 << 10;

/// Fail-closed assembly or decoding failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraduationError {
    /// The implementation belongs to a different semantic identity.
    SemanticImplementationMismatch,
    /// The supplied CEGIS result is not the source of the qualification fixtures.
    OracleFixtureMismatch,
    /// An oracle fixture does not bind to the qualified workload.
    OracleWorkloadMismatch,
    /// The graduation artifact exceeds a bounded count or byte limit.
    ExceedsLimit {
        /// Rejected field.
        field: &'static str,
        /// Rejected value.
        value: u64,
        /// Inclusive maximum.
        maximum: u64,
    },
    /// Cost estimation failed for the implementation/workload pair.
    CostModel(String),
    /// Objective construction or validation failed.
    Objective(String),
    /// E4a accepts only the provisional status implied by bounded oracle qualification.
    InvalidCorrectnessStatus,
    /// ADOPT/ADAPT require a future, stronger correctness qualification protocol.
    VerdictRequiresQualifiedCorrectness,
    /// Observed task/model quality requires task-behavior evidence.
    MissingTaskEvidence,
    /// Measured latency/energy requires hardware-cost provenance.
    MissingHardwareEvidence,
    /// A nested evidence record does not match the semantic/workload binding.
    EvidenceBindingMismatch,
    /// Two canonical E2 evidence artifacts are identical.
    DuplicateEvidence,
    /// Canonical graduation text is malformed or non-canonical.
    MalformedCanonical(String),
}

impl Display for GraduationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SemanticImplementationMismatch => formatter
                .write_str("implementation semantic identity does not match qualified semantic"),
            Self::OracleFixtureMismatch => formatter
                .write_str("CEGIS active fixtures do not reproduce the qualification fixture set"),
            Self::OracleWorkloadMismatch => {
                formatter.write_str("oracle fixture workload does not match qualification")
            }
            Self::ExceedsLimit {
                field,
                value,
                maximum,
            } => write!(formatter, "{field}={value} exceeds maximum {maximum}"),
            Self::CostModel(reason) => write!(formatter, "cost-model failure: {reason}"),
            Self::Objective(reason) => write!(formatter, "objective failure: {reason}"),
            Self::InvalidCorrectnessStatus => formatter.write_str(
                "E4a requires provisional correctness from bounded oracle qualification",
            ),
            Self::VerdictRequiresQualifiedCorrectness => {
                formatter.write_str("ADOPT/ADAPT require a stronger qualified-correctness protocol")
            }
            Self::MissingTaskEvidence => {
                formatter.write_str("observed quality requires TaskBehavior evidence")
            }
            Self::MissingHardwareEvidence => {
                formatter.write_str("measured physical cost requires HardwareCost evidence")
            }
            Self::EvidenceBindingMismatch => formatter
                .write_str("evidence semantic/workload binding does not match graduation bundle"),
            Self::DuplicateEvidence => {
                formatter.write_str("duplicate canonical semantic evidence artifact")
            }
            Self::MalformedCanonical(reason) => {
                write!(formatter, "malformed graduation artifact: {reason}")
            }
        }
    }
}

impl std::error::Error for GraduationError {}

impl From<CostModelError> for GraduationError {
    fn from(value: CostModelError) -> Self {
        Self::CostModel(value.to_string())
    }
}

impl From<ObjectiveError> for GraduationError {
    fn from(value: ObjectiveError) -> Self {
        Self::Objective(value.to_string())
    }
}

/// Serializable fingerprint of one exact oracle fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OracleFixtureFingerprint {
    primary: u64,
    secondary: u64,
    length: u64,
}

impl OracleFixtureFingerprint {
    fn from_fixture<I>(fixture: &Fixture<I>) -> Self {
        let fingerprint = fixture.fingerprint();
        Self {
            primary: fingerprint.primary(),
            secondary: fingerprint.secondary(),
            length: fingerprint.length(),
        }
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

    /// Canonical identity byte length.
    #[must_use]
    pub const fn length(self) -> u64 {
        self.length
    }
}

impl Display for OracleFixtureFingerprint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:016x}-{:016x}-{:016x}",
            self.primary, self.secondary, self.length
        )
    }
}

/// Exact CEGIS fixture identity and canonical qualification-case artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleFixtureArtifact {
    id: String,
    fingerprint: OracleFixtureFingerprint,
    canonical_text: String,
}

impl OracleFixtureArtifact {
    fn from_fixture(
        fixture: &Fixture<SemanticWorkloadCase>,
        workload: EvidenceWorkloadFingerprint,
    ) -> Result<Self, GraduationError> {
        validate_qualification_case_workload(fixture.canonical_text(), workload)?;
        Ok(Self {
            id: fixture.id().to_owned(),
            fingerprint: OracleFixtureFingerprint::from_fixture(fixture),
            canonical_text: fixture.canonical_text().to_owned(),
        })
    }

    fn from_parts(
        id: String,
        fingerprint: OracleFixtureFingerprint,
        canonical_text: String,
        workload: EvidenceWorkloadFingerprint,
    ) -> Result<Self, GraduationError> {
        let rebuilt = Fixture::new(id.clone(), canonical_text.clone(), ()).map_err(|error| {
            GraduationError::MalformedCanonical(format!("invalid oracle fixture: {error}"))
        })?;
        if OracleFixtureFingerprint::from_fixture(&rebuilt) != fingerprint {
            return Err(GraduationError::MalformedCanonical(
                "oracle fixture fingerprint mismatch".into(),
            ));
        }
        validate_qualification_case_workload(&canonical_text, workload)?;
        Ok(Self {
            id,
            fingerprint,
            canonical_text,
        })
    }

    /// Caller-owned fixture identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Stable CEGIS fixture fingerprint.
    #[must_use]
    pub const fn fingerprint(&self) -> OracleFixtureFingerprint {
        self.fingerprint
    }

    /// Exact qualification-case artifact retained by CEGIS.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }
}

/// Fully bound A11-E4 graduation artifact.
///
/// Construction requires an already evidence-bound oracle qualification plus
/// the exact completed CEGIS result that produced its active fixtures. A12 cost
/// estimates are recomputed from the workload and implementation rather than
/// accepted as caller-supplied measured data.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatGraduationBundle {
    semantic: SemanticProgram,
    workload: WorkloadContract,
    oracle_fixtures: Vec<OracleFixtureArtifact>,
    implementation: ImplementationPlan,
    operation_profile: OperationProfile,
    cost_assumptions: CostAssumptions,
    objectives: ObjectiveVector,
    evidence: Vec<SemanticEvidenceRecord>,
    verdict: QualificationVerdict,
}

/// Caller-supplied evidence-backed objective dimensions not generated by A12.
///
/// Correctness is deliberately not caller-settable in E4a: a
/// [`BoundedOracleQualification`] always maps to [`CorrectnessStatus::Provisional`].
/// Numerical error remains represented by the exact retained oracle fixtures
/// until ADA gains a separately provenance-bound numerical evidence record.
#[derive(Debug, Clone, PartialEq)]
pub struct GraduationObjectives {
    /// Physical measurements. These require HardwareCost E2 provenance.
    pub measured: MeasuredCost,
    /// Task/model quality metrics. Observed values require TaskBehavior E2 evidence.
    pub quality: Vec<QualityMetric>,
}

mod codec;
mod policy;

impl FlatGraduationBundle {
    /// Assemble one graduation artifact from qualified semantic evidence.
    ///
    /// # Errors
    ///
    /// Fails closed on semantic/implementation mismatch, missing exact oracle
    /// fixtures, unsupported A12 cost domains, invalid objective policy, missing
    /// evidence provenance, or an oversized bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qualification: &EvidenceBoundQualification,
        cegis: &CegisResult<SemanticProgram, SemanticWorkloadCase>,
        implementation: ImplementationPlan,
        operation_profile: OperationProfile,
        cost_assumptions: CostAssumptions,
        objective_input: GraduationObjectives,
        verdict: QualificationVerdict,
    ) -> Result<Self, GraduationError> {
        let oracle = qualification.oracle();
        let semantic = oracle.candidate().candidate().clone();
        let workload = oracle.workload().clone();
        if implementation.id().semantic() != semantic.descriptor().id() {
            return Err(GraduationError::SemanticImplementationMismatch);
        }

        let oracle_fixtures = policy::collect_oracle_fixtures(qualification, cegis)?;
        let report = estimate_cost(
            &workload,
            &implementation,
            operation_profile,
            cost_assumptions,
        )?;
        let objectives = policy::objectives_from_report(report, objective_input)?;
        let evidence = qualification.evidence().to_vec();
        policy::validate_bundle_policy(&semantic, &workload, &objectives, &evidence, verdict)?;

        let bundle = Self {
            semantic,
            workload,
            oracle_fixtures,
            implementation,
            operation_profile,
            cost_assumptions,
            objectives,
            evidence,
            verdict,
        };
        bundle.validate_internal()?;
        if bundle.to_canonical_text().len() > MAX_GRADUATION_BUNDLE_BYTES {
            return Err(GraduationError::ExceedsLimit {
                field: "graduation_bundle_bytes",
                value: u64::try_from(bundle.to_canonical_text().len()).unwrap_or(u64::MAX),
                maximum: u64::try_from(MAX_GRADUATION_BUNDLE_BYTES).unwrap_or(u64::MAX),
            });
        }
        Ok(bundle)
    }

    /// Exact executable semantic definition.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticProgram {
        &self.semantic
    }

    /// Exact workload contract covered by qualification/evidence.
    #[must_use]
    pub const fn workload(&self) -> &WorkloadContract {
        &self.workload
    }

    /// Exact oracle fixtures retained from the completed CEGIS corpus.
    #[must_use]
    pub fn oracle_fixtures(&self) -> &[OracleFixtureArtifact] {
        &self.oracle_fixtures
    }

    /// Backend-neutral implementation candidate kept separate from semantic identity.
    #[must_use]
    pub const fn implementation(&self) -> &ImplementationPlan {
        &self.implementation
    }

    /// Caller-declared semantic operation-cost convention used by A12.
    #[must_use]
    pub const fn operation_profile(&self) -> OperationProfile {
        self.operation_profile
    }

    /// Explicit A12 traffic/pass assumptions.
    #[must_use]
    pub const fn cost_assumptions(&self) -> CostAssumptions {
        self.cost_assumptions
    }

    /// Typed objectives. Logical/estimated sections are reproducibly derived from A12.
    #[must_use]
    pub const fn objectives(&self) -> &ObjectiveVector {
        &self.objectives
    }

    /// Full E2 evidence records in canonical order.
    #[must_use]
    pub fn evidence(&self) -> &[SemanticEvidenceRecord] {
        &self.evidence
    }

    /// Explicit research verdict.
    #[must_use]
    pub const fn verdict(&self) -> QualificationVerdict {
        self.verdict
    }

    /// Composite semantic/workload/implementation identity for Pareto archiving.
    ///
    /// Objectives, evidence, and verdict are deliberately excluded so later
    /// measurements do not redefine candidate identity.
    ///
    /// # Errors
    ///
    /// Propagates the bounded `CandidateKey` contract.
    pub fn candidate_key(&self) -> Result<CandidateKey, ObjectiveError> {
        CandidateKey::new(format!(
            "ADA-GRADUATION-CANDIDATE-V1\nsemantic={}\nworkload={}\nimplementation={}\n",
            codec::hex_encode(&self.semantic.to_canonical_text()),
            codec::hex_encode(&self.workload.to_canonical_text()),
            codec::hex_encode(&self.implementation.to_canonical_text()),
        ))
    }
}

#[cfg(test)]
mod tests;
