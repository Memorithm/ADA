//! Fail-closed bridge from SciRust Hub task admission to `ElasticXxx` budgets.
//!
//! ADA owns the scientific disposition of an attention candidate. SciRust Hub
//! owns task/workspace identity and provenance; `ElasticXxx` owns generic resource
//! envelope validation. This module binds those already-published contracts
//! without claiming execution, sandbox enforcement, capacity, performance,
//! adoption, or novelty.

use std::fmt::{Display, Formatter};

use ada_core::QualificationVerdict;

/// SciRust Hub commit whose `TaskSpec` and workspace-evidence v1 contracts were reviewed.
pub const HUB_CONTRACT_SOURCE_COMMIT: &str = "01cf5f1e6d96573d57eef8d2aea22a5a8b0da02f";
/// `ElasticXxx` commit whose `TaskResourceEnvelopeV1` contract was reviewed.
pub const ELASTIC_CONTRACT_SOURCE_COMMIT: &str = "958ce6301730ceeec4bb1252ff12222e5bb7d431";
/// Hub task schema version consumed by this bridge.
pub const HUB_TASK_SPEC_SCHEMA_VERSION: u32 = 1;
/// Hub workspace materialization schema version consumed by this bridge.
pub const HUB_WORKSPACE_MATERIALIZATION_SCHEMA_VERSION: u32 = 1;
/// Stable `ElasticXxx` contract identity emitted by this bridge.
pub const ELASTIC_TASK_RESOURCE_ENVELOPE_V1: &str = "elastic.task-resource-envelope@1.0.0";
/// Capability required for an ADA research evaluation task.
pub const ADA_RESEARCH_EVALUATE_CAPABILITY: &str = "ada.research.evaluate.v1";
/// Repository identity accepted by this ADA-specific bridge.
pub const ADA_REPOSITORY: &str = "Memorithm/ADA";

const MAX_LABEL_BYTES: usize = 256;

/// Hub isolation levels in increasing strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HubIsolationLevelV1 {
    /// Supervised process only; not a hostile-code boundary.
    Process,
    /// Dedicated OS container.
    Container,
    /// Userspace-kernel isolation.
    Gvisor,
    /// Hardware-virtualized microVM.
    MicroVm,
}

/// Resource bounds from SciRust Hub `TaskSpec::budget`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HubResourceBudgetV1 {
    pub cpu_millis: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub wall_clock_ms: Option<u64>,
    pub gpu_devices: Option<u32>,
    pub model_tokens: Option<u64>,
}

impl HubResourceBudgetV1 {
    fn validate(self) -> Result<(), EcosystemTaskBridgeError> {
        for (dimension, value) in [
            ("cpu_millis", self.cpu_millis),
            ("memory_bytes", self.memory_bytes),
            ("wall_clock_ms", self.wall_clock_ms),
            ("model_tokens", self.model_tokens),
        ] {
            if value == Some(0) {
                return Err(EcosystemTaskBridgeError::ZeroBound { dimension });
            }
        }
        Ok(())
    }
}

/// ADA-relevant projection of the Hub task contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubAdaTaskSpecV1 {
    pub schema_version: u32,
    pub task_id: String,
    pub identity_task_id: String,
    pub principal: String,
    pub repository: String,
    pub revision: String,
    pub read_only: bool,
    pub capabilities: Vec<String>,
    pub budget: HubResourceBudgetV1,
    pub minimum_isolation: HubIsolationLevelV1,
    pub network_default_deny: bool,
    pub allowed_endpoints: Vec<String>,
    pub writable_workspace: bool,
}

impl HubAdaTaskSpecV1 {
    /// Validate the exact Hub-owned task identity and ADA workspace request.
    ///
    /// # Errors
    /// Rejects schema drift, identity mismatch, floating source identity,
    /// missing capability, weak isolation, network egress, writable workspaces,
    /// and invalid resource bounds.
    pub fn validate(&self) -> Result<(), EcosystemTaskBridgeError> {
        if self.schema_version != HUB_TASK_SPEC_SCHEMA_VERSION {
            return Err(EcosystemTaskBridgeError::UnsupportedHubTaskSchema {
                actual: self.schema_version,
            });
        }
        validate_label("task_id", &self.task_id)?;
        validate_label("principal", &self.principal)?;
        if self.task_id != self.identity_task_id {
            return Err(EcosystemTaskBridgeError::TaskIdentityMismatch);
        }
        if self.repository != ADA_REPOSITORY {
            return Err(EcosystemTaskBridgeError::UnexpectedRepository);
        }
        if !is_exact_git_object_id(&self.revision) {
            return Err(EcosystemTaskBridgeError::FloatingOrMalformedRevision);
        }
        if !self
            .capabilities
            .iter()
            .any(|capability| capability == ADA_RESEARCH_EVALUATE_CAPABILITY)
        {
            return Err(EcosystemTaskBridgeError::MissingResearchCapability);
        }
        if self.capabilities.len() > 128 {
            return Err(EcosystemTaskBridgeError::TooManyCapabilities);
        }
        for capability in &self.capabilities {
            validate_label("capability", capability)?;
        }
        if !self.read_only || self.writable_workspace {
            return Err(EcosystemTaskBridgeError::WorkspaceMustBeReadOnly);
        }
        if self.minimum_isolation < HubIsolationLevelV1::Container {
            return Err(EcosystemTaskBridgeError::IsolationTooWeak);
        }
        if !self.network_default_deny || !self.allowed_endpoints.is_empty() {
            return Err(EcosystemTaskBridgeError::NetworkMustBeClosed);
        }
        self.budget.validate()
    }
}

/// Hub-retained proof that the requested ADA checkout was materialized exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubAdaWorkspaceEvidenceV1 {
    pub schema_version: u32,
    pub repository: String,
    pub requested_revision: String,
    pub observed_revision: String,
    pub tree_id: String,
    pub content_digest: String,
    pub read_only_requested: bool,
}

impl HubAdaWorkspaceEvidenceV1 {
    fn validate_against(&self, task: &HubAdaTaskSpecV1) -> Result<(), EcosystemTaskBridgeError> {
        if self.schema_version != HUB_WORKSPACE_MATERIALIZATION_SCHEMA_VERSION {
            return Err(
                EcosystemTaskBridgeError::UnsupportedWorkspaceEvidenceSchema {
                    actual: self.schema_version,
                },
            );
        }
        if self.repository != task.repository
            || self.requested_revision != task.revision
            || self.observed_revision != task.revision
            || self.read_only_requested != task.read_only
        {
            return Err(EcosystemTaskBridgeError::WorkspaceEvidenceMismatch);
        }
        if !is_exact_git_object_id(&self.tree_id) {
            return Err(EcosystemTaskBridgeError::MalformedTreeIdentity);
        }
        if !is_lower_hex(&self.content_digest, 64) {
            return Err(EcosystemTaskBridgeError::MalformedContentDigest);
        }
        Ok(())
    }
}

/// `ElasticXxx` v1 hard bounds produced from the admitted Hub task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElasticTaskResourceBudgetV1 {
    pub cpu_millis: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub wall_clock_ms: Option<u64>,
    pub gpu_devices: Option<u32>,
    pub model_tokens: Option<u64>,
    pub concurrency: Option<u32>,
}

/// Caller-provided estimates checked before an `ElasticXxx` plan may be requested.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticTaskResourceEstimateV1 {
    pub cpu_millis: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub wall_clock_ms: Option<u64>,
    pub gpu_devices: Option<u32>,
    pub model_tokens: Option<u64>,
    pub concurrency: Option<u32>,
}

/// Bounded preflight plan tied to one exact envelope identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElasticTaskResourcePlanV1 {
    envelope_binding: String,
    pub estimates: ElasticTaskResourceEstimateV1,
}

impl ElasticTaskResourcePlanV1 {
    /// Whether this plan was produced from the exact supplied envelope.
    #[must_use]
    pub fn is_bound_to(&self, envelope: &ElasticTaskResourceEnvelopeV1) -> bool {
        self.envelope_binding == envelope.binding()
    }
}

/// `ElasticXxx` resource envelope bound to Hub task and workspace provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElasticTaskResourceEnvelopeV1 {
    pub contract: &'static str,
    pub task_ref: String,
    pub workspace_ref: String,
    pub budget: ElasticTaskResourceBudgetV1,
}

impl ElasticTaskResourceEnvelopeV1 {
    fn binding(&self) -> String {
        self.to_canonical_text()
    }

    /// Validate estimates against every declared hard bound.
    ///
    /// # Errors
    /// Missing, zero, or over-bound requested dimensions fail closed. Zero GPU
    /// remains a valid explicit CPU-only estimate, matching `ElasticXxx` v1.
    pub fn preflight(
        &self,
        estimates: ElasticTaskResourceEstimateV1,
    ) -> Result<ElasticTaskResourcePlanV1, EcosystemTaskBridgeError> {
        check_dimension(
            "cpu_millis",
            self.budget.cpu_millis,
            estimates.cpu_millis,
            false,
        )?;
        check_dimension(
            "memory_bytes",
            self.budget.memory_bytes,
            estimates.memory_bytes,
            false,
        )?;
        check_dimension(
            "wall_clock_ms",
            self.budget.wall_clock_ms,
            estimates.wall_clock_ms,
            false,
        )?;
        check_dimension(
            "gpu_devices",
            self.budget.gpu_devices.map(u64::from),
            estimates.gpu_devices.map(u64::from),
            true,
        )?;
        check_dimension(
            "model_tokens",
            self.budget.model_tokens,
            estimates.model_tokens,
            false,
        )?;
        check_dimension(
            "concurrency",
            self.budget.concurrency.map(u64::from),
            estimates.concurrency.map(u64::from),
            false,
        )?;
        Ok(ElasticTaskResourcePlanV1 {
            envelope_binding: self.binding(),
            estimates,
        })
    }

    /// Canonical destination fixture used by cross-repository contract tests.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        format!(
            concat!(
                "ADA-ELASTIC-TASK-ENVELOPE-V1\n",
                "contract={}\n",
                "elastic_source_commit={}\n",
                "task_ref={}\n",
                "workspace_ref={}\n",
                "cpu_millis={}\n",
                "memory_bytes={}\n",
                "wall_clock_ms={}\n",
                "gpu_devices={}\n",
                "model_tokens={}\n",
                "concurrency={}\n"
            ),
            self.contract,
            ELASTIC_CONTRACT_SOURCE_COMMIT,
            self.task_ref,
            self.workspace_ref,
            optional_u64(self.budget.cpu_millis),
            optional_u64(self.budget.memory_bytes),
            optional_u64(self.budget.wall_clock_ms),
            optional_u32(self.budget.gpu_devices),
            optional_u64(self.budget.model_tokens),
            optional_u32(self.budget.concurrency),
        )
    }
}

/// Successfully admitted research-only handoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedAdaResearchTaskV1 {
    pub task_id: String,
    pub source_revision: String,
    pub source_tree: String,
    pub content_digest: String,
    pub verdict: QualificationVerdict,
    pub elastic_envelope: ElasticTaskResourceEnvelopeV1,
}

impl AdmittedAdaResearchTaskV1 {
    /// Canonical Hub-side destination fixture with immutable provenance.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        format!(
            concat!(
                "ADA-HUB-ADMISSION-V1\n",
                "hub_source_commit={}\n",
                "task_id={}\n",
                "repository={}\n",
                "revision={}\n",
                "tree={}\n",
                "content_digest={}\n",
                "capability={}\n",
                "verdict=CONTINUE_RESEARCH\n"
            ),
            HUB_CONTRACT_SOURCE_COMMIT,
            self.task_id,
            ADA_REPOSITORY,
            self.source_revision,
            self.source_tree,
            self.content_digest,
            ADA_RESEARCH_EVALUATE_CAPABILITY,
        )
    }
}

/// Bind Hub admission/provenance to an `ElasticXxx` budget envelope.
///
/// Only `CONTINUE_RESEARCH` may execute through this research bridge. A
/// rejected candidate remains a preserved negative result; `ADOPT` and `ADAPT`
/// require a stronger qualification protocol and are refused.
///
/// # Errors
/// Any schema, identity, provenance, isolation, egress, capability, budget, or
/// verdict mismatch fails closed.
pub fn admit_research_task(
    task: &HubAdaTaskSpecV1,
    workspace: &HubAdaWorkspaceEvidenceV1,
    verdict: QualificationVerdict,
) -> Result<AdmittedAdaResearchTaskV1, EcosystemTaskBridgeError> {
    task.validate()?;
    workspace.validate_against(task)?;
    match verdict {
        QualificationVerdict::ContinueResearch => {}
        QualificationVerdict::Reject => {
            return Err(EcosystemTaskBridgeError::RejectedCandidate);
        }
        QualificationVerdict::Adopt | QualificationVerdict::Adapt => {
            return Err(EcosystemTaskBridgeError::PromotionRequiresQualifiedCorrectness);
        }
    }

    let task_ref = format!("ada-task:{}", task.task_id);
    let workspace_ref = format!("ada-workspace:{}", workspace.content_digest);
    validate_label("elastic task_ref", &task_ref)?;
    validate_label("elastic workspace_ref", &workspace_ref)?;
    let elastic_envelope = ElasticTaskResourceEnvelopeV1 {
        contract: ELASTIC_TASK_RESOURCE_ENVELOPE_V1,
        task_ref,
        workspace_ref,
        budget: ElasticTaskResourceBudgetV1 {
            cpu_millis: task.budget.cpu_millis,
            memory_bytes: task.budget.memory_bytes,
            wall_clock_ms: task.budget.wall_clock_ms,
            gpu_devices: task.budget.gpu_devices,
            model_tokens: task.budget.model_tokens,
            concurrency: Some(1),
        },
    };
    Ok(AdmittedAdaResearchTaskV1 {
        task_id: task.task_id.clone(),
        source_revision: workspace.observed_revision.clone(),
        source_tree: workspace.tree_id.clone(),
        content_digest: workspace.content_digest.clone(),
        verdict,
        elastic_envelope,
    })
}

/// Fail-closed bridge validation error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EcosystemTaskBridgeError {
    UnsupportedHubTaskSchema { actual: u32 },
    UnsupportedWorkspaceEvidenceSchema { actual: u32 },
    InvalidLabel { field: &'static str },
    TaskIdentityMismatch,
    UnexpectedRepository,
    FloatingOrMalformedRevision,
    MissingResearchCapability,
    TooManyCapabilities,
    WorkspaceMustBeReadOnly,
    IsolationTooWeak,
    NetworkMustBeClosed,
    ZeroBound { dimension: &'static str },
    WorkspaceEvidenceMismatch,
    MalformedTreeIdentity,
    MalformedContentDigest,
    RejectedCandidate,
    PromotionRequiresQualifiedCorrectness,
    MissingEstimate { dimension: &'static str },
    ZeroEstimate { dimension: &'static str },
    ExceedsBound { dimension: &'static str },
}

impl Display for EcosystemTaskBridgeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedHubTaskSchema { actual } => {
                write!(formatter, "unsupported Hub task schema {actual}")
            }
            Self::UnsupportedWorkspaceEvidenceSchema { actual } => {
                write!(
                    formatter,
                    "unsupported Hub workspace evidence schema {actual}"
                )
            }
            Self::InvalidLabel { field } => write!(formatter, "invalid bounded {field}"),
            Self::TaskIdentityMismatch => formatter.write_str("Hub task identity mismatch"),
            Self::UnexpectedRepository => formatter.write_str("task is not for Memorithm/ADA"),
            Self::FloatingOrMalformedRevision => {
                formatter.write_str("workspace revision is not an exact Git object id")
            }
            Self::MissingResearchCapability => {
                formatter.write_str("ADA research evaluation capability is missing")
            }
            Self::TooManyCapabilities => formatter.write_str("too many capability grants"),
            Self::WorkspaceMustBeReadOnly => {
                formatter.write_str("ADA research workspace must be read-only")
            }
            Self::IsolationTooWeak => {
                formatter.write_str("container-or-stronger isolation required")
            }
            Self::NetworkMustBeClosed => {
                formatter.write_str("default-deny network with no endpoints required")
            }
            Self::ZeroBound { dimension } => write!(formatter, "zero {dimension} bound"),
            Self::WorkspaceEvidenceMismatch => formatter.write_str("workspace evidence mismatch"),
            Self::MalformedTreeIdentity => formatter.write_str("malformed exact tree identity"),
            Self::MalformedContentDigest => formatter.write_str("malformed Hub content digest"),
            Self::RejectedCandidate => {
                formatter.write_str("rejected candidate is not dispatchable")
            }
            Self::PromotionRequiresQualifiedCorrectness => {
                formatter.write_str("ADOPT/ADAPT require qualified correctness")
            }
            Self::MissingEstimate { dimension } => {
                write!(formatter, "missing {dimension} estimate")
            }
            Self::ZeroEstimate { dimension } => write!(formatter, "zero {dimension} estimate"),
            Self::ExceedsBound { dimension } => write!(formatter, "{dimension} exceeds bound"),
        }
    }
}

impl std::error::Error for EcosystemTaskBridgeError {}

fn check_dimension(
    dimension: &'static str,
    bound: Option<u64>,
    estimate: Option<u64>,
    allow_zero: bool,
) -> Result<(), EcosystemTaskBridgeError> {
    let Some(bound) = bound else {
        if estimate.is_some() {
            return Err(EcosystemTaskBridgeError::ExceedsBound { dimension });
        }
        return Ok(());
    };
    let estimate = estimate.ok_or(EcosystemTaskBridgeError::MissingEstimate { dimension })?;
    if estimate == 0 && !allow_zero {
        return Err(EcosystemTaskBridgeError::ZeroEstimate { dimension });
    }
    if estimate > bound {
        return Err(EcosystemTaskBridgeError::ExceedsBound { dimension });
    }
    Ok(())
}

fn validate_label(field: &'static str, value: &str) -> Result<(), EcosystemTaskBridgeError> {
    if value.is_empty()
        || value.len() > MAX_LABEL_BYTES
        || value.trim() != value
        || !value.is_ascii()
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(EcosystemTaskBridgeError::InvalidLabel { field });
    }
    Ok(())
}

fn is_exact_git_object_id(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64) && is_lower_hex(value, value.len())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

fn optional_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REVISION: &str = "7958439630ff5e092d32ac8d618411785d885d59";
    const TREE: &str = "548047ae73bf2c0546851386cea6a9b8d6a4e7ba";
    const DIGEST: &str = "5a59bc0d5391a643ce75f9b9d27c7cb5bfa784c2f423d5b2f680184783f06ea1";

    fn task() -> HubAdaTaskSpecV1 {
        HubAdaTaskSpecV1 {
            schema_version: HUB_TASK_SPEC_SCHEMA_VERSION,
            task_id: "018f0f77-32d4-7a26-8e46-2d7049781234".into(),
            identity_task_id: "018f0f77-32d4-7a26-8e46-2d7049781234".into(),
            principal: "spiffe://memorithm.test/ada/research".into(),
            repository: ADA_REPOSITORY.into(),
            revision: REVISION.into(),
            read_only: true,
            capabilities: vec![ADA_RESEARCH_EVALUATE_CAPABILITY.into()],
            budget: HubResourceBudgetV1 {
                cpu_millis: Some(2_000),
                memory_bytes: Some(1 << 30),
                wall_clock_ms: Some(60_000),
                gpu_devices: Some(0),
                model_tokens: Some(4_096),
            },
            minimum_isolation: HubIsolationLevelV1::Container,
            network_default_deny: true,
            allowed_endpoints: Vec::new(),
            writable_workspace: false,
        }
    }

    fn evidence() -> HubAdaWorkspaceEvidenceV1 {
        HubAdaWorkspaceEvidenceV1 {
            schema_version: HUB_WORKSPACE_MATERIALIZATION_SCHEMA_VERSION,
            repository: ADA_REPOSITORY.into(),
            requested_revision: REVISION.into(),
            observed_revision: REVISION.into(),
            tree_id: TREE.into(),
            content_digest: DIGEST.into(),
            read_only_requested: true,
        }
    }

    #[test]
    fn destination_fixtures_match_exact_contract_output() {
        let admitted =
            admit_research_task(&task(), &evidence(), QualificationVerdict::ContinueResearch)
                .unwrap();
        assert_eq!(
            admitted.to_canonical_text(),
            include_str!("../tests/fixtures/scirust_hub_ada_admission_v1.txt")
        );
        assert_eq!(
            admitted.elastic_envelope.to_canonical_text(),
            include_str!("../tests/fixtures/elastic_ada_task_envelope_v1.txt")
        );
    }

    #[test]
    fn exact_provenance_and_budget_preflight_succeed() {
        let admitted =
            admit_research_task(&task(), &evidence(), QualificationVerdict::ContinueResearch)
                .unwrap();
        let plan = admitted
            .elastic_envelope
            .preflight(ElasticTaskResourceEstimateV1 {
                cpu_millis: Some(1_500),
                memory_bytes: Some(1 << 29),
                wall_clock_ms: Some(30_000),
                gpu_devices: Some(0),
                model_tokens: Some(2_048),
                concurrency: Some(1),
            })
            .unwrap();
        assert!(plan.is_bound_to(&admitted.elastic_envelope));
    }

    #[test]
    fn floating_source_and_materialization_drift_fail_closed() {
        let mut floating = task();
        floating.revision = "main".into();
        assert_eq!(
            admit_research_task(
                &floating,
                &evidence(),
                QualificationVerdict::ContinueResearch
            ),
            Err(EcosystemTaskBridgeError::FloatingOrMalformedRevision)
        );

        let mut drift = evidence();
        drift.observed_revision = "7034491cc408ff77820589957349aa4fad1ee097".into();
        assert_eq!(
            admit_research_task(&task(), &drift, QualificationVerdict::ContinueResearch),
            Err(EcosystemTaskBridgeError::WorkspaceEvidenceMismatch)
        );
    }

    #[test]
    fn weak_runtime_or_open_network_fails_closed() {
        let mut weak = task();
        weak.minimum_isolation = HubIsolationLevelV1::Process;
        assert_eq!(
            admit_research_task(&weak, &evidence(), QualificationVerdict::ContinueResearch),
            Err(EcosystemTaskBridgeError::IsolationTooWeak)
        );
        let mut egress = task();
        egress.allowed_endpoints.push("https://example.test".into());
        assert_eq!(
            admit_research_task(&egress, &evidence(), QualificationVerdict::ContinueResearch),
            Err(EcosystemTaskBridgeError::NetworkMustBeClosed)
        );
    }

    #[test]
    fn promotions_and_rejected_candidates_never_dispatch() {
        for verdict in [QualificationVerdict::Adopt, QualificationVerdict::Adapt] {
            assert_eq!(
                admit_research_task(&task(), &evidence(), verdict),
                Err(EcosystemTaskBridgeError::PromotionRequiresQualifiedCorrectness)
            );
        }
        assert_eq!(
            admit_research_task(&task(), &evidence(), QualificationVerdict::Reject),
            Err(EcosystemTaskBridgeError::RejectedCandidate)
        );
    }

    #[test]
    fn estimates_must_be_present_and_within_exact_bounds() {
        let envelope =
            admit_research_task(&task(), &evidence(), QualificationVerdict::ContinueResearch)
                .unwrap()
                .elastic_envelope;
        assert_eq!(
            envelope.preflight(ElasticTaskResourceEstimateV1 {
                cpu_millis: Some(2_001),
                memory_bytes: Some(1 << 29),
                wall_clock_ms: Some(30_000),
                gpu_devices: Some(0),
                model_tokens: Some(2_048),
                concurrency: Some(1),
            }),
            Err(EcosystemTaskBridgeError::ExceedsBound {
                dimension: "cpu_millis"
            })
        );
        assert_eq!(
            envelope.preflight(ElasticTaskResourceEstimateV1 {
                cpu_millis: Some(1_000),
                memory_bytes: None,
                wall_clock_ms: Some(30_000),
                gpu_devices: Some(0),
                model_tokens: Some(2_048),
                concurrency: Some(1),
            }),
            Err(EcosystemTaskBridgeError::MissingEstimate {
                dimension: "memory_bytes"
            })
        );
    }
}
