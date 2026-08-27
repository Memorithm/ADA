//! ADA-A11 semantic identity and cross-project evidence contracts.
//!
//! This module deliberately separates **what an attention mechanism computes**
//! from **how one implementation computes it**.  ADA can therefore search or
//! qualify multiple implementations of one semantic without allowing a kernel
//! detail to silently redefine the scientific hypothesis under test.

/// Broad semantic families currently relevant to the Memorithm attention
/// programme.
///
/// A family is a classification aid, not evidence that the corresponding
/// mechanism is useful or implemented. `Experimental` is the fail-safe bucket
/// for a candidate that does not yet deserve a stable family of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticFamily {
    StandardSoftmax,
    DifferentialSigned,
    ToeplitzStructured,
    ProlateConcentration,
    GroundStateGreen,
    SpectralFlow,
    RecurrentMemory,
    Hybrid,
    Experimental,
}

/// Mask/visibility semantics that are part of a reference semantic contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskContract {
    Bidirectional,
    Causal,
    ExternalMask,
}

/// Whether the interaction rule carries explicit state between declared steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateContract {
    Stateless,
    Recurrent,
}

/// High-level constraint on the weights or linear coefficients used to mix
/// information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightContract {
    /// Non-negative coefficients normalized to a unit row sum.
    ProbabilitySimplex,
    /// Signed coefficients are allowed by the semantic definition.
    Signed,
    /// A structured linear operator is used without a simplex requirement.
    StructuredLinear,
    /// Coefficients depend on an explicit carried state.
    StateDependent,
}

/// Validation failures for ADA-A11 identity/evidence metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticContractError {
    EmptyName,
    InvalidName,
    ZeroRevision,
    EmptyRepository,
    EmptyArtifact,
    EmptyRevisionBinding,
}

fn validate_name(name: &str) -> Result<(), SemanticContractError> {
    if name.is_empty() {
        return Err(SemanticContractError::EmptyName);
    }

    let valid = name.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'-' | b'_' | b'.')
    });
    if !valid {
        return Err(SemanticContractError::InvalidName);
    }

    Ok(())
}

/// Stable identity of one semantic hypothesis.
///
/// The identity intentionally contains no kernel name, device, benchmark
/// result, ITD/TDI descriptor, or hardware evidence. Those belong to separate
/// records so that evidence cannot redefine semantic identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticId {
    family: SemanticFamily,
    name: String,
    revision: u32,
}

impl SemanticId {
    /// Construct a validated semantic identity.
    ///
    /// `name` is a stable lowercase ASCII slug. Revision numbering starts at 1.
    ///
    /// # Errors
    ///
    /// Returns a validation error for an empty/invalid slug or revision zero.
    pub fn new(
        family: SemanticFamily,
        name: impl Into<String>,
        revision: u32,
    ) -> Result<Self, SemanticContractError> {
        let name = name.into();
        validate_name(&name)?;
        if revision == 0 {
            return Err(SemanticContractError::ZeroRevision);
        }

        Ok(Self {
            family,
            name,
            revision,
        })
    }

    #[must_use]
    pub const fn family(&self) -> SemanticFamily {
        self.family
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn revision(&self) -> u32 {
        self.revision
    }
}

/// Minimal reference-level properties of a semantic candidate.
///
/// This is deliberately smaller than a future executable semantic IR. A11-E0
/// needs a stable identity boundary before ADA extends its synthesis grammar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticDescriptor {
    id: SemanticId,
    mask: MaskContract,
    state: StateContract,
    weights: WeightContract,
}

impl SemanticDescriptor {
    #[must_use]
    pub const fn new(
        id: SemanticId,
        mask: MaskContract,
        state: StateContract,
        weights: WeightContract,
    ) -> Self {
        Self {
            id,
            mask,
            state,
            weights,
        }
    }

    #[must_use]
    pub const fn id(&self) -> &SemanticId {
        &self.id
    }

    #[must_use]
    pub const fn mask(&self) -> MaskContract {
        self.mask
    }

    #[must_use]
    pub const fn state(&self) -> StateContract {
        self.state
    }

    #[must_use]
    pub const fn weights(&self) -> WeightContract {
        self.weights
    }
}

/// Identity of one implementation candidate for a declared semantic.
///
/// Two different values of this type may intentionally point at the same
/// [`SemanticId`]. This is the key distinction between semantic research and
/// implementation/kernel research.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImplementationCandidateId {
    semantic: SemanticId,
    name: String,
    revision: u32,
}

impl ImplementationCandidateId {
    /// Construct an implementation identity bound to an existing semantic.
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid implementation slug or zero
    /// revision.
    pub fn new(
        semantic: SemanticId,
        name: impl Into<String>,
        revision: u32,
    ) -> Result<Self, SemanticContractError> {
        let name = name.into();
        validate_name(&name)?;
        if revision == 0 {
            return Err(SemanticContractError::ZeroRevision);
        }
        Ok(Self {
            semantic,
            name,
            revision,
        })
    }

    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn revision(&self) -> u32 {
        self.revision
    }
}

/// Kind of external evidence attached to a semantic qualification record.
///
/// ITD and TDI deliberately appear as evidence kinds rather than dependencies
/// of `ada-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticEvidenceKind {
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

/// Stable reference to evidence produced by ADA or another repository.
///
/// `revision_binding` is intentionally opaque: it may be a Git commit SHA,
/// frozen manifest digest, evidence-record digest, or another immutable revision
/// identifier defined by the producing project.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiagnosticEvidenceRef {
    kind: DiagnosticEvidenceKind,
    repository: String,
    artifact: String,
    revision_binding: String,
}

impl DiagnosticEvidenceRef {
    /// Construct a validated external evidence reference.
    ///
    /// # Errors
    ///
    /// Returns an error when any provenance field is empty.
    pub fn new(
        kind: DiagnosticEvidenceKind,
        repository: impl Into<String>,
        artifact: impl Into<String>,
        revision_binding: impl Into<String>,
    ) -> Result<Self, SemanticContractError> {
        let repository = repository.into();
        let artifact = artifact.into();
        let revision_binding = revision_binding.into();
        if repository.trim().is_empty() {
            return Err(SemanticContractError::EmptyRepository);
        }
        if artifact.trim().is_empty() {
            return Err(SemanticContractError::EmptyArtifact);
        }
        if revision_binding.trim().is_empty() {
            return Err(SemanticContractError::EmptyRevisionBinding);
        }

        Ok(Self {
            kind,
            repository,
            artifact,
            revision_binding,
        })
    }

    #[must_use]
    pub const fn kind(&self) -> DiagnosticEvidenceKind {
        self.kind
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

/// ADA's research verdict for a candidate before or during FLAT graduation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualificationVerdict {
    ContinueResearch,
    Adopt,
    Adapt,
    Reject,
}

/// Artifact that binds a qualified semantic to its oracle/evidence without
/// confusing that evidence with semantic identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatGraduationRecord {
    semantic: SemanticDescriptor,
    reference_oracle: DiagnosticEvidenceRef,
    evidence: Vec<DiagnosticEvidenceRef>,
    verdict: QualificationVerdict,
}

impl FlatGraduationRecord {
    #[must_use]
    pub fn new(
        semantic: SemanticDescriptor,
        reference_oracle: DiagnosticEvidenceRef,
        evidence: Vec<DiagnosticEvidenceRef>,
        verdict: QualificationVerdict,
    ) -> Self {
        Self {
            semantic,
            reference_oracle,
            evidence,
            verdict,
        }
    }

    #[must_use]
    pub const fn semantic(&self) -> &SemanticDescriptor {
        &self.semantic
    }

    #[must_use]
    pub const fn reference_oracle(&self) -> &DiagnosticEvidenceRef {
        &self.reference_oracle
    }

    #[must_use]
    pub fn evidence(&self) -> &[DiagnosticEvidenceRef] {
        &self.evidence
    }

    #[must_use]
    pub const fn verdict(&self) -> QualificationVerdict {
        self.verdict
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn softmax_semantic() -> SemanticId {
        SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1)
            .expect("fixture identity is valid")
    }

    #[test]
    fn implementation_identity_cannot_redefine_semantic_identity() {
        let semantic = softmax_semantic();
        let scalar = ImplementationCandidateId::new(semantic.clone(), "scalar-oracle", 1)
            .expect("valid implementation");
        let fused = ImplementationCandidateId::new(semantic.clone(), "fused-kernel", 7)
            .expect("valid implementation");

        assert_ne!(scalar, fused);
        assert_eq!(scalar.semantic(), &semantic);
        assert_eq!(fused.semantic(), &semantic);
    }

    #[test]
    fn different_semantic_hypotheses_remain_distinct_with_same_implementation_slug() {
        let softmax = softmax_semantic();
        let signed = SemanticId::new(SemanticFamily::DifferentialSigned, "signed-difference", 1)
            .expect("valid semantic");

        let softmax_impl = ImplementationCandidateId::new(softmax, "reference", 1)
            .expect("valid implementation");
        let signed_impl = ImplementationCandidateId::new(signed, "reference", 1)
            .expect("valid implementation");

        assert_ne!(softmax_impl.semantic(), signed_impl.semantic());
    }

    #[test]
    fn mechanistic_evidence_is_attached_without_entering_semantic_identity() {
        let semantic_id = SemanticId::new(
            SemanticFamily::GroundStateGreen,
            "green-kernel-experimental",
            1,
        )
        .expect("valid semantic");
        let descriptor = SemanticDescriptor::new(
            semantic_id.clone(),
            MaskContract::Causal,
            StateContract::Stateless,
            WeightContract::StructuredLinear,
        );
        let oracle = DiagnosticEvidenceRef::new(
            DiagnosticEvidenceKind::StaticOperator,
            "Memorithm/ADA",
            "a11-e1-oracle",
            "sha256:oracle",
        )
        .expect("valid evidence");
        let tdi = DiagnosticEvidenceRef::new(
            DiagnosticEvidenceKind::TdiRecovery,
            "Memorithm/TDI",
            "tdi-ai-gate-b",
            "commit:fixture",
        )
        .expect("valid evidence");
        let itd = DiagnosticEvidenceRef::new(
            DiagnosticEvidenceKind::ItdStructural,
            "Memorithm/itd-simulator",
            "attention-structure-evidence",
            "commit:fixture",
        )
        .expect("valid evidence");

        let record = FlatGraduationRecord::new(
            descriptor,
            oracle,
            vec![tdi, itd],
            QualificationVerdict::ContinueResearch,
        );

        assert_eq!(record.semantic().id(), &semantic_id);
        assert_eq!(record.evidence().len(), 2);
        assert_eq!(record.verdict(), QualificationVerdict::ContinueResearch);
    }

    #[test]
    fn identifiers_fail_closed_on_ambiguous_names_or_zero_revision() {
        assert_eq!(
            SemanticId::new(SemanticFamily::Experimental, "Has Spaces", 1),
            Err(SemanticContractError::InvalidName)
        );
        assert_eq!(
            SemanticId::new(SemanticFamily::Experimental, "candidate", 0),
            Err(SemanticContractError::ZeroRevision)
        );
    }

    #[test]
    fn evidence_requires_explicit_provenance_fields() {
        assert_eq!(
            DiagnosticEvidenceRef::new(
                DiagnosticEvidenceKind::TdiRecovery,
                "Memorithm/TDI",
                "",
                "commit:abc",
            ),
            Err(SemanticContractError::EmptyArtifact)
        );
    }
}
