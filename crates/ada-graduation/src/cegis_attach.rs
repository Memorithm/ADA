//! Fail-closed attachment of explicit mechanistic task-quality evidence onto a
//! completed CEGIS outcome.
//!
//! Non-claims / non-mappings:
//! - CEGIS survival, rejection counts, and counterexample absence never become
//!   task quality or algorithmic-error values;
//! - ITD/TDI diagnostics and cost fields cannot disguise as quality;
//! - attaching evidence does not adopt, promote to FLAT, or claim usefulness;
//! - this is not a training loop and makes no FLAT claims.

use ada_cegis::{CegisResult, CegisStats};
use ada_objective::{
    AlgorithmicError, AlgorithmicErrorSource, CorrectnessStatus, EstimatedCost,
    LaneSeparatedEvidence, LaneSeparatedEvidenceSpec, LogicalCost, MeasuredCost,
    NumericalObjectives, TaskQualityContract, TaskQualityFill, accept_algorithmic_error,
};
use ada_workload::{AttentionSurfaceIdentity, WorkloadFingerprint};

use crate::{GraduationError, GraduationObjectives, accept_lane_separated_quality};

/// Opaque CEGIS run counters retained for provenance only.
///
/// Values here must never be remapped into [`LaneSeparatedEvidence`] quality or
/// algorithmic lanes. Use [`crate::task_quality_from_cegis_survival`] /
/// [`crate::algorithmic_error_from_cegis_survival`] to see the explicit refusals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CegisRunContext {
    survivor_count: u64,
    rejected_count: u64,
    active_fixture_count: u64,
    counterexample_count: u64,
    stats: CegisStats,
}

impl CegisRunContext {
    /// Snapshot opaque run context from a completed CEGIS result.
    ///
    /// Counts are recorded for provenance; they are not quality evidence.
    #[must_use]
    pub fn from_cegis_result<C, I>(result: &CegisResult<C, I>) -> Self {
        Self {
            survivor_count: u64::try_from(result.survivors().len()).unwrap_or(u64::MAX),
            rejected_count: u64::try_from(result.rejected().len()).unwrap_or(u64::MAX),
            active_fixture_count: u64::try_from(result.active_fixtures().len()).unwrap_or(u64::MAX),
            counterexample_count: u64::try_from(result.counterexamples().len()).unwrap_or(u64::MAX),
            stats: result.stats(),
        }
    }

    /// Construct a context from explicit counts (tests / offline replay).
    #[must_use]
    pub fn from_counts(
        survivor_count: u64,
        rejected_count: u64,
        active_fixture_count: u64,
        counterexample_count: u64,
    ) -> Self {
        Self {
            survivor_count,
            rejected_count,
            active_fixture_count,
            counterexample_count,
            stats: CegisStats::default(),
        }
    }

    /// Number of surviving candidates (opaque; not task quality).
    #[must_use]
    pub const fn survivor_count(self) -> u64 {
        self.survivor_count
    }

    /// Number of rejected candidates (opaque; not task quality).
    #[must_use]
    pub const fn rejected_count(self) -> u64 {
        self.rejected_count
    }

    /// Active fixture corpus size (opaque; not task quality).
    #[must_use]
    pub const fn active_fixture_count(self) -> u64 {
        self.active_fixture_count
    }

    /// Retained counterexample count (opaque; not task quality).
    #[must_use]
    pub const fn counterexample_count(self) -> u64 {
        self.counterexample_count
    }

    /// Full CEGIS stats snapshot (opaque; not task quality).
    #[must_use]
    pub const fn stats(self) -> CegisStats {
        self.stats
    }
}

/// Provenance-bound attachment of separately graded task-quality evidence to a
/// CEGIS outcome. Construction never consults survivor/rejection counts when
/// filling lanes.
#[derive(Debug, Clone, PartialEq)]
pub struct CegisTaskQualityAttachment {
    context: CegisRunContext,
    contract: TaskQualityContract,
    evidence: LaneSeparatedEvidence,
    surface: Option<AttentionSurfaceIdentity>,
    surface_fingerprint: Option<WorkloadFingerprint>,
}

impl CegisTaskQualityAttachment {
    /// Opaque CEGIS run context (never mapped into quality).
    #[must_use]
    pub const fn context(&self) -> CegisRunContext {
        self.context
    }

    /// Contract that validated the attached quality slots.
    #[must_use]
    pub const fn contract(&self) -> &TaskQualityContract {
        &self.contract
    }

    /// Explicitly graded lane-separated evidence.
    #[must_use]
    pub const fn evidence(&self) -> &LaneSeparatedEvidence {
        &self.evidence
    }

    /// Optional attention-surface identity bound at attach time.
    #[must_use]
    pub const fn surface(&self) -> Option<&AttentionSurfaceIdentity> {
        self.surface.as_ref()
    }

    /// Optional surface fingerprint bound at attach time.
    #[must_use]
    pub const fn surface_fingerprint(&self) -> Option<WorkloadFingerprint> {
        self.surface_fingerprint
    }

    /// Algorithmic-error lane from the attached evidence.
    #[must_use]
    pub const fn algorithmic(&self) -> AlgorithmicError {
        self.evidence.algorithmic()
    }

    /// Project into graduation objective inputs (quality only; A12 owns cost).
    ///
    /// # Errors
    ///
    /// Propagates schema mismatches from the lane adapter.
    pub fn to_graduation_objectives(
        &self,
        measured: MeasuredCost,
    ) -> Result<GraduationObjectives, GraduationError> {
        GraduationObjectives::from_lane_separated(measured, &self.contract, &self.evidence)
    }
}

/// Builder that requires a typed [`TaskQualityContract`] and separately graded
/// evidence. Survival-only and disguised diagnostic/cost payloads are rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct CegisTaskQualityAttachBuilder {
    contract: TaskQualityContract,
    context: CegisRunContext,
    expected_surface: Option<AttentionSurfaceIdentity>,
    claimed_surface: Option<AttentionSurfaceIdentity>,
}

impl CegisTaskQualityAttachBuilder {
    /// Start an attachment builder for one CEGIS outcome + contract.
    #[must_use]
    pub fn new(contract: TaskQualityContract, context: CegisRunContext) -> Self {
        Self {
            contract,
            context,
            expected_surface: None,
            claimed_surface: None,
        }
    }

    /// Require the attached evidence to match this surface identity.
    #[must_use]
    pub fn expect_surface(mut self, surface: AttentionSurfaceIdentity) -> Self {
        self.expected_surface = Some(surface);
        self
    }

    /// Record the surface identity the evidence claims to grade against.
    #[must_use]
    pub fn claim_surface(mut self, surface: AttentionSurfaceIdentity) -> Self {
        self.claimed_surface = Some(surface);
        self
    }

    /// Attach previously graded [`LaneSeparatedEvidence`].
    ///
    /// Requires:
    /// - quality slots match `contract` with present values;
    /// - algorithmic lane is populated (not empty);
    /// - algorithmic values pass the mechanistic-oracle provenance gate;
    /// - expected vs claimed surface fingerprints match when both are set.
    ///
    /// Never infers quality from `context` survivor/rejection/counterexample
    /// counts, cost fields, or ITD/TDI blobs.
    ///
    /// # Errors
    ///
    /// Returns a graduation error on missing lanes, schema mismatch, forbidden
    /// algorithmic provenance, or surface identity mismatch.
    pub fn attach_task_quality(
        self,
        evidence: LaneSeparatedEvidence,
    ) -> Result<CegisTaskQualityAttachment, GraduationError> {
        // Schema + present quality values (fail-closed on empty/mismatch).
        let _quality = accept_lane_separated_quality(&self.contract, &evidence)?;
        if self.contract.slots().is_empty() {
            return Err(GraduationError::TaskQualitySchemaMismatch);
        }
        if evidence.quality().is_empty() {
            return Err(GraduationError::MissingTaskQualityLane);
        }

        let algorithmic = evidence.algorithmic();
        if algorithmic.is_empty() {
            return Err(GraduationError::MissingAlgorithmicLane);
        }
        // Attachment path only accepts mechanistic/structural algorithmic fills.
        accept_algorithmic_error(
            AlgorithmicErrorSource::MechanisticOracleGrading,
            algorithmic,
        )
        .map_err(GraduationError::from)?;

        let (surface, surface_fingerprint) =
            resolve_surface_identity(self.expected_surface, self.claimed_surface)?;

        // Context is retained for provenance only; counts are never remapped.
        let _ = (
            self.context.survivor_count(),
            self.context.rejected_count(),
            self.context.counterexample_count(),
        );

        Ok(CegisTaskQualityAttachment {
            context: self.context,
            contract: self.contract,
            evidence,
            surface,
            surface_fingerprint,
        })
    }

    /// Materialize fills through the contract, then attach.
    ///
    /// Forbidden sources (ITD/TDI, cost, numerical diagnostic, CEGIS survival)
    /// fail closed inside contract materialization before attachment.
    ///
    /// # Errors
    ///
    /// Propagates materialization / lane / identity failures.
    pub fn attach_task_quality_fills(
        self,
        fills: &[TaskQualityFill],
        algorithmic: AlgorithmicError,
        correctness: CorrectnessStatus,
    ) -> Result<CegisTaskQualityAttachment, GraduationError> {
        let evidence = LaneSeparatedEvidence::bind(
            &self.contract,
            &LaneSeparatedEvidenceSpec {
                correctness,
                algorithmic,
                numerical: NumericalObjectives::default(),
                logical: LogicalCost::default(),
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: fills.to_vec(),
            },
        )?;
        self.attach_task_quality(evidence)
    }
}

/// Attach explicit lane evidence to a CEGIS outcome in one call.
///
/// # Errors
///
/// Same fail-closed conditions as
/// [`CegisTaskQualityAttachBuilder::attach_task_quality`].
pub fn attach_task_quality(
    context: CegisRunContext,
    contract: TaskQualityContract,
    evidence: LaneSeparatedEvidence,
) -> Result<CegisTaskQualityAttachment, GraduationError> {
    CegisTaskQualityAttachBuilder::new(contract, context).attach_task_quality(evidence)
}

/// Explicit non-path: a survival-only CEGIS payload cannot become an attachment.
///
/// # Errors
///
/// Always returns [`GraduationError::SurvivalIsNotTaskQuality`].
#[allow(unused_variables)]
pub fn attach_task_quality_from_survival_only(
    context: CegisRunContext,
    contract: &TaskQualityContract,
) -> Result<CegisTaskQualityAttachment, GraduationError> {
    let _ = (context, contract);
    Err(GraduationError::SurvivalIsNotTaskQuality)
}

fn resolve_surface_identity(
    expected: Option<AttentionSurfaceIdentity>,
    claimed: Option<AttentionSurfaceIdentity>,
) -> Result<
    (
        Option<AttentionSurfaceIdentity>,
        Option<WorkloadFingerprint>,
    ),
    GraduationError,
> {
    match (expected, claimed) {
        (None, None) => Ok((None, None)),
        (Some(surface), None) | (None, Some(surface)) => {
            let fingerprint = surface.fingerprint();
            Ok((Some(surface), Some(fingerprint)))
        }
        (Some(expected), Some(claimed)) => {
            if expected.fingerprint() != claimed.fingerprint() {
                return Err(GraduationError::AttachmentIdentityMismatch);
            }
            let fingerprint = expected.fingerprint();
            Ok((Some(expected), Some(fingerprint)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithmic_error_from_cegis_survival, reject_survival_fills,
        task_quality_from_cegis_survival,
    };
    use ada_objective::{
        CopyTokenAttentionTask, CopyTokenCandidate, MaskedPositionCandidate,
        MaskedPositionRetrievalTask, ObjectiveDirection, QualityValueSource,
        RelativeOffsetCandidate, RelativeOffsetSelectionTask, TaskQualitySlot,
    };
    use ada_workload::{
        AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskKind, MaskSpec,
        SequenceLengths, WorkloadMode,
    };

    fn surface_a() -> AttentionSurfaceIdentity {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 1, 2).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: Some(1),
            value_dimension: 1,
            topology: AttentionTopology::CrossAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        AttentionSurfaceIdentity::new(
            geometry,
            MaskSpec::new(MaskKind::None).unwrap(),
            WorkloadMode::TrainingForward,
        )
        .unwrap()
    }

    fn surface_b() -> AttentionSurfaceIdentity {
        let geometry = AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 2, 4).unwrap(),
            query_heads: 2,
            kv_heads: 2,
            qk_dimension: Some(2),
            value_dimension: 2,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap();
        AttentionSurfaceIdentity::new(
            geometry,
            MaskSpec::new(MaskKind::Causal).unwrap(),
            WorkloadMode::Decode,
        )
        .unwrap()
    }

    #[test]
    fn happy_path_attaches_mechanistic_fixture_evidence() {
        let task =
            MaskedPositionRetrievalTask::new(vec![1.0, 0.5], vec![3.0, 4.0], vec![true, true])
                .unwrap();
        let evidence = task
            .grade(
                MaskedPositionCandidate {
                    selected_index: 0,
                    selected_value: 3.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        let contract = MaskedPositionRetrievalTask::quality_contract().unwrap();
        let context = CegisRunContext::from_counts(3, 7, 2, 4);
        let attachment = CegisTaskQualityAttachBuilder::new(contract, context)
            .expect_surface(surface_a())
            .claim_surface(surface_a())
            .attach_task_quality(evidence.clone())
            .unwrap();

        assert_eq!(attachment.context().survivor_count(), 3);
        assert_eq!(attachment.context().rejected_count(), 7);
        assert_eq!(attachment.evidence().quality()[0].value(), Some(1.0));
        assert_eq!(attachment.algorithmic().selection_failures, Some(0));
        assert!(!attachment.algorithmic().is_empty());
        // Survival counts were not remapped into quality.
        assert_ne!(
            attachment.evidence().quality()[0].value(),
            Some(f64::from(
                u8::try_from(context.survivor_count()).unwrap_or(u8::MAX)
            ))
        );

        let objectives = attachment
            .to_graduation_objectives(MeasuredCost::default())
            .unwrap();
        assert_eq!(objectives.quality[0].name(), "exact_retrieval");
        assert_eq!(objectives.quality[0].value(), Some(1.0));
    }

    #[test]
    fn reject_survival_only_attachment() {
        let contract = MaskedPositionRetrievalTask::quality_contract().unwrap();
        let context = CegisRunContext::from_counts(5, 0, 1, 0);
        assert_eq!(
            attach_task_quality_from_survival_only(context, &contract),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
        assert_eq!(
            task_quality_from_cegis_survival(
                context.survivor_count(),
                context.rejected_count(),
                "exact_retrieval"
            ),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
        assert_eq!(
            algorithmic_error_from_cegis_survival(
                context.survivor_count(),
                context.rejected_count()
            ),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
    }

    #[test]
    fn reject_itd_tdi_cost_disguised_as_quality() {
        let contract = RelativeOffsetSelectionTask::quality_contract().unwrap();
        let context = CegisRunContext::from_counts(1, 1, 1, 1);
        let algorithmic = AlgorithmicError {
            support_mismatches: Some(0),
            selection_failures: Some(0),
            invariant_violations: Some(0),
        };
        for source in [
            QualityValueSource::ItdDiagnostic,
            QualityValueSource::TdiDiagnostic,
            QualityValueSource::LogicalCostField,
            QualityValueSource::EstimatedCostField,
            QualityValueSource::MeasuredCostField,
            QualityValueSource::NumericalDiagnostic,
            QualityValueSource::SurvivalOrCegisDisposition,
        ] {
            let fill = TaskQualityFill::new("exact_selection", 1.0, source).unwrap();
            let err = CegisTaskQualityAttachBuilder::new(contract.clone(), context)
                .attach_task_quality_fills(
                    std::slice::from_ref(&fill),
                    algorithmic,
                    CorrectnessStatus::Provisional,
                );
            assert!(
                matches!(err, Err(GraduationError::TaskQuality(_))),
                "source {source} must fail closed, got {err:?}"
            );
            if source == QualityValueSource::SurvivalOrCegisDisposition {
                assert_eq!(
                    reject_survival_fills(std::slice::from_ref(&fill)),
                    Err(GraduationError::SurvivalIsNotTaskQuality)
                );
            }
        }
    }

    #[test]
    fn reject_missing_algorithmic_lane() {
        let contract = TaskQualityContract::new(vec![
            TaskQualitySlot::new("exact_retrieval", ObjectiveDirection::Maximize).unwrap(),
        ])
        .unwrap();
        let fill = TaskQualityFill::new(
            "exact_retrieval",
            1.0,
            QualityValueSource::MechanisticFixtureEvaluation,
        )
        .unwrap();
        // Bind with empty algorithmic lane — materialization of quality succeeds,
        // but attach_task_quality must still refuse the missing algorithmic lane.
        let evidence = LaneSeparatedEvidence::bind(
            &contract,
            &LaneSeparatedEvidenceSpec {
                correctness: CorrectnessStatus::Provisional,
                algorithmic: AlgorithmicError::empty(),
                numerical: NumericalObjectives::default(),
                logical: LogicalCost::default(),
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: vec![fill],
            },
        )
        .unwrap();
        assert!(evidence.algorithmic().is_empty());
        assert_eq!(
            attach_task_quality(CegisRunContext::from_counts(0, 0, 0, 0), contract, evidence),
            Err(GraduationError::MissingAlgorithmicLane)
        );
    }

    #[test]
    fn reject_mismatched_surface_identity() {
        let task = CopyTokenAttentionTask::new(vec![1.0], vec![2.0]).unwrap();
        let evidence = task
            .grade(
                CopyTokenCandidate {
                    selected_index: 0,
                    selected_value: 2.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        let contract = CopyTokenAttentionTask::quality_contract().unwrap();
        assert_eq!(
            CegisTaskQualityAttachBuilder::new(contract, CegisRunContext::from_counts(1, 0, 1, 0))
                .expect_surface(surface_a())
                .claim_surface(surface_b())
                .attach_task_quality(evidence),
            Err(GraduationError::AttachmentIdentityMismatch)
        );
    }

    #[test]
    fn reject_schema_mismatch_across_fixture_contracts() {
        let offset = RelativeOffsetSelectionTask::new(vec![1.0, 2.0, 3.0], 1, 1).unwrap();
        let evidence = offset
            .grade(
                RelativeOffsetCandidate {
                    selected_index: 2,
                    selected_value: 3.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        let other_contract = MaskedPositionRetrievalTask::quality_contract().unwrap();
        assert_eq!(
            attach_task_quality(
                CegisRunContext::from_counts(2, 1, 1, 1),
                other_contract,
                evidence
            ),
            Err(GraduationError::TaskQualitySchemaMismatch)
        );
    }
}
