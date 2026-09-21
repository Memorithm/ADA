//! Thin fail-closed adapter from lane-separated task quality into graduation.
//!
//! Non-claims:
//! - accepting [`LaneSeparatedEvidence`] does not adopt, promote to FLAT, or
//!   claim usefulness/novelty;
//! - CEGIS survival / rejection counts cannot fill task-quality slots;
//! - ITD/TDI diagnostics and cost fields remain rejected by the contract
//!   materializer before they reach this adapter;
//! - algorithmic / numerical / cost lanes on the evidence packet are not copied
//!   into objective quality metrics here.

use ada_objective::{
    LaneSeparatedEvidence, MeasuredCost, QualityMetric, QualityValueSource, TaskQualityContract,
    TaskQualityError, TaskQualityFill,
};

use crate::{GraduationError, GraduationObjectives};

impl From<TaskQualityError> for GraduationError {
    fn from(value: TaskQualityError) -> Self {
        Self::TaskQuality(value.to_string())
    }
}

impl GraduationObjectives {
    /// Build graduation objective inputs whose quality metrics come only from a
    /// previously bound [`LaneSeparatedEvidence`] packet.
    ///
    /// The packet must already satisfy `contract` (same slot names/directions in
    /// order). Algorithmic, numerical, and cost lanes on the packet are ignored
    /// here: A12 still owns logical/estimated cost inside the graduation bundle.
    /// Algorithmic error is an `ObjectiveVector` codec lane but is not copied into
    /// graduation quality metrics by this adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when the evidence quality schema does not match the
    /// contract.
    pub fn from_lane_separated(
        measured: MeasuredCost,
        contract: &TaskQualityContract,
        evidence: &LaneSeparatedEvidence,
    ) -> Result<Self, GraduationError> {
        let quality = accept_lane_separated_quality(contract, evidence)?;
        Ok(Self { measured, quality })
    }
}

/// Validate and extract task-quality metrics from lane-separated evidence.
///
/// # Errors
///
/// Returns [`GraduationError::TaskQualitySchemaMismatch`] when slot names or
/// directions disagree with `contract`, or when a quality value is missing.
pub fn accept_lane_separated_quality(
    contract: &TaskQualityContract,
    evidence: &LaneSeparatedEvidence,
) -> Result<Vec<QualityMetric>, GraduationError> {
    contract.validate()?;
    let slots = contract.slots();
    let quality = evidence.quality();
    if slots.len() != quality.len() {
        return Err(GraduationError::TaskQualitySchemaMismatch);
    }
    for (slot, metric) in slots.iter().zip(quality.iter()) {
        if slot.name() != metric.name() || slot.direction() != metric.direction() {
            return Err(GraduationError::TaskQualitySchemaMismatch);
        }
        if metric.value().is_none() {
            return Err(GraduationError::TaskQualitySchemaMismatch);
        }
    }
    Ok(quality.to_vec())
}

/// Materialize task-quality fills through a contract for graduation inputs.
///
/// Forbidden sources (ITD/TDI, cost, numerical diagnostics, CEGIS survival) are
/// rejected by [`TaskQualityContract::materialize_quality`].
///
/// # Errors
///
/// Propagates contract materialization failures.
pub fn materialize_graduation_quality(
    contract: &TaskQualityContract,
    fills: &[TaskQualityFill],
) -> Result<Vec<QualityMetric>, GraduationError> {
    Ok(contract.materialize_quality(fills)?)
}

/// Explicit non-mapping: CEGIS survivor/rejection counts never become
/// task-quality fills.
///
/// Call sites that need task quality must grade a mechanistic fixture or task
/// oracle into [`LaneSeparatedEvidence`] separately. Survivor count is not
/// accuracy; rejection count is not a loss metric.
///
/// # Errors
///
/// Always returns [`GraduationError::SurvivalIsNotTaskQuality`].
#[allow(unused_variables)]
pub fn task_quality_from_cegis_survival(
    survivors: u64,
    rejected: u64,
    slot: &str,
) -> Result<TaskQualityFill, GraduationError> {
    // Parameters document the forbidden mapping at the call site; no numeric
    // remapping to task quality is defined.
    Err(GraduationError::SurvivalIsNotTaskQuality)
}

/// Explicit non-mapping: CEGIS survivor/rejection counts never become
/// algorithmic-error lane values (nor task quality).
///
/// # Errors
///
/// Always returns [`GraduationError::SurvivalIsNotTaskQuality`].
#[allow(unused_variables)]
pub fn algorithmic_error_from_cegis_survival(
    survivors: u64,
    rejected: u64,
) -> Result<ada_objective::AlgorithmicError, GraduationError> {
    Err(GraduationError::SurvivalIsNotTaskQuality)
}

/// Reject any fill whose provenance is CEGIS survival / disposition.
///
/// # Errors
///
/// Returns [`GraduationError::SurvivalIsNotTaskQuality`] when a forbidden
/// survival source is present.
pub fn reject_survival_fills(fills: &[TaskQualityFill]) -> Result<(), GraduationError> {
    for fill in fills {
        if fill.source() == QualityValueSource::SurvivalOrCegisDisposition {
            return Err(GraduationError::SurvivalIsNotTaskQuality);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ada_objective::{
        AlgorithmicError, CopyTokenAttentionTask, CopyTokenCandidate, CorrectnessStatus,
        EstimatedCost, LaneSeparatedEvidenceSpec, LogicalCost, MaskedPositionCandidate,
        MaskedPositionRetrievalTask, NumericalObjectives, ObjectiveDirection,
        RelativeOffsetCandidate, RelativeOffsetSelectionTask, TaskQualitySlot,
    };

    #[test]
    fn lane_evidence_from_fixtures_is_accepted_for_graduation_objectives() {
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
        let objectives = GraduationObjectives::from_lane_separated(
            MeasuredCost::default(),
            &MaskedPositionRetrievalTask::quality_contract().unwrap(),
            &evidence,
        )
        .unwrap();
        assert_eq!(objectives.quality.len(), 1);
        assert_eq!(objectives.quality[0].name(), "exact_retrieval");
        assert_eq!(objectives.quality[0].value(), Some(1.0));
        // Algorithmic lane stayed off the quality vector.
        assert_eq!(evidence.algorithmic().selection_failures, Some(0));
    }

    #[test]
    fn cegis_survival_cannot_become_task_quality_fills() {
        assert_eq!(
            task_quality_from_cegis_survival(3, 7, "exact_retrieval"),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
        let fill = TaskQualityFill::new(
            "exact_retrieval",
            1.0,
            QualityValueSource::SurvivalOrCegisDisposition,
        )
        .unwrap();
        assert_eq!(
            reject_survival_fills(std::slice::from_ref(&fill)),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
        let contract = TaskQualityContract::new(vec![
            TaskQualitySlot::new("exact_retrieval", ObjectiveDirection::Maximize).unwrap(),
        ])
        .unwrap();
        assert!(matches!(
            materialize_graduation_quality(&contract, &[fill]),
            Err(GraduationError::TaskQuality(_))
        ));
    }

    #[test]
    fn schema_mismatch_and_cost_source_fail_closed() {
        let contract = RelativeOffsetSelectionTask::quality_contract().unwrap();
        let other = CopyTokenAttentionTask::new(vec![1.0], vec![2.0]).unwrap();
        let evidence = other
            .grade(
                CopyTokenCandidate {
                    selected_index: 0,
                    selected_value: 2.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        assert_eq!(
            accept_lane_separated_quality(&contract, &evidence),
            Err(GraduationError::TaskQualitySchemaMismatch)
        );

        let cost_fill =
            TaskQualityFill::new("exact_selection", 1.0, QualityValueSource::LogicalCostField)
                .unwrap();
        assert!(matches!(
            materialize_graduation_quality(&contract, &[cost_fill]),
            Err(GraduationError::TaskQuality(_))
        ));

        // Empty quality against a non-empty contract is also rejected.
        let empty = LaneSeparatedEvidence::bind(
            &TaskQualityContract::new(Vec::new()).unwrap(),
            &LaneSeparatedEvidenceSpec {
                correctness: CorrectnessStatus::Unknown,
                algorithmic: AlgorithmicError::empty(),
                numerical: NumericalObjectives::default(),
                logical: LogicalCost::default(),
                estimated: EstimatedCost::default(),
                measured: MeasuredCost::default(),
                fills: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(
            accept_lane_separated_quality(&contract, &empty),
            Err(GraduationError::TaskQualitySchemaMismatch)
        );

        let offset = RelativeOffsetSelectionTask::new(vec![1.0, 2.0, 3.0], 1, 1).unwrap();
        let ok = offset
            .grade(
                RelativeOffsetCandidate {
                    selected_index: 2,
                    selected_value: 3.0,
                },
                LogicalCost::default(),
            )
            .unwrap();
        assert_eq!(
            accept_lane_separated_quality(&contract, &ok).unwrap()[0].value(),
            Some(1.0)
        );
    }

    #[test]
    fn cegis_survival_cannot_become_algorithmic_error_or_task_quality() {
        assert_eq!(
            algorithmic_error_from_cegis_survival(3, 7),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
        assert_eq!(
            task_quality_from_cegis_survival(3, 7, "exact_retrieval"),
            Err(GraduationError::SurvivalIsNotTaskQuality)
        );
    }
}
