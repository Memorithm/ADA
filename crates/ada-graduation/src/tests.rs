use super::*;
use ada_a10_evidence_schema::SemanticEvidenceSpec;
use ada_cegis::{CegisConfig, CegisEngine};
use ada_core::{
    DiagnosticEvidenceKind, ImplementationCandidateId, SemanticFamily, SemanticId,
};
use ada_implementation::{
    AlgorithmPlan, Buffering, ExpStrategy, MemoryLevel, MemoryPlan, ReductionTopology,
    SchedulePlan, TileShape, WorkPartition,
};
use ada_objective::{ObjectiveDirection, QualityMetric};
use ada_qualification::{
    BoundedOracleQualification, NoAdversarialGenerator, SemanticWorkloadOracle,
};
use ada_search::{
    MAX_PROGRAM_COST, SearchBudget, SearchEngine, SemanticSearchConfig, SemanticSearchSpace,
};
use ada_semantic::{
    InputTransform, MaskRule, ReferenceInput, ReferenceInputSpec, SelectionRule, WeightRule,
};
use ada_workload::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskKind, MaskSpec,
    PrecisionPolicy, ScalarPrecision, SequenceLengths, WorkloadOptions,
};

fn workload() -> WorkloadContract {
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
    WorkloadContract::new(
        geometry,
        WorkloadOptions {
            mask: MaskSpec::new(MaskKind::None).unwrap(),
            precision: PrecisionPolicy::new(
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
            ),
            ..WorkloadOptions::default()
        },
    )
    .unwrap()
}

fn reference_input() -> ReferenceInput {
    ReferenceInput::new(ReferenceInputSpec {
        query_count: 1,
        key_count: 2,
        q_dimension: 1,
        value_dimension: 1,
        queries: vec![0.0],
        keys: vec![1.0, -1.0],
        values: vec![2.0, 4.0],
        external_mask: None,
    })
    .unwrap()
}

fn semantic_search() -> SearchEngine<SemanticSearchSpace> {
    let space = SemanticSearchSpace::new(SemanticSearchConfig {
        seed: 19,
        input_transforms: vec![InputTransform::Identity],
        affinity_scales: vec![1.0],
        masks: vec![MaskRule::Unmasked],
        selections: vec![SelectionRule::All],
        weights: vec![WeightRule::Softmax],
    })
    .unwrap();
    SearchEngine::new(space, SearchBudget::new(4, 4, MAX_PROGRAM_COST).unwrap()).unwrap()
}

fn cegis_result() -> CegisResult<SemanticProgram, SemanticWorkloadCase> {
    let case = SemanticWorkloadCase::new(
        workload(),
        reference_input(),
        "q=[0];k=[1,-1];v=[2,4]",
        vec![3.0],
        0.0,
    )
    .unwrap();
    CegisEngine::new(
        semantic_search(),
        SemanticWorkloadOracle,
        NoAdversarialGenerator,
        CegisConfig::default(),
        vec![case.into_fixture("equal-score-control").unwrap()],
    )
    .unwrap()
    .run_to_end()
    .unwrap()
}

fn evidence(
    semantic: SemanticId,
    workload: EvidenceWorkloadFingerprint,
    kind: DiagnosticEvidenceKind,
    artifact: &str,
) -> SemanticEvidenceRecord {
    SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic,
        workload,
        kind,
        producer_repository: "Memorithm/ADA".into(),
        producer_revision: "a".repeat(40),
        artifact_identity: artifact.into(),
        intervention_identity: None,
        observation_horizon: None,
        metric_identity: "graduation-test-v1".into(),
        sha256_evidence: "b".repeat(64),
        metrics: vec![("score".into(), 1.0)],
    })
    .unwrap()
}

fn qualified(
    result: &CegisResult<SemanticProgram, SemanticWorkloadCase>,
    kinds: &[DiagnosticEvidenceKind],
) -> EvidenceBoundQualification {
    let survivor = &result.survivors()[0];
    let oracle = BoundedOracleQualification::from_cegis_result(
        result,
        survivor.fingerprint(),
        &workload(),
    )
    .unwrap();
    let semantic = survivor.candidate().descriptor().id().clone();
    let workload_fingerprint = oracle.workload_fingerprint();
    let records = kinds
        .iter()
        .enumerate()
        .map(|(index, &kind)| {
            evidence(
                semantic.clone(),
                workload_fingerprint,
                kind,
                &format!("artifact-{index}"),
            )
        })
        .collect();
    oracle.attach_evidence(records).unwrap()
}

fn implementation(semantic: SemanticId) -> ImplementationPlan {
    let id = ImplementationCandidateId::new(semantic, "reference-blocked", 1).unwrap();
    ImplementationPlan::new(
        id,
        AlgorithmPlan::DenseBlocked,
        SchedulePlan {
            tile: TileShape {
                queries: 1,
                keys: 2,
                values: 1,
            },
            partition: WorkPartition::Serial,
            reduction: ReductionTopology::Serial,
            exp_strategy: ExpStrategy::Standard,
            pipeline_stages: 1,
            vector_width: 1,
            buffering: Buffering::Single,
        },
        MemoryPlan {
            query: MemoryLevel::Global,
            key: MemoryLevel::Global,
            value: MemoryLevel::Global,
            output: MemoryLevel::Global,
            accumulator: MemoryLevel::Register,
            workspace_bytes: 0,
            alignment_bytes: 8,
            kv_page_rows: None,
        },
    )
    .unwrap()
}

fn objective_input(measured: MeasuredCost) -> GraduationObjectives {
    GraduationObjectives {
        measured,
        quality: vec![
            QualityMetric::new("task_accuracy", Some(0.75), ObjectiveDirection::Maximize)
                .unwrap(),
        ],
    }
}

fn bundle(
    result: &CegisResult<SemanticProgram, SemanticWorkloadCase>,
    measured: MeasuredCost,
    verdict: QualificationVerdict,
    kinds: &[DiagnosticEvidenceKind],
) -> Result<FlatGraduationBundle, GraduationError> {
    let qualification = qualified(result, kinds);
    let semantic = qualification
        .oracle()
        .candidate()
        .candidate()
        .descriptor()
        .id()
        .clone();
    FlatGraduationBundle::new(
        &qualification,
        result,
        implementation(semantic),
        OperationProfile::scaled_dot_softmax(1).unwrap(),
        CostAssumptions::default(),
        objective_input(measured),
        verdict,
    )
}

#[test]
fn continue_research_bundle_round_trips_with_exact_oracle_and_a12_cost() {
    let result = cegis_result();
    let bundle = bundle(
        &result,
        MeasuredCost::default(),
        QualificationVerdict::ContinueResearch,
        &[DiagnosticEvidenceKind::TaskBehavior],
    )
    .unwrap();
    assert_eq!(bundle.oracle_fixtures().len(), 1);
    assert_eq!(bundle.objectives().logical().qk_evaluations, Some(2));
    assert_eq!(
        bundle.objectives().correctness(),
        CorrectnessStatus::Provisional
    );
    assert_eq!(
        bundle.objectives().numerical(),
        NumericalObjectives::default()
    );
    assert!(bundle.objectives().measured().latency_ns.is_none());
    let text = bundle.to_canonical_text();
    let decoded = FlatGraduationBundle::from_canonical_text(&text).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(decoded.to_canonical_text(), text);
    assert_eq!(decoded.candidate_key().unwrap(), bundle.candidate_key().unwrap());
}

#[test]
fn bounded_oracle_qualification_cannot_self_promote_to_adopt_or_adapt() {
    let result = cegis_result();
    for verdict in [QualificationVerdict::Adopt, QualificationVerdict::Adapt] {
        assert_eq!(
            bundle(
                &result,
                MeasuredCost::default(),
                verdict,
                &[DiagnosticEvidenceKind::TaskBehavior],
            )
            .unwrap_err(),
            GraduationError::VerdictRequiresQualifiedCorrectness
        );
    }
}

#[test]
fn measured_cost_requires_hardware_evidence() {
    let result = cegis_result();
    let measured = MeasuredCost {
        latency_ns: Some(1234),
        energy_nj: None,
    };
    assert_eq!(
        bundle(
            &result,
            measured,
            QualificationVerdict::ContinueResearch,
            &[DiagnosticEvidenceKind::TaskBehavior],
        )
        .unwrap_err(),
        GraduationError::MissingHardwareEvidence
    );
    bundle(
        &result,
        measured,
        QualificationVerdict::ContinueResearch,
        &[
            DiagnosticEvidenceKind::TaskBehavior,
            DiagnosticEvidenceKind::HardwareCost,
        ],
    )
    .unwrap();
}

#[test]
fn implementation_for_another_semantic_is_rejected() {
    let result = cegis_result();
    let qualification = qualified(&result, &[DiagnosticEvidenceKind::TaskBehavior]);
    let wrong = SemanticId::new(SemanticFamily::Experimental, "wrong-semantic", 1).unwrap();
    assert_eq!(
        FlatGraduationBundle::new(
            &qualification,
            &result,
            implementation(wrong),
            OperationProfile::scaled_dot_softmax(1).unwrap(),
            CostAssumptions::default(),
            objective_input(MeasuredCost::default()),
            QualificationVerdict::ContinueResearch,
        )
        .unwrap_err(),
        GraduationError::SemanticImplementationMismatch
    );
}

#[test]
fn decoder_recomputes_a12_cost_and_rejects_tampered_objectives() {
    let result = cegis_result();
    let mut bundle = bundle(
        &result,
        MeasuredCost::default(),
        QualificationVerdict::ContinueResearch,
        &[DiagnosticEvidenceKind::TaskBehavior],
    )
    .unwrap();
    let mut estimated = bundle.objectives.estimated();
    estimated.workspace_bytes = Some(1);
    bundle.objectives = bundle
        .objectives
        .clone()
        .with_estimated(estimated)
        .unwrap();
    let text = bundle.to_canonical_text();
    assert!(matches!(
        FlatGraduationBundle::from_canonical_text(&text),
        Err(GraduationError::MalformedCanonical(_))
    ));
}

#[test]
fn observed_quality_requires_task_behavior_provenance() {
    let result = cegis_result();
    assert_eq!(
        bundle(
            &result,
            MeasuredCost::default(),
            QualificationVerdict::ContinueResearch,
            &[DiagnosticEvidenceKind::StaticOperator],
        )
        .unwrap_err(),
        GraduationError::MissingTaskEvidence
    );
}
