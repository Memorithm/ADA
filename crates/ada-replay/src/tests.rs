use super::{
    ReplayCaseSpec, ReplayError, ReplayReferenceInput, decode_graduation_fixtures,
    verify_ada_reference_replay,
};
use ada_a10_evidence_schema::{
    EvidenceWorkloadFingerprint, SemanticEvidenceRecord, SemanticEvidenceSpec,
};
use ada_cegis::{CegisConfig, CegisEngine};
use ada_core::{DiagnosticEvidenceKind, ImplementationCandidateId, QualificationVerdict, SemanticId};
use ada_cost_model::{CostAssumptions, OperationProfile};
use ada_graduation::{FlatGraduationBundle, GraduationObjectives};
use ada_implementation::{
    AlgorithmPlan, Buffering, ExpStrategy, ImplementationPlan, MemoryLevel, MemoryPlan,
    ReductionTopology, SchedulePlan, TileShape, WorkPartition,
};
use ada_objective::{MeasuredCost, ObjectiveDirection, QualityMetric};
use ada_qualification::{
    BoundedOracleQualification, EvidenceBoundQualification, NoAdversarialGenerator,
    SemanticWorkloadCase, SemanticWorkloadOracle,
};
use ada_search::{
    MAX_PROGRAM_COST, SearchBudget, SearchEngine, SemanticSearchConfig, SemanticSearchSpace,
};
use ada_semantic::{
    InputTransform, MaskRule, ReferenceInput, ReferenceInputSpec, SelectionRule, SemanticProgram,
    WeightRule,
};
use ada_workload::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, MaskKind, MaskSpec,
    PrecisionPolicy, ScalarPrecision, SequenceLengths, WorkloadContract, WorkloadOptions,
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

fn replay_input(query: f64) -> ReplayReferenceInput {
    ReplayReferenceInput::new(ReferenceInputSpec {
        query_count: 1,
        key_count: 2,
        q_dimension: 1,
        value_dimension: 1,
        queries: vec![query],
        keys: vec![1.0, -1.0],
        values: vec![2.0, 4.0],
        external_mask: None,
    })
    .unwrap()
}

fn search() -> SearchEngine<SemanticSearchSpace> {
    let space = SemanticSearchSpace::new(SemanticSearchConfig {
        seed: 29,
        input_transforms: vec![InputTransform::Identity],
        affinity_scales: vec![1.0],
        masks: vec![MaskRule::Unmasked],
        selections: vec![SelectionRule::All],
        weights: vec![WeightRule::Softmax],
    })
    .unwrap();
    SearchEngine::new(space, SearchBudget::new(4, 4, MAX_PROGRAM_COST).unwrap()).unwrap()
}

fn replayable_cegis() -> ada_cegis::CegisResult<SemanticProgram, SemanticWorkloadCase> {
    let fixture = ReplayCaseSpec {
        workload: workload(),
        input: replay_input(0.0),
        expected_output: vec![3.0],
        max_abs_tolerance: 0.0,
    }
    .into_fixture("equal-score-control")
    .unwrap();
    CegisEngine::new(
        search(),
        SemanticWorkloadOracle,
        NoAdversarialGenerator,
        CegisConfig::default(),
        vec![fixture],
    )
    .unwrap()
    .run_to_end()
    .unwrap()
}

fn opaque_cegis() -> ada_cegis::CegisResult<SemanticProgram, SemanticWorkloadCase> {
    let input = ReferenceInput::new(ReferenceInputSpec {
        query_count: 1,
        key_count: 2,
        q_dimension: 1,
        value_dimension: 1,
        queries: vec![0.0],
        keys: vec![1.0, -1.0],
        values: vec![2.0, 4.0],
        external_mask: None,
    })
    .unwrap();
    let fixture = SemanticWorkloadCase::new(workload(), input, "opaque", vec![3.0], 0.0)
        .unwrap()
        .into_fixture("legacy-opaque")
        .unwrap();
    CegisEngine::new(
        search(),
        SemanticWorkloadOracle,
        NoAdversarialGenerator,
        CegisConfig::default(),
        vec![fixture],
    )
    .unwrap()
    .run_to_end()
    .unwrap()
}

fn evidence(
    semantic: SemanticId,
    workload_fingerprint: EvidenceWorkloadFingerprint,
) -> SemanticEvidenceRecord {
    SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic,
        workload: workload_fingerprint,
        kind: DiagnosticEvidenceKind::TaskBehavior,
        producer_repository: "Memorithm/ADA".into(),
        producer_revision: "a".repeat(40),
        artifact_identity: "replay-test-v1".into(),
        intervention_identity: None,
        observation_horizon: None,
        metric_identity: "exact-replay-v1".into(),
        sha256_evidence: "b".repeat(64),
        metrics: vec![("fixture_pass".into(), 1.0)],
    })
    .unwrap()
}

fn qualified(
    result: &ada_cegis::CegisResult<SemanticProgram, SemanticWorkloadCase>,
) -> EvidenceBoundQualification {
    let survivor = &result.survivors()[0];
    let oracle = BoundedOracleQualification::from_cegis_result(
        result,
        survivor.fingerprint(),
        &workload(),
    )
    .unwrap();
    let record = evidence(
        survivor.candidate().descriptor().id().clone(),
        oracle.workload_fingerprint(),
    );
    oracle.attach_evidence(vec![record]).unwrap()
}

fn implementation(semantic: SemanticId) -> ImplementationPlan {
    ImplementationPlan::new(
        ImplementationCandidateId::new(semantic, "reference-blocked", 1).unwrap(),
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

fn graduation(
    result: &ada_cegis::CegisResult<SemanticProgram, SemanticWorkloadCase>,
) -> FlatGraduationBundle {
    let qualification = qualified(result);
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
        GraduationObjectives {
            measured: MeasuredCost::default(),
            quality: vec![
                QualityMetric::new("fixture_pass", Some(1.0), ObjectiveDirection::Maximize).unwrap(),
            ],
        },
        QualificationVerdict::ContinueResearch,
    )
    .unwrap()
}

#[test]
fn exact_reference_input_round_trips_bits_and_mask() {
    let original = ReplayReferenceInput::new(ReferenceInputSpec {
        query_count: 2,
        key_count: 2,
        q_dimension: 2,
        value_dimension: 1,
        queries: vec![-0.0, f64::from_bits(1), 1.5, -2.0],
        keys: vec![3.0, -4.0, 5.0, 6.0],
        values: vec![7.0, -8.0],
        external_mask: Some(vec![true, false, false, true]),
    })
    .unwrap();
    let text = original.to_canonical_text();
    let decoded = ReplayReferenceInput::from_canonical_text(&text).unwrap();
    assert_eq!(decoded.to_canonical_text(), text);
    assert_eq!(decoded.queries()[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(decoded.queries()[1].to_bits(), 1);
    assert_eq!(decoded.external_mask(), Some([true, false, false, true].as_slice()));
}

#[test]
fn replayable_fixture_fingerprint_binds_exact_query_bits() {
    let first = ReplayCaseSpec {
        workload: workload(),
        input: replay_input(0.0),
        expected_output: vec![3.0],
        max_abs_tolerance: 0.0,
    }
    .into_fixture("same-id")
    .unwrap();
    let second = ReplayCaseSpec {
        workload: workload(),
        input: replay_input(f64::from_bits(1)),
        expected_output: vec![3.0],
        max_abs_tolerance: 0.0,
    }
    .into_fixture("same-id")
    .unwrap();
    assert_ne!(first.fingerprint(), second.fingerprint());
    assert_ne!(first.canonical_text(), second.canonical_text());
}

#[test]
fn graduated_replayable_fixture_reconstructs_and_replays_exactly() {
    let result = replayable_cegis();
    let bundle = graduation(&result);
    let cases = decode_graduation_fixtures(&bundle).unwrap();
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].id(), "equal-score-control");
    assert_eq!(cases[0].input().queries()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(cases[0].expected_output(), &[3.0]);
    let report = verify_ada_reference_replay(&bundle).unwrap();
    assert_eq!(report.fixture_count(), 1);
    assert_eq!(report.worst_max_abs_error().to_bits(), 0.0_f64.to_bits());
}

#[test]
fn legacy_opaque_fixture_is_valid_qualification_but_not_replayable() {
    let result = opaque_cegis();
    let bundle = graduation(&result);
    assert_eq!(
        decode_graduation_fixtures(&bundle).unwrap_err(),
        ReplayError::NonReplayableFixture
    );
}

#[test]
fn decoder_rejects_nonfinite_bit_pattern_and_noncanonical_trailing_field() {
    let text = replay_input(0.0).to_canonical_text();
    let nonfinite = text.replacen("0000000000000000", "7ff0000000000000", 1);
    assert!(ReplayReferenceInput::from_canonical_text(&nonfinite).is_err());

    let mut trailing = text;
    trailing.push_str("extra=1\n");
    assert!(ReplayReferenceInput::from_canonical_text(&trailing).is_err());
}
