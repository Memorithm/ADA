use crate::{
    EvidenceBinding, ExperimentError, ExperimentIndex, ExperimentRecord, ExperimentSpec,
    ProducerProvenance,
};
use ada_core::{
    DiagnosticEvidenceKind, DiagnosticEvidenceRef, ImplementationCandidateId, SemanticFamily,
    SemanticId,
};
use ada_implementation::{
    AlgorithmPlan, Buffering, ExpStrategy, ImplementationPlan, MemoryLevel, MemoryPlan,
    ReductionTopology, SchedulePlan, TileShape, WorkPartition,
};
use ada_objective::{
    CorrectnessStatus, EstimatedCost, LogicalCost, MeasuredCost, NumericalObjectives,
    ObjectiveDirection, ObjectiveVector, QualityMetric,
};
use ada_semantic::{MaskRule, SelectionRule, SemanticProgram};
use ada_workload::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, PrecisionPolicy,
    ScalarPrecision, SequenceLengths, WorkloadContract, WorkloadOptions,
};

fn components() -> (SemanticProgram, WorkloadContract, ImplementationPlan) {
    let semantic_id =
        SemanticId::new(SemanticFamily::StandardSoftmax, "experiment-softmax", 1).unwrap();
    let semantic = SemanticProgram::standard_softmax(
        semantic_id.clone(),
        MaskRule::Unmasked,
        SelectionRule::All,
        1.0,
    )
    .unwrap();
    let geometry = AttentionGeometry::new(GeometrySpec {
        sequence_lengths: SequenceLengths::uniform(1, 2, 2).unwrap(),
        query_heads: 1,
        kv_heads: 1,
        qk_dimension: Some(4),
        value_dimension: 4,
        topology: AttentionTopology::SelfAttention,
        head_grouping: HeadGrouping::MultiHead,
    })
    .unwrap();
    let workload = WorkloadContract::new(
        geometry,
        WorkloadOptions {
            precision: PrecisionPolicy::new(
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
            ),
            ..WorkloadOptions::default()
        },
    )
    .unwrap();
    let implementation = ImplementationPlan::new(
        ImplementationCandidateId::new(semantic_id, "blocked", 1).unwrap(),
        AlgorithmPlan::DenseBlocked,
        SchedulePlan {
            tile: TileShape {
                queries: 2,
                keys: 2,
                values: 4,
            },
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
    )
    .unwrap();
    (semantic, workload, implementation)
}

fn provenance() -> ProducerProvenance {
    ProducerProvenance::new(
        "Memorithm/ADA",
        "0123456789abcdef0123456789abcdef01234567",
        "unit-fixture",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    )
    .unwrap()
}

fn evidence(kind: DiagnosticEvidenceKind, artifact: &str) -> EvidenceBinding {
    let reference =
        DiagnosticEvidenceRef::new(kind, "Memorithm/ADA", artifact, "git:0123456789abcdef")
            .unwrap();
    EvidenceBinding::from_reference(&reference).unwrap()
}

#[test]
fn experiment_round_trip_binds_all_identity_layers() {
    let (semantic, workload, implementation) = components();
    let objective = ObjectiveVector::from_parts(
        CorrectnessStatus::Provisional,
        NumericalObjectives {
            max_abs_error: Some(0.0),
            ..NumericalObjectives::default()
        },
        LogicalCost {
            flops: Some(128),
            ..LogicalCost::default()
        },
        EstimatedCost {
            bytes_moved: Some(1_024),
            ..EstimatedCost::default()
        },
        MeasuredCost::default(),
        Vec::new(),
    )
    .unwrap();
    let record = ExperimentRecord::new(ExperimentSpec {
        semantic,
        workload,
        implementation,
        objective,
        provenance: provenance(),
        evidence: vec![evidence(
            DiagnosticEvidenceKind::LogicalCost,
            "logical-cost",
        )],
    })
    .unwrap();
    let text = record.to_canonical_text();
    let decoded = ExperimentRecord::from_canonical_text(&text).unwrap();
    assert_eq!(decoded, record);
    assert_eq!(decoded.fingerprint(), record.fingerprint());
}

#[test]
fn measured_cost_fails_closed_without_hardware_evidence() {
    let (semantic, workload, implementation) = components();
    let objective = ObjectiveVector::new(CorrectnessStatus::Provisional)
        .with_measured(MeasuredCost {
            latency_ns: Some(42),
            energy_nj: None,
        })
        .unwrap();
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
    let objective = ObjectiveVector::new(CorrectnessStatus::Provisional)
        .with_quality(vec![quality])
        .unwrap();
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
    let workload_fingerprint = workload.fingerprint();
    let implementation_id = implementation.id().clone();
    let objective = ObjectiveVector::new(CorrectnessStatus::Provisional)
        .with_measured(MeasuredCost {
            latency_ns: Some(42),
            energy_nj: Some(7),
        })
        .unwrap();
    let record = ExperimentRecord::new(ExperimentSpec {
        semantic,
        workload,
        implementation,
        objective,
        provenance: provenance(),
        evidence: vec![evidence(DiagnosticEvidenceKind::HardwareCost, "hardware")],
    })
    .unwrap();
    let mut index = ExperimentIndex::new();
    let fingerprint = index.insert(record).unwrap();
    assert_eq!(index.records_for_semantic(&semantic_id).len(), 1);
    assert_eq!(index.records_for_workload(workload_fingerprint).len(), 1);
    assert_eq!(
        index.records_for_implementation(&implementation_id).len(),
        1
    );
    assert_eq!(index.records_with_measured_cost().len(), 1);
    assert!(index.get(fingerprint).is_some());
    let text = index.to_canonical_text();
    let decoded = ExperimentIndex::from_canonical_text(&text).unwrap();
    assert_eq!(decoded.to_canonical_text(), text);
}
