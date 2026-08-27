use ada_core::{DiagnosticEvidenceKind, SemanticFamily, SemanticId};
use ada_workload::{
    AttentionGeometry, AttentionTopology, GeometrySpec, HeadGrouping, InputRepresentation,
    MaskKind, MaskSpec, PrecisionPolicy, ScalarPrecision, SequenceLengths, StateSpec,
    WorkloadContract, WorkloadMode, WorkloadOptions,
};

use super::{
    EvidenceWorkloadFingerprint, SemanticEvidenceError, SemanticEvidenceRecord,
    SemanticEvidenceSpec,
};

fn workload(mask: MaskKind) -> WorkloadContract {
    WorkloadContract::new(
        AttentionGeometry::new(GeometrySpec {
            sequence_lengths: SequenceLengths::uniform(1, 3, 3).unwrap(),
            query_heads: 1,
            kv_heads: 1,
            qk_dimension: None,
            value_dimension: 1,
            topology: AttentionTopology::SelfAttention,
            head_grouping: HeadGrouping::MultiHead,
        })
        .unwrap(),
        WorkloadOptions {
            mode: WorkloadMode::Prefill,
            mask: MaskSpec::new(mask).unwrap(),
            precision: PrecisionPolicy::new(
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
                ScalarPrecision::F64,
            ),
            inputs: InputRepresentation::PrecomputedScores {
                identity: "ada-a11-e1-fixed-mixer".into(),
            },
            state: StateSpec::Stateless,
            ..WorkloadOptions::default()
        },
    )
    .unwrap()
}

fn semantic() -> SemanticId {
    SemanticId::new(
        SemanticFamily::Experimental,
        "balanced-three-token-mixer",
        1,
    )
    .unwrap()
}

fn golden_tdi() -> SemanticEvidenceRecord {
    SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic: semantic(),
        workload: EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::Bidirectional)),
        kind: DiagnosticEvidenceKind::TdiRecovery,
        producer_repository: "Memorithm/TDI".into(),
        producer_revision: "a".repeat(40),
        artifact_identity: "tdi-ai-gate-b".into(),
        intervention_identity: Some("balanced-antisymmetric-mode".into()),
        observation_horizon: Some(3),
        metric_identity: "reciprocal-linf-recovery".into(),
        sha256_evidence: "b".repeat(64),
        metrics: vec![("recovery_h3".into(), 8.0 / 9.0), ("linf_h3".into(), 0.125)],
    })
    .unwrap()
}

fn assert_metric_bits(actual: &[(String, f64)], expected: &[(String, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for ((actual_name, actual_value), (expected_name, expected_value)) in
        actual.iter().zip(expected)
    {
        assert_eq!(actual_name, expected_name);
        assert_eq!(actual_value.to_bits(), expected_value.to_bits());
    }
}

#[test]
fn tdi_record_round_trips_canonically_and_preserves_exact_metric_bits() {
    let record = golden_tdi();
    let text = record.to_canonical_text();
    let decoded = SemanticEvidenceRecord::from_canonical_text(&text).unwrap();

    assert_eq!(decoded.to_canonical_text(), text);
    assert_eq!(decoded.semantic(), record.semantic());
    assert_eq!(decoded.workload(), record.workload());
    assert_eq!(decoded.kind(), DiagnosticEvidenceKind::TdiRecovery);
    assert_eq!(decoded.producer_repository(), "Memorithm/TDI");
    assert_eq!(
        decoded.intervention_identity(),
        Some("balanced-antisymmetric-mode")
    );
    assert_eq!(decoded.observation_horizon(), Some(3));
    assert_metric_bits(decoded.metrics(), record.metrics());
}

#[test]
fn workload_binding_changes_when_experimental_visibility_changes() {
    let bidirectional =
        EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::Bidirectional));
    let unmasked = EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::None));
    assert_ne!(bidirectional, unmasked);
}

#[test]
fn tdi_recovery_requires_intervention_and_horizon() {
    let mut spec = SemanticEvidenceSpec {
        semantic: semantic(),
        workload: EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::Bidirectional)),
        kind: DiagnosticEvidenceKind::TdiRecovery,
        producer_repository: "Memorithm/TDI".into(),
        producer_revision: "a".repeat(40),
        artifact_identity: "gate-b".into(),
        intervention_identity: None,
        observation_horizon: Some(3),
        metric_identity: "recovery".into(),
        sha256_evidence: "b".repeat(64),
        metrics: vec![],
    };
    assert!(matches!(
        SemanticEvidenceRecord::new(spec.clone()),
        Err(SemanticEvidenceError::MissingIntervention)
    ));

    spec.intervention_identity = Some("balanced-mode".into());
    spec.observation_horizon = None;
    assert!(matches!(
        SemanticEvidenceRecord::new(spec),
        Err(SemanticEvidenceError::MissingObservationHorizon)
    ));
}

#[test]
fn structural_itd_evidence_can_be_non_interventional() {
    let record = SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic: semantic(),
        workload: EvidenceWorkloadFingerprint::from_workload(&workload(MaskKind::Bidirectional)),
        kind: DiagnosticEvidenceKind::ItdStructural,
        producer_repository: "Memorithm/itd-simulator".into(),
        producer_revision: "c".repeat(40),
        artifact_identity: "attention-structure-v1".into(),
        intervention_identity: None,
        observation_horizon: None,
        metric_identity: "structural-concentration".into(),
        sha256_evidence: "d".repeat(64),
        metrics: vec![("descriptor".into(), 0.25)],
    })
    .unwrap();

    assert_eq!(record.kind(), DiagnosticEvidenceKind::ItdStructural);
    assert_eq!(record.intervention_identity(), None);
    assert_eq!(record.observation_horizon(), None);
}

#[test]
fn metric_order_is_canonical_and_duplicate_names_fail_closed() {
    let record = golden_tdi();
    assert_eq!(record.metrics()[0].0, "linf_h3");
    assert_eq!(record.metrics()[1].0, "recovery_h3");

    let mut duplicate = record.metrics().to_vec();
    duplicate.push(("linf_h3".into(), 1.0));
    let result = SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic: record.semantic().clone(),
        workload: record.workload(),
        kind: record.kind(),
        producer_repository: record.producer_repository().into(),
        producer_revision: record.producer_revision().into(),
        artifact_identity: record.artifact_identity().into(),
        intervention_identity: record.intervention_identity().map(str::to_string),
        observation_horizon: record.observation_horizon(),
        metric_identity: record.metric_identity().into(),
        sha256_evidence: record.sha256_evidence().into(),
        metrics: duplicate,
    });
    assert!(matches!(
        result,
        Err(SemanticEvidenceError::DuplicateMetric)
    ));
}

#[test]
fn non_finite_summary_and_invalid_digests_are_rejected() {
    let record = golden_tdi();
    let bad_metric = SemanticEvidenceRecord::new(SemanticEvidenceSpec {
        semantic: record.semantic().clone(),
        workload: record.workload(),
        kind: record.kind(),
        producer_repository: record.producer_repository().into(),
        producer_revision: record.producer_revision().into(),
        artifact_identity: record.artifact_identity().into(),
        intervention_identity: record.intervention_identity().map(str::to_string),
        observation_horizon: record.observation_horizon(),
        metric_identity: record.metric_identity().into(),
        sha256_evidence: record.sha256_evidence().into(),
        metrics: vec![("bad".into(), f64::NAN)],
    });
    assert!(matches!(
        bad_metric,
        Err(SemanticEvidenceError::NonFiniteMetric)
    ));

    let mut text = record.to_canonical_text();
    text = text.replace(&"b".repeat(64), "deadbeef");
    assert!(SemanticEvidenceRecord::from_canonical_text(&text).is_err());
}

#[test]
fn unknown_or_duplicate_canonical_fields_fail_closed() {
    let record = golden_tdi();
    let mut unknown = record.to_canonical_text();
    unknown.push_str("unknown=field\n");
    assert!(SemanticEvidenceRecord::from_canonical_text(&unknown).is_err());

    let mut duplicate = record.to_canonical_text();
    duplicate.push_str("version=1\n");
    assert!(SemanticEvidenceRecord::from_canonical_text(&duplicate).is_err());
}

#[test]
fn diagnostic_reference_keeps_evidence_out_of_semantic_identity() {
    let record = golden_tdi();
    let reference = record.diagnostic_reference().unwrap();
    assert_eq!(reference.kind(), DiagnosticEvidenceKind::TdiRecovery);
    assert_eq!(reference.repository(), "Memorithm/TDI");
    assert_eq!(reference.artifact(), "tdi-ai-gate-b");
    assert_eq!(record.semantic().name(), "balanced-three-token-mixer");
    assert!(reference.revision_binding().contains("git:"));
    assert!(reference.revision_binding().contains(";sha256:"));
}
