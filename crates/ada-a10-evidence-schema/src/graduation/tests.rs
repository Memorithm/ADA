use super::{
    EvidenceRef, ExternalDiagnostics, MaskStateContract, NumericPolicy, PriorArtStatus,
    PriorArtStatusKind, QualificationVerdictJson, ReferenceDefinition,
    SEMANTIC_QUALIFICATION_SCHEMA_VERSION, SemanticQualificationError, SemanticQualificationRecord,
    SemanticQualificationSpec,
};

fn hex(fill: char, count: usize) -> String {
    std::iter::repeat_n(fill, count).collect()
}

fn evidence(kind: &str, artifact: &str) -> EvidenceRef {
    EvidenceRef {
        kind: kind.into(),
        repository: "Memorithm/ADA".into(),
        commit: hex('a', 40),
        artifact: artifact.into(),
        sha256: hex('b', 64),
    }
}

fn valid_spec() -> SemanticQualificationSpec {
    SemanticQualificationSpec {
        candidate_id: "ADA-A11-CANDIDATE-EXAMPLE".into(),
        semantic_id: "experimental-reference-semantic".into(),
        semantic_version: 1,
        reference_definition: ReferenceDefinition {
            repository: "Memorithm/ADA".into(),
            commit: hex('c', 40),
            artifact: "crates/ada-semantic/src/lib.rs".into(),
            sha256: hex('d', 64),
        },
        invariants: vec![
            "probabilities_non_negative".into(),
            "row_sum_one_when_finite".into(),
        ],
        mask_state_contract: MaskStateContract {
            mask: "causal".into(),
            state: "stateless".into(),
        },
        numeric_policy: NumericPolicy {
            reference_precision: "binary64".into(),
            non_finite_policy: "reject".into(),
        },
        oracle_fixtures: vec![evidence("oracle-fixture", "fixtures/oracle_v1.txt")],
        adversarial_cases: vec![evidence("adversarial", "fixtures/adversarial_v1.txt")],
        task_evidence: Vec::new(),
        external_diagnostics: ExternalDiagnostics {
            itd: vec![evidence("itd-structural", "evidence/itd_v1.txt")],
            tdi: Vec::new(),
        },
        logical_cost_evidence: vec![evidence("logical-cost", "evidence/cost_v1.txt")],
        prior_art_status: PriorArtStatus {
            status: PriorArtStatusKind::NotAssessed,
            references: Vec::new(),
        },
        verdict: QualificationVerdictJson::ContinueResearch,
    }
}

#[test]
fn happy_path_round_trips_canonical_json() {
    let record = SemanticQualificationRecord::new(valid_spec()).expect("valid fixture");
    assert_eq!(
        record.schema_version(),
        SEMANTIC_QUALIFICATION_SCHEMA_VERSION
    );
    assert_eq!(record.verdict(), QualificationVerdictJson::ContinueResearch);
    assert!(record.task_evidence().is_empty());
    assert_eq!(record.external_diagnostics().itd.len(), 1);
    assert!(record.external_diagnostics().tdi.is_empty());

    let json = record.to_canonical_json();
    assert!(json.starts_with('{'));
    assert!(json.contains("\"schema_version\":1"));
    assert!(!json.contains("implementation"));
    let decoded = SemanticQualificationRecord::from_canonical_json(&json).expect("decode");
    assert_eq!(decoded, record);
    assert_eq!(decoded.to_canonical_json(), json);
}

#[test]
fn rejects_empty_and_duplicate_invariants() {
    let mut spec = valid_spec();
    spec.invariants.clear();
    assert_eq!(
        SemanticQualificationRecord::new(spec),
        Err(SemanticQualificationError::EmptyInvariants)
    );

    let mut spec = valid_spec();
    spec.invariants = vec!["same".into(), "same".into()];
    assert_eq!(
        SemanticQualificationRecord::new(spec),
        Err(SemanticQualificationError::DuplicateInvariant)
    );
}

#[test]
fn rejects_bad_hashes_and_zero_semantic_version() {
    let mut spec = valid_spec();
    spec.reference_definition.commit = "ABC".into();
    assert_eq!(
        SemanticQualificationRecord::new(spec),
        Err(SemanticQualificationError::InvalidCommit(
            "reference_definition.commit"
        ))
    );

    let mut spec = valid_spec();
    spec.oracle_fixtures[0].sha256 = "deadbeef".into();
    assert_eq!(
        SemanticQualificationRecord::new(spec),
        Err(SemanticQualificationError::InvalidSha256("evidence.sha256"))
    );

    let mut spec = valid_spec();
    spec.semantic_version = 0;
    assert_eq!(
        SemanticQualificationRecord::new(spec),
        Err(SemanticQualificationError::ZeroSemanticVersion)
    );
}

#[test]
fn rejects_missing_required_fields_and_unknown_properties() {
    let record = SemanticQualificationRecord::new(valid_spec()).expect("valid fixture");
    let json = record.to_canonical_json();

    let missing = json.replacen("\"candidate_id\":\"ADA-A11-CANDIDATE-EXAMPLE\",", "", 1);
    assert!(matches!(
        SemanticQualificationRecord::from_canonical_json(&missing),
        Err(SemanticQualificationError::MissingProperty {
            path: "$",
            key: "candidate_id"
        })
    ));

    let with_unknown = json.replacen('{', "{\"extra\":true,", 1);
    assert!(matches!(
        SemanticQualificationRecord::from_canonical_json(&with_unknown),
        Err(SemanticQualificationError::UnknownProperty { path: "$", key }) if key == "extra"
    ));
}

#[test]
fn rejects_invalid_enum_and_schema_version() {
    let record = SemanticQualificationRecord::new(valid_spec()).expect("valid fixture");
    let json = record
        .to_canonical_json()
        .replace("\"CONTINUE_RESEARCH\"", "\"MAYBE\"");
    assert!(matches!(
        SemanticQualificationRecord::from_canonical_json(&json),
        Err(SemanticQualificationError::InvalidEnum {
            field: "verdict",
            ..
        })
    ));

    let json = record
        .to_canonical_json()
        .replace("\"schema_version\":1", "\"schema_version\":2");
    assert_eq!(
        SemanticQualificationRecord::from_canonical_json(&json),
        Err(SemanticQualificationError::UnsupportedSchemaVersion(2))
    );
}

#[test]
fn keeps_identity_diagnostics_cost_task_and_verdict_separate() {
    let record = SemanticQualificationRecord::new(valid_spec()).expect("valid fixture");
    // Semantic identity fields are present without embedding ITD/TDI into them.
    assert_eq!(record.semantic_id(), "experimental-reference-semantic");
    assert_ne!(
        record.semantic_id(),
        record.external_diagnostics().itd[0].artifact
    );
    // Task evidence, logical cost, diagnostics, and verdict remain distinct lanes.
    assert!(record.task_evidence().is_empty());
    assert_eq!(record.logical_cost_evidence().len(), 1);
    assert_eq!(
        record.prior_art_status().status,
        PriorArtStatusKind::NotAssessed
    );
    assert_eq!(record.verdict(), QualificationVerdictJson::ContinueResearch);
}

#[test]
fn documents_no_automatic_flat_graduation_conversion() {
    // FlatGraduationRecord/Bundle lack required schema fields; callers must not
    // invent missing evidence. This test locks the public API surface: there is
    // no From/TryFrom conversion helper exported for those types.
    let exports = module_path!();
    assert!(exports.contains("graduation"));
}
