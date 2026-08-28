use super::{
    BTreeSet, CegisResult, CorrectnessStatus, DiagnosticEvidenceKind, EstimatedCostReport,
    EvidenceBoundQualification, EvidenceWorkloadFingerprint, GraduationError, GraduationObjectives,
    LogicalCost, MAX_GRADUATION_EVIDENCE, MAX_GRADUATION_FIXTURES, NumericalObjectives,
    ObjectiveVector, OracleFixtureArtifact, QUALIFICATION_CASE_VERSION, QualificationVerdict,
    SemanticEvidenceRecord, SemanticProgram, SemanticWorkloadCase, WorkloadContract,
};

pub(super) fn collect_oracle_fixtures(
    qualification: &EvidenceBoundQualification,
    cegis: &CegisResult<SemanticProgram, SemanticWorkloadCase>,
) -> Result<Vec<OracleFixtureArtifact>, GraduationError> {
    let oracle = qualification.oracle();
    let expected = oracle
        .fixture_fingerprints()
        .iter()
        .map(|fingerprint| {
            (
                fingerprint.primary(),
                fingerprint.secondary(),
                fingerprint.length(),
            )
        })
        .collect::<BTreeSet<_>>();
    let workload = oracle.workload_fingerprint();
    let mut fixtures = Vec::new();
    for fixture in cegis.active_fixtures() {
        let fingerprint = fixture.fingerprint();
        let key = (
            fingerprint.primary(),
            fingerprint.secondary(),
            fingerprint.length(),
        );
        if expected.contains(&key) {
            if fixture.input().workload_fingerprint() != workload {
                return Err(GraduationError::OracleWorkloadMismatch);
            }
            fixtures.push(OracleFixtureArtifact::from_fixture(fixture, workload)?);
        }
    }
    if fixtures.len() != expected.len() {
        return Err(GraduationError::OracleFixtureMismatch);
    }
    check_count_limit(
        "oracle_fixture_count",
        fixtures.len(),
        MAX_GRADUATION_FIXTURES,
    )?;
    fixtures.sort_by_key(OracleFixtureArtifact::fingerprint);
    if !strictly_sorted_fixtures(&fixtures) {
        return Err(GraduationError::OracleFixtureMismatch);
    }
    Ok(fixtures)
}

pub(super) fn objectives_from_report(
    report: EstimatedCostReport,
    input: GraduationObjectives,
) -> Result<ObjectiveVector, GraduationError> {
    ObjectiveVector::from_parts(
        CorrectnessStatus::Provisional,
        NumericalObjectives::default(),
        LogicalCost {
            flops: Some(report.logical_flops()),
            qk_evaluations: Some(report.score_pairs()),
            transcendental_operations: Some(report.transcendental_operations()),
            value_operations: Some(report.value_elements()),
        },
        report.to_objective_estimate()?,
        input.measured,
        input.quality,
    )
    .map_err(GraduationError::from)
}

pub(super) fn validate_cost_objectives(
    report: EstimatedCostReport,
    objectives: &ObjectiveVector,
) -> Result<(), GraduationError> {
    let expected_logical = LogicalCost {
        flops: Some(report.logical_flops()),
        qk_evaluations: Some(report.score_pairs()),
        transcendental_operations: Some(report.transcendental_operations()),
        value_operations: Some(report.value_elements()),
    };
    let expected_estimated = report.to_objective_estimate()?;
    if objectives.logical() != expected_logical || objectives.estimated() != expected_estimated {
        return Err(GraduationError::MalformedCanonical(
            "objective logical/estimated cost does not reproduce A12".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_bundle_policy(
    semantic: &SemanticProgram,
    workload: &WorkloadContract,
    objectives: &ObjectiveVector,
    evidence: &[SemanticEvidenceRecord],
    verdict: QualificationVerdict,
) -> Result<(), GraduationError> {
    objectives.validate()?;
    semantic.validate_for_workload(workload).map_err(|error| {
        GraduationError::MalformedCanonical(format!("semantic/workload: {error}"))
    })?;
    let workload_fingerprint = EvidenceWorkloadFingerprint::from_workload(workload);
    let mut canonical = BTreeSet::new();
    for record in evidence {
        if record.semantic() != semantic.descriptor().id()
            || record.workload() != workload_fingerprint
        {
            return Err(GraduationError::EvidenceBindingMismatch);
        }
        if !canonical.insert(record.to_canonical_text()) {
            return Err(GraduationError::DuplicateEvidence);
        }
    }
    let evidence_count = u64::try_from(evidence.len()).unwrap_or(u64::MAX);
    if evidence_count == 0 {
        return Err(GraduationError::MalformedCanonical(
            "evidence set is empty".into(),
        ));
    }
    if evidence_count > MAX_GRADUATION_EVIDENCE {
        return Err(GraduationError::ExceedsLimit {
            field: "evidence_count",
            value: evidence_count,
            maximum: MAX_GRADUATION_EVIDENCE,
        });
    }
    if !evidence
        .windows(2)
        .all(|pair| pair[0].to_canonical_text() < pair[1].to_canonical_text())
    {
        return Err(GraduationError::MalformedCanonical(
            "evidence records are not in canonical order".into(),
        ));
    }

    if objectives.correctness() != CorrectnessStatus::Provisional {
        return Err(GraduationError::InvalidCorrectnessStatus);
    }
    if matches!(
        verdict,
        QualificationVerdict::Adopt | QualificationVerdict::Adapt
    ) {
        return Err(GraduationError::VerdictRequiresQualifiedCorrectness);
    }

    let has_task = has_evidence_kind(evidence, DiagnosticEvidenceKind::TaskBehavior);
    let has_hardware = has_evidence_kind(evidence, DiagnosticEvidenceKind::HardwareCost);
    let has_observed_quality = objectives
        .quality()
        .iter()
        .any(|metric| metric.value().is_some());
    if has_observed_quality && !has_task {
        return Err(GraduationError::MissingTaskEvidence);
    }
    let measured = objectives.measured();
    if (measured.latency_ns.is_some() || measured.energy_nj.is_some()) && !has_hardware {
        return Err(GraduationError::MissingHardwareEvidence);
    }
    Ok(())
}

fn has_evidence_kind(evidence: &[SemanticEvidenceRecord], kind: DiagnosticEvidenceKind) -> bool {
    evidence.iter().any(|record| record.kind() == kind)
}

pub(super) fn validate_qualification_case_workload(
    text: &str,
    workload: EvidenceWorkloadFingerprint,
) -> Result<(), GraduationError> {
    let mut lines = text.lines();
    let expected_header = format!("ADA-QUALIFICATION-CASE-V{QUALIFICATION_CASE_VERSION}");
    if lines.next() != Some(expected_header.as_str()) {
        return Err(GraduationError::OracleWorkloadMismatch);
    }
    let Some(line) = lines.next() else {
        return Err(GraduationError::OracleWorkloadMismatch);
    };
    let expected = format!(
        "{:016x}-{:016x}-{:016x}",
        workload.primary(),
        workload.secondary(),
        workload.length()
    );
    if line.strip_prefix("workload=") != Some(expected.as_str()) {
        return Err(GraduationError::OracleWorkloadMismatch);
    }
    Ok(())
}

pub(super) fn strictly_sorted_fixtures(fixtures: &[OracleFixtureArtifact]) -> bool {
    fixtures
        .windows(2)
        .all(|pair| pair[0].fingerprint() < pair[1].fingerprint())
}

pub(super) fn check_count_limit(
    field: &'static str,
    value: usize,
    maximum: u64,
) -> Result<(), GraduationError> {
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    if value > maximum {
        return Err(GraduationError::ExceedsLimit {
            field,
            value,
            maximum,
        });
    }
    Ok(())
}
