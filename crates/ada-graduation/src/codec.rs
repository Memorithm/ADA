use super::{
    CostAssumptions, Display, EvidenceWorkloadFingerprint, FLAT_GRADUATION_BUNDLE_HEADER,
    FLAT_GRADUATION_BUNDLE_VERSION, FlatGraduationBundle, GraduationError, ImplementationPlan,
    MAX_GRADUATION_BUNDLE_BYTES, MAX_GRADUATION_EVIDENCE, MAX_GRADUATION_FIXTURES, ObjectiveVector,
    OperationProfile, OracleFixtureArtifact, OracleFixtureFingerprint, QualificationVerdict,
    SemanticEvidenceRecord, SemanticProgram, WorkloadContract, estimate_cost,
};
use crate::policy::{
    check_count_limit, strictly_sorted_fixtures, validate_bundle_policy, validate_cost_objectives,
    validate_qualification_case_workload,
};

impl FlatGraduationBundle {
    /// Encode the complete graduation bundle in strict deterministic text.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("{FLAT_GRADUATION_BUNDLE_HEADER}\n");
        append_field(&mut text, "version", FLAT_GRADUATION_BUNDLE_VERSION);
        append_field(
            &mut text,
            "semantic",
            hex_encode(&self.semantic.to_canonical_text()),
        );
        append_field(
            &mut text,
            "workload",
            hex_encode(&self.workload.to_canonical_text()),
        );
        append_field(
            &mut text,
            "oracle_fixture_count",
            self.oracle_fixtures.len(),
        );
        for (index, fixture) in self.oracle_fixtures.iter().enumerate() {
            append_field(
                &mut text,
                &format!("oracle_fixture_{index}_id"),
                hex_encode(fixture.id()),
            );
            append_field(
                &mut text,
                &format!("oracle_fixture_{index}_fingerprint"),
                fixture.fingerprint(),
            );
            append_field(
                &mut text,
                &format!("oracle_fixture_{index}_text"),
                hex_encode(fixture.canonical_text()),
            );
        }
        append_field(
            &mut text,
            "implementation",
            hex_encode(&self.implementation.to_canonical_text()),
        );
        append_field(
            &mut text,
            "operation_score_flops_per_pair",
            self.operation_profile.score_flops_per_pair,
        );
        append_field(
            &mut text,
            "operation_transcendentals_per_pair",
            self.operation_profile.transcendentals_per_pair,
        );
        append_field(
            &mut text,
            "operation_value_flops_per_element",
            self.operation_profile.value_flops_per_element,
        );
        append_field(
            &mut text,
            "operation_finalize_flops_per_output",
            self.operation_profile.finalize_flops_per_output,
        );
        append_field(
            &mut text,
            "cost_score_passes",
            self.cost_assumptions.score_passes,
        );
        append_field(
            &mut text,
            "cost_value_passes",
            self.cost_assumptions.value_passes,
        );
        append_field(
            &mut text,
            "cost_reload_query_per_kv_tile",
            bool_text(self.cost_assumptions.reload_query_per_kv_tile),
        );
        append_field(
            &mut text,
            "cost_reuse_shared_kv_across_query_heads",
            bool_text(self.cost_assumptions.reuse_shared_kv_across_query_heads),
        );
        append_field(
            &mut text,
            "objectives",
            hex_encode(&self.objectives.to_canonical_text()),
        );
        append_field(&mut text, "evidence_count", self.evidence.len());
        for (index, record) in self.evidence.iter().enumerate() {
            append_field(
                &mut text,
                &format!("evidence_{index}"),
                hex_encode(&record.to_canonical_text()),
            );
        }
        append_field(&mut text, "verdict", verdict_text(self.verdict));
        text
    }

    /// Decode and fully revalidate a canonical graduation artifact.
    ///
    /// A12 logical/estimated objectives are recomputed from the nested workload,
    /// implementation, operation profile and cost assumptions; tampered cost
    /// fields therefore fail closed.
    ///
    /// # Errors
    ///
    /// Rejects malformed/non-canonical text, invalid nested artifacts, cost
    /// mismatches, evidence mismatches, or policy violations.
    #[allow(clippy::too_many_lines)]
    pub fn from_canonical_text(text: &str) -> Result<Self, GraduationError> {
        if text.is_empty() || text.len() > MAX_GRADUATION_BUNDLE_BYTES || text.contains('\r') {
            return Err(GraduationError::MalformedCanonical(
                "artifact exceeds limit, is empty, or contains CR".into(),
            ));
        }
        if !text.ends_with('\n') {
            return Err(GraduationError::MalformedCanonical(
                "artifact must end with newline".into(),
            ));
        }
        let mut lines = text.lines();
        if lines.next() != Some(FLAT_GRADUATION_BUNDLE_HEADER) {
            return Err(GraduationError::MalformedCanonical("invalid header".into()));
        }
        let version = parse_u16(next_value(&mut lines, "version")?, "version")?;
        if version != FLAT_GRADUATION_BUNDLE_VERSION {
            return Err(GraduationError::MalformedCanonical(
                "unsupported graduation version".into(),
            ));
        }

        let semantic_text = hex_decode(next_value(&mut lines, "semantic")?)?;
        let semantic = SemanticProgram::from_canonical_text(&semantic_text).map_err(|error| {
            GraduationError::MalformedCanonical(format!("invalid semantic: {error}"))
        })?;
        let workload_text = hex_decode(next_value(&mut lines, "workload")?)?;
        let workload = WorkloadContract::from_canonical_text(&workload_text).map_err(|error| {
            GraduationError::MalformedCanonical(format!("invalid workload: {error}"))
        })?;
        let workload_fingerprint = EvidenceWorkloadFingerprint::from_workload(&workload);

        let fixture_count = parse_usize(
            next_value(&mut lines, "oracle_fixture_count")?,
            "oracle_fixture_count",
        )?;
        check_count_limit(
            "oracle_fixture_count",
            fixture_count,
            MAX_GRADUATION_FIXTURES,
        )?;
        if fixture_count == 0 {
            return Err(GraduationError::MalformedCanonical(
                "oracle fixture set is empty".into(),
            ));
        }
        let mut oracle_fixtures = Vec::with_capacity(fixture_count);
        for index in 0..fixture_count {
            let id = hex_decode(next_value(
                &mut lines,
                &format!("oracle_fixture_{index}_id"),
            )?)?;
            let fingerprint = parse_fixture_fingerprint(next_value(
                &mut lines,
                &format!("oracle_fixture_{index}_fingerprint"),
            )?)?;
            let canonical_text = hex_decode(next_value(
                &mut lines,
                &format!("oracle_fixture_{index}_text"),
            )?)?;
            oracle_fixtures.push(OracleFixtureArtifact::from_parts(
                id,
                fingerprint,
                canonical_text,
                workload_fingerprint,
            )?);
        }
        if !strictly_sorted_fixtures(&oracle_fixtures) {
            return Err(GraduationError::MalformedCanonical(
                "oracle fixtures are not in canonical fingerprint order".into(),
            ));
        }

        let implementation_text = hex_decode(next_value(&mut lines, "implementation")?)?;
        let implementation = ImplementationPlan::from_canonical_text(&implementation_text)
            .map_err(|error| {
                GraduationError::MalformedCanonical(format!("invalid implementation: {error}"))
            })?;
        let operation_profile = OperationProfile {
            score_flops_per_pair: parse_u64(
                next_value(&mut lines, "operation_score_flops_per_pair")?,
                "operation_score_flops_per_pair",
            )?,
            transcendentals_per_pair: parse_u64(
                next_value(&mut lines, "operation_transcendentals_per_pair")?,
                "operation_transcendentals_per_pair",
            )?,
            value_flops_per_element: parse_u64(
                next_value(&mut lines, "operation_value_flops_per_element")?,
                "operation_value_flops_per_element",
            )?,
            finalize_flops_per_output: parse_u64(
                next_value(&mut lines, "operation_finalize_flops_per_output")?,
                "operation_finalize_flops_per_output",
            )?,
        };
        let cost_assumptions = CostAssumptions {
            score_passes: parse_u16(
                next_value(&mut lines, "cost_score_passes")?,
                "cost_score_passes",
            )?,
            value_passes: parse_u16(
                next_value(&mut lines, "cost_value_passes")?,
                "cost_value_passes",
            )?,
            reload_query_per_kv_tile: parse_bool(next_value(
                &mut lines,
                "cost_reload_query_per_kv_tile",
            )?)?,
            reuse_shared_kv_across_query_heads: parse_bool(next_value(
                &mut lines,
                "cost_reuse_shared_kv_across_query_heads",
            )?)?,
        };
        let objective_text = hex_decode(next_value(&mut lines, "objectives")?)?;
        let objectives = ObjectiveVector::from_canonical_text(&objective_text)?;

        let evidence_count =
            parse_usize(next_value(&mut lines, "evidence_count")?, "evidence_count")?;
        check_count_limit("evidence_count", evidence_count, MAX_GRADUATION_EVIDENCE)?;
        if evidence_count == 0 {
            return Err(GraduationError::MalformedCanonical(
                "evidence set is empty".into(),
            ));
        }
        let mut evidence = Vec::with_capacity(evidence_count);
        for index in 0..evidence_count {
            let record_text = hex_decode(next_value(&mut lines, &format!("evidence_{index}"))?)?;
            let record =
                SemanticEvidenceRecord::from_canonical_text(&record_text).map_err(|error| {
                    GraduationError::MalformedCanonical(format!("invalid E2 record: {error:?}"))
                })?;
            evidence.push(record);
        }
        let verdict = parse_verdict(next_value(&mut lines, "verdict")?)?;
        if lines.next().is_some() {
            return Err(GraduationError::MalformedCanonical(
                "unexpected trailing field".into(),
            ));
        }

        let bundle = Self {
            semantic,
            workload,
            oracle_fixtures,
            implementation,
            operation_profile,
            cost_assumptions,
            objectives,
            evidence,
            verdict,
        };
        bundle.validate_internal()?;
        if bundle.to_canonical_text() != text {
            return Err(GraduationError::MalformedCanonical(
                "artifact is not canonical".into(),
            ));
        }
        Ok(bundle)
    }

    pub(super) fn validate_internal(&self) -> Result<(), GraduationError> {
        if self.implementation.id().semantic() != self.semantic.descriptor().id() {
            return Err(GraduationError::SemanticImplementationMismatch);
        }
        let workload_fingerprint = EvidenceWorkloadFingerprint::from_workload(&self.workload);
        if self.oracle_fixtures.is_empty() {
            return Err(GraduationError::OracleFixtureMismatch);
        }
        for fixture in &self.oracle_fixtures {
            validate_qualification_case_workload(fixture.canonical_text(), workload_fingerprint)?;
        }
        if !strictly_sorted_fixtures(&self.oracle_fixtures) {
            return Err(GraduationError::MalformedCanonical(
                "oracle fixtures are not canonical".into(),
            ));
        }
        let report = estimate_cost(
            &self.workload,
            &self.implementation,
            self.operation_profile,
            self.cost_assumptions,
        )?;
        validate_cost_objectives(report, &self.objectives)?;
        validate_bundle_policy(
            &self.semantic,
            &self.workload,
            &self.objectives,
            &self.evidence,
            self.verdict,
        )
    }
}

fn append_field(text: &mut String, key: &str, value: impl Display) {
    text.push_str(key);
    text.push('=');
    text.push_str(&value.to_string());
    text.push('\n');
}

fn next_value<'a>(
    lines: &mut std::str::Lines<'a>,
    expected_key: &str,
) -> Result<&'a str, GraduationError> {
    let line = lines.next().ok_or_else(|| {
        GraduationError::MalformedCanonical(format!("missing field {expected_key}"))
    })?;
    let (key, value) = line.split_once('=').ok_or_else(|| {
        GraduationError::MalformedCanonical(format!("field {expected_key} lacks '='"))
    })?;
    if key != expected_key || value.is_empty() || value.contains('=') {
        return Err(GraduationError::MalformedCanonical(format!(
            "expected canonical field {expected_key}"
        )));
    }
    Ok(value)
}

fn parse_u64(value: &str, field: &str) -> Result<u64, GraduationError> {
    value
        .parse::<u64>()
        .map_err(|_| GraduationError::MalformedCanonical(format!("invalid integer field {field}")))
}

fn parse_u16(value: &str, field: &str) -> Result<u16, GraduationError> {
    value
        .parse::<u16>()
        .map_err(|_| GraduationError::MalformedCanonical(format!("invalid integer field {field}")))
}

fn parse_usize(value: &str, field: &str) -> Result<usize, GraduationError> {
    value
        .parse::<usize>()
        .map_err(|_| GraduationError::MalformedCanonical(format!("invalid integer field {field}")))
}

fn parse_bool(value: &str) -> Result<bool, GraduationError> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(GraduationError::MalformedCanonical(
            "boolean field must be 0 or 1".into(),
        )),
    }
}

const fn bool_text(value: bool) -> &'static str {
    if value { "1" } else { "0" }
}

fn verdict_text(verdict: QualificationVerdict) -> &'static str {
    match verdict {
        QualificationVerdict::ContinueResearch => "continue-research",
        QualificationVerdict::Adopt => "adopt",
        QualificationVerdict::Adapt => "adapt",
        QualificationVerdict::Reject => "reject",
    }
}

fn parse_verdict(value: &str) -> Result<QualificationVerdict, GraduationError> {
    match value {
        "continue-research" => Ok(QualificationVerdict::ContinueResearch),
        "adopt" => Ok(QualificationVerdict::Adopt),
        "adapt" => Ok(QualificationVerdict::Adapt),
        "reject" => Ok(QualificationVerdict::Reject),
        _ => Err(GraduationError::MalformedCanonical(
            "unknown qualification verdict".into(),
        )),
    }
}

fn parse_fixture_fingerprint(value: &str) -> Result<OracleFixtureFingerprint, GraduationError> {
    let mut parts = value.split('-');
    let primary = parse_hex_u64(parts.next(), "fixture primary")?;
    let secondary = parse_hex_u64(parts.next(), "fixture secondary")?;
    let length = parse_hex_u64(parts.next(), "fixture length")?;
    if parts.next().is_some() {
        return Err(GraduationError::MalformedCanonical(
            "invalid fixture fingerprint".into(),
        ));
    }
    Ok(OracleFixtureFingerprint {
        primary,
        secondary,
        length,
    })
}

fn parse_hex_u64(value: Option<&str>, field: &str) -> Result<u64, GraduationError> {
    let value =
        value.ok_or_else(|| GraduationError::MalformedCanonical(format!("missing {field}")))?;
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(GraduationError::MalformedCanonical(format!(
            "invalid {field}"
        )));
    }
    u64::from_str_radix(value, 16)
        .map_err(|_| GraduationError::MalformedCanonical(format!("invalid {field}")))
}

pub(super) fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len().saturating_mul(2));
    for byte in value.bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(value: &str) -> Result<String, GraduationError> {
    if value.len() % 2 != 0
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(GraduationError::MalformedCanonical(
            "invalid lowercase hex text".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        bytes.push((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?);
    }
    String::from_utf8(bytes)
        .map_err(|_| GraduationError::MalformedCanonical("hex text is not UTF-8".into()))
}

fn hex_nibble(byte: u8) -> Result<u8, GraduationError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(GraduationError::MalformedCanonical(
            "invalid lowercase hex digit".into(),
        )),
    }
}
