use std::collections::{BTreeMap, BTreeSet};

use ada_core::{DiagnosticEvidenceKind, SemanticFamily, SemanticId};

use super::{
    EvidenceWorkloadFingerprint, MAX_SEMANTIC_EVIDENCE_BYTES, MAX_SUMMARY_METRICS,
    SEMANTIC_EVIDENCE_HEADER, SEMANTIC_EVIDENCE_VERSION, SemanticEvidenceError,
    SemanticEvidenceRecord, SemanticEvidenceSpec, is_lower_hex_exact,
};

type FieldMap<'a> = BTreeMap<String, &'a str>;

impl SemanticEvidenceRecord {
    /// Canonical deterministic interchange text.
    ///
    /// Floating-point summaries are encoded by exact IEEE-754 bit pattern so
    /// re-serialization does not depend on decimal formatting.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(SEMANTIC_EVIDENCE_HEADER);
        text.push('\n');
        append_field(&mut text, "version", &self.version.to_string());
        append_field(
            &mut text,
            "semantic_family",
            semantic_family_text(self.semantic.family()),
        );
        append_field(
            &mut text,
            "semantic_name",
            &hex_encode(self.semantic.name()),
        );
        append_field(
            &mut text,
            "semantic_revision",
            &self.semantic.revision().to_string(),
        );
        append_field(
            &mut text,
            "workload_primary",
            &format!("{:016x}", self.workload.primary()),
        );
        append_field(
            &mut text,
            "workload_secondary",
            &format!("{:016x}", self.workload.secondary()),
        );
        append_field(
            &mut text,
            "workload_length",
            &format!("{:016x}", self.workload.length()),
        );
        append_field(&mut text, "evidence_kind", evidence_kind_text(self.kind));
        append_field(
            &mut text,
            "producer_repository",
            &hex_encode(&self.producer_repository),
        );
        append_field(&mut text, "producer_revision", &self.producer_revision);
        append_field(
            &mut text,
            "artifact_identity",
            &hex_encode(&self.artifact_identity),
        );
        append_field(
            &mut text,
            "intervention_identity",
            &self
                .intervention_identity
                .as_deref()
                .map_or_else(|| "-".into(), hex_encode),
        );
        append_field(
            &mut text,
            "observation_horizon",
            &self
                .observation_horizon
                .map_or_else(|| "-".into(), |value| value.to_string()),
        );
        append_field(
            &mut text,
            "metric_identity",
            &hex_encode(&self.metric_identity),
        );
        append_field(&mut text, "sha256_evidence", &self.sha256_evidence);
        append_field(&mut text, "metrics_count", &self.metrics.len().to_string());
        append_metrics(&mut text, &self.metrics);
        text
    }

    /// Decode strict canonical interchange text.
    ///
    /// # Errors
    ///
    /// Rejects unknown/duplicate fields, unsupported versions, malformed
    /// fingerprints or float bits, invalid semantic identity, incomplete metric
    /// sets and all constructor-level provenance violations.
    pub fn from_canonical_text(text: &str) -> Result<Self, SemanticEvidenceError> {
        let fields = parse_fields(text)?;
        let metrics_count = validate_field_set(&fields)?;
        let field = |key: &str| required_field(&fields, key);

        let version = parse_u16("version", field("version")?)?;
        if version != SEMANTIC_EVIDENCE_VERSION {
            return Err(SemanticEvidenceError::UnsupportedVersion(version));
        }

        let semantic = SemanticId::new(
            parse_semantic_family(field("semantic_family")?)?,
            hex_decode("semantic_name", field("semantic_name")?)?,
            parse_u32("semantic_revision", field("semantic_revision")?)?,
        )?;
        let workload = EvidenceWorkloadFingerprint::from_parts(
            parse_fixed_u64_hex("workload_primary", field("workload_primary")?)?,
            parse_fixed_u64_hex("workload_secondary", field("workload_secondary")?)?,
            parse_fixed_u64_hex("workload_length", field("workload_length")?)?,
        );
        let metrics = parse_metrics(&fields, metrics_count)?;

        Self::new(SemanticEvidenceSpec {
            semantic,
            workload,
            kind: parse_evidence_kind(field("evidence_kind")?)?,
            producer_repository: hex_decode("producer_repository", field("producer_repository")?)?,
            producer_revision: field("producer_revision")?.to_string(),
            artifact_identity: hex_decode("artifact_identity", field("artifact_identity")?)?,
            intervention_identity: parse_optional_hex_identifier(
                "intervention_identity",
                field("intervention_identity")?,
            )?,
            observation_horizon: parse_optional_u32(
                "observation_horizon",
                field("observation_horizon")?,
            )?,
            metric_identity: hex_decode("metric_identity", field("metric_identity")?)?,
            sha256_evidence: field("sha256_evidence")?.to_string(),
            metrics,
        })
    }
}

fn append_field(text: &mut String, key: &str, value: &str) {
    text.push_str(key);
    text.push('=');
    text.push_str(value);
    text.push('\n');
}

fn append_metrics(text: &mut String, metrics: &[(String, f64)]) {
    for (index, (name, value)) in metrics.iter().enumerate() {
        append_field(text, &format!("metric_{index}_name"), &hex_encode(name));
        append_field(
            text,
            &format!("metric_{index}_bits"),
            &format!("{:016x}", value.to_bits()),
        );
    }
}

fn parse_fields(text: &str) -> Result<FieldMap<'_>, SemanticEvidenceError> {
    if text.len() > MAX_SEMANTIC_EVIDENCE_BYTES || text.contains('\r') || !text.ends_with('\n') {
        return Err(SemanticEvidenceError::MalformedCanonicalText(
            "artifact exceeds its limit, contains CR, or lacks final newline".into(),
        ));
    }

    let mut lines = text.lines();
    if lines.next() != Some(SEMANTIC_EVIDENCE_HEADER) {
        return Err(SemanticEvidenceError::MalformedCanonicalText(
            "missing ADA-SEMANTIC-EVIDENCE-V1 header".into(),
        ));
    }

    let mut fields = BTreeMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Err(SemanticEvidenceError::MalformedCanonicalText(
                "field is missing '='".into(),
            ));
        };
        if key.is_empty() || value.contains('=') || fields.insert(key.to_string(), value).is_some()
        {
            return Err(SemanticEvidenceError::MalformedCanonicalText(
                "empty, duplicate, or ambiguous field".into(),
            ));
        }
    }
    Ok(fields)
}

fn validate_field_set(fields: &FieldMap<'_>) -> Result<usize, SemanticEvidenceError> {
    let metrics_count = parse_usize("metrics_count", required_field(fields, "metrics_count")?)?;
    if metrics_count > MAX_SUMMARY_METRICS {
        return Err(SemanticEvidenceError::TooManyMetrics);
    }

    let fixed_fields = [
        "version",
        "semantic_family",
        "semantic_name",
        "semantic_revision",
        "workload_primary",
        "workload_secondary",
        "workload_length",
        "evidence_kind",
        "producer_repository",
        "producer_revision",
        "artifact_identity",
        "intervention_identity",
        "observation_horizon",
        "metric_identity",
        "sha256_evidence",
        "metrics_count",
    ];
    let mut expected: BTreeSet<String> = fixed_fields.into_iter().map(str::to_string).collect();
    for index in 0..metrics_count {
        expected.insert(format!("metric_{index}_name"));
        expected.insert(format!("metric_{index}_bits"));
    }
    let actual: BTreeSet<String> = fields.keys().cloned().collect();
    if actual != expected {
        return Err(SemanticEvidenceError::MalformedCanonicalText(
            "canonical field set is incomplete or contains unknown keys".into(),
        ));
    }
    Ok(metrics_count)
}

fn required_field<'a>(
    fields: &'a FieldMap<'a>,
    key: &str,
) -> Result<&'a str, SemanticEvidenceError> {
    fields.get(key).copied().ok_or_else(|| {
        SemanticEvidenceError::MalformedCanonicalText(format!("missing field {key}"))
    })
}

fn parse_metrics(
    fields: &FieldMap<'_>,
    metrics_count: usize,
) -> Result<Vec<(String, f64)>, SemanticEvidenceError> {
    let mut metrics = Vec::with_capacity(metrics_count);
    for index in 0..metrics_count {
        let name = hex_decode(
            "metric_name",
            required_field(fields, &format!("metric_{index}_name"))?,
        )?;
        let bits = parse_fixed_u64_hex(
            "metric_bits",
            required_field(fields, &format!("metric_{index}_bits"))?,
        )?;
        metrics.push((name, f64::from_bits(bits)));
    }
    Ok(metrics)
}

fn semantic_family_text(family: SemanticFamily) -> &'static str {
    match family {
        SemanticFamily::StandardSoftmax => "standard-softmax",
        SemanticFamily::DifferentialSigned => "differential-signed",
        SemanticFamily::ToeplitzStructured => "toeplitz-structured",
        SemanticFamily::ProlateConcentration => "prolate-concentration",
        SemanticFamily::GroundStateGreen => "ground-state-green",
        SemanticFamily::SpectralFlow => "spectral-flow",
        SemanticFamily::RecurrentMemory => "recurrent-memory",
        SemanticFamily::Hybrid => "hybrid",
        SemanticFamily::Experimental => "experimental",
    }
}

fn parse_semantic_family(value: &str) -> Result<SemanticFamily, SemanticEvidenceError> {
    match value {
        "standard-softmax" => Ok(SemanticFamily::StandardSoftmax),
        "differential-signed" => Ok(SemanticFamily::DifferentialSigned),
        "toeplitz-structured" => Ok(SemanticFamily::ToeplitzStructured),
        "prolate-concentration" => Ok(SemanticFamily::ProlateConcentration),
        "ground-state-green" => Ok(SemanticFamily::GroundStateGreen),
        "spectral-flow" => Ok(SemanticFamily::SpectralFlow),
        "recurrent-memory" => Ok(SemanticFamily::RecurrentMemory),
        "hybrid" => Ok(SemanticFamily::Hybrid),
        "experimental" => Ok(SemanticFamily::Experimental),
        _ => Err(SemanticEvidenceError::MalformedCanonicalText(
            "unknown semantic family".into(),
        )),
    }
}

fn evidence_kind_text(kind: DiagnosticEvidenceKind) -> &'static str {
    match kind {
        DiagnosticEvidenceKind::TaskBehavior => "task-behavior",
        DiagnosticEvidenceKind::StaticOperator => "static-operator",
        DiagnosticEvidenceKind::ItdStructural => "itd-structural",
        DiagnosticEvidenceKind::TdiRecovery => "tdi-recovery",
        DiagnosticEvidenceKind::Adversarial => "adversarial",
        DiagnosticEvidenceKind::LogicalCost => "logical-cost",
        DiagnosticEvidenceKind::HardwareCost => "hardware-cost",
        DiagnosticEvidenceKind::Generalization => "generalization",
        DiagnosticEvidenceKind::PriorArt => "prior-art",
    }
}

fn parse_evidence_kind(value: &str) -> Result<DiagnosticEvidenceKind, SemanticEvidenceError> {
    match value {
        "task-behavior" => Ok(DiagnosticEvidenceKind::TaskBehavior),
        "static-operator" => Ok(DiagnosticEvidenceKind::StaticOperator),
        "itd-structural" => Ok(DiagnosticEvidenceKind::ItdStructural),
        "tdi-recovery" => Ok(DiagnosticEvidenceKind::TdiRecovery),
        "adversarial" => Ok(DiagnosticEvidenceKind::Adversarial),
        "logical-cost" => Ok(DiagnosticEvidenceKind::LogicalCost),
        "hardware-cost" => Ok(DiagnosticEvidenceKind::HardwareCost),
        "generalization" => Ok(DiagnosticEvidenceKind::Generalization),
        "prior-art" => Ok(DiagnosticEvidenceKind::PriorArt),
        _ => Err(SemanticEvidenceError::MalformedCanonicalText(
            "unknown evidence kind".into(),
        )),
    }
}

fn parse_u16(field: &str, value: &str) -> Result<u16, SemanticEvidenceError> {
    value.parse::<u16>().map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not an unsigned 16-bit integer"
        ))
    })
}

fn parse_u32(field: &str, value: &str) -> Result<u32, SemanticEvidenceError> {
    value.parse::<u32>().map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not an unsigned 32-bit integer"
        ))
    })
}

fn parse_usize(field: &str, value: &str) -> Result<usize, SemanticEvidenceError> {
    value.parse::<usize>().map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!("{field} is not an unsigned integer"))
    })
}

fn parse_optional_u32(field: &str, value: &str) -> Result<Option<u32>, SemanticEvidenceError> {
    if value == "-" {
        Ok(None)
    } else {
        parse_u32(field, value).map(Some)
    }
}

fn parse_fixed_u64_hex(field: &str, value: &str) -> Result<u64, SemanticEvidenceError> {
    if !is_lower_hex_exact(value, 16) {
        return Err(SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} must be exactly 16 lowercase hex characters"
        )));
    }
    u64::from_str_radix(value, 16).map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!("{field} is not valid hex"))
    })
}

fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(field: &str, value: &str) -> Result<String, SemanticEvidenceError> {
    if value.len() % 2 != 0 {
        return Err(SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} has an odd-length hex value"
        )));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut chars = value.bytes();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = hex_digit(high).ok_or_else(|| {
            SemanticEvidenceError::MalformedCanonicalText(format!(
                "{field} contains a non-hex digit"
            ))
        })?;
        let low = hex_digit(low).ok_or_else(|| {
            SemanticEvidenceError::MalformedCanonicalText(format!(
                "{field} contains a non-hex digit"
            ))
        })?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).map_err(|_| {
        SemanticEvidenceError::MalformedCanonicalText(format!(
            "{field} is not UTF-8 after hex decoding"
        ))
    })
}

fn parse_optional_hex_identifier(
    field: &str,
    value: &str,
) -> Result<Option<String>, SemanticEvidenceError> {
    if value == "-" {
        Ok(None)
    } else {
        hex_decode(field, value).map(Some)
    }
}

const fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}
