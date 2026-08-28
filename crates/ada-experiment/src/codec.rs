use crate::model::{
    EXPERIMENT_HEADER, EXPERIMENT_VERSION, EvidenceBinding, EvidenceKind, ExperimentError,
    ExperimentRecord, ExperimentSpec, MAX_EVIDENCE_BINDINGS, MAX_EXPERIMENT_BYTES,
    ProducerProvenance, malformed,
};
use ada_implementation::ImplementationPlan;
use ada_objective::ObjectiveVector;
use ada_semantic::SemanticProgram;
use ada_workload::WorkloadContract;
use std::fmt::Display;

impl ExperimentRecord {
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = String::from(EXPERIMENT_HEADER);
        text.push('\n');
        append_field(&mut text, "version", EXPERIMENT_VERSION);
        append_field(
            &mut text,
            "producer_repository",
            hex_encode(self.provenance.repository()),
        );
        append_field(
            &mut text,
            "producer_revision",
            self.provenance.git_revision(),
        );
        append_field(
            &mut text,
            "artifact_identity",
            hex_encode(self.provenance.artifact_identity()),
        );
        append_field(
            &mut text,
            "artifact_sha256",
            self.provenance.artifact_sha256(),
        );
        append_field(
            &mut text,
            "semantic_text",
            hex_encode(&self.semantic.to_canonical_text()),
        );
        append_field(
            &mut text,
            "workload_text",
            hex_encode(&self.workload.to_canonical_text()),
        );
        append_field(
            &mut text,
            "implementation_text",
            hex_encode(&self.implementation.to_canonical_text()),
        );
        append_field(
            &mut text,
            "objective_text",
            hex_encode(&self.objective.to_canonical_text()),
        );
        append_field(&mut text, "evidence_count", self.evidence.len());
        for (index, evidence) in self.evidence.iter().enumerate() {
            append_field(
                &mut text,
                &format!("evidence_{index}_kind"),
                evidence.kind.as_text(),
            );
            append_field(
                &mut text,
                &format!("evidence_{index}_repository"),
                hex_encode(evidence.repository()),
            );
            append_field(
                &mut text,
                &format!("evidence_{index}_artifact"),
                hex_encode(evidence.artifact()),
            );
            append_field(
                &mut text,
                &format!("evidence_{index}_revision"),
                hex_encode(evidence.revision_binding()),
            );
        }
        text
    }

    /// Decode and fully revalidate one canonical `ADA-EXPERIMENT-V2` artifact.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError`] for malformed/non-canonical text, unsupported
    /// versions, invalid nested semantic/workload/implementation/objective
    /// artifacts, inconsistent semantic/implementation identity, invalid
    /// provenance, or missing evidence required by measured objectives.
    pub fn from_canonical_text(text: &str) -> Result<Self, ExperimentError> {
        if text.len() > MAX_EXPERIMENT_BYTES || text.contains('\r') || !text.ends_with('\n') {
            return malformed("artifact exceeds limit, contains CR, or lacks final newline");
        }
        let mut lines = text.lines();
        if lines.next() != Some(EXPERIMENT_HEADER) {
            return malformed("invalid experiment header");
        }
        let version = parse_u16(next_value(&mut lines, "version")?)?;
        if version != EXPERIMENT_VERSION {
            return Err(ExperimentError::UnsupportedVersion(version));
        }
        let repository = hex_decode(next_value(&mut lines, "producer_repository")?)?;
        let revision = next_value(&mut lines, "producer_revision")?.to_string();
        let artifact_identity = hex_decode(next_value(&mut lines, "artifact_identity")?)?;
        let artifact_sha256 = next_value(&mut lines, "artifact_sha256")?.to_string();
        let semantic_text = hex_decode(next_value(&mut lines, "semantic_text")?)?;
        let workload_text = hex_decode(next_value(&mut lines, "workload_text")?)?;
        let implementation_text = hex_decode(next_value(&mut lines, "implementation_text")?)?;
        let objective_text = hex_decode(next_value(&mut lines, "objective_text")?)?;
        let evidence_count = parse_usize(next_value(&mut lines, "evidence_count")?)?;
        if evidence_count > MAX_EVIDENCE_BINDINGS {
            return Err(ExperimentError::TooManyEvidenceBindings);
        }
        let mut evidence = Vec::with_capacity(evidence_count);
        for index in 0..evidence_count {
            let kind =
                EvidenceKind::parse(next_value(&mut lines, &format!("evidence_{index}_kind"))?)?;
            let repository = hex_decode(next_value(
                &mut lines,
                &format!("evidence_{index}_repository"),
            )?)?;
            let artifact = hex_decode(next_value(
                &mut lines,
                &format!("evidence_{index}_artifact"),
            )?)?;
            let revision_binding = hex_decode(next_value(
                &mut lines,
                &format!("evidence_{index}_revision"),
            )?)?;
            evidence.push(EvidenceBinding {
                kind,
                repository,
                artifact,
                revision_binding,
            });
        }
        if lines.next().is_some() {
            return malformed("unexpected trailing experiment field");
        }

        let record = Self::new(ExperimentSpec {
            semantic: SemanticProgram::from_canonical_text(&semantic_text)
                .map_err(|error| ExperimentError::Semantic(error.to_string()))?,
            workload: WorkloadContract::from_canonical_text(&workload_text)
                .map_err(|error| ExperimentError::Workload(error.to_string()))?,
            implementation: ImplementationPlan::from_canonical_text(&implementation_text)
                .map_err(|error| ExperimentError::Implementation(error.to_string()))?,
            objective: ObjectiveVector::from_canonical_text(&objective_text)
                .map_err(|error| ExperimentError::Objective(error.to_string()))?,
            provenance: ProducerProvenance::new(
                repository,
                revision,
                artifact_identity,
                artifact_sha256,
            )?,
            evidence,
        })?;
        if record.to_canonical_text() != text {
            return malformed("experiment artifact is not canonical");
        }
        Ok(record)
    }
}

pub(crate) fn append_field(text: &mut String, key: &str, value: impl Display) {
    text.push_str(key);
    text.push('=');
    text.push_str(&value.to_string());
    text.push('\n');
}

pub(crate) fn next_value<'a>(
    lines: &mut std::str::Lines<'a>,
    key: &str,
) -> Result<&'a str, ExperimentError> {
    let line = lines
        .next()
        .ok_or_else(|| ExperimentError::MalformedCanonical(format!("missing field {key}")))?;
    let prefix = format!("{key}=");
    line.strip_prefix(&prefix)
        .ok_or_else(|| ExperimentError::MalformedCanonical(format!("expected field {key}")))
}

pub(crate) fn parse_u16(value: &str) -> Result<u16, ExperimentError> {
    value
        .parse()
        .map_err(|_| ExperimentError::MalformedCanonical("invalid u16".into()))
}

pub(crate) fn parse_usize(value: &str) -> Result<usize, ExperimentError> {
    value
        .parse()
        .map_err(|_| ExperimentError::MalformedCanonical("invalid usize".into()))
}

pub(crate) fn hex_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value.bytes() {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

pub(crate) fn hex_decode(value: &str) -> Result<String, ExperimentError> {
    if value.len() % 2 != 0 {
        return malformed("odd-length hex field");
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = nibble(pair[0])?;
        let low = nibble(pair[1])?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes)
        .map_err(|_| ExperimentError::MalformedCanonical("hex field is not UTF-8".into()))
}

fn nibble(byte: u8) -> Result<u8, ExperimentError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(ExperimentError::MalformedCanonical(
            "non-canonical hex".into(),
        )),
    }
}
