//! Deterministic hand-rolled JSON codec for `SemanticQualificationRecord`.
//!
//! No third-party JSON crates are used. The encoder emits compact UTF-8 JSON
//! with fixed key order from the schema. The decoder rejects unknown
//! properties (`additionalProperties: false`), trailing data, and non-schema
//! value kinds.

use std::fmt::Write;

use super::{
    EvidenceRef, ExternalDiagnostics, MaskStateContract, NumericPolicy, PriorArtStatus,
    PriorArtStatusKind, QualificationVerdictJson, ReferenceDefinition,
    SEMANTIC_QUALIFICATION_SCHEMA_VERSION, SemanticQualificationError, SemanticQualificationRecord,
    SemanticQualificationSpec,
};

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(i64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

struct Parser<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            index: 0,
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, SemanticQualificationError> {
        self.skip_ws();
        let Some(byte) = self.peek() else {
            return Err(SemanticQualificationError::MalformedJson(
                "unexpected end of input".into(),
            ));
        };
        match byte {
            b'n' => self.parse_literal(b"null", JsonValue::Null),
            b't' => self.parse_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.parse_literal(b"false", JsonValue::Bool(false)),
            b'"' => Ok(JsonValue::String(self.parse_string()?)),
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'-' | b'0'..=b'9' => Ok(JsonValue::Number(self.parse_integer()?)),
            _ => Err(SemanticQualificationError::MalformedJson(format!(
                "unexpected byte 0x{byte:02x}"
            ))),
        }
    }

    fn parse_document(&mut self) -> Result<JsonValue, SemanticQualificationError> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.index != self.bytes.len() {
            return Err(SemanticQualificationError::MalformedJson(
                "trailing data after JSON value".into(),
            ));
        }
        Ok(value)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn bump(&mut self) -> Result<u8, SemanticQualificationError> {
        let byte = self.peek().ok_or_else(|| {
            SemanticQualificationError::MalformedJson("unexpected end of input".into())
        })?;
        self.index += 1;
        Ok(byte)
    }

    fn skip_ws(&mut self) {
        while let Some(byte) = self.peek() {
            match byte {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    self.index += 1;
                }
                _ => break,
            }
        }
    }

    fn parse_literal(
        &mut self,
        literal: &[u8],
        value: JsonValue,
    ) -> Result<JsonValue, SemanticQualificationError> {
        for expected in literal {
            let got = self.bump()?;
            if got != *expected {
                return Err(SemanticQualificationError::MalformedJson(
                    "invalid literal".into(),
                ));
            }
        }
        Ok(value)
    }

    fn parse_integer(&mut self) -> Result<i64, SemanticQualificationError> {
        let start = self.index;
        if self.peek() == Some(b'-') {
            self.index += 1;
        }
        let Some(first) = self.peek() else {
            return Err(SemanticQualificationError::MalformedJson(
                "truncated number".into(),
            ));
        };
        if !first.is_ascii_digit() {
            return Err(SemanticQualificationError::MalformedJson(
                "invalid number".into(),
            ));
        }
        if first == b'0' {
            self.index += 1;
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(SemanticQualificationError::MalformedJson(
                    "leading zeros are not allowed".into(),
                ));
            }
        } else {
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.index += 1;
            }
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(SemanticQualificationError::MalformedJson(
                "floating-point numbers are not accepted by this schema codec".into(),
            ));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.index])
            .map_err(|_| SemanticQualificationError::MalformedJson("number is not UTF-8".into()))?;
        text.parse::<i64>().map_err(|_| {
            SemanticQualificationError::MalformedJson(format!("integer out of range: {text}"))
        })
    }

    fn parse_string(&mut self) -> Result<String, SemanticQualificationError> {
        if self.bump()? != b'"' {
            return Err(SemanticQualificationError::MalformedJson(
                "string must start with quote".into(),
            ));
        }
        let mut out = String::new();
        loop {
            let byte = self.bump()?;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self.bump()?;
                    match escaped {
                        b'"' | b'\\' | b'/' => out.push(char::from(escaped)),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut code = 0_u32;
                            for _ in 0..4 {
                                let hex = self.bump()?;
                                code <<= 4;
                                code |= u32::from(hex_nibble(hex)?);
                            }
                            let ch = char::from_u32(code).ok_or_else(|| {
                                SemanticQualificationError::MalformedJson(
                                    "invalid unicode escape".into(),
                                )
                            })?;
                            out.push(ch);
                        }
                        _ => {
                            return Err(SemanticQualificationError::MalformedJson(
                                "invalid string escape".into(),
                            ));
                        }
                    }
                }
                0x00..=0x1f => {
                    return Err(SemanticQualificationError::MalformedJson(
                        "unescaped control character in string".into(),
                    ));
                }
                _ => {
                    // Restart UTF-8 decode from this byte.
                    self.index -= 1;
                    let (ch, len) = decode_utf8_char(&self.bytes[self.index..])?;
                    out.push(ch);
                    self.index += len;
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, SemanticQualificationError> {
        if self.bump()? != b'[' {
            return Err(SemanticQualificationError::MalformedJson(
                "array must start with '['".into(),
            ));
        }
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.index += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.bump()? {
                b']' => return Ok(JsonValue::Array(items)),
                b',' => self.skip_ws(),
                _ => {
                    return Err(SemanticQualificationError::MalformedJson(
                        "invalid array separator".into(),
                    ));
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, SemanticQualificationError> {
        if self.bump()? != b'{' {
            return Err(SemanticQualificationError::MalformedJson(
                "object must start with '{'".into(),
            ));
        }
        self.skip_ws();
        let mut fields = Vec::new();
        if self.peek() == Some(b'}') {
            self.index += 1;
            return Ok(JsonValue::Object(fields));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bump()? != b':' {
                return Err(SemanticQualificationError::MalformedJson(
                    "object entry missing ':'".into(),
                ));
            }
            let value = self.parse_value()?;
            if fields.iter().any(|(existing, _)| existing == &key) {
                return Err(SemanticQualificationError::MalformedJson(format!(
                    "duplicate object key `{key}`"
                )));
            }
            fields.push((key, value));
            self.skip_ws();
            match self.bump()? {
                b'}' => return Ok(JsonValue::Object(fields)),
                b',' => {}
                _ => {
                    return Err(SemanticQualificationError::MalformedJson(
                        "invalid object separator".into(),
                    ));
                }
            }
        }
    }
}

fn hex_nibble(byte: u8) -> Result<u8, SemanticQualificationError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(SemanticQualificationError::MalformedJson(
            "invalid hex digit in unicode escape".into(),
        )),
    }
}

fn decode_utf8_char(bytes: &[u8]) -> Result<(char, usize), SemanticQualificationError> {
    let Some(first) = bytes.first().copied() else {
        return Err(SemanticQualificationError::MalformedJson(
            "unexpected end of string".into(),
        ));
    };
    let width = match first {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => {
            return Err(SemanticQualificationError::MalformedJson(
                "invalid UTF-8 in string".into(),
            ));
        }
    };
    if bytes.len() < width {
        return Err(SemanticQualificationError::MalformedJson(
            "truncated UTF-8 sequence".into(),
        ));
    }
    let slice = &bytes[..width];
    let text = std::str::from_utf8(slice)
        .map_err(|_| SemanticQualificationError::MalformedJson("invalid UTF-8 in string".into()))?;
    let ch = text
        .chars()
        .next()
        .ok_or_else(|| SemanticQualificationError::MalformedJson("empty UTF-8 decode".into()))?;
    Ok((ch, width))
}

fn escape_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn encode_evidence_ref(value: &EvidenceRef) -> String {
    let mut out = String::from('{');
    out.push_str("\"kind\":");
    out.push_str(&escape_string(&value.kind));
    out.push_str(",\"repository\":");
    out.push_str(&escape_string(&value.repository));
    out.push_str(",\"commit\":");
    out.push_str(&escape_string(&value.commit));
    out.push_str(",\"artifact\":");
    out.push_str(&escape_string(&value.artifact));
    out.push_str(",\"sha256\":");
    out.push_str(&escape_string(&value.sha256));
    out.push('}');
    out
}

fn encode_evidence_list(values: &[EvidenceRef]) -> String {
    let mut out = String::from('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&encode_evidence_ref(value));
    }
    out.push(']');
    out
}

pub(super) fn encode_record(record: &SemanticQualificationRecord) -> String {
    let mut out = String::from('{');
    out.push_str("\"schema_version\":");
    out.push_str(&record.schema_version().to_string());
    out.push_str(",\"candidate_id\":");
    out.push_str(&escape_string(record.candidate_id()));
    out.push_str(",\"semantic_id\":");
    out.push_str(&escape_string(record.semantic_id()));
    out.push_str(",\"semantic_version\":");
    out.push_str(&record.semantic_version().to_string());

    let reference = record.reference_definition();
    out.push_str(",\"reference_definition\":{");
    out.push_str("\"repository\":");
    out.push_str(&escape_string(&reference.repository));
    out.push_str(",\"commit\":");
    out.push_str(&escape_string(&reference.commit));
    out.push_str(",\"artifact\":");
    out.push_str(&escape_string(&reference.artifact));
    out.push_str(",\"sha256\":");
    out.push_str(&escape_string(&reference.sha256));
    out.push('}');

    out.push_str(",\"invariants\":[");
    for (index, invariant) in record.invariants().iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&escape_string(invariant));
    }
    out.push(']');

    let mask = record.mask_state_contract();
    out.push_str(",\"mask_state_contract\":{");
    out.push_str("\"mask\":");
    out.push_str(&escape_string(&mask.mask));
    out.push_str(",\"state\":");
    out.push_str(&escape_string(&mask.state));
    out.push('}');

    let numeric = record.numeric_policy();
    out.push_str(",\"numeric_policy\":{");
    out.push_str("\"reference_precision\":");
    out.push_str(&escape_string(&numeric.reference_precision));
    out.push_str(",\"non_finite_policy\":");
    out.push_str(&escape_string(&numeric.non_finite_policy));
    out.push('}');

    out.push_str(",\"oracle_fixtures\":");
    out.push_str(&encode_evidence_list(record.oracle_fixtures()));
    out.push_str(",\"adversarial_cases\":");
    out.push_str(&encode_evidence_list(record.adversarial_cases()));
    out.push_str(",\"task_evidence\":");
    out.push_str(&encode_evidence_list(record.task_evidence()));

    let diagnostics = record.external_diagnostics();
    out.push_str(",\"external_diagnostics\":{");
    out.push_str("\"itd\":");
    out.push_str(&encode_evidence_list(&diagnostics.itd));
    out.push_str(",\"tdi\":");
    out.push_str(&encode_evidence_list(&diagnostics.tdi));
    out.push('}');

    out.push_str(",\"logical_cost_evidence\":");
    out.push_str(&encode_evidence_list(record.logical_cost_evidence()));

    let prior = record.prior_art_status();
    out.push_str(",\"prior_art_status\":{");
    out.push_str("\"status\":");
    out.push_str(&escape_string(prior.status.as_str()));
    out.push_str(",\"references\":");
    out.push_str(&encode_evidence_list(&prior.references));
    out.push('}');

    out.push_str(",\"verdict\":");
    out.push_str(&escape_string(record.verdict().as_str()));
    out.push('}');
    out
}

struct ObjectView<'a> {
    path: &'static str,
    fields: &'a [(String, JsonValue)],
}

impl<'a> ObjectView<'a> {
    fn require(&'a self, key: &'static str) -> Result<&'a JsonValue, SemanticQualificationError> {
        self.fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
            .ok_or(SemanticQualificationError::MissingProperty {
                path: self.path,
                key,
            })
    }

    fn ensure_only(&self, allowed: &[&str]) -> Result<(), SemanticQualificationError> {
        for (key, _) in self.fields {
            if !allowed.iter().any(|name| name == key) {
                return Err(SemanticQualificationError::UnknownProperty {
                    path: self.path,
                    key: key.clone(),
                });
            }
        }
        Ok(())
    }
}

fn as_object<'a>(
    value: &'a JsonValue,
    field: &'static str,
) -> Result<ObjectView<'a>, SemanticQualificationError> {
    match value {
        JsonValue::Object(fields) => Ok(ObjectView {
            path: field,
            fields,
        }),
        _ => Err(SemanticQualificationError::TypeMismatch {
            field,
            expected: "object",
        }),
    }
}

fn as_array<'a>(
    value: &'a JsonValue,
    field: &'static str,
) -> Result<&'a [JsonValue], SemanticQualificationError> {
    match value {
        JsonValue::Array(items) => Ok(items),
        _ => Err(SemanticQualificationError::TypeMismatch {
            field,
            expected: "array",
        }),
    }
}

fn as_string<'a>(
    value: &'a JsonValue,
    field: &'static str,
) -> Result<&'a str, SemanticQualificationError> {
    match value {
        JsonValue::String(text) => Ok(text),
        _ => Err(SemanticQualificationError::TypeMismatch {
            field,
            expected: "string",
        }),
    }
}

fn as_u16(value: &JsonValue, field: &'static str) -> Result<u16, SemanticQualificationError> {
    match value {
        JsonValue::Number(number) => {
            u16::try_from(*number).map_err(|_| SemanticQualificationError::TypeMismatch {
                field,
                expected: "unsigned 16-bit integer",
            })
        }
        _ => Err(SemanticQualificationError::TypeMismatch {
            field,
            expected: "integer",
        }),
    }
}

fn as_u32(value: &JsonValue, field: &'static str) -> Result<u32, SemanticQualificationError> {
    match value {
        JsonValue::Number(number) => {
            u32::try_from(*number).map_err(|_| SemanticQualificationError::TypeMismatch {
                field,
                expected: "unsigned 32-bit integer",
            })
        }
        _ => Err(SemanticQualificationError::TypeMismatch {
            field,
            expected: "integer",
        }),
    }
}

fn decode_evidence_ref(
    value: &JsonValue,
    path: &'static str,
) -> Result<EvidenceRef, SemanticQualificationError> {
    let object = as_object(value, path)?;
    object.ensure_only(&["kind", "repository", "commit", "artifact", "sha256"])?;
    Ok(EvidenceRef {
        kind: as_string(object.require("kind")?, "evidence.kind")?.to_owned(),
        repository: as_string(object.require("repository")?, "evidence.repository")?.to_owned(),
        commit: as_string(object.require("commit")?, "evidence.commit")?.to_owned(),
        artifact: as_string(object.require("artifact")?, "evidence.artifact")?.to_owned(),
        sha256: as_string(object.require("sha256")?, "evidence.sha256")?.to_owned(),
    })
}

fn decode_evidence_list(
    value: &JsonValue,
    path: &'static str,
) -> Result<Vec<EvidenceRef>, SemanticQualificationError> {
    let items = as_array(value, path)?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(decode_evidence_ref(item, path)?);
    }
    Ok(out)
}

fn decode_reference_definition(
    value: &JsonValue,
) -> Result<ReferenceDefinition, SemanticQualificationError> {
    let object = as_object(value, "reference_definition")?;
    object.ensure_only(&["repository", "commit", "artifact", "sha256"])?;
    Ok(ReferenceDefinition {
        repository: as_string(
            object.require("repository")?,
            "reference_definition.repository",
        )?
        .to_owned(),
        commit: as_string(object.require("commit")?, "reference_definition.commit")?.to_owned(),
        artifact: as_string(object.require("artifact")?, "reference_definition.artifact")?
            .to_owned(),
        sha256: as_string(object.require("sha256")?, "reference_definition.sha256")?.to_owned(),
    })
}

#[allow(clippy::too_many_lines)]
pub(super) fn decode_record(
    text: &str,
) -> Result<SemanticQualificationRecord, SemanticQualificationError> {
    if text.starts_with('\u{feff}') {
        return Err(SemanticQualificationError::MalformedJson(
            "BOM is not accepted".into(),
        ));
    }
    let value = Parser::new(text).parse_document()?;
    let object = as_object(&value, "$")?;
    object.ensure_only(&[
        "schema_version",
        "candidate_id",
        "semantic_id",
        "semantic_version",
        "reference_definition",
        "invariants",
        "mask_state_contract",
        "numeric_policy",
        "oracle_fixtures",
        "adversarial_cases",
        "task_evidence",
        "external_diagnostics",
        "logical_cost_evidence",
        "prior_art_status",
        "verdict",
    ])?;

    let schema_version = as_u16(object.require("schema_version")?, "schema_version")?;
    if schema_version != SEMANTIC_QUALIFICATION_SCHEMA_VERSION {
        return Err(SemanticQualificationError::UnsupportedSchemaVersion(
            schema_version,
        ));
    }

    let invariants_value = as_array(object.require("invariants")?, "invariants")?;
    let mut invariants = Vec::with_capacity(invariants_value.len());
    for item in invariants_value {
        invariants.push(as_string(item, "invariants")?.to_owned());
    }

    let mask_object = as_object(
        object.require("mask_state_contract")?,
        "mask_state_contract",
    )?;
    mask_object.ensure_only(&["mask", "state"])?;
    let mask_state_contract = MaskStateContract {
        mask: as_string(mask_object.require("mask")?, "mask_state_contract.mask")?.to_owned(),
        state: as_string(mask_object.require("state")?, "mask_state_contract.state")?.to_owned(),
    };

    let numeric_object = as_object(object.require("numeric_policy")?, "numeric_policy")?;
    numeric_object.ensure_only(&["reference_precision", "non_finite_policy"])?;
    let numeric_policy = NumericPolicy {
        reference_precision: as_string(
            numeric_object.require("reference_precision")?,
            "numeric_policy.reference_precision",
        )?
        .to_owned(),
        non_finite_policy: as_string(
            numeric_object.require("non_finite_policy")?,
            "numeric_policy.non_finite_policy",
        )?
        .to_owned(),
    };

    let diagnostics_object = as_object(
        object.require("external_diagnostics")?,
        "external_diagnostics",
    )?;
    diagnostics_object.ensure_only(&["itd", "tdi"])?;
    let external_diagnostics = ExternalDiagnostics {
        itd: decode_evidence_list(
            diagnostics_object.require("itd")?,
            "external_diagnostics.itd",
        )?,
        tdi: decode_evidence_list(
            diagnostics_object.require("tdi")?,
            "external_diagnostics.tdi",
        )?,
    };

    let prior_object = as_object(object.require("prior_art_status")?, "prior_art_status")?;
    prior_object.ensure_only(&["status", "references"])?;
    let prior_art_status = PriorArtStatus {
        status: PriorArtStatusKind::parse(as_string(
            prior_object.require("status")?,
            "prior_art_status.status",
        )?)?,
        references: decode_evidence_list(
            prior_object.require("references")?,
            "prior_art_status.references",
        )?,
    };

    let verdict =
        QualificationVerdictJson::parse(as_string(object.require("verdict")?, "verdict")?)?;

    SemanticQualificationRecord::new(SemanticQualificationSpec {
        candidate_id: as_string(object.require("candidate_id")?, "candidate_id")?.to_owned(),
        semantic_id: as_string(object.require("semantic_id")?, "semantic_id")?.to_owned(),
        semantic_version: as_u32(object.require("semantic_version")?, "semantic_version")?,
        reference_definition: decode_reference_definition(object.require("reference_definition")?)?,
        invariants,
        mask_state_contract,
        numeric_policy,
        oracle_fixtures: decode_evidence_list(
            object.require("oracle_fixtures")?,
            "oracle_fixtures",
        )?,
        adversarial_cases: decode_evidence_list(
            object.require("adversarial_cases")?,
            "adversarial_cases",
        )?,
        task_evidence: decode_evidence_list(object.require("task_evidence")?, "task_evidence")?,
        external_diagnostics,
        logical_cost_evidence: decode_evidence_list(
            object.require("logical_cost_evidence")?,
            "logical_cost_evidence",
        )?,
        prior_art_status,
        verdict,
    })
}
