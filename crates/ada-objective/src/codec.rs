//! Strict canonical codec for objective vectors.

use super::{
    CorrectnessStatus, EstimatedCost, LogicalCost, MAX_OBJECTIVE_TEXT_BYTES, MAX_QUALITY_METRICS,
    MeasuredCost, NumericalObjectives, ObjectiveDirection, ObjectiveError, ObjectiveVector,
    QualityMetric,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Canonical objective-vector header.
pub const OBJECTIVE_TEXT_HEADER: &str = "ADA-OBJECTIVE-V1";
/// Canonical objective-vector schema version.
pub const OBJECTIVE_VECTOR_VERSION: u16 = 1;

const BASE_FIELDS: &[&str] = &[
    "correctness",
    "max_abs_error",
    "max_ulp_error",
    "normalization_error",
    "logical_flops",
    "logical_qk_evaluations",
    "logical_transcendental_operations",
    "logical_value_operations",
    "estimated_bytes_moved",
    "estimated_workspace_bytes",
    "estimated_kv_cache_bytes",
    "estimated_index_construction",
    "estimated_communication_bytes",
    "estimated_reduction_operations",
    "measured_latency_ns",
    "measured_energy_nj",
    "quality_count",
];

impl ObjectiveVector {
    /// Encode a validated vector in canonical field order.
    pub(crate) fn encode_canonical(&self) -> String {
        let mut text = format!("{OBJECTIVE_TEXT_HEADER}\n");
        append(&mut text, "correctness", self.correctness);
        append(
            &mut text,
            "max_abs_error",
            encode_float(self.numerical.max_abs_error),
        );
        append(
            &mut text,
            "max_ulp_error",
            encode_u64(self.numerical.max_ulp_error),
        );
        append(
            &mut text,
            "normalization_error",
            encode_float(self.numerical.normalization_error),
        );
        append(&mut text, "logical_flops", encode_u64(self.logical.flops));
        append(
            &mut text,
            "logical_qk_evaluations",
            encode_u64(self.logical.qk_evaluations),
        );
        append(
            &mut text,
            "logical_transcendental_operations",
            encode_u64(self.logical.transcendental_operations),
        );
        append(
            &mut text,
            "logical_value_operations",
            encode_u64(self.logical.value_operations),
        );
        append(
            &mut text,
            "estimated_bytes_moved",
            encode_u64(self.estimated.bytes_moved),
        );
        append(
            &mut text,
            "estimated_workspace_bytes",
            encode_u64(self.estimated.workspace_bytes),
        );
        append(
            &mut text,
            "estimated_kv_cache_bytes",
            encode_u64(self.estimated.kv_cache_bytes),
        );
        append(
            &mut text,
            "estimated_index_construction",
            encode_u64(self.estimated.index_construction),
        );
        append(
            &mut text,
            "estimated_communication_bytes",
            encode_u64(self.estimated.communication_bytes),
        );
        append(
            &mut text,
            "estimated_reduction_operations",
            encode_u64(self.estimated.reduction_operations),
        );
        append(
            &mut text,
            "measured_latency_ns",
            encode_u64(self.measured.latency_ns),
        );
        append(
            &mut text,
            "measured_energy_nj",
            encode_u64(self.measured.energy_nj),
        );
        append(&mut text, "quality_count", self.quality.len());
        for (index, metric) in self.quality.iter().enumerate() {
            append(
                &mut text,
                &format!("quality_{index}_name"),
                hex_encode(metric.name.as_bytes()),
            );
            append(
                &mut text,
                &format!("quality_{index}_direction"),
                metric.direction,
            );
            append(
                &mut text,
                &format!("quality_{index}_value"),
                encode_float(metric.value),
            );
        }
        text
    }

    /// Decode canonical objective-vector text.
    pub(crate) fn decode_canonical(text: &str) -> Result<Self, ObjectiveError> {
        let fields = parse_fields(text)?;
        let quality_count = parse_u64(&fields, "quality_count")?;
        if quality_count > MAX_QUALITY_METRICS {
            return Err(ObjectiveError::ExceedsLimit {
                field: "quality_count",
                value: quality_count,
                maximum: MAX_QUALITY_METRICS,
            });
        }
        let count = usize::try_from(quality_count).unwrap_or(usize::MAX);
        let expected_fields = BASE_FIELDS.len().saturating_add(count.saturating_mul(3));
        if fields.len() != expected_fields {
            return Err(ObjectiveError::MalformedCanonical(
                "field set is incomplete or has unknown keys".into(),
            ));
        }
        for index in 0..count {
            for suffix in ["name", "direction", "value"] {
                let key = format!("quality_{index}_{suffix}");
                if !fields.contains_key(key.as_str()) {
                    return Err(ObjectiveError::MalformedCanonical(
                        "quality fields are not contiguous".into(),
                    ));
                }
            }
        }
        let mut quality = Vec::with_capacity(count);
        for index in 0..count {
            let name = String::from_utf8(hex_decode(field(
                &fields,
                &format!("quality_{index}_name"),
            )?)?)
            .map_err(|_| ObjectiveError::MalformedCanonical("quality name is not UTF-8".into()))?;
            let direction =
                parse_direction(field(&fields, &format!("quality_{index}_direction"))?)?;
            let value = decode_float(field(&fields, &format!("quality_{index}_value"))?)?;
            quality.push(QualityMetric::new(name, value, direction)?);
        }
        let vector = ObjectiveVector::from_parts(
            parse_correctness(field(&fields, "correctness")?)?,
            NumericalObjectives {
                max_abs_error: decode_float(field(&fields, "max_abs_error")?)?,
                max_ulp_error: decode_u64(field(&fields, "max_ulp_error")?)?,
                normalization_error: decode_float(field(&fields, "normalization_error")?)?,
            },
            LogicalCost {
                flops: decode_u64(field(&fields, "logical_flops")?)?,
                qk_evaluations: decode_u64(field(&fields, "logical_qk_evaluations")?)?,
                transcendental_operations: decode_u64(field(
                    &fields,
                    "logical_transcendental_operations",
                )?)?,
                value_operations: decode_u64(field(&fields, "logical_value_operations")?)?,
            },
            EstimatedCost {
                bytes_moved: decode_u64(field(&fields, "estimated_bytes_moved")?)?,
                workspace_bytes: decode_u64(field(&fields, "estimated_workspace_bytes")?)?,
                kv_cache_bytes: decode_u64(field(&fields, "estimated_kv_cache_bytes")?)?,
                index_construction: decode_u64(field(&fields, "estimated_index_construction")?)?,
                communication_bytes: decode_u64(field(&fields, "estimated_communication_bytes")?)?,
                reduction_operations: decode_u64(field(
                    &fields,
                    "estimated_reduction_operations",
                )?)?,
            },
            MeasuredCost {
                latency_ns: decode_u64(field(&fields, "measured_latency_ns")?)?,
                energy_nj: decode_u64(field(&fields, "measured_energy_nj")?)?,
            },
            quality,
        )?;
        if vector.encode_canonical() != text {
            return Err(ObjectiveError::MalformedCanonical(
                "text is valid but not canonical".into(),
            ));
        }
        Ok(vector)
    }
}

fn append(text: &mut String, key: &str, value: impl std::fmt::Display) {
    let _ = writeln!(text, "{key}={value}");
}

fn encode_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".into(), |value| value.to_string())
}

fn parse_u64(fields: &BTreeMap<&str, &str>, key: &str) -> Result<u64, ObjectiveError> {
    field(fields, key)?.parse::<u64>().map_err(|_| {
        ObjectiveError::MalformedCanonical(format!("{key} is not an unsigned decimal integer"))
    })
}

fn decode_u64(value: &str) -> Result<Option<u64>, ObjectiveError> {
    if value == "none" {
        return Ok(None);
    }
    value
        .parse::<u64>()
        .map(Some)
        .map_err(|_| ObjectiveError::MalformedCanonical("invalid unsigned objective".into()))
}

fn encode_float(value: Option<f64>) -> String {
    value.map_or_else(
        || "none".into(),
        |value| format!("{:016x}", value.to_bits()),
    )
}

fn decode_float(value: &str) -> Result<Option<f64>, ObjectiveError> {
    if value == "none" {
        return Ok(None);
    }
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(ObjectiveError::MalformedCanonical(
            "float objective is not a lowercase bit lane".into(),
        ));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|_| ObjectiveError::MalformedCanonical("invalid float bits".into()))?;
    Ok(Some(f64::from_bits(bits)))
}

fn parse_correctness(value: &str) -> Result<CorrectnessStatus, ObjectiveError> {
    match value {
        "unknown" => Ok(CorrectnessStatus::Unknown),
        "falsified" => Ok(CorrectnessStatus::Falsified),
        "provisional" => Ok(CorrectnessStatus::Provisional),
        "qualified" => Ok(CorrectnessStatus::Qualified),
        _ => Err(ObjectiveError::MalformedCanonical(
            "unknown correctness status".into(),
        )),
    }
}

fn parse_direction(value: &str) -> Result<ObjectiveDirection, ObjectiveError> {
    match value {
        "min" => Ok(ObjectiveDirection::Minimize),
        "max" => Ok(ObjectiveDirection::Maximize),
        _ => Err(ObjectiveError::MalformedCanonical(
            "unknown objective direction".into(),
        )),
    }
}

fn parse_fields(text: &str) -> Result<BTreeMap<&str, &str>, ObjectiveError> {
    if text.len() > MAX_OBJECTIVE_TEXT_BYTES || text.contains('\r') {
        return Err(ObjectiveError::MalformedCanonical(
            "text exceeds its limit or contains CR".into(),
        ));
    }
    if !text.ends_with('\n') {
        return Err(ObjectiveError::MalformedCanonical(
            "text must end with a newline".into(),
        ));
    }
    let mut lines = text.lines();
    if lines.next() != Some(OBJECTIVE_TEXT_HEADER) {
        return Err(ObjectiveError::MalformedCanonical(
            "unsupported or non-canonical header".into(),
        ));
    }
    let mut fields = BTreeMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Err(ObjectiveError::MalformedCanonical(
                "field is missing '='".into(),
            ));
        };
        if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
            return Err(ObjectiveError::MalformedCanonical(
                "empty, duplicate, or ambiguous field".into(),
            ));
        }
    }
    if BASE_FIELDS.iter().any(|key| !fields.contains_key(key)) {
        return Err(ObjectiveError::MalformedCanonical(
            "base field set is incomplete".into(),
        ));
    }
    Ok(fields)
}

fn field<'a>(fields: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str, ObjectiveError> {
    fields
        .get(key)
        .copied()
        .ok_or_else(|| ObjectiveError::MalformedCanonical(format!("missing field {key}")))
}

fn hex_encode(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(value: &str) -> Result<Vec<u8>, ObjectiveError> {
    if value.is_empty() || value.len() % 2 != 0 {
        return Err(ObjectiveError::MalformedCanonical(
            "hex field is empty or odd-sized".into(),
        ));
    }
    let mut decoded = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = hex_value(pair[0]).ok_or_else(|| {
            ObjectiveError::MalformedCanonical("hex field contains non-canonical digits".into())
        })?;
        let low = hex_value(pair[1]).ok_or_else(|| {
            ObjectiveError::MalformedCanonical("hex field contains non-canonical digits".into())
        })?;
        decoded.push((high << 4) | low);
    }
    Ok(decoded)
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}
