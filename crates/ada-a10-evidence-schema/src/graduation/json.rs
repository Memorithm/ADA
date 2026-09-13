//! Unicode-interoperable wrapper around the frozen A11 JSON codec.
//!
//! The original dependency-free codec is retained verbatim in `json_legacy.rs`.
//! This wrapper performs one narrowly scoped JSON-string normalization before
//! decoding: valid UTF-16 surrogate pairs written as `\uXXXX\uXXXX` are
//! combined into the corresponding Unicode scalar value. Lone or malformed
//! surrogates fail closed. Other JSON escapes are left for the legacy parser.

use super::{
    EvidenceRef, ExternalDiagnostics, MaskStateContract, NumericPolicy, PriorArtStatus,
    PriorArtStatusKind, QualificationVerdictJson, ReferenceDefinition,
    SEMANTIC_QUALIFICATION_SCHEMA_VERSION, SemanticQualificationError, SemanticQualificationRecord,
    SemanticQualificationSpec,
};

#[path = "json_legacy.rs"]
mod legacy;

pub(super) fn encode_record(record: &SemanticQualificationRecord) -> String {
    legacy::encode_record(record)
}

pub(super) fn decode_record(
    text: &str,
) -> Result<SemanticQualificationRecord, SemanticQualificationError> {
    let normalized = normalize_surrogate_pairs(text)?;
    legacy::decode_record(&normalized)
}

fn normalize_surrogate_pairs(text: &str) -> Result<String, SemanticQualificationError> {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0usize;
    let mut in_string = false;

    while index < bytes.len() {
        let byte = bytes[index];
        if !in_string {
            let ch = decode_char_at(text, index)?;
            out.push(ch);
            index += ch.len_utf8();
            if ch == '"' {
                in_string = true;
            }
            continue;
        }

        if byte == b'"' {
            out.push('"');
            index += 1;
            in_string = false;
            continue;
        }

        if byte != b'\\' {
            let ch = decode_char_at(text, index)?;
            out.push(ch);
            index += ch.len_utf8();
            continue;
        }

        let Some(&escaped) = bytes.get(index + 1) else {
            return Err(malformed("truncated string escape"));
        };
        if escaped != b'u' {
            out.push('\\');
            out.push(char::from(escaped));
            index += 2;
            continue;
        }

        let first = parse_hex_quad(bytes, index + 2)?;
        if (0xd800..=0xdbff).contains(&first) {
            if bytes.get(index + 6) != Some(&b'\\') || bytes.get(index + 7) != Some(&b'u') {
                return Err(malformed("high surrogate is not followed by a low surrogate"));
            }
            let second = parse_hex_quad(bytes, index + 8)?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err(malformed("high surrogate is not followed by a low surrogate"));
            }
            let scalar = 0x1_0000
                + (((u32::from(first) - 0xd800) << 10) | (u32::from(second) - 0xdc00));
            let ch = char::from_u32(scalar)
                .ok_or_else(|| malformed("invalid unicode surrogate pair"))?;
            out.push(ch);
            index += 12;
            continue;
        }

        if (0xdc00..=0xdfff).contains(&first) {
            return Err(malformed("low surrogate appears without a leading high surrogate"));
        }

        out.push_str(&text[index..index + 6]);
        index += 6;
    }

    Ok(out)
}

fn parse_hex_quad(bytes: &[u8], start: usize) -> Result<u16, SemanticQualificationError> {
    let end = start
        .checked_add(4)
        .ok_or_else(|| malformed("unicode escape index overflow"))?;
    let digits = bytes
        .get(start..end)
        .ok_or_else(|| malformed("truncated unicode escape"))?;
    let mut value = 0u16;
    for &digit in digits {
        value = (value << 4) | u16::from(hex_nibble(digit)?);
    }
    Ok(value)
}

fn hex_nibble(byte: u8) -> Result<u8, SemanticQualificationError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(malformed("invalid hex digit in unicode escape")),
    }
}

fn decode_char_at(text: &str, index: usize) -> Result<char, SemanticQualificationError> {
    text.get(index..)
        .and_then(|tail| tail.chars().next())
        .ok_or_else(|| malformed("invalid UTF-8 boundary"))
}

fn malformed(message: &str) -> SemanticQualificationError {
    SemanticQualificationError::MalformedJson(message.into())
}

#[cfg(test)]
mod tests {
    use super::normalize_surrogate_pairs;

    #[test]
    fn combines_valid_non_bmp_surrogate_pair() {
        let normalized = normalize_surrogate_pairs(r#"{"candidate_id":"\ud83d\ude00"}"#).unwrap();
        assert_eq!(normalized, "{\"candidate_id\":\"😀\"}");
    }

    #[test]
    fn preserves_literal_escaped_backslash_u_sequence() {
        let input = r#"{"candidate_id":"\\ud83d\\ude00"}"#;
        assert_eq!(normalize_surrogate_pairs(input).unwrap(), input);
    }

    #[test]
    fn rejects_lone_high_surrogate() {
        assert!(normalize_surrogate_pairs(r#"{"candidate_id":"\ud83d"}"#).is_err());
    }

    #[test]
    fn rejects_lone_low_surrogate() {
        assert!(normalize_surrogate_pairs(r#"{"candidate_id":"\ude00"}"#).is_err());
    }

    #[test]
    fn rejects_high_surrogate_followed_by_non_low_surrogate() {
        assert!(normalize_surrogate_pairs(r#"{"candidate_id":"\ud83d\u0041"}"#).is_err());
    }
}
