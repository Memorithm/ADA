//! Strict line-oriented codec for deterministic search checkpoints.

use super::{
    MAX_CHECKPOINT_CANDIDATE_BYTES, MAX_CHECKPOINT_SEEN, MAX_CHECKPOINT_TEXT_BYTES,
    SEARCH_CHECKPOINT_VERSION, SearchBudget, SearchCheckpoint, SearchError, SearchFingerprint,
    SearchStats,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;

const BASE_FIELDS: &[&str] = &[
    "space_primary",
    "space_secondary",
    "space_length",
    "max_expansions",
    "max_candidates",
    "max_program_cost",
    "next_ordinal",
    "generated",
    "statically_rejected",
    "duplicate",
    "oracle_falsified",
    "adversarial_falsified",
    "cost_dominated",
    "surviving",
    "seen_count",
];

impl SearchCheckpoint {
    /// Canonical checkpoint text suitable for an evidence artifact.
    #[must_use]
    pub fn to_canonical_text(&self) -> String {
        let mut text = format!("ADA-SEARCH-CHECKPOINT-V{SEARCH_CHECKPOINT_VERSION}\n");
        append(
            &mut text,
            "space_primary",
            format_args!("{:016x}", self.space_fingerprint.primary()),
        );
        append(
            &mut text,
            "space_secondary",
            format_args!("{:016x}", self.space_fingerprint.secondary()),
        );
        append(
            &mut text,
            "space_length",
            format_args!("{}", self.space_fingerprint.length()),
        );
        append(
            &mut text,
            "max_expansions",
            format_args!("{}", self.budget.max_expansions()),
        );
        append(
            &mut text,
            "max_candidates",
            format_args!("{}", self.budget.max_candidates()),
        );
        append(
            &mut text,
            "max_program_cost",
            format_args!("{}", self.budget.max_program_cost()),
        );
        append(
            &mut text,
            "next_ordinal",
            format_args!("{}", self.next_ordinal),
        );
        append(
            &mut text,
            "generated",
            format_args!("{}", self.stats.generated()),
        );
        append(
            &mut text,
            "statically_rejected",
            format_args!("{}", self.stats.statically_rejected()),
        );
        append(
            &mut text,
            "duplicate",
            format_args!("{}", self.stats.duplicate()),
        );
        append(
            &mut text,
            "oracle_falsified",
            format_args!("{}", self.stats.oracle_falsified()),
        );
        append(
            &mut text,
            "adversarial_falsified",
            format_args!("{}", self.stats.adversarial_falsified()),
        );
        append(
            &mut text,
            "cost_dominated",
            format_args!("{}", self.stats.cost_dominated()),
        );
        append(
            &mut text,
            "surviving",
            format_args!("{}", self.stats.surviving()),
        );
        append(&mut text, "seen_count", format_args!("{}", self.seen.len()));
        for (index, candidate) in self.seen.iter().enumerate() {
            append(
                &mut text,
                &format!("seen_{index}"),
                format_args!("{}", hex_encode(candidate)),
            );
        }
        text
    }

    /// Decode and structurally validate a canonical checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed fields, unsupported versions, exceeded
    /// limits, duplicate keys, or inconsistent counters.
    pub fn from_canonical_text(text: &str) -> Result<Self, SearchError> {
        let fields = parse_fields(text)?;
        let field = |key: &str| {
            fields
                .get(key)
                .copied()
                .ok_or_else(|| SearchError::InvalidCheckpoint(format!("missing field {key}")))
        };
        let seen_count = parse_u64("seen_count", field("seen_count")?)?;
        if seen_count > MAX_CHECKPOINT_SEEN {
            return Err(SearchError::ExceedsLimit {
                field: "seen_count",
                value: seen_count,
                maximum: MAX_CHECKPOINT_SEEN,
            });
        }
        let expected_fields = BASE_FIELDS.len() + usize::try_from(seen_count).unwrap_or(usize::MAX);
        if fields.len() != expected_fields {
            return Err(SearchError::InvalidCheckpoint(
                "field set is incomplete or has unknown keys".into(),
            ));
        }
        for index in 0..seen_count {
            let key = format!("seen_{index}");
            if !fields.contains_key(key.as_str()) {
                return Err(SearchError::InvalidCheckpoint(
                    "seen fields are not contiguous".into(),
                ));
            }
        }

        let space_fingerprint = SearchFingerprint::from_parts(
            parse_hex_u64("space_primary", field("space_primary")?)?,
            parse_hex_u64("space_secondary", field("space_secondary")?)?,
            parse_u64("space_length", field("space_length")?)?,
        );
        let budget = SearchBudget::new(
            parse_u64("max_expansions", field("max_expansions")?)?,
            parse_u64("max_candidates", field("max_candidates")?)?,
            parse_u32("max_program_cost", field("max_program_cost")?)?,
        )?;
        let stats = SearchStats::with_values([
            parse_u64("generated", field("generated")?)?,
            parse_u64("statically_rejected", field("statically_rejected")?)?,
            parse_u64("duplicate", field("duplicate")?)?,
            parse_u64("oracle_falsified", field("oracle_falsified")?)?,
            parse_u64("adversarial_falsified", field("adversarial_falsified")?)?,
            parse_u64("cost_dominated", field("cost_dominated")?)?,
            parse_u64("surviving", field("surviving")?)?,
        ]);
        let mut seen = std::collections::BTreeSet::new();
        for index in 0..seen_count {
            let key = format!("seen_{index}");
            let candidate = hex_decode(field(&key)?)?;
            if candidate.is_empty() || candidate.len() > MAX_CHECKPOINT_CANDIDATE_BYTES {
                return Err(SearchError::InvalidCheckpoint(
                    "seen candidate text is empty or oversized".into(),
                ));
            }
            if !seen.insert(candidate) {
                return Err(SearchError::InvalidCheckpoint(
                    "seen candidate text is duplicated".into(),
                ));
            }
        }
        let checkpoint = Self {
            space_fingerprint,
            budget,
            next_ordinal: parse_u64("next_ordinal", field("next_ordinal")?)?,
            stats,
            seen,
        };
        checkpoint.validate_basic()?;
        Ok(checkpoint)
    }
}

fn parse_fields(text: &str) -> Result<BTreeMap<&str, &str>, SearchError> {
    if text.len() > MAX_CHECKPOINT_TEXT_BYTES || text.contains('\r') {
        return Err(SearchError::InvalidCheckpoint(
            "text exceeds its limit or contains CR".into(),
        ));
    }
    if !text.ends_with('\n') {
        return Err(SearchError::InvalidCheckpoint(
            "text must end with a newline".into(),
        ));
    }
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return Err(SearchError::InvalidCheckpoint(
            "missing version header".into(),
        ));
    };
    let Some(version_text) = header.strip_prefix("ADA-SEARCH-CHECKPOINT-V") else {
        return Err(SearchError::InvalidCheckpoint(
            "invalid version header".into(),
        ));
    };
    let version = parse_u16("version", version_text)?;
    if version != SEARCH_CHECKPOINT_VERSION {
        return Err(SearchError::InvalidCheckpoint(format!(
            "unsupported checkpoint version {version}"
        )));
    }
    let mut fields = BTreeMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Err(SearchError::InvalidCheckpoint(
                "field is missing '='".into(),
            ));
        };
        if key.is_empty() || value.contains('=') || fields.insert(key, value).is_some() {
            return Err(SearchError::InvalidCheckpoint(
                "empty, duplicate, or ambiguous field".into(),
            ));
        }
    }
    if BASE_FIELDS.iter().any(|key| !fields.contains_key(key)) {
        return Err(SearchError::InvalidCheckpoint(
            "base field set is incomplete".into(),
        ));
    }
    Ok(fields)
}

fn append(text: &mut String, key: &str, value: impl std::fmt::Display) {
    let _ = writeln!(text, "{key}={value}");
}

fn parse_u16(field: &str, value: &str) -> Result<u16, SearchError> {
    value
        .parse::<u16>()
        .map_err(|_| SearchError::InvalidCheckpoint(format!("{field} is not u16")))
}

fn parse_u32(field: &str, value: &str) -> Result<u32, SearchError> {
    value
        .parse::<u32>()
        .map_err(|_| SearchError::InvalidCheckpoint(format!("{field} is not u32")))
}

fn parse_u64(field: &str, value: &str) -> Result<u64, SearchError> {
    value
        .parse::<u64>()
        .map_err(|_| SearchError::InvalidCheckpoint(format!("{field} is not u64")))
}

fn parse_hex_u64(field: &str, value: &str) -> Result<u64, SearchError> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(SearchError::InvalidCheckpoint(format!(
            "{field} is not a 16-digit hexadecimal lane"
        )));
    }
    u64::from_str_radix(value, 16)
        .map_err(|_| SearchError::InvalidCheckpoint(format!("{field} has invalid hex")))
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

fn hex_decode(value: &str) -> Result<String, SearchError> {
    if value.len() % 2 != 0 {
        return Err(SearchError::InvalidCheckpoint(
            "hex candidate text has odd length".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut chars = value.bytes();
    while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
        let high = hex_digit(high).ok_or_else(|| {
            SearchError::InvalidCheckpoint("hex candidate text has a non-hex digit".into())
        })?;
        let low = hex_digit(low).ok_or_else(|| {
            SearchError::InvalidCheckpoint("hex candidate text has a non-hex digit".into())
        })?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes)
        .map_err(|_| SearchError::InvalidCheckpoint("hex candidate text is not UTF-8".into()))
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
