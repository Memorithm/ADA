//! ADA-A10: offline schema validation for ADA evidence records.
//!
//! The historical hardware evidence protocol (see `scripts/thor_a1_l2.sh`)
//! records SHA-256 bindings between a git commit and measured artifacts. A10
//! keeps that validator intact and also hosts versioned A11 interchange
//! contracts used to bind external mechanistic and mathematical evidence to
//! explicit provenance without upgrading its scientific status.

#![forbid(unsafe_code)]

mod semantic;
mod structured_operator;

pub use semantic::{
    EvidenceWorkloadFingerprint, MAX_EVIDENCE_IDENTIFIER_BYTES, MAX_SEMANTIC_EVIDENCE_BYTES,
    MAX_SUMMARY_METRICS, SEMANTIC_EVIDENCE_HEADER, SEMANTIC_EVIDENCE_VERSION,
    SemanticEvidenceError, SemanticEvidenceRecord, SemanticEvidenceSpec,
};
pub use structured_operator::{
    MAX_OPERATOR_IDENTIFIER_BYTES, OperatorEvidenceClass, OperatorFixtureRef, OperatorSourceRef,
    PreferredDownstreamRoute, STRUCTURED_OPERATOR_IMPORT_VERSION, StructuredOperatorImportError,
    StructuredOperatorImportV1,
};

/// One structured historical hardware evidence record.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceRecord {
    /// Algorithm identifier, e.g. `ADA-A1` or `ADA-A4-ENTMAX`.
    pub algorithm_id: String,
    /// Host fingerprint as produced by the evidence script.
    pub host_fingerprint: String,
    /// UTC timestamp in compact ISO-8601 basic format: `YYYYMMDDTHHMMSSZ`.
    pub timestamp_utc: String,
    /// Toolchain descriptor, e.g. `stable-1.89.0`.
    pub toolchain: String,
    /// Full lowercase 40-hex git commit this evidence is bound to.
    pub git_commit: String,
    /// Lowercase 64-hex SHA-256 of the raw artifact bytes.
    pub sha256_evidence: String,
    /// Numeric metric rows `(name, value)`; values must be finite.
    pub metrics: Vec<(String, f64)>,
}

fn is_lower_hex(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_timestamp(timestamp: &str) -> bool {
    let bytes = timestamp.as_bytes();
    if timestamp.len() != 16 || bytes[8] != b'T' || bytes[15] != b'Z' {
        return false;
    }
    if !timestamp[..8].bytes().all(|b| b.is_ascii_digit())
        || !timestamp[9..15].bytes().all(|b| b.is_ascii_digit())
    {
        return false;
    }

    let digits = |range: std::ops::Range<usize>| -> u32 {
        timestamp[range].parse::<u32>().unwrap_or(u32::MAX)
    };
    let year = digits(0..4);
    let month = digits(4..6);
    let day = digits(6..8);
    let hour = digits(9..11);
    let minute = digits(11..13);
    let second = digits(13..15);

    matches!(month, 1..=12)
        && matches!(day, 1..=31)
        && hour <= 23
        && minute <= 59
        && second <= 59
        && (2020..=2100).contains(&year)
}

impl EvidenceRecord {
    /// Validate every structural and lexical rule; fail closed with a precise
    /// static error naming the first violated field.
    ///
    /// # Errors
    ///
    /// Returns one message per violated contract clause, in field order.
    pub fn validate(&self) -> Result<(), &'static str> {
        let id_ok = self.algorithm_id.starts_with("ADA-")
            && self.algorithm_id.len() <= 32
            && self
                .algorithm_id
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-');
        if !id_ok {
            return Err("algorithm_id must be 'ADA-' followed by A-Z, 0-9 or '-' (max 32 chars)");
        }
        if self.host_fingerprint.trim() != self.host_fingerprint || self.host_fingerprint.is_empty()
        {
            return Err("host_fingerprint must be non-empty without surrounding whitespace");
        }
        if !valid_timestamp(&self.timestamp_utc) {
            return Err("timestamp_utc must be YYYYMMDDTHHMMSSZ with plausible fields");
        }
        if self.toolchain.trim() != self.toolchain || self.toolchain.is_empty() {
            return Err("toolchain must be non-empty without surrounding whitespace");
        }
        if self.git_commit.len() != 40 || !is_lower_hex(&self.git_commit) {
            return Err("git_commit must be exactly 40 lowercase hex characters");
        }
        if self.sha256_evidence.len() != 64 || !is_lower_hex(&self.sha256_evidence) {
            return Err("sha256_evidence must be exactly 64 lowercase hex characters");
        }
        for (name, value) in &self.metrics {
            if name.trim() != name || name.is_empty() {
                return Err("metric names must be non-empty without surrounding whitespace");
            }
            if !value.is_finite() {
                return Err("metric values must be finite");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> EvidenceRecord {
        EvidenceRecord {
            algorithm_id: "ADA-A1".into(),
            host_fingerprint: "thor-l2-6a8779eed181".into(),
            timestamp_utc: "20260824T224152Z".into(),
            toolchain: "stable-1.89.0".into(),
            git_commit: "a278533deadbeef00112233445566778899aabbcc"
                .chars()
                .take(40)
                .collect(),
            sha256_evidence: "3f2c0ed44ee14ff8946b18844e38562fefb02676295dc7bc2344e38fb4c86798"
                .into(),
            metrics: vec![
                ("speedup_ppm_min".into(), 1.084),
                ("speedup_ppm_max".into(), 1.242),
                ("parity_violations".into(), 0.0),
            ],
        }
    }

    #[test]
    fn golden_record_validates_and_matches_real_evidence_shape() {
        assert_eq!(golden().validate(), Ok(()));
    }

    #[test]
    fn each_field_contract_is_enforced() {
        let mut record = golden();
        record.algorithm_id = "ada-a1".into();
        assert!(record.validate().is_err());

        let mut record = golden();
        record.timestamp_utc = "20260824 224152Z".into();
        assert!(record.validate().is_err());

        let mut record = golden();
        record.timestamp_utc = "20261324T224152Z".into();
        assert!(record.validate().is_err());

        let mut record = golden();
        record.git_commit = "ABCDEF".into();
        assert!(record.validate().is_err());

        let mut record = golden();
        record.sha256_evidence = "3f".into();
        assert!(record.validate().is_err());

        let mut record = golden();
        record.metrics.push(("nan_metric".into(), f64::NAN));
        assert_eq!(record.validate(), Err("metric values must be finite"));

        let mut record = golden();
        record.host_fingerprint = " padded ".into();
        assert!(record.validate().is_err());
    }
}
