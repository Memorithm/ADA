//! `a10-validate`: offline validator for ADA evidence metadata sidecars.
//!
//! A sidecar is a UTF-8 file of `key=value` lines (UTF-8, no quoting):
//!
//! ```text
//! algorithm_id=ADA-A1
//! host_fingerprint=thor-l2-912f37dca2e6
//! timestamp_utc=20260825T180839Z
//! toolchain=stable-1.89.0
//! git_commit=912f37dca2e6...
//! sha256_evidence=a4077126...
//! metric.speedup_ppm_min=1.103
//! ```
//!
//! Required keys: the six record fields. Lines prefixed `metric.` become
//! named finite metrics. Unknown keys are rejected (fail closed), so a
//! typo can never silently drop a binding. Exit 0 on valid, 1 on invalid,
//! 2 on usage/IO errors.

#![forbid(unsafe_code)]

use ada_a10_evidence_schema::EvidenceRecord;
use std::collections::BTreeMap;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: a10-validate <evidence.meta>");
        return ExitCode::from(2);
    };

    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) => {
            eprintln!("error: cannot read {path}: {error}");
            return ExitCode::from(2);
        }
    };

    match parse_sidecar(&raw) {
        Ok(record) => match record.validate() {
            Ok(()) => {
                println!("VALID {path}");
                println!(
                    "algorithm={} commit={} digest={}",
                    record.algorithm_id,
                    &record.git_commit[..12.min(record.git_commit.len())],
                    &record.sha256_evidence[..16.min(record.sha256_evidence.len())]
                );
                println!("metrics={}", record.metrics.len());
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("INVALID {path}: {reason}");
                ExitCode::FAILURE
            }
        },
        Err(reason) => {
            eprintln!("INVALID {path}: {reason}");
            ExitCode::FAILURE
        }
    }
}

fn parse_sidecar(raw: &str) -> Result<EvidenceRecord, String> {
    let mut fields: BTreeMap<&str, String> = BTreeMap::new();
    let mut metrics: Vec<(String, f64)> = Vec::new();

    for (line_number, line) in raw.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {}: expected key=value", line_number + 1));
        };
        if value.is_empty() {
            return Err(format!("line {}: empty value", line_number + 1));
        }

        if let Some(metric_name) = key.strip_prefix("metric.") {
            let parsed: f64 = value
                .parse()
                .map_err(|_| format!("line {}: metric value is not an f64", line_number + 1))?;
            metrics.push((metric_name.to_owned(), parsed));
        } else if fields.insert(key, value.to_owned()).is_some() {
            return Err(format!("line {}: duplicate key {key}", line_number + 1));
        } else if !matches!(
            key,
            "algorithm_id"
                | "host_fingerprint"
                | "timestamp_utc"
                | "toolchain"
                | "git_commit"
                | "sha256_evidence"
        ) {
            return Err(format!("line {}: unknown key {key}", line_number + 1));
        }
    }

    let missing = |name: &str| format!("missing required key {name}");
    let record = EvidenceRecord {
        algorithm_id: fields
            .remove("algorithm_id")
            .ok_or_else(|| missing("algorithm_id"))?,
        host_fingerprint: fields
            .remove("host_fingerprint")
            .ok_or_else(|| missing("host_fingerprint"))?,
        timestamp_utc: fields
            .remove("timestamp_utc")
            .ok_or_else(|| missing("timestamp_utc"))?,
        toolchain: fields
            .remove("toolchain")
            .ok_or_else(|| missing("toolchain"))?,
        git_commit: fields
            .remove("git_commit")
            .ok_or_else(|| missing("git_commit"))?,
        sha256_evidence: fields
            .remove("sha256_evidence")
            .ok_or_else(|| missing("sha256_evidence"))?,
        metrics,
    };

    if !fields.is_empty() {
        return Err("unrecognized keys remain after parsing".to_string());
    }

    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> String {
        [
            "algorithm_id=ADA-A1",
            "host_fingerprint=thor-l2-912f37dca2e6",
            "timestamp_utc=20260825T180839Z",
            "toolchain=stable-1.89.0",
            "git_commit=912f37dca2e600112233445566778899aabbccdd",
            "sha256_evidence=a40771263d93690e3627c1d279e5dbfcb243fb42fcff4e70400ee1522eefd579",
            "metric.speedup_ppm_min=1.103",
        ]
        .join("\n")
    }

    #[test]
    fn golden_sidecar_parses_and_validates() {
        let record = parse_sidecar(&golden()).expect("golden sidecar must parse");
        assert_eq!(record.validate(), Ok(()));
        assert_eq!(record.metrics.len(), 1);
    }

    #[test]
    fn sidecar_failures_are_typed() {
        // Unknown key is rejected, not ignored.
        let hostile = format!("{}\nsneaky_key=1", golden());
        assert!(
            parse_sidecar(&hostile)
                .err()
                .is_some_and(|message| message.contains("unknown"))
        );

        // Duplicate key rejected.
        let duplicated = format!("{}\nalgorithm_id=ADA-B2", golden());
        assert!(
            parse_sidecar(&duplicated)
                .err()
                .is_some_and(|m| m.contains("duplicate"))
        );

        // Missing required key rejected.
        let missing: String = golden()
            .lines()
            .filter(|line| !line.starts_with("toolchain="))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            parse_sidecar(&missing)
                .err()
                .is_some_and(|m| m.contains("missing"))
        );

        // Non-numeric metric value rejected.
        let bad_metric = format!("{}\nmetric.bad=fast", golden());
        assert!(
            parse_sidecar(&bad_metric)
                .err()
                .is_some_and(|m| m.contains("f64"))
        );

        // A structurally parsed record can still fail schema validation.
        let wrong_timestamp = golden().replace("20260825", "20261325");
        let record = parse_sidecar(&wrong_timestamp).unwrap();
        assert!(record.validate().is_err());
    }
}
