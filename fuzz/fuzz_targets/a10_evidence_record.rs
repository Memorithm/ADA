#![no_main]
#![forbid(unsafe_code)]

use ada_a10_evidence_schema::EvidenceRecord;
use libfuzzer_sys::fuzz_target;

fn hex_like(bytes: &[u8], length: usize) -> String {
    bytes
        .iter()
        .take(length)
        .map(|byte| format!("{:02x}", byte % 16))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    // Build structurally arbitrary records: every field may be malformed.
    let id_source = usize::from(data[0]) % 4;
    let algorithm_id = match id_source {
        0 => "ADA-A1".into(),
        1 => "not-ada".into(),
        2 => format!("ADA-{}", hex_like(&data[1..], 40)),
        _ => String::new(),
    };

    let timestamp = if data[2] % 2 == 0 {
        format!(
            "{}{}T{}{}Z",
            hex_like(&data[3..5], 4),
            hex_like(&data[5..7], 4),
            hex_like(&data[7..9], 6),
            "",
        )
        .replace('x', "0")
    } else {
        "garbage".into()
    };

    let metric_value = i16::from_le_bytes([data[4], data[5]]);
    let record = EvidenceRecord {
        algorithm_id,
        host_fingerprint: hex_like(&data[6..12], 6),
        timestamp_utc: timestamp,
        toolchain: if data[12] % 2 == 0 { "stable-1.89.0".into() } else { String::new() },
        git_commit: hex_like(&data[13..20], 41),
        sha256_evidence: hex_like(&data[20..30], 65),
        metrics: vec![("fuzz_metric".into(), f64::from(metric_value))],
    };

    // Validation must be total: any input maps to Ok or a typed static error,
    // never a panic, regardless of field contents.
    let _ = record.validate();
});
