#![no_main]
#![forbid(unsafe_code)]

use ada_a2_real_v_trace::parse_value_trace_bytes;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Same fail-closed contract as the Q/K parser: typed errors only, never a
    // panic, never an out-of-bounds read driven by untrusted counts.
    if let Ok(corpus) = parse_value_trace_bytes(data) {
        let metadata = corpus.metadata();
        assert_eq!(metadata.record_count, corpus.records().len());
        for record in corpus.records() {
            if let Some(prefix) = record.value_count.checked_sub(1) {
                let rows = record
                    .prefix_values(prefix)
                    .expect("prefix within count must parse");
                assert_eq!(rows.len(), prefix * record.head_dim);
            }
            // `row()` addresses absolute positions in
            // [value_start_position, value_start_position + value_count).
            let first_out_of_range = u64::try_from(record.value_count)
                .ok()
                .and_then(|count| record.value_start_position.checked_add(count))
                .unwrap_or(u64::MAX);
            assert!(
                record.row(first_out_of_range).is_err(),
                "row() past the stored interval must be a typed error"
            );
        }
    }
});
