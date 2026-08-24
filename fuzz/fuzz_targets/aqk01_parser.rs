#![no_main]
#![forbid(unsafe_code)]

use ada_a5_real_qk_trace::parse_trace_bytes;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The parser must classify arbitrary bytes into either a well-formed
    // corpus or a typed error; it must never panic, allocate unboundedly from
    // header fields, or read past the buffer.
    if let Ok(corpus) = parse_trace_bytes(data) {
        let metadata = corpus.metadata();
        assert_eq!(metadata.record_count, corpus.len());
        for record in corpus.records() {
            // Parser-produced records must satisfy the replay contract for
            // every admissible page size and alpha in the candidate domain.
            let page_size = (record.head_dim).max(1);
            let case = record
                .to_query_key_case(page_size, 1.5)
                .expect("parser-produced record must convert");
            case.validate()
                .expect("parser-produced record must validate");
        }
    }
});
