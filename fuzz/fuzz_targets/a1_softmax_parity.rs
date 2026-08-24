#![no_main]
#![forbid(unsafe_code)]

use ada_core::{AttentionCase, LogicalMetrics};
use ada_oracle::{online_softmax_baseline, online_softmax_one_exp};
use libfuzzer_sys::fuzz_target;

fn f32_pairs(data: &[u8]) -> impl Iterator<Item = f32> + '_ {
    data.chunks_exact(4).map(|chunk| {
        let bits = u32::from_le_bytes(chunk.try_into().unwrap());
        f32::from_bits(bits & 0x4fff_ffff) // strip sign of exponent extremes
    })
}

fn assert_close(left: f32, right: f32, tolerance: f32) {
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= tolerance * scale,
        "outputs diverge beyond tolerance: {left} vs {right}"
    );
}

fuzz_target!(|data: &[u8]| {
    let mut values_stream = f32_pairs(data);
    let head_dim = 4_usize;

    while let Some(first) = values_stream.next() {
        let logits: Vec<f32> = std::iter::once(first)
            .chain(values_stream.by_ref().take(15))
            .collect();
        let values: Vec<f32> = values_stream.by_ref().take(logits.len() * head_dim).collect();
        if values.len() < logits.len() * head_dim {
            break;
        }

        let case = AttentionCase {
            logits,
            values,
            head_dim,
        };

        match case.validate() {
            Err(_) => continue,
            Ok(()) => {}
        }

        let baseline = online_softmax_baseline(&case).expect("validated case");
        let candidate = online_softmax_one_exp(&case).expect("validated case");

        // ADA-A1 acceptance: exact recurrence parity within declared f32
        // tolerance, finite outputs, and the logical exp-count reduction.
        assert_eq!(baseline.output.len(), candidate.output.len());
        for (&expected, &actual) in baseline.output.iter().zip(&candidate.output) {
            assert_close(expected, actual, 2.0e-5);
            assert!(actual.is_finite());
        }
        assert_close(baseline.lse, candidate.lse, 2.0e-5);
        assert!(candidate.lse.is_finite());

        let n = case.logits.len();
        assert_eq!(
            baseline.metrics.exp_evaluations,
            2 * n - 1,
            "baseline must evaluate 2n-1 exponentials"
        );
        assert_eq!(
            candidate.metrics.exp_evaluations,
            n - 1,
            "candidate must evaluate n-1 exponentials"
        );
        assert_eq!(baseline.metrics.log_evaluations, 1);
        assert_eq!(candidate.metrics.log_evaluations, 1);

        let _ = LogicalMetrics::default();
    }
});
