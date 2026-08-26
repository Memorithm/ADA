#![no_main]
#![forbid(unsafe_code)]

use ada_a3_certified_softmax::budgeted_softmax;
use ada_core::AttentionCase;
use libfuzzer_sys::fuzz_target;

fn f32_from(bytes: &[u8]) -> f32 {
    let mut value = 0_u32;
    for byte in bytes.iter().take(4) {
        value = (value << 8) | u32::from(*byte);
    }
    // Bounded finite magnitudes: exercise the certificate arithmetic, not
    // host-level inf/NaN propagation.
    let mantissa = (value & ((1_u32 << 23) - 1)) as f32 / (1_u32 << 23) as f32;
    #[allow(clippy::cast_precision_loss)]
    let exponent = (value >> 23) as i32 % 30 - 15;
    mantissa * (2.0_f32).powi(exponent)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }

    // Budget from the leading bytes, spanning impossible-to-meetable values.
    let budget_bits = u64::from(data[0]) | (u64::from(data[1]) << 8);
    #[allow(clippy::cast_precision_loss)]
    let epsilon = (budget_bits % 2000) as f64 * 1.0e-12;

    let token_count = usize::from(data[2] % 17) + 1;
    let head_dim = usize::from(data[3] % 5) + 1;

    let needed = token_count * (1 + head_dim) * 4 + 4;
    if data.len() < needed {
        return;
    }

    let mut offset = 4_usize;
    let mut logits = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        logits.push(f32_from(&data[offset..]));
        offset += 4;
    }
    let values = data[offset..offset + token_count * head_dim * 4]
        .chunks_exact(4)
        .map(f32_from)
        .collect();

    let case = AttentionCase {
        logits,
        values,
        head_dim,
    };

    match budgeted_softmax(&case, epsilon) {
        Ok(certified) => {
            // Any certified result must respect its own contract.
            assert!(certified.certified_relative_error_bound <= certified.epsilon);
            assert!(certified.certified_lse_abs_bound <= certified.epsilon);
            assert_eq!(certified.result.metrics.exp_evaluations, token_count);
            for component in &certified.result.output {
                assert!(component.is_finite());
            }
        }
        Err(message) => {
            // Fail-closed paths must be one of the documented errors.
            assert!(
                message == "ADA-A3 certified error bound exceeds the supplied budget"
                    || message.starts_with("ADA-A3 epsilon")
                    || message.contains("ADA-A1"),
                "unexpected error: {message}"
            );
        }
    }
});
