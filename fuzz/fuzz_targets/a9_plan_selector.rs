#![no_main]
#![forbid(unsafe_code)]

use ada_a9_plan_selector::{select_plan, PlanSignals};
use libfuzzer_sys::fuzz_target;

fn bounded_f64(bytes: &[u8]) -> f64 {
    let mut value = 0_u64;
    for byte in bytes.iter().take(4) {
        value = (value << 8) | u64::from(*byte);
    }
    let mantissa = (value & ((1_u64 << 40) - 1)) as f64 / (1_u64 << 40) as f64;
    #[allow(clippy::cast_precision_loss)]
    let exponent = (value >> 40) as i32 % 120 - 60;
    mantissa * (2.0_f64).powi(exponent)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }

    // key_count/head_dim/page_size span zero, tiny, and large values.
    let key_count = usize::from(u16::from_le_bytes([data[0], data[1]])) % 6000;
    let head_dim = usize::from(data[2]) % 300;
    let page_size = usize::from(data[3]) % 700;
    let max_abs_logit = if data[4] % 8 == 0 {
        f64::INFINITY
    } else {
        bounded_f64(&data[5..])
    };
    let alpha_raw = bounded_f64(&data[9..]);

    let signals = PlanSignals {
        key_count,
        head_dim,
        page_size,
        max_abs_logit,
        alpha: alpha_raw,
    };

    match select_plan(&signals) {
        Ok((plan, _rationale)) => {
            // Selected plans must respect the documented precedence: extreme
            // magnitude forces Dense, then crossover, then paging geometry.
            let scale = alpha_raw - 1.0;
            let extreme = {
                let max_scaled = scale * max_abs_logit.abs();
                max_scaled.is_finite()
                    && max_scaled > 0.0
                    && {
                        let next = f64::from_bits(max_scaled.to_bits() + 1);
                        next - max_scaled >= 0.5
                    }
            };
            use ada_a9_plan_selector::ExecutionPlan::*;
            let contract_ok = key_count > 0
                && head_dim > 0
                && max_abs_logit.is_finite()
                && (1.0 < alpha_raw && alpha_raw <= 2.0);
            if extreme {
                assert_eq!(plan, Dense);
            } else if contract_ok && key_count < 256 {
                assert_eq!(plan, Dense);
            } else if page_size == 0 {
                assert_eq!(plan, Dense);
            } else if key_count / page_size.max(1) >= 8 {
                assert_eq!(plan, ContentAware);
            } else {
                assert_eq!(plan, Hierarchical);
            }
        }
        Err(_) => {
            // Rejections must correspond to contract violations only.
            let structurally_valid =
                key_count > 0 && head_dim > 0 && max_abs_logit.is_finite() && (1.0 < alpha_raw && alpha_raw <= 2.0);
            assert!(!structurally_valid);
        }
    }
});
