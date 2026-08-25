//! Explicit NEON kernels for the ADA-A1 online-softmax laboratory.
//!
//! The rest of the ADA workspace keeps a workspace-wide
//! `unsafe_code = "forbid"` contract. This crate is the single, deliberate
//! exception: explicit `AArch64` SIMD requires vendor intrinsics that are
//! `unsafe` in Rust. The unsafe surface is confined to this module, gated
//! behind `target_arch = "aarch64"`, and every block carries a `// SAFETY:`
//! justification. On every other target the public API compiles and fails
//! closed with a static error, so cross-compilation stays green.

#![cfg_attr(not(target_arch = "aarch64"), forbid(unsafe_code))]
#![cfg_attr(target_arch = "aarch64", allow(unsafe_code))]

#[cfg(target_arch = "aarch64")]
use ada_core::LogicalMetrics;
use ada_core::{AttentionCase, AttentionResult};

/// Error returned on hosts without the required architecture support.
/// Only present on targets where the fail-closed stubs exist.
#[cfg(not(target_arch = "aarch64"))]
const UNSUPPORTED_TARGET: &str =
    "ADA-A1 NEON kernels require an aarch64 host (compile for target_arch = aarch64)";

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon {
    use std::arch::aarch64::{
        vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vmulq_n_f32, vst1q_f32,
    };

    /// Vector lane width of the kernels below.
    const LANES: usize = 4;

    /// `acc <- alpha * acc + p * v` over full 4-lane chunks plus a scalar tail.
    ///
    /// # Safety
    ///
    /// `acc` and `v` must each contain at least `len` initialized `f32`
    /// values; `acc` and `v` must not alias. The NEON loads read exactly the
    /// first `len` elements in-bounds.
    pub(crate) unsafe fn fused_scale_accumulate(acc: &mut [f32], v: &[f32], alpha: f32, p: f32) {
        let len = acc.len();
        let mut chunks = len / LANES * LANES;

        // SAFETY: vdupq is a pure constant broadcast with no memory access.
        let alpha_vec = unsafe { vdupq_n_f32(alpha) };

        let mut offset = 0;
        while offset < chunks {
            // SAFETY: offset + 4 <= chunks <= len, so both 16-byte reads stay
            // inside their respective slices; slices are disjoint by caller
            // contract; f32 alignment (4) satisfies vld1q alignment rules.
            let (acc_vec, v_vec) = unsafe {
                (
                    vld1q_f32(acc.as_ptr().add(offset)),
                    vld1q_f32(v.as_ptr().add(offset)),
                )
            };
            let mixed = unsafe { vfmaq_f32(vmulq_n_f32(v_vec, p), acc_vec, alpha_vec) };
            // SAFETY: same bounds as the load above; exclusive mutable borrow
            // guarantees no concurrent writes.
            unsafe { vst1q_f32(acc.as_mut_ptr().add(offset), mixed) };
            offset += LANES;
        }

        while chunks < len {
            acc[chunks] = alpha * acc[chunks] + p * v[chunks];
            chunks += 1;
        }
    }

    /// `acc <- alpha * acc` elementwise, same chunking contract as above.
    ///
    /// # Safety
    ///
    /// `acc` must contain at least `len` initialized `f32` values.
    pub(crate) unsafe fn scale_in_place(acc: &mut [f32], alpha: f32) {
        let len = acc.len();
        let mut chunks = len / LANES * LANES;

        // SAFETY: pure broadcast, no memory access.
        let alpha_vec = unsafe { vdupq_n_f32(alpha) };
        let mut offset = 0;
        while offset < chunks {
            // SAFETY: offset + 4 <= len; exclusive mutable borrow.
            let acc_vec = unsafe { vld1q_f32(acc.as_ptr().add(offset)) };
            let scaled = unsafe { vmulq_f32(acc_vec, alpha_vec) };
            unsafe { vst1q_f32(acc.as_mut_ptr().add(offset), scaled) };
            offset += LANES;
        }

        while chunks < len {
            acc[chunks] *= alpha;
            chunks += 1;
        }
    }
}

/// Baseline streaming online Softmax with the numerator update vectorized on
/// NEON. Transcendentals remain scalar, so logical metrics are identical to
/// the scalar baseline and only the accumulate bandwidth changes.
///
/// # Errors
///
/// Returns an error when the case violates the ADA-A1 input contract, or when
/// running on a non-aarch64 host.
pub fn online_softmax_baseline_neon(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = case;
        Err(UNSUPPORTED_TARGET)
    }

    #[cfg(target_arch = "aarch64")]
    {
        case.validate()?;

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        let mut numerator = vec![0.0_f32; case.head_dim];
        let mut metrics = LogicalMetrics::default();

        for (key, &score) in case.logits.iter().enumerate() {
            metrics.qk_pairs_evaluated += 1;
            let new_max = running_max.max(score);
            let alpha = if running_max.is_finite() {
                metrics.exp_evaluations += 1;
                (running_max - new_max).exp()
            } else {
                0.0
            };
            metrics.exp_evaluations += 1;
            let probability_numerator = (score - new_max).exp();
            running_sum = alpha * running_sum + probability_numerator;

            let value = &case.values[key * case.head_dim..(key + 1) * case.head_dim];
            // SAFETY: numerator and value are disjoint slices of length
            // head_dim; all elements are initialized.
            unsafe {
                neon::fused_scale_accumulate(&mut numerator, value, alpha, probability_numerator);
            }
            metrics.value_accumulate_elements += case.head_dim;
            running_max = new_max;
        }

        let inv_sum = running_sum.recip();
        // SAFETY: numerator is fully initialized.
        unsafe { neon::scale_in_place(&mut numerator, inv_sum) };
        metrics.log_evaluations += 1;
        let lse = running_max + running_sum.ln();

        Ok(AttentionResult {
            output: numerator,
            lse,
            metrics,
        })
    }
}

/// One-exp candidate recurrence with the numerator update vectorized on NEON.
/// Branch structure and transcendental count match the scalar candidate.
///
/// # Errors
///
/// Returns an error when the case violates the ADA-A1 input contract, or when
/// running on a non-aarch64 host.
pub fn online_softmax_one_exp_neon(case: &AttentionCase) -> Result<AttentionResult, &'static str> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = case;
        Err(UNSUPPORTED_TARGET)
    }

    #[cfg(target_arch = "aarch64")]
    {
        case.validate()?;

        let first_score = case.logits[0];
        let mut running_max = first_score;
        let mut running_sum = 1.0_f32;
        let mut numerator = case.values[..case.head_dim].to_vec();
        let mut metrics = LogicalMetrics {
            qk_pairs_evaluated: 1,
            exp_evaluations: 0,
            log_evaluations: 0,
            value_accumulate_elements: case.head_dim,
        };

        for (key, &score) in case.logits.iter().enumerate().skip(1) {
            metrics.qk_pairs_evaluated += 1;
            let value = &case.values[key * case.head_dim..(key + 1) * case.head_dim];

            if score <= running_max {
                metrics.exp_evaluations += 1;
                let p = (score - running_max).exp();
                running_sum += p;
                // SAFETY: numerator and value are disjoint, fully initialized,
                // and exactly head_dim long.
                unsafe {
                    neon::fused_scale_accumulate(&mut numerator, value, 1.0, p);
                }
                metrics.value_accumulate_elements += case.head_dim;
            } else {
                metrics.exp_evaluations += 1;
                let alpha = (running_max - score).exp();
                running_sum = alpha * running_sum + 1.0;
                // SAFETY: see above; rescale then add raw value row.
                unsafe {
                    neon::scale_in_place(&mut numerator, alpha);
                    neon::fused_scale_accumulate(&mut numerator, value, 1.0, 1.0);
                }
                metrics.value_accumulate_elements += case.head_dim;
                running_max = score;
            }
        }

        let inv_sum = running_sum.recip();
        // SAFETY: numerator is fully initialized.
        unsafe { neon::scale_in_place(&mut numerator, inv_sum) };
        metrics.log_evaluations += 1;
        let lse = running_max + running_sum.ln();

        Ok(AttentionResult {
            output: numerator,
            lse,
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(logits: &[f32], head_dim: usize) -> AttentionCase {
        let mut values = Vec::with_capacity(logits.len() * head_dim);
        for key in 0..logits.len() {
            for lane in 0..head_dim {
                let linear_index = key * head_dim + lane;
                #[allow(clippy::cast_precision_loss)]
                {
                    values.push(linear_index as f32 * 0.031_25 - 0.5);
                }
            }
        }
        AttentionCase {
            logits: logits.to_vec(),
            values,
            head_dim,
        }
    }

    #[test]
    fn unsupported_targets_fail_closed_without_touching_the_case() {
        // On aarch64 this test asserts parity instead (see below); on other
        // targets both kernels must reject deterministically.
        #[cfg(not(target_arch = "aarch64"))]
        {
            let c = case(&[1.0], 2);
            assert_eq!(
                online_softmax_baseline_neon(&c).unwrap_err(),
                UNSUPPORTED_TARGET
            );
            assert_eq!(
                online_softmax_one_exp_neon(&c).unwrap_err(),
                UNSUPPORTED_TARGET
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_parity_with_odd_head_dims_and_logits() {
        let ada_oracle_case = |logits: &[f32], head_dim| case(logits, head_dim);

        for logits in [
            vec![-8.0, -4.0, -1.0, 0.0, 3.0, 9.0],
            vec![9.0, 3.0, 0.0, -1.0, -4.0, -8.0],
            vec![2.0; 9],
            vec![7.0],
        ] {
            for head_dim in [1, 3, 4, 5, 8] {
                let c = ada_oracle_case(&logits, head_dim);

                let reference_baseline = crate_scalar_baseline(&c);
                let kernel_baseline = online_softmax_baseline_neon(&c).unwrap();
                assert_close_slice(&reference_baseline.output, &kernel_baseline.output);
                assert_eq!(
                    kernel_baseline.metrics.exp_evaluations,
                    2 * c.logits.len() - 1
                );

                let reference_one_exp = crate_scalar_one_exp(&c);
                let kernel_one_exp = online_softmax_one_exp_neon(&c).unwrap();
                assert_close_slice(&reference_one_exp.output, &kernel_one_exp.output);
                assert_eq!(kernel_one_exp.metrics.exp_evaluations, c.logits.len() - 1);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn crate_scalar_baseline(c: &AttentionCase) -> ada_core::AttentionResult {
        let mut m = f32::NEG_INFINITY;
        let mut s = 0.0_f32;
        let mut out = vec![0.0_f32; c.head_dim];
        for (key, &score) in c.logits.iter().enumerate() {
            let new_max = m.max(score);
            let alpha = if m.is_finite() {
                (m - new_max).exp()
            } else {
                0.0
            };
            let p = (score - new_max).exp();
            s = alpha * s + p;
            let value = &c.values[key * c.head_dim..(key + 1) * c.head_dim];
            for (acc, &v) in out.iter_mut().zip(value) {
                *acc = alpha * *acc + p * v;
            }
            m = new_max;
        }
        let inv = s.recip();
        for x in &mut out {
            *x *= inv;
        }
        ada_core::AttentionResult {
            output: out,
            lse: m + s.ln(),
            metrics: ada_core::LogicalMetrics::default(),
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn crate_scalar_one_exp(c: &AttentionCase) -> ada_core::AttentionResult {
        let mut m = c.logits[0];
        let mut s = 1.0_f32;
        let mut out = c.values[..c.head_dim].to_vec();
        for (key, &score) in c.logits.iter().enumerate().skip(1) {
            let value = &c.values[key * c.head_dim..(key + 1) * c.head_dim];
            if score <= m {
                let p = (score - m).exp();
                s += p;
                for (acc, &v) in out.iter_mut().zip(value) {
                    *acc += p * v;
                }
            } else {
                let alpha = (m - score).exp();
                s = alpha * s + 1.0;
                for (acc, &v) in out.iter_mut().zip(value) {
                    *acc = alpha * *acc + v;
                }
                m = score;
            }
        }
        let inv = s.recip();
        for x in &mut out {
            *x *= inv;
        }
        ada_core::AttentionResult {
            output: out,
            lse: m + s.ln(),
            metrics: ada_core::LogicalMetrics::default(),
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn assert_close_slice(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right) {
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= 1.0e-6 * scale, "{a} != {b}");
        }
    }
}
