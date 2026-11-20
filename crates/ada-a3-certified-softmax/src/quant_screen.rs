//! Quantized-key screening with certified exact rerank (ADA-A3).
//!
//! Research reference only. At prefill each key row is quantized to a
//! `bits`-bit symmetric code `c_i` with per-row step `Δ_i`, and the exact L2
//! reconstruction error `e_i = ‖k_i − Δ_i c_i‖₂` is stored. At query time the
//! cheap screen `z̃_i = s⟨q, Δ_i c_i⟩` satisfies, by Cauchy–Schwarz,
//! `|z_i − z̃_i| ≤ s‖q‖ e_i = δ_i`, hence `z_i ≤ h_i = z̃_i + δ_i`.
//! Keys are then refined exactly in decreasing `h_i` order. After `j` exact
//! rows (read mass `S`, in log domain `ln S = m + ln Σ e^{z−m}`), the unread
//! mass obeys `T ≤ Σ_{rest} e^{h_i}` (an exact suffix sum, not `count × max`)
//! and truncation with renormalization gives `‖p − q‖₁ ≤ 2T/(S+T)`.
//! The scan stops at the first `j` whose certificate is `≤ ε`.
//! All arithmetic is evaluated in log domain so large scores cannot overflow.

/// Relative inflation applied to every interval half-width.
const DELTA_INFLATION: f64 = 1.0e-9;
/// Relative inflation applied to the unread-mass upper bound.
const MASS_INFLATION: f64 = 1.0e-9;
/// Largest supported head dimension (keeps the f64 dot rounding slack valid).
const MAX_HEAD_DIM: usize = 4096;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QuantScreenCost {
    /// Rows quantized at prefill.
    pub prefill_quantized_rows: usize,
    /// Quantization bit width of the screen.
    pub bits: u32,
    /// Quantized rows dotted with the query (always `n`).
    pub screen_rows_read: usize,
    /// Full-precision rows dotted with the query.
    pub exact_rows_read: usize,
    /// Full-precision rows never read.
    pub exact_rows_skipped: usize,
    /// Elements read at screen precision (`n · d`).
    pub screen_elements_read: usize,
    /// Elements read at full precision (`exact_rows_read · d`).
    pub exact_elements_read: usize,
    /// Comparison-class work: sort of the `n` upper bounds.
    pub sorted_keys: usize,
    /// `exp`/`ln` evaluations for bounds and certificates.
    pub transcendental_evaluations: usize,
}

/// Quantized-screen certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantScreenSoftmax {
    /// Renormalized probabilities in input order; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when all rows were read exactly).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was read at full precision.
    pub read: Vec<bool>,
    /// Positive scale applied to every score.
    pub score_scale: f64,
}

fn validate(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    bits: u32,
) -> Result<usize, &'static str> {
    if head_dim == 0 || head_dim > MAX_HEAD_DIM {
        return Err("ADA-A3 quant-screen head_dim must be in 1..=4096");
    }
    if !(2..=16).contains(&bits) {
        return Err("ADA-A3 quant-screen bits must be in 2..=16");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 quant-screen query must contain head_dim elements");
    }
    if query.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 quant-screen query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 quant-screen requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 quant-screen keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 quant-screen keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 quant-screen score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 quant-screen epsilon must be finite and non-negative");
    }
    Ok(keys.len() / head_dim)
}

/// Symmetric per-row quantization: returns `(step, codes, l2_error)`.
pub(crate) fn quantize_row(row: &[f64], bits: u32) -> (f64, Vec<i32>, f64) {
    let levels = f64::from((1_i32 << (bits - 1)) - 1);
    let max_abs = row.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if max_abs == 0.0 {
        return (0.0, vec![0; row.len()], 0.0);
    }
    let step = max_abs / levels;
    #[allow(clippy::cast_possible_truncation)]
    let codes: Vec<i32> = row.iter().map(|v| (v / step).round() as i32).collect();
    let err = row
        .iter()
        .zip(&codes)
        .map(|(v, &c)| {
            let r = v - step * f64::from(c);
            r * r
        })
        .sum::<f64>()
        .sqrt();
    (step, codes, err)
}

/// Evaluate quantized-screen certified softmax with costs.
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, invalid
/// `bits`, `score_scale` or `epsilon`, or a non-finite score.
#[must_use = "the certified distribution should be checked"]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn certified_quant_screen_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    bits: u32,
) -> Result<(QuantScreenSoftmax, QuantScreenCost), &'static str> {
    let n = validate(query, keys, head_dim, score_scale, epsilon, bits)?;
    let q_norm = query.iter().map(|v| v * v).sum::<f64>().sqrt();
    // Rounding slack of the f64 screen dot, relative to s‖q‖‖k‖ (d ≤ 4096).
    let dot_slack = 1.0e-11;

    let mut hi = Vec::with_capacity(n);
    for row in keys.chunks_exact(head_dim) {
        let (step, codes, err) = quantize_row(row, bits);
        let approx: f64 = query
            .iter()
            .zip(&codes)
            .map(|(a, &c)| a * f64::from(c))
            .sum::<f64>()
            * step
            * score_scale;
        let k_norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let delta = score_scale * q_norm * (err * (1.0 + DELTA_INFLATION) + dot_slack * k_norm);
        let h = approx + delta;
        if !h.is_finite() {
            return Err("ADA-A3 quant-screen scores must be finite");
        }
        hi.push(h);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| hi[b].total_cmp(&hi[a]).then(a.cmp(&b)));
    let h_max = hi[order[0]];
    // suffix[j] = Σ_{r ≥ j} exp(h_{order[r]} − h_max)
    let mut suffix = vec![0.0_f64; n + 1];
    for r in (0..n).rev() {
        suffix[r] = suffix[r + 1] + (hi[order[r]] - h_max).exp();
    }

    let mut cost = QuantScreenCost {
        prefill_quantized_rows: n,
        bits,
        screen_rows_read: n,
        screen_elements_read: n * head_dim,
        sorted_keys: n,
        transcendental_evaluations: 2 * n,
        ..QuantScreenCost::default()
    };
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64;
    let mut certificate = 0.0_f64;
    let mut read_count = n;

    for (rank, &idx) in order.iter().enumerate() {
        let row = &keys[idx * head_dim..(idx + 1) * head_dim];
        let score = score_scale * query.iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
        if !score.is_finite() {
            return Err("ADA-A3 quant-screen scores must be finite");
        }
        cost.exact_rows_read += 1;
        cost.exact_elements_read += head_dim;
        if score > running_max {
            mass = if running_max == f64::NEG_INFINITY {
                1.0
            } else {
                mass * (running_max - score).exp() + 1.0
            };
            running_max = score;
        } else {
            mass += (score - running_max).exp();
        }
        cost.transcendental_evaluations += 1;
        scores.push((idx, score));
        let read_so_far = rank + 1;
        if read_so_far == n {
            certificate = 0.0;
            break;
        }
        // ln S and ln T in the same absolute units.
        let ln_s = running_max + mass.ln();
        let ln_t = h_max + (suffix[read_so_far] * (1.0 + MASS_INFLATION)).ln();
        cost.transcendental_evaluations += 3;
        let cert = 2.0 / (1.0 + (ln_s - ln_t).exp());
        if cert <= epsilon {
            certificate = cert;
            read_count = read_so_far;
            break;
        }
    }
    cost.exact_rows_skipped = n - read_count;

    let total: f64 = scores.iter().map(|&(_, s)| (s - running_max).exp()).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("ADA-A3 quant-screen mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        QuantScreenSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
            score_scale,
        },
        cost,
    ))
}

/// Evaluate quantized-screen certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_quant_screen_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_quant_screen_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    bits: u32,
) -> Result<QuantScreenSoftmax, &'static str> {
    certified_quant_screen_softmax_accounted(query, keys, head_dim, score_scale, epsilon, bits)
        .map(|(result, _)| result)
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]
mod tests {
    use super::{
        certified_quant_screen_softmax, certified_quant_screen_softmax_accounted, quantize_row,
    };
    use crate::{certified_norm_order_softmax_accounted, dense_qk_scores};

    fn oracle(query: &[f64], keys: &[f64], d: usize, scale: f64) -> Vec<f64> {
        let scores = dense_qk_scores(query, keys, d, scale).unwrap();
        let shift = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let terms: Vec<f64> = scores.iter().map(|s| (s - shift).exp()).collect();
        let mass: f64 = terms.iter().sum();
        terms.into_iter().map(|t| t / mass).collect()
    }

    fn l1(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u = self.next().max(1e-300);
            let v = self.next();
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    #[test]
    fn rejects_invalid_input() {
        let k = [1.0, 2.0];
        let ok = certified_quant_screen_softmax(&[1.0, 0.0], &k, 2, 1.0, 0.1, 8);
        assert!(ok.is_ok());
        assert!(certified_quant_screen_softmax(&[1.0, 0.0], &k, 0, 1.0, 0.1, 8).is_err());
        assert!(certified_quant_screen_softmax(&[1.0, 0.0], &k, 5000, 1.0, 0.1, 8).is_err());
        assert!(certified_quant_screen_softmax(&[1.0], &k, 2, 1.0, 0.1, 8).is_err());
        assert!(certified_quant_screen_softmax(&[1.0, 0.0], &[], 2, 1.0, 0.1, 8).is_err());
        assert!(
            certified_quant_screen_softmax(&[1.0, 0.0], &[1.0, 2.0, 3.0], 2, 1.0, 0.1, 8).is_err()
        );
        assert!(certified_quant_screen_softmax(&[f64::NAN, 0.0], &k, 2, 1.0, 0.1, 8).is_err());
        assert!(
            certified_quant_screen_softmax(&[1.0, 0.0], &[f64::INFINITY, 0.0], 2, 1.0, 0.1, 8)
                .is_err()
        );
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(certified_quant_screen_softmax(&[1.0, 0.0], &k, 2, bad, 0.1, 8).is_err());
        }
        for bad in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(certified_quant_screen_softmax(&[1.0, 0.0], &k, 2, 1.0, bad, 8).is_err());
        }
        for bad in [0, 1, 17] {
            assert!(certified_quant_screen_softmax(&[1.0, 0.0], &k, 2, 1.0, 0.1, bad).is_err());
        }
    }

    #[test]
    fn quantization_error_is_exact_and_bounded() {
        let mut rng = Rng(7);
        for bits in 2..=16 {
            for _ in 0..50 {
                let row: Vec<f64> = (0..9).map(|_| rng.gauss() * 3.0).collect();
                let (step, codes, err) = quantize_row(&row, bits);
                let direct = row
                    .iter()
                    .zip(&codes)
                    .map(|(v, &c)| (v - step * f64::from(c)).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((direct - err).abs() <= 1e-12);
                // per-coordinate error ≤ step/2
                assert!(err <= step * 0.5 * 3.0 + 1e-12);
            }
        }
        let (s, c, e) = quantize_row(&[0.0, 0.0], 8);
        assert!(s == 0.0 && c == vec![0, 0] && e == 0.0);
    }

    #[test]
    fn zero_query_and_equal_keys_are_uniform_and_read_all() {
        let keys = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let r = certified_quant_screen_softmax(&[0.0, 0.0], &keys, 2, 1.0, 1e-3, 8).unwrap();
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        assert!(r.certified_l1_bound.abs() < 1e-300);
    }

    #[test]
    fn needle_witness_skips_distractor_exact_rows() {
        // one aligned needle, 63 weak distractors, d = 8
        let d = 8;
        let mut keys = vec![0.0; 64 * d];
        keys[0] = 6.0;
        for i in 1..64 {
            keys[i * d + 1 + (i % 7)] = 1.0;
        }
        let mut q = vec![0.0; d];
        q[0] = 3.0;
        let (r, c) = certified_quant_screen_softmax_accounted(&q, &keys, d, 1.0, 1e-3, 8).unwrap();
        assert_eq!(c.exact_rows_read, 1);
        assert_eq!(c.exact_rows_skipped, 63);
        assert!(r.probabilities[0] > 0.999);
        assert!(l1(&r.probabilities, &oracle(&q, &keys, d, 1.0)) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn adversarial_outlier_rows_never_violate() {
        // a huge outlier coordinate makes the per-row step coarse (large e_i)
        let d = 6;
        let mut rng = Rng(99);
        for bits in [2_u32, 3, 4, 8] {
            for eps in [1e-6, 1e-3, 0.1, 0.5] {
                for _ in 0..20 {
                    let n = 24;
                    let mut keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
                    for i in 0..n {
                        if i % 3 == 0 {
                            keys[i * d] *= 1000.0;
                        }
                    }
                    let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                    let r = certified_quant_screen_softmax(&q, &keys, d, 0.05, eps, bits).unwrap();
                    let o = oracle(&q, &keys, d, 0.05);
                    assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
                    assert!(
                        r.certified_l1_bound <= eps.max(0.0) + 1e-15 || r.certified_l1_bound == 0.0
                    );
                    assert!((r.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn overflow_scale_stays_finite() {
        let d = 2;
        let keys = [1e150, 1e150, -1e150, 1e150, 1.0, 1.0];
        let q = [1e150, 1e150];
        if let Ok(r) = certified_quant_screen_softmax(&q, &keys, d, 1.0, 0.1, 8) {
            assert!(r.probabilities.iter().all(|p| p.is_finite()));
        }
        let q2 = [700.0, 0.0];
        let keys2 = [1.0, 0.0, 0.9, 0.0, -1.0, 0.0];
        let r = certified_quant_screen_softmax(&q2, &keys2, d, 1.0, 0.1, 8).unwrap();
        assert!(r.probabilities.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn exhaustive_grid_has_no_violation() {
        let alphabet: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.3, -2.0], [-1.5, 1.0]];
        let queries: [[f64; 2]; 3] = [[1.0, 0.5], [3.0, -1.0], [0.0, 0.0]];
        let mut cases = 0;
        let mut skips = 0;
        let max_len = if cfg!(miri) { 2_usize } else { 4 };
        for len in 1..=max_len {
            for code in 0..4_usize.pow(len as u32) {
                let mut keys = Vec::new();
                let mut c = code;
                for _ in 0..len {
                    keys.extend_from_slice(&alphabet[c % 4]);
                    c /= 4;
                }
                for q in &queries {
                    for eps in [1e-3, 5e-2, 0.5] {
                        for bits in if cfg!(miri) {
                            vec![4_u32]
                        } else {
                            vec![2_u32, 4]
                        } {
                            let (r, cost) = certified_quant_screen_softmax_accounted(
                                q, &keys, 2, 1.0, eps, bits,
                            )
                            .unwrap();
                            let o = oracle(q, &keys, 2, 1.0);
                            assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
                            cases += 1;
                            if cost.exact_rows_skipped > 0 {
                                skips += 1;
                            }
                        }
                    }
                }
            }
        }
        println!("quant-screen grid cases={cases} skipping={skips}");
        assert!(cases > 0);
    }

    #[test]
    fn random_sweep_and_cost_study() {
        let (d, n, samples) = (32_usize, 128_usize, if cfg!(miri) { 1 } else { 30 });
        let mut rng = Rng(2026);
        for &logit_std in &[1.0_f64, 3.0, 6.0] {
            for &bits in &[4_u32, 6, 8] {
                for &eps in &[1e-2_f64, 1e-1] {
                    let mut frac = 0.0;
                    let mut norm_frac = 0.0;
                    let mut max_l1 = 0.0_f64;
                    let mut max_cert = 0.0_f64;
                    for _ in 0..samples {
                        let keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
                        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                        let scale = logit_std / (d as f64).sqrt();
                        let (r, c) = certified_quant_screen_softmax_accounted(
                            &q, &keys, d, scale, eps, bits,
                        )
                        .unwrap();
                        let o = oracle(&q, &keys, d, scale);
                        let l = l1(&r.probabilities, &o);
                        assert!(l <= r.certified_l1_bound + 1e-12);
                        max_l1 = max_l1.max(l);
                        max_cert = max_cert.max(r.certified_l1_bound);
                        frac += c.exact_rows_read as f64 / n as f64;
                        norm_frac +=
                            certified_norm_order_softmax_accounted(&q, &keys, d, scale, eps)
                                .map_or(1.0, |(_, nc)| {
                                    nc.query_time_key_rows_read as f64 / n as f64
                                });
                    }
                    let s = samples as f64;
                    println!(
                        "quant-screen logit_std={logit_std} bits={bits} eps={eps}: exact_rows_frac={:.4} net_vs_fp16={:.4} norm_order={:.4} max_l1={max_l1:.3e} max_cert={max_cert:.3e}",
                        frac / s,
                        f64::from(bits) / 16.0 + frac / s,
                        norm_frac / s
                    );
                }
            }
        }
    }
}
