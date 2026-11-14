//! Norm-ordered Cauchy–Schwarz early-termination certified softmax (ADA-A3).
//!
//! Research reference only. Keys are visited in non-increasing `‖k‖` order
//! (an A4-style prefill index). Every unread key `i` satisfies
//! `z_i = s⟨q, k_i⟩ ≤ s‖q‖‖k_i‖ ≤ B_j` where `B_j` is the bound of the
//! largest unread norm, so the unread mass relative to the running maximum
//! `m` of the exact scores read so far obeys `T ≤ (n − j) exp(B_j − m)`.
//! With read mass `S ≥ 1` (in units of `exp(m)`), truncating and
//! renormalizing gives `‖p − q‖₁ = 2 t ≤ 2 T_hi / (S + T_hi)`.
//! The scan stops at the first `j` whose certificate fits the budget, so
//! the tail keys are never dotted with the query. Position-independent:
//! unlike contiguous blocks, the skipped set is chosen by norm rank.

/// Relative inflation applied to the Cauchy–Schwarz bound for rounding.
const BOUND_INFLATION: f64 = 1.0e-12;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormOrderCost {
    /// Key norms computed at prefill (one pass over all keys).
    pub prefill_norm_evaluations: usize,
    /// Comparisons-class work at prefill: `n` keys sorted by norm.
    pub prefill_sorted_keys: usize,
    /// Key rows dotted with the query at inference time.
    pub query_time_key_rows_read: usize,
    /// Key rows that were not read.
    pub key_rows_skipped: usize,
    /// `exp` evaluations (scores read plus certificate evaluations).
    pub exp_evaluations: usize,
    /// Certificate evaluations along the scan.
    pub bound_checks: usize,
}

/// Norm-ordered certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct NormOrderSoftmax {
    /// Renormalized probabilities in input order; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate `2 T_hi / (S + T_hi)` (0 when all read).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was read, in input order.
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
) -> Result<usize, &'static str> {
    if head_dim == 0 {
        return Err("ADA-A3 norm-order softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 norm-order softmax query must contain head_dim elements");
    }
    if query.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 norm-order softmax query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 norm-order softmax requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 norm-order softmax keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 norm-order softmax keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 norm-order softmax score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 norm-order softmax epsilon must be finite and non-negative");
    }
    Ok(keys.len() / head_dim)
}

fn norm(row: &[f64]) -> f64 {
    row.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Evaluate norm-ordered early-termination certified softmax with costs.
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, a
/// non-positive `score_scale`, a negative or non-finite `epsilon`, or a
/// non-finite score or mass.
#[must_use = "the certified distribution should be checked"]
pub fn certified_norm_order_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<(NormOrderSoftmax, NormOrderCost), &'static str> {
    let n = validate(query, keys, head_dim, score_scale, epsilon)?;
    let q_norm = norm(query);
    let norms: Vec<f64> = keys.chunks_exact(head_dim).map(norm).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]).then(a.cmp(&b)));
    let bound_of = |idx: usize| score_scale * q_norm * norms[idx] * (1.0 + BOUND_INFLATION);

    let mut cost = NormOrderCost {
        prefill_norm_evaluations: n,
        prefill_sorted_keys: n,
        ..NormOrderCost::default()
    };
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64; // sum exp(z - running_max) over read keys
    let mut certificate = 0.0_f64;
    let mut read_count = n;

    for (rank, &idx) in order.iter().enumerate() {
        let row = &keys[idx * head_dim..(idx + 1) * head_dim];
        let score = score_scale * query.iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
        if !score.is_finite() {
            return Err("ADA-A3 norm-order softmax scores must be finite");
        }
        cost.query_time_key_rows_read += 1;
        if score > running_max {
            mass = mass * (running_max - score).exp() + 1.0;
            if running_max == f64::NEG_INFINITY {
                mass = 1.0;
            }
            running_max = score;
        } else {
            mass += (score - running_max).exp();
        }
        cost.exp_evaluations += 1;
        scores.push((idx, score));

        let read_so_far = rank + 1;
        if read_so_far == n {
            certificate = 0.0;
            break;
        }
        let remaining = n - read_so_far;
        #[allow(clippy::cast_precision_loss)]
        let tail_hi = remaining as f64 * (bound_of(order[read_so_far]) - running_max).exp();
        cost.exp_evaluations += 1;
        cost.bound_checks += 1;
        let cert = if tail_hi.is_finite() {
            2.0 * tail_hi / (mass + tail_hi)
        } else {
            2.0
        };
        if cert <= epsilon {
            certificate = cert;
            read_count = read_so_far;
            break;
        }
    }
    cost.key_rows_skipped = n - read_count;

    let total: f64 = scores.iter().map(|&(_, s)| (s - running_max).exp()).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("ADA-A3 norm-order softmax mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        NormOrderSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
            score_scale,
        },
        cost,
    ))
}

/// Evaluate norm-ordered early-termination certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_norm_order_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_norm_order_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<NormOrderSoftmax, &'static str> {
    certified_norm_order_softmax_accounted(query, keys, head_dim, score_scale, epsilon)
        .map(|(result, _)| result)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::{certified_norm_order_softmax, certified_norm_order_softmax_accounted};
    use crate::{certified_qk_box_two_stat_softmax_accounted, dense_qk_scores};

    /// Independent oracle: dense stable softmax from scalar dots.
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
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[allow(clippy::cast_precision_loss)]
            {
                ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            }
        }
        fn gauss(&mut self) -> f64 {
            let (a, b) = (self.unit(), self.unit());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    fn check(query: &[f64], keys: &[f64], d: usize, scale: f64, eps: f64) -> usize {
        let r = certified_norm_order_softmax(query, keys, d, scale, eps).unwrap();
        let o = oracle(query, keys, d, scale);
        let observed = l1(&r.probabilities, &o);
        assert!(
            observed <= r.certified_l1_bound + 1e-12,
            "observed {observed} > bound {}",
            r.certified_l1_bound
        );
        assert!(r.certified_l1_bound <= eps.max(0.0) || r.read.iter().all(|&x| x));
        r.read.iter().filter(|&&x| x).count()
    }

    #[test]
    fn fail_closed_on_invalid_inputs() {
        let f = certified_norm_order_softmax;
        assert!(f(&[1.0], &[], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 0, 1.0, 0.1).is_err());
        assert!(f(&[1.0, 2.0], &[1.0], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0, 2.0, 3.0], 2, 1.0, 0.1).is_err());
        assert!(f(&[f64::NAN], &[1.0], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[f64::INFINITY], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 0.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, f64::NAN, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, -1.0).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn singleton_and_equal_keys_read_everything_or_stay_exact() {
        let r = certified_norm_order_softmax(&[1.0, 0.0], &[0.5, 0.5], 2, 1.0, 0.0).unwrap();
        assert_eq!(r.probabilities, vec![1.0]);
        let keys = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let r = certified_norm_order_softmax(&[1.0, 0.0], &keys, 2, 1.0, 1e-6).unwrap();
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn zero_query_cannot_skip_and_is_uniform() {
        let keys = [1.0, 2.0, -3.0, 4.0, 5.0, 6.0];
        let r = certified_norm_order_softmax(&[0.0, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.read.iter().all(|&x| x));
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn overflowing_bound_is_handled_without_nan() {
        let keys = [1.0e3, 0.0, 1.0e-3, 0.0, 5.0e2, 0.0];
        let r = certified_norm_order_softmax(&[1.0e3, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.probabilities.iter().all(|p| p.is_finite()));
        assert!(r.certified_l1_bound.is_finite());
    }

    #[test]
    fn designed_witness_big_norm_needle_among_small_norm_distractors() {
        // Needle norm 20 aligned with q; 63 distractors of norm 1.
        let d = 4;
        let mut keys = vec![0.0; 64 * d];
        keys[0] = 20.0;
        for i in 1..64 {
            keys[i * d + 1] = 1.0;
        }
        let q = [1.0, 0.5, 0.0, 0.0];
        let (r, c) = certified_norm_order_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        assert_eq!(c.query_time_key_rows_read, 1);
        assert_eq!(c.key_rows_skipped, 63);
        assert!(r.probabilities[0] > 0.999);
        let o = oracle(&q, &keys, d, 1.0);
        assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn exhaustive_small_grid_has_zero_certificate_violations() {
        let alphabet: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]];
        let q = [1.0, 0.25];
        let mut cases = 0usize;
        let mut skipped_cases = 0usize;
        for len in 1..=(if cfg!(miri) { 3usize } else { 4 }) {
            let total = 4usize.pow(u32::try_from(len).unwrap());
            for code in 0..total {
                let mut keys = Vec::new();
                let mut c = code;
                for _ in 0..len {
                    keys.extend_from_slice(&alphabet[c % 4]);
                    c /= 4;
                }
                for &eps in &[1e-3, 5e-2, 0.5] {
                    let read = check(&q, &keys, 2, 1.0, eps);
                    cases += 1;
                    if read < len {
                        skipped_cases += 1;
                    }
                }
            }
        }
        println!("norm-order grid: cases={cases} skipped_cases={skipped_cases}");
        if !cfg!(miri) {
            assert_eq!(cases, 340 * 3);
        }
    }

    #[test]
    fn random_certificate_sweep_including_heavy_tailed_norms() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (d, n) = (8, 48);
        for sigma in [0.0, 0.5, 1.0, 1.5] {
            for &eps in &[1e-4, 1e-2, 1e-1] {
                for _ in 0..(if cfg!(miri) { 2 } else { 40 }) {
                    let mut keys = Vec::new();
                    for _ in 0..n {
                        let s = (sigma * rng.gauss()).exp();
                        keys.extend((0..d).map(|_| s * rng.gauss()));
                    }
                    let q: Vec<f64> = (0..d).map(|_| 2.0 * rng.gauss()).collect();
                    check(&q, &keys, d, 1.0, eps);
                }
            }
        }
    }

    #[test]
    fn measured_regimes_versus_contiguous_two_stat() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let (d, n, b): (usize, usize, usize) = (16, 256, 16);
        let samples: usize = if cfg!(miri) { 2 } else { 60 };
        for sigma in [0.0, 0.5, 1.0, 1.5] {
            for &eps in &[1e-4, 1e-2, 1e-1] {
                let (mut norm_rows, mut box_rows, mut max_l1, mut max_cert) =
                    (0.0, 0.0, 0.0_f64, 0.0_f64);
                for _ in 0..samples {
                    let mut keys = Vec::new();
                    for _ in 0..n {
                        let s = (sigma * rng.gauss()).exp();
                        keys.extend((0..d).map(|_| s * rng.gauss()));
                    }
                    let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                    let (r, c) =
                        certified_norm_order_softmax_accounted(&q, &keys, d, 1.0, eps).unwrap();
                    let o = oracle(&q, &keys, d, 1.0);
                    let l = l1(&r.probabilities, &o);
                    assert!(l <= r.certified_l1_bound + 1e-12);
                    max_l1 = max_l1.max(l);
                    max_cert = max_cert.max(r.certified_l1_bound);
                    #[allow(clippy::cast_precision_loss)]
                    {
                        norm_rows += c.query_time_key_rows_read as f64 / n as f64;
                        // A non-finite box bound fails closed: count as all rows read.
                        box_rows +=
                            certified_qk_box_two_stat_softmax_accounted(&q, &keys, d, 1.0, b, eps)
                                .map_or(1.0, |(_, bc)| {
                                    bc.query_time_key_rows_read as f64 / n as f64
                                });
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                let s = samples as f64;
                println!(
                    "norm-order sigma={sigma} eps={eps}: rows_read_frac norm_order={:.4} two_stat_box={:.4} max_l1={max_l1:.3e} max_cert={max_cert:.3e}",
                    norm_rows / s,
                    box_rows / s
                );
            }
        }
    }
}
