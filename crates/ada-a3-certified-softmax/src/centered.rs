//! Mean-centered Cauchy–Schwarz certified softmax with per-query bound ordering (ADA-A3).
//!
//! Research reference only. With the key mean `μ` (a prefill statistic) and
//! `c_i = k_i − μ`, every score satisfies the exact identity
//! `z_i = s⟨q,μ⟩ + s⟨q,c_i⟩ ≤ s⟨q,μ⟩ + s‖q‖‖c_i‖`. Combined with the plain
//! `s‖q‖‖k_i‖` bound, `b_i = min(·,·)` is valid and never looser than
//! [`crate::certified_norm_sum_softmax`]. Keys are visited in non-increasing
//! `b_i` (per-query sort, no MACs); the unread mass obeys
//! `T ≤ Σ_{r≥j} exp(b_r − m)`; the scan stops when `2T/(S+T) ≤ ε`.
//! The only extra MACs are one `⟨q,μ⟩` dot (`d` MACs, counted).

/// Relative inflation applied to the Cauchy–Schwarz bound for rounding.
const BOUND_INFLATION: f64 = 1.0e-12;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CenteredCost {
    /// Key norms and centered norms computed at prefill (2 per key).
    pub prefill_norm_evaluations: usize,
    /// Extra MACs per query beyond key reads: the `⟨q,μ⟩` dot.
    pub extra_query_macs: usize,
    /// Per-query sorted keys (non-MAC comparisons-class work, `n log n`).
    pub query_sorted_keys: usize,
    /// Per-query `exp`/`ln` evaluations building the log-suffix bound table.
    pub bound_table_evaluations: usize,
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
pub struct CenteredSoftmax {
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
        return Err("ADA-A3 centered softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 centered softmax query must contain head_dim elements");
    }
    if query.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 centered softmax query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 centered softmax requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 centered softmax keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 centered softmax keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 centered softmax score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 centered softmax epsilon must be finite and non-negative");
    }
    Ok(keys.len() / head_dim)
}

fn norm(row: &[f64]) -> f64 {
    row.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Evaluate mean-centered certified softmax with costs.
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, a
/// non-positive `score_scale`, a negative or non-finite `epsilon`, or a
/// non-finite score or mass.
#[must_use = "the certified distribution should be checked"]
#[allow(clippy::too_many_lines)]
pub fn certified_centered_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<(CenteredSoftmax, CenteredCost), &'static str> {
    let n = validate(query, keys, head_dim, score_scale, epsilon)?;
    let q_norm = norm(query);
    let mut mean = vec![0.0_f64; head_dim];
    for row in keys.chunks_exact(head_dim) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n as f64;
    for m in &mut mean {
        *m *= inv_n;
    }
    let mean_norm = norm(&mean);
    if !mean_norm.is_finite() {
        return Err("ADA-A3 centered softmax mean must be finite");
    }
    let norms: Vec<f64> = keys.chunks_exact(head_dim).map(norm).collect();
    let cnorms: Vec<f64> = keys
        .chunks_exact(head_dim)
        .map(|r| {
            r.iter()
                .zip(&mean)
                .map(|(a, m)| (a - m) * (a - m))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    // The single extra query-time dot.
    let q_mean = query.iter().zip(&mean).map(|(a, b)| a * b).sum::<f64>();
    let bound_of = |idx: usize| {
        let plain = score_scale * q_norm * norms[idx] * (1.0 + BOUND_INFLATION);
        // Slack covers rounding in the computed mean, centered row and dot.
        let slack = BOUND_INFLATION * (norms[idx] + mean_norm);
        let centered = score_scale
            * (q_mean.abs().mul_add(BOUND_INFLATION, q_mean) + q_norm * (cnorms[idx] + slack));
        plain.min(centered)
    };
    let bounds: Vec<f64> = (0..n).map(bound_of).collect();
    if bounds.iter().any(|b| b.is_nan()) {
        return Err("ADA-A3 centered softmax bounds must not be NaN");
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| bounds[b].total_cmp(&bounds[a]).then(a.cmp(&b)));
    let mut log_suffix = vec![f64::NEG_INFINITY; n + 1];
    for r in (0..n).rev() {
        let b = bounds[order[r]];
        let prev = log_suffix[r + 1];
        log_suffix[r] = if prev == f64::NEG_INFINITY {
            b
        } else {
            let hi = b.max(prev);
            hi + ((b - hi).exp() + (prev - hi).exp()).ln()
        };
    }

    let mut cost = CenteredCost {
        prefill_norm_evaluations: 2 * n,
        extra_query_macs: head_dim,
        query_sorted_keys: n,
        bound_table_evaluations: 2 * n,
        ..CenteredCost::default()
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
            return Err("ADA-A3 centered softmax scores must be finite");
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
        // Per-key tail sum Σ exp(b_i − m) instead of (n−j)·exp(b_j − m).
        // Inflate by the log-sum rounding slack (relative 1e-9).
        let tail_hi = (log_suffix[read_so_far] - running_max).exp() * (1.0 + 1.0e-9);
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
        return Err("ADA-A3 centered softmax mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        CenteredSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
            score_scale,
        },
        cost,
    ))
}

/// Evaluate mean-centered certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_centered_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_centered_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<CenteredSoftmax, &'static str> {
    certified_centered_softmax_accounted(query, keys, head_dim, score_scale, epsilon)
        .map(|(result, _)| result)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::{certified_centered_softmax, certified_centered_softmax_accounted};
    use crate::{certified_norm_sum_softmax_accounted, dense_qk_scores};

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
        let r = certified_centered_softmax(query, keys, d, scale, eps).unwrap();
        let o = oracle(query, keys, d, scale);
        let observed = l1(&r.probabilities, &o);
        assert!(
            observed <= r.certified_l1_bound + 1e-12,
            "observed {observed} > bound {}",
            r.certified_l1_bound
        );
        assert!(r.certified_l1_bound <= eps.max(0.0) + 1e-12 || r.read.iter().all(|&x| x));
        r.read.iter().filter(|&&x| x).count()
    }

    /// Keys = common offset (scaled) + noise, with heavy-tailed noise norms.
    fn offset_keys(rng: &mut Rng, n: usize, d: usize, off: f64, sigma: f64) -> Vec<f64> {
        let mu: Vec<f64> = (0..d).map(|_| off * rng.gauss()).collect();
        let mut keys = Vec::new();
        for _ in 0..n {
            let s = (sigma * rng.gauss()).exp();
            keys.extend(mu.iter().map(|m| m + s * rng.gauss()));
        }
        keys
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let f = certified_centered_softmax;
        assert!(f(&[1.0], &[], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 0, 1.0, 0.1).is_err());
        assert!(f(&[1.0, 2.0], &[1.0], 1, 1.0, 0.1).is_err());
        assert!(f(&[f64::NAN], &[1.0], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[f64::INFINITY], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 0.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, -1.0).is_err());
    }

    #[test]
    fn identical_keys_and_zero_query_stay_exact_uniform() {
        let keys = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0];
        let r = certified_centered_softmax(&[1.0, 0.0], &keys, 2, 1.0, 1e-6).unwrap();
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        let keys = [1.0, 2.0, -3.0, 4.0, 5.0, 6.0];
        let r = certified_centered_softmax(&[0.0, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.read.iter().all(|&x| x));
    }

    #[test]
    fn overflow_and_cancellation_do_not_break_the_certificate() {
        let keys = [1.0e3, 0.0, 1.0e-3, 0.0, 5.0e2, 0.0];
        let r = certified_centered_softmax(&[1.0e3, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.probabilities.iter().all(|p| p.is_finite()));
        // Huge common offset: centered rows suffer catastrophic cancellation.
        let mut rng = Rng(7);
        let mut keys = Vec::new();
        for _ in 0..32 {
            keys.extend([
                1.0e9 + rng.gauss(),
                -1.0e9 + rng.gauss(),
                1.0e9 + rng.gauss(),
            ]);
        }
        for eps in [0.0, 1e-3, 0.3] {
            check(&[1.0e-9, 1.0e-9, -1.0e-9], &keys, 3, 1.0, eps);
        }
    }

    #[test]
    fn designed_witness_common_offset_hides_a_needle() {
        // All keys share a huge offset; plain norms are all ≈ equal (no pruning),
        // centered norms expose one outlier aligned with q.
        let d = 4;
        let n = 64;
        let mut keys = Vec::new();
        for i in 0..n {
            let mut row = [30.0, 30.0, 0.0, 0.0];
            if i == 0 {
                row[2] = 20.0;
            } else {
                row[3] = 0.01 * f64::from(u8::try_from(i % 5).unwrap());
            }
            keys.extend(row);
        }
        let q = [0.0, 0.0, 1.0, 0.0];
        let (rc, cc) = certified_centered_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        let (_, cn) = certified_norm_sum_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        println!(
            "witness centered_rows={} norm_sum_rows={}",
            cc.query_time_key_rows_read, cn.query_time_key_rows_read
        );
        assert!(cc.query_time_key_rows_read < cn.query_time_key_rows_read);
        let o = oracle(&q, &keys, d, 1.0);
        assert!(l1(&rc.probabilities, &o) <= rc.certified_l1_bound + 1e-12);
    }

    #[test]
    fn exhaustive_small_grid_has_zero_certificate_violations() {
        let alphabet: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]];
        let q = [1.0, 0.25];
        let mut cases = 0usize;
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
                    check(&q, &keys, 2, 1.0, eps);
                    cases += 1;
                }
            }
        }
        if !cfg!(miri) {
            assert_eq!(cases, 340 * 3);
        }
    }

    #[test]
    fn random_certificate_sweep_with_offsets_and_heavy_tails() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (d, n) = (8, 48);
        let draws = if cfg!(miri) { 1 } else { 30 };
        for off in [0.0, 1.0, 5.0, 50.0] {
            for sigma in [0.0, 1.0, 1.5] {
                for &eps in &[0.0, 1e-4, 1e-2, 1e-1] {
                    for _ in 0..draws {
                        let keys = offset_keys(&mut rng, n, d, off, sigma);
                        let q: Vec<f64> = (0..d).map(|_| 2.0 * rng.gauss()).collect();
                        check(&q, &keys, d, 1.0, eps);
                    }
                }
            }
        }
    }

    #[test]
    fn measured_regimes_versus_norm_sum() {
        if cfg!(miri) {
            return;
        }
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let (d, n) = (64, 256);
        for off in [0.0, 0.5, 1.0, 2.0] {
            for &(scale, eps) in &[(0.125, 1e-2), (0.5, 1e-2), (0.125, 1e-4)] {
                let (mut rc, mut rn) = (0.0, 0.0);
                let draws = 40_u32;
                for _ in 0..draws {
                    let keys = offset_keys(&mut rng, n, d, off, 0.5);
                    // Queries correlated with the offset direction half the time is
                    // not assumed: use a plain Gaussian query.
                    let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                    let (r, c) =
                        certified_centered_softmax_accounted(&q, &keys, d, scale, eps).unwrap();
                    let o = oracle(&q, &keys, d, scale);
                    assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
                    let (_, cn) =
                        certified_norm_sum_softmax_accounted(&q, &keys, d, scale, eps).unwrap();
                    // Net MACs/dense including the ⟨q,μ⟩ dot, in units of key rows.
                    #[allow(clippy::cast_precision_loss)]
                    {
                        rc += (c.query_time_key_rows_read * d + c.extra_query_macs) as f64
                            / (n * d) as f64;
                        rn += cn.query_time_key_rows_read as f64 / n as f64;
                    }
                }
                println!(
                    "STAT offset={off} scale={scale} eps={eps} centered_macs={:.3} norm_sum={:.3}",
                    rc / f64::from(draws),
                    rn / f64::from(draws)
                );
            }
        }
    }
}
