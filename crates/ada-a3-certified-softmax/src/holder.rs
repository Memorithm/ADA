//! Hölder-mixed (ℓ2/ℓ∞·ℓ1) certified softmax with per-query bound ordering (ADA-A3).
//!
//! Research reference only. For any key and query the exact inequalities
//! `⟨q,k⟩ ≤ ‖q‖₂‖k‖₂`, `⟨q,k⟩ ≤ ‖q‖∞‖k‖₁` and `⟨q,k⟩ ≤ ‖q‖₁‖k‖∞` hold, so
//! `b_i = s·min(·,·,·)` is a valid score upper bound, never looser than the
//! Cauchy–Schwarz bound of [`crate::certified_norm_sum_softmax`]. Keys are
//! visited in non-increasing `b_i` (per-query sort, no MACs); the scan stops
//! when `2T/(S+T) ≤ ε` with `T ≤ Σ_{unread} exp(b_r − m)`. Extra cost: three
//! query norms (`3d` non-MAC abs/add/max ops) and three prefill norms per key.

/// Relative inflation applied to the Cauchy–Schwarz bound for rounding.
const BOUND_INFLATION: f64 = 1.0e-12;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HolderCost {
    /// Key norms computed at prefill (ℓ2, ℓ1, ℓ∞ per key).
    pub prefill_norm_evaluations: usize,
    /// Per-query sorted keys (comparison-class work, `n log n`).
    pub query_sorted_keys: usize,
    /// Per-query non-MAC norm operations (`‖q‖₂`, `‖q‖₁`, `‖q‖∞`).
    pub query_norm_ops: usize,
    /// MACs spent beyond key-row dots (always 0 here: norms only).
    pub extra_query_macs: usize,
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
pub struct HolderSoftmax {
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
        return Err("ADA-A3 holder softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 holder softmax query must contain head_dim elements");
    }
    if query.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 holder softmax query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 holder softmax requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 holder softmax keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 holder softmax keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 holder softmax score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 holder softmax epsilon must be finite and non-negative");
    }
    Ok(keys.len() / head_dim)
}

fn norm(row: &[f64]) -> f64 {
    row.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Evaluate holdered early-termination certified softmax with costs.
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, a
/// non-positive `score_scale`, a negative or non-finite `epsilon`, or a
/// non-finite score or mass.
#[must_use = "the certified distribution should be checked"]
pub fn certified_holder_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<(HolderSoftmax, HolderCost), &'static str> {
    let n = validate(query, keys, head_dim, score_scale, epsilon)?;
    let q2 = norm(query);
    let q1: f64 = query.iter().map(|v| v.abs()).sum();
    let qinf = query.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let k2: Vec<f64> = keys.chunks_exact(head_dim).map(norm).collect();
    let k1: Vec<f64> = keys
        .chunks_exact(head_dim)
        .map(|r| r.iter().map(|v| v.abs()).sum())
        .collect();
    let kinf: Vec<f64> = keys
        .chunks_exact(head_dim)
        .map(|r| r.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
        .collect();
    let bound_of = |idx: usize| {
        let b = (q2 * k2[idx]).min(qinf * k1[idx]).min(q1 * kinf[idx]);
        score_scale * b * (1.0 + BOUND_INFLATION)
    };
    let bounds: Vec<f64> = (0..n).map(bound_of).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| bounds[b].total_cmp(&bounds[a]).then(a.cmp(&b)));
    // log_suffix[j] = ln Σ_{r ≥ j} exp(bound_of(order[r])), computed in the log
    // domain (no MACs; n exp/ln evaluations) so huge bounds cannot overflow.
    let mut log_suffix = vec![f64::NEG_INFINITY; n + 1];
    for r in (0..n).rev() {
        let b = bound_of(order[r]);
        let prev = log_suffix[r + 1];
        log_suffix[r] = if prev == f64::NEG_INFINITY {
            b
        } else {
            let hi = b.max(prev);
            hi + ((b - hi).exp() + (prev - hi).exp()).ln()
        };
    }

    let mut cost = HolderCost {
        prefill_norm_evaluations: 3 * n,
        query_sorted_keys: n,
        query_norm_ops: 3 * head_dim,
        bound_table_evaluations: 2 * n,
        ..HolderCost::default()
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
            return Err("ADA-A3 holder softmax scores must be finite");
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
        return Err("ADA-A3 holder softmax mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        HolderSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
            score_scale,
        },
        cost,
    ))
}

/// Evaluate holdered early-termination certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_holder_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_holder_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<HolderSoftmax, &'static str> {
    certified_holder_softmax_accounted(query, keys, head_dim, score_scale, epsilon)
        .map(|(result, _)| result)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::{certified_holder_softmax, certified_holder_softmax_accounted};
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
        let r = certified_holder_softmax(query, keys, d, scale, eps).unwrap();
        let o = oracle(query, keys, d, scale);
        let observed = l1(&r.probabilities, &o);
        assert!(
            observed <= r.certified_l1_bound + 1e-12,
            "observed {observed} > bound {}",
            r.certified_l1_bound
        );
        assert!(r.certified_l1_bound <= eps || r.read.iter().all(|&x| x));
        r.read.iter().filter(|&&x| x).count()
    }

    /// Sparse keys: `m` nonzero Gaussian coordinates among `d`, optional heavy-tail scale.
    fn sparse_keys(rng: &mut Rng, n: usize, d: usize, m: usize, sigma: f64) -> Vec<f64> {
        let mut keys = vec![0.0; n * d];
        for i in 0..n {
            let s = (sigma * rng.gauss()).exp();
            for _ in 0..m {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let j = (rng.unit() * d as f64) as usize % d;
                keys[i * d + j] = s * rng.gauss();
            }
        }
        keys
    }

    #[test]
    fn fail_closed_on_invalid_inputs() {
        let f = certified_holder_softmax;
        assert!(f(&[1.0], &[], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 0, 1.0, 0.1).is_err());
        assert!(f(&[1.0, 2.0], &[1.0], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0, 2.0, 3.0], 2, 1.0, 0.1).is_err());
        assert!(f(&[f64::NAN], &[1.0], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[f64::INFINITY], 1, 1.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 0.0, 0.1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, -1.0).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn zero_query_cannot_skip_and_is_uniform() {
        let keys = [1.0, 2.0, -3.0, 4.0, 5.0, 6.0];
        let r = certified_holder_softmax(&[0.0, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.read.iter().all(|&x| x));
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn overflowing_bound_is_handled_without_nan() {
        let keys = [1.0e3, 0.0, 1.0e-3, 0.0, 5.0e2, 0.0];
        let r = certified_holder_softmax(&[1.0e3, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.probabilities.iter().all(|p| p.is_finite()));
        assert!(r.certified_l1_bound.is_finite());
    }

    #[test]
    fn each_holder_pair_is_a_valid_upper_bound() {
        // Adversarial vectors: one-hot, constant sign, alternating, outliers.
        let mut rng = Rng(11);
        let d = 12;
        for _ in 0..(if cfg!(miri) { 20 } else { 2000 }) {
            let q: Vec<f64> = (0..d)
                .map(|_| 1e3_f64.powf(rng.gauss().tanh()) * rng.gauss())
                .collect();
            let k: Vec<f64> = (0..d)
                .map(|_| 1e3_f64.powf(rng.gauss().tanh()) * rng.gauss())
                .collect();
            let dot: f64 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
            let q2 = q.iter().map(|v| v * v).sum::<f64>().sqrt();
            let k2 = k.iter().map(|v| v * v).sum::<f64>().sqrt();
            let q1: f64 = q.iter().map(|v| v.abs()).sum();
            let k1: f64 = k.iter().map(|v| v.abs()).sum();
            let qi = q.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let ki = k.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let slack = 1.0 + 1e-12;
            assert!(dot <= q2 * k2 * slack);
            assert!(dot <= qi * k1 * slack);
            assert!(dot <= q1 * ki * slack);
        }
    }

    #[test]
    fn designed_witness_sparse_needle_beats_cauchy_schwarz() {
        // d=64, distractors are 1-sparse (norm 3), needle is aligned and large.
        // Gaussian-like dense q: ‖q‖∞ ≪ ‖q‖₂, so ℓ∞·ℓ1 is far tighter.
        let d = 64;
        let n = 64;
        let mut keys = vec![0.0; n * d];
        keys[0] = 25.0;
        for i in 1..n {
            keys[i * d + (i % d)] = 3.0;
        }
        let q = vec![1.0; d];
        let (r, c) = certified_holder_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        let (_, cn) = certified_norm_sum_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        assert!(c.query_time_key_rows_read < cn.query_time_key_rows_read);
        let o = oracle(&q, &keys, d, 1.0);
        assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn exhaustive_small_grid_has_zero_certificate_violations() {
        let alphabet: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]];
        let qs = [[1.0, 0.25], [0.0, 3.0], [-1.0, -1.0]];
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
                for q in &qs {
                    for &eps in &[0.0, 1e-3, 5e-2, 0.5] {
                        check(q, &keys, 2, 1.0, eps);
                        cases += 1;
                    }
                }
            }
        }
        println!("holder grid: cases={cases}");
        if !cfg!(miri) {
            assert_eq!(cases, 340 * 12);
        }
    }

    #[test]
    fn random_sweep_dense_sparse_heavy_tailed_zero_violations() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (d, n) = (16, 48);
        for m in [1usize, 3, 16] {
            for sigma in [0.0, 1.0] {
                for &eps in &[1e-4, 1e-2, 1e-1] {
                    for _ in 0..(if cfg!(miri) { 2 } else { 30 }) {
                        let keys = sparse_keys(&mut rng, n, d, m, sigma);
                        let q: Vec<f64> = (0..d).map(|_| 2.0 * rng.gauss()).collect();
                        check(&q, &keys, d, 1.0, eps);
                    }
                }
            }
        }
    }

    #[test]
    fn never_reads_more_than_norm_sum() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let (d, n) = (16, 48);
        let mut strictly_fewer = 0usize;
        for m in [1usize, 2, 4, 16] {
            for &eps in &[1e-3, 1e-1] {
                for _ in 0..(if cfg!(miri) { 2 } else { 30 }) {
                    let keys = sparse_keys(&mut rng, n, d, m, 1.0);
                    let q: Vec<f64> = (0..d).map(|_| 0.5 * rng.gauss()).collect();
                    let a = check(&q, &keys, d, 1.0, eps);
                    let (r, _) =
                        certified_norm_sum_softmax_accounted(&q, &keys, d, 1.0, eps).unwrap();
                    let b = r.read.iter().filter(|&&x| x).count();
                    // Same bound family after min(): fewer-or-equal rows up to ties.
                    assert!(a <= b, "holder read {a} > norm-sum {b}");
                    strictly_fewer += usize::from(a < b);
                }
            }
        }
        println!("holder strictly fewer than norm-sum in {strictly_fewer} draws");
    }

    #[test]
    fn cost_accounting_is_consistent() {
        let mut rng = Rng(5);
        let (d, n) = (8, 20);
        let keys = sparse_keys(&mut rng, n, d, 2, 0.5);
        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let (_, c) = certified_holder_softmax_accounted(&q, &keys, d, 1.0, 1e-2).unwrap();
        assert_eq!(c.query_time_key_rows_read + c.key_rows_skipped, n);
        assert_eq!(c.prefill_norm_evaluations, 3 * n);
        assert_eq!(c.query_norm_ops, 3 * d);
        assert_eq!(c.extra_query_macs, 0);
    }

    /// `cargo test -p ada-a3-certified-softmax --release -- --ignored --nocapture holder_stats`.
    #[test]
    #[ignore = "measurement"]
    fn holder_stats() {
        let mut rng = Rng(0x00AB_CDEF);
        let (d, n) = (64, 256);
        for m in [1usize, 4, 8, 16, 64] {
            for &(scale, eps) in &[(0.5, 1e-2), (0.125, 1e-2), (0.5, 1e-4)] {
                let (mut ra, mut rb) = (0usize, 0usize);
                let draws: u32 = 40;
                for _ in 0..draws {
                    let keys = sparse_keys(&mut rng, n, d, m, 0.5);
                    let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                    ra += check(&q, &keys, d, scale, eps);
                    let (r, _) =
                        certified_norm_sum_softmax_accounted(&q, &keys, d, scale, eps).unwrap();
                    rb += r.read.iter().filter(|&&x| x).count();
                }
                #[allow(clippy::cast_precision_loss)]
                let tot = f64::from(draws * 256);
                #[allow(clippy::cast_precision_loss)]
                {
                    println!(
                        "STAT nnz={m} scale={scale} eps={eps} holder={:.3} norm_sum={:.3}",
                        ra as f64 / tot,
                        rb as f64 / tot
                    );
                }
            }
        }
    }
}
