//! Mean-direction projection + residual certified softmax (ADA-A3).
//!
//! Research reference only. At prefill we fix a unit direction `u` (the
//! normalized mean key; `e_1` if the mean is zero) and store, per key,
//! `p_i = <k_i, u>` and the residual norm `r_i = ||k_i - p_i u||`.
//! For a query `q = a u + q_perp` (`a = s<q,u>`, `b = s||q_perp||`),
//! orthogonality gives `z_i = s<q,k_i> = a p_i + s<q_perp,k_i_perp>
//! <= a p_i + b r_i`. Keys are visited by decreasing `a p_i` (increasing or
//! decreasing `p` depending on the sign of `a`); every unread key obeys
//! `z_i <= a p_next + b R_j` where `R_j` is the maximum residual among the
//! unread keys (a suffix maximum precomputed for both scan directions).
//! This is a two-term Cauchy-Schwarz refinement of the norm-order bound of
//! `norm_order.rs`: the norm bound is recovered when `u` is ignored, and
//! the projection term separates equal-norm keys by direction. The scan
//! stops once `2 T_hi / (S + T_hi) <= epsilon`.

/// Relative slack applied to the bound for floating-point rounding.
const BOUND_INFLATION: f64 = 1.0e-12;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProjOrderCost {
    /// Prefill passes over the key matrix element-wise (mean, projections, residuals).
    pub prefill_key_passes: usize,
    /// Keys sorted at prefill (both scan directions share one sort).
    pub prefill_sorted_keys: usize,
    /// Dots of the query with the fixed direction `u` (one per query).
    pub query_time_direction_dots: usize,
    /// Key rows dotted with the query at inference time.
    pub query_time_key_rows_read: usize,
    /// Key rows that were not read.
    pub key_rows_skipped: usize,
    /// `exp` evaluations (scores read plus certificate evaluations).
    pub exp_evaluations: usize,
    /// Certificate evaluations along the scan.
    pub bound_checks: usize,
}

/// Projection-ordered certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjOrderSoftmax {
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
        return Err("ADA-A3 proj-order softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 proj-order softmax query must contain head_dim elements");
    }
    if query.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 proj-order softmax query must be finite");
    }
    if keys.is_empty() {
        return Err("ADA-A3 proj-order softmax requires at least one key");
    }
    if !keys.chunks_exact(head_dim).remainder().is_empty() {
        return Err("ADA-A3 proj-order softmax keys must be row-major [key_count, head_dim]");
    }
    if keys.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 proj-order softmax keys must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 proj-order softmax score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 proj-order softmax epsilon must be finite and non-negative");
    }
    Ok(keys.len() / head_dim)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Evaluate projection-ordered early-termination certified softmax with costs.
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, a
/// non-positive `score_scale`, a negative or non-finite `epsilon`, or a
/// non-finite score or mass.
#[must_use = "the certified distribution should be checked"]
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub fn certified_proj_order_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<(ProjOrderSoftmax, ProjOrderCost), &'static str> {
    let n = validate(query, keys, head_dim, score_scale, epsilon)?;
    let d = head_dim;

    // ---- prefill: direction, projections, residual norms, sort by p ----
    let mut u = vec![0.0_f64; d];
    for row in keys.chunks_exact(d) {
        for (acc, v) in u.iter_mut().zip(row) {
            *acc += v;
        }
    }
    let un = dot(&u, &u).sqrt();
    if un.is_finite() && un > 0.0 {
        for v in &mut u {
            *v /= un;
        }
    } else {
        u.fill(0.0);
        u[0] = 1.0;
    }
    let proj: Vec<f64> = keys.chunks_exact(d).map(|r| dot(r, &u)).collect();
    let resid: Vec<f64> = keys
        .chunks_exact(d)
        .zip(&proj)
        .map(|(r, &p)| {
            r.iter()
                .zip(&u)
                .map(|(k, uu)| (k - p * uu).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let max_key_norm = keys
        .chunks_exact(d)
        .map(|r| dot(r, r).sqrt())
        .fold(0.0_f64, f64::max);
    let mut asc: Vec<usize> = (0..n).collect();
    asc.sort_by(|&a, &b| proj[a].total_cmp(&proj[b]).then(a.cmp(&b)));
    // suffix maxima of residuals in both scan orders
    // desc order = reverse of asc; suffix_desc[j] = max resid over desc[j..]
    //            = max resid over asc[..n-j]  => prefix max of asc.
    let mut prefix_max_asc = vec![0.0_f64; n + 1];
    for (i, &idx) in asc.iter().enumerate() {
        prefix_max_asc[i + 1] = prefix_max_asc[i].max(resid[idx]);
    }
    let mut suffix_max_asc = vec![0.0_f64; n + 1];
    for i in (0..n).rev() {
        suffix_max_asc[i] = suffix_max_asc[i + 1].max(resid[asc[i]]);
    }

    // ---- query ----
    let a = score_scale * dot(query, &u);
    let q_perp: f64 = query
        .iter()
        .zip(&u)
        .map(|(q, uu)| (q - (dot(query, &u)) * uu).powi(2))
        .sum::<f64>()
        .sqrt();
    let b = score_scale * q_perp;
    let slack = BOUND_INFLATION * score_scale * dot(query, query).sqrt() * max_key_norm;
    let descending = a >= 0.0;
    // scan position j -> key index
    let at = |j: usize| if descending { asc[n - 1 - j] } else { asc[j] };
    let tail_resid = |j: usize| {
        // max residual among scan positions j..n
        if descending {
            prefix_max_asc[n - j]
        } else {
            suffix_max_asc[j]
        }
    };
    let bound_of = |j: usize| a * proj[at(j)] + b * tail_resid(j) + slack;

    let mut cost = ProjOrderCost {
        prefill_key_passes: 3,
        prefill_sorted_keys: n,
        query_time_direction_dots: 1,
        ..ProjOrderCost::default()
    };
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64;
    let mut certificate = 0.0_f64;
    let mut read_count = n;

    for rank in 0..n {
        let idx = at(rank);
        let row = &keys[idx * d..(idx + 1) * d];
        let score = score_scale * dot(query, row);
        if !score.is_finite() {
            return Err("ADA-A3 proj-order softmax scores must be finite");
        }
        cost.query_time_key_rows_read += 1;
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
        cost.exp_evaluations += 1;
        scores.push((idx, score));

        let read_so_far = rank + 1;
        if read_so_far == n {
            certificate = 0.0;
            break;
        }
        let remaining = n - read_so_far;
        #[allow(clippy::cast_precision_loss)]
        let tail_hi = remaining as f64 * (bound_of(read_so_far) - running_max).exp();
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
        return Err("ADA-A3 proj-order softmax mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        ProjOrderSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
            score_scale,
        },
        cost,
    ))
}

/// Evaluate projection-ordered early-termination certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_proj_order_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_proj_order_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
) -> Result<ProjOrderSoftmax, &'static str> {
    certified_proj_order_softmax_accounted(query, keys, head_dim, score_scale, epsilon)
        .map(|(result, _)| result)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::{certified_proj_order_softmax, certified_proj_order_softmax_accounted};
    use crate::{certified_norm_order_softmax_accounted, dense_qk_scores};

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
        let r = certified_proj_order_softmax(query, keys, d, scale, eps).unwrap();
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
        let f = certified_proj_order_softmax;
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
    fn singleton_equal_keys_zero_query_and_zero_keys() {
        let r = certified_proj_order_softmax(&[1.0, 0.0], &[0.5, 0.5], 2, 1.0, 0.0).unwrap();
        assert_eq!(r.probabilities, vec![1.0]);
        let keys = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let r = certified_proj_order_softmax(&[1.0, 0.0], &keys, 2, 1.0, 1e-6).unwrap();
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        let keys = [1.0, 2.0, -3.0, 4.0, 5.0, 6.0];
        let r = certified_proj_order_softmax(&[0.0, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.read.iter().all(|&x| x));
        // all-zero keys: mean zero, fallback direction, uniform output
        let r = certified_proj_order_softmax(&[1.0, 1.0], &[0.0; 6], 2, 1.0, 1e-3).unwrap();
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        // antipodal keys: mean zero
        check(
            &[1.0, 0.3],
            &[1.0, 0.0, -1.0, 0.0, 0.0, 2.0, 0.0, -2.0],
            2,
            1.0,
            0.1,
        );
    }

    #[test]
    fn overflowing_bound_is_handled_without_nan() {
        let keys = [1.0e3, 0.0, 1.0e-3, 0.0, 5.0e2, 0.0];
        let r = certified_proj_order_softmax(&[1.0e3, 0.0], &keys, 2, 1.0, 1e-3).unwrap();
        assert!(r.probabilities.iter().all(|p| p.is_finite()));
        assert!(r.certified_l1_bound.is_finite());
    }

    #[test]
    fn negative_alignment_scans_the_other_direction() {
        // Mean direction ~ +e1; query anti-aligned: the best key is the one with smallest p.
        let d = 2;
        let mut keys = Vec::new();
        for i in 0..40 {
            keys.extend_from_slice(&[3.0 + 0.1 * f64::from(i), 0.5]);
        }
        let q = [-4.0, 0.0];
        let (r, c) = certified_proj_order_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        assert!(c.query_time_key_rows_read < 40);
        let argmax = r
            .probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(argmax, 0);
        let o = oracle(&q, &keys, d, 1.0);
        assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn designed_witness_equal_norm_keys_where_norm_order_reads_everything() {
        // 65 keys of identical norm 5: needle (5,0) aligned with the query;
        // 64 distractors at (0, +-5), balanced so the mean direction is e_1.
        let d = 2;
        let mut keys = vec![5.0, 0.0];
        for i in 1..65 {
            keys.extend_from_slice(&[0.0, if i % 2 == 0 { 5.0 } else { -5.0 }]);
        }
        let q = [4.0, 0.5];
        let (r, c) = certified_proj_order_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        let (_, nc) = certified_norm_order_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        println!(
            "proj witness: proj_rows={} norm_rows={}",
            c.query_time_key_rows_read, nc.query_time_key_rows_read
        );
        assert_eq!(nc.query_time_key_rows_read, 65);
        assert_eq!(c.query_time_key_rows_read, 1);
        assert!(r.probabilities[0] > 0.999);
        let o = oracle(&q, &keys, d, 1.0);
        assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn exhaustive_small_grid_has_zero_certificate_violations() {
        let alphabet: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0]];
        let queries = [[1.0, 0.25], [-1.0, 0.5]];
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
                for q in &queries {
                    for &eps in &[1e-3, 5e-2, 0.5] {
                        let read = check(q, &keys, 2, 1.0, eps);
                        cases += 1;
                        if read < len {
                            skipped_cases += 1;
                        }
                    }
                }
            }
        }
        println!("proj-order grid: cases={cases} skipped_cases={skipped_cases}");
        if !cfg!(miri) {
            assert_eq!(cases, 340 * 6);
        }
    }

    #[test]
    fn random_certificate_sweep_with_mean_shift_and_heavy_norms() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let (d, n) = (8, 48);
        for shift in [0.0, 1.0, 3.0] {
            for sigma in [0.0, 1.0] {
                for &eps in &[1e-4, 1e-2, 1e-1] {
                    for _ in 0..(if cfg!(miri) { 2 } else { 30 }) {
                        let mut keys = Vec::new();
                        for _ in 0..n {
                            let s = (sigma * rng.gauss()).exp();
                            keys.extend(
                                (0..d).map(|j| s * rng.gauss() + if j == 0 { shift } else { 0.0 }),
                            );
                        }
                        let q: Vec<f64> = (0..d).map(|_| 2.0 * rng.gauss()).collect();
                        check(&q, &keys, d, 1.0, eps);
                    }
                }
            }
        }
    }

    #[test]
    fn measured_regimes_versus_norm_order() {
        let mut rng = Rng(0xA5A5_5A5A_C3C3_3C3C);
        let (d, n): (usize, usize) = (16, 256);
        let samples: usize = if cfg!(miri) { 2 } else { 60 };
        // shift = mean-key magnitude along e_1; rho = query component along e_1
        for (shift, rho) in [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (4.0, 2.0),
            (4.0, 4.0),
            (8.0, 4.0),
        ] {
            for &eps in &[1e-4, 1e-2, 1e-1] {
                let (mut pr, mut nr, mut max_l1, mut max_cert) = (0.0, 0.0, 0.0_f64, 0.0_f64);
                for _ in 0..samples {
                    let mut keys = Vec::new();
                    for _ in 0..n {
                        keys.extend((0..d).map(|j| rng.gauss() + if j == 0 { shift } else { 0.0 }));
                    }
                    let q: Vec<f64> = (0..d)
                        .map(|j| rng.gauss() + if j == 0 { rho } else { 0.0 })
                        .collect();
                    let (r, c) =
                        certified_proj_order_softmax_accounted(&q, &keys, d, 1.0, eps).unwrap();
                    let (_, nc) =
                        certified_norm_order_softmax_accounted(&q, &keys, d, 1.0, eps).unwrap();
                    let l = l1(&r.probabilities, &oracle(&q, &keys, d, 1.0));
                    assert!(l <= r.certified_l1_bound + 1e-12);
                    max_l1 = max_l1.max(l);
                    max_cert = max_cert.max(r.certified_l1_bound);
                    #[allow(clippy::cast_precision_loss)]
                    {
                        pr += c.query_time_key_rows_read as f64 / n as f64;
                        nr += nc.query_time_key_rows_read as f64 / n as f64;
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                let s = samples as f64;
                println!(
                    "proj-order shift={shift} rho={rho} eps={eps}: rows_read_frac proj={:.4} norm={:.4} max_l1={max_l1:.3e} max_cert={max_cert:.3e}",
                    pr / s,
                    nr / s
                );
            }
        }
    }
}
