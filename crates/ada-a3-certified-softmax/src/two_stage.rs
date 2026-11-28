//! Two-stage reduced-rank lazy refinement with certified exact completion
//! (ADA-A3, research reference).
//!
//! Conjecture: the reduced-rank screen of [`crate::certified_reduced_rank_softmax_accounted`]
//! pays `n·r` MACs for every key at a single rank. Use the leading `r1` rows of
//! one basis to screen all keys (`n·r1`), then refine a key to rank `r2 > r1`
//! only when it reaches the top of the bound frontier (extra `r2−r1` MACs, since
//! the first `r1` coefficients are a prefix), and only afterwards complete it
//! exactly (`d` MACs). Both ranks give valid upper bounds `h¹ᵢ`, `h²ᵢ` (same
//! derivation as `reduced_rank`: `|⟨q,k⟩ − ⟨q_r,c_k⟩| ≤ ‖u‖ρ_k + ν‖q_r‖‖c_k‖`,
//! valid for ANY basis), so `min(h¹ᵢ,h²ᵢ)` is valid. Stop when
//! `2T/(S+T) ≤ ε`, `T = Σ_{unread} e^{bound}` recomputed from scratch each step
//! (positive terms only, no cancellation). The bound scans are non-MAC work and
//! are reported separately.

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

const MAX_DIM: usize = 512;
const INFL: f64 = 1.0e-9;
const DOT_SLACK: f64 = 1.0e-11;
const U2_SLACK: f64 = 1.0e-12;

/// Logical (not hardware) costs of one two-stage query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TwoStageCost {
    /// Stage ranks `(r1, r2)`.
    pub ranks: (usize, usize),
    /// `q_r = P q` at the larger rank (`r2·d` MACs, prefix gives `r1`).
    pub query_projection_macs: usize,
    /// Residual norm MACs for both ranks (`r1² + r2²`).
    pub residual_norm_macs: usize,
    /// Stage-1 screen MACs (`n·r1`).
    pub stage1_macs: usize,
    /// Stage-2 refinement MACs (`refinements·(r2−r1)`, prefix reuse).
    pub stage2_macs: usize,
    /// Exact completion MACs (`rows_completed·d`).
    pub completion_macs: usize,
    /// Keys refined to rank `r2`.
    pub refinements: usize,
    /// Keys completed exactly.
    pub rows_completed: usize,
    /// Keys never completed.
    pub rows_skipped: usize,
    /// Non-MAC frontier work: bound scans (`n` per step).
    pub frontier_scan_ops: usize,
    /// One-time prefill MACs (`2·n·r2·d + r2²·d`).
    pub prefill_macs: usize,
    /// Dense reference MACs (`n·d`).
    pub dense_macs: usize,
}

impl TwoStageCost {
    /// Per-query MACs excluding amortised prefill.
    #[must_use]
    pub fn per_query_macs(&self) -> usize {
        self.query_projection_macs
            + self.residual_norm_macs
            + self.stage1_macs
            + self.stage2_macs
            + self.completion_macs
    }
}

/// Two-stage certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoStageSoftmax {
    /// Renormalised probabilities; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when every key was completed).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was completed exactly.
    pub read: Vec<bool>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Per-rank prefill data (prefix of one basis).
struct Rank {
    r: usize,
    nu: f64,
    coeff: Vec<f64>, // n × r
    rho: Vec<f64>,
    u_norm: f64,
    qr: Vec<f64>,
    qr_norm: f64,
}

#[allow(clippy::too_many_arguments)]
fn build_rank(
    query: &[f64],
    keys: &[f64],
    d: usize,
    basis: &[f64],
    r: usize,
    knorm: &[f64],
) -> Rank {
    let n = keys.len() / d;
    let b = &basis[..r * d];
    let mut gram = vec![0.0_f64; r * r];
    for a in 0..r {
        for c in 0..r {
            gram[a * r + c] = dot(&b[a * d..(a + 1) * d], &b[c * d..(c + 1) * d]);
        }
    }
    let mut acc = 0.0_f64;
    for a in 0..r {
        for c in 0..r {
            let e = gram[a * r + c] - f64::from(u8::from(a == c));
            acc += e * e;
        }
    }
    let nu = acc.sqrt();
    let mut coeff = vec![0.0_f64; n * r];
    let mut rho = vec![0.0_f64; n];
    for (i, row) in keys.chunks_exact(d).enumerate() {
        let mut res = row.to_vec();
        for a in 0..r {
            let c = dot(&b[a * d..(a + 1) * d], row);
            coeff[i * r + a] = c;
            for j in 0..d {
                res[j] -= c * b[a * d + j];
            }
        }
        rho[i] = dot(&res, &res).sqrt() * (1.0 + INFL) + DOT_SLACK * knorm[i];
    }
    let qr: Vec<f64> = (0..r).map(|a| dot(&b[a * d..(a + 1) * d], query)).collect();
    let q2 = dot(query, query);
    let qr2 = dot(&qr, &qr);
    let mut qgq = 0.0_f64;
    for a in 0..r {
        qgq += qr[a] * dot(&gram[a * r..(a + 1) * r], &qr);
    }
    let u2 = (q2 - 2.0 * qr2 + qgq).max(0.0) + U2_SLACK * (q2 + qgq.abs());
    Rank {
        r,
        nu,
        coeff,
        rho,
        u_norm: u2.sqrt(),
        qr,
        qr_norm: qr2.sqrt(),
    }
}

impl Rank {
    /// Upper bound on the scaled score of key `i`.
    fn upper(&self, i: usize, scale: f64, q_norm: f64, knorm: f64) -> f64 {
        let c = &self.coeff[i * self.r..(i + 1) * self.r];
        let p = dot(&self.qr, c) * scale;
        let cn = dot(c, c).sqrt();
        let delta = scale
            * ((self.u_norm * self.rho[i] + self.nu * (1.0 + INFL) * self.qr_norm * cn)
                * (1.0 + INFL)
                + DOT_SLACK * q_norm * knorm);
        p + delta
    }
}

/// Certified softmax with two-stage rank refinement.
///
/// `basis` is row-major `r2×head_dim`; the first `r1` rows are the stage-1
/// basis. Any finite matrix is accepted and only affects tightness.
///
/// # Errors
///
/// Inconsistent layout, non-finite input, invalid `score_scale`/`epsilon`,
/// ranks outside `1 ≤ r1 ≤ r2 ≤ d`, or non-finite scores.
#[must_use = "the certified distribution should be checked"]
pub fn certified_two_stage_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    ranks: (usize, usize),
    basis: &[f64],
) -> Result<(TwoStageSoftmax, TwoStageCost), &'static str> {
    let d = head_dim;
    let (r1, r2) = ranks;
    if d == 0 || d > MAX_DIM {
        return Err("ADA-A3 two-stage head_dim must be in 1..=512");
    }
    if r1 == 0 || r1 > r2 || r2 > d {
        return Err("ADA-A3 two-stage ranks must satisfy 1 <= r1 <= r2 <= head_dim");
    }
    if basis.len() != r2 * d || basis.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 two-stage basis must be finite row-major r2×d");
    }
    if query.len() != d || keys.is_empty() || keys.len() % d != 0 {
        return Err("ADA-A3 two-stage query/keys layout mismatch");
    }
    if query.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err("ADA-A3 two-stage inputs must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 two-stage score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 two-stage epsilon must be finite and non-negative");
    }
    let n = keys.len() / d;
    let knorm: Vec<f64> = keys
        .chunks_exact(d)
        .map(|row| dot(row, row).sqrt())
        .collect();
    let q_norm = dot(query, query).sqrt();
    let s1 = build_rank(query, keys, d, basis, r1, &knorm);
    let s2 = build_rank(query, keys, d, basis, r2, &knorm);

    let mut bound = vec![0.0_f64; n];
    for i in 0..n {
        let h = s1.upper(i, score_scale, q_norm, knorm[i]);
        if !h.is_finite() {
            return Err("ADA-A3 two-stage scores must be finite");
        }
        bound[i] = h;
    }
    let mut refined = vec![r1 == r2; n];
    let mut is_read = vec![false; n];
    let mut cost = TwoStageCost {
        ranks,
        query_projection_macs: r2 * d,
        residual_norm_macs: r1 * r1 + r2 * r2,
        stage1_macs: n * r1,
        prefill_macs: 2 * n * r2 * d + r2 * r2 * d,
        dense_macs: n * d,
        ..TwoStageCost::default()
    };
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64;
    let mut certificate = 0.0_f64;
    let mut done = n;
    loop {
        // Frontier scan (non-MAC): best unread key and unread mass.
        cost.frontier_scan_ops += n;
        let mut top = usize::MAX;
        let mut hmax = f64::NEG_INFINITY;
        for i in 0..n {
            if !is_read[i] && bound[i] > hmax {
                hmax = bound[i];
                top = i;
            }
        }
        if top == usize::MAX {
            break; // everything read
        }
        if !scores.is_empty() {
            let mut t = 0.0_f64;
            for i in 0..n {
                if !is_read[i] {
                    t += (bound[i] - hmax).exp();
                }
            }
            let ln_t = hmax + t.ln() + INFL.ln_1p() + 1.0e-9;
            let ln_s = running_max + mass.ln();
            let cert = 2.0 / (1.0 + (ln_s - ln_t).exp());
            if cert <= epsilon {
                certificate = cert;
                done = scores.len();
                break;
            }
        }
        if !refined[top] {
            let h2 = s2.upper(top, score_scale, q_norm, knorm[top]);
            if !h2.is_finite() {
                return Err("ADA-A3 two-stage scores must be finite");
            }
            bound[top] = bound[top].min(h2);
            refined[top] = true;
            cost.refinements += 1;
            cost.stage2_macs += r2 - r1;
            continue;
        }
        let row = &keys[top * d..(top + 1) * d];
        let score = dot(query, row) * score_scale;
        if !score.is_finite() {
            return Err("ADA-A3 two-stage scores must be finite");
        }
        cost.rows_completed += 1;
        cost.completion_macs += d;
        is_read[top] = true;
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
        scores.push((top, score));
    }
    cost.rows_skipped = n - done;

    let total: f64 = scores.iter().map(|&(_, s)| (s - running_max).exp()).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("ADA-A3 two-stage mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        TwoStageSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
        },
        cost,
    ))
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
mod tests {
    use super::certified_two_stage_softmax_accounted as run;
    use crate::{certified_reduced_rank_softmax_accounted, dense_qk_scores, reduced_rank_basis};

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

    /// Independent oracle: dense stable softmax on the original data.
    fn oracle(q: &[f64], keys: &[f64], d: usize, scale: f64) -> Vec<f64> {
        let sc = dense_qk_scores(q, keys, d, scale).unwrap();
        let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let t: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = t.iter().sum();
        t.into_iter().map(|x| x / z).collect()
    }

    fn l1(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }

    fn random_orthogonal(rng: &mut Rng, d: usize) -> Vec<f64> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        while rows.len() < d {
            let mut v: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
            for _ in 0..2 {
                for r in &rows {
                    let dt: f64 = v.iter().zip(r).map(|(a, b)| a * b).sum();
                    for (x, y) in v.iter_mut().zip(r) {
                        *x -= dt * y;
                    }
                }
            }
            let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nn > 1e-6 {
                rows.push(v.into_iter().map(|x| x / nn).collect());
            }
        }
        rows.concat()
    }

    fn rot(r: &[f64], v: &[f64], d: usize) -> Vec<f64> {
        (0..d)
            .map(|i| (0..d).map(|j| r[i * d + j] * v[j]).sum())
            .collect()
    }

    /// Anisotropic Gaussian rows (std e^{-decay j}) optionally hidden-rotated.
    fn sample(
        rng: &mut Rng,
        rotm: Option<&[f64]>,
        rows: usize,
        d: usize,
        decay: f64,
        amp: f64,
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows * d);
        for _ in 0..rows {
            let v: Vec<f64> = (0..d)
                .map(|j| amp * rng.gauss() * (-decay * j as f64).exp())
                .collect();
            out.extend(rotm.map_or(v.clone(), |r| rot(r, &v, d)));
        }
        out
    }

    #[test]
    fn rejects_invalid_input() {
        let k = vec![1.0; 8];
        let q = vec![1.0; 4];
        let b = vec![1.0; 8];
        assert!(run(&q, &k, 4, 1.0, 0.1, (0, 2), &b).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, (3, 2), &b).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, (1, 5), &b).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, (1, 2), &b[..7]).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, (1, 2), &[f64::NAN; 8]).is_err());
        assert!(run(&q, &k, 4, 0.0, 0.1, (1, 2), &b).is_err());
        assert!(run(&q, &k, 4, 1.0, -1.0, (1, 2), &b).is_err());
        assert!(run(&q[..3], &k, 4, 1.0, 0.1, (1, 2), &b).is_err());
        assert!(run(&q, &k[..7], 4, 1.0, 0.1, (1, 2), &b).is_err());
        let mut kn = k.clone();
        kn[0] = f64::INFINITY;
        assert!(run(&q, &kn, 4, 1.0, 0.1, (1, 2), &b).is_err());
    }

    #[test]
    fn full_rank_identity_epsilon_zero_equals_dense() {
        let mut rng = Rng(3);
        let d = 8;
        let keys = sample(&mut rng, None, 40, d, 0.0, 1.0);
        let q = sample(&mut rng, None, 1, d, 0.0, 1.0);
        let mut id = vec![0.0; d * d];
        for i in 0..d {
            id[i * d + i] = 1.0;
        }
        let (res, cost) = run(&q, &keys, d, 0.4, 0.0, (2, d), &id).unwrap();
        assert!(l1(&res.probabilities, &oracle(&q, &keys, d, 0.4)) < 1e-12);
        assert_eq!(cost.rows_completed, 40);
        assert_eq!(cost.completion_macs, 40 * d);
    }

    #[test]
    fn equal_ranks_match_single_stage_cost_class() {
        let mut rng = Rng(9);
        let (n, d, r) = (96, 16, 4);
        let hid = random_orthogonal(&mut rng, d);
        let keys = sample(&mut rng, Some(&hid), n, d, 0.3, 1.0);
        let q = sample(&mut rng, Some(&hid), 1, d, 0.3, 1.0);
        let basis = reduced_rank_basis(&keys, d, r).unwrap();
        let (a, ca) = run(&q, &keys, d, 0.5, 1e-2, (r, r), &basis).unwrap();
        let (b, cb) =
            certified_reduced_rank_softmax_accounted(&q, &keys, d, 0.5, 1e-2, r, &basis).unwrap();
        assert_eq!(ca.refinements, 0);
        assert_eq!(ca.rows_completed, cb.rows_completed);
        assert!(l1(&a.probabilities, &b.probabilities) < 1e-12);
    }

    #[test]
    fn garbage_bases_never_violate() {
        let mut rng = Rng(21);
        let d = 12;
        let iters = if cfg!(miri) { 4 } else { 250 };
        for t in 0..iters {
            let n = 4 + (t % 40);
            let r2 = 1 + (t % d);
            let r1 = 1 + (t % r2);
            let keys = sample(&mut rng, None, n, d, 0.1, 1.0 + (t % 5) as f64);
            let q = sample(&mut rng, None, 1, d, 0.1, 1.0 + (t % 3) as f64);
            let mag = [1e-3, 1.0, 50.0][t % 3];
            let mut basis: Vec<f64> = (0..r2 * d).map(|_| rng.gauss() * mag).collect();
            if t % 7 == 0 {
                basis.fill(0.0);
            }
            if t % 11 == 0 {
                basis = reduced_rank_basis(&sample(&mut rng, None, n, d, 0.5, 1.0), d, r2).unwrap();
            }
            for eps in [0.0, 1e-4, 1e-2, 0.3] {
                let scale = 0.1 + (t % 4) as f64 * 0.3;
                let (res, _) = run(&q, &keys, d, scale, eps, (r1, r2), &basis).unwrap();
                let err = l1(&res.probabilities, &oracle(&q, &keys, d, scale));
                assert!(
                    err <= res.certified_l1_bound + 1e-12,
                    "t={t} eps={eps} err={err} cert={}",
                    res.certified_l1_bound
                );
                assert!(res.certified_l1_bound <= eps + 1e-15);
            }
        }
    }

    #[test]
    fn adversarial_draws_never_violate_and_costs_are_consistent() {
        let mut rng = Rng(77);
        let d = 16;
        let iters = if cfg!(miri) { 3 } else { 150 };
        for t in 0..iters {
            let n = 8 + (t % 60);
            let hid = random_orthogonal(&mut rng, d);
            let mut keys = sample(&mut rng, Some(&hid), n, d, 0.3, 1.0);
            let mut q = sample(&mut rng, Some(&hid), 1, d, 0.3, 2.0);
            if t % 5 == 0 {
                for j in 0..d {
                    keys[(n / 2) * d + j] *= 1000.0;
                }
            }
            if t % 6 == 0 {
                let first: Vec<f64> = keys[..d].to_vec();
                keys[d..2 * d].copy_from_slice(&first);
            }
            if t % 9 == 0 {
                for v in &mut q {
                    *v *= 1e-6;
                }
            }
            let r2 = 2 + (t % 7);
            let r1 = 1 + (t % r2);
            let basis = reduced_rank_basis(&keys, d, r2).unwrap();
            for eps in [0.0, 1e-4, 1e-2, 0.3] {
                let (res, c) = run(&q, &keys, d, 0.3, eps, (r1, r2), &basis).unwrap();
                let err = l1(&res.probabilities, &oracle(&q, &keys, d, 0.3));
                assert!(
                    err <= res.certified_l1_bound + 1e-12,
                    "t={t} eps={eps} err={err}"
                );
                assert_eq!(c.rows_completed + c.rows_skipped, n);
                assert_eq!(c.completion_macs, c.rows_completed * d);
                assert_eq!(c.stage1_macs, n * r1);
                assert_eq!(c.stage2_macs, c.refinements * (r2 - r1));
                assert!(
                    c.refinements <= n
                        && c.rows_completed <= c.refinements.max(if r1 == r2 { n } else { 0 })
                );
            }
        }
    }

    /// `cargo test --release -p ada-a3-certified-softmax two_stage::tests::print_results_table -- --ignored --nocapture`
    #[test]
    #[ignore = "result table generator"]
    fn print_results_table() {
        println!(
            "n decay aligned eps std | best single-rank MACs/dense | best two-stage (r1,r2) MACs/dense | dense=1"
        );
        let singles = [2_usize, 4, 8, 16, 32];
        let pairs = [(2_usize, 8_usize), (2, 16), (4, 16), (4, 32), (8, 32)];
        for &(n, d) in &[(256_usize, 64_usize), (4096, 64)] {
            for &decay in &[0.15, 0.05] {
                for &aligned in &[true, false] {
                    for &(eps, std) in &[(1e-2, 3.0), (1e-2, 1.0), (1e-4, 3.0)] {
                        let mut rng = Rng(1234);
                        let draws = if n > 1000 { 4 } else { 10 };
                        let mut ssum = [0.0_f64; 5];
                        let mut psum = [0.0_f64; 5];
                        for _ in 0..draws {
                            let hid = random_orthogonal(&mut rng, d);
                            let keys = sample(&mut rng, Some(&hid), n, d, decay, 1.0);
                            let q = if aligned {
                                sample(&mut rng, Some(&hid), 1, d, decay, 1.0)
                            } else {
                                sample(&mut rng, None, 1, d, 0.0, 1.0)
                            };
                            let sc = oracle_scale(&q, &keys, d, std);
                            let truth = oracle(&q, &keys, d, sc);
                            for (i, &r) in singles.iter().enumerate() {
                                let basis = reduced_rank_basis(&keys, d, r).unwrap();
                                let (res, c) = certified_reduced_rank_softmax_accounted(
                                    &q, &keys, d, sc, eps, r, &basis,
                                )
                                .unwrap();
                                assert!(
                                    l1(&res.probabilities, &truth)
                                        <= res.certified_l1_bound + 1e-12
                                );
                                ssum[i] += c.per_query_macs() as f64 / c.dense_macs as f64;
                            }
                            for (i, &(r1, r2)) in pairs.iter().enumerate() {
                                let basis = reduced_rank_basis(&keys, d, r2).unwrap();
                                let (res, c) =
                                    run(&q, &keys, d, sc, eps, (r1, r2), &basis).unwrap();
                                assert!(
                                    l1(&res.probabilities, &truth)
                                        <= res.certified_l1_bound + 1e-12
                                );
                                psum[i] += c.per_query_macs() as f64 / c.dense_macs as f64;
                            }
                        }
                        let bs = (0..5).min_by(|&a, &b| ssum[a].total_cmp(&ssum[b])).unwrap();
                        let bp = (0..5).min_by(|&a, &b| psum[a].total_cmp(&psum[b])).unwrap();
                        println!(
                            "{n} {decay} {aligned} {eps} {std} | r={} {:.3} | ({},{}) {:.3}",
                            singles[bs],
                            ssum[bs] / f64::from(draws),
                            pairs[bp].0,
                            pairs[bp].1,
                            psum[bp] / f64::from(draws)
                        );
                    }
                }
            }
        }
    }

    fn oracle_scale(q: &[f64], keys: &[f64], d: usize, std: f64) -> f64 {
        let sc = dense_qk_scores(q, keys, d, 1.0).unwrap();
        let m = sc.iter().sum::<f64>() / sc.len() as f64;
        let v = sc.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / sc.len() as f64;
        std / v.sqrt().max(1e-12)
    }
}
