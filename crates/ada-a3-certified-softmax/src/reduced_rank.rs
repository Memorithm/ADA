//! Reduced-rank projected screening with certified exact completion
//! (ADA-A3, research reference).
//!
//! Conjecture: the full PCA rotation of [`crate::certified_pca_rotated_softmax_accounted`]
//! pays `d²` MACs per query, which exceeded its saving. Only the leading `r`
//! rows `P` (`r×d`) are needed to screen. Prefill stores, per key, the
//! coefficients `c_k = P k` (`r` values) and the residual norm
//! `ρ_k = ‖k − Pᵀc_k‖`. Per query: `q_r = P q` (`r·d` MACs) and the residual
//! query norm `‖u‖`, `u = q − Pᵀq_r`, from the prefill Gram `G = PPᵀ` (`r²` MACs,
//! no second `d`-sized pass).
//!
//! Exact identity for ANY `P` (no orthonormality assumed):
//! `⟨q,k⟩ = ⟨q_r,c_k⟩ + ⟨u,k⟩` because `⟨Pᵀq_r,k⟩ = ⟨q_r,Pk⟩`. Writing
//! `k = Pᵀc_k + (k − Pᵀc_k)` gives `⟨u,k⟩ = ⟨u,k−Pᵀc_k⟩ + ⟨Pu,c_k⟩` with
//! `Pu = (I − G)q_r`, hence
//! `|⟨q,k⟩ − ⟨q_r,c_k⟩| ≤ ‖u‖ρ_k + ν‖q_r‖‖c_k‖`, `ν = ‖I − G‖_F`.
//! `ν = 0` for orthonormal `P`; a garbage `P` only loosens the bound.
//! Screening costs `n·r` MACs; keys are completed in decreasing upper-bound
//! order by the exact `d`-MAC dot (no reuse of `c_k`: the exact score is
//! computed from `q` and `k` directly, which keeps exactness independent of `P`).
//! Stop when `2T/(S+T) ≤ ε` with `T = Σ_{unread} e^{h_i}`.

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

const MAX_DIM: usize = 512;
const INFL: f64 = 1.0e-9;
/// Relative slack covering f64 rounding of the dot products and norms.
const DOT_SLACK: f64 = 1.0e-11;
/// Relative (to `‖q‖²`) additive slack inside the cancellation-prone `‖u‖²`.
const U2_SLACK: f64 = 1.0e-12;

/// Logical (not hardware) costs of one reduced-rank query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReducedRankCost {
    /// Rank `r`.
    pub rank: usize,
    /// `q_r = P q` (`r·d` MACs).
    pub query_projection_macs: usize,
    /// `‖u‖²` from the Gram matrix (`r²` MACs).
    pub residual_norm_macs: usize,
    /// Screening dots `⟨q_r,c_k⟩` (`n·r` MACs).
    pub screen_macs: usize,
    /// Exact completion (`rows_completed·d` MACs).
    pub completion_macs: usize,
    /// Keys completed exactly.
    pub rows_completed: usize,
    /// Keys never completed.
    pub rows_skipped: usize,
    /// One-time prefill MACs (`c_k`, residual vectors, Gram): `2·n·r·d + r²·d`.
    pub prefill_macs: usize,
    /// Dense reference MACs (`n·d`).
    pub dense_macs: usize,
}

impl ReducedRankCost {
    /// Per-query MACs excluding amortised prefill.
    #[must_use]
    pub fn per_query_macs(&self) -> usize {
        self.query_projection_macs
            + self.residual_norm_macs
            + self.screen_macs
            + self.completion_macs
    }
}

/// Reduced-rank certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedRankSoftmax {
    /// Renormalised probabilities; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when every key was completed).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was completed exactly.
    pub read: Vec<bool>,
}

/// Leading-`rank` PCA rows (`rank×d`, row-major) of the key Gram matrix.
///
/// # Errors
///
/// Layout/finite errors, or `rank` outside `1..=d`.
pub fn reduced_rank_basis(
    keys: &[f64],
    head_dim: usize,
    rank: usize,
) -> Result<Vec<f64>, &'static str> {
    if rank == 0 || rank > head_dim {
        return Err("ADA-A3 reduced-rank rank must be in 1..=head_dim");
    }
    let full = crate::pca_rotation(keys, head_dim)?;
    Ok(full[..rank * head_dim].to_vec())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Certified softmax with reduced-rank screening.
///
/// `basis` is a row-major `rank×head_dim` matrix (see [`reduced_rank_basis`]);
/// any finite matrix is accepted and only affects tightness.
///
/// # Errors
///
/// Inconsistent layout, non-finite input, invalid `score_scale`/`epsilon`/rank,
/// or non-finite scores.
#[must_use = "the certified distribution should be checked"]
pub fn certified_reduced_rank_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    rank: usize,
    basis: &[f64],
) -> Result<(ReducedRankSoftmax, ReducedRankCost), &'static str> {
    let d = head_dim;
    if d == 0 || d > MAX_DIM {
        return Err("ADA-A3 reduced-rank head_dim must be in 1..=512");
    }
    if rank == 0 || rank > d {
        return Err("ADA-A3 reduced-rank rank must be in 1..=head_dim");
    }
    if basis.len() != rank * d || basis.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 reduced-rank basis must be finite row-major rank×d");
    }
    if query.len() != d || keys.is_empty() || keys.len() % d != 0 {
        return Err("ADA-A3 reduced-rank query/keys layout mismatch");
    }
    if query.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err("ADA-A3 reduced-rank inputs must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 reduced-rank score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 reduced-rank epsilon must be finite and non-negative");
    }
    let n = keys.len() / d;
    let r = rank;

    // ---- prefill (conceptually stored; counted separately) ----
    let mut gram = vec![0.0_f64; r * r];
    for a in 0..r {
        for b in 0..r {
            gram[a * r + b] = dot(&basis[a * d..(a + 1) * d], &basis[b * d..(b + 1) * d]);
        }
    }
    let nu = {
        let mut acc = 0.0_f64;
        for a in 0..r {
            for b in 0..r {
                let e = gram[a * r + b] - f64::from(u8::from(a == b));
                acc += e * e;
            }
        }
        acc.sqrt()
    };
    let mut coeff = vec![0.0_f64; n * r];
    let mut rho = vec![0.0_f64; n];
    let mut knorm = vec![0.0_f64; n];
    for (i, row) in keys.chunks_exact(d).enumerate() {
        for a in 0..r {
            coeff[i * r + a] = dot(&basis[a * d..(a + 1) * d], row);
        }
        let mut res = row.to_vec();
        for a in 0..r {
            let c = coeff[i * r + a];
            for j in 0..d {
                res[j] -= c * basis[a * d + j];
            }
        }
        knorm[i] = dot(row, row).sqrt();
        rho[i] = dot(&res, &res).sqrt() * (1.0 + INFL) + DOT_SLACK * knorm[i];
    }

    // ---- query ----
    let qr: Vec<f64> = (0..r)
        .map(|a| dot(&basis[a * d..(a + 1) * d], query))
        .collect();
    let q2 = dot(query, query);
    let qr2 = dot(&qr, &qr);
    // ‖u‖² = ‖q‖² − 2‖q_r‖² + q_rᵀ G q_r.
    let mut qgq = 0.0_f64;
    for a in 0..r {
        qgq += qr[a] * dot(&gram[a * r..(a + 1) * r], &qr);
    }
    let u2 = (q2 - 2.0 * qr2 + qgq).max(0.0) + U2_SLACK * (q2 + qgq.abs());
    let u_norm = u2.sqrt();
    let qr_norm = qr2.sqrt();
    let q_norm = q2.sqrt();

    let mut screen = vec![0.0_f64; n];
    let mut hi = vec![0.0_f64; n];
    for i in 0..n {
        let c = &coeff[i * r..(i + 1) * r];
        let p = dot(&qr, c) * score_scale;
        let cn = dot(c, c).sqrt();
        let delta = score_scale
            * ((u_norm * rho[i] + nu * (1.0 + INFL) * qr_norm * cn) * (1.0 + INFL)
                + DOT_SLACK * q_norm * knorm[i]);
        let h = p + delta;
        if !h.is_finite() {
            return Err("ADA-A3 reduced-rank scores must be finite");
        }
        screen[i] = p;
        hi[i] = h;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| hi[b].total_cmp(&hi[a]).then(a.cmp(&b)));
    let mut ln_suffix = vec![f64::NEG_INFINITY; n + 1];
    for k in (0..n).rev() {
        let (a, b) = (hi[order[k]], ln_suffix[k + 1]);
        ln_suffix[k] = if b == f64::NEG_INFINITY {
            a
        } else {
            a.max(b) + (-(a - b).abs()).exp().ln_1p()
        };
    }

    let mut cost = ReducedRankCost {
        rank: r,
        query_projection_macs: r * d,
        residual_norm_macs: r * r,
        screen_macs: n * r,
        prefill_macs: 2 * n * r * d + r * r * d,
        dense_macs: n * d,
        ..ReducedRankCost::default()
    };
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64;
    let mut certificate = 0.0_f64;
    let mut done = n;
    for (rk, &idx) in order.iter().enumerate() {
        let row = &keys[idx * d..(idx + 1) * d];
        let score = dot(query, row) * score_scale;
        if !score.is_finite() {
            return Err("ADA-A3 reduced-rank scores must be finite");
        }
        cost.rows_completed += 1;
        cost.completion_macs += d;
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
        scores.push((idx, score));
        let read_so_far = rk + 1;
        if read_so_far == n {
            break;
        }
        let ln_s = running_max + mass.ln();
        let ln_t = ln_suffix[read_so_far] + INFL.ln_1p();
        let cert = 2.0 / (1.0 + (ln_s - ln_t).exp());
        if cert <= epsilon {
            certificate = cert;
            done = read_so_far;
            break;
        }
    }
    cost.rows_skipped = n - done;

    let total: f64 = scores.iter().map(|&(_, s)| (s - running_max).exp()).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("ADA-A3 reduced-rank mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        ReducedRankSoftmax {
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
    use super::{certified_reduced_rank_softmax_accounted as run, reduced_rank_basis};
    use crate::dense_qk_scores;

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
        let b = vec![1.0; 4];
        assert!(run(&q, &k, 4, 1.0, 0.1, 0, &[]).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, 5, &b).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, 1, &[1.0; 3]).is_err());
        assert!(run(&q, &k, 4, 1.0, 0.1, 1, &[f64::NAN; 4]).is_err());
        assert!(run(&q, &k, 4, 0.0, 0.1, 1, &b).is_err());
        assert!(run(&q, &k, 4, 1.0, -1.0, 1, &b).is_err());
        assert!(run(&q[..3], &k, 4, 1.0, 0.1, 1, &b).is_err());
        assert!(run(&q, &k[..7], 4, 1.0, 0.1, 1, &b).is_err());
        let mut kn = k.clone();
        kn[0] = f64::INFINITY;
        assert!(run(&q, &kn, 4, 1.0, 0.1, 1, &b).is_err());
    }

    #[test]
    fn full_rank_identity_equals_dense() {
        let mut rng = Rng(3);
        let d = 8;
        let keys = sample(&mut rng, None, 40, d, 0.0, 1.0);
        let q = sample(&mut rng, None, 1, d, 0.0, 1.0);
        let mut id = vec![0.0; d * d];
        for i in 0..d {
            id[i * d + i] = 1.0;
        }
        let (res, cost) = run(&q, &keys, d, 0.4, 0.0, d, &id).unwrap();
        assert!(l1(&res.probabilities, &oracle(&q, &keys, d, 0.4)) < 1e-12);
        assert_eq!(cost.rows_completed, 40);
        assert_eq!(
            cost.per_query_macs(),
            cost.query_projection_macs + cost.residual_norm_macs + cost.screen_macs + 40 * d
        );
    }

    #[test]
    fn identity_screen_certifies_with_zero_residual() {
        // r = d orthonormal: residual ≈ 0, so screen scores are exact and
        // a peaked softmax must skip rows.
        let mut rng = Rng(5);
        let (n, d) = (128, 8);
        let mut keys = sample(&mut rng, None, n, d, 0.0, 0.3);
        let q: Vec<f64> = (0..d).map(|_| 4.0).collect();
        for j in 0..d {
            keys[j] = 4.0;
        }
        let basis = reduced_rank_basis(&keys, d, d).unwrap();
        let (res, cost) = run(&q, &keys, d, 1.0, 1e-2, d, &basis).unwrap();
        assert!(cost.rows_skipped > 0);
        assert!(
            l1(&res.probabilities, &oracle(&q, &keys, d, 1.0)) <= res.certified_l1_bound + 1e-12
        );
    }

    #[test]
    fn witness_wins_on_hidden_low_rank_structure() {
        let mut rng = Rng(11);
        let (n, d, r) = (1024, 64, 8);
        let hid = random_orthogonal(&mut rng, d);
        let mut keys = sample(&mut rng, Some(&hid), n, d, 0.15, 1.0);
        let q = sample(&mut rng, Some(&hid), 1, d, 0.15, 1.0);
        // Planted dominant key (3q) in the structured subspace: peaked softmax.
        for j in 0..d {
            keys[j] = 3.0 * q[j];
        }
        let sc = oracle_scale(&q, &keys, d, 3.0);
        let basis = reduced_rank_basis(&keys, d, r).unwrap();
        let (res, cost) = run(&q, &keys, d, sc, 1e-2, r, &basis).unwrap();
        assert!(
            l1(&res.probabilities, &oracle(&q, &keys, d, sc)) <= res.certified_l1_bound + 1e-12
        );
        assert!(cost.per_query_macs() < cost.dense_macs, "{cost:?}");
    }

    #[test]
    fn garbage_bases_never_violate() {
        let mut rng = Rng(21);
        let d = 12;
        let iters = if cfg!(miri) { 6 } else { 300 };
        for t in 0..iters {
            let n = 4 + (t % 40);
            let r = 1 + (t % d);
            let keys = sample(&mut rng, None, n, d, 0.1, 1.0 + (t % 5) as f64);
            let q = sample(&mut rng, None, 1, d, 0.1, 1.0 + (t % 3) as f64);
            let mag = [1e-3, 1.0, 50.0][t % 3];
            let mut basis: Vec<f64> = (0..r * d).map(|_| rng.gauss() * mag).collect();
            if t % 7 == 0 {
                basis.fill(0.0); // singular
            }
            if t % 11 == 0 {
                basis = reduced_rank_basis(&sample(&mut rng, None, n, d, 0.5, 1.0), d, r).unwrap(); // stale
            }
            for eps in [0.0, 1e-4, 1e-2, 0.3] {
                let scale = 0.1 + (t % 4) as f64 * 0.3;
                let (res, _) = run(&q, &keys, d, scale, eps, r, &basis).unwrap();
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
    fn adversarial_draws_never_violate() {
        let mut rng = Rng(77);
        let d = 16;
        let iters = if cfg!(miri) { 3 } else { 200 };
        for t in 0..iters {
            let n = 8 + (t % 60);
            let hid = random_orthogonal(&mut rng, d);
            let mut keys = sample(&mut rng, Some(&hid), n, d, 0.3, 1.0);
            let mut q = sample(&mut rng, Some(&hid), 1, d, 0.3, 2.0);
            if t % 5 == 0 {
                for j in 0..d {
                    keys[(n / 2) * d + j] *= 1000.0; // outlier
                }
            }
            if t % 6 == 0 {
                let first: Vec<f64> = keys[..d].to_vec();
                keys[d..2 * d].copy_from_slice(&first); // duplicate
            }
            if t % 9 == 0 {
                for v in &mut q {
                    *v *= 1e-6;
                } // near-zero query
            }
            let r = 1 + (t % 8);
            let basis = reduced_rank_basis(&keys, d, r).unwrap();
            for eps in [0.0, 1e-4, 1e-2, 0.3] {
                let (res, cost) = run(&q, &keys, d, 0.3, eps, r, &basis).unwrap();
                let err = l1(&res.probabilities, &oracle(&q, &keys, d, 0.3));
                assert!(
                    err <= res.certified_l1_bound + 1e-12,
                    "t={t} eps={eps} err={err}"
                );
                assert_eq!(cost.rows_completed + cost.rows_skipped, n);
                assert_eq!(cost.completion_macs, cost.rows_completed * d);
            }
        }
    }

    /// `cargo test --release -p ada-a3-certified-softmax reduced_rank::tests::print_results_table -- --ignored --nocapture`
    #[test]
    #[ignore = "result table generator"]
    fn print_results_table() {
        println!("n d decay aligned eps std | best-rank MACs/dense (rows frac) | dense=1");
        for &(n, d) in &[(256_usize, 64_usize), (4096, 64)] {
            for &decay in &[0.15, 0.05] {
                for &aligned in &[true, false] {
                    for &(eps, std) in &[(1e-2, 3.0), (1e-2, 1.0), (1e-4, 3.0)] {
                        let mut rng = Rng(1234);
                        let draws = 10;
                        let mut best_sum = [0.0_f64; 5];
                        let mut rows_sum = [0.0_f64; 5];
                        let ranks = [2_usize, 4, 8, 16, 32];
                        for _ in 0..draws {
                            let hid = random_orthogonal(&mut rng, d);
                            let keys = sample(&mut rng, Some(&hid), n, d, decay, 1.0);
                            let q = if aligned {
                                sample(&mut rng, Some(&hid), 1, d, decay, 1.0)
                            } else {
                                sample(&mut rng, None, 1, d, 0.0, 1.0)
                            };
                            // scale so logit std ≈ std
                            let sc = oracle_scale(&q, &keys, d, std);
                            for (ri, &r) in ranks.iter().enumerate() {
                                let basis = reduced_rank_basis(&keys, d, r).unwrap();
                                let (res, c) = run(&q, &keys, d, sc, eps, r, &basis).unwrap();
                                assert!(
                                    l1(&res.probabilities, &oracle(&q, &keys, d, sc))
                                        <= res.certified_l1_bound + 1e-12
                                );
                                best_sum[ri] += c.per_query_macs() as f64 / c.dense_macs as f64;
                                rows_sum[ri] += c.rows_completed as f64 / n as f64;
                            }
                        }
                        let (bi, _) = best_sum
                            .iter()
                            .enumerate()
                            .min_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap();
                        println!(
                            "{n} {d} {decay} {aligned} {eps} {std} | r={} {:.3} ({:.3})",
                            ranks[bi],
                            best_sum[bi] / f64::from(draws),
                            rows_sum[bi] / f64::from(draws)
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
