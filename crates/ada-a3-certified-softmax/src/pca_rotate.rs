//! PCA-rotated split-dimension certified softmax (ADA-A3, research reference).
//!
//! Conjecture: [`crate::certified_split_dim_softmax`] only wins when the
//! coordinate axes already concentrate key energy in a prefix. An orthogonal
//! rotation `R` (rows = eigenvectors of the key Gram matrix `KᵀK`, descending
//! eigenvalue) leaves every score unchanged, `⟨q,k⟩ = ⟨Rq,Rk⟩`, but moves the
//! energy into the leading coordinates for *any* rotated anisotropy. The
//! rotated keys are stored at prefill; per query `Rq` costs `d²` MACs.
//!
//! Soundness: the caller-supplied `R` is never trusted. Its departure from
//! orthonormality `ν = ‖RRᵀ − I‖_F` is measured and `|⟨Rq,Rk⟩ − ⟨q,k⟩| ≤ ν‖q‖‖k‖`
//! (plus rounding) is added to the per-key slack, so any `R` (even a random
//! non-orthogonal one) yields a valid certificate, only a looser one.
//! Rotation is data-dependent but score-preserving: it changes the schedule,
//! not the semantics.

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use crate::split_dim::{SplitDimCost, SplitDimSoftmax, split_dim_core};

const MAX_ROT_DIM: usize = 512;

/// Logical (not hardware) costs of one PCA-rotated query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PcaRotateCost {
    /// Split-dimension costs in the rotated basis.
    pub split: SplitDimCost,
    /// Per-query rotation of `q` (`d²` MACs).
    pub query_rotation_macs: usize,
    /// One-time prefill rotation of the keys (`n·d²` MACs), amortised over queries.
    pub prefill_rotation_macs: usize,
}

impl PcaRotateCost {
    /// Per-query MACs excluding the amortised prefill.
    #[must_use]
    pub fn per_query_macs(&self) -> usize {
        self.split.total_macs() + self.query_rotation_macs
    }
}

/// Deterministic PCA rotation (cyclic Jacobi on the uncentred Gram matrix).
/// Returns a row-major `d×d` matrix whose rows are eigenvectors sorted by
/// descending eigenvalue (ties by index).
///
/// # Errors
///
/// Returns an error for a bad layout, non-finite input or `d` outside `1..=512`.
pub fn pca_rotation(keys: &[f64], head_dim: usize) -> Result<Vec<f64>, &'static str> {
    let d = head_dim;
    if d == 0 || d > MAX_ROT_DIM {
        return Err("ADA-A3 pca head_dim must be in 1..=512");
    }
    if keys.is_empty() || keys.len() % d != 0 {
        return Err("ADA-A3 pca keys must be non-empty row-major [key_count, head_dim]");
    }
    if keys.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 pca keys must be finite");
    }
    let mut g = vec![0.0_f64; d * d];
    for row in keys.chunks_exact(d) {
        for a in 0..d {
            for b in a..d {
                g[a * d + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..d {
        for b in 0..a {
            g[a * d + b] = g[b * d + a];
        }
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 pca Gram matrix overflow");
    }
    // v accumulates eigenvectors as columns (identity start).
    let mut v = vec![0.0_f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..60 {
        let off: f64 = (0..d)
            .flat_map(|a| (0..d).filter(move |&b| b != a).map(move |b| (a, b)))
            .map(|(a, b)| g[a * d + b] * g[a * d + b])
            .sum();
        let diag: f64 = (0..d).map(|a| g[a * d + a] * g[a * d + a]).sum();
        if off <= 1.0e-30 * diag.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = g[p * d + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (g[q * d + q] - g[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let (gkp, gkq) = (g[k * d + p], g[k * d + q]);
                    g[k * d + p] = c * gkp - s * gkq;
                    g[k * d + q] = s * gkp + c * gkq;
                }
                for k in 0..d {
                    let (gpk, gqk) = (g[p * d + k], g[q * d + k]);
                    g[p * d + k] = c * gpk - s * gqk;
                    g[q * d + k] = s * gpk + c * gqk;
                }
                for k in 0..d {
                    let (vkp, vkq) = (v[k * d + p], v[k * d + q]);
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| g[b * d + b].total_cmp(&g[a * d + a]).then(a.cmp(&b)));
    let mut rot = vec![0.0_f64; d * d];
    for (r, &col) in order.iter().enumerate() {
        for k in 0..d {
            rot[r * d + k] = v[k * d + col];
        }
    }
    Ok(rot)
}

fn rotate_row(rot: &[f64], row: &[f64], d: usize) -> Vec<f64> {
    (0..d)
        .map(|r| {
            rot[r * d..(r + 1) * d]
                .iter()
                .zip(row)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// Frobenius departure `‖RRᵀ − I‖_F` of a row-major `d×d` matrix.
fn orthonormality_defect(rot: &[f64], d: usize) -> f64 {
    let mut acc = 0.0_f64;
    for a in 0..d {
        for b in 0..d {
            let dot: f64 = rot[a * d..(a + 1) * d]
                .iter()
                .zip(&rot[b * d..(b + 1) * d])
                .map(|(x, y)| x * y)
                .sum();
            let e = dot - f64::from(u8::from(a == b));
            acc += e * e;
        }
    }
    acc.sqrt()
}

/// Certified softmax on PCA-rotated keys with split-dimension completion.
///
/// `rotation` is a row-major `d×d` matrix (see [`pca_rotation`]); any matrix
/// is accepted and its non-orthonormality is folded into the certificate.
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, invalid
/// `split`, `score_scale`, `epsilon`, a non-finite rotation, or non-finite scores.
#[allow(clippy::too_many_arguments)]
#[must_use = "the certified distribution should be checked"]
pub fn certified_pca_rotated_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    split: usize,
    rotation: &[f64],
) -> Result<(SplitDimSoftmax, PcaRotateCost), &'static str> {
    let d = head_dim;
    if d == 0 || d > MAX_ROT_DIM {
        return Err("ADA-A3 pca head_dim must be in 1..=512");
    }
    if rotation.len() != d * d || rotation.iter().any(|v| !v.is_finite()) {
        return Err("ADA-A3 pca rotation must be finite row-major d×d");
    }
    if query.len() != d || keys.is_empty() || keys.len() % d != 0 {
        return Err("ADA-A3 pca query/keys layout mismatch");
    }
    if query.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err("ADA-A3 pca inputs must be finite");
    }
    let nu = orthonormality_defect(rotation, d);
    // |⟨Rq,Rk⟩−⟨q,k⟩| = |qᵀ(RᵀR−I)k| ≤ ν‖q‖‖k‖ (RᵀR−I and RRᵀ−I share a spectrum,
    // so the Frobenius defect ν bounds the spectral norm), plus rotation rounding
    // γ‖q‖‖k‖ with γ = 8 d^{3/2} ε_mach (1+ν)².
    let gamma = 8.0 * (d as f64).powf(1.5) * f64::EPSILON * (1.0 + nu) * (1.0 + nu);
    let rel = (nu * (1.0 + 1.0e-9) + gamma) * 2.0;
    let q_norm = query.iter().map(|v| v * v).sum::<f64>().sqrt();
    let knorms: Vec<f64> = keys
        .chunks_exact(d)
        .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect();
    let max_k = knorms.iter().copied().fold(0.0_f64, f64::max);
    // Perturbation of every score by ≤ η changes the softmax by ≤ 2(e^{2η}−1) in L1;
    // it is charged to the certificate (also for keys that are read exactly).
    let eta = score_scale * rel * q_norm * max_k;
    let extra_l1 = 2.0 * (2.0 * eta).exp_m1();
    if !extra_l1.is_finite() {
        return Err("ADA-A3 pca rotation defect too large to certify");
    }
    let rq = rotate_row(rotation, query, d);
    let rk: Vec<f64> = keys
        .chunks_exact(d)
        .flat_map(|row| rotate_row(rotation, row, d))
        .collect();
    let n = keys.len() / d;
    let eps_eff = (epsilon - extra_l1).max(0.0);
    let (mut res, split_cost) = split_dim_core(
        &rq,
        &rk,
        d,
        score_scale,
        eps_eff,
        split,
        2.0e-11,
        Some((q_norm, &knorms, rel)),
    )?;
    res.certified_l1_bound += extra_l1;
    Ok((
        res,
        PcaRotateCost {
            split: split_cost,
            query_rotation_macs: d * d,
            prefill_rotation_macs: n * d * d,
        },
    ))
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::{certified_pca_rotated_softmax_accounted, orthonormality_defect, pca_rotation};
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

    /// Independent oracle: dense stable softmax on the ORIGINAL (unrotated) data.
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

    /// Random orthogonal matrix by Gram–Schmidt of Gaussian rows.
    fn random_orthogonal(rng: &mut Rng, d: usize) -> Vec<f64> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        while rows.len() < d {
            let mut v: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
            for _ in 0..2 {
                for r in &rows {
                    let dot: f64 = v.iter().zip(r).map(|(a, b)| a * b).sum();
                    for (x, y) in v.iter_mut().zip(r) {
                        *x -= dot * y;
                    }
                }
            }
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n > 1e-6 {
                rows.push(v.into_iter().map(|x| x / n).collect());
            }
        }
        rows.concat()
    }

    fn rot_vec(r: &[f64], v: &[f64], d: usize) -> Vec<f64> {
        (0..d)
            .map(|i| (0..d).map(|j| r[i * d + j] * v[j]).sum())
            .collect()
    }

    #[test]
    fn pca_rotation_is_orthonormal_and_sorted() {
        let mut rng = Rng(7);
        let d = 12;
        let keys: Vec<f64> = (0..50 * d)
            .map(|i| rng.gauss() * (-(0.3) * (i % d) as f64).exp())
            .collect();
        let r = pca_rotation(&keys, d).unwrap();
        assert!(orthonormality_defect(&r, d) < 1e-12);
        // eigenvalue (energy along row) must be non-increasing
        let energy: Vec<f64> = (0..d)
            .map(|i| {
                keys.chunks_exact(d)
                    .map(|k| {
                        let p: f64 = (0..d).map(|j| r[i * d + j] * k[j]).sum();
                        p * p
                    })
                    .sum()
            })
            .collect();
        for w in energy.windows(2) {
            assert!(w[0] + 1e-9 >= w[1]);
        }
    }

    #[test]
    fn rejects_invalid_input() {
        let k = [1.0, 2.0];
        let id = [1.0, 0.0, 0.0, 1.0];
        assert!(
            certified_pca_rotated_softmax_accounted(&[1.0, 0.0], &k, 2, 1.0, 0.1, 1, &id).is_ok()
        );
        assert!(
            certified_pca_rotated_softmax_accounted(&[1.0, 0.0], &k, 2, 1.0, 0.1, 0, &id).is_err()
        );
        assert!(
            certified_pca_rotated_softmax_accounted(&[1.0, 0.0], &k, 2, 1.0, 0.1, 1, &[1.0])
                .is_err()
        );
        assert!(
            certified_pca_rotated_softmax_accounted(
                &[1.0, 0.0],
                &k,
                2,
                1.0,
                0.1,
                1,
                &[f64::NAN; 4]
            )
            .is_err()
        );
        assert!(pca_rotation(&[], 2).is_err());
        assert!(pca_rotation(&[f64::NAN, 0.0], 2).is_err());
        assert!(pca_rotation(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(pca_rotation(&k, 0).is_err());
    }

    #[test]
    fn identity_rotation_equals_split_dim() {
        let mut rng = Rng(11);
        let d = 8;
        let keys: Vec<f64> = (0..40 * d).map(|_| rng.gauss()).collect();
        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let mut id = vec![0.0; d * d];
        for i in 0..d {
            id[i * d + i] = 1.0;
        }
        let (a, _) =
            certified_pca_rotated_softmax_accounted(&q, &keys, d, 0.3, 1e-2, 3, &id).unwrap();
        let b = crate::certified_split_dim_softmax(&q, &keys, d, 0.3, 1e-2, 3).unwrap();
        assert_eq!(a.read, b.read);
    }

    #[test]
    fn hidden_rotation_witness_beats_axis_split() {
        // Energy sits in 2 hidden directions; axes look isotropic.
        let (n, d) = (128, 16);
        let mut rng = Rng(99);
        let basis = random_orthogonal(&mut rng, d);
        let sd: Vec<f64> = (0..d).map(|j| if j < 2 { 3.0 } else { 0.02 }).collect();
        let mut keys = Vec::new();
        for _ in 0..n {
            let z: Vec<f64> = (0..d).map(|j| rng.gauss() * sd[j]).collect();
            for i in 0..d {
                keys.push((0..d).map(|j| basis[j * d + i] * z[j]).sum::<f64>());
            }
        }
        let zq: Vec<f64> = (0..d).map(|j| rng.gauss() * sd[j]).collect();
        let q: Vec<f64> = (0..d)
            .map(|i| (0..d).map(|j| basis[j * d + i] * zq[j]).sum::<f64>())
            .collect();
        let rot = pca_rotation(&keys, d).unwrap();
        let scale = 0.5;
        let (r, c) =
            certified_pca_rotated_softmax_accounted(&q, &keys, d, scale, 1e-2, 2, &rot).unwrap();
        let (_, axis) =
            crate::certified_split_dim_softmax_accounted(&q, &keys, d, scale, 1e-2, 2).unwrap();
        assert!(l1(&r.probabilities, &oracle(&q, &keys, d, scale)) <= r.certified_l1_bound + 1e-12);
        assert!(c.split.total_macs() < axis.total_macs());
    }

    #[test]
    fn garbage_and_degenerate_rotations_never_violate() {
        // Non-orthogonal, singular, huge, or random rotations: only looseness allowed.
        let mut rng = Rng(5);
        let iters = if cfg!(miri) { 5 } else { 300 };
        for it in 0..iters {
            let d = 2 + (rng.next() * 10.0) as usize;
            let n = 1 + (rng.next() * 30.0) as usize;
            let keys: Vec<f64> = (0..n * d).map(|_| rng.gauss() * 2.0).collect();
            let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
            let rot: Vec<f64> = match it % 4 {
                0 => (0..d * d).map(|_| rng.gauss()).collect(), // non-orthogonal
                1 => vec![0.0; d * d],                          // singular
                2 => (0..d * d).map(|_| rng.gauss() * 1e3).collect(), // huge
                _ => pca_rotation(&keys, d).unwrap(),
            };
            let m = 1 + (rng.next() * d as f64) as usize % d;
            for eps in [0.0, 1e-3, 0.2] {
                let Ok((r, _)) =
                    certified_pca_rotated_softmax_accounted(&q, &keys, d, 0.4, eps, m, &rot)
                else {
                    continue;
                };
                let err = l1(&r.probabilities, &oracle(&q, &keys, d, 0.4));
                assert!(err <= r.certified_l1_bound + 1e-12, "viol {err} it {it}");
            }
        }
    }

    #[test]
    fn adversarial_rotated_outliers_and_duplicates() {
        let mut rng = Rng(0xBEEF);
        let iters = if cfg!(miri) { 5 } else { 200 };
        for _ in 0..iters {
            let d = 2 + (rng.next() * 12.0) as usize;
            let n = 2 + (rng.next() * 40.0) as usize;
            let mag = [1.0, 10.0, 1000.0][(rng.next() * 3.0) as usize % 3];
            let mut keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
            for i in 0..n {
                if rng.next() < 0.3 {
                    keys[i * d] *= mag;
                }
                if i > 0 && rng.next() < 0.2 {
                    let (a, b) = keys.split_at_mut(i * d);
                    b[..d].copy_from_slice(&a[(i - 1) * d..i * d]);
                }
            }
            let q: Vec<f64> = (0..d)
                .map(|_| rng.gauss() * if rng.next() < 0.3 { mag } else { 1.0 })
                .collect();
            // rotation fit on a *different* key set (stale PCA) must stay sound
            let stale: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
            for rot in [
                pca_rotation(&keys, d).unwrap(),
                pca_rotation(&stale, d).unwrap(),
            ] {
                let m = 1 + (rng.next() * d as f64) as usize % d;
                for eps in [0.0, 1e-4, 1e-2, 0.3] {
                    let scale = 0.05 + rng.next();
                    let Ok((r, _)) =
                        certified_pca_rotated_softmax_accounted(&q, &keys, d, scale, eps, m, &rot)
                    else {
                        continue;
                    };
                    let Ok(sc) = dense_qk_scores(&q, &keys, d, scale) else {
                        continue;
                    };
                    if sc.iter().any(|s| !s.is_finite()) {
                        continue;
                    }
                    let err = l1(&r.probabilities, &oracle(&q, &keys, d, scale));
                    assert!(err <= r.certified_l1_bound + 1e-12, "viol {err}");
                }
            }
        }
    }

    #[test]
    fn rotation_preserves_scores() {
        let mut rng = Rng(3);
        let d = 10;
        let r = random_orthogonal(&mut rng, d);
        let a: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let b: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let (ra, rb) = (rot_vec(&r, &a, d), rot_vec(&r, &b, d));
        let x: f64 = a.iter().zip(&b).map(|(p, q)| p * q).sum();
        let y: f64 = ra.iter().zip(&rb).map(|(p, q)| p * q).sum();
        assert!((x - y).abs() < 1e-12);
    }

    /// Prints the results table of the doc (`--ignored --nocapture`).
    #[test]
    #[ignore = "result table generator"]
    fn print_results_table() {
        let (n, d) = (256, 64);
        for decay in [0.0_f64, 0.05, 0.15] {
            for rotated in [false, true] {
                for eps in [1e-2, 1e-4] {
                    for std in [1.0_f64, 3.0] {
                        let (mut axis, mut pca, mut rows_a, mut rows_p) = (0.0, 0.0, 0.0, 0.0);
                        let draws = 20;
                        for s in 0..draws {
                            let mut rng = Rng(2000 + s);
                            let basis = if rotated {
                                random_orthogonal(&mut rng, d)
                            } else {
                                let mut id = vec![0.0; d * d];
                                for i in 0..d {
                                    id[i * d + i] = 1.0;
                                }
                                id
                            };
                            let sd: Vec<f64> = (0..d).map(|j| (-decay * j as f64).exp()).collect();
                            let sample = |rng: &mut Rng| -> Vec<f64> {
                                let z: Vec<f64> = (0..d).map(|j| rng.gauss() * sd[j]).collect();
                                (0..d)
                                    .map(|i| (0..d).map(|j| basis[j * d + i] * z[j]).sum::<f64>())
                                    .collect()
                            };
                            let keys: Vec<f64> = (0..n).flat_map(|_| sample(&mut rng)).collect();
                            let q = sample(&mut rng);
                            let lv: f64 = sd.iter().map(|x| x * x).sum();
                            let scale = std / lv;
                            let rot = pca_rotation(&keys, d).unwrap();
                            // best split over a fixed menu (oracle-free, per-regime)
                            let best = |f: &dyn Fn(usize) -> (usize, usize)| {
                                [4_usize, 8, 16, 32]
                                    .iter()
                                    .map(|&m| f(m))
                                    .min_by_key(|x| x.0)
                                    .unwrap()
                            };
                            let a = best(&|m| {
                                let (r, c) = crate::certified_split_dim_softmax_accounted(
                                    &q, &keys, d, scale, eps, m,
                                )
                                .unwrap();
                                assert!(
                                    l1(&r.probabilities, &oracle(&q, &keys, d, scale))
                                        <= r.certified_l1_bound + 1e-12
                                );
                                (c.total_macs(), c.rows_completed)
                            });
                            let p = best(&|m| {
                                let (r, c) = certified_pca_rotated_softmax_accounted(
                                    &q, &keys, d, scale, eps, m, &rot,
                                )
                                .unwrap();
                                assert!(
                                    l1(&r.probabilities, &oracle(&q, &keys, d, scale))
                                        <= r.certified_l1_bound + 1e-12
                                );
                                (c.per_query_macs(), c.split.rows_completed)
                            });
                            axis += a.0 as f64 / (n * d) as f64;
                            pca += p.0 as f64 / (n * d) as f64;
                            rows_a += a.1 as f64 / n as f64;
                            rows_p += p.1 as f64 / n as f64;
                        }
                        let k = draws as f64;
                        println!(
                            "decay {decay} hidden_rot {rotated} eps {eps} std {std}: axis {:.4} (rows {:.3}) pca+rot {:.4} (rows {:.3})",
                            axis / k,
                            rows_a / k,
                            pca / k,
                            rows_p / k
                        );
                    }
                }
            }
        }
    }
}
