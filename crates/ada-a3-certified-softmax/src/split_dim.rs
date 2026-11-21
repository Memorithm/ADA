//! Split-dimension prefix screening with certified exact completion (ADA-A3).
//!
//! Research reference only. The head dimension is split at `m`. Prefill stores
//! one scalar per key, the tail norm `r_i = ‖k_i[m..]‖₂`. At query time the
//! prefix dot `p_i = s⟨q[..m], k_i[..m]⟩` is computed for every key
//! (`n·m` MACs) and, by Cauchy–Schwarz on the tail,
//! `|z_i − p_i| ≤ s‖q[m..]‖ r_i = δ_i`, so `z_i ≤ h_i = p_i + δ_i`.
//! Keys are completed exactly in decreasing `h_i` order, *reusing* `p_i` and
//! reading only the remaining `d − m` coordinates (no re-read, unlike a
//! separate low-precision screen). With `T ≤ Σ_{unread} e^{h_i}` (exact
//! suffix sum) the truncated, renormalised softmax satisfies
//! `‖p − q‖₁ ≤ 2T/(S+T)`; the scan stops at the first rank whose certificate
//! is `≤ ε`. Total MACs are `n·m + read·(d−m) ≤ n·d`. Log domain throughout.

const DELTA_INFLATION: f64 = 1.0e-9;
const MASS_INFLATION: f64 = 1.0e-9;
const MAX_HEAD_DIM: usize = 4096;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SplitDimCost {
    /// Tail norms computed at prefill (`n`).
    pub prefill_tail_norms: usize,
    /// Split point `m`.
    pub split: usize,
    /// Prefix multiply-accumulates (`n·m`).
    pub prefix_macs: usize,
    /// Completion multiply-accumulates (`completed_rows·(d−m)`).
    pub completion_macs: usize,
    /// Keys completed exactly.
    pub rows_completed: usize,
    /// Keys never completed.
    pub rows_skipped: usize,
    /// Comparison-class work: sort of `n` upper bounds.
    pub sorted_keys: usize,
    /// `exp`/`ln` evaluations.
    pub transcendental_evaluations: usize,
}

impl SplitDimCost {
    /// Total MACs of the query.
    #[must_use]
    pub fn total_macs(&self) -> usize {
        self.prefix_macs + self.completion_macs
    }
}

/// Split-dimension certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitDimSoftmax {
    /// Renormalised probabilities in input order; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when all keys were completed).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was completed exactly.
    pub read: Vec<bool>,
    /// Positive scale applied to every score.
    pub score_scale: f64,
}

/// Evaluate split-dimension certified softmax with costs.
///
/// `split` must be in `1..=head_dim` (`split == head_dim` degenerates to
/// dense with a zero tail).
///
/// # Errors
///
/// Returns an error for an inconsistent layout, non-finite input, invalid
/// `split`, `score_scale` or `epsilon`, or a non-finite score.
#[must_use = "the certified distribution should be checked"]
pub fn certified_split_dim_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    split: usize,
) -> Result<(SplitDimSoftmax, SplitDimCost), &'static str> {
    split_dim_core(
        query,
        keys,
        head_dim,
        score_scale,
        epsilon,
        split,
        1.0e-11,
        None,
    )
}

/// Shared implementation; `dot_slack` is the relative (to `‖q‖‖k‖`) slack
/// covering f64 dot rounding. `original_norms = (‖q‖, ‖k_i‖ per key, rel)` adds
/// `s·rel·‖q‖‖k_i‖` per key for callers whose scored vectors are a transform
/// of the originals (the transform error is relative to the ORIGINAL norms).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn split_dim_core(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    split: usize,
    dot_slack: f64,
    original_norms: Option<(f64, &[f64], f64)>,
) -> Result<(SplitDimSoftmax, SplitDimCost), &'static str> {
    if head_dim == 0 || head_dim > MAX_HEAD_DIM {
        return Err("ADA-A3 split-dim head_dim must be in 1..=4096");
    }
    if split == 0 || split > head_dim {
        return Err("ADA-A3 split-dim split must be in 1..=head_dim");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 split-dim query must contain head_dim elements");
    }
    if keys.is_empty() || keys.len() % head_dim != 0 {
        return Err("ADA-A3 split-dim keys must be non-empty row-major [key_count, head_dim]");
    }
    if query.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err("ADA-A3 split-dim inputs must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 split-dim score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 split-dim epsilon must be finite and non-negative");
    }
    let n = keys.len() / head_dim;
    let q_tail = query[split..].iter().map(|v| v * v).sum::<f64>().sqrt();
    let q_norm = query.iter().map(|v| v * v).sum::<f64>().sqrt();

    let mut prefix = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    for (i, row) in keys.chunks_exact(head_dim).enumerate() {
        let p: f64 = query[..split]
            .iter()
            .zip(&row[..split])
            .map(|(a, b)| a * b)
            .sum();
        let p = p * score_scale;
        // r_i is the prefill-stored tail norm.
        let r = row[split..].iter().map(|v| v * v).sum::<f64>().sqrt();
        let k_norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let extra = original_norms.map_or(0.0, |(qn, kn, rel)| rel * qn * kn[i]);
        let delta = score_scale
            * (q_tail * r * (1.0 + DELTA_INFLATION) + dot_slack * q_norm * k_norm + extra);
        let h = p + delta;
        if !h.is_finite() {
            return Err("ADA-A3 split-dim scores must be finite");
        }
        prefix.push(p);
        hi.push(h);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| hi[b].total_cmp(&hi[a]).then(a.cmp(&b)));
    // ln_suffix[j] = ln Σ_{r ≥ j} e^{h_{order[r]}}, accumulated in log domain
    // so that a loose, huge `h_max` cannot underflow the unread mass to zero.
    let mut ln_suffix = vec![f64::NEG_INFINITY; n + 1];
    for r in (0..n).rev() {
        let (a, b) = (hi[order[r]], ln_suffix[r + 1]);
        ln_suffix[r] = if b == f64::NEG_INFINITY {
            a
        } else {
            a.max(b) + (-(a - b).abs()).exp().ln_1p()
        };
    }

    let mut cost = SplitDimCost {
        prefill_tail_norms: n,
        split,
        prefix_macs: n * split,
        sorted_keys: n,
        transcendental_evaluations: 2 * n,
        ..SplitDimCost::default()
    };
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64;
    let mut certificate = 0.0_f64;
    let mut done = n;
    for (rank, &idx) in order.iter().enumerate() {
        let row = &keys[idx * head_dim..(idx + 1) * head_dim];
        let tail: f64 = query[split..]
            .iter()
            .zip(&row[split..])
            .map(|(a, b)| a * b)
            .sum();
        let score = prefix[idx] + score_scale * tail;
        if !score.is_finite() {
            return Err("ADA-A3 split-dim scores must be finite");
        }
        cost.rows_completed += 1;
        cost.completion_macs += head_dim - split;
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
            break;
        }
        let ln_s = running_max + mass.ln();
        let ln_t = ln_suffix[read_so_far] + MASS_INFLATION.ln_1p();
        cost.transcendental_evaluations += 3;
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
        return Err("ADA-A3 split-dim mass must be positive and finite");
    }
    let mut probabilities = vec![0.0_f64; n];
    let mut read = vec![false; n];
    for &(idx, s) in &scores {
        probabilities[idx] = (s - running_max).exp() / total;
        read[idx] = true;
    }
    Ok((
        SplitDimSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
            score_scale,
        },
        cost,
    ))
}

/// Evaluate split-dimension certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_split_dim_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_split_dim_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    split: usize,
) -> Result<SplitDimSoftmax, &'static str> {
    certified_split_dim_softmax_accounted(query, keys, head_dim, score_scale, epsilon, split)
        .map(|(r, _)| r)
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_sign_loss
)]
mod tests {
    use super::{certified_split_dim_softmax, certified_split_dim_softmax_accounted};
    use crate::dense_qk_scores;

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
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 1.0, 0.1, 1).is_ok());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 1.0, 0.1, 2).is_ok());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 1.0, 0.1, 0).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 1.0, 0.1, 3).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 0, 1.0, 0.1, 1).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 5000, 1.0, 0.1, 1).is_err());
        assert!(certified_split_dim_softmax(&[1.0], &k, 2, 1.0, 0.1, 1).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &[], 2, 1.0, 0.1, 1).is_err());
        assert!(
            certified_split_dim_softmax(&[1.0, 0.0], &[1.0, 2.0, 3.0], 2, 1.0, 0.1, 1).is_err()
        );
        assert!(certified_split_dim_softmax(&[f64::NAN, 0.0], &k, 2, 1.0, 0.1, 1).is_err());
        assert!(
            certified_split_dim_softmax(&[1.0, 0.0], &[f64::INFINITY, 0.0], 2, 1.0, 0.1, 1)
                .is_err()
        );
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 0.0, 0.1, 1).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, f64::NAN, 0.1, 1).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 1.0, -0.1, 1).is_err());
        assert!(certified_split_dim_softmax(&[1.0, 0.0], &k, 2, 1.0, f64::NAN, 1).is_err());
    }

    #[test]
    fn full_split_is_exact_and_costs_dense() {
        let mut rng = Rng(7);
        let (n, d) = (16, 8);
        let keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let (r, c) = certified_split_dim_softmax_accounted(&q, &keys, d, 0.4, 0.0, d).unwrap();
        // ε = 0 never certifies early except by rounding slack; must be correct.
        assert!(l1(&r.probabilities, &oracle(&q, &keys, d, 0.4)) <= r.certified_l1_bound + 1e-12);
        assert!(c.total_macs() <= n * d);
    }

    #[test]
    fn epsilon_zero_reads_everything_unless_certified() {
        let mut rng = Rng(11);
        let (n, d) = (24, 8);
        let keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let (r, c) = certified_split_dim_softmax_accounted(&q, &keys, d, 0.3, 0.0, 3).unwrap();
        assert_eq!(c.rows_completed + c.rows_skipped, n);
        assert_eq!(c.total_macs(), n * 3 + c.rows_completed * 5);
        assert!(l1(&r.probabilities, &oracle(&q, &keys, d, 0.3)) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn witness_skips_distractors_on_prefix_energy() {
        // Needle aligned with q on the prefix; distractors have tiny tails.
        let (n, d, m) = (64, 8, 4);
        let mut keys = vec![0.0_f64; n * d];
        keys[0] = 3.0;
        for i in 1..n {
            keys[i * d] = -3.0;
            keys[i * d + 7] = 0.01;
        }
        let mut q = vec![0.0_f64; d];
        q[0] = 3.0;
        q[7] = 0.1;
        let (r, c) = certified_split_dim_softmax_accounted(&q, &keys, d, 1.0, 1e-3, m).unwrap();
        assert_eq!(c.rows_completed, 1);
        assert!(r.probabilities[0] > 0.999);
        assert!(c.total_macs() < n * d);
        assert!(l1(&r.probabilities, &oracle(&q, &keys, d, 1.0)) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn exhaustive_grid_has_no_violation() {
        let vals = [-2.0, 0.0, 1.5];
        let max_len = if cfg!(miri) { 2_usize } else { 4 };
        let d = 3;
        let mut cases = 0_usize;
        let mut skipping = 0_usize;
        for len in 1..=max_len {
            let cells = len * d;
            let total = vals.len().pow(cells as u32);
            let step = if cfg!(miri) { 37 } else { 1 };
            let mut code = 0_usize;
            while code < total {
                let mut c = code;
                let keys: Vec<f64> = (0..cells)
                    .map(|_| {
                        let v = vals[c % vals.len()];
                        c /= vals.len();
                        v
                    })
                    .collect();
                for q in [[1.0, 0.5, -1.0], [0.0, 0.0, 0.0], [2.0, -1.0, 3.0]] {
                    for eps in [0.0, 0.01, 0.2] {
                        for m in 1..=d {
                            let (r, c) =
                                certified_split_dim_softmax_accounted(&q, &keys, d, 0.7, eps, m)
                                    .unwrap();
                            let err = l1(&r.probabilities, &oracle(&q, &keys, d, 0.7));
                            assert!(err <= r.certified_l1_bound + 1e-12, "viol {err}");
                            assert!(
                                r.certified_l1_bound <= eps.max(0.0) + 1e-15 || c.rows_skipped == 0
                            );
                            cases += 1;
                            skipping += usize::from(c.rows_skipped > 0);
                        }
                    }
                }
                code += step;
            }
        }
        assert!(cases > if cfg!(miri) { 100 } else { 10_000 });
        if !cfg!(miri) {
            assert!(skipping > 0);
        }
    }

    #[test]
    fn adversarial_random_sweep_has_no_violation() {
        let mut rng = Rng(0x5EED);
        let iters = if cfg!(miri) { 10 } else { 400 };
        for _ in 0..iters {
            let d = 2 + (rng.next() * 14.0) as usize;
            let n = 1 + (rng.next() * 40.0) as usize;
            let m = 1 + (rng.next() * d as f64) as usize % d;
            let mag = [1.0, 10.0, 1000.0][(rng.next() * 3.0) as usize % 3];
            let mut keys: Vec<f64> = (0..n * d).map(|_| rng.gauss()).collect();
            // outlier tail coordinates and duplicated rows
            for i in 0..n {
                if rng.next() < 0.3 {
                    keys[i * d + d - 1] *= mag;
                }
                if i > 0 && rng.next() < 0.2 {
                    let (a, b) = keys.split_at_mut(i * d);
                    b[..d].copy_from_slice(&a[(i - 1) * d..i * d]);
                }
            }
            let q: Vec<f64> = (0..d)
                .map(|_| rng.gauss() * if rng.next() < 0.3 { mag } else { 1.0 })
                .collect();
            let scale = 0.05 + rng.next();
            for eps in [0.0, 1e-4, 1e-2, 0.3] {
                let Ok((r, _)) = certified_split_dim_softmax_accounted(&q, &keys, d, scale, eps, m)
                else {
                    continue; // fail-closed on overflow is allowed
                };
                let Ok(sc) = dense_qk_scores(&q, &keys, d, scale) else {
                    continue;
                };
                if sc.iter().any(|s| !s.is_finite()) {
                    continue;
                }
                let err = l1(&r.probabilities, &oracle(&q, &keys, d, scale));
                assert!(
                    err <= r.certified_l1_bound + 1e-12,
                    "viol {err} d {d} n {n} m {m} eps {eps} mag {mag} scale {scale} cert {}",
                    r.certified_l1_bound
                );
                assert!(r.certified_l1_bound <= eps || r.read.iter().all(|&x| x));
            }
        }
    }

    /// Prints the cost table recorded in the doc (`--ignored --nocapture`).
    #[test]
    #[ignore = "result table generator"]
    fn print_results_table() {
        let (n, d) = (256, 64);
        for decay in [0.0_f64, 0.05, 0.15] {
            for std in [1.0_f64, 3.0] {
                for eps in [1e-2, 1e-4] {
                    for frac in [4_usize, 2] {
                        let m = d / frac;
                        let (mut macs, mut rows, mut maxl1, mut maxc) =
                            (0.0, 0.0, 0.0_f64, 0.0_f64);
                        let draws = 30;
                        for s in 0..draws {
                            let mut rng = Rng(1000 + s);
                            let sd: Vec<f64> = (0..d).map(|j| (-decay * j as f64).exp()).collect();
                            let keys: Vec<f64> =
                                (0..n * d).map(|i| rng.gauss() * sd[i % d]).collect();
                            let q: Vec<f64> = (0..d).map(|j| rng.gauss() * sd[j]).collect();
                            let lv = sd.iter().map(|x| x * x).sum::<f64>();
                            let scale = std / lv;
                            let (r, c) =
                                certified_split_dim_softmax_accounted(&q, &keys, d, scale, eps, m)
                                    .unwrap();
                            let e = l1(&r.probabilities, &oracle(&q, &keys, d, scale));
                            assert!(e <= r.certified_l1_bound + 1e-12);
                            macs += c.total_macs() as f64 / (n * d) as f64;
                            rows += c.rows_completed as f64 / n as f64;
                            maxl1 = maxl1.max(e);
                            maxc = maxc.max(r.certified_l1_bound);
                        }
                        println!(
                            "decay {decay} std {std} eps {eps} m {m}: macs/dense {:.4} rows {:.4} maxL1 {maxl1:.2e} certmax {maxc:.2e}",
                            macs / draws as f64,
                            rows / draws as f64
                        );
                    }
                }
            }
        }
    }
}
