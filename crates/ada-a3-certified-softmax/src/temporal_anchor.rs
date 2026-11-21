//! Temporal anchor certificate reuse for decoding (ADA-A3).
//!
//! Research reference only. Consecutive decode queries are often close, so an
//! exact score computed for an *earlier* query `a` bounds the score under the
//! current query `q` without touching the key again:
//! `z_i(q) ≤ z_i(a) + s·‖q − a‖·‖k_i‖` (Cauchy–Schwarz on `q − a`).
//! Each key keeps one anchor (the last query for which its exact score was
//! computed, plus that score). A step computes the distance from `q` to each
//! distinct anchor (`d` MACs per anchor), forms the upper bounds, completes
//! keys exactly in decreasing bound order and stops once the truncation
//! certificate `2T/(S+T) ≤ ε` holds (`T` = exact log-domain sum of the unread
//! bounds). Keys without an anchor (newly appended) are always completed.
//! Completed keys re-anchor at `q`. Contract: the key store is append-only;
//! a per-row fingerprint check fails closed if a stored row changed.
//! Cost per step: `anchors·d + completed·d` MACs against dense `n·d`; the
//! anchor-distance term can exceed the saving, so the net can be worse than
//! dense and that is reported, not hidden.

const DELTA_INFLATION: f64 = 1.0e-9;
const MASS_INFLATION: f64 = 1.0e-9;
const MAX_HEAD_DIM: usize = 4096;
const DOT_SLACK: f64 = 1.0e-11;

/// Logical (not hardware) cost counters for one step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TemporalAnchorCost {
    /// Keys in the store at this step.
    pub keys: usize,
    /// Keys with no anchor (completed unconditionally).
    pub new_rows: usize,
    /// Distinct anchor queries referenced.
    pub distinct_anchors: usize,
    /// `distinct_anchors·d` MACs for `‖q − a‖`.
    pub anchor_distance_macs: usize,
    /// Rows completed exactly this step (including new rows).
    pub rows_completed: usize,
    /// Rows never completed this step.
    pub rows_skipped: usize,
    /// `rows_completed·d` MACs.
    pub completion_macs: usize,
    /// Key-store integrity-guard reads (`old_rows·d` loads, not MACs; a
    /// production caller relying on the append-only contract would skip it).
    pub integrity_reads: usize,
    /// Sort size (comparison-class work).
    pub sorted_keys: usize,
    /// `exp`/`ln` evaluations.
    pub transcendental_evaluations: usize,
}

impl TemporalAnchorCost {
    /// Total MACs of the step.
    #[must_use]
    pub fn total_macs(&self, head_dim: usize) -> usize {
        let _ = head_dim;
        self.anchor_distance_macs + self.completion_macs
    }
}

/// Result of one anchored step.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalAnchorSoftmax {
    /// Renormalised probabilities in key order; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when every key was completed).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was completed this step.
    pub read: Vec<bool>,
}

#[derive(Clone, Copy, Debug)]
struct KeyMeta {
    norm: f64,
    fingerprint: u64,
    /// `(anchor id, exact scaled score at that anchor)`.
    anchor: Option<(usize, f64)>,
}

/// Stateful anchored screen over an append-only key store.
#[derive(Clone, Debug)]
pub struct TemporalAnchorScreen {
    head_dim: usize,
    score_scale: f64,
    anchors: Vec<Option<Vec<f64>>>,
    meta: Vec<KeyMeta>,
}

impl TemporalAnchorScreen {
    /// Create an empty screen.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid `head_dim` or `score_scale`.
    pub fn new(head_dim: usize, score_scale: f64) -> Result<Self, &'static str> {
        if head_dim == 0 || head_dim > MAX_HEAD_DIM {
            return Err("ADA-A3 temporal-anchor head_dim must be in 1..=4096");
        }
        if !score_scale.is_finite() || score_scale <= 0.0 {
            return Err("ADA-A3 temporal-anchor score_scale must be finite and positive");
        }
        Ok(Self {
            head_dim,
            score_scale,
            anchors: Vec::new(),
            meta: Vec::new(),
        })
    }

    /// Evaluate one decode step against the full current key store.
    ///
    /// `keys` is row-major `[n, head_dim]` and must extend the previously
    /// seen store (same leading rows).
    ///
    /// # Errors
    ///
    /// Returns an error for a bad layout, non-finite input, invalid `epsilon`,
    /// a shrunk or modified key store, or a non-finite score. State is left
    /// unchanged on error.
    #[must_use = "the certified distribution should be checked"]
    #[allow(clippy::too_many_lines, clippy::many_single_char_names)]
    pub fn step(
        &mut self,
        query: &[f64],
        keys: &[f64],
        epsilon: f64,
    ) -> Result<(TemporalAnchorSoftmax, TemporalAnchorCost), &'static str> {
        let d = self.head_dim;
        let s = self.score_scale;
        if query.len() != d {
            return Err("ADA-A3 temporal-anchor query must contain head_dim elements");
        }
        if keys.is_empty() || keys.len() % d != 0 {
            return Err("ADA-A3 temporal-anchor keys must be non-empty row-major [n, head_dim]");
        }
        if query.iter().chain(keys).any(|v| !v.is_finite()) {
            return Err("ADA-A3 temporal-anchor inputs must be finite");
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err("ADA-A3 temporal-anchor epsilon must be finite and non-negative");
        }
        let n = keys.len() / d;
        if n < self.meta.len() {
            return Err("ADA-A3 temporal-anchor key store must be append-only");
        }
        let norm_of = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fp_of = |r: &[f64]| {
            r.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
                (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3)
            })
        };
        for (i, m) in self.meta.iter().enumerate() {
            let row = &keys[i * d..(i + 1) * d];
            if norm_of(row).to_bits() != m.norm.to_bits() || fp_of(row) != m.fingerprint {
                return Err("ADA-A3 temporal-anchor stored key row changed");
            }
        }
        let q_norm = norm_of(query);

        // Anchor distances (one per referenced anchor id).
        let mut dist: Vec<Option<(f64, f64)>> = vec![None; self.anchors.len()];
        let mut distinct = 0_usize;
        for m in &self.meta {
            if let Some((id, _)) = m.anchor {
                if dist[id].is_none() {
                    let a = self.anchors[id].as_ref().ok_or("ADA-A3 anchor missing")?;
                    let dd = query
                        .iter()
                        .zip(a)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f64>()
                        .sqrt();
                    dist[id] = Some((dd, norm_of(a)));
                    distinct += 1;
                }
            }
        }

        // Bounds for anchored rows; new rows are completed first.
        let mut new_rows: Vec<usize> = (self.meta.len()..n).collect();
        let mut old: Vec<usize> = Vec::new();
        let mut hi = vec![0.0_f64; n];
        for (i, m) in self.meta.iter().enumerate() {
            match m.anchor {
                Some((id, a_score)) => {
                    let (dd, a_norm) = dist[id].ok_or("ADA-A3 anchor distance missing")?;
                    let h = a_score
                        + s * (dd * m.norm * (1.0 + DELTA_INFLATION)
                            + DOT_SLACK * (q_norm + a_norm) * m.norm);
                    if !h.is_finite() {
                        return Err("ADA-A3 temporal-anchor bounds must be finite");
                    }
                    hi[i] = h;
                    old.push(i);
                }
                None => new_rows.push(i),
            }
        }
        old.sort_by(|&a, &b| hi[b].total_cmp(&hi[a]).then(a.cmp(&b)));
        let m_old = old.len();
        let mut ln_suffix = vec![f64::NEG_INFINITY; m_old + 1];
        for r in (0..m_old).rev() {
            let (a, b) = (hi[old[r]], ln_suffix[r + 1]);
            ln_suffix[r] = if b == f64::NEG_INFINITY {
                a
            } else {
                a.max(b) + (-(a - b).abs()).exp().ln_1p()
            };
        }

        let mut cost = TemporalAnchorCost {
            keys: n,
            integrity_reads: self.meta.len() * d,
            distinct_anchors: distinct,
            anchor_distance_macs: distinct * d,
            sorted_keys: m_old,
            transcendental_evaluations: 2 * m_old,
            ..TemporalAnchorCost::default()
        };
        cost.new_rows = new_rows.len();

        let exact = |idx: usize| -> Result<f64, &'static str> {
            let row = &keys[idx * d..(idx + 1) * d];
            let z = s * query.iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
            if z.is_finite() {
                Ok(z)
            } else {
                Err("ADA-A3 temporal-anchor scores must be finite")
            }
        };

        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
        for &idx in &new_rows {
            scores.push((idx, exact(idx)?));
            cost.rows_completed += 1;
            cost.completion_macs += d;
        }
        let mut certificate = 0.0_f64;
        let mut done_old = m_old;
        let mut running_max = f64::NEG_INFINITY;
        let mut mass = 0.0_f64;
        let absorb = |score: f64, running_max: &mut f64, mass: &mut f64| {
            if score > *running_max {
                *mass = if *running_max == f64::NEG_INFINITY {
                    1.0
                } else {
                    *mass * (*running_max - score).exp() + 1.0
                };
                *running_max = score;
            } else {
                *mass += (score - *running_max).exp();
            }
        };
        for &(_, sc) in &scores {
            absorb(sc, &mut running_max, &mut mass);
        }
        cost.transcendental_evaluations += scores.len();
        if m_old > 0 {
            let mut r = 0_usize;
            loop {
                // Check certificate before reading more (possible only when
                // something has been read already).
                if running_max > f64::NEG_INFINITY {
                    let ln_s = running_max + mass.ln();
                    let ln_t = ln_suffix[r] + MASS_INFLATION.ln_1p();
                    cost.transcendental_evaluations += 3;
                    let cert = 2.0 / (1.0 + (ln_s - ln_t).exp());
                    if cert <= epsilon {
                        certificate = cert;
                        done_old = r;
                        break;
                    }
                }
                if r == m_old {
                    done_old = m_old;
                    break;
                }
                let idx = old[r];
                let sc = exact(idx)?;
                scores.push((idx, sc));
                cost.rows_completed += 1;
                cost.completion_macs += d;
                absorb(sc, &mut running_max, &mut mass);
                cost.transcendental_evaluations += 1;
                r += 1;
            }
        }
        cost.rows_skipped = m_old - done_old;

        let total: f64 = scores.iter().map(|&(_, sc)| (sc - running_max).exp()).sum();
        if !total.is_finite() || total <= 0.0 {
            return Err("ADA-A3 temporal-anchor mass must be positive and finite");
        }
        let mut probabilities = vec![0.0_f64; n];
        let mut read = vec![false; n];
        for &(idx, sc) in &scores {
            probabilities[idx] = (sc - running_max).exp() / total;
            read[idx] = true;
        }

        // Commit state: new anchor for completed rows, then prune anchors.
        let new_id = self.anchors.len();
        self.anchors.push(Some(query.to_vec()));
        for i in self.meta.len()..n {
            let row = &keys[i * d..(i + 1) * d];
            self.meta.push(KeyMeta {
                norm: norm_of(row),
                fingerprint: fp_of(row),
                anchor: None,
            });
        }
        for &(idx, sc) in &scores {
            self.meta[idx].anchor = Some((new_id, sc));
        }
        let mut used = vec![false; self.anchors.len()];
        for m in &self.meta {
            if let Some((id, _)) = m.anchor {
                used[id] = true;
            }
        }
        for (id, u) in used.iter().enumerate() {
            if !u {
                self.anchors[id] = None;
            }
        }
        Ok((
            TemporalAnchorSoftmax {
                probabilities,
                certified_l1_bound: certificate,
                epsilon,
                read,
            },
            cost,
        ))
    }
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
    use super::TemporalAnchorScreen;
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

    #[test]
    fn rejects_invalid_input_and_changed_store() {
        assert!(TemporalAnchorScreen::new(0, 1.0).is_err());
        assert!(TemporalAnchorScreen::new(2, 0.0).is_err());
        assert!(TemporalAnchorScreen::new(2, f64::NAN).is_err());
        let mut s = TemporalAnchorScreen::new(2, 1.0).unwrap();
        assert!(s.step(&[1.0], &[1.0, 0.0], 0.1).is_err());
        assert!(s.step(&[1.0, 0.0], &[], 0.1).is_err());
        assert!(s.step(&[1.0, f64::NAN], &[1.0, 0.0], 0.1).is_err());
        assert!(s.step(&[1.0, 0.0], &[1.0, 0.0], -1.0).is_err());
        assert!(s.step(&[1.0, 0.0], &[1.0, 0.0, 2.0, 2.0], 0.1).is_ok());
        // shrunk and modified stores fail closed
        assert!(s.step(&[1.0, 0.0], &[1.0, 0.0], 0.1).is_err());
        assert!(s.step(&[1.0, 0.0], &[1.0, 0.5, 2.0, 2.0], 0.1).is_err());
        // permutation preserving norm and coordinate sum is also caught
        assert!(s.step(&[1.0, 0.0], &[0.0, 1.0, 2.0, 2.0], 0.1).is_err());
        assert!(s.step(&[1.0, 0.0], &[1.0, 0.0, 2.0, 2.0], 0.1).is_ok());
    }

    #[test]
    fn identical_repeated_query_skips_all_but_needed() {
        // needle + distractors; same query twice: second step uses anchors
        let (n, d) = (64, 8);
        let mut keys = vec![0.0_f64; n * d];
        keys[0] = 3.0;
        for i in 1..n {
            keys[i * d] = -3.0;
        }
        let mut q = vec![0.0_f64; d];
        q[0] = 3.0;
        let mut s = TemporalAnchorScreen::new(d, 1.0).unwrap();
        let (_, c1) = s.step(&q, &keys, 1e-3).unwrap();
        assert_eq!(c1.new_rows, n);
        assert_eq!(c1.rows_completed, n);
        let (r2, c2) = s.step(&q, &keys, 1e-3).unwrap();
        assert!(c2.rows_completed <= 2, "{c2:?}");
        assert!(c2.total_macs(d) < n * d / 4);
        assert!(l1(&r2.probabilities, &oracle(&q, &keys, d, 1.0)) <= r2.certified_l1_bound + 1e-12);
    }

    #[test]
    fn walk_with_appends_has_no_violation() {
        let mut rng = Rng(0xA11C);
        let iters = if cfg!(miri) { 2 } else { 60 };
        for _ in 0..iters {
            let d = 2 + (rng.next() * 14.0) as usize;
            let mag = [1.0, 10.0, 1000.0][(rng.next() * 3.0) as usize % 3];
            let step_sz = [0.0, 0.01, 0.3, 3.0][(rng.next() * 4.0) as usize % 4];
            let scale = 0.05 + rng.next();
            let mut keys: Vec<f64> = Vec::new();
            let mut q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
            let mut s = TemporalAnchorScreen::new(d, scale).unwrap();
            let steps = if cfg!(miri) { 3 } else { 25 };
            for t in 0..steps {
                let add = if t == 0 {
                    1 + (rng.next() * 20.0) as usize
                } else {
                    (rng.next() * 3.0) as usize
                };
                for _ in 0..add {
                    let big = rng.next() < 0.2;
                    for _ in 0..d {
                        keys.push(rng.gauss() * if big { mag } else { 1.0 });
                    }
                    if rng.next() < 0.2 && keys.len() >= 2 * d {
                        let l = keys.len();
                        let (a, b) = keys.split_at_mut(l - d);
                        b.copy_from_slice(&a[l - 2 * d..l - d]);
                    }
                }
                for v in &mut q {
                    *v += step_sz * rng.gauss();
                }
                for eps in [0.0, 1e-3, 0.1] {
                    // each epsilon uses a cloned screen so histories are equal
                    let mut s2 = s.clone();
                    let Ok((r, c)) = s2.step(&q, &keys, eps) else {
                        continue;
                    };
                    let Ok(sc) = dense_qk_scores(&q, &keys, d, scale) else {
                        continue;
                    };
                    if sc.iter().any(|x| !x.is_finite()) {
                        continue;
                    }
                    let err = l1(&r.probabilities, &oracle(&q, &keys, d, scale));
                    assert!(
                        err <= r.certified_l1_bound + 1e-12,
                        "viol {err} d {d} t {t} eps {eps} mag {mag} step {step_sz} cert {}",
                        r.certified_l1_bound
                    );
                    assert!(r.certified_l1_bound <= eps || r.read.iter().all(|&x| x));
                    assert_eq!(c.rows_completed + c.rows_skipped, c.keys);
                }
                // advance the real history with a moderate epsilon
                if s.step(&q, &keys, 1e-3).is_err() {
                    break;
                }
            }
        }
    }

    /// Prints the cost table recorded in the doc (`--ignored --nocapture`).
    #[test]
    #[ignore = "result table generator"]
    fn print_results_table() {
        let (n0, d, steps) = (256_usize, 64_usize, 64_usize);
        for step_sz in [0.0_f64, 0.02, 0.1, 0.3, 1.0] {
            for std in [3.0_f64, 8.0, 16.0] {
                for eps in [1e-2, 1e-4] {
                    let (mut ratio, mut rows, mut maxl1, mut maxc) = (0.0, 0.0, 0.0_f64, 0.0_f64);
                    let draws = 10;
                    for sd in 0..draws {
                        let mut rng = Rng(500 + sd);
                        let scale = std / d as f64;
                        let mut keys: Vec<f64> = (0..n0 * d).map(|_| rng.gauss()).collect();
                        let mut q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                        let mut s = TemporalAnchorScreen::new(d, scale).unwrap();
                        let (mut macs, mut dense, mut rw, mut cnt) = (0.0, 0.0, 0.0, 0.0);
                        for t in 0..steps {
                            if t > 0 {
                                keys.extend((0..d).map(|_| rng.gauss()));
                                // random step, renormalised to the initial norm
                                let n0 = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                                for v in &mut q {
                                    *v += step_sz * rng.gauss();
                                }
                                let n1 = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                                for v in &mut q {
                                    *v *= n0 / n1;
                                }
                            }
                            let (r, c) = s.step(&q, &keys, eps).unwrap();
                            let e = l1(&r.probabilities, &oracle(&q, &keys, d, scale));
                            assert!(e <= r.certified_l1_bound + 1e-12);
                            maxl1 = maxl1.max(e);
                            maxc = maxc.max(r.certified_l1_bound);
                            if t > 0 {
                                macs += c.total_macs(d) as f64;
                                dense += (keys.len()) as f64;
                                rw += c.rows_completed as f64 / c.keys as f64;
                                cnt += 1.0;
                            }
                        }
                        ratio += macs / dense;
                        rows += rw / cnt;
                    }
                    println!(
                        "step {step_sz} std {std} eps {eps}: macs/dense {:.4} rows {:.4} maxL1 {maxl1:.2e} certmax {maxc:.2e}",
                        ratio / draws as f64,
                        rows / draws as f64
                    );
                }
            }
        }
    }
}
