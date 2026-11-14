//! Angular-cone norm-band bucket certified softmax (ADA-A3).
//!
//! Research reference only. Prefill groups keys into buckets: a norm band
//! (norms within a ratio `rho` of the band's largest) split by greedy
//! angular leaders (unit-direction cosine to the leader `>= tau`). Each
//! bucket stores its member count, its largest and smallest norm, a unit
//! centroid `c` and the cosine `cos_theta` of its widest member angle to `c`.
//! For a query with angle `phi` to `c`, the triangle inequality on the sphere
//! bounds every member angle to `q` below by `phi - theta`, hence
//! `s<q,k> <= s|q||k| g` with `g = cos(phi - theta)` when `phi > theta` and
//! `g = 1` otherwise (`nmax` if `g >= 0`, `nmin` if `g < 0`).
//! Buckets are read in non-increasing bound order; after each bucket the
//! unread mass is bounded by `sum_b n_b exp(U_b - m)` and the scan stops once
//! `2 T / (S + T) <= epsilon`. Only one centroid dot per bucket is paid for
//! the buckets that stay unread. The scheme degrades to #71 norm order when
//! cones are wide and gains where directions cluster at equal norms.

const SLACK: f64 = 1.0e-9;
const BOUND_INFLATION: f64 = 1.0e-12;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConeBucketCost {
    /// Buckets built at prefill.
    pub buckets: usize,
    /// Prefill leader-cosine evaluations (greedy angular clustering).
    pub prefill_leader_dots: usize,
    /// Query-time centroid dots (one per bucket).
    pub query_time_centroid_dots: usize,
    /// Key rows dotted with the query at inference time.
    pub query_time_key_rows_read: usize,
    /// Key rows that were not read.
    pub key_rows_skipped: usize,
    /// `exp` evaluations.
    pub exp_evaluations: usize,
}

/// Cone-bucket certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct ConeBucketSoftmax {
    /// Renormalized probabilities in input order; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when all keys were read).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was read.
    pub read: Vec<bool>,
}

struct Bucket {
    members: Vec<usize>,
    nmax: f64,
    nmin: f64,
    centroid: Vec<f64>, // unit, or empty when the bucket has no usable cone
    cos_theta: f64,
}

#[allow(clippy::too_many_arguments)]
fn validate(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    norm_ratio: f64,
    cos_threshold: f64,
) -> Result<usize, &'static str> {
    if head_dim == 0 {
        return Err("ADA-A3 cone-bucket softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 cone-bucket softmax query must contain head_dim elements");
    }
    if keys.is_empty() || keys.len() % head_dim != 0 {
        return Err("ADA-A3 cone-bucket softmax keys must be non-empty row-major [n, head_dim]");
    }
    if query.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err("ADA-A3 cone-bucket softmax inputs must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 cone-bucket softmax score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 cone-bucket softmax epsilon must be finite and non-negative");
    }
    if !norm_ratio.is_finite() || norm_ratio < 1.0 {
        return Err("ADA-A3 cone-bucket softmax norm_ratio must be finite and at least 1");
    }
    if !cos_threshold.is_finite() || !(-1.0..=1.0).contains(&cos_threshold) {
        return Err("ADA-A3 cone-bucket softmax cos_threshold must lie in [-1, 1]");
    }
    Ok(keys.len() / head_dim)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn build_buckets(
    keys: &[f64],
    d: usize,
    norm_ratio: f64,
    cos_threshold: f64,
    cost: &mut ConeBucketCost,
) -> Vec<Bucket> {
    let n = keys.len() / d;
    let norms: Vec<f64> = keys.chunks_exact(d).map(norm).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]).then(a.cmp(&b)));
    let unit = |i: usize| -> Vec<f64> {
        keys[i * d..(i + 1) * d]
            .iter()
            .map(|v| v / norms[i])
            .collect()
    };

    let mut buckets = Vec::new();
    let mut pos = 0;
    while pos < n {
        // Norm band.
        let top = norms[order[pos]];
        let mut end = pos;
        while end < n && norms[order[end]] * norm_ratio >= top {
            end += 1;
        }
        if top == 0.0 {
            end = n;
        }
        let band: Vec<usize> = order[pos..end].to_vec();
        pos = end;
        // Greedy angular leaders inside the band (zero-norm keys form one group).
        let mut groups: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
        for &i in &band {
            if norms[i] == 0.0 {
                groups.push((Vec::new(), vec![i]));
                continue;
            }
            let u = unit(i);
            let mut placed = false;
            for (leader, members) in &mut groups {
                if leader.is_empty() {
                    continue;
                }
                cost.prefill_leader_dots += 1;
                if dot(leader, &u) >= cos_threshold {
                    members.push(i);
                    placed = true;
                    break;
                }
            }
            if !placed {
                groups.push((u, vec![i]));
            }
        }
        for (_, members) in groups {
            let nmax = members.iter().map(|&i| norms[i]).fold(0.0, f64::max);
            let nmin = members
                .iter()
                .map(|&i| norms[i])
                .fold(f64::INFINITY, f64::min);
            let mut sum = vec![0.0; d];
            for &i in members.iter().filter(|&&i| norms[i] > 0.0) {
                for (s, v) in sum.iter_mut().zip(unit(i)) {
                    *s += v;
                }
            }
            let sn = norm(&sum);
            let (centroid, cos_theta) = if sn > 0.0 && sn.is_finite() {
                let c: Vec<f64> = sum.iter().map(|v| v / sn).collect();
                let ct = members
                    .iter()
                    .filter(|&&i| norms[i] > 0.0)
                    .map(|&i| dot(&c, &unit(i)))
                    .fold(1.0, f64::min);
                (c, ct)
            } else {
                (Vec::new(), -1.0)
            };
            buckets.push(Bucket {
                members,
                nmax,
                nmin,
                centroid,
                cos_theta,
            });
        }
    }
    cost.buckets = buckets.len();
    buckets
}

/// Upper bound on `score_scale * <q, k>` over all members of the bucket.
fn bucket_bound(b: &Bucket, q: &[f64], q_norm: f64, scale: f64, cost: &mut ConeBucketCost) -> f64 {
    if b.nmax == 0.0 {
        return 0.0;
    }
    let mut g = 1.0;
    if !b.centroid.is_empty() && q_norm > 0.0 && b.cos_theta > SLACK {
        cost.query_time_centroid_dots += 1;
        let cos_phi = (dot(q, &b.centroid) / q_norm).clamp(-1.0, 1.0);
        let cos_wide = b.cos_theta - SLACK; // widen the cone
        if cos_phi < cos_wide {
            let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
            let sin_wide = (1.0 - cos_wide * cos_wide).max(0.0).sqrt();
            g = (cos_phi * cos_wide + sin_phi * sin_wide + SLACK).min(1.0);
        }
    }
    let n_used = if g >= 0.0 { b.nmax } else { b.nmin };
    let v = scale * q_norm * n_used * g;
    if v >= 0.0 {
        v * (1.0 + BOUND_INFLATION)
    } else {
        v * (1.0 - BOUND_INFLATION)
    }
}

/// Evaluate the cone-bucket certified softmax with logical costs.
///
/// # Errors
///
/// Returns an error for invalid layout, non-finite input, or invalid
/// `score_scale`, `epsilon`, `norm_ratio` (`>= 1`) or `cos_threshold`
/// (`[-1, 1]`), or non-finite scores/mass.
#[must_use = "the certified distribution should be checked"]
pub fn certified_cone_bucket_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    norm_ratio: f64,
    cos_threshold: f64,
) -> Result<(ConeBucketSoftmax, ConeBucketCost), &'static str> {
    let n = validate(
        query,
        keys,
        head_dim,
        score_scale,
        epsilon,
        norm_ratio,
        cos_threshold,
    )?;
    let mut cost = ConeBucketCost::default();
    let buckets = build_buckets(keys, head_dim, norm_ratio, cos_threshold, &mut cost);
    let q_norm = norm(query);
    let mut bounded: Vec<(f64, usize)> = buckets
        .iter()
        .enumerate()
        .map(|(i, b)| (bucket_bound(b, query, q_norm, score_scale, &mut cost), i))
        .collect();
    bounded.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut running_max = f64::NEG_INFINITY;
    let mut mass = 0.0_f64;
    let mut certificate = 0.0_f64;
    let mut read_count = n;
    for (pos, &(_, bi)) in bounded.iter().enumerate() {
        for &idx in &buckets[bi].members {
            let row = &keys[idx * head_dim..(idx + 1) * head_dim];
            let score = score_scale * dot(query, row);
            if !score.is_finite() {
                return Err("ADA-A3 cone-bucket softmax scores must be finite");
            }
            cost.query_time_key_rows_read += 1;
            cost.exp_evaluations += 1;
            if running_max == f64::NEG_INFINITY {
                mass = 1.0;
                running_max = score;
            } else if score > running_max {
                mass = mass * (running_max - score).exp() + 1.0;
                running_max = score;
            } else {
                mass += (score - running_max).exp();
            }
            scores.push((idx, score));
        }
        if pos + 1 == bounded.len() {
            certificate = 0.0;
            break;
        }
        #[allow(clippy::cast_precision_loss)]
        let tail: f64 = bounded[pos + 1..]
            .iter()
            .map(|&(u, j)| buckets[j].members.len() as f64 * (u - running_max).exp())
            .sum();
        cost.exp_evaluations += bounded.len() - pos - 1;
        let cert = if tail.is_finite() {
            2.0 * tail / (mass + tail)
        } else {
            2.0
        };
        if cert <= epsilon {
            certificate = cert;
            read_count = scores.len();
            break;
        }
    }
    cost.key_rows_skipped = n - read_count;
    let total: f64 = scores.iter().map(|&(_, s)| (s - running_max).exp()).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("ADA-A3 cone-bucket softmax mass must be positive and finite");
    }
    let mut probabilities = vec![0.0; n];
    let mut read = vec![false; n];
    for &(i, s) in &scores {
        probabilities[i] = (s - running_max).exp() / total;
        read[i] = true;
    }
    Ok((
        ConeBucketSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
        },
        cost,
    ))
}

/// Evaluate the cone-bucket certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_cone_bucket_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_cone_bucket_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    norm_ratio: f64,
    cos_threshold: f64,
) -> Result<ConeBucketSoftmax, &'static str> {
    certified_cone_bucket_softmax_accounted(
        query,
        keys,
        head_dim,
        score_scale,
        epsilon,
        norm_ratio,
        cos_threshold,
    )
    .map(|(r, _)| r)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
mod tests {
    use super::{certified_cone_bucket_softmax, certified_cone_bucket_softmax_accounted};
    use crate::{certified_norm_order_softmax_accounted, dense_qk_scores};

    fn oracle(q: &[f64], keys: &[f64], d: usize, scale: f64) -> Vec<f64> {
        let s = dense_qk_scores(q, keys, d, scale).unwrap();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let t: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = t.iter().sum();
        t.into_iter().map(|v| v / z).collect()
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
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let (a, b) = (self.unit(), self.unit());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    fn check(q: &[f64], keys: &[f64], d: usize, eps: f64, rho: f64, tau: f64) -> usize {
        let r = certified_cone_bucket_softmax(q, keys, d, 1.0, eps, rho, tau).unwrap();
        let o = oracle(q, keys, d, 1.0);
        let obs = l1(&r.probabilities, &o);
        assert!(
            obs <= r.certified_l1_bound + 1e-12,
            "observed {obs} > {}",
            r.certified_l1_bound
        );
        assert!(r.certified_l1_bound <= eps || r.read.iter().all(|&x| x));
        r.read.iter().filter(|&&x| x).count()
    }

    #[test]
    fn fail_closed_on_invalid_inputs() {
        let f = |q: &[f64], k: &[f64], d, s, e, r, t| {
            certified_cone_bucket_softmax(q, k, d, s, e, r, t)
        };
        assert!(f(&[1.0], &[], 1, 1.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 0, 1.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[1.0, 2.0], &[1.0], 1, 1.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[1.0, 2.0, 3.0], 2, 1.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[f64::NAN], &[1.0], 1, 1.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[f64::INFINITY], 1, 1.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 0.0, 0.1, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, -1.0, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, f64::NAN, 1.0, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, 0.1, 0.5, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, 0.1, f64::NAN, 0.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, 0.1, 1.0, 1.5).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, 0.1, 1.0, f64::NAN).is_err());
    }

    #[test]
    fn singleton_equal_keys_zero_query_and_zero_keys_stay_exact() {
        let r =
            certified_cone_bucket_softmax(&[1.0, 0.0], &[0.5, 0.5], 2, 1.0, 0.0, 2.0, 0.9).unwrap();
        assert_eq!(r.probabilities, vec![1.0]);
        let keys = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let r = certified_cone_bucket_softmax(&[1.0, 0.0], &keys, 2, 1.0, 1e-6, 2.0, 0.9).unwrap();
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        let keys = [1.0, 2.0, -3.0, 4.0, 0.0, 0.0];
        let r = certified_cone_bucket_softmax(&[0.0, 0.0], &keys, 2, 1.0, 1e-3, 2.0, 0.5).unwrap();
        assert!(r.read.iter().all(|&x| x));
        for p in &r.probabilities {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        // Zero keys mixed with live keys, many parameter choices.
        let keys = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 3.0];
        for rho in [1.0, 2.0, 100.0] {
            for tau in [-1.0, 0.0, 0.5, 1.0] {
                check(&[1.0, 0.3], &keys, 2, 1e-2, rho, tau);
            }
        }
    }

    #[test]
    fn overflowing_bound_is_handled_without_nan() {
        let keys = [1.0e3, 0.0, 1.0e-3, 0.0, 5.0e2, 0.0];
        let r =
            certified_cone_bucket_softmax(&[1.0e3, 0.0], &keys, 2, 1.0, 1e-3, 2.0, 0.9).unwrap();
        assert!(r.probabilities.iter().all(|p| p.is_finite()));
        assert!(r.certified_l1_bound.is_finite());
    }

    /// Two equal-norm clusters: one aligned with the query, one opposite.
    /// Norm order (#71) cannot skip anything; cone buckets drop the opposite
    /// cluster without dotting its rows.
    #[test]
    fn designed_witness_equal_norm_opposite_cluster() {
        let (d, per) = (4, 32);
        let mut rng = Rng(0xA5A5_1234_5678_9ABC);
        let mut keys = Vec::new();
        for sign in [1.0, -1.0] {
            for _ in 0..per {
                let mut v = [
                    sign * 1.0,
                    0.05 * rng.gauss(),
                    0.05 * rng.gauss(),
                    0.05 * rng.gauss(),
                ];
                let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                for x in &mut v {
                    *x *= 4.0 / nn;
                }
                keys.extend(v);
            }
        }
        let q = [4.0, 0.0, 0.0, 0.0];
        let (r, c) =
            certified_cone_bucket_softmax_accounted(&q, &keys, d, 1.0, 1e-3, 1.01, 0.95).unwrap();
        let (_, nc) = certified_norm_order_softmax_accounted(&q, &keys, d, 1.0, 1e-3).unwrap();
        println!(
            "witness: cone rows={} centroid_dots={} buckets={} norm-order rows={}",
            c.query_time_key_rows_read,
            c.query_time_centroid_dots,
            c.buckets,
            nc.query_time_key_rows_read
        );
        assert_eq!(nc.query_time_key_rows_read, 2 * per);
        assert_eq!(c.query_time_key_rows_read, per);
        assert!(c.query_time_key_rows_read + c.query_time_centroid_dots < 2 * per);
        let o = oracle(&q, &keys, d, 1.0);
        assert!(l1(&r.probabilities, &o) <= r.certified_l1_bound + 1e-12);
    }

    #[test]
    fn exhaustive_small_grid_has_zero_certificate_violations() {
        let alphabet: [[f64; 2]; 5] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [1.0, 1.0]];
        let q = [1.0, 0.25];
        let (mut cases, mut skipped) = (0usize, 0usize);
        for len in 1..=(if cfg!(miri) { 3usize } else { 4 }) {
            let total = 5usize.pow(u32::try_from(len).unwrap());
            for code in 0..total {
                let mut keys = Vec::new();
                let mut c = code;
                for _ in 0..len {
                    keys.extend_from_slice(&alphabet[c % 5]);
                    c /= 5;
                }
                for &eps in &[1e-3, 5e-2, 0.5] {
                    for &(rho, tau) in &[(1.0, 1.0), (2.0, 0.9), (100.0, 0.0), (1.5, -1.0)] {
                        let read = check(&q, &keys, 2, eps, rho, tau);
                        cases += 1;
                        if read < len {
                            skipped += 1;
                        }
                    }
                }
            }
        }
        println!("cone grid: cases={cases} skipped_cases={skipped}");
    }

    #[test]
    fn random_sweep_with_adversarial_bucket_parameters() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (d, n) = (6, 40);
        let mut cases = 0usize;
        for sigma in [0.0, 1.0] {
            for &eps in &[1e-4, 1e-2, 1e-1] {
                for &(rho, tau) in &[(1.0, 0.99), (1.2, 0.5), (3.0, 0.0), (1e6, -1.0), (1.0, 1.0)] {
                    for _ in 0..(if cfg!(miri) { 1 } else { 12 }) {
                        let mut keys = Vec::new();
                        for _ in 0..n {
                            let s = (sigma * rng.gauss()).exp();
                            keys.extend((0..d).map(|_| s * rng.gauss()));
                        }
                        let q: Vec<f64> = (0..d).map(|_| 2.0 * rng.gauss()).collect();
                        check(&q, &keys, d, eps, rho, tau);
                        cases += 1;
                    }
                }
            }
        }
        println!("cone random sweep: cases={cases}");
    }

    fn clustered(
        rng: &mut Rng,
        d: usize,
        clusters: usize,
        per: usize,
        norm: f64,
        jitter: f64,
    ) -> Vec<f64> {
        let centers: Vec<Vec<f64>> = (0..clusters)
            .map(|_| {
                let v: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                v.into_iter().map(|x| x / nn).collect()
            })
            .collect();
        let mut keys = Vec::new();
        for c in &centers {
            for _ in 0..per {
                let v: Vec<f64> = c.iter().map(|x| x + jitter * rng.gauss()).collect();
                let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                keys.extend(v.into_iter().map(|x| x * norm / nn));
            }
        }
        keys
    }

    #[test]
    fn measured_regimes_versus_norm_order() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let (d, clusters, per) = (16usize, 8usize, 32usize);
        let n = clusters * per;
        let samples: usize = if cfg!(miri) { 1 } else { 40 };
        for (label, jitter, norm) in [
            ("iid-gauss", -1.0, 0.0),
            ("clustered j=0.05", 0.05, 4.0),
            ("clustered j=0.2", 0.2, 4.0),
            ("clustered j=0.5", 0.5, 4.0),
        ] {
            for &eps in &[1e-4, 1e-2, 1e-1] {
                let (mut cone, mut cone_total, mut nord, mut maxl1, mut maxcert) =
                    (0.0, 0.0, 0.0, 0.0_f64, 0.0_f64);
                for _ in 0..samples {
                    let keys = if jitter < 0.0 {
                        (0..n * d).map(|_| rng.gauss()).collect()
                    } else {
                        clustered(&mut rng, d, clusters, per, norm, jitter)
                    };
                    let q: Vec<f64> = (0..d)
                        .map(|_| {
                            if jitter < 0.0 {
                                rng.gauss()
                            } else {
                                1.0 * rng.gauss()
                            }
                        })
                        .collect();
                    let q: Vec<f64> = if jitter < 0.0 {
                        q
                    } else {
                        let nn = q.iter().map(|x| x * x).sum::<f64>().sqrt();
                        q.into_iter().map(|x| 4.0 * x / nn).collect()
                    };
                    let (r, c) =
                        certified_cone_bucket_softmax_accounted(&q, &keys, d, 1.0, eps, 1.05, 0.7)
                            .unwrap();
                    let (_, nc) =
                        certified_norm_order_softmax_accounted(&q, &keys, d, 1.0, eps).unwrap();
                    let o = oracle(&q, &keys, d, 1.0);
                    let l = l1(&r.probabilities, &o);
                    assert!(l <= r.certified_l1_bound + 1e-12);
                    maxl1 = maxl1.max(l);
                    maxcert = maxcert.max(r.certified_l1_bound);
                    cone += c.query_time_key_rows_read as f64 / n as f64;
                    cone_total +=
                        (c.query_time_key_rows_read + c.query_time_centroid_dots) as f64 / n as f64;
                    nord += nc.query_time_key_rows_read as f64 / n as f64;
                }
                let s = samples as f64;
                println!(
                    "cone {label} eps={eps}: rows_frac cone={:.4} cone+centroids={:.4} norm_order={:.4} max_l1={maxl1:.3e} max_cert={maxcert:.3e}",
                    cone / s,
                    cone_total / s,
                    nord / s
                );
            }
        }
    }
}
