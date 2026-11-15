//! Multi-level angular cone tree certified softmax (ADA-A3).
//!
//! Research reference only. Prefill builds a binary tree over the non-zero
//! keys: each node stores its member count, largest and smallest norm, a unit
//! centroid `c` and `cos_theta` = smallest member cosine to `c`. A node is
//! split by two farthest-apart unit seeds (one pass of 2-means assignment;
//! degenerate splits fall back to an index halving) until it holds at most
//! `leaf_size` keys. Zero-norm keys form one extra leaf (score exactly 0).
//!
//! For a query with angle `phi` to `c`, the spherical triangle inequality
//! bounds every member angle to `q` below by `phi - theta`, so
//! `s<q,k> <= s|q| n g`, `g = cos(phi - theta)` if `phi > theta` else 1,
//! `n = nmax` if `g >= 0` else `nmin`. Query time is best-first: the frontier
//! node with the largest bound is expanded (children cost one centroid dot
//! each) or, if a leaf, its rows are dotted. The scan stops when
//! `2T / (S + T) <= epsilon`, `T = sum_frontier n_b exp(U_b - m)`. Unlike the
//! flat buckets of the angular-cone bucket module, a whole subtree is pruned
//! with the single dot of its root. Correctness does not depend on how the
//! tree was built.

const SLACK: f64 = 1.0e-9;
const BOUND_INFLATION: f64 = 1.0e-12;

/// Logical (not hardware) cost counters for one query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConeTreeCost {
    /// Tree nodes built at prefill (internal + leaves).
    pub nodes: usize,
    /// Prefill seed/assignment dot evaluations.
    pub prefill_split_dots: usize,
    /// Query-time centroid dots (one per visited node bound).
    pub query_time_centroid_dots: usize,
    /// Key rows dotted with the query at inference time.
    pub query_time_key_rows_read: usize,
    /// Key rows that were not read.
    pub key_rows_skipped: usize,
    /// `exp` evaluations.
    pub exp_evaluations: usize,
}

/// Cone-tree certified softmax result.
#[derive(Clone, Debug, PartialEq)]
pub struct ConeTreeSoftmax {
    /// Renormalized probabilities in input order; unread tokens are `0.0`.
    pub probabilities: Vec<f64>,
    /// Conservative L1 certificate (0 when all keys were read).
    pub certified_l1_bound: f64,
    /// Caller budget.
    pub epsilon: f64,
    /// Whether each token was read.
    pub read: Vec<bool>,
}

struct Node {
    members: Vec<usize>,
    nmax: f64,
    nmin: f64,
    centroid: Vec<f64>,
    cos_theta: f64,
    children: Option<(usize, usize)>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

struct Builder<'a> {
    keys: &'a [f64],
    d: usize,
    norms: Vec<f64>,
    leaf_size: usize,
    nodes: Vec<Node>,
    split_dots: usize,
}

#[allow(clippy::many_single_char_names)]
impl Builder<'_> {
    fn unit(&self, i: usize) -> Vec<f64> {
        self.keys[i * self.d..(i + 1) * self.d]
            .iter()
            .map(|v| v / self.norms[i])
            .collect()
    }

    fn make(&mut self, members: Vec<usize>) -> usize {
        let nmax = members.iter().map(|&i| self.norms[i]).fold(0.0, f64::max);
        let nmin = members
            .iter()
            .map(|&i| self.norms[i])
            .fold(f64::INFINITY, f64::min);
        let mut sum = vec![0.0; self.d];
        let units: Vec<Vec<f64>> = members
            .iter()
            .filter(|&&i| self.norms[i] > 0.0)
            .map(|&i| self.unit(i))
            .collect();
        for u in &units {
            for (s, v) in sum.iter_mut().zip(u) {
                *s += v;
            }
        }
        let sn = norm(&sum);
        let (centroid, cos_theta) = if sn > 0.0 && sn.is_finite() {
            let c: Vec<f64> = sum.iter().map(|v| v / sn).collect();
            let ct = units.iter().map(|u| dot(&c, u)).fold(1.0, f64::min);
            (c, ct)
        } else {
            (Vec::new(), -1.0)
        };
        let id = self.nodes.len();
        self.nodes.push(Node {
            members,
            nmax,
            nmin,
            centroid,
            cos_theta,
            children: None,
        });
        id
    }

    #[allow(clippy::needless_pass_by_value)]
    fn build(&mut self, members: Vec<usize>) -> usize {
        let id = self.make(members.clone());
        if members.len() <= self.leaf_size {
            return id;
        }
        // Farthest-pair seeds (unit directions; zero keys never reach here).
        let a = self.unit(members[0]);
        let mut sb = members[0];
        let mut best = f64::INFINITY;
        for &i in &members {
            self.split_dots += 1;
            let c = dot(&a, &self.unit(i));
            if c < best {
                best = c;
                sb = i;
            }
        }
        let b = self.unit(sb);
        let mut sc = members[0];
        best = f64::INFINITY;
        for &i in &members {
            self.split_dots += 1;
            let c = dot(&b, &self.unit(i));
            if c < best {
                best = c;
                sc = i;
            }
        }
        let c = self.unit(sc);
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for &i in &members {
            let u = self.unit(i);
            self.split_dots += 2;
            if dot(&b, &u) >= dot(&c, &u) {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        if left.is_empty() || right.is_empty() {
            let half = members.len() / 2;
            left = members[..half].to_vec();
            right = members[half..].to_vec();
        }
        let l = self.build(left);
        let r = self.build(right);
        self.nodes[id].children = Some((l, r));
        id
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    leaf_size: usize,
) -> Result<usize, &'static str> {
    if head_dim == 0 {
        return Err("ADA-A3 cone-tree softmax head_dim must be positive");
    }
    if query.len() != head_dim {
        return Err("ADA-A3 cone-tree softmax query must contain head_dim elements");
    }
    if keys.is_empty() || keys.len() % head_dim != 0 {
        return Err("ADA-A3 cone-tree softmax keys must be non-empty row-major [n, head_dim]");
    }
    if query.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err("ADA-A3 cone-tree softmax inputs must be finite");
    }
    if !score_scale.is_finite() || score_scale <= 0.0 {
        return Err("ADA-A3 cone-tree softmax score_scale must be finite and positive");
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("ADA-A3 cone-tree softmax epsilon must be finite and non-negative");
    }
    if leaf_size == 0 {
        return Err("ADA-A3 cone-tree softmax leaf_size must be positive");
    }
    Ok(keys.len() / head_dim)
}

fn node_bound(nd: &Node, q: &[f64], q_norm: f64, scale: f64, cost: &mut ConeTreeCost) -> f64 {
    if nd.nmax == 0.0 {
        return 0.0;
    }
    let mut g = 1.0;
    if !nd.centroid.is_empty() && q_norm > 0.0 && nd.cos_theta > SLACK {
        cost.query_time_centroid_dots += 1;
        let cos_phi = (dot(q, &nd.centroid) / q_norm).clamp(-1.0, 1.0);
        let cos_wide = nd.cos_theta - SLACK;
        if cos_phi < cos_wide {
            let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
            let sin_wide = (1.0 - cos_wide * cos_wide).max(0.0).sqrt();
            g = (cos_phi * cos_wide + sin_phi * sin_wide + SLACK).min(1.0);
        }
    }
    let n_used = if g >= 0.0 { nd.nmax } else { nd.nmin };
    let v = scale * q_norm * n_used * g;
    if v >= 0.0 {
        v * (1.0 + BOUND_INFLATION)
    } else {
        v * (1.0 - BOUND_INFLATION)
    }
}

/// Evaluate the cone-tree certified softmax with logical costs.
///
/// # Errors
///
/// Returns an error for invalid layout, non-finite input, invalid
/// `score_scale`, `epsilon` or `leaf_size` (`>= 1`), or non-finite scores/mass.
#[must_use = "the certified distribution should be checked"]
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub fn certified_cone_tree_softmax_accounted(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    leaf_size: usize,
) -> Result<(ConeTreeSoftmax, ConeTreeCost), &'static str> {
    let n = validate(query, keys, head_dim, score_scale, epsilon, leaf_size)?;
    let mut cost = ConeTreeCost::default();
    let norms: Vec<f64> = keys.chunks_exact(head_dim).map(norm).collect();
    let mut b = Builder {
        keys,
        d: head_dim,
        norms,
        leaf_size,
        nodes: Vec::new(),
        split_dots: 0,
    };
    let nonzero: Vec<usize> = (0..n).filter(|&i| b.norms[i] > 0.0).collect();
    let zero: Vec<usize> = (0..n).filter(|&i| b.norms[i] == 0.0).collect();
    let mut roots = Vec::new();
    if !nonzero.is_empty() {
        roots.push(b.build(nonzero));
    }
    if !zero.is_empty() {
        roots.push(b.make(zero));
    }
    cost.nodes = b.nodes.len();
    cost.prefill_split_dots = b.split_dots;
    let nodes = b.nodes;

    let q_norm = norm(query);
    let mut frontier: Vec<(f64, usize)> = roots
        .iter()
        .map(|&r| {
            (
                node_bound(&nodes[r], query, q_norm, score_scale, &mut cost),
                r,
            )
        })
        .collect();
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let certificate;
    loop {
        let max_read = scores
            .iter()
            .map(|&(_, s)| s)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_u = frontier
            .iter()
            .map(|&(u, _)| u)
            .fold(f64::NEG_INFINITY, f64::max);
        let m = max_read.max(max_u);
        #[allow(clippy::cast_precision_loss)]
        let tail: f64 = frontier
            .iter()
            .map(|&(u, j)| nodes[j].members.len() as f64 * (u - m).exp())
            .sum();
        cost.exp_evaluations += frontier.len();
        if frontier.is_empty() {
            certificate = 0.0;
            break;
        }
        let mass: f64 = scores.iter().map(|&(_, s)| (s - m).exp()).sum();
        cost.exp_evaluations += scores.len();
        let cert = if tail.is_finite() && mass + tail > 0.0 {
            2.0 * tail / (mass + tail)
        } else {
            2.0
        };
        if cert <= epsilon {
            certificate = cert;
            break;
        }
        // Expand the frontier node with the largest bound (ties: lowest id).
        let mut pick = 0;
        for k in 1..frontier.len() {
            let (u, j) = frontier[k];
            let (bu, bj) = frontier[pick];
            if u.total_cmp(&bu).then(bj.cmp(&j)).is_gt() {
                pick = k;
            }
        }
        let (_, id) = frontier.swap_remove(pick);
        if let Some((l, r)) = nodes[id].children {
            for c in [l, r] {
                let u = node_bound(&nodes[c], query, q_norm, score_scale, &mut cost);
                frontier.push((u, c));
            }
        } else {
            for &idx in &nodes[id].members {
                let row = &keys[idx * head_dim..(idx + 1) * head_dim];
                let score = score_scale * dot(query, row);
                if !score.is_finite() {
                    return Err("ADA-A3 cone-tree softmax scores must be finite");
                }
                cost.query_time_key_rows_read += 1;
                scores.push((idx, score));
            }
        }
    }
    cost.key_rows_skipped = n - scores.len();
    let m = scores
        .iter()
        .map(|&(_, s)| s)
        .fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = scores.iter().map(|&(_, s)| (s - m).exp()).sum();
    cost.exp_evaluations += scores.len();
    if scores.is_empty() || !total.is_finite() || total <= 0.0 {
        return Err("ADA-A3 cone-tree softmax mass must be positive and finite");
    }
    let mut probabilities = vec![0.0; n];
    let mut read = vec![false; n];
    for &(i, s) in &scores {
        probabilities[i] = (s - m).exp() / total;
        read[i] = true;
    }
    Ok((
        ConeTreeSoftmax {
            probabilities,
            certified_l1_bound: certificate,
            epsilon,
            read,
        },
        cost,
    ))
}

/// Evaluate the cone-tree certified softmax.
///
/// # Errors
///
/// Same failure conditions as [`certified_cone_tree_softmax_accounted`].
#[must_use = "the certified distribution should be checked"]
pub fn certified_cone_tree_softmax(
    query: &[f64],
    keys: &[f64],
    head_dim: usize,
    score_scale: f64,
    epsilon: f64,
    leaf_size: usize,
) -> Result<ConeTreeSoftmax, &'static str> {
    certified_cone_tree_softmax_accounted(query, keys, head_dim, score_scale, epsilon, leaf_size)
        .map(|(r, _)| r)
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::cloned_instead_of_copied,
    clippy::single_match,
    clippy::too_many_lines
)]
mod tests {
    use super::*;

    fn oracle(q: &[f64], keys: &[f64], d: usize, s: f64) -> Vec<f64> {
        let z: Vec<f64> = keys
            .chunks_exact(d)
            .map(|k| s * k.iter().zip(q).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
        let t: f64 = e.iter().sum();
        e.iter().map(|v| v / t).collect()
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
            #[allow(clippy::cast_precision_loss)]
            let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            u
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.next().max(1e-300);
            let u2 = self.next();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    fn check(q: &[f64], keys: &[f64], d: usize, s: f64, eps: f64, leaf: usize) -> ConeTreeCost {
        let (r, c) = certified_cone_tree_softmax_accounted(q, keys, d, s, eps, leaf).unwrap();
        let o = oracle(q, keys, d, s);
        let e = l1(&r.probabilities, &o);
        assert!(
            e <= r.certified_l1_bound + 1e-12,
            "L1 {e} > cert {}",
            r.certified_l1_bound
        );
        assert!(r.certified_l1_bound <= eps + 1e-15 || r.certified_l1_bound == 0.0);
        assert_eq!(c.query_time_key_rows_read + c.key_rows_skipped, o.len());
        assert_eq!(
            r.read.iter().filter(|&&b| b).count(),
            c.query_time_key_rows_read
        );
        c
    }

    #[test]
    fn rejects_invalid() {
        let f = certified_cone_tree_softmax;
        assert!(f(&[1.0], &[1.0], 0, 1.0, 0.1, 1).is_err());
        assert!(f(&[1.0, 2.0], &[1.0], 1, 1.0, 0.1, 1).is_err());
        assert!(f(&[1.0], &[], 1, 1.0, 0.1, 1).is_err());
        assert!(f(&[1.0], &[1.0, 2.0, 3.0], 2, 1.0, 0.1, 1).is_err());
        assert!(f(&[f64::NAN], &[1.0], 1, 1.0, 0.1, 1).is_err());
        assert!(f(&[1.0], &[f64::INFINITY], 1, 1.0, 0.1, 1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 0.0, 0.1, 1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, -0.1, 1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, f64::NAN, 1).is_err());
        assert!(f(&[1.0], &[1.0], 1, 1.0, 0.1, 0).is_err());
    }

    #[test]
    fn degenerate_inputs_are_exact_or_finite() {
        // singleton, equal keys, zero query, zero keys, overflow
        check(&[1.0, 0.0], &[3.0, 4.0], 2, 1.0, 0.1, 1);
        check(&[1.0, 0.0], &[1.0, 0.0].repeat(8), 2, 1.0, 1e-3, 1);
        check(
            &[0.0, 0.0],
            &[1.0, 2.0, -1.0, 0.5, 3.0, 3.0],
            2,
            1.0,
            1e-3,
            1,
        );
        check(&[1.0, 1.0], &[0.0; 12], 2, 1.0, 1e-3, 2);
        let r = certified_cone_tree_softmax(&[1.0], &[1e200, -1e200, 5.0], 1, 1.0, 0.1, 1);
        match r {
            Ok(r) => assert!(r.probabilities.iter().all(|p| p.is_finite())),
            Err(_) => {}
        }
    }

    #[test]
    fn witness_prunes_subtrees_with_few_dots() {
        // 4 equal-norm clusters of 32 in d=4; query aligned with cluster 0.
        let d = 4;
        let mut keys = Vec::new();
        for c in 0..4 {
            for j in 0..32 {
                let mut k = vec![0.0; d];
                k[c] = 8.0;
                k[(c + 1) % d] = 0.01 * j as f64;
                keys.extend(k);
            }
        }
        let q = [1.0, 0.0, 0.0, 0.0];
        let c = check(&q, &keys, d, 4.0, 1e-3, 4);
        let total = c.query_time_key_rows_read + c.query_time_centroid_dots;
        assert!(total < 128 / 2, "total dots {total}");
    }

    #[test]
    fn exhaustive_grid_has_no_violation() {
        let alphabet: [[f64; 2]; 5] = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-1.0, 1.0], [3.0, 0.5]];
        let queries: [[f64; 2]; 3] = [[1.0, 0.0], [-0.5, 1.5], [0.0, 0.0]];
        let mut cases = 0;
        let mut skipped = 0;
        for len in 1..=5usize {
            let total = 5usize.pow(len as u32);
            for code in 0..total {
                let mut keys = Vec::new();
                let mut x = code;
                for _ in 0..len {
                    keys.extend(alphabet[x % 5]);
                    x /= 5;
                }
                for q in &queries {
                    for &eps in &[0.0, 1e-3, 0.3] {
                        for &leaf in &[1usize, 2] {
                            let c = check(q, &keys, 2, 1.0, eps, leaf);
                            cases += 1;
                            if c.key_rows_skipped > 0 {
                                skipped += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 50_000);
        assert!(skipped > 0);
        eprintln!("grid cases={cases} skipped={skipped}");
    }

    #[test]
    fn random_adversarial_sweep_has_no_violation() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..400 {
            let d = 1 + (rng.next() * 8.0) as usize;
            let n = 1 + (rng.next() * 60.0) as usize;
            let heavy = rng.next() < 0.5;
            let keys: Vec<f64> = (0..n * d)
                .map(|_| {
                    let g = rng.gauss();
                    if heavy {
                        g * (4.0 * rng.gauss()).exp()
                    } else {
                        g
                    }
                })
                .collect();
            let q: Vec<f64> = (0..d).map(|_| rng.gauss() * 3.0).collect();
            let s = 0.1 + rng.next() * 2.0;
            let leaf = 1 + (rng.next() * 8.0) as usize;
            let eps = [0.0, 1e-6, 1e-2, 0.5][(rng.next() * 4.0) as usize];
            check(&q, &keys, d, s, eps, leaf);
        }
    }

    #[test]
    fn eps_zero_reads_everything_or_certifies_zero() {
        let mut rng = Rng(42);
        let d = 4;
        let keys: Vec<f64> = (0..64 * d).map(|_| rng.gauss()).collect();
        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
        let (r, c) = certified_cone_tree_softmax_accounted(&q, &keys, d, 1.0, 0.0, 4).unwrap();
        assert_eq!(c.query_time_key_rows_read, 64);
        assert_eq!(r.certified_l1_bound, 0.0);
    }

    /// Regime report (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "reports regime numbers; not a gate"]
    fn regime_report() {
        let d = 16;
        let n = 256;
        for (name, jitter) in [("iid", -1.0), ("j0.05", 0.05), ("j0.2", 0.2), ("j0.5", 0.5)] {
            for &eps in &[1e-4, 1e-2, 1e-1] {
                for &leaf in &[4usize, 16] {
                    let (mut rows, mut tot, mut maxl1, mut maxcert) = (0.0, 0.0, 0.0f64, 0.0f64);
                    let draws = 40;
                    for t in 0..draws {
                        let mut rng = Rng(1000 + t as u64 * 7919);
                        let centers: Vec<Vec<f64>> = (0..8)
                            .map(|_| {
                                let v: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                                let nn = norm(&v);
                                v.iter().map(|x| 4.0 * x / nn).collect()
                            })
                            .collect();
                        let mut keys = Vec::new();
                        for i in 0..n {
                            if jitter < 0.0 {
                                keys.extend((0..d).map(|_| rng.gauss()));
                            } else {
                                let c = &centers[i / 32];
                                keys.extend(
                                    c.iter().map(|x| {
                                        x + 4.0 * jitter * rng.gauss() / (d as f64).sqrt()
                                    }),
                                );
                            }
                        }
                        let q: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                        let s = 1.0 / (d as f64).sqrt();
                        let (r, c) =
                            certified_cone_tree_softmax_accounted(&q, &keys, d, s, eps, leaf)
                                .unwrap();
                        let o = oracle(&q, &keys, d, s);
                        maxl1 = maxl1.max(l1(&r.probabilities, &o));
                        maxcert = maxcert.max(r.certified_l1_bound);
                        rows += c.query_time_key_rows_read as f64 / n as f64;
                        tot += (c.query_time_key_rows_read + c.query_time_centroid_dots) as f64
                            / n as f64;
                    }
                    eprintln!(
                        "REPORT {name} eps={eps} leaf={leaf} rows={:.3} total={:.3} maxL1={maxl1:.2e} maxcert={maxcert:.2e}",
                        rows / draws as f64,
                        tot / draws as f64
                    );
                }
            }
        }
    }
}
