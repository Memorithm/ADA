//! ADA1 research path: exact rational alpha = 1 + 1/k entmax on integer scores.
//!
//! With `alpha = 1 + 1/k` the entmax rule is `p_i = ((z_i - u) / k)^k` on the
//! support, with `u = k * tau`. The normalization `sum p_i = 1` over a support
//! prefix of size `rho` is the integer-coefficient polynomial equation
//! `sum (z_i - u)^k = k^k` of degree `k` and leading coefficient `±rho`.
//! By the rational root theorem any rational root `u = a / b` has `b | rho`.
//! For each divisor `b` the integer equation `sum (b z_i - a)^k = (k b)^k`
//! is strictly monotone in `a` on the admissible window, so an exact integer
//! binary search decides whether a root exists. No floating point is used.
//!
//! `k = 1` is sparsemax (`alpha = 2`), `k = 2` is 1.5-entmax; softmax is the
//! `k -> infinity` limit. Inputs without a rational threshold on the true
//! support, or whose checked `i128` arithmetic overflows, fail closed.
//! `ada_a4_entmax_bnb::dense_entmax` remains the independent oracle. No
//! usefulness, novelty, or FLAT adoption claim is made.

use crate::exact::{ExactDistribution, ExactRational};

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LatticeEntmaxCost {
    /// Scores sorted.
    pub scores_sorted: usize,
    /// Support prefixes examined.
    pub prefixes_tested: usize,
    /// Denominator candidates `b | rho` examined.
    pub denominators_tested: usize,
    /// Evaluations of the integer power sum.
    pub power_sum_evaluations: usize,
}

const MSG_EMPTY: &str = "ADA-A6 lattice entmax needs a non-empty score vector";
const MSG_K: &str = "ADA-A6 lattice entmax needs 1 <= k <= 16";
const MSG_OVERFLOW: &str = "ADA-A6 lattice entmax integer arithmetic overflowed";
const MSG_NONE: &str = "ADA-A6 lattice entmax has no rational threshold on the true support";
const MSG_AMBIG: &str = "ADA-A6 lattice entmax found two admissible prefixes";

fn ipow(base: i128, k: u32) -> Result<i128, &'static str> {
    base.checked_pow(k).ok_or(MSG_OVERFLOW)
}

fn power_sum(
    sorted: &[i64],
    rho: usize,
    a: i128,
    b: i128,
    k: u32,
    cost: &mut LatticeEntmaxCost,
) -> Result<i128, &'static str> {
    cost.power_sum_evaluations += 1;
    let mut total = 0_i128;
    for &z in &sorted[..rho] {
        let w = b
            .checked_mul(i128::from(z))
            .and_then(|v| v.checked_sub(a))
            .ok_or(MSG_OVERFLOW)?;
        total = total.checked_add(ipow(w, k)?).ok_or(MSG_OVERFLOW)?;
    }
    Ok(total)
}

/// Exact `alpha = 1 + 1/k` entmax for integer scores.
///
/// # Errors
///
/// Fails closed on empty input, `k` outside `1..=16`, overflow, or when the
/// true support has no rational threshold.
pub fn lattice_entmax_i64(scores: &[i64], k: u32) -> Result<ExactDistribution, &'static str> {
    lattice_entmax_i64_accounted(scores, k).map(|(d, _)| d)
}

/// As [`lattice_entmax_i64`], also returning logical counters.
///
/// # Errors
///
/// Same as [`lattice_entmax_i64`].
pub fn lattice_entmax_i64_accounted(
    scores: &[i64],
    k: u32,
) -> Result<(ExactDistribution, LatticeEntmaxCost), &'static str> {
    if scores.is_empty() {
        return Err(MSG_EMPTY);
    }
    if !(1..=16).contains(&k) {
        return Err(MSG_K);
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&x, &y| scores[y].cmp(&scores[x]).then(x.cmp(&y)));
    let sorted: Vec<i64> = order.iter().map(|&i| scores[i]).collect();
    let mut cost = LatticeEntmaxCost {
        scores_sorted: sorted.len(),
        ..LatticeEntmaxCost::default()
    };
    let kk = i128::from(k);
    let target_base = ipow(kk, k)?;
    let top = i128::from(sorted[0]);
    let mut found: Option<(usize, i128, i128)> = None;
    for rho in 1..=sorted.len() {
        cost.prefixes_tested += 1;
        let low_score = i128::from(sorted[rho - 1]);
        for b in 1..=rho {
            if rho % b != 0 {
                continue;
            }
            cost.denominators_tested += 1;
            let bi = i128::try_from(b).map_err(|_| MSG_OVERFLOW)?;
            let target = ipow(kk.checked_mul(bi).ok_or(MSG_OVERFLOW)?, k)?;
            debug_assert!(target >= target_base);
            // Window: u in [top - k, low_score): a in [b(top-k), b*low-1].
            let mut lo = bi.checked_mul(top - kk).ok_or(MSG_OVERFLOW)?;
            let mut hi = bi.checked_mul(low_score).ok_or(MSG_OVERFLOW)? - 1;
            if lo > hi {
                continue;
            }
            // Smallest a in [lo, hi] with power_sum <= target (decreasing in a).
            if power_sum(&sorted, rho, hi, bi, k, &mut cost)? > target {
                continue;
            }
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if power_sum(&sorted, rho, mid, bi, k, &mut cost)? <= target {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            if power_sum(&sorted, rho, lo, bi, k, &mut cost)? != target {
                continue;
            }
            let excluded_ok = sorted
                .get(rho)
                .is_none_or(|&z| bi.checked_mul(i128::from(z)).is_some_and(|v| v <= lo));
            if !excluded_ok {
                continue;
            }
            if let Some((prev_rho, pa, pb)) = found {
                let same = prev_rho == rho && pa * bi == lo * pb;
                if !same {
                    return Err(MSG_AMBIG);
                }
            } else {
                found = Some((rho, lo, bi));
            }
            break;
        }
    }
    let (rho, a, b) = found.ok_or(MSG_NONE)?;
    let denom = ipow(kk.checked_mul(b).ok_or(MSG_OVERFLOW)?, k)?;
    let mut probabilities = vec![ExactRational::reduced(0, 1)?; scores.len()];
    for (pos, &original) in order.iter().enumerate().take(rho) {
        let w = b
            .checked_mul(i128::from(sorted[pos]))
            .and_then(|v| v.checked_sub(a))
            .ok_or(MSG_OVERFLOW)?;
        probabilities[original] = ExactRational::reduced(ipow(w, k)?, denom)?;
    }
    let tau = ExactRational::reduced(a, kk.checked_mul(b).ok_or(MSG_OVERFLOW)?)?;
    Ok((
        ExactDistribution {
            probabilities,
            tau,
            support_size: rho,
        },
        cost,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{entmax15_exact_i64, sparsemax_exact_i64};
    use ada_a4_entmax_bnb::dense_entmax;

    fn rat(n: i128, d: i128) -> ExactRational {
        ExactRational::reduced(n, d).unwrap()
    }

    fn to_f(r: ExactRational) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = r.numerator() as f64 / r.denominator() as f64;
        v
    }

    fn vectors(max_len: usize, max_score: i64) -> Vec<Vec<i64>> {
        let mut out = Vec::new();
        for len in 1..=max_len {
            let mut v = vec![0_i64; len];
            loop {
                out.push(v.clone());
                let mut i = 0;
                while i < len {
                    if v[i] < max_score {
                        v[i] += 1;
                        break;
                    }
                    v[i] = 0;
                    i += 1;
                }
                if i == len {
                    break;
                }
            }
        }
        out
    }

    /// Returns (accepted, rejected) and checks every accepted vector.
    fn grid(k: u32, max_len: usize, max_score: i64) -> (usize, usize) {
        let (mut ok, mut rej) = (0, 0);
        for v in vectors(max_len, max_score) {
            match lattice_entmax_i64(&v, k) {
                Ok(d) => {
                    ok += 1;
                    #[allow(clippy::cast_precision_loss)]
                    let scores: Vec<f64> = v.iter().map(|&z| z as f64).collect();
                    let alpha = 1.0 + 1.0 / f64::from(k);
                    let oracle = dense_entmax(&scores, alpha).unwrap();
                    let mut total = 0.0;
                    for (p, q) in d.probabilities.iter().zip(&oracle.probabilities) {
                        assert!((to_f(*p) - q).abs() <= 1e-9, "{v:?} k={k}");
                        total += to_f(*p);
                    }
                    assert!((total - 1.0).abs() <= 1e-12, "{v:?} k={k}");
                    assert!((to_f(d.tau) - oracle.tau).abs() <= 1e-9, "{v:?} k={k}");
                }
                Err(e) => {
                    assert_eq!(e, MSG_NONE, "{v:?} k={k}");
                    rej += 1;
                }
            }
        }
        (ok, rej)
    }

    #[test]
    fn k1_is_exact_sparsemax_everywhere() {
        for v in vectors(4, 4) {
            let a = lattice_entmax_i64(&v, 1).unwrap();
            let b = sparsemax_exact_i64(&v).unwrap();
            assert_eq!(a.probabilities, b.probabilities, "{v:?}");
            assert_eq!(a.tau, b.tau, "{v:?}");
        }
    }

    #[test]
    fn k2_matches_discriminant_path_domain_exactly() {
        for v in vectors(5, 3) {
            let a = lattice_entmax_i64(&v, 2);
            let b = entmax15_exact_i64(&v);
            assert_eq!(a.is_ok(), b.is_ok(), "{v:?}");
            if let (Ok(a), Ok(b)) = (a, b) {
                assert_eq!(a.probabilities, b.probabilities, "{v:?}");
                assert_eq!(a.tau, b.tau, "{v:?}");
            }
        }
    }

    #[test]
    fn k3_fixtures() {
        let d = lattice_entmax_i64(&[0; 8], 3).unwrap();
        assert!(d.probabilities.iter().all(|p| *p == rat(1, 8)));
        assert_eq!(d.tau, rat(-1, 2));
        let d = lattice_entmax_i64(&[3, 0], 3).unwrap();
        assert_eq!(d.probabilities, vec![rat(1, 1), rat(0, 1)]);
        assert_eq!(d.tau, rat(0, 1));
        assert_eq!(lattice_entmax_i64(&[0; 4], 3), Err(MSG_NONE));
        assert_eq!(lattice_entmax_i64(&[0, 0], 3), Err(MSG_NONE));
    }

    #[test]
    fn grids_agree_with_independent_oracle() {
        let (len, score) = if cfg!(miri) { (3, 2) } else { (5, 4) };
        for k in 1..=4 {
            let (ok, rej) = grid(k, len, score);
            assert!(ok > 0 && ok + rej == vectors(len, score).len());
            if !cfg!(miri) {
                println!("lattice-grid k={k} accepted={ok} rejected={rej}");
            }
        }
    }

    #[test]
    fn permutation_and_translation_invariance() {
        let base = [0_i64; 8];
        let shifted: Vec<i64> = base.iter().map(|z| z + 5).collect();
        let a = lattice_entmax_i64(&base, 3).unwrap();
        let b = lattice_entmax_i64(&shifted, 3).unwrap();
        assert_eq!(a.probabilities, b.probabilities);
        assert_eq!(b.tau, rat(7, 6));
    }

    #[test]
    fn adversarial_inputs_fail_closed() {
        assert_eq!(lattice_entmax_i64(&[], 2), Err(MSG_EMPTY));
        assert_eq!(lattice_entmax_i64(&[1], 0), Err(MSG_K));
        assert_eq!(lattice_entmax_i64(&[1], 17), Err(MSG_K));
        let wide = lattice_entmax_i64(&[i64::MAX, i64::MIN], 16).unwrap();
        assert_eq!(wide.probabilities, vec![rat(1, 1), rat(0, 1)]);
        assert!(lattice_entmax_i64(&[i64::MAX, i64::MAX], 16).is_err());
        assert!(lattice_entmax_i64(&[i64::MAX], 3).is_ok());
    }

    #[test]
    fn cost_counters_are_logical() {
        let (_, c) = lattice_entmax_i64_accounted(&[0; 8], 3).unwrap();
        assert_eq!(c.scores_sorted, 8);
        assert!(c.prefixes_tested >= 8 && c.power_sum_evaluations > 0);
    }

    #[test]
    fn k3_has_unequal_multi_support_members() {
        let mut found = Vec::new();
        for v in vectors(5, 4) {
            if let Ok(d) = lattice_entmax_i64(&v, 3) {
                let mut top: Vec<i64> = v
                    .iter()
                    .zip(&d.probabilities)
                    .filter(|(_, p)| p.numerator() > 0)
                    .map(|(z, _)| *z)
                    .collect();
                top.sort_unstable();
                top.dedup();
                if top.len() > 1 {
                    found.push(v);
                }
            }
        }
        println!("k3 unequal multi-support members: {}", found.len());
        assert!(!found.is_empty() || cfg!(miri));
        println!("example: {:?}", found.first());
    }
}
