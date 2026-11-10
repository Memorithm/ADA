//! ADA7 research path: support-pruned composable exact 1.5-entmax.
//!
//! For alpha = 1.5 with `p_i = (z_i - t)_+^2 / 4`, the normalizer
//! `F(t) = sum_i (z_i - t)_+^2 / 4` is decreasing in `t`, and adding scores can
//! only increase `F`. Hence the global threshold of a union is at least every
//! part-local threshold, and a score at or below its part-local threshold is
//! outside the global support. A part may therefore be replaced by its own
//! local support (a multiset of plateaus) without changing the merged result.
//!
//! Local support is decided with exact integer inequalities, with no
//! perfect-square requirement: only the final solve needs the rational domain
//! of `entmax15_exact_i64`, and it fails closed outside it. This is a
//! reference-arithmetic identity, not a usefulness, novelty, or FLAT claim.

use std::collections::BTreeMap;

use ada_a6_tau_solvers::{ExactDistribution, entmax15_exact_i64};

/// Largest expanded multiset the final exact solve will materialize.
pub const MAX_EXPANDED_SOLVE: u64 = 1 << 16;

/// Logical counters (not hardware measurements).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PruneCost {
    /// Distinct input plateaus examined.
    pub levels_examined: usize,
    /// Plateau-prefix support tests performed.
    pub prefix_tests: usize,
    /// Scores (with multiplicity) discarded as provably outside the support.
    pub scores_discarded: u64,
}

/// Support-pruned multiset summary for 1.5-entmax.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entmax15Summary {
    levels: BTreeMap<i64, u64>,
}

impl Entmax15Summary {
    /// Summarize scores, keeping only the local support.
    ///
    /// # Errors
    ///
    /// Empty input or checked-arithmetic overflow.
    pub fn from_scores(scores: &[i64]) -> Result<Self, &'static str> {
        let mut levels = BTreeMap::new();
        for &score in scores {
            let count = levels.entry(score).or_insert(0_u64);
            *count = count
                .checked_add(1)
                .ok_or("ADA-A7 entmax15 count overflow")?;
        }
        Self::prune(&levels).map(|(summary, _)| summary)
    }

    /// Like [`Self::from_scores`] with cost counters.
    ///
    /// # Errors
    ///
    /// Empty input or checked-arithmetic overflow.
    pub fn from_scores_accounted(scores: &[i64]) -> Result<(Self, PruneCost), &'static str> {
        let mut levels = BTreeMap::new();
        for &score in scores {
            let count = levels.entry(score).or_insert(0_u64);
            *count = count
                .checked_add(1)
                .ok_or("ADA-A7 entmax15 count overflow")?;
        }
        Self::prune(&levels)
    }

    /// Commutative merge followed by re-pruning.
    ///
    /// # Errors
    ///
    /// Overflow in counts or moments.
    pub fn compose(left: &Self, right: &Self) -> Result<Self, &'static str> {
        Self::compose_accounted(left, right).map(|(summary, _)| summary)
    }

    /// Merge with cost counters.
    ///
    /// # Errors
    ///
    /// Overflow in counts or moments.
    pub fn compose_accounted(left: &Self, right: &Self) -> Result<(Self, PruneCost), &'static str> {
        let mut levels = left.levels.clone();
        for (&score, &count) in &right.levels {
            let entry = levels.entry(score).or_insert(0);
            *entry = entry
                .checked_add(count)
                .ok_or("ADA-A7 entmax15 count overflow")?;
        }
        Self::prune(&levels)
    }

    /// Retained plateaus as `(score, count)`, descending.
    #[must_use]
    pub fn plateaus(&self) -> Vec<(i64, u64)> {
        self.levels.iter().rev().map(|(&s, &c)| (s, c)).collect()
    }

    /// Retained scores with multiplicity.
    #[must_use]
    pub fn retained_count(&self) -> u64 {
        self.levels.values().sum()
    }

    /// Exact solve on the retained multiset, in descending-plateau order.
    ///
    /// # Errors
    ///
    /// Fails closed outside the rational discriminant domain or when the
    /// expanded multiset exceeds [`MAX_EXPANDED_SOLVE`].
    pub fn solve_exact(&self) -> Result<(Vec<i64>, ExactDistribution), &'static str> {
        if self.retained_count() > MAX_EXPANDED_SOLVE {
            return Err("ADA-A7 entmax15 expanded solve exceeds cap");
        }
        let mut expanded = Vec::new();
        for (score, count) in self.plateaus() {
            for _ in 0..count {
                expanded.push(score);
            }
        }
        let dist = entmax15_exact_i64(&expanded)?;
        Ok((expanded, dist))
    }

    fn prune(levels: &BTreeMap<i64, u64>) -> Result<(Self, PruneCost), &'static str> {
        if levels.is_empty() {
            return Err("ADA-A7 entmax15 requires at least one score");
        }
        let mut cost = PruneCost::default();
        let mut rho: i128 = 0;
        let mut s1: i128 = 0;
        let mut s2: i128 = 0;
        let mut keep_levels = 0_usize;
        for (index, (&score, &count)) in levels.iter().rev().enumerate() {
            cost.levels_examined += 1;
            let z = i128::from(score);
            let c = i128::from(count);
            rho = rho.checked_add(c).ok_or("ADA-A7 entmax15 overflow")?;
            s1 = s1
                .checked_add(c.checked_mul(z).ok_or("ADA-A7 entmax15 overflow")?)
                .ok_or("ADA-A7 entmax15 overflow")?;
            let zz = z.checked_mul(z).ok_or("ADA-A7 entmax15 overflow")?;
            s2 = s2
                .checked_add(c.checked_mul(zz).ok_or("ADA-A7 entmax15 overflow")?)
                .ok_or("ADA-A7 entmax15 overflow")?;
            cost.prefix_tests += 1;
            // D = S1^2 - rho*S2 + 4*rho ; valid iff D >= 0 and sqrt(D) > S1 - rho*z.
            let d = s1
                .checked_mul(s1)
                .and_then(|a| rho.checked_mul(s2).and_then(|b| a.checked_sub(b)))
                .and_then(|a| rho.checked_mul(4).and_then(|b| a.checked_add(b)))
                .ok_or("ADA-A7 entmax15 overflow")?;
            if d < 0 {
                continue;
            }
            let r = s1
                .checked_sub(rho.checked_mul(z).ok_or("ADA-A7 entmax15 overflow")?)
                .ok_or("ADA-A7 entmax15 overflow")?;
            let valid = r < 0 || d > r.checked_mul(r).ok_or("ADA-A7 entmax15 overflow")?;
            if valid {
                keep_levels = index + 1;
            }
        }
        if keep_levels == 0 {
            return Err("ADA-A7 entmax15 found an empty support");
        }
        let mut kept = BTreeMap::new();
        for (index, (&score, &count)) in levels.iter().rev().enumerate() {
            if index < keep_levels {
                kept.insert(score, count);
            } else {
                cost.scores_discarded += count;
            }
        }
        Ok((Self { levels: kept }, cost))
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use ada_a4_entmax_bnb::dense_entmax;

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    /// Independent oracle: f64 dense entmax(1.5) support must equal retained
    /// local support superset relation and probabilities must match.
    fn check_vs_f64_oracle(scores: &[i64], summary_scores: &[i64], dist: &ExactDistribution) {
        let f: Vec<f64> = summary_scores.iter().map(|&s| s as f64).collect();
        let oracle = dense_entmax(&f, 1.5).unwrap();
        for (p, q) in dist.probabilities.iter().zip(&oracle.probabilities) {
            assert!((p.to_f64().unwrap() - q).abs() < 1e-9);
        }
        // Full (unpruned) oracle: discarded scores must have ~0 mass.
        let full: Vec<f64> = scores.iter().map(|&s| s as f64).collect();
        let full_oracle = dense_entmax(&full, 1.5).unwrap();
        let total: f64 = full_oracle.probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        let positive_full = full_oracle
            .probabilities
            .iter()
            .filter(|&&p| p > 1e-12)
            .count();
        assert!(positive_full <= summary_scores.len());
    }

    fn expanded_distribution_by_score(
        scores: &[i64],
    ) -> Option<BTreeMap<i64, ada_a6_tau_solvers::ExactRational>> {
        let dist = entmax15_exact_i64(scores).ok()?;
        let mut map = BTreeMap::new();
        for (&s, &p) in scores.iter().zip(&dist.probabilities) {
            map.insert(s, p);
        }
        Some(map)
    }

    #[test]
    fn hand_case_two_parts_equal_concatenation() {
        // [1,0,0,0,0] is in the rational domain (p0 = 16/25).
        let a = Entmax15Summary::from_scores(&[1, 0]).unwrap();
        let b = Entmax15Summary::from_scores(&[0, 0, 0]).unwrap();
        let merged = Entmax15Summary::compose(&a, &b).unwrap();
        let (exp, dist) = merged.solve_exact().unwrap();
        assert_eq!(exp, vec![1, 0, 0, 0, 0]);
        assert_eq!(dist.probabilities[0].numerator(), 16);
        assert_eq!(dist.probabilities[0].denominator(), 25);
    }

    #[test]
    fn pruning_discards_far_scores() {
        let (s, cost) = Entmax15Summary::from_scores_accounted(&[100, 0, -50, -50]).unwrap();
        assert_eq!(s.plateaus(), vec![(100, 1)]);
        assert_eq!(cost.scores_discarded, 3);
        assert!(Entmax15Summary::from_scores(&[]).is_err());
    }

    #[test]
    fn exhaustive_split_equals_concatenation_and_oracle() {
        // all vectors length 1..=6 over {0..3}, every split point and swap.
        let mut checked = 0_u32;
        let mut rational = 0_u32;
        let mut pruned_somewhere = 0_u32;
        for len in 2..=6_usize {
            let total = 4_u32.pow(u32::try_from(len).unwrap());
            for code in 0..total {
                let mut v = Vec::new();
                let mut c = code;
                for _ in 0..len {
                    v.push(i64::from(c % 4));
                    c /= 4;
                }
                let full = Entmax15Summary::from_scores(&v).unwrap();
                if full.retained_count() < u64::try_from(len).unwrap() {
                    pruned_somewhere += 1;
                }
                for split in 1..len {
                    let l = Entmax15Summary::from_scores(&v[..split]).unwrap();
                    let r = Entmax15Summary::from_scores(&v[split..]).unwrap();
                    let lr = Entmax15Summary::compose(&l, &r).unwrap();
                    let rl = Entmax15Summary::compose(&r, &l).unwrap();
                    assert_eq!(lr, rl, "commutativity {v:?}");
                    assert_eq!(lr, full, "merge == direct support {v:?} @ {split}");
                    checked += 1;
                }
                // Rational-domain identity against the unpruned exact solver.
                if let Some(direct) = expanded_distribution_by_score(&v) {
                    let (exp, dist) = full.solve_exact().unwrap();
                    rational += 1;
                    for (s, p) in exp.iter().zip(&dist.probabilities) {
                        assert_eq!(direct[s], *p, "{v:?}");
                    }
                    check_vs_f64_oracle(&v, &exp, &dist);
                } else {
                    // pruned solve must not succeed where the unpruned one
                    // is out of domain with an identical support... it may,
                    // since pruning removes zero-mass scores only; but then
                    // probabilities must match the f64 oracle.
                    if let Ok((exp, dist)) = full.solve_exact() {
                        check_vs_f64_oracle(&v, &exp, &dist);
                    }
                }
            }
        }
        println!("ADA7-E15 splits={checked} rational={rational} pruned_vectors={pruned_somewhere}");
        assert!(checked > 0 && rational > 0 && pruned_somewhere > 0);
    }

    #[test]
    fn random_blocks_associativity_and_oracle() {
        let mut state = 7_u64;
        for _ in 0..300 {
            let blocks = 2 + (lcg(&mut state) % 4) as usize;
            let mut all = Vec::new();
            let mut parts = Vec::new();
            for _ in 0..blocks {
                let n = 1 + (lcg(&mut state) % 12) as usize;
                let v: Vec<i64> = (0..n).map(|_| (lcg(&mut state) % 9) as i64 - 4).collect();
                all.extend_from_slice(&v);
                parts.push(Entmax15Summary::from_scores(&v).unwrap());
            }
            let direct = Entmax15Summary::from_scores(&all).unwrap();
            let left_fold = parts[1..]
                .iter()
                .try_fold(parts[0].clone(), |a, b| Entmax15Summary::compose(&a, b))
                .unwrap();
            let right_fold = parts[..blocks - 1]
                .iter()
                .rev()
                .try_fold(parts[blocks - 1].clone(), |a, b| {
                    Entmax15Summary::compose(&a, b)
                })
                .unwrap();
            assert_eq!(left_fold, direct);
            assert_eq!(right_fold, direct);
            if let Ok((exp, dist)) = direct.solve_exact() {
                check_vs_f64_oracle(&all, &exp, &dist);
            }
        }
    }

    #[test]
    fn adversarial_extremes_and_overflow_fail_closed() {
        // Squares of i64 extremes overflow i128 moments: must fail closed.
        let a = Entmax15Summary::from_scores(&[i64::MAX]).unwrap();
        let b = Entmax15Summary::from_scores(&[i64::MIN]).unwrap();
        assert!(Entmax15Summary::compose(&a, &b).is_err());
        assert!(Entmax15Summary::from_scores(&[i64::MAX, i64::MAX - 1]).is_err());
        // Large but safe magnitudes prune the far score and keep the top.
        let big = 1_i64 << 40;
        let c = Entmax15Summary::from_scores(&[big, 0]).unwrap();
        assert_eq!(c.plateaus(), vec![(big, 1)]);
        // Equal plateau of 9 zeros (4*9 is a square): atomic, p = 1/9.
        let z = Entmax15Summary::from_scores(&[0; 9]).unwrap();
        assert_eq!(z.plateaus(), vec![(0, 9)]);
        let (_, d) = z.solve_exact().unwrap();
        assert!(
            d.probabilities
                .iter()
                .all(|p| p.numerator() == 1 && p.denominator() == 9)
        );
        // Outside the rational domain: pruned summary exists, solve fails closed.
        let o = Entmax15Summary::from_scores(&[0, 0]).unwrap();
        assert!(o.solve_exact().is_err());
    }

    #[test]
    fn domain_is_not_closed_under_composition() {
        // Negative result: both parts rational, union not (or vice versa).
        let mut part_ok_union_bad = 0_u32;
        let mut union_ok_part_bad = 0_u32;
        let mut pairs = 0_u32;
        for ll in 1..=3_usize {
            for lr in 1..=3_usize {
                for lc in 0..4_u32.pow(u32::try_from(ll).unwrap()) {
                    for rc in 0..4_u32.pow(u32::try_from(lr).unwrap()) {
                        let dec = |mut c: u32, n: usize| -> Vec<i64> {
                            (0..n)
                                .map(|_| {
                                    let d = i64::from(c % 4);
                                    c /= 4;
                                    d
                                })
                                .collect()
                        };
                        let l = dec(lc, ll);
                        let r = dec(rc, lr);
                        let mut u = l.clone();
                        u.extend_from_slice(&r);
                        let lo = Entmax15Summary::from_scores(&l)
                            .unwrap()
                            .solve_exact()
                            .is_ok();
                        let ro = Entmax15Summary::from_scores(&r)
                            .unwrap()
                            .solve_exact()
                            .is_ok();
                        let uo = Entmax15Summary::from_scores(&u)
                            .unwrap()
                            .solve_exact()
                            .is_ok();
                        pairs += 1;
                        if lo && ro && !uo {
                            part_ok_union_bad += 1;
                        }
                        if uo && !(lo && ro) {
                            union_ok_part_bad += 1;
                        }
                    }
                }
            }
        }
        println!(
            "ADA7-E15 domain pairs={pairs} parts_ok_union_bad={part_ok_union_bad} union_ok_parts_bad={union_ok_part_bad}"
        );
        assert!(part_ok_union_bad > 0, "closure unexpectedly holds");
        assert!(union_ok_part_bad > 0);
    }

    #[test]
    fn compression_statistics() {
        let mut state = 99_u64;
        for &(block, range) in &[(64_usize, 64_i64), (256, 64), (256, 1024)] {
            let mut kept = 0_u64;
            let mut total = 0_u64;
            for _ in 0..200 {
                let v: Vec<i64> = (0..block)
                    .map(|_| (lcg(&mut state) % range as u64) as i64)
                    .collect();
                let s = Entmax15Summary::from_scores(&v).unwrap();
                kept += s.retained_count();
                total += block as u64;
            }
            println!("ADA7-E15 compress block={block} range={range} kept={kept}/{total}");
        }
    }
}
