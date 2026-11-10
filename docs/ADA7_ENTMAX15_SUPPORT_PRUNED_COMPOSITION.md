# ADA7: support-pruned composable exact 1.5-entmax

Status: RESEARCH (not a qualified mechanism). No novelty, utility, hardware or FLAT claim.

## Conjecture
For alpha = 1.5, `p_i = (z_i - t)_+^2 / 4`. Adding scores only increases the
normalizer, so the union threshold is >= each part-local threshold. A score at
or below its part-local threshold is therefore outside the global support, and a
part can be replaced by its local support (multiset of plateaus). Merge =
multiset union + re-prune: commutative, associative, equal to pruning the
concatenation. Local support is decided by exact integer inequalities
(`D = S1^2 - rho*S2 + 4*rho >= 0` and `D > (S1 - rho*z)^2` when the right side
is non-negative), with no perfect-square requirement; only the final solve
(`entmax15_exact_i64`) needs the rational domain and fails closed otherwise.

## Implementation
`Entmax15Summary` (`ada-a7-composable-entmax`): `from_scores[_accounted]`,
`compose[_accounted]`, `solve_exact` (expansion capped at 65536 scores).
Checked i128 everywhere; i64 extremes fail closed (moments overflow).
Oracles: unpruned `entmax15_exact_i64` (exact) and `ada_a4_entmax_bnb::dense_entmax(.., 1.5)` (f64, independent).

## Results (observed, `cargo test -- --nocapture`)
- (Sous Miri la grille est réduite : longueurs ≤4, domaine ≤2, 10 cas aléatoires ; chiffres ci-dessous = exécution native.) Exhaustive: lengths 2..=6, scores {0..3}, every split point: **25488** splits, **0** failures of `merge(l,r) == merge(r,l) == direct summary`.
- 734 vectors in the rational domain: pruned solve equals unpruned exact solve (identical reduced rationals), f64 oracle parity <= 1e-9, mass = 1 within 1e-9, support never larger than retained set.
- 300 random multi-block cases (2-5 blocks, 1-12 scores, range -4..4): left-fold, right-fold and direct all identical (associativity).
- Pruning active on **5172** of the 5456 enumerated vectors (length>=2).
- Compression (200 trials, uniform random integer scores): block 64 / range 64: 516 of 12800 scores retained (4.0%); block 256 / range 64: 1145/51200 (2.2%); block 256 / range 1024: 283/51200 (0.55%).
- **Negative result:** the rational domain is not closed under composition. Over 7056 (left,right) pairs (lengths 1..=3, scores {0..3}): 545 have both parts and not the union in domain; 879 have the union in domain but not both parts. Parts need only local support (exact inequality), never rationality.
- Adversarial: i64::MAX/MIN moments overflow -> `Err`; empty input -> `Err`; 9 equal zeros atomic p=1/9; `[0,0]` summary exists but solve fails closed.

## Cost accounting (logical, not hardware)
`PruneCost`: levels examined, prefix tests, scores discarded. Prune is
O(distinct plateaus) after the map merge; retained size is data dependent.
Compression numbers above are on synthetic uniform integer scores only; no real
task, no wall-clock, no memory measurement.

## Verdict
**ADAPT (reference identity).** Exact blockwise 1.5-entmax summaries with provable
safe pruning are confirmed on the tested domain. Open: real attention-score
traces, quantization needed to land in the rational domain, prior-art review.
