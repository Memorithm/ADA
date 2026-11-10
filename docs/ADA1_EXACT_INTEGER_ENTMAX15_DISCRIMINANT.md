# ADA1 — Exact integer 1.5-entmax discriminant domain (ada-a6)

Status: **research reference path / no usefulness claim**.

## Conjecture

For a non-empty integer score vector, the alpha = 1.5 entmax threshold and
probabilities are rational if and only if some descending support prefix of
size `ρ` has a perfect-square discriminant

```text
D = S₁² - ρ S₂ + 4ρ = 4ρ - ∑_{i < j in prefix} (zᵢ - zⱼ)²
```

in the threshold quadratic `ρ t² - 2 S₁ t + S₂ - 4 = 0`, the smaller root
`t = (S₁ - √D) / ρ` satisfies the complementary inequalities (every prefix
score strictly above `t`, every excluded score at most `t`), and checked
integer arithmetic does not overflow. Then

```text
pᵢ = (zᵢ - t)₊² / 4,     τ = t / 2
```

are reduced rationals and coincide with 1.5-entmax. This domain strictly
contains the historical E1 subset (equal scores of perfect-square length, or
a unique maximum at least 2 above every other score).

The conjecture is falsified if a historical-domain vector is rejected, if an
accepted vector disagrees with the independent A4 `dense_entmax(..., 1.5)`
oracle beyond tolerance, if a vector with no perfect-square admissible
prefix is accepted, if two distinct prefixes are both admissible, or if
hand-derived rational fixtures fail to match.

## Method

`entmax15_exact_i64` scans descending prefixes, forms `S₁` and `S₂` in
checked `i128` arithmetic, accepts a prefix only when `D` is a
non-negative perfect square and the complementary inequalities hold, and
fails closed otherwise. There is no bisection fallback. Logical counters in
`ExactEntmax15Cost` record scores sorted, prefixes tested, perfect-square
discriminants, and inequality checks. Those counters are not hardware
instruction, bandwidth, or wall-clock claims.

`ada_a4_entmax_bnb::dense_entmax` remains the independent general f64
oracle and is not called by the candidate.

## Results

Hand-derived fixtures:

- Historical one-hot and square-length cases are unchanged, including
  `[7]`, `[4, 1, 0]`, gap-exactly-2 `[-1, -3]`, and `[0; 4]`, `[1; 9]`.
- New unit-spike family (`3n + 1` a perfect square): `[1, 0, 0, 0, 0]`
  yields `p = (16/25, 9/100, 9/100, 9/100, 9/100)` and `τ = -3/10`;
  `[1, 0⁷]` yields `p₀ = 9/16`, `p_{i>0} = 1/16`, `τ = -1/4`.
- Permutations and translations preserve probabilities and shift `τ` by
  `c/2`. The partial-support vector `[1, 0, 0, 0, 0, -1]` keeps the same
  five-point distribution and zeros the excluded score.
- Previously rejected non-rational cases stay rejected: empty input,
  `[0, 0]`, `[3, 2]`, `[1, 1, 0]`, `[4, 4, 0]`, `[2, 1, 0]`, `[5, 5, 3]`.

Independent A4 oracle parity holds on every accepted hand fixture (absolute
probability difference at most `1e-9`, and the reconstructed mass identity
`∑ (zᵢ/2 - τ)₊² = 1` within `1e-9`).

Exhaustive grid, lengths `1..=5` and scores in `{0, 1, 2, 3}` (1364
vectors): 195 accepted, of which 150 lie in the historical E1 subset and 45
are new; 1169 rejected; 0 dual admissible prefixes; 0 oracle mismatches; 0
historical-domain regressions. On `[1, 0, 0, 0, 0]` the logical cost is 5
scores sorted, 5 prefixes tested, 2 perfect-square discriminants, and 7
inequality checks (early exit on the failing `ρ = 1` inactive test).

## Verdict

**ADAPT.** The discriminant scan is a strictly stronger exact rational
decision procedure than the E1 syntactic subset, with independent oracle
parity on the tested domain and fail-closed behavior outside it. This does
not replace the A4 f64 oracle for non-rational 1.5-entmax, and it is not a
usefulness, novelty, hardware, or FLAT-adoption claim. A complete
number-theoretic classification of all integer families with perfect-square
`D` remains open.

## Non-claims

No usefulness, novelty, FLAT adoption, or hardware performance claims.
Logical operation counts are not physical measurements.
