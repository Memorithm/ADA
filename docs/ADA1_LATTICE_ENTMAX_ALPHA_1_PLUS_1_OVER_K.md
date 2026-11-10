# ADA1 — Exact rational alpha = 1 + 1/k entmax ("lattice entmax") (ada-a6)

Status: **research reference path / no usefulness, novelty, or FLAT claim**.

## Conjecture

For integer scores and `alpha = 1 + 1/k` (`k = 1` sparsemax, `k = 2`
1.5-entmax, softmax as `k -> infinity`; the previous log entry had the
limits reversed), entmax is `p_i = ((z_i - u)/k)^k` on the support with
`u = k tau`. The threshold on a support prefix of size `rho` solves the
integer polynomial `sum (z_i - u)^k = k^k` (degree `k`, leading coefficient
`±rho`). By the rational root theorem a rational `u = a/b` has `b | rho`.
For each `b | rho`, `sum (b z_i - a)^k = (k b)^k` is strictly monotone in
`a` on the window `a in [b(z_max - k), b z_min)`, so an exact integer binary
search decides root existence. Conjecture: this decides the rational domain
exactly for every `k`, matches the independent A4 oracle on every accepted
vector, has at most one admissible prefix, reduces to exact sparsemax at
`k = 1` and to the discriminant path's domain at `k = 2`, and the domain
shrinks quickly with `k` (arithmetic quantisation of scores onto a lattice).

Falsified by: an oracle disagreement, `k = 1` or `k = 2` mismatch with the
existing exact paths (including accept/reject domain), two admissible
prefixes, or a rejected vector that is in fact rational.

## Method

`lattice_entmax_i64[_accounted]` (crate `ada-a6-tau-solvers`, 1 <= k <= 16):
sort descending, scan prefixes, scan divisors `b | rho`, integer binary
search, check excluded scores `<= u`, fail closed otherwise (no float
fallback). Logical counters only (`LatticeEntmaxCost`), no hardware claim.
Independent oracle: `ada_a4_entmax_bnb::dense_entmax(.., 1 + 1/k)`.

## Results (native grid, length 1..=5, scores in {0..4}, 3905 vectors)

| k | accepted | rejected | oracle disagreements |
|---|----------|----------|----------------------|
| 1 | 3905 | 0 | 0 |
| 2 | 768 | 3137 | 0 |
| 3 | 363 | 3542 | 0 |
| 4 | 19 | 3886 | 0 |

- `k = 1`: identical to `sparsemax_exact_i64` (length<=4, scores<=4).
- `k = 2`: accept/reject domain and values identical to `entmax15_exact_i64`
  (length<=5, scores<=3). Two independent decision procedures agree.
- `k = 3`: 216 accepted vectors have unequal scores on the support (e.g.
  `[2,1,0,0]`), so the lattice is not only equal-score blocks; fixture
  `[0;8]` gives `p = 1/8`, `tau = -1/2`; `[3,0]` gives `p = (1,0)`.
- Rejections: `[0;4]`, `[0,0]` for k=3 (not a perfect cube support length).
- Translation by c shifts tau by c/k; permutation invariance holds.
- Miri runs a reduced grid (length<=3, scores<=2).

## Verdict

**ADAPT (research reference only).** An exact integer decision procedure for
the whole `alpha = 1 + 1/k` family with independent-oracle parity. Negative
finding: the rational domain collapses fast with `k` (3905 -> 19 accepted
at k=4), so as an attention mechanism it needs score quantisation onto the
lattice and is not a general replacement for the f64 oracle. Overflow is
nearly unreachable because the window bounds `|z - u| <= k` before powers
are formed. No prior-art review was done, so no novelty claim.

## Open

Closed-form description of the lattice for k >= 3; composable (A7) merge
for k >= 2; whether quantising to the lattice keeps useful task behaviour.
