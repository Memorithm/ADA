# ADA-A3: mean-direction projection + residual certified softmax

Status: research reference. Not a production runtime, not a novelty claim
(two-term Cauchy–Schwarz / projection ordering is close to classical MIPS
bounds; prior-art review required), logical counts only.

## Conjecture

Norm ordering (#71) cannot separate equal-norm keys. Fix at prefill a unit
direction `u` (normalized mean key; `e_1` if zero) and store `p_i = <k_i,u>`,
`r_i = ||k_i - p_i u||`. For `q = a u + q_perp` (`a = s<q,u>`, `b = s||q_perp||`):
`z_i <= a p_i + b r_i`. Scan keys by decreasing `a p_i` (direction chosen from
the sign of `a`); unread keys obey `z_i <= a p_next + b R_j`, `R_j` = max
residual among unread keys (prefix/suffix maxima, prefill). Stop when
`2 T_hi/(S+T_hi) <= epsilon` with `T_hi = (n-j) exp(bound - m)`.
Falsification: any certificate violation; no regime where it reads fewer rows
than #71; query cost beyond one extra dot with `u`.

## Implementation

`crates/ada-a3-certified-softmax/src/proj_order.rs`
(`certified_proj_order_softmax[_accounted]`). Independent oracle: dense stable
softmax from scalar dots (`dense_qk_scores`). Cost: prefill 3 passes + 1 sort;
per query 1 direction dot + key rows read + exp/bound checks.

## Results

- Fail-closed on invalid layout/inputs/scale/epsilon. Singleton, equal keys,
  zero query, all-zero keys (fallback direction), antipodal keys (zero mean),
  overflowing bound: finite and valid.
- Negative-alignment test (`a < 0`, reverse scan): fewer than all rows read,
  argmax correct, certificate holds.
- Designed witness (65 keys of equal norm 5: one needle, 64 distractors at
  `(0,±5)`, ε=1e-3): proj-order reads **1** row, norm-order (#71) reads **65**.
- Exhaustive grid (d=2, len 1..=4, 4-key alphabet, 2 queries, 3 budgets,
  **2040** cases): **0** violations; skipping in **594** cases.
- Random sweep (d=8, n=48, mean shift {0,1,3}, norm σ {0,1}, 3 budgets, 30 draws): **0** violations.
- Gaussian keys `N(0,I)+shift·e_1`, query `N(0,I)+rho·e_1`, d=16, n=256, 60
  draws, mean fraction of rows read (proj vs norm-order #71):

| shift, rho | ε=1e-4 | ε=1e-2 | ε=1e-1 |
|---|---|---|---|
| 0, 0 | 1.0000 / 1.0000 | 0.9999 / 0.9989 | 0.9985 / 0.9959 |
| 2, 0 | 0.9999 / 0.9999 | 0.9996 / 0.9993 | 0.9991 / 0.9960 |
| 2, 2 | 0.9991 / 0.9991 | 0.9961 / 0.9920 | 0.9930 / 0.9764 |
| 4, 2 | 0.9988 / 0.9960 | 0.9954 / 0.9804 | 0.9919 / 0.9682 |
| 4, 4 | 0.9876 / 0.9707 | 0.9695 / 0.9174 | 0.9387 / 0.8195 |
| 8, 4 | 0.9875 / 0.9725 | 0.9604 / 0.9103 | 0.9535 / 0.8896 |

Max observed L1 always ≤ certificate (e.g. 6.7e-5 vs ≤0.1).

## Verdict

**ADAPT as a reference with a clear NEGATIVE on iid-like data.** The
certificate is valid and the designed equal-norm witness shows a strict gain
over #71 (1 vs 65 rows). But on mean-shifted Gaussian keys the projection
bound is *not* better than plain norm ordering — it reads *more* rows in
every measured row of the table (e.g. 0.9387 vs 0.8195 at shift 4, rho 4,
ε=0.1). Reasons: the residual maximum `R_j` is a worst case over the unread
set and is close to the full norm in d=16, so the added `b R_j` term eats the
projection gain, and the mean direction is not the direction that matters
for the query. Mean direction also fragile (balanced clusters can rotate it).
No real Q/K traces were tested. No hardware claim.

## Next

1. Per-query adaptive direction (`u = q/||q||`): ordering by exact score is the
   sorted-tail limit; the cost is a full dot per key, so only multi-level
   cone-tree pruning (subtrees by one centroid dot) can pay.
2. Several residual maxima (per-block R_j) instead of one suffix max.
3. Real Q/K traces (Qwen3 capture) for norm/mean-direction structure.
