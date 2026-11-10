# ADA-A3: two-statistic QK-box certified block-sparse softmax

Research reference only. This document records a Q/K-level attention
mechanism and its falsification evidence. It is not a usefulness, novelty,
hardware, or FLAT-adoption claim.

## Conjecture

The score-level operators of cycles #67 and #68 read every score before
they can decide to skip a block. When those scores are scaled dots
`z_i = scale · <q, k_i>`, that first pass already pays the inner-product
work the skip was meant to avoid.

A coordinate box on a key block (A4) bounds every score of that block
from the query and the per-coordinate extrema alone:

- `hi = scale * Σ_j max(q_j kmin_j, q_j kmax_j)`
- `lo = scale * Σ_j min(q_j kmin_j, q_j kmax_j)`

A single box cannot separate the maximum from the second maximum: the
same corner bound applies to every key, so the first-order mass upper
bound is `U_b = n_b * exp(hi - M_lo)`. Splitting the block into two
contiguous halves (a depth-1 A5 hierarchy) yields a two-statistic bound
that is never looser:

- `U_b = Σ_{h ∈ b} n_h * exp(hi_h - M_lo)`
- `L_b = Σ_{h ∈ b} [exp(w_h - M_hi) + (n_h - 1) * exp(lo_h - M_hi)]`

`M_lo` is the maximum of one witness score per half (the first key of
the half). `M_hi` is the maximum half-box `hi`. Then `M_lo ≤ M ≤ M_hi`,
`L = Σ_b L_b` lower-bounds the true total mass `S`, and skipping a set
`K` of blocks whose box-max is strictly below `M_lo` (they cannot hold a
top-plateau token) produces `||p - q||_1 = 2 S_K / S ≤ 2 (Σ_{b ∈ K} U_b) / L`.

Query-time work materializes a key row only as a witness or as a member
of a kept block. A skipped block contributes at most its (one or two)
witnesses.

## Falsification

- The certificate fails if any finite valid input yields `||p - q||_1`
  larger than `certified_l1_bound + 1e-12`, where `p` is an independently
  implemented dense stable softmax of the true scaled dots.
- Totality fails if a finite non-empty Q/K layout, positive `head_dim`
  and `block_size`, positive finite `score_scale`, and finite
  non-negative `epsilon` returns an unexplained `Err`.
- Safety fails if a skipped block contains a true global-maximum token.
- Mechanistic drop-in fails if unique-needle hit@1 or equal-pair
  interpolation is lost at the tested margins while dense softmax succeeds.
- The claimed improvement fails if there is no finite witness where the
  two-half operator skips a block that the first-order full-block box
  keeps under the same `(B, epsilon)`, or if the two-half bound ever
  skips fewer blocks than first-order on that witness.
- The compute-saving claim (logical, not hardware) fails if every skip
  still requires reading every key row at query time.

## Independent oracle and cost

The oracle lives only in the crate tests: dense scaled dots followed by
a separate max-shift / `exp` / normalize path that never calls the
QK-box operator. Logical counters (`QkBoxCost`) count box evaluations,
witness dots, materialized kept-block dots, distinct query-time key
rows, bound checks, and `exp` evaluations. They are not hardware
measurements. Box construction scans keys once to form min/max metadata
and is the A4-style prefill cost, not a query-time inner product.

## Results (native)

Native execution of the crate tests, recorded for the cycle that merged
this path. Under Miri the exhaustive grid and Gaussian survey shrink to
a qualitative slice; the figures below are native.

- Fail-closed on empty Q or K, shape mismatch, non-finite values,
  `head_dim = 0`, `block_size = 0`, non-positive `score_scale`, and
  non-finite or negative `epsilon`.
- Singleton and all-equal keys never skip; probabilities are `{1}` or
  uniform `1/n`.
- Anti-aligned prefix: twelve dim-2 keys, eight at `(-20, 0)` and a
  needle `(1, 0)` at index 8 with the rest of its block at `(-20, 0)`,
  `B = 4`, `epsilon = 1e-6`, query `(1, 0)`. The two cold blocks are
  skipped; only four kept keys are materialized; query-time rows read
  are strictly fewer than 12; the needle is the unique mode with mass
  `> 0.999999`.
- Witness (eight dim-2 keys, `B = 4`, `epsilon = 0.2`, query `(1, 0)`):
  block 0 is four copies of `(4, 0)`; block 1 is two copies of `(2, 0)`
  and two copies of `(-20, 0)`. First-order keeps both blocks; two-stat
  skips the mixed block.
- Unique-needle hit@1 and equal-pair interpolation succeed at every
  tested margin `{0.1, 0.4, 0.6, 1, 2, 4}` with `B = 2` and
  `epsilon = 1e-4`.
- Exhaustive grid, dim 2, lengths `1..=4`, keys drawn from
  `{(0,0), (1,0), (0,1), (-2,0)}`, query `(1, 0.25)`, every block size
  (**1252** cases): **0** certificate violations. Tight budget
  `epsilon = 1e-3` skips in **0** cases. Loose budget `epsilon = 5e-2`
  skips in **57** cases versus **51** for the first-order box (**6**
  extra). On this small grid every skipped block was short enough that
  the two witnesses already covered it, so query-time rows read never
  fell below `n` (**0** row-saving cases).
- Iid unit-Gaussian keys, `n = 32`, `d = 4`, `B = 8`, 80 LCG/Box-Muller
  samples, query `e_1`: mean skip fraction **0** and mean query-time
  row fraction **1** at `epsilon` in `{1e-4, 1e-2, 1e-1}`. Unstructured
  boxes are too loose to certify a skip (the same ~gap barrier as
  #67/#68, now at the box rather than the exact block max).
- The same draws rewritten as two polarity clusters (block 0 keys near
  `+6 e_1`, remaining blocks near `-6 e_1`, 0.05 Gaussian jitter): mean
  skip fraction **0.750000** at every tested budget (three of four
  blocks), mean query-time row fraction **0.437500** (8 kept keys plus
  2 witnesses from each of 3 skipped blocks), max observed L1
  `3.889461e-5` versus max certificate `5.364497e-5`. First-order matches
  the skip rate on this strongly separated regime. Zero violations.

## Non-claims

This does not replace [`budgeted_softmax`](../crates/ada-a3-certified-softmax/src/lib.rs),
[`certified_block_sparse_softmax`](ADA3_CERTIFIED_BLOCK_SPARSE_SOFTMAX.md),
or [`certified_two_stat_softmax`](ADA3_TWO_STAT_BLOCK_SPARSE_SOFTMAX.md).
It does not certify floating-point rounding, does not measure wall-clock
or memory traffic, and does not promote the mechanism to FLAT. Logical
row savings are not hardware speedups.
