# ADA-A3: norm-ordered Cauchy–Schwarz early-termination certified softmax

Research reference only. This is a Q/K-level attention mechanism and its
falsification evidence. It is not a usefulness, novelty, hardware, or
FLAT-adoption claim. No prior-art review was done.

## Conjecture

Cycles #67–#70 skip *contiguous position blocks*. Their bottleneck is the
gap between block maxima (or box bounds): on iid keys no block is ever
skipped. A position-independent order removes that bottleneck. Visit keys in
non-increasing `‖k‖` (an A4-style prefill index). For every unread key,
`z_i = s⟨q,k_i⟩ ≤ s‖q‖‖k_i‖ ≤ B_j`, with `B_j` from the largest unread norm.
With `m` the max exact score read so far, `S ≥ 1` the read mass (units of
`exp m`) and `j` keys read,

`T_hi = (n − j) exp(B_j − m)`, `‖p − q‖₁ = 2t ≤ 2 T_hi / (S + T_hi)`

(`t/(1-t)`-monotone, so the unread mass can be replaced by its upper bound).
Stop at the first `j` whose certificate is `≤ ε`; tail rows are never dotted.
`B_j` is inflated by `1e-12` relative for rounding. The conjecture predicts
savings exactly when key norms are heterogeneous and `‖q‖` is large relative
to score gaps, and none for equal-norm keys.

## Falsification

- L1 vs an independent dense stable softmax of scalar dots exceeds
  `certified_l1_bound + 1e-12`.
- Unexplained `Err` on valid finite input, or NaN/inf output (overflowing bound).
- No designed witness where norm order reads far fewer rows than #69 two-stat.
- Equal-norm iid keys unexpectedly show large savings (would indicate a
  broken bound).
- Note: the argmax may legitimately be left unread within budget; only the
  L1 certificate is claimed.

## Implementation

`certified_norm_order_softmax[_accounted]` in
`crates/ada-a3-certified-softmax/src/norm_order.rs`. Cost counters
(`NormOrderCost`) are logical: prefill norms and sort (n log n, amortized
over queries), query-time rows read/skipped, exps, bound checks. Not hardware
measurements. The scan is sequential and data dependent, which is hostile to
GPU tiling; the caveat is unmeasured here.

## Results (native run)

- Fail-closed on empty/misshaped/non-finite input, `head_dim=0`,
  non-positive `score_scale`, bad `epsilon`. Zero query: all keys read,
  uniform. Equal keys: uniform. Overflowing bound: finite output.
- Designed witness (one norm-20 aligned needle, 63 norm-1 distractors,
  d=4, ε=1e-3): **1** row read, **63** skipped, needle mass > 0.999, no violation.
- Exhaustive grid (d=2, len 1..=4, 4-key alphabet, ε ∈ {1e-3, 5e-2, 0.5},
  **1020** cases): **0** certificate violations; skipping in **125** cases.
- Random sweep (d=8, n=48, norm σ ∈ {0,.5,1,1.5}, 3 budgets, 40 draws each):
  **0** violations.
- Gaussian keys `k = exp(σ·N)·g`, d=16, n=256, query `N(0,I)`, scale 1, 60
  draws; mean fraction of key rows read, norm-order vs #69 two-stat
  (block 16; box-bound overflow counted as all rows read):

| σ | ε | norm-order | two-stat box | max L1 | max cert |
|---|---|---|---|---|---|
| 0 | 1e-4 / 1e-2 / 1e-1 | 1.0000 / 0.9995 / 0.9965 | 1.0 / 1.0 / 1.0 | 1.3e-15 / 2.5e-5 / 1.0e-3 | 0 / 9.1e-3 / 9.8e-2 |
| 0.5 | 1e-4 / 1e-2 / 1e-1 | 0.835 / 0.712 / 0.616 | 1.0 / 1.0 / 1.0 | 6.9e-6 / 2.7e-4 / 1.1e-3 | ≤ ε |
| 1 | 1e-4 / 1e-2 / 1e-1 | 0.196 / 0.183 / 0.150 | 0.978 / 0.986 / 0.999 | 2.8e-8 / 1.6e-7 / 3.6e-6 | ≤ ε |
| 1.5 | 1e-4 / 1e-2 / 1e-1 | 0.077 / 0.069 / 0.066 | 0.977 / 0.985 / 0.928 | 1.4e-14 / 8.9e-14 / 2.4e-16 | ≤ ε |

## Verdict

**ADAPT (research reference).** The certificate holds on every tested case
and, unlike contiguous blocks, saves rows on iid keys whenever norms vary
(σ≥1: 80–93 % of rows unread, logical). Negatives: with equal norms
(σ=0) savings are ≈0 (0.35 % at ε=0.1); the certificate is loose (observed L1
is 10²–10⁷× below it at σ≥1); savings depend on synthetic log-normal norms —
real Qwen traces are not in the repo and were not tested, so real-model
norm spread is unknown; prefill needs a sort; the sequential scan is not
shown to be hardware-friendly. Not a softmax replacement, nor a novelty
claim (MIPS norm-ordering is a classical idea; prior-art review needed).

## Next

1. Tighten the tail with per-norm-bucket angular boxes (A4 boxes inside norm
   buckets) to combine norm order with direction bounds.
2. Test on real Q/K traces when available.
3. Adaptive witnesses; thresholded/sorted-tail on exact scores.
