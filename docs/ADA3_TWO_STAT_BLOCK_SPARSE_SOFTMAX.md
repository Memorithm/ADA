# ADA-A3: two-statistic certified block-sparse softmax

Research reference only. This document records a score-level attention
mechanism and its falsification evidence. It is not a usefulness, novelty,
hardware, or FLAT-adoption claim.

## Conjecture

The first-order operator `certified_block_sparse_softmax` bounds a contiguous
block from its maximum alone: `U_b = n_b * exp(m_b - M)` and
`L = sum_b exp(m_b - M)`. That certificate cannot skip a block whose maximum
is only a few nats below the global max `M`, even when almost every other
token in the block is far colder, because the upper bound pretends every
token equals `m_b`.

The two-statistic operator keeps the same partition, the same ban on skipping
a top-plateau block (`m_b = M`), and the same greedy visit order (increasing
`U_b`, then index). It stores three order statistics per block: the maximum
`m_b`, the second maximum `s_b`, and the minimum `ℓ_b`. Then

- `U_b = exp(m_b - M) + (n_b - 1) * exp(s_b - M)` upper-bounds the
  unnormalized block mass,
- `L_b = exp(m_b - M) + (n_b - 1) * exp(ℓ_b - M)` lower-bounds it,
- `L = sum_b L_b` lower-bounds the true total mass `S` (witnesses from every
  block, including blocks that will be skipped),
- skipping a set `K` of non-top-plateau blocks and renormalizing the
  complement yields `||p - q||_1 = 2 * S_K / S <= 2 * (sum_{b in K} U_b) / L`.

For any fixed skip set the two-stat bound is never looser than the first-order
bound, because `s_b <= m_b` and `ℓ_b <= m_b`. A singleton collapses to the
first-order pair. A length-2 block has `s_b = ℓ_b`, so `U_b = L_b` is the
exact block mass.

## Falsification

- The certificate fails if any finite input yields `||p - q||_1` larger
  than `certified_l1_bound + 1e-12`, where `p` is an independently
  implemented dense stable softmax.
- Totality fails if a finite non-empty input, positive `block_size`, and
  finite non-negative `epsilon` returns an unexplained `Err`.
- Mechanistic drop-in fails if unique-needle hit@1 or equal-pair
  interpolation is lost at the tested margins while dense softmax succeeds.
- The claimed improvement fails if there is no finite witness where
  two-stat skips a block that the first-order operator keeps under the
  same `(B, epsilon)`, or if two-stat violates a certificate that
  first-order would have respected.

## Independent oracle and cost

The oracle lives only in the crate tests: a separate max-shift / `exp` /
normalize path that never calls either block-sparse implementation. Logical
counters (`TwoStatCost`) count scanned tokens, skipped blocks, bound checks,
and `exp` evaluations (at most three distinct bound `exp`s per block, plus
kept tokens). They are not hardware measurements.

Pass 1 reads every score to form `(m_b, s_b, ℓ_b)` and spends one `exp` per
distinct order statistic used by `(U_b, L_b)`. Pass 2 exponentiates only
kept tokens.

## Results (native)

Native execution of the crate tests, recorded for the cycle that merged
this path. Under Miri the exhaustive grid and Gaussian survey shrink to a
qualitative slice; the figures below are native.

- Fail-closed on empty input, non-finite scores, `block_size = 0`, and
  non-finite or negative `epsilon`.
- All-equal length-8 and singleton inputs never skip; probabilities are
  uniform `1/n` or `{1}`.
- Wide gap: twelve scores at `-20` with a single `0` at index 9, `B = 4`,
  `epsilon = 1e-6` skips the two cold blocks, exponentiates 4 kept tokens,
  and keeps the needle as the unique mode with mass `> 0.999999`.
- Unique-needle hit@1 and equal-pair interpolation succeed at every tested
  margin `{0.1, 0.4, 0.6, 1, 2, 4}` with `B = 2` and `epsilon = 1e-4`.
- Witness (length 16, `B = 8`, `epsilon = 0.1`): block 0 is eight zeros
  (the unique top plateau) and block 1 is `[-3, -20⁷]`. First-order keeps
  both blocks (`U_1 = 8 e^{-3} ≈ 0.398`, `2 U_1 / L ≈ 0.76 > 0.1`).
  Two-stat skips the cold block (`U_1 = e^{-3} + 7 e^{-20} ≈ 0.0498`).
- Exhaustive grid, lengths `1..=5`, scores in `{-3, -1, 0, 2}`, every
  block size (**6372** cases): **0** certificate violations. Tight budget
  `epsilon = 1e-3` skips in **0** cases (same barrier as first-order: the
  largest domain gap is 5). Loose budget `epsilon = 5e-2` skips in
  **1621** cases versus **1567** for first-order: **54** extra skip cases,
  and the skip *set* is identical in **6318** cases. No case on this grid
  had two-stat skip fewer blocks than first-order.
- Iid unit Gaussians, `n = 32`, `B = 8`, 200 LCG/Box-Muller samples: mean
  skip fraction **0** at `epsilon` in `{1e-6, 1e-4, 1e-2, 1e-1}`, matching
  first-order. Tighter block statistics do not remove the ~5-nat barrier
  on unstructured block maxima.
- The same draws with a `+20` spike at the midpoint: mean skip fraction
  **0.717500** at `epsilon = 1e-6` (first-order **0.672500**; max observed
  L1 `6.278911e-7` versus max certificate `9.942326e-7`) and **0.750000**
  at `epsilon` in `{1e-4, 1e-2, 1e-1}` (same as first-order; three of four
  blocks; max observed L1 `2.587330e-6` versus max certificate
  `4.243676e-6`, tighter than first-order's `7.754115e-6`). Zero
  violations.
- Structured "medium outlier" rewrite of the same draws: every non-max
  block is replaced by one token at `M - 3` and seven tokens at `M - 20`.
  Mean skip fraction is **0** at `epsilon` in `{1e-6, 1e-4, 1e-2}` for
  both operators (a single such block has `2 U / L ≈ 0.08`, above 0.01).
  At `epsilon = 0.1`, two-stat reaches mean skip fraction **0.258750**
  (max observed L1 `8.640491e-2` versus max certificate `9.979679e-2`)
  while first-order remains at **0**. Typically one of the three cold
  blocks fits the budget; a second does not.

## Non-claims

This does not replace [`budgeted_softmax`](../crates/ada-a3-certified-softmax/src/lib.rs)
or [`certified_block_sparse_softmax`](ADA3_CERTIFIED_BLOCK_SPARSE_SOFTMAX.md),
does not certify floating-point rounding of the skipped operator, does not
measure wall-clock or memory traffic, and does not promote the mechanism
to FLAT.
