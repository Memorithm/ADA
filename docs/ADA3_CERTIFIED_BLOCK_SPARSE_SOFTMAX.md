# ADA-A3: certified block-sparse softmax

Research reference only. This document records a score-level attention
mechanism and its falsification evidence. It is not a usefulness, novelty,
hardware, or FLAT-adoption claim.

## Conjecture

Softmax never has exact zeros, so the A4/A5 support-pruning certificates
cannot drop a token while remaining identical to dense softmax. A different
mechanism is still well-defined: partition finite scores `z` into contiguous
blocks of length at most `B`, bound each block from its maximum alone, and
omit a block when a conservative L1 certificate stays inside a caller budget
`epsilon`.

Let `M = max_i z_i` and `m_b = max_{i in b} z_i` for a block `b` of length
`n_b`. The shifted terms `t_i = exp(z_i - M)` lie in `(0, 1]`. Then

- `U_b = n_b * exp(m_b - M)` upper-bounds the unnormalized mass of `b`,
- `L = sum_b exp(m_b - M)` lower-bounds the total mass `S` (one witness
  token per block),
- skipping a set `K` of blocks that do not contain a global maximum and
  renormalizing the complement produces an approximate distribution `q`
  whose distance to the exact softmax `p` satisfies
  `||p - q||_1 = 2 * S_K / S <= 2 * (sum_{b in K} U_b) / L`.

The operator never skips a top-plateau block (`m_b = M`). Remaining cold
blocks are visited in increasing `U_b` (then index) and skipped greedily
while that certificate stays at most `epsilon`.

## Falsification

- The certificate fails if any finite input yields `||p - q||_1` larger
  than `certified_l1_bound + 1e-12`, where `p` is an independently
  implemented dense stable softmax.
- Totality fails if a finite non-empty input, positive `block_size`, and
  finite non-negative `epsilon` returns an unexplained `Err`.
- Mechanistic drop-in fails if unique-needle hit@1 or equal-pair
  interpolation is lost at the tested margins while dense softmax succeeds.
  The skip rule is designed not to erase a global maximum, so a miss would
  indicate a bug rather than a quantization artifact.

## Independent oracle and cost

The oracle lives only in the crate tests: a separate max-shift / `exp` /
normalize path that never calls the block-sparse implementation. Logical
counters (`BlockSparseCost`) count scanned tokens, skipped blocks, bound
checks, and `exp` evaluations. They are not hardware measurements.

Pass 1 reads every score to form block maxima and spends one `exp` per
block on `(U_b, L)`. Pass 2 exponentiates only kept tokens. The certificate
itself needs only per-block maxima, so a later QK-box bound could replace
the exact scan; that composition is not claimed here.

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
- Exhaustive grid, lengths `1..=5`, scores in `{-3, -1, 0, 2}`, every
  block size (`6372` cases): **0** certificate violations. Tight budget
  `epsilon = 1e-3` skips in **0** cases (the largest domain gap is 5, and
  `2 exp(-5) ≈ 0.0135 > 1e-3`). Loose budget `epsilon = 5e-2` skips in
  **1567** cases.
- Iid unit Gaussians, `n = 32`, `B = 8`, 200 LCG/Box-Muller samples:
  mean skip fraction **0** at `epsilon` in `{1e-6, 1e-4, 1e-2, 1e-1}`.
  Unstructured block maxima almost never differ by the ~5 nats required
  even for a 0.1 L1 budget under the conservative `(U_b, L)` bound.
- The same draws with a `+20` spike at the midpoint: mean skip fraction
  **0.672500** at `epsilon = 1e-6` (max observed L1 `4.806604e-7` versus
  max certificate `9.975798e-7`) and **0.750000** at `epsilon` in
  `{1e-4, 1e-2, 1e-1}` (three of four blocks; max observed L1
  `2.587330e-6` versus max certificate `7.754115e-6`). Zero violations.

## Non-claims

This does not replace [`budgeted_softmax`](../crates/ada-a3-certified-softmax/src/lib.rs),
does not certify floating-point rounding of the skipped operator, does not
measure wall-clock or memory traffic, and does not promote the mechanism
to FLAT.
