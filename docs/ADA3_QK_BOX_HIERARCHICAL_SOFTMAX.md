# ADA-A3: hierarchical QK-box certified block-sparse softmax

Research reference only. This document records a Q/K-level attention
mechanism and its falsification evidence. It is not a usefulness, novelty,
hardware, or FLAT-adoption claim.

## Conjecture

Cycle #69 bounds each contiguous key block from depth-1 half-boxes and
then skips *whole blocks*. A mixed block whose warm half reaches `M_lo`
can never be omitted, so every key of that block is materialized even
when the other half is far colder than the budget would allow. On keys
that are only moderately structured, the same all-or-nothing rule also
prevents any query-time row saving.

A deeper A5 split (depth `D` binary cuts inside each top-level block)
produces at most `2^D` leaf boxes. Treating each leaf as an independent
skip unit gives

- `U_ℓ = n_ℓ exp(hi_ℓ − M_lo)`
- `L_ℓ = exp(w_ℓ − M_hi) + (n_ℓ − 1) exp(lo_ℓ − M_hi)`

with `M_lo` the maximum of one witness score per leaf (the first key of
the leaf) and `M_hi` the maximum leaf-box `hi`. Then `M_lo ≤ M ≤ M_hi`,
`L = Σ_ℓ L_ℓ` lower-bounds the true total mass `S`, and skipping a set
`K` of leaves whose box-max is strictly below `M_lo` produces
`||p − q||_1 = 2 S_K / S ≤ 2 (Σ_{ℓ ∈ K} U_ℓ) / L`. Depth 0 is one leaf
per block; depth 1 recovers the halves of #69, but the skip decision is
now per leaf rather than per parent block.

Finer boxes are contained in their parent, so a leaf `hi` cannot exceed
its parent's `hi`. Extra witnesses can only raise `M_lo`. The leaf-level
bound is therefore never looser than the parent-block bound of #69 on the
same omitted region. A skipped leaf contributes only its witness at
query time.

The frozen Qwen3 E4 trace files are not stored in this repository (only
the `ada-a5-real-qk-trace` parser). This cycle therefore measures
synthetic regimes that isolate the geometries the hierarchy can and
cannot resolve: iid keys, strongly clustered blocks (the #69 witness),
moderately clustered blocks, and pair-interleaved hot/cold keys.

## Falsification

- The certificate fails if any finite valid input yields `||p − q||_1`
  larger than `certified_l1_bound + 1e-12`, where `p` is an independently
  implemented dense stable softmax of the true scaled dots.
- Totality fails if a finite non-empty Q/K layout, positive `head_dim`
  and `block_size`, `hierarchy_depth ≤ 8`, positive finite `score_scale`,
  and finite non-negative `epsilon` returns an unexplained `Err`.
- Safety fails if a skipped leaf contains a true global-maximum token.
- Mechanistic drop-in fails if unique-needle hit@1 or equal-pair
  interpolation is lost at the tested margins while dense softmax succeeds.
- The claimed improvement fails if there is no finite witness where
  leaf-level depth-1 skip saves query-time rows that the #69 two-stat
  block skip still reads, or if depth-2 leaf skip cannot isolate
  pair-interleaved cold keys that depth-1 mixed halves keep.
- The compute-saving claim (logical, not hardware) fails if every skipped
  leaf still requires reading every key row of that leaf at query time.

## Independent oracle and cost

The oracle lives only in the crate tests: dense scaled dots followed by
a separate max-shift / `exp` / normalize path that never calls the
hierarchical operator. Logical counters (`QkHierarchyCost`) count leaf
box evaluations, witness dots, materialized kept-leaf dots, distinct
query-time key rows, bound checks, and `exp` evaluations. They are not
hardware measurements. Box construction scans keys once to form min/max
metadata and is the A4-style prefill cost, not a query-time inner product.

## Results (native)

Native execution of the crate tests, recorded for the cycle that merged
this path. Under Miri the exhaustive grid and Gaussian survey shrink to
a qualitative slice; the figures below are native.

- Fail-closed on empty Q or K, shape mismatch, non-finite values,
  `head_dim = 0`, `block_size = 0`, `hierarchy_depth > 8`, non-positive
  `score_scale`, and non-finite or negative `epsilon`.
- Singleton and all-equal keys never skip; probabilities are `{1}` or
  uniform `1/n`.
- Mixed-block witness (eight dim-2 keys, `B = 4`, `epsilon = 0.2`,
  query `(1, 0)`): four hot keys, then two hot keys and two keys at
  `(-20, 0)`. The #69 two-stat operator keeps both blocks because the
  warm half of block 1 reaches `M_lo`. Depth-1 leaf skip omits only the
  cold half (one of four leaves) and reads strictly fewer query-time
  rows than #69.
- Pair-interleaved witness (`[4, 4, -20, -20, 4, 4, -20, -20]`, `B = 8`,
  `epsilon = 0.2`): depth 1 produces two mixed halves and skips nothing;
  depth 2 skips the two cold pairs, reads strictly fewer rows than
  depth 1, and first-order skips no block.
- Unique-needle hit@1 and equal-pair interpolation succeed at every
  margin `{0.1, 0.4, 0.6, 1, 2, 4}` with `B = 4`, depth 2 and
  `epsilon = 1e-4`.
- Exhaustive grid, dim 2, lengths `1..=4`, keys in
  `{(0,0), (1,0), (0,1), (-2,0)}`, query `(1, 0.25)`, all block sizes,
  depth 2 (**1252** cases): **0** certificate violations. Budget
  `epsilon = 5e-2`: leaf skips in **193** cases against **57** block
  skips for #69. On this short-block grid there were **0** cases of
  query-time row saving (witnesses already cover skipped leaves).
- Synthetic surveys, `n = 32`, `d = 4`, `B = 8`, 80 samples, query
  `e_1`, **0** violations in every cell:

  | regime | depth | epsilon | mean leaf skip | #69 mean block skip | mean row frac | #69 mean row frac | max L1 | max bound |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | iid | 1 | 1e-4 / 1e-2 / 1e-1 | 0 | 0 | 1 | 1 | 0 | 0 |
  | iid | 2 | 1e-4 / 1e-2 | 0 | 0 | 1 | 1 | 0 | 0 |
  | iid | 2 | 1e-1 | 0.066406 | 0 | 0.966797 | 1 | 6.713262e-2 | 9.935700e-2 |
  | clustered ±6 | 1 | all tested | 0.750000 | 0.750000 | 0.437500 | 0.437500 | 3.889461e-5 | 5.364497e-5 |
  | clustered ±6 | 2 | all tested | 0.750000 | 0.750000 | 0.625000 | 0.437500 | 3.889461e-5 | 4.408063e-5 |
  | moderate ±2.5 | 1 | 1e-2 | 0.053125 | 0 | 0.960156 | 1 | 7.282357e-3 | 9.965150e-3 |
  | moderate ±2.5 | 2 | 1e-2 | 0.150781 | 0 | 0.924609 | 1 | 8.877152e-3 | 9.990056e-3 |
  | moderate ±2.5 | 1 | 1e-1 | 0.684375 | 0.653125 | 0.486719 | 0.510156 | 5.503066e-2 | 9.862578e-2 |
  | moderate ±2.5 | 2 | 1e-1 | 0.747656 | 0.653125 | 0.626172 | 0.510156 | 5.503066e-2 | 9.832966e-2 |
  | pair-interleaved | 1 | all tested | 0 | 0 | 1 | 1 | 0 | 0 |
  | pair-interleaved | 2 | all tested | 0.500000 | 0 | 0.750000 | 1 | 1.291358e-5 | 1.407934e-5 |

- Strong clustering is a negative result for extra depth: depth 2 skips
  the same 75 % of units as #69 but reads more rows (0.625 vs 0.4375)
  because each skipped region pays more witnesses. Depth 2 also tightens
  the certificate slightly (`4.408063e-5` vs `5.364497e-5`).
- The frozen Qwen3 E4 traces are not present in the repository, so no
  natural-slice numbers are claimed.

## Verdict

**ADAPT (research reference).** The L1 certificate holds, leaf skip
recovers cold tokens inside mixed blocks that #69 must keep, and
depth 2 isolates pair-interleaved cold keys (50 % of leaves skipped,
75 % of rows read vs 100 % for #69). Extra depth is not uniformly
cheaper: on strongly clustered keys it increases query-time reads
through extra witnesses. On unstructured Gaussians the saving remains
tiny and appears only at `epsilon = 0.1`. This is not a replacement of
softmax, `budgeted_softmax`, or A5 support pruning, and not a
usefulness, novelty, hardware, or FLAT-adoption claim.
