# ADA-A3: quantized-key screening with certified exact rerank softmax

Research reference only. Q/K-level mechanism and falsification evidence. Not a
usefulness, novelty, hardware or FLAT-adoption claim. No prior-art review was
done (low-precision screening plus rerank is close to known quantized-KV and
MIPS practice; no novelty is claimed).

## Conjecture

Cycles #71–#74 order keys by geometric bounds (norm, angle, projection) and
fail on iid keys. A different bound source: a cheap *low-precision screen*. At
prefill each key row is quantized to `bits` bits (symmetric, per-row step) and
its exact L2 reconstruction error `e_i` stored. For any query,
`|z_i − z̃_i| ≤ s‖q‖e_i = δ_i` (Cauchy–Schwarz), so `z_i ≤ h_i = z̃_i + δ_i`.
Refine keys exactly in decreasing `h_i` order; the unread mass is bounded by
the exact suffix sum `Σ_rest e^{h_i}`; with read mass `S`,
`‖p − q‖₁ ≤ 2T/(S+T)`. Stop when `≤ ε`. Computed in log domain (no overflow).
The prediction: the exact-row fraction falls as the softmax gets peaky
(large logit spread) and as `bits` grow, and the net traffic
`bits/16 + exact_rows_frac` (against an fp16 baseline) beats 1 only for
peaky attention.

## Falsification

- L1 vs independent dense stable softmax > certificate + 1e-12.
- Interval `|z−z̃| ≤ δ` violated (would appear as an L1 violation).
- Net traffic ≥ 1 in all regimes (negative result).
- Unexplained `Err` or non-finite output on valid input.

## Implementation

`crates/ada-a3-certified-softmax/src/quant_screen.rs`,
`certified_quant_screen_softmax[_accounted]`. Slack: δ inflated by 1e‑9
relative plus an f64 dot-rounding slack `1e‑11·s‖q‖‖k‖` (head_dim ≤ 4096),
unread mass inflated by 1e‑9. `QuantScreenCost` counts logical work only:
`n` screen rows, exact rows read/skipped, elements at each precision, sort of
`n` bounds, transcendental evaluations. The screen is simulated in f64 with
integer codes; no real low-precision kernel and no hardware measurement. The
per-query sort of `n` bounds is not free and is not in the traffic number.

## Results (native run)

- Fail-closed on bad dimension (0, >4096), `bits` ∉ 2..=16, empty/misshaped/
  non-finite input, bad scale/epsilon. Zero query and equal keys: uniform, all
  read. Huge-magnitude inputs: finite output or `Err`.
- Quantization error recomputed independently for bits 2..=16: matches, and
  per-coordinate error ≤ step/2.
- Witness (1 needle, 63 weak distractors, d=8, 8 bits, ε=1e‑3): **1** exact row
  read, **63** skipped, needle mass > 0.999.
- Exhaustive grid (d=2, len 1..=4, 3 queries, 3 ε, bits {2,4}, **6120** cases):
  **0** violations; skipping in **2053**.
- Adversarial outlier keys (coordinate ×1000 on a third of the rows, bits
  {2,3,4,8}, 4 budgets, 20 draws each): **0** violations.
- Gaussian keys, d=32, n=128, 30 draws, `scale = logit_std/√d`. `frac` = mean
  exact rows read / n; `net` = bits/16 + frac (fp16 baseline, n·d elements);
  norm-order #71 reads 1.0000 in every cell.

| logit_std | bits | ε | frac | net | max L1 | cert max |
|---|---|---|---|---|---|---|
| 1 | 4 | 0.01 | 0.9615 | 1.2115 | 6.3e-3 | 1.0e-2 |
| 1 | 4 | 0.1 | 0.8203 | 1.0703 | 6.7e-2 | 1.0e-1 |
| 1 | 8 | 0.1 | 0.7529 | 1.2529 | 9.7e-2 | 1.0e-1 |
| 3 | 4 | 0.01 | 0.6302 | 0.8802 | 2.9e-3 | 1.0e-2 |
| 3 | 4 | 0.1 | 0.3440 | 0.5940 | 3.7e-2 | 1.0e-1 |
| 3 | 6 | 0.1 | 0.2461 | 0.6211 | 7.8e-2 | 1.0e-1 |
| 3 | 8 | 0.1 | 0.1812 | 0.6813 | 9.2e-2 | 1.0e-1 |
| 6 | 4 | 0.01 | 0.1945 | 0.4445 | 1.4e-3 | 1.0e-2 |
| 6 | 4 | 0.1 | 0.1117 | 0.3617 | 1.0e-2 | 1.0e-1 |
| 6 | 6 | 0.1 | 0.0458 | 0.4208 | 5.8e-2 | 1.0e-1 |
| 6 | 8 | 0.1 | 0.0299 | 0.5299 | 8.3e-2 | 1.0e-1 |

(Full 18-cell grid is printed by `random_sweep_and_cost_study`.) The
certificate is tight: observed max L1 is 0.6–0.99 of the budget in most
cells, unlike #71's 10²–10⁷× looseness.

## Verdict

**ADAPT (research reference), mixed.** First mechanism since #69 that skips
rows on iid Gaussian keys, with a tight, never-violated certificate. Skipping
is real only when attention is peaky (logit std ≥ 3); at std 1 the net traffic
is 1.07–1.45 (worse than dense fp16). Low bit widths (4) win net because the
screen cost `bits/16` dominates at high selectivity. Caveats: synthetic keys
only (no real Q/K traces), per-query sort of `n` bounds and the screen
pass over all keys are logical-count only, no hardware benchmark, exact rows
must be stored separately (memory is not reduced, only traffic), V reads not
modelled. Next: real Q/K traces, a selection rule that avoids the sort, and a
combined norm-bucket + quantization bound.
