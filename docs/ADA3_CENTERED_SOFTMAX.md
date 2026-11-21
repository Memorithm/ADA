# ADA-A3 mean-centered Cauchy–Schwarz certified softmax (research, mixed/limited)

Status: research reference, **ADAPT / mixed**. Not a production runtime, no
hardware measurement, no novelty claim (no prior-art review; mean-centering
before norm bounds is a standard trick in maximum-inner-product search).

## Conjecture

With key mean `μ` and `c_i = k_i − μ`: `z_i = s⟨q,μ⟩ + s⟨q,c_i⟩ ≤ s⟨q,μ⟩ + s‖q‖‖c_i‖`.
Real keys often share a large common component, so `‖c_i‖ ≪ ‖k_i‖` and the
centered bound can be far tighter than plain `s‖q‖‖k_i‖`. Use
`b_i = min(plain, centered)`, visit keys by per-query descending `b_i`
(non-MAC sort), stop at `2T/(S+T) ≤ ε`, `T = Σ_unread exp(b_i − m)`.
Extra cost: one `⟨q,μ⟩` dot (`d` MACs). Rounding slack added to cover the
computed mean/centered rows (catastrophic cancellation tested).

Falsification: L1 vs independent dense softmax > certificate + 1e-12; or no
regime where net MACs/dense < norm_sum.

## Implementation

`crates/ada-a3-certified-softmax/src/centered.rs`:
`certified_centered_softmax[_accounted]`, `CenteredCost` (prefill norms,
`extra_query_macs`, per-query sorted keys, key rows read, exp counts).
Oracle: `dense_qk_scores` dense stable softmax.

## Results (debug build, deterministic seeds)

- Certificate: exhaustive grid (1020 cases), 4×3×4×30 random offset/heavy-tail
  sweeps, ±1e9 offset cancellation case: 0 violations.
- Designed witness (offset (30,30) common, one outlier needle): centered reads 1
  row, norm_sum reads 64/64.
- n=256, d=64, keys = offset·N(0,I) + e^{0.5 N}·N(0,I), Gaussian q; net MACs/dense
  incl. `⟨q,μ⟩` vs norm_sum rows/dense:

| offset | scale | ε | centered | norm_sum |
|---|---|---|---|---|
| 0 | .125 | 1e-2 | 0.996 | 0.992 |
| 0 | .5 | 1e-2 | 0.931 | 0.928 |
| 0.5 | .5 | 1e-2 | 0.932 | 0.974 |
| 1 | .5 | 1e-2 | 0.929 | 0.999 |
| 2 | .5 | 1e-2 | 0.924 | 1.000 |
| any | .125 | 1e-2 | ≈1.00 | ≈1.00 |
| any | .125 | 1e-4 | 1.004 | 1.000 |

## Verdict

Valid certificate. Centered bound removes the common-offset penalty (norm_sum
degrades to 1.0 as offset grows; centered stays ≈ offset-0 level), a
7–8 % MAC saving only at scale 0.5/ε=1e-2; no gain at low scale or tight ε and
up to +0.4 % overhead there (the `μ` dot). Not measured: real Q/K traces
(where offsets are real but q is correlated with them), hardware. Gain bounded
by softmax concentration, as in prior cycles.
