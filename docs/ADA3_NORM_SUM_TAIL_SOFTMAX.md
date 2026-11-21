# ADA-A3 norm-order softmax with per-key tail sum (research)

Status: research reference, not production; no novelty claim (no prior-art review).

## Conjecture

The norm-ordered early-termination certificate (`norm_order.rs`) bounds the unread mass by
`(n−j)·exp(B_j − m)` with `B_j` the bound of the *largest* unread norm. Replacing it with the
per-key sum `Σ_{r≥j} exp(b_r − m)` (suffix log-sum-exp over the norm-sorted bounds, no MACs) is
valid and never looser, so the scan stops no later. Falsification: any L1 violation versus a dense
oracle, or any case where it reads more rows than `norm_order`.

## Implementation / evidence

`crates/ada-a3-certified-softmax/src/norm_sum.rs`. Log-domain suffix table (overflow-safe),
1e-9 relative slack on the tail. Independent oracle: dense stable softmax from scalar dots.
Tests: invalid inputs, exhaustive grid (1020 cases, 0 violations), heavy-tailed random sweeps,
extreme-bound (key norm ×1000, scales up to 8) sweep, and a dominance test (reads ≤ norm_order in
every draw; strictly fewer in 136 draws). Heavy loops reduced under `cfg!(miri)`.

## Measurements (n=256, d=64, 40 draws, fraction of rows read; sum / norm_order)

| key log-norm σ | scale | ε | sum | order |
|---|---|---|---|---|
| 0.5 | .125 | 1e-2 | 0.999 | 0.999 |
| 0.5 | .5 | 1e-2 | 0.890 | 0.911 |
| 1 | .125 | 1e-2 | 0.660 | 0.715 |
| 1 | .5 | 1e-2 | 0.371 | 0.403 |
| 1 | .125 | 1e-4 | 0.759 | 0.793 |
| 1.5 | .125 | 1e-2 | 0.204 | 0.241 |
| 1.5 | .5 | 1e-2 | 0.122 | 0.127 |
| 1.5 | .125 | 1e-4 | 0.186 | 0.211 |

Cost: saves up to ~5.5 points of key-row reads (d MACs each); adds a per-query `2n` exp/ln table
(`bound_table_evaluations`, no MACs). Logical counts only; no hardware claim. With equal key norms
it gains nothing (bound is still loose; ≈1.0).

## Verdict

**Small positive, ADAPT-level.** A strictly dominant, cheap refinement of an existing
certificate, worth 0–8 % relative fewer rows in heavy-tailed-norm regimes; does not fix the
Cauchy–Schwarz looseness. Not tested: real Q/K traces, hardware.
