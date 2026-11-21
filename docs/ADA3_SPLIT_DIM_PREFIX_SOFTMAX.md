# ADA-A3 split-dimension prefix screening (research, negative on cost)

Status: research reference, **ADAPT/negative**. Not a production runtime, no
hardware measurement, no novelty claim (prior-art review not done; partial-dot
early abandoning with Cauchy–Schwarz tail bounds is a classical idea in
similarity search).

## Conjecture

Split the head dimension at `m`. Prefill stores one tail norm `r_i = ‖k_i[m..]‖`
per key. Per query, the prefix dot `p_i` for all keys (`n·m` MACs) gives
`z_i ≤ p_i + s‖q[m..]‖ r_i`. Keys are completed in decreasing bound order,
*reusing* `p_i` and reading only `d−m` more coordinates, until
`2T/(S+T) ≤ ε` (T = exact log-domain suffix sum of unread bounds). Unlike the
quantized screen (#75) nothing is re-read: MACs `= n·m + completed·(d−m) ≤ n·d`.
Expected win: anisotropic (decaying-spectrum) keys/queries where tail energy is small.

Falsification: L1 vs independent dense softmax > certificate + 1e-12; MACs > n·d;
no regime with MACs/dense materially < 1.

## Implementation

`crates/ada-a3-certified-softmax/src/split_dim.rs`,
`certified_split_dim_softmax[_accounted]`, `SplitDimCost` (prefix MACs,
completion MACs, rows completed/skipped, sort size, transcendentals).
Oracle: dense stable softmax (`dense_qk_scores`). Sort of `n` bounds not counted
in MACs.

## Bug found and fixed (also in merged #75 `quant_screen`)

The unread-mass suffix sum was accumulated as `Σ exp(h_i − h_max)`. When one
loose bound `h_max` sits thousands of nats above the rest (huge-magnitude
outlier coordinates), the sum underflows to 0, `ln T = −∞`, and the certificate
is reported as `0` although a heavy key was skipped (observed L1 error 2 with
certificate 0). Fixed in both modules by a log-domain backward log-sum-exp.
Regression test `loose_huge_upper_bound_does_not_underflow_unread_mass`
(fails on the old code: `viol 2`, passes now). Other A3 modules use different
bounds and do not share this pattern (checked by grep for suffix sums).

## Results (native, f64)

- Fail-closed on bad dim, split ∉ 1..=d, empty/misshaped/non-finite input, bad
  scale/epsilon.
- Exhaustive grid (d=3, len 1..=4, values {−2,0,1.5}, 3 queries, 3 ε, split 1..3):
  0 violations, skipping occurs.
- Adversarial sweep (400 draws, outlier tail coordinates ×1000, duplicated rows,
  huge queries, ε ∈ {0,1e-4,1e-2,0.3}): 0 violations after the fix (1 before).
- Witness (1 needle, 63 distractors, d=8, m=4, ε=1e-3): 1 row completed.
- Gaussian, n=256, d=64, 30 draws, per-coordinate std `exp(−decay·j)`, scale
  `std/Σσ²`; MACs/dense (rows completed fraction); all max L1 ≤ certificate:

| decay | logit std | ε | m=16 | m=32 |
|---|---|---|---|---|
| 0 | 1 | 1e-2 | 0.9999 | 0.9984 |
| 0 | 3 | 1e-2 | 1.0000 | 0.9999 |
| 0.05 | 3 | 1e-2 | 0.9884 | 0.9886 |
| 0.15 | 1 | 1e-2 | 0.9906 | 0.9938 |
| 0.15 | 3 | 1e-2 | **0.9294** | 0.9523 |
| 0.15 | 3 | 1e-4 | 0.9965 | 0.9976 |
| any | any | 1e-4 (others) | 1.0000 | 1.0000 |

(Reproduce: `cargo test --release -p ada-a3-certified-softmax print_results_table -- --ignored --nocapture`.)

## Verdict

**Negative on cost.** Certificate valid, never worse than dense in MACs, but the
best observed saving is 7 % (strongly decaying spectrum, ε=1e-2); isotropic and
tight budgets give none, because Cauchy–Schwarz tail bounds are loose by a
factor ~√(tail dim). Untested: real Q/K traces (absent), learned rotations
(PCA) that would concentrate energy in the prefix, hardware.
Positive byproduct: latent underflow bug in #75 fixed.
