# ADA-A3 two-stage rank lazy-refinement softmax (research, NEGATIVE)

Status: research reference, **REJECT as a cost improvement** (sound, but never beats the
best single-rank screen). Not a production runtime, no hardware measurement, no novelty
claim (no prior-art review; multi-resolution/cascaded partial-distance bounds are classical).

## Conjecture

Single-rank screening (#82) pays `n·r` MACs for all keys. Screen with rank `r1` (`n·r1`),
refine a key to rank `r2 > r1` only when it reaches the top of the bound frontier
(`r2−r1` extra MACs, first `r1` coefficients reused as a prefix), then complete exactly
(`d` MACs). Falsification: certificate violation vs dense oracle, or no regime where
two-stage MACs/dense < best single-rank MACs/dense.

## Soundness

Both `h¹ᵢ` and `h²ᵢ` are valid score upper bounds for ANY basis (same derivation as
`reduced_rank`), hence so is `min(h¹ᵢ, h²ᵢ)`. Stop rule `2T/(S+T) ≤ ε`, `T = Σ_{unread}
e^{bound}` recomputed each step (positive terms, no cancellation).

## Implementation

`crates/ada-a3-certified-softmax/src/two_stage.rs`: `certified_two_stage_softmax_accounted`,
`TwoStageCost` (stage-1, stage-2, completion MACs, refinements, prefill, dense, and the
non-MAC frontier scan work: `n` per step in this reference — a real implementation would
use a heap; excluded from MAC totals).

## Tests (all pass)

Independent dense oracle; invalid input rejection; full-rank identity at ε=0 ≡ dense;
equal ranks ≡ single-stage rows/probabilities; 250 garbage-basis draws (random, singular,
1e-3/50 magnitude, stale) × ε∈{0,1e-4,1e-2,.3}: 0 violations; 150 adversarial draws
(hidden rotation, ×1000 outlier, duplicate, near-zero query) × 4 ε: 0 violations; cost
identities (`stage1=n·r1`, `stage2=refinements·(r2−r1)`, `completion=rows·d`).

## Results (synthetic, logical MACs, same generator as reduced-rank doc)

Best single rank r∈{2,4,8,16,32} vs best pair (r1,r2)∈{(2,8),(2,16),(4,16),(4,32),(8,32)},
each chosen post hoc per cell (equally optimistic). 10 draws (n=256), 4 draws (n=4096).
MACs/dense includes projection, residual norm, screening, refinement, completion; excludes prefill.

| n | decay | aligned | ε | std | best single | best two-stage |
|---|---|---|---|---|---|---|
| 256 | .15 | yes | 1e-2 | 3 | r=8 **0.654** | (2,8) 0.672 |
| 256 | .15 | no | 1e-2 | 3 | r=16 0.912 | (2,16) 0.913 |
| 256 | .05 | yes | 1e-2 | 3 | r=16 1.015 | (2,16) 1.015 |
| 4096 | .15 | yes | 1e-2 | 3 | r=8 **0.618** | (2,16) 0.628 |
| 4096 | .15 | no | 1e-2 | 3 | r=16 0.719 | (2,16) 0.720 |
| 4096 | .05 | yes | 1e-2 | 3 | r=32 0.955 | (4,32) 0.956 |
| all flat/tight-ε/weak-anisotropy cells | | | | | 1.008–1.039 | 1.009–1.160 |

Full 24-row output: `cargo test --release -p ada-a3-certified-softmax
two_stage::tests::print_results_table -- --ignored --nocapture`.

## Verdict

**Negative.** Two-stage was never better than the best single rank in any of 24 cells
(equal within 0.01 in the winning regimes, up to +0.13 worse where screening does not
pay). Reason: in the winning regime, almost every key that reaches the frontier must be
completed anyway, so refinement is mostly wasted `r2−r1` MACs, while the low `r1` bound is
too loose to prune the tail on its own. The frontier is the bottleneck, not bound
tightness per key. Not tested: real Q/K traces (still unavailable), basis staleness,
hardware. Logical MACs ≠ physical performance.

## Next

1. Real Q/K traces (alignment, peakedness) gate all screening work.
2. Sorted-tail softmax / adaptive witnesses; or move away from certified skipping to
   a different mechanism class (e.g. exact-with-shared-state across queries).
