# ADA-A3 temporal anchor certificate reuse (research, negative/limited)

Status: research reference, **ADAPT/negative on net cost**. Not a production
runtime, no hardware measurement, no novelty claim (no prior-art review;
incremental/cached score bounds and triangle-inequality pruning across
successive queries are classical in similarity search and caching).

## Conjecture

Decode queries `q_t` drift slowly. If the exact score `a_i = s⟨a,k_i⟩` of key
`i` was computed for an earlier query `a`, then
`z_i(q) ≤ a_i + s‖q−a‖‖k_i‖` with no read of `k_i`. Each key stores one anchor
(`a`, `a_i`); a step costs one `d`-MAC distance per distinct anchor, ranks keys
by bound, completes exactly in decreasing bound order (completed keys re-anchor
at `q`), and stops when `2T/(S+T) ≤ ε` (`T` = log-domain sum of unread bounds).
New (appended) keys are always completed. Cost per step
`distinct_anchors·d + completed·d` vs dense `n·d`.

Falsification: L1 vs independent dense softmax > certificate + 1e-12; unread
key count inconsistent; no regime with MACs/dense materially < 1.

## Implementation

`crates/ada-a3-certified-softmax/src/temporal_anchor.rs`:
`TemporalAnchorScreen::{new, step}`, `TemporalAnchorCost`. State is untouched on
error. The key store must be append-only; a per-row FNV-1a fingerprint + norm
check fails closed if a stored row changed (this guard is `old_rows·d` loads per
step, reported as `integrity_reads`, not MACs; it would defeat the memory-traffic
benefit if kept in production, so any such benefit would rely on the contract).
Oracle: dense stable softmax (`dense_qk_scores`), independent of the screen.

## Results (native, f64)

- Fail-closed: bad dim/scale/epsilon/query/layout/non-finite, shrunk store,
  modified row (including a permutation preserving norm and coordinate sum, which
  the first fingerprint — norm + sum — missed; replaced by a bitwise hash).
- Witness (needle + 63 distractors, same query repeated, ε=1e-3): second step
  completes ≤ 2 rows, MACs < n·d/4.
- Adversarial random walks (60 draws, steps 0/0.01/0.3/3, magnitudes ×1/10/1000
  outlier keys, duplicated rows, appends, ε ∈ {0,1e-3,0.1}, every ε on a cloned
  state): 0 violations.
- Table: n=256→319 over 64 steps (1 key appended per step), d=64, iid Gaussian,
  query random walk renormalised to constant norm, scale = std/d, 10 draws,
  MACs/dense (all max L1 ≤ certificate). `step` = per-coordinate query noise
  (relative ‖Δq‖/‖q‖ ≈ step); step 0 is the ceiling with exact anchors.

| step | std 3 ε1e-2 | std 8 ε1e-2 | std 16 ε1e-2 | std 16 ε1e-4 |
|---|---|---|---|---|
| 0 | 0.998 | 0.963 | **0.776** | 0.986 |
| 0.02 | 1.002 | 0.986 | 0.861 | 0.999 |
| 0.1 | 1.001 | 0.988 | 0.927 | 1.001 |
| 0.3 | 1.003 | 1.000 | 0.988 | 1.003 |
| 1 | 1.004 | 1.004 | 1.004 | 1.004 |

(ε=1e-4 and std ≤ 8 are ≈1.00 everywhere; the `distinct_anchors·d` overhead
makes the ratio up to 1.0035 > 1, i.e. worse than dense.)

## Verdict

**Negative.** The certificate is valid, but even with *exact* anchors (step 0)
the gain is bounded by how concentrated the softmax is: for logit std ≤ 8 and
n=256 almost every key is needed for ε ≤ 1e-2. Only very peaked attention
(std 16) with near-static queries gives 0.78–0.93; relative query drift ≥ 0.3
removes the gain, and the distance overhead makes it slightly worse than dense.
Real decode query autocorrelation is unmeasured (no Q traces here), as are
peakedness and hardware. Next: measure real Q drift/peakedness; per-layer
anchors with lower bounds for exact-pruned keys.
