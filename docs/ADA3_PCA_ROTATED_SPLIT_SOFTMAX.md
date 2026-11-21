# ADA-A3 PCA-rotated split-dimension softmax (research, negative on cost)

Status: research reference, **ADAPT/negative**. Not a production runtime, no
hardware measurement, no novelty claim (no prior-art review; PCA/KLT rotation
before partial-distance pruning is classical in similarity search).

## Conjecture

`split_dim` (#merged) only helps when coordinate axes already concentrate key
energy in a prefix. An orthogonal `R` (rows = eigenvectors of `KᵀK`, Jacobi,
deterministic) preserves all scores but moves energy to leading coordinates for
*any* hidden rotation of an anisotropic spectrum. Cost: `d²` MACs per query to
rotate `q` (+ `n·d²` one-time prefill). Falsification: certificate violation vs
independent dense oracle on the ORIGINAL data; or no regime with net per-query
MACs/dense < axis `split_dim`.

## Implementation

`crates/ada-a3-certified-softmax/src/pca_rotate.rs`: `pca_rotation`,
`certified_pca_rotated_softmax_accounted`, `PcaRotateCost`. `split_dim` was
refactored to a shared `split_dim_core` (public behaviour unchanged).
The rotation is never trusted: non-orthonormality `ν=‖RRᵀ−I‖_F` and rotation
rounding `γ=8d^{3/2}ε` are charged as `|Δz| ≤ s·2(ν+γ)‖q‖‖k‖` per key and as
`2(e^{2η}−1)` L1 on the whole certificate.

## Falsification found during development (fixed)

1. A relative slack taken from the *rotated* norms is unsound for singular /
   non-orthogonal `R` (viol L1 = 0.954 with a Gaussian `R`): the error is relative
   to the ORIGINAL `‖q‖‖k‖`. Fixed via `original_norms` slack.
2. With every key read the certificate was 0 yet rotation rounding at magnitude
   ×1000 gave L1 error 1.7e-12. Fixed by charging `2(e^{2η}−1)` to the certificate.

## Results

Tests (all pass): rotation orthonormal/sorted; score preservation; identity
rotation ≡ `split_dim`; hidden-rotation witness (n=128,d=16, 2 hidden strong
directions: PCA split MACs < axis split MACs); 300 garbage/singular/huge/stale
rotations and 200 adversarial draws (outliers ×1000, duplicates, ε∈{0,1e-4,1e-2,.3}):
0 violations; fail-closed on bad layout/NaN rotation.

n=256, d=64, 20 draws, Gaussian keys/queries with std `e^{-decay·j}` optionally
hidden-rotated by a random orthogonal matrix; split m∈{4,8,16,32} best per draw.
MACs/dense (rows completed fraction); "pca+rot" includes the `d²` query rotation
(= +0.25 of `n·d`) but not amortised prefill:

| decay | hidden rot | ε | std | axis split | pca+rot |
|---|---|---|---|---|---|
| 0.15 | no | 1e-2 | 3 | 0.933 (0.922) | 1.183 (0.922) |
| 0.15 | yes | 1e-2 | 3 | 0.993 (0.987) | 1.183 (0.917) |
| 0.15 | yes | 1e-2 | 1 | 0.997 (0.996) | 1.240 (0.988) |
| 0.05 | yes | 1e-2 | 3 | 0.998 (0.996) | 1.237 (0.978) |
| 0 | any | 1e-2 | 1 | 0.998 | 1.248 |
| any | any | 1e-4 | any | ≈1.000 | ≈1.250 |

(Full 24-row table: `cargo test --release -p ada-a3-certified-softmax
pca_rotate::tests::print_results_table -- --ignored --nocapture`.)

## Verdict

**Negative (ADAPT reference).** The rotation does what it should structurally:
under hidden rotation the completed-row fraction returns to the axis-aligned
level (0.917 vs 0.922 at decay .15, std 3), recovering ~7 % rows that axis split
misses (0.987). But the `d²` per-query rotation (+25 % of `n·d` at n=256, d=64)
exceeds the saving in every tested regime: net MACs/dense 1.18–1.25 > 1. Breakeven
needs `n ≫ d` (overhead `d/n`) *and* strongly concentrated softmax; the row
fraction itself stays ≈1 at ε=1e-4. Possible follow-up: reduced rank rotation
(only `r` leading rows, `r·d` MACs, residual norm bound) — untested here.
Not measured: real Q/K traces (no trace data in this checkout), real q–key
covariance sharing, hardware. Logical MACs ≠ physical performance.
