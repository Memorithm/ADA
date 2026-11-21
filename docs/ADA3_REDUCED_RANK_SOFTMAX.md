# ADA-A3 reduced-rank projected screening softmax (research, mixed/regime-limited)

Status: research reference, **ADAPT**. Not a production runtime, no hardware
measurement, no novelty claim (no prior-art review; PCA/low-rank projection with
residual-norm bounds is classical in similarity search and partial-distance pruning).

## Conjecture

The full PCA rotation (#81) failed on cost because rotating `q` costs `d²` MACs.
Keep only the leading `r` rows `P` (`r×d`). Prefill stores `c_k = Pk` and
`ρ_k = ‖k − Pᵀc_k‖`. Per query: `q_r = Pq` (`r·d`), `‖u‖²` (`u = q − Pᵀq_r`) from the
prefill Gram `G = PPᵀ` (`r²`), screen `n·r`, then exact completion (`d` MACs per
completed key, computed from `q`,`k` directly) in decreasing upper-bound order.
Falsification: certificate violation vs independent dense oracle; or no regime with
per-query MACs/dense < 1.

## Soundness (any `P`, orthonormality never trusted)

`⟨q,k⟩ = ⟨q_r,c_k⟩ + ⟨u,k⟩` exactly (since `⟨Pᵀq_r,k⟩ = ⟨q_r,Pk⟩`), and
`⟨u,k⟩ = ⟨u,k−Pᵀc_k⟩ + ⟨(I−G)q_r,c_k⟩`, so
`|⟨q,k⟩ − ⟨q_r,c_k⟩| ≤ ‖u‖ρ_k + ν‖q_r‖‖c_k‖`, `ν = ‖I−G‖_F`. Rounding: `1e-11` relative
slack on dot/norm terms; `1e-12‖q‖²` additive slack inside the cancellation-prone `‖u‖²`.
Stop rule `2T/(S+T) ≤ ε` with log-domain unread-mass suffix sums (as `split_dim`).

## Implementation

`crates/ada-a3-certified-softmax/src/reduced_rank.rs`: `reduced_rank_basis`
(leading rows of `pca_rotation`), `certified_reduced_rank_softmax_accounted`,
`ReducedRankCost` (query projection, residual norm, screen, completion, prefill, dense).

## Tests (all pass)

Independent dense oracle on original data; invalid input rejection; `r=d` identity
basis ≡ dense (`ε=0`); zero-residual screen skips rows; 300 garbage bases (random
non-orthogonal, singular zero, magnitude 1e-3/50, stale) × ε∈{0,1e-4,1e-2,.3}: 0
violations; 200 adversarial draws (hidden rotation, outlier ×1000, duplicates,
near-zero query, `r`∈1..8): 0 violations; cost identities; planted-key witness
(n=1024,d=64,r=8): MACs < dense.

## Results (synthetic, logical MACs)

Keys: Gaussian, std `e^{-decay·j}`, hidden random rotation; "aligned" = query from the same
covariance, else isotropic. Scale set so logit std = `std`. 10 draws, best rank
`r∈{2,4,8,16,32}` per cell (chosen post hoc, so slightly optimistic). MACs/dense
includes `q_r`, `‖u‖`, screening and completion, **excludes** prefill
(`2nrd + r²d`, amortised over queries). Rows fraction in parentheses.

| n | decay | q aligned | ε | std | best r | MACs/dense (rows) |
|---|---|---|---|---|---|---|
| 256 | .15 | yes | 1e-2 | 3 | 8 | **0.654** (0.493) |
| 256 | .15 | no | 1e-2 | 3 | 16 | 0.912 (0.584) |
| 256 | .05 | yes | 1e-2 | 3 | 16 | 1.015 (0.687) |
| 256 | .15 | yes | 1e-2 | 1 | 2 | 1.021 (0.981) |
| 256 | .15 | yes | 1e-4 | 3 | 2 | 1.008 (0.969) |
| 4096 | .15 | yes | 1e-2 | 3 | 8 | **0.607** (0.479) |
| 4096 | .15 | no | 1e-2 | 3 | 16 | **0.794** (0.539) |
| 4096 | .05 | yes | 1e-2 | 3 | 32 | 0.960 (0.448) |
| any | 0.05 | no | any | any | 2 | 1.032–1.039 |
| any | any | any | 1e-4 or std 1 | | 2 | 1.004–1.039 |

Full 24-row table: `cargo test --release -p ada-a3-certified-softmax
reduced_rank::tests::print_results_table -- --ignored --nocapture`.

## Verdict

**Mixed, regime-limited (ADAPT).** First screening variant in this line to beat dense
MACs by a clear margin (0.61–0.65 of dense) after paying all per-query costs, versus
1.18–1.25 for the full rotation on a comparable (not identical) setup in #81. It
only wins when (a) key energy is concentrated in a low-rank subspace (decay .15),
(b) the softmax is peaked (logit std 3) and ε is loose (1e-2); with query not
aligned it degrades (0.79–0.91); flat attention (std 1), tight ε (1e-4) or weak
anisotropy yields ≈1.0–1.04 (slight loss from `n·r` screening). Because the exact
completion recomputes the full `d`-MAC dot, rows completed cost `d`, not `d−r`.
Not tested: real Q/K traces (still unavailable in this checkout; the central open
question is whether real queries are aligned with the key spectrum and how peaked
real heads are), basis staleness under decoding drift, hardware. Logical MACs ≠ physical performance.

## Next

1. Real Q/K traces to measure alignment/peakedness (gates all screening work).
2. Two-stage ranks (r₁ ≪ r₂) refining only survivors.
3. Sorted-tail softmax / adaptive witnesses.
