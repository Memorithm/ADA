# ADA-A3 Hölder-mixed certified softmax (research, small positive / regime-limited)

Status: research reference, **ADAPT / mixed**. Not a production runtime, no
hardware measurement, no novelty claim (no prior-art review; ℓ1/ℓ∞ dual-norm
bounds are textbook, and norm-based MIPS pruning is standard).

## Conjecture

`z_i = s⟨q,k_i⟩ ≤ s·min(‖q‖₂‖k_i‖₂, ‖q‖∞‖k_i‖₁, ‖q‖₁‖k_i‖∞)`. For sparse keys against a
dense query, `‖q‖∞‖k‖₁` can be much tighter than Cauchy–Schwarz
(`‖q‖∞ ≈ 2.4σ` vs `‖q‖₂ ≈ 8σ` at d=64). Keys are visited in per-query descending
bound order; stop at `2T/(S+T) ≤ ε`, `T = Σ_unread exp(b_i − m)`.
Cost: 3 prefill norms per key, 3 query norms (`3d` non-MAC ops), **0 extra MACs**,
`n log n` per-query sort. The bound is never looser than `norm_sum`.

Falsification: L1 vs independent dense softmax > certificate + 1e-12; or reading
more rows than `norm_sum`; or no regime with fewer rows.

## Implementation

`crates/ada-a3-certified-softmax/src/holder.rs`:
`certified_holder_softmax[_accounted]`, `HolderCost`. Oracle: `dense_qk_scores`
dense stable softmax.

## Results (deterministic seeds)

- Each of the three inequalities checked on 2000 adversarial vector pairs
  (magnitudes spread over ×1e±3): 0 violations.
- Exhaustive grid (340 key sets × 3 queries × 4 ε incl. 0 = 4080 cases): 0 violations.
- Random sweep (nnz∈{1,3,16}, heavy-tail σ∈{0,1}, ε∈{1e-4,1e-2,1e-1}): 0 violations;
  never reads more rows than `norm_sum` (strictly fewer in a subset of draws).
- Witness (d=64, 63 one-sparse distractors, aligned needle): fewer rows than norm_sum.
- n=256, d=64, heavy-tail σ=.5, Gaussian q, 40 draws; rows read / n (= MACs/dense since no extra MACs):

| key nnz | scale | ε | holder | norm_sum |
|---|---|---|---|---|
| 1 | .5 | 1e-2 | 0.994 | 0.994 |
| 4 | .5 | 1e-2 | 0.974 | 0.986 |
| 8 | .5 | 1e-2 | 0.938 | 0.963 |
| 16 | .5 | 1e-2 | 0.941 | 0.946 |
| 64 (dense) | .5 | 1e-2 | 0.908 | 0.908 |
| any | .125 | 1e-2 | 0.996–1.000 | 0.997–1.000 |
| 8 | .5 | 1e-4 | 0.998 | 0.998 |

## Verdict

Valid certificate, free in MACs. Gain over `norm_sum` is real but small
(≤ 2.5 percentage points of rows, at moderate sparsity with scale .5/ε=1e-2); nil for
dense keys, nil at low scale or tight ε. Absolute savings stay bounded by softmax
concentration, as in prior cycles. Not measured: real Q/K sparsity (K is rarely
coordinate-sparse; rotations destroy sparsity), hardware, sort cost vs saved dots.
