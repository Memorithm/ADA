# ADA-A4 E1 — Query/Key Coordinate-Box Bounds

## Scope

A4-E1 replaces the synthetic score upper bounds used by A4-E0 with query-specific page bounds derived from precomputed coordinate-wise key minima/maxima.

This remains a scalar deterministic correctness laboratory. It does not claim novelty, production readiness, bandwidth savings, or GPU speedup.

## Prior-art boundary

EntmaxKV (arXiv:2605.21649) uses page-level key-range statistics so pages can be scored without loading their full cached keys, and frames entmax sparse decoding as support recovery: when selected candidates contain the full entmax support, the sparse result is exact.

ADA-A4 does not claim coordinate-box page bounds as new. The research contribution being tested here is the composition of such a conservative page-score upper bound with the A4-E0 subset-threshold branch-and-bound certificate.

## Metadata

For each key page `p` and coordinate `j`, precompute

`k_min[p,j] = min_{i in p} k[i,j]`

and

`k_max[p,j] = max_{i in p} k[i,j]`.

This metadata can be constructed while page keys are already available, such as during prefill. Query-time bound evaluation then needs only the query and the min/max metadata.

## Query-time page upper bound

For a positive attention score scale `c`, define

`u_p(q) = c * sum_j max(q_j * k_min[p,j], q_j * k_max[p,j])`.

For every token `i` in page `p`, each coordinate satisfies

`q_j * k[i,j] <= max(q_j * k_min[p,j], q_j * k_max[p,j])`.

Summing coordinates and multiplying by positive `c` gives

`c * q dot k_i <= u_p(q)`.

Therefore `u_p(q)` is a conservative upper bound on every exact attention score in page `p` in real arithmetic.

E1 uses deterministic f64 evaluation and validates every generated bound against dense f64 scores in its oracle tests. This is a numerical qualification convention, not a formal IEEE-754 proof for arbitrary implementations or lower-precision metadata.

## Composition with A4-E0

A4-E0 proved that, after loading a subset `C`, the entmax threshold of that subset satisfies

`tau_C <= tau_full`.

Therefore an unloaded page `p` is safely prunable when

`(alpha - 1) * u_p(q) <= tau_C_lower`,

where `tau_C_lower` is the conservative lower endpoint retained by the subset threshold solver.

If the coordinate box is loose, the page remains unresolved and is loaded. The bound can therefore reduce pruning efficiency but cannot intentionally convert the algorithm into an approximate selector.

## Implementation split

E1 is isolated in `crates/ada-a4-qk-box` and depends on the already-qualified score-level `ada-a4-entmax-bnb` crate.

The split is deliberate:

1. `build_page_key_boxes` models prefill-time min/max metadata construction;
2. `query_box_upper_bounds` evaluates query-time bounds without page-key access;
3. `dense_qk_scores` exists only as an independent E1 oracle path;
4. `qk_box_entmax_case` verifies the generated bounds against the existing score-level conservative-bound validator;
5. `branch_and_bound_entmax_qk_box` reuses the unchanged A4-E0 exact branch-and-bound controller.

## E1 gates

A4-E1 may advance only if:

1. mixed-sign query coordinates still produce bounds above every dense score in each page;
2. exhaustive small Q/K fixtures find no bound violation;
3. alpha=1.5 and alpha=2.0 Q/K fixtures match the dense entmax oracle within the declared f64 tolerance;
4. no dense-support token is located in a pruned page;
5. an intentionally loose coordinate box degrades safely by loading more pages without changing the result;
6. positive custom/default attention scales are supported and non-positive scales are rejected by the E1 contract;
7. workspace fmt, strict clippy, tests, and doc tests are green.

Only after these gates should ADA measure page-pruning rates on representative Q/K distributions, separate metadata cost from avoided K/V traffic, or consider A5 hierarchical bounds and A2 V-late execution.