# ADA-A5 E1 — Flat Box vs Hierarchical Bound Survey

## Purpose

A4-E2 established two simultaneous facts on the deterministic 126-case synthetic fixture family:

1. exact subset-threshold Entmax pruning is useful when Q/K page bounds are tight enough;
2. flat coordinate-wise page boxes can be extremely loose in high-dimensional unstructured regimes, including `iid_uniform` where sparsemax support is small but every page is still loaded.

A5-E0 introduced nested coordinate boxes inside each outer KV page and passed scalar correctness qualification. E1 measures whether that refinement actually reduces logical score loading relative to the flat A4-E1 page-box controller.

This is an **algorithmic work survey**, not a wall-clock benchmark.

## Frozen fixture family

E1 intentionally reuses the A4-E2 synthetic family without changing its distributions:

- seven `(N, D, page_size)` shapes;
- `iid_uniform`, `page_clustered`, and `dominant_page` regimes;
- alpha = 1.5 and alpha = 2.0;
- the same three deterministic seeds;
- attention scale `1/sqrt(D)`.

That is 126 base fixtures.

For each base fixture, E1 evaluates three hierarchy granularities:

- `leaf_divisor = 2`;
- `leaf_divisor = 4`;
- `leaf_divisor = 8`.

`leaf_size = ceil(page_size / leaf_divisor)`, producing 378 flat-vs-hierarchy comparison cases.

## Correctness gate

For every comparison case, E1 computes:

- the dense Entmax oracle;
- the flat A4 Q/K-box branch-and-bound result;
- the A5 hierarchical branch-and-bound result.

The survey aborts if either candidate exceeds the declared probability/tau tolerances or if either candidate removes any token/page containing dense-oracle support.

The hierarchy still uses the exact A4 certificate

\[
(\alpha-1)u_v(q) \le \tau_{C,\mathrm{lower}}
\]

before pruning an entire node/subtree.

## Primary metrics

Per case and aggregate, E1 reports:

- flat logical score avoidance;
- hierarchical logical score avoidance;
- additional score avoidance from hierarchy relative to flat boxes;
- hierarchical loaded-token count relative to flat loaded-score count;
- hierarchy bound evaluations per token;
- expanded nodes;
- pruned subtrees;
- threshold solves;
- dense-oracle probability/tau differences.

A positive `additional_score_avoidance` means the hierarchy avoided scores that the flat page-box controller still had to load. Negative values are retained as evidence rather than hidden.

## Critical caveat: eager bounds

A5-E0 currently evaluates every hierarchy-node bound before traversal. Therefore

`bound_evaluations = node_count`

for a query/index pair.

E1 must not interpret reduced token loading as a wall-clock, FLOP, memory-traffic, or hardware speedup. It explicitly measures the tradeoff between tighter hierarchical pruning and additional bound work.

If E1 shows useful pruning but excessive bound evaluations, the next candidate should make child-bound evaluation lazy/on-demand while preserving the same exact pruning certificate.

## Primary hypothesis

The key falsifiable hypothesis is that hierarchical subdivision materially improves the `iid_uniform` regime where flat coordinate boxes failed despite sparse Entmax support.

Possible outcomes:

- **Hierarchy improves `iid_uniform` materially:** A5 is justified; next optimize bound evaluation (lazy traversal / better tree construction) before hardware work.
- **Hierarchy improves only already-clustered regimes:** the simple contiguous binary hierarchy is insufficient; investigate geometry-aware partitioning/index construction.
- **Hierarchy rarely improves flat boxes:** reject this A5 mapping and preserve the negative result.

None of these outcomes establishes novelty or production feasibility.
