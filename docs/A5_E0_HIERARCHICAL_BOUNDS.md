# ADA-A5 E0 — Hierarchical Safe Pre-KV Bounds

## Motivation from A4-E2

A4-E2 showed that flat coordinate boxes are exact/conservative in the scalar laboratory but can be too loose to expose already-sparse entmax support. In the clean synthetic survey, `iid_uniform, alpha=2.0` retained only `2.2879%` of tokens on average while the flat page boxes still forced `100%` page loading. By contrast, page-coherent regimes achieved substantial certified score avoidance.

A5 asks whether nested boxes can preserve A4 exactness while reducing the geometric looseness of one flat page bound.

## E0 hypothesis

Let an outer KV page be recursively partitioned into nested token subsets. Every node `v` stores coordinate-wise key extrema over its subtree:

`K_min[v,j] = min_{i in subtree(v)} K[i,j]`

`K_max[v,j] = max_{i in subtree(v)} K[i,j]`.

For positive attention scale `c`, query-time node bound is

`u_v(q) = c * sum_j max(q_j K_min[v,j], q_j K_max[v,j])`.

In exact arithmetic, `u_v(q)` upper-bounds every scaled QK score in node `v`.

A4 already provides a conservative lower endpoint `tau_lower` for the entmax threshold on the loaded subset. Therefore the unchanged exact pruning rule is

`(alpha - 1) * u_v(q) <= tau_lower  =>  every token in subtree(v) is outside the full entmax support`.

If the inequality does not hold for an internal node, A5 expands that node and tests tighter child boxes. Only an unresolved leaf causes score loading.

## Separation from A4

A5 does not change:

- the alpha-entmax support condition;
- the subset-threshold monotonicity argument;
- the conservative use of the lower endpoint of the threshold bracket;
- the dense fallback guarantee.

A5 changes only the bound hierarchy and the granularity at which unresolved key ranges are loaded.

## E0 data structure

`ada-a5-hierarchical-bounds` builds one binary hierarchy inside each outer KV page.

- Leaves contain at most `leaf_size` contiguous keys.
- Parent min/max metadata is merged from children.
- Outer page boundaries remain explicit.
- A deterministic key fingerprint prevents accidentally pairing an index with another key matrix in the laboratory.

This is a research representation, not a production KV-cache format.

## Query-time search

E0 uses a deterministic highest-upper-bound descent to obtain the first leaf, then iterates:

1. solve entmax on currently loaded scores and retain `tau_lower`;
2. prune any frontier subtree satisfying the exact certificate;
3. choose the unresolved frontier node with highest upper bound;
4. expand it if internal, or load it if it is a leaf;
5. repeat until every token range is either loaded or certified pruned.

The final distribution is computed only on loaded scores and scattered back into full token order with zero probability on pruned ranges.

## Numerical safety in E0

The coordinate-box inequality is an exact-arithmetic fact. Ordinary `f64` evaluation is not automatically a directed-rounding proof. Therefore E0 deliberately computes dense QK scores in the laboratory and checks every hierarchy-node bound against the actual dense maximum before allowing pruning.

A production implementation must replace this oracle-side check with a numerically certified bound implementation (for example, outward rounding or a proved conservative error envelope). E0 makes no production numerical-certification claim.

## Declared correctness gates

E0 must pass:

- workspace formatting;
- strict Clippy;
- all existing A4/A1 workspace tests;
- hierarchy bounds dominate dense scores on mixed-sign Q/K;
- child bounds do not exceed parent bounds beyond the declared scalar tolerance in the test fixture;
- dense-oracle probability/tau parity for alpha=1.5 and alpha=2.0;
- support preservation;
- a fixture where hierarchy loads fewer tokens than the flat outer-page bound;
- a fixture that safely degrades to loading every leaf;
- index/key mismatch rejection;
- exhaustive small hierarchy states for two leaf granularities.

No A5 correctness status is granted before these gates run green on the target checkout.

## Metrics

The E0 result reports logical algorithmic work:

- hierarchy nodes total;
- internal nodes expanded;
- certified subtrees pruned;
- node bounds evaluated;
- leaves total/loaded;
- tokens loaded/pruned;
- threshold solves.

These are not hardware traffic or latency measurements.

## Prior art and novelty discipline

Hierarchical branch-and-bound for inner-product search is established prior art. Ram & Gray, *Maximum Inner-Product Search using Tree Data-structures* (2012), develops exact tree-based branch-and-bound for MIPS. EntmaxKV (2026) provides the entmax-specific pre-KV support-recovery setting and a flat page min/max bound.

A5-E0 is therefore treated as an engineering/research composition to test inside ADA. No novelty claim is made. Any later novelty assessment would require a substantially broader literature and implementation comparison.

## Promotion criterion

If E0 correctness is green, A5-E1 should compare flat A4 boxes against hierarchical bounds on exactly the same deterministic Q/K survey fixtures, including the `iid_uniform` failure regime from A4-E2. The primary question is not wall-clock speed yet, but whether hierarchy materially lowers tokens loaded after accounting for its additional metadata/bound evaluations.
