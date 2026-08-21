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

## Primary hypothesis

The key falsifiable hypothesis was that hierarchical subdivision would materially improve the `iid_uniform` regime where flat coordinate boxes failed despite sparse Entmax support.

The protocol declared the following interpretation before the run:

- **Hierarchy improves `iid_uniform` materially:** A5 is justified; next optimize bound evaluation (lazy traversal / better tree construction) before hardware work.
- **Hierarchy improves only already-clustered regimes:** the simple contiguous binary hierarchy is insufficient; investigate geometry-aware partitioning/index construction.
- **Hierarchy rarely improves flat boxes:** reject this A5 mapping and preserve the negative result.

None of these outcomes establishes novelty or production feasibility.

## Qualified run

The first clean committed E1 run used:

- survey source commit: `bfcda58bfa6af390bb900c01eb18f8187c6a7843`;
- 378 comparison cases;
- `survey_status=complete`;
- clean Git tree before and after the survey;
- preserved raw evidence: `evidence/a5-hierarchical-bounds/ada-a5-e1-flat-vs-hierarchy-bfcda58bfa6a-20260821T211306Z.txt`;
- raw SHA-256: `dc79857dfd9dab8dfa06f33d90bb068cb5bf0a71cf0a42bfccc7f6509b365d30`.

Dense-oracle probability/tau differences remained at f64 roundoff scale; the largest aggregate-reported difference was `1.110e-16`.

## Aggregate result

Mean additional logical score avoidance from the contiguous hierarchy relative to the flat page box:

| Regime | alpha | leaf /2 | leaf /4 | leaf /8 |
| --- | ---: | ---: | ---: | ---: |
| `iid_uniform` | 1.5 | 0.000000 | 0.000000 | 0.000000 |
| `iid_uniform` | 2.0 | 0.000000 | 0.000000 | 0.006696 |
| `page_clustered` | 1.5 | 0.015625 | 0.023810 | 0.066778 |
| `page_clustered` | 2.0 | 0.007440 | 0.036086 | 0.067336 |
| `dominant_page` | 1.5 | 0.000000 | 0.000000 | 0.000000 |
| `dominant_page` | 2.0 | 0.000000 | 0.000000 | 0.000000 |

The eager metadata cost rises with refinement. Mean hierarchy bound evaluations per token were:

- `/2`: `0.077009`;
- `/4`: `0.179688`;
- `/8`: `0.385045`.

For `page_clustered`, `/8` raises mean score avoidance from `0.447917` to `0.514695` at alpha=1.5 and from `0.666667` to `0.734003` at alpha=2.0. For `dominant_page`, the flat page box was already strong (`0.897321` mean score avoidance), so deeper contiguous refinement adds no logical pruning.

For the critical `iid_uniform` regime, the primary hypothesis is falsified for this mapping: alpha=1.5 gains nothing at any tested depth, while alpha=2.0 gains only `0.006696` at `/8` despite `0.385045` bound evaluations per token.

## Decision

E1 qualifies two distinct findings:

1. **Hierarchical refinement itself is useful when key geometry is locally coherent.** The `page_clustered` gains show that tighter sub-node bounds can eliminate work that a flat page box cannot.
2. **Contiguous binary subdivision is not a general solution to high-dimensional unstructured keys.** The `iid_uniform` result remains essentially dense even at `/8`.

Therefore the next A5 experiment must change geometry before optimizing traversal. A purely lazy version of the same contiguous tree could reduce metadata work in favorable cases, but it cannot make non-prunable `iid_uniform` subtrees become prunable.

The next candidate is a geometry-aware hierarchy with an independently safe bound family. A useful exact laboratory construction is a content-aware partition plus a hybrid bound that takes the minimum of independently conservative coordinate-box and enclosing-ball MIPS upper bounds. If that geometry materially improves the failed `iid_uniform` regime, lazy/on-demand bound evaluation becomes the next optimization step.

This result is synthetic algorithmic evidence only. It is not a model-distribution result, hardware benchmark, novelty claim, or production qualification.
