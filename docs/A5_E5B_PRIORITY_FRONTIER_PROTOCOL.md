# ADA-A5 E5b — Exact Priority-Frontier Lazy Bounds

## Purpose

ADA-A5 E5 qualified exact lazy hierarchical bound evaluation on the frozen
Qwen3 natural Q/K slice, but exposed a controller-cost problem.

At the globally Pareto-optimal tested configuration
`page_size=16 / leaf_divisor=8`, E5 obtained:

- alpha 1.5:
  - weighted score avoidance: `0.708539`
  - weighted bound avoidance: `0.167199`
  - weighted bound evaluations / pruned token: `1.101916`
  - weighted bound requests / pruned token: `80.490612`

- alpha 2.0:
  - weighted score avoidance: `0.774436`
  - weighted bound avoidance: `0.208171`
  - weighted bound evaluations / pruned token: `0.958555`
  - weighted bound requests / pruned token: `61.499173`

The large gap between bound evaluations and bound requests comes from repeated
frontier rescans of already-known bounds.

E5b asks:

> Can the same exact hierarchical certificate be driven by an ordered frontier
> without repeatedly requesting cached node bounds?

E5b is an algorithmic controller experiment. It is not a wall-clock,
GPU, bandwidth, latency, or production qualification.

## Historical E5 controller

The E5 lazy controller remains unchanged and preserved.

Its frontier is a `Vec<node_index>`.

For a fixed threshold lower bound it repeatedly:

1. scans all frontier nodes for pruning;
2. rebuilds the unresolved frontier;
3. scans all unresolved nodes again to locate the maximum bound;
4. expands one node;
5. repeats until a leaf is loaded.

Each node bound is cached, so repeated scans avoid recomputing the
D-dimensional coordinate-box bound, but they still produce a large number of
logical bound requests.

## E5b ordered frontier

E5b introduces a second controller using an ordered frontier.

Every frontier entry stores:

- node index;
- already-computed safe upper bound.

The frontier is ordered by bound.

It must support:

- access/removal of the minimum-bound node for threshold pruning;
- access/removal of the maximum-bound node for expansion.

The initial implementation may use Rust `BTreeSet`.

No approximation is introduced.

## Bound-evaluation invariant

A hierarchy-node bound may be evaluated at most once.

The E5b laboratory implementation keeps an explicit per-node evaluated bit.

A second attempt to evaluate the same node is an error.

Therefore E5b does not use the historical repeated `lazy_bound()` cache-hit
path.

For every evaluated node the bound remains:

\[
u_v(q)
=
c
\sum_j
\max(q_j k^{v,j}_{min}, q_j k^{v,j}_{max}).
\]

As in E5, the evaluated f64 bound is checked against the dense research oracle.

## Pruning invariant

The pruning certificate is unchanged:

\[
(\alpha - 1)u_v(q)
\le
\tau_C^{lower}.
\]

Because the loaded subset only grows, the subset threshold lower bound is
monotone non-decreasing.

At a fixed threshold, if the smallest frontier bound cannot be pruned, no
larger frontier bound can be pruned.

Therefore an ordered frontier can prune from its minimum endpoint and expand
from its maximum endpoint without rescanning every already-resolved bound.

## Expansion

When an unresolved internal node is selected:

1. remove the maximum-bound frontier entry;
2. evaluate each child bound once;
3. insert each child as a bound-carrying frontier entry;
4. continue pruning/selection at the current threshold.

When a leaf is selected, load its tokens and recompute the subset Entmax
threshold exactly as before.

## Tie semantics

The historical `Vec` controller chooses the first maximum in its current vector,
whose order can be modified by `swap_remove`.

E5b instead uses a deterministic ordered-frontier tie-break by node index.

Exact equal-bound ties may therefore produce a different valid traversal and a
different loaded-token superset.

Consequently:

- dense Entmax parity is mandatory;
- dense-support preservation is mandatory;
- deterministic behavior is mandatory;
- equality with historical E5 loaded-token sets is measured and preferred,
  but is not a mathematical correctness requirement under exact bound ties.

Any natural-trace loaded-set difference must be reported explicitly.

## E5b metrics

Report at least:

- nodes total;
- nodes expanded;
- subtrees pruned;
- bound evaluations;
- nodes never evaluated;
- frontier insertions;
- frontier minimum checks;
- frontier maximum pops;
- leaves total;
- leaves loaded;
- tokens loaded;
- tokens pruned;
- threshold solves.

Derived logical metrics may include:

- bound evaluation fraction;
- bound avoidance;
- score avoidance;
- frontier logical operations;
- frontier logical operations / pruned token.

These are logical counters, not hardware instruction counts.

`BTreeSet` internal comparisons are not represented by those counters and must
not be silently interpreted as physical cost.

## E5b-E0 correctness gates

Local correctness requires:

1. alpha 1.5 and alpha 2.0;
2. dense probability/tau parity;
3. no dense-support false negative;
4. deterministic priority traversal;
5. every node bound evaluated at most once;
6. safe dense fallback;
7. exhaustive small hierarchies;
8. strict fmt / Clippy / workspace tests.

The historical E5 API must remain unchanged.

## Natural replay

After E5b-E0 correctness passes, replay the already-frozen E4 Q/K trace.

First compare at:

- page size `16`;
- leaf divisor `8`;
- alpha `{1.5, 2.0}`.

Then, only if useful, reuse the full E5 18,432-case matrix.

Compare against the preserved E5 cost-frontier evidence SHA-256:

`121372a6751dda83725cd54102925541ac7c37cbdad80f3608f75f291f1b3099`

Required comparison includes:

- dense parity;
- support preservation;
- loaded-token-set match rate versus E5;
- score-avoidance delta versus E5;
- bound-evaluation delta;
- historical E5 bound requests;
- E5b frontier logical operations.

## Decision rule

E5b is positive only if it preserves exactness and materially removes the
historical rescan burden without sacrificing useful score pruning.

A reduction in the custom logical counter alone is not a speedup claim.

If ordered-frontier bookkeeping remains too expensive in later hardware
measurement, A5 remains a conditional ADA-A9 plan rather than a universal
execution policy.

## Non-claims

E5b does not establish:

- wall-clock speedup;
- GPU viability;
- KV bandwidth savings;
- cache-locality benefit;
- production floating-point certification;
- model-quality suitability of Entmax;
- novelty.
