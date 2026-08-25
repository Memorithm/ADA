# ADA-A5 E5 — Lazy / On-Demand Hierarchical Bound Protocol

## Purpose

ADA-A5 E4 established that exact hierarchical Q/K bounds can produce
substantial logical score avoidance on a frozen natural Qwen3 Q/K slice.

The strongest tested configuration, page size 16 with leaf divisor 8, reached
approximately:

- `0.639074` contiguous score avoidance for alpha 1.5;
- `0.712097` contiguous score avoidance for alpha 2.0;
- `0.660899` content-aware box-only score avoidance for alpha 1.5;
- `0.736760` content-aware box-only score avoidance for alpha 2.0.

However, the current laboratory controller eagerly evaluates every hierarchy
node bound before branch-and-bound traversal.

E5 asks the next algorithmic question:

> How many hierarchy-node bounds must actually be evaluated to obtain the same
> exact pruning decisions?

E5 is not a wall-clock benchmark and does not yet claim physical K/V traffic
avoidance.

## Core hypothesis

A useful hierarchy should avoid both:

1. token-score evaluation for pruned leaves/subtrees;
2. bound evaluation for hierarchy nodes that are never needed by traversal.

For a hierarchy containing `N_nodes` nodes, define:

`bound_evaluation_fraction = evaluated_bounds / N_nodes`

and:

`score_load_fraction = loaded_tokens / key_count`.

The corresponding avoidance fractions are:

`bound_avoidance = 1 - bound_evaluation_fraction`

and:

`score_avoidance = 1 - score_load_fraction`.

E5 succeeds only if exact score pruning is preserved while a material fraction
of node bounds remains unevaluated.

## Historical eager controller

The existing A5 controller remains unchanged and is the reference behavior.

Its query-time sequence is conceptually:

1. compute every hierarchy-node upper bound;
2. validate every bound against the dense Q/K oracle;
3. select the highest-bound seed path;
4. solve the subset Entmax threshold;
5. prune resolved nodes;
6. expand the highest unresolved node;
7. load unresolved leaves;
8. repeat until all unloaded tokens are certified outside support.

The historical API and metrics must remain reproducible.

## Lazy bound state

The E5 controller maintains one state per hierarchy node:

- `Unknown`
- `Known(upper_bound)`

A node bound is evaluated only on first demand.

Repeated requests for the same node must reuse the cached value and must not
increment the bound-evaluation counter.

No approximation or speculative pruning is permitted.

## When a bound may be demanded

A bound is required only when one of the following operations needs it:

1. comparing candidate root nodes to choose the seed root;
2. comparing sibling children while descending the initial seed path;
3. testing a frontier node against the current Entmax pruning certificate;
4. selecting the highest-bound unresolved frontier node;
5. comparing or ordering newly exposed children when required by traversal.

A child whose parent is pruned must never have its bound evaluated.

A descendant of a pruned subtree must therefore remain `Unknown`.

## Exactness invariant

Lazy evaluation changes only *when* a safe bound is computed.

For every evaluated node `v`, the bound remains the same coordinate-box bound:

\[
u_v(q)
=
c
\sum_j
\max(q_j k^{v,j}_{min}, q_j k^{v,j}_{max}).
\]

The pruning certificate remains:

\[
(\alpha - 1)u_v(q)
\le
\tau_C^{lower}.
\]

Therefore E5 must produce the same loaded-token set and final Entmax
distribution as the historical eager controller for the same hierarchy and
case.

## Oracle-side validation

E5 remains a scalar research implementation.

Dense Q/K scores may still be computed once for:

- independent dense Entmax parity;
- support-preservation checks;
- validating every bound at the moment that bound is first evaluated.

Crucially, dense oracle construction is not counted as E5 candidate work.

E5 must not pre-validate unknown node bounds, because doing so would destroy the
measurement of lazy bound demand.

A production implementation will require a separate numerically certified
bound path and must not rely on dense-score validation.

## Initial scope

E5 first targets the contiguous A5 hierarchy only.

Reasons:

- E4 showed contiguous hierarchy already obtains strong natural-Q/K pruning;
- it provides the simplest causal baseline for measuring lazy traversal;
- content-aware partitioning introduces a separate offline/index construction
  question;
- the current enclosing-ball component is already deprioritized by E4 because
  it changed pruning in zero of 18,432 replay cases.

Content-aware box-only lazy traversal may be added after the contiguous
mechanism is qualified.

## Required metrics

The lazy result must report at least:

- `nodes_total`;
- `bound_evaluations`;
- `bound_cache_hits`;
- `nodes_never_evaluated`;
- `nodes_expanded`;
- `subtrees_pruned`;
- `leaves_total`;
- `leaves_loaded`;
- `tokens_loaded`;
- `tokens_pruned`;
- `threshold_solves`.

Derived metrics:

- `bound_evaluation_fraction`;
- `bound_avoidance`;
- `score_load_fraction`;
- `score_avoidance`;
- `bound_evaluations_per_loaded_token`;
- `bound_evaluations_per_pruned_token`.

The historical eager metrics remain available separately.

## Correctness gates

For every tested case, lazy and eager contiguous controllers must have:

- identical loaded-token bitsets;
- identical token counts;
- identical subtree/token pruning totals where traversal semantics imply it;
- dense support preservation;
- probability parity within the existing E4 tolerance;
- tau parity within the existing E4 tolerance.

The strongest correctness target is exact equality of eager/lazy loaded-token
sets and final `EntmaxDistribution` whenever both execute the same traversal
ordering.

If implementation details alter traversal ordering while preserving exactness,
that difference must be documented rather than silently normalized.

## Adversarial tests

E5 local tests must include:

1. a hierarchy where the seed path requires evaluating only a subset of nodes;
2. an early-pruned subtree whose descendants remain unevaluated;
3. loose bounds that force dense fallback;
4. alpha 1.5;
5. alpha 2.0;
6. exhaustive small hierarchies;
7. repeated access proving cache hits do not re-evaluate bounds.

## Natural replay

After local correctness qualification, E5 reuses the already frozen E4 trace:

- trace SHA-256:
  `d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`;
- Qwen3 records: `768`;
- natural replay corpus: frozen WikiText-2 validation slice.

The initial E5 natural replay should first focus on the E4-best contiguous
configuration:

- page size `16`;
- leaf divisor `8`;
- alpha `{1.5, 2.0}`.

Only after interpreting that focused result should the full page/divisor matrix
be considered.

## E5 decision rule

### Strong positive result

Preserve lazy A5 and proceed toward A2 K-first / V-late if:

- exact E4 score pruning is retained;
- a substantial fraction of node bounds remains unevaluated;
- the remaining bound-evaluation count is plausibly smaller than the token work
  it enables the controller to avoid.

### Mixed result

If bound demand is strongly layer/head/position dependent, preserve E5 as a
conditional mechanism and expose its observable state to ADA-A9 / ElasticXxx.

### Negative result

If nearly every hierarchy-node bound must still be evaluated to obtain the E4
score pruning, reject the current lazy traversal strategy even if logical score
avoidance remains high.

That outcome would mean the hierarchy moves work from dense Q/K scores into
metadata-bound evaluation without enough algorithmic reduction.

## Non-claims

E5 does not by itself establish:

- latency or throughput speedup;
- GPU viability;
- physical K/V bandwidth reduction;
- metadata cache locality;
- production floating-point certification;
- model-quality suitability of Entmax;
- novelty.
