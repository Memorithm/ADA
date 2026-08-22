# ADA-A2 E1 — Natural Logical K/V Accounting

## Purpose

A2-E1 measures the ideal logical work decomposition enabled by the exact A5
priority-bound controller followed by exact V-late support materialization.

The experiment replays the already-frozen A5 E4 natural Q/K trace.

It does not contain or reconstruct V tensors.

Therefore E1 qualifies logical row accounting only.

## Qualified upstream configuration

A2-E1 uses the A5 E5b qualified natural configuration:

- page size: `16`
- leaf divisor: `8`
- leaf size: `2`
- alpha: `{1.5, 2.0}`.

A5 E5b already showed exact loaded-token and final-distribution identity with
the historical exact controller on all 1,536 corresponding cases.

## Exact decomposition

For every query record:

\[
N =
K_{\rm pruned}
+
(K_{\rm loaded}-V_{\rm loaded})
+
V_{\rm loaded}.
\]

Definitions:

- `N`: all visible KV tokens;
- `K_pruned`: tokens certified away by the A5 hierarchy before exact score
  loading;
- `K_loaded`: tokens for which the candidate loads an exact K score;
- `V_loaded`: tokens with strictly positive probability in the exact final
  Entmax distribution;
- `K_loaded - V_loaded`: tokens whose K score was required but whose V row can
  still be skipped after exact support resolution.

The final support must be a subset of `K_loaded`.

## Reported fractions

E1 reports at least:

- weighted K pruning fraction;
- weighted K load fraction;
- weighted final support / logical V-load fraction;
- weighted total logical V avoidance;
- weighted additional V avoidance after K loading;
- weighted fraction of loaded-K rows whose V row is unnecessary.

It also carries A5 controller diagnostics:

- bound evaluations;
- frontier insertions;
- frontier minimum checks;
- frontier maximum pops;
- threshold solves.

## Exactness

For every record and alpha:

1. compute the dense Q/K score vector as research oracle;
2. solve dense Entmax;
3. run the exact A5 E5b priority controller;
4. require probability/tau parity within the existing natural-trace tolerance;
5. require the exact zero/nonzero support mask to match;
6. require every positive-support token to belong to the K-loaded set;
7. require the decomposition identity exactly.

## Frozen trace

The expected trace SHA-256 is:

`d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`

Model:

`Qwen/Qwen3-0.6B`

Model revision:

`c1899de289a04d12100db370d81485cdf75e47ca`

Capture:

`qwen3-0.6b-e4-wikitext2raw-val16`

The corpus contains 768 Q/K records.

With two alpha values the E1 replay contains 1,536 comparison cases.

## Aggregation

Report:

- global;
- layer;
- query head;
- query position.

Weighted ratios use raw token counts rather than the arithmetic mean of
per-case fractions.

This matters because records at larger query positions contain more visible
tokens.

## GQA limitation

Qwen3-0.6B uses grouped-query attention.

Several query heads share a KV head.

Per-query-head logical V rows therefore cannot be summed and interpreted as
physical unique V-memory fetches.

A future physical implementation may reuse V across heads, queries, warps,
blocks, cache levels, or kernel phases.

## Non-claims

A2-E1 does not establish:

- physical V bytes saved;
- memory transactions saved;
- cache traffic saved;
- wall-clock speedup;
- GPU performance;
- sparse-gather efficiency;
- production scheduling;
- model-quality suitability;
- novelty.

## Decision

E1 is positive if:

1. all exactness and decomposition invariants pass;
2. the final V-load fraction is materially below the already reduced K-load
   fraction on the frozen natural slice.

A positive result motivates a later physical K-first/V-late realization.

A weak difference between K-loaded and V-loaded would deprioritize A2 even if
the Entmax support itself is sparse.
