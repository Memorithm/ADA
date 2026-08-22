# ADA-A2 E1 — Natural Logical K/V Accounting Results

## Status

**E1-NATURAL-LOGICAL-KV-ACCOUNTING-QUALIFIED**

This status qualifies exact logical K/V work accounting on the declared frozen
natural Q/K slice.

It is not a physical memory-bandwidth, cache-traffic, GPU, wall-clock,
production, or novelty claim.

## Input

Model:

`Qwen/Qwen3-0.6B`

Model revision:

`c1899de289a04d12100db370d81485cdf75e47ca`

Capture:

`qwen3-0.6b-e4-wikitext2raw-val16`

Frozen Q/K trace SHA-256:

`d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`

Configuration:

- records: `768`
- alpha: `{1.5, 2.0}`
- page size: `16`
- leaf divisor: `8`
- leaf size: `2`
- comparison cases: `1,536`

Raw E1 replay SHA-256:

`a6a16d505d9b8b5a3b72b7536ab8021c9223a163d221b8a671dd9513a2d9bbe2`

## Exact decomposition

For every case E1 verifies:

\[
N =
K_{\rm pruned}
+
(K_{\rm loaded}-V_{\rm loaded})
+
V_{\rm loaded}.
\]

The exact final Entmax support is required to be a subset of the A5 K-loaded
set.

All 1,536 cases passed.

## Global result

Each alpha covers `184,320` visible token instances.

### alpha 1.5

Counts:

- K loaded: `53,722`
- K pruned: `130,598`
- exact logical V loaded: `2,833`
- total logical V skipped: `181,487`
- V skipped after K was already loaded: `50,889`

Weighted fractions:

- K load fraction: `0.291461`
- K pruning fraction: `0.708539`
- V load fraction: `0.015370`
- total logical V avoidance: `0.984630`
- additional V avoidance after K loading: `0.276090`
- V avoidance within loaded-K rows: `0.947266`

Thus A5 accounts for about `70.85%` of the decomposition through K pruning,
while A2 contributes another about `27.61%` of all visible token rows through
post-score V elimination.

Only about `1.54%` of visible V rows remain logically required.

### alpha 2.0

Counts:

- K loaded: `41,576`
- K pruned: `142,744`
- exact logical V loaded: `1,538`
- total logical V skipped: `182,782`
- V skipped after K was already loaded: `40,038`

Weighted fractions:

- K load fraction: `0.225564`
- K pruning fraction: `0.774436`
- V load fraction: `0.008344`
- total logical V avoidance: `0.991656`
- additional V avoidance after K loading: `0.217220`
- V avoidance within loaded-K rows: `0.963008`

Thus A5 accounts for about `77.44%` through K pruning and A2 contributes
another about `21.72%` of all visible rows through post-score V elimination.

Only about `0.83%` of visible V rows remain logically required.

## Layer dependence

### Layer 0

alpha 1.5:

- K pruning: `0.511491`
- V load: `0.021794`
- additional V avoidance after K: `0.466715`
- V avoidance within loaded K: `0.955387`

alpha 2.0:

- K pruning: `0.605827`
- V load: `0.009505`
- additional V avoidance after K: `0.384668`
- V avoidance within loaded K: `0.975886`

### Layer 13

alpha 1.5:

- K pruning: `0.656283`
- V load: `0.015951`
- additional V avoidance after K: `0.327767`
- V avoidance within loaded K: `0.953594`

alpha 2.0:

- K pruning: `0.745410`
- V load: `0.008480`
- additional V avoidance after K: `0.246110`
- V avoidance within loaded K: `0.966692`

### Layer 27

alpha 1.5:

- K pruning: `0.957845`
- V load: `0.008366`
- additional V avoidance after K: `0.033789`
- V avoidance within loaded K: `0.801544`

alpha 2.0:

- K pruning: `0.972070`
- V load: `0.007048`
- additional V avoidance after K: `0.020882`
- V avoidance within loaded K: `0.747669`

Late-layer absolute A2 opportunity is smaller because A5 has already removed
most tokens before exact K scoring.

## Context-position dependence

At query position 63:

alpha 1.5:

- K pruning: `0.496257`
- V load: `0.055583`
- V avoidance within loaded K: `0.889661`

alpha 2.0:

- K pruning: `0.576009`
- V load: `0.031982`
- V avoidance within loaded K: `0.924568`

At query position 511:

alpha 1.5:

- K pruning: `0.770935`
- V load: `0.007294`
- V avoidance within loaded K: `0.968159`

alpha 2.0:

- K pruning: `0.828451`
- V load: `0.003764`
- V avoidance within loaded K: `0.978060`

On this slice, longer contexts therefore make the final support fraction even
smaller while also improving A5 K pruning.

## Numerical exactness

Maximum dense-vs-candidate differences:

alpha 1.5:

- probability: `6.106e-16`
- tau: `4.441e-16`

alpha 2.0:

- probability: `0`
- tau: `0`

No support-mask mismatch or decomposition failure occurred.

## Interpretation

E1 answers the algorithmic question motivating A2 positively.

A5 K pruning alone does not exhaust the available sparsity benefit.

A large majority of the K rows that still require an exact score nevertheless
end outside the final exact Entmax support.

Therefore a true K-first / support-resolve / V-late execution schedule has a
large logical V-elimination opportunity on this frozen slice.

## Critical limitation

The E1 trace contains Q/K but not physical V traffic.

Qwen3 grouped-query attention also allows multiple query heads to share a KV
head.

Consequently the reported row counts must not be interpreted directly as:

- unique DRAM V rows;
- bytes transferred;
- cache-line transactions;
- HBM/DRAM bandwidth saved;
- GPU load instructions avoided;
- wall-clock speedup.

A physical implementation may reuse data across heads and queries, and sparse
V gathers may have non-trivial transaction and scheduling costs.

## Decision

Qualify A2 E1 as:

**E1-NATURAL-LOGICAL-KV-ACCOUNTING-QUALIFIED**

The next A2 stage should move from logical row opportunity to an explicit
physical-access experiment.

That experiment must measure actual V access/materialization behavior and must
retain dense-output correctness.
