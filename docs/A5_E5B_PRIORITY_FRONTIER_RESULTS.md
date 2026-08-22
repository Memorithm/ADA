# ADA-A5 E5b — Priority-Frontier Natural Replay Results

## Status

**E5B-PRIORITY-FRONTIER-FOCUSED-NATURAL-QUALIFIED**

This is algorithmic natural-trace evidence only.

It is not a wall-clock benchmark, hardware-speedup result, production
floating-point qualification, KV-bandwidth measurement, or novelty claim.

## Motivation

ADA-A5 E5 showed that exact lazy hierarchical bounds preserve strong natural
Q/K pruning, but the historical Vec-based frontier repeatedly rescanned already
known bounds.

At page size 16 / leaf divisor 8, the preserved E5 full cost-frontier evidence
reported:

### alpha 1.5

- weighted score avoidance: `0.708539`
- weighted bound avoidance: `0.167199`
- bound evaluations / pruned token: `1.101916`
- historical bound requests / pruned token: `80.490612`

### alpha 2.0

- weighted score avoidance: `0.774436`
- weighted bound avoidance: `0.208171`
- bound evaluations / pruned token: `0.958555`
- historical bound requests / pruned token: `61.499173`

The E5 full matrix identified page size 16 / leaf divisor 8 as the sole global
Pareto point under the declared score-pruning/bound-cost objectives.

## E5b mechanism

E5b keeps the exact same coordinate-box bound and Entmax pruning certificate,
but replaces repeated frontier rescans with a deterministic ordered
`BTreeSet`.

Each evaluated hierarchy-node bound is computed at most once.

The minimum frontier bound drives pruning.

The maximum frontier bound drives expansion.

The laboratory implementation retains dense Q/K only as an independent oracle.

## Frozen natural trace

Model:

`Qwen/Qwen3-0.6B`

Model revision:

`c1899de289a04d12100db370d81485cdf75e47ca`

Capture:

`qwen3-0.6b-e4-wikitext2raw-val16`

Trace SHA-256:

`d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`

Replay configuration:

- records: `768`
- alpha: `{1.5, 2.0}`
- page size: `16`
- leaf divisor: `8`
- leaf size: `2`
- comparison cases: `1,536`

Raw E5b replay SHA-256:

`b7a865fe962b15f3ed63f39ec3e22a7ae85dd10b18c70f29e9b66d0cf8415e6b`

## Exactness result

Across all 1,536 cases:

- loaded-token-set match fraction versus historical E5: `1.000000`
- distribution bitwise match fraction versus historical E5: `1.000000`
- mean loaded-set symmetric difference: `0.000000`
- loaded-set mismatch cases: `0`
- non-bitwise distribution cases: `0`
- no dense-support false negative
- replay status: `complete`

The priority controller also matched the historical controller's:

- score avoidance;
- bound avoidance;
- bound-evaluation count;
- node expansions;
- threshold-solve count.

Maximum historical-vs-priority probability and tau differences were exactly
zero in the replay.

## Global scheduling result

### alpha 1.5

Historical E5:

- weighted score avoidance: `0.708539`
- weighted bound avoidance: `0.167199`
- bound evaluations / pruned token: `1.101916`
- bound requests / pruned token: `80.490612`

E5b:

- priority frontier operations / pruned token: `2.857555`
- bound evaluations + frontier operations / pruned token: `3.959471`

Relative reduction of the conservative combined logical-action counter versus
historical E5 bound requests:

approximately `95.08%`, or about `20.3x`.

### alpha 2.0

Historical E5:

- weighted score avoidance: `0.774436`
- weighted bound avoidance: `0.208171`
- bound evaluations / pruned token: `0.958555`
- bound requests / pruned token: `61.499173`

E5b:

- priority frontier operations / pruned token: `2.447865`
- bound evaluations + frontier operations / pruned token: `3.406420`

Relative reduction of the conservative combined logical-action counter versus
historical E5 bound requests:

approximately `94.46%`, or about `18.1x`.

## Layer slices

All sampled layers retained exact loaded-token and distribution identity.

### Layer 0

alpha 1.5:

- historical requests / pruned token: `170.635620`
- E5b combined logical actions / pruned token: `6.537549`

alpha 2.0:

- historical requests / pruned token: `123.008463`
- E5b combined logical actions / pruned token: `5.342781`

### Layer 13

alpha 1.5:

- historical requests / pruned token: `101.363300`
- E5b combined logical actions / pruned token: `4.864416`

alpha 2.0:

- historical requests / pruned token: `74.337810`
- E5b combined logical actions / pruned token: `4.088192`

### Layer 27

alpha 1.5:

- historical requests / pruned token: `18.051776`
- E5b combined logical actions / pruned token: `1.962736`

alpha 2.0:

- historical requests / pruned token: `13.319503`
- E5b combined logical actions / pruned token: `1.676813`

## Interpretation

E5b resolves the main logical-controller pathology exposed by E5.

The historical result was not dominated by repeated D-dimensional bound
evaluation; it was dominated by repeated requests caused by full-frontier
rescanning.

The ordered frontier preserves the exact historical natural-trace behavior at
the E5 Pareto configuration while reducing the declared logical scheduling
counter by roughly an order of magnitude or more.

This is a stronger algorithmic result than E5, but it remains insufficient for
a physical performance claim.

`BTreeSet` performs internal comparisons and tree operations that are not
represented by the custom frontier-operation counter.

Therefore the next stage must account for actual K/V work, metadata access,
frontier bookkeeping, and hardware realization.

## Decision

E5 remains preserved as:

**E5-LAZY-COST-FRONTIER-MIXED**

E5b is qualified as:

**E5B-PRIORITY-FRONTIER-FOCUSED-NATURAL-QUALIFIED**

A full 18,432-case E5b replay is not required at this stage because:

1. the preceding E5 full matrix already identified page 16 / divisor 8 as the
   sole tested global Pareto configuration; and
2. E5b reproduced historical E5 exactly on all 1,536 natural cases at that
   configuration.

The full matrix may be revisited if later hardware constraints make another
page/leaf configuration relevant.

## Next step

Proceed to the A5 → A2 interface:

**K-first / V-late candidate work accounting**

The next experiment must distinguish:

- K score work actually performed;
- K score work pruned;
- V rows never loaded because support pages/leaves were certified absent;
- hierarchy metadata traffic;
- priority-frontier bookkeeping;
- threshold-solving work.

Only after that accounting should a physical CPU/GPU realization be selected.
