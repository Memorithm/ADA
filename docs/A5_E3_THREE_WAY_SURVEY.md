# ADA-A5 E3 — Flat vs Contiguous vs Content-Aware Hybrid Survey

## Purpose

A5-E1 falsified the primary hypothesis that simple contiguous binary refinement would materially recover the `iid_uniform` regime. It improved `page_clustered` but gave no gain for alpha=1.5 and only a very small gain for alpha=2.0 at the deepest tested subdivision.

A5-E2 therefore changed two mechanisms while preserving the exact A4 subset-threshold Entmax certificate:

1. **content-aware intra-page partitioning** rather than purely contiguous subdivision;
2. a **hybrid upper bound** equal to the minimum of a coordinate box bound and an enclosing-ball MIPS bound.

E3 measures whether those changes improve logical pruning on the exact same deterministic fixture family.

This is an **algorithmic work survey**, not a wall-clock benchmark.

## Frozen fixture family

E3 reuses the A4-E2/A5-E1 family unchanged:

- seven `(N, D, page_size)` shapes;
- `iid_uniform`, `page_clustered`, and `dominant_page` regimes;
- alpha = 1.5 and alpha = 2.0;
- the same three deterministic seeds;
- score scale `1/sqrt(D)`;
- hierarchy `leaf_divisor` values 2, 4, and 8.

That gives 126 base fixtures and 378 three-way comparison cases.

## Compared candidates

For every case E3 evaluates:

1. **Flat** — A4 coordinate-wise page boxes;
2. **Contiguous** — A5-E0 contiguous binary hierarchy using coordinate boxes;
3. **Content-aware hybrid** — A5-E2 content-aware intra-page hierarchy using
   `min(box_upper, ball_upper)`.

All three retain the exact A4 pruning rule:

\[
(\alpha-1)u \le \tau_{C,\mathrm{lower}}.
\]

## Correctness gate

For every comparison case E3 computes the dense Entmax oracle and aborts if:

- any candidate exceeds the declared probability or tau tolerance;
- the flat candidate prunes a page containing dense support;
- either hierarchy leaves a dense-support token unloaded.

The survey therefore treats correctness as a gate rather than an aggregate metric.

## Primary metrics

Per case and aggregate E3 reports:

- flat logical score avoidance;
- contiguous logical score avoidance;
- content-aware logical score avoidance;
- content-aware gain over flat;
- content-aware gain over contiguous;
- contiguous bound evaluations per token;
- content-aware hybrid-bound evaluations per token;
- fraction of content-aware nodes where the ball bound is tighter than the box bound;
- threshold solves and expanded nodes for both hierarchy variants;
- maximum dense-oracle probability/tau error for all candidates.

The work counters are deliberately separate from score avoidance. A candidate that prunes more scores by evaluating many more metadata bounds is not automatically faster.

## Main falsifiable question

The critical E3 question is whether content-aware geometry materially improves `iid_uniform`, where both flat boxes and contiguous subdivision were weak despite sparse Entmax support.

Possible outcomes:

- **Material `iid_uniform` gain:** retain content-aware geometry and next investigate lazy/on-demand bound evaluation before any GPU work.
- **Gain only on already-clustered regimes:** exact geometric bounds remain strongly distribution-dependent; preserve the result and investigate different metadata/partition geometry rather than merely deeper trees.
- **Little or no gain anywhere:** reject this A5-E2 mapping as a useful pruning mechanism and keep it as negative evidence.

None of these outcomes establishes wall-clock speedup, production feasibility, or novelty.
