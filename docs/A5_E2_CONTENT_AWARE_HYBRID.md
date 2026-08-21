# ADA-A5 E2 — Content-Aware Hybrid Safe Bounds

## Motivation from qualified E1 evidence

A5-E1 qualified the contiguous hierarchy on 378 synthetic flat-vs-hierarchy comparisons. It showed that contiguous subdivision can improve already coherent `page_clustered` keys, but it essentially fails to recover the critical `iid_uniform` case:

- alpha=1.5: no additional score avoidance at `/2`, `/4`, or `/8`;
- alpha=2.0: only `0.006696` additional mean score avoidance at `/8`;
- `/8` eagerly evaluates `0.385045` hierarchy bounds per token.

The E1 protocol explicitly classified this outcome as evidence that the **contiguous binary geometry is insufficient**, not as evidence that hierarchical refinement itself is invalid.

E2 therefore changes geometry before attempting lazy traversal.

## Scope

E2 is an isolated scalar CPU research candidate. It does not change A4, A5-E0, FLAT-ATTENTION, or any GPU path.

Included:

- deterministic content-aware subdivision **inside each original outer KV page**;
- original token ids preserved through a per-page permutation;
- coordinate-wise min/max boxes per node;
- enclosing Euclidean balls per node;
- hybrid node bound `min(box, ball)`;
- exact A4 subset-threshold pruning certificate;
- dense-oracle f64 conservativeness validation;
- work counters for loaded/pruned tokens and which bound component wins.

Excluded:

- cross-page token clustering;
- KV physical re-layout;
- lazy/on-demand child-bound evaluation;
- model-distribution claims;
- wall-clock/GPU benchmarking;
- production directed-rounding certification;
- novelty claims.

## Content-aware partition

For every node inside an outer page:

1. compute the arithmetic mean of the node keys;
2. choose a first pivot farthest from that mean;
3. choose a second pivot farthest from the first pivot;
4. use the pivot-to-pivot vector as a deterministic approximate-diameter direction;
5. sort the node token ids by projection on that direction, tie-breaking by original token id;
6. split at the median and recurse until `leaf_size`.

This is a laboratory geometry probe, not a claim that approximate-diameter median trees are optimal or novel. Tree-based exact maximum-inner-product search and ball-tree bounds are prior art.

## Coordinate-box bound

For node `v` with coordinate extrema `k_min` and `k_max`:

\[
u_{\mathrm{box},v}(q)
=
\mathrm{scale}\sum_j
\max(q_j k_{\min,j}, q_j k_{\max,j}).
\]

As in A4-E1/A5-E0, this is a deterministic real-arithmetic upper bound on every score represented by the node.

## Enclosing-ball bound

Let `mu_v` be the node key mean and

\[
R_v = \max_{k\in v}\lVert k-\mu_v\rVert_2.
\]

For every key in the node, Cauchy-Schwarz gives

\[
q^\top k
=
q^\top\mu_v + q^\top(k-\mu_v)
\le
q^\top\mu_v + \lVert q\rVert_2 R_v.
\]

Therefore

\[
u_{\mathrm{ball},v}(q)
=
\mathrm{scale}
\left(q^\top\mu_v + \lVert q\rVert_2 R_v\right)
\]

is also a real-arithmetic MIPS upper bound.

The E2 f64 implementation adds a small numerical guard to the ball expression and then validates both component bounds against dense QK scores before allowing any pruning. This validation is an oracle-side research safeguard, **not** a production floating-point proof.

## Hybrid bound

If `u_box` and `u_ball` are both valid upper bounds, then

\[
\boxed{
 u_{\mathrm{hybrid},v}(q)
 =
 \min(u_{\mathrm{box},v}(q),u_{\mathrm{ball},v}(q))
}
\]

is also a valid upper bound and cannot be looser than either component.

E2 records how often the ball or box component supplies the smaller bound.

## Entmax certificate remains unchanged

No semantic change is made to A4. Given the conservative loaded-subset threshold lower endpoint `tau_lower`, E2 prunes a node only if

\[
\boxed{
(\alpha-1)u_{\mathrm{hybrid},v}(q)
\le
\tau_{C,\mathrm{lower}}
}
\]

which certifies that the entire node is outside full Entmax support.

Loose or unhelpful bounds only force further descent/loading; they do not authorize approximate support removal.

## E2-E0 correctness gates

The first content-aware candidate declares the following local gates:

1. index construction is deterministic and its permutation contains every original token exactly once;
2. box, ball, and hybrid f64 bounds dominate dense node scores in tested fixtures;
3. `hybrid <= box` and `hybrid <= ball`;
4. a cross-coordinate fixture demonstrates a ball bound tighter than its coordinate box;
5. alpha=1.5 and alpha=2.0 candidate outputs match the dense Entmax oracle;
6. an interleaved-cluster adversarial fixture demonstrates content-aware grouping can load fewer tokens than the qualified contiguous hierarchy at equal leaf size;
7. identical-key adversarial data safely degrades to loading all tokens;
8. exhaustive small content-aware trees preserve dense-oracle parity/support at multiple leaf sizes;
9. workspace fmt, strict Clippy, unit tests, and doc tests are green.

Passing these gates qualifies only **E2 content-aware/hybrid scalar correctness**. It does not yet establish that the new geometry fixes `iid_uniform` on the 126-case survey family.

## Next quantitative gate

Only after E2-E0 correctness is green should a survey compare, on the frozen A4/A5 fixture family:

- flat page box;
- contiguous hierarchy;
- content-aware box-only behavior where useful for attribution;
- content-aware hybrid box/ball behavior;
- loaded tokens;
- node/bound evaluations;
- box-vs-ball wins;
- probability/tau parity.

The key falsifiable target remains `iid_uniform`. If content-aware hybrid geometry materially improves that regime, then lazy/on-demand child-bound evaluation becomes justified. If it does not, deeper traversal alone should not be promoted and the negative geometric result should be preserved.
