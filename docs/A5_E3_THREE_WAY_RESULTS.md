# ADA-A5 E3 — Qualified Three-Way Synthetic Survey

## Qualification record

E3 compared three exact Entmax support-pruning controllers on the frozen A4-E2 synthetic fixture family:

1. flat coordinate page boxes (`ADA-A4 E1`);
2. contiguous hierarchical coordinate boxes (`ADA-A5 E0/E1`);
3. content-aware intra-page hierarchy with hybrid coordinate-box/enclosing-ball bounds (`ADA-A5 E2`).

The qualified run used clean committed source:

- git commit: `c71473e1e9a55c4ce0147a94502f43597e31040d`;
- comparison cases: 378;
- raw evidence: `evidence/a5-hierarchical-bounds/ada-a5-e3-three-way-c71473e1e9a5-20260821T213931Z.txt`;
- SHA-256: `26c75aa53833935e1609d081d06b4457a1c76ea1d2ee7268c9a84b993c6599d2`;
- survey status: `complete`;
- survey return code: 0;
- final Git tree: clean.

All three candidates preserved dense-oracle support and stayed inside the declared probability/tau tolerances. Observed differences were at f64 roundoff scale.

This is an **algorithmic synthetic work survey**, not a wall-clock benchmark or hardware-speedup claim.

## Main result

The E3 result is deliberately mixed.

### `iid_uniform`

The content-aware hierarchy does **not** solve the critical high-dimensional unstructured case.

For alpha=1.5:

| Leaf divisor | Flat avoidance | Contiguous avoidance | Content-aware avoidance |
| ---: | ---: | ---: | ---: |
| 2 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 0.000000 | 0.000000 | 0.000000 |
| 8 | 0.000000 | 0.000000 | 0.000000 |

For alpha=2.0:

| Leaf divisor | Flat avoidance | Contiguous avoidance | Content-aware avoidance |
| ---: | ---: | ---: | ---: |
| 2 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 0.000000 | 0.000000 | 0.000000 |
| 8 | 0.000000 | 0.006696 | 0.017113 |

At divisor 8, content-aware therefore improves mean avoidance over contiguous by only `0.010417` while both perform `0.385045` eager node-bound evaluations per token.

This remains a negative result for the original generality hypothesis.

### `page_clustered`

The content-aware hierarchy is materially better on the structured regime.

For alpha=1.5:

| Leaf divisor | Flat | Contiguous | Content-aware | Content minus contiguous |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.447917 | 0.463542 | 0.519345 | +0.055804 |
| 4 | 0.447917 | 0.471726 | 0.531622 | +0.059896 |
| 8 | 0.447917 | 0.514695 | 0.561012 | +0.046317 |

For alpha=2.0:

| Leaf divisor | Flat | Contiguous | Content-aware | Content minus contiguous |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.666667 | 0.674107 | 0.757440 | +0.083333 |
| 4 | 0.666667 | 0.702753 | 0.768973 | +0.066220 |
| 8 | 0.666667 | 0.734003 | 0.789993 | +0.055990 |

The shallow divisor-2 configuration is especially important: it gains substantial pruning while keeping eager bound density at only `0.077009` evaluations/token. For alpha=2.0 it also reduces mean expanded nodes from `4.000000` contiguous to `2.761905` content-aware and threshold solves from `7.809524` to `5.380952`.

### `dominant_page`

All three methods obtain the same mean logical score avoidance (`0.897321`) across the tested alpha and hierarchy granularities. The outer page bound is already sufficient, so hierarchy refinement has no additional pruning opportunity.

## Box versus ball observation

The enclosing-ball component is frequently tighter than the coordinate box:

- about `0.988095` ball-win fraction at divisor 2 on `iid_uniform`;
- about `0.927721` at divisor 4;
- about `0.788095` at divisor 8.

Yet `iid_uniform` still gets essentially no pruning. Therefore the failure cannot be attributed only to coordinate-box looseness or only to contiguous partition order.

The evidence supports the narrower conclusion:

> For the tested high-dimensional `iid_uniform` fixture family, even substantially tighter exact group upper bounds remain too loose relative to the Entmax subset-threshold certificate to remove useful groups at the tested granularities.

This is not a theorem about all high-dimensional distributions.

## Research decision

E3 does **not** justify immediate GPU implementation or lazy-bound optimization.

The next discriminator is real model Q/K structure:

`A5-E4 — Real Q/K Trace Qualification`.

E4 must determine whether post-position-encoding Q/K vectors from actual transformer attention resemble the structured regime enough for A5 to be useful. If real traces show meaningful exact pruning, lazy evaluation and A2 K-first/V-late become justified follow-ons. If real traces behave like `iid_uniform`, the current exact group-bound family should be deprioritized rather than optimized prematurely.
