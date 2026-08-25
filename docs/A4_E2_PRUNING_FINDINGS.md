# ADA-A4 E2 — Clean Synthetic Pruning Survey Findings

## Evidence identity

- Tested commit: `198021274195ad73878f1aa4af5ec1bb411d510c`
- Target: `aarch64`
- Rust: `rustc 1.89.0 (29483883e 2025-08-04)`
- Cargo: `cargo 1.89.0 (c24e10642 2025-06-23)`
- UTC: `20260821T205630Z`
- Git tree declared clean before the survey and remained clean afterwards.
- Raw evidence: `evidence/a4-entmax-bnb/a4-e2-pruning-198021274195-20260821T205630Z.txt`
- Raw SHA-256: `f6e021e127a89afed15b33b48dd8971bd7c0feb761e406d67fdccd16ddb62a9a`
- Survey cases: 126 deterministic synthetic cases.
- This is not a wall-clock benchmark and does not claim hardware speedup.

## Aggregate results

| Regime | alpha | Mean page-load ratio | Mean score avoidance | Mean support-token fraction | Mean support-page fraction | Mean bound slack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iid_uniform | 1.5 | 1.000000 | 0.000000 | 0.184756 | 0.998512 | 3.638321794 |
| iid_uniform | 2.0 | 1.000000 | 0.000000 | 0.022879 | 0.599702 | 3.638321794 |
| page_clustered | 1.5 | 0.552083 | 0.447917 | 0.240141 | 0.293155 | 0.291853455 |
| page_clustered | 2.0 | 0.333333 | 0.666667 | 0.087333 | 0.139881 | 0.291853455 |
| dominant_page | 1.5 | 0.102679 | 0.897321 | 0.102679 | 0.102679 | 0.412566775 |
| dominant_page | 2.0 | 0.102679 | 0.897321 | 0.092029 | 0.102679 | 0.412566775 |

Maximum dense/candidate probability difference over all aggregate groups was `5.551e-17`; maximum tau difference was `5.551e-17`. No dense-support page was pruned, and the survey completed with `survey_status=complete`.

## Main finding

The E1 coordinate-box bound is exact/conservative in the tested laboratory, but its usefulness is distribution dependent.

The strongest diagnostic is `iid_uniform, alpha=2.0`: the dense entmax support contains only `2.2879%` of tokens on average, yet the coordinate boxes still force `100%` of pages to be loaded. The sparse support therefore exists, but the flat page bound is too loose to certify it before loading scores.

The same branch-and-bound controller becomes useful when page geometry is coherent:

- `page_clustered`: `44.7917%` mean logical score avoidance for alpha=1.5 and `66.6667%` for alpha=2.0;
- `dominant_page`: `89.7321%` mean logical score avoidance for both tested alpha values.

This isolates bound quality, rather than the entmax support theorem or the A4 subset-threshold controller, as the next bottleneck.

## Why the flat coordinate box can become loose

For a page box, the query bound

`u_p(q) = scale * sum_j max(q_j * k_min[p,j], q_j * k_max[p,j])`

chooses an extremum independently in every coordinate. Those extrema can originate from different key rows, so the maximizing corner need not correspond to any actual key token. The resulting gap can grow with dimension and within-page dispersion.

E2 is consistent with this mechanism: mean bound slack is approximately `3.64` for the iid regime but only `0.292` for the page-clustered regime.

## Consequence for the roadmap

E2 motivates ADA-A5: hierarchical safe pre-KV bounds. A5 should keep the same exact pruning certificate

`(alpha - 1) * upper_bound <= tau_lower`

but recursively replace one loose page box by tighter boxes over nested subsets. If a parent is certifiably outside support, the whole subtree is rejected. If it is unresolved, the algorithm descends to tighter child bounds before deciding whether any leaf/block needs score evaluation.

This is an engineering/research adaptation of established exact branch-and-bound ideas for inner-product search; no novelty claim is made. Relevant prior art includes Ram & Gray, *Maximum Inner-Product Search using Tree Data-structures* (2012), while EntmaxKV (2026) supplies the entmax-specific pre-KV support-recovery context and flat coordinate-box page bound.

## Status interpretation

E2 provides deterministic synthetic algorithmic evidence only. `score_avoidance` counts scores the certified algorithm would not need to materialize in a sparse implementation; the current laboratory still computes dense QK scores for oracle verification. No memory-traffic, latency, throughput, GPU, or production claim follows from this survey.
