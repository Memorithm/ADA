# ADA-A4 E2 — Deterministic Q/K Box Pruning Survey

## Purpose

A4-E0 established the exact subset-threshold branch-and-bound rule at score level. A4-E1 established that coordinate-wise per-page key minima/maxima can provide conservative query-specific score upper bounds that preserve the E0 exactness contract in the tested scalar `f64` laboratory.

A4-E2 asks the next falsifiable question:

> Are those safe coordinate-box bounds tight enough to avoid a meaningful fraction of page/score loads under controlled Q/K distributions?

E2 is **not** a wall-clock benchmark and is **not** evidence about real model distributions. It is an algorithmic survey of bound tightness and certified work avoidance.

## Metrics

For page `p`, let

`u_p(q) = scale * sum_j max(q_j * k_min[p,j], q_j * k_max[p,j])`

and let

`m_p(q) = max_{i in page p} score(q, k_i)`.

The primary bound metric is

`slack_p = u_p - m_p >= 0`

in the exact arithmetic contract. The harness reports mean, p95, and maximum observed slack.

The branch-and-bound metrics are:

- `page_load_ratio = pages_loaded / pages_total`;
- `score_avoidance = 1 - scores_loaded / scores_total`;
- dense entmax support fraction;
- exact dense-vs-candidate maximum probability difference;
- exact dense-vs-candidate threshold difference.

The harness treats any support token placed in an unloaded page as a hard failure.

## Synthetic regimes

E2 deliberately uses deterministic synthetic regimes, not claims about transformer activations:

1. `iid_uniform` — query/key coordinates are independent centered bounded values. This is expected to stress coordinate boxes because each page can span a broad axis-aligned region.
2. `page_clustered` — keys within a page share a random centroid plus small noise. This isolates the effect of page-local geometric coherence on box tightness.
3. `dominant_page` — one page is strongly aligned with the query while other pages are mostly anti-aligned with small noise. This is a controlled sparse-support regime intended to demonstrate the best-case value of exact pruning.

## Survey grid

The initial deterministic grid spans:

- sequence lengths from 128 to 2048;
- head dimensions from 32 to 128;
- page sizes from 16 to 128;
- alpha in `{1.5, 2.0}`;
- three fixed RNG seeds;
- all three regimes above.

The attention score scale is `1 / sqrt(head_dim)`.

## Interpretation rules

A4-E2 may support statements such as:

- coordinate boxes are tight/loose on a named synthetic regime;
- a named fraction of pages/scores is certified unnecessary on the survey grid;
- exactness and support preservation hold for the measured cases.

It must **not** be used to claim:

- end-to-end inference speedup;
- GPU speedup;
- expected pruning on a real LLM;
- superiority over EntmaxKV, AdaSplash-2, or another implementation;
- novelty of the coordinate-box bound itself.

If E2 shows useful pruning only in tightly clustered or dominant synthetic cases, the next research question belongs naturally to A5: whether stronger/hierarchical safe metadata can tighten bounds without excessive metadata or query-time cost.

If E2 shows useful pruning broadly, the next bridge is A2: delay V traffic until A4 has certified the page set.

## Qualification gate

E2 remains a candidate until:

1. `cargo fmt --all -- --check` is green;
2. strict workspace Clippy is green;
3. all workspace tests are green;
4. the release survey completes deterministically;
5. every surveyed dense support token is contained in a loaded page;
6. dense/candidate probability and threshold differences stay within the declared `f64` tolerance;
7. the raw survey output is preserved with Git SHA and SHA-256.
