# ADA-A3: multi-level angular cone tree certified softmax

Research reference only. Q/K-level mechanism and falsification evidence; not
a usefulness, novelty, hardware, or FLAT-adoption claim. No prior-art review
was done (ball/cone-tree MIPS bounds are classical).

## Conjecture

The flat cone buckets of the angular-cone bucket module pay one centroid dot per
bucket, so fine buckets cost about `n` dots. A binary cone tree prunes a whole
subtree with the single dot of its root. Node = count, `nmax`, `nmin`, unit
centroid `c`, `cos_theta` (min member cosine to `c`); bound
`s<q,k> <= s|q| n g`, `g = cos(phi-theta)` if `phi > theta` else 1 (`nmax` if
`g >= 0` else `nmin`), widened by `1e-9` (cosine) and `1e-12` (relative).
Query: best-first on the frontier, expanding the largest bound (children cost
one centroid dot each, leaves cost their rows); stop when
`2T/(S+T) <= eps`, `T = sum_frontier n_b exp(U_b - m)`. Zero-norm keys form a
separate leaf (score 0). Correctness is independent of the tree shape.

## Falsification

- L1 vs independent dense stable softmax exceeds certificate + 1e-12.
- No designed witness where subtree pruning beats `n` total dots.
- Net query-time dots (rows + centroid dots) not below the flat buckets / `n`
  in any natural regime.

## Results (native)

- Fail-closed on invalid layout/values/params (`leaf_size = 0`, non-finite,
  `score_scale <= 0`, `eps < 0`). Singleton, equal keys, zero query, zero
  keys, overflow: exact / finite; `eps = 0` reads everything, certificate 0.
- Independent oracle: dense scores + stable softmax.
- Witness (4 equal-norm clusters of 32, d=4, scale 4, eps=1e-3, leaf 4):
  total dots (rows + centroid) below 64 of 128 (test asserts it).
- Exhaustive grid (d=2, len 1..5, 5-key alphabet incl. zero key, 3 queries incl.
  zero, eps in {0,1e-3,0.3}, leaf in {1,2}): **70290** cases, **0** violations,
  skipping in **13874**. Random adversarial sweep (heavy-tailed norms, d<=8,
  n<=60): **400** cases, **0** violations.
- d=16, n=256 (8 clusters x 32 or iid), 40 draws, scale 1/sqrt(d); fraction of
  n; "total" = rows + centroid dots (`regime_report`, ignored test):

| regime | eps | leaf | rows | total | max L1 | max cert |
|---|---|---|---|---|---|---|
| iid | 1e-4 / 1e-2 / 1e-1 | 4 | 1.000/.988/.950 | 1.664/1.652/1.613 | 4.9e-5/9.7e-3/7.7e-2 | <= eps |
| iid | same | 16 | 1.000/1.000/1.000 | 1.150 | ~0 | 0 |
| j=0.05 | 1e-4 / 1e-2 / 1e-1 | 4 | 1.000/.971/.807 | 1.710/1.678/1.430 | <= eps | <= eps |
| j=0.05 | same | 16 | 1.000/.982/.830 | 1.189/1.171/1.001 | <= eps | <= eps |
| j=0.2 | same | 4 | 1.000/.975/.854 | 1.710/1.685/1.520 | <= eps | <= eps |
| j=0.5 | same | 4 | 1.000/.979/.892 | 1.703/1.683/1.583 | <= eps | <= eps |
| j=0.5 | same | 16 | 1.000/.996/.956 | 1.201/1.198/1.156 | <= eps | <= eps |

## Verdict

**REJECT as a net-saving mechanism in natural regimes; keep as a valid
certified reference.** The certificate holds everywhere (0 violations in
70 690 adversarial cases) and the designed witness (well-separated tight
clusters, sharp scale) saves dots. But in d=16 even 32-key clusters have
node cones too wide to separate maxima, so the tree walks nearly to the
leaves: rows read 0.81-1.0 of `n` and total dots 1.0-1.7x `n`, i.e. **worse
than the flat buckets** (0.71-1.63) because internal nodes add about `2 n /
leaf` dots. A smaller leaf prunes a bit more but adds more internal dots.
Cause: a centroid/cone bound is a single-direction bound, loose in moderate
dimension, and iid/jittered synthetic data has no direction structure at
the score scale used. Real Q/K traces are still untested; logical counts
only, no hardware claim.

## Next

1. Real Q/K traces (the cone bound may only pay under real anisotropy).
2. Thresholded / sorted-tail softmax on exact scores.
3. Tree cut by gain: stop splitting where the cone does not narrow.
