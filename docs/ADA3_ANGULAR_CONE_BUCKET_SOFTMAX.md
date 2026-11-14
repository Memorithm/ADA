# ADA-A3: angular-cone norm-band bucket certified softmax

Research reference only. Q/K-level mechanism and falsification evidence; not
a usefulness, novelty, hardware, or FLAT-adoption claim. No prior-art review
was done (cone/ball-tree style MIPS bounds are classical).

## Conjecture

#71 norm order is blind to direction: at equal norms nothing is skipped.
Group keys into buckets = (norm band within ratio `rho`) x (greedy angular
leader with cosine `>= tau`). Per bucket store count, `nmax`, `nmin`, a unit
centroid `c` and `cos_theta` = min cosine of members to `c`. With
`phi = angle(q, c)`, spherical triangle inequality gives member angle to `q`
`>= phi - theta`, so `s<q,k> <= s|q| n g`, `g = cos(phi-theta)` if
`phi > theta` else 1 (`n = nmax` if `g >= 0`, else `nmin`; zero-norm keys have
score 0 and are covered). Bounds are widened by `1e-9` in cosine and `1e-12`
relative. Buckets are read by non-increasing bound; stop when
`2T/(S+T) <= eps`, `T = sum_unread n_b exp(U_b - m)`. Cost: one centroid dot
per bucket (counted) plus the key rows read. Correctness is independent of
how buckets were built (tested with adversarial `rho`, `tau`).

## Falsification

- L1 vs independent dense stable softmax exceeds certificate + 1e-12.
- No designed witness where it reads fewer rows than #71.
- Net query-time dots (centroid dots + rows read) not below `n` in any
  natural regime (would make the scheme a no-op).

## Results (native)

- Fail-closed on invalid layout/values/params (`norm_ratio < 1`,
  `cos_threshold` outside `[-1,1]`). Singleton, equal keys, zero query, zero
  keys, overflow: exact / finite.
- Witness (2 equal-norm-4 clusters of 32, one aligned with q, one opposite,
  d=4, eps=1e-3): cone reads **32** rows + **2** centroid dots (34 < 64);
  #71 norm order reads **64**.
- Exhaustive grid (d=2, len 1..4, 5-key alphabet incl. zero key, 3 budgets,
  4 bucket parameter settings): **9360** cases, **0** violations, skipping in
  **2909**. Random sweep (adversarial params incl. rho=1e6, tau=-1): **360**
  cases, **0** violations.
- d=16, n=256 (8 clusters x 32), 40 draws, rho=1.05, tau=0.7; fraction of n.
  "total" = rows + centroid dots; norm-order = #71 rows:

| regime | eps | cone rows | cone total | norm-order | max L1 | max cert |
|---|---|---|---|---|---|---|
| iid Gaussian | 1e-4/1e-2/1e-1 | .644/.225/.077 | **1.633/1.216/1.069** | 1.0/.9995/.9959 | <= eps | <= eps |
| clustered j=0.05 | 1e-4/1e-2/1e-1 | .950/.792/.681 | .981/.824/.713 | 1.0 | 3.2e-6/2.4e-4/2.8e-3 | 9.8e-5/9.1e-3/9.8e-2 |
| clustered j=0.2 | 1e-4/1e-2/1e-1 | .933/.820/.744 | 1.236/1.133/1.047 | 1.0 | <= eps | <= eps |
| clustered j=0.5 | 1e-4/1e-2/1e-1 | .721/.406/.242 | 1.568/1.248/1.080 | 1.0 | <= eps | <= eps |

## Verdict

**ADAPT as a research reference, NEGATIVE on net savings in most regimes.**
The certificate holds everywhere and the designed equal-norm witness beats
#71 (34 vs 64 dots). But when buckets are tight enough to separate
directions (iid keys, wide-jitter clusters) the bucket count approaches `n`,
so the centroid dots cost as much as the saved rows: total dots exceed `n`
(1.07-1.63x). Net savings appear only for tight clusters (j=0.05: 0.71-0.98 of
n) and are modest. The cone adds nothing a hierarchy over buckets would not
need to fix; a multi-level cone tree (centroid pruning of whole subtrees) is the
natural next test. Logical counts only; no hardware claim.
