# ADA-A4 — Exact Entmax Branch-and-Bound

## Mission

ADA-A4 investigates whether exact alpha-entmax support can be recovered while loading only a subset of score/KV pages.

This is an algorithm-discovery track. It does not claim novelty, production readiness, or GPU speedup.

## Prior-art boundary

For alpha > 1, alpha-entmax is

`p_i = [ (alpha - 1) s_i - tau* ]_+^(1/(alpha - 1))`,

with the unique threshold `tau*` chosen so that the probabilities sum to one. Therefore

`support(p) = { i : (alpha - 1) s_i > tau* }`.

EntmaxKV (arXiv:2605.21649) establishes two ingredients used here:

1. sparse entmax attention is exact if the selected candidates contain the full entmax support;
2. coordinate-wise min/max key metadata yields a deterministic page score upper bound `u_p >= max_{i in page p} s_i`, and any conservative threshold estimate `tau_hat <= tau*` can be combined with that upper bound to select a support superset without false negatives.

AdaSplash-2 (arXiv:2604.15180) independently shows how conservative lower bounds on the entmax threshold can be useful for exact normalization, including a histogram lower bound.

ADA-A4 does **not** claim these ingredients as new.

## A4-E0 candidate: subset-threshold branch-and-bound

Let `C` be the set of token scores already loaded exactly. Define

`f_C(tau) = -1 + sum_{i in C} [ (alpha - 1) s_i - tau ]_+^(1/(alpha - 1))`.

Let `tau_C` be its unique root. Let `tau_full` be the root over all tokens.

### Lemma 1 — subset threshold is a lower bound

For every `tau`,

`f_full(tau) >= f_C(tau)`

because the omitted terms are non-negative. Both objectives are monotonically decreasing. Since `f_C(tau_C)=0`, we have `f_full(tau_C) >= 0`, hence

`tau_C <= tau_full`.

Moreover, if `C` grows, its threshold cannot decrease.

This gives a conservative threshold estimate using only exact scores already loaded.

### Lemma 2 — safe page elimination

Let `u_p` be a valid upper bound on every score in an unloaded page `p`.

If

`(alpha - 1) u_p <= tau_C`,

then for every token `i` in page `p`,

`(alpha - 1) s_i <= (alpha - 1) u_p <= tau_C <= tau_full`.

Therefore no token in page `p` can belong to the full entmax support.

### Theorem — exact termination

Suppose the algorithm stops when every unloaded page satisfies

`(alpha - 1) u_p <= tau_C`.

Every omitted token contributes zero at `tau_C`. Therefore

`f_full(tau_C) = f_C(tau_C) = 0`.

By uniqueness of the entmax root,

`tau_C = tau_full`,

and entmax computed only over the loaded scores is exactly equal, in real arithmetic, to full dense entmax.

The method therefore has a dense fallback: loose page bounds can force all pages to be loaded, but cannot change the mathematical result.

## E0 algorithm

1. Partition scores into pages and obtain a conservative upper bound `u_p` for each page.
2. Load one seed page (E0 chooses the page with largest `u_p`).
3. Solve entmax on the loaded subset and retain a conservative lower endpoint of the threshold bracket.
4. Permanently prune any unloaded page satisfying `(alpha - 1) u_p <= tau_lower`.
5. If unresolved pages remain, load the unresolved page with largest `u_p` and repeat.
6. When no unresolved page remains, compute the final entmax distribution on the loaded subset and emit zeros for pruned pages.

Using the lower endpoint of a valid numerical root bracket, rather than an unconstrained midpoint, keeps the pruning decision conservative. This numerical convention is an engineering safeguard; it is not a formal floating-point proof.

## E0 scope

Included:

- scalar deterministic f64 dense entmax oracle;
- alpha in `(1, 2]`;
- externally supplied conservative page score bounds;
- one-page-at-a-time deterministic branch-and-bound;
- exact-support/parity tests against the dense oracle;
- logical counters for pages/scores loaded and threshold solves;
- adversarial and exhaustive small-state tests;
- dense fallback when bounds are loose.

Excluded:

- Q/K vector computation and coordinate-box metadata construction;
- V loading or output accumulation;
- GPU kernels;
- histogram acceleration;
- approximate/Gaussian page selection;
- A2 K-first/V-late scheduling;
- hardware performance claims.

## Promotion gates

A4-E0 may advance only if all of the following hold:

1. subset-threshold monotonicity tests pass;
2. no pruned page contains a dense-oracle support token in adversarial fixtures;
3. final probabilities and threshold match the dense oracle within the declared f64 tolerance;
4. exhaustive small-state tests pass for alpha = 1.5 and alpha = 2.0;
5. loose-bound cases provably fall back to loading all pages without changing the result;
6. fmt, strict clippy, and workspace tests are green.

Only after E0 correctness should ADA consider page-bound construction from real Q/K vectors, batched expansion policies, value-traffic accounting, or GPU realization.

## Industrial audit notes (2026-08-24)

A source-hardening pass touched this crate without changing any qualified
algorithmic behavior on admissible inputs:

- `probabilities_at_tau` now fails closed with a typed error when a powered
  term is non-finite, instead of publishing infinities. On admissible inputs
  (finite scores, alpha in `(1, 2]`, converged bracket) the emitted
  distribution was already finite; the guard closes a theoretical hole.
- `branch_and_bound_entmax` finalizes from the terminating round's bracket
  instead of re-solving it through the dense oracle. `metrics.threshold_solves`
  now equals exactly the number of bracket solves performed; historical runs
  undercounted that counter by one because the terminating solve was counted
  twice in execution but once in metrics. No preserved A4-E2 survey artifact
  consumed the E0 candidate's counter, so no recorded evidence changes.
- Miri's software-float `powf` does not preserve the IEEE identity
  `powf(x, 1.0) == x`, so the native rounding-repair test is ignored under
  Miri only (`#[cfg_attr(miri, ignore)]`). It remains strict on real
  toolchains.
- Former domain limit, now closed (2026-08-25): when
  `ulp((alpha-1) * max(scores)) >= 0.5` the nominal `[m-1, m]` initial bracket
  collapses. The solver now takes a certified extreme-magnitude path instead:
  it returns the single representable step `[next_down(m), m]`, skips
  bisection, and consumers finalize probabilities through mass normalization,
  which is exact in real arithmetic (`p_i(tau)/sum_j p_j(tau)` is independent
  of residual threshold error). When no term is representable above zero the
  finalization falls back to the exact limit distribution: uniform over the
  ties at `(alpha-1) * max(score)`. The pruning predicate `scale * bound <=
  tau_lower` remains sound because `next_down(m)` keeps every support page out
  of the pruned set. The normal regime is untouched bit-for-bit; extreme-path
  coverage lives in the dedicated A4 tests (singleton support, ties, and
  branch-and-bound parity at `1e200` magnitudes).
- Streaming monitor (2026-08-25): `StreamingEntmax` re-solves the exact
  bracket per absorbed block and fails closed when the prefix threshold
  sequence violates monotonicity beyond a four-ulp allowance, turning the
  real-arithmetic monotonicity fact into a checked certificate.
- Miri note (2026-08-25): Miri's software-float `powf` may differ from the
  native libm by one ulp between two evaluations of the same expression, so
  tests that compare two independent solver runs now compare bit-exactly on
  native toolchains and within a small tolerance under Miri only.
