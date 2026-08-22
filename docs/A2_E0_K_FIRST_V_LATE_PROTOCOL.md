# ADA-A2 E0 — Exact K-first / V-late Contract

## Purpose

ADA-A2 studies whether sparse exact attention can stage K before V and avoid
materializing V rows that provably have zero final attention weight.

A2 does not modify the Entmax semantics qualified by A4/A5.

The initial bridge is:

\[
\text{A5 priority-bound K discovery}
\rightarrow
\text{exact Entmax distribution}
\rightarrow
\text{V-late support materialization}.
\]

## Set relationship

Three token sets must remain conceptually distinct.

1. all visible KV tokens;
2. tokens whose exact K score is loaded by A5;
3. tokens in the final exact Entmax support.

A5 may have to load the K score of a token that ultimately receives zero
probability.

A2-E0 therefore does not define V loading as "all loaded K tokens".

It defines V loading from the final exact support:

\[
\mathcal V_{\mathrm{load}}
=
\{i : p_i > 0\}.
\]

Consequently,

\[
|\mathcal V_{\mathrm{load}}|
\le
|\mathcal K_{\mathrm{load}}|
\le
N.
\]

## Dense eager-V oracle

The E0 dense oracle evaluates

\[
O=\sum_i p_i V_i
\]

while logically reading every V row, including rows for which `p_i = 0`.

This is a semantic reference and eager-load logical baseline.

It is not a production implementation.

## V-late candidate

The candidate first obtains the exact final distribution.

For every token:

- if `p_i == 0`, its V row is skipped before any scalar from that row is
  inspected;
- if `p_i > 0`, its V row is loaded and accumulated.

Thus the real-arithmetic output remains

\[
O=\sum_{i:p_i>0}p_iV_i.
\]

Because omitted terms have exactly zero coefficient, this is mathematically
identical to the dense sum.

Binary floating-point equality is not assumed as a general theorem; E0 checks
the outputs numerically against the dense oracle.

## Logical V metrics

E0 records:

- V rows total;
- V rows loaded;
- V rows skipped;
- V scalars loaded;
- V scalars skipped;
- V row load fraction;
- V row avoidance;
- V scalar load fraction;
- V scalar avoidance.

For a fixed `value_dim`, row and scalar fractions are numerically equal, but
both counters are retained because later hardware realizations may have
different transaction granularities.

## Structural skipped-row test

E0 includes a laboratory access test in which V rows with zero probability
contain non-finite sentinels.

The V-late candidate must succeed because those rows are never inspected.

A non-finite value in a positive-probability row must fail validation.

This establishes source-level logical access behavior only.

It does not establish what a compiler, cache, prefetcher, vector load, GPU
memory transaction, or physical memory subsystem will do.

## A5 bridge

The E0 integrated candidate invokes the exact A5 E5b priority-frontier
controller.

The resulting exact Entmax distribution then drives V-late materialization.

The A5 implementation still constructs dense Q/K scores internally for
laboratory oracle validation.

Those dense oracle accesses are not candidate work.

Therefore E0 is algorithmic correctness and work-accounting evidence only.

## Correctness gates

A2-E0 requires:

1. dense eager-V versus V-late output parity;
2. alpha 1.5 and alpha 2.0 integrated A5 → A2 cases;
3. final Entmax distribution equality with dense Entmax;
4. `V_loaded == exact support size`;
5. `K_loaded >= V_loaded`;
6. zero-probability V rows are not inspected;
7. non-finite loaded V rows are rejected;
8. dense-support fallback loads every V row;
9. exhaustive small support-mask weighted-sum checks;
10. workspace fmt, strict Clippy, unit/doc tests.

## Natural Q/K follow-on

The frozen E4 Q/K trace contains no V tensors.

It can nevertheless support the next logical-accounting experiment because the
exact support size is known after Entmax.

That replay may report the number and fraction of V rows that an ideal exact
V-late schedule would need.

It must not call those rows physical memory transactions.

## GQA and reuse limitation

The current trace records individual query-head attention-score inputs.

Qwen3 uses grouped-query attention.

Multiple query heads may share a KV head, and real kernels may reuse K/V data
across queries, heads, warps, blocks, caches, or kernel stages.

Therefore summing per-record skipped V rows does not directly equal physical
KV bandwidth saved.

Any later physical claim must measure the actual implementation.

## E0 non-claims

A2-E0 does not establish:

- wall-clock speedup;
- GPU speedup;
- physical V bandwidth reduction;
- cache-transaction reduction;
- optimal scheduling;
- optimal sparse gather geometry;
- production floating-point certification;
- model-quality suitability;
- novelty.

## Next step if E0 passes

Replay the frozen natural Q/K trace at the A5 E5b qualified setting:

- page size 16;
- leaf divisor 8;
- alpha `{1.5, 2.0}`.

Report separately:

- K tokens total;
- K tokens loaded;
- K tokens pruned;
- exact support tokens;
- logical V rows loaded;
- logical V rows avoided;
- `K_loaded - V_loaded`;
- A5 metadata/frontier counters.

This becomes A2-E1 logical natural-trace qualification before any physical
kernel work.
