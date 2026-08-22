# ADA-A2 E2 — Physical V Access Laboratory

## Status

Qualified stage:

`A2-E2-THOR-PHYSICAL-V-LATE-QUALIFIED`

Qualification applies to the isolated CPU physical V-access mechanism on the
NVIDIA Jetson AGX Thor target.

It does not imply end-to-end attention, GPU, DRAM/HBM, model-quality, or
production qualification.

The qualifying campaign used five independent processes from committed harness
SHA `516527971fe2311e85af849813b990e130ec2b3c`.

## Purpose

A2-E1 established that the final exact Entmax support is much smaller than the
set of tokens whose exact K scores must still be loaded by A5.

A2-E2 asks whether that *additional* sparsity produces a measurable physical
CPU benefit when V materialization is delayed until the exact support is known.

## Attribution correction

The original E2 preflight compared:

\[
\text{FullDense-V}
\rightarrow
\text{Support-V}.
\]

That comparison mixes two different mechanisms:

1. A5 removes tokens before exact K-score loading;
2. A2 avoids V rows for K-loaded tokens that ultimately have zero Entmax
   probability.

The corrected physical experiment therefore uses:

\[
\boxed{
\text{FullDense-V}
\rightarrow
\text{A5-KLoaded-V}
\rightarrow
\text{A2-Support-V}
}
\]

with

\[
S
\subseteq
K_{\rm loaded}
\subseteq
\{0,\ldots,N-1\}.
\]

All three kernels compute the same weighted output:

\[
O=\sum_i p_iV_i.
\]

## Kernels

### FullDense-V

Reads every visible V row.

Rows with zero probability are still visited and their V values participate
in the executed row loop.

### KLoaded-V

Reads every row belonging to the declared A5 K-loaded set.

For

\[
i\in K_{\rm loaded}\setminus S
\]

the probability is exactly zero, but the V row is still read.

This is the physical baseline for measuring the additional contribution of A2.

### Support-V

Reads only rows in the exact final positive support.

This models V-late execution once Entmax support has been resolved exactly.

## Ratios

A5-side V-access ratio:

\[
G_{A5}
=
\frac{T_{\rm FullDense}}
     {T_{\rm KLoaded}}.
\]

Incremental A2 ratio after A5:

\[
\boxed{
G_{A2|A5}
=
\frac{T_{\rm KLoaded}}
     {T_{\rm Support}}.
}
\]

Total V-only ratio:

\[
G_{\rm total}
=
\frac{T_{\rm FullDense}}
     {T_{\rm Support}}.
\]

Only \(G_{A2|A5}\) is the isolated A2 physical metric.

A large \(G_{\rm total}\) alone is insufficient evidence for A2.

## Natural E1 anchors

Frozen A2-E1 logical accounting produced:

### alpha = 2

- total visible token instances: 184320
- K loaded: 41576
- final support V rows: 1538
- K fraction: approximately 22.5564%
- support fraction: approximately 0.8344%

### alpha = 1.5

- total visible token instances: 184320
- K loaded: 53722
- final support V rows: 2833
- K fraction: approximately 29.1461%
- support fraction: approximately 1.5370%

E2 uses these fractions as synthetic physical anchor densities.

They do not reconstruct the exact natural support geometry.

## Matrix

`value_dim = 128`, `f32`.

Token counts:

- 64
- 128
- 256
- 512
- 2048
- 8192
- 32768
- 65536

K-load densities include:

- 22.5564%
- 29.1461%
- 100%

Support densities include:

- 0.5%
- 0.8344%
- 1.5370%
- 2%
- 5%

Each K density additionally receives an `S=K` control.

Requested and realized densities are both reported because integer row counts
matter for small N.

## Locality patterns

### prefix

K rows form a contiguous prefix and support is nested inside that prefix.

### spread

K rows are distributed across the visible token interval.

Support rows are distributed over the K set and remain a strict subset of K.

These two cases are deterministic locality probes, not measurements of natural
Qwen support geometry.

## Cache-footprint labels

The target Thor environment documents:

- private L2: 1 MiB per CPU core;
- shared system L3: 16 MiB.

The runner reports:

- `l2_capacity`
- `l3_capacity`
- `beyond_l3`

according to the full V tensor footprint.

These labels compare footprint to documented cache capacities.

They do not prove actual residency or cache-hit behavior.

## Timing

Timing uses:

- `std::time::Instant`;
- `std::hint::black_box`;
- no allocation inside the measured kernels;
- rotation of Full/K/Support order by round;
- median;
- p95;
- MAD.

### Warm mode

Warm samples repeatedly execute the same tensor configuration.

Different kernels use different iteration counts so sparse paths accumulate
enough work to reduce timer quantization.

This means warm measurements deliberately permit strong reuse of the same V
rows.

Therefore very large warm ratios, especially for tiny support sets, must not be
interpreted as direct one-pass DRAM or end-to-end speedups.

Warm mode is primarily a kernel-cost/locality measurement.

### Evicted mode

Before each individually timed kernel call, the runner touches a separate
32 MiB eviction buffer.

The eviction work is outside the timed interval.

The term is deliberately `evicted`, not `cold`.

This procedure creates software cache pressure but does not establish a
specific hardware cache state.

Focused evicted probes cover N=512, N=8192, and N=65536 for natural-like and
stress configurations.

The evicted natural-anchor results are the more conservative physical guard
against overinterpreting warm batching.

## Correctness

Every generated case verifies:

\[
O_{\rm FullDense}
\approx
O_{\rm KLoaded}
\approx
O_{\rm Support}.
\]

The current deterministic f32 tolerance is:

\[
2\times10^{-5}.
\]

Support membership is constructed and checked so that:

\[
S\subseteq K_{\rm loaded}.
\]

## Negative control

When:

\[
S=K_{\rm loaded},
\]

KLoaded-V and Support-V visit the same indexed physical row set.

Therefore:

\[
G_{A2|A5}\approx1
\]

is expected.

A large systematic speedup in this control would indicate a benchmark defect or
measurement bias.

## Current non-qualifying smoke

The current one-process, seven-round Thor smoke produced:

- 306 result records;
- one complete survey;
- zero observed Full/K and K/Support output difference;
- zero A2 non-wins on the alpha=2 natural-like anchor;
- zero A2 non-wins on the alpha=1.5 natural-like anchor.

Observed isolated A2-after-A5 minima:

- alpha=2: 4.343910x;
- alpha=1.5: 7.207511x.

Observed medians:

- alpha=2: 21.011277x;
- alpha=1.5: 17.051900x.

The `S=K` control median was 1.000777x.

These values are preflight evidence only.

## Qualified Thor evidence

Evidence directory:

`evidence/a2-k-first-v-late/e2-thor-three-level-516527971fe2-20260822T142253Z`

Campaign:

- 5 independent processes;
- 21 rounds per configuration;
- 4,000,000 target scalar operations for warm batching;
- 306 records per process;
- 1530 aggregate records;
- zero observed numerical difference across all three kernels;
- zero natural-anchor A2 non-wins.

For the alpha=2 natural-like anchor:

- minimum isolated A2-after-A5 ratio: 4.945682x;
- median: 21.123188x;
- maximum: 67.185331x.

For the alpha=1.5 natural-like anchor:

- minimum isolated A2-after-A5 ratio: 4.882465x;
- median: 17.026632x;
- maximum: 52.448346x.

The global worst natural-anchor result was therefore:

\[
G_{A2|A5}=4.882465.
\]

Every independent process qualified individually.

The `S=K` negative control had aggregate median:

\[
G_{A2|A5}=1.000000,
\]

with per-process medians all approximately unity.

Raw evidence manifest SHA-256:

`3a5f428ee0fa90a2aff1259b0b38f0f5b8cab5dbe4548311f9d51999e52e8b37`

The raw manifest is preserved unchanged.

See the evidence-local `QUALIFICATION.md` for the complete classification
summary and scope restrictions.

## Reproduction campaign

Qualification must run from a clean committed harness.

Required environment:

- NVIDIA Jetson AGX Thor Developer Kit;
- MAXN;
- fixed min/max CPU frequency on the pinned core;
- default pinned core 13 unless explicitly overridden;
- deterministic release binary;
- multiple independent benchmark processes.

Before measurement:

- `cargo fmt --all -- --check`;
- strict workspace Clippy;
- workspace tests;
- release build;
- analyzer Python compile;
- shell syntax validation.

The qualification script records:

- Git SHA;
- system/kernel/toolchain information;
- CPU topology;
- NVIDIA power mode;
- `jetson_clocks --show`;
- thermal state before and after;
- process snapshot;
- one raw benchmark file per process;
- aggregate analyzer output;
- SHA-256 hashes for evidence files.

Performance outcomes are never used as a shell success condition.

Negative evidence must be preserved.

## Qualification criterion

A2-E2 is a positive physical CPU mechanism only if:

\[
G_{A2|A5}>1
\]

robustly across the natural E1 anchors, including both locality patterns and
the focused evicted cases.

The `S=K` control must remain consistent with approximately equal KLoaded and
Support work.

Qualification of this stage does not imply promotion to production.

## Non-claims

A2-E2 does not establish:

- end-to-end attention speedup;
- GPU speedup;
- HBM or DRAM byte counts;
- cache-line transaction counts;
- PMU events;
- exact natural-model V traffic;
- GQA-aware physical reuse;
- model-quality impact;
- production readiness;
- novelty.

The machine currently has no qualified `perf`/PMU path for this experiment.

## Next evidence stage

If E2 qualifies, the next stronger experiment should use actual natural support
masks and eventually real/captured V data with GQA-aware deduplication.

The frozen `ADAQK01` trace must not be silently changed.

A new trace format or companion V artifact must be versioned explicitly.
