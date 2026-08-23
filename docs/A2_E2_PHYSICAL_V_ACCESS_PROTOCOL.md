# ADA-A2 E2 — Three-Level Physical V-Access Microbenchmark

## Status

Physical CPU wall-clock microbenchmark. Supersedes the earlier E2 preflight
comparison `FullDense-V -> Support-V` (kept for history in
`crates/ada-a2-k-first-v-late/examples/e2_v_access_microbench.rs`), which is
methodologically insufficient and must not be cited as isolated A2 evidence.

## Why `FullDense -> Support` was insufficient

The preflight benchmark compared a dense V scan against an exact-support
gather. On the target CPU it produced strongly positive results. Those results
are real wall-clock numbers, but they are not attributable to A2 alone,
because the compared kernels differ in two independent row sets at once:

1. **A5 K pruning** removes rows from the loaded K set. Any row pruned by A5
   cannot contribute V traffic either, whether or not V-late is used.
2. **A2 V-late** shrinks the *V* read set to the exact positive support
   *within* the rows that survived A5.

Comparing `FullDense-V` directly against `Support-V` therefore measures the
composition of both effects. Attributing that composite speedup to A2 would
double-count work that A5 already removed. This is the methodological flaw E2
exists to fix.

## Correct three-level decomposition

The physically meaningful chain is:

```
FullDense-V  ->  A5-KLoaded-V  ->  A2-Support-V
```

with the strict set nesting

```
Support  subset  KLoaded  subset  AllVisibleTokens
```

Three distinct kernels are timed:

### FullDense

Reads every visible V row and multiplies every row — including rows whose
probability is exactly zero — into the output. This is the no-A5, no-A2
upper-bound workload.

### KLoaded

Reads exactly the A5-surviving K-loaded rows. Rows in `KLoaded \ Support`
have probability exactly zero but their V rows are still physically read;
zero-weight rows are never optimized away before access. This kernel models
the state where A5 pruning has happened but A2 V-late has **not**.

### Support

Reads only the final positive-support V rows. This models A2 V-late after
exact support resolution.

All three kernels compute

\[
O=\sum_i p_iV_i
\]

and every generated case must satisfy structural correctness
(`Support subset KLoaded`) and numerical agreement within a declared f32
tolerance before any timing is recorded.

## Key ratios

With kernel times `T_FullDense`, `T_KLoaded`, `T_Support`:

| Ratio | Definition | Physical meaning |
|---|---|---|
| `G_A5` | `T_FullDense / T_KLoaded` | isolated physical effect of A5 K pruning on V traffic |
| `G_A2_after_A5` | `T_KLoaded / T_Support` | isolated physical effect of A2 V-late **given** A5 |
| `G_total` | `T_FullDense / T_Support` | composed effect of A5 + A2 |

The **primary A2 metric is `G_A2_after_A5`**. `G_total` is reported for
completeness and must never be attributed to A2 alone. Because the nesting
above holds by construction, the identity
`G_total = G_A5 x G_A2_after_A5` is expected up to measurement noise, and the
analyzer reports all three so over-attribution is detectable.

The isolated A2 physical promotion criterion is:

> `G_A2_after_A5 > 1`, robustly, across the natural E1 anchor regimes and
> both locality patterns, in warm mode and in focused evicted probes.

Negative or near-unity results are valid findings and must be reported, not
tuned away.

## Natural E1 anchors

E2 embeds the frozen natural Qwen3-0.6B accounting from A2-E1 as central
cases:

| Anchor | K density | Support density |
|---|---|---|
| alpha = 2.0 | 22.5564% (`225564` ppm) | 0.8344% (`8344` ppm) |
| alpha = 1.5 | 29.1461% (`291461` ppm) | 1.5370% (`15370` ppm) |

Additional stress/control densities: 0.5%, 2%, 5%; `K = 100%` as a
no-A5-pruning control; and `Support = KLoaded` as a no-A2-residual control.

For `Support = KLoaded` the KLoaded and Support kernels perform the identical
indexed physical row set, so `G_A2_after_A5 ~= 1` **must** be observed. A
large artificial speedup in this control indicates a broken benchmark, not a
discovery. For `K = 100%`, `G_A5 ~= 1` is likewise expected.

Because small token counts round integer row counts upward, every record
reports both requested and realized densities.

## Locality patterns

Two deterministic synthetic layouts bracket locality behavior:

- `prefix`: K rows contiguous from token 0; support nested contiguously
  inside K.
- `spread`: K rows distributed across the full visible interval; support
  distributed across K, remaining a subset of K.

These are physical locality probes. They do **not** reproduce the natural
Qwen support geometry measured in E1; they bound it from friendly and
adversarial directions.

## Shapes and cache regimes

`value_dim = 128`, `f32`. Token counts and nominal V footprints:

| N | V footprint bytes | descriptive region |
|---|---|---|
| 64 | 32768 | l2_capacity |
| 128 | 65536 | l2_capacity |
| 256 | 131072 | l2_capacity |
| 512 | 262144 | l2_capacity |
| 2048 | 1048576 (~1 MiB) | l2_capacity |
| 8192 | 4194304 (~4 MiB) | l3_capacity |
| 32768 | 16777216 (~16 MiB) | l3_capacity |
| 65536 | 33554432 (~32 MiB) | beyond_l3 |

`cache_region` labels are descriptive footprint classifications against the
documented 1 MiB private L2 / 16 MiB shared L3 of the target. They are **not**
claims of measured cache residency.

## Timing protocol

- `f32`, fixed seed, deterministic layouts, no allocation inside timed
  kernels.
- `std::time::Instant`; `std::hint::black_box` on case and outputs.
- Warm mode uses batched timing: each sample times many back-to-back kernel
  invocations chosen to reach a configurable scalar-work target, divided by
  the batch iteration count, so samples are not dominated by timer overhead.
  Batch iteration counts are reported per record.
- Kernel timing order rotates per round (`Full,K,Support` /
  `K,Support,Full` / `Support,Full,K`) to reduce ordering bias.
- Per kernel: median, p95, MAD over rounds, plus iteration counts.
- Speedups are emitted as integer ppm ratios
  (`*_speedup_ppm = numerator/denominator x 1e6`).

### Modes

- `warm`: repeated accesses to the same tensors; caches are allowed to help.
- `evicted`: a dedicated eviction buffer larger than shared L3 is touched
  before each individually timed kernel call, outside the timed interval.
  Single-call timing is required here because the eviction must precede each
  timed access; timer overhead inflates all three kernels approximately
  equally and therefore biases ratios toward 1, i.e. against A2, which is the
  conservative direction.

This mode is deliberately named `evicted`, not `cold`: touching a software
buffer does not prove a precise hardware cache state.

Focused evicted probes cover representative shapes `N=512`, `N=8192`,
`N=65536` for both natural anchors under both patterns.

## Output format

Stable line-oriented `key=value` records prefixed `result,`, including mode,
shape, footprint/region labels, requested and realized densities with row
counts, per-kernel iteration counts, median/p95/MAD ns, the three speedup
ratios in ppm, and maximum absolute correctness errors. The sentinel line
`survey_status=complete` is printed only after the full matrix finishes
successfully.

`tools/analyze_a2_e2.py` (Python standard library only) parses one or more
raw logs, rejects malformed records, verifies survey completion and
correctness limits, summarizes the three ratios by anchor/pattern/mode/region,
counts `G_A2_after_A5 <= 1` non-wins, identifies the worst natural-anchor
`G_A2_after_A5`, and validates the `Support=KLoaded` and `K=100%` controls.

## Qualification environment

`scripts/thor_a2_e2.sh` runs qualification on the Thor and requires: clean
Git tree, committed SHA, MAXN, pinned core, fixed CPU frequency, fmt check,
strict workspace clippy, workspace tests, release build of the three-level
runner, multiple independent pinned benchmark processes, system metadata,
power mode, jetson_clocks output, thermal snapshots, process snapshots, raw
evidence file, and SHA-256. `perf` is not available on this target and is not
required.

## Interpretation boundaries

An E2 result means only:

> On this named Thor CPU, under this protocol, with support already resolved,
> the measured indexed support-gather V kernel was faster than the measured
> K-loaded V kernel (or dense scan), by the stated ratio.

## Non-claims

E2 does not establish:

- logical row avoidance (that is E1's accounting domain);
- physical memory transactions, cache-line or DRAM byte counts (no PMU);
- measured cache residency;
- natural model V-traffic reduction on real Qwen execution;
- GPU execution benefit;
- end-to-end attention speedup;
- production-kernel quality, GQA reuse, or model-quality suitability;
- novelty.

Logical avoidance numbers come from A2-E1. Physical memory-transaction claims
would require a qualified PMU path. GPU and end-to-end questions belong to
later, separate experiments.

## Promotion rule

Proceed toward integrated physical execution only if `G_A2_after_A5 > 1`
holds robustly on the natural E1 anchor regimes and both locality patterns,
with clean `Support=KLoaded` controls. If the isolated A2 effect vanishes
once A5's contribution is correctly factored out, the V-late physical
strategy must be revised even though the earlier composite preflight looked
favorable.
