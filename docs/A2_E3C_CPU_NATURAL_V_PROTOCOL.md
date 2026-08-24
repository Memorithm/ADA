# ADA-A2 E3c — Natural GQA CPU V-Access Protocol

## Scope

E3c is the first physical wall-clock experiment using the frozen
natural Qwen3 V corpus qualified by E3b.

E3c does not time A5 search, Entmax threshold solving, Q/K scoring,
metadata construction, or model execution.

Its timed scope is only the final value weighted-sum stage.

## Frozen predecessor

E3b qualified commit:

`2616704e9dcf38e76c6cbbe587bcb8ab9c68b190`

E3b established natural Q/K/V output correctness and exact zero
difference between the A5-KLoadedV and A2-SupportV outputs across
6,144 query-head/alpha cases.

E3c does not reopen that mathematical correctness result.

## Three physical V traversal levels

For one natural GQA pair sharing one physical KV head:

1. `FullDenseV`
2. `A5-KUnionV`
3. `A2-SupportUnionV`

The two query heads retain separate probability distributions.

Each unique V row is fetched once by the benchmark kernel and is used
to update both query-head outputs.

The GQA union therefore affects the physical row traversal only. It
does not alter attention semantics.

## Timed representation

The frozen ADAV01 source tensor is bfloat16 and is serialized as
little-endian f32.

E3c converts each selected visible V prefix once to f32 before timing.

The timed kernels therefore use f32 V rows and f32 probabilities.

This is not a bfloat16 hardware-kernel qualification.

## Thor CPU target

The E3c preflight observed:

- NVIDIA Jetson AGX Thor Developer Kit;
- Ubuntu 24.04.4;
- kernel 6.8.12-tegra;
- aarch64;
- 14 online CPU cores;
- CPU frequency fixed at 2.601 GHz;
- MAXN;
- 64 KiB L1d per core;
- 1 MiB unified L2 per core;
- no L3 reported by `lscpu`;
- 64-byte coherency line;
- EMC fixed at 4.266 GHz;
- `taskset` available;
- `perf` unavailable.

The largest frozen f32 V prefix is:

`512 * 128 * 4 = 262144 bytes`

which is below the 1 MiB private L2 capacity.

E3c therefore distinguishes:

- warm/L2-resident behavior;
- cache-evicted refill behavior.

It does not claim direct DRAM-byte measurement.

## Order-bias hardening

The historical E2 harness rotated only three cyclic kernel orders.

E3c must instead execute all six permutations:

1. Full, K, Support
2. Full, Support, K
3. K, Full, Support
4. K, Support, Full
5. Support, Full, K
6. Support, K, Full

The round count must be divisible by six.

This gives every kernel equal exposure to first, middle, and last
position within the timing sequence.

## Equal-work control

Every natural case is paired with an `Support=K` control.

In that control, the KUnion and SupportUnion kernels execute the same
indexed traversal with the same rows and weights.

The K-to-S timing ratio therefore acts as an order/noise sentinel.

It is not expected to prove perfect equality for each noisy timing
sample. Qualification must analyze its distribution across independent
processes.

## Cache-evicted mode

Before each timed kernel call, E3c touches a separate eviction buffer
larger than the private 1 MiB L2.

Eviction work itself is outside the timed interval.

This mode is described as cache-evicted refill.

It is not called a measured DRAM access because no PMU counters are
available.

## Qualification plan

The first run is a small smoke test.

Qualification will require:

- a deterministic natural case panel;
- rounds divisible by six;
- multiple independent pinned processes;
- the same frozen Q/K and V artifact SHAs;
- exact output equality in the benchmark representation;
- natural Full->K, K->Support, and Full->Support timing distributions;
- Support=K control distributions;
- warm and cache-evicted modes analyzed separately;
- environment capture;
- preserved raw logs and SHA manifests.

No production or GPU claim follows from E3c-CPU.
