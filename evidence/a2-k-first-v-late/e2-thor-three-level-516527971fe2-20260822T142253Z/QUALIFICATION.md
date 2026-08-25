# ADA-A2 E2 — Thor Physical V-Late Qualification

## Classification

`A2-E2-THOR-PHYSICAL-V-LATE-QUALIFIED`

This classification applies specifically to the CPU physical V-access
mechanism measured by ADA-A2 E2.

It does not constitute an end-to-end attention, GPU, DRAM/HBM, model-quality,
or production-performance qualification.

## Harness

Harness commit:

`516527971fe2311e85af849813b990e130ec2b3c`

Experiment:

`ADA-A2-E2-three-level-physical-v-access`

Target:

NVIDIA Jetson AGX Thor Developer Kit.

Pinned CPU:

`13`

CPU frequency:

`2601000 kHz`

Power mode:

`MAXN`

PMU counters:

not used.

## Physical attribution

The qualified experiment isolates:

\[
\text{FullDense-V}
\rightarrow
\text{A5-KLoaded-V}
\rightarrow
\text{A2-Support-V}.
\]

The A2-specific physical metric is:

\[
G_{A2|A5}
=
\frac{T_{\rm KLoaded}}
     {T_{\rm Support}}.
\]

The FullDense-to-Support ratio is not attributed solely to A2.

## Campaign

Independent processes:

`5`

Rounds per benchmark configuration:

`21`

Target scalar work:

`4000000`

Records per process:

`306`

Aggregate records:

`1530`

All five surveys completed.

All five processes independently satisfied the natural-anchor A2 criterion.

## Numerical correctness

Across all 1530 records:

\[
\max |O_{\rm FullDense}-O_{\rm KLoaded}|=0
\]

and

\[
\max |O_{\rm KLoaded}-O_{\rm Support}|=0.
\]

The configured deterministic f32 qualification tolerance was:

\[
2\times10^{-5}.
\]

## alpha = 2 natural-like anchor

Cases:

`110`

K density:

approximately `22.5564%`

Support density:

approximately `0.8344%`

Isolated A2-after-A5 results:

- minimum: `4.945682x`
- median: `21.123188x`
- maximum: `67.185331x`
- non-wins: `0`

## alpha = 1.5 natural-like anchor

Cases:

`110`

K density:

approximately `29.1461%`

Support density:

approximately `1.5370%`

Isolated A2-after-A5 results:

- minimum: `4.882465x`
- median: `17.026632x`
- maximum: `52.448346x`
- non-wins: `0`

## Worst qualified natural-anchor case

The minimum observed natural-anchor A2-after-A5 ratio across all five
independent processes was:

\[
\boxed{4.882465\times}
\]

Configuration:

- mode: `evicted`
- tokens: `512`
- footprint region: `l2_capacity`
- pattern: `prefix`
- K density: alpha=1.5-like
- support density: alpha=1.5-like

No natural-anchor A2 non-win occurred.

## Cache-footprint strata

Natural-anchor minimum A2-after-A5 ratios:

- L2-capacity-labelled cases: `4.882465x`
- L3-capacity-labelled cases: `9.308961x`
- beyond-L3-labelled cases: `10.644142x`

These labels compare V footprint against documented cache capacities and do not
prove actual cache residency.

## Evicted / spread guard

The more locality-adverse focused cases remained positive.

Examples include:

- alpha=2, evicted spread:
  minimum `6.348457x`,
  median `12.433828x`;

- alpha=1.5, evicted spread:
  minimum `6.416581x`,
  median `10.644142x`.

`evicted` means that a separate 32 MiB buffer was touched before each timed
kernel. It is not a proof of a specific cold-cache state.

## S = K negative control

When final Support equals KLoaded, the two indexed kernels perform the same
logical row work.

Across 240 controls:

- A2-after-A5 minimum: `0.839423x`
- median: `1.000000x`
- maximum: `1.307530x`

Per-process control medians were:

- run 01: `1.000000x`
- run 02: `1.000342x`
- run 03: `1.001241x`
- run 04: `1.000000x`
- run 05: `1.000000x`

The control is therefore centered essentially at unity.

The observed control spread is treated as physical measurement variability,
not as an A2 benefit.

## Per-process natural minima

- run 01: `5.220146x`
- run 02: `6.288681x`
- run 03: `6.243024x`
- run 04: `4.882465x`
- run 05: `5.309556x`

Every process had:

- 306 records;
- one complete survey;
- zero natural-anchor non-wins;
- zero observed correctness difference.

## Evidence integrity

Raw evidence manifest:

`SHA256SUMS.txt`

Manifest SHA-256:

`3a5f428ee0fa90a2aff1259b0b38f0f5b8cab5dbe4548311f9d51999e52e8b37`

Aggregate analysis SHA-256:

`7caecd968527a7029faba7126833715bae2eaf94042624afed37d0fe7da630c2`

The original `SHA256SUMS.txt` is intentionally left unchanged after
qualification verification.

`QUALIFICATION.md` is a derived classification document and is not part of the
original raw-evidence manifest.

## Interpretation

A2-E2 establishes that, for the measured sparse-Entmax regimes on Thor,
delaying V access until exact final support resolution retains a substantial
physical CPU advantage even after A5 has already received credit for K
pruning.

The measured statement is therefore specifically:

\[
T_{\rm KLoaded}
>
T_{\rm Support}
\]

with substantial margin in all qualified natural-like cases.

## Non-claims

This qualification does not establish:

- complete attention-kernel speedup;
- transformer inference speedup;
- GPU speedup;
- HBM or DRAM traffic reduction measured by counters;
- exact cache-line transaction reduction;
- natural GQA physical reuse;
- real-model V tensor access geometry;
- model-quality effects;
- production readiness;
- algorithmic novelty.

## Next research stage

The appropriate next strengthening stage is natural V/GQA evidence.

The frozen `ADAQK01` trace must remain unchanged.

Any V-bearing extension should use a new explicitly versioned artifact, such
as a companion V trace or a new QKV trace format.
