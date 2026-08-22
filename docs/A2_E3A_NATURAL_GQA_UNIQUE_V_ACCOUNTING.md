# ADA-A2 E3a — Natural GQA Unique-V-Row Accounting

Status:

`A2-E3A-NATURAL-GQA-UNIQUE-V-ROW-ACCOUNTING-QUALIFIED`

## Question

A2-E1 counted K-loaded and final-support rows independently for each
query head. Qwen3-0.6B uses grouped-query attention with 16 query heads
and 8 key/value heads, so two query heads share one physical K/V head.

E3a asks whether the logical V-late opportunity survives after rows are
deduplicated at the actual GQA group boundary.

This is a unique-row accounting experiment. It is not a physical
memory-traffic benchmark.

## Frozen provenance

Model:

`Qwen/Qwen3-0.6B`

Immutable model/tokenizer revision:

`c1899de289a04d12100db370d81485cdf75e47ca`

Frozen WikiText sample JSONL SHA-256:

`8b3cb29d52850020134bd37c1e58dac2ff79508144db49f87ca2801e4e0b4bb0`

Historical four-head E4 trace SHA-256:

`d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`

E3a all-16-head `ADAQK01` trace SHA-256:

`4d4e7b175bc0711f0acb15e3891eacedb2aaada3e11c896e1483123e0cc10ca8`

Capture metadata SHA-256:

`a8333f678c24286c04c4670972cb497fe60bf10e9daaee44cb4e42e2a593f647`

Capture log SHA-256:

`5e972de16036348f2485d0c44e56f5920f12d2ebec1d43f80c665b20667317ae`

Natural replay log SHA-256:

`64352e29f5fd6beabf3f860481fd4ca1de868bcf59fd6a84b7936249538ae935`

The all-head trace contains 3,072 Q-head records and forms 1,536
natural GQA groups:

\[
16\ samples \times 3\ layers \times 8\ KV\ heads \times 4\ positions.
\]

The historical E4 query heads 0, 5, 10, and 15 were filtered from the
new all-head trace and matched the historical 768-record stream exactly,
record for record and byte for byte. The common record-stream SHA-256 was:

`a1798f860e31451f27928215a1bcd9d91c8c4f542a1f5cf78212ee14ffcb53dd`

## Exact GQA accounting

For each natural GQA pair \(q_0,q_1\) sharing one KV head:

\[
K_\cup = K_{q_0}\cup K_{q_1},
\]

\[
S_\cup = S_{q_0}\cup S_{q_1}.
\]

The residual A2 opportunity after A5 is:

\[
A_{2,GQA}
=
1-\frac{|S_\cup|}{|K_\cup|}.
\]

The total logical unique-V-row avoidance relative to all visible rows is:

\[
A_{total,GQA}
=
1-\frac{|S_\cup|}{N}.
\]

No V tensors, cache lines, memory transactions, DRAM bytes, or hardware
counters are measured.

## Global result

### Alpha 1.5

Across 1,536 GQA groups:

- visible rows: 368,640;
- summed per-Q K rows: 202,542;
- summed per-Q support rows: 11,202;
- unique GQA K rows: 137,098;
- unique GQA support rows: 8,555;
- weighted unique K fraction: 0.371902;
- weighted unique support fraction: 0.023207;
- weighted residual A2 V-row avoidance after A5: **0.937599381**;
- weighted total unique-V-row avoidance: **0.976793077**;
- per-Q naive residual avoidance: 0.944692953;
- GQA effect on residual A2 opportunity: **-0.709357 percentage points**.

GQA deduplicated 32.3113231% of the naively summed K rows and
23.6297090% of the naively summed support rows.

### Alpha 2.0

Across 1,536 GQA groups:

- visible rows: 368,640;
- summed per-Q K rows: 157,448;
- summed per-Q support rows: 6,004;
- unique GQA K rows: 110,536;
- unique GQA support rows: 4,466;
- weighted unique K fraction: 0.299848;
- weighted unique support fraction: 0.012115;
- weighted residual A2 V-row avoidance after A5: **0.959596873**;
- weighted total unique-V-row avoidance: **0.987885200**;
- per-Q naive residual avoidance: 0.961866775;
- GQA effect on residual A2 opportunity: **-0.226990 percentage points**.

GQA deduplicated 29.7952340% of the naively summed K rows and
25.6162558% of the naively summed support rows.

## Exception analysis

The experiment does not require every individual GQA group to retain
residual A2 work.

For alpha 1.5, 6 of 1,536 groups had \(K_\cup=S_\cup\), or 0.390625%.
All six occurred in layer 27. Four had \(K_\cup=2\), and two had
\(K_\cup=4\).

For alpha 2.0, 13 of 1,536 groups had \(K_\cup=S_\cup\), or 0.846354%.
All thirteen occurred in layer 27, and every one had \(K_\cup=2\).

These cases are not support-containment failures. They are cases where
A5 has already reduced the unique K-loaded set to a tiny set whose rows
are all members of the final exact support.

Layer 27 still retains weighted residual A2 avoidance of approximately
0.884410 / 0.846082 for alpha 1.5 / 2.0, while total unique-V-row
avoidance is approximately 0.990129 / 0.991943.

## Exactness

The replay preserves exact support containment and all set-cardinality
identities:

\[
S_\cup \subseteq K_\cup,
\]

\[
|K_{q_0}|+|K_{q_1}|
=
|K_\cup|+|K_{q_0}\cap K_{q_1}|,
\]

\[
|S_{q_0}|+|S_{q_1}|
=
|S_\cup|+|S_{q_0}\cap S_{q_1}|.
\]

Observed maximum dense-vs-priority numerical differences were on the
order of f64 oracle noise:

- alpha 1.5 probability: 3.442e-15;
- alpha 1.5 tau: 1.776e-15;
- alpha 2.0 probability: 0;
- alpha 2.0 tau: 0.

## Interpretation

E3a demonstrates that the large natural A2 logical opportunity observed
per query head in E1 survives actual Qwen3 grouped-query sharing.

GQA reduces the residual percentage slightly because the two query heads
sharing a KV head overlap more strongly in their A5-loaded K sets than
in their final sparse supports. The reduction is small relative to the
remaining opportunity.

## Qualified claim

On the frozen Qwen3-0.6B natural Q/K slice, after grouping query heads
by their actual shared KV head and deduplicating requested rows, exact
sparse Entmax support leaves approximately **93.76%** (alpha 1.5) and
**95.96%** (alpha 2.0) of the A5-loaded unique V-row set logically
avoidable.

## Non-claims

This status is not:

- measured physical V traffic;
- measured cache-line traffic;
- measured DRAM/HBM bytes;
- a CPU or GPU speedup;
- end-to-end attention timing;
- a production kernel qualification;
- a model-quality claim;
- a novelty claim.

Physical execution with natural GQA/V data belongs to a later A2 stage.
