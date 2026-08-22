# ADA-A5 E4 — Real Q/K Trace Qualification Results

## Scope

E4 evaluates exact A4/A5 Entmax support-certification candidates on Q/K
vectors captured from a frozen natural-text slice using immutable Qwen3 model
and tokenizer revisions.

This is an algorithmic replay qualification. It is not a wall-clock benchmark,
GPU qualification, physical KV-traffic measurement, model-quality claim, or
production floating-point certification.

## Frozen source corpus

Dataset:

- id: `Salesforce/wikitext`
- configuration: `wikitext-2-raw-v1`
- split: `validation`
- immutable revision:
  `b08601e04326c79dfdd32d625aee71d232d685c3`
- source rows: `3760`
- source parquet SHA-256:
  `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`

Generated natural-text slice:

- samples: `16`
- adapter tokens per sample: `512`
- JSONL SHA-256:
  `8b3cb29d52850020134bd37c1e58dac2ff79508144db49f87ca2801e4e0b4bb0`
- manifest SHA-256:
  `2d6587980f4db0322067e954022c38e8b445ea99e0252c219738580c561d5362`

The slice was frozen before observing the E4 model results.

## Model and capture provenance

Model/tokenizer:

- id: `Qwen/Qwen3-0.6B`
- immutable revision:
  `c1899de289a04d12100db370d81485cdf75e47ca`
- source activation dtype: `bfloat16`

Capture:

- capture id: `qwen3-0.6b-e4-wikitext2raw-val16`
- tensor stage: `attention_score_input`
- device used for capture: CPU
- layers: `0, 13, 27`
- Q heads: `0, 5, 10, 15`
- query positions: `63, 127, 255, 511`
- records: `768`

Artifact identities:

- trace SHA-256:
  `d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`
- capture metadata SHA-256:
  `ba7f4acc42cec152a2bc6d851dbcddbebc401c2317cffe97745c3de9cc08b710`
- capture log SHA-256:
  `0d6d63a0446bd5aa6875b746f9f868cf395a218656389fe5d29f135e77760fa2`
- replay log SHA-256:
  `553cb1dba5401f67469d9e3dda0cb0e5f475b1564c6d71ab21893dd02b69dbf1`

## Replay matrix

The replay contains `18,432` comparison cases:

- alpha: `{1.5, 2.0}`
- page size: `{16, 32, 64, 128}`
- leaf divisor: `{2, 4, 8}`

Candidates:

1. dense Entmax oracle;
2. flat A4 coordinate-box pages;
3. contiguous A5 hierarchy;
4. content-aware A5 hierarchy with box-only bounds;
5. the identical content-aware hierarchy with hybrid `min(box, ball)` bounds.

The content-aware box-only and hybrid candidates reuse exactly the same
content-aware index and differ only in their selected node bound.

## Correctness

All `18,432` replay cases completed.

Maximum observed absolute differences against the dense oracle:

- flat probability: `1.776e-15`
- contiguous probability: `2.998e-15`
- content-box probability: `2.998e-15`
- content-hybrid probability: `2.998e-15`
- flat tau: `1.776e-15`
- contiguous tau: `1.776e-15`
- content-box tau: `1.776e-15`
- content-hybrid tau: `1.776e-15`

No dense-support false negative was reported.

Replay completion marker:

`survey_status=complete`

## Global pruning result

The strongest tested global configuration is page size 16 with leaf divisor 8.

For alpha 1.5:

- mean dense support fraction: `0.026568`
- flat score avoidance: `0.000854`
- contiguous score avoidance: `0.639074`
- content-box score avoidance: `0.660899`
- content-hybrid score avoidance: `0.660899`
- content partition gain over contiguous: `+0.021825`

For alpha 2.0:

- mean dense support fraction: `0.014832`
- flat score avoidance: `0.002970`
- contiguous score avoidance: `0.712097`
- content-box score avoidance: `0.736760`
- content-hybrid score avoidance: `0.736760`
- content partition gain over contiguous: `+0.024663`

Flat coordinate-box pages therefore remain largely unable to exploit the sparse
support, while hierarchical refinement produces substantial exact score
avoidance on this natural Q/K slice.

## Layer dependence

At page size 16 and leaf divisor 8:

### Alpha 1.5

Layer 0:

- support: `0.037529`
- contiguous: `0.425247`
- content-box: `0.486404`
- partition gain: `+0.061157`

Layer 13:

- support: `0.027000`
- contiguous: `0.560074`
- content-box: `0.570602`
- partition gain: `+0.010529`

Layer 27:

- support: `0.015175`
- contiguous: `0.931900`
- content-box: `0.925690`
- partition gain: `-0.006210`

### Alpha 2.0

Layer 0:

- support: `0.017174`
- contiguous: `0.519882`
- content-box: `0.586624`
- partition gain: `+0.066742`

Layer 13:

- support: `0.014755`
- contiguous: `0.662964`
- content-box: `0.674179`
- partition gain: `+0.011215`

Layer 27:

- support: `0.012566`
- contiguous: `0.953445`
- content-box: `0.949478`
- partition gain: `-0.003967`

The preferred partition is therefore layer-dependent. The early and middle
sampled layers benefit from the current content-aware partition, while the late
sampled layer is already better served by contiguous hierarchy.

## Context-position dependence

At page size 16 and leaf divisor 8, pruning remains strong and increases rather
than collapsing at longer visible prefixes.

### Alpha 1.5

- position 63: contiguous `0.496257`, content-box `0.519368`
- position 127: contiguous `0.596029`, content-box `0.617513`
- position 255: contiguous `0.693075`, content-box `0.714762`
- position 511: contiguous `0.770935`, content-box `0.791952`

### Alpha 2.0

- position 63: contiguous `0.576009`, content-box `0.602865`
- position 127: contiguous `0.681396`, content-box `0.705485`
- position 255: contiguous `0.762533`, content-box `0.788574`
- position 511: contiguous `0.828451`, content-box `0.850118`

The tested A5 benefit therefore cannot be explained solely by very short
prefixes.

## Content-aware partition ablation

Content-aware partitioning is not universally better.

At page size 16 / divisor 8 it is beneficial globally and especially on layers
0 and 13, but contiguous hierarchy is slightly better on layer 27.

Across the full global matrix, content partition gain ranges from:

- maximum: `+0.024663` at alpha 2.0, page 16, divisor 8;
- minimum: `-0.021891` at alpha 1.5, page 32, divisor 8.

This supports conditional plan selection rather than a universal
content-aware policy.

## Enclosing-ball ablation

The enclosing-ball component did not alter score avoidance in any replay case.

Across all `18,432` cases:

- positive hybrid gain over content-box: `0`
- negative hybrid gain over content-box: `0`
- zero hybrid gain over content-box: `18,432`
- maximum hybrid gain: exactly `0`

Some nodes still have a ball upper bound tighter than their coordinate-box
upper bound, but that tightening never crosses the Entmax pruning certificate
threshold in this corpus.

Therefore `ball_upper < box_upper` is not by itself evidence of useful pruning.

The current enclosing-ball component should be preserved as an ablation and
prior-art reference but deprioritized as an A5 execution mechanism unless a
new distribution or bound construction produces decision-relevant tightening.

## E4 decision

The frozen natural-text slice does not behave like the critical synthetic
`iid_uniform` failure mode.

Exact hierarchical bounds obtain substantial pruning across multiple sampled
layers, Q heads, and query positions.

A5 is therefore retained for the next research stage.

Recommended sequence:

1. design lazy/on-demand hierarchical bound evaluation;
2. combine candidate reduction with ADA-A2 K-first / V-late staging;
3. measure actual avoided K/V work and metadata overhead;
4. only then implement and benchmark a hardware realization.

The observed layer/head dependence also promotes the need for ADA-A9
distribution/state-aware plan selection:

- contiguous hierarchy is sometimes best;
- content-aware partitioning is sometimes better;
- no single current partition should be forced universally.

## Qualification limits

This E4 result qualifies only the reported algorithmic observations on the
frozen 16-sample WikiText-2 validation slice.

It does not establish:

- general Qwen3-distribution behavior;
- model-quality suitability of Entmax;
- production floating-point certification;
- physical KV bandwidth savings;
- latency or throughput speedup;
- GPU viability;
- novelty.
