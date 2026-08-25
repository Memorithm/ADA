# ADA-A5 E4 — Real Q/K Trace Qualification Protocol

## Purpose

A5-E3 established that exact hierarchical group bounds are strongly distribution-dependent:

- structured `page_clustered` synthetic keys benefit materially from content-aware hybrid bounds;
- high-dimensional `iid_uniform` keys remain essentially unprunable despite sparse Entmax support and despite enclosing-ball bounds often beating coordinate boxes.

E4 therefore does **not** add another synthetic distribution. It measures the same exact candidates on query/key vectors captured from real transformer attention.

The question is:

> Are the post-transform Q/K vectors actually consumed by model attention structured enough for exact A4/A5 support certificates to avoid useful score work?

E4 remains an algorithmic replay experiment. It is not a wall-clock GPU benchmark.

## Capture boundary

A trace record must contain Q and K vectors at the exact numerical stage used by the model's attention score dot product.

That means capture occurs **after** every model-specific transformation that affects the score vectors, including when applicable:

- learned Q/K projection;
- Q/K normalization;
- RoPE or another positional transform;
- model-specific scaling folded into the vectors, if that is how the implementation represents it.

The trace stores the remaining scalar `score_scale` separately.

The capture must occur **before** the QK dot product itself.

E4 must not reconstruct transformed Q/K later from earlier hidden states because that would add framework/model implementation ambiguity.

## Supported attention mask class

E4 v1 supports a contiguous visible key interval for each query:

`[key_start_position, key_start_position + key_count)`.

Ordinary causal decoding is represented by `key_start_position = 0` and `key_count = query_position + 1`.

A contiguous sliding window may also be represented with a non-zero start position.

Arbitrary sparse/non-contiguous masks are outside E4 v1 and must not be silently flattened into this format.

## GQA / MQA provenance

Every record stores both:

- `query_head_index`;
- `kv_head_index`.

The capture tool is responsible for recording the actual Q-to-KV-head mapping used by the model. The replay must never infer that mapping from head counts.

## Numerical storage

E4 v1 stores trace vectors as little-endian IEEE-754 `f32`.

This choice is deliberate:

- native fp32 Q/K are preserved exactly;
- fp16/bfloat16 values cast to fp32 are exactly representable;
- storage is substantially smaller than text or f64.

The global header records the source activation dtype as text. Replay converts stored f32 values to f64 for the existing scalar ADA dense oracle and exact-candidate laboratory checks.

This does **not** qualify a lower-precision production implementation.

## Binary format: `ADAQK01`

All integers are little-endian unsigned values. All strings are UTF-8 and encoded as:

`u32 byte_length` followed by exactly `byte_length` bytes.

### Global header

1. 8-byte magic: ASCII `ADAQK01` followed by one `0x00` byte
2. `u32 format_version` = 1
3. string `model_id`
4. string `model_revision`
5. string `tokenizer_id`
6. string `tokenizer_revision`
7. string `capture_id`
8. string `source_dtype`
9. string `tensor_stage` — E4 v1 requires exact value `attention_score_input`
10. `u32 record_count`

Model and tokenizer revisions must be immutable commit/revision identifiers when the source system exposes them. Floating tags such as `main` are insufficient for qualification evidence.

### Per-record fields

For each record:

1. string `sample_id`
2. `u32 layer_index`
3. `u32 query_head_index`
4. `u32 kv_head_index`
5. `u64 query_position`
6. `u64 key_start_position`
7. `u32 head_dim`
8. `u32 key_count`
9. `f64 score_scale`
10. `head_dim` little-endian f32 query values
11. `key_count * head_dim` little-endian f32 key values in visible-key order

No V tensor is stored in E4 because the current question is support certification before value loading.

## Record validity

Replay rejects a corpus if any of the following holds:

- bad magic/version;
- invalid UTF-8 metadata;
- `tensor_stage != attention_score_input`;
- zero `head_dim` or zero `key_count`;
- non-finite Q/K value or score scale;
- non-positive score scale;
- integer overflow while computing tensor lengths;
- truncated or trailing bytes;
- `query_position` lies before the visible key interval;
- for an ordinary causal-prefix capture declared by the corpus protocol, the stored interval does not match the capture declaration.

The generic v1 parser allows contiguous windows; corpus-level qualification documentation must state whether each corpus uses full causal prefixes or sliding windows.

## Required provenance alongside the binary file

A qualified corpus must record at minimum:

- model repository/id;
- immutable model revision;
- tokenizer id/revision;
- inference/capture software revision;
- model configuration relevant to attention (layer count, Q-head count, KV-head count, head dimension, positional method, attention scaling);
- source activation dtype;
- prompts or immutable dataset/sample identifiers;
- sampling rule for layers, heads, and query positions;
- whether Q/K were captured during prefill or decode;
- mask/window semantics;
- SHA-256 of the `.adaqk` file.

Prompt text may be omitted from the repository when licensing/privacy requires it, but the corpus must then use stable non-sensitive sample identifiers and document the source dataset/revision sufficiently for reproducibility.

## Replay matrix

For every accepted trace record, E4 compares:

1. dense Entmax oracle;
2. flat A4 Q/K page boxes;
3. contiguous A5 hierarchy;
4. content-aware A5 hierarchy using coordinate-box bounds only;
5. content-aware A5 hierarchy using hybrid `min(box, ball)` bounds.

The two content-aware candidates must reuse the exact same deterministic
content-aware index. Their only difference is the node upper bound used for
traversal and pruning. This ablation isolates the effect of content-aware
partition geometry from the effect of the enclosing-ball component.

Initial replay parameters:

- alpha in `{1.5, 2.0}`;
- page size in `{16, 32, 64, 128}`, restricted to useful sizes for the record length;
- hierarchy `leaf_divisor` in `{2, 4, 8}`;
- `leaf_size = ceil(page_size / leaf_divisor)`.

All candidates must preserve dense support and remain within the declared probability/tau tolerances.

## Metrics

E4 reports per record and aggregates:

- dense support size/fraction;
- flat score avoidance;
- contiguous score avoidance;
- content-aware box-only score avoidance;
- content-aware hybrid score avoidance;
- content-aware box-only gain over contiguous;
- content-aware hybrid gain over flat and contiguous;
- hybrid gain over the identical content-aware box-only tree;
- contiguous/content-aware bound evaluations per token;
- content-aware ball-bound win fraction;
- node expansions;
- threshold solves;
- dense probability/tau error;
- layer/head/query-position provenance.

Aggregates must be available at least globally and by layer. Corpus-specific analysis may add head and query-position buckets.

## Frozen natural-corpus qualification slice

The first E4 natural-text qualification slice is frozen before model capture.

Dataset provenance:

- dataset id: `Salesforce/wikitext`;
- configuration: `wikitext-2-raw-v1`;
- split: `validation`;
- immutable dataset revision:
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- source file:
  `wikitext-2-raw-v1/validation-00000-of-00001.parquet`;
- source row count: `3760`;
- source file SHA-256:
  `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`.

Tokenizer provenance:

- tokenizer id: `Qwen/Qwen3-0.6B`;
- immutable tokenizer revision:
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- `add_special_tokens = true`;
- `truncation = true`;
- `max_length = 512`;
- `padding = false`.

Selection rule:

1. divide the 3760 validation rows into 16 equal contiguous row strata using
   integer boundaries `floor(i * 3760 / 16)`;
2. inside each stratum, begin at its first non-empty row;
3. append successive non-empty rows in source order;
4. stop when tokenization under the frozen tokenizer contract reaches the
   truncation length of exactly 512 tokens;
5. retain the full accumulated source text in the JSONL so the capture adapter
   performs the final deterministic truncation itself.

This produces 16 samples, each verifying to exactly 512 adapter tokens.

Frozen generated-corpus identities:

- sample count: `16`;
- JSONL SHA-256:
  `8b3cb29d52850020134bd37c1e58dac2ff79508144db49f87ca2801e4e0b4bb0`;
- manifest SHA-256:
  `2d6587980f4db0322067e954022c38e8b445ea99e0252c219738580c561d5362`.

The selected source-row intervals are:

- sample 00: rows `1..10`;
- sample 01: rows `235..247`;
- sample 02: rows `470..478`;
- sample 03: rows `705..713`;
- sample 04: rows `940..949`;
- sample 05: rows `1175..1188`;
- sample 06: rows `1411..1423`;
- sample 07: rows `1645..1653`;
- sample 08: rows `1880..1882`;
- sample 09: rows `2115..2123`;
- sample 10: rows `2352..2360`;
- sample 11: rows `2585..2603`;
- sample 12: rows `2821..2823`;
- sample 13: rows `3055..3063`;
- sample 14: rows `3290..3298`;
- sample 15: rows `3527..3534`.

The local filenames used to create these hashes are not part of the identity.
Qualification relies on the immutable dataset revision, source SHA-256,
selection algorithm, tokenizer revision, JSONL SHA-256, and manifest SHA-256.

This 16-sample slice is a deterministic natural-text qualification corpus. It
is broader than the synthetic/smoke prompt used during capture debugging, but
it is still a limited WikiText-2 validation slice and must not be described as
a general model-distribution benchmark.

## Decision rule

E4 is intended to discriminate among three outcomes.

### Real traces behave like structured synthetic keys

If exact content-aware bounds obtain meaningful pruning across multiple layers/heads/query positions, preserve A5 and proceed to:

1. lazy/on-demand bound evaluation;
2. A2 K-first / V-late integration;
3. only then a hardware implementation/benchmark.

### Real traces behave like `iid_uniform`

If pruning remains near zero while eager metadata work is substantial, deprioritize the current exact group-bound family rather than optimizing it for GPU.

### Mixed model/layer/head behavior

If usefulness is strongly localized, preserve the candidate as a conditional mechanism and hand the selection problem to later ADA-A9 / ElasticXxx policy work rather than forcing a universal attention path.

## Non-claims

E4 does not by itself establish:

- model-quality suitability of Entmax versus the model's trained Softmax;
- production floating-point certification;
- hardware speedup;
- reduced physical KV traffic;
- novelty.

It answers only whether real Q/K geometry makes the existing exact support-certification mechanism algorithmically promising enough to justify the next implementation stage.
