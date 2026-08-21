# ADA-A5 E4 — Qwen3 real-Q/K capture adapter

## Scope

This document defines the first model-specific capture adapter used by A5-E4.
It does **not** qualify A5 on real model distributions by itself. Qualification
requires a frozen input corpus, a clean committed capture implementation, the
binary trace and metadata sidecar, a successful Rust replay, and preserved raw
evidence with hashes.

## First target

The initial target is:

- model: `Qwen/Qwen3-0.6B`
- immutable model revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- tokenizer: same repository and revision
- architecture: Qwen3 causal LM
- hidden layers: 28
- query heads: 16
- KV heads: 8
- head dimension: 128
- source checkpoint dtype: bfloat16
- sliding window: disabled

The model is intentionally small enough for a first physical capture while still
exercising RoPE, Q/K normalization and grouped-query attention.

## Capture boundary

`tools/capture_qwen3_e4.py` loads the model with
`attn_implementation="eager"` and patches only the Qwen3 module-level eager
attention fallback.

The Qwen3 attention layer computes, in order:

1. Q/K/V projections;
2. Q/K RMS normalization;
3. RoPE;
4. optional cache update;
5. attention backend call.

The wrapper observes the `query` and `key` tensors passed at step 5 and then
immediately delegates to the original eager backend. The adapter therefore does
not reimplement Q/K normalization, RoPE, masking, softmax or the output path.

For E4 v1 the capture uses `use_cache=False` and one unpadded sequence per
forward pass. The visible interval for query position `p` is recorded as
`[0, p + 1)`.

## GQA mapping

The attention backend receives Q as `[1, 16, T, 128]` and K as
`[1, 8, T, 128]` for this model. Qwen3 eager attention repeats each KV head over
its Q-head group. The capture adapter records the concrete KV-head index used by
each selected Q-head in every trace record. The Rust replay never reconstructs
or infers this mapping.

## Initial stratification

The default capture selects:

- layers: `0,13,27`;
- query heads: `0,5,10,15`;
- query positions: `63,127,255,511`;
- maximum tokenized sample length: 512.

For Qwen3-0.6B this covers beginning/middle/end depth and maps the selected
Q-heads to four distinct KV heads. A sample produces:

`3 layers * 4 Q heads * 4 positions = 48 records`.

This is a first stratified qualification slice, not an exhaustive model survey.
A later expansion may add layers/heads/positions only after the first real-trace
result is understood.

## Input file

The capture tool consumes UTF-8 JSONL. Each non-empty line is exactly one object:

```json
{"sample_id":"sample-0001","text":"..."}
```

`sample_id` values must be unique. Every selected sample must tokenize to at
least 512 tokens because position 511 is part of the default slice.

The capture sidecar records the SHA-256 of the entire JSONL file. A qualification
run must use a preserved or independently reproducible corpus file; ad-hoc text
pasted into the command line is not acceptable evidence.

## Smoke versus qualification corpus

Two phases are deliberately separate.

### Adapter smoke

A local throwaway JSONL may be used to prove that:

- the model loads on the target;
- the Qwen3 attention hook is actually reached;
- 48 records per sample are produced;
- the Rust `e4_trace_replay` parser accepts the file;
- dense/candidate correctness checks run.

A smoke file is **not** model-distribution evidence.

### Qualification corpus

The first intended public corpus is WikiText-2 raw validation data from
`Salesforce/wikitext`, frozen to an immutable dataset revision/file hash before
capture. Corpus construction must be deterministic and documented separately.
The goal is natural text with reproducible provenance, not benchmark quality.

## Binary provenance

The capture writes the already-qualified `ADAQK01\0` v1 format and a JSON
sidecar. The sidecar includes:

- model/tokenizer immutable revisions;
- source activation dtype;
- input JSONL SHA-256;
- selected layers/heads/positions;
- record count;
- trace SHA-256;
- Python, Torch and Transformers versions;
- target platform/device;
- relevant model configuration.

The sidecar is supplementary evidence. The Rust parser continues to treat the
binary header as the normative trace contract.

## Qualification sequence

1. Qualify the Python adapter source mechanically (`py_compile` plus repository
   gates).
2. Commit that exact source.
3. Perform one adapter smoke on the target.
4. Replay the smoke with the Rust E4 tool.
5. Freeze a deterministic WikiText qualification corpus and its SHA-256.
6. Capture Q/K from the immutable Qwen3 revision.
7. Replay all trace records with Flat / Contiguous / Content-aware candidates.
8. Preserve trace metadata, replay output and hashes.
9. Only then promote the E4 real-model status.

## Non-claims

The capture/replay experiment does not establish:

- GPU speedup;
- physical KV-memory traffic savings;
- production-safe floating-point directed rounding;
- generalization beyond the captured model/corpus/layers/heads/positions;
- novelty of hierarchical or MIPS bounds.
