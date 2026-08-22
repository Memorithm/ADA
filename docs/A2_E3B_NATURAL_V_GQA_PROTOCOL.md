# ADA-A2 E3b — Natural V / GQA Protocol

## Objective

E3b extends the qualified A2 E3a natural Q/K GQA accounting
experiment with real Qwen3 value tensors.

The experiment keeps the frozen `ADAQK01` Q/K format unchanged and
introduces a separate companion value trace:

`ADAV01`

This separation is deliberate.

`ADAQK01` records query-specific Q/K score inputs.

`ADAV01` records the unique physical KV-head value representation
before Transformers repeats a KV head across its grouped query heads.

## Frozen model

Model:

`Qwen/Qwen3-0.6B`

Immutable model/tokenizer revision:

`c1899de289a04d12100db370d81485cdf75e47ca`

Natural corpus:

the same frozen 16-sample WikiText validation slice used by A5 E4
and A2 E3a.

Sample JSONL SHA-256:

`8b3cb29d52850020134bd37c1e58dac2ff79508144db49f87ca2801e4e0b4bb0`

## Value tensor stage

The installed Qwen3 attention path computes:

1. `query_states = q_norm(q_proj(...))`
2. `key_states = k_norm(k_proj(...))`
3. `value_states = v_proj(...)`
4. RoPE is applied to Q/K only.
5. The attention implementation receives Q/K/V.
6. Eager attention performs `repeat_kv` on K and V.

`ADAV01` observes V at step 5:

`attention_value_input_pre_repeat_kv`

For the frozen Qwen3-0.6B configuration:

- query heads: 16;
- physical KV heads: 8;
- Q heads per KV head: 2;
- head dimension: 128;
- sequence length: 512.

The captured V tensor at each selected layer has shape:

`[1, 8, 512, 128]`

and source dtype `bfloat16`.

## Record identity

One `ADAV01` record is identified by:

`(sample_id, layer_index, kv_head_index)`

The frozen trace therefore contains:

`16 samples * 3 layers * 8 KV heads = 384 records`.

Each record stores the complete 512 x 128 V matrix for that physical
KV head exactly once.

Query positions 63, 127, 255, and 511 use prefixes of the same
record rather than duplicating V per query head or position.

## Binary format

Magic:

`ADAV01\0`

Version:

`1`

Header fields:

- model id;
- model revision;
- tokenizer id;
- tokenizer revision;
- capture id;
- source dtype;
- tensor stage;
- record count.

Each record stores:

- sample id;
- layer index (`u32`);
- KV head index (`u32`);
- value start position (`u64`);
- value row count (`u32`);
- head dimension (`u32`);
- row-major V tensor as little-endian IEEE-754 `f32`.

The capture converts source `bfloat16` V values to stored `f32`
without changing their numeric values.

All serialized values must be finite.

## Determinism prototype

Before committing the permanent format, two completely separate model
loads and captures produced byte-identical traces.

Canonical prototype trace SHA-256:

`d007b56a7c4588568bb9211c58250d8b214d8eb31ee2483eb0d1b033ffac11c5`

Canonical prototype metadata SHA-256:

`9b0c9a362023203bb0660590bd9556f4be340027104cbd78722fdbcde5ab05a7`

Prototype trace size:

`100686186` bytes.

The independent binary-contract parser confirmed:

- 384 records;
- 16 samples;
- 384 unique record identities;
- layers 0, 13, 27;
- 48 records per KV-head index;
- 512 V rows per record;
- head dimension 128;
- zero trailing bytes;
- metadata/trace SHA agreement.

## E3b numerical experiment

The next E3b stage joins:

`ADAQK01 + ADAV01`

by:

`(sample_id, layer_index, kv_head_index)`.

For every natural GQA pair and query position, E3b will evaluate
the exact sparse Entmax distributions for both Q heads against the
same unique V matrix.

It will distinguish three logical V sets:

1. `FullDenseV`
2. `A5-KLoaded-V`
3. `A2-Support-V`

For the two Q heads sharing one KV head:

`K_union = K_q0 union K_q1`

and

`S_union = S_q0 union S_q1`.

The exact output for each individual Q head remains:

`O_q = sum_i p_q(i) V_i`.

The union determines which unique V rows need to be resident/loaded
for the pair; each Q head still uses its own probabilities.

## Claim boundaries

This protocol does not by itself establish:

- physical cache-line traffic;
- DRAM/HBM bytes;
- CPU/GPU speedup;
- end-to-end attention speedup;
- production memory scheduling;
- model-quality effects;
- novelty.

The first E3b milestone is exact natural-V numerical qualification.

Wall-clock and physical-traffic studies are separate later gates.

## Qualified E3b result

Classification:

`A2-E3B-NATURAL-GQA-V-OUTPUT-CORRECTNESS-QUALIFIED`

The frozen all-head `ADAQK01` corpus and the natural `ADAV01`
corpus were joined by:

`(sample_id, layer_index, kv_head_index)`.

The experiment evaluated:

- 1,536 natural GQA groups;
- 3,072 group/alpha cases;
- 3,072 query-head records;
- 6,144 query-head/alpha output cases.

The experiment reproduced the complete E3a unique-row accounting
exactly.

### Alpha 1.5

- unique A5 K-loaded V rows: 137,098;
- unique final-support V rows: 8,555;
- residual A2 V-row avoidance after A5: 0.937599381;
- total unique-V-row avoidance: 0.976793077;
- groups without residual A2 opportunity: 6 / 1,536.

Maximum dense-vs-priority probability difference:

`3.44169137633798528e-15`

Maximum tau difference:

`1.77635683940025046e-15`

Maximum `FullDenseV` versus `A5-KLoadedV` output L-infinity error:

`3.37507799486047588e-14`

Maximum `FullDenseV` versus `A2-SupportV` output L-infinity error:

`3.37507799486047588e-14`

Maximum `A5-KLoadedV` versus `A2-SupportV` output L-infinity error:

`0`

### Alpha 2.0

- unique A5 K-loaded V rows: 110,536;
- unique final-support V rows: 4,466;
- residual A2 V-row avoidance after A5: 0.959596873;
- total unique-V-row avoidance: 0.987885200;
- groups without residual A2 opportunity: 13 / 1,536.

All observed probability, tau, and V-output differences were exactly zero.

## Interpretation of the V-output result

For each Q head, the exact output is

`O_q = sum_i p_q(i) V_i`.

The GQA union is used only to account for which unique rows of the
shared physical KV head would need to be available.

It does not combine or alter the two query-head probability
distributions.

The exact zero difference between `A5-KLoadedV` and
`A2-SupportV` is the direct A2 property:

rows loaded by A5 but outside the final exact Entmax support have
probability exactly zero and therefore make exactly zero contribution
to the output.

The small alpha-1.5 `FullDenseV` difference comes from the already
measured f64 dense-versus-priority Entmax numerical difference, not from
dropping zero-probability V rows.

## Qualified claim

On the frozen Qwen3-0.6B natural Q/K/V slice, exact sparse Entmax
K-first/V-late execution preserves the attention-head value output
when V rows outside final support are omitted.

After actual GQA unique-row accounting, 93.7599381% (alpha 1.5) and
95.9596873% (alpha 2.0) of the unique V rows remaining after A5 are
logically avoidable.

## Non-claims

This qualification is not evidence of:

- measured physical V traffic;
- cache-line traffic;
- DRAM/HBM byte reduction;
- memory-bandwidth reduction;
- CPU speedup;
- GPU speedup;
- end-to-end attention speedup;
- production scheduling behavior;
- model-quality preservation outside the tested exact attention
  outputs;
- algorithmic novelty.
