# E5c geometry survey + E4 dispatch replay — natural trace provenance

Date: 2026-08-26 (UTC). Host: Jetson Thor, CPU capture (eager attention).

## Trace artifact

- File: `qwen3-e4-natural.adaqk` (94,810,585 bytes) — **not committed**;
  regenerate with the procedure below.
- SHA-256: `9e7eda5dd8cd513342c97af734a2770bc0e863b03fcf6723d8a54dc3a7c928d1`
- Format: ADAQK01 v1; 768 records (16 samples x 48), head_dim 128.
- Sidecar: `capture-metadata.json` (committed).

## Reconstruction procedure

1. Frozen parquet downloaded at the immutable dataset revision
   `b08601e04326c79dfdd32d625aee71d232d685c3`
   (`wikitext-2-raw-v1/validation-00000-of-00001.parquet`);
   SHA-256 verified:
   `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`.
2. JSONL rebuilt with the documented selection rule (16 strata, first
   non-empty row onward, concatenated non-empty cells until truncation).
   Verified: every sample tokenizes to exactly 512 Qwen3 tokens at revision
   `c1899de289a04d12100db370d81485cdf75e47ca`.
   - Rebuilt JSONL SHA-256:
     `7372e74eb2a1fd17be42134b95c433e48c931238234fe162931837bf666120aa`
   - Historical frozen identity:
     `8b3cb29d52850020134bd37c1e58dac2ff79508144db49f87ca2801e4e0b4bb0`
   The byte-level serialization of the historical file (id naming, JSON
   separators, trailing newline) is not recorded in the protocol docs and a
   768-variant search did not reproduce it. The selection RULE is verified
   exactly; only cosmetic serialization may differ. Token stream equivalence
   was checked per sample (512/512).
3. Capture: `tools/capture_qwen3_e4.py --device cpu` on
   `Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca`,
   default stratification (layers 0,13,27; q-heads 0,5,10,15; positions
   63,127,255,511).

## Campaign results (page 16, leaf 8)

### e4_dispatch_replay (selector -> controller end-to-end)

| alpha | plans chosen        | worst abs O | worst abs tau |
| ----- | ------------------- | ----------: | ------------: |
| 2.0   | 384 dense / 384 CA  |      0.0    |         0.0   |
| 1.5   | 384 dense / 384 CA  |   5.0e-16   |       8.9e-16 |

### e5c_geometry_survey (legacy vs PCA/shrunk-ball, both exact on 768/768)

| alpha | legacy pruned | v2 pruned | delta   | records where v2 wins |
| ----- | ------------: | --------: | ------: | --------------------: |
| 2.0   |        17,280 |    19,184 | +1,904  |               116/768 |
| 1.5   |        10,712 |    12,256 | +1,544  |     (see survey files) |

Conclusion: the v2 geometry strictly dominates the historical one on this
natural slice (never loses a case, gains ~11-14% additional token pruning),
answering the E4-era observation that the old mean-ball won 0/18,432 cases.
