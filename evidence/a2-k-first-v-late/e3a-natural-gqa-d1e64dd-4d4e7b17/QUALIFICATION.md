# ADA-A2 E3a Qualification

Classification:

`A2-E3A-NATURAL-GQA-UNIQUE-V-ROW-ACCOUNTING-QUALIFIED`

Base ADA commit:

`d1e64dd767031dc35d9faa73fc1fa7b9c760d364`

All-head trace SHA-256:

`4d4e7b175bc0711f0acb15e3891eacedb2aaada3e11c896e1483123e0cc10ca8`

Historical four-head trace SHA-256:

`d205e242d781c56799565a41abaad2d36d991f29519578f7c7c2bbb477bc8c49`

Historical/new selected-record common stream SHA-256:

`a1798f860e31451f27928215a1bcd9d91c8c4f542a1f5cf78212ee14ffcb53dd`

Replay SHA-256:

`64352e29f5fd6beabf3f860481fd4ca1de868bcf59fd6a84b7936249538ae935`

## Qualification result

The 3,072 all-head records formed exactly 1,536 GQA groups and 3,072
group/alpha cases.

For alpha 1.5:

- unique K rows: 137,098;
- unique support rows: 8,555;
- residual A2 V-row avoidance after A5: 0.937599381;
- total unique-V-row avoidance: approximately 0.976793;
- GQA delta versus naive per-Q accounting: -0.709357 percentage points;
- no-residual groups: 6 / 1,536.

For alpha 2.0:

- unique K rows: 110,536;
- unique support rows: 4,466;
- residual A2 V-row avoidance after A5: 0.959596873;
- total unique-V-row avoidance: approximately 0.987885;
- GQA delta versus naive per-Q accounting: -0.226990 percentage points;
- no-residual groups: 13 / 1,536.

All no-residual cases occurred in layer 27 after A5 had already reduced
the unique K-loaded set to 2 or 4 rows.

No support-union containment violation or union/intersection cardinality
failure was observed.

Maximum numerical differences were 3.442e-15 probability and 1.776e-15
tau for alpha 1.5, and zero for alpha 2.0.

## Scope

This qualification establishes natural Qwen3 GQA-aware logical unique-V-row
accounting.

It does not establish physical memory traffic, bandwidth, cache behavior,
wall-clock speedup, GPU viability, end-to-end speedup, production readiness,
model quality, or novelty.
