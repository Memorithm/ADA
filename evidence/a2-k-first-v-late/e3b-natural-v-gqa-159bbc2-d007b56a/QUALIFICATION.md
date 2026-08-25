# ADA-A2 E3b Qualification

Classification:

`A2-E3B-NATURAL-GQA-V-OUTPUT-CORRECTNESS-QUALIFIED`

Base ADA commit:

`159bbc27c49fc922ae06ab3040ff55ca57b4052c`

## Frozen inputs

All-head Q/K trace SHA-256:

`4d4e7b175bc0711f0acb15e3891eacedb2aaada3e11c896e1483123e0cc10ca8`

Natural pre-repeat-GQA V trace SHA-256:

`d007b56a7c4588568bb9211c58250d8b214d8eb31ee2483eb0d1b033ffac11c5`

V capture metadata SHA-256:

`9b0c9a362023203bb0660590bd9556f4be340027104cbd78722fdbcde5ab05a7`

Join replay SHA-256:

`e2b5fc270bc8eb5a5e17f9a371d3f39ed47fc6afac25fe99416f83901c32e144`

## Research-source identities

E3b runner SHA-256:

`a9a5dad8ec8fdab383a87af76bf96fae6ad83a936775a0b7ebabde7265b75b89`

Qwen3 V capture source SHA-256:

`68b02a51dfab28c8fbe3b08f1aeda45e172806cc8f25cf06ad38d7c5860dd658`

E3b analyzer SHA-256:

`7ef7c4530dfba6a0da4bb5e05b267aec59afaabd4a29bd9383d87038b818fffa`

Permanent capture log SHA-256:

`1eee422fc0720a4352637d783c783a3ab83eb60454fd3e45b5aeba23f8dbd774`

Rust parser log SHA-256:

`b48666bc4d727d7cae8b21e30306267341551bc1cac08a0b3d776e1134c4526a`

## Structural result

- Q/K records: 3,072;
- unique V records: 384;
- natural GQA groups: 1,536;
- group/alpha cases: 3,072;
- query-head/alpha cases: 6,144;
- page size: 16;
- hierarchy leaf size: 2.

## Alpha 1.5

- A5 unique K-loaded rows: 137,098;
- final unique support rows: 8,555;
- residual A2 V avoidance after A5: 0.937599381;
- total unique-V avoidance: 0.976793077;
- no-residual groups: 6;
- max probability difference: 3.44169137633798528e-15;
- max tau difference: 1.77635683940025046e-15;
- max FullDenseV vs A5-KLoadedV L-infinity:
  3.37507799486047588e-14;
- max FullDenseV vs A2-SupportV L-infinity:
  3.37507799486047588e-14;
- max A5-KLoadedV vs A2-SupportV L-infinity: 0.

## Alpha 2.0

- A5 unique K-loaded rows: 110,536;
- final unique support rows: 4,466;
- residual A2 V avoidance after A5: 0.959596873;
- total unique-V avoidance: 0.987885200;
- no-residual groups: 13;
- all observed probability, tau and V-output differences: 0.

## Exact A2 result

Across all 6,144 query-head/alpha cases:

`A5-KLoadedV == A2-SupportV`

at the recorded f64 output level.

This directly confirms that V rows loaded during A5 search but lying
outside final exact Entmax support contribute zero to the attention
value output.

## Scope

This qualification establishes exact natural Qwen3 V-output
correctness under GQA-aware logical row omission.

It does not establish physical traffic reduction, cache behavior,
bandwidth reduction, wall-clock speedup, GPU performance, end-to-end
speedup, production readiness, broad model-quality equivalence, or
novelty.
