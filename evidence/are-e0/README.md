# ARE-E0 online-softmax discovery evidence

## Result

The committed-source E0 run is a **bounded negative result**. The automated
grammar enumerator and seeded evolutionary proposer did not rediscover a
candidate that survived the six-case probe corpus. Nothing reached the oracle,
adversarial holdout, structural-cost qualification, or Pareto archive gates.

This is not evidence that the target recurrence is absent from the grammar,
and it is not a proof that a larger or different search would fail.

## Frozen run

- engine source revision: `eb2222e740cfe9cd807bd4743817b5abaa3d1752`
- archive schema: `3`
- seed: `20260822`
- experiment ID: `0d1d0528da631ae827e1d20e2259680ebdb49b947c0ecb6d9d1271e04161facf`
- archive digest: `627c273f300798ab66109fb175c8d37c91602a30cbd7fb2a55b08599db66f565`
- archive file SHA-256: `31ee68ff30072674e3a19d5005883061260665e87bd2511707d0fbb75d17ca5f`
- archive: `are-e0-online-softmax-seed-20260822.json`
- target environment recorded by the archive: `aarch64-linux`

Configuration:

- 10,000 raw-proposal cap;
- 5,000 canonical-unique evaluation cap;
- enumerator with at most five scalar nodes and 4,000 possible emissions;
- evolutionary seed `3790090774`, population 128, 64-generation cap, and
  eight mutation retries;
- grammar maximum 30 aggregate nodes and depth 10;
- 96 discovery, 6 probe, 686 oracle, and 106 adversarial-holdout cases;
- fixed relative-error tolerance `1e-9` at every numerical gate.

## Coverage and negative evidence

| Measure | Count |
| --- | ---: |
| Raw proposals | 6,199 |
| Canonical unique | 5,000 |
| Canonical duplicates | 1,199 |
| Static rejections | 0 |
| Probe mismatches | 4,929 |
| Probe non-finite executions | 71 |
| Probe survivors | 0 |
| Oracle survivors | 0 |
| Adversarial survivors | 0 |
| Pareto members | 0 |

The round-robin stream contained 3,100 enumerative proposals (3,036 unique)
and 3,099 evolutionary proposals (1,964 unique). Search stopped at the declared
canonical-candidate budget. Every unique candidate is retained in the archive;
no failed candidate was silently dropped.

The deterministic best-ranked failed candidate had digest
`b9a1d83d6dcaab7a539e59d26ec2ac2bbcd3495d79e0b2e35aa664138e1b859b`,
discovery loss `3.66071127893173340e1`, and was falsified by a probe mismatch.
It is evidence about the search trajectory, not a surviving algorithm.

## Replay

The experiment was executed twice with identical arguments. The second run
used `--replay evidence/are-e0/are-e0-online-softmax-seed-20260822.json` and
reported `replay_identical=true`. The two JSON files were byte-identical
(`cmp` exit 0) and both had file SHA-256
`31ee68ff30072674e3a19d5005883061260665e87bd2511707d0fbb75d17ca5f`.
Elapsed time was observed but is absent from fitness and archive identity.

## Interpretation

The reference recurrence is expressible in the declared grammar, as the
test-only oracle fixture demonstrates, but its unshared tree form needs 18
nodes and repeats the same maximum subexpression. The five-node enumerator
cannot cover that structure, while the bounded genetic-programming run did not
assemble it. The least target-specific next expansion is a preregistered,
typed DAG/register form that permits general subexpression sharing or
references to earlier ordered outputs. Such an expansion must remain grammar
driven and pass the same leakage and evidence gates.

