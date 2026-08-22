# ARE-E0 — ADA Research Engine E0

Status: implemented research infrastructure; human promotion remains external.

## Claim boundary

ARE-E0 automates a bounded loop:

`PROPOSE → STATIC VALIDATE → FALSIFY → ORACLE CHECK → ADVERSARIAL CHECK → COST → PARETO → ARCHIVE`

It can falsify a candidate with a finite counterexample. A candidate that passes
all declared cases is recorded as `SurvivedDeclaredGatesWithinTolerance` or
`SurvivedDeclaredGatesExactly`. Neither state means mathematical proof,
production readiness, novelty, or promotion. The engine has no promotion API.

Hardware timing is not a fitness objective. ARE-E0 records deterministic
structural costs; physical benchmark evidence is a later, separate gate.

## Architectural position

- ElasticSoftMax and other semantic controllers decide **what** attention
  semantics are desired.
- ADA discovers, falsifies, qualifies, ranks, and archives candidate
  algorithms implementing declared semantics.
- ElasticXxx chooses **which already-qualified** execution plan fits current
  hardware and state.
- FLAT-ATTENTION receives only mechanisms that have completed the required
  external qualification and promotion path.

ARE-E0 does not modify FLAT-ATTENTION, SciRust, or ADA's existing qualified
mechanisms.

## Crate layout

The implementation is additive in `crates/ada-research-engine`:

| Module | Responsibility |
| --- | --- |
| `candidate`, `expr`, `grammar` | Restricted recurrence IR and static contract |
| `canon`, `digest_writer`, `float_serde` | Conservative normalization, stable identity, exact archive transport |
| `proposer`, `proposers/*` | Pluggable proposal-only boundary, enumeration, seeded evolution, manual test ingress |
| `problem`, `corpus` | Budgets, tolerances, and opaque evidence case sets |
| `engine`, `gates` | Ordered evaluation, falsification, finalization, and epistemic states |
| `cost`, `pareto` | Structural objective vector and nondominated ranking |
| `archive` | Manifest, per-proposal ledger, counterexamples, integrity, and replay comparison |
| `online_softmax` | Independent E0 oracle and partitioned challenge corpora |
| `benchmark` | Deliberately empty downstream physical-evidence boundary |

## Core contracts

`ResearchProblem` owns the grammar, seed, counter budgets, tolerances, and four
opaque corpora. Corpus fields are private. A proposer receives only a
`GrammarSpec`, `SearchBudget`, and loss feedback for its own earlier emissions.

`Candidate` is an ordered output tuple. Output order participates in identity.
For E0, a candidate receives `[m_old, l_old, score]` and emits
`[m_new, l_new]`; future state is never supplied as an input.

`CandidateProposer` returns data, never a verdict. Built-in deterministic
enumeration and seeded genetic programming are real grammar procedures, not
handwritten candidate vectors. Manual and future external proposers enter the
same validation path and have no gate bypass.

`SearchBudget` bounds raw proposals, canonical-unique evaluations, AST size and
depth, final gate evaluations, oracle and holdout case counts, generations,
mutation retries, retained counterexamples, and archive hall capacity. Search
termination never depends on wall-clock time.

`CandidateEvaluation` records proposal index, canonical form, digest,
provenance, discovery loss, structural cost, every gate disposition, final
empirical class, and Pareto membership. Static and procedural failures are
`Rejected`; numerical counterexamples are `Falsified`.

## Candidate language and interpreter

E0 permits only:

- finite declared constants;
- indexed input variables;
- `Add`, `Sub`, `Mul`, `Exp`, and symmetric `Max`;
- a fixed ordered output tuple.

It has no division, loops, arbitrary memory, branching, filesystem, shell,
callbacks, FFI, dynamic source execution, or unsafe code. The crate uses
`#![forbid(unsafe_code)]`.

Evaluation is recursive in explicit left-to-right tree order. Every
intermediate must be finite. NaN or infinity produces a precise execution
failure at the active gate. The E0 `Max` tie rule returns positive zero for a
signed-zero tie, making commutative child ordering bit-stable.

Static validation occurs on the raw candidate before normalization. It checks
output arity, aggregate node count, maximum depth, variable indices, declared
constant bit patterns, and enabled operators. Therefore simplification cannot
erase an illegal or over-budget raw subtree.

## Canonical identity

Canonical text is an ordered candidate S-expression. Constants use exact
64-bit hexadecimal encodings. Candidate IDs are SHA-256 over a versioned domain
and the normalized candidate text.

Normalization is deliberately narrow:

- exact finite constant folding;
- `x * 1 → x`, `x - +0 → x`, and `max(x, x) → x`;
- stable child order for finite `Add`, `Mul`, and symmetric `Max`.

There is no associativity rewrite and no `x + 0 → x` rewrite because signed
zero makes the latter unsound. ARE-E0 also deliberately does not rewrite
`x - x → +0`: an overflowing `x` must remain an execution failure rather than
being hidden by normalization.

A digest collision is never structural equality. Deduplication uses a digest
bucket followed by the complete canonical string. A synthetic-collision unit
test exercises that rule.

## Proposal mechanisms

### Deterministic enumeration

The enumerator constructs scalar expressions level by level from grammar
leaves and enabled operators, normalizes/deduplicates those forms, and performs
a bounded generic Cartesian composition into the declared output arity. It
contains no online-softmax variable names, target AST, or oracle call.

### Seeded evolutionary search

The evolutionary proposer uses a local fixed `SplitMix64` stream, random
grammar growth, tournament selection, elitism, subtree crossover, and bounded
mutation. It evolves the complete ordered output tuple. Its only fitness signal
is the engine-assigned discovery loss for candidates it previously emitted.
Configuration is recorded exactly, including probability bit patterns.

The engine caps proposer configuration with manifest-level generation and
mutation budgets. A separate raw-generation limit guarantees termination even
for an external proposer that emits the same duplicate forever.

## Leakage controls

The challenge uses three evidence partitions:

1. discovery cases affect only engine-assigned proposal feedback;
2. oracle cases are evaluated after cheap probe survival;
3. adversarial holdout cases are touched only after proposal search ends and an
   oracle candidate survives.

The explicit oracle grid avoids exact discovery-grid duplicates. Holdout uses
unseen extremes and multi-step repeated, alternating, monotone, and seeded-walk
trajectories. Corpus roles must be non-empty and distinct.

The proposer trait mentions neither `ProblemCorpus` nor expected outputs.
Built-in proposer modules do not import `online_softmax` or `corpus`. A tracked
holdout test asserts that all proposal calls finish before the first holdout
evaluation. Source-structure tests reject built-in proposer dependencies on
the oracle or holdout builders.

The trusted recurrence appears only in the evaluation module and a test-only
known-valid fixture. It is not a generator seed, candidate list, AST pattern,
rewrite, or ranking input. Candidate ranking sees discovery loss and structural
cost, not hidden expected formulas.

The E0 oracle implements its maximum rule independently rather than calling
the candidate interpreter's `Max` helper. Candidate and oracle trajectories
advance separate state, so a locally plausible but unstable recurrence is
exposed by rollout drift.

## Gate order

1. Raw static validation.
2. Canonical deduplication with collision-safe equality.
3. Discovery loss for proposal feedback (not a qualification verdict).
4. Six-case probe falsification.
5. Post-search deterministic ranking of probe survivors.
6. Oracle evaluation, capped by the manifest.
7. Adversarial holdout evaluation, never during proposal.
8. Optional structural-cost budget.
9. Pareto extraction over candidates that survived every configured evidence
   gate.

Incorrect candidates do not participate in the qualified Pareto set.

## Structural cost and Pareto semantics

`CostVector` records total operators, `Exp`, `Max`, `Mul`, combined `Add/Sub`,
tree depth, output-state count, and a deterministic temporary upper bound.
There is no wall-clock measurement.

The Pareto objective vector is discovery error plus every structural cost
dimension. Dominance is component-wise with at least one strict improvement.
A deterministic total order—loss, cost vector, canonical text—is used only for
stable ordering within and across fronts; it is not presented as a scientific
magic score.

## Archive and replay semantics

Archive schema v2 contains:

- engine/schema version and committed source revision;
- declared numeric semantics and target architecture/OS;
- experiment ID, seed, grammar digest, all counter budgets, and tolerances;
- role, case count, and content digest for every corpus;
- full proposer descriptors and configuration digests;
- search termination and aggregate gate accounting;
- a proposal-trajectory digest committing to raw emission order;
- every generated proposal's canonical/raw-invalid representation,
  provenance, gate history, rejection or counterexample, and final class;
- structural costs, nondominated set, hall of fame, and best-evidence record;
- bounded per-gate counterexamples and rejection counts;
- an explicit final experiment outcome;
- an unkeyed SHA-256 integrity digest over a versioned binary encoding of all
  preceding fields.

Archive JSON transports every evidence `f64` as a 16-digit hexadecimal bit
string. This avoids one-ULP changes from decimal parser rounding. The digest is
integrity evidence, not authentication; a party able to edit an archive and
recompute its digest is outside this threat model.

`ExperimentArchive::from_json` verifies schema and digest. Replay reruns the
experiment from the manifest-equivalent configuration and compares all fields,
including proposal order and the complete evaluation ledger. Same-seed tests
also compare pretty JSON byte for byte.

The logical determinism contract is strongest on the recorded target and
numeric environment. IEEE-754 operations use ordered evaluation, but Rust's
platform `exp` implementation is not claimed bit-identical across every
libm/architecture. The manifest makes that limitation explicit rather than
silently claiming universal cross-platform bit identity.

## First discovery challenge

The target behavior is the stable online-softmax state transition. The target
formula is intentionally omitted from this discovery section; it is documented
only in the independent oracle source and mission specification. The grammar
knows three input names, two output names, no constants, and the five general
operators.

Declared case sets:

| Partition | Cases | Purpose |
| --- | ---: | --- |
| Discovery | 96 | Structured regimes plus seeded finite jitter |
| Probe | 6 | Cheapest regime-spanning counterexamples |
| Oracle | 686 | Independent unseen single-step grid |
| Adversarial holdout | 106 | 96 extreme steps plus 10 multi-step trajectories |

Fixed relative-error tolerance is `1e-9` at probe, oracle, and holdout. Oracle
and holdout case counts are fixed before the run; tolerance is never relaxed in
response to results.

The curated E0 evidence run uses:

- seed `20260822`;
- 5,000 canonical-candidate evaluations;
- 10,000 raw-proposal cap;
- enumerator: at most five scalar nodes, 4,000 emissions;
- evolutionary proposer seed `3790090774`, population 128, at most 64
  generations and eight mutation retries;
- 64 final gate evaluations;
- grammar maximum 30 aggregate nodes and depth 10.

The evidence result and exact committed source revision are added to this
document only after the source commit and two byte-identical runs. Until that
entry exists, no successful rediscovery claim is authorized.

<!-- ARE_E0_RESULT_BEGIN -->
Result: pending committed-source evidence run.
<!-- ARE_E0_RESULT_END -->

## Runner

Example:

```bash
cargo run --release -p ada-research-engine \
  --example discover_online_softmax -- \
  --seed 20260822 \
  --budget 5000 \
  --generated-budget 10000 \
  --population 128 \
  --generations 64 \
  --source-revision <committed-sha> \
  --archive /tmp/are-e0-run-a.json
```

Pass `--replay /tmp/are-e0-run-a.json` on the second identical run. The summary
prints `experiment_id`, `seed`, `generated`, `canonical_unique`,
`rejected_static`, `falsified`, `survived_oracle`, `survived_adversarial`,
`pareto_count`, `best_candidate_digest`, `status`, termination, and archive
digest. Elapsed time is stderr-only evidence and never affects fitness or the
archive.

## External intelligence and benchmarking

A future SciRust, symbolic, LLM, Muse, Ox, human, or SciAgent adapter implements
only `CandidateProposer`. It cannot set gate results or insert directly into
the qualified archive. ADA remains the evaluator.

`BenchmarkEvidenceAvailable` is reserved for a separately supplied physical
artifact. ARE-E0 cannot construct that state from structural cost or elapsed
runner time. No speedup claim follows from this experiment.

## Known limitations

- This is bounded empirical search, not theorem proving.
- The grammar has scalar trees and duplicated subexpressions, not DAG sharing,
  output-to-output dependencies, tensors, loops, or learned constants.
- Genetic programming is a baseline, not a claim of search optimality.
- Full per-proposal ledgers make large JSON archives intentionally verbose.
- Checkpoint resume is not implemented; deterministic replay is implemented.
- Cross-platform `exp` bit identity is not guaranteed.
- Archive SHA-256 is unkeyed integrity, not provenance authentication.
- No hardware benchmark, prior-art review, or promotion is performed.

## Hostile self-review record

The pre-evidence review treated the implementation as adversarial, not merely
as a passing test suite:

| Question | Finding and action |
| --- | --- |
| Is the target secretly seeded? | No built-in proposer imports the challenge module or contains its variable names. The inherited supplied-`m_new` formulation was rejected and preserved only as legacy evidence. |
| Is search automatic? | Yes. Production runs use grammar enumeration and seeded tuple-level genetic programming. The only fixed-candidate source is labeled `ManualList` and is used by tests/external submissions, not the E0 runner. |
| Can oracle or holdout leak into proposal? | Proposer context contains grammar, budgets, and own prior loss only. A tracked test proves holdout evaluation begins after all proposal calls finish. |
| Is holdout distinct? | Yes at the interface and role/digest level. Explicit oracle grid points avoid exact discovery-grid duplicates; holdout uses unseen ranges and trajectories. |
| Is determinism real? | Sequential ordering, fixed RNG, counter stopping, ordered maps/sorts, trajectory digest, same-seed archive equality, and two-run CLI replay are tested. Elapsed time is excluded. |
| Is archive identity stable? | An initial decimal-JSON replay changed one error by one ULP and was rejected by the digest. Schema v2 now transports every evidence float by exact bits and has a real-E0 round-trip regression test. |
| Can a digest collision imply equality? | No. Dedup checks the full canonical form inside each digest bucket; a synthetic-collision test covers this. |
| Are floating rewrites sound? | `x + 0` was removed for signed-zero semantics. Hostile review also removed `x - x → 0`, which could hide an overflowing intermediate. |
| Can malformed candidates escape? | Raw arity/resources/operators/variables/constants are validated before normalization. A regression test uses an oversized tree that would otherwise collapse. |
| Can external sources bypass gates? | No engine API accepts an external verdict. Every proposer emits only candidate data and provenance. Archive hashes are integrity rather than authentication, which is documented. |
| Are claims too strong? | Outcome names describe finite-case survival only; no `Proven` state, speedup claim, novelty claim, or automatic promotion exists. |
