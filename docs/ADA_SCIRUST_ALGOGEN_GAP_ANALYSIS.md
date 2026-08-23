# ADA / SciRust Algorithm-Generation Gap Analysis

## Scope and inspection basis

This analysis is read-only. No SciRust file was modified.

Relevant clean paths were inspected in the SciRust checkout at committed HEAD
`e096b474b8d95e271abb98c53773a473f27d829d`. The wider SciRust worktree had
unrelated changes, but the inspected `scirust-algogen`, `scirust-symbolic`,
`scirust-symreg`, `scirust-solvers`, and relevant `scirust-core` paths were
clean relative to that HEAD.

The classification vocabulary is:

- **ADOPT** — a concept or implementation pattern is usable now without
  changing ADA's trust boundary;
- **ADAPT** — useful after an explicit adapter or semantic hardening;
- **MISSING** — capability required by ADA recurrence discovery is absent;
- **INAPPROPRIATE FOR THIS LAYER** — useful elsewhere, but not inside the
  candidate proposal/evidence boundary.

ARE-E0 intentionally has no path dependency on `/root/scirust`. Cross-repository
absolute paths would make replay depend on ambient checkout state.

## Summary matrix

| SciRust area | Classification | ADA decision |
| --- | --- | --- |
| algogen deterministic RNG and counter-bounded evolution | ADOPT (pattern) | Reimplemented locally with a versioned proposer descriptor |
| algogen verifier/interpreter separation | ADOPT (architecture) | Mirrored in raw validation before safe interpretation |
| algogen canonical bytes and collision-safe equality | ADOPT (architecture) | Canonical string remains authoritative after digest bucketing |
| algogen cost, Pareto ranking, hall/archive/replay | ADOPT / ADAPT | Concepts adopted; ADA-specific gates and evidence schema implemented locally |
| algogen current tensor IR | MISSING for recurrence search | Future adapter only after a stronger stateful scalar/tensor IR exists |
| symbolic expression tree | ADAPT | Possible offline proposer after IEEE-safe semantics and `Max` are added |
| symbolic current simplifier/evaluator | INAPPROPRIATE FOR TRUSTED GATES | Rewrites and evaluation contract do not match ADA finite-intermediate semantics |
| symreg GP and complexity/error Pareto ideas | ADAPT | Useful search ideas; current implementation is not a stable public ADA component |
| symreg current constants/division/function set | INAPPROPRIATE for E0 | Enlarges hazards/search space without need and permits different non-finite behavior |
| solvers | INAPPROPRIATE FOR THIS LAYER | Potential downstream parameter fitting or certificates, not proposal/gate authority |
| core reproducible reductions | ADAPT for future tensor objectives | E0 scalar evaluation already has fixed sequential order |

## `scirust-algogen`

### Immediately reusable ideas — ADOPT

The tensor algogen already establishes several sound research-engine patterns:

- an explicitly seeded local `SplitMix64` generator independent of OS entropy;
- a verifier separate from the interpreter;
- generation, mutation, crossover, population ranking, and bounded evolution;
- deterministic structural costs and fitness reports;
- nondominated/Pareto ordering;
- a capacity-bounded hall of fame;
- canonical program bytes and fingerprints where canonical equality, not the
  fingerprint alone, decides duplicates;
- an archive with explicit schema/digest versions, full provenance, integrity
  verification, and deterministic experiment replay;
- optional parallel evaluation designed to preserve deterministic result order.

ARE-E0 adopts these concepts. Local implementation is justified because ADA
needs a different IR, different floating-point failure semantics, distinct
discovery/oracle/holdout gates, negative-evidence ledgers, and an independent
repository history. This is not a claim that the ideas were invented anew.

### Reusable after adaptation — ADAPT

A future SciRust adapter can map a stronger generated program into ADA's
`Candidate` data boundary. It must provide:

1. a versioned proposer descriptor;
2. a deterministic candidate stream;
3. a total translation into ADA's declared grammar or a static rejection;
4. no reference to ADA corpora or expected outputs;
5. no direct archive or survival verdict insertion.

SciRust fitness, success, and archive labels cannot substitute for ADA gates.
The external engine proposes; ADA revalidates and evaluates.

### Required capability absent today — MISSING

`scirust-algogen/src/tensor/ir.rs` currently provides a linear single-output
tensor program with operations equivalent to:

- `Input`;
- elementwise `Add`;
- `MatMul`;
- `Transpose2d`;
- `Relu`;
- scalar `Scale`.

It has no scalar recurrent state tuple, `Sub`, `Mul` between scalar values,
`Exp`, `Max`, output arity, streaming transition contract, or ADA finite-case
semantics. It therefore cannot naturally express the E0 stable online-softmax
transition. Encoding the target through tensor-shape tricks or target-specific
macros would be leakage, not reuse.

A future general integration needs at least:

- typed scalar and tensor values;
- ordered multi-output/state transitions;
- `Add`, `Sub`, `Mul`, `Exp`, and symmetric `Max`;
- exact constant encoding;
- explicit non-finite behavior;
- aggregate node/depth/resource verification;
- stable cross-version canonical serialization;
- adapter-visible IR/schema versions.

## `scirust-symbolic`

The symbolic tree includes constants, named variables, arithmetic, division,
powers, trigonometric functions, `Exp`, logarithm, square root, and absolute
value. It supports parsing, evaluation, differentiation, and simplification.

### ADAPT

It could eventually act as an offline proposal or transformation source if an
adapter translates a strictly allowed subset into ADA candidates and ADA then
revalidates the result. Differentiation could later help fit declared constants
outside the E0 constant-free challenge.

### MISSING

The current symbolic expression enum has no `Max`, ordered output tuple, state
transition type, candidate resource contract, or ADA archive provenance.

### INAPPROPRIATE FOR TRUSTED GATES

Current operator overloads simplify expressions such as multiplication by zero
and zero divided by an arbitrary denominator. Those are familiar real-number
identities but can hide NaN, infinity, division-by-zero, or intermediate
execution failure under ADA's declared floating-point semantics. Evaluation
uses named-variable maps rather than the restricted indexed environment.

Consequently ADA must not use the current simplifier as its canonical identity
or its trusted oracle/interpreter path. Any symbolic proposal must cross the
ordinary ADA static and numerical gates.

## `scirust-symreg`

The current symbolic-regression crate contains a compact genetic-programming
implementation with subtree replacement, mutation/crossover, error/complexity
tradeoffs, fitted constants, and a Pareto concept.

### ADAPT

The broad search ideas are relevant to richer ADA proposal sources. A mature
public API could provide candidate syntax trees behind the adapter described
above.

### MISSING / INAPPROPRIATE FOR E0

- Most implementation pieces are crate-private rather than a stable adapter
  API.
- The local RNG uses modulo reduction for index selection rather than the
  algogen multiply-shift pattern.
- The language includes arbitrary constants, division, powers, trigonometric
  functions, logarithm, square root, and absolute value but lacks `Max` and a
  recurrent output tuple.
- Non-finite and domain-error behavior is not ADA's fail-closed interpreter
  contract.
- It has no discovery/oracle/holdout separation or ADA evidence ledger.

Directly importing it into E0 would enlarge the grammar and trust surface
without increasing the target's legitimate expressibility.

## `scirust-solvers`

SciRust solvers cover numerical equations, optimization, quadrature, ODEs, and
combinatorial methods. They are not a recurrence grammar, safe candidate
interpreter, provenance system, or evidence gate.

Classification: **INAPPROPRIATE FOR THIS LAYER** today. A future problem with
declared free constants could use a solver behind a proposal adapter, and a
future proof/certificate layer could use suitable certified routines. Solver
success would still not bypass ADA oracle or adversarial evaluation.

## `scirust-core` reproducibility facilities

SciRust core contains deterministic/certified numerical accumulation patterns
and reproducibility-oriented facilities useful for parallel tensor objectives.

Classification: **ADAPT**. ARE-E0 evaluates scalar expressions and case sets in
a fixed single-threaded order, so importing a large core dependency would add
no present benefit. When ADA adds parallel reductions, an adapter should adopt
canonical partitioning/reduction order and record the reduction policy in the
manifest.

These facilities do not grant candidate code callbacks or runtime authority.

## Adapter trust boundary

The stable ADA-side boundary is deliberately small:

```text
SciRust / symbolic / LLM / human
          |
          | deterministic Candidate + ProposalDescriptor + ProposalSource
          v
ADA raw static validation
  -> canonical dedup
  -> falsification
  -> independent oracle
  -> adversarial holdout
  -> structural cost
  -> Pareto/archive
```

An adapter cannot receive `ProblemCorpus`, gate setters, archive mutation, or a
promotion handle. Translation failures become static rejections with preserved
provenance. Adapter and upstream IR versions must participate in the proposer
descriptor digest.

This boundary lets a future SciRust IR improve without rewriting the ADA
engine: only the proposer adapter changes. Conversely, ADA does not freeze or
fork SciRust's current tensor IR merely to force a premature dependency.

