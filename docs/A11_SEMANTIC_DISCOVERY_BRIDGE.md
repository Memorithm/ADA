# ADA-A11 — Semantic discovery bridge

Status: **A11-E0 architecture / open research**.

ADA historically separates algorithm discovery from semantic attention research.
That separation remains valuable: an optimized recurrence or kernel must not be
allowed to redefine what an attention mechanism means.

A11 adds a narrow bridge between the two layers. ADA may now assign stable
identity to a **semantic hypothesis**, search or qualify implementations of that
hypothesis, and attach external mechanistic evidence without making the evidence
part of the semantic identity.

## 1. The distinction A11 makes explicit

Before A11, most ADA missions can be read as searches for better algorithms or
execution plans under an already-declared attention contract.

For example, two online-softmax recurrences can be different programs while
still implementing the same reference semantic.

The new attention programme also needs to compare genuinely different questions:

```text
implementation candidate
    "how do we compute this semantic?"

semantic candidate
    "what interaction rule should be computed?"
```

These identities must never be interchangeable.

## 2. Identity model

A11-E0 adds `SemanticId`:

```text
SemanticId
├── SemanticFamily
├── stable lowercase name
└── positive revision
```

A semantic identity deliberately contains no:

- implementation name;
- kernel name;
- device identity;
- latency result;
- ITD descriptor;
- TDI recovery profile;
- task score;
- novelty verdict.

Those quantities can change while the semantic under test remains the same.

`ImplementationCandidateId` therefore contains its own implementation name and
revision **plus an explicit `SemanticId` binding**.

Two implementation candidates can point to the same semantic. Conversely, two
semantic hypotheses remain different even when their reference implementations
happen to share an implementation slug such as `reference`.

## 3. Minimal semantic descriptor

`SemanticDescriptor` records only reference-level properties needed at A11-E0:

- semantic identity;
- mask contract;
- state contract;
- weight contract.

The initial family vocabulary is intentionally descriptive rather than
prescriptive:

- `StandardSoftmax`;
- `DifferentialSigned`;
- `ToeplitzStructured`;
- `ProlateConcentration`;
- `GroundStateGreen`;
- `SpectralFlow`;
- `RecurrentMemory`;
- `Hybrid`;
- `Experimental`.

Presence in this enum is **not evidence of usefulness, novelty, implementability
or mathematical validity**. It only gives the research system a stable category
for candidate identity.

A later A11 gate may introduce an executable semantic IR. E0 deliberately does
not extend `ada-ir` yet.

## 4. Mechanistic evidence remains external

A11 introduces `DiagnosticEvidenceRef`, not ITD or TDI dependencies.

The evidence kinds currently distinguish:

- task behavior;
- static operator evidence;
- ITD structural evidence;
- TDI intervention/recovery evidence;
- adversarial evidence;
- logical cost;
- hardware cost;
- generalization evidence;
- prior-art evidence.

Each reference carries:

```text
kind
repository
artifact
revision_binding
```

`revision_binding` is intentionally opaque. The producing project may bind an
artifact with a Git commit, SHA-256 manifest, evidence-record digest or another
immutable identifier.

This design avoids three dangerous couplings:

1. ADA core does not import ITD/TDI research code merely to identify a candidate;
2. a search objective cannot silently redefine semantic identity by changing a
   diagnostic;
3. historical evidence remains owned by the project that produced it.

## 5. FLAT graduation

`FlatGraduationRecord` binds:

```text
semantic descriptor
reference-oracle evidence
additional evidence references
qualification verdict
```

The current verdict vocabulary is:

- `ContinueResearch`;
- `Adopt`;
- `Adapt`;
- `Reject`.

This is a research handoff artifact. It does not make ADA a production runtime
and it does not authorize automatic modification of FLAT-ATTENTION.

The desired lifecycle is:

```text
ADA candidate
   ↓
reference semantic frozen
   ↓
mathematical / numerical / task / mechanistic evidence
   ↓
FlatGraduationRecord
   ↓
FLAT deterministic reference implementation
   ↓
optimized kernel(s)
```

A later optimized kernel remains an implementation of the declared semantic; it
must not become the source of semantic truth.

## 6. Ecosystem roles

The intended research topology is:

```text
RiemannBench
  structured operator hypotheses
        ↓
ADA
  candidate identity / generation / search / qualification
     ↙                         ↘
ITD Simulator                   TDI
structural diagnostics         intervention/recovery dynamics
     ↘                         ↙
        versioned evidence
             ↓
        FLAT-ATTENTION
  executable semantic + kernels
             ↓
          ElasticXxx
 eventual adaptive runtime policy
```

SciRust supplies reusable mathematical, numerical, statistical, spectral and
optimization primitives when a capability becomes general infrastructure.

### RiemannBench

RiemannBench may provide independently meaningful Toeplitz, prolate,
ground-state/Green-kernel or perturbative operator hypotheses. ADA may turn a
small, explicit mathematical parameterization into a semantic candidate.

No Riemann/zeta interpretation transfers to AI by this path.

### ITD Simulator

ITD is intended to provide structural/mechanistic observations. Those
observations can become ADA evidence only after their own AI validation. ADA
must never reward a candidate merely for maximizing an ITD descriptor unless a
separate experiment has shown that descriptor to be task-relevant.

### TDI

`tdi-ai` now defines a generic intervention/recovery contract. ADA can attach TDI
recovery evidence to a candidate without making TDI a dependency of `ada-core`.

The deterministic attention fixture being developed in TDI Gate B is a useful
future A11-E1 integration case because its perturbation trajectory is
hand-derived. It is not, by itself, a candidate that has earned adoption.

### FLAT-ATTENTION

FLAT remains the executable semantic and kernel target. ADA is the discovery and
qualification system.

### ElasticXxx

ADA-A9 can discover distribution-dependent execution structure offline. If a
runtime selection principle becomes sufficiently general and evidenced,
ElasticXxx is the natural downstream home for adaptive policy.

## 7. A11 gate sequence

### A11-E0 — semantic identity

This change implements the first gate:

- stable semantic identity;
- separate implementation identity;
- minimal reference-level descriptor;
- external evidence references;
- FLAT graduation record;
- tests of the identity boundary.

It does **not** search a new semantic.

### A11-E1 — deterministic non-softmax fixture

Use one analytically tractable semantic candidate and establish:

- hand-derived or exact oracle fixtures;
- deterministic reference evaluation;
- explicit invariants;
- TDI intervention compatibility;
- negative result path.

### A11-E2 — evidence adapters

Define versioned artifact adapters between ADA and ITD/TDI. Avoid circular core
dependencies.

### A11-E3 — bounded semantic search

Extend a grammar only after the semantic and evidence contracts are stable.
Vary one constrained semantic component at a time rather than opening an
unbounded combinatorial search.

### A11-E4 — FLAT promotion

Export one surviving semantic to a deterministic FLAT reference implementation.
Hardware specialization remains a later evidence gate.

## 8. Non-goals of E0

A11-E0 does not:

- change the semantics of any existing ADA A1–A10 result;
- extend the A8 instruction grammar;
- add an ITD dependency;
- add a TDI dependency;
- add a FLAT dependency;
- add SciRust or RiemannBench dependencies;
- add GPU code;
- search a new attention mechanism;
- claim novelty;
- claim model-quality improvement;
- rank candidate semantics.

Its sole purpose is to make the identity/evidence boundary explicit enough that
future semantic search can be falsifiable and auditable.
