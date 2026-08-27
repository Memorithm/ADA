# ADA-A11-E2 — Semantic evidence interchange

Status: **schema / no scientific qualification claim**.

## Purpose

A11-E0 separated semantic identity from implementation and evidence. A11-E1
added one deterministic non-softmax semantic fixture. E2 now defines the
artifact boundary through which external experimental systems can attach
mechanistic evidence to an ADA semantic/workload pair without becoming runtime
dependencies of ADA core.

The schema lives in `ada-a10-evidence-schema` rather than creating a second
independent evidence subsystem. The historical A10 hardware `EvidenceRecord`
remains unchanged; E2 adds a separate `SemanticEvidenceRecord` with its own
versioned canonical format.

## Canonical artifact

The interchange header is:

```text
ADA-SEMANTIC-EVIDENCE-V1
```

A record binds:

```text
SemanticId
+ WorkloadFingerprint
+ DiagnosticEvidenceKind
+ producer repository
+ exact producer Git revision
+ producer artifact identity
+ optional intervention identity
+ optional observation horizon
+ metric/protocol identity
+ SHA-256 of preserved raw evidence
+ bounded scalar summary metrics
```

The raw evidence is not replaced by scalar summaries. The SHA-256 field binds
this metadata to the preserved artifact produced by the external laboratory.

## Workload binding

`ada-workload` already defines a canonical deterministic workload text and a
stable three-lane fingerprint:

```text
primary u64
secondary u64
canonical byte length u64
```

E2 copies those three lanes into the evidence record. A change in sequence
geometry, mask, representation, precision, mode, input identity or another
canonical workload property therefore changes the evidence binding.

This prevents evidence obtained on one experimental workload from being
silently reused for another.

## Producer boundary

The producer is identified by:

```text
owner/repository
40-hex Git commit
artifact identity
64-hex SHA-256
```

ADA does not import producer code to validate identity. TDI, ITD, an adversarial
bench, a task suite, a hardware bench or another producer can emit the same
versioned envelope.

The intended relationship is:

```text
ADA SemanticId + WorkloadContract
              ↓
external producer executes its own protocol
              ↓
raw evidence artifact
              ↓ SHA-256 + metadata
ADA-SEMANTIC-EVIDENCE-V1
              ↓
DiagnosticEvidenceRef
              ↓
ADA qualification / graduation record
```

## TDI recovery requirement

A `TdiRecovery` record fails closed unless it declares both:

- `intervention_identity`;
- `observation_horizon`.

This encodes the minimum meaning of an intervention/recovery claim. A recovery
number without a declared perturbation or horizon is not accepted as a TDI
recovery artifact.

The schema does not define TDI's recovery metric. TDI remains responsible for
that scientific definition and for its raw evidence.

## ITD structural evidence

An `ItdStructural` record may omit intervention and horizon because a static
structural descriptor need not be intervention-based.

This does not make an ITD descriptor a target objective or semantic truth. It
only makes a validated structural measurement attachable to the same semantic
and workload used by other evidence producers.

## Determinism

Canonical interchange uses:

- fixed field names and order;
- hex encoding for external UTF-8 identifiers;
- fixed-width lowercase hexadecimal workload lanes;
- exact producer Git SHA;
- exact evidence SHA-256;
- lexicographically sorted summary metric names;
- IEEE-754 `f64::to_bits()` encoding for scalar metric values.

The decoder rejects:

- missing, duplicate or unknown fields;
- unsupported versions;
- malformed semantic identities;
- malformed producer repository/revision;
- malformed SHA-256 values;
- missing TDI intervention/horizon;
- duplicate metric names;
- non-finite metrics;
- oversized records or metric sets.

## Relation to historical A10 evidence

Historical A10 hardware evidence remains:

```text
EvidenceRecord
  algorithm_id
  host_fingerprint
  timestamp
  toolchain
  git_commit
  sha256_evidence
  metrics
```

E2 does not reinterpret those records and does not require historical artifacts
to migrate.

The two formats answer different questions:

```text
A10 historical record:
  "what hardware/algorithm evidence was measured on this committed ADA state?"

A11-E2 semantic record:
  "what external evidence was produced for this semantic on this exact workload?"
```

## A11 integration

`SemanticEvidenceRecord::diagnostic_reference()` projects the fully bound E2
record into the lighter A11 `DiagnosticEvidenceRef` used by qualification and
future FLAT graduation records.

The resulting reference carries the evidence kind, producer repository,
artifact identity and a revision binding containing both Git revision and raw
artifact SHA-256.

The semantic identity itself remains unchanged.

## Non-goals

E2 does not:

- execute ITD or TDI;
- define ITD/TDI metrics;
- decide whether an evidence result is favorable;
- aggregate different metrics into one scalar score;
- rank semantic candidates;
- make evidence part of `SemanticId`;
- establish novelty;
- claim model quality;
- promote a semantic to FLAT;
- add a TDI or ITD code dependency.

## Next gate

With E0 identity, E1 deterministic semantics and E2 evidence interchange in
place, the next research step can connect bounded `ada-search` candidates to
**qualification** rather than merely generation.

That gate should preserve an explicit distinction between:

```text
generated
statically valid
oracle-qualified
mechanistically evaluated
surviving
rejected
```

and should use controlled counterexamples/ablations before any larger language
model experiment.
