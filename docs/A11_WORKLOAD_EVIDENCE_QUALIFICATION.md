# ADA-A11 workload-bound qualification

## Purpose

ADA already has separate layers for:

- semantic identity (`ada-core`);
- workload identity (`ada-workload`);
- executable semantic reference programs (`ada-semantic`);
- bounded candidate generation (`ada-search`);
- deterministic CEGIS/counterexamples (`ada-cegis`);
- versioned external evidence (`ADA-SEMANTIC-EVIDENCE-V1` in A10/E2).

`ada-qualification` joins those layers without redefining any of them.

The missing invariant was:

```text
candidate survived an oracle somewhere
```

is not sufficient.  Qualification must say **which workload was actually in the
active corpus**, and later evidence must refer to **the same semantic and the
same workload fingerprint**.

## Lifecycle

```text
SemanticSearchSpace
        ↓
SearchCandidate<SemanticProgram>
        ↓
CegisEngine
        ├── retained counterexample → rejected
        └── bounded survivor
                 ↓
BoundedOracleQualification
  candidate fingerprint
  semantic identity
  exact WorkloadContract
  workload fingerprint
  matching active-fixture fingerprints
                 ↓
SemanticEvidenceRecord (E2)
  exact same SemanticId
  exact same workload fingerprint
                 ↓
EvidenceBoundQualification
```

A CEGIS survivor is still only bounded research evidence.  This layer does not
claim novelty, model usefulness, hardware performance, or FLAT readiness.

## SemanticWorkloadCase

A qualification fixture contains:

```text
WorkloadContract
ReferenceInput
caller-owned canonical input artifact
independent expected output
maximum absolute error tolerance
```

The typed input is never serialized implicitly.  The caller supplies canonical
input text, consistent with the existing `ada-cegis::Fixture` contract.

The qualification-case canonical identity binds:

- the three-lane workload fingerprint;
- caller-owned canonical input text;
- Q/K/V/output shape;
- exact IEEE-754 bits of the expected output;
- exact IEEE-754 bits of the tolerance.

## Workload-aware differential oracle

`SemanticWorkloadOracle` first calls:

```text
SemanticProgram::validate_for_workload(workload)
```

A candidate-specific contract mismatch is a falsification.  A workload outside
the executable v1 reference domain is an oracle/setup error and stops the run
rather than silently rejecting every candidate.

Only after the workload contract passes does the adapter call the independent
f64 semantic evaluator and compare its output against caller-owned oracle truth.

## No resurrection rule

The strongest rule in this slice is:

```text
retained CEGIS counterexample
        + arbitrary later E2 evidence
        = rejected candidate
```

`BoundedOracleQualification::from_cegis_result` checks the rejection archive
before the survivor archive.  A rejected fingerprint returns
`CandidateFalsified` and cannot reach evidence binding.

## Workload coverage rule

A survivor can be bound to a workload only when at least one active CEGIS
fixture contains the same `EvidenceWorkloadFingerprint`.

This prevents a candidate that survived workload A from being relabeled as
oracle-qualified on workload B merely because its semantic program can execute
there.

The matching active-fixture fingerprints are retained by the qualification for
later audit/reconstruction.

## E2 evidence rule

`attach_evidence` rejects:

- an empty evidence set;
- another `SemanticId`;
- another workload fingerprint;
- duplicate canonical E2 artifacts;
- an E2 record that cannot project to `DiagnosticEvidenceRef`.

Evidence records are canonicalized deterministically before storage.  Their
lightweight `DiagnosticEvidenceRef` projections are retained separately for
future graduation records.

Evidence never enters semantic identity.

## Deterministic control

The initial regression control uses one query, two keys, and scalar values:

```text
Q = [0]
K = [1, -1]
V = [2, 4]
```

Both affinity scores are exactly zero.

For ordinary softmax:

```text
weights = [1/2, 1/2]
output  = 3
```

For the current signed-difference rule with two softmax branches, both branches
are uniform on this fixture, so their difference is exactly zero:

```text
weights = [0, 0]
output  = 0
```

The expected oracle output is exactly `3`.  Therefore the softmax candidate
survives with zero tolerance while the signed-difference candidate is retained
as an explicit CEGIS rejection.

This fixture is a plumbing/control test, not a claim that signed attention is
scientifically invalid in general.

## Relationship to ITD and TDI

ITD/TDI remain external evidence producers.  They do not become dependencies of
semantic identity, the semantic evaluator, or the CEGIS engine.

A future ITD/TDI artifact can be attached only after:

1. the semantic candidate survives the declared bounded oracle corpus;
2. the exact workload being claimed was actually evaluated;
3. the E2 record names the same semantic and workload.

Thus mechanistic evidence can refine a surviving research hypothesis but cannot
repair a deterministic semantic failure.

## Relationship to A12 and FLAT

A12 implementation/schedule identity is orthogonal:

```text
semantic qualification ≠ implementation schedule qualification
```

A later stage may bind one qualified semantic to multiple implementation plans,
then collect measured implementation evidence independently.

No FLAT-ATTENTION code is modified by this slice.  A future FLAT graduation
record should consume the already separated semantic, oracle and evidence
identities rather than infer semantics from a selected kernel.
