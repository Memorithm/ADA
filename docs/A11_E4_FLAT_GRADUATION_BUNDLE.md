# A11-E4a — fail-closed FLAT graduation bundle

## Purpose

A11-E4a introduces a deterministic handoff artifact between ADA research and a future FLAT reference-semantic ingestion path.

The canonical artifact is:

```text
ADA-FLAT-GRADUATION-V1
```

It does **not** lower GPU kernels, modify FLAT-ATTENTION, or claim that a searched semantic is useful, novel, production-ready, or faster than an existing implementation.

## Inputs

A bundle can be assembled only from an existing `EvidenceBoundQualification` and the exact completed CEGIS result that produced that qualification. It binds:

```text
exact SemanticProgram
+ exact WorkloadContract
+ exact qualified CEGIS fixture artifacts
+ ImplementationPlan
+ explicit OperationProfile
+ explicit CostAssumptions
+ A12-derived logical/estimated objectives
+ evidence-backed task/measured objectives
+ full E2 SemanticEvidenceRecord artifacts
+ explicit research verdict
```

Semantic, workload, implementation, evidence, and objectives remain separate typed identities.

## Exact oracle preservation

The qualification layer historically retained the fingerprints of active fixtures that covered a workload. E4a additionally copies each matching CEGIS fixture identifier and complete canonical `ADA-QUALIFICATION-CASE-V1` text into the handoff bundle.

On decode, ADA reconstructs the generic CEGIS fixture identity and verifies its fingerprint. The qualification-case workload fingerprint embedded in the fixture must match the bundle workload.

This gives a future FLAT importer exact deterministic oracle material rather than an opaque claim that an oracle once passed.

## Reproducible A12 cost boundary

Logical and estimated cost are not caller-supplied fields in E4a.

The constructor runs the existing backend-neutral A12 cost model from:

```text
WorkloadContract
+ ImplementationPlan
+ OperationProfile
+ CostAssumptions
```

The decoder repeats the same calculation and requires the canonical `ObjectiveVector` logical and estimated sections to match exactly. Editing an estimated byte count, FLOP count, Q/K evaluation count, transcendental count, value-operation count, workspace estimate, or reduction estimate therefore invalidates the bundle.

This remains an estimate. It is not DRAM traffic, latency, occupancy, throughput, energy, or a speedup claim.

## Measured cost boundary

Physical latency or energy may appear only in the measured section of `ObjectiveVector` and only when the same semantic/workload evidence set contains `DiagnosticEvidenceKind::HardwareCost` provenance.

The presence of a hardware evidence reference does not make an A12 estimate a measurement. Estimated and measured sections remain distinct.

## Task-quality boundary

An observed task/model quality value requires a `TaskBehavior` E2 record for the same semantic and workload.

A diagnostic value from ITD, TDI, static operator analysis, or another producer does not silently become task-quality evidence.

## Correctness and verdict boundary

`BoundedOracleQualification` explicitly means survival of a declared finite CEGIS corpus. It is not a proof of general correctness.

Therefore E4a derives:

```text
CorrectnessStatus::Provisional
```

and does not allow the caller to set correctness to `Qualified`.

For the same reason, E4a rejects `ADOPT` and `ADAPT`. The allowed useful handoff state is currently `CONTINUE_RESEARCH`; `REJECT` remains representable for a preserved negative decision. A later version may permit stronger verdicts only after ADA has a distinct provenance-bound correctness qualification protocol.

## Canonical identity

The bundle has a strict line-oriented codec. Nested semantic, workload, implementation, objective, fixture, and E2 artifacts are preserved as hex-encoded canonical text.

The Pareto `CandidateKey` is built from only:

```text
SemanticProgram
+ WorkloadContract
+ ImplementationPlan
```

Evidence, measurements, objective values, and verdict are excluded from candidate identity so later evidence cannot redefine what candidate was evaluated.

## What E4a proves

Green tests for this slice establish only that ADA can produce and validate a deterministic, internally consistent research handoff whose semantic/workload/implementation/oracle/evidence/cost identities cannot silently diverge.

They do not establish:

- semantic novelty;
- language-model quality;
- general correctness;
- hardware speedup;
- FLAT kernel parity;
- GPU readiness;
- ITD/TDI predictive usefulness;
- Riemann relevance to AI.

## Next gate

After this artifact is stable, the next narrow step is a FLAT-side **reference ingestion** path that reads the frozen semantic/workload/oracle contract and checks a deterministic FLAT reference implementation against the retained oracle fixtures.

Portable/optimized GPU kernels and hardware promotion remain later gates.
