# ADA-A11 Registry Reconciliation

Status: **REFERENCE-PIPELINE-IMPLEMENTED / RESEARCH-ONLY / NOT-ADOPTED / NO-NOVELTY-CLAIM**

Audited default-branch commit: `6ecc9e5b3ee6f8cf8ec3115400d64aa99113280c`.

## Purpose

`docs/ALGORITHM_REGISTRY.md` historically enumerates ADA-A1 through ADA-A10, while the A11 reference qualification infrastructure was implemented in later merged work. This document records that state without rewriting the historical A1-A10 evidence record and without promoting any A11 semantic candidate.

The registry interpretation for ADA-A11 is:

> ADA-A11 has an implemented reference research pipeline for semantic identity, deterministic reference execution, versioned evidence, bounded generation, workload-bound CEGIS qualification, fail-closed FLAT graduation artifacts, and bit-exact replay. This is infrastructure qualification only. It is not evidence that any A11 semantic is useful, novel, production-qualified, or adopted by FLAT-ATTENTION.

## Implemented evidence slices

| Slice | Evidence | Registry meaning |
| --- | --- | --- |
| A11-E0 semantic identity | PR #8, merge `6d0fabb339cbed5d93d672d5e8b0df641160378f` | Semantic identity is separated from implementation identity and diagnostic evidence. |
| A11-E1 deterministic non-softmax oracle | PR #12, merge `1ceb083762e0c88bb5901baaddd934de887176fb` | A deterministic reference execution fixture exists; analytic tractability is not usefulness evidence. |
| A11-E2 versioned semantic evidence | PR #15, merge `a0eac6d88f73b4b67c3ae47dda90b92191958b21` | Evidence interchange is versioned and provenance-bound rather than embedded into semantic identity. |
| A11-E3 bounded semantic generation | PR #14, merge `8e9a12d2a82964adc089f60a4e76069f578830b5` | Candidate generation is bounded and inspectable; generation does not imply qualification. |
| A11-E3 workload-bound CEGIS qualification | PR #21, merge `134ba22662af90917ef56a9e343c8d81c83b46f5` | Qualification is workload/evidence bound and retains counterexamples. |
| A11-E4a fail-closed FLAT graduation bundle | PR #22, merge `bb8d711252d78de43f006aba0cf2debe3f634f1b` | ADA can export a qualified research handoff artifact without performing production lowering. |
| A11-E4b bit-exact replayable fixtures | PR #23, merge `f5075ac505df68f14199103eef189a061c4f3d3e` | Reference evidence can be replayed with exact fixture identity. |
| RB5 structured-operator import hardening | PR #40, merge `6ecc9e5b3ee6f8cf8ec3115400d64aa99113280c` | External mathematical operators can enter ADA through a typed, fail-closed research import contract while preserving open gaps and non-transferable interpretations. |

## Qualification boundary

The implemented A11 pipeline must continue to follow ADA's research ladder:

`GENERATE -> PROVE_OR_FALSIFY -> INDEPENDENT_ORACLE -> ADVERSARIAL_TEST -> COST -> REAL_HARDWARE_WHEN_RELEVANT -> PRIOR_ART_REVIEW_IF_NOVELTY_IS_CANDIDATE -> ADOPT/ADAPT/REJECT`

The following statements are therefore explicitly **not** implied by the implemented pipeline:

- an A11 semantic is better than standard attention;
- an A11 semantic improves model quality;
- an A11 semantic is novel;
- an imported RiemannBench/RB5 operator transfers its source-domain interpretation to sequence modelling;
- an ITD or TDI diagnostic proves task usefulness;
- a logical operation-count reduction is a physical speedup;
- a candidate is ready for FLAT GPU execution;
- a candidate is eligible for runtime selection by ElasticXxx.

## Ownership and handoff

ADA owns semantic identity, reference execution, attention-specific falsification/CEGIS, adversarial qualification, evidence registration, and backend-neutral candidate study.

FLAT-ATTENTION owns production lowering, GPU correctness, device qualification, and performance promotion. Forge may search implementations or parameters only after semantic constraints are frozen. TDI and ITD may contribute mechanistic evidence, but neither replaces ADA semantic/task gates. SciRust remains the preferred owner for genuinely reusable mathematical or tensor primitives.

## Registry administration

This document is a reconciliation supplement for the current A11 state. It does not alter or erase any negative result in the historical A1-A10 registry, and it must not be cited as scientific evidence for a candidate semantic.

A future consolidation of `docs/ALGORITHM_REGISTRY.md` may add an A11 row using the conservative status above, but must preserve the same non-promotion boundary and all existing historical evidence.