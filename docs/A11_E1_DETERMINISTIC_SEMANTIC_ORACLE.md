# ADA-A11-E1 — Deterministic non-softmax semantic oracle

Status: **research oracle / no usefulness claim**.

## Purpose

A11-E0 separated semantic identity, implementation identity and evidence.
`ada-workload` then added a versioned workload contract that can describe
explicit Q/K/V workloads or named precomputed interaction rules without
silently reinterpreting one as the other.

E1 combines those two layers for the first time with one tiny executable
semantic whose downstream behavior is independently derivable.

The objective is not to propose a competitive attention mechanism. The
objective is to prove that ADA can now represent:

```text
semantic identity
+ explicit workload
+ deterministic reference implementation
+ independent mathematical oracle
```

before the semantic search grammar is expanded.

## Frozen semantic

The scalar sequence state contains three token positions:

```text
x = [x0, x1, x2]^T.
```

One semantic step applies

```text
      [ 1/2  1/2   0  ]
M  =  [ 1/4  1/2  1/4 ]
      [  0   1/2  1/2 ].
```

Every row is non-negative and sums exactly to one in binary floating point.
The resulting semantic descriptor is therefore:

```text
family   = Experimental
name     = balanced-three-token-mixer
revision = 1
mask     = Bidirectional
state    = Stateless
weights  = ProbabilitySimplex
```

`Experimental` is intentional. The fixture has not earned a more specific
scientific family and is not presented as standard softmax attention.

## Workload contract

The `ada-workload` description freezes:

```text
batch_count       = 1
query_length      = 3
kv_length         = 3
query_heads       = 1
kv_heads          = 1
value_dimension   = 1
qk_dimension      = none
topology           = self-attention
head_grouping      = MHA
mode               = prefill
mask               = bidirectional
precision          = f64 throughout
state              = stateless
inputs              = precomputed scores / interaction artifact
input identity      = ada-a11-e1-fixed-mixer
```

The missing Q/K dimension is a feature, not an omission. This semantic is a
fixed interaction matrix and must not masquerade as a dot-product Q/K
construction.

## Independent oracle

Define

```text
v = [1, 0, -1]^T.
```

Direct multiplication gives

```text
M v = (1/2) v.
```

Therefore

```text
M^h v = 2^-h v.
```

The first states are:

```text
h=0 : [1,     0, -1]
h=1 : [1/2,   0, -1/2]
h=2 : [1/4,   0, -1/4]
h=3 : [1/8,   0, -1/8]
```

The crate implements two separate paths:

1. `advance_horizon(...)` repeatedly executes the matrix evaluator;
2. `antisymmetric_oracle(...)` computes the closed-form dyadic trajectory and
   never calls the evaluator.

Tests compare both paths for horizons `0..=16`.

This separation matters. A test that derives its expected result through the
same evaluator is not an independent oracle.

## Constant-mode control

Because the rows sum to one,

```text
M [c,c,c]^T = [c,c,c]^T.
```

The test suite verifies this invariant independently of the antisymmetric mode.
This catches a different class of operator/transcription errors.

## Relationship to TDI Gate B

TDI Gate B independently uses the same three-token matrix and the same
antisymmetric perturbation to validate intervention/recovery dynamics.

The ADA E1 crate intentionally does **not** copy TDI's reciprocal-L-infinity
recovery metric. ADA owns the reference semantic and its deterministic state
trajectory; TDI owns the intervention/recovery measurement.

That separation gives the future cross-project experiment the desired form:

```text
ADA reference semantic
      ↓ state trajectory
TDI intervention/recovery adapter
      ↓ dynamic evidence artifact
ADA qualification record
```

The TDI PR containing Gate B has been merged, but its Jetson validation run was
still queued when this E1 work was prepared. A merge is therefore not treated
as hardware/runner qualification evidence.

## Relationship to ITD

No ITD descriptor is computed in E1.

The same deterministic trajectory can later be exposed to an ITD-AI structural
descriptor adapter. That must happen through an evidence interface, not by
making ITD part of the semantic evaluator.

## Relationship to RiemannBench

E1 deliberately uses an elementary operator rather than importing a prolate,
Toeplitz or Green-kernel construction.

This establishes the experimental plumbing first. Once the oracle/evidence
pipeline is trustworthy, richer independently meaningful operator hypotheses
from RiemannBench can enter as distinct semantic candidates.

No Riemann/zeta semantics transfer to this fixture.

## Relationship to FLAT-ATTENTION

E1 is CPU/reference-only and framework-independent.

It does not create a FLAT kernel. If a future semantic survives task,
mechanistic and numerical qualification, FLAT can implement it against the
frozen semantic identity and fixtures.

## What E1 proves if green

A green E1 proves only that:

- ADA can bind a non-softmax semantic to a stable `SemanticId`;
- an implementation identity can be bound separately;
- `ada-workload` can describe the case without inventing Q/K;
- the scalar evaluator reproduces the hand-derived eigenmode trajectory;
- the probability-simplex row and constant-mode controls hold;
- non-finite input fails closed.

## What E1 does not prove

It does not establish:

- model-quality gain;
- usefulness on associative recall or language modeling;
- novelty;
- superiority over softmax;
- causal relevance of the TDI recovery score;
- usefulness of ITD descriptors;
- hardware advantage;
- a new FLAT semantic ready for production;
- transfer from any RiemannBench result.

## Next gate

After E1 is green, the next narrow step should be **A11-E2 evidence
interchange**, not a broad semantic search.

E2 should define a versioned artifact that can bind:

```text
SemanticId
Workload fingerprint
producer repository/revision
intervention identity
observation horizon
metric identity
raw/summary evidence digest
```

and should support TDI and ITD producers without adding either project as an
`ada-core` dependency.

Only after that interface is stable should A11-E3 open a bounded semantic
search dimension.
