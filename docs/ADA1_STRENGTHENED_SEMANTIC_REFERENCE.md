# ADA1 — Strengthened semantic reference evaluator

Status: **research reference path / no usefulness claim**.

## Purpose

`ada-semantic` historically evaluated scaled-dot-product affinity, softmax /
signed-difference weighting, and value mixing with naive left-to-right f64
accumulation. That path remains the default `SemanticProgram::evaluate`
contract used by CEGIS differential checks.

This slice adds `SemanticProgram::evaluate_strengthened`: a higher-assurance
reference path that does not replace the naive evaluator and does not expand
the executable semantic grammar.

## Strengthened behaviors

1. **Exact integer/dyadic affinity** — when every Q/K entry is an exact binary64
   integer and the affinity scale is an exact non-positive power of two
   (`2^-k`), the scaled dot product is accumulated in `i128` and converted back
   only when the dyadic result is exactly representable in finite f64.
2. **Exact equal-score weights** — when every selected score is bit-identical,
   softmax weights are the exact uniform value `1/n` (dyadic when `n` is a
   power of two) instead of evaluating `exp`.
3. **Neumaier-compensated accumulation** — otherwise softmax mass and value
   mixes use Neumaier compensation; non-finite intermediates fail closed.

## Fail-closed rules

- Non-integer Q/K or non-dyadic scale fall back to compensated f64 affinity
  rather than silently claiming exactness.
- Exact-path `i128` overflow returns `SemanticIrError::Overflow`.
- Non-finite affinity, softmax, or value-mix stages return
  `SemanticIrError::NonFiniteValue`.

## Independence

Based on `main`. Does not depend on ADA2 task-quality PRs (#49–#52) or on the
A11 exact mixer in PR #46.

## Non-claims

No usefulness, novelty, FLAT adoption, or hardware performance claims.
