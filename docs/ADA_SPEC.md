# ADA Specification

## Purpose

ADA (Algorithm Discovery for Attention) is an offline research system for producing and qualifying attention algorithms. It is separate from semantic attention research (what should be computed) and runtime execution policy (which qualified implementation should run on a device).

## Fundamental separations

1. Candidate generation != evidence.
2. Real-number equivalence != floating-point equivalence.
3. Logical cost != physical hardware cost.
4. Approximation != exact optimization.
5. Research candidate != FLAT production kernel.
6. Prior-art absence has to be established before novelty claims.

## Candidate lifecycle

`GENERATED → STATICALLY_VALID → MATH_VALIDATED → NUMERICALLY_FALSIFIED/VALIDATED → COSTED → HARDWARE_MEASURED → ADOPT/ADAPT/REJECT`

`DISCOVER` is a separate research outcome requiring stronger proof and prior-art review.

## Initial IR direction

ADA-A1 starts without a general synthesis IR. Once the bench can reproduce and falsify hand-written candidates, `ada-ir` and `ada-search` may be introduced with a restricted grammar covering scalar state, comparisons, select/max, arithmetic, exp/log, reductions, and vector accumulation.

The general search system must generate inspectable programs and fail closed on unsupported operations.

## Integration boundaries

- FLAT-ATTENTION remains the production target and external correctness/performance reference.
- SciRust supplies research capabilities such as symbolic algebra, solvers, autodiff, reproducible numerics, algorithm generation, and autotuning when a mission actually requires them.
- ADA does not require SciAgent.

## Reproducibility

Every hardware evidence record should eventually bind at minimum:

- ADA commit SHA
- FLAT commit SHA when applicable
- SciRust commit SHA when applicable
- device and driver identity
- algorithm/candidate identity
- shape/mode/precision
- deterministic seed/data identity
- warmup and measured iteration counts
- correctness metrics
- latency distribution
- verdict
