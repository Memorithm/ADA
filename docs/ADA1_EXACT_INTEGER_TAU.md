# ADA1 — Exact integer tau domain (ada-a6)

Status: **research reference path / no usefulness claim**.

## Purpose

ADA-A6 already had `sparsemax_sorted`, an f64 sorted-projection sparsemax
checked against the A4 bisection oracle. That path remains the floating-point
laboratory candidate, including its certified extreme-magnitude fallback.

This slice adds an exact rational domain on integer scores:

- `sparsemax_exact_i64` — alpha = 2 for every non-empty `i64` vector whose
  prefix sums fit in `i128`. Threshold and probabilities are reduced
  rationals. They sum to one with no f64 rounding and no mass renormalization.
- `entmax15_exact_i64` — alpha = 1.5 only on a subset where both the
  distribution and the threshold are rational:
  - all scores equal and the length is a perfect square, or
  - a unique maximum at least 2 above every other score (gap exactly 2 is
    one-hot; the other score contributes zero).

## Fail-closed rules

- Empty input fails.
- Checked `i128` overflow fails.
- 1.5-entmax inputs outside the subset fail. There is no bisection fallback.
  In particular, equal scores of length 2 are rejected: the probabilities
  would be `1/2`, but the threshold is irrational.
- A support member that is not strictly positive, or an inactive sparsemax
  score that is strictly positive, fails instead of being repaired.

## What this does not change

`ada-a4-entmax-bnb::dense_entmax` stays the general f64 oracle for arbitrary
alpha. `sparsemax_sorted` is unchanged. This slice does not search semantics,
does not attach task quality, and does not promote a candidate.

## Non-claims

No usefulness, novelty, FLAT adoption, or hardware performance claims.
