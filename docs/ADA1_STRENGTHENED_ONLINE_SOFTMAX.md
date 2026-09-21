# ADA1 — Strengthened online Softmax reference (ada-oracle)

Status: **research reference path / no usefulness claim**.

## Purpose

`ada-oracle` historically evaluated streaming online Softmax with naive f32
accumulation (`online_softmax_baseline`) and a one-exp candidate
(`online_softmax_one_exp`). Those paths remain the ADA-A1 E0 laboratory
contracts.

This slice adds `online_softmax_strengthened`: a higher-assurance reference
path that does not replace the naive baseline and does not change the
`AttentionResult` f32 API surface.

## Strengthened behaviors

1. **Exact singleton** — when `seq_len == 1`, output equals the sole value row
   and LSE equals the sole logit (no `exp` / `ln`).
2. **Exact equal-logit weights** — when every logit is bit-identical, softmax
   weights are the exact uniform value `1/n` (dyadic when `n` is a power of
   two); value mixing uses Neumaier compensation; LSE = `m + ln(n)`.
3. **Neumaier-compensated streaming** — otherwise the online Softmax recurrence
   runs in f64 with Neumaier-compensated mass and value accumulation; rescale
   multiplies both the sum and its compensation; non-finite intermediates fail
   closed. Results narrow to finite f32 for API parity.

## Fail-closed rules

- Invalid `AttentionCase` (empty, shape, non-finite inputs) fails as before.
- Exact uniform `1/n` refuses counts outside the exact binary64 integer range.
- Non-finite exp, rescale, mass, value mix, LSE, or f32 narrowing returns an
  error rather than a silent NaN/Inf result.

## Independence

Based on `main`. Does not depend on ADA2 task-quality PRs (#49–#52), on the
A11 exact mixer in PR #46, or on the semantic strengthened evaluator in PR #53.

## Non-claims

No usefulness, novelty, FLAT adoption, or hardware performance claims.
