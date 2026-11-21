# ADA-A3 value-radius output-error certificate (research, mixed/modest)

Status: research reference, **ADAPT**. Not a production runtime, no hardware
measurement, no novelty claim (centring/Chebyshev-radius style bounds on convex
combinations are classical; prior-art review not done).

## Conjecture

Probability-L1 certificates ignore the values. For the attention *output*
`o = Σ p_i v_i`, with unread fraction `t = T/(S+T) ≤ t_hi` and truncated output
`ô`: `o − ô = t(õ − ô)`, `õ` a convex combination of unread values. With a
prefill centre `c` (mean of V) and unread-suffix radius `R_j = max ‖v_i − c‖`:
`‖o − ô‖₂ ≤ t_hi (R_j + ‖ô − c‖)` (ô is known exactly at run time). Compare with
the L1-route bound `t_hi · 2·max‖v‖` ("Uncentered"). Expected win: values that
cluster away from the origin (large mean, small spread), with norm-varying keys.

Falsification: ‖o−ô‖ > certificate; no regime where centred modes read fewer rows.

## Implementation

`crates/ada-a3-certified-softmax/src/value_radius.rs`
(`certified_value_radius_attention[_accounted]`, `ValueRadiusMode::{Uncentered,
Centered, SuffixCentered}`, `ValueRadiusCost`). Keys scanned in norm order
(as #71); unread-mass bound in log domain (lesson of the #76 underflow bug).
Oracle: independent dense stable softmax-weighted mean (separate code in tests).

## Results (native f64)

- Fail-closed on 12 invalid-input cases.
- Exhaustive grid (n ≤ 3, d=2, dv=1, 3 key values, 3 value levels, 3 queries incl.
  zero, ε_out ∈ {0, .05, 1, 100}, 3 modes): 0 violations; skipping occurs.
- Adversarial sweep (300 draws × 5 ε × 3 modes; key scale up to 1000, query ×100,
  duplicated keys, 1e6 value outlier on a small-norm key, values clustered at 7):
  0 violations.
- Witness (128 keys, values 100 ± 0.01, ε_out=1e-2): centred reads strictly fewer rows
  than uncentred (test asserts it).
- Fraction of rows read, n=256, d=16, dv=4, 40 draws, ε_out = ε_rel·σ_v
  (unc / centred / suffix-centred):

| values | key log-norm σ | ε_rel | unc | cen | suf |
|---|---|---|---|---|---|
| zero-mean | 0 | .01 | 1.000 | 1.000 | 0.999 |
| zero-mean | 1 | .01 | 0.682 | 0.674 | 0.671 |
| zero-mean | 1.5 | .01 | 0.154 | 0.144 | 0.144 |
| offset 5, σ_v .1 | 1 | .1 | 0.697 | 0.600 | 0.592 |
| offset 5, σ_v .1 | 1.5 | .01 | 0.243 | 0.144 | 0.144 |
| offset 50, σ_v .01 | 1.5 | .01 | 0.448 | 0.144 | 0.144 |
| any | 0 (equal norms) | any | ≈1.000 | ≈1.000 | ≈0.999 |

Cost: identical per-row cost for all modes; centred modes add one O(n·dv) prefill
pass and an O(dv) distance per certificate check; prefill sort O(n log n).

## Verdict

**Modest positive, conditional.** The output certificate is valid and strictly
no worse than the uncentred bound; it matters when values are offset from the
origin and spread is small (up to 3× fewer rows at σ=1.5, offset 50). Suffix
radius adds ≤ 0.01 over global radius. It does nothing for the real bottleneck:
with equal key norms the Cauchy–Schwarz tail bound never lets the scan stop
(1.000). Not tested: real Q/K/V traces (synthetic only), hardware, and
combination with tighter key bounds (cone/box orders), which is the natural next step.
