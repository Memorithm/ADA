# ADA1 — Quantized lattice attention (ada-a6)

Status: **research reference path / no usefulness, novelty, or FLAT claim**.

## Conjecture

Cycle #63 showed that exact rational `alpha = 1 + 1/k` entmax on integer
scores is a partial map: the domain collapses from every vector at `k = 1`
to 19 / 3905 at `k = 4` on a small grid. The follow-up mechanism
(*quantized lattice attention*, QLA) is:

1. Shift scores so the maximum is zero and round `scale * (z_i - max z)`
   onto the integer lattice (fail closed on overflow / non-finite input).
   `scale` amplifies before rounding; the integers are then treated as
   raw lattice scores, so `scale > 1` is a sharpness parameter, not a
   reconstructed `q / scale` approximation of the original operator.
2. Apply `lattice_entmax_i64` (exact rational `p_i = ((q_i - u)/k)^k`).
3. If the rounded vector is outside the rational domain, *gap-complete*:
   keep the first argmax and push every other score down to at most
   `max - k`. After the max-shift the peak is `0`, so this is always
   defined without underflow of `i64::MIN` for `k <= 16`, and the solver
   then accepts a one-hot distribution.

**Claims under test (any one falsifies the corresponding claim):**

- *Totality.* Every finite non-empty score vector with a finite positive
  scale and `k in 1..=16` yields a non-negative probability vector that
  sums to 1 (or a documented overflow / invalid-argument error). An
  unexplained `Err` falsifies totality.
- *Oracle parity on hits.* When step 2 accepts, the exact rationals match
  `ada_a4_entmax_bnb::dense_entmax` on the *quantized* integers within
  `1e-9` L1. A larger disagreement falsifies parity.
- *Task drop-in.* On two tiny mechanistic tasks — unique-needle majority
  / hit@1, and equal-pair interpolation — QLA matches softmax and
  sparsemax success. A systematic gap (QLA failing a task the baselines
  solve, or the reverse) falsifies the drop-in claim. Negative results
  are recorded, not repaired.

No novelty or hardware claim is made; no prior-art review was performed.

## Method

`quantized_lattice_attention[_accounted]` in `ada-a6-tau-solvers`.
Independent oracles: A4 `dense_entmax` on original and quantized scores,
and a stable softmax baseline used only for task comparison. Logical
counters (`QlaCost`) are not physical measurements. Under Miri the
Gaussian survey uses 8 samples of length 4 and skips the task sweep;
recorded figures come from the native run.

## Results

Native Gaussian survey (`n = 8`, 200 LCG/Box-Muller samples, independent
A4 oracle). Lattice-hit parity against A4 *on the quantized integers*
had 0 disagreements (L1 ~ 1e-17 or smaller) in every cell.

| k | scale | lattice hits | gap completions | mean L1 vs unquantized A4 |
|---|-------|--------------|-----------------|---------------------------|
| 1 | 1 | 200 | 0 | 0.271397 |
| 1 | 2 | 200 | 0 | 0.304220 |
| 1 | 4 | 200 | 0 | 0.407621 |
| 1 | 8 | 200 | 0 | 0.547996 |
| 2 | 1 | 61 | 139 | 0.741019 |
| 2 | 2 | 59 | 141 | 0.887652 |
| 2 | 4 | 102 | 98 | 0.861232 |
| 2 | 8 | 154 | 46 | 0.828809 |
| 3 | 1 | 11 | 189 | 1.000065 |
| 3 | 2 | 28 | 172 | 0.967157 |
| 3 | 4 | 84 | 116 | 0.948977 |
| 3 | 8 | 123 | 77 | 0.959782 |

Increasing `scale` does not reduce L1 against the original-score oracle
(for `k = 1` it rises from 0.27 to 0.55): amplification-then-entmax is
not a finer discretisation of the same map. For `k >= 2`, larger scale
raises the lattice-hit rate (more unique peaks with gap `>= k`) but
mean L1 stays around 0.83-1.00 because completions are one-hot and even
hits are peaked integer-score entmax.

Mechanistic tasks, length 8, scale 1, margins `{0.1, 0.4, 0.6, 1, 2, 4}`:

- Unique needle at index 3, hit@1: softmax and sparsemax succeed at every
  margin. QLA (`k = 1, 2, 3`) fails at margins 0.1 and 0.4 (everything
  rounds to a tie; `k = 2` gap-completes onto index 0) and succeeds from
  margin 0.6 upward (the peak survives rounding).
- Equal-pair interpolation at indices 2 and 5 (success: both peaks have
  mass `> 0.05` and `|p_2 - p_5| <= 0.05`): softmax, sparsemax, f64
  1.5-entmax, and QLA `k = 1` succeed at every tested margin. QLA
  `k = 2` and `k = 3` fail once the two peaks survive rounding (margin
  `>= 0.6`): the pair is outside the rational domain, gap completion
  one-hots the first peak, and the decode jumps to `+1`. At sub-bin
  margins `k = 2` also fails (gap completion one-hots index 0), while
  `k = 3` accidentally succeeds because length 8 is a perfect cube and
  the all-tied vector is in-domain (uniform `1/8` on every index).

Hand fixtures:

- In-domain unit spike `[1, 0, 0, 0, 0]` at `k = 2`, scale `1`: lattice
  hit, matches both `lattice_entmax_i64` and the A4 1.5-entmax oracle.
- Two-way tie `[0, 0]` at `k = 2`: gap completion to `[0, -2]`, one-hot
  on the first index.
- Equal-pair interpolation `scores = [2, 2, 0, 0]`, values `[+1, -1, 4, -4]`:
  softmax, sparsemax, f64 1.5-entmax, and QLA `k = 1` all decode to `0`;
  QLA `k = 2` and `k = 3` gap-complete and decode to `+1`.
- Unique needle of margin `5`: softmax, sparsemax, QLA `k = 1` and
  `k = 2` all put majority mass on the needle.
- Sub-bin needle of margin `0.2` at scale `1`: softmax hit@1 succeeds;
  QLA `k = 1` sees a four-way tie and returns the uniform vector
  (hit@1 picks index 0).

## Verdict

**ADAPT (research reference) / drop-in claim REJECTED.** Totality and
lattice-hit oracle parity hold on the tested domain. The mechanism is
not a drop-in replacement for softmax or sparsemax: coarse quantisation
erases unique-needle hit@1, gap completion destroys equal-pair
interpolation for `k >= 2`, and larger `scale` does not approximate the
unquantized operator more closely. No usefulness, novelty, hardware, or
FLAT-adoption claim.

## Open

A reconstructed solver with target `(k * scale)^k` and
`p_i = (q_i - u)^k / (k * scale)^k` (true `q / scale` entmax); a
nearest in-domain multi-support completion that preserves pair
interpolation; scale adapted to local gaps; the same tasks wired through
the ADA2 `CausalArgmaxRetrievalTask` / `CopyTokenAttentionTask` lane
separation.
