# ADA1 — Multi-support quantized lattice attention (ada-a6)

Status: **research reference path / no usefulness, novelty, or FLAT claim**.

## Conjecture

Cycle #65 made quantized lattice attention (QLA) total by one-hot gap
completion. That completion destroys equal-pair interpolation for
`k >= 2`, and a fixed scale erases unique needles whose margin is below
half a rounding bin. The follow-up mechanism (*multi-support QLA*,
MS-QLA) is:

1. Optionally adapt the scale to the top-plateau gap:
   `effective_scale = max(scale, (1/2) / (max z - next distinct z))`.
   After max-shift, background scores then round to at most `-1`.
2. Quantize as in QLA and apply `lattice_entmax_i64`.
3. On a miss, *plateau-complete*: keep every quantized maximum and push
   every other score down to at most `max - k`, then retry the solver.
   When the argmax set is a singleton this is QLA's one-hot completion.
4. On a second miss, return the exact uniform distribution on the
   quantized argmax set (exact sparsemax of a `{0, -1}` indicator).

Step 4 is required, not a convenience. For `k = 2` the two-score
discriminant is `D = 8 - (z_1 - z_2)^2`, which is a perfect square only
at `|z_1 - z_2| = 2`, and then the complementary inequalities exclude
the lower score. No integer vector therefore has a two-point
lattice-entmax support, so no in-domain score completion can preserve a
tied pair.

**Claims under test (any one falsifies the corresponding claim):**

- *Totality.* Every finite non-empty score vector with a finite positive
  scale and `k in 1..=16` yields a non-negative probability vector that
  sums to 1 (or a documented overflow / invalid-argument error).
- *Oracle parity on hits.* When step 2 accepts, the exact rationals
  match `ada_a4_entmax_bnb::dense_entmax` on the *quantized* integers
  within `1e-9` L1. On the uniform path, the exact rationals match A4
  sparsemax (`alpha = 2`) on the `{0, -1}` argmax indicator within
  `1e-9` L1.
- *Task recovery.* On unique-needle hit@1 and equal-pair interpolation,
  gap-adaptive MS-QLA matches softmax and sparsemax success. The same
  unique needle is graded through ADA2 `CausalArgmaxRetrievalTask` and
  `CopyTokenAttentionTask` (hard retrieval; first index on ties). Those
  fixtures cannot express soft pair interpolation, which remains a
  separate mechanistic check. A systematic gap against the baselines
  falsifies the recovery claim.

No novelty or hardware claim is made; no prior-art review was performed.

## Method

`multi_support_quantized_lattice_attention[_accounted]` in
`ada-a6-tau-solvers`, with `MsqlaScaleMode::{Fixed, GapAdaptive}`.
Independent oracles: A4 `dense_entmax` on original and quantized scores,
A4 sparsemax on the argmax indicator, and a stable softmax baseline used
only for task comparison. Logical counters (`MsqlaCost`) are not
physical measurements. Under Miri the two-point-support grid and the
Gaussian survey are reduced and the task sweep is skipped; recorded
figures come from the native run.

## Results

Native enumeration of integer vectors, lengths `1..=5`, scores `{0..4}`
(**3905** vectors): `lattice_entmax_i64(..., k = 2)` accepts a nonempty
subset and reports **0** supports of size exactly 2, matching the
two-score discriminant argument.

Native Gaussian survey (`n = 8`, 200 LCG/Box-Muller samples). Lattice-hit
parity against A4 on the quantized integers, and uniform-path parity
against A4 sparsemax on the argmax indicator, had **0 disagreements** in
every cell.

Fixed scale (no adaptation):

| k | scale | lattice hits | plateau | argmax-uniform | mean L1 vs unquantized A4 |
|---|-------|--------------|---------|----------------|---------------------------|
| 1 | 1 | 200 | 0 | 0 | 0.315564 |
| 1 | 2 | 200 | 0 | 0 | 0.323620 |
| 1 | 4 | 200 | 0 | 0 | 0.432874 |
| 1 | 8 | 200 | 0 | 0 | 0.501848 |
| 2 | 1 | 51 | 67 | 82 | 0.384945 |
| 2 | 2 | 61 | 77 | 62 | 0.572686 |
| 2 | 4 | 107 | 57 | 36 | 0.695225 |
| 2 | 8 | 149 | 32 | 19 | 0.753601 |
| 3 | 1 | 23 | 80 | 97 | 0.495098 |
| 3 | 2 | 36 | 102 | 62 | 0.706796 |
| 3 | 4 | 68 | 85 | 47 | 0.827436 |
| 3 | 8 | 117 | 65 | 18 | 0.921776 |

Gap-adaptive scale (`effective_scale = max(scale, 1/2 / gap)`):

| k | scale | lattice hits | plateau | argmax-uniform | adaptations | mean L1 vs unquantized A4 |
|---|-------|--------------|---------|----------------|-------------|---------------------------|
| 1 | 1 | 200 | 0 | 0 | 104 | 0.513567 |
| 1 | 2 | 200 | 0 | 0 | 64 | 0.503558 |
| 1 | 4 | 200 | 0 | 0 | 35 | 0.549942 |
| 1 | 8 | 200 | 0 | 0 | 23 | 0.611295 |
| 2 | 1 | 32 | 160 | 8 | 109 | 0.758650 |
| 2 | 2 | 73 | 117 | 10 | 46 | 0.713938 |
| 2 | 4 | 115 | 78 | 7 | 22 | 0.821113 |
| 2 | 8 | 142 | 56 | 2 | 21 | 0.856984 |
| 3 | 1 | 8 | 175 | 17 | 111 | 0.907390 |
| 3 | 2 | 37 | 159 | 4 | 51 | 0.898320 |
| 3 | 4 | 80 | 116 | 4 | 30 | 0.891097 |
| 3 | 8 | 137 | 63 | 0 | 7 | 0.927677 |

For `k = 2`, scale `1`, replacing one-hot completion with plateau /
argmax-uniform lowers mean L1 against the unquantized A4 oracle from
QLA's 0.741019 to 0.384945 (fixed MS-QLA). Gap adaptation then *raises*
L1 (0.758650 at the same cell) because it sharpens surviving peaks; it
is not a finer discretisation of the original operator.

Mechanistic tasks, length 8, scale floor 1, margins
`{0.1, 0.4, 0.6, 1, 2, 4}`:

- Unique needle at index 3, hit@1: softmax and sparsemax succeed at
  every margin. QLA and *fixed-scale* MS-QLA fail at margins 0.1 and
  0.4 (everything rounds to a tie). *Gap-adaptive* MS-QLA succeeds at
  every tested margin for `k = 1, 2, 3`. At margin 0.1 and `k = 2` the
  adapted vector is a unit-spike (`[0, -1^7]` up to permutation) and
  hits the lattice solver with needle mass `9/16 = 0.5625`. At the same
  margin and `k = 3` the path is plateau completion to a one-hot.
- Equal-pair interpolation at indices 2 and 5: softmax, sparsemax,
  f64 1.5-entmax, QLA `k = 1`, and MS-QLA (`k = 1, 2, 3`, both scale
  modes) succeed at every tested margin. QLA `k = 2` and `k = 3` still
  fail once the two peaks survive rounding (and at some sub-bin
  margins), because one-hot gap completion picks the first peak.
  Fixed-scale MS-QLA succeeds at sub-bin pair margins by emitting the
  uniform vector on the collapsed all-tie; gap-adaptive MS-QLA keeps
  the two peaks and puts mass `1/2` on each (`ArgmaxUniform` for
  `k >= 2`).
- ADA2 corridor, same unique needle, values one-hot at index 3,
  `query_index = n - 1` for `CausalArgmaxRetrievalTask` and raw scores
  as affinities for `CopyTokenAttentionTask`: softmax, sparsemax, and
  gap-adaptive MS-QLA (`k = 1, 2, 3`) grade `Provisional` at every
  margin; QLA and fixed-scale MS-QLA grade `Falsified` at margins 0.1
  and 0.4. A causal cut at `query_index = 2` (needle hidden) is
  `Provisional` for the visible-prefix softmax decoder at every
  margin — the corridor is lane-separated retrieval, not a claim that
  MS-QLA implements masking internally.

Hand fixtures:

- In-domain unit spike `[1, 0, 0, 0, 0]` at `k = 2`, scale `1`: lattice
  hit, matches both `lattice_entmax_i64` and the A4 1.5-entmax oracle.
- Two-way tie `[0, 0]` at `k = 2`: QLA gap-completes to a one-hot on
  the first index; MS-QLA returns `ArgmaxUniform` `(1/2, 1/2)`.
- Equal-pair interpolation `scores = [2, 2, 0, 0]`, values
  `[+1, -1, 4, -4]`: softmax, sparsemax, f64 1.5-entmax, and MS-QLA
  `k = 2` and `k = 3` decode to `0`; QLA `k = 2` and `k = 3` decode to
  `+1`.
- Sub-bin needle of margin `0.2` at scale floor `1`: fixed MS-QLA
  `k = 1` still sees a four-way tie (hit@1 picks index 0); gap-adaptive
  MS-QLA raises the scale to `2.5` and recovers index 1.
- The same needle of margin `0.4` at `k = 2` with gap adaptation
  becomes `[0, -1, ...]` after rounding, misses the lattice domain, and
  plateau-completes to a one-hot on the needle.

Adversarial inputs (empty vector, `k` outside `1..=16`, non-positive or
non-finite scale, non-finite scores, overflow-scale, a subnormal
plateau gap that sends `1/2 / gap` to infinity) fail closed.

## Verdict

**ADAPT (research reference).** Totality and the stated oracle parities
hold on the tested domain. Gap-adaptive MS-QLA recovers the two
mechanistic tasks that rejected QLA's drop-in claim, and the unique
needle additionally grades provisionally through the ADA2 causal-argmax
and copy-token fixtures. The recovery depends on *both* ingredients:
argmax-preserving completion alone does not save sub-bin needles, and
gap adaptation alone does not save `k >= 2` tied pairs (the integer
lattice has no two-point `k = 2` support). Gap adaptation increases L1
against the unquantized operator. No usefulness, novelty, hardware, or
FLAT-adoption claim.

## Open

A reconstructed solver with target `(k * scale)^k` and
`p_i = (q_i - u)^k / (k * scale)^k` (true `q / scale` entmax); a search
for in-domain completions other than the equal-plateau / uniform
projector (none exist for two-point `k = 2` support); traces of real
attention scores; composition of MS-QLA summaries.
