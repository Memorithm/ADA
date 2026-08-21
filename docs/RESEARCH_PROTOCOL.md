# ADA Research Protocol

## Evidence hierarchy

1. **Definition** — specify the mathematical contract and admissible domain.
2. **Independent oracle** — provide a reference that does not call the candidate.
3. **Adversarial falsification** — target numerical and structural edge cases.
4. **Logical cost** — count algorithmic operations without presenting them as physical instructions or bandwidth.
5. **Experimental realization** — create an isolated implementation or kernel.
6. **Hardware evidence** — benchmark on a named physical target with a reproducible protocol.
7. **Prior-art review** — required before any novelty statement.
8. **Promotion** — ADOPT, ADAPT, REJECT, or DISCOVER.

## Hardware evidence levels

These levels are internal ADA qualification labels; they are not claims about external benchmark standards.

### L1 — exploratory hardware evidence

Requires:

- a named physical target;
- the exact ADA commit SHA;
- deterministic fixtures and explicit algorithmic metrics;
- raw timing output retained as evidence;
- enough system metadata to identify the machine and toolchain.

L1 may run under normal DVFS/governor behavior. It can establish that a hardware signal exists, but it is not sufficient for a strong reproducible performance claim.

### L2 — controlled hardware evidence

Requires all L1 properties plus:

- correctness gates before timing (`fmt`, strict `clippy`, tests);
- fixed power mode and fixed frequency on the measured CPU core;
- explicit CPU affinity for the timed process;
- multiple independent process runs, each retaining its internal repeated measurements;
- pre/post thermal and clock-state capture;
- raw evidence stored verbatim and addressed by SHA-256.

For Thor CPU ADA-A1 campaigns, `scripts/thor_a1_l2.sh` enforces MAXN, requires the selected core to have `scaling_min_freq == scaling_max_freq`, pins the release runner with `taskset`, and records repeated process runs. L2 CPU evidence still does **not** imply a GPU or FLAT kernel performance claim.

## Non-negotiables

- No performance claim from operation counts alone.
- No GPU claim from CPU timings.
- No exactness claim for an approximation without a proof/certificate.
- No production mutation by the search process.
- No candidate can benchmark its way around failed correctness gates.
- Negative results are retained as evidence.
- A blocked or administratively unavailable CI is not equivalent to a green CI; merge policy remains unchanged.

## ADA-A1 acceptance

The first mission validates the bench itself.

The one-exp recurrence must:

- match the baseline online Softmax over deterministic normal and adversarial fixtures within an explicit floating-point tolerance;
- preserve finite outputs and LSE;
- reduce the scalar logical `exp` count from `2n-1` to `n-1` for a query with `n` admissible keys;
- make no GPU speed claim until an isolated FLAT candidate is measured on real hardware.
