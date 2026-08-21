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

## Non-negotiables

- No performance claim from operation counts alone.
- No GPU claim from CPU timings.
- No exactness claim for an approximation without a proof/certificate.
- No production mutation by the search process.
- No candidate can benchmark its way around failed correctness gates.
- Negative results are retained as evidence.

## ADA-A1 acceptance

The first mission validates the bench itself.

The one-exp recurrence must:

- match the baseline online Softmax over deterministic normal and adversarial fixtures within an explicit floating-point tolerance;
- preserve finite outputs and LSE;
- reduce the scalar logical `exp` count from `2n-1` to `n-1` for a query with `n` admissible keys;
- make no GPU speed claim until an isolated FLAT candidate is measured on real hardware.
