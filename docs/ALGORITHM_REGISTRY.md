# ADA Algorithm Mission Registry

| ID | Mission | Current status |
| --- | --- | --- |
| ADA-A1 | Exact Online Softmax recurrence search | CPU-L2-QUALIFIED / GPU-Q4-BRANCH-REJECTED / ADAPT |
| ADA-A2 | K-first / V-late staging and scheduling | PLANNED |
| ADA-A3 | Certified error-budgeted Softmax | PLANNED |
| ADA-A4 | Exact Entmax branch-and-bound | RESEARCH |
| ADA-A5 | Hierarchical safe Pre-KV bounds | RESEARCH |
| ADA-A6 | Specialized tau solvers | RESEARCH |
| ADA-A7 | Moment / composable Entmax | INVESTIGATE |
| ADA-A8 | Attention recurrence program synthesis | PLANNED |
| ADA-A9 | Distribution-aware execution selection | PLANNED |
| ADA-A10 | Reproducible numerical oracle/certification | PLANNED |

## Status semantics

- `CPU-L2-QUALIFIED` means the candidate has passed the ADA CPU L2 evidence protocol on a named physical target with a clean Git tree, fixed power/frequency context, CPU affinity, correctness gates, repeated processes, raw evidence, and a SHA-256 digest.
- `GPU-Q4-BRANCH-REJECTED` means the branch-specialized Q4 GPU realization was correct but slower than the qualified Q4 baseline across the first physical-Thor timestamp smoke matrix. This rejects that realization as a performance candidate; it does not invalidate the underlying exact recurrence.
- `ADAPT` means the mission remains active only through a materially different mapping that preserves the mathematical contract, such as a branchless one-exp formulation; the rejected mapping must not be silently re-labelled as qualified.
- None of these statuses means production-qualified, novel, or adopted by FLAT-ATTENTION.

Statuses are research administration only; they are not claims of novelty or feasibility.
