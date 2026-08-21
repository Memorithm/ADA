# ADA Algorithm Mission Registry

| ID | Mission | Current status |
| --- | --- | --- |
| ADA-A1 | Exact Online Softmax recurrence search | CPU-L2-QUALIFIED / GPU-Q4-DIRECT-MAPPINGS-REJECTED / NVIDIA-BACKEND-INVESTIGATE |
| ADA-A2 | K-first / V-late staging and scheduling | PLANNED |
| ADA-A3 | Certified error-budgeted Softmax | PLANNED |
| ADA-A4 | Exact Entmax branch-and-bound | RESEARCH / E0-CANDIDATE |
| ADA-A5 | Hierarchical safe Pre-KV bounds | RESEARCH |
| ADA-A6 | Specialized tau solvers | RESEARCH |
| ADA-A7 | Moment / composable Entmax | INVESTIGATE |
| ADA-A8 | Attention recurrence program synthesis | PLANNED |
| ADA-A9 | Distribution-aware execution selection | PLANNED |
| ADA-A10 | Reproducible numerical oracle/certification | PLANNED |

## Status semantics

- `CPU-L2-QUALIFIED` means the candidate has passed the ADA CPU L2 evidence protocol on a named physical target with a clean Git tree, fixed power/frequency context, CPU affinity, correctness gates, repeated processes, raw evidence, and a SHA-256 digest.
- `GPU-Q4-DIRECT-MAPPINGS-REJECTED` means two correctness-qualified direct Q4 GPU realizations of the exact one-exp recurrence were slower than the qualified Q4 baseline across the same physical-Thor GPU-timestamp smoke matrix: the branch-specialized mapping and the adapted steady-state branchless mapping. The branchless mapping recovered part of the branch penalty but still lost in all 12 cases. This rejects these direct same-geometry/same-staging Q4 mappings as performance candidates; it does not invalidate the exact recurrence, the CPU result, or materially different GPU implementations.
- `NVIDIA-BACKEND-INVESTIGATE` means the remaining A1 GPU work has been localized below the generic WGSL/Naga/SPIR-V optimization layer. Preserved evidence shows Q4 retains two static SPIR-V `Exp` instructions while A1B retains one both before and after `spirv-opt -O`, yet Q4 is faster; Vulkan executable statistics also show equal register count, shared-memory allocation, stack allocation, and subgroup size. Any further A1 GPU work therefore requires NVIDIA-specific backend/machine evidence rather than another unmotivated direct WGSL recurrence rewrite.
- `E0-CANDIDATE` means an isolated, falsifiable research candidate and its oracle/tests have been specified on a dedicated branch, but no correctness or performance qualification is implied until the local gates and declared adversarial tests have run green.
- The rejected branch-specialized and branchless A1 mappings must remain preserved as negative evidence and must not be silently re-labelled as qualified.
- None of these statuses means production-qualified, novel, or adopted by FLAT-ATTENTION.

Statuses are research administration only; they are not claims of novelty or feasibility.
