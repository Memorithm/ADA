# ADA Algorithm Mission Registry

| ID | Mission | Current status |
| --- | --- | --- |
| ADA-A1 | Exact Online Softmax recurrence search | CPU-L2-QUALIFIED / GPU-Q4-DIRECT-MAPPINGS-REJECTED / NVIDIA-BACKEND-INVESTIGATE |
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
- `GPU-Q4-DIRECT-MAPPINGS-REJECTED` means two correctness-qualified direct Q4 GPU realizations of the exact one-exp recurrence were slower than the qualified Q4 baseline across the same physical-Thor GPU-timestamp smoke matrix: the branch-specialized mapping and the adapted steady-state branchless mapping. The branchless mapping recovered part of the branch penalty but still lost in all 12 cases. This rejects these direct same-geometry/same-staging Q4 mappings as performance candidates; it does not invalidate the exact recurrence, the CPU result, or materially different GPU implementations.
- `NVIDIA-BACKEND-INVESTIGATE` means the remaining A1 GPU work is mechanistic NVIDIA backend / machine scheduling investigation, not another direct WGSL mapping search. Preserved Naga->SPIR-V evidence and a generic `spirv-opt -O` pass both leave Q4 with two static `Exp` instructions and A1B with one, while Q4 remains faster. Vulkan executable statistics report the same register count, shared-memory allocation, stack allocation, and subgroup size for all three variants. Further explanation therefore requires NVIDIA-specific lower-level evidence such as native compiler/profiler output, cubin/SASS, or equivalent machine scheduling data.
- The rejected branch-specialized and branchless mappings must remain preserved as negative evidence and must not be silently re-labelled as qualified.
- None of these statuses means production-qualified, novel, or adopted by FLAT-ATTENTION.

Statuses are research administration only; they are not claims of novelty or feasibility.
