# ADA Algorithm Mission Registry

| ID | Mission | Current status |
| --- | --- | --- |
| ADA-A1 | Exact Online Softmax recurrence search | CPU-L2-QUALIFIED / GPU-Q4-DIRECT-MAPPINGS-REJECTED / NVIDIA-BACKEND-INVESTIGATE |
| ADA-A2 | K-first / V-late staging and scheduling | PLANNED |
| ADA-A3 | Certified error-budgeted Softmax | PLANNED |
| ADA-A4 | Exact Entmax branch-and-bound | CPU-E0-CORRECTNESS / E1-QK-BOX-CORRECTNESS / E2-SYNTHETIC-SURVEY-QUALIFIED |
| ADA-A5 | Hierarchical safe Pre-KV bounds | E0-HIERARCHICAL-BOUND-CORRECTNESS / E1-FLAT-VS-HIERARCHY-SURVEY-CANDIDATE |
| ADA-A6 | Specialized tau solvers | RESEARCH |
| ADA-A7 | Moment / composable Entmax | INVESTIGATE |
| ADA-A8 | Attention recurrence program synthesis | PLANNED |
| ADA-A9 | Distribution-aware execution selection | PLANNED |
| ADA-A10 | Reproducible numerical oracle/certification | PLANNED |

## Status semantics

- `CPU-L2-QUALIFIED` means the candidate has passed the ADA CPU L2 evidence protocol on a named physical target with a clean Git tree, fixed power/frequency context, CPU affinity, correctness gates, repeated processes, raw evidence, and a SHA-256 digest.
- `GPU-Q4-DIRECT-MAPPINGS-REJECTED` means two correctness-qualified direct Q4 GPU realizations of the exact one-exp recurrence were slower than the qualified Q4 baseline across the same physical-Thor GPU-timestamp smoke matrix: the branch-specialized mapping and the adapted steady-state branchless mapping. The branchless mapping recovered part of the branch penalty but still lost in all 12 cases. This rejects these direct same-geometry/same-staging Q4 mappings as performance candidates; it does not invalidate the exact recurrence, the CPU result, or materially different GPU implementations.
- `NVIDIA-BACKEND-INVESTIGATE` means the remaining A1 GPU work has been localized below the generic WGSL/Naga/SPIR-V optimization layer. Preserved evidence shows Q4 retains two static SPIR-V `Exp` instructions while A1B retains one both before and after `spirv-opt -O`, yet Q4 is faster; Vulkan executable statistics also show equal register count, shared-memory allocation, stack allocation, and subgroup size. Any further A1 GPU work therefore requires NVIDIA-specific backend/machine evidence rather than another unmotivated direct WGSL recurrence rewrite.
- `CPU-E0-CORRECTNESS` means the isolated A4 score-level subset-threshold branch-and-bound candidate has passed its declared local correctness gates on CPU: workspace fmt, strict clippy, all unit/doc tests, subset-threshold monotonicity, adversarial support preservation, dense-oracle parity, exhaustive small states for alpha=1.5 and alpha=2.0, and safe dense fallback under loose bounds. This is correctness qualification only, not a performance or production claim.
- `E1-QK-BOX-CORRECTNESS` means the A4 coordinate-box follow-on has passed its declared local correctness gates: query/key min-max page bounds dominate every dense page score in the tested mixed-sign and exhaustive fixtures; exact branch-and-bound parity and support preservation hold for alpha=1.5 and alpha=2.0; positive attention scaling is covered; loose boxes degrade safely; and strict crate/workspace Clippy plus unit/doc tests are green. This is still scalar CPU correctness evidence, not a claim that the boxes are tight enough to save useful work on model distributions.
- `E2-SYNTHETIC-SURVEY-QUALIFIED` means the deterministic A4 pruning survey ran from a clean committed tree, passed its build/test gates, completed all 126 declared synthetic cases, preserved no dense-support false negatives, and has raw evidence with a recorded SHA-256 digest. It qualifies the reported synthetic algorithmic pruning observations only. It is not a wall-clock benchmark, model-distribution qualification, hardware speedup claim, or production qualification.
- `E0-HIERARCHICAL-BOUND-CORRECTNESS` means the isolated A5 scalar hierarchy has passed fmt, strict crate/workspace Clippy, all A5 unit/doc tests, dense-oracle parity for alpha=1.5 and alpha=2.0, hierarchy-bound conservativeness/tightening checks, adversarial flat-box-vs-hierarchy pruning, safe all-leaf fallback, invalid-index rejection, and exhaustive small hierarchies at multiple leaf sizes. This qualifies scalar CPU correctness only. The current f64 dense-score cross-check remains an oracle-side safety check rather than a production directed-rounding proof.
- `E1-FLAT-VS-HIERARCHY-SURVEY-CANDIDATE` means A5 has a deterministic synthetic survey intended to compare flat Q/K page boxes against hierarchical refinement on the exact A4-E2 fixture family, across multiple hierarchy leaf granularities. It reports token/score avoidance and hierarchy metadata work separately. Because the E0 implementation eagerly evaluates all hierarchy-node bounds, E1 is an algorithmic work survey and not a wall-clock or memory-traffic speedup claim. E1 is not qualified until its gates and first clean committed survey run are green and preserved.
- The rejected branch-specialized and branchless A1 mappings must remain preserved as negative evidence and must not be silently re-labelled as qualified.
- None of these statuses means production-qualified, novel, or adopted by FLAT-ATTENTION.

Statuses are research administration only; they are not claims of novelty or feasibility.
