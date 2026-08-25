# ADA Algorithm Mission Registry

| ID | Mission | Current status |
| --- | --- | --- |
| ADA-A1 | Exact Online Softmax recurrence search | CPU-L2-QUALIFIED / GPU-Q4-DIRECT-MAPPINGS-REJECTED / NVIDIA-BACKEND-INVESTIGATE |
| ADA-A2 | K-first / V-late staging and scheduling | E0-K-FIRST-V-LATE-CORRECTNESS / E1-NATURAL-LOGICAL-KV-ACCOUNTING-QUALIFIED / E2-THOR-PHYSICAL-V-LATE-QUALIFIED / E3A-NATURAL-GQA-UNIQUE-V-ROW-ACCOUNTING-QUALIFIED / E3B-NATURAL-GQA-V-OUTPUT-CORRECTNESS-QUALIFIED |
| ADA-A3 | Certified error-budgeted Softmax | E0-CERTIFIED-BUDGET-CORRECTNESS (`ada-a3-certified-softmax`) |
| ADA-A4 | Exact Entmax branch-and-bound | CPU-E0-CORRECTNESS / E1-QK-BOX-CORRECTNESS / E2-SYNTHETIC-SURVEY-QUALIFIED |
| ADA-A5 | Hierarchical safe Pre-KV bounds | E0-HIERARCHICAL-BOUND-CORRECTNESS / E1-CONTIGUOUS-HIERARCHY-SURVEY-QUALIFIED / E2-CONTENT-AWARE-HYBRID-CORRECTNESS / E3-THREE-WAY-SYNTHETIC-SURVEY-QUALIFIED / E4-TRACE-CONTRACT-CORRECTNESS / E4-NATURAL-QK-SLICE-QUALIFIED / E5-LAZY-COST-FRONTIER-MIXED / E5B-PRIORITY-FRONTIER-FOCUSED-NATURAL-QUALIFIED |
| ADA-A6 | Specialized tau solvers | E0-SPARSEMAX-SORTED-RESEARCH (`ada-a6-tau-solvers`) |
| ADA-A7 | Moment / composable Entmax | INVESTIGATE |
| ADA-A8 | Attention recurrence program synthesis | E0-IR-AND-SEARCH-RESEARCH (`ada-ir` + `ada-search`) |
| ADA-A9 | Distribution-aware execution selection | E0-SIGNAL-RULES-RESEARCH (`ada-a9-plan-selector`) / E0-DISPATCH-PARITY (`ada-a9-dispatch`) |
| ADA-A10 | Reproducible numerical oracle/certification | E0-SCHEMA-VALIDATOR (`ada-a10-evidence-schema`) |

## Status semantics

- `E0-K-FIRST-V-LATE-CORRECTNESS` means the isolated A2 exact K-first/V-late scalar contract passed its declared local correctness gates: workspace fmt, strict crate/workspace Clippy, all workspace unit/doc tests, exhaustive small support-mask weighted-sum parity, integrated alpha 1.5 and alpha 2.0 A5-priority-to-A2 cases, dense Entmax/output parity, exact `V_loaded == final support size`, `K_loaded >= V_loaded`, safe dense-support fallback, rejection of non-finite values in actually loaded V rows, and a structural sentinel test showing that zero-probability V rows are skipped before their scalars are inspected. This qualifies source-level logical V-late semantics and accounting only. It is not a physical memory-traffic, cache, GPU, wall-clock, model-quality, production floating-point, or novelty claim.

- `E1-NATURAL-LOGICAL-KV-ACCOUNTING-QUALIFIED` means A2 replayed the frozen Qwen3 natural Q/K slice at the A5 E5b qualified page-size-16 / leaf-divisor-8 configuration for alpha 1.5 and 2.0, completing all 1,536 cases with dense Entmax parity, exact support containment inside the K-loaded set, and the exact decomposition `N = K_pruned + (K_loaded - V_loaded) + V_loaded`. Weighted K pruning was `0.708539` / `0.774436`, while only `0.015370` / `0.008344` of visible rows remained in the final logical V support. Among K-loaded rows, `0.947266` / `0.963008` had zero final probability and were therefore logically avoidable on a V-late path. This qualifies natural-trace logical row opportunity only. The trace does not measure V tensors or memory transactions, and GQA/cache/reuse effects are not modeled, so this status is not a bandwidth, GPU, wall-clock, production, or novelty claim.

- `E2-THOR-PHYSICAL-V-LATE-QUALIFIED` means the isolated three-level FullDense-V -> A5-KLoaded-V -> A2-Support-V CPU mechanism completed five independent pinned Thor processes with 1,530 aggregate records, exact V-output parity, zero natural anchor A2 non-wins, and a worst observed natural `G_A2_after_A5` of `4.882465x`. The `Support=KLoaded` control was centered at `1.000000x`. This qualifies only the synthetic V-access CPU wall-clock mechanism on the named Thor target. It does not measure PMU/DRAM bytes, natural GQA V traffic, GPU performance, end-to-end attention speedup, or production behavior.

- `E3A-NATURAL-GQA-UNIQUE-V-ROW-ACCOUNTING-QUALIFIED` means A2 replayed an all-16-query-head extension of the frozen Qwen3 natural Q/K trace and grouped the 3,072 records into 1,536 actual two-Q-head/one-KV-head GQA groups. The historical four-head E4 record stream reproduced byte-for-byte inside the expanded trace. After exact union/deduplication, weighted residual V-row avoidance inside the A5-loaded unique K set was `0.937599381` / `0.959596873` for alpha 1.5 / 2.0, while total unique-V-row avoidance was about `0.976793` / `0.987885`. GQA reduced the residual A2 percentage by only `0.709357` / `0.226990` percentage points relative to naive per-query counting. Six / thirteen groups had no residual A2 opportunity; all occurred in layer 27 after A5 had already reduced the unique K set to only 2 or 4 rows. All support-containment and union/intersection identities held. This status qualifies natural unique-row logical accounting only; it is not measured V traffic, bandwidth, wall-clock, GPU, production, model-quality, or novelty evidence.

- `E3B-NATURAL-GQA-V-OUTPUT-CORRECTNESS-QUALIFIED` means A2 joined the frozen all-16-head natural Q/K corpus with a deterministic pre-repeat-GQA Qwen3 V corpus containing 384 unique physical KV-head value matrices. Across 1,536 GQA groups, two alpha values, and 6,144 query-head/alpha output cases, the E3a unique-row accounting reproduced exactly. `A5-KLoadedV` and `A2-SupportV` were output-identical in every case. Maximum FullDenseV-versus-sparse L-infinity error was `3.37507799486047588e-14` for alpha 1.5 and exactly zero for alpha 2.0. Residual unique-V-row avoidance after A5 remained `0.937599381` / `0.959596873` for alpha 1.5 / 2.0. This qualifies natural exact V-output correctness and GQA-aware logical row omission only; it does not measure physical traffic, bandwidth, wall-clock, GPU performance, end-to-end speedup, production behavior, model quality beyond the tested attention outputs, or novelty.

- `CPU-L2-QUALIFIED` means the candidate has passed the ADA CPU L2 evidence protocol on a named physical target with a clean Git tree, fixed power/frequency context, CPU affinity, correctness gates, repeated processes, raw evidence, and a SHA-256 digest.
- `GPU-Q4-DIRECT-MAPPINGS-REJECTED` means two correctness-qualified direct Q4 GPU realizations of the exact one-exp recurrence were slower than the qualified Q4 baseline across the same physical-Thor GPU-timestamp smoke matrix: the branch-specialized mapping and the adapted steady-state branchless mapping. The branchless mapping recovered part of the branch penalty but still lost in all 12 cases. This rejects these direct same-geometry/same-staging Q4 mappings as performance candidates; it does not invalidate the exact recurrence, the CPU result, or materially different GPU implementations.
- `NVIDIA-BACKEND-INVESTIGATE` means the remaining A1 GPU work has been localized below the generic WGSL/Naga/SPIR-V optimization layer. Preserved evidence shows Q4 retains two static SPIR-V `Exp` instructions while A1B retains one both before and after `spirv-opt -O`, yet Q4 is faster; Vulkan executable statistics also show equal register count, shared-memory allocation, stack allocation, and subgroup size. Any further A1 GPU work therefore requires NVIDIA-specific backend/machine evidence rather than another unmotivated direct WGSL recurrence rewrite.
- `CPU-E0-CORRECTNESS` means the isolated A4 score-level subset-threshold branch-and-bound candidate has passed its declared local correctness gates on CPU: workspace fmt, strict clippy, all unit/doc tests, subset-threshold monotonicity, adversarial support preservation, dense-oracle parity, exhaustive small states for alpha=1.5 and alpha=2.0, and safe dense fallback under loose bounds. This is correctness qualification only, not a performance or production claim.
- `E1-QK-BOX-CORRECTNESS` means the A4 coordinate-box follow-on has passed its declared local correctness gates: query/key min-max page bounds dominate every dense page score in the tested mixed-sign and exhaustive fixtures; exact branch-and-bound parity and support preservation hold for alpha=1.5 and alpha=2.0; positive attention scaling is covered; loose boxes degrade safely; and strict crate/workspace Clippy plus unit/doc tests are green. This is still scalar CPU correctness evidence, not a claim that the boxes are tight enough to save useful work on model distributions.
- `E2-SYNTHETIC-SURVEY-QUALIFIED` means the deterministic A4 pruning survey ran from a clean committed tree, passed its build/test gates, completed all 126 declared synthetic cases, preserved no dense-support false negatives, and has raw evidence with a recorded SHA-256 digest. It qualifies the reported synthetic algorithmic pruning observations only. It is not a wall-clock benchmark, model-distribution qualification, hardware speedup claim, or production qualification.
- `E0-HIERARCHICAL-BOUND-CORRECTNESS` means the isolated A5 scalar hierarchy has passed fmt, strict crate/workspace Clippy, all A5 unit/doc tests, dense-oracle parity for alpha=1.5 and alpha=2.0, hierarchy-bound conservativeness/tightening checks, adversarial flat-box-vs-hierarchy pruning, safe all-leaf fallback, invalid-index rejection, and exhaustive small hierarchies at multiple leaf sizes. This qualifies scalar CPU correctness only. The current f64 dense-score cross-check remains an oracle-side safety check rather than a production directed-rounding proof.
- `E1-CONTIGUOUS-HIERARCHY-SURVEY-QUALIFIED` means the deterministic 378-case A5 flat-vs-contiguous-hierarchy survey ran from clean commit `bfcda58bfa6af390bb900c01eb18f8187c6a7843`, completed with dense-oracle parity/support preservation, and has preserved raw evidence with SHA-256 `dc79857dfd9dab8dfa06f33d90bb068cb5bf0a71cf0a42bfccc7f6509b365d30`. It qualifies a mixed result: contiguous refinement improves `page_clustered` logical score avoidance (up to about +0.067 mean at leaf divisor 8) but essentially fails the critical `iid_uniform` hypothesis (0 gain for alpha=1.5 and only +0.006696 for alpha=2.0 at divisor 8), while eager bound work rises with depth. This is synthetic algorithmic evidence, not a speedup claim.
- `E2-CONTENT-AWARE-HYBRID-CORRECTNESS` means the isolated A5 content-aware scalar follow-on has passed fmt, strict crate/workspace Clippy, all unit/doc tests, deterministic index construction, enclosing-ball/coordinate-box hybrid-bound conservativeness against the dense oracle, exact Entmax parity for alpha=1.5 and alpha=2.0, safe all-token fallback, an interleaved-cluster case where content-aware partitioning beats the contiguous tree, and exhaustive small content-aware trees. The candidate retains the exact A4 subset-threshold certificate. This is scalar CPU correctness only; its f64 ball/box checks are laboratory oracle safeguards rather than a production directed-rounding proof.
- `E3-THREE-WAY-SYNTHETIC-SURVEY-QUALIFIED` means the deterministic 378-case flat-vs-contiguous-vs-content-aware survey ran from clean commit `c71473e1e9a55c4ce0147a94502f43597e31040d`, completed with dense-oracle parity/support preservation, and has preserved raw evidence with SHA-256 `26c75aa53833935e1609d081d06b4457a1c76ea1d2ee7268c9a84b993c6599d2`. It qualifies a deliberately mixed synthetic result. On `page_clustered`, the content-aware hybrid candidate improves mean logical score avoidance over the contiguous tree by about +0.046 to +0.083 across the tested alpha/leaf settings and reduces node expansions/threshold solves in representative cases. On `dominant_page`, all methods are already near the same outer-page optimum. On `iid_uniform`, the critical generality hypothesis remains largely falsified: alpha=1.5 gets zero pruning at every tested granularity and alpha=2.0 reaches only 0.017113 mean avoidance at divisor 8 versus 0.006696 contiguous, despite the enclosing-ball bound beating the coordinate box on most nodes. This indicates that high-dimensional exact group upper bounds, not merely contiguous partition geometry, are the limiting mechanism on that synthetic regime. This is synthetic algorithmic evidence only, not a wall-clock, model-distribution, memory-traffic, hardware-speedup, or production claim.
- `E4-TRACE-CONTRACT-CORRECTNESS` means the `ADAQK01\0` binary trace parser/replay contract passed its declared Rust fmt, strict crate/workspace Clippy, parser unit tests, workspace unit/doc tests, and format documentation gates on the exact committed source. This qualifies the trace laboratory plumbing only; it is not real-model evidence.
- `E4-NATURAL-QK-SLICE-QUALIFIED` means the Qwen3 E4 capture/replay path has completed algorithmic qualification on the frozen 16-sample `Salesforce/wikitext` `wikitext-2-raw-v1` validation slice declared in `A5_E4_REAL_QK_TRACE_PROTOCOL.md`, using immutable `Qwen/Qwen3-0.6B` model/tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`. The preserved trace contains 768 real post-QK-normalization/post-RoPE attention-score-input records and the replay completed all 18,432 declared alpha/page/leaf comparisons with no dense-support false negative and only f64-scale dense-oracle differences (maximum observed probability difference about `2.998e-15`). In the strongest tested global setting, page size 16 with leaf divisor 8, contiguous hierarchy avoided about `0.639074` / `0.712097` of score work for alpha 1.5 / 2.0, while content-aware box-only reached about `0.660899` / `0.736760`. The benefit is not uniform: content-aware partitioning helps sampled early/middle layers and several heads but is slightly worse than contiguous hierarchy on sampled layer 27, supporting later ADA-A9 conditional plan selection. The enclosing-ball component changed score avoidance in exactly 0 of 18,432 cases, so the current ball bound is preserved as an ablation/prior-art reference but deprioritized as an execution mechanism. This status qualifies only the reported algorithmic observations on the frozen natural-text slice; it is not general Qwen3-distribution qualification, model-quality evidence, production floating-point certification, physical KV-traffic evidence, GPU viability, wall-clock speedup, or novelty.
- `E5-LAZY-COST-FRONTIER-MIXED` means the exact lazy contiguous A5 controller was evaluated on the frozen E4 natural Q/K trace across the full 18,432-case alpha/page/leaf matrix. It preserved exact historical pruning behavior while evaluating node bounds only on demand. Page size 16 / leaf divisor 8 was the sole tested global Pareto point under the declared logical score-pruning and bound-cost objectives, with weighted score avoidance about `0.708539` / `0.774436` and weighted bound avoidance about `0.167199` / `0.208171` for alpha 1.5 / 2.0. However, repeated Vec-frontier rescans produced about `80.490612` / `61.499173` logical bound requests per pruned token. This status records a mixed controller-cost result, not a speedup or hardware qualification.
- `E5B-PRIORITY-FRONTIER-FOCUSED-NATURAL-QUALIFIED` means the exact ordered priority-frontier follow-on was replayed on all 1,536 frozen natural Q/K cases at the E5 Pareto configuration, page size 16 / leaf divisor 8, for alpha 1.5 and 2.0. It matched the historical E5 loaded-token set and final Entmax distribution bitwise in every case, preserved identical score/bound avoidance, bound-evaluation count, node expansions, and threshold solves, and had no dense-support false negative. The conservative logical counter of bound evaluations plus declared frontier operations fell from historical E5 request ratios of about `80.490612` / `61.499173` per pruned token to `3.959471` / `3.406420`. This qualifies the algorithmic scheduling observation only. `BTreeSet` internal comparison/tree costs are not represented by the custom counter, so this is not a wall-clock, GPU, bandwidth, or production-speed claim.
- The rejected branch-specialized and branchless A1 mappings must remain preserved as negative evidence and must not be silently re-labelled as qualified.
- None of these statuses means production-qualified, novel, or adopted by FLAT-ATTENTION.

Statuses are research administration only; they are not claims of novelty or feasibility.

## 2026-08-25 follow-up 2: A6 candidate and E5c ablation harness

- `E0-SPARSEMAX-SORTED-RESEARCH` (ADA-A6): sorted-projection sparsemax
  (alpha = 2) as a specialized tau solver. Exhaustive parity against the
  canonical bisection oracle over all 3^n score states up to n=6 plus wide
  dynamics/tie cases; certified fallback to the A4 extreme semantics on
  degenerate magnitudes; fail-closed on invalid inputs.
- E5c geometry-ablation example (`e5c_geometry_survey`): replays either a
  frozen natural trace or a deterministic synthetic grid, comparing legacy
  pivot-diameter/mean-ball versus PCA-cut/shrunk-ball pruning fractions under
  an exactness gate (dense support must be loaded). First synthetic run:
  both geometries exact, identical pruning on strongly separated workloads;
  the natural-slice measurement awaits the `.adaqk` artifact.
## 2026-08-25 follow-up: fuzz coverage and end-to-end dispatch

- Four new libFuzzer harnesses (60k smoke execs each, no crashes):
  `ir_interpreter` (arbitrary programs must validate/interpret-finite or
  fail closed with typed errors), `a3_budget_softmax` (certified results
  respect their own bounds), `a10_evidence_record` (total validation),
  `a9_plan_selector` (precedence table cross-checked against an independent
  reimplementation).
- IR text codec (A8): canonical s-expression pretty-printing
  (`to_ir_text`) with a fail-closed parser (`from_ir_text`); constants are
  exact `0x`-prefixed bit patterns so round trips are bit-preserving.
  Round-trip and malformed-input tests included.
- A10 CLI: `a10-validate` validates evidence metadata sidecars (`key=value`)
  fail-closed — unknown/duplicate keys rejected, metrics must be finite —
  and is now invoked by `scripts/thor_a1_l2.sh` as a post-artifact schema
  gate (exit 4 on violation), leaving measurements untouched.
- `E0-DISPATCH-PARITY` (ADA-A9): the `ada-a9-dispatch` crate wires selector
  to controllers end-to-end. Integration tests prove plan parity across
  {dense, paged BnB, hierarchical, content-aware} on crossing workloads and
  certified Dense fallback in the degenerate-magnitude regime; an
  `e4_dispatch_replay` example replays frozen natural traces through the
  dispatcher wherever the `.adaqk` artifact is available.

## 2026-08-25 hardening and research additions

- `E0-CERTIFIED-BUDGET-CORRECTNESS` (ADA-A3): `budgeted_softmax` computes
  softmax/LSE in f64 compensated arithmetic, derives a rigorous relative-error
  certificate from documented per-operation bounds, and fails closed when the
  achieved bound exceeds the caller budget. Zero or negative budgets are
  rejected before any work; the achieved bound is a property of the
  computation, not of the requested budget.
- `E0-IR-AND-SEARCH-RESEARCH` (ADA-A8): `ada-ir` provides the restricted
  straight-line grammar (scalar state, comparisons, select/max, arithmetic,
  exp/log, reductions, broadcast/zip vector algebra, accumulate) with a
  fail-closed interpreter (non-finite intermediates rejected) and structural
  validation. `ada-search` instantiates the qualified two-exp / one-exp online
  recurrences as real IR programs and verifies them against the dense f64
  reference; candidates deviating beyond tolerance are rejected loudly.
  This is source-level research scaffolding only: no performance claim.
- `E0-SIGNAL-RULES-RESEARCH` (ADA-A9): deterministic plan selection over
  {dense, A4 paged BnB, A5 hierarchical, A5 content-aware} from measurable
  signals, including the certified degenerate-magnitude check mirroring the
  A4 collapse threshold. Thresholds trace to the qualified Thor evidence but
  the selector itself is unqualified for production dispatch.
- `E0-SCHEMA-VALIDATOR` (ADA-A10): dependency-free fail-closed validation of
  evidence-record metadata (identifier grammar, compact ISO-8601 timestamp,
  40-hex commit binding, 64-hex SHA-256 digest, finite metrics).
- Workspace hardening: key-index fingerprints upgraded to a dual-lane
  digest with length sentinel (`ada-core::KeyFingerprint`); post-hoc support
  exactness certificates added to all A5 controllers; honest bound-access
  counter in content-aware metrics; PCA-cut/shrunk-ball geometry variant for
  the content-aware hierarchy (conservative by construction); explicit NEON
  kernels isolated in the single unsafe-scoped crate `ada-a1-neon`; runner
  environment overrides that preserve byte-compatible default output;
  dependency-free A1 bench example (`cargo run --release -p ada-oracle
  --example bench_a1`).
