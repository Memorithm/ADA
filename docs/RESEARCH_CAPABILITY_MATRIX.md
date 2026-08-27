# ADA research capability matrix

This matrix is deliberately conservative. A capability is marked yes only
when the repository has an executable, validated, tested path for it. A field
in the workload contract is not, by itself, an oracle, a searchable semantic,
or hardware evidence.

The current slices add versioned geometry and experiment-mode metadata, a
small executable semantic IR, and a bounded cost-ordered semantic generator
with canonical deduplication and checkpoint/resume. The semantic reference
path currently covers single-batch, single-head, explicit-Q/K/V, full-KV,
row-major f64 reference execution for scaled dot products, masking, bounded
selection, softmax or signed weighting, and weighted value mixing. Generation
is now consumable by a bounded generic CEGIS runner that checks explicit seed
fixtures, asks a caller-owned adversarial generator for bounded fixtures,
re-tests prior survivors, and retains candidate/counterexample artifacts. The
semantic reference path and CEGIS runner still make no novelty or hardware
claim. They do not turn any declared low-precision, latent-KV, recurrent,
paged, or distributed field into an implementation, and they do not add
implementation schedules to semantic identity.

| Research family | Semantic | Reference oracle | Searchable | Forward | Backward | Prefill | Decode | GQA/MQA | Paged KV | Low precision | Distributed | Hardware evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exact dense / online softmax and FlashAttention-style blocking | partial | partial | partial | partial | no | partial | no | no | no | no | no | L1 |
| Static block sparse attention | no | no | no | no | no | no | no | no | no | no | no | none |
| Dynamic sparse pattern selection | no | no | no | no | no | no | no | no | no | no | no | none |
| Hierarchical / trainable sparse attention | no | no | no | no | no | no | no | no | no | no | no | none |
| Routed block selection | no | no | no | no | no | no | no | no | no | no | no | none |
| MQA / GQA | partial | no | no | no | no | partial | partial | partial | partial | no | no | none |
| Latent / compressed KV | partial | no | no | no | no | no | partial | no | partial | partial | no | none |
| Recurrent / delta-rule linear attention | partial | no | no | no | no | no | no | no | no | no | no | none |
| Hybrid full / linear attention | no | no | no | no | no | no | no | no | no | no | no | none |
| Paged attention / block-table decode | partial | no | no | no | no | no | partial | partial | partial | no | no | none |
| Low-precision attention | partial | no | no | no | no | partial | partial | no | no | partial | no | none |
| Distributed / ring-style block attention | no | no | no | no | no | no | no | no | no | no | partial | none |

## Interpretation

- partial means that a bounded contract, executable semantic/reference slice,
  or historical fixture exists, but the complete family-specific
  executable/reference/search/evidence path is not present.
- L1 means existing ADA evidence or source-level investigation, not a general
  hardware qualification. It does not imply a speedup for any unmeasured
  device or workload.
- Existing A1-A10 fixtures remain authoritative for their original contracts.
  The historical A1 adapter in ada-workload is explicitly tagged as
  precomputed scalar logits and is not evidence that A1 has a general Q/K/V
  oracle.

## Current boundary

The next layers are intentionally separate:

    validated workload geometry
            ↓
    executable semantic IR
            ↓
    bounded search + CEGIS / oracle / differential evidence
            ↓
    implementation + schedule/memory IR
            ↓
    measured backend evidence

Adding an enum or metadata field must not move a row to yes. The row changes
only when the corresponding reference evaluator, validation tests, and
evidence protocol exist.
