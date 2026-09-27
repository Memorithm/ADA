# AX-inspired research task boundary

ADA may generate and evaluate bounded algorithm candidates as declared tasks.
The declaration improves reproducibility; it does not turn an execution plan
into a qualified algorithm.

## Required identity

Every candidate task records the exact protocol revision, semantic identity,
implementation identity, workspace object IDs, seed domain, resource
envelope, evaluator revision and evidence references.

Semantic identity and implementation identity remain separate. A runtime
lowering, tile choice or backend change must not silently change the candidate
semantic.

## Gate order

```text
static validity → mathematical/numerical checks → adversarial checks
→ task quality → cost evidence → hardware evidence → promotion decision
```

REJECT and INCONCLUSIVE are valid outcomes. Holdout isolation and provenance
are mandatory even when a candidate is inexpensive to execute.

## Ownership

SciRust Hub controls task admission and provenance, RemoteOps controls
execution enforcement, and ElasticXxx controls adaptive resources. ADA owns
candidate generation, falsification and qualification records.
