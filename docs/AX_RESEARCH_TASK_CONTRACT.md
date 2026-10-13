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

## Executable admission bridge

`ada_graduation::ecosystem_task` implements the ADA side of this boundary. It
consumes the reviewed SciRust Hub `TaskSpec` v1 and workspace-materialization
evidence shapes, requires an exact ADA commit/tree/content digest, a read-only
workspace, the `ada.research.evaluate.v1` capability, container-or-stronger
isolation and closed network policy, then emits an ElasticXxx
`elastic.task-resource-envelope@1.0.0` budget envelope.

The bridge is pinned to Hub commit
`01cf5f1e6d96573d57eef8d2aea22a5a8b0da02f` and ElasticXxx commit
`958ce6301730ceeec4bb1252ff12222e5bb7d431`. Contract changes require an
explicit adapter update and fixture review. Destination fixtures live under
`crates/ada-graduation/tests/fixtures/`.

Only `CONTINUE_RESEARCH` may pass this bridge. `REJECT` remains a preserved
negative result, while `ADOPT` and `ADAPT` fail closed until ADA has the
stronger correctness qualification already required by graduation policy.
Successful preflight remains a bounded plan input: it does not prove worker
capacity, resource enforcement, sandbox security, performance, adoption or
novelty.
