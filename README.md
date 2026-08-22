# ADA — Algorithm Discovery for Attention

ADA is a deterministic research bench for discovering, falsifying, qualifying, and benchmarking attention algorithms.

ADA is **not** a production attention runtime and it does not silently modify FLAT-ATTENTION or SciRust. Candidates become interesting only after they survive mathematical, numerical, adversarial, and hardware evidence gates.

## Core rule

> Generation is cheap. Evidence is expensive.

The qualification pipeline is:

`Generate → Prove/Falsify → Oracle → Adversarial Tests → Cost → Hardware Benchmark → ADOPT / ADAPT / REJECT`

A candidate that appears absent from prior art may additionally enter `DISCOVER`, which requires a dedicated prior-art review before any novelty claim.

## Initial mission

The first mission is **ADA-A1: exact Online Softmax recurrence search**. The initial hand-seeded candidate is a branch-specialized recurrence that evaluates one non-trivial exponential per score after initialization, rather than the baseline online recurrence's two candidate exponentials per subsequent score.

The first bench is intentionally CPU/reference-only. GPU kernels and FLAT integration are promotion stages, not assumptions.

## Workspace

- `crates/ada-core` — shared contracts and logical operation metrics.
- `crates/ada-oracle` — deterministic reference implementations and candidate algorithms.
- `crates/ada-runner` — reproducible local benchmark/falsification runner.
- `crates/ada-research-engine` — deterministic grammar search, evidence gates,
  Pareto ranking, and replayable ARE-E0 archives.
- `docs/ADA_SPEC.md` — architecture and research invariants.
- `docs/RESEARCH_PROTOCOL.md` — evidence and promotion rules.
- `docs/ALGORITHM_REGISTRY.md` — ADA-A1…A10 mission registry.
- `docs/ADA_RESEARCH_ENGINE_E0.md` — ARE-E0 architecture and first automated
  discovery experiment.
- `docs/ADA_SCIRUST_ALGOGEN_GAP_ANALYSIS.md` — SciRust reuse and adapter analysis.

## Quick start

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p ada-runner --release
```

Hardware performance claims require recorded device evidence; logical operation counts are not physical bandwidth or instruction counts.
