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
- `crates/ada-search` — historical A8 recurrence fixtures plus bounded,
  cost-ordered semantic generation, canonical deduplication, statistics, and
  checkpoint/resume infrastructure.
- `crates/ada-workload` — versioned, validated geometry and experiment-mode
  contracts, with an explicit adapter for historical A1 fixtures.
- `crates/ada-semantic` — bounded executable semantic programs with an
  independent f64 reference evaluator, canonical text, and stable identity
  fingerprints.
- `crates/ada-cegis` — bounded CEGIS orchestration with deterministic fixture
  checks, adversarial counterexample insertion, survivor re-evaluation, and
  persisted rejection artifacts.
- `crates/ada-objective` — typed multi-objective vectors and a deterministic
  Pareto archive that keeps correctness, numerical, logical, estimated,
  measured, and task-quality dimensions separate.
- `crates/ada-implementation` — backend-neutral implementation, schedule, and
  memory IR. It binds concrete implementation identities to existing semantic
  identities while keeping tiles, partitioning, reductions, buffering, memory
  placement, and paging metadata outside semantic identity.
- `docs/ADA_SPEC.md` — architecture and research invariants.
- `docs/RESEARCH_PROTOCOL.md` — evidence and promotion rules.
- `docs/ALGORITHM_REGISTRY.md` — ADA-A1…A10 mission registry.
- `docs/RESEARCH_CAPABILITY_MATRIX.md` — conservative boundary of executable
  research capabilities.

## Quick start

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p ada-runner --release
```

## Verification ladder

Beyond the quick-start gates, the workspace is qualified with:

- `cargo miri test` — UB and arithmetic-overflow detection on the oracle and
  trace-parser crates (the one IEEE-rounding-contract test in `ada-a4-entmax-bnb`
  is ignored under Miri because Miri's soft-float `powf` does not preserve
  `powf(x, 1.0) == x`; see `docs/A4_EXACT_ENTMAX_BNB.md`).
- `cargo fuzz` — ASan-backed libFuzzer campaigns over both binary trace
  parsers (`ADAQK01`, `ADAV01`) and a differential A1 softmax parity target;
  harnesses live under `fuzz/`.
- `cargo audit` / `cargo deny check advisories sources` — supply-chain gates.
  The workspace has no third-party runtime dependencies.

Hardware performance claims require recorded device evidence; logical operation counts are not physical bandwidth or instruction counts.
