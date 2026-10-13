# Native AArch64 qualification

The `Native AArch64 qualification` workflow is the hardware evidence gate for
`ada-a1-neon`. It is distinct from the generic x86_64 CI and runs only on the
internal Jetson Thor runner labels:

`self-hosted, Linux, ARM64, scirust-arm64, jetson-thor, internal`.

## Security and reproducibility boundary

- Pull requests from forks never execute on the self-hosted runner.
- Checkout, Rust setup and artifact upload actions are pinned to full commit
  SHAs.
- The checked-out commit must equal the exact PR head or integrated push SHA.
- Rust 1.85.0, the workspace MSRV, and the committed `Cargo.lock` are required.
- The workflow refuses implicit source or lockfile mutation.

## Executed contract

The job lists and then runs the AArch64-only
`neon_parity_with_odd_head_dims_and_logits` test. Dimensions 1, 3 and 5
exercise scalar tails; dimensions 4 and 8 exercise full NEON lanes. Ascending,
descending, equal and singleton score sequences are compared with independent
scalar references. Clippy also runs with warnings denied.

A successful job is correctness evidence for that source SHA and recorded host;
it is not a performance, GPU, production-runtime, adoption or novelty claim.

## Retained evidence

Each run uploads a bounded, secret-free artifact named
`ada-native-aarch64-<source-sha>` containing:

- the exact source SHA, workflow identity and run identity;
- bounded CPU architecture/model/count and byte-order provenance;
- exact Rust/Cargo versions;
- SHA-256 identities for `Cargo.lock` and this workflow;
- the selected test list and qualification result;
- a relative SHA-256 manifest and a run-local Ed25519 detached signature.

The private signing key is destroyed before upload. Evidence from a previous
SHA must not be reused for a new candidate or integrated revision.
