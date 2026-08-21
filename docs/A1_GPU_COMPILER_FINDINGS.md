# ADA-A1 GPU Compiler Findings

## Scope

This note records the mechanistic follow-up after the direct Q4 GPU mappings of ADA-A1 were correctness-qualified but slower than the qualified FLAT Q4 baseline on a physical Jetson AGX Thor.

It does not promote any GPU candidate and does not make a novelty claim.

## Physical Thor performance result

FLAT research branch: `research/ada-a1-one-exp-gpu`

Three-way timestamp harness revision:

`d7aacd1516a1cab8c8602df12c58116d704a27e0`

Physical target / driver:

- NVIDIA Tegra NVIDIA Thor
- Vulkan
- NVIDIA driver 580.00
- subgroup size 32
- GPU timestamp queries
- fixed MAXN / jetson_clocks context for the timing smoke

Across the 12-case matrix `(N in {32,128}) x (D in {8,64,128}) x (causal in {false,true})`:

- Q4 beat the branch-specialized A1 mapping in 12/12 cases.
- Q4 beat the branchless A1B mapping in 12/12 cases.
- A1B beat A1 in 12/12 cases.

Geometric-mean median ratios:

- `Q4 / A1 = 0.935899x`
- `Q4 / A1B = 0.950040x`
- `A1 / A1B = 1.015110x`

Interpretation: removing the dynamic steady-state branch recovered only about 1.5% relative to A1 and did not recover the remaining roughly 5% gap to Q4.

Raw three-way evidence SHA-256:

`7e21387230a483f642d1bab4b84da5649ab8d1a539e341476db0c0ee3216dda2`

## Vulkan pipeline executable probe

Probe revision:

`a61d224ed0ef00d2cfd75a8d41713e294a3a50ba`

Probe log SHA-256:

`5b57bb02c70a762a9cccf106881118bf233c63011685ec748337ba256b9d3b00`

The NVIDIA Vulkan driver reported one compute executable for each variant and the same apparent resource usage:

| Metric | Q4 | A1 branched | A1B branchless |
| --- | ---: | ---: | ---: |
| Register Count | 45 | 45 | 45 |
| Shared Memory Size | 11328 B | 11328 B | 11328 B |
| Stack Size | 0 | 0 | 0 |
| Subgroup Size | 32 | 32 | 32 |
| Driver Binary Size | 59776 B | 60032 B | 59904 B |

The driver exposed no internal representations through `VK_KHR_pipeline_executable_properties` (`internal_representation_count=0`).

The reported `Local Memory Size = 68719476736` was identical across all variants and is treated as non-comparative / suspect reporting rather than a per-thread local-memory conclusion.

Mechanistic consequence: there is no evidence here that the measured Q4 advantage is caused by a different register count, shared-memory allocation, stack allocation, or subgroup size.

The driver-binary-size order was:

`Q4 < A1B < A1`

which matches the observed performance order, but the binary-size differences are much smaller than the timing differences and therefore are not by themselves a causal explanation.

## Naga -> SPIR-V inspection

Preserved SPIR-V SHA-256 values:

- Q4: `ff20d02534e2e88f78bab2cf092956c967b9bd5ead7f1b47273d7d315c4af536`
- A1 branched: `3cb6e78c14c1526ecbbab69b2ba90caf1acc3aee4a92040d85b652ef5c7b6a49`
- A1B branchless: `d3be0625c1f6638113a9ce3b4d115163243383332a0402c60a1ee092160d31b6`

SPIR-V sizes:

| Variant | Size |
| --- | ---: |
| Q4 | 13848 B |
| A1 branched | 14364 B |
| A1B branchless | 14260 B |

Static key-op counts from `spirv-dis`:

| Op | Q4 | A1 branched | A1B branchless |
| --- | ---: | ---: | ---: |
| `Exp` | 2 | 2 | 1 |
| `FAbs` | 0 | 0 | 1 |
| `FMax` | 1 | 0 | 1 |
| `OpSelect` | 44 | 45 | 46 |
| `OpBranchConditional` | 42 | 44 | 43 |
| `OpPhi` | 0 | 0 | 0 |

Important distinction: the two static `Exp` instructions in A1 branched are in mutually exclusive control-flow paths, so the static SPIR-V instruction count is not a claim that both execute for every participating lane. The source-level A1 recurrence still has one nontrivial exponential on each non-first logical update.

The crucial result is A1B: it reaches the NVIDIA driver with one static SPIR-V `Exp`, while Q4 reaches the driver with two static SPIR-V `Exp`, yet Q4 is still faster on the measured Thor matrix.

Therefore Naga is not already collapsing Q4 to the same one-`Exp` SPIR-V form. The unexplained performance difference is localized below or within the NVIDIA compilation / scheduling layer (or to dependency structure presented to that compiler), not to a WGSL-to-SPIR-V elimination of the second Q4 exponential.

## Current falsified / weakened hypotheses

- **Falsified for these direct mappings:** halving the logical/static exponential count is sufficient to improve this Q4 GPU kernel on Thor.
- **Strongly weakened:** dynamic branch divergence is the dominant cause of the A1 slowdown. A1B removes the steady-state branch and recovers only a small part of the loss.
- **Not supported by the Vulkan statistics:** increased register count or shared-memory allocation explains the slowdown.

## Active hypotheses

The remaining serious mechanisms include:

1. NVIDIA backend optimization of the Q4 dependency graph despite two SPIR-V `Exp` instructions.
2. Better instruction-level parallelism / latency hiding in Q4 than in the one-`Exp` forms.
3. A longer serialized dependency chain in A1B (`sub -> abs -> exp -> compare/select -> recurrence update`).
4. Different lower-level instruction selection or scheduling for `select`, `abs`, comparisons, and control flow.
5. SFU/transcendental throughput not being the limiting resource for this kernel geometry.

## Next experiment

Run an offline `spirv-opt -O` pass on the three already-preserved SPIR-V modules, then disassemble and compare:

- optimized module size;
- static `Exp`, `FAbs`, `FMax`, `OpSelect`, `OpBranchConditional`, and `OpPhi` counts;
- whether generic SPIR-V optimization changes Q4 toward A1B or preserves the same qualitative structure.

If optimized SPIR-V still leaves Q4 with two `Exp` and A1B with one, then further explanation requires NVIDIA-specific lower-level evidence (for example cubin/SASS or a native NVIDIA profiling/compiler path), not another unmotivated WGSL recurrence variant.
