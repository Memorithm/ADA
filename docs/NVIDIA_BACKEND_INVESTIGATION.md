# ADA-A1 NVIDIA backend investigation

Status context: `GPU-Q4-DIRECT-MAPPINGS-REJECTED / NVIDIA-BACKEND-INVESTIGATE`.

## Question

Can the A1 online-softmax recurrence execute on the Jetson Thor's NVIDIA
iGPU with parity against the CPU oracle path?

## Finding

The sequential one-exp recurrence itself cannot map directly (the token
loop is inherently serial - consistent with the prior
`GPU-Q4-DIRECT-MAPPINGS-REJECTED` status). The standard two-pass
reformulation maps cleanly:

```
pass 1: m   = max_i s_i                       (grid reduction)
pass 2: p_i = expf(s_i - m)                   (elementwise)
        l   = sum p_i ; O[h] = sum p_i V[i][h] (block reductions + gemv)
epilogue: O /= l ; LSE = m + ln(l)
```

This is no longer "one-exp online" in the A1 sense: it evaluates one `exp`
per token unconditionally and performs three kernel launches, so the
algorithmic exp-count advantage (2n-1 -> n-1) is preserved but the
sequential-state advantage is not.

## Measurements (Thor sm=110, CUDA 13.0, driver 580.00)

Deterministic xorshift inputs, 5 rounds per shape, median event timing.
Artifact: `thor-sm110-two-pass-parity.txt`.

| shape      | max abs O dev-vs-host | max abs LSE dev-vs-host | host-vs-f64 LSE error | median ms |
| ---------- | --------------------: | ----------------------: | --------------------: | --------: |
| n=128 d=64 |          1.19e-07     |            9.54e-07     |  7.05e-07             |    0.0205 |
| n=512 d=128|          1.34e-07     |            9.54e-07     |  1.13e-06             |    0.0294 |
| n=2048 d=64|          1.34e-07     |            1.91e-06     |  9.58e-07             |    0.0725 |
| n=4096 d=128|         2.46e-07     |            1.91e-06     |  2.88e-06             |    0.1297 |

Device-vs-host deviations are the same order of magnitude as the float
host path's own distance to an f64 reference, i.e. no GPU-specific
numerical pathology was observed. Absolute times are microsecond-scale and
carry no CPU-comparison claim.

## Conclusions

1. Feasible: a two-pass CUDA mapping reproduces the CPU float results
   within ordinary f32 noise on Thor.
2. Not a drop-in: the mapping abandons the sequential one-exp state
   machine; it is a different algorithm with equal exp count but parallel
   structure. Any adoption must be argued on end-to-end latency/bandwidth,
   which is out of scope for this investigation.
3. The direct-mapping rejection stands; what this investigation adds is a
   parity-clean fallback formulation should a GPU execution target ever be
   required.

Build/run locally:

```sh
nvcc -O3 -arch=native -std=c++17 one_exp_cuda.cu -o one_exp_cuda
./one_exp_cuda
```
