// ADA-A1 NVIDIA backend investigation (INVESTIGATE deliverable).
//
// The sequential one-exp recurrence cannot map directly to a GPU (token
// loop is inherently serial). The investigated mapping is the standard
// two-pass reformulation:
//
//   pass 1: m      = max_i s_i                (grid reduction)
//   pass 2: p_i    = expf(s_i - m); l = sum p; O[h] = sum p_i * V[i][h]
//   epilogue: O /= l ; LSE = m + ln(l)
//
// This program measures, on deterministic inputs:
//   1. device-vs-host parity (max |dO-hO|, |dLSE-hLSE|)
//   2. device-vs-f64-reference quality
//   3. median kernel wall time per shape
//
// It makes NO performance claim against the CPU evidence path; it only
// establishes mapping feasibility and numerical agreement.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                         cudaGetErrorString(err), __FILE__, __LINE__);  \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

struct Xorshift64 {
    unsigned long long state;
    explicit Xorshift64(unsigned long long seed) : state(seed) {}
    unsigned long long next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    float f32Signed() {
        unsigned int mantissa =
            static_cast<unsigned int>((next() >> 41) & 0x007fffffull);
        unsigned int bits = 0x3f800000u | mantissa;
        float unit;
        std::memcpy(&unit, &bits, sizeof(unit));
        unit -= 1.0f;
        return 2.0f * unit - 1.0f;
    }
};

__global__ void reduceMaxKernel(const float* logits, int n, float* outMax,
                                float* blockPartials) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    float value = -INFINITY;
    if (index < n) value = logits[index];
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) blockPartials[blockIdx.x] = shared[0];
}

__global__ void finalizeMaxKernel(float* blockPartials, int blocks,
                                  float* outMax) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float value = -INFINITY;
    if (tid < blocks) value = blockPartials[tid];
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) *outMax = shared[0];
}

__global__ void probsAndSumKernel(const float* logits, const float* maxPtr,
                                  int n, float* probs, float* partialSums) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    float maxLogit = *maxPtr;
    // Exact single-precision expf (the fast __expf approximation is
    // intentionally avoided for parity quality).
    float probability = 0.0f;
    if (index < n) {
        probability = expf(logits[index] - maxLogit);
        probs[index] = probability;
    } else {
        probs[index] = 0.0f;
    }
    shared[tid] = probability;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partialSums, shared[0]);
}

__global__ void outputKernel(const float* probs, const float* values, int n,
                             int headDim, const float* sumPtr, float* output,
                             float* lseOut, const float* maxPtr) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    float denominator = *sumPtr;
    if (column < headDim) {
        float accumulator = 0.0f;
        for (int i = 0; i < n; ++i) {
            accumulator += probs[i] * values[i * headDim + column];
        }
        output[column] = accumulator / denominator;
    }
    if (column == 0) {
        *lseOut = *maxPtr + logf(denominator);
    }
}

struct ShapeResult {
    int seqLen;
    int headDim;
    double hostVsDeviceO;
    double hostVsDeviceLse;
    double hostVsF64Lse;
    float msMedian;
};

int main() {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device=%s sm=%d%d multiprocessors=%d\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount);
    std::printf(
        "shape,n,max_abs_O_dev_vs_host,max_abs_LSE_dev_vs_host,"
        "max_abs_LSE_host_vs_f64,median_ms\n");

    const int kRounds = 5;
    const int kBlock = 256;

    for (int n : {128, 512, 2048, 4096}) {
        for (int headDim : {64, 128}) {
            Xorshift64 rng(0xADA0BEEFULL);
            std::vector<float> logits(n);
            for (auto& l : logits) l = 12.0f * rng.f32Signed();
            std::vector<float> values(static_cast<size_t>(n) * headDim);
            for (auto& v : values) v = rng.f32Signed();

            // Host reference in float (same formulas).
            float hostMax = *std::max_element(logits.begin(), logits.end());
            std::vector<float> hostProbs(n);
            float hostSum = 0.0f;
            for (int i = 0; i < n; ++i) {
                hostProbs[i] = std::exp(logits[i] - hostMax);
                hostSum += hostProbs[i];
            }
            std::vector<float> hostOutput(headDim);
            for (int h = 0; h < headDim; ++h) {
                float acc = 0.0f;
                for (int i = 0; i < n; ++i)
                    acc += hostProbs[i] * values[i * headDim + h];
                hostOutput[h] = acc / hostSum;
            }
            float hostLse = hostMax + std::log(hostSum);

            // f64 reference for quality context.
            double sum64 = 0.0;
            for (int i = 0; i < n; ++i)
                sum64 += std::exp(static_cast<double>(logits[i]) -
                                  static_cast<double>(hostMax));
            double lse64 = static_cast<double>(hostMax) + std::log(sum64);

            // Device buffers.
            float *dLogits, *dValues, *dProbs, *dMax, *dSum, *dLse, *dPartials,
                *dOutput;
            size_t valueBytes = sizeof(float) * n * headDim;
            CHECK(cudaMalloc(&dLogits, sizeof(float) * n));
            CHECK(cudaMalloc(&dValues, valueBytes));
            CHECK(cudaMalloc(&dProbs, sizeof(float) * (n + 1)));
            CHECK(cudaMalloc(&dMax, sizeof(float)));
            CHECK(cudaMalloc(&dSum, sizeof(float)));
            CHECK(cudaMalloc(&dLse, sizeof(float)));
            CHECK(cudaMalloc(&dOutput, sizeof(float) * headDim));

            int blocks = (n + kBlock - 1) / kBlock;
            CHECK(cudaMalloc(&dPartials, sizeof(float) * blocks));

            std::vector<float> deviceOutput(headDim);
            float deviceMax = 0.0f, deviceSum = 0.0f, deviceLse = 0.0f;

            cudaEvent_t start, stop;
            CHECK(cudaEventCreate(&start));
            CHECK(cudaEventCreate(&stop));
            std::vector<float> roundMs;

            for (int round = 0; round < kRounds; ++round) {
                CHECK(cudaMemcpy(dLogits, logits.data(), sizeof(float) * n,
                                 cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dValues, values.data(), valueBytes,
                                 cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dSum, 0, sizeof(float)));

                CHECK(cudaEventRecord(start));
                reduceMaxKernel<<<blocks, kBlock, kBlock * sizeof(float)>>>(
                    dLogits, n, dMax, dPartials);
                finalizeMaxKernel<<<1, kBlock, kBlock * sizeof(float)>>>(
                    dPartials, blocks, dMax);
                probsAndSumKernel<<<blocks, kBlock, kBlock * sizeof(float)>>>(
                    dLogits, dMax, n, dProbs, dSum);
                int outBlocks = (headDim + kBlock - 1) / kBlock;
                outputKernel<<<outBlocks, kBlock>>>(
                    dProbs, dValues, n, headDim, dSum, dOutput, dLse, dMax);
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CHECK(cudaEventElapsedTime(&ms, start, stop));
                roundMs.push_back(ms);

                CHECK(cudaMemcpy(deviceOutput.data(), dOutput,
                                 sizeof(float) * headDim,
                                 cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(&deviceMax, dMax, sizeof(float),
                                 cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(&deviceSum, dSum, sizeof(float),
                                 cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(&deviceLse, dLse, sizeof(float),
                                 cudaMemcpyDeviceToHost));
            }
            std::sort(roundMs.begin(), roundMs.end());
            float medianMs = roundMs[roundMs.size() / 2];

            double worstO = 0.0;
            for (int h = 0; h < headDim; ++h)
                worstO = std::max(worstO,
                                  std::abs(static_cast<double>(deviceOutput[h]) -
                                           static_cast<double>(hostOutput[h])));
            double worstLse = std::abs(static_cast<double>(deviceLse) -
                                       static_cast<double>(hostLse));
            double hostQuality = std::abs(static_cast<double>(hostLse) - lse64);

            (void)deviceMax;
            std::printf("%d,%d,%.9e,%.9e,%.9e,%.6f\n", n, headDim, worstO,
                        worstLse, hostQuality, medianMs);

            CHECK(cudaEventDestroy(start));
            CHECK(cudaEventDestroy(stop));
            cudaFree(dLogits); cudaFree(dValues); cudaFree(dProbs);
            cudaFree(dMax); cudaFree(dSum); cudaFree(dLse);
            cudaFree(dOutput); cudaFree(dPartials);
        }
    }
    return 0;
}
