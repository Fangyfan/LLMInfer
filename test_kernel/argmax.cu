#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstdint>
#include <limits>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " code=" << static_cast<int>(err)                      \
                      << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int32_t SIZE = 151936;
constexpr int WARMUP_ITERS = 100;
constexpr int BENCH_ITERS = 10000;

// ============================================================
// CPU reference argmax
// tie-breaking rule:
// 1. larger value wins
// 2. if value equal, smaller index wins
// ============================================================
int32_t argmax_cpu(const float* input, int32_t size) {
    float max_val = -std::numeric_limits<float>::infinity();
    int32_t max_idx = -1;

    for (int32_t i = 0; i < size; ++i) {
        float v = input[i];
        if (max_idx == -1 || max_val < v) {
            max_val = v;
            max_idx = i;
        } else if (max_val == v && i < max_idx) {
            max_idx = i;
        }
    }

    return max_idx;
}

double benchmark_cpu_ms(const std::vector<float>& input, int bench_iters) {
    volatile int32_t sink = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < bench_iters; ++i) {
        sink = argmax_cpu(input.data(), static_cast<int32_t>(input.size()));
    }

    auto end = std::chrono::high_resolution_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return total_ms / bench_iters;
}

// ============================================================
// single_block
// ============================================================
namespace single_block {

template <int32_t WARP_SIZE>
static __device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
        int32_t other_idx = __shfl_down_sync(0xffffffff, idx, delta, WARP_SIZE);

        if (other_idx != -1) {
            if (idx == -1 || val < other_val) {
                val = other_val;
                idx = other_idx;
            } else if (val == other_val && idx > other_idx) {
                idx = other_idx;
            }
        }
    }
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ void block_reduce_argmax(float& val, int32_t& idx) {
    __shared__ float shared_vals[WARP_NUM];
    __shared__ int32_t shared_idxs[WARP_NUM];

    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;

    warp_reduce_argmax<32>(val, idx);
    if (lane == 0) {
        shared_vals[warp] = val;
        shared_idxs[warp] = idx;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
            idx = shared_idxs[lane];
        }
        warp_reduce_argmax<WARP_NUM>(val, idx);
    }
}

template <int32_t BLOCK_DIM>
static __global__ void argmax_kernel(
    const float* __restrict__ input,
    int32_t size,
    int32_t* __restrict__ output
) {
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    float val = -INFINITY;
    int32_t idx = -1;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        float other_val = input[i];
        if (val < other_val) {
            val = other_val;
            idx = i;
        }
    }
    block_reduce_argmax<WARP_NUM>(val, idx);

    if (threadIdx.x == 0) {
        *output = idx;
    }
}

void argmax(const float* input, int32_t size, int32_t* output, cudaStream_t stream) {
    dim3 gridDim(1);
    dim3 blockDim(256);
    argmax_kernel<256><<<gridDim, blockDim, 0, stream>>>(input, size, output);
}

} // namespace single_block

// ============================================================
// multi_block_32
// ============================================================
namespace multi_block_32 {

template <int32_t WARP_SIZE>
static __device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
        int32_t other_idx = __shfl_down_sync(0xffffffff, idx, delta, WARP_SIZE);

        if (other_idx != -1) {
            if (idx == -1 || val < other_val) {
                val = other_val;
                idx = other_idx;
            } else if (val == other_val && idx > other_idx) {
                idx = other_idx;
            }
        }
    }
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ void block_reduce_argmax(float& val, int32_t& idx) {
    __shared__ float shared_vals[WARP_NUM];
    __shared__ int32_t shared_idxs[WARP_NUM];

    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;

    warp_reduce_argmax<32>(val, idx);
    if (lane == 0) {
        shared_vals[warp] = val;
        shared_idxs[warp] = idx;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
            idx = shared_idxs[lane];
        }
        warp_reduce_argmax<WARP_NUM>(val, idx);
    }
}

struct __align__(8) val_idx {
    float val;
    int32_t idx;
};

template <int32_t BLOCK_DIM>
static __global__ void argmax_kernel_1(
    const float* __restrict__ input,
    int32_t size,
    val_idx* __restrict__ temp
) {
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    float val = -INFINITY;
    int32_t idx = -1;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float other_val = input[i];
        if (val < other_val) {
            val = other_val;
            idx = i;
        }
    }
    block_reduce_argmax<WARP_NUM>(val, idx);

    if (threadIdx.x == 0) {
        temp[blockIdx.x] = {val, idx};
    }
}

static __global__ void argmax_kernel_2(
    const val_idx* __restrict__ temp,
    int32_t* __restrict__ output
) {
    float val = -INFINITY;
    int32_t idx = -1;
    int32_t tid = threadIdx.x << 2;

#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        val_idx other = temp[tid + i];
        if (val < other.val) {
            val = other.val;
            idx = other.idx;
        } else if (val == other.val && idx > other.idx) {
            idx = other.idx;
        }
    }
    warp_reduce_argmax<32>(val, idx);

    if (threadIdx.x == 0) {
        *output = idx;
    }
}

void argmax(
    const float* input,
    int32_t size,
    val_idx* temp,
    int32_t* output,
    cudaStream_t stream
) {
    dim3 gridDim(128);
    dim3 blockDim(256);
    argmax_kernel_1<256><<<gridDim, blockDim, 0, stream>>>(input, size, temp);
    argmax_kernel_2<<<1, 32, 0, stream>>>(temp, output);
}

} // namespace multi_block_32

// ============================================================
// multi_block_128
// ============================================================
namespace multi_block_128 {

template <int32_t WARP_SIZE>
static __device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
        int32_t other_idx = __shfl_down_sync(0xffffffff, idx, delta, WARP_SIZE);

        if (other_idx != -1) {
            if (idx == -1 || val < other_val) {
                val = other_val;
                idx = other_idx;
            } else if (val == other_val && idx > other_idx) {
                idx = other_idx;
            }
        }
    }
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ void block_reduce_argmax(float& val, int32_t& idx) {
    __shared__ float shared_vals[WARP_NUM];
    __shared__ int32_t shared_idxs[WARP_NUM];

    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;

    warp_reduce_argmax<32>(val, idx);
    if (lane == 0) {
        shared_vals[warp] = val;
        shared_idxs[warp] = idx;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
            idx = shared_idxs[lane];
        }
        warp_reduce_argmax<WARP_NUM>(val, idx);
    }
}

struct __align__(8) val_idx {
    float val;
    int32_t idx;
};

template <int32_t BLOCK_DIM>
static __global__ void argmax_kernel_1(
    const float* __restrict__ input,
    int32_t size,
    val_idx* __restrict__ temp
) {
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    float val = -INFINITY;
    int32_t idx = -1;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float other_val = input[i];
        if (val < other_val) {
            val = other_val;
            idx = i;
        }
    }
    block_reduce_argmax<WARP_NUM>(val, idx);

    if (threadIdx.x == 0) {
        temp[blockIdx.x] = {val, idx};
    }
}

template <int32_t BLOCK_DIM>
static __global__ void argmax_kernel_2(
    const val_idx* __restrict__ temp,
    int32_t* __restrict__ output
) {
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    val_idx cur = temp[threadIdx.x];
    float val = cur.val;
    int32_t idx = cur.idx;

    block_reduce_argmax<WARP_NUM>(val, idx);

    if (threadIdx.x == 0) {
        *output = idx;
    }
}

void argmax(
    const float* input,
    int32_t size,
    val_idx* temp,
    int32_t* output,
    cudaStream_t stream
) {
    dim3 gridDim(128);
    dim3 blockDim1(256);
    dim3 blockDim2(128);

    argmax_kernel_1<256><<<gridDim, blockDim1, 0, stream>>>(input, size, temp);
    argmax_kernel_2<128><<<1, blockDim2, 0, stream>>>(temp, output);
}

} // namespace multi_block_128

// ============================================================
// Benchmark wrappers
// ============================================================
struct BenchResult {
    std::string name;
    int32_t cpu_idx;
    int32_t gpu_idx;
    bool correct;
    float total_gpu_ms;
    float avg_gpu_us;
    double cpu_avg_us;
    double speedup;
};

BenchResult benchmark_single_block(
    const std::vector<float>& h_input,
    int32_t cpu_idx,
    double cpu_avg_ms
) {
    float* d_input = nullptr;
    int32_t* d_output = nullptr;
    int32_t h_output = -1;

    CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0xff, sizeof(int32_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        single_block::argmax(d_input, SIZE, d_output, stream);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    single_block::argmax(d_input, SIZE, d_output, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < BENCH_ITERS; ++i) {
        single_block::argmax(d_input, SIZE, d_output, stream);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    double gpu_avg_ms = static_cast<double>(total_ms) / BENCH_ITERS;

    return BenchResult{
        "single_block",
        cpu_idx,
        h_output,
        h_output == cpu_idx,
        total_ms,
        total_ms * 1000.0f / BENCH_ITERS,
        cpu_avg_ms * 1000.0,
        cpu_avg_ms / gpu_avg_ms
    };
}

BenchResult benchmark_multi_block_32(
    const std::vector<float>& h_input,
    int32_t cpu_idx,
    double cpu_avg_ms
) {
    float* d_input = nullptr;
    int32_t* d_output = nullptr;
    multi_block_32::val_idx* d_temp = nullptr;
    int32_t h_output = -1;

    CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_temp, 128 * sizeof(multi_block_32::val_idx)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0xff, sizeof(int32_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        multi_block_32::argmax(d_input, SIZE, d_temp, d_output, stream);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    multi_block_32::argmax(d_input, SIZE, d_temp, d_output, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < BENCH_ITERS; ++i) {
        multi_block_32::argmax(d_input, SIZE, d_temp, d_output, stream);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));

    double gpu_avg_ms = static_cast<double>(total_ms) / BENCH_ITERS;

    return BenchResult{
        "multi_block_32",
        cpu_idx,
        h_output,
        h_output == cpu_idx,
        total_ms,
        total_ms * 1000.0f / BENCH_ITERS,
        cpu_avg_ms * 1000.0,
        cpu_avg_ms / gpu_avg_ms
    };
}

BenchResult benchmark_multi_block_128(
    const std::vector<float>& h_input,
    int32_t cpu_idx,
    double cpu_avg_ms
) {
    float* d_input = nullptr;
    int32_t* d_output = nullptr;
    multi_block_128::val_idx* d_temp = nullptr;
    int32_t h_output = -1;

    CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_temp, 128 * sizeof(multi_block_128::val_idx)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0xff, sizeof(int32_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        multi_block_128::argmax(d_input, SIZE, d_temp, d_output, stream);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    multi_block_128::argmax(d_input, SIZE, d_temp, d_output, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < BENCH_ITERS; ++i) {
        multi_block_128::argmax(d_input, SIZE, d_temp, d_output, stream);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));

    double gpu_avg_ms = static_cast<double>(total_ms) / BENCH_ITERS;

    return BenchResult{
        "multi_block_128",
        cpu_idx,
        h_output,
        h_output == cpu_idx,
        total_ms,
        total_ms * 1000.0f / BENCH_ITERS,
        cpu_avg_ms * 1000.0,
        cpu_avg_ms / gpu_avg_ms
    };
}

// ============================================================
// Main
// ============================================================
int main() {
    int device_id = 3;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "CUDA device : " << prop.name << "\n";
    std::cout << "Size        : " << SIZE << "\n";
    std::cout << "Warmup iters: " << WARMUP_ITERS << "\n";
    std::cout << "Bench iters : " << BENCH_ITERS << "\n\n";

    std::vector<float> h_input(SIZE);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int32_t i = 0; i < SIZE; ++i) {
        h_input[i] = dist(rng);
    }

    // Optional: create a deterministic unique maximum to avoid accidental ties.
    h_input[SIZE / 3] = 10000.0f;

    int32_t cpu_idx = argmax_cpu(h_input.data(), SIZE);
    double cpu_avg_ms = benchmark_cpu_ms(h_input, BENCH_ITERS);

    std::vector<BenchResult> results;

    results.push_back(
        benchmark_single_block(h_input, cpu_idx, cpu_avg_ms)
    );

    results.push_back(
        benchmark_multi_block_32(h_input, cpu_idx, cpu_avg_ms)
    );

    results.push_back(
        benchmark_multi_block_128(h_input, cpu_idx, cpu_avg_ms)
    );

    std::cout << "CPU argmax index: " << cpu_idx << "\n";
    std::cout << "CPU avg time    : " << std::fixed << std::setprecision(6)
              << cpu_avg_ms * 1000.0 << " us\n\n";

    std::cout << std::left
              << std::setw(20) << "Kernel"
              << std::setw(12) << "Correct"
              << std::setw(12) << "CPU idx"
              << std::setw(12) << "GPU idx"
              << std::setw(18) << "Total GPU ms"
              << std::setw(18) << "Avg GPU us"
              << std::setw(12) << "Speedup"
              << "\n";

    std::cout << std::string(104, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(20) << r.name
                  << std::setw(12) << (r.correct ? "true" : "false")
                  << std::setw(12) << r.cpu_idx
                  << std::setw(12) << r.gpu_idx
                  << std::setw(18) << r.total_gpu_ms
                  << std::setw(18) << r.avg_gpu_us
                  << std::setw(12) << r.speedup
                  << "\n";
    }

    return 0;
}