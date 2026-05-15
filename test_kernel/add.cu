#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstdint>

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

constexpr int32_t SIZE = 2560;
constexpr int WARMUP_ITERS = 100;
constexpr int BENCH_ITERS = 10000;
constexpr float EPS = 1e-5f;

// ============================================================
// Kernel 1: fp32, block 256
// ============================================================
namespace add_fp32_b256 {

static __global__ void add_kernel(
    const float* in1,
    const float* __restrict__ in2,
    float* out,
    int32_t size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = in1[tid] + in2[tid];
    }
}

void add(
    const float* in1,
    const float* in2,
    float* out,
    int32_t size,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    add_kernel<<<gridDim, blockDim, 0, stream>>>(in1, in2, out, size);
}

} // namespace add_fp32_b256

// ============================================================
// Kernel 2: fp32, block 512
// ============================================================
namespace add_fp32_b512 {

static __global__ void add_kernel(
    const float* in1,
    const float* __restrict__ in2,
    float* out,
    int32_t size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = in1[tid] + in2[tid];
    }
}

void add(
    const float* in1,
    const float* in2,
    float* out,
    int32_t size,
    cudaStream_t stream
) {
    dim3 blockDim(512);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    add_kernel<<<gridDim, blockDim, 0, stream>>>(in1, in2, out, size);
}

} // namespace add_fp32_b512

// ============================================================
// Kernel 3: fp32 x4, block 256
// ============================================================
namespace add_fp32x4_b256 {

static __global__ void add_kernel(
    const float* in1,
    const float* __restrict__ in2,
    float* out,
    int32_t size
) {
    const int32_t tid = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    const float4 in1_v4 = reinterpret_cast<const float4*>(in1 + tid)[0];
    const float4 in2_v4 = reinterpret_cast<const float4*>(in2 + tid)[0];
    float4 out_v4 = make_float4(
        in1_v4.x + in2_v4.x, 
        in1_v4.y + in2_v4.y, 
        in1_v4.z + in2_v4.z, 
        in1_v4.w + in2_v4.w
    );

    if (tid + 3 < size) {
        reinterpret_cast<float4*>(out + tid)[0] = out_v4;
    } else {
        for (int i = tid; i < size; ++i) {
            out[i] = in1[i] + in2[i];
        }
    }
}

void add(
    const float* in1,
    const float* in2,
    float* out,
    int32_t size,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((size + 4 * blockDim.x - 1) / (4 * blockDim.x));
    add_kernel<<<gridDim, blockDim, 0, stream>>>(in1, in2, out, size);
}

} // namespace add_fp32x4_b256

// ============================================================
// Kernel 4: fp32 x4, block 128
// ============================================================
namespace add_fp32x4_b128 {

static __global__ void add_kernel(
    const float* in1,
    const float* __restrict__ in2,
    float* out,
    int32_t size
) {
    const int32_t tid = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    const float4 in1_v4 = reinterpret_cast<const float4*>(in1 + tid)[0];
    const float4 in2_v4 = reinterpret_cast<const float4*>(in2 + tid)[0];
    float4 out_v4 = make_float4(
        in1_v4.x + in2_v4.x, 
        in1_v4.y + in2_v4.y, 
        in1_v4.z + in2_v4.z, 
        in1_v4.w + in2_v4.w
    );

    if (tid + 3 < size) {
        reinterpret_cast<float4*>(out + tid)[0] = out_v4;
    } else {
        for (int i = tid; i < size; ++i) {
            out[i] = in1[i] + in2[i];
        }
    }
}

void add(
    const float* in1,
    const float* in2,
    float* out,
    int32_t size,
    cudaStream_t stream
) {
    dim3 blockDim(128);
    dim3 gridDim((size + 4 * blockDim.x - 1) / (4 * blockDim.x));
    add_kernel<<<gridDim, blockDim, 0, stream>>>(in1, in2, out, size);
}

} // namespace add_fp32x4_b128

// ============================================================
// CPU reference
// ============================================================
void add_cpu(
    const float* in1,
    const float* in2,
    float* out,
    int32_t size
) {
    for (int32_t i = 0; i < size; ++i) {
        out[i] = in1[i] + in2[i];
    }
}

// ============================================================
// Correctness check
// ============================================================
bool check_result(
    const std::vector<float>& ref,
    const std::vector<float>& got,
    float eps = EPS
) {
    if (ref.size() != got.size()) {
        return false;
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(ref[i] - got[i]);
        if (diff > eps) {
            std::cerr << "Mismatch at index " << i
                      << ", CPU = " << ref[i]
                      << ", GPU = " << got[i]
                      << ", diff = " << diff
                      << std::endl;
            return false;
        }
    }

    return true;
}

// ============================================================
// CPU benchmark
// ============================================================
double benchmark_cpu_ms(
    const std::vector<float>& in1,
    const std::vector<float>& in2,
    std::vector<float>& out,
    int iters
) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; ++i) {
        add_cpu(in1.data(), in2.data(), out.data(), static_cast<int32_t>(out.size()));
    }

    auto end = std::chrono::high_resolution_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return total_ms / iters;
}

// ============================================================
// GPU benchmark
// ============================================================
using AddFunc = void (*)(
    const float*,
    const float*,
    float*,
    int32_t,
    cudaStream_t
);

struct BenchResult {
    std::string name;
    bool correct;
    float total_kernel_ms;
    float avg_kernel_us;
    double speedup;
};

BenchResult benchmark_gpu(
    const std::string& name,
    AddFunc add_func,
    const std::vector<float>& h_in1,
    const std::vector<float>& h_in2,
    const std::vector<float>& h_ref,
    double cpu_avg_ms,
    int32_t size,
    int warmup_iters,
    int bench_iters
) {
    float* d_in1 = nullptr;
    float* d_in2 = nullptr;
    float* d_out = nullptr;

    std::vector<float> h_out(size, 0.0f);

    CUDA_CHECK(cudaMalloc(&d_in1, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in2, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in1, h_in1.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in2, h_in2.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, size * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // warmup
    for (int i = 0; i < warmup_iters; ++i) {
        add_func(d_in1, d_in2, d_out, size, stream);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // correctness check
    add_func(d_in1, d_in2, d_out, size, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool correct = check_result(h_ref, h_out);

    // benchmark kernel only
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < bench_iters; ++i) {
        add_func(d_in1, d_in2, d_out, size, stream);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_us = total_ms * 1000.0f / bench_iters;

    double gpu_avg_ms = static_cast<double>(total_ms) / bench_iters;
    double speedup = cpu_avg_ms / gpu_avg_ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_in1));
    CUDA_CHECK(cudaFree(d_in2));
    CUDA_CHECK(cudaFree(d_out));

    return BenchResult{
        name,
        correct,
        total_ms,
        avg_us,
        speedup
    };
}

// ============================================================
// Main
// ============================================================
int main() {
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "CUDA device: " << prop.name << std::endl;
    std::cout << "Size       : " << SIZE << std::endl;
    std::cout << "Warmup     : " << WARMUP_ITERS << std::endl;
    std::cout << "Bench iters: " << BENCH_ITERS << std::endl;
    std::cout << std::endl;

    std::vector<float> h_in1(SIZE);
    std::vector<float> h_in2(SIZE);
    std::vector<float> h_ref(SIZE);
    std::vector<float> h_cpu_out(SIZE);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int32_t i = 0; i < SIZE; ++i) {
        h_in1[i] = dist(rng);
        h_in2[i] = dist(rng);
    }

    add_cpu(h_in1.data(), h_in2.data(), h_ref.data(), SIZE);

    double cpu_avg_ms = benchmark_cpu_ms(
        h_in1,
        h_in2,
        h_cpu_out,
        BENCH_ITERS
    );

    std::vector<BenchResult> results;

    results.push_back(
        benchmark_gpu(
            "add_fp32_b256",
            add_fp32_b256::add,
            h_in1,
            h_in2,
            h_ref,
            cpu_avg_ms,
            SIZE,
            WARMUP_ITERS,
            BENCH_ITERS
        )
    );

    results.push_back(
        benchmark_gpu(
            "add_fp32_b512",
            add_fp32_b512::add,
            h_in1,
            h_in2,
            h_ref,
            cpu_avg_ms,
            SIZE,
            WARMUP_ITERS,
            BENCH_ITERS
        )
    );

    results.push_back(
        benchmark_gpu(
            "add_fp32x4_b256",
            add_fp32x4_b256::add,
            h_in1,
            h_in2,
            h_ref,
            cpu_avg_ms,
            SIZE,
            WARMUP_ITERS,
            BENCH_ITERS
        )
    );

    results.push_back(
        benchmark_gpu(
            "add_fp32x4_b128",
            add_fp32x4_b128::add,
            h_in1,
            h_in2,
            h_ref,
            cpu_avg_ms,
            SIZE,
            WARMUP_ITERS,
            BENCH_ITERS
        )
    );

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "CPU avg time: "
              << cpu_avg_ms * 1000.0
              << " us"
              << std::endl;

    std::cout << std::endl;

    std::cout << std::left
              << std::setw(20) << "Kernel"
              << std::setw(12) << "Correct"
              << std::setw(18) << "Total GPU ms"
              << std::setw(18) << "Avg GPU us"
              << std::setw(12) << "Speedup"
              << std::endl;

    std::cout << std::string(80, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(20) << r.name
                  << std::setw(12) << (r.correct ? "true" : "false")
                  << std::setw(18) << r.total_kernel_ms
                  << std::setw(18) << r.avg_kernel_us
                  << std::setw(12) << r.speedup
                  << std::endl;
    }

    return 0;
}