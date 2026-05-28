// bench_qwen3_gemv_bf16_cublaslt.cu
// Independent benchmark for Qwen3-4B GEMV-related bf16 operators.
//
// Build example:
//   nvcc -O3 -std=c++17 -arch=sm_80 bench_qwen3_gemv_bf16_cublaslt.cu -lcublasLt -lcublas -o bench_qwen3_gemv_bf16_cublaslt
//
// Run example:
//   ./bench_qwen3_gemv_bf16_cublaslt --iters 200 --warmup 20 --workspace-mb 64
//   ./bench_qwen3_gemv_bf16_cublaslt --case qkv
//   ./bench_qwen3_gemv_bf16_cublaslt --skip-lm-head
//
// Notes:
//   1. This benchmark uses Qwen3-4B default dense config dimensions.
//   2. cuBLASLt path uses bf16 inputs and fp32 compute.
//   3. Row-major layouts are used to match hand-written weight layout [M, K].
//   4. Timings are CUDA event GPU times, not CPU wall-clock launch overhead.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLASLT_CHECK(expr) do { \
    cublasStatus_t _st = (expr); \
    if (_st != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLASLt error status " << static_cast<int>(_st) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

static uint16_t fp32_to_bf16_rn(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t rounding_bias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

static float bf16_to_fp32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

static void fill_bf16_device(void* d_ptr, size_t n, uint32_t seed, float scale = 0.02f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = fp32_to_bf16_rn(dist(rng));
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

static void fill_float_device(float* d_ptr, size_t n, uint32_t seed, float scale = 0.02f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

static void* cuda_malloc_bytes(size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}

template <int32_t WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
    }
    return val;
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(WARP_NUM > 0 && WARP_NUM <= 32, "WARP_NUM must be in [1, 32]");
    __shared__ float shared_vals[WARP_NUM];

    const int32_t lane = threadIdx.x & 31;
    const int32_t warp = threadIdx.x >> 5;

    val = warp_reduce_sum<32>(val);
    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < WARP_NUM) ? shared_vals[lane] : 0.0f;
        block_sum = warp_reduce_sum<32>(block_sum);
        if (lane == 0) {
            shared_vals[0] = block_sum;
        }
    }
    __syncthreads();
    return shared_vals[0];
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void manual_gemv_bf16_out_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ out,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = __float2bfloat16(sum);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void manual_gemv_fp32_out_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    float* __restrict__ out,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void manual_fused_gemv_add_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ residual_add,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);
    if (threadIdx.x == 0) {
        float residual = __bfloat162float(residual_add[blockIdx.x]);
        residual_add[blockIdx.x] = __float2bfloat16(residual + sum);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void manual_fused_qkv_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int32_t K,
    int32_t q_dim,
    int32_t kv_dim
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(input);
    const uint4* wei8 = reinterpret_cast<const uint4*>(weight + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        if (blockIdx.x < q_dim) {
            query[blockIdx.x] = __float2bfloat16(sum);
        } else if (blockIdx.x < q_dim + kv_dim) {
            key[blockIdx.x - q_dim] = __float2bfloat16(sum);
        } else {
            value[blockIdx.x - q_dim - kv_dim] = __float2bfloat16(sum);
        }
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void manual_fused_gate_up_swiglu_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ out,
    int32_t intermediate_dim,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* gate8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);
    const uint4* up8 = reinterpret_cast<const uint4*>(wei + (blockIdx.x + intermediate_dim) * K);

    float gate = 0.0f;
    float up = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = gate8[i];
        uint4 c = up8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
        const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&c);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            gate += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            gate += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
            up += __bfloat162float(a2[j].x) * __bfloat162float(c2[j].x);
            up += __bfloat162float(a2[j].y) * __bfloat162float(c2[j].y);
        }
    }
    gate = block_reduce_sum<WARP_NUM>(gate);
    up = block_reduce_sum<WARP_NUM>(up);

    if (threadIdx.x == 0) {
        float gate_silu = gate / (1.0f + __expf(-gate));
        out[blockIdx.x] = __float2bfloat16(gate_silu * up);
    }
}

static __global__ void split_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int32_t q_dim,
    int32_t kv_dim
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total = q_dim + 2 * kv_dim;
    if (idx >= total) return;
    if (idx < q_dim) {
        q[idx] = qkv[idx];
    } else if (idx < q_dim + kv_dim) {
        k[idx - q_dim] = qkv[idx];
    } else {
        v[idx - q_dim - kv_dim] = qkv[idx];
    }
}

static __global__ void swiglu_from_fp32_tmp_kernel(
    const float* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ out,
    int32_t intermediate_dim
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intermediate_dim) return;
    float gate = gate_up[idx];
    float up = gate_up[idx + intermediate_dim];
    float gate_silu = gate / (1.0f + __expf(-gate));
    out[idx] = __float2bfloat16(gate_silu * up);
}

struct LtGemvPlan {
    cublasLtHandle_t lt{};
    cublasLtMatmulDesc_t op_desc{};
    cublasLtMatrixLayout_t a_desc{};
    cublasLtMatrixLayout_t b_desc{};
    cublasLtMatrixLayout_t c_desc{};
    cublasLtMatrixLayout_t d_desc{};
    cublasLtMatmulAlgo_t algo{};
    bool has_algo = false;
    int32_t M = 0;
    int32_t K = 0;
    cudaDataType_t out_type = CUDA_R_16BF;
    float beta = 0.0f;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    LtGemvPlan(
        cublasLtHandle_t lt_,
        int32_t M_,
        int32_t K_,
        cudaDataType_t out_type_,
        float beta_,
        void* workspace_,
        size_t workspace_bytes_
    ) : lt(lt_), M(M_), K(K_), out_type(out_type_), beta(beta_), workspace(workspace_), workspace_bytes(workspace_bytes_) {
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

        // Row-major A[M, K] * B[K, 1] = D[M, 1].
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, M, K, K));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, K, 1, 1));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, out_type, M, 1, 1));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&d_desc, out_type, M, 1, 1));

        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

        cublasLtMatmulPreference_t pref;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)
        ));

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            lt,
            op_desc,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            pref,
            1,
            &heuristic,
            &returned
        ));
        CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));

        if (returned <= 0) {
            std::cerr << "No cuBLASLt heuristic algorithm found for M=" << M << " K=" << K << std::endl;
            std::exit(EXIT_FAILURE);
        }
        algo = heuristic.algo;
        has_algo = true;
    }

    ~LtGemvPlan() {
        if (d_desc) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    }

    void run(cudaStream_t stream, const __nv_bfloat16* W, const __nv_bfloat16* x, const void* C, void* D) const {
        const float alpha = 1.0f;
        const void* C_ptr = C ? C : D;
        CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            W,
            a_desc,
            x,
            b_desc,
            &beta,
            C_ptr,
            c_desc,
            D,
            d_desc,
            &algo,
            workspace,
            workspace_bytes,
            stream
        ));
    }
};

static float time_gpu_ms(cudaStream_t stream, int warmup, int iters, const std::function<void()>& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(iters);
}

struct DiffStats {
    float max_abs = 0.0f;
    double mean_abs = 0.0;
};

static DiffStats compare_bf16_device(const void* a, const void* b, size_t n) {
    std::vector<uint16_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    DiffStats s;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float da = bf16_to_fp32(ha[i]);
        float db = bf16_to_fp32(hb[i]);
        float d = std::fabs(da - db);
        s.max_abs = std::max(s.max_abs, d);
        sum += d;
    }
    s.mean_abs = sum / static_cast<double>(n);
    return s;
}

static DiffStats compare_fp32_device(const float* a, const float* b, size_t n) {
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    DiffStats s;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(ha[i] - hb[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum += d;
    }
    s.mean_abs = sum / static_cast<double>(n);
    return s;
}

static void print_result(const std::string& name, int32_t M, int32_t K, float hand_ms, float lt_ms, const DiffStats& diff) {
    double speedup = static_cast<double>(lt_ms) / static_cast<double>(hand_ms);
    std::cout << std::left << std::setw(24) << name
              << " M=" << std::setw(7) << M
              << " K=" << std::setw(7) << K
              << " hand=" << std::setw(10) << std::fixed << std::setprecision(4) << hand_ms << " ms"
              << " cublasLt=" << std::setw(10) << lt_ms << " ms"
              << " lt/hand=" << std::setw(8) << std::setprecision(3) << speedup
              << " max_abs=" << std::scientific << diff.max_abs
              << " mean_abs=" << diff.mean_abs
              << std::defaultfloat << std::endl;
}

static void bench_basic_gemv(
    cublasLtHandle_t lt,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    const std::string& name,
    int32_t M,
    int32_t K,
    bool fp32_out,
    int warmup,
    int iters
) {
    std::cout << "\n[Case] " << name << std::endl;
    size_t w_bytes = static_cast<size_t>(M) * K * sizeof(uint16_t);
    size_t x_bytes = static_cast<size_t>(K) * sizeof(uint16_t);

    auto* d_W = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(w_bytes));
    auto* d_x = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(x_bytes));
    fill_bf16_device(d_W, static_cast<size_t>(M) * K, 123);
    fill_bf16_device(d_x, K, 456);

    dim3 block(256);
    dim3 grid(M);

    if (fp32_out) {
        auto* d_hand = static_cast<float*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(float)));
        auto* d_lt = static_cast<float*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(float)));
        LtGemvPlan plan(lt, M, K, CUDA_R_32F, 0.0f, workspace, workspace_bytes);

        manual_gemv_fp32_out_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, K);
        CUDA_CHECK(cudaGetLastError());
        plan.run(stream, d_W, d_x, d_lt, d_lt);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        DiffStats diff = compare_fp32_device(d_hand, d_lt, M);

        float hand_ms = time_gpu_ms(stream, warmup, iters, [&]() {
            manual_gemv_fp32_out_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, K);
        });
        float lt_ms = time_gpu_ms(stream, warmup, iters, [&]() {
            plan.run(stream, d_W, d_x, d_lt, d_lt);
        });
        print_result(name, M, K, hand_ms, lt_ms, diff);
        cudaFree(d_hand);
        cudaFree(d_lt);
    } else {
        auto* d_hand = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(uint16_t)));
        auto* d_lt = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(uint16_t)));
        LtGemvPlan plan(lt, M, K, CUDA_R_16BF, 0.0f, workspace, workspace_bytes);

        manual_gemv_bf16_out_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, K);
        CUDA_CHECK(cudaGetLastError());
        plan.run(stream, d_W, d_x, d_lt, d_lt);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        DiffStats diff = compare_bf16_device(d_hand, d_lt, M);

        float hand_ms = time_gpu_ms(stream, warmup, iters, [&]() {
            manual_gemv_bf16_out_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, K);
        });
        float lt_ms = time_gpu_ms(stream, warmup, iters, [&]() {
            plan.run(stream, d_W, d_x, d_lt, d_lt);
        });
        print_result(name, M, K, hand_ms, lt_ms, diff);
        cudaFree(d_hand);
        cudaFree(d_lt);
    }

    cudaFree(d_W);
    cudaFree(d_x);
}

static void bench_fused_add(
    cublasLtHandle_t lt,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    const std::string& name,
    int32_t M,
    int32_t K,
    int warmup,
    int iters
) {
    std::cout << "\n[Case] " << name << std::endl;
    size_t w_bytes = static_cast<size_t>(M) * K * sizeof(uint16_t);
    auto* d_W = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(w_bytes));
    auto* d_x = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(K) * sizeof(uint16_t)));
    auto* d_res0 = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(uint16_t)));
    auto* d_hand = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(uint16_t)));
    auto* d_lt = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(uint16_t)));

    fill_bf16_device(d_W, static_cast<size_t>(M) * K, 111);
    fill_bf16_device(d_x, K, 222);
    fill_bf16_device(d_res0, M, 333);
    CUDA_CHECK(cudaMemcpyAsync(d_hand, d_res0, static_cast<size_t>(M) * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lt, d_res0, static_cast<size_t>(M) * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));

    dim3 block(256);
    dim3 grid(M);
    LtGemvPlan plan(lt, M, K, CUDA_R_16BF, 1.0f, workspace, workspace_bytes);

    manual_fused_gemv_add_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, K);
    CUDA_CHECK(cudaGetLastError());
    plan.run(stream, d_W, d_x, d_lt, d_lt);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    DiffStats diff = compare_bf16_device(d_hand, d_lt, M);

    float hand_ms = time_gpu_ms(stream, warmup, iters, [&]() {
        manual_fused_gemv_add_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, K);
    });
    float lt_ms = time_gpu_ms(stream, warmup, iters, [&]() {
        plan.run(stream, d_W, d_x, d_lt, d_lt);
    });
    print_result(name, M, K, hand_ms, lt_ms, diff);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_res0);
    cudaFree(d_hand);
    cudaFree(d_lt);
}

static void bench_qkv(
    cublasLtHandle_t lt,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    int32_t hidden_size,
    int32_t q_dim,
    int32_t kv_dim,
    int warmup,
    int iters
) {
    const int32_t M = q_dim + 2 * kv_dim;
    const int32_t K = hidden_size;
    std::cout << "\n[Case] qkv_proj split outputs" << std::endl;

    auto* d_W = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * K * sizeof(uint16_t)));
    auto* d_x = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(K) * sizeof(uint16_t)));
    auto* q_hand = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(q_dim) * sizeof(uint16_t)));
    auto* k_hand = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(kv_dim) * sizeof(uint16_t)));
    auto* v_hand = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(kv_dim) * sizeof(uint16_t)));
    auto* q_lt = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(q_dim) * sizeof(uint16_t)));
    auto* k_lt = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(kv_dim) * sizeof(uint16_t)));
    auto* v_lt = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(kv_dim) * sizeof(uint16_t)));
    auto* qkv_tmp = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(uint16_t)));

    fill_bf16_device(d_W, static_cast<size_t>(M) * K, 777);
    fill_bf16_device(d_x, K, 888);

    dim3 block(256);
    dim3 grid(M);
    LtGemvPlan plan(lt, M, K, CUDA_R_16BF, 0.0f, workspace, workspace_bytes);

    manual_fused_qkv_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, q_hand, k_hand, v_hand, K, q_dim, kv_dim);
    CUDA_CHECK(cudaGetLastError());
    plan.run(stream, d_W, d_x, qkv_tmp, qkv_tmp);
    split_qkv_bf16_kernel<<<(M + 255) / 256, 256, 0, stream>>>(qkv_tmp, q_lt, k_lt, v_lt, q_dim, kv_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DiffStats dq = compare_bf16_device(q_hand, q_lt, q_dim);
    DiffStats dk = compare_bf16_device(k_hand, k_lt, kv_dim);
    DiffStats dv = compare_bf16_device(v_hand, v_lt, kv_dim);
    DiffStats diff;
    diff.max_abs = std::max({dq.max_abs, dk.max_abs, dv.max_abs});
    diff.mean_abs = (dq.mean_abs * q_dim + dk.mean_abs * kv_dim + dv.mean_abs * kv_dim) / static_cast<double>(M);

    float hand_ms = time_gpu_ms(stream, warmup, iters, [&]() {
        manual_fused_qkv_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, q_hand, k_hand, v_hand, K, q_dim, kv_dim);
    });
    float lt_ms = time_gpu_ms(stream, warmup, iters, [&]() {
        plan.run(stream, d_W, d_x, qkv_tmp, qkv_tmp);
        split_qkv_bf16_kernel<<<(M + 255) / 256, 256, 0, stream>>>(qkv_tmp, q_lt, k_lt, v_lt, q_dim, kv_dim);
    });
    print_result("qkv_proj", M, K, hand_ms, lt_ms, diff);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(q_hand);
    cudaFree(k_hand);
    cudaFree(v_hand);
    cudaFree(q_lt);
    cudaFree(k_lt);
    cudaFree(v_lt);
    cudaFree(qkv_tmp);
}

static void bench_gate_up_swiglu(
    cublasLtHandle_t lt,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    int32_t hidden_size,
    int32_t intermediate_dim,
    int warmup,
    int iters
) {
    const int32_t M = 2 * intermediate_dim;
    const int32_t K = hidden_size;
    std::cout << "\n[Case] gate_up_proj + swiglu" << std::endl;

    auto* d_W = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(M) * K * sizeof(uint16_t)));
    auto* d_x = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(K) * sizeof(uint16_t)));
    auto* d_hand = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(intermediate_dim) * sizeof(uint16_t)));
    auto* d_lt = static_cast<__nv_bfloat16*>(cuda_malloc_bytes(static_cast<size_t>(intermediate_dim) * sizeof(uint16_t)));
    auto* gate_up_tmp = static_cast<float*>(cuda_malloc_bytes(static_cast<size_t>(M) * sizeof(float)));

    fill_bf16_device(d_W, static_cast<size_t>(M) * K, 999);
    fill_bf16_device(d_x, K, 1001);

    dim3 block(256);
    dim3 grid(intermediate_dim);
    LtGemvPlan plan(lt, M, K, CUDA_R_32F, 0.0f, workspace, workspace_bytes);

    manual_fused_gate_up_swiglu_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, intermediate_dim, K);
    CUDA_CHECK(cudaGetLastError());
    plan.run(stream, d_W, d_x, gate_up_tmp, gate_up_tmp);
    swiglu_from_fp32_tmp_kernel<<<(intermediate_dim + 255) / 256, 256, 0, stream>>>(gate_up_tmp, d_lt, intermediate_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    DiffStats diff = compare_bf16_device(d_hand, d_lt, intermediate_dim);

    float hand_ms = time_gpu_ms(stream, warmup, iters, [&]() {
        manual_fused_gate_up_swiglu_kernel<256><<<grid, block, 0, stream>>>(d_x, d_W, d_hand, intermediate_dim, K);
    });
    float lt_ms = time_gpu_ms(stream, warmup, iters, [&]() {
        plan.run(stream, d_W, d_x, gate_up_tmp, gate_up_tmp);
        swiglu_from_fp32_tmp_kernel<<<(intermediate_dim + 255) / 256, 256, 0, stream>>>(gate_up_tmp, d_lt, intermediate_dim);
    });
    print_result("gate_up_swiglu", M, K, hand_ms, lt_ms, diff);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_hand);
    cudaFree(d_lt);
    cudaFree(gate_up_tmp);
}

struct Args {
    int warmup = 20;
    int iters = 200;
    int workspace_mb = 64;
    bool skip_lm_head = false;
    std::string case_name = "all";
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc) {
            args.warmup = std::stoi(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            args.iters = std::stoi(argv[++i]);
        } else if (a == "--workspace-mb" && i + 1 < argc) {
            args.workspace_mb = std::stoi(argv[++i]);
        } else if (a == "--case" && i + 1 < argc) {
            args.case_name = argv[++i];
        } else if (a == "--skip-lm-head") {
            args.skip_lm_head = true;
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [--iters N] [--warmup N] [--workspace-mb N] [--case all|qkv|o|gate|down|lm|gemv] [--skip-lm-head]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << a << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    return args;
}

static bool wants_case(const Args& args, const std::string& c) {
    return args.case_name == "all" || args.case_name == c;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    CUDA_CHECK(cudaSetDevice(1));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device: " << prop.name << " sm_" << prop.major << prop.minor << std::endl;

    if (prop.major < 8) {
        std::cerr << "Warning: native bf16 tensor core performance generally requires Ampere or newer." << std::endl;
    }

    // Qwen3-4B dense config.
    constexpr int32_t hidden_size = 2560;
    constexpr int32_t intermediate_size = 9728;
    constexpr int32_t head_dim = 128;
    constexpr int32_t num_attention_heads = 32;
    constexpr int32_t num_key_value_heads = 8;
    constexpr int32_t vocab_size = 151936;

    constexpr int32_t q_dim = num_attention_heads * head_dim;       // 4096
    constexpr int32_t kv_dim = num_key_value_heads * head_dim;      // 1024

    static_assert(hidden_size % 8 == 0, "hidden_size must be divisible by 8");
    static_assert(q_dim % 8 == 0, "q_dim must be divisible by 8");
    static_assert(kv_dim % 8 == 0, "kv_dim must be divisible by 8");
    static_assert(intermediate_size % 8 == 0, "intermediate_size must be divisible by 8");

    std::cout << "Qwen3-4B shapes:" << std::endl;
    std::cout << "  hidden_size=" << hidden_size << std::endl;
    std::cout << "  intermediate_size=" << intermediate_size << std::endl;
    std::cout << "  q_dim=num_attention_heads*head_dim=" << q_dim << std::endl;
    std::cout << "  kv_dim=num_key_value_heads*head_dim=" << kv_dim << std::endl;
    std::cout << "  vocab_size=" << vocab_size << std::endl;
    std::cout << "  warmup=" << args.warmup << " iters=" << args.iters << " workspace_mb=" << args.workspace_mb << std::endl;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasLtHandle_t lt;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    size_t workspace_bytes = static_cast<size_t>(args.workspace_mb) * 1024ULL * 1024ULL;
    void* workspace = nullptr;
    if (workspace_bytes > 0) {
        workspace = cuda_malloc_bytes(workspace_bytes);
    }

    std::cout << "\nColumns:" << std::endl;
    std::cout << "  lt/hand < 1 means cuBLASLt is faster; lt/hand > 1 means hand-written kernel is faster.\n";

    if (wants_case(args, "qkv")) {
        // qkv weight: [q_dim + 2 * kv_dim, hidden_size]
        bench_qkv(lt, stream, workspace, workspace_bytes, hidden_size, q_dim, kv_dim, args.warmup, args.iters);
    }

    if (wants_case(args, "o")) {
        // o_proj weight: [hidden_size, q_dim], usually followed by residual add.
        bench_fused_add(lt, stream, workspace, workspace_bytes, "o_proj_add", hidden_size, q_dim, args.warmup, args.iters);
    }

    if (wants_case(args, "gate")) {
        // gate_up_proj weight: [2 * intermediate_size, hidden_size], followed by SwiGLU.
        bench_gate_up_swiglu(lt, stream, workspace, workspace_bytes, hidden_size, intermediate_size, args.warmup, args.iters);
    }

    if (wants_case(args, "down")) {
        // down_proj weight: [hidden_size, intermediate_size], usually followed by residual add.
        bench_fused_add(lt, stream, workspace, workspace_bytes, "down_proj_add", hidden_size, intermediate_size, args.warmup, args.iters);
    }

    if (wants_case(args, "gemv")) {
        // Generic hidden-to-hidden GEMV baseline.
        bench_basic_gemv(lt, stream, workspace, workspace_bytes, "generic_bf16_gemv", hidden_size, hidden_size, false, args.warmup, args.iters);
    }

    if (!args.skip_lm_head && wants_case(args, "lm")) {
        // lm_head weight: [vocab_size, hidden_size], output logits fp32.
        bench_basic_gemv(lt, stream, workspace, workspace_bytes, "lm_head_fp32", vocab_size, hidden_size, true, args.warmup, args.iters);
    }

    if (workspace) {
        cudaFree(workspace);
    }
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
