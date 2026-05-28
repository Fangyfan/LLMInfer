#include "matmul_kernel.cuh"
#include <cuda_bf16.h>
#include <cublasLt.h>

namespace kernel {
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
    __shared__ float shared_vals[WARP_NUM];
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    val = warp_reduce_sum<32>(val);
    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
        }
        val = warp_reduce_sum<WARP_NUM>(val);
    }
    if (threadIdx.x == 0) {
        shared_vals[0] = val;
    }
    __syncthreads();
    return shared_vals[0];
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void gemv_bf16x8_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,   // [N]
    const __nv_bfloat16* __restrict__ wei,  // [M, N]
    __nv_bfloat16* __restrict__ out,        // [M]
    int32_t N
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t N8 = (N >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * N);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < N8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) { // bf16 * bf16 -> fp32 * fp32 -> fp32
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
static __global__ __launch_bounds__(BLOCK_DIM) void gemv_bf16x8_fp32_kernel(
    const __nv_bfloat16* __restrict__ in,   // [N]
    const __nv_bfloat16* __restrict__ wei,  // [M, N]
    float* __restrict__ out,                // [M]
    int32_t N
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t N8 = (N >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * N);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < N8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) { // bf16 * bf16 -> fp32 * fp32 -> fp32
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

void gemv_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    bool lm_head, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == (!lm_head ? base::DataType::DataTypeBf16 : base::DataType::DataTypeFp32));

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(input.get_dim(0), N);
    CHECK_EQ(output.get_dim(0), M);
    CHECK(N % 8 == 0);
    
    dim3 gridDim(M);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    if (!lm_head) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
        gemv_bf16x8_bf16_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, N);
    } else {
        float* out = const_cast<float*>(output.ptr<float>());
        gemv_bf16x8_fp32_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, N);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gemv_add_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,       // [N]
    const __nv_bfloat16* __restrict__ wei,      // [M, N]
    __nv_bfloat16* __restrict__ residual_add,   // [M]
    int32_t N
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t N8 = (N >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * N);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < N8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) { // bf16 * bf16 -> fp32 * fp32 -> fp32
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        float residual = __bfloat162float(residual_add[blockIdx.x]);
        residual_add[blockIdx.x] = __float2bfloat16(residual + sum); // fp32 + fp32 -> bf16
    }
}

void fused_gemv_add_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, // residual_add
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == base::DataType::DataTypeBf16);

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    __nv_bfloat16* res_add = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(input.get_dim(0), N);
    CHECK_EQ(output.get_dim(0), M);
    CHECK(N % 8 == 0);

    dim3 gridDim(M);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_gemv_add_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, res_add, N);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qkv_gemv_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int32_t N, int32_t dim, int32_t kv_dim
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t N8 = (N >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(input);
    const uint4* wei8 = reinterpret_cast<const uint4*>(weight + blockIdx.x * N);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < N8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) { // bf16 * bf16 -> fp32 * fp32 -> fp32
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        if (blockIdx.x < dim) {
            query[blockIdx.x] = __float2bfloat16(sum);
        } else if (blockIdx.x < dim + kv_dim) {
            key[blockIdx.x - dim] = __float2bfloat16(sum);
        } else {
            value[blockIdx.x - dim - kv_dim] = __float2bfloat16(sum);
        }
    }
}

void fused_qkv_gemv_kernel_cu(
    const tensor::Tensor& input,    // [hidden_dim]
    const tensor::Tensor& weight,   // [dim + 2 * kv_dim, hidden_dim]
    const tensor::Tensor& query,    // [dim]
    const tensor::Tensor& key,      // [kv_dim]
    const tensor::Tensor& value,    // [kv_dim]
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!query.is_empty() && query.dims_size() == 1);
    CHECK(!key.is_empty() && key.dims_size() == 1);
    CHECK(!value.is_empty() && value.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeBf16);
    CHECK(query.data_type() == base::DataType::DataTypeBf16);
    CHECK(key.data_type() == base::DataType::DataTypeBf16);
    CHECK(value.data_type() == base::DataType::DataTypeBf16);

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(value.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t hidden_dim = input.get_dim(0);
    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const int32_t dim = query.get_dim(0);
    const int32_t kv_dim = key.get_dim(0);

    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    __nv_bfloat16* q = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(query.ptr<uint16_t>()));
    __nv_bfloat16* k = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(key.ptr<uint16_t>()));
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(value.ptr<uint16_t>()));

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(value.get_dim(0), kv_dim);
    CHECK_EQ(M, dim + 2 * kv_dim);
    CHECK_EQ(N, hidden_dim);
    CHECK(N % 8 == 0);

    dim3 gridDim(M);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_qkv_gemv_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, q, k, v, N, dim, kv_dim);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gate_up_gemv_swiglu_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ out,
    int32_t immediate_dim, 
    int32_t N
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t N8 = (N >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* gate8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * N);
    const uint4* up8 = reinterpret_cast<const uint4*>(wei + (blockIdx.x + immediate_dim) * N);
    
    float gate = 0.0f;
    float up = 0.0f;
    for (int32_t i = threadIdx.x; i < N8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = gate8[i];
        uint4 c = up8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
        const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&c);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) { // bf16 * bf16 -> fp32 * fp32 -> fp32
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

void fused_gate_up_gemv_swiglu_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t immediate_dim, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == base::DataType::DataTypeBf16);

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK(M == 2 * immediate_dim);
    CHECK_EQ(output.get_dim(0), immediate_dim);
    CHECK_EQ(input.get_dim(0), N);
    CHECK(N % 8 == 0);

    dim3 gridDim(immediate_dim);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_gate_up_gemv_swiglu_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, immediate_dim, N);
}

template <int32_t BLOCK_DIM>
static __global__ void gemv_int8_kernel(
    const float* __restrict__ in,    // [N]
    const int8_t* __restrict__ wei,  // [M, N]
    float* __restrict__ out,         // [M]
    const float* __restrict__ scales, 
    int32_t group_size, int32_t N
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);

    int32_t N4 = (N >> 2);
    float sum = 0.0f;
    const float4* in8 = reinterpret_cast<const float4*>(in);
    for (int32_t i = threadIdx.x; i < N4; i += blockDim.x) {
        int32_t idx = blockIdx.x * N + (i << 2);
        int32_t group_id = idx / group_size;
        float scale = scales[group_id];
        float4 a = in8[i];
        float4 b = make_float4(
            scale * static_cast<float>(wei[idx]),
            scale * static_cast<float>(wei[idx + 1]),
            scale * static_cast<float>(wei[idx + 2]),
            scale * static_cast<float>(wei[idx + 3])
        );
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

void gemv_int8_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, const tensor::Tensor& scales, int32_t group_size, void* stream) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 1);
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const float* in = input.ptr<float>();
    const int8_t* wei = weight.ptr<int8_t>();
    const float* scl = scales.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<float>()) % 16, 0);
    CHECK_EQ(input.get_dim(0), N);
    CHECK_EQ(output.get_dim(0), M);
    CHECK(group_size == 64);
    CHECK(N % group_size == 0);

    dim3 gridDim(M);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    gemv_int8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, scl, group_size, N);
}
}  // namespace kernel