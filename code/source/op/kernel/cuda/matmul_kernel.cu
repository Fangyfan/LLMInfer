#include "matmul_kernel.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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
    const __nv_bfloat16* __restrict__ in,   // [K]
    const __nv_bfloat16* __restrict__ wei,  // [N, K]
    __nv_bfloat16* __restrict__ out,        // [N]
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
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
    const __nv_bfloat16* __restrict__ in,   // [K]
    const __nv_bfloat16* __restrict__ wei,  // [N, K]
    float* __restrict__ out,                // [N]
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
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

    const int32_t N = weight.get_dim(0);
    const int32_t K = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(input.get_dim(0), K);
    CHECK_EQ(output.get_dim(0), N);
    CHECK(K % 8 == 0);
    
    dim3 gridDim(N);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    if (!lm_head) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
        gemv_bf16x8_bf16_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, K);
    } else {
        float* out = const_cast<float*>(output.ptr<float>());
        gemv_bf16x8_fp32_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, K);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gemv_add_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,       // [K]
    const __nv_bfloat16* __restrict__ wei,      // [N, K]
    __nv_bfloat16* __restrict__ residual_add,   // [N]
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
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

    const int32_t N = weight.get_dim(0);
    const int32_t K = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    __nv_bfloat16* res_add = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(input.get_dim(0), K);
    CHECK_EQ(output.get_dim(0), N);
    CHECK(K % 8 == 0);

    dim3 gridDim(N);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_gemv_add_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, res_add, K);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qkv_gemv_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int32_t K, int32_t dim, int32_t kv_dim
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(input);
    const uint4* wei8 = reinterpret_cast<const uint4*>(weight + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
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
    const int32_t N = weight.get_dim(0);
    const int32_t K = weight.get_dim(1);
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
    CHECK_EQ(N, dim + 2 * kv_dim);
    CHECK_EQ(K, hidden_dim);
    CHECK(K % 8 == 0);

    dim3 gridDim(N);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_qkv_gemv_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, q, k, v, K, dim, kv_dim);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gate_up_gemv_swiglu_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ out,
    int32_t immediate_dim, 
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* gate8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);
    const uint4* up8 = reinterpret_cast<const uint4*>(wei + (blockIdx.x + immediate_dim) * K);
    
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

    const int32_t N = weight.get_dim(0);
    const int32_t K = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK(N == 2 * immediate_dim);
    CHECK_EQ(output.get_dim(0), immediate_dim);
    CHECK_EQ(input.get_dim(0), K);
    CHECK(K % 8 == 0);

    dim3 gridDim(immediate_dim);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_gate_up_gemv_swiglu_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, immediate_dim, K);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gemv_add_int4x8_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,       // [K]
    const int32_t* __restrict__ wei,            // [N, K], each int32 packs 8 output rows
    const int32_t* __restrict__ zeros,          // [N, K / 128], each int32 packs 8 zero-points
    const half* __restrict__ scales,            // [8 * N, K / 128]
    __nv_bfloat16* __restrict__ residual_add,   // [8 * N]
    int32_t group_size, int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t n_pack_id = blockIdx.x;
    const int32_t out_base = n_pack_id << 3;
    const int32_t group_num = K / group_size;

    const int32_t* __restrict__ wei_row = wei + n_pack_id * K;
    const int32_t* __restrict__ zero_row = zeros + n_pack_id * group_num;

    float sum[8] = { 0.0f };

    const int32_t K8 = K >> 3;
    const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(in);

    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        const int32_t k_base = i << 3;
        const int32_t group_id = k_base / group_size;

        const uint4 a = in8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);

        float x[8] = {
            __bfloat162float(a2[0].x), __bfloat162float(a2[0].y),
            __bfloat162float(a2[1].x), __bfloat162float(a2[1].y),
            __bfloat162float(a2[2].x), __bfloat162float(a2[2].y),
            __bfloat162float(a2[3].x), __bfloat162float(a2[3].y)
        };

        const uint32_t zero_pack8 = static_cast<uint32_t>(zero_row[group_id]);
        float zero[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            zero[j] = static_cast<float>((zero_pack8 >> (j << 2)) & 0xF);
        }

        float scale[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            scale[j] = __half2float(scales[(out_base + j) * group_num + group_id]);
        }

        const int32_t* wei_ptr = wei_row + k_base;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t wei_pack8 = static_cast<uint32_t>(wei_ptr[j]);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float wei_val = static_cast<float>((wei_pack8 >> (k << 2)) & 0xF);
                sum[k] += x[j] * ((wei_val - zero[k]) * scale[k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum[i] = block_reduce_sum<WARP_NUM>(sum[i]);
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float add = __bfloat162float(residual_add[out_base + i]);
            residual_add[out_base + i] = __float2bfloat16(add + sum[i]);
        }
    }
}

void fused_gemv_add_int4_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, // residual_add
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 1);
    CHECK(!zeros.is_empty() && zeros.dims_size() == 1);
    CHECK(!scales.is_empty() && scales.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeInt4x8);
    CHECK(zeros.data_type() == base::DataType::DataTypeInt4x8);
    CHECK(scales.data_type() == base::DataType::DataTypeFp16);

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(zeros.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(scales.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t N = weight.get_dim(0);   // packed output rows, real output = 8 * N
    const int32_t K = weight.get_dim(1);   // input dim
    CHECK_EQ(input.get_dim(0), K);
    CHECK_EQ(output.get_dim(0), 8 * N);
    CHECK(group_size == 128);
    CHECK(K % group_size == 0);
    const int32_t group_num = K / group_size;

    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    __nv_bfloat16* res_add = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    const int32_t* wei = weight.ptr<int32_t>();
    const int32_t* zeros_ptr = zeros.ptr<int32_t>();
    const half* scales_ptr = reinterpret_cast<const half*>(scales.ptr<uint16_t>());

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(zeros.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(scales.ptr<uint16_t>()) % 16, 0);

    // zeros logical shape should be [N, K / 128] flattened
    CHECK_EQ(zeros.get_dim(0), N * group_num);

    // scales logical shape should be [8 * N, K / 128] flattened
    CHECK_EQ(scales.get_dim(0), 8 * N * group_num);

    dim3 gridDim(N);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    fused_gemv_add_int4x8_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, zeros_ptr, scales_ptr, res_add, group_size, K);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qkv_gemv_int4x8_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,       // [hidden_dim]
    const int32_t* __restrict__ wei,            // [(dim + 2 * kv_dim) / 8, hidden_dim      ], each int32 packs 8 output rows
    const int32_t* __restrict__ zeros,          // [(dim + 2 * kv_dim) / 8, hidden_dim / 128], each int32 packs 8 zero-points
    const half* __restrict__ scales,            // [dim + 2 * kv_dim,       hidden_dim / 128]
    __nv_bfloat16* __restrict__ query,          // [dim]
    __nv_bfloat16* __restrict__ key,            // [kv_dim]
    __nv_bfloat16* __restrict__ value,          // [kv_dim]
    int32_t group_size, int32_t K, int32_t dim, int32_t kv_dim
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t n_pack_id = blockIdx.x;
    const int32_t out_base = n_pack_id << 3;
    const int32_t group_num = K / group_size;

    const int32_t* __restrict__ wei_row = wei + n_pack_id * K;
    const int32_t* __restrict__ zero_row = zeros + n_pack_id * group_num;

    float sum[8] = { 0.0f };

    const int32_t K8 = K >> 3;
    const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(in);

    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        const int32_t k_base = i << 3;
        const int32_t group_id = k_base / group_size;

        const uint4 a = in8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);

        float x[8] = {
            __bfloat162float(a2[0].x), __bfloat162float(a2[0].y),
            __bfloat162float(a2[1].x), __bfloat162float(a2[1].y),
            __bfloat162float(a2[2].x), __bfloat162float(a2[2].y),
            __bfloat162float(a2[3].x), __bfloat162float(a2[3].y)
        };

        const uint32_t zero_pack8 = static_cast<uint32_t>(zero_row[group_id]);
        float zero[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            zero[j] = static_cast<float>((zero_pack8 >> (j << 2)) & 0xF);
        }

        float scale[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            scale[j] = __half2float(scales[(out_base + j) * group_num + group_id]);
        }

        const int32_t* wei_ptr = wei_row + k_base;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t wei_pack8 = static_cast<uint32_t>(wei_ptr[j]);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float wei_val = static_cast<float>((wei_pack8 >> (k << 2)) & 0xF);
                sum[k] += x[j] * ((wei_val - zero[k]) * scale[k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum[i] = block_reduce_sum<WARP_NUM>(sum[i]);
    }

    if (threadIdx.x == 0) {
        if (out_base < dim) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                query[out_base + i] = __float2bfloat16(sum[i]);
        } else if (out_base < dim + kv_dim) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                key[out_base + i - dim] = __float2bfloat16(sum[i]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                value[out_base + i - dim - kv_dim] = __float2bfloat16(sum[i]);
        }
    }
}

void fused_qkv_gemv_int4_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& value, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!query.is_empty() && query.dims_size() == 1);
    CHECK(!key.is_empty() && key.dims_size() == 1);
    CHECK(!value.is_empty() && value.dims_size() == 1);
    CHECK(!zeros.is_empty() && zeros.dims_size() == 1);
    CHECK(!scales.is_empty() && scales.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(query.data_type() == base::DataType::DataTypeBf16);
    CHECK(key.data_type() == base::DataType::DataTypeBf16);
    CHECK(value.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeInt4x8);
    CHECK(zeros.data_type() == base::DataType::DataTypeInt4x8);
    CHECK(scales.data_type() == base::DataType::DataTypeFp16);

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(value.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(zeros.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(scales.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t hidden_dim = input.get_dim(0);
    const int32_t N = weight.get_dim(0);    // packed output rows, real output = 8 * N
    const int32_t K = weight.get_dim(1);
    const int32_t dim = query.get_dim(0);
    const int32_t kv_dim = key.get_dim(0);

    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    const int32_t* wei = weight.ptr<int32_t>();
    const int32_t* z = zeros.ptr<int32_t>();
    const half* s = reinterpret_cast<const half*>(scales.ptr<uint16_t>());
    __nv_bfloat16* q = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(query.ptr<uint16_t>()));
    __nv_bfloat16* k = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(key.ptr<uint16_t>()));
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(value.ptr<uint16_t>()));

    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(zeros.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(scales.ptr<uint16_t>()) % 16, 0);

    CHECK_EQ(value.get_dim(0), kv_dim);
    CHECK_EQ(dim + 2 * kv_dim, 8 * N);
    CHECK_EQ(K, hidden_dim);

    CHECK(group_size == 128);
    CHECK(K % group_size == 0);
    const int32_t group_num = K / group_size;
    CHECK_EQ(zeros.get_dim(0), N * group_num);
    CHECK_EQ(scales.get_dim(0), 8 * N * group_num);

    dim3 gridDim(N);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_qkv_gemv_int4x8_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, z, s, q, k, v, group_size, K, dim, kv_dim);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gate_up_gemv_swiglu_int4x8_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,   // [K]
    const int32_t* __restrict__ wei,        // [2 * immediate_dim / 8,  K      ], each int32 packs 8 output rows
    const int32_t* __restrict__ zeros,      // [2 * immediate_dim / 8,  K / 128], each int32 packs 8 zero-points
    const half* __restrict__ scales,        // [2 * immediate_dim,      K / 128]
    __nv_bfloat16* __restrict__ out,        // [immediate_dim]
    int32_t group_size, int32_t immediate_dim, int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t n_pack_id = blockIdx.x;
    const int32_t out_base = n_pack_id << 3;
    const int32_t group_num = K / group_size;

    const int32_t* __restrict__ gate_wei_row = wei + n_pack_id * K;
    const int32_t* __restrict__ gate_zero_row = zeros + n_pack_id * group_num;

    const int32_t* __restrict__ up_wei_row = gate_wei_row + (immediate_dim >> 3) * K;
    const int32_t* __restrict__ up_zero_row = gate_zero_row + (immediate_dim >> 3) * group_num;

    float gate[8] = { 0.0f };
    float up[8] = { 0.0f };

    const int32_t K8 = K >> 3;
    const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(in);

    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        const int32_t k_base = i << 3;
        const int32_t group_id = k_base / group_size;

        const uint4 a = in8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);

        float x[8] = {
            __bfloat162float(a2[0].x), __bfloat162float(a2[0].y),
            __bfloat162float(a2[1].x), __bfloat162float(a2[1].y),
            __bfloat162float(a2[2].x), __bfloat162float(a2[2].y),
            __bfloat162float(a2[3].x), __bfloat162float(a2[3].y)
        };

        const uint32_t gate_zero_pack8 = static_cast<uint32_t>(gate_zero_row[group_id]);
        const uint32_t up_zero_pack8 = static_cast<uint32_t>(up_zero_row[group_id]);
        float gate_zero[8];
        float up_zero[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            gate_zero[j] = static_cast<float>((gate_zero_pack8 >> (j << 2)) & 0xF);
            up_zero[j] = static_cast<float>((up_zero_pack8 >> (j << 2)) & 0xF);
        }

        float gate_scale[8];
        float up_scale[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            gate_scale[j] = __half2float(scales[(out_base + j) * group_num + group_id]);
            up_scale[j] = __half2float(scales[(out_base + immediate_dim + j) * group_num + group_id]);
        }

        const int32_t* gate_wei_ptr = gate_wei_row + k_base;
        const int32_t* up_wei_ptr = up_wei_row + k_base;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t gate_wei_pack8 = static_cast<uint32_t>(gate_wei_ptr[j]);
            uint32_t up_wei_pack8 = static_cast<uint32_t>(up_wei_ptr[j]);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                gate[k] += x[j] * ((static_cast<float>((gate_wei_pack8 >> (k << 2)) & 0xF) - gate_zero[k]) * gate_scale[k]);
                up[k] += x[j] * ((static_cast<float>((up_wei_pack8 >> (k << 2)) & 0xF) - up_zero[k]) * up_scale[k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        gate[i] = block_reduce_sum<WARP_NUM>(gate[i]);
        up[i] = block_reduce_sum<WARP_NUM>(up[i]);
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float gate_silu = gate[i] / (1.0f + __expf(-gate[i]));
            out[out_base + i] = __float2bfloat16(gate_silu * up[i]);
        }
    }
}

void fused_gate_up_gemv_swiglu_int4_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    int32_t immediate_dim, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 1);
    CHECK(!zeros.is_empty() && zeros.dims_size() == 1);
    CHECK(!scales.is_empty() && scales.dims_size() == 1);

    CHECK(input.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == base::DataType::DataTypeBf16);
    CHECK(weight.data_type() == base::DataType::DataTypeInt4x8);
    CHECK(zeros.data_type() == base::DataType::DataTypeInt4x8);
    CHECK(scales.data_type() == base::DataType::DataTypeFp16);

    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(zeros.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(scales.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t N = weight.get_dim(0);
    const int32_t K = weight.get_dim(1);
    const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(input.ptr<uint16_t>());
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    const int32_t* wei = weight.ptr<int32_t>();
    const int32_t* z = zeros.ptr<int32_t>();
    const half* s = reinterpret_cast<const half*>(scales.ptr<uint16_t>());
    
    CHECK_EQ(reinterpret_cast<uintptr_t>(input.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(zeros.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(scales.ptr<uint16_t>()) % 16, 0);

    CHECK_EQ(output.get_dim(0), immediate_dim);
    CHECK_EQ(input.get_dim(0), K);
    CHECK_EQ(2 * immediate_dim, 8 * N);

    CHECK(group_size == 128);
    CHECK(K % group_size == 0);
    const int32_t group_num = K / group_size;
    CHECK_EQ(zeros.get_dim(0), N * group_num);
    CHECK_EQ(scales.get_dim(0), 8 * N * group_num);

    dim3 gridDim(immediate_dim / 8);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_gate_up_gemv_swiglu_int4x8_bf16x8_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, z, s, out, group_size, immediate_dim, K);
}

}  // namespace kernel