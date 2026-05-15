#include "rmsnorm_kernel.cuh"

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
    return val;
}

template <int32_t BLOCK_DIM>
static __global__ void rmsnorm_kernel_fp32(
    const float* in, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t size, 
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const float4* in4 = reinterpret_cast<const float4*>(in);
    const float4* wei4 = reinterpret_cast<const float4*>(wei);
    float4* out4 = reinterpret_cast<float4*>(out);
    int32_t size4 = size >> 2;
    
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 v = in4[i];
        sum += (v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w);
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    float scale = shared_scale;

    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 a = in4[i];
        float4 b = wei4[i];
        out4[i] = make_float4(
            a.x * b.x * scale, 
            a.y * b.y * scale, 
            a.z * b.z * scale, 
            a.w * b.w * scale
        );
    }
}

void rmsnorm_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    int32_t size = static_cast<int32_t>(input.size());

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    constexpr float eps = 1e-6f;
#else
    constexpr float eps = 1e-5f;
#endif

    CHECK(size % 4 == 0);
    dim3 gridDim(1);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rmsnorm_kernel_fp32<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, size, eps);
}

template <int32_t BLOCK_DIM>
static __global__ void rmsnorm_2d_kernel_fp32(
    const float* in, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t dim, 
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    in += blockIdx.x * dim;
    out += blockIdx.x * dim;
    
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = in[i];
        sum += val * val;
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / dim + eps);
    }
    __syncthreads();
    float scale = shared_scale;

    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = in[i] * wei[i] * scale;
    }
}

void rmsnorm_2d_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t dim, 
    void* stream
) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    int32_t size = static_cast<int32_t>(input.size());

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    constexpr float eps = 1e-6f;
#else
    constexpr float eps = 1e-5f;
#endif
    
    dim3 blockDim(128);
    dim3 gridDim(size / dim);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rmsnorm_2d_kernel_fp32<128><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, dim, eps);
}
}  // namespace kernel