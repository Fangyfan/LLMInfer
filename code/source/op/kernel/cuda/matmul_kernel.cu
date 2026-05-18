#include "matmul_kernel.cuh"

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
static __global__ __launch_bounds__(BLOCK_DIM) void matmul_kernel_fp32(
    const float* __restrict__ in,    // [N, 1]
    const float* __restrict__ wei,   // [M, N]
    float* __restrict__ out,         // [M, 1]
    float scale, int32_t M, int32_t N
) {
    constexpr int WARP_NUM = (BLOCK_DIM >> 5);

    int32_t N4 = (N >> 2);
    float sum = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    const float4* wei4 = reinterpret_cast<const float4*>(wei + blockIdx.x * N);
    for (int32_t i = threadIdx.x; i < N4; i += blockDim.x) {
        float4 a = in4[i];
        float4 b = wei4[i];
        sum += (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum * scale;
    }
}

void matmul_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    float scale, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    CHECK_EQ(input.get_dim(0), N);
    CHECK(N % 4 == 0);

    dim3 gridDim(M);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    matmul_kernel_fp32<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, scale, M, N);
}

template <int32_t BLOCK_DIM>
static __global__ void matmul_kernel_int8(
    const float* __restrict__ in,    // [N, 1]
    const int8_t* __restrict__ wei,  // [M, N]
    float* __restrict__ out,         // [M, 1]
    const float* __restrict__ scales, 
    int32_t group_size, int32_t M, int32_t N
) {
    constexpr int WARP_NUM = (BLOCK_DIM >> 5);

    int32_t N4 = (N >> 2);
    float sum = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (int32_t i = threadIdx.x; i < N4; i += blockDim.x) {
        int32_t idx = blockIdx.x * N + (i << 2);
        int32_t group_id = idx / group_size;
        float scale = scales[group_id];
        float4 a = in4[i];
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

void matmul_kernel_cu_quant8(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, const tensor::Tensor& scales, int32_t group_size, void* stream) {
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t M = weight.get_dim(0);
    const int32_t N = weight.get_dim(1);
    const float* in = input.ptr<float>();
    const int8_t* wei = weight.ptr<int8_t>();
    const float* scl = scales.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    CHECK_EQ(input.get_dim(0), N);
    CHECK(group_size == 64);
    CHECK(N % group_size == 0);

    dim3 gridDim(M);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    matmul_kernel_int8<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, scl, group_size, M, N);
}
}  // namespace kernel