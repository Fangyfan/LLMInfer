#include "rmsnorm_kernel.cuh"
#include <cub/block/block_reduce.cuh>

namespace kernel {
template<int32_t WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int32_t mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<int32_t THREAD_NUM>
static __device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t WARP_NUM = (THREAD_NUM + WARP_SIZE - 1) / WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    static __shared__ float shared_val[WARP_NUM];
    val = warp_reduce_sum<WARP_SIZE>(val);
    if (lane_id == 0) {
        shared_val[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = lane_id < WARP_NUM ? shared_val[lane_id] : 0.0f;
        val = warp_reduce_sum<WARP_NUM>(val);
    }
    return val;
}

template<int32_t THREAD_NUM>
static __global__ void rmsnorm_kernel_fp32(
    const float* in, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t size, 
    float eps
) {
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = size / pack_size;
    const int32_t tail_off = pack_num * pack_size;

    // 每个线程分工，计算部分平方和，无重复 + 全覆盖 + 负载均衡 + 向量化(float4)存取
    float sum = 0.0f;
    const float4* in_pack = reinterpret_cast<const float4*>(in);
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in_pack[i];
        sum += (in4.x * in4.x) + (in4.y * in4.y) + (in4.z * in4.z) + (in4.w * in4.w);
    }
    for (int32_t i = tail_off + threadIdx.x; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    // Block 级别规约对 Block 内所有线程的值 sum 求和，这段代码等价于:
    // sum = block_reduce_sum<THREAD_NUM>(sum);
    using BlockReduce = cub::BlockReduce<float, THREAD_NUM>;
    __shared__ typename BlockReduce::TempStorage temp;
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    // sum = BlockReduce(temp).Sum(sum);

    // 把全局 scale 存到共享内存，让 Block 内所有线程都能读到
    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / static_cast<float>(size) + eps);
    }
    __syncthreads(); // 这里涉及到共享变量 shared_val 的读取，必须要同步，避免读取脏数据
    const float scale = shared_scale;

    // 每个线程再分工，计算输出值，无重复 + 全覆盖 + 负载均衡 + 向量化(float4)存取
    const float4* wei_pack = reinterpret_cast<const float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in_pack[i];
        float4 wei4 = wei_pack[i];
        out_pack[i] = make_float4(
            scale * in4.x * wei4.x,
            scale * in4.y * wei4.y,
            scale * in4.z * wei4.z,
            scale * in4.w * wei4.w
        );
    }
    for (int32_t i = tail_off + threadIdx.x; i < size; i += blockDim.x) {
        out[i] = scale * in[i] * wei[i];
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
    const int32_t size = static_cast<int32_t>(input.size());

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    constexpr float eps = 1e-6f;
#else
    constexpr float eps = 1e-5f;
#endif

    constexpr int32_t block_num = 1;
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = size / pack_size;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (pack_num < 256) {
        constexpr int32_t thread_num = 128;
        rmsnorm_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, size, eps);
    } else if (pack_num < 512) {
        constexpr int32_t thread_num = 256;
        rmsnorm_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, size, eps);
    } else {
        constexpr int32_t thread_num = 512;
        rmsnorm_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, size, eps);
    }
}

template<int32_t THREAD_NUM>
static __global__ void rmsnorm_2d_kernel_fp32(
    const float* in, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t dim, 
    float eps
) {
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = dim / pack_size;
    const int32_t tail_off = pack_num * pack_size;

    // 每个线程分工，计算部分平方和，无重复 + 全覆盖 + 负载均衡 + 向量化(float4)存取
    float sum = 0.0f;
    const float* row_in = in + blockIdx.x * dim;
    const float4* in_pack = reinterpret_cast<const float4*>(row_in);
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in_pack[i];
        sum += (in4.x * in4.x) + (in4.y * in4.y) + (in4.z * in4.z) + (in4.w * in4.w);
    }
    for (int32_t i = tail_off + threadIdx.x; i < dim; i += blockDim.x) {
        sum += row_in[i] * row_in[i];
    }

    // Block 级别规约对 Block 内所有线程的值 sum 求和
    using BlockReduce = cub::BlockReduce<float, THREAD_NUM>;
    __shared__ typename BlockReduce::TempStorage temp;
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());

    // 把全局 scale 存到共享内存，让 Block 内所有线程都能读到
    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / static_cast<float>(dim) + eps);
    }
    __syncthreads(); // 这里涉及到共享变量 shared_val 的读取，必须要同步，避免读取脏数据
    const float scale = shared_scale;

    // 每个线程再分工，计算输出值，无重复 + 全覆盖 + 负载均衡 + 向量化(float4)存取
    const float4* wei_pack = reinterpret_cast<const float4*>(wei);
    float* row_out = out + blockIdx.x * dim;
    float4* out_pack = reinterpret_cast<float4*>(row_out);
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in_pack[i];
        float4 wei4 = wei_pack[i];
        out_pack[i] = make_float4(
            scale * in4.x * wei4.x,
            scale * in4.y * wei4.y,
            scale * in4.z * wei4.z,
            scale * in4.w * wei4.w
        );
    }
    for (int32_t i = tail_off + threadIdx.x; i < dim; i += blockDim.x) {
        row_out[i] = scale * row_in[i] * wei[i];
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
    const int32_t size = static_cast<int32_t>(input.size());

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    constexpr float eps = 1e-6f;
#else
    constexpr float eps = 1e-5f;
#endif

    const int32_t block_num = size / dim;
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = dim / pack_size;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    if (pack_num < 256) {
        constexpr int32_t thread_num = 128;
        rmsnorm_2d_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, dim, eps);
    } else if (pack_num < 512) {
        constexpr int32_t thread_num = 256;
        rmsnorm_2d_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, dim, eps);
    } else {
        constexpr int32_t thread_num = 512;
        rmsnorm_2d_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, dim, eps);
    }
}
}  // namespace kernel