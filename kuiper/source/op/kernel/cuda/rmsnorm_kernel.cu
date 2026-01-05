#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {    
template<int32_t WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int32_t mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<int32_t NUM_THREADS>
static __device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int32_t warp = threadIdx.x / WARP_SIZE;
    int32_t lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    val = warp_reduce_sum<WARP_SIZE>(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = lane < NUM_WARPS ? shared[lane] : 0.f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

template<int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_fp32_old(float *in, float* wei, float* out, const int32_t size, const float eps) {
    // 每个线程分工，计算部分平方和，无重复 + 全覆盖 + 负载均衡
    float sum = 0.f;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    // Block 级别规约对 Block 内所有线程的值 sum 求和，这段代码等价于:
    // sum = block_reduce_sum<BLOCK_DIM>(sum);
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    sum = BlockReduce(temp).Sum(sum);
    
    // 把全局总和存到共享内存，让 Block 内所有线程都能读到
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum;
    }
    __syncthreads(); // 这里涉及到共享变量 shared_sum 的读取，必须要同步
    sum = shared_sum;

    // 每个线程再分工，计算输出值，无重复 + 全覆盖 + 负载均衡
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        out[i] = scale * in[i] * wei[i];
    }
}

template<int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_fp32(float* in, float* wei, float* out, const int32_t size, const float eps) {
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = size / pack_size;
    const int32_t pack_off = pack_num * pack_size;

    // 每个线程分工，计算部分平方和，无重复 + 全覆盖 + 负载均衡 + 向量化(float4)存取
    float sum = 0.f;
    float4* in_pack = reinterpret_cast<float4*>(in);
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in_float4 = in_pack[i];
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }
    for (int32_t i = pack_off + threadIdx.x; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    // Block 级别规约对 Block 内所有线程的值 sum 求和，这段代码等价于:
    // sum = block_reduce_sum<BLOCK_DIM>(sum);
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    sum = BlockReduce(temp).Sum(sum);

    // 把全局总和存到共享内存，让 Block 内所有线程都能读到
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum;
    }
    __syncthreads(); // 这里涉及到共享变量 shared_sum 的读取，必须要同步
    sum = shared_sum;
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    // 每个线程再分工，计算输出值，无重复 + 全覆盖 + 负载均衡 + 向量化(float4)存取
    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(out);
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in_float4 = in_pack[i];
        float4 wei_float4 = wei_pack[i];
        out_pack[i] = make_float4(
            scale * in_float4.x * wei_float4.x,
            scale * in_float4.y * wei_float4.y,
            scale * in_float4.z * wei_float4.z,
            scale * in_float4.w * wei_float4.w
        );
    }
    for (int32_t i = pack_off + threadIdx.x; i < size; i += blockDim.x) {
        out[i] = scale * in[i] * wei[i];
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    float* in = const_cast<float*>(input.ptr<float>());
    float* wei = const_cast<float*>(weight.ptr<float>());
    float* out = const_cast<float*>(output.ptr<float>());
    const int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-5f;
    
    cudaStream_t stream_ = nullptr;
    if (stream) {
        stream_ = static_cast<cudaStream_t>(stream);
    }

    if (size <= 1024) {
        constexpr int32_t thread_num = 128;
        // LOG(INFO) << "size = " << size << ", " << "thread_num = " << thread_num << std::endl;
        if (stream_) {
            row_rmsnorm_fp32<thread_num><<<1, thread_num, 0, stream_>>>(in, wei, out, size, eps);
        } else {
            row_rmsnorm_fp32<thread_num><<<1, thread_num>>>(in, wei, out, size, eps);
        }
    } else {
        constexpr int32_t thread_num = 1024;
        // LOG(INFO) << "size = " << size << ", " << "thread_num = " << thread_num << std::endl;
        if (stream_) {
            row_rmsnorm_fp32<thread_num><<<1, thread_num, 0, stream_>>>(in, wei, out, size, eps);
        } else {
            row_rmsnorm_fp32<thread_num><<<1, thread_num>>>(in, wei, out, size, eps);
        }
    }
}
}  // namespace kernel