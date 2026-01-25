#include "softmax_kernel.cuh"
#include "cub/block/block_reduce.cuh"
#include <cfloat>

namespace kernel {
template<int32_t THREAD_PER_BLOCK>
static __global__ void softmax_kernel_fp32(float* __restrict__ x, int32_t size) {
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // 每个线程求自己负责的局部最大值
    float max_val = -FLT_MAX;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        if (max_val < x[i]) {
            max_val = x[i];
        }
    }

    // 块级规约求全局最大值
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());

    // 通过块内共享变量，将全局最大值由 thread 0 广播到块内所有线程
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // 每个线程对自己负责的值更新: 减去全局最大值 x = (x - max)，以至于 exp 后数值稳定，并进行局部求和
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // 块级规约进行全局求和
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());

    // 通过块内共享变量，将全局和由 thread 0 广播到块内所有线程
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // 每个线程对自己负责的值更新: 除以全局和，即 softmax(x) = e^{-x} / sum(e^{-x})
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] /= sum;
    }
}

void softmax_kernel_cu(const tensor::Tensor& input, void* stream) {
    CHECK(!input.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);

    float* in = const_cast<float*>(input.ptr<float>());
    const int32_t size = static_cast<int32_t>(input.size());
    
    constexpr int32_t block_num = 1;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    if (size < 256) {
        constexpr int32_t thread_num = 128;
        softmax_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, size);
    } else if (size < 512) {
        constexpr int32_t thread_num = 256;
        softmax_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, size);
    } else {
        constexpr int32_t thread_num = 512;
        softmax_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, size);
    }
}
}  // namespace kernel