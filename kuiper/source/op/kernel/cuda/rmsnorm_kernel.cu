#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

namespace kernel {
template<int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_fp32(const float *in, const float* wei, float* out, const int32_t size, const float eps) {
    // 每个线程分工，计算部分平方和，无重复 + 全覆盖 + 负载均衡
    float sum = 0.f;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    // Block 级别规约对 Block 内所有线程的值 sum 求和
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    sum = BlockReduce(temp).Sum(sum);
    
    // 把全局总和存到共享内存，让 Block 内所有线程都能读到
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum;
    }
    __syncthreads();
    sum = shared_sum;

    // 每个线程再分工，计算输出值，无重复 + 全覆盖 + 负载均衡
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
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

    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    const int32_t size = static_cast<int32_t>(input.size());
    const float eps = 1e-5f;

    if (size <= 1024) {
        constexpr int32_t thread_num = 128;
        // LOG(INFO) << "size = " << size << ", " << "thread_num = " << thread_num << "\n";
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            row_rmsnorm_fp32<thread_num><<<1, thread_num, 0, stream_>>>(in, wei, out, size, eps);
        } else {
            row_rmsnorm_fp32<thread_num><<<1, thread_num>>>(in, wei, out, size, eps);
        }
    } else {
        constexpr int32_t thread_num = 1024;
        // LOG(INFO) << "size = " << size << ", " << "thread_num = " << thread_num << "\n";
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            row_rmsnorm_fp32<thread_num><<<1, thread_num, 0, stream_>>>(in, wei, out, size, eps);
        } else {
            row_rmsnorm_fp32<thread_num><<<1, thread_num>>>(in, wei, out, size, eps);
        }
    }
}
}  // namespace kernel