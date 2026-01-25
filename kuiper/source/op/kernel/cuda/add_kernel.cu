#include "add_kernel.cuh"

namespace kernel {
static __global__ void add_kernel_fp32(
    const float* in1, 
    const float* __restrict__ in2, 
    float* out, 
    int32_t size
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    out[tid] = in1[tid] + in2[tid];
}

void add_kernel_cu(
    const tensor::Tensor& input1, 
    const tensor::Tensor& input2, 
    const tensor::Tensor& output, 
    void* stream
) {
    CHECK(!input1.is_empty());
    CHECK(!input2.is_empty());
    CHECK(!output.is_empty());
    CHECK(input1.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(input2.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    int32_t size = static_cast<int32_t>(output.size());
    CHECK_EQ(input1.size(), size);
    CHECK_EQ(input2.size(), size);

    const float* in1 = input1.ptr<float>();
    const float* in2 = input2.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    constexpr int32_t thread_num = 256;
    const int32_t block_num = (size + thread_num - 1) / thread_num;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    
    // 块数量: block_num
    // 每个块内的线程数量: thread_num
    // 共享内存大小: 0 表示不需要共享内存
    // 执行核函数的 CUDA 流（任务队列）: stream_ 异步执行，当 stream_ 为 0 就是默认流同步执行省略
    add_kernel_fp32<<<block_num, thread_num, 0, stream_>>>(in1, in2, out, size);
}
}  // namespace kernel