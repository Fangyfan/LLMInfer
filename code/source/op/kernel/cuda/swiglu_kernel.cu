#include "swiglu_kernel.cuh"

namespace kernel {
static __global__ void swiglu_kernel_fp32(
    const float* in1, 
    const float* __restrict__ in2, 
    float* out, 
    int32_t size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    float x1 = in1[tid];
    float x2 = in2[tid];
    out[tid] = (x1 / (1 + expf(-x1))) * x2;
}

void swiglu_kernel_cu(
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

    const int32_t size = output.size();
    CHECK_EQ(input1.size(), size);
    CHECK_EQ(input2.size(), size);

    const float* in1 = input1.ptr<float>();
    const float* in2 = input2.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    constexpr int32_t thread_num = 256;
    const int32_t block_num = (size + thread_num - 1) / thread_num;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    swiglu_kernel_fp32<<<block_num, thread_num, 0, stream_>>>(in1, in2, out, size);
}
}  // namespace kernel