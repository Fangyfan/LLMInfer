#include "add_kernel.cuh"

namespace kernel {
static __global__ void add_kernel_fp32(
    const float* in1, 
    const float* __restrict__ in2, 
    float* out, 
    int32_t size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = in1[tid] + in2[tid];
    }
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

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    
    add_kernel_fp32<<<gridDim, blockDim, 0, stream_>>>(in1, in2, out, size);
}
}  // namespace kernel