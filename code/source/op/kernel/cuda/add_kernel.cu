#include "add_kernel.cuh"
#include <cuda_bf16.h>

namespace kernel {
static __global__ void add_bf16_kernel(
    const __nv_bfloat16* in1, 
    const __nv_bfloat16* __restrict__ in2, 
    __nv_bfloat16* out, 
    int32_t size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float v1 = __bfloat162float(in1[tid]);
        float v2 = __bfloat162float(in2[tid]);
        out[tid] = __float2bfloat16(v1 + v2);
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

    CHECK(input1.data_type() == base::DataType::DataTypeBf16);
    CHECK(input2.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == base::DataType::DataTypeBf16);

    CHECK(input1.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(input2.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    int32_t size = static_cast<int32_t>(output.size());
    CHECK_EQ(input1.size(), size);
    CHECK_EQ(input2.size(), size);

    const __nv_bfloat16* in1 = reinterpret_cast<const __nv_bfloat16*>(input1.ptr<uint16_t>());
    const __nv_bfloat16* in2 = reinterpret_cast<const __nv_bfloat16*>(input2.ptr<uint16_t>());
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    add_bf16_kernel<<<gridDim, blockDim, 0, stream_>>>(in1, in2, out, size);
}
}  // namespace kernel