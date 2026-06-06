#include "embedding_kernel.cuh"
#include <cuda_bf16.h>

namespace kernel {
static __global__ void embedding_bf16x8_kernel(
    const int32_t* __restrict__ input, 
    const __nv_bfloat16* __restrict__ weight, 
    __nv_bfloat16* output, 
    int32_t hidden_dim
) {
    int32_t hidden_dim8 = (hidden_dim >> 3);
    int32_t token_id = input[blockIdx.x];
    const uint4* wei8 = reinterpret_cast<const uint4*>(weight + token_id * hidden_dim);
    uint4* out8 = reinterpret_cast<uint4*>(output + blockIdx.x * hidden_dim);
    for (int32_t i = threadIdx.x; i < hidden_dim8; i += blockDim.x) {
        out8[i] = wei8[i];
    }
}

void embedding_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 2);

    CHECK(input.data_type() == base::DataType::DataTypeInt32);
    CHECK(weight.data_type() == base::DataType::DataTypeBf16);
    CHECK(output.data_type() == base::DataType::DataTypeBf16);

    CHECK(input.device_type() == base::DeviceType::DeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    tensor::Tensor input_cu = input.clone();
    input_cu.to_cuda(stream_);
    
    const int32_t token_num = input.get_dim(0);
    const int32_t vocab_size = weight.get_dim(0);
    const int32_t hidden_dim = weight.get_dim(1);
    CHECK_EQ(output.get_dim(0), token_num);
    CHECK_EQ(output.get_dim(1), hidden_dim);

    CHECK_EQ(reinterpret_cast<uintptr_t>(weight.ptr<uint16_t>()) % 16, 0);
    CHECK_EQ(reinterpret_cast<uintptr_t>(output.ptr<uint16_t>()) % 16, 0);

    for (int32_t i = 0; i < input.size(); ++i) {
        int32_t token_id = input.index<int32_t>(i);
        CHECK_GE(token_id, 0);
        CHECK_LT(token_id, vocab_size);
    }
    
    const int32_t* in = input_cu.ptr<int32_t>();
    const __nv_bfloat16* wei = reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(output.ptr<uint16_t>()));
    
    CHECK(hidden_dim % 8 == 0);
    dim3 gridDim(token_num);
    dim3 blockDim(512);
    embedding_bf16x8_kernel<<<gridDim, blockDim, 0, stream_>>>(in, wei, out, hidden_dim);
    cudaStreamSynchronize(stream_);
}
}  // namespace kernel