#include "embedding_kernel.cuh"

namespace kernel {
static __global__ void embedding_kernel(
    const int32_t* __restrict__ input, 
    const float* __restrict__ weight, 
    float* output, 
    int32_t hidden_dim
) {
    int32_t hidden_dim4 = hidden_dim >> 2;
    int32_t token_id = input[blockIdx.x];
    const float4* wei4 = reinterpret_cast<const float4*>(weight + token_id * hidden_dim);
    float4* out4 = reinterpret_cast<float4*>(output + blockIdx.x * hidden_dim);
    for (int32_t i = threadIdx.x; i < hidden_dim4; i += blockDim.x) {
        out4[i] = wei4[i];
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

    for (int32_t i = 0; i < input.size(); ++i) {
        int32_t token_id = input.index<int32_t>(i);
        CHECK_GE(token_id, 0);
        CHECK_LT(token_id, vocab_size);
    }
    
    const int32_t* in = input_cu.ptr<int32_t>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    
    CHECK(hidden_dim % 4 == 0);
    dim3 gridDim(token_num);
    dim3 blockDim(512);
    embedding_kernel<<<gridDim, blockDim, 0, stream_>>>(in, wei, out, hidden_dim);
}
}  // namespace kernel