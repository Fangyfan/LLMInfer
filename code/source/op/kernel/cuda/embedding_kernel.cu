#include "embedding_kernel.cuh"

namespace kernel {
static __global__ void embedding_kernel(
    const int32_t* __restrict__ input, 
    const float* __restrict__ weight, 
    float* output, 
    int32_t dim
) {
    // 每个 Block 负责一个 token 的 embedding 拷贝
    int32_t token_id = input[blockIdx.x];
    
    // 定位 token_id 对应的 Embedding 起始地址
    const float* weight_ptr = weight + token_id * dim;
    
    // 定位 output 第 i 行 (token) 的起始地址
    float* output_ptr = output + blockIdx.x * dim;
    
    // 直接把 weight 这一行 embedding 拷贝到 output 第 i 行
    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        output_ptr[i] = weight_ptr[i];
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
    
    const int32_t token_num = input_cu.get_dim(0);
    const int32_t vocab_size = weight.get_dim(0);
    const int32_t dim = weight.get_dim(1);
    CHECK_EQ(output.get_dim(0), token_num);
    CHECK_EQ(output.get_dim(1), dim);

    for (int32_t i = 0; i < input.size(); ++i) {
        int32_t token_id = input.index<int32_t>(i);
        CHECK_GE(token_id, 0);
        CHECK_LT(token_id, vocab_size);
    }
    
    const int32_t* in = input_cu.ptr<int32_t>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    
    const int32_t block_num = token_num;
    constexpr int32_t thread_num = 256;
    embedding_kernel<<<block_num, thread_num, 0, stream_>>>(in, wei, out, dim);
}
}  // namespace kernel