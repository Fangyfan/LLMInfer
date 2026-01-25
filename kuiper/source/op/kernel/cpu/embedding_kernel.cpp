#include "embedding_kernel.h"

namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty() && input.dims_size() == 1);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty() && output.dims_size() == 2);
    CHECK(input.device_type() == base::DeviceType::DeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::DeviceCPU);
    CHECK(output.device_type() == base::DeviceType::DeviceCPU);
    
    const int32_t token_num = input.get_dim(0);
    const int32_t vocab_size = weight.get_dim(0);
    const int32_t dim = weight.get_dim(1);
    CHECK_EQ(output.get_dim(0), token_num);
    CHECK_EQ(output.get_dim(1), dim);

    for (int32_t i = 0; i < token_num; ++i) {
        int32_t token_id = input.index<int32_t>(i);
        if (token_id < 0 || token_id >= vocab_size) {
            LOG(FATAL) << "Token index is greater than vocab size." << std::endl;
            return;
        }
        float* weight_ptr = const_cast<float*>(weight.ptr<float>(token_id * dim)); // 定位 token_id 对应的 Embedding 起始地址
        float* output_ptr = const_cast<float*>(output.ptr<float>(i * dim)); // 定位 output 第 i 行的起始地址
        memcpy(output_ptr, weight_ptr, dim * sizeof(float)); // 直接把 weight 这一行 embedding 拷贝到 output 第 i 行
    }
}
}  // namespace kernel