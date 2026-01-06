#include "op/embedding.h"
#include "kernel/kernel_interface.h"

namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size)
: LayerParam(device_type, LayerType::LayerEmbedding, false, "Embedding"), dim_(dim), seq_len_(seq_len), vocab_size_(vocab_size) {
    reset_inputs_size(2);
    reset_weights_size(1);
    reset_outputs_size(1);
}

base::Status EmbeddingLayer::check() const {
    base::Status status;
    const tensor::Tensor& input = get_input(0); // 输入 tokens，每个 token 是离散的 id，在 [0, vocab_size) 范围内
    int32_t token_num = static_cast<int32_t>(get_input(1).size()); // 输入 token 的数量
    status = check_tensor_with_dim(input, base::DeviceType::DeviceCPU, base::DataType::DataTypeInt32, token_num);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the embedding layer.";
        return status;
    }
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }
    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_num, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }
    return base::error::success();
}

base::Status EmbeddingLayer::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    const tensor::Tensor& input = get_input(0);
    const tensor::Tensor& weight = get_weight(0);
    const tensor::Tensor& output = get_output(0);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    kernel::get_embedding_kernel(device_type_)(input, weight, output, vocab_size_, cuda_config_ ? cuda_config_->stream() : nullptr);
    return base::error::success();
}
}  // namespace op