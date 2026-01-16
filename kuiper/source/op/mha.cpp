#include "op/mha.h"
#include "kernel/kernel_interface.h"

namespace op {
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t kv_dim, int32_t kv_mul, int32_t head_num, int32_t head_size, int32_t max_seq_len)
: Layer(device_type, LayerType::LayerMHA, "MHA"), kv_dim_(kv_dim), kv_mul_(kv_mul), head_num_(head_num), head_size_(head_size), max_seq_len_(max_seq_len) {
    reset_inputs_size(4);
    reset_outputs_size(1);
}

base::Status MultiHeadAttention::check() const {
    base::Status status;
    for (int32_t i = 0; i < inputs_size(); i++) {
        status = check_tensor(get_input(i), device_type_, data_type_);
        if (!status) {
            LOG(ERROR) << "The input tensor " << i << " error in the matmul layer." << std::endl;
            return status;
        }
    }
    status = check_tensor(get_output(0), device_type_, data_type_);
    if (!status) {
        LOG(ERROR) << "The mha output tensor error in the matmul layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status MultiHeadAttention::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    const tensor::Tensor& query = get_input(0);
    const tensor::Tensor& score = get_input(1);
    const tensor::Tensor& key_cache = get_input(2);
    const tensor::Tensor& value_cache = get_input(3);
    const tensor::Tensor& output = get_output(0);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    kernel::get_mha_kernel(device_type_)(layer_id_, pos_, kv_dim_, kv_mul_, head_num_, head_size_, max_seq_len_, 
                                         query, score, key_cache, value_cache, output, 
                                         cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::success();
}

void MultiHeadAttention::set_pos(int32_t pos) {
    pos_ = pos;
}

void MultiHeadAttention::set_layer_id(int32_t layer_id) {
    layer_id_ = layer_id;
}
}  // namespace op