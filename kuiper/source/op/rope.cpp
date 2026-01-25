#include "op/rope.h"
#include "kernel/kernel_interface.h"

namespace op {
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_dim) 
: Layer(device_type, LayerType::LayerRoPE, "RoPE"), dim_(dim), kv_dim_(kv_dim), head_dim_(head_dim) {
    reset_inputs_size(5);
    reset_outputs_size(1); // 虽然没有用 output 但是需要匹配基类 Layer 层的 forward 方法格式
}

base::Status RoPELayer::check() const {
    base::Status status;
    const tensor::Tensor& query = get_input(0);
    const tensor::Tensor& key = get_input(1);
    const tensor::Tensor& token_pos = get_input(2);
    const tensor::Tensor& sin_cache = get_input(3);
    const tensor::Tensor& cos_cache = get_input(4);
    status = check_tensor_with_dim(query, device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The input query error in the rope layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(key, device_type_, data_type_, kv_dim_);
    if (!status) {
        LOG(ERROR) << "The input key error in the rope layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(token_pos, base::DeviceType::DeviceCPU, base::DataType::DataTypeInt32, 1);
    if (!status) {
        LOG(ERROR) << "The input token pos error in the rope layer." << std::endl;
        return status;
    }
    status = check_tensor(sin_cache, device_type_, data_type_);
    if (!status) {
        LOG(ERROR) << "The sin cache error in the rope layer." << std::endl;
        return status;
    }
    status = check_tensor(cos_cache, device_type_, data_type_);
    if (!status) {
        LOG(ERROR) << "The cos cache error in the rope layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status RoPELayer::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    const tensor::Tensor& query = get_input(0);
    const tensor::Tensor& key = get_input(1);
    const tensor::Tensor& token_pos = get_input(2);
    const tensor::Tensor& sin_cache = get_input(3);
    const tensor::Tensor& cos_cache = get_input(4);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    kernel::get_rope_kernel(device_type_)(query, key, token_pos, sin_cache, cos_cache, dim_, kv_dim_, head_dim_, 
                                          cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::success();
}
}  // namespace op