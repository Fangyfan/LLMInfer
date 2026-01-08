#include "op/rope.h"
#include "kernel/kernel_interface.h"

namespace op {
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size) 
: Layer(device_type, LayerType::LayerRoPE, "RoPE"), dim_(dim), kv_dim_(kv_dim), head_size_(head_size) {
    reset_inputs_size(5);
}

base::Status RoPELayer::check() const {
    base::Status status;
    const tensor::Tensor& input_q = get_input(0);
    const tensor::Tensor& input_k = get_input(1);
    const tensor::Tensor& input_pos = get_input(2);
    const tensor::Tensor& sin_cache = get_input(3);
    const tensor::Tensor& cos_cache = get_input(4);
    status = check_tensor_with_dim(input_q, device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The input q error in the rope layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(input_k, device_type_, data_type_, kv_dim_);
    if (!status) {
        LOG(ERROR) << "The input k error in the rope layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(input_pos, base::DeviceType::DeviceCPU, base::DataType::DataTypeInt32, 1);
    if (!status) {
        LOG(ERROR) << "The input pos error in the rope layer." << std::endl;
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
    const tensor::Tensor& input_q = get_input(0);
    const tensor::Tensor& input_k = get_input(1);
    const tensor::Tensor& input_pos = get_input(2);
    const tensor::Tensor& sin_cache = get_input(3);
    const tensor::Tensor& cos_cache = get_input(4);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    kernel::get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, 
                                          input_q, input_k, input_pos, sin_cache, cos_cache, 
                                          cuda_config_ ? cuda_config_->stream() : nullptr);
    return base::error::success();
}
}  // namespace op