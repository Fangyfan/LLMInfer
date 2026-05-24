#include "op/qkv_matmul.h"
#include "kernel/kernel_interface.h"

namespace op {
QKVMatmulLayer::QKVMatmulLayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t hidden_dim)
: LayerParam(device_type, LayerType::LayerMatmul, false, "Matmul"), dim_(dim), kv_dim_(kv_dim), hidden_dim_(hidden_dim) {
    reset_inputs_size(4);
    reset_weights_size(1);
    reset_outputs_size(1); // 虽然没有用 output 但是需要匹配基类 Layer 层的 forward 方法格式
}

base::Status QKVMatmulLayer::check() const {
    base::Status status = check_tensor_with_dim(get_input(0), device_type_, data_type_, hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_ + 2 * kv_dim_, hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The output query tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_output(1), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The query tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_output(2), device_type_, data_type_, kv_dim_);
    if (!status) {
        LOG(ERROR) << "The key tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_output(3), device_type_, data_type_, kv_dim_);
    if (!status) {
        LOG(ERROR) << "The value tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status QKVMatmulLayer::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    const tensor::Tensor& input = get_input(0);
    const tensor::Tensor& query = get_input(1);
    const tensor::Tensor& key   = get_input(2);
    const tensor::Tensor& value = get_input(3);
    const tensor::Tensor& weight = get_weight(0);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    // kernel::get_fused_qkv_gemv_kernel(device_type_)(input, weight, query, key, value, cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::success();
}

}  // namespace op