#include "op/qkv_matmul.h"
#include "kernel/kernel_interface.h"

namespace op {
QKVMatmulLayer::QKVMatmulLayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t hidden_dim, bool is_quant_layer)
: LayerParam(device_type, LayerType::LayerMatmul, is_quant_layer, "Matmul"), dim_(dim), kv_dim_(kv_dim), hidden_dim_(hidden_dim) {
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
    if (!is_quant_layer_) {
        status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_ + 2 * kv_dim_, hidden_dim_);
        if (!status) {
            LOG(ERROR) << "The output query tensor error in the qkv-matmul layer." << std::endl;
            return status;
        }
    } else {
        status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::DataTypeInt4x8, (dim_ + 2 * kv_dim_) / 8, hidden_dim_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the awq int4 qkv-matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(zeros_, device_type_, base::DataType::DataTypeInt4x8, (dim_ + 2 * kv_dim_) / 8 * hidden_dim_ / 128);
        if (!status) {
            LOG(ERROR) << "The zeros tensor error in the awq int4 qkv-matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(scales_, device_type_, base::DataType::DataTypeFp16, (dim_ + 2 * kv_dim_) * hidden_dim_ / 128);
        if (!status) {
            LOG(ERROR) << "The scales tensor error in the awq int4 qkv-matmul layer." << std::endl;
            return status;
        }
    }
    status = check_tensor_with_dim(get_input(1), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The query tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_input(2), device_type_, data_type_, kv_dim_);
    if (!status) {
        LOG(ERROR) << "The key tensor error in the qkv-matmul layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_input(3), device_type_, data_type_, kv_dim_);
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
    if (!is_quant_layer_) {
        kernel::get_fused_qkv_gemv_kernel(device_type_)(input, weight, query, key, value, cuda_config_ ? cuda_config_->stream : nullptr);
    } else {
        kernel::get_fused_qkv_gemv_int4_kernel(device_type_)(input, weight, query, key, value, zeros_, scales_, group_size_, 
            cuda_config_ ? cuda_config_->stream : nullptr);
    }
    return base::error::success();
}

}  // namespace op