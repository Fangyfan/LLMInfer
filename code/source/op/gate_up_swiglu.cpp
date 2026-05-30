#include "op/gate_up_swiglu.h"
#include "kernel/kernel_interface.h"

namespace op {
GateUpSwigluLayer::GateUpSwigluLayer(base::DeviceType device_type, int32_t immediate_dim, int32_t hidden_dim, bool is_quant_layer)
: LayerParam(device_type, LayerType::LayerMatmul, is_quant_layer, "Matmul"), immediate_dim_(immediate_dim), hidden_dim_(hidden_dim) {
    reset_inputs_size(1);
    reset_weights_size(1);
    reset_outputs_size(1);
}

base::Status GateUpSwigluLayer::check() const {
    base::Status status;
    const tensor::Tensor& input = get_input(0);
    const tensor::Tensor& weight = get_weight(0);
    const tensor::Tensor& output = get_output(0);
    status = check_tensor_with_dim(input, device_type_, data_type_, hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the gate-up-swiglu layer." << std::endl;
        return status;
    }
    if (!is_quant_layer_) {
        status = check_tensor_with_dim(weight, device_type_, data_type_, 2 * immediate_dim_, hidden_dim_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the gate-up-swiglu layer." << std::endl;
            return status;
        }
    } else {
        status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::DataTypeInt4x8, 2 * immediate_dim_ / 8, hidden_dim_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the awq int4 gate-up-swiglu layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(zeros_, device_type_, base::DataType::DataTypeInt4x8, 2 * immediate_dim_ / 8 * hidden_dim_ / 128);
        if (!status) {
            LOG(ERROR) << "The zeros tensor error in the awq int4 gate-up-swiglu layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(scales_, device_type_, base::DataType::DataTypeFp16, 2 * immediate_dim_ * hidden_dim_ / 128);
        if (!status) {
            LOG(ERROR) << "The scales tensor error in the awq int4 gate-up-swiglu layer." << std::endl;
            return status;
        }
    }
    status = check_tensor_with_dim(output, device_type_, data_type_, immediate_dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the gate-up-swiglu layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status GateUpSwigluLayer::forward() {
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
    if (!is_quant_layer_) {
        kernel::get_fused_gate_up_gemv_swiglu_kernel(device_type_)(input, weight, output, immediate_dim_, cuda_config_ ? cuda_config_->stream : nullptr);
    } else {
        kernel::get_fused_gate_up_gemv_swiglu_int4_kernel(device_type_)(input, weight, output, zeros_, scales_, group_size_, 
            immediate_dim_, cuda_config_ ? cuda_config_->stream : nullptr);
    }
    return base::error::success();
}

}  // namespace op