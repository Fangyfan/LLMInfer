#include "op/matmul.h"
#include "kernel/kernel_interface.h"

namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, bool lm_head, bool fuse_add, bool is_quant_layer)
: LayerParam(device_type, LayerType::LayerMatmul, is_quant_layer, "Matmul"), dim0_(dim0), dim1_(dim1), lm_head_(lm_head), fuse_add_(fuse_add) {
    reset_inputs_size(1);
    reset_weights_size(1);
    reset_outputs_size(1);
}

base::Status MatmulLayer::check() const {
    base::Status status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the matmul layer." << std::endl;
        return status;
    }
    if (!is_quant_layer_) {
        status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(get_output(0), device_type_, !lm_head_ ? data_type_ : base::DataType::DataTypeFp32, dim0_);
        if (!status) {
            LOG(ERROR) << "The output tensor error in the matmul layer." << std::endl;
            return status;
        }
    } else {
        status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::DataTypeInt4x8, dim0_, dim1_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the awq int4 matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(zeros_, device_type_, base::DataType::DataTypeInt4x8, dim0_ * dim1_ / 128);
        if (!status) {
            LOG(ERROR) << "The zeros tensor error in the awq int4 matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(scales_, device_type_, base::DataType::DataTypeFp16, 8 * dim0_ * dim1_ / 128);
        if (!status) {
            LOG(ERROR) << "The scales tensor error in the awq int4 matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(get_output(0), device_type_, !lm_head_ ? data_type_ : base::DataType::DataTypeFp32, 8 * dim0_);
        if (!status) {
            LOG(ERROR) << "The output tensor error in the awq int4 matmul layer." << std::endl;
            return status;
        }
    }
    return base::error::success();
}

base::Status MatmulLayer::forward() {
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
        if (!fuse_add_) {
            kernel::get_gemv_kernel(device_type_)(input, weight, output, lm_head_, cuda_config_ ? cuda_config_->stream : nullptr);
        } else {
            kernel::get_fused_gemv_add_kernel(device_type_)(input, weight, output, cuda_config_ ? cuda_config_->stream : nullptr);
        }
    } else {
        if (!fuse_add_) {
            CHECK(lm_head_);
            kernel::get_gemv_kernel(device_type_)(input, weight, output, lm_head_, cuda_config_ ? cuda_config_->stream : nullptr);
        } else {
            kernel::get_fused_gemv_add_int4_kernel(device_type_)(input, weight, output, zeros_, scales_, group_size_, cuda_config_->stream);
        }
    }
    return base::error::success();
}

}  // namespace op