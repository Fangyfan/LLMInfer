#include "op/matmul.h"
#include "kernel/kernel_interface.h"

namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, bool is_quant_layer, bool has_bias)
: LayerParam(device_type, LayerType::LayerMatmul, is_quant_layer, "Matmul"), dim0_(dim0), dim1_(dim1), has_bias_(has_bias) {
    reset_inputs_size(1);
    reset_weights_size(1);
    reset_outputs_size(1);
    if (has_bias_) {
        bias_.resize(1);
    }
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
    } else {
        status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::DataTypeInt8, dim0_, dim1_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the matmul layer." << std::endl;
            return status;
        }
        status = check_tensor_with_dim(scales_, device_type_, base::DataType::DataTypeFp32, scales_.size());
        if (!status) {
            LOG(ERROR) << "The scales tensor error in the matmul layer." << std::endl;
            return status;
        }
    }
    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the matmul layer." << std::endl;
        return status;
    }
    if (has_bias_) {
        status = check_tensor_with_dim(get_bias(0), device_type_, data_type_, dim0_);
        if (!status) {
            LOG(ERROR) << "The bias tensor error in the matmul layer." << std::endl;
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
    if (is_quant_layer_) {
        CHECK(device_type_ == base::DeviceType::DeviceCUDA);
        kernel::get_matmul_kernel_quant8(device_type_)(input, weight, output, scales_, group_size_, cuda_config_->stream);
    } else {
        kernel::get_matmul_kernel(device_type_)(input, weight, output, cuda_config_ ? cuda_config_->stream : nullptr);
    }
    if (has_bias_) {
        kernel::get_add_kernel(device_type_)(output, get_bias(0), output, cuda_config_ ? cuda_config_->stream : nullptr);
    }
    return base::error::success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t dim, const void* bias_ptr, base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK_NE(bias_ptr, nullptr);

    // 设置偏置 (bias) 时，确保是非量化层
    CHECK(!is_quant_layer_);

    // 创建 Fp32 类型的 Tensor 偏置参数，同时创建 Buffer 使用 bias_ptr 所指向的外部内存/显存
    tensor::Tensor bias(base::DataType::DataTypeFp32, dim, false, nullptr, const_cast<void*>(bias_ptr));
    bias.set_device_type(device_type);
    bias_.at(idx) = bias;

    return base::error::success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
    LayerParam::to_cuda();
    if (has_bias_) {
        for (tensor::Tensor& bias : bias_) {
            if (!bias.is_empty()) {
                bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
            }
        }
    }
}
}  // namespace op