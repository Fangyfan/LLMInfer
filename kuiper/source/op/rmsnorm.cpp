#include "op/rmsnorm.h"
#include "kernel/kernel_interface.h"

namespace op {
RMSNormLaryer::RMSNormLaryer(base::DeviceType device_type, int32_t dim)
: LayerParam(device_type, LayerType::LayerRMSNorm, false, "RMSNorm"), dim_(dim) {
    reset_inputs_size(1);
    reset_outputs_size(1);
    reset_weights_size(1);
}

base::Status RMSNormLaryer::check() const {
    base::Status status;
    const tensor::Tensor& input = get_input(0); // 输入向量
    const tensor::Tensor& weight = get_weight(0); // 可学习的缩放因子
    const tensor::Tensor& output = get_output(0); // 输出均方根归一化向量
    status = check_tensor_with_dim(input, device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the rmsnorm layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(weight, device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the rmsnorm layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(output, device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the rmsnorm layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status RMSNormLaryer::forward() {
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
    kernel::get_rmsnorm_kernel(device_type_)(input, weight, output, cuda_config_ ? cuda_config_->stream() : nullptr);
    return base::error::success();
}
}  // namespace op