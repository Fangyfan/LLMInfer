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
    base::Status status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the rmsnorm layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the rmsnorm layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
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