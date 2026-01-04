#include "op/add.h"
#include "kernel/kernel_interface.h"

namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type) : Layer(device_type, LayerType::LayerAdd, "Add") {
    reset_inputs_size(2);
    reset_outputs_size(1);
}

base::Status VecAddLayer::check() const {
    base::Status status;
    const tensor::Tensor& input1 = get_input(0);
    const tensor::Tensor& input2 = get_input(1);
    const tensor::Tensor& output = get_output(0);
    size_t size = input1.size();
    status = check_tensor_with_dim(input1, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The input tensor 1 error in the add layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(input2, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The input tensor 2 error in the add layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(output, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the add layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status VecAddLayer::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    tensor::Tensor& input1 = get_input(0);
    tensor::Tensor& input2 = get_input(1);
    tensor::Tensor& output = get_output(0);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    kernel::get_add_kernel(device_type_)(input1, input2, output, cuda_config_ ? cuda_config_->stream() : nullptr);
    return base::error::success();
}
}  // namespace op