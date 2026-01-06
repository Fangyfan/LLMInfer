#include "op/swiglu.h"
#include "kernel/kernel_interface.h"

namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
: Layer(device_type, LayerType::LayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
    reset_inputs_size(2);
    reset_outputs_size(1);
}

base::Status SwiGLULayer::check() const {
    base::Status status;
    const tensor::Tensor& input1 = get_input(0);
    const tensor::Tensor& input2 = get_input(1);
    const tensor::Tensor& output = get_output(0);
    status = check_tensor_with_dim(input1, device_type_, data_type_, hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor 1 error in the swiglu layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(input2, device_type_, data_type_, hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The input tensor 2 error in the swiglu layer." << std::endl;
        return status;
    }
    status = check_tensor_with_dim(output, device_type_, data_type_, hidden_dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the swiglu layer." << std::endl;
        return status;
    }
    return base::error::success();
}

base::Status SwiGLULayer::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    const tensor::Tensor& input1 = get_input(0);
    const tensor::Tensor& input2 = get_input(1);
    const tensor::Tensor& output = get_output(0);
    if (device_type_ == base::DeviceType::DeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
    }
    kernel::get_swiglu_kernel(device_type_)(input1, input2, output, cuda_config_ ? cuda_config_->stream() : nullptr);
    return base::error::success();
}
}  // namespace op