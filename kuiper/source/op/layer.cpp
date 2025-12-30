#include <cstdarg>
#include <numeric>
#include "op/layer.h"

namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name)
: device_type_(device_type), layer_type_(layer_type), data_type_(data_type), layer_name_(std::move(layer_name)) {}

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
    return base::error::function_not_implement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, base::DeviceType device_type) {
    return base::error::function_not_implement();
}

const std::string& BaseLayer::layer_name() const {
    return layer_name_;
}

LayerType BaseLayer::layer_type() const {
    return layer_type_;
}

base::DataType BaseLayer::data_type() const {
    return data_type_;
}

base::DeviceType BaseLayer::device_type() const {
    return device_type_;
}

void BaseLayer::set_layer_name(const std::string& layer_name) {
    layer_name_ = layer_name;
}

void BaseLayer::set_device_type(base::DeviceType device_type) {
    device_type_ = device_type;
}

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
: BaseLayer(device_type, layer_type, base::DataType::DataTypeFp32, layer_name) {}

base::Status Layer::init() {
    return base::error::success();
}

base::Status Layer::check() const {
    return base::error::function_not_implement("The check function is not implement yet");
}

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const {
    if (tensor.is_empty()) {
        return base::error::invalid_argument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::invalid_argument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::invalid_argument("The tensor has a wrong data type.");
    }
    return base::error::success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type, ...) const {
    // 开头声明 va_list，确保作用域覆盖整个函数
    std::va_list args;

    if (tensor.is_empty()) {
        return base::error::invalid_argument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::invalid_argument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::invalid_argument("The tensor has a wrong data type.");
    }
    
    va_start(args, data_type); // 初始化可变参数：从 data_type 之后开始读取可变参数
    int32_t size = tensor.dims_size();
    for (int32_t i = 0; i < size; i++) {
        // 从可变参数列表中读取一个 int32_t 类型的参数（即期望的第 i 维大小）
        int32_t dim = va_arg(args, int32_t);
        if (dim != tensor.get_dim(i)) {
            va_end(args);  // 关键：提前 return 前先释放 va_list
            return base::error::invalid_argument("The tensor has a wrong dim in dim" + std::to_string(i));
        }
    }
    va_end(args); // 结束可变参数列表的遍历（必须调用，释放资源）
    
    return base::error::success();
}

base::Status Layer::forward() {
    return base::error::function_not_implement();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
    set_input(0, input1);
    set_output(0, output1);
    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output1) {
    set_input(0, input1);
    set_input(1, input2);
    set_output(0, output1);
    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3, const tensor::Tensor& output1) {
    set_input(0, input1);
    set_input(1, input2);
    set_input(2, input3);
    set_output(0, output1);
    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3, const tensor::Tensor& input4, const tensor::Tensor& output1) {
    set_input(0, input1);
    set_input(1, input2);
    set_input(2, input3);
    set_input(3, input4);
    set_output(0, output1);
    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3, const tensor::Tensor& input4, const tensor::Tensor& input5, const tensor::Tensor& output1) {
    set_input(0, input1);
    set_input(1, input2);
    set_input(2, input3);
    set_input(3, input4);
    set_input(4, input5);
    set_output(0, output1);
    return forward();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    CHECK(input.data_type() == data_type_);
    if (!input.is_empty()) {
        CHECK(input.device_type() == device_type_);
    }
    inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    CHECK(output.data_type() == data_type_);
    if (!output.is_empty()) {
        CHECK(output.device_type() == device_type_);
    }
    outputs_.at(idx) = output;
}

size_t Layer::input_size() const {
    return inputs_.size();
}

size_t Layer::output_size() const {
    return outputs_.size();
}

tensor::Tensor& Layer::get_input(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) {
    inputs_.resize(size);
}

void Layer::reset_output_size(size_t size) {
    outputs_.resize(size);
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> cuda_config) {
    if (!cuda_config) {
        return;
    }
    cuda_config_ = cuda_config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const {
    return cuda_config_;
}

void Layer::to_cuda() {
    for (tensor::Tensor& input : inputs_) {
        if (!input.is_empty()) {
            input.to_cuda(cuda_config_ ? cuda_config_->stream() : nullptr);
        }
    }
    for (tensor::Tensor& output : outputs_) {
        if (!output.is_empty()) {
            output.to_cuda(cuda_config_ ? cuda_config_->stream() : nullptr);
        }
    }
}

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

size_t LayerParam::weight_size() const {
    return weights_.size();
}

void LayerParam::reset_weight_size(size_t size) {
    weights_.resize(size);
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK(weight.data_type() == base::DataType::DataTypeFp32);
    if (!weight.is_empty()) {
        CHECK(weight.device_type() == device_type_);
    }
    weights_.at(idx) = weight;
    return base::error::success();
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK_NE(weight_ptr, nullptr);
    
    size_t data_size = base::data_type_size(base::DataType::DataTypeFp32);
    size_t byte_size = std::accumulate(dims.begin(), dims.end(), data_size, std::multiplies<size_t>());
    auto buffer = std::make_shared<base::Buffer>(byte_size, nullptr, const_cast<void*>(weight_ptr), true); // 使用外部内存/显存
    if (device_type != base::DeviceType::DeviceUnknown) {
        buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_) {
        tensor::Tensor weight(base::DataType::DataTypeFp32, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assgin(buffer));
        weights_.at(idx) = weight;
    }

    return base::error::success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
    CHECK(!scales.is_empty());
    CHECK(scales.data_type() == data_type_);
    CHECK(scales.device_type() == device_type_);
    scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) {
    group_size_ = group_size;
}

int32_t LayerParam::get_scale_num() const {
    CHECK(!scales_.is_empty());
    return static_cast<int32_t>(scales_.size());
}

void LayerParam::to_cuda() {
    Layer::to_cuda();
    for (tensor::Tensor& weight : weights_) {
        if (!weight.is_empty()) {
            weight.to_cuda(cuda_config_ ? cuda_config_->stream() : nullptr);
        }
    }
    if (!scales_.is_empty()) {
        scales_.to_cuda(cuda_config_ ? cuda_config_->stream() : nullptr);
    }
}
}  // namespace op