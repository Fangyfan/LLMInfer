#ifndef KUIPER_INCLUDE_OP_LAYER_H
#define KUIPER_INCLUDE_OP_LAYER_H

#include "tensor/tensor.h"
#include "base/cuda_config.h"

namespace op {
enum class LayerType : uint8_t {
    LayerUnknown = 0,   // 未知层（默认值）
    LayerLinear = 1,    // 线性层（全连接层）
    LayerEncode = 2,    // 编码层
    LayerEmbedding = 3, // 嵌入层
    LayerRMSNorm = 4,   // RMS归一化层
    LayerMatmul = 5,    // 矩阵乘法层
    LayerRope = 6,      // RoPE位置编码层
    LayerMHA = 7,       // 多头注意力层
    LayerSoftmax = 8,   // Softmax层
    LayerAdd = 9,       // 加法层
    LayerSwiGLU = 10,   // SwiGLU激活层
};

class BaseLayer {
public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name = "");
    
    virtual base::Status init() = 0; // 层的初始化（如初始化权重、分配内存）
    virtual base::Status check() const = 0; // 检查层的状态（如输入输出维度是否合法、权重是否初始化）

    virtual base::Status forward() = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3,
                                 const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3,
                                 const tensor::Tensor& input4, const tensor::Tensor& output1) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3,
                                 const tensor::Tensor& input4, const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0; // 设置指定索引的输入张量
    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0; // 设置指定索引的输出张量

    virtual size_t inputs_size() const = 0; // 获取输入张量的数量
    virtual size_t outputs_size() const = 0; // 获取输出张量的数量

    virtual tensor::Tensor& get_input(int32_t idx) = 0; // 获取指定索引的输入张量
    virtual tensor::Tensor& get_output(int32_t idx) = 0; // 获取指定索引的输出张量
    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;
    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);
    virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                                    base::DeviceType device_type = base::DeviceType::DeviceUnknown);

    const std::string& layer_name() const;
    LayerType layer_type() const;
    base::DataType data_type() const;
    base::DeviceType device_type() const;

    void set_layer_name(const std::string& layer_name);
    void set_device_type(base::DeviceType device_type);

protected:
    base::DeviceType device_type_ = base::DeviceType::DeviceUnknown; // 层设备类型
    LayerType layer_type_ = LayerType::LayerUnknown; // 层类型
    base::DataType data_type_ = base::DataType::DataTypeUnknown; // 层数据类型
    std::string layer_name_; // 层名
};

// 不带参（权重）算子类
class Layer : public BaseLayer {
public:
    // data_type 需要默认初始化为 fp32
    explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");
    
    base::Status init() override;
    base::Status check() const override;

    // 检查单个张量的设备类型、数据类型是否合法
    base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type) const;

    // ...（可变参数）：用来传入 “期望的维度列表”，比如检查张量是 2 维且维度为 [32, 1024]，就可以传 32, 1024
    base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type, base::DataType data_type, ...) const;

    base::Status forward() override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output1) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3,
                                 const tensor::Tensor& output1) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3,
                                 const tensor::Tensor& input4, const tensor::Tensor& output1) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& input3,
                                 const tensor::Tensor& input4, const tensor::Tensor& input5, const tensor::Tensor& output1) override;
    // set_input(0, x1)
    // set_input(0, x2)
    void set_input(int32_t idx, const tensor::Tensor& input) override; // 设置指定索引的输入张量（传入输入）
    
    // set_output(0, y)
    void set_output(int32_t idx, const tensor::Tensor& output) override; // 设置指定索引的输出张量（传入输出）

    size_t inputs_size() const override; // 获取输入张量的数量 inputs_.size()
    size_t outputs_size() const override; // 获取输出张量的数量 outputs_.size()

    tensor::Tensor& get_input(int32_t idx) override; // 获取指定索引的输入张量
    tensor::Tensor& get_output(int32_t idx) override; // 获取指定索引的输出张量
    const tensor::Tensor& get_input(int32_t idx) const override;
    const tensor::Tensor& get_output(int32_t idx) const override;

    void reset_inputs_size(size_t size); // 重置输入张量的数量 inputs_.size()
    void reset_outputs_size(size_t size); // 重置输出张量的数量 outputs_.size()

    void set_cuda_config(std::shared_ptr<kernel::CudaConfig> cuda_config);
    std::shared_ptr<kernel::CudaConfig> cuda_config() const;

    virtual void to_cuda();

protected:
    std::vector<tensor::Tensor> inputs_; // 存放每个算子中的输入张量
    std::vector<tensor::Tensor> outputs_; // 存放每个算子中的输出张量
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

class LayerParam : public Layer {
public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer = false, std::string layer_name = "");
    
    size_t weights_size() const; // 获取权重的数量 weights_.size()
    void reset_weights_size(size_t size); // 重置权重的数量 weights_.size()
    
    // 通过索引获取权重，比如 Linear 层 get_weight(0) = W 且 get_weight(1) = b
    tensor::Tensor& get_weight(int32_t idx);
    const tensor::Tensor& get_weight(int32_t idx) const;
    
    // 直接设置指定索引的权重张量（存入 weights_）
    // 用途：直接传入一个已初始化的张量作为权重（比如从其他层复用权重）
    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;
    
    // 从原始内存指针加载权重（指定维度、数据指针、设备类型）
    // 用途：从模型文件加载权重，比如 从磁盘读取 权重数据 到内存指针 weight_ptr，指定张量维度 dims，就能直接 初始化 权重张量 并存入 weights_
    base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, base::DeviceType device_type) override;
    
    void set_scales(const tensor::Tensor& scales);
    void set_group_size(int32_t group_size);
    int32_t get_scales_size() const; // 获取 scales 张量的大小 (元素个数)

    void to_cuda() override; // 重写 Layer 的 to_cuda 函数，把层的所有数据转移到 cuda

protected:
    bool is_quant_layer_ = false;           // 标记是否为: 量化层
    int32_t group_size_ = 0;                // 分组量化: 每次取连续的 group_size_ 个权重参数进行 Int8 对称量化，共享 1 个 scale 系数
    tensor::Tensor scales_;                 // 缩放因子张量 (x_{fp32} = x_{int8} * scale) 
    std::vector<tensor::Tensor> weights_;   // 算子中的所有权重（可能有多个，比如 Linear 层需要 2 个权重 W 和 b）
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_LAYER_H