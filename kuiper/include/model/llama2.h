#ifndef KUIPER_INCLUDE_MODEL_LLAMA2_H_
#define KUIPER_INCLUDE_MODEL_LLAMA2_H_

#include "model.h"

namespace model {
struct Llama2Layers {
    // 基础算子 (Non-Parametric Layers): 这些层通常没有训练参数，或者参数很少，主要是执行数学运算
    // embedding_layer_: 嵌入层。负责将输入的 Token ID（比如 392）转换成高维向量（Embedding Vector）
    // rope_layer_: RoPE (Rotary Positional Embeddings) 旋转位置编码。这是 Llama 系列模型的核心特征，用于告诉模型每个 Token 的相对位置
    // add_layer_: 执行残差连接（Residual Connection）中的“加法”操作
    // swiglu_layer_: SwiGLU 激活函数。这是 Llama 对传统 ReLU/GELU 的改进
    // rmsnorm_layers_: RMSNorm (Root Mean Square Normalization)。Llama 使用 RMSNorm 而不是标准的 LayerNorm，用于稳定训练和推理。这是一个 vector，因为每一层 Transformer Block 都有两个 Norm 操作
    std::unique_ptr<op::Layer> embedding_layer_;
    std::unique_ptr<op::Layer> rope_layer_;
    std::unique_ptr<op::Layer> add_layer_;
    std::unique_ptr<op::Layer> swiglu_layer_;
    std::vector<std::unique_ptr<op::Layer>> rmsnorm_layers_;

    // 注意力机制权重 (Attention Weights)
    // 这些是 Transformer 注意力模块的核心矩阵。因为 Llama 2 有很多层（比如 7B 模型有 32 层），所以它们都是 std::vector
    // wq_layers_: W_Q (Query 权重矩阵)
    // wk_layers_: W_K (Key 权重矩阵)
    // wv_layers_: W_V (Value 权重矩阵)
    // wo_layers_: W_O (Output 权重矩阵，注意力计算完后的投影)
    // mha_layer_: Multi-Head Attention 算子。负责执行 Softmax (QK^T / sqrt(d)) * V 的核心计算
    std::vector<std::unique_ptr<op::Layer>> wq_layers_;
    std::vector<std::unique_ptr<op::Layer>> wk_layers_;
    std::vector<std::unique_ptr<op::Layer>> wv_layers_;
    std::vector<std::unique_ptr<op::Layer>> wo_layers_;
    std::unique_ptr<op::Layer> mha_layer_;
    
    // 前馈网络权重 (Feed-Forward Weights - SwiGLU)
    // 标准的 Transformer FFN 只有两个矩阵，但 Llama 使用 SwiGLU 结构，所以有三个矩阵:
    // w1_layers_ (Gate Projection): 门控投影层
    // w2_layers_ (Down Projection): 下降投影层（将维度映射回 hidden_size）
    // w3_layers_ (Up Projection): 上升投影层（通常与 Gate 做点乘）
    std::vector<std::unique_ptr<op::Layer>> w1_layers_;
    std::vector<std::unique_ptr<op::Layer>> w2_layers_;
    std::vector<std::unique_ptr<op::Layer>> w3_layers_;
    
    // 输出头 cls_layer_: Classification Layer (LM Head)。最后的线性层，将向量映射回词表大小（Vocab Size），计算每个词的概率
    std::unique_ptr<op::Layer> cls_layer_;

    // 工具函数 to_cuda(...): 一个辅助函数，用于将上述所有层的数据从 CPU 内存移动到 GPU 显存
    void to_cuda(std::shared_ptr<kernel::CudaConfig> cuda_config);
};

class Llama2Model : public Model {
public:
    explicit Llama2Model(base::TokenizerType tokenizer_type, std::string tokenizer_path, std::string model_path, bool is_quant_model);

    // Llama2 模型初始化
    base::Status init(base::DeviceType device_type) override;

    // Llama2 模型推理的前向传播 + 后处理/采样
    base::Status predict(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos, bool is_prompt, int32_t& next_token_id) const override;

    // Llama2 模型推理的前向传播
    base::Status forward(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos) const override;

    // 把离散的 token id 变成连续的向量
    op::EmbeddingResult embedding(const std::vector<int32_t>& token_ids) const override;

private:
    // 模型缓冲区 (Buffer) 内存分配: 一次性申请并分配推理过程中所需的所有显存/内存
    void allocate_model_buffers() override;

    // 构建网络层与校验: 负责实例化所有层，并检查模型结构是否完整
    base::Status create_layers() override;

    // 创建无参数层: 把不需要训练参数的算子实例化
    void create_nonparam_layers() override;
    
    // 创建非量化 Fp32 可学习参数层: 把需要训练参数的算子实例化
    void create_param_layers() override;

    // 创建量化 Int8 可学习参数层: 把需要训练参数的算子实例化
    void create_param_quant_layers() override;

    // 对输入数据做 RMSNorm: Llama 2 采用 Pre-Norm 结构。这意味着数据在进入注意力模块之前，必须先被归一化，这有助于数值稳定
    void attention_rmsnorm(int32_t layer_id, const tensor::Tensor& input) const;

    // 投影 QKV、缓存 KV、RoPE 旋转位置编码
    void attention_qkv_rope(int32_t layer_id, const tensor::Tensor& token_pos) const;

    // 执行多头注意力机制: 读取 KV Cache（上一轮对话的历史），计算注意力分数，并融合 V 向量，最后乘上 wo 输出矩阵
    void attention_mha(int32_t layer_id, const tensor::Tensor& token_pos) const;

    // 执行一次 Add，即残差连接 x = x + attention_output
    // 再次进行 RMSNorm，执行 SwiGLU 也就是 MLP 块: output = (SiLU(x * w1) @ (x * w3)) w2 ，这里对应代码中的 w1, w3, w2 层
    // 最后再执行一次 Add，即残差连接
    void feed_forward(int32_t layer_id, const tensor::Tensor& input) const;

    // 分类头 (Classification): 最后一层 Transformer 跑完后，将向量映射到词表大小，输出叫 Logits（未归一化的概率分值）
    void cls_logits(const tensor::Tensor& input) const;

    // 后处理/采样 (Sampling): argmax 采样会选择 Logits 最大的 token id
    int32_t post_process(bool is_prompt) const override;

private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
    std::unique_ptr<Llama2Layers> llama2_layers_;
};
}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_LLAMA2_H_