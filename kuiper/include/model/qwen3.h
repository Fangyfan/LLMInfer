#ifndef KUIPER_INCLUDE_MODEL_QWEN3_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_H_

#include "model.h"

namespace model {
struct Qwen3Layers {
    std::unique_ptr<op::Layer> embedding_layer_;
    std::unique_ptr<op::Layer> rope_layer_;
    std::unique_ptr<op::Layer> add_layer_;
    std::unique_ptr<op::Layer> swiglu_layer_;
    std::vector<std::unique_ptr<op::Layer>> rmsnorm_layers_;

    std::vector<std::unique_ptr<op::Layer>> wq_layers_;
    std::vector<std::unique_ptr<op::Layer>> wk_layers_;
    std::vector<std::unique_ptr<op::Layer>> wv_layers_;
    std::vector<std::unique_ptr<op::Layer>> wo_layers_;
    std::unique_ptr<op::Layer> mha_layer_;
    
    std::vector<std::unique_ptr<op::Layer>> w1_layers_;
    std::vector<std::unique_ptr<op::Layer>> w2_layers_;
    std::vector<std::unique_ptr<op::Layer>> w3_layers_;
    
    std::unique_ptr<op::Layer> cls_layer_;

    void to_cuda(std::shared_ptr<kernel::CudaConfig> cuda_config);
};

class Qwen3Model : public Model {
public:
    explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string tokenizer_path, std::string model_path, bool is_quant_model);

    // Qwen3 模型初始化
    base::Status init(base::DeviceType device_type) override;

    // Qwen3 模型推理的前向传播 + 后处理/采样
    base::Status predict(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos, bool is_prompt, int32_t& next_token_id) const override;

    // Qwen3 模型推理的前向传播
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

    // 对输入数据做 RMSNorm: Qwen3 采用 Pre-Norm 结构。这意味着数据在进入注意力模块之前，必须先被归一化，这有助于数值稳定
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
    std::unique_ptr<Qwen3Layers> qwen3_layers_;
};
}  // namespace model

#endif