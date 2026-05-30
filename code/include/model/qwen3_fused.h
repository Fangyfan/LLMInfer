#ifndef CODE_INCLUDE_MODEL_QWEN3_FUSED_H_
#define CODE_INCLUDE_MODEL_QWEN3_FUSED_H_

#include "model.h"

namespace model {
struct Qwen3FusedLayers {
    std::unique_ptr<op::Layer> embedding_layer_;
    
    std::vector<std::unique_ptr<op::Layer>> pre_rmsnorm_layers_;
    std::vector<std::unique_ptr<op::Layer>> fused_qkv_proj_layers_;
    std::vector<std::unique_ptr<op::Layer>> fused_qk_norm_rope_layers_;
    std::unique_ptr<op::Layer> flashdecoding_gqa_layer_;
    std::vector<std::unique_ptr<op::Layer>> fused_o_proj_add_layers_;
    
    std::vector<std::unique_ptr<op::Layer>> ffn_rmsnorm_layers_;
    std::vector<std::unique_ptr<op::Layer>> fused_gate_up_swiglu_layers_;
    std::vector<std::unique_ptr<op::Layer>> fused_down_proj_add_layers_;
    
    std::unique_ptr<op::Layer> final_rmsnorm_layer_;
    std::unique_ptr<op::Layer> lm_head_layer_;

    void to_cuda(std::shared_ptr<kernel::CudaConfig> cuda_config);
};

class Qwen3FusedModel : public Model {
public:
    explicit Qwen3FusedModel(base::TokenizerType tokenizer_type, std::string tokenizer_path, std::string model_path, bool is_quant_model);

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
    
    // 创建非量化 BF16 可学习参数层: 把需要训练参数的算子实例化
    void create_param_layers() override;

    // 创建 AWQ INT4 量化可学习参数层: 把需要训练参数的算子实例化
    void create_param_awq_int4_layers() override;

    // 对输入数据做 RMSNorm、投影 QKV、缓存 KV、RoPE 旋转位置编码
    void rmsnorm_qkv_rope(int32_t layer_id, const tensor::Tensor& input, const tensor::Tensor& token_pos) const;

    // 执行多头注意力机制: 读取 KV Cache（上一轮对话的历史），计算注意力分数，并融合 V 向量，最后乘上 wo 输出矩阵
    void flash_decoding_gqa(int32_t layer_id, const tensor::Tensor& residual_add, const tensor::Tensor& token_pos) const;

    // 执行一次 Add，即残差连接 x = x + attention_output
    // 再次进行 RMSNorm，执行 SwiGLU 也就是 MLP 块: output = (SiLU(x * w1) @ (x * w3)) w2 ，这里对应代码中的 w1, w3, w2 层
    // 最后再执行一次 Add，即残差连接
    void feed_forward(int32_t layer_id, const tensor::Tensor& residual_add) const;

    // 分类头 (Classification): 最后一层 Transformer 跑完后，将向量映射到词表大小，输出叫 Logits（未归一化的概率分值）
    void cls_logits(const tensor::Tensor& input) const;

    // 后处理/采样 (Sampling): argmax 采样会选择 Logits 最大的 token id
    int32_t post_process(bool is_prompt) const override;

private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
    std::unique_ptr<Qwen3FusedLayers> qwen3_fused_layers_;
};
}  // namespace model

#endif