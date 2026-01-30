#ifndef KUIPER_INCLUDE_MODEL_MODEL_H
#define KUIPER_INCLUDE_MODEL_MODEL_H

#include "config.h"
#include "op/encode.h"
#include "op/embedding.h"
#include "sampler/argmax_sampler.h"
#include "raw_model_data.h"

namespace model {
class Model { // 所有大模型的统一抽象基类
public:
    explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string tokenizer_path, std::string model_path, bool is_quant_model);

    // 初始化模型，决定是在 CPU 还是 CUDA 上运行，并分配内存
    virtual base::Status init(base::DeviceType device_type) = 0;

    // 最高层级的推理入口: 输入一个 token 的 embedding + position，是否属于 prompt 阶段，输出生成的下一个 token
    virtual base::Status predict(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos, bool is_prompt, int32_t& next_token_id) const = 0;

    // 前向传播主函数: 它描述了整个 LLM 的计算图，它会循环调用每一层 Transformer Block
    virtual base::Status forward(const tensor::Tensor& token_embedding, const tensor::Tensor& token_pos) const = 0;

    // 简单的查表操作，获取词向量
    virtual op::EmbeddingResult embedding(const std::vector<int32_t>& token_ids) const = 0;

    base::ModelType model_type() const;
    const base::TokenizerType tokenizer_type() const;
    const std::string& tokenizer_path() const;
    const std::string& model_path() const;
    virtual tensor::Tensor& get_buffer(ModelBufferType model_buffer_type);
    virtual const tensor::Tensor& get_buffer(ModelBufferType model_buffer_type) const;

    virtual bool is_sentence_end(int32_t token_id) const;
    virtual std::string decode(int32_t token_id) const;
    virtual std::string decode(const std::vector<int32_t>& token_ids) const;
    virtual std::vector<int32_t> encode(const std::string& sentence) const;

    virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_id, int32_t token_pos) const;
    virtual tensor::Tensor get_embedding(const tensor::Tensor& token_pos, const op::EmbeddingResult& embedding_output, bool is_prompt) const;

protected:
    virtual base::Status insert_buffer(ModelBufferType model_buffer_type, const tensor::Tensor& model_buffer);
    virtual base::Status read_model_file();
    virtual base::Status create_encode_layer();
    virtual base::Status create_model();
    virtual base::Status generate_model_info(const ModelConfig& config) const;
    virtual int32_t post_process(bool is_prompt) const = 0;

private:
    virtual void allocate_model_buffers() = 0;
    // 这些函数负责在程序启动时读取文件并创建对象
    virtual base::Status create_layers() = 0; // 总入口
    virtual void create_param_layers() = 0; // 创建带权重的层（Linear 层等）
    virtual void create_nonparam_layers() = 0; // 创建算子层（RoPE 层等）
    virtual void create_param_quant_layers() = 0; // 如果是 Int8 量化模型，则调用此函数加载量化后的权重

protected:
    int32_t group_size_ = 0; // 量化分组大小
    bool is_quant_model_ = false; // 是否为量化模型
    std::unique_ptr<TransformerConfig> config_; // 模型配置

    std::string tokenizer_path_; // tokenizer 文件路径
    std::string model_path_; // 模型权重文件路径

    std::map<ModelBufferType, tensor::Tensor> buffers_; // 存储推理过程中各种数据 (如 KV Cache)
    std::unique_ptr<op::BaseEncodeLayer> encode_layer_; // encode 分词层 (sentence <=> token ids)
    std::unique_ptr<sampler::ArgmaxSampler> sampler_; // argmax 采样器
    std::unique_ptr<RawModelData> raw_model_data_; // 模型权重内存管理结构

    base::DeviceType device_type_ = base::DeviceType::DeviceUnknown; // 设备类型 (CPU / CUDA)
    base::ModelType model_type_ = base::ModelType::ModelTypeUnknown; // 模型类型（Llama / Qwen)
    base::TokenizerType tokenizer_type_ = base::TokenizerType::EncodeUnknown; // tokenizer 类型 (BPE / SentencePiece)
};
}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_MODEL_H