#ifndef KUIPER_INCLUDE_OP_ENCODE_H
#define KUIPER_INCLUDE_OP_ENCODE_H

#include <sentencepiece_processor.h>
#include "op/layer.h"

namespace op {
class BaseEncodeLayer : public Layer { // 接口层（抽象基类），解耦 tokenizer 的具体实现，允许未来支持 SentencePiece / SPE
public:
    explicit BaseEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos);

    // 文本(sentence) -> token ids
    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

    // token id -> 子词(sub word)
    virtual std::string decode(int32_t token_id) const = 0;
    // token ids -> 文本(sentence)
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

    // 判断是否为 EOS（推理停止条件）
    virtual bool is_sentence_end(int32_t token_id) const = 0;

    // 词表大小（Embedding 维度需要）
    virtual int32_t vocab_size() const = 0;
    
protected:
    // 这些是所有 tokenizer 都共用的配置
    std::string tokenizer_path_;  // SentencePiece 模型文件路径 (.model)
    bool has_bos_ = false;  // 是否在开头插入 <bos>
    bool has_eos_ = false;  // 是否在结尾插入 <eos>
};

class SpeEncodeLayer : public BaseEncodeLayer {
public:
    explicit SpeEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos); // 构造函数：加载 tokenizer 模型
    virtual std::vector<int32_t> encode(const std::string& sentence) const override;
    virtual std::string decode(int32_t token_id) const override;
    virtual std::string decode(const std::vector<int32_t>& token_ids) const override;
    virtual bool is_sentence_end(int32_t token_id) const override;
    virtual int32_t vocab_size() const override;
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe_; // SentencePiece 官方处理器 + RAII 管理资源
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_ENCODE_H