#ifndef KUIPER_INCLUDE_OP_ENCODE_H
#define KUIPER_INCLUDE_OP_ENCODE_H

#include <sentencepiece_processor.h>
#include "op/layer.h"
#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "base/tiktoken.h"
#include "base/unordered_dense.h"
#include "nlohmann/json.hpp"
#endif

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
    std::vector<int32_t> encode(const std::string& sentence) const override;
    std::string decode(int32_t token_id) const override;
    std::string decode(const std::vector<int32_t>& token_ids) const override;
    bool is_sentence_end(int32_t token_id) const override;
    int32_t vocab_size() const override;
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe_; // SentencePiece 官方处理器 + RAII 管理资源
};

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
class BpeEncodeLayer : public BaseEncodeLayer {
public:
    explicit BpeEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos);
    std::vector<int32_t> encode(const std::string& sentence) const override;
    std::string decode(int32_t token_id) const override;
    std::string decode(const std::vector<int32_t>& token_ids) const override;
    bool is_sentence_end(int32_t token_id) const override;
    int32_t vocab_size() const override;

protected:
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t stop_token1_ = -1;
    int32_t stop_token2_ = -1;
    int32_t num_token_ = 0;
    std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

class QwenEncodeLayer : public BpeEncodeLayer {
public:
    explicit QwenEncodeLayer(std::string tokenizer_path, bool has_bos, bool has_eos);
};
#endif

}  // namespace op

#endif  // KUIPER_INCLUDE_OP_ENCODE_H