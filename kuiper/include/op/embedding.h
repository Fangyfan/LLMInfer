#ifndef KUIPER_INCLUDE_OP_EMBEDDING_H
#define KUIPER_INCLUDE_OP_EMBEDDING_H

#include "op/layer.h"

namespace op {
// 输入: 向量 in 表示 tokens_seq (seq_len)，in[i] 属于 [0, vocab_size)，词嵌入张量 weight (vocab_size, dim)
// 输出: 张量 out (seq_len, dim)
// 计算公式: out = Embedding(in, weight) = weight[in[i]]
class EmbeddingLayer : public LayerParam {
public:
    EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size);

    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t dim_ = 0;        // 词嵌入 (embedding) 的长度，其实就是 d_model
    int32_t seq_len_ = 0;    // prefill 阶段 prompt 的长度
    int32_t vocab_size_ = 0; // 词表大小 (token 在词表中最大索引)
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_EMBEDDING_H