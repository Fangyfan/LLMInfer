#ifndef KUIPER_INCLUDE_OP_EMBEDDING_H
#define KUIPER_INCLUDE_OP_EMBEDDING_H

#include "op/layer.h"

namespace op {
class EmbeddingLayer : public LayerParam {
public:
    EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size);

    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t dim_ = 0; // 词嵌入 (token embedding) 的长度
    int32_t seq_len_ = 0; // 序列总长度 (token 的实际数量)
    int32_t vocab_size_ = 0; // 词表大小 (token 的最大数量)
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_EMBEDDING_H