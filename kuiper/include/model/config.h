#ifndef KUIPER_INCLUDE_MODEL_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_CONFIG_H_

#include <cstdint>

namespace model {
struct ModelConfig {
    int32_t dim = 0;            // 模型的隐藏层维度，也就是 Embedding 长度，即 d_model
    int32_t hidden_dim = 0;     // 前馈层的中间维度，即 FFN-SwiGLU 隐藏层维度 = (2/3) * 4d
    int32_t layer_num = 0;      // 模型的 Transformer 总层数
    int32_t head_num = 0;       // 注意力机制的总头数 (Query 头数量)
    int32_t kv_head_num = 0;    // 分组注意力的 KV 头数
    int32_t vocab_size = 0;     // 词表大小
    int32_t max_seq_len = 0;    // 模型能处理的最长文本长度
};

struct TransformerConfig {
    int32_t layer_num = 0;      // 模型的 Transformer 总层数
    int32_t vocab_size = 0;     // 词表大小
    int32_t hidden_dim = 0;     // 前馈层的中间维度，即 FFN-SwiGLU 隐藏层维度 = (2/3) * 4d
    int32_t dim = 0;            // 模型的隐藏层维度，也就是 Embedding 长度，即 d_model
    int32_t kv_dim = 0;         // KV 的总维度
    int32_t kv_mul = 0;         // 每个 KV 头对应多少个 Query 头
    int32_t head_dim = 0;       // 每个注意力头的大小
    int32_t head_num = 0;       // 注意力机制的总头数 (Query 头数量)
    int32_t kv_head_num = 0;    // 分组注意力的 KV 头的数量
    int32_t max_seq_len = 0;    // 模型能处理的最长文本长度
    bool is_shared_weight = false;  // Embedding 层 (vocab_size, dim) 和 Output 层 (dim, vocab_size) 是否共享权重
};
}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_CONFIG_H_