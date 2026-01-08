#ifndef KUIPER_INLCUDE_MHA_H
#define KUIPER_INLCUDE_MHA_H

#include "op/layer.h"

namespace op {
class MultiHeadAttention : public Layer {
public:
    explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index, int32_t pos, int32_t kv_mul, int32_t kv_dim, 
                                int32_t seq_len, int32_t head_num, int32_t head_size);
    
    base::Status check() const override;

    base::Status forward() override;

    void set_pos(int32_t pos);
    void set_layer_index(int32_t layer_index);

private:
    int32_t layer_index_ = 0;   // 当前层在模型中的索引
    int32_t pos_ = 0;           // 当前位置（在序列中的位置）
    int32_t kv_mul_ = 0;        // KV 头的倍数（通常用于分组查询注意力）
    int32_t kv_dim_ = 0;        // KV 的维度
    int32_t seq_len_ = 0;       // 序列长度
    int32_t head_num_ = 0;      // 注意力头 (Q) 的数量
    int32_t head_size_ = 0;     // 每个注意力头 (QKV) 的大小
};
}  // namespace op

#endif  // KUIPER_INLCUDE_MHA_H