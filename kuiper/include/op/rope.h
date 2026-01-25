#ifndef KUIPER_INCLUDE_OP_ROPE_H
#define KUIPER_INCLUDE_OP_ROPE_H

#include "op/layer.h"

namespace op {
// 输入: 向量 q (dim) 、向量 k (kv_dim) 、当前位置 pos 、sin_cache (token_num * head_dim) 、 cos_cache (token_num * head_dim)
// 输出: 向量 RoPE(q) (dim) 和向量 RoPE(k) (kv_dim)
// 计算公式: RoPE(x): for k = 0, 1, 2, ... head_dim/2 - 1 :
// x[2k]   = cos(pos × theta[pos,2k]) * x[2k] - sin(pos × theta[pos,2k]) * x[2k+1]
// x[2k+1] = sin(pos × theta[pos,2k]) * x[2k] + cos(pos × theta[pos,2k]) * x[2k+1]
// theta[pos,2k] = 1 / base^{2k / head_dim}
class RoPELayer : public Layer {
public:
    explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_dim);
    
    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t dim_ = 0;       // Q 向量总长度，即 embedding 的长度 d_model
    int32_t kv_dim_ = 0;    // KV 向量总长度，在 GQA 中 dim_ % kv_dim_ = 0 (dim_ > kv_dim_)
    int32_t head_dim_ = 0;  // 每个注意力头 (QKV head) 大小 (head_num = dim_ / head_dim_, kv_head_num = kv_dim_ / head_dim_)
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_ROPE_H
