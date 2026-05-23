#ifndef CODE_INCLUDE_OP_QK_NORM_ROPE_H
#define CODE_INCLUDE_OP_QK_NORM_ROPE_H

#include "op/layer.h"

namespace op {
class QKNormRoPELaryer : public LayerParam {
public:
    explicit QKNormRoPELaryer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_dim);

    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t dim_ = 0;       // 向量 Q 总长度，即 embedding 的长度 d_model
    int32_t kv_dim_ = 0;    // 向量 KV 总长度，在 GQA 中 dim_ % kv_dim_ = 0 (dim_ > kv_dim_)
    int32_t head_dim_ = 0;  // 每个注意力头 (QKV head) 大小 (head_num = dim_ / head_dim_, kv_head_num = kv_dim_ / head_dim_)
};
}  // namespace op

#endif  // CODE_INCLUDE_OP_QK_NORM_ROPE_H