#ifndef KUIPER_INCLUDE_OP_RMSNORM_H
#define KUIPER_INCLUDE_OP_RMSNORM_H

#include "op/layer.h"

namespace op {
// 输入: 向量 in (dim)，可学习的缩放权重向量 wei (dim)
// 输出: 向量 out (dim)
// 计算公式: out = RMSNorm(in) = wei * in / RMS(in)
// 其中 RMS(in) = 1 / sqrt(mean(in^2) + eps) = 1 / sqrt(sum(in^2) / dim + eps)
class RMSNormLaryer : public LayerParam {
public:
    explicit RMSNormLaryer(base::DeviceType device_type, int32_t dim);

    base::Status check() const override;

    base::Status forward() override;
private:
    int32_t dim_ = 0;
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_RMSNORM_H