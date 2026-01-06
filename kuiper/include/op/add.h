#ifndef KUIPER_INCLUDE_OP_ADD_H
#define KUIPER_INCLUDE_OP_ADD_H

#include "op/layer.h"

namespace op {
// 输入: 输入向量 x (dim) 和 y (dim)
// 输出: 结果向量 z (dim)
// 计算公式: 向量逐位相加 z = x + y
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;
    
    base::Status forward() override;
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_ADD_H