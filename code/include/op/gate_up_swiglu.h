#ifndef CODE_INCLUDE_OP_GATE_UP_SWIGLU_H
#define CODE_INCLUDE_OP_GATE_UP_SWIGLU_H

#include "op/layer.h"

namespace op {
// 输入: 向量 x (dim1)，权重矩阵 w (dim0, dim1) 其中 dim0 = (immediate_dim + immediate_dim), dim1 = hidden_dim
// 输出: 向量 y (dim0)
// 计算公式为: 矩阵-向量乘法 y = wx = row_concat(w_gate * x, w_up * x)
class GateUpSwigluLayer : public LayerParam {
public:
    explicit GateUpSwigluLayer(base::DeviceType device_type, int32_t immediate_dim, int32_t hidden_dim, bool is_quant_layer = false);
    
    base::Status check() const override;
    
    base::Status forward() override;

private:
    int32_t immediate_dim_; // 向量 x_gate = w_gate * x 和 x_up = w_up * x 的维度
    int32_t hidden_dim_;    // 向量 x 的维度
};
}  // namespace op

#endif  // CODE_INCLUDE_OP_GATE_UP_SWIGLU_H