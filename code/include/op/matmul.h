#ifndef CODE_INCLUDE_OP_MATMUL_H
#define CODE_INCLUDE_OP_MATMUL_H

#include "op/layer.h"

namespace op {
// 输入: 向量 x (dim1)，权重矩阵 w (dim0, dim1)，偏置 b (dim0)
// 输出: 向量 y (dim0)
// 计算公式为: 矩阵-向量乘法 y = wx + b
class MatmulLayer : public LayerParam {
public:
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, bool fuse_add = false, bool is_quant_layer = false);
    
    base::Status check() const override;
    
    base::Status forward() override;

private:
    int32_t dim0_ = 0;      // 权重矩阵 w 的第 1 个维度
    int32_t dim1_ = 0;      // 权重矩阵 w 的第 2 个维度
    bool fuse_add_ = false; // 是否与 add 融合，即 GEMV + Add (dim0)
};
}  // namespace op

#endif  // CODE_INCLUDE_OP_MATMUL_H