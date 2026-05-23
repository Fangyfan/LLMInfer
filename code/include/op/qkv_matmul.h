#ifndef CODE_INCLUDE_OP_QKV_MATMUL_H
#define CODE_INCLUDE_OP_QKV_MATMUL_H

#include "op/layer.h"

namespace op {
// 输入: 向量 x (dim1)，权重矩阵 w (dim0, dim1) 其中 dim0 = (dim + kv_dim + kv_dim)
// 输出: 向量 y (dim0)
// 计算公式为: 矩阵-向量乘法 y = wx = row_concat(w_q * x, w_k * x, w_v * x)
class QKVMatmulLayer : public LayerParam {
public:
    explicit QKVMatmulLayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t dim0, int32_t dim1);
    
    base::Status check() const override;
    
    base::Status forward() override;

private:
    int32_t dim_;       // 向量 Q 的维度: head_num * head_dim
    int32_t kv_dim_;    // 向量 KV 的维度: kv_head_num * head_dim
    int32_t dim0_ = 0;  // 权重矩阵 w 的第 1 个维度
    int32_t dim1_ = 0;  // 权重矩阵 w 的第 2 个维度
};
}  // namespace op

#endif  // CODE_INCLUDE_OP_QKV_MATMUL_H