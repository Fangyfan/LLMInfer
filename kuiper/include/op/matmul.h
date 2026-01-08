#ifndef KUIPER_INCLUDE_OP_MATMUL_H
#define KUIPER_INCLUDE_OP_MATMUL_H

#include "op/layer.h"

namespace op {
// 输入: 向量 x (dim1)，权重矩阵 w (dim0, dim1)，偏置 b (dim0)
// 输出: 向量 y (dim0)
// 计算公式为: 矩阵-向量乘法 y = wx + b
class MatmulLayer : public LayerParam {
public:
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, bool is_quant_layer = false, bool has_bias = false);
    
    base::Status check() const override;
    
    base::Status forward() override;

    // 把外部传入的偏置数据 bias_ptr，拷贝到当前层的 bias_[idx] 中，完成偏置的初始化 / 更新，返回执行状态
    base::Status set_bias(int32_t idx, int32_t dim, const void* bias_ptr, base::DeviceType device_type);

    tensor::Tensor& get_bias(int32_t idx);
    const tensor::Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;

private:
    int32_t dim0_ = 0;                  // 权重矩阵 w 的第 1 个维度
    int32_t dim1_ = 0;                  // 权重矩阵 w 的第 2 个维度
    bool has_bias_ = false;             // 当前层是否有偏置 bias
    std::vector<tensor::Tensor> bias_;  // 偏置张量 bias
};
}  // namespace op

#endif  // KUIPER_INCLUDE_OP_MATMUL_H