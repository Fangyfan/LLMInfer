#ifndef LLAMA_INFER_INCLUDE_OP_SWIGLU_H
#define LLAMA_INFER_INCLUDE_OP_SWIGLU_H

#include "op/layer.h"

namespace op {
// 输入: 向量 in1 = w1 * x (hidden_dim) 和向量 in2 = w2 * x (hidden_dim)
// 输出: 向量 out (hidden_dim)
// 计算公式: 其中 @ 表示逐位相乘
// SwiGLU(x, w1, w2) = SiLU(w1 * x) @ (w2 * x) = [(w1 * x) * Sigmoid(w1 * x)] @ (w2 * x)
// SwiGLU(in1, in2) = SiLU(in1) @ in2 = (in1 * Sigmoid(in1)) @ in2
class SwiGLULayer : public Layer {
public:
    explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);
    
    base::Status check() const override;

    base::Status forward() override;
private:
    int32_t hidden_dim_ = 0; // 隐藏层大小: 当输入张量大小为 d 时，隐藏层大小为 (2/3) * 4d
};
}  // namespace op

#endif  // LLAMA_INFER_INCLUDE_OP_SWIGLU_H