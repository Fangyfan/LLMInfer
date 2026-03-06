#ifndef SCALE_SUM_KERNEL_H
#define SCALE_SUM_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void scale_sum_kernel_cpu(
    const tensor::Tensor& score, 
    const tensor::Tensor& value, 
    const tensor::Tensor& output, 
    int32_t pos, 
    int32_t head_dim, 
    int32_t kv_dim
);
}  // namespace kernel

#endif  // SCALE_SUM_KERNEL_H