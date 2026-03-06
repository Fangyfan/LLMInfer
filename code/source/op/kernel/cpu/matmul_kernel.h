#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void matmul_kernel_cpu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    float scale, 
    void* stream
);
}  // namespace kernel

#endif  // MATMUL_KERNEL_H