#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void gemv_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    float scale, 
    void* stream
);

void fused_gemv_add_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);

void gemv_int8_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CUH