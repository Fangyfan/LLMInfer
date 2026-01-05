#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void matmul_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);

void matmul_kernel_cu_quant8(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& sacles, 
    int32_t group_size, 
    void* stream
);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CUH