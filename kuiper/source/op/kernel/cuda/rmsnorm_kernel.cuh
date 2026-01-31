#ifndef RMSNORM_KERNEL_CUH
#define RMSNORM_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void rmsnorm_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);

void rmsnorm_2d_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t dim, 
    void* stream
);
}  // namespace kernel

#endif  // RMSNORM_KERNEL_CUH