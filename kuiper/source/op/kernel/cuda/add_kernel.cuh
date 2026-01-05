#ifndef ADD_KERNEL_CUH
#define ADD_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void add_kernel_cu(
    const tensor::Tensor& input1, 
    const tensor::Tensor& input2, 
    const tensor::Tensor& output, 
    void* stream
);
}  // namespace kernel

#endif  // ADD_KERNEL_CUH