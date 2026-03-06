#ifndef SOFTMAX_KERNEL_CUH
#define SOFTMAX_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void softmax_kernel_cu(
    const tensor::Tensor& input, 
    void* stream
);
}  // namespace kernel

#endif  // SOFTMAX_KERNEL_CUH