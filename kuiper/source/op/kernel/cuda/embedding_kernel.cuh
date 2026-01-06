#ifndef EMBEDDING_KERNEL_CUH
#define EMBEDDING_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void embedding_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t vocab_size, 
    void* stream
);
}  // namespace kernel

#endif  // EMBEDDING_KERNEL_CUH