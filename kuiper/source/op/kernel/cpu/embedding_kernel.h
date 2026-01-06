#ifndef EMBEDDING_KERNEL_H
#define EMBEDDING_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void embedding_kernel_cpu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t vocab_size, 
    void* stream
);
}  // namespace kernel

#endif  // EMBEDDING_KERNEL_H