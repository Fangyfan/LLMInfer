#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input, void* stream);

}  // namespace kernel

#endif  // SOFTMAX_KERNEL_H