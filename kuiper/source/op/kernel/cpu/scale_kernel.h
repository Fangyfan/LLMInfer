#ifndef SCALE_KERNEL_H
#define SCALE_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void scale_kernel_cpu(float scale, const tensor::Tensor& input);

}  // namespace kernel

#endif  // SCALE_KERNEL_H