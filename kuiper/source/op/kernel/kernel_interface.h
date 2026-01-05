#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include "tensor/tensor.h"

namespace kernel {
// 函数指针类型的别名定义
// 用 typedef / using 定义了一个名为 XXX Kernel (比如 AddKernel) 的函数指针类型
// typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output, void* stream);
using AddKernel = void (*)(
    const tensor::Tensor& input1, 
    const tensor::Tensor& input2, 
    const tensor::Tensor& output, 
    void* stream
);
AddKernel get_add_kernel(base::DeviceType device_type);

using RMSNormKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

using MatmulKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);
MatmulKernel get_matmul_kernel(base::DeviceType device_type);

using MatmulKernelQuant = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& sacles, 
    int32_t group_size, 
    void* stream
);
MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

}  // namespace kernel

#endif  // KERNEL_INTERFACE_H