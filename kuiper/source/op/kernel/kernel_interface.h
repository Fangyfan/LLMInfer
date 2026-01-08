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
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
);
MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

using SwigluKernel = void (*)(
    const tensor::Tensor& input1, 
    const tensor::Tensor& input2, 
    const tensor::Tensor& output, 
    void* stream
);
SwigluKernel get_swiglu_kernel(base::DeviceType device_type);

using EmbeddingKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t vocab_size, 
    void* stream
);
EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);

using RoPEKernel = void (*)(
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_size, 
    const tensor::Tensor& input_q, 
    const tensor::Tensor& input_k, 
    const tensor::Tensor& input_pos, 
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    void* stream
);
RoPEKernel get_rope_kernel(base::DeviceType device_type);

using MHAKernel = void (*)(
    int32_t layer_index, 
    int32_t pos, 
    int32_t kv_mul, 
    int32_t kv_dim, 
    int32_t seq_len, 
    int32_t head_num, 
    int32_t head_size, 
    const tensor::Tensor& query, 
    const tensor::Tensor& score, 
    const tensor::Tensor& key_cache, 
    const tensor::Tensor& value_cache, 
    const tensor::Tensor& mha_out, 
    void* stream
);
MHAKernel get_mha_kernel(base::DeviceType device_type);

}  // namespace kernel

#endif  // KERNEL_INTERFACE_H