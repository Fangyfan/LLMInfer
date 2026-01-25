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
    float scale, 
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
    void* stream
);
EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);

using SinCosCacheKernel = void (*)(
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t head_dim, 
    int32_t max_seq_len, 
    void* stream
);
SinCosCacheKernel get_sin_cos_cache_kernel(base::DeviceType device_type);

using RoPEKernel = void (*)(
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& token_pos, 
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_dim, 
    void* stream
);
RoPEKernel get_rope_kernel(base::DeviceType device_type);

using MHAKernel = void (*)(
    const tensor::Tensor& query, 
    const tensor::Tensor& score, 
    const tensor::Tensor& key_cache, 
    const tensor::Tensor& value_cache, 
    const tensor::Tensor& output, 
    int32_t layer_id, 
    int32_t pos, 
    int32_t kv_dim, 
    int32_t kv_mul, 
    int32_t head_num, 
    int32_t head_dim, 
    int32_t max_seq_len, 
    void* stream
);
MHAKernel get_mha_kernel(base::DeviceType device_type);

using SoftmaxKernel = void (*)(
    const tensor::Tensor& input, 
    void* stream
);
SoftmaxKernel get_softmax_kernel(base::DeviceType device_type);

using ScaleKernel = void (*)(const tensor::Tensor& input, float scale);
ScaleKernel get_scale_kernel(base::DeviceType device_type);

using ScaleSumKernel = void (*)(
    const tensor::Tensor& score, 
    const tensor::Tensor& value, 
    const tensor::Tensor& output, 
    int32_t pos, 
    int32_t head_dim, 
    int32_t kv_dim
);
ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

}  // namespace kernel

#endif  // KERNEL_INTERFACE_H