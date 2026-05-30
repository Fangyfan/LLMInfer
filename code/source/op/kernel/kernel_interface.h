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

using FusedAddRMSNormKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& residual_add,  
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);
FusedAddRMSNormKernel get_fused_add_rmsnorm_kernel(base::DeviceType device_type);

using RMSNorm2DKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t dim, 
    void* stream
);
RMSNorm2DKernel get_rmsnorm_2d_kernel(base::DeviceType device_type);

using GEMVKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    bool lm_head, 
    void* stream
);
GEMVKernel get_gemv_kernel(base::DeviceType device_type);

using FusedGEMVAddKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);
FusedGEMVAddKernel get_fused_gemv_add_kernel(base::DeviceType device_type);

using FusedGEMVAddInt4Kernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
);
FusedGEMVAddInt4Kernel get_fused_gemv_add_int4_kernel(base::DeviceType device_type);

using FusedQKVGEMVKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& value, 
    void* stream
);
FusedQKVGEMVKernel get_fused_qkv_gemv_kernel(base::DeviceType device_type);

using FusedQKVGEMVInt4Kernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& value, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
);
FusedQKVGEMVInt4Kernel get_fused_qkv_gemv_int4_kernel(base::DeviceType device_type);

using FusedGateUpSwiGLUKernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t immediate_dim, 
    void* stream
);
FusedGateUpSwiGLUKernel get_fused_gate_up_gemv_swiglu_kernel(base::DeviceType device_type);

using FusedGateUpSwiGLUInt4Kernel = void (*)(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    int32_t immediate_dim, 
    void* stream
);
FusedGateUpSwiGLUInt4Kernel get_fused_gate_up_gemv_swiglu_int4_kernel(base::DeviceType device_type);

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

using FusedQKNormRoPEKernel = void (*)(
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& token_pos, 
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_dim, 
    void* stream
);
FusedQKNormRoPEKernel get_fused_qk_norm_rope_kernel(base::DeviceType device_type);

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