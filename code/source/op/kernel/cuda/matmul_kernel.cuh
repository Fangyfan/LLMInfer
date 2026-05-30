#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void gemv_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    bool lm_head, 
    void* stream
);

void fused_gemv_add_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);

void fused_qkv_gemv_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& value, 
    void* stream
);

void fused_gate_up_gemv_swiglu_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t immediate_dim, 
    void* stream
);

void fused_gemv_add_int4_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
);

void fused_qkv_gemv_int4_kernel_cu(
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

void fused_gate_up_gemv_swiglu_int4_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& zeros, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    int32_t immediate_dim, 
    void* stream
);

void gemv_int8_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    const tensor::Tensor& scales, 
    int32_t group_size, 
    void* stream
);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CUH