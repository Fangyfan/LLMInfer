#ifndef RMSNORM_KERNEL_CUH
#define RMSNORM_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void rmsnorm_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);

void fused_add_rmsnorm_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& residual_add, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
);

void fused_qk_norm_rope_kernel_cu(
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

void rmsnorm_2d_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t dim, 
    void* stream
);
}  // namespace kernel

#endif  // RMSNORM_KERNEL_CUH