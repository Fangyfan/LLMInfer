#ifndef ROPE_KERNEL_H
#define ROPE_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
void sin_cos_cache_precompute_cpu(
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t head_dim, 
    int32_t max_seq_len, 
    void* stream
);

void rope_kernel_cpu(
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
}  // namespace kernel

#endif  // ROPE_KERNEL_H