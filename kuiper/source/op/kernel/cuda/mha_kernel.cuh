#ifndef MHA_KERNEL_CUH
#define MHA_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void mha_kernel_cu(
    int32_t layer_id, 
    int32_t pos, 
    int32_t kv_dim, 
    int32_t kv_mul, 
    int32_t head_num, 
    int32_t head_size, 
    int32_t max_seq_len, 
    const tensor::Tensor& query, 
    const tensor::Tensor& score, 
    const tensor::Tensor& key_cache, 
    const tensor::Tensor& value_cache, 
    const tensor::Tensor& output, 
    void* stream
);
}  // namespace kernel

#endif  // MHA_KERNEL_CUH