#ifndef MHA_KERNEL_CUH
#define MHA_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void mha_kernel_cu(
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
}  // namespace kernel

#endif  // MHA_KERNEL_CUH