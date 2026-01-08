#include "rope_kernel.cuh"

namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                     const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream) {
    
}
}  // namespace kernel