#include "matmul_kernel.cuh"

namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, void* stream) {
    
}

void matmul_kernel_cu_quant8(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, const tensor::Tensor& sacles, int32_t group_size, void* stream) {
    
}
}  // namespace kernel