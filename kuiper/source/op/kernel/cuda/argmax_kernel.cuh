#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH

namespace kernel {
int32_t argmax_kernel_cu(const float* input_ptr, int32_t size, void* stream);

}  // namespace kernel

#endif  // ARGMAX_KERNEL_CUH
