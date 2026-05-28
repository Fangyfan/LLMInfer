#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include <cstdint>

namespace kernel {
int32_t argmax_kernel_cu(
    const float* input, 
    int32_t size, /* 151936 */
    int32_t* argmax_token, 
    void* argmax_buffer, 
    void* stream
);

}  // namespace kernel

#endif  // ARGMAX_KERNEL_CUH
