#include <algorithm>
#include "sampler/argmax_sampler.h"
#include "../op/kernel/cuda/argmax_kernel.cuh"

namespace sampler {
ArgmaxSampler::ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

int32_t ArgmaxSampler::sample(const float* logits, int32_t size, void* stream) const {
    CHECK_NE(logits, nullptr);
    CHECK_GT(size, 0);
    int32_t next_token_id = 0;
    if (device_type_ == base::DeviceType::DeviceCPU) {
        next_token_id = std::distance(logits, std::max_element(logits, logits + size));
    } else if (device_type_ == base::DeviceType::DeviceCUDA) {
        next_token_id = kernel::argmax_kernel_cu(logits, size, stream);
    } else {
        LOG(FATAL) << "Unknown device type for argmax sampler." << std::endl;
    }
    return next_token_id;
}
}  // namespace sampler
