#ifndef KUIPER_INCLUDE_SAMPLER_H
#define KUIPER_INCLUDE_SAMPLER_H

#include "base/base.h"

namespace sampler {
class Sampler {
public:
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

    // argmax 贪心采样: 模型输出的概率分布 logits，其长度为 size，输出概率最高的那个 token 的索引
    // 输入 logits = [-0.2, 2.3, 0.5, 1.8, -1.0]，输出为索引 1
    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) const = 0;

protected:
    base::DeviceType device_type_ = base::DeviceType::DeviceUnknown;
};
}  // namespace sampler

#endif  // KUIPER_INCLUDE_SAMPLER_H
