#ifndef KUIPER_INCLUDE_ARGMAX_SAMPLER_H
#define KUIPER_INCLUDE_ARGMAX_SAMPLER_H

#include "sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
public:
    explicit ArgmaxSampler(base::DeviceType device_type);

    // argmax 贪心采样: 模型输出的概率分布 logits，其长度为 size，输出概率最高的那个 token 的索引
    // 输入 logits = [-0.2, 2.3, 0.5, 1.8, -1.0]，输出为索引 1
    // logits: 模型输出的原始分数数组（未经过 softmax）
    // size: 数组大小，即词汇表大小
    int32_t sample(const float* logits, int32_t size, void* stream) const override;
};
}  // namespace sampler

#endif  // KUIPER_INCLUDE_ARGMAX_SAMPLER_H