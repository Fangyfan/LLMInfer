#include "rope_kernel.h"
#include <cmath>

namespace kernel {
void sin_cos_cache_precompute_cpu(const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, 
                                  int32_t head_dim, int32_t max_seq_len, void* stream) {
    int32_t half = head_dim / 2;
    std::vector<float> freq(half);
    const float inv_head_dim = 1.0f / static_cast<float>(head_dim);
    const float ln_base = std::log(1000000.0f);
    for (int32_t i = 0; i < half; ++i) {
        freq[i] = std::exp(-2.0 * i * inv_head_dim * ln_base);
        // freq[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }
    for (int32_t pos = 0; pos < max_seq_len; ++pos) {
        float* sin_cache_ptr = const_cast<float*>(sin_cache.ptr<float>(pos * head_dim / 2));
        float* cos_cache_ptr = const_cast<float*>(cos_cache.ptr<float>(pos * head_dim / 2));
        for (int32_t i = 0; i < head_dim / 2; ++i) {
            float theta = static_cast<float>(pos) * freq[i];
            sincosf(theta, sin_cache_ptr + i, cos_cache_ptr + i);
        }
    }
}

void rope_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& key, const tensor::Tensor& token_pos, 
const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, int32_t dim, int32_t kv_dim, int32_t head_dim, void* stream) {
    UNUSED(stream);
    CHECK(query.device_type() == base::DeviceType::DeviceCPU);
    CHECK(key.device_type() == base::DeviceType::DeviceCPU);
    CHECK(token_pos.device_type() == base::DeviceType::DeviceCPU);
    CHECK(sin_cache.device_type() == base::DeviceType::DeviceCPU);
    CHECK(cos_cache.device_type() == base::DeviceType::DeviceCPU);

    const int32_t pos = token_pos.index<int32_t>(0);
    float* q = const_cast<float*>(query.ptr<float>());
    float* k = const_cast<float*>(key.ptr<float>());
    float* sptr = const_cast<float*>(sin_cache.ptr<float>(pos * head_dim / 2)); // sptr 索引到第 pos 行
    float* cptr = const_cast<float*>(cos_cache.ptr<float>(pos * head_dim / 2)); // cptr 索引到第 pos 行

    int32_t half = head_dim / 2;
    // 遍历每个 Head 的首地址偏移量 head_start
    for (int32_t head_start = 0; head_start < dim; head_start += head_dim) {
        // 遍历 i = 0, 1, 2, ... , head_dim/2，让 (i, i + head_dim/2) 配对，进行二维旋转变换
        // 这里的 i 正好就是每个注意力头内部的 pair(2k, 2k+1) 中的 k，表示 sin/cos pair 的索引
        for (int32_t i = 0; i < half; ++i) {
            float sin_theta = sptr[i];
            float cos_theta = cptr[i];
            // 当 head_start < kv_dim 时需要旋转 q 和 k，否则只需要旋转 q
            int32_t rotn = head_start < kv_dim ? 2 : 1;
            for (int32_t r = 0; r < rotn; ++r) {
                float* v = const_cast<float*>(r == 0 ? q : k) + head_start + i;
                float v0 = v[0];
                float v1 = v[half];
                v[0] = v0 * cos_theta - v1 * sin_theta;
                v[half] = v0 * sin_theta + v1 * cos_theta;
            }
        }
    }
}
}  // namespace kernel