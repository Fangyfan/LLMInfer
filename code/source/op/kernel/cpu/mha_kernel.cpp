#include "mha_kernel.h"
#include "../kernel_interface.h"
#include <cmath>

namespace kernel {
void mha_kernel_cpu(
    const tensor::Tensor& query, 
    const tensor::Tensor& score, 
    const tensor::Tensor& key_cache, 
    const tensor::Tensor& value_cache, 
    const tensor::Tensor& output, 
    int32_t layer_id, 
    int32_t pos, 
    int32_t kv_dim, 
    int32_t kv_mul, 
    int32_t head_num, 
    int32_t head_dim, 
    int32_t max_seq_len, 
    void* stream
) {
    UNUSED(stream);
    CHECK(!query.is_empty());
    CHECK(!score.is_empty());
    CHECK(!key_cache.is_empty());
    CHECK(!value_cache.is_empty());
    CHECK(!output.is_empty());
    CHECK(query.device_type() == base::DeviceType::DeviceCPU);
    CHECK(score.device_type() == base::DeviceType::DeviceCPU);
    CHECK(key_cache.device_type() == base::DeviceType::DeviceCPU);
    CHECK(value_cache.device_type() == base::DeviceType::DeviceCPU);
    CHECK(output.device_type() == base::DeviceType::DeviceCPU);

    // 计算当前 Layer 在 KV Cache 中的起始偏移量
    int32_t layer_offset = layer_id * max_seq_len * kv_dim;

    // 缩放因子: 1 / sqrt(head_dim)，防止点积结果过大导致 Softmax 梯度消失
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // 遍历每一个注意力头 (Head)，独立计算
    for (int32_t h = 0; h < head_num; ++h) {
        // 定位当前头的 Score 存放地址: Score 形状为 [head_num, max_seq_len] 分成了 head_num 个注意力头 [1, max_seq_len]
        // 用于存计算出的注意力分数
        float* score_head_ptr = const_cast<float*>(score.ptr<float>(h * max_seq_len));

        // 定位当前头的 Query 向量地址: Query 形状为 [head_num, head_dim] 分成了 head_num 个注意力头 [1, head_dim]
        float* query_head_ptr = const_cast<float*>(query.ptr<float>(h * head_dim));

        // 包装成 Tensor 对象，方便后续传给 Kernel 函数（这只是一个视图，不拷贝数据）
        tensor::Tensor query_head(base::DataType::DataTypeFp32, head_dim, false, nullptr, query_head_ptr);
        query_head.set_device_type(base::DeviceType::DeviceCPU);

        // 遍历每个历史 token 的位置 t = 0..pos，逐个计算 Query * Key[t] 注意力分数
        for (int32_t t = 0; t <= pos; ++t) {
            // --- 核心逻辑：KV Cache 寻址 ---
            // h / kv_mul: 这是 GQA / MQA 的精髓
            // 如果 kv_mul = 1 (标准 MHA)，第 h 个 Query 对应第 h 个 Key
            // 如果 kv_mul = 8 (MQA)，所有 h (0 ~ 7) 都对应第 0 个 Key (0/8 = 0)
            // t * kv_dim: 定位到时间步 t
            int32_t key_cache_offset = layer_offset + t * kv_dim + (h / kv_mul) * head_dim;

            // 找到过去第 t 个时刻的 Key 向量地址
            float* key_head_ptr = const_cast<float*>(key_cache.ptr<float>(key_cache_offset));

            // 包装 Key 向量
            tensor::Tensor key_head(base::DataType::DataTypeFp32, 1, head_dim, false, nullptr, key_head_ptr);
            key_head.set_device_type(base::DeviceType::DeviceCPU);

            // 包装 Score 存放地址（只存这一个时间步 t 的分数）
            tensor::Tensor score_head_t(base::DataType::DataTypeFp32, 1, false, nullptr, score_head_ptr + t);
            score_head_t.set_device_type(base::DeviceType::DeviceCPU);

            // --- 执行点积 ---
            // 计算一维向量点积：Score[t] = Query * Key[t] * scale
            // 其中 Query 和 Key[t] 均是长度为 head_dim 的一维向量，这里调用的是一个底层的矩阵乘法算子
            kernel::get_matmul_kernel(base::DeviceType::DeviceCPU)(query_head, key_head, score_head_t, scale, nullptr);
        }

        // 包装整个 score 数组（长度为 pos + 1）
        tensor::Tensor score_head(base::DataType::DataTypeFp32, pos + 1, false, nullptr, score_head_ptr);
        score_head.set_device_type(base::DeviceType::DeviceCPU);

        // 执行 Softmax：e^{x_i-max} / sum(e^{x_j-max})
        // 结果直接原地更新在 score_head_tensor 中
        get_softmax_kernel(base::DeviceType::DeviceCPU)(score_head, nullptr);

        // 定位当前头的 Output 向量地址: Output 形状为 [head_num, head_dim] 分成了 head_num 个注意力头 [1, head_dim]
        float* output_head_ptr = const_cast<float*>(output.ptr<float>(h * head_dim));

        // 先清零输出内存，因为后面是累加操作
        memset(output_head_ptr, 0, head_dim * sizeof(float));

        // 包装 Output Head Tensor
        tensor::Tensor output_head(base::DataType::DataTypeFp32, head_dim, false, nullptr, output_head_ptr);
        output_head.set_device_type(base::DeviceType::DeviceCPU);

        // --- 准备 Value Cache 的地址 ---
        // 注意：这里没有循环 t
        // 这里的 value_cache_offset 指向了当前 Head 对应的 Value Cache 的起始列（第 0 个时间步）
        // 那么第 t 个时间步的 offset 应该在此基础上加 t * kv_dim，即步长 stride = kv_dim
        // GQA 逻辑 (h / kv_mul) 依然适用，找到对应的 Value Head
        int32_t value_cache_offset = layer_offset + (h / kv_mul) * head_dim;
        float* value_head_ptr = const_cast<float*>(value_cache.ptr<float>(value_cache_offset));
        tensor::Tensor value_head(base::DataType::DataTypeFp32, head_dim, false, nullptr, value_head_ptr);
        value_head.set_device_type(base::DeviceType::DeviceCPU);

        // --- 执行加权求和 ---
        // 这个 kernel 内部会自己遍历 t = 0..pos
        // 计算：Output += Score[t] * Value[t]，其中 Score[t] 是 float 类型标量，Ouput 和 Value[t] 是长度为 head_dim 的向量
        get_scale_sum_kernel(base::DeviceType::DeviceCPU)(score_head, value_head, output_head, pos, head_dim, kv_dim);
    }
}
}  // namespace kernel