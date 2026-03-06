#include "scale_sum_kernel.h"
#include <armadillo>

namespace kernel {
// 对于每个注意力头独立计算: score_head[0..pos] × value_head[0..pos, head_dim] = mha_head_out[head_dim]
void scale_sum_kernel_cpu(const tensor::Tensor& score, const tensor::Tensor& value, const tensor::Tensor& output, 
                          int32_t pos, int32_t head_dim, int32_t kv_dim) {
    CHECK(!score.is_empty());
    CHECK(!value.is_empty());
    CHECK(!output.is_empty());
    CHECK(score.device_type() == base::DeviceType::DeviceCPU);
    CHECK(value.device_type() == base::DeviceType::DeviceCPU);
    CHECK(output.device_type() == base::DeviceType::DeviceCPU);
    CHECK_EQ(score.size(), pos + 1);
    CHECK_EQ(value.size(), head_dim);
    CHECK_EQ(output.size(), head_dim);

    // arma::fvec 表示 单精度浮点 一维向量（float vector），封装了连续的浮点数组，支持高效的逐元素运算
    // 不拷贝数据，直接把张量的内存映射成 Armadillo 向量，避免内存拷贝的性能损耗
    // arma::fvec(
    //     float* mem_ptr,        // 参数1：内存指针 → CPU中存放数据的首地址 (指针需要去掉只读 const)
    //     const int n_elem,      // 参数2：向量长度 → 数据的总元素个数
    //     const bool copy_mem,   // 参数3：是否拷贝内存 → false = 不拷贝，共享内存，避免拷贝开销
    //     const bool strict_mem  // 参数4：是否严格绑定内存 → true = 严格绑定，不扩容/缩容，向量的长度永远是dim，不会越界访问内存
    // );
    arma::fvec scale(const_cast<float*>(score.ptr<float>()), score.size(), false, true);
    arma::fvec out(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

    for (int32_t t = 0; t <= pos; ++t) {
        // 取出当前注意力头中 value 第 t 个时间步长度为 head_dim 的向量 val
        arma::fvec val(const_cast<float*>(value.ptr<float>(t * kv_dim)), value.size(), false, true);
        // 对于当前时间步 t 的向量 val[t] 中所有元素，乘以系数 scale[t]
        // 将所有时间步 t 的向量 val[t] 加起来，就是 output 向量，长度均为 head_dim
        out += scale[t] * val;
    }
}
}  // namespace kernel