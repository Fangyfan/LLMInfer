#include <armadillo>
#include "rmsnorm_kernel.h"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::DeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::DeviceCPU);
    CHECK(output.device_type() == base::DeviceType::DeviceCPU);

    const float* input_ptr = input.ptr<float>();
    const float* weight_ptr = weight.ptr<float>();
    const float* output_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(input.size());

    // arma::fvec 表示 单精度浮点 一维向量（float vector），封装了连续的浮点数组，支持高效的逐元素运算
    // 不拷贝数据，直接把张量的内存映射成 Armadillo 向量，避免内存拷贝的性能损耗
    // arma::fvec(
    //     float* mem_ptr,        // 参数1：内存指针 → CPU中存放数据的首地址 (指针需要去掉只读 const)
    //     const int n_elem,      // 参数2：向量长度 → 数据的总元素个数
    //     const bool copy_mem,   // 参数3：是否拷贝内存 → false = 不拷贝，共享内存，避免拷贝开销
    //     const bool strict_mem  // 参数4：是否严格绑定内存 → true = 严格绑定，不扩容/缩容，向量的长度永远是dim，不会越界访问内存
    // );
    arma::fvec in(const_cast<float*>(input_ptr), dim, false, true);
    arma::fvec wei(const_cast<float*>(weight_ptr), dim, false, true);
    arma::fvec out(const_cast<float*>(output_ptr), dim, false, true);

    // RMSNorm 计算公式: output[i] = input[i] * weight[i] / sqrt(sum_{j=0}^{dim-1} input[j]^2 / dim + eps)

    // 步骤1: arma::pow(input, 2) -> 对向量中的每一个元素做平方运算(输出为新的 arma::fvec 向量) -> in[j]^2
    // 步骤2: arma::mean(...) -> 对向量的所有元素求算术平均值(输出为 arma::fscalar 单精度浮点标量) -> (Σin[j]^2) / dim
    // 步骤3: arma::as_scalar(...) -> 把 arma::fscalar 类型转换成 C++ 原生的 float 类型
    // 步骤4: +eps -> 极小值，防止后续计算倒数平方根时分母为 0
    const float eps = 1e-5f;
    const float mean = arma::as_scalar(arma::mean(arma::pow(in, 2))) + eps;

    // 计算缩放因子: scale = 1 / sqrt(sum_{j=0}^{dim-1} input[j]^2 / dim + eps)
    const float scale = 1.f / std::sqrt(mean);

    // 执行顺序：先算括号内 -> (scale * in) -> 再算 % wei
    // (scale * in): 标量 × 向量，arma 中会自动做 广播运算 -> temp[i] = scale * in[i]
    // wei % (scale * in): 运算符 % 表示逐元素相乘 -> out[i] = wei[i] * temp[i]
    out = wei % (scale * in);
}
}  // namespace kernel