#include <armadillo>
#include "add_kernel.h"

namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK(!input1.is_empty());
    CHECK(!input2.is_empty());
    CHECK(!output.is_empty());

    CHECK(input1.device_type() == base::DeviceType::DeviceCPU);
    CHECK(input2.device_type() == base::DeviceType::DeviceCPU);
    CHECK(output.device_type() == base::DeviceType::DeviceCPU);
    
    CHECK_EQ(input1.size(), output.size());
    CHECK_EQ(input2.size(), output.size());

    // arma::fvec 表示 单精度浮点 一维向量（float vector），封装了连续的浮点数组，支持高效的逐元素运算
    // 不拷贝数据，直接把张量的内存映射成 Armadillo 向量，避免内存拷贝的性能损耗
    // arma::fvec(
    //     float* mem_ptr,        // 参数1：内存指针 → CPU中存放数据的首地址 (指针需要去掉只读 const)
    //     const int n_elem,      // 参数2：向量长度 → 数据的总元素个数
    //     const bool copy_mem,   // 参数3：是否拷贝内存 → false = 不拷贝，共享内存，避免拷贝开销
    //     const bool strict_mem  // 参数4：是否严格绑定内存 → true = 严格绑定，不扩容/缩容，向量的长度永远是dim，不会越界访问内存
    // );
    arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

    // 逐元素相加
    output_vec = input1_vec + input2_vec;
}
}  // namespace kernel