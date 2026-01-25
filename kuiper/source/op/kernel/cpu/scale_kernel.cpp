#include "scale_kernel.h"
#include <armadillo>

namespace kernel {
void scale_kernel_cpu(const tensor::Tensor& input, float scale) {
    CHECK(!input.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCPU);

    // arma::fvec 表示 单精度浮点 一维向量（float vector），封装了连续的浮点数组，支持高效的逐元素运算
    // 不拷贝数据，直接把张量的内存映射成 Armadillo 向量，避免内存拷贝的性能损耗
    // arma::fvec(
    //     float* mem_ptr,        // 参数1：内存指针 → CPU中存放数据的首地址 (指针需要去掉只读 const)
    //     const int n_elem,      // 参数2：向量长度 → 数据的总元素个数
    //     const bool copy_mem,   // 参数3：是否拷贝内存 → false = 不拷贝，共享内存，避免拷贝开销
    //     const bool strict_mem  // 参数4：是否严格绑定内存 → true = 严格绑定，不扩容/缩容，向量的长度永远是dim，不会越界访问内存
    // );
    arma::fvec in(const_cast<float*>(input.ptr<float>()), input.size(), false, true);
    in *= scale;
}
}  // namespace kernel