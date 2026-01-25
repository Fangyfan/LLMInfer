#include <armadillo>
#include "matmul_kernel.h"

namespace kernel {
// 矩阵乘法: weight[wei_dim0, wei_dim1] × input[in_dim0, in_dim1] = output[wei_dim0, in_dim1]
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, float scale, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::DeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::DeviceCPU);
    CHECK(output.device_type() == base::DeviceType::DeviceCPU);

    int32_t in_dim0 = 1;
    int32_t in_dim1 = 1;
    if (input.dims_size() == 2) {
        in_dim0 = input.get_dim(0);
        in_dim1 = input.get_dim(1);
    } else if (input.dims_size() == 1) {
        in_dim0 = input.get_dim(0);
    } else {
        LOG(FATAL) << "The input tensor has a wrong dim size." << std::endl;
    }
    
    CHECK_EQ(weight.dims_size(), 2);
    const int32_t wei_dim0 = weight.get_dim(0);
    const int32_t wei_dim1 = weight.get_dim(1);
    CHECK_EQ(wei_dim1, in_dim0);
    
    CHECK_EQ(output.size(), wei_dim0 * in_dim1);

    // arma::fmat 表示 单精度浮点 二维矩阵（float matrix），封装了连续的浮点数组，支持高效的逐元素运算，列优先存储!!!
    // 不拷贝数据，直接把张量的内存映射成 Armadillo 矩阵，避免内存拷贝的性能损耗
    // arma::fmat(
    //     float* mem_ptr,          // 参数1：内存指针 → CPU中存放数据的首地址 (指针需要去掉只读 const)
    //     const int dim0,          // 参数2：矩阵行数
    //     const int dim1,          // 参数3：矩阵列数
    //     const bool copy_aux_mem, // 参数4：是否拷贝内存 → false = 不拷贝，共享内存，避免拷贝开销
    //     const bool strict_mem    // 参数5：是否严格绑定内存 → true = 严格绑定，不扩容/缩容，矩阵维度永远是 (dim0, dim1)，不会越界访问内存
    // );

    // Tensor 通常使用 行优先 (Row-Major) 存储
    // Armadillo 库默认使用 列优先 (Column-Major) 存储
    // 为了避免耗时的数据拷贝，代码使用了“欺骗”维度的技巧：将行优先矩阵封装为列优先矩阵，在数学上等同于对矩阵进行了转置 (A^T)
    // y = Ax => y^T = (Ax)^T = x^T × A^T
    arma::fmat in_mat(const_cast<float*>(input.ptr<float>()), in_dim1, in_dim0, false, true);
    arma::fmat wei_mat(const_cast<float*>(weight.ptr<float>()), wei_dim1, wei_dim0, false, true);
    arma::fmat out_mat(const_cast<float*>(output.ptr<float>()), in_dim1, wei_dim0, false, true);
    out_mat = (in_mat * wei_mat) * scale;
}
}  // namespace kernel