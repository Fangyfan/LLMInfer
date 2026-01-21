#include "add_kernel.cuh"

namespace kernel {
static __global__ void add_kernel_fp32(const int32_t size, const float* in1, const float* in2, float* out) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) {
        return;
    }
    out[tid] = in1[tid] + in2[tid];
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2, const tensor::Tensor& output, void* stream) {
    CHECK(!input1.is_empty());
    CHECK(!input2.is_empty());
    CHECK(!output.is_empty());

    CHECK(input1.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(input2.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    int32_t size = static_cast<int32_t>(output.size());
    CHECK_EQ(input1.size(), size);
    CHECK_EQ(input2.size(), size);

    cudaStream_t stream_ = nullptr;
    if (stream) {
        stream_ = static_cast<cudaStream_t>(stream);
    }

    const float* in1 = input1.ptr<float>();
    const float* in2 = input2.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    int32_t thread_num = 512;
    int32_t block_num = (size + thread_num - 1) / thread_num;
    if (stream_) { // 共享内存大小（这里用 0，不需要共享内存），stream_: 执行核函数的 CUDA 流（异步执行用，同步执行省略）
        add_kernel_fp32<<<block_num, thread_num, 0, stream_>>>(size, in1, in2, out);
    } else {
        add_kernel_fp32<<<block_num, thread_num>>>(size, in1, in2, out);
    }

    // 核函数启动后不会立即报错，需手动检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    CHECK(err == cudaSuccess) << "CUDA ADD kernel launch failed: " << cudaGetErrorString(err) << std::endl;
}
}  // namespace kernel