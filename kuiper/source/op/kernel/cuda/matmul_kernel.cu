#include "matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>

namespace kernel {
template<int32_t THREAD_NUM>
static __global__ void matmul_kernel_fp32(
    const float* __restrict__ input,    // [M, 1]
    const float* __restrict__ weight,   // [N, M]
    float* __restrict__ output,         // [N, 1]
    float scale, 
    int32_t N, int32_t M
) {
    using BlockReduce = cub::BlockReduce<float, THREAD_NUM>;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t row = blockIdx.x; // 每个 Block 处理一行点积
    int32_t row_offset = row * M;

    constexpr int32_t pack_size = 4;
    int32_t pack_num = M / pack_size;
    int32_t tail_off = pack_num * pack_size;
    
    float sum = 0.0f; // 每线程寄存器局部和
    const float4* in_pack = reinterpret_cast<const float4*>(input);
    const float4* wei_pack = reinterpret_cast<const float4*>(weight + row_offset);

    // 每线程处理多个 pack: i = tid, tid + blockDim, ...
    for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in_pack[i];
        float4 wei4 = wei_pack[i];

        // 点积累加到寄存器 sum
        sum += (in4.x * wei4.x) + (in4.y * wei4.y) + (in4.z * wei4.z) + (in4.w * wei4.w);
    }

    // 尾部不足 pack_size 的处理
    for (int32_t i = tail_off + threadIdx.x; i < M; i += blockDim.x) {
        sum += input[i] * weight[row_offset + i];
    }

    // 直接对寄存器 sum 做 Block 级别的求和归约
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());

    // Block 级别规约后结果存储在 thread 0 中，直接将结果计算到对应行 row 即可
    if (threadIdx.x == 0) {
        output[row] = sum * scale;
    }
}

void matmul_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    float scale, 
    void* stream
) {
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t N = weight.get_dim(0);
    const int32_t M = weight.get_dim(1);
    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    // CHECK_EQ(M % pack_size, 0);
    CHECK_EQ(input.get_dim(0), M);

    const int32_t block_num = N; // 每个 Block 处理矩阵向量乘法中的一行点积
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = M / pack_size;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    
    if (pack_num < 256) {
        constexpr int32_t thread_num = 128;
        matmul_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, scale, N, M);
    } else if (pack_num < 512) {
        constexpr int32_t thread_num = 256;
        matmul_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, scale, N, M);
    } else {
        constexpr int32_t thread_num = 512;
        matmul_kernel_fp32<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, scale, N, M);
    }
}

template<int32_t THREAD_NUM>
static __global__ void matmul_kernel_int8(
    const float* __restrict__ input,    // [M, 1]
    const int8_t* __restrict__ weight,  // [N, M]
    float* __restrict__ output,         // [N, 1]
    const float* __restrict__ scales, 
    int32_t group_size, 
    int32_t N, int32_t M
) {
    using BlockReduce = cub::BlockReduce<float, THREAD_NUM>;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t row = blockIdx.x; // 每个 Block 处理一行点积
    int32_t row_offset = row * M;

    // 每线程处理多个列: i = tid, tid + blockDim, ..., M - 1
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < M; i += blockDim.x) {
        int32_t idx = row_offset + i;
        int32_t group_id = idx / group_size;
        float real_weight = static_cast<float>(weight[idx]) * scales[group_id];
        sum += input[i] * real_weight;
    }

    // 直接对寄存器 sum 做 Block 级别的求和归约
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());

    // Block 级别规约后结果存储在 thread 0 中，直接将结果计算到对应行 row 即可
    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

void matmul_kernel_cu_quant8(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, const tensor::Tensor& scales, int32_t group_size, void* stream) {
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const int32_t N = weight.get_dim(0);
    const int32_t M = weight.get_dim(1);
    const float* in = input.ptr<float>();
    const int8_t* wei = weight.ptr<int8_t>();
    const float* scl = scales.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    constexpr int32_t pack_size = 4;
    // CHECK_EQ(M % pack_size, 0);
    CHECK_EQ(input.get_dim(0), M);
    CHECK(group_size % pack_size == 0);

    const int32_t block_num = N;
    constexpr int32_t thread_num = 128;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    matmul_kernel_int8<thread_num><<<block_num, thread_num, 0, stream_>>>(in, wei, out, scl, group_size, N, M);
}
}  // namespace kernel