#include "matmul_kernel.cuh"
#include <cub/block/block_reduce.cuh>

namespace kernel {
template<int32_t THREAD_PER_BLOCK, int32_t ROW_PER_BLOCK>
static __global__ void matmul_kernel_fp32(
    const float* __restrict__ input,    // [M, 1]
    const float* __restrict__ weight,   // [N, M]
    float* __restrict__ output,         // [N, 1]
    const int32_t N, const int32_t M
) {
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int32_t start_row = blockIdx.x * ROW_PER_BLOCK;
    const int32_t end_row = start_row + ROW_PER_BLOCK;

    constexpr int32_t pack_size = 4;
    const int32_t pack_num = M / pack_size;
    const int32_t tail_off = pack_num * pack_size;

    // 每行循环复用同一个 block
    #pragma unroll
    for (int32_t row = start_row; row < end_row; row++) {
        float sum = 0.f; // 每线程寄存器局部和
        const int32_t row_offset = M * row;

        const float4* in_pack = reinterpret_cast<const float4*>(input);
        const float4* wei_pack = reinterpret_cast<const float4*>(weight + row_offset);

        // 每线程处理多个 pack: i = tid, tid + blockDim, ...
        #pragma unroll
        for (int32_t i = threadIdx.x; i < pack_num; i += blockDim.x) {
            float4 in4 = in_pack[i];
            float4 wei4 = wei_pack[i];

            // 点积累加到寄存器 sum
            sum += (in4.x * wei4.x) + (in4.y * wei4.y) + (in4.z * wei4.z) + (in4.w * wei4.w);
        }

        // 尾部不足 pack_size (4) 的处理
        #pragma unroll
        for (int32_t i = tail_off + threadIdx.x; i < M; i += blockDim.x) {
            sum += input[i] * weight[row_offset + i];
        }

        // CUB block reduce：直接对寄存器 sum 做求和归约
        sum = BlockReduce(temp).Reduce(sum, cub::Sum());

        if (threadIdx.x == 0) {
            output[row] = sum;
        }

        __syncthreads(); // 关键：多行复用 temp，必须保证下一行归约前不会踩共享临时区
    }
}

template<int32_t THREAD_PER_BLOCK, int32_t ROW_PER_BLOCK>
static __global__ void matmul_kernel_int8(
    const float* __restrict__ input,    // [M, 1]
    const int8_t* __restrict__ weight,  // [N, M]
    float* __restrict__ output,         // [N, 1]
    const float* __restrict__ scales, 
    const int32_t group_size, const int32_t N, const int32_t M
) {
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int32_t start_row = blockIdx.x * ROW_PER_BLOCK;
    const int32_t end_row = start_row + ROW_PER_BLOCK;

    // 每行循环复用同一个 block
    #pragma unroll
    for (int32_t row = start_row; row < end_row; row++) {
        float sum = 0.f; // 每线程寄存器局部和
        const int32_t row_offset = M * row;

        // 每线程处理多个列: i = tid, tid + blockDim, ..., M - 1
        #pragma unroll
        for (int32_t i = threadIdx.x; i < M; i += blockDim.x) {
            const int32_t group_id = (row_offset + i) / group_size;
            const float real_weight = static_cast<float>(weight[i]) * scales[group_id];
            sum += input[i] * real_weight;
        }

        // CUB block reduce：直接对寄存器 sum 做求和归约
        sum = BlockReduce(temp).Reduce(sum, cub::Sum());

        if (threadIdx.x == 0) {
            output[row] = sum;
        }

        __syncthreads(); // 关键：多行复用 temp，必须保证下一行归约前不会踩共享临时区
    }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, void* stream) {
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty());
    
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    cudaStream_t stream_ = nullptr;
    if (stream) {
        stream_ = static_cast<cudaStream_t>(stream);
    }

    const int32_t N = weight.get_dim(0);
    const int32_t M = weight.get_dim(1);
    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());

    // constexpr int32_t pack_size = 4;
    // CHECK_EQ(M % pack_size, 0);
    CHECK_EQ(input.get_dim(0), M);

    constexpr int32_t thread_num = 128;
    constexpr int32_t row_per_block = 1;
    const int32_t block_num = (N + row_per_block - 1) / row_per_block;
    if (stream_) {
        matmul_kernel_fp32<thread_num, row_per_block><<<block_num, thread_num, 0, stream_>>>(in, wei, out, N, M);
    } else {
        matmul_kernel_fp32<thread_num, row_per_block><<<block_num, thread_num>>>(in, wei, out, N, M);
    }
}

void matmul_kernel_cu_quant8(const tensor::Tensor& input, const tensor::Tensor& weight, const tensor::Tensor& output, const tensor::Tensor& scales, int32_t group_size, void* stream) {
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(!output.is_empty());
    
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    cudaStream_t stream_ = nullptr;
    if (stream) {
        stream_ = static_cast<cudaStream_t>(stream);
    }

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

    constexpr int32_t thread_num = 128;
    constexpr int32_t row_per_block = 1;
    const int32_t block_num = (N + row_per_block - 1) / row_per_block;
    if (stream_) {
        matmul_kernel_int8<thread_num, row_per_block><<<block_num, thread_num, 0, stream_>>>(in, wei, out, scl, group_size, N, M);
    } else {
        matmul_kernel_int8<thread_num, row_per_block><<<block_num, thread_num>>>(in, wei, out, scl, group_size, N, M);
    }
}
}  // namespace kernel