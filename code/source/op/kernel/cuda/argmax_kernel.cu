#include "argmax_kernel.cuh"
#include "cuda_runtime.h"
#include "base/buffer.h"
#include <cfloat>

namespace kernel {
static __device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
    // unsigned active_mask = __activemask(); // 获取当前活跃线程的掩码
    #pragma unroll
    for (int32_t delta = warpSize >> 1; delta >= 1; delta >>= 1) {
        // 树形规约: 当前线程为 id，获取对应线程 (id + delta) 的值
        // 最终只有 lane0 的 val/idx 是整个 warp 的规约结果
        float other_val = __shfl_down_sync(0xffffffff, val, delta, warpSize);
        int32_t other_idx = __shfl_down_sync(0xffffffff, idx, delta, warpSize);
        
        // 比较并更新当前线程的最大值 val 及其下标 idx，当存在多个最大值时取最小下标
        if (other_idx != -1) { // 有效索引
            if (idx == -1 || val < other_val) {
                val = other_val;
                idx = other_idx;
            } else if (val == other_val && idx > other_idx) { // 值相同时取较小下标
                idx = other_idx;
            }
        }
    }
}

static __device__ __forceinline__ void block_reduce_argmax(float& val, int32_t& idx, float* __restrict__ shared_val, int32_t* __restrict__ shared_idx) {
    int32_t lane_id = threadIdx.x % warpSize;
    int32_t warp_id = threadIdx.x / warpSize;
    int32_t warp_num = blockDim.x / warpSize;
    
    // 对每个线程负责的局部最大值 val 及其下标 idx 进行 warp 内树形规约
    warp_reduce_argmax(val, idx);

    // 每个 warp 内 lane 0 线程取得了 warp 内的最大值 val 及其下标 idx
    if (lane_id == 0) {
        shared_val[warp_id] = val; // 第几个 warp 的规约结果 存储到第几个 shared
        shared_idx[warp_id] = idx;
    }
    __syncthreads(); // 共享内存写入之后需要同步，确保后续读取正确
    
    // 只对 warp 0 进行 warp 内树形规约，Block 内最大值规约结果存储到 thread 0 中
    if (warp_id == 0) {
        if (lane_id < warp_num) {
            val = shared_val[lane_id];
            idx = shared_idx[lane_id];
        } else {
            val = -FLT_MAX;
            idx = -1;
        }
        warp_reduce_argmax(val, idx);
    }
}

static __global__ void argmax_kernel_fp32(const float* __restrict__ input, int32_t size, int32_t* __restrict__ output_idx_cu_ptr) {
    // 每个线程求自己负责的局部最大值 val 及其下标 idx
    float val = -FLT_MAX;
    int32_t idx = -1;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        if (val < input[i]) {
            val = input[i];
            idx = i;
        }
    }

    // 块级规约需要共享内存
    static __shared__ float shared_val[32];
    static __shared__ int32_t shared_idx[32];

    // 块级规约求全局最大值 val 及其下标 idx
    block_reduce_argmax(val, idx, shared_val, shared_idx);

    // 只有 thread 0 拿着全局最大值 val 及其下标 idx，直接赋值即可
    if (threadIdx.x == 0) {
        *output_idx_cu_ptr = idx;
    }
}

int32_t argmax_kernel_cu(const float* input, int32_t size, void* stream) {
    int32_t output_idx_cpu = -1;
    auto allocator_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto output_idx_cu = std::make_unique<base::Buffer>(sizeof(int32_t), allocator_cu); // RAII 自动管理显存
    int32_t* output_idx_cu_ptr = static_cast<int32_t*>(output_idx_cu->ptr());

    constexpr int32_t block_num = 1;
    constexpr int32_t thread_num = 256;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    argmax_kernel_fp32<<<block_num, thread_num, 0, stream_>>>(input, size, output_idx_cu_ptr);
    if (stream_) {
        cudaMemcpyAsync(&output_idx_cpu, output_idx_cu_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        // 因为这里是异步 (Async) 调用，调用函数后会立马返回，所有 cuda 操作进入流任务队列，按照顺序执行
        // 这里必须要进行 cuda 流同步，保证流任务队列执行完成，即 output_idx_cu 结果计算完成
        cudaStreamSynchronize(stream_);
    } else {
        cudaMemcpy(&output_idx_cpu, output_idx_cu_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }
    return output_idx_cpu;
}
}  // namespace kernel