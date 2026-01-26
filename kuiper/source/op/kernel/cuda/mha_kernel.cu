#include "mha_kernel.cuh"
#include "cub/block/block_reduce.cuh"
#include <cfloat>

namespace kernel {
template<int32_t THREAD_PER_BLOCK>
static __device__ void softmax_kernel_fp32(float* __restrict__ score_head, int32_t size) {
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    
    // 每个线程求自己负责的局部最大值
    float max_val = -FLT_MAX;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        if (max_val < score_head[i]) {
            max_val = score_head[i];
        }
    }

    // 块级规约求全局最大值
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());

    // 通过块内共享变量，将全局最大值由 thread 0 广播到块内所有线程
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // 每个线程对自己负责的值更新: 减去全局最大值 x = (x - max)，以至于 exp 后数值稳定，并进行局部求和
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        score_head[i] = expf(score_head[i] - max_val);
        sum += score_head[i];
    }

    // 块级规约进行全局求和
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());

    // 通过块内共享变量，将全局和由 thread 0 广播到块内所有线程
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // 每个线程对自己负责的值更新: 除以全局和，即 softmax(x) = e^{-x} / sum(e^{-x})
    for (int32_t i = threadIdx.x; i < size; i += blockDim.x) {
        score_head[i] /= sum;
    }
}

template<int32_t THREAD_PER_BLOCK>
static __global__ void mha_kernel_fp32(
    const float* __restrict__ query, 
    float* __restrict__ score, 
    const float* __restrict__ key_cache, 
    const float* __restrict__ value_cache, 
    float* __restrict__ output, 
    int32_t layer_offset, 
    int32_t pos, 
    int32_t kv_dim, 
    int32_t kv_mul, 
    int32_t head_dim, 
    int32_t max_seq_len
) {
    // 每个 Block 负责一个注意力头 Head 的独立计算
    int32_t h = blockIdx.x;
    constexpr int32_t pack_size = 4;
    const int32_t pack_num = head_dim / pack_size;
    float scale = rsqrtf(static_cast<float>(head_dim));
    int32_t head_offset = layer_offset + (h / kv_mul) * head_dim;
    
    extern __shared__ float shared_mem[]; // 动态块内共享数组: 用来临时存放 query_head，长度: head_dim
    float* query_head = shared_mem;
    for (int32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        query_head[i] = query[h * head_dim + i]; // 预加载 query_head 到共享内存
    }
    __syncthreads();
    
    float* score_head = score + h * max_seq_len;
    const float4* query_head_pack = reinterpret_cast<const float4*>(query_head);
    for (int32_t t = threadIdx.x; t <= pos; t += blockDim.x) {
        int32_t key_cache_offset = head_offset + t * kv_dim;
        const float4* key_head_pack = reinterpret_cast<const float4*>(key_cache + key_cache_offset);

        float sum = 0.0f;
        for (int32_t i = 0; i < pack_num; ++i) {
            float4 q4 = query_head_pack[i];
            float4 k4 = key_head_pack[i];
            sum += (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
        }
        score_head[t] = sum * scale;
    }
    
    __syncthreads();
    softmax_kernel_fp32<THREAD_PER_BLOCK>(score_head, pos + 1);
    __syncthreads();

    float* output_head = output + h * head_dim;
    for (int32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int32_t t = 0; t <= pos; ++t) {
            int32_t value_cache_offset = head_offset + t * kv_dim;
            float value_head = *(value_cache + value_cache_offset + i);
            sum += score_head[t] * value_head;
        }
        output_head[i] = sum;
    }
}

void mha_kernel_cu(
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
    UNUSED(score);
    CHECK(!query.is_empty());
    CHECK(!key_cache.is_empty());
    CHECK(!value_cache.is_empty());
    CHECK(!output.is_empty());
    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(value_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);
    
    constexpr int32_t pack_size = 4;
    CHECK_EQ(head_dim % pack_size, 0);

    const float* query_ptr = query.ptr<float>();
    float* score_ptr = const_cast<float*>(score.ptr<float>());
    const float* key_cache_ptr = key_cache.ptr<float>();
    const float* value_cache_ptr = value_cache.ptr<float>();
    float* output_ptr = const_cast<float*>(output.ptr<float>());
    const int32_t layer_offset = layer_id * max_seq_len * kv_dim;
    
    constexpr int32_t thread_num = 128;
    const int32_t block_num = head_num;
    const int32_t shared_size = head_dim * sizeof(float); // query_head + score_head
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    mha_kernel_fp32<thread_num><<<block_num, thread_num, shared_size, stream_>>>(
        query_ptr, score_ptr, key_cache_ptr, value_cache_ptr, output_ptr, layer_offset, pos, kv_dim, kv_mul, head_dim, max_seq_len
    );
}
}  // namespace kernel