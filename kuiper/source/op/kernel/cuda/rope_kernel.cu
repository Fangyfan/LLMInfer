#include "rope_kernel.cuh"

namespace kernel {
static __global__ void sin_cos_cache_kernel_fp32(
    float* __restrict__ sin_cache, 
    float* __restrict__ cos_cache, 
    int32_t head_dim, 
    int32_t max_seq_len
) {
    int32_t i = blockIdx.x;
    int32_t half = head_dim / 2;

    // 使用共享变量让 thread 0 算一次 freq，然后广播到其他线程复用
    __shared__ float shared_freq;
    if (threadIdx.x == 0) {
        shared_freq = 1.0f / powf(10000.0f, (2.0f * i) / head_dim);
    }
    __syncthreads();
    const float freq = shared_freq;

    for (int32_t pos = threadIdx.x; pos < max_seq_len; pos += blockDim.x) {
        float theta = static_cast<float>(pos) * freq;
        sincosf(theta, sin_cache + pos * half + i, cos_cache + pos * half + i);
        // *(sin_cache_ptr + pos * half + i) = sinf(theta);
        // *(cos_cache_ptr + pos * half + i) = cosf(theta);
    }
}

void sin_cos_cache_precompute_cu(
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t head_dim, 
    int32_t max_seq_len, 
    void* stream
) {
    CHECK(!sin_cache.is_empty());
    CHECK(!cos_cache.is_empty());
    CHECK(sin_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(cos_cache.device_type() == base::DeviceType::DeviceCUDA);

    float* sin_cache_ptr = const_cast<float*>(sin_cache.ptr<float>());
    float* cos_cache_ptr = const_cast<float*>(cos_cache.ptr<float>());
    
    const int32_t block_num = head_dim / 2;
    constexpr int32_t thread_num = 256;
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    sin_cos_cache_kernel_fp32<<<block_num, thread_num, 0, stream_>>>(sin_cache_ptr, cos_cache_ptr, head_dim, max_seq_len);
}

static __device__ __forceinline__ void rotate(float* __restrict__ v, float sin_theta, float cos_theta) {
    float2* v_pack = reinterpret_cast<float2*>(v);
    float2 v2 = *v_pack;
    *v_pack = make_float2(
        v2.x * cos_theta - v2.y * sin_theta, 
        v2.x * sin_theta + v2.y * cos_theta
    );
}

static __global__ void rope_kernel_fp32(
    float* __restrict__ q, 
    float* __restrict__ k, 
    const float* __restrict__ sptr, 
    const float* __restrict__ cptr, 
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_dim
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t i = 2 * tid; // i 指的是每个线程负责的全局 pair(2k, 2k+1) 中的 2k
    if (i >= dim) {
        return;
    }
    
    // 计算注意力头内部的 pair(2k, 2k+1) 中的 k
    // 因为 sin/cos cache 在每个注意力头内部都是相同的，k 表示 sin/cos pair 的索引
    int32_t pair_idx = (i % head_dim) / 2;
    float sin_theta = sptr[pair_idx];
    float cos_theta = cptr[pair_idx];

    rotate(q + i, sin_theta, cos_theta); // 旋转全局 (query[i], query[i + 1])
    if (i >= kv_dim) {
        return;
    }
    rotate(k + i, sin_theta, cos_theta); // 旋转全局 (key[i], key[i + 1])
}

void rope_kernel_cu(
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& token_pos, 
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_dim, 
    void* stream
) {
    CHECK(!query.is_empty());
    CHECK(!key.is_empty());
    CHECK(!token_pos.is_empty());
    CHECK(!sin_cache.is_empty());
    CHECK(!cos_cache.is_empty());
    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(token_pos.device_type() == base::DeviceType::DeviceCPU);
    CHECK(sin_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(cos_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(head_dim % 2 == 0);

    float* q = const_cast<float*>(query.ptr<float>());
    float* k = const_cast<float*>(key.ptr<float>());
    const int32_t pos = token_pos.index<int32_t>(0);
    const float* sptr = sin_cache.ptr<float>(pos * head_dim / 2); // sptr 索引到第 pos 行
    const float* cptr = cos_cache.ptr<float>(pos * head_dim / 2); // cptr 索引到第 pos 行

    constexpr int32_t thread_num = 256;
    const int32_t block_num = (dim / 2 + thread_num - 1) / thread_num; // 总共需要旋转 dim/2 个 pair
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rope_kernel_fp32<<<block_num, thread_num, 0, stream_>>>(q, k, sptr, cptr, dim, kv_dim, head_dim);
}
}  // namespace kernel