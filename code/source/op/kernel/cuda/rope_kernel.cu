#include "rope_kernel.cuh"
#include <cuda_bf16.h>

namespace kernel {

static __global__ void sin_cos_cache_kernel(
    float* __restrict__ sin_cache, 
    float* __restrict__ cos_cache, 
    int32_t head_dim, 
    int32_t max_seq_len
) {
    int32_t i = blockIdx.x;
    int32_t half_dim = head_dim / 2;

    const float freq = 1.0f / powf(1000000.0f, (2.0f * i) / head_dim);

    for (int32_t pos = threadIdx.x; pos < max_seq_len; pos += blockDim.x) {
        float theta = static_cast<float>(pos) * freq;
        sincosf(theta, sin_cache + pos * half_dim + i, cos_cache + pos * half_dim + i);
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
    
    CHECK(sin_cache.data_type() == base::DataType::DataTypeFp32);
    CHECK(cos_cache.data_type() == base::DataType::DataTypeFp32);

    CHECK(sin_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(cos_cache.device_type() == base::DeviceType::DeviceCUDA);

    float* sin_cache_ptr = const_cast<float*>(sin_cache.ptr<float>());
    float* cos_cache_ptr = const_cast<float*>(cos_cache.ptr<float>());
    
    dim3 blockDim(256);
    dim3 gridDim(head_dim / 2);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    sin_cos_cache_kernel<<<gridDim, blockDim, 0, stream_>>>(sin_cache_ptr, cos_cache_ptr, head_dim, max_seq_len);
}

static __global__ void rope_kernel(
    __nv_bfloat16* __restrict__ q, 
    __nv_bfloat16* __restrict__ k, 
    const float* __restrict__ sptr, 
    const float* __restrict__ cptr, 
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_dim
) {    
    // 当前线程负责第 tid 个 pair
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim / 2) {
        return;
    }

    // 每个 head 里面有 half_dim 个 pair(i, i + half_dim)
    int32_t half_dim = head_dim / 2;

    // 计算第 tid 个 pair 在第几个 head 以及 head 内的偏移量 i，表示 head 内的 pair(i, i + half_dim)
    int32_t head_id = tid / half_dim;
    int32_t pair_id = tid % half_dim;

    // 当 head_offset < kv_dim 时需要旋转 q 和 k，否则只需要旋转 q
    int32_t head_offset = head_id * head_dim;
    int32_t rotn = head_offset < kv_dim ? 2 : 1;

    // sin/cos cache
    float sin_theta = sptr[pair_id];
    float cos_theta = cptr[pair_id];

    for (int32_t r = 0; r < rotn; ++r) {
        __nv_bfloat16* v = static_cast<__nv_bfloat16*>(r == 0 ? q : k) + head_offset;
        float a = __bfloat162float(v[pair_id]);
        float b = __bfloat162float(v[pair_id + half_dim]);
        float a1 = a * cos_theta - b * sin_theta;
        float b1 = a * sin_theta + b * cos_theta;
        v[pair_id] = __float2bfloat16(a1);
        v[pair_id + half_dim] = __float2bfloat16(b1);
    }
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

    CHECK(query.data_type() == base::DataType::DataTypeBf16);
    CHECK(key.data_type() == base::DataType::DataTypeBf16);
    CHECK(token_pos.data_type() == base::DataType::DataTypeInt32);
    CHECK(sin_cache.data_type() == base::DataType::DataTypeFp32);
    CHECK(cos_cache.data_type() == base::DataType::DataTypeFp32);

    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(token_pos.device_type() == base::DeviceType::DeviceCPU);
    CHECK(sin_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(cos_cache.device_type() == base::DeviceType::DeviceCUDA);
    
    CHECK(head_dim % 2 == 0);

    __nv_bfloat16* q = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(query.ptr<uint16_t>()));
    __nv_bfloat16* k = reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(key.ptr<uint16_t>()));
    const int32_t pos = token_pos.index<int32_t>(0);
    const float* sptr = sin_cache.ptr<float>(pos * head_dim / 2); // sptr 索引到第 pos 行
    const float* cptr = cos_cache.ptr<float>(pos * head_dim / 2); // cptr 索引到第 pos 行

    dim3 blockDim(256);
    dim3 gridDim((dim / 2 + blockDim.x - 1) / blockDim.x); // 总共需要旋转 dim/2 个 pair
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rope_kernel<<<gridDim, blockDim, 0, stream_>>>(q, k, sptr, cptr, dim, kv_dim, head_dim);
}

}  // namespace kernel