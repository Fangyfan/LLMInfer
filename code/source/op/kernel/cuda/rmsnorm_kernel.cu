#include "rmsnorm_kernel.cuh"

namespace kernel {
template <int32_t WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
    }
    return val;
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared_vals[WARP_NUM];
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    val = warp_reduce_sum<32>(val);
    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
        }
        val = warp_reduce_sum<WARP_NUM>(val);
    }
    return val;
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void rmsnorm_kernel(
    const float* in, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t size, 
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const float4* in4 = reinterpret_cast<const float4*>(in);
    const float4* wei4 = reinterpret_cast<const float4*>(wei);
    float4* out4 = reinterpret_cast<float4*>(out);
    int32_t size4 = size >> 2;
    
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 v = in4[i];
        sum += (v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w);
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    float scale = shared_scale;

    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 a = in4[i];
        float4 b = wei4[i];
        out4[i] = make_float4(
            a.x * b.x * scale, 
            a.y * b.y * scale, 
            a.z * b.z * scale, 
            a.w * b.w * scale
        );
    }
}

void rmsnorm_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-6f;

    CHECK(size % 4 == 0);
    dim3 gridDim(1);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rmsnorm_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, size, eps);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_add_rmsnorm_kernel(
    const float* in, 
    const float* __restrict__ add, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t size, 
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const float4* in4 = reinterpret_cast<const float4*>(in);
    const float4* add4 = reinterpret_cast<const float4*>(add);
    const float4* wei4 = reinterpret_cast<const float4*>(wei);
    float4* out4 = reinterpret_cast<float4*>(out);
    int32_t size4 = size >> 2;
    
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 a = in4[i];
        float4 b = add4[i];
        sum += (a.x + b.x) * (a.x + b.x) + (a.y + b.y) * (a.y + b.y) + (a.z + b.z) * (a.z + b.z) + (a.w + b.w) * (a.w + b.w);
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    float scale = shared_scale;

    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 a = in4[i];
        float4 b = add4[i];
        float4 c = wei4[i];
        out4[i] = make_float4(
            (a.x + b.x) * c.x * scale, 
            (a.y + b.y) * c.y * scale, 
            (a.z + b.z) * c.z * scale, 
            (a.w + b.w) * c.w * scale
        );
    }
}

void fused_add_rmsnorm_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& residual_add, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    void* stream
) {
    CHECK(!input.is_empty());
    CHECK(!residual_add.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(residual_add.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const float* in = input.ptr<float>();
    const float* add = residual_add.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-6f;

    CHECK(size % 4 == 0);
    dim3 gridDim(1);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_add_rmsnorm_kernel<256><<<gridDim, blockDim, 0, stream_>>>(in, add, wei, out, size, eps);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qk_norm_rope_kernel(
    float* __restrict__ query, 
    float* __restrict__ key, 
    const float* __restrict__ weight, 
    const float* __restrict__ sin_cache, 
    const float* __restrict__ cos_cache, 
    int32_t head_num, 
    int32_t head_dim, 
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    float* in = (blockIdx.x < head_num) ? (query + blockIdx.x * head_dim) : (key + (blockIdx.x - head_num) * head_dim);
    const float* wei = (blockIdx.x < head_num) ? weight : (weight + head_dim);
    float* out = in;

    float val = in[threadIdx.x];
    float sum = val * val;
    sum = block_reduce_sum<WARP_NUM>(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / head_dim + eps);
    }
    __syncthreads();

    if (threadIdx.x < 64) {
        int32_t pair_id = threadIdx.x;
        float sin_theta = sin_cache[pair_id];
        float cos_theta = cos_cache[pair_id];
        
        float scale = shared_scale;
        float a = in[pair_id] * wei[pair_id] * scale;
        float b = in[pair_id + 64] * wei[pair_id + 64] * scale;
        
        out[pair_id] = a * cos_theta - b * sin_theta;
        out[pair_id + 64] = a * sin_theta + b * cos_theta;
    }
}

void fused_qk_norm_rope_kernel_cu(
    const tensor::Tensor& query, 
    const tensor::Tensor& key, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& token_pos, 
    const tensor::Tensor& sin_cache, 
    const tensor::Tensor& cos_cache, 
    int32_t dim, 
    int32_t kv_dim, 
    int32_t head_dim, 
    void* stream
) {
    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(token_pos.device_type() == base::DeviceType::DeviceCPU);
    CHECK(sin_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(cos_cache.device_type() == base::DeviceType::DeviceCUDA);
    
    CHECK(head_dim == 128);
    const int32_t head_num = dim / head_dim;
    const int32_t kv_head_num = kv_dim / head_dim;

    float* q = const_cast<float*>(query.ptr<float>());
    float* k = const_cast<float*>(key.ptr<float>());
    const float* wei = weight.ptr<float>();
    const int32_t pos = token_pos.index<int32_t>(0);
    const float* sptr = sin_cache.ptr<float>(pos * head_dim / 2); // sptr 索引到第 pos 行
    const float* cptr = cos_cache.ptr<float>(pos * head_dim / 2); // cptr 索引到第 pos 行
    constexpr float eps = 1e-6f;
    
    dim3 blockDim(head_dim);
    dim3 gridDim(head_num + kv_head_num);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    fused_qk_norm_rope_kernel<128><<<gridDim, blockDim, 0, stream_>>>(q, k, wei, sptr, cptr, head_num, head_dim, eps);
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void rmsnorm_2d_kernel(
    const float* in, 
    const float* __restrict__ wei, 
    float* out, 
    int32_t dim, 
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    in += blockIdx.x * dim;
    out += blockIdx.x * dim;
    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = in[i];
        sum += val * val;
    }
    sum = block_reduce_sum<WARP_NUM>(sum);
    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / dim + eps);
    }
    __syncthreads();
    float scale = shared_scale;
    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = in[i] * wei[i] * scale;
    }
}

void rmsnorm_2d_kernel_cu(
    const tensor::Tensor& input, 
    const tensor::Tensor& weight, 
    const tensor::Tensor& output, 
    int32_t dim, 
    void* stream
) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    const float* in = input.ptr<float>();
    const float* wei = weight.ptr<float>();
    float* out = const_cast<float*>(output.ptr<float>());
    int32_t size = static_cast<int32_t>(input.size());
    constexpr float eps = 1e-6f;
    
    dim3 blockDim(128);
    dim3 gridDim(size / dim);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    rmsnorm_2d_kernel<128><<<gridDim, blockDim, 0, stream_>>>(in, wei, out, dim, eps);
}
}  // namespace kernel