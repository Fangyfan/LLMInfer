#include "mha_kernel.cuh"
#include <cub/block/block_reduce.cuh>

namespace kernel {

template <int32_t WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, delta, WARP_SIZE);
    }
    return val;
}

template <int32_t BLOCK_DIM, int32_t Bc>
static __global__ __launch_bounds__(BLOCK_DIM) void flashattention_gqa_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ output,
    int32_t layer_offset,
    int32_t seq_len,
    int32_t kv_dim,
    int32_t kv_mul,
    int32_t head_dim,
    float scale
) {
    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem;            // [head_dim]
    float* s_K = s_Q + head_dim;        // [Bc, head_dim]
    float* s_V = s_K + Bc * head_dim;   // [Bc, head_dim]
    float* s_O = s_V + Bc * head_dim;   // [head_dim]
    float* s_S = s_O + head_dim;        // [Bc]
    float* s_P = s_S + Bc;              // [Bc]

    // 单 block 负责一个 Q head，适合短上下文，避免 split path 的二次 kernel launch
    const int32_t head_id = blockIdx.x;
    const int32_t kv_head_id = head_id / kv_mul;
    const int32_t head_offset = layer_offset + kv_head_id * head_dim;

    const int32_t lane = threadIdx.x & 31;
    const int32_t warp = threadIdx.x >> 5;

    // 搬运 Q tile: thread_num = head_dim = 128
    s_Q[threadIdx.x] = query[head_id * head_dim + threadIdx.x];
    s_O[threadIdx.x] = 0.0f;
    __syncthreads();
    float4 q4 = reinterpret_cast<float4*>(s_Q)[lane];
    
    float old_max = -INFINITY;
    float old_exp_sum = 0.0f;
    
    const int32_t seq_len_remain = seq_len % Bc;
    const int32_t seq_len_multi_Bc = seq_len - seq_len_remain;
    for (int32_t seq_offset = 0; seq_offset < seq_len_multi_Bc; seq_offset += Bc) {
        // 搬运 K/V tile: thread_num = head_dim = 128
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            s_K[i * head_dim + threadIdx.x] = key_cache[head_offset + (seq_offset + i) * kv_dim + threadIdx.x];
            s_V[i * head_dim + threadIdx.x] = value_cache[head_offset + (seq_offset + i) * kv_dim + threadIdx.x];
        }
        __syncthreads();
        
        // 每个 warp 计算一行点积: dot(q, k^T)，1 个 block 有 4 个 warp，每次计算 4 行 k
#pragma unroll
        for (int32_t i = 0; i < Bc; i += 4) {
            float4 k4 = reinterpret_cast<float4*>(s_K + (i + warp) * head_dim)[lane];
            float qk = (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
            qk = warp_reduce_sum<32>(qk);
            if (lane == 0) {
                s_S[i + warp] = scale * qk;
            }
        }
        __syncthreads();

        float row_max = -INFINITY;
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            row_max = fmaxf(row_max, s_S[i]);
        }
        float new_max = fmaxf(old_max, row_max);
        float exp_scale = __expf(old_max - new_max);
        if (warp == 0 && lane < Bc) {
            s_P[lane] = __expf(s_S[lane] - new_max);
        }
        __syncthreads();

        float row_exp_sum = 0.0f;
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            row_exp_sum += s_P[i];
        }
        old_exp_sum = exp_scale * old_exp_sum + row_exp_sum;
        old_max = new_max;

        // 计算 O = PV: thread_num = head_dim = 128
        float new_O = exp_scale * s_O[threadIdx.x];
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            new_O += s_P[i] * s_V[i * head_dim + threadIdx.x];
        }
        s_O[threadIdx.x] = new_O;
        __syncthreads();
    }

    if (seq_len_remain > 0) {
        for (int32_t i = 0; i < seq_len_remain; ++i) {
            s_K[i * head_dim + threadIdx.x] = key_cache[head_offset + (seq_len_multi_Bc + i) * kv_dim + threadIdx.x];
            s_V[i * head_dim + threadIdx.x] = value_cache[head_offset + (seq_len_multi_Bc + i) * kv_dim + threadIdx.x];
        }
        __syncthreads();
        
        for (int32_t i = 0; i < seq_len_remain; i += 4) {
            float4 k4 = reinterpret_cast<float4*>(s_K + (i + warp) * head_dim)[lane];
            float qk = (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
            qk = warp_reduce_sum<32>(qk);
            if (lane == 0) {
                s_S[i + warp] = scale * qk;
            }
        }
        __syncthreads();

        float row_max = -INFINITY;
        for (int32_t i = 0; i < seq_len_remain; ++i) {
            row_max = fmaxf(row_max, s_S[i]);
        }
        float new_max = fmaxf(old_max, row_max);
        float exp_scale = __expf(old_max - new_max);
        if (warp == 0 && lane < seq_len_remain) {
            s_P[lane] = __expf(s_S[lane] - new_max);
        }
        __syncthreads();

        float row_exp_sum = 0.0f;
        for (int32_t i = 0; i < seq_len_remain; ++i) {
            row_exp_sum += s_P[i];
        }
        old_exp_sum = exp_scale * old_exp_sum + row_exp_sum;
        old_max = new_max;

        float new_O = exp_scale * s_O[threadIdx.x];
        for (int32_t i = 0; i < seq_len_remain; ++i) {
            new_O += s_P[i] * s_V[i * head_dim + threadIdx.x];
        }
        s_O[threadIdx.x] = new_O;
        __syncthreads();
    }

    output[head_id * head_dim + threadIdx.x] = s_O[threadIdx.x] / old_exp_sum;
}

template <int32_t BLOCK_DIM, int32_t Bc>
static __global__ __launch_bounds__(BLOCK_DIM) void flashdecoding_gqa_split_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ kv_split_output,
    int32_t layer_offset,
    int32_t seq_len,
    int32_t kv_dim,
    int32_t kv_mul,
    int32_t head_dim,
    int32_t kv_split_size,
    int32_t kv_split_num,
    float scale
) {
    extern __shared__ float shared_mem[];
    float* s_Q = shared_mem;            // [head_dim]
    float* s_K = s_Q + head_dim;        // [Bc, head_dim]
    float* s_V = s_K + Bc * head_dim;   // [Bc, head_dim]
    float* s_O = s_V + Bc * head_dim;   // [head_dim]
    float* s_S = s_O + head_dim;        // [Bc]
    float* s_P = s_S + Bc;              // [Bc]
    
    // grid.y = q head
    // grid.x = kv split id
    const int32_t head_id = blockIdx.y;
    const int32_t kv_split_id = blockIdx.x;
    const int32_t kv_head_id = head_id / kv_mul;
    const int32_t head_offset = layer_offset + kv_head_id * head_dim;

    const int32_t lane = threadIdx.x & 31;
    const int32_t warp = threadIdx.x >> 5;

    int32_t kv_split_begin = kv_split_id * kv_split_size;
    int32_t kv_split_end = kv_split_begin + kv_split_size;
    if (kv_split_end > seq_len) {
        kv_split_end = seq_len;
    }
    int32_t kv_split_end_remain = kv_split_end % Bc;
    int32_t kv_split_end_multi_Bc = kv_split_end - kv_split_end_remain;

    s_Q[threadIdx.x] = query[head_id * head_dim + threadIdx.x];
    s_O[threadIdx.x] = 0.0f;
    __syncthreads();
    float4 q4 = reinterpret_cast<float4*>(s_Q)[lane];
    
    float old_max = -INFINITY;
    float old_exp_sum = 0.0f;

    for (int32_t seq_offset = kv_split_begin; seq_offset < kv_split_end_multi_Bc; seq_offset += Bc) {
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            s_K[i * head_dim + threadIdx.x] = key_cache[head_offset + (seq_offset + i) * kv_dim + threadIdx.x];
            s_V[i * head_dim + threadIdx.x] = value_cache[head_offset + (seq_offset + i) * kv_dim + threadIdx.x];
        }
        __syncthreads();
        
#pragma unroll
        for (int32_t i = 0; i < Bc; i += 4) {
            float4 k4 = reinterpret_cast<float4*>(s_K + (i + warp) * head_dim)[lane];
            float qk = (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
            qk = warp_reduce_sum<32>(qk);
            if (lane == 0) {
                s_S[i + warp] = scale * qk;
            }
        }
        __syncthreads();

        float row_max = -INFINITY;
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            row_max = fmaxf(row_max, s_S[i]);
        }
        float new_max = fmaxf(old_max, row_max);
        float exp_scale = __expf(old_max - new_max);
        if (warp == 0 && lane < Bc) {
            s_P[lane] = __expf(s_S[lane] - new_max);
        }
        __syncthreads();

        float row_exp_sum = 0.0f;
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            row_exp_sum += s_P[i];
        }
        old_exp_sum = exp_scale * old_exp_sum + row_exp_sum;
        old_max = new_max;

        float new_O = exp_scale * s_O[threadIdx.x];
#pragma unroll
        for (int32_t i = 0; i < Bc; ++i) {
            new_O += s_P[i] * s_V[i * head_dim + threadIdx.x];
        }
        s_O[threadIdx.x] = new_O;
        __syncthreads();
    }

    if (kv_split_end_remain > 0) {
        for (int32_t i = 0; i < kv_split_end_remain; ++i) {
            s_K[i * head_dim + threadIdx.x] = key_cache[head_offset + (kv_split_end_multi_Bc + i) * kv_dim + threadIdx.x];
            s_V[i * head_dim + threadIdx.x] = value_cache[head_offset + (kv_split_end_multi_Bc + i) * kv_dim + threadIdx.x];
        }
        __syncthreads();
        
        for (int32_t i = 0; i < kv_split_end_remain; i += 4) {
            float4 k4 = reinterpret_cast<float4*>(s_K + (i + warp) * head_dim)[lane];
            float qk = (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
            qk = warp_reduce_sum<32>(qk);
            if (lane == 0) {
                s_S[i + warp] = scale * qk;
            }
        }
        __syncthreads();

        float row_max = -INFINITY;
        for (int32_t i = 0; i < kv_split_end_remain; ++i) {
            row_max = fmaxf(row_max, s_S[i]);
        }
        float new_max = fmaxf(old_max, row_max);
        float exp_scale = __expf(old_max - new_max);
        if (warp == 0 && lane < kv_split_end_remain) {
            s_P[lane] = __expf(s_S[lane] - new_max);
        }
        __syncthreads();

        float row_exp_sum = 0.0f;
        for (int32_t i = 0; i < kv_split_end_remain; ++i) {
            row_exp_sum += s_P[i];
        }
        old_exp_sum = exp_scale * old_exp_sum + row_exp_sum;
        old_max = new_max;

        float new_O = exp_scale * s_O[threadIdx.x];
        for (int32_t i = 0; i < kv_split_end_remain; ++i) {
            new_O += s_P[i] * s_V[i * head_dim + threadIdx.x];
        }
        s_O[threadIdx.x] = new_O;
        __syncthreads();
    }

    kv_split_output += head_id * kv_split_num * (head_dim + 2);
    kv_split_output[kv_split_id * head_dim + threadIdx.x] = s_O[threadIdx.x];
    if (threadIdx.x == 0) {
        kv_split_output[kv_split_num * head_dim + kv_split_id] = old_max;
        kv_split_output[kv_split_num * (head_dim + 1) + kv_split_id] = old_exp_sum;
    }
}

template <int32_t WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, delta, WARP_SIZE));
    }
    return val;
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void flashdecoding_gqa_combine_kernel(
    const float* __restrict__ kv_split_output,
    float* __restrict__ output,
    int32_t head_dim,
    int32_t kv_split_size,
    int32_t kv_split_num
) {
    const int32_t head_id = blockIdx.x;
    const int32_t lane = threadIdx.x & 31;
    kv_split_output += head_id * kv_split_num * (head_dim + 2);

    extern __shared__ float shared_mem[];
    float* partial_max = shared_mem;
    float* partial_exp_sum = partial_max + kv_split_num;

    for (int32_t kv_split_id = threadIdx.x; kv_split_id < kv_split_num; kv_split_id += BLOCK_DIM) {
        partial_max[kv_split_id] = kv_split_output[kv_split_num * head_dim + kv_split_id];
        partial_exp_sum[kv_split_id] = kv_split_output[kv_split_num * (head_dim + 1) + kv_split_id];
    }
    __syncthreads();
    
    float max = -INFINITY;
    for (int32_t kv_split_id = lane; kv_split_id < kv_split_num; kv_split_id += 32) {
        max = fmaxf(max, partial_max[kv_split_id]);
    }
    max = warp_reduce_max<32>(max);

    float exp_sum = 0.0f;
    for (int32_t kv_split_id = lane; kv_split_id < kv_split_num; kv_split_id += 32) {
        exp_sum += partial_exp_sum[kv_split_id] * __expf(partial_max[kv_split_id] - max);
    }
    exp_sum = warp_reduce_sum<32>(exp_sum);

    float final_out = 0.0f;
    for (int32_t kv_split_id = 0; kv_split_id < kv_split_num; ++kv_split_id) {
        final_out += kv_split_output[kv_split_id * head_dim + threadIdx.x] * __expf(partial_max[kv_split_id] - max);
    }
    output[head_id * head_dim + threadIdx.x] = final_out / exp_sum;
}

void mha_kernel_cu(
    const tensor::Tensor& query,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    const tensor::Tensor& kv_split_output,
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
    CHECK(!query.is_empty());
    CHECK(!kv_split_output.is_empty());
    CHECK(!key_cache.is_empty());
    CHECK(!value_cache.is_empty());
    CHECK(!output.is_empty());

    CHECK(query.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(kv_split_output.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(key_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(value_cache.device_type() == base::DeviceType::DeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::DeviceCUDA);

    CHECK(pos >= 0);
    CHECK(pos < max_seq_len);

    // 让一个 thread 负责一个 output channel，所以要求 head_dim = 128
    // Qwen3-4B dense decode 常见 head_dim = 128，正好匹配
    constexpr int32_t thread_num = 128;
    CHECK(head_dim == thread_num);

    const float* query_ptr = query.ptr<float>();
    float* kv_split_output_ptr = const_cast<float*>(kv_split_output.ptr<float>());
    const float* key_cache_ptr = key_cache.ptr<float>();
    const float* value_cache_ptr = value_cache.ptr<float>();
    float* output_ptr = const_cast<float*>(output.ptr<float>());

    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int32_t layer_offset = layer_id * max_seq_len * kv_dim;
    const int32_t seq_len = pos + 1;

    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;

    // shared memory:
    //   s_Q / s_O: [head_dim] floats
    //   s_K / s_V: [Bc * head_dim] floats
    //   s_S / s_P: [Bc] floats
    constexpr int32_t Bc = 32;
    const int32_t shared_size = 2 * (head_dim + Bc * head_dim + Bc) * sizeof(float);
    if (seq_len <= 128) {
        flashattention_gqa_kernel<thread_num, Bc><<<head_num, thread_num, shared_size, stream_>>>(
            query_ptr, key_cache_ptr, value_cache_ptr, output_ptr, layer_offset, seq_len, kv_dim, kv_mul, head_dim, scale
        );
        return;
    }

    // 短上下文走单 kernel，避免额外 combine kernel launch
    // 长上下文走 split-KV，提高 block 数量，避免只有 head_num 个 block 导致 SM 利用率低
    constexpr int32_t kv_split_size = Bc;
    const int32_t kv_split_num = (seq_len + kv_split_size - 1) / kv_split_size;

    dim3 gridDim(kv_split_num, head_num);
    flashdecoding_gqa_split_kernel<thread_num, Bc><<<gridDim, thread_num, shared_size, stream_>>>(
        query_ptr, key_cache_ptr, value_cache_ptr, kv_split_output_ptr, layer_offset, seq_len, kv_dim, kv_mul, head_dim,
        kv_split_size, kv_split_num, scale
    );

    const int32_t combine_shared_size = 2 * kv_split_num * sizeof(float);
    flashdecoding_gqa_combine_kernel<thread_num><<<head_num, thread_num, combine_shared_size, stream_>>>(
        kv_split_output_ptr, output_ptr, head_dim, kv_split_size, kv_split_num
    );
}

}  // namespace kernel