#include "softmax_kernel.cuh"

namespace kernel {
struct __align__(8) MD {
    float m;
    float d;
};

static __device__ __forceinline__ MD merge(const MD& a, const MD& b) {
    if (a.d == 0.0f) return b;
    if (b.d == 0.0f) return a;

    MD res;
    res.m = fmaxf(a.m, b.m);
    res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
    return res;
}

template <int WARP_SIZE>
static __device__ __forceinline__ MD warp_reduce(MD val) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val = merge(val, MD{
            __shfl_down_sync(0xffffffff, val.m, delta, WARP_SIZE), 
            __shfl_down_sync(0xffffffff, val.d, delta, WARP_SIZE)
        });
    }
    return val;
}

template <int WARP_NUM>
static __device__ __forceinline__ MD block_reduce(MD val) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    __shared__ MD shared_vals[WARP_NUM];
    __shared__ MD shared_val;
    val = warp_reduce<32>(val);
    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
        }
        val = warp_reduce<WARP_NUM>(val);
    }
    if (threadIdx.x == 0) {
        shared_val = val;
    }
    __syncthreads();
    val = shared_val;
    return val;
}

template <int32_t BLOCK_DIM>
static __global__ void softmax_kernel_fp32(float* __restrict__ input, int32_t size) {
    input += blockIdx.x * size;
    int32_t size4 = size >> 2;
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    MD md_temp[4], md_val;
    md_val.m = -INFINITY;
    md_val.d = 0.0f;

    float4* in4 = reinterpret_cast<float4*>(input);
    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 v = in4[i];
        md_temp[0] = {v.x, 1.0f};
        md_temp[1] = {v.y, 1.0f};
        md_temp[2] = {v.z, 1.0f};
        md_temp[3] = {v.w, 1.0f};

        md_val = merge(md_val, md_temp[0]);
        md_val = merge(md_val, md_temp[1]);
        md_val = merge(md_val, md_temp[2]);
        md_val = merge(md_val, md_temp[3]);
    }
    md_val = block_reduce<WARP_NUM>(md_val);

    for (int32_t i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 v = in4[i];
        in4[i] = make_float4(
            __expf(v.x - md_val.m) / md_val.d,
            __expf(v.y - md_val.m) / md_val.d,
            __expf(v.z - md_val.m) / md_val.d,
            __expf(v.w - md_val.m) / md_val.d
        );
    }
}

void softmax_kernel_cu(const tensor::Tensor& input, void* stream) {
    CHECK(!input.is_empty());
    CHECK(input.device_type() == base::DeviceType::DeviceCUDA);

    float* in = const_cast<float*>(input.ptr<float>());
    const int32_t size = static_cast<int32_t>(input.size());

    CHECK(size % 4 == 0);

    dim3 gridDim(1);
    dim3 blockDim(256);
    cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
    softmax_kernel_fp32<256><<<gridDim, blockDim, 0, stream_>>>(in, size);
}
}  // namespace kernel