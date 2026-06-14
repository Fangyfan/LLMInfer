#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision test for the BF16 CUDA kernels used by Qwen3-4B-style single-token decode ops.

Run:
    python test_qwen3_bf16_ops.py

Requirements:
    - Linux
    - PyTorch with CUDA
    - nvcc / CUDA toolkit available for torch.utils.cpp_extension
    - NVIDIA GPU with BF16 support, usually SM80+
"""

import os
import math
import time
import traceback
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------
# Qwen3-4B official shapes
# -------------------------
VOCAB_SIZE = 151936
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 9728
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM          # 4096
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM         # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM                    # 6144
MAX_POSITION_EMBEDDINGS = 40960
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6

SEED = 20260614
BLOCK_DIM = 256
LM_HEAD_REF_CHUNK_ROWS = 2048
REF_CHUNK_ROWS = 2048

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be torch.bfloat16")
#define CHECK_FP32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be torch.float32")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be torch.int32")
#define CHECK_ALIGN16(x) TORCH_CHECK((reinterpret_cast<uintptr_t>((x).data_ptr()) & 15) == 0, #x " data_ptr must be 16-byte aligned")

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
    if (threadIdx.x == 0) {
        shared_vals[0] = val;
    }
    __syncthreads();
    return shared_vals[0];
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void gemv_bf16x8_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ out,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = __float2bfloat16(sum);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void gemv_bf16x8_fp32_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    float* __restrict__ out,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gemv_add_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ residual_add,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        float residual = __bfloat162float(residual_add[blockIdx.x]);
        residual_add[blockIdx.x] = __float2bfloat16(residual + sum);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qkv_gemv_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int32_t K, int32_t dim, int32_t kv_dim
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(input);
    const uint4* wei8 = reinterpret_cast<const uint4*>(weight + blockIdx.x * K);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            sum += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            sum += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    if (threadIdx.x == 0) {
        if (blockIdx.x < dim) {
            query[blockIdx.x] = __float2bfloat16(sum);
        } else if (blockIdx.x < dim + kv_dim) {
            key[blockIdx.x - dim] = __float2bfloat16(sum);
        } else {
            value[blockIdx.x - dim - kv_dim] = __float2bfloat16(sum);
        }
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gate_up_gemv_swiglu_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* __restrict__ out,
    int32_t immediate_dim,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    int32_t K8 = (K >> 3);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* gate8 = reinterpret_cast<const uint4*>(wei + blockIdx.x * K);
    const uint4* up8 = reinterpret_cast<const uint4*>(wei + (blockIdx.x + immediate_dim) * K);

    float gate = 0.0f;
    float up = 0.0f;
    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = gate8[i];
        uint4 c = up8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
        const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(&c);
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            gate += __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x);
            gate += __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y);
            up += __bfloat162float(a2[j].x) * __bfloat162float(c2[j].x);
            up += __bfloat162float(a2[j].y) * __bfloat162float(c2[j].y);
        }
    }
    gate = block_reduce_sum<WARP_NUM>(gate);
    up = block_reduce_sum<WARP_NUM>(up);

    if (threadIdx.x == 0) {
        float gate_silu = gate / (1.0f + __expf(-gate));
        out[blockIdx.x] = __float2bfloat16(gate_silu * up);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void rmsnorm_kernel(
    const __nv_bfloat16* in,
    const __nv_bfloat16* __restrict__ wei,
    __nv_bfloat16* out,
    int32_t size,
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const uint4* in8 = reinterpret_cast<const uint4*>(in);
    const uint4* wei8 = reinterpret_cast<const uint4*>(wei);
    uint4* out8 = reinterpret_cast<uint4*>(out);
    int32_t size8 = (size >> 3);

    float sum = 0.0f;
    for (int32_t i = threadIdx.x; i < size8; i += blockDim.x) {
        uint4 v = in8[i];
        const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum += __bfloat162float(v2[j].x) * __bfloat162float(v2[j].x);
            sum += __bfloat162float(v2[j].y) * __bfloat162float(v2[j].y);
        }
    }
    sum = block_reduce_sum<WARP_NUM>(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    float scale = shared_scale;

    uint32_t c[4];
    union {
        __nv_bfloat162 bf;
        uint32_t u;
    } cvt;
    for (int32_t i = threadIdx.x; i < size8; i += blockDim.x) {
        uint4 a = in8[i];
        uint4 b = wei8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x = __bfloat162float(a2[j].x) * __bfloat162float(b2[j].x) * scale;
            float y = __bfloat162float(a2[j].y) * __bfloat162float(b2[j].y) * scale;
            cvt.bf = __floats2bfloat162_rn(x, y);
            c[j] = cvt.u;
        }
        out8[i] = make_uint4(c[0], c[1], c[2], c[3]);
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache,
    int32_t head_num,
    int32_t head_dim,
    float eps
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    __nv_bfloat16* in = (blockIdx.x < head_num) ? (query + blockIdx.x * head_dim) : (key + (blockIdx.x - head_num) * head_dim);
    const __nv_bfloat16* wei = (blockIdx.x < head_num) ? weight : (weight + head_dim);
    __nv_bfloat16* out = in;

    float val = __bfloat162float(in[threadIdx.x]);
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
        float a = __bfloat162float(in[pair_id]) * __bfloat162float(wei[pair_id]) * scale;
        float b = __bfloat162float(in[pair_id + 64]) * __bfloat162float(wei[pair_id + 64]) * scale;

        float a1 = a * cos_theta - b * sin_theta;
        float b1 = a * sin_theta + b * cos_theta;

        out[pair_id] = __float2bfloat16(a1);
        out[pair_id + 64] = __float2bfloat16(b1);
    }
}

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

template <int32_t WARP_SIZE>
static __device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
#pragma unroll
    for (int32_t delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
        int32_t other_idx = __shfl_down_sync(0xffffffff, idx, delta, WARP_SIZE);

        if (other_idx != -1) {
            if (idx == -1 || val < other_val) {
                val = other_val;
                idx = other_idx;
            } else if (val == other_val && idx > other_idx) {
                idx = other_idx;
            }
        }
    }
}

template <int32_t WARP_NUM>
static __device__ __forceinline__ void block_reduce_argmax(float& val, int32_t& idx) {
    __shared__ float shared_vals[WARP_NUM];
    __shared__ int32_t shared_idxs[WARP_NUM];

    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;

    warp_reduce_argmax<32>(val, idx);
    if (lane == 0) {
        shared_vals[warp] = val;
        shared_idxs[warp] = idx;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
            idx = shared_idxs[lane];
        }
        warp_reduce_argmax<WARP_NUM>(val, idx);
    }
}

struct __align__(8) val_idx {
    float val;
    int32_t idx;
};

template <int32_t SM_NUM, int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void argmax_kernel_1(
    const float* __restrict__ input,
    int32_t size,
    val_idx* __restrict__ temp
) {
    constexpr int32_t WARP_NUM = BLOCK_DIM >> 5;

    int32_t size4 = size >> 2;
    const float4* __restrict__ input4 = reinterpret_cast<const float4*>(input);

    float val = -INFINITY;
    int32_t idx = -1;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size4; i += SM_NUM * BLOCK_DIM) {
        float4 other_val4 = input4[i];
        if (val < other_val4.x) { val = other_val4.x; idx = (i << 2); }
        if (val < other_val4.y) { val = other_val4.y; idx = (i << 2) + 1; }
        if (val < other_val4.z) { val = other_val4.z; idx = (i << 2) + 2; }
        if (val < other_val4.w) { val = other_val4.w; idx = (i << 2) + 3; }
    }
    block_reduce_argmax<WARP_NUM>(val, idx);

    if (threadIdx.x == 0) {
        temp[blockIdx.x] = {val, idx};
    }
}

static __global__ void argmax_kernel_2(
    const val_idx* __restrict__ temp,
    int32_t* __restrict__ output
) {
    float val = -INFINITY;
    int32_t idx = -1;
    int32_t tid = threadIdx.x << 2;

#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        val_idx other = temp[tid + i];
        if (val < other.val) {
            val = other.val;
            idx = other.idx;
        } else if (val == other.val && idx > other.idx) {
            idx = other.idx;
        }
    }
    warp_reduce_argmax<32>(val, idx);

    if (threadIdx.x == 0) {
        *output = idx;
    }
}

static inline const __nv_bfloat16* bf16_ptr_const(const torch::Tensor& x) {
    return reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
}

static inline __nv_bfloat16* bf16_ptr(torch::Tensor& x) {
    return reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
}

torch::Tensor gemv_bf16(torch::Tensor input, torch::Tensor weight) {
    CHECK_CUDA(input); CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight);
    CHECK_BF16(input); CHECK_BF16(weight);
    CHECK_ALIGN16(input); CHECK_ALIGN16(weight);
    TORCH_CHECK(input.dim() == 1 && weight.dim() == 2, "input [K], weight [N,K] expected");
    const int32_t K = (int32_t)input.size(0);
    const int32_t N = (int32_t)weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal K");
    TORCH_CHECK((K % 8) == 0, "K must be divisible by 8");
    auto out = torch::empty({N}, input.options());
    CHECK_ALIGN16(out);
    dim3 grid(N);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemv_bf16x8_bf16_kernel<256><<<grid, block, 0, stream>>>(bf16_ptr_const(input), bf16_ptr_const(weight), bf16_ptr(out), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gemv_fp32(torch::Tensor input, torch::Tensor weight) {
    CHECK_CUDA(input); CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight);
    CHECK_BF16(input); CHECK_BF16(weight);
    CHECK_ALIGN16(input); CHECK_ALIGN16(weight);
    TORCH_CHECK(input.dim() == 1 && weight.dim() == 2, "input [K], weight [N,K] expected");
    const int32_t K = (int32_t)input.size(0);
    const int32_t N = (int32_t)weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal K");
    TORCH_CHECK((K % 8) == 0, "K must be divisible by 8");
    auto out = torch::empty({N}, input.options().dtype(torch::kFloat32));
    dim3 grid(N);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemv_bf16x8_fp32_kernel<256><<<grid, block, 0, stream>>>(bf16_ptr_const(input), bf16_ptr_const(weight), out.data_ptr<float>(), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gemv_add_inplace(torch::Tensor input, torch::Tensor weight, torch::Tensor residual_add) {
    CHECK_CUDA(input); CHECK_CUDA(weight); CHECK_CUDA(residual_add);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(residual_add);
    CHECK_BF16(input); CHECK_BF16(weight); CHECK_BF16(residual_add);
    CHECK_ALIGN16(input); CHECK_ALIGN16(weight); CHECK_ALIGN16(residual_add);
    TORCH_CHECK(input.dim() == 1 && weight.dim() == 2 && residual_add.dim() == 1, "input [K], weight [N,K], residual [N] expected");
    const int32_t K = (int32_t)input.size(0);
    const int32_t N = (int32_t)weight.size(0);
    TORCH_CHECK(weight.size(1) == K && residual_add.size(0) == N, "shape mismatch");
    TORCH_CHECK((K % 8) == 0, "K must be divisible by 8");
    dim3 grid(N);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_gemv_add_bf16x8_kernel<256><<<grid, block, 0, stream>>>(bf16_ptr_const(input), bf16_ptr_const(weight), bf16_ptr(residual_add), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return residual_add;
}

std::vector<torch::Tensor> fused_qkv(torch::Tensor input, torch::Tensor weight, int64_t dim, int64_t kv_dim) {
    CHECK_CUDA(input); CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight);
    CHECK_BF16(input); CHECK_BF16(weight);
    CHECK_ALIGN16(input); CHECK_ALIGN16(weight);
    TORCH_CHECK(input.dim() == 1 && weight.dim() == 2, "input [hidden], weight [dim+2*kv_dim, hidden] expected");
    const int32_t K = (int32_t)input.size(0);
    const int32_t N = (int32_t)weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "weight.size(1) must equal K");
    TORCH_CHECK(N == dim + 2 * kv_dim, "weight.size(0) must equal dim + 2*kv_dim");
    TORCH_CHECK((K % 8) == 0, "K must be divisible by 8");
    auto query = torch::empty({dim}, input.options());
    auto key = torch::empty({kv_dim}, input.options());
    auto value = torch::empty({kv_dim}, input.options());
    CHECK_ALIGN16(query); CHECK_ALIGN16(key); CHECK_ALIGN16(value);
    dim3 grid(N);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_qkv_gemv_bf16x8_kernel<256><<<grid, block, 0, stream>>>(
        bf16_ptr_const(input), bf16_ptr_const(weight), bf16_ptr(query), bf16_ptr(key), bf16_ptr(value), K, (int32_t)dim, (int32_t)kv_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {query, key, value};
}

torch::Tensor fused_gate_up_swiglu(torch::Tensor input, torch::Tensor weight, int64_t immediate_dim) {
    CHECK_CUDA(input); CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight);
    CHECK_BF16(input); CHECK_BF16(weight);
    CHECK_ALIGN16(input); CHECK_ALIGN16(weight);
    TORCH_CHECK(input.dim() == 1 && weight.dim() == 2, "input [K], weight [2*I,K] expected");
    const int32_t K = (int32_t)input.size(0);
    TORCH_CHECK(weight.size(0) == 2 * immediate_dim && weight.size(1) == K, "shape mismatch");
    TORCH_CHECK((K % 8) == 0, "K must be divisible by 8");
    auto out = torch::empty({immediate_dim}, input.options());
    CHECK_ALIGN16(out);
    dim3 grid((int32_t)immediate_dim);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_gate_up_gemv_swiglu_kernel<256><<<grid, block, 0, stream>>>(bf16_ptr_const(input), bf16_ptr_const(weight), bf16_ptr(out), (int32_t)immediate_dim, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor rmsnorm(torch::Tensor input, torch::Tensor weight, double eps) {
    CHECK_CUDA(input); CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight);
    CHECK_BF16(input); CHECK_BF16(weight);
    CHECK_ALIGN16(input); CHECK_ALIGN16(weight);
    TORCH_CHECK(input.dim() == 1 && weight.dim() == 1, "input [size], weight [size] expected");
    const int32_t size = (int32_t)input.numel();
    TORCH_CHECK(weight.numel() == size, "weight.numel() must equal input.numel()");
    TORCH_CHECK((size % 8) == 0, "size must be divisible by 8");
    auto out = torch::empty_like(input);
    CHECK_ALIGN16(out);
    dim3 grid(1);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_kernel<256><<<grid, block, 0, stream>>>(bf16_ptr_const(input), bf16_ptr_const(weight), bf16_ptr(out), size, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> fused_qk_norm_rope_inplace(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor weight,
    torch::Tensor sin_cache_row,
    torch::Tensor cos_cache_row,
    int64_t dim,
    int64_t kv_dim,
    int64_t head_dim,
    double eps
) {
    CHECK_CUDA(query); CHECK_CUDA(key); CHECK_CUDA(weight); CHECK_CUDA(sin_cache_row); CHECK_CUDA(cos_cache_row);
    CHECK_CONTIGUOUS(query); CHECK_CONTIGUOUS(key); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(sin_cache_row); CHECK_CONTIGUOUS(cos_cache_row);
    CHECK_BF16(query); CHECK_BF16(key); CHECK_BF16(weight); CHECK_FP32(sin_cache_row); CHECK_FP32(cos_cache_row);
    TORCH_CHECK(head_dim == 128, "this kernel assumes head_dim == 128");
    TORCH_CHECK(query.numel() == dim && key.numel() == kv_dim, "q/k shape mismatch");
    TORCH_CHECK(weight.numel() == 2 * head_dim, "weight must contain q_norm_weight + k_norm_weight, shape [2*head_dim]");
    TORCH_CHECK(sin_cache_row.numel() == head_dim / 2 && cos_cache_row.numel() == head_dim / 2, "sin/cos row shape mismatch");
    int32_t head_num = (int32_t)(dim / head_dim);
    int32_t kv_head_num = (int32_t)(kv_dim / head_dim);
    dim3 grid(head_num + kv_head_num);
    dim3 block((int32_t)head_dim);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_qk_norm_rope_kernel<128><<<grid, block, 0, stream>>>(
        bf16_ptr(query), bf16_ptr(key), bf16_ptr_const(weight), sin_cache_row.data_ptr<float>(), cos_cache_row.data_ptr<float>(), head_num, (int32_t)head_dim, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {query, key};
}

std::vector<torch::Tensor> sin_cos_cache(int64_t head_dim, int64_t max_seq_len) {
    TORCH_CHECK(head_dim == 128, "this test is intended for head_dim=128");
    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto sin_cache = torch::empty({max_seq_len, head_dim / 2}, opts);
    auto cos_cache = torch::empty({max_seq_len, head_dim / 2}, opts);
    dim3 block(256);
    dim3 grid((int32_t)(head_dim / 2));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sin_cos_cache_kernel<<<grid, block, 0, stream>>>(sin_cache.data_ptr<float>(), cos_cache.data_ptr<float>(), (int32_t)head_dim, (int32_t)max_seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {sin_cache, cos_cache};
}

torch::Tensor argmax_token(torch::Tensor input) {
    CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_FP32(input); CHECK_ALIGN16(input);
    TORCH_CHECK(input.dim() == 1, "input must be 1D");
    const int32_t size = (int32_t)input.numel();
    TORCH_CHECK((size % 4) == 0, "size must be divisible by 4 for float4 loads");
    constexpr int32_t SM_NUM = 128;
    constexpr int32_t ARG_BLOCK_DIM = 512;
    auto temp_bytes = torch::empty({SM_NUM * 2}, input.options().dtype(torch::kInt32));
    auto out = torch::empty({1}, input.options().dtype(torch::kInt32));
    val_idx* temp = reinterpret_cast<val_idx*>(temp_bytes.data_ptr<int32_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    argmax_kernel_1<SM_NUM, ARG_BLOCK_DIM><<<SM_NUM, ARG_BLOCK_DIM, 0, stream>>>(input.data_ptr<float>(), size, temp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    argmax_kernel_2<<<1, 32, 0, stream>>>(temp, out.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''

CPP_DECLS = r'''
#include <torch/extension.h>
#include <vector>

torch::Tensor gemv_bf16(torch::Tensor input, torch::Tensor weight);
torch::Tensor gemv_fp32(torch::Tensor input, torch::Tensor weight);
torch::Tensor fused_gemv_add_inplace(torch::Tensor input, torch::Tensor weight, torch::Tensor residual_add);
std::vector<torch::Tensor> fused_qkv(torch::Tensor input, torch::Tensor weight, int64_t dim, int64_t kv_dim);
torch::Tensor fused_gate_up_swiglu(torch::Tensor input, torch::Tensor weight, int64_t immediate_dim);
torch::Tensor rmsnorm(torch::Tensor input, torch::Tensor weight, double eps);
std::vector<torch::Tensor> fused_qk_norm_rope_inplace(torch::Tensor query, torch::Tensor key, torch::Tensor weight, torch::Tensor sin_cache_row, torch::Tensor cos_cache_row, int64_t dim, int64_t kv_dim, int64_t head_dim, double eps);
std::vector<torch::Tensor> sin_cos_cache(int64_t head_dim, int64_t max_seq_len);
torch::Tensor argmax_token(torch::Tensor input);
'''


def build_extension():
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    print(f"[build] compiling inline CUDA extension for sm_{major}{minor} ...")
    t0 = time.time()
    ext = load_inline(
        name="qwen3_bf16_ops_precision_ext",
        cpp_sources=CPP_DECLS,
        cuda_sources=CUDA_SRC,
        functions=[
            "gemv_bf16",
            "gemv_fp32",
            "fused_gemv_add_inplace",
            "fused_qkv",
            "fused_gate_up_swiglu",
            "rmsnorm",
            "fused_qk_norm_rope_inplace",
            "sin_cos_cache",
            "argmax_token",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    torch.cuda.synchronize()
    print(f"[build] done in {time.time() - t0:.1f}s")
    return ext


def rand_bf16(shape, scale=1.0):
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * scale).to(torch.bfloat16).contiguous()


@torch.no_grad()
def ref_gemv_chunked(weight_bf16, input_bf16, chunk_rows=REF_CHUNK_ROWS):
    """FP32 accumulation reference without materializing the whole weight as fp32."""
    x = input_bf16.float()
    outs = []
    for s in range(0, weight_bf16.shape[0], chunk_rows):
        w = weight_bf16[s:s + chunk_rows].float()
        outs.append((w * x).sum(dim=1))
    return torch.cat(outs, dim=0)


def bf16_exact_rate(a, b):
    try:
        aa = a.detach().cpu().view(torch.int16)
        bb = b.detach().cpu().view(torch.int16)
        return (aa == bb).float().mean().item()
    except Exception:
        return float("nan")


@dataclass
class TestResult:
    name: str
    ok: bool
    max_abs: float
    mean_abs: float
    max_rel: float
    extra: str = ""


def compare(name, got, ref, atol, rtol, bf16=False, extra=""):
    torch.cuda.synchronize()
    gf = got.float()
    rf = ref.float()
    diff = (gf - rf).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    max_rel = (diff / rf.abs().clamp_min(1e-8)).max().item() if diff.numel() else 0.0
    ok = bool(torch.allclose(gf, rf, atol=atol, rtol=rtol))
    if bf16:
        er = bf16_exact_rate(got, ref)
        extra = (extra + " " if extra else "") + f"bf16_exact={er*100:.2f}%"
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:<28} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} max_rel={max_rel:.6g} atol={atol:g} rtol={rtol:g} {extra}")
    return TestResult(name, ok, max_abs, mean_abs, max_rel, extra)


@torch.no_grad()
def reference_qk_norm_rope(query, key, weight, sin_row, cos_row):
    q_weight = weight[:HEAD_DIM].float()
    k_weight = weight[HEAD_DIM:].float()

    def one(x, w):
        x = x.float().view(-1, HEAD_DIM)
        scale = torch.rsqrt((x * x).mean(dim=1, keepdim=True) + RMS_EPS)
        y = x * w.view(1, -1) * scale
        a = y[:, :HEAD_DIM // 2]
        b = y[:, HEAD_DIM // 2:]
        sr = sin_row.view(1, -1)
        cr = cos_row.view(1, -1)
        y0 = a * cr - b * sr
        y1 = a * sr + b * cr
        return torch.cat([y0, y1], dim=1).reshape(-1).to(torch.bfloat16)

    return one(query, q_weight), one(key, k_weight)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    major, minor = torch.cuda.get_device_capability()
    print(f"[env] torch={torch.__version__}, device={torch.cuda.get_device_name()}, capability=sm_{major}{minor}")
    if major < 8:
        raise RuntimeError("These BF16 kernels require an Ampere-or-newer GPU, usually sm80+.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("[shape] Qwen3-4B:", {
        "vocab": VOCAB_SIZE,
        "hidden": HIDDEN_SIZE,
        "intermediate": INTERMEDIATE_SIZE,
        "q_dim": Q_DIM,
        "kv_dim": KV_DIM,
        "head_dim": HEAD_DIM,
        "max_pos": MAX_POSITION_EMBEDDINGS,
    })

    ext = build_extension()
    results = []

    # 1) lm_head: [vocab, hidden] -> fp32 logits
    print("\n[test] lm_head gemv bf16x8 -> fp32")
    x_hidden = rand_bf16((HIDDEN_SIZE,), scale=1.0)
    w_lm = rand_bf16((VOCAB_SIZE, HIDDEN_SIZE), scale=0.02)
    got_lm = ext.gemv_fp32(x_hidden, w_lm)
    ref_lm = ref_gemv_chunked(w_lm, x_hidden, chunk_rows=LM_HEAD_REF_CHUNK_ROWS)
    results.append(compare("gemv_fp32_lm_head", got_lm, ref_lm, atol=8e-4, rtol=8e-4))

    # Also test bf16 output path on a real Qwen3 hidden projection size.
    print("\n[test] generic gemv bf16x8 -> bf16, N=hidden, K=hidden")
    w_hidden = rand_bf16((HIDDEN_SIZE, HIDDEN_SIZE), scale=0.02)
    got_gemv_bf16 = ext.gemv_bf16(x_hidden, w_hidden)
    ref_gemv_bf16 = ref_gemv_chunked(w_hidden, x_hidden).to(torch.bfloat16)
    results.append(compare("gemv_bf16_hidden", got_gemv_bf16, ref_gemv_bf16, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    del w_lm, got_lm, ref_lm, w_hidden, got_gemv_bf16, ref_gemv_bf16
    torch.cuda.empty_cache()

    # 2) qkv_proj: hidden -> q_dim + 2*kv_dim
    print("\n[test] fused_qkv_gemv_bf16x8")
    w_qkv = rand_bf16((QKV_DIM, HIDDEN_SIZE), scale=0.02)
    q, k, v = ext.fused_qkv(x_hidden, w_qkv, Q_DIM, KV_DIM)
    ref_qkv = ref_gemv_chunked(w_qkv, x_hidden).to(torch.bfloat16)
    results.append(compare("fused_qkv/query", q, ref_qkv[:Q_DIM], atol=2.0e-2, rtol=2.0e-2, bf16=True))
    results.append(compare("fused_qkv/key", k, ref_qkv[Q_DIM:Q_DIM + KV_DIM], atol=2.0e-2, rtol=2.0e-2, bf16=True))
    results.append(compare("fused_qkv/value", v, ref_qkv[Q_DIM + KV_DIM:], atol=2.0e-2, rtol=2.0e-2, bf16=True))

    # 3) qk-norm + qk-rope, including sin/cos cache
    print("\n[test] sin_cos_cache and fused_qk_norm_rope")
    sin_cache, cos_cache = ext.sin_cos_cache(HEAD_DIM, MAX_POSITION_EMBEDDINGS)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device="cuda", dtype=torch.float32) / HEAD_DIM))
    pos_ids = torch.arange(MAX_POSITION_EMBEDDINGS, device="cuda", dtype=torch.float32)
    freqs = torch.outer(pos_ids, inv_freq)
    ref_sin = torch.sin(freqs)
    ref_cos = torch.cos(freqs)
    results.append(compare("sin_cache", sin_cache, ref_sin, atol=2e-6, rtol=2e-6))
    results.append(compare("cos_cache", cos_cache, ref_cos, atol=2e-6, rtol=2e-6))

    pos = 12345
    q_norm_weight = rand_bf16((HEAD_DIM,), scale=0.05) + torch.ones((HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = rand_bf16((HEAD_DIM,), scale=0.05) + torch.ones((HEAD_DIM,), device="cuda", dtype=torch.bfloat16)
    qk_norm_weight = torch.cat([q_norm_weight, k_norm_weight]).contiguous()
    q_in = q.clone()
    k_in = k.clone()
    ref_q_rope, ref_k_rope = reference_qk_norm_rope(q_in, k_in, qk_norm_weight, sin_cache[pos], cos_cache[pos])
    q_out, k_out = ext.fused_qk_norm_rope_inplace(q_in.clone(), k_in.clone(), qk_norm_weight, sin_cache[pos].contiguous(), cos_cache[pos].contiguous(), Q_DIM, KV_DIM, HEAD_DIM, RMS_EPS)
    results.append(compare("q_norm_rope/query", q_out, ref_q_rope, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    results.append(compare("q_norm_rope/key", k_out, ref_k_rope, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    del w_qkv, ref_qkv, q, k, v, q_out, k_out, q_in, k_in, ref_q_rope, ref_k_rope
    torch.cuda.empty_cache()

    # 4) o_proj + residual add: input [q_dim], weight [hidden, q_dim], residual [hidden]
    print("\n[test] fused_gemv_add for o_proj + residual")
    x_attn = rand_bf16((Q_DIM,), scale=1.0)
    w_o = rand_bf16((HIDDEN_SIZE, Q_DIM), scale=0.02)
    residual_o = rand_bf16((HIDDEN_SIZE,), scale=1.0)
    got_o = ext.fused_gemv_add_inplace(x_attn, w_o, residual_o.clone())
    ref_o = (residual_o.float() + ref_gemv_chunked(w_o, x_attn)).to(torch.bfloat16)
    results.append(compare("fused_gemv_add/o_proj", got_o, ref_o, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    del x_attn, w_o, residual_o, got_o, ref_o
    torch.cuda.empty_cache()

    # 5) gate/up projection + SwiGLU: hidden -> intermediate
    print("\n[test] fused_gate_up_gemv_swiglu")
    w_gate_up = rand_bf16((2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), scale=0.02)
    got_swiglu = ext.fused_gate_up_swiglu(x_hidden, w_gate_up, INTERMEDIATE_SIZE)
    ref_gate = ref_gemv_chunked(w_gate_up[:INTERMEDIATE_SIZE], x_hidden)
    ref_up = ref_gemv_chunked(w_gate_up[INTERMEDIATE_SIZE:], x_hidden)
    ref_swiglu = (F.silu(ref_gate) * ref_up).to(torch.bfloat16)
    results.append(compare("fused_gate_up_swiglu", got_swiglu, ref_swiglu, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    del w_gate_up, got_swiglu, ref_gate, ref_up, ref_swiglu
    torch.cuda.empty_cache()

    # 6) down_proj + residual add: intermediate -> hidden
    print("\n[test] fused_gemv_add for down_proj + residual")
    x_mlp = rand_bf16((INTERMEDIATE_SIZE,), scale=0.15)
    w_down = rand_bf16((HIDDEN_SIZE, INTERMEDIATE_SIZE), scale=0.02)
    residual_down = rand_bf16((HIDDEN_SIZE,), scale=1.0)
    got_down = ext.fused_gemv_add_inplace(x_mlp, w_down, residual_down.clone())
    ref_down = (residual_down.float() + ref_gemv_chunked(w_down, x_mlp)).to(torch.bfloat16)
    results.append(compare("fused_gemv_add/down", got_down, ref_down, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    del x_mlp, w_down, residual_down, got_down, ref_down
    torch.cuda.empty_cache()

    # 7) RMSNorm over hidden size and over intermediate size.
    print("\n[test] rmsnorm")
    rms_w_hidden = rand_bf16((HIDDEN_SIZE,), scale=0.05) + torch.ones((HIDDEN_SIZE,), device="cuda", dtype=torch.bfloat16)
    got_rms_hidden = ext.rmsnorm(x_hidden, rms_w_hidden, RMS_EPS)
    ref_rms_hidden = (x_hidden.float() * rms_w_hidden.float() * torch.rsqrt((x_hidden.float() * x_hidden.float()).mean() + RMS_EPS)).to(torch.bfloat16)
    results.append(compare("rmsnorm/hidden", got_rms_hidden, ref_rms_hidden, atol=2.0e-2, rtol=2.0e-2, bf16=True))

    x_inter = rand_bf16((INTERMEDIATE_SIZE,), scale=1.0)
    rms_w_inter = rand_bf16((INTERMEDIATE_SIZE,), scale=0.05) + torch.ones((INTERMEDIATE_SIZE,), device="cuda", dtype=torch.bfloat16)
    got_rms_inter = ext.rmsnorm(x_inter, rms_w_inter, RMS_EPS)
    ref_rms_inter = (x_inter.float() * rms_w_inter.float() * torch.rsqrt((x_inter.float() * x_inter.float()).mean() + RMS_EPS)).to(torch.bfloat16)
    results.append(compare("rmsnorm/intermediate", got_rms_inter, ref_rms_inter, atol=2.0e-2, rtol=2.0e-2, bf16=True))
    del rms_w_hidden, got_rms_hidden, ref_rms_hidden, x_inter, rms_w_inter, got_rms_inter, ref_rms_inter
    torch.cuda.empty_cache()

    # 8) argmax over vocab logits. Add ties intentionally to test lower-index tie break.
    print("\n[test] argmax")
    logits = torch.randn((VOCAB_SIZE,), device="cuda", dtype=torch.float32).contiguous()
    logits[777] = 10.0
    logits[1234] = 10.0
    got_idx = int(ext.argmax_token(logits).item())
    ref_idx = int(torch.argmax(logits).item())
    ok_argmax = got_idx == ref_idx == 777
    print(f"[{'PASS' if ok_argmax else 'FAIL'}] argmax_token                  got={got_idx} ref={ref_idx}")
    results.append(TestResult("argmax_token", ok_argmax, 0.0, 0.0, 0.0, f"got={got_idx} ref={ref_idx}"))

    # Summary
    failed = [r for r in results if not r.ok]
    print("\n========== SUMMARY ==========")
    print(f"total={len(results)}, pass={len(results) - len(failed)}, fail={len(failed)}")
    if failed:
        print("failed tests:")
        for r in failed:
            print(f"  - {r.name}: max_abs={r.max_abs:.6g}, max_rel={r.max_rel:.6g} {r.extra}")
        raise SystemExit(1)
    print("All precision tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", repr(e))
        traceback.print_exc()
        raise