#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy smoke test for the custom Int4x8/BF16x8 GEMV kernels used by Qwen3-4B-AWQ.

The dimensions are fixed to Qwen3-4B:
  hidden_size        = 2560
  head_dim           = 128
  num_attention_heads= 32  -> q_dim  = 4096
  num_kv_heads       = 8   -> kv_dim = 1024
  intermediate_size  = 9728
  AWQ group_size     = 128

Tested fused ops:
  1. qkv_proj:               [6144, 2560] -> q[4096], k[1024], v[1024]
  2. o_proj + residual add:  [2560, 4096] -> [2560]
  3. gate_up_proj + SwiGLU:  [19456,2560] -> [9728]
  4. down_proj + add:        [2560, 9728] -> [2560]

By default this script generates synthetic BF16 weights with the same shapes, quantizes them into
exactly the layout expected by the kernels, runs the kernels, and compares against two references:
  - implementation check: kernel output vs dequantized-weight fp32 reference rounded to BF16
  - quantization loss:    dequantized-weight fp32 reference vs original-BF16-weight fp32 reference

This is intended to validate dimensions, packing layout, and numerical loss before plugging in
real exported Qwen3-4B-AWQ tensors from your converter.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


# -----------------------------
# Qwen3-4B fixed dimensions
# -----------------------------
@dataclass(frozen=True)
class Qwen3_4B_Dims:
    hidden_size: int = 2560
    head_dim: int = 128
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 9728
    group_size: int = 128

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim  # 4096

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 1024

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim  # 6144


DIMS = Qwen3_4B_Dims()


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <vector>

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
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gemv_add_int4x8_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const int32_t* __restrict__ wei,
    const int32_t* __restrict__ zeros,
    const half* __restrict__ scales,
    __nv_bfloat16* __restrict__ residual_add,
    int32_t group_size,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t n_pack_id = blockIdx.x;
    const int32_t out_base = n_pack_id << 3;
    const int32_t group_num = K / group_size;

    const int32_t* __restrict__ wei_row = wei + n_pack_id * K;
    const int32_t* __restrict__ zero_row = zeros + n_pack_id * group_num;

    float sum[8] = {0.0f};

    const int32_t K8 = K >> 3;
    const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(in);

    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        const int32_t k_base = i << 3;
        const int32_t group_id = k_base / group_size;

        const uint4 a = in8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);

        float x[8] = {
            __bfloat162float(a2[0].x), __bfloat162float(a2[0].y),
            __bfloat162float(a2[1].x), __bfloat162float(a2[1].y),
            __bfloat162float(a2[2].x), __bfloat162float(a2[2].y),
            __bfloat162float(a2[3].x), __bfloat162float(a2[3].y)
        };

        const uint32_t zero_pack8 = static_cast<uint32_t>(zero_row[group_id]);

        float zero[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            zero[j] = static_cast<float>((zero_pack8 >> (j << 2)) & 0xF);
        }

        float scale[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            scale[j] = __half2float(scales[(out_base + j) * group_num + group_id]);
        }

        const int32_t* wei_ptr = wei_row + k_base;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t wei_pack8 = static_cast<uint32_t>(wei_ptr[j]);

#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float wei_val = static_cast<float>((wei_pack8 >> (k << 2)) & 0xF);
                sum[k] += x[j] * ((wei_val - zero[k]) * scale[k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum[i] = block_reduce_sum<WARP_NUM>(sum[i]);
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float add = __bfloat162float(residual_add[out_base + i]);
            residual_add[out_base + i] = __float2bfloat16(add + sum[i]);
        }
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_qkv_gemv_int4x8_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const int32_t* __restrict__ wei,
    const int32_t* __restrict__ zeros,
    const half* __restrict__ scales,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int32_t group_size,
    int32_t K,
    int32_t dim,
    int32_t kv_dim
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t n_pack_id = blockIdx.x;
    const int32_t out_base = n_pack_id << 3;
    const int32_t group_num = K / group_size;

    const int32_t* __restrict__ wei_row = wei + n_pack_id * K;
    const int32_t* __restrict__ zero_row = zeros + n_pack_id * group_num;

    float sum[8] = {0.0f};

    const int32_t K8 = K >> 3;
    const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(in);

    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        const int32_t k_base = i << 3;
        const int32_t group_id = k_base / group_size;

        const uint4 a = in8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);

        float x[8] = {
            __bfloat162float(a2[0].x), __bfloat162float(a2[0].y),
            __bfloat162float(a2[1].x), __bfloat162float(a2[1].y),
            __bfloat162float(a2[2].x), __bfloat162float(a2[2].y),
            __bfloat162float(a2[3].x), __bfloat162float(a2[3].y)
        };

        const uint32_t zero_pack8 = static_cast<uint32_t>(zero_row[group_id]);

        float zero[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            zero[j] = static_cast<float>((zero_pack8 >> (j << 2)) & 0xF);
        }

        float scale[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            scale[j] = __half2float(scales[(out_base + j) * group_num + group_id]);
        }

        const int32_t* wei_ptr = wei_row + k_base;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t wei_pack8 = static_cast<uint32_t>(wei_ptr[j]);

#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float wei_val = static_cast<float>((wei_pack8 >> (k << 2)) & 0xF);
                sum[k] += x[j] * ((wei_val - zero[k]) * scale[k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum[i] = block_reduce_sum<WARP_NUM>(sum[i]);
    }

    if (threadIdx.x == 0) {
        if (out_base < dim) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                query[out_base + i] = __float2bfloat16(sum[i]);
            }
        } else if (out_base < dim + kv_dim) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                key[out_base + i - dim] = __float2bfloat16(sum[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                value[out_base + i - dim - kv_dim] = __float2bfloat16(sum[i]);
            }
        }
    }
}

template <int32_t BLOCK_DIM>
static __global__ __launch_bounds__(BLOCK_DIM) void fused_gate_up_gemv_swiglu_int4x8_bf16x8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const int32_t* __restrict__ wei,
    const int32_t* __restrict__ zeros,
    const half* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    int32_t group_size,
    int32_t immediate_dim,
    int32_t K
) {
    constexpr int32_t WARP_NUM = (BLOCK_DIM >> 5);
    const int32_t n_pack_id = blockIdx.x;
    const int32_t out_base = n_pack_id << 3;
    const int32_t group_num = K / group_size;

    const int32_t* __restrict__ gate_wei_row = wei + n_pack_id * K;
    const int32_t* __restrict__ gate_zero_row = zeros + n_pack_id * group_num;

    const int32_t* __restrict__ up_wei_row = gate_wei_row + (immediate_dim >> 3) * K;
    const int32_t* __restrict__ up_zero_row = gate_zero_row + (immediate_dim >> 3) * group_num;

    float gate[8] = {0.0f};
    float up[8] = {0.0f};

    const int32_t K8 = K >> 3;
    const uint4* __restrict__ in8 = reinterpret_cast<const uint4*>(in);

    for (int32_t i = threadIdx.x; i < K8; i += blockDim.x) {
        const int32_t k_base = i << 3;
        const int32_t group_id = k_base / group_size;

        const uint4 a = in8[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);

        float x[8] = {
            __bfloat162float(a2[0].x), __bfloat162float(a2[0].y),
            __bfloat162float(a2[1].x), __bfloat162float(a2[1].y),
            __bfloat162float(a2[2].x), __bfloat162float(a2[2].y),
            __bfloat162float(a2[3].x), __bfloat162float(a2[3].y)
        };

        const uint32_t gate_zero_pack8 = static_cast<uint32_t>(gate_zero_row[group_id]);
        const uint32_t up_zero_pack8 = static_cast<uint32_t>(up_zero_row[group_id]);

        float gate_zero[8];
        float up_zero[8];

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            gate_zero[j] = static_cast<float>((gate_zero_pack8 >> (j << 2)) & 0xF);
            up_zero[j] = static_cast<float>((up_zero_pack8 >> (j << 2)) & 0xF);
        }

        float gate_scale[8];
        float up_scale[8];

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            gate_scale[j] = __half2float(scales[(out_base + j) * group_num + group_id]);
            up_scale[j] = __half2float(scales[(out_base + immediate_dim + j) * group_num + group_id]);
        }

        const int32_t* gate_wei_ptr = gate_wei_row + k_base;
        const int32_t* up_wei_ptr = up_wei_row + k_base;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t gate_wei_pack8 = static_cast<uint32_t>(gate_wei_ptr[j]);
            uint32_t up_wei_pack8 = static_cast<uint32_t>(up_wei_ptr[j]);

#pragma unroll
            for (int k = 0; k < 8; ++k) {
                gate[k] += x[j] * ((static_cast<float>((gate_wei_pack8 >> (k << 2)) & 0xF) - gate_zero[k]) * gate_scale[k]);
                up[k] += x[j] * ((static_cast<float>((up_wei_pack8 >> (k << 2)) & 0xF) - up_zero[k]) * up_scale[k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        gate[i] = block_reduce_sum<WARP_NUM>(gate[i]);
        up[i] = block_reduce_sum<WARP_NUM>(up[i]);
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float gate_silu = gate[i] / (1.0f + __expf(-gate[i]));
            out[out_base + i] = __float2bfloat16(gate_silu * up[i]);
        }
    }
}

static void check_cuda_bf16(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

static void check_cuda_i32(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32");
}

static void check_cuda_f16(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be float16");
}

torch::Tensor fused_gemv_add_int4(
    torch::Tensor in,
    torch::Tensor wei,
    torch::Tensor zeros,
    torch::Tensor scales,
    torch::Tensor residual,
    int64_t group_size
) {
    check_cuda_bf16(in, "in");
    check_cuda_i32(wei, "wei");
    check_cuda_i32(zeros, "zeros");
    check_cuda_f16(scales, "scales");
    check_cuda_bf16(residual, "residual");

    const int32_t N_pack = static_cast<int32_t>(wei.size(0));
    const int32_t K = static_cast<int32_t>(wei.size(1));
    const int32_t group_num = K / static_cast<int32_t>(group_size);

    TORCH_CHECK(in.numel() == K, "in length must equal K");
    TORCH_CHECK(residual.numel() == 8LL * N_pack, "residual length must equal 8 * wei.size(0)");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(K % 8 == 0, "K must be divisible by 8");
    TORCH_CHECK(zeros.numel() == static_cast<int64_t>(N_pack) * group_num, "zeros shape mismatch");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(8 * N_pack) * group_num, "scales shape mismatch");

    auto out = residual.clone();

    const auto* in_ptr = reinterpret_cast<const __nv_bfloat16*>(in.data_ptr<c10::BFloat16>());
    const auto* wei_ptr = wei.data_ptr<int32_t>();
    const auto* zero_ptr = zeros.data_ptr<int32_t>();
    const auto* scale_ptr = reinterpret_cast<const half*>(scales.data_ptr<c10::Half>());
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_gemv_add_int4x8_bf16x8_kernel<256>
        <<<N_pack, 256, 0, stream>>>(
            in_ptr, wei_ptr, zero_ptr, scale_ptr, out_ptr,
            static_cast<int32_t>(group_size), K
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> fused_qkv_int4(
    torch::Tensor in,
    torch::Tensor wei,
    torch::Tensor zeros,
    torch::Tensor scales,
    int64_t group_size,
    int64_t dim,
    int64_t kv_dim
) {
    check_cuda_bf16(in, "in");
    check_cuda_i32(wei, "wei");
    check_cuda_i32(zeros, "zeros");
    check_cuda_f16(scales, "scales");

    const int32_t N_pack = static_cast<int32_t>(wei.size(0));
    const int32_t K = static_cast<int32_t>(wei.size(1));
    const int32_t group_num = K / static_cast<int32_t>(group_size);

    TORCH_CHECK(in.numel() == K, "in length must equal K");
    TORCH_CHECK(dim % 8 == 0, "dim must be divisible by 8");
    TORCH_CHECK(kv_dim % 8 == 0, "kv_dim must be divisible by 8");
    TORCH_CHECK(dim + 2 * kv_dim == 8LL * N_pack, "dim + 2 * kv_dim must equal 8 * N_pack");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(K % 8 == 0, "K must be divisible by 8");
    TORCH_CHECK(zeros.numel() == static_cast<int64_t>(N_pack) * group_num, "zeros shape mismatch");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(8 * N_pack) * group_num, "scales shape mismatch");

    auto q = torch::empty({dim}, in.options());
    auto k = torch::empty({kv_dim}, in.options());
    auto v = torch::empty({kv_dim}, in.options());

    const auto* in_ptr = reinterpret_cast<const __nv_bfloat16*>(in.data_ptr<c10::BFloat16>());
    const auto* wei_ptr = wei.data_ptr<int32_t>();
    const auto* zero_ptr = zeros.data_ptr<int32_t>();
    const auto* scale_ptr = reinterpret_cast<const half*>(scales.data_ptr<c10::Half>());
    auto* q_ptr = reinterpret_cast<__nv_bfloat16*>(q.data_ptr<c10::BFloat16>());
    auto* k_ptr = reinterpret_cast<__nv_bfloat16*>(k.data_ptr<c10::BFloat16>());
    auto* v_ptr = reinterpret_cast<__nv_bfloat16*>(v.data_ptr<c10::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_qkv_gemv_int4x8_bf16x8_kernel<256>
        <<<N_pack, 256, 0, stream>>>(
            in_ptr, wei_ptr, zero_ptr, scale_ptr, q_ptr, k_ptr, v_ptr,
            static_cast<int32_t>(group_size), K,
            static_cast<int32_t>(dim), static_cast<int32_t>(kv_dim)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, k, v};
}

torch::Tensor fused_gate_up_swiglu_int4(
    torch::Tensor in,
    torch::Tensor wei,
    torch::Tensor zeros,
    torch::Tensor scales,
    int64_t group_size,
    int64_t immediate_dim
) {
    check_cuda_bf16(in, "in");
    check_cuda_i32(wei, "wei");
    check_cuda_i32(zeros, "zeros");
    check_cuda_f16(scales, "scales");

    const int32_t N_pack = static_cast<int32_t>(wei.size(0));
    const int32_t K = static_cast<int32_t>(wei.size(1));
    const int32_t group_num = K / static_cast<int32_t>(group_size);

    TORCH_CHECK(in.numel() == K, "in length must equal K");
    TORCH_CHECK(immediate_dim % 8 == 0, "immediate_dim must be divisible by 8");
    TORCH_CHECK(2 * immediate_dim == 8LL * N_pack, "2 * immediate_dim must equal 8 * N_pack");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(K % 8 == 0, "K must be divisible by 8");
    TORCH_CHECK(zeros.numel() == static_cast<int64_t>(N_pack) * group_num, "zeros shape mismatch");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(8 * N_pack) * group_num, "scales shape mismatch");

    auto out = torch::empty({immediate_dim}, in.options());

    const auto* in_ptr = reinterpret_cast<const __nv_bfloat16*>(in.data_ptr<c10::BFloat16>());
    const auto* wei_ptr = wei.data_ptr<int32_t>();
    const auto* zero_ptr = zeros.data_ptr<int32_t>();
    const auto* scale_ptr = reinterpret_cast<const half*>(scales.data_ptr<c10::Half>());
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<c10::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_gate_up_gemv_swiglu_int4x8_bf16x8_kernel<256>
        <<<static_cast<int32_t>(immediate_dim / 8), 256, 0, stream>>>(
            in_ptr, wei_ptr, zero_ptr, scale_ptr, out_ptr,
            static_cast<int32_t>(group_size),
            static_cast<int32_t>(immediate_dim),
            K
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemv_add_int4", &fused_gemv_add_int4);
    m.def("fused_qkv_int4", &fused_qkv_int4);
    m.def("fused_gate_up_swiglu_int4", &fused_gate_up_swiglu_int4);
}
'''


def build_ext(verbose: bool = False):
    return load_inline(
        name="qwen3_4b_awq_int4_accuracy_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=verbose,
    )


# -----------------------------
# Packing helper matching your kernel layout
# -----------------------------
def quant_pack_int4x8_cpu(weight_bf16_as_fp32: torch.Tensor, group_size: int):
    """
    Per-output-row, per-group asymmetric int4 quantization.

    Input:
      weight_bf16_as_fp32: CPU float32, shape [M, K]. This should already represent
                           BF16 weights, i.e. original_weight.to(bfloat16).float().
    Output layout consumed by the custom kernels:
      wei:    int32 [M/8, K], each int32 packs 8 output rows at the same K column.
      zeros:  int32 [M/8, K/group_size], each int32 packs 8 zero-points.
      scales: fp16  [M, K/group_size] flattened row-major.
      W_deq:  fp32  [M, K], dequantized weight using fp16 scales.
    """
    assert weight_bf16_as_fp32.device.type == "cpu"
    assert weight_bf16_as_fp32.dtype == torch.float32

    M, K = weight_bf16_as_fp32.shape
    assert M % 8 == 0, f"M={M} must be divisible by 8"
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    G = K // group_size
    Wg = weight_bf16_as_fp32.reshape(M, G, group_size)

    w_min = Wg.amin(dim=2)
    w_max = Wg.amax(dim=2)

    scale = (w_max - w_min) / 15.0
    scale = torch.clamp(scale, min=1e-8)

    # Kernel stores scale as fp16, so quantize scale to fp16 before computing q/zp.
    scale_h = scale.to(torch.float16)
    scale_f = scale_h.float().clamp_min(torch.finfo(torch.float16).tiny)

    zp = torch.round(-w_min / scale_f).clamp(0, 15).to(torch.int32)

    q = torch.round(Wg / scale_f[..., None] + zp.float()[..., None])
    q = q.clamp(0, 15).to(torch.int32)  # [M, G, group_size]
    q2 = q.reshape(M, K)

    W_deq = ((q.float() - zp.float()[..., None]) * scale_f[..., None]).reshape(M, K).contiguous()

    M_pack = M // 8
    wei_pack = torch.zeros((M_pack, K), dtype=torch.int32)
    for lane in range(8):
        wei_pack |= q2[lane::8, :] << (4 * lane)

    zero_pack = torch.zeros((M_pack, G), dtype=torch.int32)
    for lane in range(8):
        zero_pack |= zp[lane::8, :] << (4 * lane)

    scales_flat = scale_h.contiguous().reshape(-1)
    return wei_pack.contiguous(), zero_pack.contiguous(), scales_flat.contiguous(), W_deq


# -----------------------------
# Metrics
# -----------------------------
def calc_metrics(actual: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    a = actual.float().flatten().detach().cpu()
    r = ref.float().flatten().detach().cpu()
    diff = a - r
    ref_norm = torch.linalg.vector_norm(r).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    denom = r.abs().clamp_min(1e-6)
    cos = F.cosine_similarity(a.double(), r.double(), dim=0).item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(diff.float() ** 2)).item(),
        "rel_l2": diff_norm / max(ref_norm, 1e-12),
        "max_rel": (diff.abs() / denom).max().item(),
        "cosine": cos,
    }


def print_metrics(title: str, actual: torch.Tensor, ref: torch.Tensor):
    m = calc_metrics(actual, ref)
    print(
        f"{title:<48} "
        f"max_abs={m['max_abs']:.6e}  "
        f"mean_abs={m['mean_abs']:.6e}  "
        f"rmse={m['rmse']:.6e}  "
        f"rel_l2={m['rel_l2']:.6e}  "
        f"max_rel={m['max_rel']:.6e}  "
        f"cos={m['cosine']:.8f}"
    )
    return m


def make_bf16_weight_cpu(M: int, K: int, std: float) -> torch.Tensor:
    # Generate fp32, then round to BF16 and return fp32 values representing BF16 weights.
    W = torch.randn(M, K, dtype=torch.float32) * std
    return W.to(torch.bfloat16).float().contiguous()


def make_bf16_activation_cuda(K: int, std: float, device: str) -> torch.Tensor:
    return (torch.randn(K, device=device, dtype=torch.float32) * std).to(torch.bfloat16).contiguous()


def to_cuda_pack(wei, zeros, scales, W_deq, W, device):
    return (
        wei.to(device, non_blocking=True),
        zeros.to(device, non_blocking=True),
        scales.to(device, non_blocking=True),
        W_deq.to(device, non_blocking=True),
        W.to(device, non_blocking=True),
    )


@torch.no_grad()
def run_qkv_proj(ext, args):
    M = DIMS.qkv_dim
    K = DIMS.hidden_size
    print(f"\n[1] qkv_proj: W=[{M}, {K}], input=[{K}], q=[{DIMS.q_dim}], k/v=[{DIMS.kv_dim}]")

    W = make_bf16_weight_cpu(M, K, args.weight_std)
    wei, zeros, scales, W_deq = quant_pack_int4x8_cpu(W, DIMS.group_size)
    wei, zeros, scales, W_deq, W = to_cuda_pack(wei, zeros, scales, W_deq, W, args.device)

    impl_metrics = []
    quant_metrics = []
    kernel_full_metrics = []

    for t in range(args.trials):
        x = make_bf16_activation_cuda(K, args.activation_std, args.device)
        q, k, v = ext.fused_qkv_int4(x, wei, zeros, scales, DIMS.group_size, DIMS.q_dim, DIMS.kv_dim)
        out = torch.cat([q, k, v], dim=0)
        torch.cuda.synchronize()

        ref_deq = W_deq.matmul(x.float())
        ref_full = W.matmul(x.float())

        if t == 0:
            impl_metrics.append(print_metrics("impl: kernel vs dequant-ref(BF16 out)", out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(print_metrics("quant: dequant-ref vs original-bf16-ref", ref_deq, ref_full))
            kernel_full_metrics.append(print_metrics("total: kernel vs original-bf16-ref(BF16 out)", out, ref_full.to(torch.bfloat16)))
        else:
            impl_metrics.append(calc_metrics(out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(calc_metrics(ref_deq, ref_full))
            kernel_full_metrics.append(calc_metrics(out, ref_full.to(torch.bfloat16)))

    return impl_metrics, quant_metrics, kernel_full_metrics


@torch.no_grad()
def run_o_proj_add(ext, args):
    M = DIMS.hidden_size
    K = DIMS.q_dim
    print(f"\n[2] o_proj + add: W=[{M}, {K}], input=[{K}], residual/out=[{M}]")

    W = make_bf16_weight_cpu(M, K, args.weight_std)
    wei, zeros, scales, W_deq = quant_pack_int4x8_cpu(W, DIMS.group_size)
    wei, zeros, scales, W_deq, W = to_cuda_pack(wei, zeros, scales, W_deq, W, args.device)

    impl_metrics = []
    quant_metrics = []
    kernel_full_metrics = []

    for t in range(args.trials):
        x = make_bf16_activation_cuda(K, args.activation_std, args.device)
        residual = make_bf16_activation_cuda(M, args.residual_std, args.device)

        out = ext.fused_gemv_add_int4(x, wei, zeros, scales, residual, DIMS.group_size)
        torch.cuda.synchronize()

        ref_deq = residual.float() + W_deq.matmul(x.float())
        ref_full = residual.float() + W.matmul(x.float())

        if t == 0:
            impl_metrics.append(print_metrics("impl: kernel vs dequant-ref(BF16 out)", out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(print_metrics("quant: dequant-ref vs original-bf16-ref", ref_deq, ref_full))
            kernel_full_metrics.append(print_metrics("total: kernel vs original-bf16-ref(BF16 out)", out, ref_full.to(torch.bfloat16)))
        else:
            impl_metrics.append(calc_metrics(out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(calc_metrics(ref_deq, ref_full))
            kernel_full_metrics.append(calc_metrics(out, ref_full.to(torch.bfloat16)))

    return impl_metrics, quant_metrics, kernel_full_metrics


@torch.no_grad()
def run_gate_up_swiglu(ext, args):
    M = 2 * DIMS.intermediate_size
    K = DIMS.hidden_size
    I = DIMS.intermediate_size
    print(f"\n[3] gate_up_proj + SwiGLU: W=[{M}, {K}], input=[{K}], out=[{I}]")

    W = make_bf16_weight_cpu(M, K, args.weight_std)
    wei, zeros, scales, W_deq = quant_pack_int4x8_cpu(W, DIMS.group_size)
    wei, zeros, scales, W_deq, W = to_cuda_pack(wei, zeros, scales, W_deq, W, args.device)

    impl_metrics = []
    quant_metrics = []
    kernel_full_metrics = []

    for t in range(args.trials):
        x = make_bf16_activation_cuda(K, args.activation_std, args.device)

        out = ext.fused_gate_up_swiglu_int4(x, wei, zeros, scales, DIMS.group_size, I)
        torch.cuda.synchronize()

        gate_deq = W_deq[:I].matmul(x.float())
        up_deq = W_deq[I:].matmul(x.float())
        ref_deq = F.silu(gate_deq) * up_deq

        gate_full = W[:I].matmul(x.float())
        up_full = W[I:].matmul(x.float())
        ref_full = F.silu(gate_full) * up_full

        if t == 0:
            impl_metrics.append(print_metrics("impl: kernel vs dequant-ref(BF16 out)", out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(print_metrics("quant: dequant-ref vs original-bf16-ref", ref_deq, ref_full))
            kernel_full_metrics.append(print_metrics("total: kernel vs original-bf16-ref(BF16 out)", out, ref_full.to(torch.bfloat16)))
        else:
            impl_metrics.append(calc_metrics(out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(calc_metrics(ref_deq, ref_full))
            kernel_full_metrics.append(calc_metrics(out, ref_full.to(torch.bfloat16)))

    return impl_metrics, quant_metrics, kernel_full_metrics


@torch.no_grad()
def run_down_proj_add(ext, args):
    M = DIMS.hidden_size
    K = DIMS.intermediate_size
    print(f"\n[4] down_proj + add: W=[{M}, {K}], input=[{K}], residual/out=[{M}]")

    W = make_bf16_weight_cpu(M, K, args.weight_std)
    wei, zeros, scales, W_deq = quant_pack_int4x8_cpu(W, DIMS.group_size)
    wei, zeros, scales, W_deq, W = to_cuda_pack(wei, zeros, scales, W_deq, W, args.device)

    impl_metrics = []
    quant_metrics = []
    kernel_full_metrics = []

    for t in range(args.trials):
        x = make_bf16_activation_cuda(K, args.activation_std, args.device)
        residual = make_bf16_activation_cuda(M, args.residual_std, args.device)

        out = ext.fused_gemv_add_int4(x, wei, zeros, scales, residual, DIMS.group_size)
        torch.cuda.synchronize()

        ref_deq = residual.float() + W_deq.matmul(x.float())
        ref_full = residual.float() + W.matmul(x.float())

        if t == 0:
            impl_metrics.append(print_metrics("impl: kernel vs dequant-ref(BF16 out)", out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(print_metrics("quant: dequant-ref vs original-bf16-ref", ref_deq, ref_full))
            kernel_full_metrics.append(print_metrics("total: kernel vs original-bf16-ref(BF16 out)", out, ref_full.to(torch.bfloat16)))
        else:
            impl_metrics.append(calc_metrics(out, ref_deq.to(torch.bfloat16)))
            quant_metrics.append(calc_metrics(ref_deq, ref_full))
            kernel_full_metrics.append(calc_metrics(out, ref_full.to(torch.bfloat16)))

    return impl_metrics, quant_metrics, kernel_full_metrics


def summarize(name: str, metrics_list):
    keys = ["max_abs", "mean_abs", "rmse", "rel_l2", "max_rel", "cosine"]
    print(f"\nSummary: {name}")
    for key in keys:
        vals = torch.tensor([m[key] for m in metrics_list], dtype=torch.float64)
        print(f"  {key:<9} mean={vals.mean().item():.6e}  max={vals.max().item():.6e}  min={vals.min().item():.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3, help="Number of random activation trials per op")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--activation-std", type=float, default=0.5)
    parser.add_argument("--residual-std", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--assert-thresholds", action="store_true")
    parser.add_argument("--impl-rel-l2-threshold", type=float, default=5e-3)
    parser.add_argument("--quant-rel-l2-threshold", type=float, default=5e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Disable TF32 for cleaner references.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Qwen3-4B fixed dimensions:")
    print(f"  hidden_size         = {DIMS.hidden_size}")
    print(f"  head_dim            = {DIMS.head_dim}")
    print(f"  num_attention_heads = {DIMS.num_attention_heads}")
    print(f"  num_key_value_heads = {DIMS.num_key_value_heads}")
    print(f"  q_dim               = {DIMS.q_dim}")
    print(f"  kv_dim              = {DIMS.kv_dim}")
    print(f"  qkv_dim             = {DIMS.qkv_dim}")
    print(f"  intermediate_size   = {DIMS.intermediate_size}")
    print(f"  group_size          = {DIMS.group_size}")

    ext = build_ext(verbose=args.compile_verbose)

    all_impl = []
    all_quant = []
    all_total = []

    for fn in [run_qkv_proj, run_o_proj_add, run_gate_up_swiglu, run_down_proj_add]:
        impl, quant, total = fn(ext, args)
        all_impl.extend(impl)
        all_quant.extend(quant)
        all_total.extend(total)

    summarize("implementation check: kernel vs dequant-ref(BF16 out)", all_impl)
    summarize("pure quantization loss: dequant-ref vs original-bf16-ref", all_quant)
    summarize("end-to-end: kernel vs original-bf16-ref(BF16 out)", all_total)

    if args.assert_thresholds:
        worst_impl = max(m["rel_l2"] for m in all_impl)
        worst_quant = max(m["rel_l2"] for m in all_quant)
        if worst_impl > args.impl_rel_l2_threshold:
            raise AssertionError(
                f"implementation rel_l2 too high: {worst_impl:.6e} > {args.impl_rel_l2_threshold:.6e}. "
                f"Likely packing/layout/kernel bug."
            )
        if worst_quant > args.quant_rel_l2_threshold:
            raise AssertionError(
                f"quantization rel_l2 too high: {worst_quant:.6e} > {args.quant_rel_l2_threshold:.6e}. "
                f"Quantization loss may be too large for your chosen threshold."
            )


if __name__ == "__main__":
    main()