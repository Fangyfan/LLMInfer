#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-4B-AWQ HuggingFace/AutoAWQ -> custom .bin exporter.

This version explicitly handles AutoAWQ INT4 nibble order on the Python
export side, so the C++ kernel/loader only needs to unpack every int32 as:

    v0 = (x >>  0) & 0xF
    v1 = (x >>  4) & 0xF
    v2 = (x >>  8) & 0xF
    v3 = (x >> 12) & 0xF
    v4 = (x >> 16) & 0xF
    v5 = (x >> 20) & 0xF
    v6 = (x >> 24) & 0xF
    v7 = (x >> 28) & 0xF

No AWQ_REVERSE_ORDER / interleaved nibble handling is required in C++.

Header:
    int32[8]
    [
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq_len,
        intermediate_size,
    ]

Payload:
    Non-quantized tensors are written as raw BF16 bits.

    Quantized Linear tensors are written as:
        qweight int32 payload
        qzeros  int32 payload
        scales  fp16  payload

    For each logical Linear W[N, K], where:
        N = out_features
        K = in_features
        G = group_size, normally 128

    HF/AutoAWQ GEMM/N-packed layout is assumed to be:
        qweight: [K,   N / 8]      int32, 8 int4 values packed along N
        qzeros : [K/G, N / 8]      int32, 8 int4 values packed along N
        scales : [K/G, N]          fp16/bf16/fp32

    AutoAWQ physical nibble order inside each int32 is assumed to be:
        AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    Therefore, when reading HF/AutoAWQ qweight/qzeros, this exporter maps:

        logical nibble 0 <- physical nibble 0
        logical nibble 1 <- physical nibble 4
        logical nibble 2 <- physical nibble 1
        logical nibble 3 <- physical nibble 5
        logical nibble 4 <- physical nibble 2
        logical nibble 5 <- physical nibble 6
        logical nibble 6 <- physical nibble 3
        logical nibble 7 <- physical nibble 7

    Then this converter rewrites to the current C++ loader layout:
        qweight: [N / 8, K]        int32, 8 int4 values packed along N/output dim
        qzeros : [N / 8, K / G]    int32, 8 int4 values packed along N/output dim
        scales : [N,     K / G]    fp16

    The exported qweight/qzeros are packed in plain 01234567 nibble order.

Important:
    This is NOT a simple int32 matrix transpose. INT4 payloads are unpacked,
    reordered from AutoAWQ physical nibble order into logical 01234567 order,
    transposed at nibble level, then repacked in plain 01234567 order.

Expected HF tensor names for quantized Linear layers:
    xxx.qweight
    xxx.qzeros
    xxx.scales

Expected HF tensor names for non-quantized layers:
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.q_norm.weight
    model.layers.{i}.self_attn.k_norm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.norm.weight

Written order:
    1. Embedding BF16
    2. all input_layernorm BF16
    3. per-layer fused QKV AWQ blob: q/k/v concatenated along output dim N
    4. per-layer q_norm BF16 + k_norm BF16
    5. all o_proj AWQ blobs
    6. all post_attention_layernorm BF16
    7. per-layer fused gate/up AWQ blob: gate/up concatenated along output dim N
    8. all down_proj AWQ blobs
    9. final norm BF16

LM head is intentionally not written. The C++ side ties lm_head to embedding.

Examples:
    python qwen3_awq_converter_awq_reverse_handled.py \
        --model_dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ \
        --out_file /home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_awq.bin

Install:
    pip install torch safetensors tqdm
"""

import argparse
import gc
import json
import struct
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError as e:
    raise ImportError("Please install safetensors first: pip install safetensors") from e


# ============================================================
# default paths
# ============================================================

DEFAULT_QWEN3_4B_AWQ_MODEL = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ"
DEFAULT_OUT_FILE = "/home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_awq_int4_bf16.bin"


# ============================================================
# AWQ INT4 nibble order
# ============================================================

# AutoAWQ's physical nibble order inside each int32.
#
# Meaning used here:
#   logical_values[..., j] = physical_nibbles[..., AWQ_REVERSE_ORDER[j]]
#
# After this Python-side conversion, exported qweight/qzeros are repacked in
# normal physical order [0,1,2,3,4,5,6,7], so the C++ side does not need this
# table anymore.
AWQ_REVERSE_ORDER: Tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)
PLAIN_INT4_ORDER: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


# ============================================================
# low-level serialization
# ============================================================


def write_bf16(file_obj, tensor: torch.Tensor) -> int:
    """Write tensor as raw little-endian BF16 bits. Return bytes written."""
    bf16_tensor = tensor.detach().cpu().contiguous().view(-1).to(torch.bfloat16)
    uint16_tensor = bf16_tensor.view(torch.uint16)
    arr = uint16_tensor.numpy()
    data = arr.tobytes()
    file_obj.write(data)
    return len(data)


def write_fp16(file_obj, tensor: torch.Tensor) -> int:
    """Write tensor as raw little-endian FP16 bits. Return bytes written."""
    fp16_tensor = tensor.detach().cpu().contiguous().view(-1).to(torch.float16)
    arr = fp16_tensor.numpy()
    data = arr.tobytes()
    file_obj.write(data)
    return len(data)


def write_int32(file_obj, tensor: torch.Tensor) -> int:
    """Write tensor as raw little-endian int32. Return bytes written."""
    int32_tensor = tensor.detach().cpu().contiguous().view(-1).to(torch.int32)
    arr = int32_tensor.numpy()
    data = arr.tobytes()
    file_obj.write(data)
    return len(data)


# ============================================================
# safetensors lazy loader
# ============================================================


class ShardedTensorLoader:
    """
    Lazy loader for HuggingFace safetensors.

    Supports:
      - model.safetensors
      - model.safetensors.index.json + model-xxxxx-of-yyyyy.safetensors
      - model-*.safetensors without index, by scanning tensor names
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.weight_map: Dict[str, str] = {}
        self.current_shard_name: Optional[str] = None
        self.current_tensors: Optional[Dict[str, torch.Tensor]] = None

        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index_json = json.load(f)
            self.weight_map = index_json["weight_map"]
            shard_names = sorted(set(self.weight_map.values()))
            print("Detected indexed sharded checkpoint:")
            for shard_name in shard_names:
                print("   ", shard_name)

        elif single_path.exists():
            tensors = safe_load_file(str(single_path), device="cpu")
            self.weight_map = {name: single_path.name for name in tensors.keys()}
            del tensors
            gc.collect()
            print("Detected single safetensors checkpoint:")
            print("   ", single_path.name)

        else:
            shard_paths = sorted(self.model_dir.glob("model-*.safetensors"))
            if not shard_paths:
                raise FileNotFoundError(
                    "Cannot find model.safetensors.index.json, "
                    "model.safetensors, or model-*.safetensors"
                )

            print("Detected safetensors shard files without index:")
            for shard_path in shard_paths:
                print("   ", shard_path.name)

            print("Scanning shard tensor names...")
            for shard_path in shard_paths:
                tensors = safe_load_file(str(shard_path), device="cpu")
                for tensor_name in tensors.keys():
                    if tensor_name in self.weight_map:
                        raise RuntimeError(f"Duplicate tensor name found: {tensor_name}")
                    self.weight_map[tensor_name] = shard_path.name
                del tensors
                gc.collect()

    def _switch_shard(self, shard_name: str) -> None:
        if shard_name == self.current_shard_name:
            return

        self.current_tensors = None
        gc.collect()

        shard_path = self.model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")

        print(f"Loading shard: {shard_name}")
        self.current_tensors = safe_load_file(str(shard_path), device="cpu")
        self.current_shard_name = shard_name

    def get(self, tensor_name: str) -> torch.Tensor:
        if tensor_name not in self.weight_map:
            raise KeyError(f"Tensor not found: {tensor_name}")

        shard_name = self.weight_map[tensor_name]
        self._switch_shard(shard_name)
        assert self.current_tensors is not None

        if tensor_name not in self.current_tensors:
            raise KeyError(f"Tensor {tensor_name} missing inside shard {shard_name}")

        return self.current_tensors[tensor_name]

    def has(self, tensor_name: str) -> bool:
        return tensor_name in self.weight_map


# ============================================================
# INT4 pack / unpack helpers
# ============================================================


def _as_unsigned_int64(x: torch.Tensor) -> torch.Tensor:
    """
    Interpret int32 storage as unsigned 32-bit values, represented in int64.

    This avoids sign-extension issues when shifting packed int32 values whose
    highest bit is set.
    """
    return x.to(torch.int64) & 0xFFFFFFFF


def _validate_int4_order(order: Sequence[int]) -> Tuple[int, ...]:
    order = tuple(int(x) for x in order)
    if len(order) != 8 or sorted(order) != list(range(8)):
        raise ValueError(f"INT4 order must be a permutation of 0..7, got {order}")
    return order


def unpack_int4_along_last_dim(
    packed: torch.Tensor,
    unpacked_last_dim: int,
    source_order: Sequence[int] = AWQ_REVERSE_ORDER,
) -> torch.Tensor:
    """
    Unpack int32 tensor whose last dimension packs groups of 8 uint4 values.

    Input shape:
        [..., unpacked_last_dim / 8]

    Output shape:
        [..., unpacked_last_dim]

    Physical nibble convention:
        physical nibble 0 is bits [3:0],
        physical nibble 1 is bits [7:4],
        ...
        physical nibble 7 is bits [31:28].

    source_order controls how physical nibbles become logical values:

        logical[..., j] = physical[..., source_order[j]]

    For HF/AutoAWQ tensors, source_order should be AWQ_REVERSE_ORDER:
        [0, 4, 1, 5, 2, 6, 3, 7]

    This explicitly removes AutoAWQ's interleaved nibble order on the Python
    side. Downstream C++ can then use plain 01234567 unpacking.
    """
    source_order = _validate_int4_order(source_order)

    if unpacked_last_dim % 8 != 0:
        raise ValueError(f"unpacked_last_dim must be divisible by 8, got {unpacked_last_dim}")

    packed = packed.detach().cpu().contiguous()
    expected_packed_last = unpacked_last_dim // 8
    if packed.shape[-1] != expected_packed_last:
        raise RuntimeError(
            f"Packed last dim mismatch: got {packed.shape[-1]}, "
            f"expected {expected_packed_last} for unpacked_last_dim={unpacked_last_dim}"
        )

    u = _as_unsigned_int64(packed)

    # physical[..., i] is the uint4 stored in physical nibble i.
    physical = [((u >> (4 * i)) & 0xF).to(torch.uint8) for i in range(8)]

    # Convert physical AutoAWQ nibble order to logical 01234567 order.
    logical = [physical[source_order[j]] for j in range(8)]

    stacked = torch.stack(logical, dim=-1)  # [..., packed_last, 8]
    return stacked.reshape(*packed.shape[:-1], unpacked_last_dim).contiguous()


def pack_int4_along_first_dim(
    unpacked: torch.Tensor,
    target_order: Sequence[int] = PLAIN_INT4_ORDER,
) -> torch.Tensor:
    """
    Pack a uint4 matrix [N, K] into int32 matrix [N/8, K].

    By default this writes plain 01234567 physical nibble order:

        physical nibble 0 <- logical value 0
        physical nibble 1 <- logical value 1
        ...
        physical nibble 7 <- logical value 7

    For output row r = n // 8 and default target_order=[0,1,2,3,4,5,6,7]:

        int32[r, k] = sum_{i=0..7} unpacked[8*r+i, k] << (4*i)

    target_order controls where each logical value is physically written:

        physical nibble target_order[j] <- logical value j

    Keep target_order as PLAIN_INT4_ORDER to make the exported file simple for
    C++ kernels that unpack int4x8 in 01234567 order.
    """
    target_order = _validate_int4_order(target_order)

    if unpacked.dim() != 2:
        raise ValueError(f"Expected rank-2 tensor, got shape {tuple(unpacked.shape)}")

    n, k = unpacked.shape
    if n % 8 != 0:
        raise ValueError(f"First dimension must be divisible by 8, got {n}")

    x = unpacked.detach().cpu().contiguous().to(torch.int64)
    if torch.any((x < 0) | (x > 15)):
        raise RuntimeError("INT4 values must be in [0, 15]")

    x = x.reshape(n // 8, 8, k)
    packed = torch.zeros((n // 8, k), dtype=torch.int64)

    for logical_i in range(8):
        physical_i = target_order[logical_i]
        packed |= (x[:, logical_i, :] & 0xF) << (4 * physical_i)

    return packed.to(torch.int32).contiguous()


# ============================================================
# AWQ tensor conversion
# ============================================================


def awq_prefix(base: str) -> Tuple[str, str, str]:
    return base + ".qweight", base + ".qzeros", base + ".scales"


def validate_awq_shapes(
    base: str,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    n_out: int,
    k_in: int,
    group_size: int,
) -> None:
    if k_in % group_size != 0:
        raise RuntimeError(f"{base}: K={k_in} is not divisible by group_size={group_size}")
    if n_out % 8 != 0:
        raise RuntimeError(f"{base}: N={n_out} is not divisible by 8")

    expected_qweight = (k_in, n_out // 8)
    expected_qzeros = (k_in // group_size, n_out // 8)
    expected_scales = (k_in // group_size, n_out)

    if tuple(qweight.shape) != expected_qweight:
        raise RuntimeError(
            f"{base}.qweight shape mismatch\n"
            f"got      : {tuple(qweight.shape)}\n"
            f"expected : {expected_qweight}"
        )
    if tuple(qzeros.shape) != expected_qzeros:
        raise RuntimeError(
            f"{base}.qzeros shape mismatch\n"
            f"got      : {tuple(qzeros.shape)}\n"
            f"expected : {expected_qzeros}"
        )
    if tuple(scales.shape) != expected_scales:
        raise RuntimeError(
            f"{base}.scales shape mismatch\n"
            f"got      : {tuple(scales.shape)}\n"
            f"expected : {expected_scales}"
        )


def convert_one_awq_linear_to_cpp_layout(
    loader: ShardedTensorLoader,
    base: str,
    n_out: int,
    k_in: int,
    group_size: int,
    no_shape_check: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert one HF AutoAWQ Linear from:
        qweight [K, N/8], qzeros [K/G, N/8], scales [K/G, N]

    to C++ layout:
        qweight [N/8, K], qzeros [N/8, K/G], scales [N, K/G]

    Important for qweight/qzeros:
        1. Unpack HF/AutoAWQ physical nibble order using AWQ_REVERSE_ORDER.
        2. Get logical uint4 matrix in normal 01234567 order.
        3. Transpose at nibble level.
        4. Repack in plain 01234567 order for simple C++ kernels.
    """
    qweight_name, qzeros_name, scales_name = awq_prefix(base)

    qweight = loader.get(qweight_name)
    qzeros = loader.get(qzeros_name)
    scales = loader.get(scales_name)

    if not no_shape_check:
        validate_awq_shapes(base, qweight, qzeros, scales, n_out, k_in, group_size)

    # qweight:
    #   HF physical [K, N/8] int32 with AutoAWQ nibble order
    #   -> logical [K, N] uint4 in 01234567 order
    #   -> logical [N, K]
    #   -> exported [N/8, K] int32 in plain 01234567 order
    qweight_u4_k_n = unpack_int4_along_last_dim(
        qweight,
        n_out,
        source_order=AWQ_REVERSE_ORDER,
    )
    qweight_u4_n_k = qweight_u4_k_n.transpose(0, 1).contiguous()
    cpp_qweight = pack_int4_along_first_dim(
        qweight_u4_n_k,
        target_order=PLAIN_INT4_ORDER,
    )

    del qweight_u4_k_n, qweight_u4_n_k
    gc.collect()

    # qzeros:
    #   HF physical [K/G, N/8] int32 with AutoAWQ nibble order
    #   -> logical [K/G, N] uint4 in 01234567 order
    #   -> logical [N, K/G]
    #   -> exported [N/8, K/G] int32 in plain 01234567 order
    qzeros_u4_kg_n = unpack_int4_along_last_dim(
        qzeros,
        n_out,
        source_order=AWQ_REVERSE_ORDER,
    )
    qzeros_u4_n_kg = qzeros_u4_kg_n.transpose(0, 1).contiguous()
    cpp_qzeros = pack_int4_along_first_dim(
        qzeros_u4_n_kg,
        target_order=PLAIN_INT4_ORDER,
    )

    del qzeros_u4_kg_n, qzeros_u4_n_kg
    gc.collect()

    # scales: [K/G, N] -> [N, K/G], stored as fp16
    cpp_scales = scales.detach().cpu().transpose(0, 1).contiguous().to(torch.float16)

    return cpp_qweight, cpp_qzeros, cpp_scales


def concat_awq_linears_along_output(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate converted AWQ linears along logical output dimension N.

    In C++ layout:
        qweight [N/8, K]      -> concat dim 0
        qzeros  [N/8, K/G]    -> concat dim 0
        scales  [N,   K/G]    -> concat dim 0
    """
    qweights = [p[0] for p in parts]
    qzeros = [p[1] for p in parts]
    scales = [p[2] for p in parts]
    return (
        torch.cat(qweights, dim=0).contiguous(),
        torch.cat(qzeros, dim=0).contiguous(),
        torch.cat(scales, dim=0).contiguous(),
    )


def write_awq_blob(file_obj, blob: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> int:
    qweight, qzeros, scales = blob
    written = 0
    written += write_int32(file_obj, qweight)
    written += write_int32(file_obj, qzeros)
    written += write_fp16(file_obj, scales)
    return written


# ============================================================
# shape helpers for Qwen3 dense metadata
# ============================================================


def get_qwen3_dims(cfg: dict) -> Dict[str, int]:
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    vocab_size = int(cfg["vocab_size"])
    max_seq_len = int(cfg["max_position_embeddings"])
    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))

    return {
        "hidden_size": hidden_size,
        "hidden_dim": hidden_size,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "head_dim": head_dim,
        "dim": num_heads * head_dim,
        "kv_dim": num_kv_heads * head_dim,
    }


def expected_nonquant_shape(name: str, dims: Dict[str, int]) -> Optional[Tuple[int, ...]]:
    hidden = dims["hidden_size"]
    vocab = dims["vocab_size"]
    layers = dims["num_layers"]
    head_dim = dims["head_dim"]

    if name == "model.embed_tokens.weight":
        return (vocab, hidden)
    if name == "model.norm.weight":
        return (hidden,)

    for i in range(layers):
        prefix = f"model.layers.{i}."
        if name in (
            prefix + "input_layernorm.weight",
            prefix + "post_attention_layernorm.weight",
        ):
            return (hidden,)
        if name in (
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
        ):
            return (head_dim,)

    return None


def write_nonquant_bf16(
    file_obj,
    loader: ShardedTensorLoader,
    name: str,
    dims: Dict[str, int],
    no_shape_check: bool,
) -> int:
    tensor = loader.get(name)
    if not no_shape_check:
        expected = expected_nonquant_shape(name, dims)
        if expected is not None and tuple(tensor.shape) != expected:
            raise RuntimeError(
                f"Shape mismatch for {name}\n"
                f"got      : {tuple(tensor.shape)}\n"
                f"expected : {expected}"
            )
    return write_bf16(file_obj, tensor)


def awq_blob_sizes_bytes(n_out: int, k_in: int, group_size: int) -> int:
    """Expected bytes for C++ layout qweight + qzeros + scales."""
    if n_out % 8 != 0 or k_in % group_size != 0:
        raise ValueError(f"Invalid AWQ dims: N={n_out}, K={k_in}, G={group_size}")
    kg = k_in // group_size
    qweight_int32 = (n_out // 8) * k_in
    qzeros_int32 = (n_out // 8) * kg
    scales_fp16 = n_out * kg
    return qweight_int32 * 4 + qzeros_int32 * 4 + scales_fp16 * 2


def expected_file_size_bytes(dims: Dict[str, int], group_size: int) -> int:
    hidden = dims["hidden_size"]
    layers = dims["num_layers"]
    vocab = dims["vocab_size"]
    head_dim = dims["head_dim"]
    dim = dims["dim"]
    kv_dim = dims["kv_dim"]
    intermediate = dims["intermediate_size"]

    total = 8 * 4

    # embedding
    total += vocab * hidden * 2

    # input_layernorm, all layers
    total += layers * hidden * 2

    # fused qkv: N = dim + 2 * kv_dim, K = hidden
    total += layers * awq_blob_sizes_bytes(dim + 2 * kv_dim, hidden, group_size)

    # q_norm + k_norm
    total += layers * 2 * head_dim * 2

    # o_proj: N = hidden, K = dim
    total += layers * awq_blob_sizes_bytes(hidden, dim, group_size)

    # post_attention_layernorm
    total += layers * hidden * 2

    # fused gate/up: N = 2 * intermediate, K = hidden
    total += layers * awq_blob_sizes_bytes(2 * intermediate, hidden, group_size)

    # down_proj: N = hidden, K = intermediate
    total += layers * awq_blob_sizes_bytes(hidden, intermediate, group_size)

    # final norm
    total += hidden * 2

    return total


# ============================================================
# main export
# ============================================================


def infer_group_size(cfg: dict, cli_group_size: Optional[int]) -> int:
    if cli_group_size is not None:
        return int(cli_group_size)

    qcfg = cfg.get("quantization_config", {}) or {}
    for key in ("group_size", "q_group_size"):
        if key in qcfg:
            return int(qcfg[key])

    # Some configs nest under weights.
    weights_cfg = qcfg.get("weights", {}) or {}
    if "group_size" in weights_cfg:
        return int(weights_cfg["group_size"])

    return 128


def export_qwen3_awq(
    model_dir: str,
    out_file: str,
    group_size: Optional[int] = None,
    no_shape_check: bool = False,
) -> None:
    model_dir_path = Path(model_dir)
    config_path = model_dir_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dims = get_qwen3_dims(cfg)
    group_size = infer_group_size(cfg, group_size)

    if group_size != 128:
        print(
            f"WARNING: group_size={group_size}. Your C++ code was discussed with group_size=128. "
            "Use this only if the C++ side also matches."
        )

    hidden = dims["hidden_size"]
    layers = dims["num_layers"]
    dim = dims["dim"]
    kv_dim = dims["kv_dim"]
    intermediate = dims["intermediate_size"]

    print("=" * 70)
    print("Model config")
    print("=" * 70)
    for k in [
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "dim",
        "kv_dim",
        "vocab_size",
        "max_seq_len",
    ]:
        print(f"{k:18s}: {dims[k]}")
    print(f"group_size        : {group_size}")
    print("nonquant dtype    : bf16")
    print("quant payload     : qweight int32 + qzeros int32 + scales fp16")
    print(f"HF AWQ nibble map : {list(AWQ_REVERSE_ORDER)} -> exported plain {list(PLAIN_INT4_ORDER)}")

    qcfg = cfg.get("quantization_config", {}) or {}
    if qcfg:
        print("quantization_config:")
        print(json.dumps(qcfg, indent=2, ensure_ascii=False))

    loader = ShardedTensorLoader(str(model_dir_path))

    expected_size = expected_file_size_bytes(dims, group_size)

    print("=" * 70)
    print("Export layout")
    print("=" * 70)
    print("write_lm_head     : False")
    print("cpp int4 unpack   : plain 01234567, no AWQ_REVERSE_ORDER required")
    print(f"expected_size     : {expected_size} bytes")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0

    with open(out_path, "wb") as f:
        header = struct.pack(
            "iiiiiiii",
            dims["dim"],
            dims["hidden_dim"],
            dims["num_layers"],
            dims["num_heads"],
            dims["num_kv_heads"],
            dims["vocab_size"],
            dims["max_seq_len"],
            dims["intermediate_size"],
        )
        f.write(header)
        bytes_written += len(header)

        print("=" * 70)
        print("Writing non-quantized embedding")
        print("=" * 70)
        bytes_written += write_nonquant_bf16(
            f, loader, "model.embed_tokens.weight", dims, no_shape_check
        )

        print("=" * 70)
        print("Writing input_layernorm weights")
        print("=" * 70)
        for i in tqdm(range(layers), desc="input_layernorm"):
            name = f"model.layers.{i}.input_layernorm.weight"
            bytes_written += write_nonquant_bf16(f, loader, name, dims, no_shape_check)

        print("=" * 70)
        print("Writing fused QKV AWQ blobs")
        print("=" * 70)
        for i in tqdm(range(layers), desc="qkv"):
            prefix = f"model.layers.{i}.self_attn"
            q = convert_one_awq_linear_to_cpp_layout(
                loader, prefix + ".q_proj", dim, hidden, group_size, no_shape_check
            )
            k = convert_one_awq_linear_to_cpp_layout(
                loader, prefix + ".k_proj", kv_dim, hidden, group_size, no_shape_check
            )
            v = convert_one_awq_linear_to_cpp_layout(
                loader, prefix + ".v_proj", kv_dim, hidden, group_size, no_shape_check
            )
            qkv = concat_awq_linears_along_output([q, k, v])
            bytes_written += write_awq_blob(f, qkv)
            del q, k, v, qkv
            gc.collect()

        print("=" * 70)
        print("Writing q_norm / k_norm weights")
        print("=" * 70)
        for i in tqdm(range(layers), desc="qk_norm"):
            qn = f"model.layers.{i}.self_attn.q_norm.weight"
            kn = f"model.layers.{i}.self_attn.k_norm.weight"
            bytes_written += write_nonquant_bf16(f, loader, qn, dims, no_shape_check)
            bytes_written += write_nonquant_bf16(f, loader, kn, dims, no_shape_check)

        print("=" * 70)
        print("Writing O projection AWQ blobs")
        print("=" * 70)
        for i in tqdm(range(layers), desc="o_proj"):
            base = f"model.layers.{i}.self_attn.o_proj"
            blob = convert_one_awq_linear_to_cpp_layout(
                loader, base, hidden, dim, group_size, no_shape_check
            )
            bytes_written += write_awq_blob(f, blob)
            del blob
            gc.collect()

        print("=" * 70)
        print("Writing post_attention_layernorm weights")
        print("=" * 70)
        for i in tqdm(range(layers), desc="post_attention_layernorm"):
            name = f"model.layers.{i}.post_attention_layernorm.weight"
            bytes_written += write_nonquant_bf16(f, loader, name, dims, no_shape_check)

        print("=" * 70)
        print("Writing fused Gate/Up AWQ blobs")
        print("=" * 70)
        for i in tqdm(range(layers), desc="gate_up"):
            prefix = f"model.layers.{i}.mlp"
            gate = convert_one_awq_linear_to_cpp_layout(
                loader, prefix + ".gate_proj", intermediate, hidden, group_size, no_shape_check
            )
            up = convert_one_awq_linear_to_cpp_layout(
                loader, prefix + ".up_proj", intermediate, hidden, group_size, no_shape_check
            )
            gate_up = concat_awq_linears_along_output([gate, up])
            bytes_written += write_awq_blob(f, gate_up)
            del gate, up, gate_up
            gc.collect()

        print("=" * 70)
        print("Writing Down projection AWQ blobs")
        print("=" * 70)
        for i in tqdm(range(layers), desc="down_proj"):
            base = f"model.layers.{i}.mlp.down_proj"
            blob = convert_one_awq_linear_to_cpp_layout(
                loader, base, hidden, intermediate, group_size, no_shape_check
            )
            bytes_written += write_awq_blob(f, blob)
            del blob
            gc.collect()

        print("=" * 70)
        print("Writing final norm")
        print("=" * 70)
        bytes_written += write_nonquant_bf16(f, loader, "model.norm.weight", dims, no_shape_check)

    actual_size = out_path.stat().st_size

    print("=" * 70)
    print("Export completed")
    print("=" * 70)
    print(f"Output file       : {out_path}")
    print(f"Bytes written     : {bytes_written}")
    print(f"Actual size       : {actual_size}")
    print(f"Expected size     : {expected_size}")

    if bytes_written != actual_size:
        raise RuntimeError(
            "Internal byte counter mismatch!\n"
            f"bytes_written: {bytes_written}\n"
            f"actual_size  : {actual_size}"
        )

    if actual_size != expected_size:
        raise RuntimeError(
            "Output file size mismatch!\n"
            f"actual   : {actual_size}\n"
            f"expected : {expected_size}"
        )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-4B-AWQ HF/AutoAWQ -> custom BF16 + AWQ INT4 .bin exporter. "
            "AutoAWQ nibble order is handled in Python; C++ unpacks plain 01234567."
        )
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_QWEN3_4B_AWQ_MODEL,
        help="Path to HF Qwen3-4B-AWQ model directory",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default=DEFAULT_OUT_FILE,
        help="Output .bin path",
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="AWQ group size. Default: read from config.json, fallback to 128",
    )

    parser.add_argument(
        "--no_shape_check",
        action="store_true",
        help="Disable strict tensor shape validation",
    )

    args = parser.parse_args()

    export_qwen3_awq(
        model_dir=args.model_dir,
        out_file=args.out_file,
        group_size=args.group_size,
        no_shape_check=args.no_shape_check,
    )


if __name__ == "__main__":
    main()