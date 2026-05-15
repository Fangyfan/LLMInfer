#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Qwen3 dense HF -> fp32 .bin exporter.

Supports:
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Other dense Qwen3 variants following the same HF naming convention

The generated binary format is intentionally kept IDENTICAL to the original
reference exporter:

header = int32[8]
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

followed by fp32 weights written in the exact same tensor order.

Supports:
- single model.safetensors
- sharded model-00001-of-0000X.safetensors
- automatic shard lookup via model.safetensors.index.json

Examples
--------

Qwen3-1.7B:
python qwen3_dense_unified_converter.py \
  --model_dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-1.7B \
  --out_file /home/yifanfang/LLMInfer/models/qwen3/qwen3_1.7b_fp32.bin

Qwen3-4B:
python qwen3_dense_unified_converter.py \
  --model_dir /home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B \
  --out_file /home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_fp32.bin

Install dependencies:
pip install torch safetensors tqdm
"""

import argparse
import gc
import json
import os
import struct
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError as e:
    raise ImportError(
        "Please install safetensors first: pip install safetensors"
    ) from e


DEFAULT_17B_MODEL = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-1.7B"
DEFAULT_17B_OUT = "/home/yifanfang/LLMInfer/models/qwen3/qwen3_1.7b_fp32.bin"

DEFAULT_4B_MODEL = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B"
DEFAULT_4B_OUT = "/home/yifanfang/LLMInfer/models/qwen3/qwen3_4b_fp32.bin"


# ============================================================
# binary serialization
# ============================================================


def serialize_fp32(file_obj, tensor: torch.Tensor) -> None:
    """Write tensor as contiguous little-endian fp32."""

    arr = (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(-1)
        .to(torch.float32)
        .numpy()
    )

    file_obj.write(struct.pack(f"{arr.size}f", *arr))


# ============================================================
# lazy shard loader
# ============================================================


class ShardedTensorLoader:
    """
    Lazy loader for HuggingFace safetensors.

    Works for:
    - single-file checkpoints
    - multi-shard checkpoints

    Qwen3-4B typically contains 3 shards:
        model-00001-of-00003.safetensors
        model-00002-of-00003.safetensors
        model-00003-of-00003.safetensors

    We DO NOT load all shards into RAM simultaneously.

    Instead:
    - read model.safetensors.index.json
    - find which shard contains a tensor
    - load only that shard
    - automatically switch shards when needed
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

        # ----------------------------------------------------
        # sharded checkpoint
        # ----------------------------------------------------

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index_json = json.load(f)

            self.weight_map = index_json["weight_map"]

            shard_names = sorted(set(self.weight_map.values()))

            print("Detected sharded checkpoint:")
            for s in shard_names:
                print("   ", s)

        # ----------------------------------------------------
        # single-file checkpoint
        # ----------------------------------------------------

        elif single_path.exists():
            tensors = safe_load_file(str(single_path), device="cpu")

            self.weight_map = {
                tensor_name: single_path.name
                for tensor_name in tensors.keys()
            }

            del tensors
            gc.collect()

            print("Detected single safetensors checkpoint")

        else:
            raise FileNotFoundError(
                "Cannot find model.safetensors.index.json or model.safetensors"
            )

    def _switch_shard(self, shard_name: str):
        """Load a new shard only when necessary."""

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
        """
        Return tensor by name.

        Some Qwen3 checkpoints tie lm_head.weight with embedding.
        If lm_head.weight is absent, reuse embedding weight.
        """

        original_name = tensor_name

        if tensor_name not in self.weight_map:
            if tensor_name == "lm_head.weight":
                tensor_name = "model.embed_tokens.weight"
            else:
                raise KeyError(f"Tensor not found: {original_name}")

        shard_name = self.weight_map[tensor_name]

        self._switch_shard(shard_name)

        assert self.current_tensors is not None

        if tensor_name not in self.current_tensors:
            raise KeyError(
                f"Tensor {tensor_name} missing inside shard {shard_name}"
            )

        return self.current_tensors[tensor_name]


# ============================================================
# tensor order
# ============================================================


def build_export_order(num_layers: int):
    """
    IMPORTANT:
    Keep EXACTLY identical to the original reference exporter.
    """

    names = []

    # RMSNorms
    names += [
        f"model.layers.{i}.input_layernorm.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.post_attention_layernorm.weight"
        for i in range(num_layers)
    ]

    names += ["model.norm.weight"]

    # embedding
    names += ["model.embed_tokens.weight"]

    # attention
    names += [
        f"model.layers.{i}.self_attn.q_proj.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.self_attn.q_norm.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.self_attn.k_proj.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.self_attn.k_norm.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.self_attn.v_proj.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.self_attn.o_proj.weight"
        for i in range(num_layers)
    ]

    # mlp
    names += [
        f"model.layers.{i}.mlp.gate_proj.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.mlp.down_proj.weight"
        for i in range(num_layers)
    ]

    names += [
        f"model.layers.{i}.mlp.up_proj.weight"
        for i in range(num_layers)
    ]

    # lm head
    names += ["lm_head.weight"]

    return names


# ============================================================
# shape checking
# ============================================================


def expected_shape(name: str, cfg: dict):
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])

    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])

    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))

    vocab_size = int(cfg["vocab_size"])

    if name == "model.norm.weight":
        return (hidden_size,)

    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (vocab_size, hidden_size)

    for i in range(num_layers):
        prefix = f"model.layers.{i}."

        if name in (
            prefix + "input_layernorm.weight",
            prefix + "post_attention_layernorm.weight",
        ):
            return (hidden_size,)

        if name == prefix + "self_attn.q_proj.weight":
            return (num_heads * head_dim, hidden_size)

        if name == prefix + "self_attn.k_proj.weight":
            return (num_kv_heads * head_dim, hidden_size)

        if name == prefix + "self_attn.v_proj.weight":
            return (num_kv_heads * head_dim, hidden_size)

        if name == prefix + "self_attn.o_proj.weight":
            return (hidden_size, num_heads * head_dim)

        if name in (
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
        ):
            return (head_dim,)

        if name in (
            prefix + "mlp.gate_proj.weight",
            prefix + "mlp.up_proj.weight",
        ):
            return (intermediate_size, hidden_size)

        if name == prefix + "mlp.down_proj.weight":
            return (hidden_size, intermediate_size)

    return None


# ============================================================
# main export
# ============================================================


def export_model(model_dir: str, out_file: str, no_shape_check: bool = False):

    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])

    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])

    vocab_size = int(cfg["vocab_size"])
    max_seq_len = int(cfg["max_position_embeddings"])

    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # Keep exactly same semantics as original exporter.
    #
    # dim = num_heads * head_dim
    # hidden_dim = hidden_size
    # --------------------------------------------------------

    dim = num_heads * head_dim
    hidden_dim = hidden_size

    print("=" * 60)
    print("Model config")
    print("=" * 60)

    print(f"hidden_size      : {hidden_size}")
    print(f"intermediate_size: {intermediate_size}")
    print(f"num_layers       : {num_layers}")
    print(f"num_heads        : {num_heads}")
    print(f"num_kv_heads     : {num_kv_heads}")
    print(f"head_dim         : {head_dim}")
    print(f"vocab_size       : {vocab_size}")
    print(f"max_seq_len      : {max_seq_len}")

    loader = ShardedTensorLoader(str(model_dir))

    export_order = build_export_order(num_layers)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:

        # ====================================================
        # header
        # ====================================================

        header = struct.pack(
            "iiiiiiii",
            dim,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            max_seq_len,
            intermediate_size,
        )

        f.write(header)

        print("=" * 60)
        print("Writing weights")
        print("=" * 60)

        for tensor_name in tqdm(export_order):

            tensor = loader.get(tensor_name)

            if not no_shape_check:
                expected = expected_shape(tensor_name, cfg)

                if expected is not None:
                    if tuple(tensor.shape) != expected:
                        raise RuntimeError(
                            f"Shape mismatch for {tensor_name}\n"
                            f"got      : {tuple(tensor.shape)}\n"
                            f"expected : {expected}"
                        )

            serialize_fp32(f, tensor)

    print("=" * 60)
    print("Export completed")
    print("=" * 60)
    print(f"Output file: {out_path}")


# ============================================================
# cli
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description="Qwen3 dense HF -> fp32 .bin exporter"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default=DEFAULT_17B_MODEL,
        help="Path to HF Qwen3 model directory",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        required=False,
        default=DEFAULT_17B_OUT,
        help="Output fp32 .bin path",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[None, "1.7b", "4b"],
        help="Convenience preset",
    )

    parser.add_argument(
        "--no_shape_check",
        action="store_true",
        help="Disable tensor shape validation",
    )

    args = parser.parse_args()

    # convenience presets
    if args.preset == "1.7b":
        args.model_dir = DEFAULT_17B_MODEL
        args.out_file = DEFAULT_17B_OUT

    elif args.preset == "4b":
        args.model_dir = DEFAULT_4B_MODEL
        args.out_file = DEFAULT_4B_OUT

    export_model(
        model_dir=args.model_dir,
        out_file=args.out_file,
        no_shape_check=args.no_shape_check,
    )


if __name__ == "__main__":
    main()
