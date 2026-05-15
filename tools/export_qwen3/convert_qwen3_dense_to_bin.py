#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Qwen3 dense HF weights to the same fp32 .bin layout used by the Qwen3-0.6B reference code.

Default input : /home/yifanfang/LLMInfer/models/qwen3/Qwen3-1.7B
Default output: /home/yifanfang/LLMInfer/models/qwen3/qwen3_1.7b_fp32.bin

The script supports HuggingFace safetensors sharded checkpoints, e.g.
model-00001-of-00002.safetensors + model-00002-of-00002.safetensors, by reading
model.safetensors.index.json and loading each shard lazily.
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
    raise ImportError("Please install safetensors first: pip install safetensors") from e


DEFAULT_MODEL_DIR = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-1.7B"
DEFAULT_OUT_FILE = "/home/yifanfang/LLMInfer/models/qwen3/qwen3_1.7b_fp32.bin"


def serialize_fp32(file_obj, tensor: torch.Tensor) -> None:
    """Write one tensor as contiguous little-endian fp32 values."""
    arr = tensor.detach().cpu().contiguous().view(-1).to(torch.float32).numpy()
    file_obj.write(struct.pack(f"{arr.size}f", *arr))


class ShardedTensorLoader:
    """Lazy tensor loader for single-file or sharded HF safetensors checkpoints."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model dir does not exist: {self.model_dir}")

        self.weight_map: Dict[str, str] = {}
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"

        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            self.weight_map = index["weight_map"]
        elif single_path.exists():
            tensors = safe_load_file(str(single_path), device="cpu")
            self.weight_map = {name: single_path.name for name in tensors.keys()}
            del tensors
        else:
            raise FileNotFoundError(
                "No safetensors checkpoint found. Expected model.safetensors.index.json "
                "or model.safetensors under: " + str(self.model_dir)
            )

        self._current_file: Optional[str] = None
        self._current_tensors: Optional[Dict[str, torch.Tensor]] = None

    def _load_shard(self, shard_name: str) -> None:
        if shard_name == self._current_file:
            return
        self._current_tensors = None
        gc.collect()
        shard_path = self.model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard listed in index is missing: {shard_path}")
        self._current_tensors = safe_load_file(str(shard_path), device="cpu")
        self._current_file = shard_name

    def get(self, name: str) -> torch.Tensor:
        # Qwen3 dense checkpoints have tied word embeddings. Some exports omit lm_head.weight.
        if name not in self.weight_map and name == "lm_head.weight":
            name = "model.embed_tokens.weight"

        if name not in self.weight_map:
            raise KeyError(f"Tensor not found in checkpoint: {name}")

        shard_name = self.weight_map[name]
        self._load_shard(shard_name)
        assert self._current_tensors is not None
        if name not in self._current_tensors:
            raise KeyError(f"Tensor {name} not found inside shard {shard_name}")
        return self._current_tensors[name]


def required_weight_names(num_layers: int):
    names = []

    # Same order as the Qwen3-0.6B reference exporter:
    # 2 * rmsnorm + final norm
    names += [f"model.layers.{i}.input_layernorm.weight" for i in range(num_layers)]
    names += [f"model.layers.{i}.post_attention_layernorm.weight" for i in range(num_layers)]
    names += ["model.norm.weight"]

    names += ["model.embed_tokens.weight"]

    names += [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(num_layers)]
    names += [f"model.layers.{i}.self_attn.q_norm.weight" for i in range(num_layers)]

    names += [f"model.layers.{i}.self_attn.k_proj.weight" for i in range(num_layers)]
    names += [f"model.layers.{i}.self_attn.k_norm.weight" for i in range(num_layers)]

    names += [f"model.layers.{i}.self_attn.v_proj.weight" for i in range(num_layers)]
    names += [f"model.layers.{i}.self_attn.o_proj.weight" for i in range(num_layers)]

    names += [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(num_layers)]
    names += [f"model.layers.{i}.mlp.down_proj.weight" for i in range(num_layers)]
    names += [f"model.layers.{i}.mlp.up_proj.weight" for i in range(num_layers)]

    # If tied and absent in checkpoint, ShardedTensorLoader maps it to embed_tokens.weight.
    names += ["lm_head.weight"]
    return names


def expected_shape(name: str, cfg: dict):
    n_layers = int(cfg["num_hidden_layers"])
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    head_dim = int(cfg.get("head_dim", hidden_size // int(cfg["num_attention_heads"])))
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    vocab_size = int(cfg["vocab_size"])

    if name in ("model.norm.weight",):
        return (hidden_size,)
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return (vocab_size, hidden_size)

    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        if name in (prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"):
            return (hidden_size,)
        if name == prefix + "self_attn.q_proj.weight":
            return (n_heads * head_dim, hidden_size)
        if name in (prefix + "self_attn.q_norm.weight", prefix + "self_attn.k_norm.weight"):
            return (head_dim,)
        if name in (prefix + "self_attn.k_proj.weight", prefix + "self_attn.v_proj.weight"):
            return (n_kv_heads * head_dim, hidden_size)
        if name == prefix + "self_attn.o_proj.weight":
            return (hidden_size, n_heads * head_dim)
        if name in (prefix + "mlp.gate_proj.weight", prefix + "mlp.up_proj.weight"):
            return (intermediate_size, hidden_size)
        if name == prefix + "mlp.down_proj.weight":
            return (hidden_size, intermediate_size)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3 dense HF safetensors to reference fp32 .bin layout.")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE)
    parser.add_argument("--no_shape_check", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim", int(cfg["hidden_size"]) // n_heads))

    # Keep the reference format exactly:
    # header = int32[8]: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, intermediate_size
    dim = n_heads * head_dim
    hidden_dim = int(cfg["hidden_size"])
    vocab_size = int(cfg["vocab_size"])
    max_seq_len = int(cfg["max_position_embeddings"])
    intermediate_size = int(cfg["intermediate_size"])

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loader = ShardedTensorLoader(str(model_dir))
    names = required_weight_names(n_layers)

    with open(out_path, "wb") as f:
        header = struct.pack(
            "iiiiiiii",
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            max_seq_len,
            intermediate_size,
        )
        f.write(header)

        for name in tqdm(names, desc="Writing fp32 weights"):
            tensor = loader.get(name)
            if not args.no_shape_check:
                exp = expected_shape(name, cfg)
                if exp is not None and tuple(tensor.shape) != exp:
                    raise RuntimeError(f"Shape mismatch for {name}: got {tuple(tensor.shape)}, expected {exp}")
            serialize_fp32(f, tensor)

    print(f"wrote {out_path}")
    print(
        "header:",
        {
            "dim": dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "intermediate_size": intermediate_size,
        },
    )


if __name__ == "__main__":
    main()