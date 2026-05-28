#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Qwen3 dense HF -> .bin exporter.

This exporter is aligned with the following C++ loading layout:

1. Embedding:
   model.embed_tokens.weight

2. Pre RMSNorm:
   model.layers.{i}.input_layernorm.weight

3. Attention QKV per layer:
   model.layers.{i}.self_attn.q_proj.weight
   model.layers.{i}.self_attn.k_proj.weight
   model.layers.{i}.self_attn.v_proj.weight

4. Q/K RMSNorm per layer:
   model.layers.{i}.self_attn.q_norm.weight
   model.layers.{i}.self_attn.k_norm.weight

5. Attention O projection:
   model.layers.{i}.self_attn.o_proj.weight

6. FFN RMSNorm:
   model.layers.{i}.post_attention_layernorm.weight

7. FFN Gate/Up per layer:
   model.layers.{i}.mlp.gate_proj.weight
   model.layers.{i}.mlp.up_proj.weight

8. FFN Down:
   model.layers.{i}.mlp.down_proj.weight

9. Final RMSNorm:
   model.norm.weight

10. LM Head:
   Not written.
   The C++ loader ties lm_head to embedding by using weight_ptr(0).

Binary format:

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

followed by weights.

Supported output dtype:
- fp32: each weight element is 4 bytes
- bf16: each weight element is 2 bytes, stored as raw little-endian bfloat16 bits

Important:
The header is unchanged for fp32 and bf16.
Your C++ loader must know externally whether the weight payload is fp32 or bf16.

Supports:
- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Other dense Qwen3 variants following the same HF naming convention

Supports:
- single model.safetensors
- sharded model-00001-of-0000X.safetensors
- automatic shard lookup via model.safetensors.index.json
- shard files without index, such as model-00001-of-00001.safetensors

Examples
--------

Qwen3-0.6B fp32:
python qwen3_dense_unified_converter.py --preset 0.6b --out_dtype fp32

Qwen3-0.6B bf16:
python qwen3_dense_unified_converter.py --preset 0.6b --out_dtype bf16

Qwen3-1.7B bf16:
python qwen3_dense_unified_converter.py --preset 1.7b --out_dtype bf16

Qwen3-4B bf16:
python qwen3_dense_unified_converter.py --preset 4b --out_dtype bf16

Install dependencies:
pip install torch safetensors tqdm
"""

import argparse
import gc
import json
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError as e:
    raise ImportError(
        "Please install safetensors first: pip install safetensors"
    ) from e


# ============================================================
# default paths / presets
# ============================================================

DEFAULT_06B_MODEL = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-0.6B"
DEFAULT_17B_MODEL = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-1.7B"
DEFAULT_4B_MODEL = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B"

DEFAULT_OUT_DIR = "/home/yifanfang/LLMInfer/models/qwen3"

PRESETS: Dict[str, str] = {
    "0.6b": DEFAULT_06B_MODEL,
    "1.7b": DEFAULT_17B_MODEL,
    "4b": DEFAULT_4B_MODEL,
}


def default_out_file_for_preset(preset: str, out_dtype: str) -> str:
    return str(Path(DEFAULT_OUT_DIR) / f"qwen3_{preset}_{out_dtype}.bin")


# ============================================================
# binary serialization
# ============================================================


def serialize_weight(file_obj, tensor: torch.Tensor, out_dtype: str) -> None:
    """
    Write tensor payload.

    fp32:
        stored as little-endian float32, 4 bytes per element.

    bf16:
        stored as raw bfloat16 bits, 2 bytes per element.
        This preserves bf16 storage format.
    """

    if out_dtype == "fp32":
        arr = (
            tensor.detach()
            .cpu()
            .contiguous()
            .view(-1)
            .to(torch.float32)
            .numpy()
        )
        file_obj.write(arr.tobytes())
        return

    if out_dtype == "bf16":
        # Convert to bf16 if the source tensor is not already bf16.
        #
        # Direct .numpy() on torch.bfloat16 is not supported in many PyTorch
        # versions, so we reinterpret the bf16 tensor as uint16 and write the
        # raw 16-bit bf16 payload.
        bf16_tensor = (
            tensor.detach()
            .cpu()
            .contiguous()
            .view(-1)
            .to(torch.bfloat16)
        )

        uint16_tensor = bf16_tensor.view(torch.uint16)
        arr = uint16_tensor.numpy()
        file_obj.write(arr.tobytes())
        return

    raise ValueError(f"Unsupported out_dtype: {out_dtype}")


def bytes_per_element(out_dtype: str) -> int:
    if out_dtype == "fp32":
        return 4
    if out_dtype == "bf16":
        return 2
    raise ValueError(f"Unsupported out_dtype: {out_dtype}")


# ============================================================
# lazy shard loader
# ============================================================


class ShardedTensorLoader:
    """
    Lazy loader for HuggingFace safetensors.

    Works for:
    - single-file checkpoints
    - multi-shard checkpoints
    - one-file shard checkpoints such as model-00001-of-00001.safetensors

    We do not load all shards into RAM simultaneously.
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

            self.weight_map = {
                tensor_name: single_path.name
                for tensor_name in tensors.keys()
            }

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
                        raise RuntimeError(
                            "Duplicate tensor name found while scanning shards: "
                            f"{tensor_name}"
                        )

                    self.weight_map[tensor_name] = shard_path.name

                del tensors
                gc.collect()

    def _switch_shard(self, shard_name: str) -> None:
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
        """Return tensor by HF tensor name."""

        if tensor_name not in self.weight_map:
            raise KeyError(f"Tensor not found: {tensor_name}")

        shard_name = self.weight_map[tensor_name]

        self._switch_shard(shard_name)

        assert self.current_tensors is not None

        if tensor_name not in self.current_tensors:
            raise KeyError(
                f"Tensor {tensor_name} missing inside shard {shard_name}"
            )

        return self.current_tensors[tensor_name]


# ============================================================
# tensor order aligned with C++ loader
# ============================================================


def build_export_order(num_layers: int):
    """
    Export tensor order must match C++ offset order exactly.

    This order supports both:

    1. Qwen3Model::create_param_layers()
    2. Qwen3FusedModel::create_param_layers()

    because the fused loader reads several consecutive tensors as one fused block:
    - q + k + v
    - q_norm + k_norm
    - gate + up
    """

    names = []

    # 1. Embedding
    names += ["model.embed_tokens.weight"]

    # 2. Pre RMSNorm / input_layernorm
    names += [
        f"model.layers.{i}.input_layernorm.weight"
        for i in range(num_layers)
    ]

    # 3. Attention QKV per layer
    for i in range(num_layers):
        names += [
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
        ]

    # 4. q_norm / k_norm per layer
    for i in range(num_layers):
        names += [
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight",
        ]

    # 5. Attention O projection
    names += [
        f"model.layers.{i}.self_attn.o_proj.weight"
        for i in range(num_layers)
    ]

    # 6. FFN RMSNorm / post_attention_layernorm
    names += [
        f"model.layers.{i}.post_attention_layernorm.weight"
        for i in range(num_layers)
    ]

    # 7. FFN gate + up per layer
    for i in range(num_layers):
        names += [
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
        ]

    # 8. FFN down projection
    names += [
        f"model.layers.{i}.mlp.down_proj.weight"
        for i in range(num_layers)
    ]

    # 9. Final RMSNorm
    names += ["model.norm.weight"]

    # 10. lm_head.weight is intentionally NOT written.
    # C++ ties lm_head to embedding by using weight_ptr(0).

    return names


# ============================================================
# shape checking
# ============================================================


def expected_shape(name: str, cfg: dict) -> Optional[Tuple[int, ...]]:
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])

    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])

    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))

    vocab_size = int(cfg["vocab_size"])

    if name == "model.norm.weight":
        return (hidden_size,)

    if name == "model.embed_tokens.weight":
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


def expected_element_count_from_order(export_order, cfg: dict) -> int:
    total = 0

    for name in export_order:
        shape = expected_shape(name, cfg)

        if shape is None:
            raise RuntimeError(f"Cannot infer expected shape for tensor: {name}")

        numel = 1
        for d in shape:
            numel *= d

        total += numel

    return total


# ============================================================
# main export
# ============================================================


def export_model(
    model_dir: str,
    out_file: str,
    out_dtype: str = "fp32",
    no_shape_check: bool = False,
) -> None:

    if out_dtype not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")

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

    # Qwen3-0.6B:
    #   hidden_size = 1024
    #   num_attention_heads = 16
    #   head_dim = 128
    #   dim = 16 * 128 = 2048
    #
    # So dim must be num_heads * head_dim, not hidden_size.

    dim = num_heads * head_dim
    hidden_dim = hidden_size
    kv_dim = num_kv_heads * head_dim

    print("=" * 60)
    print("Model config")
    print("=" * 60)

    print(f"hidden_size      : {hidden_size}")
    print(f"intermediate_size: {intermediate_size}")
    print(f"num_layers       : {num_layers}")
    print(f"num_heads        : {num_heads}")
    print(f"num_kv_heads     : {num_kv_heads}")
    print(f"head_dim         : {head_dim}")
    print(f"dim              : {dim}")
    print(f"hidden_dim       : {hidden_dim}")
    print(f"kv_dim           : {kv_dim}")
    print(f"vocab_size       : {vocab_size}")
    print(f"max_seq_len      : {max_seq_len}")
    print(f"out_dtype        : {out_dtype}")

    loader = ShardedTensorLoader(str(model_dir))

    export_order = build_export_order(num_layers)

    expected_element_count = expected_element_count_from_order(export_order, cfg)
    expected_file_size = 8 * 4 + expected_element_count * bytes_per_element(out_dtype)

    print("=" * 60)
    print("Export layout")
    print("=" * 60)
    print(f"num_tensors      : {len(export_order)}")
    print(f"write_lm_head    : False")
    print(f"elements         : {expected_element_count}")
    print(f"bytes_per_element: {bytes_per_element(out_dtype)}")
    print(f"expected_size    : {expected_file_size} bytes")

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

                if expected is not None and tuple(tensor.shape) != expected:
                    raise RuntimeError(
                        f"Shape mismatch for {tensor_name}\n"
                        f"got      : {tuple(tensor.shape)}\n"
                        f"expected : {expected}"
                    )

            serialize_weight(f, tensor, out_dtype)

    actual_file_size = out_path.stat().st_size

    print("=" * 60)
    print("Export completed")
    print("=" * 60)
    print(f"Output file      : {out_path}")
    print(f"Output dtype     : {out_dtype}")
    print(f"Actual size      : {actual_file_size} bytes")
    print(f"Expected size    : {expected_file_size} bytes")

    if actual_file_size != expected_file_size:
        raise RuntimeError(
            "Output file size mismatch!\n"
            f"actual   : {actual_file_size}\n"
            f"expected : {expected_file_size}"
        )


# ============================================================
# cli
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description="Qwen3 dense HF -> .bin exporter aligned with C++ loader"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to HF Qwen3 model directory",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Output .bin path",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Convenience preset: 0.6b, 1.7b, 4b",
    )

    parser.add_argument(
        "--out_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Output weight dtype: fp32 or bf16",
    )

    parser.add_argument(
        "--no_shape_check",
        action="store_true",
        help="Disable tensor shape validation",
    )

    args = parser.parse_args()

    if args.preset is not None:
        model_dir = PRESETS[args.preset]
        out_file = args.out_file or default_out_file_for_preset(
            args.preset,
            args.out_dtype,
        )
    else:
        model_dir = args.model_dir or DEFAULT_17B_MODEL
        out_file = args.out_file or str(
            Path(DEFAULT_OUT_DIR) / f"qwen3_1.7b_{args.out_dtype}.bin"
        )

    export_model(
        model_dir=model_dir,
        out_file=out_file,
        out_dtype=args.out_dtype,
        no_shape_check=args.no_shape_check,
    )


if __name__ == "__main__":
    main()