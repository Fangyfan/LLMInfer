import os
import json
from glob import glob
from safetensors import safe_open
from collections import Counter

# 如果是原始 FP16/BF16 模型：
# model_dir = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B"

# 如果是 AWQ INT4 量化模型：
model_dir = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ"

files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
assert files, f"No safetensors files found in {model_dir}"

config_path = os.path.join(model_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("===== Qwen3 Config =====")
    print("model_type:", config.get("model_type"))
    print("hidden_size:", config.get("hidden_size"))
    print("intermediate_size:", config.get("intermediate_size"))
    print("num_hidden_layers:", config.get("num_hidden_layers"))
    print("num_attention_heads:", config.get("num_attention_heads"))
    print("num_key_value_heads:", config.get("num_key_value_heads"))
    print("head_dim:", config.get("head_dim"))
    print("vocab_size:", config.get("vocab_size"))

    quant_config = config.get("quantization_config", None)
    if quant_config:
        print("\n===== Quantization Config =====")
        print(json.dumps(quant_config, indent=2, ensure_ascii=False))
    else:
        print("\nNo quantization_config found. This is likely FP16/BF16 Qwen3-4B.")
else:
    print(f"Warning: config.json not found in {model_dir}")

print("\n===== Safetensors Files =====")
for f in files:
    print(os.path.basename(f))

# Qwen3-4B / Qwen3-4B-AWQ 关键权重名
target_keywords = [
    # embedding / output
    "embed_tokens",
    "lm_head",

    # final norm
    "model.norm",

    # attention
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",

    # MLP
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",

    # layer norms
    "input_layernorm",
    "post_attention_layernorm",

    # AWQ-specific
    "qweight",
    "qzeros",
    "scales",
]

dtype_counter = Counter()
shape_counter = Counter()
total_tensors = 0

print("\n===== Matched Tensors =====")

for f in files:
    print(f"\n--- {os.path.basename(f)} ---")

    with safe_open(f, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if any(s in k for s in target_keywords):
                t = sf.get_tensor(k)

                print(k, tuple(t.shape), t.dtype)

                dtype_counter[str(t.dtype)] += 1
                shape_counter[tuple(t.shape)] += 1
                total_tensors += 1

print("\n===== Summary =====")
print("matched tensor count:", total_tensors)

print("\nDType count:")
for dtype, count in dtype_counter.items():
    print(dtype, count)

print("\nMost common shapes:")
for shape, count in shape_counter.most_common(30):
    print(shape, count)