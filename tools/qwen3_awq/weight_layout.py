from safetensors import safe_open
from glob import glob
from collections import defaultdict

model_dir = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ"
files = sorted(glob(model_dir + "/*.safetensors"))

targets = defaultdict(int)

for f in files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if k.endswith(".qweight"):
                if ".self_attn." in k:
                    targets["attention_qweight"] += 1
                if ".mlp." in k:
                    targets["mlp_qweight"] += 1
                if "lm_head" in k:
                    targets["lm_head_qweight"] += 1

print(targets)