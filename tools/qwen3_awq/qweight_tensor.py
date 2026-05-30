from safetensors import safe_open
from glob import glob
from collections import Counter

model_dir = "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ"

files = sorted(glob(model_dir + "/*.safetensors"))
dtype_counter = Counter()
suffix_counter = Counter()

for f in files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            t = sf.get_tensor(k)
            dtype_counter[str(t.dtype)] += 1
            if k.endswith(".qweight"):
                suffix_counter["qweight"] += 1
                print(k, t.shape, t.dtype)
            elif k.endswith(".qzeros"):
                suffix_counter["qzeros"] += 1
            elif k.endswith(".scales"):
                suffix_counter["scales"] += 1
            elif k.endswith(".g_idx"):
                suffix_counter["g_idx"] += 1
            elif k.endswith(".weight"):
                suffix_counter["normal_weight"] += 1

print("dtype_counter:", dtype_counter)
print("suffix_counter:", suffix_counter)