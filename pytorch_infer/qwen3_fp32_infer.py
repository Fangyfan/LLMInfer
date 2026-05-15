# qwen3_fp32_infer.py
import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def fill_template(prompt: str) -> str:
    """
    对齐你 C++ 代码里的模板：
    <|im_start|>user
    prompt<|im_end|>
    <|im_start|>assistant
    """
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def sync_device(device: torch.device) -> None:
    """
    CUDA 是异步执行的，不同步会导致 TTFT / TPOT 偏小。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_stop_ids(tokenizer) -> Set[int]:
    """
    C++ 代码里过滤了 151645 和 151644。
    这里同时从 tokenizer 中动态获取特殊 token id，避免硬编码失效。
    """
    stop_ids: Set[int] = set()

    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_ids.update(tokenizer.eos_token_id)
        else:
            stop_ids.add(tokenizer.eos_token_id)

    for token in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            stop_ids.add(token_id)

    return stop_ids


def build_skip_ids(tokenizer) -> Set[int]:
    """
    生成文本时不把这些特殊 token 放进 decode 列表。
    对齐 C++ 中：
    if (next_token_id != 151645 && next_token_id != 151644)
    """
    skip_ids: Set[int] = set()

    for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            skip_ids.add(token_id)

    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            skip_ids.update(tokenizer.eos_token_id)
        else:
            skip_ids.add(tokenizer.eos_token_id)

    return skip_ids


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    默认 temperature=0，执行 greedy decode。
    如果 temperature > 0，则执行采样。
    返回 shape: [1, 1]
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_manual(
    model,
    tokenizer,
    prompt: str,
    total_steps: int = 2560,
    need_output: bool = True,
    use_chat_template: bool = False,
    enable_thinking: Optional[bool] = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Dict[str, float]:
    """
    手动 prefill + decode，尽量对齐 C++ generate()。

    total_steps 的含义对齐 C++：
    C++ 里 pos 从 0 开始，prompt 阶段也计入 pos。
    第一个生成 token 发生在 pos == prompt_len - 1 时。
    因此最大新 token 数近似为：
        max_new_tokens = total_steps - prompt_len + 1
    """
    device = next(model.parameters()).device

    stop_ids = build_stop_ids(tokenizer)
    skip_ids = build_skip_ids(tokenizer)

    sync_device(device)
    ttft_start = time.perf_counter()

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]

        # Qwen3-2504 支持 enable_thinking；
        # 部分新版 Instruct 模型不再需要或不支持这个参数，所以做兼容处理。
        try:
            if enable_thinking is None:
                sentence = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                sentence = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
        except TypeError:
            sentence = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        sentence = fill_template(prompt)

    model_inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)

    if input_ids.numel() == 0:
        raise RuntimeError("input token ids is empty!")

    prompt_len = input_ids.shape[-1]
    max_new_tokens = max(1, total_steps - prompt_len + 1)

    generated_token_ids: List[int] = []

    total_latency = 0.0
    gen_token_count = 0

    # -------------------------
    # 1. Prefill：整段 prompt 一次性送入模型
    # -------------------------
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )

    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    next_token = sample_next_token(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    sync_device(device)
    ttft_end = time.perf_counter()

    TTFT = ttft_end - ttft_start
    last_token_time = ttft_end
    gen_token_count = 1

    token_id = int(next_token.item())
    if token_id not in skip_ids:
        generated_token_ids.append(token_id)

    # 如果第一个 token 就是结束符，直接结束
    if token_id in stop_ids:
        TPOT = 0.0
        decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        if need_output:
            print(decoded_text)

        return {
            "prompt_len": float(prompt_len),
            "generated_tokens": float(gen_token_count),
            "processed_steps": float(prompt_len + gen_token_count - 1),
            "TTFT": TTFT,
            "TPOT": TPOT,
        }

    # -------------------------
    # 2. Decode：逐 token 生成
    # -------------------------
    cur_token = next_token

    for _ in range(1, max_new_tokens):
        outputs = model(
            input_ids=cur_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        sync_device(device)
        current_time = time.perf_counter()

        token_latency = current_time - last_token_time
        total_latency += token_latency
        last_token_time = current_time

        gen_token_count += 1

        token_id = int(next_token.item())
        if token_id not in skip_ids:
            generated_token_ids.append(token_id)

        cur_token = next_token

        if token_id in stop_ids:
            break

    if gen_token_count > 1:
        TPOT = total_latency / (gen_token_count - 1)
    else:
        TPOT = 0.0

    decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    if need_output:
        print(decoded_text)

    return {
        "prompt_len": float(prompt_len),
        "generated_tokens": float(gen_token_count),
        "processed_steps": float(prompt_len + gen_token_count - 1),
        "TTFT": TTFT,
        "TPOT": TPOT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="本地 Hugging Face 格式 Qwen3 模型路径，例如 ./Qwen3-0.6B",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is AI?",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=2560,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="使用 tokenizer.apply_chat_template，而不是手写模板",
    )
    parser.add_argument(
        "--enable-thinking",
        type=int,
        default=-1,
        help="-1 表示不传；0 表示 False；1 表示 True",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 表示 greedy decode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please use --device cpu")

    device = torch.device(args.device)

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    print(f"Loading Qwen3 FP32 model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    print(args.prompt)
    print("Qwen3 FP32 model generating...")

    enable_thinking: Optional[bool]
    if args.enable_thinking < 0:
        enable_thinking = None
    else:
        enable_thinking = bool(args.enable_thinking)

    sync_device(device)
    start = time.perf_counter()

    metrics = generate_manual(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        total_steps=args.total_steps,
        need_output=not args.no_output,
        use_chat_template=args.use_chat_template,
        enable_thinking=enable_thinking,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    sync_device(device)
    end = time.perf_counter()

    duration = end - start

    processed_steps = metrics["processed_steps"]
    generated_tokens = metrics["generated_tokens"]

    print("\n--------------- Performance Metrics ---------------")
    print(f"prompt_len: {int(metrics['prompt_len'])}")
    print(f"generated_tokens: {int(generated_tokens)}")
    print(f"processed_steps: {int(processed_steps)}")
    print(f"time(s): {duration:.6f}")
    print(f"processed_steps/s: {processed_steps / duration:.6f}")
    print(f"generated_tokens/s: {generated_tokens / duration:.6f}")
    print(f"TTFT (First Token Latency): {metrics['TTFT'] * 1000:.6f} ms")
    print(f"TPOT (Average Token Latency): {metrics['TPOT'] * 1000:.6f} ms")


if __name__ == "__main__":
    main()