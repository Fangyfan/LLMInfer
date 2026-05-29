# qwen3_4b_bf16_thinking_infer.py
import argparse
import time
from typing import Dict, List, Optional, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync_device(device: torch.device) -> None:
    """
    CUDA 是异步执行的，不同步会导致 TTFT / TPOT 偏小。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_stop_ids(tokenizer) -> Set[int]:
    """
    Qwen Chat 模型通常使用 <|im_end|> 作为停止符。
    同时加入 eos_token_id，避免模型提前输出 EOS 时无法停止。
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
    解码时跳过 ChatML 特殊 token。
    注意：不跳过 <think> / </think>，因为这里要求开启思考模式，
    所以允许打印模型的 thinking 内容。
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


def build_qwen3_thinking_prompt(tokenizer, prompt: str) -> str:
    """
    使用 Qwen3 官方 chat template，并强制开启 thinking mode。
    """
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        sentence = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "当前 tokenizer.apply_chat_template 不支持 enable_thinking=True。"
            "请确认使用的是 Qwen3 系列 tokenizer，或升级 transformers。"
        ) from exc

    return sentence


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
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_manual_bf16_thinking(
    model,
    tokenizer,
    prompt: str = "What is AI?",
    total_steps: int = 2560,
    need_output: bool = True,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Dict[str, float]:
    """
    Qwen3-4B BF16 thinking mode 手动推理。

    total_steps 的含义对齐你的 C++ / FP32 版本：
    prompt 阶段也计入 steps。
    最大新 token 数近似为：

        max_new_tokens = total_steps - prompt_len + 1
    """
    device = next(model.parameters()).device

    stop_ids = build_stop_ids(tokenizer)
    skip_ids = build_skip_ids(tokenizer)

    sync_device(device)
    ttft_start = time.perf_counter()

    sentence = build_qwen3_thinking_prompt(tokenizer, prompt)

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

    if token_id in stop_ids:
        TPOT = 0.0
        decoded_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

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

    decoded_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

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
        default="./Qwen3-4B",
        help="本地 Hugging Face 格式 Qwen3-4B 模型路径，例如 ./Qwen3-4B",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is AI?",
        help="默认 prompt 为 What is AI?",
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

    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "当前 CUDA 设备不支持 BF16。"
                "原生 BF16 推理通常需要 Ampere 或更新架构 GPU。"
            )

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    print(f"Loading Qwen3-4B native BF16 model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    print(args.prompt)
    print("Qwen3-4B BF16 thinking model generating...")

    sync_device(device)
    start = time.perf_counter()

    metrics = generate_manual_bf16_thinking(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        total_steps=args.total_steps,
        need_output=not args.no_output,
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