# qwen3_4b_vllm_awq_int4_thinking_argmax_benchmark.py
"""
vLLM benchmark for local Qwen3-4B-AWQ INT4 weight-only inference.

目标：
1. 单 batch
2. 单请求
3. 本地 Qwen3-4B-AWQ 模型
4. AWQ INT4 weight-only 量化权重推理
5. 非量化张量 / 激活计算使用 FP16，默认 dtype=half
6. 开启 Qwen3 thinking mode
7. prompt 固定为: What is AI?
8. argmax / greedy decode
9. 默认 max_tokens = 1024
10. 输出 TTFT / TPOT / 总延迟 / 吞吐
"""

import os
import time
import argparse

# 尽量关闭 torch.compile，减少首次编译对 benchmark 的影响
os.environ.setdefault("VLLM_TORCH_COMPILE_LEVEL", "0")

# 使用 V0 engine，便于读取 metrics，并避免部分 V1 行为差异
os.environ.setdefault("VLLM_USE_V1", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_qwen3_thinking_prompt(tokenizer, prompt: str) -> str:
    """
    使用 Qwen3 tokenizer 的 chat_template 构造 thinking mode 输入。
    """
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "当前 tokenizer 不支持 enable_thinking=True。"
            "请确认本地模型是 Qwen3 系列，并且 transformers 版本支持 Qwen3 thinking mode。"
        ) from exc


def read_vllm_metrics(output, total_time_ms: float, output_tokens: int):
    """
    尽量从 vLLM RequestOutput.metrics 中读取 TTFT / TPOT。
    不同 vLLM 版本 metrics 字段可能略有差异，所以这里做兼容。
    """
    metrics = getattr(output, "metrics", None)

    ttft_ms = None
    decode_time_ms = None

    # 如果拿不到细粒度 metrics，就用总时延 / 输出 token 数作为近似 TPOT
    tpot_ms = total_time_ms / max(output_tokens, 1)

    if metrics is None:
        return ttft_ms, decode_time_ms, tpot_ms

    first_token_time = getattr(metrics, "first_token_time", None)
    first_scheduled_time = getattr(metrics, "first_scheduled_time", None)
    finished_time = getattr(metrics, "finished_time", None)

    if first_token_time is not None and first_scheduled_time is not None:
        ttft_ms = (first_token_time - first_scheduled_time) * 1000

    if first_token_time is not None and finished_time is not None:
        decode_time_ms = (finished_time - first_token_time) * 1000
        tpot_ms = decode_time_ms / max(output_tokens - 1, 1)

    return ttft_ms, decode_time_ms, tpot_ms


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "本地 Qwen3-4B-AWQ 模型路径，例如 "
            "/home/yifanfang/LLMInfer/models/qwen3/Qwen3-4B-AWQ"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="最大生成 token 数，默认 1024",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="warmup 次数",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM 最大上下文长度",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM 可使用的 GPU 显存比例",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        choices=["auto", "half", "float16"],
        help=(
            "AWQ 推理的非量化张量 / activation 计算 dtype。"
            "vLLM AWQ 通常推荐 half/float16，默认 half。"
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="开启后禁用 CUDA graph，使用 eager 模式",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="不打印生成文本，只打印性能指标",
    )

    args = parser.parse_args()

    prompt = "What is AI?"

    print(f"Loading local tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    formatted_prompt = build_qwen3_thinking_prompt(tokenizer, prompt)

    print(f"Loading local Qwen3-4B-AWQ model with vLLM from: {args.model}")
    print("Quantization: AWQ INT4 weight-only")
    print(f"Compute dtype: {args.dtype}")
    print("Thinking mode: enabled")
    print("Decode strategy: argmax / greedy")
    print("Batch size: 1")
    print("Request count: 1")
    print(f"Max output tokens: {args.max_tokens}")

    llm = LLM(
        model=args.model,
        tokenizer=args.model,

        # 关键点：
        # 1. AWQ 不是 dtype='int4'
        # 2. INT4 信息来自模型 config.json 里的 quantization_config
        # 3. 这里显式指定 quantization='awq'，避免 vLLM 自动识别失败
        quantization="awq",

        # AWQ 是 weight-only INT4，activation / 非量化权重仍使用 half/float16 计算
        dtype=args.dtype,

        trust_remote_code=True,

        # 单卡、单请求、单 batch 对照
        tensor_parallel_size=1,
        max_num_seqs=1,

        # 上下文长度。需要保证：
        # prompt_tokens + max_tokens <= max_model_len
        max_model_len=args.max_model_len,

        # 显存占用比例
        gpu_memory_utilization=args.gpu_memory_utilization,

        # 默认 False：允许 CUDA graph
        # 加 --enforce-eager 后禁用 CUDA graph
        enforce_eager=args.enforce_eager,
    )

    # argmax / greedy decode
    # temperature=0.0 即 greedy；这里不强行设置 top_k，避免不同 vLLM 版本对 top_k=0/-1 的兼容差异
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    print("\nRaw prompt:")
    print(prompt)

    print("\nFormatted prompt:")
    print(formatted_prompt)

    print(f"\nWarming up for {args.warmup} iterations...")
    for i in range(args.warmup):
        llm.generate([formatted_prompt], sampling_params)
        print(f"  warmup {i + 1}/{args.warmup} done")
    print("Warmup done.")

    print("\n" + "=" * 70)
    print("Benchmark: Qwen3-4B-AWQ vLLM INT4 Weight-Only Thinking Mode Argmax")
    print("=" * 70)

    start = time.perf_counter()
    outputs = llm.generate([formatted_prompt], sampling_params)
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000

    output = outputs[0]
    completion = output.outputs[0]

    generated_text = completion.text
    prompt_tokens = len(output.prompt_token_ids)
    output_tokens = len(completion.token_ids)

    ttft_ms, decode_time_ms, tpot_ms = read_vllm_metrics(
        output=output,
        total_time_ms=total_time_ms,
        output_tokens=output_tokens,
    )

    if not args.no_output:
        print("\nGenerated text:")
        print(generated_text)

    throughput = output_tokens / max(total_time_ms / 1000, 1e-9)

    print("\n" + "-" * 70)
    print("Performance Metrics")
    print("-" * 70)
    print(f"Prompt tokens:       {prompt_tokens}")
    print(f"Output tokens:       {output_tokens}")
    print(f"Total latency:       {total_time_ms:.3f} ms")

    if ttft_ms is not None:
        print(f"TTFT:                {ttft_ms:.3f} ms")
    else:
        print("TTFT:                unavailable")

    if decode_time_ms is not None:
        print(f"Decode latency:      {decode_time_ms:.3f} ms")
    else:
        print("Decode latency:      unavailable")

    print(f"TPOT:                {tpot_ms:.3f} ms/token")
    print(f"Throughput:          {throughput:.3f} tokens/s")
    print("-" * 70)


if __name__ == "__main__":
    main()