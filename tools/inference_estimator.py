#!/usr/bin/env python3
"""
LLM 推理性能估算器

根据模型规格和硬件配置，估算推理吞吐量、延迟和显存需求。
覆盖 prefill/decode 阶段分析、KV Cache 显存、batch size 上限等。

用法:
    python inference_estimator.py --model llama-7b --gpu a100-80gb
    python inference_estimator.py --params 70 --hidden 8192 --layers 80 --kv-heads 8 --gpu a100-80gb
"""

import argparse
import math
import sys
from dataclasses import dataclass


# GPU 规格
GPU_SPECS = {
    "a100-40gb": {"memory_gb": 40, "fp16_tflops": 312, "hbm_bw_gbs": 1555, "nvlink_bw_gbs": 600},
    "a100-80gb": {"memory_gb": 80, "fp16_tflops": 312, "hbm_bw_gbs": 2039, "nvlink_bw_gbs": 600},
    "h100-sxm":  {"memory_gb": 80, "fp16_tflops": 990, "hbm_bw_gbs": 3350, "nvlink_bw_gbs": 900},
    "h200":      {"memory_gb": 141, "fp16_tflops": 990, "hbm_bw_gbs": 4800, "nvlink_bw_gbs": 900},
    "a16":       {"memory_gb": 16, "fp16_tflops": 14.7, "hbm_bw_gbs": 300, "nvlink_bw_gbs": 0},
    "l40s":      {"memory_gb": 48, "fp16_tflops": 362, "hbm_bw_gbs": 864, "nvlink_bw_gbs": 0},
}

# 模型规格
MODEL_SPECS = {
    "llama-7b":   {"params_b": 7, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 32, "vocab": 32000},
    "llama2-70b": {"params_b": 70, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8,  "vocab": 32000},
    "llama3-8b":  {"params_b": 8, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8,  "vocab": 128256},
    "llama3-70b": {"params_b": 70, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8,  "vocab": 128256},
    "qwen-72b":   {"params_b": 72, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8,  "vocab": 151936},
    "mistral-7b": {"params_b": 7, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 1,  "vocab": 32000},
}


@dataclass
class EstimateResult:
    model_name: str
    gpu_name: str
    num_gpus: int
    dtype_bytes: int

    # Model memory
    model_memory_gb: float
    model_memory_per_gpu_gb: float

    # KV Cache
    kv_per_token_per_layer_kb: float
    kv_per_token_total_kb: float
    max_kv_cache_gb: float
    max_tokens_in_kv: int
    max_seq_per_batch: int

    # Performance
    prefill_time_ms: float
    decode_time_per_token_ms: float
    decode_bandwidth_limited_tok_s: float

    # Batch analysis
    max_batch_size: int
    throughput_tok_s: float


def estimate_inference(
    params_b: float,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    vocab: int,
    gpu_spec: dict,
    num_gpus: int = 1,
    seq_len: int = 2048,
    dtype: str = "fp16",
    gpu_util: float = 0.9,
    max_batch: int = 0,
) -> EstimateResult:
    """估算推理性能"""

    dtype_bytes = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}[dtype]

    # 1. 模型权重显存
    model_memory_gb = params_b * 1e9 * dtype_bytes / (1024**3)
    model_memory_per_gpu_gb = model_memory_gb / num_gpus

    # 2. KV Cache
    head_dim = hidden // heads
    kv_per_token_per_layer = 2 * kv_heads * head_dim * dtype_bytes  # bytes
    kv_per_token_total = kv_per_token_per_layer * layers  # bytes

    available_for_kv = (gpu_spec["memory_gb"] * gpu_util * num_gpus) - model_memory_gb
    if available_for_kv < 0:
        available_for_kv = 0

    max_tokens_in_kv = int(available_for_kv * (1024**3) / kv_per_token_total) if kv_per_token_total > 0 else 0
    max_seq_per_batch = max_tokens_in_kv // seq_len if seq_len > 0 else 0
    if max_batch > 0:
        max_seq_per_batch = min(max_seq_per_batch, max_batch)

    kv_total_gb = max_seq_per_batch * seq_len * kv_per_token_total / (1024**3)

    # 3. Prefill 性能 (compute-bound)
    # FLOPs ≈ 2 × params × seq_len (forward pass for each token)
    prefill_flops = 2 * params_b * 1e9 * seq_len
    prefill_flops_per_gpu = prefill_flops / num_gpus
    # 假设 50% 峰值利用率 (attention 不完全受限于矩阵乘法)
    prefill_time_s = prefill_flops_per_gpu / (gpu_spec["fp16_tflops"] * 1e12 * 0.5)
    prefill_time_ms = prefill_time_s * 1000

    # 4. Decode 性能 (memory-bandwidth-bound)
    # 每步读模型权重 + KV Cache
    decode_bytes_per_step = model_memory_gb * (1024**3) + max_seq_per_batch * seq_len * kv_per_token_total
    total_bandwidth_gbs = gpu_spec["hbm_bw_gbs"] * num_gpus
    # 假设 60% 带宽利用率
    decode_time_s = decode_bytes_per_step / (total_bandwidth_gbs * 1e9 * 0.6)
    decode_time_per_token_ms = decode_time_s * 1000

    # 理论 decode 吞吐 (bandwidth-limited)
    decode_bandwidth_limited = total_bandwidth_gbs * 1e9 * 0.6 / (model_memory_gb * (1024**3)) if model_memory_gb > 0 else 0

    throughput = max_seq_per_batch / (decode_time_s) if decode_time_s > 0 else 0

    return EstimateResult(
        model_name=f"{params_b}B",
        gpu_name=list(GPU_SPECS.keys())[list(GPU_SPECS.values()).index(gpu_spec)] if gpu_spec in GPU_SPECS.values() else "custom",
        num_gpus=num_gpus,
        dtype_bytes=dtype_bytes,
        model_memory_gb=model_memory_gb,
        model_memory_per_gpu_gb=model_memory_per_gpu_gb,
        kv_per_token_per_layer_kb=kv_per_token_per_layer / 1024,
        kv_per_token_total_kb=kv_per_token_total / 1024,
        max_kv_cache_gb=available_for_kv,
        max_tokens_in_kv=max_tokens_in_kv,
        max_seq_per_batch=max_seq_per_batch,
        prefill_time_ms=prefill_time_ms,
        decode_time_per_token_ms=decode_time_per_token_ms,
        decode_bandwidth_limited_tok_s=decode_bandwidth_limited,
        max_batch_size=max_seq_per_batch,
        throughput_tok_s=throughput,
    )


def print_result(r: EstimateResult, seq_len: int, dtype: str):
    """格式化输出结果"""
    print(f"\n{'='*60}")
    print(f"  LLM 推理性能估算")
    print(f"{'='*60}")
    print(f"\n  模型: {r.model_name} ({dtype})")
    print(f"  硬件: {r.gpu_name} × {r.num_gpus}")
    print(f"  序列长度: {seq_len}")

    print(f"\n--- 显存分析 ---")
    print(f"  模型权重:    {r.model_memory_gb:.1f} GB ({r.model_memory_per_gpu_gb:.1f} GB/GPU)")
    print(f"  KV Cache 可用: {r.max_kv_cache_gb:.1f} GB")
    print(f"  每 token KV:  {r.kv_per_token_total_kb:.1f} KB ({r.kv_per_token_per_layer_kb:.2f} KB/layer)")
    print(f"  KV 最大 tokens: {r.max_tokens_in_kv:,}")
    print(f"  最大 batch:   {r.max_batch_size} (seq_len={seq_len})")

    print(f"\n--- 性能估算 ---")
    print(f"  Prefill:       {r.prefill_time_ms:.1f} ms (seq_len={seq_len})")
    print(f"  Decode:        {r.decode_time_per_token_ms:.2f} ms/token")
    print(f"  Decode 吞吐:   {r.throughput_tok_s:.0f} tokens/s (batch={r.max_batch_size})")
    print(f"  单流吞吐:      {1000/r.decode_time_per_token_ms:.0f} tokens/s (bs=1, 不含KV读取)")
    print(f"  带宽极限:      {r.decode_bandwidth_limited_tok_s:.0f} tokens/s (理论)")

    print(f"\n--- 单请求延迟 (bs=1) ---")
    prompt_len = seq_len
    gen_len = 256
    single_decode_ms = r.model_memory_gb * (1024**3) / (30 * 1e9 * 0.6) * 1000  # approximate
    total_ms = r.prefill_time_ms + single_decode_ms * gen_len
    print(f"  Prompt: {prompt_len} tokens → Prefill {r.prefill_time_ms:.1f} ms")
    print(f"  Generate: {gen_len} tokens → Decode {single_decode_ms:.1f} ms × {gen_len} = {single_decode_ms*gen_len:.0f} ms")
    print(f"  总延迟: {total_ms/1000:.1f} s")


def compare_gpus(model_key: str, seq_len: int = 2048):
    """对比不同 GPU 上同一模型的推理性能"""
    model = MODEL_SPECS[model_key]
    print(f"\n{'='*70}")
    print(f"  GPU 对比: {model_key} (seq_len={seq_len})")
    print(f"{'='*70}")
    print(f"{'GPU':<12} {'模型/GPU':<10} {'KV可用':<10} {'最大Batch':<10} {'吞吐tok/s':<12} {'Decode ms':<10}")
    print(f"{'-'*70}")

    for gpu_name, gpu_spec in GPU_SPECS.items():
        num_gpus = max(1, math.ceil(model["params_b"] * 2 / (gpu_spec["memory_gb"] * 0.9)))
        r = estimate_inference(
            params_b=model["params_b"],
            hidden=model["hidden"],
            layers=model["layers"],
            heads=model["heads"],
            kv_heads=model["kv_heads"],
            vocab=model["vocab"],
            gpu_spec=gpu_spec,
            num_gpus=num_gpus,
            seq_len=seq_len,
        )
        print(f"{gpu_name:<12} {r.model_memory_per_gpu_gb:<10.1f} {r.max_kv_cache_gb:<10.1f} {r.max_batch_size:<10} {r.throughput_tok_s:<12.0f} {r.decode_time_per_token_ms:<10.2f}")


def compare_models(gpu_key: str, seq_len: int = 2048):
    """对比同一 GPU 上不同模型的推理性能"""
    gpu_spec = GPU_SPECS[gpu_key]
    print(f"\n{'='*80}")
    print(f"  模型对比: {gpu_key} (seq_len={seq_len})")
    print(f"{'='*80}")
    print(f"{'模型':<14} {'GPU数':<6} {'KV/token':<10} {'最大Batch':<10} {'吞吐tok/s':<12} {'Prefill ms':<10}")
    print(f"{'-'*80}")

    for model_name, model in MODEL_SPECS.items():
        num_gpus = max(1, math.ceil(model["params_b"] * 2 / (gpu_spec["memory_gb"] * 0.9)))
        if num_gpus > 8:
            continue  # 跳过需要太多 GPU 的组合
        r = estimate_inference(
            params_b=model["params_b"],
            hidden=model["hidden"],
            layers=model["layers"],
            heads=model["heads"],
            kv_heads=model["kv_heads"],
            vocab=model["vocab"],
            gpu_spec=gpu_spec,
            num_gpus=num_gpus,
            seq_len=seq_len,
        )
        print(f"{model_name:<14} {num_gpus:<6} {r.kv_per_token_total_kb:<10.1f} {r.max_batch_size:<10} {r.throughput_tok_s:<12.0f} {r.prefill_time_ms:<10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 推理性能估算器")
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), help="预定义模型")
    parser.add_argument("--gpu", choices=list(GPU_SPECS.keys()), help="GPU 类型")
    parser.add_argument("--num-gpus", type=int, default=0, help="GPU 数量 (0=自动计算)")
    parser.add_argument("--seq-len", type=int, default=2048, help="序列长度")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8", "int4"])
    parser.add_argument("--compare-gpus", action="store_true", help="对比不同 GPU")
    parser.add_argument("--compare-models", action="store_true", help="对比不同模型")

    # 自定义模型参数
    parser.add_argument("--params", type=float, help="模型参数量 (B)")
    parser.add_argument("--hidden", type=int, help="Hidden dim")
    parser.add_argument("--layers", type=int, help="层数")
    parser.add_argument("--heads", type=int, help="Q heads")
    parser.add_argument("--kv-heads", type=int, help="KV heads")
    parser.add_argument("--vocab", type=int, default=32000, help="词表大小")

    args = parser.parse_args()

    if args.compare_gpus and args.model:
        compare_gpus(args.model, args.seq_len)
    elif args.compare_models and args.gpu:
        compare_models(args.gpu, args.seq_len)
    elif args.params and args.gpu:
        gpu_spec = GPU_SPECS[args.gpu]
        num_gpus = args.num_gpus or max(1, math.ceil(args.params * 2 / (gpu_spec["memory_gb"] * 0.9)))
        r = estimate_inference(
            params_b=args.params, hidden=args.hidden, layers=args.layers,
            heads=args.heads, kv_heads=args.kv_heads or args.heads,
            vocab=args.vocab, gpu_spec=gpu_spec, num_gpus=num_gpus,
            seq_len=args.seq_len, dtype=args.dtype,
        )
        print_result(r, args.seq_len, args.dtype)
    elif args.model and args.gpu:
        model = MODEL_SPECS[args.model]
        gpu_spec = GPU_SPECS[args.gpu]
        num_gpus = args.num_gpus or max(1, math.ceil(model["params_b"] * 2 / (gpu_spec["memory_gb"] * 0.9)))
        r = estimate_inference(
            params_b=model["params_b"], hidden=model["hidden"], layers=model["layers"],
            heads=model["heads"], kv_heads=model["kv_heads"], vocab=model["vocab"],
            gpu_spec=gpu_spec, num_gpus=num_gpus, seq_len=args.seq_len, dtype=args.dtype,
        )
        print_result(r, args.seq_len, args.dtype)
    else:
        # 默认：演示模式
        print("LLM 推理性能估算器")
        print("=" * 40)
        print("\n示例用法:")
        print("  python inference_estimator.py --model llama-7b --gpu a100-80gb")
        print("  python inference_estimator.py --model llama2-70b --gpu h100-sxm")
        print("  python inference_estimator.py --model llama-7b --gpu a100-80gb --compare-gpus")
        print("  python inference_estimator.py --gpu a100-80gb --compare-models")
        print()
        print("默认演示: llama-7b on A100-80GB\n")
        model = MODEL_SPECS["llama-7b"]
        gpu_spec = GPU_SPECS["a100-80gb"]
        r = estimate_inference(
            params_b=model["params_b"], hidden=model["hidden"], layers=model["layers"],
            heads=model["heads"], kv_heads=model["kv_heads"], vocab=model["vocab"],
            gpu_spec=gpu_spec, num_gpus=1, seq_len=2048,
        )
        print_result(r, 2048, "fp16")
        print("\n")
        compare_models("a100-80gb", 2048)
