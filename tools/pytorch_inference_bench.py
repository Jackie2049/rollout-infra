#!/usr/bin/env python3
"""
PyTorch 原生 LLM 推理性能基准测试

不依赖 vLLM 等推理引擎，直接用 PyTorch + HuggingFace Transformers 进行推理，
用于理解 LLM 推理的核心概念：KV Cache、Prefill vs Decode、Batch 推理、吞吐分析。

用法:
    python pytorch_inference_bench.py
    python pytorch_inference_bench.py --model gpt2 --max-tokens 64 --batch-sizes 1 4 8
"""

import argparse
import time
import json
import gc
from pathlib import Path

import torch
import numpy as np


def get_device_info():
    """获取 GPU 设备信息"""
    if not torch.cuda.is_available():
        return {"device": "CPU", "cuda_available": False}

    props = torch.cuda.get_device_properties(0)
    return {
        "device": props.name,
        "cuda_available": True,
        "total_memory_gb": props.total_memory / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
    }


def estimate_model_memory(model_name, dtype=torch.float16):
    """估算模型显存占用"""
    bytes_per_param = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
    known_models = {
        "gpt2": (124_439_808, 1024, 12, 768, 12, 50257),       # params, max_seq, layers, hidden, heads, vocab
        "gpt2-medium": (354_823_168, 1024, 24, 1024, 16, 50257),
        "gpt2-large": (774_030_080, 1024, 36, 1280, 20, 50257),
        "facebook/opt-125m": (125_239_808, 2048, 12, 768, 12, 50272),
        "facebook/opt-350m": (331_196_416, 2048, 24, 1024, 16, 50272),
    }

    if model_name not in known_models:
        return None

    params, max_seq, layers, hidden, heads, vocab = known_models[model_name]
    bpb = bytes_per_param.get(dtype, 2)

    model_mem = params * bpb  # 模型权重
    # KV Cache: 2 * layers * 2 * batch * seq * hidden * dtype_bytes
    kv_per_token = 2 * layers * 2 * hidden * bpb  # per token KV cache bytes

    return {
        "params": params,
        "params_M": params / 1e6,
        "model_memory_MB": model_mem / 1e6,
        "kv_cache_per_token_KB": kv_per_token / 1e3,
        "max_seq": max_seq,
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
    }


class KVCacheManager:
    """手动管理 KV Cache，演示 KV Cache 的工作原理"""

    def __init__(self, num_layers, num_heads, head_dim, max_batch_size, max_seq_len, dtype=torch.float16, device="cuda"):
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # 为每一层分配 KV Cache
        # Shape: [max_batch, num_heads, max_seq_len, head_dim]
        self.k_cache = []
        self.v_cache = []
        for _ in range(num_layers):
            k = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim,
                          dtype=dtype, device=device)
            v = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim,
                          dtype=dtype, device=device)
            self.k_cache.append(k)
            self.v_cache.append(v)

        self.current_seq_len = 0

    def update(self, layer_idx, new_k, new_v, start_pos):
        """更新 KV Cache 并返回完整的 key, value"""
        batch_size, num_heads, seq_len, head_dim = new_k.shape

        # 写入新 KV 到 cache
        self.k_cache[layer_idx][:, :, start_pos:start_pos + seq_len, :] = new_k
        self.v_cache[layer_idx][:, :, start_pos:start_pos + seq_len, :] = new_v

        # 返回从 0 到当前位置的完整 KV
        end_pos = start_pos + seq_len
        return (
            self.k_cache[layer_idx][:, :, :end_pos, :],
            self.v_cache[layer_idx][:, :, :end_pos, :],
        )

    def memory_usage_mb(self):
        """KV Cache 占用的显存"""
        total = sum(k.nelement() + v.nelement() for k, v in zip(self.k_cache, self.v_cache))
        bytes_per_elem = {torch.float16: 2, torch.float32: 4, torch.bfloat16: 2}
        return total * bytes_per_elem.get(self.dtype, 2) / 1e6


def benchmark_prefill_decode(model_name, max_new_tokens=64, batch_size=1, device="cuda"):
    """
    分别测量 Prefill 和 Decode 阶段的延迟
    展示 KV Cache 带来的加速效果
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Prefill vs Decode Benchmark")
    print(f"  Model: {model_name}, Batch: {batch_size}, MaxTokens: {max_new_tokens}")
    print(f"{'='*60}")

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备输入
    prompts = ["The key to good software engineering is"] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    print(f"  Input length: {input_len} tokens")

    # === 方法 1: 不使用 KV Cache (每步重新计算) ===
    print(f"\n  --- Without KV Cache ---")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()

    generated = inputs["input_ids"].clone()
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(generated, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_no_cache = time.time() - t0

    # === 方法 2: 使用 KV Cache ===
    print(f"  --- With KV Cache ---")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()

    past_key_values = None
    generated_cache = inputs["input_ids"].clone()
    with torch.no_grad():
        # Prefill 阶段
        outputs = model(generated_cache, use_cache=True)
        past_key_values = outputs.past_key_values

        # Decode 阶段
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_cache = torch.cat([generated_cache, next_token], dim=1)

        decode_times = []
        for step in range(max_new_tokens - 1):
            t_step = time.time()
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_cache = torch.cat([generated_cache, next_token], dim=1)
            decode_times.append(time.time() - t_step)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_with_cache = time.time() - t0

    # 计算吞吐量
    total_tokens = batch_size * max_new_tokens
    no_cache_tps = total_tokens / t_no_cache
    with_cache_tps = total_tokens / t_with_cache
    avg_decode_latency = np.mean(decode_times) * 1000 if decode_times else 0

    print(f"\n  Results:")
    print(f"    Without KV Cache: {t_no_cache:.3f}s, {no_cache_tps:.1f} tok/s")
    print(f"    With KV Cache:    {t_with_cache:.3f}s, {with_cache_tps:.1f} tok/s")
    print(f"    Speedup:          {t_no_cache/t_with_cache:.2f}x")
    print(f"    Avg decode step:  {avg_decode_latency:.2f} ms")
    print(f"    Decode TPS:       {batch_size/np.mean(decode_times):.1f} tok/s" if decode_times else "")

    # 输出样例
    sample = tokenizer.decode(generated_cache[0], skip_special_tokens=True)
    print(f"\n  Sample output:")
    print(f"    {sample[:200]}...")

    # 释放显存
    del model, past_key_values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "max_new_tokens": max_new_tokens,
        "no_cache_time": t_no_cache,
        "with_cache_time": t_with_cache,
        "speedup": t_no_cache / t_with_cache,
        "with_cache_tps": with_cache_tps,
        "avg_decode_ms": avg_decode_latency,
    }


def benchmark_batch_scaling(model_name, max_new_tokens=32, batch_sizes=None):
    """
    测量不同 batch size 下的吞吐量变化
    展示 GPU 利用率如何随 batch size 提升
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    print(f"\n{'='*60}")
    print(f"  Batch Scaling Benchmark")
    print(f"  Model: {model_name}, Tokens: {max_new_tokens}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    prompt_text = "The meaning of life is"
    results = []

    for bs in batch_sizes:
        prompts = [prompt_text] * bs
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
            model.device
        )

        # Warmup
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, use_cache=True)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                use_cache=True, do_sample=False,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - t0

        total_tokens = bs * max_new_tokens
        tps = total_tokens / elapsed

        result = {
            "batch_size": bs,
            "time": elapsed,
            "total_tokens": total_tokens,
            "tok_per_sec": tps,
        }
        results.append(result)
        print(f"  Batch={bs}: {elapsed:.3f}s, {tps:.1f} tok/s")

    # 分析扩展效率
    if len(results) > 1:
        base_tps = results[0]["tok_per_sec"]
        print(f"\n  Scaling Efficiency:")
        for r in results:
            linear = r["batch_size"] * base_tps
            actual = r["tok_per_sec"]
            eff = actual / linear * 100 if linear > 0 else 0
            print(f"    Batch {r['batch_size']}: "
                  f"actual={actual:.1f} vs linear={linear:.1f}, "
                  f"efficiency={eff:.1f}%")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def analyze_kv_cache(model_name, seq_lengths=None):
    """
    分析不同序列长度下的 KV Cache 显存占用
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"\n{'='*60}")
    print(f"  KV Cache Memory Analysis: {model_name}")
    print(f"{'='*60}")

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.n_layer if hasattr(config, 'n_layer') else config.num_hidden_layers
    hidden = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
    num_heads = config.n_head if hasattr(config, 'n_head') else config.num_attention_heads
    head_dim = hidden // num_heads

    print(f"  Config: layers={num_layers}, hidden={hidden}, heads={num_heads}, head_dim={head_dim}")
    print(f"\n  {'Seq Len':>8} {'KV Cache (MB)':>14} {'per token (KB)':>15} {'Model+KV (MB)':>14}")
    print(f"  {'-'*55}")

    model_mem = sum(p.nelement() * 2 for p in
                    AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).parameters()) / 1e6

    for seq_len in seq_lengths:
        # KV Cache = 2 * layers * batch * seq * hidden * 2 bytes (FP16)
        kv_per_seq = 2 * num_layers * 1 * seq_len * hidden * 2  # bytes
        kv_mb = kv_per_seq / 1e6
        per_token_kb = kv_per_seq / seq_len / 1e3
        total_mb = model_mem + kv_mb

        print(f"  {seq_len:>8} {kv_mb:>14.1f} {per_token_kb:>15.1f} {total_mb:>14.1f}")

    return {
        "model": model_name,
        "num_layers": num_layers,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "model_memory_mb": model_mem,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch 原生 LLM 推理基准测试")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="模型名称 (gpt2, facebook/opt-125m 等)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 设备信息
    info = get_device_info()
    print(f"\nDevice: {info}")

    if info.get("cuda_available"):
        print(f"  Memory: {info['total_memory_gb']:.1f} GB")
        print(f"  CUDA: {info['cuda_version']}, SM: {info['compute_capability']}")

    # 模型估算
    est = estimate_model_memory(args.model)
    if est:
        print(f"\nModel Estimate ({args.model}):")
        print(f"  Parameters: {est['params_M']:.1f}M")
        print(f"  Model memory: {est['model_memory_MB']:.1f} MB")
        print(f"  KV Cache/token: {est['kv_cache_per_token_KB']:.1f} KB")

    # KV Cache 分析
    analyze_kv_cache(args.model)

    # Prefill vs Decode
    benchmark_prefill_decode(args.model, args.max_tokens, batch_size=1, device=device)

    # Batch Scaling
    benchmark_batch_scaling(args.model, max_new_tokens=32, batch_sizes=args.batch_sizes)

    print(f"\n{'='*60}")
    print(f"  Benchmark Complete!")
    print(f"{'='*60}")
