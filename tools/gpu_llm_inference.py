#!/usr/bin/env python3
"""LLM 推理全流程 Benchmark — OPT-125M/350M on A16

5 个实验:
1. Prefill vs Decode 延迟对比
2. Batch Size 吞吐量曲线
3. KV Cache 内存实测
4. Continuous Batching 模拟
5. Speculative Decoding 模拟

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  python gpu_llm_inference.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")
print(f"CUDA: {torch.version.cuda}")


def bench_ms(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def load_opt_model(model_name="facebook/opt-125m"):
    """Load OPT model (offline)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True,
            torch_dtype=torch.float16, device_map="cuda",
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        return None, None


# ============================================================
# 实验 1: Prefill vs Decode 延迟
# ============================================================

def exp1_prefill_decode(model, tokenizer, model_name=""):
    print("\n" + "=" * 60)
    print(f"实验1: Prefill vs Decode ({model_name})")
    print("=" * 60)

    results = []

    # Prefill: process entire prompt at once
    prompt_lengths = [32, 64, 128, 256, 512, 1024]

    print(f"\n  Prefill (单次前向, 无 KV cache 复用):")
    print(f"  {'Prompt Len':<14} {'Time ms':<12} {'Tokens/s':<12} {'Mem MB':<10}")
    print("  " + "-" * 48)

    for plen in prompt_lengths:
        input_ids = torch.randint(100, 5000, (1, plen), device="cuda")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        ms = bench_ms(lambda: model(input_ids), warmup=2, rep=10)
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        toks = plen / ms * 1000

        print(f"  {plen:<14} {ms:<12.2f} {toks:<12.0f} {mem_mb:<10.0f}")
        results.append({
            "prompt_len": plen, "prefill_ms": round(ms, 2),
            "tokens_per_s": round(toks, 0), "mem_mb": round(mem_mb, 0),
        })
        del input_ids
        torch.cuda.empty_cache()

    # Decode: generate one token at a time (with KV cache)
    print(f"\n  Decode (逐 token, 使用 KV cache):")
    print(f"  {'Batch':<8} {'Time/token ms':<14} {'Tokens/s':<12} {'Mem MB':<10}")
    print("  " + "-" * 44)

    for batch in [1, 4, 8, 16, 32, 64]:
        plen = 64
        input_ids = torch.randint(100, 5000, (batch, plen), device="cuda")

        try:
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            torch.cuda.reset_peak_memory_stats()

            ms = bench_ms(
                lambda: model(next_token, past_key_values=past_kv, use_cache=True),
                warmup=2, rep=20
            )
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            toks = batch / ms * 1000

            print(f"  {batch:<8} {ms:<14.3f} {toks:<12.0f} {mem_mb:<10.0f}")
            results.append({
                "batch": batch, "decode_ms": round(ms, 3),
                "tokens_per_s": round(toks, 0), "mem_mb": round(mem_mb, 0),
            })

            del past_kv, out, next_token
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  {batch:<8} OOM")
            torch.cuda.empty_cache()
            break

    return results


# ============================================================
# 实验 2: Batch Size 吞吐量曲线
# ============================================================

def exp2_batch_throughput(model, model_name=""):
    print("\n" + "=" * 60)
    print(f"实验2: Batch Throughput ({model_name})")
    print("=" * 60)

    results = []
    seq_len = 128

    # Baseline for efficiency calc
    input_ids_1 = torch.randint(100, 5000, (1, seq_len), device="cuda")
    baseline_ms = bench_ms(lambda: model(input_ids_1), warmup=2, rep=5)
    baseline_tps = seq_len / baseline_ms * 1000
    del input_ids_1

    print(f"\n  Prefill throughput (seq={seq_len}):")
    print(f"  {'Batch':<8} {'Time ms':<12} {'Throughput tok/s':<18} {'Efficiency'}")
    print("  " + "-" * 50)

    for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
        try:
            input_ids = torch.randint(100, 5000, (batch, seq_len), device="cuda")

            ms = bench_ms(lambda: model(input_ids), warmup=2, rep=10)
            throughput = batch * seq_len / ms * 1000
            eff = throughput / (batch * baseline_tps) * 100

            print(f"  {batch:<8} {ms:<12.2f} {throughput:<18.0f} {eff:.0f}%")
            results.append({
                "batch": batch, "ms": round(ms, 2),
                "throughput": round(throughput, 0), "efficiency_pct": round(eff, 0),
            })

            del input_ids
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  {batch:<8} OOM")
            torch.cuda.empty_cache()
            break

    return results


# ============================================================
# 实验 3: KV Cache 内存实测
# ============================================================

def exp3_kv_cache_memory(model, model_name=""):
    print("\n" + "=" * 60)
    print(f"实验3: KV Cache Memory ({model_name})")
    print("=" * 60)

    results = []

    if hasattr(model, 'config'):
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads
        print(f"\n  Config: {n_layers} layers, {n_heads} heads, head_dim={head_dim}")

    print(f"\n  KV Cache 内存 (batch=1):")
    print(f"  {'Seq Len':<10} {'KV Size MB':<14} {'Bytes/token':<14} {'Theory'}")
    print("  " + "-" * 55)

    for seq_len in [64, 128, 256, 512, 1024, 2048]:
        try:
            input_ids = torch.randint(100, 5000, (1, seq_len), device="cuda")

            base_mem = torch.cuda.memory_allocated()

            with torch.no_grad():
                out = model(input_ids, use_cache=True)

            kv_mem = torch.cuda.memory_allocated() - base_mem
            kv_mb = kv_mem / 1e6
            bytes_per_token = kv_mem / seq_len
            theory_bytes = 2 * n_layers * n_heads * head_dim * 2  # K+V, FP16

            print(f"  {seq_len:<10} {kv_mb:<14.1f} {bytes_per_token:<14.0f} {theory_bytes}")
            results.append({
                "seq_len": seq_len, "kv_mb": round(kv_mb, 1),
                "bytes_per_token": int(bytes_per_token), "theory_bytes": theory_bytes,
            })

            del input_ids, out
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  {seq_len:<10} OOM")
            torch.cuda.empty_cache()
            break

    return results


# ============================================================
# 实验 4: Continuous Batching 模拟
# ============================================================

def exp4_continuous_batching(model, model_name=""):
    print("\n" + "=" * 60)
    print(f"实验4: Continuous Batching ({model_name})")
    print("=" * 60)

    results = []
    max_batch = 32
    prompt_len = 64

    input_ids = torch.randint(100, 5000, (max_batch, prompt_len), device="cuda")

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Full batch baseline
    full_ms = bench_ms(
        lambda: model(tokens, past_key_values=past_kv, use_cache=True),
        warmup=3, rep=50
    )
    full_tps = max_batch / full_ms * 1000

    print(f"\n  Full batch ({max_batch}) decode: {full_ms:.3f} ms ({full_tps:.0f} tok/s)")
    print(f"\n  {'Active Reqs':<14} {'Time ms':<12} {'Throughput':<14} {'Utilization'}")
    print("  " + "-" * 50)

    for active in [1, 2, 4, 8, 16, 32]:
        if active > max_batch:
            break

        active_past = tuple(
            (kv[0][:active], kv[1][:active]) for kv in past_kv
        )
        active_tokens = tokens[:active]

        ms = bench_ms(
            lambda: model(active_tokens, past_key_values=active_past, use_cache=True),
            warmup=3, rep=30
        )
        throughput = active / ms * 1000
        util = throughput / full_tps * 100

        print(f"  {active:<14} {ms:<12.3f} {throughput:<14.0f} {util:.0f}%")
        results.append({
            "active": active, "ms": round(ms, 3),
            "throughput": round(throughput, 0), "util_pct": round(util, 0),
        })

    del input_ids, out, past_kv, tokens
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 5: Speculative Decoding 模拟
# ============================================================

def exp5_speculative_decoding(model, model_name=""):
    print("\n" + "=" * 60)
    print(f"实验5: Speculative Decoding ({model_name})")
    print("=" * 60)

    results = []
    prompt_len = 64
    input_ids = torch.randint(100, 5000, (1, prompt_len), device="cuda")

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Baseline: 1 token decode
    one_ms = bench_ms(
        lambda: model(next_tok, past_key_values=past_kv, use_cache=True),
        warmup=3, rep=50
    )

    print(f"\n  1-token decode: {one_ms:.3f} ms")
    print(f"\n  {'K (draft)':<12} {'Verify ms':<12} {'ms/token':<12} {'Speedup (all accept)'}")
    print("  " + "-" * 50)

    for K in [2, 3, 4, 5, 8]:
        # Verify K tokens: prefill-style forward
        draft_tokens = torch.randint(100, 5000, (1, K), device="cuda")
        verify_ids = torch.cat([input_ids, draft_tokens], dim=1)

        verify_ms = bench_ms(
            lambda: model(verify_ids, use_cache=True),
            warmup=2, rep=20
        )

        ms_per_tok = verify_ms / K
        speedup = one_ms / ms_per_tok

        print(f"  {K:<12} {verify_ms:<12.2f} {ms_per_tok:<12.3f} {speedup:.2f}x")
        results.append({
            "K": K, "verify_ms": round(verify_ms, 2),
            "ms_per_tok": round(ms_per_tok, 3), "speedup": round(speedup, 2),
        })

        del draft_tokens, verify_ids

    del input_ids, out, past_kv, next_tok
    torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM 推理全流程 Benchmark")
    print("=" * 60)

    all_results = OrderedDict()

    # Try OPT-125M
    model, tokenizer = load_opt_model("facebook/opt-125m")

    if model is not None:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Loaded OPT-125M: {n_params/1e6:.1f}M params")

        all_results["prefill_decode_125m"] = exp1_prefill_decode(model, tokenizer, "OPT-125M")
        all_results["batch_throughput_125m"] = exp2_batch_throughput(model, "OPT-125M")
        all_results["kv_cache_125m"] = exp3_kv_cache_memory(model, "OPT-125M")
        all_results["continuous_batching_125m"] = exp4_continuous_batching(model, "OPT-125M")
        all_results["speculative_125m"] = exp5_speculative_decoding(model, "OPT-125M")

        del model, tokenizer
        torch.cuda.empty_cache()

        # Try OPT-350M
        model2, tok2 = load_opt_model("facebook/opt-350m")
        if model2 is not None:
            n_params2 = sum(p.numel() for p in model2.parameters())
            print(f"\n  Loaded OPT-350M: {n_params2/1e6:.1f}M params")
            all_results["prefill_decode_350m"] = exp1_prefill_decode(model2, tok2, "OPT-350M")
            all_results["batch_throughput_350m"] = exp2_batch_throughput(model2, "OPT-350M")
            del model2, tok2
            torch.cuda.empty_cache()
    else:
        print("\n  Models not available (need to download first)")
        print("  huggingface-cli download facebook/opt-125m")

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. Prefill: 延迟 ∝ seq² (compute-bound), 吞吐量随 seq↑
  2. Decode: 延迟 ∝ batch (memory-bound), batch↑吞吐↑但延迟也↑
  3. KV Cache: 2 * n_layers * n_heads * head_dim * 2 bytes/tok (FP16)
  4. Continuous Batching: 小 batch 时利用率低, 大 batch 接近线性
  5. Speculative Decoding: K=3-5 最优, verify 比 K×decode 更高效
""")

    with open("/root/llm_inference_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved.")
