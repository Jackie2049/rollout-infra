#!/usr/bin/env python3
"""vLLM GPU Benchmark 脚本

在 GPU 服务器上运行，测试 vLLM 不同配置下的性能:
1. max_model_len 对显存/吞吐的影响
2. 不同模型 (OPT-125M vs OPT-350M)
3. Batch size 对吞吐的影响
4. gpu_memory_utilization 对 KV Cache 的影响

用法 (在 GPU 服务器上):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python gpu_bench_vllm.py
"""

import time
import json
import os
import gc

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from vllm import LLM, SamplingParams


def run_benchmark(model_name, max_len, gpu_util=0.9, num_prompts=8, max_tokens=32):
    """运行单个 benchmark 配置"""
    torch.cuda.empty_cache()
    gc.collect()

    llm = LLM(
        model=model_name,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
    )

    engine = llm.llm_engine
    cache_config = engine.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size

    gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
    gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9

    prompts = ["The key to understanding artificial intelligence is"] * num_prompts
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0)

    # Warmup
    _ = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0))

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    avg_latency = elapsed / num_prompts * 1000

    result = {
        "model": model_name,
        "max_len": max_len,
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": block_size,
        "max_concurrent": (num_gpu_blocks * block_size) / max_len if max_len > 0 else 0,
        "gpu_mem_allocated_gb": round(gpu_mem_allocated, 2),
        "gpu_mem_reserved_gb": round(gpu_mem_reserved, 2),
        "time_s": round(elapsed, 3),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "avg_latency_ms": round(avg_latency, 1),
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 70)
    print("vLLM GPU Benchmark")
    print("=" * 70)

    # GPU 信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}, {gpu_total:.1f} GB")

    all_results = {}

    # 实验1: OPT-125M, 不同 max_model_len
    print("\n--- 实验1: OPT-125M max_model_len 影响 ---")
    exp1 = []
    for max_len in [256, 512]:
        try:
            r = run_benchmark("facebook/opt-125m", max_len)
            exp1.append(r)
            print(f"  max_len={max_len}: blocks={r['num_gpu_blocks']}, "
                  f"max_concurrent={r['max_concurrent']:.0f}, "
                  f"mem={r['gpu_mem_reserved_gb']:.2f}GB, "
                  f"throughput={r['throughput_tok_s']:.0f} tok/s, "
                  f"latency={r['avg_latency_ms']:.1f}ms")
        except Exception as e:
            print(f"  max_len={max_len}: FAILED - {e}")
            torch.cuda.empty_cache()
    all_results["exp1_max_model_len"] = exp1

    # 实验2: OPT-350M
    print("\n--- 实验2: OPT-350M (max_len=512) ---")
    exp2 = []
    try:
        r = run_benchmark("facebook/opt-350m", 512)
        exp2.append(r)
        print(f"  blocks={r['num_gpu_blocks']}, max_concurrent={r['max_concurrent']:.0f}, "
              f"mem={r['gpu_mem_reserved_gb']:.2f}GB, "
              f"throughput={r['throughput_tok_s']:.0f} tok/s, "
              f"latency={r['avg_latency_ms']:.1f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
        torch.cuda.empty_cache()
    all_results["exp2_opt350m"] = exp2

    # 实验3: 不同 batch size (OPT-125M, max_len=512)
    print("\n--- 实验3: Batch size 影响 (OPT-125M, max_len=512) ---")
    exp3 = []
    for bs in [1, 4, 8, 16, 32, 64]:
        try:
            r = run_benchmark("facebook/opt-125m", 512, num_prompts=bs, max_tokens=32)
            exp3.append(r)
            print(f"  batch={bs}: throughput={r['throughput_tok_s']:.0f} tok/s, "
                  f"latency={r['avg_latency_ms']:.1f}ms")
        except Exception as e:
            print(f"  batch={bs}: FAILED - {e}")
            torch.cuda.empty_cache()
    all_results["exp3_batch_size"] = exp3

    # 实验4: gpu_memory_utilization (OPT-125M, max_len=512, batch=16)
    print("\n--- 实验4: gpu_memory_utilization 影响 (OPT-125M, max_len=512) ---")
    exp4 = []
    for util in [0.7, 0.8, 0.9, 0.95]:
        try:
            r = run_benchmark("facebook/opt-125m", 512, gpu_util=util, num_prompts=16, max_tokens=32)
            exp4.append(r)
            print(f"  util={util}: blocks={r['num_gpu_blocks']}, "
                  f"max_concurrent={r['max_concurrent']:.0f}, "
                  f"throughput={r['throughput_tok_s']:.0f} tok/s")
        except Exception as e:
            print(f"  util={util}: FAILED - {e}")
            torch.cuda.empty_cache()
    all_results["exp4_gpu_util"] = exp4

    # 实验5: 生成长度影响 (OPT-125M, max_len=512, batch=8)
    print("\n--- 实验5: 生成长度影响 (OPT-125M, max_len=512, batch=8) ---")
    exp5 = []
    for gen_len in [16, 32, 64, 128]:
        try:
            r = run_benchmark("facebook/opt-125m", 512, num_prompts=8, max_tokens=gen_len)
            exp5.append(r)
            print(f"  gen_len={gen_len}: throughput={r['throughput_tok_s']:.0f} tok/s, "
                  f"latency={r['avg_latency_ms']:.1f}ms, "
                  f"total_tokens={r['total_tokens']}")
        except Exception as e:
            print(f"  gen_len={gen_len}: FAILED - {e}")
            torch.cuda.empty_cache()
    all_results["exp5_gen_length"] = exp5

    # 保存结果
    with open("vllm_bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("实验完成! 结果保存到 vllm_bench_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
