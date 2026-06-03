#!/usr/bin/env python3
"""
vLLM 推理实验脚本

在 GPU 服务器上运行 vLLM 推理 benchmark，测试不同配置下的吞吐和延迟。
要求: pip install vllm transformers

用法:
    python vllm_experiment.py --model gpt2 --max-model-len 512
    python vllm_experiment.py --model facebook/opt-1.3b --max-model-len 1024

注意: 此脚本需要在有 GPU 的环境下运行 (A16 或其他 NVIDIA GPU)
"""

import argparse
import json
import time
import sys


def check_environment():
    """检查运行环境"""
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


def benchmark_offline(model_name, max_model_len, batch_sizes, seq_lengths, max_tokens=64):
    """离线推理 benchmark"""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"  vLLM Offline Benchmark")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Max model len: {max_model_len}")

    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    # 准备 prompts
    prompts = ["The meaning of life is"] * max(batch_sizes)

    print(f"\n{'bs':<6} {'seq':<8} {'gen':<6} {'total_tokens':<14} {'time_s':<10} {'tok/s':<10}")
    print("-" * 60)

    for bs in batch_sizes:
        for seq_len in seq_lengths:
            # 用 padding 凑 seq_len
            prompt = "The meaning of life is " + "hello " * max(0, seq_len // 6)
            batch_prompts = [prompt] * bs

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,  # greedy
            )

            start = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed

            print(f"{bs:<6} {seq_len:<8} {max_tokens:<6} {total_tokens:<14} {elapsed:<10.2f} {throughput:<10.1f}")


def benchmark_prefix_caching(model_name, max_model_len, prompt_len=256, num_requests=8):
    """Prefix Caching benchmark"""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"  Prefix Caching Benchmark")
    print(f"{'='*60}")

    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    # 共享前缀 + 不同后缀
    shared_prefix = "In the field of artificial intelligence, " * (prompt_len // 10)

    prompts = []
    for i in range(num_requests):
        suffix = f"Question {i}: What is the answer to everything? "
        prompts.append(shared_prefix + suffix)

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    # 第一次运行（无缓存）
    start = time.time()
    outputs1 = llm.generate(prompts, sampling_params)
    time1 = time.time() - start

    # 第二次运行（应该命中 prefix cache）
    # 用相同前缀 + 不同后缀
    prompts2 = []
    for i in range(num_requests):
        suffix = f"Different question {i}: How does the universe work? "
        prompts2.append(shared_prefix + suffix)

    start = time.time()
    outputs2 = llm.generate(prompts2, sampling_params)
    time2 = time.time() - start

    print(f"  Shared prefix: ~{len(shared_prefix.split())} tokens")
    print(f"  First run (no cache):  {time1:.2f}s")
    print(f"  Second run (cached):   {time2:.2f}s")
    if time2 > 0:
        print(f"  Speedup from cache:    {time1/time2:.2f}x")


def test_basic_inference(model_name, max_model_len):
    """基础推理测试"""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"  Basic Inference Test")
    print(f"{'='*60}")

    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    prompts = [
        "Once upon a time,",
        "The capital of France is",
        "In machine learning,",
    ]

    sampling_params = SamplingParams(max_tokens=50, temperature=0.7)

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        print(f"\n  Prompt: {prompts[i]}")
        print(f"  Output: {generated[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM 推理实验")
    parser.add_argument("--model", default="gpt2", help="模型名称 (HuggingFace)")
    parser.add_argument("--max-model-len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--mode", choices=["basic", "benchmark", "prefix", "all"],
                        default="all", help="运行模式")

    # 检查环境
    try:
        check_environment()
    except ImportError:
        print("Error: PyTorch not installed. This script requires a GPU environment.")
        print("Install: pip install torch")
        sys.exit(1)

    args = parser.parse_args()

    try:
        from vllm import LLM
    except ImportError:
        print("Error: vLLM not installed.")
        print("Install: pip install vllm==0.3.3")
        sys.exit(1)
    except Exception as e:
        print(f"Error importing vLLM: {e}")
        print("This may be a compatibility issue. Check torch/triton versions.")
        sys.exit(1)

    if args.mode in ("basic", "all"):
        test_basic_inference(args.model, args.max_model_len)

    if args.mode in ("benchmark", "all"):
        benchmark_offline(
            model_name=args.model,
            max_model_len=args.max_model_len,
            batch_sizes=[1, 4, 8],
            seq_lengths=[64, 128, 256],
            max_tokens=32,
        )

    if args.mode in ("prefix", "all"):
        benchmark_prefix_caching(
            model_name=args.model,
            max_model_len=args.max_model_len,
            prompt_len=128,
            num_requests=4,
        )
