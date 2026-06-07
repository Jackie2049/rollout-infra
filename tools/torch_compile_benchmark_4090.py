#!/usr/bin/env python3
"""torch.compile Benchmark — RTX 4090

4 experiments:
1. Graph break analysis: torch._dynamo.explain on MiniGQATransformer
2. Compile modes: default/reduce-overhead/max-autotune vs eager
3. Fwd+Bwd compile: training step comparison
4. MiniGRPO compile: GRPO training step with compile

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/torch_compile_benchmark_4090.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    generate_arithmetic_prompt, generate_sft_dataset,
)


def measure_time(fn, warmup=5, repeat=20):
    """Measure function execution time with warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times), np.mean(times), np.std(times)


def exp1_graph_breaks(device):
    """Analyze graph breaks in MiniGQATransformer."""
    print("\n" + "="*50)
    print("Exp 1: Graph Break Analysis")
    print("="*50)

    model = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    # Test with typical input
    prompt_tokens = [TOKENS['2'], TOKENS['+'], TOKENS['1'], TOKENS['=']]
    input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>'], TOKENS['3']]],
                              dtype=torch.long, device=device)

    # Explain graph breaks
    try:
        explanation = torch._dynamo.explain(model, input_ids)
        print(f"  Graph breaks: {explanation.graph_break_count}")
        print(f"  Ops captured: {len(explanation.ops)}")
        for gb in explanation.graph_break_reasons:
            print(f"    Break: {gb}")
    except Exception as e:
        print(f"  Explain failed: {e}")

    # Compile and measure
    compiled = torch.compile(model, mode="default")
    compiled(input_ids)  # warmup

    median_eager, _, _ = measure_time(lambda: model(input_ids))
    median_compile, _, _ = measure_time(lambda: compiled(input_ids))
    speedup = median_eager / median_compile
    print(f"  Eager: {median_eager:.2f}ms, Compile: {median_compile:.2f}ms, Speedup: {speedup:.2f}x")

    return {'graph_breaks': getattr(explanation, 'graph_break_count', 'unknown'),
            'eager_ms': median_eager, 'compile_ms': median_compile, 'speedup': speedup}


def exp2_compile_modes(device):
    """Compare compile modes: default/reduce-overhead/max-autotune."""
    print("\n" + "="*50)
    print("Exp 2: Compile Modes Comparison")
    print("="*50)

    model = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    prompt_tokens = [TOKENS['2'], TOKENS['+'], TOKENS['1'], TOKENS['=']]
    input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>'], TOKENS['3']]],
                              dtype=torch.long, device=device)

    results = {}

    # Eager baseline
    median, mean, std = measure_time(lambda: model(input_ids))
    results['eager'] = {'median_ms': median, 'mean_ms': mean, 'std_ms': std}

    for mode in ['default', 'reduce-overhead', 'max-autotune']:
        compiled = torch.compile(model, mode=mode)
        # Warmup (compilation happens here)
        for _ in range(10):
            compiled(input_ids)
        torch.cuda.synchronize()

        median, mean, std = measure_time(lambda: compiled(input_ids))
        speedup = results['eager']['median_ms'] / median
        results[mode] = {'median_ms': median, 'mean_ms': mean, 'std_ms': std, 'speedup': speedup}
        print(f"  {mode}: {median:.2f}ms ({speedup:.2f}x)")

    return results


def exp3_training_step(device):
    """Compare training step (fwd+bwd) with compile."""
    print("\n" + "="*50)
    print("Exp 3: Training Step (Fwd+Bwd) Compile vs Eager")
    print("="*50)

    model = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    dataset = generate_sft_dataset(100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    def train_step_eager():
        optimizer.zero_grad()
        idx = np.random.randint(len(dataset))
        full_tokens, prompt_len = dataset[idx]
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
            targets[:, prompt_len:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Compile model and optimizer step
    compiled_model = torch.compile(model, mode='default')
    compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=2e-3)

    def train_step_compile():
        compiled_optimizer.zero_grad()
        idx = np.random.randint(len(dataset))
        full_tokens, prompt_len = dataset[idx]
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()
        logits = compiled_model(input_ids)
        loss = F.cross_entropy(
            logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
            targets[:, prompt_len:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=1.0)
        compiled_optimizer.step()

    # Warmup both
    for _ in range(20):
        train_step_eager()
    for _ in range(20):
        train_step_compile()
    torch.cuda.synchronize()

    median_eager, _, _ = measure_time(lambda: train_step_eager(), warmup=3, repeat=30)
    median_compile, _, _ = measure_time(lambda: train_step_compile(), warmup=3, repeat=30)

    speedup = median_eager / median_compile
    print(f"  Eager train step: {median_eager:.2f}ms")
    print(f"  Compile train step: {median_compile:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    return {'eager_ms': median_eager, 'compile_ms': median_compile, 'speedup': speedup}


def exp4_batch_scaling(device):
    """Compile speedup across batch sizes."""
    print("\n" + "="*50)
    print("Exp 4: Batch Size × Compile Speedup")
    print("="*50)

    model = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    compiled = torch.compile(model, mode='default')

    prompt_tokens = [TOKENS['2'], TOKENS['+'], TOKENS['1'], TOKENS['=']]
    batch_sizes = [1, 4, 16, 32, 64]

    results = {}
    for b in batch_sizes:
        input_ids = torch.tensor(
            [[prompt_tokens + [TOKENS['<eos>'], TOKENS['3']]] * b],
            dtype=torch.long, device=device).squeeze(0)

        # Warmup
        model(input_ids)
        compiled(input_ids)
        torch.cuda.synchronize()

        median_eager, _, _ = measure_time(lambda: model(input_ids), warmup=3, repeat=20)
        median_compile, _, _ = measure_time(lambda: compiled(input_ids), warmup=3, repeat=20)
        speedup = median_eager / median_compile
        results[f'batch_{b}'] = {'eager_ms': median_eager, 'compile_ms': median_compile, 'speedup': speedup}
        print(f"  Batch={b}: Eager {median_eager:.2f}ms → Compile {median_compile:.2f}ms ({speedup:.2f}x)")

    return results


def main():
    device = torch.device('cuda:0')
    print("=" * 70)
    print("torch.compile Benchmark — RTX 4090")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    all_results = {}

    all_results['exp1_graph_breaks'] = exp1_graph_breaks(device)
    all_results['exp2_compile_modes'] = exp2_compile_modes(device)
    all_results['exp3_training_step'] = exp3_training_step(device)
    all_results['exp4_batch_scaling'] = exp4_batch_scaling(device)

    print("\n" + "=" * 70)
    print("Summary: torch.compile Performance on RTX 4090")
    print("=" * 70)

    e1 = all_results['exp1_graph_breaks']
    print(f"  Graph breaks: {e1['graph_breaks']}")
    print(f"  Forward speedup: {e1['speedup']:.2f}x")

    e2 = all_results['exp2_compile_modes']
    print(f"  Mode comparison:")
    for mode in ['default', 'reduce-overhead', 'max-autotune']:
        if mode in e2:
            print(f"    {mode}: {e2[mode]['speedup']:.2f}x")

    e3 = all_results['exp3_training_step']
    print(f"  Training step speedup: {e3['speedup']:.2f}x")

    e4 = all_results['exp4_batch_scaling']
    print(f"  Batch scaling:")
    for b in [1, 4, 16, 32, 64]:
        key = f'batch_{b}'
        if key in e4:
            print(f"    B={b}: {e4[key]['speedup']:.2f}x")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'torch_compile_benchmark.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()