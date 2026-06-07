#!/usr/bin/env python3
"""ZeRO Distributed Training Experiment — RTX 4090 PCIe Cluster

Compare ZeRO stages 0/1/2/3 with different DP sizes on 8×RTX 4090 PCIe.
Uses PyTorch DDP + manual ZeRO simulation (shard optimizer/gradients/parameters).

Key questions:
1. How much memory does each ZeRO stage save?
2. What's the throughput cost of ZeRO communication?
3. What DP size + ZeRO stage combination is optimal for RTX 4090 PCIe?

Usage: On GPU server with 8 GPUs:
  python tools/zero_training_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    generate_arithmetic_prompt, generate_sft_dataset,
)


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def measure_memory(model, zero_stage=0, dp=1):
    """Measure GPU memory usage for different ZeRO configurations."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    params = sum(p.numel() for p in model.parameters())
    param_bytes = params * 2  #FP16

    #ZeRO savings (per GPU)
    if zero_stage >= 1:
        optimizer_bytes_per_gpu = params * 12 / dp  #ZeRO-1: shard optimizer
    else:
        optimizer_bytes_per_gpu = params * 12

    if zero_stage >= 2:
        grad_bytes_per_gpu = params * 2 / dp  #ZeRO-2: shard gradients
    else:
        grad_bytes_per_gpu = params * 2

    if zero_stage >= 3:
        param_bytes_per_gpu = params * 2 / dp  #ZeRO-3: shard parameters
    else:
        param_bytes_per_gpu = params * 2

    activation_bytes_per_gpu = params * 1  #Approximate

    total_per_gpu = (param_bytes_per_gpu + optimizer_bytes_per_gpu +
                     grad_bytes_per_gpu + activation_bytes_per_gpu) / 1e9

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        'params': params,
        'total_per_gpu_gb': total_per_gpu,
        'peak_gpu_mem_gb': peak_mem,
        'model_gb': param_bytes_per_gpu / 1e9,
        'optimizer_gb': optimizer_bytes_per_gpu / 1e9,
        'grad_gb': grad_bytes_per_gpu / 1e9,
        'activation_gb': activation_bytes_per_gpu / 1e9,
        'zero_stage': zero_stage,
        'dp': dp,
    }


def train_step_sft(model, dataset, optimizer, device, zero_stage=0, dp=1):
    """SFT training step with ZeRO simulation."""
    model.train()

    #Pick random batch
    idx = np.random.randint(len(dataset))
    full_tokens, prompt_len = dataset[idx]
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    targets = input_ids.clone()

    #Forward
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                           targets[:, prompt_len:].reshape(-1))

    #ZeRO-2: shard gradients (gradient accumulation across DP)
    #For real ZeRO-2, each GPU only stores 1/dp of gradients
    #We simulate by scaling loss by 1/dp (equivalent to gradient averaging)
    if dp > 1 and zero_stage >= 2:
        loss = loss / dp  #Gradient averaging for DP

    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Gradient sync for DDP
    if dp > 1:
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    #ZeRO-1: optimizer step (sharded optimizer — we do full step here since
    #we can't easily shard optimizer in pure PyTorch)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def run_zero_experiment(rank, world_size, model_size='76k', zero_stage=0,
                         num_steps=100, return_dict=None):
    """Run ZeRO training experiment on one GPU."""
    setup_distributed(rank, world_size)
    device = torch.cuda.current_device()

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    #Model config based on size
    if model_size == '76k':
        model = MiniGQATransformer(
            hidden_dim=64, num_layers=2, num_heads=4,
            num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    elif model_size == '2.28m':
        model = MiniGQATransformer(
            hidden_dim=256, num_layers=4, num_heads=8,
            num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)
    elif model_size == '3.3m':
        model = MiniGQATransformer(
            hidden_dim=256, num_layers=6, num_heads=8,
            num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    #Wrap in DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    dataset = generate_sft_dataset(500)

    #Measure memory
    mem_info = measure_memory(model.module, zero_stage, world_size)

    #Training
    losses = []
    start_time = time.time()

    for step in range(num_steps):
        loss = train_step_sft(model, dataset, optimizer, device, zero_stage, world_size)
        losses.append(float(loss))

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed

    #Eval (only on rank 0)
    eval_acc = 0.0
    if rank == 0:
        model.module.eval()
        correct = 0
        for _ in range(50):
            prompt_tokens, correct_sum = generate_arithmetic_prompt()
            input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                      dtype=torch.long, device=device)
            with torch.no_grad():
                pred = model.module(input_ids)[0, -2, :].argmax().item()
            if pred == correct_sum:
                correct += 1
        eval_acc = correct / 50

    #Gather eval_acc from rank 0
    eval_tensor = torch.tensor([eval_acc], device=device)
    dist.broadcast(eval_tensor, src=0)
    eval_acc = eval_tensor.item()

    cleanup_distributed()

    result = {
        'model_size': model_size,
        'zero_stage': zero_stage,
        'dp': world_size,
        'final_loss': losses[-1],
        'peak_loss': max(losses),
        'eval_accuracy': eval_acc,
        'throughput_steps_s': throughput,
        'total_time_s': elapsed,
        'memory_per_gpu_gb': mem_info['total_per_gpu_gb'],
        'peak_gpu_mem_gb': mem_info['peak_gpu_mem_gb'],
    }

    if return_dict is not None:
        return_dict[f'zero{zero_stage}_dp{world_size}_{model_size}'] = result

    return result


def run_single_gpu_benchmark(model_size='76k', zero_stage=0, num_steps=100):
    """Run benchmark on single GPU (no DDP)."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    if model_size == '76k':
        model = MiniGQATransformer(
            hidden_dim=64, num_layers=2, num_heads=4,
            num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    elif model_size == '2.28m':
        model = MiniGQATransformer(
            hidden_dim=256, num_layers=4, num_heads=8,
            num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)
    elif model_size == '3.3m':
        model = MiniGQATransformer(
            hidden_dim=256, num_layers=6, num_heads=8,
            num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    dataset = generate_sft_dataset(500)

    mem_info = measure_memory(model, zero_stage, 1)

    losses = []
    start_time = time.time()
    for step in range(num_steps):
        idx = np.random.randint(len(dataset))
        full_tokens, prompt_len = dataset[idx]
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()

        logits = model(input_ids)
        loss = F.cross_entropy(logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                               targets[:, prompt_len:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss))

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed

    #Eval
    model.eval()
    correct = 0
    for _ in range(50):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    eval_acc = correct / 50

    return {
        'model_size': model_size,
        'zero_stage': zero_stage,
        'dp': 1,
        'final_loss': losses[-1],
        'peak_loss': max(losses),
        'eval_accuracy': eval_acc,
        'throughput_steps_s': throughput,
        'total_time_s': elapsed,
        'memory_per_gpu_gb': mem_info['total_per_gpu_gb'],
        'peak_gpu_mem_gb': mem_info['peak_gpu_mem_gb'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--output', default='zero_training_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("ZeRO Distributed Training Experiment — RTX 4090 PCIe")
    print("=" * 70)

    results = {}
    model_sizes = ['76k', '2.28m', '3.3m']

    #Phase 1: Single GPU benchmarks (ZeRO stages 0-3, memory simulation)
    print("\n--- Phase 1: Memory Analysis (Single GPU) ---")
    for model_size in model_sizes:
        for zero in [0, 1, 2, 3]:
            dp_vals = [1] if zero == 0 else [1, 2, 4, 8]
            for dp in dp_vals:
                mem = measure_memory(
                    MiniGQATransformer(
                        hidden_dim=64 if model_size == '76k' else 256,
                        num_layers=2 if model_size == '76k' else (4 if model_size == '2.28m' else 6),
                        num_heads=4 if model_size == '76k' else 8,
                        num_kv_heads=2 if model_size == '76k' else 4,
                        vocab_size=VOCAB_SIZE).to(device),
                    zero, dp)
                key = f'mem_zero{zero}_dp{dp}_{model_size}'
                results[key] = {
                    'model_size': model_size,
                    'zero_stage': zero,
                    'dp': dp,
                    'total_per_gpu_gb': mem['total_per_gpu_gb'],
                    'peak_gpu_mem_gb': mem['peak_gpu_mem_gb'],
                    'model_gb': mem['model_gb'],
                    'optimizer_gb': mem['optimizer_gb'],
                    'grad_gb': mem['grad_gb'],
                    'activation_gb': mem['activation_gb'],
                }
                fits = mem['total_per_gpu_gb'] <= 24
                print(f"  {model_size} ZeRO-{zero} DP={dp}: "
                      f"{mem['total_per_gpu_gb']:.2f}GB, fits={fits}")

    #Phase 2: Single GPU training benchmark
    print("\n--- Phase 2: Training Throughput (Single GPU) ---")
    for model_size in model_sizes:
        result = run_single_gpu_benchmark(model_size, 0, args.num_steps)
        key = f'train_zero0_dp1_{model_size}'
        results[key] = result
        print(f"  {model_size}: {result['throughput_steps_s']:.1f} steps/s, "
              f"loss={result['final_loss']:.3f}, eval={result['eval_accuracy']:.1%}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("ZeRO Training Summary")
    print("=" * 70)
    print("\nMemory (per GPU):")
    for model_size in model_sizes:
        for zero in [0, 1, 2, 3]:
            dp = 1 if zero == 0 else 8
            key = f'mem_zero{zero}_dp{dp}_{model_size}'
            if key in results:
                r = results[key]
                print(f"  {model_size} ZeRO-{zero} DP={dp}: "
                      f"{r['total_per_gpu_gb']:.2f}GB, fits={r['total_per_gpu_gb'] <= 24}")


if __name__ == '__main__':
    main()