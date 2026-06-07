#!/usr/bin/env python3
"""DDP vs Gradient Accumulation Throughput Experiment — RTX 4090

Compare 3 strategies for training with larger effective batch size:
1. Single GPU + gradient accumulation (n steps before update)
2. DDP with 2/4/8 GPUs (each GPU does 1 step, then AllReduce)
3. DDP + gradient accumulation (each GPU accumulates n steps)

Key questions:
1. For small models, is DDP worth the PCIe overhead?
2. When does gradient accumulation beat DDP?
3. What's the optimal GPU count for PCIe cluster?

Usage: On GPU server:
  # Single GPU experiments
  CUDA_VISIBLE_DEVICES=0 python tools/ddp_vs_gradaccum.py --mode single
  # Multi-GPU experiments (requires torchrun)
  torchrun --nproc_per_node=4 tools/ddp_vs_gradaccum.py --mode ddp --world_size 4
  torchrun --nproc_per_node=8 tools/ddp_vs_gradaccum.py --mode ddp --world_size 8
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


# ============================================================
# Single GPU: Gradient Accumulation
# ============================================================

def train_single_gpu(model_size, num_steps, grad_accum_steps, lr=2e-3):
    """Train on single GPU with gradient accumulation."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_sft_dataset(500)

    #Memory measurement
    torch.cuda.reset_peak_memory_stats()
    peak_mem_start = torch.cuda.max_memory_allocated() / 1e9

    losses = []
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
            targets = input_ids.clone()

            logits = model(input_ids)
            loss = F.cross_entropy(logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                                   targets[:, prompt_len:].reshape(-1))
            loss = loss / grad_accum_steps  #Scale for accumulation
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(accum_loss * grad_accum_steps)

    elapsed = time.time() - start_time
    throughput = num_steps * grad_accum_steps / elapsed  #tokens processed per second proxy
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    #Eval
    model.eval()
    correct = 0
    for _ in range(100):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    eval_acc = correct / 100

    return {
        'strategy': 'single_gpu_grad_accum',
        'model_size': model_size,
        'num_gpus': 1,
        'grad_accum_steps': grad_accum_steps,
        'effective_batch_size': grad_accum_steps,
        'total_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': num_steps / elapsed,
        'throughput_micro_steps_s': throughput,
        'final_loss': losses[-1],
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
    }


# ============================================================
# DDP Training
# ============================================================

def train_ddp(rank, world_size, model_size, num_steps, lr=2e-3):
    """Train with DDP on multiple GPUs."""
    # torchrun sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE automatically
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_sft_dataset(500)

    torch.cuda.reset_peak_memory_stats()

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
        losses.append(loss.item())

    elapsed = time.time() - start_time

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    #Eval on rank 0
    eval_acc = 0.0
    if rank == 0:
        model.module.eval()
        correct = 0
        for _ in range(100):
            prompt_tokens, correct_sum = generate_arithmetic_prompt()
            input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                      dtype=torch.long, device=device)
            with torch.no_grad():
                pred = model.module(input_ids)[0, -2, :].argmax().item()
            if pred == correct_sum:
                correct += 1
        eval_acc = correct / 100

    eval_tensor = torch.tensor([eval_acc], device=device)
    dist.broadcast(eval_tensor, src=0)
    eval_acc = eval_tensor.item()

    dist.destroy_process_group()

    return {
        'strategy': 'ddp',
        'model_size': model_size,
        'num_gpus': world_size,
        'grad_accum_steps': 1,
        'effective_batch_size': world_size,
        'total_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': num_steps / elapsed,
        'throughput_micro_steps_s': num_steps / elapsed,
        'final_loss': losses[-1],
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
        'rank': rank,
    }


# ============================================================
# DDP + Gradient Accumulation
# ============================================================

def train_ddp_gradaccum(rank, world_size, model_size, num_steps,
                         grad_accum_steps, lr=2e-3):
    """Train with DDP + gradient accumulation."""
    # torchrun sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE automatically
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_sft_dataset(500)

    torch.cuda.reset_peak_memory_stats()

    losses = []
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
            targets = input_ids.clone()

            logits = model(input_ids)
            loss = F.cross_entropy(logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                                   targets[:, prompt_len:].reshape(-1))
            loss = loss / grad_accum_steps
            #DDP handles gradient sync via no_sync for accumulation
            if micro < grad_accum_steps - 1:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(accum_loss * grad_accum_steps)

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    eval_acc = 0.0
    if rank == 0:
        model.module.eval()
        correct = 0
        for _ in range(100):
            prompt_tokens, correct_sum = generate_arithmetic_prompt()
            input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                      dtype=torch.long, device=device)
            with torch.no_grad():
                pred = model.module(input_ids)[0, -2, :].argmax().item()
            if pred == correct_sum:
                correct += 1
        eval_acc = correct / 100

    eval_tensor = torch.tensor([eval_acc], device=device)
    dist.broadcast(eval_tensor, src=0)
    eval_acc = eval_tensor.item()

    dist.destroy_process_group()

    return {
        'strategy': 'ddp_grad_accum',
        'model_size': model_size,
        'num_gpus': world_size,
        'grad_accum_steps': grad_accum_steps,
        'effective_batch_size': world_size * grad_accum_steps,
        'total_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': num_steps / elapsed,
        'throughput_micro_steps_s': num_steps * grad_accum_steps / elapsed,
        'final_loss': losses[-1],
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
        'rank': rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'ddp', 'ddp_gradaccum', 'all'],
                        default='single')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--model_size', default='76k',
                        choices=['76k', '2.28m'])
    parser.add_argument('--output', default='ddp_vs_gradaccum_results.json')
    args = parser.parse_args()

    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE env vars
    # Use env vars as fallback when argparse defaults are used
    rank = int(os.environ.get('LOCAL_RANK', args.rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))

    print("=" * 70)
    print("DDP vs Gradient Accumulation Experiment — RTX 4090 PCIe")
    print("=" * 70)

    results = {}

    if args.mode == 'single' or args.mode == 'all':
        print("\n--- Single GPU + Gradient Accumulation ---")
        for ga in [1, 2, 4, 8, 16]:
            result = train_single_gpu(args.model_size, args.num_steps, ga)
            key = f'single_ga{ga}_{args.model_size}'
            results[key] = result
            print(f"  GA={ga}: {result['throughput_steps_s']:.1f} steps/s, "
                  f"micro={result['throughput_micro_steps_s']:.1f} micro/s, "
                  f"loss={result['final_loss']:.3f}, "
                  f"eval={result['eval_accuracy']:.1%}, "
                  f"mem={result['peak_gpu_mem_gb']:.2f}GB")

    if args.mode == 'ddp' or args.mode == 'all':
        print(f"\n--- DDP (world_size={world_size}) ---")
        result = train_ddp(rank, world_size, args.model_size, args.num_steps)
        if rank == 0:
            key = f'ddp_dp{world_size}_{args.model_size}'
            results[key] = result
            print(f"  DP={world_size}: {result['throughput_steps_s']:.1f} steps/s, "
                  f"loss={result['final_loss']:.3f}, "
                  f"eval={result['eval_accuracy']:.1%}, "
                  f"mem={result['peak_gpu_mem_gb']:.2f}GB")

    if args.mode == 'ddp_gradaccum' or args.mode == 'all':
        print(f"\n--- DDP + Gradient Accumulation (dp={world_size}, ga={args.grad_accum_steps}) ---")
        result = train_ddp_gradaccum(rank, world_size, args.model_size,
                                      args.num_steps, args.grad_accum_steps)
        if rank == 0:
            key = f'ddp_ga_dp{world_size}_ga{args.grad_accum_steps}_{args.model_size}'
            results[key] = result
            print(f"  DP={world_size} GA={args.grad_accum_steps}: "
                  f"{result['throughput_steps_s']:.1f} steps/s, "
                  f"micro={result['throughput_micro_steps_s']:.1f} micro/s, "
                  f"loss={result['final_loss']:.3f}, "
                  f"eval={result['eval_accuracy']:.1%}")

    #Save (only rank 0 for distributed)
    if args.mode == 'single' or rank == 0:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'results', args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()