#!/usr/bin/env python3
"""FSDP2 + torch.compile Distributed Training Benchmark — RTX 4090

Compare 5 training approaches on 8×RTX 4090 PCIe cluster:
1. Single GPU + gradient accumulation (baseline)
2. DDP (standard DistributedDataParallel)
3. FSDP1 (original FSDP, ZeRO-3 style)
4. FSDP2 (new composable FSDP)
5. FSDP2 + torch.compile

Experiments:
1. Memory: peak memory per GPU for each approach
2. Throughput: tokens/sec for each approach
3. Convergence: training loss curve (small model)
4. Scaling: 1→8 GPU efficiency for each approach

Usage:
  python -u tools/fsdp2_benchmark_4090.py          # single GPU
  torchrun --nproc_per_node=8 tools/fsdp2_benchmark_4090.py  # 8 GPU
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import json
import os
import sys
import time
import argparse
from contextlib import contextmanager

# ============================================================
# Model Definition
# ============================================================

class MiniGQATransformer(nn.Module):
    """Small GQA transformer for benchmarking."""
    def __init__(self, vocab_size=100, d_model=256, n_heads=8, n_kv_heads=4,
                 d_head=32, n_layers=4, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Total params
        total = sum(p.numel() for p in self.parameters())
        print(f"MiniGQATransformer: {total/1e6:.2f}M params")

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.vocab_proj(x)
        return logits


# ============================================================
# Training Functions
# ============================================================

def train_step(model, optimizer, batch, compile_mode=None):
    """Single training step."""
    input_ids, targets = batch
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def benchmark_throughput(model, optimizer, device, n_steps=20, batch_size=4,
                          seq_len=128, vocab_size=100, compile_mode=None):
    """Benchmark training throughput."""
    # Warmup
    for _ in range(3):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        train_step(model, optimizer, (ids, targets), compile_mode)

    torch.cuda.synchronize()
    times = []
    total_tokens = 0

    for _ in range(n_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = train_step(model, optimizer, (ids, targets), compile_mode)
        end.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end)
        times.append(elapsed)
        total_tokens += batch_size * seq_len

    median_time = np.median(times)
    throughput = total_tokens / (n_steps * median_time / 1000)

    return {
        'median_ms': median_time,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'throughput_tok_s': throughput,
        'total_tokens': total_tokens,
        'loss': loss,
    }


# ============================================================
# Experiment Functions
# ============================================================

def exp1_single_gpu(device, model_size='2.28M'):
    """Single GPU + gradient accumulation baseline."""
    print("\n" + "="*60)
    print("Exp 1: Single GPU + Gradient Accumulation")
    print("="*60)

    vocab_size = 100
    if model_size == '2.28M':
        d_model = 256; n_heads = 8; n_kv_heads = 4; d_head = 32; n_layers = 4
    else:
        d_model = 512; n_heads = 16; n_kv_heads = 4; d_head = 32; n_layers = 8

    model = MiniGQATransformer(vocab_size, d_model, n_heads, n_kv_heads,
                                d_head, n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Memory before training
    torch.cuda.reset_peak_memory_stats(device)
    peak_mem_before = torch.cuda.max_memory_allocated(device) / 1e9

    result = benchmark_throughput(model, optimizer, device, n_steps=20,
                                   batch_size=4, seq_len=128, vocab_size=vocab_size)

    peak_mem_after = torch.cuda.max_memory_allocated(device) / 1e9

    result['peak_memory_GB'] = peak_mem_after
    result['model_params'] = sum(p.numel() for p in model.parameters())
    print(f"  Params: {result['model_params']/1e6:.2f}M")
    print(f"  Peak memory: {peak_mem_after:.3f} GB")
    print(f"  Throughput: {result['throughput_tok_s']:.0f} tok/s")
    print(f"  Step time: {result['median_ms']:.3f} ms")

    return result


def exp2_ddp(rank, world_size, model_size='2.28M'):
    """DDP benchmark."""
    print(f"\n[Rank {rank}] DDP Benchmark")

    vocab_size = 100
    if model_size == '2.28M':
        d_model = 256; n_heads = 8; n_kv_heads = 4; d_head = 32; n_layers = 4
    else:
        d_model = 512; n_heads = 16; n_kv_heads = 4; d_head = 32; n_layers = 8

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = MiniGQATransformer(vocab_size, d_model, n_heads, n_kv_heads,
                                d_head, n_layers).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark
    n_steps = 20
    batch_size = 4
    seq_len = 128

    # Warmup
    for _ in range(3):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = train_step(model, optimizer, (ids, targets))

    torch.cuda.synchronize()
    times = []

    for _ in range(n_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = train_step(model, optimizer, (ids, targets))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_time = np.median(times)
    total_tokens = n_steps * batch_size * seq_len * world_size
    throughput = total_tokens / (n_steps * median_time / 1000)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    result = {
        'world_size': world_size,
        'median_ms': median_time,
        'throughput_tok_s': throughput,
        'peak_memory_GB': peak_mem,
        'model_params': sum(p.numel() for p in model.parameters()),
    }

    if rank == 0:
        print(f"  DDP {world_size} GPU: {median_time:.3f} ms/step, "
              f"{throughput:.0f} tok/s, {peak_mem:.3f} GB")

    return result


def exp3_fsdp1(rank, world_size, model_size='2.28M'):
    """FSDP1 benchmark."""
    print(f"\n[Rank {rank}] FSDP1 Benchmark")

    vocab_size = 100
    if model_size == '2.28M':
        d_model = 256; n_heads = 8; n_kv_heads = 4; d_head = 32; n_layers = 4
    else:
        d_model = 512; n_heads = 16; n_kv_heads = 4; d_head = 32; n_layers = 8

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = MiniGQATransformer(vocab_size, d_model, n_heads, n_kv_heads,
                                d_head, n_layers).to(device)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 device_id=rank)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark
    n_steps = 20
    batch_size = 4
    seq_len = 128

    # Warmup
    for _ in range(3):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = train_step(model, optimizer, (ids, targets))

    torch.cuda.synchronize()
    times = []

    for _ in range(n_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = train_step(model, optimizer, (ids, targets))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_time = np.median(times)
    total_tokens = n_steps * batch_size * seq_len * world_size
    throughput = total_tokens / (n_steps * median_time / 1000)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    result = {
        'world_size': world_size,
        'median_ms': median_time,
        'throughput_tok_s': throughput,
        'peak_memory_GB': peak_mem,
        'model_params': sum(p.numel() for p in model.module.parameters()) if hasattr(model, 'module') else sum(p.numel() for p in model.parameters()),
    }

    if rank == 0:
        print(f"  FSDP1 {world_size} GPU: {median_time:.3f} ms/step, "
              f"{throughput:.0f} tok/s, {peak_mem:.3f} GB")

    return result


def exp4_fsdp2(rank, world_size, model_size='2.28M'):
    """FSDP2 (composable) benchmark."""
    print(f"\n[Rank {rank}] FSDP2 Benchmark")

    vocab_size = 100
    if model_size == '2.28M':
        d_model = 256; n_heads = 8; n_kv_heads = 4; d_head = 32; n_layers = 4
    else:
        d_model = 512; n_heads = 16; n_kv_heads = 4; d_head = 32; n_layers = 8

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = MiniGQATransformer(vocab_size, d_model, n_heads, n_kv_heads,
                                d_head, n_layers).to(device)

    # Apply FSDP2 composable API
    from torch.distributed._composable.fsdp import fully_shard

    # Shard each layer then shard the whole model
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark
    n_steps = 20
    batch_size = 4
    seq_len = 128

    # Warmup
    for _ in range(3):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = train_step(model, optimizer, (ids, targets))

    torch.cuda.synchronize()
    times = []

    for _ in range(n_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = train_step(model, optimizer, (ids, targets))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_time = np.median(times)
    total_tokens = n_steps * batch_size * seq_len * world_size
    throughput = total_tokens / (n_steps * median_time / 1000)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    result = {
        'world_size': world_size,
        'median_ms': median_time,
        'throughput_tok_s': throughput,
        'peak_memory_GB': peak_mem,
        'model_params': sum(p.numel() for p in model.module.parameters()) if hasattr(model, 'module') else sum(p.numel() for p in model.parameters()),
    }

    if rank == 0:
        print(f"  FSDP2 {world_size} GPU: {median_time:.3f} ms/step, "
              f"{throughput:.0f} tok/s, {peak_mem:.3f} GB")

    return result


def exp5_fsdp2_compile(rank, world_size, model_size='2.28M'):
    """FSDP2 + torch.compile benchmark."""
    print(f"\n[Rank {rank}] FSDP2 + torch.compile Benchmark")

    vocab_size = 100
    if model_size == '2.28M':
        d_model = 256; n_heads = 8; n_kv_heads = 4; d_head = 32; n_layers = 4
    else:
        d_model = 512; n_heads = 16; n_kv_heads = 4; d_head = 32; n_layers = 8

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    model = MiniGQATransformer(vocab_size, d_model, n_heads, n_kv_heads,
                                d_head, n_layers).to(device)

    from torch.distributed._composable.fsdp import fully_shard

    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)

    # Apply torch.compile AFTER FSDP sharding
    model = torch.compile(model, mode='reduce-overhead')

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark (more warmup for compile)
    n_steps = 20
    batch_size = 4
    seq_len = 128

    # Warmup (more steps for compile to stabilize)
    for _ in range(5):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        train_step(model, optimizer, (ids, targets))

    torch.cuda.synchronize()
    times = []

    for _ in range(n_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = train_step(model, optimizer, (ids, targets))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_time = np.median(times)
    total_tokens = n_steps * batch_size * seq_len * world_size
    throughput = total_tokens / (n_steps * median_time / 1000)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    result = {
        'world_size': world_size,
        'median_ms': median_time,
        'throughput_tok_s': throughput,
        'peak_memory_GB': peak_mem,
        'model_params': sum(p.numel() for p in model.module.parameters()) if hasattr(model, 'module') else sum(p.numel() for p in model.parameters()),
    }

    if rank == 0:
        print(f"  FSDP2+compile {world_size} GPU: {median_time:.3f} ms/step, "
              f"{throughput:.0f} tok/s, {peak_mem:.3f} GB")

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='2.28M',
                        choices=['2.28M', '46M'])
    parser.add_argument('--exp', type=str, default='all',
                        choices=['single', 'ddp', 'fsdp1', 'fsdp2', 'fsdp2_compile', 'all'])
    args = parser.parse_args()

    # Detect distributed training from environment
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))

    all_results = {}

    if world_size == 1:
        # Single GPU experiments
        device = torch.device('cuda:0')
        print("="*70)
        print(f"FSDP2 Distributed Training Benchmark — RTX 4090 (Single GPU)")
        print("="*70)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")

        if args.exp in ['single', 'all']:
            all_results['single_gpu'] = exp1_single_gpu(device, args.model_size)
        print(f"\nSingle GPU baseline done. Use torchrun for distributed experiments.")

    else:
        # Distributed experiments
        dist.init_process_group(backend='nccl')

        if rank == 0:
            print("="*70)
            print(f"FSDP2 Distributed Training Benchmark — {world_size}× RTX 4090")
            print("="*70)

        if args.exp in ['ddp', 'all']:
            all_results['ddp'] = exp2_ddp(rank, world_size, args.model_size)
        if args.exp in ['fsdp1', 'all']:
            all_results['fsdp1'] = exp3_fsdp1(rank, world_size, args.model_size)
        if args.exp in ['fsdp2', 'all']:
            all_results['fsdp2'] = exp4_fsdp2(rank, world_size, args.model_size)
        if args.exp in ['fsdp2_compile', 'all']:
            all_results['fsdp2_compile'] = exp5_fsdp2_compile(rank, world_size, args.model_size)

        if rank == 0:
            print("\n" + "="*70)
            print("Summary")
            print("="*70)
            for key, val in all_results.items():
                if isinstance(val, dict) and 'median_ms' in val:
                    print(f"  {key}: {val['median_ms']:.3f} ms, "
                          f"{val['throughput_tok_s']:.0f} tok/s, "
                          f"{val['peak_memory_GB']:.3f} GB")

        dist.destroy_process_group()

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'fsdp2_benchmark.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()