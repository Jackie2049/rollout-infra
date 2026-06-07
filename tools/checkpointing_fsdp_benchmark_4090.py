#!/usr/bin/env python3
"""Activation Checkpointing + FSDP Interaction Benchmark — RTX 4090

Key questions:
1. Does checkpointing save memory under FSDP (where optimizer is already sharded)?
2. Checkpointing overhead under DDP vs FSDP1 vs FSDP2
3. Optimal checkpointing strategy (every/every_2nd/selective) under FSDP
4. Can FSDP + checkpointing fit a larger model on single GPU?

Experiments:
1. Memory: peak memory per strategy × parallelism method
2. Throughput: tok/s per strategy × parallelism method
3. Strategy comparison: every/every_2nd/every_3rd/selective under FSDP
4. Large model: 46M model with FSDP+checkpointing on 2 GPU

Usage:
  python -u tools/checkpointing_fsdp_benchmark_4090.py --exp single
  torchrun --nproc_per_node=2 tools/checkpointing_fsdp_benchmark_4090.py --exp fsdp
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

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.checkpoint import checkpoint


class MiniTransformer(nn.Module):
    """Transformer for checkpointing benchmark."""
    def __init__(self, vocab_size=100, d_model=256, n_heads=8,
                 d_head=32, n_layers=8, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0, batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.vocab_proj(x)


class MiniTransformerCheckpointed(nn.Module):
    """Transformer with configurable checkpointing."""
    def __init__(self, vocab_size=100, d_model=256, n_heads=8,
                 d_head=32, n_layers=8, checkpoint_every=2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.checkpoint_every = checkpoint_every
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0, batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            if self.checkpoint_every > 0 and (i % self.checkpoint_every == 0):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.final_norm(x)
        return self.vocab_proj(x)


def train_step(model, optimizer, input_ids, targets):
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def measure_step(model, optimizer, device, vocab_size, batch_size=4,
                  seq_len=128, n_steps=20, warmup=5):
    # Warmup
    for _ in range(warmup):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        train_step(model, optimizer, ids, targets)

    torch.cuda.synchronize()
    times = []
    for _ in range(n_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        train_step(model, optimizer, ids, targets)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        'median_ms': np.median(times),
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'throughput': batch_size * seq_len * n_steps / (n_steps * np.median(times) / 1000),
    }


def exp1_memory_throughput(device):
    """Memory and throughput: checkpointing strategies × model sizes on single GPU."""
    print("\n" + "="*60)
    print("Exp 1: Checkpointing × Model Size — Single GPU")
    print("="*60)

    results = {}
    configs = [
        # (d_model, n_layers, checkpoint_every, label)
        (256, 4, 0, "2.3M_no_ckpt"),
        (256, 4, 1, "2.3M_every"),
        (256, 4, 2, "2.3M_every2"),
        (256, 8, 0, "4.5M_no_ckpt"),
        (256, 8, 1, "4.5M_every"),
        (256, 8, 2, "4.5M_every2"),
        (512, 8, 0, "25M_no_ckpt"),
        (512, 8, 1, "25M_every"),
        (512, 8, 2, "25M_every2"),
    ]

    vocab_size = 100
    batch_size = 4
    seq_len = 128

    for d_model, n_layers, ckpt_every, label in configs:
        if ckpt_every == 0:
            model = MiniTransformer(vocab_size, d_model, 8, 32, n_layers).to(device)
        else:
            model = MiniTransformerCheckpointed(vocab_size, d_model, 8, 32, n_layers, ckpt_every).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        total_params = sum(p.numel() for p in model.parameters())

        torch.cuda.reset_peak_memory_stats(device)
        result = measure_step(model, optimizer, device, vocab_size, batch_size, seq_len)
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

        result['peak_memory_GB'] = peak_mem
        result['total_params'] = total_params
        results[label] = result

        print(f"  {label}: {total_params/1e6:.2f}M params, {peak_mem:.3f} GB, "
              f"{result['median_ms']:.2f} ms, {result['throughput']:.0f} tok/s")

    return results


def exp2_fsdp_checkpointing(rank, world_size):
    """FSDP + checkpointing interaction on 2 GPU."""
    print(f"\n[Rank {rank}] Exp 2: FSDP1 + Checkpointing — 2 GPU")

    results = {}
    vocab_size = 100
    batch_size = 4
    seq_len = 128
    d_model = 512
    n_layers = 8

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    ckpt_configs = [
        (0, "no_ckpt"),
        (1, "every"),
        (2, "every2"),
        (3, "every3"),
    ]

    for ckpt_every, ckpt_label in ckpt_configs:
        if ckpt_every == 0:
            model = MiniTransformer(vocab_size, d_model, 8, 32, n_layers).to(device)
        else:
            model = MiniTransformerCheckpointed(vocab_size, d_model, 8, 32, n_layers, ckpt_every).to(device)

        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=rank)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        torch.cuda.reset_peak_memory_stats(device)

        # Warmup (more for FSDP)
        for _ in range(5):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            train_step(model, optimizer, ids, targets)

        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            train_step(model, optimizer, ids, targets)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        median_ms = np.median(times)
        throughput = batch_size * seq_len * 20 / (20 * median_ms / 1000)
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        total_params = sum(p.numel() for p in model.module.parameters()) if hasattr(model, 'module') else sum(p.numel() for p in model.parameters())

        results[f"fsdp1_{ckpt_label}"] = {
            'median_ms': median_ms, 'throughput': throughput,
            'peak_memory_GB': peak_mem, 'total_params': total_params,
            'checkpoint_every': ckpt_every,
        }

        if rank == 0:
            print(f"  FSDP1 {ckpt_label}: {median_ms:.2f} ms, {throughput:.0f} tok/s, "
                  f"{peak_mem:.3f} GB")

    return results


def exp3_ddp_checkpointing(rank, world_size):
    """DDP + checkpointing for comparison."""
    print(f"\n[Rank {rank}] Exp 3: DDP + Checkpointing — 2 GPU")

    results = {}
    vocab_size = 100
    batch_size = 4
    seq_len = 128
    d_model = 512
    n_layers = 8

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    ckpt_configs = [
        (0, "no_ckpt"),
        (1, "every"),
        (2, "every2"),
    ]

    for ckpt_every, ckpt_label in ckpt_configs:
        if ckpt_every == 0:
            model = MiniTransformer(vocab_size, d_model, 8, 32, n_layers).to(device)
        else:
            model = MiniTransformerCheckpointed(vocab_size, d_model, 8, 32, n_layers, ckpt_every).to(device)

        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(5):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            train_step(model, optimizer, ids, targets)

        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            train_step(model, optimizer, ids, targets)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        median_ms = np.median(times)
        throughput = batch_size * seq_len * 20 * world_size / (20 * median_ms / 1000)
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

        results[f"ddp_{ckpt_label}"] = {
            'median_ms': median_ms, 'throughput': throughput,
            'peak_memory_GB': peak_mem, 'checkpoint_every': ckpt_every,
        }

        if rank == 0:
            print(f"  DDP {ckpt_label}: {median_ms:.2f} ms, {throughput:.0f} tok/s, "
                  f"{peak_mem:.3f} GB")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['single', 'fsdp', 'ddp', 'all'])
    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))

    all_results = {}

    if world_size == 1:
        device = torch.device('cuda:0')
        print("="*70)
        print(f"Checkpointing + FSDP Benchmark — Single GPU RTX 4090")
        print("="*70)
        print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch: {torch.__version__}")

        if args.exp in ['single', 'all']:
            all_results['exp1_single'] = exp1_memory_throughput(device)

    else:
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        if rank == 0:
            print("="*70)
            print(f"Checkpointing + FSDP Benchmark — {world_size}× RTX 4090")
            print("="*70)

        if args.exp in ['fsdp', 'all']:
            all_results['exp2_fsdp'] = exp2_fsdp_checkpointing(rank, world_size)
        if args.exp in ['ddp', 'all']:
            all_results['exp3_ddp'] = exp3_ddp_checkpointing(rank, world_size)

        if rank == 0:
            print("\n" + "="*70)
            print("Summary")
            print("="*70)
            for key, val in all_results.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        if isinstance(subval, dict) and 'median_ms' in subval:
                            print(f"  {subkey}: {subval['median_ms']:.2f} ms, "
                                  f"{subval['throughput']:.0f} tok/s, "
                                  f"{subval['peak_memory_GB']:.3f} GB")

        dist.destroy_process_group()

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'checkpointing_fsdp_benchmark.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()