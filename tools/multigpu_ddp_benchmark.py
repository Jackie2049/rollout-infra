#!/usr/bin/env python3
"""Multi-GPU Distributed Training Benchmark — 8× RTX 4090
==========================================================
Implements DDP (Distributed Data Parallel) from scratch and benchmarks:
1. Scaling efficiency: 1, 2, 4 GPUs
2. Communication overhead measurement (AllReduce)
3. Gradient synchronization analysis
4. Batch size scaling effect

Key concepts:
- DDP: Each GPU has a copy of the model, processes different data
- AllReduce: Synchronize gradients across GPUs
- Scaling efficiency = speedup / ideal_speedup

Usage: torchrun --nproc_per_node=N tools/multigpu_ddp_benchmark.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import time
import json
import os
import sys
from datetime import datetime


# ============================================================
# 1. Model
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, block_size=256):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        hidden = int(2 / 3 * 4 * d_model)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

        self.register_buffer("mask",
            torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        # Attention
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)

        # FFN
        h = self.ln2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, n_head=8, n_layer=8, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================
# 2. Synthetic Data
# ============================================================

def get_synthetic_batch(batch_size, block_size, vocab_size, device):
    """Generate random training data."""
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    return x, y


# ============================================================
# 3. Communication Benchmark
# ============================================================

def benchmark_allreduce(world_size, device, msg_size_mb=10, n_iters=100):
    """Benchmark AllReduce performance."""
    if world_size <= 1:
        return {'bandwidth_gbps': 0, 'latency_ms': 0}

    n_elements = msg_size_mb * 1024 * 1024 // 4  # float32
    data = torch.randn(n_elements, device=device)

    # Warmup
    for _ in range(10):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(n_iters):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iters

    # Data volume: each GPU sends (world_size-1)/world_size of data
    # Ring AllReduce: 2 * (n-1)/n * data_size
    data_bytes = n_elements * 4
    bandwidth = data_bytes * 2 * (world_size - 1) / world_size / elapsed / 1e9

    return {
        'latency_ms': elapsed * 1000,
        'bandwidth_gbps': bandwidth,
        'msg_size_mb': msg_size_mb,
    }


# ============================================================
# 4. DDP Training Benchmark
# ============================================================

def benchmark_ddp_training(rank, world_size, device, model_config, n_warmup=10, n_iters=50):
    """Benchmark DDP training throughput."""
    vocab_size = 256
    d_model, n_head, n_layer, block_size = model_config

    # Create model
    model = MiniGPT(vocab_size, d_model, n_head, n_layer, block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Wrap in DDP
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Per-GPU batch size
    batch_per_gpu = 32

    # Warmup
    for _ in range(n_warmup):
        x, y = get_synthetic_batch(batch_per_gpu, block_size, vocab_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    total_tokens = 0
    t0 = time.time()
    for _ in range(n_iters):
        x, y = get_synthetic_batch(batch_per_gpu, block_size, vocab_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_tokens += batch_per_gpu * block_size

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Results
    throughput = total_tokens / elapsed  # tokens/sec per GPU
    global_throughput = throughput * world_size  # total tokens/sec
    samples_per_sec = batch_per_gpu * world_size / (elapsed / n_iters)

    mem_used = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"  [{world_size} GPU] Throughput: {global_throughput/1e3:.1f}K tok/s "
              f"({throughput/1e3:.1f}K tok/s/GPU), "
              f"Mem: {mem_used:.2f}GB, "
              f"Time: {elapsed:.2f}s")

    return {
        'n_gpus': world_size,
        'n_params_M': n_params / 1e6,
        'per_gpu_tok_s': throughput,
        'global_tok_s': global_throughput,
        'mem_gb': mem_used,
        'elapsed_s': elapsed,
    }


# ============================================================
# 5. Main
# ============================================================

def main():
    # Initialize distributed
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    if rank == 0:
        print("=" * 70)
        print(f"Multi-GPU DDP Benchmark — {world_size} GPU(s)")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print("=" * 70)

    results = {}

    # ----------------------------------------------------------
    # Experiment 1: AllReduce Communication Benchmark
    # ----------------------------------------------------------
    if rank == 0:
        print("\n--- Experiment 1: AllReduce Communication ---")

    for msg_mb in [1, 10, 50, 100]:
        comm = benchmark_allreduce(world_size, device, msg_mb, n_iters=200)
        if rank == 0 and world_size > 1:
            print(f"  {msg_mb}MB: latency={comm['latency_ms']:.3f}ms, "
                  f"bandwidth={comm['bandwidth_gbps']:.2f} GB/s")
            results[f'comm_{msg_mb}mb'] = comm

    if rank == 0 and world_size == 1:
        print("  (Single GPU — no communication needed)")

    # ----------------------------------------------------------
    # Experiment 2: DDP Scaling Efficiency
    # ----------------------------------------------------------
    if rank == 0:
        print("\n--- Experiment 2: DDP Scaling (4M model) ---")

    model_config = (256, 8, 8, 256)  # d_model, n_head, n_layer, block_size
    ddp_result = benchmark_ddp_training(rank, world_size, device, model_config)
    if rank == 0:
        results[f'ddp_{world_size}gpu'] = ddp_result

    # ----------------------------------------------------------
    # Experiment 3: Batch Size Scaling
    # ----------------------------------------------------------
    if rank == 0:
        print("\n--- Experiment 3: Batch Size Scaling ---")

    vocab_size, block_size = 256, 256
    d_model, n_head, n_layer = 256, 8, 8
    model = MiniGPT(vocab_size, d_model, n_head, n_layer, block_size).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for bs in [8, 16, 32, 64, 128]:
        torch.cuda.reset_peak_memory_stats()
        # Warmup
        for _ in range(5):
            x, y = get_synthetic_batch(bs, block_size, vocab_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            x, y = get_synthetic_batch(bs, block_size, vocab_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        tok_s = bs * block_size * 30 / elapsed * world_size
        mem = torch.cuda.max_memory_allocated() / 1e9

        if rank == 0:
            print(f"  BS={bs:3d} per GPU: {tok_s/1e3:.1f}K tok/s (global), "
                  f"{mem:.2f}GB, {elapsed:.2f}s")
            results[f'bs_{bs}'] = {'tok_s': tok_s, 'mem_gb': mem, 'time': elapsed}

    # ----------------------------------------------------------
    # Experiment 4: Model Size Scaling
    # ----------------------------------------------------------
    if rank == 0:
        print("\n--- Experiment 4: Model Size Scaling ---")

    configs = [
        ('small',  128, 4, 4, 256),
        ('medium', 256, 8, 8, 256),
        ('large',  384, 8, 12, 256),
        ('xlarge', 512, 8, 16, 256),
    ]

    for name, d, h, l, bs in configs:
        torch.cuda.reset_peak_memory_stats()
        model = MiniGPT(vocab_size, d, h, l, bs).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Warmup + Benchmark
        for _ in range(5):
            x, y = get_synthetic_batch(32, bs, vocab_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            x, y = get_synthetic_batch(32, bs, vocab_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        tok_s = 32 * bs * 30 / elapsed * world_size
        mem = torch.cuda.max_memory_allocated() / 1e9

        if rank == 0:
            print(f"  {name:7s} ({n_params/1e6:.1f}M): {tok_s/1e3:.1f}K tok/s, "
                  f"{mem:.2f}GB, {elapsed:.2f}s")
            results[f'model_{name}'] = {
                'params_M': n_params/1e6, 'tok_s': tok_s, 'mem_gb': mem
            }

    # Save results (rank 0 only)
    if rank == 0:
        with open('multigpu_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to multigpu_results.json")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
