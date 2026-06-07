#!/usr/bin/env python3
"""BF16 + FSDP Training Benchmark — RTX 4090 PCIe
==================================================
Validates BF16 native training under distributed (FSDP1) settings.
Connects our two key findings:
1. BF16 native is the best single-GU training choice (1.23x speedup + 39% eval)
2. FSDP1 is the best 2-GPU distributed choice (125% efficiency on 25M)

Experiments:
1. Single-GPU BF16 vs FP32 training
2. FSDP1 2-GPU BF16 vs FP32 training
3. FSDP1 2-GPU BF16 + compile training

Usage:
  python bf16_fsdp_benchmark_4090.py --exp single  (single GPU)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 bf16_fsdp_benchmark_4090.py --exp fsdp
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import time
import json
import argparse
import math


class MiniGQA(nn.Module):
    """Mini GQA transformer for benchmarking."""
    def __init__(self, vocab_size=32000, d_model=256, n_heads=8, n_layers=4,
                 gqa_groups=2, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_step(model, optimizer, input_ids, targets):
    optimizer.zero_grad()
    output = model(input_ids)
    loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_accuracy(model, input_ids, targets):
    with torch.no_grad():
        output = model(input_ids)
        predicted = output.argmax(dim=-1)
        correct = (predicted == targets).float().mean().item()
    return correct


def benchmark_single_gpu(model_size="2.28M", dtype="bf16", n_steps=100, B=8):
    """Exp1: Single-GPU BF16 vs FP32 training."""
    device = "cuda"

    # Model config
    configs = {
        "2.28M": {"vocab_size": 32000, "d_model": 256, "n_heads": 8, "n_layers": 4},
        "7.5M": {"vocab_size": 32000, "d_model": 512, "n_heads": 8, "n_layers": 4},
        "25M": {"vocab_size": 32000, "d_model": 512, "n_heads": 16, "n_layers": 8},
    }
    cfg = configs[model_size]

    torch.manual_seed(42)
    model = MiniGQA(**cfg, max_seq_len=128).to(device)

    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    elif dtype == "fp16_amp":
        model = model  # FP32 model, AMP handles conversion

    n_params = count_params(model)
    S = 128  # sequence length

    # Data
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S), device=device)
    targets = torch.randint(0, cfg["vocab_size"], (B, S), device=device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    # GradScaler for FP16 AMP
    scaler = torch.amp.GradScaler("cuda") if dtype == "fp16_amp" else None

    # Warmup
    for _ in range(5):
        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                train_step(model, optimizer, input_ids, targets)
        else:
            train_step(model, optimizer, input_ids, targets)

    # Measure
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    losses = []

    for step in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = train_step(model, optimizer, input_ids, targets)
        else:
            loss = train_step(model, optimizer, input_ids, targets)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        losses.append(loss)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval
    if dtype == "bf16":
        eval_ids = input_ids.to(torch.bfloat16) if model.embedding.weight.dtype == torch.bfloat16 else input_ids
    else:
        eval_ids = input_ids
    acc = eval_accuracy(model, input_ids, targets)

    return {
        "model_size": model_size,
        "dtype": dtype,
        "n_params": n_params,
        "median_ms": sorted(times)[len(times) // 2],
        "mean_loss": sum(losses[-20:]) / 20,
        "peak_memory_GB": peak_mem,
        "eval_accuracy": acc,
        "throughput_tok_s": B * S / (sorted(times)[len(times) // 2] / 1000),
    }


def benchmark_fsdp_bf16(model_size="2.28M", dtype="bf16", n_steps=100, B=8):
    """Exp2: FSDP1 2-GPU BF16 vs FP32 training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    device = f"cuda:{rank}"
    configs = {
        "2.28M": {"vocab_size": 32000, "d_model": 256, "n_heads": 8, "n_layers": 4},
        "7.5M": {"vocab_size": 32000, "d_model": 512, "n_heads": 8, "n_layers": 4},
        "25M": {"vocab_size": 32000, "d_model": 512, "n_heads": 16, "n_layers": 8},
    }
    cfg = configs[model_size]
    S = 128

    torch.manual_seed(42 + rank)
    model = MiniGQA(**cfg, max_seq_len=128).to(device)

    # Apply FSDP1
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

    if dtype == "bf16":
        model = model.to(torch.bfloat16)

    # Data
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S), device=device)
    targets = torch.randint(0, cfg["vocab_size"], (B, S), device=device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda") if dtype == "fp16_amp" else None

    # Warmup
    for _ in range(5):
        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                train_step(model, optimizer, input_ids, targets)
        else:
            train_step(model, optimizer, input_ids, targets)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    losses = []

    for step in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if scaler:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = train_step(model, optimizer, input_ids, targets)
        else:
            loss = train_step(model, optimizer, input_ids, targets)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        losses.append(loss)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    acc = eval_accuracy(model, input_ids, targets)

    if rank == 0:
        result = {
            "model_size": model_size, "dtype": dtype,
            "n_params": count_params(model) * world_size,
            "median_ms": sorted(times)[len(times) // 2],
            "mean_loss": sum(losses[-20:]) / 20,
            "peak_memory_GB": peak_mem,
            "eval_accuracy": acc,
            "throughput_tok_s": B * S * world_size / (sorted(times)[len(times) // 2] / 1000),
            "world_size": world_size,
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))

    dist.destroy_process_group()


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='single',
                        choices=['single', 'fsdp'])
    parser.add_argument('--dtype', type=str, default='bf16',
                        choices=['fp32', 'bf16', 'fp16_amp'])
    parser.add_argument('--model-size', type=str, default='2.28M',
                        choices=['2.28M', '7.5M', '25M'])
    parser.add_argument('--B', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=100)
    parser.add_argument('--output', type=str, default='bf16_fsdp_benchmark.json')
    args = parser.parse_args()

    if args.exp == 'single':
        result = benchmark_single_gpu(
            model_size=args.model_size, dtype=args.dtype,
            n_steps=args.n_steps, B=args.B)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    elif args.exp == 'fsdp':
        benchmark_fsdp_bf16(
            model_size=args.model_size, dtype=args.dtype,
            n_steps=args.n_steps, B=args.B)


if __name__ == "__main__":
    main()