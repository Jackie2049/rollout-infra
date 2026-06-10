#!/usr/bin/env python3
"""FSDP Scaling Benchmark — OPT-125M on RTX 4090 PCIe

Benchmarks FSDP vs DDP scaling for small model (125M):
1. Single GPU baseline (training throughput)
2. FSDP 2/4/8 GPU scaling
3. DDP 2/4/8 GPU scaling
4. Communication overhead estimation

Expected: Small model → FSDP/DDP viable (comm < 1ms)
          7B → disaster (comm 1536ms) — already verified
"""

import json
import os
import time
import torch
import torch.distributed as dist

MODEL_PATH = "/home/zxw/rollout-infra/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"

def setup(rank, world_size):
    # Let torchrun handle MASTER_ADDR/PORT via env vars
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def benchmark_training(rank, world_size, mode='ddp'):
    """Benchmark training throughput for OPT-125M"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, local_files_only=True,
        weights_only=False,
    ).to(f'cuda:{rank}')
    model.train()

    if mode == 'fsdp':
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model, device_id=rank)
    elif mode == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Create dummy input
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=f'cuda:{rank}')
    labels = input_ids.clone()

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    n_steps = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    step_time = total_time / n_steps
    throughput = batch_size * seq_len * world_size / step_time  # tokens/s total
    per_gpu_throughput = batch_size * seq_len / step_time

    mem_peak = torch.cuda.max_memory_allocated(f'cuda:{rank}') / 1024 / 1024

    results = {
        'mode': mode,
        'n_gpu': world_size,
        'step_time_s': round(step_time, 4),
        'total_throughput_tok_s': round(throughput, 1),
        'per_gpu_throughput_tok_s': round(per_gpu_throughput, 1),
        'mem_peak_mb': round(mem_peak, 1),
        'speedup_vs_1gpu': 0,  # will calculate later
    }

    if rank == 0:
        with open(f'fsdp_scaling_{mode}_{world_size}gpu.json', 'w') as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()
    return results

def single_gpu_baseline():
    """Single GPU training baseline"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # CUDA_VISIBLE_DEVICES remaps, so use cuda:0
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=device,
        local_files_only=True, weights_only=False,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Benchmark
    n_steps = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    step_time = total_time / n_steps
    throughput = batch_size * seq_len / step_time
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    results = {
        'mode': 'single_gpu',
        'n_gpu': 1,
        'step_time_s': round(step_time, 4),
        'per_gpu_throughput_tok_s': round(throughput, 1),
        'mem_peak_mb': round(mem_peak, 1),
    }

    print(f"Single GPU: step={step_time:.3f}s, throughput={throughput:.0f} tok/s, peak={mem_peak:.0f}MB")

    del model
    torch.cuda.empty_cache()

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'fsdp', 'ddp'], default='single')
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    if args.mode == 'single':
        print("=== Single GPU Baseline ===")
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        single_results = single_gpu_baseline()
        with open('fsdp_scaling_125m_single.json', 'w') as f:
            json.dump(single_results, f, indent=2)
        print(f"Saved to fsdp_scaling_125m_single.json")
    else:
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = args.world_size
        print(f"=== {args.mode.upper()} {world_size} GPU ===")
        results = benchmark_training(rank, world_size, mode=args.mode)
        if rank == 0:
            print(f"Results: {results}")