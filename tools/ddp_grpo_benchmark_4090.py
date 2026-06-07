#!/usr/bin/env python3
"""GRPO DDP Scaling Benchmark — 8×RTX 4090 PCIe

Uses multiprocessing to spawn DDP processes (no torchrun needed).
Measures: step time, throughput, memory, scaling efficiency.
"""

import json
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

DTYPE = torch.float32


class SmallGQATransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, num_heads=4, num_kv_heads=2, vocab_size=1000):
        super().__init__()
        self.head_dim = hidden_dim // num_heads
        self.g = num_heads // num_kv_heads
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'q_proj': nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False),
                'k_proj': nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False),
                'v_proj': nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False),
                'o_proj': nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False),
                'ln2': nn.LayerNorm(hidden_dim),
                'gate_proj': nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                'up_proj': nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                'down_proj': nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            }))
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            residual = x
            x = layer['ln1'](x)
            B, S, H = x.shape
            Q = layer['q_proj'](x).view(B, S, -1, self.head_dim)
            K = layer['k_proj'](x).view(B, S, -1, self.head_dim)
            V = layer['v_proj'](x).view(B, S, -1, self.head_dim)
            if self.g > 1:
                K = K.repeat_interleave(self.g, dim=2)
                V = V.repeat_interleave(self.g, dim=2)
            attn_out = F.scaled_dot_product_attention(
                Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2), is_causal=True
            )
            attn_out = attn_out.transpose(1,2).contiguous().view(B, S, -1)
            x = residual + layer['o_proj'](attn_out)
            residual = x
            x = layer['ln2'](x)
            gate = torch.sigmoid(layer['gate_proj'](x))
            x = residual + layer['down_proj'](gate * layer['up_proj'](x))
        x = self.final_ln(x)
        return self.lm_head(x)


def grpo_step(model, input_ids, response_mask, group_ids, optimizer):
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * response_mask

    batch_size = input_ids.shape[0]
    rewards = torch.randn(batch_size, device=input_ids.device, dtype=DTYPE)

    unique_groups = group_ids.unique()
    advantages = torch.zeros_like(token_log_probs)
    for gid in unique_groups:
        idx = (group_ids == gid)
        group_rewards = rewards[idx]
        mean_r = group_rewards.mean()
        std_r = group_rewards.std()
        if std_r < 1e-8:
            std_r = 1.0
        normalized = (group_rewards - mean_r) / std_r
        advantages[idx] = normalized.unsqueeze(-1) * response_mask[idx]

    loss = -(advantages * token_log_probs).sum() / response_mask.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def ddp_worker(rank, world_size, config, batch_size, seq_len, n_samples, num_steps, port, result_queue):
    """DDP worker process."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = SmallGQATransformer(**config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    vocab_size = config["vocab_size"]
    local_batch = batch_size // world_size
    param_count = sum(p.numel() for p in model.parameters())

    # Warmup
    warmup_ids = torch.randint(0, vocab_size, (local_batch, seq_len), device=rank)
    warmup_mask = torch.ones(local_batch, seq_len, device=rank)
    warmup_groups = torch.arange(local_batch // n_samples, device=rank).repeat(n_samples)
    for _ in range(5):
        grpo_step(ddp_model, warmup_ids, warmup_mask, warmup_groups, optimizer)

    torch.cuda.reset_peak_memory_stats(rank)
    torch.cuda.synchronize()

    times = []
    mem_before = torch.cuda.memory_allocated(rank) / 1e6

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (local_batch, seq_len), device=rank)
        response_mask = torch.ones(local_batch, seq_len, device=rank)
        group_ids = torch.arange(local_batch // n_samples, device=rank).repeat(n_samples)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        grpo_step(ddp_model, input_ids, response_mask, group_ids, optimizer)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    mem_peak = torch.cuda.max_memory_allocated(rank) / 1e6
    avg_time = sum(times) / len(times)

    result_queue.put({
        "rank": rank,
        "avg_step_ms": avg_time,
        "times": times,
        "mem_before_mb": mem_before,
        "mem_peak_mb": mem_peak,
        "param_count": param_count,
        "local_batch": local_batch,
    })

    dist.destroy_process_group()


def benchmark_single_gpu(config, batch_size, seq_len, n_samples, num_steps):
    device = "cuda:0"
    vocab_size = config["vocab_size"]

    model = SmallGQATransformer(**config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    warmup_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    warmup_mask = torch.ones(batch_size, seq_len, device=device)
    warmup_groups = torch.arange(batch_size // n_samples, device=device).repeat(n_samples)
    for _ in range(5):
        grpo_step(model, warmup_ids, warmup_mask, warmup_groups, optimizer)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    times = []
    mem_before = torch.cuda.memory_allocated(device) / 1e6

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        response_mask = torch.ones(batch_size, seq_len, device=device)
        group_ids = torch.arange(batch_size // n_samples, device=device).repeat(n_samples)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        grpo_step(model, input_ids, response_mask, group_ids, optimizer)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    mem_peak = torch.cuda.max_memory_allocated(device) / 1e6
    avg_time = sum(times) / len(times)

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "param_count": param_count,
        "param_mb": param_mb,
        "avg_step_ms": avg_time,
        "times": times,
        "mem_before_mb": mem_before,
        "mem_peak_mb": mem_peak,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "throughput_ktok_s": batch_size * seq_len / (avg_time/1000) / 1000,
    }


def benchmark_ddp(num_gpus, config, batch_size, seq_len, n_samples, num_steps, port=29500):
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=ddp_worker, args=(
            rank, num_gpus, config, batch_size, seq_len, n_samples, num_steps, port, result_queue
        ))
        p.start()
        processes.append(p)

    results = []
    for _ in range(num_gpus):
        results.append(result_queue.get())

    for p in processes:
        p.join()

    # Aggregate
    avg_times = [r['avg_step_ms'] for r in results]
    avg_step = sum(avg_times) / len(avg_times)
    global_batch = batch_size
    throughput = global_batch * seq_len / (avg_step/1000) / 1000

    return {
        "num_gpus": num_gpus,
        "global_batch": global_batch,
        "local_batch": batch_size // num_gpus,
        "avg_step_ms": avg_step,
        "per_rank_avg_ms": avg_times,
        "throughput_ktok_s": throughput,
        "mem_peak_mb": max(r['mem_peak_mb'] for r in results),
        "param_count": results[0]['param_count'],
    }


def main():
    print("=" * 70)
    print("GRPO DDP Scaling Benchmark — 8×RTX 4090 PCIe")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}\n")

    configs = {
        "small_3M": {"hidden_dim": 256, "num_layers": 4, "num_heads": 4, "num_kv_heads": 2, "vocab_size": 1000},
        "medium_46M": {"hidden_dim": 512, "num_layers": 6, "num_heads": 8, "num_kv_heads": 2, "vocab_size": 32000},
    }

    seq_len = 128
    n_samples = 4
    num_steps = 30
    base_batch = 8
    all_results = {}

    for model_name, config in configs.items():
        print(f"\n--- {model_name} model ---")

        # Single GPU baseline
        print("  Running single GPU baseline...")
        r1 = benchmark_single_gpu(config, base_batch, seq_len, n_samples, num_steps)
        print(f"  Params: {r1['param_count']:,} ({r1['param_mb']:.1f}MB)")
        print(f"  Avg step: {r1['avg_step_ms']:.2f}ms, {r1['throughput_ktok_s']:.1f}K tok/s")
        print(f"  Peak mem: {r1['mem_peak_mb']:.1f}MB")
        all_results[f"{model_name}_1gpu"] = r1

        # DDP scaling
        for num_gpus in [2, 4, 8]:
            # Use different ports for each run to avoid conflicts
            port = 40000 + num_gpus * 1000
            batch = base_batch * num_gpus
            print(f"\n  Running {num_gpus}-GPU DDP (batch={batch}, port={port})...")
            torch.cuda.empty_cache()
            try:
                r_ddp = benchmark_ddp(num_gpus, config, batch, seq_len, n_samples, num_steps, port)
                speedup = r1['avg_step_ms'] * num_gpus / r_ddp['avg_step_ms']
                efficiency = speedup / num_gpus * 100
                comm_overhead = (1 - efficiency/100) * 100

                print(f"    Avg step: {r_ddp['avg_step_ms']:.2f}ms")
                print(f"    Throughput: {r_ddp['throughput_ktok_s']:.1f}K tok/s")
                print(f"    Speedup: {speedup:.2f}x (ideal={num_gpus}x)")
                print(f"    Efficiency: {efficiency:.1f}%")
                print(f"    Comm overhead: {comm_overhead:.1f}%")
                print(f"    Peak mem: {r_ddp['mem_peak_mb']:.1f}MB")

                r_ddp['speedup'] = speedup
                r_ddp['efficiency_pct'] = efficiency
                r_ddp['comm_overhead_pct'] = comm_overhead
                all_results[f"{model_name}_{num_gpus}gpu"] = r_ddp
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results[f"{model_name}_{num_gpus}gpu"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'GPUs':<5} {'Step(ms)':<10} {'Speedup':<10} {'Eff%':<8} {'Comm%':<8} {'Tok/s':<10}")
    print("-" * 70)
    for model_name in configs:
        r1 = all_results[f"{model_name}_1gpu"]
        print(f"{model_name:<12} {'1':<5} {r1['avg_step_ms']:<10.2f} {'1.00':<10} {'100':<8} {'0':<8} {r1['throughput_ktok_s']:<10.1f}")
        for num_gpus in [2, 4, 8]:
            key = f"{model_name}_{num_gpus}gpu"
            if key in all_results and "error" not in all_results[key]:
                r = all_results[key]
                print(f"{model_name:<12} {num_gpus:<5} {r['avg_step_ms']:<10.2f} {r['speedup']:<10.2f} {r['efficiency_pct']:<8.1f} {r['comm_overhead_pct']:<8.1f} {r['throughput_ktok_s']:<10.1f}")

    # Save results
    output = {
        "gpu": gpu_name,
        "configs": configs,
        "seq_len": seq_len, "n_samples": n_samples, "num_steps": num_steps, "base_batch": base_batch,
        "results": all_results,
    }
    with open("grpo_ddp_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to grpo_ddp_benchmark_results.json")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()