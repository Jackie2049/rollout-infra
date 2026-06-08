"""
FSDP2 Multi-GPU Scaling Benchmark — RTX 4090 (PCIe)

Tests DDP vs FSDP1 vs FSDP2 scaling across 1/2/4/8 GPUs.
Measures throughput, memory, and communication overhead.
"""

import os, sys, json, time, argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy, size_based_auto_wrap_policy

def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class SimpleTransformer(torch.nn.Module):
    """Simple transformer-like model for benchmarking"""
    def __init__(self, hidden_size=2560, num_layers=8, vocab_size=32000, nhead=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, dim_feedforward=4*hidden_size,
                dropout=0.0, batch_first=True, norm_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_training(model, optimizer, input_ids, warmup=3, iters=10):
    """Benchmark training step latency"""
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        if dist.is_initialized():
            dist.barrier()

    torch.cuda.synchronize()
    latencies = []
    peak_mem = 0

    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        start = time.perf_counter()
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append(end - start)
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)

    return {
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "std_ms": (sum((l - sum(latencies)/len(latencies))**2 for l in latencies) / len(latencies))**0.5 * 1000,
        "peak_mem_GB": peak_mem,
        "throughput_tok_s": input_ids.shape[0] * input_ids.shape[1] / (sum(latencies) / len(latencies)),
    }

def main():
    rank, world_size, local_rank = setup_distributed()

    # Config
    configs = [
        {"name": "25M", "hidden": 512, "layers": 4, "vocab": 32000, "nhead": 8, "batch": 8, "seq": 512},
        {"name": "125M", "hidden": 1024, "layers": 8, "vocab": 32000, "nhead": 8, "batch": 4, "seq": 512},
    ]

    results = {}
    device_info = {
        "world_size": world_size,
        "gpu_name": torch.cuda.get_device_name(local_rank),
    }

    if rank == 0:
        print(f"GPU: {device_info['gpu_name']}, World size: {world_size}")

    for cfg in configs:
        hidden = cfg["hidden"]
        num_layers = cfg["layers"]
        vocab = cfg["vocab"]
        B = cfg["batch"]
        S = cfg["seq"]

        nhead = cfg["nhead"]

        model_raw = SimpleTransformer(hidden_size=hidden, num_layers=num_layers, vocab_size=vocab, nhead=nhead)
        num_params = count_params(model_raw)
        param_size_gb = num_params * 4 / 1e9  # BF16 = 2 bytes but Adam uses FP32 params

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Model: {cfg['name']} ({num_params/1e6:.1f}M params, {param_size_gb:.2f}GB FP32)")
            print(f"{'='*60}")

        input_ids = torch.randint(0, vocab, (B, S), device=torch.device(f"cuda:{local_rank}"))

        # === Single GPU (no distributed) ===
        if world_size == 1:
            model_raw = SimpleTransformer(hidden_size=hidden, num_layers=num_layers, vocab_size=vocab, nhead=nhead).to(f"cuda:{local_rank}")
            optimizer_raw = torch.optim.AdamW(model_raw.parameters(), lr=1e-4)
            single_result = benchmark_training(model_raw, optimizer_raw, input_ids)
            single_result["num_params_M"] = num_params / 1e6
            single_result["param_size_GB"] = param_size_gb
            print(f"  SingleGPU: {single_result['mean_ms']:.2f}ms, peak_mem={single_result['peak_mem_GB']:.3f}GB, throughput={single_result['throughput_tok_s']:.0f} tok/s")
            results[cfg["name"]] = {"SingleGPU": single_result}
            del model_raw, optimizer_raw
            torch.cuda.empty_cache()
            continue

        # === DDP ===
        model_ddp = SimpleTransformer(hidden_size=hidden, num_layers=num_layers, vocab_size=vocab, nhead=nhead).to(f"cuda:{local_rank}")
        model_ddp = DDP(model_ddp, device_ids=[local_rank])
        optimizer_ddp = torch.optim.AdamW(model_ddp.parameters(), lr=1e-4)

        ddp_result = benchmark_training(model_ddp, optimizer_ddp, input_ids)
        ddp_result["num_params_M"] = num_params / 1e6
        ddp_result["param_size_GB"] = param_size_gb

        if rank == 0:
            print(f"  DDP: {ddp_result['mean_ms']:.2f}ms, peak_mem={ddp_result['peak_mem_GB']:.3f}GB, throughput={ddp_result['throughput_tok_s']:.0f} tok/s")

        del model_ddp, optimizer_ddp
        torch.cuda.empty_cache()

        # === FSDP1 (Full Shard) ===
        model_fsdp1 = SimpleTransformer(hidden_size=hidden, num_layers=num_layers, vocab_size=vocab, nhead=nhead).to(f"cuda:{local_rank}")
        model_fsdp1 = FSDP(
            model_fsdp1,
            sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
        )
        optimizer_fsdp1 = torch.optim.AdamW(model_fsdp1.parameters(), lr=1e-4)

        fsdp1_result = benchmark_training(model_fsdp1, optimizer_fsdp1, input_ids)
        fsdp1_result["num_params_M"] = num_params / 1e6

        if rank == 0:
            print(f"  FSDP1(Full): {fsdp1_result['mean_ms']:.2f}ms, peak_mem={fsdp1_result['peak_mem_GB']:.3f}GB, throughput={fsdp1_result['throughput_tok_s']:.0f} tok/s")

        del model_fsdp1, optimizer_fsdp1
        torch.cuda.empty_cache()

        # === FSDP2 (Per-parameter shard, if available in PyTorch 2.9) ===
        try:
            # Try FSDP2 API (torch 2.6+)
            from torch.distributed._composable.fsdp import FSDP as FSDP2_mod, fully_shard
            model_fsdp2 = SimpleTransformer(hidden_size=hidden, num_layers=num_layers, vocab_size=vocab, nhead=nhead).to(f"cuda:{local_rank}")
            # Apply per-parameter sharding
            for layer in model_fsdp2.layers:
                fully_shard(layer)
            fully_shard(model_fsdp2)

            optimizer_fsdp2 = torch.optim.AdamW(model_fsdp2.parameters(), lr=1e-4)
            fsdp2_result = benchmark_training(model_fsdp2, optimizer_fsdp2, input_ids)
            fsdp2_result["num_params_M"] = num_params / 1e6
            fsdp2_result["strategy"] = "FSDP2_per_param"

            if rank == 0:
                print(f"  FSDP2(PerParam): {fsdp2_result['mean_ms']:.2f}ms, peak_mem={fsdp2_result['peak_mem_GB']:.3f}GB, throughput={fsdp2_result['throughput_tok_s']:.0f} tok/s")

            del model_fsdp2, optimizer_fsdp2
            torch.cuda.empty_cache()
        except ImportError:
            if rank == 0:
                print(f"  FSDP2 not available (need torch 2.6+)")
            fsdp2_result = None

        results[cfg["name"]] = {
            "DDP": ddp_result,
            "FSDP1": fsdp1_result,
            "FSDP2": fsdp2_result,
        }

    # Save results on rank 0
    if rank == 0:
        all_output = {
            "device": device_info,
            "results": results,
        }
        os.makedirs("results", exist_ok=True)
        with open("results/fsdp2_scaling_benchmark_4090.json", "w") as f:
            json.dump(all_output, f, indent=2)
        print(f"\nResults saved to results/fsdp2_scaling_benchmark_4090.json")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for cfg_name, res in results.items():
            ddp = res["DDP"]
            fsdp1 = res["FSDP1"]
            print(f"\n{cfg_name} ({ddp['num_params_M']:.1f}M params):")
            print(f"  DDP:   {ddp['mean_ms']:.2f}ms, {ddp['peak_mem_GB']:.3f}GB, {ddp['throughput_tok_s']:.0f}tok/s")
            print(f"  FSDP1: {fsdp1['mean_ms']:.2f}ms, {fsdp1['peak_mem_GB']:.3f}GB, {fsdp1['throughput_tok_s']:.0f}tok/s")
            print(f"  FSDP1 vs DDP: {fsdp1['mean_ms']/ddp['mean_ms']:.2f}x time, {fsdp1['peak_mem_GB']/ddp['peak_mem_GB']:.2f}x mem")
            if res["FSDP2"]:
                fsdp2 = res["FSDP2"]
                print(f"  FSDP2: {fsdp2['mean_ms']:.2f}ms, {fsdp2['peak_mem_GB']:.3f}GB, {fsdp2['throughput_tok_s']:.0f}tok/s")
                print(f"  FSDP2 vs DDP: {fsdp2['mean_ms']/ddp['mean_ms']:.2f}x time, {fsdp2['peak_mem_GB']/ddp['peak_mem_GB']:.2f}x mem")

    cleanup_distributed()

if __name__ == "__main__":
    main()