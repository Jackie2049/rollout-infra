#!/usr/bin/env python3
"""
ZeRO / FSDP Memory Peak Simulator

Simulates peak GPU memory usage for different distributed training strategies:
- DDP (standard data parallelism)
- ZeRO-1 (optimizer state sharding)
- ZeRO-2 (optimizer + gradient sharding)
- ZeRO-3 (optimizer + gradient + parameter sharding)
- FSDP1 (FlatParameter sharding, similar to ZeRO-3)
- FSDP2 (DTensor per-parameter sharding)
- LoRA + ZeRO-2 (practical for RTX 4090)
- LoRA + CPU Adam (offload optimizer to CPU)

Usage:
    python3 zero_memory_peak_simulator.py
    python3 zero_memory_peak_simulator.py --model-size 7 --gpu-count 8 --gpu-memory 24
"""

import argparse
import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def simulate_memory(model_size_b, gpu_count, gpu_memory_gb, lora_rank=0):
    """
    Simulate peak GPU memory for different training strategies.

    model_size_b: model size in billions of parameters
    gpu_count: number of GPUs
    gpu_memory_gb: per-GPU memory in GB
    lora_rank: LoRA rank (0 = full parameter training)

    Key formulas:
    - Model params: Ψ = model_size_b × 1e9
    - BF16 model: 2Ψ bytes (params) + 2Ψ bytes (gradients) + KΨ bytes (optimizer, K=12 for Adam)
    - FP32 master weights: 4Ψ bytes (for mixed precision)
    - Activation memory: varies by batch size and model architecture
    """
    Ψ = model_size_b * 1e9  # total parameters
    bytes_per_gb = 1e9

    # Base memory components (in bytes)
    bf16_params = 2 * Ψ              # BF16 model parameters
    bf16_grads = 2 * Ψ               # BF16 gradients
    fp32_params = 4 * Ψ              # FP32 master copy (mixed precision)
    fp32_grads = 4 * Ψ               # FP32 gradients (for optimizer)
    optimizer_states = 4 * Ψ + 4 * Ψ # FP32 momentum + variance (Adam: 2 states)
    # Total without sharding: 2Ψ (model) + 2Ψ (grads) + (4+4+4+4)Ψ (optimizer+master) = 20Ψ

    # LoRA parameters
    if lora_rank > 0:
        # LoRA adds low-rank matrices: for each target layer, 2 × (in_dim × r + out_dim × r)
        # Approximate: LoRA params ≈ 0.1% to 0.5% of total params for r=16-64
        lora_fraction = 2 * lora_rank / 4096  # rough estimate based on hidden=4096
        lora_params = Ψ * lora_fraction
        lora_bf16_params = 2 * lora_params
        lora_bf16_grads = 2 * lora_params
        lora_optimizer = 12 * lora_params  # FP32 master + momentum + variance
        # Base model is frozen → no gradients, no optimizer states
        # Only LoRA parameters need gradients + optimizer
    else:
        lora_params = 0
        lora_bf16_params = 0
        lora_bf16_grads = 0
        lora_optimizer = 0

    # Activation memory estimate (per microbatch)
    # For transformer: ~2 × seq_len × hidden × batch_size (for each layer)
    # Rough: 1-2 GB per microbatch for 7B model with seq=2048, bs=1
    activation_per_microbatch_gb = model_size_b * 0.2  # rough estimate

    results = {}

    N = gpu_count  # DP size

    # 1. DDP (standard)
    ddp_model = bf16_params
    ddp_grads = bf16_grads + fp32_grads  # gradients stored in FP32 for optimizer
    ddp_optimizer = optimizer_states + fp32_params
    ddp_peak = ddp_model + ddp_grads + ddp_optimizer + activation_per_microbatch_gb * bytes_per_gb
    ddp_peak_gb = ddp_peak / bytes_per_gb
    results["DDP"] = {
        "description": "Standard Data Parallelism",
        "model_gb": bf16_params / bytes_per_gb,
        "gradients_gb": (bf16_grads + fp32_grads) / bytes_per_gb,
        "optimizer_gb": (optimizer_states + fp32_params) / bytes_per_gb,
        "activation_gb": activation_per_microbatch_gb,
        "peak_per_gpu_gb": round(ddp_peak_gb, 2),
        "fits_gpu": ddp_peak_gb <= gpu_memory_gb,
    }

    # 2. ZeRO-1 (optimizer state sharding)
    zero1_model = bf16_params + fp32_params  # full model on each GPU
    zero1_grads = bf16_grads + fp32_grads    # full gradients
    zero1_optimizer = (optimizer_states) / N  # optimizer states sharded across N GPUs
    zero1_peak = zero1_model + zero1_grads + zero1_optimizer + activation_per_microbatch_gb * bytes_per_gb
    zero1_peak_gb = zero1_peak / bytes_per_gb
    results["ZeRO-1"] = {
        "description": "Optimizer state sharding across N GPUs",
        "model_gb": (bf16_params + fp32_params) / bytes_per_gb,
        "gradients_gb": (bf16_grads + fp32_grads) / bytes_per_gb,
        "optimizer_gb": optimizer_states / N / bytes_per_gb,
        "activation_gb": activation_per_microbatch_gb,
        "peak_per_gpu_gb": round(zero1_peak_gb, 2),
        "memory_saving_vs_ddp": round((ddp_peak_gb - zero1_peak_gb) / ddp_peak_gb * 100, 1),
        "fits_gpu": zero1_peak_gb <= gpu_memory_gb,
        "comm_overhead": "Same as DDP (AllReduce gradients)",
    }

    # 3. ZeRO-2 (optimizer + gradient sharding)
    zero2_model = bf16_params + fp32_params  # full model
    zero2_grads = (bf16_grads + fp32_grads) / N  # gradients sharded (ReduceScatter)
    zero2_optimizer = optimizer_states / N   # optimizer sharded
    zero2_peak = zero2_model + zero2_grads + zero2_optimizer + activation_per_microbatch_gb * bytes_per_gb
    zero2_peak_gb = zero2_peak / bytes_per_gb
    results["ZeRO-2"] = {
        "description": "Optimizer + gradient sharding",
        "model_gb": (bf16_params + fp32_params) / bytes_per_gb,
        "gradients_gb": (bf16_grads + fp32_grads) / N / bytes_per_gb,
        "optimizer_gb": optimizer_states / N / bytes_per_gb,
        "activation_gb": activation_per_microbatch_gb,
        "peak_per_gpu_gb": round(zero2_peak_gb, 2),
        "memory_saving_vs_ddp": round((ddp_peak_gb - zero2_peak_gb) / ddp_peak_gb * 100, 1),
        "fits_gpu": zero2_peak_gb <= gpu_memory_gb,
        "comm_overhead": "ReduceScatter gradients → same bandwidth as AllReduce",
    }

    # 4. ZeRO-2 + CPU Adam (optimizer offload to CPU)
    zero2_cpu_model = bf16_params  # BF16 model on GPU (no FP32 master on GPU)
    zero2_cpu_grads = bf16_grads / N  # BF16 gradients sharded
    zero2_cpu_optimizer = 0  # optimizer on CPU!
    zero2_cpu_peak = zero2_cpu_model + zero2_cpu_grads + zero2_cpu_optimizer + activation_per_microbatch_gb * bytes_per_gb
    zero2_cpu_peak_gb = zero2_cpu_peak / bytes_per_gb
    results["ZeRO-2+CPU_Adam"] = {
        "description": "ZeRO-2 with CPU optimizer offload",
        "model_gb": bf16_params / bytes_per_gb,
        "gradients_gb": bf16_grads / N / bytes_per_gb,
        "optimizer_gpu_gb": 0,
        "optimizer_cpu_gb": round((optimizer_states + fp32_params) / bytes_per_gb, 2),
        "activation_gb": activation_per_microbatch_gb,
        "peak_per_gpu_gb": round(zero2_cpu_peak_gb, 2),
        "fits_gpu": zero2_cpu_peak_gb <= gpu_memory_gb,
        "comm_overhead": "ReduceScatter + CPU↔GPU transfer for optimizer",
        "rtx4090_recommended": True,
    }

    # 5. ZeRO-3 (optimizer + gradient + parameter sharding)
    zero3_model = (bf16_params + fp32_params) / N  # parameters sharded
    zero3_grads = (bf16_grads + fp32_grads) / N    # gradients sharded
    zero3_optimizer = optimizer_states / N           # optimizer sharded
    # But need temp buffers for AllGather during forward/backward
    zero3_allgather_temp = bf16_params  # temporary full params during forward
    zero3_peak = zero3_model + zero3_grads + zero3_optimizer + zero3_allgather_temp + activation_per_microbatch_gb * bytes_per_gb
    zero3_peak_gb = zero3_peak / bytes_per_gb
    results["ZeRO-3"] = {
        "description": "Optimizer + gradient + parameter sharding",
        "model_gb": (bf16_params + fp32_params) / N / bytes_per_gb,
        "gradients_gb": (bf16_grads + fp32_grads) / N / bytes_per_gb,
        "optimizer_gb": optimizer_states / N / bytes_per_gb,
        "allgather_temp_gb": bf16_params / bytes_per_gb,
        "activation_gb": activation_per_microbatch_gb,
        "peak_per_gpu_gb": round(zero3_peak_gb, 2),
        "memory_saving_vs_ddp": round((ddp_peak_gb - zero3_peak_gb) / ddp_peak_gb * 100, 1),
        "fits_gpu": zero3_peak_gb <= gpu_memory_gb,
        "comm_overhead": "3×: AllGather(params fwd) + ReduceScatter(grads bwd) + AllGather(params bwd) = 3Ψ",
        "note": "Peak includes AllGather temp buffer for current layer params",
    }

    # 6. FSDP1 (FlatParameter, similar to ZeRO-3)
    fsdp1_peak = zero3_peak  # approximately same as ZeRO-3
    fsdp1_peak_gb = fsdp1_peak / bytes_per_gb
    results["FSDP1"] = {
        "description": "PyTorch FSDP1 (FlatParameter sharding)",
        "peak_per_gpu_gb": round(fsdp1_peak_gb, 2),
        "fits_gpu": fsdp1_peak_gb <= gpu_memory_gb,
        "comm_overhead": "Same as ZeRO-3: 3Ψ",
        "vs_zero3": "Similar peak, but FlatParameter may have padding waste",
        "note": "FlatParameter → fixed-size shard → may waste memory on small params",
    }

    # 7. FSDP2 (DTensor per-parameter sharding)
    fsdp2_model = bf16_params / N  # no padding waste!
    fsdp2_grads = bf16_grads / N   # per-parameter ReduceScatter
    fsdp2_optimizer = optimizer_states / N  # per-parameter optimizer shard
    fsdp2_allgather_temp = bf16_params  # still need temp for forward
    fsdp2_peak = fsdp2_model + fsdp2_grads + fsdp2_optimizer + fsdp2_allgather_temp + activation_per_microbatch_gb * bytes_per_gb
    fsdp2_peak_gb = fsdp2_peak / bytes_per_gb
    results["FSDP2"] = {
        "description": "PyTorch FSDP2 (DTensor per-parameter sharding)",
        "peak_per_gpu_gb": round(fsdp2_peak_gb, 2),
        "fits_gpu": fsdp2_peak_gb <= gpu_memory_gb,
        "comm_overhead": "2Ψ: ReduceScatter(grads) + AllGather(params) = 2Ψ (no backward AllGather needed)",
        "vs_fsdp1": "No padding waste + torch.compile compatible + TP compatible",
        "compile_compatible": True,
        "note": "DTensor → per-param sharding → no padding → compile sees static shapes",
    }

    # 8. LoRA + ZeRO-2 (RTX 4090 recommended)
    if lora_rank > 0:
        lora_model_base = bf16_params  # frozen base model (no gradients)
        lora_trainable = lora_bf16_params + lora_bf16_grads + lora_optimizer  # trainable LoRA
        lora_peak = lora_model_base + lora_trainable + activation_per_microbatch_gb * bytes_per_gb
        lora_peak_gb = lora_peak / bytes_per_gb
        lora_key = f"LoRA_r{lora_rank}_ZeRO-2"
        results[lora_key] = {
            "description": f"LoRA r={lora_rank} with ZeRO-2 gradient sharding",
            "base_model_gb": bf16_params / bytes_per_gb,
            "lora_trainable_gb": (lora_bf16_params + lora_bf16_grads + lora_optimizer) / bytes_per_gb,
            "activation_gb": activation_per_microbatch_gb,
            "peak_per_gpu_gb": round(lora_peak_gb, 2),
            "fits_gpu": lora_peak_gb <= gpu_memory_gb,
            "rtx4090_recommended": True,
        }

    # 9. LoRA + CPU Adam (RTX 4090 single GPU)
    if lora_rank > 0:
        lora_cpu_model_base = bf16_params  # frozen base on GPU
        lora_cpu_trainable_gpu = lora_bf16_params  # LoRA params on GPU (BF16)
        lora_cpu_grads_gpu = lora_bf16_grads  # LoRA grads on GPU
        lora_cpu_optimizer_cpu = lora_optimizer  # optimizer on CPU
        lora_cpu_peak = lora_cpu_model_base + lora_cpu_trainable_gpu + lora_cpu_grads_gpu + activation_per_microbatch_gb * bytes_per_gb
        lora_cpu_peak_gb = lora_cpu_peak / bytes_per_gb
        lora_cpu_key = f"LoRA_r{lora_rank}_CPU_Adam"
        results[lora_cpu_key] = {
            "description": f"LoRA r={lora_rank} with CPU Adam optimizer",
            "base_model_gpu_gb": bf16_params / bytes_per_gb,
            "lora_gpu_gb": (lora_bf16_params + lora_bf16_grads) / bytes_per_gb,
            "optimizer_cpu_gb": round(lora_optimizer / bytes_per_gb, 2),
            "activation_gpu_gb": activation_per_microbatch_gb,
            "peak_per_gpu_gb": round(lora_cpu_peak_gb, 2),
            "fits_gpu": lora_cpu_peak_gb <= gpu_memory_gb,
            "rtx4090_recommended": True,
            "note": "Single GPU optimal: no distributed ops needed!",
        }

    # 10. GRPO training (no critic!)
    if lora_rank > 0:
        # GRPO: no value function → saves ~same as base model memory
        # Alternating forward/backward → peak = model + LoRA grads + activation
        grpo_peak = bf16_params + lora_bf16_params + lora_bf16_grads + activation_per_microbatch_gb * bytes_per_gb * 2  # double activation for fwd+bwd
        grpo_peak_gb = grpo_peak / bytes_per_gb
        results["GRPO_LoRA_r{}".format(lora_rank)] = {
            "description": f"GRPO with LoRA r={lora_rank} (no critic!)",
            "base_model_gb": bf16_params / bytes_per_gb,
            "lora_gb": (lora_bf16_params + lora_bf16_grads) / bytes_per_gb,
            "activation_gb": activation_per_microbatch_gb * 2,
            "no_critic_saving_gb": round(bf16_params / bytes_per_gb, 2),
            "peak_per_gpu_gb": round(grpo_peak_gb, 2),
            "fits_gpu": grpo_peak_gb <= gpu_memory_gb,
            "rtx4090_recommended": True,
            "note": "GRPO advantage = group mean → no critic model → saves ~Ψ memory!",
        }

    # Summary comparison
    summary = {}
    for name, data in results.items():
        summary[name] = {
            "peak_per_gpu_gb": data["peak_per_gpu_gb"],
            "fits_gpu": data["fits_gpu"],
        }

    results["_summary"] = summary
    results["_config"] = {
        "model_size_b": model_size_b,
        "gpu_count": gpu_count,
        "gpu_memory_gb": gpu_memory_gb,
        "lora_rank": lora_rank,
        "total_params_Ψ": Ψ,
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="ZeRO/FSDP Memory Peak Simulator")
    parser.add_argument("--model-size", type=float, default=7, help="Model size in billions of params")
    parser.add_argument("--gpu-count", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--gpu-memory", type=float, default=24, help="Per-GPU memory in GB")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (0=no LoRA)")
    args = parser.parse_args()

    results = simulate_memory(args.model_size, args.gpu_count, args.gpu_memory, args.lora_rank)

    output_path = RESULTS_DIR / "zero_memory_peak_simulator.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    gpu_mem = args.gpu_memory
    print("=" * 70)
    print(f"ZeRO/FSDP Memory Peak Simulator ({args.model_size}B model, {args.gpu_count} GPUs, {gpu_mem}GB/GPU)")
    print("=" * 70)

    print(f"\n{'Strategy':<22} {'Peak/GPU':>10} {'Fits?':>8} {'Comm Cost':>10} {'Notes'}")
    print("-" * 70)

    for name, data in results.items():
        if name.startswith("_"):
            continue
        peak = data.get("peak_per_gpu_gb", 0)
        fits = "✓" if data.get("fits_gpu", False) else "✗"
        comm = data.get("comm_overhead", data.get("vs_zero3", data.get("vs_fsdp1", "")))
        if len(comm) > 20:
            comm = comm[:20]
        notes = data.get("rtx4090_recommended", False)
        note_str = "RTX4090✓" if notes else ""
        print(f"{name:<22} {peak:>8.2f}GB {fits:>8} {comm:>10} {note_str}")

    print("\n--- Key Insights ---")
    print(f"1. DDP peak = ~20Ψ/N → {results['DDP']['peak_per_gpu_gb']}GB/GPU → {'✓' if results['DDP']['fits_gpu'] else '✗'} fits {gpu_mem}GB")
    print(f"2. ZeRO-3 saves most memory but 3Ψ comm overhead + compile incompatible")
    print(f"3. FSDP2: 2Ψ comm + compile compatible + no padding waste → best for research")
    print(f"4. LoRA+CPU_Adam: single GPU → no distributed ops → RTX 4090 optimal")
    print(f"5. GRPO LoRA: no critic → saves Ψ memory → RTX 4090 recommended")

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
