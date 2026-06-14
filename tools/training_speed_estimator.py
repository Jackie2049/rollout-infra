#!/usr/bin/env python3
"""
Training Speed Estimator

Estimates training throughput (tokens/sec, steps/sec, time-to-train)
for different distributed training strategies, combining:
- Computation time (forward + backward)
- Communication overhead (AllReduce, AllGather, etc.)
- Compile acceleration (FSDP2+compile vs ZeRO-3 eager)
- LoRA reduction (fewer trainable params → faster backward)
- GRPO savings (no critic → fewer forward passes)

Key formulas:
- Compute time: Ψ × GEMM_throughput × 3 (fwd + bwd + optimizer)
- Communication: depends on strategy (see comm_cost_calculator)
- Compile speedup: 2-3x for FSDP2+compile (kernel fusion)
- LoRA speedup: trainable_fraction × backward reduction

Usage:
    python3 training_speed_estimator.py
    python3 training_speed_estimator.py --gpu-count 8 --interconnect nvlink --model-size 7 --strategy fsdp2-compile
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# GPU compute throughput estimates (TFLOPS for BF16 GEMM)
GPU_GEMM_TFLOPS = {
    "rtx4090": 169.6,   # SM 8.9 Ada
    "a100_80": 312,     # SM 8.0 Ampere
    "a100_40": 312,     # Same GEMM, less memory
    "h100_sxm": 990,    # SM 9.0 Hopper
    "h100_pcie": 756,   # PCIe variant
    "h200": 990,        # Same compute, more memory
    "b200": 2250,       # SM 100 Blackwell (est)
}

# Communication bandwidth (GB/s bidirectional)
COMM_BANDWIDTH = {
    "nvlink_a100": 300,
    "nvlink_h100": 450,
    "nvlink_h200": 600,
    "pcie_gen4": 25,
    "pcie_gen5": 32,
    " roce_200": 25,
}

# Typical batch sizes and sequence lengths for 7B model
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 2048

def estimate_training_speed(model_size_b, gpu_count, gpu_type, interconnect,
                            strategy, lora_rank, training_type,
                            batch_size, seq_len):
    """Estimate training speed for given configuration"""

    Ψ = model_size_b * 1e9
    bf16_model_gb = 2 * Ψ / 1e9
    gemm_tflops = GPU_GEMM_TFLOPS.get(gpu_type, GPU_GEMM_TFLOPS["rtx4090"])

    # Tokens per step
    tokens_per_step = batch_size * seq_len * gpu_count

    # === Computation time estimate ===
    # Forward: 2Ψ FLOPS per token (matmul)
    # Backward: 4Ψ FLOPS per token (2x forward for gradient computation)
    # Total: 6Ψ FLOPS per token for full parameter training
    fwd_flops_per_token = 2 * Ψ  # approximate
    bwd_flops_per_token = 4 * Ψ  # approximate (2x forward)

    # LoRA reduction
    if lora_rank > 0:
        lora_fraction = 2 * lora_rank / 4096
        # Forward: still need full model forward (base is frozen)
        # Backward: only LoRA params need gradients → much less
        bwd_flops_per_token = fwd_flops_per_token * lora_fraction * 2 + fwd_flops_per_token * 0.1  # LoRA backward + base gradient checkpointing overhead
        total_flops_per_token = fwd_flops_per_token + bwd_flops_per_token
    else:
        total_flops_per_token = fwd_flops_per_token + bwd_flops_per_token

    # GRPO: no critic model
    if training_type == "grpo":
        # GRPO needs: actor forward + actor backward (no critic)
        # PPO needs: actor forward + critic forward + actor backward + critic backward
        # GRPO saves: critic forward + critic backward ≈ 50% of compute
        grpo_factor = 0.55  # slight overhead for rollout generation
    elif training_type == "ppo":
        grpo_factor = 1.0   # full compute (actor + critic)
    else:
        grpo_factor = 1.0

    # Compute time per step (seconds)
    # FLOPS per step = total_flops_per_token * tokens_per_step * grpo_factor
    total_flops = total_flops_per_token * tokens_per_step * grpo_factor
    # GPU can overlap compute across GPUs, so total compute ≈ per-GPU compute
    per_gpu_flops = total_flops / gpu_count
    compute_time_s = per_gpu_flops / (gemm_tflops * 1e12)

    # === Communication time estimate ===
    # See comm_cost_calculator for detailed calculations
    N = gpu_count
    comm_bw = COMM_BANDWIDTH.get(interconnect, COMM_BANDWIDTH["pcie_gen4"])

    comm_time_s = 0
    if strategy == "ddp":
        # AllReduce gradients: data = bf16_model_gb * lora_fraction (if LoRA) or bf16_model_gb
        grad_gb = bf16_model_gb * (2 * lora_rank / 4096 if lora_rank > 0 else 1)
        # Ring AllReduce: 2*(N-1)/N * data / (bw * N/2)
        comm_time_s = grad_gb / (comm_bw * N / 2)  # simplified
    elif strategy == "zero2":
        # ReduceScatter gradients ≈ similar to AllReduce in time
        grad_gb = bf16_model_gb * (2 * lora_rank / 4096 if lora_rank > 0 else 1)
        comm_time_s = grad_gb / (comm_bw * N / 2)
    elif strategy == "zero2-cpu-adam":
        grad_gb = bf16_model_gb * (2 * lora_rank / 4096 if lora_rank > 0 else 1)
        comm_time_s = grad_gb / (comm_bw * N / 2)
        # CPU optimizer transfer overhead
        optimizer_gb = 12 * Ψ * (2 * lora_rank / 4096 if lora_rank > 0 else 1) / 1e9 / N
        cpu_transfer_s = optimizer_gb / 50  # CPU-GPU PCIe ~50GB/s
        comm_time_s += cpu_transfer_s
    elif strategy == "zero3":
        # 3Ψ: AllGather fwd + RS bwd + AllGather bwd
        total_comm_gb = bf16_model_gb * 3
        comm_time_s = total_comm_gb / (comm_bw * N / 2)
    elif strategy == "fsdp2-compile":
        # 2Ψ: AllGather fwd + RS bwd (no backward AllGather)
        total_comm_gb = bf16_model_gb * 2
        comm_time_s = total_comm_gb / (comm_bw * N / 2)
    elif strategy == "fsdp2-eager":
        total_comm_gb = bf16_model_gb * 2
        comm_time_s = total_comm_gb / (comm_bw * N / 2)
    elif strategy == "single-gpu-lora":
        comm_time_s = 0  # no distributed communication!
    elif strategy == "tinker-grpo-lora":
        comm_time_s = 0  # in-process, no distributed communication!

    # === Compile speedup ===
    compile_speedup = 1.0
    if strategy == "fsdp2-compile":
        # torch.compile fusion: RMSNorm+SiLU+Residual→1 Triton kernel
        # Reduces kernel launch overhead + memory round-trips
        # Estimate: 2-3x speedup on compute portion
        compile_speedup = 2.5  # moderate estimate
        compute_time_s /= compile_speedup

    # === Total time per step ===
    # Communication and computation can overlap to some degree
    # For NVLink: overlap is effective → total ≈ max(compute, comm)
    # For PCIe: overlap is limited → total ≈ compute + comm
    if interconnect in ["nvlink_a100", "nvlink_h100", "nvlink_h200"]:
        overlap_efficiency = 0.7  # 70% overlap possible
        total_time_s = compute_time_s + comm_time_s * (1 - overlap_efficiency)
    else:
        overlap_efficiency = 0.2  # 20% overlap on PCIe
        total_time_s = compute_time_s + comm_time_s * (1 - overlap_efficiency)

    # === Throughput metrics ===
    tokens_per_sec = tokens_per_step / total_time_s if total_time_s > 0 else 0
    steps_per_sec = 1.0 / total_time_s if total_time_s > 0 else 0
    samples_per_sec = batch_size * gpu_count / total_time_s if total_time_s > 0 else 0

    # === Time to train estimates ===
    # Assume 1B tokens for fine-tuning, 100B tokens for full training
    total_tokens_ft = 1e9   # fine-tuning
    total_tokens_full = 1e11  # full training (rough)
    time_to_ft_hours = total_tokens_ft / tokens_per_sec / 3600
    time_to_full_hours = total_tokens_full / tokens_per_sec / 3600

    results = {
        "strategy": strategy,
        "config": {
            "model_size_b": model_size_b,
            "gpu_count": gpu_count,
            "gpu_type": gpu_type,
            "interconnect": interconnect,
            "strategy": strategy,
            "lora_rank": lora_rank,
            "training_type": training_type,
            "batch_size": batch_size,
            "seq_len": seq_len,
        },
        "compute": {
            "total_flops_per_step": total_flops,
            "per_gpu_flops": per_gpu_flops,
            "compute_time_s": round(compute_time_s, 4),
            "gemm_tflops": gemm_tflops,
            "lora_fraction": 2 * lora_rank / 4096 if lora_rank > 0 else 1.0,
            "grpo_factor": grpo_factor,
            "compile_speedup": compile_speedup,
        },
        "communication": {
            "comm_time_s": round(comm_time_s, 4),
            "comm_volume_gb": round(bf16_model_gb * (3 if strategy == "zero3" else 2 if strategy in ["fsdp2-compile", "fsdp2-eager"] else 0.1 if lora_rank > 0 else 1), 2),
            "overlap_efficiency": overlap_efficiency,
        },
        "throughput": {
            "total_time_per_step_s": round(total_time_s, 4),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "steps_per_sec": round(steps_per_sec, 4),
            "samples_per_sec": round(samples_per_sec, 2),
        },
        "time_to_train": {
            "fine_tuning_1B_tokens_hours": round(time_to_ft_hours, 2),
            "full_training_100B_tokens_hours": round(time_to_full_hours, 2),
        },
        "memory_peak_gb": estimate_memory_peak(model_size_b, gpu_count, strategy, lora_rank),
        "fits_rtx4090": estimate_memory_peak(model_size_b, gpu_count, strategy, lora_rank) <= 24 if gpu_type == "rtx4090" else None,
    }

    return results

def estimate_memory_peak(model_size_b, gpu_count, strategy, lora_rank):
    """Quick memory peak estimate for strategy"""
    Ψ = model_size_b * 1e9
    bf16_model_gb = 2 * Ψ / 1e9
    N = gpu_count
    activation_gb = model_size_b * 0.2  # rough estimate

    if strategy == "ddp":
        return bf16_model_gb + bf16_model_gb + 12 * bf16_model_gb / 1 + activation_gb
    elif strategy == "zero2":
        return bf16_model_gb + bf16_model_gb / N + 12 * bf16_model_gb / N + activation_gb
    elif strategy == "zero2-cpu-adam":
        if lora_rank > 0:
            lora_gb = bf16_model_gb * 2 * lora_rank / 4096
            return bf16_model_gb + lora_gb + activation_gb
        else:
            return bf16_model_gb + bf16_model_gb / N + activation_gb
    elif strategy == "zero3":
        return bf16_model_gb / N + 12 * bf16_model_gb / N + bf16_model_gb + activation_gb
    elif strategy == "fsdp2-compile" or strategy == "fsdp2-eager":
        return bf16_model_gb / N + bf16_model_gb / N + 12 * bf16_model_gb / N + bf16_model_gb + activation_gb
    elif strategy == "single-gpu-lora":
        lora_gb = bf16_model_gb * 2 * lora_rank / 4096
        return bf16_model_gb + lora_gb + activation_gb
    elif strategy == "tinker-grpo-lora":
        lora_gb = bf16_model_gb * 2 * lora_rank / 4096
        return bf16_model_gb + lora_gb + activation_gb * 2  # GRPO double activation
    else:
        return bf16_model_gb * 20 / N

def main():
    parser = argparse.ArgumentParser(description="Training Speed Estimator")
    parser.add_argument("--model-size", type=float, default=7)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--gpu-type", type=str, default="rtx4090",
                       choices=list(GPU_GEMM_TFLOPS.keys()))
    parser.add_argument("--interconnect", type=str, default="pcie_gen4",
                       choices=list(COMM_BANDWIDTH.keys()) + ["none"])
    parser.add_argument("--strategy", type=str, default="tinker-grpo-lora",
                       choices=["ddp", "zero2", "zero2-cpu-adam", "zero3",
                                "fsdp2-compile", "fsdp2-eager",
                                "single-gpu-lora", "tinker-grpo-lora"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--training-type", type=str, default="grpo",
                       choices=["full", "lora", "grpo", "ppo"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    results = estimate_training_speed(
        args.model_size, args.gpu_count, args.gpu_type,
        args.interconnect, args.strategy, args.lora_rank,
        args.training_type, args.batch_size, args.seq_len
    )

    output_path = RESULTS_DIR / "training_speed_estimator.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print("=" * 70)
    print(f"Training Speed Estimator ({args.model_size}B, {args.gpu_count}×{args.gpu_type}, {args.strategy})")
    print("=" * 70)

    c = results["config"]
    print(f"\nConfig: {c['model_size_b']}B model, {c['gpu_count']}×{c['gpu_type']}, {c['interconnect']}")
    print(f"  Strategy: {c['strategy']}, LoRA r={c['lora_rank']}, {c['training_type']}")
    print(f"  Batch={c['batch_size']}, Seq={c['seq_len']}")

    comp = results["compute"]
    print(f"\nCompute:")
    print(f"  FLOPS/step: {comp['total_flops_per_step']:.2e}")
    print(f"  Compute time: {comp['compute_time_s']:.4f}s")
    print(f"  LoRA fraction: {comp['lora_fraction']*100:.1f}%")
    print(f"  GRPO factor: {comp['grpo_factor']:.2f}")
    print(f"  Compile speedup: {comp['compile_speedup']:.1f}x")

    comm = results["communication"]
    print(f"\nCommunication:")
    print(f"  Comm time: {comm['comm_time_s']:.4f}s")
    print(f"  Comm volume: {comm['comm_volume_gb']:.2f}GB")
    print(f"  Overlap efficiency: {comm['overlap_efficiency']*100:.0f}%")

    thr = results["throughput"]
    print(f"\nThroughput:")
    print(f"  Time/step: {thr['total_time_per_step_s']:.4f}s")
    print(f"  Tokens/sec: {thr['tokens_per_sec']:.2f}")
    print(f"  Steps/sec: {thr['steps_per_sec']:.4f}")
    print(f"  Samples/sec: {thr['samples_per_sec']:.2f}")

    ttt = results["time_to_train"]
    print(f"\nTime to train:")
    print(f"  Fine-tune 1B tokens: {ttt['fine_tuning_1B_tokens_hours']:.1f} hours")
    print(f"  Full train 100B tokens: {ttt['full_training_100B_tokens_hours']:.1f} hours")

    mem = results["memory_peak_gb"]
    fits = results["fits_rtx4090"]
    print(f"\nMemory: {mem:.2f}GB/GPU peak {'✓ fits 24GB' if fits else '✗ exceeds 24GB' if fits is not None else ''}")

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
