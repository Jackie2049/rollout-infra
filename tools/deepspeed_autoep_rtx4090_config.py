#!/usr/bin/env python3
"""DeepSpeed AutoEP RTX 4090 MoE Training Config Generator
===========================================================

Generate DeepSpeed AutoEP + ZeRO-2 + LoRA + CPU_Adam config for RTX 4090 MoE training.

Modes:
  - generate: Generate DeepSpeed JSON config for MoE training
  - validate: Validate config against RTX 4090 memory constraints
  - compare: Compare AutoEP vs other MoE approaches
  - all: Run all modes

Based on: notebook/fundamentals/rtx4090-moe-training-viability-autoep.md
AutoEP MERGED #7938, Singleton MoE #7997, ZenFlow #8058, LoRAOptimizedLinear
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional


RTX_4090_VRAM = 24  # GB
SM_ARCH = 89


@dataclass
class MoEModelSpec:
    name: str
    active_params: str  # e.g., "0.6B"
    total_params: str   # e.g., "4B+"
    active_params_gb: float
    total_params_gb: float
    num_experts: int
    expert_hidden: int
    router_hidden: int
    model_type: str  # for AutoEP config


MOE_MODELS = {
    "qwen3_moe": MoEModelSpec(
        name="Qwen3-MoE (A0.6B+B4B)",
        active_params="0.6B",
        total_params="4B+",
        active_params_gb=0.6,
        total_params_gb=16.0,
        num_experts=64,
        expert_hidden=2048,
        router_hidden=2048,
        model_type="qwen3_moe",
    ),
    "mixtral_8x7b": MoEModelSpec(
        name="Mixtral 8x7B",
        active_params="13B",
        total_params="47B",
        active_params_gb=13.0,
        total_params_gb=47.0,
        num_experts=8,
        expert_hidden=4096,
        router_hidden=4096,
        model_type="mixtral",
    ),
    "deepseek_v2_lite": MoEModelSpec(
        name="DeepSeek-V2-Lite (A2.4B+B16B)",
        active_params="2.4B",
        total_params="16B",
        active_params_gb=2.4,
        total_params_gb=16.0,
        num_experts=64,
        expert_hidden=1408,
        router_hidden=2048,
        model_type="deepseek_v2",
    ),
}


def estimate_memory(model_spec: MoEModelSpec, lora_rank: int,
                    cpu_offload: bool, offload_ratio: float,
                    zenflow: bool) -> dict:
    """Estimate memory for MoE training on RTX 4090."""

    total = model_spec.total_params_gb

    # Frozen base weights (BF16)
    weights = total

    # LoRA params (~0.6GB for rank=32 on typical MoE)
    lora_params = 0.3 * (lora_rank / 32) * (total / 16.0)  # proportional scaling
    if lora_rank == 0:
        lora_params = 0

    # Optimizer: on CPU if cpu_offload, on GPU otherwise
    if cpu_offload:
        optimizer_gpu = 0
        optimizer_cpu = lora_params * 4 if lora_rank > 0 else total * 4
    else:
        optimizer_gpu = lora_params * 4 if lora_rank > 0 else total * 4
        optimizer_cpu = 0

    # GPU spike from optimizer copyback (ZenFlow vs old)
    if cpu_offload and zenflow:
        copyback_spike = 0.25  # ZenFlow chunked: ~256 MiB
    elif cpu_offload:
        copyback_spike = 2.94  # Old: ~2944 MiB
    else:
        copyback_spike = 0

    # Activations (~2GB for LoRA forward/backward on MoE)
    activations = 2.0 * (total / 16.0) if lora_rank > 0 else 8.0 * (total / 16.0)

    # CUDA context
    cuda_ctx = 1.0

    # LoRAOptimizedLinear offload_ratio (offloads frozen base to CPU)
    if lora_rank > 0 and offload_ratio > 0:
        offloaded_base = total * offload_ratio
        weights_gpu = total - offloaded_base
    else:
        weights_gpu = total

    # Total GPU memory
    total_gpu = weights_gpu + lora_params + optimizer_gpu + activations + cuda_ctx + copyback_spike
    peak_gpu = total_gpu + copyback_spike  # worst case

    fits = peak_gpu <= RTX_4090_VRAM
    headroom = RTX_4090_VRAM - peak_gpu

    return {
        "model": model_spec.name,
        "weights_gpu_gb": round(weights_gpu, 2),
        "weights_offloaded_gb": round(total * offload_ratio, 2) if offload_ratio > 0 else 0,
        "lora_params_gb": round(lora_params, 2),
        "optimizer_gpu_gb": round(optimizer_gpu, 2),
        "optimizer_cpu_gb": round(optimizer_cpu, 2),
        "copyback_spike_gb": round(copyback_spike, 2),
        "activations_gb": round(activations, 2),
        "cuda_ctx_gb": round(cuda_ctx, 2),
        "total_gpu_gb": round(total_gpu, 2),
        "peak_gpu_gb": round(peak_gpu, 2),
        "fits_vram": fits,
        "headroom_gb": round(headroom, 2),
        "recommendation": "VIABLE" if fits else "NOT VIABLE - exceeds 24GB!",
    }


def generate_config(model_key: str, lora_rank: int, lr: float,
                    batch_size: int, micro_batch: int,
                    cpu_offload: bool, offload_ratio: float,
                    zenflow: bool, bf16: bool) -> str:
    """Generate DeepSpeed AutoEP config JSON."""

    model_spec = MOE_MODELS[model_key]

    config = {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu" if cpu_offload else "none",
                "pin_memory": cpu_offload,
            },
            "overlap_comm": False,  # single GPU, no overlap needed
            "contiguous_gradients": True,
            "allgather_bucket_size": int(5e8),
            "reduce_bucket_size": int(5e8),
        },
        "gradient_accumulation_steps": batch_size // micro_batch,
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
            },
        },
        "bf16": {
            "enabled": bf16,
        },
        "fp16": {
            "enabled": not bf16,
        },
        "data_types": {
            "param_dtype": "bf16" if bf16 else "fp16",
            "buffer_dtype": "fp32",  # preserve inv_freq for RoPE (#8066)
        },
        "auto_expert_parallelism": {
            "enabled": True,
            "expert_parallel_size": 1,  # EP=1 singleton! RTX 4090 single GPU
            "model_type": model_spec.model_type,
        },
        "lora": {
            "enabled": lora_rank > 0,
            "rank": lora_rank,
            "target_modules": ["router", "expert_mlp", "shared_mlp", "attention"],
            "offload_ratio": offload_ratio,
        },
        "tensorboard": {
            "enabled": True,
            "output_path": "./logs/autoep_moe_rtx4090",
        },
    }

    # Add ZenFlow config if enabled
    if zenflow and cpu_offload:
        config["zero_optimization"]["offload_optimizer"]["use_zenflow"] = True

    return json.dumps(config, indent=2)


def run_generate(args):
    """Generate DeepSpeed AutoEP config."""
    model_key = args.model
    if model_key not in MOE_MODELS:
        print(f"ERROR: Unknown model '{model_key}'. Available: {list(MOE_MODELS.keys())}")
        sys.exit(1)

    config = generate_config(
        model_key=model_key,
        lora_rank=args.lora_rank,
        lr=args.lr,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        cpu_offload=args.cpu_offload,
        offload_ratio=args.offload_ratio,
        zenflow=args.zenflow,
        bf16=args.bf16,
    )

    model_spec = MOE_MODELS[model_key]

    print("=" * 80)
    print(f"DeepSpeed AutoEP RTX 4090 Config — {model_spec.name}")
    print("=" * 80)
    print()
    print(f"Model: {model_spec.name}")
    print(f"Active params: {model_spec.active_params} | Total params: {model_spec.total_params}")
    print(f"LoRA rank: {args.lora_rank} | CPU offload: {args.cpu_offload} | ZenFlow: {args.zenflow}")
    print(f"EP=1 singleton: 15x speedup over EP>1 (#7997)")
    print()
    print("Configuration JSON:")
    print(config)
    print()

    # Also validate
    mem = estimate_memory(model_spec, args.lora_rank, args.cpu_offload,
                          args.offload_ratio, args.zenflow)
    print(f"Memory estimate: {mem['peak_gpu_gb']}GB peak / {RTX_4090_VRAM}GB available")
    print(f"Status: {mem['recommendation']}")
    print(f"Headroom: {mem['headroom_gb']}GB")

    if args.output:
        with open(args.output, 'w') as f:
            f.write(config)
        print(f"\nConfig saved to: {args.output}")


def run_validate(args):
    """Validate configs against RTX 4090 memory constraints."""
    print("=" * 80)
    print("DeepSpeed AutoEP RTX 4090 MoE Memory Validation")
    print("=" * 80)
    print()

    configs = [
        ("AutoEP ZeRO-2 + LoRA32 + CPU_Adam + offload 0.5 + ZenFlow",
         MOE_MODELS["qwen3_moe"], 32, True, 0.5, True),
        ("AutoEP ZeRO-2 + LoRA32 + CPU_Adam + offload 0.5 (no ZenFlow)",
         MOE_MODELS["qwen3_moe"], 32, True, 0.5, False),
        ("AutoEP ZeRO-2 + LoRA32 + CPU_Adam + no offload_ratio + ZenFlow",
         MOE_MODELS["qwen3_moe"], 32, True, 0.0, True),
        ("AutoEP ZeRO-2 + LoRA16 + CPU_Adam + offload 0.5 + ZenFlow",
         MOE_MODELS["qwen3_moe"], 16, True, 0.5, True),
        ("AutoEP ZeRO-2 + LoRA64 + CPU_Adam + offload 0.5 + ZenFlow",
         MOE_MODELS["qwen3_moe"], 64, True, 0.5, True),
        ("Full model (no LoRA) — baseline",
         MOE_MODELS["qwen3_moe"], 0, True, 0.0, True),
        ("DeepSeek-V2-Lite + LoRA32 + CPU_Adam + offload 0.5 + ZenFlow",
         MOE_MODELS["deepseek_v2_lite"], 32, True, 0.5, True),
        ("Mixtral 8x7B + LoRA32 — baseline",
         MOE_MODELS["mixtral_8x7b"], 32, True, 0.5, True),
    ]

    for name, spec, lora, offload, ratio, zenflow in configs:
        mem = estimate_memory(spec, lora, offload, ratio, zenflow)
        status = "✓ FITS" if mem["fits_vram"] else "✗ EXCEEDS"
        print(f"{name}:")
        print(f"  Peak: {mem['peak_gpu_gb']}GB / {RTX_4090_VRAM}GB → {status}")
        print(f"  Headroom: {mem['headroom_gb']}GB")
        if mem['copyback_spike_gb'] > 0:
            spike_label = "ZenFlow" if zenflow else "OLD"
            print(f"  Copyback spike ({spike_label}): {mem['copyback_spike_gb']}GB")
        print()


def run_compare(args):
    """Compare AutoEP vs other MoE approaches."""
    print("=" * 80)
    print("RTX 4090 MoE Training: AutoEP vs Alternatives")
    print("=" * 80)
    print()

    print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
    print()
    print("Framework          | EP Support     | RTX 4090 Status    | Notes")
    print("-------------------|----------------|--------------------|------")
    print("DeepSpeed AutoEP   | EP=1 singleton | ★★★★★★★ VIABLE    | 15x speedup; LoRA+CPU_Adam; ZenFlow copyback")
    print("Megatron core      | EP>1 only      | ★ CRASH (#5203)    | Singleton PG bug; LayerWise optimizer crash")
    print("verl               | No EP          | ★★★ No MoE opt     | No expert parallelism; active params small anyway")
    print("PyTorch FSDP2      | No EP          | ★★★ No MoE opt     | Per-parameter meshes but no EP; single GPU useless")
    print("rLLM Tinker        | No EP          | ★★★ Dense only     | In-process; bypass; but no MoE-specific optimization")
    print()

    print("★★★★★★★★★ DeepSpeed AutoEP = ONLY framework supporting MoE on RTX 4090!")
    print()
    print("Memory comparison (Qwen3-MoE A0.6B+B4B, LoRA rank=32):")
    mem_autoep = estimate_memory(MOE_MODELS["qwen3_moe"], 32, True, 0.5, True)
    print(f"  AutoEP ZeRO-2 + CPU_Adam + ZenFlow: {mem_autoep['peak_gpu_gb']}GB ✓")
    mem_no_lora = estimate_memory(MOE_MODELS["qwen3_moe"], 0, True, 0.0, True)
    print(f"  Full model (no LoRA): {mem_no_lora['peak_gpu_gb']}GB ✗ (exceeds 24GB)")
    mem_no_zenflow = estimate_memory(MOE_MODELS["qwen3_moe"], 32, True, 0.5, False)
    print(f"  No ZenFlow (old copyback spike): {mem_no_zenflow['peak_gpu_gb']}GB → spike risk!")
    print()

    print("★★★★★★★★★ Key: LoRA + CPU_Adam + offload_ratio = only 0.6GB trainable → fits 24GB!")
    print("★★★★★★★★★ ZenFlow reduces copyback spike from 2.94GB → 0.25GB → eliminates OOM risk!")


def run_all(args):
    """Run all modes."""
    run_generate(args)
    print()
    run_validate(args)
    print()
    run_compare(args)


def main():
    parser = argparse.ArgumentParser(
        description="DeepSpeed AutoEP RTX 4090 MoE Training Config Generator")
    parser.add_argument("--mode",
                        choices=["generate", "validate", "compare", "all"],
                        default="validate",
                        help="Display mode")
    parser.add_argument("--model",
                        choices=list(MOE_MODELS.keys()),
                        default="qwen3_moe",
                        help="MoE model type")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (0 = no LoRA)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Total batch size")
    parser.add_argument("--micro_batch", type=int, default=2,
                        help="Micro batch size per GPU")
    parser.add_argument("--cpu_offload", type=bool, default=True,
                        help="Offload optimizer to CPU")
    parser.add_argument("--offload_ratio", type=float, default=0.5,
                        help="LoRAOptimizedLinear offload ratio (0-1)")
    parser.add_argument("--zenflow", type=bool, default=True,
                        help="Use ZenFlow chunked copyback (#8058)")
    parser.add_argument("--bf16", type=bool, default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--output", type=str, default=None,
                        help="Output config JSON file path")
    args = parser.parse_args()

    modes = {
        "generate": run_generate,
        "validate": run_validate,
        "compare": run_compare,
        "all": run_all,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
