#!/usr/bin/env python3
"""
Training Strategy Decision Tool

Given hardware constraints and model size, recommends the optimal
distributed training strategy across 7 frameworks.

Decision factors:
- GPU count and memory
- GPU interconnect (NVLink, PCIe, none)
- Model size
- Training type (full, LoRA, GRPO)
- Inference vs training

Usage:
    python3 training_strategy_decision.py
    python3 training_strategy_decision.py --gpu-count 1 --gpu-memory 24 --model-size 7
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def decide_strategy(gpu_count, gpu_memory_gb, interconnect, model_size_b,
                    training_type, lora_rank):
    """Decide optimal training strategy based on constraints"""

    Ψ = model_size_b * 1e9
    bf16_model_gb = 2 * Ψ / 1e9  # BF16 model size in GB

    decisions = {}

    # ===== Framework-specific decisions =====

    # 1. DeepSpeed
    ds_options = {}
    if gpu_count == 1:
        if lora_rank > 0 and training_type in ["lora", "grpo"]:
            ds_options["ZeRO-2+CPU_Adam+LoRA"] = {
                "peak_gb": round(bf16_model_gb + 0.3 + 3, 1),
                "fits": gpu_memory_gb >= bf16_model_gb + 0.3 + 3,
                "recommended": True,
                "reason": "Single GPU: CPU Adam offload optimizer → LoRA trainable params only",
            }
        ds_options["ZeRO-3"] = {
            "peak_gb": "N/A (needs multi-GPU)",
            "fits": False,
            "recommended": False,
            "reason": "ZeRO-3 needs AllGather across GPUs → single GPU pointless",
        }
    elif interconnect == "nvlink":
        if lora_rank > 0:
            ds_options["ZeRO-2+LoRA"] = {
                "peak_gb": round(bf16_model_gb + 0.3 + 3, 1),
                "fits": gpu_memory_gb >= bf16_model_gb + 0.3 + 3,
                "recommended": gpu_memory_gb < bf16_model_gb * 2 + 10,
                "reason": "NVLink fast enough for ZeRO-2 ReduceScatter",
            }
        ds_options["ZeRO-3+overlap"] = {
            "peak_gb": round(bf16_model_gb / gpu_count + bf16_model_gb + 3, 1),
            "fits": True,
            "recommended": bf16_model_gb > gpu_memory_gb * 0.7,
            "reason": "ZeRO-3 with comm overlap → NVLink handles 3Ψ traffic",
        }
    else:  # PCIe
        ds_options["ZeRO-2+CPU_Adam+LoRA"] = {
            "peak_gb": round(bf16_model_gb + 0.3 + 3, 1),
            "fits": gpu_memory_gb >= bf16_model_gb + 0.3 + 3,
            "recommended": True,
            "reason": "PCIe bottleneck → avoid ZeRO-3 AllGather → LoRA+CPU Adam",
        }
        ds_options["ZeRO-3"] = {
            "fits": False,
            "recommended": False,
            "reason": "PCIe bottleneck for 3Ψ AllGather traffic → too slow",
        }
    decisions["DeepSpeed"] = ds_options

    # 2. Megatron-LM
    megatron_options = {}
    if gpu_count == 1:
        megatron_options["Single_GPU"] = {
            "recommended": True,
            "reason": "TP=1,PP=1,DP=1 → no distributed → fallback to PyTorch",
            "note": "Megatron优势在多GPU分布式 → 单GPU无优势",
        }
    elif interconnect == "nvlink":
        megatron_options["TP+DP"] = {
            "recommended": bf16_model_gb > gpu_memory_gb * 0.5,
            "reason": "NVLink fast for TP AllReduce → DP for data parallelism",
            "config": f"TP={min(gpu_count, 8)}, DP={gpu_count // min(gpu_count, 8)}",
        }
    else:  # PCIe
        megatron_options["DP_only"] = {
            "recommended": False,
            "reason": "PCIe bottleneck → TP>1灾难 → PP>1灾难 → 只能DP",
            "note": "DP=gpu_count但scaling差",
        }
    decisions["Megatron-LM"] = megatron_options

    # 3. vLLM (inference only)
    vllm_options = {}
    if training_type == "inference" or training_type == "serving":
        vllm_options["INT4+INT8KV+GQA"] = {
            "recommended": True,
            "reason": "Decode memory-bound → INT4 weights+INT8 KV → max throughput",
            "estimated_tps": f"~{model_size_b * 700} tok/s (7B INT4 baseline)",
        }
        vllm_options["BF16+FlashInfer"] = {
            "recommended": gpu_memory_gb >= bf16_model_gb + 4,
            "reason": "BF16 inference if memory permits → best accuracy",
        }
    else:
        vllm_options["rollout_engine"] = {
            "recommended": True,
            "reason": "vLLM作为verl/rLLM的rollout引擎 → 不独立训练",
        }
    decisions["vLLM"] = vllm_options

    # 4. verl
    verl_options = {}
    if gpu_count == 1:
        verl_options["GRPO+LoRA+CPU_Adam"] = {
            "recommended": True,
            "peak_gb": round(bf16_model_gb + 0.3 + 3 + bf16_model_gb * 0.1, 1),
            "fits": gpu_memory_gb >= bf16_model_gb + 4,
            "reason": "GRPO no critic → saves ~50% memory vs PPO → LoRA r=16 → CPU Adam",
            "note": "Single GPU → no FSDP/DDP needed → simplest path",
        }
        verl_options["PPO"] = {
            "recommended": False,
            "reason": f"PPO needs 2×model ({bf16_model_gb*2}GB actor+critic) → doesn't fit {gpu_memory_gb}GB",
        }
    elif interconnect == "nvlink":
        verl_options["GRPO+FSDP2+LoRA"] = {
            "recommended": True,
            "reason": "FSDP2 per-param sharding + NVLink fast → compile compatible",
        }
    else:
        verl_options["GRPO+LoRA+CPU_Adam"] = {
            "recommended": True,
            "reason": "PCIe bottleneck → single-GPU-like setup per node → GRPO+LoRA",
        }
    decisions["verl"] = verl_options

    # 5. rLLM
    rllm_options = {}
    if gpu_count == 1:
        rllm_options["UnifiedWorkflowEngine+TinkerBackend+GRPO+LoRA"] = {
            "recommended": True,
            "reason": "Tinker in-process → no Ray overhead → single GPU fastest",
            "peak_gb": round(bf16_model_gb + 0.3 + 3, 1),
            "fits": gpu_memory_gb >= bf16_model_gb + 4,
        }
    elif interconnect == "nvlink" and gpu_count >= 4:
        rllm_options["UnifiedWorkflowEngine+VerlBackend+GRPO+LoRA"] = {
            "recommended": True,
            "reason": "Ray distributed → colocated actor+rollout → NVLink fast",
        }
    else:
        rllm_options["TinkerBackend+GRPO+LoRA"] = {
            "recommended": True,
            "reason": "PCIe → treat as multiple single-GPU → Tinker per node",
        }
    decisions["rLLM"] = rllm_options

    # 6. MindIE
    mindie_options = {}
    if interconnect == "hccs":  # Ascend NPU
        mindie_options["MindIE-serving"] = {
            "recommended": True,
            "reason": "Ascend NPU → MindIE native → ATB+HCCL optimized",
        }
    else:  # NVIDIA GPU
        mindie_options["Not_applicable"] = {
            "recommended": False,
            "reason": "MindIE = Ascend NPU专用 → NVIDIA GPU用vLLM替代",
            "alternative": "vLLM-Ascend (if Ascend available)",
        }
    decisions["MindIE"] = mindie_options

    # 7. PyTorch
    pytorch_options = {}
    if gpu_count == 1:
        pytorch_options["DDP+LoRA"] = {
            "recommended": False,
            "reason": "DDP on single GPU = no parallelism → just standard training",
        }
        pytorch_options["FSDP2+compile+LoRA"] = {
            "recommended": gpu_count >= 2 and interconnect == "nvlink",
            "reason": "FSDP2 per-param + compile → NVLink needed for sharding",
        }
    elif interconnect == "nvlink":
        pytorch_options["FSDP2+compile"] = {
            "recommended": True,
            "reason": "FSDP2+compile = best combination → compile sees static shapes → max fusion",
            "note": "ZeRO-3+compile = incompatible (dynamic AllGather → graph break)",
        }
    else:
        pytorch_options["DDP+CPU_Adam+LoRA"] = {
            "recommended": True,
            "reason": "PCIe → only DDP(AllReduce gradients) → slow but works",
        }
    decisions["PyTorch"] = pytorch_options

    # ===== Overall recommendation =====
    best = None
    best_reason = ""
    if gpu_count == 1 and training_type in ["lora", "grpo"]:
        if interconnect == "none":
            best = "rLLM (TinkerBackend+GRPO+LoRA)"
            best_reason = "In-process → no distributed overhead → single GPU optimal"
        else:
            best = "verl (GRPO+LoRA+CPU_Adam)"
            best_reason = "GRPO saves 50% memory → CPU Adam offload → simplest"
    elif gpu_count >= 4 and interconnect == "nvlink":
        best = "PyTorch (FSDP2+compile) or verl (GRPO+FSDP2)"
        best_reason = "NVLink → FSDP2+compile compatible → max fusion speed"
    elif interconnect == "pcie":
        best = "DeepSpeed (ZeRO-2+CPU_Adam+LoRA) or rLLM (Tinker+GRPO)"
        best_reason = "PCIe bottleneck → avoid sharding → LoRA+CPU offload"
    elif training_type == "inference":
        best = "vLLM (INT4+INT8KV+GQA+FlashInfer)"
        best_reason = "Decode memory-bound → INT4 max throughput"

    decisions["_overall_recommendation"] = {
        "best_framework": best,
        "reason": best_reason,
        "config": {
            "gpu_count": gpu_count,
            "gpu_memory_gb": gpu_memory_gb,
            "interconnect": interconnect,
            "model_size_b": model_size_b,
            "training_type": training_type,
            "lora_rank": lora_rank,
        },
    }

    return decisions

def main():
    parser = argparse.ArgumentParser(description="Training Strategy Decision Tool")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--gpu-memory", type=float, default=24)
    parser.add_argument("--interconnect", type=str, default="pcie",
                       choices=["nvlink", "pcie", "hccs", "none"])
    parser.add_argument("--model-size", type=float, default=7)
    parser.add_argument("--training-type", type=str, default="grpo",
                       choices=["full", "lora", "grpo", "inference", "serving"])
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    results = decide_strategy(
        args.gpu_count, args.gpu_memory, args.interconnect,
        args.model_size, args.training_type, args.lora_rank
    )

    output_path = RESULTS_DIR / "training_strategy_decision.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print("=" * 65)
    print(f"Training Strategy Decision ({args.model_size}B, {args.gpu_count}×{args.gpu_memory}GB, {args.interconnect})")
    print("=" * 65)

    for framework, options in results.items():
        if framework.startswith("_"):
            continue
        print(f"\n--- {framework} ---")
        for strategy, info in options.items():
            rec = "✓推荐" if info.get("recommended", False) else "✗不推荐"
            reason = info.get("reason", "")
            fits = ""
            if "fits" in info:
                fits = f" | fits: {'✓' if info['fits'] else '✗'}"
            print(f"  {strategy}: {rec}{fits}")
            print(f"    → {reason}")

    rec = results["_overall_recommendation"]
    print(f"\n{'='*65}")
    print(f"★ 最佳方案: {rec['best_framework']}")
    print(f"  → {rec['reason']}")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
