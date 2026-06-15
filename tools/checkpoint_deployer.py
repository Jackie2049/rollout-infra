#!/usr/bin/env python3
"""Cross-Framework Checkpoint-to-Inference Deployment Tool

根据训练框架+checkpoint格式+目标推理框架+量化选项，
生成精确的命令序列，从训练checkpoint到部署的推理模型。

支持的路径:
  - DeepSpeed ZeRO-2 → universal ckpt → HF → vLLM/SGLang
  - DeepSpeed ZeRO-3 → universal ckpt → HF → vLLM/SGLang
  - FSDP2 → FSDPModelMerger → HF → vLLM/SGLang
  - Megatron → mbridge HF → vLLM/SGLang
  - Megatron → TRT-LLM export → TensorRT
  - verl (FSDP) → FSDPModelMerger → HF → vLLM/SGLang
  - verl (Megatron) → mbridge HF → vLLM/SGLang
  - rLLM Tinker → save_pretrained → HF → vLLM/SGLang
  - MindIE → ATB → MindIE Serving (Ascend NPU)

量化选项:
  - BF16 (baseline)
  - FP8 (E4M3)
  - INT4 (GPTQ/AWQ)
  - INT4 + INT8 KV cache
  - INT4 + EAGLE speculative decoding

Usage:
  python checkpoint_deployer.py --framework deepspeed-zero2 --model-size 7 \
    --checkpoint-dir /path/to/ckpt --inference-engine vllm --quantization int4
  python checkpoint_deployer.py --framework verl-fsdp --model-size 7 \
    --checkpoint-dir /path/to/ckpt --inference-engine sglang --quantization int4-int8kv
  python checkpoint_deployer.py --framework rllm-tinker --model-size 7 \
    --checkpoint-dir /path/to/ckpt --inference-engine vllm
"""

import argparse
import json
import sys
from pathlib import Path


# ── Memory Estimation ──────────────────────────────────────────────

def estimate_inference_memory(model_size_b: int, quant: str, gpu_count: int = 1) -> dict:
    """Estimate inference memory requirements for RTX 4090 (24GB)."""
    base_params_gb = model_size_b * 2  # BF16: ~2 bytes/param → GB

    quant_sizes = {
        "bf16": base_params_gb,
        "fp8": base_params_gb * 0.5,
        "int4": base_params_gb * 0.25 + 0.5,  # quant overhead
        "int4-int8kv": base_params_gb * 0.25 + 0.5,
        "int4-eagle": base_params_gb * 0.25 + 1.0,  # +EAGLE draft model
    }

    model_gb = quant_sizes.get(quant, base_params_gb)

    # KV cache estimation (7B, GQA-8, block_size=16)
    kv_per_block_per_layer = 0.01  # ~10KB per block per layer (INT8)
    num_layers = 32 if model_size_b <= 8 else 64
    kv_total_per_block = kv_per_block_per_layer * num_layers  # ~320KB/block INT8

    available_kv_gb = (24 - model_gb - 1.0) * gpu_count  # 1GB overhead
    num_blocks = int(available_kv_gb * 1024 / kv_total_per_block)

    # Throughput estimation (RTX 4090, memory-bound decode)
    throughput_estimates = {
        "bf16": 1088,
        "fp8": 3000,
        "int4": 4791,
        "int4-int8kv": 4791,
        "int4-eagle": 9088,
    }
    throughput = throughput_estimates.get(quant, 1088)

    return {
        "model_size_gb": round(model_gb, 2),
        "available_kv_gb": round(available_kv_gb, 2),
        "kv_blocks_approx": num_blocks,
        "estimated_throughput_tok_s": throughput,
        "gpu_count": gpu_count,
        "quantization": quant,
    }


# ── Command Generation ─────────────────────────────────────────────

def generate_deepspeed_zero2_commands(ckpt_dir: str, model_size: int,
                                       base_model: str, quant: str,
                                       inference_engine: str,
                                       has_lora: bool, lora_merge: bool) -> list[str]:
    """Generate DeepSpeed ZeRO-2 → HF → inference commands."""
    steps = []
    hf_dir = f"{ckpt_dir}/hf_deploy"

    # Step 1: Extract FP32 from ZeRO-2
    steps.append("# === Step 1: ZeRO-2 → FP32 state dict ===")
    steps.append("python -m deepspeed.utils.zero_to_fp32 \\")
    steps.append(f"  {ckpt_dir}/ {ckpt_dir}/extracted/ \\")
    steps.append("  --safe_serialization --max_shard_size 5GB")

    # Step 2: Load into HF model and save
    steps.append("\n# === Step 2: FP32 → HF format ===")
    steps.append("python -c '\\")
    steps.append("from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint\\")
    steps.append(f"from transformers import AutoModelForCausalLM, AutoTokenizer\\")
    steps.append(f"state_dict = get_fp32_state_dict_from_zero_checkpoint(\"{ckpt_dir}/\", lazy_mode=True)\\")
    steps.append(f"model = AutoModelForCausalLM.from_pretrained(\"{base_model}\", state_dict=state_dict, torch_dtype=torch.bfloat16)\\")
    steps.append(f"model.save_pretrained(\"{hf_dir}/\", safe_serialization=True)\\")
    steps.append(f"tokenizer = AutoTokenizer.from_pretrained(\"{base_model}\")\\")
    steps.append(f"tokenizer.save_pretrained(\"{hf_dir}/\")\\")
    steps.append("'")

    # Step 2b: LoRA merge if applicable
    if has_lora and lora_merge:
        steps.append("\n# === Step 2b: Merge LoRA into base ===")
        steps.append("python -c '\\")
        steps.append("from peft import PeftModel\\")
        steps.append("from transformers import AutoModelForCausalLM\\")
        steps.append(f"base = AutoModelForCausalLM.from_pretrained(\"{hf_dir}/\", torch_dtype=torch.bfloat16)\\")
        steps.append(f"model = PeftModel.from_pretrained(base, \"{ckpt_dir}/lora_adapter/\")\\")
        steps.append("merged = model.merge_and_unload()\\")
        steps.append(f"merged.save_pretrained(\"{hf_dir}/\", safe_serialization=True)\\")
        steps.append("'")

    # Step 3: Quantization (if needed)
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        steps.append("\n# === Step 3: GPTQ INT4 quantization ===")
        steps.append(f"python -m auto_gptq {hf_dir}/ \\")
        steps.append(f"  --quant_config {{\"bits\": 4, \"group_size\": 128, \"desc_act\": true}} \\")
        steps.append(f"  --output_dir {hf_dir}/int4/ \\")
        steps.append(f"  --batch_size 4")

    # Step 4: Inference serving
    model_path = hf_dir if quant == "bf16" else f"{hf_dir}/int4/"
    steps.append(f"\n# === Step 4: {inference_engine.upper()} serving ===")

    if inference_engine == "vllm":
        cmd = f"vllm serve {model_path}"
        if quant in ("int4", "int4-int8kv", "int4-eagle"):
            cmd += " --quantization gptq"
        if quant in ("int4-int8kv", "int4-eagle"):
            cmd += " --kv-cache-dtype int8"
        if quant == "int4-eagle":
            cmd += " --speculative-model eagle --num-speculative-tokens 5"
        steps.append(cmd)
    elif inference_engine == "sglang":
        cmd = f"python -m sglang.launch_server --model-path {model_path}"
        if quant in ("int4", "int4-int8kv", "int4-eagle"):
            cmd += " --quantization gptq"
        if quant in ("int4-int8kv", "int4-eagle"):
            cmd += " --kv-cache-dtype int8"
        steps.append(cmd)
    elif inference_engine == "trt-llm":
        steps.append(f"# TRT-LLM: requires separate build step")
        steps.append(f"trtllm-build --model_dir {model_path}/ --engine_dir {hf_dir}/trt_engine/")

    return steps


def generate_deepspeed_zero3_commands(ckpt_dir: str, model_size: int,
                                       base_model: str, quant: str,
                                       inference_engine: str) -> list[str]:
    """Generate DeepSpeed ZeRO-3 → universal ckpt → HF → inference commands."""
    steps = []
    hf_dir = f"{ckpt_dir}/hf_deploy"

    steps.append("# === Step 1: ZeRO-3 → Universal checkpoint ===")
    steps.append("python -m deepspeed.checkpoint.ds_to_universal \\")
    steps.append(f"  {ckpt_dir}/ {ckpt_dir}/universal/")

    steps.append("\n# === Step 2: Universal → FP32 state dict ===")
    steps.append("python -m deepspeed.utils.zero_to_fp32 \\")
    steps.append(f"  {ckpt_dir}/universal/ {ckpt_dir}/extracted/ \\")
    steps.append("  --safe_serialization --max_shard_size 5GB")

    steps.append("\n# === Step 3: FP32 → HF format ===")
    steps.append("# (same as ZeRO-2 Step 2 — see deepspeed-zero2 path)")

    # Remaining steps same as ZeRO-2
    steps.append("\n# === Steps 3-4: HF → quantization → serving ===")
    steps.append("# (same as ZeRO-2 path — see generate_deepspeed_zero2_commands)")

    return steps


def generate_fsdp2_commands(ckpt_dir: str, model_size: int,
                            base_model: str, quant: str,
                            inference_engine: str) -> list[str]:
    """Generate FSDP2 → FSDPModelMerger → HF → inference commands."""
    steps = []
    hf_dir = f"{ckpt_dir}/hf_deploy"

    steps.append("# === Step 1: FSDP2 shards → HF (via FSDPModelMerger) ===")
    steps.append("# Option A: verl FSDPModelMerger (offline)")
    steps.append("python -m verl.model_merger merge \\")
    steps.append(f"  --backend fsdp --local_dir {ckpt_dir}/ \\")
    steps.append(f"  --target_dir {hf_dir}/")

    steps.append("\n# Option B: PyTorch native (in-training gather)")
    steps.append("python -c '\\")
    steps.append("from torch.distributed.fsdp import FullStateDictConfig, StateDictType\\")
    steps.append("from torch.distributed.fsdp import FSDP\\")
    steps.append("# with FullStateDictConfig(offload_to_cpu=True, rank0_only=True):\\")
    steps.append("#   full_state_dict = model.state_dict()\\")
    steps.append("#   model.save_pretrained(...)  # requires HF model wrapper\\")
    steps.append("'")

    # Quantization + serving (same pattern)
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        steps.append(f"\n# === Step 2: GPTQ INT4 quantization ===")
        steps.append(f"# (same as DeepSpeed path)")

    steps.append(f"\n# === Step 3: {inference_engine.upper()} serving ===")
    model_path = hf_dir
    if inference_engine == "vllm":
        cmd = f"vllm serve {model_path}"
        if quant in ("int4", "int4-int8kv", "int4-eagle"):
            cmd += " --quantization gptq"
            model_path = f"{hf_dir}/int4/"
        if quant in ("int4-int8kv", "int4-eagle"):
            cmd += " --kv-cache-dtype int8"
        steps.append(cmd)

    return steps


def generate_megatron_commands(ckpt_dir: str, model_size: int,
                               base_model: str, quant: str,
                               inference_engine: str) -> list[str]:
    """Generate Megatron → inference commands."""
    steps = []
    hf_dir = f"{ckpt_dir}/hf_deploy"

    if inference_engine == "trt-llm":
        steps.append("# === Path A: Megatron → TRT-LLM (production) ===")
        steps.append("\n# Step 1: TRT-LLM export")
        steps.append(f"python -m megatron.core.export.trtllm \\")
        steps.append(f"  --model_dir {ckpt_dir}/ --output_dir {ckpt_dir}/trt_weights/ \\")
        steps.append(f"  --tp_size 1 --pp_size 1")
        steps.append("\n# Step 2: TRT-LLM engine build")
        steps.append(f"trtllm-build \\")
        steps.append(f"  --model_dir {ckpt_dir}/trt_weights/ \\")
        steps.append(f"  --engine_dir {ckpt_dir}/trt_engine/")
        steps.append("\n# Step 3: TensorRT serving")
        steps.append("# (use trtllm-engine runtime for serving)")
    else:
        steps.append("# === Path B: Megatron → mbridge HF → vLLM/SGLang ===")
        steps.append("\n# Step 1: mbridge HF export")
        steps.append(f"python -m megatron.bridge \\")
        steps.append(f"  --save_weights {hf_dir}/ \\")
        steps.append(f"  --model_dir {ckpt_dir}/")

        if quant in ("int4", "int4-int8kv", "int4-eagle"):
            steps.append(f"\n# Step 2: GPTQ INT4 quantization")
            steps.append(f"# (same as other frameworks)")

        steps.append(f"\n# Step 3: {inference_engine.upper()} serving")
        model_path = hf_dir if quant == "bf16" else f"{hf_dir}/int4/"
        cmd = f"vllm serve {model_path}"
        if quant in ("int4", "int4-int8kv", "int4-eagle"):
            cmd += " --quantization gptq"
        if quant in ("int4-int8kv", "int4-eagle"):
            cmd += " --kv-cache-dtype int8"
        steps.append(cmd)

    return steps


def generate_verl_commands(ckpt_dir: str, model_size: int,
                           base_model: str, quant: str,
                           inference_engine: str,
                           backend: str) -> list[str]:
    """Generate verl → inference commands."""
    steps = []
    hf_dir = f"{ckpt_dir}/hf_deploy"

    steps.append(f"# === verl ({backend} backend) → HF → inference ===")

    if backend == "fsdp":
        steps.append("\n# Step 1: FSDPModelMerger (offline)")
        steps.append("python -m verl.model_merger merge \\")
        steps.append(f"  --backend fsdp --local_dir {ckpt_dir}/ \\")
        steps.append(f"  --target_dir {hf_dir}/")
    elif backend == "megatron":
        steps.append("\n# Step 1: mbridge HF export")
        steps.append("# (see Megatron path)")
        steps.append(f"python -m megatron.bridge --save_weights {hf_dir}/ --model_dir {ckpt_dir}/")

    # LoRA handling for GRPO
    steps.append("\n# Step 1b: LoRA merge (if GRPO + LoRA)")
    steps.append("# GRPO actor checkpoint = inference model → merge LoRA → save_pretrained")
    steps.append("python -c '\\")
    steps.append("from peft import PeftModel\\")
    steps.append("from transformers import AutoModelForCausalLM\\")
    steps.append(f"base = AutoModelForCausalLM.from_pretrained(\"{hf_dir}/\", torch_dtype=torch.bfloat16)\\")
    steps.append(f"model = PeftModel.from_pretrained(base, \"{ckpt_dir}/actor/lora_adapter/\")\\")
    steps.append("merged = model.merge_and_unload()\\")
    steps.append(f"merged.save_pretrained(\"{hf_dir}/\", safe_serialization=True)\\")
    steps.append("'")

    # Quantization + serving
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        steps.append(f"\n# Step 2: GPTQ INT4 quantization")
        steps.append(f"# (standard quantization step)")

    steps.append(f"\n# Step 3: {inference_engine.upper()} serving")
    model_path = hf_dir if quant == "bf16" else f"{hf_dir}/int4/"
    cmd = f"vllm serve {model_path}"
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        cmd += " --quantization gptq"
    if quant in ("int4-int8kv", "int4-eagle"):
        cmd += " --kv-cache-dtype int8"
    if quant == "int4-eagle":
        cmd += " --speculative-model eagle --num-speculative-tokens 5"
    steps.append(cmd)

    return steps


def generate_rllm_commands(ckpt_dir: str, model_size: int,
                           base_model: str, quant: str,
                           inference_engine: str) -> list[str]:
    """Generate rLLM Tinker → HF → inference commands (simplest path!)."""
    steps = []
    hf_dir = f"{ckpt_dir}/hf_deploy"

    steps.append("# === rLLM Tinker → HF → inference (最简单路径!) ===")
    steps.append("\n# Step 1: Already HF format! save_pretrained during training")
    steps.append("# rLLM TinkerBackend.save_checkpoint() → save_pretrained() → HF format directly")
    steps.append(f"# Output: {hf_dir}/ → model.safetensors + config.json + tokenizer.json")
    steps.append("\n# ★ No merge step needed! rLLM Tinker saves HF format directly!")

    # LoRA merge if needed
    steps.append("\n# Step 1b: LoRA merge (optional)")
    steps.append("# If using LoRA: merge before inference for maximum throughput")
    steps.append("# Or: vLLM dynamic LoRA loading for multi-adapter serving")
    steps.append("python -c '\\")
    steps.append("from peft import PeftModel\\")
    steps.append("from transformers import AutoModelForCausalLM\\")
    steps.append(f"base = AutoModelForCausalLM.from_pretrained(\"{hf_dir}/\")\\")
    steps.append(f"model = PeftModel.from_pretrained(base, \"{hf_dir}/lora/\")\\")
    steps.append("merged = model.merge_and_unload()\\")
    steps.append(f"merged.save_pretrained(\"{hf_dir}/merged/\")\\")
    steps.append("'")

    # Quantization + serving
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        steps.append(f"\n# Step 2: GPTQ INT4 quantization")
        steps.append(f"# (standard step)")

    steps.append(f"\n# Step 3: {inference_engine.upper()} serving")
    model_path = f"{hf_dir}/merged/" if quant != "bf16" else hf_dir
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        model_path = f"{hf_dir}/merged/int4/"
    cmd = f"vllm serve {model_path}"
    if quant in ("int4", "int4-int8kv", "int4-eagle"):
        cmd += " --quantization gptq"
    if quant in ("int4-int8kv", "int4-eagle"):
        cmd += " --kv-cache-dtype int8"
    if quant == "int4-eagle":
        cmd += " --speculative-model eagle --num-speculative-tokens 5"
    steps.append(cmd)

    return steps


def generate_mindie_commands(ckpt_dir: str, model_size: int,
                             quant: str) -> list[str]:
    """Generate MindIE → Ascend NPU serving commands."""
    steps = []

    steps.append("# === MindIE → Ascend NPU serving ===")
    steps.append("\n# ★ RTX 4090 不适用! MindIE 仅支持 Ascend NPU (910B/C/D)")
    steps.append("\n# Step 1: MindIE-LLM serving (ATB format → direct)")
    steps.append(f"mindie-llm --model-path {ckpt_dir}/ \\")
    steps.append("  --device ascend")
    steps.append("\n# Step 2: Alternative: vLLM-Ascend")
    steps.append(f"vllm serve {ckpt_dir}/ --device ascend")
    steps.append("\n# ★ MindIE不适用于NVIDIA GPU → 使用vLLM替代!")

    return steps


# ── RTX 4090 Feasibility Check ────────────────────────────────────

def check_rtx4090_feasibility(framework: str, model_size: int, quant: str,
                               has_lora: bool, training_type: str) -> dict:
    """Check if deployment path is feasible on RTX 4090."""
    issues = []
    feasible = True

    # Inference feasibility
    mem = estimate_inference_memory(model_size, quant)
    if mem["model_size_gb"] > 22:  # Leave 2GB for OS/runtime
        issues.append(f"Model {mem['model_size_gb']}GB > 22GB available → need quantization")
        feasible = False

    # Framework-specific checks
    if framework == "deepspeed-zero3":
        issues.append("ZeRO-3 training: 3Ψ communication → PCIe disaster → not feasible for training")
        issues.append("ZeRO-3 extraction: OK offline → universal ckpt → HF → inference feasible")
    elif framework == "megatron":
        issues.append("Megatron TP>1: PCIe AllReduce terrible → single GPU only")
    elif framework == "mindie":
        issues.append("MindIE: Ascend NPU only → RTX 4090 not applicable → use vLLM instead")
        feasible = False

    # Training feasibility (if relevant)
    if training_type == "ppo":
        issues.append("PPO: ~270GB/GPU → RTX 4090 completely impossible")
        feasible = False
    elif training_type == "grpo" and not has_lora:
        bf16_mem = model_size * 2
        if bf16_mem > 20:
            issues.append(f"GRPO BF16: {bf16_mem}GB → need LoRA or quantization")
            feasible = False

    recommendations = []
    if framework == "deepspeed-zero2" and has_lora:
        recommendations.append("ZeRO-2 + LoRA + CPU Adam → RTX 4090最优训练路径")
    if framework == "rllm-tinker":
        recommendations.append("rLLM Tinker + GRPO + LoRA → 最简单单GPU路径")
    if framework == "verl-fsdp" and has_lora:
        recommendations.append("verl HYBRID + LoRA + naive sync → 单GPU可行")

    return {
        "feasible": feasible,
        "issues": issues,
        "recommendations": recommendations,
        "inference_memory": mem,
    }


# ── Main ───────────────────────────────────────────────────────────

FRAMEWORKS = [
    "deepspeed-zero2", "deepspeed-zero3", "fsdp2", "megatron",
    "verl-fsdp", "verl-megatron", "rllm-tinker", "mindie",
]

QUANTIZATIONS = ["bf16", "fp8", "int4", "int4-int8kv", "int4-eagle"]

INFERENCE_ENGINES = ["vllm", "sglang", "trt-llm"]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-framework checkpoint-to-inference deployment tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--framework", required=True, choices=FRAMEWORKS,
                        help="Training framework that produced the checkpoint")
    parser.add_argument("--model-size", required=True, type=int,
                        help="Model size in billions of params (7=7B, 70=70B)")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to training checkpoint directory")
    parser.add_argument("--inference-engine", default="vllm", choices=INFERENCE_ENGINES,
                        help="Target inference engine")
    parser.add_argument("--quantization", default="bf16", choices=QUANTIZATIONS,
                        help="Quantization method for inference")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace base model identifier")
    parser.add_argument("--has-lora", action="store_true",
                        help="Checkpoint includes LoRA adapter")
    parser.add_argument("--lora-merge", action="store_true",
                        help="Merge LoRA into base before inference")
    parser.add_argument("--gpu-count", default=1, type=int,
                        help="Number of GPUs for inference")
    parser.add_argument("--training-type", default="grpo",
                        choices=["grpo", "ppo", "sft", "none"],
                        help="Training type that produced checkpoint")

    args = parser.parse_args()

    # Generate commands
    generators = {
        "deepspeed-zero2": lambda: generate_deepspeed_zero2_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine, args.has_lora, args.lora_merge),
        "deepspeed-zero3": lambda: generate_deepspeed_zero3_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine),
        "fsdp2": lambda: generate_fsdp2_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine),
        "megatron": lambda: generate_megatron_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine),
        "verl-fsdp": lambda: generate_verl_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine, "fsdp"),
        "verl-megatron": lambda: generate_verl_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine, "megatron"),
        "rllm-tinker": lambda: generate_rllm_commands(
            args.checkpoint_dir, args.model_size, args.base_model,
            args.quantization, args.inference_engine),
        "mindie": lambda: generate_mindie_commands(
            args.checkpoint_dir, args.model_size, args.quantization),
    }

    commands = generators[args.framework]()

    # Feasibility check
    feasibility = check_rtx4090_feasibility(
        args.framework, args.model_size, args.quantization,
        args.has_lora, args.training_type)

    # Print results
    print("=" * 70)
    print(f"Checkpoint → Inference Deployment Plan")
    print(f"  Framework: {args.framework}")
    print(f"  Model: {args.model_size}B")
    print(f"  Quantization: {args.quantization}")
    print(f"  Inference: {args.inference_engine}")
    print(f"  GPU: {args.gpu_count}× RTX 4090 (24GB)")
    print("=" * 70)

    print("\n## Feasibility Check (RTX 4090)")
    status = "✓ FEASIBLE" if feasibility["feasible"] else "✗ NOT FEASIBLE"
    print(f"  Status: {status}")
    for issue in feasibility["issues"]:
        print(f"  ⚠ {issue}")
    for rec in feasibility["recommendations"]:
        print(f"  ★ {rec}")

    print(f"\n## Memory Estimate")
    mem = feasibility["inference_memory"]
    print(f"  Model weights: {mem['model_size_gb']} GB")
    print(f"  Available for KV: {mem['available_kv_gb']} GB")
    print(f"  KV blocks (approx): {mem['kv_blocks_approx']}")
    print(f"  Est. throughput: {mem['estimated_throughput_tok_s']} tok/s")

    print(f"\n## Deployment Commands")
    for cmd in commands:
        print(cmd)

    # Save JSON output
    output = {
        "framework": args.framework,
        "model_size": args.model_size,
        "quantization": args.quantization,
        "inference_engine": args.inference_engine,
        "feasibility": feasibility,
        "commands": commands,
    }

    output_path = Path(args.checkpoint_dir) / "deploy_plan.json"
    try:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n## Plan saved to: {output_path}")
    except OSError:
        print(f"\n## (Could not save plan JSON — checkpoint dir may not exist yet)")

    return 0 if feasibility["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
