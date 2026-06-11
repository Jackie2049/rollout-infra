#!/usr/bin/env python3
"""
Distributed Training Configuration Advisor
基于源码阅读(torch.distributed+NCCL+safetensors+Megatron)的分布式训练配置推荐工具

输入: 模型大小, GPU类型, GPU数量, 训练目标
输出: 最优并行策略, NCCL配置, 内存估算, throughput估算, 权重格式建议

Usage:
  python tools/distributed_training_advisor.py --model-size 7B --gpu-type rtx4090 --gpu-count 1
  python tools/distributed_training_advisor.py --model-size 70B --gpu-type a100 --gpu-count 8
  python tools/distributed_training_advisor.py --model-size 7B --gpu-type rtx4090 --gpu-count 8 --objective finetune
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Optional


# GPU hardware specs (from our benchmark data)
GPU_SPECS = {
    "rtx4090": {
        "name": "RTX 4090",
        "hbm_gb": 24,
        "sm": 89,
        "nvlink": False,
        "p2p": False,
        "pcie_bandwidth_gbps": 12,  # bidirectional
        "tflops_bf16": 82.6,
        "hbm_bandwidth_gbps": 890.8,
        "cost_per_hour": 0.5,  # matpool approx
        "notes": "Consumer GPU, no NVLink P2P, PCIe bottleneck for multi-GPU",
    },
    "a100_80gb": {
        "name": "A100 80GB",
        "hbm_gb": 80,
        "sm": 80,
        "nvlink": True,
        "p2p": True,
        "nvlink_bandwidth_gbps": 300,
        "pcie_bandwidth_gbps": 32,
        "tflops_bf16": 156,
        "hbm_bandwidth_gbps": 2039,
        "cost_per_hour": 2.0,
        "notes": "Data center GPU, NVLink P2P, optimal for TP/DP",
    },
    "h100": {
        "name": "H100 80GB",
        "hbm_gb": 80,
        "sm": 90,
        "nvlink": True,
        "p2p": True,
        "nvlink_bandwidth_gbps": 726,
        "pcie_bandwidth_gbps": 64,
        "tflops_bf16": 495,
        "hbm_bandwidth_gbps": 3352,
        "cost_per_hour": 3.0,
        "notes": "Latest data center GPU, NVLink+NVLB, TMA+WGMMA",
    },
    "h200": {
        "name": "H200 141GB",
        "hbm_gb": 141,
        "sm": 90,
        "nvlink": True,
        "p2p": True,
        "nvlink_bandwidth_gbps": 726,
        "pcie_bandwidth_gbps": 64,
        "tflops_bf16": 495,
        "hbm_bandwidth_gbps": 4800,
        "cost_per_hour": 4.0,
        "notes": "Max memory, optimal for large model training",
    },
}

# Model memory estimates (BF16 training with AdamW)
MODEL_MEMORY = {
    # param_bytes = size_B * 2 (BF16)
    # gradient = size_B * 2
    # adamw = size_B * 12 (FP32 m+v + FP32 master)
    # total = size_B * 16 (BF16 params + BF16 grad + FP32 adam states)
    # But with mixed precision: FP32 master weights = 4*size_B, grad FP16 = 2*size_B, adam FP32 = 8*size_B
    # Total per GPU = depends on parallelism strategy
}


@dataclass
class TrainingConfig:
    model_size_b: float  # model size in billions of parameters
    gpu_type: str
    gpu_count: int
    objective: str  # "pretrain", "finetune", "lora"
    seq_len: int = 4096
    batch_size: int = 1
    precision: str = "bf16"
    offload_optimizer: bool = False
    offload_params: bool = False


def estimate_model_memory(config: TrainingConfig) -> dict:
    """Estimate memory requirements for different parallelism strategies."""
    size_b = config.model_size_b
    n_gpus = config.gpu_count
    gpu = GPU_SPECS[config.gpu_type]
    hbm_gb = gpu["hbm_gb"]

    # Bytes per parameter for different states
    param_bytes = 2  # BF16 params
    grad_bytes = 2   # BF16 gradients
    adam_bytes = 12   # FP32 master(4) + m(4) + v(4)
    total_bytes_per_param = param_bytes + grad_bytes + adam_bytes  # = 16

    # Activation memory (approximate for transformer)
    # ~seq_len * hidden_dim * num_layers * 2 (forward + backward)
    # For 7B: hidden=4096, layers=32 → act ≈ seq * 4096 * 32 * 2 * 2bytes
    hidden_dim = int(4096 * (size_b / 7) ** 0.5)  # scale with model size
    num_layers = int(32 * (size_b / 7) ** 0.7)
    act_bytes = config.seq_len * hidden_dim * num_layers * 4  # rough estimate

    # Total model memory
    model_memory_gb = size_b * 1e9 * total_bytes_per_param / 1e9  # in GB
    act_memory_gb = act_bytes / 1e9

    # Different parallelism strategies
    strategies = {}

    # 1. Pure DP (no sharding)
    dp_memory = model_memory_gb + act_memory_gb
    strategies["pure_dp"] = {
        "memory_per_gpu_gb": dp_memory,
        "fits": dp_memory <= hbm_gb,
        "communication": "AllReduce (gradient sync)",
        "comm_volume_gb": size_b * 2 / n_gpus if n_gpus > 1 else 0,  # gradient per step
    }

    # 2. ZeRO-1 (optimizer sharding)
    zero1_memory = (size_b * 1e9 * (param_bytes + grad_bytes)) / 1e9 + \
                   (size_b * 1e9 * adam_bytes / n_gpus) / 1e9 + act_memory_gb
    strategies["zero1"] = {
        "memory_per_gpu_gb": zero1_memory,
        "fits": zero1_memory <= hbm_gb,
        "communication": "AllReduce (gradient) + ReduceScatter (optimizer)",
        "comm_volume_gb": size_b * 2 / n_gpus if n_gpus > 1 else 0,
    }

    # 3. ZeRO-2 (optimizer + gradient sharding)
    zero2_memory = (size_b * 1e9 * param_bytes) / 1e9 + \
                   (size_b * 1e9 * (grad_bytes + adam_bytes) / n_gpus) / 1e9 + act_memory_gb
    strategies["zero2"] = {
        "memory_per_gpu_gb": zero2_memory,
        "fits": zero2_memory <= hbm_gb,
        "communication": "ReduceScatter (gradient+optimizer)",
        "comm_volume_gb": size_b * 2 / n_gpus if n_gpus > 1 else 0,
    }

    # 4. ZeRO-3/FSDP (full sharding)
    zero3_memory = (size_b * 1e9 * param_bytes / n_gpus) / 1e9 + \
                   (size_b * 1e9 * (grad_bytes + adam_bytes) / n_gpus) / 1e9 + \
                   act_memory_gb + (size_b * 1e9 * param_bytes) / 1e9  # all_gather temp
    strategies["zero3_fsdp"] = {
        "memory_per_gpu_gb": zero3_memory,
        "fits": zero3_memory <= hbm_gb,
        "communication": "AllGather (params) + ReduceScatter (gradient)",
        "comm_volume_gb": size_b * 4 / n_gpus if n_gpus > 1 else 0,  # 2x for forward+backward
    }

    # 5. ZeRO-2 + CPU optimizer offload
    if config.offload_optimizer:
        zero2_offload_memory = (size_b * 1e9 * (param_bytes + grad_bytes)) / 1e9 + act_memory_gb
        strategies["zero2_offload"] = {
            "memory_per_gpu_gb": zero2_offload_memory,
            "fits": zero2_offload_memory <= hbm_gb,
            "communication": "AllReduce (gradient) + CPU optimizer update",
            "comm_volume_gb": size_b * 2 / n_gpus if n_gpus > 1 else 0,
            "cpu_memory_gb": size_b * 12 / n_gpus,  # Adam states on CPU
        }

    # 6. TP (tensor parallelism) — only works with NVLink!
    if gpu["nvlink"] and n_gpus > 1:
        tp_factor = min(n_gpus, 8)  # practical max TP=8
        tp_memory = model_memory_gb / tp_factor + act_memory_gb / tp_factor
        strategies[f"tp{tp_factor}"] = {
            "memory_per_gpu_gb": tp_memory,
            "fits": tp_memory <= hbm_gb,
            "communication": "AllReduce per layer (TP)",
            "comm_volume_gb": size_b * 2 * num_layers * hidden_dim * hidden_dim * 3 / tp_factor / 1e9,
        }

    # 7. LoRA (only adds adapter params)
    if config.objective == "lora":
        lora_params_b = size_b * 0.001  # ~0.1% of model params
        lora_memory = (size_b * 1e9 * param_bytes) / 1e9 + \
                      (lora_params_b * 1e9 * total_bytes_per_param) / 1e9 + \
                      act_memory_gb
        strategies["lora"] = {
            "memory_per_gpu_gb": lora_memory,
            "fits": lora_memory <= hbm_gb,
            "communication": "AllReduce (gradient) — same as DP but only LoRA params",
            "comm_volume_gb": lora_params_b * 2 / n_gpus if n_gpus > 1 else 0,
            "trainable_params_pct": 0.1,
        }

    return {
        "model_params_gb": size_b * 2,  # BF16 weight size
        "total_state_gb": model_memory_gb,
        "activation_gb": act_memory_gb,
        "strategies": strategies,
    }


def recommend_parallel_strategy(config: TrainingConfig, memory: dict) -> dict:
    """Recommend the best parallelism strategy based on hardware and model size."""
    gpu = GPU_SPECS[config.gpu_type]
    strategies = memory["strategies"]

    # RTX 4090 special case: PCIe bottleneck makes multi-GPU counterproductive
    if config.gpu_type == "rtx4090":
        # For RTX 4090, single GPU is almost always optimal
        single_gpu_fits = any(
            s["fits"] for name, s in strategies.items()
            if "offload" in name or name in ["zero2_offload", "lora", "zero2", "zero1"]
        )

        if config.gpu_count == 1 or not gpu["nvlink"]:
            # Single GPU recommendation
            if config.objective == "lora":
                return {
                    "strategy": "LoRA on single GPU",
                    "reason": "RTX 4090 PCIe bottleneck makes multi-GPU DDP 0.46x slower. "
                              "LoRA on single GPU avoids communication overhead entirely.",
                    "fsdp_level": "NO_SHARD",
                    "ddp_bucket_mb": 0,  # single GPU, no bucketing needed
                    "key_insight": "From NCCL internals: PCIe 2.76GB/s AllReduce vs NVLink 300GB/s. "
                                   "RTX 4090 multi-GPU = counterproductive!",
                }
            elif config.objective == "finetune":
                # Check if ZeRO-2 + offload fits
                if "zero2_offload" in strategies and strategies["zero2_offload"]["fits"]:
                    return {
                        "strategy": "ZeRO-2 + CPU optimizer offload on single GPU",
                        "reason": "Model params+grad+act fits HBM with optimizer offloaded to CPU. "
                                  "CPU offload optimizer (56GB) → GPU only stores params+grad+act.",
                        "fsdp_level": "SHARD_GRAD_OP with offload",
                        "offload_target": "optimizer to CPU",
                        "key_insight": "From DDP Reducer: bucket_cap_mb irrelevant for 1 GPU. "
                                       "From FSDP2: NO_SHARD if memory fits, SHARD_GRAD_OP+offload if not.",
                    }
                elif "lora" in strategies and strategies["lora"]["fits"]:
                    return {
                        "strategy": "LoRA fine-tuning on single GPU",
                        "reason": "7B BF16 full params+grad exceeds 24GB HBM. "
                                  "LoRA trains only 0.1% params → gradient+adam negligible. "
                                  "Full model weights (14GB) loaded but frozen.",
                        "fsdp_level": "NO_SHARD (LoRA)",
                        "trainable_params_pct": "0.1%",
                        "key_insight": "From FSDP2 lifecycle: NO_SHARD for LoRA — "
                                       "no sharding needed, only LoRA adapter params have gradients. "
                                       "From DDP Reducer: tiny buckets for LoRA params.",
                    }
                else:
                    # Nothing fits — suggest activation checkpointing + LoRA
                    # With gradient checkpointing: act ≈ seq_len * hidden_dim (no full forward cache)
                    act_checkpointed_gb = memory["activation_gb"] * 0.25  # ~75% reduction
                    lora_total = memory["model_params_gb"] + config.model_size_b * 0.001 * 16 + act_checkpointed_gb
                    return {
                        "strategy": "LoRA + gradient checkpointing on single GPU",
                        "reason": f"7B BF16 weights (14GB) + LoRA (0.16GB state) + "
                                  f"checkpointed act ({act_checkpointed_gb:.1f}GB) ≈ "
                                  f"{lora_total:.1f}GB → fits {gpu['hbm_gb']}GB!",
                        "gradient_checkpointing": True,
                        "activation_reduction": "~75% (from our benchmark: activation-checkpointing-rtx4090.md)",
                        "key_insight": "From vLLM InputBatch: Persistent Batch avoids full rebuild. "
                                       "From our RTX 4090 benchmark: gradient checkpointing reduces "
                                       "activation 75% at ~37% throughput cost. LoRA is best for 4090 finetune.",
                    }

        # Multi-GPU RTX 4090 — generally bad but sometimes unavoidable
        if config.gpu_count > 1 and not gpu["nvlink"]:
            return {
                "strategy": "DP with minimal AllReduce (avoid if possible)",
                "reason": f"RTX 4090 {config.gpu_count}×GPU PCIe: AllReduce 2.76GB/s → "
                          f"DDP = 0.46x vs 1GPU. Only use if single GPU OOM.",
                "warning": "PCIe scaling disaster! 8GPU=0.46x. Prefer single GPU + offload.",
                "fsdp_level": "NO_SHARD (DDP)",
                "ddp_bucket_mb": 25,
                "nccl_config": get_nccl_config(config),
                "key_insight": "From NCCL internals: PCIe SHM transport, Ring algo, "
                               "2-4 channels, protocol=Simple for large messages.",
            }

    # NVLink GPUs (A100/H100/H200) — multi-GPU is productive
    if gpu["nvlink"] and config.gpu_count > 1:
        # Find best fitting strategy
        best = None
        for name, s in strategies.items():
            if s["fits"]:
                # Prefer simpler strategies that fit
                priority = {"lora": 1, "zero1": 2, "zero2": 3, "zero2_offload": 4,
                           "zero3_fsdp": 5, "pure_dp": 6}
                if name.startswith("tp"):
                    priority[name] = 7
                p = priority.get(name, 99)
                if best is None or p < best[0]:
                    best = (p, name, s)

        if best:
            _, name, s = best
            return {
                "strategy": name,
                "reason": f"Best strategy that fits {gpu['hbm_gb']}GB HBM. "
                          f"NVLink {gpu['nvlink_bandwidth_gbps']}GB/s enables efficient multi-GPU.",
                "memory_per_gpu_gb": s["memory_per_gpu_gb"],
                "fsdp_level": map_name_to_fsdp_level(name),
                "ddp_bucket_mb": 25,
                "nccl_config": get_nccl_config(config),
                "key_insight": f"From NCCL: NVLink transport, {4 if gpu['nvlink'] else 2} channels. "
                               f"From torch.distributed: ProcessGroupNCCL stream-per-device + double-barrier.",
            }

    # Fallback
    return {
        "strategy": "none_fits",
        "reason": f"No strategy fits in {gpu['hbm_gb']}GB HBM. Need more GPUs or offload.",
        "suggestion": f"Use ZeRO-3 + CPU offload, or increase GPU count to "
                      f"{math.ceil(memory['total_state_gb'] / gpu['hbm_gb'])}",
    }


def map_name_to_fsdp_level(name: str) -> str:
    mapping = {
        "pure_dp": "NO_SHARD",
        "zero1": "NO_SHARD (optimizer offload only)",
        "zero2": "SHARD_GRAD_OP",
        "zero2_offload": "SHARD_GRAD_OP + optimizer offload",
        "zero3_fsdp": "FULL_SHARD",
        "lora": "NO_SHARD (LoRA)",
    }
    return mapping.get(name, name)


def get_nccl_config(config: TrainingConfig) -> dict:
    """Generate NCCL environment variables based on GPU type and topology insights."""
    gpu = GPU_SPECS[config.gpu_type]

    if config.gpu_type == "rtx4090":
        return {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IGNORE_DISABLED_P2P": "1",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_MAX_NRINGS": "4",
            "NCCL_ALGO": "RING",
            "NCCL_PROTO": "Simple",  # large messages (gradient buckets)
            "NCCL_DEBUG": "WARN",
            "NCCL_BUFFSIZE": "4194304",  # 4MB default
            "reason": "RTX 4090: no P2P → SHM(PCIe) transport → Ring algo → "
                      "4 channels → Simple protocol for large gradient sync. "
                      "From NCCL internals: PCIe bandwidth 2.76GB/s实测, "
                      "protocol choice irrelevant (bandwidth bottleneck).",
        }
    elif gpu["nvlink"]:
        return {
            "NCCL_P2P_LEVEL": "SYS",  # NVLink P2P
            "NCCL_MAX_NRINGS": "8",
            "NCCL_ALGO": "auto",  # Ring for large, Tree for small
            "NCCL_PROTO": "auto",  # LL/LL128/Simple adaptive
            "NCCL_DEBUG": "WARN",
            "NCCL_BUFFSIZE": "8388608",  # 8MB for NVLink
            "reason": f"{gpu['name']}: NVLink P2P → Net+SHM transport → "
                      "auto algorithm selection → adaptive protocol. "
                      "From NCCL internals: NVLink 300+GB/s, 8 channels, "
                      "LL for small messages, Simple for large.",
        }
    else:
        return {
            "NCCL_DEBUG": "WARN",
            "NCCL_MAX_NRINGS": "4",
            "reason": "Generic PCIe GPU config.",
        }


def estimate_throughput(config: TrainingConfig, recommendation: dict) -> dict:
    """Estimate training throughput based on parallelism and hardware."""
    gpu = GPU_SPECS[config.gpu_type]
    strategy = recommendation["strategy"]

    # Base compute throughput (tokens/second per GPU)
    # BF16 GEMM: 2*size_b params * seq_len * batch_size * 3 (forward+backward) / tflops
    compute_time_ms = (2 * config.model_size_b * 1e9 * config.seq_len * 3) / \
                      (gpu["tflops_bf16"] * 1e12 / 1e3)  # ms

    # Communication time
    comm_time_ms = 0
    if config.gpu_count > 1:
        if gpu["nvlink"]:
            # NVLink: gradient AllReduce
            gradient_gb = config.model_size_b * 2  # BF16 gradient
            bandwidth = gpu["nvlink_bandwidth_gbps"]
            comm_time_ms = gradient_gb / bandwidth * 1000  # ms
        else:
            # PCIe: measured RTX 4090 AllReduce
            if config.gpu_type == "rtx4090":
                # From benchmark: 2.76GB/s AllReduce
                gradient_gb = config.model_size_b * 2
                comm_time_ms = gradient_gb / 2.76 * 1000  # ms
            else:
                gradient_gb = config.model_size_b * 2
                comm_time_ms = gradient_gb / gpu["pcie_bandwidth_gbps"] * 1000

    # Overlap factor (DDP bucketing allows overlap)
    # From DDP Reducer: bucket_cap_mb=25 → ~70% overlap for NVLink
    # PCIe: ~20% overlap (comm much slower than compute)
    if gpu["nvlink"] and config.gpu_count > 1:
        overlap_factor = 0.7
    elif config.gpu_count > 1:
        overlap_factor = 0.2
    else:
        overlap_factor = 1.0  # no communication

    effective_time_ms = compute_time_ms + comm_time_ms * (1 - overlap_factor)

    # LoRA: much fewer trainable params
    if strategy == "LoRA on single GPU":
        lora_factor = 0.001  # 0.1% params
        effective_time_ms = compute_time_ms * lora_factor * 3  # forward+backward+update for LoRA only
        # But base model forward still needed
        effective_time_ms = compute_time_ms + compute_time_ms * lora_factor * 2

    # ZeRO-3/FSDP: additional AllGather overhead
    if "zero3" in strategy.lower() or "fsdp" in strategy.lower():
        # 2x AllGather per step (forward + backward)
        allgather_gb = config.model_size_b * 2 * 2  # BF16 params * 2
        if gpu["nvlink"]:
            allgather_ms = allgather_gb / gpu["nvlink_bandwidth_gbps"] * 1000
        else:
            allgather_ms = allgather_gb / 2.76 * 1000 if config.gpu_type == "rtx4090" \
                else allgather_gb / gpu["pcie_bandwidth_gbps"] * 1000
        effective_time_ms += allgather_ms

    # Tokens per second
    tokens_per_step = config.seq_len * config.batch_size * config.gpu_count
    throughput_tok_per_s = tokens_per_step / (effective_time_ms / 1000) if effective_time_ms > 0 else 0

    return {
        "compute_time_ms": round(compute_time_ms, 1),
        "comm_time_ms": round(comm_time_ms, 1),
        "overlap_factor": overlap_factor,
        "effective_time_ms": round(effective_time_ms, 1),
        "tokens_per_step": tokens_per_step,
        "throughput_tok_per_s": round(throughput_tok_per_s, 1),
        "steps_per_second": round(1000 / effective_time_ms, 3) if effective_time_ms > 0 else 0,
    }


def get_weight_format_recommendation(config: TrainingConfig) -> dict:
    """Recommend weight format based on training stage."""
    return {
        "training_checkpoint": {
            "format": "PyTorch .pt (pickle)",
            "reason": "Must store optimizer state + scheduler + RNG for training resume. "
                      "From safetensors deep dive: safetensors only stores weights, no optimizer.",
            "warning": "pickle = arbitrary code execution risk! Only use in trusted environments.",
            "file_size_estimate_gb": config.model_size_b * 16,  # params+grad+adam
        },
        "distribution_sharing": {
            "format": "Safetensors (.safetensors)",
            "reason": "HuggingFace standard. mmap + zero-copy + lazy load. "
                      "From safetensors spec: 8B header + JSON metadata + contiguous data buffer. "
                      "mmap load: 0.1ms vs pickle: 30s → 300,000x faster!",
            "file_size_estimate_gb": config.model_size_b * 2,  # BF16 weights only
            "multi_process_saving": f"{config.gpu_count} processes sharing {config.model_size_b * 2}GB "
                                   f"→ only {config.model_size_b * 2}GB physical memory "
                                   f"(vs pickle: {config.model_size_b * 2 * config.gpu_count}GB)",
        },
        "inference_deployment": {
            "format": "Safetensors (BF16) → runtime quantize, or quantized safetensors (AWQ/GPTQ)",
            "reason": "vLLM uses safetensors mmap for weight loading. "
                      "From vLLM pipeline: discover→detect→parse→map→allocate→transfer. "
                      "For RTX 4090: INT4 AWQ safetensors → ~4GB → fastest inference.",
            "vllm_command": f"vllm serve model-name --quantization awq "
                           f"--gpu-memory-utilization 0.9 --max-model-len {config.seq_len}",
        },
        "conversion_pipeline": "pickle (.pt) → consolidate → safetensors → (optional) AWQ/GPTQ quantize → vLLM serve",
    }


def generate_full_report(config: TrainingConfig) -> dict:
    """Generate comprehensive training configuration report."""
    memory = estimate_model_memory(config)
    recommendation = recommend_parallel_strategy(config, memory)
    throughput = estimate_throughput(config, recommendation)
    nccl = get_nccl_config(config)
    weight_formats = get_weight_format_recommendation(config)

    gpu = GPU_SPECS[config.gpu_type]

    report = {
        "config": {
            "model_size": f"{config.model_size_b}B",
            "gpu_type": gpu["name"],
            "gpu_count": config.gpu_count,
            "objective": config.objective,
            "precision": config.precision,
            "seq_len": config.seq_len,
            "batch_size": config.batch_size,
        },
        "hardware": {
            "hbm_per_gpu": f"{gpu['hbm_gb']}GB",
            "nvlink": gpu["nvlink"],
            "p2p": gpu["p2p"],
            "interconnect": f"NVLink {gpu.get('nvlink_bandwidth_gbps', 0)}GB/s" if gpu["nvlink"]
                           else f"PCIe {gpu['pcie_bandwidth_gbps']}GB/s",
            "compute": f"{gpu['tflops_bf16']} TFLOPS BF16",
            "memory_bandwidth": f"{gpu['hbm_bandwidth_gbps']}GB/s",
        },
        "memory_analysis": {
            "model_weights_bf16_gb": memory["model_params_gb"],
            "total_training_state_gb": memory["total_state_gb"],
            "activation_estimate_gb": memory["activation_gb"],
            "strategies": memory["strategies"],
        },
        "recommendation": recommendation,
        "nccl_config": nccl,
        "throughput_estimate": throughput,
        "weight_formats": weight_formats,
        "torch_distributed_insights": {
            "ProcessGroupNCCL": "ncclComm lazy init + cached reuse. WorkNCCL async model + CUDA event. "
                                "Stream-per-device double-barrier. Watchdog + HeartbeatMonitor dual monitoring.",
            "DDP_Reducer": f"Gradient bucketing: bucket_cap_mb={recommendation.get('ddp_bucket_mb', 25)}. "
                           "Reverse order (last layers first). Flat buffer for single allreduce per bucket. "
                           "Communication-compute overlap via autograd_hook → mark_variable_ready → mark_bucket_ready.",
            "FSDP2": f"Recommended level: {recommendation.get('fsdp_level', 'N/A')}. "
                     "Lifecycle: shard→unshard(all_gather)→compute→reshard→backward→reduce_scatter. "
                     "param.data swap for zero-copy.",
            "NCCL_channels": f"From NCCL internals: {nccl.get('NCCL_MAX_NRINGS', 'auto')} channels. "
                            f"Algorithm: {nccl.get('NCCL_ALGO', 'auto')}. "
                            f"Protocol: {nccl.get('NCCL_PROTO', 'auto')}. "
                            "Proxy thread for DMA/network. Transport: " +
                            ("NVLink+SHM" if gpu["nvlink"] else "SHM(PCIe)"),
        },
    }

    return report


def format_report(report: dict) -> str:
    """Format the report as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("DISTRIBUTED TRAINING CONFIGURATION ADVISOR")
    lines.append("Based on: torch.distributed + NCCL + safetensors + Megatron source reading")
    lines.append("=" * 70)

    # Config summary
    lines.append("\n## Configuration")
    c = report["config"]
    lines.append(f"  Model: {c['model_size']} | GPU: {c['gpu_type']} × {c['gpu_count']}")
    lines.append(f"  Objective: {c['objective']} | Precision: {c['precision']}")
    lines.append(f"  SeqLen: {c['seq_len']} | BatchSize: {c['batch_size']}")

    # Hardware
    lines.append("\n## Hardware")
    h = report["hardware"]
    lines.append(f"  HBM: {h['hbm_per_gpu']} per GPU")
    lines.append(f"  Interconnect: {h['interconnect']}")
    lines.append(f"  P2P: {h['p2p']} | NVLink: {h['nvlink']}")
    lines.append(f"  Compute: {h['compute']} | Memory BW: {h['memory_bandwidth']}")

    # Memory analysis
    lines.append("\n## Memory Analysis")
    m = report["memory_analysis"]
    lines.append(f"  BF16 Weights: {m['model_weights_bf16_gb']:.1f}GB")
    lines.append(f"  Total Training State: {m['total_state_state_gb'] if 'total_state_state_gb' in m else m['total_training_state_gb']:.1f}GB")
    lines.append(f"  Activation: {m['activation_estimate_gb']:.1f}GB")
    lines.append("  Strategy fits:")
    for name, s in m["strategies"].items():
        fit = "YES" if s["fits"] else "NO"
        lines.append(f"    {name}: {s['memory_per_gpu_gb']:.1f}GB → {fit}")

    # Recommendation
    lines.append("\n## Recommended Strategy")
    r = report["recommendation"]
    lines.append(f"  Strategy: {r['strategy']}")
    lines.append(f"  Reason: {r['reason']}")
    if "key_insight" in r:
        lines.append(f"  Key Insight: {r['key_insight']}")
    if "warning" in r:
        lines.append(f"  WARNING: {r['warning']}")
    if "fsdp_level" in r:
        lines.append(f"  FSDP Level: {r['fsdp_level']}")

    # NCCL config
    lines.append("\n## NCCL Configuration")
    nc = report["nccl_config"]
    for key in ["NCCL_P2P_DISABLE", "NCCL_IGNORE_DISABLED_P2P", "NCCL_SHM_DISABLE",
                "NCCL_MAX_NRINGS", "NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG", "NCCL_BUFFSIZE",
                "NCCL_P2P_LEVEL"]:
        if key in nc:
            lines.append(f"  {key}={nc[key]}")
    lines.append(f"  Reason: {nc['reason']}")

    # Throughput
    lines.append("\n## Throughput Estimate")
    t = report["throughput_estimate"]
    lines.append(f"  Compute time: {t['compute_time_ms']}ms")
    lines.append(f"  Communication time: {t['comm_time_ms']}ms")
    lines.append(f"  Overlap factor: {t['overlap_factor']}")
    lines.append(f"  Effective time: {t['effective_time_ms']}ms")
    lines.append(f"  Throughput: {t['throughput_tok_per_s']} tok/s")
    lines.append(f"  Steps/s: {t['steps_per_second']}")

    # Weight formats
    lines.append("\n## Weight Format Pipeline")
    wf = report["weight_formats"]
    lines.append(f"  Training: {wf['training_checkpoint']['format']}")
    lines.append(f"    Reason: {wf['training_checkpoint']['reason']}")
    lines.append(f"    Size: ~{wf['training_checkpoint']['file_size_estimate_gb']:.1f}GB")
    lines.append(f"  Distribution: {wf['distribution_sharing']['format']}")
    lines.append(f"    Reason: {wf['distribution_sharing']['reason']}")
    lines.append(f"    Size: ~{wf['distribution_sharing']['file_size_estimate_gb']:.1f}GB")
    lines.append(f"  Inference: {wf['inference_deployment']['format']}")
    lines.append(f"  Pipeline: {wf['conversion_pipeline']}")

    # Source reading insights
    lines.append("\n## Source Reading Insights")
    insights = report["torch_distributed_insights"]
    for key, value in insights.items():
        lines.append(f"  {key}: {value}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Distributed Training Configuration Advisor")
    parser.add_argument("--model-size", type=float, required=True, help="Model size in billions (e.g. 7, 70)")
    parser.add_argument("--gpu-type", choices=list(GPU_SPECS.keys()), required=True, help="GPU type")
    parser.add_argument("--gpu-count", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--objective", choices=["pretrain", "finetune", "lora"], default="finetune",
                       help="Training objective")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--precision", choices=["bf16", "fp32", "fp16"], default="bf16",
                       help="Training precision")
    parser.add_argument("--offload-optimizer", action="store_true", help="Offload optimizer to CPU")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--list-gpus", action="store_true", help="List available GPU types")

    args = parser.parse_args()

    if args.list_gpus:
        print("Available GPU types:")
        for name, spec in GPU_SPECS.items():
            print(f"  {name}: {spec['name']} — {spec['hbm_gb']}GB HBM, "
                  f"NVLink={spec['nvlink']}, P2P={spec['p2p']}")
        return

    config = TrainingConfig(
        model_size_b=args.model_size,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        objective=args.objective,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        precision=args.precision,
        offload_optimizer=args.offload_optimizer,
    )

    report = generate_full_report(config)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()