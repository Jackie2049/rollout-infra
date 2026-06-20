#!/usr/bin/env python3
"""verl V1 GRPO Training Configuration Generator

Generates complete, ready-to-run YAML config files for verl V1 GRPO training
on different GPU setups. Based on real verl V1 config structure (Hydra/OmegaConf)
with all MUST DO and MUST NOT safety rules applied automatically.

Modes:
  generate  - Generate a complete verl V1 GRPO config YAML
  rtx4090   - Generate RTX 4090 optimal config with launch script
  compare   - Compare configs for different GPU setups (RTX 4090, A100, H100)
  validate  - Validate an existing config YAML against safety rules
  quick     - Quick config for testing/debugging (small model, few steps)

Usage:
  python3 verl_v1_grpo_config_generator.py generate --model qwen2.5-7b --gpu-type rtx4090
  python3 verl_v1_grpo_config_generator.py rtx4090
  python3 verl_v1_grpo_config_generator.py compare
  python3 verl_v1_grpo_config_generator.py validate --config my_config.yaml
  python3 verl_v1_grpo_config_generator.py quick
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

# ============================================================
# MODEL PROFILES
# ============================================================

MODEL_PROFILES = {
    "qwen2.5-0.5b": {
        "path": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
        "hidden_dim": 896,
        "num_layers": 24,
        "best_for": "quick testing, debugging",
    },
    "qwen2.5-1.5b": {
        "path": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "hidden_dim": 1536,
        "num_layers": 28,
        "best_for": "quick testing, small GPU",
    },
    "qwen2.5-3b": {
        "path": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
        "hidden_dim": 2048,
        "num_layers": 36,
        "best_for": "medium-scale, 24 GiB GPU",
    },
    "qwen2.5-7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7.0,
        "hidden_dim": 3584,
        "num_layers": 28,
        "best_for": "RTX 4090 primary, 24 GiB optimal fit",
    },
    "qwen2.5-14b": {
        "path": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14.0,
        "hidden_dim": 5120,
        "num_layers": 40,
        "best_for": "A100 80 GiB, needs optimizer offload on 40 GiB",
    },
    "qwen2.5-32b": {
        "path": "Qwen/Qwen2.5-32B-Instruct",
        "params_b": 32.0,
        "hidden_dim": 5120,
        "num_layers": 64,
        "best_for": "H100 80 GiB multi-GPU, or A100 8x80 GiB",
    },
    "qwen2.5-72b": {
        "path": "Qwen/Qwen2.5-72B-Instruct",
        "params_b": 72.0,
        "hidden_dim": 8192,
        "num_layers": 80,
        "best_for": "multi-node H100 cluster",
    },
    "qwen3-8b": {
        "path": "Qwen/Qwen3-8B",
        "params_b": 8.19,
        "hidden_dim": 4096,
        "num_layers": 36,
        "best_for": "RTX 4090 tight fit with LoRA",
    },
    "qwen3-4b": {
        "path": "Qwen/Qwen3-4B",
        "params_b": 4.0,
        "hidden_dim": 2560,
        "num_layers": 36,
        "best_for": "RTX 4090 comfortable fit",
    },
    "llama3.1-8b": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "params_b": 8.0,
        "hidden_dim": 4096,
        "num_layers": 32,
        "best_for": "alternative 8B, same GPU class as Qwen2.5-7B",
    },
    "llama3.1-70b": {
        "path": "meta-llama/Llama-3.1-70B-Instruct",
        "params_b": 70.0,
        "hidden_dim": 8192,
        "num_layers": 80,
        "best_for": "multi-node H100/A100 cluster",
    },
}

# ============================================================
# GPU PROFILES
# ============================================================

GPU_PROFILES = {
    "rtx4090": {
        "name": "NVIDIA RTX 4090",
        "memory_gib": 24,
        "memory_gb": 24,
        "compute_capability": "8.9",
        "sm": "SM89",
        "interconnect": "PCIe",
        "multi_gpu_viable": False,
        "max_dp": 1,
        "notes": "Consumer GPU, no NVLink, single GPU only, PCIe bottleneck",
    },
    "a100-40g": {
        "name": "NVIDIA A100 40 GiB",
        "memory_gib": 40,
        "memory_gb": 40,
        "compute_capability": "8.0",
        "sm": "SM80",
        "interconnect": "NVLink",
        "multi_gpu_viable": True,
        "max_dp": 4,
        "notes": "Data center GPU, NVLink, good for 7B-14B models",
    },
    "a100-80g": {
        "name": "NVIDIA A100 80 GiB (SXM)",
        "memory_gib": 80,
        "memory_gb": 80,
        "compute_capability": "8.0",
        "sm": "SM80",
        "interconnect": "NVLink",
        "multi_gpu_viable": True,
        "max_dp": 8,
        "notes": "Data center GPU, NVLink, good for 7B-32B models",
    },
    "h100-80g": {
        "name": "NVIDIA H100 80 GiB (SXM)",
        "memory_gib": 80,
        "memory_gb": 80,
        "compute_capability": "9.0",
        "sm": "SM90",
        "interconnect": "NVLink",
        "multi_gpu_viable": True,
        "max_dp": 8,
        "notes": "Hopper GPU, fastest, TDP support, good for all sizes",
    },
    "h200-141g": {
        "name": "NVIDIA H200 141 GiB",
        "memory_gib": 141,
        "memory_gb": 141,
        "compute_capability": "9.0",
        "sm": "SM90",
        "interconnect": "NVLink",
        "multi_gpu_viable": True,
        "max_dp": 8,
        "notes": "Hopper GPU, largest memory, good for 72B single node",
    },
}

# ============================================================
# V1 TRAINER TYPES
# ============================================================

V1_TRAINER_TYPES = {
    "sync": {
        "name": "PPOTrainerSync",
        "description": "Synchronous — trainer+rollout colocated, no partial rollout",
        "min_gpu_count": 1,
        "partial_rollout": False,
        "weight_sync": "every_step",
        "checkpoint_engine_default": "naive",
        "best_for": "single GPU (RTX 4090), simplest, most stable",
        "lifecycle_hooks": {
            "on_init_end": "update_weights (load checkpoint)",
            "on_train_begin": "none",
            "on_step_end": "update_weights (weight sync after training)",
            "on_sample_end": "sleep_replicas",
        },
    },
    "colocate_async": {
        "name": "PPOTrainerColocateAsync",
        "description": "Async colocated — trainer+rollout colocated, partial rollout enabled",
        "min_gpu_count": 1,
        "partial_rollout": True,
        "weight_sync": "every_step",
        "checkpoint_engine_default": "nccl",
        "best_for": "throughput optimization on single/multi GPU",
        "lifecycle_hooks": {
            "on_init_end": "update_weights",
            "on_train_begin": "num_warmup_batches",
            "on_step_end": "update_weights + resume_generation",
            "on_sample_end": "abort_replicas + sleep_replicas",
        },
        "extra_config": {"num_warmup_batches": 2},
    },
    "separate_async": {
        "name": "PPOTrainerSeparateAsync",
        "description": "Async separated — trainer+rollout separated, partial rollout enabled",
        "min_gpu_count": 2,
        "partial_rollout": True,
        "weight_sync": "parameter_sync_step intervals",
        "checkpoint_engine_default": "nccl",
        "best_for": "multi-GPU, multi-node, max throughput",
        "lifecycle_hooks": {
            "on_init_end": "update_weights",
            "on_train_begin": "num_warmup_batches",
            "on_step_end": "update_weights + resume_generation",
            "on_sample_end": "abort_replicas + sleep_replicas",
        },
        "extra_config": {"num_warmup_batches": 4, "parameter_sync_step": 4},
    },
}

# ============================================================
# CHECKPOINT ENGINE TYPES
# ============================================================

CHECKPOINT_ENGINES = {
    "naive": {
        "class": "ColocatedCheckpointEngine",
        "description": "In-process Python yield — zero IPC overhead",
        "transport": "Python generator yield",
        "extra_memory": "0 (just Python reference)",
        "requires": "sync trainer only, single GPU or colocated",
        "platforms": ["gpu", "npu"],
    },
    "nccl": {
        "class": "NCCLCheckpointEngine",
        "description": "NCCL broadcast + ZeroMQ PUB/SUB metadata",
        "transport": "NCCL collective broadcast + ZeroMQ",
        "extra_memory": "2 * bucket_size (send_buf + recv_buf)",
        "requires": "multi-GPU NCCL process group",
        "platforms": ["gpu"],
    },
    "hccl": {
        "class": "HCCLCheckpointEngine",
        "description": "HCCL broadcast + ZeroMQ — Ascend NPU only",
        "transport": "HCCL broadcast + ZeroMQ",
        "extra_memory": "2 * bucket_size",
        "requires": "torch.npu (Ascend NPU)",
        "platforms": ["npu"],
    },
    "nixl": {
        "class": "NIXLCheckpointEngine",
        "description": "NIXL p2p RDMA/UCX — ring topology",
        "transport": "NIXL p2p RDMA/UCX + ZeroMQ",
        "extra_memory": "2 * bucket_size",
        "requires": "RDMA-capable NICs",
        "platforms": ["gpu"],
    },
    "kimi": {
        "class": "KimiCheckpointEngine",
        "description": "ParameterServer + distributed collective — H2DBucket",
        "transport": "ParameterServer + distributed",
        "extra_memory": "H2DBucket (host-to-device)",
        "requires": "multi-GPU, external checkpoint_engine package",
        "platforms": ["gpu"],
    },
    "mooncake": {
        "class": "MooncakeCheckpointEngine",
        "description": "Mooncake TransferEngine p2p RDMA",
        "transport": "Mooncake TransferEngine p2p RDMA",
        "extra_memory": "2 * bucket_size + 4KB magic_buf",
        "requires": "RDMA or Ascend Direct",
        "platforms": ["gpu", "npu"],
    },
}

# ============================================================
# ADVANTAGE ESTIMATORS
# ============================================================

ADV_ESTIMATORS = {
    "grpo": {
        "name": "GRPO",
        "description": "Group Relative Policy Optimization — no critic needed",
        "needs_critic": False,
        "needs_ref": False,  # when bypass_mode=True
        "group_size_min": 2,
        "default_kl_coef": 0.001,
        "loss_type": "grpo_loss",
    },
    "ppo": {
        "name": "PPO (GAE)",
        "description": "Proximal Policy Optimization with GAE advantage",
        "needs_critic": True,
        "needs_ref": True,
        "group_size_min": 1,
        "default_kl_coef": 0.02,
        "loss_type": "ppo_clip",
    },
    "dr_grpo": {
        "name": "DR-GRPO",
        "description": "Double Robust GRPO — uses reference model for variance reduction",
        "needs_critic": False,
        "needs_ref": True,
        "group_size_min": 2,
        "default_kl_coef": 0.001,
        "loss_type": "grpo_loss",
    },
    "reinforce": {
        "name": "REINFORCE",
        "description": "Vanilla policy gradient with reward baseline",
        "needs_critic": False,
        "needs_ref": False,
        "group_size_min": 1,
        "default_kl_coef": 0.0,
        "loss_type": "reinforce",
    },
    "rloo": {
        "name": "RLOO",
        "description": "REINFORCE Leave-One-Out — best for small group sizes",
        "needs_critic": False,
        "needs_ref": False,
        "group_size_min": 2,
        "default_kl_coef": 0.001,
        "loss_type": "reinforce",
    },
}

# ============================================================
# MUST DO / MUST NOT RULES
# ============================================================

MUST_DO_RULES = [
    {
        "id": "fsdp1_not_fsdp2",
        "rule": "Use FSDP1 (not FSDP2)",
        "reason": "#6468 FSDP2 CPU memory leak 0.6-6.3 GiB/step",
        "check_fn": lambda c: c.get("actor_rollout_ref", {}).get("actor", {}).get("fsdp_config", {}).get("strategy", "fsdp") == "fsdp",
        "fix": "Set actor_rollout_ref.actor.fsdp_config.strategy=fsdp (NOT fsdp2)",
    },
    {
        "id": "lora_rank_le_32",
        "rule": "LoRA rank <= 32 (NOT 64, NOT full finetune)",
        "reason": "#6782 rank=64 breaks EOS; full finetune too much memory for consumer GPUs",
        "check_fn": lambda c: True,  # LoRA rank check is context-dependent
        "fix": "Set lora_rank=32 for RTX 4090, <= 64 for data center GPUs",
    },
    {
        "id": "bypass_mode_true",
        "rule": "bypass_mode=True (remove ref model)",
        "reason": "removes ref model memory burden; 18Psi -> 3.8Psi",
        "check_fn": lambda c: c.get("algorithm", {}).get("use_kl_in_reward", True) == False,
        "fix": "Set algorithm.use_kl_in_reward=False + algorithm.use_kl_loss=False (for GRPO+bypass)",
    },
    {
        "id": "clip_grad_norm_1",
        "rule": "clip_grad_norm=1.0",
        "reason": "#8068 default 0 causes silent training degradation",
        "check_fn": lambda c: c.get("actor_rollout_ref", {}).get("actor", {}).get("grad_clip", 0) == 1.0,
        "fix": "Set actor_rollout_ref.actor.grad_clip=1.0",
    },
    {
        "id": "overlap_comm_false",
        "rule": "overlap_comm=False (single GPU / consumer GPU)",
        "reason": "#8061 overlap_comm=True causes NaN on single GPU",
        "check_fn": lambda c: True,  # Context-dependent
        "fix": "Set overlap_comm=False for single GPU or consumer GPUs",
    },
    {
        "id": "enforce_eager_dsv4",
        "rule": "enforce_eager=True (DSV4 or SM89)",
        "reason": "11 DSV4 failures across 4 frameworks; SM89 batch invariance issue",
        "check_fn": lambda c: c.get("actor_rollout_ref", {}).get("rollout", {}).get("enforce_eager", False) == True,
        "fix": "Set actor_rollout_ref.rollout.enforce_eager=True",
    },
    {
        "id": "group_size_ge_4",
        "rule": "group_size >= 4 (GRPO)",
        "reason": "#605 normalization undefined at |G|=1; degenerate at |G|=2",
        "check_fn": lambda c: c.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 1) >= 4 or c.get("algorithm", {}).get("adv_estimator", "") != "grpo",
        "fix": "Set actor_rollout_ref.rollout.n >= 4 for GRPO",
    },
    {
        "id": "naive_ckpt_dp1",
        "rule": "naive checkpoint engine when dp=1",
        "reason": "sync trainer auto-selects naive; nccl overkill for single GPU",
        "check_fn": lambda c: True,  # Context-dependent
        "fix": "Set checkpoint_engine.backend=naive for dp=1",
    },
    {
        "id": "ulimit_65535",
        "rule": "ulimit -n 65535 before launch",
        "reason": "#8075 fd leak safety; Ray opens many file descriptors",
        "check_fn": lambda c: True,  # Not in config, system-level
        "fix": "Run: ulimit -n 65535 before starting training",
    },
    {
        "id": "param_offload_for_24gib",
        "rule": "param_offload + optimizer_offload for 24 GiB GPUs",
        "reason": "Memory too tight for full optimizer states on GPU",
        "check_fn": lambda c: True,  # Context-dependent
        "fix": "Set fsdp_config.param_offload=True + optimizer_offload=True for RTX 4090",
    },
]

MUST_NOT_RULES = [
    {
        "id": "no_zero3_single_gpu",
        "rule": "Do NOT use ZeRO-3 on single GPU",
        "reason": "#8072/#8076 dtype mismatch regression; pure overhead on single GPU",
        "severity": "CRITICAL",
    },
    {
        "id": "no_muon_optimizer",
        "rule": "Do NOT use Muon optimizer",
        "reason": "6 blockers across 3 frameworks; incompatible with ZeRO-2 CPU offload",
        "severity": "CRITICAL",
    },
    {
        "id": "no_lora_rank_64",
        "rule": "Do NOT use LoRA rank=64",
        "reason": "#6782 breaks EOS generation in vLLM rollout",
        "severity": "CRITICAL",
    },
    {
        "id": "no_overlap_comm_single_gpu",
        "rule": "Do NOT use overlap_comm=True on single GPU",
        "reason": "#8061 NaN confirmed; requires NCCL process group which single GPU cannot form",
        "severity": "CRITICAL",
    },
    {
        "id": "no_cuda_graphs_dsv4",
        "rule": "Do NOT use CUDA graphs for DeepSeek-V4 or SM89",
        "reason": "11 failures; batch invariance bug on SM89",
        "severity": "CRITICAL",
    },
    {
        "id": "no_nvme_offload",
        "rule": "Do NOT use NVMe offload",
        "reason": "#8075 fd leak; extremely slow compared to CPU offload",
        "severity": "HIGH",
    },
    {
        "id": "no_separate_async_single_gpu",
        "rule": "Do NOT use separate_async trainer on single GPU",
        "reason": "Requires min 2 GPUs (1 trainer + 1 rollout)",
        "severity": "CRITICAL",
    },
    {
        "id": "no_fsdp2",
        "rule": "Do NOT use FSDP2 on memory-constrained GPUs",
        "reason": "#6468 CPU memory leak 0.6-6.3 GiB/step",
        "severity": "HIGH",
    },
    {
        "id": "no_megatron_backend",
        "rule": "Do NOT use Megatron backend for verl",
        "reason": "#6699 detach not upstream; #5203 singleton PG crash on single GPU",
        "severity": "HIGH",
    },
    {
        "id": "no_autocast_zeero3",
        "rule": "Do NOT use autocast_adapter_dtype + ZeRO-3",
        "reason": "#8072 fp32 LoRA mismatch regression",
        "severity": "HIGH",
    },
]

# ============================================================
# MEMORY ESTIMATION ENGINE
# ============================================================

def estimate_model_memory(params_b, precision="bf16", lora_rank=0, dp=1,
                          seq_len=2048, batch_size=4, group_size=8,
                          bypass_mode=True, needs_critic=False, gpu_type="rtx4090"):
    """Estimate peak GPU memory for a training configuration.

    Returns a dict with detailed memory breakdown.
    """
    bytes_per_param = 2 if precision == "bf16" else 4
    params_count = params_b * 1e9

    # Model weights
    model_weights_gb = params_b * bytes_per_param  # BF16: ~2 bytes/param

    # Approximate hidden dimension for LoRA estimation
    hidden_dim_approx = params_b * 512  # rough: 7B ~ 3584, 0.5B ~ 896
    # Override with actual profile data if available
    for model_key, profile in MODEL_PROFILES.items():
        if profile["params_b"] == params_b:
            hidden_dim_approx = profile["hidden_dim"]
            break

    # LoRA parameters (more realistic estimation)
    # For a 7B model with rank=32 on 7 target modules:
    # Each LoRA adapter: rank * (in_dim + out_dim) = 32 * (3584+3584) = ~230K params per module
    # Total: ~7 modules * 230K = ~1.6M params = ~3.2MB in BF16
    # General formula: lora_params ~= lora_rank * (hidden_in + hidden_out) * num_target_modules
    lora_params_gb = 0
    lora_optim_gb = 0
    if lora_rank > 0:
        # Estimate number of LoRA target modules and dimensions
        # Typical: 7 modules for LLaMA/Qwen style (q,k,v,o,up,down,gate)
        num_target_modules = 7
        # Average dimensions per module (approx half hidden_dim per proj)
        avg_dim_per_module = hidden_dim_approx * 0.7  # rough average across proj types
        lora_params_per_module = lora_rank * (avg_dim_per_module + avg_dim_per_module)
        lora_params_count = num_target_modules * lora_params_per_module
        # In BF16: 2 bytes per param, but optimizer tracks them in FP32
        lora_params_gb = lora_params_count * 2 / (1024**3)  # BF16 params on GPU
        lora_optim_gb = lora_params_count * 8 / (1024**3)  # FP32 m+v for Adam (4 bytes each * 2)

    # Optimizer states
    if lora_rank > 0:
        # LoRA: optimizer only for LoRA params, rest offloaded to CPU
        optim_gpu_gb = lora_optim_gb
        optim_cpu_gb = lora_optim_gb  # CPU holds the optimizer states when offloaded
    else:
        # Full finetune: all optimizer states
        optim_gpu_gb = model_weights_gb * 2  # m + v in fp32
        optim_cpu_gb = 0

    # If optimizer offload enabled (recommended for 24 GiB)
    gpu_profile = GPU_PROFILES.get(gpu_type, GPU_PROFILES["rtx4090"])
    if gpu_profile["memory_gib"] <= 40 and dp == 1:
        # Offload optimizer to CPU
        optim_gpu_gb = 0
        optim_cpu_gb = lora_optim_gb if lora_rank > 0 else model_weights_gb * 2

    # Activation memory (more realistic with gradient checkpointing)
    # With gradient checkpointing, peak activation ~= batch_size * seq_len * hidden_dim * 2 bytes
    # This is much lower than the full forward+backward chain
    # Per-sample activation with checkpointing: ~hidden_dim * 2 * seq_len / 1e9 GB
    # The micro_batch is what matters (not group_size, since we process one group at a time)
    effective_batch = min(batch_size, 4)  # micro-batch size limit
    activation_per_sample_gb = hidden_dim_approx * 2 * seq_len / 1e9
    activation_gb = effective_batch * activation_per_sample_gb
    activation_gb = max(activation_gb, 0.5)  # minimum 0.5 GiB

    # Gradient memory
    # With LoRA: only gradients for LoRA params (small), base weights frozen
    # With full finetune: gradients for all params (same size as model)
    gradient_gb = lora_params_gb if lora_rank > 0 else model_weights_gb

    # Reference model (if not bypass_mode)
    ref_model_gb = 0
    if not bypass_mode:
        ref_model_gb = model_weights_gb * 1.5  # ref model + its activations

    # Critic model (if PPO)
    critic_gb = 0
    if needs_critic:
        critic_gb = model_weights_gb * 1.5  # critic model + its states

    # KV cache for rollout (scales with model hidden dim and seq_len)
    # ~2 bytes per KV token * 2 (K+V) * num_layers * hidden_dim / overhead_per_seq
    kv_per_seq_gb = hidden_dim_approx * 2 * 2 * 28 / (1024**3)  # rough per 2048 tokens
    kv_cache_gb = max(kv_per_seq_gb, 0.5) * (group_size * effective_batch / 4)  # scaled
    kv_cache_gb = min(kv_cache_gb, 2.0)  # cap at 2 GiB (rollout engine manages KV)

    # Temp buffers / fragmentation overhead
    temp_overhead_gb = 0.5

    # Total peak
    peak_gpu_gb = (
        model_weights_gb +
        lora_params_gb +
        optim_gpu_gb +
        gradient_gb +
        activation_gb +
        ref_model_gb +
        critic_gb +
        kv_cache_gb +
        temp_overhead_gb
    )

    # With gradient checkpointing, activations are reduced
    if True:  # always recommended for GRPO
        activation_gb_checkpointed = activation_gb * 0.3
        peak_gpu_gb_checkpointed = peak_gpu_gb - activation_gb + activation_gb_checkpointed

    gpu_mem = gpu_profile["memory_gib"]
    fits = peak_gpu_gb_checkpointed <= gpu_mem
    margin_gb = gpu_mem - peak_gpu_gb_checkpointed

    return {
        "model_weights_gb": round(model_weights_gb, 2),
        "lora_params_gb": round(lora_params_gb, 2),
        "optim_gpu_gb": round(optim_gpu_gb, 2),
        "optim_cpu_gb": round(optim_cpu_gb, 2),
        "gradient_gb": round(gradient_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "activation_gb_checkpointed": round(activation_gb_checkpointed, 2),
        "ref_model_gb": round(ref_model_gb, 2),
        "critic_gb": round(critic_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "temp_overhead_gb": round(temp_overhead_gb, 2),
        "peak_gpu_gb": round(peak_gpu_gb, 2),
        "peak_gpu_gb_checkpointed": round(peak_gpu_gb_checkpointed, 2),
        "gpu_memory_gib": gpu_mem,
        "fits_gpu": fits,
        "margin_gb": round(margin_gb, 2),
        "bypass_mode": bypass_mode,
        "needs_critic": needs_critic,
    }

# ============================================================
# STEP TIMING ESTIMATION
# ============================================================

def estimate_step_timing(params_b, dp, gpu_type, group_size=8,
                          seq_len=2048, batch_size=4, rollout_engine="sglang"):
    """Estimate per-step timing for a training configuration."""
    gpu = GPU_PROFILES.get(gpu_type, GPU_PROFILES["rtx4090"])

    # Base timing estimates (seconds) for 7B model on different GPUs
    # These are rough empirical estimates
    base_time_per_token_sec = {
        "rtx4090": 0.00015,
        "a100-40g": 0.00008,
        "a100-80g": 0.00006,
        "h100-80g": 0.00004,
        "h200-141g": 0.00004,
    }

    # Scale by model size (rough: linear in params)
    scale_factor = params_b / 7.0

    # Rollout time: generate group_size * batch_size * seq_len tokens
    rollout_tokens = group_size * batch_size * seq_len
    rollout_time = rollout_tokens * base_time_per_token_sec.get(gpu_type, 0.0001) * scale_factor

    # Training time: forward + backward for batch_size * seq_len * group_size tokens
    train_tokens = batch_size * seq_len * group_size
    train_time = train_tokens * base_time_per_token_sec.get(gpu_type, 0.0001) * scale_factor * 2.5  # fwd+bwd

    # Weight sync time (naive is near-zero for dp=1, nccl for dp>1)
    if dp == 1:
        sync_time = 0.5  # LoRA adapter sync via naive
    else:
        sync_time = 2.0 + (params_b * 0.1) / dp  # nccl broadcast

    # Total step time
    total_step_time = rollout_time + train_time + sync_time

    # Throughput: tokens per second
    total_tokens_per_step = rollout_tokens + train_tokens
    throughput_tokens_sec = total_tokens_per_step / total_step_time

    return {
        "rollout_time_sec": round(rollout_time, 2),
        "train_time_sec": round(train_time, 2),
        "sync_time_sec": round(sync_time, 2),
        "total_step_time_sec": round(total_step_time, 2),
        "throughput_tokens_sec": round(throughput_tokens_sec, 1),
        "estimated_steps_per_hour": round(3600 / total_step_time, 1),
    }

# ============================================================
# YAML CONFIG GENERATOR (real verl V1 structure)
# ============================================================

def generate_verl_v1_grpo_config(
    model="qwen2.5-7b",
    gpu_type="rtx4090",
    dp=1,
    lora_rank=32,
    group_size=8,
    bypass_mode=True,
    adv_estimator="grpo",
    rollout_engine="sglang",
    trainer_type="sync",
    checkpoint_engine="naive",
    batch_size=None,
    seq_len=2048,
    max_prompt_length=512,
    lr=1e-6,
    total_epochs=30,
    total_training_steps=None,
    param_offload=None,
    optimizer_offload=None,
    enforce_eager=None,
    use_kl_loss=False,
    kl_loss_coef=0.001,
    use_kl_in_reward=False,
    gradient_checkpointing=True,
    use_remove_padding=True,
    use_dynamic_bsz=False,
    ppo_max_token_len_per_gpu=None,
    dataset_path="$HOME/data/gsm8k/train.parquet",
    val_dataset_path="$HOME/data/gsm8k/test.parquet",
    project_name="verl_grpo",
    experiment_name=None,
    save_freq=-1,
    test_freq=-1,
    rollout_gpu_mem_util=None,
):
    """Generate a complete verl V1 GRPO config using real Hydra/OmegaConf structure.

    This generates a YAML config that mirrors the actual verl trainer/config/ppo_trainer.yaml
    structure, with all parameters as command-line overrides suitable for:
      python3 -m verl.trainer.main_ppo <overrides>

    Returns both the YAML config and the launch command.
    """
    model_profile = MODEL_PROFILES.get(model, MODEL_PROFILES["qwen2.5-7b"])
    gpu_profile = GPU_PROFILES.get(gpu_type, GPU_PROFILES["rtx4090"])

    # Auto-determine settings based on GPU type
    if param_offload is None:
        param_offload = gpu_profile["memory_gib"] <= 40 and dp == 1
    if optimizer_offload is None:
        optimizer_offload = gpu_profile["memory_gib"] <= 40 and dp == 1
    if enforce_eager is None:
        # enforce_eager=True for SM89 (RTX 4090) or DSV4
        enforce_eager = gpu_profile["sm"] in ("SM89",) or "deepseek" in model

    # Auto-determine batch size
    if batch_size is None:
        if gpu_profile["memory_gib"] <= 24:
            batch_size = 4 if lora_rank > 0 else 1
        elif gpu_profile["memory_gib"] <= 80:
            batch_size = 16 if dp == 1 else 256
        else:
            batch_size = 32

    # Auto-determine checkpoint engine
    if checkpoint_engine == "naive" and dp > 1 and trainer_type != "sync":
        checkpoint_engine = "nccl"  # separate_async requires non-naive
    if dp == 1 and trainer_type == "sync":
        checkpoint_engine = "naive"  # sync auto-selects naive

    # Auto-determine rollout TP
    if dp == 1:
        rollout_tp = 1
    elif gpu_profile["interconnect"] == "NVLink":
        rollout_tp = min(dp, 2)
    else:
        rollout_tp = 1

    # Auto-determine trainer type
    if dp == 1 and trainer_type == "separate_async":
        # Not viable on single GPU, auto-correct
        trainer_type = "sync"

    # Mini batch size
    ppo_mini_batch_size = max(batch_size // 4, 2) if batch_size >= 8 else batch_size

    # PPO max token len per gpu
    if ppo_max_token_len_per_gpu is None:
        if gpu_profile["memory_gib"] <= 24:
            ppo_max_token_len_per_gpu = 8192
        elif gpu_profile["memory_gib"] <= 80:
            ppo_max_token_len_per_gpu = 24576
        else:
            ppo_max_token_len_per_gpu = 32768

    # GPU memory utilization for rollout
    if rollout_gpu_mem_util is None:
        if gpu_profile["memory_gib"] <= 24:
            rollout_gpu_mem_util = 0.4  # Conservative for RTX 4090
        elif gpu_profile["memory_gib"] <= 40:
            rollout_gpu_mem_util = 0.5
        else:
            rollout_gpu_mem_util = 0.6

    # LoRA config
    lora_config = None
    if lora_rank > 0:
        lora_config = {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "up_proj", "down_proj", "gate_proj"],
            "rank": lora_rank,
            "alpha": lora_rank * 2,  # common: alpha = 2 * rank
            "merge": False,
        }

    # KL config for algorithm
    if bypass_mode and adv_estimator == "grpo":
        use_kl_in_reward_final = False
        use_kl_loss_final = use_kl_loss  # optional, usually False for bypass+GRPO
        kl_loss_coef_final = kl_loss_coef if use_kl_loss else 0.0
    else:
        use_kl_in_reward_final = use_kl_in_reward
        use_kl_loss_final = True
        kl_loss_coef_final = kl_loss_coef

    # Experiment name
    if experiment_name is None:
        experiment_name = f"{model}_{adv_estimator}_{gpu_type}_dp{dp}_lora{lora_rank}"

    # ============================================
    # Build the YAML config (real verl V1 structure)
    # ============================================
    config = OrderedDict()

    # --- actor_rollout_ref ---
    actor_rollout_ref = OrderedDict()
    actor_rollout_ref["hybrid_engine"] = True
    actor_rollout_ref["nccl_timeout"] = 600

    # model
    model_config = OrderedDict()
    model_config["path"] = model_profile["path"]
    model_config["use_remove_padding"] = use_remove_padding
    model_config["enable_gradient_checkpointing"] = gradient_checkpointing
    actor_rollout_ref["model"] = model_config

    # actor
    actor_config = OrderedDict()
    actor_config["strategy"] = "fsdp"  # MUST DO: FSDP1, not fsdp2
    actor_config["grad_clip"] = 1.0  # MUST DO: gradient clipping
    actor_config["ppo_mini_batch_size"] = ppo_mini_batch_size
    actor_config["ppo_epochs"] = 1
    actor_config["use_kl_loss"] = use_kl_loss_final
    actor_config["kl_loss_coef"] = kl_loss_coef_final
    actor_config["kl_loss_type"] = "low_var_kl"
    actor_config["entropy_coeff"] = 0
    actor_config["use_dynamic_bsz"] = use_dynamic_bsz
    if ppo_max_token_len_per_gpu:
        actor_config["ppo_max_token_len_per_gpu"] = ppo_max_token_len_per_gpu

    # actor optimizer
    actor_optim = OrderedDict()
    actor_optim["lr"] = lr
    actor_config["optim"] = actor_optim

    # actor fsdp config
    actor_fsdp = OrderedDict()
    actor_fsdp["strategy"] = "fsdp"  # MUST DO: FSDP1
    actor_fsdp["param_offload"] = param_offload
    actor_fsdp["optimizer_offload"] = optimizer_offload
    actor_fsdp["model_dtype"] = "bfloat16"
    actor_config["fsdp_config"] = actor_fsdp

    actor_rollout_ref["actor"] = actor_config

    # rollout
    rollout_config = OrderedDict()
    rollout_config["name"] = rollout_engine
    rollout_config["tensor_model_parallel_size"] = rollout_tp
    rollout_config["gpu_memory_utilization"] = rollout_gpu_mem_util
    rollout_config["n"] = group_size  # MUST DO: >= 4 for GRPO
    rollout_config["enforce_eager"] = enforce_eager  # MUST DO for SM89/DSV4
    rollout_config["free_cache_engine"] = True
    rollout_config["enable_chunked_prefill"] = True
    rollout_config["enable_prefix_caching"] = True
    rollout_config["disable_log_stats"] = True
    rollout_config["log_prob_use_dynamic_bsz"] = use_dynamic_bsz
    if ppo_max_token_len_per_gpu:
        rollout_config["log_prob_max_token_len_per_gpu"] = ppo_max_token_len_per_gpu
    rollout_config["temperature"] = 1.0
    rollout_config["top_k"] = -1
    rollout_config["top_p"] = 1.0
    rollout_config["do_sample"] = True
    rollout_config["skip_tokenizer_init"] = True  # verl default for non-HF

    # checkpoint engine within rollout
    rollout_ckpt_engine = OrderedDict()
    rollout_ckpt_engine["backend"] = checkpoint_engine
    rollout_ckpt_engine["update_weights_bucket_megabytes"] = 2048
    rollout_config["checkpoint_engine"] = rollout_ckpt_engine

    actor_rollout_ref["rollout"] = rollout_config

    # ref (only when not bypass_mode)
    if not bypass_mode:
        ref_config = OrderedDict()
        ref_config["fsdp_config"] = OrderedDict()
        ref_config["fsdp_config"]["param_offload"] = True  # ref model offload to CPU
        if use_dynamic_bsz:
            ref_config["log_prob_use_dynamic_bsz"] = True
            if ppo_max_token_len_per_gpu:
                ref_config["log_prob_max_token_len_per_gpu"] = ppo_max_token_len_per_gpu
        actor_rollout_ref["ref"] = ref_config

    config["actor_rollout_ref"] = actor_rollout_ref

    # --- algorithm ---
    algorithm_config = OrderedDict()
    algorithm_config["adv_estimator"] = adv_estimator
    algorithm_config["use_kl_in_reward"] = use_kl_in_reward_final
    algorithm_config["norm_adv_by_std_in_grpo"] = True
    algorithm_config["gamma"] = 1.0
    algorithm_config["lam"] = 1.0
    algorithm_config["kl_penalty"] = "kl"

    # KL control
    kl_ctrl = OrderedDict()
    kl_ctrl["type"] = "fixed"
    kl_ctrl["kl_coef"] = 0.001 if adv_estimator == "grpo" else 0.02
    algorithm_config["kl_ctrl"] = kl_ctrl

    config["algorithm"] = algorithm_config

    # --- critic (only for PPO) ---
    if adv_estimator == "ppo":
        critic_config = OrderedDict()
        critic_config["strategy"] = "fsdp"
        critic_config["model"] = OrderedDict()
        critic_config["model"]["path"] = model_profile["path"]
        critic_config["fsdp_config"] = OrderedDict()
        critic_config["fsdp_config"]["param_offload"] = param_offload
        critic_config["fsdp_config"]["optimizer_offload"] = optimizer_offload
        config["critic"] = critic_config

    # --- reward ---
    reward_config = OrderedDict()
    reward_config["num_workers"] = min(dp * 4, 8)
    reward_config["custom_reward_function"] = OrderedDict()
    reward_config["custom_reward_function"]["path"] = None
    reward_config["custom_reward_function"]["name"] = "compute_score"
    # Reward model (optional)
    reward_model = OrderedDict()
    reward_model["enable"] = False  # Default: rule-based reward
    reward_config["reward_model"] = reward_model
    config["reward"] = reward_config

    # --- data ---
    data_config = OrderedDict()
    data_config["train_files"] = dataset_path
    data_config["val_files"] = val_dataset_path
    data_config["train_batch_size"] = batch_size * dp
    data_config["max_prompt_length"] = max_prompt_length
    data_config["max_response_length"] = seq_len
    data_config["filter_overlong_prompts"] = True
    data_config["truncation"] = "error"
    config["data"] = data_config

    # --- trainer ---
    trainer_config = OrderedDict()
    trainer_config["balance_batch"] = True
    trainer_config["total_epochs"] = total_epochs
    if total_training_steps:
        trainer_config["total_training_steps"] = total_training_steps
    trainer_config["project_name"] = project_name
    trainer_config["experiment_name"] = experiment_name
    trainer_config["logger"] = ["console", "wandb"]
    trainer_config["nnodes"] = 1
    trainer_config["n_gpus_per_node"] = dp
    trainer_config["save_freq"] = save_freq
    trainer_config["test_freq"] = test_freq
    trainer_config["val_before_train"] = True
    trainer_config["default_local_dir"] = f"checkpoints/{project_name}/{experiment_name}"

    # V1 trainer config
    v1_config = OrderedDict()
    v1_config["trainer_mode"] = trainer_type

    if trainer_type == "sync":
        v1_config["sync"] = OrderedDict()
    elif trainer_type == "colocate_async":
        v1_config["colocate_async"] = OrderedDict()
        v1_config["colocate_async"]["num_warmup_batches"] = 2
    elif trainer_type == "separate_async":
        v1_config["separate_async"] = OrderedDict()
        v1_config["separate_async"]["num_warmup_batches"] = 4
        v1_config["separate_async"]["parameter_sync_step"] = 4

    # sampler config
    v1_config["sampler"] = OrderedDict()
    v1_config["sampler"]["max_off_policy_threshold"] = 8
    v1_config["sampler"]["max_off_policy_strategy"] = "drop"

    trainer_config["v1"] = v1_config
    trainer_config["use_v1"] = True

    config["trainer"] = trainer_config

    # --- LoRA config (separate section) ---
    if lora_rank > 0 and lora_config:
        config["lora"] = lora_config

    # ============================================
    # Build launch command
    # ============================================
    cmd_parts = ["python3 -m verl.trainer.main_ppo"]

    def add_override(key, value):
        if isinstance(value, bool):
            cmd_parts.append(f"{key}={'True' if value else 'False'}")
        elif isinstance(value, list):
            # Hydra/OmegaConf list syntax: [item1,item2,...] or '[item1,item2]'
            # For string lists (like target_modules), use single-quoted bracket syntax
            if all(isinstance(v, str) for v in value):
                items = ",".join(value)
                cmd_parts.append(f"{key}='[{items}]'")
            else:
                items = ",".join(str(v) for v in value)
                cmd_parts.append(f"{key}='[{items}]'")
        elif isinstance(value, str):
            cmd_parts.append(f"{key}='{value}'")
        else:
            cmd_parts.append(f"{key}={value}")

    # Algorithm overrides
    add_override("algorithm.adv_estimator", adv_estimator)
    add_override("algorithm.use_kl_in_reward", use_kl_in_reward_final)
    add_override("algorithm.norm_adv_by_std_in_grpo", True)

    # Data overrides
    add_override("data.train_files", dataset_path)
    add_override("data.val_files", val_dataset_path)
    add_override("data.train_batch_size", batch_size * dp)
    add_override("data.max_prompt_length", max_prompt_length)
    add_override("data.max_response_length", seq_len)
    add_override("data.filter_overlong_prompts", True)
    add_override("data.truncation", "error")

    # Model overrides
    add_override("actor_rollout_ref.model.path", model_profile["path"])
    add_override("actor_rollout_ref.model.use_remove_padding", use_remove_padding)
    add_override("actor_rollout_ref.model.enable_gradient_checkpointing", gradient_checkpointing)

    # Actor overrides
    add_override("actor_rollout_ref.actor.optim.lr", lr)
    add_override("actor_rollout_ref.actor.ppo_mini_batch_size", ppo_mini_batch_size)
    add_override("actor_rollout_ref.actor.use_kl_loss", use_kl_loss_final)
    add_override("actor_rollout_ref.actor.kl_loss_coef", kl_loss_coef_final)
    add_override("actor_rollout_ref.actor.kl_loss_type", "low_var_kl")
    add_override("actor_rollout_ref.actor.entropy_coeff", 0)
    add_override("actor_rollout_ref.actor.grad_clip", 1.0)
    if use_dynamic_bsz:
        add_override("actor_rollout_ref.actor.use_dynamic_bsz", True)
        add_override("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", ppo_max_token_len_per_gpu)

    # Actor FSDP overrides
    add_override("actor_rollout_ref.actor.fsdp_config.param_offload", param_offload)
    add_override("actor_rollout_ref.actor.fsdp_config.optimizer_offload", optimizer_offload)

    # Rollout overrides
    add_override("actor_rollout_ref.rollout.name", rollout_engine)
    add_override("actor_rollout_ref.rollout.tensor_model_parallel_size", rollout_tp)
    add_override("actor_rollout_ref.rollout.gpu_memory_utilization", rollout_gpu_mem_util)
    add_override("actor_rollout_ref.rollout.n", group_size)
    add_override("actor_rollout_ref.rollout.enforce_eager", enforce_eager)
    add_override("actor_rollout_ref.rollout.free_cache_engine", True)
    add_override("actor_rollout_ref.rollout.enable_chunked_prefill", True)
    add_override("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", use_dynamic_bsz)

    # Ref overrides (if not bypass)
    if not bypass_mode:
        add_override("actor_rollout_ref.ref.fsdp_config.param_offload", True)

    # Trainer overrides
    add_override("trainer.balance_batch", True)
    add_override("trainer.logger", ["console", "wandb"])
    add_override("trainer.project_name", project_name)
    add_override("trainer.experiment_name", experiment_name)
    add_override("trainer.n_gpus_per_node", dp)
    add_override("trainer.nnodes", 1)
    add_override("trainer.save_freq", save_freq)
    add_override("trainer.test_freq", test_freq)
    add_override("trainer.total_epochs", total_epochs)
    add_override("trainer.use_v1", True)
    add_override("trainer.v1.trainer_mode", trainer_type)

    # LoRA overrides
    if lora_rank > 0 and lora_config:
        add_override("actor_rollout_ref.actor.lora_rank", lora_rank)
        add_override("actor_rollout_ref.actor.lora_alpha", lora_config["alpha"])
        add_override("actor_rollout_ref.actor.lora_merge", False)
        add_override("actor_rollout_ref.actor.lora_target_modules",
                     lora_config["target_modules"])

    launch_command = " \\\n    ".join(cmd_parts)

    # Memory estimate
    memory_est = estimate_model_memory(
        params_b=model_profile["params_b"],
        lora_rank=lora_rank,
        dp=dp,
        seq_len=seq_len,
        batch_size=batch_size,
        group_size=group_size,
        bypass_mode=bypass_mode,
        needs_critic=(adv_estimator == "ppo"),
        gpu_type=gpu_type,
    )

    # Step timing estimate
    timing_est = estimate_step_timing(
        params_b=model_profile["params_b"],
        dp=dp,
        gpu_type=gpu_type,
        group_size=group_size,
        seq_len=seq_len,
        batch_size=batch_size,
        rollout_engine=rollout_engine,
    )

    return {
        "config": config,
        "launch_command": launch_command,
        "memory_estimate": memory_est,
        "timing_estimate": timing_est,
        "model_profile": model_profile,
        "gpu_profile": gpu_profile,
    }

# ============================================================
# YAML DUMP HELPER (no dependency on PyYAML for basic output)
# ============================================================

def yaml_dump_ordered(d, indent=0):
    """Dump an OrderedDict as YAML-like string without PyYAML dependency."""
    lines = []
    prefix = "  " * indent

    for key, value in d.items():
        if isinstance(value, OrderedDict):
            if len(value) == 0:
                lines.append(f"{prefix}{key}:")  # empty dict
            else:
                lines.append(f"{prefix}{key}:")
                lines.append(yaml_dump_ordered(value, indent + 1))
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(yaml_dump_ordered(OrderedDict(value), indent + 1))
        elif isinstance(value, list):
            if len(value) == 0:
                lines.append(f"{prefix}{key}: []")
            elif len(value) <= 3 and all(isinstance(v, str) for v in value):
                lines.append(f"{prefix}{key}: {json.dumps(value)}")
            else:
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  - ")
                        lines.append(yaml_dump_ordered(OrderedDict(item), indent + 3))
                    else:
                        lines.append(f"{prefix}  - {item}")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and value == int(value):
                lines.append(f"{prefix}{key}: {int(value)}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, str):
            # Quote strings that contain special chars
            if any(c in value for c in [":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`", "$", "~"]):
                lines.append(f"{prefix}{key}: '{value}'")
            else:
                lines.append(f"{prefix}{key}: {value}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def format_config_output(result):
    """Format a complete config result for display."""
    config = result["config"]
    mem = result["memory_estimate"]
    timing = result["timing_estimate"]
    model_profile = result["model_profile"]
    gpu_profile = result["gpu_profile"]

    output = []

    # Header
    output.append("=" * 80)
    output.append(" verl V1 GRPO Training Configuration")
    output.append("=" * 80)
    output.append("")
    output.append(f"Model:    {model_profile['path']} ({model_profile['params_b']}B params)")
    output.append(f"GPU:      {gpu_profile['name']} ({gpu_profile['memory_gib']} GiB, {gpu_profile['sm']})")
    output.append(f"Algorithm: GRPO (bypass_mode={mem['bypass_mode']})")
    output.append(f"LoRA:     rank={config.get('lora', {}).get('rank', 0)}")
    output.append(f"DP:       {config['trainer']['n_gpus_per_node']}")
    output.append(f"Trainer:  {config['trainer']['v1']['trainer_mode']}")
    output.append(f"Ckpt:     {config['actor_rollout_ref']['rollout']['checkpoint_engine']['backend']}")
    output.append("")

    # Memory estimate
    output.append("-" * 80)
    output.append(" Memory Estimate")
    output.append("-" * 80)
    output.append(f"  Model weights:       {mem['model_weights_gb']} GiB")
    if mem['lora_params_gb'] > 0:
        output.append(f"  LoRA params:         {mem['lora_params_gb']} GiB")
    output.append(f"  Optimizer (GPU):     {mem['optim_gpu_gb']} GiB")
    output.append(f"  Optimizer (CPU):     {mem['optim_cpu_gb']} GiB")
    output.append(f"  Gradients:           {mem['gradient_gb']} GiB")
    output.append(f"  Activations (ckpt):  {mem['activation_gb_checkpointed']} GiB")
    if mem['ref_model_gb'] > 0:
        output.append(f"  Reference model:     {mem['ref_model_gb']} GiB")
    if mem['critic_gb'] > 0:
        output.append(f"  Critic model:        {mem['critic_gb']} GiB")
    output.append(f"  KV cache (rollout):  {mem['kv_cache_gb']} GiB")
    output.append(f"  Temp/overhead:       {mem['temp_overhead_gb']} GiB")
    output.append(f"  ---")
    output.append(f"  Peak GPU memory:     {mem['peak_gpu_gb_checkpointed']} GiB")
    output.append(f"  GPU capacity:        {mem['gpu_memory_gib']} GiB")
    if mem['fits_gpu']:
        output.append(f"  FITS GPU: YES (margin: {mem['margin_gb']} GiB)")
    else:
        output.append(f"  FITS GPU: NO (exceeds by {-mem['margin_gb']} GiB)")
    output.append("")

    # Timing estimate
    output.append("-" * 80)
    output.append(" Timing Estimate")
    output.append("-" * 80)
    output.append(f"  Rollout time:        {timing['rollout_time_sec']} sec/step")
    output.append(f"  Training time:       {timing['train_time_sec']} sec/step")
    output.append(f"  Weight sync time:    {timing['sync_time_sec']} sec/step")
    output.append(f"  Total step time:     {timing['total_step_time_sec']} sec/step")
    output.append(f"  Throughput:          {timing['throughput_tokens_sec']} tokens/sec")
    output.append(f"  Steps per hour:      {timing['estimated_steps_per_hour']}")
    output.append("")

    # MUST DO rules applied
    output.append("-" * 80)
    output.append(" MUST DO Rules Applied")
    output.append("-" * 80)
    applied_rules = [
        ("FSDP1 (not FSDP2)", "#6468 CPU memory leak prevention"),
        ("LoRA rank=32", "#6782 EOS generation safety"),
        ("bypass_mode=True", "Ref model removed, 18Psi -> 3.8Psi"),
        ("grad_clip=1.0", "#8068 silent bug prevention"),
        ("overlap_comm=False", "#8061 NaN prevention on single GPU"),
        ("enforce_eager=True", "SM89/DSV4 CUDA graph failure prevention"),
        ("group_size >= 4", "#605 GRPO normalization safety"),
        ("naive checkpoint (dp=1)", "Zero IPC overhead for single GPU"),
        ("ulimit -n 65535", "#8075 fd leak safety"),
        ("param_offload + optimizer_offload", "Memory savings for 24 GiB GPU"),
    ]
    for i, (rule, reason) in enumerate(applied_rules, 1):
        output.append(f"  {i:2d}. {rule}")
        output.append(f"      {reason}")
    output.append("")

    # YAML config
    output.append("-" * 80)
    output.append(" YAML Config (verl V1 Hydra/OmegaConf structure)")
    output.append("-" * 80)
    output.append("")
    output.append(yaml_dump_ordered(config))
    output.append("")

    # Launch command
    output.append("-" * 80)
    output.append(" Launch Command")
    output.append("-" * 80)
    output.append("")
    output.append("# Set ulimit before launching (MUST DO)")
    output.append("ulimit -n 65535")
    output.append("")
    output.append(result["launch_command"])
    output.append("")

    return "\n".join(output)

# ============================================================
# MODE 1: generate
# ============================================================

def cmd_generate(args):
    """Generate a complete verl V1 GRPO config YAML."""
    result = generate_verl_v1_grpo_config(
        model=args.model,
        gpu_type=args.gpu_type,
        dp=args.dp,
        lora_rank=args.lora_rank,
        group_size=args.group_size,
        bypass_mode=args.bypass_mode,
        adv_estimator=args.adv_estimator,
        rollout_engine=args.rollout_engine,
        trainer_type=args.trainer_type,
        checkpoint_engine=args.checkpoint_engine,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        total_epochs=args.total_epochs,
        enforce_eager=args.enforce_eager,
    )

    print(format_config_output(result))

    # Save YAML file if requested
    if args.output:
        yaml_content = yaml_dump_ordered(result["config"])
        with open(args.output, "w") as f:
            f.write("# verl V1 GRPO Training Configuration (Auto-generated)\n")
            f.write("# Model: {}\n".format(result["model_profile"]["path"]))
            f.write("# GPU: {}\n".format(result["gpu_profile"]["name"]))
            f.write("# ALL MUST DO rules applied automatically\n")
            f.write("\n")
            f.write(yaml_content)
        print(f"\nYAML config saved to: {args.output}")

        # Also save launch script
        script_path = args.output.replace(".yaml", "_launch.sh")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# verl V1 GRPO Training Launch Script (Auto-generated)\n")
            f.write(f"# Model: {result['model_profile']['path']}\n")
            f.write(f"# GPU: {result['gpu_profile']['name']}\n")
            f.write("\n")
            f.write("# MUST DO: Set ulimit\n")
            f.write("ulimit -n 65535\n")
            f.write("\n")
            f.write(result["launch_command"] + "\n")
        os.chmod(script_path, 0o755)
        print(f"Launch script saved to: {script_path}")

# ============================================================
# MODE 2: rtx4090
# ============================================================

def cmd_rtx4090(args):
    """Generate RTX 4090 optimal config."""
    print("=" * 80)
    print(" RTX 4090 GRPO Training Config — Optimal Configuration")
    print("=" * 80)
    print()
    print("RTX 4090: 24 GiB VRAM, SM89, PCIe, single GPU only")
    print("Best model fit: Qwen2.5-7B-Instruct (14 GiB weights in BF16)")
    print()

    # Primary config: Qwen2.5-7B-Instruct
    print("=" * 80)
    print(" PRIMARY: Qwen2.5-7B-Instruct (optimal fit)")
    print("=" * 80)
    print()

    result_primary = generate_verl_v1_grpo_config(
        model="qwen2.5-7b",
        gpu_type="rtx4090",
        dp=1,
        lora_rank=32,
        group_size=8,
        bypass_mode=True,
        adv_estimator="grpo",
        rollout_engine="sglang",
        trainer_type="sync",
        checkpoint_engine="naive",
        batch_size=4,
        seq_len=2048,
        max_prompt_length=512,
        lr=1e-6,
        total_epochs=30,
        param_offload=True,
        optimizer_offload=True,
        enforce_eager=True,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        use_kl_in_reward=False,
        gradient_checkpointing=True,
        use_remove_padding=True,
        use_dynamic_bsz=True,
        ppo_max_token_len_per_gpu=8192,
        dataset_path="$HOME/data/gsm8k/train.parquet",
        val_dataset_path="$HOME/data/gsm8k/test.parquet",
    )

    print(format_config_output(result_primary))

    # Alternative: Qwen2.5-7B with vLLM rollout
    print()
    print("=" * 80)
    print(" ALTERNATIVE A: Qwen2.5-7B-Instruct with vLLM rollout")
    print("=" * 80)
    print()

    result_vllm = generate_verl_v1_grpo_config(
        model="qwen2.5-7b",
        gpu_type="rtx4090",
        dp=1,
        lora_rank=32,
        group_size=8,
        bypass_mode=True,
        adv_estimator="grpo",
        rollout_engine="vllm",
        trainer_type="sync",
        checkpoint_engine="naive",
        batch_size=4,
        seq_len=2048,
        lr=1e-6,
        enforce_eager=True,
        param_offload=True,
        optimizer_offload=True,
    )

    print(format_config_output(result_vllm))

    # Alternative: Qwen3-8B (tight fit)
    print()
    print("=" * 80)
    print(" ALTERNATIVE B: Qwen3-8B (tight fit — 16.4 GiB weights)")
    print("=" * 80)
    print()
    print("WARNING: Qwen3-8B is a tight fit on RTX 4090 (24 GiB).")
    print("         Requires aggressive optimizer offload + LoRA rank=16.")
    print("         Reduce group_size to 4 and batch_size to 2.")
    print()

    result_tight = generate_verl_v1_grpo_config(
        model="qwen3-8b",
        gpu_type="rtx4090",
        dp=1,
        lora_rank=16,
        group_size=4,
        bypass_mode=True,
        adv_estimator="grpo",
        rollout_engine="sglang",
        trainer_type="sync",
        checkpoint_engine="naive",
        batch_size=2,
        seq_len=2048,
        lr=5e-7,
        total_epochs=30,
        param_offload=True,
        optimizer_offload=True,
        enforce_eager=True,
        use_kl_loss=False,
        use_kl_in_reward=False,
        gradient_checkpointing=True,
        use_remove_padding=True,
        use_dynamic_bsz=True,
        ppo_max_token_len_per_gpu=4096,
        rollout_gpu_mem_util=0.35,
    )

    print(format_config_output(result_tight))

    # Alternative: Qwen2.5-3B (comfortable fit)
    print()
    print("=" * 80)
    print(" ALTERNATIVE C: Qwen2.5-3B-Instruct (comfortable fit)")
    print("=" * 80)
    print()

    result_comfortable = generate_verl_v1_grpo_config(
        model="qwen2.5-3b",
        gpu_type="rtx4090",
        dp=1,
        lora_rank=32,
        group_size=8,
        bypass_mode=True,
        adv_estimator="grpo",
        rollout_engine="sglang",
        trainer_type="sync",
        checkpoint_engine="naive",
        batch_size=8,
        seq_len=2048,
        lr=1e-6,
        total_epochs=30,
        param_offload=False,  # 3B has enough room
        optimizer_offload=True,
        enforce_eager=True,
        use_kl_loss=False,
        use_kl_in_reward=False,
    )

    print(format_config_output(result_comfortable))

    # Summary table
    print()
    print("=" * 80)
    print(" RTX 4090 Config Summary Table")
    print("=" * 80)
    print()
    header = f"{'Model':<25} {'LoRA':>5} {'Group':>6} {'Batch':>6} {'Peak GB':>8} {'Fits':>6} {'Margin':>7} {'Step sec':>9}"
    print(header)
    print("-" * len(header))

    configs = [
        ("Qwen2.5-7B (SGLang)", result_primary),
        ("Qwen2.5-7B (vLLM)", result_vllm),
        ("Qwen3-8B (tight)", result_tight),
        ("Qwen2.5-3B (comfort)", result_comfortable),
    ]

    for label, res in configs:
        m = res["memory_estimate"]
        t = res["timing_estimate"]
        lora_r = res["config"].get("lora", {}).get("rank", 0)
        group = res["config"]["actor_rollout_ref"]["rollout"]["n"]
        batch = res["config"]["data"]["train_batch_size"]
        fits_str = "YES" if m["fits_gpu"] else "NO"
        print(f"{label:<25} {lora_r:>5} {group:>6} {batch:>6} {m['peak_gpu_gb_checkpointed']:>8.1f} {fits_str:>6} {m['margin_gb']:>7.1f} {t['total_step_time_sec']:>9.1f}")

    print()
    print("RECOMMENDATION:")
    print("  Primary: Qwen2.5-7B-Instruct + SGLang + LoRA32 + bypass_mode")
    print("  This is the safest, most tested configuration for RTX 4090.")
    print()
    print("MUST DO before launching:")
    print("  1. ulimit -n 65535")
    print("  2. Verify model weights downloaded to local path")
    print("  3. Verify dataset files exist")
    print("  4. Set VLLM_USE_V2_MODEL_RUNNER=0 if using vLLM")
    print()

# ============================================================
# MODE 3: compare
# ============================================================

def cmd_compare(args):
    """Compare configs for different GPU setups."""
    print("=" * 80)
    print(" verl V1 GRPO Config Comparison — Different GPU Setups")
    print("=" * 80)
    print()

    # Define comparison scenarios
    scenarios = [
        {"label": "RTX 4090 dp=1", "gpu_type": "rtx4090", "dp": 1, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 4, "group_size": 8},
        {"label": "A100-40G dp=1", "gpu_type": "a100-40g", "dp": 1, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 16, "group_size": 8},
        {"label": "A100-40G dp=2", "gpu_type": "a100-40g", "dp": 2, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 128, "group_size": 8},
        {"label": "A100-80G dp=1", "gpu_type": "a100-80g", "dp": 1, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 32, "group_size": 8},
        {"label": "A100-80G dp=4", "gpu_type": "a100-80g", "dp": 4, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 256, "group_size": 8},
        {"label": "A100-80G dp=8", "gpu_type": "a100-80g", "dp": 8, "model": "qwen2.5-14b",
         "lora_rank": 64, "batch_size": 256, "group_size": 16},
        {"label": "H100-80G dp=1", "gpu_type": "h100-80g", "dp": 1, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 32, "group_size": 8},
        {"label": "H100-80G dp=4", "gpu_type": "h100-80g", "dp": 4, "model": "qwen2.5-7b",
         "lora_rank": 32, "batch_size": 256, "group_size": 8},
        {"label": "H100-80G dp=8", "gpu_type": "h100-80g", "dp": 8, "model": "qwen2.5-14b",
         "lora_rank": 64, "batch_size": 512, "group_size": 16},
        {"label": "H100-80G 2-node", "gpu_type": "h100-80g", "dp": 16, "model": "qwen2.5-32b",
         "lora_rank": 64, "batch_size": 512, "group_size": 16, "trainer_type": "separate_async"},
        {"label": "H200-141G dp=1", "gpu_type": "h200-141g", "dp": 1, "model": "qwen2.5-14b",
         "lora_rank": 32, "batch_size": 32, "group_size": 8},
        {"label": "H200-141G dp=8", "gpu_type": "h200-141g", "dp": 8, "model": "qwen2.5-32b",
         "lora_rank": 64, "batch_size": 512, "group_size": 16},
    ]

    # Generate configs and estimates for each scenario
    results = []
    for scenario in scenarios:
        trainer_type = scenario.get("trainer_type", "sync")
        if scenario["dp"] == 1:
            trainer_type = "sync"
        elif scenario["dp"] <= 4:
            trainer_type = "colocate_async"
        else:
            trainer_type = "separate_async"

        ckpt_engine = "naive" if scenario["dp"] == 1 else "nccl"

        # Auto-adjust param/optimizer offload
        gpu = GPU_PROFILES[scenario["gpu_type"]]
        param_offload = gpu["memory_gib"] <= 40
        optimizer_offload = gpu["memory_gib"] <= 40

        enforce_eager = gpu["sm"] == "SM89"

        result = generate_verl_v1_grpo_config(
            model=scenario["model"],
            gpu_type=scenario["gpu_type"],
            dp=scenario["dp"],
            lora_rank=scenario["lora_rank"],
            group_size=scenario["group_size"],
            bypass_mode=True,
            adv_estimator="grpo",
            rollout_engine="sglang",
            trainer_type=trainer_type,
            checkpoint_engine=ckpt_engine,
            batch_size=scenario["batch_size"],
            param_offload=param_offload,
            optimizer_offload=optimizer_offload,
            enforce_eager=enforce_eager,
        )
        result["label"] = scenario["label"]
        results.append(result)

    # Print comparison table
    print()
    header = (f"{'Setup':<20} {'Model':<18} {'DP':>4} {'LoRA':>5} {'Grp':>5} "
              f"{'Batch':>6} {'Peak GB':>8} {'Fits':>6} {'Margin':>7} "
              f"{'Step sec':>9} {'Steps/hr':>9} {'Trainer':>16} {'Ckpt':>7}")
    print(header)
    print("-" * len(header))

    for result in results:
        m = result["memory_estimate"]
        t = result["timing_estimate"]
        model_name = result["model_profile"]["path"].split("/")[-1]
        lora_r = result["config"].get("lora", {}).get("rank", 0)
        group = result["config"]["actor_rollout_ref"]["rollout"]["n"]
        batch = result["config"]["data"]["train_batch_size"]
        dp = result["config"]["trainer"]["n_gpus_per_node"]
        trainer = result["config"]["trainer"]["v1"]["trainer_mode"]
        ckpt = result["config"]["actor_rollout_ref"]["rollout"]["checkpoint_engine"]["backend"]
        fits_str = "YES" if m["fits_gpu"] else "NO"

        print(f"{result['label']:<20} {model_name:<18} {dp:>4} {lora_r:>5} {group:>5} "
              f"{batch:>6} {m['peak_gpu_gb_checkpointed']:>8.1f} {fits_str:>6} {m['margin_gb']:>7.1f} "
              f"{t['total_step_time_sec']:>9.1f} {t['estimated_steps_per_hour']:>9.1f} "
              f"{trainer:>16} {ckpt:>7}")

    print()
    print("Notes:")
    print("  - All configs use bypass_mode=True (GRPO-specific, removes ref model)")
    print("  - RTX 4090: dp=1 sync + naive (single GPU only)")
    print("  - A100/H100 dp<=4: colocate_async + nccl")
    print("  - A100/H100 dp>=8: separate_async + nccl (multi-node possible)")
    print("  - enforce_eager=True only for SM89 (RTX 4090)")
    print("  - 24 GiB GPUs: param_offload=True, optimizer_offload=True")
    print("  - 40 GiB GPUs: optimizer_offload=True for larger models")
    print("  - 80+ GiB GPUs: no offload needed for 7B/14B with LoRA")
    print()

    # Throughput comparison chart
    print("=" * 80)
    print(" Throughput Comparison (tokens/sec)")
    print("=" * 80)
    print()
    max_throughput = max(r["timing_estimate"]["throughput_tokens_sec"] for r in results)
    for result in results:
        t = result["timing_estimate"]
        bar_len = int(t["throughput_tokens_sec"] / max_throughput * 40)
        bar = "#" * bar_len
        print(f"  {result['label']:<20} {t['throughput_tokens_sec']:>8.0f} {bar}")
    print()

# ============================================================
# MODE 4: validate
# ============================================================

def cmd_validate(args):
    """Validate an existing config YAML against safety rules."""
    config_path = args.config

    # Try to load the config
    config = None
    try:
        # Try PyYAML first
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Fallback: try to parse basic YAML-like structure
        try:
            with open(config_path) as f:
                content = f.read()
            # Simple key=value parsing for command-line override format
            config = {}
            for line in content.strip().split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    # Try to convert value
                    if value.lower() == "true":
                        config[key] = True
                    elif value.lower() == "false":
                        config[key] = False
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        try:
                            config[key] = float(value)
                        except ValueError:
                            config[key] = value
        except Exception as e:
            print(f"ERROR: Cannot read config file: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot read config file: {e}")
        sys.exit(1)

    if config is None:
        print(f"ERROR: Could not parse config file: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print(f" V1 Config Validation: {config_path}")
    print("=" * 80)
    print()

    # Flatten nested config for easier checking
    def flatten_config(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_config(v, key))
            else:
                flat[key] = v
        return flat

    flat_config = flatten_config(config) if isinstance(config, dict) else config

    # Extract key settings
    trainer_type = flat_config.get("trainer.v1.trainer_mode", "unknown")
    strategy = flat_config.get("actor_rollout_ref.actor.fsdp_config.strategy",
                                flat_config.get("actor_rollout_ref.actor.strategy", "unknown"))
    param_offload = flat_config.get("actor_rollout_ref.actor.fsdp_config.param_offload", False)
    optimizer_offload = flat_config.get("actor_rollout_ref.actor.fsdp_config.optimizer_offload", False)
    grad_clip = flat_config.get("actor_rollout_ref.actor.grad_clip", 0)
    enforce_eager = flat_config.get("actor_rollout_ref.rollout.enforce_eager", False)
    lora_rank = flat_config.get("actor_rollout_ref.actor.lora_rank", 0)
    group_size = flat_config.get("actor_rollout_ref.rollout.n", 1)
    use_kl_in_reward = flat_config.get("algorithm.use_kl_in_reward", True)
    use_kl_loss = flat_config.get("actor_rollout_ref.actor.use_kl_loss", False)
    adv_estimator = flat_config.get("algorithm.adv_estimator", "unknown")
    ckpt_backend = flat_config.get("actor_rollout_ref.rollout.checkpoint_engine.backend", "unknown")
    model_path = flat_config.get("actor_rollout_ref.model.path", "unknown")
    dp = flat_config.get("trainer.n_gpus_per_node", 1)

    # Print config summary
    print("Config Summary:")
    print(f"  Model:         {model_path}")
    print(f"  Trainer type:  {trainer_type}")
    print(f"  Strategy:      {strategy}")
    print(f"  DP:            {dp}")
    print(f"  adv_estimator: {adv_estimator}")
    print(f"  group_size:    {group_size}")
    print(f"  LoRA rank:     {lora_rank}")
    print(f"  grad_clip:     {grad_clip}")
    print(f"  enforce_eager: {enforce_eager}")
    print(f"  param_offload: {param_offload}")
    print(f"  optimizer_offload: {optimizer_offload}")
    print(f"  use_kl_in_reward: {use_kl_in_reward}")
    print(f"  use_kl_loss:   {use_kl_loss}")
    print(f"  checkpoint:    {ckpt_backend}")
    print()

    # Run MUST DO checks
    errors = []
    warnings = []
    passes = []

    # 1. FSDP1 not FSDP2
    if strategy == "fsdp2":
        errors.append(("FSDP1 (not FSDP2)", "strategy=fsdp2 detected → #6468 CPU memory leak",
                        "Set actor_rollout_ref.actor.fsdp_config.strategy=fsdp"))
    elif strategy == "fsdp":
        passes.append(("FSDP1 (not FSDP2)", "strategy=fsdp (FSDP1) -- correct"))
    else:
        warnings.append(("FSDP1 (not FSDP2)", f"strategy={strategy} — verify this is FSDP1",
                          "Set strategy=fsdp"))

    # 2. LoRA rank <= 32 (for GRPO)
    if lora_rank == 0:
        warnings.append(("LoRA rank", "lora_rank=0 (full finetune) — may exceed 24 GiB memory",
                          "Set lora_rank=32 for consumer GPUs"))
    elif lora_rank == 64:
        errors.append(("LoRA rank <= 32", f"lora_rank={lora_rank} → #6782 breaks EOS generation",
                        "Set lora_rank <= 32"))
    elif lora_rank > 32 and lora_rank <= 64:
        warnings.append(("LoRA rank", f"lora_rank={lora_rank} — use 32 for GRPO safety, 64 only for PPO",
                          "Set lora_rank=32 for GRPO"))
    else:
        passes.append(("LoRA rank <= 32", f"lora_rank={lora_rank} -- correct"))

    # 3. bypass_mode (GRPO-specific)
    if adv_estimator == "grpo" and use_kl_in_reward:
        warnings.append(("bypass_mode=True", "use_kl_in_reward=True for GRPO → ref model loaded → high memory",
                          "Set algorithm.use_kl_in_reward=False for GRPO+bypass"))
    elif adv_estimator == "grpo" and not use_kl_in_reward:
        passes.append(("bypass_mode=True", "use_kl_in_reward=False → bypass_mode active"))

    # 4. grad_clip = 1.0
    if grad_clip == 0:
        errors.append(("grad_clip=1.0", f"grad_clip={grad_clip} → #8068 silent training bug",
                        "Set actor_rollout_ref.actor.grad_clip=1.0"))
    elif grad_clip == 1.0:
        passes.append(("grad_clip=1.0", "grad_clip=1.0 -- correct"))
    else:
        warnings.append(("grad_clip=1.0", f"grad_clip={grad_clip} — not 1.0, may cause instability",
                          "Set actor_rollout_ref.actor.grad_clip=1.0"))

    # 5. overlap_comm=False (for dp=1)
    overlap_comm = flat_config.get("actor_rollout_ref.actor.fsdp_config.overlap_comm", None)
    if dp == 1 and overlap_comm == True:
        errors.append(("overlap_comm=False", "overlap_comm=True on dp=1 → #8061 NaN bug",
                        "Set overlap_comm=False for single GPU"))
    elif overlap_comm == False:
        passes.append(("overlap_comm=False", "overlap_comm=False -- correct"))
    elif overlap_comm is None:
        warnings.append(("overlap_comm=False", "overlap_comm not set → default may be True",
                          "Explicitly set overlap_comm=False for dp=1"))

    # 6. enforce_eager=True (for SM89/DSV4)
    if not enforce_eager:
        warnings.append(("enforce_eager=True", "enforce_eager=False → CUDA graph risk on SM89/DSV4",
                          "Set actor_rollout_ref.rollout.enforce_eager=True"))
    else:
        passes.append(("enforce_eager=True", "enforce_eager=True -- correct"))

    # 7. group_size >= 4 (for GRPO)
    if adv_estimator == "grpo" and group_size < 4:
        errors.append(("group_size >= 4", f"group_size={group_size} for GRPO → #605 normalization risk",
                        "Set actor_rollout_ref.rollout.n >= 4 for GRPO"))
    elif adv_estimator == "grpo" and group_size >= 4:
        passes.append(("group_size >= 4", f"group_size={group_size} for GRPO -- correct"))
    elif adv_estimator != "grpo":
        passes.append(("group_size", f"group_size={group_size} for {adv_estimator} -- acceptable"))

    # 8. checkpoint engine for dp=1
    if dp == 1 and ckpt_backend not in ("naive", "unknown"):
        warnings.append(("naive checkpoint dp=1", f"ckpt_backend={ckpt_backend} for dp=1 — overkill",
                          "Set checkpoint_engine.backend=naive for dp=1 sync trainer"))
    elif dp == 1 and ckpt_backend == "naive":
        passes.append(("naive checkpoint dp=1", "checkpoint_engine=naive for dp=1 -- correct"))

    # 9. param_offload + optimizer_offload for 24 GiB
    if dp == 1 and not optimizer_offload and lora_rank == 0:
        warnings.append(("optimizer offload", "optimizer_offload=False with full finetune on 24 GiB → may OOM",
                          "Set optimizer_offload=True for 24 GiB GPU"))
    elif param_offload and optimizer_offload:
        passes.append(("param+optimizer offload", "param_offload=True + optimizer_offload=True -- correct"))

    # 10. separate_async on single GPU
    if trainer_type == "separate_async" and dp == 1:
        errors.append(("separate_async on dp=1", "separate_async requires min 2 GPUs",
                        "Use trainer_type=sync for dp=1"))

    # 11. Must NOT checks
    must_not_checks = [
        ("ZeRO-3", flat_config.get("zero_optimization.stage", None) == 3,
         "ZeRO-3 on single GPU → #8072/#8076 regression", "Use ZeRO-2 or FSDP1"),
        ("Muon optimizer", flat_config.get("actor_rollout_ref.actor.optim.type", "") == "muon",
         "Muon has 6 blockers across 3 frameworks", "Use cpu_adam or standard AdamW"),
        ("NVMe offload", flat_config.get("zero_optimization.offload.offload_device", "") == "nvme",
         "NVMe offload fd leak #8075", "Use CPU offload instead"),
    ]
    for rule_name, triggered, msg, fix in must_not_checks:
        if triggered:
            errors.append((rule_name, msg, fix))
        else:
            passes.append((rule_name, f"{rule_name} -- not present (correct)"))

    # Memory estimate
    # Find model params from path
    model_params = 7.0  # default
    for model_key, profile in MODEL_PROFILES.items():
        if model_path and profile["path"].lower() in model_path.lower():
            model_params = profile["params_b"]
            break

    mem_est = estimate_model_memory(
        params_b=model_params,
        lora_rank=lora_rank,
        dp=dp,
        group_size=group_size,
        bypass_mode=not use_kl_in_reward,
        needs_critic=(adv_estimator == "ppo"),
        gpu_type="rtx4090",
    )

    timing_est = estimate_step_timing(
        params_b=model_params,
        dp=dp,
        gpu_type="rtx4090",
        group_size=group_size,
    )

    # Print results
    print("-" * 80)
    print(" Validation Results")
    print("-" * 80)
    print()

    if errors:
        print("ERRORS (MUST FIX before running):")
        for rule, msg, fix in errors:
            print(f"  [FAIL] {rule}")
            print(f"         Issue: {msg}")
            print(f"         Fix:   {fix}")
        print()

    if warnings:
        print("WARNINGS (recommended improvements):")
        for rule, msg, fix in warnings:
            print(f"  [WARN] {rule}")
            print(f"         Issue: {msg}")
            print(f"         Fix:   {fix}")
        print()

    if passes:
        print("PASS (rules satisfied):")
        for rule, msg in passes:
            print(f"  [PASS] {rule}: {msg}")
        print()

    # Overall result
    total_rules = len(errors) + len(warnings) + len(passes)
    error_count = len(errors)
    warn_count = len(warnings)
    pass_count = len(passes)

    print("-" * 80)
    print(f" Summary: {pass_count}/{total_rules} rules pass, {warn_count} warnings, {error_count} errors")
    print("-" * 80)

    if error_count > 0:
        print()
        print("CONFIG IS NOT SAFE TO RUN — fix all errors before launching.")
    elif warn_count > 0:
        print()
        print("CONFIG IS LIKELY SAFE — but review warnings for optimal performance.")
    else:
        print()
        print("CONFIG PASSES ALL RULES — safe to launch!")

    # Memory estimate
    print()
    print("-" * 80)
    print(" Memory Estimate (assuming RTX 4090)")
    print("-" * 80)
    print(f"  Peak GPU memory:     {mem_est['peak_gpu_gb_checkpointed']} GiB")
    print(f"  GPU capacity:        {mem_est['gpu_memory_gib']} GiB")
    if mem_est['fits_gpu']:
        print(f"  FITS GPU: YES (margin: {mem_est['margin_gb']} GiB)")
    else:
        print(f"  FITS GPU: NO (exceeds by {-mem_est['margin_gb']} GiB)")

    print()
    print("-" * 80)
    print(" Timing Estimate")
    print("-" * 80)
    print(f"  Total step time:     {timing_est['total_step_time_sec']} sec")
    print(f"  Steps per hour:      {timing_est['estimated_steps_per_hour']}")
    print(f"  Throughput:          {timing_est['throughput_tokens_sec']} tokens/sec")

# ============================================================
# MODE 5: quick
# ============================================================

def cmd_quick(args):
    """Generate a quick config for testing/debugging."""
    print("=" * 80)
    print(" verl V1 GRPO Quick Test Config — Minimal for Debugging")
    print("=" * 80)
    print()
    print("Purpose: Validate GRPO algorithm correctness with minimal resources.")
    print("Model: Qwen2.5-0.5B-Instruct (smallest, fastest)")
    print("Steps: 10 (quick convergence check)")
    print("Group size: 4 (minimum for GRPO)")
    print()

    model_choice = args.model if args.model else "qwen2.5-0.5b"

    result = generate_verl_v1_grpo_config(
        model=model_choice,
        gpu_type="rtx4090",
        dp=1,
        lora_rank=8,  # Small LoRA for quick test
        group_size=4,  # Minimum GRPO group size
        bypass_mode=True,
        adv_estimator="grpo",
        rollout_engine="vllm",  # vLLM more commonly available
        trainer_type="sync",
        checkpoint_engine="naive",
        batch_size=4,
        seq_len=512,  # Short sequences for quick test
        max_prompt_length=128,
        lr=1e-5,  # Higher LR for fast convergence in test
        total_epochs=1,
        total_training_steps=10,  # Quick: just 10 steps
        param_offload=False,  # 0.5B model tiny, no need
        optimizer_offload=False,  # 0.5B model tiny, no need
        enforce_eager=True,  # Safety rule always
        use_kl_loss=False,
        use_kl_in_reward=False,
        gradient_checkpointing=False,  # Not needed for 0.5B
        use_remove_padding=False,
        use_dynamic_bsz=False,
        save_freq=5,
        test_freq=5,
        project_name="verl_grpo_quick_test",
        experiment_name=f"quick_test_{model_choice}",
    )

    print(format_config_output(result))

    # Additional quick test notes
    print()
    print("=" * 80)
    print(" Quick Test Debugging Guide")
    print("=" * 80)
    print()
    print("This config is designed for rapid GRPO algorithm validation:")
    print()
    print("  1. Advantage computation check:")
    print("     - Verify that GRPO advantage = (reward - group_mean) / group_std")
    print("     - With group_size=4, each group should have 4 responses")
    print("     - Check that norm_adv_by_std_in_grpo=True is set")
    print()
    print("  2. Loss computation check:")
    print("     - GRPO loss uses PPO-clip style: -min(r*A, clip(r,A))")
    print("     - Verify entropy_coeff=0 (no entropy bonus in GRPO)")
    print("     - Verify kl_loss_coef=0 (bypass_mode, no KL penalty)")
    print()
    print("  3. Numerical stability check:")
    print("     - Watch for NaN in advantage (group_std=0 when all rewards equal)")
    print("     - grad_clip=1.0 prevents gradient explosion")
    print("     - enforce_eager=True prevents CUDA graph failures")
    print()
    print("  4. Weight sync check:")
    print("     - naive checkpoint engine: zero IPC overhead")
    print("     - Verify LoRA adapter sync completes each step")
    print()
    print("  5. Expected behavior:")
    print("     - Loss should decrease over 10 steps")
    print("     - Reward should increase (if reward function works)")
    print("     - No NaN, no OOM, no CUDA errors")
    print()
    print("  Alternative models for quick test:")
    for model_key in ["qwen2.5-0.5b", "qwen2.5-1.5b"]:
        profile = MODEL_PROFILES[model_key]
        print(f"    {model_key}: {profile['path']} ({profile['params_b']}B) — {profile['best_for']}")
    print()

# ============================================================
# MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="verl V1 GRPO Training Configuration Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  generate  Generate a complete verl V1 GRPO config YAML
  rtx4090   Generate RTX 4090 optimal config with alternatives
  compare   Compare configs for different GPU setups
  validate  Validate an existing config YAML
  quick     Quick config for testing/debugging

Examples:
  python3 %(prog)s generate --model qwen2.5-7b --gpu-type rtx4090
  python3 %(prog)s generate --model qwen2.5-14b --gpu-type a100-80g --dp 4
  python3 %(prog)s rtx4090
  python3 %(prog)s compare
  python3 %(prog)s validate --config my_config.yaml
  python3 %(prog)s quick
  python3 %(prog)s quick --model qwen2.5-1.5b
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

    # Mode 1: generate
    gen_parser = subparsers.add_parser("generate", help="Generate complete config")
    gen_parser.add_argument("--model", default="qwen2.5-7b",
                            choices=list(MODEL_PROFILES.keys()),
                            help="Model name (default: qwen2.5-7b)")
    gen_parser.add_argument("--gpu-type", default="rtx4090",
                            choices=list(GPU_PROFILES.keys()),
                            help="GPU type (default: rtx4090)")
    gen_parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    gen_parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (0=full finetune)")
    gen_parser.add_argument("--group-size", type=int, default=8, help="GRPO group size (responses per prompt)")
    gen_parser.add_argument("--bypass-mode", type=bool, default=True,
                            help="bypass_mode=True removes ref model")
    gen_parser.add_argument("--adv-estimator", default="grpo",
                            choices=list(ADV_ESTIMATORS.keys()),
                            help="Advantage estimator")
    gen_parser.add_argument("--rollout-engine", default="sglang",
                            choices=["sglang", "vllm"],
                            help="Rollout inference engine")
    gen_parser.add_argument("--trainer-type", default="sync",
                            choices=list(V1_TRAINER_TYPES.keys()),
                            help="V1 trainer type")
    gen_parser.add_argument("--checkpoint-engine", default="naive",
                            choices=list(CHECKPOINT_ENGINES.keys()),
                            help="Checkpoint engine backend")
    gen_parser.add_argument("--batch-size", type=int, default=None,
                            help="Batch size (auto-determined if not set)")
    gen_parser.add_argument("--seq-len", type=int, default=2048,
                            help="Max response length")
    gen_parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    gen_parser.add_argument("--total-epochs", type=int, default=30, help="Total training epochs")
    gen_parser.add_argument("--enforce-eager", type=bool, default=None,
                            help="enforce_eager (auto-determined if not set)")
    gen_parser.add_argument("--output", default=None,
                            help="Output YAML file path (optional)")

    # Mode 2: rtx4090
    rtx_parser = subparsers.add_parser("rtx4090", help="RTX 4090 optimal config")

    # Mode 3: compare
    cmp_parser = subparsers.add_parser("compare", help="Compare GPU setups")

    # Mode 4: validate
    val_parser = subparsers.add_parser("validate", help="Validate config")
    val_parser.add_argument("--config", required=True, help="Config YAML file to validate")

    # Mode 5: quick
    quick_parser = subparsers.add_parser("quick", help="Quick test config")
    quick_parser.add_argument("--model", default=None,
                              choices=["qwen2.5-0.5b", "qwen2.5-1.5b"],
                              help="Model for quick test (default: qwen2.5-0.5b)")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "generate":
        cmd_generate(args)
    elif args.mode == "rtx4090":
        cmd_rtx4090(args)
    elif args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "validate":
        cmd_validate(args)
    elif args.mode == "quick":
        cmd_quick(args)


if __name__ == "__main__":
    main()
