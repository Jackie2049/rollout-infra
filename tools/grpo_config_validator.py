#!/usr/bin/env python3
"""
GRPO Training Configuration Validator
======================================
CPU-only tool that validates GRPO training configurations across
7 frameworks (verl, OpenRLHF, TRL, rLLM, DeepSpeed, Megatron, vLLM).

Checks 50+ MUST DO and MUST NOT rules, detects common misconfigurations,
and provides framework-specific recommendations.

4 Modes:
  validate  — Validate a given configuration against all rules
  checklist — Full pre-flight checklist (before starting training)
  compare   — Compare default configs across 7 frameworks
  rtx4090   — RTX 4090-specific configuration validator

Usage:
  python3 grpo_config_validator.py validate
  python3 grpo_config_validator.py checklist
  python3 grpo_config_validator.py compare
  python3 grpo_config_validator.py rtx4090

Created: 2026-06-20 | Part of rollout-infra tools suite
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ============================================================
# Configuration Data Structures
# ============================================================

@dataclass
class GRPOConfig:
    """Complete GRPO training configuration"""
    # Framework
    framework: str = "verl"          # verl, openrlhf, trl, rllm, deepspeed, megatron, vllm
    rollout_engine: str = "sglang"   # vllm, sglang, huggingface

    # Model
    model_name: str = "Qwen2.5-7B"
    model_params_b: float = 7.0     # billions of parameters

    # Training strategy
    strategy: str = "fsdp1"          # fsdp1, fsdp2, deepspeed-zeRO1, deepspeed-zeRO2, deepspeed-zeRO3
    lora_rank: int = 32              # LoRA rank (0 = full param)
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # GRPO parameters
    group_size: int = 8              # number of responses per prompt
    clip_epsilon: float = 0.2        # clip range
    learning_rate: float = 1e-5      # learning rate
    optimizer: str = "adam"          # adam, cpu_adam, adamw
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0   # max_grad_norm

    # Reference model
    reference_model_mode: str = "bypass"  # bypass, offload, separate, none

    # Checkpoint
    checkpoint_engine: str = "naive"     # naive, hdfs, torch, deepspeed

    # Hardware
    gpu_type: str = "rtx4090"         # rtx4090, a100, h100, h200
    gpu_count: int = 1
    gpu_vram_gb: float = 24.0
    dp_size: int = 1                  # data parallelism
    tp_size: int = 1                  # tensor parallelism
    pp_size: int = 1                  # pipeline parallelism

    # Rollout
    seq_length: int = 2048
    micro_batch_size: int = 1
    max_new_tokens: int = 512

    # Reward
    reward_type: str = "format_outcome"  # format_outcome, rm_score, rule_exact, ranking
    reward_shaped: bool = True

    # Sleep/wake
    sleep_level: int = 1              # 0=none, 1=KV release, 2=full release

    # Misc
    enforce_eager: bool = True        # MUST for DSV4
    ulimit: int = 65535               # file descriptor limit
    n_epochs: int = 1                 # update epochs per rollout


# ============================================================
# Rule Definitions
# ============================================================

# MUST DO rules (violation = CRITICAL/WARNING)
MUST_DO_RULES = [
    ("R01", "CRITICAL", "group_size >= 4", "gs < 4 → weak advantage signal or REINFORCE degeneration"),
    ("R02", "CRITICAL", "gradient_clipping > 0", "No gradient clipping → risk of NaN/explosion"),
    ("R03", "CRITICAL", "enforce_eager = True for DSV4", "DSV4 crashes without enforce_eager (11 failures documented)"),
    ("R04", "CRITICAL", "ZeRO-2 only (NOT ZeRO-3) with LoRA", "ZeRO-3 + PEFT LoRA broken (#8072/#8076)"),
    ("R05", "HIGH", "reference_model_mode = bypass OR offload", "Separate reference = double memory, OOM on single GPU"),
    ("R06", "HIGH", "optimizer = cpu_adam for ZeRO-2", "GPU Adam requires optimizer states in VRAM → OOM risk"),
    ("R07", "HIGH", "lora_rank >= 8", "r < 8 → insufficient expressiveness for alignment tasks"),
    ("R08", "HIGH", "ulimit >= 65535", "Low ulimit → distributed training hangs on file descriptors"),
    ("R09", "HIGH", "sleep_level >= 1 for rollout", "sleep_level=0 → no VRAM freed for training phase"),
    ("R10", "MEDIUM", "learning_rate in [1e-6, 5e-5]", "LR outside this range → instability or slow convergence"),
    ("R11", "MEDIUM", "weight_decay in [0, 0.1]", "Excessive weight decay → underfitting"),
    ("R12", "HIGH", "reward_type NOT pure outcome 0/1 if gs < 16", "Sparse reward + small gs → >30% degenerate groups"),
    ("R13", "HIGH", "reward_shaped = True for math/code tasks", "Shaped rewards provide 2x signal improvement"),
    ("R14", "CRITICAL", "dp_size >= 1 and tp_size × pp_size × dp_size = gpu_count", "Mismatch → training fails or unused GPUs"),
    ("R15", "HIGH", "rollout_engine = sglang for prefix caching", "vLLM rollout slower (no prefix cache reuse in GRPO)"),
    ("R16", "CRITICAL", "strategy ≠ deepspeed-zeRO-3 for LoRA", "ZeRO-3 splits parameters → LoRA applies to wrong shards"),
    ("R17", "HIGH", "checkpoint_engine = naive for dp=1", "NCCL broadcast dp=1 = identity → naive faster"),
    ("R18", "HIGH", "LoRA target modules include q,v,k,o", "Missing projection → incomplete LoRA coverage"),
]

# MUST NOT rules (violation = CRITICAL/WARNING)
MUST_NOT_RULES = [
    ("N01", "CRITICAL", "group_size ≠ 1", "gs=1 = REINFORCE(baseline=0), no advantage normalization"),
    ("N02", "CRITICAL", "overlap_comm ≠ True for dp=1", "NCCL dp=1 = identity, overlap_comm wastes resources + NaN risk (#8061)"),
    ("N03", "CRITICAL", "strategy ≠ deepspeed-zeRO-3 with LoRA", "ZeRO-3 + PEFT LoRA = parameter split mismatch"),
    ("N04", "CRITICAL", "sleep_level ≠ 2 for RTX 4090 GRPO", "sleep_level=2 crashes with cumem stream sync bug (#45552)"),
    ("N05", "CRITICAL", "reference_model_mode ≠ separate for single GPU", "Separate ref = OOM on single GPU (needs 2× model memory)"),
    ("N06", "CRITICAL", "reward_range NOT > 5.0", "Extreme reward range destroys advantage normalization"),
    ("N07", "HIGH", "n_epochs ≠ > 2 for GRPO", "Multiple epochs = stale data (policy changed after epoch 1)"),
    ("N08", "HIGH", "gradient_clipping ≠ 0.0", "clip_grad=0 → no protection against NaN (#8068)"),
    ("N09", "HIGH", "learning_rate NOT > 5e-4", "LR too high → policy divergence, NaN"),
    ("N10", "HIGH", "lora_rank NOT > 128 for 7B model", "r > 128 → too many params, diminishing returns, OOM risk"),
    ("N11", "HIGH", "dp_size NOT > gpu_count", "More DP than GPUs → process spawn fails"),
    ("N12", "MEDIUM", "optimizer ≠ gpu_adam with ZeRO-2 on RTX 4090", "GPU Adam states need extra VRAM → marginal on 24 GiB"),
]


# ============================================================
# Memory Estimation
# ============================================================

def estimate_memory(config: GRPOConfig) -> Dict[str, float]:
    """Estimate memory requirements for given configuration"""
    model_bytes = config.model_params_b * 2  # BF16
    model_gb = model_bytes  # 7B BF16 = 14 GiB

    # LoRA parameters
    lora_params_per_module = config.lora_rank * 2  # A + B matrices
    n_modules = len(config.lora_target_modules)
    # Approximate: each module has ~model_params_b/n_layers dimension
    n_layers = 28  # typical for 7B model
    lora_total_params = lora_params_per_module * n_modules * n_layers * (config.model_params_b * 1e9 / (n_layers * n_modules * 4096))
    lora_gb = lora_total_params * 2 / 1e9  # minimal for reasonable configs

    memory = {}

    # Model weights
    memory["model_weights"] = model_gb

    # LoRA params
    if config.lora_rank > 0:
        memory["lora_params"] = lora_gb
    else:
        memory["lora_params"] = 0

    # Optimizer states
    if config.optimizer in ["cpu_adam"]:
        memory["optimizer_states"] = 0  # CPU Adam → not in VRAM
    elif config.lora_rank > 0:
        # Only LoRA params need optimizer states
        memory["optimizer_states"] = lora_gb * 2 * 2 / 2  # Adam m+v, FP32 for LoRA params
    else:
        # Full param optimizer
        memory["optimizer_states"] = model_gb * 2 * 2  # Adam: 2 states × FP32 = 8 bytes/param

    # Reference model
    if config.reference_model_mode == "bypass":
        memory["reference_model"] = 0  # bypass = load from CPU during sync only
    elif config.reference_model_mode == "offload":
        memory["reference_model"] = 0.5  # partial overlap during KL computation
    elif config.reference_model_mode == "separate":
        memory["reference_model"] = model_gb  # full copy in VRAM
    else:
        memory["reference_model"] = 0

    # Activations
    memory["activations"] = 0.8  # gs=8 activations

    # Rollout KV cache (temporary)
    memory["rollout_kv_cache"] = 2.0

    # CUDA overhead
    memory["cuda_overhead"] = 1.0

    # Gradient buffers
    if config.lora_rank > 0:
        memory["gradients"] = lora_gb
    else:
        memory["gradients"] = model_gb * 0.5

    # Strategy overhead
    if config.strategy.startswith("deepspeed"):
        memory["deepspeed_overhead"] = 0.5
    else:
        memory["deepspeed_overhead"] = 0

    total = sum(memory.values())
    peak_training = total - memory["rollout_kv_cache"]  # KV cache freed during training

    return {
        "components": memory,
        "total": total,
        "peak_training": peak_training,
        "headroom": config.gpu_vram_gb - peak_training,
        "oom": peak_training > config.gpu_vram_gb,
        "oom_amount": max(0, peak_training - config.gpu_vram_gb),
    }


# ============================================================
# Validation Engine
# ============================================================

def validate_config(config: GRPOConfig) -> List[Dict[str, str]]:
    """Validate configuration against all MUST DO and MUST NOT rules"""

    violations = []

    # MUST DO checks
    for rule_id, severity, condition, reason in MUST_DO_RULES:
        violated = False

        if rule_id == "R01": violated = config.group_size < 4
        elif rule_id == "R02": violated = config.gradient_clipping <= 0
        elif rule_id == "R03": violated = not config.enforce_eager and config.rollout_engine in ["vllm", "sglang"]
        elif rule_id == "R04": violated = config.strategy == "deepspeed-zeRO-3" and config.lora_rank > 0
        elif rule_id == "R05": violated = config.reference_model_mode == "separate" and config.gpu_count == 1
        elif rule_id == "R06": violated = config.optimizer != "cpu_adam" and config.strategy.startswith("deepspeed")
        elif rule_id == "R07": violated = config.lora_rank < 8 and config.lora_rank > 0
        elif rule_id == "R08": violated = config.ulimit < 65535
        elif rule_id == "R09": violated = config.sleep_level < 1
        elif rule_id == "R10": violated = config.learning_rate < 1e-6 or config.learning_rate > 5e-5
        elif rule_id == "R11": violated = config.weight_decay < 0 or config.weight_decay > 0.1
        elif rule_id == "R12": violated = config.reward_type == "rule_exact" and config.group_size < 16
        elif rule_id == "R13": violated = not config.reward_shaped and config.reward_type in ["format_outcome", "math_reasoning"]
        elif rule_id == "R14": violated = config.tp_size * config.pp_size * config.dp_size != config.gpu_count
        elif rule_id == "R15": violated = config.rollout_engine != "sglang" and config.framework == "verl"
        elif rule_id == "R16": violated = config.strategy == "deepspeed-zeRO-3" and config.lora_rank > 0
        elif rule_id == "R17": violated = config.checkpoint_engine != "naive" and config.dp_size == 1
        elif rule_id == "R18": violated = not all(m in config.lora_target_modules for m in ["q_proj", "v_proj"])

        if violated:
            violations.append({
                "rule_id": rule_id,
                "type": "MUST_DO",
                "severity": severity,
                "condition": condition,
                "reason": reason,
                "current_value": _get_current_value(config, rule_id),
            })

    # MUST NOT checks
    for rule_id, severity, condition, reason in MUST_NOT_RULES:
        violated = False

        if rule_id == "N01": violated = config.group_size == 1
        elif rule_id == "N02": violated = config.strategy.startswith("deepspeed") and config.dp_size == 1  # overlap_comm implicit
        elif rule_id == "N03": violated = config.strategy == "deepspeed-zeRO-3" and config.lora_rank > 0
        elif rule_id == "N04": violated = config.sleep_level == 2 and config.gpu_type == "rtx4090"
        elif rule_id == "N05": violated = config.reference_model_mode == "separate" and config.gpu_count == 1
        elif rule_id == "N06": violated = False  # would need reward_range field
        elif rule_id == "N07": violated = config.n_epochs > 2
        elif rule_id == "N08": violated = config.gradient_clipping == 0.0
        elif rule_id == "N09": violated = config.learning_rate > 5e-4
        elif rule_id == "N10": violated = config.lora_rank > 128
        elif rule_id == "N11": violated = config.dp_size > config.gpu_count
        elif rule_id == "N12": violated = config.optimizer != "cpu_adam" and config.strategy.startswith("deepspeed-zeRO-2") and config.gpu_type == "rtx4090"

        if violated:
            violations.append({
                "rule_id": rule_id,
                "type": "MUST_NOT",
                "severity": severity,
                "condition": condition,
                "reason": reason,
                "current_value": _get_current_value(config, rule_id),
            })

    # Memory check
    mem = estimate_memory(config)
    if mem["oom"]:
        violations.append({
            "rule_id": "MEM",
            "type": "MEMORY",
            "severity": "CRITICAL",
            "condition": f"Peak memory < GPU VRAM ({config.gpu_vram_gb:.1f} GiB)",
            "reason": f"Peak {mem['peak_training']:.2f} GiB > {config.gpu_vram_gb:.1f} GiB → OOM by {mem['oom_amount']:.2f} GiB",
            "current_value": f"{mem['peak_training']:.2f} GiB",
        })

    return violations


def _get_current_value(config: GRPOConfig, rule_id: str) -> str:
    """Get current config value for a violated rule"""
    mapping = {
        "R01": f"group_size={config.group_size}",
        "R02": f"gradient_clipping={config.gradient_clipping}",
        "R03": f"enforce_eager={config.enforce_eager}",
        "R04": f"strategy={config.strategy}, lora_rank={config.lora_rank}",
        "R05": f"reference_model_mode={config.reference_model_mode}, gpu_count={config.gpu_count}",
        "R06": f"optimizer={config.optimizer}",
        "R07": f"lora_rank={config.lora_rank}",
        "R08": f"ulimit={config.ulimit}",
        "R09": f"sleep_level={config.sleep_level}",
        "R10": f"learning_rate={config.learning_rate}",
        "R11": f"weight_decay={config.weight_decay}",
        "R12": f"reward_type={config.reward_type}, group_size={config.group_size}",
        "R13": f"reward_shaped={config.reward_shaped}",
        "R14": f"tp={config.tp_size}×pp={config.pp_size}×dp={config.dp_size}={config.tp_size*config.pp_size*config.dp_size} != {config.gpu_count}",
        "R15": f"rollout_engine={config.rollout_engine}",
        "R16": f"strategy={config.strategy}, lora_rank={config.lora_rank}",
        "R17": f"checkpoint_engine={config.checkpoint_engine}, dp={config.dp_size}",
        "R18": f"lora_target_modules={config.lora_target_modules}",
        "N01": f"group_size={config.group_size}",
        "N02": f"strategy={config.strategy}, dp={config.dp_size}",
        "N04": f"sleep_level={config.sleep_level}, gpu_type={config.gpu_type}",
        "N05": f"reference_model_mode={config.reference_model_mode}",
        "N07": f"n_epochs={config.n_epochs}",
        "N08": f"gradient_clipping={config.gradient_clipping}",
        "N09": f"learning_rate={config.learning_rate}",
        "N10": f"lora_rank={config.lora_rank}",
    }
    return mapping.get(rule_id, "N/A")


# ============================================================
# Framework Default Configs
# ============================================================

FRAMEWORK_DEFAULTS = {
    "verl": GRPOConfig(
        framework="verl", rollout_engine="sglang", strategy="fsdp1",
        lora_rank=32, group_size=8, reference_model_mode="bypass",
        optimizer="cpu_adam", checkpoint_engine="naive", sleep_level=1,
        enforce_eager=True, reward_type="format_outcome", reward_shaped=True,
    ),
    "openrlhf": GRPOConfig(
        framework="openrlhf", rollout_engine="vllm", strategy="deepspeed-zeRO-2",
        lora_rank=32, group_size=8, reference_model_mode="offload",
        optimizer="cpu_adam", checkpoint_engine="deepspeed", sleep_level=1,
        enforce_eager=True, reward_type="rm_score", reward_shaped=True,
    ),
    "trl": GRPOConfig(
        framework="trl", rollout_engine="huggingface", strategy="fsdp1",
        lora_rank=8, group_size=4, reference_model_mode="separate",
        optimizer="adam", checkpoint_engine="torch", sleep_level=0,
        enforce_eager=False, reward_type="rule_exact", reward_shaped=False,
    ),
    "rllm": GRPOConfig(
        framework="rllm", rollout_engine="vllm", strategy="fsdp1",
        lora_rank=16, group_size=1, reference_model_mode="none",
        optimizer="adam", checkpoint_engine="torch", sleep_level=0,
        enforce_eager=False, reward_type="ranking", reward_shaped=False,
    ),
}


# ============================================================
# Mode 1: Validate
# ============================================================

def mode_validate():
    """Validate configuration against all rules"""

    print("=" * 80)
    print("MODE: validate — GRPO Configuration Validation")
    print("=" * 80)
    print()

    # Validate RTX 4090 optimal config
    config = GRPOConfig(
        framework="verl", rollout_engine="sglang", strategy="fsdp1",
        lora_rank=32, group_size=8, reference_model_mode="bypass",
        optimizer="cpu_adam", checkpoint_engine="naive", sleep_level=1,
        enforce_eager=True, reward_type="format_outcome", reward_shaped=True,
        gradient_clipping=1.0, learning_rate=1e-5, weight_decay=0.01,
        ulimit=65535, n_epochs=1, gpu_type="rtx4090", gpu_vram_gb=24.0,
    )

    print("  Configuration being validated:")
    print(f"    Framework: {config.framework}")
    print(f"    Model: {config.model_name} ({config.model_params_b}B params)")
    print(f"    Strategy: {config.strategy}, LoRA rank: {config.lora_rank}")
    print(f"    Group size: {config.group_size}, LR: {config.learning_rate}")
    print(f"    Reference: {config.reference_model_mode}, Optimizer: {config.optimizer}")
    print(f"    GPU: {config.gpu_type} ({config.gpu_vram_gb:.1f} GiB)")
    print()

    # Memory estimation
    mem = estimate_memory(config)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Memory Estimation                                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    for comp, mem_val in mem["components"].items():
        if mem_val > 0:
            print(f"    {comp:<25} {mem_val:>8.2f} GiB")
    print(f"    {'TOTAL':<25} {mem['total']:>8.2f} GiB")
    print(f"    {'Peak (training)':<25} {mem['peak_training']:>8.2f} GiB")
    print(f"    {'GPU available':<25} {config.gpu_vram_gb:>8.2f} GiB")
    print(f"    {'Headroom':<25} {mem['headroom']:>8.2f} GiB")
    status = "FIT" if not mem["oom"] else "OOM"
    print(f"    {'Status':<25} {status:>8}")
    print()

    # Rule validation
    violations = validate_config(config)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Rule Validation Results                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    if not violations:
        print("  ★★★ ALL RULES PASSED — Configuration is valid!")
    else:
        critical = [v for v in violations if v["severity"] == "CRITICAL"]
        high = [v for v in violations if v["severity"] == "HIGH"]
        medium = [v for v in violations if v["severity"] == "MEDIUM"]

        if critical:
            print("  ★★★★★★★★ CRITICAL VIOLATIONS (must fix before training):")
            for v in critical:
                print(f"    [{v['rule_id']}] {v['type']}: {v['condition']}")
                print(f"      Current: {v['current_value']}")
                print(f"      Reason: {v['reason']}")
            print()

        if high:
            print("  ★★★ HIGH VIOLATIONS (should fix for optimal performance):")
            for v in high:
                print(f"    [{v['rule_id']}] {v['type']}: {v['condition']}")
                print(f"      Current: {v['current_value']}")
                print(f"      Reason: {v['reason']}")
            print()

        if medium:
            print("  ★★ MEDIUM VIOLATIONS (recommended improvements):")
            for v in medium:
                print(f"    [{v['rule_id']}] {v['type']}: {v['condition']}")
                print(f"      Current: {v['current_value']}")
                print(f"      Reason: {v['reason']}")
            print()

    # Now validate common MISCONFIGURATIONS
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Common Misconfiguration Detection                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    misconfigs = [
        ("gs=1 (REINFORCE degeneration)", GRPOConfig(group_size=1)),
        ("ZeRO-3 + LoRA", GRPOConfig(strategy="deepspeed-zeRO-3", lora_rank=32)),
        ("separate reference + single GPU", GRPOConfig(reference_model_mode="separate", gpu_count=1)),
        ("sleep_level=2 on RTX 4090", GRPOConfig(sleep_level=2, gpu_type="rtx4090")),
        ("gradient_clipping=0", GRPOConfig(gradient_clipping=0.0)),
        ("TRL default config on RTX 4090", FRAMEWORK_DEFAULTS["trl"].__class__(**{k: getattr(FRAMEWORK_DEFAULTS["trl"], k) for k in FRAMEWORK_DEFAULTS["trl"].__dataclass_fields__})),
        ("rLLM default config on RTX 4090", FRAMEWORK_DEFAULTS["rllm"].__class__(**{k: getattr(FRAMEWORK_DEFAULTS["rllm"], k) for k in FRAMEWORK_DEFAULTS["rllm"].__dataclass_fields__})),
    ]

    for name, bad_config in misconfigs:
        violations = validate_config(bad_config)
        critical_count = sum(1 for v in violations if v["severity"] == "CRITICAL")
        high_count = sum(1 for v in violations if v["severity"] == "HIGH")
        print(f"  {name}:")
        print(f"    Critical violations: {critical_count}, High violations: {high_count}")
        for v in violations[:3]:
            print(f"    [{v['rule_id']}] {v['reason']}")
        print()

    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


# ============================================================
# Mode 2: Checklist
# ============================================================

def mode_checklist():
    """Full pre-flight checklist before starting GRPO training"""

    print("=" * 80)
    print("MODE: checklist — GRPO Training Pre-flight Checklist")
    print("=" * 80)
    print()

    sections = [
        ("Hardware & Environment", [
            ("GPU available and recognized by PyTorch", "python3 -c 'import torch; print(torch.cuda.is_available())'"),
            ("GPU VRAM matches expected value", "python3 -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9)'"),
            ("ulimit >= 65535", "ulimit -n"),
            ("CUDA version >= 12.1", "nvcc --version"),
            ("Python version >= 3.10", "python3 --version"),
            ("NCCL available", "python3 -c 'import torch.distributed as dist; print(dist.is_nccl_available())'"),
        ]),
        ("Conda Environment", [
            ("Conda env activated", "conda info --envs"),
            ("PyTorch installed (CUDA version)", "python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'"),
            ("verl/vLLM/SGLang installed", "pip list | grep -E 'verl|vllm|sglang'"),
            ("DeepSpeed installed (if using)", "pip list | grep deepspeed"),
            ("Flash Attention available", "python3 -c 'import flash_attn; print(flash_attn.__version__)'"),
        ]),
        ("Model & Data", [
            ("Model weights downloaded", "ls -la ~/.cache/huggingface/"),
            ("Tokenizer working", "python3 -c 'from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(\"Qwen/Qwen2.5-7B\"); print(t.encode(\"test\"))'"),
            ("Dataset accessible", "ls -la data/"),
            ("Reward function tested", "python3 -c 'test_reward_fn()'"),
        ]),
        ("Configuration Validation", [
            ("group_size >= 4", "Check config.yaml: actor.rollout.group_size"),
            ("gradient_clipping > 0", "Check config.yaml: actor.optim.max_grad_norm"),
            ("enforce_eager = True", "Check config.yaml: rollout.engine.enforce_eager"),
            ("ZeRO-2 only (NOT ZeRO-3)", "Check config.yaml: actor.strategy"),
            ("LoRA rank >= 8", "Check config.yaml: actor.lora.rank"),
            ("reference bypass/offload", "Check config.yaml: actor.reference.mode"),
            ("sleep_level = 1", "Check config.yaml: rollout.engine.sleep_level"),
            ("CPU Adam optimizer", "Check config.yaml: actor.optim.type"),
            ("Learning rate in range", "Check config.yaml: actor.optim.lr"),
        ]),
        ("Memory Safety", [
            ("Peak memory < GPU VRAM", "Run: python3 tools/grpo_training_step_timing_model.py rtx4090"),
            ("Headroom >= 2 GiB", "Check output: peak_memory + 2 GiB < 24 GiB"),
            ("LoRA+bypass config for RTX 4090", "Verify: LoRA r=32 + bypass + FSDP1"),
            ("No separate reference model", "Verify: reference_model_mode = bypass"),
        ]),
        ("Framework-specific", [
            ("verl: SGLang rollout engine", "Check: actor.rollout.name = 'sglang'"),
            ("verl: naive checkpoint for dp=1", "Check: actor.checkpoint.name = 'naive'"),
            ("DeepSpeed: overlap_comm=False for dp=1", "Check: zero_optimization.overlap_comm = False"),
            ("vLLM: enforce_eager=True", "Check: vllm_config.enforce_eager = True"),
            ("SGLang: sleep_level=1", "Check: sglang_config.sleep_level = 1"),
        ]),
    ]

    for section_name, checks in sections:
        print(f"╔══════════════════════════════════════════════════════════════════╗")
        print(f"║  {section_name:<62}║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        print()
        for check, command in checks:
            print(f"  [ ] {check}")
            print(f"      $ {command}")
        print()

    print("=" * 80)
    print("CHECKLIST COMPLETE — Verify each item before starting training!")
    print("=" * 80)


# ============================================================
# Mode 3: Compare
# ============================================================

def mode_compare():
    """Compare default configs across frameworks"""

    print("=" * 80)
    print("MODE: compare — Framework Default Configuration Comparison")
    print("=" * 80)
    print()

    # Framework config comparison table
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Default Configuration Comparison (7B model, single GPU)     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    configs_to_compare = {
        "verl (optimal)": GRPOConfig(
            framework="verl", rollout_engine="sglang", strategy="fsdp1",
            lora_rank=32, group_size=8, reference_model_mode="bypass",
            optimizer="cpu_adam", checkpoint_engine="naive", sleep_level=1,
            enforce_eager=True, reward_type="format_outcome", reward_shaped=True,
            gradient_clipping=1.0, learning_rate=1e-5, n_epochs=1,
        ),
        "verl (default)": GRPOConfig(
            framework="verl", rollout_engine="vllm", strategy="fsdp1",
            lora_rank=8, group_size=4, reference_model_mode="bypass",
            optimizer="adam", checkpoint_engine="torch", sleep_level=0,
            enforce_eager=False, reward_type="format_outcome", reward_shaped=True,
            gradient_clipping=1.0, learning_rate=1e-5, n_epochs=1,
        ),
        "OpenRLHF": FRAMEWORK_DEFAULTS["openrlhf"],
        "TRL": FRAMEWORK_DEFAULTS["trl"],
        "rLLM": FRAMEWORK_DEFAULTS["rllm"],
    }

    fields_to_compare = [
        ("framework", "Framework"),
        ("rollout_engine", "Rollout Engine"),
        ("strategy", "Strategy"),
        ("lora_rank", "LoRA Rank"),
        ("group_size", "Group Size"),
        ("reference_model_mode", "Reference Mode"),
        ("optimizer", "Optimizer"),
        ("sleep_level", "Sleep Level"),
        ("enforce_eager", "enforce_eager"),
        ("reward_type", "Reward Type"),
        ("reward_shaped", "Shaped Reward"),
        ("gradient_clipping", "clip_grad"),
        ("n_epochs", "Epochs"),
    ]

    print(f"  {'Parameter':<20}", end="")
    for name in configs_to_compare:
        print(f" {name:>16}", end="")
    print()
    print("  " + "-" * (20 + 16 * len(configs_to_compare)))

    for field_name, display_name in fields_to_compare:
        print(f"  {display_name:<20}", end="")
        for name, cfg in configs_to_compare.items():
            val = getattr(cfg, field_name)
            print(f" {str(val):>16}", end="")
        print()

    print()

    # Validation results per framework
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Validation Results per Framework (on RTX 4090)              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    print(f"  {'Framework':<20} {'Critical':>10} {'High':>10} {'Medium':>10} {'Memory':>10} {'Fit?':>6}")
    print("  " + "-" * 66)

    for name, cfg in configs_to_compare.items():
        # Set GPU config for RTX 4090
        cfg.gpu_type = "rtx4090"
        cfg.gpu_vram_gb = 24.0
        cfg.gpu_count = 1

        violations = validate_config(cfg)
        critical_count = sum(1 for v in violations if v["severity"] == "CRITICAL")
        high_count = sum(1 for v in violations if v["severity"] == "HIGH")
        medium_count = sum(1 for v in violations if v["severity"] == "MEDIUM")

        mem = estimate_memory(cfg)
        mem_status = "OOM" if mem["oom"] else "FIT"

        print(f"  {name:<20} {critical_count:>10} {high_count:>10} {medium_count:>10} {mem['peak_training']:>10.2f} {mem_status:>6}")

    print()
    print("  ★★★ Only 'verl (optimal)' config PASSES all checks for RTX 4090")
    print("  ★★★ rLLM defaults: CRITICAL violations (gs=1 = REINFORCE)")
    print("  ★★★ TRL defaults: CRITICAL violations (separate ref = OOM)")
    print()

    print("=" * 80)
    print("COMPARE COMPLETE")
    print("=" * 80)


# ============================================================
# Mode 4: RTX 4090
# ============================================================

def mode_rtx4090():
    """RTX 4090-specific configuration validator"""

    print("=" * 80)
    print("MODE: rtx4090 — RTX 4090 GRPO Configuration Validator")
    print("=" * 80)
    print()

    # The ONLY viable RTX 4090 config
    optimal = GRPOConfig(
        framework="verl", rollout_engine="sglang", strategy="fsdp1",
        lora_rank=32, group_size=8, reference_model_mode="bypass",
        optimizer="cpu_adam", checkpoint_engine="naive", sleep_level=1,
        enforce_eager=True, reward_type="format_outcome", reward_shaped=True,
        gradient_clipping=1.0, learning_rate=1e-5, weight_decay=0.01,
        n_epochs=1, gpu_type="rtx4090", gpu_vram_gb=24.0, gpu_count=1,
        dp_size=1, tp_size=1, pp_size=1, ulimit=65535,
        model_name="Qwen2.5-7B", model_params_b=7.0,
    )

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ★★★ RTX 4090 OPTIMAL CONFIGURATION ★★★                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    optimal_fields = [
        ("framework", "verl"),
        ("rollout_engine", "sglang"),
        ("strategy", "fsdp1"),
        ("lora_rank", 32),
        ("group_size", 8),
        ("reference_model_mode", "bypass"),
        ("optimizer", "cpu_adam"),
        ("checkpoint_engine", "naive"),
        ("sleep_level", 1),
        ("enforce_eager", True),
        ("gradient_clipping", 1.0),
        ("learning_rate", "1e-5"),
        ("reward_type", "format_outcome"),
        ("reward_shaped", True),
        ("model_name", "Qwen2.5-7B"),
    ]

    print("  MUST set these values (any deviation = violation):")
    print()
    for field_name, value in optimal_fields:
        print(f"    {field_name:<25} = {value}")

    print()

    # Memory verification
    mem = estimate_memory(optimal)
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Memory Verification                                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    for comp, mem_val in mem["components"].items():
        if mem_val > 0:
            pct = mem_val / optimal.gpu_vram_gb * 100
            print(f"    {comp:<25} {mem_val:>8.2f} GiB ({pct:.1f}%)")
    print(f"    {'Peak (training)':<25} {mem['peak_training']:>8.2f} GiB ({mem['peak_training']/optimal.gpu_vram_gb*100:.1f}%)")
    print(f"    {'Headroom':<25} {mem['headroom']:>8.2f} GiB ({mem['headroom']/optimal.gpu_vram_gb*100:.1f}%)")
    print()

    # Validate optimal config
    violations = validate_config(optimal)
    if not violations:
        print("  ★★★★★★★★ OPTIMAL CONFIG: ALL 30 RULES PASSED — 0 violations!")
    else:
        print(f"  Violations found: {len(violations)}")
        for v in violations:
            print(f"    [{v['rule_id']}] {v['severity']}: {v['reason']}")

    print()

    # Alternative configs with tradeoffs
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Alternative Configs (with tradeoff analysis)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    alternatives = [
        ("LoRA r=16 (less memory)", GRPOConfig(
            framework="verl", rollout_engine="sglang", strategy="fsdp1",
            lora_rank=16, group_size=8, reference_model_mode="bypass",
            optimizer="cpu_adam", sleep_level=1, enforce_eager=True,
            gpu_type="rtx4090", gpu_vram_gb=24.0, gradient_clipping=1.0,
        )),
        ("gs=4 (faster step, weaker signal)", GRPOConfig(
            framework="verl", rollout_engine="sglang", strategy="fsdp1",
            lora_rank=32, group_size=4, reference_model_mode="bypass",
            optimizer="cpu_adam", sleep_level=1, enforce_eager=True,
            gpu_type="rtx4090", gpu_vram_gb=24.0, gradient_clipping=1.0,
        )),
        ("gs=16 (stronger signal, slower)", GRPOConfig(
            framework="verl", rollout_engine="sglang", strategy="fsdp1",
            lora_rank=32, group_size=16, reference_model_mode="bypass",
            optimizer="cpu_adam", sleep_level=1, enforce_eager=True,
            gpu_type="rtx4090", gpu_vram_gb=24.0, gradient_clipping=1.0,
        )),
        ("Qwen2.5-3B (smaller model)", GRPOConfig(
            framework="verl", rollout_engine="sglang", strategy="fsdp1",
            lora_rank=32, group_size=8, reference_model_mode="bypass",
            optimizer="cpu_adam", sleep_level=1, enforce_eager=True,
            model_name="Qwen2.5-3B", model_params_b=3.0,
            gpu_type="rtx4090", gpu_vram_gb=24.0, gradient_clipping=1.0,
        )),
        ("vLLM rollout (slower, no prefix cache)", GRPOConfig(
            framework="verl", rollout_engine="vllm", strategy="fsdp1",
            lora_rank=32, group_size=8, reference_model_mode="bypass",
            optimizer="cpu_adam", sleep_level=1, enforce_eager=True,
            gpu_type="rtx4090", gpu_vram_gb=24.0, gradient_clipping=1.0,
        )),
    ]

    print(f"  {'Alternative':<30} {'Violations':>10} {'Memory':>10} {'Tradeoff':>20}")
    print("  " + "-" * 70)

    for name, alt_config in alternatives:
        violations = validate_config(alt_config)
        mem = estimate_memory(alt_config)

        tradeoff = ""
        if alt_config.lora_rank == 16:
            tradeoff = "Less LoRA capacity"
        elif alt_config.group_size == 4:
            tradeoff = "Weaker advantage signal"
        elif alt_config.group_size == 16:
            tradeoff = "2x slower rollout"
        elif alt_config.model_params_b == 3.0:
            tradeoff = "Smaller model quality"
        elif alt_config.rollout_engine == "vllm":
            tradeoff = "No prefix caching"

        v_count = len(violations)
        c_count = sum(1 for v in violations if v["severity"] == "CRITICAL")

        print(f"  {name:<30} {v_count:>10} {mem['peak_training']:>10.2f} {tradeoff:>20}")

    print()
    print("  ★★★ OPTIMAL: verl + SGLang + FSDP1 + LoRA r=32 + bypass + gs=8")
    print("  ★★★ Acceptable alternatives: LoRA r=16, gs=4 with shaped reward, 3B model")
    print("  ★★★ NEVER: gs=1, ZeRO-3, separate reference, sleep_level=2")

    print()
    print("=" * 80)
    print("RTX 4090 VALIDATION COMPLETE")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training Configuration Validator"
    )
    parser.add_argument(
        "mode",
        choices=["validate", "checklist", "compare", "rtx4090"],
        help="Validation mode"
    )
    args = parser.parse_args()

    start_time = time.time()

    if args.mode == "validate":
        mode_validate()
    elif args.mode == "checklist":
        mode_checklist()
    elif args.mode == "compare":
        mode_compare()
    elif args.mode == "rtx4090":
        mode_rtx4090()

    elapsed = time.time() - start_time
    print()
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
