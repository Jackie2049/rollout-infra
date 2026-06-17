#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Rollout-Infra Team
"""
RTX 4090 GRPO Training Configuration Reference
Quick reference for optimal GRPO training configs across all 7 frameworks.
Modes: show (display recommended configs), validate (check user config), estimate (memory/time estimate)

Usage:
  python rtx4090_grpo_config_reference.py show           # Show all recommended configs
  python rtx4090_grpo_config_reference.py show --framework verl  # Show verl configs only
  python rtx4090_grpo_config_reference.py validate --config my_config.yaml
  python rtx4090_grpo_config_reference.py estimate --model qwen3-4b --framework verl
"""

import argparse
import json
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Recommended GRPO configs per framework
# ═══════════════════════════════════════════════════════════════════════════

RECOMMENDED_CONFIGS = {
    "verl_tinker_primitives": {
        "name": "verl HYBRID + Tinker primitives (RTX 4090 #1.5 — split training IN verl)",
        "framework": "verl",
        "rank": "#1.5",
        "description": "Tinker-style split training IN verl — optimizer_zero_grad/forward_backward/optimizer_step primitives",
        "training_framework": "verl HYBRID + TinkerTrainingWorker",
        "rollout_backend": "vLLM (in-process)",
        "determinism_level": "EAGER or COMPILE (Fusion Guard)",
        "memory_peak_gb": 16.2,
        "fits_24gb": True,
        "margin_gb": 7.8,
        "config": {
            "rollout_mode": "hybrid",
            "bypass_mode": True,
            "advantage_estimator": "grpo",
            "policy_loss_type": "ppo_clip",
            "enforce_eager": True,
            "gradient_clipping": 1.0,
            "lora_rank": 16,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "n_grpo_responses": 8,
            "use_tinker_training_worker": True,
            "sleep_level": 1,
        },
        "warnings": [
            "#6717 TinkerTrainingWorker MERGED June 15 → split training primitives available",
            "No framework switch needed → rLLM Tinker features IN verl",
            "MUST use verl >= commit with #6699 detach fix → 4x memory reduction",
        ],
    },
    "verl_hybrid_bypass_grpo": {
        "name": "verl HYBRID + bypass + GRPO (RTX 4090 #2)",
        "framework": "verl",
        "rank": "#2",
        "description": "Standard GRPO with bypass_mode — simplest viable config",
        "training_framework": "verl HYBRID",
        "rollout_backend": "vLLM (in-process)",
        "determinism_level": "EAGER (enforce_eager=True) or COMPILE (Fusion Guard, when merged)",
        "memory_peak_gb": 16.2,
        "fits_24gb": True,
        "margin_gb": 7.8,
        "config": {
            "rollout_mode": "hybrid",
            "bypass_mode": True,
            "advantage_estimator": "grpo",
            "policy_loss_type": "ppo_clip",
            "enforce_eager": True,
            "gradient_clipping": 1.0,
            "lora_rank": 16,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "n_grpo_responses": 8,
            "sleep_level": 1,
        },
        "warnings": [
            "enforce_eager=True → ~2x slower inference → fallback before Fusion Guard",
            "gradient_clipping=1.0 → MUST set explicitly (default 0 = silent bug)",
            "MUST use verl >= commit with #6699 detach fix → 4x memory reduction (64→16.2 GiB)",
        ],
    },
    "verl_hybrid_bypass_cppo": {
        "name": "verl HYBRID + bypass + CPPO (RTX 4090 #1 BEST)",
        "framework": "verl",
        "rank": "#1",
        "description": "Best trust region — CPPO position-weighted cumulative budget, best determinism path",
        "training_framework": "verl HYBRID",
        "rollout_backend": "vLLM (in-process)",
        "determinism_level": "EAGER or COMPILE (Fusion Guard)",
        "memory_peak_gb": 16.2,
        "fits_24gb": True,
        "margin_gb": 7.8,
        "config": {
            "rollout_mode": "hybrid",
            "bypass_mode": True,
            "advantage_estimator": "grpo",
            "policy_loss_type": "cppo",
            "enforce_eager": True,
            "gradient_clipping": 1.0,
            "use_kl_loss": False,  # CPPO+bypass makes KL redundant
            "entropy_coeff": 0.01,
            "lora_rank": 16,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "n_grpo_responses": 8,
            "sleep_level": 1,
        },
        "warnings": [
            "CPPO requires bypass_mode — mathematical necessity",
            "CPPO PR #6731 NOT merged → need branch checkout",
            "use_kl_loss=False with CPPO+bypass → KL penalty redundant",
            "MUST use verl >= commit with #6699 detach fix → 4x memory reduction (64→16.2 GiB)",
        ],
    },
    "rllm_tinker_sglang": {
        "name": "rLLM Tinker + SGLang (RTX 4090 #3 — BLOCKED by #605 bug)",
        "framework": "rLLM",
        "rank": "#3",
        "description": "BLOCKED by #605 GRPO grouping bug → fix pending → potential #1 when fixed",
        "training_framework": "rLLM Tinker",
        "rollout_backend": "SGLang (client-server SDK)",
        "determinism_level": "KERNEL (tl.constexpr → gold standard)",
        "memory_peak_gb": 19.0,
        "fits_24gb": True,
        "margin_gb": 5.0,
        "config": {
            "serving_backend": "sglang",
            "enable_deterministic_inference": True,
            "bypass_mode": True,
            "gradient_clipping": 1.0,
            "lora_rank": 16,
            "n_grpo_responses": 8,
        },
        "warnings": [
            "#605 CRITICAL: GRPO grouping bug → trajectory.uid vs task_ids → group size 1 → BROKEN!",
            "Tinker checkpoint NOT standard PEFT → export gap needs resolution",
            "SGLang integration stability → check #6117",
            "Client-server SDK → NOT in-process → weight sync via checkpoint",
            "Rank downgraded #1→#3 until #605 fixed → potential #1 when fixed",
        ],
    },
    "deepspeed_zero2_lora": {
        "name": "DeepSpeed ZeRO-2 + LoRA (RTX 4090 #2.5 — training ONLY)",
        "framework": "DeepSpeed",
        "rank": "#2.5",
        "description": "For supervised fine-tuning only — NOT for RL/inference",
        "training_framework": "DeepSpeed ZeRO-2",
        "rollout_backend": "vLLM/SGLang (external — separate serving)",
        "determinism_level": "EAGER (no compile) + frozen router (MoE)",
        "memory_peak_gb": 10.0,
        "fits_24gb": True,
        "margin_gb": 14.0,
        "config": {
            "zero_stage": 2,
            "optimizer": "cpu_adam",
            "offload_optimizer": True,
            "overlap_comm": False,  # MUST! #8061 NaN bug
            "gradient_clipping": 1.0,
            "lora_rank": 16,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
        },
        "warnings": [
            "overlap_comm=False MANDATORY on single GPU (#8061 NaN bug)",
            "gradient_clipping=1.0 MUST set (default 0 = silent bug #8068)",
            "NOT for RL training — use verl/rLLM for RL",
            "For inference: use vLLM/SGLang, NOT DeepSpeed",
            "ZeRO-3+PEFT regression in v0.19.2 (#8072/#8073) → ZeRO-2 unaffected → safe!",
        ],
    },
    "deepspeed_zero2_moe_lora": {
        "name": "DeepSpeed ZeRO-2 + AutoEP MoE + LoRA (RTX 4090 MoE training)",
        "framework": "DeepSpeed",
        "rank": "#2.5 MoE",
        "description": "MoE training viable with EP=1 + freeze router + LoRA",
        "training_framework": "DeepSpeed ZeRO-2 + AutoEP",
        "rollout_backend": "vLLM/SGLang (external)",
        "determinism_level": "EAGER + freeze router (0 LOC deterministic)",
        "memory_peak_gb": 19.85,
        "fits_24gb": True,
        "margin_gb": 4.15,
        "config": {
            "zero_stage": 2,
            "optimizer": "cpu_adam",
            "offload_optimizer": True,
            "overlap_comm": False,
            "gradient_clipping": 1.0,
            "ep_size": 1,  # MUST on single GPU
            "freeze_moe_router": True,  # 0 LOC — immediate deterministic routing
            "lora_rank": 16,
            "lora_target_modules": ["router", "expert_mlp", "shared_mlp", "attention"],
            "offload_ratio": 0.5,  # LoRAOptimizedLinear — 50% base weights to CPU
        },
        "warnings": [
            "ep_size=1 MANDATORY on single GPU (no AllToAll)",
            "freeze_moe_router = 0 LOC deterministic routing",
            "offload_ratio=0.5 critical for memory budget",
            "RouterReplay needed for CUDA graph future (~300 LOC)",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Memory estimation models
# ═══════════════════════════════════════════════════════════════════════════

MEMORY_ESTIMATES = {
    "qwen3-4b": {"params_gb": 8.0, "lora_trainable_pct": 0.2},
    "qwen3-8b": {"params_gb": 16.0, "lora_trainable_pct": 0.2},
    "qwen3-moe-a0.6b+b4b": {"params_gb": 4.8, "lora_trainable_pct": 0.3, "moe": True},
    "llama3.1-8b": {"params_gb": 16.0, "lora_trainable_pct": 0.2},
}


def estimate_memory(model_name, framework_key):
    """Estimate peak memory for a model + framework combination."""
    if model_name in MEMORY_ESTIMATES:
        m = MEMORY_ESTIMATES[model_name]
    else:
        return None

    config = RECOMMENDED_CONFIGS.get(framework_key)
    if not config:
        return None

    # Base memory: model params + gradients + activations
    model_mem = m["params_gb"]
    grad_mem = model_mem  # Same size as model (fp32 gradients)
    act_mem = 2.0  # Typical activation memory

    if m.get("moe"):
        # MoE: offload_ratio reduces base weight memory
        offload_ratio = config["config"].get("offload_ratio", 0)
        base_mem = model_mem * (1 - offload_ratio)
        total = base_mem + grad_mem + act_mem + 2.0  # LoRA optimizer overhead
    else:
        # Standard: LoRA reduces optimizer overhead dramatically
        lora_pct = m["lora_trainable_pct"]
        # ZeRO-2+CPU_Adam: optimizer on CPU → 0 on GPU
        if config["config"].get("offload_optimizer"):
            optimizer_gpu_mem = 0
        else:
            optimizer_gpu_mem = model_mem * lora_pct * 2  # m+v for LoRA params only

        total = model_mem + grad_mem + act_mem + optimizer_gpu_mem

    return {
        "model": model_name,
        "framework": config["name"],
        "peak_memory_gb": round(total, 1),
        "fits_24gb": total < 24,
        "margin_gb": round(24 - total, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Config validation rules
# ═══════════════════════════════════════════════════════════════════════════

VALIDATION_RULES = {
    "overlap_comm_single_gpu": {
        "check": lambda c: c.get("zero_stage") and c.get("overlap_comm") == True,
        "message": "overlap_comm=True on single GPU → #8061 NaN bug → MUST set overlap_comm=False",
        "severity": "CRITICAL",
    },
    "gradient_clipping_default": {
        "check": lambda c: c.get("gradient_clipping") is None or c.get("gradient_clipping") == 0,
        "message": "gradient_clipping=0 (default) → silent bug #8068 → MUST set gradient_clipping=1.0",
        "severity": "CRITICAL",
    },
    "zero3_single_gpu": {
        "check": lambda c: c.get("zero_stage") == 3,
        "message": "ZeRO-3 on single GPU = pure overhead → ALWAYS use ZeRO-2",
        "severity": "HIGH",
    },
    "no_bypass_rl": {
        "check": lambda c: c.get("bypass_mode") is False and c.get("rollout_mode") == "hybrid",
        "message": "bypass_mode=False → 18Ψ memory → MUST use bypass_mode=True for RTX 4090",
        "severity": "HIGH",
    },
    "no_offload_optimizer": {
        "check": lambda c: c.get("offload_optimizer") is False and c.get("zero_stage") == 2,
        "message": "ZeRO-2 without optimizer offload → 18Ψ memory → MUST offload_optimizer=True",
        "severity": "HIGH",
    },
    "moe_ep_size_gt1": {
        "check": lambda c: c.get("ep_size") and c.get("ep_size") > 1,
        "message": "ep_size>1 on single GPU → requires AllToAll → NOT viable → MUST ep_size=1",
        "severity": "CRITICAL",
    },
    "muon_cpu_offload_blocked": {
        "check": lambda c: c.get("optimizer") == "muon" and c.get("zero_stage") == 2,
        "message": "Muon+ZeRO-2 CPU offload BLOCKED (#7939 closed without merge) → MUST use cpu_adam instead",
        "severity": "CRITICAL",
    },
    "muon_global_clipping": {
        "check": lambda c: c.get("optimizer") == "muon" and c.get("gradient_clipping") != 0,
        "message": "Muon + gradient_clipping != 0 → global clipping degenerates Newton-Schulz (#7776/#5394) → MUST gradient_clipping=0 for Muon groups",
        "severity": "CRITICAL",
    },
    "zero3_peft_regression": {
        "check": lambda c: c.get("zero_stage") == 3 and c.get("lora_rank") is not None,
        "message": "ZeRO-3+PEFT regression in v0.19.2 (#8072/#8073) → TypeError in _allgather_params_coalesced → use ZeRO-2 instead",
        "severity": "HIGH",
    },
    "cppo_without_bypass": {
        "check": lambda c: c.get("policy_loss_type") == "cppo" and c.get("bypass_mode") is False,
        "message": "CPPO requires bypass_mode → mathematical necessity → divergence measured against μ",
        "severity": "CRITICAL",
    },
    "lora_rank_over_32": {
        "check": lambda c: c.get("lora_rank") is not None and c.get("lora_rank") > 32,
        "message": "LoRA rank>32 breaks EOS generation in vLLM rollout (#6782) → MUST lora_rank<=32 for GRPO!",
        "severity": "CRITICAL",
    },
}


def validate_config(user_config):
    """Validate a user config against RTX 4090 safety rules."""
    issues = []
    for rule_name, rule in VALIDATION_RULES.items():
        if rule["check"](user_config):
            issues.append({
                "rule": rule_name,
                "severity": rule["severity"],
                "message": rule["message"],
            })
    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

SEVERITY_ICONS = {"CRITICAL": "!!!", "HIGH": "!!", "MEDIUM": "!", "LOW": "."}


def show_configs(framework_filter=None):
    """Display recommended configs, optionally filtered by framework."""
    for key, cfg in RECOMMENDED_CONFIGS.items():
        if framework_filter and cfg["framework"] != framework_filter:
            continue
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}  [Rank: {cfg['rank']}]")
        print(f"{'='*60}")
        print(f"  Framework:    {cfg['framework']}")
        print(f"  Training:     {cfg['training_framework']}")
        print(f"  Rollout:      {cfg['rollout_backend']}")
        print(f"  Determinism:  {cfg['determinism_level']}")
        print(f"  Peak memory:  {cfg['memory_peak_gb']}GB (fits 24GB: {cfg['fits_24gb']}, margin: {cfg['margin_gb']}GB)")
        print(f"\n  Recommended config:")
        for k, v in cfg["config"].items():
            print(f"    {k}: {v}")
        if cfg["warnings"]:
            print(f"\n  Warnings:")
            for w in cfg["warnings"]:
                print(f"    - {w}")
        print()


def show_estimate(model_name, framework_filter=None):
    """Show memory estimates for a model across frameworks."""
    print(f"\nMemory estimates for {model_name} on RTX 4090 (24GB):")
    print(f"{'='*60}")
    for key, cfg in RECOMMENDED_CONFIGS.items():
        if framework_filter and cfg["framework"] != framework_filter:
            continue
        est = estimate_memory(model_name, key)
        if est:
            fits = "YES" if est["fits_24gb"] else "NO"
            print(f"\n  {cfg['name']}:")
            print(f"    Peak: {est['peak_memory_gb']}GB | Fits 24GB: {fits} | Margin: {est['margin_gb']}GB")
        else:
            print(f"\n  {cfg['name']}: no estimate available")


def show_all_estimates(model_name=None):
    """Show estimates for all models across all frameworks."""
    for model in MEMORY_ESTIMATES:
        if model_name and model != model_name:
            continue
        print(f"\n{'#'*60}")
        print(f"  {model}")
        print(f"{'#'*60}")
        show_estimate(model)


def run_validate(config_file=None):
    """Validate a config file or show common pitfalls."""
    if config_file:
        try:
            with open(config_file) as f:
                if config_file.endswith('.json'):
                    user_config = json.load(f)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    import yaml
                    user_config = yaml.safe_load(f)
                else:
                    print("Unsupported format. Use .json or .yaml")
                    return
        except Exception as e:
            print(f"Error reading config: {e}")
            return

        issues = validate_config(user_config)
        if not issues:
            print("No issues found — config looks safe for RTX 4090!")
        else:
            print(f"\nFound {len(issues)} issues:")
            for issue in issues:
                icon = SEVERITY_ICONS.get(issue["severity"], "?")
                print(f"  [{icon}] {issue['severity']}: {issue['message']}")
    else:
        print("\nCommon RTX 4090 GRPO pitfalls to")
        print("=" * 50)
        for rule_name, rule in VALIDATION_RULES.items():
            icon = SEVERITY_ICONS.get(rule["severity"], "?")
            print(f"  [{icon}] {rule['severity']}: {rule['message']}")


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 GRPO Config Reference")
    parser.add_argument("mode", choices=["show", "validate", "estimate"],
                        help="Mode: show configs, validate config, estimate memory")
    parser.add_argument("--framework", choices=["verl", "rLLM", "DeepSpeed"],
                        help="Filter by framework")
    parser.add_argument("--config", help="Config file to validate (.json/.yaml)")
    parser.add_argument("--model", choices=list(MEMORY_ESTIMATES.keys()),
                        help="Model to estimate memory for")

    args = parser.parse_args()

    if args.mode == "show":
        show_configs(args.framework)
    elif args.mode == "validate":
        run_validate(args.config)
    elif args.mode == "estimate":
        if args.model:
            show_estimate(args.model, args.framework)
        else:
            show_all_estimates()


if __name__ == "__main__":
    main()
