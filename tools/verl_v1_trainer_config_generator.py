#!/usr/bin/env python3
"""
verl V1 Trainer Config Generator for RTX 4090

Generates YAML configs for verl's new V1 unified trainer architecture.
The V1 trainer uses register_trainer() pattern with 3 types:
  - sync: colocated, synchronous (RTX 4090 BEST)
  - colocate_async: colocated, asynchronous (experimental on single GPU)
  - separate_async: separated, asynchronous (multi-GPU only, NOT viable on RTX 4090)

Based on 7-framework deep research (11 DSV4 failures, 10 MUST DO + 10 MUST NOT rules).

Usage:
  python verl_v1_trainer_config_generator.py generate <trainer_type> <model_name>
  python verl_v1_trainer_config_generator.py compare
  python verl_v1_trainer_config_generator.py validate <config.yaml>
  python verl_v1_trainer_config_generator.py rtx4090
"""

import argparse
import json
import sys

# ============================================================
# V1 TRAINER ARCHITECTURE DATA
# ============================================================

V1_TRAINERS = {
    "sync": {
        "name": "PPOTrainerSync",
        "description": "Synchronous PPO — trainer+rollout colocated, no partial rollout",
        "rtx4090_viable": True,
        "rtx4090_ranking": "★★★★★★★★ #1 BEST for RTX 4090",
        "lifecycle_hooks": {
            "on_init_end": "update_weights (load checkpoint)",
            "on_train_begin": "none (no warmup batches)",
            "on_step_end": "update_weights (weight sync after training)",
            "on_sample_end": "sleep_replicas (discard weights + KV cache)",
        },
        "min_gpu_count": 1,
        "weight_sync_frequency": "every_step",
        "partial_rollout": False,
        "recommended_for": "single GPU, RTX 4090, dp=1",
    },
    "colocate_async": {
        "name": "PPOTrainerColocateAsync",
        "description": "Async PPO — trainer+rollout colocated, partial rollout enabled",
        "rtx4090_viable": True,  # experimental
        "rtx4090_ranking": "★★★★★★★★ experimental, may work on single GPU",
        "lifecycle_hooks": {
            "on_init_end": "update_weights (load checkpoint)",
            "on_train_begin": "num_warmup_batches (pre-fill rollout queue)",
            "on_step_end": "update_weights + resume_generation",
            "on_sample_end": "abort_replicas + sleep_replicas",
        },
        "min_gpu_count": 1,
        "weight_sync_frequency": "every_step",
        "partial_rollout": True,
        "recommended_for": "single GPU with throughput optimization, future exploration",
    },
    "separate_async": {
        "name": "PPOTrainerSeparateAsync",
        "description": "Async PPO — trainer+rollout separated, partial rollout enabled",
        "rtx4090_viable": False,
        "rtx4090_ranking": "NOT viable on RTX 4090 dp=1",
        "lifecycle_hooks": {
            "on_init_end": "update_weights",
            "on_train_begin": "num_warmup_batches",
            "on_step_end": "update_weights + resume_generation",
            "on_sample_end": "abort_replicas + sleep_replicas",
        },
        "min_gpu_count": 2,  # NEEDS at least 2 GPUs (1 trainer + 1 rollout)
        "weight_sync_frequency": "every_step",
        "partial_rollout": True,
        "recommended_for": "multi-GPU, multi-node, NOT single GPU",
    },
}

CHECKPOINT_ENGINES = {
    "naive": {
        "backend": "ColocatedCheckpointEngine",
        "rtx4090": True,
        "rtx4090_ranking": "★★★★★★★★ BEST for RTX 4090 single-GPU (sync trainer)",
        "description": "In-process Python yield — zero IPC overhead, no process groups",
        "transport": "Python generator yield",
        "memory": "0 extra (just Python reference)",
        "forced_by": "sync trainer automatically sets backend=naive",
        "platform": "Any (CPU, GPU, NPU)",
    },
    "nccl": {
        "backend": "NCCLCheckpointEngine",
        "rtx4090": True,
        "rtx4090_ranking": "Good for multi-GPU, overkill for single GPU",
        "description": "NCCL broadcast + ZeroMQ PUB/SUB metadata",
        "transport": "NCCL collective broadcast + ZeroMQ",
        "memory": "2 * bucket_size (send_buf + recv_buf), cupy on master",
        "forced_by": "separate_async requires non-naive backend",
        "platform": "NVIDIA GPU only",
    },
    "hccl": {
        "backend": "HCCLCheckpointEngine",
        "rtx4090": False,
        "rtx4090_ranking": "NOT usable on RTX 4090 — requires torch.npu",
        "description": "HCCL collective broadcast + ZeroMQ — Ascend NPU only",
        "transport": "HCCL broadcast + ZeroMQ",
        "memory": "2 * bucket_size",
        "forced_by": "Ascend NPU deployments only",
        "platform": "Ascend NPU only (torch.npu)",
    },
    "nixl": {
        "backend": "NIXLCheckpointEngine",
        "rtx4090": False,
        "rtx4090_ranking": "NOT viable — requires RDMA NIC (not on consumer RTX 4090)",
        "description": "NIXL p2p (RDMA/UCX/UCCL/Mooncake) + ZeroMQ — ring topology",
        "transport": "NIXL p2p RDMA/UCX + ZeroMQ",
        "memory": "2 * bucket_size, cupy or CPU pinned",
        "forced_by": "separate_async requires non-naive backend",
        "platform": "GPU + RDMA-capable NICs",
    },
    "kimi": {
        "backend": "KimiCheckpointEngine",
        "rtx4090": False,
        "rtx4090_ranking": "Complex, multi-GPU only — uses external checkpoint_engine package",
        "description": "ParameterServer + distributed collective — H2DBucket",
        "transport": "ParameterServer + distributed",
        "memory": "H2DBucket (host-to-device)",
        "forced_by": "separate_async requires non-naive backend",
        "platform": "NVIDIA GPU, multi-GPU",
    },
    "mooncake": {
        "backend": "MooncakeCheckpointEngine",
        "rtx4090": False,
        "rtx4090_ranking": "NOT viable — requires RDMA or Ascend Direct",
        "description": "Mooncake TransferEngine p2p RDMA — supports Ascend NPU via ascend_direct",
        "transport": "Mooncake TransferEngine p2p RDMA",
        "memory": "2 * bucket_size + 4KB magic_buf",
        "forced_by": "separate_async requires non-naive backend",
        "platform": "GPU with RDMA or Ascend NPU",
    },
}

# ============================================================
# MODEL PROFILES FOR RTX 4090
# ============================================================

RTX4090_MODELS = {
    "qwen3-8b": {
        "full_name": "Qwen/Qwen3-8B",
        "size_b": 8,
        "params_b": 8.19,
        "peak_gpu_gib": 6,
        "peak_gpu_gib_bypass": 4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "algorithm": "grpo",
        "best_algorithm": "cppo",
        "dynamic_routing": 0,
        "enforce_eager": True,  # safety rule
        "zero_stage": 2,
        "optimizer": "CPU_Adam",
        "lr": 1e-6,
    },
    "qwen3-30b-a3b": {
        "full_name": "Qwen/Qwen3-30B-A3B",
        "size_b": 30,
        "params_b": 30.6,  # 3B active with MoE
        "peak_gpu_gib": 6,
        "peak_gpu_gib_bypass": 4,
        "lora_rank": 8,  # Grouped LoRA for MoE (163x reduction!)
        "lora_alpha": 16,
        "algorithm": "grpo",
        "best_algorithm": "cppo",
        "dynamic_routing": 1,  # MoE expert routing
        "enforce_eager": True,
        "zero_stage": 2,
        "optimizer": "CPU_Adam",
        "lr": 5e-7,
    },
}

# ============================================================
# MUST DO / MUST NOT RULES (from 7-framework research)
# ============================================================

MUST_DO = [
    ("ZeRO-2 (NEVER ZeRO-3)", "#8072/#8076 dtype mismatch regression"),
    ("CPU_Adam optimizer offload", "18Ψ→3.8Ψ, only viable optimizer"),
    ("gradient_clipping=1.0", "#8068 default 0→1.0 regression"),
    ("enforce_eager=True", "11 DSV4 failures across 4 frameworks"),
    ("bypass_mode=True", "removes ref model → 18Ψ→3.8Ψ"),
    ("LoRA rank=32 (NOT 64)", "#6782 rank=64 breaks EOS"),
    ("overlap_comm=False", "#8061 NaN on single GPU"),
    ("cosine decay + warmup", "standard LR schedule"),
    ("group_size≥2", "#605 normalization undefined at |G|=1 → ALL frameworks degenerate to REINFORCE!"),
    ("reset_prefix+encoder_cache after weight update", "#45093/#46125 stale cache = silent corruption in RLHF"),
    ("FSDP1 (NOT FSDP2)", "#6468 FSDP2 CPU memory leak 0.6-6.3 GiB/step"),
    ("ulimit -n 65536", "#8075 fd leak safety"),
]

MUST_NOT = [
    ("ZeRO-3 on single GPU", "#8072/#8076 regression"),
    ("Muon optimizer", "6 blockers across 3 frameworks"),
    ("LoRA rank=64", "#6782 breaks EOS"),
    ("overlap_comm=True", "#8061 NaN confirmed"),
    ("CUDA graphs for DSV4", "11 failures"),
    ("NVMe offload", "#8075 fd leak"),
    ("autocast_adapter_dtype+ZeRO-3", "#8072 fp32 LoRA mismatch"),
    ("separate_async trainer on RTX 4090", "multi-GPU only"),
    ("Megatron backend for verl", "#6699 detach not upstream"),
    ("vLLM-Ascend backend", "sleep_level=1 NOT supported"),
]

# ============================================================
# CONFIG GENERATION
# ============================================================

def generate_config(trainer_type, model_name):
    """Generate YAML config for V1 trainer + model on RTX 4090."""

    if trainer_type not in V1_TRAINERS:
        print(f"ERROR: Unknown trainer type '{trainer_type}'. Available: sync, colocate_async, separate_async")
        sys.exit(1)

    if model_name not in RTX4090_MODELS:
        print(f"ERROR: Unknown model '{model_name}'. Available: {', '.join(RTX4090_MODELS.keys())}")
        sys.exit(1)

    trainer = V1_TRAINERS[trainer_type]
    model = RTX4090_MODELS[model_name]

    if not trainer["rtx4090_viable"]:
        print(f"ERROR: Trainer '{trainer_type}' NOT viable on RTX 4090!")
        print(f"  Reason: requires {trainer['min_gpu_count']}+ GPUs, RTX 4090 is single GPU")
        sys.exit(1)

    config = {
        "trainer": {
            "v1": {
                "type": trainer_type,
                "backend": "fsdp",
            },
        },
        "algorithm": {
            "name": model["best_algorithm"],
            "bypass_mode": True,
            "clip_ratio": 0.2,
            "group_size": 2,
            # CPPO-specific (if algorithm=cppo)
            "position_weight": True,  # w_t = max(0, 1 - t/T)
        },
        "model": {
            "path": f"/root/rollout-infra/models/{model_name}",
            "name": model["full_name"],
            "max_seq_len": 4096,
        },
        "rollout": {
            "name": "sglang",
            "sleep_level": 1,
            "lora_rank": model["lora_rank"],
            "lora_alpha": model["lora_alpha"],
            "merge": False,
            "enforce_eager": True,
        },
        "checkpoint_engine": {
            "type": "nccl",
        },
        "deepspeed": {
            "config": {
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True,
                    },
                    "overlap_comm": False,
                    "gradient_clipping": 1.0,
                },
                "gradient_accumulation_steps": 1,
                "train_micro_batch_size_per_gpu": 1,
                "optimizer": {
                    "type": "CPUAdam",
                    "params": {
                        "lr": model["lr"],
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    },
                },
                "scheduler": {
                    "type": "CosineWithWarmup",
                    "params": {
                        "warmup_steps": 10,
                        "max_steps": 500,
                    },
                },
                "bf16": {"enabled": True},
            },
        },
        "data": {
            "train_files": "/root/rollout-infra/data/gsm8k/train.jsonl",
        },
        "output_dir": f"/root/rollout-infra/experiments/v1_{trainer_type}_{model_name}",
        "max_steps": 500,
        "lr": model["lr"],
        "lr_scheduler": "cosine",
    }

    # Add trainer-specific config
    if trainer_type == "colocate_async":
        config["trainer"]["v1"]["colocate_async"] = {
            "num_warmup_batches": 2,
        }

    return config


def show_config(config):
    """Pretty-print config as YAML-like format."""
    print("# ★★★★★★★★ RTX 4090 V1 Trainer Config ★★★★★★★★")
    print(f"# Trainer: {config['trainer']['v1']['type']}")
    print(f"# Algorithm: {config['algorithm']['name']} + bypass_mode")
    print(f"# Model: {config['model']['name']}")
    print(f"# Based on 7-framework research (11 DSV4 failures, 10 MUST DO + 10 MUST NOT)")
    print()

    import yaml
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))


def compare_trainers():
    """Compare V1 trainer types for RTX 4090."""
    print("=" * 80)
    print(" verl V1 Trainer Comparison for RTX 4090")
    print("=" * 80)
    print()

    print(f"{'Trainer':<20} {'RTX 4090':>12} {'GPU Count':>10} {'Partial':>10} {'Ranking':>20}")
    print("-" * 80)

    for name, data in V1_TRAINERS.items():
        viable = "✓" if data["rtx4090_viable"] else "✗"
        print(f"{name:<20} {viable:>12} {data['min_gpu_count']:>10} {str(data['partial_rollout']):>10} {data['rtx4090_ranking']:>20}")

    print()
    print("★★★★★★★★★ RTX 4090 #1 BEST: trainer_sync (simplest, safest, production-tested)")
    print("★★★★★★★★★ Experimental: trainer_colocate_async (may improve throughput)")
    print("★★★★★★★★★ NOT viable: trainer_separate_async (multi-GPU only)")
    print()

    print("Checkpoint Engines:")
    for name, data in CHECKPOINT_ENGINES.items():
        viable = "✓" if data["rtx4090"] else "✗"
        print(f"  {name:<12} {viable:>5} {data['description']}")

    print()
    print("★★★★★★★★★ RTX 4090 uses nccl_checkpoint_engine (standard CUDA)")

    print()
    print("=" * 80)
    print(" Lifecycle Hook Comparison")
    print("=" * 80)
    print()

    hooks = ["on_init_end", "on_train_begin", "on_step_end", "on_sample_end"]
    print(f"{'Hook':<20} {'sync':<30} {'colocate_async':<30} {'separate_async':<30}")
    print("-" * 100)
    for hook in hooks:
        row = []
        for trainer_name in ["sync", "colocate_async", "separate_async"]:
            row.append(V1_TRAINERS[trainer_name]["lifecycle_hooks"][hook])
        print(f"{hook:<20} {row[0]:<30} {row[1]:<30} {row[2]:<30}")


def validate_config(config_path):
    """Validate a V1 config file against RTX 4090 safety rules."""
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Cannot read config file: {e}")
        sys.exit(1)

    errors = []
    warnings = []

    # Check trainer type
    trainer_type = config.get("trainer", {}).get("v1", {}).get("type", "unknown")
    if trainer_type == "separate_async":
        errors.append("trainer_separate_async NOT viable on RTX 4090 (requires multi-GPU)")
    elif trainer_type not in V1_TRAINERS:
        errors.append(f"Unknown trainer type: {trainer_type}")

    # Check MUST NOT rules
    ds_config = config.get("deepspeed", {}).get("config", {})
    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    if zero_stage == 3:
        errors.append("ZeRO-3 on single GPU (#8072/#8076 regression)")

    overlap_comm = ds_config.get("zero_optimization", {}).get("overlap_comm", True)
    if overlap_comm:
        errors.append("overlap_comm=True (#8061 NaN on single GPU)")

    grad_clip = ds_config.get("zero_optimization", {}).get("gradient_clipping", 0)
    if grad_clip != 1.0:
        errors.append(f"gradient_clipping={grad_clip} (#8068 MUST be 1.0)")

    # Check MUST DO rules
    optimizer = ds_config.get("optimizer", {}).get("type", "")
    if optimizer != "CPUAdam":
        warnings.append(f"optimizer={optimizer} (CPU_Adam recommended for 18Ψ→3.8Ψ)")

    bypass = config.get("algorithm", {}).get("bypass_mode", False)
    if not bypass:
        warnings.append("bypass_mode=False (ref model consumes 18Ψ extra)")

    enforce_eager = config.get("rollout", {}).get("enforce_eager", False)
    if not enforce_eager:
        errors.append("enforce_eager=False (11 DSV4 failures, MUST True)")

    lora_rank = config.get("rollout", {}).get("lora_rank", 64)
    if lora_rank == 64:
        errors.append("lora_rank=64 (#6782 breaks EOS, MUST use 32)")

    group_size = config.get("algorithm", {}).get("group_size", 1)
    if group_size < 2:
        errors.append("group_size<2 (#605 normalization undefined)")

    # Check checkpoint engine
    checkpoint_engine = config.get("checkpoint_engine", {}).get("type", "nccl")
    if checkpoint_engine != "nccl":
        warnings.append(f"checkpoint_engine={checkpoint_engine} (nccl recommended for RTX 4090)")

    backend = config.get("trainer", {}).get("v1", {}).get("backend", "")
    if backend != "fsdp":
        errors.append(f"backend={backend} (FSDP ONLY backend with detach fix #6699)")

    # Print results
    print("=" * 80)
    print(f" V1 Config Validation: {config_path}")
    print("=" * 80)
    print(f"\nTrainer type: {trainer_type}")
    print(f"Backend: {backend}")
    print(f"ZeRO stage: {zero_stage}")
    print(f"Optimizer: {optimizer}")
    print()

    if errors:
        print("★★★★★★★★★ ERRORS (MUST FIX before running on RTX 4090):")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
    else:
        print("✓ No errors found — config passes RTX 4090 safety checks!")

    if warnings:
        print("\n⚠ WARNINGS (recommended improvements):")
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
    else:
        print("✓ No warnings — config is optimal for RTX 4090!")


def show_rtx4090():
    """Show complete RTX 4090 best config."""
    config = generate_config("sync", "qwen3-8b")
    show_config(config)

    print()
    print("=" * 80)
    print(" RTX 4090 MUST DO Rules (10 rules with mathematical proof)")
    print("=" * 80)
    for i, (rule, evidence) in enumerate(MUST_DO, 1):
        print(f"  {i:2d}. {rule}")
        print(f"      Evidence: {evidence}")

    print()
    print("=" * 80)
    print(" RTX 4090 MUST NOT Rules (10 rules with mathematical proof)")
    print("=" * 80)
    for i, (rule, evidence) in enumerate(MUST_NOT, 1):
        print(f"  {i:2d}. {rule}")
        print(f"      Evidence: {evidence}")

    print()
    print("=" * 80)
    print(" V1 Trainer Architecture Summary")
    print("=" * 80)
    print()
    print("  ★★★★★★★★ V1 = verl's future unified trainer")
    print("  ★★★★★★★★ trainer_sync = RTX 4090 #1 BEST")
    print("  ★★★★★★★★ CPPO can register via register_trainer('cppo')")
    print("  ★★★★★★★★ nccl_checkpoint_engine = RTX 4090 default")
    print()
    print("  Legacy: main_ppo.py (maintenance mode)")
    print("  Future: V1 trainer (production-ready)")
    print()
    print("  Migration path: main_ppo → V1 trainer_sync → V1 trainer_colocate_async")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="verl V1 Trainer Config Generator for RTX 4090")
    parser.add_argument("command", choices=["generate", "compare", "validate", "rtx4090"],
                        help="Command to run")
    parser.add_argument("args", nargs="*",
                        help="Arguments: trainer_type model_name (generate), config.yaml (validate)")

    args = parser.parse_args()

    if args.command == "generate":
        if len(args.args) < 2:
            print("Usage: generate <trainer_type> <model_name>")
            print(f"  Trainer types: {', '.join(V1_TRAINERS.keys())}")
            print(f"  Models: {', '.join(RTX4090_MODELS.keys())}")
            sys.exit(1)
        config = generate_config(args.args[0], args.args[1])
        show_config(config)

    elif args.command == "compare":
        compare_trainers()

    elif args.command == "validate":
        if not args.args:
            print("Usage: validate <config.yaml>")
            sys.exit(1)
        validate_config(args.args[0])

    elif args.command == "rtx4090":
        show_rtx4090()


if __name__ == "__main__":
    main()
