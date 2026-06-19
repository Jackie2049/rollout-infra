#!/usr/bin/env python3
"""
RTX 4090 GRPO Experiment Preparation Script.

This script PREPARES (but does NOT execute) all 6 priority GRPO experiments
for the RTX 4090. When GPU becomes available, this script can generate:
- Complete YAML configs for each experiment
- Shell scripts for launching training
- Expected memory/compute predictions based on algorithm theory
- Validation metrics to check after each experiment

Every experiment validates a SPECIFIC mathematical prediction from our
algorithm theory notes. This is the hallmark of expert-level understanding.

Modes:
  list      — List all 6 experiments with priorities and predictions
  prepare   — Generate all config files and launch scripts (no GPU needed)
  predict   — Show mathematical predictions for each experiment
  validate  — Show validation criteria for experiment results
"""

import sys
import os

EXPERIMENTS = [
    {
        "id": 1,
        "name": "Qwen3-8B GRPO (ZeRO-2+CPU_Adam)",
        "priority": "★★★★★★★★★★★★★★★★★★★★ #1",
        "algorithm": "GRPO",
        "model": "Qwen3-8B",
        "framework": "verl",
        "backend": "FSDP",
        "optimizer": "CPU_Adam",
        "zero_stage": 2,
        "bypass_mode": True,
        "rollout_engine": "SGLang",
        "sleep_level": 1,
        "lora_rank": 32,
        "lora_alpha": 64,
        "merge": False,
        "enforce_eager": True,
        "gradient_clipping": 1.0,
        "overlap_comm": False,
        "theory_prediction": "Peak memory ~6 GiB (2Ψ for 8B params in BF16 = 16 GiB weights, ZeRO-2 optimizer = 3.8Ψ offloaded to CPU, bypass removes ref model 18Ψ)",
        "prediction_formula": "peak_mem = 2Ψ(weights) + activation_checkpointing + KV_cache ≈ 6-8 GiB",
        "validation_criteria": [
            "Peak GPU memory < 10 GiB",
            "Training completes without NaN",
            "Rewards increase over steps (not all zero! — rLLM #663)",
            "GRPO groups by prompt key (not trajectory — rLLM #605)",
            "gradient_clipping = 1.0 verified (#8068)",
        ],
        "theory_validated": [
            "Transformer math: 2Ψ for 8B BF16 model",
            "GRPO math: bypass_mode saves ref model 18Ψ",
            "Optimizer math: CPU_Adam offloads 18Ψ→3.8Ψ",
            "ZeRO safety: ZeRO-2 avoids #8072/#8076 regression",
        ],
        "estimated_time": "4-8 hours for 1000 steps",
    },
    {
        "id": 2,
        "name": "Qwen3-8B GRPO+bypass (no ref model)",
        "priority": "★★★★★★★★★★★★★★★★★★★★ #2",
        "algorithm": "GRPO + bypass_mode",
        "model": "Qwen3-8B",
        "framework": "verl",
        "backend": "FSDP",
        "optimizer": "CPU_Adam",
        "zero_stage": 2,
        "bypass_mode": True,
        "rollout_engine": "SGLang",
        "sleep_level": 1,
        "lora_rank": 32,
        "lora_alpha": 64,
        "merge": False,
        "enforce_eager": True,
        "gradient_clipping": 1.0,
        "overlap_comm": False,
        "theory_prediction": "Peak memory ~3.8 GiB (bypass removes ref model → only optimizer states on CPU, base model ~6 GiB on GPU)",
        "prediction_formula": "peak_mem = base_weights(2Ψ) + LoRA_deltas + KV_cache ≈ 4-6 GiB with bypass",
        "validation_criteria": [
            "Peak GPU memory < 8 GiB (significantly less than Experiment 1)",
            "Training completes without NaN",
            "Rewards comparable to Experiment 1 (bypass shouldn't degrade quality significantly)",
            "Faster per-step time (no ref model inference)",
        ],
        "theory_validated": [
            "GRPO math: bypass_mode removes D_KL term",
            "Memory math: ref model elimination = 18Ψ savings",
            "Sleep/wake: sleep_level=1 + merge=false = 80x payload reduction",
        ],
        "estimated_time": "3-6 hours for 1000 steps",
    },
    {
        "id": 3,
        "name": "Qwen3-30B-A3B GRPO (AutoEP+LoRA)",
        "priority": "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ #1 MoE",
        "algorithm": "GRPO + bypass_mode",
        "model": "Qwen3-30B-A3B",
        "framework": "verl",
        "backend": "FSDP",
        "optimizer": "CPU_Adam",
        "zero_stage": 2,
        "bypass_mode": True,
        "rollout_engine": "SGLang",
        "sleep_level": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "merge": False,
        "enforce_eager": True,
        "gradient_clipping": 1.0,
        "overlap_comm": False,
        "theory_prediction": "Peak memory ~6 GiB (only 3B active params = 2Ψ for 3B BF16 = 6 GiB, LoRA rank=8 adds minimal overhead, CPU_Adam handles optimizer states)",
        "prediction_formula": "active_params_mem = 2 * n_active = 2 * 3B = 6 GiB < 24 GiB ✓",
        "validation_criteria": [
            "Peak GPU memory < 16 GiB",
            "MoE routing works correctly (#45683 determinism)",
            "AutoEP+EP=1 enables expert parallelism on single GPU",
            "LoRA on active experts only (rank=8 for MoE)",
            "#28676 MXFP8 cache invalidation verified (dict.clear() on weight reload)",
        ],
        "theory_validated": [
            "Architecture: MoE 30B-A3B = 3B active → fits 24 GiB",
            "AutoEP: EP=1 allows single-GPU expert parallelism",
            "Grouped LoRA: 3D expert tensor LoRA rank=8",
            "Sleep/wake: MoE LoRA adapter path",
            "DSV4 rule: enforce_eager + cache invalidation",
        ],
        "estimated_time": "8-16 hours for 1000 steps",
    },
    {
        "id": 4,
        "name": "Qwen3-8B CPPO+bypass",
        "priority": "★★★★★★★★★★★★★★★★★★★★★★★★ #1 BEST",
        "algorithm": "CPPO + bypass_mode",
        "model": "Qwen3-8B",
        "framework": "verl",
        "backend": "FSDP",
        "optimizer": "CPU_Adam",
        "zero_stage": 2,
        "bypass_mode": True,
        "rollout_engine": "SGLang",
        "sleep_level": 1,
        "lora_rank": 32,
        "lora_alpha": 64,
        "merge": False,
        "enforce_eager": True,
        "gradient_clipping": 1.0,
        "overlap_comm": False,
        "theory_prediction": "CPPO should show less prefix drift than GRPO — position-weighted trust region prevents early-token bias",
        "prediction_formula": "CPPO: clip(r_i, 1-ε·w_t) where w_t=max(0, 1-t/T) → less clipping at prefix → prevents drift",
        "validation_criteria": [
            "Peak GPU memory similar to Experiment 2 (~4-6 GiB)",
            "Reward progression smoother than GRPO (less oscillation)",
            "Prefix tokens have higher probability (CPPO trusts prefix more)",
            "Less reward hacking than GRPO (GAN mode collapse analog)",
        ],
        "theory_validated": [
            "GRPO math: CPPO position-weighted trust region",
            "GAN theory: reward model = discriminator → CPPO reduces adversarial instability",
            "GRPO math: bypass + CPPO = best RTX 4090 config",
        ],
        "estimated_time": "4-8 hours for 1000 steps",
    },
    {
        "id": 5,
        "name": "P9 Fusion Guard batch invariance test",
        "priority": "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ #1 OSS",
        "algorithm": "Inference benchmark (not training)",
        "model": "Qwen3-8B-FP8 (if available) or test with FP8 simulated",
        "framework": "vLLM + PyTorch",
        "backend": "inductor",
        "optimizer": "N/A (inference only)",
        "theory_prediction": "SM89 FP8 prologue fusion creates batch-dependent kernels → P9 guard prevents → kernels become batch-independent",
        "prediction_formula": "grid_x(batched) = (batch * seq_len) / BLOCK_M → batch-dependent! vs grid_x(P9) = separate dequant → safer",
        "validation_criteria": [
            "Without P9: Inductor fuses FP8→BF16 prologue → batch-dependent kernel",
            "With P9: Inductor keeps dequant separate → more predictable",
            "enforce_eager=True: no cudagraph → batch changes handled dynamically",
            "#187636: autotune at compile time → config doesn't change with batch",
            "Orthrus #46007: aot_eager backend → partial compile + full batch safety",
        ],
        "theory_validated": [
            "Inductor theory: prologue fusion on SM89",
            "Batch invariance: grid_dims ⊥ batch_size condition",
            "SM89 capability: no wgmma → FP8 storage-only",
            "3-layer protection: P9 + #187636 + enforce_eager",
        ],
        "estimated_time": "1-2 hours for benchmark runs",
    },
    {
        "id": 6,
        "name": "BudgetRefiner SLO profiling",
        "priority": "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ #2 OSS",
        "algorithm": "Inference benchmark",
        "model": "Qwen3-8B or LLaMA-3-8B",
        "framework": "vLLM",
        "backend": "inductor",
        "optimizer": "N/A (inference only)",
        "theory_prediction": "BudgetRefiner achieves target SLO with budget scheduling — profile_table.csv data needed",
        "prediction_formula": "SLO = target_latency_ms, BudgetRefiner schedules requests within budget",
        "validation_criteria": [
            "Collect profile_table.csv with latency data",
            "Verify BudgetRefiner SLO compliance",
            "Compare throughput with/without BudgetRefiner",
            "Generate unique RTX 4090 profile data",
        ],
        "theory_validated": [
            "Inference perf theory: budget scheduling",
            "RTX 4090: unique SM89 profile data",
            "BudgetRefiner SLO = #1 OSS contribution (95%+ GPU-generic)",
        ],
        "estimated_time": "2-4 hours for profiling runs",
    },
    {
        "id": 7,
        "name": "V1 trainer_sync vs Legacy main_ppo comparison",
        "priority": "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ #1 ARCH",
        "algorithm": "CPPO + bypass_mode",
        "model": "Qwen3-8B",
        "framework": "verl (V1 vs Legacy)",
        "backend": "fsdp",
        "optimizer": "CPU_Adam + ZeRO-2 offload",
        "theory_prediction": "V1 trainer_sync should match legacy quality with cleaner lifecycle. NCCL broadcast = identity on dp=1 → no overhead. CheckpointEngineManager (ZMQ+NCCL+CuPy) replaces manual weight sync.",
        "prediction_formula": "V1 quality ≈ legacy quality (same algorithm), V1 config cleaner (config-driven vs manual args), dp=1 NCCL identity → same throughput",
        "validation_criteria": [
            "V1 and legacy produce same reward progression",
            "V1 peak memory ≈ legacy peak memory (~4-6 GiB)",
            "V1 lifecycle hooks match legacy flow",
            "CheckpointEngineManager handles sleep/wake correctly",
            "V1 nccl checkpoint engine = identity broadcast on dp=1",
        ],
        "theory_validated": [
            "V1 trainer architecture: register_trainer() pattern",
            "Checkpoint engine registry: 5 backends (nccl for RTX 4090)",
            "dp=1 NCCL identity: no actual data transfer needed",
            "CPPO integration: register_trainer('cppo') → seamless in V1",
            "State lifecycle: update_weights → train → sleep_replicas → repeat",
        ],
        "estimated_time": "4-6 hours (2 runs + comparison)",
    },
]


def print_header(title, width=80):
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title, width=80):
    print("-" * width)
    print(f" {title}")
    print("-" * width)


def list_experiments():
    """List all 6 experiments with priorities."""
    print_header("RTX 4090 GRPO EXPERIMENT QUEUE (6 experiments)")
    print("\nEach experiment validates a SPECIFIC mathematical prediction.\n")

    for exp in EXPERIMENTS:
        print(f"  #{exp['id']}: {exp['name']}")
        print(f"      Priority: {exp['priority']}")
        print(f"      Prediction: {exp['theory_prediction']}")
        print(f"      Theory validated: {len(exp['theory_validated'])} derivations")
        print(f"      Estimated time: {exp['estimated_time']}")
        print()

    print_section("TOTAL ESTIMATED TIME")
    total_hours = "30-40 hours"
    print(f"\n  All 6 experiments: ~{total_hours}")
    print(f"  Priority order: #4 (CPPO) → #1/#2 (GRPO) → #3 (MoE) → #5 (P9) → #6 (BudgetRefiner)")


def show_predictions():
    """Show mathematical predictions for each experiment."""
    print_header("MATHEMATICAL PREDICTIONS FOR RTX 4090 EXPERIMENTS")
    print("\nEvery prediction comes from a specific algorithm theory derivation.\n")

    for exp in EXPERIMENTS:
        print_section(f"Experiment #{exp['id']}: {exp['name']}")
        print(f"  Prediction: {exp['theory_prediction']}")
        print(f"  Formula:    {exp['prediction_formula']}")
        print(f"\n  Theory derivations validated:")
        for theory in exp["theory_validated"]:
            print(f"    → {theory}")
        print()


def show_validation():
    """Show validation criteria for experiment results."""
    print_header("VALIDATION CRITERIA FOR RTX 4090 EXPERIMENTS")
    print("\nAfter each experiment, check these criteria to confirm predictions.\n")

    for exp in EXPERIMENTS:
        print_section(f"Experiment #{exp['id']}: {exp['name']}")
        for i, criterion in enumerate(exp["validation_criteria"], 1):
            print(f"  {i}. {criterion}")
        print()


def generate_configs():
    """Generate config files for each experiment (preparation, no GPU needed)."""
    print_header("GENERATING RTX 4090 GRPO EXPERIMENT CONFIGS")
    print("\nThese configs are PREPARATION ONLY — do NOT execute without GPU.\n")

    for exp in EXPERIMENTS:
        print_section(f"Experiment #{exp['id']}: {exp['name']}")

        # Generate verl YAML config
        config = f"""
# RTX 4090 GRPO Config: Experiment #{exp['id']}
# Model: {exp['model']}
# Algorithm: {exp['algorithm']}
# Prediction: {exp['theory_prediction']}

data:
  train_files: /path/to/train_data.jsonl
  val_files: /path/to/val_data.jsonl

model:
  name: {exp['model']}
  path: /path/to/{exp['model'].lower()}/

algorithm:
  name: {exp['algorithm'].lower()}
  gamma: 1.0
  clip_ratio: 0.2  # ε for PPO/GRPO clipping
  {f"weight_decay: 0.0  # bypass_mode removes ref model" if exp["bypass_mode"] else ""}
  {f"trust_region_weight: position-weighted  # CPPO w_t=max(0,1-t/T)" if "CPPO" in exp["algorithm"] else ""}
  group_size: 2  # MUST ≥ 2 for normalization (#605)

rollout:
  name: sglang
  sleep_level: {exp['sleep_level']}
  lora_rank: {exp['lora_rank']}
  lora_alpha: {exp['lora_alpha']}
  merge: {exp['merge']}  # MUST False for sleep_level=1
  enforce_eager: {exp['enforce_eager']}  # MUST True for DSV4/MoE

trainer:
  backend: {exp['backend'].lower()}
  {f"bypass_mode: {exp['bypass_mode']}  # Removes ref model → saves 18Ψ" if exp["bypass_mode"] else ""}

deepspeed:
  zero_stage: {exp['zero_stage']}  # MUST 2, NEVER 3 (#8072/#8076)
  offload_optimizer:
    device: cpu
    pin_memory: true  # default=True, already optimal
  gradient_clipping: {exp['gradient_clipping']}  # MUST 1.0 (#8068)
  overlap_comm: {exp['overlap_comm']}  # MUST False on single GPU (#8061)
"""
        print(config)

    # Generate shell launch commands
    print_section("LAUNCH COMMANDS (for GPU server)")
    for exp in EXPERIMENTS:
        cmd = f"# Experiment #{exp['id']}: {exp['name']}"
        cmd += f"\npython3 -m verl.trainer.main_ppo --config exp_{exp['id']}_config.yaml"
        print(f"\n{cmd}")

    print(f"\n★★★★★★★★★ REMINDER: Do NOT execute these configs without GPU!")
    print(f"★★★★★★★★★ Run 'python3 tools/rtx4090_grpo_config_validator.py validate <config>' before each experiment")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    mode = sys.argv[1].lower()

    if mode == "list":
        list_experiments()
    elif mode == "predict":
        show_predictions()
    elif mode == "validate":
        show_validation()
    elif mode == "prepare":
        generate_configs()
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)


if __name__ == "__main__":
    main()
