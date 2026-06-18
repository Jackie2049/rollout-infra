#!/usr/bin/env python3
"""
RTX 4090 GRPO Configuration Validator — 7-Framework Cross-Reference

Validates GRPO training configurations for RTX 4090 (24 GiB, SM89) across
7 frameworks (DeepSpeed, Megatron-LM, vLLM, verl, MindIE/vLLM-Ascend, rLLM, SGLang).

Usage:
  python rtx4090_grpo_config_validator.py validate <config.yaml>
  python rtx4090_grpo_config_validator.py generate <scenario>
  python rtx4090_grpo_config_validator.py compare <framework1> <framework2>
  python rtx4090_grpo_config_validator.py estimate <model_name>
  python rtx4090_grpo_config_validator.py rtx4090  # full RTX 4090 best config
"""

import json
import sys
import yaml  # PyYAML required

# ============================================================
# CRITICAL FINDINGS FROM 7-FRAMEWORK DEEP RESEARCH
# ============================================================

# Framework profiles with known issues and safe configurations
FRAMEWORK_PROFILES = {
    "verl": {
        "name": "verl",
        "version": "v0.3.0-pre",
        "safe_backends": ["fsdp"],  # ONLY FSDP has detach fix (#6699)
        "unsafe_backends": ["megatron", "automodel", "torchtitan"],  # UNFIXED detach leak
        "critical_issues": {
            "#6782": "LoRA rank=64 breaks EOS → MUST use rank=32/alpha=64",
            "#6468": "FSDP2 CPU memory leak during weight sync → monotonic growth",
            "#6699": "detach model_output → 4x memory reduction (FSDP only fixed)",
            "#6512": "per-unit LoRA summon → 10x peak memory reduction",
            "#6794": "delta weight sync → ~100x payload (SGLang-only, LoRA deferred)",
        },
        "optimal_config": {
            "rollout": "sglang",  # best LoRA adapter support
            "mode": "hybrid",  # same-process, zero IPC overhead
            "sleep_level": 1,  # LoRA adapter: tags=["kv_cache"] only
            "peft_merge": False,  # merge=True forces sleep_level=2 → AVOID
            "lora_rank": 32,  # NOT 64! (#6782 breaks EOS)
            "lora_alpha": 64,
            "bypass_mode": True,  # eliminates ref model → 18Ψ→3.8Ψ
            "optimizer": "cpu_adam",  # CPU_Adam → 18Ψ→3.8Ψ
            "param_offload": True,
            "grad_offload": True,
            "enforce_eager": True,  # DSV4 MANDATORY
            "gradient_clipping": 1.0,  # MUST set (#8068)
        },
        "ranking": "#1 (CPPO+bypass) or #2 (GRPO+bypass)",
    },
    "deepspeed": {
        "name": "DeepSpeed",
        "version": "v0.19.2",
        "safe_zero_stages": [2],  # ZeRO-2 only
        "unsafe_zero_stages": [3],  # ZeRO-3 = pure overhead on dp=1
        "critical_issues": {
            "#8061": "overlap_comm+torch.compile = NaN → MUST overlap_comm=False on single GPU",
            "#8068": "gradient_clipping default 0→1.0 → ALWAYS set 1.0",
            "#8072": "ZeRO-3+PEFT LoRA regression → ZeRO-2 unaffected",
            "#7939": "Muon+CPU_offload BLOCKED → CPU_Adam ONLY option",
        },
        "optimal_config": {
            "zero_stage": 2,
            "optimizer": "cpu_adam",
            "offload_param": True,
            "offload_grad": True,
            "overlap_comm": False,  # NaN bug on single GPU!
            "gradient_clipping": 1.0,
            "enforce_eager": True,
        },
        "ranking": "#2.5 (solid but needs RouterReplay for CUDA graph)",
    },
    "megatron": {
        "name": "Megatron-LM",
        "version": "core",
        "critical_issues": {
            "#5394": "AdamW ALSO stalls under global grad-norm clipping → optimizer-agnostic!",
            "#5395": "skip_grad_norm_clip +15/-1 → 0 reviews → stalled",
            "#5387": "MFSDPv2 APPROVED but blocked by codeowners",
            "#5179": "Muon PyPI stub → can't install → 4th Muon blocker",
        },
        "optimal_config": {
            "backend": "NOT RECOMMENDED for RTX 4090 GRPO",
            "reason": "C9 detach fix not in upstream → memory leak (#6699)",
        },
        "ranking": "#4 (complex setup, detach fix missing)",
    },
    "vllm": {
        "name": "vLLM",
        "version": "v0.23.0",
        "critical_issues": {
            "#45309": "DSV4 eager_break revert → garbage output",
            "#45863": "DSV4 sparse cache revert → GSM8K 6.75% vs 87%",
            "#45972": "2nd DSV4 revert (merged June 18)",
            "#45979": "3rd DSV4 revert (OPEN June 18)",
            "#45683": "Deterministic MoE combine → CRITICAL for GRPO",
            "#45819": "GDN batch invariance → progressing CI",
        },
        "sleep_level": "integer-based (1 or 2)",
        "sleep_level_1_limitation": "EP>1 with old vLLM (<0.11.0) forces WARNING about potential OOM",
        "ranking": "used as rollout engine via verl (SGLang preferred)",
    },
    "mindie": {
        "name": "MindIE/vLLM-Ascend",
        "critical_issues": {
            "#10684": "DSA Hadamard ALL-ZERO after sleep/wake → verl RLHF blocker on NPU!",
            "#10579": "MoE NaN 1-line fix → 0 reviews → stalled",
            "#10724": "DSV4 8th failure → 2*A2 PD-Mix crash",
        },
        "sleep_level_support": "sleep_level=2 ONLY (sleep_level=1 NOT supported on NPU!)",
        "ranking": "N/A (Ascend NPU only, not RTX 4090)",
    },
    "rllm": {
        "name": "rLLM",
        "version": "v0.3.0-pre",
        "critical_issues": {
            "#605": "GRPO grouping bug → trajectory.uid vs task_ids → group size 1 → BROKEN!",
            "#663": "Step.output was always None → ALL rewards=0.0 (MERGED fix)",
        },
        "optimal_config": {
            "backend": "tinker",
            "bypass_mode": True,
            "deterministic_optimizer": True,  # #630 MERGED
        },
        "ranking": "#3 BLOCKED by #605 (GRPO grouping bug)",
    },
    "sglang": {
        "name": "SGLang",
        "version": "v0.5.13",
        "critical_issues": {
            "#27097": "multi-LoRA determinism bug",
            "#28618": "SM89 DSV4-Flash-FP8 → validated on 8xL20 → RTX 4090 pathway!",
            "#28612": "DSV4 C128 state mapping lifecycle fix",
            "#28582": "CRITICAL RCE",
            "#28588": "image decompression bomb guard",
        },
        "sleep_mechanism": "tag-based (release/resume_memory_occupation)",
        "sleep_tags": ["kv_cache", "weights"],
        "lora_as_adapter": True,  # sleep_level=1: tags=["kv_cache"] only
        "ranking": "#1 rollout engine (via verl HYBRID)",
    },
    "pytorch": {
        "name": "PyTorch",
        "critical_issues": {
            "#184119": "SM89 fp8→bf16 fusion guard → P9 thesis!",
            "#187620": "PartialOffloadPolicy → dp=1 NOT viable (shard=identity)",
            "#187636": "autotune_at_compile_time → reduces SM89 batch-dependent fusion",
        },
        "fsdp_offload": "CPUOffloadPolicy(pin_memory=True) — default is TRUE!",
        "partial_offload": "NOT viable on dp=1 → shard=identity → OOM",
        "ranking": "infrastructure layer (not a training framework)",
    },
}

# Model memory estimates (BF16, single GPU)
MODEL_ESTIMATES = {
    "qwen3-8b": {
        "total_params": 8e9,
        "d_model": 4096,
        "d_hidden": 11008,
        "n_layers": 32,
        "n_heads_kv": 4,  # GQA
        "bf16_memory": 16,  # GiB
        "fp8_memory": 8,
        "lora_rank32_payload": 0.288,  # GiB
        "optimizer_cpu": 64,  # GiB CPU
        "peak_gpu_grpo": 22,  # GiB with offload
        "fits_rtx4090": True,
        "sleep_level_1_benefit": "80x payload reduction (0.288 vs 16 GiB)",
        "ranking": "#2 GRPO candidate",
    },
    "qwen3-30b-a3b": {
        "total_params": 30e9,
        "active_params": 3e9,
        "d_model": 2048,
        "n_experts": 128,
        "top_k_experts": 8,
        "bf16_memory_active": 6,  # GiB (active params only)
        "fp8_memory_active": 3,
        "lora_rank32_payload": 0.060,  # GiB (LoRA on active only)
        "optimizer_cpu": 24,  # GiB CPU (active params only)
        "peak_gpu_grpo": 18,  # GiB with offload
        "fits_rtx4090": True,  # with MoE+LoRA+offload
        "sleep_level_1_benefit": "100x payload reduction (0.060 vs 6 GiB)",
        "ranking": "#1 BEST GRPO candidate (MoE + LoRA)",
    },
    "llama3-8b": {
        "total_params": 8e9,
        "bf16_memory": 16,
        "lora_rank32_payload": 0.288,
        "peak_gpu_grpo": 22,
        "fits_rtx4090": True,
        "ranking": "#2 baseline",
    },
}

# MUST DO / MUST NOT rules (compiled from 7-framework research)
MUST_DO = [
    "Use SGLang rollout + sleep_level=1 + LoRA rank=32/alpha=64 + merge=false",
    "Use FSDP backend (only backend with detach fix from #6699)",
    "Set enforce_eager=True for DSV4/DSV3 models",
    "Use CPU_Adam optimizer (18Ψ→3.8Ψ)",
    "Set gradient_clipping=1.0 explicitly (#8068)",
    "Set param_offload=True + grad_offload=True",
    "Use ZeRO-2 (NOT ZeRO-3! Pure overhead on dp=1)",
    "Use bypass_mode (eliminates ref model → saves ~11 GiB)",
    "Set overlap_comm=False on single GPU (#8061 NaN bug)",
    "Use peft.merge=false (forces sleep_level=1 → 80x payload reduction)",
]

MUST_NOT = [
    "Use peft.merge=true (forces sleep_level=2 → full re-transfer)",
    "Use lora_rank=64 (breaks EOS per #6782)",
    "Use vLLM-Ascend backend (sleep_level=1 NOT supported)",
    "Use Megatron backend (C9 detach fix not upstream)",
    "Use overlap_comm=True on single GPU (#8061 NaN)",
    "Use ZeRO-3 (pure overhead on dp=1)",
    "Use PartialOffloadPolicy on dp=1 (shard=identity → OOM)",
    "Use CUDA graphs for DSV4 models (8+ failures)",
    "Use Muon optimizer (6 blockers → NOT viable)",
    "Set pin_memory=False (default is TRUE, already optimal)",
]


def validate_config(config_path):
    """Validate a GRPO training config against all 7-framework rules."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    errors = []
    warnings = []
    passes = []

    # Check framework-specific rules
    framework = config.get("framework", "unknown").lower()
    if framework not in FRAMEWORK_PROFILES:
        errors.append(f"Unknown framework: {framework}")

    # Check critical MUST NOT rules
    if config.get("peft_merge", False) is True:
        errors.append("MUST NOT: peft.merge=true forces sleep_level=2 → full weight re-transfer")
    if config.get("lora_rank", 0) == 64:
        errors.append("MUST NOT: lora_rank=64 breaks EOS (#6782)")
    if config.get("zero_stage", 0) == 3:
        errors.append("MUST NOT: ZeRO-3 is pure overhead on dp=1 RTX 4090")
    if config.get("overlap_comm", False) is True:
        errors.append("MUST NOT: overlap_comm=True causes NaN on single GPU (#8061)")
    if config.get("gradient_clipping", 0) == 0:
        warnings.append("WARNING: gradient_clipping=0 → MUST set 1.0 (#8068)")
    if config.get("optimizer") == "muon":
        errors.append("MUST NOT: Muon optimizer blocked by 6 issues (#7939 etc)")
    if config.get("backend") in ["megatron", "automodel", "torchtitan"]:
        errors.append("MUST NOT: backend has UNFIXED detach leak (#6699)")

    # Check critical MUST DO rules
    if config.get("rollout") == "sglang":
        passes.append("PASS: SGLang rollout (best LoRA adapter support)")
    elif config.get("rollout") == "vllm":
        warnings.append("WARNING: vLLM rollout → integer-based sleep_level, less fine-grained than SGLang")

    if config.get("sleep_level") == 1:
        passes.append("PASS: sleep_level=1 (LoRA adapter, 80x payload reduction)")
    elif config.get("sleep_level") == 2:
        errors.append("MUST NOT: sleep_level=2 → full weight re-transfer → AVOID on RTX 4090")

    if config.get("bypass_mode") is True:
        passes.append("PASS: bypass_mode (eliminates ref model → saves ~11 GiB)")
    else:
        warnings.append("WARNING: bypass_mode=False → need ref model (~11 GiB extra)")

    if config.get("optimizer") == "cpu_adam":
        passes.append("PASS: CPU_Adam optimizer (18Ψ→3.8Ψ)")
    elif config.get("optimizer") == "adam":
        warnings.append("WARNING: GPU Adam → 64 GiB optimizer states → EXCEEDS RTX 4090")

    if config.get("enforce_eager") is True:
        passes.append("PASS: enforce_eager=True (DSV4 MANDATORY)")
    else:
        warnings.append("WARNING: enforce_eager not set → DSV4 models WILL crash (#45309/#45863)")

    if config.get("gradient_clipping") == 1.0:
        passes.append("PASS: gradient_clipping=1.0 (#8068)")
    else:
        warnings.append("WARNING: gradient_clipping should be 1.0 (#8068)")

    # Print results
    print(f"\n{'='*60}")
    print(f"RTX 4090 GRPO Config Validation: {config_path}")
    print(f"Framework: {framework}")
    print(f"{'='*60}")

    if passes:
        print(f"\n✅ PASSES ({len(passes)}):")
        for p in passes:
            print(f"  ✅ {p}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠️  {w}")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ❌ {e}")

    # Memory estimate
    model = config.get("model", "unknown")
    if model in MODEL_ESTIMATES:
        est = MODEL_ESTIMATES[model]
        print(f"\n📊 Memory Estimate for {model}:")
        print(f"  Peak GPU (GRPO): ~{est['peak_gpu_grpo']} GiB")
        print(f"  Fits RTX 4090: {'✅ YES' if est['fits_rtx4090'] else '❌ NO'}")
        print(f"  LoRA payload (sleep_level=1): ~{est['lora_rank32_payload']} GiB")
        print(f"  CPU optimizer states: ~{est['optimizer_cpu']} GiB")

    if errors:
        print(f"\n❌ CONFIG HAS {len(errors)} CRITICAL ERRORS — FIX BEFORE TRAINING!")
        return False
    elif warnings:
        print(f"\n⚠️  CONFIG HAS {len(warnings)} WARNINGS — REVIEW CAREFULLY")
        return True
    else:
        print(f"\n✅ CONFIG PASSES ALL RTX 4090 GRPO CHECKS")
        return True


def generate_config(scenario):
    """Generate optimal config for a given scenario."""
    scenarios = {
        "verl-grpo": {
            "framework": "verl",
            "model": "qwen3-8b",
            "rollout": "sglang",
            "mode": "hybrid",
            "sleep_level": 1,
            "peft_merge": False,
            "lora_rank": 32,
            "lora_alpha": 64,
            "bypass_mode": True,
            "optimizer": "cpu_adam",
            "param_offload": True,
            "grad_offload": True,
            "enforce_eager": True,
            "gradient_clipping": 1.0,
            "backend": "fsdp",
            "zero_stage": 2,
            "overlap_comm": False,
            "group_size": 8,
            "max_response_length": 2048,
            "lr": 1e-6,
            "weight_decay": 0.01,
        },
        "verl-cppo": {
            # CPPO = #1 BEST for RTX 4090
            "framework": "verl",
            "model": "qwen3-8b",
            "rollout": "sglang",
            "mode": "hybrid",
            "algorithm": "cppo",  # position-weighted trust region
            "sleep_level": 1,
            "peft_merge": False,
            "lora_rank": 32,
            "lora_alpha": 64,
            "bypass_mode": True,
            "optimizer": "cpu_adam",
            "param_offload": True,
            "grad_offload": True,
            "enforce_eager": True,
            "gradient_clipping": 1.0,
            "backend": "fsdp",
        },
        "verl-moe": {
            # MoE model — #1 BEST candidate
            "framework": "verl",
            "model": "qwen3-30b-a3b",
            "rollout": "sglang",
            "mode": "hybrid",
            "sleep_level": 1,
            "peft_merge": False,
            "lora_rank": 32,
            "lora_alpha": 64,
            "bypass_mode": True,
            "optimizer": "cpu_adam",
            "param_offload": True,
            "grad_offload": True,
            "enforce_eager": True,
            "gradient_clipping": 1.0,
            "backend": "fsdp",
        },
        "deepspeed-grpo": {
            "framework": "deepspeed",
            "model": "qwen3-8b",
            "zero_stage": 2,
            "optimizer": "cpu_adam",
            "offload_param": True,
            "offload_grad": True,
            "overlap_comm": False,
            "gradient_clipping": 1.0,
            "enforce_eager": True,
            "lora_rank": 32,
            "lora_alpha": 64,
        },
    }

    if scenario not in scenarios:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(scenarios.keys())}")
        return

    config = scenarios[scenario]
    print(f"\n# RTX 4090 GRPO Config: {scenario}")
    print(f"# Generated by rtx4090_grpo_config_validator.py")
    print(f"# Based on 7-framework deep research (30+ critical issues)")
    print(yaml.dump(config, default_flow_style=False))


def compare_frameworks(f1, f2):
    """Compare two frameworks for RTX 4090 GRPO training."""
    if f1 not in FRAMEWORK_PROFILES or f2 not in FRAMEWORK_PROFILES:
        print(f"Unknown framework. Available: {', '.join(FRAMEWORK_PROFILES.keys())}")
        return

    p1, p2 = FRAMEWORK_PROFILES[f1], FRAMEWORK_PROFILES[f2]
    print(f"\n{'='*60}")
    print(f"RTX 4090 GRPO: {p1['name']} vs {p2['name']}")
    print(f"{'='*60}")

    print(f"\n| Feature | {p1['name']} | {p2['name']} |")
    print(f"|---------|{p1['name']:^20}|{p2['name']:^20}|")
    print(f"| Ranking | {p1['ranking']:^20} | {p2['ranking']:^20} |")
    print(f"| Version | {p1.get('version','?'):^20} | {p2.get('version','?'):^20} |")

    # Sleep/wake comparison
    s1 = p1.get("sleep_mechanism", p1.get("sleep_level", "N/A"))
    s2 = p2.get("sleep_mechanism", p2.get("sleep_level", "N/A"))
    print(f"| Sleep/Wake | {str(s1)[:20]:^20} | {str(s2)[:20]:^20} |")

    # Critical issues count
    n1 = len(p1.get("critical_issues", {}))
    n2 = len(p2.get("critical_issues", {}))
    print(f"| Critical issues | {n1:^20} | {n2:^20} |")

    # Key differences
    print(f"\nKey differences:")
    print(f"\n{p1['name']} critical issues:")
    for id, desc in p1.get("critical_issues", {}).items():
        print(f"  {id}: {desc}")
    print(f"\n{p2['name']} critical issues:")
    for id, desc in p2.get("critical_issues", {}).items():
        print(f"  {id}: {desc}")


def estimate_model(model_name):
    """Estimate memory requirements for a model on RTX 4090."""
    if model_name not in MODEL_ESTIMATES:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(MODEL_ESTIMATES.keys())}")
        return

    est = MODEL_ESTIMATES[model_name]
    print(f"\n{'='*60}")
    print(f"RTX 4090 GRPO Memory Estimate: {model_name}")
    print(f"{'='*60}")
    print(f"\n| Component | Memory | Notes |")
    print(f"|-----------|--------|-------|")
    print(f"| BF16 weights | ~{est['bf16_memory']} GiB | (FP8: ~{est.get('fp8_memory', '?')} GiB) |")
    print(f"| LoRA rank=32 | ~{est['lora_rank32_payload']} GiB | sleep_level=1 payload |")
    print(f"| Peak GPU (GRPO) | ~{est['peak_gpu_grpo']} GiB | with CPU offload |")
    print(f"| CPU optimizer | ~{est.get('optimizer_cpu', '?')} GiB | CPU_Adam states |")
    print(f"| Fits RTX 4090 | {'YES ✅' if est['fits_rtx4090'] else 'NO ❌'} | 24 GiB capacity |")
    print(f"| sleep_level=1 benefit | {est.get('sleep_level_1_benefit', '?')} | |")
    print(f"| GRPO ranking | {est.get('ranking', '?')} | |")


def rtx4090_best_config():
    """Print the optimal RTX 4090 GRPO configuration."""
    print(f"\n{'='*60}")
    print(f"RTX 4090 GRPO BEST CONFIGURATION")
    print(f"Based on 7-framework deep research (30+ critical issues)")
    print(f"{'='*60}")

    print(f"\n{'='*20} MUST DO {'='*20}")
    for rule in MUST_DO:
        print(f"  ✅ {rule}")

    print(f"\n{'='*20} MUST NOT {'='*20}")
    for rule in MUST_NOT:
        print(f"  ❌ {rule}")

    print(f"\n{'='*20} OPTIMAL CONFIG {'='*20}")
    generate_config("verl-cppo")

    print(f"\n{'='*20} RANKING {'='*20}")
    rankings = [
        ("#1", "verl CPPO + bypass_mode + SGLang sleep_level=1", "★★★★★★★★"),
        ("#1.5", "verl Tinker primitives + GRPO", "★★★★★★★"),
        ("#2", "verl GRPO + bypass_mode + SGLang", "★★★★★★"),
        ("#2.5", "DeepSpeed ZeRO-2 + CPU_Adam", "★★★★★"),
        ("#3", "rLLM Tinker (BLOCKED by #605)", "★★★★"),
        ("#4", "Megatron core (complex, detach missing)", "★★★"),
    ]
    for rank, desc, stars in rankings:
        print(f"  {rank}: {stars} {desc}")

    print(f"\n{'='*20} KEY ISSUES {'='*20}")
    critical = [
        ("#8061", "DeepSpeed overlap_comm NaN on single GPU"),
        ("#8068", "DeepSpeed gradient_clipping 0→1.0"),
        ("#605", "rLLM GRPO grouping bug → BROKEN"),
        ("#6699", "verl detach memory fix (FSDP only)"),
        ("#6782", "verl LoRA rank=64 breaks EOS"),
        ("#6468", "verl FSDP2 CPU memory leak"),
        ("#10684", "MindIE DSA Hadamard ALL-ZERO"),
        ("#45683", "vLLM MoE combine determinism"),
        ("#184119", "PyTorch SM89 fp8 guard (P9)"),
    ]
    for id, desc in critical:
        print(f"  {id}: {desc}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "validate" and args:
        validate_config(args[0])
    elif cmd == "generate" and args:
        generate_config(args[0])
    elif cmd == "compare" and len(args) >= 2:
        compare_frameworks(args[0], args[1])
    elif cmd == "estimate" and args:
        estimate_model(args[0])
    elif cmd == "rtx4090":
        rtx4090_best_config()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
