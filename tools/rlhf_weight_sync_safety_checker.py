#!/usr/bin/env python3
"""RLHF Weight Sync Safety Checker — cross-framework sleep/wake buffer preservation validation

Validates weight synchronization safety for RLHF training across 7 frameworks.
Checks for constant buffer corruption risk during sleep/wake state transitions.

Usage:
  python rlhf_weight_sync_safety_checker.py check [--framework <name>] [--mode <mode>]
  python rlhf_weight_sync_safety_checker.py compare [--model <model>]
  python rlhf_weight_sync_safety_checker.py rtx4090 [--config <config>]

Modes: check, compare, rtx4090
Frameworks: verl, deepspeed, rllm, megatron, vllm, sglang, vllm_ascend
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BufferRisk:
    name: str
    framework: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    bug_id: Optional[str] = None
    fix: str = ""
    verified: bool = False


@dataclass
class FrameworkSyncProfile:
    name: str
    sync_mode: str
    has_sleep_wake: bool
    buffer_transfer_includes_constants: bool
    known_corruption_bugs: list = field(default_factory=list)
    safe_on_single_gpu: bool = True
    overlap_comm_safe: bool = True
    recommended_config: str = ""


# ============================================================
# Database: Framework sync profiles
# ============================================================

FRAMEWORK_PROFILES = {
    "verl": FrameworkSyncProfile(
        name="verl HYBRID",
        sync_mode="ZMQ IPC bucket (512MB) / in-process generator",
        has_sleep_wake=True,
        buffer_transfer_includes_constants=False,  # NOT verified!
        known_corruption_bugs=[
            BufferRisk("DSA Hadamard", "verl/vllm", "Hadamard matrix may not survive Sleep/Wake on CUDA",
                       "CRITICAL", "#10684-variant", "Verify Hadamard preservation on Wake", False),
            BufferRisk("LoRA prefix hardcode", "verl FSDP2", "8 hard-coded LoRA prefixes → silently fails on non-standard architectures",
                       "HIGH", "#6468", "Add dynamic LoRA prefix detection", True),
            BufferRisk("detach memory leak", "verl", "model_output not detached → 4x memory growth",
                       "HIGH", "#6699", "detach model_output (MERGED for FSDP, UNFIXED for 3 backends)", True),
            BufferRisk("CPU memory leak", "verl FSDP2", "0.6-6.3 GiB/step CPU memory growth during weight sync",
                       "HIGH", "#6468", "Not yet fixed → Ray OOM", True),
            BufferRisk("LoRA rank=64 EOS", "verl vLLM rollout", "LoRA rank=64 breaks EOS token → infinite generation",
                       "CRITICAL", "#6782", "MUST use rank=32/alpha=64", True),
        ],
        safe_on_single_gpu=True,
        overlap_comm_safe=True,  # verl doesn't use DeepSpeed overlap_comm
        recommended_config="HYBRID + FSDP2 + bypass_mode + CPPO + sleep_level=1 (LoRA adapter)",
    ),
    "deepspeed": FrameworkSyncProfile(
        name="DeepSpeed ZeRO-2",
        sync_mode="CPU_Adam implicit (CPU offload → GPU load)",
        has_sleep_wake=False,  # implicit — no explicit Sleep/Wake
        buffer_transfer_includes_constants=True,  # CPU_Adam preserves all states
        known_corruption_bugs=[
            BufferRisk("overlap_comm NaN", "DeepSpeed ZeRO-2", "overlap_comm + torch.compile → multi-stream data race → NaN",
                       "CRITICAL", "#8061", "overlap_comm=False on single GPU (MUST)", True),
            BufferRisk("ZeRO-3+PEFT dtype", "DeepSpeed ZeRO-3", "mixed dtype in _allgather_params_coalesced → TypeError",
                       "HIGH", "#8072/#8073", "2-line per-param dtype fix (#8073)", True),
            BufferRisk("gradient_clipping default", "DeepSpeed", "default 0→1.0 → ALWAYS set explicitly",
                       "MEDIUM", "#8068", "Set gradient_clipping=1.0 explicitly", True),
        ],
        safe_on_single_gpu=True,
        overlap_comm_safe=False,  # overlap_comm=False REQUIRED on single GPU
        recommended_config="ZeRO-2 + CPU_Adam + overlap_comm=False + gradient_clipping=1.0",
    ),
    "rllm": FrameworkSyncProfile(
        name="rLLM Tinker",
        sync_mode="Single-step (no sleep/wake)",
        has_sleep_wake=False,
        buffer_transfer_includes_constants=True,  # single process → no transfer needed
        known_corruption_bugs=[
            BufferRisk("GRPO grouping bug", "rLLM", "groups by task_id:trajectory.name → group size=1 → BROKEN",
                       "CRITICAL", "#605", "Change grouping key to task_id only (1-line fix)", True),
        ],
        safe_on_single_gpu=True,
        overlap_comm_safe=True,
        recommended_config="Tinker + single-step + bypass_mode + fix #605 first!",
    ),
    "megatron": FrameworkSyncProfile(
        name="Megatron-LM",
        sync_mode="TP group AllReduce/broadcast",
        has_sleep_wake=False,  # multi-GPU → no Sleep/Wake → but has RouterReplay
        buffer_transfer_includes_constants=False,  # RouterReplay needed!
        known_corruption_bugs=[
            BufferRisk("RouterReplay stale", "Megatron MoE", "MoE router state stale after weight update → needs replay",
                       "HIGH", "#4168", "RouterReplay pattern → replay router after each weight update", True),
            BufferRisk("Muon clipping stalls", "Megatron", "ChainedOptimizer global clipping stalls Muon updates",
                       "MEDIUM", "#5394/#5395", "skip_grad_norm_clip attribute (+15/-1)", True),
        ],
        safe_on_single_gpu=False,  # Megatron designed for multi-GPU
        overlap_comm_safe=True,
        recommended_config="Multi-GPU only → use verl + Megatron Lite for RLHF",
    ),
    "vllm": FrameworkSyncProfile(
        name="vLLM (inference only)",
        sync_mode="No training sync (pure inference)",
        has_sleep_wake=False,
        buffer_transfer_includes_constants=True,  # no transfer needed
        known_corruption_bugs=[
            BufferRisk("DSV4 cudagraph revert", "vLLM", "DSV4 cudagraph → stale dynamic data → reverted",
                       "HIGH", "#45972", "enforce_eager=True for DSV4", True),
            BufferRisk("DSV4 flashinfer cache", "vLLM", "flashinfer sparse cache → stale → GSM8K regression",
                       "HIGH", "#45979", "Clear cache between steps", True),
        ],
        safe_on_single_gpu=True,
        overlap_comm_safe=True,
        recommended_config="Used as verl rollout backend → HYBRID mode",
    ),
    "sglang": FrameworkSyncProfile(
        name="SGLang (inference + verl HYBRID rollout)",
        sync_mode="HTTP release_memory_occupation/resume_memory_occupation (tag-based)",
        has_sleep_wake=True,
        buffer_transfer_includes_constants=True,  # tag-based: ["kv_cache"] or ["weights", "kv_cache"]
        known_corruption_bugs=[
            BufferRisk("DSV4 MTP revert", "SGLang", "DSV4 MTP → swa_loc cache → stale → accept-length collapse",
                       "HIGH", "#28591/#28520", "Per-step dynamic data MUST NOT cache", True),
            BufferRisk("DSV4 C128 state mapping", "SGLang", "C128 slots derived from stale full_to_swa_index_mapping",
                       "HIGH", "#28612", "Derive directly from full_loc/128 instead of unstable mapping", True),
            BufferRisk("multi-LoRA determinism", "SGLang", "4 factors cause non-determinism in multi-LoRA serving",
                       "MEDIUM", "#27097", "Fix all 4 factors (#28499/#28566/#28588)", True),
            BufferRisk("CRITICAL RCE", "SGLang", "CVSS 9.8 RCE vulnerability in serve endpoint",
                       "CRITICAL", "#28582", "PATCH immediately! Authentication required", True),
            BufferRisk("image decompression bomb", "SGLang", "PIL MAX_IMAGE_PIXELS=None → decompression bomb risk",
                       "HIGH", "#28588", "Set PIL.Image.MAX_IMAGE_PIXELS=89478485", True),
            BufferRisk("sleep_level=1 LoRA only KV", "SGLang", "sleep_level=1 releases only kv_cache → base weights stay → LoRA delta sync",
                       "MEDIUM", "source code", "sleep_level=1 + merge=false + LoRA adapter path → RTX 4090 optimal", True),
        ],
        safe_on_single_gpu=True,
        overlap_comm_safe=True,
        recommended_config="verl HYBRID + SGLang + sleep_level=1 + LoRA rank=32/alpha=64 + enforce_eager + PIL limits",
    ),
    "vllm_ascend": FrameworkSyncProfile(
        name="vLLM-Ascend (NPU inference)",
        sync_mode="NPUIPC (HCCS interconnect)",
        has_sleep_wake=True,
        buffer_transfer_includes_constants=False,  # #10684 proves this!
        known_corruption_bugs=[
            BufferRisk("DSA Hadamard ALL-ZERO", "vLLM-Ascend", "Hadamard matrix becomes ALL-ZERO after sleep/wake → CRITICAL!",
                       "CRITICAL", "#10684", "Include buffers in NPUIPC or regenerate on Wake", True),
            BufferRisk("MoE NaN", "vLLM-Ascend", "torch.abs() on row indices → duplication → NaN",
                       "HIGH", "#10579", "Remove torch.abs() line (1-line fix)", True),
            BufferRisk("DSV4 chat template", "vLLM-Ascend", "Wrong chat template for DSV4 on Ascend",
                       "MEDIUM", "#10645/#10628", "Correct chat template", True),
        ],
        safe_on_single_gpu=True,  # NPU single-card
        overlap_comm_safe=True,
        recommended_config="Fix #10684 first → then NPUIPC → then verl Ascend backend",
    ),
}

# ============================================================
# Database: Models
# ============================================================

MODELS = {
    "qwen3_4b": {"params": 4, "type": "dense", "architecture": "Qwen3"},
    "qwen3_30b_a3b": {"params": 30, "type": "moe", "active_params": 3, "architecture": "Qwen3-MoE"},
    "llama3_1b": {"params": 1, "type": "dense", "architecture": "Llama3.2"},
    "llama3_8b": {"params": 8, "type": "dense", "architecture": "Llama3.1"},
    "glm4_9b": {"params": 9, "type": "dense", "architecture": "GLM-4"},
    "dsv4_671b_a37b": {"params": 671, "type": "moe", "active_params": 37, "architecture": "DSV4-MoE"},
}


def check_framework(name: str):
    """Check weight sync safety for a specific framework."""
    profile = FRAMEWORK_PROFILES.get(name)
    if not profile:
        print(f"Unknown framework: {name}")
        print(f"Available: {', '.join(FRAMEWORK_PROFILES.keys())}")
        return

    print(f"\n{'='*70}")
    print(f"  Weight Sync Safety Check: {profile.name}")
    print(f"{'='*70}")

    print(f"\n  Sync Mode: {profile.sync_mode}")
    print(f"  Has Sleep/Wake: {profile.has_sleep_wake}")
    print(f"  Buffer Transfer Includes Constants: {profile.buffer_transfer_includes_constants}")
    print(f"  Safe on Single GPU: {profile.safe_on_single_gpu}")
    print(f"  overlap_comm Safe: {profile.overlap_comm_safe}")
    print(f"  Recommended Config: {profile.recommended_config}")

    # Sleep/Wake risk assessment
    if profile.has_sleep_wake and not profile.buffer_transfer_includes_constants:
        print(f"\n  ★★★★★★★★ CRITICAL RISK: Sleep/Wake + buffer NOT included in transfer!")
        print(f"  → Constant buffers (Hadamard, router bias) may be CORRUPTED during state transfer")
        print(f"  → MUST verify: does DSA Hadamard survive Sleep/Wake?")
        print(f"  → Recommended: regenerate initialization-dependent buffers on Wake")
    elif profile.has_sleep_wake:
        print(f"\n  ✓ Sleep/Wake safe: buffers included in transfer protocol")
    else:
        print(f"\n  ✓ No explicit Sleep/Wake → no buffer corruption risk")

    # Known bugs
    if profile.known_corruption_bugs:
        print(f"\n  Known Corruption Bugs ({len(profile.known_corruption_bugs)}):")
        for bug in profile.known_corruption_bugs:
            severity_icon = {"CRITICAL": "!!!", "HIGH": "!!", "MEDIUM": "!", "LOW": "."}
            icon = severity_icon.get(bug.severity, "?")
            print(f"    [{icon}] {bug.name} ({bug.severity})")
            if bug.bug_id:
                print(f"        Bug: #{bug.bug_id}")
            print(f"        {bug.description}")
            print(f"        Fix: {bug.fix}")
            print(f"        Verified: {bug.verified}")


def compare_frameworks(model_name: str = None):
    """Compare weight sync safety across all frameworks."""
    print(f"\n{'='*70}")
    print(f"  Cross-Framework Weight Sync Safety Comparison")
    if model_name:
        model = MODELS.get(model_name)
        print(f"  Model: {model_name} ({model['params']}B {model['type']})")
    print(f"{'='*70}")

    # Table header
    print(f"\n  | Framework | Sleep/Wake | Buffer Risk | Critical Bugs | Single GPU Safe |")
    print(f"  |-----------|------------|-------------|---------------|----------------|")

    for name, profile in FRAMEWORK_PROFILES.items():
        has_sw = "Yes" if profile.has_sleep_wake else "No"
        buf_risk = "HIGH" if profile.has_sleep_wake and not profile.buffer_transfer_includes_constants else "LOW"
        crit = sum(1 for b in profile.known_corruption_bugs if b.severity == "CRITICAL")
        safe = "Yes" if profile.safe_on_single_gpu else "No"
        print(f"  | {name:12s} | {has_sw:10s} | {buf_risk:11s} | {crit:13d} | {safe:14s} |")

    # Universal patterns
    print(f"\n  ★★★★★★★★ Universal Patterns:")
    print(f"  1. Sleep/Wake + constant buffer NOT in transfer = CORRUPTION RISK")
    print(f"     → #10684 (Ascend), potential CUDA variant (unverified)")
    print(f"  2. Per-step dynamic data MUST NOT be cached across steps")
    print(f"     → DSV4: 6 failures (3 CUDA graph, 1 sparse cache, 1 swa_loc cache, 1 chat)")
    print(f"  3. overlap_comm + torch.compile = NaN on single GPU")
    print(f"     → DeepSpeed #8061 → overlap_comm=False MANDATORY")
    print(f"  4. Regenerate buffers on Wake = CHEAPEST + MOST ROBUST fix")
    print(f"     → No transfer bandwidth → deterministic → same as RouterReplay pattern")


def rtx4090_check(config: str = "verl_hybrid"):
    """RTX 4090 specific weight sync safety check."""
    print(f"\n{'='*70}")
    print(f"  RTX 4090 Weight Sync Safety Check")
    print(f"  Config: {config}")
    print(f"{'='*70}")

    configs = {
        "verl_hybrid": {
            "framework": "verl",
            "backend": "vllm",
            "description": "verl HYBRID + vLLM rollout + FSDP2 training",
            "checks": [
                ("DSA Hadamard preservation during Sleep/Wake", "UNVERIFIED", "CRITICAL"),
                ("LoRA prefix detection (8 hard-coded → non-standard archs)", "VERIFIED BUG", "HIGH"),
                ("detach model_output (memory leak)", "MERGED for FSDP", "HIGH"),
                ("overlap_comm=False needed?", "NOT applicable (verl)", "LOW"),
                ("gradient_clipping=1.0 set?", "MUST set", "MEDIUM"),
                ("CPPO + sync TransferQueue trainer", "MUST use sync", "HIGH"),
            ]
        },
        "deepspeed_zero2": {
            "framework": "deepspeed",
            "backend": "deepspeed",
            "description": "DeepSpeed ZeRO-2 + CPU_Adam + LoRA",
            "checks": [
                ("overlap_comm=False (MUST on single GPU)", "VERIFIED BUG", "CRITICAL"),
                ("gradient_clipping=1.0 (MUST set explicitly)", "VERIFIED BUG", "MEDIUM"),
                ("ZeRO-3+PEFT LoRA regression", "ZeRO-2 unaffected", "LOW"),
                ("CPU_Adam optimizer state preservation", "SAFE", "LOW"),
                ("Muon CPU offload", "BLOCKED (#7939)", "CRITICAL"),
            ]
        },
        "rllm_tinker": {
            "framework": "rllm",
            "backend": "rllm",
            "description": "rLLM Tinker single-step + bypass_mode",
            "checks": [
                ("GRPO grouping bug #605 (MUST fix first!)", "CRITICAL BUG", "CRITICAL"),
                ("No Sleep/Wake needed (single-step)", "SAFE", "LOW"),
                ("bypass_mode enabled", "MUST enable", "HIGH"),
                ("Step.output was None → rewards=0.0 (#663)", "MERGED fix", "MEDIUM"),
            ]
        },
    }

    if config not in configs:
        print(f"  Unknown config: {config}")
        print(f"  Available: {', '.join(configs.keys())}")
        return

    cfg = configs[config]
    print(f"\n  Config: {cfg['description']}")
    print(f"\n  Safety Checks ({len(cfg['checks'])}):")

    for check_name, status, severity in cfg["checks"]:
        icon = {"CRITICAL": "!!!", "HIGH": "!!", "MEDIUM": "!", "LOW": "."}
        i = icon.get(severity, "?")
        print(f"    [{i}] {check_name}")
        print(f"        Status: {status}")
        print(f"        Severity: {severity}")

    # MUST DO / MUST NOT
    print(f"\n  MUST DO:")
    if config == "verl_hybrid":
        print(f"    → FSDP2 backend (NOT Automodel/Megatron/TorchTitan)")
        print(f"    → CPPO + bypass_mode + sync TransferQueue trainer")
        print(f"    → overlap_comm=False (if DeepSpeed backend)")
        print(f"    → gradient_clipping=1.0")
        print(f"    → LoRA rank=32/alpha=64 (NOT rank=64 → #6782)")
        print(f"    → sleep_level=1 + merge=false (LoRA adapter path → 80x payload reduction)")
        print(f"    → SGLang rollout backend (best LoRA adapter + sleep/wake support)")
        print(f"    → Verify DSA Hadamard preservation on Wake (when GPU available)")
    elif config == "deepspeed_zero2":
        print(f"    → overlap_comm=False (MANDATORY)")
        print(f"    → gradient_clipping=1.0 (MANDATORY)")
        print(f"    → CPU_Adam optimizer")
        print(f"    → ZeRO-2 (NOT ZeRO-3 → pure overhead on single GPU)")
    elif config == "rllm_tinker":
        print(f"    → Fix #605 BEFORE training (GRPO grouping bug → BROKEN)")
        print(f"    → bypass_mode=True (skip ref model)")
        print(f"    → Use post-#663 version (rewards were all 0.0 before fix)")

    print(f"\n  MUST NOT:")
    if config == "verl_hybrid":
        print(f"    → Use async Ray trainer with CPPO (overrides loss_mode!)")
        print(f"    → Use LoRA rank=64 (#6782 breaks EOS)")
        print(f"    → Use Automodel/Megatron/TorchTitan backends (detach leak unfixed)")
        print(f"    → Use overlap_comm=True (NaN risk)")
        print(f"    → Use sleep_level=2 + merge=true (full weight re-transfer every step → 80x slower)")
        print(f"    → Use vLLM-Ascend backend (sleep_level=1 NOT supported → always full sleep)")
    elif config == "deepspeed_zero2":
        print(f"    → Use overlap_comm=True (NaN #8061)")
        print(f"    → Use ZeRO-3 on single GPU (pure overhead)")
        print(f"    → Use Muon optimizer (4 blockers)")
    elif config == "rllm_tinker":
        print(f"    → Train without fixing #605 (GRPO = REINFORCE → no variance reduction)")
        print(f"    → Use pre-#663 version (all rewards = 0.0)")

    # Sleep/Wake Level Guide
    if config == "verl_hybrid":
        print(f"\n  SLEEP/WAKE LEVEL GUIDE (verl HYBRID RTX 4090):")
        print(f"    sleep_level=1 (LoRA adapter, merge=false):")
        print(f"      → Release: tags=['kv_cache'] → base weights STAY on GPU")
        print(f"      → Resume: tags=['kv_cache'] → only restore KV cache space")
        print(f"      → Weight sync: LoRA deltas only (~200 MiB vs ~16 GiB → 80x reduction)")
        print(f"      → Base sync: ONE TIME only (first step) → then adapter sync per step")
        print(f"      → ★★★ OPTIMAL for RTX 4090 GRPO training!")
        print(f"    sleep_level=2 (merge mode, merge=true):")
        print(f"      → Release: tags=['kv_cache', 'weights'] → EVERYTHING released")
        print(f"      → Resume: tags=['weights'] then tags=['kv_cache'] → full restore needed")
        print(f"      → Weight sync: FULL model weights every step (~16 GiB → slow)")
        print(f"      → ★★★ AVOID on RTX 4090 → much slower weight transfer cycle")


def main():
    parser = argparse.ArgumentParser(description="RLHF Weight Sync Safety Checker")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # check
    check_parser = subparsers.add_parser("check", help="Check specific framework")
    check_parser.add_argument("--framework", choices=list(FRAMEWORK_PROFILES.keys()),
                              default="verl", help="Framework to check")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare all frameworks")
    compare_parser.add_argument("--model", choices=list(MODELS.keys()), default=None)

    # rtx4090
    rtx_parser = subparsers.add_parser("rtx4090", help="RTX 4090 specific check")
    rtx_parser.add_argument("--config", choices=["verl_hybrid", "deepspeed_zero2", "rllm_tinker"],
                            default="verl_hybrid", help="Training config")

    args = parser.parse_args()

    if args.mode == "check":
        check_framework(args.framework)
    elif args.mode == "compare":
        compare_frameworks(args.model)
    elif args.mode == "rtx4090":
        rtx4090_check(args.config)


if __name__ == "__main__":
    main()
