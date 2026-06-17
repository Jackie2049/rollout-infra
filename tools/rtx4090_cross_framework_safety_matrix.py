#!/usr/bin/env python3
"""
RTX 4090 Cross-Framework Configuration Safety Matrix
=====================================================
Consolidates known pitfalls across all 7 frameworks for RTX 4090 (24GB VRAM, SM89).

Categories:
  - CRITICAL: Must fix before training, causes NaN/crash/OOM
  - STABILITY: Recommended for training stability
  - CONFIG: Performance/efficiency optimization
  - MONITOR: Track for future impact
  - EXPERIMENTAL: Use with caution, compare baseline

Usage:
  python tools/rtx4090_cross_framework_safety_matrix.py --mode matrix
  python tools/rtx4090_cross_framework_safety_matrix.py --mode framework deepspeed
  python tools/rtx4090_cross_framework_safety_matrix.py --mode category critical
  python tools/rtx4090_cross_framework_safety_matrix.py --mode summary
"""

import argparse
import sys

# ============================================================
# Safety Matrix Data
# ============================================================

MATRIX = {
    "deepspeed": {
        "critical": [
            {
                "id": "#8061",
                "title": "overlap_comm + torch.compile = NaN from step 1",
                "fix": "overlap_comm=False on single GPU (zero penalty + safer)",
                "source": "stage_1_and_2.py:1230 average_tensor() only waits current_stream",
                "tool": "deepspeed_zero_safety_checker.py --mode check",
            },
            {
                "id": "#44879/#45038",
                "title": "FP8 compressed-tensors KV cache crash on SM89",
                "fix": "Use Triton FP8 KV (#43914) or INT8 FlashInfer KV (NOT compressed-tensors)",
                "source": "vLLM FP8 KV 3-path distinction",
                "tool": "sm89_kv_cache_cost_analyzer.py",
            },
        ],
        "stability": [
            {
                "id": "#8068",
                "title": "gradient_clipping default 0.0 (disabled)",
                "fix": "Always set gradient_clipping=1.0 explicitly",
                "source": "DeepSpeed GRADIENT_CLIPPING_DEFAULT = 0",
                "tool": "deepspeed_zero_safety_checker.py --mode check",
            },
            {
                "id": "#8072/#8073",
                "title": "ZeRO-3+PEFT LoRA regression in v0.19.2",
                "fix": "Use ZeRO-2 (unaffected) or autocast_adapter_dtype=False workaround",
                "source": "#8066 dtype cast → mixed bf16+fp32 → TypeError in _allgather_params_coalesced",
                "tool": "deepspeed_zero_safety_checker.py --mode explain --check zero3_peft_regression",
            },
            {
                "id": "#7776",
                "title": "Muon gradient clipping orthogonalization bug",
                "fix": "gradient_clipping=0 for Muon groups, or clip AFTER Muon step",
                "source": "Clip BEFORE Newton-Schulz → near-zero vectors → degenerate",
                "tool": "deepspeed_zero_safety_checker.py --mode explain --check muon_plus_zero2_blocked",
            },
            {
                "id": "ZeRO-3",
                "title": "ZeRO-3 meaningless overhead on single GPU",
                "fix": "Use ZeRO-2 + CPU_Adam + offload_optimizer",
                "source": "partition_size=full with dp=1, all_gather skips",
                "tool": "deepspeed_zero_safety_checker.py --mode check",
            },
        ],
        "config": [
            {
                "id": "ZeRO-2+LoRA",
                "title": "LoRAOptimizedLinear + ZeRO-2 optimal config",
                "fix": "offload_ratio=0.5, LoRA rank=32, CPU_Adam optimizer",
                "source": "split forward + cumulative offload + A=kaiming B=zeros",
                "tool": "deepspeed_zero_safety_checker.py --mode generate --scenario lora-grpo",
            },
            {
                "id": "AutoEP+EP=1",
                "title": "MoE training viable on RTX 4090",
                "fix": "AutoEP + singleton EP=1 + LoRA + CPU_Adam",
                "source": "AutoEP merged #7938, EP=1 skip identity collectives #7997",
                "tool": "deepspeed_zero_safety_checker.py --mode generate --scenario moe-autoep",
            },
            {
                "id": "Muon+LoRA",
                "title": "Muon optimizer for LoRA (experimental)",
                "fix": "Gram NS method, aux_adam_lr=1e-5, gradient_clipping=1.0",
                "source": "#7953 merged, Muon operates on 2D matrices",
                "tool": "deepspeed_zero_safety_checker.py --mode generate --scenario lora-grpo-muon",
            },
        ],
        "monitor": [
            {
                "id": "#8060",
                "title": "AutoEP+ZeRO-3 (open) — limited single GPU benefit",
                "fix": "Track for future multi-GPU use",
            },
            {
                "id": "#8064",
                "title": "AutoEP+AutoTP folding — TP+EP共存",
                "fix": "Track for future multi-GPU use",
            },
            {
                "id": "#8058",
                "title": "ZenFlow native CPU optimizer — GPU spike 2944→256 MiB",
                "fix": "Track for CPU_Adam efficiency improvement on RTX 4090",
            },
            {
                "id": "#7992",
                "title": "coalesce_grad_reduction — N reduce-scatters → 1 for GRPO",
                "fix": "Use engine.coalesce_grad_reduction() for multi-loss backward",
            },
        ],
        "experimental": [
            {
                "id": "#8027",
                "title": "OPD distillation trainer (DRAFT) — LoRA+OPD gap exists",
                "fix": "~15 LOC to add LoRAConfig to StudentConfig → 60x optimizer reduction",
            },
        ],
    },
    "vllm": {
        "critical": [
            {
                "id": "#39096",
                "title": "SM89 batch invariance bug — Inductor RMSNorm fusion",
                "fix": "enforce_eager=True (disable compile) OR Inductor Fusion Guard (PR draft ready)",
                "source": "CachingAutotuner XBLOCK varies, tl.sum() accumulation order varies",
                "tool": "sm89_batch_invariance_diagnostic.py --mode check",
            },
        ],
        "stability": [
            {
                "id": "FP8 KV paths",
                "title": "3 FP8 KV paths must be distinguished on SM89",
                "fix": "Triton FP8 ALLOWED (#43914) / FlashInfer FP8 NOT / compressed-tensors crash",
                "tool": "sm89_kv_cache_cost_analyzer.py",
            },
        ],
        "config": [
            {
                "id": "BudgetRefiner",
                "title": "BudgetRefiner SLO scheduling — UNIQUE contribution",
                "fix": "profile_table.csv collection needed (P10)",
                "tool": "profile_vllm_budget.py --mode collect",
            },
            {
                "id": "Watermark",
                "title": "Watermark preemption prevention — merged v0.23.0",
                "fix": "Set watermark=0.05 for RTX 4090",
            },
        ],
        "monitor": [
            {
                "id": "#45731",
                "title": "PyTorch 2.13.0 proposed — Triton 3.7.1 may change autotuning",
                "fix": "Monitor for SM89 batch invariance root cause shift",
            },
            {
                "id": "#45683",
                "title": "Deterministic MoE combine under VLLM_BATCH_INVARIANT (89 additions)",
                "fix": "Track for GRPO MoE → DP+EP needs deterministic combine for stable rewards",
            },
            {
                "id": "#45819",
                "title": "GDN attention batch invariance support (13 additions)",
                "fix": "Track for Qwen3.6 hybrid Mamba+GDN deterministic inference",
            },
            {
                "id": "MRv2",
                "title": "MRv2 default expanding — verl may be safe",
                "fix": "VLLM_USE_V2_MODEL_RUNNER=0 as conservative fallback",
            },
        ],
    },
    "verl": {
        "critical": [
            {
                "id": "bypass_mode",
                "title": "bypass_mode=True mandatory on RTX 4090 — skip ref model (14GB savings)",
                "fix": "bypass_mode=True in all RTX 4090 configs",
                "tool": "grpo_troubleshooter_4090.py --mode check",
            },
            {
                "id": "#6699",
                "title": "model_output detach fix — 4x memory reduction (64→16.2 GiB)",
                "fix": "Use verl >= commit with #6699 (MERGED June 12) — detach tensors before building output dict",
                "source": "log_probs/entropy STILL ATTACHED to autograd graph → pinned checkpoint frames → OOM at micro-batch #400",
                "tool": "rtx4090_grpo_config_reference.py --mode validate",
            },
        ],
        "stability": [
            {
                "id": "detach_metrics",
                "title": "detach_metrics_per_micro_batch=True prevents 0.27GiB/step OOM",
                "fix": "Add to config → 28→18GiB",
            },
            {
                "id": "#6735",
                "title": "Cap micro-batch tokens at max_token_len",
                "fix": "Prevents OOM from Karmarkar-Karp imbalance",
            },
        ],
        "config": [
            {
                "id": "CPPO+bypass",
                "title": "CPPO (#6731) + bypass_mode = RTX 4090 #1 BEST trust region",
                "fix": "CPPO MUST use bypass_mode (divergence vs rollout mu, not pi_old)",
            },
            {
                "id": "#6717",
                "title": "TinkerTrainingWorker — split training primitives IN verl",
                "fix": "optimizer_zero_grad/forward_backward/optimizer_step → no framework switch needed",
            },
            {
                "id": "#6736",
                "title": "Off-policy staleness metrics for async GRPO",
                "fix": "trajectory_spans + trajectory_staleness → monitor async training",
            },
            {
                "id": "#6729",
                "title": "Prepare actor weights before rollout wakeup",
                "fix": "Reduces peak memory overlap → 24GB relief",
            },
        ],
        "monitor": [
            {
                "id": "#6738",
                "title": "SGLang weight sync OOM fix — skip redundant clone",
                "fix": "Track for SGLang rollout path → MoE weight sync on RTX 4090",
            },
            {
                "id": "#6688",
                "title": "LoRA IPC buffer crash fix — cudaErrorIllegalAddress",
                "fix": "MERGED June 12 → unmerged LoRA + vLLM + free_cache_engine=True",
            },
            {
                "id": "#6572",
                "title": "Full determinism vLLM rollout — bitwise-aligned reward curves",
                "fix": "OPEN → SM89 requires P9 Fusion Guard → track for deployment",
            },
            {
                "id": "#6790",
                "title": "Separate async trainer — colocated group never switches",
                "fix": "MERGED June 17 → enables separate rollout+training memory optimization",
            },
            {
                "id": "#6782",
                "title": "LoRA rank>32 breaks EOS in vLLM rollout — MUST rank<=32!",
                "fix": "CRITICAL → lora_rank=64/alpha=128 never emits EOS → MUST lora_rank=32/alpha=64",
                "severity": "CRITICAL",
            },
            {
                "id": "#6512",
                "title": "Per-unit LoRA summon — peak memory bounded by largest FSDP unit",
                "fix": "OPEN → reduces peak LoRA sync memory → RTX 4090 benefit",
            },
        ],
    },
    "megatron": {
        "critical": [
            {
                "id": "#5203",
                "title": "Singleton PG crash — dp_cp_params_list=None on single GPU",
                "fix": "Avoid LayerWise optimizer on single GPU",
            },
            {
                "id": "#5394",
                "title": "ChainedOptimizer Muon clipping stalls — same pattern as #8068/#7776",
                "fix": "Skip global norm-clipping for Muon groups → gradient_clipping=0 per-group",
                "source": "Global grad_norm across all sub-optimizers → Newton-Schulz degenerates on near-zero vectors",
            },
        ],
        "stability": [
            {
                "id": "FlashAttention",
                "title": "SM89 FA3/FA4 unavailable, batch_invariant_mode slow",
                "fix": "TE mode = FA2 on SM89, local mode = no FA (pure PyTorch)",
            },
        ],
        "config": [
            {
                "id": "#5349",
                "title": "Quantile Balancing MoE routing — replaces aux loss",
                "fix": "qb_beta per-expert bias, moe_aux_loss_coeff MUST 0",
            },
            {
                "id": "Mamba prefix",
                "title": "Mamba SSM state 40x smaller than KV cache",
                "fix": "Hybrid Mamba-Transformer = 3.7x more concurrent on RTX 4090",
            },
            {
                "id": "DistributedOptimizer",
                "title": "DistributedOptimizer useless on single GPU (dp=1)",
                "fix": "Avoid on RTX 4090",
            },
        ],
        "monitor": [
            {
                "id": "#5309",
                "title": "SSM dtype configurable (APPROVED) — bf16/fp32 choice",
                "fix": "Track for hybrid model memory optimization",
            },
            {
                "id": "#5391",
                "title": "Compact LayerWise DDP — removes dp padding → RTX 4090 memory efficiency",
                "fix": "Track for Muon+ZeRO-2 memory optimization",
            },
            {
                "id": "#5386/#5384",
                "title": "DSA Indexer Replay for RL — extends RouterReplay to sparse attention",
                "fix": "3404 LOC → DeepSeek V4 stability → same singleton-registry pattern",
            },
            {
                "id": "#5219",
                "title": "Single-GPU Muon crash fix — close to merge",
                "fix": "None guard for dp_cp_params_list → essential for single GPU",
            },
        ],
    },
    "sglang": {
        "critical": [],
        "stability": [
            {
                "id": "#27097",
                "title": "Concurrent multi-LoRA diverge under deterministic inference",
                "fix": "Sequential is stable → concurrency problem → needs watching for GRPO LoRA",
            },
        ],
        "config": [
            {
                "id": "deterministic",
                "title": "Deterministic inference → batch-invariant by design (9+ aten overrides KERNEL-level)",
                "fix": "--enable-deterministic-inference, Triton backend recommended SM89",
                "source": "#24459 MERGED → rms_norm + mm.dtype overrides → 9+ overrides total",
            },
            {
                "id": "RadixAttention",
                "title": "Prefix KV reuse via tree-based cache",
                "fix": "GRPO prefix reuse benefit, Triton backend has radix + deterministic",
            },
        ],
        "monitor": [
            {
                "id": "#28354",
                "title": "NVFP4 MoE quantization → RTX 5090 NEXT-PHASE window",
                "fix": "SM120 FP4/MXFP4 kernel gap = contribution opportunity",
            },
            {
                "id": "#28355",
                "title": "Cutlass FP8 MoE EP1 regression -43~50%",
                "fix": "Validates Triton MoE runner as RTX 4090 correct choice",
            },
        ],
    },
    "rllm": {
        "critical": [
            {
                "id": "#605",
                "title": "GRPO grouping bug — trajectory.uid vs task_ids → group size 1 → BROKEN",
                "fix": "1 line fix for enable=False → few lines for per_step → ranking downgraded #1→#3",
                "source": "batch.non_tensor_batch['uid'] = batch.non_tensor_batch['step_ids'] → wrong grouping",
            },
        ],
        "stability": [],
        "config": [
            {
                "id": "Tinker",
                "title": "Tinker #1 RTX 4090 GRPO — in-process, auto LoRA, bypass default",
                "fix": "lora_rank=32, group_size=4, batch_size=8, bypass=true",
                "tool": "train_tinker_rtx4090.sh",
            },
            {
                "id": "async",
                "title": "Async Trainer — truly concurrent (2 asyncio loops)",
                "fix": "enable=true, mini_batch=8, fwd_bwd=4, staleness=0",
            },
        ],
        "monitor": [
            {
                "id": "#653",
                "title": "SWE-RL cookbook — SWE-bench Verified",
                "fix": "Track for agent RL environment",
            },
        ],
    },
    "mindie": {
        "critical": [],
        "stability": [],
        "config": [
            {
                "id": "ATB compose",
                "title": "Compose-level fusion = Ascend unique (NVIDIA has NO compose API)",
                "fix": "Track for NPU inference architecture",
            },
            {
                "id": "BudgetRefiner SLO",
                "title": "58 lines GPU-generic — RTX 4090 profile data unique",
                "fix": "Contribute to vLLM upstream (#1 priority)",
            },
        ],
        "monitor": [
            {
                "id": "DeepEP-Ascend",
                "title": "HCCL DeepEP planned (#8550) — not yet implemented",
                "fix": "Track for Ascend MoE EP future",
            },
        ],
    },
    "pytorch": {
        "critical": [],
        "stability": [],
        "config": [
            {
                "id": "Fusion Guard",
                "title": "Inductor SM<90 Fusion Guard — P9 PyTorch upstream PR",
                "fix": "5-line choices.py: props.major < 9 → WhyNoFuse → return False",
                "tool": "sm89_batch_invariance_diagnostic.py --mode diagnose",
            },
        ],
        "monitor": [
            {
                "id": "#187275",
                "title": "Persistent reduction RBLOCK fix — confirms batch invariance root cause",
                "fix": "Strengthens our Inductor Fusion Guard case",
            },
            {
                "id": "#187435",
                "title": "no_fuse_region per-op fusion barrier — complementary to P9",
                "fix": "804 LOC per-op barrier + 5 LOC P9 global = complete solution",
            },
            {
                "id": "#184119",
                "title": "SM89 fp8→bf16 prologue fusion guard → VALIDATES P9 thesis!",
                "fix": "Blocks fp8-to-bf16 fusion into mm template on pre-sm90 → same Inductor SM89 class",
            },
            {
                "id": "Non-TMA",
                "title": "Non-TMA Triton templates for SM89 — Phase 1 of 2-phase fix",
                "fix": "Does NOT fix batch invariance root cause directly",
            },
            {
                "id": "v2.12",
                "title": "max_autotune layout deferral now opt-in — reduces SM89 risk by default",
                "fix": "P9 still needed for opt-in users → issue draft ready to file",
            },
        ],
    },
}


def print_matrix(framework=None, category=None):
    """Print the safety matrix filtered by framework/category."""
    severity_order = ["critical", "stability", "config", "monitor", "experimental"]
    icons = {"critical": "✗", "stability": "⚠", "config": "◆", "monitor": "◉", "experimental": "?"}
    colors = {"critical": "\033[0;31m", "stability": "\033[1;33m", "config": "\033[0;34m",
              "monitor": "\033[0;35m", "experimental": "\033[0;36m"}

    frameworks = [framework] if framework else list(MATRIX.keys())
    categories = [category] if category else severity_order

    total_items = 0
    for fw in frameworks:
        fw_data = MATRIX.get(fw, {})
        for cat in categories:
            items = fw_data.get(cat, [])
            if not items:
                continue
            color = colors.get(cat, "\033[0m")
            icon = icons.get(cat, "?")
            nc = "\033[0m"
            for item in items:
                total_items += 1
                print(f"{color}[{cat.upper()}] {icon} {fw}/{item['id']}: {item['title']}{nc}")
                print(f"  FIX: {item['fix']}")
                if 'source' in item:
                    print(f"  SOURCE: {item['source']}")
                if 'tool' in item:
                    print(f"  TOOL: {item['tool']}")
                print()

    return total_items


def print_summary():
    """Print summary statistics."""
    print("=" * 70)
    print("RTX 4090 Cross-Framework Safety Matrix — Summary")
    print("=" * 70)
    print()

    severity_counts = {"critical": 0, "stability": 0, "config": 0, "monitor": 0, "experimental": 0}
    fw_counts = {}

    for fw, cats in MATRIX.items():
        fw_count = 0
        for cat, items in cats.items():
            severity_counts[cat] += len(items)
            fw_count += len(items)
        fw_counts[fw] = fw_count

    print("Framework coverage:")
    for fw, count in fw_counts.items():
        print(f"  {fw}: {count} items")
    print()

    print("Severity distribution:")
    for cat, count in severity_counts.items():
        icon = {"critical": "✗", "stability": "⚠", "config": "◆", "monitor": "◉", "experimental": "?"}[cat]
        print(f"  {icon} {cat}: {count}")
    print()

    print("★★★★★★★★★ Top 3 CRITICAL items:")
    for fw, cats in MATRIX.items():
        for item in cats.get("critical", []):
            print(f"  ✗ {fw}/{item['id']}: {item['title']}")

    print()
    print("★★★★★★★★★ Recommended immediate actions:")
    print("  1. DeepSpeed: overlap_comm=False + gradient_clipping=1.0")
    print("  2. vLLM: enforce_eager=True (or Inductor Fusion Guard when merged)")
    print("  3. verl: bypass_mode=True + detach_metrics=True")
    print()
    print("★★★★★★★★★ GPU needed for:")
    print("  1. BudgetRefiner profile_table.csv (P10 UNIQUE)")
    print("  2. SM89 batch invariance repro (P9)")
    print("  3. SGLang deterministic vs vLLM enforce_eager (P7)")


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 Cross-Framework Safety Matrix")
    parser.add_argument("--mode", choices=["matrix", "framework", "category", "summary"],
                        required=True)
    parser.add_argument("--framework",
                        choices=list(MATRIX.keys()),
                        help="Filter by framework")
    parser.add_argument("--category",
                        choices=["critical", "stability", "config", "monitor", "experimental"],
                        help="Filter by category")

    args = parser.parse_args()

    if args.mode == "summary":
        print_summary()
    elif args.mode == "framework":
        print_matrix(framework=args.framework)
    elif args.mode == "category":
        print_matrix(category=args.category)
    else:
        print_matrix()


if __name__ == "__main__":
    main()
