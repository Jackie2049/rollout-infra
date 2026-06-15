#!/usr/bin/env python3
"""7-Framework AI Infra Quick Advisor for RTX 4090

Generates personalized recommendations across DeepSpeed, Megatron-LM, vLLM,
verl, MindIE, rLLM, and PyTorch based on workload type and constraints.

Modes:
  - advisor: Interactive advisor — pick a workload, get cross-framework recommendation
  - compare: Compare all 7 frameworks for a specific task
  - roadmap: Show RTX 4090 learning roadmap with priority ordering
  - gaps: Identify knowledge gaps and suggest next study topics
  - matrix: Print the full RTX 4090 framework compatibility matrix

Based on: notebook/projects/ (7 framework reading notes)
"""

import argparse
import sys
from pathlib import Path

# ============================================================
# 7-Framework Data Model
# ============================================================

FRAMEWORKS = {
    "vLLM": {
        "version": "v0.23.0",
        "notes_count": 38,
        "coverage": "★★★★★ Very thorough",
        "rtx4090_viability": "★★★★★ BEST for inference serving",
        "key_strength": "V1 architecture, MRv2, INT4 Triton fallback, HMA, prefix caching",
        "key_limitation": "FP8 KV crash (#44879), batch invariance (#39096), MRv2 no INT4",
        "best_workload": "INT4 GPTQ inference, GRPO rollout engine, BF16 dense serving",
        "production_path": "INT8 KV + prefix caching + INT4 Marlin/Triton → full production",
        "oss_contribution_opportunities": 6,
        "tier1_issues": ["#44879 FP8 KV", "#44701 prefix hash", "#39096 batch invariance"],
    },
    "verl": {
        "version": "v0.8+",
        "notes_count": 22,
        "coverage": "★★★★ Very thorough",
        "rtx4090_viability": "★★★★ BEST for GRPO training (with bypass_mode)",
        "key_strength": "GRPO/PPO, bypass_mode=True, TransferQueue zero-copy KV, async rollout",
        "key_limitation": "ZERO MRv2 handling, detach_metrics needed for OOM, HYBRID mode complexity",
        "best_workload": "GRPO training with vLLM rollout, PPO training, async RL training",
        "production_path": "bypass_mode=True + detach_metrics + INT4 + INT8 KV → RTX 4090 GRPO",
        "oss_contribution_opportunities": 2,
        "tier1_issues": ["#6401 RFC full-model PS"],
    },
    "rLLM": {
        "version": "v0.3.0-pre",
        "notes_count": 6,
        "coverage": "★★★ Decent",
        "rtx4090_viability": "★★★★★ BEST for single-GPU GRPO (Tinker zero-copy)",
        "key_strength": "TinkerBackend in-process zero-copy, bypass default, auto-safe, VLM support",
        "key_limitation": "Smaller community, fewer model recipes, agent RL still experimental",
        "best_workload": "Single-GPU GRPO training, agent RL, math/code RL, VLM RL",
        "production_path": "TinkerBackend + GRPO + LoRA-32 + bypass_mode → single GPU optimal",
        "oss_contribution_opportunities": 1,
        "tier1_issues": [],
    },
    "DeepSpeed": {
        "version": "v0.19+",
        "notes_count": 13,
        "coverage": "★★★★★ Very thorough",
        "rtx4090_viability": "★★★ ZeRO-3 useful for multi-GPU, single GPU less useful",
        "key_strength": "ZeRO-1/2/3, NVMe offload, communication overlap, DeepCompile, AutoEP",
        "key_limitation": "ZeRO-3 MoE conflict, single GPU less useful than LoRA+compile",
        "best_workload": "Multi-GPU distributed training, MoE training (ZeRO-1+EP), NVMe offload",
        "production_path": "ZeRO-2 + NVMe offload → multi-GPU training; ZeRO-3 → large model",
        "oss_contribution_opportunities": 0,
        "tier1_issues": [],
    },
    "Megatron-LM": {
        "version": "v0.17+",
        "notes_count": 9,
        "coverage": "★★★★ Good",
        "rtx4090_viability": "★★ NOT viable for RTX 4090 (single GPU crash, no LoRA)",
        "key_strength": "TP+PP+EP, inference engine, MCore, TensorRT-LLM export",
        "key_limitation": "★ Single-GPU LayerWise CRASH (#5203), NO LoRA, SM90-only kernels",
        "best_workload": "Multi-node distributed training (8+ GPUs), MoE EP, export to TRT-LLM",
        "production_path": "NOT viable for single RTX 4090 → use verl/rLLM instead",
        "oss_contribution_opportunities": 1,
        "tier1_issues": ["#5203 single GPU crash"],
    },
    "MindIE": {
        "version": "MindIE-Service latest",
        "notes_count": 5,
        "coverage": "★★★ Good (growing)",
        "rtx4090_viability": "✗✗ Not applicable (Ascend NPU only)",
        "key_strength": "ATB kernel, vLLM-Ascend bridge, MC2+EPLB MoE EP, MXFP4 on Ascend",
        "key_limitation": "Ascend-only, CANN dependency, less community than NVIDIA ecosystem",
        "best_workload": "Ascend NPU inference (A5/950B/910C), MoE EP on Ascend",
        "production_path": "vLLM-Ascend op-level patch → production serving on Ascend NPUs",
        "oss_contribution_opportunities": 0,
        "tier1_issues": [],
    },
    "PyTorch": {
        "version": "v2.12.0",
        "notes_count": 10,
        "coverage": "★★★★ Good",
        "rtx4090_viability": "★★★★ Foundation for all frameworks",
        "key_strength": "FSDP2, torch.compile, DTensor, Inductor, custom ops, CUDA 13.0",
        "key_limitation": "FSDP2 useless for single GPU, Inductor batch invariance on SM89",
        "best_workload": "Foundation layer for all frameworks, torch.compile+LoRA on RTX 4090",
        "production_path": "torch.compile + LoRA-32 → single GPU fine-tuning; FSDP2 → multi-GPU",
        "oss_contribution_opportunities": 0,
        "tier1_issues": [],
    },
}

WORKLOAD_MATRIX = {
    "GRPO training (single GPU)": {
        "best": "rLLM Tinker ★★★★★",
        "alternatives": ["verl bypass_mode ★★★★", "Megatron ✗✗ (crash)"],
        "config": "rllm train → TinkerBackend → GRPO+LoRA-32+bypass_mode → 7B INT4",
        "key_insight": "rLLM Tinker = in-process zero-copy → no detach needed → auto-safe",
    },
    "GRPO training (multi GPU)": {
        "best": "verl HYBRID ★★★★★",
        "alternatives": ["rLLM ★★★★", "DeepSpeed ZeRO+verl ★★★"],
        "config": "verl → HYBRID mode → vLLM rollout + FSDP training → bypass_mode=True",
        "key_insight": "verl HYBRID = dedicated rollout GPU + training GPUs → most scalable",
    },
    "INT4 inference serving": {
        "best": "vLLM ★★★★★",
        "alternatives": ["SGLang ★★★★", "MindIE-Ascend ✗ (Ascend only)"],
        "config": "vLLM v0.23 → INT8 KV + prefix caching + INT4 Marlin → MRv1",
        "key_insight": "INT4 Triton fallback (#43731) → more models work on RTX 4090",
    },
    "BF16 dense inference": {
        "best": "vLLM MRv2 ★★★★★",
        "alternatives": ["SGLang ★★★★", "TensorRT-LLM ★★★"],
        "config": "vLLM v0.23 → MRv2 auto for Llama/Mistral/Qwen3 → INT8 KV",
        "key_insight": "MRv2 FlashInfer sampler + BCG → better throughput on SM89",
    },
    "MoE inference": {
        "best": "vLLM ★★★★ (with HMA)",
        "alternatives": ["DeepSpeed inference ★★★", "MindIE-Ascend ★★★ (Ascend)"],
        "config": "vLLM → HMA-by-default → INT8 KV → max-num-seqs=48",
        "key_insight": "HMA (#41847) prevents startup OOM for MoE on 24GB VRAM",
    },
    "Multi-GPU distributed training": {
        "best": "DeepSpeed ZeRO ★★★★",
        "alternatives": ["PyTorch FSDP2 ★★★★", "Megatron TP+PP ★★★★"],
        "config": "DeepSpeed ZeRO-2 → gradient checkpointing → communication overlap",
        "key_insight": "ZeRO-2 vs FSDP2: similar memory savings, FSDP2 = 2Ψ vs ZeRO-3 = 3Ψ",
    },
    "Single-GPU fine-tuning": {
        "best": "PyTorch compile+LoRA ★★★★★",
        "alternatives": ["DeepSpeed ZeRO-1 ★★★", "verl LoRA ★★★★"],
        "config": "torch.compile + LoRA-32 + BF16 → 7B on 24GB",
        "key_insight": "torch.compile+LoRA > FSDP2 for single GPU (FSDP2 needs 2+ GPUs)",
    },
    "Ascend NPU inference": {
        "best": "vLLM-Ascend ★★★★★",
        "alternatives": ["SGLang-Ascend ★★★", "MindIE raw ★★★"],
        "config": "vLLM-Ascend → op-level patch → A5/950B/910C → MC2+EPLB for MoE",
        "key_insight": "vLLM-Ascend = op-level patch → more scheduling control than SGLang-Ascend",
    },
}

LEARNING_ROADMAP = [
    ("Phase 1: Foundation (current)", [
        "vLLM V1 architecture → serving backbone for all RL training",
        "verl GRPO training → bypass_mode + detach_metrics → RTX 4090 path",
        "rLLM Tinker backend → single GPU zero-copy → GRPO optimal",
        "PyTorch compile+LoRA → single GPU fine-tuning foundation",
    ]),
    ("Phase 2: Deep Dive (next 2 weeks)", [
        "vLLM batch invariance (#39096) → deep research → possible OSS PR",
        "vLLM QuantKey refactor (#32268) → systematic SM89 FP8 guard",
        "verl MRv2 interaction → update vLLM → need explicit handling",
        "MindIE/vLLM-Ascend → Ascend serving → alternative ecosystem",
        "DeepSpeed ZeRO MoE conflict → resolve or document workaround",
    ]),
    ("Phase 3: OSS Contribution (next 1-2 months)", [
        "Tier 1: vLLM #39096 batch invariance → deep analysis → PR attempt",
        "Tier 2: vLLM #32268 QuantKey refactor → code contribution",
        "Tier 3: vLLM #43204/#44931 quick merges → easy wins",
        "NEXT-PHASE: SM120 FP4/MXFP4 kernel → RTX 5090 contribution window",
    ]),
    ("Phase 4: Expert Level (3+ months)", [
        "vLLM MRv2 INT4/GPTQ support → make quantized models MRv2-compatible",
        "verl + rLLM integration → bridge the two RL frameworks",
        "PyTorch Inductor SM89 → fix batch invariance in Inductor layer",
        "Ascend ecosystem → vLLM-Ascend op patches → contribute",
    ]),
]

KNOWLEDGE_GAPS = {
    "MindIE": {
        "current_notes": 5,
        "target_notes": 10,
        "gap_topics": [
            "vLLM-Ascend production deployment guide",
            "Ascend NPU vs NVIDIA GPU performance comparison",
            "CANN 8.x kernel optimization details",
            "DeepEP-Ascend HCCL integration deep dive",
            "MXFP4 on Ascend → FP4 future direction",
        ],
    },
    "rLLM": {
        "current_notes": 6,
        "target_notes": 10,
        "gap_topics": [
            "Tinker VLM training detailed guide",
            "rLLM vs verl performance benchmark",
            "rLLM agent RL cookbook",
            "rLLM snapshot+warm-pool production guide",
            "rLLM cookbook contributions",
        ],
    },
    "verl": {
        "current_notes": 22,
        "target_notes": 25,
        "gap_topics": [
            "verl v0.9+ latest features",
            "verl SGLang PD disaggregation production",
            "verl reward model improvements",
            "verl worker pool lifecycle deep dive",
        ],
    },
    "Megatron-LM": {
        "current_notes": 9,
        "target_notes": 12,
        "gap_topics": [
            "Megatron inference engine deep dive",
            "Megatron Lite (#4885) lightweight runtime",
            "Megatron→TRT-LLM export pipeline",
            "Single-GPU crash workaround (#5203)",
        ],
    },
}


# ============================================================
# Mode implementations
# ============================================================

def mode_advisor(args):
    """Interactive advisor mode."""
    workload = args.workload
    if not workload:
        print("Available workloads:")
        for i, name in enumerate(WORKLOAD_MATRIX, 1):
            print(f"  {i}. {name}")
        print("\nUsage: python tools/seven_framework_advisor.py --mode advisor --workload '<workload_name>'")
        return

    if workload not in WORKLOAD_MATRIX:
        print(f"Unknown workload: {workload}")
        print(f"Options: {', '.join(WORKLOAD_MATRIX.keys())}")
        return

    data = WORKLOAD_MATRIX[workload]
    print(f"=== 7-Framework Advisor: {workload} ===")
    print()
    print(f"★★★★★ Best framework: {data['best']}")
    print(f"Alternatives: {', '.join(data['alternatives'])}")
    print()
    print(f"Recommended config:")
    print(f"  {data['config']}")
    print()
    print(f"Key insight: {data['key_insight']}")
    print()

    # Show framework-specific details
    print("Framework viability for this workload:")
    for fw_name, fw_data in FRAMEWORKS.items():
        viability = fw_data["rtx4090_viability"]
        strength = fw_data["key_strength"]
        limitation = fw_data["key_limitation"]
        print(f"  {fw_name}: {viability}")
        print(f"    Strength: {strength}")
        print(f"    Limitation: {limitation}")
    print()


def mode_compare(args):
    """Compare all 7 frameworks for a task."""
    task = args.workload or "GRPO training"
    print(f"=== 7-Framework Comparison for: {task} ===")
    print()

    # Header
    print(f"{'Framework':<12} {'Version':<12} {'RTX 4090':<20} {'Best For':<30} {'Notes':<6}")
    print("-" * 80)
    for fw_name, fw_data in FRAMEWORKS.items():
        print(f"{fw_name:<12} {fw_data['version']:<12} {fw_data['rtx4090_viability']:<20} "
              f"{fw_data['best_workload'][:30]:<30} {fw_data['notes_count']:<6}")
    print()

    # Recommendation
    if task in WORKLOAD_MATRIX:
        data = WORKLOAD_MATRIX[task]
        print(f"★★★★★ RECOMMENDED: {data['best']}")
        print(f"Config: {data['config']}")
        print(f"Insight: {data['key_insight']}")
    else:
        print("Task not in matrix — showing general RTX 4090 recommendations:")
        print("  GRPO training: rLLM Tinker #1 > verl #2 > Megatron ✗")
        print("  INT4 inference: vLLM #1 (INT8 KV + prefix caching)")
        print("  BF16 serving: vLLM MRv2 #1")
        print("  Fine-tuning: PyTorch compile+LoRA #1")
    print()


def mode_roadmap(args):
    """Show learning roadmap."""
    print("=== RTX 4090 AI Infra Learning Roadmap ===")
    print()

    for phase_name, topics in LEARNING_ROADMAP:
        print(f"## {phase_name}")
        for topic in topics:
            print(f"  • {topic}")
        print()

    print("Current progress:")
    for fw_name, fw_data in FRAMEWORKS.items():
        print(f"  {fw_name}: {fw_data['notes_count']} notes, coverage {fw_data['coverage']}")
    print()

    print("Next immediate actions:")
    print("  1. Deepen MindIE coverage (5 → 10 notes)")
    print("  2. vLLM #39096 batch invariance deep research → possible PR")
    print("  3. GPU experiments when servers come online")
    print("  4. Prepare Tier 1 OSS drafts for posting")
    print()


def mode_gaps(args):
    """Identify knowledge gaps."""
    print("=== 7-Framework Knowledge Gaps ===")
    print()

    for fw_name, gap_data in KNOWLEDGE_GAPS.items():
        current = gap_data["current_notes"]
        target = gap_data["target_notes"]
        progress = f"{current}/{target}"
        gap_pct = int((target - current) / target * 100)
        print(f"## {fw_name} — Progress: {progress} ({gap_pct}% gap)")
        print(f"  Gap topics:")
        for topic in gap_data["gap_topics"]:
            print(f"    • {topic}")
        print()

    # Also show frameworks that are well-covered
    well_covered = []
    for fw_name, fw_data in FRAMEWORKS.items():
        if fw_name not in KNOWLEDGE_GAPS and fw_data["notes_count"] >= 10:
            well_covered.append((fw_name, fw_data["notes_count"], fw_data["coverage"]))
    if well_covered:
        print("Well-covered frameworks (no major gaps):")
        for fw_name, count, coverage in well_covered:
            print(f"  ✓ {fw_name}: {count} notes, {coverage}")
    print()


def mode_matrix(args):
    """Print RTX 4090 framework compatibility matrix."""
    print("=== RTX 4090 Framework Compatibility Matrix ===")
    print()

    categories = [
        ("GRPO Training (single)", "rLLM ★★★★★", "verl ★★★★", "Megatron ✗✗"),
        ("GRPO Training (multi)", "verl ★★★★★", "rLLM ★★★★", "DeepSpeed ★★★"),
        ("INT4 Inference", "vLLM ★★★★★", "SGLang ★★★★", "MindIE ✗"),
        ("BF16 Dense Serving", "vLLM MRv2 ★★★★★", "SGLang ★★★★", "TRT-LLM ★★★"),
        ("MoE Inference", "vLLM ★★★★", "DeepSpeed ★★★", "MindIE ★★★"),
        ("Multi-GPU Training", "DeepSpeed ★★★★", "FSDP2 ★★★★", "Megatron ★★★★"),
        ("Single-GPU Fine-tune", "PyTorch ★★★★★", "verl LoRA ★★★★", "DeepSpeed ★★★"),
        ("Ascend Serving", "vLLM-Ascend ★★★★★", "SGLang-Ascend ★★★", "MindIE ★★★"),
    ]

    print(f"{'Task':<22} {'Best':<18} {'Alt 1':<16} {'Alt 2':<16}")
    print("-" * 72)
    for task, best, alt1, alt2 in categories:
        print(f"{task:<22} {best:<18} {alt1:<16} {alt2:<16}")
    print()

    # Key constraints
    print("RTX 4090 SM89 constraints:")
    print("  ✗ FP8 KV (compressed-tensors override → crash #44879)")
    print("  ✗ Batch invariance (CUDA graphs + Inductor → #39096)")
    print("  ✗ MRv2 for INT4 (quantized = MRv1 only)")
    print("  ✗ DeepSeek-V4-Flash (SM90 exclusive)")
    print("  ✓ INT8 KV (FlashInfer backend → production path)")
    print("  ✓ Triton FP8 (allowed #43914, but slower than INT8)")
    print("  ✓ Prefix caching (7x compute savings for GRPO)")
    print("  ✓ INT4 Marlin + Triton fallback (#43731)")
    print("  ✓ HMA-by-default (#41847 → prevents MoE OOM)")
    print()

    # Strategic insight
    print("★★★★★ Strategic insight:")
    print("  RTX 4090 GRPO optimal stack:")
    print("  rLLM Tinker (training) + vLLM (rollout) → in-process → zero-copy → bypass")
    print("  OR: verl HYBRID (training) + vLLM (rollout) → multi-GPU scaling")
    print()
    print("  RTX 5090 SM120 future direction:")
    print("  FP4/MXFP4 → replaces INT4 → vLLM contribution window")
    print()


def main():
    parser = argparse.ArgumentParser(description="7-Framework AI Infra Quick Advisor for RTX 4090")
    parser.add_argument("--mode", choices=["advisor", "compare", "roadmap", "gaps", "matrix"],
                        default="matrix", help="Output mode")
    parser.add_argument("--workload", default=None,
                        help="Target workload type (for advisor/compare modes)")
    parser.add_argument("--output", default=None, help="Output file path")

    args = parser.parse_args()

    if args.mode == "advisor":
        text = mode_advisor(args)
    elif args.mode == "compare":
        text = mode_compare(args)
    elif args.mode == "roadmap":
        text = mode_roadmap(args)
    elif args.mode == "gaps":
        text = mode_gaps(args)
    elif args.mode == "matrix":
        text = mode_matrix(args)

    if args.output and text:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
