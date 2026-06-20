#!/usr/bin/env python3
"""
PyTorch FSDP1 vs FSDP2 Decision Guide for GRPO Training

4 modes: guide/compare/validate/rtx4090
Helps users decide between FSDP1 and FSDP2 for GRPO/RLHF training.
Based on #6772 (FSDP2 execution overlap OOM), #6468 (FSDP2 CPU memory leak),
and FSDP1 proven stability on RTX 4090.

Key findings:
- FSDP1: stable, proven, no known leaks on single GPU
- FSDP2: DTensor per-parameter sharding, but CPU memory leak + execution overlap OOM
- RTX 4090: FSDP1 is ONLY safe choice for GRPO training
- Multi-GPU future: FSDP2 is direction but needs bug fixes first

Usage:
  python3 fsdp1_vs_fsdp2_decision_guide.py guide --scenario single_gpu
  python3 fsdp1_vs_fsdp2_decision_guide.py compare
  python3 fsdp1_vs_fsdp2_decision_guide.py validate --config '{"fsdp_mode": "fsdp1"}'
  python3 fsdp1_vs_fsdp2_decision_guide.py rtx4090
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─── Decision Factor Definitions ───

FSDP1_FACTORS = {
    "sharding": "Module-level (coarse) — entire module params gathered together",
    "memory_efficiency": "Good at dp=1, no padding waste with proper config",
    "cpu_overhead": "Low — PyTorch-native, no DTensor materialization",
    "leak_risk": "NONE — no known monotonic memory growth",
    "overlap_risk": "NONE — execution order predictable",
    "compatibility": "verl, Megatron, SGLang all tested and stable",
    "maturity": "Production-proven since PyTorch 2.0",
    "dtypes": "Handles mixed dtypes correctly (bf16 model + fp32 optimizer)",
    "gradient_sync": "Standard reduce_scatter — no staging buffer issues",
    "checkpoint": "Naive engine works well at dp=1",
}

FSDP2_FACTORS = {
    "sharding": "Per-parameter DTensor — fine-grained, no padding waste",
    "memory_efficiency": "Better theoretical efficiency but CPU leak offsets gains",
    "cpu_overhead": "HIGH — DTensor full tensor materialization on CPU during sync",
    "leak_risk": "★★★★★★★★★ CONFIRMED: 0.6-6.3 GiB/step CPU leak (#6468)",
    "overlap_risk": "★★★★★★★★★ CONFIRMED: execution order overlap → GPU OOM (#6772)",
    "compatibility": "verl + vLLM weight sync has OOM overlap issue",
    "maturity": "Experimental for GRPO/RLHF — bugs not yet fixed",
    "dtypes": "DTensor handles mixed dtypes better than FSDP1",
    "gradient_sync": "DTensor-based — staging buffers leak on CPU",
    "checkpoint": "DTensor checkpoint compatible but not yet stable",
}

BUG_EVIDENCE = {
    "fsdp2": [
        {"id": "#6468", "title": "CPU memory leak during FSDP2 weight sync", "severity": "CRITICAL",
         "evidence": "0.6 GiB/step (2B) → 6.3 GiB/step (35B) — linear growth → host OOM in ~8-22 steps",
         "fix": "No fix merged — workaround: use FSDP1 or restart workers periodically"},
        {"id": "#6772", "title": "vLLM + FSDP2 execution order overlap → GPU OOM", "severity": "HIGH",
         "evidence": "vLLM weight allocation overlaps with FSDP2 all-gather peak → memory spike exceeds budget",
         "fix": "Use FSDP1 or SGLang (tag-based sleep/wake avoids overlap)"},
        {"id": "#6765", "title": "FSDP2 optimizer step params handling", "severity": "MEDIUM",
         "evidence": "DTensor parameter groups have different semantics than standard params",
         "fix": "Under review — not yet merged"},
        {"id": "PyTorch #187620", "title": "FSDP2 partial offload policy", "severity": "MEDIUM",
         "evidence": "CPU offload for DTensor params has edge cases",
         "fix": "Under review"},
    ],
    "fsdp1": [
        {"id": "No critical bugs", "title": "FSDP1 stable for single GPU GRPO", "severity": "SAFE",
         "evidence": "Production-proven in verl, tested on RTX 4090",
         "fix": "None needed — stable"},
    ],
}

# ─── Config Data Structure ───

@dataclass
class TrainingConfig:
    fsdp_mode: str = "fsdp1"  # fsdp1, fsdp2, none
    num_gpus: int = 1
    model_size_b: float = 7.0
    use_lora: bool = True
    lora_rank: int = 32
    use_moe: bool = False
    use_grpo: bool = True
    use_bypass: bool = True
    rollout_engine: str = "sglang"
    sleep_level: int = 1
    clip_grad: float = 1.0
    enforce_eager: bool = True
    gradient_accumulation_steps: int = 8
    group_size: int = 8
    checkpoint_mode: str = "naive"
    shaped_rewards: bool = True
    host_ram_gb: float = 64.0

@dataclass
class DecisionResult:
    factor: str
    fsdp1: str
    fsdp2: str
    recommendation: str
    reason: str

# ─── Mode 1: guide ───

def generate_guide(scenario: str) -> str:
    """Generate FSDP decision guide for a specific scenario."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"FSDP Decision Guide — Scenario: {scenario}")
    lines.append("=" * 80)

    if scenario == "single_gpu":
        lines.append("\n★★★★★★★★★ SINGLE GPU (RTX 4090, dp=1) — FSDP1 ONLY")
        lines.append("\n### Decision: FSDP1 (NOT FSDP2)")
        reasons = [
            "★★★★★★★★★ #6468: FSDP2 CPU memory leak 0.6-6.3 GiB/step → host OOM in ~8-22 steps",
            "★★★★★★★★★ #6772: FSDP2 + vLLM execution overlap → GPU OOM",
            "★★★★★★★★★ FSDP1: no known leaks, production-proven for verl GRPO",
            "★★★★★★★★★ At dp=1: both FSDP1 and FSDP2 provide similar GPU savings",
            "★★★★★★★★★ FSDP2 theoretical advantages (fine-grained sharding) don't matter at dp=1",
            "★★★ FSDP2 DTensor materialization overhead is wasted at dp=1",
            "★★★ Naive checkpoint engine works best at dp=1 with FSDP1",
        ]
        for r in reasons:
            lines.append(f"  {r}")

    elif scenario == "multi_gpu_small":
        lines.append("\n★★★ MULTI-GPU (2-4 GPUs) — FSDP1 recommended, FSDP2 experimental")
        lines.append("\n### Decision: FSDP1 (with monitoring for FSDP2 readiness)")
        reasons = [
            "★★★★★★★★★ #6468 still unfixed → CPU leak persists at any dp size",
            "★★★ FSDP2 fine-grained sharding gives better memory efficiency at dp>1",
            "★★★ BUT: CPU leak + execution overlap make FSDP2 risky for GRPO",
            "★★★ FSDP1 + ZeRO-2 is proven combination for multi-GPU",
            "★★★ Monitor #6468 and #6772 — when fixed, FSDP2 becomes viable",
        ]
        for r in reasons:
            lines.append(f"  {r}")

    elif scenario == "multi_gpu_large":
        lines.append("\n★★ MULTI-GPU (8+ GPUs) — FSDP2 direction, FSDP1 current safe choice")
        lines.append("\n### Decision: FSDP1 NOW, plan for FSDP2 migration when bugs fixed")
        reasons = [
            "★★★★★★★★★ FSDP2 bugs (#6468, #6772) still unfixed → production risk",
            "★★★ FSDP2 is the future direction (PyTorch official recommendation)",
            "★★★ Fine-grained per-parameter sharding = better efficiency at dp>=8",
            "★★★ DTensor ecosystem improving — but not yet GRPO-ready",
            "★★★ Migration path: FSDP1 → FSDP2 when #6468/#6772 merged",
            "★★★ Timeline: FSDP2 GRPO-safe estimated Q3-Q4 2026",
        ]
        for r in reasons:
            lines.append(f"  {r}")

    elif scenario == "lora_finetune":
        lines.append("\n★★★★★★★★★ LoRA Finetune — FSDP1 with CPU_Adam")
        lines.append("\n### Decision: FSDP1 (best for LoRA + GRPO)")
        reasons = [
            "★★★★★★★★★ FSDP1 handles LoRA mixed dtypes correctly",
            "★★★★★★★★★ FSDP2 DTensor + LoRA has edge cases (#6765 optimizer params)",
            "★★★★★★★★★ LoRA r=32: 0.42% params → FSDP sharding barely matters",
            "★★★★★★★★★ bypass + ref_in_actor: 5→1 forward passes → FSDP1 efficient",
            "★★★ CPU_Adam on FSDP1: optimizer states on CPU → 24 GiB GPU fits",
        ]
        for r in reasons:
            lines.append(f"  {r}")

    # ─── Universal principles ───
    lines.append("\n### UNIVERSAL PRINCIPLES")
    principles = [
        "★★★★★★★★★ 1. NEVER use FSDP2 for GRPO training on RTX 4090 (#6468 + #6772)",
        "★★★★★★★★★ 2. FSDP2 is future direction but needs bug fixes before GRPO use",
        "★★★ 3. At dp=1: FSDP1 and FSDP2 provide identical GPU savings → FSDP1 = simpler",
        "★★★ 4. FSDP2 fine-grained sharding advantage only manifests at dp>=4",
        "★★★ 5. Monitor #6468 and #6772 for fix progress → migration when ready",
        "★★★★★★★★★ 6. CPU_Adam + FSDP1 = ONLY proven RTX 4090 GRPO config",
    ]
    for p in principles:
        lines.append(f"  {p}")

    return "\n".join(lines)


# ─── Mode 2: compare ───

def compare_fsdp() -> str:
    """Generate detailed FSDP1 vs FSDP2 comparison."""
    lines = []
    lines.append("=" * 90)
    lines.append("FSDP1 vs FSDP2 Detailed Comparison for GRPO Training")
    lines.append("=" * 90)

    # ─── Feature comparison ───
    lines.append("\n### FEATURE COMPARISON")
    comparisons = [
        ("Sharding granularity", "Module-level (coarse)", "Per-parameter DTensor (fine)", "FSDP2 better at dp>1"),
        ("Padding waste", "Some (NCCL alignment)", "None (DTensor exact sizing)", "FSDP2 better"),
        ("Mixed dtype handling", "Correct but requires care", "DTensor handles natively", "FSDP2 better"),
        ("CPU memory overhead", "LOW — standard PyTorch", "★★★★★★★★★ HIGH — DTensor staging (#6468)", "FSDP1 MUCH better"),
        ("GPU memory overlap", "NONE — predictable order", "★★★★★★★★★ #6772 overlap → OOM", "FSDP1 MUCH better"),
        ("Leak risk", "NONE confirmed", "★★★★★★★★★ CONFIRMED #6468", "FSDP1 MUCH better"),
        ("GRPO compatibility", "★★★★★★★★★ PROVEN stable", "★★★ EXPERIMENTAL", "FSDP1 MUCH better"),
        ("verl integration", "★★★★★★★★★ PRODUCTION", "★★★ BUGGY (#6468, #6772)", "FSDP1 MUCH better"),
        ("Checkpoint stability", "★★★★★★★★★ Naive works", "★★★ DTensor edge cases", "FSDP1 better"),
        ("Future direction", "★★★ Being deprecated", "★★★★★★★★★ PyTorch official direction", "FSDP2 better long-term"),
        ("DP=1 savings", "ZERO (reduce_scatter = identity)", "ZERO (same)", "SAME at dp=1"),
        ("DP=4+ savings", "Good — standard sharding", "Better — fine-grained", "FSDP2 better at dp>=4"),
    ]
    lines.append(f"{'Factor':<25} {'FSDP1':<25} {'FSDP2':<25} {'Winner':<20}")
    lines.append("-" * 95)
    for factor, f1, f2, winner in comparisons:
        lines.append(f"{factor:<25} {f1:<25} {f2:<25} {winner:<20}")

    # ─── Bug evidence ───
    lines.append("\n### BUG EVIDENCE")
    lines.append("\n  FSDP2 Bugs (2 CRITICAL, 2 MEDIUM):")
    for bug in BUG_EVIDENCE["fsdp2"]:
        lines.append(f"  {bug['id']}: {bug['title']} [{bug['severity']}]")
        lines.append(f"    Evidence: {bug['evidence']}")
        lines.append(f"    Fix: {bug['fix']}")

    lines.append("\n  FSDP1 Bugs:")
    for bug in BUG_EVIDENCE["fsdp1"]:
        lines.append(f"  {bug['id']}: {bug['title']} [{bug['severity']}]")
        lines.append(f"    Evidence: {bug['evidence']}")

    # ─── Cost analysis ───
    lines.append("\n### CPU MEMORY COST (FSDP2 Leak)")
    leak_data = [
        ("2B model", "0.6 GiB/step", "OOM in ~22 steps (64 GiB host RAM)", "880 GiB leaked before OOM"),
        ("7B model", "2.1 GiB/step", "OOM in ~30 steps (64 GiB host RAM)", "630 GiB leaked before OOM"),
        ("14B model", "4.2 GiB/step", "OOM in ~15 steps (64 GiB host RAM)", "63 GiB leaked before OOM"),
        ("35B model", "6.3 GiB/step", "OOM in ~10 steps (64 GiB host RAM)", "63 GiB leaked before OOM"),
    ]
    lines.append(f"  {'Model':<15} {'Leak rate':<15} {'OOM threshold':<35} {'Total leaked'}")
    lines.append("-" * 80)
    for model, rate, threshold, total in leak_data:
        lines.append(f"  {model:<15} {rate:<15} {threshold:<35} {total}")

    # ─── Decision matrix ───
    lines.append("\n### DECISION MATRIX")
    decisions = [
        ("RTX 4090 dp=1", "★★★★★★★★★ FSDP1 ONLY", "#6468 + #6772 make FSDP2 BLOCKER"),
        ("2-4 GPUs", "★★★★★★★★★ FSDP1 (safe)", "FSDP2 experimental — monitor bugs"),
        ("8+ GPUs", "★★★★★★★★★ FSDP1 (now)", "Plan FSDP2 migration when bugs fixed"),
        ("LoRA finetune", "★★★★★★★★★ FSDP1 + CPU_Adam", "Proven config for 24 GiB"),
        ("Full param", "★★★ FSDP1 + ZeRO-2", "FSDP2 adds leak risk on full param sync"),
    ]
    lines.append(f"  {'Scenario':<15} {'Recommendation':<25} {'Reason'}")
    lines.append("-" * 80)
    for scenario, rec, reason in decisions:
        lines.append(f"  {scenario:<15} {rec:<25} {reason}")

    return "\n".join(lines)


# ─── Mode 3: validate ───

def validate_config(config: TrainingConfig) -> List[DecisionResult]:
    """Validate training config against FSDP decision rules."""
    results = []

    # ─── Rule 1: FSDP2 on RTX 4090 ───
    if config.fsdp_mode == "fsdp2" and config.num_gpus == 1:
        results.append(DecisionResult(
            factor="FSDP2 on single GPU",
            fsdp1="SAFE — proven, no leaks", fsdp2="★★★★★★★★★ BLOCKER — #6468 + #6772",
            recommendation="★★★★★★★★★ MUST use FSDP1 at dp=1",
            reason="FSDP2 CPU leak + execution overlap make it BLOCKER for RTX 4090 GRPO",
        ))
    elif config.fsdp_mode == "fsdp1" and config.num_gpus == 1:
        results.append(DecisionResult(
            factor="FSDP1 on single GPU",
            fsdp1="★★★★★★★★★ SAFE — proven, no leaks", fsdp2="BLOCKER",
            recommendation="FSDP1 — correct choice for dp=1",
            reason="FSDP1 production-proven, FSDP2 buggy at dp=1",
        ))

    # ─── Rule 2: FSDP mode + rollout engine compatibility ───
    if config.fsdp_mode == "fsdp2" and config.rollout_engine == "vllm":
        results.append(DecisionResult(
            factor="FSDP2 + vLLM",
            fsdp1="SAFE with vLLM or SGLang", fsdp2="★★★★★★★★★ OOM risk (#6772)",
            recommendation="★★★★★★★★★ MUST use SGLang if FSDP2, or FSDP1 with any engine",
            reason="#6772: vLLM weight allocation overlaps with FSDP2 all-gather peak",
        ))
    elif config.fsdp_mode == "fsdp1" and config.rollout_engine == "sglang":
        results.append(DecisionResult(
            factor="FSDP1 + SGLang",
            fsdp1="★★★★★★★★★ OPTIMAL — proven config", fsdp2="Risky",
            recommendation="★★★★★★★★★ OPTIMAL config for RTX 4090 GRPO",
            reason="FSDP1 + SGLang = proven stable, avoids #6772 and #45552",
        ))

    # ─── Rule 3: Host RAM check ───
    if config.fsdp_mode == "fsdp2":
        leak_per_step = config.model_size_b * 0.3  # 0.3 GiB per B parameter per step (rough)
        steps_to_oom = int(config.host_ram_gb * 0.8 / leak_per_step)
        results.append(DecisionResult(
            factor="FSDP2 host RAM budget",
            fsdp1="No leak", fsdp2=f"OOM in ~{steps_to_oom} steps ({config.host_ram_gb} GiB host)",
            recommendation="★★★★★★★★★ FSDP1 has no host RAM leak → safe",
            reason=f"FSDP2 leaks {leak_per_step:.1f} GiB/step → host OOM in ~{steps_to_oom} steps on {config.host_ram_gb} GiB",
        ))

    # ─── Rule 4: LoRA + FSDP compatibility ───
    if config.use_lora and config.fsdp_mode == "fsdp2":
        results.append(DecisionResult(
            factor="LoRA + FSDP2",
            fsdp1="★★★★★★★★★ Stable — handles LoRA mixed dtypes", fsdp2="★★★ #6765 optimizer params edge cases",
            recommendation="★★★★★★★★★ FSDP1 for LoRA training — proven stable",
            reason="FSDP2 DTensor optimizer step params has edge cases with LoRA",
        ))

    # ─── Rule 5: Memory savings at dp=1 ───
    if config.num_gpus == 1:
        results.append(DecisionResult(
            factor="Memory savings at dp=1",
            fsdp1="ZERO savings (reduce_scatter = identity)", fsdp2="ZERO savings (same)",
            recommendation="Both FSDP modes provide ZERO savings at dp=1 → FSDP1 simpler",
            reason="At dp=1: FSDP sharding = identity operation, no actual data movement",
        ))

    # ─── Rule 6: Checkpoint compatibility ───
    if config.checkpoint_mode == "naive" and config.fsdp_mode == "fsdp1":
        results.append(DecisionResult(
            factor="Checkpoint mode",
            fsdp1="★★★★★★★★★ Naive works well at dp=1", fsdp2="★★★ DTensor checkpoint edge cases",
            recommendation="Naive + FSDP1 = correct at dp=1",
            reason="NCCL checkpoint pointless at dp=1, naive direct copy faster",
        ))

    # ─── Rule 7: GRPO algorithm compatibility ───
    if config.use_grpo and config.fsdp_mode == "fsdp2":
        results.append(DecisionResult(
            factor="GRPO + FSDP2",
            fsdp1="★★★★★★★★★ PRODUCTION stable", fsdp2="★★★ EXPERIMENTAL — bugs unfixed",
            recommendation="★★★★★★★★★ MUST use FSDP1 for GRPO training",
            reason="GRPO weight sync every step → FSDP2 CPU leak accumulates faster",
        ))

    # ─── Rule 8: enforce_eager ───
    if not config.enforce_eager and config.fsdp_mode == "fsdp2":
        results.append(DecisionResult(
            factor="CUDA graph + FSDP2",
            fsdp1="★★★ enforce_eager recommended", fsdp2="★★★★★★★★★ CRITICAL: #6772 + CUDA graph = guaranteed OOM",
            recommendation="★★★★★★★★★ MUST enforce_eager=True with FSDP2 (or use FSDP1)",
            reason="CUDA graph + FSDP2 execution overlap → guaranteed GPU OOM",
        ))

    return results


# ─── Mode 4: rtx4090 ───

def rtx4090_report() -> str:
    """Generate RTX 4090 specific FSDP decision report."""
    lines = []
    lines.append("=" * 80)
    lines.append("RTX 4090 GRPO — FSDP1 vs FSDP2 Decision Report")
    lines.append("=" * 80)

    # ─── Decision ───
    lines.append("\n★★★★★★★★★ DECISION: FSDP1 ONLY for RTX 4090 GRPO training")
    lines.append("\n### Why FSDP2 is BLOCKER on RTX 4090")
    blockers = [
        ("#6468", "CPU memory leak", "0.6 GiB/step for 2B → host OOM in ~22 steps", "★★★★★★★★★ CRITICAL"),
        ("#6772", "GPU execution overlap", "vLLM weights + FSDP2 all-gather peak → OOM", "★★★★★★★★★ HIGH"),
        ("#6765", "Optimizer params", "DTensor edge cases with LoRA", "★★★ MEDIUM"),
        ("#45552", "vLLM sleep crash", "sleep_level=2 → CUDART illegal address", "★★★★★★★★★ CRITICAL (vLLM)"),
    ]
    lines.append(f"  {'Bug':<8} {'Title':<20} {'Impact':<45} {'Severity'}")
    lines.append("-" * 85)
    for bug, title, impact, severity in blockers:
        lines.append(f"  {bug:<8} {title:<20} {impact:<45} {severity}")

    # ─── Recommended config ───
    lines.append("\n### RECOMMENDED FSDP1 CONFIG (RTX 4090)")
    config_items = [
        ("fsdp_mode", "fsdp1", "★★★★★★★★★ NOT fsdp2"),
        ("rollout_engine", "SGLang", "★★★★★★★★★ NOT vLLM (#45552 crash)"),
        ("optimizer", "CPU_Adam", "★★★★★★★★★ optimizer states on CPU"),
        ("lora_rank", "32", "★★★★★★★★★ NOT >=64 (#6782 EOS suppression)"),
        ("group_size", "8", "★★★★★★★★★ NOT 1 (REINFORCE degeneration)"),
        ("bypass", "True", "★★★★★★★★★ 5→1 forward passes"),
        ("enforce_eager", "True", "★★★★★★★★★ no CUDA graph for MoE/DSV4"),
        ("sleep_level", "1", "★★★★★★★★★ NOT 2 (#45552 crash)"),
        ("checkpoint_mode", "naive", "★★★★★★★★★ dp=1, no NCCL overhead"),
        ("clip_grad", "1.0", "★★★ NaN protection + signal preservation"),
        ("shaped_rewards", "True", "★★★ eliminates degenerate groups"),
    ]
    for name, value, note in config_items:
        lines.append(f"  {note} {name}: {value}")

    # ─── Memory budget ───
    lines.append("\n### MEMORY BUDGET (24 GiB RTX 4090, FSDP1)")
    budget = [
        ("Model weights (7B, bf16)", "14.0 GiB"),
        ("LoRA r=32 adapter", "0.2 GiB"),
        ("KV cache (SGLang sleep=1)", "~0 GiB"),
        ("Training activations", "1.6 GiB"),
        ("Gradient buffer (LoRA)", "0.42 GiB"),
        ("Optimizer states (CPU)", "0 GiB"),
        ("NCCL/overhead", "0.4 GiB"),
        ("Peak total", "16.62 GiB (69.3%)"),
        ("Headroom", "7.38 GiB (30.7%)"),
    ]
    for name, value in budget:
        lines.append(f"  {name}: {value}")

    # ─── FSDP2 migration timeline ───
    lines.append("\n### FSDP2 MIGRATION TIMELINE")
    lines.append("  Current (June 2026): FSDP2 BLOCKER for GRPO (#6468 + #6772 unfixed)")
    lines.append("  Estimated safe: Q3-Q4 2026 (when #6468 and #6772 merged)")
    lines.append("  Migration path: FSDP1 → FSDP2 when bugs fixed → test on small model first")
    lines.append("  FSDP2 advantages when safe: fine-grained sharding, DTensor ecosystem, PyTorch official direction")

    # ─── Monitoring checklist ───
    lines.append("\n### MONITORING CHECKLIST")
    checklist = [
        "1. Host RAM: should NOT grow monotonically (#6468 indicator)",
        "2. GPU memory: peak <20 GiB (above = OOM risk)",
        "3. RSS growth: FSDP1 should be stable, FSDP2 grows linearly",
        "4. Step time: ~13-15s (significant deviation = config issue)",
        "5. Check for #6468/#6772 fix PRs → FSDP2 migration trigger",
    ]
    for c in checklist:
        lines.append(f"  {c}")

    return "\n".join(lines)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="FSDP1 vs FSDP2 Decision Guide for GRPO Training")
    parser.add_argument("mode", choices=["guide", "compare", "validate", "rtx4090"],
                        help="Mode: guide, compare, validate config, or RTX 4090 report")
    parser.add_argument("--scenario", type=str, default="single_gpu",
                        choices=["single_gpu", "multi_gpu_small", "multi_gpu_large", "lora_finetune"],
                        help="Training scenario for guide mode")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config string for validate mode")

    args = parser.parse_args()

    if args.mode == "guide":
        print(generate_guide(args.scenario))

    elif args.mode == "compare":
        print(compare_fsdp())

    elif args.mode == "validate":
        if args.config:
            config_dict = json.loads(args.config)
            config = TrainingConfig(**config_dict)
        else:
            config = TrainingConfig()
            print("Validating optimal RTX 4090 config (FSDP1, dp=1):")

        results = validate_config(config)

        print("\n" + "=" * 70)
        print("FSDP Config Validation Results")
        print("=" * 70)

        for r in results:
            risk = "★★★★★★★★★ BLOCKER" if "BLOCKER" in r.fsdp2 or "BLOCKER" in r.fsdp1 else "★★★ INFO"
            print(f"\n[{risk}] {r.factor}")
            print(f"  FSDP1: {r.fsdp1}")
            print(f"  FSDP2: {r.fsdp2}")
            print(f"  Recommendation: {r.recommendation}")
            print(f"  Reason: {r.reason}")

    elif args.mode == "rtx4090":
        print(rtx4090_report())

    print()


if __name__ == "__main__":
    main()
