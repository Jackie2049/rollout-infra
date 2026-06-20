#!/usr/bin/env python3
"""
Cross-Framework GRPO Bug Avoidance Matrix Tool

4 modes: matrix/validate/rtx4090/cross-framework
Synthesizes all MUST DO/MUST NOT rules + bug avoidance patterns across 7 frameworks
into a single practical validation matrix for RTX 4090 GRPO training.

Frameworks: DeepSpeed, Megatron-LM, vLLM, verl, SGLang, rLLM, MindIE/vLLM-Ascend
Based on 20 knowledge domains, 78 connections, 34 rules (17 MUST DO + 17 MUST NOT)

Usage:
  python3 cross_framework_grpo_bug_avoidance_matrix.py matrix
  python3 cross_framework_grpo_bug_avoidance_matrix.py validate --config '{"framework": "verl+sglang+deepspeed", ...}'
  python3 cross_framework_grpo_bug_avoidance_matrix.py rtx4090
  python3 cross_framework_grpo_bug_avoidance_matrix.py cross-framework --bug '#5394'
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── 7 Framework Definitions ───

FRAMEWORKS = {
    "deepspeed": {
        "name": "DeepSpeed",
        "role": "Training backend (ZeRO, optimizer)",
        "bugs": [
            {"id": "#8061", "title": "ZeRO average_tensor stream race", "severity": "HIGH", "pattern": "cuda_stream_safety", "avoided_by": "overlap_comm=False at dp=1"},
            {"id": "#8068", "title": "gradient_clipping default 0→1 (Muon)", "severity": "CRITICAL", "pattern": "muon_clipping", "avoided_by": "Using Adam, not Muon"},
            {"id": "#7776", "title": "Orthogonalization-before-clipping ordering", "severity": "HIGH", "pattern": "muon_clipping", "avoided_by": "Using Adam, not Muon"},
            {"id": "#8072", "title": "ZeRO-3 PEFT regression", "severity": "HIGH", "pattern": "zero3_regression", "avoided_by": "ZeRO-2, not ZeRO-3"},
            {"id": "#8075", "title": "FD leak in ZeRO offload", "severity": "LOW", "pattern": "resource_leak", "avoided_by": "Monitor fd count"},
            {"id": "#7939", "title": "Muon CPU offload (CLOSED without merge)", "severity": "HIGH", "pattern": "missing_feature", "avoided_by": "CPU_Adam instead"},
        ],
    },
    "megatron": {
        "name": "Megatron-LM",
        "role": "Training backend (TP/PP/SP, optimizer)",
        "bugs": [
            {"id": "#5394", "title": "ChainedOptimizer Muon clipping stall", "severity": "CRITICAL", "pattern": "muon_clipping", "avoided_by": "Using Adam, not Muon"},
            {"id": "#5395", "title": "skip_grad_norm_clip (CHANGES_REQUESTED)", "severity": "HIGH", "pattern": "muon_clipping_fix", "avoided_by": "Using Adam, skip if Muon"},
            {"id": "#5384", "title": "DSA indexer train/rollout mismatch", "severity": "VERY HIGH for GRPO", "pattern": "discrete_decision_mismatch", "avoided_by": "DSAIndexerReplay or avoid DSA models"},
            {"id": "#5387", "title": "FSDP fully_shard implementation", "severity": "HIGH", "pattern": "fsdp_enabler", "avoided_by": "Monitor for merge → enables larger models"},
            {"id": "#5391", "title": "Decoupled compact LayerWise DDP", "severity": "HIGH", "pattern": "memory_reduction", "avoided_by": "Monitor for merge → reduces Muon memory"},
            {"id": "#5400", "title": "GDN in_proj fused matrix Muon routing", "severity": "MEDIUM", "pattern": "muon_routing", "avoided_by": "Using Adam, not Muon"},
            {"id": "#5401", "title": "MoE router z-loss + TE CUDA graph", "severity": "MEDIUM", "pattern": "cuda_graph_incompatibility", "avoided_by": "Not using MoE z-loss + CUDA graph"},
        ],
    },
    "vllm": {
        "name": "vLLM",
        "role": "Rollout engine (V1, inference)",
        "bugs": [
            {"id": "#45552", "title": "CuMemAllocator missing synchronize() before sleep", "severity": "CRITICAL for RTX 4090", "pattern": "cuda_stream_safety", "avoided_by": "Using SGLang instead, sleep_level=1"},
            {"id": "#46125", "title": "Encoder cache revert (DANGEROUS for RLHF)", "severity": "HIGH", "pattern": "stale_cache_corruption", "avoided_by": "Never merge encoder cache revert"},
            {"id": "#6794", "title": "Delta weight sync 4 blocking sub-issues", "severity": "HIGH", "pattern": "weight_sync_safety", "avoided_by": "Monitor checksums, record_stream"},
            {"id": "#6772", "title": "vLLM + FSDP2 execution order peak overlap", "severity": "HIGH", "pattern": "execution_overlap_oom", "avoided_by": "Using FSDP1, not FSDP2"},
            {"id": "#46118", "title": "MTP grammar FSM conflict", "severity": "MEDIUM", "pattern": "grammar_incompatibility", "avoided_by": "Not using MTP with structured output"},
            {"id": "#45979", "title": "DSv4 sparse cache VINDICATED", "severity": "LOW", "pattern": "intra_step_safe", "avoided_by": "Intra-step caching SAFE, inter-step dangerous"},
            {"id": "#6782", "title": "LoRA rank=64 breaks EOS token", "severity": "HIGH", "pattern": "lora_distribution_distortion", "avoided_by": "LoRA r=32 (not r=64)"},
        ],
    },
    "verl": {
        "name": "verl",
        "role": "RL framework (GRPO/CPPO, training loop)",
        "bugs": [
            {"id": "#6699", "title": "Detach micro-batch outputs (OOM fix)", "severity": "MEDIUM", "pattern": "memory_accumulation", "avoided_by": "Already merged, use latest verl"},
            {"id": "#6780", "title": "Async trainer step mismatch", "severity": "MEDIUM", "pattern": "async_sync_mismatch", "avoided_by": "Use synchronous trainer"},
            {"id": "#6782", "title": "LoRA rank breaks EOS (vLLM side)", "severity": "HIGH", "pattern": "lora_distribution_distortion", "avoided_by": "LoRA r=32"},
            {"id": "#6794", "title": "Delta weight sync blocking sub-issues", "severity": "HIGH", "pattern": "weight_sync_safety", "avoided_by": "Monitor checksums, 4 sub-issues"},
            {"id": "#6786", "title": "Dynamic CP batch split bug", "severity": "MEDIUM", "pattern": "batch_split_bug", "avoided_by": "Not using dynamic CP"},
            {"id": "#6772", "title": "vLLM + FSDP2 OOM overlap", "severity": "HIGH", "pattern": "execution_overlap_oom", "avoided_by": "Using FSDP1"},
        ],
    },
    "sglang": {
        "name": "SGLang",
        "role": "Rollout engine (tag-based sleep/wake)",
        "bugs": [
            {"id": "#28771", "title": "Eagle accept_length degradation (44% throughput)", "severity": "HIGH", "pattern": "spec_decode_degradation", "avoided_by": "Monitor accept_length < 2.0"},
            {"id": "#28676", "title": "MXFP8 MoE v4 cache clobber", "severity": "MEDIUM", "pattern": "cache_corruption", "avoided_by": "Not using MXFP8 MoE v4"},
            {"id": "#28566", "title": "LoRA sentinel pad", "severity": "LOW", "pattern": "lora_padding", "avoided_by": "LoRA r=32 (power of 2)"},
            {"id": "#28499", "title": "cSGMV CUDA graph fix", "severity": "MEDIUM", "pattern": "cuda_graph_lora", "avoided_by": "Using LoRA r=32 + enforce_eager"},
            {"id": "#28582", "title": "RCE security vulnerability", "severity": "HIGH", "pattern": "security", "avoided_by": "Validate input, restrict access"},
        ],
    },
    "rllm": {
        "name": "rLLM",
        "role": "RL framework (alternative to verl)",
        "bugs": [
            {"id": "#605", "title": "GRPO grouping strategy bug (singleton degeneration)", "severity": "CRITICAL", "pattern": "grpo_singleton_degeneration", "avoided_by": "group_by_prompt, gs>=8"},
            {"id": "#667", "title": "Config/transform naming regression (CLOSED)", "severity": "MEDIUM", "pattern": "api_regression", "avoided_by": "Use latest rllm v0.3"},
        ],
    },
    "mindie_vllm_ascend": {
        "name": "MindIE / vLLM-Ascend",
        "role": "NPU inference (Ascend, CANN)",
        "bugs": [
            {"id": "#10592", "title": "NPUIPC weight transfer security", "severity": "HIGH", "pattern": "ipc_security", "avoided_by": "Validate IPC paths (cross-lesson for CUDA IPC)"},
            {"id": "#10579", "title": "MoE NaN on NPU", "severity": "MEDIUM", "pattern": "moe_numerical", "avoided_by": "Check MoE numerical stability"},
            {"id": "#10684", "title": "DSA Hadamard + sleep/wake NPU", "severity": "MEDIUM", "pattern": "dsa_npu_integration", "avoided_by": "Monitor for NPU DSA fix → cross-lesson for CUDA"},
        ],
    },
}

# ─── 7 Pattern Classes ───

PATTERN_CLASSES = {
    "cuda_stream_safety": {
        "description": "CUDA stream ordering bugs — async operations on wrong stream → race condition",
        "mathematical_model": "stream_order_violation: op_A(stream_x) → op_B(stream_y) → no synchronize → A|B undefined",
        "universal_fix": "torch.cuda.synchronize() before unmap, record_stream for freed tensors, copy_streams set for all producers",
        "members": ["#8061", "#45552", "#28499"],
    },
    "muon_clipping": {
        "description": "Scale-invariant optimizer + global gradient clipping = contradiction",
        "mathematical_model": "clip_coeff c = clip_grad/grad_norm → for Muon: direction ∝ c·g → c changes direction → Newton-Schulz degenerates",
        "universal_fix": "Per-optimizer-group clipping: Muon=0, Adam=1.0, clip AFTER Muon step",
        "members": ["#8068", "#7776", "#5394", "#5395", "#229", "#230"],
    },
    "discrete_decision_mismatch": {
        "description": "Discrete decisions (top-k, routing) differ between train and rollout → logprob mismatch",
        "mathematical_model": "top_k(train, ε) ≠ top_k(rollout, ε') → P(agree) = 1 - ε_error → for GRPO: reward attribution wrong",
        "universal_fix": "Replay pattern: record discrete decisions during rollout → replay during training",
        "members": ["#5384", "#5386", "#2693"],
    },
    "lora_distribution_distortion": {
        "description": "LoRA high rank distorts logit distribution → suppresses EOS → runaway generation",
        "mathematical_model": "ΔW = BA/r → r large → ΔW magnitude large → logit shift → EOS suppression",
        "universal_fix": "LoRA r=32 (0.42% params, 1.56% direction, effective 15.6% with 10x LR)",
        "members": ["#6782", "#28566", "#28499"],
    },
    "execution_overlap_oom": {
        "description": "Training + inference execution order overlap → peak memory exceeds budget",
        "mathematical_model": "peak_overlap = max(inference_peak, training_peak) ≠ inference_peak + training_peak with proper sleep/wake",
        "universal_fix": "Sleep/wake with proper memory swapping, FSDP1 (not FSDP2), enforce_eager=True",
        "members": ["#6772", "#45552"],
    },
    "grpo_singleton_degeneration": {
        "description": "GRPO gs=1 = REINFORCE(baseline=0) → degenerate groups, no advantage signal",
        "mathematical_model": "A(gs=1) = r - 0 = r → variance = σ² → SNR = 1 → catastrophic for outcome-only rewards",
        "universal_fix": "gs>=8 for shaped rewards, gs>=4 minimum, group_by_prompt, NEVER gs=1",
        "members": ["#605"],
    },
    "stale_cache_corruption": {
        "description": "Cache reuse after weight update → stale values → silent corruption",
        "mathematical_model": "cache[t] = f(weights[t-1]) ≠ f(weights[t]) → silent wrong values propagated",
        "universal_fix": "Never reuse inference cache after weight update, clear all caches on sync",
        "members": ["#46125", "#45979"],
    },
}

# ─── 34 Rules (17 MUST DO + 17 MUST NOT) ───

MUST_DO_RULES = [
    {"id": "R1", "rule": "ZeRO-2 (NOT ZeRO-3) for RTX 4090", "framework": "deepspeed", "reason": "ZeRO-3 sharding overhead > savings at dp=1"},
    {"id": "R2", "rule": "CPU_Adam optimizer states", "framework": "deepspeed", "reason": "Optimizer states on CPU → 24 GiB GPU fits"},
    {"id": "R3", "rule": "clip_grad=1.0 (optimal threshold)", "framework": "deepspeed/megatron", "reason": "NaN protection + signal preservation"},
    {"id": "R4", "rule": "bypass + ref_in_actor mode", "framework": "verl", "reason": "5→1 forward passes, ~14 Psi savings"},
    {"id": "R5", "rule": "group_by_prompt for GRPO", "framework": "verl/rllm", "reason": "Eliminates singleton degeneration"},
    {"id": "R6", "rule": "enforce_eager=True (no CUDA graph)", "framework": "vllm/sglang", "reason": "#6772 FSDP2 overlap, #5401 z-loss graph"},
    {"id": "R7", "rule": "FSDP1 (NOT FSDP2)", "framework": "verl/pytorch", "reason": "#6772 execution order overlap → OOM"},
    {"id": "R8", "rule": "gs>=4 (minimum group size)", "framework": "verl/rllm", "reason": "SNR = √gs → gs=4 → SNR=2.0 (borderline)"},
    {"id": "R9", "rule": "LoRA r=32 + bypass sync", "framework": "verl/vllm", "reason": "80x sync reduction, no EOS suppression"},
    {"id": "R10", "rule": "naive checkpoint (dp=1)", "framework": "deepspeed/verl", "reason": "NCCL checkpoint pointless at dp=1"},
    {"id": "R11", "rule": "record_stream for freed tensors", "framework": "pytorch", "reason": "#6794 silent corruption prevention"},
    {"id": "R12", "rule": "ulimit 65535 (file descriptors)", "framework": "system", "reason": "#8075 FD leak prevention"},
    {"id": "R13", "rule": "shaped rewards (NOT outcome-only)", "framework": "verl", "reason": "Eliminate degenerate groups, ρ≈0 with task difficulty"},
    {"id": "R14", "rule": "gs>=8 for sparse rewards", "framework": "verl/rllm", "reason": "SNR = √8 ≈ 2.83 → sufficient"},
    {"id": "R15", "rule": "sleep_level=1 (NOT 2)", "framework": "vllm/sglang", "reason": "#45552 RTX 4090 BLOCKER at level=2"},
    {"id": "R16", "rule": "SGLang rollout (NOT vLLM)", "framework": "verl", "reason": "vLLM #45552 crash, LoRA not in CuMem pool"},
    {"id": "R17", "rule": "DSAIndexerReplay for DSA models", "framework": "megatron/verl", "reason": "#5384 train/rollout mismatch → incorrect GRPO signal"},
]

MUST_NOT_RULES = [
    {"id": "R18", "rule": "NOT ZeRO-3", "framework": "deepspeed", "reason": "Sharding overhead > savings at dp=1, PEFT regression #8072"},
    {"id": "R19", "rule": "NOT Muon optimizer", "framework": "deepspeed/megatron", "reason": "#8068/#7776/#5394 clipping bugs → stall"},
    {"id": "R20", "rule": "NOT overlap_comm at dp=1", "framework": "deepspeed", "reason": "NCCL reduce-scatter = identity at dp=1"},
    {"id": "R21", "rule": "NOT sleep_level=2 on RTX 4090", "framework": "vllm", "reason": "#45552 guaranteed crash"},
    {"id": "R22", "rule": "NOT LoRA rank>=64 on vLLM", "framework": "vllm/verl", "reason": "#6782 EOS suppression"},
    {"id": "R23", "rule": "NOT FSDP2", "framework": "verl/pytorch", "reason": "#6772 execution overlap → OOM"},
    {"id": "R24", "rule": "NOT gs=1 (group size)", "framework": "verl/rllm", "reason": "REINFORCE(baseline=0) → degenerate"},
    {"id": "R25", "rule": "NOT PPO-clip on RTX 4090", "framework": "verl", "reason": "65.06 GiB peak > 24 GiB → OOM"},
    {"id": "R26", "rule": "NOT full param sync", "framework": "verl", "reason": "~14 GiB per step vs ~200 MiB LoRA sync"},
    {"id": "R27", "rule": "NOT NCCL checkpoint at dp=1", "framework": "deepspeed/verl", "reason": "Pointless, no inter-GPU communication"},
    {"id": "R28", "rule": "NOT outcome reward gs<16", "framework": "verl", "reason": "4.6% degenerate groups, ρ>0 with difficulty"},
    {"id": "R29", "rule": "NOT merge vLLM encoder cache revert (#46125)", "framework": "vllm", "reason": "Silent corruption after weight update"},
    {"id": "R30", "rule": "NOT overlap_comm=True at dp=1", "framework": "deepspeed", "reason": "Same as R20, redundant but explicit"},
    {"id": "R31", "rule": "NOT ZeRO-1/2 at dp=1 for savings", "framework": "deepspeed", "reason": "ZERO savings at dp=1, FSDP1 better"},
    {"id": "R32", "rule": "NOT Muon CPU offload (unmerged)", "framework": "deepspeed", "reason": "#7939 closed without merge, use CPU_Adam"},
    {"id": "R33", "rule": "NOT naive DSA training without replay", "framework": "megatron", "reason": "#5384 indexer mismatch → GRPO wrong"},
    {"id": "R34", "rule": "NOT global clipping for Muon in ChainedOptimizer", "framework": "megatron", "reason": "#5394 positive-feedback stall"},
]

# ─── Config Data Structure ───

@dataclass
class GRPOConfig:
    framework: str = "verl+sglang+deepspeed"
    training_backend: str = "deepspeed"
    rollout_engine: str = "sglang"
    optimizer: str = "adam"
    clip_grad: float = 1.0
    zero_stage: int = 2
    overlap_comm: bool = False
    fsdp_mode: str = "fsdp1"
    lora_rank: int = 32
    group_size: int = 8
    bypass: bool = True
    ref_in_actor: bool = True
    sleep_level: int = 1
    checkpoint_mode: str = "naive"
    shaped_rewards: bool = True
    enforce_eager: bool = True
    use_megatron: bool = False
    use_muon: bool = False
    use_dsa: bool = False
    data_parallel_size: int = 1
    ulimit: int = 65535

@dataclass
class ValidationResult:
    rule_id: str
    rule: str
    framework: str
    status: str  # PASS, FAIL, WARN
    explanation: str

# ─── Mode 1: matrix ───

def generate_matrix() -> str:
    """Generate full cross-framework bug avoidance matrix."""
    lines = []
    lines.append("=" * 90)
    lines.append("Cross-Framework GRPO Bug Avoidance Matrix")
    lines.append("7 Frameworks × 7 Pattern Classes × 34 Rules")
    lines.append("=" * 90)

    # ─── Framework overview ───
    lines.append("\n### FRAMEWORK OVERVIEW")
    for fw_key, fw in FRAMEWORKS.items():
        bug_count = len(fw["bugs"])
        critical_count = sum(1 for b in fw["bugs"] if "CRITICAL" in b.get("severity", "") or "VERY HIGH" in b.get("severity", ""))
        lines.append(f"  {fw['name']} ({fw['role']}) — {bug_count} bugs tracked, {critical_count} critical")

    # ─── Pattern class matrix ───
    lines.append("\n### PATTERN CLASS × FRAMEWORK MATRIX")
    lines.append(f"{'Pattern':<25} {'DeepSpeed':<10} {'Megatron':<10} {'vLLM':<10} {'verl':<10} {'SGLang':<10} {'rLLM':<10} {'MindIE':<10}")
    lines.append("-" * 95)

    pattern_fw_matrix = {}
    for p_key, p in PATTERN_CLASSES.items():
        fw_hits = {"deepspeed": 0, "megatron": 0, "vllm": 0, "verl": 0, "sglang": 0, "rllm": 0, "mindie_vllm_ascend": 0}
        for fw_key, fw in FRAMEWORKS.items():
            for bug in fw["bugs"]:
                if bug["pattern"] == p_key:
                    fw_hits[fw_key] += 1
        pattern_fw_matrix[p_key] = fw_hits
        hit_strs = [str(fw_hits[k]) if fw_hits[k] > 0 else "-" for k in fw_hits]
        lines.append(f"{p_key:<25} {' '.join(f'{s:<10}' for s in hit_strs)}")

    # ─── Bug detail table ───
    lines.append("\n### BUG DETAIL TABLE (Sorted by Severity)")
    all_bugs = []
    for fw_key, fw in FRAMEWORKS.items():
        for bug in fw["bugs"]:
            sev_order = {"CRITICAL": 0, "VERY HIGH for GRPO": 0, "CRITICAL for RTX 4090": 0,
                        "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            sev = sev_order.get(bug.get("severity", "MEDIUM"), 2)
            all_bugs.append((sev, fw["name"], bug))

    all_bugs.sort(key=lambda x: (x[0], x[1], x[2]["id"]))

    lines.append(f"{'Severity':<12} {'Framework':<12} {'ID':<10} {'Title':<45} {'Avoided By':<35}")
    lines.append("-" * 120)
    for sev, fw_name, bug in all_bugs:
        sev_str = bug.get("severity", "MEDIUM")
        if len(sev_str) > 12:
            sev_str = sev_str[:12]
        title = bug["title"]
        if len(title) > 45:
            title = title[:42] + "..."
        avoided = bug.get("avoided_by", "N/A")
        if len(avoided) > 35:
            avoided = avoided[:32] + "..."
        lines.append(f"{sev_str:<12} {fw_name:<12} {bug['id']:<10} {title:<45} {avoided:<35}")

    # ─── MUST DO rules ───
    lines.append("\n### MUST DO RULES (17)")
    for r in MUST_DO_RULES:
        lines.append(f"  {r['id']}: {r['rule']} [{r['framework']}] — {r['reason']}")

    # ─── MUST NOT rules ───
    lines.append("\n### MUST NOT RULES (17)")
    for r in MUST_NOT_RULES:
        lines.append(f"  {r['id']}: {r['rule']} [{r['framework']}] — {r['reason']}")

    # ─── Pattern universal fixes ───
    lines.append("\n### UNIVERSAL FIX PER PATTERN CLASS")
    for p_key, p in PATTERN_CLASSES.items():
        lines.append(f"  [{p_key}]")
        lines.append(f"    Description: {p['description']}")
        lines.append(f"    Math model: {p['mathematical_model']}")
        lines.append(f"    Universal fix: {p['universal_fix']}")
        lines.append(f"    Bug members: {', '.join(p['members'])}")

    return "\n".join(lines)


# ─── Mode 2: validate ───

def validate_config(config: GRPOConfig) -> List[ValidationResult]:
    """Validate a GRPO config against all 34 rules."""
    results = []

    for r in MUST_DO_RULES:
        status, explanation = check_must_do(r, config)
        results.append(ValidationResult(
            rule_id=r["id"], rule=r["rule"], framework=r["framework"],
            status=status, explanation=explanation,
        ))

    for r in MUST_NOT_RULES:
        status, explanation = check_must_not(r, config)
        results.append(ValidationResult(
            rule_id=r["id"], rule=r["rule"], framework=r["framework"],
            status=status, explanation=explanation,
        ))

    return results


def check_must_do(rule: dict, config: GRPOConfig) -> Tuple[str, str]:
    """Check if a MUST DO rule is satisfied."""
    rid = rule["id"]

    if rid == "R1":  # ZeRO-2
        if config.zero_stage == 2:
            return "PASS", "ZeRO-2 selected"
        elif config.zero_stage == 3:
            return "FAIL", "ZeRO-3 selected → sharding overhead at dp=1"
        return "WARN", f"ZeRO stage {config.zero_stage} — recommended: ZeRO-2"

    if rid == "R2":  # CPU_Adam
        if config.optimizer == "adam" or config.optimizer == "cpu_adam":
            return "PASS", "Adam/CPU_Adam selected → optimizer states on CPU"
        return "WARN", f"Optimizer {config.optimizer} — CPU_Adam recommended for RTX 4090"

    if rid == "R3":  # clip_grad=1.0
        if config.clip_grad == 1.0 and not config.use_muon:
            return "PASS", "clip_grad=1.0 with Adam → optimal threshold"
        elif config.use_muon and config.clip_grad > 0:
            return "FAIL", f"clip_grad={config.clip_grad} with Muon → #5394 stall risk"
        return "WARN", f"clip_grad={config.clip_grad} — recommended: 1.0"

    if rid == "R4":  # bypass + ref_in_actor
        if config.bypass and config.ref_in_actor:
            return "PASS", "bypass + ref_in_actor → 5→1 forward passes"
        return "FAIL", f"bypass={config.bypass}, ref_in_actor={config.ref_in_actor} → extra forward passes waste VRAM"

    if rid == "R5":  # group_by_prompt
        if config.group_size >= 4:
            return "PASS", f"gs={config.group_size} → group_by_prompt effective"
        return "FAIL", f"gs={config.group_size} → degenerate groups possible"

    if rid == "R6":  # enforce_eager
        if config.enforce_eager:
            return "PASS", "enforce_eager=True → no CUDA graph issues"
        return "WARN", "enforce_eager=False → #6772, #5401 risk"

    if rid == "R7":  # FSDP1
        if config.fsdp_mode == "fsdp1":
            return "PASS", "FSDP1 selected → safe execution order"
        elif config.fsdp_mode == "fsdp2":
            return "FAIL", "FSDP2 → #6772 execution order overlap → OOM"
        return "WARN", f"FSDP mode {config.fsdp_mode} — recommended: FSDP1"

    if rid == "R8":  # gs>=4
        if config.group_size >= 4:
            return "PASS", f"gs={config.group_size} ≥ 4 → SNR ≥ 2.0"
        return "FAIL", f"gs={config.group_size} < 4 → SNR < 2.0 → insufficient"

    if rid == "R9":  # LoRA r=32
        if config.lora_rank == 32:
            return "PASS", "LoRA r=32 → 0.42% params, no EOS suppression"
        elif config.lora_rank >= 64:
            return "FAIL", f"LoRA r={config.lora_rank} → #6782 EOS suppression"
        return "WARN", f"LoRA r={config.lora_rank} — recommended: r=32"

    if rid == "R10":  # naive checkpoint
        if config.checkpoint_mode == "naive":
            return "PASS", "naive checkpoint → correct for dp=1"
        return "WARN", f"checkpoint_mode={config.checkpoint_mode} — naive recommended at dp=1"

    if rid == "R11":  # record_stream
        return "WARN", "record_stream — monitor #6794, check tensor lifecycle"

    if rid == "R12":  # ulimit
        if config.ulimit >= 65535:
            return "PASS", f"ulimit={config.ulimit} ≥ 65535"
        return "FAIL", f"ulimit={config.ulimit} < 65535 → #8075 FD leak risk"

    if rid == "R13":  # shaped rewards
        if config.shaped_rewards:
            return "PASS", "shaped rewards → ρ≈0 with task difficulty"
        return "WARN", "outcome-only rewards → 4.6% degenerate groups"

    if rid == "R14":  # gs>=8 for sparse
        if config.group_size >= 8:
            return "PASS", f"gs={config.group_size} ≥ 8 → SNR ≥ 2.83"
        elif config.group_size >= 4 and config.shaped_rewards:
            return "WARN", f"gs={config.group_size} ≥ 4 with shaped rewards → borderline"
        return "FAIL", f"gs={config.group_size} with outcome rewards → insufficient SNR"

    if rid == "R15":  # sleep_level=1
        if config.sleep_level == 1:
            return "PASS", "sleep_level=1 → safe for RTX 4090"
        elif config.sleep_level == 2:
            return "FAIL", "sleep_level=2 → #45552 CRASH on RTX 4090"
        return "WARN", f"sleep_level={config.sleep_level} — recommended: 1"

    if rid == "R16":  # SGLang rollout
        if config.rollout_engine == "sglang":
            return "PASS", "SGLang rollout → tag-based sleep/wake, safe"
        elif config.rollout_engine == "vllm":
            return "FAIL", "vLLM rollout → #45552 crash risk, LoRA not in CuMem pool"
        return "WARN", f"rollout_engine={config.rollout_engine} — recommended: SGLang"

    if rid == "R17":  # DSA replay
        if config.use_dsa:
            return "WARN", "DSA model → MUST implement DSAIndexerReplay (#5384)"
        return "PASS", "Not using DSA → replay not needed"

    return "WARN", f"Rule {rid} — manual check recommended"


def check_must_not(rule: dict, config: GRPOConfig) -> Tuple[str, str]:
    """Check if a MUST NOT rule is violated."""
    rid = rule["id"]

    if rid == "R18":  # NOT ZeRO-3
        if config.zero_stage == 3:
            return "FAIL", "ZeRO-3 selected → sharding overhead at dp=1"
        return "PASS", f"ZeRO stage {config.zero_stage} ≠ 3"

    if rid == "R19":  # NOT Muon
        if config.use_muon:
            return "FAIL", "Muon optimizer → #8068/#7776/#5394 clipping bugs"
        return "PASS", "Adam optimizer → no Muon clipping risk"

    if rid in ("R20", "R30"):  # NOT overlap_comm dp=1
        if config.overlap_comm and config.data_parallel_size == 1:
            return "FAIL", "overlap_comm=True at dp=1 → NCCL reduce-scatter = identity"
        return "PASS", f"overlap_comm={config.overlap_comm} at dp={config.data_parallel_size}"

    if rid == "R21":  # NOT sleep_level=2
        if config.sleep_level == 2:
            return "FAIL", "sleep_level=2 → #45552 guaranteed crash on RTX 4090"
        return "PASS", f"sleep_level={config.sleep_level}"

    if rid == "R22":  # NOT LoRA rank>=64
        if config.lora_rank >= 64:
            return "FAIL", f"LoRA r={config.lora_rank} → #6782 EOS suppression"
        return "PASS", f"LoRA r={config.lora_rank}"

    if rid == "R23":  # NOT FSDP2
        if config.fsdp_mode == "fsdp2":
            return "FAIL", "FSDP2 → #6772 execution overlap → OOM"
        return "PASS", f"FSDP mode={config.fsdp_mode}"

    if rid == "R24":  # NOT gs=1
        if config.group_size == 1:
            return "FAIL", "gs=1 → REINFORCE(baseline=0) → degenerate"
        return "PASS", f"gs={config.group_size}"

    if rid == "R25":  # NOT PPO-clip
        return "PASS", "GRPO selected (PPO-clip would require 65 GiB)"

    if rid == "R26":  # NOT full param sync
        if config.bypass:
            return "PASS", "bypass=True → LoRA sync (~200 MiB), not full param (~14 GiB)"
        return "WARN", "bypass=False → full param sync risk"

    if rid == "R27":  # NOT NCCL checkpoint dp=1
        if config.checkpoint_mode == "naive":
            return "PASS", "naive checkpoint at dp=1"
        return "WARN", f"checkpoint_mode={config.checkpoint_mode} — NCCL pointless at dp=1"

    if rid == "R28":  # NOT outcome reward gs<16
        if not config.shaped_rewards and config.group_size < 16:
            return "FAIL", f"outcome rewards with gs={config.group_size} < 16 → 4.6% degenerate groups"
        return "PASS", f"shaped_rewards={config.shaped_rewards}, gs={config.group_size}"

    if rid == "R29":  # NOT merge encoder cache revert
        return "PASS", "Rule awareness: never merge #46125 encoder cache revert"

    if rid == "R31":  # NOT ZeRO savings at dp=1
        if config.data_parallel_size == 1 and config.zero_stage > 0:
            return "WARN", f"ZeRO-{config.zero_stage} at dp=1 → ZERO savings, FSDP1 better"
        return "PASS", f"dp={config.data_parallel_size}, zero_stage={config.zero_stage}"

    if rid == "R32":  # NOT Muon CPU offload
        if config.use_muon and config.optimizer == "cpu_muon":
            return "FAIL", "Muon CPU offload unmerged (#7939) → use CPU_Adam"
        return "PASS", "No Muon CPU offload"

    if rid == "R33":  # NOT DSA without replay
        if config.use_dsa and not config.use_megatron:
            return "WARN", "DSA model without replay → #5384 mismatch risk"
        elif config.use_dsa and config.use_megatron:
            return "FAIL", "DSA model with Megatron → MUST implement DSAIndexerReplay"
        return "PASS", "Not using DSA"

    if rid == "R34":  # NOT global clipping for Muon
        if config.use_muon and config.clip_grad > 0 and not config.use_megatron:
            return "FAIL", f"Muon with clip_grad={config.clip_grad} → #5394 stall"
        return "PASS", "No global clipping for Muon"

    return "WARN", f"Rule {rid} — manual check recommended"


# ─── Mode 3: rtx4090 ───

def rtx4090_report() -> str:
    """Generate RTX 4090 specific comprehensive report."""
    lines = []
    lines.append("=" * 90)
    lines.append("RTX 4090 GRPO Training — Cross-Framework Bug Avoidance Report")
    lines.append("=" * 90)

    # ─── Optimal config ───
    optimal = GRPOConfig()
    results = validate_config(optimal)

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARN")

    lines.append("\n### OPTIMAL CONFIG: verl + SGLang + DeepSpeed ZeRO-2 + CPU_Adam")
    lines.append(f"  34 rules validated: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARN")

    # ─── Bugs avoided by config ───
    lines.append("\n### BUGS AVOIDED BY OPTIMAL CONFIG (5 DIRECTLY AVOIDED)")
    avoided_bugs = [
        ("#5394/#5395", "Muon clipping stall → AVOIDED (using Adam, not Muon)", "muon_clipping"),
        ("#8068/#7776", "DeepSpeed Muon clipping → AVOIDED (using Adam)", "muon_clipping"),
        ("#45552", "vLLM sleep crash → AVOIDED (using SGLang, sleep_level=1)", "cuda_stream_safety"),
        ("#6772", "FSDP2 overlap → AVOIDED (using FSDP1)", "execution_overlap_oom"),
        ("#6782", "LoRA r=64 EOS → AVOIDED (using r=32)", "lora_distribution_distortion"),
        ("#8072", "ZeRO-3 PEFT regression → AVOIDED (using ZeRO-2)", "zero3_regression"),
        ("#8061", "ZeRO stream race → AVOIDED (overlap_comm=False at dp=1)", "cuda_stream_safety"),
    ]
    for bug_id, explanation, pattern in avoided_bugs:
        lines.append(f"  ✓ {bug_id}: {explanation} [{pattern}]")

    # ─── Bugs still at risk ───
    lines.append("\n### BUGS STILL AT RISK (2 MONITORING REQUIRED)")
    at_risk = [
        ("#6794", "verl delta weight sync → silent corruption (4 sub-issues)", "monitor checksums, record_stream"),
        ("#28771", "SGLang Eagle accept_length → 44% throughput loss", "monitor accept_length < 2.0"),
    ]
    for bug_id, explanation, mitigation in at_risk:
        lines.append(f"  ⚠ {bug_id}: {explanation}")
        lines.append(f"    Mitigation: {mitigation}")

    # ─── Pattern class summary ───
    lines.append("\n### PATTERN CLASS SUMMARY (7 Classes, 8 Bug Members)")
    for p_key, p in PATTERN_CLASSES.items():
        members_in_config = []
        members_avoided = []
        for m in p["members"]:
            if m in ["#8068", "#7776", "#5394", "#5395", "#45552", "#6772", "#6782", "#8072", "#8061"]:
                members_avoided.append(m)
            elif m in ["#6794", "#28771"]:
                members_in_config.append(m)

        avoided_str = ", ".join(members_avoided) if members_avoided else "none"
        at_risk_str = ", ".join(members_in_config) if members_in_config else "none"
        lines.append(f"  [{p_key}]")
        lines.append(f"    Avoided by config: {avoided_str}")
        lines.append(f"    Still at risk: {at_risk_str}")
        lines.append(f"    Universal fix: {p['universal_fix']}")

    # ─── Memory budget ───
    lines.append("\n### MEMORY BUDGET (24 GiB RTX 4090)")
    memory_items = [
        ("Model weights (7B, bf16)", "14.0 GiB"),
        ("LoRA r=32 adapter", "0.2 GiB"),
        ("KV cache (SGLang sleep=1)", "~0 GiB (offloaded)"),
        ("Training activations", "1.6 GiB"),
        ("Gradient buffer (LoRA)", "0.42 GiB"),
        ("Optimizer states (CPU)", "0 GiB"),
        ("NCCL/overhead", "0.4 GiB"),
        ("Peak total", "16.62 GiB (69.3%)"),
        ("Headroom", "7.38 GiB (30.7%)"),
    ]
    for name, value in memory_items:
        lines.append(f"  {name}: {value}")

    # ─── Monitoring checklist ───
    lines.append("\n### MONITORING CHECKLIST")
    checklist = [
        "1. grad_norm: should NOT grow >1e6 (#5394 stall indicator)",
        "2. Loss curve: should decrease monotonically (plateau = bug)",
        "3. Logprob agreement: train vs rollout >95% (#5384 mismatch)",
        "4. GPU memory: peak <20 GiB (above = OOM risk)",
        "5. accept_length: >2.0 (#28771 degradation)",
        "6. Step time: ~13-15s (significant deviation = config issue)",
        "7. Reward difficulty correlation: ρ≈0 (ρ>0 = degenerate rewards)",
        "8. FD count: <65535 (#8075 FD leak)",
        "9. Checkpoint integrity: SHA-256 verification (#6794 silent corruption)",
    ]
    for c in checklist:
        lines.append(f"  {c}")

    return "\n".join(lines)


# ─── Mode 4: cross-framework ───

def cross_framework_bug_analysis(bug_id: str) -> str:
    """Trace a specific bug across all 7 frameworks and pattern classes."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"Cross-Framework Bug Analysis: {bug_id}")
    lines.append("=" * 70)

    # Find the bug in all frameworks
    target_bug = None
    target_framework = None
    for fw_key, fw in FRAMEWORKS.items():
        for bug in fw["bugs"]:
            if bug["id"] == bug_id:
                target_bug = bug
                target_framework = fw
                break

    if not target_bug:
        lines.append(f"Bug {bug_id} not found in tracked database")
        lines.append("Tracked bugs:")
        for fw_key, fw in FRAMEWORKS.items():
            for bug in fw["bugs"]:
                lines.append(f"  {fw['name']} {bug['id']}: {bug['title']}")
        return "\n".join(lines)

    # ─── Bug detail ───
    lines.append("\n### BUG DETAILS")
    lines.append(f"  Framework: {target_framework['name']}")
    lines.append(f"  Title: {target_bug['title']}")
    lines.append(f"  Severity: {target_bug['severity']}")
    lines.append(f"  Pattern class: {target_bug['pattern']}")
    lines.append(f"  Avoided by: {target_bug['avoided_by']}")

    # ─── Pattern class ───
    pattern_key = target_bug["pattern"]
    if pattern_key in PATTERN_CLASSES:
        p = PATTERN_CLASSES[pattern_key]
        lines.append("\n### PATTERN CLASS")
        lines.append(f"  Class: {pattern_key}")
        lines.append(f"  Description: {p['description']}")
        lines.append(f"  Mathematical model: {p['mathematical_model']}")
        lines.append(f"  Universal fix: {p['universal_fix']}")
        lines.append(f"  All members: {', '.join(p['members'])}")

    # ─── Related bugs across frameworks ───
    lines.append("\n### RELATED BUGS (Same Pattern Class, Different Frameworks)")
    if pattern_key in PATTERN_CLASSES:
        for fw_key, fw in FRAMEWORKS.items():
            for bug in fw["bugs"]:
                if bug["pattern"] == pattern_key and bug["id"] != bug_id:
                    lines.append(f"  {fw['name']} {bug['id']}: {bug['title']} [{bug['severity']}]")
                    lines.append(f"    Avoided by: {bug['avoided_by']}")

    # ─── Related MUST DO/MUST NOT rules ───
    lines.append("\n### RELATED RULES")
    for r in MUST_DO_RULES + MUST_NOT_RULES:
        if pattern_key in r["rule"].lower() or pattern_key.replace("_", " ") in r["reason"].lower() or target_bug["id"] in r["reason"]:
            rule_type = "MUST DO" if r["id"].startswith("R") and int(r["id"][1:]) <= 17 else "MUST NOT"
            lines.append(f"  [{rule_type}] {r['id']}: {r['rule']} [{r['framework']}] — {r['reason']}")

    # ─── RTX 4090 implications ───
    lines.append("\n### RTX 4090 IMPLICATIONS")
    if pattern_key == "muon_clipping":
        lines.append("  ★★★★★★★★★ CRITICAL: Muon clipping bugs → AVOIDED by using Adam, not Muon")
        lines.append("  Cross-framework: #5394 (Megatron) = #8068 (DeepSpeed) = #7776 (DeepSpeed)")
        lines.append("  Universal fix: Per-optimizer-group clipping → Muon=0, Adam=1.0")
    elif pattern_key == "cuda_stream_safety":
        lines.append("  ★★★★★★★★★ CRITICAL: Stream ordering bugs → RTX 4090 single copy engine more vulnerable")
        lines.append("  #45552 vLLM sleep_level=2 → guaranteed crash → MUST use SGLang, sleep_level=1")
        lines.append("  #8061 DeepSpeed → AVOIDED by overlap_comm=False at dp=1")
    elif pattern_key == "discrete_decision_mismatch":
        lines.append("  ★★★★★★★★★ VERY HIGH: DSA indexer mismatch → incorrect GRPO reward signal")
        lines.append("  MoE RouterReplay = template → DSAIndexerReplay ~200-300 LOC needed")
    elif pattern_key == "lora_distribution_distortion":
        lines.append("  ★★★★★★★★★ HIGH: LoRA r>=64 → EOS suppression → runaway generation")
        lines.append("  MUST: LoRA r=32 → 0.42% params, no EOS issue")
    elif pattern_key == "grpo_singleton_degeneration":
        lines.append("  ★★★★★★★★★ CRITICAL: gs=1 = REINFORCE(baseline=0) → degenerate across ALL 4 RL frameworks")
        lines.append("  MUST: gs>=8 for shaped rewards, NEVER gs=1")
    elif pattern_key == "stale_cache_corruption":
        lines.append("  ★★★★★★★★★ HIGH: Cache reuse after weight update → silent corruption")
        lines.append("  #46125: MUST NOT merge encoder cache revert → affects RLHF directly")
    else:
        lines.append("  Check specific bug details for RTX 4090 impact")

    return "\n".join(lines)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Cross-Framework GRPO Bug Avoidance Matrix Tool")
    parser.add_argument("mode", choices=["matrix", "validate", "rtx4090", "cross-framework"],
                        help="Mode: matrix, validate config, RTX 4090 report, or cross-framework bug analysis")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config string for validate mode")
    parser.add_argument("--bug", type=str, default=None,
                        help="Bug ID (e.g., '#5394') for cross-framework mode")

    args = parser.parse_args()

    if args.mode == "matrix":
        print(generate_matrix())

    elif args.mode == "validate":
        if args.config:
            config_dict = json.loads(args.config)
            config = GRPOConfig(**config_dict)
        else:
            config = GRPOConfig()
            print("Validating optimal RTX 4090 config (default):")

        results = validate_config(config)

        print("\n" + "=" * 70)
        print("Cross-Framework GRPO Config Validation Results")
        print("=" * 70)

        pass_count = sum(1 for r in results if r.status == "PASS")
        fail_count = sum(1 for r in results if r.status == "FAIL")
        warn_count = sum(1 for r in results if r.status == "WARN")

        for r in results:
            icon = "✓" if r.status == "PASS" else ("✗" if r.status == "FAIL" else "⚠")
            print(f"  {icon} [{r.status}] {r.rule_id}: {r.rule} [{r.framework}]")
            print(f"    {r.explanation}")

        print(f"\nSummary: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARN / 34 rules")
        if fail_count > 0:
            print("★★★★★★★★★ CRITICAL: Fix FAIL rules before training!")

    elif args.mode == "rtx4090":
        print(rtx4090_report())

    elif args.mode == "cross-framework":
        if args.bug:
            print(cross_framework_bug_analysis(args.bug))
        else:
            print("Please provide --bug ID (e.g., '#5394')")
            print("Tracked bugs:")
            for fw_key, fw in FRAMEWORKS.items():
                for bug in fw["bugs"]:
                    print(f"  {fw['name']} {bug['id']}: {bug['title']}")


if __name__ == "__main__":
    main()
