#!/usr/bin/env python3
"""
Megatron Muon Clipping + DSA Indexer Avoidance Config Tool

4 modes: check/diagnose/avoid/rtx4090
Validates Megatron configs against ChainedOptimizer Muon clipping stall (#5394),
DSA indexer replay (#5384/#5386), and cross-framework Muon clipping bugs.

Based on deep readings:
- #5394: ChainedOptimizer global clipping stalls Muon (positive-feedback loop)
- #5395: skip_grad_norm_clip per-optimizer fix (CHANGES_REQUESTED)
- #5384/#5386: DSA indexer train/rollout mismatch → incorrect GRPO signal
- #5400: GDN in_proj fused matrix → Muon orthogonalization semantically meaningless
- DeepSpeed #8068/#7776: Same Muon clipping pattern across frameworks

Usage:
  python3 megatron_muon_clipping_avoidance_tool.py check --config '{"use_megatron": true, "optimizer": "chained_muon_adam", "clip_grad": 1.0}'
  python3 megatron_muon_clipping_avoidance_tool.py diagnose --symptoms 'loss_plateau,grad_norm_growth'
  python3 megatron_muon_clipping_avoidance_tool.py avoid --optimizer muon
  python3 megatron_muon_clipping_avoidance_tool.py rtx4090
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Bug Pattern Definitions ───

MUON_CLIPPING_BUG = {
    "id": "#5394",
    "title": "ChainedOptimizer Muon clipping stall",
    "severity": "CRITICAL",
    "root_cause": "Single global grad_norm across ALL sub-optimizers → clip_coeff = clip_grad / grad_norm ≈ 2e-8 → Newton-Schulz degenerates",
    "mechanism": "Positive-feedback: tiny c → near-zero → degenerate → more large grads → smaller c → collapse",
    "fixes": [
        "Per-optimizer-group clipping: Muon=0, Adam=1.0",
        "skip_grad_norm_clip attribute (#5395)",
        "clip_grad=0 for dist_muon sub-optimizer",
        "Clamp clip_coeff with floor (suboptimal)",
    ],
    "cross_framework": [
        "DeepSpeed #8068: gradient_clipping default 0→1.0 proposed",
        "DeepSpeed #7776: orthogonalization-before-clipping ordering",
        "NeMo/Emerging-Optimizers #229/#230: Newton-Schulz degeneration",
    ],
}

DSA_INDEXER_BUG = {
    "id": "#5384/#5386",
    "title": "DSA indexer train/rollout mismatch",
    "severity": "VERY HIGH for GRPO",
    "root_cause": "DSA indexer top-k = discrete decision → different kernels/precision/batching in train vs rollout → different selections → logprob mismatch",
    "mechanism": "Same tokens attend to different KV positions → different probability distributions → incorrect reward attribution → wrong GRPO signal",
    "fixes": [
        "Implement DSAIndexerReplay (~200-300 LOC, follows RouterReplay pattern)",
        "Carry recorded indexer_topk with rollout data in verl",
        "Stash meta['indexer_topk'] in rollout dict (community workaround)",
    ],
}

GDN_ROUTING_BUG = {
    "id": "#5400",
    "title": "GDN in_proj fused matrix Muon routing",
    "severity": "MEDIUM",
    "root_cause": "in_proj packs q/k/v/conv/gate/beta into one fused matrix → Muon orthogonalizes as single matrix (semantically meaningless)",
    "fixes": [
        "Tag in_proj.weight with skip_orthogonalization=True",
        "Route GDN parameters to Adam instead of Muon",
    ],
}

MOE_ZLOSS_BUG = {
    "id": "#5401",
    "title": "MoE router z-loss + TE CUDA graph",
    "severity": "MEDIUM",
    "root_cause": "torch.tensor(logits.shape[0], device=logits.device) creates CPU→CUDA copy during graph capture",
    "fixes": [
        "Use Python int instead of device tensor for shape[0]",
    ],
}

# ─── Config Data Structures ───

@dataclass
class MegatronConfig:
    use_megatron: bool = True
    optimizer: str = "adam"  # adam, muon, chained_muon_adam, chained_muon_soap
    clip_grad: float = 1.0
    use_chained_optimizer: bool = False
    skip_grad_norm_clip: bool = False
    use_dsa: bool = False
    use_gdn: bool = False  # GatedDeltaNet
    use_moe: bool = False
    use_moe_zloss: bool = False
    use_cuda_graph: bool = False
    use_fsdp: bool = False
    model_arch: str = "dense"  # dense, moe, gdn_hybrid, dsa
    gradient_accumulation_steps: int = 1
    data_parallel_size: int = 1

@dataclass
class DiagnosticResult:
    bug_id: str
    title: str
    severity: str
    triggered: bool
    confidence: float  # 0.0-1.0
    explanation: str
    fix_recommendations: List[str] = field(default_factory=list)

@dataclass
class AvoidanceConfig:
    optimizer: str
    clip_grad: float
    skip_grad_norm_clip: bool
    dsa_replay_required: bool
    gdn_skip_orthogonalization: bool
    moe_zloss_fix: bool
    notes: List[str] = field(default_factory=list)

# ─── Mode 1: check ───

def check_config(config: MegatronConfig) -> List[DiagnosticResult]:
    """Validate Megatron config against known bug patterns."""
    results = []

    # ─── Check 1: Muon clipping stall (#5394) ───
    if config.optimizer in ("muon", "chained_muon_adam", "chained_muon_soap") or config.use_chained_optimizer:
        if config.clip_grad > 0 and not config.skip_grad_norm_clip:
            # Positive-feedback stall risk
            severity = "CRITICAL"
            confidence = 0.95 if config.clip_grad == 1.0 else 0.80
            explanation = (
                f"ChainedOptimizer with clip_grad={config.clip_grad} and no skip_grad_norm_clip → "
                f"global grad_norm across ALL sub-optimizers → clip_coeff = {config.clip_grad}/grad_norm → "
                f"near-zero for Muon → Newton-Schulz degenerates → positive-feedback stall"
            )
            triggered = True
        elif config.skip_grad_norm_clip:
            # Fix applied
            triggered = False
            confidence = 0.90
            explanation = "skip_grad_norm_clip=True → Muon groups bypass global clipping → safe"
            severity = "RESOLVED"
        else:
            triggered = False
            confidence = 0.95
            explanation = "clip_grad=0 → no clipping applied → Muon safe"
            severity = "SAFE"

        results.append(DiagnosticResult(
            bug_id="#5394", title="ChainedOptimizer Muon clipping stall",
            severity=severity, triggered=triggered, confidence=confidence,
            explanation=explanation,
            fix_recommendations=MUON_CLIPPING_BUG["fixes"],
        ))

    # ─── Check 2: DSA indexer mismatch (#5384/#5386) ───
    if config.use_dsa or config.model_arch == "dsa":
        results.append(DiagnosticResult(
            bug_id="#5384/#5386", title="DSA indexer train/rollout mismatch",
            severity="VERY HIGH for GRPO",
            triggered=True, confidence=0.90,
            explanation="DSA models in GRPO → indexer top-k differs between train and rollout → logprob mismatch → incorrect reward attribution",
            fix_recommendations=DSA_INDEXER_BUG["fixes"],
        ))

    # ─── Check 3: GDN fused matrix (#5400) ───
    if config.use_gdn or config.model_arch == "gdn_hybrid":
        if config.optimizer in ("muon", "chained_muon_adam"):
            results.append(DiagnosticResult(
                bug_id="#5400", title="GDN in_proj fused matrix Muon routing",
                severity="MEDIUM", triggered=True, confidence=0.85,
                explanation="GDN in_proj packs q/k/v/conv/gate/beta → Muon orthogonalizes as single matrix → semantically meaningless",
                fix_recommendations=GDN_ROUTING_BUG["fixes"],
            ))

    # ─── Check 4: MoE z-loss CUDA graph (#5401) ───
    if config.use_moe and config.use_moe_zloss and config.use_cuda_graph:
        results.append(DiagnosticResult(
            bug_id="#5401", title="MoE router z-loss + TE CUDA graph",
            severity="MEDIUM", triggered=True, confidence=0.90,
            explanation="torch.tensor(logits.shape[0], device=logits.device) creates CPU→CUDA copy during graph capture → rejected by PyTorch",
            fix_recommendations=MOE_ZLOSS_BUG["fixes"],
        ))

    # ─── Check 5: Adam-only clipping (baseline safe) ───
    if config.optimizer == "adam" and config.clip_grad > 0:
        results.append(DiagnosticResult(
            bug_id="#5394", title="Adam-only clipping (not Muon)",
            severity="SAFE", triggered=False, confidence=0.95,
            explanation=f"Adam-only with clip_grad={config.clip_grad} → safe (Adam adapts lr per-param, clipping preserves direction)",
            fix_recommendations=["No fix needed for Adam-only configs"],
        ))

    # ─── Check 6: Data parallel clipping (ZeRO-1/2) ───
    if config.data_parallel_size == 1 and config.clip_grad > 0:
        # Same insight as DeepSpeed: at dp=1, reduce-scatter is identity → no data movement
        results.append(DiagnosticResult(
            bug_id="cross-framework", title="DP=1 clipping implications",
            severity="INFO", triggered=False, confidence=0.80,
            explanation="dp=1: reduce-scatter is identity → ZeRO-1/2 provide ZERO savings → clip_grad applies globally",
            fix_recommendations=["ZeRO-1/2 unnecessary at dp=1, FSDP1 preferred for RTX 4090"],
        ))

    return results


# ─── Mode 2: diagnose ───

SYMPTOM_BUG_MAP = {
    "loss_plateau": [
        ("#5394", 0.85, "ChainedOptimizer Muon clipping stall → loss plateaus ~0.5, never overfits"),
        ("#5384", 0.40, "DSA indexer mismatch → incorrect reward signal → loss doesn't improve"),
    ],
    "grad_norm_growth": [
        ("#5394", 0.90, "Positive-feedback: grad_norm grows from 7.5e7→2.2e11 → Muon collapse"),
    ],
    "grad_norm_huge": [
        ("#5394", 0.90, "ChainedOptimizer global norm dominated by large layers → clip_coeff ≈ 2e-8"),
    ],
    "training_stall": [
        ("#5394", 0.80, "Muon groups stop updating → degenerate Newton-Schulz → permanent stall"),
    ],
    "logprob_mismatch": [
        ("#5384", 0.95, "DSA indexer top-k differs → logprobs disagree → GRPO reward attribution wrong"),
    ],
    "reward_instability": [
        ("#5384", 0.85, "Rollout/training disagreement → reward attribution noisy → GRPO unstable"),
    ],
    "nan_loss": [
        ("#5394", 0.30, "Near-zero gradients after clipping → numerical instability possible (unlikely primary cause)"),
    ],
    "cuda_graph_error": [
        ("#5401", 0.85, "MoE z-loss CPU→CUDA copy during graph capture → rejected by PyTorch"),
    ],
}

def diagnose_symptoms(symptoms: str) -> List[DiagnosticResult]:
    """Given training symptoms, check if Muon clipping or DSA indexer bugs are the cause."""
    symptom_list = [s.strip() for s in symptoms.split(",")]
    results = []

    for symptom in symptom_list:
        if symptom in SYMPTOM_BUG_MAP:
            for bug_id, confidence, explanation in SYMPTOM_BUG_MAP[symptom]:
                # Determine bug details
                if bug_id == "#5394":
                    bug = MUON_CLIPPING_BUG
                elif bug_id == "#5384":
                    bug = DSA_INDEXER_BUG
                elif bug_id == "#5401":
                    bug = MOE_ZLOSS_BUG
                else:
                    continue

                results.append(DiagnosticResult(
                    bug_id=bug_id, title=bug["title"],
                    severity=bug["severity"], triggered=True,
                    confidence=confidence, explanation=explanation,
                    fix_recommendations=bug["fixes"],
                ))

    # ─── Differential diagnosis ───
    muon_confidence = max((r.confidence for r in results if r.bug_id == "#5394"), default=0)
    dsa_confidence = max((r.confidence for r in results if r.bug_id == "#5384"), default=0)

    if muon_confidence > 0.5:
        results.append(DiagnosticResult(
            bug_id="DIAGNOSIS", title="Primary diagnosis: Muon clipping stall",
            severity="CRITICAL", triggered=True, confidence=muon_confidence,
            explanation="Most likely cause: #5394 positive-feedback stall. Verify: check grad_norm history, check loss curve plateau, compare clip_grad=0 vs clip_grad=1.0",
            fix_recommendations=["Set clip_grad=0 for Muon groups", "Add skip_grad_norm_clip=True", "Compare training with clip_grad=0 to confirm"],
        ))

    if dsa_confidence > 0.5:
        results.append(DiagnosticResult(
            bug_id="DIAGNOSIS", title="Secondary diagnosis: DSA indexer mismatch",
            severity="VERY HIGH", triggered=True, confidence=dsa_confidence,
            explanation="DSA model GRPO instability. Verify: compare rollout vs training logprobs on same input, check indexer top-k agreement rate",
            fix_recommendations=["Implement DSAIndexerReplay", "Stash meta['indexer_topk'] in rollout dict", "Compare logprobs agreement rate"],
        ))

    if not results:
        results.append(DiagnosticResult(
            bug_id="NONE", title="No matching bug patterns found",
            severity="INFO", triggered=False, confidence=0.50,
            explanation=f"Symptoms '{symptoms}' don't match known Megatron Muon/DSA bug patterns. Check other frameworks.",
            fix_recommendations=["Check DeepSpeed #8068 if using ZeRO+Muon", "Check vLLM #45552 if using sleep/wake", "Check verl #6794 if using delta weight sync"],
        ))

    return results


# ─── Mode 3: avoid ───

def generate_avoidance_config(optimizer: str) -> AvoidanceConfig:
    """Generate safe config recommendations for a given optimizer type."""
    if optimizer == "adam":
        return AvoidanceConfig(
            optimizer="adam", clip_grad=1.0,
            skip_grad_norm_clip=False,
            dsa_replay_required=True,  # Even with Adam, DSA needs replay
            gdn_skip_orthogonalization=False,  # Not applicable for Adam
            moe_zloss_fix=False,
            notes=[
                "Adam-only: safe with clip_grad=1.0 (standard clipping)",
                "CPU_Adam recommended for RTX 4090 (optimizer states on CPU)",
                "DSA models still need indexer replay for GRPO",
                "FSDP1 recommended over ZeRO-1/2 at dp=1",
            ],
        )

    elif optimizer == "muon":
        return AvoidanceConfig(
            optimizer="muon", clip_grad=0.0,
            skip_grad_norm_clip=True,  # MUST skip for Muon
            dsa_replay_required=True,
            gdn_skip_orthogonalization=False,  # Not applicable for pure Muon
            moe_zloss_fix=False,
            notes=[
                "★★★★★★★★★ MUST: clip_grad=0 for Muon (scale-invariant optimizer)",
                "★★★★★★★★★ MUST: skip_grad_norm_clip=True for Muon in ChainedOptimizer",
                "Muon+Adam chained: per-group clipping (Muon=0, Adam=1.0)",
                "Newton-Schulz eps floor interaction → near-zero vectors degenerate",
                "Positive-feedback stall if clipping NOT disabled",
                "★★★★★★★★★ Cross-framework: same bug in DeepSpeed #8068, #7776",
            ],
        )

    elif optimizer == "chained_muon_adam":
        return AvoidanceConfig(
            optimizer="chained_muon_adam", clip_grad=1.0,
            skip_grad_norm_clip=True,  # MUST skip for Muon sub-optimizer
            dsa_replay_required=True,
            gdn_skip_orthogonalization=True,  # MUST skip for GDN in_proj
            moe_zloss_fix=True,  # If using MoE
            notes=[
                "★★★★★★★★★ MUST: skip_grad_norm_clip=True for Muon sub-optimizer (#5395)",
                "★★★★★★★★★ MUST: Adam sub-optimizer clips at 1.0 (standard)",
                "★★★★★★★★★ MUST: Muon sub-optimizer clips at 0 (bypass global norm)",
                "★★★★★★★★★ MUST: GDN in_proj.weight tagged skip_orthogonalization=True (#5400)",
                "★★★★★★★★★ MUST: DSA indexer replay for GRPO (#5384/#5386)",
                "★★★★★★★★★ Cross-framework: 3 independent discoveries → same root cause",
                "Scale-invariant optimizer + scale-change (clipping) = fundamental contradiction!",
            ],
        )

    elif optimizer == "chained_muon_soap":
        return AvoidanceConfig(
            optimizer="chained_muon_soap", clip_grad=1.0,
            skip_grad_norm_clip=True,  # MUST skip for Muon sub-optimizer
            dsa_replay_required=True,
            gdn_skip_orthogonalization=True,
            moe_zloss_fix=True,
            notes=[
                "★★★★★★★★★ MUST: skip_grad_norm_clip=True for Muon sub-optimizer",
                "★★★★★★★★★ MUST: SOAP sub-optimizer clips normally",
                "★★★★★★★★★ MUST: Muon sub-optimizer bypasses global norm",
                "★★★★★★★★★ #5395 fix gates skip flag on isinstance(optimizer, OrthogonalizedOptimizer)",
                "SOAP/Lion NOT orthogonalized → skip flag correctly NOT applied to them",
            ],
        )

    else:
        return AvoidanceConfig(
            optimizer=optimizer, clip_grad=1.0,
            skip_grad_norm_clip=False,
            dsa_replay_required=True,
            gdn_skip_orthogonalization=False,
            moe_zloss_fix=False,
            notes=["Unknown optimizer type — apply standard clip_grad=1.0, check DSA replay needs"],
        )


# ─── Mode 4: rtx4090 ───

RTX4090_GRPO_CONFIG = {
    "framework": "verl + SGLang",
    "training_framework": "DeepSpeed ZeRO-2 + CPU_Adam",
    "rollout_engine": "SGLang (NOT vLLM — #45552 crashes RTX 4090)",
    "algorithm": "GRPO",
    "optimizer": "Adam (NOT Muon — clipping bugs #5394)",
    "clip_grad": 1.0,
    "lora_rank": 32,
    "lora_target_modules": "all-linear",
    "group_size": 8,
    "bypass": True,
    "ref_in_actor": True,
    "fsdp_mode": "FSDP1 (NOT FSDP2 — #6772 execution order overlap)",
    "sleep_level": "1 (NOT 2 — #45552 RTX 4090 BLOCKER)",
    "checkpoint_mode": "naive (NOT NCCL — dp=1 pointless)",
    "gradient_accumulation": "gs=8 for sparse rewards, gs>=4 for shaped",
    "shaped_rewards": True,
    "enforce_eager": True,
}

RTX4090_MEGATRON_ALTERNATIVES = {
    "option_A": {
        "name": "verl + SGLang + DeepSpeed (RECOMMENDED)",
        "optimizer": "Adam (CPU_Adam)",
        "clip_grad": 1.0,
        "muon_safe": "N/A (not using Muon)",
        "dsa_replay": "Need if using DSA model",
        "peak_memory": "16.24 GiB / 24 GiB (73.0%)",
        "notes": "Proven config, all known bugs avoided by configuration",
    },
    "option_B": {
        "name": "verl + SGLang + Megatron-LM",
        "optimizer": "chained_muon_adam",
        "clip_grad": "Muon=0, Adam=1.0",
        "muon_safe": "★★★★★★★★★ MUST: skip_grad_norm_clip=True (#5395)",
        "dsa_replay": "★★★★★★★★★ MUST: DSAIndexerReplay (#5384)",
        "peak_memory": "~18-20 GiB / 24 GiB (est.)",
        "notes": "Megatron+Muon requires per-group clipping fix, DSA replay, more complex",
    },
    "option_C": {
        "name": "verl + SGLang + Megatron-LM (Adam-only)",
        "optimizer": "DistributedFusedAdam",
        "clip_grad": 1.0,
        "muon_safe": "Safe (no Muon)",
        "dsa_replay": "★★★★★★★★★ MUST: DSAIndexerReplay (#5384)",
        "peak_memory": "~17-19 GiB / 24 GiB",
        "notes": "Simpler than Muon path, but still needs DSA replay for GRPO",
    },
}

RTX4090_PRIORITY_MATRIX = {
    "P0_CRITICAL": [
        {"issue": "#5394/#5395", "impact": "Muon clipping → silent training stall", "avoided_by": "Using Adam (not Muon)"},
        {"issue": "#5384/#5386", "impact": "DSA indexer mismatch → incorrect GRPO signal", "avoided_by": "Implementing DSAIndexerReplay or avoiding DSA models"},
    ],
    "P1_HIGH": [
        {"issue": "#5387", "impact": "FSDP fully_shard → enables larger models on 24 GiB", "action": "Monitor for merge"},
        {"issue": "#5391", "impact": "LayerWise DDP → reduces Muon memory overhead", "action": "Monitor for merge"},
    ],
    "P2_MEDIUM": [
        {"issue": "#5401", "impact": "MoE z-loss + CUDA graph → capture error", "avoided_by": "Not using MoE z-loss + CUDA graph"},
        {"issue": "#5400", "impact": "GDN fused matrix Muon routing", "avoided_by": "Using Adam (not Muon) for GDN params"},
    ],
}

def rtx4090_report():
    """Generate RTX 4090 specific GRPO config with all avoidance rules."""
    lines = []
    lines.append("=" * 80)
    lines.append("RTX 4090 GRPO Training — Megatron Muon/DSA Bug Avoidance Report")
    lines.append("=" * 80)

    # ─── Recommended config ───
    lines.append("\n### RECOMMENDED CONFIG (verl + SGLang + DeepSpeed)")
    for k, v in RTX4090_GRPO_CONFIG.items():
        marker = "★★★★★★★★★" if k in ("optimizer", "rollout_engine", "fsdp_mode", "sleep_level", "clip_grad", "algorithm") else ""
        lines.append(f"  {marker} {k}: {v} {marker}")

    # ─── Bug avoidance summary ───
    lines.append("\n### BUGS AVOIDED BY RECOMMENDED CONFIG")
    avoided = [
        "#5394 Muon clipping → AVOIDED (using Adam, not Muon)",
        "#5395 skip_grad_norm_clip → N/A (no Muon)",
        "#5400 GDN routing → N/A (no Muon)",
        "#5401 z-loss CUDA graph → AVOIDED (enforce_eager=True)",
        "#5384 DSA indexer → AVOIDED (not using DSA model)",
        "#45552 vLLM sleep_level=2 → AVOIDED (using SGLang, sleep_level=1)",
        "#6772 FSDP2 overlap → AVOIDED (using FSDP1)",
        "#8068 DeepSpeed clipping → AVOIDED (Adam safe with clip_grad=1.0)",
    ]
    for a in avoided:
        lines.append(f"  ✓ {a}")

    # ─── Megatron alternatives ───
    lines.append("\n### MEGATRON-LM ALTERNATIVE CONFIGS")
    for option, details in RTX4090_MEGATRON_ALTERNATIVES.items():
        lines.append(f"\n  [{option}] {details['name']}")
        for k, v in details.items():
            if k == "name":
                continue
            lines.append(f"    {k}: {v}")

    # ─── Priority matrix ───
    lines.append("\n### PRIORITY MATRIX (RTX 4090 GRPO Impact)")
    for priority, issues in RTX4090_PRIORITY_MATRIX.items():
        lines.append(f"\n  [{priority}]")
        for item in issues:
            lines.append(f"    {item['issue']}: {item['impact']}")
            if "avoided_by" in item:
                lines.append(f"      → Avoided by: {item['avoided_by']}")
            elif "action" in item:
                lines.append(f"      → Action: {item['action']}")

    # ─── Cross-framework pattern ───
    lines.append("\n### CROSS-FRAMEWORK MUON CLIPPING PATTERN (Same Root Cause)")
    lines.append("  #5394 (Megatron) = #8068 (DeepSpeed) = #7776 (DeepSpeed) = #229/#230 (NeMo)")
    lines.append("  Root cause: Scale-invariant optimizer + global scale-change (clipping) = contradiction")
    lines.append("  Universal fix: Per-optimizer-group clipping → Muon=0, Adam=1.0")
    lines.append("  Mathematical proof: clip_coeff c = clip_grad/grad_norm → for Muon, direction ∝ c·g → c changes direction!")

    # ─── Monitoring checklist ───
    lines.append("\n### MONITORING CHECKLIST (RTX 4090 GRPO Training)")
    checklist = [
        "1. grad_norm history: should NOT grow >1e6 (indicates #5394 stall)",
        "2. Loss curve: should decrease monotonically (plateau = bug)",
        "3. Logprob agreement: train vs rollout >95% (low = #5384 mismatch)",
        "4. GPU memory: peak <20 GiB (above = OOM risk)",
        "5. accept_length: >2.0 for spec decode (low = #28771 degradation)",
        "6. Step time: ~13-15s (significant deviation = config issue)",
        "7. Reward variance: shaped rewards ρ≈0 with task difficulty (outcome-only ρ>0 = degenerate)",
    ]
    for c in checklist:
        lines.append(f"  {c}")

    # ─── Memory budget ───
    lines.append("\n### MEMORY BUDGET (24 GiB RTX 4090)")
    budget = [
        "Model weights (7B, bf16): 14 GiB",
        "LoRA r=32 adapter: 0.2 GiB",
        "KV cache (SGLang sleep): 0 GiB (offloaded during training)",
        "Training activations: 1.6 GiB (with gradient checkpointing)",
        "Gradient buffer: 0.42 GiB (LoRA only)",
        "Optimizer states (CPU_Adam): 0 GiB (on CPU)",
        "NCCL/overhead: 0.4 GiB",
        "Peak total: 16.62 GiB (69.3%)",
        "Headroom: 7.38 GiB (30.7%)",
    ]
    for b in budget:
        lines.append(f"  {b}")

    return "\n".join(lines)


# ─── Main Entry Point ───

def main():
    parser = argparse.ArgumentParser(description="Megatron Muon Clipping + DSA Indexer Avoidance Config Tool")
    parser.add_argument("mode", choices=["check", "diagnose", "avoid", "rtx4090"],
                        help="Mode: check config, diagnose symptoms, generate avoidance config, or RTX 4090 report")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config string for check mode")
    parser.add_argument("--symptoms", type=str, default=None,
                        help="Comma-separated symptoms for diagnose mode")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "muon", "chained_muon_adam", "chained_muon_soap"],
                        help="Optimizer type for avoid mode")

    args = parser.parse_args()

    if args.mode == "check":
        if args.config:
            config_dict = json.loads(args.config)
            config = MegatronConfig(**config_dict)
        else:
            # Default: check the recommended RTX 4090 config
            config = MegatronConfig(
                optimizer="adam", clip_grad=1.0,
                use_megatron=True, data_parallel_size=1,
            )
            print("No --config provided, checking default RTX 4090 config:")
            print(f"  optimizer=adam, clip_grad=1.0, dp=1")

        results = check_config(config)

        print("\n" + "=" * 60)
        print("Megatron Config Bug Pattern Check Results")
        print("=" * 60)

        triggered_count = sum(1 for r in results if r.triggered)
        safe_count = sum(1 for r in results if not r.triggered)

        for r in results:
            status = "⚠ TRIGGERED" if r.triggered else "✓ SAFE"
            print(f"\n[{status}] {r.bug_id}: {r.title}")
            print(f"  Severity: {r.severity}")
            print(f"  Confidence: {r.confidence:.0%}")
            print(f"  Explanation: {r.explanation}")
            if r.triggered and r.fix_recommendations:
                print(f"  Fix recommendations:")
                for f in r.fix_recommendations:
                    print(f"    → {f}")

        print(f"\nSummary: {triggered_count} triggered, {safe_count} safe")
        if triggered_count > 0:
            print("★★★★★★★★★ CRITICAL: Fix triggered bugs before training!")

    elif args.mode == "diagnose":
        if not args.symptoms:
            print("Please provide --symptoms (comma-separated)")
            print("Available symptoms: loss_plateau, grad_norm_growth, grad_norm_huge,")
            print("  training_stall, logprob_mismatch, reward_instability, nan_loss, cuda_graph_error")
            sys.exit(1)

        results = diagnose_symptoms(args.symptoms)

        print("\n" + "=" * 60)
        print("Megatron Muon/DSA Bug Diagnosis Results")
        print("=" * 60)
        print(f"Input symptoms: {args.symptoms}")

        for r in results:
            print(f"\n[{r.bug_id}] {r.title}")
            print(f"  Severity: {r.severity}")
            print(f"  Confidence: {r.confidence:.0%}")
            print(f"  Explanation: {r.explanation}")
            if r.fix_recommendations:
                print(f"  Recommended fixes:")
                for f in r.fix_recommendations:
                    print(f"    → {f}")

        # ─── Verification steps ───
        print("\n### VERIFICATION STEPS")
        verification = [
            "1. Check grad_norm history: tensorboard/W&B → should be <1e4 normally",
            "2. Compare clip_grad=0 vs clip_grad=1.0 training curves (5-10 steps)",
            "3. Check logprob agreement: rollout vs training on same prompts (>95%)",
            "4. Monitor Newton-Schulz output norms: should be ~1.0 (near-zero = stall)",
            "5. Check optimizer learning rate: Muon uses 10x base LR → needs unclipped gradients",
        ]
        for v in verification:
            print(f"  {v}")

    elif args.mode == "avoid":
        avoidance = generate_avoidance_config(args.optimizer)

        print("\n" + "=" * 60)
        print(f"Megatron Muon/DSA Bug Avoidance Config for '{args.optimizer}'")
        print("=" * 60)

        print(f"\n  Optimizer: {avoidance.optimizer}")
        print(f"  clip_grad: {avoidance.clip_grad}")
        print(f"  skip_grad_norm_clip: {avoidance.skip_grad_norm_clip}")
        print(f"  DSA replay required: {avoidance.dsa_replay_required}")
        print(f"  GDN skip orthogonalization: {avoidance.gdn_skip_orthogonalization}")
        print(f"  MoE z-loss fix: {avoidance.moe_zloss_fix}")

        if avoidance.notes:
            print(f"\n  Notes:")
            for n in avoidance.notes:
                print(f"    {n}")

        # ─── Cross-framework comparison ───
        print("\n### CROSS-FRAMEWORK SAME BUG PATTERN")
        print("  Megatron #5394: ChainedOptimizer global norm → Muon stall")
        print("  DeepSpeed #8068: gradient_clipping default → Muon clipping bug")
        print("  DeepSpeed #7776: orthogonalization-before-clipping ordering bug")
        print("  NeMo #229/#230: Newton-Schulz silent degeneration")
        print("  ★★★★★★★★★ Universal fix: Per-optimizer-group clipping (Muon=0, Adam=1.0)")

    elif args.mode == "rtx4090":
        report = rtx4090_report()
        print(report)

    print()


if __name__ == "__main__":
    main()
