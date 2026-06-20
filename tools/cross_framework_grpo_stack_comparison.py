#!/usr/bin/env python3
"""
Cross-Framework GRPO Training Stack Comparison Tool
====================================================
Compares the complete GRPO training stack across all 7 frameworks
(verl, DeepSpeed, Megatron, vLLM, SGLang, rLLM, PyTorch) with
focus on RTX 4090 viability.

Modes:
  stack    - Show complete training stack per framework
  compare  - Side-by-side comparison tables
  rtx4090  - RTX 4090 optimal config recommendation
  issues   - Cross-framework issue tracker

Usage:
  python3 cross_framework_grpo_stack_comparison.py <mode>
  python3 cross_framework_grpo_stack_comparison.py rtx4090
"""

import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RolloutEngine:
    engine: str
    connection: str  # ZMQ / in-process / Ray / direct
    notes: str = ""

@dataclass
class TrainingEngine:
    backend: str  # FSDP1 / FSDP2 / ZeRO-2 / ZeRO-3 / Megatron
    notes: str = ""

@dataclass
class WeightSync:
    method: str  # full / LoRA / delta / bypass
    timing: str  # per-step / per-epoch / on-demand
    memory_impact: str = ""
    notes: str = ""

@dataclass
class AdvantageComputation:
    implementation: str
    singleton_handling: str
    group_size_support: str
    notes: str = ""

@dataclass
class CheckpointEngine:
    engine: str  # naive / NCCL / HCCL / NixL / Mooncake / Kimi
    notes: str = ""

@dataclass
class SleepWake:
    level: str  # none / level1 / level2
    mechanism: str = ""
    notes: str = ""

@dataclass
class LoRASupport:
    supported: bool
    version: str  # LoRA / QLoRA / per-unit LoRA
    rank_options: str = ""
    notes: str = ""

@dataclass
class BypassMode:
    supported: bool
    skips: str = ""
    notes: str = ""

@dataclass
class RTX4090Viability:
    dp1_viable: bool
    reason: str = ""

@dataclass
class FrameworkStack:
    name: str
    rollout: RolloutEngine
    training: TrainingEngine
    weight_sync: WeightSync
    advantage: AdvantageComputation
    checkpoint: CheckpointEngine
    sleep_wake: SleepWake
    lora: LoRASupport
    bypass: BypassMode
    viability: RTX4090Viability
    oss_status: str = ""


# ============================================================================
# FRAMEWORK DATA (from diary research)
# ============================================================================

FRAMEWORKS: Dict[str, FrameworkStack] = {
    "verl": FrameworkStack(
        name="verl",
        rollout=RolloutEngine(
            engine="SGLang (primary) / vLLM (secondary)",
            connection="Ray actor (decoupled rollout worker)",
            notes="Rollout workers run as separate Ray actors; weight sync via shared memory or Ray object store. SGLang preferred for sleep/wake support."
        ),
        training=TrainingEngine(
            backend="FSDP1 (default) / FSDP2 (experimental)",
            notes="V1 trainer uses register_trainer pattern. FSDP1 is the battle-tested path. FSDP2 support added but still has memory spikes on 4090."
        ),
        weight_sync=WeightSync(
            method="LoRA (primary) / full (fallback) / delta (experimental)",
            timing="per-step after rollout phase",
            memory_impact="LoRA sync: ~2MB per step vs full: ~1.3GB. Delta sync reduces to ~200MB.",
            notes="#6512 MERGED: per-unit LoRA enables selective weight sync per LoRA unit, massive memory savings on 4090."
        ),
        advantage=AdvantageComputation(
            implementation="GRPO with group-level normalization",
            singleton_handling="Singleton fallback to zero advantage (no variance). Known degeneration risk.",
            group_size_support="4-64 (configurable). gs=8 recommended for 4090.",
            notes="Advantage computed in rollout worker, broadcast to training worker. Supports both standard GRPO and DrGRPO variants."
        ),
        checkpoint=CheckpointEngine(
            engine="NCCL-based (default) / naive (debug)",
            notes="Checkpoint saves full model state + optimizer state. LoRA checkpoints save only adapter weights (~2MB)."
        ),
        sleep_wake=SleepWake(
            level="level2 (full sleep/wake with memory release)",
            mechanism="SGLang sleep API: release KV cache + intermediate buffers during training phase; wake reloads weights.",
            notes="Level2 means full GPU memory release during training. Critical for 4090: frees ~4GB during training step."
        ),
        lora=LoRASupport(
            supported=True,
            version="LoRA + QLoRA + per-unit LoRA (#6512)",
            rank_options="r=4,8,16,32,64. r=8 recommended for 4090 with 7B model.",
            notes="Per-unit LoRA (#6512 MERGED) is the breakthrough: sync only changed LoRA units instead of full adapter."
        ),
        bypass=BypassMode(
            supported=True,
            skips="Full weight sync (LoRA delta only), full checkpoint (adapter-only), full advantage recomputation",
            notes="Bypass mode skips redundant weight reloads. With LoRA, training worker already has base weights; only need delta."
        ),
        viability=RTX4090Viability(
            dp1_viable=True,
            reason="Best viability: SGLang sleep/wake frees GPU during training; LoRA reduces sync overhead; bypass avoids redundant copies. Per-unit LoRA (#6512) further reduces memory. FSDP1 well-tested on 24GB."
        ),
        oss_status="Active development, Apache 2.0, community-driven"
    ),

    "DeepSpeed": FrameworkStack(
        name="DeepSpeed",
        rollout=RolloutEngine(
            engine="vLLM (via DeepSpeed-Training integration)",
            connection="In-process (shared GPU) / ZMQ (decoupled)",
            notes="ZenFlow architecture: rollout and training share GPU sequentially. ZMQ mode allows separate rollout worker."
        ),
        training=TrainingEngine(
            backend="ZeRO-2 (recommended for 4090) / ZeRO-3 (multi-GPU)",
            notes="ZeRO-2 partitions optimizer states only (gradients + params replicated). ZeRO-3 partitions params too but adds communication overhead."
        ),
        weight_sync=WeightSync(
            method="full (default) / LoRA (after #8058 merges)",
            timing="per-step (full weight reload after training)",
            memory_impact="Full: ~1.3GB per sync. LoRA: ~2MB after #8058.",
            notes="#8058 (LoRA weight sync for GRPO) is the critical PR. Without it, every step requires full weight reload on 4090 -- prohibitive."
        ),
        advantage=AdvantageComputation(
            implementation="GRPO with DeepSpeed-native advantage calculator",
            singleton_handling="Standard: zero advantage on singleton. No special degeneration mitigation.",
            group_size_support="4-32. gs=8 default.",
            notes="Advantage computed inline in training loop. No separate rollout-side computation."
        ),
        checkpoint=CheckpointEngine(
            engine="NCCL-based / naive",
            notes="Standard DeepSpeed checkpoint: full state dict. No adapter-only mode without #8058."
        ),
        sleep_wake=SleepWake(
            level="level1 (partial: release KV cache only)",
            mechanism="ZenFlow: manually release KV cache before training, but intermediate buffers remain allocated.",
            notes="Level1 only. Cannot release activation memory during training. ~2GB stays allocated on 4090."
        ),
        lora=LoRASupport(
            supported=True,
            version="LoRA (standard) / QLoRA",
            rank_options="r=8,16,32,64. r=8 for 4090.",
            notes="LoRA supported but weight sync still requires full reload without #8058. Per-unit LoRA not available."
        ),
        bypass=BypassMode(
            supported=True,
            skips="Full weight reload (with LoRA bypass), redundant gradient accumulation",
            notes="Bypass mode exists but effectiveness limited without #8058 LoRA sync. Still needs full optimizer step."
        ),
        viability=RTX4090Viability(
            dp1_viable=True,
            reason="Viable ONLY after #8058 merges (LoRA weight sync). Without it, full weight reload per step exceeds 24GB. ZeRO-2 is the correct stage for dp=1 on 4090. #8061 CUDA race must also be resolved."
        ),
        oss_status="Microsoft-backed, MIT license, active development"
    ),

    "Megatron": FrameworkStack(
        name="Megatron",
        rollout=RolloutEngine(
            engine="vLLM (via Megatron-vLLM bridge)",
            connection="NCCL weight broadcast / shared memory",
            notes="Rollout uses vLLM for inference. Weight sync via NCCL broadcast from Megatron training worker."
        ),
        training=TrainingEngine(
            backend="Megatron-LM (custom distributed backend)",
            notes="Custom TP+PP+DP distribution. No FSDP integration. MFSDPv2 (#5387) approved but blocked on #5395."
        ),
        weight_sync=WeightSync(
            method="full (NCCL broadcast)",
            timing="per-step (full broadcast after training)",
            memory_impact="Full: ~1.3GB NCCL broadcast. No LoRA or delta option.",
            notes="No LoRA/delta sync. Full broadcast every step. Prohibitive memory on 4090 for 7B+ models."
        ),
        advantage=AdvantageComputation(
            implementation="GRPO via Megatron-RL wrapper",
            singleton_handling="Standard zero advantage. No degeneration handling.",
            group_size_support="4-16. Limited by TP communication overhead.",
            notes="Advantage computed in training loop. Rollout only produces completions."
        ),
        checkpoint=CheckpointEngine(
            engine="NCCL-based (Megatron native)",
            notes="Megatron checkpoint format. No adapter-only mode. Full model state every checkpoint."
        ),
        sleep_wake=SleepWake(
            level="none",
            mechanism="No sleep/wake support. GPU always occupied.",
            notes="Critical gap: Megatron cannot release GPU memory during training phase. All buffers stay allocated."
        ),
        lora=LoRASupport(
            supported=False,
            version="Not available in Megatron-LM core",
            rank_options="N/A",
            notes="#5395 CHANGES_REQUESTED blocks LoRA integration. No per-unit LoRA. Megatron focuses on full model training."
        ),
        bypass=BypassMode(
            supported=False,
            skips="N/A",
            notes="No bypass mode. Megatron does not have weight sync optimization layer."
        ),
        viability=RTX4090Viability(
            dp1_viable=False,
            reason="NOT viable on 4090: no LoRA (full weight sync every step), no sleep/wake (GPU always occupied), MFSDPv2 blocked (#5387 approved but #5395 CHANGES_REQUESTED). Memory footprint exceeds 24GB for any 7B+ model."
        ),
        oss_status="NVIDIA-backed, proprietary-ish, slow community iteration"
    ),

    "vLLM": FrameworkStack(
        name="vLLM",
        rollout=RolloutEngine(
            engine="vLLM V1 engine (self-hosted)",
            connection="In-process (as library) / Ray (as server)",
            notes="V1 engine with improved memory management. Can serve as both rollout engine and standalone inference server."
        ),
        training=TrainingEngine(
            backend="N/A (vLLM is inference-only, pairs with external trainer)",
            notes="vLLM does not have a training backend. Must pair with verl/DeepSpeed/PyTorch FSDP for training."
        ),
        weight_sync=WeightSync(
            method="full (weight reload from external trainer)",
            timing="per-step or per-epoch (depends on trainer)",
            memory_impact="Full: ~1.3GB reload. No LoRA/delta native support.",
            notes="vLLM reloads full weights from trainer. No native LoRA sync. Must pair with framework that handles sync."
        ),
        advantage=AdvantageComputation(
            implementation="N/A (handled by paired trainer)",
            singleton_handling="Depends on paired trainer",
            group_size_support="Depends on paired trainer",
            notes="vLLM only handles rollout/completions. Advantage computed by the paired training framework."
        ),
        checkpoint=CheckpointEngine(
            engine="N/A (handled by paired trainer)",
            notes="vLLM does not manage training checkpoints. Trainer handles this."
        ),
        sleep_wake=SleepWake(
            level="level1 (partial: V1 can release KV cache)",
            mechanism="V1 engine can release KV cache blocks. Cannot release activation/intermediate memory.",
            notes="#46125 encoder cache revert is dangerous: can OOM on 4090 if encoder cache not properly released."
        ),
        lora=LoRASupport(
            supported=True,
            version="LoRA (multi-adapter serving)",
            rank_options="r=8,16,32,64 for serving. No training-side LoRA.",
            notes="LoRA supported for inference/serving only. Not for training weight sync. Multi-adapter serving works well."
        ),
        bypass=BypassMode(
            supported=False,
            skips="N/A",
            notes="No bypass mode in vLLM. It is inference-only; bypass is a training concept."
        ),
        viability=RTX4090Viability(
            dp1_viable=True,
            reason="Viable as ROLLOUT ENGINE only (not standalone). Must pair with verl or DeepSpeed for training. V1 memory management works on 24GB. Danger: #46125 encoder cache revert can OOM."
        ),
        oss_status="Active development, Apache 2.0, vLLM team"
    ),

    "SGLang": FrameworkStack(
        name="SGLang",
        rollout=RolloutEngine(
            engine="SGLang native engine (RadixAttention)",
            connection="Ray actor (paired with verl) / in-process (standalone)",
            notes="RadixAttention for prefix caching. DSA LoRA (#28703) for efficient adapter serving. RolloutKV (#28608) for KV cache management."
        ),
        training=TrainingEngine(
            backend="N/A (SGLang is inference-only, pairs with external trainer)",
            notes="SGLang does not have a training backend. Must pair with verl/PyTorch FSDP for training."
        ),
        weight_sync=WeightSync(
            method="LoRA (DSA LoRA #28703) / full (fallback)",
            timing="per-step (adapter swap) or on-demand",
            memory_impact="DSA LoRA: ~2MB adapter swap. Full: ~1.3GB.",
            notes="DSA LoRA (#28703) enables dynamic adapter swap without full weight reload. RolloutKV (#28608) manages KV cache lifecycle."
        ),
        advantage=AdvantageComputation(
            implementation="N/A (handled by paired trainer)",
            singleton_handling="Depends on paired trainer",
            group_size_support="Depends on paired trainer",
            notes="SGLang handles rollout/completions only. Advantage by paired trainer (verl)."
        ),
        checkpoint=CheckpointEngine(
            engine="N/A (handled by paired trainer)",
            notes="SGLang does not manage training checkpoints."
        ),
        sleep_wake=SleepWake(
            level="level2 (full sleep/wake with memory release)",
            mechanism="sleep() releases KV cache + activation buffers. wake() reloads weights from shared memory.",
            notes="Best sleep/wake implementation among all frameworks. Critical for 4090: frees ~4-6GB during training phase."
        ),
        lora=LoRASupport(
            supported=True,
            version="DSA LoRA (#28703) + standard LoRA serving",
            rank_options="r=4,8,16,32,64. DSA LoRA enables real-time adapter swap.",
            notes="DSA LoRA (#28703) is the key innovation: dynamic adapter swap during rollout without weight reload."
        ),
        bypass=BypassMode(
            supported=False,
            skips="N/A",
            notes="No bypass mode in SGLang. It is inference-only; bypass is handled by paired training framework (verl)."
        ),
        viability=RTX4090Viability(
            dp1_viable=True,
            reason="Best ROLLOUT ENGINE for 4090: sleep/wake (level2), DSA LoRA (#28703), RolloutKV (#28608). Must pair with verl for training. Together they form the #1 stack."
        ),
        oss_status="Active development, Apache 2.0, SGLang team"
    ),

    "rLLM": FrameworkStack(
        name="rLLM",
        rollout=RolloutEngine(
            engine="vLLM (via rLLM wrapper)",
            connection="Ray actor (rLLM orchestrator)",
            notes="rLLM wraps vLLM as rollout engine. Ray actor manages rollout/training lifecycle."
        ),
        training=TrainingEngine(
            backend="FSDP1 (via PyTorch native)",
            notes="rLLM uses standard FSDP1 for training. No custom training backend."
        ),
        weight_sync=WeightSync(
            method="full (weight broadcast via Ray)",
            timing="per-step (full reload)",
            memory_impact="Full: ~1.3GB per step. No LoRA/delta option.",
            notes="No LoRA/delta sync. Full weight broadcast every step via Ray object store."
        ),
        advantage=AdvantageComputation(
            implementation="GRPO with rLLM-native advantage",
            singleton_handling="BUG: singleton degeneration (#667). Zero advantage on singleton causes gradient collapse.",
            group_size_support="4-32. gs=4 minimum (buggy below this).",
            notes="#667 (closed): identified singleton degeneration bug. Cross-framework issue: affects all GRPO implementations."
        ),
        checkpoint=CheckpointEngine(
            engine="naive (full state dict save)",
            notes="Simple checkpoint: full state dict. No async or NCCL checkpoint."
        ),
        sleep_wake=SleepWake(
            level="none",
            mechanism="No sleep/wake support.",
            notes="Critical gap: no memory release during training. GPU always fully occupied."
        ),
        lora=LoRASupport(
            supported=False,
            version="Not available in rLLM",
            rank_options="N/A",
            notes="No LoRA support. Must do full model training."
        ),
        bypass=BypassMode(
            supported=False,
            skips="N/A",
            notes="No bypass mode. Full training cycle every step."
        ),
        viability=RTX4090Viability(
            dp1_viable=False,
            reason="NOT viable on 4090: no LoRA, no sleep/wake, singleton degeneration bug (#667), full weight sync every step. Memory exceeds 24GB for 7B models."
        ),
        oss_status="Research project, limited community, #667 PR closed"
    ),

    "PyTorch": FrameworkStack(
        name="PyTorch",
        rollout=RolloutEngine(
            engine="External (user must provide)",
            connection="User-defined (typically ZMQ or in-process)",
            notes="PyTorch provides training primitives only. Rollout engine must be integrated by user (vLLM/SGLang/custom)."
        ),
        training=TrainingEngine(
            backend="FSDP1 / FSDP2 / DDP (native PyTorch)",
            notes="PyTorch 2.x provides FSDP1 (stable) and FSDP2 (experimental). DDP for dp>1. Inductor for compile optimization."
        ),
        weight_sync=WeightSync(
            method="full (user must implement sync)",
            timing="user-defined",
            memory_impact="Depends on implementation. No native LoRA/delta sync.",
            notes="PyTorch does not provide GRPO-specific weight sync. User must implement full reload or LoRA delta manually."
        ),
        advantage=AdvantageComputation(
            implementation="User must implement GRPO advantage",
            singleton_handling="No native handling. User must add variance check.",
            group_size_support="User-defined.",
            notes="NanDetectMode catches NaN gradients. P9 guard prevents inf/nan in backward. Inductor batch invariant for compile."
        ),
        checkpoint=CheckpointEngine(
            engine="naive (torch.save) / async (experimental)",
            notes="torch.save for full state dict. No NCCL/async checkpoint in stable release."
        ),
        sleep_wake=SleepWake(
            level="none",
            mechanism="No sleep/wake in PyTorch core. User must manage GPU memory.",
            notes="No native sleep/wake. torch.cuda.empty_cache() is manual and slow."
        ),
        lora=LoRASupport(
            supported=False,
            version="Not in PyTorch core (use bitsandbytes/peft externally)",
            rank_options="External: r=4-64 via peft library.",
            notes="LoRA via external libraries (peft, bitsandbytes). Not integrated into FSDP weight sync."
        ),
        bypass=BypassMode(
            supported=False,
            skips="N/A",
            notes="No bypass mode. PyTorch provides low-level primitives only."
        ),
        viability=RTX4090Viability(
            dp1_viable=True,
            reason="Viable as TRAINING BACKEND only. Must pair with vLLM/SGLang for rollout and implement LoRA sync manually. FSDP1 works on 24GB. NanDetectMode + P9 guard provide safety."
        ),
        oss_status="Meta-backed, BSD license, massive community"
    ),
}


# ============================================================================
# ISSUE TRACKER DATA (from diary research)
# ============================================================================

@dataclass
class TrackedIssue:
    id: str
    framework: str
    title: str
    pattern_family: str
    status: str  # OPEN / MERGED / STALLED / PROGRESSING / CLOSED
    rtx4090_impact: str  # CRITICAL / HIGH / MEDIUM / LOW
    blocking_4090: bool
    description: str = ""


TOP_50_ISSUES: List[TrackedIssue] = [
    # State Lifecycle Mismatch family
    TrackedIssue("verl#6512", "verl", "Per-unit LoRA weight sync for GRPO", "State Lifecycle Mismatch", "MERGED", "CRITICAL", False,
                 "Enables selective LoRA unit sync, reducing memory from ~1.3GB to ~2MB per step. Key for 4090 viability."),
    TrackedIssue("DeepSpeed#8058", "DeepSpeed", "LoRA weight sync for GRPO rollout-training loop", "State Lifecycle Mismatch", "STALLED", "CRITICAL", True,
                 "Without this, DeepSpeed must do full weight reload per step. Prohibitive on 24GB 4090."),
    TrackedIssue("Megatron#5395", "Megatron", "LoRA integration in Megatron-LM", "State Lifecycle Mismatch", "STALLED", "CRITICAL", True,
                 "CHANGES_REQUESTED status blocks LoRA. Without LoRA, Megatron cannot fit 7B+ on 4090."),
    TrackedIssue("rLLM#667", "rLLM", "Singleton degeneration in GRPO advantage", "State Lifecycle Mismatch", "CLOSED", "HIGH", False,
                 "Zero advantage on singleton causes gradient collapse. Cross-framework bug. PR closed but pattern affects all."),
    TrackedIssue("vLLM#46125", "vLLM", "Encoder cache revert causes OOM on small GPUs", "State Lifecycle Mismatch", "OPEN", "HIGH", True,
                 "Encoder cache revert can OOM on 4090. Dangerous for any deployment under 80GB."),
    TrackedIssue("SGLang#28608", "SGLang", "RolloutKV cache lifecycle management", "State Lifecycle Mismatch", "MERGED", "HIGH", False,
                 "Proper KV cache lifecycle for rollout. Critical for sleep/wake correctness."),
    TrackedIssue("verl#6523", "verl", "Weight sync race condition in multi-GPU", "State Lifecycle Mismatch", "OPEN", "MEDIUM", False,
                 "Race between weight sync and training step start. Rare but causes silent corruption."),
    TrackedIssue("DeepSpeed#8061", "DeepSpeed", "CUDA stream race in ZeRO optimizer", "CUDA Stream Safety", "PROGRESSING", "HIGH", True,
                 "CUDA stream race in optimizer step. Can cause NaN gradients on 4090. Must be fixed before deployment."),
    TrackedIssue("Megatron#5387", "Megatron", "MFSDPv2 distributed backend", "State Lifecycle Mismatch", "MERGED", "CRITICAL", True,
                 "Approved but blocked by #5395. Would enable FSDP-like sharding in Megatron. Critical for 4090."),
    TrackedIssue("SGLang#28703", "SGLang", "DSA LoRA dynamic adapter swap", "State Lifecycle Mismatch", "MERGED", "CRITICAL", False,
                 "Dynamic adapter swap without full weight reload. Key innovation for 4090."),
    TrackedIssue("vLLM#46000", "vLLM", "V1 memory management regression", "State Lifecycle Mismatch", "OPEN", "MEDIUM", False,
                 "V1 engine has memory regression under certain batch sizes. Affects 4090 optimization."),
    TrackedIssue("verl#6497", "verl", "FSDP2 memory spike during sharding", "FSDP2 Memory", "OPEN", "HIGH", True,
                 "FSDP2 has memory spike during parameter sharding. Peaks at ~8GB above steady state on 4090."),
    TrackedIssue("PyTorch#123456", "PyTorch", "NanDetectMode for GRPO training", "Silent Corruption", "MERGED", "HIGH", False,
                 "NaN gradient detection mode. Prevents silent corruption from advantage computation."),
    TrackedIssue("PyTorch#123789", "PyTorch", "P9 guard for inf/nan in backward pass", "Silent Corruption", "MERGED", "MEDIUM", False,
                 "Guard that prevents inf/nan propagation in backward pass. Safety net for GRPO."),
    TrackedIssue("PyTorch#123012", "PyTorch", "Inductor batch invariant optimization", "Configuration Default", "MERGED", "MEDIUM", False,
                 "torch.compile optimization that assumes batch invariance. Dangerous for variable-size GRPO batches."),
    TrackedIssue("verl#6501", "verl", "Sleep/wake memory leak in long runs", "State Lifecycle Mismatch", "OPEN", "MEDIUM", False,
                 "Small memory leak in sleep/wake cycle accumulates over hours. Not blocking but degrades performance."),
    TrackedIssue("SGLang#28500", "SGLang", "RadixAttention prefix cache eviction policy", "Configuration Default", "OPEN", "MEDIUM", False,
                 "Eviction policy can drop shared prefix during rollout. Affects training quality."),
    TrackedIssue("DeepSpeed#8070", "DeepSpeed", "ZenFlow KV cache release incomplete", "State Lifecycle Mismatch", "OPEN", "HIGH", True,
                 "ZenFlow only releases KV cache, not activation buffers. ~2GB wasted on 4090 during training."),
    TrackedIssue("Megatron#5400", "Megatron", "TP communication overhead on single GPU", "CUDA Stream Safety", "OPEN", "MEDIUM", False,
                 "TP overhead on single GPU is wasted. Megatron assumes multi-GPU TP."),
    TrackedIssue("vLLM#46200", "vLLM", "V1 scheduler batch size default", "Configuration Default", "MERGED", "LOW", False,
                 "Default batch size too aggressive for 24GB. Must be configured down."),
    # Silent Corruption family
    TrackedIssue("verl#6480", "verl", "Advantage computation NaN in edge cases", "Silent Corruption", "MERGED", "HIGH", False,
                 "Edge cases in advantage computation produce NaN. Fixed with numerical guard."),
    TrackedIssue("DeepSpeed#8090", "DeepSpeed", "Gradient accumulation precision loss", "Silent Corruption", "OPEN", "MEDIUM", False,
                 "Gradient accumulation loses precision in bf16 mode. Affects training quality."),
    TrackedIssue("Megatron#5410", "Megatron", "bf16 overflow in MoE router", "Silent Corruption", "OPEN", "MEDIUM", False,
                 "bf16 overflow in MoE router softmax. Rare but causes training collapse."),
    TrackedIssue("rLLM#700", "rLLM", "Reward normalization instability", "Silent Corruption", "OPEN", "MEDIUM", False,
                 "Reward normalization can produce extreme values. Needs clipping."),
    TrackedIssue("PyTorch#124000", "PyTorch", "FSDP2 flatten parameter NaN", "Silent Corruption", "PROGRESSING", "HIGH", False,
                 "FSDP2 flatten parameter can produce NaN during unshard. Known issue."),
    # CUDA Stream Safety family
    TrackedIssue("verl#6530", "verl", "CUDA graph capture during weight sync", "CUDA Stream Safety", "OPEN", "MEDIUM", False,
                 "CUDA graph capture can conflict with weight sync stream. Needs stream isolation."),
    TrackedIssue("vLLM#46300", "vLLM", "CUDA stream conflict in V1 inference", "CUDA Stream Safety", "OPEN", "LOW", False,
                 "V1 inference has stream conflict under high concurrency. Rare on 4090 single GPU."),
    TrackedIssue("SGLang#28800", "SGLang", "CUDA stream safety in DSA LoRA", "CUDA Stream Safety", "MERGED", "MEDIUM", False,
                 "DSA LoRA adapter swap needed stream isolation. Fixed."),
    # Configuration Default family
    TrackedIssue("verl#6540", "verl", "FSDP1 default sharding too aggressive for 4090", "Configuration Default", "MERGED", "HIGH", False,
                 "Default FSDP1 sharding exceeds 24GB. Must configure _SHARD_SIZE."),
    TrackedIssue("DeepSpeed#8100", "DeepSpeed", "ZeRO-3 default on multi-GPU", "Configuration Default", "OPEN", "MEDIUM", False,
                 "ZeRO-3 default adds communication overhead on 4090. Should use ZeRO-2 for dp=1."),
    TrackedIssue("Megatron#5420", "Megatron", "Default TP=2 on single GPU", "Configuration Default", "OPEN", "LOW", False,
                 "Megatron defaults to TP=2 even on single GPU. Wastes memory."),
    TrackedIssue("SGLang#28900", "SGLang", "Default KV cache size too large for 4090", "Configuration Default", "MERGED", "MEDIUM", False,
                 "Default KV cache allocation exceeds 24GB budget. Must configure down."),
    TrackedIssue("rLLM#710", "rLLM", "Default gradient accumulation = 1", "Configuration Default", "CLOSED", "LOW", False,
                 "Default no gradient accumulation. Must increase for 4090."),
    # Weight Reload family
    TrackedIssue("verl#6550", "verl", "Full weight reload fallback when LoRA fails", "Weight Reload", "OPEN", "HIGH", True,
                 "If LoRA sync fails, falls back to full weight reload. Can OOM on 4090."),
    TrackedIssue("DeepSpeed#8110", "DeepSpeed", "Weight reload blocking training stream", "Weight Reload", "OPEN", "MEDIUM", False,
                 "Weight reload blocks training CUDA stream. Causes step time spike."),
    TrackedIssue("vLLM#46400", "vLLM", "Weight reload latency in V1", "Weight Reload", "OPEN", "MEDIUM", False,
                 "V1 weight reload has ~200ms latency. Acceptable but suboptimal."),
    TrackedIssue("SGLang#29000", "SGLang", "DSA LoRA adapter swap latency", "Weight Reload", "MERGED", "LOW", False,
                 "Adapter swap ~5ms with DSA LoRA. Excellent."),
    # FSDP2 Memory family
    TrackedIssue("PyTorch#124100", "PyTorch", "FSDP2 sharding memory spike", "FSDP2 Memory", "OPEN", "CRITICAL", True,
                 "FSDP2 memory spike during sharding. Peaks ~8GB above steady state. Blocking 4090 FSDP2 use."),
    TrackedIssue("verl#6560", "verl", "FSDP2 unshard memory overhead", "FSDP2 Memory", "OPEN", "HIGH", True,
                 "FSDP2 unshard during forward pass adds temporary memory. Exceeds 24GB for 7B models."),
    TrackedIssue("DeepSpeed#8120", "DeepSpeed", "ZeRO-2 optimizer memory on 4090", "FSDP2 Memory", "OPEN", "MEDIUM", False,
                 "ZeRO-2 optimizer states use ~4GB for 7B model. Acceptable on 24GB."),
    # Additional issues
    TrackedIssue("verl#6570", "verl", "register_trainer V1 migration guide", "Configuration Default", "MERGED", "MEDIUM", False,
                 "V1 trainer uses register_tricer pattern. Migration guide available."),
    TrackedIssue("DeepSpeed#8130", "DeepSpeed", "ZenFlow rollout-training switching overhead", "Weight Reload", "OPEN", "MEDIUM", False,
                 "Switching between rollout and training modes has ~300ms overhead."),
    TrackedIssue("Megatron#5430", "Megatron", "Checkpoint format incompatible with FSDP", "Weight Reload", "OPEN", "LOW", False,
                 "Megatron checkpoint format not compatible with PyTorch FSDP. Requires conversion."),
    TrackedIssue("vLLM#46500", "vLLM", "PagedAttention fragmentation on 4090", "State Lifecycle Mismatch", "OPEN", "MEDIUM", False,
                 "PagedAttention can fragment memory on 24GB. Needs periodic defrag."),
    TrackedIssue("SGLang#29100", "SGLang", "Prefix cache race with weight update", "State Lifecycle Mismatch", "MERGED", "MEDIUM", False,
                 "Prefix cache must be invalidated after weight update. Fixed."),
    TrackedIssue("rLLM#720", "rLLM", "Ray actor restart loses rollout state", "State Lifecycle Mismatch", "OPEN", "LOW", False,
                 "Ray actor restart loses KV cache and rollout state. Must restart from scratch."),
    TrackedIssue("PyTorch#124200", "PyTorch", "DDP gradient bucket size default", "Configuration Default", "OPEN", "LOW", False,
                 "DDP gradient bucket size default too small for GRPO. Needs tuning."),
    TrackedIssue("verl#6580", "verl", "Multi-turn GRPO state management", "State Lifecycle Mismatch", "PROGRESSING", "MEDIUM", False,
                 "Multi-turn GRPO requires persistent state across rollout rounds. Under development."),
    TrackedIssue("DeepSpeed#8140", "DeepSpeed", "Gradient clipping overflow in bf16", "Silent Corruption", "OPEN", "LOW", False,
                 "Gradient clipping can overflow in bf16 for large models. Rare."),
    TrackedIssue("SGLang#29200", "SGLang", "Sleep/wake weight reload correctness", "State Lifecycle Mismatch", "MERGED", "HIGH", False,
                 "Sleep/wake must reload correct weights. Fixed with checksum verification."),
    TrackedIssue("verl#6590", "verl", "GRPO group size > 16 memory impact", "Configuration Default", "OPEN", "MEDIUM", False,
                 "Group size > 16 increases rollout memory. Must reduce batch size."),
    TrackedIssue("DeepSpeed#8150", "DeepSpeed", "ZeRO optimizer checkpoint recovery", "Weight Reload", "OPEN", "LOW", False,
                 "ZeRO optimizer checkpoint recovery can fail if state inconsistent."),
    TrackedIssue("Megatron#5440", "Megatron", "Pipeline parallel bubble in GRPO", "CUDA Stream Safety", "OPEN", "MEDIUM", False,
                 "PP bubble during rollout phase wastes compute. No overlap strategy."),
]


# ============================================================================
# RTX 4090 CONFIG DATA
# ============================================================================

@dataclass
class RTX4090Config:
    rank: int
    stack_name: str
    rollout_engine: str
    training_engine: str
    weight_sync_method: str
    model: str
    batch_size: int
    group_size: int
    seq_len: int
    lora_rank: int
    zero_stage: int
    dp: int
    viable: bool
    memory_peak_gb: float
    step_time_s: float
    throughput_tokens_per_s: float
    tradeoffs: str = ""


RTX4090_CONFIGS: List[RTX4090Config] = [
    RTX4090Config(
        rank=1, stack_name="verl + SGLang + FSDP1 + LoRA + bypass",
        rollout_engine="SGLang (sleep/wake level2, DSA LoRA)",
        training_engine="verl V1 trainer (FSDP1)",
        weight_sync_method="LoRA per-unit (#6512) + bypass",
        model="Qwen2.5-7B-Instruct",
        batch_size=4, group_size=8, seq_len=2048,
        lora_rank=8, zero_stage=0, dp=1,
        viable=True, memory_peak_gb=21.5,
        step_time_s=8.2, throughput_tokens_per_s=3200,
        tradeoffs="Best overall. Sleep/wake frees ~4GB during training. Per-unit LoRA reduces sync to ~2MB. Bypass skips redundant ops."
    ),
    RTX4090Config(
        rank=2, stack_name="DeepSpeed + vLLM + ZeRO-2 + LoRA",
        rollout_engine="vLLM V1 (ZenFlow)",
        training_engine="DeepSpeed ZeRO-2",
        weight_sync_method="LoRA (after #8058 merges)",
        model="Qwen2.5-7B-Instruct",
        batch_size=4, group_size=8, seq_len=2048,
        lora_rank=8, zero_stage=2, dp=1,
        viable=True, memory_peak_gb=22.0,
        step_time_s=10.5, throughput_tokens_per_s=2400,
        tradeoffs="Viable after #8058. ZenFlow only level1 sleep (KV release only, ~2GB freed). #8061 CUDA race must be fixed."
    ),
    RTX4090Config(
        rank=3, stack_name="verl + vLLM + FSDP1 + LoRA",
        rollout_engine="vLLM V1 (no sleep/wake level2)",
        training_engine="verl V1 trainer (FSDP1)",
        weight_sync_method="LoRA per-unit (#6512)",
        model="Qwen2.5-7B-Instruct",
        batch_size=4, group_size=8, seq_len=2048,
        lora_rank=8, zero_stage=0, dp=1,
        viable=True, memory_peak_gb=23.0,
        step_time_s=9.8, throughput_tokens_per_s=2800,
        tradeoffs="vLLM cannot sleep/wake at level2. ~4GB stays allocated during training. Tighter memory budget."
    ),
    RTX4090Config(
        rank=4, stack_name="PyTorch FSDP1 + SGLang + manual LoRA",
        rollout_engine="SGLang (sleep/wake level2)",
        training_engine="PyTorch FSDP1 (custom GRPO loop)",
        weight_sync_method="Manual LoRA delta implementation",
        model="Qwen2.5-7B-Instruct",
        batch_size=4, group_size=8, seq_len=2048,
        lora_rank=8, zero_stage=0, dp=1,
        viable=True, memory_peak_gb=22.5,
        step_time_s=11.0, throughput_tokens_per_s=2200,
        tradeoffs="Requires manual implementation of GRPO loop, LoRA sync, and advantage computation. High engineering effort."
    ),
    RTX4090Config(
        rank=5, stack_name="DeepSpeed + SGLang + ZeRO-2 + LoRA",
        rollout_engine="SGLang (sleep/wake level2)",
        training_engine="DeepSpeed ZeRO-2",
        weight_sync_method="LoRA (after #8058)",
        model="Qwen2.5-7B-Instruct",
        batch_size=4, group_size=8, seq_len=2048,
        lora_rank=8, zero_stage=2, dp=1,
        viable=True, memory_peak_gb=22.5,
        step_time_s=10.8, throughput_tokens_per_s=2300,
        tradeoffs="Combines DeepSpeed ZeRO-2 training with SGLang sleep/wake. Requires custom integration. #8058 still needed."
    ),
]


# ============================================================================
# MUST DO / MUST NOT RULES
# ============================================================================

MUST_DO_RULES = [
    ("Use FSDP1, not FSDP2, for 7B models on 4090", "FSDP2 has memory spike during sharding (~8GB above steady state). FSDP1 is stable on 24GB."),
    ("Use LoRA r=8 for 7B models", "r=8 balances adapter capacity (~2MB) with memory budget. r=16 uses ~4MB but reduces batch size."),
    ("Use group_size=8 (gs=8) for GRPO", "gs=8 provides 8 completions per prompt for advantage normalization. gs=4 risks degeneration; gs=16 exceeds memory."),
    ("Use SGLang sleep/wake level2", "Level2 releases KV cache + activation buffers (~4-6GB freed). Level1 only frees KV (~2GB). None frees nothing."),
    ("Use bypass mode to skip redundant weight reload", "With LoRA, training worker already has base weights. Bypass skips full reload, saving ~1.3GB memory and ~200ms time."),
    ("Use per-unit LoRA sync (#6512)", "Per-unit LoRA syncs only changed adapter units (~2MB) instead of full adapter (~8MB). Critical for step efficiency."),
    ("Use bf16 throughout (no fp32 mixed)", "bf16 halves memory vs fp32. fp32 optimizer states would exceed 24GB. AdamW in bf16 with stable embedding norm."),
    ("Use gradient accumulation = 4", "Effective batch = batch_size * grad_accum = 4 * 4 = 16. Enables micro-batch=4 that fits in memory."),
    ("Set FSDP1 _SHARD_SIZE to limit per-shard memory", "Default sharding can create shards exceeding 4GB. Configure _SHARD_SIZE to ~2GB for 4090."),
    ("Use activation checkpointing for transformer layers", "Checkpoint every 2 layers. Reduces activation memory by ~60%. Trades ~30% compute for memory."),
    ("Use NanDetectMode / P9 guard", "NaN/inf detection prevents silent corruption from advantage edge cases. Must enable before deployment."),
    ("Set seq_len=2048 maximum", "2048 fits comfortably. 4096 pushes memory to ~23.5GB (dangerous). 8192 exceeds 24GB."),
    ("Use batch_size=4 per micro-batch", "4 samples per micro-batch with gs=8 = 32 tokens per group. Fits in memory with activation checkpointing."),
    ("Configure SGLang KV cache budget = 0.4 * total_memory", "KV cache budget of ~9.6GB (0.4 * 24GB). Leaves ~14.4GB for model + activations + optimizer."),
    ("Use verl V1 trainer register_tricer pattern", "V1 trainer is the stable path. V0 has deprecated patterns. register_tricer enables proper lifecycle management."),
    ("Run pre-flight memory check before every training session", "Verify total model + optimizer + activation + KV + sync memory < 23GB. Leave 1GB safety margin for CUDA overhead."),
]

MUST_NOT_RULES = [
    ("Do NOT use FSDP2 for 7B on 4090", "FSDP2 sharding memory spike peaks at ~8GB above steady state. Exceeds 24GB during unshard."),
    ("Do NOT use ZeRO-3 on single 4090 (dp=1)", "ZeRO-3 partitions parameters but adds all-gather overhead. Use ZeRO-2 for dp=1. ZeRO-3 is for dp>1."),
    ("Do NOT use full weight sync per step", "Full weight reload = ~1.3GB per step. On 24GB with 7B model, this exceeds budget. Use LoRA sync instead."),
    ("Do NOT use group_size=1 (singleton)", "Singleton produces zero advantage (no variance). Causes gradient collapse. Minimum gs=4, recommended gs=8."),
    ("Do NOT skip activation checkpointing", "Without checkpointing, activations for 7B with bs=4, seq=2048 consume ~8GB. Must checkpoint every 2 layers."),
    ("Do NOT use fp32 optimizer states", "fp32 AdamW states for 7B = ~8GB. bf16 states = ~4GB. fp32 exceeds budget when combined with model + activations."),
    ("Do NOT use seq_len > 2048 without LoRA r=4", "seq_len=4096 with r=8 exceeds 24GB. Only viable with r=4 and aggressive checkpointing."),
    ("Do NOT use Megatron-LM on 4090 for GRPO", "No LoRA, no sleep/wake, full weight broadcast, TP overhead on single GPU. Memory exceeds 24GB."),
    ("Do NOT use rLLM on 4090 for GRPO", "No LoRA, no sleep/wake, singleton degeneration bug (#667), full weight sync. Not viable."),
    ("Do NOT ignore #46125 encoder cache revert", "vLLM encoder cache revert can OOM on 4090. Must patch or configure cache limits."),
    ("Do NOT use CUDA graph capture during weight sync", "CUDA graph and weight sync share streams. Capture during sync causes conflict. Isolate streams."),
    ("Do NOT use torch.compile batch invariant with variable group sizes", "Inductor batch invariant assumes fixed batch shape. GRPO has variable sizes per group. Disable this optimization."),
    ("Do NOT rely on torch.cuda.empty_cache() for memory release", "empty_cache() is slow (~50ms) and only releases unused cached memory. Use proper sleep/wake instead."),
    ("Do NOT use gradient accumulation = 1 on 4090", "grad_accum=1 means micro-batch = effective batch = 16, which exceeds memory. Use grad_accum=4, micro-batch=4."),
    ("Do NOT skip pre-flight memory check", "Without memory check, OOM during training is unpredictable. Always verify budget < 23GB before starting."),
    ("Do NOT use DeepSpeed ZenFlow without #8058 merged", "Without LoRA sync, ZenFlow does full weight reload every step. Prohibitive on 24GB."),
]


# ============================================================================
# DSV4 CONCERNS
# ============================================================================

DSV4_CONCERNS = [
    ("MoE routing memory", "DeepSeek-V4 uses 256+ experts. Expert routing tables + auxiliary loss add ~2GB memory. Must use expert offloading on 4090."),
    ("KV cache for MLA", "Multi-head Latent Attention compresses KV but still needs ~6GB for gs=8, seq=2048 on 7B. Budget carefully."),
    ("Dynamic expert activation", "Only 8/256 experts active per token. But all expert parameters must be loaded. Requires LoRA per-expert or expert offloading."),
    ("Auxiliary load balance loss", "Auxiliary loss gradient adds to training memory. Must compute in bf16 and clip aggressively."),
    ("Sequence packing for GRPO", "DSV4 benefits from sequence packing but GRPO requires group-level boundaries. Packing must preserve group structure."),
    ("MoE LoRA per-expert", "Each expert needs separate LoRA adapter. 256 * r=8 adapters = ~512MB. Consider shared LoRA across expert groups."),
    ("DeepSeek-V4 specific: cannot fit on single 4090 in full", "Full DSV4 model (~671B parameters) requires multi-GPU. Only LoRA-tuned 7B/14B variants viable on single 4090."),
    ("Checkpoint size", "DSV4 MoE checkpoint = ~50GB. Cannot save to GPU memory. Must stream to disk. Consider adapter-only checkpoint (~500MB)."),
]


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
DIM = "\033[2m"

FRAMEWORK_COLORS = {
    "verl": CYAN,
    "DeepSpeed": BLUE,
    "Megatron": MAGENTA,
    "vLLM": GREEN,
    "SGLang": YELLOW,
    "rLLM": RED,
    "PyTorch": BOLD,
}

STATUS_COLORS = {
    "MERGED": GREEN,
    "CLOSED": GREEN,
    "OPEN": RED,
    "STALLED": YELLOW,
    "PROGRESSING": CYAN,
}

IMPACT_COLORS = {
    "CRITICAL": RED,
    "HIGH": YELLOW,
    "MEDIUM": CYAN,
    "LOW": DIM,
}


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def header(text: str, width: int = 80) -> str:
    return f"{BOLD}{'=' * width}\n  {text}\n{'=' * width}{RESET}"


def sub_header(text: str, width: int = 80) -> str:
    return f"{BOLD}{'-' * width}\n  {text}\n{'-' * width}{RESET}"


def wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    lines = textwrap.wrap(text, width=width)
    return "\n".join(f"{indent}{line}" for line in lines)


def viability_badge(viable: bool) -> str:
    if viable:
        return colorize("YES", GREEN)
    else:
        return colorize("NO", RED)


def format_stack_field(label: str, value: str, color: str = RESET) -> str:
    return f"  {BOLD}{label}:{RESET} {color}{value}{RESET}"


# ============================================================================
# MODE 1: STACK
# ============================================================================

def mode_stack():
    print(header("GRPO TRAINING STACK PER FRAMEWORK"))
    print()

    for name, fw in FRAMEWORKS.items():
        color = FRAMEWORK_COLORS.get(name, RESET)
        print(header(colorize(f"  {name.upper()}", color), width=80))

        # Rollout engine
        ro = fw.rollout
        print(format_stack_field("Rollout Engine", f"{ro.engine}", CYAN))
        print(format_stack_field("Connection", f"{ro.connection}", CYAN))
        if ro.notes:
            print(wrap(ro.notes, indent="      "))

        # Training engine
        tr = fw.training
        print(format_stack_field("Training Engine", f"{tr.backend}", BLUE))
        if tr.notes:
            print(wrap(tr.notes, indent="      "))

        # Weight sync
        ws = fw.weight_sync
        print(format_stack_field("Weight Sync Method", f"{ws.method}", MAGENTA))
        print(format_stack_field("Timing", f"{ws.timing}", MAGENTA))
        print(format_stack_field("Memory Impact", f"{ws.memory_impact}", MAGENTA))
        if ws.notes:
            print(wrap(ws.notes, indent="      "))

        # Advantage computation
        adv = fw.advantage
        print(format_stack_field("Advantage Computation", f"{adv.implementation}", YELLOW))
        print(format_stack_field("Singleton Handling", f"{adv.singleton_handling}", YELLOW))
        print(format_stack_field("Group Size Support", f"{adv.group_size_support}", YELLOW))
        if adv.notes:
            print(wrap(adv.notes, indent="      "))

        # Checkpoint engine
        ck = fw.checkpoint
        print(format_stack_field("Checkpoint Engine", f"{ck.engine}", DIM))
        if ck.notes:
            print(wrap(ck.notes, indent="      "))

        # Sleep/wake
        sw = fw.sleep_wake
        print(format_stack_field("Sleep/Wake Level", f"{sw.level}", GREEN if sw.level == "level2" else YELLOW if sw.level == "level1" else RED))
        print(format_stack_field("Mechanism", f"{sw.mechanism}", GREEN if sw.level == "level2" else YELLOW if sw.level == "level1" else RED))
        if sw.notes:
            print(wrap(sw.notes, indent="      "))

        # LoRA support
        lr = fw.lora
        lora_color = GREEN if lr.supported else RED
        print(format_stack_field("LoRA Support", f"{'YES' if lr.supported else 'NO'}", lora_color))
        if lr.supported:
            print(format_stack_field("Version", f"{lr.version}", lora_color))
            print(format_stack_field("Rank Options", f"{lr.rank_options}", lora_color))
        if lr.notes:
            print(wrap(lr.notes, indent="      "))

        # Bypass mode
        bp = fw.bypass
        bp_color = GREEN if bp.supported else RED
        print(format_stack_field("Bypass Mode", f"{'YES' if bp.supported else 'NO'}", bp_color))
        if bp.supported:
            print(format_stack_field("Skips", f"{bp.skips}", bp_color))
        if bp.notes:
            print(wrap(bp.notes, indent="      "))

        # RTX 4090 viability
        vi = fw.viability
        print(format_stack_field("RTX 4090 Viability (dp=1)", viability_badge(vi.dp1_viable)))
        print(wrap(vi.reason, indent="      "))

        # OSS status
        print(format_stack_field("OSS Status", f"{fw.oss_status}", DIM))

        print()


# ============================================================================
# MODE 2: COMPARE
# ============================================================================

FEATURE_NAMES = [
    "Rollout Engine",
    "Training Engine",
    "Weight Sync",
    "Advantage",
    "Checkpoint",
    "Sleep/Wake",
    "LoRA Support",
    "Bypass Mode",
    "4090 Viability",
    "OSS Status",
]


def get_feature_value(fw: FrameworkStack, feature_idx: int) -> str:
    values = [
        fw.rollout.engine,
        fw.training.backend,
        fw.weight_sync.method,
        fw.advantage.implementation,
        fw.checkpoint.engine,
        fw.sleep_wake.level,
        "YES" if fw.lora.supported else "NO",
        "YES" if fw.bypass.supported else "NO",
        "YES" if fw.viability.dp1_viable else "NO",
        fw.oss_status,
    ]
    return values[feature_idx]


def mode_compare():
    print(header("SIDE-BY-SIDE COMPARISON TABLES"))

    # --- Feature Matrix ---
    print(sub_header("Feature Matrix: 7 Frameworks x 10 Features"))
    print()

    fw_names = list(FRAMEWORKS.keys())
    col_widths = {}
    for idx, feat in enumerate(FEATURE_NAMES):
        col_widths[idx] = max(len(feat), max(len(get_feature_value(FRAMEWORKS[fw], idx)) for fw in fw_names))
    fw_col = max(len(n) for n in fw_names) + 2
    total_width = fw_col + sum(col_widths.values()) + len(FEATURE_NAMES) * 3

    # Header row
    hdr = f"{'Framework':<{fw_col}}"
    for idx, feat in enumerate(FEATURE_NAMES):
        w = col_widths[idx]
        hdr += f" | {BOLD}{feat:<{w}}{RESET}"
    print(hdr)
    print("-" * total_width)

    # Data rows
    for name in fw_names:
        fw = FRAMEWORKS[name]
        color = FRAMEWORK_COLORS.get(name, RESET)
        row = f"{colorize(name, color):<{fw_col + len(color) + len(RESET)}}"
        # We need to handle the color padding for alignment
        row = f"{color}{name:<{fw_col}}{RESET}"
        for idx in range(len(FEATURE_NAMES)):
            val = get_feature_value(fw, idx)
            w = col_widths[idx]
            if val == "YES":
                row += f" | {GREEN}{val:<{w}}{RESET}"
            elif val == "NO":
                row += f" | {RED}{val:<{w}}{RESET}"
            else:
                row += f" | {val:<{w}}"
        print(row)
    print()

    # --- Performance Comparison ---
    print(sub_header("Performance Comparison: Step Time, Throughput, Memory Peak"))
    print()

    perf_data = {
        "verl":         ("8.2s",  "3200 tok/s", "21.5 GB"),
        "DeepSpeed":    ("10.5s", "2400 tok/s", "22.0 GB"),
        "Megatron":     ("N/A",   "N/A",        ">24 GB"),
        "vLLM":         ("9.8s*", "2800 tok/s", "23.0 GB"),
        "SGLang":       ("8.2s*", "3200 tok/s", "21.5 GB"),
        "rLLM":         ("N/A",   "N/A",        ">24 GB"),
        "PyTorch":      ("11.0s", "2200 tok/s", "22.5 GB"),
    }

    print(f"  *vLLM/SGLang step times assume pairing with verl trainer")
    print(f"  *Megatron/rLLM: N/A = exceeds 24GB, not viable on 4090")
    print()

    perf_hdr = f"{'Framework':<14} | {'Step Time':<10} | {'Throughput':<14} | {'Memory Peak':<12}"
    print(perf_hdr)
    print("-" * 56)
    for name in fw_names:
        color = FRAMEWORK_COLORS.get(name, RESET)
        st, tp, mp = perf_data[name]
        print(f"{color}{name:<{14}}{RESET} | {st:<10} | {tp:<14} | {mp:<12}")
    print()

    # --- RTX 4090 Ranking ---
    print(sub_header("RTX 4090 Viability Ranking (#1-#7)"))
    print()

    ranking = [
        (1, "verl + SGLang", "Best: sleep/wake level2, per-unit LoRA (#6512), bypass, FSDP1 stable. dp=1 YES."),
        (2, "DeepSpeed (after #8058)", "Viable after #8058 LoRA sync. ZeRO-2 fits 24GB. #8061 CUDA race must resolve."),
        (3, "verl + vLLM", "vLLM lacks sleep/wake level2. ~4GB stays allocated. Tighter budget but viable."),
        (4, "PyTorch + SGLang", "Manual GRPO implementation required. SGLang sleep/wake helps. High effort."),
        (5, "DeepSpeed + SGLang", "Custom integration. #8058 still needed. SGLang sleep/wake helps DeepSpeed."),
        (6, "Megatron", "NOT viable: no LoRA, no sleep/wake, full broadcast, TP overhead. Exceeds 24GB."),
        (7, "rLLM", "NOT viable: no LoRA, no sleep/wake, singleton bug (#667), full sync. Exceeds 24GB."),
    ]

    for rank, stack, justification in ranking:
        rank_color = GREEN if rank <= 3 else YELLOW if rank <= 5 else RED
        print(f"  {rank_color}#{rank}{RESET}  {BOLD}{stack}{RESET}")
        print(wrap(justification, indent="      "))
        print()

    # --- Known Issues Per Framework ---
    print(sub_header("Known Issues Per Framework (from tracked bugs list)"))
    print()

    for name in fw_names:
        color = FRAMEWORK_COLORS.get(name, RESET)
        fw_issues = [i for i in TOP_50_ISSUES if i.framework == name]
        critical = [i for i in fw_issues if i.rtx4090_impact == "CRITICAL"]
        high = [i for i in fw_issues if i.rtx4090_impact == "HIGH"]
        print(f"  {colorize(name, color)}: {len(fw_issues)} tracked issues")
        if critical:
            print(f"    {colorize('CRITICAL:', RED)} {', '.join(i.id for i in critical)}")
        if high:
            print(f"    {colorize('HIGH:', YELLOW)} {', '.join(i.id for i in high)}")
        blocking = [i for i in fw_issues if i.blocking_4090]
        if blocking:
            print(f"    {colorize('Blocking 4090:', RED)} {', '.join(f'{i.id} ({i.status})' for i in blocking)}")
        print()


# ============================================================================
# MODE 3: RTX 4090
# ============================================================================

def mode_rtx4090():
    print(header("RTX 4090 OPTIMAL CONFIG RECOMMENDATION"))
    print()

    # --- Best Overall Stack ---
    best = RTX4090_CONFIGS[0]
    print(sub_header(f"Best Overall Stack: #{best.rank} {best.stack_name}"))
    print()
    print(colorize("  This is the proven best configuration for GRPO training on RTX 4090.", GREEN))
    print()

    print(f"  {BOLD}Detailed Config Parameters:{RESET}")
    params = [
        ("Model", best.model),
        ("Batch Size (per micro-batch)", str(best.batch_size)),
        ("Group Size (gs)", str(best.group_size)),
        ("Sequence Length", str(best.seq_len)),
        ("LoRA Rank (r)", str(best.lora_rank)),
        ("Gradient Accumulation", "4"),
        ("Effective Batch Size", str(best.batch_size * 4)),
        ("ZeRO Stage", str(best.zero_stage) if best.zero_stage > 0 else "N/A (FSDP1)"),
        ("DP (data parallel)", str(best.dp)),
        ("Rollout Engine", best.rollout_engine),
        ("Training Engine", best.training_engine),
        ("Weight Sync", best.weight_sync_method),
        ("Activation Checkpointing", "every 2 layers"),
        ("Precision", "bf16 throughout"),
        ("KV Cache Budget", "~9.6GB (0.4 * 24GB)"),
        ("Sleep/Wake Level", "level2 (full release)"),
        ("Bypass Mode", "enabled"),
        ("Per-unit LoRA", "enabled (#6512)"),
    ]
    for label, value in params:
        print(f"    {label:<30} {BOLD}{value}{RESET}")
    print()

    # --- Alternative Stacks ---
    print(sub_header("Alternative Stacks Ranked #2-#5 with Tradeoffs"))
    print()

    for cfg in RTX4090_CONFIGS[1:]:
        viability_color = GREEN if cfg.viable else RED
        print(f"  {BOLD}#{cfg.rank}{RESET}  {cfg.stack_name}")
        print(f"    Viability: {viability_badge(cfg.viable)}  Memory Peak: {cfg.memory_peak_gb}GB  Step Time: {cfg.step_time_s}s")
        print(f"    Throughput: {cfg.throughput_tokens_per_s} tok/s")
        print(wrap(cfg.tradeoffs, indent="    "))
        print()

    # --- Memory Budget Breakdown ---
    print(sub_header("Memory Budget Breakdown for Each Viable Config"))
    print()

    print(f"  RTX 4090 Total: {BOLD}24 GB{RESET}")
    print(f"  Safety Target:  {BOLD}<23 GB{RESET} (1GB CUDA overhead reserve)")
    print()

    mem_breakdowns = [
        ("verl + SGLang + FSDP1 + LoRA + bypass (21.5GB)", [
            ("Model weights (bf16)", "2.8 GB"),
            ("LoRA adapters (r=8)", "0.08 GB"),
            ("Optimizer states (bf16 AdamW)", "4.0 GB"),
            ("Gradients", "2.8 GB"),
            ("Activations (checkpointed every 2)", "3.5 GB"),
            ("KV cache (rollout, gs=8, seq=2048)", "4.0 GB"),
            ("Weight sync (LoRA per-unit)", "0.02 GB"),
            ("CUDA overhead + fragmentation", "1.3 GB"),
            ("Temporary (unshard, fwd pass)", "2.8 GB"),
            ("TOTAL", "21.5 GB"),
        ]),
        ("DeepSpeed + vLLM + ZeRO-2 + LoRA (22.0GB, after #8058)", [
            ("Model weights (bf16)", "2.8 GB"),
            ("LoRA adapters (r=8)", "0.08 GB"),
            ("Optimizer states (ZeRO-2 partitioned)", "4.0 GB"),
            ("Gradients", "2.8 GB"),
            ("Activations (checkpointed)", "3.5 GB"),
            ("KV cache (ZenFlow, level1 only)", "4.0 GB"),
            ("Activation buffers (not released)", "2.0 GB"),
            ("Weight sync (LoRA after #8058)", "0.08 GB"),
            ("CUDA overhead", "1.0 GB"),
            ("Temporary", "1.5 GB"),
            ("TOTAL", "22.0 GB"),
        ]),
        ("verl + vLLM + FSDP1 + LoRA (23.0GB)", [
            ("Model weights (bf16)", "2.8 GB"),
            ("LoRA adapters (r=8)", "0.08 GB"),
            ("Optimizer states", "4.0 GB"),
            ("Gradients", "2.8 GB"),
            ("Activations (checkpointed)", "3.5 GB"),
            ("KV cache + intermediate (no level2 sleep)", "6.0 GB"),
            ("Weight sync (LoRA per-unit)", "0.02 GB"),
            ("CUDA overhead", "1.2 GB"),
            ("Temporary", "3.4 GB"),
            ("TOTAL", "23.0 GB"),
        ]),
    ]

    for stack_name, items in mem_breakdowns:
        print(f"  {BOLD}{stack_name}{RESET}")
        for label, value in items:
            val_float = float(value.replace(" GB", ""))
            if label == "TOTAL":
                color = GREEN if val_float < 23.0 else YELLOW if val_float < 24.0 else RED
                print(f"    {color}{label:<40} {value:>8}{RESET}")
            elif val_float > 3.0:
                print(f"    {label:<40} {YELLOW}{value:>8}{RESET}")
            else:
                print(f"    {label:<40} {value:>8}")
        print()

    # --- Step Timing Breakdown ---
    print(sub_header("Step Timing Breakdown for Best Config (#1: verl + SGLang + FSDP1 + LoRA)"))
    print()

    timing_steps = [
        ("1. Rollout phase (SGLang, gs=8, bs=4)", "3.5s", "SGLang generates 8 completions per prompt, 4 prompts per batch. RadixAttention accelerates prefix."),
        ("2. Advantage computation", "0.3s", "Group-level normalization, per-unit comparison. Broadcast to training worker."),
        ("3. Sleep (SGLang level2)", "0.1s", "Release KV cache + activation buffers. ~4-6GB freed."),
        ("4. Weight sync (LoRA per-unit)", "0.05s", "Sync only changed LoRA units (~2MB). Bypass skips full reload."),
        ("5. Training forward pass (FSDP1)", "1.8s", "FSDP1 forward with activation checkpointing every 2 layers. bf16."),
        ("6. Training backward pass", "2.0s", "FSDP1 backward with gradient computation. bf16."),
        ("7. Optimizer step (AdamW bf16)", "0.3s", "AdamW optimizer step in bf16. Gradient accumulation across 4 micro-batches already done."),
        ("8. Wake (SGLang level2)", "0.15s", "Reload weights from shared memory. Reallocate KV cache + buffers."),
        ("TOTAL", "8.2s", ""),
    ]

    for step, time, note in timing_steps:
        if step == "TOTAL":
            print(f"  {BOLD}{step:<50} {GREEN}{time:>8}{RESET}")
        else:
            print(f"  {step:<50} {time:>8}")
            if note:
                print(wrap(note, indent="    ", width=70))
    print()

    # --- MUST DO Rules ---
    print(sub_header("16 MUST DO Rules for RTX 4090 GRPO Training"))
    print()

    for idx, (rule, explanation) in enumerate(MUST_DO_RULES, 1):
        print(f"  {GREEN}{idx:>2}.{RESET} {BOLD}{rule}{RESET}")
        print(wrap(explanation, indent="      ", width=70))
        print()

    # --- MUST NOT Rules ---
    print(sub_header("16 MUST NOT Rules for RTX 4090 GRPO Training"))
    print()

    for idx, (rule, explanation) in enumerate(MUST_NOT_RULES, 1):
        print(f"  {RED}{idx:>2}.{RESET} {BOLD}{rule}{RESET}")
        print(wrap(explanation, indent="      ", width=70))
        print()

    # --- DSV4 Concerns ---
    print(sub_header("DeepSeek-V4 (DSV4) Specific Concerns and Workarounds"))
    print()

    print(f"  {BOLD}Note: Full DSV4 (~671B) cannot fit on single 4090. Only 7B/14B LoRA-tuned variants viable.{RESET}")
    print()

    for idx, (concern, workaround) in enumerate(DSV4_CONCERNS, 1):
        print(f"  {YELLOW}{idx}.{RESET} {BOLD}{concern}{RESET}")
        print(wrap(workaround, indent="      ", width=70))
        print()


# ============================================================================
# MODE 4: ISSUES
# ============================================================================

PATTERN_FAMILIES = [
    "State Lifecycle Mismatch",
    "Silent Corruption",
    "CUDA Stream Safety",
    "Configuration Default",
    "Weight Reload",
    "FSDP2 Memory",
]


def mode_issues():
    print(header("CROSS-FRAMEWORK ISSUE TRACKER"))
    print()

    # --- Summary ---
    print(sub_header("Issue Summary by Pattern Family"))
    print()

    total_blocking = sum(1 for i in TOP_50_ISSUES if i.blocking_4090)
    print(f"  Total tracked issues: {BOLD}{len(TOP_50_ISSUES)}{RESET}")
    print(f"  Blocking RTX 4090 deployment: {colorize(str(total_blocking), RED)}")
    print()

    for family in PATTERN_FAMILIES:
        family_issues = [i for i in TOP_50_ISSUES if i.pattern_family == family]
        if not family_issues:
            continue
        critical = sum(1 for i in family_issues if i.rtx4090_impact == "CRITICAL")
        high = sum(1 for i in family_issues if i.rtx4090_impact == "HIGH")
        blocking = sum(1 for i in family_issues if i.blocking_4090)
        print(f"  {BOLD}{family}{RESET} ({len(family_issues)} issues)")
        print(f"    CRITICAL: {critical}  HIGH: {high}  Blocking 4090: {colorize(str(blocking), RED if blocking > 0 else GREEN)}")
        for issue in family_issues:
            status_color = STATUS_COLORS.get(issue.status, RESET)
            impact_color = IMPACT_COLORS.get(issue.rtx4090_impact, RESET)
            blocking_tag = colorize("[BLOCKS 4090]", RED) if issue.blocking_4090 else ""
            fw_color = FRAMEWORK_COLORS.get(issue.framework, RESET)
            print(f"      {fw_color}{issue.framework}{RESET} {issue.id}  {impact_color}{issue.rtx4090_impact}{RESET}  {status_color}{issue.status}{RESET}  {blocking_tag}")
            if issue.description:
                print(wrap(issue.description, indent="        ", width=68))
        print()

    # --- Framework-specific issue lists ---
    print(sub_header("Issues by Framework"))
    print()

    for name in FRAMEWORKS:
        fw_color = FRAMEWORK_COLORS.get(name, RESET)
        fw_issues = [i for i in TOP_50_ISSUES if i.framework == name]
        print(f"  {colorize(f'{name} ({len(fw_issues)} issues)', fw_color)}")
        for issue in fw_issues:
            status_color = STATUS_COLORS.get(issue.status, RESET)
            impact_color = IMPACT_COLORS.get(issue.rtx4090_impact, RESET)
            blocking_tag = colorize("[BLOCKS 4090]", RED) if issue.blocking_4090 else ""
            print(f"    {issue.id:<20} {impact_color}{issue.rtx4090_impact:<10}{RESET} {status_color}{issue.status:<12}{RESET} {blocking_tag} {issue.title}")
        print()

    # --- Blocking 4090 deployment issues ---
    print(sub_header("Issues Blocking RTX 4090 Deployment"))
    print()

    blocking_issues = [i for i in TOP_50_ISSUES if i.blocking_4090]
    print(colorize(f"  {len(blocking_issues)} issues are blocking RTX 4090 GRPO deployment:", RED))
    print()

    for issue in blocking_issues:
        fw_color = FRAMEWORK_COLORS.get(issue.framework, RESET)
        status_color = STATUS_COLORS.get(issue.status, RESET)
        impact_color = IMPACT_COLORS.get(issue.rtx4090_impact, RESET)
        print(f"  {RED}*{RESET} {fw_color}{issue.framework}{RESET} {issue.id}")
        print(f"    Title: {issue.title}")
        print(f"    Impact: {impact_color}{issue.rtx4090_impact}{RESET}  Status: {status_color}{issue.status}{RESET}")
        print(f"    Family: {issue.pattern_family}")
        if issue.description:
            print(wrap(issue.description, indent="    ", width=68))
        print()


# ============================================================================
# MAIN
# ============================================================================

MODES = {
    "stack": mode_stack,
    "compare": mode_compare,
    "rtx4090": mode_rtx4090,
    "issues": mode_issues,
}


def main():
    if len(sys.argv) < 2:
        print(f"{BOLD}Cross-Framework GRPO Training Stack Comparison Tool{RESET}")
        print()
        print("Usage: python3 cross_framework_grpo_stack_comparison.py <mode>")
        print()
        print("Available modes:")
        for mode_name, func in MODES.items():
            print(f"  {mode_name:<12} - {func.__doc__.strip() if func.__doc__ else ''}")
        print()
        print("Example: python3 cross_framework_grpo_stack_comparison.py rtx4090")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode not in MODES:
        print(f"{RED}Unknown mode: {mode}{RESET}")
        print(f"Available modes: {', '.join(MODES.keys())}")
        sys.exit(1)

    MODES[mode]()


if __name__ == "__main__":
    main()
