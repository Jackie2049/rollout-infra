#!/usr/bin/env python3
"""RTX 4090 GRPO NaN Debugging Guide

Comprehensive NaN detection and root cause analysis for GRPO training
on RTX 4090. Integrates PyTorch NanDetectMode (#187653) with
framework-specific NaN patterns.

Usage:
  python3 tools/grpo_nan_debugging_guide.py --mode info
  python3 tools/grpo_nan_debugging_guide.py --mode checklist
  python3 tools/grpo_nan_debugging_guide.py --mode patterns
  python3 tools/grpo_nan_debugging_guide.py --mode rtx4090
  python3 tools/grpo_nan_debugging_guide.py --mode detect
  python3 tools/grpo_nan_debugging_guide.py --mode analyze
  python3 tools/grpo_nan_debugging_guide.py --mode combined
"""

import argparse
from dataclasses import dataclass
from enum import Enum


class NaNSource(Enum):
    SM89_BATCH_INVAR = "sm89_batch_invariance"
    LORA_RANK_MISMATCH = "lora_rank_mismatch"
    ZERO_OVERLAP_COMM = "zero_overlap_comm"
    FSDP2_CPU_LEAK = "fsdp2_cpu_leak"
    DSV4_DSA_INDEXER = "dsv4_dsa_indexer"
    MOE_ROUTER_OVERFLOW = "moe_router_overflow"
    OVERFLOW_LOGITS = "overflow_logits"
    ZERO_GRAD_CLIPPING = "zero_grad_clipping"
    TRITON_INPLACE_AUTOGRAD_BYPASS = "triton_inplace_autograd_bypass"
    MXFP8_MOE_CACHE_CLOBBER = "mxfp8_moe_cache_clobber"
    MTP_GRAMMAR_FSM_CONFLICT = "mtp_grammar_fsm_conflict"


@dataclass
class NaNPattern:
    source: NaNSource
    framework: str
    symptom: str
    first_op: str  # which torch op first produces NaN
    fix: str
    priority: int  # 1-10, higher = more common on RTX 4090


NAN_PATTERNS = [
    NaNPattern(
        source=NaNSource.SM89_BATCH_INVAR,
        framework="PyTorch/vLLM",
        symptom="Different numerical results per batch size → gradual NaN accumulation",
        first_op="aten.mean.dim or aten.sum.dim_IntList (Inductor-fused RMSNorm)",
        fix="enforce_eager=True or Inductor SM<90 Fusion Guard (#P9)",
        priority=9,
    ),
    NaNPattern(
        source=NaNSource.LORA_RANK_MISMATCH,
        framework="verl",
        symptom="NaN in LoRA matmul → wrong weight shapes",
        first_op="aten.addmm (LoRA A*B computation)",
        fix="LoRA rank=32/alpha=64 MANDATORY (rank=64 breaks EOS #6782)",
        priority=8,
    ),
    NaNPattern(
        source=NaNSource.ZERO_OVERLAP_COMM,
        framework="DeepSpeed",
        symptom="NaN in gradient → overlap_comm+torch.compile multi-stream race condition (#8061)",
        first_op="aten.all_reduce or aten.reduce_scatter (overlap_comm ops)",
        fix="overlap_comm=False MANDATORY on single GPU (#8061)",
        priority=7,
    ),
    NaNPattern(
        source=NaNSource.FSDP2_CPU_LEAK,
        framework="verl",
        symptom="Stale parameters → NaN in forward after many steps",
        first_op="aten.mm or aten.linear (stale weight matmul)",
        fix="Monitor CPU memory growth per step (#6468), restart periodically",
        priority=6,
    ),
    NaNPattern(
        source=NaNSource.DSV4_DSA_INDEXER,
        framework="vLLM/vLLM-Ascend",
        symptom="DSA indexer selects wrong positions → garbage attention → NaN",
        first_op="aten.sparse_softmax or attention computation",
        fix="enforce_eager=True MANDATORY for DSV4",
        priority=6,
    ),
    NaNPattern(
        source=NaNSource.MOE_ROUTER_OVERFLOW,
        framework="vLLM/Megatron/DeepSpeed",
        symptom="Router logits overflow → Inf → top-k selects wrong experts → NaN",
        first_op="aten.topk or aten.softmax (router computation)",
        fix="z-loss regularization or logit clamping",
        priority=5,
    ),
    NaNPattern(
        source=NaNSource.OVERFLOW_LOGITS,
        framework="All",
        symptom="Large logits → exp overflow → NaN in softmax/cross_entropy",
        first_op="aten.softmax or aten.cross_entropy",
        fix="logit clamping or FP32 logits in BF16 model",
        priority=4,
    ),
    NaNPattern(
        source=NaNSource.ZERO_GRAD_CLIPPING,
        framework="DeepSpeed/Megatron",
        symptom="Global grad clipping at 0 → all gradients zero → NaN in optimizer step",
        first_op="aten.norm or aten.clip (gradient clipping)",
        fix="gradient_clipping=1.0 MANDATORY for AdamW, skip for Muon (#8068)",
        priority=3,
    ),
    NaNPattern(
        source=NaNSource.TRITON_INPLACE_AUTOGRAD_BYPASS,
        framework="Megatron-LM",
        symptom="NaN at iter 2 — Triton in-place rotary_fwd_q_kernel bypasses autograd version counter → MQA aliasing amplifies corruption (NEW class D, #5317)",
        first_op="aten.add.Tensor or aten.mm.default (accumulated corruption from iter 1)",
        fix="apply_rope_fusion=False MANDATORY for DSV4-Hybrid/MLA (#5317)",
        priority=6,
    ),
    NaNPattern(
        source=NaNSource.MXFP8_MOE_CACHE_CLOBBER,
        framework="SGLang",
        symptom="MXFP8 MoE shuffle cache physically CLOBBERED on RL weight reload → 64x accuracy blowup (#28676)",
        first_op="aten.moe_dispatch or expert computation (cache returns clobbered data)",
        fix="dict.clear() on cache + weight-load funnel call (#28676, +28/-2)",
        priority=7,
    ),
    NaNPattern(
        source=NaNSource.MTP_GRAMMAR_FSM_CONFLICT,
        framework="vLLM",
        symptom="MTP+grammar FSM conflict → 58% request failure rate → state lifecycle mismatch (#46118)",
        first_op="aten.embedding or token generation (MTP draft conflicts with FSM constraints)",
        fix="Reset FSM state at MTP boundary — fix PR #44297 (+455/-15)",
        priority=6,
    ),
]


NAN_DETECT_CODE = """
# === PyTorch NanDetectMode Usage (#187653) ===
# Requires: PyTorch with #187653 merged (or patch manually)

import torch

# Basic: NaN detection in forward pass
with torch.utils.nan_detect.NanDetectMode():
    output = model(input_ids)
    # RuntimeError if any op produces NaN: "Function aten.X returned NaN values"

# Advanced: also detect Inf (useful for MoE router overflow)
with torch.utils.nan_detect.NanDetectMode(check_inf=True):
    output = model(input_ids)
    # Also catches ±Inf → router logits overflow detection

# GRPO-specific: wrap entire training step
def training_step(model, batch, optimizer):
    with torch.utils.nan_detect.NanDetectMode(check_inf=True):
        # Forward + loss
        output = model(batch["input_ids"])
        loss = compute_grpo_loss(output, batch)

    # If forward passes NaN check → backward should be safe
    loss.backward()
    optimizer.step()

# === Alternative: manual NaN checking (works on any PyTorch version) ===

def check_nan_gradients(model):
    \"\"\"Check all parameter gradients for NaN.\"\"\"
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in: {name}")
            return True
    return False

def check_nan_outputs(output_dict):
    \"\"\"Check model output tensors for NaN.\"\"\"
    for key, value in output_dict.items():
        if torch.is_tensor(value) and torch.isnan(value).any():
            print(f"NaN in output: {key} → first NaN at index {torch.isnan(value).nonzero()[0]}")
            return True
    return False
"""


COMBINED_NAN_DEBUG_CODE = """
# === Combined NanDetectMode + detect_anomaly for Comprehensive NaN Debugging ===
# Requires: PyTorch with #187653 merged (or patch manually)

import torch

def debug_full_nan(model, inputs, loss_fn):
    \"\"\"Comprehensive NaN debugging: forward (NanDetectMode) + backward (detect_anomaly).\"\"\"
    results = {"forward": None, "backward": None}

    # Phase 1: Forward NaN detection (NanDetectMode)
    #   → Checks every ATen op output for NaN/Inf
    #   → Raises RuntimeError with EXACT op name if NaN found
    #   → 7/10+ common GRPO NaN sources are forward-detectable
    try:
        from torch.utils.nan_detect import NanDetectMode
        with NanDetectMode(check_inf=True):  # Also catch Inf (MoE router overflow)
            output = model(inputs)
        results["forward"] = "clean"
    except RuntimeError as e:
        if "returned NaN values" in str(e) or "returned non-finite" in str(e):
            results["forward"] = f"NaN at: {e}"
            return results  # No point running backward if forward has NaN
        else:
            # RuntimeError from NanDetectMode import failure (PR not merged)
            # Fall back to manual forward check
            output = model(inputs)
            if not torch.isfinite(output).all():
                results["forward"] = "NaN/Inf in model output (manual check)"
                return results
            results["forward"] = "clean (NanDetectMode unavailable)"

    # Phase 2: Loss computation (fp32 for GRPO!)
    loss = loss_fn(output)
    if not torch.isfinite(loss):
        results["forward"] = f"NaN/Inf in loss: {loss.item()}"
        return results

    # Phase 3: Backward NaN detection (detect_anomaly)
    #   → Only needed if forward is clean
    #   → Checks gradient computation for NaN
    #   → Catches backward-only issues like gradient_clipping=0 (#8068)
    try:
        with torch.autograd.detect_anomaly():
            loss.backward()
        results["backward"] = "clean"
    except RuntimeError as e:
        results["backward"] = f"NaN in backward: {str(e)}"

    return results

# === Production: periodic NaN monitoring ===
NAN_CHECK_INTERVAL = 100  # Check every 100 steps (not every step!)

def periodic_nan_check(model_engine, batch, step):
    \"\"\"Lightweight NaN check at periodic intervals.\"\"\"
    if step % NAN_CHECK_INTERVAL != 0:
        return  # Skip most steps (zero overhead)

    try:
        from torch.utils.nan_detect import NanDetectMode
        with NanDetectMode(check_inf=True):
            test_loss = model_engine(batch)
        print(f"Step {step}: forward NaN check PASSED")
    except RuntimeError as e:
        print(f"Step {step}: FORWARD NaN DETECTED: {e}")
        # Enter full debugging mode with combined mode
        return "nan_detected"
    except ImportError:
        # NanDetectMode not available (PR not merged) → manual check
        test_loss = model_engine(batch)
        if torch.isfinite(test_loss):
            print(f"Step {step}: forward NaN check PASSED (manual)")
        else:
            print(f"Step {step}: FORWARD NaN DETECTED (manual)")
            return "nan_detected"
"""


RTX4090_CHECKLIST = """
# === RTX 4090 GRPO NaN Prevention Checklist ===

## MUST DO (before training):
1. enforce_eager=True → prevents CUDA graph replay → SM89 batch invariance safe
2. LoRA rank=32/alpha=64 → prevents rank=64 EOS break (#6782)
3. gradient_clipping=1.0 → prevents zero gradients (#8068)
4. overlap_comm=False → prevents multi-stream NaN (#8061)
5. bypass_mode=True → eliminates ref model → reduces memory pressure
6. detach model_output → prevents per-micro-batch graph retention (#6699/#C9)

## MUST NOT DO:
1. LoRA rank=64 → breaks EOS (#6782)
2. overlap_comm=True on single GPU → NaN guaranteed (#8061)
3. torch.compile without Fusion Guard → batch-dependent results (#39096)
4. CUDA graphs with DSV4 → garbage output (#45972)
5. gradient_clipping=0 → zero gradients → NaN in optimizer

## NaN Debugging Workflow:
1. Wrap model forward in NanDetectMode → locate FIRST NaN-producing op
2. If NaN in matmul → check LoRA shapes, stale weights, dtype mismatch
3. If NaN in softmax → check logit overflow, add clamping
4. If NaN in gradient → check overlap_comm, gradient_clipping setting
5. If NaN after many steps → monitor FSDP2 CPU memory growth (#6468)
6. If NaN only at specific batch sizes → Inductor fusion issue (#39096)
"""


def print_info():
    print("=" * 80)
    print("RTX 4090 GRPO NaN Debugging Guide")
    print("=" * 80)
    print()
    print("NanDetectMode (#187653): Forward-pass NaN detection")
    print("  → Complements detect_anomaly (backward-only)")
    print("  → Locates FIRST NaN-producing operation in forward pass")
    print()
    print("Top 3 RTX 4090 NaN sources:")
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority)[:3]:
        print(f"  [{p.priority}] {p.source.value} ({p.framework})")
        print(f"    First op: {p.first_op}")
        print(f"    Fix: {p.fix}")
    print()
    print("NanDetectMode code snippet:")
    print(NAN_DETECT_CODE)


def print_checklist():
    print("=" * 80)
    print("RTX 4090 GRPO NaN Prevention Checklist")
    print("=" * 80)
    print(RTX4090_CHECKLIST)


def print_patterns():
    print("=" * 80)
    print("GRPO NaN Patterns — Cross-Framework")
    print("=" * 80)
    print()
    print("| Priority | Source | Framework | First NaN Op | Fix |")
    print("|----------|--------|-----------|-------------|-----|")
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority):
        print(f"| P{p.priority} | {p.source.value} | {p.framework} | {p.first_op[:40]} | {p.fix[:50]} |")


def print_rtx4090():
    print("=" * 80)
    print("RTX 4090 GRPO NaN Debugging — Quick Reference")
    print("=" * 80)
    print()
    print("## MUST DO (6 rules)")
    print("  1. enforce_eager=True")
    print("  2. LoRA rank=32/alpha=64")
    print("  3. gradient_clipping=1.0")
    print("  4. overlap_comm=False")
    print("  5. bypass_mode=True")
    print("  6. detach model_output")
    print()
    print("## MUST NOT DO (5 rules)")
    print("  1. LoRA rank=64 (breaks EOS)")
    print("  2. overlap_comm=True on single GPU")
    print("  3. torch.compile without Fusion Guard")
    print("  4. CUDA graphs with DSV4")
    print("  5. gradient_clipping=0")
    print()
    print("## Debugging: NanDetectMode workflow")
    print("  Step 1: NanDetectMode(check_inf=True) → find FIRST NaN op")
    print("  Step 2: Based on op type → apply pattern-specific fix")
    print()
    print("## Top 3 NaN sources on RTX 4090:")
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority)[:3]:
        print(f"  P{p.priority}: {p.source.value} → {p.fix}")


# Classification: which patterns NanDetectMode can catch (forward-pass NaN)
FORWARD_DETECTABLE = {
    NaNSource.SM89_BATCH_INVAR,
    NaNSource.LORA_RANK_MISMATCH,
    NaNSource.ZERO_OVERLAP_COMM,  # indirectly: NaN in subsequent forward after gradient corruption
    NaNSource.FSDP2_CPU_LEAK,
    NaNSource.DSV4_DSA_INDEXER,
    NaNSource.MOE_ROUTER_OVERFLOW,  # with check_inf=True
    NaNSource.OVERFLOW_LOGITS,      # with check_inf=True
    NaNSource.TRITON_INPLACE_AUTOGRAD_BYPASS,  # iter 2 forward NaN
    NaNSource.MXFP8_MOE_CACHE_CLOBBER,
    NaNSource.MTP_GRAMMAR_FSM_CONFLICT,
}

# Map ATen op name prefixes to likely NaN patterns
OP_TO_PATTERN_MAP = {
    "aten.div": [NaNSource.ZERO_OVERLAP_COMM, NaNSource.OVERFLOW_LOGITS],
    "aten.mm": [NaNSource.FSDP2_CPU_LEAK, NaNSource.TRITON_INPLACE_AUTOGRAD_BYPASS,
                NaNSource.SM89_BATCH_INVAR],
    "aten.addmm": [NaNSource.LORA_RANK_MISMATCH],
    "aten.linear": [NaNSource.LORA_RANK_MISMATCH, NaNSource.FSDP2_CPU_LEAK],
    "aten.softmax": [NaNSource.MOE_ROUTER_OVERFLOW, NaNSource.OVERFLOW_LOGITS],
    "aten.add": [NaNSource.ZERO_OVERLAP_COMM, NaNSource.TRITON_INPLACE_AUTOGRAD_BYPASS],
    "aten.flash_attention": [NaNSource.DSV4_DSA_INDEXER],
    "aten.moe_dispatch": [NaNSource.MXFP8_MOE_CACHE_CLOBBER],
    "aten.embedding": [NaNSource.MTP_GRAMMAR_FSM_CONFLICT],
    "aten.reduce_scatter": [NaNSource.ZERO_OVERLAP_COMM],
    "aten.all_reduce": [NaNSource.ZERO_OVERLAP_COMM],
}

# Framework-specific recommendations
FRAMEWORK_RECOMMENDATIONS = {
    "verl": [
        "Use FSDP backend (NOT Automodel/Megatron/TorchTitan — have memory leak #6699)",
        "bypass_mode=True → eliminates ref model memory",
        "LoRA rank=32/alpha=64 → prevents EOS break (#6782)",
        "Monitor FSDP2 CPU memory growth per step (#6468)",
        "detach model_output → prevents per-micro-batch graph retention (#6699)",
        "sleep_level=1 (LoRA adapter) → 80x payload reduction vs sleep_level=2",
    ],
    "DeepSpeed": [
        "ZeRO-2 + CPU_Adam ONLY (ZeRO-3 = pure overhead on single GPU)",
        "overlap_comm=False MANDATORY (#8061)",
        "gradient_clipping=1.0 MANDATORY (default 0 → no clipping #8068)",
        "pin_memory=True (CPUOffloadPolicy default, already optimal)",
        "Avoid ZeRO-3+PEFT LoRA (regression #8072/#8073)",
        "AutoEP+LoRA+EP=1 → MoE training viable on RTX 4090",
    ],
    "Megatron-LM": [
        "apply_rope_fusion=False for DSV4-Hybrid/MLA (#5317)",
        "skip_grad_norm_clip for Muon optimizer (#5395)",
        "gradient_clipping=1.0 for AdamW (#8068)",
        "MFSDPv2 (DBuffer primitives) for FSDP integration (#5387)",
        "test_mla_yarn_rope_apply.py uses .detach() → ONLY verifies arithmetic, NOT autograd",
    ],
    "vLLM": [
        "enforce_eager=True MANDATORY for DSV4",
        "MTP+structured_output currently BROKEN (#46118)",
        "kv-cache-dtype auto can produce garbage with MTP (#46088)",
        "MoE is_sym guard regression fixed (#45656)",
        "aot_eager = batch-invariant BY DESIGN on SM89 (#46085)",
        "Disable CUDA graphs with DSV4 (#45972)",
    ],
    "SGLang": [
        "MXFP8 MoE cache MUST be cleared at weight-reload boundary (#28676)",
        "Ring cursors MUST be reset at weight-reload boundary for GRPO (#28695)",
        "GDN intermittent degeneracy worsens over uptime (#28679)",
        "enforce_eager=True for DSV4",
        "Disable CUDA graphs with DSV4",
    ],
}


def print_detect():
    print("=" * 80)
    print("NanDetectMode Forward-Pass NaN Detection — GRPO Debugging")
    print("=" * 80)
    print()
    print("KEY INSIGHT: 7/10 common GRPO NaN sources are forward-pass detectable.")
    print("NanDetectMode (#187653) catches them by intercepting EVERY ATen")
    print("operation and checking outputs for NaN/Inf BEFORE propagation.")
    print()
    print("=" * 80)
    print("Forward-Pass NaN Detection Workflow (6 Steps)")
    print("=" * 80)
    print()
    print("  Step 1: REPRODUCE NaN")
    print("    → Run training until NaN appears in loss/reward")
    print("    → Record: which step, which batch, which model state")
    print()
    print("  Step 2: WRAP forward pass in NanDetectMode")
    print("    with NanDetectMode(check_inf=True):")
    print("        outputs = policy_model(queries)")
    print("    → If RuntimeError → EXACTLY which ATen op produced NaN")
    print("    → If no error → NaN originates in BACKWARD pass")
    print("    → (fall back to detect_anomaly for backward-only NaN)")
    print()
    print("  Step 3: EXAMINE the offending op with pdb")
    print("    → Break at NanDetectMode.__torch_dispatch__ line")
    print("    → Inspect func (op name) and args (input tensors)")
    print("    → Example: 'Function aten.div.Tensor returned NaN values'")
    print("      → division by zero: advantage std=0, overlap_comm race")
    print()
    print("  Step 4: MAP op to known NaN pattern")
    print("    → Use 'analyze' mode for pattern matching")
    print("    → Common mappings on RTX 4090:")
    print("      aten.div.Tensor   → division by zero (advantage std=0, #8061)")
    print("      aten.mm.default   → corrupted weights (FSDP2 leak, LoRA mismatch)")
    print("      aten.addmm        → LoRA weight shape mismatch (#6782)")
    print("      aten.softmax      → logit overflow (MoE router, FP16 overflow)")
    print("      aten.flash_attn   → DSA indexer stale positions")
    print("      aten.add.Tensor   → accumulated corruption from iter 1 (#5317)")
    print()
    print("  Step 5: APPLY pattern-specific fix")
    print("    → See 'rtx4090' mode for MUST DO rules")
    print("    → See 'analyze' mode for framework-specific recommendations")
    print()
    print("  Step 6: VERIFY fix with NanDetectMode")
    print("    → Re-run forward pass → should complete without RuntimeError")
    print()
    print("=" * 80)
    print("How TorchDispatchMode Works for GRPO Debugging")
    print("=" * 80)
    print()
    print("  TorchDispatchMode intercepts at the ATen dispatch layer:")
    print("    Python API → Autograd dispatch → __torch_dispatch__ [HERE]")
    print("                                     → Backend dispatch (CUDA kernel)")
    print()
    print("  NanDetectMode.__torch_dispatch__ flow:")
    print("    1. Calls func(*args, **kwargs) → operation executes on GPU")
    print("    2. Flattens result via tree_flatten")
    print("    3. For each floating-point tensor: checks isnan() or isfinite()")
    print("    4. If NaN found: raise RuntimeError IMMEDIATELY (stops propagation)")
    print("    5. If clean: return result (transparent pass-through)")
    print()
    print("  Design choices:")
    print("    → check_inf=True is opt-in (attention masks use -inf intentionally)")
    print("    → Integer/boolean/empty tensors skipped (no NaN possible)")
    print("    → Meta/fake tensors skipped (torch.compile compatible)")
    print("    → Auto-disabled by Dynamo (__init_subclass__ wraps with _disable_dynamo)")
    print("    → ~10-20% overhead → acceptable for debugging, MUST disable for production")
    print()
    print("=" * 80)
    print("GRPO NaN Sources: Forward-Detectable vs Backward-Only")
    print("=" * 80)
    print()
    fwd_count = len(FORWARD_DETECTABLE)
    total_count = len(NAN_PATTERNS)
    bwd_count = total_count - fwd_count
    print(f"  {fwd_count}/{total_count} tracked NaN patterns are forward-pass detectable:")
    print()
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority):
        if p.source in FORWARD_DETECTABLE:
            print(f"    P{p.priority}: {p.source.value} ({p.framework})")
            print(f"      First op: {p.first_op[:60]}")
            print(f"      NanDetectMode: YES")
    print()
    print(f"  {bwd_count}/{total_count} are backward-only (need detect_anomaly):")
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority):
        if p.source not in FORWARD_DETECTABLE:
            print(f"    P{p.priority}: {p.source.value} ({p.framework})")
            print(f"      First op: {p.first_op[:60]}")
            print(f"      NanDetectMode: NO (backward issue)")
    print()
    print("=" * 80)
    print("NanDetectMode Code Snippet")
    print("=" * 80)
    print(NAN_DETECT_CODE)


def print_analyze():
    print("=" * 80)
    print("GRPO NaN Analysis — Pattern Matching + NanDetectMode Integration")
    print("=" * 80)
    print()
    print("=" * 80)
    print("ATen Op → NaN Pattern Lookup Table")
    print("=" * 80)
    print()
    print("When NanDetectMode reports 'Function aten.X returned NaN values',")
    print("map the ATen op to known patterns below:")
    print()
    print("| ATen Op Prefix | Likely Patterns | Priority Range |")
    print("|---------------|----------------|---------------|")
    for op, sources in sorted(OP_TO_PATTERN_MAP.items()):
        patterns = [p for p in NAN_PATTERNS if p.source in sources]
        p_str = ", ".join(f"{p.source.value}(P{p.priority})" for p in sorted(patterns, key=lambda x: -x.priority))
        pri_range = f"P{min(p.priority for p in patterns)}-P{max(p.priority for p in patterns)}"
        print(f"| {op} | {p_str} | {pri_range} |")
    print()
    print("=" * 80)
    print("Pattern Matching: NanDetectMode Error → Root Cause")
    print("=" * 80)
    print()
    print("Given NanDetectMode output 'Function aten.X returned NaN values':")
    print("  1. Extract op name: aten.X")
    print("  2. Look up in OP_TO_PATTERN_MAP above")
    print("  3. Check each candidate pattern's symptom against your scenario")
    print("  4. Apply highest-priority matching pattern's fix first")
    print()
    print("Example resolution workflow:")
    print()
    print("  NanDetectMode reports: 'Function aten.addmm.default returned NaN values'")
    print("  → OP_TO_PATTERN_MAP: LORA_RANK_MISMATCH (P8)")
    print("  → Symptom match: 'NaN in LoRA matmul → wrong weight shapes'")
    print("  → Fix: LoRA rank=32/alpha=64 MANDATORY (rank=64 breaks EOS #6782)")
    print()
    print("  NanDetectMode reports: 'Function aten.softmax.default returned NaN values'")
    print("  → OP_TO_PATTERN_MAP: MOE_ROUTER_OVERFLOW (P5), OVERFLOW_LOGITS (P4)")
    print("  → If MoE model → MOE_ROUTER_OVERFLOW → cast router to fp32")
    print("  → If dense model → OVERFLOW_LOGITS → add logit clamping")
    print()
    print("  NanDetectMode reports: 'Function aten.add.Tensor returned NaN values'")
    print("  → OP_TO_PATTERN_MAP: ZERO_OVERLAP_COMM (P7), TRITON_INPLACE_AUTOGRAD_BYPASS (P6)")
    print("  → If DeepSpeed ZeRO → overlap_comm race → overlap_comm=False (#8061)")
    print("  → If Megatron DSV4 → fused RoPE corruption → apply_rope_fusion=False (#5317)")
    print()
    print("=" * 80)
    print("Framework-Specific NaN Recommendations")
    print("=" * 80)
    print()
    for fw, recs in FRAMEWORK_RECOMMENDATIONS.items():
        print(f"## {fw}")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
        print()
    print("=" * 80)
    print("All NaN Patterns (Full Table)")
    print("=" * 80)
    print()
    print("| Priority | Source | Framework | Symptom | First Op | Fix | Fwd-Detect? |")
    print("|----------|--------|-----------|---------|----------|-----|-------------|")
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority):
        fwd = "YES" if p.source in FORWARD_DETECTABLE else "NO"
        print(f"| P{p.priority} | {p.source.value[:25]} | {p.framework[:12]} | {p.symptom[:45]} | {p.first_op[:35]} | {p.fix[:40]} | {fwd} |")


def print_combined():
    print("=" * 80)
    print("Combined Forward+Backward NaN Debugging — NanDetectMode + detect_anomaly")
    print("=" * 80)
    print()
    print("RECOMMENDATION: Use BOTH NanDetectMode and detect_anomaly together")
    print("for comprehensive NaN debugging. They cover different pass directions:")
    print()
    print("  NanDetectMode (#187653):  Forward-pass NaN → catches 7/10+ patterns")
    print("  detect_anomaly:           Backward-pass NaN → catches gradient issues")
    print("  Combined:                 Both directions → full coverage")
    print()
    print("=" * 80)
    print("Why Combined Debugging Is Necessary")
    print("=" * 80)
    print()
    print("  detect_anomaly alone has 4 critical limitations (#160016):")
    print("    1. Un-pdb-able traceback (raises inside C++ autograd internals)")
    print("    2. Misleading backward-first detection (points to wrong op)")
    print("    3. Silent NaN pass-through (forward NaN with finite gradient → no error)")
    print("    4. Backward-only NaN unreported (-Inf gradient → NaN only in optimizer)")
    print()
    print("  NanDetectMode alone misses backward-only NaN (e.g., #8068 grad clipping)")
    print()
    print("  Combined approach: NanDetectMode fires FIRST for forward NaN,")
    print("  detect_anomaly fires for backward NaN if forward is clean.")
    print()
    print("=" * 80)
    print("Combined Debugging Code")
    print("=" * 80)
    print()
    print(COMBINED_NAN_DEBUG_CODE)
    print()
    print("=" * 80)
    print("Decision Tree: Which Tool to Use First")
    print("=" * 80)
    print()
    print("  ALWAYS start with NanDetectMode (forward NaN is more common):")
    print()
    print("  1. NanDetectMode(check_inf=True) → forward pass")
    print("     → If NaN detected: you have EXACT op → use 'analyze' mode")
    print("     → If forward clean: proceed to step 2")
    print()
    print("  2. detect_anomaly → backward pass only")
    print("     → If NaN detected: backward op → gradient computation issue")
    print("     → If backward clean: proceed to step 3")
    print()
    print("  3. Check optimizer state (CPU_Adam overflow, Muon clipping)")
    print("     → NaN may appear only after optimizer.step()")
    print("     → Check per-parameter grad norms before and after optimizer step")
    print()
    print("  4. If NaN after many steps → monitor CPU memory per step (#6468)")
    print("     → FSDP2 leak: 0.6-6.3 GiB/step → monotonic growth")
    print("     → Restart periodically as workaround")
    print()
    print("  NEVER use detect_anomaly as first tool → misleading traceback for forward NaN")
    print("  ALWAYS start with NanDetectMode → pinpoint location for forward NaN")
    print("  ONLY fall back to detect_anomaly if NanDetectMode shows clean forward")
    print()
    print("=" * 80)
    print("Pattern Coverage: Forward vs Backward vs Combined")
    print("=" * 80)
    print()
    print("| Pattern | Source | NanDetectMode? | detect_anomaly? | Combined? |")
    print("|---------|--------|---------------|----------------|-----------|")
    for p in sorted(NAN_PATTERNS, key=lambda x: -x.priority):
        fwd = "YES" if p.source in FORWARD_DETECTABLE else "NO"
        bwd = "NO" if p.source in FORWARD_DETECTABLE else "YES"
        combined = "YES"
        if p.source == NaNSource.ZERO_OVERLAP_COMM:
            bwd = "YES (indirect)"  # backward origin, forward symptom
        print(f"| P{p.priority} | {p.source.value[:25]} | {fwd} | {bwd} | {combined} |")
    print()
    print("Summary: Combined mode catches ALL patterns. NanDetectMode catches")
    print(f"  {len(FORWARD_DETECTABLE)}/{len(NAN_PATTERNS)}. detect_anomaly alone catches only")
    print(f"  {len(NAN_PATTERNS) - len(FORWARD_DETECTABLE)}/{len(NAN_PATTERNS)} directly (backward-only patterns).")



def main():
    parser = argparse.ArgumentParser(description="RTX 4090 GRPO NaN Debugging Guide")
    parser.add_argument("--mode", choices=["info", "checklist", "patterns", "rtx4090", "detect", "analyze", "combined"], default="info")
    args = parser.parse_args()

    if args.mode == "info":
        print_info()
    elif args.mode == "checklist":
        print_checklist()
    elif args.mode == "patterns":
        print_patterns()
    elif args.mode == "rtx4090":
        print_rtx4090()
    elif args.mode == "detect":
        print_detect()
    elif args.mode == "analyze":
        print_analyze()
    elif args.mode == "combined":
        print_combined()


if __name__ == "__main__":
    main()
