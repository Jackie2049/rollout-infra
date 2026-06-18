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
        symptom="NaN in gradient → multi-stream race condition",
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


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 GRPO NaN Debugging Guide")
    parser.add_argument("--mode", choices=["info", "checklist", "patterns", "rtx4090"], default="info")
    args = parser.parse_args()

    if args.mode == "info":
        print_info()
    elif args.mode == "checklist":
        print_checklist()
    elif args.mode == "patterns":
        print_patterns()
    elif args.mode == "rtx4090":
        print_rtx4090()


if __name__ == "__main__":
    main()
