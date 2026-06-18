#!/usr/bin/env python3
"""7-Framework Critical Issues Dashboard — RTX 4090 Consultant Quick Check

Quick-access dashboard for monitoring critical issues across 7 frameworks.
Organized by urgency and RTX 4090 impact.

Usage:
  python3 tools/seven_framework_critical_dashboard.py          # Full dashboard
  python3 tools/seven_framework_critical_dashboard.py --brief   # Brief summary
  python3 tools/seven_framework_critical_dashboard.py --filter BLOCKED  # Filter by status
"""

import argparse
import sys
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CRITICAL ISSUES DATABASE
# ═══════════════════════════════════════════════════════════════

CRITICAL_ISSUES = [
    # ── DeepSpeed ──
    {
        "id": "DS-1",
        "framework": "DeepSpeed",
        "issue": "#8072",
        "title": "ZeRO-3+PEFT LoRA regression",
        "severity": "CRITICAL",
        "status": "BLOCKED",
        "rtx4090": "ZeRO-3+LoRA BROKEN on v0.19.2! ZeRO-2 SAFE",
        "must": "Use ZeRO-2 + CPU_Adam only",
        "days_open": 3,
        "comments": 0,
        "fix_pr": "#8073 (2-line, 0 reviews, STALLED)",
        "url": "https://github.com/microsoft/DeepSpeed/issues/8072",
    },
    {
        "id": "DS-2",
        "framework": "DeepSpeed",
        "issue": "#8061",
        "title": "overlap_comm + torch.compile = NaN",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "MUST overlap_comm=False on single GPU!",
        "must": "Set overlap_comm=False for dp=1",
        "days_open": 14,
        "comments": 3,
        "fix_pr": "None (root cause confirmed: reduction_stream timing)",
        "url": "https://github.com/microsoft/DeepSpeed/issues/8061",
    },
    {
        "id": "DS-3",
        "framework": "DeepSpeed",
        "issue": "#8068",
        "title": "gradient_clipping default 0→1.0",
        "severity": "HIGH",
        "status": "STALLED",
        "rtx4090": "MUST set clip_grad=1.0 explicitly for GRPO",
        "must": "Set gradient_clipping=1.0 in config",
        "days_open": 7,
        "comments": 0,
        "fix_pr": "None (0 reviews)",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8068",
    },
    {
        "id": "DS-4",
        "framework": "DeepSpeed",
        "issue": "#8058",
        "title": "ZenFlow CPU optimizer (2944→256 MiB)",
        "severity": "HIGH",
        "status": "PROGRESSING",
        "rtx4090": "BEST optimizer for tight 24GB",
        "must": "Monitor for merge",
        "days_open": 14,
        "comments": 4,
        "fix_pr": "delock reviewing",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8058",
    },
    {
        "id": "DS-5",
        "framework": "DeepSpeed",
        "issue": "#8075",
        "title": "fd leak in deepspeed_io_handle_t::wait()",
        "severity": "MEDIUM",
        "status": "OPEN",
        "rtx4090": "File descriptor leak → long training runs → eventual failure",
        "must": "Monitor for merge",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "#8075 (close fd in wait())",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8075",
    },
    # ── Megatron-LM ──
    {
        "id": "MG-1",
        "framework": "Megatron-LM",
        "issue": "#5394",
        "title": "ChainedOptimizer Muon clipping stalls",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "Muon NOT viable without skip_grad_norm_clip",
        "must": "Wait for #5395 merge or skip clipping",
        "days_open": 1,
        "comments": 2,
        "fix_pr": "#5395 skip_grad_norm_clip (+15/-1, 0 reviews)",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5394",
    },
    {
        "id": "MG-2",
        "framework": "Megatron-LM",
        "issue": "#5219",
        "title": "Single-GPU Muon crash fix",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "Blocks Muon on RTX 4090",
        "must": "Avoid Muon until merged",
        "days_open": 30,
        "comments": 5,
        "fix_pr": "Final Review, progressing",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5219",
    },
    {
        "id": "MG-3",
        "framework": "Megatron-LM",
        "issue": "#5179",
        "title": "Muon PyPI placeholder stub",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "4th Muon blocker — can't even install!",
        "must": "Use AdamW only on RTX 4090",
        "days_open": 30,
        "comments": 3,
        "fix_pr": "None",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5179",
    },
    {
        "id": "MG-4",
        "framework": "Megatron-LM",
        "issue": "#5400",
        "title": "GatedDeltaNet in_proj Muon routing → Adam",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "6th Muon blocker — skip_orthogonalization attribute",
        "must": "AdamW only on RTX 4090",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "#5400 (+14/-1, DRAFT)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5400",
    },
    {
        "id": "MG-5",
        "framework": "Megatron-LM",
        "issue": "#5227",
        "title": "Recompute memory leak (autograd references)",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "Gradient growth during backward with recompute",
        "must": "Monitor for fix",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "#5197 (MoE activation free in recompute)",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5227",
    },
    {
        "id": "MG-6",
        "framework": "Megatron-LM",
        "issue": "#5401",
        "title": "MoE z-loss + CUDA graph capture failure",
        "severity": "MEDIUM",
        "status": "OPEN",
        "rtx4090": "z-loss + CUDA graph → padding_mask=None → CPU-to-CUDA during capture",
        "must": "Monitor for merge → affects MoE+z-loss+CUDA graph combo",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "#5401 (keep no-padding token count as Python int)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5401",
    },
    # ── vLLM ──
    {
        "id": "VL-1",
        "framework": "vLLM",
        "issue": "#45972",
        "title": "REVERT: DSV4 cudagraph garbage output",
        "severity": "CRITICAL",
        "status": "MERGED",
        "rtx4090": "cudagraph + DSV4 = correctness regression → enforce_eager!",
        "must": "Use enforce_eager=True on SM89",
        "days_open": 0,
        "comments": 2,
        "fix_pr": "MERGED (revert of #45309)",
        "url": "https://github.com/vllm-project/vllm/pull/45972",
    },
    {
        "id": "VL-2",
        "framework": "vLLM",
        "issue": "#45683",
        "title": "MoE deterministic combine",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "CRITICAL for GRPO MoE stability",
        "must": "Monitor for merge",
        "days_open": 7,
        "comments": 3,
        "fix_pr": "OPEN, 89 additions",
        "url": "https://github.com/vllm-project/vllm/pull/45683",
    },
    {
        "id": "VL-3",
        "framework": "vLLM",
        "issue": "#39096",
        "title": "SM<90 batch invariance UNFIXED",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "Inductor fuses RMSNorm on SM89 → batch-dependent!",
        "must": "Use enforce_eager=True or VLLM_BATCH_INVARIANT=1",
        "days_open": 60,
        "comments": 10,
        "fix_pr": "P9 Inductor Guard (draft ready)",
        "url": "https://github.com/vllm-project/vllm/issues/39096",
    },
    {
        "id": "VL-4",
        "framework": "vLLM",
        "issue": "#45979",
        "title": "3rd DSV4 revert: sparse cache GSM8K 6.75%",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "DSV4 systematic instability continues — enforce_eager MANDATORY",
        "must": "enforce_eager=True for DSV4 models",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "Revert of #45863 (sparse index cache)",
        "url": "https://github.com/vllm-project/vllm/pull/45979",
    },
    # ── verl ──
    {
        "id": "VE-1",
        "framework": "verl",
        "issue": "#6782",
        "title": "LoRA rank=64 breaks EOS",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "MUST rank=32/alpha=64 for vLLM rollout!",
        "must": "Set lora_rank=32, lora_alpha=64",
        "days_open": 3,
        "comments": 3,
        "fix_pr": "None yet",
        "url": "https://github.com/volcengine/verl/pull/6782",
    },
    {
        "id": "VE-2",
        "framework": "verl",
        "issue": "#6468",
        "title": "FSDP2 CPU memory leak (6.3 GiB/step)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "Devastating for 24 GiB GPU! Scales with model size.",
        "must": "Monitor for fix, reduce training steps",
        "days_open": 7,
        "comments": 5,
        "fix_pr": "None yet (suspected DTensor staging buffers)",
        "url": "https://github.com/volcengine/verl/issues/6468",
    },
    {
        "id": "VE-3",
        "framework": "verl",
        "issue": "#6699",
        "title": "detach model_output (4x memory reduction)",
        "severity": "HIGH",
        "status": "PARTIAL",
        "rtx4090": "FSDP fixed, 3 other backends UNFIXED!",
        "must": "Use FSDP backend ONLY",
        "days_open": 0,
        "comments": 8,
        "fix_pr": "MERGED for FSDP, C9 PR draft for 3 others",
        "url": "https://github.com/volcengine/verl/pull/6699",
    },
    # ── rLLM ──
    {
        "id": "RL-1",
        "framework": "rLLM",
        "issue": "#605",
        "title": "GRPO grouping bug (group size=1)",
        "severity": "CRITICAL",
        "status": "BLOCKED",
        "rtx4090": "GRPO COMPLETELY BROKEN! 18+ days, 0 comments!",
        "must": "DO NOT use rLLM for GRPO until fixed",
        "days_open": 18,
        "comments": 0,
        "fix_pr": "1-line fix verified (transform.py:127)",
        "url": "https://github.com/rllm-org/rllm/issues/605",
    },
    {
        "id": "RL-2",
        "framework": "rLLM",
        "issue": "#663",
        "title": "Step.output was None (ALL rewards=0.0)",
        "severity": "CRITICAL",
        "status": "MERGED",
        "rtx4090": "ALL prior training produced ZERO rewards!",
        "must": "Never use pre-June 17 training data",
        "days_open": 0,
        "comments": 2,
        "fix_pr": "MERGED June 17",
        "url": "https://github.com/rllm-org/rllm/pull/663",
    },
    # ── SGLang ──
    {
        "id": "SG-1",
        "framework": "SGLang",
        "issue": "#28582",
        "title": "RCE CVSS 9.8 (LoRA endpoint)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "Unauthenticated LoRA load → RCE! Source verified!",
        "must": "Apply @auth_level or restrict network access",
        "days_open": 1,
        "comments": 0,
        "fix_pr": "None (0 maintainer response)",
        "url": "https://github.com/sgl-project/sglang/pull/28582",
    },
    {
        "id": "SG-2",
        "framework": "SGLang",
        "issue": "#28588",
        "title": "Image decompression bomb guard",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "2nd security issue same week as #28582",
        "must": "Apply pixel-count guard for image inputs",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "OPEN (June 18)",
        "url": "https://github.com/sgl-project/sglang/pull/28588",
    },
    {
        "id": "SG-3",
        "framework": "SGLang",
        "issue": "#27097",
        "title": "multi-LoRA determinism bug (4 factors)",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "LoRA serving non-deterministic → affects GRPO reward",
        "must": "Use --enable-deterministic-inference",
        "days_open": 14,
        "comments": 5,
        "fix_pr": "#28499 partial fix (Factor 2), #28566 sentinel-pad",
        "url": "https://github.com/sgl-project/sglang/issues/27097",
    },
    {
        "id": "SG-4",
        "framework": "SGLang",
        "issue": "#28612",
        "title": "DSV4 C128 state mapping lifecycle fix",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "DSV4 correctness fix — co-authored with shiyu7",
        "must": "Monitor for merge — DSV4 systematic instability",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "Fix for #28591 (DSV4 MTP revert)",
        "url": "https://github.com/sgl-project/sglang/pull/28612",
    },
    {
        "id": "SG-5",
        "framework": "SGLang",
        "issue": "#28618",
        "title": "RFC: SM89/L20 support for DSV4-Flash-FP8",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★ DIRECTLY RELEVANT! SM89 DSV4 path validated on L20 (8xL20 TP=8)",
        "must": "Monitor for merge → opens DSV4-Flash-FP8 on RTX 4090!",
        "days_open": 0,
        "comments": 0,
        "fix_pr": "RFC stage — upstream SM89-compatible DSV4 path",
        "url": "https://github.com/sgl-project/sglang/issues/28618",
    },
    # ── PyTorch ──
    {
        "id": "PT-1",
        "framework": "PyTorch",
        "issue": "#187484",
        "title": "vLLM Inductor breaks on torch 2.13",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "Blocks vLLM torch 2.13 upgrade → stay on 2.12!",
        "must": "DO NOT upgrade to torch 2.13",
        "days_open": 2,
        "comments": 3,
        "fix_pr": "None (#187581 revert NOT accepted)",
        "url": "https://github.com/pytorch/pytorch/issues/187484",
    },
    {
        "id": "PT-2",
        "framework": "PyTorch",
        "issue": "#184119",
        "title": "SM89 fp8→bf16 prologue fusion guard",
        "severity": "HIGH",
        "status": "PROGRESSING",
        "rtx4090": "VALIDATES P9 thesis! jansel pushing CI!",
        "must": "Monitor for merge → validates our contribution",
        "days_open": 30,
        "comments": 10,
        "fix_pr": "5-line choices.py, progressing",
        "url": "https://github.com/pytorch/pytorch/pull/184119",
    },
    # ── vLLM-Ascend ──
    {
        "id": "VA-1",
        "framework": "vLLM-Ascend",
        "issue": "#10684",
        "title": "DSA Hadamard ALL-ZERO after sleep/wake",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "BLOCKER for verl RLHF on Ascend! Same pattern as RouterReplay",
        "must": "Monitor → in-place mutation + buffer transfer exclusion = double failure",
        "days_open": 3,
        "comments": 0,
        "fix_pr": "None yet → Option A: copy before in-place, Option C: regenerate on wake",
        "url": "https://github.com/vllm-project/vllm-ascend/issues/10684",
    },
    {
        "id": "VA-2",
        "framework": "vLLM-Ascend",
        "issue": "#10579",
        "title": "MoE NaN: torch.abs() on row indices → duplication",
        "severity": "HIGH",
        "status": "STALLED",
        "rtx4090": "Any MoE model on Ascend → potential NaN during inference!",
        "must": "Monitor for merge → 1-line fix, 0 reviews",
        "days_open": 5,
        "comments": 0,
        "fix_pr": "1-line: remove torch.abs() before npu_moe_token_unpermute",
        "url": "https://github.com/vllm-project/vllm-ascend/issues/10579",
    },
    {
        "id": "VA-3",
        "framework": "vLLM-Ascend",
        "issue": "#10592",
        "title": "NPUIPC weight transfer engine (+787 lines)",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "verl Ascend integration pathway → weight sync between processes",
        "must": "Monitor for merge → enables verl HYBRID on Ascend NPU",
        "days_open": 5,
        "comments": 2,
        "fix_pr": "New feature → NPU-native IPC for weight transfer",
        "url": "https://github.com/vllm-project/vllm-ascend/issues/10592",
    },
]

# ═══════════════════════════════════════════════════════════════
# DASHBOARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════

SEVERITY_COLORS = {
    "CRITICAL": "\033[91m",  # Red
    "HIGH": "\033[93m",      # Yellow
    "MEDIUM": "\033[94m",    # Blue
}

STATUS_SYMBOLS = {
    "BLOCKED": "[X]",
    "STALLED": "[!]",
    "OPEN": "[ ]",
    "PROGRESSING": "[~]",
    "PARTIAL": "[/]",
    "MERGED": "[V]",
}

RESET = "\033[0m"

def print_full_dashboard(issues, filter_status=None):
    """Print full dashboard with all details."""
    if filter_status:
        issues = [i for i in issues if i["status"] == filter_status]

    # Group by framework
    frameworks = {}
    for issue in issues:
        fw = issue["framework"]
        if fw not in frameworks:
            frameworks[fw] = []
        frameworks[fw].append(issue)

    print("=" * 80)
    print("7-Framework Critical Issues Dashboard — RTX 4090 Consultant")
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Issues: {len(issues)}")
    print("=" * 80)

    # Priority order
    fw_order = ["DeepSpeed", "Megatron-LM", "vLLM", "verl", "rLLM", "SGLang", "PyTorch"]

    for fw in fw_order:
        if fw not in frameworks:
            continue
        fw_issues = frameworks[fw]
        print(f"\n{'─' * 80}")
        print(f"  {fw} ({len(fw_issues)} issues)")
        print(f"{'─' * 80}")

        for issue in fw_issues:
            sev = issue["severity"]
            status = issue["status"]
            color = SEVERITY_COLORS.get(sev, "")
            sym = STATUS_SYMBOLS.get(status, "[?]")

            print(f"  {color}{sym} {issue['id']} | {sev} | {status}{RESET}")
            print(f"      {issue['issue']}: {issue['title']}")
            print(f"      RTX 4090: {issue['rtx4090']}")
            print(f"      MUST: {issue['must']}")
            print(f"      Days open: {issue['days_open']} | Comments: {issue['comments']} | Fix: {issue['fix_pr']}")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    blocked = [i for i in issues if i["status"] == "BLOCKED"]
    stalled = [i for i in issues if i["status"] == "STALLED"]
    critical = [i for i in issues if i["severity"] == "CRITICAL"]
    print(f"  BLOCKED: {len(blocked)} → {', '.join(i['id'] for i in blocked)}")
    print(f"  STALLED: {len(stalled)} → {', '.join(i['id'] for i in stalled)}")
    print(f"  CRITICAL: {len(critical)} → {', '.join(i['id'] for i in critical)}")
    print()

    # RTX 4090 MUST list
    print("=" * 80)
    print("RTX 4090 GRPO TRAINING — MUST DO / MUST AVOID")
    print("=" * 80)
    print("\n  MUST DO:")
    must_do = [i for i in issues if i["status"] != "MERGED"]
    for i in must_do:
        print(f"    {i['id']}: {i['must']}")
    print("\n  MUST AVOID:")
    print("    torch 2.13 (PT-1: Inductor breaks)")
    print("    rLLM GRPO (RL-1: grouping bug → BROKEN)")
    print("    LoRA rank=64 (VE-1: breaks EOS)")
    print("    overlap_comm=True (DS-2: NaN on single GPU)")
    print("    ZeRO-3+LoRA (DS-1: regression on v0.19.2)")
    print("    cudagraph+DSV4 (VL-1: garbage output)")
    print("    torch.compile on SM89 (VL-3: batch-dependent)")
    print()


def print_brief_dashboard(issues):
    """Print brief summary only."""
    print("=" * 60)
    print("7-Framework Critical Issues — Brief Summary")
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    for issue in issues:
        sev = issue["severity"]
        status = issue["status"]
        color = SEVERITY_COLORS.get(sev, "")
        sym = STATUS_SYMBOLS.get(status, "[?]")
        print(f"  {color}{sym} {issue['id']} {sev} {status}{RESET} | {issue['framework']} {issue['issue']}: {issue['title'][:50]}")

    print()
    blocked = len([i for i in issues if i["status"] == "BLOCKED"])
    critical = len([i for i in issues if i["severity"] == "CRITICAL"])
    print(f"  Total: {len(issues)} issues | {critical} CRITICAL | {blocked} BLOCKED")


def main():
    parser = argparse.ArgumentParser(description="7-Framework Critical Issues Dashboard")
    parser.add_argument("--brief", action="store_true", help="Brief summary only")
    parser.add_argument("--filter", choices=["BLOCKED", "STALLED", "OPEN", "PROGRESSING", "MERGED", "PARTIAL"],
                       help="Filter by status")
    args = parser.parse_args()

    if args.brief:
        print_brief_dashboard(CRITICAL_ISSUES)
    else:
        print_full_dashboard(CRITICAL_ISSUES, args.filter)


if __name__ == "__main__":
    main()
