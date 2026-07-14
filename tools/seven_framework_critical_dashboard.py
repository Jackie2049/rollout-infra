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
        "issue": "#8072/#8073",
        "title": "ZeRO-3+PEFT LoRA regression (RESOLVED)",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "Now RESOLVED! #8073 MERGED (June 23). Still: ZeRO-2 recommended over ZeRO-3 for dp=1",
        "must": "Use ZeRO-2 + CPU_Adam only (still best practice)",
        "days_open": 20,
        "comments": 0,
        "fix_pr": "#8073 MERGED (+2/-0)",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8073",
    },
    {
        "id": "DS-2",
        "framework": "DeepSpeed",
        "issue": "#8061",
        "title": "overlap_comm + torch.compile = NaN (multi-stream race)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "MUST overlap_comm=False! 4-point evidence matrix confirmed. Maintainers hwchen2017/cx2009 engaged, production workload confirmed. Same root cause as verl #6794 record_stream (CUDA stream safety)",
        "must": "Set overlap_comm=False for dp=1",
        "days_open": 40,
        "comments": 4,
        "fix_pr": "None (root cause confirmed: multi-stream data race in gradient bucket copy_)",
        "url": "https://github.com/microsoft/DeepSpeed/issues/8061",
    },
    {
        "id": "DS-3",
        "framework": "DeepSpeed",
        "issue": "#8068",
        "title": "gradient_clipping default 0→1.0 (RESOLVED)",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "MERGED June 23! Now defaults to 1.0. Still: ALWAYS set explicitly for GRPO",
        "must": "Set gradient_clipping=1.0 in config (now default, but explicit is safer)",
        "days_open": 14,
        "comments": 2,
        "fix_pr": "MERGED (+2/-0)",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8068",
    },
    {
        "id": "DS-4",
        "framework": "DeepSpeed",
        "issue": "#8058",
        "title": "ZenFlow CPU optimizer (2944→256 MiB) — RESOLVED",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "MERGED July 7! ZenFlow CPU optimizer available. Note: chunked copyback ONLY for Stage 1/2, NOT Stage 3",
        "must": "Use ZeRO-2+ZenFlow for best CPU offload. Still avoid ZeRO-3 on single GPU",
        "days_open": 28,
        "comments": 5,
        "fix_pr": "MERGED (delock reviewed, Antlera addressed all comments)",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8058",
    },
    {
        "id": "DS-5",
        "framework": "DeepSpeed",
        "issue": "#8075",
        "title": "fd leak in deepspeed_io_handle_t::wait() — RESOLVED",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "MERGED June 23! 1-line fix (+1/-1). Long-running GRPO safe now.",
        "must": "Still ensure ulimit -n ≥ 65536 for long GRPO runs",
        "days_open": 3,
        "comments": 1,
        "fix_pr": "MERGED (+1/-1: close(fd) was missing)",
        "url": "https://github.com/microsoft/DeepSpeed/pull/8075",
    },
    {
        "id": "DS-6",
        "framework": "DeepSpeed",
        "issue": "#8104",
        "title": "Q3 Roadmap: AutoEP, DeepCompile, On-Policy Distillation Trainer",
        "severity": "MEDIUM",
        "status": "OPEN",
        "rtx4090": "AutoEP + LoRA viable on RTX 4090. DeepCompile = torch.compile for DeepSpeed. OPD Trainer = GRPO-compatible distillation",
        "must": "Monitor roadmap — AutoEP opens MoE training on single GPU",
        "days_open": 11,
        "comments": 0,
        "fix_pr": "Roadmap RFC",
        "url": "https://github.com/microsoft/DeepSpeed/issues/8104",
    },
    # ── Megatron-LM ──
    {
        "id": "MG-1",
        "framework": "Megatron-LM",
        "issue": "#5394/#5395",
        "title": "ChainedOptimizer Muon/AdamW clipping stalls → fix progressing",
        "severity": "CRITICAL",
        "status": "PROGRESSING",
        "rtx4090": "Muon NOT viable without skip_grad_norm_clip. AdamW ALSO stalls (optimizer-agnostic!)",
        "must": "Wait for #5395 merge. ALL 4 review findings addressed (July 10)!",
        "days_open": 30,
        "comments": 8,
        "fix_pr": "#5395 skip_grad_norm_clip (+15/-1, APPROVED by ShauryaaSharma, waiting NVIDIA CI)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5395",
    },
    {
        "id": "MG-2",
        "framework": "Megatron-LM",
        "issue": "#5387",
        "title": "MFSDPv2 fully_shard — RESOLVED",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "MERGED June 29! Megatron-native FSDP with DBuffer primitives. Gradient accumulation contract for GRPO micro-batching.",
        "must": "Evaluate MFSDPv2 for GRPO training (DBuffer minimizes peak memory)",
        "days_open": 14,
        "comments": 5,
        "fix_pr": "MERGED (+993/-3, approved by shjwudp)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5387",
    },
    {
        "id": "MG-3",
        "framework": "Megatron-LM",
        "issue": "#5219",
        "title": "Single-GPU Muon crash fix",
        "severity": "HIGH",
        "status": "PROGRESSING",
        "rtx4090": "Blocks Muon on RTX 4090",
        "must": "Avoid Muon until merged",
        "days_open": 45,
        "comments": 6,
        "fix_pr": "Final Review, progressing",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5219",
    },
    {
        "id": "MG-4",
        "framework": "Megatron-LM",
        "issue": "#5179",
        "title": "Muon PyPI placeholder stub",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "4th Muon blocker — can't even install!",
        "must": "Use AdamW only on RTX 4090",
        "days_open": 45,
        "comments": 3,
        "fix_pr": "None",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5179",
    },
    {
        "id": "MG-5",
        "framework": "Megatron-LM",
        "issue": "#5401",
        "title": "MoE z-loss + CUDA graph capture failure — RESOLVED",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "MERGED June 23! z-loss + CUDA graph → padding_mask=None → CPU-to-CUDA during capture fixed",
        "must": "z-loss now safe with CUDA graphs",
        "days_open": 3,
        "comments": 1,
        "fix_pr": "MERGED (+6/-1, keep no-padding token count as Python int)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5401",
    },
    {
        "id": "MG-6",
        "framework": "Megatron-LM",
        "issue": "#5227",
        "title": "Recompute memory leak (autograd references)",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "Gradient growth during backward with recompute",
        "must": "Monitor for fix",
        "days_open": 20,
        "comments": 1,
        "fix_pr": "#5197 (MoE activation free in recompute)",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5227",
    },
    {
        "id": "MG-7",
        "framework": "Megatron-LM",
        "issue": "#5400",
        "title": "GatedDeltaNet in_proj Muon routing → Adam",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "6th Muon blocker — skip_orthogonalization attribute",
        "must": "AdamW only on RTX 4090",
        "days_open": 15,
        "comments": 2,
        "fix_pr": "#5400 (+14/-1, DRAFT)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5400",
    },
    {
        "id": "MG-8",
        "framework": "Megatron-LM",
        "issue": "#5396",
        "title": "GDN L2-norm fold (24 GiB savings at 128K)",
        "severity": "MEDIUM",
        "status": "OPEN",
        "rtx4090": "+7/-4 lines, numerically lossless. ~384 MiB savings at 4K for RTX 4090",
        "must": "Monitor for merge — enables longer sequences",
        "days_open": 20,
        "comments": 2,
        "fix_pr": "#5396 (+7/-4, OPEN, updated July 10)",
        "url": "https://github.com/NVIDIA/Megatron-LM/pull/5396",
    },
    {
        "id": "MG-9",
        "framework": "Megatron-LM",
        "issue": "#5749",
        "title": "Expert Parallelism deadlock with Gemma4 (EP=4, DP≥16)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "NEW July 11! EP deadlock on multi-GPU. Single GPU (EP=1) unaffected.",
        "must": "Monitor — EP deadlock affects multi-GPU training only",
        "days_open": 1,
        "comments": 0,
        "fix_pr": "None yet (NEW)",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5749",
    },
    {
        "id": "MG-10",
        "framework": "Megatron-LM",
        "issue": "#5747",
        "title": "CI determinism_perf regression",
        "severity": "MEDIUM",
        "status": "OPEN",
        "rtx4090": "NEW July 10. CI regression, not directly RTX 4090 impacting but signals instability",
        "must": "Monitor",
        "days_open": 2,
        "comments": 0,
        "fix_pr": "None yet (NEW)",
        "url": "https://github.com/NVIDIA/Megatron-LM/issues/5747",
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
    {
        "id": "VL-5",
        "framework": "vLLM",
        "issue": "#46125",
        "title": "REVERT encoder cache fix — RLHF CRITICAL",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "MERGED June 20 (reverted #45093). DANGEROUS for RLHF: reverts cache reset after weight update → stale KV/encoder outputs = SILENT CORRUPTION. Same pattern family as SGLang #28676, #28679, vLLM #44395",
        "must": "★★★★★★★★ MUST reset prefix+encoder cache after EVERY weight update in GRPO loop. Configurable cache reset needed.",
        "days_open": 22,
        "comments": 0,
        "fix_pr": "#45093 (+12/-0, reset cache after weight update) → #46125 REVERTS it!",
        "url": "https://github.com/vllm-project/vllm/pull/46125",
    },
    {
        "id": "VL-6",
        "framework": "vLLM",
        "issue": "#45552",
        "title": "CuMemAllocator sleep/wake stream sync bug",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★ CuMemAllocator sleep/wake missing cuda.synchronize() → CUDART illegal-memory crash! In-flight kernels race cuMemUnmap + cudaMemcpy. 6th weight_reload pattern member.",
        "must": "★★★★★★★★ MUST add cuda.synchronize() before cuMemUnmap in sleep() and after D2H copies in wake_up().",
        "days_open": 20,
        "comments": 2,
        "fix_pr": "None yet (torch.cuda.synchronize() fix in sleep/wake)",
        "url": "https://github.com/vllm-project/vllm/issues/45552",
    },
    {
        "id": "VE-1",
        "framework": "verl",
        "issue": "#6782",
        "title": "LoRA rank=64 breaks EOS",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "MUST rank=32/alpha=64 for vLLM rollout!",
        "must": "Set lora_rank=32, lora_alpha=64",
        "days_open": 22,
        "comments": 5,
        "fix_pr": "None yet (updated July 11, maintainers engaged)",
        "url": "https://github.com/volcengine/verl/pull/6782",
    },
    {
        "id": "VE-2",
        "framework": "verl",
        "issue": "#6468",
        "title": "FSDP2 CPU memory leak (0.6-6.3 GiB/step, MUST use FSDP1!)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "Devastating for 24 GiB GPU! Scales with model size.",
        "must": "Monitor for fix, reduce training steps",
        "days_open": 30,
        "comments": 5,
        "fix_pr": "None yet (suspected DTensor staging buffers, multi-user confirmed)",
        "url": "https://github.com/volcengine/verl/issues/6468",
    },
    {
        "id": "VE-3",
        "framework": "verl",
        "issue": "#6699",
        "title": "detach model_output (4x memory reduction) — RESOLVED",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "VERIFIED 4x memory reduction! Still: only FSDP backend fixed. Automodel/Megatron/TorchTitan have SAME unfixed leak.",
        "must": "Use FSDP backend ONLY for GRPO. C9 PR draft for 3 other backends.",
        "days_open": 14,
        "comments": 8,
        "fix_pr": "MERGED (+22/-6, detach in Actor.forward)",
        "url": "https://github.com/volcengine/verl/pull/6699",
    },
    {
        "id": "VE-4",
        "framework": "verl",
        "issue": "#6794",
        "title": "Delta weight sync (~100x payload reduction, SGLang-only)",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★ Weight sync bandwidth reduction! 2 CRITICAL review issues: record_stream + disk race. SGLang-only rollout.",
        "must": "Monitor for merge. MUST fix record_stream and disk_race before production use",
        "days_open": 10,
        "comments": 4,
        "fix_pr": "#6794 (delta sync e2e, WIP)",
        "url": "https://github.com/volcengine/verl/pull/6794",
    },
    {
        "id": "VE-5",
        "framework": "verl",
        "issue": "#7016",
        "title": "Qwen3-MoE FSDP2 backward failure (grad ckpt crash + SIGSEGV)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★★ CRITICAL for RTX 4090 GRPO! Two failure modes: (A) grad ckpt ON → data-dependent saved-tensor count (router dispatch path), (B) grad ckpt OFF → SIGSEGV no traceback. FSDP1 works, dense models work. ROOT CAUSE: FSDP2 assumes SPMD but MoE router creates different gradient graphs per rank → NCCL reduce-scatter sees mismatched input sizes (PyTorch PR #174862). On dp=1: Failure A still occurs locally, Failure B avoided (single rank).",
        "must": "MUST use FSDP1 (not FSDP2) with Qwen3-MoE. Transformers PR #41580 workaround: consolidate experts into single nn.Parameter. Monitor PyTorch PR #174862 for FSDP2 fix.",
        "days_open": 1,
        "comments": 0,
        "fix_pr": "None yet (NEW July 11)",
        "url": "https://github.com/volcengine/verl/issues/7016",
    },
    {
        "id": "VE-6",
        "framework": "verl",
        "issue": "#7007",
        "title": "Sync trainer SkipManager enablement",
        "severity": "MEDIUM",
        "status": "OPEN",
        "rtx4090": "NEW July 10. Sync trainer optimization. SkipManager enables selective computation skipping. RTX 4090 relevant for training efficiency.",
        "must": "Monitor for merge",
        "days_open": 2,
        "comments": 0,
        "fix_pr": "#7007 (Sync trainer SkipManager enablement)",
        "url": "https://github.com/volcengine/verl/pull/7007",
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
        "days_open": 39,
        "comments": 0,
        "fix_pr": "1-line fix verified (transform.py:127, fork: jackie2049/rllm fix/grpo-configurable-grouping)",
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
    {
        "id": "SG-6",
        "framework": "SGLang",
        "issue": "#28676",
        "title": "MXFP8 MoE shuffle cache CLOBBERED — RESOLVED",
        "severity": "FIXED",
        "status": "MERGED",
        "rtx4090": "MERGED July 1! MXFP8 MoE cache clobbered on RL weight reload → 64x accuracy blowup. Fix: dict.clear() on cache + weight-load funnel call (+28/-2). RTX 4090 MoE GRPO UNLOCKED!",
        "must": "Update SGLang to include this fix before MoE GRPO training",
        "days_open": 14,
        "comments": 3,
        "fix_pr": "MERGED (+28/-2, dict.clear() on cache + weight-load funnel call)",
        "url": "https://github.com/sgl-project/sglang/pull/28676",
    },
    {
        "id": "SG-7",
        "framework": "SGLang",
        "issue": "#28679",
        "title": "GDN intermittent decode degeneracy (worsens over uptime)",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★ Silent corruption in long-running GRPO! Dense model issue (NOT DSV4) but same state lifecycle mismatch pattern. Worsens over uptime, clears on restart.",
        "must": "Restart SGLang server periodically during long GRPO runs. Monitor for fix.",
        "days_open": 15,
        "comments": 2,
        "fix_pr": "None yet — periodic flush (#28695 ReplaySSM) mitigates partially",
        "url": "https://github.com/sgl-project/sglang/issues/28679",
    },
    {
        "id": "SG-8",
        "framework": "SGLang",
        "issue": "#28703",
        "title": "DSA LoRA targets for GLM-5.1/DSv3.2 → GRPO critical",
        "severity": "CRITICAL",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★ GRPO CRITICAL! DSA LoRA targets for GLM-5.1/DSv3.2. e2e verified.",
        "must": "Monitor for merge — enables GRPO on DSA models via SGLang",
        "days_open": 5,
        "comments": 0,
        "fix_pr": "#28703 (DSA LoRA targets, e2e verified)",
        "url": "https://github.com/sgl-project/sglang/pull/28703",
    },
    {
        "id": "SG-9",
        "framework": "SGLang",
        "issue": "#28608",
        "title": "RolloutKV prefix KV pinning for RL (GRPO critical)",
        "severity": "HIGH",
        "status": "OPEN",
        "rtx4090": "★★★★★★★★ Prefix KV pinning for GRPO rollout. Prevents KV cache eviction during training steps. +768/-5.",
        "must": "Monitor for merge — enables stable long-context GRPO rollout",
        "days_open": 12,
        "comments": 2,
        "fix_pr": "#28608 (+768/-5, RolloutKV prefix KV pinning)",
        "url": "https://github.com/sgl-project/sglang/pull/28608",
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
    print("    rLLM GRPO (RL-1: grouping bug → BROKEN, 39+ days stale)")
    print("    LoRA rank=64 (VE-1: breaks EOS on vLLM rollout)")
    print("    overlap_comm=True on single GPU (DS-2: NaN with torch.compile)")
    print("    FSDP2 for GRPO training (VE-2: CPU memory leak 0.6-6.3 GiB/step)")
    print("    FSDP2+Qwen3-MoE (VE-5: backward crash, use FSDP1 instead)")
    print("    cudagraph+DSV4 (VL-1: garbage output → enforce_eager MANDATORY)")
    print("    torch.compile on SM89 (VL-3: batch-dependent fusion)")
    print("    Muon optimizer (MG-1/MG-3/MG-4/MG-7: 6 blockers, AdamW only)")
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
