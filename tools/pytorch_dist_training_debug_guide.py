#!/usr/bin/env python3
"""PyTorch Distributed Training Debugging Guide Tool.

A practical reference for debugging distributed training issues on RTX 4090
and other GPUs. Covers NaN/Inf, OOM, performance, stability, and accuracy
problems with concrete diagnostic commands and fix recommendations.

Modes:
  diagnose  — Given a symptom, diagnose likely root causes and fixes
  checklist — Pre-flight checklist before starting GRPO training
  debug     — Step-by-step debugging workflow for specific issues
  rtx4090   — RTX 4090 specific debugging guide with config templates

Usage:
  python3 pytorch_dist_training_debug_guide.py diagnose <symptom>
  python3 pytorch_dist_training_debug_guide.py checklist
  python3 pytorch_dist_training_debug_guide.py debug <nan|oom|convergence>
  python3 pytorch_dist_training_debug_guide.py rtx4090

Requires: Python 3.10+
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

class Color(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def c(color: Color, text: str) -> str:
    return f"{color.value}{text}{Color.RESET.value}"


def header(title: str, width: int = 80) -> str:
    line = "=" * width
    return f"\n{line}\n{c(Color.BOLD, c(Color.CYAN, title.center(width)))}\n{line}\n"


def section(title: str) -> str:
    return f"\n{c(Color.BOLD, c(Color.BLUE, f'--- {title} ---'))}\n"


def must_do(text: str) -> str:
    return c(Color.RED, c(Color.BOLD, f"[MUST DO] {text}"))


def warning(text: str) -> str:
    return c(Color.YELLOW, c(Color.BOLD, f"[WARNING] {text}"))


def fix(text: str) -> str:
    return c(Color.GREEN, f"[FIX] {text}")


def code_block(text: str) -> str:
    return c(Color.DIM, textwrap.indent(text, "  "))


def bullet(text: str, level: int = 0) -> str:
    indent = "  " * level
    marker = "-" if level == 0 else "*"
    return f"{indent}{marker} {text}"


# ---------------------------------------------------------------------------
# Data: Symptoms, causes, fixes
# ---------------------------------------------------------------------------

class SymptomCategory(Enum):
    NAN_INF = "nan_inf"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    STABILITY = "stability"
    ACCURACY = "accuracy"


@dataclass
class Cause:
    name: str
    probability: str  # "high", "medium", "low"
    description: str
    framework_issue: str = ""  # issue reference like #8061
    diagnostic_cmds: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    must_do_rules: list[str] = field(default_factory=list)


SYMPTOM_DB: dict[str, dict[str, Any]] = {
    # --- NaN/Inf ---
    "training_nan": {
        "category": SymptomCategory.NAN_INF,
        "description": "Training produces NaN values in model weights or activations",
        "causes": [
            Cause(
                name="CUDA stream race condition in DeepSpeed overlap_comm",
                probability="high",
                description="overlap_comm=True with dp=1 creates silent data corruption that manifests as NaN",
                framework_issue="#8061",
                diagnostic_cmds=[
                    "grep overlap_comm your_config.json",
                    "python3 -c 'import torch; print(torch.isnan(your_tensor).any())'",
                    "torch.autograd.set_detect_anomaly(True)",
                ],
                fixes=["Set overlap_comm=False when dp=1"],
                must_do_rules=["overlap_comm=False when dp=1 — NO EXCEPTIONS (#8061)"],
            ),
            Cause(
                name="dtype mismatch (fp32/bf16 mix)",
                probability="medium",
                description="Mixing fp32 and bf16 tensors in the same computation causes NaN",
                framework_issue="#8058",
                diagnostic_cmds=[
                    "python3 -c 'for n,p in model.named_parameters(): print(n, p.dtype)'",
                ],
                fixes=["Ensure consistent dtype across all parameters and buffers"],
                must_do_rules=["All model params MUST be same dtype before training starts"],
            ),
            Cause(
                name="Gradient explosion without clipping",
                probability="medium",
                description="Large gradients overflow bf16 range, producing NaN in weight updates",
                framework_issue="#8068",
                diagnostic_cmds=[
                    "python3 -c 'print(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))'",
                ],
                fixes=["Set clip_grad_norm=1.0 in config"],
                must_do_rules=["clip_grad_norm MUST be 1.0, not 0.0 (#8068)"],
            ),
            Cause(
                name="DSV4 CUDA graph instability",
                probability="medium",
                description="DeepSpeed v4 CUDA graphs cause NaN on consumer GPUs",
                framework_issue="#8061, #28679",
                diagnostic_cmds=[
                    "grep cuda_graph your_config.json",
                    "grep enforce_eager your_config.json",
                ],
                fixes=["Set enforce_eager=True, cuda_graph=False"],
                must_do_rules=["enforce_eager=True MUST be set for DSV4 on RTX 4090"],
            ),
            Cause(
                name="Numerical overflow in bf16 accumulation",
                probability="low",
                description="bf16 has only 8 exponent bits; accumulation can overflow in long sequences",
                framework_issue="",
                diagnostic_cmds=[
                    "python3 -c 'import torch; x=torch.randn(10000,dtype=torch.bfloat16); print(x.sum())'",
                ],
                fixes=["Use fp32 accumulation in loss computation, or reduce sequence length"],
                must_do_rules=["Loss computation MUST use fp32 accumulation"],
            ),
        ],
    },
    "gradient_nan": {
        "category": SymptomCategory.NAN_INF,
        "description": "NaN detected in gradients during backward pass",
        "causes": [
            Cause(
                name="Silent corruption from overlap_comm",
                probability="high",
                description="Data corrupted by overlap_comm race condition propagates to backward pass",
                framework_issue="#8061",
                diagnostic_cmds=[
                    "torch.autograd.set_detect_anomaly(True)",
                    "for n,p in model.named_parameters(): if p.grad is not None: print(n, torch.isnan(p.grad).sum())",
                ],
                fixes=["Set overlap_comm=False when dp=1"],
                must_do_rules=["overlap_comm=False when dp=1 (#8061)"],
            ),
            Cause(
                name="Forward NaN propagating backward",
                probability="medium",
                description="NaN in forward pass activations propagates into gradient computation",
                framework_issue="",
                diagnostic_cmds=[
                    "torch.autograd.detect_anomaly()",
                    "Check intermediate activations for NaN layer-by-layer",
                ],
                fixes=["Debug forward pass first — see NaN debugging workflow"],
                must_do_rules=["Always debug forward NaN before backward NaN"],
            ),
        ],
    },
    "loss_nan": {
        "category": SymptomCategory.NAN_INF,
        "description": "Loss becomes NaN during training",
        "causes": [
            Cause(
                name="Reward/advantage NaN propagation",
                probability="high",
                description="NaN in reward computation propagates through advantage to loss",
                framework_issue="",
                diagnostic_cmds=[
                    "print(f'reward: {reward}, advantage: {advantage}, loss: {loss}')",
                ],
                fixes=["Check reward function for NaN inputs, clamp reward values"],
                must_do_rules=["Reward values MUST be clamped to [-10, 10] range"],
            ),
            Cause(
                name="log(0) or log(negative) in loss",
                probability="medium",
                description="Logarithm of zero or negative probability causes NaN",
                framework_issue="",
                diagnostic_cmds=[
                    "torch.clamp(log_probs, min=-1e8, max=1e8)",
                ],
                fixes=["Clamp log probabilities before loss computation"],
                must_do_rules=["log_probs MUST be clamped before loss computation"],
            ),
        ],
    },
    "output_nan": {
        "category": SymptomCategory.NAN_INF,
        "description": "Model outputs contain NaN values at inference",
        "causes": [
            Cause(
                name="Corrupted weights from training NaN",
                probability="high",
                description="Weights already contain NaN from training — check with weight checksum",
                framework_issue="#8061, #8058",
                diagnostic_cmds=[
                    "for n,p in model.named_parameters(): if torch.isnan(p).any(): print(f'NaN in {n}')",
                ],
                fixes=["Restart training with overlap_comm=False, enforce_eager=True"],
                must_do_rules=["Run weight checksum after every 100 steps"],
            ),
            Cause(
                name="KV cache corruption on weight reload",
                probability="medium",
                description="Weight reload without resetting KV cache causes stale/corrupted cache",
                framework_issue="#46125, #28676",
                diagnostic_cmds=[
                    "Check if model reload resets KV cache and encoder cache",
                ],
                fixes=["Reset KV cache and encoder cache after weight reload"],
                must_do_rules=["MUST reset KV cache + encoder cache after weight reload (#46125)"],
            ),
        ],
    },
    # --- Memory ---
    "oom": {
        "category": SymptomCategory.MEMORY,
        "description": "Out of memory error on GPU",
        "causes": [
            Cause(
                name="Full fine-tuning of 8B model exceeds 24GB VRAM",
                probability="high",
                description="8B model full fine-tuning needs ~48GB; RTX 4090 has only 24GB",
                framework_issue="",
                diagnostic_cmds=[
                    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv",
                    "python3 -c 'params=sum(p.numel() for p in model.parameters()); print(f\"{params/1e9:.1f}B params, ~{params*2*4/1e9:.1f} GiB fp32\")'",
                ],
                fixes=["Use LoRA + bypass_mode + FSDP1 (peak 22.9 GiB for 8B)"],
                must_do_rules=["8B on RTX 4090 MUST use LoRA+bypass+FSDP1"],
            ),
            Cause(
                name="DeepSpeed ZeRO-2 memory higher than expected",
                probability="medium",
                description="ZeRO-2 shards optimizer but keeps full gradients; peak higher than FSDP1",
                framework_issue="",
                diagnostic_cmds=[
                    "python3 -c 'from deepspeed.utils import get_global_norm; print(get_global_norm(group))'",
                    "nvidia-smi dmon -s m -d 1",
                ],
                fixes=["Use FSDP1 instead of ZeRO-2 for RTX 4090 single-node"],
                must_do_rules=["FSDP1 preferred over ZeRO-2 for single-node RTX 4090"],
            ),
            Cause(
                name="CUDA graphs consume extra memory",
                probability="medium",
                description="CUDA graphs capture kernel sequences, allocating extra memory for replay",
                framework_issue="",
                diagnostic_cmds=[
                    "torch.cuda.memory_allocated() / 1e9",
                    "torch.cuda.memory_reserved() / 1e9",
                ],
                fixes=["Set cuda_graph=False, enforce_eager=True"],
                must_do_rules=["cuda_graph=False on RTX 4090 with DSV4"],
            ),
        ],
    },
    "host_ram_growth": {
        "category": SymptomCategory.MEMORY,
        "description": "Host RAM usage grows continuously during training",
        "causes": [
            Cause(
                name="FSDP2 memory leak",
                probability="high",
                description="FSDP2 has known host RAM leak; memory grows without bound",
                framework_issue="#6468",
                diagnostic_cmds=[
                    "ps -o rss= -p $(pgrep python) | awk '{print $1/1024 \" MiB\"}'",
                    "Watch RSS over multiple training steps",
                ],
                fixes=["Use FSDP1 instead of FSDP2"],
                must_do_rules=["Do NOT use FSDP2 — use FSDP1 (#6468)"],
            ),
            Cause(
                name="Gradient accumulation buffers not freed",
                probability="medium",
                description="Gradient accumulation keeps extra buffers in host RAM",
                framework_issue="",
                diagnostic_cmds=[
                    "Check gradient_accumulation_steps config",
                    "Monitor /proc/meminfo MemAvailable over time",
                ],
                fixes=["Reduce gradient_accumulation_steps, or use FSDP1 with proper sharding"],
                must_do_rules=[],
            ),
        ],
    },
    "gpu_memory_spike": {
        "category": SymptomCategory.MEMORY,
        "description": "GPU memory spikes suddenly during training",
        "causes": [
            Cause(
                name="Unsharded full param gather in FSDP",
                probability="high",
                description="FSDP1 gathers full parameters for forward/backward, causing memory spikes",
                framework_issue="",
                diagnostic_cmds=[
                    "torch.cuda.memory_allocated() before and after forward pass",
                ],
                fixes=["Use gradient_checkpointing, reduce batch_size, increase micro_batch_size"],
                must_do_rules=["Gradient checkpointing MUST be enabled for 8B models"],
            ),
            Cause(
                name="KV cache allocation during rollout",
                probability="medium",
                description="Rollout generation allocates large KV cache that spikes memory",
                framework_issue="",
                diagnostic_cmds=[
                    "Profile rollout phase memory separately",
                ],
                fixes=["Use paged_attention, limit max_seq_len during rollout"],
                must_do_rules=["Limit rollout max_seq_len to avoid OOM"],
            ),
        ],
    },
    # --- Performance ---
    "slow_training": {
        "category": SymptomCategory.PERFORMANCE,
        "description": "Training throughput is significantly lower than expected",
        "causes": [
            Cause(
                name="NCCL configuration for consumer GPU",
                probability="high",
                description="Default NCCL settings assume InfiniBand; RTX 4090 uses PCIe/NVLink",
                framework_issue="",
                diagnostic_cmds=[
                    "python3 -c 'import torch.distributed as dist; print(dist.get_backend())'",
                    "nccl-net=Socket for PCIe-only systems",
                    "export NCCL_DEBUG=INFO",
                ],
                fixes=["Set NCCL_NET=Socket, NCCL_SOCKET_IFNAME=eth0 for PCIe systems"],
                must_do_rules=["NCCL_NET MUST be Socket for PCIe-only RTX 4090"],
            ),
            Cause(
                name="Enforce eager disables CUDA graphs",
                probability="medium",
                description="enforce_eager=True is needed for stability but costs ~10-15% throughput",
                framework_issue="",
                diagnostic_cmds=[
                    "Compare timing with enforce_eager=True vs False (only test, not production)",
                ],
                fixes=["Accept ~10-15% throughput loss for stability; DO NOT disable enforce_eager"],
                must_do_rules=["enforce_eager=True MUST stay — stability over throughput"],
            ),
        ],
    },
    "slow_convergence": {
        "category": SymptomCategory.PERFORMANCE,
        "description": "Training converges much slower than expected",
        "causes": [
            Cause(
                name="Singleton GRPO degenerates to REINFORCE",
                probability="high",
                description="group_size=1 makes GRPO identical to REINFORCE, losing advantage normalization",
                framework_issue="#605",
                diagnostic_cmds=[
                    "grep group_size your_config",
                    "Verify advantage computation has variance",
                ],
                fixes=["Set group_size >= 4 (minimum 4 for meaningful advantage)"],
                must_do_rules=["group_size MUST be >= 4 (#605). gs=1 = REINFORCE!"],
            ),
            Cause(
                name="Learning rate too high or too low",
                probability="medium",
                description="LR outside optimal range for LoRA GRPO on 8B models",
                framework_issue="",
                diagnostic_cmds=[
                    "print current LR: optimizer.param_groups[0]['lr']",
                ],
                fixes=["Use LR ~1e-6 for LoRA GRPO; use cosine schedule"],
                must_do_rules=["LoRA GRPO LR MUST be ~1e-6 with cosine decay"],
            ),
        ],
    },
    "low_throughput": {
        "category": SymptomCategory.PERFORMANCE,
        "description": "Throughput (tokens/sec or samples/sec) is low",
        "causes": [
            Cause(
                name="Rollout phase bottleneck",
                probability="high",
                description="Generation phase is slower than training phase, creating imbalance",
                framework_issue="",
                diagnostic_cmds=[
                    "Time rollout vs training separately",
                    "Profile with torch.profiler",
                ],
                fixes=["Use vllm/sglang for fast rollout, tune max_new_tokens"],
                must_do_rules=["Use vllm or sglang engine for rollout generation"],
            ),
            Cause(
                name="Overlap of comm/compute not working",
                probability="medium",
                description="overlap_comm=False (required for stability) loses ~5% throughput",
                framework_issue="#8061",
                diagnostic_cmds=[
                    "Profile communication vs compute ratio",
                ],
                fixes=["Accept small throughput loss; DO NOT enable overlap_comm with dp=1"],
                must_do_rules=[],
            ),
        ],
    },
    # --- Stability ---
    "intermittent_crashes": {
        "category": SymptomCategory.STABILITY,
        "description": "Training crashes intermittently with no consistent error message",
        "causes": [
            Cause(
                name="File descriptor exhaustion",
                probability="high",
                description="DeepSpeed/vllm opens many file descriptors; default ulimit is too low",
                framework_issue="#8075",
                diagnostic_cmds=[
                    "ulimit -n",
                    "ls /proc/self/fd | wc -l",
                ],
                fixes=["Set ulimit -n 65535 before training"],
                must_do_rules=["ulimit -n MUST be >= 65535 (#8075)"],
            ),
            Cause(
                name="CUDA stream race condition",
                probability="medium",
                description="overlap_comm creates race conditions that crash intermittently",
                framework_issue="#8061",
                diagnostic_cmds=[
                    "grep overlap_comm config",
                    "export NCCL_DEBUG=INFO",
                ],
                fixes=["Set overlap_comm=False"],
                must_do_rules=["overlap_comm=False when dp=1"],
            ),
        ],
    },
    "random_errors": {
        "category": SymptomCategory.STABILITY,
        "description": "Random errors during training: NCCL timeouts, CUDA errors, etc.",
        "causes": [
            Cause(
                name="NCCL timeout from slow rank",
                probability="high",
                description="One rank slower than others causes NCCL collective timeout",
                framework_issue="",
                diagnostic_cmds=[
                    "export NCCL_DEBUG=INFO",
                    "export NCCL_DEBUG_SUBSYS=ALL",
                    "Check if all ranks have same GPU and config",
                ],
                fixes=["Increase NCCL_TIMEOUT, ensure all ranks have identical hardware"],
                must_do_rules=["All ranks MUST have identical GPU model and config"],
            ),
            Cause(
                name="Weight reload inconsistency across ranks",
                probability="medium",
                description="Weight reload not synchronized across ranks causes mismatch",
                framework_issue="#46125",
                diagnostic_cmds=[
                    "Check weight checksum across all ranks",
                ],
                fixes=["Synchronize weight reload across all ranks; reset caches"],
                must_do_rules=["Weight reload MUST be synchronized across all ranks"],
            ),
        ],
    },
    "fd_exhaustion": {
        "category": SymptomCategory.STABILITY,
        "description": "Process runs out of file descriptors",
        "causes": [
            Cause(
                name="Default ulimit too low",
                probability="high",
                description="macOS/Linux default ulimit is 256/1024; DeepSpeed needs 65535+",
                framework_issue="#8075",
                diagnostic_cmds=[
                    "ulimit -n",
                    "ls /proc/self/fd | wc -l",
                ],
                fixes=["ulimit -n 65535; add to ~/.bashrc or training script"],
                must_do_rules=["ulimit -n MUST be 65535+ (#8075)"],
            ),
        ],
    },
    # --- Accuracy ---
    "reward_zero": {
        "category": SymptomCategory.ACCURACY,
        "description": "Reward function returns 0 for all samples",
        "causes": [
            Cause(
                name="Reward function not matching format",
                probability="high",
                description="Reward function expects specific format (e.g., <answer>42</answer>) but model outputs differently",
                framework_issue="",
                diagnostic_cmds=[
                    "Print raw model output and expected format",
                    "Test reward function on known correct/incorrect answers",
                ],
                fixes=["Debug reward function separately; verify format matching"],
                must_do_rules=["Test reward function on golden examples before training"],
            ),
            Cause(
                name="Singleton GRPO — all rewards same",
                probability="medium",
                description="With gs=1, all samples get same reward so advantage=0, reward_effective=0",
                framework_issue="#605",
                diagnostic_cmds=[
                    "Verify reward variance across group",
                ],
                fixes=["Set group_size >= 4"],
                must_do_rules=["group_size >= 4 (#605)"],
            ),
        ],
    },
    "advantage_zero": {
        "category": SymptomCategory.ACCURACY,
        "description": "Advantage is always 0, making GRPO degenerate",
        "causes": [
            Cause(
                name="group_size=1 eliminates advantage normalization",
                probability="high",
                description="GRPO advantage = (reward - mean(rewards_group)) / std(rewards_group); gs=1 means std=0",
                framework_issue="#605",
                diagnostic_cmds=[
                    "Check group_size config",
                    "Print advantage values to verify they are non-zero",
                ],
                fixes=["Set group_size >= 4"],
                must_do_rules=["group_size MUST be >= 4. gs=1 means advantage=0 (#605)"],
            ),
            Cause(
                name="All rewards identical within group",
                probability="medium",
                description="If all gs samples get same reward, std=0, advantage=0",
                framework_issue="",
                diagnostic_cmds=[
                    "Print reward values within each group",
                ],
                fixes=["Increase group_size or improve reward function discrimination"],
                must_do_rules=["Reward function MUST produce variation within groups"],
            ),
        ],
    },
    "training_not_improving": {
        "category": SymptomCategory.ACCURACY,
        "description": "Training metrics not improving over steps",
        "causes": [
            Cause(
                name="GRPO degenerated to REINFORCE (gs=1)",
                probability="high",
                description="group_size=1 eliminates advantage normalization, making GRPO=REINFORCE",
                framework_issue="#605",
                diagnostic_cmds=[
                    "Verify group_size >= 4",
                    "Print advantage distribution statistics",
                ],
                fixes=["Set group_size >= 4"],
                must_do_rules=["gs >= 4 is MANDATORY for GRPO (#605)"],
            ),
            Cause(
                name="Learning rate schedule issue",
                probability="medium",
                description="Warmup too long, or LR already at minimum from scheduler",
                framework_issue="",
                diagnostic_cmds=[
                    "optimizer.param_groups[0]['lr'] at each step",
                ],
                fixes=["Use warmup_ratio=0.1, cosine schedule, LR=1e-6"],
                must_do_rules=["LR MUST be ~1e-6 for LoRA GRPO with cosine decay"],
            ),
            Cause(
                name="Gradient clipping too aggressive",
                probability="medium",
                description="clip_grad_norm too small (or 0.0 which disables clipping, allowing explosion)",
                framework_issue="#8068",
                diagnostic_cmds=[
                    "Print grad_norm before and after clipping",
                ],
                fixes=["Set clip_grad_norm=1.0"],
                must_do_rules=["clip_grad_norm MUST be 1.0 (#8068)"],
            ),
        ],
    },
}

# Map symptom aliases
SYMPTOM_ALIASES: dict[str, str] = {
    "nan": "training_nan",
    "grad_nan": "gradient_nan",
    "loss_nan": "loss_nan",
    "out_nan": "output_nan",
    "oom": "oom",
    "ram": "host_ram_growth",
    "ram_leak": "host_ram_growth",
    "mem_spike": "gpu_memory_spike",
    "slow": "slow_training",
    "convergence": "slow_convergence",
    "throughput": "low_throughput",
    "crash": "intermittent_crashes",
    "random": "random_errors",
    "fd": "fd_exhaustion",
    "ulimit": "fd_exhaustion",
    "reward0": "reward_zero",
    "advantage0": "advantage_zero",
    "not_improving": "training_not_improving",
    "no_improve": "training_not_improving",
}


# ---------------------------------------------------------------------------
# Mode 1: diagnose
# ---------------------------------------------------------------------------

def run_diagnose(symptom: str) -> None:
    key = SYMPTOM_ALIASES.get(symptom, symptom)
    info = SYMPTOM_DB.get(key)
    if info is None:
        print(c(Color.RED, f"Unknown symptom: '{symptom}'"))
        print(c(Color.YELLOW, "Available symptoms:"))
        for k, v in SYMPTOM_DB.items():
            cat = v["category"].value
            print(bullet(f"{k} ({cat}): {v['description']}"))
        print(c(Color.YELLOW, "\nAliases: " + ", ".join(SYMPTOM_ALIASES.keys())))
        return

    category = info["category"]
    print(header(f"DIAGNOSIS: {key}"))
    print(f"Category: {c(Color.BOLD, category.value)}")
    print(f"Description: {info['description']}")
    print(section("Likely Root Causes (ranked by probability)"))

    for i, cause in enumerate(info["causes"], 1):
        prob_color = {
            "high": Color.RED,
            "medium": Color.YELLOW,
            "low": Color.DIM,
        }.get(cause.probability, Color.RESET)
        prob_tag = c(prob_color, c(Color.BOLD, f"[{cause.probability.upper()}]"))
        fw = f" ({c(Color.MAGENTA, cause.framework_issue)})" if cause.framework_issue else ""

        print(f"\n  {i}. {prob_tag} {cause.name}{fw}")
        print(f"     {cause.description}")

        if cause.diagnostic_cmds:
            print(f"     {c(Color.BOLD, 'Diagnostic commands:')}")
            for cmd in cause.diagnostic_cmds:
                print(code_block(cmd))

        if cause.fixes:
            print(f"     {c(Color.BOLD, 'Recommended fixes:')}")
            for fx in cause.fixes:
                print(f"     {fix(fx)}")

        if cause.must_do_rules:
            print(f"     {c(Color.BOLD, 'MUST DO rules:')}")
            for rule in cause.must_do_rules:
                print(f"     {must_do(rule)}")

    print(section("Quick diagnostic — run all of these first"))
    all_cmds = []
    for cause in info["causes"]:
        all_cmds.extend(cause.diagnostic_cmds)
    for cmd in all_cmds:
        print(code_block(cmd))

    all_must = []
    for cause in info["causes"]:
        all_must.extend(cause.must_do_rules)
    if all_must:
        print(section("ALL MUST DO RULES for this symptom"))
        for rule in all_must:
            print(must_do(rule))


# ---------------------------------------------------------------------------
# Mode 2: checklist
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    detail: str = ""
    fix_recommendation: str = ""


def run_checklist() -> None:
    print(header("GRPO TRAINING PRE-FLIGHT CHECKLIST"))

    results: list[CheckResult] = []

    # --- System checks ---
    print(section("System Checks"))

    # ulimit
    try:
        ulimit_val = int(subprocess.run(["ulimit", "-n"], capture_output=True, text=True).stdout.strip())
    except Exception:
        ulimit_val = -1

    if ulimit_val >= 65535:
        results.append(CheckResult("ulimit -n", "PASS", f"{ulimit_val}", ""))
    elif ulimit_val > 0:
        results.append(CheckResult("ulimit -n", "FAIL", f"{ulimit_val} (need 65535+)", "ulimit -n 65535 (#8075)"))
    else:
        results.append(CheckResult("ulimit -n", "FAIL", "could not determine", "ulimit -n 65535"))

    # GPU available
    gpu_count = 0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        lines = out.stdout.strip().split("\n")
        gpu_count = len([l for l in lines if l.strip()])
        gpu_info = lines[0] if lines else "N/A"
    except Exception:
        gpu_info = "N/A"

    if gpu_count > 0:
        results.append(CheckResult("GPU available", "PASS", f"{gpu_count} GPUs: {gpu_info}", ""))
    else:
        results.append(CheckResult("GPU available", "FAIL", "No GPUs detected", "Install NVIDIA drivers"))

    # CUDA version
    cuda_ver = "N/A"
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        for line in out.stdout.split("\n"):
            if "release" in line:
                cuda_ver = line.strip().split("release")[1].strip().split(",")[0].strip()
    except Exception:
        pass
    if cuda_ver != "N/A":
        results.append(CheckResult("CUDA version", "PASS", cuda_ver, ""))
    else:
        results.append(CheckResult("CUDA version", "WARN", "nvcc not found", "Install CUDA toolkit or use torch.cuda.is_available()"))

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        results.append(CheckResult("Python version", "PASS", py_ver, ""))
    else:
        results.append(CheckResult("Python version", "FAIL", py_ver, "Upgrade to Python 3.10+"))

    # --- Package checks ---
    print(section("Package Checks"))

    pkg_checks = {
        "torch": ("PyTorch", True),
        "vllm": ("vLLM", False),
        "sglang": ("SGLang", False),
        "deepspeed": ("DeepSpeed", False),
        "trl": ("TRL", False),
        "verl": ("VERL", False),
    }
    for pkg, (display, required) in pkg_checks.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            results.append(CheckResult(display, "PASS", f"v{ver}", ""))
        except ImportError:
            status = "FAIL" if required else "WARN"
            msg = "install with pip" if required else "optional — install if needed"
            results.append(CheckResult(display, status, "not installed", msg))

    # version compatibility warnings
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        torch_cuda = torch.version.cuda or "N/A"
        if not cuda_avail:
            results.append(CheckResult("torch.cuda", "FAIL", "not available", "PyTorch CUDA build required"))
        else:
            results.append(CheckResult("torch.cuda", "PASS", f"CUDA {torch_cuda}", ""))

        # bf16 support
        bf16_ok = torch.cuda.is_bf16_supported()
        if bf16_ok:
            results.append(CheckResult("bf16 support", "PASS", "available", ""))
        else:
            results.append(CheckResult("bf16 support", "FAIL", "NOT available", "Use Ampere+ GPU (RTX 3090/4090) for bf16"))
    except ImportError:
        results.append(CheckResult("torch.cuda", "SKIP", "torch not installed", ""))

    # --- Config checks (static guidance since we can't read arbitrary configs) ---
    print(section("Config Checks (Verify in your training config)"))

    config_must_checks = [
        ("LoRA rank", "lora_rank >= 8 for 8B models; use LoRA for RTX 4090 24GB", "PASS if LoRA; FAIL if full finetune"),
        ("bypass_mode", "bypass_mode=True for actor+ref model sharing (saves ~50% memory)", "Must verify in config"),
        ("zero_stage", "zero_stage=3 for multi-GPU; FSDP1 for single-node", "Verify zero_stage matches hardware"),
        ("overlap_comm", "overlap_comm=False when dp=1 (#8061)", "MUST be False for dp=1"),
        ("gradient_clipping", "clip_grad_norm=1.0 (#8068)", "MUST be 1.0, NOT 0.0"),
        ("group_size", "group_size >= 4 (#605)", "MUST be >= 4"),
        ("enforce_eager", "enforce_eager=True for DSV4 (#8061)", "MUST be True"),
        ("cuda_graph", "cuda_graph=False for DSV4 stability", "MUST be False"),
    ]
    for name, desc, detail in config_must_checks:
        results.append(CheckResult(name, "WARN", detail, desc))

    # --- Memory checks ---
    print(section("Memory Checks"))

    # Known RTX 4090 memory budgets
    mem_budgets = {
        "8B LoRA+bypass+FSDP1": "22.9 GiB peak",
        "8B LoRA+bypass+ZeRO3": "~23.5 GiB peak",
        "8B full finetune": "~48 GiB (IMPOSSIBLE on 4090)",
        "1.5B full finetune": "~12 GiB",
        "1.5B LoRA+FSDP1": "~8 GiB",
    }
    print(c(Color.BOLD, "Estimated peak memory on RTX 4090 (24 GiB VRAM):"))
    for config, peak in mem_budgets.items():
        feasible = "OK" if "IMPOSSIBLE" not in peak else c(Color.RED, "FAIL")
        print(bullet(f"{config}: {peak} [{feasible}]"))

    results.append(CheckResult("Memory budget", "WARN", "Verify against your GPU VRAM", "Use LoRA+bypass+FSDP1 for 8B on RTX 4090"))

    # Host RAM
    host_ram_gb = 0
    try:
        if platform.system() == "Linux":
            out = subprocess.run(["free", "-g"], capture_output=True, text=True)
            for line in out.stdout.split("\n"):
                if "Mem:" in line:
                    host_ram_gb = int(line.split()[1])
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            host_ram_gb = int(out.stdout.strip()) // (1024 ** 3)
    except Exception:
        pass

    if host_ram_gb >= 64:
        results.append(CheckResult("Host RAM", "PASS", f"{host_ram_gb} GiB", ""))
    elif host_ram_gb >= 32:
        results.append(CheckResult("Host RAM", "WARN", f"{host_ram_gb} GiB (recommend 64+)", "Upgrade RAM if possible"))
    else:
        results.append(CheckResult("Host RAM", "FAIL", f"{host_ram_gb} GiB (need 32+)", "Upgrade host RAM to 64 GiB minimum"))

    # --- Network checks ---
    print(section("Network Checks"))

    nccl_env_vars = {
        "NCCL_NET": "Should be 'Socket' for PCIe-only RTX 4090 systems",
        "NCCL_SOCKET_IFNAME": "Set to your Ethernet interface (e.g., eth0, en0)",
        "NCCL_IB_DISABLE": "Should be '1' if no InfiniBand available",
        "NCCL_DEBUG": "Set to 'INFO' for debugging; 'WARN' for production",
    }
    for var, desc in nccl_env_vars.items():
        val = os.environ.get(var, "NOT SET")
        results.append(CheckResult(var, "WARN", val, desc))

    # --- DSV4 checks ---
    print(section("DeepSpeed V4 Specific Checks"))

    dsv4_checks = [
        ("enforce_eager", "MUST be True — prevents 11 failures across 4 frameworks", "enforce_eager=True"),
        ("cuda_graph", "MUST be False — CUDA graphs cause NaN on consumer GPUs", "cuda_graph=False"),
        ("overlap_comm", "MUST be False when dp=1 — race condition (#8061)", "overlap_comm=False"),
        ("fp8_quantization", "WARN: fp8 on SM89 (Ada) has known issues; use bf16", "Avoid fp8 on RTX 4090"),
    ]
    for name, desc, fix_rec in dsv4_checks:
        results.append(CheckResult(name, "FAIL", desc, fix_rec))

    # --- Summary ---
    print(section("CHECKLIST SUMMARY"))

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARN")

    print(f"  {c(Color.GREEN, f'PASS: {pass_count}')}")
    print(f"  {c(Color.RED, f'FAIL: {fail_count}')}")
    print(f"  {c(Color.YELLOW, f'WARN: {warn_count}')}")

    if fail_count > 0:
        print(c(Color.RED, c(Color.BOLD, "\n  DO NOT START TRAINING until all FAIL items are resolved!")))
    else:
        print(c(Color.GREEN, "\n  All critical checks passed. Review WARN items before proceeding."))

    # Full detail table
    print(section("Detailed Results"))
    for r in results:
        color = {
            "PASS": Color.GREEN,
            "FAIL": Color.RED,
            "WARN": Color.YELLOW,
            "SKIP": Color.DIM,
        }.get(r.status, Color.RESET)
        status_tag = c(color, c(Color.BOLD, f"[{r.status}]"))
        print(f"  {status_tag} {r.name}: {r.detail}")
        if r.fix_recommendation:
            print(f"    {fix(r.fix_recommendation)}")


# ---------------------------------------------------------------------------
# Mode 3: debug
# ---------------------------------------------------------------------------

def run_debug_nan() -> None:
    print(header("NaN DEBUGGING WORKFLOW: 5 Steps"))
    print(c(Color.BOLD, "detect -> isolate -> reproduce -> root cause -> fix"))

    print(section("Step 1: DETECT — Find where NaN appears"))
    print("""
Two detection approaches:

  A) torch.autograd.set_detect_anomaly(True)
     - Runs forward+backward in debug mode
     - Throws RuntimeError at the operation that produced NaN
     - Slows training ~3x but pinpoints exact op

  B) NanDetectMode (verl-specific)
     - Forward mode: check every intermediate activation
     - Backward mode: check every gradient
     - More precise than set_detect_anomaly
""")
    print(code_block("import torch"))
    print(code_block("torch.autograd.set_detect_anomaly(True)"))
    print(code_block("# Run one training step — will throw at NaN-producing op"))
    print(code_block("#"))
    print(code_block("# Or for verl/DeepSpeed:"))
    print(code_block("# config.nan_detect_mode = 'forward'  # or 'backward'"))
    print(code_block("# This adds NaN checks after every layer forward/backward"))

    print(must_do("Always use detect_anomaly or NanDetectMode before trying ANY fix"))

    print(section("Step 2: ISOLATE — Narrow down which layer/component"))
    print("""
Layer-by-layer NaN check:
""")
    print(code_block("""for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in parameter: {name}")
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN in gradient: {name}")"""))
    print(code_block(""))
    print(code_block("""# Check intermediate activations:
with torch.no_grad():
    for name, module in model.named_modules():
        if hasattr(module, '_saved_output'):
            out = module._saved_output
            if torch.isnan(out).any():
                print(f"NaN after module: {name}")"""))

    print(section("Step 3: REPRODUCE — Create minimal reproducer"))
    print("""
Strip config to minimum to reproduce NaN:
""")
    print(must_do("overlap_comm=False (required for dp=1, #8061)"))
    print(must_do("enforce_eager=True (required for DSV4)"))
    print(must_do("cuda_graph=False"))
    print(code_block("""
# Minimal reproduction config:
{
    "overlap_comm": false,
    "enforce_eager": true,
    "cuda_graph": false,
    "gradient_clipping": 1.0,
    "zero_stage": 3,
    "bf16": { "enabled": true }
}

# If NaN still appears with this minimal config, the issue is in:
#   - Model architecture (dtype mismatch)
#   - Data (NaN in input)
#   - Loss computation (log(0), overflow)
"""))

    print(section("Step 4: ROOT CAUSE — Match to taxonomy"))
    print("""
NaN Root Cause Taxonomy:

  1. CUDA stream race condition (#8061)
     - overlap_comm=True with dp=1
     - Silent corruption that appears as NaN
     - Fix: overlap_comm=False

  2. dtype mismatch (#8058)
     - Mixed fp32/bf16 tensors in same computation
     - Some buffers not converted to bf16
     - Fix: ensure consistent dtype

  3. bf16 overflow (#8068)
     - Large gradient values exceed bf16 range (max ~3.4e38)
     - Without gradient clipping, updates overflow
     - Fix: clip_grad_norm=1.0

  4. CUDA graph instability
     - CUDA graphs capture operations that later produce different shapes/values
     - Consumer GPUs more susceptible
     - Fix: enforce_eager=True, cuda_graph=False

  5. Numerical underflow
     - log(0), log(negative) in loss computation
     - Division by zero in advantage normalization (gs=1)
     - Fix: clamp values, ensure gs>=4

  6. Weight corruption
     - Silent corruption accumulates in weights over time
     - Detect with weight checksum every 100 steps
     - Fix: overlap_comm=False + NanDetectMode
""")

    print(section("Step 5: FIX — Apply corrections with MUST DO rules"))
    print(must_do("overlap_comm=False when dp=1 (#8061)"))
    print(must_do("clip_grad_norm=1.0, NOT 0.0 (#8068)"))
    print(must_do("enforce_eager=True for DSV4 on RTX 4090"))
    print(must_do("cuda_graph=False for DSV4 on RTX 4090"))
    print(must_do("group_size >= 4 (#605)"))
    print(must_do("All params MUST be same dtype before training"))
    print(must_do("Loss MUST use fp32 accumulation"))
    print(must_do("log_probs MUST be clamped before loss computation"))
    print(must_do("Weight checksum every 100 steps (#8061, #8058, #28679)"))
    print(must_do("Reset KV cache + encoder cache after weight reload (#46125)"))

    print(section("Verification — Confirm fix worked"))
    print(code_block("""
# After applying fixes, verify:
# 1. Run 500 steps with detect_anomaly=True — should complete without error
# 2. Check weight checksums at step 100, 200, 500
# 3. Monitor loss curve — should be smooth, no spikes
# 4. Check gradient norms — should be <10.0 consistently
"""))


def run_debug_oom() -> None:
    print(header("OOM DEBUGGING WORKFLOW: 5 Steps"))
    print(c(Color.BOLD, "measure -> profile -> reduce -> offload -> scale"))

    print(section("Step 1: MEASURE — Quantify current memory usage"))
    print(code_block("""
# GPU memory monitoring
nvidia-smi dmon -s m -d 1  # continuous GPU memory monitor

# Python memory tracking
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GiB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GiB")
print(f"Max alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GiB")

# Peak memory during training step
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
print(f"Peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GiB")
"""))

    print(section("Step 2: PROFILE — Identify memory hotspots"))
    print(code_block("""
# PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # run one training step
    ...

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

# Memory snapshot (PyTorch 2.1+)
torch.cuda.memory._snapshot()
# Save and visualize with https://pytorch.org/memory_snapshot
"""))

    print(section("Step 3: REDUCE — Lower memory footprint"))
    print("""
Memory reduction strategies (ordered by impact):

  1. Use LoRA (rank 8-16) instead of full finetune
     - Saves ~80% param memory for 8B model
     - Peak 22.9 GiB vs ~48 GiB full finetune

  2. Use bypass_mode=True (actor+ref model sharing)
     - Saves ~50% memory by sharing base weights
     - Only LoRA adapter weights are separate

  3. Use gradient_checkpointing=True
     - Trades compute for memory
     - Reduces activation memory by ~60-70%

  4. Reduce batch_size / micro_batch_size
     - Most direct memory reduction
     - Use gradient_accumulation to maintain effective batch size

  5. Use FSDP1 (not ZeRO-2, not FSDP2)
     - FSDP1: proper sharding, no RAM leak (#6468)
     - ZeRO-2: keeps full gradients, higher peak
     - FSDP2: host RAM leak bug
""")
    print(must_do("8B on RTX 4090 MUST use LoRA+bypass+FSDP1 (peak 22.9 GiB)"))
    print(must_do("FSDP1 not FSDP2 (#6468)"))
    print(must_do("gradient_checkpointing=True for 8B models"))

    print(section("Step 4: OFFLOAD — Move data to host RAM"))
    print(code_block("""
# DeepSpeed ZeRO-3 offload config
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}

# WARNING: CPU offload costs ~2-3x throughput on consumer GPUs
# Only use if you CANNOT fit with LoRA+bypass+FSDP1
"""))

    print(section("Step 5: SCALE — Multi-GPU if single GPU insufficient"))
    print(code_block("""
# FSDP1 multi-GPU config
torchrun --nproc_per_node=2 train.py  # 2x RTX 4090 = 48 GiB effective

# FSDP1 wrapping strategy:
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
# SHARD_GRAD_OP = ZeRO-2 equivalent (recommended)
# FULL_SHARD = ZeRO-3 equivalent (more memory savings)
"""))

    print(section("RTX 4090 Memory Budgets"))
    budgets = [
        ("8B LoRA+bypass+FSDP1, gs=8, bs=1", "22.9 GiB", "OK on 24 GiB"),
        ("8B LoRA+bypass+FSDP1, gs=16, bs=2", "~23.8 GiB", "TIGHT on 24 GiB"),
        ("8B LoRA+FSDP1, no bypass, gs=4", "~23.5 GiB", "TIGHT on 24 GiB"),
        ("8B full finetune", "~48 GiB", "IMPOSSIBLE on 24 GiB"),
        ("1.5B LoRA+FSDP1", "~8 GiB", "OK on 24 GiB"),
        ("1.5B full finetune", "~12 GiB", "OK on 24 GiB"),
    ]
    for config, peak, status in budgets:
        color = Color.GREEN if "OK" in status else (Color.RED if "IMPOSSIBLE" in status else Color.YELLOW)
        print(f"  {c(color, f'{status}')}: {config} -> {peak}")


def run_debug_convergence() -> None:
    print(header("CONVERGENCE DEBUGGING WORKFLOW: 5 Steps"))
    print(c(Color.BOLD, "advantage -> loss -> gradient -> learning_rate -> batch"))

    print(section("Step 1: ADVANTAGE — Check advantage computation"))
    print("""
The GRPO advantage formula:
  advantage_i = (reward_i - mean(rewards)) / std(rewards)

Critical checks:
""")
    print(code_block("""
# Verify advantage values
print(f"Rewards: {rewards}")
print(f"Mean: {rewards.mean()}, Std: {rewards.std()}")
print(f"Advantages: {advantages}")
print(f"Advantage mean: {advantages.mean()} (should be ~0)")
print(f"Advantage std: {advantages.std()} (should be ~1)")
"""))
    print(must_do("group_size MUST be >= 4 (#605). gs=1 -> std=0 -> advantage=0 -> REINFORCE"))
    print(must_do("Reward function MUST produce variation within groups"))
    print(must_do("Reward values MUST be clamped to [-10, 10]"))

    print(section("Step 2: LOSS — Check loss computation"))
    print(code_block("""
# GRPO loss formula:
# loss = -log_prob * advantage * mask
# Verify:
print(f"Log probs: {log_probs}")
print(f"Are any log_probs NaN? {torch.isnan(log_probs).any()}")
print(f"Are any log_probs -inf? {(log_probs == float('-inf')).any()}")
print(f"Clamped log probs: {torch.clamp(log_probs, min=-1e8, max=1e8)}")
"""))
    print(must_do("log_probs MUST be clamped before loss computation"))
    print(must_do("Loss MUST use fp32 accumulation"))

    print(section("Step 3: GRADIENT — Check gradient health"))
    print(code_block("""
# Gradient diagnostics
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"Gradient norm before clipping: {grad_norm}")
# If grad_norm > 100: gradient explosion
# If grad_norm < 0.001: gradient vanishing
# If grad_norm == 0: model is not learning

# Per-layer gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}, grad_mean={param.grad.mean():.6f}")
"""))
    print(must_do("clip_grad_norm MUST be 1.0 (#8068)"))

    print(section("Step 4: LEARNING_RATE — Check LR schedule"))
    print(code_block("""
# Monitor LR over training
for step in range(max_steps):
    lr = optimizer.param_groups[0]['lr']
    print(f"Step {step}: LR={lr:.6f}")

# Recommended LR for LoRA GRPO:
# - Initial: 1e-6
# - Warmup: 10% of total steps
# - Schedule: cosine decay
# - Min LR: 1e-7
"""))
    print(must_do("LoRA GRPO LR MUST be ~1e-6 with cosine decay"))

    print(section("Step 5: BATCH — Check batch/group configuration"))
    print("""
GRPO batch configuration:
  - group_size (gs): number of samples per prompt
  - batch_size: number of prompts per step
  - total_samples = batch_size * group_size
  - effective_batch = total_samples * gradient_accumulation_steps

Recommended for 8B on RTX 4090:
  - gs >= 4 (minimum for advantage normalization)
  - bs = 1-2 (memory constrained)
  - gradient_accumulation_steps to reach effective batch ~32-64
""")
    print(must_do("group_size >= 4 (#605)"))
    print(must_do("Effective batch >= 32 for stable GRPO training"))


def run_debug(issue: str) -> None:
    issue_map = {
        "nan": run_debug_nan,
        "oom": run_debug_oom,
        "convergence": run_debug_convergence,
        "conv": run_debug_convergence,
    }
    handler = issue_map.get(issue)
    if handler is None:
        print(c(Color.RED, f"Unknown debug workflow: '{issue}'"))
        print(c(Color.YELLOW, "Available workflows: nan, oom, convergence"))
        return
    handler()


# ---------------------------------------------------------------------------
# Mode 4: rtx4090
# ---------------------------------------------------------------------------

def run_rtx4090() -> None:
    print(header("RTX 4090 DISTRIBUTED TRAINING DEBUGGING GUIDE"))
    print(c(Color.BOLD, "Specific to NVIDIA RTX 4090 (24 GiB VRAM, SM89 Ada Lovelace)"))

    # --- Common issues ---
    print(section("Common RTX 4090 Issues with Solutions"))

    issues = [
        {
            "title": "OOM on 8B model",
            "symptom": "RuntimeError: CUDA out of memory",
            "cause": "8B full finetune needs ~48 GiB; RTX 4090 has only 24 GiB",
            "fix": "Use LoRA + bypass_mode + FSDP1 (peak 22.9 GiB)",
            "must_do": "8B on RTX 4090 MUST use LoRA+bypass+FSDP1",
            "ref": "",
        },
        {
            "title": "NaN with DeepSpeed overlap_comm",
            "symptom": "Silent NaN in weights/activations, intermittent crashes",
            "cause": "overlap_comm=True with dp=1 creates CUDA stream race -> silent data corruption",
            "fix": "Set overlap_comm=False when dp=1",
            "must_do": "overlap_comm=False when dp=1 -- NO EXCEPTIONS (#8061)",
            "ref": "#8061",
        },
        {
            "title": "Gradient explosion",
            "symptom": "Loss spikes, NaN in gradients, unstable training",
            "cause": "No gradient clipping allows gradients to overflow bf16 range",
            "fix": "Set clip_grad_norm=1.0",
            "must_do": "clip_grad_norm MUST be 1.0, not default 0.0 (#8068)",
            "ref": "#8068",
        },
        {
            "title": "Host RAM leak",
            "symptom": "RSS grows continuously, eventually OOM on host",
            "cause": "FSDP2 has known host RAM leak bug",
            "fix": "Use FSDP1 instead of FSDP2",
            "must_do": "Do NOT use FSDP2 -- use FSDP1 (#6468)",
            "ref": "#6468",
        },
        {
            "title": "DSV4 instability",
            "symptom": "NaN, crashes, silent corruption with DeepSpeed v4",
            "cause": "CUDA graphs on consumer GPUs cause instability (11 failures across 4 frameworks)",
            "fix": "Set enforce_eager=True, cuda_graph=False",
            "must_do": "enforce_eager=True MUST be set for DSV4 on RTX 4090",
            "ref": "",
        },
        {
            "title": "Singleton GRPO degeneration",
            "symptom": "advantage=0, reward_effective=0, training not improving",
            "cause": "group_size=1 makes GRPO identical to REINFORCE (no advantage normalization)",
            "fix": "Set group_size >= 4",
            "must_do": "gs >= 4 is MANDATORY; gs=1 = REINFORCE (#605)",
            "ref": "#605",
        },
        {
            "title": "File descriptor exhaustion",
            "symptom": "OSError: Too many open files, intermittent crashes",
            "cause": "Default ulimit (256 on macOS, 1024 on Linux) too low for DeepSpeed/vllm",
            "fix": "ulimit -n 65535 before training",
            "must_do": "ulimit -n MUST be >= 65535 (#8075)",
            "ref": "#8075",
        },
        {
            "title": "Silent corruption",
            "symptom": "Weights slowly corrupted without visible NaN",
            "cause": "CUDA stream race + no detection mechanism",
            "fix": "Enable NanDetectMode + weight checksum every 100 steps",
            "must_do": "Run NanDetectMode + weight checksum (#8061, #8058, #28679)",
            "ref": "#8061, #8058, #28679",
        },
        {
            "title": "Weight reload crash",
            "symptom": "Crash or corrupted output after weight reload (vllm/sglang)",
            "cause": "Stale KV cache and encoder cache after weight reload",
            "fix": "Reset KV cache + encoder cache after weight reload",
            "must_do": "MUST reset KV cache + encoder cache after weight reload (#46125, #28676)",
            "ref": "#46125, #28676",
        },
    ]

    for issue in issues:
        ref = f" ({c(Color.MAGENTA, issue['ref'])})" if issue["ref"] else ""
        print(f"\n  {c(Color.BOLD, c(Color.RED, issue['title']))}{ref}")
        print(f"    Symptom: {issue['symptom']}")
        print(f"    Cause:   {issue['cause']}")
        print(f"    {fix(issue['fix'])}")
        print(f"    {must_do(issue['must_do'])}")

    # --- Config template ---
    print(section("RTX 4090 GRPO Training Config Template"))
    print("All MUST DO rules applied:")

    config_template = {
        "model": {
            "name": "Qwen/Qwen2.5-8B-Instruct",
            "lora_rank": 16,
            "lora_alpha": 32,
            "bypass_mode": True,
        },
        "training": {
            "group_size": 8,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-6,
            "lr_schedule": "cosine",
            "warmup_ratio": 0.1,
            "clip_grad_norm": 1.0,
            "max_steps": 500,
            "bf16": True,
        },
        "distributed": {
            "strategy": "fsdp1",
            "zero_stage": 3,
            "overlap_comm": False,          # MUST DO: #8061
            "enforce_eager": True,          # MUST DO: DSV4
            "cuda_graph": False,            # MUST DO: DSV4
            "gradient_checkpointing": True,
            "sharding_strategy": "FULL_SHARD",
        },
        "rollout": {
            "engine": "vllm",
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "reset_kv_cache_on_reload": True,  # MUST DO: #46125
        },
        "system": {
            "ulimit_n": 65535,               # MUST DO: #8075
            "nccl_net": "Socket",             # MUST DO: PCIe-only
            "nccl_socket_ifname": "eth0",
            "nccl_ib_disable": 1,
        },
        "monitoring": {
            "nan_detect_mode": "backward",    # MUST DO: #8061
            "weight_checksum_interval": 100,  # MUST DO: #8058
            "log_grad_norm": True,
            "log_advantage_stats": True,
        },
    }

    print(code_block(json.dumps(config_template, indent=2)))

    # --- Memory budget template ---
    print(section("RTX 4090 Memory Budget Templates"))

    mem_templates = [
        ("8B LoRA r=16 + bypass + FSDP1", "22.9 GiB", "gs=8, bs=1", "Safe on 24 GiB"),
        ("8B LoRA r=8 + bypass + FSDP1", "21.5 GiB", "gs=8, bs=1", "Safe on 24 GiB"),
        ("8B LoRA r=16 + bypass + ZeRO-3", "23.5 GiB", "gs=8, bs=1", "TIGHT on 24 GiB"),
        ("8B LoRA r=16 + no bypass + FSDP1", "23.5 GiB", "gs=4, bs=1", "TIGHT on 24 GiB"),
        ("8B full finetune + FSDP1", "~48 GiB", "any config", "IMPOSSIBLE on 24 GiB"),
        ("1.5B LoRA + bypass + FSDP1", "~8 GiB", "gs=16, bs=4", "Safe on 24 GiB"),
        ("1.5B full finetune", "~12 GiB", "gs=8, bs=2", "Safe on 24 GiB"),
        ("0.5B full finetune", "~4 GiB", "gs=16, bs=4", "Safe on 24 GiB"),
    ]

    for config, peak, params, status in mem_templates:
        color = Color.GREEN if "Safe" in status else (Color.RED if "IMPOSSIBLE" in status else Color.YELLOW)
        print(f"  {c(color, status)} | {peak} peak | {config} | {params}")

    # --- Timing estimates ---
    print(section("RTX 4090 Timing Estimates (approximate)"))

    timing_data = [
        ("8B LoRA+bypass+FSDP1, gs=8, bs=1, 1 GPU", "~45 sec/step", "~1100 tokens/sec", "~4 hours/500 steps"),
        ("8B LoRA+bypass+FSDP1, gs=8, bs=1, 2 GPU", "~28 sec/step", "~1800 tokens/sec", "~2.5 hours/500 steps"),
        ("1.5B LoRA+bypass+FSDP1, gs=8, bs=2, 1 GPU", "~12 sec/step", "~2600 tokens/sec", "~1 hour/500 steps"),
        ("1.5B full finetune, gs=8, bs=2, 1 GPU", "~18 sec/step", "~1700 tokens/sec", "~1.5 hours/500 steps"),
    ]

    for config, step_time, throughput, total_time in timing_data:
        print(f"  {config}")
        print(f"    Step time: {step_time}  |  Throughput: {throughput}  |  500 steps: {total_time}")

    # --- MUST DO rules summary ---
    print(section("ALL RTX 4090 MUST DO RULES"))

    must_do_rules = [
        "1. overlap_comm=False when dp=1 (#8061)",
        "2. clip_grad_norm=1.0, NOT 0.0 (#8068)",
        "3. enforce_eager=True for DSV4",
        "4. cuda_graph=False for DSV4",
        "5. FSDP1 not FSDP2 (#6468)",
        "6. LoRA+bypass+FSDP1 for 8B on 24 GiB",
        "7. group_size >= 4 (#605)",
        "8. ulimit -n 65535 (#8075)",
        "9. NanDetectMode + weight checksum (#8061, #8058, #28679)",
        "10. Reset KV cache after weight reload (#46125, #28676)",
        "11. All params same dtype before training",
        "12. Loss uses fp32 accumulation",
        "13. log_probs clamped before loss computation",
        "14. Reward values clamped to [-10, 10]",
        "15. LR ~1e-6 with cosine decay for LoRA GRPO",
    ]

    for rule in must_do_rules:
        print(must_do(rule))

    # --- Quick diagnostic commands ---
    print(section("Quick Diagnostic Commands for RTX 4090"))

    quick_cmds = [
        ("Check GPU memory", "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"),
        ("Check ulimit", "ulimit -n"),
        ("Check CUDA version", "nvcc --version | grep release"),
        ("Check NCCL env", "echo $NCCL_NET $NCCL_SOCKET_IFNAME $NCCL_IB_DISABLE"),
        ("Check host RAM", "free -g  # or: sysctl -n hw.memsize on macOS"),
        ("Check file descriptors", "ls /proc/self/fd | wc -l  # or: lsof -p $$ | wc -l"),
        ("Check NaN in weights", "python3 -c 'import torch; m=your_model; for n,p in m.named_parameters(): print(n, torch.isnan(p).any())'"),
        ("Check gradient norms", "python3 -c 'import torch; g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); print(g)'"),
        ("Enable anomaly detection", "python3 -c 'import torch; torch.autograd.set_detect_anomaly(True)'"),
        ("Monitor GPU memory live", "nvidia-smi dmon -s m -d 1"),
        ("Profile memory", "python3 -c 'import torch; print(torch.cuda.memory_summary())'"),
    ]

    for name, cmd in quick_cmds:
        print(f"  {c(Color.BOLD, name)}:")
        print(code_block(cmd))

    # --- Framework-specific notes ---
    print(section("Framework-Specific Notes for RTX 4090"))

    fw_notes = [
        ("verl", "Primary framework for GRPO on RTX 4090. Uses FSDP1, vllm backend. Set enforce_eager=True."),
        ("DeepSpeed v4", "Most issues documented. overlap_comm=False, enforce_eager=True, cuda_graph=False. 11 failures across 4 frameworks."),
        ("TRL", "Simple GRPO trainer. Limited multi-GPU support. Good for single-GPU 1.5B experiments."),
        ("vllm (rollout)", "Fast generation engine. Reset KV cache after weight reload (#46125). Use PagedAttention."),
        ("SGLang (rollout)", "Alternative to vllm. Similar KV cache reset requirement. May have different memory profile."),
    ]
    for fw, note in fw_notes:
        print(f"  {c(Color.BOLD, fw)}: {note}")

    print(section("SM89 (Ada Lovelace) Architecture Notes"))
    print("""
  - bf16 native support (no fp16 emulation needed)
  - fp8 (E4M3/E5M2) available BUT has known issues on consumer GPUs
  - 24 GiB GDDR6X (not HBM — different memory characteristics)
  - 128 CUDA cores per SM, 128 SMs total
  - PCIe Gen4 x16 (no NVLink bridge for consumer models)
  - TDP: 450W — ensure adequate power supply and cooling

  WARNING: fp8 quantization on SM89 has reported issues.
  Use bf16 for training stability on RTX 4090.
""")
    print(warning("Avoid fp8 quantization on RTX 4090 for training. Use bf16."))

    print(header("END OF RTX 4090 GUIDE", 80))
    print(c(Color.GREEN, "Apply ALL MUST DO rules before starting training. Stability over throughput."))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Training Debugging Guide Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Modes:
              diagnose <symptom>  Diagnose likely root causes for a symptom
              checklist           Pre-flight checklist before GRPO training
              debug <workflow>    Step-by-step debugging workflow (nan|oom|convergence)
              rtx4090             RTX 4090 specific debugging guide

            Symptom names for diagnose mode:
              training_nan, gradient_nan, loss_nan, output_nan,
              oom, host_ram_growth, gpu_memory_spike,
              slow_training, slow_convergence, low_throughput,
              intermittent_crashes, random_errors, fd_exhaustion,
              reward_zero, advantage_zero, training_not_improving

            Aliases: nan, grad_nan, loss_nan, out_nan, oom, ram, ram_leak,
              mem_spike, slow, convergence, throughput, crash, random, fd,
              ulimit, reward0, advantage0, not_improving, no_improve
        """),
    )
    parser.add_argument(
        "mode",
        choices=["diagnose", "checklist", "debug", "rtx4090"],
        help="Operating mode",
    )
    parser.add_argument(
        "arg",
        nargs="?",
        default=None,
        help="Symptom name (for diagnose) or workflow name (for debug)",
    )

    args = parser.parse_args()

    if args.mode == "diagnose":
        if args.arg is None:
            print(c(Color.RED, "diagnose mode requires a symptom name"))
            print(c(Color.YELLOW, "Run with --help for available symptoms"))
            sys.exit(1)
        run_diagnose(args.arg)
    elif args.mode == "checklist":
        run_checklist()
    elif args.mode == "debug":
        if args.arg is None:
            print(c(Color.RED, "debug mode requires a workflow name: nan, oom, convergence"))
            sys.exit(1)
        run_debug(args.arg)
    elif args.mode == "rtx4090":
        run_rtx4090()


if __name__ == "__main__":
    main()
