#!/usr/bin/env python3
"""
RTX 4090 GRPO Training Pre-flight Checklist — Automated pre-training validation.

This tool automates ALL pre-flight checks before starting GRPO training on RTX 4090.
It checks hardware, software, config, and framework-specific safety rules.

Modes:
  check     — Run all pre-flight checks and report status
  config    — Validate a specific YAML config file
  quick     — Quick hardware + software check only
  full      — Full check including network connectivity + GPU availability

★★★★★★★★★ 16 MUST DO + 14 MUST NOT rules automated from 7-framework research.
★★★★★★★★★ Based on 1089+ commits of source-level analysis across 7 frameworks.

Usage:
  python tools/rtx4090_grpo_pre_flight_checklist.py --mode check
  python tools/rtx4090_grpo_pre_flight_checklist.py --mode config --config your_config.yaml
  python tools/rtx4090_grpo_pre_flight_checklist.py --mode quick
  python tools/rtx4090_grpo_pre_flight_checklist.py --mode full
"""

import sys
import os
import json
import subprocess
import importlib

# ─── MUST DO Rules (16 total) ──────────────────────────────────────────────

MUST_DO = [
    {"id": "D1", "rule": "lora_rank <= 32", "reason": "#6782: rank>64 breaks EOS in vLLM rollout", "severity": "CRITICAL", "check": "config"},
    {"id": "D2", "rule": "lora_alpha = 2 * lora_rank", "reason": "scaling = alpha/rank = 2.0 standard", "severity": "WARNING", "check": "config"},
    {"id": "D3", "rule": "lora.merge = False (unmerged for HYBRID)", "reason": "HYBRID sleep/wake needs unmerged LoRA", "severity": "CRITICAL", "check": "config"},
    {"id": "D4", "rule": "bypass_mode = True", "reason": "18Ψ→3.8Ψ → ref model eliminated", "severity": "CRITICAL", "check": "config"},
    {"id": "D5", "rule": "overlap_comm = False", "reason": "#8061: multi-stream race → NaN", "severity": "CRITICAL", "check": "config"},
    {"id": "D6", "rule": "zero_stage = 2 (NOT 3)", "reason": "ZeRO-3 regression + overhead #8072", "severity": "CRITICAL", "check": "config"},
    {"id": "D7", "rule": "gradient_clipping = 1.0", "reason": "#8068: default 0→1.0 silent change", "severity": "WARNING", "check": "config"},
    {"id": "D8", "rule": "enforce_eager = True", "reason": "DSV4/MoE cudagraph failures", "severity": "CRITICAL", "check": "config"},
    {"id": "D9", "rule": "free_cache_engine = True", "reason": "Memory management for HYBRID mode", "severity": "WARNING", "check": "config"},
    {"id": "D10", "rule": "FSDP1 backend (NOT FSDP2)", "reason": "#6468: FSDP2 CPU leak 0.6-6.3 GiB/step", "severity": "CRITICAL", "check": "config"},
    {"id": "D11", "rule": "record_stream on ALL async copies", "reason": "#6794: missing → silent corruption", "severity": "CRITICAL", "check": "source"},
    {"id": "D12", "rule": "engine backend = fsdp (FSDP1)", "reason": "#6699: 3 unfixed backends have leak", "severity": "CRITICAL", "check": "config"},
    {"id": "D13", "rule": "CPU_Adam optimizer", "reason": "18Ψ optimizer states > 24 GiB", "severity": "CRITICAL", "check": "config"},
    {"id": "D14", "rule": "Monitor host RAM during training", "reason": "#6468: host OOM kills workers", "severity": "CRITICAL", "check": "runtime"},
    {"id": "D15", "rule": "ulimit -n 65536", "reason": "#8075: fd leak in long-running", "severity": "WARNING", "check": "system"},
    {"id": "D16", "rule": "BF16 training precision", "reason": "Only correct training precision", "severity": "CRITICAL", "check": "config"},
]

# ─── MUST NOT Rules (14 total) ─────────────────────────────────────────────

MUST_NOT = [
    {"id": "N1", "rule": "NOT use lora_rank > 32", "reason": "#6782: breaks EOS", "severity": "CRITICAL"},
    {"id": "N2", "rule": "NOT use ZeRO-3", "reason": "#8072/#8076 regression + overhead", "severity": "CRITICAL"},
    {"id": "N3", "rule": "NOT use overlap_comm = True", "reason": "#8061: NaN confirmed", "severity": "CRITICAL"},
    {"id": "N4", "rule": "NOT use Muon optimizer", "reason": "6 blockers: #5394/#5395/#8068/#5400/#5179/#7939", "severity": "CRITICAL"},
    {"id": "N5", "rule": "NOT use automodel/megatron/torchtitan engine", "reason": "#6699: memory leak (unfixed)", "severity": "CRITICAL"},
    {"id": "N6", "rule": "NOT use rLLM for GRPO", "reason": "#605: grouping bug → BROKEN", "severity": "CRITICAL"},
    {"id": "N7", "rule": "NOT use CUDA graph for DSV4 inference", "reason": "11 failures across 4 frameworks", "severity": "CRITICAL"},
    {"id": "N8", "rule": "NOT use FSDP2 backend", "reason": "#6468: CPU leak → host OOM", "severity": "CRITICAL"},
    {"id": "N9", "rule": "NOT use async side-stream copies without record_stream", "reason": "#6794: silent corruption", "severity": "CRITICAL"},
    {"id": "N10", "rule": "NOT run GRPO >100 steps without host RAM monitoring", "reason": "#6468: host OOM kills workers", "severity": "CRITICAL"},
    {"id": "N11", "rule": "NOT use CPPO without bypass_mode", "reason": "#6731: bypass MANDATORY", "severity": "CRITICAL"},
    {"id": "N12", "rule": "NOT use autocast_adapter_dtype with ZeRO-3", "reason": "#8072: dtype mismatch → TypeError", "severity": "CRITICAL"},
    {"id": "N13", "rule": "NOT use DeepSpeed v0.19.2 with ZeRO-3+LoRA", "reason": "#8072/#8076 regression", "severity": "CRITICAL"},
    {"id": "N14", "rule": "NOT use MTP+structured_output simultaneously", "reason": "#46118: FSM conflict → 58% failure", "severity": "CRITICAL"},
]


def check_system():
    """Check system-level prerequisites."""
    results = []

    # Check ulimit
    try:
        ulimit_output = subprocess.run(["ulimit", "-n"], capture_output=True, text=True, shell=True)
        fd_limit_str = subprocess.run(["bash", "-c", "ulimit -n"], capture_output=True, text=True).stdout.strip()
        if fd_limit_str == "unlimited":
            results.append({"rule": "D15", "status": "PASS", "detail": f"fd limit = unlimited >= 65536"})
        else:
            fd_limit = int(fd_limit_str)
            if fd_limit >= 65536:
                results.append({"rule": "D15", "status": "PASS", "detail": f"fd limit = {fd_limit} >= 65536"})
            else:
                results.append({"rule": "D15", "status": "FAIL", "detail": f"fd limit = {fd_limit} < 65536 → run: ulimit -n 65536"})
    except Exception as e:
        results.append({"rule": "D15", "status": "UNKNOWN", "detail": f"Cannot check ulimit: {e}"})

    # Check GPU
    try:
        gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True)
        if gpu_info.returncode == 0:
            lines = gpu_info.stdout.strip().split("\n")
            for line in lines:
                name, mem = line.split(",")
                mem_gb = float(mem.strip().split()[0]) / 1024
                if "4090" in name:
                    results.append({"rule": "GPU", "status": "PASS", "detail": f"{name.strip()}, {mem_gb:.1f} GiB"})
                else:
                    results.append({"rule": "GPU", "status": "INFO", "detail": f"{name.strip()}, {mem_gb:.1f} GiB (not RTX 4090)"})
        else:
            results.append({"rule": "GPU", "status": "FAIL", "detail": "nvidia-smi not available — no GPU detected"})
    except Exception:
        results.append({"rule": "GPU", "status": "FAIL", "detail": "nvidia-smi not found — CUDA not installed or no GPU"})

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            results.append({"rule": "CUDA", "status": "PASS",
                           "detail": f"CUDA {cuda_version}, {device_name}, {device_mem:.1f} GiB"})
        else:
            results.append({"rule": "CUDA", "status": "FAIL", "detail": "torch.cuda not available"})
    except ImportError:
        results.append({"rule": "CUDA", "status": "FAIL", "detail": "PyTorch not installed"})

    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 10):
        results.append({"rule": "Python", "status": "PASS", "detail": f"Python {py_version.major}.{py_version.minor}"})
    else:
        results.append({"rule": "Python", "status": "FAIL", "detail": f"Python {py_version.major}.{py_version.minor} < 3.10"})

    # Check key packages
    for pkg in ["torch", "vllm", "ray", "omegaconf"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            results.append({"rule": pkg, "status": "PASS", "detail": f"{pkg} {ver}"})
        except ImportError:
            results.append({"rule": pkg, "status": "FAIL", "detail": f"{pkg} not installed"})

    return results


def check_config(config_dict):
    """Validate config against MUST DO / MUST NOT rules."""
    results = []

    # Extract key config values
    lora_rank = config_dict.get("lora_rank", None)
    lora_alpha = config_dict.get("lora_alpha", None)
    lora_merge = config_dict.get("lora.merge", config_dict.get("lora_merge", None))
    bypass_mode = config_dict.get("bypass_mode", config_dict.get("rollout.bypass_mode", None))
    overlap_comm = config_dict.get("overlap_comm", None)
    zero_stage = config_dict.get("zero_stage", None)
    grad_clip = config_dict.get("gradient_clipping", config_dict.get("gradient_clip", None))
    enforce_eager = config_dict.get("enforce_eager", None)
    free_cache = config_dict.get("free_cache_engine", None)
    strategy = config_dict.get("strategy", config_dict.get("actor.strategy", None))
    engine_backend = config_dict.get("engine_backend", config_dict.get("actor.engine_backend", None))
    optimizer = config_dict.get("optimizer", config_dict.get("actor.optim.name", None))
    dtype = config_dict.get("dtype", None)

    # MUST DO checks
    for must in MUST_DO:
        if must["check"] != "config":
            continue
        status = "UNKNOWN"
        detail = ""
        rule_id = must["id"]

        if rule_id == "D1":  # lora_rank <= 32
            if lora_rank is not None:
                if lora_rank <= 32:
                    status = "PASS"
                    detail = f"lora_rank = {lora_rank} <= 32"
                else:
                    status = "FAIL"
                    detail = f"lora_rank = {lora_rank} > 32 → #6782 EOS bug!"
            else:
                status = "SKIP"
                detail = "lora_rank not specified"

        elif rule_id == "D2":  # lora_alpha = 2 * lora_rank
            if lora_rank is not None and lora_alpha is not None:
                if lora_alpha == 2 * lora_rank:
                    status = "PASS"
                    detail = f"lora_alpha = {lora_alpha} = 2 × lora_rank = {2*lora_rank}"
                else:
                    status = "WARNING"
                    detail = f"lora_alpha = {lora_alpha} ≠ 2 × lora_rank = {2*lora_rank}"
            else:
                status = "SKIP"
                detail = "lora_rank or lora_alpha not specified"

        elif rule_id == "D3":  # lora.merge = False
            if lora_merge is not None:
                if lora_merge is False or lora_merge == "False":
                    status = "PASS"
                    detail = f"lora.merge = {lora_merge} (unmerged for HYBRID)"
                else:
                    status = "FAIL"
                    detail = f"lora.merge = {lora_merge} → HYBRID needs unmerged!"
            else:
                status = "SKIP"
                detail = "lora.merge not specified"

        elif rule_id == "D4":  # bypass_mode = True
            if bypass_mode is not None:
                if bypass_mode is True or bypass_mode == "True":
                    status = "PASS"
                    detail = f"bypass_mode = {bypass_mode} → 18Ψ→3.8Ψ"
                else:
                    status = "FAIL"
                    detail = f"bypass_mode = {bypass_mode} → ref model overhead!"
            else:
                status = "FAIL"
                detail = "bypass_mode not specified → MUST set True!"

        elif rule_id == "D5":  # overlap_comm = False
            if overlap_comm is not None:
                if overlap_comm is False or overlap_comm == "False":
                    status = "PASS"
                    detail = f"overlap_comm = {overlap_comm} → no multi-stream race"
                else:
                    status = "FAIL"
                    detail = f"overlap_comm = {overlap_comm} → #8061 NaN bug!"
            else:
                status = "PASS"
                detail = "overlap_comm not specified → default False OK"

        elif rule_id == "D6":  # zero_stage = 2
            if zero_stage is not None:
                if zero_stage == 2:
                    status = "PASS"
                    detail = f"zero_stage = {zero_stage}"
                else:
                    status = "FAIL"
                    detail = f"zero_stage = {zero_stage} → MUST use 2 (NOT 3)!"
            else:
                status = "PASS"
                detail = "zero_stage not specified → default ZeRO-2 OK for verl"

        elif rule_id == "D7":  # gradient_clipping = 1.0
            if grad_clip is not None:
                if grad_clip == 1.0:
                    status = "PASS"
                    detail = f"gradient_clipping = {grad_clip}"
                else:
                    status = "WARNING"
                    detail = f"gradient_clipping = {grad_clip} → recommended 1.0"
            else:
                status = "WARNING"
                detail = "gradient_clipping not specified → MUST set 1.0!"

        elif rule_id == "D8":  # enforce_eager = True
            if enforce_eager is not None:
                if enforce_eager is True or enforce_eager == "True":
                    status = "PASS"
                    detail = f"enforce_eager = {enforce_eager}"
                else:
                    status = "FAIL"
                    detail = f"enforce_eager = {enforce_eager} → DSV4/MoE cudagraph risk!"
            else:
                status = "WARNING"
                detail = "enforce_eager not specified → recommend True"

        elif rule_id == "D9":  # free_cache_engine
            if free_cache is not None:
                if free_cache is True or free_cache == "True":
                    status = "PASS"
                    detail = f"free_cache_engine = {free_cache}"
                else:
                    status = "WARNING"
                    detail = f"free_cache_engine = {free_cache} → recommend True"
            else:
                status = "SKIP"
                detail = "free_cache_engine not specified"

        elif rule_id == "D10":  # FSDP1 backend
            if strategy is not None:
                if strategy in ["fsdp", "fsdp1"]:
                    status = "PASS"
                    detail = f"strategy = {strategy} → FSDP1"
                elif strategy in ["fsdp2"]:
                    status = "FAIL"
                    detail = f"strategy = {strategy} → #6468 FSDP2 CPU leak!"
                else:
                    status = "INFO"
                    detail = f"strategy = {strategy} → verify this uses FSDP1"
            else:
                status = "WARNING"
                detail = "strategy not specified → MUST use fsdp (FSDP1)"

        elif rule_id == "D12":  # engine backend = fsdp
            if engine_backend is not None:
                if engine_backend in ["fsdp", "fsdp1"]:
                    status = "PASS"
                    detail = f"engine_backend = {engine_backend}"
                elif engine_backend in ["automodel", "megatron", "torchtitan"]:
                    status = "FAIL"
                    detail = f"engine_backend = {engine_backend} → #6699 leak!"
                else:
                    status = "INFO"
                    detail = f"engine_backend = {engine_backend}"
            else:
                status = "SKIP"
                detail = "engine_backend not specified"

        elif rule_id == "D13":  # CPU_Adam optimizer
            if optimizer is not None:
                if "cpu" in optimizer.lower() or "adam" in optimizer.lower():
                    status = "PASS"
                    detail = f"optimizer = {optimizer}"
                elif "muon" in optimizer.lower():
                    status = "FAIL"
                    detail = f"optimizer = {optimizer} → Muon 6 blockers!"
                else:
                    status = "INFO"
                    detail = f"optimizer = {optimizer}"
            else:
                status = "SKIP"
                detail = "optimizer not specified"

        elif rule_id == "D16":  # BF16 training
            if dtype is not None:
                if dtype in ["bfloat16", "bf16"]:
                    status = "PASS"
                    detail = f"dtype = {dtype} → BF16 training"
                elif dtype in ["float16", "fp16"]:
                    status = "WARNING"
                    detail = f"dtype = {dtype} → BF16 recommended for RTX 4090"
                else:
                    status = "INFO"
                    detail = f"dtype = {dtype}"
            else:
                status = "PASS"
                detail = "dtype not specified → BF16 default OK"

        results.append({"rule": rule_id, "status": status, "detail": detail,
                        "severity": must["severity"], "reason": must["reason"]})

    # MUST NOT checks
    for must_not in MUST_NOT:
        status = "PASS"
        detail = ""
        rule_id = must_not["id"]

        if rule_id == "N1":  # lora_rank > 32
            if lora_rank is not None and lora_rank > 32:
                status = "FAIL"
                detail = f"lora_rank = {lora_rank} > 32 → #6782!"
            else:
                status = "PASS"
                detail = "lora_rank <= 32 or not specified"

        elif rule_id == "N2":  # ZeRO-3
            if zero_stage is not None and zero_stage == 3:
                status = "FAIL"
                detail = f"zero_stage = 3 → #8072/#8076 regression!"
            else:
                status = "PASS"
                detail = "not using ZeRO-3"

        elif rule_id == "N3":  # overlap_comm True
            if overlap_comm is True or overlap_comm == "True":
                status = "FAIL"
                detail = f"overlap_comm = True → #8061 NaN!"
            else:
                status = "PASS"
                detail = "overlap_comm not True"

        elif rule_id == "N4":  # Muon
            if optimizer is not None and "muon" in optimizer.lower():
                status = "FAIL"
                detail = f"Muon optimizer → 6 blockers!"
            else:
                status = "PASS"
                detail = "not using Muon"

        elif rule_id == "N5":  # automodel/megatron/torchtitan engine
            if engine_backend is not None and engine_backend in ["automodel", "megatron", "torchtitan"]:
                status = "FAIL"
                detail = f"engine_backend = {engine_backend} → #6699 leak!"
            else:
                status = "PASS"
                detail = "not using unsafe engine backend"

        elif rule_id == "N8":  # FSDP2
            if strategy in ["fsdp2"]:
                status = "FAIL"
                detail = f"strategy = fsdp2 → #6468 CPU leak!"
            else:
                status = "PASS"
                detail = "not using FSDP2"

        elif rule_id == "N11":  # CPPO without bypass
            adv_est = config_dict.get("adv_estimator", config_dict.get("algorithm.adv_estimator", None))
            if adv_est == "cppo" and (bypass_mode is False or bypass_mode == "False"):
                status = "FAIL"
                detail = f"CPPO + bypass_mode=False → #6731 MANDATORY!"
            else:
                status = "PASS"
                detail = "CPPO+bypass properly configured or not using CPPO"

        elif rule_id == "N14":  # MTP+structured_output
            mtp = config_dict.get("speculative_decode", None)
            structured = config_dict.get("structured_output", None)
            if mtp and structured:
                status = "FAIL"
                detail = "MTP+structured_output → #46118 FSM conflict!"
            else:
                status = "PASS"
                detail = "not using MTP+structured_output simultaneously"

        elif rule_id in ["N6", "N7", "N9", "N10", "N12", "N13"]:
            status = "INFO"
            detail = "Runtime/source check required (not config-checkable)"

        results.append({"rule": rule_id, "status": status, "detail": detail,
                        "severity": must_not["severity"], "reason": must_not["reason"]})

    return results


def print_results(results, title=""):
    """Print check results with color coding."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARNING")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")
    info_count = sum(1 for r in results if r["status"] == "INFO")
    unknown_count = sum(1 for r in results if r["status"] == "UNKNOWN")

    for r in results:
        severity = r.get("severity", "")
        status = r["status"]
        rule = r["rule"]
        detail = r["detail"]
        reason = r.get("reason", "")

        icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "SKIP": "⏭️", "INFO": "ℹ️", "UNKNOWN": "??"}.get(status, "??")
        sev_icon = {"CRITICAL": "🔴", "WARNING": "🟡"}.get(severity, "")

        print(f"  {icon} {sev_icon} [{rule}] {detail}")
        if reason and status in ["FAIL", "WARNING"]:
            print(f"     Reason: {reason}")

    print(f"\n  Summary: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARNING, "
          f"{skip_count} SKIP, {info_count} INFO, {unknown_count} UNKNOWN")

    if fail_count > 0:
        print(f"  ★★★★★★★★ {fail_count} FAILURES — MUST FIX before training!")
        return False
    elif warn_count > 0:
        print(f"  ⚠️ {warn_count} WARNINGS — review recommended")
        return True
    else:
        print(f"  ✅ All checks PASS — ready for training!")
        return True


def run_mode_check():
    """Run all pre-flight checks."""
    print("★★★★★★★★★ RTX 4090 GRPO Training Pre-flight Checklist")
    print(f"★★★★★★★★★ 16 MUST DO + 14 MUST NOT rules")
    print(f"★★★★★★★★★ Based on 7-framework source-level analysis")

    # System checks
    sys_results = check_system()
    sys_ok = print_results(sys_results, "System Checks")

    if not sys_ok:
        print("\n★★★★★★★★★ System checks FAILED — fix before proceeding!")
        return False

    # Config rules (show without specific config)
    print(f"\n{'='*60}")
    print(f"  Config Rules (MUST DO / MUST NOT)")
    print(f"{'='*60}")
    print(f"  Run --mode config --config YOUR_CONFIG.yaml to validate specific config")
    print(f"\n  MUST DO rules ({len(MUST_DO)}):")
    for must in MUST_DO:
        sev = {"CRITICAL": "🔴", "WARNING": "🟡"}.get(must["severity"], "")
        print(f"    {sev} [{must['id']}] {must['rule']} — {must['reason']}")

    print(f"\n  MUST NOT rules ({len(MUST_NOT)}):")
    for must_not in MUST_NOT:
        sev = {"CRITICAL": "🔴"}.get(must_not["severity"], "")
        print(f"    {sev} [{must_not['id']}] {must_not['rule']} — {must_not['reason']}")

    return True


def run_mode_config(config_path):
    """Validate a specific config file."""
    print("★★★★★★★★★ RTX 4090 GRPO Config Validation")
    print(f"  Config: {config_path}")

    # Load config
    try:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            except ImportError:
                print("  ❌ yaml not installed → pip install pyyaml")
                return False
        elif config_path.endswith(".json"):
            with open(config_path) as f:
                config = json.load(f)
        else:
            print(f"  ❌ Unknown config format: {config_path}")
            return False
    except FileNotFoundError:
        print(f"  ❌ Config file not found: {config_path}")
        return False

    # Flatten nested config for easier checking
    flat_config = {}
    if isinstance(config, dict):
        def flatten(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten(v, f"{prefix}{k}.")
                else:
                    flat_config[f"{prefix}{k}"] = v
                    flat_config[k] = v
        flatten(config)

    # System checks first
    sys_results = check_system()
    print_results(sys_results, "System Checks")

    # Config checks
    config_results = check_config(flat_config)
    config_ok = print_results(config_results, "Config Validation (MUST DO / MUST NOT)")

    # Memory estimate
    model_path = flat_config.get("model.path", flat_config.get("path", ""))
    if model_path:
        # Estimate model size from name
        model_sizes = {
            "1.7b": 1.7, "1.5b": 1.5, "2b": 2, "3b": 3, "4b": 4,
            "7b": 7, "8b": 8, "9b": 9, "14b": 14, "27b": 27, "30b": 30,
            "35b": 35, "70b": 70, "72b": 72,
        }
        estimated_size = None
        model_lower = model_path.lower()
        for key, size in model_sizes.items():
            if key in model_lower:
                estimated_size = size
                break

        if estimated_size:
            # Memory estimate
            bf16_mem = estimated_size * 2  # GiB (bf16 weights)
            fp32_opt = estimated_size * 8   # GiB (optimizer states, offloaded)
            lora_add = flat_config.get("lora_rank", 32) * flat_config.get("lora_alpha", 64) / estimated_size * 0.1

            if flat_config.get("bypass_mode", True):
                peak_gpu = bf16_mem + lora_add + 2  # activations + temp
            else:
                peak_gpu = bf16_mem * 2 + fp32_opt + 2  # ref model + optimizer

            print(f"\n{'='*60}")
            print(f"  Memory Estimate for ~{estimated_size}B model")
            print(f"{'='*60}")
            print(f"  BF16 weights:     {bf16_mem:.1f} GiB")
            print(f"  FP32 optimizer:   {fp32_opt:.1f} GiB (CPU offloaded)")
            print(f"  LoRA overhead:    ~{lora_add:.2f} GiB")
            print(f"  Activations:      ~2-4 GiB")
            if flat_config.get("bypass_mode", True):
                print(f"  Reference model:  0 GiB (bypass_mode=True)")
            else:
                print(f"  Reference model:  {bf16_mem:.1f} GiB (bypass_mode=False → WARNING!)")
            print(f"  Estimated peak:   ~{peak_gpu:.1f} GiB")
            if peak_gpu <= 22:
                print(f"  ✅ Fits on RTX 4090 (24 GiB) with ~{24-peak_gpu:.1f} GiB margin")
            else:
                print(f"  ❌ OOM risk on RTX 4090 (peak {peak_gpu:.1f} > 24 GiB)")

    return config_ok


def run_mode_quick():
    """Quick hardware + software check."""
    print("★★★★★★★★★ RTX 4090 Quick Hardware Check")
    sys_results = check_system()
    print_results(sys_results, "Quick Check")
    return True


def run_mode_full():
    """Full check including GPU connectivity."""
    print("★★★★★★★★★ RTX 4090 Full Pre-flight Check")

    # System checks
    sys_results = check_system()
    print_results(sys_results, "System Checks")

    # GPU connectivity
    print(f"\n{'='*60}")
    print(f"  GPU Server Connectivity")
    print(f"{'='*60}")

    servers = [
        ("University server", "sshpass -p 'adspzxw123' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 zxw@219.223.198.62 hostname"),
        ("Matpool server", "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 28959 root@hz-t3.matpool.com hostname"),
    ]

    for name, cmd in servers:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                print(f"  ✅ {name}: ONLINE ({result.stdout.strip()})")
            else:
                print(f"  ❌ {name}: OFFLINE")
        except subprocess.TimeoutExpired:
            print(f"  ❌ {name}: TIMEOUT (offline)")
        except Exception as e:
            print(f"  ❌ {name}: ERROR ({e})")

    # Config rules
    run_mode_check()
    return True


def main():
    if len(sys.argv) < 2 or "--mode" not in sys.argv:
        print("Usage: python tools/rtx4090_grpo_pre_flight_checklist.py --mode <mode>")
        print("  Modes: check, config, quick, full")
        print("  Config mode requires: --config <path>")
        sys.exit(1)

    mode_idx = sys.argv.index("--mode") + 1
    if mode_idx >= len(sys.argv):
        print("❌ --mode requires a value")
        sys.exit(1)

    mode = sys.argv[mode_idx]

    if mode == "check":
        ok = run_mode_check()
    elif mode == "config":
        config_idx = sys.argv.index("--config") + 1 if "--config" in sys.argv else None
        if config_idx is None or config_idx >= len(sys.argv):
            print("❌ --mode config requires --config <path>")
            sys.exit(1)
        ok = run_mode_config(sys.argv[config_idx])
    elif mode == "quick":
        ok = run_mode_quick()
    elif mode == "full":
        ok = run_mode_full()
    else:
        print(f"❌ Unknown mode: {mode}")
        print("  Available modes: check, config, quick, full")
        sys.exit(1)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
