"""
GPU Experiment Helpers - Common utilities for PR validation experiments.

Provides shared functions for:
- GPU setup and verification
- Memory tracking (peak, current, per-device)
- Timing utilities (wall-clock, per-step)
- Result saving (JSON, CSV, summary reports)
- NaN detection in tensors
- Gradient norm computation
- Convergence metric tracking
"""

import os
import sys
import json
import time
import csv
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# GPU Setup and Verification
# ---------------------------------------------------------------------------

def check_gpu_available() -> bool:
    """Check if nvidia-smi is accessible and at least one GPU is present."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print("[ERROR] nvidia-smi returned non-zero exit code")
            return False
        gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not gpus:
            print("[ERROR] No GPUs detected by nvidia-smi")
            return False
        print(f"[OK] Detected {len(gpus)} GPU(s): {gpus}")
        return True
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found — no NVIDIA driver installed")
        return False
    except subprocess.TimeoutExpired:
        print("[ERROR] nvidia-smi timed out")
        return False


def get_gpu_info() -> Dict[str, Any]:
    """Return detailed GPU information from nvidia-smi."""
    info = {"gpus": [], "driver_version": "", "cuda_version": ""}
    try:
        # Driver and CUDA version
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            if "Driver Version" in line:
                info["driver_version"] = line.split("Driver Version:")[1].strip().split()[0]
            if "CUDA Version" in line:
                info["cuda_version"] = line.split("CUDA Version:")[1].strip().split()[0]

        # Per-GPU details
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                info["gpus"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]),
                    "memory_free_mb": float(parts[3]),
                    "memory_used_mb": float(parts[4]),
                    "utilization_pct": float(parts[5]),
                })
    except Exception as e:
        print(f"[WARN] Could not get full GPU info: {e}")
    return info


def check_cuda_pytorch_compat() -> Dict[str, Any]:
    """Check CUDA version compatibility between driver, runtime, and PyTorch."""
    compat = {"driver_cuda": "", "pytorch_cuda": "", "pytorch_version": "", "compatible": False, "errors": []}
    try:
        import torch
        compat["pytorch_version"] = torch.__version__
        compat["pytorch_cuda"] = torch.version.cuda or "N/A"
        compat["pytorch_cudnn"] = torch.version.cudnn or "N/A"

        gpu_info = get_gpu_info()
        compat["driver_cuda"] = gpu_info["cuda_version"]

        # Rough compatibility check: PyTorch CUDA should be <= driver CUDA
        if compat["pytorch_cuda"] != "N/A" and compat["driver_cuda"]:
            pt_major = int(compat["pytorch_cuda"].split(".")[0])
            drv_major = int(compat["driver_cuda"].split(".")[0])
            compat["compatible"] = drv_major >= pt_major
            if not compat["compatible"]:
                compat["errors"].append(
                    f"Driver CUDA {compat['driver_cuda']} < PyTorch CUDA {compat['pytorch_cuda']}"
                )

        compat["gpu_available"] = torch.cuda.is_available()
        compat["gpu_count"] = torch.cuda.device_count()
        if compat["gpu_available"] and compat["gpu_count"] > 0:
            compat["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(compat["gpu_count"])
            ]
            compat["gpu_memory_gb"] = [
                torch.cuda.get_device_properties(i).total_mem / (1024**3)
                for i in range(compat["gpu_count"])
            ]
    except ImportError:
        compat["errors"].append("PyTorch not installed")
    except Exception as e:
        compat["errors"].append(str(e))

    return compat


def set_gpu_device(device_id: int = 0) -> None:
    """Set the CUDA device and print confirmation."""
    import torch
    torch.cuda.set_device(device_id)
    print(f"[OK] Set CUDA device to {device_id}: {torch.cuda.get_device_name(device_id)}")


# ---------------------------------------------------------------------------
# Memory Tracking
# ---------------------------------------------------------------------------

class MemoryTracker:
    """Track GPU memory usage over time."""

    def __init__(self, device: int = 0):
        self.device = device
        self.samples: List[Dict[str, float]] = []
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0

    def sample(self, label: str = "") -> Dict[str, float]:
        """Take a memory snapshot and record it."""
        import torch
        allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**2)

        self.peak_allocated_mb = max(self.peak_allocated_mb, peak_allocated)
        self.peak_reserved_mb = max(self.peak_reserved_mb, peak_reserved)

        snapshot = {
            "label": label,
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "peak_allocated_mb": round(peak_allocated, 2),
            "peak_reserved_mb": round(peak_reserved, 2),
            "timestamp": time.time(),
        }
        self.samples.append(snapshot)
        return snapshot

    def reset_peak(self) -> None:
        """Reset peak memory trackers."""
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_allocated_mb = 0.0
        self.peak_reserved_mb = 0.0

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all memory samples."""
        if not self.samples:
            return {"peak_allocated_mb": 0, "peak_reserved_mb": 0, "samples": []}
        return {
            "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            "peak_reserved_mb": round(self.peak_reserved_mb, 2),
            "peak_allocated_gib": round(self.peak_allocated_mb / 1024, 3),
            "peak_reserved_gib": round(self.peak_reserved_mb / 1024, 3),
            "samples": self.samples,
        }


# ---------------------------------------------------------------------------
# Timing Utilities
# ---------------------------------------------------------------------------

class StepTimer:
    """Track timing per training step with wall-clock and throughput."""

    def __init__(self):
        self.step_times: List[float] = []
        self.step_labels: List[str] = []
        self._start: Optional[float] = None
        self._total_start: Optional[float] = None

    def start_run(self) -> None:
        """Mark the start of the entire run."""
        self._total_start = time.time()

    def start_step(self) -> None:
        """Mark the start of a single step."""
        self._start = time.time()

    def end_step(self, label: str = "") -> float:
        """Mark the end of a step and record elapsed time."""
        elapsed = time.time() - (self._start or time.time())
        self.step_times.append(elapsed)
        self.step_labels.append(label)
        self._start = None
        return elapsed

    def end_run(self) -> float:
        """Mark the end of the entire run."""
        total = time.time() - (self._total_start or time.time())
        self._total_start = None
        return total

    def summary(self) -> Dict[str, Any]:
        """Return timing summary statistics."""
        if not self.step_times:
            return {"total_steps": 0, "mean_step_s": 0, "median_step_s": 0}
        mean_step = sum(self.step_times) / len(self.step_times)
        sorted_times = sorted(self.step_times)
        n = len(sorted_times)
        median_step = sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return {
            "total_steps": len(self.step_times),
            "total_wall_s": round(sum(self.step_times), 2),
            "mean_step_s": round(mean_step, 2),
            "median_step_s": round(median_step, 2),
            "min_step_s": round(min(self.step_times), 2),
            "max_step_s": round(max(self.step_times), 2),
            "steps_per_hour": round(3600 / mean_step, 1) if mean_step > 0 else 0,
            "step_times": self.step_times,
            "step_labels": self.step_labels,
        }


# ---------------------------------------------------------------------------
# NaN and Gradient Utilities
# ---------------------------------------------------------------------------

def check_tensor_for_nan(tensor, name: str = "tensor") -> Dict[str, Any]:
    """Check a tensor for NaN values and return detailed info."""
    import torch
    result = {
        "name": name,
        "has_nan": False,
        "nan_count": 0,
        "total_count": 0,
        "nan_fraction": 0.0,
        "has_inf": False,
        "inf_count": 0,
        "min_val": None,
        "max_val": None,
        "mean_val": None,
    }
    if tensor is None:
        result["error"] = "tensor is None"
        return result
    with torch.no_grad():
        flat = tensor.flatten().float()
        result["total_count"] = flat.numel()
        nan_mask = torch.isnan(flat)
        result["has_nan"] = nan_mask.any().item()
        result["nan_count"] = nan_mask.sum().item()
        result["nan_fraction"] = result["nan_count"] / result["total_count"] if result["total_count"] > 0 else 0.0
        inf_mask = torch.isinf(flat)
        result["has_inf"] = inf_mask.any().item()
        result["inf_count"] = inf_mask.sum().item()
        valid = flat[~nan_mask & ~inf_mask]
        if valid.numel() > 0:
            result["min_val"] = valid.min().item()
            result["max_val"] = valid.max().item()
            result["mean_val"] = valid.mean().item()
    return result


def compute_gradient_norm(model, norm_type: float = 2.0) -> Tuple[float, Dict[str, float]]:
    """Compute the total gradient norm of a model and per-layer norms."""
    import torch
    total_norm = 0.0
    per_layer = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type).item()
            total_norm += param_norm ** norm_type
            per_layer[name] = round(param_norm, 6)
    total_norm = total_norm ** (1.0 / norm_type)
    return round(total_norm, 6), per_layer


def clip_gradients_and_report(model, max_norm: float = 1.0, norm_type: float = 2.0) -> Dict[str, Any]:
    """Clip gradients and return before/after norm comparison."""
    import torch
    before_norm, before_per_layer = compute_gradient_norm(model, norm_type)
    total_norm_before = torch.tensor(before_norm)
    clip_coef = max_norm / (total_norm_before + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(clip_coef_clamped.item())
    after_norm, after_per_layer = compute_gradient_norm(model, norm_type)
    return {
        "before_norm": before_norm,
        "after_norm": after_norm,
        "max_norm": max_norm,
        "clip_coef": round(clip_coef.item(), 6),
        "was_clipped": clip_coef.item() < 1.0,
        "before_per_layer": before_per_layer,
        "after_per_layer": after_per_layer,
    }


# ---------------------------------------------------------------------------
# Advantage Statistics (for GRPO experiments)
# ---------------------------------------------------------------------------

def compute_advantage_stats(advantages: List[float]) -> Dict[str, Any]:
    """Compute statistics on a list of advantage values."""
    if not advantages:
        return {"error": "empty advantage list"}
    import math
    n = len(advantages)
    mean = sum(advantages) / n
    variance = sum((a - mean) ** 2 for a in advantages) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "count": n,
        "mean": round(mean, 6),
        "variance": round(variance, 6),
        "std": round(std, 6),
        "min": round(min(advantages), 6),
        "max": round(max(advantages), 6),
        "range": round(max(advantages) - min(advantages), 6),
    }


# ---------------------------------------------------------------------------
# Result Saving
# ---------------------------------------------------------------------------

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save a dictionary as JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[OK] Saved JSON to {filepath}")


def save_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    """Save a list of dicts as CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] No rows to save for {filepath}")
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Saved CSV to {filepath}")


def write_summary_report(
    experiment_name: str,
    results: Dict[str, Any],
    output_dir: str,
    expected: Optional[str] = None,
    observed: Optional[str] = None,
    pass_fail: Optional[str] = None,
) -> str:
    """Write a human-readable summary report for an experiment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "summary_report.txt")

    lines = [
        "=" * 80,
        f"EXPERIMENT: {experiment_name}",
        f"Date: {datetime.now().isoformat()}",
        "=" * 80,
        "",
        "EXPECTED RESULT:",
        f"  {expected or 'N/A'}",
        "",
        "OBSERVED RESULT:",
        f"  {observed or 'N/A'}",
        "",
        "PASS/FAIL:",
        f"  {pass_fail or 'N/A'}",
        "",
        "-" * 80,
        "DETAILED RESULTS:",
        "",
    ]

    # Flatten nested results for readability
    def flatten(d: Dict, prefix: str = "") -> List[str]:
        items = []
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(flatten(v, f"{key}."))
            elif isinstance(v, list):
                items.append(f"  {key}: [{v[0] if v else 'empty'}...] (len={len(v)})")
            else:
                items.append(f"  {key}: {v}")
        return items

    lines.extend(flatten(results))
    lines.append("")
    lines.append("=" * 80)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Summary report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Environment Helpers
# ---------------------------------------------------------------------------

def setup_ulimit() -> Dict[str, Any]:
    """Set recommended ulimits for GPU training and return current values."""
    result = {}
    try:
        import resource

        # Soft limit for open files
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        result["nofile_soft_before"] = soft
        result["nofile_hard"] = hard
        desired = min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
        soft_new, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        result["nofile_soft_after"] = soft_new
        print(f"[OK] ulimit -n set to {soft_new} (was {soft}, hard={hard})")
    except Exception as e:
        result["error"] = str(e)
        print(f"[WARN] Could not set ulimit: {e}")

    return result


def activate_conda_env(env_name: str) -> bool:
    """Attempt to activate a conda environment (for use within Python)."""
    try:
        conda_prefix = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not conda_prefix:
            print("[WARN] conda not found")
            return False

        env_path = os.path.join(conda_prefix, "envs", env_name)
        if not os.path.isdir(env_path):
            print(f"[WARN] Conda env '{env_name}' not found at {env_path}")
            return False

        # Add conda env to PATH
        bin_path = os.path.join(env_path, "bin")
        os.environ["PATH"] = bin_path + ":" + os.environ.get("PATH", "")
        os.environ["CONDA_DEFAULT_ENV"] = env_name
        os.environ["CONDA_PREFIX"] = env_path
        print(f"[OK] Conda env '{env_name}' activated (prefix: {env_path})")
        return True
    except Exception as e:
        print(f"[WARN] Could not activate conda env: {e}")
        return False


# ---------------------------------------------------------------------------
# Experiment Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_experiment(
    experiment_name: str,
    output_dir: str,
    gpu_device: int = 0,
    conda_env: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all common setup for an experiment and return a config dict."""
    config = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "gpu_device": gpu_device,
        "conda_env": conda_env,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "setup_results": {},
    }

    print(f"\n{'='*80}")
    print(f"BOOTSTRAPPING: {experiment_name}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # GPU check
    gpu_ok = check_gpu_available()
    config["setup_results"]["gpu_available"] = gpu_ok
    if not gpu_ok:
        print("[FATAL] No GPU available — cannot run experiment")
        config["setup_results"]["fatal_error"] = "no_gpu"
        save_json(config, os.path.join(output_dir, "bootstrap_config.json"))
        return config

    # GPU info
    gpu_info = get_gpu_info()
    config["setup_results"]["gpu_info"] = gpu_info

    # CUDA / PyTorch compat
    compat = check_cuda_pytorch_compat()
    config["setup_results"]["cuda_compat"] = compat
    if compat.get("errors"):
        for err in compat["errors"]:
            print(f"[WARN] CUDA compat issue: {err}")

    # Set GPU device
    if compat.get("gpu_available"):
        set_gpu_device(gpu_device)

    # Ulimit
    ulimit_result = setup_ulimit()
    config["setup_results"]["ulimit"] = ulimit_result

    # Conda env (optional)
    if conda_env:
        activate_conda_env(conda_env)

    # Save bootstrap config
    save_json(config, os.path.join(output_dir, "bootstrap_config.json"))

    print(f"\n[OK] Bootstrap complete for {experiment_name}")
    return config


def finalize_experiment(
    experiment_name: str,
    output_dir: str,
    results: Dict[str, Any],
    timer: Optional[StepTimer] = None,
    mem_tracker: Optional[MemoryTracker] = None,
    expected: str = "",
    observed: str = "",
    pass_fail: str = "",
) -> None:
    """Save all final results for an experiment."""
    final = {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
        "expected": expected,
        "observed": observed,
        "pass_fail": pass_fail,
        "results": results,
    }
    if timer:
        final["timing"] = timer.summary()
    if mem_tracker:
        final["memory"] = mem_tracker.summary()

    save_json(final, os.path.join(output_dir, "final_results.json"))
    write_summary_report(
        experiment_name=experiment_name,
        results=final,
        output_dir=output_dir,
        expected=expected,
        observed=observed,
        pass_fail=pass_fail,
    )
    print(f"\n[OK] Experiment '{experiment_name}' finalized. Results in {output_dir}")


# ---------------------------------------------------------------------------
# Minimal Model Creation (for experiments that need a small trainable model)
# ---------------------------------------------------------------------------

def create_small_transformer_model(
    vocab_size: int = 1000,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 512,
    seq_len: int = 32,
    batch_size: int = 4,
    device: str = "cuda",
) -> Tuple[Any, Any, Any]:
    """Create a small Transformer model for fast experiments."""
    import torch
    import torch.nn as nn

    class SmallTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            emb = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
            encoded = self.encoder(emb)
            logits = self.head(encoded)
            return logits

    model = SmallTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Random data
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return model, optimizer, (inputs, targets)


def create_small_moe_model(
    vocab_size: int = 1000,
    d_model: int = 256,
    n_experts: int = 4,
    top_k: int = 2,
    num_layers: int = 2,
    device: str = "cuda",
) -> Tuple[Any, Any, Any]:
    """Create a small MoE (Mixture of Experts) model for cache experiments."""
    import torch
    import torch.nn as nn

    class Expert(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )

        def forward(self, x):
            return self.net(x)

    class MoELayer(nn.Module):
        def __init__(self, d_model, n_experts, top_k):
            super().__init__()
            self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])
            self.gate = nn.Linear(d_model, n_experts)
            self.top_k = top_k

        def forward(self, x):
            gate_logits = self.gate(x)
            topk_vals, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_vals, dim=-1)
            output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                mask = (topk_indices == i).any(dim=-1)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = expert(expert_input)
                    weight_mask = (topk_indices[mask] == i)
                    weights = topk_weights[mask]
                    weighted_output = expert_output * weights[weight_mask].unsqueeze(-1)
                    output[mask] += weighted_output
            return output

    class SmallMoEModel(nn.Module):
        def __init__(self, vocab_size, d_model, n_experts, top_k, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.moe_layers = nn.ModuleList([
                MoELayer(d_model, n_experts, top_k) for _ in range(num_layers)
            ])
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            emb = self.embedding(x)
            for layer in self.moe_layers:
                emb = emb + layer(emb)
            return self.head(emb)

    model = SmallMoEModel(vocab_size, d_model, n_experts, top_k, num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(0, vocab_size, (4, 32), device=device)
    targets = torch.randint(0, vocab_size, (4, 32), device=device)
    return model, optimizer, (inputs, targets)
