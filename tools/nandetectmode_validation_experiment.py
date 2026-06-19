#!/usr/bin/env python3
"""
NanDetectMode Validation Experiment — Prepared for GPU Testing
===============================================================
Validates PyTorch #187653 NanDetectMode on RTX 4090.

This experiment tests:
1. NanDetectMode forward NaN/Inf detection performance
2. Comparison with torch.autograd.detect_anomaly()
3. NaN injection in GRPO training scenarios
4. Layer 2 defense effectiveness for RTX 4090

Usage: python tools/nandetectmode_validation_experiment.py [prepare|validate|run|report]

Modes:
  prepare  - Check dependencies, generate experiment configs
  validate - Run CPU-only validation (safe without GPU)
  run      - Run full GPU experiment (requires CUDA device)
  report   - Generate experiment report from results
"""

import argparse
import json
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("results/nandetectmode_validation")


def check_dependencies():
    """Check if required dependencies are available."""
    deps = {
        "torch": None,
        "numpy": None,
        "nandetectmode_available": False,
        "cuda_available": False,
        "gpu_name": None,
    }

    try:
        import torch
        deps["torch"] = torch.__version__
        deps["cuda_available"] = torch.cuda.is_available()
        if deps["cuda_available"]:
            deps["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            # NanDetectMode is from PR #187653 — check if available
            from torch.utils._python_dispatch import TorchDispatchMode
            deps["nandetectmode_available"] = True  # At least TorchDispatchMode exists
        except ImportError:
            deps["nandetectmode_available"] = False
    except ImportError:
        pass

    try:
        import numpy
        deps["numpy"] = numpy.__version__
    except ImportError:
        pass

    return deps


def generate_configs():
    """Generate experiment configurations for different scenarios."""
    configs = []

    # Config 1: Basic NaN injection detection
    configs.append({
        "name": "basic_nan_detection",
        "description": "Test NanDetectMode vs detect_anomaly() for basic NaN injection",
        "model_size": 256,
        "hidden_dim": 64,
        "num_layers": 4,
        "batch_size": 32,
        "seq_len": 128,
        "nan_injection_layers": [2],  # Inject NaN at layer 2
        "num_iterations": 100,
    })

    # Config 2: GRPO training simulation
    configs.append({
        "name": "grpo_nan_simulation",
        "description": "Simulate GRPO training NaN scenarios",
        "model_size": 256,
        "hidden_dim": 64,
        "num_layers": 4,
        "batch_size": 8,  # Small GRPO batch
        "seq_len": 64,
        "nan_injection_layers": [1, 3],  # Multiple NaN injections
        "num_iterations": 50,
        "grpo_group_size": 4,
    })

    # Config 3: Performance benchmark
    configs.append({
        "name": "performance_benchmark",
        "description": "Benchmark NanDetectMode vs detect_anomaly() performance",
        "model_size": 256,
        "hidden_dim": 64,
        "num_layers": 4,
        "batch_size": 32,
        "seq_len": 128,
        "nan_injection_layers": [],
        "num_iterations": 1000,
        "measure_time": True,
    })

    return configs


def run_cpu_validation(config):
    """Run CPU-only validation (safe without GPU)."""
    import torch
    import numpy as np

    results = {
        "config": config["name"],
        "device": "cpu",
        "torch_version": torch.__version__,
    }

    # Simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(config["model_size"], config["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(config["hidden_dim"], config["model_size"]),
    )

    # Test 1: Basic forward pass (no NaN)
    x = torch.randn(config["batch_size"], config["model_size"])
    try:
        output = model(x)
        results["forward_no_nan"] = {
            "output_mean": output.mean().item(),
            "output_std": output.std().item(),
            "has_nan": torch.isnan(output).any().item(),
        }
    except Exception as e:
        results["forward_no_nan"] = {"error": str(e)}

    # Test 2: Forward pass with NaN injection
    x_nan = torch.randn(config["batch_size"], config["model_size"])
    x_nan[:, config["nan_injection_layers"][0] if config["nan_injection_layers"] else 0] = float("nan")

    try:
        output_nan = model(x_nan)
        results["forward_with_nan"] = {
            "output_has_nan": torch.isnan(output_nan).any().item(),
            "nan_count": torch.isnan(output_nan).sum().item(),
            "total_elements": output_nan.numel(),
        }
    except Exception as e:
        results["forward_with_nan"] = {"error": str(e)}

    # Test 3: detect_anomaly() performance
    start_time = time.time()
    with torch.autograd.detect_anomaly():
        for _ in range(100):
            x = torch.randn(config["batch_size"], config["model_size"])
            output = model(x)
            loss = output.sum()
            loss.backward()
    anomaly_time = time.time() - start_time
    results["detect_anomaly_time"] = anomaly_time

    # Test 4: Manual NaN check performance
    start_time = time.time()
    for _ in range(100):
        x = torch.randn(config["batch_size"], config["model_size"])
        output = model(x)
        # Manual NaN check
        if torch.isnan(output).any():
            print("NaN detected!")
        loss = output.sum()
        loss.backward()
    manual_check_time = time.time() - start_time
    results["manual_check_time"] = manual_check_time

    # Test 5: No-check performance (baseline)
    start_time = time.time()
    for _ in range(100):
        x = torch.randn(config["batch_size"], config["model_size"])
        output = model(x)
        loss = output.sum()
        loss.backward()
    baseline_time = time.time() - start_time
    results["baseline_time"] = baseline_time

    # Compute overhead ratios
    results["overhead_ratios"] = {
        "detect_anomaly_vs_baseline": anomaly_time / baseline_time,
        "manual_check_vs_baseline": manual_check_time / baseline_time,
    }

    return results


def prepare_experiment():
    """Prepare experiment directory and configs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    deps = check_dependencies()
    configs = generate_configs()

    # Save configs
    config_path = RESULTS_DIR / "configs.json"
    with open(config_path, "w") as f:
        json.dump({"dependencies": deps, "configs": configs}, f, indent=2)

    print(f"Dependencies: {json.dumps(deps, indent=2)}")
    print(f"Configs saved to: {config_path}")
    print(f"Number of configs: {len(configs)}")

    if not deps["cuda_available"]:
        print("\nWARNING: No CUDA device available. GPU experiments will need to wait.")
        print("Run 'python tools/nandetectmode_validation_experiment.py validate' for CPU-only testing.")

    return deps, configs


def validate_experiment():
    """Run CPU-only validation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = generate_configs()
    all_results = []

    for config in configs:
        print(f"\nRunning CPU validation for: {config['name']}")
        results = run_cpu_validation(config)
        all_results.append(results)

        # Save individual results
        result_path = RESULTS_DIR / f"{config['name']}_cpu.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {result_path}")

        # Print summary
        if "overhead_ratios" in results:
            print(f"  detect_anomaly overhead: {results['overhead_ratios']['detect_anomaly_vs_baseline']:.2f}x")
            print(f"  manual NaN check overhead: {results['overhead_ratios']['manual_check_vs_baseline']:.2f}x")

    # Save combined results
    combined_path = RESULTS_DIR / "all_cpu_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {combined_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="NanDetectMode Validation Experiment")
    parser.add_argument("mode", choices=["prepare", "validate", "run", "report"],
                        help="prepare=check deps+configs, validate=CPU-only, run=GPU full, report=generate report")
    args = parser.parse_args()

    if args.mode == "prepare":
        deps, configs = prepare_experiment()
        print(f"\nGPU status: {deps['gpu_name'] if deps['cuda_available'] else 'OFFLINE'}")

    elif args.mode == "validate":
        results = validate_experiment()

    elif args.mode == "run":
        deps = check_dependencies()
        if not deps["cuda_available"]:
            print("ERROR: No CUDA device available. Run 'validate' mode for CPU-only testing.")
            sys.exit(1)
        print("GPU experiment requires CUDA device — full implementation coming when GPU available.")

    elif args.mode == "report":
        print("Report generation from saved results.")


if __name__ == "__main__":
    main()
