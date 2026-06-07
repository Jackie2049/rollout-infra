#!/usr/bin/env python3
"""FP8 Quantization Visualization — RTX 4090

Visualizes FP8 E4M3/E5M2 quantization formats, compares DelayedScaling
vs Float8CurrentScaling, and demonstrates the scaling factor dynamics.

This is a CPU-only educational tool (no GPU required).

Usage: python tools/fp8_quantization_visualizer.py
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path

import torch
import numpy as np


# ============================================================
# FP8 Format Definitions
# ============================================================

class FP8Format:
    """FP8 format specification (matching TransformerEngine recipe/__init__.py)."""

    E4M3 = {
        "name": "E4M3",
        "sign_bits": 1,
        "exp_bits": 4,
        "mantissa_bits": 3,
        "bias": 7,  # exponent bias
        "max_exp": 8,  # max normal exponent (no inf/nan)
        "max_value": 448.0,
        "min_subnormal": 2**(-10),  # 2^(bias - max_exp - mantissa_bits)
        "description": "Forward专用, 精度高, 范围小",
    }

    E5M2 = {
        "name": "E5M2",
        "sign_bits": 1,
        "exp_bits": 5,
        "mantissa_bits": 2,
        "bias": 15,  # exponent bias
        "max_exp": 28,  # max normal exponent
        "max_value": 57344.0,  # 2^(max_exp-bias) * (1 + 2^-1 + 2^-2)
        "min_subnormal": 2**(-16),
        "description": "Backward专用, 范围大, 精度低",
    }

    HYBRID = {
        "name": "HYBRID",
        "forward": "E4M3",
        "backward": "E5M2",
        "description": "Forward=E4M3, Backward=E5M2 → 精度+范围兼顾",
    }


def fp8_e4m3_to_float(sign, exponent, mantissa):
    """Convert FP8 E4M3 (1-4-3) to float value."""
    bias = 7
    if exponent == 0:
        # Subnormal: value = (-1)^sign * 2^(1-bias) * 0.mantissa
        value = (-1)**sign * 2**(1 - bias) * mantissa / 8
    elif exponent <= 8:
        # Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mantissa/8)
        value = (-1)**sign * 2**(exponent - bias) * (1 + mantissa / 8)
    else:
        # Special: NaN/Inf (E4M3 uses exp=15 for NaN only, no Inf)
        return float('nan') if exponent == 15 else float('inf')
    return value


def fp8_e5m2_to_float(sign, exponent, mantissa):
    """Convert FP8 E5M2 (1-5-2) to float value."""
    bias = 15
    if exponent == 0:
        # Subnormal: value = (-1)^sign * 2^(1-bias) * 0.mantissa
        value = (-1)**sign * 2**(1 - bias) * mantissa / 4
    elif exponent <= 30:
        # Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mantissa/4)
        value = (-1)**sign * 2**(exponent - bias) * (1 + mantissa / 4)
    else:
        # Special: exp=31, mantissa=0→Inf, mantissa≠0→NaN
        return float('inf') if mantissa == 0 else float('nan')
    return value


def generate_fp8_values(format_spec, num_samples=100):
    """Generate all representable FP8 values for a format."""
    values = []
    sign_bits = format_spec["sign_bits"]
    exp_bits = format_spec["exp_bits"]
    mant_bits = format_spec["mantissa_bits"]

    for sign in range(2**sign_bits):
        for exp in range(2**exp_bits):
            for mant in range(2**mant_bits):
                if format_spec["name"] == "E4M3":
                    val = fp8_e4m3_to_float(sign, exp, mant)
                elif format_spec["name"] == "E5M2":
                    val = fp8_e5m2_to_float(sign, exp, mant)
                if not math.isnan(val) and not math.isinf(val):
                    values.append(val)

    values.sort()
    return values


def analyze_fp8_distribution(format_spec):
    """Analyze the distribution of FP8 representable values."""
    values = generate_fp8_values(format_spec)
    positive = [v for v in values if v > 0]

    # Bucket analysis
    buckets = {}
    for v in positive:
        # Find the power of 2 range
        if v < 1:
            exp_range = f"2^({int(math.log2(v))}) to 2^({int(math.log2(v))+1})"
        else:
            exp_range = f"2^({int(math.log2(v))}) to 2^({int(math.log2(v))+1})"
        if exp_range not in buckets:
            buckets[exp_range] = []
        buckets[exp_range].append(v)

    return {
        "name": format_spec["name"],
        "total_values": len(values),
        "positive_values": len(positive),
        "max_value": max(positive) if positive else 0,
        "min_positive": min(positive) if positive else 0,
        "dynamic_range": max(positive) / min(positive) if positive else 0,
        "buckets": {k: len(v) for k, v in buckets.items()},
    }


def simulate_delayed_scaling(tensor_values, margin=0, amax_history_len=1024, num_steps=10):
    """Simulate DelayedScaling quantization over multiple steps."""
    FP8_MAX = 448.0  # E4M3 max for forward

    history = []
    results = []

    for step in range(num_steps):
        # Current tensor's actual amax
        current_amax = max(abs(v) for v in tensor_values[step])

        # Use previous step's amax for scaling (delayed!)
        if step == 0:
            # First step: use current amax (no history yet)
            scale_amax = current_amax
        else:
            # Use max of history window
            scale_amax = max(history[-amax_history_len:]) if history else current_amax

        # Compute scaling factor
        # scale = FP8_MAX / amax / (2^margin)
        scale = FP8_MAX / scale_amax / (2**margin)

        # Quantize each value
        quantized = []
        for v in tensor_values[step]:
            q = round(v * scale)
            q = max(-FP8_MAX, min(FP8_MAX, q))  # clamp
            deq = q / scale  # dequantize
            quantized.append(deq)

        # Compute error
        errors = [abs(v - q) for v, q in zip(tensor_values[step], quantized)]
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)

        # Update history
        history.append(current_amax)

        results.append({
            "step": step,
            "current_amax": current_amax,
            "scale_amax": scale_amax,
            "scale_factor": scale,
            "max_error": max_error,
            "mean_error": mean_error,
            "relative_error_pct": mean_error / max(abs(v) for v in tensor_values[step]) * 100 if tensor_values[step] else 0,
        })

    return results


def simulate_current_scaling(tensor_values, num_steps=10, epsilon=0.0):
    """Simulate Float8CurrentScaling quantization over multiple steps."""
    FP8_MAX = 448.0  # E4M3 max for forward

    results = []

    for step in range(num_steps):
        # Current tensor's amax (no delay!)
        current_amax = max(abs(v) for v in tensor_values[step])
        scale_amax = current_amax + epsilon  # epsilon prevents zero amax

        # Compute scaling factor
        scale = FP8_MAX / scale_amax

        # Quantize each value
        quantized = []
        for v in tensor_values[step]:
            q = round(v * scale)
            q = max(-FP8_MAX, min(FP8_MAX, q))  # clamp
            deq = q / scale  # dequantize
            quantized.append(deq)

        # Compute error
        errors = [abs(v - q) for v, q in zip(tensor_values[step], quantized)]
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)

        results.append({
            "step": step,
            "current_amax": current_amax,
            "scale_factor": scale,
            "max_error": max_error,
            "mean_error": mean_error,
            "relative_error_pct": mean_error / max(abs(v) for v in tensor_values[step]) * 100 if tensor_values[step] else 0,
        })

    return results


def generate_synthetic_training_data(num_steps=10, num_elements=100, max_val=10.0):
    """Generate synthetic tensor values simulating training dynamics."""
    torch.manual_seed(42)
    tensors = []
    for step in range(num_steps):
        # Gradually changing distribution (simulating training)
        scale = max_val * (1 + 0.3 * math.sin(step * 0.5))  # oscillating amax
        tensor = torch.randn(num_elements) * scale
        tensors.append(tensor.tolist())
    return tensors


def run_analysis(output_dir="results"):
    """Run complete FP8 quantization analysis."""
    results = {}

    print("=" * 60)
    print("FP8 Quantization Visualizer — RTX 4090 (Educational)")
    print("=" * 60)

    # 1. FP8 format analysis
    print("\n--- FP8 Format Analysis ---")
    e4m3_analysis = analyze_fp8_distribution(FP8Format.E4M3)
    e5m2_analysis = analyze_fp8_distribution(FP8Format.E5M2)
    results["e4m3_analysis"] = e4m3_analysis
    results["e5m2_analysis"] = e5m2_analysis

    print(f"  E4M3: {e4m3_analysis['positive_values']} positive values, "
          f"max={e4m3_analysis['max_value']}, "
          f"dynamic_range={e4m3_analysis['dynamic_range']:.1f}x")
    print(f"  E5M2: {e5m2_analysis['positive_values']} positive values, "
          f"max={e5m2_analysis['max_value']}, "
          f"dynamic_range={e5m2_analysis['dynamic_range']:.1f}x")
    print(f"  HYBRID: forward=E4M3(max=448), backward=E5M2(max=57344)")

    # Key comparison
    print(f"\n  E4M3 vs E5M2:")
    print(f"    Range: E5M2覆盖{e5m2_analysis['dynamic_range']/e4m3_analysis['dynamic_range']:.1f}x更大范围")
    print(f"    精度: E4M3有3bit mantissa→8级精度, E5M2有2bit→4级精度")
    print(f"    用途: E4M3→forward(精度优先), E5M2→backward(范围优先)")

    # 2. DelayedScaling vs CurrentScaling simulation
    print("\n--- Scaling Recipe Simulation ---")
    tensors = generate_synthetic_training_data(num_steps=10, num_elements=100)

    ds_results = simulate_delayed_scaling(tensors, margin=0)
    cs_results = simulate_current_scaling(tensors, epsilon=0.001)

    results["delayed_scaling"] = ds_results
    results["current_scaling"] = cs_results

    print(f"\n  DelayedScaling (margin=0):")
    for r in ds_results[:5]:
        print(f"    Step {r['step']}: amax={r['current_amax']:.2f}, "
              f"scale={r['scale_factor']:.4f}, "
              f"rel_error={r['relative_error_pct']:.2f}%")

    print(f"\n  Float8CurrentScaling (epsilon=0.001):")
    for r in cs_results[:5]:
        print(f"    Step {r['step']}: amax={r['current_amax']:.2f}, "
              f"scale={r['scale_factor']:.4f}, "
              f"rel_error={r['relative_error_pct']:.2f}%")

    # 3. Margin effect
    print("\n--- Margin Effect (DelayedScaling) ---")
    margin_results = {}
    for margin in [0, 1, 2, 3, 4]:
        mr = simulate_delayed_scaling(tensors, margin=margin)
        avg_error = sum(r['relative_error_pct'] for r in mr) / len(mr)
        margin_results[margin] = {"avg_relative_error_pct": avg_error, "results": mr}
        print(f"    margin={margin}: avg_rel_error={avg_error:.2f}%")
    results["margin_sweep"] = margin_results

    # 4. Dynamic range comparison
    print("\n--- Dynamic Range Comparison ---")
    # FP8 vs BF16 vs FP16 vs INT8
    formats = {
        "FP8 E4M3": {"max": 448, "min_pos": 2**(-9), "bits": 8},
        "FP8 E5M2": {"max": 57344, "min_pos": 2**(-16), "bits": 8},
        "FP16": {"max": 65504, "min_pos": 2**(-24), "bits": 16},
        "BF16": {"max": 3.39e38, "min_pos": 2**(-126), "bits": 16},
        "INT8": {"max": 127, "min_pos": 1, "bits": 8},
    }
    for name, spec in formats.items():
        dr = spec["max"] / spec["min_pos"]
        print(f"    {name}: max={spec['max']}, min_pos={spec['min_pos']:.6f}, "
              f"dynamic_range={dr:.1f}x, bits={spec['bits']}")
    results["format_comparison"] = formats

    # 5. RTX 4099 recipe availability
    print("\n--- RTX 4090 Recipe Availability ---")
    recipes = {
        "DelayedScaling": {"sm_required": "8.9+", "cuda_required": "12.1+", "available": True},
        "Float8CurrentScaling": {"sm_required": "8.9+", "cuda_required": "12.1+", "available": True},
        "MXFP8BlockScaling": {"sm_required": "10.0+ (Blackwell)", "cuda_required": "any", "available": False},
        "Float8BlockScaling": {"sm_required": "9.0+ (Hopper)", "cuda_required": "12.9+", "available": False},
        "NVFP4BlockScaling": {"sm_required": "10.0+ (Blackwell)", "cuda_required": "any", "available": False},
    }
    for name, spec in recipes.items():
        status = "✅ AVAILABLE" if spec["available"] else "❌ NOT AVAILABLE"
        print(f"    {name}: {status} (needs SM {spec['sm_required']}, CUDA {spec['cuda_required']})")
    results["rtx4090_recipe_availability"] = recipes

    # Save results
    output_path = Path(output_dir) / "fp8_quantization_visualizer.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FP8 Quantization Visualizer")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    run_analysis(args.output)


if __name__ == "__main__":
    main()