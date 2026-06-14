#!/usr/bin/env python3
"""
Megatron Pipeline Parallelism Bubble Simulator

Simulates pipeline bubble sizes for different PP configurations:
- GPipe (naive)
- 1F1B (non-interleaved)
- 1F1B (interleaved with VP)
- With overlap_p2p_comm

Also models RTX 4090 PCIe P2P communication overhead.

Usage:
    python3 megatron_pp_bubble_simulator.py
    python3 megatron_pp_bubble_simulator.py --pp 4 --vp 2 --microbatches 16
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Results output directory
RESULTS_DIR = Path(__file__).parent.parent / "results"

def calc_gpipe_bubble(pp, m):
    """GPipe bubble fraction: (P-1)/(P-1+M)"""
    if m == 0:
        return 1.0
    return (pp - 1) / (pp - 1 + m)

def calc_1f1b_bubble(pp, m):
    """1F1B (non-interleaved) bubble fraction: min(1.0, 2*(P-1)/M)
    When M < 2*(PP-1), pipeline can't be fully utilized → max bubble = 1.0"""
    if m == 0:
        return 1.0
    raw = 2 * (pp - 1) / m
    return min(1.0, raw)

def calc_interleaved_bubble(pp, m, vp):
    """Interleaved 1F1B bubble fraction: min(1.0, 2*(P-1)/(M*VP))"""
    if m == 0:
        return 1.0
    raw = 2 * (pp - 1) / (m * vp)
    return min(1.0, raw)

def calc_warmup_steps(pp, rank, m):
    """1F1B warmup microbatches for given rank"""
    return min(pp - rank - 1, m)

def calc_interleaved_warmup(pp, rank, vp, microbatch_group_size):
    """Interleaved 1F1B warmup steps"""
    warmup = (pp - rank - 1) * 2
    warmup += (vp - 1) * microbatch_group_size
    return warmup

def calc_memory_peak_gpipe(pp, m, activation_per_microbatch):
    """GPipe peak memory: M * activation (all microbatch activations stored)"""
    return m * activation_per_microbatch

def calc_memory_peak_1f1b(pp, rank, m, activation_per_microbatch):
    """1F1B peak memory: num_warmup * activation (only warmup activations stored)"""
    num_warmup = calc_warmup_steps(pp, rank, m)
    return num_warmup * activation_per_microbatch

def calc_p2p_traffic_non_interleaved(pp, m, activation_bytes, bandwidth_gbps):
    """P2P traffic for non-interleaved 1F1B
    Each steady iteration: 2 P2P ops (send_fwd_recv_bwd + send_bwd_recv_fwd)
    Warmup: 1 P2P per step (send_fwd or recv_fwd)
    Cooldown: 1 P2P per step"""
    # Forward pass: each microbatch crosses PP-1 boundaries
    fwd_p2p = m * (pp - 1) * activation_bytes
    # Backward pass: same
    bwd_p2p = m * (pp - 1) * activation_bytes
    total_bytes = fwd_p2p + bwd_p2p
    # With batched_p2p: send+recv in one call → effectively halve round-trips
    total_time_sec = total_bytes / (bandwidth_gbps * 1e9 / 8)  # convert Gbps to GB/s
    return total_bytes, total_time_sec

def calc_p2p_traffic_interleaved(pp, m, vp, activation_bytes, bandwidth_gbps):
    """Interleaved P2P traffic: 2x compared to non-interleaved"""
    non_interleaved_bytes, _ = calc_p2p_traffic_non_interleaved(pp, m, activation_bytes, bandwidth_gbps)
    # Interleaved doubles P2P because each microbatch crosses VP*PP boundaries
    total_bytes = non_interleaved_bytes * vp
    total_time_sec = total_bytes / (bandwidth_gbps * 1e9 / 8)
    return total_bytes, total_time_sec

def simulate_pp_configurations():
    """Run comprehensive PP simulation across multiple configurations"""
    results = {}

    # Configuration sweeps
    pp_sizes = [2, 4, 8, 16]
    microbatch_counts = [4, 8, 16, 32, 64]
    vp_sizes = [1, 2, 4]  # VP=1 means non-interleaved

    # Hardware parameters
    hw_configs = {
        "NVLink_A100": {"bandwidth_gbps": 300, "name": "A100 NVLink 300GB/s"},
        "NVLink_H100": {"bandwidth_gbps": 450, "name": "H100 NVLink 450GB/s"},
        "PCIe_RTX4090": {"bandwidth_gbps": 12, "name": "RTX 4090 PCIe 12GB/s"},
        "RoCE_200Gbps": {"bandwidth_gbps": 200, "name": "RoCE v2 200Gbps (inter-node)"},
    }

    # Model parameters (7B model-like)
    activation_bytes = 2 * 1024 * 1024  # ~2MB per microbatch activation (BF16, seq=2048, hidden=4096)

    # Sweep 1: Bubble fractions across PP and M
    bubble_sweep = {}
    for pp in pp_sizes:
        for m in microbatch_counts:
            gpipe = calc_gpipe_bubble(pp, m)
            one_f1b_bubble = calc_1f1b_bubble(pp, m)
            key = f"PP={pp}_M={m}"
            bubble_sweep[key] = {
                "pp": pp, "m": m,
                "gpipe_bubble_pct": round(gpipe * 100, 2),
                "1f1b_bubble_pct": round(one_f1b_bubble * 100, 2),
                "1f1b_reduction_vs_gpipe": round((gpipe - one_f1b_bubble) / gpipe * 100, 1) if gpipe > 0 else 0,
            }
    results["bubble_sweep"] = bubble_sweep

    # Sweep 2: Interleaved bubble reduction
    interleaved_sweep = {}
    for pp in [2, 4, 8]:
        for vp in [2, 4]:
            for m in [8, 16, 32]:
                non_interleaved = calc_1f1b_bubble(pp, m)
                interleaved = calc_interleaved_bubble(pp, m, vp)
                key = f"PP={pp}_VP={vp}_M={m}"
                interleaved_sweep[key] = {
                    "pp": pp, "vp": vp, "m": m,
                    "non_interleaved_pct": round(non_interleaved * 100, 2),
                    "interleaved_pct": round(interleaved * 100, 2),
                    "reduction_factor": round(non_interleaved / interleaved, 2) if interleaved > 0 else 0,
                }
    results["interleaved_sweep"] = interleaved_sweep

    # Sweep 3: P2P communication overhead by hardware
    p2p_sweep = {}
    for pp in [2, 4, 8]:
        for m in [8, 16]:
            for hw_name, hw in hw_configs.items():
                bytes_non, time_non = calc_p2p_traffic_non_interleaved(
                    pp, m, activation_bytes, hw["bandwidth_gbps"])
                bytes_int, time_int = calc_p2p_traffic_interleaved(
                    pp, m, 2, activation_bytes, hw["bandwidth_gbps"])
                key = f"PP={pp}_M={m}_{hw_name}"
                p2p_sweep[key] = {
                    "pp": pp, "m": m, "hardware": hw["name"],
                    "non_interleaved_bytes_mb": round(bytes_non / 1e6, 2),
                    "non_interleaved_time_ms": round(time_non * 1000, 3),
                    "interleaved_VP2_bytes_mb": round(bytes_int / 1e6, 2),
                    "interleaved_VP2_time_ms": round(time_int * 1000, 3),
                }
    results["p2p_sweep"] = p2p_sweep

    # Sweep 4: Warmup and memory peak per rank
    warmup_sweep = {}
    for pp in [4, 8]:
        for m in [16, 32]:
            for rank in range(pp):
                num_warmup = calc_warmup_steps(pp, rank, m)
                peak_mem = calc_memory_peak_1f1b(pp, rank, m, activation_bytes)
                key = f"PP={pp}_M={m}_rank={rank}"
                warmup_sweep[key] = {
                    "pp": pp, "m": m, "rank": rank,
                    "warmup_microbatches": num_warmup,
                    "steady_microbatches": m - num_warmup,
                    "cooldown_microbatches": num_warmup,
                    "peak_activation_mb": round(peak_mem / 1e6, 2),
                }
    results["warmup_sweep"] = warmup_sweep

    # Sweep 5: RTX 4090 PCIe scaling analysis
    rtx4090_analysis = {
        "hardware": "RTX 4090 24GB, PCIe Gen4 x16",
        "p2p_bandwidth": "12 GB/s (PCIe, no NVLink)",
        "conclusions": [
            "PP=2, M=4: bubble=50% + PCIe P2P 2ms → terrible",
            "PP=2, M=16: bubble=12.5% + P2P overhead still significant",
            "Interleaved PP=2,VP=2: bubble halved but 2x P2P → PCIe worse",
            "RTX 4090不适合PP → 单GPU最优(PP=1)",
            "NVLink GPU(A100/H100): PP可行因NVLink 300-450GB/s",
        ],
    }
    # Specific calculations
    for pp in [2, 4]:
        for m in [4, 8, 16, 32]:
            bubble = calc_1f1b_bubble(pp, m)
            bytes_total, time_total = calc_p2p_traffic_non_interleaved(
                pp, m, activation_bytes, 12)  # 12 GB/s PCIe
            rtx4090_analysis[f"PP={pp}_M={m}"] = {
                "bubble_pct": round(bubble * 100, 2),
                "p2p_time_ms": round(time_total * 1000, 3),
                "usable": bubble < 0.1 and time_total < 0.01,  # <10% bubble and <10ms P2P
            }
    results["rtx4090_analysis"] = rtx4090_analysis

    # Summary table
    summary = {
        "GPipe": "Bubble=(P-1)/(P-1+M), peak memory=M×activation, simplest but worst",
        "1F1B": "Bubble=2(P-1)/M, peak=warmup×activation, same bubble but lower memory",
        "Interleaved": "Bubble=2(P-1)/(M×VP), 2×P2P traffic, needs NVLink",
        "recommendation": {
            "RTX 4090": "PP=1 only (PCIe bottleneck)",
            "A100 NVLink": "PP≤4 with M≥16 (NVLink handles P2P)",
            "H100 NVLink": "PP≤8 with M≥32 (NVLink + higher bandwidth)",
            "Multi-node": "PP≤16 with RoCE inter-node + NVLink intra-node",
        },
    }
    results["summary"] = summary

    return results

def main():
    parser = argparse.ArgumentParser(description="Megatron PP Bubble Simulator")
    parser.add_argument("--pp", type=int, default=0, help="Pipeline parallel size (0=sweep)")
    parser.add_argument("--vp", type=int, default=0, help="Virtual pipeline size (0=sweep)")
    parser.add_argument("--microbatches", type=int, default=0, help="Number of microbatches (0=sweep)")
    parser.add_argument("--output", type=str, default="", help="Output file path")
    args = parser.parse_args()

    results = simulate_pp_configurations()

    # If specific config requested, also add detailed calculation
    if args.pp > 0:
        m = args.microbatches if args.microbatches > 0 else 16
        vp = args.vp if args.vp > 0 else 1
        specific = {
            "pp": args.pp, "m": m, "vp": vp,
            "gpipe_bubble_pct": round(calc_gpipe_bubble(args.pp, m) * 100, 2),
            "1f1b_bubble_pct": round(calc_1f1b_bubble(args.pp, m) * 100, 2),
            "interleaved_bubble_pct": round(calc_interleaved_bubble(args.pp, m, vp) * 100, 2) if vp > 1 else None,
        }
        # Per-rank warmup
        specific["per_rank"] = {}
        for rank in range(args.pp):
            specific["per_rank"][str(rank)] = {
                "warmup": calc_warmup_steps(args.pp, rank, m),
                "steady": m - calc_warmup_steps(args.pp, rank, m),
                "cooldown": calc_warmup_steps(args.pp, rank, m),
            }
        results["specific_config"] = specific

    # Output
    output_path = Path(args.output) if args.output else RESULTS_DIR / "megatron_pp_bubble_simulator.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("=" * 60)
    print("Megatron Pipeline Parallelism Bubble Simulator")
    print("=" * 60)

    print("\n--- Bubble Fraction Comparison ---")
    print(f"{'Config':<20} {'GPipe':>10} {'1F1B':>10} {'Interleaved':>12}")
    print("-" * 54)
    for pp in [2, 4, 8, 16]:
        for m in [8, 16, 32]:
            gpipe = calc_gpipe_bubble(pp, m)
            bubble_1f1b = calc_1f1b_bubble(pp, m)
            interleaved = calc_interleaved_bubble(pp, m, 2)
            print(f"PP={pp} M={m:<4}     {gpipe*100:>8.1f}% {bubble_1f1b*100:>8.1f}% {interleaved*100:>10.1f}%")

    print("\n--- RTX 4090 PCIe Impact ---")
    print(f"{'Config':<20} {'Bubble':>10} {'P2P Time':>10} {'Usable?':>10}")
    print("-" * 52)
    for pp in [2, 4]:
        for m in [4, 8, 16, 32]:
            bubble = calc_1f1b_bubble(pp, m)
            _, time = calc_p2p_traffic_non_interleaved(pp, m, 2*1024*1024, 12)
            usable = "✗" if bubble > 0.1 or time > 0.01 else "✓"
            print(f"PP={pp} M={m:<4}     {bubble*100:>8.1f}% {time*1000:>8.3f}ms {usable:>10}")

    print("\n--- Key Findings ---")
    print("1. GPipe bubble = (P-1)/(P-1+M) → worst, but simplest")
    print("2. 1F1B bubble = 2(P-1)/M → same fraction, lower memory peak")
    print("3. Interleaved bubble = 2(P-1)/(M×VP) → VP× reduction, 2× P2P")
    print("4. RTX 4090 PCIe 12GB/s → PP unusable → single GPU optimal")
    print("5. NVLink 300-450GB/s → PP viable for A100/H100 clusters")

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
