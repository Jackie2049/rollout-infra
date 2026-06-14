#!/usr/bin/env python3
"""Parameter Sharding Strategy Simulator.

Simulates and compares 3 parameter sharding approaches:
1. DeepSpeed ZeRO-3 Init Metaclass — "born partitioned"
2. FSDP1 FlatParameter — "flatten then shard"
3. FSDP2 DTensor — "per-parameter shard"

This script simulates a simple model's memory lifecycle through one
training step under each approach, showing memory usage at each phase.

Reference: notebook/fundamentals/zero-algorithm-deep-dive.md
"""

import numpy as np

# ============================================================================
# Model simulation: 3-layer MLP with different-sized layers
# ============================================================================

def simulate_model():
    """Create a simulated model with parameter tensors."""
    # Mimic a 3-layer model: embedding + 2 transformer layers
    layers = [
        ("embed_tokens", (32000, 4096)),    # 128M params = 256MB BF16
        ("self_attn.q_proj", (4096, 4096)),  # 16M params = 32MB BF16
        ("self_attn.k_proj", (4096, 4096)),  # 16M params = 32MB BF16
        ("self_attn.v_proj", (4096, 4096)),  # 16M params = 32MB BF16
        ("self_attn.o_proj", (4096, 4096)),  # 16M params = 32MB BF16
        ("mlp.gate_proj", (4096, 14336)),    # 56M params = 112MB BF16
        ("mlp.up_proj", (4096, 14336)),      # 56M params = 112MB BF16
        ("mlp.down_proj", (14336, 4096)),    # 56M params = 112MB BF16
        ("lm_head", (4096, 32000)),          # 128M params = 256MB BF16
    ]
    total_params = sum(s[0] * s[1] for _, s in layers)
    total_bytes_bf16 = total_params * 2  # 2 bytes per BF16 param
    return layers, total_params, total_bytes_bf16


def simulate_deepspeed_zero3(layers, total_params, total_bytes, N_dp):
    """DeepSpeed ZeRO-3 Init Metaclass simulation.

    Key: params are born partitioned → never exist in full on any rank.
    Each param gets ds_tensor (1/N shard) + ds_status lifecycle.
    """
    print("=" * 70)
    print("DeepSpeed ZeRO-3 — Init Metaclass Injection")
    print("=" * 70)

    # Phase 1: Model Initialization (with zero.Init())
    print("\nPhase 1: Init (params born partitioned)")
    phase1_mem = 0
    param_shards = []
    for name, shape in layers:
        numel = shape[0] * shape[1]
        shard_numel = numel // N_dp + (1 if numel % N_dp else 0)  # ceil division
        shard_bytes = shard_numel * 2  # BF16 shard
        param_shards.append((name, shape, shard_bytes))
        phase1_mem += shard_bytes
        print(f"  {name}: shape={shape}, numel={numel}, shard={shard_numel} params, "
              f"shard_mem={shard_bytes/1e6:.1f}MB")

    print(f"\n  Total after init: {phase1_mem/1e6:.1f}MB (only 1/N shards exist)")
    print(f"  vs DDP init: {total_bytes/1e6:.1f}MB (full params)")
    print(f"  Memory savings: {(1-phase1_mem/total_bytes)*100:.1f}%")

    # Phase 2: Forward — AllGather coalesced per layer
    print("\nPhase 2: Forward (AllGather coalesced + prefetch)")
    peak_forward_mem = phase1_mem  # Start with sharded params
    gathered_mem_per_layer = 0
    for name, shape, shard_bytes in param_shards:
        numel = shape[0] * shape[1]
        full_bytes = numel * 2  # Full param temporarily gathered
        gathered_mem_per_layer += full_bytes

    # With coalescing: 1 AllGather per layer group
    # Without: N params × N AllGather calls
    print(f"  AllGather per layer group: {gathered_mem_per_layer/1e6:.1f}MB gathered")
    print(f"  Communication: 1× AllGather(coalesced) per group")
    print(f"  Prefetch: next layer params gathered during current computation")

    peak_forward_mem += gathered_mem_per_layer  # Temporarily gathered
    print(f"  Peak forward memory: {peak_forward_mem/1e6:.1f}MB")

    # Phase 3: Backward — AllGather + ReduceScatter + partition
    print("\nPhase 3: Backward (AllGather + ReduceScatter + partition)")
    peak_backward_mem = phase1_mem  # Start with sharded
    # Need to gather params again for backward
    peak_backward_mem += gathered_mem_per_layer  # Gathered params
    # ReduceScatter produces 1/N gradient shard
    grad_shard_bytes = phase1_mem  # Same size as param shards
    peak_backward_mem += grad_shard_bytes  # Gradient shards

    print(f"  AllGather: same as forward ({gathered_mem_per_layer/1e6:.1f}MB)")
    print(f"  ReduceScatter: gradients 1/N → {grad_shard_bytes/1e6:.1f}MB per rank")
    print(f"  Peak backward memory: {peak_backward_mem/1e6:.1f}MB")

    # Phase 4: Optimizer step
    print("\nPhase 4: Optimizer (1/N optimizer states)")
    # FP32 master params, momentum, variance — all 1/N
    opt_bytes_per_rank = phase1_mem * 6  # 6× shard_bytes (fp32 master+m+v+grad)
    print(f"  Optimizer states: {opt_bytes_per_rank/1e6:.1f}MB per rank (FP32 1/N)")
    print(f"  vs DDP optimizer: {total_bytes*6/1e6:.1f}MB (FP32 full)")

    # Peak memory
    total_peak = peak_backward_mem + opt_bytes_per_rank
    print(f"\n  Total peak memory: {total_peak/1e6:.1f}MB per rank")
    print(f"  Formula: (16Ψ/N + 4Ψ) = ({phase1_mem*8/1e6:.1f} + "
          f"{gathered_mem_per_layer*2/1e6:.1f}) = {total_peak/1e6:.1f}MB")

    return {
        "phase1": phase1_mem,
        "peak_forward": peak_forward_mem,
        "peak_backward": peak_backward_mem,
        "optimizer": opt_bytes_per_rank,
        "peak_total": total_peak,
        "comm_per_step": 3 * total_bytes,  # 3Ψ = AllGather(fwd) + AllGather(bwd) + ReduceScatter(bwd)
    }


def simulate_fsdp1(layers, total_params, total_bytes, N_dp):
    """FSDP1 FlatParameter simulation.

    Key: params initialized fully → then wrapped by FSDP → flattened into
    FlatParameter buffer → shard into 1/N → unshard/reshard lifecycle.
    """
    print("\n" + "=" * 70)
    print("FSDP1 — FlatParameter Architecture")
    print("=" * 70)

    # Phase 1: Model Initialization (full params first!)
    print("\nPhase 1: Init (params created fully, then flattened)")
    # Model created with full parameters
    init_mem = total_bytes  # All params exist in full initially
    print(f"  Model init (full params): {init_mem/1e6:.1f}MB")

    # Then FSDP wraps → Flatten → Shard → Free originals
    # FlatParameter = concat all params in a module into 1D buffer
    flat_buffer_bytes = total_bytes  # Same total, just reshaped
    shard_bytes = flat_buffer_bytes // N_dp
    # Padding needed for even division
    padding_bytes = (N_dp - (total_params % N_dp)) * 2 if total_params % N_dp else 0
    padded_shard_bytes = (total_params + (N_dp - total_params % N_dp if total_params % N_dp else 0)) * 2 // N_dp

    print(f"  FlatParameter buffer: {flat_buffer_bytes/1e6:.1f}MB (1D concat)")
    print(f"  Padding overhead: {padding_bytes/1e6:.2f}MB ({padding_bytes/(flat_buffer_bytes+padding_bytes)*100:.1f}% waste)")
    print(f"  After shard: {padded_shard_bytes/1e6:.1f}MB per rank (includes padding)")
    print(f"  After freeing originals: {padded_shard_bytes/1e6:.1f}MB per rank")

    phase1_mem = padded_shard_bytes
    # FlatParamHandle state: SHARD (only shards exist)

    # Phase 2: Forward — Unshard (AllGather flat buffer)
    print("\nPhase 2: Forward (Unshard flat buffer)")
    # AllGather flat buffer → all params restored
    # Problem: ALL params gathered at once (even those not needed yet)
    unshard_mem = flat_buffer_bytes + padding_bytes
    peak_forward_mem = phase1_mem + unshard_mem
    print(f"  Unshard: {unshard_mem/1e6:.1f}MB (all params at once)")
    print(f"  Peak forward: {peak_forward_mem/1e6:.1f}MB (shard + unshard)")

    # Phase 3: Reshard after forward (free gathered params)
    print(f"\n  Reshard: free gathered params → back to {phase1_mem/1e6:.1f}MB (shard only)")

    # Phase 4: Backward — Unshard + ReduceScatter + Reshard
    print("\nPhase 3: Backward (Unshard + ReduceScatter)")
    peak_backward_mem = phase1_mem + unshard_mem
    grad_shard_bytes = phase1_mem  # Same size (with padding)
    peak_backward_mem += grad_shard_bytes
    print(f"  Unshard: {unshard_mem/1e6:.1f}MB")
    print(f"  ReduceScatter: {grad_shard_bytes/1e6:.1f}MB per rank")
    print(f"  Peak backward: {peak_backward_mem/1e6:.1f}MB")

    # Phase 5: Optimizer
    opt_bytes = phase1_mem * 6  # FP32 1/N (with padding waste!)
    print(f"\nPhase 4: Optimizer (1/N with padding waste)")
    print(f"  Optimizer states: {opt_bytes/1e6:.1f}MB (includes padding waste!)")

    total_peak = peak_backward_mem + opt_bytes
    print(f"\n  Total peak: {total_peak/1e6:.1f}MB")
    print(f"  Padding waste in optimizer: ~{padding_bytes*6/1e6:.2f}MB")

    return {
        "phase1": phase1_mem,
        "peak_forward": peak_forward_mem,
        "peak_backward": peak_backward_mem,
        "optimizer": opt_bytes,
        "peak_total": total_peak,
        "comm_per_step": 2 * total_bytes,  # AllGather(fwd) + ReduceScatter(bwd) = AllReduce equivalent
        "padding_waste": padding_bytes * 6,  # 6× in optimizer
    }


def simulate_fsdp2(layers, total_params, total_bytes, N_dp):
    """FSDP2 DTensor simulation.

    Key: per-parameter sharding → DTensor replaces param.data → no
    flattening → no padding → torch.compile compatible.
    """
    print("\n" + "=" * 70)
    print("FSDP2 — Per-Parameter DTensor Architecture")
    print("=" * 70)

    # Phase 1: Init (params full, then per-param shard via DTensor)
    print("\nPhase 1: Init (params full → per-param DTensor shard)")
    print(f"  Model init (full params): {total_bytes/1e6:.1f}MB (temporary!)")

    # Per-parameter shard: no flattening, no padding!
    phase1_mem = 0
    param_dtensor_shards = []
    for name, shape in layers:
        numel = shape[0] * shape[1]
        shard_numel = numel // N_dp  # Even division per param
        shard_bytes = shard_numel * 2  # BF16
        param_dtensor_shards.append((name, shape, shard_bytes, shard_numel))
        phase1_mem += shard_bytes
        print(f"  {name}: DTensor shard={shard_numel} params, "
              f"mem={shard_bytes/1e6:.1f}MB (NO padding!)")

    print(f"  After DTensor shard: {phase1_mem/1e6:.1f}MB per rank")
    print(f"  vs FSDP1 shard: includes padding → slightly more")
    print(f"  DTensor preserves param identity → debug-friendly!")

    # Phase 2: Forward — per-param Unshard
    print("\nPhase 2: Forward (per-param AllGather)")
    # Each param gathered individually or in groups
    # FSDP2 can do per-param prefetch (more flexible than FSDP1)
    unshard_mem = total_bytes  # All params gathered
    peak_forward_mem = phase1_mem + unshard_mem
    print(f"  Per-param AllGather: {unshard_mem/1e6:.1f}MB (same total, but per-param)")
    print(f"  Advantage: can prefetch only needed params → more flexible")
    print(f"  Peak forward: {peak_forward_mem/1e6:.1f}MB")

    # Phase 3: Backward — Unshard + ReduceScatter
    print("\nPhase 3: Backward (per-param Unshard + ReduceScatter)")
    peak_backward_mem = phase1_mem + unshard_mem
    grad_shard_bytes = phase1_mem
    peak_backward_mem += grad_shard_bytes
    print(f"  Per-param AllGather: {unshard_mem/1e6:.1f}MB")
    print(f"  Per-param ReduceScatter: {grad_shard_bytes/1e6:.1f}MB")
    print(f"  Peak backward: {peak_backward_mem/1e6:.1f}MB")

    # Phase 4: Optimizer (NO padding waste!)
    opt_bytes = phase1_mem * 6  # FP32 1/N — exact, no padding
    print(f"\nPhase 4: Optimizer (1/N, NO padding)")
    print(f"  Optimizer states: {opt_bytes/1e6:.1f}MB (exact per-param, no waste)")
    print(f"  vs FSDP1: saves padding waste in optimizer")

    total_peak = peak_backward_mem + opt_bytes
    print(f"\n  Total peak: {total_peak/1e6:.1f}MB")
    print(f"  Formula: (16Ψ/N + 4Ψ) — same as ZeRO-3 formula")
    print(f"  But: no padding waste + torch.compile compatible + TP compatible!")

    return {
        "phase1": phase1_mem,
        "peak_forward": peak_forward_mem,
        "peak_backward": peak_backward_mem,
        "optimizer": opt_bytes,
        "peak_total": total_peak,
        "comm_per_step": 2 * total_bytes,  # AllGather + ReduceScatter = AllReduce
        "padding_waste": 0,  # No padding!
    }


def compare_results(ds_results, fsdp1_results, fsdp2_results, total_bytes, N_dp):
    """Print a comparison table of all 3 approaches."""
    print("\n" + "=" * 70)
    print("Comparison: DeepSpeed ZeRO-3 vs FSDP1 vs FSDP2")
    print("=" * 70)

    print(f"\nDP world size: N={N_dp}")
    print(f"Total params BF16: {total_bytes/1e6:.1f}MB")

    headers = ["Phase", "DeepSpeed ZeRO-3", "FSDP1 FlatParam", "FSDP2 DTensor"]
    rows = [
        ("After Init", f"{ds_results['phase1']/1e6:.1f}MB", f"{fsdp1_results['phase1']/1e6:.1f}MB", f"{fsdp2_results['phase1']/1e6:.1f}MB"),
        ("Peak Forward", f"{ds_results['peak_forward']/1e6:.1f}MB", f"{fsdp1_results['peak_forward']/1e6:.1f}MB", f"{fsdp2_results['peak_forward']/1e6:.1f}MB"),
        ("Peak Backward", f"{ds_results['peak_backward']/1e6:.1f}MB", f"{fsdp1_results['peak_backward']/1e6:.1f}MB", f"{fsdp2_results['peak_backward']/1e6:.1f}MB"),
        ("Optimizer", f"{ds_results['optimizer']/1e6:.1f}MB", f"{fsdp1_results['optimizer']/1e6:.1f}MB", f"{fsdp2_results['optimizer']/1e6:.1f}MB"),
        ("Peak Total", f"{ds_results['peak_total']/1e6:.1f}MB", f"{fsdp1_results['peak_total']/1e6:.1f}MB", f"{fsdp2_results['peak_total']/1e6:.1f}MB"),
        ("Comm/Step", f"{ds_results['comm_per_step']/1e6:.1f}MB (3Ψ)", f"{fsdp1_results['comm_per_step']/1e6:.1f}MB (2Ψ)", f"{fsdp2_results['comm_per_step']/1e6:.1e6:.1f}MB (2Ψ)" if False else f"{fsdp2_results['comm_per_step']/1e6:.1f}MB (2Ψ)"),
        ("Padding Waste", "None", f"{fsdp1_results['padding_waste']/1e6:.2f}MB", "None"),
        ("compile Compatible", "❌ (monkey-patch)", "❌ (hook-driven)", "✅ (DTensor native)"),
        ("TP Compatible", "❌ (ZeROOrderedDict)", "❌ (FlatParam)", "✅ (DTensor mesh)"),
        ("Prefetch", "trace+coalesced", "basic", "basic+per-param"),
        ("NVMe Offload", "✅ ZeRO-Infinity", "❌", "❌ (2026 dev)"),
    ]

    # Print as formatted table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    print(f"\n{header_line}")
    print(sep_line)
    for row in rows:
        line = "| " + " | ".join(v.ljust(w) for v, w in zip(row, col_widths)) + " |"
        print(line)

    print(f"\nKey insights:")
    print(f"  1. FSDP2 = FSDP1's memory savings + no padding waste + torch.compile + TP")
    print(f"  2. DeepSpeed ZeRO-3 has highest comm (3Ψ) but NVMe offload unique")
    print(f"  3. FSDP2 comm = 2Ψ (AllGather+ReduceScatter = AllReduce) → less than ZeRO-3!")
    print(f"  4. On RTX 4090 single GPU: all 3 require LoRA (全参数不可能)")
    print(f"  5. Recommendation: FSDP2 + LoRA + BF16 + torch.compile (modern approach)")

    # RTX 4090 analysis
    print(f"\nRTX 4090 24GB analysis (N=1, single GPU):")
    print(f"  DDP peak: {total_bytes*8/1e6:.1f}MB → ❌ exceeds 24GB")
    print(f"  ZeRO-3 N=1: {(total_bytes*8 + total_bytes*4)/1e6:.1f}MB → ❌ worse than DDP!")
    print(f"  FSDP2 N=1: {(total_bytes*8 + total_bytes*4)/1e6:.1f}MB → ❌ same as ZeRO-3 N=1")
    print(f"  → Single GPU: sharding doesn't help! Must use LoRA + CPU Adam")
    print(f"  → LoRA r=16: ~{(total_bytes*0.001*8 + 4*1e6)/1e6:.1f}MB → fits 24GB ✅")


def main():
    print("Parameter Sharding Strategy Simulator")
    print("Simulates DeepSpeed ZeRO-3 vs FSDP1 vs FSDP2 memory lifecycle")
    print()

    layers, total_params, total_bytes = simulate_model()

    N_dp = 8  # Simulate 8-GPU DP (like 8× RTX 4090)

    print(f"Model: 336M params ({total_params/1e6:.0f}M), {total_bytes/1e6:.1f}MB BF16")
    print(f"DP world size: {N_dp}")
    print(f"Per-layer details:")
    for name, shape in layers:
        numel = shape[0] * shape[1]
        print(f"  {name}: {shape} = {numel/1e6:.1f}M params = {numel*2/1e6:.1f}MB")
    print()

    # Simulate all 3 approaches
    ds_results = simulate_deepspeed_zero3(layers, total_params, total_bytes, N_dp)
    fsdp1_results = simulate_fsdp1(layers, total_params, total_bytes, N_dp)
    fsdp2_results = simulate_fsdp2(layers, total_params, total_bytes, N_dp)

    # Compare
    compare_results(ds_results, fsdp1_results, fsdp2_results, total_bytes, N_dp)

    # Also simulate N=1 (single GPU)
    print("\n\n" + "=" * 70)
    print("Single GPU Analysis (N=1) — RTX 4090 scenario")
    print("=" * 70)
    ds_n1 = simulate_deepspeed_zero3(layers, total_params, total_bytes, 1)
    fsdp1_n1 = simulate_fsdp1(layers, total_params, total_bytes, 1)
    fsdp2_n1 = simulate_fsdp2(layers, total_params, total_bytes, 1)
    compare_results(ds_n1, fsdp1_n1, fsdp2_n1, total_bytes, 1)


if __name__ == "__main__":
    main()
