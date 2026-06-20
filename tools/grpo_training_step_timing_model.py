#!/usr/bin/env python3
"""
verl GRPO Training Step Timing Model

Models the complete GRPO training step timing for different GPU configurations,
based on the verl V1 architecture (10-phase lifecycle).

Modes:
  simulate  - Simulate a complete GRPO training step
  compare   - Compare across GPU configs (RTX 4090, A100, H100, A16)
  scale     - Scaling analysis with different dp values
  rtx4090   - RTX 4090 specific comprehensive analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── GPU Specifications ───────────────────────────────────────────────────

GPU_SPECS = {
    "RTX_4090": {"vram": 24, "tflops_bf16": 82.6, "hbm_bw": 1008, "pcie_bw": 32},
    "A100":     {"vram": 80, "tflops_bf16": 312, "hbm_bw": 2039, "nvlink_bw": 600},
    "H100":     {"vram": 80, "tflops_bf16": 990, "hbm_bw": 3352, "nvlink_bw": 900},
    "A16":      {"vram": 16, "tflops_bf16": 70,  "hbm_bw": 1555, "pcie_bw": 32},
}

# ─── Model Specifications ────────────────────────────────────────────────

MODEL_SPECS = {
    "7B":      {"bf16_gib": 14, "hidden": 4096, "layers": 32, "heads": 32, "params": 7e9},
    "8B":      {"bf16_gib": 16, "hidden": 4096, "layers": 32, "heads": 32, "params": 8e9},
    "14B":     {"bf16_gib": 28, "hidden": 5120, "layers": 40, "heads": 40, "params": 14e9},
    "30B-A3B": {"bf16_gib": 60, "hidden": 6144, "layers": 48, "heads": 48, "params": 30e9, "active_params": 3e9},
}


# ─── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    phase_id: int
    name: str
    time_s: float
    peak_mem_gib: float
    gpu_util_pct: float
    description: str = ""


@dataclass
class StepResult:
    phases: List[PhaseResult] = field(default_factory=list)
    total_time_s: float = 0.0
    bottleneck_phase: str = ""
    bottleneck_pct: float = 0.0


# ─── Timing Helpers ───────────────────────────────────────────────────────

def model_bf16_bytes(mspec: dict) -> float:
    """Return bf16 model size in bytes."""
    return mspec["params"] * 2  # bf16 = 2 bytes per param


def model_bf16_gib(mspec: dict) -> float:
    """Return bf16 model size in GiB."""
    return model_bf16_bytes(mspec) / (1024 ** 3)


def forward_pass_time_compute_bound(mspec: dict, gpu: dict, batch: int = 1) -> float:
    """
    Compute-bound forward pass time.
    Forward pass FLOPs ~ 2 * model_params * seq_tokens (matmul: input @ weight).
    For a single token through the full model: ~2 * params FLOPs.
    throughput = gpu_tflops * 1e12 FLOP/s
    """
    params = mspec.get("active_params", mspec["params"])
    flops = 2 * params  # per token
    tflops = gpu["tflops_bf16"] * 1e12
    # With batch, we get better GPU utilization but same per-token compute
    # Compute-bound: time = total_flops / throughput
    # For batch: total_flops = batch * 2 * params * seq_len, but we compute per-token
    return flops / tflops


def forward_pass_time_memory_bound(mspec: dict, gpu: dict) -> float:
    """
    Memory-bound forward pass time (weight loading).
    Must read all weights from HBM: model_size / hbm_bw.
    """
    params = mspec.get("active_params", mspec["params"])
    weight_bytes = params * 2  # bf16
    hbm_bw_bytes = gpu["hbm_bw"] * 1e9  # GB/s -> bytes/s (approx, 1 GiB/s ≈ 1e9 B/s)
    return weight_bytes / hbm_bw_bytes


def is_memory_bound(mspec: dict, gpu: dict) -> bool:
    """Determine if forward pass is memory-bound on this GPU."""
    compute_time = forward_pass_time_compute_bound(mspec, gpu)
    memory_time = forward_pass_time_memory_bound(mspec, gpu)
    return memory_time > compute_time


def effective_forward_pass_time(mspec: dict, gpu: dict, batch: int = 1, seq_len: int = 1024) -> float:
    """
    Effective forward pass time per batch.
    For memory-bound regime with batch: time ≈ memory_bound_time (weights loaded once).
    For compute-bound regime: time ≈ batch * seq_len * compute_time_per_token.
    Reality: mix of both. Use the dominant bound with arithmetic intensity considerations.
    """
    params = mspec.get("active_params", mspec["params"])
    mem_time = forward_pass_time_memory_bound(mspec, gpu)
    compute_per_token = 2 * params / (gpu["tflops_bf16"] * 1e12)
    compute_time = batch * seq_len * compute_per_token

    # Arithmetic intensity = FLOPs / bytes_loaded = (2 * params * batch * seq_len) / (params * 2)
    #                      = batch * seq_len
    # Roofline: if arithmetic_intensity > ops:byte_ratio, compute-bound; else memory-bound
    ops_byte_ratio = gpu["tflops_bf16"] * 1e12 / (gpu["hbm_bw"] * 1e9)
    arithmetic_intensity = batch * seq_len

    if arithmetic_intensity >= ops_byte_ratio:
        # Compute-bound
        return compute_time
    else:
        # Memory-bound: weight load dominates, some compute overlapped
        return max(mem_time, compute_time)


def gradient_step_time(mspec: dict, gpu: dict, batch: int = 1, seq_len: int = 1024, lora: bool = False) -> float:
    """
    Gradient step time = 3 * forward_pass_time (forward + backward + optimizer).
    Backward is ~2x forward (gradient for each param).
    Optimizer step is ~1x forward (read/write params + momentum/variance).
    Total ≈ 3x forward for full, slightly less for LoRA.
    """
    fwd_time = effective_forward_pass_time(mspec, gpu, batch, seq_len)
    if lora:
        # LoRA only updates a small fraction of params (~0.4% for typical rank)
        lora_fraction = 0.004  # rank=16 on 4096 hidden ≈ 0.4%
        # Forward is still full model (all weights read), backward only for LoRA params
        # Optimizer step only for LoRA params
        # Actual: fwd unchanged, bwd ≈ fwd * (1 + lora_fraction * 2), opt ≈ lora_fraction * fwd
        return fwd_time * (1 + 2 * lora_fraction + lora_fraction)
    return 3 * fwd_time


def nccl_allreduce_time(mspec: dict, gpu: dict, dp: int) -> float:
    """
    NCCL AllReduce time for gradient synchronization.
    AllReduce = ReduceScatter + AllGather.
    Time ≈ 2 * model_size * (dp-1)/dp / bandwidth.
    """
    if dp <= 1:
        return 0.0

    params = mspec.get("active_params", mspec["params"])
    model_bytes = params * 2  # bf16

    # Choose bandwidth: NVLink if available, else PCIe
    if "nvlink_bw" in gpu:
        bw = gpu["nvlink_bw"] * 1e9  # GB/s -> bytes/s
    else:
        bw = gpu.get("pcie_bw", 32) * 1e9

    # Ring AllReduce: 2 * (n-1)/n * message_size / bandwidth
    # With dp GPUs, each GPU sends (dp-1)/dp fraction of data twice (reduce-scatter + all-gather)
    return 2 * model_bytes * (dp - 1) / dp / bw


def inference_throughput(mspec: dict, gpu: dict, batch: int, seq_len: int = 1024) -> float:
    """
    Estimate inference throughput (tokens/second) for batch generation.
    For 8B bf16 on RTX 4090: ~50 tokens/s with batch=1, ~200 tokens/s with batch=4.
    """
    params = mspec.get("active_params", mspec["params"])

    # Base throughput per token (memory-bound single-batch decode)
    # tokens/s = 1 / time_per_token
    time_per_token_mem = forward_pass_time_memory_bound(mspec, gpu)
    time_per_token_compute = 2 * params / (gpu["tflops_bf16"] * 1e12)

    # For decode (single token generation step), memory-bound dominates at small batch
    base_tps = 1.0 / max(time_per_token_mem, time_per_token_compute)

    # Batch scaling: larger batches amortize weight loading, shift toward compute-bound
    # Throughput scales sub-linearly with batch due to compute becoming bottleneck
    # Empirical fit: throughput ≈ base_tps * batch^0.7 for moderate batches
    batch_factor = batch ** 0.7 if batch > 1 else 1.0

    # Cap at compute-bound throughput ceiling
    compute_ceiling = gpu["tflops_bf16"] * 1e12 / (2 * params)  # max tokens/s compute-bound
    estimated_tps = min(base_tps * batch_factor, compute_ceiling)

    return estimated_tps


# ─── Phase Timing Models ──────────────────────────────────────────────────

def phase0_weight_sync(mspec: dict, gpu: dict, dp: int) -> PhaseResult:
    """Phase 0: Weight sync (sleep → transfer → wake → validate)."""
    params = mspec.get("active_params", mspec["params"])
    model_bytes = params * 2

    # Sleep time: CPU triggers sleep signal, GPU enters low-power (~0.1s overhead)
    sleep_time = 0.05

    # Transfer time: depends on interconnect
    if "nvlink_bw" in gpu:
        bw = gpu["nvlink_bw"] * 1e9
    else:
        bw = gpu.get("pcie_bw", 32) * 1e9
    transfer_time = model_bytes / bw

    # Wake time: GPU resumes from sleep state (~0.1s)
    wake_time = 0.05

    # Validate: quick hash check of weights (~0.02s)
    validate_time = 0.02

    total_time = sleep_time + transfer_time + wake_time + validate_time
    # During transfer, GPU memory is being written; peak = model size
    peak_mem = model_bf16_gib(mspec) if dp == 1 else model_bf16_gib(mspec) / dp

    return PhaseResult(
        phase_id=0, name="weight_sync",
        time_s=total_time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=10.0,  # mostly idle, just receiving data
        description=f"sleep({sleep_time:.3f}s) + transfer({transfer_time:.3f}s) + wake({wake_time:.3f}s) + validate({validate_time:.3f}s)"
    )


def phase1_rollout(mspec: dict, gpu: dict, batch: int, gs: int, seq_len: int = 1024) -> PhaseResult:
    """Phase 1: Rollout - SGLang/vLLM inference generation, gs samples per prompt."""
    total_samples = batch * gs
    total_tokens = total_samples * seq_len

    # Inference throughput with rollout batch
    tps = inference_throughput(mspec, gpu, total_samples, seq_len)
    rollout_time = total_tokens / tps

    # Memory during rollout: model weights + KV cache for all samples
    # KV cache per sample: ~2 * layers * heads * head_dim * seq_len * 2 bytes
    head_dim = mspec["hidden"] // mspec["heads"]
    kv_per_sample_bytes = 2 * mspec["layers"] * mspec["heads"] * head_dim * seq_len * 2
    kv_total_gib = kv_per_sample_bytes * total_samples / (1024 ** 3)
    peak_mem = model_bf16_gib(mspec) + kv_total_gib

    return PhaseResult(
        phase_id=1, name="rollout",
        time_s=rollout_time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=85.0,  # inference generation is intensive
        description=f"batch={batch}, gs={gs}, total_samples={total_samples}, tokens={total_tokens}, tps={tps:.1f}"
    )


def phase2_replay_buffer(mspec: dict, gpu: dict) -> PhaseResult:
    """Phase 2: Replay buffer update (CPU-backed, negligible)."""
    return PhaseResult(
        phase_id=2, name="replay_buffer_update",
        time_s=0.01,  # CPU operation, negligible
        peak_mem_gib=0.0,  # CPU memory, not GPU
        gpu_util_pct=0.0,
        description="CPU-backed buffer update, negligible GPU time"
    )


def phase3_sleep_replicas(mspec: dict, gpu: dict) -> PhaseResult:
    """Phase 3: Sleep replicas - free GPU memory after rollout."""
    return PhaseResult(
        phase_id=3, name="sleep_replicas",
        time_s=0.05,  # memory deallocation overhead
        peak_mem_gib=model_bf16_gib(mspec),  # before freeing
        gpu_util_pct=0.0,
        description="Free rollout GPU memory, prepare for training phases"
    )


def phase4_reward(mspec: dict, gpu: dict, batch: int, gs: int, seq_len: int = 1024,
                   reward_type: str = "rule") -> PhaseResult:
    """Phase 4: Reward computation (model-based or rule-based)."""
    total_samples = batch * gs

    if reward_type == "rule":
        # Rule-based: very fast, CPU computation
        time = 0.05  # nearly instant
        peak_mem = 0.0
        gpu_util = 0.0
    else:
        # Model-based reward: forward pass through reward model (~1B params typically)
        reward_model_params = 1e9  # typical reward model size
        reward_mspec = {"params": reward_model_params, "active_params": reward_model_params,
                        "hidden": 2048, "layers": 16, "heads": 16}
        # Batch through reward model
        fwd_time = effective_forward_pass_time(reward_mspec, gpu, total_samples, seq_len)
        time = fwd_time
        peak_mem = reward_model_params * 2 / (1024 ** 3)  # reward model weights
        gpu_util = 60.0

    return PhaseResult(
        phase_id=4, name="reward_computation",
        time_s=time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=gpu_util,
        description=f"reward_type={reward_type}, samples={total_samples}"
    )


def phase5_batch_balancing(mspec: dict, gpu: dict) -> PhaseResult:
    """Phase 5: Batch balancing (CPU, negligible)."""
    return PhaseResult(
        phase_id=5, name="batch_balancing",
        time_s=0.01,
        peak_mem_gib=0.0,
        gpu_util_pct=0.0,
        description="CPU batch balancing, negligible GPU time"
    )


def phase6_old_log_prob(mspec: dict, gpu: dict, batch: int, gs: int,
                         seq_len: int = 1024, dp: int = 1) -> PhaseResult:
    """Phase 6: Old log prob - forward pass through actor."""
    total_samples = batch * gs
    fwd_time = effective_forward_pass_time(mspec, gpu, total_samples, seq_len)
    comm_time = nccl_allreduce_time(mspec, gpu, dp) if dp > 1 else 0.0

    # Activation memory: ~2 * total_samples * seq_len * hidden * layers bytes (approximate)
    act_mem = 2 * total_samples * seq_len * mspec["hidden"] * mspec["layers"] / (1024 ** 3)
    peak_mem = model_bf16_gib(mspec) / dp + act_mem / dp

    return PhaseResult(
        phase_id=6, name="old_log_prob",
        time_s=fwd_time + comm_time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=70.0,
        description=f"actor forward, samples={total_samples}, fwd={fwd_time:.3f}s, comm={comm_time:.3f}s"
    )


def phase7_ref_log_prob(mspec: dict, gpu: dict, batch: int, gs: int,
                         seq_len: int = 1024, dp: int = 1) -> PhaseResult:
    """Phase 7: Ref log prob - forward pass through reference model."""
    total_samples = batch * gs
    fwd_time = effective_forward_pass_time(mspec, gpu, total_samples, seq_len)
    comm_time = nccl_allreduce_time(mspec, gpu, dp) if dp > 1 else 0.0

    # Need both actor and ref model in memory
    act_mem = 2 * total_samples * seq_len * mspec["hidden"] * mspec["layers"] / (1024 ** 3)
    peak_mem = 2 * model_bf16_gib(mspec) / dp + act_mem / dp  # actor + ref

    return PhaseResult(
        phase_id=7, name="ref_log_prob",
        time_s=fwd_time + comm_time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=70.0,
        description=f"ref forward, samples={total_samples}, fwd={fwd_time:.3f}s, comm={comm_time:.3f}s"
    )


def phase8_advantage(mspec: dict, gpu: dict, batch: int, gs: int) -> PhaseResult:
    """Phase 8: Advantage computation (GRPO normalization)."""
    total_samples = batch * gs
    # GRPO advantage: group normalization, very fast (just statistics on small tensors)
    # ~1ms per group, total = batch groups
    time = batch * 0.001  # very fast
    peak_mem = 0.1  # negligible extra memory
    gpu_util = 5.0

    return PhaseResult(
        phase_id=8, name="advantage_computation",
        time_s=time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=gpu_util,
        description=f"GRPO normalization, groups={batch}, samples/group={gs}"
    )


def phase9_actor_update(mspec: dict, gpu: dict, batch: int, gs: int,
                         seq_len: int = 1024, dp: int = 1,
                         lora: bool = False, mini_batch: int = 4) -> PhaseResult:
    """Phase 9: Actor update - mini-batch gradient + optimizer step."""
    total_samples = batch * gs

    # Mini-batch gradient steps
    num_mini_batches = max(1, total_samples // mini_batch)
    step_time = gradient_step_time(mspec, gpu, mini_batch, seq_len, lora=lora)
    total_grad_time = num_mini_batches * step_time

    # Gradient AllReduce
    comm_time = nccl_allreduce_time(mspec, gpu, dp) if dp > 1 else 0.0

    # Optimizer memory: momentum + variance states (fp32) ≈ 8 bytes per param
    params = mspec.get("active_params", mspec["params"])
    if lora:
        opt_mem = params * 0.004 * 8 / (1024 ** 3)  # only LoRA params
    else:
        opt_mem = params * 8 / (1024 ** 3)

    # Gradient memory: bf16, 2 bytes per param
    grad_mem = params * 2 / (1024 ** 3) if not lora else params * 0.004 * 2 / (1024 ** 3)

    act_mem = 2 * mini_batch * seq_len * mspec["hidden"] * mspec["layers"] / (1024 ** 3)
    peak_mem = model_bf16_gib(mspec) / dp + opt_mem / dp + grad_mem / dp + act_mem / dp

    return PhaseResult(
        phase_id=9, name="actor_update",
        time_s=total_grad_time + comm_time,
        peak_mem_gib=peak_mem,
        gpu_util_pct=95.0,
        description=f"mini_batches={num_mini_batches}, mini_batch={mini_batch}, "
                    f"grad_step={total_grad_time:.3f}s, comm={comm_time:.3f}s, lora={lora}"
    )


# ─── Full Step Simulation ─────────────────────────────────────────────────

def simulate_step(mspec: dict, gpu: dict, batch: int = 4, gs: int = 4,
                  seq_len: int = 1024, dp: int = 1, reward_type: str = "rule",
                  lora: bool = False, mini_batch: int = 4) -> StepResult:
    """Simulate a complete GRPO training step (10 phases)."""
    phases = [
        phase0_weight_sync(mspec, gpu, dp),
        phase1_rollout(mspec, gpu, batch, gs, seq_len),
        phase2_replay_buffer(mspec, gpu),
        phase3_sleep_replicas(mspec, gpu),
        phase4_reward(mspec, gpu, batch, gs, seq_len, reward_type),
        phase5_batch_balancing(mspec, gpu),
        phase6_old_log_prob(mspec, gpu, batch, gs, seq_len, dp),
        phase7_ref_log_prob(mspec, gpu, batch, gs, seq_len, dp),
        phase8_advantage(mspec, gpu, batch, gs),
        phase9_actor_update(mspec, gpu, batch, gs, seq_len, dp, lora, mini_batch),
    ]

    total_time = sum(p.time_s for p in phases)
    # Find bottleneck
    bottleneck = max(phases, key=lambda p: p.time_s)

    result = StepResult(
        phases=phases,
        total_time_s=total_time,
        bottleneck_phase=bottleneck.name,
        bottleneck_pct=bottleneck.time_s / total_time * 100,
    )
    return result


# ─── Display Helpers ──────────────────────────────────────────────────────

def fmt_time(t: float) -> str:
    if t < 0.001:
        return f"{t*1e6:.1f}us"
    elif t < 1.0:
        return f"{t*1e3:.2f}ms"
    else:
        return f"{t:.3f}s"


def fmt_mem(m: float) -> str:
    if m < 0.01:
        return "~0"
    elif m < 1.0:
        return f"{m:.2f}MiB" if m * 1024 < 1024 else f"{m:.3f}GiB"
    else:
        return f"{m:.2f}GiB"


def print_phase_table(phases: List[PhaseResult], title: str = ""):
    """Print a formatted table of phase results."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    header = f"{'Phase':>5} {'Name':<22} {'Time':>10} {'Peak Mem':>10} {'GPU Util':>9} {'% Total':>8} {'Details'}"
    print(header)
    print("-" * len(header))

    total_time = sum(p.time_s for p in phases)
    for p in phases:
        pct = p.time_s / total_time * 100 if total_time > 0 else 0
        print(f"  {p.phase_id:>3}  {p.name:<22} {fmt_time(p.time_s):>10} "
              f"{fmt_mem(p.peak_mem_gib):>10} {p.gpu_util_pct:>8.1f}% {pct:>7.1f}%  {p.description}")

    print("-" * len(header))
    print(f"  {'TOTAL':>3}  {'':<22} {fmt_time(total_time):>10}")


def print_step_summary(result: StepResult, mspec: dict, gpu: dict,
                       batch: int, gs: int, seq_len: int, dp: int):
    """Print step summary statistics."""
    total_tokens = batch * gs * seq_len
    steps_per_hour = 3600 / result.total_time_s if result.total_time_s > 0 else 0
    tokens_per_hour = total_tokens * steps_per_hour

    print(f"\n{'='*80}")
    print(f"  STEP SUMMARY")
    print(f"{'='*80}")
    print(f"  Model:            {mspec.get('name', '?')} ({mspec['params']/1e9:.0f}B params)")
    print(f"  GPU:              {gpu.get('name', '?')} ({gpu['vram']} GiB VRAM, {gpu['tflops_bf16']} TFLOPS bf16)")
    print(f"  Config:           batch={batch}, gs={gs}, seq_len={seq_len}, dp={dp}")
    print(f"  Total step time:  {fmt_time(result.total_time_s)}")
    print(f"  Bottleneck:       {result.bottleneck_phase} ({result.bottleneck_pct:.1f}% of step)")
    print(f"  Steps/hour:       {steps_per_hour:.2f}")
    print(f"  Tokens/step:      {total_tokens:,}")
    print(f"  Tokens/hour:      {tokens_per_hour:,.0f}")

    # Memory check
    max_mem_phase = max(result.phases, key=lambda p: p.peak_mem_gib)
    vram = gpu["vram"]
    mem_usage = max_mem_phase.peak_mem_gib
    mem_pct = mem_usage / vram * 100

    print(f"  Peak GPU memory:  {fmt_mem(mem_usage)} / {vram} GiB ({mem_pct:.1f}%)")
    if mem_pct > 100:
        print(f"  *** OOM WARNING: Peak memory exceeds GPU VRAM! ***")
    elif mem_pct > 85:
        print(f"  *** HIGH MEMORY: Close to GPU VRAM limit ***")
    else:
        print(f"  Memory headroom:  {vram - mem_usage:.1f} GiB available")


# ─── Mode 1: simulate ─────────────────────────────────────────────────────

def run_simulate(args):
    """Mode 1: Simulate a complete GRPO training step."""
    mspec = MODEL_SPECS[args.model].copy()
    mspec["name"] = args.model
    gpu = GPU_SPECS[args.gpu].copy()
    gpu["name"] = args.gpu

    result = simulate_step(mspec, gpu, batch=args.batch, gs=args.gs,
                           seq_len=args.seq_len, dp=args.dp,
                           reward_type=args.reward, lora=args.lora,
                           mini_batch=args.mini_batch)

    print_phase_table(result.phases, f"GRPO Training Step Simulation - {args.model} on {args.gpu}")
    print_step_summary(result, mspec, gpu, args.batch, args.gs, args.seq_len, args.dp)


# ─── Mode 2: compare ──────────────────────────────────────────────────────

def run_compare(args):
    """Mode 2: Compare across GPU configs."""
    mspec = MODEL_SPECS[args.model].copy()
    mspec["name"] = args.model

    gpu_names = ["RTX_4090", "A100", "H100", "A16"]

    print(f"\n{'='*100}")
    print(f"  GPU COMPARISON for {args.model} GRPO Training")
    print(f"  Config: batch={args.batch}, gs={args.gs}, seq_len={args.seq_len}")
    print(f"{'='*100}")

    # Header
    print(f"\n{'GPU':<12} {'VRAM':>6} {'TFLOPS':>8} {'Step Time':>12} "
          f"{'Steps/hr':>10} {'Tokens/hr':>12} {'Peak Mem':>10} "
          f"{'Bottleneck':<18} {'Viability':<12}")
    print("-" * 100)

    total_tokens = args.batch * args.gs * args.seq_len

    for gpu_name in gpu_names:
        gpu = GPU_SPECS[gpu_name].copy()
        gpu["name"] = gpu_name

        result = simulate_step(mspec, gpu, batch=args.batch, gs=args.gs,
                               seq_len=args.seq_len, dp=1,
                               reward_type=args.reward, lora=args.lora,
                               mini_batch=args.mini_batch)

        steps_per_hour = 3600 / result.total_time_s if result.total_time_s > 0 else 0
        tokens_per_hour = total_tokens * steps_per_hour

        max_mem_phase = max(result.phases, key=lambda p: p.peak_mem_gib)
        mem_pct = max_mem_phase.peak_mem_gib / gpu["vram"] * 100

        if mem_pct > 100:
            viability = "OOM"
        elif mem_pct > 85:
            viability = "TIGHT"
        elif result.total_time_s > 60:
            viability = "SLOW"
        elif result.total_time_s > 10:
            viability = "VIABLE"
        else:
            viability = "FAST"

        print(f"  {gpu_name:<12} {gpu['vram']:>5}G {gpu['tflops_bf16']:>7.1f} "
              f"{fmt_time(result.total_time_s):>12} {steps_per_hour:>9.2f} "
              f"{tokens_per_hour:>11,.0f} {fmt_mem(max_mem_phase.peak_mem_gib):>10} "
              f"{result.bottleneck_phase:<18} {viability:<12}")

    # Detailed phase breakdown per GPU
    print(f"\n{'='*100}")
    print(f"  DETAILED PHASE BREAKDOWN BY GPU")
    print(f"{'='*100}")

    for gpu_name in gpu_names:
        gpu = GPU_SPECS[gpu_name].copy()
        gpu["name"] = gpu_name

        result = simulate_step(mspec, gpu, batch=args.batch, gs=args.gs,
                               seq_len=args.seq_len, dp=1,
                               reward_type=args.reward, lora=args.lora,
                               mini_batch=args.mini_batch)

        print_phase_table(result.phases, f"{args.model} on {gpu_name} ({gpu['vram']} GiB, {gpu['tflops_bf16']} TFLOPS)")

    # Memory comparison
    print(f"\n{'='*100}")
    print(f"  MEMORY BUDGET COMPARISON")
    print(f"{'='*100}")

    print(f"\n{'GPU':<12} {'VRAM':>6} {'Phase 1':>10} {'Phase 6':>10} {'Phase 7':>10} "
          f"{'Phase 9':>10} {'Peak':>10} {'Headroom':>10}")
    print("-" * 80)

    for gpu_name in gpu_names:
        gpu = GPU_SPECS[gpu_name].copy()
        gpu["name"] = gpu_name

        result = simulate_step(mspec, gpu, batch=args.batch, gs=args.gs,
                               seq_len=args.seq_len, dp=1,
                               reward_type=args.reward, lora=args.lora,
                               mini_batch=args.mini_batch)

        p1_mem = result.phases[1].peak_mem_gib
        p6_mem = result.phases[6].peak_mem_gib
        p7_mem = result.phases[7].peak_mem_gib
        p9_mem = result.phases[9].peak_mem_gib
        peak = max(p.peak_mem_gib for p in result.phases)
        headroom = gpu["vram"] - peak

        print(f"  {gpu_name:<12} {gpu['vram']:>5}G {p1_mem:>9.2f} {p6_mem:>9.2f} "
              f"{p7_mem:>9.2f} {p9_mem:>9.2f} {peak:>9.2f} {headroom:>9.2f}")


# ─── Mode 3: scale ────────────────────────────────────────────────────────

def run_scale(args):
    """Mode 3: Scaling analysis across dp values."""
    mspec = MODEL_SPECS[args.model].copy()
    mspec["name"] = args.model
    gpu = GPU_SPECS[args.gpu].copy()
    gpu["name"] = args.gpu

    dp_values = [1, 2, 4, 8]

    print(f"\n{'='*100}")
    print(f"  SCALING ANALYSIS for {args.model} on {args.gpu}")
    print(f"  Config: batch={args.batch}, gs={args.gs}, seq_len={args.seq_len}")
    print(f"{'='*100}")

    # Scaling table
    print(f"\n{'DP':>4} {'Step Time':>12} {'Comm Overhead':>14} {'Steps/hr':>10} "
          f"{'Tokens/hr':>12} {'Mem/GPU':>10} {'Mem %':>8} {'Efficiency':>10}")
    print("-" * 100)

    total_tokens = args.batch * args.gs * args.seq_len
    baseline_time = None

    results = []
    for dp in dp_values:
        result = simulate_step(mspec, gpu, batch=args.batch, gs=args.gs,
                               seq_len=args.seq_len, dp=dp,
                               reward_type=args.reward, lora=args.lora,
                               mini_batch=args.mini_batch)

        comm_overhead = sum(p.time_s for p in result.phases
                           if "comm" in p.description.lower() or p.name == "weight_sync")

        steps_per_hour = 3600 / result.total_time_s if result.total_time_s > 0 else 0
        tokens_per_hour = total_tokens * steps_per_hour

        max_mem_phase = max(result.phases, key=lambda p: p.peak_mem_gib)
        mem_per_gpu = max_mem_phase.peak_mem_gib
        mem_pct = mem_per_gpu / gpu["vram"] * 100

        if dp == 1:
            baseline_time = result.total_time_s
            efficiency = 100.0
        else:
            # Ideal scaling: step_time = baseline_time / dp
            ideal_time = baseline_time / dp
            efficiency = ideal_time / result.total_time_s * 100 if result.total_time_s > 0 else 0

        results.append({
            "dp": dp, "step_time": result.total_time_s,
            "comm_overhead": comm_overhead, "steps_per_hour": steps_per_hour,
            "tokens_per_hour": tokens_per_hour, "mem_per_gpu": mem_per_gpu,
            "mem_pct": mem_pct, "efficiency": efficiency, "result": result
        })

        print(f"  {dp:>3}  {fmt_time(result.total_time_s):>12} {fmt_time(comm_overhead):>14} "
              f"{steps_per_hour:>9.2f} {tokens_per_hour:>11,.0f} "
              f"{mem_per_gpu:>9.2f} {mem_pct:>7.1f}% {efficiency:>9.1f}%")

    # Communication overhead breakdown
    print(f"\n{'='*100}")
    print(f"  COMMUNICATION OVERHEAD BREAKDOWN")
    print(f"{'='*100}")

    print(f"\n{'DP':>4} {'AllReduce (grad)':>18} {'Weight Sync':>14} {'Total Comm':>14} "
          f"{'Comm % of Step':>16}")
    print("-" * 80)

    for r in results:
        dp = r["dp"]
        # Phase 0 (weight_sync) and phase 9 (actor_update has AllReduce)
        p0_time = r["result"].phases[0].time_s
        # AllReduce time in phases 6, 7, 9
        allreduce_time = nccl_allreduce_time(mspec, gpu, dp) * 3  # phases 6, 7, 9
        total_comm = p0_time + allreduce_time
        comm_pct = total_comm / r["step_time"] * 100 if r["step_time"] > 0 else 0

        print(f"  {dp:>3}  {fmt_time(allreduce_time):>18} {fmt_time(p0_time):>14} "
              f"{fmt_time(total_comm):>14} {comm_pct:>15.1f}%")

    # Scaling efficiency curve
    print(f"\n{'='*100}")
    print(f"  SCALING EFFICIENCY CURVE")
    print(f"{'='*100}")

    print(f"\n  DP  |  Ideal Time  |  Actual Time  |  Efficiency  |  Speedup")
    print(f"  ----+-------------+--------------+-------------+----------")

    for r in results:
        dp = r["dp"]
        ideal = baseline_time / dp
        actual = r["step_time"]
        speedup = baseline_time / actual if actual > 0 else 0
        print(f"   {dp:>2}  |  {fmt_time(ideal):>11}  |  {fmt_time(actual):>11}  |  "
              f"{r['efficiency']:>8.1f}%  |  {speedup:>6.2f}x")

    # Detailed phase breakdown for each dp
    print(f"\n{'='*100}")
    print(f"  PHASE TIMING BY DP")
    print(f"{'='*100}")

    for r in results:
        print_phase_table(r["result"].phases,
                         f"{args.model} on {args.gpu}, dp={r['dp']}")

    # Memory scaling
    print(f"\n{'='*100}")
    print(f"  MEMORY PER GPU SCALING")
    print(f"{'='*100}")

    print(f"\n{'DP':>4} {'Model/GPU':>10} {'Act/GPU':>10} {'Opt/GPU':>10} "
          f"{'Peak/GPU':>10} {'VRAM':>6} {'Headroom':>10} {'Status'}")
    print("-" * 80)

    model_mem = model_bf16_gib(mspec)
    params = mspec.get("active_params", mspec["params"])

    for r in results:
        dp = r["dp"]
        m_per = model_mem / dp
        # Optimizer states
        if args.lora:
            opt_per = params * 0.004 * 8 / (1024 ** 3) / dp
        else:
            opt_per = params * 8 / (1024 ** 3) / dp
        # Activation peak
        act_per = r["result"].phases[9].peak_mem_gib - m_per - opt_per
        peak = r["mem_per_gpu"]
        headroom = gpu["vram"] - peak
        status = "OK" if headroom > 0 else "OOM!"

        print(f"  {dp:>3}  {m_per:>9.2f} {act_per:>9.2f} {opt_per:>9.2f} "
              f"{peak:>9.2f} {gpu['vram']:>5} {headroom:>9.2f} {status}")


# ─── Mode 4: rtx4090 ──────────────────────────────────────────────────────

def run_rtx4090(args):
    """Mode 4: RTX 4090 specific comprehensive analysis."""
    mspec = MODEL_SPECS[args.model].copy()
    mspec["name"] = args.model
    gpu = GPU_SPECS["RTX_4090"].copy()
    gpu["name"] = "RTX_4090"

    print(f"\n{'='*100}")
    print(f"  RTX 4090 COMPREHENSIVE GRPO ANALYSIS")
    print(f"  Model: {args.model} ({mspec['params']/1e9:.0f}B params, {mspec['bf16_gib']} GiB bf16)")
    print(f"  GPU:   RTX 4090 (24 GiB VRAM, 82.6 TFLOPS bf16, 1008 GB/s HBM)")
    print(f"{'='*100}")

    # ─── Section 1: Complete 10-Phase Timeline ────────────────────────────

    print(f"\n{'='*100}")
    print(f"  SECTION 1: COMPLETE 10-PHASE TIMELINE")
    print(f"  Config: batch=4, gs=4, seq_len=1024, dp=1, reward=rule, full update")
    print(f"{'='*100}")

    result = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                           reward_type="rule", lora=False, mini_batch=4)

    print_phase_table(result.phases, "Default Config: batch=4, gs=4, seq_len=1024")
    print_step_summary(result, mspec, gpu, 4, 4, 1024, 1)

    # Timeline visualization
    print(f"\n  TIMELINE VISUALIZATION (proportional):")
    total = result.total_time_s
    for p in result.phases:
        bar_len = int(p.time_s / total * 60)
        bar = "#" * max(bar_len, 1)
        print(f"  P{p.phase_id} {p.name:<22} |{bar}| {p.time_s/total*100:.1f}%")

    # ─── Section 2: Memory Budget Breakdown ──────────────────────────────

    print(f"\n{'='*100}")
    print(f"  SECTION 2: MEMORY BUDGET BREAKDOWN PER PHASE")
    print(f"{'='*100}")

    print(f"\n  Phase  |  Components                     |  Peak Mem  |  vs 24GiB  |  Headroom")
    print(f"  -------+---------------------------------+------------+------------+----------")

    components = {
        0: "model weights",
        1: "model + KV cache (rollout batch)",
        2: "CPU only (no GPU mem)",
        3: "model weights (pre-free)",
        4: "none (rule-based reward)",
        5: "CPU only (no GPU mem)",
        6: "actor model + activations",
        7: "actor + ref model + activations",
        8: "negligible extra",
        9: "actor + optimizer states + gradients + activations",
    }

    for p in result.phases:
        pct = p.peak_mem_gib / 24 * 100
        headroom = 24 - p.peak_mem_gib
        print(f"  P{p.phase_id:>2}    |  {components[p.phase_id]:<31} |  {p.peak_mem_gib:>8.2f}G  |  "
              f"{pct:>7.1f}%   |  {headroom:>7.2f}G")

    # Memory challenge analysis
    print(f"\n  *** RTX 4090 MEMORY CHALLENGE ***")
    print(f"  VRAM: 24 GiB")
    model_mem = model_bf16_gib(mspec)
    print(f"  Model bf16: {model_mem:.2f} GiB ({model_mem/24*100:.1f}% of VRAM)")
    print(f"  Optimizer (fp32 momentum+variance): {mspec['params']*8/(1024**3):.2f} GiB")
    print(f"  Gradients (bf16): {mspec['params']*2/(1024**3):.2f} GiB")
    print(f"  Training total (model+opt+grad): {model_mem + mspec['params']*8/(1024**3) + mspec['params']*2/(1024**3):.2f} GiB")
    print(f"  Phase 7 (actor+ref): {2*model_mem:.2f} GiB ({2*model_mem/24*100:.1f}% of VRAM)")

    if 2 * model_mem > 24:
        print(f"  >>> Phase 7 (ref log prob) EXCEEDS 24 GiB: actor({model_mem:.2f}G) + ref({model_mem:.2f}G) = {2*model_mem:.2f}G > 24G")
        print(f"  >>> Must use reference model offload or bypass mode")

    # ─── Section 3: Config Comparisons ────────────────────────────────────

    print(f"\n{'='*100}")
    print(f"  SECTION 3: CONFIGURATION COMPARISONS")
    print(f"{'='*100}")

    configs = [
        ("Default (full, gs=4)",    {"batch": 4, "gs": 4, "seq_len": 1024, "lora": False, "reward": "rule", "mini_batch": 4}),
        ("Default (full, gs=8)",    {"batch": 4, "gs": 8, "seq_len": 1024, "lora": False, "reward": "rule", "mini_batch": 4}),
        ("Bypass mode (no ref fwd)", {"batch": 4, "gs": 4, "seq_len": 1024, "lora": False, "reward": "rule", "mini_batch": 4, "bypass": True}),
        ("LoRA (rank=16)",          {"batch": 4, "gs": 4, "seq_len": 1024, "lora": True,  "reward": "rule", "mini_batch": 4}),
        ("LoRA + gs=8",            {"batch": 4, "gs": 8, "seq_len": 1024, "lora": True,  "reward": "rule", "mini_batch": 4}),
        ("Model reward",           {"batch": 4, "gs": 4, "seq_len": 1024, "lora": False, "reward": "model", "mini_batch": 4}),
        ("Smaller batch=2, gs=4",  {"batch": 2, "gs": 4, "seq_len": 1024, "lora": False, "reward": "rule", "mini_batch": 4}),
        ("Shorter seq=512",        {"batch": 4, "gs": 4, "seq_len": 512,  "lora": False, "reward": "rule", "mini_batch": 4}),
    ]

    print(f"\n{'Config':<25} {'Step Time':>12} {'Steps/hr':>10} {'Tokens/hr':>12} "
          f"{'Peak Mem':>10} {'Mem %':>8} {'Bottleneck':<18}")
    print("-" * 100)

    total_tokens_default = 4 * 4 * 1024

    for name, cfg in configs:
        bypass = cfg.get("bypass", False)

        result = simulate_step(mspec, gpu, batch=cfg["batch"], gs=cfg["gs"],
                               seq_len=cfg["seq_len"], dp=1,
                               reward_type=cfg["reward"], lora=cfg["lora"],
                               mini_batch=cfg["mini_batch"])

        if bypass:
            # Bypass mode: skip phase 7 (ref_log_prob) - reference values computed once and cached
            result.phases[7].time_s = 0.0
            result.phases[7].peak_mem_gib = 0.0
            result.phases[7].gpu_util_pct = 0.0
            result.total_time_s = sum(p.time_s for p in result.phases)
            bottleneck = max(result.phases, key=lambda p: p.time_s)
            result.bottleneck_phase = bottleneck.name
            result.bottleneck_pct = bottleneck.time_s / result.total_time_s * 100

        steps_per_hour = 3600 / result.total_time_s if result.total_time_s > 0 else 0
        total_tokens = cfg["batch"] * cfg["gs"] * cfg["seq_len"]
        tokens_per_hour = total_tokens * steps_per_hour

        max_mem = max(p.peak_mem_gib for p in result.phases)
        mem_pct = max_mem / 24 * 100

        print(f"  {name:<25} {fmt_time(result.total_time_s):>12} {steps_per_hour:>9.2f} "
              f"{tokens_per_hour:>11,.0f} {fmt_mem(max_mem):>10} {mem_pct:>7.1f}% "
              f"{result.bottleneck_phase:<18}")

    # Bypass vs no-bypass detailed comparison
    print(f"\n{'='*100}")
    print(f"  BYPASS vs NO-BYPASS COMPARISON")
    print(f"{'='*100}")

    result_nobypass = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                                    reward_type="rule", lora=False, mini_batch=4)
    result_bypass = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                                  reward_type="rule", lora=False, mini_batch=4)
    # Bypass: remove ref forward
    result_bypass.phases[7].time_s = 0.0
    result_bypass.phases[7].peak_mem_gib = 0.0
    result_bypass.total_time_s = sum(p.time_s for p in result_bypass.phases)
    bottleneck = max(result_bypass.phases, key=lambda p: p.time_s)
    result_bypass.bottleneck_phase = bottleneck.name
    result_bypass.bottleneck_pct = bottleneck.time_s / result_bypass.total_time_s * 100

    print(f"\n  {'Phase':>5} {'No-Bypass Time':>16} {'Bypass Time':>14} {'Saved':>10} {'No-Bypass Mem':>14} {'Bypass Mem':>12}")
    print(f"  {'-----':>5} {'---------------':>16} {'-------------':>14} {'----------':>10} {'--------------':>14} {'------------':>12}")

    for i in range(10):
        p_nb = result_nobypass.phases[i]
        p_b = result_bypass.phases[i]
        saved = p_nb.time_s - p_b.time_s
        print(f"  P{i:>2}   {fmt_time(p_nb.time_s):>16} {fmt_time(p_b.time_s):>14} "
              f"{fmt_time(saved):>10} {fmt_mem(p_nb.peak_mem_gib):>14} {fmt_mem(p_b.peak_mem_gib):>12}")

    time_saved = result_nobypass.total_time_s - result_bypass.total_time_s
    mem_saved = max(p.peak_mem_gib for p in result_nobypass.phases) - max(p.peak_mem_gib for p in result_bypass.phases)

    print(f"\n  Total time saved:  {fmt_time(time_saved)} ({time_saved/result_nobypass.total_time_s*100:.1f}%)")
    print(f"  Peak mem saved:    {fmt_mem(mem_saved)} GiB")
    print(f"  Bypass removes ref model forward (phase 7), saving {model_mem:.2f} GiB memory")
    print(f"  Critical for RTX 4090: ref model ({model_mem:.2f} GiB) cannot coexist with actor ({model_mem:.2f} GiB) in 24 GiB")

    # LoRA vs Full comparison
    print(f"\n{'='*100}")
    print(f"  LoRA vs FULL UPDATE COMPARISON")
    print(f"{'='*100}")

    result_full = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                                reward_type="rule", lora=False, mini_batch=4)
    result_lora = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                                reward_type="rule", lora=True, mini_batch=4)

    print(f"\n  {'Phase':>5} {'Full Update':>14} {'LoRA Update':>14} {'Diff':>10}")
    print(f"  {'-----':>5} {'-------------':>14} {'-------------':>14} {'----------':>10}")

    for i in range(10):
        p_f = result_full.phases[i]
        p_l = result_lora.phases[i]
        diff = p_f.time_s - p_l.time_s
        print(f"  P{i:>2}   {fmt_time(p_f.time_s):>14} {fmt_time(p_l.time_s):>14} {fmt_time(diff):>10}")

    # Memory comparison
    print(f"\n  {'Memory':>20} {'Full':>10} {'LoRA':>10} {'Saved':>10}")
    full_opt = mspec["params"] * 8 / (1024**3)
    lora_opt = mspec["params"] * 0.004 * 8 / (1024**3)
    full_grad = mspec["params"] * 2 / (1024**3)
    lora_grad = mspec["params"] * 0.004 * 2 / (1024**3)
    print(f"  {'Optimizer states':>20} {full_opt:>9.2f} {lora_opt:>9.2f} {full_opt-lora_opt:>9.2f}")
    print(f"  {'Gradients':>20} {full_grad:>9.2f} {lora_grad:>9.2f} {full_grad-lora_grad:>9.2f}")
    print(f"  {'Total training':>20} {model_mem+full_opt+full_grad:>9.2f} {model_mem+lora_opt+lora_grad:>9.2f} "
          f"{full_opt+full_grad-lora_opt-lora_grad:>9.2f}")
    print(f"  LoRA reduces optimizer+gradient memory by {((full_opt+full_grad)-(lora_opt+lora_grad)):.2f} GiB")

    # gs=4 vs gs=8
    print(f"\n{'='*100}")
    print(f"  gs=4 vs gs=8 COMPARISON")
    print(f"{'='*100}")

    result_gs4 = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                                reward_type="rule", lora=False, mini_batch=4)
    result_gs8 = simulate_step(mspec, gpu, batch=4, gs=8, seq_len=1024, dp=1,
                                reward_type="rule", lora=False, mini_batch=4)

    print(f"\n  {'Phase':>5} {'gs=4':>14} {'gs=8':>14} {'Ratio':>10}")
    print(f"  {'-----':>5} {'-------------':>14} {'-------------':>14} {'----------':>10}")

    for i in range(10):
        p4 = result_gs4.phases[i]
        p8 = result_gs8.phases[i]
        ratio = p8.time_s / p4.time_s if p4.time_s > 0 else 0
        print(f"  P{i:>2}   {fmt_time(p4.time_s):>14} {fmt_time(p8.time_s):>14} {ratio:>9.2f}x")

    print(f"\n  gs=4 total: {fmt_time(result_gs4.total_time_s)}, tokens={4*4*1024:,}")
    print(f"  gs=8 total: {fmt_time(result_gs8.total_time_s)}, tokens={4*8*1024:,}")
    print(f"  gs=8 produces more samples per step but rollout dominates (roughly 2x rollout time)")
    print(f"  GRPO advantage quality improves with larger gs (more diverse comparisons)")

    # ─── Section 4: Throughput Estimation ────────────────────────────────

    print(f"\n{'='*100}")
    print(f"  SECTION 4: THROUGHPUT ESTIMATION")
    print(f"{'='*100}")

    configs_throughput = [
        ("Default (full, gs=4)",    4, 4, 1024, False, False),
        ("Default (full, gs=8)",    4, 8, 1024, False, False),
        ("Bypass (full, gs=4)",     4, 4, 1024, False, True),
        ("LoRA (gs=4)",             4, 4, 1024, True,  False),
        ("LoRA + bypass (gs=4)",    4, 4, 1024, True,  True),
        ("LoRA + bypass (gs=8)",    4, 8, 1024, True,  True),
    ]

    print(f"\n{'Config':<25} {'Step Time':>12} {'Steps/hr':>10} {'Tokens/step':>12} "
          f"{'Tokens/hr':>14} {'Effective tok/hr':>16}")
    print("-" * 110)

    for name, b, g, s, lora, bypass in configs_throughput:
        result = simulate_step(mspec, gpu, batch=b, gs=g, seq_len=s, dp=1,
                               reward_type="rule", lora=lora, mini_batch=4)
        if bypass:
            result.phases[7].time_s = 0.0
            result.phases[7].peak_mem_gib = 0.0
            result.total_time_s = sum(p.time_s for p in result.phases)
            bottleneck = max(result.phases, key=lambda p: p.time_s)
            result.bottleneck_phase = bottleneck.name
            result.bottleneck_pct = bottleneck.time_s / result.total_time_s * 100

        steps_per_hour = 3600 / result.total_time_s if result.total_time_s > 0 else 0
        tokens_per_step = b * g * s
        tokens_per_hour = tokens_per_step * steps_per_hour
        # Effective tokens: only unique prompt tokens (not gs copies)
        effective_tokens_per_step = b * s  # unique prompt processing
        effective_tok_per_hour = effective_tokens_per_step * steps_per_hour

        print(f"  {name:<25} {fmt_time(result.total_time_s):>12} {steps_per_hour:>9.2f} "
              f"{tokens_per_step:>11,} {tokens_per_hour:>13,} {effective_tok_per_hour:>15,}")

    # ─── Section 5: Bottleneck Identification ────────────────────────────

    print(f"\n{'='*100}")
    print(f"  SECTION 5: BOTTLENECK IDENTIFICATION")
    print(f"{'='*100}")

    result_default = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                                   reward_type="rule", lora=False, mini_batch=4)

    # Sort phases by time
    sorted_phases = sorted(result_default.phases, key=lambda p: p.time_s, reverse=True)

    print(f"\n  Top bottlenecks (sorted by time contribution):")
    for i, p in enumerate(sorted_phases[:5]):
        pct = p.time_s / result_default.total_time_s * 100
        print(f"  #{i+1}: P{p.phase_id} {p.name} - {fmt_time(p.time_s)} ({pct:.1f}% of step)")

    # Bottleneck analysis
    rollout_pct = result_default.phases[1].time_s / result_default.total_time_s * 100
    training_pct = (result_default.phases[6].time_s + result_default.phases[7].time_s +
                   result_default.phases[9].time_s) / result_default.total_time_s * 100

    print(f"\n  Rollout (P1):       {rollout_pct:.1f}% of step time")
    print(f"  Training (P6+P7+P9): {training_pct:.1f}% of step time")
    print(f"  Other:              {100-rollout_pct-training_pct:.1f}% of step time")

    if rollout_pct > 50:
        print(f"\n  >>> ROLLOUT IS DOMINANT BOTTLENECK")
        print(f"  >>> Optimization: batch more prompts, increase throughput via prefix caching")
        print(f"  >>> Or use disaggregated inference (separate GPU for rollout)")
    elif training_pct > 50:
        print(f"\n  >>> TRAINING IS DOMINANT BOTTLENECK")
        print(f"  >>> Optimization: LoRA to reduce gradient/optimizer overhead")
        print(f"  >>> Or gradient accumulation to reduce update frequency")

    # Memory bottleneck
    p7_mem = result_default.phases[7].peak_mem_gib
    print(f"\n  Memory bottleneck: Phase 7 peak = {p7_mem:.2f} GiB / 24 GiB ({p7_mem/24*100:.1f}%)")
    if p7_mem > 24:
        print(f"  >>> Phase 7 OOM: actor({model_mem:.2f}G) + ref({model_mem:.2f}G) = {2*model_mem:.2f}G > 24G")
        print(f"  >>> Must bypass ref forward or offload ref model to CPU")

    # ─── Section 6: Optimization Recommendations ─────────────────────────

    print(f"\n{'='*100}")
    print(f"  SECTION 6: OPTIMIZATION RECOMMENDATIONS FOR RTX 4090")
    print(f"{'='*100}")

    recommendations = [
        ("CRITICAL", "Use bypass mode (skip ref log prob phase)",
         f"Saves {model_mem:.2f} GiB memory and ref forward time. Essential for 24 GiB."),
        ("CRITICAL", "Use LoRA instead of full parameter update",
         f"Reduces optimizer+gradient memory by {(mspec['params']*8+mspec['params']*2)/(1024**3) - (mspec['params']*0.004*8+mspec['params']*0.004*2)/(1024**3):.2f} GiB."),
        ("HIGH", "Optimize rollout throughput",
         f"Rollout is {rollout_pct:.1f}% of step. Use prefix caching, batch efficiently."),
        ("HIGH", "Use SGLang with RadixAttention for shared prefix",
         "Reduces redundant computation for prompt prefix across gs samples."),
        ("MEDIUM", "Use gradient accumulation (2-4 steps)",
         "Amortize optimizer step across multiple batches, reduce peak memory."),
        ("MEDIUM", "Offload reference model to CPU during training phases",
         f"Ref model ({model_mem:.2f} GiB) only needed briefly for phase 7."),
        ("MEDIUM", "Reduce seq_len from 1024 to 512 for initial training",
         "Halves KV cache, activation memory, and rollout time."),
        ("LOW", "Use mixed precision with gradient scaling",
         "May reduce memory slightly, but bf16 is already efficient."),
        ("LOW", "Consider dp=2 with PCIe (limited by 32 GB/s bandwidth)",
         f"NCCL overhead significant: {fmt_time(nccl_allreduce_time(mspec, gpu, 2))} per AllReduce."),
    ]

    for priority, rec, detail in recommendations:
        print(f"\n  [{priority}] {rec}")
        print(f"       {detail}")

    # Final verdict
    print(f"\n{'='*100}")
    print(f"  FINAL VERDICT: RTX 4090 GRPO VIABILITY")
    print(f"{'='*100}")

    # Check various configurations for viability
    viable_configs = []

    # Default
    r = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                      reward_type="rule", lora=False, mini_batch=4)
    peak = max(p.peak_mem_gib for p in r.phases)
    if peak > 24:
        print(f"\n  Default config: NOT VIABLE (peak {peak:.2f} GiB > 24 GiB)")
    else:
        print(f"\n  Default config: VIABLE (peak {peak:.2f} GiB < 24 GiB)")
        viable_configs.append(("Default", r))

    # Bypass
    r = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                      reward_type="rule", lora=False, mini_batch=4)
    r.phases[7].time_s = 0.0
    r.phases[7].peak_mem_gib = 0.0
    r.total_time_s = sum(p.time_s for p in r.phases)
    peak = max(p.peak_mem_gib for p in r.phases)
    if peak > 24:
        print(f"  Bypass config: NOT VIABLE (peak {peak:.2f} GiB > 24 GiB)")
    else:
        print(f"  Bypass config: VIABLE (peak {peak:.2f} GiB < 24 GiB, headroom {24-peak:.2f} GiB)")
        viable_configs.append(("Bypass", r))

    # LoRA
    r = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                      reward_type="rule", lora=True, mini_batch=4)
    peak = max(p.peak_mem_gib for p in r.phases)
    if peak > 24:
        print(f"  LoRA config: NOT VIABLE (peak {peak:.2f} GiB > 24 GiB)")
    else:
        print(f"  LoRA config: VIABLE (peak {peak:.2f} GiB < 24 GiB, headroom {24-peak:.2f} GiB)")
        viable_configs.append(("LoRA", r))

    # LoRA + bypass
    r = simulate_step(mspec, gpu, batch=4, gs=4, seq_len=1024, dp=1,
                      reward_type="rule", lora=True, mini_batch=4)
    r.phases[7].time_s = 0.0
    r.phases[7].peak_mem_gib = 0.0
    r.total_time_s = sum(p.time_s for p in r.phases)
    peak = max(p.peak_mem_gib for p in r.phases)
    if peak > 24:
        print(f"  LoRA+bypass config: NOT VIABLE (peak {peak:.2f} GiB > 24 GiB)")
    else:
        print(f"  LoRA+bypass config: VIABLE (peak {peak:.2f} GiB < 24 GiB, headroom {24-peak:.2f} GiB)")
        viable_configs.append(("LoRA+bypass", r))

    # LoRA + bypass + smaller batch
    r = simulate_step(mspec, gpu, batch=2, gs=4, seq_len=512, dp=1,
                      reward_type="rule", lora=True, mini_batch=4)
    r.phases[7].time_s = 0.0
    r.phases[7].peak_mem_gib = 0.0
    r.total_time_s = sum(p.time_s for p in r.phases)
    peak = max(p.peak_mem_gib for p in r.phases)
    if peak > 24:
        print(f"  LoRA+bypass+small config: NOT VIABLE (peak {peak:.2f} GiB > 24 GiB)")
    else:
        print(f"  LoRA+bypass+small config: VIABLE (peak {peak:.2f} GiB < 24 GiB, headroom {24-peak:.2f} GiB)")
        viable_configs.append(("LoRA+bypass+small", r))

    print(f"\n  Best viable configurations:")
    for name, r in viable_configs:
        steps_hr = 3600 / r.total_time_s
        tok_hr = 4 * 4 * 1024 * steps_hr  # normalized
        print(f"    {name}: {fmt_time(r.total_time_s)}/step, {steps_hr:.1f} steps/hr")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="verl GRPO Training Step Timing Model")
    parser.add_argument("mode", choices=["simulate", "compare", "scale", "rtx4090"],
                        help="Operation mode")
    parser.add_argument("--model", default="8B", choices=list(MODEL_SPECS.keys()),
                        help="Model size (default: 8B)")
    parser.add_argument("--gpu", default="RTX_4090", choices=list(GPU_SPECS.keys()),
                        help="GPU type (default: RTX_4090)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (prompts)")
    parser.add_argument("--gs", type=int, default=4, help="Group size (samples per prompt)")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--reward", default="rule", choices=["rule", "model"],
                        help="Reward type")
    parser.add_argument("--lora", action="store_true", help="Use LoRA update")
    parser.add_argument("--mini_batch", type=int, default=4, help="Mini-batch size for gradient update")

    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulate(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "scale":
        run_scale(args)
    elif args.mode == "rtx4090":
        run_rtx4090(args)


if __name__ == "__main__":
    main()
