#!/usr/bin/env python3
"""Weight Sync Timing Simulator for RLHF/GRPO Training.

Simulates timing and memory behavior of weight synchronization across
different GPU configurations, with special focus on RTX 4090 (dp=1).

Modes:
  simulate  - Simulate a complete weight sync cycle
  compare   - Compare across configurations (models, dp, methods, engines)
  rtx4090   - RTX 4090 specific analysis with 24 GiB budget
  lifecycle - Complete training step lifecycle (10-phase verl V1)

Mathematical foundations:
  Full param transfer:  time = 2 * model_size_bf16 / PCIe_bandwidth (CPU->GPU roundtrip)
  LoRA transfer:        time = 2 * lora_params_bf16 / PCIe_bandwidth (80x smaller than full)
  Delta transfer:       time = 2 * delta_params / PCIe_bandwidth (depends on dp)
  NCCL overhead:        dp=1 -> 0 (identity broadcast), dp>1 -> AllReduce time
  Sleep/wake overhead:  sleep_level=1 ~2s, sleep_level=2 ~5s
  Memory peak:          max(rollout_peak, training_peak) NOT sum (TransferQueue decouples)
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants and hardware specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUSpec:
    name: str
    vram_gib: float
    pcie_bandwidth_gbps: float          # GB/s (PCIe bandwidth)
    cuda_cores: int
    bf16_tflops: float
    pcie_gen: int
    pcie_lanes: int


GPUS = {
    "RTX4090": GPUSpec(
        name="RTX 4090",
        vram_gib=24.0,
        pcie_bandwidth_gbps=32.0,       # PCIe 4.0 x16
        cuda_cores=16384,
        bf16_tflops=82.6,
        pcie_gen=4,
        pcie_lanes=16,
    ),
    "RTX3090": GPUSpec(
        name="RTX 3090",
        vram_gib=24.0,
        pcie_bandwidth_gbps=25.0,       # PCIe 4.0 x16 (slower controller)
        cuda_cores=10496,
        bf16_tflops=35.6,
        pcie_gen=4,
        pcie_lanes=16,
    ),
    "A100_40": GPUSpec(
        name="A100 40GiB",
        vram_gib=40.0,
        pcie_bandwidth_gbps=32.0,
        cuda_cores=6912,
        bf16_tflops=156.0,
        pcie_gen=4,
        pcie_lanes=16,
    ),
    "A100_80": GPUSpec(
        name="A100 80GiB",
        vram_gib=80.0,
        pcie_bandwidth_gbps=64.0,       # NVLink 600 GB/s but PCIe fallback 32
        cuda_cores=6912,
        bf16_tflops=156.0,
        pcie_gen=5,
        pcie_lanes=16,
    ),
    "H100": GPUSpec(
        name="H100 SXM",
        vram_gib=80.0,
        pcie_bandwidth_gbps=64.0,
        cuda_cores=16896,
        bf16_tflops=393.0,
        pcie_gen=5,
        pcie_lanes=16,
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    bf16_gib: float                     # Total bf16 parameter memory in GiB
    active_params_gib: float            # Active params (for MoE, only active expert params)
    is_moe: bool = False
    num_experts: int = 0
    lora_default_rank: int = 32


MODELS = {
    "7B": ModelSpec(name="7B", bf16_gib=14.0, active_params_gib=14.0, lora_default_rank=32),
    "8B": ModelSpec(name="8B", bf16_gib=16.0, active_params_gib=16.0, lora_default_rank=32),
    "14B": ModelSpec(name="14B", bf16_gib=28.0, active_params_gib=28.0, lora_default_rank=32),
    "30B-A3B": ModelSpec(
        name="30B-A3B (MoE)",
        bf16_gib=60.0,
        active_params_gib=6.0,          # 3B active * 2 bytes = 6 GiB
        is_moe=True,
        num_experts=8,
        lora_default_rank=16,           # MoE uses lower LoRA rank
    ),
}

# Sync methods
SYNC_METHODS = ["full", "lora", "delta", "bypass"]

# Checkpoint engines
CHECKPOINT_ENGINES = ["naive", "nccl", "hccl"]

# Sleep levels for SGLang
SLEEP_LEVELS = {
    1: {"name": "sleep_level=1 (light)", "time_s": 2.0, "mem_ops_gib": 0.5},
    2: {"name": "sleep_level=2 (deep)", "time_s": 5.0, "mem_ops_gib": 1.0},
}

# verl V1 10-phase lifecycle phases
LIFECYCLE_PHASES = [
    ("weight_sync",   "Weight sync (sleep -> transfer -> wake -> validate)"),
    ("rollout",       "Rollout / inference generation"),
    ("replay",        "Replay buffer update"),
    ("sleep",         "Sleep (free GPU for next phase)"),
    ("reward",        "Reward model scoring"),
    ("balance",       "Balance / advantage normalization"),
    ("old_log_prob",  "Old log-prob computation"),
    ("ref_log_prob",  "Reference log-prob computation"),
    ("advantage",     "Advantage computation (GRPO)"),
    ("actor_update",  "Actor update (gradient + optimizer step)"),
]


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def lora_params_gib(model: ModelSpec, rank: int) -> float:
    """Compute LoRA adapter size in GiB for a given rank.

    LoRA adds 2 low-rank matrices (A and B) per target layer.
    For a model with hidden_dim h and rank r:
      Per layer: 2 * h * r params
    Typical target layers: q_proj, k_proj, v_proj, o_proj (4 attention projections)
    For 7B: h=4096, ~32 layers, 4 projections each
      Total LoRA params = 32 * 4 * 2 * 4096 * rank = 1,048,576 * rank
    Generalized: ~num_layers * 4 * 2 * hidden_dim * rank
    """
    # Estimate hidden_dim and num_layers from model size
    # 7B: 4096 hidden, 32 layers
    # 8B: 4096 hidden, 32 layers (Llama 3)
    # 14B: 5120 hidden, 40 layers
    # 30B-A3B: 4096 hidden, 32 layers (MoE, only active experts)
    hidden_dims = {"7B": 4096, "8B": 4096, "14B": 5120, "30B-A3B": 4096}
    num_layers = {"7B": 32, "8B": 32, "14B": 40, "30B-A3B": 32}

    h = hidden_dims.get(model.name.split()[0], 4096)
    nl = num_layers.get(model.name.split()[0], 32)

    # 4 attention projections + optionally 2 MLP projections
    num_target_projections = 4  # q, k, v, o

    total_params = nl * num_target_projections * 2 * h * rank
    # bf16: 2 bytes per param
    size_bytes = total_params * 2
    return size_bytes / (1024 ** 3)


def delta_params_gib(model: ModelSpec, dp: int) -> float:
    """Compute delta weight size in GiB.

    Delta weights = only the parameters that changed between sync points.
    For dp>1, each rank holds shard = model_bf16 / dp.
    Delta is typically 10-20% of the shard for GRPO (small updates).
    """
    base = model.bf16_gib / dp if dp > 1 else model.active_params_gib
    # GRPO updates are small; ~10% of shard changes
    delta_fraction = 0.10
    return base * delta_fraction


def transfer_time_s(size_gib: float, bandwidth_gbps: float) -> float:
    """Compute transfer time for CPU -> GPU roundtrip.

    time = 2 * size / bandwidth  (roundtrip: send to GPU, then receive back)
    """
    if bandwidth_gbps <= 0:
        return float("inf")
    return 2.0 * size_gib / bandwidth_gbps


def nccl_broadcast_time_s(model: ModelSpec, dp: int, gpu: GPUSpec) -> float:
    """Compute NCCL broadcast/AllReduce time.

    dp=1: NCCL broadcast = identity (same GPU, no actual transfer) -> 0s
    dp>1: AllReduce time proportional to model_size / (interconnect_bandwidth * (dp-1)/dp)
    """
    if dp <= 1:
        return 0.0  # Identity broadcast on single GPU
    # For dp>1, AllReduce latency
    # Ring AllReduce: 2*(dp-1)/dp * model_size / bandwidth
    size_gib = model.bf16_gib / dp  # Each rank's shard
    ring_factor = 2.0 * (dp - 1) / dp
    # Assume NVLink-like bandwidth for multi-GPU or PCIe fallback
    # For consumer GPUs (RTX), interconnect is PCIe only
    bw = gpu.pcie_bandwidth_gbps
    return ring_factor * size_gib / bw


def hccl_broadcast_time_s(model: ModelSpec, dp: int, gpu: GPUSpec) -> float:
    """HCCL broadcast time (Huawei Ascend NPUs).

    Similar scaling to NCCL but with different interconnect characteristics.
    For simulation: use 1.2x NCCL time (HCCL is typically slightly slower
    on comparable hardware due to less mature optimization).
    """
    nccl_t = nccl_broadcast_time_s(model, dp, gpu)
    if nccl_t == 0.0:
        return 0.0
    return nccl_t * 1.2


def checkpoint_engine_overhead_s(engine: str, dp: int) -> float:
    """Additional overhead from checkpoint engine.

    naive: direct memcpy, minimal overhead (~0.1s setup)
    nccl:  dp=1 -> 0 (identity), dp>1 -> AllReduce time + 0.5s setup
    hccl:  similar to NCCL with 1.2x multiplier
    """
    if engine == "naive":
        return 0.1  # Minimal memcpy setup
    elif engine == "nccl":
        if dp <= 1:
            return 0.1  # NCCL identity is basically free
        return 0.5  # NCCL group setup + broadcast coordination
    elif engine == "hccl":
        return 0.6  # HCCL setup overhead
    return 0.1


def memory_at_phase(
    phase: str,
    model: ModelSpec,
    gpu: GPUSpec,
    dp: int,
    sync_method: str,
    lora_rank: int,
    engine: str,
) -> float:
    """Estimate memory (GiB) at each phase of weight sync cycle.

    TransferQueue backbone: GPU only used during compute phases,
    memory peak = max(rollout_peak, training_peak) NOT sum.
    """
    model_mem = model.bf16_gib / dp if dp > 1 else model.bf16_gib

    # Activation memory estimates (rough)
    # Rollout: model + KV cache + activations
    kv_cache_gib = 2.0  # ~2 GiB for typical sequence lengths
    rollout_act_gib = 1.5  # Generation activations

    # Training: model + gradients + optimizer states + activations
    # FSDP1+bypass: shards model, but dp=1 means full model on one GPU
    grad_gib = model_mem  # Gradients same size as params
    optimizer_gib = model_mem * 2  # Adam: 2x param size (m + v)
    train_act_gib = 3.0  # Training activations

    # LoRA reduces training memory
    lora_mem = lora_params_gib(model, lora_rank) if sync_method in ("lora", "bypass") else 0.0

    phase_memories = {
        "sleep": 0.5 + SLEEP_LEVELS[1]["mem_ops_gib"],  # GPU mostly freed, small overhead
        "transfer": model_mem * 0.1,  # Buffer for incoming weights
        "wake": model_mem + 1.0,  # Rebuilding state, model + temp
        "validate": model_mem + 0.5,  # Model + checksum buffer
        "rollout_peak": model_mem + kv_cache_gib + rollout_act_gib,
        "training_peak": model_mem + grad_gib + optimizer_gib * 0.0 + train_act_gib + lora_mem,
    }

    # For bypass method: training only uses LoRA params for gradient
    if sync_method == "bypass":
        phase_memories["training_peak"] = model_mem + lora_mem + train_act_gib + lora_mem

    # FSDP1 with dp=1: no sharding benefit, full optimizer states
    if dp == 1:
        phase_memories["training_peak"] = model_mem + grad_gib + optimizer_gib + train_act_gib + lora_mem
        # But bypass+LoRA: only LoRA gradients and optimizer
        if sync_method == "bypass":
            phase_memories["training_peak"] = model_mem + lora_mem * 2 + lora_mem * 2 + train_act_gib

    return phase_memories.get(phase, 0.0)


def total_memory_peak(
    model: ModelSpec,
    gpu: GPUSpec,
    dp: int,
    sync_method: str,
    lora_rank: int,
) -> float:
    """Total memory peak = max(rollout_peak, training_peak).

    TransferQueue decouples rollout and training, so peak is max not sum.
    """
    rollout_peak = memory_at_phase("rollout_peak", model, gpu, dp, sync_method, lora_rank, "naive")
    training_peak = memory_at_phase("training_peak", model, gpu, dp, sync_method, lora_rank, "naive")
    return max(rollout_peak, training_peak)


# ---------------------------------------------------------------------------
# Mode 1: simulate
# ---------------------------------------------------------------------------

def run_simulate(
    model_name: str = "8B",
    gpu_name: str = "RTX4090",
    dp: int = 1,
    sync_method: str = "lora",
    lora_rank: int = 32,
    engine: str = "naive",
    sleep_level: int = 1,
) -> str:
    model = MODELS[model_name]
    gpu = GPUS[gpu_name]

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"  WEIGHT SYNC SIMULATION: {model.name} on {gpu.name}, dp={dp}")
    lines.append(f"  Method: {sync_method}, Engine: {engine}, LoRA rank: {lora_rank}")
    lines.append("=" * 78)
    lines.append("")

    # Phase 1: Sleep (free GPU)
    sleep_info = SLEEP_LEVELS[sleep_level]
    sleep_time = sleep_info["time_s"]
    sleep_mem_ops = sleep_info["mem_ops_gib"]
    lines.append("--- Phase 1: SLEEP (free GPU) ---")
    lines.append(f"  SGLang sleep_level={sleep_level}")
    lines.append(f"  Time:       {sleep_time:.1f} s")
    lines.append(f"  Mem ops:    {sleep_mem_ops:.1f} GiB (cleanup)")
    lines.append(f"  GPU freed:  ~{model.bf16_gib:.1f} GiB reclaimed")
    lines.append("")

    # Phase 2: Weight transfer (CPU -> GPU)
    if sync_method == "full":
        transfer_size = model.bf16_gib
        transfer_desc = "Full parameter sync (all bf16 weights)"
    elif sync_method == "lora":
        transfer_size = lora_params_gib(model, lora_rank)
        transfer_desc = f"LoRA adapter sync (rank={lora_rank}, {transfer_size:.4f} GiB)"
    elif sync_method == "delta":
        transfer_size = delta_params_gib(model, dp)
        transfer_desc = f"Delta weight sync ({transfer_size:.4f} GiB, ~10% of shard)"
    elif sync_method == "bypass":
        transfer_size = lora_params_gib(model, lora_rank)
        transfer_desc = f"Bypass: LoRA adapter only (rank={lora_rank}, {transfer_size:.4f} GiB)"
    else:
        transfer_size = model.bf16_gib
        transfer_desc = "Unknown method, assuming full"

    pcie_time = transfer_time_s(transfer_size, gpu.pcie_bandwidth_gbps)
    nccl_time = nccl_broadcast_time_s(model, dp, gpu) if engine == "nccl" else 0.0
    hccl_time = hccl_broadcast_time_s(model, dp, gpu) if engine == "hccl" else 0.0
    engine_overhead = checkpoint_engine_overhead_s(engine, dp)
    total_transfer_time = pcie_time + nccl_time + hccl_time + engine_overhead

    lines.append("--- Phase 2: WEIGHT TRANSFER (CPU -> GPU) ---")
    lines.append(f"  Method:     {transfer_desc}")
    lines.append(f"  Transfer:   {transfer_size:.4f} GiB")
    lines.append(f"  PCIe time:  {pcie_time:.3f} s (roundtrip @ {gpu.pcie_bandwidth_gbps:.0f} GB/s)")
    if engine == "nccl":
        lines.append(f"  NCCL time:  {nccl_time:.3f} s (dp={dp}: {'identity=0' if dp==1 else 'AllReduce'})")
    elif engine == "hccl":
        lines.append(f"  HCCL time:  {hccl_time:.3f} s")
    lines.append(f"  Engine oh:  {engine_overhead:.1f} s ({engine} setup)")
    lines.append(f"  Total:      {total_transfer_time:.3f} s")
    lines.append("")

    # Phase 3: Wake (rebuild state)
    wake_time = 1.0  # SGLang wake: rebuild CUDA graphs, KV cache state
    wake_mem = memory_at_phase("wake", model, gpu, dp, sync_method, lora_rank, engine)
    lines.append("--- Phase 3: WAKE (rebuild state) ---")
    lines.append(f"  Time:       {wake_time:.1f} s (SGLang wake: rebuild CUDA graphs)")
    lines.append(f"  Memory:     {wake_mem:.1f} GiB (model + rebuild buffers)")
    lines.append("")

    # Phase 4: Validate (checksum)
    validate_time = 0.5  # SHA256 or simple checksum over params
    validate_mem = memory_at_phase("validate", model, gpu, dp, sync_method, lora_rank, engine)
    lines.append("--- Phase 4: VALIDATE (checksum) ---")
    lines.append(f"  Time:       {validate_time:.1f} s")
    lines.append(f"  Memory:     {validate_mem:.1f} GiB (model + checksum buffer)")
    lines.append("")

    # Total sync cycle
    total_sync_time = sleep_time + total_transfer_time + wake_time + validate_time
    lines.append("--- TOTAL SYNC CYCLE ---")
    lines.append(f"  Sleep:      {sleep_time:.1f} s")
    lines.append(f"  Transfer:   {total_transfer_time:.3f} s")
    lines.append(f"  Wake:       {wake_time:.1f} s")
    lines.append(f"  Validate:   {validate_time:.1f} s")
    lines.append(f"  TOTAL:      {total_sync_time:.3f} s")
    lines.append("")

    # Memory timeline
    lines.append("--- MEMORY TIMELINE ---")
    mem_sleep = memory_at_phase("sleep", model, gpu, dp, sync_method, lora_rank, engine)
    mem_transfer = memory_at_phase("transfer", model, gpu, dp, sync_method, lora_rank, engine)
    peak = total_memory_peak(model, gpu, dp, sync_method, lora_rank)
    oom = peak > gpu.vram_gib

    lines.append(f"  Phase         Memory (GiB)   vs Budget ({gpu.vram_gib:.0f} GiB)")
    lines.append(f"  Sleep:         {mem_sleep:>6.1f}         {'OK' if mem_sleep <= gpu.vram_gib else 'OOM'}")
    lines.append(f"  Transfer:      {mem_transfer:>6.1f}         {'OK' if mem_transfer <= gpu.vram_gib else 'OOM'}")
    lines.append(f"  Wake:          {wake_mem:>6.1f}         {'OK' if wake_mem <= gpu.vram_gib else 'OOM'}")
    lines.append(f"  Validate:      {validate_mem:>6.1f}         {'OK' if validate_mem <= gpu.vram_gib else 'OOM'}")
    lines.append(f"  Peak (max):    {peak:>6.1f}         {'*** OOM ***' if oom else 'OK'}")
    lines.append("")

    # Comparison of all sync methods
    lines.append("--- SYNC METHOD COMPARISON ---")
    lines.append(f"  {'Method':<10} {'Size (GiB)':<12} {'PCIe Time (s)':<16} {'Total (s)':<12} {'Viability'}")
    for m in SYNC_METHODS:
        if m == "full":
            sz = model.bf16_gib
        elif m == "lora":
            sz = lora_params_gib(model, lora_rank)
        elif m == "delta":
            sz = delta_params_gib(model, dp)
        elif m == "bypass":
            sz = lora_params_gib(model, lora_rank)
        else:
            sz = model.bf16_gib
        t_pcie = transfer_time_s(sz, gpu.pcie_bandwidth_gbps)
        t_engine = checkpoint_engine_overhead_s(engine, dp)
        t_nccl = nccl_broadcast_time_s(model, dp, gpu) if engine == "nccl" else 0.0
        t_total = sleep_time + t_pcie + t_nccl + t_engine + wake_time + validate_time
        p = total_memory_peak(model, gpu, dp, m, lora_rank)
        viable = "VIABLE" if p <= gpu.vram_gib else "OOM"
        lines.append(f"  {m:<10} {sz:<12.4f} {t_pcie:<16.3f} {t_total:<12.3f} {viable}")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode 2: compare
# ---------------------------------------------------------------------------

def run_compare() -> str:
    lines: List[str] = []
    lines.append("=" * 110)
    lines.append("  WEIGHT SYNC COMPARISON ACROSS CONFIGURATIONS")
    lines.append("=" * 110)
    lines.append("")

    # Table header
    hdr = (
        f"  {'Model':<14} {'dp':<4} {'Method':<8} {'Engine':<6} "
        f"{'Size(GiB)':<10} {'PCIe(s)':<10} {'NCCL(s)':<10} "
        f"{'Total(s)':<10} {'Peak(GiB)':<10} {'Budget':<8} {'Status'}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 104)

    for model_name, model in MODELS.items():
        for dp in [1, 2, 4]:
            for sync_method in SYNC_METHODS:
                for engine in CHECKPOINT_ENGINES:
                    gpu = GPUS["RTX4090"]  # Default comparison on RTX 4090

                    # Compute transfer size
                    if sync_method == "full":
                        sz = model.bf16_gib
                    elif sync_method == "lora":
                        sz = lora_params_gib(model, model.lora_default_rank)
                    elif sync_method == "delta":
                        sz = delta_params_gib(model, dp)
                    elif sync_method == "bypass":
                        sz = lora_params_gib(model, model.lora_default_rank)
                    else:
                        sz = model.bf16_gib

                    t_pcie = transfer_time_s(sz, gpu.pcie_bandwidth_gbps)
                    t_nccl = nccl_broadcast_time_s(model, dp, gpu) if engine == "nccl" else 0.0
                    t_hccl = hccl_broadcast_time_s(model, dp, gpu) if engine == "hccl" else 0.0
                    t_engine = checkpoint_engine_overhead_s(engine, dp)

                    # Full sync cycle
                    sleep_t = SLEEP_LEVELS[1]["time_s"]
                    total_t = sleep_t + t_pcie + t_nccl + t_hccl + t_engine + 1.0 + 0.5

                    peak = total_memory_peak(model, gpu, dp, sync_method, model.lora_default_rank)
                    budget = gpu.vram_gib
                    status = "VIABLE" if peak <= budget else "OOM"

                    nccl_str = f"{t_nccl + t_hccl:.3f}" if (t_nccl + t_hccl) > 0 else "0 (dp=1)"

                    lines.append(
                        f"  {model.name:<14} {dp:<4} {sync_method:<8} {engine:<6} "
                        f"{sz:<10.4f} {t_pcie:<10.3f} {nccl_str:<10} "
                        f"{total_t:<10.3f} {peak:<10.1f} {budget:<8.0f} {status}"
                    )
        lines.append("  " + "-" * 104)

    lines.append("")
    lines.append("  Key observations:")
    lines.append("    - dp=1 with NCCL: broadcast = identity, zero NCCL overhead")
    lines.append("    - LoRA/bypass: ~80x smaller transfer than full param sync")
    lines.append("    - 30B-A3B MoE: total 60 GiB, but only 3B active params (6 GiB bf16)")
    lines.append("    - TransferQueue: peak = max(rollout, training), NOT sum")
    lines.append("    - OOM threshold: peak > GPU VRAM budget")
    lines.append("")
    lines.append("=" * 110)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode 3: rtx4090
# ---------------------------------------------------------------------------

def run_rtx4090() -> str:
    gpu = GPUS["RTX4090"]
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  RTX 4090 WEIGHT SYNC ANALYSIS")
    lines.append(f"  VRAM Budget: {gpu.vram_gib:.0f} GiB | PCIe: {gpu.pcie_bandwidth_gbps:.0f} GB/s (4.0 x16)")
    lines.append("=" * 78)
    lines.append("")

    # Known optimal config
    lines.append("--- KNOWN OPTIMAL CONFIG (verl V1 on RTX 4090) ---")
    lines.append("  sync_method:   lora (LoRA adapter sync)")
    lines.append("  engine:        naive (direct memcpy, no NCCL overhead)")
    lines.append("  fsdp:          FSDP1 (full shard, dp=1)")
    lines.append("  bypass:        bypass (skip full param rebuild, only LoRA)")
    lines.append("  lora_rank:     32 (good quality/speed tradeoff)")
    lines.append("")

    # Per-model viability
    lines.append("--- PER-MODEL VIABILITY (dp=1, naive, LoRA rank=32) ---")
    lines.append(f"  {'Model':<14} {'bf16':<8} {'LoRA':<10} {'Peak':<10} {'Budget':<8} {'Status':<10} {'OOM Risk'}")
    lines.append("  " + "-" * 60)

    for model_name, model in MODELS.items():
        lr = model.lora_default_rank
        lora_sz = lora_params_gib(model, lr)

        # Rollout peak: model weights + KV cache + activations
        rollout_peak = model.bf16_gib + 2.0 + 1.5

        # Training peak with bypass+LoRA+naive+FSDP1 (dp=1):
        # Model on GPU + LoRA gradients + LoRA optimizer states + training activations
        # FSDP1 dp=1: no sharding, full model resident
        # But bypass means we only compute gradients for LoRA params
        # Training peak = model_mem + lora_grad + lora_opt + train_act
        #   model_mem = bf16_gib (dp=1, full model)
        #   lora_grad = lora_sz (same size as LoRA params)
        #   lora_opt  = lora_sz * 2 (Adam m+v)
        #   train_act = ~3 GiB
        training_peak = model.bf16_gib + lora_sz + lora_sz * 2 + 3.0

        peak = max(rollout_peak, training_peak)
        budget = gpu.vram_gib
        status = "FIT" if peak <= budget else "OOM"
        risk = "None" if peak <= budget * 0.8 else "Moderate" if peak <= budget else "CRITICAL"

        lines.append(
            f"  {model.name:<14} {model.bf16_gib:<8.1f} {lora_sz:<10.4f} "
            f"{peak:<10.1f} {budget:<8.0f} {status:<10} {risk}"
        )
    lines.append("  " + "-" * 60)
    lines.append("")

    # Detailed memory budget breakdown for best viable config (8B)
    lines.append("--- MEMORY BUDGET BREAKDOWN: 8B (dp=1, naive, bypass+LoRA r=32) ---")
    model = MODELS["8B"]
    lr = 32
    lora_sz = lora_params_gib(model, lr)
    budget = gpu.vram_gib

    components = [
        ("Model weights (bf16)",   model.bf16_gib),
        ("KV cache (rollout)",     2.0),
        ("Rollout activations",    1.5),
        ("LoRA params (r=32)",     lora_sz),
        ("LoRA gradients",         lora_sz),
        ("LoRA optimizer (Adam)",  lora_sz * 2),
        ("Training activations",   3.0),
        ("CUDA context + misc",    1.0),
    ]

    rollout_components = [
        ("Model weights (bf16)",   model.bf16_gib),
        ("KV cache",               2.0),
        ("Rollout activations",    1.5),
        ("CUDA context + misc",    1.0),
    ]

    training_components = [
        ("Model weights (bf16)",   model.bf16_gib),
        ("LoRA params (r=32)",     lora_sz),
        ("LoRA gradients",         lora_sz),
        ("LoRA optimizer (Adam)",  lora_sz * 2),
        ("Training activations",   3.0),
        ("CUDA context + misc",    1.0),
    ]

    lines.append(f"  {'Component':<30} {'Size (GiB)':<12} {'Cumulative':<12} {'vs Budget'}")
    lines.append("  " + "-" * 65)

    # Rollout peak breakdown
    lines.append("  [Rollout phase (generation)]")
    cum = 0.0
    for name, sz in rollout_components:
        cum += sz
        pct = cum / budget * 100
        lines.append(f"  {name:<30} {sz:<12.2f} {cum:<12.2f} {pct:.1f}%")
    rollout_peak = cum
    lines.append(f"  {'ROLLOUT PEAK':<30} {'':12} {rollout_peak:<12.2f} {rollout_peak/budget*100:.1f}%")
    lines.append("")

    # Training peak breakdown
    lines.append("  [Training phase (actor update)]")
    cum = 0.0
    for name, sz in training_components:
        cum += sz
        pct = cum / budget * 100
        lines.append(f"  {name:<30} {sz:<12.2f} {cum:<12.2f} {pct:.1f}%")
    training_peak = cum
    lines.append(f"  {'TRAINING PEAK':<30} {'':12} {training_peak:<12.2f} {training_peak/budget*100:.1f}%")
    lines.append("")

    # Actual peak (TransferQueue decouples)
    actual_peak = max(rollout_peak, training_peak)
    remaining = budget - actual_peak
    lines.append(f"  Actual peak (TransferQueue decoupled) = max(rollout, training)")
    lines.append(f"  Peak: {actual_peak:.2f} GiB | Remaining: {remaining:.2f} GiB | {actual_peak/budget*100:.1f}% of budget")
    lines.append("")

    # Sync timing for optimal config
    lines.append("--- SYNC TIMING (8B, dp=1, naive, bypass+LoRA r=32) ---")
    lora_sz = lora_params_gib(model, lr)
    sleep_t = SLEEP_LEVELS[1]["time_s"]
    pcie_t = transfer_time_s(lora_sz, gpu.pcie_bandwidth_gbps)
    engine_t = checkpoint_engine_overhead_s("naive", 1)
    wake_t = 1.0
    validate_t = 0.5
    total_t = sleep_t + pcie_t + engine_t + wake_t + validate_t

    lines.append(f"  Sleep (level=1):   {sleep_t:.1f} s")
    lines.append(f"  LoRA transfer:     {pcie_t:.3f} s ({lora_sz:.4f} GiB @ {gpu.pcie_bandwidth_gbps:.0f} GB/s roundtrip)")
    lines.append(f"  Engine (naive):    {engine_t:.1f} s")
    lines.append(f"  Wake:              {wake_t:.1f} s")
    lines.append(f"  Validate:          {validate_t:.1f} s")
    lines.append(f"  TOTAL SYNC:        {total_t:.3f} s")
    lines.append("")

    # Compare sync methods on RTX 4090 for 8B
    lines.append("--- SYNC METHOD COMPARISON (8B on RTX 4090) ---")
    lines.append(f"  {'Method':<10} {'Size (GiB)':<12} {'Transfer (s)':<14} {'NCCL (s)':<10} {'Total (s)':<10} {'Peak (GiB)':<10} {'Viable'}")
    lines.append("  " + "-" * 70)

    for m in SYNC_METHODS:
        if m == "full":
            sz = model.bf16_gib
        elif m == "lora":
            sz = lora_params_gib(model, lr)
        elif m == "delta":
            sz = delta_params_gib(model, 1)
        elif m == "bypass":
            sz = lora_params_gib(model, lr)
        else:
            sz = model.bf16_gib

        t_pcie = transfer_time_s(sz, gpu.pcie_bandwidth_gbps)
        t_nccl = nccl_broadcast_time_s(model, 1, gpu)  # dp=1 -> 0
        t_engine = checkpoint_engine_overhead_s("naive", 1)
        total = SLEEP_LEVELS[1]["time_s"] + t_pcie + t_nccl + t_engine + 1.0 + 0.5

        peak = total_memory_peak(model, gpu, 1, m, lr)
        viable = "YES" if peak <= gpu.vram_gib else "OOM"

        lines.append(
            f"  {m:<10} {sz:<12.4f} {t_pcie:<14.3f} {t_nccl:<10.3f} "
            f"{total:<10.3f} {peak:<10.1f} {viable}"
        )
    lines.append("  " + "-" * 70)
    lines.append("")

    # OOM risk points
    lines.append("--- OOM RISK POINTS (RTX 4090, 24 GiB) ---")
    for model_name, model in MODELS.items():
        peak = total_memory_peak(model, gpu, 1, "bypass", model.lora_default_rank)
        if peak > gpu.vram_gib:
            lines.append(f"  {model.name}: OOM at {peak:.1f} GiB > {gpu.vram_gib:.0f} GiB budget")
            lines.append(f"    - Cannot fit even with bypass+LoRA; need dp>=2 or smaller model")
        elif peak > gpu.vram_gib * 0.8:
            lines.append(f"  {model.name}: Near-OOM at {peak:.1f} GiB ({peak/gpu.vram_gib*100:.0f}% of budget)")
            lines.append(f"    - Tight margin; any activation spike could OOM")
        else:
            lines.append(f"  {model.name}: Safe at {peak:.1f} GiB ({peak/gpu.vram_gib*100:.0f}% of budget)")
            lines.append(f"    - Adequate headroom for activation variance")
    lines.append("")

    # Recommended config
    lines.append("--- RECOMMENDED CONFIG (RTX 4090 dp=1) ---")
    lines.append("  Model:          8B (Llama 3)")
    lines.append("  Sync method:    bypass (LoRA adapter only)")
    lines.append("  Engine:         naive (no NCCL overhead)")
    lines.append("  FSDP:           FSDP1 (dp=1, full model resident)")
    lines.append("  LoRA rank:      32")
    lines.append("  Sleep level:    1 (2s overhead)")
    lines.append("  Peak memory:    {:.1f} GiB / {:.0f} GiB budget".format(
        total_memory_peak(MODELS["8B"], gpu, 1, "bypass", 32), gpu.vram_gib
    ))
    lines.append("  Sync time:      {:.3f} s per cycle".format(
        SLEEP_LEVELS[1]["time_s"] + transfer_time_s(lora_params_gib(MODELS["8B"], 32), gpu.pcie_bandwidth_gbps)
        + checkpoint_engine_overhead_s("naive", 1) + 1.0 + 0.5
    ))
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode 4: lifecycle
# ---------------------------------------------------------------------------

def run_lifecycle(
    model_name: str = "8B",
    gpu_name: str = "RTX4090",
    dp: int = 1,
    sync_method: str = "bypass",
    lora_rank: int = 32,
    engine: str = "naive",
) -> str:
    model = MODELS[model_name]
    gpu = GPUS[gpu_name]

    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(f"  TRAINING STEP LIFECYCLE (verl V1)")
    lines.append(f"  {model.name} on {gpu.name}, dp={dp}, {sync_method}+{engine}, LoRA r={lora_rank}")
    lines.append("=" * 90)
    lines.append("")

    # Compute timing per phase based on model and config
    # These are empirical estimates scaled by model size

    # Base timing for 8B reference model on RTX 4090
    # Then scale proportionally for other model sizes
    size_scale = model.bf16_gib / 16.0  # 8B = 16 GiB is reference

    # Phase timing estimates
    phase_timings: Dict[str, Tuple[float, str]] = {}

    # 1. weight_sync: from simulate mode
    if sync_method == "full":
        transfer_sz = model.bf16_gib
    elif sync_method in ("lora", "bypass"):
        transfer_sz = lora_params_gib(model, lora_rank)
    elif sync_method == "delta":
        transfer_sz = delta_params_gib(model, dp)
    else:
        transfer_sz = model.bf16_gib

    sleep_t = SLEEP_LEVELS[1]["time_s"]
    transfer_t = transfer_time_s(transfer_sz, gpu.pcie_bandwidth_gbps)
    engine_t = checkpoint_engine_overhead_s(engine, dp)
    nccl_t = nccl_broadcast_time_s(model, dp, gpu) if engine == "nccl" else 0.0
    hccl_t = hccl_broadcast_time_s(model, dp, gpu) if engine == "hccl" else 0.0

    sync_total = sleep_t + transfer_t + nccl_t + hccl_t + engine_t + 1.0 + 0.5
    phase_timings["weight_sync"] = (sync_total, "sleep+transfer+wake+validate")

    # 2. rollout: inference generation, ~5-15s depending on prompt/batch size
    # Scales with active params (for MoE, only active params)
    rollout_time = 8.0 * (model.active_params_gib / 16.0) / gpu.bf16_tflops * 82.6
    rollout_time = max(5.0, min(20.0, rollout_time))  # Clamp to reasonable range
    phase_timings["rollout"] = (rollout_time, f"generation ({model.active_params_gib:.1f} GiB active)")

    # 3. replay: buffer update, fast
    phase_timings["replay"] = (0.2, "buffer insert + sort")

    # 4. sleep: free GPU (this is the sleep within lifecycle, not sync sleep)
    phase_timings["sleep"] = (SLEEP_LEVELS[1]["time_s"] * 0.3, "partial GPU free for reward")

    # 5. reward: reward model scoring
    # Reward model is typically smaller (~1-3B), but batch scoring
    reward_time = 3.0 * size_scale / max(1, dp)
    phase_timings["reward"] = (reward_time, "reward model inference")

    # 6. balance: advantage normalization, fast compute
    phase_timings["balance"] = (0.1, "Whiten/normalize advantages")

    # 7. old_log_prob: forward pass on old policy (no grad)
    old_lp_time = 2.0 * model.active_params_gib / 16.0 / max(1, dp)
    phase_timings["old_log_prob"] = (old_lp_time, "forward pass (no grad)")

    # 8. ref_log_prob: forward pass on reference policy (no grad)
    ref_lp_time = 2.0 * model.active_params_gib / 16.0 / max(1, dp)
    phase_timings["ref_log_prob"] = (ref_lp_time, "forward pass (no grad)")

    # 9. advantage: compute GRPO advantages, fast
    phase_timings["advantage"] = (0.15, "GRPO advantage = (r - mean) / std")

    # 10. actor_update: LoRA backward + optimizer step
    if sync_method in ("lora", "bypass"):
        # LoRA backward is much faster than full
        actor_time = 4.0 * lora_params_gib(model, lora_rank) / lora_params_gib(MODELS["8B"], 32)
    else:
        # Full backward
        actor_time = 8.0 * size_scale / max(1, dp)
    phase_timings["actor_update"] = (actor_time, "backward + optimizer step")

    # Memory per phase
    phase_memories: Dict[str, float] = {}
    model_mem = model.bf16_gib / dp if dp > 1 else model.bf16_gib
    lora_sz = lora_params_gib(model, lora_rank)

    # TransferQueue backbone: GPU only during compute phases
    phase_memories["weight_sync"] = 0.5 + SLEEP_LEVELS[1]["mem_ops_gib"]
    phase_memories["rollout"] = model_mem + 2.0 + 1.5  # model + KV + act
    phase_memories["replay"] = 0.5  # CPU buffer, minimal GPU
    phase_memories["sleep"] = 0.5  # GPU mostly freed
    phase_memories["reward"] = 3.0  # Reward model (small, ~1-3B)
    phase_memories["balance"] = 0.3  # Tensor ops, minimal
    phase_memories["old_log_prob"] = model_mem + 2.0  # Forward, model + act
    phase_memories["ref_log_prob"] = model_mem + 2.0  # Forward, model + act
    phase_memories["advantage"] = 0.2  # Small tensors
    phase_memories["actor_update"] = model_mem + lora_sz + lora_sz * 2 + 3.0  # model + LoRA + opt + act

    # Render lifecycle table
    lines.append(f"  {'Phase':<22} {'Description':<36} {'Time (s)':<10} {'Mem (GiB)':<10} {'GPU':<6} {'vs Budget'}")
    lines.append("  " + "-" * 84)

    total_time = 0.0
    gpu_active_time = 0.0

    for phase_id, (phase_name, description) in enumerate(LIFECYCLE_PHASES, 1):
        key = phase_name
        time_s, time_desc = phase_timings[key]
        mem_gib = phase_memories[key]
        gpu_state = "GPU" if mem_gib > 2.0 else "free"
        budget_pct = mem_gib / gpu.vram_gib * 100
        budget_str = f"{budget_pct:.0f}% {'OK' if mem_gib <= gpu.vram_gib else 'OOM'}"

        total_time += time_s
        if gpu_state == "GPU":
            gpu_active_time += time_s

        lines.append(
            f"  {phase_id}. {phase_name:<20} {description:<36} "
            f"{time_s:<10.3f} {mem_gib:<10.1f} {gpu_state:<6} {budget_str}"
        )

    lines.append("  " + "-" * 84)
    lines.append(f"  {'TOTAL STEP':<22} {'':36} {total_time:<10.3f} {'(peak below)':10} {'GPU':6}")
    lines.append(f"  GPU active:    {gpu_active_time:.3f} s / {total_time:.3f} s = {gpu_active_time/total_time*100:.1f}% utilization")
    lines.append("")

    # Peak memory analysis
    peak_phase = max(phase_memories, key=phase_memories.get)
    peak_mem = phase_memories[peak_phase]
    lines.append("--- MEMORY PEAK ANALYSIS ---")
    lines.append(f"  Peak phase:    {peak_phase} at {peak_mem:.1f} GiB")
    lines.append(f"  Budget:        {gpu.vram_gib:.0f} GiB")
    lines.append(f"  Headroom:      {gpu.vram_gib - peak_mem:.1f} GiB ({(gpu.vram_gib - peak_mem)/gpu.vram_gib*100:.0f}%)")
    lines.append(f"  OOM risk:      {'LOW' if peak_mem <= gpu.vram_gib * 0.8 else 'MODERATE' if peak_mem <= gpu.vram_gib else 'HIGH'}")
    lines.append("")

    # How weight sync fits
    lines.append("--- WEIGHT SYNC IN CONTEXT ---")
    sync_pct = phase_timings["weight_sync"][0] / total_time * 100
    lines.append(f"  Weight sync:    {phase_timings['weight_sync'][0]:.3f} s ({sync_pct:.1f}% of step)")
    lines.append(f"  Compute:        actor_update = {phase_timings['actor_update'][0]:.3f} s ({phase_timings['actor_update'][0]/total_time*100:.1f}%)")
    lines.append(f"  Rollout:        {phase_timings['rollout'][0]:.3f} s ({phase_timings['rollout'][0]/total_time*100:.1f}%)")
    lines.append(f"  Sync overhead is {'DOMINANT' if sync_pct > 30 else 'SIGNIFICANT' if sync_pct > 15 else 'MINIMAL'}")
    lines.append("  With LoRA bypass, sync overhead drops from ~30s (full) to <4s")
    lines.append("  TransferQueue ensures GPU is freed during non-compute phases")
    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Weight Sync Timing Simulator for RLHF/GRPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  simulate   Simulate a complete weight sync cycle
  compare    Compare across configurations (models, dp, methods, engines)
  rtx4090    RTX 4090 specific analysis with 24 GiB budget
  lifecycle  Complete training step lifecycle (10-phase verl V1)

Examples:
  python weight_sync_timing_simulator.py simulate
  python weight_sync_timing_simulator.py simulate --model 14B --method delta --dp 2
  python weight_sync_timing_simulator.py compare
  python weight_sync_timing_simulator.py rtx4090
  python weight_sync_timing_simulator.py lifecycle
  python weight_sync_timing_simulator.py lifecycle --model 30B-A3B --method lora --dp 4
""",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Simulation mode")

    # Mode 1: simulate
    sim_parser = subparsers.add_parser("simulate", help="Simulate a complete weight sync cycle")
    sim_parser.add_argument("--model", default="8B", choices=list(MODELS.keys()),
                            help="Model size (default: 8B)")
    sim_parser.add_argument("--gpu", default="RTX4090", choices=list(GPUS.keys()),
                            help="GPU type (default: RTX4090)")
    sim_parser.add_argument("--dp", type=int, default=1, choices=[1, 2, 4, 8],
                            help="Data parallelism (default: 1)")
    sim_parser.add_argument("--method", default="lora", choices=SYNC_METHODS,
                            help="Sync method (default: lora)")
    sim_parser.add_argument("--lora-rank", type=int, default=32,
                            help="LoRA rank (default: 32)")
    sim_parser.add_argument("--engine", default="naive", choices=CHECKPOINT_ENGINES,
                            help="Checkpoint engine (default: naive)")
    sim_parser.add_argument("--sleep-level", type=int, default=1, choices=[1, 2],
                            help="SGLang sleep level (default: 1)")

    # Mode 2: compare
    cmp_parser = subparsers.add_parser("compare", help="Compare across configurations")
    cmp_parser.add_argument("--gpu", default="RTX4090", choices=list(GPUS.keys()),
                            help="GPU type for comparison (default: RTX4090)")

    # Mode 3: rtx4090
    rtx_parser = subparsers.add_parser("rtx4090", help="RTX 4090 specific analysis")

    # Mode 4: lifecycle
    lc_parser = subparsers.add_parser("lifecycle", help="Complete training step lifecycle")
    lc_parser.add_argument("--model", default="8B", choices=list(MODELS.keys()),
                           help="Model size (default: 8B)")
    lc_parser.add_argument("--gpu", default="RTX4090", choices=list(GPUS.keys()),
                           help="GPU type (default: RTX4090)")
    lc_parser.add_argument("--dp", type=int, default=1, choices=[1, 2, 4, 8],
                           help="Data parallelism (default: 1)")
    lc_parser.add_argument("--method", default="bypass", choices=SYNC_METHODS,
                           help="Sync method (default: bypass)")
    lc_parser.add_argument("--lora-rank", type=int, default=32,
                           help="LoRA rank (default: 32)")
    lc_parser.add_argument("--engine", default="naive", choices=CHECKPOINT_ENGINES,
                           help="Checkpoint engine (default: naive)")

    args = parser.parse_args()

    if args.mode == "simulate":
        output = run_simulate(
            model_name=args.model,
            gpu_name=args.gpu,
            dp=args.dp,
            sync_method=args.method,
            lora_rank=args.lora_rank,
            engine=args.engine,
            sleep_level=args.sleep_level,
        )
    elif args.mode == "compare":
        output = run_compare()
    elif args.mode == "rtx4090":
        output = run_rtx4090()
    elif args.mode == "lifecycle":
        output = run_lifecycle(
            model_name=args.model,
            gpu_name=args.gpu,
            dp=args.dp,
            sync_method=args.method,
            lora_rank=args.lora_rank,
            engine=args.engine,
        )
    else:
        parser.print_help()
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    main()
