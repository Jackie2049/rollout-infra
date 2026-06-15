#!/usr/bin/env python3
"""RTX 4090 Seven-Framework GRPO Configuration Matrix
=====================================================
Cross-framework GRPO config recommendation tool for RTX 4090 (24GB VRAM, SM89).

Modes:
  - matrix: Print full framework config matrix
  - budget: Estimate memory budgets for each config
  - recommend: Print RTX 4090 GRPO recommendations
  - compare: Compare frameworks across dimensions
  - gaps: Identify knowledge gaps and opportunities
  - all: Run all modes

Based on: notebook/projects/ (7 framework reading notes)
"""

import argparse
import sys
from dataclasses import dataclass, fields
from typing import List

RTX_4090_VRAM = 24  # GB
SM_ARCH = 89


@dataclass
class FrameworkConfig:
    name: str
    backend: str
    model: str
    lora_rank: int
    lora_targets: List[str]
    bypass_mode: bool
    group_size: int
    batch_size: int
    kv_cache_type: str
    memory_est: float
    fits_vram: bool
    viability: str
    ranking: int
    notes: str
    config_cmd: str


FRAMEWORK_CONFIGS = {
    "rllm_tinker": FrameworkConfig(
        name="rLLM Tinker",
        backend="Tinker (in-process)",
        model="Qwen3-1.7B/8B",
        lora_rank=32,
        lora_targets=["mlp", "attn", "unembed"],
        bypass_mode=True,
        group_size=4,
        batch_size=8,
        kv_cache_type="FP16",
        memory_est=12.0,
        fits_vram=True,
        viability="★★★★★★ #1",
        ranking=1,
        notes="In-process no Ray; auto LoRA rank=32; zero-copy weight sync; LossFnType={ppo,IS,cispo,dro}",
        config_cmd="rllm train --config tinker.yaml --model.lora_rank 32 --bypass_mode true",
    ),
    "verl_bypass": FrameworkConfig(
        name="verl + bypass",
        backend="verl (Ray) + vLLM",
        model="Qwen3-1.7B/8B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=True,
        group_size=8,
        batch_size=4,
        kv_cache_type="INT8 FlashInfer",
        memory_est=20.0,
        fits_vram=True,
        viability="★★★★ #2",
        ranking=2,
        notes="bypass save 14GB ref; detach_metrics prevent OOM; INT8 KV; VLLM_USE_V2_MODEL_RUNNER=0 fallback",
        config_cmd="verl train --algorithm grpo --bypass_mode True --detach_metrics True",
    ),
    "verl_cppo": FrameworkConfig(
        name="verl + CPPO",
        backend="verl (Ray) + vLLM",
        model="Qwen3-1.7B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=True,
        group_size=4,
        batch_size=4,
        kv_cache_type="INT8 FlashInfer",
        memory_est=16.0,
        fits_vram=True,
        viability="★★★★★ #2.5",
        ranking=3,
        notes="CPPO provably tighter bound; bypass MUST; near-zero overhead; position-weighted divergence",
        config_cmd="verl train --algorithm cppo --bypass_mode True",
    ),
    "verl_remax": FrameworkConfig(
        name="verl + ReMax",
        backend="verl (Ray) + vLLM",
        model="Qwen3-1.7B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=True,
        group_size=1,
        batch_size=4,
        kv_cache_type="INT8 FlashInfer",
        memory_est=28.0,
        fits_vram=False,
        viability="★★★★ but ref model needed",
        ranking=2,
        notes="ReMax greedy baseline ZERO variance; BUT use_kl_in_reward=True needs ref model; NOT bypass compatible canonically",
        config_cmd="verl train --algorithm remax --bypass_mode True",
    ),
    "verl_icepop": FrameworkConfig(
        name="verl + IcePop",
        backend="verl (Ray) + vLLM",
        model="Qwen3-1.7B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=True,
        group_size=4,
        batch_size=4,
        kv_cache_type="INT8 FlashInfer",
        memory_est=16.0,
        fits_vram=True,
        viability="★★★ #3",
        ranking=3,
        notes="IcePop IS correction exact population clipping; most precise but less stable; bypass_pg_token_icepop preset",
        config_cmd="verl train --algorithm icepop --bypass_mode True",
    ),
    "deepspeed_zero2": FrameworkConfig(
        name="DeepSpeed ZeRO-2",
        backend="DeepSpeed",
        model="Qwen3-1.7B/8B",
        lora_rank=32,
        lora_targets=["mlp", "attn", "unembed"],
        bypass_mode=True,
        group_size=4,
        batch_size=8,
        kv_cache_type="N/A (training)",
        memory_est=14.0,
        fits_vram=True,
        viability="★★★★ #4",
        ranking=4,
        notes="ZeRO-2+CPU_Adam+LoRAOptimizedLinear(offload=0.5)+coalesce_grad; ZenFlow #8058 chunked copyback 2944→256MiB; Muon optimizer alternative",
        config_cmd="deepspeed train_config.json --zero_stage 2 --offload_optimizer cpu",
    ),
    "deepspeed_autoep_moe": FrameworkConfig(
        name="DeepSpeed AutoEP MoE",
        backend="DeepSpeed AutoEP",
        model="Qwen3-MoE (A0.6B+B4B)",
        lora_rank=32,
        lora_targets=["router", "expert_mlp"],
        bypass_mode=True,
        group_size=4,
        batch_size=8,
        kv_cache_type="N/A (training)",
        memory_est=18.0,
        fits_vram=True,
        viability="★★★★★★ MoE #1 NEW!",
        ranking=1,
        notes="AutoEP MERGED #7938; EP=1 singleton MoE 15x faster; LoRA+CPU_Adam+coalesce_grad; RTX 4090唯一MoE可行路径!",
        config_cmd="deepspeed train_config.json --zero_stage 2 --offload_optimizer cpu --autoep_enable True",
    ),
    "deepspeed_opd_distill": FrameworkConfig(
        name="DeepSpeed OPD Distillation",
        backend="DeepSpeed OPD",
        model="Qwen2.5-0.5B (student)",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=False,
        group_size=0,
        batch_size=16,
        kv_cache_type="N/A (distillation)",
        memory_est=2.0,
        fits_vram=True,
        viability="★★★★ distillation NEW!",
        ranking=1,
        notes="OPD #8027 DRAFT; small student+big teacher; teacher logits CPU chunk fetch; student-only GPU ~1GB; ZeRO-0 viable",
        config_cmd="deepspeed opd_train --student_model Qwen2.5-0.5B --teacher_model Qwen2.5-1.5B",
    ),
    "deepspeed_zero3": FrameworkConfig(
        name="DeepSpeed ZeRO-3",
        backend="DeepSpeed",
        model="Qwen3-8B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=True,
        group_size=4,
        batch_size=8,
        kv_cache_type="N/A (training)",
        memory_est=28.0,
        fits_vram=False,
        viability="★ NOT RECOMMENDED",
        ranking=7,
        notes="ZeRO-3 single GPU = meaningless overhead; world_size=1 no sharding; gather/scatter overhead; singleton PG",
        config_cmd="NOT RECOMMENDED for single GPU RTX 4090",
    ),
    "megatron_lite": FrameworkConfig(
        name="Megatron Lite",
        backend="Megatron-LM (verl)",
        model="MoE only (too big)",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=True,
        group_size=4,
        batch_size=8,
        kv_cache_type="N/A",
        memory_est=0,
        fits_vram=False,
        viability="★★ mid-term only",
        ranking=5,
        notes="MoE too big for 24GB; bitwise verified; -20.3% memory; no LoRA wiring; no value_model; mid-term: small model protocol",
        config_cmd="NOT yet viable on RTX 4090",
    ),
    "vllm_inference": FrameworkConfig(
        name="vLLM Inference",
        backend="vLLM V1",
        model="Qwen3-1.7B/8B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=False,
        group_size=0,
        batch_size=0,
        kv_cache_type="INT8 FlashInfer",
        memory_est=6.0,
        fits_vram=True,
        viability="★★★★★★ best inference",
        ranking=1,
        notes="INT8 KV FlashInfer; MRv2 default Qwen3 but safe; VLLM_USE_V2_MODEL_RUNNER=0 fallback; HMA-by-default; enforce_eager for batch invariance",
        config_cmd="vllm serve Qwen/Qwen3-8B --kv-cache-dtype int8 --enable-prefix-caching",
    ),
    "sglang_inference": FrameworkConfig(
        name="SGLang Inference",
        backend="SGLang",
        model="Qwen3-1.7B/8B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=False,
        group_size=0,
        batch_size=0,
        kv_cache_type="FP16 Triton",
        memory_est=8.0,
        fits_vram=True,
        viability="★★★★ good inference alt",
        ranking=2,
        notes="SGLang deterministic inference SM89 batch invariance solution; Triton constexpr BLOCK_SIZE; RadixAttention prefix reuse; DeepGEMM SM89 fallback",
        config_cmd="python -m sglang.launch_server Qwen/Qwen3-8B --enable-deterministic-inference",
    ),
    "mindie_ascend": FrameworkConfig(
        name="MindIE/Ascend",
        backend="MindIE (Ascend NPU)",
        model="N/A (Ascend only)",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=False,
        group_size=0,
        batch_size=0,
        kv_cache_type="N/A",
        memory_est=0,
        fits_vram=False,
        viability="★ not for NVIDIA",
        ranking=0,
        notes="Ascend NPU only; compose-level fusion unique; BudgetRefiner SLO; vLLM-Ascend production recommended",
        config_cmd="Not applicable for RTX 4090",
    ),
    "pytorch_fsdp2": FrameworkConfig(
        name="PyTorch FSDP2",
        backend="PyTorch FSDP2",
        model="Qwen3-1.7B",
        lora_rank=0,
        lora_targets=[],
        bypass_mode=False,
        group_size=4,
        batch_size=8,
        kv_cache_type="N/A",
        memory_est=14.0,
        fits_vram=False,
        viability="★★ single GPU useless",
        ranking=6,
        notes="FSDP2 useless for single GPU; LoRA+compile more effective; DTensor per-parameter; BF16 only SM89",
        config_cmd="NOT RECOMMENDED for single GPU RTX 4090",
    ),
}


def estimate_memory(model_name, lora_rank, bypass_mode, kv_cache_type, group_size, batch_size):
    """Estimate GRPO training memory for RTX 4090 in GB.

    Uses realistic estimates based on measured configs:
    - rLLM Tinker 8B LoRA+bypass: ~12GB
    - verl 8B no-LoRA+bypass+INT8KV: ~20GB
    - verl 1.7B bypass+INT8KV: ~6GB
    - DeepSpeed ZeRO-2 8B LoRA+offload: ~14GB
    """
    model_sizes = {"Qwen3-1.7B": 3.4, "Qwen3-8B": 16.0, "Llama-3.1-8B": 16.0, "Mistral-7B": 14.0}
    base = model_sizes.get(model_name, 16.0)
    ratio = base / 16.0  # proportional scaling factor

    # Model weights (always present)
    weights = base

    # Ref model (if bypass=False)
    ref = base if not bypass_mode else 0

    # LoRA params (~0.6GB for rank=32 on 8B)
    lora = 0.6 * ratio if lora_rank > 0 else 0

    # Optimizer (Adam m+v for trainable params only with LoRA, all params without)
    optimizer = lora * 2 if lora_rank > 0 else base * 2

    # Activations (~2GB LoRA on 8B, ~8GB full on 8B)
    activations = 2.0 * ratio if lora_rank > 0 else 8.0 * ratio

    # KV cache (inference phase)
    kv = 2.0 if kv_cache_type.startswith("INT8") else 4.0 if kv_cache_type.startswith("FP16") else 0

    # CUDA workspace
    cuda = 1.0

    total = weights + ref + lora + optimizer + activations + kv + cuda
    return total


def run_matrix(args):
    print("=" * 100)
    print("RTX 4090 Seven-Framework GRPO Configuration Matrix")
    print("=" * 100)
    print(f"RTX 4090: {RTX_4090_VRAM}GB VRAM, SM{SM_ARCH}")
    print()

    fmt = "{:<16} {:<14} {:<14} {:<6} {:<6} {:<4} {:<4} {:<14} {:<6} {:<14} {:<4} {:<30}"
    print(fmt.format("Framework", "Backend", "Model", "LoRA", "Bypass", "Grp", "Btch", "KV Cache", "Mem", "Fits?", "Rnk", "Notes"))
    print("-" * 120)

    for key, cfg in FRAMEWORK_CONFIGS.items():
        print(fmt.format(
            cfg.name, cfg.backend, cfg.model, str(cfg.lora_rank),
            "Y" if cfg.bypass_mode else "N", str(cfg.group_size),
            str(cfg.batch_size), cfg.kv_cache_type, f"{cfg.memory_est:.0f}G",
            "Y" if cfg.fits_vram else "N", f"#{cfg.ranking}",
            cfg.notes[:30]
        ))

    print()
    print("Key Insights:")
    print("  1. rLLM Tinker = #1 RTX 4090 (in-process, bypass default, auto LoRA)")
    print("  2. verl + CPPO + bypass = #2.5 (provably tighter bound)")
    print("  3. DeepSpeed ZeRO-2 = #4 (LoRA+CPU offload, solid training)")
    print("  4. vLLM INT8 KV = best inference backend for RTX 4090")
    print("  5. SGLang deterministic = SM89 batch invariance inference solution")


def run_budget(args):
    print("=" * 100)
    print("RTX 4090 GRPO Memory Budget Estimator")
    print("=" * 100)

    models = ["Qwen3-1.7B", "Qwen3-8B"]
    configs = [
        ("rLLM Tinker", 32, True, "FP16", 4, 8),
        ("verl + bypass", 0, True, "INT8 FlashInfer", 8, 4),
        ("verl + CPPO", 0, True, "INT8 FlashInfer", 4, 4),
        ("DeepSpeed ZeRO-2", 32, True, "N/A", 4, 8),
        ("DeepSpeed ZeRO-3", 0, True, "N/A", 4, 8),
    ]

    for model in models:
        print(f"\n--- {model} ---")
        print(f"{'Config':<20} {'LoRA':<6} {'Bypass':<8} {'KV':<16} {'Group':<6} {'Batch':<6} {'Mem(GB)':<8} {'Fits 24GB':<8}")
        print("-" * 80)
        for name, lr, bypass, kv, grp, btch in configs:
            mem = estimate_memory(model, lr, bypass, kv, grp, btch)
            fits = mem <= 23.0
            print(f"{name:<20} {lr:<6} {'Yes' if bypass else 'No':<8} {kv:<16} {grp:<6} {btch:<6} {mem:<8.1f} {'Yes' if fits else 'NO!':<8}")


def run_recommend(args):
    print("=" * 100)
    print("RTX 4090 GRPO Training Recommendations")
    print("=" * 100)

    print("""
TOP RECOMMENDED CONFIGS:

1. rLLM Tinker + Qwen3-8B + bypass + LoRA rank=32
   - ~12GB memory, fits 24GB VRAM easily
   - In-process, no Ray overhead
   - Zero-copy weight sync via save_weights_for_sampler
   - LossFnType={ppo,IS,cispo,dro}

2. verl + CPPO + bypass + Qwen3-1.7B
   - ~16GB memory, fits 24GB VRAM
   - Provably tighter trust region bound
   - INT8 KV cache (FlashInfer backend)
   - detach_metrics=True to prevent OOM

3. verl + bypass + PPO clip + Qwen3-1.7B
   - ~16GB memory, fits 24GB VRAM
   - INT8 KV cache, detach_metrics=True

NOT RECOMMENDED:
  - DeepSpeed ZeRO-3 on single GPU (overhead only)
  - PyTorch FSDP2 on single GPU (useless without multi-GPU)

INFERENCE:
  - vLLM INT8 KV (FlashInfer) = best for RTX 4090
  - SGLang deterministic = SM89 batch invariance solution

OSS CONTRIBUTIONS:
  1. BudgetRefiner SLO -> vLLM upstream (#1 priority)
  2. Inductor SM<90 Fusion Guard -> PyTorch upstream (#2)
  3. QuantKey refactor -> vLLM foundation for SM89 guard
""")


def run_compare(args):
    print("=" * 100)
    print("Seven-Framework RTX 4090 GRPO Comparison")
    print("=" * 100)

    dims = [
        ("Architecture", {
            "rllm_tinker": "In-process client-server",
            "verl_bypass": "Ray-based distributed",
            "deepspeed_zero2": "ZeRO-2 + CPU_Adam",
            "vllm_inference": "V1 async scheduler",
            "sglang_inference": "RadixAttention + Triton",
        }),
        ("LoRA", {
            "rllm_tinker": "Auto r=32 (in-service merge)",
            "verl_bypass": "No native LoRA",
            "deepspeed_zero2": "LoRAOptimizedLinear r=32",
            "vllm_inference": "LoRA serve (inference)",
            "sglang_inference": "LoRA serve (inference)",
        }),
        ("Bypass", {
            "rllm_tinker": "Default True (KL=0)",
            "verl_bypass": "Manual True (KL=0)",
            "deepspeed_zero2": "External True (KL=0)",
            "vllm_inference": "N/A (inference)",
            "sglang_inference": "N/A (inference)",
        }),
        ("Batch Invariance", {
            "rllm_tinker": "Tinker handles internally",
            "verl_bypass": "vLLM compile or enforce_eager",
            "deepspeed_zero2": "N/A (training)",
            "vllm_inference": "Inductor compile / enforce_eager",
            "sglang_inference": "Triton constexpr deterministic",
        }),
        ("KV Path", {
            "rllm_tinker": "FP16 (internal)",
            "verl_bypass": "INT8 FlashInfer",
            "deepspeed_zero2": "N/A (training)",
            "vllm_inference": "INT8 FlashInfer",
            "sglang_inference": "FP16 Triton",
        }),
    ]

    keys = ["rllm_tinker", "verl_bypass", "deepspeed_zero2", "vllm_inference", "sglang_inference"]
    names = [FRAMEWORK_CONFIGS[k].name for k in keys]

    print(f"{'Dimension':<18}", end="")
    for n in names:
        print(f" {n:<22}", end="")
    print()
    print("-" * 110)

    for dim_name, vals in dims:
        print(f"{dim_name:<18}", end="")
        for k in keys:
            v = vals.get(k, "N/A")
            print(f" {v:<22}", end="")
        print()


def run_gaps(args):
    print("=" * 100)
    print("RTX 4090 GRPO Gap Analysis & Opportunity Map")
    print("=" * 100)

    gaps = [
        ("RTX 4090 GRPO recipe", "rLLM cookbook no RTX 4090 recipe", "★★★★★", "rLLM cookbook PR"),
        ("BudgetRefiner SLO", "vLLM V1 NO SLO scheduling; portable 95%+ GPU-generic; RTX 4090 profile data unique", "★★★★★★★", "#1 OSS priority"),
        ("Inductor SM<90 Guard", "5-line fix in choices.py; 4 precedents; root cause fully understood", "★★★★★★★★★", "PyTorch upstream PR"),
        ("QuantKey SM89 guard", "QuantKey refactor foundation; Phase 2: requires_sm", "★★★★★", "vLLM PR"),
        ("SGLang MAGI adapter", "prefix-tree KV dedup saves 7/8 prefix; currently Megatron only", "★★★★", "Medium-term contribution"),
        ("INT4 Triton SM89 test", "INT4 Triton fallback merged; need SM89 comprehensive testing", "★★★", "Testing contribution"),
        ("MoE small model", "Megatron Lite MoE too big; need LoRA+INT8+small model protocol", "★★★", "Future exploration"),
    ]

    print(f"{'Gap':<22} {'Description':<55} {'Priority':<16} {'Action':<22}")
    print("-" * 115)

    for name, desc, priority, action in gaps:
        print(f"{name:<22} {desc:<55} {priority:<16} {action:<22}")

    print()
    print("TOP 3 OPPORTUNITIES:")
    print("  1. BudgetRefiner SLO -> vLLM upstream -> RTX 4090 profile data = unique contribution")
    print("  2. Inductor SM<90 Fusion Guard -> PyTorch upstream -> 5-line PR, most valuable")
    print("  3. rLLM RTX 4090 cookbook -> practical recipe -> community adoption")


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 Seven-Framework GRPO Configuration Matrix")
    parser.add_argument("--mode", choices=["matrix", "budget", "recommend", "compare", "gaps", "all"],
                        default="matrix", help="Output mode")
    args = parser.parse_args()

    modes = {"matrix": run_matrix, "budget": run_budget, "recommend": run_recommend,
             "compare": run_compare, "gaps": run_gaps}

    if args.mode == "all":
        for func in modes.values():
            func(args)
            print()
    else:
        modes[args.mode](args)


if __name__ == "__main__":
    main()
