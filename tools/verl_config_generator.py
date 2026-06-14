#!/usr/bin/env python3
"""verl Training Configuration Generator

根据硬件约束自动生成最优verl训练配置YAML。
涵盖: RolloutMode(HYBRID/COLOCATED/STANDALONE) + Weight Sync后端 + FSDP策略 + LoRA + GRPO/PPO选择

Usage:
  python verl_config_generator.py --gpu-count 1 --gpu-memory 24 --interconnect none --model-size 7
  python verl_config_generator.py --gpu-count 8 --gpu-memory 80 --interconnect nvlink --model-size 7 --training-type ppo
  python verl_config_generator.py --gpu-count 8 --gpu-memory 24 --interconnect pcie --model-size 7 --lora-rank 16
"""

import argparse
import json
import sys


def estimate_model_memory(model_size_billion: int, precision: str = "bf16", lora_rank: int = 0) -> dict:
    """Estimate model memory requirements.

    model_size_billion: number of parameters in billions (7 = 7B model)
    For 7B BF16: 7×10^9 params × 2 bytes = 14×10^9 bytes ≈ 14GB
    """
    bytes_per_param = 2 if precision == "bf16" else 4  # fp32
    # Model weights: params × bytes_per_param / (1024^3) → GB
    model_weights_gb = model_size_billion * bytes_per_param  # ≈GB (since 10^9 bytes ≈ 1 GB)

    # LoRA adds minimal trainable params but model weights still need full memory
    lora_params_gb = 0
    if lora_rank > 0:
        # Approximate: ~0.8% trainable params for r=16
        lora_ratio = lora_rank / 512
        lora_params_gb = model_weights_gb * lora_ratio * 2  # LoRA params counted in fp32 for optimizer

    # Optimizer states (Adam = 2 × param_bytes)
    # Full training: optimizer states for all params
    # LoRA training: optimizer states only for LoRA params (fp32), but base weights still in bf16
    if lora_rank == 0:
        optimizer_gb = model_weights_gb * 4  # fp32 master weights + m + v = 4×
    else:
        optimizer_gb = lora_params_gb * 4  # fp32 LoRA master weights + m + v = 4×

    # Activations (rough: ~10% of model per batch for seq_len=2048, batch=4)
    # GRPO uses ~2× activations (rollout_n sequences)
    activation_per_batch_gb = model_weights_gb * 0.1
    activation_gb = activation_per_batch_gb * 4  # batch_size=4

    # PPO needs 2× model memory (actor + critic)
    # GRPO needs 1× model memory (only actor)

    # Total for single model with LoRA (what actually sits on GPU):
    # = base model weights(BF16) + LoRA weights(BF16) + optimizer(LoRA fp32) + activations
    total_with_lora_gb = model_weights_gb + lora_params_gb + optimizer_gb + activation_gb

    # Total for single model full training:
    # = model weights(BF16) + optimizer(fp32 master+m+v) + activations
    total_single_gpu_gb = model_weights_gb + optimizer_gb + activation_gb

    return {
        "model_weights_gb": model_weights_gb,
        "params_gb": model_weights_gb,
        "lora_params_gb": lora_params_gb,
        "optimizer_gb": optimizer_gb,
        "activation_gb": activation_gb,
        "total_single_gpu_gb": total_single_gpu_gb,
        "total_with_lora_gb": total_with_lora_gb,
        "ppo_peak_gb": total_single_gpu_gb * 2,  # actor + critic
        "ppo_with_lora_peak_gb": total_with_lora_gb * 2,
    }


def recommend_rollout_mode(gpu_count: int, interconnect: str, lora_rank: int) -> dict:
    """Recommend RolloutMode based on hardware."""
    if gpu_count == 1:
        return {
            "mode": "HYBRID",
            "reason": "单GPU→HYBRID同进程最优→naive weight sync零拷贝→无IPC/Ray overhead",
            "weight_sync_backend": "naive",
            "sleep_level": 1 if lora_rank > 0 else 2,
        }
    elif interconnect == "nvlink":
        if gpu_count <= 4:
            return {
                "mode": "HYBRID",
                "reason": "小规模NVLink→HYBRID仍最优→naive零拷贝→每GPU actor+rollout",
                "weight_sync_backend": "naive",
                "sleep_level": 1 if lora_rank > 0 else 2,
            }
        else:
            return {
                "mode": "COLOCATED",
                "reason": "大规模NVLink→COLOCATED→CUDA IPC weight sync→vLLM独立进程更稳定",
                "weight_sync_backend": "nccl",
                "sleep_level": 1,
            }
    elif interconnect == "pcie":
        return {
            "mode": "HYBRID",
            "reason": "PCIe瓶颈→HYBRID避免NCCL→naive零拷贝→单GPU最优; 多GPUPCIe无法有效sharding",
            "weight_sync_backend": "naive",
            "sleep_level": 1 if lora_rank > 0 else 2,
        }
    else:
        return {
            "mode": "STANDALONE",
            "reason": "RDMA集群→STANDALONE→NIXL RDMA weight sync→最快跨节点",
            "weight_sync_backend": "nixl",
            "sleep_level": None,  # STANDALONE不需要sleep/wake
        }


def recommend_algorithm(gpu_memory: int, model_mem: dict, lora_rank: int, training_type: str) -> dict:
    """Recommend GRPO vs PPO."""
    total_mem = model_mem["total_with_lora_gb"] if lora_rank > 0 else model_mem["total_single_gpu_gb"]

    if training_type == "ppo":
        # PPO needs 2× model memory (actor + critic)
        ppo_mem = total_mem * 2
        if ppo_mem > gpu_memory:
            return {
                "algorithm": "grpo",
                "reason": f"PPO需2×内存={ppo_mem:.1f}GB > GPU {gpu_memory}GB → 不可行! GRPO只需1×={total_mem:.1f}GB",
                "need_critic": False,
                "ppo_feasible": False,
            }
        else:
            return {
                "algorithm": "ppo",
                "reason": f"PPO内存={ppo_mem:.1f}GB ≤ GPU {gpu_memory}GB → 可行",
                "need_critic": True,
                "ppo_feasible": True,
            }
    elif training_type == "grpo":
        if total_mem > gpu_memory:
            return {
                "algorithm": "grpo_with_lora",
                "reason": f"GRPO内存={total_mem:.1f}GB > GPU {gpu_memory}GB → 需LoRA减内存",
                "need_critic": False,
                "ppo_feasible": False,
            }
        else:
            return {
                "algorithm": "grpo",
                "reason": f"GRPO内存={total_mem:.1f}GB ≤ GPU {gpu_memory}GB → 可行",
                "need_critic": False,
                "ppo_feasible": True,
            }
    else:  # auto
        ppo_mem = total_mem * 2
        if total_mem <= gpu_memory and ppo_mem <= gpu_memory:
            return {
                "algorithm": "ppo",
                "reason": f"PPO内存={ppo_mem:.1f}GB ≤ GPU {gpu_memory}GB → 可行, 选PPO(更精确)",
                "need_critic": True,
                "ppo_feasible": True,
            }
        elif total_mem <= gpu_memory:
            return {
                "algorithm": "grpo",
                "reason": f"PPO={ppo_mem:.1f}GB > GPU, GRPO={total_mem:.1f}GB ≤ GPU → GRPO唯一可行",
                "need_critic": False,
                "ppo_feasible": False,
            }
        else:
            return {
                "algorithm": "grpo_with_lora",
                "reason": f"GRPO={total_mem:.1f}GB > GPU → LoRA必需",
                "need_critic": False,
                "ppo_feasible": False,
            }


def recommend_fsdp_strategy(gpu_count: int, interconnect: str) -> dict:
    """Recommend FSDP sharding strategy."""
    if gpu_count == 1:
        return {
            "strategy": "no_sharding",
            "fsdp_size": -1,
            "reason": "单GPU → 无分布式 → FSDP退化为普通训练",
        }
    elif interconnect == "nvlink":
        if gpu_count <= 4:
            return {
                "strategy": "FULL_SHARD",
                "fsdp_size": -1,
                "reason": "NVLink小规模 → 全分片(FSDP2) → 2Ψ通信可承受 → compile兼容",
            }
        else:
            return {
                "strategy": "HYBRID_SHARD",
                "fsdp_size": 4,
                "reason": "NVLink大规模 → HSDP(4GPU FSDP + DP) → 减少通信量 → compile兼容",
            }
    elif interconnect == "pcie":
        return {
            "strategy": "no_sharding",
            "fsdp_size": -1,
            "reason": "PCIe → 任何sharding通信灾难 → 单GPU+LoRA唯一出路",
        }
    else:
        return {
            "strategy": "HYBRID_SHARD",
            "fsdp_size": 4,
            "reason": "RDMA集群 → HSDP → 减少跨节点通信 → NIXL加速",
        }


def generate_yaml_config(model_size: int, gpu_count: int, gpu_memory: int, interconnect: str,
                         lora_rank: int, training_type: str, rollout_mode: dict,
                         algorithm: dict, fsdp: dict) -> str:
    """Generate verl training config YAML."""
    algo = algorithm["algorithm"]
    need_critic = algorithm["need_critic"]
    lora_enabled = lora_rank > 0 or algo == "grpo_with_lora"
    effective_lora = lora_rank if lora_rank > 0 else 16 if algo == "grpo_with_lora" else 0

    # Rollout settings
    rollout_tp = min(gpu_count, 1)  # Default TP=1 for single GPU
    if interconnect == "nvlink" and gpu_count > 1:
        rollout_tp = min(gpu_count, 2)

    # GRPO rollout_n
    rollout_n = 8 if "grpo" in algo else 1

    # Sleep level
    sleep_level = rollout_mode.get("sleep_level", 1)
    sleep_config = ""
    if sleep_level is not None:
        sleep_config = f"""
  rollout:
    name: vllm
    tensor_parallel_size: {rollout_tp}
    sleep_level: {sleep_level}
    free_cache_engine: true"""

    # LoRA config
    lora_config = ""
    if lora_enabled:
        lora_config = f"""
  actor:
    strategy: fsdp2
    lora_rank: {effective_lora}
    lora_dropout: 0.0
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    train_unembed: false"""

    # Critic config
    critic_config = ""
    if need_critic:
        critic_config = """
  critic:
    strategy: fsdp2
    model:
      path: same_as_actor"""

    # Weight sync
    weight_backend = rollout_mode["weight_sync_backend"]

    yaml = f"""# verl Training Configuration (Auto-generated)
# Hardware: {gpu_count}×{gpu_memory}GB GPU ({interconnect}), Model: {model_size}B
# Algorithm: {algo}, RolloutMode: {rollout_mode['mode']}, WeightSync: {weight_backend}
# FSDP Strategy: {fsdp['strategy']}, LoRA: {lora_enabled}

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-{model_size}B
{sleep_config}
  ref:
    strategy: fsdp2
    cpu_offload: true  # Ref model → CPU offload to save GPU memory
{lora_config}

algorithm:
  adv_estimator: {algo.replace('_with_lora', '')}
  rollout_n: {rollout_n}  # GRPO: 8 responses per prompt
  kl_ctrl:
    type: kl
    kl_coef: {"0.001" if "grpo" in algo else "0.02"}
  grpo_n: {rollout_n}{"  # Only for GRPO" if "grpo" in algo else ""}

trainer:
  total_epochs: 1
  rollout_batch_size: {4 if gpu_count == 1 else 32 * gpu_count}
  mini_batch_size: {2 if gpu_count == 1 else 4}
  max_mini_batch_count: 2
  weight_sync_backend: {weight_backend}

checkpoint_engine:
  backend: {weight_backend}
  update_weights_bucket_megabytes: 2048

resource_pool:
  max_colocate_count: {3 if rollout_mode['mode'] == 'HYBRID' else 1}
{critic_config}
"""
    return yaml


def main():
    parser = argparse.ArgumentParser(description="verl Training Configuration Generator")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu-memory", type=int, default=24, help="GPU memory in GB")
    parser.add_argument("--interconnect", choices=["none", "pcie", "nvlink", "rdma"], default="none",
                        help="GPU interconnect type")
    parser.add_argument("--model-size", type=int, default=7, help="Model size in billions of params")
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0=full training)")
    parser.add_argument("--training-type", choices=["grpo", "ppo", "auto"], default="auto",
                        help="Training algorithm type")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of YAML")
    args = parser.parse_args()

    # Compute recommendations
    model_mem = estimate_model_memory(args.model_size, "bf16", args.lora_rank)
    rollout_mode = recommend_rollout_mode(args.gpu_count, args.interconnect, args.lora_rank)
    algorithm = recommend_algorithm(args.gpu_memory, model_mem, args.lora_rank, args.training_type)
    fsdp = recommend_fsdp_strategy(args.gpu_count, args.interconnect)

    # Check feasibility
    total_mem = model_mem["total_with_lora_gb"] if args.lora_rank > 0 or algorithm["algorithm"] == "grpo_with_lora" else model_mem["total_single_gpu_gb"]
    fits = total_mem <= args.gpu_memory

    if args.json:
        result = {
            "hardware": {"gpu_count": args.gpu_count, "gpu_memory_gb": args.gpu_memory,
                         "interconnect": args.interconnect},
            "model": {"size_billion": args.model_size, "memory_estimate": model_mem},
            "recommendations": {"rollout_mode": rollout_mode, "algorithm": algorithm, "fsdp_strategy": fsdp},
            "feasibility": {"fits_gpu_memory": fits, "total_memory_gb": total_mem,
                            "gpu_memory_gb": args.gpu_memory},
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 70)
        print(f"verl 配置推荐: {args.gpu_count}×{args.gpu_memory}GB GPU ({args.interconnect})")
        print(f"模型: {args.model_size}B参数, LoRA rank={args.lora_rank}")
        print("=" * 70)

        print("\n--- 内存估算 ---")
        print(f"  模型参数:     {model_mem['params_gb']:.2f} GB")
        if args.lora_rank > 0 or algorithm["algorithm"] == "grpo_with_lora":
            print(f"  LoRA参数:     {model_mem['lora_params_gb']:.2f} GB")
        print(f"  优化器状态:   {model_mem['optimizer_gb']:.2f} GB")
        print(f"  激活内存:     {model_mem['activation_gb']:.2f} GB")
        print(f"  总计:         {total_mem:.2f} GB → {'✓ FIT' if fits else '✗ EXCEEDS'} {args.gpu_memory}GB GPU")

        print("\n--- Rollout模式 ---")
        print(f"  推荐: {rollout_mode['mode']}")
        print(f"  原因: {rollout_mode['reason']}")
        print(f"  Weight Sync: {rollout_mode['weight_sync_backend']}")
        if rollout_mode.get("sleep_level"):
            print(f"  Sleep Level: {rollout_mode['sleep_level']} ({'LoRA只释放KV' if rollout_mode['sleep_level'] == 1 else '释放全部'}")

        print("\n--- 算法选择 ---")
        print(f"  推荐: {algorithm['algorithm']}")
        print(f"  原因: {algorithm['reason']}")
        print(f"  需要Critic: {algorithm['need_critic']}")
        print(f"  PPO可行: {algorithm['ppo_feasible']}")

        print("\n--- FSDP策略 ---")
        print(f"  推荐: {fsdp['strategy']}")
        print(f"  原因: {fsdp['reason']}")
        print(f"  FSDP size: {fsdp['fsdp_size']}")

        print("\n--- verl YAML配置 ---")
        yaml = generate_yaml_config(args.model_size, args.gpu_count, args.gpu_memory,
                                    args.interconnect, args.lora_rank, args.training_type,
                                    rollout_mode, algorithm, fsdp)
        print(yaml)

        # RTX 4090 special notes
        if args.gpu_memory == 24 and args.interconnect in ("none", "pcie"):
            print("\n--- RTX 4090 特殊提醒 ---")
            print("  ✓ HYBRID+naive+GRPO+LoRA → 最优配置(17GB peak)")
            print("  ✓ Sleep level=1 → 只释放KV cache → 快速wake_up")
            print("  ✓ LoRA adapter weight sync → ~2.6GB → 极快")
            print("  ✗ PPO → 需要2×模型内存 → 28GB > 24GB → 不可行!")
            print("  ✗ ZeRO-3/FSDP2多GPU → PCIe通信灾难 → 不可行!")
            print("  ✗ NCCL/NIXL weight sync → PCIe RDMA → 不可行!")


if __name__ == "__main__":
    main()
