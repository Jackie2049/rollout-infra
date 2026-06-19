#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Rollout-Infra Team
"""
verl V1 Sync PPOTrainer GRPO Training Step Simulator

Simulates the 10-phase GRPO training step lifecycle on CPU/MPS (no GPU needed).
Validates deep reading knowledge from the V1 sync trainer architecture.

Based on:
  - verl V1 sync PPOTrainer GRPO training loop deep reading
  - 10-phase training step lifecycle with TransferQueue-centric data flow
  - bypass_mode = skip entire old_log_prob forward pass (18Psi -> 3.8 Psi)
  - GRPO singleton degeneration (group_size=1 -> REINFORCE)
  - RTX 4090 24 GiB budget analysis

Modes:
  simulate   — Run a full GRPO training step simulation
  compare    — Compare bypass vs full, group sizes, loss types
  rtx4090    — RTX 4090-specific simulation with 24 GiB budget
  lifecycle  — Show the 10-phase lifecycle with memory/timing estimates

Usage:
  python verl_grpo_step_simulator.py simulate
  python verl_grpo_step_simulator.py simulate --model qwen3-4b --group_size 8 --bypass_mode True
  python verl_grpo_step_simulator.py compare
  python verl_grpo_step_simulator.py rtx4090
  python verl_grpo_step_simulator.py lifecycle
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# MODEL SIZE DATA (param counts + memory estimates)
# ==============================================================================

MODEL_PROFILES = {
    "qwen2.5-0.5b": {
        "name": "Qwen2.5-0.5B",
        "params": 0.5e9,
        "hidden_dim": 896,
        "num_layers": 24,
        "num_heads": 14,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "bf16_weights_gb": 1.0,
        "fp32_optimizer_gb": 4.0,
        "lora_params_m_rank16": 0.75,
        "lora_params_m_rank32": 1.5,   # rank=32 LoRA trainable params (M)
        "lora_params_m_rank64": 3.0,
    },
    "qwen2.5-1.5b": {
        "name": "Qwen2.5-1.5B",
        "params": 1.5e9,
        "hidden_dim": 1536,
        "num_layers": 28,
        "num_heads": 12,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "bf16_weights_gb": 3.0,
        "fp32_optimizer_gb": 12.0,
        "lora_params_m_rank16": 1.05,
        "lora_params_m_rank32": 2.1,
        "lora_params_m_rank64": 4.2,
    },
    "qwen2.5-3b": {
        "name": "Qwen2.5-3B",
        "params": 3e9,
        "hidden_dim": 2048,
        "num_layers": 36,
        "num_heads": 16,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "bf16_weights_gb": 6.0,
        "fp32_optimizer_gb": 24.0,
        "lora_params_m_rank16": 1.4,
        "lora_params_m_rank32": 2.8,
        "lora_params_m_rank64": 5.6,
    },
    "qwen2.5-7b": {
        "name": "Qwen2.5-7B",
        "params": 7e9,
        "hidden_dim": 3584,
        "num_layers": 28,
        "num_heads": 28,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "bf16_weights_gb": 14.0,
        "fp32_optimizer_gb": 56.0,
        "lora_params_m_rank16": 2.25,
        "lora_params_m_rank32": 4.5,
        "lora_params_m_rank64": 9.0,
    },
    "qwen3-4b": {
        "name": "Qwen3-4B (Dense)",
        "params": 4e9,
        "hidden_dim": 2560,
        "num_layers": 36,
        "num_heads": 20,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "bf16_weights_gb": 8.0,
        "fp32_optimizer_gb": 32.0,
        "lora_params_m_rank16": 1.6,
        "lora_params_m_rank32": 3.2,
        "lora_params_m_rank64": 6.4,
    },
    "qwen3-8b": {
        "name": "Qwen3-8B",
        "params": 8e9,
        "hidden_dim": 4096,
        "num_layers": 36,
        "num_heads": 32,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "bf16_weights_gb": 16.0,
        "fp32_optimizer_gb": 64.0,
        "lora_params_m_rank16": 2.5,
        "lora_params_m_rank32": 5.0,
        "lora_params_m_rank64": 10.0,
    },
}

# ==============================================================================
# 10-PHASE LIFECYCLE DATA
# ==============================================================================

PHASE_INFO = {
    0: {
        "name": "Weight Sync (Previous Step)",
        "method": "PPOTrainerSync.on_step_end() -> checkpoint_manager.update_weights()",
        "description": "Wake up rollout replicas, sync weights from actor engine. LoRA: two-phase (base + delta).",
        "gpu_active": True,
        "memory_type": "weights",
        "timing_s": 0.5,
        "notes": "naive backend = in-process, zero IPC overhead",
    },
    1: {
        "name": "Rollout Generation",
        "method": "_add_batch_to_generate -> agent_loop_manager.generate_sequences()",
        "description": "Sample prompts from dataloader, assign uid, fire-and-forget n trajectory generation per prompt.",
        "gpu_active": True,
        "memory_type": "kv_cache",
        "timing_s": 5.0,
        "notes": "Peak memory during rollout = weights + KV cache. Reward computed inside AgentLoop.",
    },
    2: {
        "name": "Replay Buffer Sampling",
        "method": "replay_buffer.sample(global_steps, partition, batch_size)",
        "description": "Poll TransferQueue until enough finished prompts. Select oldest first. Apply off-policy threshold.",
        "gpu_active": False,
        "memory_type": "cpu",
        "timing_s": 0.1,
        "notes": "CPU-only operation on TransferQueue metadata",
    },
    3: {
        "name": "Sleep Replicas",
        "method": "PPOTrainerSync.on_sample_end() -> checkpoint_manager.sleep_replicas()",
        "description": "Free rollout replica GPU memory (weights + KV cache). Essential for RTX 4090 memory budget.",
        "gpu_active": False,
        "memory_type": "freed",
        "timing_s": 0.2,
        "notes": "After sampling, rollout sleeps to free memory for training phase",
    },
    4: {
        "name": "Reward Computation (OPTIONAL)",
        "method": "_compute_reward_colocate()",
        "description": "NotImplementedError in sync trainer — rewards computed inside AgentLoop during Phase 1.",
        "gpu_active": False,
        "memory_type": "none",
        "timing_s": 0.0,
        "notes": "Skipped in practice; reward scores already in TransferQueue",
    },
    5: {
        "name": "Batch Balancing",
        "method": "_balance_batch(batch, metrics)",
        "description": "Upsample to divisible size, sequence-length balancing across DP ranks.",
        "gpu_active": False,
        "memory_type": "cpu",
        "timing_s": 0.05,
        "notes": "CPU reordering of batch metadata",
    },
    6: {
        "name": "Old Log Prob Computation",
        "method": "_compute_old_log_prob(batch, metrics)",
        "description": "bypass_mode=True: reuse rollout_log_probs (no forward). bypass_mode=False: full actor forward pass.",
        "gpu_active": "depends_on_bypass",
        "memory_type": "depends_on_bypass",
        "timing_s_bypass": 0.01,
        "timing_s_full": 3.0,
        "notes": "bypass_mode=True = skip entire forward pass -> 18Psi -> 3.8Psi",
    },
    7: {
        "name": "Ref Log Prob (OPTIONAL)",
        "method": "_compute_ref_log_prob(batch)",
        "description": "LoRA mode: ref from actor with LoRA disabled. Otherwise: separate ref_policy_wg.",
        "gpu_active": True,
        "memory_type": "activations",
        "timing_s": 2.0,
        "notes": "Optional, ref_in_actor=True reduces overhead for LoRA GRPO",
    },
    8: {
        "name": "Advantage Computation",
        "method": "_compute_advantage(batch, metrics)",
        "description": "GRPO: group by uid, compute outcome reward, normalize per group. Singleton = REINFORCE degeneration.",
        "gpu_active": False,
        "memory_type": "cpu",
        "timing_s": 0.02,
        "notes": "CPU-only. group_size=1 -> mean=0,std=1 -> REINFORCE degeneration (cross-framework bug)",
    },
    9: {
        "name": "Critic Update (OPTIONAL)",
        "method": "_update_critic(batch)",
        "description": "Not used in pure GRPO (outcome-only, no critic needed).",
        "gpu_active": False,
        "memory_type": "none",
        "timing_s": 0.0,
        "notes": "use_critic=False for GRPO",
    },
    10: {
        "name": "Actor Update",
        "method": "_update_actor(batch, metrics) -> actor.train_mini_batch(data)",
        "description": "Mini-batch training loop: forward -> PPO-clip loss -> backward -> optimizer step.",
        "gpu_active": True,
        "memory_type": "peak_training",
        "timing_s": 8.0,
        "notes": "FSDP: unshard -> forward -> reshard -> unshard -> backward -> reshard -> optimizer. Per-unit LoRA summon: 60GiB->6-8GiB.",
    },
}


# ==============================================================================
# SIMULATION DATA STRUCTURES
# ==============================================================================

@dataclass
class SimConfig:
    model: str = "qwen3-4b"
    group_size: int = 8
    bypass_mode: bool = True
    loss_type: str = "ppo_clip"       # "ppo_clip" or "reinforce"
    lora_rank: int = 32
    num_prompts: int = 4               # batch of prompts per step
    max_response_len: int = 512        # max tokens per response
    ppo_epochs: int = 1                # PPO update epochs
    mini_batch_size: int = 1           # mini-batches per epoch
    use_critic: bool = False
    use_reference_policy: bool = True   # ref_in_actor = LoRA-detached
    enforce_eager: bool = True
    gradient_clipping: float = 1.0
    sleep_level: int = 1
    checkpoint_backend: str = "naive"


@dataclass
class PhaseResult:
    phase: int
    name: str
    timing_s: float
    memory_peak_gb: float
    memory_type: str
    gpu_active: bool
    details: Dict = field(default_factory=dict)


@dataclass
class StepResult:
    config: SimConfig
    phases: List[PhaseResult] = field(default_factory=list)
    total_time_s: float = 0.0
    peak_memory_gb: float = 0.0
    advantage_quality: Dict = field(default_factory=dict)
    feasibility: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ==============================================================================
# CORE SIMULATION ENGINE
# ==============================================================================

class GRPOStepSimulator:
    """Simulates one complete GRPO training step (10 phases) on CPU/MPS."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.model_profile = MODEL_PROFILES.get(config.model, MODEL_PROFILES["qwen3-4b"])
        self.total_tokens = config.num_prompts * config.group_size * config.max_response_len
        self.total_trajectories = config.num_prompts * config.group_size

    def _estimate_weight_memory_gb(self) -> float:
        """Estimate model weight memory (bf16)."""
        return self.model_profile["bf16_weights_gb"]

    def _estimate_optimizer_memory_gb(self) -> float:
        """Estimate optimizer state memory (fp32 Adam: 2x weights for m+v states)."""
        # LoRA optimizer: only trainable params in fp32
        lora_m = self.model_profile[f"lora_params_m_rank{self.config.lora_rank}"]
        # fp32 optimizer states: param + m + v = 3 * lora_params * 4 bytes / 1e9
        return lora_m * 3 * 4 / 1e9  # ~small for LoRA

    def _estimate_kv_cache_memory_gb(self) -> float:
        """Estimate KV cache memory during rollout generation."""
        # KV cache: 2 * num_layers * hidden_dim * num_tokens * 2 bytes (bf16)
        hidden = self.model_profile["hidden_dim"]
        layers = self.model_profile["num_layers"]
        # n parallel generations per prompt, max_response_len tokens
        total_kv_tokens = self.config.group_size * self.config.max_response_len
        # per-token KV: 2 * hidden_dim * 2 bytes per layer, summed over layers
        kv_bytes = 2 * hidden * 2 * layers * total_kv_tokens
        return kv_bytes / (1024 ** 3)

    def _estimate_activation_memory_gb(self) -> float:
        """Estimate activation memory for training forward+backward."""
        hidden = self.model_profile["hidden_dim"]
        layers = self.model_profile["num_layers"]
        # Per token per layer: ~4 * hidden * 2 bytes (Q, K, V, O + residual)
        # Mini-batch size determines peak
        mb_tokens = self.config.num_prompts * self.config.max_response_len
        act_bytes = 4 * hidden * 2 * layers * mb_tokens
        return act_bytes / (1024 ** 3)

    def _estimate_training_peak_gb(self) -> float:
        """Estimate peak GPU memory during training (actor update)."""
        weights = self._estimate_weight_memory_gb()
        optimizer = self._estimate_optimizer_memory_gb()
        # Per-unit LoRA summon (#6512): only summon LoRA params, not full model
        # With FSDP, peak = LoRA unshard + gradients + optimizer + activations
        if self.config.bypass_mode:
            # bypass: skip old_log_prob forward, activations smaller
            activations = self._estimate_activation_memory_gb() * 0.3
        else:
            activations = self._estimate_activation_memory_gb() * 0.6
        # FSDP sharding on dp=1: full model resident + LoRA overhead
        # With per-unit LoRA summon: ~6-8 GiB peak for 7B model
        lora_overhead = 0.5  # LoRA adapter delta memory (GiB)
        return weights + optimizer + activations + lora_overhead

    def _estimate_rollout_peak_gb(self) -> float:
        """Estimate peak GPU memory during rollout generation."""
        weights = self._estimate_weight_memory_gb()
        kv_cache = self._estimate_kv_cache_memory_gb()
        return weights + kv_cache

    def _simulate_advantage(self) -> Dict:
        """Simulate GRPO advantage computation and quality metrics."""
        group_size = self.config.group_size
        num_groups = self.config.num_prompts

        # Generate simulated rewards per trajectory
        # Each group has group_size trajectories from the same prompt
        import random
        random.seed(42)

        all_rewards = []
        all_advantages = []
        group_stats = []

        for g in range(num_groups):
            # Simulate outcome rewards with controlled variance
            base_reward = random.gauss(0.5, 0.3)
            group_rewards = [base_reward + random.gauss(0, 0.2) for _ in range(group_size)]
            all_rewards.extend(group_rewards)

            # Compute GRPO advantage
            if group_size == 1:
                # Singleton: mean=0, std=1 -> REINFORCE degeneration
                mean = 0.0
                std = 1.0
                advantages = [(r - mean) / (std + 1e-6) for r in group_rewards]
                degeneration = True
            else:
                mean = sum(group_rewards) / group_size
                variance = sum((r - mean) ** 2 for r in group_rewards) / (group_size - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
                advantages = [(r - mean) / (std + 1e-6) for r in group_rewards]
                degeneration = False

            all_advantages.extend(advantages)
            group_stats.append({
                "group_id": g,
                "mean_reward": round(mean, 4),
                "std_reward": round(std, 4),
                "advantages": [round(a, 4) for a in advantages],
                "degeneration": degeneration,
            })

        # Compute quality metrics
        adv_mean = sum(all_advantages) / len(all_advantages) if all_advantages else 0
        adv_variance = sum((a - adv_mean) ** 2 for a in all_advantages) / len(all_advantages) if all_advantages else 0
        adv_std = math.sqrt(adv_variance) if adv_variance > 0 else 0

        # Signal-to-noise: |mean| / std (higher = better learning signal)
        snr = abs(adv_mean) / (adv_std + 1e-6) if adv_std > 0 else 0

        # Effective group size: variance reduction factor
        # GRPO with group_size n reduces variance by factor n compared to REINFORCE
        effective_variance_reduction = group_size if group_size > 1 else 1.0

        # Advantages always sum to 0 within each group (for group_size > 1)
        sum_within_groups = []
        for gs in group_stats:
            if gs["degeneration"]:
                sum_within_groups.append(sum(gs["advantages"]))  # NOT 0 for singleton
            else:
                sum_within_groups.append(sum(gs["advantages"]))  # ~0 (normalized)

        return {
            "group_size": group_size,
            "num_groups": num_groups,
            "total_trajectories": self.total_trajectories,
            "degeneration_detected": any(gs["degeneration"] for gs in group_stats),
            "degeneration_fraction": sum(1 for gs in group_stats if gs["degeneration"]) / num_groups,
            "advantage_mean": round(adv_mean, 4),
            "advantage_variance": round(adv_variance, 4),
            "advantage_std": round(adv_std, 4),
            "signal_to_noise": round(snr, 4),
            "effective_variance_reduction": effective_variance_reduction,
            "advantage_sum_per_group": [round(s, 4) for s in sum_within_groups],
            "group_stats_sample": group_stats[:2],  # Show first 2 groups
            "warning": "SINGLETON DEGENERATION: group_size=1 -> REINFORCE (mean=0, std=1, no variance reduction)"
                       if group_size == 1 else None,
        }

    def _simulate_phase(self, phase_num: int) -> PhaseResult:
        """Simulate a single phase with timing and memory estimates."""
        info = PHASE_INFO[phase_num]
        model = self.model_profile
        cfg = self.config

        timing = 0.0
        memory = 0.0
        gpu_active = False
        details = {}

        if phase_num == 0:
            # Weight sync from previous step
            timing = 0.3 + 0.1 * (model["bf16_weights_gb"] / 8)  # Scales with model size
            if cfg.sleep_level == 1:
                # LoRA adapter delta only: ~80x payload reduction
                memory = model["bf16_weights_gb"] + 0.5  # base resident + LoRA delta
            else:
                # sleep_level=2: full re-transfer
                memory = model["bf16_weights_gb"] * 2
            gpu_active = True
            details = {
                "backend": cfg.checkpoint_backend,
                "sleep_level": cfg.sleep_level,
                "payload": "LoRA delta (~0.5 GiB)" if cfg.sleep_level == 1 else "full weights (~{:.1f} GiB)".format(model["bf16_weights_gb"]),
                "two_phase_lora": True,
            }

        elif phase_num == 1:
            # Rollout generation
            timing = 2.0 + 3.0 * (cfg.max_response_len / 512) * (cfg.group_size / 8) * (model["params"] / 4e9)
            if cfg.enforce_eager:
                timing *= 1.5  # ~2x slower inference without cudagraph
            memory = self._estimate_rollout_peak_gb()
            gpu_active = True
            details = {
                "num_prompts": cfg.num_prompts,
                "group_size": cfg.group_size,
                "total_trajectories": self.total_trajectories,
                "max_response_len": cfg.max_response_len,
                "enforce_eager": cfg.enforce_eager,
                "peak_note": "Rollout peak = weights + KV cache",
            }

        elif phase_num == 2:
            # Replay buffer sampling (CPU)
            timing = 0.05 + 0.01 * (cfg.num_prompts * cfg.group_size / 32)
            memory = 0.0  # CPU-only
            gpu_active = False
            details = {
                "batch_size": cfg.num_prompts * cfg.group_size,
                "off_policy_threshold": "drop (recommended)",
            }

        elif phase_num == 3:
            # Sleep replicas
            timing = 0.1
            memory = 0.0  # Memory FREED
            gpu_active = False
            details = {
                "freed_memory_gb": round(self._estimate_rollout_peak_gb(), 2),
                "sleep_level": cfg.sleep_level,
                "tags": ["kv_cache"] if cfg.sleep_level == 1 else ["kv_cache", "weights"],
            }

        elif phase_num == 4:
            # Reward computation (skipped)
            timing = 0.0
            memory = 0.0
            gpu_active = False
            details = {"skipped": True, "reason": "Reward computed inside AgentLoop during Phase 1"}

        elif phase_num == 5:
            # Batch balancing (CPU)
            timing = 0.03
            memory = 0.0  # CPU reordering
            gpu_active = False
            details = {
                "upsample_to_divisible": True,
                "seqlen_balance": True,
                "dp_size": 1,
            }

        elif phase_num == 6:
            # Old log prob computation
            if cfg.bypass_mode:
                timing = 0.01  # TransferQueue rename operation
                memory = 0.0   # CPU-only TransferQueue ops
                gpu_active = False
                details = {
                    "mode": "bypass",
                    "operation": "tq.kv_batch_get -> rename rollout_log_probs -> tq.kv_batch_put",
                    "forward_pass_skipped": True,
                    "memory_saved": "~15 Psi equivalent",
                }
            else:
                timing = 2.0 + 1.0 * (model["params"] / 4e9) * (cfg.max_response_len / 512)
                memory = model["bf16_weights_gb"] * 0.6  # Partial forward activations
                gpu_active = True
                details = {
                    "mode": "full_recomputation",
                    "operation": "actor.infer_batch -> engine forward -> log_probs extraction",
                    "forward_pass_needed": True,
                    "memory_overhead": "Full actor forward pass + activations",
                }

        elif phase_num == 7:
            # Ref log prob (optional)
            if cfg.use_reference_policy and cfg.lora_rank > 0:
                # ref_in_actor = True: LoRA-detached ref
                timing = 1.0 + 0.5 * (model["params"] / 4e9)
                memory = model["bf16_weights_gb"] * 0.3  # Reduced with LoRA-detached
                gpu_active = True
                details = {
                    "ref_in_actor": True,
                    "mode": "LoRA-detached (no_lora=True)",
                    "no_separate_worker": True,
                }
            else:
                timing = 0.0
                memory = 0.0
                gpu_active = False
                details = {"skipped": True}

        elif phase_num == 8:
            # Advantage computation (CPU)
            timing = 0.01 + 0.005 * (cfg.num_prompts * cfg.group_size / 32)
            memory = 0.0  # CPU-only
            gpu_active = False
            adv_quality = self._simulate_advantage()
            details = {
                "estimator": "grpo",
                "group_size": cfg.group_size,
                "degeneration": adv_quality["degeneration_detected"],
                "advantage_mean": adv_quality["advantage_mean"],
                "advantage_std": adv_quality["advantage_std"],
                "signal_to_noise": adv_quality["signal_to_noise"],
                "bypass_impact": "No IS weight computation (pi_theta/pi_rollout, no gap to correct)" if cfg.bypass_mode else "IS weight computation needed (decoupled mode)",
            }

        elif phase_num == 9:
            # Critic update (skipped for GRPO)
            timing = 0.0
            memory = 0.0
            gpu_active = False
            details = {"skipped": True, "reason": "GRPO is outcome-only, use_critic=False"}

        elif phase_num == 10:
            # Actor update
            timing = 3.0 + 5.0 * (model["params"] / 4e9) * (cfg.ppo_epochs / 1) * (cfg.mini_batch_size / 1)
            memory = self._estimate_training_peak_gb()
            gpu_active = True
            details = {
                "ppo_epochs": cfg.ppo_epochs,
                "mini_batch_size": cfg.mini_batch_size,
                "loss_type": cfg.loss_type,
                "clip_ratio": 0.2,
                "gradient_clipping": cfg.gradient_clipping,
                "per_unit_lora_summon": True,
                "peak_note": "Training peak = FSDP unshard + LoRA summon + gradients + optimizer",
            }

        return PhaseResult(
            phase=phase_num,
            name=info["name"],
            timing_s=round(timing, 3),
            memory_peak_gb=round(memory, 2),
            memory_type=info["memory_type"],
            gpu_active=gpu_active,
            details=details,
        )

    def simulate_step(self) -> StepResult:
        """Run a full 10-phase GRPO training step simulation."""
        result = StepResult(config=self.config)
        warnings = []

        # Simulate all 10 phases (0-10)
        for phase_num in range(11):
            phase_result = self._simulate_phase(phase_num)
            result.phases.append(phase_result)

        # Compute totals
        result.total_time_s = round(sum(p.timing_s for p in result.phases), 3)

        # Peak memory = max of rollout peak and training peak
        rollout_peak = self._estimate_rollout_peak_gb()
        training_peak = self._estimate_training_peak_gb()
        # In practice, sleep/wake means peaks don't overlap
        # True peak = max(rollout_peak, training_peak)
        result.peak_memory_gb = round(max(rollout_peak, training_peak), 2)

        # Advantage quality
        result.advantage_quality = self._simulate_advantage()

        # Generate warnings
        if self.config.group_size == 1:
            warnings.append("CRITICAL: group_size=1 -> REINFORCE degeneration (mean=0, std=1, no variance reduction)")
        if self.config.lora_rank == 64:
            warnings.append("CRITICAL: LoRA rank=64 breaks EOS in vLLM rollout (#6782), MUST use rank=32")
        if not self.config.bypass_mode:
            warnings.append("WARNING: bypass_mode=False wastes ~15 Psi on old_log_prob forward pass")
        if not self.config.enforce_eager:
            warnings.append("WARNING: enforce_eager=False -> cudagraph fails on SM89 with DSV4")
        if self.config.gradient_clipping == 0:
            warnings.append("CRITICAL: gradient_clipping=0 -> ALWAYS set 1.0 (#8068)")
        if self.config.sleep_level == 2:
            warnings.append("WARNING: sleep_level=2 -> full weight re-transfer, use sleep_level=1 (LoRA adapter)")
        if self.config.loss_type == "reinforce":
            warnings.append("WARNING: REINFORCE loss_type is inferior to PPO-clip for variance reduction")

        # RTX 4090 feasibility check (24 GiB budget)
        budget = 24.0
        fits = result.peak_memory_gb <= budget
        margin = budget - result.peak_memory_gb

        result.feasibility = {
            "gpu_budget_gb": budget,
            "peak_memory_gb": result.peak_memory_gb,
            "fits_rtx4090": fits,
            "margin_gb": round(margin, 2),
            "rollout_peak_gb": round(rollout_peak, 2),
            "training_peak_gb": round(training_peak, 2),
            "sleep_wake_nonoverlap": True,  # Peaks don't overlap due to sleep/wake
            "status": "VIABLE" if fits and margin > 1.0 else "MARGINAL" if fits else "NOT VIABLE",
        }

        if not fits:
            warnings.append(f"NOT VIABLE on RTX 4090: peak {result.peak_memory_gb:.1f} GiB > 24 GiB budget")
        elif margin < 2.0:
            warnings.append(f"MARGINAL on RTX 4090: only {margin:.1f} GiB margin")

        result.warnings = warnings
        return result


# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def _phase_bar(timing_s: float, max_timing: float, width: int = 30) -> str:
    """Generate a simple text bar for timing visualization."""
    if max_timing == 0:
        return ""
    filled = int(width * timing_s / max_timing)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def display_simulate(result: StepResult) -> str:
    """Format a simulate result for display."""
    lines = []
    cfg = result.config
    mp = MODEL_PROFILES[cfg.model]

    lines.append("=" * 80)
    lines.append("verl V1 Sync PPOTrainer GRPO Training Step Simulation")
    lines.append("=" * 80)
    lines.append(f"  Model: {mp['name']} ({cfg.model})")
    lines.append(f"  Params: {mp['params']/1e9:.1f}B | Hidden: {mp['hidden_dim']} | Layers: {mp['num_layers']}")
    lines.append(f"  Config: group_size={cfg.group_size} | bypass_mode={cfg.bypass_mode} | loss_type={cfg.loss_type}")
    lines.append(f"  LoRA: rank={cfg.lora_rank} | enforce_eager={cfg.enforce_eager} | sleep_level={cfg.sleep_level}")
    lines.append(f"  Batch: {cfg.num_prompts} prompts x {cfg.group_size} responses = {cfg.num_prompts*cfg.group_size} trajectories")
    lines.append(f"  Response length: {cfg.max_response_len} tokens | PPO epochs: {cfg.ppo_epochs}")
    lines.append("")

    # Phase-by-phase results
    lines.append("-" * 80)
    lines.append("  Phase-by-Phase Simulation")
    lines.append("-" * 80)

    max_timing = max(p.timing_s for p in result.phases) if result.phases else 1
    total_active_time = sum(p.timing_s for p in result.phases if p.gpu_active)
    total_cpu_time = sum(p.timing_s for p in result.phases if not p.gpu_active)

    for p in result.phases:
        gpu_tag = "GPU" if p.gpu_active else "CPU"
        mem_tag = f"{p.memory_peak_gb:.2f} GiB" if p.memory_peak_gb > 0 else "---"
        bar = _phase_bar(p.timing_s, max_timing)
        lines.append(f"  Phase {p.phase:2d}: {p.name}")
        lines.append(f"          Time: {p.timing_s:.3f}s {bar} | Memory: {mem_tag} | {gpu_tag}")

        # Show key details
        for k, v in p.details.items():
            if k not in ("skipped",) and v is not None:
                val_str = str(v)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                lines.append(f"          {k}: {val_str}")
        lines.append("")

    # Summary
    lines.append("=" * 80)
    lines.append("  Step Summary")
    lines.append("=" * 80)
    lines.append(f"  Total step time:   {result.total_time_s:.3f} s")
    lines.append(f"  GPU active time:   {total_active_time:.3f} s")
    lines.append(f"  CPU-only time:     {total_cpu_time:.3f} s")
    lines.append(f"  Peak memory:       {result.peak_memory_gb:.2f} GiB")
    lines.append(f"  Rollout peak:      {result.feasibility['rollout_peak_gb']:.2f} GiB (weights + KV cache)")
    lines.append(f"  Training peak:     {result.feasibility['training_peak_gb']:.2f} GiB (FSDP + LoRA + optimizer)")
    lines.append(f"  Sleep/wake:        Peaks DO NOT overlap (memory freed between phases)")
    lines.append("")

    # Advantage quality
    aq = result.advantage_quality
    lines.append("-" * 80)
    lines.append("  Advantage Quality Metrics")
    lines.append("-" * 80)
    lines.append(f"  Group size:           {aq['group_size']}")
    lines.append(f"  Num groups:           {aq['num_groups']}")
    lines.append(f"  Total trajectories:   {aq['total_trajectories']}")
    lines.append(f"  Degeneration:         {aq['degeneration_detected']}")
    lines.append(f"  Advantage mean:       {aq['advantage_mean']}")
    lines.append(f"  Advantage variance:   {aq['advantage_variance']}")
    lines.append(f"  Advantage std:        {aq['advantage_std']}")
    lines.append(f"  Signal-to-noise:      {aq['signal_to_noise']}")
    lines.append(f"  Var. reduction:       {aq['effective_variance_reduction']}x")
    if aq["warning"]:
        lines.append(f"  *** {aq['warning']}")

    # Sample group stats
    for gs in aq["group_stats_sample"]:
        lines.append(f"  Group {gs['group_id']}: mean_reward={gs['mean_reward']}, std_reward={gs['std_reward']}, "
                      f"advantages={gs['advantages'][:4]}, degeneration={gs['degeneration']}")

    # RTX 4090 feasibility
    lines.append("")
    lines.append("-" * 80)
    lines.append("  RTX 4090 Feasibility")
    lines.append("-" * 80)
    f = result.feasibility
    status_emoji = "[OK]" if f["status"] == "VIABLE" else "[!!]" if f["status"] == "MARGINAL" else "[XX]"
    lines.append(f"  Status:     {status_emoji} {f['status']}")
    lines.append(f"  Budget:     {f['gpu_budget_gb']} GiB")
    lines.append(f"  Peak:       {f['peak_memory_gb']:.2f} GiB")
    lines.append(f"  Margin:     {f['margin_gb']:.2f} GiB")
    lines.append(f"  Fits:       {f['fits_rtx4090']}")

    # Warnings
    if result.warnings:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Warnings")
        lines.append("-" * 80)
        for w in result.warnings:
            lines.append(f"  * {w}")

    return "\n".join(lines)


def display_compare() -> str:
    """Compare bypass vs full, group sizes, loss types."""
    lines = []
    lines.append("=" * 80)
    lines.append("verl V1 GRPO Step Comparison")
    lines.append("=" * 80)

    # 1. Bypass mode comparison
    lines.append("")
    lines.append("-" * 80)
    lines.append("  1. bypass_mode Comparison (model=qwen3-4b, group_size=8, rank=32)")
    lines.append("-" * 80)

    configs_bypass = [
        SimConfig(bypass_mode=True),
        SimConfig(bypass_mode=False),
    ]

    lines.append(f"  {'Mode':<20} {'Time(s)':<10} {'Peak(GiB)':<12} {'Margin(GiB)':<12} {'Status':<10}")
    lines.append(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for cfg in configs_bypass:
        sim = GRPOStepSimulator(cfg)
        r = sim.simulate_step()
        mode_str = f"bypass={cfg.bypass_mode}"
        lines.append(f"  {mode_str:<20} {r.total_time_s:<10.3f} {r.peak_memory_gb:<12.2f} "
                      f"{r.feasibility['margin_gb']:<12.2f} {r.feasibility['status']:<10}")

    # 2. Group size comparison
    lines.append("")
    lines.append("-" * 80)
    lines.append("  2. group_size Comparison (model=qwen3-4b, bypass=True, rank=32)")
    lines.append("-" * 80)

    configs_group = [
        SimConfig(group_size=1),
        SimConfig(group_size=2),
        SimConfig(group_size=4),
        SimConfig(group_size=8),
        SimConfig(group_size=16),
    ]

    lines.append(f"  {'Group Size':<12} {'Degeneration':<15} {'Var.Red.':<10} {'SNR':<10} {'Adv.Std':<10} {'Margin':<10}")
    lines.append(f"  {'-'*12} {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for cfg in configs_group:
        sim = GRPOStepSimulator(cfg)
        r = sim.simulate_step()
        aq = r.advantage_quality
        deg_str = "YES" if aq["degeneration_detected"] else "no"
        lines.append(f"  {cfg.group_size:<12} {deg_str:<15} {aq['effective_variance_reduction']:<10} "
                      f"{aq['signal_to_noise']:<10.4f} {aq['advantage_std']:<10.4f} "
                      f"{r.feasibility['margin_gb']:<10.2f}")

    lines.append("")
    lines.append("  Key insight: group_size=1 -> REINFORCE degeneration (mean=0, std=1)")
    lines.append("  group_size >= 4 provides adequate variance reduction")
    lines.append("  group_size >= 8 is RECOMMENDED for stable GRPO training")

    # 3. Loss type comparison
    lines.append("")
    lines.append("-" * 80)
    lines.append("  3. loss_type Comparison (model=qwen3-4b, bypass=True, group_size=8)")
    lines.append("-" * 80)

    configs_loss = [
        SimConfig(loss_type="ppo_clip"),
        SimConfig(loss_type="reinforce"),
    ]

    lines.append(f"  {'Loss Type':<15} {'Clip Ratio':<12} {'Var.Red.':<10} {'Note':<40}")
    lines.append(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*40}")

    for cfg in configs_loss:
        if cfg.loss_type == "ppo_clip":
            note = "Clipped surrogate objective, bounded policy updates"
            clip = "0.2"
            vr = "clip+group"
        else:
            note = "Unbounded policy gradient, can overshoot"
            clip = "none"
            vr = "group only"
        lines.append(f"  {cfg.loss_type:<15} {clip:<12} {vr:<10} {note:<40}")

    # 4. LoRA rank comparison
    lines.append("")
    lines.append("-" * 80)
    lines.append("  4. LoRA rank Comparison (model=qwen3-4b, bypass=True, group_size=8)")
    lines.append("-" * 80)

    configs_lora = [
        SimConfig(lora_rank=16),
        SimConfig(lora_rank=32),
        SimConfig(lora_rank=64),
    ]

    lines.append(f"  {'LoRA Rank':<12} {'Params(M)':<10} {'Opt.GiB':<10} {'Peak(GiB)':<10} {'Margin':<10} {'Note':<30}")
    lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*30}")

    for cfg in configs_lora:
        sim = GRPOStepSimulator(cfg)
        r = sim.simulate_step()
        mp = MODEL_PROFILES[cfg.model]
        lora_m = mp[f"lora_params_m_rank{cfg.lora_rank}"]
        opt_gb = sim._estimate_optimizer_memory_gb()
        note = "#6782: rank=64 breaks EOS" if cfg.lora_rank == 64 else "recommended" if cfg.lora_rank == 32 else "minimal"
        lines.append(f"  {cfg.lora_rank:<12} {lora_m:<10.1f} {opt_gb:<10.3f} {r.peak_memory_gb:<10.2f} "
                      f"{r.feasibility['margin_gb']:<10.2f} {note:<30}")

    lines.append("")
    lines.append("  RECOMMENDED: LoRA rank=32 (safe for vLLM rollout, adequate expressiveness)")
    lines.append("  MUST NOT: rank=64 (breaks EOS token in vLLM rollout, #6782)")

    # 5. Model size comparison
    lines.append("")
    lines.append("-" * 80)
    lines.append("  5. Model Size Comparison (bypass=True, group_size=8, rank=32)")
    lines.append("-" * 80)

    configs_model = [
        SimConfig(model="qwen2.5-0.5b"),
        SimConfig(model="qwen2.5-1.5b"),
        SimConfig(model="qwen2.5-3b"),
        SimConfig(model="qwen3-4b"),
        SimConfig(model="qwen2.5-7b"),
        SimConfig(model="qwen3-8b"),
    ]

    lines.append(f"  {'Model':<15} {'Params':<8} {'Time(s)':<10} {'Peak(GiB)':<12} {'Margin':<10} {'Status':<10}")
    lines.append(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

    for cfg in configs_model:
        sim = GRPOStepSimulator(cfg)
        r = sim.simulate_step()
        mp = MODEL_PROFILES[cfg.model]
        lines.append(f"  {cfg.model:<15} {mp['params']/1e9:<8.1f} {r.total_time_s:<10.3f} "
                      f"{r.peak_memory_gb:<12.2f} {r.feasibility['margin_gb']:<10.2f} {r.feasibility['status']:<10}")

    return "\n".join(lines)


def display_rtx4090() -> str:
    """RTX 4090-specific simulation with 24 GiB budget analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("RTX 4090 GRPO Training — V1 Sync PPOTrainer Feasibility Analysis")
    lines.append("=" * 80)
    lines.append("  GPU: NVIDIA RTX 4090 | VRAM: 24 GiB | SM: 89 (Ada Lovelace)")
    lines.append("  Trainer: verl V1 PPOTrainerSync (colocated, synchronous)")
    lines.append("  Backend: FSDP + naive checkpoint + vLLM (in-process)")
    lines.append("")

    # Optimal config
    optimal = SimConfig(
        model="qwen3-4b",
        group_size=8,
        bypass_mode=True,
        loss_type="ppo_clip",
        lora_rank=32,
        num_prompts=4,
        max_response_len=512,
        ppo_epochs=1,
        mini_batch_size=1,
        enforce_eager=True,
        gradient_clipping=1.0,
        sleep_level=1,
    )

    sim = GRPOStepSimulator(optimal)
    r = sim.simulate_step()

    # Memory breakdown
    lines.append("-" * 80)
    lines.append("  Memory Budget Breakdown (24 GiB)")
    lines.append("-" * 80)

    weights = sim._estimate_weight_memory_gb()
    kv_cache = sim._estimate_kv_cache_memory_gb()
    optimizer = sim._estimate_optimizer_memory_gb()
    rollout_peak = r.feasibility["rollout_peak_gb"]
    training_peak = r.feasibility["training_peak_gb"]

    lines.append(f"  Model weights (bf16):        {weights:.2f} GiB")
    lines.append(f"  KV cache (rollout peak):      {kv_cache:.2f} GiB")
    lines.append(f"  Optimizer (LoRA fp32):        {optimizer:.3f} GiB")
    lines.append(f"  LoRA adapter delta:           ~0.50 GiB")
    lines.append(f"  Activations (training):       ~2.00 GiB")
    lines.append(f"  ---")
    lines.append(f"  Rollout peak:                 {rollout_peak:.2f} GiB (weights + KV cache)")
    lines.append(f"  Training peak:                {training_peak:.2f} GiB (FSDP + LoRA + optimizer)")
    lines.append(f"  Effective peak:               {r.peak_memory_gb:.2f} GiB (max of rollout/training)")
    lines.append(f"  Margin:                       {r.feasibility['margin_gb']:.2f} GiB")
    lines.append(f"  Status:                       {r.feasibility['status']}")
    lines.append("")
    lines.append("  NOTE: Sleep/wake ensures rollout and training peaks DO NOT overlap")
    lines.append("  Phase 3: sleep_replicas() frees KV cache + rollout weights before training")
    lines.append("  Effective peak = max(rollout_peak, training_peak), NOT sum")

    # Optimal config details
    lines.append("")
    lines.append("-" * 80)
    lines.append("  Optimal RTX 4090 GRPO Config")
    lines.append("-" * 80)
    lines.append(f"  trainer_mode:           sync (MUST, #1 for RTX 4090)")
    lines.append(f"  group_size (rollout.n): {optimal.group_size} (MUST >= 8)")
    lines.append(f"  bypass_mode:            True (MUST, 18Psi -> 3.8Psi)")
    lines.append(f"  loss_type:              ppo_clip (RECOMMENDED)")
    lines.append(f"  lora_rank:              {optimal.lora_rank} (MUST NOT 64, #6782)")
    lines.append(f"  enforce_eager:          True (MUST, SM89 cudagraph incompatible)")
    lines.append(f"  gradient_clipping:      {optimal.gradient_clipping} (MUST NOT 0, #8068)")
    lines.append(f"  sleep_level:            {optimal.sleep_level} (MUST NOT 2)")
    lines.append(f"  checkpoint_backend:     naive (MUST, in-process zero IPC)")
    lines.append(f"  use_critic:             False (GRPO outcome-only)")
    lines.append(f"  strategy:               fsdp (FSDP backend)")
    lines.append(f"  ref_in_actor:           True (LoRA-detached, no separate worker)")

    # Phase timing for optimal config
    lines.append("")
    lines.append("-" * 80)
    lines.append("  Phase Timing (Optimal Config)")
    lines.append("-" * 80)

    max_t = max(p.timing_s for p in r.phases)
    for p in r.phases:
        if p.timing_s > 0:
            bar = _phase_bar(p.timing_s, max_t, width=25)
            lines.append(f"  Phase {p.phase:2d}: {p.timing_s:.3f}s {bar}  {p.name}")

    lines.append(f"  Total:  {r.total_time_s:.3f}s")

    # MUST DO / MUST NOT rules
    lines.append("")
    lines.append("=" * 80)
    lines.append("  RTX 4090 GRPO MUST DO (10 rules)")
    lines.append("=" * 80)
    must_do = [
        "1.  trainer_mode = 'sync' (colocated, no disaggregation)",
        "2.  rollout.n >= 8 (GRPO group size, n=1 = REINFORCE degeneration)",
        "3.  bypass_mode = True (skip old_log_prob forward, 18Psi -> 3.8Psi)",
        "4.  enforce_eager = True (SM89, no cudagraph for DSV4)",
        "5.  checkpoint_engine.backend = 'naive' (in-process, zero IPC overhead)",
        "6.  actor.strategy = 'fsdp' (FSDP for training)",
        "7.  use_critic = False (GRPO is outcome-only)",
        "8.  LoRA rank = 32 (NOT 64, see #6782 EOS bug)",
        "9.  ref_in_actor = True (LoRA detached as ref, no separate worker)",
        "10. free_cache_engine = True (free KV cache during sleep)",
    ]
    for rule in must_do:
        lines.append(f"  {rule}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  RTX 4090 GRPO MUST NOT (10 rules)")
    lines.append("=" * 80)
    must_not = [
        "1.  rollout.n = 1 (REINFORCE degeneration, no variance reduction)",
        "2.  ZeRO-3 on single GPU (pure overhead, use ZeRO-2 + CPU_Adam)",
        "3.  ZeRO-2 + Muon (BLOCKED, #7939 closed)",
        "4.  LoRA rank = 64 (breaks EOS in vLLM rollout, #6782)",
        "5.  bypass_mode = False (wastes 15Psi on old_log_prob forward)",
        "6.  checkpoint_engine.backend != 'naive' (IPC overhead for sync)",
        "7.  enforce_eager = False (cudagraph fails on SM89 with DSV4)",
        "8.  overlap_comm = True (multi-stream NaN on single GPU, #8061)",
        "9.  gradient_clipping = 0 (ALWAYS set 1.0, #8068)",
        "10. sleep_level = 2 (full re-transfer, use sleep_level=1 LoRA adapter)",
    ]
    for rule in must_not:
        lines.append(f"  {rule}")

    # Warnings from simulation
    if r.warnings:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Active Warnings")
        lines.append("-" * 80)
        for w in r.warnings:
            lines.append(f"  * {w}")

    return "\n".join(lines)


def display_lifecycle() -> str:
    """Show the 10-phase lifecycle with memory/timing estimates."""
    lines = []
    lines.append("=" * 80)
    lines.append("verl V1 Sync PPOTrainer GRPO — 10-Phase Training Step Lifecycle")
    lines.append("=" * 80)
    lines.append("  Based on deep reading of verl/trainer/ppo/v1/trainer_base.py:404-456")
    lines.append("  Call tree: PPOTrainer.step(metrics, timing_raw)")
    lines.append("  Data flow: TransferQueue-centric (tq.kv_batch_put/get/clear)")
    lines.append("")

    # Optimal config for timing reference
    cfg = SimConfig(model="qwen3-4b", bypass_mode=True)
    sim = GRPOStepSimulator(cfg)
    ref_result = sim.simulate_step()

    max_t = max(p.timing_s for p in ref_result.phases) if ref_result.phases else 1

    for phase_num in range(11):
        info = PHASE_INFO[phase_num]
        phase_r = ref_result.phases[phase_num]

        lines.append("-" * 80)
        lines.append(f"  Phase {phase_num}: {info['name']}")
        lines.append("-" * 80)
        lines.append(f"    Method: {info['method']}")
        lines.append(f"    Description: {info['description']}")
        lines.append(f"    GPU active: {phase_r.gpu_active}")
        lines.append(f"    Timing: {phase_r.timing_s:.3f}s {_phase_bar(phase_r.timing_s, max_t, width=20)}")
        lines.append(f"    Memory peak: {phase_r.memory_peak_gb:.2f} GiB ({phase_r.memory_type})")
        lines.append(f"    Notes: {info['notes']}")

        # Show details
        for k, v in phase_r.details.items():
            if v is not None:
                lines.append(f"    {k}: {v}")
        lines.append("")

    # Sleep/wake lifecycle pattern
    lines.append("=" * 80)
    lines.append("  Sleep/Wake Lifecycle Pattern")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  [Init] -> sleep_replicas() -> weights freed, KV cache freed")
    lines.append("    |")
    lines.append("  [Step N starts] -> Phase 0: wake for weight sync")
    lines.append("    |              -> Phase 1: rollout generation (PEAK: weights + KV cache)")
    lines.append("    |")
    lines.append("  [Phase 3] -> on_sample_end -> sleep_replicas() (free rollout memory)")
    lines.append("    |              -> Advantage on CPU, no GPU needed")
    lines.append("    |")
    lines.append("  [Phase 6] -> bypass: CPU-only (TransferQueue rename)")
    lines.append("    |         full: GPU forward pass needed")
    lines.append("    |")
    lines.append("  [Phase 10] -> actor update (PEAK: FSDP + LoRA + optimizer + activations)")
    lines.append("    |")
    lines.append("  [Post-Step] -> update_weights() -> wake + weight sync")
    lines.append("    |")
    lines.append("  [Step N+1 starts] -> Phase 0 again...")
    lines.append("")
    lines.append("  KEY: Peaks DO NOT overlap (sleep/wake pattern essential for RTX 4090)")
    lines.append("  Effective peak = max(rollout_peak, training_peak), NOT sum")

    # TransferQueue operations
    lines.append("")
    lines.append("=" * 80)
    lines.append("  TransferQueue Operations Per Step")
    lines.append("=" * 80)
    tq_ops = [
        ("Phase 1", "PUT: prompts with uid, tags={is_prompt, status=pending}"),
        ("Phase 1", "PUT: rollout outputs (async, AgentLoop)"),
        ("Phase 2", "GET: sample keys from finished prompts"),
        ("Phase 6", "GET: rollout_log_probs (bypass) OR compute (full)"),
        ("Phase 6", "PUT: old_log_probs"),
        ("Phase 7", "GET+PUT: ref_log_prob (optional, LoRA-detached)"),
        ("Phase 8", "GET: uid, response_mask, rm_scores, old_log_probs, ref_log_prob"),
        ("Phase 8", "PUT: advantages, returns"),
        ("Phase 10", "GET: actor update data"),
        ("Post-Step", "CLEAR: all keys, partition_id"),
    ]
    lines.append(f"  {'Phase':<12} {'Operation':<65}")
    lines.append(f"  {'-'*12} {'-'*65}")
    for phase, op in tq_ops:
        lines.append(f"  {phase:<12} {op:<65}")

    # Source files reference
    lines.append("")
    lines.append("=" * 80)
    lines.append("  Key Source Files")
    lines.append("=" * 80)
    sources = [
        "trainer_base.py:299-456    — PPOTrainer.fit() + step()",
        "trainer_sync.py:31-42      — PPOTrainerSync hooks",
        "replay_buffer.py           — ReplayBuffer + TransferQueue",
        "utils.py:23-92             — compute_advantage_for_multi_trajectories",
        "agent_loop_tq.py           — AgentLoopWorkerTQ, TransferQueue-based",
        "core_algos.py:268-331      — compute_grpo_outcome_advantage",
        "core_algos.py:1279-1369    — compute_policy_loss_vanilla (PPO-clip)",
        "losses.py:57-144           — ppo_loss function",
        "checkpoint_engine/base.py  — CheckpointEngineManager (sleep/wake/update)",
    ]
    for s in sources:
        lines.append(f"  verl/trainer/ppo/v1/{s}")

    return "\n".join(lines)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="verl V1 Sync PPOTrainer GRPO Training Step Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  simulate   Run a full GRPO training step simulation
  compare    Compare bypass vs full, group sizes, loss types
  rtx4090    RTX 4090-specific simulation with 24 GiB budget
  lifecycle  Show the 10-phase lifecycle with memory/timing estimates

Examples:
  python verl_grpo_step_simulator.py simulate
  python verl_grpo_step_simulator.py simulate --model qwen2.5-7b --group_size 16
  python verl_grpo_step_simulator.py simulate --model qwen3-8b --bypass_mode False
  python verl_grpo_step_simulator.py compare
  python verl_grpo_step_simulator.py rtx4090
  python verl_grpo_step_simulator.py lifecycle
        """,
    )
    parser.add_argument("mode", choices=["simulate", "compare", "rtx4090", "lifecycle"],
                        help="Simulation mode")
    parser.add_argument("--model", default="qwen3-4b",
                        choices=list(MODEL_PROFILES.keys()),
                        help="Model to simulate")
    parser.add_argument("--group_size", type=int, default=8,
                        help="GRPO group size (rollout.n)")
    parser.add_argument("--bypass_mode", type=str, default="True",
                        help="bypass_mode: True or False")
    parser.add_argument("--loss_type", default="ppo_clip",
                        choices=["ppo_clip", "reinforce"],
                        help="Policy loss type")
    parser.add_argument("--lora_rank", type=int, default=32,
                        choices=[16, 32, 64],
                        help="LoRA rank (MUST NOT 64)")
    parser.add_argument("--num_prompts", type=int, default=4,
                        help="Number of prompts per step")
    parser.add_argument("--max_response_len", type=int, default=512,
                        help="Max response tokens per trajectory")
    parser.add_argument("--ppo_epochs", type=int, default=1,
                        help="PPO update epochs per step")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                        help="Mini-batches per PPO epoch")
    parser.add_argument("--sleep_level", type=int, default=1,
                        choices=[1, 2],
                        help="Sleep level (1=LoRA adapter, 2=full weight transfer)")
    parser.add_argument("--enforce_eager", type=str, default="True",
                        help="enforce_eager: True or False")
    parser.add_argument("--gradient_clipping", type=float, default=1.0,
                        help="Gradient clipping value (MUST NOT 0)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted text")

    args = parser.parse_args()

    # Parse bool args
    bypass = args.bypass_mode.lower() in ("true", "1", "yes")
    eager = args.enforce_eager.lower() in ("true", "1", "yes")

    if args.mode == "simulate":
        config = SimConfig(
            model=args.model,
            group_size=args.group_size,
            bypass_mode=bypass,
            loss_type=args.loss_type,
            lora_rank=args.lora_rank,
            num_prompts=args.num_prompts,
            max_response_len=args.max_response_len,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
            enforce_eager=eager,
            gradient_clipping=args.gradient_clipping,
            sleep_level=args.sleep_level,
        )
        sim = GRPOStepSimulator(config)
        result = sim.simulate_step()

        if args.json:
            output = {
                "config": {
                    "model": config.model,
                    "group_size": config.group_size,
                    "bypass_mode": config.bypass_mode,
                    "loss_type": config.loss_type,
                    "lora_rank": config.lora_rank,
                    "num_prompts": config.num_prompts,
                    "max_response_len": config.max_response_len,
                },
                "phases": [
                    {
                        "phase": p.phase,
                        "name": p.name,
                        "timing_s": p.timing_s,
                        "memory_peak_gb": p.memory_peak_gb,
                        "gpu_active": p.gpu_active,
                        "details": p.details,
                    }
                    for p in result.phases
                ],
                "total_time_s": result.total_time_s,
                "peak_memory_gb": result.peak_memory_gb,
                "advantage_quality": result.advantage_quality,
                "feasibility": result.feasibility,
                "warnings": result.warnings,
            }
            print(json.dumps(output, indent=2))
        else:
            print(display_simulate(result))

    elif args.mode == "compare":
        print(display_compare())

    elif args.mode == "rtx4090":
        print(display_rtx4090())

    elif args.mode == "lifecycle":
        print(display_lifecycle())


if __name__ == "__main__":
    main()
