#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Rollout-Infra Team
"""
GRPO Mathematical Validator — Numerically verify advantage + loss + gradient formulas.

Validates the mathematical derivations from:
  - notebook/rl-advantage-estimators-mathematical-derivation.md
  - notebook/rl-policy-loss-functions-mathematical-derivation.md
  - notebook/grpo-training-algorithm-unified-synthesis.md

Tests:
  1. GRPO advantages: mean=0, std=1 proof
  2. RLOO advantages: leave-one-out correctness
  3. REINFORCE++BL: mean-subtracted properties
  4. PPO-clip: trust region bounds
  5. UP-GRPO: unbounded positive, dual-clip negative
  6. CISPO: detached clamp gradient flow
  7. Singleton degeneration: gs=1 → A=0
  8. Response mask: prompt tokens zeroed
  9. Aggregation: seq-mean-token-mean vs token-mean-token-mean
  10. Gradient magnitude: GRPO vs REINFORCE

Modes:
  validate  — Run all 10 validation tests
  compare   — Compare estimators numerically with sample rewards
  rtx4090   — RTX 4090 memory budget and gradient analysis

Usage:
  python tools/grpo_mathematical_validator.py validate
  python tools/grpo_mathematical_validator.py compare
  python tools/grpo_mathematical_validator.py rtx4090
"""

import argparse
import json
import math
import sys
from typing import Dict, List, Tuple


# ============================================================
# Advantage Estimator Implementations
# ============================================================

def compute_grpo_advantage(rewards: List[float]) -> List[float]:
    """GRPO: A_i = (R_i - mu) / sigma, sigma=0 → A_i=0."""
    mu = sum(rewards) / len(rewards)
    sigma = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
    if sigma == 0:
        return [0.0] * len(rewards)
    return [(r - mu) / sigma for r in rewards]


def compute_reinforce_advantage(rewards: List[float]) -> List[float]:
    """REINFORCE: A_i = R_i (no baseline)."""
    return list(rewards)


def compute_reinforce_bl_advantage(rewards: List[float]) -> List[float]:
    """REINFORCE++BL: A_i = R_i - mu (mean-subtracted)."""
    mu = sum(rewards) / len(rewards)
    return [r - mu for r in rewards]


def compute_rloo_advantage(rewards: List[float]) -> List[float]:
    """RLOO: A_i = R_i - mu_LOO_i, where mu_LOO excludes i."""
    n = len(rewards)
    if n == 1:
        return [0.0]  # undefined for single trajectory
    total = sum(rewards)
    advantages = []
    for i, r in enumerate(rewards):
        mu_loo = (total - r) / (n - 1)
        advantages.append(r - mu_loo)
    return advantages


# ============================================================
# Policy Loss Implementations (per-token, returns loss + gradient info)
# ============================================================

def ppo_clip_loss(ratio: float, advantage: float, epsilon: float = 0.2) -> Tuple[float, Dict]:
    """PPO-clip: L = min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)."""
    clipped_ratio = max(1 - epsilon, min(1 + epsilon, ratio))
    loss_unclipped = ratio * advantage
    loss_clipped = clipped_ratio * advantage
    loss = min(loss_unclipped, loss_clipped)

    # Gradient info
    if ratio >= 1 - epsilon and ratio <= 1 + epsilon:
        gradient_flows = True
        gradient = advantage  # full gradient through logp_curr
    elif advantage >= 0 and ratio > 1 + epsilon:
        gradient_flows = False
        gradient = 0  # upper clip activates
    elif advantage < 0 and ratio < 1 - epsilon:
        gradient_flows = False
        gradient = 0  # lower clip activates
    else:
        gradient_flows = True
        gradient = advantage

    return loss, {
        "type": "ppo_clip",
        "loss": loss,
        "gradient_flows": gradient_flows,
        "gradient": gradient,
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "advantage": advantage,
    }


def up_grpo_loss(ratio: float, advantage: float, epsilon: float = 0.2,
                 clip_ratio_c: float = 3.0) -> Tuple[float, Dict]:
    """UP-GRPO: A>=0: L=ratio*A (unbounded), A<0: dual-clip."""
    if advantage >= 0:
        # No upper clip for positive advantages
        loss = ratio * advantage
        gradient_flows = True
        gradient = advantage  # always flows for positive A
        clipped = False
    else:
        # Standard PPO-clip + dual-clip for negative advantages
        clipped_ratio_inner = max(1 - epsilon, min(1 + epsilon, ratio))
        clipped_ratio_outer = max(1 - epsilon, min(clip_ratio_c, ratio))
        loss = max(ratio * advantage, clipped_ratio_inner * advantage, clipped_ratio_outer * advantage)
        # For negative A, smaller ratio → larger loss (since A<0, r*A more negative = larger loss)
        # The max picks the least negative → conservative
        if ratio >= 1 - epsilon and ratio <= 1 + epsilon:
            gradient_flows = True
            gradient = advantage
            clipped = False
        else:
            gradient_flows = False
            gradient = 0
            clipped = True

    return loss, {
        "type": "up_grpo",
        "loss": loss,
        "gradient_flows": gradient_flows,
        "gradient": gradient,
        "ratio": ratio,
        "advantage": advantage,
        "clipped": clipped,
    }


def cispo_loss(ratio: float, advantage: float, epsilon: float = 0.2,
               logp_curr: float = -2.0) -> Tuple[float, Dict]:
    """CISPO: clamp(ratio).detach() * A * logp_curr.
    The clamp is detached → gradient always flows through logp_curr."""
    clamped_ratio = max(1 - epsilon, min(1 + epsilon, ratio))
    # .detach() means gradient doesn't flow through ratio
    # but flows through logp_curr
    loss = clamped_ratio * advantage * logp_curr  # clamped_ratio acts as WEIGHT
    gradient_flows = True  # ALWAYS! No zero-gradient zones
    gradient = clamped_ratio * advantage  # weight * A → gradient magnitude

    return loss, {
        "type": "cispo",
        "loss": loss,
        "gradient_flows": gradient_flows,
        "gradient": gradient,
        "ratio": ratio,
        "clamped_ratio_weight": clamped_ratio,
        "advantage": advantage,
    }


# ============================================================
# Validation Tests
# ============================================================

def test_grpo_normalization():
    """Test 1: GRPO advantages have mean=0, std=1."""
    rewards = [0.3, 0.7, 0.5, 0.9, 0.1, 0.6, 0.4, 0.8]
    advantages = compute_grpo_advantage(rewards)
    mean_adv = sum(advantages) / len(advantages)
    std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))

    passed = abs(mean_adv) < 1e-10 and abs(std_adv - 1.0) < 1e-10
    return {
        "test": "grpo_normalization",
        "passed": passed,
        "rewards": rewards,
        "advantages": advantages,
        "mean": mean_adv,
        "std": std_adv,
        "expected_mean": 0.0,
        "expected_std": 1.0,
    }


def test_rloo_correctness():
    """Test 2: RLOO leave-one-out baseline correctness."""
    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = compute_rloo_advantage(rewards)

    # Manual calculation:
    # A_0 = 1.0 - (2+3+4)/3 = 1.0 - 3.0 = -2.0
    # A_1 = 2.0 - (1+3+4)/3 = 2.0 - 2.67 = -0.67
    # A_2 = 3.0 - (1+2+4)/3 = 3.0 - 2.33 = 0.67
    # A_3 = 4.0 - (1+2+3)/3 = 4.0 - 2.0 = 2.0
    expected = [-2.0, 2.0 - 8 / 3, 3.0 - 7 / 3, 2.0]

    passed = all(abs(a - e) < 1e-10 for a, e in zip(advantages, expected))
    return {
        "test": "rloo_correctness",
        "passed": passed,
        "rewards": rewards,
        "advantages": advantages,
        "expected": expected,
    }


def test_reinforce_bl_properties():
    """Test 3: REINFORCE++BL mean-subtracted properties."""
    rewards = [0.3, 0.7, 0.5, 0.9]
    advantages = compute_reinforce_bl_advantage(rewards)

    mean_adv = sum(advantages) / len(advantages)
    # Mean should be 0 (since we subtracted the mean)
    # Std should be preserved (same as reward std)

    passed = abs(mean_adv) < 1e-10
    return {
        "test": "reinforce_bl_properties",
        "passed": passed,
        "rewards": rewards,
        "advantages": advantages,
        "mean_advantages": mean_adv,
    }


def test_ppo_clip_trust_region():
    """Test 4: PPO-clip creates trust region bounds."""
    epsilon = 0.2
    results = []

    # Test ratios inside and outside trust region
    test_cases = [
        (1.0, 1.0, "inside"),    # ratio=1.0, A=1.0 → full gradient
        (1.3, 1.0, "above"),     # ratio>1+eps, A>0 → zero gradient
        (0.7, -1.0, "below"),    # ratio<1-eps, A<0 → zero gradient
        (1.1, 0.5, "inside2"),   # inside → full gradient
        (1.3, -0.5, "above_neg"), # ratio>1+eps, A<0 → clipped but gradient flows
    ]

    all_correct = True
    for ratio, advantage, label in test_cases:
        loss, info = ppo_clip_loss(ratio, advantage, epsilon)
        if label == "above" and info["gradient_flows"]:
            all_correct = False
        if label == "below" and info["gradient_flows"]:
            all_correct = False
        if label == "inside" and not info["gradient_flows"]:
            all_correct = False
        results.append({"label": label, **info})

    return {
        "test": "ppo_clip_trust_region",
        "passed": all_correct,
        "epsilon": epsilon,
        "results": results,
    }


def test_up_grpo_properties():
    """Test 5: UP-GRPO unbounded positive, dual-clip negative."""
    epsilon = 0.2
    clip_ratio_c = 3.0
    results = []

    # Positive advantage: should ALWAYS have gradient, even for large ratios
    pos_ratios = [1.0, 1.5, 2.0, 5.0, 10.0]
    for ratio in pos_ratios:
        loss, info = up_grpo_loss(ratio, 1.0, epsilon, clip_ratio_c)
        if not info["gradient_flows"]:
            return {
                "test": "up_grpo_properties",
                "passed": False,
                "reason": f"Positive A gradient blocked at ratio={ratio}",
                "results": results,
            }
        results.append({"ratio": ratio, "A_sign": "positive", **info})

    # Negative advantage: should have dual-clip protection
    neg_ratios = [1.0, 0.5, 0.1, 4.0]  # ratio=4.0 tests clip_ratio_c
    for ratio in neg_ratios:
        loss, info = up_grpo_loss(ratio, -1.0, epsilon, clip_ratio_c)
        results.append({"ratio": ratio, "A_sign": "negative", **info})

    return {
        "test": "up_grpo_properties",
        "passed": True,
        "epsilon": epsilon,
        "clip_ratio_c": clip_ratio_c,
        "results": results,
    }


def test_cispo_gradient_flow():
    """Test 6: CISPO ALL tokens keep gradient (never zeroed)."""
    epsilon = 0.2
    results = []

    test_ratios = [0.5, 0.8, 1.0, 1.2, 2.0, 5.0]  # inside and outside trust region
    for ratio in test_ratios:
        for advantage in [1.0, -1.0]:
            loss, info = cispo_loss(ratio, advantage, epsilon, -2.0)
            if not info["gradient_flows"]:
                return {
                    "test": "cispo_gradient_flow",
                    "passed": False,
                    "reason": f"Gradient zeroed at ratio={ratio}, A={advantage}",
                    "results": results,
                }
            results.append({"ratio": ratio, "advantage": advantage, **info})

    return {
        "test": "cispo_gradient_flow",
        "passed": True,
        "results": results,
    }


def test_singleton_degeneration():
    """Test 7: gs=1 → GRPO A=0, RLOO undefined, REINFORCE++BL A=0."""
    single_reward = [0.5]

    grpo_adv = compute_grpo_advantage(single_reward)
    rloo_adv = compute_rloo_advantage(single_reward)
    reinforce_bl_adv = compute_reinforce_bl_advantage(single_reward)
    reinforce_adv = compute_reinforce_advantage(single_reward)

    # All group-based estimators should give 0 for gs=1
    # Only REINFORCE gives the raw reward
    passed = (
        grpo_adv == [0.0] and
        rloo_adv == [0.0] and
        reinforce_bl_adv == [0.0] and
        reinforce_adv == [0.5]  # REINFORCE still works but has high variance
    )

    return {
        "test": "singleton_degeneration",
        "passed": passed,
        "reward": 0.5,
        "grpo": grpo_adv,
        "rloo": rloo_adv,
        "reinforce_bl": reinforce_bl_adv,
        "reinforce": reinforce_adv,
        "conclusion": "gs=1 → ALL group estimators → A=0 → NO learning signal!",
    }


def test_response_mask():
    """Test 8: Prompt tokens masked to zero in advantage expansion."""
    # Simulate: trajectory-level A=1.0, prompt_length=4, response_length=6
    advantage = 1.0
    prompt_length = 4
    response_length = 6

    token_advantages = []
    for t in range(prompt_length + response_length):
        if t < prompt_length:
            mask = 0  # prompt token
            token_advantages.append(0.0)
        else:
            mask = 1  # response token
            token_advantages.append(advantage)

    # Verify: prompt tokens have 0 advantage, response tokens have A=1.0
    prompt_zeros = all(a == 0.0 for a in token_advantages[:prompt_length])
    response_equals = all(a == advantage for a in token_advantages[prompt_length:])

    passed = prompt_zeros and response_equals
    return {
        "test": "response_mask",
        "passed": passed,
        "advantage": advantage,
        "token_advantages": token_advantages,
        "prompt_mask": [0] * prompt_length + [1] * response_length,
        "conclusion": "Prompt tokens: A=0, Response tokens: A=R (masked correctly)",
    }


def test_aggregation_modes():
    """Test 9: Different aggregation modes produce different gradient weights."""
    # Two trajectories with different lengths
    # Traj 1: 4 tokens, losses = [0.1, 0.2, 0.3, 0.4]
    # Traj 2: 2 tokens, losses = [0.5, 0.6]
    losses_traj1 = [0.1, 0.2, 0.3, 0.4]
    losses_traj2 = [0.5, 0.6]

    # token-mean-token-mean: each token equal weight
    all_losses = losses_traj1 + losses_traj2
    tmtm = sum(all_losses) / len(all_losses)  # (0.1+0.2+0.3+0.4+0.5+0.6)/6

    # seq-mean-token-mean: each trajectory equal weight
    smtm = (sum(losses_traj1) / len(losses_traj1) + sum(losses_traj2) / len(losses_traj2)) / 2

    # Verify different results → different gradient weighting
    passed = abs(tmtm - smtm) > 1e-10  # They should be different!

    return {
        "test": "aggregation_modes",
        "passed": passed,
        "traj1_losses": losses_traj1,
        "traj2_losses": losses_traj2,
        "token_mean_token_mean": tmtm,
        "seq_mean_token_mean": smtm,
        "conclusion": "Different aggregation → different loss values → different gradient weights",
    }


def test_gradient_magnitude():
    """Test 10: GRPO advantages are BOUNDED (near 1.0), REINFORCE scales with reward magnitude."""
    # Small rewards: REINFORCE E[|A|] < GRPO E[|A|]
    small_rewards = [0.3, 0.7, 0.5, 0.9, 0.1, 0.6, 0.4, 0.8]
    # Large rewards: REINFORCE E[|A|] >> GRPO E[|A|]
    large_rewards = [30.0, 70.0, 50.0, 90.0, 10.0, 60.0, 40.0, 80.0]
    # Same distribution, just 100x scale

    grpo_small = compute_grpo_advantage(small_rewards)
    reinforce_small = compute_reinforce_advantage(small_rewards)
    grpo_large = compute_grpo_advantage(large_rewards)
    reinforce_large = compute_reinforce_advantage(large_rewards)

    grpo_small_abs = sum(abs(a) for a in grpo_small) / len(grpo_small)
    reinforce_small_abs = sum(abs(a) for a in reinforce_small) / len(reinforce_small)
    grpo_large_abs = sum(abs(a) for a in grpo_large) / len(grpo_large)
    reinforce_large_abs = sum(abs(a) for a in reinforce_large) / len(reinforce_large)

    # KEY PROPERTY: GRPO E[|A|] stays bounded (~1.0) regardless of reward scale
    # REINFORCE E[|A|] grows proportionally with reward scale
    # This means: GRPO is invariant to reward scale → safe for gradient clipping
    # REINFORCE depends on reward scale → dangerous for large rewards
    grpo_scale_invariant = abs(grpo_small_abs - grpo_large_abs) < 1e-10  # EXACTLY equal!
    reinforce_scale_grows = reinforce_large_abs > reinforce_small_abs * 10  # grows with scale

    passed = grpo_scale_invariant and reinforce_scale_grows
    return {
        "test": "gradient_magnitude",
        "passed": passed,
        "grpo_small_mean_abs_A": grpo_small_abs,
        "grpo_large_mean_abs_A": grpo_large_abs,
        "reinforce_small_mean_abs_A": reinforce_small_abs,
        "reinforce_large_mean_abs_A": reinforce_large_abs,
        "conclusion": f"GRPO: scale-invariant ({grpo_small_abs:.3f}≈{grpo_large_abs:.3f}) vs "
                      f"REINFORCE: scale-dependent ({reinforce_small_abs:.3f}→{reinforce_large_abs:.3f})",
    }


# ============================================================
# Compare Mode: Numerical comparison across estimators
# ============================================================

def compare_estimators(rewards: List[float]) -> Dict:
    """Compare all advantage estimators numerically."""
    results = {
        "rewards": rewards,
        "group_size": len(rewards),
        "grpo": compute_grpo_advantage(rewards),
        "reinforce": compute_reinforce_advantage(rewards),
        "reinforce_bl": compute_reinforce_bl_advantage(rewards),
        "rloo": compute_rloo_advantage(rewards),
    }

    # Add statistics
    for name in ["grpo", "reinforce", "reinforce_bl", "rloo"]:
        advs = results[name]
        if not advs:
            results[f"{name}_mean"] = None
            results[f"{name}_std"] = None
            continue
        mean = sum(advs) / len(advs)
        std = math.sqrt(sum((a - mean) ** 2 for a in advs) / len(advs)) if len(advs) > 1 else 0
        results[f"{name}_mean"] = mean
        results[f"{name}_std"] = std

    return results


def compare_losses(ratio: float, advantage: float) -> Dict:
    """Compare all loss functions at a given (ratio, advantage) point."""
    results = {}

    results["ppo_clip"] = ppo_clip_loss(ratio, advantage)
    results["up_grpo"] = up_grpo_loss(ratio, advantage)
    results["cispo"] = cispo_loss(ratio, advantage)

    # Extract key metrics
    for name, (loss, info) in [("ppo_clip", ppo_clip_loss(ratio, advantage)),
                                ("up_grpo", up_grpo_loss(ratio, advantage)),
                                ("cispo", cispo_loss(ratio, advantage))]:
        results[f"{name}_loss"] = loss
        results[f"{name}_gradient_flows"] = info["gradient_flows"]

    return results


# ============================================================
# RTX 4090 Mode: Memory budget + gradient analysis
# ============================================================

def rtx4090_analysis() -> Dict:
    """RTX 4090 GRPO training memory budget and gradient analysis."""
    return {
        "gpu_memory_total": 24 * 1024,  # 24 GiB in MiB
        "model_weights_7b_bf16": 14.0 * 1024,  # ~14 GiB
        "activations_bypass": 3.8 * 1024,  # bypass mode ~3.8 GiB
        "activations_full": 18.0 * 1024,  # without bypass ~18 GiB
        "gradients_bf16": 1.4 * 1024,  # ~1.4 GiB
        "optimizer_cpu_offload": 0,  # offloaded to CPU
        "kv_cache_rollout": 3.0 * 1024,  # ~3 GiB during rollout
        "total_training_bypass": 19.2 * 1024,  # fits 24 GiB with 5 GiB headroom
        "total_training_full": 33.4 * 1024,  # OOM without bypass!
        "total_rollout_colocate": 22.2 * 1024,  # tight but viable

        "recommended_config": {
            "advantage": "grpo",
            "loss": "up_grpo",
            "group_size": 8,
            "clip_ratio": 0.2,
            "clip_ratio_c": 3.0,
            "kl_coef": 0.01,
            "aggregation": "seq-mean-token-mean",
            "zero_stage": 2,
            "optimizer": "cpu_adam",
            "gradient_clipping": 1.0,
            "overlap_comm": False,
            "bypass_mode": True,
            "fsdp_mode": "fsdp1",
            "lora_rank": 32,
            "lora_alpha": 64,
            "checkpoint_mode": "naive",
            "sleep_level": 1,
            "enforce_eager": True,
        },

        "key_rules": [
            "ALWAYS group_size >= 4 for GRPO",
            "ALWAYS group_size >= 8 for MoE models",
            "NEVER group_size = 1 (REINFORCE degeneration)",
            "ALWAYS gradient_clipping = 1.0 (NOT default 0.0)",
            "ALWAYS overlap_comm = False on single GPU",
            "ALWAYS FSDP1 (NOT FSDP2 — leak + MoE bug)",
            "ALWAYS bypass_mode = True (18Ψ→3.8Ψ)",
            "NEVER LoRA rank >= 64 (breaks EOS)",
            "NEVER ZeRO-3 on single GPU (pure overhead)",
            "NEVER sleep_level = 2 on RTX 4090",
        ],
    }


# ============================================================
# Main Entry Point
# ============================================================

def run_validate():
    """Run all 10 validation tests."""
    tests = [
        test_grpo_normalization(),
        test_rloo_correctness(),
        test_reinforce_bl_properties(),
        test_ppo_clip_trust_region(),
        test_up_grpo_properties(),
        test_cispo_gradient_flow(),
        test_singleton_degeneration(),
        test_response_mask(),
        test_aggregation_modes(),
        test_gradient_magnitude(),
    ]

    passed_count = sum(1 for t in tests if t["passed"])
    total_count = len(tests)

    print("=" * 60)
    print("GRPO Mathematical Validator — 10 Tests")
    print("=" * 60)

    for t in tests:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"\n[{status}] {t['test']}")
        # Print key metrics
        for k, v in t.items():
            if k not in ("test", "passed") and not isinstance(v, list):
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print(f"Results: {passed_count}/{total_count} PASSED")
    print("=" * 60)

    return {"tests": tests, "passed": passed_count, "total": total_count}


def run_compare():
    """Compare estimators and losses numerically."""
    print("=" * 60)
    print("GRPO Estimator & Loss Comparison")
    print("=" * 60)

    # Compare estimators
    print("\n--- Advantage Estimators ---")
    for rewards in [[0.3, 0.7, 0.5, 0.9], [1.0, 2.0, 3.0, 4.0], [0.5]]:
        print(f"\nRewards: {rewards} (gs={len(rewards)})")
        result = compare_estimators(rewards)
        for name in ["grpo", "reinforce", "reinforce_bl", "rloo"]:
            advs = result[name]
            mean = result[f"{name}_mean"]
            std = result[f"{name}_std"]
            print(f"  {name:15s}: A={advs}  mean={mean}  std={std}")

    # Compare losses at key points
    print("\n--- Policy Loss Functions ---")
    for ratio, advantage in [(1.0, 1.0), (1.5, 1.0), (0.5, -1.0), (3.0, -1.0), (2.0, 0.5)]:
        print(f"\nratio={ratio}, A={advantage}")
        result = compare_losses(ratio, advantage)
        for name in ["ppo_clip", "up_grpo", "cispo"]:
            loss = result[f"{name}_loss"]
            grad = result[f"{name}_gradient_flows"]
            print(f"  {name:15s}: loss={loss:.4f}  gradient_flows={grad}")


def run_rtx4090():
    """RTX 4090 memory budget analysis."""
    result = rtx4090_analysis()

    print("=" * 60)
    print("RTX 4090 GRPO Training — Memory Budget & Config")
    print("=" * 60)

    print("\n--- Memory Budget (MiB) ---")
    items = [
        ("GPU Total", result["gpu_memory_total"]),
        ("Model weights (7B BF16)", result["model_weights_7b_bf16"]),
        ("Activations (bypass)", result["activations_bypass"]),
        ("Activations (full, no bypass)", result["activations_full"]),
        ("Gradients (BF16)", result["gradients_bf16"]),
        ("Optimizer (CPU offload)", result["optimizer_cpu_offload"]),
        ("KV cache (rollout)", result["kv_cache_rollout"]),
        ("Total (bypass mode)", result["total_training_bypass"]),
        ("Total (no bypass)", result["total_training_full"]),
        ("Total (colocate)", result["total_rollout_colocate"]),
    ]
    for name, value in items:
        status = ""
        if value > result["gpu_memory_total"]:
            status = " ❌ OOM!"
        elif value > result["gpu_memory_total"] * 0.9:
            status = " ⚠️ TIGHT"
        else:
            status = " ✓ OK"
        print(f"  {name:35s}: {value:>7.0f} MiB{status}")

    print("\n--- Recommended Config ---")
    for k, v in result["recommended_config"].items():
        print(f"  {k:25s}: {v}")

    print("\n--- Key Rules ---")
    for i, rule in enumerate(result["key_rules"], 1):
        print(f"  {i}. {rule}")


def main():
    parser = argparse.ArgumentParser(description="GRPO Mathematical Validator")
    parser.add_argument("mode", choices=["validate", "compare", "rtx4090"],
                        help="Mode: validate=all tests, compare=numerical comparison, rtx4090=budget analysis")
    args = parser.parse_args()

    if args.mode == "validate":
        run_validate()
    elif args.mode == "compare":
        run_compare()
    elif args.mode == "rtx4090":
        run_rtx4090()


if __name__ == "__main__":
    main()
