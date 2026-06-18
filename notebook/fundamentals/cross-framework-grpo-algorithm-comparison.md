# Cross-Framework GRPO Algorithm Comparison — RTX 4090 Consultant Reference

> 2026-06-18 | Synthesis of GRPO/policy optimization algorithms across all 7 frameworks
> ★★★★★★★★ verl CPPO+GRPO+bypass = RTX 4090 #1 BEST | rLLM GRPO BROKEN (#605) | DeepSpeed ZeRO-2 only
> ★★★★★★★★ Trust region comparison: CPPO (position-weighted cumulative) > DPPO (pointwise divergence) > PPO (heuristic clip) > GRPO (none)

---

## 1. Policy Optimization Algorithms — Cross-Framework Comparison

```
★★★★★★★★★ Algorithm landscape across 7 frameworks:

| Algorithm | Trust Region | Ref Model | Critic | Framework | RTX 4090 Rank |
|-----------|-------------|-----------|--------|-----------|---------------|
| PPO (standard) | Heuristic ratio clip | Yes (~14GB) | Yes (~2GB) | vLLM+verl (legacy) | #5 (needs ref+critic) |
| GRPO | None (group-relative) | No (bypass_mode) | No (group) | verl, rLLM | #2-3 (no extras) |
| CPPO | Position-weighted cumulative divergence | No (bypass) | No (GRPO adv) | verl #6731 | ★★★★★★★★ #1 BEST! |
| DPPO | Pointwise binary-TV divergence | Yes (unless bypass) | Yes | verl (config) | #4 (needs careful config) |
| DAPO | Dynamic advantage norm | No | No | verl (DAPO config) | #2 (same as GRPO+norm) |
| REINFORCE | None | No | No | rLLM (fallback) | #6 (no variance reduction) |

★★★★★★★★★ RTX 4090 GRPO training ranking (updated):
  #1: verl CPPO + GRPO advantage + bypass_mode → best trust region + no ref/critic
  #1.5: verl Tinker primitives + GRPO → single-step training + simple
  #2: verl GRPO + bypass_mode → good but heuristic trust region
  #2.5: DeepSpeed ZeRO-2 + CPU_Adam → solid but needs RouterReplay for CUDA graph
  #3: rLLM Tinker → BLOCKED by #605 (GRPO grouping bug)!
  #4: Megatron core → powerful but complex setup
```

---

## 2. Trust Region Comparison — Detailed

```
★★★★★★★★★ Trust region quality comparison:

Standard PPO:
  → Heuristic ratio clip: |ρ_t - 1| <= ε
  → Uniform, position-independent → ALL tokens constrained equally
  → Does NOT consider prefix drift → cascading divergence risk
  → ε = 0.2 typical → conservative → can prevent good updates too

GRPO:
  → No explicit trust region → relies on group-relative advantage normalization
  → Group normalization implicitly constrains → but NOT a principled trust region
  → Same ratio clip as PPO → when used → uniform → same issues
  → For short responses: group normalization works well → sufficient
  → For long responses (>4k): prefix drift risk → CPPO better

DPPO (verl divergence-based PPO):
  → Binary-TV divergence: D_t = |π(y_t|s_t) - μ(y_t|s_t)|
  → Principled divergence threshold: D_t <= δ
  → BUT: uniform, pointwise, per-token → does NOT consider prefix accumulation
  → Better than PPO heuristic → but still misses cascading drift

CPPO (verl #6731):
  → Position-weighted threshold: w_t = w_min + (1-w_min)*(T-t)/(T-1)
  → → Early tokens constrained TIGHTER → prevents cascading drift from the start!
  → Cumulative prefix budget: c_t = min(δ, δ + δ_b*W_{t-1} - S_{t-1})
  → → As prefix diverges → threshold shrinks → dynamic tightening!
  → ★★★★★★★★ Can catch tokens that satisfy Z_t <= δ yet exceed Z_t > c_t!
  → → PPO/GRPO/DPPO ALL miss these → CPPO catches them → better stability!
  → Theorem 1: provably tighter policy-improvement bound than PPO/GRPO/DPPO!
  → ★★★★★★★★ Near-zero compute/memory overhead → no cost to use!

★★★★★★★★★ Trust region quality ranking:
  CPPO > DPPO > PPO > GRPO (no explicit trust region)
  CPPO = position-weighted + cumulative prefix → MOST principled
  DPPO = pointwise divergence → principled but missing prefix accumulation
  PPO = heuristic ratio clip → simple but unprincipled
  GRPO = group normalization only → implicit, weakest explicit constraint
```

---

## 3. Framework-Specific GRPO Implementation Details

```
★★★★★★★★★ verl GRPO implementation:

Algorithm flow (HYBRID mode):
  → Rollout: vLLM/SGLang generates n=8 responses per prompt (same batch)
  → Sleep: GPU memory freed → only actor weights kept
  → Reward: external reward model or rule-based (GSM8K)
  → Advantage: GRPO group-relative normalization across n=8 responses
  → Update: CPPO/GRPO loss → actor parameters updated
  → Wake: GPU memory restored → next iteration

Key configs for RTX 4090:
  → algorithm.adv_estimator: grpo (no critic)
  → algorithm.rollout_correction.bypass_mode: True (no ref model)
  → actor.policy_loss.loss_mode: cppo (better trust region)
  → actor.clip_ratio: 0.15 (dense model)
  → actor.policy_loss.cppo_w_min: 0.8
  → actor.policy_loss.cppo_delta_b: 0.02
  → ★★★★★★★★ CPPO + bypass_mode MUST use sync TransferQueue trainer (NOT async Ray!)

★★★★★★★★★ rLLM GRPO implementation:

Algorithm flow (Tinker mode):
  → Single-step training → no multi-iteration → simpler
  → Step merge: MergedSegment (#576) → groups trajectories by task
  → Advantage: group-relative normalization → same formula as verl
  → ★★★★★★★★ BUG #605: groups by task_id:trajectory.name → group size = 1 → BROKEN!
  → → Fix: change grouping key from task_id:name to task_id (1 line)
  → → Without fix: rLLM GRPO = REINFORCE (no variance reduction)

★★★★★★★★★ DeepSpeed GRPO implementation:

Algorithm flow (ZeRO-2):
  → Uses standard PPO loss (no CPPO/GRPO built-in)
  → Can implement GRPO advantage manually → group-relative normalization
  → Needs: ZeRO-2 + CPU_Adam + freeze_moe_router
  → For MoE: AutoEP EP=1 → but LoRA gap for 3D expert tensors
  → ★★★★★★★★ DeepSpeed does NOT have built-in GRPO loss → need custom implementation!

★★★★★★★★★ Megatron-LM GRPO:

Algorithm flow:
  → Uses verl orchestration → Megatron as training backend
  → verl #6791: Megatron Lite backend → DSv4/GLM5/KimiK2.5
  → RouterReplay needed for MoE CUDA graph stability
  → DSA indexer replay needed for DSV4 (#5384 feature request)
  → ★★★★★★★★ Megatron = training backend → verl = orchestration → combined = full stack
```

---

## 4. Memory Budget Comparison — RTX 4090 24 GiB

```
★★★★★★★★★ RTX 4090 memory budget for GRPO training (1-4B dense model):

| Component | verl HYBRID+bypass+CPPO | DeepSpeed ZeRO-2 | rLLM Tinker |
|-----------|------------------------|-------------------|-------------|
| Actor weights (bf16) | ~2-4 GB | ~2-4 GB (CPU offload 50%) | ~2-4 GB |
| Ref model | 0 (bypass!) | 0 (bypass or KL=0) | 0 (single-step) |
| Critic | 0 (GRPO) | 0 (manual GRPO) | 0 (group-relative) |
| LoRA params | ~50-300 MB | ~50-300 MB | ~50-300 MB |
| Optimizer states | ~1-2 GB (FSDP) | ~0.5-1 GB (CPU_Adam) | ~1-2 GB |
| Activations | ~2-4 GB | ~2-4 GB | ~2-4 GB |
| KV cache (rollout) | ~1-2 GB | ~1-2 GB | ~1-2 GB |
| CPPO overhead | ~0.1 GB (near-zero) | N/A | N/A |
| Total estimate | ~6-12 GB | ~5-9 GB | ~6-12 GB |
| Margin from 24 GB | ~12-18 GB ★★★★★ | ~15-19 GB ★★★★★ | ~12-18 GB |

★★★★★★★★★ All 3 frameworks fit in 24 GiB for 1-4B dense models!
  → verl CPPO+bypass: BEST trust region + simplest config + most margin
  → DeepSpeed ZeRO-2: MOST CPU offload flexibility + 0 ref model
  → rLLM Tinker: simplest training loop BUT GRPO BROKEN (#605)!
```

---

## Key Findings Summary

★★★★★★★★★ CPPO = RTX 4090 #1 BEST: position-weighted + cumulative prefix → provably tighter bound
★★★★★★★★★ CPPO+GRPO advantage+bypass_mode = optimal RTX 4090 combination (no ref, no critic, best trust region)
★★★★★★★★★ rLLM GRPO BROKEN (#605): group size=1 → no variance reduction → must fix before training
★★★★★★★★★ DeepSpeed needs custom GRPO loss (not built-in) + Grouped LoRA for MoE (3D tensor gap)
★★★★★★★★★ All 3 frameworks fit 1-4B dense models in 24 GiB → margin 12-18 GB
★★★★★★★★★ CPPO overhead = near-zero (0.1 GB, cumsum/quantile) → always safe to use instead of GRPO
★★★★★★★★★ CPPO MUST use sync TransferQueue trainer (async overrides loss_mode → destroys mask!)

---

## References

- CPPO algorithm: notebook/projects/verl-cppo-algorithm-reading.md
- GRPO training flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- GRPO runbook: notebook/projects/rtx4090-grpo-training-runbook.md
- rLLM #605 bug: notebook/projects/rllm-605-grpo-grouping-bug-source-reading.md
- AutoEP LoRA gap: notebook/projects/deepspeed-autoep-lora-gap-3d-tensor-reading.md
- DeepSpeed ZeRO source: notebook/projects/deepspeed-zero-single-gpu-source-reading.md
- Cross-framework Muon: notebook/projects/cross-framework-muon-optimizer-status-synthesis.md
