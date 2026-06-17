# verl Latest Developments — June 2026 Comprehensive Reading

> 2026-06-18 | verl-project/verl | v0.8.0 + 21+ post-release PRs
> Core theme: verl has become the most complete RL training framework in OSS -- 14 advantage estimators, 12 policy loss types, 3 trainer backends, 3 rollout engines, 3 RolloutMode types
> ★★★★★★★★ CPPO + bypass_mode + GRPO = RTX 4090 optimal trust region + zero staleness + near-zero overhead

---

## Table of Contents

1. [v0.8.0 Release (June 1) -- Feature Map](#1-v080-release-june-1)
2. [CPPO #6731 (OPEN) -- Position-Weighted Cumulative Prefix Divergence](#2-cppo-6731-open)
3. [Off-Policy Staleness #6736 + #6778 (MERGED June 16)](#3-off-policy-staleness-6736--6778-merged-june-16)
4. [Megatron LoRA Export #6713 (DRAFT)](#4-megatron-lora-export-6713-draft)
5. [SGLang PD Disaggregation #6117 (MERGED May 6)](#5-sglang-pd-disaggregation-6117-merged-may-6)
6. [Full Determinism #6572 (OPEN)](#6-full-determinism-6572-open)
7. [Algorithm Catalog -- 14 Advantage Estimators + 12 Policy Loss Types](#7-algorithm-catalog)
8. [HYBRID RolloutMode -- RTX 4090 Optimal Architecture](#8-hybrid-rolloutmode--rtx-4090-optimal-architecture)
9. [bypass_mode Deep Dive -- 18Psi to 3.8Psi](#9-bypass-mode-deep-dive)
10. [Complete 11-Step GRPO Flow on RTX 4090](#10-complete-11-step-grpo-flow-on-rtx-4090)
11. [Memory Timeline and Budget](#11-memory-timeline-and-budget)
12. [Off-Policy on Single GPU -- HYBRID ONLY Choice](#12-off-policy-on-single-gpu--hybrid-only-choice)
13. [RTX 4090 Action Items](#13-rtx-4090-action-items)
14. [References](#references)

---

## 1. v0.8.0 Release (June 1) -- Feature Map

```
★★★★★★★★★ verl v0.8.0 released 2026-06-01 -- 5 core additions + multiple bug fixes:

Feature                          | PR      | Status      | RTX 4090 Impact
-------------------------------- | ------- | ----------- | ----------------
On-Policy Distillation (OPD)     | #5041   | MERGED      | ★★★ FSDP2 multi-GPU only
TransferQueue Sync Trainer       | #5401   | MERGED      | ★★★★★ +49.2% e2e (128 GPU)
VeOmni Engine                    | #6072   | MERGED      | ★★★ FSDP2-based, external pkg
SGLang LoRA Support              | #5564   | MERGED      | ★★★★ merge=True path ready
SGLang PD Disaggregation         | #6117   | MERGED      | ★★★★ -6.8% wall-clock (H100)
Unify Trainer ABC                | #6710   | MERGED      | ★★★★★ sync/colocate_async/separate_async
Detach Metrics Fix               | #6699   | MERGED      | ★★★★★★★★ RTX 4090生死级重要!
ROCm/HIP Platform                | #6702   | MERGED      | ✗ NVIDIA GPU only
Gemma4 Multimodal GRPO           | #6715   | MERGED      | ★★★ VLM RL with SDPA
ReMax Support                    | #6340   | MERGED      | ★★★★★ greedy baseline, lowest variance
IcePop IS Correction             | #5722   | MERGED      | ★★★★★ exact population IS bounds

★★★★★★★★★ v0.8.0 BREAKING CHANGES:
  → main_ppo.py deprecated → main_ppo_sync.py recommended
  → Legacy FSDP/Megatron workers deprecated → unified engine abstraction
  → verl/interactions removed
  → Diffusion RL migrated to verl-omni repo
  → trainer.v1.backend: "sync" | "colocate_async" | "separate_async"

★★★★★★★★★ Detach Fix = RTX 4090生死级重要:
  → LoRA + GRPO + long sequences → undetached .item() → +0.27GiB/micro-batch → progressive OOM
  → Fix: .detach().item() → peak VRAM constant at 16.2GiB → RTX 4090 24GB feasible!
  → Without this fix → RTX 4090 LoRA GRPO = guaranteed OOM → MUST use v0.8.0+
```

---

## 2. CPPO #6731 (OPEN) -- Position-Weighted Cumulative Prefix Divergence

```
★★★★★★★★★ CPPO = Clipped Position-weighted Policy Optimization (Binary-TV variant)
  → Paper: arXiv:2606.10968 "Beyond Uniform Token-Level Trust Region in LLM RL"
  → Authors: Tencent Hunyuan (Mao, Zhou, Tao, Ding, Shi, Lin, Wu, Zhu, Qiu, Zhu)
  → PR #6731: OPEN (not yet merged), +544 lines, 8 files changed
  → 12th registered policy loss type (when merged)

★★★★★★★★★ Three improvements over DPPO/GRPO/PPO-clip:
  1. Position-weighted threshold: w_t = w_min + (1-w_min)*(T-t)/(T-1)
     → Early tokens constrained tighter → Theorem 1: theoretically optimal
     → Default w_min = 0.8 → position weight ranges [0.8, 1.0]
  2. Cumulative prefix budget: c_t = min(delta, delta + delta_b * W_{t-1} - S_{t-1})
     → As prefix diverges → threshold shrinks → dynamic tightening
     → CPPO catches "budget overruns" that PPO/GRPO/DPPO all miss!
  3. Per-sequence adaptive budget: delta_b^seq = clamp(k * P90(D_t), delta_b, 2*delta_b)
     → Each sequence calibrates from its own divergence statistics → adaptive

★★★★★★★★★ Keep decision (two-clause mask):
  keep token t iff  A_t*(rho_t - 1) <= 0  OR  Z_t <= c_t

  Clause 1: A_t*(rho_t-1) <= 0 → always keep "safe" updates (toward mu)
  Clause 2: Z_t <= c_t → keep if within effective budget → dynamic check

★★★★★★★★★ CPPO + bypass_mode = mathematically REQUIRED pairing -- NOT optional!

  CPPO divergence: D_t = |pi(y_t|s_t) - mu(y_t|s_t)|
  → Measured against mu (rollout policy), NOT pi_old

  Without bypass_mode:
    → old_log_prob from separate forward → pi_old ≠ mu
    → CPPO divergence = |pi - pi_old| ≠ |pi - mu| → INCORRECT!
    → Budget calibrated against WRONG divergence → position weighting meaningless

  With bypass_mode:
    → old_log_prob = rollout_log_prob = mu
    → CPPO divergence = |pi - mu| → CORRECT!
    → Budget calibrated against true divergence → position weighting meaningful

★★★★★★★★★ CPPO + bypass_mode + GRPO advantage = RTX 4090 optimal combination:
  → algorithm.adv_estimator: grpo → no critic needed
  → algorithm.rollout_correction.bypass_mode: True → no ref model, save ~14GB
  → actor.policy_loss.loss_mode: cppo → best trust region
  → Near-zero compute/memory overhead → cumsum/quantile trivial
  → Same VRAM savings as GRPO + bypass_mode → but provably better trust region!

Config defaults:
  actor.clip_ratio: 0.15 (dense), 0.20 (MoE)
  actor.policy_loss.cppo_w_min: 0.8
  actor.policy_loss.cppo_delta_b: 0.02
  actor.policy_loss.cppo_delta_b_q: 0.9
  actor.policy_loss.cppo_delta_b_k: 1.0
```

---

## 3. Off-Policy Staleness #6736 + #6778 (MERGED June 16)

```
★★★★★★★★★ PR #6736 MERGED 2026-06-16: Off-Policy Metrics + Replay Buffer Staleness
★★★★★★★★★ PR #6778 MERGED 2026-06-16: Staleness Control (drop/wait strategies)

★★★★★★★★★ New metrics (from #6736):

  trajectory_spans = max_global_steps - min_global_steps + 1
  → How many distinct model versions a single trajectory spans
  → trajectory_spans = 1 → fully on-policy (all tokens from same model version)
  → trajectory_spans > 1 → off-policy (tokens from different model versions)

  trajectory_staleness = (global_steps - 1) - max_global_steps  # lower bound
  trajectory_staleness_worst = (global_steps - 1) - min_global_steps  # worst case
  → How many training steps the trajectory lags behind current policy

  Logged per training step:
    → training/off_policy/trajectory_spans/mean
    → training/off_policy/trajectory_spans/max
    → training/off_policy/trajectory_staleness/mean
    → training/off_policy/trajectory_staleness_worst/mean

★★★★★★★★★ Replay buffer fix (from #6736):
  Before: arbitrary sampling order → stale trajectories sampled as often as fresh
  After: prioritize oldest prompts (smallest global_steps) → oldest first → reduces staleness

★★★★★★★★★ Staleness control (from #6778): drop/wait strategies
  → drop: discard trajectories exceeding staleness threshold → zero staleness tolerance
  → wait: hold trajectory until staleness acceptable → delayed processing
  → Configurable thresholds → balance on-policy purity vs throughput
  → 3 granularity levels: trajectory-level, token-level, group-level

★★★★★★★★★ OPD terminology clarified: "one-policy distillation" → "on-policy distillation"
  → Aligns with DeepSpeed OPSD #8027 terminology

★★★★★★★★★ RTX 4090 impact:
  → HYBRID mode = on-policy by design → trajectory_spans = 1 → zero staleness
  → off_policy DOES NOT make sense on single GPU → HYBRID ONLY choice
  → Monitoring: enable trajectory_spans/staleness metrics → validate on-policy purity
  → Combine verl monitoring (#6736) with rLLM staleness control (SyncCoordinator) → optimal async GRPO
```

---

## 4. Megatron LoRA Export #6713 (DRAFT)

```
★★★★★★★★★ CRITICAL CORRECTION: PR #6713 is a verl PR, NOT Megatron-LM

★★★★★★★★★ PR #6713: Adapter-only Megatron LoRA tensor export → vLLM format for rollout
  → Two new functions in megatron_peft_utils.py:
    1. gather_ep_lora_adapter_weights_for_vllm() -- EP rank gather + local→global ID remap
    2. pack_3d_moe_lora_adapter_weights_for_vllm() -- Qwen3-Omni 3D MoE LoRA packing
  → Targets Qwen3-Omni 3D MoE architecture specifically
  → 3-repo coordinated: verl #6713 + verl-omni #169 + vllm-omni #4388 (all DRAFT, same author)

★★★★★★★★★ EP Gather mechanism:
  → When ep_size > 1: all_gather across EP ranks → remap local→global expert IDs
  → global_id = ep_rank * local_expert_count + local_id
  → ★★★★★★★★ RTX 4090 EP=1: passthrough → zero overhead, no all-gather needed!

★★★★★★★★★ 3D MoE Pack mechanism:
  → For Qwen3-Omni model type only
  → Validates: gate_proj and up_proj share same lora_A (architecture requirement)
  → Packs into 4 tensors per expert group → vLLM 3D MoE layout format
  → Standard MoE (Qwen3-30B-A3B) uses PEFT path directly, no 3D packing needed

★★★★★★★★★ Critical review issue (from Gemini Code Assist):
  → decoder_input.requires_grad_(True) on non-leaf tensor → runtime error!
  → Recommendation: detach first before requires_grad_(True)
  → Environment toggle: VERL_MEGATRON_LORA_RECOMPUTE_INPUT_GRAD (default "1")

★★★★★★★★★ RTX 4090 implications:
  → Direct training: NOT viable → verl Megatron backend requires multi-GPU + Megatron-Bridge
  → Core Megatron crashes on single GPU (#5203)
  → Indirect serving: YES viable → train on multi-GPU → export adapter-only LoRA → deploy on RTX 4090 vLLM
  → ★★★★★★★★ Bridges: multi-GPU training → single-GPU serving
  → BEST RTX 4090 training paths remain: DeepSpeed ZeRO-2 + LoRA or rLLM Tinker

★★★★★★★★★ Test coverage:
  → 5 test functions in tests/utils/test_megatron_lora_export_on_cpu.py
  → EP=1 passthrough test explicitly validates RTX 4090 scenario!
```

---

## 5. SGLang PD Disaggregation #6117 (MERGED May 6)

```
★★★★★★★★★ SGLang PD = Prefill-Decode disaggregated rollout architecture
  → PR #6117 MERGED → 1P:ND topology → 1 Prefill server + N Decode servers per replica
  → Asymmetric TP: prefill_tp ≠ decode_tp → configurable different parallelism

★★★★★★★★★ SGLang PD Benchmarks (Qwen2.5-7B, GSM8K, n=8, response_length=2048):
  Single-node 1x8 H100, PD 1P+3D:    11.41 s/step → ★★★★★ -6.8% vs colocated!
  Multi-node 2x8 H100, PD:            11.78 s/step → ★★★★ -4.75% vs colocated

★★★★★★★★★ vLLM PD Benchmarks (PR #6243, still OPEN):
  4-card 0.6B (NIXL):   16.42 s/step → ★★ +17.49% WORSE than colocated!
  8-card 7B (NIXL):     21.22 s/step → ★★ +18.82% WORSE than colocated!
  8-card 7B (Mooncake): 20.13 s/step → ★★★ +10.83% WORSE than colocated!

★★★★★★★★★ SGLang PD > vLLM PD → SGLang PD = verl PD only recommendation for production

★★★★★★★★★ Transfer backends:
  NIXL:      NVIDIA NVLink → highest throughput → RTX 4090 ✗ (no NVLink)
  Mooncake:  RDMA → ~5% slower than NIXL → RTX 4090 ✗ (no RDMA)
  Ascend:    Huawei HCCL → NPU-only → RTX 4090 ✗

★★★★★★★★★ RTX 4090 PD feasibility = ✗✗✗ NOT feasible:
  → No NVLink → NIXL unavailable → KV transfer cannot cross GPU
  → No RDMA → Mooncake unavailable → remote KV transfer impossible
  → PD needs inter-GPU KV transfer → RTX 4090 single GPU → no transfer needed → no PD!

★★★★★★★★★ RTX 4090 viable paths:
  → Colocated vLLM: enforce_eager=True → SM89 batch invariance → throughput -10-15%
  → ★★★★★★ Colocated SGLang: --enable-deterministic-inference → no throughput loss!
    → BUT: needs testing on SM89 → unvalidated → potential RTX 4090 biggest improvement!
  → ★★★★★★★ In-process (rLLM Tinker): not affected by SM89 bugs → simplest

★★★★★★★★★ SGLang deterministic inference = SM89 most attractive alternative:
  → --enable-deterministic-inference → batch-invariant operators → Thinking Machines Lab
  → No need for enforce_eager → no need to disable compile → deterministic by design!
  → sampling_seed → reproducible non-greedy sampling → GRPO critical!
  → Supported backends: FlashInfer, FA3, Triton
  → ★★★ FlashInfer does NOT support radix cache in deterministic mode → tradeoff!
```

---

## 6. Full Determinism #6572 (OPEN)

```
★★★★★★★★★ PR #6572: Full determinism for vLLM rollout → bitwise-aligned reward curves

  Goal: bitwise-aligned reward curves across training steps
  → Deterministic inference → reproducible log_probs → same input = same output
  → Critical for GRPO: reward computation must be deterministic for fair advantage estimation
  → Batch-invariant logprobs → different batch sizes produce identical per-token log_probs

★★★★★★★★★ Why determinism matters for GRPO:
  → GRPO advantage = (r_i - mu_group) / sigma_group → group-relative
  → If log_probs differ across batches → reward estimation inconsistent → advantage biased
  → Non-deterministic inference → different numerical results → reward noise → training instability
  → ★★★★★★★★ Bitwise-aligned reward curves → remove reward noise → cleaner training signal

★★★★★★★★★ RTX 4090 determinism hierarchy (KERNEL > COMPILE > NONE):
  1. KERNEL-level: SGLang deterministic inference → 7 aten overrides constexpr → gold standard
     → murmur_hash32 Gumbel-max float64 sampling → GRPO best
  2. COMPILE-level: vLLM enforce_eager=True → disable Inductor → batch-invariant
     → But -10-15% throughput penalty → SM89 hit hardest
  3. NONE-level: Standard inference → batch-dependent → NOT batch-invariant
     → Inductor 3-layer fusion gate → RMSNorm fusion root cause (#39096)

★★★★★★★★★ Connection to Inductor SM89 Fusion Guard (P9):
  → #39096: SM89 batch invariance STILL UNFIXED → Inductor root cause confirmed
  → #187275: PR OPEN → combo kernel dynamic reduction fix → progressing
  → Fusion Guard PR draft ready → 5-line choices.py → props.major<9 → WhyNoFuse
  → verl #6572 + PyTorch Fusion Guard = complementary → both needed for RTX 4090 GRPO

★★★★★★★★★ RTX 4090 recommendation:
  → verl HYBRID + vLLM + enforce_eager=True → COMPILE-level → works but -10-15%
  → verl HYBRID + SGLang + deterministic_inference → KERNEL-level → best if validated
  → rLLM Tinker → in-process → not affected by batch invariance → simplest path
```

---

## 7. Algorithm Catalog -- 14 Advantage Estimators + 12 Policy Loss Types

```
★★★★★★★★★ 14 Advantage Estimators (registered via @register_adv_est):

 #  | Name                    | Formula/Core                              | Critic? | RTX 4090?
--- | ----------------------- | ----------------------------------------- | ------- | ----------
 1  | GAE                     | A_t = Σ(γλ)^l δ_{t+l}                    | YES     | ✗ (needs critic)
 2  | GRPO                    | (r_i - μ_group) / σ_group                | NO      | ★★★★★★★ standard
 3  | GRPO_VECTORIZED         | torch-level group ops                    | NO      | ★★★★★★★ fastest
 4  | GDPO                    | Per-dimension normalize then sum         | NO      | ★★★★ multi-reward
 5  | GRPO_PASSK              | top-k samples only → pass@k oriented    | NO      | ★★★ pass@k eval
 6  | REMAX                   | r_i - greedy_baseline_reward             | NO      | ★★★★★★★ lowest var
 7  | GPG                     | Direct ∇logpi * A (REINFORCE)           | NO      | ★★ basic
 8  | REINFORCE_PLUS_PLUS     | Reward baseline normalization            | NO      | ★★ simple
 9  | REINFORCE_PLUS_PLUS_BL  | Reward + learned baseline                | YES     | ✗ (needs critic)
10  | RLOO                    | Leave-One-Out (group mean excluding i)   | NO      | ★★★ low var
11  | RLOO_VECTORIZED         | Vectorized RLOO                          | NO      | ★★★★ fast RLOO
12  | OPO                     | Online Policy Optimization               | NO      | ★★ experimental
13  | OPTIMAL_TOKEN_BASELINE  | Token-level optimal baseline (theory)    | NO      | ★★★ theoretical
14  | TIR_OPTIMAL_TOKEN_BL    | Trust-region optimized token baseline    | NO      | ★★★ theoretical

★★★★★★★★★ 12 Policy Loss Types (registered via @register_policy_loss):

 #  | Name          | Trust Region                   | Position-Aware? | Prefix-Aware? | RTX 4090?
--- | ------------- | ------------------------------ | --------------- | ------------- | ----------
 1  | vanilla       | |ρ-1| <= ε (PPO-clip)          | NO              | NO            | ★★★★★★★★
 2  | dppo_tv       | D_t <= δ (Binary-TV)           | NO              | NO            | ★★★★★ principled
 3  | dppo_kl       | KL_t <= δ                       | NO              | NO            | ★★★★★ KL-based
 4  | gspo          | Sequence-level clip             | NO              | NO            | ★★★ experimental
 5  | sapo          | Smooth advantage                | NO              | NO            | ★★★ experimental
 6  | gpg           | Direct gradient (REINFORCE)     | NO              | NO            | ★★ basic
 7  | clip_cov      | Clipped covariance              | NO              | NO            | ★★★★ cov-aware
 8  | kl_cov        | KL + covariance                 | NO              | NO            | ★★★★ KL cov
 9  | geo_mean      | Geometric mean ratio            | NO              | NO            | ★★★ experimental
10  | cispo         | Exact population IS [0.5, 5.0]  | NO              | NO            | ★★★★★★★ most precise
11  | bypass_mode   | Dispatcher → ppo_clip/reinforce | NO              | NO            | ★★★★★★★★★ RTX 4090 MUST!
12  | cppo          | w_t*D_t <= c_t (cumulative)     | YES             | YES           | ★★★★★★★★★★ BEST bound! (OPEN #6731)

★★★★★★★★★ RTX 4090 BEST combinations:
  1. bypass_mode + loss_type="ppo_clip" + GRPO advantage → ★★★★★★★★★★★★★ simplest, safe
  2. bypass_mode + loss_type="reinforce" + IcePop + GRPO → ★★★★★★★★★★★★★ most precise gradient
  3. bypass_mode + CPPO + GRPO advantage → ★★★★★★★★★★★★★★★★★★★★★★★★★ best trust region bound!

★★★★★★★★★ IcePop (CISPO) = exact population IS correction:
  → IcePop zeros out-of-range IS weights to EXACTLY zero (not clips!)
  → Threshold format: "0.5_5.0" → lower and upper bounds → exact population
  → TIS (standard): clips high weights to threshold → biased, reduces variance
  → IcePop: within bounds = EXACT weight, outside = zero → unbiased within bounds
  → Preset: bypass_pg_token_icepop → bypass_mode=True + reinforce + token-level IcePop
```

---

## 8. HYBRID RolloutMode -- RTX 4090 Optimal Architecture

```
★★★★★★★★★ verl supports 3 RolloutMode values (replica.py:54-68):

HYBRID (★★★★★★★★★ RTX 4090 OPTIMAL):
  → Rollout + training share SAME Ray actor/process per GPU
  → FusedWorker wrapper combines actor+rollout into one process
  → Sleep/Wake GPU memory management:
    → sleep_level=2: full weight sync (releases KV cache + weights + GPU memory)
    → sleep_level=1: LoRA mode (only releases KV cache; base weights remain)
  → Weight sync: naive (in-process Python generator) → ZERO-COPY
  → ★★★★★★★★ BEST for: on-policy PPO/GRPO, single GPU RTX 4090, LoRA training
  → ★★★★★★★★ trajectory_spans = 1 → zero staleness → on-policy by design

COLOCATED:
  → Rollout + training share same Ray Placement Group (same GPU) but DIFFERENT processes
  → CheckpointEngineWorker as independent Ray actor per GPU
  → Weight sync: CUDA IPC (ZMQ + torch.multiprocessing) → GPU-direct tensor transfer
  → ★★★★★ BEST for: GRM (LLM-as-judge), MoE models needing separate rollout processes

STANDALONE:
  → Rollout has own ResourcePool → independent GPU set
  → No Sleep/Wake needed → dedicated GPU memory
  → Weight sync: NCCL ProcessGroup or NIXL RDMA
  → ★★★★★ BEST for: off-policy training, disaggregated architecture, multi-GPU clusters
  → NOT feasible on single RTX 4090 (needs 2+ GPU sets)

★★★★★★★★★ HYBRID = RTX 4090 ONLY choice:
  → In-process → zero IPC overhead → no Ray actor communication latency
  → Zero-copy weight sync → Python generator → no CUDA IPC
  → Sleep/Wake memory management → GPU time-multiplexing → rollout + training share 24GB
  → On-policy by design → trajectory_spans = 1 → no staleness issues
  → ★★★★★★★★ off_policy DOES NOT make sense on single GPU → HYBRID ONLY choice!
```

---

## 9. bypass_mode Deep Dive -- 18Psi to 3.8Psi

```
★★★★★★★★★ Standard PPO memory (18Psi):
  → Actor parameters: 2Psi (bf16)
  → Actor gradients: 2Psi (bf16)
  → Optimizer states: 14Psi (fp32) → momentum + variance + master params
  → Reference model: 2Psi (bf16) → KL penalty
  → Total: 18Psi → DOES NOT fit RTX 4090 24GB for 7B

★★★★★★★★★ bypass_mode + LoRA memory (3.8Psi):
  → Frozen base parameters: 2Psi (bf16) → LoRA base weights, not trainable
  → LoRA gradients: ~0.6Psi (bf16) → only adapter params trainable
  → LoRA optimizer states: ~1.2Psi (fp32) → momentum + variance for LoRA only
  → No reference model: 0 → eliminated by bypass_mode!
  → Total: ~3.8Psi → fits RTX 4090 24GB easily

★★★★★★★★★ How bypass_mode eliminates the reference model:
  → apply_bypass_mode(batch, rollout_corr_config, policy_loss_config):
  → batch["old_log_probs"] = batch["rollout_log_probs"] ← zero-cost substitution!
  → Policy reduction: 3 policies → 2 policies → pi_old eliminated
  → ★★★★★★★★ On-policy (HYBRID same GPU): pi_rollout = pi_theta → IS ratio = 1.0 → NO correction needed

★★★★★★★★★ bypass_mode loss types:
  → ppo_clip: PPO clipped → ratio = pi_current/pi_rollout → IS correction natural
  → reinforce: REINFORCE → explicit IS weights w = pi_current/pi_rollout
  → ★★★★★★★★ GRPO recommended: ppo_clip + on-policy → IS ratio always 1.0

★★★★★★★★★ bypass_mode is NOT a separate loss -- it's a DISPATCHER!
  → Computes IS weights and rejection mask, then dispatches to:
    → loss_type="ppo_clip" → compute_policy_loss_vanilla (NO IS weights)
    → loss_type="reinforce" → compute_policy_loss_reinforce (explicit IS weights)
  → ★★★★★★★★ PPO-clip bypass = NO IS weights (clipping handles ratio)
    → REINFORCE bypass = explicit IS weights → DIFFERENT semantics → choose carefully!

★★★★★★★★★ CPPO + bypass_mode: TWO configs working together:
  → bypass_mode (RolloutCorrectionConfig): Sets old_log_probs = rollout_log_probs at BATCH level
  → CPPO (policy_loss.loss_mode): Registered as separate @register_policy_loss("cppo")
  → CPPO reads mu directly from old_log_prob → which IS rollout_log_prob thanks to bypass_mode
  → ★★★★★★★★ CPPO does NOT use loss_mode="bypass_mode" → uses loss_mode="cppo" directly!

★★★★★★★★★ Memory comparison:
  GRPO standard (verl):    ~31GB → RTX 4090 ✗ NOT feasible
  GRPO bypass_mode (verl): ~17GB → RTX 4090 ✓ feasible → 7GB headroom!
  GRPO Tinker (rLLM):      ~17GB → RTX 4090 ✓ (bypass by default)
  PPO (GAE):              ~42GB+ → RTX 4090 ✗✗✗ completely unfeasible
```

---

## 10. Complete 11-Step GRPO Flow on RTX 4090

```
★★★★★★★★★ Optimal config: RolloutMode=HYBRID + Weight Sync=naive + GRPO + LoRA + bypass_mode

Step 1: INITIALIZATION (engine_workers.py:500-632)
  → a. Ref model: BYPASS → eliminated entirely (0 memory)
  → b. Actor model: FSDP engine + LoRA + optimizer + LR scheduler → ~0.6Psi trainable
  → c. Rollout engine: vLLM ServerAdapter + 3D DeviceMesh(dp=1, infer_tp=1, infer_pp=1)
  → d. Checkpoint engine: naive (in-process Python generator → zero-copy)
  → e. aggressive_empty_cache(force_sync=True) → free GPU memory for vLLM KV cache

Step 2: ROLLOUT PHASE
  → rollout.wake_up(tags=["kv_cache", "weights"]) → vLLM occupies full GPU
  → reset_prefix_cache() → clear stale KV from previous step
  → generate_sequences() via vLLM async server
  → ★★★★★★★★ GRPO n=8: same prompt → 8 responses → sticky session → prefix caching
  → calculate_log_probs=True (bypass_mode requirement)
  → collect: input_ids + responses + rollout_log_probs

Step 3: SLEEP PHASE
  → rollout.sleep(level=1) → LoRA mode: only release KV cache
  → vLLM releases KV cache blocks → ~2-4GB freed
  → Base weights remain on GPU (LoRA mode)
  → GPU memory now available for FSDP training

Step 4: DATA ASSEMBLY
  → batch.repeat(rollout_n=8, interleave=True) → 8 responses per prompt
  → compute_response_mask() → mark valid response tokens
  → balance_batch() → equalize valid token counts across DP ranks

Step 5: REWARD COMPUTATION
  → extract_reward(batch) → rule-based or model-based
  → Outcome-level reward per response (scalar)

Step 6: BYPASS MODE ACTIVATION
  → apply_bypass_mode():
  → batch["old_log_probs"] = batch["rollout_log_probs"] ← zero-cost!
  → policy_loss_config["loss_mode"] = "bypass_mode"
  → No separate ref model forward pass → saves ~2Psi memory

Step 7: KL PENALTY (optional with bypass_mode)
  → If use_kl_loss=True (GRPO recommended):
  → KL divergence during actor forward
  → ★★★★★★★★ With bypass_mode: KL penalty = 0 → π_theta vs π_theta → zero!
  → CPPO provides its own trust region → makes explicit KL penalty redundant!

Step 8: ADVANTAGE COMPUTATION
  → compute_advantage(batch, adv_estimator="grpo"):
  → Group normalization: A_i = (r_i - mean_group) / (std_group + epsilon)
  → Outcome-only: scores = rewards.sum() → per-sequence scalar
  → ★★★★★★★★ Singleton fallback (n=1): mean=0, std=1 → advantage=raw reward

Step 9: ACTOR UPDATE
  → actor_rollout_wg.update_actor(data):
  → FSDP forward + backward on LoRA params only
  → compute_policy_loss_bypass_mode() → PPO-clip ratio=pi_theta/pi_rollout
  → OR compute_policy_loss_cppo() → CPPO mask construction (if loss_mode=cppo)
  → clip_grad_norm → optimizer.step (LoRA params only)
  → ~0.6Psi gradients + ~1.2Psi optimizer states

Step 10: WEIGHT SYNC (HYBRID naive)
  → checkpoint_manager.update_weights():
  → actor.engine.get_per_tensor_param() → Python generator of (name, tensor)
  → For LoRA: only sync adapter weights (~2.6GB) → or merge LoRA then sync full
  → rollout.update_weights_from_ipc() → receives weights
  → clear_kv_cache() → reset_prefix_cache()

Step 11: WAKE-UP for next rollout
  → rollout.wake_up(tags=["kv_cache"]) → vLLM restores KV cache allocation
  → Ready for next step's generation
```

---

## 11. Memory Timeline and Budget

```
★★★★★★★★★ Memory budget per training step (RTX 4090, 7B LoRA+GRPO+bypass):

Rollout phase:  ~17GB peak (weights + KV cache + activations)
Sleep phase:    ~12GB (weights remain, KV freed, FSDP uses freed space)
Training phase: ~14GB peak (FSDP forward+backward on LoRA)
Weight sync:    ~2.6GB transfer (LoRA adapter only)
Wake-up:        KV cache reallocated

★★★★★★★★★ Total peak: ~17GB → fits 24GB RTX 4090 with 7GB margin!

★★★★★★★★★ Detailed VRAM budget for CPPO + GRPO + bypass + LoRA-32:

| Component           | Qwen3-4B BF16 | Qwen2.5-7B BF16 | Qwen2.5-1.5B BF16 |
| -------------------- | ------------- | ---------------- | ------------------ |
| Base model           | ~8GB          | ~14GB            | ~3GB               |
| LoRA-32 adapters     | ~0.3GB        | ~0.5GB           | ~0.1GB             |
| Adam (LoRA FP32)     | ~1GB          | ~2GB             | ~0.5GB             |
| KV cache (rollout)   | ~3GB          | ~4GB             | ~1GB               |
| Training buffers     | ~1.5GB        | ~2GB             | ~0.5GB             |
| CPPO mask tensors    | ~0.01GB       | ~0.01GB          | ~0.01GB            |
| **Total**            | **~13.8GB**   | **~22.5GB**      | **~5.1GB**         |
| **Headroom**         | **~10.2GB**   | **~1.5GB**       | **~18.9GB**        |

★★★★★★★★★ Key savings from bypass_mode:
  → No ref model → save ~14GB (7B model)
  → No pi_old forward pass → save ~3-5GB activation memory
  → CPPO mask tensors → negligible (<0.01GB) → D_t, w_t, Z_t, c_t, valid_mask
  → ★★★★★★★★ CPPO near-zero overhead: same VRAM as GRPO+bypass → but better trust region!

★★★★★★★★★ Detach Metrics = RTX 4090防OOM救命补丁:
  → undetached .item() → retains backward graph → +0.27GiB/micro-batch → progressive OOM
  → Fix: .detach().item() → peak VRAM constant at 16.2GiB → saves ~10GiB peak!
  → ★★★★★★★★ MUST enable: detach_metrics_per_micro_batch=True → RTX 4090 GRPO MUST config!
  → Fixed in v0.8.0 → all subsequent versions include this fix

★★★★★★★★★ Weight Sync Backend comparison:
  naive:     Python generator (0-copy) → HYBRID → ★★★★★★★★ Optimal!
  CUDA IPC:  ZMQ + torch IPC (GPU)    → COLOCATED → feasible
  NCCL:      AllReduce/Broadcast       → STANDALONE → not feasible (PCIe)
  NIXL:      RDMA P2P ring             → STANDALONE → not applicable
```

---

## 12. Off-Policy on Single GPU -- HYBRID ONLY Choice

```
★★★★★★★★★ Why off_policy DOES NOT make sense on single GPU:

  STANDALONE mode (off-policy):
    → Rollout on GPU set A → training on GPU set B → need 2+ GPU sets
    → RTX 4090 = single GPU → cannot have 2 GPU sets → STANDALONE impossible!

  COLOCATED mode (off-policy):
    → Same GPU but different processes → still need Ray actors → IPC overhead
    → Staleness: rollout data from step t → training at step t+N → N steps stale
    → trajectory_spans > 1 → off-policy → divergence risk → advantage estimation biased
    → On single GPU → no benefit over HYBRID → same GPU → same memory → but MORE staleness!

  HYBRID mode (on-policy):
    → Same process → same GPU → same model → trajectory_spans = 1 → zero staleness
    → Rollout → sleep → train → wake → sequential → no concurrency needed
    → ★★★★★★★★ On single GPU → sequential is MANDATORY → cannot overlap rollout+training
    → → HYBRID sequential flow = same throughput as COLOCATED/STANDALONE async on single GPU!
    → → BUT: zero staleness vs N-step staleness → HYBRID wins on quality!

★★★★★★★★★ Mathematical argument:
  → On-policy: advantage estimation = exact → no IS correction needed → bypass_mode works perfectly
  → Off-policy: advantage estimation = biased → need IS/RS correction → overhead + instability
  → ★★★★★★★★ On single GPU → off_policy adds overhead WITHOUT throughput benefit!
  → → Cannot overlap rollout+training on single GPU → async = same wall-clock as sync!
  → → → HYBRID on-policy = same throughput + better quality + simpler config = WIN!

★★★★★★★★★ CPPO + bypass + GRPO = RTX 4090 best trust region + zero staleness:
  1. bypass_mode → no ref model → save ~14GB → old_log_probs = rollout_log_probs → zero-cost
  2. GRPO advantage → no critic → save value model memory → group-relative normalization
  3. CPPO trust region → position-weighted cumulative prefix divergence → provably better bound
  4. HYBRID mode → zero staleness → trajectory_spans = 1 → on-policy by design
  5. Near-zero overhead → CPPO mask negligible → same VRAM as GRPO+bypass → but better bound!

★★★★★★★★★ RTX 4090 GRPO Training Ranking (updated 2026-06-18):

  Rank | Algorithm                    | Advantage      | Trust Region      | Memory  | Variance | Staleness
  ---- | ---------------------------- | -------------- | ----------------- | ------- | -------- | ----------
  ★★★★★★★★ #1 | rLLM Tinker                | auto-safe      | bypass default    | ~18GB   | ★★★★★   | zero
  ★★★★★★★★ #2 | verl CPPO+bypass+GRPO      | GRPO group     | prefix-weighted   | ~18GB   | ★★★★    | zero
  ★★★★★★ #2 | verl ReMax+bypass_ppo_clip  | greedy base    | PPO-clip          | ~18GB   | ★★★★★★★★ | zero
  ★★★★★★ #2.5 | verl ReMax+IcePop+bypass   | greedy+IS      | IcePop bounds     | ~18GB   | ★★★★★★★★ | zero
  ★★★★ #3 | verl GRPO+bypass_ppo_clip    | group mean     | PPO-clip          | ~18GB   | ★★★     | zero

★★★★★★★★★ All HYBRID configurations have zero staleness by design!
```

---

## 13. RTX 4090 Action Items

```
★★★★★★★★★ RTX 4090 Action Items -- prioritized by impact:

  ★★★★★★★★ 1. MUST: Use verl >= v0.8.0
     → Detach metrics fix included → LoRA+GRPO+long sequence → RTX 4090 feasible
     → Without this fix → guaranteed progressive OOM

  ★★★★★★★★ 2. MUST: Enable bypass_mode=True
     → algorithm.rollout_correction.bypass_mode: True
     → Saves ~14GB (no ref model) → RTX 4090 from 31GB→17GB → feasible!
     → MUST combine with: rollout.calculate_log_probs: True

  ★★★★★★★★ 3. MUST: Enable detach_metrics_per_micro_batch=True
     → Prevents progressive OOM from undetached autograd graph retention
     → Saves ~10GiB peak VRAM → RTX 4090 24GB critical

  ★★★★★★★★ 4. MUST: Use HYBRID RolloutMode (trainer.v1.backend: sync)
     → In-process → zero IPC → zero staleness → on-policy by design
     → PPOTrainerSync → single GPU optimal

  ★★★★★★★★ 5. MUST: Use LoRA rank=32
     → Only adapter params trainable → ~0.6Psi gradients + ~1.2Psi optimizer
     → Base weights frozen → 3.8Psi total vs 18Psi full training

  ★★★★★★ 6. RECOMMENDED: Use CPPO loss_mode (when #6731 merges)
     → actor.policy_loss.loss_mode: cppo
     → Provably better trust region → near-zero overhead → same VRAM
     → Requires bypass_mode=True → mathematical necessity

  ★★★★★★ 7. RECOMMENDED: Use GRPO_VECTORIZED advantage estimator
     → algorithm.adv_estimator: grpo_vectorized
     → 10-100x faster than loop-based GRPO → GPU-level vectorization

  ★★★★★ 8. RECOMMENDED: Use naive weight sync backend
     → In-process Python generator → zero-copy → optimal for HYBRID single GPU
     → Default for HYBRID mode → no config change needed

  ★★★★★ 9. RECOMMENDED: Use rule-based reward
     → No reward model needed → CPU-side computation → saves GPU memory
     → Math/code → verifiable → deterministic → GRPO best

  ★★★ 10. OPTIONAL: Use enforce_eager=True (vLLM) or deterministic_inference (SGLang)
     → Batch-invariant log_probs → SM89 compatibility
     → vLLM: -10-15% throughput → SGLang: no penalty (if validated on SM89)

  ★★★ 11. OPTIONAL: Use INT4 quantization for base model
     → Base model ~3.5GB → more headroom for KV cache and training buffers
     → merge LoRA → INT4 quantize → vLLM serve → best deployment path

  ★★★ 12. MONITOR: Enable trajectory_spans/staleness metrics
     → Even with HYBRID (zero staleness) → validate on-policy purity
     → Watch: training/off_policy/trajectory_spans/mean → should always be 1.0

  ★★★ 13. MONITOR: Use VLLM_USE_V2_MODEL_RUNNER=0 until verl officially supports MRv2
     → MRv2 still has ZERO explicit handling in verl → potential bugs
     → INT4 = MRv1 → safe → not affected
     → BF16 dense → MRv2 auto → needs verl explicit handling → currently NOT safe

★★★★★★★★★ RTX 4090 optimal config summary:

  trainer.v1.backend: sync
  algorithm.adv_estimator: grpo_vectorized
  algorithm.rollout_correction.bypass_mode: true
  actor_rollout_ref.model.lora.merge: true
  actor_rollout_ref.model.lora.rank: 32
  actor_rollout_ref.actor.policy_loss.loss_mode: cppo  (when #6731 merges)
  actor_rollout_ref.actor.clip_ratio: 0.15
  actor_rollout_ref.actor.policy_loss.cppo_w_min: 0.8
  actor_rollout_ref.actor.policy_loss.cppo_delta_b: 0.02
  actor_rollout_ref.actor.detach_metrics_per_micro_batch: true
  actor_rollout_ref.rollout.calculate_log_probs: true
  transfer_queue.enable: true
  rollout.gpu_memory_utilization: 0.6
  reward_model.enable: false
```

---

## References

### PRs and Issues
- verl v0.8.0 release: https://github.com/verl-project/verl/releases/tag/v0.8.0
- CPPO #6731: https://github.com/verl-project/verl/pull/6731 (OPEN)
- CPPO paper: arXiv:2606.10968 "Beyond Uniform Token-Level Trust Region in LLM RL"
- CPPO project page: https://hunyuan-cppo.github.io
- Off-Policy Metrics #6736: https://github.com/verl-project/verl/pull/6736 (MERGED 2026-06-16)
- Staleness Control #6778: https://github.com/verl-project/verl/pull/6778 (MERGED 2026-06-16)
- Megatron LoRA Export #6713: https://github.com/verl-project/verl/pull/6713 (DRAFT)
- verl-omni #169: https://github.com/verl-project/verl-omni/pull/169 (DRAFT)
- vllm-omni #4388: https://github.com/vllm-project/vllm-omni/pull/4388 (DRAFT)
- SGLang PD #6117: https://github.com/verl-project/verl/pull/6117 (MERGED)
- vLLM PD #6243: https://github.com/verl-project/verl/pull/6243 (OPEN, trails colocated)
- Full Determinism #6572: https://github.com/verl-project/verl/pull/6572 (OPEN)
- Detach Metrics Fix #6699: https://github.com/verl-project/verl/pull/6699 (MERGED)
- Unify Trainer ABC #6710: https://github.com/verl-project/verl/pull/6710 (MERGED)
- TransferQueue #5401: https://github.com/verl-project/verl/pull/5401 (MERGED)
- VeOmni OPD #6072: https://github.com/verl-project/verl/pull/6072 (MERGED)
- ReMax #6340: https://github.com/verl-project/verl/pull/6340 (MERGED)
- IcePop #5722: https://github.com/verl-project/verl/pull/5722 (MERGED)
- Gemma4 #6715: https://github.com/verl-project/verl/pull/6715 (MERGED)
- ROCm #6702: https://github.com/verl-project/verl/pull/6702 (MERGED)
- Tinker Worker Primitives #6717: https://github.com/verl-project/verl/pull/6717 (MERGED)
- SGLang LoRA #5564: https://github.com/verl-project/verl/pull/5564 (MERGED)
- SGLang clone OOM #6733: https://github.com/verl-project/verl/pull/6733 (OPEN)
- vLLM SM89 batch invariance #39096: https://github.com/vllm-project/vllm/issues/39096 (STILL UNFIXED)
- PyTorch #187275: https://github.com/pytorch/pytorch/pull/187275 (OPEN)
- DeepSpeed OPD #8027: https://github.com/deepspeedai/DeepSpeed/pull/8027 (DRAFT OPEN)
- rLLM SyncCoordinator: max_quota=(1+staleness)*trigger*mini_batch

### Source Notes
- verl-cppo-algorithm-reading.md: CPPO algorithm deep analysis
- verl-cppo-bypass-source-reading.md: CPPO + bypass_mode code path trace
- verl-policy-loss-catalog-source-reading.md: 11 registered policy loss types catalog
- verl-off-policy-staleness-metrics-source-reading.md: #6736 staleness metrics analysis
- verl-pr6713-megatron-lora-export-reading.md: Megatron LoRA adapter export
- verl-sglang-pd-disaggregation-reading.md: SGLang PD architecture and benchmarks
- verl-grpo-bypass-mode-reading.md: bypass_mode + TransferQueue + detach metrics
- verl-grpo-core-algos-reading.md: 14 advantage estimators + 10 policy loss types
- verl-remax-icepop-source-reading.md: ReMax and IcePop source analysis
- verl-v0.8-features-reading.md: v0.8.0 complete feature map
- verl-veomni-opd-source-reading.md: VeOmni OPD engine analysis
- verl-tinker-worker-primitives-reading.md: Tinker split training API
- verl-rtx4090-grpo-training-flow.md: Complete 11-step GRPO flow
- rllm-async-trainer-sync-coordinator-source-reading.md: SyncCoordinator staleness control
