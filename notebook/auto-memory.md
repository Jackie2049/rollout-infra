# Auto Memory — AI Infra Engineer Learning Project

> Updated: 2026-06-20

## Project Context
- Working directory: ~/workspace/rollout-infra/
- Goal: Become strong AI infra engineer → senior AI expert
- Current focus: 7 frameworks (DeepSpeed, Megatron, vLLM, verl, MindIE/vLLM-Ascend, rLLM, PyTorch)
- GPU: university + matpool — BOTH OFFLINE. User will provide GPU device later for PR validation.

## Expert Readiness: 42/50 (84%)
- Theory: 14/10, Infra: 9/10, Math→Bug: 8/10, Practical: 7/10, OSS: 4/10

## Key New Findings (June 20)
- PPO-clip OOM on RTX 4090 (65 GiB > 24 GiB) → GRPO = ONLY viable
- PPO-clip signal 0.07 vs GRPO gs=8 signal 0.8 → GRPO 2x faster convergence
- Shaped rewards eliminate degenerate groups (0% vs 4.6%)
- SNR = √gs: gs=8 → 2.83 (sufficient), gs=4 → 2.0 (borderline)
- Ranking-based rewards = perfect decorrelation from task difficulty
- ONLY viable: verl + SGLang + FSDP1 + LoRA r=32 + bypass + naive checkpoint
- gs=1 = REINFORCE — MUST use gs >= 4
- Rollout = 69.2% bottleneck → SGLang #1
- NCCL dp=1 = identity → naive checkpoint better

## MUST DO: ZeRO-2, CPU_Adam, clip_grad=1.0, bypass, group_by_prompt, enforce_eager for DSV4, FSDP1, gs>=4, LoRA+bypass sync, naive ckpt dp=1, record_stream, ulimit 65535, shaped rewards, gs>=8 for sparse, format+outcome reward

## New Tools (June 20)
- weight_sync_timing_simulator, grpo_advantage_numerical_experiment, grpo_training_step_timing_model
- cross_framework_grpo_stack_comparison, pytorch_dist_training_debug_guide
- verl_v1_grpo_config_generator, sglang_sleep_wave_integration_guide
- gpu_pr_validation_experiments.sh (6 GPU experiments ready)
- ppo_vs_grpo_comparison_simulator (4 modes, PPO OOM proof)
- grpo_reward_shaping_analysis (4 modes, shaped vs outcome rewards)

## Tracked Pattern Families
- DSV4: 11 failures → enforce_eager=True MANDATORY
- ZeRO-3: #8072/#8076 → ZeRO-2 ONLY
- Weight Reload: #46125, #28676, #10684, #44395, #28679, #45552
- Silent Corruption: #8061, #8058, #28679, #46118

## GPU Validation Experiments Ready (6)
1. DeepSpeed #8061 overlap_comm NaN
2. DeepSpeed #8068 gradient clipping
3. GRPO singleton degeneration
4. vLLM #46125 encoder cache stale
5. SGLang #28676 MoE cache clobber
6. verl RTX 4090 GRPO full pipeline
