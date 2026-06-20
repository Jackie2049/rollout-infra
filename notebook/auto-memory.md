# Auto Memory — AI Infra Engineer Learning Project

> Updated: 2026-06-20 (Session 2 continued)

## Project Context
- Working directory: ~/workspace/rollout-infra/
- Goal: Become strong AI infra engineer → senior AI expert
- Current focus: 7 frameworks (DeepSpeed, Megatron, vLLM, verl, MindIE/vLLM-Ascend, rLLM, PyTorch)
- GPU: university + matpool — BOTH OFFLINE. User will provide GPU device later for PR validation.

## Expert Readiness: 44/50 (88%)
- Theory: 14/10, Infra: 10/10, Math→Bug: 9/10, Practical: 7/10, OSS: 4/10

## Key Findings (Session 2 — deep readings)
- ZeRO gradient flow: backward → IPG bucket → reduce_scatter → average_tensor → optimizer
- #8061: average_tensor only syncs current_stream, misses producers → #8080: copy_streams set fix
- overlap_comm pointless on dp=1 (NCCL reduce-scatter = identity)
- ZeRO-1/2 both zero savings at dp=1
- verl V1: 10-phase step pipeline with TransferQueue data fabric
- LoRA adapter path: 80x sync payload reduction (~200 MiB vs ~14 GiB)
- bypass + ref_in_actor: 5→1 forward passes, 14 Psi savings
- #6782 root cause: LoRA rank=64 distorts logit distribution → suppresses EOS
- #6699: micro-batch outputs not detached → OOM accumulation

## MUST DO: ZeRO-2, CPU_Adam, clip_grad=1.0, bypass, group_by_prompt, enforce_eager, FSDP1, gs>=4, LoRA+bypass sync, naive ckpt dp=1, record_stream, ulimit 65535, shaped rewards, gs>=8 for sparse, sleep_level=1, SGLang rollout

## MUST NOT: ZeRO-3, Muon, overlap_comm dp=1, sleep_level=2 RTX4090, LoRA rank>=64 vLLM, FSDP2, gs=1, PPO-clip RTX4090, full param sync, NCCL ckpt dp=1, outcome reward gs<16

## CUDA Stream Safety Pattern Family (8 members)
- DeepSpeed: #8061 (CRITICAL), #8080 (fix), #8072 (HIGH), #8075 (latent)
- verl: #6794-CRITICAL-1 (UNFIXED, silent corruption)
- vLLM: #45552 (CRITICAL RTX4090 BLOCKER)
- SGLang: #28676 (HIGH), #28771 (CRITICAL, 44% throughput loss)
- 5 AVOIDED by config, 2 STILL AT RISK (#6794, #28771)

## New Tools (Session 2 continued)
- verl_v1_grpo_training_loop_simulator (4 modes, 10-phase breakdown)
- verl_v1_grpo_data_flow_tracer (4 modes, TQ operations + debug checks)
- grpo_memory_planner (4 modes, GPU+model viability)
- cuda_stream_safety_pattern_synthesis (4 modes, 8 pattern members)

## Knowledge Map: 18 domains, 72 connections, 33 rules (17 MUST DO + 16 MUST NOT)
