# Auto Memory — AI Infra Engineer Learning Project

> Updated: 2026-06-20 (Session 2 final update)

## Project Context
- Working directory: ~/workspace/rollout-infra/
- Goal: Become strong AI infra engineer → senior AI expert
- Current focus: 7 frameworks (DeepSpeed, Megatron, vLLM, verl, MindIE/vLLM-Ascend, rLLM, PyTorch)
- GPU: university + matpool — BOTH OFFLINE. User will provide GPU device later for PR validation.

## Expert Readiness: 45/50 (90%)
- Theory: 14/10, Infra: 10/10, Math→Bug: 10/10, Practical: 7/10, OSS: 4/10

## Key New Findings (Session 2 continued)
- ZeRO gradient flow: backward → IPG bucket → reduce_scatter → average_tensor → optimizer
- #8061: average_tensor only syncs current_stream → #8080: copy_streams set fix
- overlap_comm pointless on dp=1 (NCCL reduce-scatter = identity)
- verl V1: 10-phase step pipeline, TransferQueue data fabric
- LoRA adapter path: 80x sync reduction (~200 MiB vs ~14 GiB)
- bypass + ref_in_actor: 5→1 forward passes, 14 Psi savings
- vLLM V1: one-process-per-GPU, ZMQ IPC, CuMemAllocator sleep/wake
- #45552: sleep_level=2 crashes RTX 4090 (missing synchronize())
- #46125: encoder cache revert = DANGEROUS for RLHF (silent corruption)
- #6794: 4 blocking sub-issues (record_stream, disk_race, big_values, makedirs)
- #6772: vLLM + FSDP2 execution order peak overlap → OOM
- #45979: DSV4 sparse cache VINDICATED (intra-step safe, inter-step dangerous)
- #46118: MTP grammar FSM conflict → 58% request failure

## MUST DO: ZeRO-2, CPU_Adam, clip_grad=1.0, bypass, group_by_prompt, enforce_eager, FSDP1, gs>=4, LoRA+bypass sync, naive ckpt dp=1, record_stream, ulimit 65535, shaped rewards, gs>=8 for sparse, sleep_level=1, SGLang rollout

## MUST NOT: ZeRO-3, Muon, overlap_comm dp=1, sleep_level=2 RTX4090, LoRA rank>=64 vLLM, FSDP2, gs=1, PPO-clip RTX4090, full param sync, NCCL ckpt dp=1, outcome reward gs<16, merge vLLM encoder cache revert

## CUDA Stream Safety Pattern Family (8 members, 5 AVOIDED by config)
- AVOIDED: #45552 (sleep_level=1), #8061 (overlap_comm=False), #8072 (FSDP1), #6782 (LoRA r=32), #5394 (cpu_adam)
- AT RISK: #6794 (silent corruption, monitor checksums), #28771 (44% throughput loss, monitor accept_length<2.0)

## Knowledge Map: 20 domains, 78 connections, 34 rules (17 MUST DO + 17 MUST NOT)
## Background agents: vLLM V1 sleep/wake source reading, Megatron latest developments (still running)
