# Auto Memory — AI Infra Engineer Learning Project

> Updated: 2026-06-20 (Session 3 final update)

## Project Context
- Working directory: ~/workspace/rollout-infra/
- Goal: Become strong AI infra engineer → senior AI expert
- Current focus: 7 frameworks (DeepSpeed, Megatron, vLLM, verl, MindIE/vLLM-Ascend, rLLM, PyTorch)
- GPU: university + matpool — BOTH OFFLINE. User will provide GPU device later for PR validation.

## Expert Readiness: 47/50 (94%)
- Theory: 14/10, Infra: 10/10, Math→Bug: 10/10, Practical: 8/10, OSS: 5/10

## Key New Findings (Session 3)
- #5394/#5395: Muon clipping stall + skip_grad_norm_clip fix (cross-framework pattern)
- #5384/#5386: DSA indexer replay for GRPO (train/rollout mismatch)
- #667: rLLM GRPO grouping fix MERGED (critical correctness fix)
- #663: rLLM Step.output null → ALL rewards were 0.0 before fix (double bug!)
- #671/#672: rLLM verl 0.8.0 upgrade (breaking changes, simplification)
- #669/#668: PRPO + ECHO advantage estimators (new algorithms in rLLM)
- #658: rLLM cumulative token mode for Tinker (drift-free multi-turn)
- #10579: MoE NaN universal pattern (FP16 softmax overflow on CUDA AND NPU)
- #10684: MindIE/vLLM-Ascend DSA Hadamard + sleep/wake
- #10592: NPUIPC weight transfer (mirrors CUDA IPC)
- Ascend bugs mirror CUDA 2014-2018 era → budget 30-40% more debugging time
- MoE gating softmax MUST use FP32 regardless of platform

## MUST DO: ZeRO-2, CPU_Adam, clip_grad=1.0, bypass, group_by_prompt, enforce_eager, FSDP1, gs>=4, LoRA+bypass sync, naive ckpt dp=1, record_stream, ulimit 65535, shaped rewards, gs>=8 for sparse, sleep_level=1, SGLang rollout, DSAIndexerReplay, FP32 gating softmax, ECHO auxiliary loss

## MUST NOT: ZeRO-3, Muon, overlap_comm dp=1, sleep_level=2 RTX4090, LoRA rank>=64 vLLM, FSDP2, gs=1, PPO-clip RTX4090, full param sync, NCCL ckpt dp=1, outcome reward gs<16, merge vLLM encoder cache revert, global clipping Muon ChainedOptimizer, FP16 gating softmax for MoE

## 7 Pattern Classes × 7 Frameworks Matrix
- cuda_stream_safety: #8061, #45552, #28499 → AVOIDED
- muon_clipping: #8068, #7776, #5394, #5395 → AVOIDED (Adam)
- discrete_decision_mismatch: #5384, #5386 → AVOIDED (not DSA)
- lora_distortion: #6782, #28566 → AVOIDED (r=32)
- execution_overlap: #6772, #45552 → AVOIDED (FSDP1+SGLang)
- singleton_degeneration: #605 → AVOIDED (gs>=8)
- stale_cache_corruption: #46125, #45979 → monitor
- NEW: MoE_FP16_NaN: #10579, Switch TF → AVOIDED (FP32 softmax)

## Knowledge Map: 24 domains, 90 connections, 36 rules (19 MUST DO + 17 MUST NOT)
## Tools: 9 session tools (simulator, tracer, planner, synthesis, playbook, Muon avoidance, cross-framework matrix, MoE NaN avoidance)
## Background agents: rLLM + MindIE completed (both comprehensive reading notes saved)
