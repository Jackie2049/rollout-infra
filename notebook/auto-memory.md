# Auto Memory — AI Infra Engineer Learning Project

> Updated: 2026-06-20 (Session 3 update)

## Project Context
- Working directory: ~/workspace/rollout-infra/
- Goal: Become strong AI infra engineer → senior AI expert
- Current focus: 7 frameworks (DeepSpeed, Megatron, vLLM, verl, MindIE/vLLM-Ascend, rLLM, PyTorch)
- GPU: university + matpool — BOTH OFFLINE. User will provide GPU device later for PR validation.

## Expert Readiness: 46/50 (92%)
- Theory: 14/10, Infra: 10/10, Math→Bug: 10/10, Practical: 8/10, OSS: 4/10

## Key New Findings (Session 3)
- #5394: ChainedOptimizer Muon clipping stall — positive-feedback loop (c≈2e-8 → NS degenerates)
- #5395: skip_grad_norm_clip fix — per-optimizer-group clipping: Muon=0, Adam=1.0
- #5384/#5386: DSA indexer replay — train/rollout top-k mismatch → incorrect GRPO signal
- #5400: GDN in_proj fused matrix → Muon orthogonalization meaningless
- Cross-framework Muon pattern: #5394 = #8068 = #7776 = #229/#230 (same root cause)
- Scale-invariant optimizer + scale-change (clipping) = fundamental contradiction
- 7 pattern classes × 7 frameworks × 34 rules = comprehensive avoidance matrix
- RTX 4090 optimal config: 32/34 PASS, 0 FAIL, 2 WARN

## MUST DO: ZeRO-2, CPU_Adam, clip_grad=1.0, bypass, group_by_prompt, enforce_eager, FSDP1, gs>=4, LoRA+bypass sync, naive ckpt dp=1, record_stream, ulimit 65535, shaped rewards, gs>=8 for sparse, sleep_level=1, SGLang rollout, DSAIndexerReplay

## MUST NOT: ZeRO-3, Muon, overlap_comm dp=1, sleep_level=2 RTX4090, LoRA rank>=64 vLLM, FSDP2, gs=1, PPO-clip RTX4090, full param sync, NCCL ckpt dp=1, outcome reward gs<16, merge vLLM encoder cache revert, global clipping Muon ChainedOptimizer

## 7 Pattern Classes (7×7 Matrix)
- cuda_stream_safety: #8061, #45552, #28499 → AVOIDED by config
- muon_clipping: #8068, #7776, #5394, #5395 → AVOIDED (Adam, not Muon)
- discrete_decision_mismatch: #5384, #5386 → AVOIDED (not using DSA)
- lora_distortion: #6782, #28566 → AVOIDED (r=32)
- execution_overlap: #6772, #45552 → AVOIDED (FSDP1 + SGLang)
- singleton_degeneration: #605 → AVOIDED (gs>=8)
- stale_cache_corruption: #46125, #45979 → monitor carefully

## Knowledge Map: 22 domains, 84 connections, 34 rules (17 MUST DO + 17 MUST NOT)
## Tools created: 6 session-2 tools + 2 session-3 tools (Muon avoidance + cross-framework matrix)
