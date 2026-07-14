---
name: grpo-debug
description: Debug and diagnose GRPO training issues across verl, Megatron, DeepSpeed, and RTX 4090. Symptom-to-bug mapping, NaN debugging, Muon clipping avoidance, and CUDA stream safety analysis.
triggers:
  - GRPO debug
  - GRPO NaN
  - training NaN
  - loss plateau
  - gradient explosion
  - Muon clipping
  - ChainedOptimizer
  - CUDA stream safety
  - record_stream
  - training crash
  - GRPO failure
  - gradient norm growth
  - logprob mismatch
  - reward all zero
  - singleton degeneration
  - verl bug
  - DeepSpeed bug
  - Megatron bug
---

# GRPO Debug & Diagnostic Skill

## When to Use

- User encounters NaN, loss plateau, gradient explosion during GRPO training
- User asks about debugging GRPO training failures
- User wants to diagnose Muon clipping or ChainedOptimizer issues
- User needs CUDA stream safety analysis for training
- User asks about verl/Megatron/DeepSpeed GRPO bugs
- User wants symptom-to-bug mapping for training issues

## 7 Pattern Classes × Bug Mapping

| Pattern Class | Bugs | Frameworks | Root Cause |
|--------------|------|------------|------------|
| **cuda_stream_safety** | #8061, #45552, #28499 | DeepSpeed, vLLM, PyTorch | record_stream missing or wrong stream |
| **muon_clipping** | #8068, #7776, #5394, #5395 | DeepSpeed, Megatron, verl | Scale-invariant optimizer + global clipping = stall |
| **discrete_decision_mismatch** | #5384, #5386 | Megatron, verl | Train/rollout top-k mismatch → wrong GRPO signal |
| **lora_distortion** | #6782, #28566 | vLLM, SGLang | LoRA rank>=64 breaks EOS / cache corruption |
| **execution_overlap_oom** | #6772, #45552 | PyTorch FSDP2, vLLM | Overlap all_gather + backward → OOM |
| **singleton_degeneration** | #605, #663 | rLLM, verl | gs=1 = REINFORCE(baseline=0) |
| **moe_fp16_nan** | #10579, Switch TF | vLLM-Ascend, CUDA | FP16 softmax overflow → NaN |

## Symptom → Bug Quick Map

| Symptom | Most Likely Bug | Fix |
|---------|----------------|-----|
| **NaN in loss** | #8061 overlap_comm+torch.compile or #10579 FP16 softmax | overlap_comm=False or FP32 gating softmax |
| **Loss plateau (no improvement)** | #5394/#5395 Muon clipping stall | Switch to Adam, remove ChainedOptimizer global clip |
| **Gradient norm → infinity** | #8068 gradient_clipping=0.0 | Set gradient_clipping=1.0 explicitly |
| **All rewards = 0.0** | #663 Step.output null or #605 gs=1 degeneration | Fix null handling, use gs>=4 |
| **logprob mismatch train vs rollout** | #5384/#5386 DSA indexer replay | Implement DSAIndexerReplay |
| **EOS token missing in output** | #6782 LoRA rank=64 distortion | Use LoRA rank=32 |
| **OOM on FSDP2** | #6772 execution overlap | Use FSDP1, NOT FSDP2 |
| **MoE NaN (any platform)** | #10579 FP16 gating softmax overflow | Always compute gating softmax in FP32 |

## Diagnostic Tools

### 1. GRPO Training Debug Playbook
```bash
python3 tools/grpo_training_debug_playbook.py playbook    # Full playbook with 7 pattern classes
python3 tools/grpo_training_debug_playbook.py symptoms     # Symptom→bug mapping
python3 tools/grpo_training_debug_playbook.py fixes        # Fix reference for each bug
python3 tools/grpo_training_debug_playbook.py rtx4090      # RTX 4090 specific debug report
```

### 2. GRPO NaN Debugging Guide
```bash
python3 tools/grpo_nan_debugging_guide.py check            # Check config for NaN risks
python3 tools/grpo_nan_debugging_guide.py diagnose         # Diagnose NaN from symptoms
python3 tools/grpo_nan_debugging_guide.py avoid            # Generate avoidance recommendations
python3 tools/grpo_nan_debugging_guide.py rtx4090          # RTX 4090 NaN risk analysis
```

### 3. Megatron Muon Clipping Avoidance
```bash
python3 tools/megatron_muon_clipping_avoidance_tool.py check       # Check Megatron config
python3 tools/megatron_muon_clipping_avoidance_tool.py diagnose    # Diagnose from symptoms
python3 tools/megatron_muon_clipping_avoidance_tool.py avoid       # Generate avoidance config
python3 tools/megatron_muon_clipping_avoidance_tool.py rtx4090     # RTX 4090 Muon analysis
```

### 4. CUDA Stream Safety Pattern Synthesis
```bash
python3 tools/cuda_stream_safety_pattern_synthesis.py map           # Map stream patterns across frameworks
python3 tools/cuda_stream_safety_pattern_synthesis.py analyze       # Analyze stream safety for a config
python3 tools/cuda_stream_safety_pattern_synthesis.py compare       # Compare frameworks
python3 tools/cuda_stream_safety_pattern_synthesis.py rtx4090       # RTX 4090 stream safety report
```

### 5. Algorithm Infra Knowledge Map (theory→bug→decision)
```bash
python3 tools/algorithm_infra_knowledge_map.py theory     # Theory derivation for a bug
python3 tools/algorithm_infra_knowledge_map.py bug         # Bug→pattern→decision chain
python3 tools/algorithm_infra_knowledge_map.py decision    # Decision→theory justification
python3 tools/algorithm_infra_knowledge_map.py rtx4090     # RTX 4090 knowledge report
python3 tools/algorithm_infra_knowledge_map.py proof       # Mathematical proof for a rule
python3 tools/algorithm_infra_knowledge_map.py cross       # Cross-framework pattern analysis
```

## MUST DO Rules (GRPO Training)

1. **ZeRO-2 + CPU_Adam** — Never ZeRO-3 on dp=1 (pure overhead)
2. **clip_grad=1.0** — Never leave default 0.0 (silently disabled → gradient explosion #8068)
3. **bypass_mode** — Skip old_log_prob forward → 18Ψ→3.8Ψ memory savings
4. **group_by_prompt** — Correct GRPO advantage normalization (not global)
5. **enforce_eager=True** — Disable CUDA graph for training (safety)
6. **FSDP1** — Only proven safe for GRPO (FSDP2 BLOCKER: #6468 CPU leak + #6772 OOM)
7. **gs>=4** — Minimum group size (gs=1 = REINFORCE degeneration)
8. **LoRA r=32 + bypass sync** — LoRA rank>=64 breaks EOS (#6782)
9. **naive checkpoint dp=1** — No distributed checkpoint overhead
10. **record_stream** — Must call for async CUDA operations
11. **ulimit 65535** — File descriptor limit for NCCL
12. **shaped rewards** — Use reward shaping, not flat outcome rewards
13. **gs>=8 for sparse** — Sparse MoE needs larger groups
14. **sleep_level=1** — Never sleep_level=2 on RTX 4090 (weight corruption)
15. **SGLang rollout** — SGLang > vLLM for GRPO (sleep/wake support)
16. **DSAIndexerReplay** — Replay DSA indices for train/rollout consistency
17. **FP32 gating softmax** — Always compute MoE gating in FP32 (#10579 universal)
18. **ECHO auxiliary loss** — Free env-observation loss on verl (#668)

## MUST NOT Rules (GRPO Training)

1. **ZeRO-3** — Pure overhead on dp=1, MUST use ZeRO-2
2. **Muon optimizer** — Scale-invariant + clipping = stall (#5394/#5395/#8068)
3. **overlap_comm=True dp=1** — NaN with torch.compile (#8061) + no benefit
4. **sleep_level=2 RTX 4090** — Weight corruption risk
5. **LoRA rank>=64 vLLM** — Breaks EOS (#6782)
6. **FSDP2** — BLOCKER: CPU leak + execution overlap OOM
7. **gs=1** — REINFORCE degeneration, MUST use gs>=4
8. **PPO-clip RTX 4090** — OOM (65.06 GiB > 24 GiB)
9. **full param sync** — LoRA delta is sufficient for weight sync
10. **NCCL checkpoint dp=1** — Use naive checkpoint
11. **outcome reward gs<16** — Too noisy for small groups
12. **merge vLLM encoder cache revert** — Known corruption bug
13. **global clipping Muon ChainedOptimizer** — Fundamental contradiction
14. **FP16 gating softmax for MoE** — Universal NaN (#10579)
15. **torch.compile + overlap_comm** — NaN (#8061)
16. **gradient_clipping=0.0** — Silently disabled (#8068)
17. **bf16+fp16 both enabled** — Choose bf16 only (SM89 native)

## Deep Reading References

- verl V1 GRPO training loop: `notebook/projects/verl-v1-sync-ppo-trainer-grpo-training-loop-deep-reading.md`
- verl V1 critical bugs: `notebook/projects/verl-v1-critical-bugs-issues-2026-06-20.md`
- DeepSpeed ZeRO gradient flow + stream safety: `notebook/projects/deepspeed-zero1-2-gradient-flow-stream-safety-deep-reading.md`
- Megatron Muon clipping: `notebook/projects/megatron-5394-chained-optimizer-muon-clipping-reading.md`
- GRPO singleton degeneration derivation: `notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md`
- RTX 4090 GRPO runbook: `notebook/projects/rtx4090-grpo-training-runbook.md`
- Always load: `notebook/auto-memory.md`

## RTX 4090 Specific Debug

**Optimal GRPO config (16.24 GiB peak / 24 GiB VRAM)**:
```
trainer_mode=sync | rollout.n=8 | bypass_mode=True | loss_type=ppo_clip
actor.strategy=fsdp | lora_rank=32 | enforce_eager=True | checkpoint=naive
use_critic=False | ref_in_actor=True | gradient_clipping=1.0
sleep_level=1 | free_cache_engine=True | overlap_comm=False
```

**If NaN appears on RTX 4090**: Check overlap_comm (must be False), gradient_clipping (must be 1.0), gating dtype (must be FP32 for MoE), LoRA rank (must be 32).

**If loss plateau on RTX 4090**: Check optimizer (not Muon), clip_grad (not 0.0), group_size (>=4).
