---
name: cross-framework-avoidance
description: Validate GRPO training configs against 36 cross-framework bug avoidance rules (19 MUST DO + 17 MUST NOT). Covers 7 frameworks (DeepSpeed, Megatron, vLLM, verl, SGLang, rLLM, MindIE), 7 pattern classes, and MoE NaN avoidance.
triggers:
  - cross-framework bug
  - GRPO config validation
  - bug avoidance matrix
  - MUST DO rules
  - MUST NOT rules
  - MoE NaN
  - FP16 softmax
  - LoRA distortion
  - config checker
  - training config validation
  - DeepSpeed config check
  - Megatron config check
  - vLLM config check
  - verl config check
  - SGLang config check
  - rLLM config check
  - MindIE config check
  - 36 rules
  - 7 pattern classes
---

# Cross-Framework Bug Avoidance Skill

## When to Use

- User wants to validate a GRPO training config before running
- User asks about cross-framework bug patterns
- User wants to check config against MUST DO / MUST NOT rules
- User asks about MoE NaN avoidance (FP16 gating softmax)
- User wants to check LoRA distortion risks
- User asks about specific framework bug avoidance
- User wants to compare bug patterns across frameworks

## 7 Frameworks × 7 Pattern Classes Matrix

| Pattern Class | DeepSpeed | Megatron | vLLM | verl | SLLM | rLLM | MindIE |
|--------------|-----------|----------|------|------|------|------|--------|
| **cuda_stream_safety** | #8061 | — | #45552 | — | #28499 | — | — |
| **muon_clipping** | #8068 | #5394/#5395 | — | #7776 | — | — | — |
| **discrete_decision_mismatch** | — | #5384/#5386 | — | — | — | — | — |
| **lora_distortion** | — | — | #6782 | — | #28566 | — | — |
| **execution_overlap_oom** | — | — | #45552 | — | — | — | #6772 |
| **singleton_degeneration** | — | — | — | — | — | #605/#663 | — |
| **moe_fp16_nan** | — | — | #10579 | — | — | — | #10579 |

All patterns marked → AVOIDED in our RTX 4090 config (except stale_cache_corruption → monitor).

## 36 Rules (19 MUST DO + 17 MUST NOT)

### 19 MUST DO Rules

| # | Rule | Why | Bug IDs |
|---|------|-----|---------|
| 1 | ZeRO-2 + CPU_Adam | dp=1 ZeRO-3 = pure overhead | #8061 |
| 2 | clip_grad=1.0 | Default 0.0 = silently disabled | #8068 |
| 3 | bypass_mode | Skip old_log_prob forward | verl V1 |
| 4 | group_by_prompt | Correct GRPO advantage norm | #667 |
| 5 | enforce_eager=True | Disable CUDA graph for training | safety |
| 6 | FSDP1 | Only proven safe for GRPO | #6468, #6772 |
| 7 | gs>=4 | gs=1 = REINFORCE degeneration | #605, #663 |
| 8 | LoRA r=32 + bypass sync | rank>=64 breaks EOS | #6782 |
| 9 | naive checkpoint dp=1 | No distributed overhead | perf |
| 10 | record_stream | Async CUDA stream safety | #8061, #45552 |
| 11 | ulimit 65535 | FD limit for NCCL | NCCL |
| 12 | shaped rewards | Not flat outcome rewards | #663 |
| 13 | gs>=8 for sparse MoE | Sparse needs larger groups | #605 |
| 14 | sleep_level=1 | Never sleep_level=2 RTX 4090 | risk |
| 15 | SGLang rollout | Sleep/wake support for GRPO | verl V1 |
| 16 | DSAIndexerReplay | Train/rollout consistency | #5384 |
| 17 | FP32 gating softmax | MoE NaN universal | #10579 |
| 18 | ECHO auxiliary loss | Free env-observation loss | #668 |
| 19 | LoRA+bypass sync | Delta sync, not full param | verl V1 |

### 17 MUST NOT Rules

| # | Rule | Why | Bug IDs |
|---|------|-----|---------|
| 1 | ZeRO-3 | Pure overhead dp=1 | #8061 |
| 2 | Muon optimizer | Scale-invariant + clip = stall | #5394, #8068 |
| 3 | overlap_comm=True dp=1 | NaN + no benefit | #8061 |
| 4 | sleep_level=2 RTX 4090 | Weight corruption | risk |
| 5 | LoRA rank>=64 vLLM | Breaks EOS | #6782 |
| 6 | FSDP2 | BLOCKER: CPU leak + OOM | #6468, #6772 |
| 7 | gs=1 | REINFORCE degeneration | #605, #663 |
| 8 | PPO-clip RTX 4090 | OOM 65.06 > 24 GiB | budget |
| 9 | full param sync | LoRA delta sufficient | perf |
| 10 | NCCL checkpoint dp=1 | Use naive checkpoint | perf |
| 11 | outcome reward gs<16 | Too noisy for small groups | #663 |
| 12 | merge vLLM encoder cache revert | Corruption bug | #46125 |
| 13 | global clipping Muon ChainedOptimizer | Fundamental contradiction | #5394 |
| 14 | FP16 gating softmax MoE | Universal NaN | #10579 |
| 15 | torch.compile + overlap_comm | NaN | #8061 |
| 16 | gradient_clipping=0.0 | Silently disabled | #8068 |
| 17 | bf16+fp16 both enabled | Choose bf16 only (SM89) | perf |

## Validation Tools

### 1. Cross-Framework GRPO Bug Avoidance Matrix (34 rules)
```bash
python3 tools/cross_framework_grpo_bug_avoidance_matrix.py matrix          # 7×7 pattern matrix
python3 tools/cross_framework_grpo_bug_avoidance_matrix.py validate        # Validate config (JSON)
python3 tools/cross_framework_grpo_bug_avoidance_matrix.py rtx4090         # RTX 4090 optimal config
python3 tools/cross_framework_grpo_bug_avoidance_matrix.py cross-framework # Cross-framework analysis
```

Validate a config:
```bash
python3 tools/cross_framework_grpo_bug_avoidance_matrix.py validate --config '{"zero_stage":2,"gradient_clipping":1.0,"overlap_comm":false,"lora_rank":32,"fsdp_mode":"fsdp1","group_size":8}'
```

### 2. MoE NaN FP32 Softmax Avoidance Tool
```bash
python3 tools/moe_nan_fp32_softmax_avoidance_tool.py check      # Check MoE config for NaN risks
python3 tools/moe_nan_fp32_softmax_avoidance_tool.py diagnose    # Diagnose MoE NaN from symptoms
python3 tools/moe_nan_fp32_softmax_avoidance_tool.py avoid       # Generate avoidance config
python3 tools/moe_nan_fp32_softmax_avoidance_tool.py rtx4090     # RTX 4090 MoE NaN report
```

### 3. Megatron Muon Clipping Avoidance
```bash
python3 tools/megatron_muon_clipping_avoidance_tool.py check      # Check Megatron config
python3 tools/megatron_muon_clipping_avoidance_tool.py diagnose   # Diagnose from symptoms
python3 tools/megatron_muon_clipping_avoidance_tool.py avoid      # Generate avoidance config
python3 tools/megatron_muon_clipping_avoidance_tool.py rtx4090    # RTX 4090 Muon report
```

### 4. FSDP1 vs FSDP2 Decision Guide
```bash
python3 tools/fsdp1_vs_fsdp2_decision_guide.py guide      # Decision guide
python3 tools/fsdp1_vs_fsdp2_decision_guide.py compare     # Compare FSDP1 vs FSDP2
python3 tools/fsdp1_vs_fsdp2_decision_guide.py validate    # Validate FSDP config
python3 tools/fsdp1_vs_fsdp2_decision_guide.py rtx4090     # RTX 4090 FSDP report
```

### 5. DeepSpeed ZeRO Safety Checker
```bash
python3 tools/deepspeed_zero_safety_checker.py --mode check --config <path>
python3 tools/deepspeed_zero_safety_checker.py --mode diagnose --symptoms <str>
python3 tools/deepspeed_zero_safety_checker.py --mode rtx4090
```

## RTX 4090 Optimal Config (32/34 PASS)

```json
{
  "zero_stage": 2,
  "optimizer": "cpu_adam",
  "gradient_clipping": 1.0,
  "overlap_comm": false,
  "lora_rank": 32,
  "fsdp_mode": "fsdp1",
  "group_size": 8,
  "bypass_mode": true,
  "enforce_eager": true,
  "use_critic": false,
  "ref_in_actor": true,
  "checkpoint_mode": "naive",
  "sleep_level": 1,
  "rollout_engine": "sglang",
  "model_dtype": "bf16",
  "gating_dtype": "fp32"
}
```

Result: **32 PASS, 0 FAIL, 2 WARN** (WARN: outcome reward gs<16, stale cache corruption → monitor)

## MoE NaN Universal Pattern

★★★★★★★★ ALWAYS compute MoE gating softmax in FP32 regardless of platform.

**Mathematical proof**:
```
FP16 max ≈ 65504
softmax_FP16(logits) = exp(x_i) / Σexp(x_j)
When any x_j > 65504 → exp(x_j) = inf → inf/inf = NaN
FP32 fix: logits.float() → softmax → result.to(dtype)
```

| Platform | Bug | Root Cause | Fix |
|----------|-----|------------|-----|
| CUDA (Switch TF 2021) | FP16 NaN | softmax overflow | FP32 gating |
| Ascend NPU (#10579) | torch.abs() + FP16 softmax | both patterns | Remove abs() + FP32 gating |
| Megatron (proactive) | FP32 gating built-in | Defense | Already safe |
| DeepSpeed (proactive) | FP32 gating built-in | Defense | Already safe |

**RTX 4090**: BF16 model dtype + FP32 gating softmax = safest config on ALL platforms.

## Knowledge Sources

Always load:
- `notebook/auto-memory.md`

On-demand load per framework:
- DeepSpeed: `notebook/projects/deepspeed-zero1-2-gradient-flow-stream-safety-deep-reading.md`
- Megatron: `notebook/projects/megatron-5394-chained-optimizer-muon-clipping-reading.md`
- vLLM: `notebook/projects/vllm-v1-architecture-critical-bugs-deep-reading.md`
- verl: `notebook/projects/verl-v1-grpo-training-loop-architecture-deep-reading.md`
- rLLM: `notebook/projects/rllm-latest-developments-2026-06-session3-reading.md`
- MindIE: `notebook/projects/mindie-vllm-ascend-latest-developments-2026-06-session3-reading.md`
- PyTorch: `notebook/projects/pytorch-fsdp2-2026-deep-reading.md`
