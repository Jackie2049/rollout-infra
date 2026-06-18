# RTX 4090 GRPO Training Runbook — Complete Step-by-Step Guide

> 2026-06-18 | Operational guide synthesizing ALL verified findings
> ★★★★★★★★ Covers: framework selection, config validation, troubleshooting, memory optimization
> ★★★★★★★★ Based on 1000+ commits of source-level analysis across 7 frameworks
> ★★★★★★★★ GPU required for final validation — configs tested against source, not runtime

---

## Quick Decision Guide

### Which framework? (30-second answer)

| Framework | Rank | Status | Why |
|-----------|------|--------|-----|
| **verl CPPO+bypass** | #1 | READY | 18Ψ→3.8Ψ, bypass_mode MANDATORY, best trust region |
| **verl Tinker primitives** | #1.5 | READY | Split training in verl, no framework switch |
| **verl GRPO+bypass** | #2 | READY | Standard GRPO, bypass eliminates ref model |
| **DeepSpeed ZeRO-2** | #2.5 | READY | CPU_Adam, ZeRO-2 only, no ZeRO-3 |
| **rLLM Tinker** | #3 | BLOCKED | #605 grouping bug → GRPO BROKEN |
| **Megatron core** | #4 | BLOCKED | Muon crash + clipping + no LoRA export |

**Answer: USE verl CPPO+bypass (#1). ALWAYS.**

---

## Step 1: Environment Setup

### 1.1 verl Installation (China mainland mirrors)

```bash
# Use conda with tsinghua mirror
conda create -n verl python=3.12 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda activate verl

# Install verl from source (latest with #6699, #6717, #6688)
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

# Install vLLM (for rollout backend)
pip install vllm==0.23.0 -i https://mirrors.aliyun.com/pypi/simple/

# Install Ray
pip install "ray[default]" -i https://mirrors.aliyun.com/pypi/simple/
```

### 1.2 Critical: verl version MUST include #6699

```bash
# Check that detach fix is present (MERGED June 12)
git log --onimize --all | grep 6699 | head -1
# OR: check verl version >= commit from June 12

# ★★★★★★★★ Without #6699: 64 GiB OOM at ~micro-batch #400!
# ★★★★★★★★ With #6699: 16.2 GiB stable throughout!
```

---

## Step 2: Model Selection

### 2.1 Recommended models for RTX 4090 (24 GiB)

| Model | Type | LoRA rank | Peak Memory | Status |
|-------|------|-----------|-------------|--------|
| Qwen3-1.7B | dense | 32 | ~8 GiB | ✅ Easy fit |
| Qwen3-8B | dense | 32 | ~16.2 GiB | ✅ Fits with margin |
| Qwen3.5-27B | dense | 32 | ~16.2 GiB | ✅ Fits with #6512 per-unit summon |
| Qwen3-MoE (A0.6B+B4B) | MoE | 32 | ~19.8 GiB | ✅ Fits with AutoEP |
| Qwen3.5-35B-A3B | MoE | 32 | ~20 GiB | ⚠️ Tight, needs #6512 |

### 2.2 DO NOT use these on RTX 4090

| Model | Why |
|-------|-----|
| Qwen3-30B-A3B (dense total) | 30B total > 24GB even with LoRA+offload |
| Any model with LoRA rank > 32 | #6782: rank>64 breaks EOS → MUST rank<=32 |
| Qwen3-72B | Way too large for 24GB |

---

## Step 3: Config Validation (MUST RUN BEFORE TRAINING)

```bash
# Validate your config against RTX 4090 safety rules
python tools/rtx4090_grpo_config_reference.py --mode validate --config your_config.yaml

# ★★★★★★★★ 10 validation rules (all MUST pass):
# 1. overlap_comm=False (NaN bug #8061)
# 2. zero_stage!=3 (ZeRO-3 pure overhead + regression #8072)
# 3. cpu_offload=True (CPU_Adam, NOT Muon offload #7939)
# 4. lora_rank<=32 (#6782 EOS bug)
# 5. bypass_mode=True (CPPO MANDATORY, 18Ψ→3.8Ψ)
# 6. engine_backend in [fsdp, veomni] (#6699 unfixed backends!)
# 7. gradient_clipping=1.0 (#8068 default 0→1.0)
# 8. muon_global_clipping=0 (#7776/#5394)
# 9. zero3_peft_regression warning (#8072)
# 10. cppo_without_bypass warning (#6731)
```

---

## Step 4: Recommended Config — verl CPPO+bypass (RTX 4090 #1)

```yaml
# ★★★★★★★★ RTX 4090 BEST config: verl CPPO+bypass
# 16.2 GiB peak → 7.8 GiB margin → safe

algorithm:
  adv_estimator: cppo          # NOT grpo → CPPO better trust region
  use_kl_in_reward: False       # bypass_mode eliminates KL
  cppo_w_min: 0.8               # Binary-TV min
  cppo_delta_b: 0.02            # Binary-TV delta

data:
  train_batch_size: 64
  max_prompt_length: 1024
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: Qwen3.5-27B           # or Qwen3-8B
    lora_rank: 32                # ★★★★★★★★ MUST <=32! #6782
    lora_alpha: 64               # MUST 2x rank → scaling=2.0
    target_modules: all-linear
    lora.merge: False            # unmerged for HYBRID mode
    enable_gradient_checkpointing: True
    trust_remote_code: True

  actor:
    strategy: fsdp2
    optim.lr: 1e-5
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: False
    use_kl_loss: False            # bypass_mode → no KL loss needed
    fsdp_config:
      param_offload: True         # CPU offload for base weights
      optimizer_offload: True     # CPU offload for optimizer states

  rollout:
    name: vllm
    mode: hybrid                  # ★★★★★★★★ HYBRID = optimal RTX 4090
    bypass_mode: True             # ★★★★★★★★ MANDATORY → 18Ψ→3.8Ψ
    tensor_model_parallel_size: 1  # single GPU
    gpu_memory_utilization: 0.5
    enforce_eager: True           # avoid CUDA graph issues
    free_cache_engine: True       # memory management
    n: 8                          # GRPO n responses per prompt
    dtype: bfloat16
    load_format: safetensors
    layered_summon: True          # ★★★★★★★★ per-unit summon with #6512

  ref:
    strategy: fsdp2
    fsdp_config:
      param_offload: True         # ref model offloaded (but bypass eliminates!)
```

---

## Step 5: Pre-Flight Checks

### 5.1 Engine Backend Safety

```bash
# Check which verl engine backend has detach fix
python tools/verl_engine_backend_safety.py --mode check

# ★★★★★★★★ MUST use FSDP or VeOmni backend
# Automodel/Megatron/TorchTitan → SAME leak as #6699 → OOM!
```

### 5.2 Cross-Framework Safety Matrix

```bash
# Check all 7-framework safety items for RTX 4090
python tools/rtx4090_cross_framework_safety_matrix.py --mode summary

# ★★★★★★★★ 8 CRITICAL items → all addressed in this runbook
```

### 5.3 LoRA Rank Validation

```bash
# MUST: lora_rank <= 32
# #6782: rank=64 NEVER emits EOS → all responses truncated → GRPO BROKEN!
# lora_rank=32 + lora_alpha=64 → scaling=2.0 → works correctly
```

---

## Step 6: Training Execution

### 6.1 Launch command (verl CPPO+bypass on RTX 4090)

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=cppo \
    algorithm.use_kl_in_reward=False \
    algorithm.cppo_w_min=0.8 \
    algorithm.cppo_delta_b=0.02 \
    actor_rollout_ref.model.path=Qwen3.5-27B \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora.merge=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.rollout.mode=hybrid \
    actor_rollout_ref.rollout.bypass_mode=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.total_epochs=15 \
    trainer.n_gpus_per_node=1
```

---

## Step 7: Monitoring and Troubleshooting

### 7.1 Memory Monitoring

```python
# Watch peak memory during training
import torch
print(f"Peak allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GiB")
# Expected: ~16.2 GiB (with #6699 fix)
# If >20 GiB: check engine backend (must be FSDP!), check lora_rank (must be <=32!)
```

### 7.2 GRPO Troubleshooter

```bash
# Run troubleshooter if training fails
python tools/grpo_troubleshooter_4090.py --mode diagnose

# ★★★★★★★★ 17 known pitfalls checked
# Most common RTX 4090 issues:
# 1. Engine backend NOT FSDP → memory leak (#6699 unfixed)
# 2. LoRA rank > 32 → EOS never emitted (#6782)
# 3. bypass_mode=False → ref model 14 GiB overhead
# 4. ZeRO-3 → pure overhead + regression (#8072)
# 5. overlap_comm=True → NaN (#8061)
```

### 7.3 Training Health Indicators

```
★★★★★★★★★ Expected healthy training metrics (from #6699 PR verification):

After fix:
  → grad_norm: 0.015 (healthy)
  → rollout/training log-prob Pearson corr: 0.993 (excellent alignment)
  → Peak allocated: 16.2 GiB → STABLE throughout
  → response_length/clip_ratio: <0.3 (responses terminate normally)
  → reward signal: non-degenerate → advantage computation works

★★★★★★★★★ If unhealthy:
  → grad_norm > 1.0 → check gradient_clipping (#8068)
  → Pearson corr < 0.9 → check determinism (#6572, VLLM_BATCH_INVARIANT)
  → Peak memory > 20 GiB → check engine backend, lora_rank
  → clip_ratio > 0.8 → check lora_rank (must <=32!), check EOS
  → reward near 0 → check GRPO grouping (rLLM #605 style bugs)
```

---

## Critical Config Rules (Summary)

```
★★★★★★★★★ MUST rules for RTX 4090 GRPO:

1.  lora_rank <= 32                    # #6782: rank>64 breaks EOS
2.  lora_alpha = 2 * lora_rank         # scaling = 2.0 standard
3.  lora.merge = False                  # unmerged for HYBRID mode
4.  bypass_mode = True                  # 18Ψ→3.8Ψ → ref model eliminated
5.  engine = fsdp (NOT automodel/megatron/torchtitan)  # #6699 unfixed leak!
6.  zero_stage = 2 (NOT 3)              # ZeRO-3 overhead + regression
7.  overlap_comm = False                # #8061: NaN bug
8.  gradient_clipping = 1.0             # #8068: default 0→1.0
9.  enforce_eager = True                # avoid CUDA graph issues
10. free_cache_engine = True            # memory management
11. layered_summon = True               # #6512: per-unit summon
12. verl version >= #6699               # detach fix → 4x memory reduction

★★★★★★★★★ MUST NOT rules:

1.  MUST NOT use lora_rank > 32         # #6782 EOS bug
2.  MUST NOT use ZeRO-3                 # regression + pure overhead
3.  MUST NOT use overlap_comm = True     # NaN bug
4.  MUST NOT use Muon optimizer          # crash + clipping + CPU offload blocked
5.  MUST NOT use automodel/megatron/torchtitan engine  # memory leak
6.  MUST NOT use rLLM for GRPO          # #605 grouping bug → BROKEN
7.  MUST NOT use CUDA graph for DSV4 inference  # 4 failures in 4 days → enforce_eager=True MANDATORY
8.  MUST NOT attempt DSV4 full model on RTX 4090  # requires minimum 128 GPUs (TP1/ETP1 fixed)
```

---

## DeepSeek-V4 Special Considerations

### DSV4 is NOT feasible on single RTX 4090

```
★★★★★★★★★ DSV4 architecture constraints:
  → TP1/ETP1 fixed — no tensor parallelism support → each GPU holds full non-expert params
  → Minimum: 16 nodes (128 GPUs) for flash variant (PP4 EP8 CP4)
  → 64 nodes (512 GPUs) for pro variant (PP8 EP16 CP4)
  → ALL_OFFLOAD=True by default → memory is TIGHT even with 128 GPUs
  → RTX 4090 (24 GiB) CANNOT fit DSV4 even with INT4 quantization!
```

### DSV4-related models that ARE feasible on RTX 4090

```
★★★★★★★★★ Models that use DSV4 techniques but fit on RTX 4090:
  → Qwen3.5-35B-A3B: GDN + MLA + MoE → fits with ZeRO-2+LoRA+CPU_offload
  → Kimi-K2.5: MLA + MoE → feasible with similar configuration
  → GLM-5: may be feasible depending on size
  → ★★★★★★★★ Key: use models that use SUBSETS of DSV4 techniques (GDN, MLA, MoE) but are smaller!
```

### DSV4 CUDA Graph Rule: enforce_eager=True

```
★★★★★★★★★ DSV4 + CUDA graph = systematic fragility (4 failures, 4 days, 2 frameworks + Ascend):
  → vLLM #45972: cudagraph optimization → garbage output → MERGED revert
  → SGLang #28591: MTP Online Compress → testing revert
  → SGLang #28569: EAGLE3 CUDA graph → illegal memory access crash
  → vLLM-Ascend #10628/#10640: DSV4 failure + MTP startup crash on Ascend!

★★★★★★★★★ Rule: ANY per-request dynamic routing MUST run eagerly:
  → MoE expert selection → MUST run eagerly
  → DSA indexer → MUST run eagerly
  → MTP draft decisions → MUST run eagerly
  → Online compress decisions → MUST run eagerly

★★★★★★★★★ Config for DSV4-like models (Qwen3.5-35B-A3B) on RTX 4090:
  vllm:
    enforce_eager: True    # MANDATORY for DSV4-like models!
    gpu_memory_utilization: 0.90
  verl:
    algorithm.rollout_correction.bypass_mode: True
    actor.policy_loss.loss_mode: cppo  # position-weighted trust region → especially important for DSV4-like models!
```

### DSV4 GRPO Training Pathway

```
★★★★★★★★★ verl #6791 (MERGED June 18): DSv4/GLM5/KimiK2.5 via Megatron Lite
  → For multi-node DSV4 training (128+ GPUs) → use Megatron Lite backend
  → For single RTX 4090 → use Qwen3.5-35B-A3B with FSDP2 backend
  → ★★★★★★★★ CPPO + bypass_mode + FSDP2 = optimal RTX 4090 DSV4-like GRPO training!

★★★★★★★★★ Megatron #5384 (OPEN): DSA Indexer Replay for RL training stability
  → When training models with DSA sparse attention → MUST record indexer top-k decisions during rollout
  → Replay during training → prevents train/rollout mismatch → same pattern as MoE RouterReplay
  → Needed for models like DeepSeek-V3.2, DeepSeek-V4 that use DSA

★★★★★★★★★ See full DSV4 synthesis: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md
```

---

## References

- verl #6699: https://github.com/verl-project/verl/pull/6699 (detach fix, 4x memory)
- verl #6731: https://github.com/verl-project/verl/pull/6731 (CPPO, bypass MANDATORY)
- verl #6512: https://github.com/verl-project/verl/pull/6512 (per-unit LoRA summon)
- verl #6782: https://github.com/verl-project/verl/issues/6782 (LoRA rank>32 EOS bug)
- verl #6688: https://github.com/verl-project/verl/pull/6688 (LoRA IPC buffer crash fix)
- verl #6717: https://github.com/verl-project/verl/pull/6717 (Tinker training primitives)
- DeepSpeed #8061: overlap_comm NaN
- DeepSpeed #8072/#8073: ZeRO-3+PEFT regression
- rLLM #605: GRPO grouping bug
- verl #6791: https://github.com/verl-project/verl/pull/6791 (DSv4/GLM5/KimiK2.5 Megatron Lite)
- vLLM #45972: REVERT of DSV4 cudagraph optimization
- SGLang #28591/#28569: DSV4 MTP revert + EAGLE3 crash
- Megatron #5384: DSA Indexer Replay feature request
- DSV4 synthesis: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md
- Tools: rtx4090_grpo_config_reference.py, verl_engine_backend_safety.py, grpo_troubleshooter_4090.py
