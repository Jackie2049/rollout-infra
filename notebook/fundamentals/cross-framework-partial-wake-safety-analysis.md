# Cross-Framework Partial Wake Safety Analysis

**Created: 2026-06-19 | Synthesis of 3 partial wake/state transfer bugs across 4 frameworks**
**★★★★★★★★★ Key insight: staged wake_up without full state validation = CUDA illegal memory access across ALL frameworks!**

---

## 1. The Problem: Partial Wake-Up in RLHF Sleep/Wake

★★★★★★★★★ In RLHF training (GRPO, PPO), the serving engine uses sleep/wake to share GPU memory with the training engine. A **partial wake_up** restores only some resources (e.g., weights but NOT KV cache), then executes a forward pass. If the forward pass accesses resources that are still asleep/released → illegal memory access → crash or silent corruption.

---

## 2. Three Bug Instances

### 2.1 vLLM #44395/#44483: wake_up(tags=["weights"]) + forward → illegal memory access

| Aspect | Details |
|--------|---------|
| **Framework** | vLLM |
| **Issue** | #44395 |
| **Fix PR** | #44483 (+1645/-24) |
| **Root cause** | `wake_up(tags=["weights"])` restores weights but leaves KV cache asleep → forward pass accesses released KV cache → CUDA illegal memory access |
| **Effect** | CUDA error: an illegal memory access was encountered → crash |
| **Trigger** | RLHF/colocate workflows with staged wake (weights-only wake before full wake) |
| **Fix** | Resume scheduling only after executor is fully resident. `model_executor.is_sleeping` stays True until all sleep tags are awake |
| **Status** | OPEN, blocked |
| **RTX 4090 risk** | HIGH — verl uses vLLM as rollout backend → partial wake = crash! |

★★★★★★★★★ **How it works**:
```
Sleep → offload KV cache + weights to CPU
Partial wake_up(tags=["weights"]) → restore weights only, KV cache still offloaded
Forward pass → model weights on GPU, but KV cache pointers still pointing to released memory
→ CUDA illegal memory access → crash!
```

### 2.2 SGLang #28676: MXFP8 MoE shuffle cache CLOBBERED on RL weight reload

| Aspect | Details |
|--------|---------|
| **Framework** | SGLang |
| **Issue** | #28676 |
| **Root cause** | MoE shuffle cache uses GPU-resident physical memory that is NOT invalidated during weight reload → stale cache clobbers new weights |
| **Effect** | 64x accuracy blowup → garbage outputs |
| **Trigger** | MoE model + RL weight reload (GRPO step transition) |
| **Fix** | dict.clear() on cache + weight-load funnel call (+28/-2) |
| **Status** | OPEN, 0 maintainer engagement |
| **RTX 4090 risk** | CRITICAL — MoE GRPO BLOCKED without this fix! |

★★★★★★★★★ **Pattern similarity**: Both vLLM #44395 and SGLang #28676 involve GPU-resident state that is NOT properly invalidated during state transfer:
- vLLM: KV cache pointers → point to released memory → forward accesses released memory
- SGLang: MoE shuffle cache → stale data clobbers new weights → forward uses stale cache

### 2.3 vLLM-Ascend #10684: DSA Hadamard ALL-ZERO after sleep/wake

| Aspect | Details |
|--------|---------|
| **Framework** | vLLM-Ascend (MindIE) |
| **Issue** | #10684 |
| **Root cause** | DSA (DeepSeek Attention) Hadamard transform constant buffer lost during NPU sleep/wake → ALL-ZERO output |
| **Effect** | DSA attention produces wrong results → quality degradation |
| **Trigger** | Ascend NPU sleep/wake cycle in RLHF training |
| **Fix** | Not yet implemented |
| **Status** | OPEN, critical |
| **RTX 4090 risk** | NOT directly (Ascend NPU only) → but same pattern family as CUDA bugs |

---

## 3. Pattern Family: State Lifecycle Mismatch in Sleep/Wake

★★★★★★★★★ All 3 bugs share the **State Lifecycle Mismatch** pattern family:

```
Sleep: state A on GPU (weights, KV cache, Hadamard constant)
Wake: state B transferred from CPU to GPU
Bug: forward pass uses state C (stale pointer/released memory/stale cache)
  → C is neither A nor B → neither old nor new → invalid state!
```

**Three manifestations**:
1. **Pointer invalidation failure**: vLLM #44395 — KV cache pointer points to released memory
2. **Cache invalidation failure**: SGLang #28676 — MoE shuffle cache not cleared on weight reload
3. **Constant buffer loss**: vLLM-Ascend #10684 — Hadamard constant lost during state transfer

★★★★★★★★★ **Common root cause**: State transfer between sleep and wake does NOT fully validate that ALL GPU-resident state is consistent. Only the explicitly transferred state (weights) is validated, but implicitly resident state (KV cache, MoE cache, Hadamard constant) is NOT.

---

## 4. RTX 4090 Defense Rules

★★★★★★★★★ **For verl HYBRID sleep/wake on RTX 4090**:

1. **sleep_level=1 (LoRA adapter path)**: SAFE
   - Base weights stay on GPU → no partial wake needed
   - LoRA adapter swapped in/out via tags=["kv_cache"] → minimal state change
   - No pointer invalidation risk → base weights always resident

2. **sleep_level=2 (merge path)**: RISKY
   - Full weights offloaded to CPU → wake_up needs to restore ALL weights
   - If partial wake (tags=["weights"] only) → KV cache still asleep → #44395 risk!
   - Must do full wake_up(tags=["kv_cache","weights"]) → safe but slow (80x slower than sleep_level=1)

★★★★★★★★★ **RTX 4090 MUST**:
- Use sleep_level=1 (LoRA adapter path) → no partial wake risk
- NEVER use partial wake_up(tags=["weights"]) alone → ALWAYS include kv_cache tag
- For MoE models: enforce_eager=True + clear MoE cache at weight-reload boundary
- For DSV4: enforce_eager=True MANDATORY (11 failures across 4 frameworks!)

★★★★★★★★★ **RTX 4090 MUST NOT**:
- NEVER use sleep_level=2 on RTX 4090 unless full wake_up guaranteed
- NEVER skip KV cache tag in wake_up → #44395 crash risk
- NEVER skip MoE cache invalidation at weight-reload boundary → #28676 clobber risk
- NEVER assume GPU-resident state survives sleep/wake without explicit validation

---

## 5. Fix Comparison

| Bug | Fix | LOC | Status |
|-----|-----|-----|--------|
| vLLM #44395 | Resume scheduling only after executor fully resident | +1645/-24 | OPEN, blocked |
| SGLang #28676 | dict.clear() on cache + weight-load funnel | +28/-2 | OPEN, stalled |
| vLLM-Ascend #10684 | Not yet implemented | unknown | OPEN, critical |

★★★★★★★★★ **Fix pattern**: All fixes require **explicit state validation** at the sleep/wake boundary:
1. vLLM: ensure executor fully resident before scheduling → validates ALL state
2. SGLang: clear cache at weight-reload boundary → invalidates stale state
3. vLLM-Ascend: preserve Hadamard constant during state transfer → retain constant state

---

## References

- vLLM #44395/#44483: https://github.com/vllm-project/vllm/issues/44395
- SGLang #28676: https://github.com/sgl-project/sglang/issues/28676
- vLLM-Ascend #10684: https://github.com/sgl-project/sglang/issues/10684 (MindIE)
- verl sleep/wake architecture: notebook/projects/verl-rl-infra-reading.md
- Silent corruption pattern: notebook/fundamentals/silent-corruption-pattern-family-analysis.md
- RTX 4090 runbook: notebook/projects/rtx4090-grpo-training-runbook.md

---

*Created 2026-06-19. Partial wake safety analysis — 3 bugs across 4 frameworks, state lifecycle mismatch pattern family, RTX 4090 sleep_level=1 SAFE, sleep_level=2 RISKY.*
