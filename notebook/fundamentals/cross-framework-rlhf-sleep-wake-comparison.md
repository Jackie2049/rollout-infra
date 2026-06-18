# Cross-Framework RLHF Sleep/Wake Architecture Comparison

## SGLang vs vLLM vs verl Integration

> 2026-06-18 | Cross-Framework Synthesis | verl v0.3.0-pre, SGLang v0.5.13, vLLM v0.23.0
> ★★★★★★★★ RTX 4090 GRPO: SGLang sleep_level=1 + LoRA adapter = 80x payload reduction vs sleep_level=2 merge> ★★★★★★★★ RTX 4090 GRPO: enforce_eager=True MANDATORY for DSV4/DSV3/DSV4 models (8+ failures across 3 frameworks)
> ★★★★★★★★ RTX 4090 GRPO: vLLM-Ascend sleep_level=1 NOT SUPPORTED → NPU-only path

> ★★★★★★★★ SGLang tag-based > vLLM integer-based: more extensible, fine-grained control

> ★★★★★★★★ verl HYBRID: same-process sleep/wake → zero IPC overhead → RTX 4090 OPTIMAL

> ★★★★★★★★ verl COLOCATED: same PG separate process → no weight sync needed → GRM/judge use case
> ★★★★★★★★ verl STANDALONE: separate GPU → full network transfer → off-policy only case

> ★★★★★★★★ DSV4 ROOT CAUSE: caching per-step dynamic data as if static → 8+ failures across vLLM, SGLang, MindIE → enforce_eager=True

> ★★★★★★★★ PartialOffloadPolicy dp=1 NOT viable: shard=identity → CPUOffloadPolicy(pin_memory=True) default=TRUE) remains ONLY dp=1 path
> ★★★★★★★★ C128 state mapping lifecycle bug (#28612): SWA mapping freed/reused → stale slot → accuracy degradation → fix: derive from full KV location directly

---

## 1. Architecture Overview: Three Sleep/Wake Systems

```
┌───────────────────────────────────────────────────────────────────────┐
│                RLHF Sleep/Wake Memory Management                    │
│                                                                     │
│  ┌─── SGLang (Tag-based) ───┐    ┌─── vLLM (Level-based) ───┐     │
│  │                            │    │                            │    │        │
│  │ tags=["kv_cache"]          │    │ sleep(level=1)             │    │        │
│  │   → release KV only        │    │   → offload weights,       │    │        │
│  │   → base weights STAY      │    │      discard KV cache      │    │        │
│  │                            │    │                            │    │        │
│  │ tags=["kv_cache","weights"]│    │ sleep(level=2)             │    │        │
│  │   → release EVERYTHING     │    │   → discard weights+KV     │    │        │
│  │   → full re-transfer       │    │   → full re-transfer       │    │        │
│  └────────────────────────────┘    └────────────────────────────┘        │
│                                                                     │
│  ┌─── verl Integration ────────────────────────────────────────┐    │
│  │                                                               │    │
│  │ HYBRID mode: same process, same GPU                          │    │
│  │   → HybridWorker.update_weights() orchestrates:              │    │
│  │     1. rollout.resume(tags=["weights"])  ← wake weights      │    │
│  │     2. get_per_tensor_param()            ← summon from FSDP  │    │
│  │     3. if LoRA: sleep_level=1, adapter sync only             │    │
│  │     4. if merge: sleep_level=2, full weight sync             │    │
│  │     5. aggressive_empty_cache()          ← reclaim GPU       │    │
│  │     6. rollout.resume(tags=["kv_cache"]) ← wake KV space    │    │
│  │                                                               │    │
│  │ COLOCATED mode: same PG, separate process                    │    │
│  │   → No weight sync needed (weights stay on GPU)              │    │
│  │   → Use case: GRM (LLM as judge)                             │    │
│  │                                                               │    │
│  │ STANDALONE mode: separate GPU allocation                     │    │
│  │   → Full network transfer (ZMQ/HTTP)                         │    │
│  │   → Use case: off-policy training                            │    │
│  └───────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Sleep/Wake Mechanism Comparison

### 2.1 SGLang Tag-Based API

**Source**: `sglang_rollout.py:278-293`, `async_sglang_server.py:469-490`

```python
# SGLang release() — tag-based granularity
async def release(self):
    if self.sleep_level == 1:      # LoRA adapter mode
        tags = ["kv_cache"]         # ★ ONLY release KV → base weights stay!
    else:                            # merge/default mode
        tags = ["kv_cache", "weights"]  # ★ Release everything
    await self._engine.release_memory_occupation(tags=tags)
```

**API primitives**:
- `ReleaseMemoryOccupationReqInput(tags=["kv_cache"])` — release KV cache only
- `ReleaseMemoryOccupationReqInput(tags=["weights"])` — release model weights only (hypothetical)
- `ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])` — release everything
- `ResumeMemoryOccupationReqInput(tags=[...])` — resume previously released resources

**Advantages**: Extensible — can add new tags for layer groups, intermediate activations, etc. SGLang's `release/resume` API is more future-proof than vLLM's fixed integer levels.

### 2.2 vLLM Level-Based API
**Source**: `vllm_rollout.py:51-95`, `vllm_async_server.py:969-974`

```python
# vLLM sleep — integer-based levels
if not is_torch_npu_available():   # CUDA path
    sleep_level = 1               # LoRA mode (offload weights, discard KV)
else:                              # Ascend NPU path
    sleep_level = 2               # ★★★★★★★★ ALWAYS full sleep on Ascend!
await self.engine.sleep(level=sleep_level)

# vLLM wake — tag-based BUT only for RLHF control
await self.engine.wake_up(tags=["weights"])      # wake weights only
await self.engine.wake_up(tags=["kv_cache"])     # wake KV only
await self.engine.wake_up(tags=["weights", "kv_cache"])  # wake both
```

**vLLM CuMemAllocator internals**:
- `sleep()` → offloads tagged allocations to CPU via `cudaMemcpy`, discards others via `unmap_and_release`
- `wake_up()` → re-maps and copies back from CPU backup tensors
- `use_memory_pool(tag)` → context manager for allocation tagging

**Caveat**: vLLM-Ascend does NOT support `sleep_level=1` — always full sleep on NPU → weight re-transfer every step → much slower for LoRA adapter path. This is a HARD BLOCKER for verl RLHF on Ascend NPU (#10684).

### 2.3 Mechanism Comparison Table

| Feature | SGLang | vLLM | vLLM-Ascend |
|---------|--------|------|------------|
| Sleep mechanism | `tokenizer_manager.release_memory_occupation()` (tag-based) | `engine.sleep(level=1|2)` (integer-based) | `engine.sleep(level=2)` ONLY |
| Wake mechanism | `tokenizer_manager.resume_memory_occupation()` (tag-based) | `engine.wake_up(tags=[...])` (tag-based for wake only) | `engine.wake_up()` (re-map + copy) |
| LoRA adapter support | ★★★★★ `lora_as_adapter` → `tags=["kv_cache"]` only | Conditional (sleep_level=1) | ❌ NOT supported! |
| Weight update path | HTTP-based `update_weights` | ZMQ IPC `update_weights` | HTTP-based (verl patched) |
| Delta sync | #6794 (NEW, deferred LoRA path) | #6794 (SGLang-only) | N/A |
| Memory pool | SGLang internal pool | CuMemAllocator (tag-based) | CuMemAllocator (vLLM-Ascend) |
| PD disaggregation | SGLangPDReplica (prefill/decode roles) | Not supported | Not supported |
| CUDA IPC | `get_named_tensor_buckets` + DTensor | ZMQ handles | NPUIPC (#10592, +787) |
| Tags granularity | `["kv_cache"]`, `["weights"]`, `["kv_cache","weights"]` | `level=1` or `level=2` | `level=2` ONLY |

---

## 3. Two-Phase Weight Sync: Base Sync vs Adapter Sync

### 3.1 First Step: Base Sync (One-Time Full Transfer)

When `do_lora_base_sync = not self.base_sync_done` (first iteration only):

```python
per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=False
)
await self.rollout.update_weights(
    per_tensor_param_base, peft_config=peft_config,
    base_sync_done=False, global_steps=global_steps
)
```

**One-time cost**: Full model weights transferred to rollout server (~16 GiB for Qwen3-8B). This happens ON the FIRST iteration only then the LoRA adapter path kicks `base_sync_done=True` for and subsequent steps only only transfer LoRA deltas.

)

### 3.2 Every Step: Adapter Sync (LoRA Deltas Only)

```python
# After base_sync, each step only only transfers LoRA deltas:
await self.rollout.update_weights(
    per_tensor_param, peft_config=peft_config,
    base_sync_done=True, global_steps=global_steps
)
```

**LoRA unload/reload cycle** (SGLang):
1. `available_models()` → check if SGLANG_LORA_NAME exists
2. If exists: `unload_lora_adapter(SGLANG_LORA_NAME)` — remove old adapter
3. `wrap_lora_params(peft_config, weights)` → serialize new LoRA params
4. `LoadLoRAAdapterFromTensorsReqInput(lora_name, config_dict, serialized_tensors)` → load new adapter

**Critical detail**: Only ONE LoRA adapter at time (constant `SGLANG_LORA_NAME`). Old adapter must be unloaded before loading new one. This means the adapter swap cycle is unload→serialize→load` per step.

)

### 3.3 Payload Size Comparison

| Component | sleep_level=2 (merge) | sleep_level=1 (LoRA) |
|-----------|----------------------|---------------------|
| Base weight transfer | ~16 GiB (every step!) | ~16 GiB (first step only!) |
| LoRA adapter transfer | N/A | ~200 MiB (every step) |
| Total per step | ~16 GiB | ~200 MiB |
| **Payload reduction** | 1x | **80x reduction** |



**MoE models even more dramatic**: Qwen3-30B-A3B active params 3B → LoRA deltas ~60 MiB vs ~6 GiB base → **100x reduction**.

 )

---

## 4. verl RolloutMode Impact on Sleep/Wake

### 4.1 Mode Comparison



| Mode | Process Layout | Weight Sync | GPU Sharing | Use Case | RTX 4090 Ranking |
|------|---------------|-------------|-------------|----------|----------------|
| **HYBRID** | Same process | Required (sleep→wake→update) | Same GPU, same process | On-policy GRPO | ★★★★★★★★ #1 |
| **COLOCATED** | Separate process, same PG | Not required (weights stay) | Same GPU, diff process | GRM (judge) | N/A (multi-GPU) |
| **STANDALONE** | Separate GPU | Full transfer via network/ZMQ | Different GPU | Off-policy | N/A (multi-GPU) |

 |

**RTX 4090 Insight**: HYBRID mode is the ONLY viable single-GPU mode. Same-process = zero IPC overhead. The sleep/wake mechanism is the memory management backbonebones for COLOCATED/STANDALONE require placement groups with multiple GPUs. RTX 4090 (dp=1) MUST use HYBRID mode.

 )

### 4.2 HYBRID Training Step Flow (engine_workers.py:695-749)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HYBRID Training Step (per iteration)          │
│                                                                    │
│  1. set_expandable_segments(False)                                 │
│  2. rollout.resume(tags=["weights"])    ← wake up weights space  │
│  3. get_per_tensor_param()             ← summon from FSDP         │
│  4. if LoRA adapter (merge=False):                                │
│     a. sleep_level = 1                 ← KEY: only release KV    │
│     b. if first time: base_sync (full weights, one-time)        │
│     c. adapter sync (LoRA deltas only, ~200 MiB)              │
│  5. if merge mode:                                                │
│     a. sleep_level = 2                 ← release weights + KV    │
│     b. full weight sync every step (~16 GiB)                   │
│  6. offload to CPU if param_offload enabled                           │
│  7. aggressive_empty_cache()             ← reclaim GPU memory     │
│  8. rollout.resume(tags=["kv_cache"])    ← wake up KV space   │
│  9. set_expandable_segments(True)                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. DSV4 Systematic Instability: Sleep/Wake Connection

### 5.1 Root Cause: Caching Dynamic Data as Static

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

**8+ failures across 3 frameworks**: All share the same root cause — per-step dynamic data (MoE routing, SWA mapping  MTP/MTP-compress state, Hadamard state) is cached as if static execution graph. With CUDA graphs, `eager_break_during_capture` must be used to split the graph into static/captured and dynamic segments. But dynamic routing produces different token-to-expert assignments each step → graph topology mismatch → stale metadata → incorrect attention/garbage output/NaN.

 )

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 5.2 Failure Catalog

| Framework | Issue | Type | Impact | RTX 4090 |
|-----------|-------|------|-------|----------|
| vLLM | #45309 eager_break | Perf optimization | Garbage output | ★★★★★★★★★ |
| vLLM | #45863 sparse cache | Perf optimization | GSM8K 6.75% vs 87% | ★★★★★★★★★ |
| vLLM | #45972 2nd revert | Revert #45309 | Restores correctness | ★★★★★★★★★ |
| vLLM | #45979 3rd revert | Revert #45863 | CI auto-detected | ★★★★★★★★★ |
| vLLM-Ascend | #10724 PD-Mix crash | Disaggregation | Crash | ★★★★★★★★★ |
| SGLang | #28591 MTP revert | State management | Lifecycle bug | ★★★★★★★★★ |
| SGLang | #28612 C128 mapping | State lifecycle | Stale slot → accuracy degradation | ★★★★★★★★★ |
| SGLang | #28618 SM89 DSV4 RFC | SM89 validation | Opens DSV4 path on ★★★★★★★★★ |
| MindIE | #10684 Hadamard | Sleep/wake | ALL-ZERO after wake | ★★★★★★★★★★ CRITICAL |
| MindIE | #10579 MoE NaN | Attention kernel | NaN, 1-line fix | ★★★★★★★★★ |
| MindIE | #10645 DSV4 chat | Template bug | Chat template issue | ★★★★★★★★★ |
| MindIE | #10193 prefix cache | Caching bug | Stale prefix data | ★★★★★★★★★ |

**Pattern**: #45309 removes `eager_break` → entire attention becomes "capturable" → but dynamic routing makes stale graph → garbage. #45863 caches sparse metadata between steps → routing changes → stale metadata → accuracy regression. #28612 derives C128 state from SWA mapping → SWA freed → stale state → accuracy degradation. #10684 Hadamard state lost during sleep/wake → constant buffer zero on ALL-ZERO output.

 )

### 5.3 enforce_eager=True: The Universal Rule

 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**ALL DSV4/DSV3 models MUST use enforce_eager=True in rollout engine**. This disables CUDA graph capture entirely, avoiding the static/dynamic mismatch. Performance penalty is but15-20% TTFT), but correctness is guaranteed.

 )

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 6. Memory Budget Analysis: RTX 4090

### 6.1 Qwen3-8B (BF16, 24 GiB)

| Component | Training Phase | Rollout Phase |
|-----------|---------------|--------------|
| Base weights | ~16 GiB (FSDP sharded) | ~16 GiB (resident, sleep_level=1) |
| LoRA adapters | ~200 MiB (rank=32) | ~200 MiB (resident) |
| Optimizer states | CPU (CPU_Adam) | 0 |
| Gradients | CPU (offload) | 0 |
| KV cache (rollout) | 0 (released during training) | ~4-6 GiB (resident) |
| Free GPU | ~8 GiB (after sleep) | ~2-4 GiB (available) |

**sleep_level=1 advantage**: Base weights STAY during training → no 16 GiB re-transfer per step. Only 200 MiB LoRA deltas need. → **80x payload reduction**. )

### 6.2 Qwen3-30B-A3B (MoE, BF16, 24 GiB)

| Component | Training Phase | Rollout Phase |
|-----------|---------------|--------------|
| Active params (3B) | ~6 GiB (FSDP sharded) | ~6 GiB (resident, sleep_level=1) |
| Expert params (27B) | CPU (offload) | CPU (offload) |
| LoRA (rank=32, 3B active) | ~60 MiB | ~60 MiB |
| Optimizer | CPU (CPU_Adam) | 0 |
| KV cache | Released | ~3-4 GiB |

 |

**sleep_level=1 advantage for MoE models**: Active params small → LoRA deltas ~60 MiB vs ~6 GiB → **100x reduction**. )

---

## 7. RTX 4090 GRPO Training: MUST DO / MUST NOT

### 7.1 MUST DO
 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. **SGLang rollout** + sleep_level=1 + LoRA rank=32/alpha=64 + merge=false — 80x payload reduction
 )
2. **FSDP backend** (only backend with detach fix from #6699)
3. **enforce_eager=True** for DSV4/DSV3/DSV4 models — correctness guaranteed
 )
4. **CPU_Adam optimizer** (18Ψ→3.8Ψ optimizer memory reduction)
4. **gradient_clipping=1.0** (#8068 default 0→1.0 → ALWAYS set explicitly)
4. **param_offload=True + grad_offload=True** — CPU offload during training
4. **ZeRO-2** (NOT ZeRO-3! Ze ZeRO-3 = pure overhead on dp=1)
4. **pin_memory=True** (CPUOffloadPolicy default= already TRUE!)

 no need to set)
4. **bypass_mode** (verl CPPO+GRPO → 18Ψ→3.8Ψ)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 7.2 MUST NOT
 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. **peft.merge=true** → forces sleep_level=2 → full weight re-transfer every step
2. **lora_rank=64** → breaks EOS (#6782)
3. **vLLM-Ascend backend** → sleep_level=1 NOT supported on → NPU-only)
4. **Megatron backend** → C9 detach fix not in upstream
5. **overlap_comm=True on single GPU** → NaN bug (#8061)
4. **ZeRO-3** → pure overhead on dp=1
4. **PartialOffloadPolicy on dp=1** → shard=identity → resident=full param → OOM
 )
4. **CUDA graphs for DSV4 models** → 8+ failures → enforce_eager=True instead)

 )

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. Connection Map: Sleep/Wake to Other Framework Issues

 | This Sleep/Wake | Connection | Sleep/Wake Relevance |
 RTX 4090 |
|-------|---------------------|--------------------|---------------------|----------|

| **DS-1 (#8072)** | ZeRO-3+LoRA regression | sleep_level=1 with ZeRO-3 is BROKEN! → ★★★★★★★★★ |
 ★★★★★★★★★ |
| **VE-1 (#6782)** | LoRA rank=64 breaks EOS | Adapter path unus unus at ★★★★★★★★★ | ★★★★★★★★★ |
| **VE-2 (#6468)** | FSDP2 CPU memory leak | W6.3 GiB/step during weight sync | ★★★★★★★★★ | ★★★★★★★★★ |

| **VE-3 (#6699)** | detach model_output | 4x reduction | Frees GPU for KV cache | ★★★★★★★★★ | ★★★★★★★★★ |

| **VE-6512** | per-unit LoRA summon | 10x memory reduction | Enables LoRA adapter | ★★★★★★★★★ | ★★★★★★★★★ |
| **SG-3 (#27097)** | multi-LoRA determinism | Single adapter path must deterministic! | ★★★★★★★★★ | ★★★★★★★★★ |
| **SG-5 (#28618)** | SM89 DSV4 RFC | Opens DSV4 inference on ★★★★★★★★★ | ★★★★★★★★★ |
 | **VL-3 (#39096)** | SM89 batch invariance | enforce_eager=True | ★★★★★★★★★ | ★★★★★★★★★ |
 | **VA-1 (#10684)** | DSA Hadamard ALL-ZERO | Sleep/wake state lost | ★★★★★★★★★ CRITICAL | ★★★★★★★★★ CRITICAL |

 **VL-#45683** | Deterministic MoE combine | Critical for GRPO | ★★★★★★★★★ | ★★★★★★★★★ |

| **MG-#5387** | MFSDPv2 | New FSDP for Megatron | ★★★★★★★★★ | ★★★★★★★★★ |
| **MG-#5396** | GDN L2-norm fold | 24 GiB savings | ★★★★★★★★★ | ★★★★★★★★★ |
| **DS-#8068** | gradient_clipping default | 0→1.0 | MUST set 1.0 | ★★★★★★★★★ | ★★★★★★★★★ |
| **rLLM #605** | GRPO grouping bug | group size 1 → GRPO BROKEN | ★★★★★★★★★ CRITICAL | ★★★★★★★★★ CRITICAL |
| **PT #187620** | PartialOffloadPolicy | dp=1 NOT viable | ★★★★★★★★★ | ★★★★★★★★★ |

| **PT #184119** | SM89 guard | Blocks fp8→bf16 fusion | ★★★★★★★★★ | ★★★★★★★★★ |


| **PT #187636** | autotune_at_compile_time | Red SM89 batch risk ★ ★★★★★★★★★ | ★★★★★★★★★ |

| **VL #45819** | GDN batch invariance | Progressing CI | ★★★★★★★★★ | ★★★★★★★★★ |



---

## 9. Sleep/Wake Decision Tree: RTX 4090 dp=1

```
                    GPU (24 GiB)
 RTX 4090)
                           │
            ┌──────────────────────────────┐
            │    Model fits in GPU?     │
            └──────────────────────────────┘
            │                             │
      YES (<8B BF16)            NO (>8B or needs offload)
            │                             │
            │                    ┌──────────────────────────┐
            │                    │ Use CPUOffloadPolicy        │
            │                    │ (pin_memory=True default!)       │
            │                    │ ZeRO-2 + CPU_Adam             │
            │                    └──────────────────────────┘
            │                             │
            │                    Sleep/Wake Cycle:
            │                    ┌──────────────────────────┐
            │                    │ LoRA adapter path           │
            │                    │ sleep_level=1                │
            │                    │ tags=["kv_cache"] only      │
            │                    │ Base weights stay resident    │
            │                    │ 80-100x payload reduction     │
            │                    └──────────────────────────┯
            │                             │
            │                    ┌──────────────────────────┐
            │                    │ Merge path                 │
            │                    │ sleep_level=2                │
            │                    │ tags=["kv_cache","weights"] │
            │                    │ Full re-transfer each step   │
            │                    │ ★★★★★ AVOID on RTX 4090     │
            │                    └──────────────────────────┘
```

 **Key decisions**:
- CPUOffloadPolicy: Already optimal (pin_memory=True default). No need to explicitly set. verl FSDP2 uses CPUOffloadPolicy.
 DeepSpeed uses ZeRO-2+CPU_Adam ( )
- Sleep/Wake: LoRA adapter path (sleep_level=1) = 80-100x reduction → RTX 4090 OPTIMAL
 )
- Sleep/Wake: Merge path (sleep_level=2) → full re-transfer → AVOID on RTX 4090 ( )
- Backend: FSDP ONLY (detach fix) + verl or Megatron, AutoModel, TorchTitan → UNFIXED! → )
- Optimizer: CPU_Adam ONLY (18Ψ→3.8Ψ). Muon+CPU_offload BLOCKED (#7939) -> )

---

## 10. Future Directions

1. **#6794 delta weight sync**: LoRA delta encoding (~100x smaller payload). but SGLang-only, LoRA deferred. 4 CRITICAL review issues)
  )
2. **#6790 separate async trainer**: Switch+offload strategies → more flexible sleep/wake coordination between different modes combinations →  )
3. **#6512 per-unit LoRA summon**: 10x memory reduction → enables sleep_level=1 with larger models on )
4. **#187620 PartialOffloadPolicy**: dp>=2 ONLY → fractional CPU offload → multi-GPU path to )
5. **SGLang #28618 SM89 DSV4**: Opens DSV4 inference path on RTX 4090 → TP=8 on )
6. **SGLang PD disaggregation**: Prefill/decode split → sleep/wake decode separately →  )
7. **Megatron #5387 MFSDPv2**: New FSDP backend for Megatron → DBuffer primitives →  )
8. **rLLM Tinker async trainer**: SyncCoordinator → client-server SDK → similar architecture to verl #6790 →  )

9. **PyTorch Inductor SM89 guard**: #184119 + blocks batch-dependent fusion → complements P9 thesis →  )

10. **vLLM #45819 GDN batch invariance**: Active review CI → GDN-compatible models →  )

11. **verl #6782 LoRA EOS fix**: rank=64 breaks EOS → MUST use rank=32/alpha=64 →  )
12. **verl #6468 FSDP2 leak**: 6.3 GiB/step → monotonic growth → Ray OOM →  )
13. **SGLang #28612 C128 state mapping lifecycle fix**: Derive from full KV location →  stable lifecycle →  )

---

## Source Files Read

 | File | Key Lines | Purpose |
|------|-----------|---------|

| `replica.py` | 54-67, 131-2226 | RolloutMode enum, RolloutReplica ABC, init_hybrid/colocated/standalone |

| `sglang_rollout.py` | 192-196, 266-293, 295-380 | sleep_level, release/resume tags, update_weights + LoRA adapter | `
| `async_sglang_server.py` | 445-490, 496-508 | SGLang server sleep/wake/release_kv_cache | `
| `engine_workers.py` | 695-749 | HybridWorker.update_weights() flow, sleep_level=1 override | `
| `vllm_rollout.py` | 51-95 | vLLM sleep_level version check | `
| `vllm_async_server.py` | 969-974 | vLLM engine.sleep()/wake_up() | `
| `weight_update_utils.py` | 18-100 | split_buffer_updates, apply_buffer_updates |

| `base.py` | 44-69, 83-87 | BaseRollout ABC, _ROLLOUT_REGISTRY | `

---

## Related Reading Notes

 | Note | Lines | Focus |
|------|-------|-------|
| `verl-hybrid-sleep-wake-architecture-reading.md` | 444 | verl HYBRID mode deep dive |
| `vllm-v1-memory-management-architecture-reading.md` | TBDBD | vLLM CuMemAllocator deep dive (
| `sglang-28618-sm89-dsv4-fp8-reading.md` | TBD | SGLang SM89 DSV4 RFC deep dive | `
 | `pytorch-187620-fsdp2-partial-offload-policy-reading.md` | 1018 | FSDP2 offload policy deep dive | `
| `dsv4-systematic-instability-pattern-synthesis.md` | existing | DSV4 root cause analysis |
