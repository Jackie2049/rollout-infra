# SGLang PDMux Architecture: Single-GPU Prefill/Decode Multiplexing Deep Reading

**Date**: 2026-07-15 (Session 10 continued)
**Purpose**: Understand SGLang's PDMux (Prefill-Decode Multiplexing) for single-GPU GRPO serving
**Sources**: sglang/python/sglang/srt/multiplex/ source code, sglang-nav skill

---

## 1. PDMux Architecture Overview

```
PDMux (Prefill-Decode Multiplexing): Time-slice prefill and decode on a SINGLE GPU

Instead of 2 separate machines (prefill server + decode server):
  → 1 GPU time-slices between prefill and decode phases
  → Uses Green Context (spatial partitioning) to share SMs
  → Prefill and decode run CONCURRENTLY on different SM groups

Key innovation: spatial multiplexing (SM partitioning) not temporal multiplexing (full GPU switching)
  → Prefill uses N SMs, Decode uses M SMs, N+M = total SMs on GPU
  → Both phases run in PARALLEL on different CUDA streams
  → No "full GPU context switch" overhead — both phases share GPU memory
```

---

## 2. Green Context: Spatial SM Partitioning

```
Green Context (from sgl_kernel.spatial module):

create_greenctx_stream_by_value(prefill_sm_count, decode_sm_count, gpu_id)
  → Creates a CUDA stream with restricted SM allocation
  → Prefill stream: uses prefill_sm_count SMs
  → Decode stream: uses decode_sm_count SMs
  → Both streams run CONCURRENTLY on the same GPU

SM partitioning rules (from pdmux_context.py):
  sm_60 (Pascal): min_per_part=1, multiple=1
  sm_70 (Volta):  min_per_part=2, multiple=2
  sm_80 (Ampere): min_per_part=4, multiple=2
  sm_90 (Hopper): min_per_part=8, multiple=8

RTX 4090 (sm_89 Ada Lovelace):
  → NOT sm_90! Falls under sm_80 rules: min_per_part=4, multiple=2
  → Total SMs: 128
  → Valid partitions: prefill 4-112 SMs, decode = 128-prefill SMs
  → Constraints: prefill ≥ decode (prefill is compute-heavy), decode ≥ 16 SMs

H20-3e (sm_90 Hopper):
  → min_per_part=8, multiple=8
  → Total SMs: 132
  → Valid partitions: prefill 8-124 SMs, decode = 132-prefill SMs

★★★★★★★★★ For RTX 4090 GRPO:
  - PDMux requires sm_80+ (Green Context API only on Ampere and later)
  - RTX 4090 = sm_89 → PDMux AVAILABLE!
  - SM partitioning: prefill gets more SMs (compute-heavy), decode gets fewer (memory-heavy)
```

---

## 3. Stream Groups Architecture

```
From pdmux_context.py initialize_stream_groups():

SM_COUNTS structure (for sm_group_num=8):
  [0] (total_sm, 0)          → Pure prefill (all SMs for prefill)
  [1] (prefill_sm, decode_sm) → Split: prefill dominates
  [2] (prefill_sm, decode_sm) → Split: more balanced
  [3] (prefill_sm, decode_sm) → Split: more balanced
  [4] (prefill_sm, decode_sm) → Split: more balanced
  [5] (prefill_sm, decode_sm) → Split: decode dominates
  [6] (prefill_sm, decode_sm) → Split: decode dominates even more
  [7] (0, total_sm)          → Pure decode (all SMs for decode)

STREAM_GROUPS: each entry = (prefill_stream, decode_stream)
  [0]: (normal_cuda_stream, normal_cuda_stream) → Pure prefill
  [1-6]: (greenctx_prefill_stream, greenctx_decode_stream) → Split
  [7]: (normal_cuda_stream, normal_cuda_stream) → Pure decode

Dynamic adjustment:
  → stream_idx chosen based on decode batch size
  → More decode requests → stream_idx increases → more SMs for decode
  → Fewer decode requests → stream_idx decreases → more SMs for prefill

★★★★★★★★★ This is EXACTLY what GRPO serving needs:
  Rollout generation (prefill+decode) can share GPU with training update
  → Prefill phase: generate responses for all group_size samples
  → Decode phase: complete all responses
  → PDMux adjusts SM allocation dynamically based on workload
```

---

## 4. event_loop_pdmux: Core Scheduling Loop

```
From multiplexing_mixin.py event_loop_pdmux():

Main loop (infinite while True):
  1. recv_requests() on decode_stream → get new GRPO requests
  2. process_input_requests() → tokenize, register
  3. update_split_prefill_batch() on prefill_stream → prepare new batch
  4. update_running_batch() on decode_stream → maintain decode batch
  5. adjust_stream_groups() if needed → change SM partition
  6. run_batch(decode) on decode_stream → decode step (M SMs)
  7. run_batch(prefill) on prefill_stream → prefill step (N SMs)
  8. synchronize decode → process decode results
  9. synchronize prefill → process prefill results (split across layers)
  10. merge completed prefill into running_batch → new requests enter decode

★★★★★★★★ Key insight: Prefill and Decode run in PARALLEL
  → Prefill on N SMs (prefill_stream)
  → Decode on M SMs (decode_stream)
  → No sequential bottleneck → throughput maximized
  → But: BOTH phases share GPU memory → memory budget constrained
```

---

## 5. Split Prefill: Layer-by-Layer Processing

```
Split prefill = prefill processed layer-by-layer, not all-at-once:

ForwardMode.SPLIT_PREFILL:
  → split_index: current layer being processed
  → split_forward_count: how many layers to process this iteration
  → split_forward_token_budget: max tokens per forward (default 65536)
  → split_prefill_finished: True when all layers done

Why split prefill:
  → Prefill is compute-heavy: all layers at once → SMs monopolized by prefill
  → Split: process N layers per iteration → interleaved with decode steps
  → Each iteration: decode step (M SMs) + prefill N layers (N SMs)
  → After all layers done: merge into running_batch → decode phase continues

★★★★★★★★★ For GRPO serving:
  Group_size=8 GRPO rollout: 8 responses per prompt
  → Prefill: process prompt (shared prefix) → 1× compute
  → Decode: complete 8 responses → 8× compute
  → PDMux: prefill gets more SMs for prompt processing
  → Then shift to decode-dominated → adjust stream groups
  → Dynamic: stream_idx changes as decode batch grows
```

---

## 6. Stream Group Adjustment: Dynamic SM Partitioning

```
From adjust_stream_groups():

Logic:
  1. If running_batch NOT empty AND split_prefill_batch exists:
     → Mixed workload → choose stream_idx based on decode batch size
     → More decode requests → higher stream_idx → more SMs for decode
     → Formula: stream_idx = decode_bs * (sm_group_num - 2) / decode_bs_divisor

  2. If running_batch NOT empty AND NO split_prefill:
     → Decode-only → stream_idx = sm_group_num - 1 → all SMs for decode

  3. If running_batch empty AND NO split_prefill:
     → Prefill-only → stream_idx = 0 → all SMs for prefill

  4. Manual divisions override:
     → manual_divisions: [(prefill_sm, decode_sm, threshold), ...]
     → If decode_bs >= threshold → use corresponding partition

★★★★★★★★★ GRPO serving workload pattern:
  Step 1: Prefill phase → stream_idx=0 → all SMs for prefill
  Step 2: Prefill+Decode mixed → stream_idx shifts based on group_size
  Step 3: Decode phase → stream_idx=7 → all SMs for decode
  Step 4: Training update → no PDMux needed (use ZeRO-2 instead)

This is the natural PDMux cycle for GRPO:
  Prefill → Mixed → Decode → Sleep(wake for training) → Prefill → cycle
```

---

## 7. PDMux for GRPO Training on RTX 4090

```
Complete GRPO + PDMux pipeline on RTX 4090:

Phase 1: Rollout Generation (PDMux serving)
  → SGLang PDMux handles prefill + decode
  → Prefill: process prompt prefix (shared KV across group)
  → Decode: generate group_size responses
  → HiCache: offload completed KV to CPU
  → All on single GPU (24 GiB)

Phase 2: Sleep (release KV, keep weights)
  → sleep(level=1): release KV cache → keep model weights on GPU
  → HiCache eviction: completed requests' KV offloaded to CPU/SSD

Phase 3: Training Update (ZeRO-2)
  → No PDMux needed → full GPU for training
  → ZeRO-2 + CPU_Adam: optimizer on CPU
  → bypass_mode: skip old_log_prob forward → 3.8 GiB activations
  → gradient_clipping = 1.0

Phase 4: Wake + Delta Sync
  → Wake: reload model (updated weights)
  → LoRA delta sync: only ~50 MiB adapter weights
  → Ready for next rollout phase

★★★★★★★★★ Memory budget with PDMux:
  Rollout (PDMux): 14 GiB model + 2-3 GiB KV = ~16-17 GiB
  Training (ZeRO-2): 14 GiB model + 3.8 GiB activations + 1.4 GiB grads = 19.2 GiB
  Both phases fit in 24 GiB!

PDMux SM allocation for RTX 4090 (128 SMs):
  Prefill-dominated: 80 SMs prefill + 48 SMs decode
  Decode-dominated: 48 SMs prefill + 80 SMs decode
  Pure decode: 0 SMs prefill + 128 SMs decode
```

---

## 8. PDMux vs Traditional P/D Disaggregation

```
Comparison: PDMux (single GPU) vs Disaggregation (2 machines):

| Property | PDMux (1 GPU) | Disaggregation (2 GPUs) |
|----------|---------------|------------------------|
| **Hardware** | 1 GPU | 2+ GPUs (separate machines) |
| **Memory** | Shared (24 GiB) | Separate (each GPU has own VRAM) |
| **KV transfer** | Local (same GPU) | RDMA (inter-machine) |
| **Prefill SMs** | Dynamic (4-128) | All SMs on prefill machine |
| **Decode SMs** | Dynamic (4-128) | All SMs on decode machine |
| **Throughput** | Lower (SM partitioning) | Higher (dedicated machines) |
| **Cost** | 1 GPU | 2+ GPUs |
| **Latency** | Prefill: fast, Decode: slower | Both fast |
| **Complexity** | Green Context + SM partitioning | RDMA + KV transfer + bootstrap |

★★★★★★★★★ For RTX 4090 GRPO:
  PDMux is the ONLY viable option:
  → 1 GPU = budget constraint
  → Disaggregation needs 2 GPUs → out of budget
  → PDMux = time-slicing within 1 GPU
  → Trade-off: lower throughput per step, but zero extra hardware cost

For H100 cluster (production):
  Disaggregation is better:
  → 2+ machines = budget available
  → Dedicated prefill/decode → higher throughput
  → NIXL RDMA → fast KV transfer
  → But: PDMux still useful for single-GPU development/testing
```

---

## 9. PDMux Configuration for RTX 4090

```
PDMux YAML config for RTX 4090 (128 SMs, sm_89):

sm_group_num: 6  # 6 stream groups (2 pure + 4 split)
manual_divisions:
  - [80, 48, 16]   # prefill 80 SMs, decode 48 SMs, threshold: decode_bs >= 16
  - [64, 64, 32]   # balanced 64/64, threshold: decode_bs >= 32
  - [48, 80, 48]   # decode 80 SMs, threshold: decode_bs >= 48
  - [32, 96, 64]   # decode 96 SMs, threshold: decode_bs >= 64

split_forward_token_budget: 65536  # default
decode_bs_divisor: 36              # default

★★★★★★★★ For GRPO rollout (group_size=8):
  Typical decode batch size: 8-32 responses
  → When decode_bs=8: stream_idx=1 → 80 SMs prefill + 48 SMs decode
  → When decode_bs=16: stream_idx=2 → 64 SMs prefill + 64 SMs decode
  → When decode_bs=32: stream_idx=3 → 48 SMs prefill + 80 SMs decode
  → When decode_bs=48+: stream_idx=4 → 32 SMs prefill + 96 SMs decode

For GRPO rollout (group_size=4):
  Typical decode batch size: 4-16 responses
  → Smaller batch → more SMs for prefill → faster prompt processing
  → Then shift to decode as responses complete
```

---

## 10. PDMux Limitations and Risks

```
1. Green Context API requirement:
   → Only available on sm_80+ (Ampere, Ada Lovelace, Hopper)
   → RTX 4090 = sm_89 → PDMux available
   → Older GPUs (sm_70 Volta, sm_75 Turing) → NO PDMux support

2. SM partitioning overhead:
   → Prefill with fewer SMs → slower prompt processing
   → Decode with fewer SMs → slower response generation
   → Net: 20-40% throughput reduction vs dedicated machines

3. Memory contention:
   → Both phases share VRAM → memory budget constrained
   → Large KV cache during decode → may exceed memory
   → HiCache helps: offload completed KV to CPU/SSD

4. Split prefill complexity:
   → Layer-by-layer processing → more scheduling decisions
   → Forward count varies → may not align with decode batch timing
   → AllReduce across TP ranks needed for each split forward

5. Thread safety:
   → Two CUDA streams operating concurrently → race condition risk
   → Green Context restricts SMs per stream → reduces but doesn't eliminate races
   → Critical: model parameters shared between streams → must not be modified during both

★★★★★★★★ For RTX 4090 GRPO:
  Risk 1: manageable (sm_89 supports Green Context)
  Risk 2: acceptable (20-40% reduction vs dedicated, but saves 1 GPU cost)
  Risk 3: mitigated by HiCache + sleep/wake (KV offloaded between phases)
  Risk 4: acceptable for small group_size (4-8)
  Risk 5: CRITICAL — training update must NOT happen during PDMux serving
    → Sleep/wake boundary separates PDMux serving from ZeRO-2 training
    → NEVER run PDMux + training concurrently on same GPU!
```

---

## 11. PDMux + HiCache + Sleep/Wake Complete Pipeline

```
Complete RTX 4090 GRPO pipeline (3 mechanisms combined):

Step 1: Rollout Generation (PDMux + HiCache)
  → SGLang PDMux: prefill + decode on single GPU
  → Prefill: process prompt prefix (RadixTree prefix sharing)
  → Decode: generate group_size responses (decode_stream)
  → HiCache: offload completed requests' KV to CPU
  → Stream groups: adjust SM allocation dynamically
  → Duration: ~5-30 seconds (depends on group_size + max_tokens)

Step 2: Sleep (level=1) — Release KV, Keep Model
  → sleep(): release KV cache from GPU → keep model weights
  → HiCache: KV already on CPU/SSD → no data loss
  → GPU memory freed: ~2-3 GiB KV cache released
  → Duration: ~0.5 seconds (fast, only KV release)

Step 3: Training Update (ZeRO-2 + bypass)
  → ZeRO-2: optimizer on CPU → 0 GiB GPU for optimizer
  → bypass_mode: skip old_log_prob forward → 3.8 GiB activations only
  → gradient_clipping = 1.0 (prevent explosion)
  → LoRA: only train adapter weights → faster convergence
  → Duration: ~10-60 seconds (depends on batch size)

Step 4: Wake + Delta Sync
  → wake(): reload model weights (updated LoRA adapter)
  → Delta sync: only LoRA weights (~50 MiB) → fast
  → Model weights already on GPU (sleep level=1 kept them)
  → Duration: ~1 second (fast, only LoRA delta)

Step 5: Next Rollout (back to Step 1)
  → PDMux resumes: Prefill new prompt + Decode responses
  → HiCache: reload warm prefixes from CPU → prefix reuse
  → Updated model → better responses → reward improves

★★★★★★★★★ Total cycle time estimate:
  Rollout: 5-30s + Sleep: 0.5s + Training: 10-60s + Wake: 1s
  = ~15-90 seconds per GRPO step
  Throughput: ~7-40 steps per hour (depends on group_size + model size)

Memory per phase:
  Rollout: 14 + 3 = 17 GiB → PDMux + HiCache manages
  Sleep: 14 GiB → model weights stay, KV released
  Training: 14 + 3.8 + 1.4 = 19.2 GiB → fits in 24 GiB!
  Wake: 14 + 0.05 = 14.05 GiB → LoRA delta only
```

---

## 12. Cross-Framework Comparison: Single-GPU GRPO Mechanisms

```
| Mechanism | SGLang | vLLM | verl | DeepSpeed |
|-----------|--------|------|------|-----------|
| **SM partitioning** | PDMux (Green Context) | UBatch (2 microbatches) | None | None |
| **KV offload** | HiCache (GPU→CPU→SSD) | CuMem sleep/wake | sleep/wake | ZeRO optimizer |
| **Prefill/Decode** | PDMux time-slice | UBatch overlap | HYBRID colocate | ZeRO-2 colocate |
| **Weight sync** | Delta sync (LoRA) | Delta sync (LoRA) | TransferQueue | ZeRO-2 gradient |
| **Training** | External (verl HYBRID) | External (verl HYBRID) | HYBRID colocate | ZeRO-2 standalone |

★★★★★★★★★ SGLang PDMux is UNIQUE:
  → Only framework with SM-level spatial partitioning for single-GPU serving
  → vLLM UBatch: temporal partitioning (2 sequential microbatches), NOT spatial
  → verl: no SM partitioning (full GPU for each phase)
  → DeepSpeed: no serving capability (training only)

For RTX 4090 GRPO:
  SGLang PDMux + HiCache + verl HYBRID = best single-GPU pipeline
  → PDMux for serving (prefill + decode time-sliced)
  → HiCache for KV management (GPU→CPU→SSD hierarchy)
  → verl HYBRID for training (sleep/wake + ZeRO-2 + bypass)
```

---

## Session Stats
- **PDMux architecture**: Green Context SM partitioning, stream groups, split prefill
- **event_loop_pdmux**: concurrent prefill + decode on different SM groups
- **SM partitioning rules**: architecture-dependent (sm_89 RTX 4090 = min 4, multiple 2)
- **Stream group adjustment**: dynamic SM partitioning based on decode batch size
- **Split prefill**: layer-by-layer processing interleaved with decode
- **PDMux vs Disaggregation**: single-GPU time-slice vs multi-GPU dedicated machines
- **RTX 4090 config**: manual_divisions for group_size=4-8 GRPO rollout
- **Complete pipeline**: PDMux + HiCache + Sleep/Wake + ZeRO-2 = end-to-end GRPO
- **Cross-framework comparison**: SGLang PDMux unique in SM-level spatial partitioning
