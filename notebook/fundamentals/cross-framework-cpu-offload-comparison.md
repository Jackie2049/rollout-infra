# Cross-Framework CPU Offload Comparison — RTX 4090 GRPO Training

**Created: 2026-06-19 | Synthesis of 4 CPU offload approaches across 3 frameworks**
**★★★★★★★★★ Key question: Which CPU offload strategy is optimal for RTX 4090 dp=1 GRPO training?**

---

## 1. The Problem: Optimizer State Memory Dominance

For any model >8B parameters on RTX 4090 (24 GiB), the optimizer states dominate memory:

```
Memory budget for 7B model:
  Model weights (BF16):       14 GiB  (2 bytes per param)
  Optimizer states (FP32):    28 GiB  (8 bytes per param: m + v + master)
  Activations (gradient):      ~4 GiB  (depends on batch size)
  Total without offload:      46 GiB  → OOM on 24 GiB GPU!

With CPU offload (optimizer to CPU):
  Model weights (BF16):       14 GiB  (on GPU)
  Optimizer states (FP32):    0 GiB   (on CPU RAM)
  Activations (gradient):      ~4 GiB
  Peak GPU:                   18 GiB  → fits on RTX 4090 with 6 GiB margin
```

★★★★★★★★★ **The math is clear**: CPU offload is MANDATORY for >8B models on RTX 4090. The question is WHICH offload mechanism is best.

---

## 2. Four CPU Offload Approaches

### 2.1 DeepSpeed CPU_Adam (Current Standard)

| Aspect | Details |
|--------|---------|
| **Location** | CPU (Python subprocess) |
| **IPC** | Python multiprocessing (pickle serialization) |
| **Copyback** | Full fp32→bf16 materialization on GPU |
| **GPU transient** | 2944 MiB (ZeRO-3 partition level) |
| **Latency** | Slow (serialization overhead per step) |
| **Stability** | Python subprocess can crash silently |
| **ZeRO compatibility** | ZeRO-2 (recommended) or ZeRO-3 |
| **RTX 4090 verdict** | #1 current choice with ZeRO-2 |

★★★★★★★★★ **How it works**:
- ZeRO-2: each GPU holds full model params (bf16), optimizer states offloaded to CPU
- Gradient computation on GPU → CPU_Adam updates fp32 params → copyback updated bf16 params
- Copyback path: fp32→bf16 conversion occurs in the GPU subprocess → materializes full fp32 partition temporarily

### 2.2 DeepSpeed ZenFlow (#8058, Proposed)

| Aspect | Details |
|--------|---------|
| **Location** | CPU (native C++ process) |
| **IPC** | POSIX semaphores + shared memory (zero-copy) |
| **Copyback** | Chunked fp32→bf16 conversion — ★★★★★★★★ ONLY for ZeRO Stage 1/2! Stage 3 STILL full materialization! |
| **GPU transient** | 256 MiB (Stage 1/2) / 2944 MiB (Stage 3 STILL!) |
| **Latency** | Fast (direct shared memory, no serialization) |
| **Stability** | C++ process (needs death detection), Linux-only (no POSIX fallback), depends on #7771 |
| **ZeRO compatibility** | ZeRO-1/2 (chunked) + ZeRO-3 (STILL full materialization!) |
| **RTX 4090 verdict** | ★★★★★★★★ Best for ZeRO-2, ZeRO-3 STILL NOT viable until Stage 3 copyback implemented |

★★★★★★★★★ **How it works**:
- Same fundamental approach as CPU_Adam: optimizer on CPU, params on GPU
- Key difference: **chunked copyback** eliminates the 2944 MiB transient spike — BUT ONLY for Stage 1/2!
- GPU_transient = 4 × chunk_size_bf16 = 4 × 64 = 256 MiB regardless of model size (Stage 1/2 only!)
- ZeRO-3 STILL uses full fp32 materialization → 2944 MiB spike REMAINS → RTX 4090 OOM risk persists
- Native C++ optimizer process → faster IPC, more robust
- ZenGroup double-buffered: `grad[2], exp_avg[2], exp_avg_sq[2]` with [0]/[1] indexing → pipelined overlap
- PinnedThreadPool with AVX alignment (`kZenAdamAlign`) → deterministic numerics
- Linux-only (POSIX semaphores no fallback), depends on #7771 (Fix ZenFlow NaN)

★★★★★★★★★ **Chunked copyback math**:
```
Full materialization: GPU transient = partition_size_fp32 = (n_params × 4) / dp
  For 7B dp=1: 28 GiB → OOM on RTX 4090
  For 7B dp=4: 7 GiB → fits but wastes 7 GiB transient

Chunked copyback: GPU transient = n_buffers × chunk_size_bf16
  n_buffers = 4 (pipeline overlap)
  chunk_size = 128 MiB (fp32) → 64 MiB (bf16)
  Total = 4 × 64 = 256 MiB → fits on ANY GPU!
```

### 2.3 PyTorch CPUOffloadPolicy (#187620, Proposed)

| Aspect | Details |
|--------|---------|
| **Location** | CPU (in-process, PyTorch native) |
| **IPC** | None (same process, pin_memory async transfer) |
| **Copyback** | Async pin_memory transfer per shard |
| **GPU transient** | ~full model size (shard = identity on dp=1) |
| **Latency** | In-process → lowest latency |
| **Stability** | PyTorch native → most stable |
| **FSDP compatibility** | FSDP2 only |
| **RTX 4090 verdict** | NOT viable on dp=1 (shard=identity → full param resident) |

★★★★★★★★★ **Critical limitation for RTX 4090 dp=1**:
```
FSDP2 shard on dp=1 = identity function → each GPU holds FULL params
CPUOffloadPolicy: offload shard to CPU, keep resident on GPU
  shard_size = model_size / dp = model_size (dp=1!)
  resident_size = model_size → STILL takes full GPU memory!
  Result: >8B model → OOM on RTX 4090 regardless of offload
```

★★★★★★★★★ **Only viable for dp≥2**: When dp=4, shard_size = model_size/4 → resident_size = model_size/4 → fits on RTX 4090 for 7B model (~3.5 GiB resident).

**But**: dp≥2 requires multi-GPU → RTX 4090 only has 1 GPU → CPUOffloadPolicy NOT viable on RTX 4090!

### 2.4 verl FSDP1 + CPU Offload (Current Best Practice)

| Aspect | Details |
|--------|---------|
| **Location** | CPU (PyTorch native, in-process) |
| **IPC** | None (FSDP1 in-process offload) |
| **Copyback** | Per-unit summon via #6512 (dynamic FSDP unit discovery) |
| **GPU transient** | 6-8 GiB peak (per-unit summon, not full-model) |
| **Latency** | In-process → low latency |
| **Stability** | PyTorch FSDP1 → stable (FSDP2 has CPU leak #6468!) |
| **FSDP compatibility** | FSDP1 only (MUST NOT use FSDP2!) |
| **RTX 4090 verdict** | ★★★★★★★★ #1 BEST for dp=1 GRPO |

★★★★★★★★★ **How it works**:
- FSDP1 + param_offload=True: model weights split into FSDP units, offloaded to CPU
- Per-unit summon (#6512): only summon needed FSDP units for forward/backward → 6-8 GiB peak
- optimizer_offload=True: optimizer states on CPU
- bypass_mode: eliminates reference model (18Ψ→3.8Ψ)
- Combined: ~16.2 GiB peak → fits on RTX 4090 with 7.8 GiB margin

★★★★★★★★★ **Per-unit summon math** (#6512):
```
Old approach: whole-model summon before forward
  GPU peak = model_size_bf16 + activations + temp_buffers
  For 8B: 14 + 4 + 2 = 20 GiB → barely fits

Per-unit summon: summon each FSDP unit separately
  GPU peak = max_unit_size + activations + temp_buffers
  For 8B: 1.5 + 4 + 0.5 = 6 GiB → safe margin!
  Dynamic unit discovery → replaces 8 hard-coded prefixes → 60→6-8 GiB peak
```

---

## 3. Comparative Analysis

### 3.1 Feature Matrix

| Feature | CPU_Adam | ZenFlow | CPUOffloadPolicy | verl FSDP1+offload |
|---------|----------|---------|-------------------|--------------------|
| **dp=1 viable** | YES | YES | NO | YES ★★★★★★★★ |
| **GPU transient** | 2944 MiB | 256 MiB (S1/2) / 2944 MiB (S3!) | ~14 GiB | 6-8 GiB |
| **IPC overhead** | High | Low | None | None |
| **Process stability** | Python (fragile) | C++ (needs death detect, Linux-only) | Native | Native |
| **ZeRO/FSDP compat** | ZeRO-2/3 | ZeRO-1/2 chunked, ZeRO-3 STILL full! | FSDP2 only | FSDP1 only |
| **FSDP2 leak risk** | None | None | HIGH (#6468) | N/A (uses FSDP1) |
| **bypass_mode compat** | No (DeepSpeed) | No (DeepSpeed) | No (PyTorch) | YES ★★★★★★★★ |
| **LoRA compat** | ZeRO-3 blocked (#8072) | ZeRO-3 STILL blocked! | FSDP2 unknown | YES ★★★★★★★★ |
| **RTX 4090 #1 for GRPO** | #2.5 | #2 (ZeRO-2 chunked) | N/A | ★★★★★★★★ #1 |

### 3.2 Memory Budget Comparison (7B model, dp=1)

| Approach | Peak GPU (GiB) | Transient (MiB) | Margin (GiB) | Fits? |
|----------|----------------|-----------------|---------------|-------|
| CPU_Adam ZeRO-2 | ~16.2 | 2944 (during copyback) | ~6 | YES (but tight during copyback) |
| ZenFlow ZeRO-2 | ~16.0 | 256 (chunked, S1/2 only!) | ~7 | YES (improved margin) |
| ZenFlow ZeRO-3 | ~16.2 | 2944 (S3 STILL full!) | ~6 | YES but copyback RISKY |
| CPUOffloadPolicy | ~24+ (resident=full) | ~14 GiB | 0 | NO |
| verl FSDP1+offload | ~16.2 | 6-8 GiB (per-unit) | ~7.8 | YES ★★★★★★★★ BEST |

### 3.3 Memory Budget Comparison (27B model, dp=1)

| Approach | Peak GPU (GiB) | Fits? |
|----------|----------------|-------|
| CPU_Adam ZeRO-2 | ~54 (27×2 bf16) | NO (requires ZeRO-3 partitioning) |
| ZenFlow ZeRO-3 | ~8 (partition + 256 MiB) | YES with dp≥4 |
| CPUOffloadPolicy | ~54 | NO |
| verl FSDP1+offload+bypass+LoRA | ~16.2 (only LoRA on GPU) | YES ★★★★★★★★ |

★★★★★★★★★ **27B key insight**: LoRA+bypass is the ONLY approach that fits 27B on RTX 4090! LoRA adds ~0.4 GiB (rank=32) while base model stays in CPU-offloaded FSDP units. Per-unit summon means only 1-2 FSDP units on GPU at peak.

---

## 4. RTX 4090 GRPO Training Decision Tree

```
RTX 4090 GRPO Training (24 GiB, dp=1):

Model ≤ 4B:
  → FSDP1 + full params on GPU + CPU optimizer offload
  → Peak: ~10-12 GiB → comfortable fit

Model 4B-8B:
  → FSDP1 + CPU offload + per-unit summon (#6512)
  → Peak: ~16-18 GiB → fits with 6 GiB margin
  → bypass_mode = True (eliminates ref model)

Model 8B-27B:
  → verl CPPO+bypass+FSDP1+LoRA-32+CPU offload + per-unit summon
  → Peak: ~16.2 GiB → fits with 7.8 GiB margin
  → ★★★★★★★★ ONLY viable approach on dp=1!

MoE models (30B-A3B, 35B-A3B):
  → verl CPPO+bypass+FSDP1+LoRA-32+CPU offload + AutoEP
  → Only 3B active params → similar to 4B dense model
  → Peak: ~10-12 GiB → comfortable fit

DSV4 (685B):
  → NOT viable on RTX 4090 (343 GiB FP8 > 24 GiB)
  → Use DSV4-like models: Qwen3.5-35B-A3B instead
```

---

## 5. ZenFlow Integration Paths

### 5.1 DeepSpeed Path (Direct)

★★★★★★★★★ ZenFlow is a DeepSpeed PR → directly applicable to DeepSpeed-based training:
- ZeRO-3 + ZenFlow CPU offload → NOW VIABLE on RTX 4090
- Previously: ZeRO-3 = OOM risk due to 2944 MiB transient
- Now: 256 MiB transient → 11.5x reduction → ZeRO-3 viable
- But: ZeRO-2 still preferred for dp=1 (no partition overhead)

### 5.2 verl Path (Indirect)

★★★★★★★★★ verl uses FSDP1, not ZeRO → ZenFlow needs adaptation:
- FSDP1 doesn't have ZeRO-3 partition-level copyback
- But FSDP1 + CPU offload still has fp32→bf16 per-unit conversion
- Chunked copyback could reduce per-unit summon transient
- Integration requires: ZenFlow concepts adapted to FSDP's summon path

### 5.3 PyTorch Path (Future)

★★★★★★★★★ CPUOffloadPolicy could borrow ZenFlow's chunked copyback concept:
- CPUOffloadPolicy currently transfers entire shard at once
- Chunked transfer: reduce GPU transient from shard_size to chunk_size
- But: dp=1 shard=identity → chunking doesn't help (still full param)
- Only useful for dp≥2 → not relevant for RTX 4090

---

## 6. Key Conclusions

★★★★★★★★★ **RTX 4090 dp=1 GRPO training — CPU offload ranking** (★ UPDATED with Stage 1/2 only correction):

```
#1: verl CPPO+bypass + FSDP1 + CPU_offload + per-unit LoRA summon (#6512)
    → 16.2 GiB peak, bypass eliminates ref model, LoRA adds minimal overhead
    → BEST: bypass_mode + per-unit summon + LoRA = triple memory optimization

#2: DeepSpeed ZeRO-2 + ZenFlow CPU optimizer (after #8058 merges)
    → ~16 GiB peak, chunked copyback for Stage 1/2, faster IPC
    → ★★★★★★★★ Chunked copyback ONLY works for Stage 1/2 (confirmed by delock review)
    → Improvement: 5-15% faster per step, safer copyback for ZeRO-2

#2.5: DeepSpeed ZeRO-2 + CPU_Adam + LoRA-32
    → ~16-18 GiB peak, no partition overhead, well-tested
    → Risk: #8072 regression if accidentally use ZeRO-3

#3: DeepSpeed ZeRO-3 + ZenFlow CPU optimizer
    → ★★★★★★★★ STILL NOT viable! Stage 3 copyback NOT implemented → 2944 MiB spike REMAINS
    → Only viable AFTER Stage 3 chunked copyback is added (future work)
    → Risk: #8072 regression (per-policy dtype mismatch)

NOT viable: PyTorch CPUOffloadPolicy (#187620)
    → dp=1 shard=identity → full model resident → OOM for >8B
    → Only viable for dp≥2 → multi-GPU required

NOT viable: FSDP2 backend (verl)
    → #6468 CPU memory leak → 0.6-6.3 GiB/step → host OOM in ~8-22 steps
    → MUST use FSDP1 backend for long-running GRPO
```

★★★★★★★★★ **Bottom line**: verl CPPO+bypass+FSDP1+CPU_offload remains #1 BEST for RTX 4090. ZenFlow makes ZeRO-2 BETTER (chunked copyback for Stage 1/2) but ZeRO-3 is STILL NOT viable (Stage 3 copyback not implemented). CPUOffloadPolicy is NOT viable on dp=1.

---

## References

- DeepSpeed #8058 ZenFlow: notebook/projects/deepspeed-8058-zenflow-native-cpu-optimizer-reading.md
- DeepSpeed #8061 overlap_comm: notebook/projects/deepspeed-8061-overlap-comm-multi-stream-race-reading.md
- PyTorch #187620 CPUOffloadPolicy: notebook/projects/pytorch-187620-fsdp2-partial-offload-policy-reading.md
- verl #6512 per-unit summon: notebook/projects/verl-6512-per-unit-lora-summon-reading.md
- verl #6468 FSDP2 leak: notebook/projects/verl-6468-fsdp2-cpu-memory-leak-reading.md
- verl #6794 delta sync: notebook/projects/verl-6794-delta-weight-sync-reading.md
- RTX 4090 runbook: notebook/projects/rtx4090-grpo-training-runbook.md
- Algorithm→infra synthesis: notebook/synthesis/algorithm-to-infra-synthesis.md
- CUDA stream safety: notebook/fundamentals/cuda-stream-safety-cross-framework-pattern.md

---

*Created 2026-06-19. Cross-framework CPU offload comparison — synthesis of 4 approaches across 3 frameworks.*
