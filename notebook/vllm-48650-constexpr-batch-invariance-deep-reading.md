# vLLM #48650: tl.constexpr Batch-Invariance Bug — Deep Reading

**Date**: 2026-07-15 (Session 10, deep reading)
**Issue**: [vllm-project/vllm#48650](https://github.com/vllm-project/vllm/issues/48650)
**Author**: sbeurnier (systematic audit via Claude Code)
**Status**: OPEN (0 comments, 0 labels, untriaged)
**Significance**: P9 thesis validation — Triton constexpr batch-invariance is the SAME bug class as Inductor SM<90 fusion guard

---

## 1. Executive Summary

Several in-tree vLLM Triton kernels declare **runtime-varying, batch-derived values as `tl.constexpr`**. Triton constexpr values are baked into the JIT compilation cache key, so each new distinct value forces a full kernel recompile (hundreds of ms) at request time. This is the exact same mechanism as our P9 thesis: any JIT compilation system that keys its cache on batch-derived values creates batch-dependent kernels, causing performance degradation or TTFT stalls on the serving path.

The issue author (sbeurnier) established this bug class in triton-lang/triton#10872, where they measured warm-phase p99 TTFT dropping **682ms -> 316ms (= p50)** from demoting two constexpr params in OpenAI's `triton_kernels` topk implementation used in gpt-oss serving.

The vLLM audit found **7 confirmed instances** across attention, LoRA, MoE, and sampling kernels. The highest-impact instance (`MAX_MM_RANGES`) recompiles one of the largest kernels in the entire vLLM codebase (1189 lines) per distinct per-batch image count, unbounded.

---

## 2. Background: Triton JIT Cache Key Mechanism

### How Triton JIT Works

Triton kernels are compiled via JIT (Just-In-Time compilation). When a kernel is launched, Triton computes a **cache key** from:
1. The kernel function's AST (source code)
2. All `tl.constexpr` parameter values
3. Specialization hints (divisibility, alignment, etc.)

If the cache key matches an existing compiled binary, the kernel launches immediately (~0.1ms). If it doesn't match, Triton must **recompile the entire kernel** from scratch (~300-700ms for complex kernels, sometimes >1s for large kernels on first cold hit).

### The Bug Pattern

When a `tl.constexpr` parameter receives a **batch-derived value** (i.e., a value that changes depending on the composition of requests in a batch), each new distinct value creates a new cache key, forcing a recompile. In LLM serving, batch composition changes continuously (different numbers of images, different token counts, different logprob requests), so these recompiles happen **repeatedly on the serving hot path**.

Key insight from triton#10872 measurements:
- Same 32-bucket (cache hit): 0.4ms
- New bucket (warm compile, disk-cache hit): 1.8ms
- Truly new bucket (cold compile): **732.6ms** (~1s stall)

Under TP, all ranks hit the cache miss simultaneously, and other GPUs **spin in the collective** while one rank compiles — amplifying the stall.

---

## 3. Upstream Precedent: triton-lang/triton#10872/#10875/#10884

### triton#10872: topk bitmatrix strides as constexpr

**File**: `triton_kernels/topk_details/_topk_forward.py`
**Bug**: `stride_rm: tl.constexpr, stride_rn: tl.constexpr` where stride values derive from `cdiv(n_rows, 32) * 32` — n_rows = per-batch token count.

**Measurement**: In SGLang gpt-oss serving on H100, warm-phase p99 TTFT = 682ms. After demoting the two strides to runtime args: p99 = 316ms (= p50). **682ms -> 316ms, 54% reduction**.

**Fix** (triton#10875): Simply remove `tl.constexpr` annotations:
```diff
-  USE_PROVIDED_INDX: tl.constexpr, Bits, stride_rm: tl.constexpr, stride_rn: tl.constexpr,
+  USE_PROVIDED_INDX: tl.constexpr, Bits, stride_rm, stride_rn,
```
Result: **1 compile total**, then ~0.1ms across 13 previously-unseen lengths. Outputs bit-identical.

### triton#10884: p_matmul NUM_SMS as constexpr

**Bug**: `_p_matmul`'s `NUM_SMS: tl.constexpr` receives the launch grid size, which below SM saturation equals `batch_size * cdiv(M, block_m) * grid_n * split_k` — a function of runtime token count. Persistent matmul recompiles every `block_m` step of M in the small-batch regime where persistent mode is selected.

**Fix**: Demote NUM_SMS to runtime argument (or use `tl.num_programs(0)` which equals it by construction).

---

## 4. Kernel Instance Deep Analysis

### 4.1 HIGHEST IMPACT: `MAX_MM_RANGES` — Unified Attention

**Kernel**: `kernel_unified_attention`
**Source**: `vllm/v1/attention/ops/triton_unified_attention.py:222`
**Helper**: `vllm/v1/attention/ops/triton_attention_helpers.py:355`
**Also in**: `vllm/v1/attention/ops/int4_per_token_head.py:360`

**Constexpr declaration**:
```python
MAX_MM_RANGES: tl.constexpr,  # int  (line 222)
```

**Launch site** (`triton_unified_attention.py:1133`):
```python
MAX_MM_RANGES=max_mm_ranges,
```

**Producer** (`triton_unified_attention.py:910-914`):
```python
max_mm_ranges = 0
if mm_prefix_range is not None:
    if mm_prefix_range.ndim == 3:
        max_mm_ranges = mm_prefix_range.shape[1]
```

**Root producer** (`vllm/v1/attention/backends/utils.py:69`):
```python
max_ranges = max(len(r) for r in range_lists)
```
Where `range_lists` are the per-request multimodal prefix ranges (image count). This is rebuilt **every scheduler step** from the batch composition.

**Kernel usage** (`triton_attention_helpers.py:355-360`):
```python
for i in range(MAX_MM_RANGES):
    range_start = tl.load(
        mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
    )
    range_end = tl.load(
        mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
    )
```

The `MAX_MM_RANGES` is used as an **unrolled loop bound** — Triton expands `for i in range(MAX_MM_RANGES)` into `MAX_MM_RANGES` separate iterations at compile time. This means:
- MAX_MM_RANGES=1 and MAX_MM_RANGES=2 produce **different compiled binaries**
- A request with 1 image joining a batch that previously had 2 images triggers a full recompile
- The kernel is **1189 lines** long (one of the largest in the entire codebase) — recompile cost is maximized

**Affected models**: All `is_mm_prefix_lm` models (Gemma3-style) under TRITON_ATTN backend (default on ROCm, available on CUDA). Same pattern in `int4_per_token_head.py:360`.

**Impact assessment**:
- **Unbounded**: Image count can be any positive integer, so cache keys grow without bound
- **Hot path**: unified attention is called on every decode/prefill step
- **ROCm critical**: TRITON_ATTN is the default on ROCm, so this affects all AMD deployments
- **CUDA path**: TRITON_ATTN is available on CUDA (used when FlashInfer is unavailable, e.g. for FP8 KV cache)

**Proposed fix**:
1. **Runtime loop bound** (masked, non-unrolled): Replace `for i in range(MAX_MM_RANGES)` with a runtime loop bound + mask for invalid ranges. This avoids recompiles but may slightly reduce peak throughput for large image counts.
2. **Pow2-bucket the range count**: Pad MAX_MM_RANGES to next power of 2 (1, 2, 4, 8, 16). Reduces cache key variety from O(N) to O(log N), but still has residual recompiles.
3. **Fixed small set**: Clamp to a fixed maximum (e.g. 8) with masking. One compile, zero recompiles, minimal masking overhead.

---

### 4.2 `prefill_tokens_with_context` — Merge Attention States

**Kernel**: `merge_attn_states_kernel`
**Source**: `vllm/v1/attention/ops/triton_merge_attn_states.py:71`
**Launch**: `triton_merge_attn_states.py:39`

**Constexpr declaration**:
```python
prefill_tokens_with_context: tl.constexpr,  (line 71)
```

**Launch code**:
```python
if prefill_tokens_with_context is None:
    prefill_tokens_with_context = num_tokens  # per-batch total token count
merge_attn_states_kernel[(num_tokens, num_query_heads)](
    ...
    prefill_tokens_with_context,  # passed as constexpr
    ...
)
```

**Batch dependency**: `prefill_tokens_with_context` defaults to `num_tokens` (total tokens in batch), which changes with every batch composition.

**Kernel usage**:
```python
prefix_mask = token_idx < prefill_tokens_with_context  # line ~80
```

This is a simple comparison — it does NOT need to be constexpr. The value is used as a runtime mask boundary, not as an unrolled loop bound or tile size.

**Critical observation**: The CUDA C++ twin (`_merge_attn_states` in C++ backend) correctly takes this as a **runtime argument**. The Triton port incorrectly made it constexpr — it just needs the same treatment as the C++ version.

**Proposed fix**: Demote to runtime argument (remove `tl.constexpr` annotation). The comparison `token_idx < prefill_tokens_with_context` works identically with a runtime value. This is the **easiest fix** among all instances.

---

### 4.3 `_pack_seq_kernel` N / `_unpack_seq_triton_kernel` B — Pure Cache-Key Pollution

**Kernel**: `_pack_seq_kernel`, `_unpack_seq_triton_kernel`
**Source**: `vllm/v1/attention/ops/common.py:265,391`

**Constexpr declarations**:
```python
# _pack_seq_kernel (line 265)
N: tl.constexpr,
D: tl.constexpr,
Lmax: tl.constexpr,

# _unpack_seq_triton_kernel (line 391)
B: tl.constexpr,
Lmax: tl.constexpr,
D: tl.constexpr,
```

**Launch code** (`pack_seq_triton` function):
```python
N, D = x.shape  # N = total tokens (per-batch)
B = lengths.numel()  # B = number of requests (per-batch)
Lmax = int(lengths.max().item())  # Lmax = max decode length (per-batch)
_pack_seq_kernel[grid](
    x_reshaped, out, lengths.int(),
    N, D, Lmax,  # all as constexpr
    ...
)
```

**Batch dependency**: `N` = total tokens, `B` = request count, `Lmax` = max sequence length — all change with every batch.

**Critical observation**: **These constexpr values are UNUSED in the kernel bodies!**

In `_pack_seq_kernel`, the kernel uses:
- `pid_b = tl.program_id(0)` (batch index from grid)
- `tl.load(lengths_ptr + pid_b)` (reads actual length from memory)
- `D` is used in pointer arithmetic but could be a runtime arg

The `N` parameter (total token count) is **never referenced** in the kernel body — it's pure cache-key pollution. Same for `B` in `_unpack_seq_triton_kernel`.

`Lmax` is used in `t_mask = off_t < Lmax`, which works perfectly as a runtime comparison.

**Proposed fix**:
1. Remove `N` from `_pack_seq_kernel` entirely (unused)
2. Remove `B` from `_unpack_seq_triton_kernel` entirely (unused)
3. Demote `Lmax`, `D` to runtime arguments
4. This is the **lowest-risk fix** — removing unused params has zero performance impact

---

### 4.4 `NUM_KV_SPLITS` — Triton MLA Decode (Already Partially Mitigated)

**Kernel**: `_fwd_grouped_kernel_stage1` and `_fwd_grouped_kernel_stage2`
**Source**: `vllm/v1/attention/ops/triton_decode_attention.py:95,308,587`

**Constexpr declarations**:
```python
NUM_KV_SPLITS: tl.constexpr,  # line 95, 308, 587
```

**Launch code** (`triton_decode_attention.py:514`):
```python
NUM_KV_SPLITS = num_kv_splits
grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)
_fwd_grouped_kernel_stage1[grid](
    ...
    NUM_KV_SPLITS=NUM_KV_SPLITS,
    ...
)
```

**Bucketing function** (`vllm/v1/attention/backends/mla/triton_mla.py:41-47`):
```python
_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2

def _compute_num_kv_splits(max_seq_len: int, sm_count: int) -> int:
    ideal_splits = triton.next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
    max_splits = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
    return min(ideal_splits, max_splits)
```

**Batch dependency**: `_compute_num_kv_splits` takes `max_seq_len` (the maximum KV sequence length in the current batch), which grows as context accumulates across requests.

**Already mitigated**: Pow2 bucketing limits the key space to powers of 2 (1, 2, 4, 8, 16, 32, 64, ...). However, as context grows from 512 tokens to 128K tokens across the serving lifetime, NUM_KV_SPLITS ratchets through ~10 distinct values, each triggering a recompile on the decode hot path.

**Batch invariance override** (`triton_mla.py:212-214`):
```python
if is_batch_invariant:
    num_kv_splits = 1
```

When `VLLM_BATCH_INVARIANT=1` is set, `NUM_KV_SPLITS=1` (single compile, no recompiles). This is the current escape hatch but sacrifices split-KV parallelism.

**Proposed fix**:
1. **Clamp to fixed small set**: Like `turboquant`, cap NUM_KV_SPLITS to a fixed small set (e.g. {1, 2, 4, 8, 16}) and use runtime masking for intermediate values.
2. **Demote to runtime**: The kernel uses `kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)` which works with runtime values. However, NUM_KV_SPLITS also determines the grid size, so demotion requires restructuring the launch.
3. **Current best**: The pow2 bucketing is a reasonable partial mitigation; full fix requires demotion or clamping.

---

### 4.5 `_count_expert_num_tokens` BLOCK_SIZE — Trivial Fix

**Kernel**: `_count_expert_num_tokens`
**Source**: `vllm/model_executor/layers/fused_moe/utils.py:42-49`

**Constexpr declaration**:
```python
BLOCK_SIZE: tl.constexpr,  # line 49
```

**Launch code** (`fused_moe/utils.py:97-98`):
```python
BLOCK_SIZE = min(topk_ids.numel(), 1024)
BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
```

**Batch dependency**: `topk_ids.numel()` = total tokens * topk, changes with every batch.

**Kernel usage**: `tl.arange(0, BLOCK_SIZE)` for offsets, `tl.cdiv(topk_numel, BLOCK_SIZE)` for loop bound. A fixed BLOCK_SIZE=1024 works identically — the masking `mask = offsets < (topk_numel - x * BLOCK_SIZE)` already handles shorter sequences.

**Proposed fix**: Replace with constant `BLOCK_SIZE = 1024`. This is a **1-line fix**:
```python
# Before:
BLOCK_SIZE = min(topk_ids.numel(), 1024)
BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
# After:
BLOCK_SIZE = 1024
```

The kernel already masks loads for positions beyond `topk_numel`, so using 1024 for all cases is safe and avoids all recompiles.

---

### 4.6 LoRA shrink/expand SPLIT_K / BLOCK_K — Batch-Dependent Config

**Kernel**: `_lora_shrink_kernel`, `_lora_expand_kernel`
**Source**: `vllm/lora/ops/triton_ops/lora_shrink_op.py:44-49`, `vllm/lora/ops/triton_ops/lora_expand_op.py:45-51`

**Constexpr declarations** (shrink):
```python
BLOCK_M: tl.constexpr,
BLOCK_N: tl.constexpr,
BLOCK_K: tl.constexpr,
EVEN_K: tl.constexpr,
SPLIT_K: tl.constexpr,
```

**Config derivation** (`vllm/lora/ops/triton_ops/utils.py:221-228`):
```python
if op_type == "shrink":
    split_k = 64 if batch < 128 else 8
    if is_batch_invariant:
        split_k = 1
    default = {
        "block_m": 32,
        "block_n": 16,
        "block_k": 256 if batch < 128 else 32,
        "split_k": split_k,
        ...
    }
```

**Batch dependency**:
- `SPLIT_K` flips between 64 and 8 at the `batch < 128` boundary
- `BLOCK_K` flips between 256 and 32 at the same boundary
- These are **config choices** derived from the current batch size, passed as constexpr

**Impact**: LoRA kernels recompile when batch size crosses the 128-token boundary. In GRPO serving, batch sizes typically stay under 128 for decode steps but exceed 128 for prefill steps — so the transition happens repeatedly.

**Connection to #48590**: The LoRA NaN bug on Hopper (#48590) is in `lora_expand` with `block_n=128`. PR #48638 fixes it by dropping `block_n` to 32 on Hopper only. This is the **same kernel family** — the constexpr config system creates both NaN bugs (via bad tiling choices) and recompile bugs (via batch-dependent configs).

**Batch invariance override**: When `VLLM_BATCH_INVARIANT=1`, `split_k = 1` — same escape hatch as NUM_KV_SPLITS.

**Proposed fix**:
1. **Fixed config per architecture**: Use architecture-specific configs that don't vary with batch size (e.g., always use SPLIT_K=8, BLOCK_K=32). Slightly less optimal for small batches but eliminates recompiles.
2. **Two-stage warmup**: Pre-compile both the "small batch" and "large batch" configs during warmup, then select at runtime. Requires Triton support for runtime config selection (not currently available).
3. **Demote BLOCK_K/SPLIT_K**: These feed into tiling decisions that Triton can optimize via runtime divisibility/alignment specialization rather than constexpr.

---

### 4.7 `_fill_logprob_token_ids_kernel` NUM_TOPK — Per-Batch Logprob Requests

**Kernel**: `_fill_logprob_token_ids_kernel`
**Source**: `vllm/v1/worker/gpu/sample/logprob.py:182-196`

**Constexpr declarations**:
```python
NUM_TOPK: tl.constexpr,   # line 196
PADDED_COLS: tl.constexpr,  # line 197
```

**Launch code** (`logprob.py:158-159`):
```python
NUM_TOPK=num_logprobs,
PADDED_COLS=triton.next_power_of_2(num_cols),
```

Where `num_logprobs` = max requested logprobs across the batch, and `num_cols = max(num_logprobs, max_per_req_token_ids)`.

**Batch dependency**: Different requests in a batch can request different numbers of top-k logprobs (0, 5, 10, 20, etc.). The kernel takes the **per-batch maximum** as constexpr, so each new distinct value triggers a recompile.

**Kernel usage**:
```python
valid = col < NUM_TOPK  # simple comparison, works with runtime arg
```

PADDED_COLS feeds into `tl.arange(0, PADDED_COLS)` which works with runtime values via masking.

**Proposed fix**:
1. **Demote NUM_TOPK**: Remove `tl.constexpr` — the comparison `col < NUM_TOPK` works identically with a runtime value.
2. **Pad to fixed set**: Bucket logprob requests to {5, 10, 20, 50} and pre-compile during warmup.
3. **PADDED_COLS**: Could remain constexpr if bucketed to pow2 set, or demoted entirely.

---

## 5. P9 Thesis Connection

### The P9 Thesis (PyTorch Inductor SM<90 Fusion Guard)

Our P9 thesis (PyTorch #184119, Jackie2049/pytorch PR #1):
- **Problem**: PyTorch Inductor creates SM<90 prologue fusions that bake batch-dependent shapes into compiled kernels. The `choices.py` scheduler decides to fuse fp8-to-bf16 conversion as a prologue on SM<90 GPUs (because SM<90 lacks native fp8 compute). This fusion makes the kernel's compiled shape dependent on batch dimensions.
- **Fix**: Add a guard in `choices.py`: `props.major < 9` → block prologue fusion. Kernels become batch-invariant, no recompiles per batch size variation.
- **Result**: 5-line fix, validated by NVIDIA PR #184119 (official SM89 fp8 guard).

### How #48650 Validates P9

| Dimension | P9 (Inductor) | #48650 (Triton constexpr) |
|-----------|---------------|---------------------------|
| JIT system | PyTorch Inductor | Triton JIT compiler |
| Cache key mechanism | Compiled kernel shape | constexpr parameter values |
| Batch-dependent input | Tensor shape (batch dim) | Runtime values (image count, token count, etc.) |
| Effect | Recompile per batch size | Recompile per batch composition |
| Stall measurement | Not measured in vLLM (stalls absorbed by aot_eager) | 682ms -> 316ms p99 TTFT reduction (triton#10872) |
| Fix principle | Block batch-dependent fusion | Demote constexpr to runtime arg |
| Fix granularity | Architecture guard (SM<90) | Per-kernel annotation change |

**The core insight is identical**: Both P9 and #48650 demonstrate that **any JIT compilation system that keys its cache on batch-derived values creates batch-dependent kernels**. The mechanism differs (fusion vs constexpr), but the root cause and the fix principle are the same.

### Validation Chain (4 independent validations of the batch-invariance thesis)

1. **P9 original** (#184119): Inductor SM<90 prologue fusion guard — 5-line fix
2. **#48613**: GDN batch-invariance broken for Qwen3.5/3.6 — `supports_batch_invariance()=True` insufficient
3. **#46085**: aot_eager = batch-invariant BY DESIGN — Dynamo+AOTAutograd WITHOUT Inductor avoids fusion entirely
4. **#48650**: Triton constexpr batch-invariance — systematic audit of 7 kernel instances

This is now the **4th independent validation** of the batch-invariance thesis.

---

## 6. Cross-Framework Pattern Analysis

### The Universal Batch-Invariance Bug Pattern

| System | Cache Key Source | Batch Dependency | Recompile Trigger | Fix |
|--------|-----------------|------------------|-------------------|-----|
| **Triton JIT** | `tl.constexpr` values | Batch-derived runtime values | New constexpr value | Demote to runtime arg |
| **PyTorch Inductor** | Compiled kernel shape | Batch-dependent fusion prologues | New batch shape | Block fusion (SM<90 guard) |
| **CUDA Graph** | Captured graph operations | Batch size at capture time | Batch size change | Pre-compile multiple graphs |
| **JAX/XLA** | Shape specialization | Dynamic shape dimensions | New shape signature | Shape polymorphism / padding |
| **torch.compile (aot_eager)** | AOTAutograd graph | No Inductor → no shape fusion | None (by design) | Already batch-invariant |

### Pattern Family Members

| Bug | System | Mechanism | Status |
|-----|--------|-----------|--------|
| P9 (#184119) | PyTorch Inductor | SM<90 prologue fusion -> batch-dependent | Jackie2049/pytorch PR #1 (OPEN) |
| #48650 | vLLM Triton | tl.constexpr batch-derived -> recompiles | OPEN (0 comments) |
| triton#10872 | Triton kernels | bitmatrix strides constexpr -> ~1s TTFT stalls | triton#10875 fix (OPEN) |
| triton#10884 | Triton kernels | NUM_SMS constexpr -> persistent matmul recompiles | OPEN |
| #46085 | vLLM | aot_eager = batch-invariant BY DESIGN | Documented |
| #48613 | vLLM | GDN batch-invariance broken for Qwen3.5/3.6 | OPEN |
| #45601 | vLLM | Warm up all top-k/top-p Triton sampler specializations | OPEN |
| CUDA Graph | vLLM/SGLang | graph captured at batch N, replay only at N | Multiple captures needed |
| vLLM-Ascend #48656 | vLLM | SP-MoE auto-enabled at dp=1 -> -24% KV cache | #48657 override (DRAFT) |

### Why This Pattern Is Universal

All JIT compilation systems (Triton, Inductor, XLA, NVCC, HIPCC) face the same fundamental trade-off:
1. **Compile-time specialization** enables optimizations (loop unrolling, tiling, register allocation) that depend on specific values
2. **Runtime generality** avoids recompiles but may sacrifice some peak throughput
3. **The bug**: When the specialized value is batch-derived, you get per-batch recompiles instead of per-architecture specialization

The correct resolution is always: **specialize on architecture/GPU properties (which are fixed), not on batch composition (which varies)**.

---

## 7. RTX 4090 / H20-3e Impact Analysis

### RTX 4090 (SM89, Ada Lovelace)

**Why RTX 4090 is particularly affected**:

1. **No FlashInfer on SM89**: FlashInfer requires SM90+ (Hopper). On RTX 4090, vLLM falls back to TRITON_ATTN backend, which means ALL the Triton constexpr instances apply directly.

2. **LoRA GRPO critical path**: GRPO training on RTX 4090 uses LoRA adapters. The `lora_shrink/lora_expand` kernels are on every training step's hot path. SPLIT_K/BLOCK_K flip at batch<128 creates recompiles at the prefill/decode transition.

3. **Connection to #48590**: The LoRA NaN bug on Hopper is `block_n=128` in lora_expand. On RTX 4090 (SM89), Triton uses `aot_eager` mode (no Inductor), so the NaN bug doesn't manifest. BUT the constexpr batch-invariance bug DOES manifest — different batch sizes cause different constexpr configs -> recompiles.

4. **GRPO batch composition variability**: GRPO batches have variable composition (different numbers of images, different token counts, different logprob requests). This maximizes the number of distinct constexpr values -> maximizes recompiles.

**Current mitigation**: `aot_eager=True` + `VLLM_BATCH_INVARIANT=1` makes Triton kernels batch-invariant by design (confirmed in #46085). However:
- `VLLM_BATCH_INVARIANT=1` forces `SPLIT_K=1` for LoRA and `NUM_KV_SPLITS=1` for MLA decode, sacrificing throughput
- `aot_eager` avoids Inductor fusion but doesn't help Triton constexpr recompiles (Triton is a separate JIT system)

**Required fix for RTX 4090**: Demote batch-derived constexprs in all 7 kernel instances. This is complementary to P9 (which fixes Inductor) — both are needed for full batch-invariance.

### H20-3e (SM90a, Hopper variant)

**Why H20-3e is also affected**:

1. **FlashInfer available**: On SM90a, FlashInfer is available, so TRITON_ATTN is NOT the default. The attention constexpr instances (#4.1, #4.2, #4.4) are less critical because FlashInfer replaces them.

2. **LoRA still on Triton**: Even with FlashInfer attention, LoRA shrink/expand kernels are Triton-based. #48590 NaN on sm_90 + #48650 constexpr batch-invariance both apply.

3. **MoE and sampling still Triton**: `_count_expert_num_tokens` (MoE) and `_fill_logprob_token_ids_kernel` (sampling) are Triton regardless of attention backend.

4. **NUM_KV_SPLITS for MLA decode**: TritonMLA decode uses Triton kernels with NUM_KV_SPLITS constexpr. On H20-3e with MLA models (DSv4, GLM-5), this is on the decode hot path.

**Priority for H20-3e**: LoRA (#4.6), MoE (#4.5), sampling (#4.7), MLA decode (#4.4) — these are Triton-based regardless of attention backend.

---

## 8. Proposed Fix Strategy

### Immediate Fixes (Easy, Low Risk)

| Instance | Fix | LOC Change | Risk |
|----------|-----|-----------|------|
| `prefill_tokens_with_context` | Demote to runtime arg | -1 annotation | Zero (C++ twin uses runtime) |
| `_pack_seq_kernel` N | Remove unused param | -1 param | Zero (unused in body) |
| `_unpack_seq_kernel` B | Remove unused param | -1 param | Zero (unused in body) |
| `_count_expert_num_tokens` BLOCK_SIZE | Constant 1024 | -2 lines | Low (masking handles it) |
| `_fill_logprob_token_ids` NUM_TOPK | Demote to runtime arg | -1 annotation | Low (simple comparison) |

### Medium Fixes (Requires Testing)

| Instance | Fix | LOC Change | Risk |
|----------|-----|-----------|------|
| `MAX_MM_RANGES` | Masked runtime loop bound | ~10-20 lines | Medium (changes unrolled loop pattern) |
| `MAX_MM_RANGES` | Pow2-bucket (1,2,4,8) | ~5 lines | Low (still unrolled, limited keys) |
| `Lmax` (pack/unpack) | Demote to runtime arg | -1 annotation | Low (comparison works) |
| `NUM_KV_SPLITS` | Clamp to fixed set {1,2,4,8,16} | ~5 lines | Medium (may affect split-KV perf) |
| LoRA SPLIT_K/BLOCK_K | Fixed config per arch | ~10 lines | Medium (may affect small-batch perf) |

### Architecture-Level Fix (Complementary to P9)

The universal fix principle: **All JIT cache keys should be independent of batch-derived values. Specialize only on architecture/GPU properties.**

This requires:
1. **Triton**: Audit all `tl.constexpr` params for batch dependency, demote or remove batch-derived ones
2. **Inductor**: P9 guard blocks SM<90 prologue fusion (already implemented)
3. **CUDA Graph**: Pre-compile multiple batch sizes during warmup (already done in vLLM via chunked prefill)
4. **aot_eager**: Already batch-invariant by design (no Inductor -> no fusion)

---

## 9. Summary and Conclusions

### Key Findings

1. **7 confirmed instances** of batch-derived constexpr in vLLM Triton kernels, ranging from trivial (unused params) to critical (MAX_MM_RANGES unrolled loop in 1189-line kernel)

2. **Measured impact**: 682ms -> 316ms p99 TTFT reduction in triton#10872 from demoting just 2 constexpr params — the mechanism and cost are established

3. **P9 thesis validation**: #48650 is the 4th independent validation that batch-dependent JIT cache keys cause performance degradation in LLM serving

4. **RTX 4090 critical**: TRITON_ATTN is the fallback on SM89, making all attention constexpr instances directly applicable. LoRA instances apply regardless of GPU.

5. **5 easy fixes** available (demote/remove unused constexprs) with zero or near-zero risk

### Open Questions

1. Will vLLM maintainers accept bulk constexpr demotion, or prefer per-instance PRs?
2. What is the performance impact of masked loop bounds for MAX_MM_RANGES vs unrolled?
3. Should Triton itself provide a `tl.runtime_const` annotation that enables optimization but doesn't key the cache? (This would be the ideal solution — specialize on divisibility/alignment without cache-key pollution)
4. How does this interact with `VLLM_BATCH_INVARIANT=1`? Should the constexpr params be automatically demoted when BIC mode is enabled?

### Relationship to Existing vLLM Issues

- **#48590 (LoRA NaN on Hopper)**: Same kernel family (lora_expand). The NaN bug is a *correctness* issue; #48650 is a *performance* issue. Both stem from Triton constexpr config choices.
- **#48613 (GDN batch-invariance broken)**: Same pattern family — batch-invariance violated. GDN is FlashInfer-based; #48650 is Triton-based.
- **#45601 (Warm up sampler kernel specializations)**: Partially addresses NUM_TOPK recompiles by pre-compiling during warmup, but doesn't fix the root cause (constexpr annotation).
- **#48656 (SP-MoE auto-enabled)**: Not constexpr-related, but same serving-path performance concern (unwanted automatic behavior that hurts dp=1 deployments).

---

## 10. References

- **vLLM #48650**: [Issue](https://github.com/vllm-project/vllm/issues/48650) — tl.constexpr batch-invariance audit
- **triton #10872**: [Issue](https://github.com/triton-lang/triton/issues/10872) — topk bitmatrix strides constexpr
- **triton #10875**: [PR](https://github.com/triton-lang/triton/pull/10875) — Fix: demote strides
- **triton #10884**: [Issue](https://github.com/triton-lang/triton/issues/10884) — p_matmul NUM_SMS constexpr
- **vLLM #48590**: LoRA NaN on Hopper sm_90 (block_n=128)
- **vLLM #48638**: Fix PR for #48590 (block_n=32 on Hopper)
- **vLLM #48613**: GDN batch-invariance broken for Qwen3.5/3.6
- **vLLM #46085**: aot_eager = batch-invariant BY DESIGN
- **vLLM #45601**: Warm up all top-k/top-p Triton sampler specializations
- **vLLM #48656**: SP-MoE auto-enabled at dp=1
- **PyTorch #184119**: SM89 fp8 guard (validates P9 thesis)
- **Jackie2049/pytorch PR #1**: P9 Inductor SM<90 Fusion Guard
- **PyTorch #187636**: autotune_at_compile_time=False by default (complements P9)
