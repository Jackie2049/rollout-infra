# vLLM PR #43914: Triton FP8 KV Cache SM89+ Enforcement — Deep Reading Note

> PR #43914: "Remove redundant Triton KV cache dtype asserts and enforce architectural support (fp8 >= sm89)"
> Author: mikekg | Reviewer: mgoin (approved)
> Merged: 2026-06-15 (v0.23.0 era)
> Files: `triton_attn.py` (+20/-0), `triton_reshape_and_cache_flash.py` (+13/-39)

---

## 1. What This PR Changes: The Three Paths

### 1.1 Removes Redundant Asserts (Cleanup)

Before #43914, `triton_reshape_and_cache_flash.py` had verbose dtype-list asserts that checked:
```python
# REMOVED: redundant assert
assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
    torch.float8_e4m3fn, torch.float8_e5m2,
    torch.uint8, torch.float8_e4m3fnuz,
]
```

These asserts were redundant because `_is_supported_kv_cache_dtype()` already validated the dtype string before the kernel launch. The uint8 assert was also removed because Triton now handles FP8 storage through view casting, not direct uint8 operations.

### 1.2 Reworks `_is_supported_kv_cache_dtype` (Two-Layer Gate)

**Before**: Simple string-based check that accepted any recognized dtype regardless of hardware capability:
```python
def _is_supported_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _NATIVE_KV_CACHE_DTYPES or is_quantized_kv_cache(kv_cache_dtype)
```

**After**: Two-layer capability gate:
```python
def _is_supported_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    # Layer 1: dtype must be recognized
    if not (kv_cache_dtype in _NATIVE_KV_CACHE_DTYPES or is_quantized_kv_cache(kv_cache_dtype)):
        return False
    # Layer 2: hardware must support the dtype's compute requirements
    if kv_cache_dtype.startswith("fp8"):
        return current_platform.has_device_capability(89)  # fp8e4nv requires SM89+
    if kv_cache_dtype == "bfloat16":
        return current_platform.has_device_capability(80)   # bf16 requires SM80+
    return True  # int8_per_token_head and nvfp4 do NOT lower to fp8e4nv
```

**Key insight**: `int8_per_token_head` and `nvfp4` do NOT lower to `fp8e4nv` and remain allowed on ALL GPUs. This means INT8 KV continues to work on SM75+, regardless of FP8 hardware support.

### 1.3 Adds Early Guard in TritonAttentionImpl.__init__

```python
cap = current_platform.get_device_capability()
cap_str = cap.as_version_str() if cap is not None else "unknown"
dev = current_platform.get_device_name()

if self.kv_cache_dtype.startswith("fp8") and not (
    current_platform.has_device_capability(89)
):
    suggested = "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
    raise ValueError(
        f"FP8 KV cache is not supported by the Triton attention backend "
        f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
        f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
    )

if self.kv_cache_dtype == "bfloat16" and not (
    current_platform.has_device_capability(80)
):
    raise ValueError(
        f"bfloat16 KV cache is not supported on {dev} (compute capability "
        f"{cap_str}); bfloat16 requires SM80+. Re-run with "
        f"--kv-cache-dtype float16."
    )
```

This guard replaces the opaque `type fp8e4nv not supported in this architecture` crash that appeared deep in Triton autotuning with a clear, actionable `ValueError` that suggests an appropriate alternative dtype.

---

## 2. Why FP8 KV Works on SM89 via Triton (but NOT via FlashInfer)

### 2.1 The FlashInfer FP8 Limitation on SM89

FlashInfer's FP8 KV cache path requires SM90+ (Hopper/B200) because FlashInfer uses **hardware FP8 tensor cores** for attention computation. On Hopper:
- FlashInfer performs attention in the FP8 domain (Q, K, V all FP8)
- This requires the Hopper TMA + FP8 tensor core instructions that don't exist on SM89

On SM89 (Ada Lovelace / RTX 4090 / L4):
- FP8 storage (fp8e4m3fn) IS supported natively — the GPU can store and load FP8 data
- FP8 tensor core attention IS NOT supported — the GPU cannot perform FP8 matmuls in attention

This is why compressed-tensors models with `kv_cache_scheme` overriding to `fp8` crash on SM89 (issue #44879, PR #45038): FlashInfer tries to use its FP8 attention kernels which don't exist on SM89.

### 2.2 The Triton FP8 KV Path on SM89 — How It Works

Triton handles FP8 KV cache differently from FlashInfer:

**Storage (reshape_and_cache)**:
```python
# In reshape_and_cache_kernel_flash (Triton JIT kernel):
if FP8_KV_CACHE:
    # tl.store will do the correct implicit cast to fp8,
    # based on the key_cache_ptr.dtype.element_ty
    key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
```

The Triton kernel:
1. Loads KV in their original dtype (BF16/FP16)
2. Divides by the scale factor (`k_scale` / `v_scale`)
3. Triton's `tl.store` performs **implicit cast** to fp8e4nv (float8_e4m3fn)
4. This works on SM89 because fp8e4m3fn store operations ARE natively supported

**Retrieval / Dequantization (_cast_kv_tile in unified_attention)**:
```python
@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed."""
    if KV_QUANT_MODE == 1:  # FP8_PER_TENSOR
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)  # Stay in FP8 if Q is also FP8
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
        # fp8 → fp32 × scale → bf16/fp16  (software dequantization)
    return data.to(Q.dtype)  # NONE, INT8/FP8_PER_TOKEN_HEAD: plain cast
```

The Triton unified attention kernel:
1. Loads KV tiles from the fp8e4m3fn KV cache
2. Dequantizes: FP8 → FP32, multiply by scale, cast to query dtype (BF16/FP16)
3. Performs attention in the query dtype (BF16/FP16), NOT in FP8
4. This works on SM89 because Triton does the dequantization in **software** — no FP8 tensor cores needed

### 2.3 The KV_QUANT_MODE Dispatch

| Mode | Value | Storage | Dequant in Attention | SM89 Support |
|------|-------|---------|---------------------|-------------|
| NONE | 0 | BF16/FP16 native | No dequant | Yes |
| FP8_PER_TENSOR | 1 | fp8e4m3fn + k_scale/v_scale (per-tensor) | fp8→fp32×scale→Q_dtype | Yes (SM89+) |
| INT8_PER_TOKEN_HEAD | 2 | int8 + per-(token,head) scale | int8→Q_dtype then apply scales separately | Yes (SM75+) |
| FP8_PER_TOKEN_HEAD | 3 | fp8e4m3fn + per-(token,head) scale | fp8→Q_dtype then apply scales separately | Yes (SM89+) |
| NVFP4 | 4 | packed fp4 + fp8 block scales | (separate path) | Yes (SM75+) |

**Critical**: INT8_PER_TOKEN_HEAD (mode 2) works on ALL GPUs (SM75+). FP8 modes require SM89+ because `fp8e4nv` has no lowering before SM89.

---

## 3. PR #43914 Test Matrix — Hardware/Dtype Sweep

| GPU | Compute Cap. | FP8 KV Cache | BF16 KV Cache | Error Behavior |
|-----|-------------|--------------|---------------|---------------|
| Tesla T4 | SM75 | **Rejected** — suggests `float16` | **Rejected** — suggests `float16` | Clean ValueError |
| A100 | SM80 | **Rejected** — suggests `bfloat16` | **Allowed** | Clean ValueError |
| RTX 3090 | SM86 | **Rejected** — suggests `bfloat16` | **Allowed** | Clean ValueError |
| **RTX 4090 / L4** | **SM89** | **Allowed** | **Allowed** | Works correctly |
| H100 | SM90 | **Allowed** | **Allowed** | Works correctly |
| B200 | SM100 | **Allowed** | **Allowed** | Works correctly |

**For RTX 4090 (SM89)**: Triton FP8 KV cache is now EXPLICITLY ALLOWED with architectural validation. No `fp8e4nv "not supported in this architecture"` crash.

---

## 4. Three FP8 KV Paths on SM89 — Must Distinguish

This is a critical update to our previous understanding. There are THREE distinct FP8 KV paths that behave differently on SM89:

| Path | Backend | SM89 Status | Mechanism | Crash Risk |
|------|---------|-------------|-----------|-----------|
| **Triton FP8 KV** | TRITON_ATTN | **ALLOWED** (#43914) | Software dequant in Triton kernels | None (guarded) |
| **FlashInfer FP8 KV** (native) | FLASHINFER | **NOT SUPPORTED** | Requires FP8 tensor cores | Crash (#44879) |
| **FlashInfer FP8 KV** (TRTLLM-GEN dequant) | FLASHINFER | **NOT SUPPORTED** (TRTLLM requires SM100) | Dequant FP8→BF16 then BF16 attention | Crash/guard |

### 4.1 Triton FP8 KV (ALLOWED on SM89)

- `TRITON_ATTN` backend selected when: heterogeneous head_dim, NVFP4 models, non-standard head sizes (not 16/32/64), encoder attention, sinks enabled
- FP8 KV stored as fp8e4m3fn (native on SM89)
- Dequantized via `_cast_kv_tile` in Triton software (FP8→FP32×scale→BF16)
- Attention computed in BF16 domain (no FP8 tensor cores needed)
- PR #43914 adds explicit SM89 guard: FP8 KV rejected below SM89, allowed at SM89+

### 4.2 FlashInfer FP8 KV (NOT SUPPORTED on SM89)

- FlashInfer backend is the DEFAULT for most decoder models
- FlashInfer requires SM90+ for FP8 KV because it uses hardware FP8 tensor cores
- Issue #44879: compressed-tensors `kv_cache_scheme` overrides `kv_cache_dtype` to `fp8`, causing CUDA illegal memory access on SM89
- PR #45038 (open): adds `has_device_capability(90)` guard so auto-override only fires on SM90+

### 4.3 FlashInfer TRTLLM-GEN FP8 KV (NOT SUPPORTED on SM89)

- FlashInfer has a Triton-based dequant kernel (`_trtllm_prefill_attn_kvfp8_dequant`) that dequantizes FP8 KV to BF16 for prefill
- But TRTLLM-GEN attention (both prefill and decode) requires SM100+ (Blackwell)
- `can_use_trtllm_attention()` checks for hardware support; returns False on SM89
- So even with the dequant workaround, TRTLLM-GEN FP8 KV doesn't work on SM89

---

## 5. Backend Selection: When Does TRITON_ATTN Get Used?

### 5.1 TritonAttentionBackend Capabilities

```python
class TritonAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "float16", "bfloat16",
        "fp8", "fp8_e4m3", "fp8_e5m2",      # FP8 requires SM89+
        "int8_per_token_head",               # Works on SM75+
        "fp8_per_token_head",                # FP8 requires SM89+
    ]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True  # Supports ALL compute capabilities (7.5+)
```

### 5.2 Triton ATTN Backend Selection Triggers

TRITON_ATTN is selected when FlashInfer cannot handle the model configuration:

| Condition | FlashInfer | TRITON_ATTN | Reason |
|-----------|-----------|-------------|--------|
| Standard head_dim (16,32,64), GQA/MQA | **Selected** | Available | FlashInfer optimized for decoder |
| NVFP4 models (head_dim=256) | **Rejected** | **Selected** | Heterogeneous head_dim |
| Non-standard head sizes | **Rejected** | **Selected** | FlashInfer only supports 16,32,64 |
| Encoder-only attention | **Rejected** | **Selected** | FlashInfer = decoder-focused |
| Sinks enabled | Partial | **Selected** | Triton supports sinks |
| Any block size | 16,32,64 only | Any block size | FlashInfer = limited block sizes |
| All attention types | Decoder only | All (Decoder, Encoder, Encoder-only) | Triton = universal |

### 5.3 For RTX 4090 Typical Usage

Most production models on RTX 4090 use **FlashInfer** backend (default):
- LLaMA, Qwen, Mistral, etc. with standard GQA → FlashInfer
- INT8 KV cache with FlashInfer → production-proven on SM89

NVFP4 models (e.g., Gemma4 NVFP4) force **TRITON_ATTN**:
- Heterogeneous head_dim (256 for attention, smaller for projection)
- Now Triton FP8 KV IS allowed on SM89 for these models

---

## 6. Triton FP8 KV Cache Internal Mechanism — Source-Level Detail

### 6.1 Write Path: reshape_and_cache_kernel_flash

```python
# Triton JIT kernel — storage operation
key_load = tl.load(key_ptr + src_key_idx + tile_pos, ...)  # Load in original dtype (BF16)
if FP8_KV_CACHE:
    # Divide by scale, then tl.store does implicit cast to fp8e4m3fn
    key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
else:
    key_tile = key_load

# Later: tl.store writes key_tile to key_cache (fp8e4m3fn dtype on SM89)
```

**Key mechanism**: Triton's `tl.store` performs implicit type casting. When the destination buffer is `fp8e4m3fn` (which SM89 supports natively), Triton can store FP8 values directly. The division by `k_scale/v_scale` (float32) converts from BF16 to the FP8 range before the implicit cast.

### 6.2 Read Path: _cast_kv_tile in unified_attention

```python
# Triton JIT function — dequantization during attention
@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    if KV_QUANT_MODE == 1:  # FP8_PER_TENSOR
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)  # Stay FP8 if Q is also FP8
        # SOFTWARE DEQUANTIZATION: fp8 → fp32 × scale → Q_dtype (BF16/FP16)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)  # INT8/FP8_PER_TOKEN_HEAD: plain cast, scales applied separately
```

**Key mechanism**: The Triton kernel loads FP8 KV values, converts them to FP32, multiplies by the per-tensor scale, then casts to the query dtype. This is a **software dequantization path** that works on SM89 because:
1. FP8 load operations ARE supported on SM89
2. FP32 multiplication IS supported (always)
3. Cast to BF16/FP16 IS supported (SM80+)
4. The attention computation then proceeds in BF16/FP16 — no FP8 tensor cores needed

### 6.3 Per-Token-Head Quantization: Both INT8 and FP8

```python
_PER_TOKEN_HEAD_QUANT_PARAMS: dict[torch.dtype, tuple[float, float]] = {
    torch.int8: (127.0, -128.0),   # INT8 per-token-head
    FP8_DTYPE: (FP8_MAX, FP8_MIN),  # fp8e4m3fn per-token-head
}
```

For `int8_per_token_head` (mode 2) and `fp8_per_token_head` (mode 3):
- Quantize per (token, head) — find max absolute value per token-head pair
- Store as 1 byte per element + float32 scale per (token, head)
- In attention: plain cast to Q dtype, then apply per-token-head scales separately

### 6.4 FlashInfer's Triton Dequant Kernel (for comparison)

```python
# _trtllm_prefill_attn_kvfp8_dequant (in flashinfer.py)
fp8_k = tl.load(kv_cache_ptr + src_k)
dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)  # fp8 → fp32 × scale → bf16
tl.store(mock_kv_cache_ptr + dst_k, dequant_k)  # Write dequantized to mock cache
```

FlashInfer's Triton dequant kernel is SIMILAR to Triton's `_cast_kv_tile` — both do FP8→FP32×scale→BF16. But FlashInfer's version creates a **mock BF16 KV cache** that FlashInfer's native BF16 attention then uses. This is only used when TRTLLM-GEN attention is available (SM100+), so it doesn't help on SM89.

---

## 7. Memory Savings Comparison on SM89 (RTX 4090)

### 7.1 Per-Element Storage

| KV Cache Mode | Bytes per Element | Scale Storage | Total Memory vs BF16 |
|---------------|-------------------|---------------|---------------------|
| BF16 (auto) | 2 bytes | None | 1.0× (baseline) |
| FP8 per-tensor | 1 byte | k_scale + v_scale (2 float32 per tensor) | ~0.50× |
| FP8 per-token-head | 1 byte | 2 float32 per (token, kv_head) | ~0.50× + small overhead |
| INT8 per-token-head | 1 byte | 2 float32 per (token, kv_head) | ~0.50× + small overhead |
| FP8_e5m2 per-tensor | 1 byte | k_scale + v_scale (2 float32 per tensor) | ~0.50× |

### 7.2 RTX 4090 Practical Example (Qwen2.5-7B, GQA-4)

| Config | KV Cache Size (per token) | 24GB VRAM Capacity |
|--------|--------------------------|---------------------|
| BF16 KV (2 bytes) | 128 KB/token (4 KV heads × 128 head_dim × 2) | ~187K tokens |
| FP8/INT8 KV (1 byte) | 64 KB/token | ~375K tokens (2× more) |
| FP8/INT8 per-token-head | ~65 KB/token (64KB data + 256 bytes scales) | ~369K tokens |

**Conclusion**: FP8 and INT8 provide identical memory savings (~50%) on RTX 4090. The choice between them is about accuracy and compute path, not memory.

### 7.3 Accuracy: FP8 vs INT8

| Format | Range | Granularity | Quantization Error |
|--------|-------|------------|-------------------|
| fp8e4m3fn | [-448, 448] | 3-bit exponent, 4-bit mantissa | Lower error for typical KV values (centered around 0) |
| int8 | [-128, 127] | 7-bit integer | Higher error for values near quantization boundaries |
| fp8e5m2 | [-57344, 57344] | 5-bit exponent, 2-bit mantissa | Very coarse — only for storage, not compute |

**fp8e4m3fn vs int8**: FP8 has better accuracy for typical KV cache distributions because:
1. FP8 has a float representation with adaptive exponent — better for the bell-curve distribution of attention key/value norms
2. INT8 has uniform quantization steps — worse for outlier values
3. Per-token-head quantization (both FP8 and INT8) mitigates this by using per-head scales

---

## 8. Performance: Triton FP8 KV vs INT8 KV vs FlashInfer INT8 KV on SM89

### 8.1 Compute Path Analysis

| Path | Backend | KV Quant | KV→Q_dtype Conversion | Attention Compute Dtype | Expected Perf |
|------|---------|----------|----------------------|------------------------|---------------|
| FlashInfer INT8 KV | FLASHINFER | INT8 per-token-head | Hardware-optimized in FlashInfer | BF16/FP16 | **Best** (optimized CUDA kernel) |
| Triton INT8 KV | TRITON_ATTN | INT8 per-token-head | Triton software dequant | BF16/FP16 | Good (Triton kernel) |
| Triton FP8 KV (per-tensor) | TRITON_ATTN | FP8 per-tensor | Triton software dequant (fp8→fp32×scale→bf16) | BF16/FP16 | Good (Triton kernel) |
| Triton FP8 KV (per-token-head) | TRITON_ATTN | FP8 per-token-head | Triton cast then per-token-head scales | BF16/FP16 | Good (Triton kernel) |
| FlashInfer FP8 KV | FLASHINFER | FP8 per-tensor | **NOT POSSIBLE on SM89** | — | **CRASH** |

### 8.2 Triton vs FlashInfer Decode Performance

From the attention_backends.md documentation:

| Backend | CUDA Graphs | Batch Invariant | MM Prefix | Sink | Attention Types |
|---------|-------------|----------------|-----------|------|-----------------|
| FLASHINFER | Yes | Yes | Yes | No (SM100+ only) | Decoder |
| TRITON_ATTN | Yes | Yes | Yes | No | All |

FlashInfer is generally faster for decode on standard configurations because:
1. FlashInfer's CUDA kernels are hand-optimized for decoder attention
2. FlashInfer supports TRTLLM-GEN attention on SM100+ (Blackwell)
3. FlashInfer has optimized INT8 KV per-token-head path with hardware-accelerated dequant

Triton is competitive but generally slower for decode because:
1. Triton kernels are JIT-compiled — may have compilation overhead on first run
2. Software dequantization adds compute steps (fp8→fp32×scale→bf16)
3. Triton attention uses a more general kernel that handles all attention types

### 8.3 mikekg's Comment on FP8 KV on Ampere (SM80)

From PR #43914 discussion (mikekg, 2026-05-28):
> "On Ampere SM80 architecture, Triton currently does not support fp8 because of lack of hardware support. To support fp8 cache on Ampere, we'd also need conversion between bf16 <-> fp8. I have a prototype implementation, but it's not clear to me at this point that this is desirable because of the overhead imposed by software conversion between bf16 and f8e4nv."

This comment highlights the performance concern: software FP8↔BF16 conversion on GPUs without native FP8 tensor cores has overhead. On SM89 (Ada Lovelace), FP8 load/store IS native, so the overhead is lower than on SM80 — but the dequantization step (fp8→fp32×scale→bf16) still adds compute compared to BF16 KV.

### 8.4 No Published Benchmarks

There are no publicly available benchmarks comparing Triton FP8 KV vs INT8 KV vs FlashInfer INT8 KV on SM89. Performance data would need to be measured empirically on RTX 4090 hardware.

---

## 9. Production Readiness Assessment for RTX 4090

### 9.1 Triton FP8 KV on SM89 — Production Status

| Dimension | Status | Detail |
|-----------|--------|--------|
| Architectural guard | **PRODUCTION** | SM89+ guard in TritonAttentionImpl.__init__ (merged #43914) |
| Hardware compatibility | **VALIDATED** | Tested on T4/A100/3090/H100/B200; SM89 bracketed by SM86 (reject) and SM90 (allow) |
| Code maturity | **STABLE** | Triton reshape-and-cache + unified_attention kernels are core vLLM code |
| Backend availability | **CONDITIONAL** | Only when TRITON_ATTN is selected (not default for most models) |
| Accuracy | **REASONABLE** | fp8e4m3fn has better range than INT8 for typical KV values |
| Performance | **UNCLEAR** | No published benchmarks; software dequant adds compute overhead |
| Edge cases | **UNKNOWN** | MTP speculative decoding with Triton FP8 KV on SM89 not tested |

### 9.2 FlashInfer INT8 KV on SM89 — Production Status

| Dimension | Status | Detail |
|-----------|--------|--------|
| Architectural guard | **NONE NEEDED** | INT8 works on SM75+ (no SM requirement) |
| Hardware compatibility | **PRODUCTION** | Widely used on RTX 4090 in vLLM deployments |
| Code maturity | **VERY STABLE** | FlashInfer INT8 KV is default quantized KV path |
| Backend availability | **DEFAULT** | FlashInfer is default backend for most decoder models |
| Accuracy | **GOOD** | Per-token-head quantization adapts to each head's range |
| Performance | **OPTIMIZED** | FlashInfer's INT8 path is hardware-optimized CUDA kernel |
| Edge cases | **KNOWN** | Works with MTP speculative decoding, prefix caching, LoRA |

### 9.3 Recommendation Matrix for RTX 4090

| Scenario | Recommended KV Cache | Backend | Reason |
|----------|---------------------|---------|--------|
| Standard decoder model (LLaMA, Qwen, Mistral) | **INT8 per-token-head** | FlashInfer | Production-proven, fastest path |
| Standard decoder + maximum memory savings | **INT8 per-token-head** | FlashInfer | Same 50% savings as FP8, but more stable |
| NVFP4 model (Gemma4, etc.) | **FP8 or INT8 per-token-head** | TRITON_ATTN (forced) | Triton FP8 now allowed on SM89 (#43914) |
| NVFP4 model + maximum accuracy | **INT8 per-token-head** | TRITON_ATTN (forced) | INT8 works everywhere, no FP8 concerns |
| NVFP4 model + best accuracy-to-size ratio | **FP8 per-token-head** | TRITON_ATTN (forced) | FP8 has better quantization than INT8 for KV norms |
| Encoder-only model | **FP8 or INT8 per-token-head** | TRITON_ATTN (forced) | FlashInfer doesn't support encoder attention |
| Compressed-tensors FP8 checkpoint | **INT8 per-token-head** (with --kv-cache-dtype override) | FlashInfer | FP8 override crashes on SM89 (#44879) |

---

## 10. Impact on Our SM89 Strategy

### 10.1 Updates to SM89 Compatibility Matrix

Before #43914, our understanding was:
- **FP8 KV on SM89 = NOT POSSIBLE** (FlashInfer requires SM90)

After #43914, the correct understanding is:
- **FP8 KV on SM89 via Triton = ALLOWED** (software dequant path)
- **FP8 KV on SM89 via FlashInfer = NOT POSSIBLE** (requires SM90 hardware tensor cores)
- **INT8 KV on SM89 via FlashInfer = ALLOWED** (production-proven)
- **INT8 KV on SM89 via Triton = ALLOWED** (works on SM75+)

### 10.2 The Three FP8 KV Paths Must Be Distinguished

Our previous note stated "FlashInfer FP8 NOT" which was correct for FlashInfer but incomplete. The full picture:

1. **Triton FP8 KV on SM89 = ALLOWED (#43914)** — software dequant, works for TRITON_ATTN backend
2. **FlashInfer FP8 KV on SM89 = NOT SUPPORTED** — requires SM90 FP8 tensor cores
3. **compressed-tensors FP8 override on SM89 = CRASH (#44879/#45038)** — kv_cache_scheme overrides to fp8, triggers FlashInfer crash

### 10.3 Contribution Opportunity Updates

PR #43914 adds architectural guards that are similar to what we were planning for the QuantKey refactor (#32268). The key difference:
- #43914: guards at TritonAttentionImpl.__init__ + reshape_and_cache — Triton-specific
- Our planned contribution: systematic QuantKey-based requires_sm guard — cross-backend

The QuantKey refactor (#32268) is still needed because:
1. #43914 only guards the Triton path — FlashInfer's FP8 SM90 requirement is handled by #45038 (not merged yet)
2. There's no unified "requires_sm" mechanism across all backends
3. QuantKey would provide a systematic framework for dtype→SM capability mapping

### 10.4 PR #40127 — Related Work (Still Open)

PR #40127: "[Kernel] Add SM>=89 guard for Triton block FP8 kernel" (by Sandermage, still OPEN)
- Adds SM89+ guard for `TritonFp8BlockScaledMMKernel` (linear layer, not attention)
- Uses `tl.dtype('fp8e4nv')` which requires SM89+
- On SM<89, Marlin kernel is selected as fallback (after #40105)
- This is for linear layers, separate from attention KV cache

---

## 11. Key Technical Details for Reference

### 11.1 fp8e4m3fn (float8_e4m3fn) — The Triton FP8 Type

| Property | Value |
|----------|-------|
| Triton name | `fp8e4nv` |
| PyTorch dtype | `torch.float8_e4m3fn` |
| Range | [-448, 448] |
| Mantissa bits | 4 |
| Exponent bits | 3 (with bias 7) |
| Zero representation | +0 and -0 |
| Inf representation | Not supported (saturated to max) |
| SM requirement | **SM89+ for native support** (Ada Lovelace, Hopper, Blackwell) |
| SM80/86 | Can be stored/loaded via Triton view casting, but NO FP8 compute |
| SM75 | Cannot be used at all (rejected by #43914 guard) |

### 11.2 Triton Implicit Cast Mechanism

In Triton JIT kernels, `tl.store` performs implicit casting based on the destination pointer's dtype. This is critical for FP8 KV cache storage:

```python
# key_cache_ptr.dtype.element_ty = fp8e4nv on SM89
# Triton automatically casts key_tile (BF16 after division) to fp8e4nv when storing
tl.store(key_cache_ptr + tgt_idx_k, key_tile, ...)
```

This implicit cast works on SM89 because the GPU has native fp8e4nv store support. On SM80/86, Triton's `parse_options` gates `fp8e4nv` behind capability >= 89, so this code path is never compiled on Ampere.

### 11.3 The uint8 View Casting Trick

Before Triton supported fp8e4m3fn natively, vLLM stored FP8 KV cache as `torch.uint8` and used `.view(torch.float8_e4m3fn)` to reinterpret the bytes. PR #43914 removes the uint8 assert because Triton now handles FP8 through direct fp8e4nv operations, not uint8 reinterpretation.

However, the `.view()` trick still exists in the code for backward compatibility and for cases where the cache is allocated as uint8:

```python
if is_quantized_kv_cache(kv_cache_dtype) and key_cache.dtype != kv_cache_torch_dtype:
    key_cache = key_cache.view(kv_cache_torch_dtype)  # uint8 → float8_e4m3fn view
    value_cache = value_cache.view(kv_cache_torch_dtype)
```

---

## 12. Open Questions and Future Work

### 12.1 Triton FP8 KV vs INT8 KV Accuracy on SM89

No empirical accuracy comparison exists for Triton FP8 KV vs INT8 KV on SM89. Theoretical analysis suggests FP8 per-token-head should have slightly better accuracy than INT8 per-token-head for typical KV distributions, but this needs measurement.

### 12.2 Triton FP8 KV Performance on SM89

The software dequantization step (fp8→fp32×scale→bf16) adds compute overhead compared to BF16 KV. The question is whether the 50% memory savings outweigh the compute overhead for RTX 4090's memory-bound decode workload.

For decode (memory-bound on RTX 4090):
- Memory savings from FP8/INT8 KV cache → more tokens per batch → higher throughput
- Dequantization overhead → slightly higher per-token latency
- Net effect: likely positive (memory savings dominate for memory-bound workloads)

### 12.3 FP8 KV on Ampere (SM80/SM86) — Future Opportunity

mikekg's prototype for bf16↔fp8 conversion on Ampere could enable FP8 KV on SM80/86. The overhead question is critical:
- On Ampere, FP8 has no native hardware support → all FP8 operations are software emulated
- The conversion overhead (bf16→fp8 during cache, fp8→bf16 during attention) could negate memory savings
- INT8 KV remains the safe choice for SM80/86

### 12.4 QuantKey Refactor (#32268) — Still Needed

PR #43914 and #45038 provide backend-specific guards, but there's no unified dtype→SM capability mapping. The QuantKey refactor would provide:
- `requires_sm` field on QuantKey → systematic SM requirement per dtype
- Cross-backend enforcement → one check covers all backends
- Foundation for future dtype additions (FP4, MXFP4, etc.)

---

## 13. Summary

PR #43914 is a significant change that corrects our understanding of FP8 KV cache support on SM89:

1. **Triton FP8 KV IS ALLOWED on SM89** — via software dequantization in Triton JIT kernels
2. **FlashInfer FP8 KV is NOT supported on SM89** — requires SM90 hardware FP8 tensor cores
3. **INT8 KV remains the safest choice for RTX 4090** — works on all backends, production-proven
4. **FP8 KV is viable on SM89 for TRITON_ATTN backend** — NVFP4 models, encoder attention, non-standard configs
5. **Three FP8 KV paths must be distinguished** — Triton FP8, FlashInfer FP8, compressed-tensors FP8 override

The PR replaces opaque autotuning crashes with clear actionable errors, adds architectural guards, and establishes the SM89 FP8 baseline for Triton attention. This is complementary to (but not a substitute for) the QuantKey refactor (#32268) which would provide systematic cross-backend dtype→SM capability enforcement.

---

*Last updated: 2026-06-15 | Based on PR #43914 (merged), related issues #44879/#45038, PR #40127 (open), and vLLM v0.23.0 source code*
