# vLLM INT4 Triton Fallback — Source-Level Reading Note

> PR #43731: `[Kernel] Enable TritonW4A16LinearKernel as CUDA fallback for non-Marlin-aligned W4A16 shapes`
> Author: Luciano Martins (lucianommartins)
> Merged: 2026-05-27 | Release: v0.23.0
> Reviewer: mgoin (approved with performance warning note)

---

## 1. The Problem: "ValueError: Failed to find a kernel that can implement the WNA16 linear layer"

★★★ Before PR#43731, certain W4A16 compressed-tensors models on **Ampere/Ada (SM80/SM89)** GPUs could not load at all. The error was a hard crash, not a degradation.

### Which shapes fail?

Marlin's fundamental tile constraint: `GPTQ_MARLIN_MIN_THREAD_K = 128`. Any linear layer where `input_size_per_partition % 128 != 0` is rejected by Marlin (when `has_g_idx=True`). Even without `g_idx`, **MoE Marlin does NOT support padding**:

```python
# marlin_utils.py:347-353
# moe marlin requires n % 128 == 0 and k % 64 == 0
supports_shape = (
    hidden_size % 128 == 0
    and intermediate_size_per_partition % max(64, group_size) == 0
)
```

**Critical comment in marlin_utils.py:314-317**: "Thread-tile misalignment is fixed by zero-padding at weight prep... Dense layers only — **MoE prep does not pad yet**."

### Kernel rejection chain on SM80/SM89

For a W4A16 compressed-tensors layer with `intermediate_size=2112` or `moe_intermediate_size=704`:

| Kernel | Min SM | Rejection Reason |
|--------|--------|-----------------|
| CutlassW4A8 | 90 | **SM90 required** (Hopper); also W4A8 not W4A16 |
| Machete | 90 | **SM90 required** |
| AllSpark | 80 | Only `group_size=-1` on Ampere; rejects other group sizes |
| Marlin | 75 | Requires `input % 128 == 0`; **MoE cannot pad** |
| Humming | 75 | External library; rarely installed |
| Conch | 80 | Only `group_size in [-1, 128]`; rejects 32/64/256 |
| Exllama | 60 | Only `float16` activations; **rejects bfloat16** |
| (TritonW4A16) | 0 | **Was ROCm-only gate** — could handle all shapes but blocked |

★★★ Every single kernel in the CUDA chain rejected certain MoE layer shapes. The result was a fatal `ValueError` with no workaround.

---

## 2. The Fix: Two Lines Changed, Enormous Impact

### Code changes (2 files, +3 lines)

**File 1: `vllm/model_executor/kernels/linear/__init__.py`**

```python
# _POSSIBLE_KERNELS PlatformEnum.CUDA list — BEFORE:
[CutlassW4A8, Machete, AllSpark, Marlin, Humming, Conch, Exllama]

# AFTER — TritonW4A16 added as LAST (lowest priority):
[CutlassW4A8, Machete, AllSpark, Marlin, Humming, Conch, Exllama, TritonW4A16]
```

**File 2: `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`**

```python
# BEFORE:
if not current_platform.is_rocm():
    return False, "TritonW4A16LinearKernel only targets ROCm"

# AFTER:
if not (current_platform.is_rocm() or current_platform.is_cuda()):
    return False, "TritonW4A16LinearKernel requires CUDA or ROCm"
```

★★★ The change is minimal but architecturally critical: the ROCm-only gate was removed, and TritonW4A16 entered the CUDA priority list at the **bottom**. This is a "catch-all fallback" pattern — higher-priority kernels are tried first, and TritonW4A16 only activates when everything else rejects.

---

## 3. TritonW4A16LinearKernel Implementation — How It Works

### Kernel architecture (345 lines total)

The kernel is a **pure Triton** implementation of fused W4A16 GEMM: `C[M,N] = A[M,K] @ dequant(B)[K,N]`

#### Core GEMM kernel (`triton_w4a16_gemm_kernel`)

```
Algorithm:
1. Tile over [M, N] using program IDs (pid_m, pid_n)
2. For each K-block tile (k_start loop):
   a. Load activation A: [BLOCK_M, BLOCK_K] fp16/bf16
   b. Load packed weight B: [BLOCK_K, BLOCK_N//8] int32
   c. Unpack int4 → [BLOCK_K, BLOCK_N] using tl.interleave trick
   d. Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N]
   e. Load/compute zeros: HAS_ZP? unpack zeros : ZP_BIAS constant
   f. Dequantize: (w_int4 - zero) * scale → b_fp
   g. Accumulate: tl.dot(a, b_fp, out_dtype=tl.float32)
3. Store output C: [BLOCK_M, BLOCK_N]
```

★★★ The **tl.interleave unpack trick** is elegant and efficient:

```python
# GPTQ sequential packing: 8 int4 values per int32, shifts [0,4,...,28]
b = tl.interleave(b_packed, b_packed)  # [BLOCK_K, BLOCK_N//8] → [BLOCK_K, BLOCK_N//4]
b = tl.interleave(b, b)                # → [BLOCK_K, BLOCK_N//2]
b = tl.interleave(b, b)                # → [BLOCK_K, BLOCK_N]
b = (b >> shifts) & 0xF                # Extract correct nibble for each column
```

Three `tl.interleave` calls replicate each int32 value 8 times, then `shift+mask` extracts the correct 4-bit nibble. This avoids element-wise indexing and is Triton-friendly.

### Shape constraints (very loose)

```python
# TritonW4A16 can_implement constraints:
N % 8 == 0          # 8 int4 values per int32 (very loose)
K % group_size == 0 # Standard group quantization requirement
group_size in [-1, 32, 64, 128, 256]
No g_idx             # No activation reordering support
act_type: float16 or bfloat16  # Both supported!
weight_type: uint4b8 (symmetric) or uint4 (asymmetric)
```

★★★ Compare with Marlin's constraints:
- Marlin: `K % 128 == 0` (strict) → Triton: `N % 8 == 0` (16x more relaxed on alignment)
- Marlin: `group_size in [-1, 32, 64, 128]` → Triton: adds `256`
- Marlin: float16/bfloat16 but limited on Ampere → Triton: both, all platforms
- Marlin: supports g_idx → Triton: **no g_idx** (limitation)

### Block size tuning

| M range | BLOCK_M | BLOCK_N | BLOCK_K |
|---------|---------|---------|---------|
| M <= 32 | 32 | 64 | 32 |
| M <= 64 | 64 | 64 | 32 |
| M > 64 | 128 | 128 | 32 |

**Special clamp**: If `group_size < BLOCK_K`, then `BLOCK_K = group_size`. This ensures one scale group per K-tile — critical for correctness.

### Weight layout transformation (`process_weights_after_loading`)

Checkpoint format → kernel format requires a **full repack** (not just transpose):

```
Checkpoint: weight_packed [N, K//8] int32  (K packed at dim 1)
Kernel:     qweight [K, N//8] int32         (N packed at dim 1)

Repack steps:
1. permute_param_layout_ (input_dim=1, output_dim=0, packed_dim=1)
2. Unpack: [N, K//8] → [N, K] using shift+mask
3. Transpose: [N, K] → [K, N]
4. Repack N: [K, N] → [K, N//8] using shift+OR
```

★★★ The repack is CPU-side, one-time cost at model loading. Scales also need transpose `[N, K//G] → [K//G, N]`. Zero points `[N//8, K//G] → [K//G, N//8]` (simple transpose).

---

## 4. The ROCm-Only → CUDA+ROCm Change: Why It Matters

### Original PR #37352 (April 2026)

TritonW4A16 was originally written **for ROCm MI300 only**. The gate `if not current_platform.is_rocm()` was intentional because:
- The kernel was tested on MI300 (gfx942) hardware
- ROCm had no good W4A16 kernels (Marlin is CUDA-only)
- Block sizes were tuned for MI300 wavefront=64

★★★ But the kernel is **pure Triton** — `tl.dot/tl.load/tl.store` — no platform-specific ops, no inline PTX, no ROCm-specific intrinsics. This makes it inherently portable to CUDA.

### PR #43731 rationale

On Ampere (SM80) and Ada (SM89), the gap was:
- **Hopper (SM90)**: Cutlass/Machete handle everything
- **Ampere/Ada (SM80/SM89)**: No kernel handles `intermediate_size % 128 != 0` + bfloat16 + `group_size in [32, 64, 256]`

The Triton kernel fills this gap with **zero impact on existing models**:
- Higher-priority kernels (Marlin, etc.) still selected first for aligned shapes
- Triton only activates as last resort when all others reject
- 29/29 existing tests pass without regression

### mgoin's review comment

> "We should maybe have a performance warning for this kernel, but LGTM since it is already integrated"

This acknowledges TritonW4A16 is slower than Marlin, but the tradeoff is correct: **slow but working** beats **crash**.

---

## 5. Kernel Selection Priority: Complete CUDA W4A16 Chain

```python
_POSSIBLE_KERNELS[PlatformEnum.CUDA] = [
    CutlassW4A8LinearKernel,     # SM90, W4A8 (not W4A16)
    MacheteLinearKernel,         # SM90, CUTLASS-based
    AllSparkLinearKernel,        # SM80, group_size=-1 on Ampere
    MarlinLinearKernel,          # SM75, inline PTX, tile-aligned
    HummingLinearKernel,         # SM75, external library
    ConchLinearKernel,           # SM80, group_size in [-1, 128]
    ExllamaLinearKernel,         # SM60, float16 only
    TritonW4A16LinearKernel,     # SM0, pure Triton, catch-all ← NEW
]
```

### Selection algorithm (`choose_mp_linear_kernel`)

```python
for kernel in platform_kernels:           # Priority order
    if kernel in VLLM_DISABLED_KERNELS:   # Env var filter
        continue
    if kernel.get_min_capability() > cc:  # SM version filter
        continue
    can_implement, reason = kernel.can_implement(config)
    if can_implement:
        return kernel                     # First match wins
    failure_reasons.append(reason)

raise ValueError("Failed to find kernel") # All rejected
```

★★★ TritonW4A16 is the **safety net**. For any W4A16 layer on SM89:
- If shape is 128-aligned + group_size standard → Marlin selected (fastest)
- If shape is non-aligned + group_size not in [-1,128] + bf16 → Triton selected (slowest but working)

### `--linear-backend` override

Users can force a specific backend:
```bash
--linear-backend triton  # Forces TritonW4A16 (and other Triton kernels)
--linear-backend marlin  # Forces Marlin only (may fail on non-aligned shapes)
```

### `VLLM_DISABLED_KERNELS` override

```bash
VLLM_DISABLED_KERNELS=TritonW4A16LinearKernel  # Disable fallback (will crash on non-aligned)
```

---

## 6. Performance Comparison: Marlin vs Triton Fallback

★★★ No official benchmarks exist for TritonW4A16 on SM89. The analysis below is based on architectural understanding.

### Why Triton is slower than Marlin

| Aspect | Marlin | TritonW4A16 |
|--------|--------|-------------|
| Implementation | Custom CUDA + inline PTX | Pure Triton (compiler-generated) |
| Tile size | TILE=16, MIN_THREAD_K=128 | BLOCK_K=32, BLOCK_M/N=32-128 |
| Memory access | Hand-optimized loads/stores | tl.load/tl.store (generic) |
| Accumulation | FP32 reduce in registers | tl.dot → compiler decides |
| Padding support | Zero-pad at weight prep | No padding needed (flexible shapes) |
| BF16 on Ampere | Supported with specific paths | tl.dot handles both fp16/bf16 |

### Estimated performance on RTX 4090

- **Marlin-aligned shapes**: Marlin ~2-5x faster than Triton for same shape
- **Decode (M=1)**: Marlin has optimized small-M path; Triton uses BLOCK_M=32 (wasteful)
- **Prefill (M>64)**: Triton BLOCK_M=128 is more reasonable; gap smaller (~1.5-3x)
- **Net effect**: Triton fallback makes previously-unloadable models loadable, at ~2-5x slower GEMM for the specific non-aligned layers. Other (aligned) layers still use Marlin.

★★★ For MoE models like DeepSeek-V2-Lite, the **gate_up_proj** (K=2048, aligned) uses Marlin, while only **down_proj** (K=704, non-aligned) falls to Triton. The overall throughput hit is proportional to the fraction of layers that need Triton.

---

## 7. Which MoE Models Benefit Most?

### Models with non-128-aligned MoE intermediate_size

★★★ **DeepSeek-V2-Lite** is the primary beneficiary (explicitly mentioned in PR):

| Layer | K | N | K%128 | Kernel Used |
|-------|---|---|-------|-------------|
| gate_up_proj | 2048 | 1408 | 0 | Marlin (aligned) |
| down_proj | 704 | 2048 | **64** | **TritonW4A16** (fallback) |
| dense layers | 2048 | 2112 | **64** | **TritonW4A16** (fallback) |

### Other potentially affected models

| Model | moe_intermediate_size | % 128 | Status |
|-------|-----------------------|-------|---------|
| DeepSeek-V2-Lite | 704 | 64 | Needs Triton |
| Qwen2-MoE-57B-A14B | 2496 | 64 | Needs Triton |
| DeepSeek-V2/V3 | 1536 | 0 | Marlin OK |
| Mixtral-8x7B | 14336 | 0 | Marlin OK |
| Qwen3-MoE | 768 | 0 | Marlin OK |
| Llama-4-Scout | 5120 | 0 | Marlin OK |
| DBRX | 6144 | 0 | Marlin OK |

★★★ The pattern: models with `moe_intermediate_size % 128 != 0` are affected. This happens when the MoE FFN hidden dimension is chosen without regard to Marlin's 128-alignment. DeepSeek-V2-Lite's 704 and Qwen2-MoE's 2496 are examples.

### Dense models with intermediate_size % 128 != 0

Dense models CAN use Marlin padding (when no g_idx), so Triton fallback is less critical for dense layers. However, if the model uses `g_idx=True` (act-order quantization), Marlin cannot pad, and Triton becomes necessary.

---

## 8. Integration with vLLM Quantization Registry

### CompressedTensorsWNA16 → choose_mp_linear_kernel

```python
# compressed_tensors_wNa16.py
class CompressedTensorsWNA16(CompressedTensorsScheme):
    def create_weights(self, layer, ...):
        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(input_size_per_partition, output_size_per_partition),
            weight_type=self.quant_type,       # uint4b8 or uint4
            act_type=params_dtype,             # bf16 or fp16
            group_size=self.group_size,        # -1, 32, 64, 128, 256
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
        # → tries Cutlass→Machete→AllSpark→Marlin→Humming→Conch→Exllama→TritonW4A16
```

★★★ The integration is transparent: `CompressedTensorsWNA16` calls `choose_mp_linear_kernel()` which iterates through the priority list. TritonW4A16 is selected automatically when needed — no user configuration required.

### When TritonW4A16 is selected

Log message: `Using TritonW4A16LinearKernel for CompressedTensorsWNA16`

This appears only for layers where higher-priority kernels reject. The same model can have **mixed kernel selection**: Marlin for aligned layers, Triton for non-aligned layers.

### TritonW4A16 also in `--linear-backend triton`

```python
_LINEAR_BACKEND_KERNEL_MAP = {
    "triton": {
        TritonInt8ScaledMMLinearKernel,
        TritonFp8BlockScaledMMKernel,
        TritonW4A16LinearKernel,   # ← Included here too
    },
}
```

Users can force Triton backend with `--linear-backend triton`, which restricts selection to Triton kernels only.

---

## 9. Limitations of TritonW4A16

1. **No g_idx support**: Cannot handle activation reordering (act-order quantization). Models with `g_idx=True` that also have non-aligned shapes have no solution at all — Triton rejects them too.

2. **No CUDA graph compatibility**: Triton kernels with dynamic shapes may not be CUDA graph-friendly. vLLM's CUDA graph dispatch may need `enforce_eager=True` for Triton-based layers.

3. **Performance warning**: mgoin noted the kernel should have a performance warning. Currently no warning is emitted when TritonW4A16 is selected instead of Marlin.

4. **group_size limited to [-1, 32, 64, 128, 256]**: Most common sizes covered, but not as flexible as arbitrary group sizes.

5. **BLOCK_K clamped to group_size**: When `group_size < BLOCK_K`, BLOCK_K shrinks. This may reduce tile efficiency for small group sizes.

---

## 10. RTX 4090 Practical Analysis

### Before PR#43731: What was impossible

★★★ On RTX 4090 (SM89), loading a W4A16 compressed-tensors DeepSeek-V2-Lite model was **impossible**. The `down_proj` expert layer (K=704) could not find any kernel. This was a hard crash.

### After PR#43731: What is now possible

DeepSeek-V2-Lite W4A16 can now load and serve on RTX 4090:
- Aligned layers (gate_up_proj, attention) → Marlin (fast)
- Non-aligned layers (down_proj, some dense) → Triton (slower but working)

### Estimated throughput impact

For a 7B-class MoE model with ~10% of layers needing Triton fallback:
- Overall throughput: ~5-15% lower than if all layers used Marlin
- Specific Triton layers: ~2-5x slower than Marlin would be (if Marlin supported them)
- This is still **far better than the previous situation (hard crash)**

### RTX 4090 specific considerations

1. **SM89 = Ada Lovelace**: TritonW4A16's `get_min_capability() = 0` means it runs everywhere
2. **BF16 supported**: RTX 4090 native BF16 → TritonW4A16 handles bf16 activations
3. **PCIe single-GPU**: No TP partition issues (TP=1 means no partition splitting)
4. **CUDA graph**: Triton kernels may need `enforce_eager=True` for stability

### Configuration recommendation

```bash
# DeepSeek-V2-Lite W4A16 on RTX 4090
python -m vllm.entrypoints.openai.api_server \
    --model <w4a16-deepseek-v2-lite> \
    --max-model-len 4096 \
    --enforce-eager          # Triton kernels may not CUDA graph well
    --quantization compressed_tensors
```

★★★ If all layers are Marlin-aligned (e.g., most popular models), TritonW4A16 is never activated and has zero impact. The fallback is purely additive — it only helps when nothing else works.

---

## 11. Future Directions

1. **MoE Marlin padding support**: The `check_moe_marlin_supports_layer` does not pad. If MoE Marlin adds padding support, Triton fallback becomes unnecessary for most MoE shapes.

2. **Performance warning**: mgoin requested this. A `logger.warning_once()` when TritonW4A16 is selected would help users understand the tradeoff.

3. **TritonW4A16 + CUDA graph**: Triton kernels can potentially be CUDA graph-friendly if block sizes are static. This needs investigation for decode use cases.

4. **TritonW4A16 + g_idx**: Adding activation reordering support would make Triton the universal fallback for ALL W4A16 shapes, including act-order models.

5. **Block size autotuning**: Current block sizes are hardcoded. Triton's `triton.autotune` could optimize for specific GPU architectures (SM89 vs SM90).

---

## Key Takeaways

★★★ **1. This PR is a correctness fix, not a performance optimization.** It changes "crash" to "slow but working" for non-Marlin-aligned W4A16 shapes on SM80/SM89.

★★★ **2. TritonW4A16 is the safety net.** It's the last entry in the CUDA priority list and only activates when all 7 higher-priority kernels reject. Zero impact on existing models.

★★★ **3. MoE models with intermediate_size % 128 != 0 are the primary beneficiaries.** DeepSeek-V2-Lite (704), Qwen2-MoE (2496) were previously unloadable on RTX 4090.

★★★ **4. The ROCm-only gate was artificial.** The kernel is pure Triton with no platform-specific ops. Removing the gate was the correct decision.

★★★ **5. For RTX 4090 users**: INT4 compressed-tensors MoE models like DeepSeek-V2-Lite now work. Use `--enforce-eager` until CUDA graph compatibility is confirmed. Expected ~5-15% throughput reduction vs fully-Marlin-aligned models.

---

## Source Files

| File | Role |
|------|------|
| `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` | TritonW4A16 kernel implementation (GEMM + weight repack) |
| `vllm/model_executor/kernels/linear/__init__.py` | Kernel priority registry (`_POSSIBLE_KERNELS`) + `choose_mp_linear_kernel()` |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | CompressedTensorsWNA16 scheme → calls `choose_mp_linear_kernel()` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py` | Marlin shape constraints + `check_moe_marlin_supports_layer()` |
| `vllm/model_executor/kernels/linear/mixed_precision/marlin.py` | MarlinLinearKernel (SM75, tile-aligned) |
| `vllm/model_executor/kernels/linear/mixed_precision/machete.py` | MacheteLinearKernel (SM90, Hopper only) |

---

## Related PRs

| PR | Description |
|----|-------------|
| #37352 | Original TritonW4A16 kernel (ROCm-only, merged 2026-04-10) |
| #43731 | **This PR**: Enable CUDA fallback (merged 2026-05-27) |
| #41394 | RDNA3 native W4A16 kernel (ROCm gfx1100, merged 2026-05-29) |
