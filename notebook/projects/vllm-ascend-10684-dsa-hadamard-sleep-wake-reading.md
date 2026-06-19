# vLLM-Ascend #10684: DSA Hadamard Sleep/Wake ALL-ZERO — Deep Reading

> 2026-06-19 | Deep analysis of DSA Hadamard class variable loss during sleep/wake
> ★★★★★★★★ CRITICAL BLOCKER for verl RLHF on Ascend NPU with DSA models
> ★★★★★★★★ Pattern family: State Lifecycle Mismatch (identical to SGLang #28676, #28679, vLLM #44395)
> ★★★★★★★★ Root cause CONFIRMED: hadamard stored as CLASS VARIABLE, NOT model buffer

---

## 1. Bug Summary

**Issue:** #10684 on vllm-project/vllm-ascend
**Title:** DSA Hadamard produces ALL-ZERO output after sleep/wake cycle
**Impact:** BLOCKER for any RLHF/GRPO training on Ascend NPU with DSA models (DSV4, GLM-5.x)
**Severity:** CRITICAL — ALL downstream DSA attention produces zero output → completely broken inference

---

## 2. Root Cause — CLASS VARIABLE vs Model Buffer

★★★★★★★★★ The hadamard transform matrix is stored as a **CLASS VARIABLE** on `AscendDSACPMetadataBuilder.hadamard`, NOT as an instance variable or model buffer.

### The Storage Problem

```python
class AscendDSACPMetadataBuilder:
    hadamard = None  # CLASS VARIABLE — shared across all instances!
    hadamard_scale = None  # Also class variable
```

This means:
1. **Not in `model.named_buffers()`**: Only instance attributes and registered buffers appear in `named_buffers()`. Class variables are invisible.
2. **Not in `model.named_parameters()`**: Same — class variables don't appear in parameter lists.
3. **Not saved by `worker.sleep()`**: The sleep mechanism explicitly saves `model.named_buffers()` (lines 221-223), which skips class variables entirely.

### The Corruption Chain

```
1. CaMemAllocator.sleep() → offloads NPU memory tagged as "weights" to CPU
2. worker.sleep() → saves model.named_buffers() → hadamard NOT included
3. wake_up() → restores saved buffers → hadamard NOT restored
4. NPU memory backing hadamard tensor → invalidated/zeroed after wake_up
5. ALL downstream DSA attention → zero output → ALL-ZERO inference
```

### Why This is Textbook State Lifecycle Mismatch

The bug follows the exact pattern described in our `state-lifecycle-mismatch-pattern-family-derivation.md`:

| Stage | Expected | Actual |
|-------|----------|--------|
| Pre-sleep | hadamard tensor on NPU | hadamard tensor on NPU ✓ |
| Sleep | hadamard saved & offloaded | hadamard NOT saved, memory offloaded ✗ |
| Wake | hadamard restored to NPU | hadamard NOT restored, memory zeroed ✗ |
| Post-wake | correct DSA attention | ALL-ZERO output ✗ |

The **state transfer boundary** (sleep/wake) drops a GPU-resident constant because it's stored in a location invisible to the save mechanism.

---

## 3. DSA Architecture on Ascend — How Hadamard is Used

### DSA Components

DSA (Deep Sparse Attention) on Ascend uses these components:
- `wq_a`, `q_norm`, `wq_b` — query projection (split into A/B for MLA-style)
- `wkv`, `kv_norm` — key-value projection
- `wo_a`, `wo_b` — output projection
- `attn_sink` — attention sink token
- `indexer` — sparse attention indexer
- `compressor` — KV compression
- `swa_cache_layer` — sliding window attention cache
- `indexer_rotary_emb` — rotary embedding for indexer

### Hadamard Transform in DSA

The hadamard transform is applied in DSA attention:

```python
# In dsa.py (291 lines)
F.linear(x, hadamard)  # Hadamard transform: orthogonal matrix multiplication
# scale = hidden_size**-0.5
```

Split into two streams:
- `hadamard_linear` — applied in main stream
- `hadamard_scale` — applied after aux_stream

**Purpose:** The Hadamard transform provides orthogonal mixing of attention features, improving numerical stability and reducing attention degeneracy in DSA.

### CUDA vs Ascend Difference

| Aspect | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|--------|-------------|----------------------|
| Hadamard storage | Model parameter/buffer | **CLASS VARIABLE** on metadata builder |
| Attention backend | `scaled_dot_product_attention` | `npu_flash_attention` / `npu_bmm_flash_attention` |
| Metadata builder | `DSAMetadataBuilder` | `AscendDSACPMetadataBuilder` (NPU-specific) |
| Multi-stream | `torch.cuda.Stream()` | `torch_npu.npu.Stream()` |

---

## 4. Pattern Family Connections

★★★★★★★★★ This bug belongs to the **State Lifecycle Mismatch** pattern family, with exact structural equivalence to:

### SGLang #28676 — MXFP8 MoE Shuffle Cache CLOBBERED

- **Pattern**: GPU-resident shuffle cache clobbered on weight reload
- **Similarity**: Constant tensor stored outside model save mechanism → lost at state transfer boundary
- **Fix**: `dict.clear()` at weight-reload boundary (invalidate ALL caches)

### SGLang #28679 — GDN Intermittent Decode Degeneracy

- **Pattern**: GPU-resident GDN state accumulates stale data over uptime
- **Similarity**: State that should be reset at boundary is NOT → progressive corruption
- **Fix**: Periodic flush mechanism (addresses #28679 pattern)

### vLLM #44395/#44483 — KV Cache Still Asleep

- **Pattern**: `wake_up(tags=["weights"])` + forward → illegal memory access
- **Similarity**: Partial wake-up leaves GPU-resident state in invalid state
- **Fix**: Staged wake — wake ALL necessary tags before forward pass

### rLLM #605 — GRPO Grouping Bug

- **Pattern**: Trajectory grouping key includes trajectory name → group size=1 → std=0
- **Similarity**: Logical grouping error causes downstream computation degeneracy
- **Fix**: Change grouping key from `task_id:name` to just `task_id`

**Severity Level 4 (Critical — Complete Failure)**: All downstream computation produces zero output, equivalent to model brain-death.

---

## 5. Fix Directions

### Option 1: Convert Hadamard to Model Buffer (Best — Automatic Save/Restore)

```python
# Instead of class variable:
class AscendDSACPMetadataBuilder:
    hadamard = None  # CLASS VARIABLE → BAD

# Register as model buffer:
# In model definition (deepseek_v4.py):
self.register_buffer('hadamard', hadamard_tensor)
# Now appears in model.named_buffers() → automatically saved/restored by sleep/wake
```

**Advantages**: Zero additional code in sleep/wake path. Automatic lifecycle management.
**Disadvantages**: Changes model architecture (adds buffer). Need to verify DSV4 model compatibility.

### Option 2: Re-compute Hadamard After wake_up (Practical — Quick Fix)

```python
# In AscendDSACPMetadataBuilder or worker:
def wake_up(self, ...):
    # Existing wake_up logic...
    # Re-compute hadamard:
    hadamard_size = self.hadamard_size
    self.hadamard = create_hadamard_matrix(hadamard_size).to(device)
    self.hadamard_scale = hidden_size**-0.5
```

**Advantages**: Minimal change, doesn't modify model architecture.
**Disadvantages**: Adds computation at wake boundary. Need to know hadamard dimensions.

### Option 3: Copy Before In-place Mutation (Workaround)

```python
# In dsa.py, wherever hadamard is modified:
hadamard_copy = self.hadamard.clone()  # Preserve original
# Use hadamard_copy for computation
```

**Advantages**: Prevents corruption of class variable.
**Disadvantages**: Doesn't solve the root cause (sleep/wake still drops it).

**Recommended**: Option 1 (model buffer) as long-term fix. Option 2 (re-compute) as immediate workaround.

---

## 6. verl RLHF Impact on Ascend NPU

★★★★★★★★★ This bug is a **BLOCKER** for verl RLHF/GRPO on Ascend NPU:

### verl Sleep/Wake Cycle on Ascend

verl's HYBRID sleep/wake architecture for RLHF training:
1. **Rollout phase**: Model generates trajectories (forward pass with DSA)
2. **Sleep**: Offload model weights to CPU (sleep_level=1: tags=["kv_cache"])
3. **Training phase**: Update model weights via PPO/GRPO
4. **Wake**: Reload model weights to NPU
5. **Repeat**: Rollout → Sleep → Train → Wake → Rollout...

If step 4 (wake) doesn't restore hadamard, ALL subsequent rollout forward passes produce zero output → ALL rewards are 0 → GRPO/PPO completely broken.

### Ascend Integration Pathway

verl Ascend integration uses NPUIPC (#10592) for weight sync, which means:
- Weight transfer happens every training step
- Sleep/wake cycle happens every training step
- Hadamard corruption happens EVERY training step
- → Complete training failure from step 1

---

## 7. RTX 4090 Pattern Transfer

★★★★★★★★★ 6 patterns that carry from Ascend NPU to CUDA (RTX 4090):

1. **State Lifecycle Mismatch** (#10684 → #28676, #28679, #44395): ANY GPU-resident constant buffer MUST be invalidated/rebuilt at weight-reload boundary.

2. **Sleep/Wake Buffer Preservation**: Ascend CaMemAllocator tag-based offload mirrors vLLM/SGLang. Both platforms face class-variable/device-constant tensor loss.

3. **MoE NaN from Sign Convention** (#10579): Operator semantics differ between hardware. ALWAYS verify when porting.

4. **MX Quant Fusion** (#10730): AddRMSNorm+DynamicMxQuant fusion. RTX 4090: MXFP8 MoE quant has same cache invalidation requirement.

5. **NPUIPC Security**: pickle.loads RCE mirrors SGLang #28582. NEVER deserialize untrusted data over network.

6. **DSV4 Instability** (#10724): 8th confirmed failure on Ascend. RTX 4090 MUST enforce_eager=True + invalidate ALL GPU-resident caches.

---

## 8. Cross-Framework Defense Stack Update

This bug confirms our 4-layer defense stack needs **Layer 1 enhancement**:

| Layer | Defense | #10684 Coverage |
|-------|---------|----------------|
| **Layer 1** | Synchronization | MUST ensure ALL GPU-resident state is saveable at boundary |
| **Layer 2** | Correctness Checks | Hadamard ALL-ZERO detectable via norm check |
| **Layer 3** | State Reset | Invalidate/recompute hadamard at wake_up |
| **Layer 4** | FSM Validation | N/A (not FSM-related) |

**New MUST DO rule**: GPU-resident constants MUST be stored as model buffers (not class variables or standalone tensors) to ensure automatic save/restore during state lifecycle transitions.

**New MUST NOT rule**: NEVER store GPU-resident tensors as class attributes outside the model hierarchy — they are invisible to `named_buffers()` / `named_parameters()` save mechanisms.

---

## References

- Issue #10684: https://github.com/vllm-project/vllm-ascend/issues/10684
- MindIE/vLLM-Ascend ecosystem research: notebook/projects/mindie-vllm-ascend-ecosystem-deep-research.md
- State Lifecycle Mismatch pattern family: notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md
- SGLang #28676 MoE cache clobber: notebook/projects/sglang-28676-mxfp8-moe-v4-reading.md
- vLLM #44395 partial wake: notebook/fundamentals/cross-framework-partial-wake-safety-analysis.md
- DSA Hadamard validator tool: tools/dsa_hadamard_sleep_wake_validator.py
