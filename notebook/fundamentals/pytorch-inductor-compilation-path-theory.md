# PyTorch Inductor Compilation Path Theory — Mathematical Derivation for SM89 Batch Invariance

> 2026-06-19 | Algorithm Theory | From torch.compile to Inductor FX Graph to Kernel Fusion → Why SM89 Breaks
> ★★★★★★★★ This is the CRITICAL theory behind P9 Fusion Guard: understanding Inductor's fusion decisions is the mathematical root of SM89 batch-dependent failures
> ★★★★★★★★ RTX 4090 Context: SM89 (Ada Lovelace) has FP8 storage but NO wgmma → Inductor attempts fp8→bf16 prologue fusion → breaks on batch-dependent shapes

---

## 1. The Compilation Pipeline: Python → FX Graph → Inductor → GPU Kernel

### 1.1 torch.compile Pipeline Overview

```
Python Model Code
  │
  ├── torch._dynamo.trace() → FX Graph (symbolic execution)
  │     ├── captures Python operations as graph nodes
  │     ├── graph = torch.fx.Graph(nodes, edges)
  │     └── each node = (op, target, args, kwargs)
  │
  ├── torch._inductor.lowering() → Scheduler Buffers
  │     ├── converts FX ops to Inductor Buffer operations
  │     ├── groups ops into "fused groups" based on memory/compute
  │     └── decides: which ops to fuse? which to keep separate?
  │
  ├── torch._inductor.scheduler() → Kernel Groups
  │     ├── creates fused kernel groups (fused_bufs)
  │     ├── determines: data dependencies, memory layout, compute intensity
  │     └── ★★★★★★★★ THIS IS WHERE SM89 FUSION DECISIONS ARE MADE
  │
  └── torch._inductor.codegen() → Triton/C++ GPU Kernel Code
        ├── generates Triton kernel for each fused group
        ├── each kernel has: grid dimensions, block dimensions, buffers
        └── kernel = fused_ops → single GPU launch → avoids intermediate memory
```

★★★★★★★★★ **The KEY question**: How does Inductor decide which ops to fuse? And why does SM89 get different fusion decisions than SM90?

### 1.2 Inductor Fusion Decision Process

Inductor's fusion decision follows a **greedy heuristic** based on:

1. **Memory intensity**: ops that read/write large tensors → benefit from fusion (avoid intermediate memory)
2. **Compute intensity**: ops with high FLOPs/byte ratio → can be separate kernels (compute-bound)
3. **Data dependency**: ops that depend on each other's outputs → must be in same kernel or sequential
4. **Hardware capability**: ops that use specific hardware features → different fusion on different GPUs

The fusion decision function (simplified):

$$\text{ShouldFuse}(op_1, op_2) = \begin{cases} \text{Yes} & \text{if } \frac{\text{bytes\_saved}}{\text{bytes\_read}} > \text{threshold} \\ \text{Yes} & \text{if } op_1.\text{output} = op_2.\text{input} \text{ and memory\_reduction} > 0 \\ \text{No} & \text{if } \text{compute\_intensity}(op_1) > \text{roofline\_threshold} \end{cases}$$

Where:
- $\text{bytes\_saved} = \text{intermediate\_tensor\_size}$ (eliminated by fusion)
- $\text{bytes\_read} = \text{total\_input\_bytes}$ for both ops
- $\text{threshold}$ depends on GPU architecture

---

## 2. SM89 vs SM90: The Hardware Capability Gap

### 2.1 SM89 (Ada Lovelace — RTX 4090)

| Feature | SM89 Capability | Impact on Inductor |
|---------|----------------|-------------------|
| **FP8 storage** | ✅ E5M2/E4M3 formats | Inductor can use FP8 for weight storage |
| **FP8 compute (wgmma)** | ❌ NOT supported | Inductor must convert FP8→BF16 before compute |
| **TMA (Tensor Memory Accelerator)** | ❌ NOT supported | Inductor must use manual memory loading |
| **Thread Block Clusters** | ❌ NOT supported | Inductor can't use distributed shared memory |
| **max_autotune** | ✅ but limited | Fewer layout options, batch-dependent |

★★★★★★★★★ **SM89's missing wgmma is the ROOT CAUSE of P9**: Inductor sees FP8 weights, knows SM89 can store but not compute, so it inserts a **prologue fusion**: dequantize FP8 → BF16 inline in the same kernel. This prologue fusion is **batch-dependent** because the FP8→BF16 conversion happens per-token, making the kernel shape depend on batch size.

### 2.2 SM90 (Hopper — H100)

| Feature | SM90 Capability | Impact on Inductor |
|---------|----------------|-------------------|
| **FP8 storage** | ✅ E5M2/E4M3 formats | Same as SM89 |
| **FP8 compute (wgmma)** | ✅ Full support | Inductor can compute in FP8 directly → NO prologue needed |
| **TMA** | ✅ Async copy | Better memory pipeline → less batch-dependent |
| **Thread Block Clusters** | ✅ Up to 8 blocks | Distributed shared memory → more flexible |
| **max_autotune** | ✅ Full | More layout options, fewer batch-dependent decisions |

★★★★★★★★★ **SM90 doesn't need prologue fusion**: When Inductor sees FP8 on SM90, it uses wgmma to compute directly in FP8. The kernel is NOT batch-dependent because wgmma operates on matrix tiles independent of total batch size. This is why H100 works but RTX 4090 breaks.

### 2.3 The Prologue Fusion: SM89's Achilles Heel

```
SM90 (H100):
  FP8 weights ──→ wgmma(FP8) ──→ FP8 output ──→ accumulate ──→ result
  ↑ NO prologue needed. Matrix tile operation. Batch-independent.

SM89 (RTX 4090):
  FP8 weights ──→ dequant(FP8→BF16) ──→ mma(BF16) ──→ BF16 output ──→ result
  ↑ PROLOGUE FUSION needed! Per-token conversion. Batch-dependent!
  ↑ If this prologue gets fused into the wrong kernel group → batch shape changes
  ↑ → cudagraph assumes constant batch → DSV4 breaks!
```

★★★★★★★★★ **The prologue fusion is the exact point where P9 applies**: Our Fusion Guard checks `props.major < 9` (SM89 = major version 8) and adds `WhyNoFuse("SM<90: fp8→bf16 prologue fusion is batch-dependent, skip on pre-sm90")` to prevent Inductor from making this dangerous fusion decision.

---

## 3. Batch Invariance: The Mathematical Condition

### 3.1 Definition

A kernel is **batch-invariant** if its compiled shape does not depend on the batch dimension:

$$\text{BatchInvariant}(K) \iff \text{grid\_dims}(K) \perp \text{batch\_size}$$

Where $\text{grid\_dims}(K)$ is the launch grid dimensions of kernel $K$.

### 3.2 Why Batch Invariance Matters for CUDA Graph

CUDA graph captures a sequence of GPU kernel launches and replays them with the SAME launch parameters. This requires:

$$\text{CUDAGraphSafe}(K_1, K_2, ..., K_n) \iff \forall i: \text{BatchInvariant}(K_i)$$

If ANY kernel in the sequence is batch-dependent, the CUDA graph will launch with the wrong grid dimensions when batch size changes → garbage output or crash.

### 3.3 The SM89 Batch Dependency Formula

For SM89 FP8 prologue fusion, the grid dimensions depend on:

$$\text{grid\_x} = \frac{\text{batch\_size} \cdot \text{seq\_len}}{\text{BLOCK\_M}}$$

Where BLOCK_M is the Triton kernel's M dimension block size. This is **batch-dependent** because both batch_size and seq_len appear in the formula.

For SM90 FP8 wgmma, the grid dimensions are:

$$\text{grid\_x} = \frac{\text{M}}{\text{wgmma\_tile\_M}}$$

Where M is the matrix dimension (independent of batch for most GEMM operations). This is **batch-independent** for matrix operations.

### 3.4 The Inductor Autotune Problem

Inductor's `max_autotune` searches for the best Triton kernel configuration by trying different BLOCK_M, BLOCK_N, BLOCK_K values. The search process:

1. Compile kernel with candidate config
2. Benchmark on current input shapes
3. Pick the fastest config

★★★★★★★★★ **Problem**: The fastest config for one batch size may be DIFFERENT from the fastest config for another batch size! This is because:
- Small batch → prefers larger BLOCK_M (more parallelism per block)
- Large batch → prefers smaller BLOCK_M (more blocks to fill GPU)

This creates a **batch-dependent optimization**: the chosen config changes with batch size → grid dimensions change → CUDA graph breaks.

### 3.5 The #187636 Solution

PyTorch #187636 (`autotune_at_compile_time`) changes the autotune behavior:

**Before**: autotune at RUNTIME → picks config based on actual batch size → batch-dependent
**After**: autotune at COMPILE TIME → picks config based on symbolic shapes → batch-independent (default)

$$\text{config\_choice}_{\text{runtime}} = \text{argmin}_{c} \text{latency}(c, \text{actual\_batch\_size})$$
$$\text{config\_choice}_{\text{compile\_time}} = \text{argmin}_{c} \text{latency}(c, \text{symbolic\_shapes})$$

★★★★★★★★★ **This COMPLEMENTS P9**: #187636 makes autotune batch-independent by default, while P9 prevents the dangerous prologue fusion on SM89. Together, they ensure batch invariance on RTX 4090.

---

## 4. The Inductor Choices.py: Where Fusion Decisions Are Made

### 4.1 The Choices.py Architecture

Inductor's fusion decisions are controlled by `torch/_inductor/choices.py`:

```python
# Key function: should_fuse_horizontal
def should_fuse_horizontal(group: list[Buffer]) -> bool:
    """Decide whether to fuse horizontally (same output -> same kernel)."""
    # Check memory savings
    bytes_saved = compute_memory_savings(group)
    if bytes_saved < threshold:
        return False
    # Check compute intensity
    compute_ratio = compute_arithmetic_intensity(group)
    if compute_ratio > roofline_threshold:
        return False  # Compute-bound → keep separate
    return True  # Memory-bound → fuse together

# Key function: should_fuse_vertical
def should_fuse_vertical(op1: Node, op2: Node) -> bool:
    """Decide whether to fuse vertically (op1 output -> op2 input)."""
    # Check data dependency
    if op1.output != op2.input:
        return False
    # Check hardware compatibility
    if not hardware_supports(op1, op2):
        return False
    return True
```

### 4.2 The P9 Fusion Guard Insertion Point

Our P9 Fusion Guard adds a hardware check in `should_fuse_vertical`:

```python
def should_fuse_vertical(op1: Node, op2: Node) -> bool:
    """Decide whether to fuse vertically (op1 output -> op2 input)."""
    if op1.output != op2.input:
        return False
    if not hardware_supports(op1, op2):
        return False

    # ★★★★★★★★ P9 Fusion Guard: SM<90 fp8→bf16 prologue fusion check
    if torch.cuda.get_device_properties(op1.device).major < 9:
        if is_fp8_prologue_fusion(op1, op2):
            WhyNoFuse("SM<90: fp8→bf16 prologue fusion is batch-dependent, "
                      "skip on pre-sm90 GPUs (RTX 4090/3090/etc)")
            return False

    return True
```

★★★★★★★★★ **5 lines of code**: This is the entire P9 contribution. It checks:
1. Is the GPU SM version < 9? (RTX 4090 = SM89 = major 8 ✓)
2. Is this a FP8→BF16 prologue fusion? (dequant FP8 → BF16 before compute ✓)
3. If both → REJECT the fusion → keep dequant as separate kernel → batch-independent

### 4.3 Why This Works: The Mathematical Proof

**Before P9** (SM89 with prologue fusion):
- Kernel: `dequant_fp8_bf16 + mma_bf16` (fused)
- Grid: `grid_x = (batch * seq_len) / BLOCK_M` → **batch-dependent**
- CUDAGraph: ✗ (batch changes → grid changes → replay fails)

**After P9** (SM89 with separate dequant):
- Kernel 1: `mma_bf16` (compute only)
- Grid: `grid_x = M / wgmma_tile_M` (but SM89 has NO wgmma → uses mma) → still batch-dependent for matmul!
- Wait... this means P9 alone isn't enough?

★★★★★★★★★ **Important nuance**: The matmul itself on SM89 IS batch-dependent (no wgmma → mma operates on full matrices). But:
1. With enforce_eager=True → NO cudagraph → batch changes are handled dynamically → SAFE
2. With P9 → dequant is NOT fused into matmul → can be computed ahead of time → more predictable
3. With #187636 → autotune at compile time → config choice is batch-independent → SAFER

**The complete solution for RTX 4090**:
- P9 (prevent prologue fusion) + #187636 (batch-independent autotune) + enforce_eager=True (no cudagraph)
- Each addresses a DIFFERENT aspect of the batch invariance problem

---

## 5. Connection Map: Inductor Theory → Framework Issues

| Theory Component | Mathematical Property | Framework Issue | RTX 4090 Decision |
|---|---|---|---|
| **Prologue fusion** | dequant(FP8→BF16) fused into matmul → batch-dependent grid | #184119 SM89 fp8 guard → P9 validated | P9 Fusion Guard prevents this on SM89 |
| **Batch-dependent autotune** | argmin latency(config, actual_batch) → config changes per batch | #187636 autotune_at_compile_time | Default compile-time → batch-independent |
| **CUDA graph constant batch** | grid_dims must be constant → replay assumes same shapes | #45309 cudagraph DSV4 crash | enforce_eager=True for DSV4 |
| **SM89 no wgmma** | FP8 storage only → requires FP8→BF16 conversion before compute | #184119, #187484 Inductor breaks | Triton fallback for FP8 on SM89 |
| **max_autotune layout deferral** | Layout choice deferred to runtime → may be batch-dependent | PyTorch v2.12 opt-in deferral | Reduces batch-dependent risk by default |
| **Inductor fx wrapper** | Non-triton fused kernels → different compile path | #187649 Inductor fx wrapper | May affect SM89 compilation |

---

## 6. vLLM Compilation Backends: 4 Paths

### 6.1 Backend Comparison

| Backend | Compilation Path | Batch Invariance | SM89 Risk | Status |
|---------|-----------------|-----------------|-----------|--------|
| **inductor** | torch.compile → Inductor → Triton kernels | LOW (prologue fusion) | ★★★★★★★★ HIGH | Default, needs P9 |
| **aot_eager** | torch.compile → piecewise AOT → eager fallbacks | HIGH (eager fallbacks) | ★★★★★★★★ LOW | #46085 NEW, promising |
| **cudagraph** | CUDA graph capture → replay | LOW (assumes constant batch) | ★★★★★★★★ HIGH | Default for non-DSV4 |
| **eager** | No compilation → direct PyTorch ops | PERFECT (no compile) | ★★★★★★★★ ZERO | enforce_eager=True |

★★★★★★★★★ **aot_eager is the SM89 middle ground**: It compiles PARTS of the model AOT (ahead of time) but falls back to eager for batch-dependent ops. This provides:
- Some compilation speedup (fused AOT parts)
- Batch invariance (eager fallbacks handle dynamic shapes)
- SM89 safety (no dangerous prologue fusion)

### 6.2 The aot_eager Piecewise Compilation (#46085)

aot_eager divides the model into "piecewise subgraphs":
- **Static subgraphs**: ops with fixed shapes → compile AOT → cudagraph-safe
- **Dynamic subgraphs**: ops with variable shapes → keep eager → batch-safe

This is mathematically equivalent to:

$$K_{\text{aot\_eager}} = K_{\text{static}} \cup K_{\text{dynamic}}$$
$$\text{grid\_dims}(K_{\text{static}}) = \text{constant} \perp \text{batch}$$
$$\text{grid\_dims}(K_{\text{dynamic}}) = \text{eager} \rightarrow \text{no\_cudagraph\_needed}$$

★★★★★★★★★ **This is potentially the BEST vLLM backend for RTX 4090**: combines compilation speedup with batch safety. Needs more investigation (#46085 deep reading pending).

---

## 7. The Complete SM89 Batch Invariance Strategy

### 7.1 Three Layers of Protection

```
Layer 1: P9 Fusion Guard (choices.py)
  → Prevents FP8→BF16 prologue fusion on SM<90
  → Keeps dequant as separate kernel
  → Makes prologue batch-independent

Layer 2: #187636 autotune_at_compile_time
  → Makes autotune choices batch-independent by default
  → Config chosen at compile time, not runtime
  → Prevents config-dependent grid changes

Layer 3: enforce_eager=True (no cudagraph)
  → Eliminates CUDA graph entirely
  → Every launch is dynamic
  → No replay with stale shapes
```

★★★★★★★★★ **All three layers are needed for RTX 4090 DSV4**:
- P9 alone: prevents prologue fusion but matmul still batch-dependent
- #187636 alone: prevents autotune batch-dependency but prologue still fused
- enforce_eager alone: no cudagraph but compilation may still produce bad kernels
- **P9 + #187636 + enforce_eager = COMPLETE SM89 safety**

### 7.2 For Dense Models (LLaMA, Qwen-3-8B)

```
Dense models on RTX 4090:
  → No FP8 → NO prologue fusion risk → P9 not needed
  → Standard matmul → batch-dependent grid → cudagraph risky
  → Options:
    1. cudagraph + padded_batch (standard vLLM) → SAFE if batch padding works
    2. enforce_eager=True → no cudagraph → SAFE but 10-15% slower
    3. aot_eager (if available) → partial compile → SAFE + some speedup
```

### 7.3 For MoE Models (Qwen-3-30B-A3B)

```
MoE models on RTX 4090:
  → FP8 weights → prologue fusion risk → P9 NEEDED
  → Dynamic routing → batch-dependent by nature → cudagraph IMPOSSIBLE
  → MUST: enforce_eager=True + P9 + #28676 cache invalidation
  → Best: aot_eager for static parts, eager for dynamic routing
```

---

## 8. Key Takeaways

★★★★★★★★★ **Inductor's fusion decisions are the mathematical root of SM89 batch invariance issues**: The prologue fusion (FP8→BF16 before compute) makes kernels batch-dependent, which breaks CUDA graph replay. P9 prevents this fusion on SM<90 GPUs.

★★★★★★★★★ **SM89 vs SM90 is a capability gap, not a bug**: SM89 LACKS wgmma, TMA, and Thread Block Clusters. These missing features force Inductor to make different (worse) fusion decisions. The fix is to REJECT those decisions on SM89, not to change the hardware.

★★★★★★★★★ **Three-layer protection is the complete solution**: P9 (prevent prologue fusion) + #187636 (batch-independent autotune) + enforce_eager=True (no cudagraph). Each layer addresses a different aspect of the batch invariance problem.

★★★★★★★★★ **aot_eager (#46085) is promising for RTX 4090**: Piecewise compilation provides partial speedup with full batch safety. This could be the optimal vLLM backend for SM89, combining performance with correctness.

★★★ **The compilation pipeline matters**: Understanding torch.compile → Inductor → Triton kernel pipeline is essential for debugging SM89 issues. The 4 vLLM backends (inductor, aot_eager, cudagraph, eager) have very different batch invariance profiles.

★★★ **autotune_at_compile_time (#187636) complements P9**: By making autotune decisions at compile time (symbolic shapes) instead of runtime (actual shapes), #187636 reduces the chance that kernel configurations change between batches. This is another layer of batch invariance protection.

---

## 9. References

- PyTorch #184119: SM89 fp8→bf16 prologue fusion guard (P9 validated)
- PyTorch #187636: autotune_at_compile_time default change
- PyTorch #187649: Inductor fx wrapper with non-triton fused kernel
- PyTorch #187484: vLLM w8a8 block-fp8 Inductor breaks on torch 2.13
- vLLM #45309: cudagraph DSV4 garbage output (1st DSV4 revert)
- vLLM #45972: eager_break DSV4 garbage output (2nd DSV4 revert)
- vLLM #46085: aot_eager piecewise compilation backend
- vLLM #45731: PyTorch 2.13 + torch.compile DRAFT
- SGLang #28618/#28620: SM89 DSV4-Flash-FP8 Triton fallback
- SGLang #28676: MXFP8 MoE shuffle cache clobber (10th DSV4 failure)
- p9-fusion-guard-integration-path-synthesis.md: 3-layer protection strategy
- P9 fork PR: jackie2049/pytorch PR #1

---

*End of theory note. Generated 2026-06-19.*
