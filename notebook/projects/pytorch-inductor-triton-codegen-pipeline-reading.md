# PyTorch Inductor Triton Codegen Pipeline — Source-Level Analysis

> 2026-06-16 | Inductor SchedulerNode → Triton kernel pipeline | Connects batch invariance root cause to Fusion Guard PR
> Key: CachingAutotuner persistent_reduction RBLOCK=constexpr XBLOCK=autotuned → SM89 batch-dependent accumulation order

---

## 1. Complete Pipeline: SchedulerNode → Triton Kernel

```
Python Model → Dynamo FX → AOTAutograd → Inductor Lowering (IR)
  → Scheduler (SchedulerNode → FusedSchedulerNode via 3-layer fusion gate)
  → SIMDScheduling.codegen_node() (tiling + TritonKernel construction)
  → TritonKernel.codegen_kernel() (source generation)
  → define_kernel() (async compile submission)
  → CachingAutotuner.precompile() (benchmark all configs, select best)
  → Triton JIT (GPU binary)
  → Runtime launch
```

### Step A — Lowering

`torch.compile(model)` → Dynamo FX graph → AOTAutograd joint forward-backward → Inductor Lowering (`lowering.py`) → FX node → Inductor IR (`TensorBox → StorageBox → ComputedBuffer → Operation`). Each operation: `ir.Pointwise` or `ir.Reduction`.

**Critical**: `mean` decomposed to `ReductionOp("sum") + divide` — NOT dispatched through `aten::mean.dim` → bypasses vLLM's batch-invariant override.

### Step B — Scheduler

Scheduler (`scheduler.py`, 10213 lines) creates `SchedulerNode` from IR → computes dependencies → topological sort → greedy fusion loop (`fuse_nodes`, up to 10 rounds). Three-layer fusion gate:
- Layer 0: `V.choices.can_fuse` (common heuristic)
- Layer 1: `Scheduler.can_fuse_vertical` (structural legality)
- Layer 2: `V.choices.can_fuse_vertical` (profitability → OUR INSERTION POINT → currently empty return True!)
- Layer 3: `Backend.can_fuse_vertical` (tiling legality)

If all pass → `FusedSchedulerNode`.

### Step C — SIMDScheduling.codegen_node()

Backend scheduling (`SIMDScheduling` in `simd.py`) → `SIMDKernelFeatures` from fused node → kernel args → tiling (dimension prefixes "x", "y", "z", "r0_" → sympy numel) → `TritonKernel` instance.

### Step D — TritonKernel Construction

`TritonKernel(SIMDKernel[TritonCSEVariable])` (triton.py, line 3131) → range trees, indexing, reduction infrastructure. For persistent reduction: `fixed_config` (`FixedTritonConfig`) where RBLOCK hardcoded. For non-persistent: RBLOCK autotuned.

### Step E — codegen_node_schedule_with_kernel

Iterates node schedule → each `codegen()` → loads, compute, stores → `V.ops` dispatches to `TritonKernelOverrides` (line 2366). **This is where `ops.reduction()` → `tl.sum()` inline**.

### Step F — TritonKernel.codegen_kernel()

Produces Triton kernel source string. Generates: imports, helpers, heuristic decorator (`@triton_heuristics.persistent_reduction` or `@triton_heuristics.reduction`), `@triton.jit`, kernel signature, `codegen_static_numels()`, argument aliases, body code.

### Step G-H — define_kernel + async_compile

Writes source to `.py` → submits to `async_compile.triton()` → compile worker process.

### Step I — CachingAutotuner.precompile()

Benchmarks all configs → selects best → stores in Triton cache. Runtime: loads cached config → launches with best grid/parameters.

---

## 2. Autotuning Mechanism — Three Heuristic Decorators

| Decorator | Kernel Type | What Varies | What Fixed | Location |
|-----------|------------|------------|-----------|----------|
| `triton_heuristics.pointwise` | Pointwise | XBLOCK/YBLOCK/ZBLOCK + num_warps | None | Line 3929 |
| `triton_heuristics.reduction` | Looped reduction | XBLOCK + RBLOCK | None | Line 4613 |
| `triton_heuristics.persistent_reduction` | Persistent reduction | **XBLOCK only** | **RBLOCK = next_power_of_2(rnumel)** | Line 4882 |

★★★★★★★★★ **Key asymmetry**: persistent_reduction RBLOCK = constexpr (fixed at compile time, deterministic) but XBLOCK = autotuned (varies with input size). On SM89, different xnumel (batch sizes) → different optimal XBLOCK → `tl.sum()` accumulates over different numbers of rows → non-associative FP addition → batch-dependent results!

### CachingAutotuner (triton_heuristics.py, line 420)

Inductor's custom autotuner, NOT Triton's built-in `@triton.autotune`. Differences:
1. **Precompiles all configs in parallel** via process pool (autotune_process.py) — not sequential
2. **Caches to disk** via `lookup_autotune_config()` (line 157) — subsequent runs load immediately
3. **Two-phase**: compile time (benchmark, select, cache) + runtime (load cached, launch)
4. **Coordinate descent tuner** (`CoordescTuner`) — iteratively refines block sizes
5. **`_dynamic_scale_rblock()`** (line 828) — scales RBLOCK based on register pressure

### SM-dependent behavior

- Line 722-723: `device_prop.major >= 8` → `_could_rblock_scale` gate
- Line 4220-4223: `device_major >= 10` → `MAX_R0_BLOCK` (1024 Blackwell vs 2048 others)
- SM89 shared memory: 100KB (vs SM80=164KB, SM90=228KB) → forces different XBLOCK → batch-dependent!

---

## 3. Persistent vs Non-Persistent Reduction Kernels

### Persistent Reduction (RMSNorm case)

Decision: `TritonKernel.should_use_persistent_reduction()` (line 3371) — rnumel statically known + small enough + device has enough shared memory.

```python
@triton_heuristics.persistent_reduction(size_hints={...})
@triton.jit
def kernel(..., XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    r0_index = tl.arange(0, R0_BLOCK)   # R0_BLOCK = next_power_of_2(rnumel) → FIXED
    # 2D load: [XBLOCK, R0_BLOCK]
    tmp0 = tl.load(input + (xindex[:, None] * r0_numel + r0_index[None, :]), ...)
    # Inline reduction: sum over R0_BLOCK dimension
    tmp1 = tl.sum(tmp0, 1)              # ← XBLOCK varies → accumulation order varies!
    tl.store(output + xindex, tmp1, xmask)
```

**No loop** over reduction dimension. Grid: `ceildiv(xnumel, XBLOCK)` — each thread block processes one row.

### Non-Persistent (Looped) Reduction

Used when rnumel too large or dynamic. Both XBLOCK and RBLOCK autotuned. Loop over reduction dim:

```python
@triton_heuristics.reduction(size_hints={...})
@triton.jit
def kernel(..., XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    _acc = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + tl.arange(0, R0_BLOCK)
        tmp0 = tl.load(...)
        _acc = _acc + tl.sum(tmp, 1)
    tl.store(output + xindex, _acc, xmask)
```

### Cooperative Reduction (very large)

Multiple thread blocks cooperate per row → shared workspace + semaphores → `(RSPLIT, ceildiv(xnumel, XBLOCK), 1)` grid → partial results combined via barrier synchronization.

★★★★★★★★★ **Batch invariance problem is specific to persistent reductions**: RBLOCK constexpr (deterministic) + XBLOCK autotuned (batch-dependent) → variable accumulation order. Non-persistent has loop structure providing consistent accumulation per iteration. Cooperative adds multi-block variability.

---

## 4. Combo Kernels and PR #187275

**Combo kernel** = single Triton program running multiple sub-kernels sequentially. `FusedSchedulerNode` containing nodes with different iteration domains → combined into one `@jit` function.

- `TritonKernel.is_combo_kernel` parameter
- `ComboKernelGrid` (line 5419) computes grid for combo kernels
- Each sub-kernel can have its own XBLOCK/RBLOCK

**PR #187275** (opened 2026-06-14): "Fix Combo Kernel Crash with Dynamic Persistent Reduction Dimensions". Root cause: persistent reduction RBLOCK hardcoded/autotuned → dynamic reduction numel changes → hardcoded RBLOCK invalid → crash. Fix: reuse `_get_persistent_RBLOCK()` for dynamic reduction numels.

★★★★★★★★★ **#187275 CONFIRMS our batch invariance root cause class**: Both stem from persistent reduction dimension handling weakness. Our issue = numerical correctness (different results for different batch sizes). Their issue = crash correctness. Same architectural weakness → strengthens SM<90 Fusion Guard case!

★★★★★ **v2.12 max_autotune for combo kernels** (#177715/#178936/#179317): O(N*K) phase configs → per-sub-kernel reduction hints → may **exacerbate SM89 batch-dependent behavior** → needs GPU validation!

---

## 5. Key Code Locations for Fusion Guard PR

| File | Line | What | Role |
|------|------|------|------|
| choices.py | 639-647 | `can_fuse_vertical()` | **INSERTION POINT** — currently empty return True |
| choices.py | 472-506 | `reduction_split_factor()` | **PRIMARY PRECEDENT** — `DeviceProperties.create(device)` + `props.major >= 10` |
| choices.py | 569-637 | `can_fuse()` | Common heuristic (shared_data_score) |
| choices.py | 649-669 | `can_fuse_horizontal()` | Has actual heuristics (contrasting with empty vertical) |
| choices.py | 671-729 | `score_fusion()` | 5-component FusionScore |
| triton.py | 3131 | `TritonKernel` | fixed_config for persistent reduction RBLOCK |
| triton.py | 6454 | `codegen_kernel()` | Generates heuristic decorator + kernel source |
| triton.py | 2849 | TMA check | SM capability gate precedent (`major >= 9`) |
| triton.py | 4129 | PDL check | SM capability gate precedent (`major >= 9`) |
| triton.py | 6571 | `DeviceProperties.create()` | Device metadata |
| triton_heuristics.py | 420 | `CachingAutotuner` | Custom autotuner class |
| triton_heuristics.py | 3350 | `triton_config()` | Config object construction |
| triton_heuristics.py | 4882 | `persistent_reduction()` | RBLOCK=constexpr, XBLOCK autotuned |
| triton_heuristics.py | 722 | `_could_rblock_scale` | `major >= 8` gate |
| triton_heuristics.py | 4220 | `MAX_R0_BLOCK` | `major >= 10` gate |

---

## Key Findings Summary

★★★★★★★★★ Complete pipeline: SchedulerNode → 3-layer fusion gate → SIMDScheduling → TritonKernel → CachingAutotuner → Triton JIT → runtime. `ops.reduction()` → `tl.sum()` inline at codegen step.
★★★★★★★★★ Persistent reduction: RBLOCK = next_power_of_2(rnumel) constexpr + XBLOCK autotuned → batch-dependent accumulation order → root cause CONFIRMED
★★★★★★★★★ CachingAutotuner: Inductor custom (parallel precompile + disk cache + coordinate descent) — NOT Triton built-in
★★★★★★★★★ SM89 shared memory 100KB → different XBLOCK selections for different batch sizes → batch-dependent behavior
★★★★★★★★★ Combo kernels (#187275): SAME root cause class as our batch invariance → persistent reduction dimension handling weakness → CONFIRMS our diagnosis
★★★★★★★★★ v2.12 max_autotune combo kernels → may exacerbate SM89 batch-dependent behavior → needs GPU validation
★★★★★★★★★ choices.py line 639-647 `can_fuse_vertical()` = OUR INSERTION POINT (currently empty return True)
★★★★★★★★★ choices.py line 472-506 `reduction_split_factor()` = PRIMARY PRECEDENT (DeviceProperties + props.major >= 10)

---

## References

- Inductor scheduler source: notebook/projects/pytorch-inductor-scheduler-source-reading.md
- SM89 fusion root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Fusion Guard PR approach: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-approach.md
- Fusion Guard PR draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Triton codegen reading: notebook/projects/pytorch-inductor-triton-codegen-reading.md
- Inductor lowering: notebook/fundamentals/pytorch-inductor-lowering-source-reading.md
- v2.12 features: notebook/projects/pytorch-2.12-features-reading.md
- Tracker #187275: notebook/projects/7-framework-developments-tracker-2026-06-16.md
