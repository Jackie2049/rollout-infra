# PyTorch Inductor Triton Kernel Codegen Pipeline -- Deep Source Reading

## 1. Inductor IR -> Triton Codegen: The Full Pipeline

### 1.1 IR Class Hierarchy (torch/_inductor/ir.py, 11765 lines)

Inductor's IR is built on three key classes that form a layered abstraction:

- **TensorBox**: The outermost wrapper. Wraps a `Buffer` and provides a "tensor view" with sizes/strides. Most operations in the lowering phase operate on `TensorBox` objects. It delegates storage to its inner `Buffer`.

- **StorageBox**: The middle layer. Wraps a `Buffer` and manages the data layout (whether the buffer is realized, its memory location, mutation tracking). A `StorageBox` can be "unrealized" (computed on the fly) or "realized" (allocated in memory).

- **Buffer / ComputedBuffer**: The innermost layer. A `Buffer` represents a named memory allocation. A `ComputedBuffer` additionally stores the `Storage` (how the data is computed) and the `Layout` (size, stride, offset). The `Storage` contains an `ir.Operation` (e.g., `ir.Pointwise`, `ir.Reduction`) which is the actual computation description.

The flow is: `TensorBox(StorageBox(ComputedBuffer(Layout, Operation)))`. When Inductor decides to "realize" a buffer, the `StorageBox` is marked as realized and the buffer gets a concrete memory allocation.

### 1.2 Lowering to Scheduler Nodes

When `torch.compile` processes a model, the following happens:

1. **Dynamo** captures the FX graph (Python bytecode -> FX IR).
2. **AOTAutograd** handles autograd decomposition, producing a joint forward-backward graph.
3. **Inductor Lowering** (torch/_inductor/lowering.py) converts each FX node into Inductor IR (`TensorBox` -> `Buffer` -> `Operation`). Each operation gets a `ir.ComputedBuffer` with either `ir.Pointwise` (for elementwise ops) or `ir.Reduction` (for reduction ops) storage.
4. **Scheduler** (torch/_inductor/scheduler.py, 10204 lines) groups these IR nodes into `SchedulerNode` and `FusedSchedulerNode` objects. It decides which nodes to fuse and which backend to use.

### 1.3 TritonKernel Class (torch/_inductor/codegen/triton.py, 7810 lines, line 3131)

`TritonKernel(SIMDKernel[TritonCSEVariable])` is the central class that generates Triton kernel source code. It inherits from `SIMDKernel` (in simd.py, line 514), which provides common infrastructure for SIMD-style codegen (flattened indexing, range trees, iteration variables).

**Key init parameters**:
- `tiling`: Dict mapping dimension prefixes ("x", "y", "z", "r0_") to sympy numel expressions. This defines the iteration space.
- `fixed_config`: Optional `FixedTritonConfig` for persistent reductions where block sizes are hardcoded.
- `is_combo_kernel`: Whether this kernel is part of a combo kernel (multiple sub-kernels in one Triton program).
- `min_elem_per_thread`: Minimum elements per thread (relevant for FP8 type conversions).

**Init sequence** (line 3147-3201):
1. Call `super().__init__()` -> `SIMDKernel.__init__()` which sets up `self.range_trees`, `self.numels`, `self.inside_reduction`, `self.persistent_reduction`, `self.cooperative_reduction`.
2. Create `TritonCSE` for common subexpression elimination.
3. Create `IndentedBuffer` sections: `self.prologue`, `self.post_loop_combine`, `self.post_loop_store`.
4. If `inside_reduction`, call `codegen_reduction_numels()` to generate reduction dimension numel code.
5. If `cooperative_reduction`, call `init_cooperative_reduction()` (semaphore setup, rsplit indexing).
6. Call `codegen_range_tree()` to emit iteration range headers (xindex, yindex, etc.).
7. If `cooperative_reduction`, call `init_cooperative_reduction_mask()`.

### 1.4 SIMDKernel Base (torch/_inductor/codegen/simd.py, line 514)

`SIMDKernel(Kernel[CSEVariableType])` provides the shared infrastructure:

- **Range Trees** (`IterationRangesRoot`): Each iteration dimension is represented as a range tree. For a pointwise kernel with numel N, there's one range tree for "x". For a reduction kernel, there are range trees for "x" (non-reduced) and "r0_" (reduced). Each range tree generates code like:
  ```python
  xoffset = tl.program_id(0) * XBLOCK
  xindex = xoffset + tl.arange(0, XBLOCK)
  xmask = xindex < xnumel
  ```

- **Iteration Ranges**: The `IterationRangesRoot` class manages the indexing for each dimension. It knows whether it's a reduction dimension (`is_reduction`), whether it should be looped (`is_loop`), and which grid dimension it maps to (`grid_dim`).

- **Key properties**:
  - `self.inside_reduction`: Whether we're currently inside a reduction loop.
  - `self.persistent_reduction`: Whether the reduction can be completed in a single program (no loop needed). When True, RBLOCK is fixed at compile time and the entire reduction happens in one pass.
  - `self.cooperative_reduction`: Whether multiple thread blocks cooperate on the same reduction using shared workspace and semaphores.
  - `self.no_x_dim`: Whether the pointwise dimension is eliminated (pure reduction like scalar reduction).

### 1.5 Code Generation Flow

The TritonKernel code generation follows this sequence:

**Step 1: codegen_node_schedule** (SIMDScheduling, line 3096)
- Takes a `SIMDKernelFeatures` object containing the fused node schedule.
- Creates kernel args, determines tiling, and constructs the `TritonKernel`.

**Step 2: TritonKernel construction**
- Sets up range trees, indexing, reduction infrastructure.

**Step 3: codegen_node_schedule_with_kernel** (SIMDScheduling, line 3212)
- Iterates over the node schedule, calling each scheduler node's `codegen()` method.
- Each node generates loads, compute, and stores into the kernel's buffer sections.

**Step 4: codegen_kernel** (TritonKernel, line 6454)
- The main method that produces the final Triton kernel source string.
- Generates:
  1. Common Triton imports (`import triton`, `import triton.language as tl`, etc.)
  2. Helper functions (from `self.helper_functions`)
  3. Heuristic decorator (`@triton_heuristics.pointwise(...)` or `@triton_heuristics.reduction(...)`)
  4. `@triton.jit` decorator
  5. Kernel function signature with all argdefs
  6. `codegen_static_numels()` - hardcodes static numel values for better Triton optimization
  7. Argument aliases
  8. Body code (from `self.body` buffer which contains prologue + indexing + loads + compute + stores)

**The generated Triton kernel looks like**:
```python
@triton_heuristics.pointwise(
    size_hints={...}, filename=__file__,
    triton_meta={...}, inductor_meta={...})
@triton.jit
def triton_poi_fused__0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (xindex), xmask)
    tmp1 = tmp0 + 1.0
    tl.store(out_ptr0 + (xindex), tmp1, xmask)
```

### 1.6 TritonPrinter and TritonOverrides (triton.py, lines 907 and 1307)

- **TritonPrinter(PythonPrinter)**: Converts sympy expressions to Triton-compatible Python strings. Handles special sympy functions like `FloorDiv`, `ModularIndexing`, `CeilDiv` by mapping them to Triton idioms (e.g., `FloorDiv` -> `//` for non-negative, `triton_helpers.div_floor_integer()` for general case).

- **TritonOverrides(OpOverrides)**: Maps PyTorch elementwise operations to Triton API calls. For example:
  - `ops.to_dtype(x, dtype)` -> `x.to(triton_type(dtype))`
  - `ops.abs(x)` -> `libdevice.abs(x)`
  - `ops.sin(x)` -> `libdevice.sin(x)`
  - `ops.exp(x)` -> `libdevice.exp(x)`
  - `ops.relu(x)` -> `triton_helpers.relu(x)`

- **TritonKernelOverrides(TritonOverrides)** (line 2366): Extends TritonOverrides with kernel-level operations like:
  - `ops.load(name, index)` -> delegates to `TritonKernel.load()`
  - `ops.store(name, index, value)` -> delegates to `TritonKernel.store()`
  - `ops.reduction(dtype, src_dtype, reduction_type, value)` -> delegates to `TritonKernel.reduction()`

The `OpsHandler` pattern uses `V.ops` (the virtualized ops handler) which dispatches to the currently active kernel's overrides. When `V.set_kernel_handler(kernel)` is called, all ops calls go to that kernel's overrides.


## 2. JIT Compilation: How Triton Kernels Are JIT-Compiled at Runtime

### 2.1 The CachingAutotuner (triton_heuristics.py, line 420)

Inductor does NOT use Triton's built-in `@triton.autotune` decorator. Instead, it uses its own `CachingAutotuner(KernelInterface)` which:

1. **Precompiles all configs**: Unlike Triton's Autotuner which compiles configs one at a time during benchmarking, Inductor's `CachingAutotuner` can precompile all configs in parallel using a process pool.

2. **Caches to disk**: The best config is cached to disk using `lookup_autotune_config()` (line 157). On subsequent runs, the cached config is loaded immediately, skipping autotuning entirely.

3. **Two-phase compilation**:
   - Phase 1 (compile time): The `async_compile.triton()` call (in TritonScheduling._emit_kernel_to_wrapper, line 7396) submits the kernel source to a process pool worker. The worker runs `CachingAutotuner.precompile()` which:
     a. Looks up cached config (if available, uses only that config)
     b. For each config, compiles the kernel using Triton's JIT compiler
     c. Benchmarks each compiled kernel variant
     d. Selects the best config
   - Phase 2 (runtime): The compiled kernel is loaded from Triton's cache and launched.

### 2.2 The Heuristic Decorators

The generated kernel source is decorated with one of three heuristic functions:

- **`triton_heuristics.pointwise(size_hints, ...)`** (line 3929): Generates configs for pointwise kernels. Configs are `triton_config()` objects specifying XBLOCK, YBLOCK, ZBLOCK sizes, num_warps, and num_stages.

- **`triton_heuristics.reduction(size_hints, reduction_hint, ...)`** (line 4613): Generates configs for reduction kernels. Configs specify XBLOCK and RBLOCK (reduction block) sizes. The reduction_hint provides information about the reduction type (e.g., small inner dim, large outer dim).

- **`triton_heuristics.persistent_reduction(size_hints, reduction_hint, ...)`** (line 4882): Generates configs for persistent reductions where RBLOCK = rnumel (the entire reduction dimension fits in one pass). The XBLOCK varies.

Each decorator calls `cached_autotune()` which creates a `CachingAutotuner` instance. The `CachingAutotuner` wraps the `@triton.jit` decorated kernel function.

### 2.3 triton_config() Function (line 3350)

The `triton_config()` function constructs a Triton Config object with block sizes. Key logic:

1. Start with the initial block sizes (e.g., `x=256`).
2. Scale up to `size_hints` or `TRITON_MAX_BLOCK` if the grid would be too large.
3. Calculate `num_warps` based on `num_elements_per_warp` (default 256 elements per warp, so a 1024-element block gets 4 warps).
4. Enforce minimum 4 warps for PTX bug workaround.

### 2.4 Grid Types

The grid (number of thread blocks to launch) is determined by `GridExpr` subclasses:

- **Grid1D** (line 5317): `grid = (ceildiv(xnumel, XBLOCK), 1, 1)`. Used for 1D pointwise kernels.
- **Grid2D** (line 5322): `grid = (ceildiv(xnumel, XBLOCK), ceildiv(ynumel, YBLOCK), 1)`. Used for 2D pointwise kernels.
- **Grid3D** (line 5328): Similar for 3D kernels.
- **CooperativeReductionGrid** (line 5377): `grid = (RSPLIT, ceildiv(xnumel, XBLOCK), 1)`. The first dimension is RSPLIT (number of thread blocks cooperating on one reduction row).
- **MixOrderReductionGrid** (line 5363): `grid = (ceildiv(xnumel, RSPLIT_SIZE), 1, 1)`. Used for mix-order reductions where the X dimension is split into RSPLIT_SIZE chunks.
- **SplitScanGrid** (line 5383): `grid = (ceildiv(r0_numel, R0_BLOCK), xnumel, 1)`. Scans with decoupled look-back.
- **FixedGrid** (line 5394): Grid sizes are precomputed and passed as arguments.
- **ComboKernelGrid** (line 5419): For combo kernels that run multiple sub-kernels in a single Triton program.

### 2.5 Async Compilation Pipeline

The full JIT compilation pipeline works as follows:

1. **First torch.compile call**: `TritonScheduling.define_kernel()` generates kernel source code.
2. The source is written to a `.py` file in the Triton cache directory via `async_compile.triton()`.
3. A **compile worker process** (`autotune_process.py`) picks up the kernel, creates a `CachingAutotuner`, and:
   a. Calls `precompile()` which compiles all configs via Triton JIT
   b. Benchmarks each config using `triton.testing.do_bench()`
   c. Selects the best config
   d. Returns the compiled kernel binary path
4. The compiled kernel is stored in Triton's cache (`~/.triton/cache/`).
5. **Subsequent calls**: The `CachingAutotuner.run()` method loads the compiled kernel from cache and launches it with the best config's grid and launch parameters.

### 2.6 Coordinate Descent Tuner (CoordescTuner)

In addition to the pre-defined configs, Inductor uses a **coordinate descent tuner** that iteratively refines block sizes starting from the best pre-defined config. It:

1. Starts with the best config from the initial benchmarking.
2. Halves the RBLOCK (for reductions) and re-benchmarks.
3. Adjusts XBLOCK up/down and re-benchmarks.
4. Uses `_dynamic_scale_rblock()` (line 828) to dynamically scale the reduction block size based on register pressure.


## 3. Autotuning: Config Search and Selection

### 3.1 Config Generation for Pointwise Kernels

For **pointwise kernels**, the `triton_heuristics.pointwise()` function generates configs:

- **1D case** (single "x" dimension): Generates configs with various XBLOCK sizes (256, 512, 1024, 2048) and num_warps (1, 2, 4, 8, 16, 32).
- **2D case** (x and y dimensions): Generates configs where one dimension gets the full block and the other gets 1, e.g., `(XBLOCK=256, YBLOCK=1)` and `(XBLOCK=1, YBLOCK=256)`.
- **3D case**: Similar patterns with one hot dimension.

The `tile_hint` parameter further constrains the search space. `TileHint` values include `SQUARE`, `NARROW`, `WIDE`, etc.

### 3.2 Config Generation for Reduction Kernels

For **reduction kernels**, `_reduction_configs()` generates configs with:

- XBLOCK sizes: typically powers of 2 from 32 to 1024.
- RBLOCK sizes: powers of 2 from 32 to 2048 (for non-persistent reductions).
- The `reduction_hint` parameter provides information about the reduction pattern:
  - `ReductionHint.INNER` (default): The reduction dimension is innermost (contiguous).
  - `ReductionHint.OUTER`: The reduction dimension is outermost.
  - `ReductionHint.SAME`: The reduction dimension is the same as the output.

### 3.3 Persistent Reduction Configs

For **persistent reductions** (where the entire reduction dimension fits in one thread block):

- `_persistent_reduction_configs()` generates configs where RBLOCK = rnumel (the full reduction size, rounded up to power of 2).
- XBLOCK varies, but typically starts small (8, 16, 32) since each thread block processes one row.
- The `no_x_dim` case (XBLOCK=1) is used for pure scalar reductions.

### 3.4 Autotune Hints (torch/_inductor/runtime/hints.py)

`AutotuneHint` values can be attached to kernel metadata to suggest specific configs:

- `ONE_ELEMENT_PER_THREAD`: Suggests configs where each thread processes exactly one element (relevant for memory-bound operations with complex indexing).

These hints are processed by `autotune_hints_to_configs()` (line 180) which generates additional configs to try.

### 3.5 Benchmarking and Selection

The `CachingAutotuner` benchmarks all configs and selects the fastest one. The benchmarking uses `triton.testing.do_bench()` which:

1. Warms up the kernel.
2. Measures execution time over multiple iterations.
3. Returns the median time.

The best config is cached to disk via `save_cache_hook`. On subsequent runs, `lookup_autotune_config()` retrieves the cached config, skipping the entire autotuning process.

### 3.6 Multi-Kernel Selection (multi_kernel.py)

When `config.triton.multi_kernel` is enabled (line 7602), Inductor creates multiple kernel variants with different reduction strategies:

1. **Persistent reduction** (default for small reductions): RBLOCK = rnumel, one thread block per row.
2. **Non-persistent reduction**: RBLOCK < rnumel, loop over the reduction dimension.
3. **Cooperative reduction**: Multiple thread blocks cooperate via workspace + semaphores.
4. **Non-cooperative reduction**: Each thread block works independently.

All variants are benchmarked, and the `MultiKernel` class (in multi_kernel.py) selects the fastest variant at runtime based on input sizes. The selection uses a hash of input sizes to map to the best kernel variant.


## 4. Kernel Fusion: Scheduler's can_fuse Logic

### 4.1 Fusion Decision Hierarchy

Fusion decisions are made at two levels:

**Level 1: Scheduler** (`scheduler.py`, `Scheduler.can_fuse()` at line 7557):
- Determines whether two nodes can legally be fused (no data hazards, no cycles, compatible devices).
- Calls the backend-specific `can_fuse()` method.

**Level 2: Backend** (`SIMDScheduling.can_fuse()` at line 2022 in `simd.py`):
- Determines whether the Triton backend can practically fuse two nodes.
- Checks size/tiling compatibility.

### 4.2 SIMDScheduling.can_fuse() (simd.py, line 2022)

The backend fusion gate checks the following conditions:

1. **ForeachKernelSchedulerNode**: Delegates to `ForeachKernelSchedulerNode.can_fuse()` for foreach (multi-tensor) operations.

2. **Split scan vs reduction**: Split scan kernels cannot fuse with reductions.

3. **Two reductions**: Can fuse if they have the same numel/rnumel. If not, checks for:
   - **MixOrderReduction.can_fuse()** (scheduler.py, line 344): Allows fusing two reductions with different loop orders if they share common read buffers and the workload is large enough (>5MB).
   - **NestedReduction.can_fuse()** (scheduler.py, line 847): Allows fusing a grouped reduction with its parent reduction.

4. **Native matmul compatibility**: If one node is a native matmul, the other must have compatible tiling (z,y,x order).

5. **Two pointwise nodes**: Can fuse if they have the same numel/rnumel. If one is a template node, always allow fusion (for prologue/epilogue).

6. **Pointwise + reduction (producer-consumer)**: Can fuse if the pointwise node's output is only consumed by the reduction. The pointwise becomes the "epilogue" of the reduction.

7. **Tiling compatibility**: For non-template fusions, checks that `select_tiling()` produces compatible tilings for both nodes. Bad combined tilings (where block sizes would be too small) are rejected.

### 4.3 FusedSchedulerNode (scheduler.py, line 2664)

When nodes are fused, they become a `FusedSchedulerNode` which:

- Contains a list of constituent `SchedulerNode` objects (`self.snodes`).
- Maintains unmet dependencies as the union of constituent nodes.
- Computes the fused group as `(product(sizes[0]), product(sizes[1]))` where `sizes[0]` is the pointwise dimensions and `sizes[1]` is the reduction dimensions.
- Delegates codegen to each constituent node.

### 4.4 WhyNoFuse (scheduler.py, line 2076)

`WhyNoFuse` is a diagnostic helper that records the reason why two nodes cannot be fused. It's used for debugging and logging:

```python
why = WhyNoFuse(node1, node2)
if numel1 != numel2:
    why("numel mismatch (%s, %s)", numel1, numel2)
    return False
```

Each `WhyNoFuse` call appends a reason string, and the full list of reasons is logged when fusion is rejected.

### 4.5 How Fused Kernels Share Memory

When multiple nodes are fused into a single Triton kernel, they share memory through:

1. **Intermediate buffer elimination**: If node1's output is only consumed by node2 (in the same fused kernel), the intermediate buffer is never materialized in global memory. Instead, the computation chain stays in registers/shared memory:
   ```python
   # Instead of:
   tmp0 = tl.load(in_ptr0 + xindex, xmask)   # node1 load
   tl.store(out_ptr0 + xindex, tmp0, xmask)   # node1 store
   tmp1 = tl.load(out_ptr0 + xindex, xmask)   # node2 load (same data!)
   tmp2 = tmp1 * 2.0                           # node2 compute
   tl.store(out_ptr1 + xindex, tmp2, xmask)   # node2 store

   # Fused kernel generates:
   tmp0 = tl.load(in_ptr0 + xindex, xmask)    # node1 load
   tmp2 = tmp0 * 2.0                           # directly use tmp0
   tl.store(out_ptr1 + xindex, tmp2, xmask)   # only final store
   ```

2. **CSE (Common Subexpression Elimination)**: `TritonCSE` (triton.py, line 2810) tracks which values have already been computed and reuses them across fused operations. When two fused nodes both load the same input buffer at the same index, CSE ensures only one `tl.load` is generated.

3. **Buffer removal**: The scheduler's `removed_buffers` set tracks intermediate buffers that don't need to be allocated because they're consumed entirely within the fused kernel.

### 4.6 Fusion Benchmarking

After creating a `FusedSchedulerNode`, Inductor optionally benchmarks the fused kernel vs. separate kernels to decide whether fusion actually improves performance. `fuse_if_speedup()` (scheduler.py, line 5972) only fuses if the benchmark shows a speedup.

The benchmarking process:
1. Generate source code for the fused kernel.
2. Generate source code for each unfused kernel separately.
3. Load both via `PyCodeCache.load()`.
4. Run `do_bench()` on each.
5. Compare execution times.


## 5. Pointwise vs Reduction Kernels

### 5.1 Pointwise Kernels

**Grid**: 1D, 2D, or 3D depending on the number of non-reduction dimensions. For example, a pointwise kernel with numel N uses `Grid1D`: `grid = (ceildiv(N, XBLOCK), 1, 1)`.

**Memory access pattern**: Each thread block processes a contiguous chunk of the output tensor. The block loads input elements at the same indices as the output, performs elementwise computation, and stores results.

**Generated code structure**:
```python
@triton_heuristics.pointwise(size_hints={...})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    # loads
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    # compute
    tmp1 = tmp0 + 1.0
    # stores
    tl.store(out_ptr0 + xindex, tmp1, xmask)
```

**Key characteristics**:
- No reduction dimension.
- No loop over the iteration space.
- Each element is processed independently.
- The heuristic is `triton_heuristics.pointwise`.

### 5.2 Reduction Kernels

**Grid**: Depends on reduction strategy:
- **Persistent reduction**: `grid = (ceildiv(xnumel, XBLOCK), 1, 1)` where each thread block processes one row.
- **Non-persistent (looped) reduction**: Same grid as persistent, but the kernel internally loops over the reduction dimension.
- **Cooperative reduction**: `grid = (RSPLIT, ceildiv(xnumel, XBLOCK), 1)` where multiple thread blocks cooperate on each row.

**Memory access pattern**: Each thread block loads a "stripe" of the input tensor along the reduction dimension, accumulates a partial result, and either:
- Stores directly (persistent: entire reduction in one pass)
- Loops and accumulates (non-persistent: reduction in multiple passes)
- Cooperates with other thread blocks (cooperative: partial results combined via workspace + barrier)

**Generated code structure (persistent reduction)**:
```python
@triton_heuristics.persistent_reduction(size_hints={...})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    r0_offset = 0
    R0_BLOCK: tl.constexpr = <next_power_of_2(r0_numel)>
    r0_index = r0_offset + tl.arange(0, R0_BLOCK)
    r0_mask = r0_index < r0_numel
    # loads (2D: [XBLOCK, R0_BLOCK])
    tmp0 = tl.load(in_ptr0 + (xindex[:, None] * r0_numel + r0_index[None, :]), xmask[:, None] & r0_mask[None, :])
    # reduction (sum over R0_BLOCK dimension)
    tmp1 = tl.sum(tmp0, 1)
    # store (1D: [XBLOCK])
    tl.store(out_ptr0 + xindex, tmp1, xmask)
```

**Generated code structure (looped reduction)**:
```python
@triton_heuristics.reduction(size_hints={...})
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    # accumulator init
    _tmp0 = tl.full([XBLOCK], 0.0, tl.float32)
    # loop over reduction dimension
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + tl.arange(0, R0_BLOCK)
        r0_mask = r0_index < r0_numel
        tmp0 = tl.load(in_ptr0 + (xindex[:, None] * r0_numel + r0_index[None, :]), ...)
        _tmp0 = _tmp0 + tl.sum(tmp0, 1)
    # store final result
    tl.store(out_ptr0 + xindex, _tmp0, xmask)
```

### 5.3 Key Differences

| Aspect | Pointwise | Reduction |
|--------|-----------|-----------|
| Grid | 1D/2D/3D (output numel) | Depends on strategy (xnumel, rnumel) |
| Heuristic | `pointwise` | `reduction` / `persistent_reduction` |
| Block sizes | XBLOCK, YBLOCK, ZBLOCK | XBLOCK + RBLOCK |
| Memory pattern | 1D contiguous per element | 2D stripe + accumulator |
| Loop | None | Optional (`tl.range` over reduction dim) |
| Output shape | Same as input | Reduced dimension |
| Persistent | N/A | RBLOCK = rnumel, no loop |
| Cooperative | N/A | Multiple blocks per row, semaphore sync |


## 6. Epilogue Fusion: GEMM + Pointwise Operations

### 6.1 Epilogue Fusion Concept

Epilogue fusion allows fusing a pointwise operation (like bias add, residual connection, ReLU) directly into a GEMM kernel's output path, avoiding the overhead of a separate kernel launch and an extra global memory round-trip.

For example, `C = A @ B + bias` can be fused so that:
- The GEMM computes `A @ B` into registers/shared memory
- The bias is added in-place before writing to global memory
- No intermediate buffer for `A @ B` result

### 6.2 TritonTemplateKernel (select_algorithm.py, line 524)

`TritonTemplateKernel(TritonKernel)` is the class for templated Triton kernels (GEMM, BMM, etc.). It extends TritonKernel with:

- `input_nodes`: The template's input tensors (A, B, and optionally bias for addmm).
- `output_node`: The template's output tensor.
- `epilogue_fn`: A function that applies the manual epilogue (e.g., `lambda acc, bias: acc + bias` for addmm).
- `subgraphs`: List of `ir.ComputedBuffer` subgraphs for prologue/epilogue fusion.
- `prefix_args` / `suffix_args`: Number of arguments before/after the main template arguments that are epilogue/prologue inputs.

### 6.3 Epilogue Fusion Mechanism

The epilogue fusion works through `store_output()` (select_algorithm.py, line 1459):

1. **Create subgraph body**: The kernel enters a subgraph context (`create_subgraph_body`) which isolates the epilogue computation.
2. **Compute epilogue**: The template output value (`val`) becomes the first epilogue argument. Additional epilogue inputs (bias, residual) are loaded via `input_node.make_loader()`.
3. **Apply epilogue_fn**: The manual epilogue function is applied (e.g., bias add for addmm).
4. **Fused epilogue ops**: If `_epilogue_nodes_by_subgraph` is set, additional pointwise ops are applied after the manual epilogue:
   ```python
   epilogue_result = self.epilogue_fn(*epilogue_args)  # manual epilogue (e.g., bias add)
   # Fused epilogue ops continue from epilogue_result
   ```
5. **Store**: The final result is stored to the output buffer.

### 6.4 _fuse_epilogue Decision (scheduler.py, line 3827)

`_fuse_epilogue()` decides whether to fuse an epilogue into a GEMM template by:

1. Checking occupancy impact: `_occupancy_before_and_after_fusion()` computes how many thread blocks can be resident on the GPU before and after fusion (based on register usage).
2. If fused kernel has >= 4 resident blocks, always fuse.
3. If occupancy drops but the ratio > 0.5 (fused/unfused), still fuse.
4. If epilogue is memory-bound and fused still has >1 block, fuse.

### 6.5 FusedExternTritonKernelSchedulerNode (scheduler.py, line 3141)

This is the scheduler node type for epilogue-fused GEMM templates. It contains:
- `kernel_node`: The `ExternKernelSchedulerNode` with the `ir.UserDefinedTritonKernel` (the GEMM template).
- `fused_epilogue`: The `SchedulerNode` containing the epilogue pointwise ops.

The `codegen()` method (line 3165):
1. Creates a `FusedUserDefinedTritonKernel` which generates Triton code for the epilogue ops.
2. The new kernel source is injected into the original Triton template's store operation.
3. The template's AST is modified to replace the store value with the epilogue computation result.

### 6.6 Triton GEMM Templates (kernel/mm.py)

The Triton GEMM templates are defined as `TritonTemplate` objects in `kernel/mm.py`. They contain hand-written Triton kernel code for matrix multiplication with various optimization strategies:

- `mm_template`: Standard GEMM with 2D tiling (BLOCK_M, BLOCK_N, BLOCK_K).
- `persistent_mm_template`: GEMM with persistent kernel (keeps accumulator across multiple tiles).
- `persistent_tma_mm_template`: GEMM with TMA (Tensor Memory Accelerator) descriptor loads.
- `blackwell_ws_persistent_device_tma_mm_template`: Blackwell-specific with warp-specialization.

Each template has a `grid_fn` that computes the grid size from the config, and `epilogue_fn` for built-in epilogues.

### 6.7 Template Heuristics (template_heuristics/triton.py)

The template heuristics file (3487 lines) generates GEMM configs with specific tile sizes:

- `BaseConfig`: `block_m, block_n, block_k, num_stages, num_warps`
- `GemmConfig`: Adds `group_m` for grouped prefetching.
- `BlackwellGPUGemmConfig`: Blackwell-specific features.

These configs are passed to the `TritonTemplate` and benchmarked via the same `CachingAutotuner` mechanism.


## 7. Persistent Reductions and Softmax

### 7.1 Persistent Reduction Decision

`TritonKernel.should_use_persistent_reduction()` (line 3371) returns True when:
1. We're inside a reduction (`self.inside_reduction`).
2. `V.choices.should_use_persistent_reduction()` returns True, which checks:
   - The reduction numel is statically known or can be bounded.
   - The numel is small enough that a single thread block can process it.
   - The device has enough shared memory for the accumulator.

When persistent reduction is used:
- `RBLOCK = next_power_of_2(rnumel)` (computed at compile time).
- No loop over the reduction dimension.
- The entire reduction happens in a single pass.
- The `tl.sum()`/`tl.max()` etc. operation reduces along the RBLOCK dimension directly.

### 7.2 Non-Persistent (Looped) Reduction

When persistent reduction is not possible (rnumel too large or dynamic), the kernel uses a loop:

```python
# accumulator init
_acc = tl.full([XBLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + tl.arange(0, R0_BLOCK)
    # load stripe
    tmp = tl.load(...)
    # accumulate
    _acc = _acc + tl.sum(tmp, 1)
# store final
tl.store(out_ptr0 + xindex, _acc, xmask)
```

The `tl.range()` loop can optionally use `num_stages` for software pipelining (prefetch next tile while computing current tile).

### 7.3 Cooperative Reduction

For very large reductions, cooperative reduction uses multiple thread blocks per row:

1. **RSPLIT**: Number of thread blocks assigned to each row. Each block processes a subset of the reduction dimension.
2. **Semaphore**: `tl.program_id(0)` identifies which block within the row. A semaphore synchronizes the blocks.
3. **Workspace**: Each block writes its partial result to a workspace buffer at `xindex * RSPLIT + rsplit_id`. After all blocks finish (barrier), each block reads all partial results and combines them.

The `init_cooperative_reduction()` (line 3277) sets up:
- Shifts grid dimensions (program_id(0) becomes rsplit_id).
- Creates semaphore args.
- Creates `CooperativeReductionWorkspaceCache` for workspace allocation.
- Generates rsplit indexing code.

### 7.4 Welford Reduction (Batch Norm)

For operations like batch norm that need mean and variance, Inductor uses **Welford's algorithm** which computes mean, M2 (sum of squared deviations), and weight in a single online pass.

`TritonKernel.welford_reduce()` generates calls to `triton_helpers.welford()` which implements Welford's combine formula:
```python
# combine two welford tuples (mean1, m2_1, weight1) and (mean2, m2_2, weight2)
new_weight = weight1 + weight2
new_mean = (weight1 * mean1 + weight2 * mean2) / new_weight
new_m2 = m2_1 + m2_2 + weight1 * (mean1 - new_mean) * (mean1 - new_mean) + weight2 * (mean2 - new_mean) * (mean2 - new_mean)
```

For persistent reductions, Welford is NOT used because it requires more registers. Instead, `welford_reduce_fallback()` takes two separate reductions (one for mean, one for variance) which is simpler but requires two passes.

For cooperative reductions, full Welford IS required because correctness demands that partial results from different thread blocks are combined correctly.

### 7.5 Online Softmax Reduction

Softmax requires computing `exp(x - max(x)) / sum(exp(x - max(x)))`. The `online_softmax_reduce` type implements this efficiently:

1. Each thread block computes a local max and a local sum of exp(x - local_max).
2. In the final reduction, `triton_helpers.online_softmax_reduce()` combines the partial max and partial sum across blocks:
   ```python
   # Combine two partial softmax results
   new_max = max(max1, max2)
   correction1 = exp(max1 - new_max)  # scale factor for sum1
   correction2 = exp(max2 - new_max)  # scale factor for sum2
   new_sum = sum1 * correction1 + sum2 * correction2
   ```
3. The final result is normalized by dividing by the global sum.

`TritonKernel.online_softmax_reduce_final_reduction()` (line 5429) generates:
```python
result_max, result_sum = triton_helpers.online_softmax_reduce(
    accumulator_max, accumulator_sum, dim, use_fast_math)
result_max = reduction_resize(result_max)
result_sum = reduction_resize(result_sum)
```

### 7.6 TritonSplitScanKernel (triton_split_scan.py, 228 lines)

For scan operations (cumsum, prefix sum), `TritonSplitScanKernel(TritonKernel)` implements the **decoupled look-back** algorithm from NVIDIA research (single-pass parallel prefix scan with decoupled look-back).

Key characteristics:
- Grid: `(ceildiv(r0_numel, R0_BLOCK), xnumel, 1)` - the first dimension indexes scan blocks, the second indexes rows.
- Each block computes a local scan (`tl.associative_scan`) and a block sum (`tl.reduce`).
- Uses a workspace buffer for inter-block communication.
- Calls `triton_helpers.exclusive_scan_decoupled_lookback()` for the decoupled look-back phase.
- The final result combines the exclusive prefix from look-back with the local scan.
- Persistent and cooperative reductions are disabled (line 53-57).

### 7.7 Mix-Order Reduction

For fusing two reductions with different loop orders (e.g., sum along rows + sum along columns), Inductor uses **mix-order reduction** which:

1. Splits the X dimension into `RSPLIT_SIZE` chunks.
2. Each chunk processes a subset of rows in parallel, with partial accumulation stored in workspace.
3. After all chunks are processed, partial results are combined.

This is useful for operations like variance computation that need both a sum and a sum-of-squares reduction along the same dimension but with different intermediate data.


## Summary: The Complete Pipeline

```
Python Model Code
    |
    v
torch.compile (Dynamo) -> FX Graph
    |
    v
AOTAutograd -> Joint fwd/bwd FX Graph
    |
    v
Inductor Lowering -> ir.TensorBox/Buffer/Operation
    |
    v
Scheduler -> SchedulerNode + FusedSchedulerNode
    |   (can_fuse decisions, fusion benchmarking)
    v
TritonScheduling.codegen_node()
    |   (select tiling, create TritonKernel)
    v
TritonKernel.codegen_kernel()
    |   (generate Triton source code)
    v
TritonScheduling.define_kernel()
    |   (write to wrapper code + submit to async_compile)
    v
async_compile.triton() -> Compile Worker Process
    |   (CachingAutotuner: precompile configs, benchmark, select best)
    v
Triton JIT Compiler -> GPU Binary
    |
    v
Runtime: CachingAutotuner.run() -> Launch with best config
```

Key files referenced:
- `/tmp/inductor-research/triton.py` (7810 lines) -- Main Triton codegen: TritonKernel, TritonScheduling, TritonOverrides, TritonPrinter
- `/tmp/inductor-research/simd.py` (4649 lines) -- SIMDKernel, SIMDScheduling base classes, can_fuse logic, codegen_node
- `/tmp/inductor-research/scheduler.py` (10204 lines) -- Scheduler, FusedSchedulerNode, WhyNoFuse, fusion benchmarking
- `/tmp/inductor-research/triton_heuristics.py` (5553 lines) -- CachingAutotuner, triton_config, Grid classes, heuristic decorators
- `/tmp/inductor-research/select_algorithm.py` (6193 lines) -- TritonTemplateKernel, epilogue fusion, store_output
- `/tmp/inductor-research/triton_combo_kernel.py` (1302 lines) -- ComboKernel for multi-sub-kernel Triton programs
- `/tmp/inductor-research/triton_split_scan.py` (228 lines) -- TritonSplitScanKernel for scan/prefix-sum operations
- `/tmp/inductor-research/common.py` (3069 lines) -- Kernel, CSE, KernelArgs base classes
- `/tmp/inductor-research/multi_kernel.py` (650 lines) -- MultiKernel for runtime variant selection
- `/tmp/inductor-research/kernel_mm.py` (1399 lines) -- Triton GEMM templates
- `/tmp/inductor-research/template_heuristics_triton.py` (3487 lines) -- GEMM config generation
- `/tmp/inductor-research/cpp_gemm_template.py` (1939 lines) -- CPU GEMM template (for comparison)
- `/tmp/inductor-research/autotune_process.py` (1593 lines) -- Compile worker process pool
