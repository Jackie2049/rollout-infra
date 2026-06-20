# PyTorch CUDA Graph Refactoring Cluster (#187740-#187750) — Deep Reading

> 2026-06-20 | Cluster of 6 PRs | Authors: ngimel (4 PRs), ahkush (1 PR), kwen2501 (1 PR)
> All OPEN, not yet merged | Stack-based (ghstack) dependency chain: #187749 -> #187746 -> #187741 -> #187740
> Priority: HIGH for vLLM/DeepSpeed/SGLang CUDA graph paths
> Relevance: vLLM V1 cudagraph capture, DeepSpeed overlap_comm (#8061/#8080), SGLang piecewise CUDA graph, verl colocated mode

---

## Cluster Overview

This cluster represents a coordinated effort by ngimel (Natalia Gimelshein, PyTorch CUDA team) to restructure PyTorch's `CUDAGraph` internals from a monolithic C++-centric design to a split Python/C++ model with explicit lifecycle phases and user-facing hooks. The changes are organized as a ghstack chain (bottom-to-top dependency order):

```
#187749 (Unify debug flag; move debug_dump to Python; add capture hooks)
  └── depends on #187746 (Split capture_end; stamp annotations via raw_cuda_graph)
      └── depends on #187741 (Fix annotation remap for keep_graph=True)
          └── depends on #187740 (Expose exec state; move lazy instantiate to Python)
```

Two additional PRs in the broader cluster are independent:
- #187747 (Replace CUDA APIs with torch.accelerator equivalents) — by ahkush
- #187750 (Tune all_gather_offset for skewed buckets) — by kwen2501, symm_mem

---

## PR #187740 — Expose CUDAGraph exec state; move replay's lazy instantiate to Python

### Basic Info
| Field | Value |
|-------|-------|
| Title | [cuda] Expose CUDAGraph exec state; move replay's lazy instantiate to Python |
| Author | ngimel |
| Status | OPEN |
| Additions/Deletions | +20/-5 |
| Changed files | 4 |

### Technical Analysis

**What it changes**: The C++ `CUDAGraph::replay()` method previously had lazy on-demand instantiation logic — when `keep_graph=true` and the `cudaGraphExec_t` had not been instantiated yet, `replay()` would call `instantiate()` internally. This PR removes that lazy instantiation from C++ and moves it to the Python `torch.cuda.CUDAGraph.replay()` wrapper.

**Key code changes**:
1. **C++ side** (`CUDAGraph.cpp`): The old `replay()` checked `if (!has_graph_exec_)` and called `instantiate()`. Now it does `TORCH_CHECK(has_graph_exec_)` — a strict contract requiring the exec graph to already exist.
2. **C++ side** (`CUDAGraph.h`): New `bool has_graph_exec() const` accessor exposed to Python bindings.
3. **Python side** (`Graph.cpp` pybind): New `def_property_readonly("_has_graph_exec", &CUDAGraph::has_graph_exec)` binding.
4. **Python side** (`graphs.py`): `replay()` now checks `if not self._has_graph_exec: self.instantiate()` before calling `super().replay()`.

**Why it matters**: This is the foundational change that makes the rest of the stack possible. By routing instantiation through Python, the annotation remap (#187741) can ride entirely on `instantiate()` rather than needing to infer whether `replay()` instantiated. It also establishes the principle that C++ `CUDAGraph` is a strict engine (no lazy behavior) while Python provides the convenience wrappers.

**Impact on downstream frameworks**: vLLM's V1 CUDA graph path calls `graph.replay()` through `CUDAGraphWrapper`. The Python-side lazy instantiation now runs in vLLM's code path, meaning vLLM gets the convenience behavior automatically. No change needed in vLLM code.

---

## PR #187741 — Fix CUDA graph kernel-annotation remap for keep_graph=True

### Basic Info
| Field | Value |
|-------|-------|
| Title | [cuda] Fix CUDA graph kernel-annotation remap for keep_graph=True |
| Author | ngimel |
| Status | OPEN |
| Additions/Deletions | +157/-20 |
| Changed files | 3 |

### Technical Analysis

**What it changes**: The kernel annotation remap system (`remap_to_exec_graph()`) was broken for `keep_graph=True` graphs because it tried to read the exec graph id via `raw_cuda_graph_exec()` during `__exit__`, but with `keep_graph=True` the exec graph does not exist yet at that point (it's only created on the first `instantiate()` or `replay()`). This PR fixes the remap by deferring it until instantiation time.

**Key code changes**:
1. **`_graph_annotations.py`**: `stamp_capture_graph_id()` moved from `mark_kernels()` (which ran during active stream capture) to a standalone function. It now reads the graph id via `cudaGraphGetId()` on `raw_cuda_graph()` rather than deriving it from a kernel node's `toolsId` high bits.
2. **`graphs.py`**: New `_remapped_exec_id: int | None` attribute tracks which key annotations are currently keyed to. New `_maybe_remap_annotations()` method called from `instantiate()`. `__exit__` only remaps for `keep_graph=False`.
3. **`_graph_annotations.py`**: `remap_to_exec_graph()` now rekeys from the *current* key (capture id or previous exec id) to the new exec id, with `_remapped_exec_id` tracking. Self-skips when exec id is unchanged (e.g., replay after instantiate).
4. **Tests**: New tests for `keep_graph=True` deferral (via both `instantiate()` and bare `replay()`), and for re-instantiation rekeying (each `instantiate()` produces a fresh exec id).

**Critical insight**: The re-instantiation rekeying logic is significant. With `keep_graph=True`, a user can modify the template and instantiate again, producing a fresh exec graph id. Annotations must follow that new id, not stay keyed to the old one. The `_remapped_exec_id` field tracks the current key across multiple instantiations.

**Impact on downstream frameworks**: vLLM uses `keep_graph=False` (default) for its CUDA graph captures, so this fix doesn't directly affect vLLM's annotation path. However, SGLang's piecewise CUDA graph approach might benefit from `keep_graph=True` for template reuse — the correct annotation remap would be important for profiling/debugging SGLang's piecewise captures.

---

## PR #187746 — Split capture_end; stamp graph annotations via raw_cuda_graph

### Basic Info
| Field | Value |
|-------|-------|
| Title | [cuda] Split capture_end; stamp graph annotations via raw_cuda_graph |
| Author | ngimel |
| Status | OPEN |
| Additions/Deletions | +75/-37 |
| Changed files | 7 |

### Technical Analysis

**What it changes**: `CUDAGraph::capture_end()` is split into two phases: `capture_end_pre()` (ends capture, leaving the `cudaGraph_t` template live) and `capture_end_post()` (instantiate + destroy for `keep_graph=false`). This creates a "post-capture / pre-finalize" window where the template graph is reachable via `raw_cuda_graph()` regardless of `keep_graph` mode.

**Key code changes**:
1. **C++ side** (`CUDAGraph.cpp`): `capture_end()` split into `capture_end_pre()` + `capture_end_post()` + composite `capture_end()`. The split gives a window where `graph_` is live for both `keep_graph` modes.
2. **C++ side** (`CUDAGraph.h`): New declarations for `capture_end_pre()` and `capture_end_post()`.
3. **C++ side** (`CUDAGraph.cpp`): `raw_cuda_graph()` relaxed to require only `has_graph_` — no longer requires `keep_graph=true`. This means the template is accessible in the `capture_end_pre` window even for `keep_graph=false`.
4. **Python side** (`Graph.cpp`): New pybind bindings for `capture_end_pre` and `capture_end_post`.
5. **Python side** (`graphs.py`): `capture_end()` now calls `capture_end_pre()`, then stamps the capture graph id via `maybe_stamp_capture_graph_id()` (reading it from `raw_cuda_graph()` in the window), then `capture_end_post()`.
6. **Python side** (`_graph_annotations.py`): `stamp_capture_graph_id` renamed to `maybe_stamp_capture_graph_id`. Now reads the capture id via `raw_cuda_graph()` + `cudaGraphGetId()` rather than fishing the in-flight graph from `cudaStreamGetCaptureInfo`.

**Why this design**: `cudaGraphGetId()` stays in Python/cuda.bindings, resolved at runtime. A C++ call would require building PyTorch against CUDA >= 13.1. By stamping in the `capture_end_pre` window (where the template is guaranteed live), the stamp works even for `keep_graph=False` (which destroys the template at `capture_end_post`).

**Impact on downstream frameworks**: This split creates a new architectural feature — a "capture-end window" where the template graph is live and inspectable. vLLM could use `register_capture_end_hook` (#187749) to dump or inspect captured graphs before finalization. SGLang's piecewise CUDA graph could use this window to validate segment counts before the graph is finalized, potentially catching issues like #28499 earlier.

---

## PR #187747 — Replace CUDA APIs with generic torch.accelerator equivalents

### Basic Info
| Field | Value |
|-------|-------|
| Title | Replace CUDA APIs with generic torch.accelerator equivalents |
| Author | ahkush |
| Status | OPEN |
| Additions/Deletions | +48/-22 |
| Changed files | 4 |
| Labels | module: inductor, open source |

### Technical Analysis

**What it changes**: Replaces CUDA-specific API calls in `torch._inductor` and `torch._functorch` with generic `torch.accelerator` equivalents so diagnostic and instrumentation code works across all backends (CUDA, XPU, MPS, third-party).

**Key replacements**:
- `torch.cuda.empty_cache()` -> `torch.accelerator.empty_cache()`
- `torch.cuda.memory.*` -> `torch.accelerator.memory.*`
- `ProfilerActivity.CUDA` / `devices=["cuda"]` -> runtime dispatch via `torch.accelerator.current_accelerator()` guarded by `torch.profiler.supported_activities()`
- `--cuda-memory-snapshot` flag -> `--memory-snapshot`

**Impact on downstream frameworks**: This is not directly about CUDA graphs but about the broader trend of making PyTorch's device abstraction layer generic. vLLM is already exploring Ascend support; this change supports that direction. For RTX 4090, no impact — `torch.accelerator` delegates to `torch.cuda` on NVIDIA GPUs.

---

## PR #187749 — Unify CUDAGraph debug flag; move debug_dump to Python; add capture hooks

### Basic Info
| Field | Value |
|-------|-------|
| Title | [cuda] Unify CUDAGraph debug flag; move debug_dump to Python; add capture hooks |
| Author | ngimel |
| Status | OPEN |
| Additions/Deletions | +224/-81 |
| Changed files | 9 |

### Technical Analysis

**What it changes**: Three major changes:

1. **Debug flag unification**: `enable_debug_mode()` now just sets `keep_graph_` (retain the template for inspection) instead of a separate process-global `_cuda_graphs_debug` flag. The old flag never actually let `debug_dump` dump a `keep_graph=false` graph. `enable_debug_mode` is now per-instance (was process-global). Debug mode defers instantiation like `keep_graph=true`.

2. **debug_dump moved to Python**: The C++ `CUDAGraph::debug_dump()` is removed entirely. Dumping now lives in Python (`CUDAGraph.debug_dump()` via cuda.bindings `cudaGraphDebugDotPrint` on `raw_cuda_graph()`). This keeps the CUDA-version-gated API runtime-resolved rather than tied to the build toolkit. New `export_dot(path)` factory returns a ready-made capture-end hook.

3. **Capture lifecycle hooks**: New `register_capture_end_hook(fn)` and `register_post_instantiate_hook(fn)` APIs. Hooks are `RemovableHandle`-based (multiple, in registration order). `register_capture_end_hook` runs in the `capture_end` window where the template is live (both `keep_graph` modes). `register_post_instantiate_hook` runs after each instantiation. No replay hook — replay is the hot path.

**Key code changes**:
- C++: `_cuda_graphs_debug` global removed; `debug_dump()` method removed; `enable_debug_mode()` sets `keep_graph_` = true; `capture_end_post()` is destroy-only (no instantiate)
- C++: `capture_end()` now calls `instantiate()` before `capture_end_post()` for `keep_graph=false` (C++ callers still get the old composite behavior)
- Python: `debug_dump()` reimplemented via `_dump_graph_dot()` helper using cuda.bindings
- Python: `export_dot(path)` factory function returns a capture-end hook
- Python: `_capture_end_hooks` and `_post_instantiate_hooks` OrderedDict fields
- Python: `capture_end()` stamps annotations, fires capture-end hooks, then instantiates (for `keep_graph=False`) via Python `instantiate()` (which fires post-instantiate hooks), then `capture_end_post()`

**Behavior changes**:
- `enable_debug_mode` is now per-instance, not process-global. Defers instantiation like `keep_graph=True`.
- `debug_dump` now requires `cuda-bindings` package.
- `debug_dump` on `keep_graph=False` graph raises unless called from a capture-end hook.

**Impact on downstream frameworks**: The capture lifecycle hooks are the most significant new feature. vLLM's V1 CUDA graph capture could use `register_capture_end_hook` to:
- Validate captured graph structure before finalization
- Dump graph topology for debugging
- Run custom validation on segment counts or node counts

SGLang could use `register_capture_end_hook` to validate piecewise segment metadata before the graph is finalized, catching issues like the csgmv segment replay bug (#28499/#28371) earlier in the pipeline.

---

## PR #187750 — Tune all_gather_offset for skewed buckets (symm_mem)

### Basic Info
| Field | Value |
|-------|-------|
| Title | Tune all_gather_offset for skewed buckets |
| Author | kwen2501 |
| Status | OPEN |
| Additions/Deletions | +113/-34 |
| Changed files | 2 |

### Technical Analysis

**What it changes**: The `nccl_all_gather_offset` kernel (symm_mem all-gather) was using one CTA per `(parameter, peer)` shard, which left most of the GPU idle for buckets with few large shards. This PR flattens the work into fixed-size byte tiles (at least `AG_MIN_BYTES_PER_TILE = 2MB`, the largest shard capped at `AG_MAX_CTAS = 32` tiles) and grid-strides CTAs over the flat tile space, load-balancing shards of any size including skewed buckets.

**Performance results (8x H100, bf16, GB/s)**:

| Shape | multimem (before -> after) | LSA push (before -> after) |
|-------|---------------------------|---------------------------|
| 1 split | 171 -> 350 (2.0x) | 76 -> 289 (3.8x) |
| 8 splits | 330 -> 355 (1.1x) | 270 -> 289 (1.1x) |
| **1 huge + 8 tiny (skew)** | **180 -> 369 (2.0x)** | **77 -> 301 (3.9x)** |
| 256 x 4KB | 58 -> 178 (3.0x) | 93 -> 75 (latency-bound) |

**Impact on downstream frameworks**: This is directly relevant to DeepSpeed's `overlap_comm` path. DeepSpeed's ZeRO stage 1/2 gradient bucket all-gather uses similar patterns where skewed bucket sizes (some parameters large, some tiny) lead to GPU underutilization. The tile-based approach could inform DeepSpeed's own bucket scheduling. For vLLM's data-parallel path, this is relevant when using symm_mem for weight sync.

---

## Synthesis: Overall Direction of PyTorch CUDA Graph Refactoring

### Architectural Shift: From Monolithic C++ to Split Python/C++ Lifecycle

The cluster represents a fundamental architectural shift in how PyTorch's CUDAGraph works:

**Before (old model)**:
```
C++ CUDAGraph::capture_end()
  ├── End capture
  ├── If keep_graph=false: instantiate + destroy template
  ├── If keep_graph=true: keep template alive
  ├── C++ replay() with lazy on-demand instantiation
  ├── C++ debug_dump() gated by process-global flag
  └── No user-facing lifecycle hooks
```

**After (new model)**:
```
Python capture_end()
  ├── capture_end_pre() (C++: end capture, template live)
  ├── stamp_capture_graph_id() (Python: cudaGraphGetId via raw_cuda_graph)
  ├── capture-end hooks (Python: user callbacks, template live for both modes)
  ├── instantiate() (Python: for keep_graph=False)
  │   ├── C++ instantiate
  │   ├── maybe_remap_annotations() (Python: rekey to exec id)
  │   └── post-instantiate hooks (Python: user callbacks)
  ├── capture_end_post() (C++: destroy template for keep_graph=False)

Python replay()
  ├── if not _has_graph_exec: instantiate() (routes through above)
  ├── C++ replay (strict: requires exec graph exists)
```

**Key principles established**:
1. C++ is the strict engine — no lazy behavior, no soft checks
2. Python is the convenience wrapper — lazy instantiation, hooks, annotations
3. The `capture_end_pre/post` split creates an explicit lifecycle window where the template is inspectable
4. cuda.bindings (runtime-resolved) replaces C++ CUDA API calls where version gating matters
5. User-facing hooks replace internal-only lifecycle points

### How This Affects vLLM's CUDA Graph Path (V1 cudagraph capture)

**vLLM V1 architecture** (`vllm/v1/worker/gpu/cudagraph_utils.py`):
- Uses `torch.cuda.CUDAGraph()` with `keep_graph=False` (default)
- Captures full model execution graphs for multiple batch sizes
- Replay is the hot path — happens on every decode step

**Direct impact**:
1. **Lazy instantiation moved to Python**: vLLM's `CUDAGraphWrapper` calls `graph.replay()` which now goes through Python's lazy instantiate check. This adds a per-replay `_has_graph_exec` property check, but since vLLM always uses `keep_graph=False` (where `capture_end` already instantiates), `_has_graph_exec` is always True at replay time — the check is a no-op.
2. **Capture lifecycle hooks**: vLLM could register `capture_end_hook` to validate graph structure or log capture metadata. This could help diagnose vLLM CUDA graph capture issues more easily.
3. **debug_dump in Python**: vLLM could now dump captured graphs to DOT format without needing a C++ debug flag. The `export_dot(path)` factory works as a capture-end hook, usable even for `keep_graph=False`.
4. **No breaking changes**: vLLM's existing CUDA graph code uses the default `keep_graph=False` path, which is unchanged in behavior. The C++ composite `capture_end()` still does pre + instantiate + post for C++ callers.

**Indirect impact via cumem (#45552)**: vLLM's CuMem sleep/wake interacts with CUDA graph state. The new `capture_end_pre/post` split and `_has_graph_exec` accessor could help vLLM implement safer sleep/wake transitions — checking whether a graph is instantiated before attempting memory operations on captured tensors. This connects to the stream sync bug (#45552) where missing `torch.cuda.synchronize()` around CuMem operations causes crashes on RTX 4090.

### How This Affects DeepSpeed's overlap_comm (#8061/#8080)

**DeepSpeed's overlap_comm architecture** uses CUDA streams for gradient reduction overlap, which is where the stream race (#8061) originates. The PyTorch CUDAGraph refactoring does not directly fix DeepSpeed's multi-stream wait bug, but it creates new APIs that DeepSpeed could leverage:

1. **`register_post_instantiate_hook`**: DeepSpeed's overlap_comm path could use this hook to validate that all gradient bucket producer streams have been recorded before the graph is instantiated. This would catch stream race conditions at graph instantiation time rather than at runtime.

2. **`_has_graph_exec` accessor**: DeepSpeed could check this before replaying a CUDA graph in the overlap_comm path, avoiding replay on uninstantiated graphs.

3. **Connection to #187750**: The `all_gather_offset` tiled kernel is directly relevant to DeepSpeed's ZeRO bucket all-gather. DeepSpeed's bucket sizes are often skewed (large parameters + tiny parameters in the same bucket). The 2.0x-3.9x bandwidth improvements from tiling suggest DeepSpeed should adopt similar tile-based launch geometry.

4. **Root cause connection**: #8061/#8080 is a multi-producer stream race, not a CUDA graph lifecycle bug. The PyTorch refactoring doesn't fix it, but the new lifecycle hooks provide a better debugging surface — DeepSpeed could register hooks to check stream synchronization state at capture boundaries.

### How This Affects SGLang's Piecewise CUDA Graph

**SGLang's piecewise CUDA graph** (`sglang/srt/layers/radix_attention.py` and LoRA backends) is the most directly affected framework because:

1. **Piecewise capture requires inspecting the graph**: SGLang's piecewise mode captures sub-graphs for compatible operations while leaving incompatible ones eager. The `capture_end_pre/post` window and `raw_cuda_graph()` relaxation (no longer requires `keep_graph=true`) give SGLang a way to inspect captured sub-graph topology before finalization.

2. **`register_capture_end_hook` for segment validation**: The csgmv bug (#28499/#28371) was caused by CUDA graph replay with stale segment counts. SGLang could register a `capture_end_hook` that validates segment metadata matches the template graph's node structure, catching mismatches before the graph is finalized.

3. **`keep_graph=True` for template reuse**: SGLang's piecewise approach might benefit from `keep_graph=True` to reuse graph templates across different batch configurations. The fixed annotation remap (#187741) and re-instantiation rekeying ensure annotations stay correct across re-instantiations.

4. **`export_dot` for piecewise debugging**: SGLang could use `export_dot(path)` as a capture-end hook to dump piecewise sub-graphs for debugging, even when using `keep_graph=False`.

### RTX 4090 Implications

**Positive implications**:
1. **No new issues introduced**: The refactoring preserves the existing `keep_graph=False` path unchanged. vLLM, DeepSpeed, and SGLang all use `keep_graph=False` by default, so the core behavior is identical.
2. **Better debugging surface**: The Python-side `debug_dump` and `export_dot` hook work with any CUDA version that supports cuda.bindings. On RTX 4090 (CUDA 12.x), cuda.bindings is available, so these tools work.
3. **Lazy instantiation in Python is harmless**: For `keep_graph=False` (vLLM's path), the `_has_graph_exec` check at replay time is always True — no extra CUDA calls, no performance impact.

**Potential concerns**:
1. **`enable_debug_mode` per-instance**: Previously a process-global flag, now per-instance. This means enabling debug mode on one graph doesn't affect other graphs. If a framework was relying on the global flag behavior (unlikely), this could be a surprise.
2. **`debug_dump` requires cuda-bindings**: On RTX 4090, cuda-bindings is typically available via pip. But if a deployment doesn't have it, `debug_dump` and `export_dot` will raise RuntimeError. The C++ `debug_dump` didn't require cuda-bindings.
3. **New `_remapped_exec_id` attribute**: Each `CUDAGraph` instance now tracks `_remapped_exec_id`. This is a small per-instance memory increase (one int), negligible on RTX 4090.
4. **`capture_end_pre/post` split in C++**: The C++ `capture_end()` composite still works as before. No risk for existing C++ callers.

**RTX 4090 CUDA graph path specifically**: vLLM's CuMem sleep/wake (#45552) crashes on RTX 4090 because of missing stream synchronization, not because of CUDA graph internals. The PyTorch refactoring doesn't fix #45552 directly, but the new `_has_graph_exec` accessor and lifecycle hooks could help vLLM implement safer state transitions. Specifically:
- vLLM could check `_has_graph_exec` before CuMem sleep — if a graph is still instantiated and replayable, sleeping on its captured memory is unsafe
- vLLM could register `capture_end_hook` to record which tensors are captured, then validate that CuMem sleep only operates on non-captured tensors

### Connection to Related Issues

#### #45552 (cumem stream sync) — CRITICAL RTX 4090 blocker

The PyTorch CUDAGraph refactoring creates new APIs that could help vLLM fix #45552:
- `_has_graph_exec` could be checked before CuMem sleep operations
- `register_capture_end_hook` could log captured tensor addresses for CuMem to skip
- But the root cause (missing `torch.cuda.synchronize()`) is orthogonal to CUDA graph lifecycle

#### #45309/#45972 (DSV4 cudagraph revert)

vLLM PR #45309 attempted to expand CUDA graph range for DeepSeek-V4 by reducing `eager_break_during_capture` decorators, achieving 26.8-27.9% E2E TTFT improvement. PR #45972 reverted it because it caused **garbage output** (correctness regression).

**Connection**: The PyTorch CUDAGraph refactoring's `capture_end_pre/post` window and hooks could provide a safer way to expand CUDA graph range. Instead of removing `eager_break_during_capture` guards wholesale (which caused the garbage output), vLLM could:
- Use `register_capture_end_hook` to validate that no unexpected nodes were captured
- Use `debug_dump` / `export_dot` to inspect what operations ended up in the graph
- Use `keep_graph=True` + re-instantiation to iteratively debug which operations break capture

This suggests that when #45309 is re-attempted (after fixing the garbage output), the new PyTorch APIs will provide better debugging tools.

#### #45309/#45972 vs. the refactoring cluster

The DSV4 cudagraph revert was about **which operations can safely be captured**. The PyTorch refactoring is about **how capture lifecycle works internally**. They're orthogonal but complementary: better lifecycle APIs make it easier to diagnose which operations break capture.

---

## Dependency Chain and Merge Order

Since these are a ghstack chain, they must merge in bottom-to-top order:

```
Merge order:
1. #187740 (Expose exec state; move lazy instantiate to Python)
2. #187741 (Fix annotation remap for keep_graph=True)
3. #187746 (Split capture_end; stamp annotations via raw_cuda_graph)
4. #187749 (Unify debug flag; move debug_dump to Python; add capture hooks)

Independent:
5. #187747 (Replace CUDA APIs with torch.accelerator equivalents)
6. #187750 (Tune all_gather_offset for skewed buckets)
```

---

## Key Takeaways for Framework Developers

1. **The C++ contract is now strict**: `CUDAGraph::replay()` requires `has_graph_exec_`. Any C++ code calling replay must instantiate first. This affects frameworks that use C++ CUDA graph paths directly (unlikely for vLLM/DeepSpeed/SGLang, which use Python).

2. **New hook API is the primary benefit**: `register_capture_end_hook` and `register_post_instantiate_hook` provide clean extension points. Frameworks should adopt these for:
   - vLLM: CuMem safety checks, graph structure validation
   - DeepSpeed: stream synchronization checks at capture boundaries
   - SGLang: segment metadata validation before finalization

3. **`export_dot` works for `keep_graph=False`**: Previously, dumping a `keep_graph=False` graph required the debug mode flag. Now, `export_dot(path)` registered as a capture-end hook works for any graph, even `keep_graph=False`. This is useful for production debugging.

4. **Annotation remap is correct for re-instantiation**: The `_remapped_exec_id` tracking ensures annotations follow each new exec graph id. This matters for frameworks that might use `keep_graph=True` for template reuse.

5. **`all_gather_offset` tiling (#187750) is a performance lesson**: The 2.0x-3.9x bandwidth improvement for skewed buckets should inform DeepSpeed's bucket scheduling and vLLM's data-parallel all-gather implementation.

6. **No immediate breaking changes**: For `keep_graph=False` (the default used by all three frameworks), behavior is identical. The refactoring is additive — new APIs, stricter internal contracts, but same external behavior for default usage.

---

## Related Reading Notes

- `vllm-45552-cumem-stream-sync-deep-reading.md` — CuMem stream sync bug on RTX 4090
- `deepspeed-8080-cuda-stream-race-fix-reading.md` — DeepSpeed overlap_comm stream race fix
- `vllm-cuda-graph-reading.md` — vLLM V1 CUDA graph source reading
- `sglang-28499-csgmv-cuda-graph-fix-reading.md` — SGLang csgmv segment replay fix
- `cuda-stream-memory-management-reading.md` — CUDA stream safety pattern family
