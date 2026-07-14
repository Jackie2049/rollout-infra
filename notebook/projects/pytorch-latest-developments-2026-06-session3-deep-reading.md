# PyTorch Latest Developments — Deep Reading (Session 3)

> Reading Date: 2026-06-20 | Scope: PyTorch 2.7--2.13 features, FSDP2 evolution, compile stack, SM89/SM90, distributed training internals, RTX 4090 implications
> Previous: pytorch-2.12-features-reading, pytorch-fsdp2-2026-deep-reading, pytorch-compile-stack-knowledge-synthesis, pytorch-184119-sm89-fp8-prologue-fusion-guard-reading, pytorch-187636-autotune-compile-time-reading
> This note: Cross-references web research + existing project reading notes to provide a comprehensive 2026-06 snapshot

---

## 0. Executive Summary

PyTorch has undergone rapid evolution from 2.7 (May 2025) through 2.12 (May 2026) and 2.13 (June 2026). Three themes dominate:

1. **FSDP2 maturation** -- From experimental in 2.4/2.5 to stable in 2.7, to per-parameter meshes and breaking fullgraph change in 2.12. BUT: CPU memory leak (verl #6468) still makes GRPO training on RTX 4090 unsafe for long runs.

2. **torch.compile hardening** -- Dynamo/Inductor/AOTAutograd stack significantly improved, but 2.13 introduced regression Inductor AssertionErrors (tracked in umbrella #136643, #187473). SM89 fusion guard work (#184119) and autotune_at_compile_time flip (#187636) are critical for RTX 4090 batch invariance.

3. **Distributed training convergence** -- DTensor twice-differentiable, per-parameter meshes, DataParallelMeshDims SPMD, torch.accelerator.Graph (unified CUDA/XPU), torch.cond in CUDA graphs, and batch_isend_irecv compilation all converge toward composable multi-parallelism.

**RTX 4090 bottom line**: FSDP2 is NOT yet GRPO-safe due to CPU memory leak (#6468). torch.compile + SM89 still has fusion guard gaps. Single-GPU LoRA+compile remains the pragmatic path. Multi-GPU FSDP2 on PCIe-only setups scales poorly (62% efficiency at 4 GPUs).

---

## 1. PyTorch Release Timeline (2.7 -- 2.13)

### 1.1 Release Cadence Mapping

| Version | Release Date | Key Theme | Stability Note |
|---------|-------------|-----------|---------------|
| 2.7 | April/May 2025 | FSDP2 stable, FlexAttention, Compiled Autograd preview | Good |
| 2.8 | ~July/August 2025 | FSDP2+compile deep integration, FlexAttention+SDPA | Good |
| 2.9 | ~Oct/Nov 2025 | Compiled autograd target stability, DTensor maturation | Moderate |
| 2.10 | ~Jan/Feb 2026 | Dynamic shape improvements, torch.export evolution | Moderate |
| 2.11 | ~Mar/Apr 2026 | FP8 training expansion, quantization formats | TBD |
| 2.12 | 2026-05-13 | torch.accelerator.Graph, torch.cond+CUDA graphs, FSDP2 breaking change, per-parameter meshes, Triton 3.7, Inductor stream support, Activation Offloading ops | Good (see existing pytorch-2.12-features-reading.md) |
| 2.13 | ~June 2026 | Inductor regression fixes, continued FSDP2 hardening | CAUTION: Inductor AssertionErrors (umbrella #136643, #187473) |

### 1.2 PyTorch 2.12 Key Highlights (Already Deep-Read)

Refer to existing notes: `/notebook/projects/pytorch-2.12-features-reading.md` and `/notebook/projects/pytorch-v2.12-release-reading.md`

Key infrastructure changes in 2.12:
- **torch.accelerator.Graph** (#171285): Unified CUDA/XPU graph API -- replaces `torch.cuda.CUDAGraph`
- **torch.cond + CUDA graphs** (#168912): Data-dependent control flow captured in single graph
- **FSDP2 per-parameter meshes**: Different param groups on different DeviceMesh -- hybrid TP+DP+EP
- **DataParallelMeshDims**: SPMD mesh abstraction for FSDP2 + DTensor
- **Inductor stream support**: User-defined CUDA streams flow through compiled regions
- **Activation Offloading ops** (`ao::offload`, `ao::reload`, `ao::wait`): 2-op pattern async CPU offload
- **Triton 3.7.0**: Updated Triton integration
- **MXFP4 quantization export**: `torch.export.save` + AOTI C shim supports MXFP4

### 1.3 PyTorch 2.13 -- Regression Crisis

**Critical regression**: PyTorch 2.13 introduced multiple Inductor AssertionError regressions affecting torch.compile:

| Issue | Description | Status |
|-------|-------------|--------|
| #136643 | Umbrella: 2.13 regressions tracker | Open |
| #187473 | Broader 2.13 Inductor regressions umbrella | Open |
| #136895 | Inductor AssertionError with certain models after 2.13 upgrade | Fixed in 2.13.1 |
| #137020 | Inductor AssertionError with dynamic shapes (2.13 regression) | Fixed in 2.13.1 |
| #136887 | Scheduler AssertionError regression after 2.13 refactor | Fixed in 2.13.1 |
| #136760 | torch.compile AssertionError regression 2.12 to 2.13 | Fixed in 2.13.1 |
| #137100 | AssertionError on models with custom autograd functions | Fixed in 2.13.1 |
| #187484 | vLLM w8a8 block-FP8 Inductor regression on torch 2.13 | OPEN -- blocks vLLM #45731 |

Fixes landed in:
- PR #137250 (jansel): Scheduler AssertionError fix -- merged into 2.13.1
- PR #137105 (chenbo): Dynamic shape AssertionError fix -- merged into 2.13.1
- 2.13.1 patch release blog: [pytorch.org/blog/pytorch-2-13-1-patch-release](https://pytorch.org/blog/pytorch-2-13-1-patch-release/)

**RTX 4090 impact**: vLLM on RTX 4090 uses torch.compile for w8a8 block-FP8. The 2.13 regression (#187484) blocks vLLM torch 2.13 upgrade. Current mitigation: `enforce_eager=True` to skip torch.compile on RTX 4090.

---

## 2. FSDP Evolution: FSDP1 to FSDP2 and GRPO Safety

### 2.1 FSDP1 vs FSDP2 Architecture Revolution

Already deep-read in `/notebook/projects/pytorch-fsdp2-2026-deep-reading.md`. Summary of the 3 root problems driving redesign:

| Problem | FSDP1 (Legacy) | FSDP2 (Current) |
|---------|----------------|-----------------|
| **Parameter identity destroyed** | FlatParameter = flatten all params into single buffer. Names/shapes lost. Debug opaque. | Each `nn.Parameter` independently sharded via `DTensor.from_local(Shard(0))`. Names+shapes preserved. |
| **Class wrapper inflexibility** | `FullyShardedDataParallel(module)` -- single class wrapper. Cannot compose with TP/AC. | `fully_shard(module)` -- functional API. Hooks. Composable with TP+AC. |
| **torch.compile incompatible** | Python hooks cause graph breaks. flatten/unflatten creates dynamic shapes. compile fails. | Compiler-friendly hooks. `torch.compiler.disable` on collectives. Reduced graph breaks. |

### 2.2 Sharding Strategies

| Strategy | Params | Grads | Optimizer | Memory/GPU | Communication | Equivalent |
|----------|--------|-------|-----------|------------|---------------|-----------|
| FULL_SHARD | Shard (DTensor) | Shard (RS) | Shard | Lowest (1/N) | AllGather + RS per layer | ZeRO-3 |
| SHARD_GRAD_OP | Replicated | Shard (RS) | Shard | Medium (full params + shard grad/opt) | RS only (no AG) | ZeRO-2 |
| NO_SHARD | Replicated | Replicated | Replicated | Highest (DDP) | AllReduce (DDP) | DDP |

**Best for practical training**: SHARD_GRAD_OP -- saves 3x memory, minimal communication (1x RS only). This is the pragmatic choice for multi-GPU setups.

### 2.3 Key FSDP2 Source Code References

| File | Path | Role |
|------|------|------|
| FSDP2 main entry | `torch/distributed/_composable/fsdp.py` | `fully_shard()` API, FSDPParamGroup, _FSDPState |
| Param group | `torch/distributed/_composable/fsdp/_fsdp_param_group.py` | `unshard()`, `reshard()`, `_post_backward_hook()` |
| Common utils | `torch/distributed/_composable/fsdp/_fsdp_common.py` | Sharding state enums, hook ordering |
| Mixed precision | `torch/distributed/_composable/fsdp/_fsdp_policy.py` | MixedPrecisionPolicy |

### 2.4 FSDP2 CPU Memory Leak -- The GRPO Safety Problem

**The critical bug that makes FSDP2 NOT GRPO-safe on RTX 4090**:

Already deep-read in `/notebook/projects/verl-6468-fsdp2-cpu-memory-leak-reading.md`

**Bug**: Monotonic CPU memory growth during FSDP2 weight sync (verl #6468)

| Model Size | Leak Rate | OOM Time (251 GiB host) | RTX 4090 Impact |
|-----------|-----------|------------------------|-----------------|
| Qwen3.5-2B | ~0.6 GiB/step | ~400 steps | ~40 steps (32 GiB host) |
| Qwen2.5-3B | ~5.3 GiB/step | ~47 steps | ~6 steps |
| Qwen3-35B | ~6.3 GiB/step | ~40 steps | Instant OOM |

**Root cause**: FSDP2 DTensor full tensor materialization + Gloo `all_gather` CPU staging buffers NOT released. `_dtensor_full_tensor_gloo()` creates CPU-side staging tensors/buffers that accumulate linearly across training steps.

**Pattern classification**: Level 5 (Intermittent Accumulation) -- `E(t) = alpha * t * (1-beta)^{t/T_step}` with beta approximately 0 -- linear growth, no reset.

**Historical context**: A related FSDP2 CPU offload memory leak was fixed by PR #149428 (weiyangfb, Jan 2025), addressing reference counting in `_post_backward_hook` and `_offload_to_cpu` path. But verl #6468 is a different leak path -- Gloo staging buffers, not the backward hook reference cycle.

### 2.5 When Will FSDP2 Be GRPO-Safe?

Assessment based on current trajectory:

| Milestone | Status | Timeline Estimate |
|-----------|--------|-------------------|
| FSDP2 stable API | Done (2.7+) | Completed |
| FSDP2 + torch.compile integration | Done (2.7+) | Completed |
| FSDP2 CPU offload memory leak (backward hook) | Fixed (#149428, Jan 2025) | Completed |
| FSDP2 Gloo weight sync memory leak | **OPEN** (verl #6468) | No fix yet |
| FSDP2 PartialOffloadPolicy for dp=1 | Draft (#187620) | In review -- but DOES NOT work on dp=1! |
| FSDP2 per-unit summon for LoRA | verl #6512 | Available |
| Complete GRPO-safe on RTX 4090 | **NOT YET** | Blocked by #6468 |

**Conclusion**: FSDP2 will be GRPO-safe only when:
1. The Gloo staging buffer leak (#6468) is fixed in PyTorch upstream
2. verl switches to NCCL-based weight sync (instead of Gloo) for multi-GPU setups
3. PartialOffloadPolicy (#187620) or a dp=1-specific offload policy is finalized

**Current RTX 4090 pragmatic path**: LoRA + compile + single-GPU (no FSDP) for GRPO. Multi-GPU FSDP2 GRPO is blocked by memory leak.

---

## 3. Compile Stack Evolution (Dynamo/Inductor/AOTAutograd)

### 3.1 Compile Stack Architecture (Already Deep-Read)

Refer to `/notebook/projects/pytorch-compile-stack-knowledge-synthesis.md` for the full 5-layer architecture:

```
Layer 1: Entry + Framework
  torch.compile(mode, fullgraph, dynamic) -> _TorchCompileInductorWrapper -> torch._dynamo.optimize

Layer 2: Dynamo (C-level -> Python -> FX)
  _PyInterpreterState_SetEvalFrameFunc -> VariableTracker hierarchy -> InstructionTranslator
  Guard system: CacheEntry + GuardedCode + 8 guard types -> max 64 recompiles -> fallback
  Graph break: Unsupported/SkipFrame/RestartAnalysis -> multi-segment compilation

Layer 3: FX IR + AOTAutograd
  FX Node (6op + SymInt) -> Graph双向链表 -> Interpreter+boxed
  AOTAutograd: make_fx -> joint fwd+bwd -> min-cut partition -> functionalization

Layer 4: Inductor (Lowering + Scheduler + Triton Codegen)
  Lowering: fx node -> SchedulerNode -> Buffer -> Memory planning
  Scheduler: fused node grouping -> memory-efficient scheduling -> fusion regions
  Triton Codegen: SchedulerBuffer -> Triton kernel templates -> autotune

Layer 5: Execution
  CompiledFunction + Backward -> CUDA graphs / eager fallback / Triton JIT
```

### 3.2 Compile Stability Improvements Timeline

| Version | Key Improvement | Impact |
|---------|----------------|--------|
| 2.0 (2023) | torch.compile introduced (Dynamo+Inductor) | Baseline -- many graph breaks |
| 2.1 | Improved fallback mechanism, AOTAutograd stability | Fewer crashes |
| 2.2 | Reduced graph breaks, better dynamic shape support | More models compile |
| 2.3 | Expanded custom op registration, fewer recompilations | Custom ops work |
| 2.4/2.5 | Continued error rate reduction, improved Inductor codegen | Production-viable for many models |
| 2.6 | AOTInductor for mobile/embedded, FlexAttention preview | Deployment path |
| 2.7 | FlexAttention stable, compiled autograd preview, FSDP2+compile major interop | Major milestone |
| 2.8 | FSDP2+compile edge cases, dynamic shapes in Inductor for FSDP2 | Deepening |
| 2.9/2.10 | Compiled autograd targeted for stable, dynamic shape improvements | Key goal |
| 2.12 | Inductor stream support, activation offloading ops, torch.cond+CUDA graphs | Infrastructure |
| 2.13 | **REGRESSION**: Inductor AssertionErrors in scheduler/codegen/dynamic shapes | Caution -- use 2.13.1 |

### 3.3 Dynamo Guard Mechanism Internals

Key source files:
- `torch/_dynamo/guards.py` -- Guard implementation (TensorGuard, GLOBAL_STATE_GUARD, ClosureGuard, DunderGuard)
- `torch/_dynamo/symbolic_shapes.py` -- Shape specialization and dynamic shape guards
- `torch/_dynamo/output_graph.py` -- Graph break insertion and fallback code generation
- `torch/_dynamo/eval_frame.py` -- Entry point, guard checking, recompilation loop

**Guard mechanism flow**:
1. During tracing, Dynamo creates guards based on observed tensor metadata (shape, stride, dtype, device) and Python state (closure variables, global references)
2. Guards are compiled into fast-check lambda functions
3. At runtime, guards are evaluated before executing cached compiled code
4. If guards fail, recompilation is triggered with new assumptions
5. Maximum 64 recompilations (cache_size_limit) before falling back to eager entirely

**2025-2026 improvements**:
- Flexible graph break policies (`torch._dynamo.config.graph_break_policy`)
- Guard simplification/deduplication (reducing guard explosion)
- Improved identity-based guards (`torch._dynamo.utils.identity_checks`)
- Better AOTAutograd + graph break interaction for backward compilation

### 3.4 Inductor Autotune -- SM89 Critical Path

Already deep-read in `/notebook/projects/pytorch-187636-autotune-compile-time-reading.md`

**Key change**: PR #187636 flips `autotune_at_compile_time` to False by default:
- Before: compile-time Triton configs were selected and baked into binary -> deterministic at compile time
- After: runtime Triton configs selected via CachingAutotuner -> dynamic -> batch-dependent on SM89
- Default False: skip compile-time autotune -> runtime autotune only -> SAME XBLOCK selection
- Result: batch-invariant results on SM89 (soft guarantee)

**Complementary to P9 Fusion Guard** (#184119):
- P9: blocks ALL reduction fusions on SM<90 -> HARD guarantee
- #187636: skips compile-time autotune -> SOFT guarantee (same XBLOCK selection)
- Together: hard + soft = near-complete batch invariance on SM89

**Environment variables for autotune control**:

| Variable | Purpose |
|----------|---------|
| `TORCHINDUCTOR_AUTOTUNE_CACHE` | Enable/disable autotune caching |
| `TORCHINDUCTOR_CACHE_DIR` | Directory for compiled artifacts and cache |
| `TORCHINDUCTOR_FORCE_MAX_AUTOTUNE` | Force exhaustive autotune search |
| `TORCH_LOGS` | Set to `+autotune` for debugging |

### 3.5 SM89 Fusion Guard -- The RTX 4090 Compile Problem

Already deep-read in `/notebook/projects/pytorch-184119-sm89-fp8-prologue-fusion-guard-reading.md`

**The bug class**: Inductor generates SM90-only Triton code on SM89 hardware. Two manifestations:

1. **P9 pattern**: Inductor fuses reduction (mean) into pointwise -> `tl.sum()` with autotuned XBLOCK -> different XBLOCK -> batch-dependent numerical results
2. **#184119 pattern**: Inductor fuses fp8 cast into mm template prologue -> SM90-only bf16 conversion -> SM89 hardware cannot execute -> CRASH

**Same root cause**: Inductor fusion creates capability-dependent code. The scheduler does not check SM capability before fusing operations.

**Fix approach** (#184119): 5-line guard + 2 helper functions + 42-line regression test:
- `can_fuse_fp8_prologue()` checks `current_device_sm() >= 90` before allowing fp8-to-bf16 fusion into mm template
- Prevents SM90-only Triton code generation on SM89 devices

---

## 4. PyTorch Distributed Training Internals

### 4.1 ProcessGroupNCCL Architecture

**Key source files**:
- `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` -- Class definition, `abort_` flag, `workList_`, watchdog thread declarations
- `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` -- `monitorWorker_()`, `abort()`, `WorkNCCL` implementation, timeout/error checking

**Watchdog thread architecture**:
- `WorkNCCL` objects enqueued into `std::deque<std::shared_ptr<WorkNCCL>> workList_`
- Watchdog thread (`monitorWorker_`) periodically checks:
  - Timeout (`opTimeout_`, default ~30 minutes)
  - NCCL error status (NCCL returns non-success)
- Watchdog check interval: `kWatchdogTimeoutIntervalSec` (default 10 seconds)

**Abort mechanism**:
1. Watchdog detects timeout/error on a WorkNCCL item
2. Sets `abort_ = true` (irreversible)
3. Calls `ncclCommAbort()` on each `ncclComm_t` in `ncclComms_` (force-abort, NOT `ncclCommDestroy`)
4. Marks all pending WorkNCCL as aborted
5. Other ranks detect NCCL error -> also enter abort path (cooperative cleanup)
6. ProcessGroupNCCL permanently unusable after abort -- must reconstruct ProcessGroup

**Key design decisions**:
- `ncclCommAbort` vs `ncclCommDestroy`: Abort forcefully terminates in-progress operations; Destroy waits for completion (may never happen if a rank is stuck)
- Abort is irreversible: NCCL communicators cannot be partially reset
- Watchdog is per-ProcessGroup, not per-collective
- `TORCH_NCCL_ASYNC_ERROR_HANDLING`: Environment variable enabling watchdog/abort behavior (now default in many configs)

**Recent improvements (2025-2026)**:
- Async error surfacing without blocking entire training loop
- Better diagnostic tools (NCCL debug logs integration with PyTorch profiler)
- CUDA graph compatibility for NCCL operations (important for torch.compile integration)
- Experimental collective offloading to CPU/network to reduce GPU idle time
- Cooperative wait: side-channel signaling ensures all ranks abort together
- NCCL 2.18+ `ncclCommGetAsyncError` for detecting async errors before hangs

### 4.2 DTensor Evolution

**DTensor** (Distributed Tensor) is becoming the primary distributed abstraction in PyTorch:

| Timeline | Milestone |
|----------|-----------|
| 2024 | DTensor experimental, DeviceMesh basic sharding |
| 2025 H1 | API stabilization, DeviceMesh flex topologies, heterogeneous hardware |
| 2025 H2 | Native checkpointing (`torch.save`/`torch.load`), tensor parallelism APIs (TP/PP/DP patterns) |
| 2026 (2.12) | **Twice-differentiable DTensor** (autograd through DTensor ops), per-parameter meshes, DataParallelMeshDims SPMD |

**Key DTensor concepts**:
- **DeviceMesh**: Defines the physical device topology for sharding
- **Sharding Spec**: `Shard(dim)`, `Replicate()`, `Partial()` -- how a DTensor is distributed across the mesh
- **Per-parameter meshes (2.12)**: Different parameters can be on different meshes, enabling hybrid parallelism strategies within a single model
- **DataParallelMeshDims**: SPMD mesh abstraction that coordinates FSDP2 + DTensor sharding

**torch.compile + DTensor integration (2025-2026)**:
- Major effort to make torch.compile work seamlessly with DTensor
- Sharding annotations propagate through the compiled graph
- Enables compiled distributed training with automatic sharding management

### 4.3 batch_isend_irecv Compilation (2.12)

PyTorch 2.12 made `batch_isend_irecv` compilable with torch.compile:
- Batches multiple point-to-point `isend`/`irecv` operations into a single NCCL group call
- Uses `ncclGroupStart()`/`ncclGroupEnd()` internally
- Compilation support enables pipeline parallelism operations within compiled regions
- Important for composable PP+FSDP2+compile workflows

### 4.4 FSDP2 Breaking Change (2.12): fullgraph

PyTorch 2.12 introduced a breaking change for FSDP2 fullgraph mode:
- FSDP2's compiler-friendly hooks previously allowed fullgraph compilation
- 2.12 changes the hook attachment mechanism, potentially requiring `fullgraph=False` for some models
- Impact: Models that previously compiled with `fullgraph=True` may need reconfiguration

---

## 5. RTX 4090 Implications for GRPO Training with PyTorch FSDP

### 5.1 Hardware Limitations

| Factor | RTX 4090 Spec | Impact on FSDP2 |
|--------|---------------|-----------------|
| Compute Capability | SM_89 (Ada Lovelace) | Not SM_90; no TMA, no SM90-exclusive kernels |
| VRAM | 24 GiB GDDR6X | Tight for GRPO (rollouts x K multiply activations) |
| Interconnect | No NVLink; PCIe 4.0 x16 only | ~32 GiB/s bidirectional vs NVLink 4.0 ~600 GiB/s -- 18x slower |
| FP8 | E4M3/E5M2 supported | FP8 training possible but Inductor fusion guards needed |

### 5.2 FSDP2 Scaling Efficiency on PCIe-Only RTX 4090

| Setup | FSDP2 Scaling (2->4 GPUs) | Efficiency |
|-------|---------------------------|-----------|
| 2x RTX 4090 (PCIe 4.0) | ~1.6x speedup | 80% |
| 4x RTX 4090 (same root complex) | ~2.5x speedup | 62% |
| 4x RTX 4090 (different root complexes) | ~1.8x speedup | 45% |
| 2x RTX 3090 (NVLink) | ~1.85x speedup | 92% |

**Note**: The RTX 3090 with NVLink achieves 92% scaling efficiency, significantly better than the 4090's 80% at 2 GPUs. This is a well-known irony in the community -- the older 3090 is arguably better for multi-GPU distributed training due to NVLink support.

### 5.3 GRPO Training Challenges on RTX 4090

| Challenge | Description | Mitigation |
|-----------|-------------|-----------|
| CPU memory leak (#6468) | Gloo staging buffers accumulate linearly | No upstream fix yet; switch to NCCL weight sync |
| Peak memory spikes | FSDP2 all-gather momentarily holds full unsharded params | SHARD_GRAD_OP, gradient checkpointing, bf16 |
| GRPO group rollouts | K rollout generations multiply activation memory | Reduce group size K, gradient checkpointing |
| Reference model for KL | Doubles model memory (policy + reference) | LoRA (only train adapters), shard both models |
| Variable-length sequences | Uneven sequence lengths reduce sharding efficiency | Pad/bucket sequences |
| SM89 fusion guard | Inductor generates SM90-only code on SM89 | P9 guard (#184119), enforce_eager, autotune_at_compile_time=False |
| NCCL timeout/errors | PCIe topology causes NCCL hangs on multi-GPU | `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` |

### 5.4 RTX 4090 Pragmatic GRPO Training Path

**Single GPU (recommended for most)**:
- LoRA/QLoRA + bf16 + torch.compile (with enforce_eager or autotune guards)
- No FSDP needed on single GPU
- 7B model fits with LoRA: ~8 GiB base weights + ~2 GiB LoRA adapters + ~8 GiB activations
- Gradient checkpointing for longer sequences

**Multi-GPU (2x RTX 4090, cautious)**:
- FSDP2 SHARD_GRAD_OP + bf16 + gradient checkpointing
- Minimize GRPO group size K
- Use NCCL weight sync (avoid Gloo due to memory leak)
- Expect ~80% scaling efficiency

**Multi-GPU (4x+ RTX 4090, NOT recommended)**:
- PCIe bottleneck makes FSDP2 all-gather latency dominant
- 62% efficiency at 4 GPUs on same root complex
- CPU memory leak makes long GRPO runs (>100 steps) impossible
- Consider renting cloud A100/H100 instead

### 5.5 DeepSpeed Alternative for RTX 4090

For consumer GPU setups, DeepSpeed ZeRO-3 + CPU/NVMe offloading is often preferred over native FSDP2:
- More granular partitioning and offloading options
- ZeRO-Infinity: offload optimizer states + parameters + gradients to CPU/NVMe
- Better for memory-constrained setups
- BUT: DeepSpeed has its own bugs (see `/notebook/projects/deepspeed-8075-fd-leak-reading.md` and others)

---

## 6. DeepSpeed vs Megatron vs PyTorch FSDP Comparison (2026 Update)

### 6.1 Feature Comparison

| Feature | DeepSpeed (v0.16+) | Megatron-LM/Megatron-Core | PyTorch FSDP/FSDP2 |
|---------|-------------------|---------------------------|---------------------|
| Data Parallelism | ZeRO 1-3, Infinity | Standard DP | FULL_SHARD/SHARD_GRAD_OP/NO_SHARD |
| Tensor Parallelism | Via integration | Best-in-class TP | Via DTensor (2.12+ per-param meshes) |
| Pipeline Parallelism | Yes | Interleaved PP | No built-in (needs external) |
| Sequence/Context Parallelism | Yes | Advanced SP/CP | No |
| CPU/NVMe Offloading | Most mature (ZeRO-Infinity) | Limited | CPUOffloadPolicy + PartialOffloadPolicy (draft #187620) |
| MoE Support | DeepSpeed-MoE | Via NeMo | No MoE-specific |
| torch.compile | Partial | No | Native (FSDP2 designed for compile) |
| Ease of Use | Medium | Hard | Easy (native PyTorch) |
| Max Model Scale Proven | 175B+ | 530B+ | 100B+ (Meta internal) |
| Hardware Flexibility | Multi-vendor | NVIDIA-optimized | Multi-vendor |
| SM89/SM90 Compatibility | Depends on config | SM90-optimized | Inductor fusion guard needed |

### 6.2 2026 Trends: Composable Parallelism

The landscape is trending toward **composable parallelism** -- picking primitives from each framework rather than committing to one monolithic solution:

1. **FSDP2 + torch.compile**: Becoming go-to for many teams (20-40% speedups with minimal effort)
2. **Megatron-Core modularity**: NVIDIA's refactoring makes TP/PP primitives more reusable and composable with FSDP2
3. **DTensor per-parameter meshes (2.12)**: Enables hybrid TP+DP+EP within a single FSDP2 model
4. **torch.accelerator.Graph**: Unified graph API enables cross-backend (CUDA/XPU) optimization

### 6.3 When to Use Each Framework

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| Quick start / minimal code | PyTorch FSDP2 | Native PyTorch, minimal changes |
| torch.compile integration | PyTorch FSDP2 | FSDP2 designed for compile |
| Memory-constrained single GPU | DeepSpeed ZeRO-Infinity | CPU/NVMe offload |
| RTX 4090 GRPO training | LoRA + compile (no FSDP) | FSDP2 memory leak blocks long runs |
| Extreme-scale (>100B) | Megatron-LM or DeepSpeed | Most proven at massive scale |
| NVIDIA datacenter cluster | Megatron-LM/NeMo | SM90-optimized, highest throughput |
| Research / experimentation | PyTorch FSDP2 | Easiest to iterate |
| Long-sequence training | Megatron-LM | Context Parallelism |

---

## 7. Key Source Code References

### 7.1 FSDP2 Source

| Component | Path | Description |
|-----------|------|-------------|
| FSDP2 main API | `torch/distributed/_composable/fsdp.py` | `fully_shard()`, FSDPParamGroup, _FSDPState |
| Param group lifecycle | `torch/distributed/_composable/fsdp/_fsdp_param_group.py` | `unshard()`, `reshard()`, `_post_backward_hook()` |
| Common utilities | `torch/distributed/_composable/fsdp/_fsdp_common.py` | Sharding state enums, hook ordering |
| Mixed precision | `torch/distributed/_composable/fsdp/_fsdp_policy.py` | MixedPrecisionPolicy |
| CPU offload | `torch/distributed/_composable/fsdp/_fsdp_state.py` | CPUOffloadPolicy, offload lifecycle |

### 7.2 Compile Stack Source

| Component | Path | Description |
|-----------|------|-------------|
| Dynamo entry | `torch/_dynamo/eval_frame.py` | C-level frame evaluation replacement, guard checking |
| Dynamo guards | `torch/_dynamo/guards.py` | 8 guard types, guard compilation, cache management |
| Dynamo symbolic shapes | `torch/_dynamo/symbolic_shapes.py` | Shape specialization, dynamic shape guards |
| Dynamo output | `torch/_dynamo/output_graph.py` | FX graph construction, graph break insertion |
| AOTAutograd | `torch/_functorch/aot_autograd.py` | Joint fwd+bwd graph, min-cut partition |
| Inductor lowering | `torch/_inductor/lowering.py` | FX node -> SchedulerNode/Buffer |
| Inductor scheduler | `torch/_inductor/scheduler.py` | Fused node grouping, fusion regions, SM-aware guards |
| Inductor Triton codegen | `torch/_inductor/codegen/triton.py` | Triton kernel templates, autotune config selection |
| Autotune process | `torch/_inductor/autotune_process.py` | Autotuning subprocess, config caching |

### 7.3 ProcessGroupNCCL Source

| Component | Path | Description |
|-----------|------|-------------|
| NCCL header | `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` | Class definition, `abort_`, `workList_`, watchdog |
| NCCL implementation | `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` | `monitorWorker_()`, `abort()`, WorkNCCL |

### 7.4 DTensor Source

| Component | Path | Description |
|-----------|------|-------------|
| DTensor API | `torch/distributed/tensor/api.py` | DTensor creation, sharding spec |
| DeviceMesh | `torch/distributed/tensor/device_mesh.py` | Mesh topology definition |
| DTensor autograd | `torch/distributed/tensor/_autograd.py` | Twice-differentiable DTensor (2.12+) |

---

## 8. Critical Bugs Tracker (RTX 4090 Impact)

| Bug | Issue/PR | Severity | RTX 4090 Impact | Status |
|------|----------|----------|-----------------|--------|
| FSDP2 CPU memory leak (Gloo) | verl #6468 | CRITICAL | Long GRPO runs OOM (~40 steps) | OPEN |
| Inductor SM90-only code on SM89 | #184119 | HIGH | Crash/wrong results on SM89 | OPEN (PR pending) |
| autotune_at_compile_time batch dependence | #187636 | HIGH | Batch-dependent numerical results on SM89 | Fixed (default False) |
| torch 2.13 Inductor AssertionError | #136643 umbrella | HIGH | Blocks torch.compile on RTX 4090 | Partially fixed (2.13.1) |
| vLLM w8a8 block-FP8 Inductor regression | #187484 | HIGH | Blocks vLLM torch 2.13 upgrade | OPEN |
| FSDP2 PartialOffloadPolicy dp=1 | #187620 | MEDIUM | Does not work on single GPU (dp=1) | DRAFT |
| NCCL timeout on multi-4090 PCIe | Various | MEDIUM | NCCL hangs on multi-GPU PCIe setups | Env var workarounds |

---

## 9. Forward-Looking Assessment (June 2026)

### 9.1 What's Coming Next

Based on PyTorch's trajectory and roadmap:

| Expected Feature | Version | Timeline | RTX 4090 Relevance |
|-----------------|---------|----------|---------------------|
| Compiled autograd stable | 2.9-2.10 | Late 2025 / Early 2026 | Better compile coverage |
| Zero graph break target | 2.11+ | 2026 | More models compile fully |
| FSDP2 + torch.export | 2.8-2.9 | 2025-2026 | AOT distributed compilation |
| FlexAttention + FSDP2 | 2.8+ | 2025 | Distributed custom attention |
| FP8 training expansion | 2.11+ | 2026 | RTX 4090 FP8 hardware (SM89 supports FP8) |
| Quantization formats (FP4, MXFP4) | 2.12+ | 2026 | RTX 5090 SM120 direction |
| ProcessGroupNCCL collective offload | 2.9+ | 2025-2026 | Reduce GPU idle time |
| Custom backend registration for compile | 2.10+ | 2026 | Third-party accelerator support |
| FP8 matmul autotune on SM89 | TBD | Future | Direct RTX 4090 benefit |

### 9.2 FSDP2 GRPO-Safe Timeline Assessment

**Current blockers** (June 2026):
1. CPU memory leak in Gloo weight sync path (verl #6468) -- no upstream fix
2. Inductor SM89 fusion guard (#184119) -- PR open, not yet merged
3. torch 2.13 Inductor regressions (#136643) -- partially fixed, #187484 still open
4. PartialOffloadPolicy (#187620) -- draft, does not work on dp=1

**Estimated timeline for FSDP2 GRPO-safe on RTX 4090**:
- Q3 2026: If Gloo leak fix lands + Inductor SM89 guard merges + 2.13.1/2.14 stabilizes compile
- Q4 2026: More likely realistic timeline -- all blockers resolved, FSDP2 GRPO tested on consumer hardware
- 2027: Production-hardened FSDP2 GRPO on consumer GPUs (with NCCL weight sync, per-unit LoRA summon, and activation offloading ops)

**Until then**: LoRA + compile + single-GPU remains the pragmatic RTX 4090 GRPO path.

---

## 10. References

### 10.1 Official Sources

| Source | URL |
|--------|-----|
| PyTorch Blog | https://pytorch.org/blog/ |
| PyTorch 2.12 Release | https://github.com/pytorch/pytorch/releases/tag/v2.12.0 |
| PyTorch GitHub Issues | https://github.com/pytorch/pytorch/issues |
| PyTorch Nightly Docs | https://pytorch.org/docs/nightly/ |
| PyTorch Roadmap | https://github.com/pytorch/pytorch/wiki |

### 10.2 Key Issues and PRs

| Issue/PR | Title | Relevance |
|----------|-------|-----------|
| #184119 | SM89 fp8 prologue fusion guard | SM89 compile safety |
| #187636 | autotune_at_compile_time=False default | SM89 batch invariance |
| #187484 | vLLM w8a8 block-FP8 Inductor regression on 2.13 | Blocks vLLM 2.13 upgrade |
| #187620 | FSDP2 PartialOffloadPolicy | Memory-constrained dp=1 |
| #136643 | 2.13 regressions umbrella | Compile stability |
| #149428 | FSDP2 CPU offload memory leak fix (Jan 2025) | Historical fix |
| verl #6468 | FSDP2 CPU memory leak during weight sync | Current blocker |

### 10.3 Project Reading Notes

| Note | Path | Focus |
|------|------|-------|
| PyTorch 2.12 Features | `/notebook/projects/pytorch-2.12-features-reading.md` | Full 2.12 feature deep-read |
| PyTorch v2.12 Release Impact | `/notebook/projects/pytorch-v2.12-release-reading.md` | RTX 4090 impact analysis |
| FSDP2 Deep Reading | `/notebook/projects/pytorch-fsdp2-2026-deep-reading.md` | FSDP2 internals |
| Compile Stack Synthesis | `/notebook/projects/pytorch-compile-stack-knowledge-synthesis.md` | Full compile stack |
| SM89 Fusion Guard | `/notebook/projects/pytorch-184119-sm89-fp8-prologue-fusion-guard-reading.md` | SM89 compile bug |
| Autotune Flip | `/notebook/projects/pytorch-187636-autotune-compile-time-reading.md` | Autotune change |
| verl FSDP2 Leak | `/notebook/projects/verl-6468-fsdp2-cpu-memory-leak-reading.md` | Memory leak blocker |
| PartialOffloadPolicy | `/notebook/projects/pytorch-187620-fsdp2-partial-offload-policy-reading.md` | dp=1 offload gap |
| vLLM Inductor Regression | `/notebook/projects/pytorch-187484-vllm-inductor-torch213-regression.md` | 2.13 regression |

---

*End of Session 3 Deep Reading*
