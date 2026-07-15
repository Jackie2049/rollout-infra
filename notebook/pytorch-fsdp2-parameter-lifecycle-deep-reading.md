# PyTorch FSDP2 Parameter Lifecycle — Comprehensive Deep Reading

> 2026-07-15 | Synthesis of 13 source readings | RTX 4090 GRPO relevance throughout
> Source files: pytorch-torch-distributed-internals-deep-dive.md, pytorch-fsdp2-2026-deep-reading.md,
>   pytorch-fsdp2-internals-reading.md, verl-7016-moe-fsdp2-backward-deep-analysis.md,
>   verl-6468-fsdp2-cpu-memory-leak-reading.md, verl-fsdp2-source-deep-reading.md,
>   pytorch-fsdp2-single-gpu-analysis.md, verl-fsdp-weight-sync-mechanism-reading.md,
>   pytorch-187620-fsdp2-partial-offload-policy-reading.md,
>   cross-framework-fsdp-evolution-synthesis.md, pytorch-compile-fsdp2-integration-reading.md,
>   verl-weight-sync-memory-leak-6468.md

---

## 1. FSDP2 Architecture: Per-Parameter Sharding via _fsdp_param_group.py

### 1.1 The Fundamental Shift: FlatParameter to DTensor

FSDP1's central design flaw was the FlatParameter -- all parameters in an FSDP
unit were flattened into a single 1D buffer, destroying parameter identity,
names, and shapes. Debugging was opaque because the flat buffer made it
impossible to inspect individual parameters.

FSDP2 replaces this with per-parameter DTensor sharding. Each nn.Parameter
becomes an independent DTensor via `DTensor.from_local(Shard(0))`, preserving
the original name, shape, and dtype. This is the foundational architectural
change that enables everything else FSDP2 does differently.

```
FSDP1 architecture:
  nn.Parameter (4096, 4096) --|
  nn.Parameter (7, 4096)    --|--> FlatParameter [16384+28+... padded to 8*N]
  nn.Parameter (4096,)      --|
  -> Names lost. Shapes lost. Padding waste proportional to world_size.
  -> One FlatParamHandle manages all params in unit.
  -> summon_full_params() materializes ALL params at once.

FSDP2 architecture:
  nn.Parameter (4096, 4096) --> DTensor(Shard(0)) shard=(512, 4096) per-rank
  nn.Parameter (7, 4096)    --> DTensor(Shard(0)) shard=(1, 4096) per-rank [rank 7 gets 0]
  nn.Parameter (4096,)      --> DTensor(Shard(0)) shard=(512,) per-rank
  -> Names preserved. Shapes preserved. Per-param padding = at most (world_size-1) elements.
  -> One FSDPParam per parameter. FSDPParamGroup manages collection.
  -> foreach_all_gather coalesces shards but splits output back to per-param.
```

### 1.2 _fsdp_param_group.py: The Core Management Class

The file `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py` is the
central orchestrator. Each FSDPParamGroup corresponds to one `fully_shard()`
call on a module. Its key attributes:

```
FSDPParamGroup:
  -> fsdp_params: list[FSDPParam]     -- per-parameter sharding state
  -> process_group: ProcessGroup      -- NCCL communication group
  -> device: Device                   -- GPU device
  -> reshard_after_forward: bool      -- post-forward reshard policy
  -> offload_policy: OffloadPolicy    -- CPU offload configuration
  -> _training_state: TrainingState   -- FORWARD / PRE_BACKWARD / BACKWARD / IDLE
```

Each FSDPParam (`_fsdp_param.py`) tracks:

```
FSDPParam:
  -> orig_param: Parameter            -- original nn.Parameter (identity preserved!)
  -> sharded_param: Parameter         -- DTensor shard (only 1/N of original)
  -> unsharded_param: Tensor          -- full param (AllGather result, temporary)
  -> _shard_state: ShardState         -- SHARDED / UNSHARDED / SHARDED_POST_FORWARD
  -> _unshard_state: UnshardState     -- NOT_UNSHARDED / UNSHARDING / UNSHARDED
  -> offload_to_cpu: bool             -- per-param CPU offload decision
  -> pin_memory: bool                 -- pinned CPU memory (DMA transfer)
  -> orig_dtype / param_dtype / reduce_dtype  -- mixed precision state
```

### 1.3 Initialization: fully_shard() -> _init_param_group()

When `fully_shard(module)` is called, the following sequence executes:

```
1. @contract(state_cls=FSDPState) creates FSDPState for module
2. mesh = DeviceMesh("cuda", (world_size,), mesh_dim_names=("shard",))
3. _get_modules_and_states() DFS collects manageable modules
4. state.init(modules, device, mp_policy, auto_reshard_after_forward)
5. _init_param_group() creates FSDPParamGroup:
   -> For each nn.Parameter in module:
     -> torch.chunk(param_data, shard_world_size, dim=shard_dim)
     -> chunks[shard_rank] = local shard
     -> Zero-pad to padded_sharded_size for AllGather alignment
     -> CPU offload: move to CPU + optional pin_memory
     -> nn.Parameter(to_sharded_dtensor(sharded_local), requires_grad=...)
     -> unsafe_setattr_param() writes directly to module._parameters
6. MRO insertion: module.__class__ = FSDP<OrigClass>(FSDPModule, OrigClass)
```

The MRO insertion (step 6) is a key design choice. Instead of wrapping the
module in a new class (FSDP1's FullyShardedDataParallel), FSDP2 modifies the
module's class to inherit from both FSDPModule and the original class. This
makes FSDP2 transparent -- __getattr__ passes through to the original module,
no surprise attribute hiding.

### 1.4 Padding Waste: FSDP1 vs FSDP2

```
FSDP1 padding (8 GPU):
  -> Flatten all params to 1D -> pad to 8*N (world_size multiple)
  -> 7B model: could waste several MB of GPU memory per FlatParamHandle
  -> Example: [16384+28+4096] total = 20508 -> pad to 20508*8 = 164064 -> waste ~164064-20508*8

FSDP2 padding (8 GPU):
  -> Per-param: [4096, 4096] weight -> shard [512, 4096] -> evenly divisible -> NO padding
  -> Per-param: [7, 4096] bias -> shard [1, 4096] (rank 0-6) or [0] (rank 7) -> pad 7 elements total
  -> Total padding waste across all params: negligible (few KB vs several MB)
  -> Per-param sharding almost eliminates padding waste entirely
```

### 1.5 DTensor Native State Dict

FSDP2 state dicts are DTensor-native. Each parameter's `.state_dict()` entry
is a DTensor whose `_local_tensor` is the local shard. This means:

- Save: `model.local_state_dict()` returns per-rank shards -> each rank saves its own shard
- Load: each rank loads its own shard -> `DTensor.from_local()` restores DTensor
- Full checkpoint: `with model.unshard(): model.state_dict()` materializes all params temporarily

vs FSDP1 which required `ShardedStateDictConfig` / `FullStateDictConfig` with
complex mapping tables. FSDP2's DTensor-native approach is simpler, more
human-readable, and less error-prone.

---

## 2. Shard -> Unshard -> Compute -> Reshard Cycle: The Forward Pass Lifecycle

### 2.1 The 4-Hook Forward Lifecycle

```
Forward lifecycle for each FSDPParamGroup:

Phase 1: Pre-Forward (unshard)
  _pre_forward_hook triggers:
  -> FSDPState._pre_forward() -> _lazy_init() (first call only)
  -> fsdp_param_group.pre_forward():
     -> TrainingState = FORWARD
     -> self.unshard() -> foreach_all_gather()
     -> self.wait_for_unshard() -> copy_out to per-param unsharded views
     -> RegisterPostBackwardFunction autograd node (backward hook registration)

Phase 2: Forward Compute
  -> Module forward runs with FULL (unsharded) parameters
  -> Standard PyTorch computation -- no sharding overhead during compute
  -> Activations computed normally

Phase 3: Post-Forward (reshard)
  _post_forward_hook triggers:
  -> fsdp_param_group.post_forward():
     -> self.reshard() -> if reshard_after_forward:
        -> _to_sharded() -> free_storage(unsharded) -> param.data = sharded_param
     -> _record_post_forward_order() -> records for backward prefetching

Phase 4: (No phase -- parameters are now sharded, waiting for backward)
```

### 2.2 AllGather Pipeline: 3-Stream Overlap

The `foreach_all_gather()` in `_fsdp_collectives.py` uses a 3-stream pipeline:

```
Stream 1: copy_in_stream
  -> dtype conversion: sharded_data -> param_dtype cast
  -> Copy shards into AllGather output buffer (contiguous, ready for NCCL)
  -> Non-blocking: overlaps with previous computation

Stream 2: all_gather_stream (NCCL dedicated stream)
  -> Wait for copy_in_stream completion (CUDA event barrier)
  -> all_gather_comm(output, input, process_group)
  -> WorkNCCL async operation -> does not block default compute stream

Stream 3: copy_out_stream (default stream)
  -> Wait for all_gather_stream completion (CUDA event barrier)
  -> foreach_all_gather_copy_out() -> split AllGather output to per-param views
  -> torch.as_strided(all_gather_output, _orig_size, _contiguous_orig_stride)
  -> VIEW, not copy! -> zero additional memory allocation

Key: as_strided creates views of the AllGather output buffer, NOT new tensors.
This means unsharded params are views of the AllGather output, and when the
AllGather output buffer is freed (reshard), the views become invalid -- which
is exactly the desired behavior (prevent accidental use of stale data).
```

### 2.3 Data Flow Diagram: Forward Pass

```
[Sharded State]                    [Unsharded State]                [Compute]

 rank_0: shard_W0    AllGather    full_W = [W0|W1|W2|...|W7]    forward(layer_i)
 rank_1: shard_W1    -------->    (per-param view via             with full_W
  ...                  NCCL        as_strided)                    activations
 rank_7: shard_W7                 stored in AllGather             computed
                                  output buffer)

[After Forward]                    [Sharded State]                 [Ready for Bwd]

 free AllGather      reshard      param.data = shard_Wi           backward will
 output buffer        -------->    free_storage(unsharded=0)      re-unshard
                                  storage.resize(0)               via AllGather

Time sequence:
  t0: copy_in (dtype cast + buffer prepare)
  t1: all_gather (NCCL collective)
  t2: copy_out (as_strided views)
  t3: forward compute (uses full params)
  t4: reshard (free AllGather output, swap param.data back to shard)
```

### 2.4 Prefetching: Communication-Compute Overlap

FSDP2 prefetch uses the recorded `post_forward_order` to predict which module
will be needed next and start its AllGather early:

```
Forward prefetch (正序):
  Computing layer i forward -> simultaneously prefetch layer i+1 AllGather
  -> When layer i+1 forward starts -> params already ready -> no stall!

Backward prefetch (逆序):
  Computing backward layer i -> simultaneously prefetch layer i-1 AllGather
  -> Backward order = reverse of forward order -> recorded from post_forward_order
  -> When backward layer i-1 starts -> params already ready -> no stall!

Prefetch policies:
  None: sequential (no overlap) -> simplest -> lowest throughput
  Backward-only: only backward prefetch -> useful when forward is lightweight
  Always: forward + backward both prefetch -> maximum overlap -> highest throughput
```

RTX 4090 implication: PCIe AllGather for 7B model takes ~1.5 seconds per
layer. Prefetching cannot fully overlap this because compute per layer is
only ~50-100ms. The ratio is 15:1 (comm vs compute) -- overlap provides
negligible benefit on PCIe. On NVLink, the ratio is ~1:1, so overlap is
critical.

---

## 3. Backward Pass: reduce_scatter Gradient Flow

### 3.1 The 4-Hook Backward Lifecycle

```
Backward lifecycle for each FSDPParamGroup:

Phase 1: Pre-Backward (unshard -- same as forward)
  RegisterPreBackwardFunction backward hook triggers:
  -> FSDPState._pre_backward():
     -> TrainingState = PRE_BACKWARD
     -> Register root module post-backward final callback
     -> fsdp_param_group.pre_backward():
        -> self.unshard() -> if reshard_after_forward=True: AllGather again
                             if False: params already unsharded -> skip
        -> self.wait_for_unshard() -> copy_out
        -> _backward_prefetch() -> start AllGather for next backward module

Phase 2: Backward Compute
  -> Module backward runs with FULL (unsharded) parameters
  -> Gradients computed for each parameter (stored on .grad attribute)

Phase 3: Post-Backward (reduce_scatter + reshard)
  fsdp_param_group.post_backward():
  -> accumulate_unsharded_grad_if_needed() -> accumulate in reduce_dtype
  -> Per FSDPParam gradient extraction:
     -> unsharded_accumulated_grad / unsharded_param.grad / zero_grad
     -> DTensor gradients: DP dimension stays Partial -> FSDP handles RS
  -> Buffer recycling: if max_input_buffers limit -> wait oldest RS -> free
  -> foreach_reduce():
     -> Pre-divide factor: ReduceScatter pre_div=1, post_div=1/world_size
     -> HSDP: pre=1/replicate_size, post=1/shard_size
     -> reduce_scatter_stream: torch.ops.fsdp.chunk_cat -> concat grads
     -> reduce_scatter_comm(output, input, process_group)
     -> HSDP: also all_reduce on replicate dimension
     -> CPU offload: RS output -> D2H -> CPU -> optional pin_memory()
  -> Reshard: if reshard_after_backward -> param.data = sharded_param

Phase 4: Finalize Backward (root module callback)
  Variable._execution_engine.queue_callback():
  -> All groups post_backward() complete
  -> finalize_backward() -> wait post-reduce events -> clear RS state
```

### 3.2 Gradient Flow Diagram: Backward Pass

```
[Unsharded State]                  [Backward Compute]               [ReduceScatter]

 full_W = [W0|W1|...|W7]          backward(layer_i)                grad_shard_0
 (AllGather output view)           computes:                        grad_shard_1
                                    full_W.grad = full_grad_W       grad_shard_2
                                    (full gradient tensor)          ...
                                                                  grad_shard_7

[After ReduceScatter]              [Sharded State]                  [Optimizer Step]

 each rank gets 1/N                 param.data = shard_Wi           shard_grad stored
 of the gradient:                   free_storage(unsharded=0)       optimizer updates
 grad_shard_i has                   back to sharded state            sharded_param only
 correct 1/N gradient                                               (1/N memory!)

ReduceScatter math:
  input:  full_grad_W (shape = orig_size, on each rank)
  output: grad_shard_i = full_grad_W[shard_start:shard_end] / world_size
  -> Each rank contributes its full gradient
  -> NCCL reduce_scatter sums across ranks AND partitions
  -> Result: each rank has its shard of the AVERAGED gradient
  -> Division: pre_div=1, post_div=1/world_size -> averaged before partitioning
```

### 3.3 Mixed Precision in Backward

```
MixedPrecisionPolicy affects backward gradient flow:

  param_dtype (e.g., BF16):
    -> AllGather output is BF16 -> forward compute in BF16
    -> Gradients computed in BF16 (autograd tracks compute dtype)

  reduce_dtype (e.g., FP32):
    -> Before ReduceScatter: grad.cast_to(reduce_dtype)
    -> accumulate_unsharded_grad_if_needed() -> FP32 accumulation buffer
    -> ReduceScatter runs in FP32 -> more numerically stable reduction
    -> After ReduceScatter: gradient shard stored in FP32 on CPU (if offloaded)

  orig_dtype (e.g., FP32):
    -> sharded_param ALWAYS stored in orig_dtype -> FP32 shard on CPU
    -> Optimizer step in FP32 -> no loss scaling needed
    -> BF16 compute + FP32 optimizer = FSDP2 recommended config -> safer than FP16 + loss scaling

Key insight: sharded params are NEVER stored in param_dtype. They stay in
orig_dtype. The cast to param_dtype happens ONLY during AllGather (copy_in).
This means:
  -> 7B model: 7B * 4 bytes (FP32 shard) = 28 GB on CPU -> but only 7B * 2 bytes (BF16) during compute
  -> Optimizer step on FP32 shard -> full precision -> no loss scaling -> simpler than FSDP1 FP16
```

### 3.4 Buffer Recycling in ReduceScatter

FSDP2 implements buffer recycling to prevent memory spikes during backward:

```
max_input_buffers limits in-flight ReduceScatter operations:
  -> Without limit: each backward layer allocates 1 RS buffer -> L layers * buffer_size
  -> With limit: only N buffers in flight -> oldest buffer freed when new one needed
  -> O(1) buffer consumption instead of O(L)
  -> Similar to DeepSpeed ZeRO-3 max_ongoing_backpressure(2 events) -> same pattern!

This is critical for RTX 4090:
  -> 32 decoder layers * ~350 MiB RS buffer = ~11.2 GiB without recycling
  -> With max_input_buffers=2: only ~700 MiB RS buffers at any time
  -> 16x memory reduction in backward buffers alone!
```

---

## 4. param.data Swap: Zero-Copy Sharding via Pointer Swap

### 4.1 The Core Mechanism

FSDP2 uses Python's `param.data` attribute swap to transition between sharded
and unsharded states. This is a zero-copy operation -- it only changes the
pointer that `param.data` references, not the underlying data.

```
Unshard (pre-forward):
  param.data = unsharded_param    # swap pointer to AllGather output view
  -> Module now sees full (unsharded) parameter
  -> Forward compute uses full param -> normal computation
  -> No data copying! Just pointer swap!

Reshard (post-forward):
  param.data = sharded_param      # swap pointer back to shard
  -> Module now sees only its shard (1/N of original)
  -> free_storage(unsharded_param) -> resize storage to 0 -> release GPU memory
  -> No data copying! Just pointer swap + storage free!

Key: unsafe_setattr_param() bypasses nn.Module.__setattr__ hooks
  -> Directly writes to module._parameters[name] = new_value
  -> Avoids __setattr__ overhead and potential side effects
  -> This is why FSDP2 can do zero-copy swaps at Python level
```

### 4.2 Storage Lifecycle: alloc_storage and free_storage

```
alloc_storage (unshard):
  -> unsharded_param.storage().resize_(unsharded_numel) -> allocate GPU memory
  -> Or: AllGather output buffer provides the storage -> view via as_strided
  -> Storage object reused across forward/backward calls -> not malloc per step
  -> Only first call allocates -> subsequent calls reuse existing storage

free_storage (reshard):
  -> unsharded_param.storage().resize_(0) -> shrink storage to 0 -> release GPU memory
  -> Storage object preserved (not deleted) -> can be resized again later
  -> PyTorch CUDA allocator: resize to 0 = free block -> returns to allocator pool
  -> empty_cache() NOT needed -> allocator manages pool efficiently

Critical subtlety: free_storage preserves the Storage object but sets size to 0.
  -> autograd may have saved views sharing this Storage
  -> resize to 0 on shared Storage: all views become empty (size 0)
  -> This is SAFE because: backward recomputes via activation checkpointing
  -> Saved tensors from forward are NOT the parameter views -- they're activation tensors
  -> Parameter views are only needed during live forward/backward computation
  -> Between phases, they can be freed safely
```

### 4.3 Three Reshard Strategies

```
Strategy 1: reshard_after_forward=True (FULL_SHARD, default for non-root)
  -> Forward: AllGather -> compute -> reshard (free unsharded)
  -> Backward: AllGather again -> compute -> ReduceScatter -> reshard
  -> Memory: peak = shard + one layer full params (during compute)
  -> Communication: 2 AllGather + 1 ReduceScatter per layer per step
  -> Best for: memory-constrained scenarios, multi-GPU with many layers

Strategy 2: reshard_after_forward=False (NO reshard, default for root)
  -> Forward: AllGather -> compute -> KEEP unsharded
  -> Backward: NO AllGather needed (already unsharded) -> compute -> ReduceScatter
  -> Memory: peak = shard + ALL forward layer full params simultaneously!
  -> Communication: 1 AllGather + 1 ReduceScatter per layer per step
  -> Best for: communication-constrained scenarios (PCIe, slow interconnect)
  -> DANGER: all forward layers' params stay on GPU -> peak = full model!

Strategy 3: Int value (HSDP partial reshard)
  -> Forward: AllGather -> compute -> reshard to sub-group (not full shard)
  -> Backward: AllGather from sub-shard -> compute -> ReduceScatter in sub-group
  -> Memory: peak = shard + one layer sub-shard params
  -> Communication: AllGather in sub-group + ReduceScatter in sub-group + all_reduce in replicate
  -> Best for: HSDP (hybrid sharded data parallelism)

Auto (default): non-root=True, root=False
  -> Non-root modules: release after forward -> save memory -> backward prefetch can overlap
  -> Root module: keep after forward -> backward starts immediately -> no AllGather stall
  -> Optimal combination: middle layers save memory + root layer saves communication
```

### 4.4 RTX 4090: reshard_after_forward=False Paradox

```
RTX 4090 single GPU (dp=1):
  -> reshard_after_forward=True: AllGather is identity (1 shard = full param) -> overhead only
  -> reshard_after_forward=False: keeps full params -> also full param (same state)
  -> BOTH strategies = same memory footprint on dp=1 (full param either way)
  -> FSDP2 on dp=1 = pure overhead with NO benefit -> USE FSDP1 or DeepSpeed ZeRO-2 instead!

RTX 4090 multi-GPU (dp>=2):
  -> reshard_after_forward=True: saves 50% memory (shard = 1/dp of param) -> meaningful!
  -> reshard_after_forward=False: keeps full params on each GPU -> 2x memory vs sharded
  -> PCIe bottleneck: AllGather for 7B model ~1.5s per layer -> reshard=True = 3s total comm
  -> reshard=False = 1.5s comm but 2x memory -> tradeoff: speed vs memory
  -> PCIe so slow that even reshard=False doesn't help enough -> multi-GPU RTX 4090 = marginal
```

---

## 5. MoE + FSDP2 Conflict (verl #7016): Gradient Graph Divergence

### 5.1 The Root Cause

MoE (Mixture of Experts) models violate FSDP2's fundamental assumption of
SPMD (Same Program, Multiple Data). The MoE router selects different experts
for different tokens, creating data-dependent execution paths that differ
across ranks.

```
The SPMD assumption:
  FSDP2 assumes: every rank executes the SAME computation graph
  -> Same parameters produce gradients -> same ReduceScatter input sizes
  -> NCCL ReduceScatter expects identical input sizes across all ranks
  -> If sizes mismatch -> NCCL collective fails -> SIGSEGV or CheckpointError

MoE violates SPMD:
  -> Router: top-K expert selection per token -> data-dependent
  -> Different ranks: different input data -> different expert selections
  -> Expert parameters that are NOT selected produce NO gradients
  -> Different ranks: different subsets of parameters with gradients
  -> ReduceScatter sees mismatched input sizes -> NCCL collective fails!
```

### 5.2 Two Failure Modes

```
Failure Mode A: CheckpointError (gradient checkpointing ON)
  -> Activation checkpointing saves intermediate activations during forward
  -> During backward: recomputation of forward pass
  -> MoE router: different expert-dispatch path during recomputation
  -> This creates non-deterministic saved-tensor count -> CheckpointError
  -> The count of saved tensors depends on which experts were visited
  -> Different recomputation paths -> different tensor counts -> mismatch

Failure Mode B: SIGSEGV (gradient checkpointing OFF)
  -> Full computation graph held in memory during backward
  -> FSDP2 tries ReduceScatter for all parameters in the FSDPParamGroup
  -> Different ranks have different parameter subsets with gradients
  -> NCCL ReduceScatter: input size mismatch across ranks -> SIGSEGV
  -> No Python traceback -> crash at NCCL C++ level -> hard to debug
```

### 5.3 The Fix: PyTorch PR #174862

The fix introduces zero-gradient buffers for unused parameters, ensuring all
ranks present the same ReduceScatter input sizes:

```
Step 1: Zero buffer initialization (one-time per FSDPParam)
  _fsdp_param.py init_dtype_attrs():
    self._zero_buf = torch.zeros(1, dtype=grad_dtype, device=self.sharded_param.device)
  -> Single element zero tensor, allocated once per param
  -> expand() used for zero-cost views -> no per-step allocation

Step 2: Unsharded zero grad data (per backward call)
  _fsdp_param.py unsharded_zero_grad_data property:
    if is_dtensor:
      return _get_grad_inner_tensor(torch.zeros_like(unsharded_param))  # actual allocation for TP
    else:
      return _get_grad_inner_tensor(_zero_buf.expand(_orig_size))  # zero-cost view!
  -> Ensures NCCL ReduceScatter sees same input size on all ranks
  -> DTensor path: needs actual allocation (TP dimension)
  -> Regular FSDP path: expand() = zero-cost view

Step 3: Post-backward tracking
  _fsdp_param_group.py post_backward():
    for each fsdp_param:
      if param has grad: append real gradient
      if param NO grad: append unsharded_zero_grad_data (zero buffer via expand)
      track index in _locally_unused_params set

Step 4: Global coordination (finalize_backward)
  globally_used = torch.ones(len(fsdp_params), ...)
  for i in _locally_unused_params: globally_used[i] = 0
  dist.all_reduce(globally_used, op=ReduceOp.MAX)
  -> If ANY rank used the param -> globally_used[i] = 1
  -> Globally unused params: grad set to None -> optimizer skips them
  -> Prevents Adam momentum/adaptive LR corruption from zero gradients

Naming confusion: _locally_unused_params actually tracks LOCALLY USED indices
  (params that had gradients). The name is misleading.
```

### 5.4 Workaround: Expert Parameter Consolidation

Transformers PR #41580 consolidates all expert parameters into a single large
nn.Parameter. This ensures every expert computation creates gradients for ALL
experts (since they're all part of the same parameter), eliminating the unused-
param problem entirely.

```
Before consolidation:
  expert_0.weight -> nn.Parameter (may or may not get gradients)
  expert_1.weight -> nn.Parameter (may or may not get gradients)
  ...
  expert_7.weight -> nn.Parameter (may or may not get gradients)
  -> Different ranks activate different experts -> different gradient sets

After consolidation:
  experts.weight -> nn.Parameter (concatenation of all expert weights)
  -> Any expert computation creates gradients for ALL experts
  -> ReduceScatter sees same input size on all ranks -> no mismatch
  -> FSDP2 + MoE works correctly without PR #174862
```

### 5.5 Silent Corruption Risk

Even when training doesn't crash, FSDP2 + MoE gradient graph mismatch can
cause **silent weight corruption**. NCCL may complete the ReduceScatter
operation but distribute incorrect gradient data -- weights that should be
zero get non-zero values from other ranks, and weights that should have
real gradients get zeros. This is WORSE than a crash because training
appears to succeed but model quality silently degrades over many steps.

### 5.6 RTX 4090 Implications

```
Single GPU (dp=1):
  -> Failure Mode B (SIGSEGV): likely does NOT occur
     -> Only one rank -> no NCCL ReduceScatter needed -> no size mismatch
  -> Failure Mode A (CheckpointError): CAN still occur
     -> Data-dependent saved-tensor count is a LOCAL issue
     -> Autograd graph reconstruction depends on expert selection during forward
     -> Different recomputation paths -> different tensor counts -> local mismatch

RTX 4090 MUST DO:
  1. Use FSDP1 instead of FSDP2 with MoE models (confirmed working by reporter)
  2. If FSDP2 required: disable gradient checkpointing AND validate backward stability
  3. Consider expert parameter consolidation (Transformers PR #41580)
  4. Monitor PyTorch PR #174862 for merge status
  5. NEVER assume FSDP2 + MoE is safe without explicit testing
```

---

## 6. CPU Memory Leak (verl #6468): FSDP2 all_gather Accumulation

### 6.1 The Leak Path

```
WorkerDict.update_weights()
  -> actor.engine.get_per_tensor_param()
    -> FSDP2 DTensor full tensor materialization
      -> _dtensor_full_tensor_gloo() (when VERL_FSDP2_WEIGHT_SYNC_GLOO=1)
        -> local_tensor = dtensor.to_local().detach().cpu().contiguous()
        -> torch.distributed.all_gather_object(...)
        -> full_tensor = torch.cat(...)
      -> staging tensors/buffers NOT released after each sync
    -> For NCCL path: param.to(device).full_tensor().to(torch.bfloat16)
      -> Creates GPU staging buffer -> then .cpu() for transfer
      -> Multiple intermediate tensors per param per step
  -> rollout.update_weights(...)
```

### 6.2 Five Leak Points

```
LEAK 1 -- DTensor.full_tensor() CPU staging buffers (PRIMARY)
  -> Each full_tensor() creates a full-sized CPU staging buffer
  -> ~340 separate CPU staging buffers per sync step (for 2B model)
  -> Pageable (not pinned) CPU tensors -> PyTorch allocator may not release to OS
  -> .to(torch.bfloat16, non_blocking=True) creates ANOTHER intermediate tensor
  -> Generator pattern -> tensors from consecutive steps may overlap in memory

LEAK 2 -- Old staging round trip (pre-PR#7005, FIXED)
  -> load_fsdp_model_to_gpu() / offload_fsdp_model_to_cpu() round trip
  -> Hundreds of small blocking H2D + D2H copies -> fragmented CPU allocator
  -> Fixed by PR #7005 (merged Jul 10, 2026)

LEAK 3 -- LoRA collect_lora_params CPU accumulation
  -> lora_params = {name: param.full_tensor().detach().cpu() ...}
  -> Creates CPU tensors for every LoRA parameter each step
  -> empty_cache() only clears GPU cache -> CPU tensors remain

LEAK 4 -- backup_base_model_weights clone leakage
  -> backup[name] = param.data.clone().cpu()
  -> Clones all parameters to CPU simultaneously -> PyTorch allocator may not release pages

LEAK 5 -- Gloo all_gather_object staging buffers
  -> torch.distributed.all_gather_object() serializes Python objects through Gloo
  -> Creates temporary buffers never explicitly freed
```

### 6.3 Leak Scaling Pattern

```
| Model Size | Leak Rate | OOM Steps (251 GiB host) | OOM Steps (32 GiB host, RTX 4090) |
|-----------|-----------|-------------------------|-----------------------------------|
| Qwen3.5-2B | ~0.6 GiB/step | ~400 | ~40 -> OOM! |
| Qwen2.5-3B | ~5.3 GiB/step | ~47 | ~6 -> instant OOM! |
| Qwen3-35B | ~6.3 GiB/step | ~40 | instant OOM |

Scaling formula: leak_rate ~ 0.3 * n_params/B GiB/step (0.3 GiB per billion active parameters)

Pattern classification: Level 5 Intermittent Accumulation
  E(t) = alpha * t * (1 - beta)^(t/T_step)
  For #6468: beta ~ 0 (nothing released) -> linear growth -> guaranteed OOM

RTX 4090 (8B model estimate):
  -> Estimated leak ~2.4 GiB/step -> host OOM in ~8-22 steps (32-64 GiB host)
  -> Standard 1000-step GRPO training IMPOSSIBLE without workaround
```

### 6.4 Why FSDP1 Doesn't Have This Leak

FSDP1 uses `summon_full_params()` context manager that properly cleans up
after each use. The context manager:
  -> Enters: materializes full parameters
  -> Exits: frees full parameters, restores sharded state
  -> Cleanup: explicit free + Python context manager guarantees cleanup

FSDP2 uses DTensor which may retain staging buffers:
  -> full_tensor() creates temporary buffers
  -> No explicit cleanup mechanism (no context manager)
  -> PyTorch CPU allocator caches freed blocks for reuse -> RSS grows
  -> Generator closures hold references to intermediate tensors

### 6.5 Workaround Options for RTX 4090

```
Option 1: NCCL weight sync (not Gloo)
  -> VERL_FSDP2_WEIGHT_SYNC_GLOO=0 -> NCCL all-gather -> staging on GPU, not CPU
  -> GPU staging may also have leaks -> needs verification
  -> On dp=1: NCCL broadcast is identity -> minimal data movement

Option 2: SGLang sleep/wake with NCCL checkpoint engine
  -> dp=1: NCCL broadcast = identity -> no DTensor materialization needed
  -> CheckpointEngineManager handles efficiently (ZMQ + NCCL + CuPy)
  -> Doesn't create CPU staging buffers -> no leak
  -> BEST workaround for RTX 4090

Option 3: Periodic process restart
  -> Restart rollout worker every 20-50 steps -> clear accumulated host memory
  -> Disrupts training flow and KV cache -> suboptimal

Option 4: Manual garbage collection
  -> gc.collect() + torch.cuda.empty_cache() after each weight sync
  -> May help but not guaranteed (PyTorch allocator may hold references)

Option 5: Use FSDP1 backend instead of FSDP2
  -> FSDP1 summon_full_params properly cleans up -> no leak
  -> FSDP1 with per-unit LoRA summon (#6512) is proven safe path
  -> RTX 4090 recommendation: USE FSDP1 until #6468 leak is fixed
```

### 6.6 Monitoring Recommendation

```python
import psutil

def check_host_ram():
    process = psutil.Process()
    rss = process.memory_info().rss / (1024**3)  # GiB
    total = psutil.virtual_memory().total / (1024**3)
    if rss > 0.8 * total:
        logger.warning(f"Host RAM {rss:.1f} GiB > 80% of {total:.1f} GiB -- FSDP2 leak risk!")
        # Consider restarting rollout worker or switching to FSDP1
```

This monitoring is MANDATORY for any RTX 4090 GRPO training using FSDP2.

---

## 7. Comparison: FSDP1 vs FSDP2

### 7.1 Architectural Differences

```
| Feature | FSDP1 | FSDP2 |
|---------|-------|-------|
| Parameter representation | FlatParameter (flatten all params to 1D) | per-param DTensor (independent) |
| Sharding unit | FlatParamHandle per FSDP unit | FSDPParam per parameter |
| Padding waste | FlatParameter padded to world_size*N -> potentially several MB | per-param padded to shard slot -> negligible |
| Hook mechanism | FullyShardedDataParallel wrapper -> override forward() | _composable @contract + MRO insertion |
| State management | _FSDPState attached to wrapper class | FSDPState via _composable_state |
| Communication | per FlatParamHandle AllGather/RS | per FSDPParamGroup foreach_all_gather/RS |
| State dict | ShardedStateDictConfig / FullStateDictConfig | DTensor native -> _local_tensor |
| torch.compile | INCOMPATIBLE -> FlatParameter captured to graph -> static views | COMPATIBLE -> original params -> dynamic -> UnspecializedNNModule |
| Backward hooks | FlatParamHandle hooks | RegisterPostBackwardFunction + pre_backward hooks |
| Mixed precision | loss scaling required for FP16 | BF16 orig_dtype + param_dtype cast -> no loss scaling |
| CPU offload | CPUOffload(offload_params=True) -> binary choice | CPUOffloadPolicy / PartialOffloadPolicy -> fractional choice (#187620) |
| API style | FSDP(model) -> class wrapper | fully_shard(model) -> functional API |
| Composable | NOT composable -> single wrapper -> blocks TP/AC | Composable -> hooks -> can combine TP+AC+FSDP |
| Summon API | summon_full_params() context manager | DTensor.full_tensor() all-gathers on demand |
```

### 7.2 When Each Is Appropriate

```
Use FSDP1 when:
  -> Using MoE models with data-dependent routing (#7016)
  -> Need summon_full_params() context manager for explicit cleanup
  -> Working with existing FSDP1 codebase (migration cost)
  -> On RTX 4090 single GPU with verl (FSDP1 proven safe, no CPU leak #6468)
  -> Need ZeRO-2 style gradient partitioning without FSDP2 overhead

Use FSDP2 when:
  -> Need torch.compile compatibility (FSDP1 completely incompatible)
  -> Need per-parameter sharding for TP+DP mesh combinations
  -> Need composable API (combine FSDP + TP + AC)
  -> Need DTensor-native state dicts (simpler, human-readable)
  -> Need model-internal mixed sharding strategies (big layers FULL_SHARD + small NO_SHARD)
  -> On NVLink multi-GPU setups where AllGather overhead is manageable
  -> Need HSDP (2D DeviceMesh) for hybrid sharded data parallelism
  -> Need per-unit LoRA summon (#6512) with DTensor (more memory-friendly)
```

### 7.3 Communication Volume Comparison

```
ZeRO-3 (FSDP1 equivalent):
  -> 3*Psi per step (AllGather forward + AllGather backward + ReduceScatter grads)
  -> AllGather + ReduceScatter = 2*Psi communication per parameter per step

FSDP2 FULL_SHARD:
  -> 2*Psi per step (AllGather forward + ReduceScatter grads)
  -> Backward AllGather included if reshard_after_forward=True
  -> 33% less communication than ZeRO-3!

FSDP2 SHARD_GRAD_OP:
  -> 1*Psi per step (ReduceScatter grads only)
  -> No forward AllGather (params already replicated)
  -> 66% less communication than ZeRO-3!

FSDP2 NO_SHARD (DDP):
  -> 1*Psi per step (AllReduce grads only)
  -> No sharding at all -> same as vanilla DDP
```

---

## 8. RTX 4090 Implications: dp=1 No Sharding, FSDP1 Safer

### 8.1 FSDP2 on RTX 4090 Single GPU = Pure Overhead

```
dp_world_size = 1 on RTX 4090:
  -> partition_size = full -> NO sharding occurs
  -> Each parameter: full copy on single GPU -> no savings
  -> AllGather: identity operation -> gather from 1 shard -> no-op
  -> ReduceScatter: identity operation -> scatter to 1 shard -> no-op

What FSDP2 ACTUALLY does on single GPU:
  -> Wraps every module in FSDPModule -> adds overhead per module
  -> Pre-forward hook -> triggers AllGather (identity) -> overhead
  -> Post-forward hook -> frees gathered shard -> overhead
  -> Backward hook -> triggers ReduceScatter (identity) -> overhead
  -> ALL hooks = no-ops in data movement -> BUT still add latency!
  -> Buffer management, callback scheduling, state tracking -> all overhead

Net effect:
  -> Memory: NO savings (full parameters anyway) -> same as vanilla DDP
  -> Compute: NO savings (no gradient partitioning) -> same as vanilla
  -> Latency: ADDS overhead (FSDP hooks, buffer management) -> WORSE than vanilla
  -> FSDP2 on single GPU = WORSE than vanilla DDP or FSDP1 -> pure overhead!
```

### 8.2 Why FSDP1 Is Safer for RTX 4090

```
FSDP1 advantages on RTX 4090:
  1. summon_full_params() context manager -> explicit cleanup -> no CPU leak (#6468)
  2. Proven compatibility with MoE models (#7016) -> no gradient graph divergence
  3. ZeRO-2 mode -> partition optimizer states -> CPU offload -> 4.7x memory reduction
  4. CPU_Adam integration -> AVX512 optimized -> 5-7x faster CPU optimizer
  5. Production-tested for 2+ years -> stable -> no ALPHA crash bugs

FSDP2 disadvantages on RTX 4090:
  1. CPUOffloadPolicy = ALPHA -> crash bugs reported -> NOT production-ready
  2. CPU memory leak during weight sync (#6468) -> 0.6-6.3 GiB/step -> guaranteed OOM
  3. MoE gradient graph divergence (#7016) -> SIGSEGV or CheckpointError
  4. Single GPU = pure overhead -> no benefit from sharding
  5. DTensor overhead on dp=1 -> unnecessary complexity
```

### 8.3 RTX 4090 GRPO Optimal Configuration

```
Configuration ranking for RTX 4090 GRPO:

#1: verl CPPO + bypass + FSDP1 ZeRO-2 + CPU_Adam + LoRA-32 + SGLang
  -> ZeRO-2: optimizer states on CPU -> GPU peak ~16.2 GiB -> fits 24 GiB
  -> CPU_Adam: AVX512 optimized -> fast CPU optimizer step
  -> LoRA rank=32: trainable params = 0.2% -> minimal optimizer load
  -> bypass_mode: skip old_log_prob forward -> 18Psi -> 3.8Psi memory
  -> SGLang: tag-based sleep/wake -> explicit memory control
  -> FSDP1: proven safe on RTX 4090 -> no #6468 leak, no #7016 MoE crash
  -> Peak GPU: ~17-19 GiB -> 5-7 GiB margin -> safe -> production-tested

#2: verl GRPO + bypass + FSDP1 ZeRO-2 + CPU_Adam + LoRA-32 + SGLang
  -> Same as #1 but without CPPO -> slightly less optimal
  -> Still safe and production-tested

#2.5: DeepSpeed ZeRO-2 + CPU_Adam + LoRA-32 (standalone)
  -> 15-25% faster per-step than FSDP2 CPUOffloadPolicy
  -> But: no CPPO, no verl integration -> less optimal for GRPO
  -> overlap_comm=False MUST (NaN risk #8061)
  -> gradient_clipping=1.0 MUST (default 0 is broken #8068)

#3: rLLM Tinker GRPO LoRA (standalone, no distributed)
  -> In-process -> no distributed overhead -> simplest
  -> LoRA-32 + BF16 + torch.compile -> good for single GPU
  -> But: limited to simple GRPO -> no CPPO, no advanced estimators

BLOCKED: verl FSDP2 backend on RTX 4090
  -> CPUOffloadPolicy ALPHA crash bugs -> NOT safe
  -> CPU memory leak #6468 -> guaranteed OOM in long-running training
  -> MoE #7016 crash -> SIGSEGV or CheckpointError
  -> Single GPU = pure overhead -> no sharding benefit
```

### 8.4 RTX 4090 Memory Budget Comparison

```
| Method | Model Params | Gradients | Optimizer | Activations | Total | Fits 24GB? |
|--------|-------------|-----------|-----------|-------------|-------|------------|
| Vanilla DDP | 8GB | 8GB | 16GB | 2-4GB | 34-36GB | NO |
| ZeRO-2+CPU_Adam (FSDP1) | 8GB | 8GB | 0(CPU) | 2-4GB | 10-12GB | YES (12GB margin) |
| FSDP2 (dp=1) | 8GB | 8GB | 16GB | 2-4GB | 34-36GB | NO (same as DDP) |
| FSDP2+CPUOffload (ALPHA) | 4GB | 4GB | 8GB(CPU) | 2-4GB | 10-12GB | YES (but crash bugs) |
| ZeRO-2+LoRA-32+bypass | ~8GB | ~0.02GB | 0(CPU) | ~3.8GB | ~12GB | YES (12GB margin) |

Best config: ZeRO-2 + CPU_Adam + LoRA-32 + bypass = ~17-19 GB peak
  -> 5-7 GB margin on 24 GB GPU -> safe -> production-tested -> FSDP1 path
```

### 8.5 FSDP2 Multi-GPU Future for RTX 4090

```
Current: RTX 4090 single GPU
  -> FSDP2 pointless -> FSDP1 ZeRO-2+CPU_Adam is proven path
  -> LoRA-32 + bypass + SGLang = optimal GRPO stack

Future: RTX 4090 x2 (dp=2, PCIe)
  -> FSDP2 SHARD_GRAD_OP -> 3x memory savings -> but PCIe bottleneck
  -> PartialOffloadPolicy (ratio=0.3) -> 70% resident, 30% CPU -> faster forward
  -> AllGather: 7B/2 = ~1.75 GB per rank -> PCIe ~650 ms -> significant overhead
  -> Net: marginal improvement over single GPU -> PCIe limits scaling

Future: RTX 4090 x8 (dp=8, PCIe)
  -> FSDP2 FULL_SHARD -> 8x memory savings -> but PCIe catastrophic
  -> AllGather: 7B/8 = ~0.22 GB per rank -> PCIe ~80 ms per layer
  -> 32 layers * 80 ms = ~2.56 s forward communication alone
  -> FSDP2 8 GPU = 0.46x throughput -> WORSE than single GPU!

Conclusion:
  -> RTX 4090 single GPU + ZeRO-2 + CPU_Adam = BEST current path
  -> RTX 4090 multi-GPU = NOT viable due to PCIe bottleneck
  -> NVLink GPU (A100/H100) needed for multi-GPU FSDP2 to be beneficial
  -> FSDP2 future value = NVLink environments, NOT PCIe RTX 4090
```

### 8.6 Complete RTX 4090 FSDP Decision Tree

```
Decision 1: Does model fit with NO offload? (peak < 24 GiB)
  -> Qwen3-1.7B: ~11 GiB -> YES -> OffloadPolicy (no offload) + FSDP1 ZeRO-2
  -> Qwen3-8B LoRA (per-unit summon): ~16.2 GiB -> YES -> FSDP1 ZeRO-2
  -> Without per-unit summon: ~64 GiB -> NO -> go to Decision 2

  If YES -> Use FSDP1 ZeRO-2 (no offload needed) + CPU_Adam for optimizer savings
  If NO -> Go to Decision 2

Decision 2: Does model fit with CPU optimizer offload? (peak < 24 GiB)
  -> ZeRO-2 + CPU_Adam: peak = model + grads + activations (optimizer on CPU)
  -> Qwen3-8B LoRA + bypass: ~17-19 GiB -> YES -> FSDP1 ZeRO-2 + CPU_Adam
  -> Qwen3.5-27B LoRA: ~20-22 GiB -> MAYBE -> tight -> need gradient checkpointing

  If YES -> Use FSDP1 ZeRO-2 + CPU_Adam -> proven path -> stable -> safe
  If MAYBE -> Add gradient checkpointing -> reduce activations by ~70%
  If NO -> Go to Decision 3

Decision 3: Model doesn't fit even with optimizer offload?
  -> Reduce batch size -> fewer activations
  -> Gradient checkpointing -> reduce activations by ~70%
  -> Multi-GPU -> dp>=2 -> but PCIe bottleneck on RTX 4090

  -> Real answer: use NVLink GPU (A100/H100) for models >27B on single GPU
  -> RTX 4090 sweet spot: 7B-14B LoRA models with ZeRO-2 + CPU_Adam
```

### 8.7 RTX 4090 MUST DO / MUST NOT Summary

```
MUST DO:
  1. Use FSDP1 backend (not FSDP2) until #6468 CPU leak is fixed
  2. Use ZeRO-2 + CPU_Adam for optimizer offload -> 4.7x memory reduction
  3. Use LoRA rank=32 (rank=64 breaks vLLM EOS, #6782)
  4. Use overlap_comm=False (overlap_comm+compile = NaN, #8061)
  5. Set gradient_clipping=1.0 (default 0 is broken, #8068)
  6. Use per-unit LoRA summon (#6512) for weight sync -> 10x peak reduction
  7. Use SGLang rollout (tag-based sleep/wake, LoRA adapter path)
  8. Monitor host RAM growth during training (psutil)
  9. Use bypass_mode for GRPO -> 18Psi -> 3.8Psi memory reduction
  10. Use lora.merge=False (adapter path) -> sleep_level=1 -> no full model re-transfer

MUST NOT:
  1. MUST NOT use FSDP2 backend for long-running GRPO (>100 steps) until #6468 fixed
  2. MUST NOT use FSDP2 CPUOffloadPolicy (ALPHA, crash bugs)
  3. MUST NOT use FSDP2 + MoE without PR #174862 or expert consolidation
  4. MUST NOT use ZeRO-3 on single GPU (pure overhead, no benefit)
  5. MUST NOT use Gloo weight sync on RTX 4090 (CPU staging buffers that leak)
  6. MUST NOT use DeepSpeed overlap_comm=True on single GPU (NaN, #8061)
  7. MUST NOT use Muon optimizer (6 blockers, not viable on RTX 4090)
  8. MUST NOT use lora.merge=True (forces sleep_level=2, OOM on 8B+)
  9. MUST NOT use load_format=dummy (forces full base sync, 16 GiB payload)
  10. MUST NOT run >100 steps without host RAM monitoring (#6468 leak)
```

---

## 9. Cross-References and Source Trail

```
Primary sources (this reading synthesizes):
  - notebook/fundamentals/pytorch-torch-distributed-internals-deep-dive.md
    -> Section 4: FSDP2 lifecycle, reshard strategies, RTX 4090 implications
  - notebook/projects/pytorch-fsdp2-2026-deep-reading.md
    -> FSDP2 vs FSDP1, 3 strategies, MixedPrecisionPolicy, HSDP, prefetch, compile
  - notebook/projects/pytorch-fsdp2-internals-reading.md
    -> Source-level: fully_shard, FSDPParam, unshard/reshard, backward RS, compile integration
  - notebook/projects/verl-7016-qwen3-moe-fsdp2-backward-deep-analysis.md
    -> MoE + FSDP2 conflict, PR #174862 fix, RTX 4090 implications
  - notebook/projects/verl-6468-fsdp2-cpu-memory-leak-reading.md
    -> CPU memory leak, leak scaling, workaround options
  - notebook/verl-weight-sync-memory-leak-6468.md
    -> 5 leak points, fix approaches, PR #7005
  - notebook/projects/verl-fsdp2-source-deep-reading.md
    -> verl FSDP2 engine source, LoRA summon, HYBRID weight sync
  - notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
    -> FSDP2 dp=1 = pure overhead, CPUOffloadPolicy ALPHA, ZeRO-2+CPU_Adam
  - notebook/projects/verl-fsdp-weight-sync-mechanism-reading.md
    -> Complete weight sync pipeline, summon mechanics, LoRA path
  - notebook/projects/pytorch-187620-fsdp2-partial-offload-policy-reading.md
    -> PartialOffloadPolicy, dp=1 issues, RTX 4090 decision tree
  - notebook/fundamentals/cross-framework-fsdp-evolution-synthesis.md
    -> Summon/release lifecycle pattern, 3 optimization levels
  - notebook/projects/pytorch-compile-fsdp2-integration-reading.md
    -> DTensor shard_dim graph breaks, 3-phase roadmap

Key PyTorch source files:
  - torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py (FSDPParamGroup)
  - torch/distributed/fsdp/_fully_shard/_fsdp_param.py (FSDPParam, DTensor sharding)
  - torch/distributed/fsdp/_fully_shard/_fsdp_state.py (FSDPState, hooks)
  - torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py (AllGather, ReduceScatter)
  - torch/distributed/fsdp/_fully_shard/_fsdp_api.py (MixedPrecisionPolicy, CPUOffloadPolicy)
  - torch/distributed/fsdp/_fully_shard/_fully_shard.py (fully_shard function, FSDPModule)
  - torch/distributed/fsdp/_dynamo_utils.py (Dynamo integration)

Key issues and PRs:
  - verl #7016: MoE + FSDP2 backward failure
  - verl #6468: FSDP2 CPU memory leak
  - verl #6512: per-unit LoRA summon (MERGED)
  - verl #6699: detach memory fix (MERGED)
  - verl #6782: LoRA rank=64 breaks EOS
  - verl #6794: delta weight sync
  - verl #7005: skip FSDP2 staging round trip (MERGED)
  - PyTorch #174862: reduce_scatter unused params fix
  - PyTorch #187620: PartialOffloadPolicy
  - PyTorch #187615: PartialOffloadPolicy scoping RFC
  - DeepSpeed #8061: overlap_comm + compile = NaN
  - DeepSpeed #8068: gradient clipping default 0 broken
```

---

*Created 2026-07-15. PyTorch FSDP2 parameter lifecycle comprehensive deep reading.*
*RTX 4090 recommendation: FSDP1 ZeRO-2 + CPU_Adam + LoRA-32 + bypass = proven safe path.*
*FSDP2 on RTX 4090 dp=1 = pure overhead + CPU leak + MoE crash risk = AVOID.*
