# Megatron #5387 — MFSDPv2 Deep Source-Level Reading

> 2026-06-18 | PR #5387 OPEN (APPROVED by shjwudp, CI triggered, merge blocked by codeowners) | +993/-3 | 8 files | branch: fsdp/minimal
> Author: wujingyue (Jingyue Wu, NVIDIA)
> Recovered replacement for #4976 (GitHub closed after base ref deletion), now rebased on main
> Foundation: #4835 (DBuffer minimal implementation, +1353/-0, MERGED June 17)
> Follow-up: #5369 (meta-parameter support, +2398/-0, OPEN draft)
> CI status: unit tests + distributed tests PASS (27 passed, 6 skipped); blocked by codeowners-approval and multi-approval-bot-summary

---

## 1. PR Overview and Context

### 1.1 What This PR Does

Adds an experimental per-module Megatron-FSDP `fully_shard` path that uses DBuffer primitives to shard parameters, materialize full weights for compute, and reduce gradients back into sharded optimizer state. This is Megatron's own FSDP implementation (MFSDPv2), distinct from PyTorch's FSDP2.

**Key design statement from the PR body:**
> "Meta-parameter materialization is intentionally split out to the follow-up draft PR at #5369."

**Commit history (17 commits, May 22 - June 17):**
- May 22: WIP: add experimental minimal FSDP path
- May 23: Refine minimal experimental FSDP path (DBuffer release/reallocate, fully_allgather_into)
- May 24: Refine experimental FSDP buffer APIs (out= support, preallocated model/gradient buffers)
- May 25: Refine experimental FSDP gradient contract (ordered parameter tuples, grad_dtype)
- May 25: Require matching FSDP main grad dtype
- May 25: Reuse FSDP model weights for matching main weights (aliasing when same dtype/placements)
- May 26: Sync FSDP model weights before unshard (next-forward optimizer visibility)
- Jun 1: Preserve autograd grad dtype in minimal FSDP
- Jun 1: Compare minimal FSDP loss curve with baseline
- Jun 10: Adapt minimal FSDP to split DBuffer API
- Jun 11: Split minimal FSDP runtime modules
- Jun 11: Split minimal FSDP module mixin
- Jun 12: Remove experimental FSDP meta parameter support (moved to #5369)
- Jun 17: Clarify FSDP version counter preservation
- Jun 17: Rename experimental FSDP runtime types
- Jun 17: Document main_grad allocation lifetime
- Jun 17: Document post-backward reshard storage choice

### 1.2 Why Megatron Built Its Own FSDP

```
★★★★★★★★★ CRITICAL DESIGN CHOICE: NOT using PyTorch FSDP2

Megatron's unique constraints justify custom FSDP:
  1. Tensor-parallel sharding metadata lives on nn.Parameter (TE integration)
     → PyTorch FSDP2's FlatParamHandle doesn't understand TP sharding
     → DBuffer comment: "FsdpParameterGroup should extend returned DTensors
       with tensor-parallel mesh axes because TP sharding metadata lives on
       nn.Parameter in Mcore/TransformerEngine" (dbuffer.py line 71)
  2. DBuffer primitives give lower-level control than FSDP2's DTensor-based API
     → DBuffer manages a group of logical tensors in one local storage tensor
     → FSDP2's FlatParamHandle manages one flat parameter per module group
  3. Per-module granularity: each module = one FSDP unit, no nested wrapping
     → PyTorch FSDP2 requires nested wrapping for per-layer sharding
  4. Custom storage lifecycle: release/reallocate without replacing Storage object
     → Preserves autograd aliases during storage resize (version counter preservation)
     → FSDP2 uses implicit Python GC + separate FlatParamHandle

★★★★★★★★★ The existing Megatron FSDP (megatron_fsdp.py) is ALSO custom:
  → MegatronFSDP class → TrainingState enum → FORWARD/PRE_BACKWARD/POST_BACKWARD/IDLE
  → ParamAndGradBuffer → AllGatherPipeline → GradReducePipeline
  → BucketingPolicy → PrefetchOrder
  → ShardingStrategy IntEnum: NO_SHARD(0), OPTIM(1), OPTIM_GRADS(2), OPTIM_GRADS_PARAMS(3)
  → fully_shard_model() in existing fully_shard.py → wraps whole model, not per-module
  → This existing code is 500+ lines and tightly coupled to Megatron's DDP config

★★★★★★★★★ MFSDPv2 is a CLEAN REIMPLEMENTATION:
  → 993 additions → minimal, experimental, clean
  → No dependency on DistributedDataParallelConfig
  → No ParamAndGradBuffer, AllGatherPipeline, GradReducePipeline
  → No BucketingPolicy, PrefetchOrder
  → Uses only DBuffer primitives + DeviceMesh + DTensor
  → Located in experimental/ directory → clearly labeled as experimental
```

---

## 2. Source-Level Architecture Analysis

### 2.1 fully_shard.py (61 lines) — Entry Point

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/fully_shard.py

★★★★★★★★★ fully_shard() function signature (line 27-42):

  def fully_shard(
      module: nn.Module,
      mesh: DeviceMesh,
      placements: Placements,
      mixed_precision_policy: MixedPrecisionPolicy | None = None,
  ) -> None:

Key parameters:
  → module: Module whose currently UNOWNED parameters become this FSDP unit
  → mesh: Device mesh used for sharding
  → placements: Per-mesh-axis parameter, gradient, optimizer placements
  → mixed_precision_policy: Optional → defaults to MixedPrecisionPolicy()
     → MixedPrecisionPolicy defaults: main_params_dtype=FP32, main_grads_dtype=None (=compute dtype)

★★★★★★★★★ _attach_mixin() — Dynamic class creation (line 52-58):

  def _attach_mixin(module: nn.Module) -> None:
      if isinstance(module, FsdpModule):
          return
      module_cls = module.__class__
      fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
      module.__class__ = fsdp_cls

★★★★★★★★★ CRITICAL: This creates a NEW class dynamically!
  → ExperimentalFsdpLinear → inherits from (FsdpModule, Linear)
  → FsdpModule methods override Linear's where they conflict
  → MRO: FsdpModule → Linear → nn.Module
  → Original class preserved for rollback on exception
  → Guard: raises ValueError if module already managed by FSDP (double-sharding prevention)

★★★★★★★★★ Comparison with PyTorch FSDP2:
  → PyTorch FSDP2: _FSDPParamGroup wraps module, replaces parameters with FlatParameter
  → Megatron MFSDPv2: FsdpModule mixin injected into module's class hierarchy
  → Key difference: MFSDPv2 PRESERVES original module class → forward() still works
  → FSDP2 replaces .data on parameters → MFSDPv2 swaps entire Parameter objects (sharded ↔ unsharded)
```

### 2.2 module.py (161 lines) — FsdpModule Mixin Lifecycle

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py

★★★★★★★★★ FsdpModule.__init__() (line 39-56):

  def __init__(self, mesh, placements, mixed_precision_policy) -> None:
      owned_parameters = _collect_owned_parameters(self)  # dict[str, nn.Parameter]
      axis_indices = tuple(_axis_index(mesh, axis) for axis in placements.dp_axes)
      assert axis_indices == tuple(range(mesh.ndim))
      parameter_groups = [
          FsdpParameterGroup(
              owning_module=self,
              parameters=group_parameters,
              mesh=mesh,
              placements=placements,
              mixed_precision_policy=mixed_precision_policy,
          )
          for group_parameters in _group_parameters(owned_parameters)
      ]
      self._parameter_groups = tuple(parameter_groups)
      self._ready_grad_parameters = set()
      self._num_training_parameters = sum(
          len(group.sharded_parameters) for group in self._parameter_groups if group.requires_grad
      )
      self._register_hooks()

★★★★★★★★★ CRITICAL: axis_indices must match every mesh axis in mesh order!
  → "FSDP requires dp_axes to match every mesh axis in mesh order for now."
  → This means HSDP/HFSDP (hybrid sharded DP) is NOT yet supported
  → Same limitation noted in parameter_group.py: "FSDP temporarily requires
    main_grad and main_weight to have the same placements until HSDP/HFSDP
    support is implemented"

★★★★★★★★★ Four lifecycle hooks (line 58-98):

  _register_hooks():
    → register_forward_pre_hook → pre_forward()
    → register_forward_hook → post_forward()
    → register_full_backward_pre_hook → pre_backward()
    → register_post_accumulate_grad_hook per parameter → completion-based gradient tracking

  pre_forward():
    → _ready_grad_parameters.clear()
    → For each group: sync_model_weight_from_main_weight() + unshard_parameters()

  post_forward():
    → For each group: reshard_parameters() (release storage)

  pre_backward():
    → For each group: unshard_parameters()

  post_backward():
    → For each group: reduce_gradients() (if requires_grad) + reshard_parameters()
    → _ready_grad_parameters.clear()

★★★★★★★★★ Completion-based gradient tracking (line 82-98):

  _make_grad_hook(parameter):
    → Adds parameter to _ready_grad_parameters set
    → When len(_ready_grad_parameters) == _num_training_parameters → post_backward()

★★★★★★★★★ CRITICAL DIFFERENCE from PyTorch FSDP2:
  → PyTorch FSDP2: post_backward hook fires on module boundary
  → Megatron MFSDPv2: fires when ALL parameters have accumulated their grad
  → Why: module full-backward hooks can fire BEFORE all grads ready (when module inputs don't require grad)
  → This is MORE ROBUST for complex models (MoE, LoRA, GDN) where not all backward paths are guaranteed

★★★★★★★★★ _collect_owned_parameters() (line 104-127):

  def _collect_owned_parameters(root_module) -> dict[str, nn.Parameter]:
      → Walks module tree, collects direct parameters (named_parameters(recurse=False))
      → SKIPS child modules that are already FsdpModule (nested ownership)
      → Raises ValueError if parameter already in a parameter group (double-ownership guard)
      → Raises ValueError if no unowned parameters found

★★★★★★★★★ _group_parameters() (line 129-137):

  def _group_parameters(parameters) -> list[dict[str, nn.Parameter]]:
      → Groups by (dtype, requires_grad) key
      → Each group becomes one FsdpParameterGroup
      → ★★★★★★★★ FROZEN parameters get their own group → main_grad=None → no gradient buffer!
```

### 2.3 parameter_group.py (271 lines) — Core State Management

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py

★★★★★★★★★ FsdpParameterGroup.__init__() (line 41-155):

Three DBuffer instances per group:

  1. main_weight (line 89-93):
     → DBuffer.distribute_tensors(params cast to main_params_dtype, mesh, optimizer placements)
     → DEFAULT: FP32, Flat sharding → optimizer's sharded copy of weights

  2. model_weight (line 95-106):
     → If main_weight.dtype == model_weight.dtype AND placements match → model_weight = main_weight (ALIAS!)
     → Else: separate DBuffer with parameter placements, compute dtype
     → ★★★★★★★★ CRITICAL: aliasing means ZERO overhead when compute dtype = optimizer dtype!
     → DEFAULT: BF16, Flat sharding → compute weights

  3. _unsharded_model_weight (line 94, line 97-105):
     → DBuffer(mesh, placements=[Replicate()] * mesh.ndim, tensor_shapes, dtype, device)
     → TEMPORARY replicated buffer → allocated on unshard, released on reshard
     → ★★★★★★★★ Peak memory optimization: this buffer only exists during forward/backward compute!

  4. main_grad (line 108-134):
     → Only allocated if requires_grad=True
     → ★★★★★★★★ FROZEN parameters: main_grad = None → no gradient buffer allocated!
     → grad_dtype = mixed_precision_policy.main_grads_dtype or self.dtype
     → Persistent allocation comment: "For micro-batch size 1, this allocation could be
       delayed until post_backward and then eagerly deallocated right after optimizer.step()...
       This version keeps the simpler persistent buffer."
     → ★★★★★★★★ Future optimization: lazy main_grad allocation for micro-batch=1

★★★★★★★★★ Parameter management (line 136-155):

  sharded_parameters and unsharded_parameters:
    → parameter.data = _unsharded_model_weight.get_local_tensor(index) → unsharded view
    → sharded_parameter = nn.Parameter(main_weight.get_dtensor(index)) → DTensor-backed
    → sharded_parameter.grad_dtype = main_grad_dtype → optimizer sees correct dtype
    → _CONTAINING_PARAMETER_GROUP_ATTR = "_mfsdp_parameter_group" → marks ownership
    → _switch_to_sharded_parameters() → installs DTensor params on module
    → _unsharded_model_weight.release_storage() → frees GPU allocation after init

★★★★★★★★★ sync_model_weight_from_main_weight() (line 164-168):

  def sync_model_weight_from_main_weight(self) -> None:
      if self.main_weight is self.model_weight:
          return  # ALIAS case → no sync needed!
      self.main_weight.cast(self.model_weight.dtype).redistribute(
          self.model_weight.placements, out=self.model_weight
      )

★★★★★★★★★ CRITICAL: Two-step sync for mixed precision:
  1. Cast: FP32 main_weight → BF16 (or compute dtype)
  2. Redistribute: Flat(optimizer placements) → Flat(parameter placements)
     → When same dtype + same placements → NO sync needed (aliasing optimization)
  → out=self.model_weight → in-place redistribution → no extra allocation

★★★★★★★★★ unshard_parameters() (line 170-183):

  def unshard_parameters(self) -> None:
      self._unsharded_model_weight.reallocate_storage()
      with torch.autograd._unsafe_preserve_version_counter(
          self._unsharded_model_weight.local_buffer
      ):
          self.model_weight.redistribute(
              self._unsharded_model_weight.placements, out=self._unsharded_model_weight
          )
      self._switch_to_unsharded_parameters()

★★★★★★★★★ CRITICAL: _unsafe_preserve_version_counter!
  → Autograd records tensor version counter when saving for backward
  → In-place redistribution (out=) increments version counter EVEN under no_grad
  → Without preserving: backward fails with "modified by an inplace operation"
  → This is a KNOWN PAIN POINT in FSDP implementations → PyTorch FSDP2 has similar version counter issues
  → ★★★★★★★★ This is the SAME pattern as verl #6699 (detach fix) — autograd graph pinning!

★★★★★★★★★ reshard_parameters() (line 185-196):

  def reshard_parameters(self) -> None:
      self._switch_to_sharded_parameters()
      self._unsharded_model_weight.release_storage()

★★★★★★★★★ Storage lifecycle pattern (RTX 4090 critical):
  Forward:  unshard → allocate → redistribute into → compute → reshard → release(0)
  Backward: unshard → allocate → redistribute into → compute → reduce → reshard → release(0)

  → release_storage() → resize(0) → frees GPU allocation while preserving Storage object
  → reallocate_storage() → resize(N) → restores allocation without replacing Storage
  → ★★★★★★★★ This is the SAME pattern as verl per-unit LoRA summon (#6512)!
  → verl: summon LoRA adapter → compute → release → next layer
  → MFSDPv2: unshard params → compute → release → next layer
  → ★★★★★★★★ CONVERGENCE: FSDP storage lifecycle = LoRA weight lifecycle!

★★★★★★★★★ reduce_gradients() (line 198-246):

  def reduce_gradients(self) -> None:
      grads = [parameter.grad for each unsharded parameter]
      partial_grad = DBuffer.distribute_tensors(grads, mesh, [Partial(ReduceOp.AVG)] * mesh.ndim)

      # Key optimization: reduce into main_grad if possible
      can_reduce_into_main_grad = (
          not has_sharded_grads and partial_grad.dtype == self.main_grad.dtype
      )
      if can_reduce_into_main_grad:
          partial_grad.redistribute(self.main_grad.placements, out=self.main_grad)
      else:
          reduced_grad = partial_grad.redistribute(self.main_grad.placements)
          if has_sharded_grads:
              self.main_grad.local_buffer.add_(reduced_grad.local_buffer)  # accumulate
          else:
              self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)  # first reduction

★★★★★★★★★ Gradient accumulation across backward calls:
  → zero_grad(set_to_none=True) → clears sharded grads → next backward reduces into main_grad directly
  → zero_grad(set_to_none=False) → leaves sharded grads → next backward ACCUMULATES into main_grad
  → ★★★★★★★★ This is the CORRECT accumulation pattern for GRPO micro-batching!
  → can_reduce_into_main_grad → in-place reduction → no extra allocation

★★★★★★★★★ _get_parameter_owner() (line 248-254):

  def _get_parameter_owner(module, name):
      module_name, separator, parameter_name = name.rpartition(".")
      owner = module.get_submodule(module_name) if separator else module
      return owner, parameter_name

  → Resolves root-module-relative parameter FQN to direct owner module
  → Enables nested module parameter management (bias on root, weight on inner)
```

### 2.4 DBuffer Primitives (dbuffer.py) — Foundation

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/dbuffer.py

★★★★★★★★★ DBuffer class (line 65-):

  class DBuffer:
      """A distributed buffer holding a group of logical tensors.

      DBuffer is analogous to DTensor, but manages a group of logical tensors
      in one local storage tensor. It stores enough metadata to return per-tensor
      views, redistribute the buffer across mesh axes, and materialize per-tensor
      DTensors for optimizer state or distributed checkpointing.
      """

★★★★★★★★★ CRITICAL: DBuffer ≠ DTensor:
  → DTensor: one logical tensor, one local shard → per-tensor
  → DBuffer: GROUP of logical tensors, one flat local_buffer → per-group
  → Why group: optimizer operates on grouped parameters → single buffer for all-gather/reduce-scatter
  → This matches Megatron's existing ParamAndGradBuffer pattern → but much cleaner

★★★★★★★★★ DBuffer fields:
  mesh: DeviceMesh → owns only DP sub-mesh (TP axes added by FsdpParameterGroup)
  placements: tuple[Placement, ...] → per-mesh-axis (Replicate, Partial, Flat)
  layout: GlobalLayout → global tensor offsets + padding
  offset: int → this rank's owned range start
  local_buffer: torch.Tensor → 1D flat contiguous storage

★★★★★★★★★ New methods added in #5387:

  reallocate_storage() (line 117-119):
    → _resize_storage(self.local_buffer.numel()) → restores full allocation
    → Used before: all-gather (unshard), compute

  release_storage() (line 121-127):
    → _resize_storage(0) → frees allocation WITHOUT replacing Storage object
    → Comment: "Autograd may save views that share this Storage object. Resizing
      the existing Storage releases the allocation while preserving those aliases
      for a later reallocate_storage()."
    → ★★★★★★★★ CRITICAL: this is the KEY memory optimization → peak = compute-only

  cast(dtype) (line 269-283):
    → Returns self if dtype matches (identity optimization)
    → Else: new DBuffer with same layout/placements in target dtype
    → Used for: FP32 main_weight → BF16 model_weight conversion

★★★★★★★★★ DBuffer redistribute() — Collective operation dispatcher:

  Supported transitions:
    Flat → Replicate: allgather() (dim-0 shard → full tensor)
    Partial → Replicate: allreduce() (unreduced → averaged)
    Partial → Flat: reduce_scatter() (unreduced → sharded)
    Replicate → Flat: scatter() (full → local chunk, NO communication)

  ★★★★★★★★ CRITICAL: scatter() is LOCAL ONLY!
    → No communication needed → just narrow the local buffer
    → Used for: splitting replicated buffer into per-rank shard

★★★★★★★★★ DBuffer.distribute_tensors() (line 197-220):

  @classmethod
  def distribute_tensors(cls, tensors, mesh, placements) -> "DBuffer":
      tensors = tuple(tensor.detach().contiguous() for tensor in tensors)
      → ★★★★★★★★ DETACH + CONTIGUIZE → strips autograd graph + ensures contiguous layout
      → Same pattern as PyTorch FSDP2's _flat_param_from_tensors
      → Only owned ranges initialized → padding gaps remain unspecified (not observable)
```

### 2.5 Placement System (placement.py) — Sharding Configuration

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/placement.py

★★★★★★★★★ Placement hierarchy (mirrors DTensor):

  Placement → base class
    Replicate → replicated local buffer (matches DTensor Replicate)
    Partial(reduce_op=ReduceOp.SUM) → unreduced replicated (matches DTensor Partial)
    Flat → dim-0 shard per-unit (matches DTensor Shard(0), but custom name)

★★★★★★★★★ MeshAxis = int | str (line 41):
  → Can reference mesh axis by index or by name
  → Enables named mesh dimensions (e.g., "dp", "tp", "cp")

★★★★★★★★★ Placements dataclass (line 66-82):

  @dataclasses.dataclass(frozen=True)
  class Placements:
      dp_axes: list[MeshAxis]
      parameter: list[Placement]     # model weight placements
      gradient: list[Placement]      # gradient buffer placements
      optimizer: list[Placement]     # main weight placements

★★★★★★★★★ Validation: all placement lists must have same length as dp_axes
  → Each placement list corresponds to each mesh axis
  → Typical: Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])
  → This means: single DP mesh axis, all buffers Flat (dim-0 shard)

★★★★★★★★★ Supported redistribution transitions (from placement.py docstring):

  Source         Destination    DBuffer operation
  sharded        Replicate      allgather()
  Partial        sharded        reduce_scatter()
  Partial        Replicate      allreduce()
  Replicate      sharded        scatter() (local only)

★★★★★★★★★ Transition table for MFSDPv2 lifecycle:

  FORWARD (unshard):
    model_weight: Flat → Replicate → allgather → full compute weights
  FORWARD (reshard):
    _unsharded_model_weight: Replicate → release_storage(0) → frees GPU
  BACKWARD (reduce_gradients):
    local grads → Partial(ReduceOp.AVG) → Flat → reduce_scatter into main_grad
  BACKWARD (reshard):
    _unsharded_model_weight: release_storage(0) again
```

### 2.6 MixedPrecisionPolicy — Precision Configuration

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/mixed_precision.py

★★★★★★★★★ MixedPrecisionPolicy dataclass:

  @dataclass(frozen=True)
  class MixedPrecisionPolicy:
      main_params_dtype: Optional[torch.dtype] = torch.float32
      → Optimizer weight buffer dtype → FP32 by default
      → If None: model weight buffer takes role (aliasing optimization)

      main_grads_dtype: Optional[torch.dtype] = None
      → Gradient buffer dtype → defaults to compute dtype (self.dtype)
      → For GRPO: main_grads_dtype=BF16 → matches compute → allows reduce_into_main_grad

      grad_comm_dtype: Optional[torch.dtype] = None
      → Gradient communication dtype → NCCL optimization
      → If None: uses main_grads_dtype → no extra allocation
      → NCCL UBR v2.27+: can enable mixed-precision comm+accumulation

★★★★★★★★★ RTX 4096 optimal policy:
  MixedPrecisionPolicy(
      main_params_dtype=torch.float32,  # FP32 optimizer weights (standard)
      main_grads_dtype=torch.bfloat16,  # BF16 gradients (matches compute)
  )
  → model_weight = separate BF16 buffer (alias=False since dtype differs)
  → main_grad = BF16 (matches compute → can_reduce_into_main_grad=True)
  → ★★★★★★★★ For PURE BF16 training (no FP32 optimizer):
    MixedPrecisionPolicy(main_params_dtype=torch.bfloat16)
    → main_weight aliases model_weight → ZERO extra buffer!
```

### 2.7 GlobalLayout — Tensor Packing

```
Source: megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/layout.py

★★★★★★★★★ GlobalLayout class:

  @dataclasses.dataclass(frozen=True)
  class GlobalLayout:
      tensor_shapes: tuple[torch.Size, ...]
      tensor_to_offset: tuple[int, ...]
      size: int

★★★★★★★★★ Layout algorithm (GlobalLayout.build):
  → chunk_size = LCM of all tensor row sizes (shape[1:].numel())
  → Regular tensors anchor the layout (tensors >= chunk_size)
  → Fragment tensors fill padding gaps left by regular tensors
  → Total size padded to chunk_size * dp_size → equal-size rank shards
  → ★★★★★★★★ Row-aligned packing → every DP shard owns full rows → no partial row splits

★★★★★★★★★ Example layout from docstring:
  Shapes: P0=(2,6), P1=(4,4), P2=(4,4), P3=(1,2), P4=(1,6)
  chunk_size = LCM(6,4,4,2,6) = 12
  5-rank DP layout:

  rank 0 [0,12):  | P0 row 0 | P0 row 1 |
  rank 1 [12,24): | P1 row 0 | P1 row 1 | P1 row 2 |
  rank 2 [24,36): | P1 row 3 | P3 | gap | P2 row 0 |
  rank 3 [36,48): | P2 row 1 | P2 row 2 | P2 row 3 |
  rank 4 [48,60): | P4 | pad |

★★★★★★★★★ CRITICAL: This is a REIMPLEMENTATION of param_and_grad_buffer.build_data_parallel_buffer_index
  → "This is a DBuffer-specific reimplementation of
    param_and_grad_buffer.build_data_parallel_buffer_index. It keeps only
    the global offset construction and final padding..."
  → Simplified: no bucketing, no prefetch order, no pipeline management
  → Compatible with Flat, TensorAtomic, BlockAtomic (future sharding modes)
```

---

## 3. Test Coverage Analysis

```
Source: tests/unit_tests/distributed/megatron_fsdp/test_experimental_fully_shard.py (+353 lines)

★★★★★★★★★ Test classes:

  TinyModel: fc1(8→16) + relu + fc2(16→4) → two separately shardable units
  NestedModel: bias on root + inner.weight → tests nested ownership
  NonLeafViewModel: weight with view_as → tests autograd version counter preservation
  SaveNonLeafWeightView: autograd Function saving non-leaf weight view

★★★★★★★★★ 9 test cases (27 passed, 6 skipped in CI):

  1. test_fully_shard_losses_match_baseline
     → 5-step SGD training → loss parity with single-rank baseline
     → ★★★★★★★★ Validates numerical correctness of entire FSDP lifecycle

  2. test_nested_fully_shard_excludes_child_owned_parameters
     → inner module sharded first → outer module owns only bias
     → inner_names=["weight"], outer_names=["bias"]
     → ★★★★★★★★ Validates nested ownership: parent skips child-owned params

  3. test_frozen_parameter_group_does_not_allocate_main_grad
     → requires_grad=False → main_grad=None → no gradient buffer
     → ★★★★★★★★ Validates memory optimization for frozen parameters

  4. test_backward_averages_across_dp_and_accumulates_across_calls
     → Two backward calls → averaged + accumulated
     → expected = world_size + 1 (rank+1 averaged across ranks, then accumulated)
     → ★★★★★★★★ Validates: ReduceOp.AVG for inter-rank, SUM for accumulation

  5. test_next_forward_uses_optimizer_updated_weights
     → BF16 model + FP32 main weights → SGD(foreach=False)
     → Second loss ≠ first loss → optimizer step visible to next forward
     → ★★★★★★★★ Validates: sync_model_weight_from_main_weight() works after optimizer.step()

  6. test_cpu_initialized_parameters_shard_to_mesh_device
     → CPU-initialized model → sharded to GPU → values preserved
     → ★★★★★★★★ Validates: CPU-to-device transfer during sharding

  7. test_non_leaf_parameter_view_survives_storage_resize
     → Non-leaf view saved for backward → survives release/reallocate
     → After forward: storage nbytes == 0 (released)
     → After backward: main_grad allocated, storage still nbytes == 0
     → ★★★★★★★★ Validates: _unsafe_preserve_version_counter + Storage resize pattern

  8. test_fully_shard_reduces_peak_training_memory
     → 16-layer Sequential(Linear(1024,1024)) → per-layer FSDP
     → baseline_peak vs sharded_peak → assert sharded_peak < baseline_peak
     → ★★★★★★★★ Validates: peak memory reduction from per-layer unshard/reshard

★★★★★★★★★ DBuffer tests (test_dbuffer.py, +77 lines):

  test_cast_to_same_dtype_returns_self → identity optimization
  test_cast_preserves_layout_and_casts_values → layout preservation
  test_release_and_reallocate_storage_preserves_buffer_views → Storage alias preservation
  test_distribute_tensors_detaches_and_contiguizes_inputs → detach + contiguous
```

---

## 4. DBuffer vs PyTorch FSDP2 FlatParamHandle — Deep Comparison

```
★★★★★★★★★ Architecture comparison:

| Feature                | MFSDPv2 DBuffer              | PyTorch FSDP2 FlatParamHandle |
|------------------------|-------------------------------|-------------------------------|
| Grouping unit          | DBuffer (group of tensors)    | FlatParameter (one flat param) |
| Storage                | local_buffer (1D flat)        | _flat_param (1D flat)         |
| Sharding metadata      | Custom Placement (Flat/Partial/Replicate) | DTensor Shard/Partial/Replicate |
| Layout                 | GlobalLayout (row-aligned, chunk_size LCM) | FlatParamHandle._param_infos |
| Unshard                | reallocate_storage + redistribute(allgather) | _unshard_flat_param (all_gather) |
| Reshard                | _switch_to_sharded + release_storage(0) | _shard_flat_param (reduce_scatter or free) |
| Mixed precision        | cast + redistribute (out=)    | _cast_flat_param (copy_ + cast) |
| Gradient reduction     | Partial(AVG) → Flat (reduce_scatter) | _reduce_scatter_flat_grad |
| Gradient accumulation  | add_ into main_grad or copy_  | _accum_grad + reduce_scatter |
| Version counter        | _unsafe_preserve_version_counter | _check_training_state (safety check) |
| Storage lifecycle      | explicit release/reallocate   | implicit Python GC + swap |
| TP integration         | get_dtensor → extend with TP axes | None (separate TP + FSDP layers) |
| Frozen params          | main_grad=None → skip gradient | Not optimized in FSDP2 |
| Module attachment       | Dynamic mixin (FsdpModule)    | _FSDPParamGroup (static wrapper) |
| Meta params            | #5369 follow-up (+2398)       | native DTensor support |
| Partial offload         | NOT yet                       | #187620 PartialOffloadPolicy |

★★★★★★★★★ KEY DIFFERENCES:

  1. DBuffer = group-of-tensors vs FlatParam = one-flat-param
     → DBuffer packs multiple parameters into one buffer → single collective call
     → FlatParamHandle also packs, but via _param_infos index → less explicit layout control
     → GlobalLayout's chunk_size LCM → row-aligned → enables clean dim-0 sharding

  2. Storage lifecycle: explicit release/reallocate vs implicit
     → DBuffer: resize(0) → release → resize(N) → reallocate → SAME Storage object
     → FlatParamHandle: allocates new tensor or uses preallocated swap buffer
     → ★★★★★★★★ DBuffer approach preserves autograd aliases → MORE CORRECT

  3. Version counter handling: _unsafe_preserve vs _check_training_state
     → DBuffer: explicitly preserves version counter → allows storage resize during forward
     → FSDP2: checks training state → asserts not in FORWARD when unshard is called
     → ★★★★★★★★ DBuffer approach is MORE FLEXIBLE for complex autograd graphs

  4. TP integration: native vs separate
     → DBuffer: "FsdpParameterGroup should extend returned DTensors with TP mesh axes"
     → FSDP2: TP is a separate layer → FSDP2 doesn't understand TP sharding
     → ★★★★★★★★ This is the PRIMARY justification for Megatron's custom FSDP
```

---

## 5. Cross-Framework Connections

### 5.1 verl FSDP2 Integration — #6512 per-unit LoRA summon

```
★★★★★★★★★ CONVERGENCE: FSDP unshard/reshard = LoRA summon/release

verl #6512 (MERGED June 18):
  → layered_summon_lora_params → iterate every FSDP unit → summon LoRA → compute → release
  → Peak memory: 60→6-8 GiB → per-unit summon avoids full model unshard
  → Dynamic FSDP unit discovery → replaces 8 hard-coded prefixes

MFSDPv2 #5387:
  → unshard_parameters → reallocate_storage + allgather → compute → reshard → release_storage
  → Peak memory: sharded_peak < baseline_peak (validated in test)

★★★★★★★★★ Pattern convergence:

  verl LoRA lifecycle:
    summon(layer) → allocate LoRA adapter → compute → release → next layer
  MFSDPv2 FSDP lifecycle:
    unshard(layer) → allocate full params → compute → reshard → release → next layer

★★★★★★★★★ Both use the SAME storage optimization:
  → Allocate only during compute → release immediately after
  → Peak memory = only one layer's full parameters at any time
  → ★★★★★★★★ This pattern is THE key to RTX 4090 GRPO training!

★★★★★★★★★ Implication for verl:
  → If verl adopts MFSDPv2 backend → summon pattern works IDENTICALLY
  → But: verl currently uses PyTorch FSDP2 → MFSDPv2 would require Megatron dependency
  → For RTX 4090: PyTorch FSDP2 (verl) remains #1 → MFSDPv2 is Megatron-only
```

### 5.2 verl #6699 — detach fix for autograd graph pinning

```
★★★★★★★★★ SAME ROOT CAUSE: autograd pinning of intermediate tensors

verl #6699 (MERGED June 18):
  → model_output (log_probs/entropy) still attached to autograd graph
  → forward_backward_batch holds per-micro-batch outputs in output_lst
  → Retained graph pins parameters → memory leak → OOM
  → Fix: detach model_output and loss metrics

MFSDPv2 #5387:
  → _unsafe_preserve_version_counter → prevents "modified by inplace operation"
  → release_storage(0) → preserves Storage object for autograd aliases
  → Both handle the same underlying issue: autograd's lifetime management

★★★★★★★★★ Key difference:
  → verl #6699: DETACH outputs → break autograd graph → release memory
  → MFSDPv2: PRESERVE version counter → keep autograd happy → but release STORAGE
  → ★★★★★★★★ Both solve memory issues → but MFSDPv2's approach is MORE subtle
  → MFSDPv2: don't break graph → just release backing storage → graph can still "exist" but with empty storage
```

### 5.3 PyTorch #187620 — PartialOffloadPolicy

```
★★★★★★★★★ RTX 4090 CRITICAL CONNECTION:

PyTorch #187620 (OPEN, draft):
  → PartialOffloadPolicy → fractional CPU parameter offload
  → Selects which parameters to offload based on ratio
  → For dp=1: shard=identity → resident=full param → OOM for >8B models
  → ★★★★★★★★ dp=1 NOT viable with PartialOffloadPolicy → ONLY helps dp>=2
  → CPUOffloadPolicy(pin_memory=True) default=TRUE → remains ONLY dp=1 path

MFSDPv2 #5387:
  → NO partial offload support yet
  → Placements only support: Flat, Partial, Replicate → all GPU
  → ★★★★★★★★ Future: could add CPU placement → analogous to CPUOffloadPolicy
  → But: DBuffer.release_storage already reduces peak memory → different approach

★★★★★★★★★ RTX 4090 decision matrix:

| Approach              | dp=1 single GPU | Peak Memory   | Viability |
|-----------------------|------------------|---------------|-----------|
| PyTorch FSDP2 (verl)  | identity shard   | full model    | #1 for dp=1 |
| CPUOffloadPolicy      | pin_memory=True  | offloaded     | dp=1 only |
| PartialOffloadPolicy  | NOT viable dp=1  | OOM for >8B  | dp>=2 only |
| MFSDPv2 (Megatron)    | identity shard   | full model    | dp=1 same |
| ZeRO-2 + CPU_Adam     | optimizer only   | CPU optimizer | #2 for dp=1 |
```

### 5.4 DeepSpeed ZeRO-2 vs MFSDPv2

```
★★★★★★★★★ DeepSpeed ZeRO-2 comparison:

| Feature                | DeepSpeed ZeRO-2            | MFSDPv2                      |
|------------------------|------------------------------|-------------------------------|
| Optimizer sharding     | CPU_Adam optimizer           | DBuffer Flat (GPU shard)      |
| Parameter sharding     | NO (ZeRO-2 = optimizer only) | YES (Flat sharding)           |
| Gradient sharding      | NO (ZeRO-2 = optimizer only) | YES (Flat sharding)           |
| CPU offload            | CPU_Adam (optimizer → CPU)   | NOT yet supported             |
| overlap_comm           | YES (but #8061 NaN risk!)    | NOT yet implemented           |
| Gradient reduction     | reduce_scatter               | reduce_scatter (same)         |
| Per-module granularity | bucket-based                 | per-module FsdpModule         |
| RTX 4090 dp=1          | ZeRO-2 = identity → CPU_Adam only benefit | Flat = identity → no benefit |

★★★★★★★★★ RTX 4090 ranking:
  → ZeRO-2 + CPU_Adam: optimizer state on CPU → ~4x optimizer memory savings
  → MFSDPv2: optimizer state sharded → but dp=1 → identity → NO savings
  → ★★★★★★★★ For dp=1: ZeRO-2's CPU offload > MFSDPv2's GPU sharding
  → But: MFSDPv2 could add CPU placement in future → could match ZeRO-2
```

---

## 6. Related Megatron PR/Issue Analysis

### 6.1 #5395 — skip_grad_norm_clip (+15/-1, OPEN)

```
★★★★★★★★★ CRITICAL for Muon optimizer on RTX 4090:

  Title: "fix(optimizer): skip grad-norm clipping for orthogonalizing (Muon) optimizers"
  Author: yuchenwang3
  Background: ms-swift + Megatron-Core SFT of Qwen3.5-35B-A3B (GatedDeltaNet)
  → ChainedOptimizer computes single global grad_norm → applies clip to ALL sub-optimizers
  → Muon: Newton-Schulz orthogonalization is scale-invariant → clipping meaningless + harmful
  → When grad_norm ~5e7 → clip coefficient ~2e-8 → per-matrix gradient → ~0 → training stalls

★★★★★★★★★ Connection to MFSDPv2:
  → MFSDPv2 gradient reduction: ReduceOp.AVG → Partial → Flat → main_grad
  → If global grad-norm clipping applied → same stall risk for Muon
  → #5395 adds skip_grad_norm_clip attribute → +15/-1 lines
  → ★★★★★★★★ MUST be merged before Muon + MFSDPv2 combination
  → Same pattern as DeepSpeed #8068 + #7776 → optimizer-agnostic bug
```

### 6.2 #5396 — GDN L2-norm fold (+7/-4, OPEN draft)

```
★★★★★★★★★ CRITICAL for RTX 4090 memory savings:

  Title: "perf(gated_delta_net): fold q/k L2-norm into the gated_delta_rule kernel"
  Author: yuchenwang3
  Background: GatedDeltaNet L2-normalizes q/k before kernel → materializes [B,T,2*Hk,128] activation
  → This activation must be kept for backward → peak memory penalty
  → Folding L2-norm INTO kernel → eliminates materialization → 24 GiB savings at 128K context
  → +7/-4 lines → numerically lossless → RTX 4090 ~384 MiB savings at 4K

★★★★★★★★★ Connection to MFSDPv2:
  → MFSDPv2's release_storage pattern → peak memory = compute-only
  → If GDN materializes [B,T,2*Hk,128] during forward → pinned even after reshard
  → L2-norm fold eliminates materialization → COMPLEMENTS MFSDPv2's memory optimization
  → ★★★★★★★★ Both reduce peak memory → but at different levels:
    MFSDPv2: parameter storage lifecycle → frees params after compute
    #5396: activation elimination → frees activations before compute
  → Combined: both params + activations → maximum memory savings
```

### 6.3 #5384 — DSA/DSv4 Indexer Replay

```
★★★★★★★★★ CRITICAL for RL training with DSV4 models:

  Title: DSA/DSv4 Indexer Replay feature request
  → Same pattern as MoE RouterReplay (#4168) → train/rollout mismatch
  → Needs ~200-300 LOC → creates replay buffer for routing indices

★★★★★★★★★ Connection to MFSDPv2:
  → DSV4 models use sparse attention → dynamic routing per step
  → MFSDPv2's DBuffer manages sharded storage → could store replay indices
  → But: Indexer Replay is about TRAINING/ROLLOUT mismatch → not FSDP storage
  → ★★★★★★★★ MFSDPv2 + Indexer Replay: both needed for DSV4 GRPO training
  → DSV4 SYSTEMATIC INSTABILITY: 8 failures across 3 frameworks → enforce_eager=True MANDATORY
```

### 6.4 #5394 — ChainedOptimizer Muon clipping stalls

```
★★★★★★★★★ ROOT CAUSE for #5395:

  Title: "[BUG] ChainedOptimizer applies global grad-norm clipping to Muon"
  → Global grad_norm across ALL chained sub-optimizers
  → clip_grad_by_total_norm_fp32 applied to EVERY sub-optimizer's parameters
  → For Muon: meaningless (Newton-Schulz is scale-invariant) + harmful (stalls training)
  → ★★★★★★★★ Optimizer-agnostic bug → AdamW ALSO stalls under large global grad_norm
  → Controlled experiments confirm → volunteer stepping up to fix

★★★★★★★★★ Connection to MFSDPv2:
  → MFSDPv2 manages parameter groups per dtype/requires_grad
  → If ChainedOptimizer wraps MFSDPv2 parameters → same clipping bug
  → Need: per-optimizer-group clipping → skip for orthogonalizing groups
  → ★★★★★★★★ MUST coordinate #5395 fix with MFSDPv2 integration
```

---

## 7. RTX 4090 GRPO Impact Analysis

### 7.1 Single-GPU (dp=1) Analysis

```
★★★★★★★★★ RTX 4090 dp=1: MFSDPv2 = identity overhead

  dp=1 → world_size=1 → single GPU
  → Flat placement → dim-0 shard → but shard = full tensor when dp=1
  → Replicate placement → same tensor on all ranks → trivial when dp=1
  → Partial placement → same as local tensor → trivial when dp=1

★★★★★★★★★ What DOES work on dp=1:
  → release_storage/reallocate → memory optimization → works on ANY GPU
  → Per-module FSDP → peak memory = one layer's full parameters → works on dp=1
  → But: with dp=1, all-gather = identity → no communication → no latency
  → ★★★★★★★★ Per-module peak memory savings: SHARED between MFSDPv2 and FSDP2

★★★★★★★★★ What DOES NOT work on dp=1:
  → Optimizer sharding → shard = full optimizer state → no savings
  → Gradient sharding → shard = full gradient → no savings
  → CPU offload → NOT supported yet → ZeRO-2 + CPU_Adam remains only option
```

### 7.2 RTX 4090 MUST DO / MUST NOT for MFSDPv2

```
★★★★★★★★★ MUST DO:

  1. Use per-module fully_shard → peak memory = one layer at a time
     → fully_shard(layer, mesh, placements=_flat_placements()) for each layer
     → NOT fully_shard(model, ...) → whole model = no peak savings

  2. Use MixedPrecisionPolicy(main_params_dtype=torch.float32)
     → FP32 optimizer weights → standard practice
     → BF16 compute weights → cast on unshard → reduce on reshard

  3. Set gradient_clipping=1.0 explicitly
     → #8068/#5394: Muon/AdamW clipping bugs → MUST set explicitly
     → Default 0→1.0 change in DeepSpeed → same risk in MFSDPv2

  4. Use zero_grad(set_to_none=True) for micro-batch=1
     → Clears sharded grads → next backward reduces directly into main_grad
     → Maximum memory efficiency

  5. Validate with test_fully_shard_reduces_peak_training_memory
     → Verify sharded_peak < baseline_peak on actual RTX 4090

★★★★★★★★★ MUST NOT:

  1. MUST NOT use MFSDPv2 as dp=1 optimizer sharding
     → dp=1 → Flat = identity → no optimizer savings
     → Use ZeRO-2 + CPU_Adam for optimizer state savings

  2. MUST NOT use overlap_comm with MFSDPv2
     → #8061: overlap_comm + torch.compile = NaN on single GPU
     → MFSDPv2 doesn't implement overlap_comm yet → but future risk

  3. MUST NOT use MFSDPv2 for models >8B on single RTX 4090 without CPU offload
     → Per-module peak savings help → but full model still OOM
     → Need: CPUOffloadPolicy or MFSDPv2 CPU placement (future)

  4. MUST NOT use HSDP/HFSDP placements yet
     → "FSDP temporarily requires main_grad and main_weight to have the same placements
       until HSDP/HFSDP support is implemented"
     → Only single dp_axes=[0] supported

  5. MUST NOT combine MFSDPv2 with verl directly (yet)
     → verl uses PyTorch FSDP2 → different sharding primitive
     → Need: adapter layer or Megatron backend for verl (#6791 Megatron Lite pathway)
```

### 7.3 RTX 4090 Memory Budget Estimates

```
★★★★★★★★★ MFSDPv2 per-layer peak memory for RTX 4090 (dp=1):

  Model: Qwen2.5-7B (bf16)
  → Per-layer params: ~400M params → ~800 MiB bf16
  → Per-layer optimizer (FP32 AdamW): ~1.6 GiB (momentum + variance + main_weight)
  → Per-layer gradient (bf16): ~800 MiB

  Without MFSDPv2 (whole model):
  → Peak = all layers unsharded → ~7B * 2 bytes = ~14 GiB (params only)
  → + optimizer state → ~7B * 12 bytes = ~84 GiB → OOM!

  With MFSDPv2 (per-layer):
  → Peak = ONE layer unsharded → ~800 MiB (params only)
  → + optimizer for all layers (sharded → but dp=1 → no savings)
  → ★★★★★★★★ Still OOM for optimizer state on dp=1!

  ★★★★★★★★ CONCLUSION: MFSDPv2 alone CANNOT solve >8B model training on RTX 4090
  → Need: ZeRO-2 + CPU_Adam for optimizer → + MFSDPv2 for per-layer peak
  → OR: verl FSDP2 + CPUOffloadPolicy → dp=1 path
  → Best: verl FSDP2 + bypass_mode + per-unit LoRA summon (#6512)
```

---

## 8. Evolution and Future Path

### 8.1 Two-Phase MFSDPv2 Development

```
★★★★★★★★★ Phase 1: #5387 (minimal, +993/-3) → MERGE PENDING
  → DBuffer foundation (#4835, MERGED June 17)
  → fully_shard entry point → FsdpModule mixin → FsdpParameterGroup
  → Forward/backward lifecycle → storage release/reallocate
  → Gradient reduction → mixed precision → version counter preservation
  → 8 test cases → 27 passed, 6 skipped

★★★★★★★★★ Phase 2: #5369 (meta params, +2398/-0) → OPEN draft
  → restore experimental fully_shard handling for meta parameters
  → to_empty() → create uninitialized tensors on correct device
  → reset_parameters() → initialize with model-specific logic
  → Then FsdpParameterGroup construction → shard initialized params
  → ★★★★★★★★ Modular: can be kept, revised, or dropped independently

★★★★★★★★★ Future extensions (not yet in any PR):
  → HSDP/HFSDP support → multiple dp_axes → hybrid sharding
  → CPU placement → analogous to CPUOffloadPolicy → critical for dp=1
  → overlap_comm → pipeline all-gather with compute → but #8061 NaN risk
  → TensorAtomic / BlockAtomic sharding → finer-grained than Flat
  → verl integration → Megatron backend for RL training
```

### 8.2 verl + MFSDPv2 Integration Path

```
★★★★★★★★★ Two pathways:

  Path 1: verl FSDP2 + MFSDPv2 coexistence
  → verl uses PyTorch FSDP2 for training → MFSDPv2 for Megatron-specific models
  → verl #6791 (MERGED June 18): DSv4/GLM5/KimiK2.5 via Megatron Lite
  → Megatron Lite = lightweight Megatron integration → could use MFSDPv2 backend

  Path 2: verl adopts MFSDPv2 as training backend
  → Need: adapter between verl's FSDPEngine and MFSDPv2's FsdpModule
  → verl #6512 per-unit summon → works with MFSDPv2's per-module FSDP
  → But: DBuffer ≠ DTensor → need conversion layer
  → ★★★★★★★★ verl FSDP2 remains #1 for RTX 4090 → MFSDPv2 is future multi-GPU path
```

---

## 9. Source File Index with Key Line Numbers

```
★★★★★★★★★ Key source files (all from PR #5387 branch fsdp/minimal):

  fully_shard.py (+61 lines):
    Line 27-42:  fully_shard() entry point signature
    Line 44-50:  FsdpModule.__init__() call
    Line 52-58:  _attach_mixin() → dynamic class creation

  module.py (+161 lines):
    Line 39-56:  FsdpModule.__init__() → parameter group creation
    Line 58-82:  _register_hooks() → forward/backward lifecycle hooks
    Line 82-98:  _make_grad_hook() → completion-based gradient tracking
    Line 100-103: pre_forward() → sync + unshard
    Line 105-108: post_forward() → reshard
    Line 110-112: pre_backward() → unshard
    Line 114-119: post_backward() → reduce + reshard
    Line 104-127: _collect_owned_parameters() → nested ownership
    Line 129-137: _group_parameters() → dtype/requires_grad grouping

  parameter_group.py (+271 lines):
    Line 41-155: FsdpParameterGroup.__init__() → main_weight + model_weight + main_grad
    Line 89-93:  main_weight = DBuffer.distribute_tensors (FP32, Flat)
    Line 94-106: model_weight = alias or separate DBuffer (BF16)
    Line 95-105: _unsharded_model_weight = Replicate temporary buffer
    Line 108-134: main_grad = persistent gradient buffer (None for frozen!)
    Line 136-155: sharded/unsharded parameter creation
    Line 164-168: sync_model_weight_from_main_weight() → cast + redistribute
    Line 170-183: unshard_parameters() → reallocate + preserve_version_counter + redistribute
    Line 185-196: reshard_parameters() → switch + release_storage
    Line 198-246: reduce_gradients() → Partial(AVG) → Flat → main_grad

  dbuffer.py (+31 lines delta):
    Line 71:  DBuffer comment → TP integration via FsdpParameterGroup
    Line 117-119: reallocate_storage() → resize to full
    Line 121-127: release_storage() → resize(0) → preserve aliases
    Line 129:  _resize_storage() → untyped_storage.resize_
    Line 269-283: cast(dtype) → identity or new DBuffer
    Line 197-220: distribute_tensors() → detach + contiguize

  placement.py (+24 lines):
    Line 41:  MeshAxis = int | str
    Line 66-82: Placements dataclass → dp_axes + parameter + gradient + optimizer
    Line 77-82: __post_init__ → validate placement list lengths

  __init__.py (+15/-2 lines):
    Line 18-30: exports → DBuffer, Flat, FsdpModule, FsdpParameterGroup, fully_shard, Placements

  GlobalLayout (from main, NOT in this PR but referenced):
    Line 1-120: GlobalLayout.build → chunk_size LCM → row-aligned packing
```

---

## 10. Key Findings Summary

```
★★★★★★★★★ #5387: Megatron's own FSDP (MFSDPv2) → DBuffer primitives → per-module fully_shard → +993/-3
★★★★★★★★★ APPROVED by shjwudp → CI passing → blocked by codeowners-approval → merge imminent
★★★★★★★★★ FsdpModule mixin: dynamic class injection → preserves original module → per-module lifecycle
★★★★★★★★★ FsdpParameterGroup: dtype/requires_grad grouping → frozen params skip gradient allocation
★★★★★★★★★ DBuffer storage lifecycle: release/reallocate → SAME pattern as verl per-unit LoRA summon (#6512)
★★★★★★★★★ _unsafe_preserve_version_counter: SAME root cause as verl #6699 detach fix → autograd pinning
★★★★★★★★★ model_weight = main_weight aliasing when same dtype/placements → ZERO overhead optimization
★★★★★★★★★ Completion-based gradient tracking → MORE ROBUST than module-level hooks for complex models
★★★★★★★★★ HSDP/HFSDP NOT yet supported → only single dp_axes → main_grad must match main_weight placements
★★★★★★★★★ CPU offload NOT yet supported → ZeRO-2 + CPU_Adam remains dp=1 optimizer solution
★★★★★★★★★ RTX 4090 dp=1: MFSDPv2 = identity overhead → per-layer peak savings = SAME as FSDP2
★★★★★★★★★ RTX 4090 ranking: verl FSDP2 + bypass #1 > MFSDPv2 future #2 > ZeRO-2 + CPU_Adam #2.5
★★★★★★★★★ #5395 MUST merge before Muon + MFSDPv2 → optimizer-agnostic clipping bug
★★★★★★★★★ #5396 L2-norm fold COMPLEMENTS MFSDPv2 → both reduce peak memory at different levels
★★★★★★★★★ verl #6791 Megatron Lite → MFSDPv2 integration pathway → DSv4/GLM5/KimiK2.5 training
★★★★★★★★★ Convergence: FSDP unshard/reshard = LoRA summon/release = SAME storage lifecycle pattern
★★★★★★★★★ DBuffer ≠ DTensor → group-of-tensors vs per-tensor → enables single collective for multiple params
★★★★★★★★★ GlobalLayout LCM chunk_size → row-aligned packing → enables clean dim-0 sharding + future TensorAtomic/BlockAtomic
```

---

## References

- Megatron #5387: https://github.com/NVIDIA/Megatron-LM/pull/5387 (MFSDPv2 fully_shard)
- Megatron #4835: https://github.com/NVIDIA/Megatron-LM/pull/4835 (DBuffer foundation, MERGED June 17)
- Megatron #4976: https://github.com/NVIDIA/Megatron-LM/pull/4976 (original PR, GitHub closed)
- Megatron #5369: https://github.com/NVIDIA/Megatron-LM/pull/5369 (meta-parameter follow-up, +2398/-0)
- Megatron #5395: https://github.com/NVIDIA/Megatron-LM/pull/5395 (skip_grad_norm_clip +15/-1)
- Megatron #5396: https://github.com/NVIDIA/Megatron-LM/pull/5396 (GDN L2-norm fold +7/-4)
- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394 (ChainedOptimizer Muon clipping)
- Megatron #5389: https://github.com/NVIDIA/Megatron-LM/pull/5389 (GDN THD MERGED June 17)
- PyTorch #187620: https://github.com/pytorch/pytorch/pull/187620 (PartialOffloadPolicy)
- verl #6512: https://github.com/volcengine/verl/pull/6512 (per-unit LoRA summon, MERGED June 18)
- verl #6699: https://github.com/volcengine/verl/pull/6699 (detach memory fix, MERGED June 18)
- verl #6791: https://github.com/volcengine/verl/pull/6791 (DSv4/Megatron Lite, MERGED June 18)
- DeepSpeed #8061: overlap_comm + torch.compile = NaN (multi-stream root cause)
- DeepSpeed #8068: gradient_clipping default change (0→1.0)
- Existing reading: notebook/projects/megatron-5387-fsdp-fully-shard-reading.md
