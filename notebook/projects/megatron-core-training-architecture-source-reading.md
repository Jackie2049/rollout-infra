# Megatron-LM Core Training Architecture Source-Level Reading

Date: 2026-06-16
Repo: NVIDIA/Megatron-LM (main branch, last pushed 2026-06-15)
Focus: megatron/core/ directory — training loop, parallel state, distributed optimizer, TransformerLayer, FlashAttention

---

## 1. MegatronCore Training Loop

### 1.1 Entry Point: megatron/training/training.py

★★★★★ The training loop lives in `megatron/training/training.py`, NOT in megatron/core/. The core library provides building blocks; the training script orchestrates them.

Key functions (line numbers from current main):

- `train()` (line 3107): Outer loop — iterations, checkpointing, validation, RL integration
- `train_step()` (line 2198): Single training step — the critical orchestrator
- `setup_model_and_optimizer()` (line 1956): Model+optimizer+DDP initialization
- `get_model()` (line 1631): Model provider wrapper, applies DDP

### 1.2 train_step() Flow (line 2198-2400)

★★★★★★★ train_step is the heart. Source-level pipeline:

```
train_step(forward_step_func, data_iterator, model, optimizer, ...):
  1. rerun_state_machine.should_run_forward_backward()  # fault tolerance loop
  2. model.zero_grad_buffer()                            # zero grad buffers
  3. optimizer.zero_grad()                                # zero optimizer state
  4. forward_backward_func(...)                           # THE MAIN STEP
  5. optimizer.step()                                     # parameter update
  6. (misc: vision grads, grad norm logging, etc.)
```

★★★★★ The `forward_backward_func` is selected by pipeline parallel configuration:

- `forward_backward_no_pipelining()` — when PP=1 (single stage, most relevant for RTX 4090)
- `forward_backward_pipelining_without_interleaving()` — PP>1, non-interleaved
- `forward_backward_pipelining_with_interleaving()` — PP>1 with virtual pipeline stages

### 1.3 forward_backward_no_pipelining (megatron/core/pipeline_parallel/schedules.py line 637)

★★★★★ This is the RTX 4090 relevant path (PP=1). Source-level:

```python
def forward_backward_no_pipelining(*, forward_step_func, data_iterator, model, ...):
    # No PP — single model, no p2p communication needed
    # Process microbatches sequentially
    for i in range(num_microbatches):
        # Forward microbatch
        output_tensor, num_tokens = forward_step(
            forward_step_func, data_iterator, model, num_microbatches,
            input_tensor=None, ...
        )
        # Backward microbatch
        input_tensor_grad = backward_step(
            input_tensor=None, output_tensor, output_tensor_grad=None, config
        )
        # Gradient accumulation happens inside DDP's grad buffer
```

★★★★★ Gradient accumulation across microbatches: handled by DDP's `param.main_grad.add_(param.grad.data)` in backward post-hook. Each microbatch adds its gradient contribution to `main_grad`. After all microbatches, reduce-scatter/all-reduce is triggered.

### 1.4 forward_step (schedules.py line 362) and backward_step (line 497)

★★★★ forward_step:
```python
def forward_step(forward_step_func, data_iterator, model, ...):
    set_input_tensor(input_tensor)  # for PP stages
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    output_tensor, num_tokens = forward_step_calc_loss(...)
```

★★★★ backward_step:
```python
def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
    # Retain grad on input_tensor
    input_tensor.retain_grad()
    # Backward pass
    torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
    # Collect input gradients
    input_tensor_grad = input_tensor.grad
```

★★★★★ Key insight: autograd.backward is called per microbatch. The gradient accumulation happens at the DDP level, NOT at the autograd level. Each backward call adds to `param.main_grad` via the backward post-hook.

---

## 2. Parallel State Initialization

### 2.1 initialize_model_parallel (megatron/core/parallel_state.py line 547)

★★★★★★★ This function creates ALL process groups. 2238 lines total. Key signature:

```python
def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    num_distributed_optimizer_instances=1,
    expert_tensor_parallel_size=None,  # defaults to tp_size
    order="tp-cp-ep-dp-pp",  # ★★★★★★ rank layout order
    ...
):
```

★★★★★★★ RankGenerator: The `order` parameter is critical. Default "tp-cp-ep-dp-pp" means:
- Adjacent ranks are in the same TP group first
- Then CP, then EP, then DP, then PP
- This ensures TP ranks are on the same node (NVLink-connected)

Two RankGenerators are created:
- `decoder_rank_generator` — for dense (non-expert) layers
- `expert_decoder_rank_generator` — for MoE expert layers (with separate tp/ep/dp)

★★★★★★★ world_size=1 (single GPU) behavior:

```python
# When world_size=1 and all parallel sizes=1:
model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
# = 1 * 1 * 1 = 1
data_parallel_size = world_size // model_size = 1 // 1 = 1
```

All groups become singleton groups of size 1. No communication needed. But:

★★★★★ ★★★★★ CRITICAL: The LayerWise optimizer CRASH (#5203) occurs exactly here. When `dp_cp_params_list=None` on single GPU, the optimizer crashes because it tries to shard parameters across a singleton group. This confirms that singleton PG degeneration has real bugs in Megatron.

### 2.2 Process Groups Created (in order of creation)

★★★★★ Process groups are created in specific order (SHARP requires first-created group):

1. **dp_cp group** (data parallel + context parallel) — first for SHARP support
2. **dp group** (data parallel only)
3. **cp group** (context parallel)
4. **mp group** (model parallel = tp+pp)
5. **tp group** (tensor model parallel)
6. **pp group** (pipeline model parallel)
7. **embedding group**
8. **position embedding group**
9. **tp_cp group** (tensor + context parallel)
10. **tp_dp_cp group** (tensor + data + context parallel, for FP8)
11. **ep group** (expert model parallel)
12. **expt_tp group** (expert tensor parallel)
13. **tp_ep group** (expert tensor + model parallel)
14. **tp_ep_pp group** (expert tensor + model + pipeline parallel)
15. **expt_dp group** (expert data parallel)
16. **intra_partial_dp_cp** — for num_distributed_optimizer_instances > 1

★★★★★ Total: 16+ process groups! Each gets both NCCL and optionally Gloo backend.

### 2.3 ProcessGroupCollection (megatron/core/process_groups_config.py)

★★★★★ New dataclass-based PG management (2025+). Replaces global variable access pattern:

```python
@dataclass
class ProcessGroupCollection:
    # Model Parallelism PGs
    tp: ProcessGroup          # tensor parallel
    pp: ProcessGroup          # pipeline parallel
    mp: ProcessGroup          # model parallel (tp+pp)
    cp: ProcessGroup          # context parallel
    tp_cp: ProcessGroup       # tensor + context
    ep: ProcessGroup          # expert model parallel
    expt_tp: ProcessGroup     # expert tensor parallel
    # Data Parallelism PGs
    dp: ProcessGroup          # data parallel
    dp_cp: ProcessGroup       # data + context parallel
    expt_dp: ProcessGroup     # expert data parallel
    intra_dp_cp: ProcessGroup # intra partial DP+CP (for multi DistOpt instances)
    dp_cp_ag: ProcessGroup    # separate AG/RS overlap communicator
    ...
```

★★★★★ This is being threaded through model construction, DDP, and pipeline schedules — replacing `parallel_state.get_*()` global calls. Still transitional: both patterns coexist.

### 2.4 Communication Overlap

★★★★★ Megatron has TWO overlap mechanisms:

**overlap_grad_reduce**: Grad sync overlaps with backward compute
- Buckets: parameters grouped into buckets (~40M elements each)
- Backward post-hook: when all grads in a bucket ready → dispatch reduce-scatter async
- `is_last_microbatch` flag: only reduce on last microbatch
- Communication stream: separate CUDA stream for NCCL ops

**overlap_param_gather**: Param all-gather overlaps with forward compute
- Forward pre-hook: before each module's forward, wait for param all-gather
- `overlap_param_gather_with_optimizer_step`: AG dispatched during optimizer step
- Param data stored in contiguous buffer, all-gathered from shards

★★★★★★★ Both mechanisms use the same bucket group structure — `_ParamAndGradBucketGroup`.

---

## 3. Distributed Optimizer (DDP + DistributedOptimizer)

### 3.1 DistributedDataParallel (megatron/core/distributed/distributed_data_parallel.py)

★★★★★★★ DDP is the core training wrapper. Key architecture:

```python
class DistributedDataParallel(_BaseDataParallel):
    def __init__(self, config, ddp_config, module, ...):
        # 1. Collect all trainable parameters
        # 2. Group by (param_dtype, grad_dtype, is_expert_parallel) → buffer_groups
        # 3. Create _ParamAndGradBuffer per group
        # 4. Partition buffers into buckets → bucket_groups
        # 5. Setup hooks:
        #    - forward pre-hook (if overlap_param_gather)
        #    - backward post-hook (for grad accumulation + overlap_grad_reduce)
```

★★★★★ Gradient scaling logic (critical for MoE):

```python
if ddp_config.average_in_collective:
    gradient_scaling_factor = 1.0           # NCCL averages internally
    expert_gradient_scaling_factor = edp_size / dp_size  # expert grads scaled before EP reduce
else:
    gradient_scaling_factor = 1.0 / dp_size  # manual scaling
```

★★★★★★★ Backward post-hook — the heart of gradient accumulation:

```python
def _make_backward_post_hook(self, param):
    def hook(*unused):
        if param in self.param_to_bucket_group:
            # Add param.grad to param.main_grad
            if param.grad is not None and not param.grad_added_to_main_grad:
                param.main_grad.add_(param.grad.data)
            param.grad = None  # Free per-microbatch grad immediately!

            if self.ddp_config.overlap_grad_reduce:
                # Register grad ready → may trigger async reduce-scatter
                self.param_to_bucket_group[param].register_grad_ready(param, force_all_reduce)
    return hook
```

★★★★★★★ Key insight: `param.grad = None` after adding to main_grad. This frees per-microbatch gradient memory immediately. Only `main_grad` in the contiguous buffer persists across microbatches.

### 3.2 _ParamAndGradBuffer (megatron/core/distributed/param_and_grad_buffer.py line 938)

★★★★★★★ The buffer is the core memory management structure:

```python
class _ParamAndGradBuffer:
    # Two contiguous tensors:
    # param_data: stores all parameters in contiguous memory
    # grad_data: stores all gradients in contiguous memory
    # Each parameter is a VIEW into these buffers (param.data → param_data slice)
    # Each gradient is a VIEW (param.main_grad → grad_data slice)
```

★★★★★★★ Key design: contiguous buffers enable:
1. Single NCCL operation per bucket (not per parameter)
2. overlap_grad_reduce: async reduce on bucket slice
3. overlap_param_gather: async all-gather on bucket slice
4. DistributedOptimizer: shard buffer by dp_rank, each rank owns 1/dp_size slice

★★★★★ NVFP4 dual-buffer layout: param buffer stores packed bytes (numel/2), grad buffer uses full numel. Two separate index maps.

### 3.3 _ParamAndGradBucketGroup (line 157)

★★★★★ BucketGroup aggregates multiple buckets for communication aggregation:

```python
class _ParamAndGradBucketGroup:
    buckets: List[_ParamAndGradBucket]  # multiple buckets in one group
    param_to_bucket: Dict               # param → which bucket
    is_last_microbatch: bool            # controls when to reduce
    num_grads_ready: int                # counter for overlap_grad_reduce
    communication_stream: torch.cuda.Stream  # separate CUDA stream
    next_param_gather_bucket_group: Self    # chain for overlap_param_gather
```

★★★★★ Overlap_grad_reduce flow:
1. Each backward post-hook increments `num_grads_ready` for the bucket
2. When all params in bucket group have grads ready → dispatch async reduce-scatter
3. On `is_last_microbatch=True` → finalize all outstanding handles

★★★★★ Overlap_param_gather flow:
1. After optimizer step → dispatch all-gather for next bucket group
2. Forward pre-hook → wait for the all-gather of current bucket group
3. Chain: bucket_groups linked via `next_param_gather_bucket_group`

### 3.4 DistributedOptimizer (megatron/core/optimizer/distrib_optimizer.py line 107)

★★★★★★★ DistributedOptimizer shards optimizer state across DP ranks:

```python
class DistributedOptimizer(MixedPrecisionOptimizer):
    """Optimizer that shards state across data-parallel ranks."""
```

★★★★★★★ Key data flow:
1. Each DP rank owns 1/dp_world_size of the param buffer
2. Optimizer states (momentum, variance) only exist for owned shard
3. Step: update owned shard in main_param → all-gather to all ranks
4. Total memory: 2Ψ model params + 2Ψ/dp_size optimizer states (vs 2Ψ + 12Ψ/dp_size without DistOpt)

★★★★★ _build_model_gbuf_param_range_map (line 128): Maps each parameter to its range in the global buffer and its shard in the local buffer. Each DP rank "owns" a contiguous region of the buffer.

★★★★★ compute_full_param_layout (classmethod): Pre-computes PerBufferParamLayout before DDP construction. This is now recommended over auto-compute inside DDP.

### 3.5 LayerWiseDistributedOptimizer (line 89)

★★★★★ Experimental optimizer that distributes by LAYER instead of by parameter position:

```python
class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer.
    1. weights split into lists, each rank keeps only its shard
    2. DDP handles allreduce grad (each rank has full model + grad)
    3. optimizer only updates params belonging to this DP rank
    4. grad_norm/zero counting reduced globally
    5. allgather updated params to every rank
    """
```

★★★★★★★ CRITICAL BUG for single GPU: `self.dp_cp_params_list = None` (line 453) when world_size=1 → crash in step(). This is Issue #5203. Singleton group degeneration is a real bug, unlike DeepSpeed's graceful handling.

---

## 4. TransformerLayer Core

### 4.1 TransformerLayer (megatron/core/transformer/transformer_layer.py)

★★★★★★★ The layer class hierarchy:

```
TransformerLayer(GraphableMegatronModule, BaseTransformerLayer)
  - input_layernorm
  - self_attention (Attention subclass)
  - self_attn_bda (bias-dropout-add)
  - pre_cross_attn_layernorm
  - cross_attention (optional)
  - cross_attn_bda
  - pre_mlp_layernorm
  - mlp (MLP or IdentityOp)
  - mlp_bda
```

★★★★★ forward flow (line 733, delegated from _forward_attention + _forward_mlp):

```python
def forward(self, *args, **kwargs):
    hidden_states, context = self._forward_attention(*args, **kwargs)
    output = self._forward_mlp(hidden_states, ...)
    return output, context

def _forward_attention(self, hidden_states, ...):
    # 1. input_layernorm (with optional offload)
    # 2. self_attention → attention_output_with_bias
    # 3. self_attn_bda (bias + dropout + add residual)
    # 4. pre_cross_attn_layernorm
    # 5. cross_attention (if applicable)
    # 6. cross_attn_bda
    return hidden_states, context

def _forward_mlp(self, hidden_states, ...):
    # 1. pre_mlp_layernorm (with optional offload)
    # 2. mlp.forward → mlp_output_with_bias
    # 3. mlp_bda (bias + dropout + add residual)
    return output
```

★★★★★ Selective recomputation (activation checkpointing):
- `recompute_granularity == 'selective'` + `"core_attn" in recompute_modules` → checkpoint core attention
- `recompute_mlp = True` → checkpoint entire MLP
- `recompute_input_layernorm` → checkpoint input layernorm output

★★★★★ Fine-grained activation offloading:
- `offload_qkv_linear`, `offload_core_attention`, `offload_attn_proj`, `offload_attn_norm`, `offload_mlp_norm`
- Uses `FineGrainedActivationOffloadingInterface` (off_interface)

★★★★★ MLP chunking for prefill/training:
- `mlp_chunks_for_prefill > 1`: chunks MLP computation during prefill
- `mlp_chunks_for_training > 1`: chunks MLP during training
- Reduces peak activation memory

### 4.2 MLP Parallel Pattern (megatron/core/transformer/mlp.py)

★★★★★★★ MLP = ColumnParallelLinear(fc1) + Activation + RowParallelLinear(fc2)

★★★★★★★★★ EXACT TP dispatch pattern:

**linear_fc1 (ColumnParallelLinear)**:
```python
# __init__: output_size_per_partition = divide(output_size, world_size)
# For GLU/SwiGLU: ffn_hidden_size *= 2, stride=2 (interleaved gate+up per TP rank)
# gather_output=False → each TP rank gets its shard of the output

# forward:
# 1. input_parallel = copy_to_tp_region(input_) OR just input_ (if sequence_parallel)
# 2. output_parallel = X @ weight_shard  # local matmul on TP partition
# 3. NO all-gather → output stays partitioned across TP
```

**linear_fc2 (RowParallelLinear)**:
```python
# __init__: input_size_per_partition = divide(input_size, world_size)
# input_is_parallel=True → input already split across TP ranks

# forward:
# 1. input_parallel = input_ (already partitioned from fc1 output)
# 2. output_parallel = input_parallel @ weight_shard  # local matmul
# 3. All-reduce/reduce-scatter across TP group → full output
#    - sequence_parallel: reduce_scatter_to_sequence_parallel_region
#    - else: reduce_from_tensor_model_parallel_region
```

★★★★★★★★★ COMPLETE TP MLP flow:
```
Input [s, b, h] → fc1 (ColumnParallel) → [s, b, 4h/p] per rank
                   (split output dim, no gather)
→ Activation (SwiGLU/GELU locally) → [s, b, 2h/p] per rank  (half for SwiGLU)
→ fc2 (RowParallel) → [s, b, h/p] per rank
                   (split input dim, all-reduce output)
→ All-reduce → [s, b, h] full output
```

★★★★★ For MoE: `is_expert=True` → uses expert TP group, `explicit_expert_comm` for EP-aware communication.

### 4.3 Parallel Self-Attention (megatron/core/transformer/attention.py)

★★★★★★★ Attention uses the same Column→Row TP pattern:

**linear_qkv (ColumnParallelLinear or TEColumnParallelLinear)**:
- Column-parallel: splits Q, K, V projections across TP
- For GQA: num_query_groups_per_partition = num_query_groups / tp_size
- gather_output=False → QKV stays partitioned

**linear_proj (RowParallelLinear or TERowParallelLinear)**:
- Row-parallel: input_is_parallel=True
- All-reduce/reduce-scatter after projection

★★★★★★★★★ COMPLETE TP Attention flow:
```
Input [s, b, h] → linear_qkv (ColumnParallel) → [s, b, (q+h_kv)/p] per rank
→ Core attention (local computation on partitioned heads)
→ linear_proj (RowParallel) → all-reduce → [s, b, h] full output
```

★★★★★ GQA handling:
```python
if self.config.num_query_groups < world_size:
    # num_kv_heads < tp_size: each TP rank gets 1 kv_head + (num_q_heads/num_kv_heads) q_heads
    self.num_query_groups_per_partition = 1
    self.num_attention_heads_per_partition = divide(num_attention_heads, num_query_groups)
else:
    # Standard: each TP rank gets num_kv_heads/tp_size kv_heads
    self.num_query_groups_per_partition = divide(num_query_groups, world_size)
```

---

## 5. FlashAttention Integration

### 5.1 Attention Backend Selection

★★★★★★★ Three levels of attention backend selection:

**Level 1: transformer_impl config field** (TransformerConfig line 1118):
```python
transformer_impl: Literal['local', 'transformer_engine', 'inference_optimized'] = 'local'
```
- `local` → Megatron-native modules (DotProductAttention, ColumnParallelLinear, etc.)
- `transformer_engine` → TE modules (TEDotProductAttention, TEColumnParallelLinear, etc.)
- `inference_optimized` → inference-specific modules

**Level 2: attention_backend config field** (TransformerConfig line 143):
```python
attention_backend: AttnBackend = AttnBackend.auto
# AttnBackend enum: flash=1, fused=2, unfused=3, local=4, auto=5
```

**Level 3: Spec provider mechanism** (TESpecProvider):
```python
class TESpecProvider(BackendSpecProvider):
    def core_attention(self) -> type:
        return TEDotProductAttention  # TE always uses flash attention
```

★★★★★★★ When transformer_impl='local' + attention_backend=auto:
- DotProductAttention (unfused, PyTorch baddbmm+bmm)
- NO flash attention — pure PyTorch matmul + FusedScaleMaskSoftmax

★★★★★★★ When transformer_impl='transformer_engine':
- TEDotProductAttention → TE's DotProductAttention with flash attention enabled
- TE internally selects FA2/FA3 based on hardware and availability

★★★★★★★ When attention_backend=flash + transformer_impl=local:
- Uses flash_attn_varlen_func from flash-attn package directly
- Only for inference (dynamic batching), not training

### 5.2 DotProductAttention (megatron/core/transformer/dot_product_attention.py line 26)

★★★★★ The unfused/local backend. Pure PyTorch implementation:

```python
class DotProductAttention(MegatronModule):
    def forward(self, query, key, value, attention_mask, ...):
        # 1. GQA: repeat_interleave key/value if num_kv_heads < num_attention_heads
        # 2. baddbmm: Q @ K^T with preallocated buffer + softmax_scale
        #    matmul_result = torch.baddbmm(buffer, Q^T, K^T, beta=0, alpha=softmax_scale)
        # 3. FusedScaleMaskSoftmax: scale + mask + softmax (fused kernel)
        # 4. Dropout (with TP-aware RNG tracker for sequence_parallel)
        # 5. bmm: attention_probs @ V → context
        # 6. Reshape: [b, np, sq, hn] → [sq, b, hp]
```

★★★★★ No flash attention in local mode training! This is pure unfused attention. Batch-dependent by nature (baddbmm batch size varies).

### 5.3 TEDotProductAttention (megatron/core/extensions/transformer_engine.py line 1583)

★★★★★★★ TE wrapper that enables flash attention:

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    """Wrapper for TE's DotProductAttention with flash attention enabled."""

    def __init__(self, config, layer_number, attn_mask_type, attention_type, ...):
        # Key parameters:
        self.num_splits = 1 if config.batch_invariant_mode else num_splits  # ★★★★★
        extra_kwargs["num_gqa_groups"] = config.num_query_groups  # GQA support
        # CP support (TE >= 1.0.0):
        extra_kwargs["cp_group"] = pg_collection.cp
        extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
        # deterministic_mode: requires NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
        # Window attention (TE >= 1.2.0)
        # KV channels (TE >= 1.10.0) — MLA support
```

★★★★★★★★★★★★★★★ batch_invariant_mode → num_splits=1 → TE uses single-split flash attention → deterministic! This is the Megatron equivalent of SGLang's deterministic inference, but for TRAINING.

### 5.4 FlashAttention Availability Checks (attention.py lines 60-90)

★★★★★★★ Four flash attention backends checked:

```python
# FA3 (flash_attn_3 or flashattn_hopper)
try: from flash_attn_3.flash_attn_interface import _flash_attn_forward → HAVE_FA3 = True
except: from flashattn_hopper.flash_attn_interface import _flash_attn_forward → HAVE_FA3 = True

# FA4 (flash_attn.cute)
try: from flash_attn.cute import flash_attn_varlen_func → HAVE_FA4 = True

# FlashMLA
try: from flash_mla import flash_mla_with_kvcache → HAVE_FMLA = True

# FA2 (flash_attn standard)
try: from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
```

★★★★★★★★★★★ Backend hierarchy for inference:
- FA4 > FA3 > FA2 for dynamic batching inference
- FlashMLA for MLA models specifically
- FA3 `_flash_attn_forward` wrapper used directly in `_flash_attention_3_forward_wrapper`

### 5.5 Flash Decode and KV Cache (attention.py)

★★★★★ Inference KV cache integration:

```python
def flash_decode(self, query_layer, key_layer, value_layer, ...):
    """Flash decoding: RoPE + KV cache update + flash attention in one kernel"""
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v,
        rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        cache_seqlens=sequence_len_offset,
        rotary_interleaved=rotary_interleaved
    )
```

★★★★★ flash_decode_and_prefill: Mixed decode+prefill batching
- Uses FA4 or FA3 for dynamic batching
- block_table for paged KV cache
- Separate paths for decode-only vs mixed batches

### 5.6 Batch Invariant Kernels (megatron/core/transformer/custom_layers/batch_invariant_kernels.py)

★★★★★ NEW (2025+, from Thinking Machines Lab). Triton persistent kernels:

```python
# batch_invariant_kernels.py
# Triton kernels with constexpr BLOCK_SIZE → NOT autotuned → batch-invariant
# _matmul_launch_metadata: fixed grid, not data-dependent
# Similar concept to SGLang's deterministic Triton kernels
```

★★★★★★★ config.batch_invariant_mode = True → num_splits=1 in TE → Triton persistent matmul in BIK → deterministic forward regardless of batch size.

★★★★★ RTX 4090 relevance: batch_invariant_mode would solve batch-dependent training results, but "significantly affects speed" — not production-viable for training on SM89. Better path: SM<90 Fusion Guard in Inductor (our proposed PyTorch upstream PR).

---

## 6. Key Architecture Insights

### 6.1 Memory Data Flow

★★★★★★★★★★★★★ COMPLETE memory flow for a training step:

```
Param buffer (contiguous, bf16/fp8):
  param.data → view into param_data buffer slice
  param.main_grad → view into grad_data buffer slice

Forward:
  param.data used for compute
  (if overlap_param_gather: all-gather from shard to full param_data before forward)

Backward (per microbatch):
  param.grad computed by autograd → temporary tensor
  backward post-hook: param.main_grad.add_(param.grad); param.grad = None
  (if overlap_grad_reduce: async reduce-scatter on bucket when all grads ready)

After all microbatches:
  reduce-scatter/all-reduce on grad_data (if not overlapped)
  DistributedOptimizer.step():
    1. Get grad norm from grad_data
    2. Scale grads by 1/grad_norm
    3. Update main_param (fp32) for owned shard
    4. Copy updated main_param to param_data shard
    5. All-gather param_data → all ranks get full params
```

★★★★★★★ Total memory per rank (with DistributedOptimizer):
- param_data: 2Ψ (full model params, bf16)
- grad_data: 2Ψ (full gradients, bf16 or fp32)
- main_param: 2Ψ/dp_size (fp32 optimizer copy, only owned shard)
- optimizer states: 12Ψ/dp_size (momentum + variance, fp32, only owned shard)

Without DistributedOptimizer: 2Ψ + 2Ψ + 2Ψ + 12Ψ = 16Ψ (all in fp32 for optimizer)
With DistributedOptimizer: 2Ψ + 2Ψ + 2Ψ/dp + 12Ψ/dp = (4Ψ + 14Ψ/dp) → much less!

★★★★★ Single GPU (dp_size=1): DistributedOptimizer provides NO benefit. All optimizer states are on one GPU. Only overhead from the sharding logic.

### 6.2 Sequence Parallel Impact

★★★★★ When sequence_parallel=True:
- TP groups use reduce-scatter (not all-reduce) for RowParallelLinear output
- Input is already partitioned along sequence dimension
- Each TP rank processes s/p tokens
- Grad reduce uses reduce-scatter_to_sequence_parallel_region

★★★★★ Without sequence_parallel:
- RowParallelLinear uses all-reduce (reduce_from_tensor_model_parallel_region)
- Full sequence processed by each rank
- More communication, but simpler

### 6.3 Communication Overlap Detail

★★★★★★★ overlap_grad_reduce=True:
1. PP rank 0 only (higher PP ranks: `self.bucket_size = None` → no bucketing)
2. Buckets created in forward order, reduced in backward order
3. Each bucket's reduce-scatter dispatched when all its grads are computed
4. Communication runs on separate CUDA stream, overlapping with backward compute

★★★★★ overlap_param_gather=True:
1. Requires DistributedOptimizer or LayerWise optimizer
2. Forward pre-hook waits for all-gather before each module
3. All-gather dispatched during optimizer step or pipeline schedule
4. Param buffer and grad buffer may share memory (reuse_grad_buf_for_mxfp8_param_ag)

### 6.4 RTX 4090 Architecture Implications

★★★★★★★★★★★★★★★★★★★ CRITICAL findings for RTX 4090:

1. **Megatron DDP on single GPU**: All process groups are singleton (size=1). No communication. DDP overhead (buffer allocation, hooks) still present but no actual NCCL ops.

2. **DistributedOptimizer on single GPU**: NO benefit. Only overhead. `dp_cp_params_list` can be None → crash risk (#5203). Do NOT use on RTX 4090.

3. **Gradient accumulation**: Works correctly on single GPU. `main_grad.add_(param.grad)` per microbatch → accumulated in contiguous buffer.

4. **TP=1, PP=1, DP=1**: All parallel groups degenerate. Megatron designed for multi-GPU; single GPU is an edge case with bugs.

5. **FlashAttention**: local mode (unfused) = no flash attention. TE mode = flash attention but requires TE installation. SM89 supports FA2 but NOT FA3 (SM90+ only).

6. **Batch invariant mode**: Available but slow. Not viable for RTX 4090 training production. Better to fix at Inductor level.

7. **MoE on RTX 4090**: EP=1 → singleton expert group → DeepSpeed handles gracefully, Megatron may crash. Need small model + LoRA for viability.

8. **CUDA graphs**: `cuda_graph_impl='local'` for local backend, `'transformer_engine'` for TE. Both work on SM89 but require enforce_eager for torch.compile.

### 6.5 Comparison with DeepSpeed

★★★★★★★★★★★ Architecture comparison:

| Feature | Megatron-LM | DeepSpeed |
|---------|-------------|-----------|
| Grad buffer | Contiguous _ParamAndGradBuffer | Flat param buffer in ZeRO |
| Grad accumulation | DDP backward post-hook | ZeRO gradient partitioning |
| Optimizer sharding | DistributedOptimizer (1/dp_size) | ZeRO-3 (1/dp_size per stage) |
| Single GPU | Singleton PG → crash risk (#5203) | ZeRO-2 works, ZeRO-3 limited |
| LoRA | NOT in core (only NeMo/Megatron-Bridge) | LoRAOptimizedLinear in core |
| Communication overlap | DDP-level bucket overlap | ZeRO comm overlap partition groups |
| Process groups | 16+ groups, global variables | ZeRO partition groups |
| MoE support | EP + expert TP/DP groups | AutoEP + ZeRO groups |

★★★★★★★★ Megatron's strength: production-grade multi-GPU training with mature TP/PP/DP/CP/EP. Weakness: single GPU edge case, no LoRA in core, no ZeRO-3-like full sharding for single GPU.

★★★★★★★★★★★ RTX 4090 ranking (from previous notes):
- rLLM Tinker #1 (in-process, auto LoRA, bypass default)
- verl CPPO+bypass #2
- DeepSpeed ZeRO-2+CPU_Adam+LoRAOptimizedLinear #2.5
- Megatron core #3 (no LoRA, singleton bugs, DDP overhead on single GPU)

---

## 7. Source File Index

| File | Lines | Key Content |
|------|-------|-------------|
| megatron/core/parallel_state.py | 2238 | initialize_model_parallel (line 547), 16+ PG globals |
| megatron/core/distributed/distributed_data_parallel.py | 635 | DDP class, backward post-hook, overlap mechanisms |
| megatron/core/distributed/param_and_grad_buffer.py | 1646 | Buffer, Bucket, BucketGroup classes |
| megatron/core/transformer/transformer_layer.py | 1600 | TransformerLayer, forward flow |
| megatron/core/transformer/mlp.py | ~350 | MLP with Column→Row TP pattern |
| megatron/core/transformer/attention.py | 1887 | Attention, FA2/FA3/FA4/FMLA, flash_decode |
| megatron/core/transformer/dot_product_attention.py | ~300 | Unfused local attention backend |
| megatron/core/transformer/enums.py | ~120 | AttnBackend, AttnMaskType, CudaGraphModule |
| megatron/core/transformer/transformer_config.py | ~2600 | batch_invariant_mode, attention_backend, transformer_impl |
| megatron/core/transformer/custom_layers/batch_invariant_kernels.py | ~? | Triton persistent batch-invariant kernels |
| megatron/core/tensor_parallel/layers.py | ~1500 | ColumnParallelLinear (line 778), RowParallelLinear (line 1142) |
| megatron/core/optimizer/optimizer.py | ~large | MegatronOptimizer, MixedPrecisionOptimizer |
| megatron/core/optimizer/distrib_optimizer.py | ~large | DistributedOptimizer, shard logic |
| megatron/core/optimizer/layer_wise_optimizer.py | ~large | LayerWiseDistributedOptimizer, dp_cp_params_list bug |
| megatron/core/extensions/transformer_engine.py | ~large | TEDotProductAttention (line 1583), TELinear wrappers |
| megatron/core/extensions/transformer_engine_spec_provider.py | ~100 | TESpecProvider (core_attention → TEDotProductAttention) |
| megatron/core/process_groups_config.py | ~200 | ProcessGroupCollection dataclass |
| megatron/core/pipeline_parallel/schedules.py | ~large | forward_backward_no_pipelining (line 637), forward_step (362), backward_step (497) |
| megatron/training/training.py | ~large | train (3107), train_step (2198), setup (1956) |

---

## 8. Key Findings Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. **Training loop**: `train_step()` → `forward_backward_func()` → per-microbatch forward+backward → gradient accumulation in DDP buffer → `optimizer.step()`. NOT in megatron/core/ — in megatron/training/.

2. **DDP is the core**: DistributedDataParallel wraps model, manages contiguous param+grad buffers, handles gradient accumulation via backward post-hooks. Each microbatch: `main_grad.add_(grad); grad=None`.

3. **Parallel state**: 16+ process groups created by `initialize_model_parallel()`. Singleton groups (size=1) for single GPU — bugs exist (#5203 LayerWise crash).

4. **TP pattern**: ColumnParallel (split output, no gather) → local compute → RowParallel (split input, all-reduce output). Same for both MLP and Attention.

5. **FlashAttention**: NOT used in local mode training. Only available via TransformerEngine (TEDotProductAttention). FA3/FA4 for inference only. SM89 gets FA2 at most.

6. **Batch invariant mode**: Triton persistent kernels with constexpr BLOCK_SIZE → deterministic but slow. Not production-viable for RTX 4090 training.

7. **DistributedOptimizer on single GPU**: No benefit, potential crash. RTX 4090 must avoid it.

8. **ProcessGroupCollection**: New dataclass-based PG management replacing global variables. Transitional — both patterns coexist.

9. **Communication overlap**: Two mechanisms — overlap_grad_reduce (backward) and overlap_param_gather (forward). Both use bucket groups with separate CUDA streams.

10. **MoE expert groups**: Separate EP/TP/DP groups for experts. EP=1 → singleton → potential crash.
