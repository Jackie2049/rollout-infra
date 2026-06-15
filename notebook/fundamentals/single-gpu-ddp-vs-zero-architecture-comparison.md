# Single-GPU DDP vs ZeRO Training Architecture Comparison

> 2026-06-16 | RTX 4090 consulting reference
> Focus: How gradient accumulation, optimizer sharding, and communication patterns differ on single GPU across Megatron DDP, DeepSpeed ZeRO, and PyTorch FSDP2
> Relevance: RTX 4090 GRPO training advisor must know which framework's architecture is most efficient on single GPU

---

## 1. Architecture Overview Comparison

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Feature | Megatron DDP | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-3 | PyTorch FSDP2 | rLLM Tinker |
|---------|-------------|-------------------|-------------------|---------------|-------------|
| Grad accumulation | Contiguous buffer + backward post-hook | Flat param buffer + ZeRO grad partition | Same as ZeRO-2 | Per-param DTensor + unshard→compute→reshard | In-process, simple |
| Optimizer | DistributedOptimizer (sharded) | CPU_Adam (offloaded) or FP16Optimizer | Same, but ZeRO-3 partitions params | Per-param optimizer (sharded via DTensor) | AdamW (LoRA params only) |
| Single GPU sharding | None (dp=1 → singleton) | None (ZeRO-2: dp=1, no partitioning) | None (ZeRO-3: dp=1, partition_size=full) | None (DTensor single GPU = no sharding) | N/A (LoRA only) |
| Grad buffer | Contiguous _ParamAndGradBuffer | Flat buffer per partition group | Same | Per-param DTensor buffers | Auto-managed |
| Communication overlap | overlap_grad_reduce + overlap_param_gather | ZeRO comm overlap partition groups | Same | AC overlap (all-gather ahead) | N/A (single GPU) |
| Process groups | 16+ PG (global vars) | ZeRO partition groups | Same | DeviceMesh (2D/3D) | N/A |
| LoRA support | NOT in core (only NeMo/Bridge) | LoRAOptimizedLinear (split forward) | Same | Not native | Auto LoRA rank=32 |
| Ref model needed | Yes (for KL penalty) | Yes | Yes | Yes | No (bypass_mode) |

---

## 2. Gradient Accumulation: The Core Difference

### 2.1 Megatron DDP Gradient Accumulation

★★★★★★★★★ Megatron uses contiguous buffers and backward post-hooks:

```python
# Per-microbatch backward:
# 1. autograd.backward() computes param.grad (temporary tensor)
# 2. backward post-hook: param.main_grad.add_(param.grad.data)
# 3. param.grad = None  # Free per-microbatch grad immediately!
# 4. After all microbatches: reduce-scatter/all-reduce on main_grad
```

★★★★★★★ Key advantage: `param.grad = None` frees per-microbatch gradient memory immediately. Only `main_grad` in contiguous buffer persists. This means peak gradient memory = main_grad size (not main_grad + per-microbatch grad).

★★★★★★★ On single GPU: No reduce-scatter needed (dp=1). But DDP overhead still present:
- Contiguous buffer allocation (~2Ψ for params + ~2Ψ for grads)
- Backward post-hook overhead (param.main_grad.add_(param.grad.data) per parameter)
- Bucket management (~40M elements per bucket)

### 2.2 DeepSpeed ZeRO-2 Gradient Accumulation

★★★★★★★★★ DeepSpeed uses flat parameter buffers and ZeRO gradient partitioning:

```python
# Per-microbatch backward:
# 1. autograd.backward() computes param.grad
# 2. DeepSpeed averages gradients across microbatches
# 3. On dp=1: no partitioning → partition_size = full model
# 4. After all microbatches: All-reduce (but dp=1 → skips entirely!)
```

★★★★★★★ On single GPU: ZeRO-2 partition_size = full model (no sharding). But:
- `coalesce_grad_reduction=True`: group gradients into contiguous buffers → reduce overhead
- `CPU_Adam offload`: optimizer state on CPU → frees GPU memory
- `LoRAOptimizedLinear(offload_ratio=0.5)`: splits base weights → offload half to CPU

★★★★★★★★★ DeepSpeed ZeRO-2 single GPU config is the BEST for RTX 4090 because:
1. CPU_Adam offload → real GPU memory savings (5-7x faster optimizer on CPU)
2. LoRAOptimizedLinear → 50% weight offloading → more GPU room for activations
3. coalesce_grad_reduction → less communication overhead even on single GPU
4. No ref model needed if using bypass_mode (not native in DeepSpeed, but achievable)

### 2.3 DeepSpeed ZeRO-3 Gradient Accumulation

★★★★★★★★★ ZeRO-3 adds parameter partitioning and all-gather:

```python
# Per-microbatch forward:
# 1. All-gather parameters from shards (dp=1: full model, no gather needed!)
# 2. Forward pass with full parameters
# 3. Discard non-owned parameters (dp=1: all owned, no discard!)

# Per-microbatch backward:
# 1. All-gather parameters again (dp=1: skip)
# 2. Backward pass
# 3. Reduce-scatter gradients (dp=1: skip)
# 4. Update owned optimizer shard (dp=1: all params)
```

★★★★★★★★★ On single GPU: ZeRO-3 is PURE OVERHEAD. partition_size = full model → no sharding benefit → all-gather/reduce-scatter calls are skipped → but the partition management logic still runs → overhead without benefit.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090 MUST NOT use ZeRO-3! Use ZeRO-2 + CPU_Adam + LoRA instead.

### 2.4 PyTorch FSDP2 Gradient Accumulation

★★★★★★★★★ FSDP2 uses per-parameter DTensor and unshard→compute→reshard:

```python
# Per-microbatch forward:
# 1. Unshard: all-gather DTensor shard → full parameter
# 2. Forward pass
# 3. Reshard: discard non-local shards (free memory!)

# Per-microbatch backward:
# 1. Unshard: all-gather DTensor shard again
# 2. Backward pass → compute gradients
# 3. Reduce-scatter gradients → each rank keeps its shard
# 4. Reshard: free non-local gradient shards
```

★★★★★★★★★ On single GPU: FSDP2 is PURE OVERHEAD. DTensor with dp_mesh of size 1 → no sharding → unshard/reshard calls are identity operations → but the DTensor management overhead still exists.

★★★★★★★ Key difference from ZeRO-3: FSDP2 is composable (AC+FSDP+TP in one line) but still useless on single GPU.

### 2.5 rLLM Tinker Gradient Accumulation

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ rLLM Tinker is the SIMPLEST and most efficient on single GPU:

```python
# No DDP wrapper needed (in-process, single GPU)
# LoRA params only → ~0.6GB trainable (vs ~16GB full model)
# Simple AdamW on LoRA params
# Zero-copy weight sync (save_weights_for_sampler)
# bypass_mode=True → no ref model → save ~16GB
```

★★★★★★★★★ rLLM Tinker wins on single GPU because:
1. No DDP/DDP overhead (in-process, no distributed training wrapper)
2. Only LoRA params trainable (~0.6GB) → tiny optimizer state (~1.2GB)
3. bypass_mode → no ref model → save ~16GB
4. Zero-copy weight sync → no IPC → no serialization overhead
5. Total memory: ~9.2GB for Qwen3-1.7B (fits RTX 4090 24GB easily)

---

## 3. Optimizer Architecture Comparison

### 3.1 Memory per Rank (Mathematical)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

Let Ψ = number of model parameters. For single GPU (dp=1):

| Framework | Param Memory | Grad Memory | Optimizer Memory | Ref Model | Total |
|-----------|-------------|-------------|-------------------|-----------|-------|
| Megatron DDP (full) | 2Ψ (bf16) | 2Ψ (bf16) | 2Ψ + 12Ψ (fp32) | 2Ψ | 18Ψ |
| Megatron DDP (DistOpt) | 2Ψ (bf16) | 2Ψ (bf16) | 2Ψ + 12Ψ (fp32) | 2Ψ | 18Ψ (same! dp=1) |
| DeepSpeed ZeRO-2 (full) | 2Ψ (bf16) | 2Ψ (bf16) | 0 (CPU) | 2Ψ | 6Ψ |
| DeepSpeed ZeRO-2+LoRA | 2Ψ (bf16, frozen) | ~0.6Ψ (LoRA) | 0 (CPU) | 2Ψ | ~5Ψ |
| DeepSpeed ZeRO-2+LoRA+bypass | 2Ψ (frozen) | ~0.6Ψ | 0 (CPU) | 0 | ~3Ψ |
| DeepSpeed ZeRO-3 (dp=1) | 2Ψ | 2Ψ | 2Ψ + 12Ψ (fp32) | 2Ψ | 18Ψ (same!) |
| PyTorch FSDP2 (dp=1) | 2Ψ | 2Ψ | 2Ψ + 12Ψ (fp32) | 2Ψ | 18Ψ (same!) |
| rLLM Tinker+LoRA+bypass | 2Ψ (frozen) | ~0.6Ψ (bf16) | ~1.2Ψ (fp32 LoRA) | 0 | ~3.8Ψ |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

For Qwen3-1.7B (Ψ ≈ 3.4B params):

| Framework | Estimated Memory | Fits RTX 4090? |
|-----------|-----------------|---------------|
| Megatron DDP (full, no bypass) | ~61GB (18Ψ × 3.4) | NO! |
| DeepSpeed ZeRO-2+LoRA+bypass | ~10.2GB (3Ψ × 3.4) | YES! |
| DeepSpeed ZeRO-3 (dp=1) | ~61GB | NO! |
| PyTorch FSDP2 (dp=1) | ~61GB | NO! |
| rLLM Tinker+LoRA+bypass | ~13GB (3.8Ψ × 3.4) | YES! |

★★★★★★★★★ Only 3 configs fit RTX 4090: DeepSpeed ZeRO-2+LoRA+bypass, rLLM Tinker+LoRA+bypass, and verl+CPPO+bypass.

### 3.2 CPU_Adam vs GPU Optimizer

★★★★★★★★★ CPU_Adam is DeepSpeed's unique advantage for single GPU:

```python
# DeepSpeed CPU_Adam:
# - C++ implementation with SIMD (AVX512) + OpenMP
# - 5-7x faster than torch.optim.Adam on CPU
# - GPU memory saved: optimizer states live on CPU (12Ψ offloaded!)
# - CPU→GPU transfer: only updated parameters (small LoRA = tiny transfer)
# - Overhead: CPU→GPU transfer latency (~1-2ms for LoRA params)

# Standard GPU Adam:
# - 2Ψ momentum + 2Ψ variance in fp32 on GPU
# - No offloading → all optimizer memory on GPU
# - Faster step time (no CPU→GPU transfer)
# - But: 12Ψ memory consumption!
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ For LoRA training: CPU_Adim's advantage is smaller because LoRA params are tiny (~0.6GB). The CPU→GPU transfer overhead may exceed the GPU memory savings. Decision depends on total memory pressure:
- If GPU memory tight → CPU_Adam worth the transfer overhead
- If GPU memory comfortable → GPU Adam faster (no transfer)

---

## 4. Communication Overlap Comparison

### 4.1 Megatron Communication Overlap

★★★★★★★★★ Megatron has TWO overlap mechanisms:

**overlap_grad_reduce**:
- Buckets of ~40M parameters
- Backward post-hook: when all grads in bucket ready → dispatch async reduce-scatter
- Separate CUDA stream for NCCL ops
- On single GPU: reduce-scatter is identity → but bucket management still runs

**overlap_param_gather**:
- Requires DistributedOptimizer
- Forward pre-hook: before each module's forward → wait for all-gather
- All-gather dispatched during optimizer step
- On single GPU: all-gather is identity → but hook management still runs

★★★★★★★ On single GPU: Both mechanisms are pure overhead. No actual NCCL ops, but hook/bucket overhead persists.

### 4.2 DeepSpeed ZeRO Comm Overlap

★★★★★★★★★ DeepSpeed overlap via partition groups:

- `overlap_comm=True`: separate communication stream
- `reduce_bucket_size`: contiguous grad reduction
- `coalesce_grad_reduction=True`: group grads for single NCCL call (even dp=1 benefits from reduced Python overhead!)
- On single GPU: coalesce_grad_reduction still useful (reduces Python-level grad accumulation overhead)

★★★★★★★ `coalesce_grad_reduction=True` is the ONLY overlap mechanism useful on single GPU. It reduces Python-level per-parameter grad processing overhead by batching into contiguous buffer operations.

### 4.3 PyTorch FSDP2 AC Overlap

★★★★★★★★★ FSDP2 AC (Activation Checkpointing) overlap:

- `use_ac_overlap=True`: unshard next layer's params while computing current layer
- On single GPU: unshard is identity → no overlap benefit

---

## 5. RTX 4090 Training Framework Decision Matrix

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Priority | Framework | Config | Memory (Qwen3-1.7B) | Speed | Ease | Score |
|----------|-----------|--------|---------------------|-------|------|-------|
| #1 | rLLM Tinker | LoRA+bypass | ~9.2GB | Fast | Very easy | ★★★★★★★★★★ |
| #2 | verl+CPPO | bypass_ppo_clip | ~10GB | Medium | Medium | ★★★★★★★ |
| #2.5 | DeepSpeed ZeRO-2 | LoRA+CPU_Adam+bypass | ~10GB | Slow (CPU offload) | Medium | ★★★★★★★ |
| #3 | Megatron core | N/A | ~61GB (full model) | N/A | Hard | ★★★ (not viable) |
| #4 | DeepSpeed ZeRO-3 | N/A | ~61GB | N/A | Hard | ✗ (not viable) |
| #5 | PyTorch FSDP2 | N/A | ~61GB | N/A | Hard | ✗ (not viable) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Decision Criteria for RTX 4090:

1. **LoRA MUST**: Full model training impossible on 24GB → LoRA is mandatory for any training framework
2. **bypass_mode MUST**: Ref model doubles memory → 14-16GB wasted → must use bypass for RL training
3. **No distributed overhead**: Single GPU → all sharding/communication mechanisms are pure overhead
4. **In-process preferred**: Ray/DDP overhead wastes compute → in-process (rLLM Tinker) wins
5. **CPU offload optional**: LoRA params small → CPU_Adam transfer overhead may exceed savings

---

## 6. Key Architectural Insight: Single-GPU Degeneration

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

ALL distributed training frameworks degenerate on single GPU. The fundamental pattern:

```
dp_world_size = 1 → partition_size = full → all-gather/reduce-scatter = identity → pure overhead
```

This applies to:
- Megatron DistributedOptimizer (dp=1 → all params owned, no sharding)
- DeepSpeed ZeRO (dp=1 → partition_size=full, get_data_parallel_partitions returns [full_tensor])
- PyTorch FSDP2 (dp_mesh size=1 → DTensor is local tensor, no sharding)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

**The ONLY thing that works on single GPU is LoRA + in-process training.**

Full model training on single GPU requires:
- 2Ψ (params) + 2Ψ (grads) + 12Ψ (optimizer) + 2Ψ (ref model) = 18Ψ ≈ 61GB for 1.7B → EXCEEDS 24GB!

LoRA + bypass reduces to:
- 2Ψ (frozen params) + ~0.6Ψ (LoRA grads) + ~1.2Ψ (LoRA optimizer) + 0 (ref) = ~3.8Ψ ≈ 13GB → FITS 24GB!

---

## 7. Practical Advice for RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 7.1 For GRPO/RL Training

Use rLLM Tinker (in-process, auto LoRA, bypass_mode default):
```bash
bash tools/train_tinker_rtx4090.sh --model Qwen3-1.7B --task math
```

### 7.2 For verl GRPO Training

Use verl + CPPO + bypass_mode:
```yaml
actor:
  bypass_mode: true
  ppo_clip_ratio: 0.2
algorithm:
  type: cppo  # Better bound than GRPO
```

### 7.3 For DeepSpeed Fine-tuning (non-RL)

Use ZeRO-2 + CPU_Adam + LoRAOptimizedLinear:
```json
{
  "zero_optimization": { "stage": 2 },
  "optimizer": { "type": "CPUAdam" },
  "lora": { "enabled": true, "rank": 32 }
}
```

### 7.4 AVOID on RTX 4090

- DeepSpeed ZeRO-3 (pure overhead, no benefit on single GPU)
- PyTorch FSDP2 (pure overhead, no benefit on single GPU)
- Megatron DistributedOptimizer (no benefit, potential crash #5203)
- Full model training (memory exceeds 24GB)
- FP8 quantization on SM89 (all paths crash)
- FlashAttention FA3/FA4 (SM90+ only)

---

## 8. Source References

| Note | Lines | Key Content |
|------|-------|-------------|
| megatron-core-training-architecture-source-reading.md | 764 | DDP backward post-hook, 16+ PG, DistOpt useless on single GPU |
| deepspeed-zero-single-gpu-source-reading.md | 974 | ZeRO partition_size=full, CPU_Adam 5-7x, LoRAOptimizedLinear split |
| pytorch-fsdp2-internals-reading.md | varies | DTensor per-param, composable API, useless single GPU |
| vllm-v1-scheduler-budgetrefiner-integration.md | 612 | 3 integration points, BudgetRefiner GPU-generic |
| rllm-tinker-source-reading.md | varies | In-process, auto LoRA, bypass, zero-copy |
