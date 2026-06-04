---
name: megatron-nav
description: Navigate Megatron-LM source code. Quick reference for subsystem locations, key classes, and data flow across TP/PP/SP/Communication/Optimizer/Checkpoint/DataLoader/MoE modules.
triggers:
  - megatron source
  - megatron architecture
  - megatron code location
  - megatron tp
  - megatron tensor parallel
  - megatron pp
  - megatron pipeline
  - megatron pipeline parallel
  - megatron 3d parallel
  - megatron sequence parallel
  - megatron communication
  - megatron optimizer
  - megatron checkpoint
  - megatron dataloader
  - megatron moe
  - megatron expert parallel
  - megatron 源码
  - megatron 张量并行
  - megatron 流水线并行
  - megatron 3d并行
  - megatron 序列并行
  - megatron 通信原语
  - megatron 优化器
  - megatron 检查点
  - megatron 数据加载
  - megatron 混合专家
---

# Megatron-LM Source Code Navigator

## Architecture: `megatron/core/`

```
megatron/core/
  ├── tensor_parallel/        # TP layers + communication primitives
  ├── pipeline_parallel/      # PP schedules + P2P communication
  ├── parallel_state.py       # Process group management (TP/PP/DP/EP/CP)
  ├── optimizer/              # Distributed optimizer + ZeRO
  ├── datasets/               # DataLoader + indexed dataset
  └── transformer/
      └── moe/                # MoE router + dispatcher + experts
```

## Parallelism Dimensions

| Type | Split Dim | Communication | Scale |
|------|-----------|--------------|-------|
| TP | Weight matrix | AllReduce (intra-layer) | 2-8 GPUs |
| PP | Model layers | P2P (inter-stage) | 4-16 stages |
| DP | Data batch | AllReduce (gradient) | Any |
| SP | Sequence length | AllGather/ReduceScatter | With TP |
| EP | MoE experts | All-to-All | MoE models |
| CP | Long sequence | AllGather/Ring Attention | Ultra-long |

Rank mapping: `global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size`

---

## Subsystem Index

### 1. Tensor Parallelism (TP)
- **Directory**: `megatron/core/tensor_parallel/`
- **Notes**: `notebook/projects/megatron-tp-source-reading.md` + `notebook/projects/megatron-tp-communication.md`

| File | Key Classes/Functions | Description |
|------|----------------------|-------------|
| `layers.py` | `ColumnParallelLinear` (L770+), `RowParallelLinear` (L1134+), `VocabParallelEmbedding` | Weight-sharded linear layers |
| `mappings.py` | `_CopyToModelParallelRegion`, `_ReduceFromModelParallelRegion`, `_GatherFromModelParallelRegion`, `_ScatterToModelParallelRegion` | Autograd-wrapped communication primitives |
| `mappings.py` | `_ScatterToSequenceParallelRegion`, `_GatherFromSequenceParallelRegion` | SP-specific primitives (split first_dim) |
| `random.py` | `CudaRNGStatesTracker` | TP-aware RNG state management |
| `cross_entropy.py` | `VocabParallelCrossEntropy` | Sharded logits softmax (3x AllReduce) |

**TP Layer Pattern**: ColumnParallel (split output dim) -> activation -> RowParallel (split input dim + AllReduce). Each Transformer layer needs 2x AllReduce (Attention output + MLP FC2).

**Communication Primitives** (each is `autograd.Function`):

| Primitive | Fwd | Bwd | Used By |
|-----------|-----|-----|---------|
| `copy_to_tp_region` | Identity | AllReduce | ColumnParallel input |
| `reduce_from_tp_region` | AllReduce | Identity | RowParallel output |
| `gather_from_tp_region` | AllGather(last_dim) | ReduceScatter(last_dim) | ColumnParallel output |
| `scatter_to_tp_region` | Split(last_dim) | AllGather(last_dim) | RowParallel input |

**Weight Init**: `initialize_affine_weight_gpu` (production, per-shard GPU init) vs `initialize_affine_weight_cpu` (debug, full weight then split). TP metadata: `weight.tensor_model_parallel=True`, `weight.partition_dim`, `weight.partition_stride`.

### 2. Pipeline Parallelism (PP)
- **Directory**: `megatron/core/pipeline_parallel/`
- **Notes**: `notebook/projects/megatron-pp-schedules-reading.md`

| File | Line | Key Functions/Classes | Description |
|------|------|----------------------|-------------|
| `schedules.py` | L48-100 | `get_forward_backward_func` | Entry point dispatcher |
| `schedules.py` | L2074+ | `forward_backward_pipelining_without_interleaving` | Standard 1F1B (non-interleaved) |
| `schedules.py` | L2268-2303 | Warmup forward loop | Stage-dependent warmup count |
| `schedules.py` | L2313-2383 | 1F1B steady state loop | Alternating forward + backward |
| `schedules.py` | L2385-2409 | Cooldown backward loop | Drain remaining backwards |
| `schedules.py` | L931+ | `forward_backward_pipelining_with_interleaving` | Virtual Pipeline (VP) interleaved |
| `p2p_communication.py` | - | `P2PCommunicator` | P2P send/recv wrapper |

**Three Schedules**: Non-Interleaved 1F1B (standard), Interleaved 1F1B (VP, bubble reduces 1/V), Combined 1F1B (flexible overlap).

**Bubble formula**: `(stages - 1) / (microbatches + stages - 1)`. Interleaved: bubble reduces by 1/V (V = virtual stages).

**P2P Primitives**: `recv_forward`, `send_forward`, `send_forward_recv_backward`, `send_backward_recv_forward`, `send_backward`, `recv_backward`. Combined ops reduce sync points. `ring_exchange` is fastest (batched isend/irecv).

**Memory**: `deallocate_output_tensor` frees activations after send. Activation checkpointing based on in-flight backward count.

### 3. Sequence Parallelism (SP)
- **Directory**: `megatron/core/tensor_parallel/mappings.py` (SP primitives)
- **Notes**: `notebook/projects/megatron-tp-source-reading.md` (Section 2)

SP replaces TP AllReduce with AllGather + ReduceScatter, splitting activations along sequence dim. LayerNorm/Dropout run independently on each shard. Same total communication volume as AllReduce, but activation memory saved by TP factor.

SP Primitives: `_ScatterToSequenceParallelRegion` (fwd: split first_dim, bwd: AllGather first_dim), `_GatherFromSequenceParallelRegion` (fwd: AllGather first_dim, bwd: ReduceScatter first_dim).

### 4. Communication
- **Directory**: `megatron/core/tensor_parallel/mappings.py` + `megatron/core/parallel_state.py`
- **Notes**: `notebook/projects/megatron-tp-communication.md`

**Async overlap**: `CUDA_DEVICE_MAX_CONNECTIONS=1` forces NCCL ops to one stream, enabling async communication to overlap with compute. Used with `async_op=True` + `handle.wait()`.

**Gradient Accumulation Fusion**: `LinearWithGradAccumulationAndAsyncCommunication` merges gradient GEMM + accumulation into single CUDA kernel (`fused_wgrad_gemm_accum_fp32`). Saves one memory roundtrip.

**Process Groups** (`parallel_state.py`): `TENSOR_MODEL_PARALLEL_GROUP`, `PIPELINE_MODEL_PARALLEL_GROUP`, `DATA_PARALLEL_GROUP`, `EMBEDDER_MODEL_PARALLEL_GROUP`, `_TENSOR_AND_CONTEXT_PARALLEL_GROUP`.

### 5. Optimizer + Checkpoint
- **Directory**: `megatron/core/optimizer/`
- **Notes**: `notebook/projects/megatron-optimizer-ckpt.md`

| File | Key Classes | Description |
|------|------------|-------------|
| `optimizer.py` | `MixedPrecisionOptimizer` | Base: FP16/BF16 fwd + FP32 optimizer |
| `distrib_optimizer.py` (~3000 lines) | `DistributedOptimizer` | ZeRO-1 sharded optimizer states |
| `grad_scaler.py` | `MegatronGradScaler` | FP16 AMP loss scaling (BF16 not needed) |
| `clip_grads.py` | - | Global gradient norm clipping |
| `param_layout.py` | `_ParamAndGradBuffer` | Bucketed parameter buffer management |
| `optimizer_config.py` | `OptimizerConfig` | Optimizer configuration |
| `cpu_offloading/` | `HybridDeviceOptimizer` | CPU offloading |

**DistributedOptimizer = ZeRO-1**: Shards Adam (m, v) across DP ranks. Each rank stores 12/DP bytes/param for optimizer states.

**Step flow**: AllReduce grads (bucket-level, overlap with next bucket) -> unscale -> clip -> optimizer.step() on local shard -> AllGather updated params.

**Checkpoint formats**: `fully_reshardable` (default), `dp_zero` (fastest), `dp_reshardable`, `fully_sharded_model_space`, `fsdp_dtensor`. Re-sharding supported when DP size changes.

**Memory (per param)**: No ZeRO = 16B. ZeRO-1 (DP=8) = 5.5B. ZeRO-2 (DP=8) = 3.75B. ZeRO-3 (DP=8) = 2B + activations.

### 6. DataLoader
- **Directory**: `megatron/core/datasets/` (16 files)
- **Notes**: `notebook/projects/megatron-dataloader.md`

| File | Key Classes | Description |
|------|------------|-------------|
| `indexed_dataset.py` | `IndexedDataset` | MMap binary format (.idx + .bin), zero-copy, O(1) random access |
| `gpt_dataset.py` | `GPTDataset` | Sequence packing + document boundary detection |
| `blended_dataset.py` | `BlendedDataset` | Multi-dataset weighted mixing |
| `blended_megatron_dataset_builder.py` | `BlendedMegatronDatasetBuilder` | Unified dataset builder |

**DataLoader config**: `num_workers=4`, `pin_memory=True` (zero-cost), `prefetch_factor=2`. DataLoader overhead <3% of training time.

**Data splitting**: DP = different data per rank (distributed sampler), TP = same data, PP = same micro-batch (different chunks), CP = sequence split.

### 7. MoE (Mixture of Experts)
- **Directory**: `megatron/core/transformer/moe/` (13 files)
- **Notes**: `notebook/projects/megatron-moe-architecture.md`

| File | Key Classes | Description |
|------|------------|-------------|
| `router.py` | `Router` | Top-K gating, aux loss, token dropping |
| `token_dispatcher.py` | `TokenDispatcher` | Dispatch (permute + all-to-all) + Combine (all-to-all + unpermute) |
| `experts.py` | `GroupedMLP` | Batched expert computation via grouped GEMM |
| `moe_layer.py` | `MoELayer` | MoE Layer wrapper |
| `fused_a2a.py` | `fused_dispatch`, `fused_combine` | Fused All-to-All (DeepEP backend) |
| `shared_experts.py` | `SharedExpertMLP` | Shared expert (DeepSeek-V2/V3 style) |
| `moe_utils.py` | - | Routing utilities, capacity calculation |

**Pipeline**: Router (Top-K) -> Dispatch (permute + A2A) -> Experts (grouped GEMM) -> Combine (A2A + unpermute). 2x All-to-All per MoE layer.

**EP config**: EP=1 all experts local, EP=E each rank 1 expert (max communication). TP+EP hybrid: TP before router, EP after router, TP after combine.

**DeepEP optimization**: NVLink RDMA, SM control (reserve SMs for communication), async dispatch/combine overlap with compute.

---

## Key Data Flow

```
Input Tokens
  -> VocabParallelEmbedding (AllReduce)
  -> Transformer Layer x N:
      -> Attention:
          ColumnParallelLinear (QKV, gather=False) -> FlashAttention -> RowParallelLinear (AllReduce)
      -> MLP:
          ColumnParallelLinear (gate+up, gather=False) -> SwiGLU -> RowParallelLinear (AllReduce)
      [SP: AllGather before ColumnParallel, ReduceScatter after RowParallel]
  -> VocabParallelCrossEntropy (3x AllReduce)
```

## Critical Paths

- **TP Forward**: ColumnParallel -> local compute -> RowParallel -> AllReduce (1x per sub-layer)
- **TP Backward**: Autograd handles communication via Function.backward (1x AllReduce per sub-layer)
- **PP Schedule**: Warmup forward -> 1F1B steady (forward + backward alternating) -> Cooldown backward
- **SP Flow**: AllGather -> ColumnParallel -> local -> RowParallel -> ReduceScatter (replaces AllReduce, saves activation memory)
- **MoE Flow**: Router -> Dispatch (A2A) -> Grouped GEMM -> Combine (A2A) -> Shared expert add
- **Optimizer Step**: AllReduce grads -> clip -> local shard optimizer.step() -> AllGather params
