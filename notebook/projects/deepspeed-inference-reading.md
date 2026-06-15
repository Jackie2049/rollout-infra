# DeepSpeed 推理优化架构深度分析 (V2 Update)

> 作者: Claude + jackie2049 | 日期: 2026-06-15 (原版 2026-06-05, V2 源码级更新)
> 论文: DeepSpeed-FastGen (2023), FP6-LLM (arXiv:2401.14112), ZeroQuant 系列
> 源码: https://github.com/deepspeedai/DeepSpeed | https://github.com/deepspeedai/DeepSpeed-MII
> 版本: DeepSpeed v0.19.1 (2026-05-27), DeepSpeed-MII v0.3.3 (2025-03)
> 用途: AI Infra 工程师学习笔记, 覆盖 ZeRO-Inference/V2 Engine/FastGen/量化/Kernel/TP/RTX4090

---

## 目录

1. [ZeRO-Inference: 超大模型推理](#1-zero-inference-超大模型推理)
2. [InferenceEngineV2: Ragged Batching 新引擎](#2-inferenceenginev2-ragged-batching-新引擎)
3. [DeepSpeed-FastGen 推理服务系统与 MII](#3-deepspeed-fastgen-推理服务系统与-mii)
4. [Dynamic SplitFuse: 动态分词融合调度](#4-dynamic-splitfuse-动态分词融合调度)
5. [量化支持: FP6/INT8/INT4/W4A16](#5-量化支持-fp6int8int4w4a16)
6. [Kernel 优化: V2 模块化 + V1 融合算子](#6-kernel-优化-v2-模块化--v1-融合算子)
7. [推理 Tensor Parallelism 与训练 TP 的差异](#7-推理-tensor-parallelism-与训练-tp-的差异)
8. [与 vLLM/SGLang 的对比分析](#8-与-vllmsglang-的对比分析)
9. [RTX 4090 实战可行性](#9-rtx-4090-实战可行性)
10. [2025-2026 最新特性与现状](#10-2025-2026-最新特性与现状)

---

## 1. ZeRO-Inference: 超大模型推理

### 1.1 问题: 模型大于单 GPU 内存

训练中 ZeRO-3 通过参数分区+按需gather使每GPU只存 1/N 的参数。推理能否复用这一机制?

**核心思路**: ZeRO-3 的参数分区+CPU offloading 同样适用于推理, 将无法全部放入GPU的模型参数offload到CPU RAM, 每次forward时按需从CPU gather到GPU。

### 1.2 V1 InferenceEngine 的 ZeRO 集成

源码位置: `deepspeed/inference/engine.py` (628行) + `deepspeed/inference/config.py`

**关键配置字段**:

```python
class DeepSpeedInferenceConfig(DeepSpeedConfigModel):
    zero: DeepSpeedZeroConfig = {}
    """
    ZeRO configuration to use with the Inference Engine.
    """

    keep_module_on_host: bool = False
    """
    When loading checkpoints, keep them on host (CPU) instead of moving
    to device. Essential for models larger than GPU memory - allows
    quantization before moving to device.
    """

    dtype: torch.dtype = torch.float16  # 支持 fp16/bf16/int8
    enable_cuda_graph: bool = False
    quant: QuantizationConfig = {}  # INT8 MoQ 量化
```

**ZeRO-Inference 工作流程**:

```
=== ZeRO-Inference (V1 Engine) ===

初始化阶段:
  1. 模型参数 → ZeRO-3 分区 → 每GPU只存 partition_size = param_size/N
  2. keep_module_on_host=True → 参数先在CPU, 不直接移到GPU
  3. 可选量化: QuantizationContext + groupwise weight quantization
     (ZeRO-3下: module.weight.all_gather() → quantize → module.weight.partition())

Forward阶段:
  每次forward:
  1. AllGather 参数分区 → 临时恢复完整参数到GPU
  2. 执行 forward 计算
  3. 释放gather的参数 → GPU内存回到partition级别
  4. ZeRO-3 partition → 参数回到1/N大小

代价:
  - 每层每次forward = 2次AllGather(attention+MLP) + 2次partition
  - 通信开销: 2 × param_size × 4B (bf16) × 层数
  - For 7B model, TP=1: 每层 ~28MB × 32层 × 2 = ~1.8GB通信/forward
  - For 70B model: ~280MB × 80层 × 2 = ~45GB通信/forward (极慢!)

=== CPU Offloading 变体 ===

如果 GPU 内存不足以放一个partition:
  1. 参数分区存储在 CPU RAM (ZeRO-Infinity)
  2. forward时: CPU→GPU H2D copy 按需加载
  3. 计算后: GPU→CPU 释放
  4. 代价更高: PCIe带宽瓶颈 (~32GB/s PCIe Gen4 vs ~900GB/s NVLink)
```

### 1.3 ZeRO-Inference 的量化协同

源码位置: `deepspeed/inference/quantization/quantization.py`

**关键**: ZeRO-3 推理可以与groupwise weight quantization结合:

```python
# _init_group_wise_weight_quantization() 核心逻辑:
is_zero3_enabled = ds_config['zero_optimization']['stage'] == 3
is_offloading_enabled = ds_config['zero_optimization']['offload_param']

if is_zero3_enabled:
    with QuantizationContext(config_dict_or_path=ds_config, param_swapper=nvme_swapper):
        # 1. module.weight.all_gather() → 恢复完整权重
        # 2. quantize → QuantizedLinear 替换
        # 3. module.weight.partition() → 回到分区状态
        # 4. 量化后的权重更小 → partition进一步缩小 → GPU内存释放更多

        # 按大小排序: 先量化小层, 再量化大层 (及时 gc.collect() 防OOM)
        module_list.sort(key=lambda named_module: named_module[1].weight.numel())
```

**QuantizedLinear**: 运行时dequantize:

```python
class QuantizedLinear(nn.Linear):
    def forward(self, input):
        # 1. 从concat存储中拆出 quantized_weight + quant_scale + quant_min
        quantized_weight, quant_scale, quant_min = self.weight.deconcat(self.weight)
        # 2. 运行时 dequantize (INT8 → FP16)
        temp_dequantized_weight = self.weight.dequantizer.dequantize(
            quantized_weight.view(torch.uint8), quant_scale, quant_min)
        # 3. 禁止 torch.functional.linear (ZeRO-3会替换为LinearFunctionForZeroStage3,
        #    临时buffer会导致内存泄漏!)
        return torch._C._nn.linear(input, temp_dequantized_weight, self.bias)
```

### 1.4 ZeRO-Inference 延迟影响分析

```
=== 延迟模型 ===

单GPU无offload (模型<GPU内存):
  T_forward = N_layers × (T_compute + 0)  ← 纯计算
  7B BF16 单GPU: ~10ms/forward (decode)

ZeRO-3 + AllGather (模型>单GPU, 多GPU):
  T_forward = N_layers × (T_compute + T_allgather + T_partition)
  7B TP=2 PCIe: AllGather ~28MB/层 × ~10us/MB × 2 = ~560us/层
                32层 × ~560us ≈ 18ms 通信开销
                总延迟: ~28ms/forward (比无offload慢~3x)

ZeRO-Infinity + CPU Offload (单GPU,模型极大):
  T_forward = N_layers × (T_compute + T_H2D + T_D2H)
  PCIe Gen4带宽: ~32GB/s (实际~25GB/s考虑协议开销)
  70B 1GPU CPU offload: 每层参数 ~280MB × 2 = ~560MB/层
                         H2D时间: ~560MB/25GB/s ≈ 22ms/层
                         80层 × 22ms ≈ 1.8s 通信开销 alone!
                         → 完全不可行作为交互式推理

结论:
  ★★★ ZeRO-Inference 延迟极高, 仅适合batch/offline推理场景
  ★★★ 对于低延迟交互推理, 必须用量化压缩模型到单GPU内存内
  ★★★ RTX 4090 24GB: INT4量化后7B模型仅需~4GB → 完全不需要ZeRO-Inference
```

### 1.5 V2 Engine: 无 ZeRO 支持

源码确认: `InferenceEngineV2` (RaggedInferenceEngineConfig) 中没有 ZeRO 配置字段。V2 引擎专注高速ragged batching, 不支持参数offloading, 假设模型参数全部在GPU上。

```
=== V1 vs V2 Inference Engine ===

V1 InferenceEngine (deepspeed/inference/engine.py, 628行):
  - 支持ZeRO-3配置 (zero: DeepSpeedZeroConfig)
  - 支持CPU offload (keep_module_on_host)
  - 支持INT8 MoQ量化 + groupwise weight quantization
  - 支持CUDA Graph (enable_cuda_graph)
  - 支持torch.compile (compile backend)
  - 固定batch推理 (无连续batching)
  - TP: AutoTP自动解析 + 手动injection_policy

V2 InferenceEngineV2 (deepspeed/inference/v2/engine_v2.py, ~200行):
  - ★ 无ZeRO配置 (假设模型全在GPU)
  - ★ Ragged batching (连续batching)
  - ★ BlockedKVCache (非连续内存, block_size=128)
  - ★ DSStateManager (sequence tracking + scheduling)
  - ★ Atom-based BlockedFlashAttn (FlashAttention V2修改版)
  - ★ MoE: scatter/gather + top_k_gating + CUTLASS MoEGEMM
  - ★ FP6 quantization: QuantizedWf6Af16Linear (wf6af16模式)
  - ★ 11种模型Policy (OPT/Llama2/Mistral/Mixtral/Falcon/Phi/Phi3/Qwen/Qwen2/Qwen2MoE/Exaone4)
  - ★ DSModuleRegistry (attention/linear/moe/norm/embeb/unembed模块注册)
```

---

## 2. InferenceEngineV2: Ragged Batching 新引擎

### 2.1 架构总览

源码位置: `deepspeed/inference/v2/` (完整子目录结构)

```
deepspeed/inference/v2/
├── engine_v2.py          ← InferenceEngineV2 (主引擎)
├── engine_factory.py     ← build_hf_engine() / build_engine_from_ds_checkpoint()
├── config_v2.py          ← RaggedInferenceEngineConfig (TP+StateManager+Quantization)
├── inference_parameter.py ← InferenceParameter (权重存储,含FP6的2bit+4bit+scales)
├── inference_utils.py    ← DtypeEnum, ActivationType, elem_size
├── scheduling_utils.py   ← SchedulingResult (5种结果) + SchedulingError
├── logging.py            ← inference_logger()
├── allocator.py          ← empty_from() (view-based输出buffer)
├── model_implementations/ ← 11种模型Policy + 基类
│   ├── inference_model_base.py   ← DSInferenceModelBase (ABC)
│   ├── inference_policy_base.py  ← InferenceV2Policy + POLICIES全局注册
│   ├── inference_transformer_base.py
│   ├── flat_model_helpers.py     ← 参数序列化
│   ├── parameter_base.py
│   ├── sharding/                 ← TP分片实现
│   │   ├── attn.py, attn_out.py, mlp.py, qkv.py, embedding.py, unembed.py
│   ├── llama_v2/, mistral/, mixtral/, opt/, falcon/, phi/, phi3/,
│   │   qwen/, qwen_v2/, qwen_v2_moe/, exaone4/
│   ├── AddingAModel.md           ← 添加新模型的指南
├── modules/              ← DSModule 可插拔模块系统
│   ├── ds_module.py      ← DSModuleConfig + DSModuleRegistry
│   ├── configs/          ← DSLinearConfig, DSSelfAttentionConfig, DSMoEConfig, etc
│   ├── interfaces/       ← DSLinearBase, DSSelfAttentionBase, DSMoEBase + Registry
│   ├── implementations/
│   │   ├── attention/dense_blocked_attention.py ← ★ DSDenseBlockedAttention
│   │   ├── linear/blas_fp_linear.py, quantized_linear.py ← ★ QuantizedWf6Af16Linear
│   │   ├── moe/cutlass_multi_gemm.py ← ★ DSMultiGemmMoE (CUTLASS ragged GEMM)
│   │   ├── embedding/, post_norm/, pre_norm/, unembed/
├── kernels/              ← DSKernelBase 模块化kernel层
│   ├── ds_kernel.py      ← ABC: __init__(compile+warmup) + __call__(execute)
│   ├── core_ops/         ← 基础kernel
│   │   ├── bias_activations/  ← CUDABiasActivation
│   │   ├── blas_kernels/      ← BlasLibLinear
│   │   ├── cuda_layer_norm/   ← CUDALayerNorm
│   │   ├── cuda_rms_norm/     ← CUDARMSNorm
│   │   ├── cuda_linear/       ← CUDAWf6Af16Linear (FP6)
│   │   ├── gated_activations/ ← CUDAGatedActivation
│   ├── cutlass_ops/      ← CUTLASS 高级kernel
│   │   ├── mixed_gemm/  ← MixedGEMM (INT4/INT8 WxA16, MoE GEMM)
│   │   ├── moe_gemm/    ← MoEGEMM (expert-level ragged GEMM)
│   ├── ragged_ops/       ← Ragged batching专用kernel (C++ + CUDA)
│   │   ├── blocked_flash/     ← ★ BlockedFlashAttn (atom-based FlashAttention)
│   │   ├── linear_blocked_kv_rotary/ ← BlockedRotaryEmbeddings / BlockedTrainedRotaryEmbeddings
│   │   ├── moe_scatter/       ← ★ MoEScatter (token→expert重排)
│   │   ├── moe_gather/        ← ★ MoEGather (expert→原序恢复+scale)
│   │   ├── top_k_gating/      ← ★ RaggedTopKGating (top-k router, k=1/2/4/8)
│   │   ├── logits_gather/     ← ★ RaggedLogitsGather (最终token gather)
│   │   ├── embed/, ragged_helpers/, atom_builder/
├── ragged/               ← Continuous batching 核心数据结构
│   ├── ragged_manager.py  ← ★ DSStateManager (序列跟踪+KV cache+调度)
│   ├── ragged_wrapper.py  ← ★ RaggedBatchWrapper (ragged batch元数据)
│   ├── blocked_allocator.py ← BlockedAllocator (linked-list free block管理)
│   ├── kv_cache.py         ← ★ BlockedKVCache (GPU KV cache + TP同步)
│   ├── manager_configs.py  ← DSStateManagerConfig + KVCacheConfig + MemoryConfig
│   ├── sequence_descriptor.py ← DSSequenceDescriptor + PlaceholderSequenceDescriptor
│   ├── csrc/               ← C++ ragged_utils实现 (fast_host_buffer等)
│   ├── includes/           ← C++ 头文件
├── checkpoint/            ← HuggingFaceCheckpointEngine
```

### 2.2 DSStateManager: 连续Batching核心

源码: `deepspeed/inference/v2/ragged/ragged_manager.py`

```python
class DSStateManager:
    _config: DSStateManagerConfig        # 调度限制
    _kv_configs: Tuple[KVCacheConfig]    # KV cache配置
    _kv_cache: BlockedKVCache            # GPU KV cache存储
    _seqs: Dict[int, DSSequenceDescriptor]  # 序列跟踪容器
    _tracking_allocator: BlockedAllocator    # 序列slot分配
    _all_block_ids: Tuple[torch.Tensor, ...]     # GPU端block ID
    _all_block_ids_shadow: Tuple[torch.Tensor, ...]  # CPU端shadow (pin_memory)
```

**配置参数**:

```python
class DSStateManagerConfig(DeepSpeedConfigModel):
    max_tracked_sequences: int = 2048     # 最大跟踪序列数
    max_ragged_batch_size: int = 768      # 单次forward最大token数
    max_ragged_sequence_count: int = 512  # 单次forward最大序列数
    max_context: int = 8192               # 单序列最大token数
    memory_config: MemoryConfig           # KV cache内存策略
    offload: bool = False                 # KV cache CPU offload (★ 目前 NotImplementedError!)
```

**KV Cache配置**:

```python
class KVCacheConfig(DeepSpeedConfigModel):
    type: KVCacheType = KVCacheType.DENSE  # dense / local (窗口attention)
    block_size: int = 128                   # ★ 每block 128 tokens (vs vLLM V1=16)
    num_allocation_groups: int = 1          # 分配组数 (GPTNeo交替local/global=2)
    cache_shape: Tuple[int, int, int]      # (num_caches, num_heads, head_size)
    cache_dtype: DtypeEnum = DtypeEnum.fp16
    max_blocks_per_allocation_group: int = 64  # 每组最大block数
```

### 2.3 BlockedKVCache: GPU KV Cache实现

源码: `deepspeed/inference/v2/ragged/kv_cache.py`

```
KV Cache 存储格式 (5D tensor):
  [num_blocks, block_size, 2, num_heads_kv, head_size]

  split_kv() → K: [num_blocks, block_size, num_heads_kv, head_size]
                V: [num_blocks, block_size, num_heads_kv, head_size]

内存分配策略 (MemoryConfig.AllocationMode):
  RESERVE: 计算GPU剩余内存 → 自动计算num_blocks
    available_kv_memory = accelerator.available_memory() - reserve_size
    num_blocks = available_kv_memory / total_per_block_footprint
    ★ TP同步: dist.all_reduce(num_blocks, op=MIN) → 确保所有rank一致
    ★ NCCL预热: dummy all_reduce before memory calculation (H100观察到NCCL占用更高)

  ALLOCATE: 用户指定num_blocks (需要手动调优)
```

### 2.4 BlockedAllocator: Linked-List Block管理

源码: `deepspeed/inference/v2/ragged/blocked_allocator.py`

```
BlockedAllocator: CPU端linked-list管理free blocks
  _blocks: torch.Tensor (int32, pin_memory=True)  ← 下一个free block索引
  _head: int                                        ← linked-list头
  _free_blocks: int                                 ← 剩余free block数

  allocate(num_blocks): O(num_blocks) 从linked-list取出
  free(blocks): O(num_blocks) 归还到linked-list, 校验block有效性

  ★ 对比 vLLM V1: FreeKVCacheBlockQueue (双向链表, O(1)中间删除, prefix hit需要)
  ★ DeepSpeed: 单向链表, 更简单, 但不支持prefix caching中间释放
```

### 2.5 Scheduling: 可调度性检查

源码: `deepspeed/inference/v2/scheduling_utils.py` + `engine_v2.py`

```python
class SchedulingResult(Enum):
    Success = 0
    EngineSequenceLimitExceeded = 1    # 超过max_tracked_sequences
    BatchSequenceLimitExceeded = 2     # 超过max_ragged_sequence_count
    BatchTokenLimitExceeded = 3        # 超过max_ragged_batch_size
    KVCacheLimitExceeded = 4           # KV block不够
    SequenceTokenLimitExceeded = 5     # 单序列超过max_context
```

**调度逻辑** (engine_v2.is_schedulable):

```python
def is_schedulable(self, uids, lengths):
    for uid, length in zip(uids, lengths):
        seq_desc = self._state_manager.get_sequence(uid)
        if seq_desc is None:
            seq_desc = PlaceholderSequenceDescriptor()  # ★ 幽灵描述符,仅用于估算

        sched_len, sched_blocks = self._model.get_kv_requirements(seq_desc, length, free_blocks)
        if sched_len != length:  # KV不够
            return SchedulingResult.KVCacheLimitExceeded
        free_blocks -= sched_blocks
        batch_len += length

    # 检查各类限制
    if cur_seqs > max_tracked_sequences: return EngineSequenceLimitExceeded
    if batch_len > max_ragged_batch_size: return BatchTokenLimitExceeded
    return SchedulingResult.Success
```

### 2.6 RaggedBatchWrapper: Ragged Batch元数据

源码: `deepspeed/inference/v2/ragged/ragged_wrapper.py`

```
RaggedBatchWrapper: 每个forward pass的batch元数据
  _input_ids_shadow / _input_ids            ← token IDs (CPU/GPU双缓冲)
  _batch_metadata_storage / shadow          ← [n_seqs, n_tokens] (int32)
  _token_to_seq_storage / shadow            ← token→序列映射 [0,0,0,0,1,2,2,2]
  _inflight_seq_descriptors / shadow        ← [start, n_tokens, history, X] per序列
  _kv_ptrs / shadow                         ← 每序列的KV block ID指针

  ★ CPU/GPU双缓冲: shadow(CPU pin_memory) → copy_ non_blocking → GPU
  ★ to_padded(): pad到64/128粒度 (≤512 pad到64, >512 pad到128)
  ★ 与vLLM V1对比: vLLM用CPU传递token IDs(D2H+H2D), DS V2用pin_memory shadow
```

### 2.7 DSSequenceDescriptor: 序列状态跟踪

源码: `deepspeed/inference/v2/ragged/sequence_descriptor.py`

```python
class DSSequenceDescriptor(BaseSequenceDescriptor):
    _seen_tokens: int          # 已完成forward的token数
    _in_flight_tokens: int     # 正在forward中的token数
    _max_context: int          # 最大token限制
    _kv_cache_ids: Tuple[torch.Tensor, ...]     # GPU端block IDs
    _kv_cache_ids_shadow: Tuple[torch.Tensor, ...]  # CPU端shadow

    def pre_forward(self, num_tokens):  self._in_flight_tokens = num_tokens
    def post_forward(self):             self._seen_tokens += self._in_flight_tokens; self._in_flight_tokens = 0

    def extend_kv_cache(self, new_ids, cache_group):  # 扩展KV block分配
        shadow_alloc_group[cur_blocks:cur_blocks+new_blocks].copy_(new_group_ids)
        alloc_group[cur_blocks:cur_blocks+new_blocks].copy_(shadow[...], non_blocking=True)
```

---

## 3. DeepSpeed-FastGen 推理服务系统与 MII

### 3.1 系统定位

DeepSpeed-FastGen 是 Microsoft DeepSpeed 团队推出的 LLM 推理服务系统, 通过 **DeepSpeed-MII** (Model Implementations for Inference) 和 **DeepSpeed-Inference** 的协同组合实现高吞吐、低延迟文本生成:

```
┌──────────────────────────────────────────────────────┐
│                 DeepSpeed-FastGen                     │
│            (MII + DeepSpeed-Inference)                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   DeepSpeed-MII │    │  DeepSpeed-Inference     │  │
│  │   (前端服务层)   │    │  (后端推理引擎)          │  │
│  │                 │    │                         │  │
│  │ - pipeline()    │    │ - V2: Ragged Batching   │  │
│  │ - serve()       │───>│ - V1: ZeRO-Inference    │  │
│  │ - client()      │    │ - Blocked KV Cache      │  │
│  │ - Load Balancer │    │ - Atom-based Attention   │  │
│  │ - RESTful API   │    │ - MoE scatter/gather     │  │
│  └─────────────────┘    └─────────────────────────┘  │
│                                                      │
│  ┌─────────────────────────────────────────────────┐  │
│  │           Dynamic SplitFuse Scheduler            │  │
│  │   (MII batching/ragged_batching.py, ZMQ IPC)    │  │
│  └─────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─────────────────────────────────────────────────┐  │
│  │            Blocked KV Cache (V2 Engine)          │  │
│  │       Continuous Batching (iteration-level)      │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 3.2 MII (Model Implementations for Inference)

MII 是开源 Python 库, 提供两层 API:

**非持久化 Pipeline** (适合交互式/测试):

```python
from mii import pipeline
pipe = pipeline("mistralai/Mistral-7B-v0.1")
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
```

**持久化部署** (适合生产环境):

```python
import mii
mii.serve("mistralai/Mistral-7B-v0.1")
client = mii.client("mistralai/Mistral-7B-v0.1")
output = client.generate("DeepSpeed is", max_new_tokens=128)
```

**高级部署选项**:

```python
# Tensor Parallelism
mii.serve("meta-llama/Llama-2-70b", tensor_parallel=4)

# 多副本 + 负载均衡 (线性扩展, 16副本=16x吞吐)
mii.serve("meta-llama/Llama-2-70b", tensor_parallel=4, replica_num=4)

# FP6 量化 (★ 70B 单卡 A100-80GB)
pipe = mii.pipeline("NousResearch/Llama-2-70b-hf", quantization_mode='wf6af16')
```

### 3.3 支持模型 (V2 Engine)

engine_factory.py 中注册的11种模型Policy:

| Policy | 模型类型 | 特点 |
|--------|---------|------|
| OPTPolicy | opt | 基础测试模型 |
| Llama2Policy | llama | Llama-1/2/3 系列 |
| MistralPolicy | mistral | Mistral-7B |
| MixtralPolicy | mixtral | ★ MoE: Mixtral-8x7B |
| FalconPolicy | falcon | Falcon-7B/40B/180B |
| PhiPolicy | phi | Phi-2 |
| Phi3Policy | phi3 | Phi-3 |
| QwenPolicy | qwen | Qwen-1 |
| Qwen2Policy | qwen2 | Qwen-2 |
| Qwen2MoePolicy | qwen2_moe | ★ MoE: Qwen2-MoE |
| Exaone4Policy | exaone4 | ★ 最新: EXAONE 4.0 |

### 3.4 MII Serving层: RaggedBatchBase

源码: `deepspeedai/DeepSpeed-MII/mii/batching/ragged_batching.py`

```
RaggedBatchBase: MII的核心调度循环
  - ZMQ IPC: rank0 PUB → 其他rank SUB (跨TP进程通信)
  - _bcast_requests(): 收集请求,广播到所有TP rank
  - flush(): 清理完成序列的KV cache
  - put(): 将新token输入InferenceEngineV2
  - generate(): 主循环 (调度→flush→put→采样→停止检测→返回)

  与V2 Engine交互:
    engine.is_schedulable() → SchedulingResult
    engine.put(uids, tokens) → logits
    engine.flush(uids) → 释放KV cache
    engine.get_remaining_block_capacity(uid) → 剩余容量

  ★ SplitFuse调度逻辑在MII层实现, V2 Engine只负责forward+KV管理
```

### 3.5 DeepSpeed-Kernels

为避免用户漫长的编译等待, DeepSpeed 团队将自定义 CUDA kernel 预编译为 Python wheel:

- **硬件要求**: NVIDIA GPU, Compute Capability 8.0+ (Ampere: A100, A6000, H100)
- **CUDA 版本**: 11.6+
- **安装**: `pip install deepspeed-mii` 时自动安装为依赖

> ★★★ RTX 4090 SM 8.9 > 8.0 → DeepSpeed-Kernels wheel 兼容

---

## 4. Dynamic SplitFuse: 动态分词融合调度

### 4.1 问题背景: 现有系统的调度缺陷

LLM 推理有两个阶段:

1. **Prefill (提示处理)**: compute-bound, 一次处理大量token
2. **Decode (生成)**: memory-bound, 每步1 token

```
=== vLLM (2023): Preemption ===
  长 prompt 处理时, 所有 decode 被暂停 → 生成延迟飙升 (P95 毛刺)

=== Orca: 混合策略 ===
  完整 prompt 加入 batch → forward pass 大小不均 → 性能波动

=== vLLM V1 (2024+): Chunked Prefill ===
  改进版: prompt分chunk, 但仍有抢占+num_computed_tokens=0全重置

=== DeepSpeed-FastGen: Dynamic SplitFuse ===
  固定token budget, 长prompt分chunk + 短prompt拼接 → decode从不暂停
```

### 4.2 三个关键性能洞察

**洞察 1**: Token数量是性能决定因素, 序列数可忽略
**洞察 2**: 吞吐量曲线呈凹函数, 存在饱和点 (memory-bound→compute-bound陡峭过渡)
**洞察 3**: 凹函数等分最优 → `2f(x) >= f(x+h) + f(x-h)` → 均分token到各forward最优

### 4.3 Dynamic SplitFuse 策略

```
=== Dynamic SplitFuse ===
目标: 每个 forward pass 固定 Token Budget = B

时间 →     T1          T2          T3          T4          T5
         ┌───────────┬───────────┬───────────┬───────────┬──────┐
  请求A  │ 64 decode │ 64 decode │ 64 decode │ 64 decode │ 64 d │
  请求B  │ 128 prefill│ 128 prefill│ 128 prefill│ 64 prefill│      │
  (长)   │  chunk 1  │  chunk 2  │  chunk 3  │  chunk 4  │ (完) │
  请求C  │ 64 prefill│           │           │           │ 64 d │
  (短)   │ (完整)    │           │           │           │      │
  总计   │ =256 tok  │ =192 tok  │ =192 tok  │ =128 tok  │=128  │

关键:
1. 长 prompt → 多个小chunk + decode混合
2. 短 prompt → 填入剩余空间
3. Decode 从不暂停 → P95延迟稳定
```

### 4.4 性能收益

| 指标 | Dynamic SplitFuse | vLLM (2023版) | 提升 |
|------|-------------------|---------------|------|
| **有效吞吐** | 1.42 qps (70B/A100x4) | 0.63 qps | **2.3x** |
| **P95 延迟** | ~110ms | ~400ms | **3.7x 降低** |
| **SLA 违规率** | <1% | 28% | 显著改善 |

> 注意: 2023年数据。vLLM V1 (2024+) 已有Chunked Prefill,差距缩小但仍存在。

---

## 5. 量化支持: FP6/INT8/INT4/W4A16

### 5.1 V2 Engine 量化: QuantizedWf6Af16Linear

源码: `deepspeed/inference/v2/modules/implementations/linear/quantized_linear.py`

**★ V2 Engine 当前唯一量化模式: wf6af16 (FP6 weight + FP16 activation)**

```python
@DSLinearRegistry.register_module
class QuantizedWf6Af16Linear(DSLinearBase):
    """
    FP6 weight-only quantization: weight is FP6, activation is FP16.
    8 FP6 data items packed in 3 FP16 tensors → 2bit + 4bit 分离存储
    """

    def transform_param(self, param):
        # 1. fp_quantize(param, num_bits=6, exp_bits=3) → quantized_fake_fp6 + scales
        # 2. preprocess_weight(quantized_fake_fp6) → weights_2bit + weights_4bit
        return InferenceParameter.initialize(weights_2bit, weights_4bit=weights_4bit, scales=scales)

    def forward(self, hidden_states, w, b):
        weights_2bit = w
        weights_4bit = w.weights_4bit  # ★ InferenceParameter扩展属性
        scales = w.scales
        self._linear_impl(output, hidden_states, weights_2bit, weights_4bit, scales, ...)
```

**fp_quantize()**: 使用qtorch的float_quantize实现FP6 (3 exp bits + 2 man bits):

```python
def fp_quantize(input, num_bits=6, exp_bits=3, group_size=-1):
    # qtorch.float_quantize: per-group quantization
    # 8 fp6 items → packed into 3 fp16 values (2bit+4bit分离)
    # scales: per-output-channel FP16
    return quantized_fake_fp6, scales
```

**V2 Config**:

```python
class QuantizationConfig(DeepSpeedConfigModel):
    quantization_mode: Optional[str] = None
    # 支持模式: 'wf6af16' (FP6 weight + FP16 activation)
```

### 5.2 V1 Engine 量化: INT8 MoQ + Groupwise Weight Quantization

源码: `deepspeed/inference/quantization/` + `deepspeed/runtime/weight_quantizer.py`

**INT8 MoQ (Model-Optimized Quantization)**:

```python
class WeightQuantization:
    def quantize_data(self, data, quantize_bits=8, groups=1):
        # Per-group symmetric INT8 quantization
        # max_d = max(g.max(), g.min().abs()) per group
        # data_scale = (1 << 8) / (2 * max_d + 1e-5)
        # data_int = (g * scale).round().clamp(-128, 127).to(torch.int8)
        return data_int, data_scale
```

**Groupwise Weight Quantization (ZeRO-3兼容)**:

- QuantizedLinear: 运行时dequantize INT8→FP16 → torch._C._nn.linear (禁止ZeRO-3替换的functional.linear)
- QuantizedEmbedding: 类似, embedding层也量化
- ★ ZeRO-3协同: all_gather → quantize → partition → 内存进一步释放

### 5.3 CUTLASS MixedGEMM: INT4/INT8 Weight量化GEMM

源码: `deepspeed/inference/v2/kernels/cutlass_ops/mixed_gemm/mixed_gemm.py`

```python
class MixedGEMM(DSKernelBase):
    """CUTLASS MoE GEMM, supports INT4 and INT8 weight quantization"""

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    supported_act_fns = [GELU, SILU, RELU, IDENTITY]

    def __init__(self, fp_dtype, act_fn, num_bits):
        if num_bits != 4 and num_bits != 8:
            raise ValueError("supported num_bits are 4 and 8")  # ★ 仅INT4和INT8

    def __call__(self, output, hidden_states, weights, scales, biases):
        self.kernel(output, hidden_states, weights, biases, self.num_bits, self.act_fn)
```

### 5.4 量化方法总结

```
=== DeepSpeed 推理量化矩阵 ===

V2 Engine (RaggedInferenceConfig):
  ★ wf6af16      ← FP6 weight + FP16 activation (唯一V2量化模式)
                    8 fp6 packed in 3 fp16, per-channel scales
                    TC-FPx kernel, qtorch runtime quantization
                    ★ 仅支持FP16 input (不支持BF16)
                    ★ SM 8.0+ (Ampere) 必须

  MixedGEMM INT4/INT8 ← CUTLASS MoE expert GEMM
                    W4A16 或 W8A16, fused activation
                    ★ 仅用于MoE expert权重, 不是通用线性层量化

V1 Engine (DeepSpeedInferenceConfig):
  INT8 MoQ       ← 对称INT8权重量化, per-group
                    WeightQuantization.quantize_data()
                    运行时dequantize
                    ★ 可与ZeRO-3 groupwise quantization协同

  ZeRO-3 Groupwise ← QuantizedLinear + QuantizedEmbedding
                    all_gather → quantize → partition
                    ★ 适合超大模型推理

对比 vLLM:
  vLLM支持30+量化方法: GPTQ/AWQ/FP8/INT4(Marlin)/INT8/SqueezellM/HQQ/...
  ★ DeepSpeed 量化方法远少于vLLM
  ★ ★★ DeepSpeed无INT4 GPTQ-Marlin通用推理量化 (只有MoE专用MixedGEMM)
```

### 5.5 FP6-LLM 性能数据

**LLaMA-70B, FP6 on 2xA100-80GB** (DeepSpeed-FP6 Blog):

| 配置 | 推理延迟 | 推理吞吐 | 说明 |
|------|---------|---------|------|
| FP16 baseline | 基线 | 基线 | 2xA100必须 |
| FP6 wf6af16 | **1.5x 降低** | **3.5x 提升** | 单A100即可运行! |

★ 关键: FP6让70B模型在**单张A100-80GB**上运行 (FP16需2张), decode阶段memory-bound优势更大。

---

## 6. Kernel 优化: V2 模块化 + V1 融合算子

### 6.1 V2 Kernel 架构: DSKernelBase 模块化

源码: `deepspeed/inference/v2/kernels/ds_kernel.py`

```python
class DSKernelBase(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """触发编译+warmup+autotuning, 验证配置兼容性"""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """执行kernel, 不做autotuning, 不做内存分配"""
```

★ 所有V2 kernel继承DSKernelBase, 编译时warmup, 运行时零开销调用。

### 6.2 BlockedFlashAttn: Atom-based FlashAttention

源码: `deepspeed/inference/v2/kernels/ragged_ops/blocked_flash/blocked_flash.py`

```python
class BlockedFlashAttn(DSKernelBase):
    """Modified flash-attn-2 for blocked KV-cache inference"""

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    # ★ 要求 CUDA CC >= 8.0 (Ampere+)
    # ★ head_size 必须 % 16 == 0

    def __init__(self, head_size, dtype):
        inf_module = RaggedUtilsBuilder().load()
        self.kernel = inf_module.flash_attn_by_atoms  # ★ atom-based接口

    def __call__(self, out, q, k, v, atoms, softmax_scale):
        # atoms: [num_atoms, 8] int32 → 见attention_atom.h
        # k/v: [n_blocks, block_size, n_heads_kv, head_size] (blocked KV cache)
```

**Atom Builder**: 将ragged batch转换为attention计算单元

```
Atom结构 (attention_atom.h, [num_atoms, 8] int32):
  每个atom包含: block索引, token偏移, q_block范围, kv_block范围等
  Atom Builder在CPU构建 → copy_ non_blocking到GPU

  q_block_size: head_size≤64→128, head_size≤160→64, head_size=192→128
  kv_block_size: head_size≤64→128, head_size≠160→64

  ★ 对比 vLLM: vLLM用FlashInfer的BatchPrefillWithPagedKV/BatchDecodeWithPagedKV
  ★ DeepSpeed: 自研atom-based blocked_flash_attn_by_atoms
  ★ 两者都基于FlashAttention-2修改, 但接口不同
```

### 6.3 MoE推理Kernel栈

源码: `deepspeed/inference/v2/modules/implementations/moe/cutlass_multi_gemm.py`

**DSMultiGemmMoE**: 完整的MoE推理pipeline:

```
MoE Forward Pipeline:
  1. _gate_proj(logits, hidden_states, gate_w)          ← BlasLibLinear (FP16/BF16)
  2. _top_1_gate(expert_counts, scores, assignments, offsets, logits, batch)
     ← ★ RaggedTopKGating CUDA kernel (支持top_k=1/2/4/8)
  3. _moe_scatter(moe_input, expert_cumsum, mapped_slots, ...)
     ← ★ MoEScatter: token按expert重排 + cumsum计算
  4. _mlp_1(gated_intermediate, moe_input, mlp_1_w, expert_cumsum)
     ← ★ MoEGEMM CUTLASS: ragged multi-expert GEMM (INT4/INT8/FP16/BF16)
  5. _activation(intermediate, gated_intermediate)
     ← CUDAGatedActivation (SiLU/GeLU)
  6. _mlp_2(output_unordered, intermediate, mlp_2_w, expert_cumsum)
     ← MoEGEMM CUTLASS
  7. _moe_gather(output, output_unordered, scores, mapped_slots, expert_counts)
     ← ★ MoEGather: expert输出→原序恢复 + score加权 + expert_counts清零

★ 关键设计:
  - 所有buffer预分配 (max_tokens * n_top_k)
  - expert_cumsum: 每个expert的token起始位置 (类似DeepEP的offset)
  - mapped_slots: token在expert输入中的位置索引 → gather时恢复原序
  - expert_counts.zero_() 在gather中fused清零 → 下次gate无需单独zero
```

### 6.4 V1 融合算子

**QKV Fusion**:
```
非融合: Q = x@W_Q, K = x@W_K, V = x@W_V → 3次GEMM + 3次input读取
融合: QKV = x@[W_Q|W_K|W_V] → 1次GEMM + 1次input读取 → decode节省~15-20%内存带宽
```

**MLP + GeLU Fusion**: Element-wise操作融入GEMM epilogue/prologue

**Residual Add + LayerNorm Fusion**: 减少intermediate tensor global memory读写

### 6.5 Logits Gather: 高效最终token提取

源码: `deepspeed/inference/v2/kernels/ragged_ops/logits_gather/logits_gather.py`

```python
class RaggedLogitsGather(DSKernelBase):
    """Gather hidden states of the FINAL token of each sequence → reduces unembedding cost"""

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def __call__(self, final_token_activations, all_activations, ragged_wrapper):
        # 从 [num_tokens, model_dim] 中gather每序列最后一个token → [num_seqs, model_dim]
        # ★ 大幅减少LM head的GEMM大小 (num_tokens → num_seqs)
        self.kernel(final_token_activations, all_activations,
                    ragged_wrapper.batch_metadata_buffer(),
                    ragged_wrapper.inflight_seq_descriptors())
```

★ ★ 对比 vLLM V1: vLLM也用类似优化 (logits_indices只计算最后token → 省compute)

---

## 7. 推理 Tensor Parallelism 与训练 TP 的差异

### 7.1 V2 Engine TP

`RaggedInferenceEngineConfig.tensor_parallel.tp_size` 配置TP度数。

TP初始化: `engine_v2._initialize_tp_group()` → `init_distributed()` → `dist.new_group(ranks)`

模型加载时: Policy通过 `sharding/` 子目录中的分片逻辑将权重按TP拆分:
- `sharding/attn.py`, `sharding/mlp.py`, `sharding/qkv.py`, `sharding/embedding.py`, `sharding/unembed.py`

KV Cache: 每个TP rank只存 1/TP 的heads → KV memory减少TP倍 → 更多并发序列

### 7.2 推理 TP vs 训练 TP 对比

| 维度 | 训练 TP | 推理 TP |
|------|---------|---------|
| 通信频率 | fwd+bwd=4次AllReduce/层 | 仅fwd=2次AllReduce/层 |
| 通信量/层 | 4×4B×H | 2×4B×H |
| Batch Size | 固定 (micro batch 1-32) | 动态 (SplitFuse混合) |
| KV Cache | 不持久化 | 持久化, 按TP分片 |
| PP | 1F1B/Interleaved | ★ 不用PP (延迟敏感) |

### 7.3 关键差异

1. **通信量减半**: 推理只需前向2次AllReduce, 但decode阶段GEMM极小, AllReduce固定延迟可能成瓶颈
2. **KV Cache分片**: 每卡存1/TP的KV heads → 并发序列数增加TP倍
3. **不用PP**: 推理延迟敏感, PP气泡不可接受
4. **动态batch**: SplitFuse混合prefill+decode → 通信模式不一致

---

## 8. 与 vLLM/SGLang 的对比分析

### 8.1 架构级对比

| 维度 | DeepSpeed-FastGen (V2) | vLLM (V1) | SGLang |
|------|------------------------|-----------|--------|
| 调度策略 | Dynamic SplitFuse (固定token budget) | Chunked Prefill (preemption+全重置) | Overlap (CPU处理上batch+GPU执行本batch) |
| KV Cache | BlockedKVCache (block_size=128) | PagedAttention (block_size=16) | RadixAttention (True Patricia Trie) |
| Attention | BlockedFlashAttn (atom-based) | FlashInfer / FA3 | FlashInfer / Triton |
| Continuous Batching | DSStateManager + SchedulingResult | Scheduler两阶段(RUNNING→WAITING) | overlap scheduling + FutureMap relay |
| Prefix Caching | ★ **无** | BlockHash prefix caching | ★ Radix tree prefix复用 |
| MoE | DSMultiGemmMoE (CUTLASS scatter/gather) | 11种router + 8种AllToAll | DeepEP + 6 dispatcher |
| Spec Dec | **无** | 8+ Proposer (Eagle/N-gram等) | 无 (计划中) |
| LoRA | **无** | Punica Multi-LoRA | 无 |
| P/D分离 | **无** | NIXL/FlexKV connector | Mooncake / RadixCache |
| 量化 | FP6 (wf6af16) + INT4/INT8 (MoE only) | 30+ 方法 (GPTQ/AWQ/FP8/INT4/Marlin) | GPTQ/AWQ/FP8/INT8 |
| 模型支持 | 11 Policy (OPT/Llama/Mistral/Mixtral/Phi/Qwen/Exaone4) | 200+ 架构 | 50+ 架构 |
| 抢占 | SchedulingResult.KVCacheLimitExceeded → 拒绝新请求 | preemption → num_computed_tokens=0全重置 | ★ radix tree保存被抢占KV → prefix可复用 |
| 硬件 | NVIDIA only (SM 8.0+) | NVIDIA/AMD/Intel/Ascend | NVIDIA/AMD |
| 可观测性 | 基础日志 | Prometheus 30+ 指标 | Prometheus |

### 8.2 ★★★ 关键差异: Prefix Caching

```
DeepSpeed V2:
  BlockedAllocator: 简单linked-list, allocate/free
  ★ 无 prefix caching → 每 sequence 的 system prompt KV 每次都重新计算
  ★ GRPO rollout_n=8 → system prompt KV 计算8次 → 7×浪费

vLLM V1:
  BlockPool + chained hash → prefix caching
  但抢占时 num_computed_tokens=0 → KV 全丢弃 → prefix也不保留

SGLang:
  ★ ★ ★ RadixAttention True Patricia Trie
  抢占时保留prefix KV → _split_node从radix tree分裂
  ★ GRPO rollout_n=8 → system prompt KV只计算1次 → 7×省prefill
```

### 8.3 ★★★ 关键差异: 抢占策略

```
DeepSpeed V2: SchedulingResult.KVCacheLimitExceeded → 拒绝新序列
  - 不抢占正在运行的序列
  - 新请求排队等待KV cache释放
  - ★ 简单但可能导致排队延迟

vLLM V1: preemption → 被抢占序列 num_computed_tokens=0
  - 强制释放KV cache → 全重置
  - ★ 重新计算时浪费已计算的prefix

SGLang: 抢占但保留prefix → radix tree管理
  - 释放non-prefix KV, 保留prefix KV
  - ★ 最优策略: 既有空间又有prefix复用
```

---

## 9. RTX 4090 实战可行性

### 9.1 DeepSpeed-Kernels 兼容性

```
RTX 4090: SM 8.9 (Ada Lovelace)
  DeepSpeed-Kernels wheel: SM 8.0+ → ★ 兼容
  BlockedFlashAttn: CC >= 8.0 → ★ 兼容

  不兼容:
  ★ FP8 E5M2 训练: ✗ (SM 8.9不支持FP8 E5M2, 仅SM 90+)
  ★ NVLS: ✗ (Hopper only)
  ★ TMA: ✗ (Hopper only)
```

### 9.2 ZeRO-Inference on RTX 4090

```
RTX 4090 24GB VRAM + PCIe Gen4 (带宽~25GB/s实际):

ZeRO-3 推理 (单GPU):
  模型参数offload到CPU → 每次forward H2D copy
  7B BF16 ~14GB → 24GB内可全放 → 不需要ZeRO-3!
  13B BF16 ~26GB → 超出24GB → ZeRO-3 TP=2需要第二张GPU (PCIe灾难!)
  ★ ★★ 13B+ 模型推理: 量化是唯一出路, 不是ZeRO-Inference

ZeRO-Infinity CPU Offload:
  PCIe带宽瓶颈: 7B每层参数~28MB × 32层 × H2D+D2H = ~1.8GB/forward
  延迟: ~72ms/forward (仅通信) → 太慢

结论:
  ★★★ RTX 4090: ZeRO-Inference 完全不推荐
  ★★★ 用 INT4 量化把模型压缩到单GPU内存内 (7B INT4 ~4GB)
  ★★★ 量化后不需要参数offload → 无PCIe瓶颈 → 纯GPU计算
```

### 9.3 V2 Engine on RTX 4090

```
V2 Engine (InferenceEngineV2):
  ★★ BlockedKVCache: 自动计算可用内存 → num_blocks
  ★ 7B INT4: ~4GB权重 + ~11GB KV cache (INT8KV+GQA-8) → 24GB内可行
  ★ FP6 wf6af16: 7B FP6 ~7GB → 剩余17GB KV cache → 可行但不如INT4省

  V2 Engine限制:
  ★★★ 无ZeRO配置 → 模型必须全在GPU → 必须量化!
  ★★★ 无prefix caching → GRPO rollout重复计算system prompt
  ★★★ 无speculative decoding → 无EAGLE加速
  ★★★ 只有wf6af16量化 → INT4 GPTQ-Marlin不支持 → 只有FP6可选
  ★★★ 无LoRA serving → 不适合多tenant推理

  MoE (Mixtral/Qwen2-MoE):
  ★ DSMultiGemmMoE TP=1 → EP=1 → 单GPU
  ★ Mixtral-8x7B INT4: ~6GB权重 → 但8个expert每次2个 → GEMM小batch → 低效
  ★ RTX 4090: MoE推理低效 (expert切换频繁 + 小batch GEMM)
```

### 9.4 ★★★ RTX 4090 推理最优路径

```
RTX 4090 推理决策树:

模型≤7B dense?
  └─→ vLLM V1 + INT4 GPTQ-Marlin + INT8KV + GQA-8 + FlashInfer
      → 4,791 tok/s (decode)
      → EAGLE speculative → 9,088 tok/s
      ★ DeepSpeed V2 只有FP6 (~7GB权重) → 剩余少 → 并发少 → 不如INT4

模型13B+ dense?
  └─→ INT4量化到~8GB → vLLM V1 → 可行但并发受限
      ★ DeepSpeed V2 FP6 ~13GB → 剩余11GB → 并发极少 → 不如vLLM INT4

MoE模型?
  └─→ vLLM MoE serving (11种router + EPLB)
      ★ DeepSpeed V2 MoE也可行但无EPLB/DeepEP → 性能不如vLLM

结论:
  ★★★ RTX 4090 推理最优 = vLLM V1 + INT4 + INT8KV + GQA + EAGLE
  ★★★ DeepSpeed-FastGen V2 on RTX 4090 = 可运行但非最优
      原因: 无INT4通用量化, 无prefix caching, 无spec dec, 无LoRA
```

### 9.5 ★★★ RTX 4090 RL训练+推理一体化

```
RTX 4090 RL训练 (GRPO):
  ★★★ rLLM TinkerBackend: in-process + LoRA auto-init + zero-copy weight sync
  ★★★ 训练后: LoRA merge → INT4 → vLLM → 4,791 tok/s → EAGLE → 9,088 tok/s

  DeepSpeed 推理在此路径中的角色:
  ★★★ DeepSpeed-FastGen 不支持 LoRA serving → 无法直接从GRPO训练过渡到推理
  ★★★ 需要额外步骤: LoRA merge → 导出HF → 再加载到MII → 量化部署
  ★★★ vs vLLM: Punica Multi-LoRA → 不需要merge → 直接serve

  ★★★ 完整路径: rLLM Tinker GRPO → merge → INT4 → vLLM → EAGLE
  ★★★ DeepSpeed推理在此路径中不优于vLLM
```

---

## 10. 2025-2026 最新特性与现状

### 10.1 DeepSpeed v0.19.1 (2026-05-27)

推理相关变更 (最近5个版本):

| 版本 | 日期 | 推理相关变更 |
|------|------|-------------|
| v0.19.1 | 2026-05 | Muon optimizer blog, BF16 CPU optimizer offload |
| v0.19.0 | 2026-05 | DeepCompile, AutoEP, AutoSP, ZenFlow |
| v0.18.9 | 2026-03 | Patch release |
| v0.18.8 | 2026-03 | Patch release |
| v0.18.7 | 2026-03 | SuperOffload blog (GH200专用) |

### 10.2 MII 版本追踪

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v0.3.3 | 2025-03 | 构建系统更新 (pyproject.toml) |
| v0.3.2 | 2025-03 | DCO替代CLA, pydantic v2 |
| v0.3.1 | 2024-10 | Streaming, 模型列表更新 |
| v0.3.0 | 2024-08 | Pydantic v2迁移, scheduling修复 |
| v0.2.4 | 2024-07 | Llama-3支持, KV cache starvation修复 |
| v0.2.0 | 2024-01 | ★ Mixtral/Phi-2/Falcon支持 |

### 10.3 项目状态评估

```
DeepSpeed 推理项目状态 (2025-2026):

活跃度:    ██░░░░░░░░░░░░░░░░░░  低
           (MII v0.3.x主要是维护性更新, V2 Engine代码在主repo但MII层更新慢)

V2 Engine: ████████░░░░░░░░░░░░  中等
           (代码质量高, 模块化设计好, 但缺少prefix caching/spec dec等关键特性)

社区参与:  ████░░░░░░░░░░░░░░░░  中等偏低
           (PR响应慢, 核心贡献者Microsoft内部)

对比 vLLM:
  vLLM 2024-2026: V1重写 + MRv2 + BCG + 30+量化 + 8+spec dec + P/D分离 + Multi-LoRA
  ★★★ vLLM生态远超DeepSpeed推理
```

### 10.4 关键技术遗产

1. **Dynamic SplitFuse**: 固定token budget + prefill/decode混合 → 被其他框架借鉴思路
2. **FP6-LLM/TC-FPx**: 首个完整FP6 GPU kernel实现 → 为非标准位宽量化提供参考
3. **BlockedKVCache + Atom-based Attention**: V2 Engine的模块化设计 → 高质量代码参考
4. **DSKernelBase 模块化kernel**: 编译时warmup + 运行时零开销 → 值得学习的设计模式
5. **DSModuleRegistry**: attention/linear/moe/norm/embeb/unembed可插拔 → 模型支持扩展容易
6. **Effective Throughput SLA**: prompt延迟+生成EMA延迟的综合评估框架 → 生产SLO参考

### 10.5 实际部署建议 (2025-2026)

```
=== 选择 DeepSpeed-FastGen 的场景 ===

1. 已在DeepSpeed训练生态中, 需要统一技术栈
2. 长 prompt 工作负载 (SplitFuse核心优势, P95延迟稳定)
3. FP6量化需求 (70B单卡A100部署)
4. Microsoft Azure 环境
5. MoE模型推理 (V2 DSMultiGemmMoE支持好)

=== 不推荐 DeepSpeed-FastGen 的场景 ===

1. 需要 Speculative Decoding (EAGLE/N-gram等)
2. 需要 Multi-LoRA serving
3. 需要 Prefix Caching (GRPO rollout重复计算system prompt)
4. 需要 INT4 GPTQ-Marlin 通用量化 (只有FP6+MoE专用INT4/INT8)
5. 需要 AMD/Intel GPU 支持
6. 需要 P/D 分离
7. 需要结构化输出
8. RTX 4090 单GPU推理 (vLLM INT4更优)

=== 替代方案 ===

- vLLM V1: 通用LLM serving, 特性最全, RTX 4090最优
- SGLang: prefix-sharing密集场景, GRPO rollout最优
- TRT-LLM: NVIDIA GPU极致性能
- rLLM Tinker: RTX 4090 RL训练+推理一体化最优
```

---

## 总结

| 主题 | 核心要点 |
|------|---------|
| **ZeRO-Inference** | ZeRO-3参数分区+CPU offload让超大模型可推理, 但延迟极高(~3x通信开销), ★★★ 仅适合offline/batch推理, RTX 4090完全不需要(量化更优) |
| **V2 Engine** | InferenceEngineV2: ragged batching + BlockedKVCache(block_size=128) + Atom-based BlockedFlashAttn + MoE scatter/gather + FP6 wf6af16, ★ 模块化设计优秀但缺少prefix caching/spec dec/LoRA |
| **FastGen + MII** | MII(前端) + V2 Engine(后端), SplitFuse在MII层实现, 11种模型Policy, ZMQ IPC跨TP通信 |
| **Dynamic SplitFuse** | 固定token budget, 长prompt分块+短prompt拼接, decode从不暂停, 2.3x有效吞吐, 3.7x P95降低 |
| **量化** | V2: wf6af16(FP6 only) + MixedGEMM INT4/INT8(MoE only); V1: INT8 MoQ + groupwise(ZeRO-3); ★★★ 量化方法远少于vLLM |
| **Kernel** | V2: DSKernelBase模块化 + BlockedFlashAttn(atom-based) + MoE scatter/gather/gating + logits_gather; V1: QKV/MLP/GeLU/Residual fusion |
| **推理 TP** | 仅前向通信减半, 不用PP, KV按TP分片, 动态batch |
| **vs vLLM** | FastGen优势: SplitFuse调度+长promptP95; vLLM优势: INT4量化+prefix caching+spec dec+LoRA+P/D分离+30+量化方法 |
| **RTX 4090** | ★★★ DeepSpeed推理可运行但非最优; 最优=vLLM INT4+INT8KV+GQA+EAGLE; ★★★ ZeRO-Inference完全不推荐 |
| **2025-2026现状** | V2 Engine代码质量高但特性不全, MII维护模式, 核心价值在SplitFuse思想+FP6研究遗产+模块化kernel设计 |

---

## 参考资料

1. DeepSpeed-FastGen Blog: https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen
2. DeepSpeed-FP6 Blog: https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fp6
3. DeepSpeed-MII: https://github.com/deepspeedai/DeepSpeed-MII
4. FP6-LLM Paper: arXiv:2401.14112
5. ZeroQuant: https://arxiv.org/abs/2206.01861
6. DeepSpeed Main Repo: https://github.com/deepspeedai/DeepSpeed (v0.19.1)
7. DeepSpeed Inference V2 Source: deepspeed/inference/v2/ (engine_v2.py, ragged/, kernels/, modules/)
8. DeepSpeed Inference V1 Source: deepspeed/inference/engine.py, config.py, quantization/
