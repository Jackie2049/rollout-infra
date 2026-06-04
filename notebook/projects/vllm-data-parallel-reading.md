# vLLM V1 Data Parallelism 源码阅读

> vLLM V1 的 Data Parallel (DP) Serving 架构，主要用于 MoE 模型的 Expert Parallelism 扩展

## 1. 核心概念

### 1.1 DP Serving vs 训练 DP

| 维度 | 训练 DP | 推理 DP Serving |
|------|---------|----------------|
| 目的 | 梯度聚合 | 吞吐扩展 / EP 扩展 |
| 通信 | AllReduce 梯度 | 同步 token 数、CUDA Graph 模式 |
| 独立性 | 每卡独立数据 | MoE 需协调 Expert 分布 |
| Dense 模型 | 直接适用 | 不适用（用独立实例代替） |

### 1.2 vLLM DP 的两种模式

1. **MoE Expert Parallelism (EP)**: MoE 模型用 DP 维度分片 Expert
   - `DPEngineCoreProc` 子类处理 DP 协调
   - TP × DP = 总 Expert 分片数
   - 需要 All-to-All 通信

2. **Dense 模型 DP**: 每个副本完全独立
   - `data_parallel_size` 设为 1（每个 rank）
   - 仅用 `data_parallel_index` 标识身份
   - 等价于启动多个独立 vLLM 实例

## 2. 架构层次

```
                         ┌─────────────┐
                         │ API Server  │ ← 负载均衡 (LB)
                         └──────┬──────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                   ▼
     ┌─────────────┐  ┌─────────────┐    ┌─────────────┐
     │ EngineCore  │  │ EngineCore  │    │ EngineCore  │  ← DP Rank 0/1/2...
     │  (DP=0)     │  │  (DP=1)     │    │  (DP=2)     │
     └──────┬──────┘  └──────┬──────┘    └──────┬──────┘
            │                │                   │
     ┌──────┴──────┐  ┌─────┴───────┐   ┌──────┴──────┐
     │  Scheduler  │  │  Scheduler  │   │  Scheduler  │  ← 独立调度
     │  + KV Cache │  │  + KV Cache │   │  + KV Cache │  ← 独立 KV
     └──────┬──────┘  └─────┬───────┘   └──────┴──────┘
            │                │                   │
     ┌──────┴──────┐  ┌─────┴───────┐   ┌──────┴──────┐
     │  Executor   │  │  Executor   │   │  Executor   │  ← 独立执行
     │  (Worker)   │  │  (Worker)   │   │  (Worker)   │
     └─────────────┘  └─────────────┘   └─────────────┘
            │                │                   │
            └───────┬────────┘                   │
                    │  DP Process Group (Gloo)    │
                    │  - AllReduce 同步            │
                    │  - Wave 协调                 │
                    └─────────────────────────────┘
```

## 3. 关键组件

### 3.1 ParallelConfig (`vllm/config/parallel.py`)

```python
@config
class ParallelConfig:
    data_parallel_size: int = 1           # DP 副本数
    data_parallel_size_local: int = 1     # 本地 DP 副本数
    data_parallel_rank: int = 0           # DP rank
    data_parallel_rank_local: int | None  # 本地 DP rank (SPMD)
    data_parallel_index: int              # 标识，不用于 process group
    data_parallel_backend: Literal["mp", "ray"]  # 后端
    data_parallel_external_lb: bool       # K8s 外部 LB 模式
    data_parallel_hybrid_lb: bool         # 混合 LB 模式
    enable_dbo: bool = False              # Dual Batch Overlap
    ubatch_size: int = 0                  # 微批次大小
```

三种 LB 模式:
- **Internal LB**: vLLM 内置负载均衡（默认）
- **External LB**: 外部 K8s 负载均衡（每 Pod 一个 rank）
- **Hybrid LB**: 节点内 vLLM LB + 节点间外部 LB

### 3.2 DPEngineCoreProc (`vllm/v1/engine/core.py:1676`)

```python
class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    # 仅用于 MoE 模型
    assert vllm_config.model_config.is_moe

    # Wave 协调
    self.current_wave = 0      # 当前 wave 编号
    self.step_counter = 0      # 步数计数器

    # 两阶段暂停协议
    self.pending_pause = False
    self.ignore_start_dp_wave = False
```

**Dense 模型处理** (core.py:1155):
```python
if data_parallel and vllm_config.model_config.is_moe:
    # MoE: 使用 DPEngineCoreProc
    engine_core = DPEngineCoreProc(*args, **kwargs)
else:
    # Dense: 每个 DP rank 独立运行，DP size 设为 1
    parallel_config.data_parallel_size = 1
    parallel_config.data_parallel_rank = 0
    engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
```

## 4. DP 协调机制

### 4.1 Wave-based 协调

```
Time →
DP Rank 0:  [Wave 0: process reqs] ──step──step── AllReduce ── [Wave 1: ...]
DP Rank 1:  [Wave 0: process reqs] ──step──step── AllReduce ── [Wave 1: ...]
DP Rank 2:  [Wave 0: process reqs] ──step──step── AllReduce ── [Wave 1: ...]

           ← START_DP_WAVE 消息协调 →
```

- 每个 wave 包含一批请求
- 所有 DP rank 必须在同一个 wave 内步调一致
- wave 结束条件：所有 rank 都没有未完成的请求
- 每 32 步做一次 AllReduce 检查全局状态

### 4.2 DP 同步 (`vllm/v1/worker/dp_utils.py`)

```python
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    ...
) -> tuple[bool, torch.Tensor | None, int]:
    """协调所有 DP rank:
    1. 是否使用 microbatching
    2. 每个 rank 处理多少 token（可能需要 padding）
    3. 统一 CUDA Graph 模式
    """
```

同步内容 (4 元素 AllReduce):
```python
tensor[0][dp_rank] = orig_num_tokens      # 原始 token 数
tensor[1][dp_rank] = padded_num_tokens     # 填充后 token 数
tensor[2][dp_rank] = should_ubatch ? 1 : 0 # 是否 microbatch
tensor[3][dp_rank] = cudagraph_mode        # CUDA Graph 模式
```

### 4.3 DP Padding

**目的**: 确保 CUDA Graph 兼容性

```python
# 所有 rank 填充到最大 token 数
if should_dp_pad:
    max_num_tokens = num_tokens_across_dp.max()
    num_tokens_after_padding = [max_num_tokens] * dp_size
```

### 4.4 CUDA Graph 同步 (`vllm/v1/worker/gpu/dp_utils.py`)

```python
def sync_cudagraph_and_dp_padding(...):
    """协调 CUDA Graph 模式和 DP padding:
    - 取所有 rank 的最小 CUDA Graph 模式
    - 如果任何 rank 用 NONE，所有 rank 都用 NONE
    - 统一 token 数到最大值
    """
```

## 5. Dual Batch Overlap (DBO)

### 5.1 微批次机制 (`vllm/v1/worker/ubatch_utils.py`)

```python
@dataclass
class UBatchSlice:
    request_slice: slice  # 请求范围
    token_slice: slice    # Token 范围
```

DBO 将一个大 batch 拆分成多个 microbatch:
- **目的**: 在 MoE EP 中重叠计算和 All-to-All 通信
- **条件**: token 数超过阈值（decode ≥ 32, prefill ≥ 512）
- **决策**: 所有 DP rank 必须一致（AllReduce 投票）

### 5.2 Attention Metadata 切分

```python
def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice,
    attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """为每个 microbatch 创建独立的 attention metadata:
    - 切分 query_start_loc
    - 处理跨 microbatch 的请求（splits_first/last_request）
    - 调整 seq_lens 和 block_table
    """
```

## 6. Elastic EP (弹性 Expert Parallelism)

### 6.1 动态扩缩容

```python
class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int
    ...
```

支持:
- **Scale Up**: 添加新的 DP rank（增加 Expert 分片）
- **Scale Down**: 移除 DP rank
- **EPLB**: Expert Parallel Load Balancing（专家重分配）

### 6.2 EPLB 配置

```python
@config
class EPLBConfig:
    window_size: int = 1000              # 负载统计窗口
    step_interval: int = 3000            # 重排间隔
    num_redundant_experts: int = 0       # 冗余专家数
    use_async: bool = True               # 异步专家迁移
    communicator: Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]
```

## 7. All-to-All Backend 选项

| Backend | 特点 | 适用场景 |
|---------|------|---------|
| `allgather_reducescatter` | 基于 NCCL 原语 | 通用 |
| `deepep_high_throughput` | DeepEP 高吞吐 | 多节点 MoE |
| `deepep_low_latency` | DeepEP 低延迟 | 在线推理 |
| `mori_high_throughput` | MoRI 高吞吐 | 多节点 |
| `nixl_ep` | NIXL EP | NVIDIA GPU |
| `flashinfer_nvlink_two_sided` | FlashInfer 双边 | MNNVL NVLink |
| `flashinfer_nvlink_one_sided` | FlashInfer 单边 | 高吞吐 |

## 8. 请求生命周期 (DP)

```
1. API Server 接收请求
   │
   ├── Internal LB: vLLM 选择 DP rank
   ├── External LB: K8s Ingress 选择
   └── Hybrid: 两层 LB
   │
2. 请求发送到目标 EngineCore (DPEngineCoreProc)
   │
3. EngineCore.add_request(request, request_wave=current_wave)
   │
4. 如果需要启动新 wave:
   ├── 发送 START_DP_WAVE 给所有 DP rank
   └── 所有 rank 进入 busy loop
   │
5. 每个 step:
   ├── 调度 (独立)
   ├── 执行 (独立)
   ├── coordinate_batch_across_dp() — 同步 token 数
   └── 每 32 步 _has_global_unfinished_reqs()
   │
6. Wave 完成:
   ├── 所有 rank 都无未完成请求
   ├── 通知 client (wave_complete)
   └── current_wave += 1
```

## 9. 关键洞察

1. **DP Serving 主要服务于 MoE**: Dense 模型不需要 DP 协调，直接启动独立实例
2. **Wave 是协调单位**: 所有 DP rank 在同一 wave 内步进，wave 结束时同步
3. **AllReduce 是核心同步原语**: token 数、CG 模式、ubatch 决策、完成状态都通过 AllReduce 同步
4. **CUDA Graph 需要 DP padding**: 所有 rank 必须处理相同数量的 token
5. **DBO 用于 EP 通信重叠**: microbatch 让计算和 All-to-All 通信重叠
6. **Elastic EP 支持动态扩缩**: 运行时增减 DP rank
7. **每 32 步同步一次完成状态**: 减少 AllReduce 开销
8. **两层暂停协议**: pending_pause → AllReduce 共识 → ignore_start_dp_wave

## 10. 配置示例

```bash
# MoE EP: 2 节点 × 8 GPU, TP=8, DP=2 (共 16 expert 分片)
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --all2all-backend deepep_high_throughput

# Dense 模型: 2 副本负载均衡
vllm serve meta-llama/Llama-3.1-70B \
    --data-parallel-size 2
    # 注意: 内部每个副本 DP=1，完全独立

# K8s External LB
vllm serve deepseek-ai/DeepSeek-V3 \
    --data-parallel-size 4 \
    --data-parallel-external-lb \
    --enable-expert-parallel

# Hybrid LB
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-hybrid-lb \
    --data-parallel-start-rank 0
```

## 参考资料

- `vllm/config/parallel.py` — ParallelConfig 定义（DP 参数）
- `vllm/v1/engine/core.py` — EngineCore / DPEngineCoreProc（DP 协调）
- `vllm/v1/worker/dp_utils.py` — DP batch 协调（token 同步）
- `vllm/v1/worker/gpu/dp_utils.py` — CUDA Graph + DP 同步
- `vllm/v1/worker/ubatch_utils.py` — Microbatch 切分
- `vllm/distributed/elastic_ep/` — Elastic EP 动态扩缩
- 相关: [V1 Architecture Map](vllm-v1-architecture-map.md), [MoE Serving](../fundamentals/moe.md)
