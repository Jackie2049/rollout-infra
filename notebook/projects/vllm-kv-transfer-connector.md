# vLLM KV Transfer / KV Connector 架构深度分析

> 源码版本: vLLM main branch (2026-06), 路径: `rollout-infra/vllm-latest/`
>
> 分析范围: `vllm/distributed/kv_transfer/` 下的 KV Connector 框架、NIXL 连接器实现、
> Metrics 系统, 以及与 Disaggregated Prefill 的关系。

---

## 一、架构总览

### 1.1 核心抽象层次

vLLM 的分布式 KV cache 传输分为三个抽象层:

```
┌─────────────────────────────────────────────┐
│          KV Connector (最高层)               │  ← 连接器框架, 对接 Scheduler & Worker
├─────────────────────────────────────────────┤
│          KV Lookup Buffer (可选)             │  ← KV 缓存查找表 (按 token 索引)
├─────────────────────────────────────────────┤
│          KV Pipe (可选, 最底层)              │  ← FIFO tensor 传输通道
└─────────────────────────────────────────────┘
```

KV Connector 是当前主要使用的层。KV Pipe 和 KV Lookup Buffer 可以被绕过 --
当底层通信库(如 NIXL)已经支持 key-value 查找时, 直接在 Connector 层实现即可。

### 1.2 已注册的连接器一览

在 `factory.py` 中注册了以下连接器:

| 名称 | 模块路径 | 用途 |
|------|----------|------|
| `NixlConnector` | `v1/nixl/` | NVIDIA NIXL RDMA 传输 (生产级) |
| `P2pNcclConnector` | `v1/p2p/` | NCCL P2P 传输 |
| `LMCacheConnectorV1` | `v1/lmcache_connector.py` | LMCache 集成 |
| `LMCacheMPConnector` | `v1/lmcache_mp_connector.py` | LMCache 多进程适配 |
| `MooncakeConnector` | `v1/mooncake/` | Mooncake RDMA 传输 |
| `MooncakeStoreConnector` | `v1/mooncake/store/` | Mooncake Store 模式 |
| `HF3FSKVConnector` | `v1/hf3fs/` | HuggingFace 3FS 文件系统 |
| `MoRIIOConnector` | `v1/moriio/` | MoRIIO 传输 |
| `OffloadingConnector` | `v1/offloading_connector.py` | KV offloading (CPU/异构存储) |
| `SimpleCPUOffloadConnector` | `v1/simple_cpu_offload_connector.py` | 简单 CPU offload |
| `FlexKVConnectorV1` | `v1/flexkv_connector.py` | FlexKV 传输 |
| `DecodeBenchConnector` | `v1/decode_bench_connector.py` | Decode 性能基准测试 |
| `ExampleConnector` | `v1/example_connector.py` | 示例连接器 |
| `MultiConnector` | `v1/multi_connector.py` | 组合连接器封装 |

---

## 二、KV Connector 框架

### 2.1 类继承关系

```
                    KVConnectorBase_V1 (ABC)          ← 基类, v1/base.py
                    ├── KVConnectorMetadata (ABC)      ← Scheduler→Worker 元数据
                    ├── KVConnectorWorkerMetadata (ABC)← Worker→Scheduler 元数据
                    ├── KVConnectorHandshakeMetadata   ← P/D Worker 间握手元数据
                    ├── KVConnectorRole (Enum)          ← SCHEDULER / WORKER
                    └── SupportsHMA (ABC)              ← 混合内存分配器支持标记

                    KVConnectorBase = KVConnectorBase_V1  ← 当前唯一基类, base.py

NixlConnector (KVConnectorBase_V1, SupportsHMA)
  ├── NixlConnectorScheduler    ← Scheduler 侧逻辑
  ├── NixlConnectorWorker       ← Worker 侧逻辑
  ├── NixlConnectorMetadata     ← 元数据结构
  ├── NixlKVConnectorStats      ← 统计指标
  └── TPMapping / ReadSpec      ← TP 映射数据结构
```

### 2.2 注册机制 (Factory Pattern)

核心在 `kv_connector/factory.py` 的 `KVConnectorFactory`:

```python
class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name, module_path, class_name):
        # 延迟加载: 存储闭包而非类本身, 避免导入不必要的依赖
        def loader():
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, config, role, kv_cache_config):
        # 根据 role 区分创建 Scheduler 还是 Worker 实例
        return connector_cls(config, role, kv_cache_config)
```

**关键设计**: 延迟加载 (lazy loading) -- 只在需要时才导入特定连接器的模块,
避免加载所有依赖(如 NIXL, Mooncake, LMCache 等可能不可用)。

### 2.3 双角色分离: Scheduler 与 Worker

每个 Connector 在构造时根据 `KVConnectorRole` 分成两个角色:

- **Scheduler 侧** (`KVConnectorRole.SCHEDULER`): 运行在调度器进程中
  - `get_num_new_matched_tokens()` -- 计算远程可匹配的 token 数
  - `update_state_after_alloc()` -- 块分配后更新状态
  - `build_connector_meta()` -- 构建发送给 Worker 的元数据
  - `request_finished()` -- 请求完成后决定块释放策略
  - `update_connector_output()` -- 处理 Worker 回传的输出

- **Worker 侧** (`KVConnectorRole.WORKER`): 运行在每个 Worker 进程中
  - `register_kv_caches()` -- 注册 KV cache 到 NIXL
  - `start_load_kv()` -- 开始异步加载 KV
  - `wait_for_layer_load()` -- 等待特定层的 KV 加载完成
  - `save_kv_layer()` -- 保存某一层的 KV
  - `wait_for_save()` -- 等待所有保存完成
  - `get_finished()` -- 查询已完成的异步传输

NixlConnector 的实现中, `connector.py` 只是一个薄外观 (thin facade),
根据 role 分别委托给 `NixlConnectorScheduler` 和 `NixlConnectorWorker`。

### 2.4 元数据流转

```
Scheduler 进程                      Worker 进程
┌───────────────────┐              ┌───────────────────┐
│ build_connector   │─── meta ───→ │ bind_connector    │
│   _meta()         │              │   _metadata()     │
│                   │              │                   │
│ update_connector  │ ←── output ──│ build_connector   │
│   _output()       │              │   _worker_meta()  │
└───────────────────┘              └───────────────────┘
```

- **Scheduler → Worker**: `NixlConnectorMetadata`, 包含 `reqs_to_recv`, `reqs_to_save`,
  `reqs_to_send`, `heartbeat_by_engine` 等
- **Worker → Scheduler**: `KVConnectorOutput`, 包含 `finished_sending`, `finished_recving`,
  `kv_connector_stats`, `invalid_block_ids` 等

---

## 三、NIXL 连接器深度分析

### 3.1 NIXL 是什么

NIXL (NVIDIA IO Transfer Library) 是 NVIDIA 提供的高性能 I/O 传输库,
支持 RDMA/NVLink 等高速互联。在 vLLM 中:

- 通过 `vllm/distributed/nixl_utils.py` 延迟加载 `nixl` 或 `rixl` (ROCm) 包
- 核心类 `NixlWrapper` (实际为 `nixl._api.nixl_agent`) 封装了 NIXL Agent API
- 支持 CUDA / XPU / CPU / TPU 等多种设备

### 3.2 关键文件职责

| 文件 | 职责 |
|------|------|
| `nixl/connector.py` | 薄外观类, 根据 role 委托给 Scheduler/Worker |
| `nixl/scheduler.py` | Scheduler 侧: 元数据构建、请求生命周期管理、ZMQ 握手监听 |
| `nixl/worker.py` | Worker 侧: NIXL Agent 管理、内存注册、异步传输、TP 映射 |
| `nixl/metadata.py` | 数据类定义: `NixlAgentMetadata`, `ReqMeta`, `NixlConnectorMetadata` 等 |
| `nixl/stats.py` | 传输统计: `NixlKVConnectorStats` (聚合/Reduce) + `NixlPromMetrics` |
| `nixl/tp_mapping.py` | 异构 TP 映射: `compute_tp_mapping()` 计算 local→remote rank 映射 |
| `nixl/utils.py` | 常量、ZMQ 辅助、设备支持矩阵 |

### 3.3 数据流: 完整的 KV 传输过程

```
                     Prefill 实例 (P)                           Decode 实例 (D)
                 ┌──────────────────────┐                ┌──────────────────────┐
  HTTP Request   │                      │  kv_transfer   │                      │
  ─────────────→ │  API Server          │   _params      │  API Server          │
                 │  (设置 do_remote_     │ ──────────────→│  (设置 do_remote_    │
                 │   prefill=True)      │                │   decode=True)       │
                 │                      │                │                      │
                 │  Scheduler           │                │  Scheduler           │
                 │  ┌────────────────┐  │                │  ┌────────────────┐  │
                 │  │ get_num_new_   │  │                │  │ get_num_new_   │  │
                 │  │ matched_tokens │  │                │  │ matched_tokens │  │
                 │  │ → (0, False)   │  │                │  │ → (N, True)    │  │
                 │  └────────────────┘  │                │  └────────────────┘  │
                 │                      │                │                      │
                 │  request_finished()  │                │  update_state_after  │
                 │  → delay_free=True   │                │  _alloc()            │
                 │  → 返回 kv_transfer  │                │  → 记录 _reqs_need   │
                 │    _params 给 D      │                │    _recv              │
                 │                      │                │                      │
                 │  Worker              │                │  build_connector     │
                 │  ┌────────────────┐  │                │  _meta() → meta      │
                 │  │ KV Cache 注册  │  │                │                      │
                 │  │ NIXL Agent     │  │  NIXL READ     │  Worker              │
                 │  │ ZMQ Listener   │←─┼─── (RDMA) ────│  ┌────────────────┐  │
                 │  │ send_notif     │←─┼─── (notif) ───│  │ start_load_kv  │  │
                 │  └────────────────┘  │                │  │ → nixl_xfer    │  │
                 │                      │                │  │                │  │
                 │  get_finished()      │                │  get_finished()   │  │
                 │  → 收到 notif        │                │  → 传输完成       │  │
                 │  → 释放 blocks       │                │  → post-process   │  │
                 └──────────────────────┘                └────────────────┘  │
                                                                                │
                                                         └──────────────────────┘
```

### 3.4 详细传输流程 (ASCII 流程图)

```
P Worker                                    D Worker
═══════════════════════════════════════════════════════════════

1. 初始化阶段:
┌─────────────────────────┐               ┌─────────────────────────┐
│ register_kv_caches()    │               │ register_kv_caches()    │
│ → 注册 KV cache 到 NIXL │               │ → 注册 KV cache 到 NIXL │
│ → 创建 NixlAgentMetadata│               │                         │
│ → 启动 ZMQ Listener     │               │                         │
│   (ROUTER socket)       │               │                         │
└─────────────────────────┘               └─────────────────────────┘

2. 请求完成 (Prefill 侧):
┌─────────────────────────┐
│ request_finished()      │
│ → delay_free = True     │
│ → 设置 lease 计时器     │
│ → 返回 kv_transfer_     │
│   params 给 D           │
│   (包含 block_ids,      │
│    engine_id, host,     │
│    port 等)             │
└─────────────────────────┘

3. 握手阶段 (D 发起):
                                          ┌─────────────────────────┐
                                          │ _ensure_handshake()     │
                                          │ → ZMQ REQ 连接 P       │
              ←──── GET_META_MSG ──────── │                         │
              ────→ NixlHandshakePayload ─→│                         │
                                          │ → 兼容性 hash 校验      │
                                          │ → add_remote_agent()    │
                                          │   注册远程 NIXL Agent   │
                                          │ → 构建 TP 映射          │
                                          │ → 注册远程内存描述符     │
                                          └─────────────────────────┘

4. 异步读取传输:
                                          ┌─────────────────────────┐
                                          │ start_load_kv()         │
                                          │ → _read_blocks_for_req()│
                                          │   → 计算 desc IDs       │
                                          │   → make_prepped_xfer() │
              ┌──── NIXL READ ──────────→ │   → transfer()          │
              │    (RDMA 直接读取 P 内存) │   → 记录 handle          │
              │                           └─────────────────────────┘
              │
              │    ... 异步传输中 ...
              │
              │                           ┌─────────────────────────┐
              ←──── NIXL notif ────────── │ get_finished()          │
                   "{req_id}:{tp_size}"   │ → _pop_done_transfers() │
                                          │ → check_xfer_state()    │
                                          │ → record_telemetry()    │
                                          │ → post-process (如需要)  │
                                          └─────────────────────────┘

5. 块释放 (P 侧):
┌─────────────────────────┐
│ _get_new_notifs()       │
│ → 收到所有 consumer     │
│   的 notif              │
│ → 从 _reqs_to_send 移除 │
│ → 释放 blocks           │
└─────────────────────────┘
```

### 3.5 NIXL 内存注册机制

NIXL 传输的核心是**内存描述符 (descriptors)** 和 **预准备传输列表 (prepped dlist)**:

1. **本地内存注册** (`register_kv_caches`):
   - 将 KV cache tensor 注册为 NIXL memory regions
   - 计算 base_addr, size, device_id
   - 调用 `nixl_wrapper.register_memory(descs)`
   - 创建 local xfer handle (`prep_xfer_dlist`)

2. **远程内存注册** (`add_remote_agent`):
   - 通过握手获取远程 `NixlAgentMetadata`
   - 调用 `nixl_wrapper.add_remote_agent(metadata)`
   - 为每个远程 TP rank 创建 remote xfer handle

3. **Block ID → Descriptor ID 映射**:
   - `_compute_desc_ids()` 将逻辑 block IDs 转换为 NIXL descriptor IDs
   - 公式: `desc_id = region_id * num_blocks + block_id`
   - 支持混合模型 (Attention + SSM/Mamba), 使用不同的 region 布局

### 3.6 KV Cache 布局与传输优化

NixlConnector 默认要求 HND (Head-NumTokens-Dim) 布局:

```python
@classmethod
def get_required_kvcache_layout(cls, vllm_config):
    return "HND"  # 更高效的传输性能
```

HND 布局使得 KV head 在内存中连续, 支持异构 TP 下的 head 切分:
- D_TP > P_TP: 每个 D rank 从 P rank 读取不同的 kv_head 切片
- P_TP > D_TP: 每个 D rank 需要读取多个 P rank 的数据

### 3.7 兼容性检查 (Compatibility Hash)

在握手阶段使用 SHA-256 hash 进行兼容性校验:

```python
def compute_nixl_compatibility_hash(vllm_config, attn_backend_name, cross_layers_blocks):
    factors = {
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,  # 当前版本: 4
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
        "cross_layers_blocks": cross_layers_blocks,
        "is_hma_enabled": is_hma_enabled,
    }
    return hash_factors(factors)
```

两阶段解码:
1. 先解码 `NixlHandshakePayload` 获取 `compatibility_hash`
2. 比较本地和远程 hash
3. 只在 hash 匹配时解码 `NixlAgentMetadata` -- 防止 schema 不兼容导致崩溃

### 3.8 心跳与租约机制

P 侧 KV blocks 使用**租约 (lease)** 机制, 防止 blocks 无限期占用:

- P 侧 `request_finished()` 设置 `_kv_lease_duration` (默认 30s) 的租约
- D 侧定期发送心跳 (`HB:req1,req2,...`) 延长租约
- 心跳间隔 = `kv_lease_duration // 6`
- 租约到期后 P 侧自动释放 blocks

### 3.9 传输后处理 (Post-processing)

接收 KV 后可能需要的后处理:

| 场景 | 操作 |
|------|------|
| block_size 不匹配 | `kv_postprocess_blksize_on_receive` -- 小 block 合并为大 block |
| 布局不匹配 (HND→NHD) | `kv_postprocess_layout_on_receive` -- 维度转置 |
| 两者同时 | `kv_postprocess_blksize_and_layout_on_receive` |
| 异构 attention 后端 | `post_process_device_kv_on_receive_heterogeneous_attn` |
| Host buffer 模式 | `sync_recved_kv_to_device` -- CPU buffer→GPU 拷贝 |

---

## 四、TP Rank 映射 (异构 TP 支持)

### 4.1 核心概念

异构 TP 指的是 Prefill 和 Decode 使用不同的 tensor parallel size。
`TransferTopology` 和 `compute_tp_mapping()` 负责处理这种情况。

```
tp_ratio = local_tp_size / remote_tp_size
  > 0: 本地 TP > 远程 TP (D>P), 多个本地 rank 从一个远程 rank 读取不同 head
  < 0: 远程 TP > 本地 TP (P>D), 一个本地 rank 从多个远程 rank 读取
```

### 4.2 TPMapping 数据结构

```python
@dataclass(frozen=True)
class TPMapping:
    source_ranks_per_group: tuple[tuple[int, ...], ...]  # 每组的源 rank 列表
    all_source_ranks: tuple[int, ...]                      # 所有源 rank 并集
    rank_to_attention_slot: dict[int, int]                 # rank → FA head slot
    rank_offset_factor: int                                # FA head 偏移因子
```

### 4.3 异构 TP 示例

```
示例: D_TP=4, P_TP=2 (tp_ratio=2)

Decoder TP workers                   Prefix TP workers
    (world_size=4)                       (world_size=2)

rank_offset   p_remote_tp_rank
────────────────────────────────────────────────────────
    0              0      Worker0 ──── 1st half of KV ────→ Worker0  [KV Cache]
                                                                       /
    1              0      Worker1 ──── 2nd half of KV ────→/

    0              1      Worker2 ──── 1st half of KV ────→ Worker1  [KV Cache]
                                                                       /
    1              1      Worker3 ──── 2nd half of KV ────→/
```

---

## 五、Metrics / Stats 系统

### 5.1 指标收集流程

```
Worker TP Ranks (每个 rank 独立收集)
┌────────────┐ ┌────────────┐ ┌────────────┐
│ Worker 0   │ │ Worker 1   │ │ Worker N   │
│ xfer_stats │ │ xfer_stats │ │ xfer_stats │
│ .record_   │ │ .record_   │ │ .record_   │
│ transfer() │ │ transfer() │ │ transfer() │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
          KVOutputAggregator.aggregate()
          (在 Executor 层合并所有 Worker 的输出)
                     │
                     ▼
          ┌────────────────────┐
          │ Aggregated Stats   │  ← 合并后的统计
          │ (单个 data dict,   │
          │  包含所有 rank 的   │
          │  观测值)           │
          └────────┬───────────┘
                   │
        ┌──────────┼──────────┐
        ▼                     ▼
  KVConnectorLogging     KVConnectorProm
  (CLI 日志输出)          (Prometheus 指标)
  .observe() → .log()    .observe()
  → reduce() 输出摘要     → histogram/counter
```

### 5.2 KVConnectorStats 基类

```python
class KVConnectorStats:
    """所有子类必须可序列化 (从 worker 发送到 logger 进程)"""
    data: dict[str, Any]

    def reset(self): ...
    def aggregate(self, other) -> KVConnectorStats: ...  # 合并两个 stats
    def reduce(self) -> dict[str, int | float]: ...       # 生成摘要统计
    def is_empty(self) -> bool: ...
```

### 5.3 NixlKVConnectorStats

NIXL 连接器的指标:

| 指标 | 含义 |
|------|------|
| `transfer_duration` | 传输耗时 (秒) |
| `post_duration` | 传输后处理耗时 (秒) |
| `bytes_transferred` | 传输字节数 |
| `num_descriptors` | 描述符数量 |
| `num_failed_transfers` | 失败传输次数 |
| `num_failed_notifications` | 失败通知次数 |
| `num_kv_expired_reqs` | KV blocks 过期请求数 |

### 5.4 TP Rank 聚合机制

`KVOutputAggregator` 在 `utils.py` 中实现:

```python
class KVOutputAggregator:
    def aggregate(self, outputs, output_rank=0):
        for model_runner_output in outputs:
            kv_output = model_runner_output.kv_connector_output
            # 1. 聚合 finished_sending/recving (引用计数)
            # 2. 聚合 kv_connector_stats (合并 data dict)
            # 3. 聚合 kv_connector_worker_meta
            # 4. 合并 kv_cache_events
            # 5. 合并 invalid_block_ids
```

关键点:
- **finished_sending/recving**: 使用引用计数, 每个 rank 报告自己的完成集合,
  只有所有 rank 都报告某个 req_id 完成时, 才认为真正完成
- **stats**: 调用 `aggregate()` 将所有 rank 的观测值合并到同一个 data dict
- **reduce()**: 在 logger 进程中计算汇总统计 (均值、P90 等)

### 5.5 Prometheus 指标

`NixlPromMetrics` 注册以下 Prometheus 指标:

| 指标名 | 类型 | 含义 |
|--------|------|------|
| `vllm:nixl_xfer_time_seconds` | Histogram | 传输耗时 |
| `vllm:nixl_post_time_seconds` | Histogram | 传输后处理耗时 |
| `vllm:nixl_bytes_transferred` | Histogram | 传输字节数 |
| `vllm:nixl_num_descriptors` | Histogram | 描述符数量 |
| `vllm:nixl_num_failed_transfers` | Counter | 失败传输计数 |
| `vllm:nixl_num_failed_notifications` | Counter | 失败通知计数 |
| `vllm:nixl_num_kv_expired_reqs` | Counter | KV 过期请求计数 |

所有指标支持 `per_engine` label, 区分不同的 engine 实例。

---

## 六、与 Disaggregated Prefill 的关系

### 6.1 工作模式

KV Connector 的核心应用场景是 **P/D disaggregation** (Prefill/Decode 分离):

1. **P 侧 (Prefill Instance)**:
   - 接收请求, 执行 prefill, 计算 KV cache
   - `request_finished()` 时设置 `do_remote_decode=True`
   - 延迟释放 KV blocks (等待 D 侧读取)
   - 启动 ZMQ listener 等待 D 侧握手

2. **D 侧 (Decode Instance)**:
   - 从客户端接收请求 (带 `kv_transfer_params`)
   - `get_num_new_matched_tokens()` 返回可远程加载的 token 数
   - Worker 侧发起 NIXL READ 拉取 KV cache
   - 传输完成后开始 decode

3. **双向 KV 传输** (实验性):
   - 通过 `bidirectional_kv_xfer=True` 启用
   - D 侧的 KV blocks 也可以传输回 P 侧

### 6.2 请求生命周期中的 Hook 点

```
请求创建
    │
    ▼
on_new_request()         ← 跟踪心跳需求
    │
    ▼
get_num_new_matched_tokens()  ← 判断远程可加载的 token 数
    │
    ▼
[Scheduler 分配 blocks]
    │
    ▼
update_state_after_alloc()    ← 记录需要 recv 的请求
    │
    ▼
build_connector_meta()        ← 构建发送给 Worker 的元数据
    │
    ▼
[Worker 执行 forward pass]
    │
    ├── start_load_kv()       ← 发起 NIXL READ
    ├── wait_for_layer_load() ← 等待加载 (NIXL: no-op, 异步)
    ├── save_kv_layer()       ← 保存 KV (NIXL: no-op, 直接传输)
    └── wait_for_save()       ← Host buffer 模式下的 D2H 拷贝
    │
    ▼
get_finished()                ← 查询已完成的传输
get_kv_connector_stats()      ← 获取传输统计
get_block_ids_with_load_errors()  ← 获取加载失败的 blocks
    │
    ▼
[请求生成完成]
    │
    ▼
request_finished()            ← P 侧: 设置延迟释放; D 侧: 正常释放
    │
    ▼
update_connector_output()     ← D 侧: 停止心跳
```

### 6.3 跨层 Block 优化 (Cross-Layer Blocks)

NixlConnector 支持 `prefer_cross_layer_blocks`:
- 当启用时, KV cache 使用单一跨层 tensor: `[num_layers, num_blocks, ...]`
- 一次 NIXL 传输可以搬运所有层的 KV, 减少传输开销
- 需要 FlashAttn/FlashInfer 后端 + HND 布局 + `enable_cross_layers_blocks=True`

---

## 七、关键配置项

| 配置 | 来源 | 默认值 | 含义 |
|------|------|--------|------|
| `kv_connector` | `KVTransferConfig` | None | 连接器名称 |
| `engine_id` | `KVTransferConfig` | None | 实例标识 |
| `kv_buffer_device` | `KVTransferConfig` | "cuda" | 传输 buffer 设备 |
| `kv_lease_duration` | extra_config | 30 | P 侧 KV 租约时长 (秒) |
| `kv_recompute_threshold` | extra_config | 64 | 最小远程拉取 token 数 |
| `bidirectional_kv_xfer` | extra_config | False | 双向 KV 传输 |
| `backends` | extra_config | ["UCX"] | NIXL 后端 |
| `num_threads` | extra_config | 4 | NIXL 线程数 |
| `enable_cross_layers_blocks` | extra_config | False | 跨层 block 优化 |
| `enforce_handshake_compat` | extra_config | True | 强制兼容性检查 |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | env var | - | ZMQ 握手主机 |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | env var | - | ZMQ 握手端口 |

---

## 八、错误处理与容错

### 8.1 传输失败处理

```python
def _handle_failed_transfer(self, req_id, handle):
    # 1. 标记该请求的所有 blocks 为 invalid
    meta = self._recving_metadata.get(req_id)
    if meta:
        self._invalid_block_ids.put(set(meta.local_block_ids[0]))
    # 2. 标记请求为失败 (加入 done_recving, 触发 recompute)
    self._failed_recv_reqs.put(req_id)
    # 3. 记录失败指标
    self.xfer_stats.record_failed_transfer()
```

Scheduler 收到 `invalid_block_ids` 后, 会将这些 blocks 标记为需要重新计算。

### 8.2 握手失败处理

- 后台线程中的握手失败通过 Future 回调处理
- 失败的请求被加入 `_failed_recv_reqs` 队列
- 不会阻塞后续请求的处理

---

## 九、总结

### 核心设计思想

1. **Scheduler/Worker 分离**: 清晰的职责划分, Scheduler 管理"哪些 KV 需要传输",
   Worker 管理"如何传输"
2. **异步传输**: NIXL 使用 RDMA READ, 传输在后台完成, 不阻塞模型执行
3. **延迟释放 + 租约**: P 侧不立即释放 KV blocks, 等待 D 侧确认或超时
4. **异构 TP 支持**: 通过 TPMapping 和 TransferTopology 支持 P/D 不同 TP 配置
5. **可扩展性**: Factory 模式 + 延迟加载, 添加新连接器只需注册即可

### 关键文件路径

```
vllm/distributed/kv_transfer/
├── __init__.py                           # 公共 API
├── kv_transfer_state.py                  # 全局 connector 单例管理
├── README.md                             # 架构说明
└── kv_connector/
    ├── __init__.py                       # 导出 KVConnectorBase
    ├── base.py                           # 当前别名 (→ v1)
    ├── factory.py                        # 注册 + 创建连接器
    ├── utils.py                          # TransferTopology, KVOutputAggregator 等
    └── v1/
        ├── __init__.py
        ├── base.py                       # KVConnectorBase_V1 基类
        ├── metrics.py                    # KVConnectorStats, KVConnectorProm 基类
        ├── nixl/
        │   ├── __init__.py
        │   ├── connector.py              # NixlConnector (外观类)
        │   ├── scheduler.py              # Scheduler 侧实现
        │   ├── worker.py                 # Worker 侧实现 (核心传输逻辑)
        │   ├── metadata.py               # 元数据结构 + 兼容性 hash
        │   ├── stats.py                  # NixlKVConnectorStats + Prometheus
        │   ├── tp_mapping.py             # 异构 TP 映射
        │   └── utils.py                  # ZMQ 辅助, 设备支持矩阵
        └── ... (其他连接器)
```
