# vLLM V1 Prefill/Decode 分离架构源码阅读

> P/D Disaggregated Serving: prefill 和 decode 在独立 GPU 组上运行，通过 KV Connector 传输 KV Cache

## 1. 动机与收益

### 为什么需要 P/D 分离

传统架构中 prefill 和 decode 共享 GPU:
- **Prefill**: compute-bound (大矩阵运算), 处理长 prompt 时耗时长
- **Decode**: memory-bound (权重+KV 读取), 需要低延迟
- **冲突**: 长 prefill 阻塞 decode 请求, 导致高延迟

### 收益

| 指标 | 共享架构 | P/D 分离 |
|------|---------|---------|
| TTFT (32K) | 13s | 94ms (chunked) |
| Decode 延迟 | 受 prefill 影响 | 稳定 |
| GPU 利用率 | 60-70% | 85-95% |
| 扩展性 | 统一扩展 | 独立扩展 P/D |

**1P+2D 配置**: 在保持吞吐的同时减少 GPU 使用量约 50%

## 2. 架构概览

```
Client → Proxy → Prefill Instance (N GPU) ──KV Transfer──→ Decode Instance (M GPU)
                   │                                              │
                   └── 计算 prompt KV cache                      └── 持续生成 tokens
                   └── KV Cache 发送给 Decode 节点              └── 只需 model weights + KV
```

### 核心抽象

```
KVConnectorBase_V1  (基类接口)
  ├── KVConnectorRole.SCHEDULER  (调度器端: 元数据管理, 握手)
  └── KVConnectorRole.WORKER     (工作端: 实际 KV 传输)
       ├── NixlConnector  (NIXL 高性能传输)
       ├── FlexKVConnector (灵活缓存策略)
       ├── LMCacheConnector (结合 prefix caching)
       └── MooncakeConnector (分布式缓存)
```

## 3. NIXL Connector 实现

**位置**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`

### 3.1 组件结构

| 文件 | 职责 |
|------|------|
| `connector.py` | 薄层门面, 按 role 委托给 scheduler 或 worker |
| `scheduler.py` | 调度器端: 管理 request 元数据, 握手协调 |
| `worker.py` | 工作端: 执行实际 KV 传输 (NIXL DMA) |

### 3.2 类设计

```python
class NixlConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlConnectorScheduler(...)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = NixlConnectorWorker(...)
```

### 3.3 数据流程

```
1. Handshake 阶段:
   - ZMQ 交换元数据 (模型配置, attention backend)
   - 验证兼容性
   - 建立 NIXL 连接

2. Prefill → Decode KV Transfer:
   - Prefill 计算 KV cache
   - 调度器设置 do_remote_prefill=True
   - NIXL 异步 DMA 传输 KV blocks
   - Decode 节点接收并加载

3. 状态管理:
   WAITING_FOR_REMOTE_KVS → 等待远程 KV
   WAITING → 正常等待调度
   RUNNING → 正在执行
```

### 3.4 异步传输

```python
# 异步加载: 不阻塞调度循环
def start_load_kv(self, forward_context):
    # 启动后台 NIXL 传输
    ...

# 状态追踪
_recving_transfers  # 跟踪进行中的传输
```

### 3.5 异构 TP 支持

P/D 节点可以使用不同 TP 大小:
- `D_TP > P_TP`: 多个 decode worker 从同一 prefill worker 读取
- `P_TP > D_TP`: 一个 prefill worker 写入多个 decode worker

### 3.6 内存描述符

```python
def _compute_desc_ids(block_ids, dst_num_blocks, block_size_ratio, ...):
    # FA 快速路径: 向量化广播
    if num_ssm_regions == 0:
        block_arr = np.concatenate(block_ids)[None, :]
        region_ids = np.arange(num_fa_regions)[:, None]
        return (region_ids * num_blocks + block_arr).flatten()

    # SSM (Mamba) 特殊处理: 4 个描述符区域 (x, B, C, ssm)
```

## 4. 调度器集成

### 4.1 KV 检查点

调度器在每个请求的调度过程中检查远程 KV:

```python
if self.connector is not None:
    ext_tokens, load_kv_async = (
        self.connector.get_num_new_matched_tokens(
            request, num_new_local_computed_tokens
        )
    )
    num_external_computed_tokens = ext_tokens
```

### 4.2 Remote Prefill 流程

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    if params.get("do_remote_prefill"):
        # Remote prefill: 获取所有 prompt blocks
        token_ids = request.prompt_token_ids or []
        actual = self._mamba_prefill_token_count(len(token_ids))
        count = actual - num_computed_tokens
        if count > 0:
            return count, True  # 异步加载
```

### 4.3 生命周期管理

```python
def request_finished(self, request, block_ids):
    # 决定是否延迟释放 KV blocks (等待 decode 节点读取)
    delay_free_blocks = any(len(group) > 0 for group in block_ids)

    if delay_free_blocks:
        # 设置 lease 时间
        self._reqs_need_send[request.request_id] = (
            time.perf_counter() + request_kv_blocks_ttl
        )
        # 返回 KV transfer 参数
        return delay_free_blocks, {
            "do_remote_prefill": is_p_node,
            "do_remote_decode": is_d_node,
            "remote_block_ids": block_ids,
            "remote_engine_id": self.engine_id,
            ...
        }
```

## 5. 部署配置

### 5.1 1P2D 配置 (推荐)

```bash
# Prefill (1 GPU)
CUDA_VISIBLE_DEVICES=0 vllm serve model \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector":"NixlConnector",
        "engine_id":"prefill",
        "kv_buffer_device":"cuda",
        "kv_connector_extra_config":{
            "backends":["UCX"],
            "kv_lease_duration":30
        }
    }'

# Decode (2 GPUs, TP=2)
CUDA_VISIBLE_DEVICES=1,2 vllm serve model \
    --port 8200 \
    --tensor-parallel-size 2 \
    --kv-transfer-config '{
        "kv_connector":"NixlConnector",
        "engine_id":"decode",
        "kv_buffer_device":"cuda",
        "kv_connector_extra_config":{
            "backends":["UCX"],
            "kv_lease_duration":30,
            "bidirectional_kv_xfer":true
        }
    }'
```

### 5.2 关键配置参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `kv_connector` | 连接器类型 | "NixlConnector" |
| `engine_id` | 唯一标识 | "prefill"/"decode" |
| `kv_buffer_device` | 缓冲设备 | "cuda" |
| `backends` | 传输后端 | ["UCX"] |
| `kv_lease_duration` | KV 租约时间(秒) | 30 |
| `kv_recompute_threshold` | 重计算阈值(tokens) | 64 |

### 5.3 环境依赖

```bash
pip install nixl ucx-py
export VLLM_NIXL_SIDE_CHANNEL_HOST=localhost
export VLLM_NIXL_SIDE_CHANNEL_PORT=55555
```

## 6. 与 SGLang 对比

| 特性 | vLLM V1 | SGLang |
|------|---------|--------|
| 架构 | 模块化 Connector | 内置 KV 传输 |
| Connector 种类 | NIXL/FlexKV/LMCache/Mooncake | 专有实现 |
| 异构 TP | ✓ (P/D 不同 TP) | ✗ |
| MLA 支持 | ✓ | ✗ |
| HMA (Host Memory) | ✓ | ✗ |
| Prefix Caching 集成 | ✓ | ✓ |
| Speculative Decoding | ✓ | ✓ |

## 7. 关键洞察

1. **Connector 抽象**: 模块化设计, 支持 NIXL/FlexKV/LMCache/Mooncake 多种实现
2. **异步传输**: NIXL DMA + ZMQ side channel, 不阻塞调度循环
3. **KV Lease**: Prefill 节点延迟释放 KV blocks, 等 Decode 节点读取完再释放
4. **异构 TP**: P/D 节点可使用不同 TP, 灵活适配硬件
5. **SSM 支持**: 不只是 Attention, Mamba 等模型也有 KV transfer
6. **与 Prefix Caching 协同**: Prefill 阶段的 prefix 可以被多个 Decode 请求复用

## 参考资料

- 源码: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`
- 相关: [KV Cache Manager](vllm-kv-cache-manager-reading.md), [V1 Scheduler](vllm-v1-executor-reading.md)
- `tools/disaggregated_serving_sim.py` — P/D 分离模拟器
