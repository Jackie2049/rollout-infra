# vLLM V1 PD Separation & KV Transfer Architecture

> 2026-06-08 | 源码分析, vLLM V1 KV connector系统 — PD分离 + KV offloading + 多后端
> 关键: 7种KV connector后端, P2P NCCL(PD分离), Offloading(swap), NIXL/Mooncake(RDMA)

## 1. KV Connector Architecture

vLLM V1的KV connector系统采用**Scheduler/Worker双角色设计**:

```
Scheduler Connector:
  - get_num_new_matched_tokens() → 查询remote已有KV tokens
  - update_state_after_alloc() → alloc后更新状态
  - request_finished() → 请求完成 → 可异步释放KV blocks

Worker Connector:
  - start_load_kv() → 开始加载KV (async)
  - wait_for_layer_load() → 阻塞等某层KV加载完 → 可pipeline
  - save_kv_layer() → 保存某层KV (async)
  - wait_for_save() → 阻塞等所有save完成
  - get_finished() → 返回已完成异步传输的请求IDs
```

**设计亮点**: layer-by-layer pipeline → prefill GPU逐层发送KV → decode GPU逐层接收 → 与compute重叠

## 2. 7种KV Connector后端

| Connector | 用途 | 特点 |
|-----------|------|------|
| **P2pNcclConnector** | PD分离(NVLink/PCIe) | NCCL P2P, request_id编码地址, layer-by-layer |
| **NIXLConnector** | NVIDIA官方KV传输 | NIXL library(高性能RDMA+NVLink) |
| **MooncakeConnector** | RDMA KV传输 | Mooncake RDMA, 跨节点KV迁移 |
| **OffloadingConnector** | **CPU swap** | HMA(Hierarchical Memory Access), scheduler+worker |
| **LMCacheConnector** | 持久KV存储 | LMCache分布式KV缓存 |
| **FlexKVConnector** | 灵活KV管理 | — |
| **SimpleCPUOffload** | 简单swap | 基础CPU offload |

## 3. P2P NCCL Connector (PD分离)

```python
class P2pNcclConnector:
  is_producer = config.is_kv_producer  # prefill/decode角色

  # Producer (Prefill GPU): save KV per layer
  save_kv_layer():
    if not is_producer: return  # 只producer保存
    kv_cache = kv_layer[block_ids, ...]  # 从paged KV buffer提取
    p2p_nccl_engine.send_tensor(kv_cache, remote_address)

  # Consumer (Decode GPU): load KV per layer
  start_load_kv():
    if is_producer: return  # 只consumer加载
    for each request in metadata:
      for each layer:
        kv_cache = p2p_nccl_engine.recv_tensor(remote_address)
        layer[block_ids, ...] = kv_cache  # 注入paged KV buffer
```

**关键**: request_id编码IP+port → producer/consumer直接连接 → NCCL P2P通信

## 4. Offloading Connector (CPU Swap)

```python
class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):
  # SupportsHMA → 支持分层内存访问(GPU HBM + CPU pinned + SSD)
  prefer_cross_layer_blocks = True  # 跨层block → 更高效offloading

  # Scheduler side: 管理offloading策略
  connector_scheduler = OffloadingConnectorScheduler(spec)

  # Worker side: 实际CPU-GPU数据传输
  connector_worker = OffloadingConnectorWorker(spec)
```

**重要**: vLLM确实有swap实现! 但scheduler preemption仍默认recomputation → swap作为connector可选 → 需配置启用

**与实测数据对比**:
- 实测: SWAP比recompute快4-44x → vLLM应默认推荐swap
- OffloadingConnector使用pinned memory → 24GB/s → 与实测一致

## 5. Scheduler Preemption vs KV Connector

```
vLLM V1 scheduler preemption (当前实现):
  _preempt_request():
    kv_cache_manager.free(request)  ← 释放GPU KV blocks
    request.num_computed_tokens = 0 ← 全部recompute
    → 默认recomputation → 无swap选项在scheduler层

KV Connector offloading (可选配置):
  OffloadingConnector → CPU pinned memory → swap in/out
  → 需要配置 kv_transfer_config → enable_offloading
  → scheduler preemption仍用recomputation → 但offloading connector可保留KV

  两种模式:
  1. recomputation only: 释放KV → recompute → 简单但慢(实测17ms for S=16)
  2. offloading: KV→CPU pinned → recompute时从CPU加载 → 快(实测0.38ms)
```

**建议**: RTX 4090用户应启用OffloadingConnector → swap比recompute快4-44x

## 6. PD分离完整流程

```
Prefill GPU (Producer):
  1. 接收用户请求 → prefill S tokens → 生成KV cache
  2. save_kv_layer() per layer → NCCL P2P发送KV到decode GPU
  3. 释放本地KV → 继续处理下一个prefill请求

Decode GPU (Consumer):
  1. 接收KV cache → start_load_kv() → 注入paged KV buffer
  2. 开始decode → 每步生成1 token → 更新KV cache
  3. 请求完成 → 通知prefill GPU释放远程KV

KV传输开销(实测数据):
  PCIe RTX 4090: 3% TTFT overhead → 可行
  NVLink H100: 0.2% TTFT → 几乎免费 → 生产标配

PD比例:
  1 prefill GPU : 4-8 decode GPU
  原因: prefill TFLOPS利用率73.5% vs decode 0.4-12.8%
  → prefill GPU处理快 → 1 GPU可服务多个decode GPU
```

## 7. 核心规律

```
vLLM V1 KV Connector系统:
  双角色设计(Scheduler/Worker) → scheduler决策 → worker执行
  7种connector后端 → 按需选择(NVLink→NCCL/RDMA→Mooncake/通用→Offloading)
  Layer-by-layer pipeline → 与compute重叠 → 减少KV传输延迟

  Preemption决策:
    当前: recomputation only → num_computed_tokens=0 → 全部重算
    实测: swap 4-44x faster → 应启用OffloadingConnector
    → 配置: kv_transfer_config.enable_offloading=True

  PD分离:
    NCCL P2P → request_id编码地址 → 直接连接 → layer-by-layer
    PCIe: 3% TTFT → 可行(修正之前不可行结论!)
    NVLink: 0.2% TTFT → 生产标配

  生产建议:
    RTX 4090单GPU: chunked prefill + swap preemption + FlashInfer
    RTX 4090 2GPU: PD分离(PCIe) → ITL稳定 → TTFT+3%
    H100集群: PD分离(NVLink) → 生产标配 → TTFT+0.2%
    大规模: 1:8 prefill:decode → Mooncake RDMA → 跨节点KV共享
```