# GPUDirect 技术在 AI 推理中的应用

> GPUDirect RDMA/Storage: GPU 内存直接访问网络和存储，消除 CPU 中间拷贝

## 1. 为什么 AI 推理需要 GPUDirect

```
P/D 分离中的 KV Cache 传输:

传统路径 (CPU Staging):
  GPU(P) → PCIe → CPU(P) → PCIe → NIC(P) → 网络 → NIC(D) → PCIe → CPU(D) → PCIe → GPU(D)
           ↖ 4 次 PCIe 交叉, 2 次 CPU 内存拷贝 ↗

GPUDirect RDMA 路径:
  GPU(P) → PCIe → NIC(P) → 网络 → NIC(D) → PCIe → GPU(D)
           ↖ 2 次 PCIe 交叉, 零 CPU 拷贝 ↗

性能差异:
  - 拷贝次数: 4 → 0
  - PCIe 交叉: 4 → 2
  - CPU 利用率: 高 → 极低
  - DDR 带宽消耗: 是 → 否
  - NVIDIA 官方声称: 10x 性能提升
```

## 2. GPUDirect 技术栈

| 技术 | 功能 | 应用场景 |
|------|------|---------|
| GPUDirect RDMA | GPU↔网络直接传输 | P/D 分离 KV Cache 传输 |
| GPUDirect Storage | GPU↔存储直接传输 | 模型加载, Checkpoint 读写 |
| GPUDirect P2P | GPU↔GPU 直接传输 | 同节点 TP 通信 |
| GPUDirect for Video | GPU↔视频设备 | 视频推理 |

### 2.1 GPUDirect RDMA 原理

```
关键步骤:
1. 内存注册 (Memory Registration)
   - GPU 内存通过 cudaMalloc 分配
   - 通过 ibv_reg_mr 注册到 RDMA NIC
   - NIC 获得 GPU 内存的物理地址映射
   - 内存页被 pin 住, 防止被 OS 换出

2. RDMA 描述符创建
   - 每个连续内存区域创建一个 RDMA 描述符
   - 描述符包含: 虚拟地址, 长度, 访问密钥 (rkey/lkey)

3. 非阻塞传输
   - 通过 ibv_post_send 提交 RDMA READ/WRITE 请求
   - NIC 通过 DMA 直接读写 GPU 内存
   - 完成后产生完成队列事件 (CQE)

4. 数据一致性
   - GPU kernel 必须在 RDMA 传输前完成
   - 通过 CUDA Event 或 Stream 同步
   - RDMA 传输期间 GPU 不能修改该内存区域
```

### 2.2 GPUDirect Storage

```
应用: 模型加载 + Checkpoint 读写

传统: NVMe → CPU DRAM → GPU HBM (2 次拷贝)
GDS:  NVMe → GPU HBM (直接 DMA)

性能数据 (Blackwell GPU):
  - ZSTD 压缩: 16-19 GB/s (1.27x 压缩比)
  - ANS 压缩: 181-190 GB/s (1.25x 压缩比)
  - GDS 本地 NVMe: 15-50+ GB/s 持续吞吐

LLM 场景收益:
  - 70B 模型加载: ~140GB (FP16)
  - 传统 PCIe: 140GB / 64GB/s ≈ 2.2s
  - GDS NVMe: 140GB / 50GB/s ≈ 2.8s (SSD 限制)
  - 主要收益: 不占用 CPU 内存带宽和 PCIe 带宽
```

## 3. vLLM NIXL Connector 深度解析

NIXL (NVIDIA IO Extension for LLM) 是 vLLM 的 KV Cache 传输引擎，基于 GPUDirect RDMA。

### 3.1 架构

```
NixlConnector (Facade)
  ├── NixlConnectorScheduler (调度侧)
  │     ├── 请求路由和块分配
  │     ├── ZMQ Side Channel 元数据交换
  │     ├── KV Block Lease 管理
  │     └── 兼容性哈希验证 (SHA-256)
  │
  └── NixlConnectorWorker (传输侧)
        ├── NIXL Agent (nixl_agent / rixl_agent)
        ├── 内存注册 (GPU VRAM → RDMA 可访问)
        ├── 异步 RDMA READ (D 从 P 拉取 KV)
        ├── 传输状态轮询
        └── 遥测数据收集

版本: NIXL_CONNECTOR_VERSION = 4
```

### 3.2 传输生命周期

```
阶段 1: 握手 (Handshake)
  P: 监听 ZMQ Side Channel (VLLM_NIXL_SIDE_CHANNEL_PORT)
  D: 发送 GET_META_MSG + NixlHandshakePayload
  P: 验证兼容性哈希 (vLLM 版本, 模型, dtype, KV heads, ...)
  P: 返回 NixlAgentMetadata (engine_id, base_addr, num_blocks, ...)

阶段 2: 内存注册
  P: register_memory(KV_cache_tensors, memory_type="cuda", backends=["UCX"])
  D: register_memory(KV_cache_tensors, memory_type="cuda", backends=["UCX"])
  → GPU 内存被 pin 住, RDMA NIC 可直接访问

阶段 3: KV Block 传输
  D-side (拉取模式):
    1. 确定需要哪些 blocks (get_num_new_matched_tokens)
    2. 构建 ReadSpec: local_blocks + remote_blocks
    3. prep_xfer_dlist("READ", ...) → 创建传输描述符
    4. make_prepped_xfer(local_handle, local_ids, remote_handle, remote_ids)
    5. transfer(handle) → 非阻塞 RDMA READ
    6. 每步 step() 轮询 check_xfer_state()

阶段 4: 完成通知
  NIXL 自动发送通知到 P
  P 收到通知 → 可以释放 KV blocks

阶段 5: Lease 续期
  D 定期发送心跳 ("HB:req1,req2,...")
  默认: lease=30s, 心跳间隔=5s
```

### 3.3 关键设计决策

| 设计 | 原因 |
|------|------|
| **HND KV Cache 布局** | Head-Number-Depth 确保连续头, 减少 RDMA 描述符 |
| **D-side 拉取 (READ)** | Decode 端知道自己需要哪些 blocks |
| **UCX Backend** | 统一抽象 InfiniBand/RoCE/共享内存 |
| **kv_lease_duration=30s** | 平衡 P-side 显存压力和 D-side 传输延迟 |
| **kv_recompute_threshold=64** | 少于 64 tokens 的 KV 传输不值得, 直接重算 |
| **num_threads=4** | 避免耗尽 Mellanox NIC 的 UAR 空间 |

### 3.4 异构 TP 支持

```
P-side: TP=4 (4 GPU)          D-side: TP=2 (2 GPU)

每个 D rank 从多个 P ranks 读取:
  D-rank-0 → P-rank-0 + P-rank-1 (head splitting)
  D-rank-1 → P-rank-2 + P-rank-3

代码处理:
  - 自动 head 维度切分
  - Block size 差异适配
  - 多源 RDMA 传输合并
```

### 3.5 Host Buffer Fallback

```
不支持 GPUDirect 的设备 (TPU):
  use_host_buffer=True

传输路径:
  GPU(P) → D2H → CPU buffer → RDMA → CPU buffer → H2D → GPU(D)

支持的设备映射:
  CUDA: cuda (直接) / cpu (staging)
  TPU:  cpu only
  XPU:  cpu / xpu
  CPU:  cpu
```

## 4. 性能模型

### 4.1 KV Cache 传输量

| 模型 | Context | KV Cache 大小 | NVLink (600GB/s) | IB NDR (50GB/s) |
|------|---------|-------------|-----------------|-----------------|
| 7B | 2K | ~0.5 GB | 0.8 ms | 10 ms |
| 7B | 8K | ~2.0 GB | 3.3 ms | 40 ms |
| 70B | 2K | ~5.2 GB | 8.7 ms | 104 ms |
| 70B | 16K | ~20.8 GB | 34.7 ms | 416 ms |
| 70B | 128K | ~166 GB | 277 ms | 3320 ms |

### 4.2 传输 vs 重算决策

```
kv_recompute_threshold = 64 tokens

传输成本:
  - RDMA 延迟: ~2-5 μs (设置) + 数据传输时间
  - 64 tokens KV: ~2 MB (7B) → 传输 ~40 μs (IB)

重算成本:
  - Prefill 64 tokens: 2 * 7B * 64 = 896 GFLOPs
  - A100 (312 TFLOPS, 50% MFU): ~5.7 ms
  - H100 (990 TFLOPS, 50% MFU): ~1.8 ms

结论:
  - 少量 tokens: 重算更快 (1.8ms vs 40μs 但含设置开销)
  - 大量 tokens: 传输更优 (线性增长 vs 平方增长)
  - 64 tokens 阈值是经验值, 平衡了设置开销和数据量
```

## 5. 遥测与可观测性

vLLM NIXL 暴露的 Prometheus 指标:

| 指标 | 类型 | 含义 |
|------|------|------|
| `vllm:nixl_xfer_time_seconds` | Histogram | RDMA 传输时间 |
| `vllm:nixl_bytes_transferred` | Histogram | 传输字节数 |
| `vllm:nixl_post_time_seconds` | Histogram | 提交到 NIC 的时间 |
| `num_failed_transfers` | Counter | 传输失败次数 |
| `num_failed_notifications` | Counter | 通知失败次数 |
| `num_kv_expired_reqs` | Counter | KV 过期请求次数 |

统计输出格式:
```
Avg transfer time: 3.2 ms
P90 transfer time: 8.1 ms
Throughput: 1250 MB/s
Avg MB per transfer: 4.2 MB
```

## 6. 实际部署注意事项

### 6.1 网络配置

```
RoCE v2 无损网络配置:
  - PFC (Priority Flow Control): 基于 802.1p 优先级
  - ECN (Explicit Congestion Notification): 拥塞标记
  - DCQCN: 拥塞控制算法
  - MTU: 4096 或更大 (减少包处理开销)

InfiniBand:
  - 原生无损 (Credit-based 流控)
  - SHARP: 交换机内 Reduce (加速 AllReduce)
  - 自适应路由: 包级别负载均衡
```

### 6.2 UAR 耗尽问题

```
Mellanox NIC UAR (User Access Region):
  - 每个 UCX 线程通过 DevX 分配 UAR
  - UAR 数量有限 (~256 per PF)
  - 与 NVSHMEM (DeepEP) 竞争

vLLM 处理:
  - UCX only: num_threads=4
  - 有其他 backend: num_threads=1
  - 避免 UAR 耗尽导致 NVSHMEM 失败
```

### 6.3 GPU 内存布局

```
KV Cache 布局对 RDMA 的影响:

NHD (Number-Head-Depth): [num_tokens, num_heads, head_dim]
  - 每个 head 数据在内存中连续
  - 但多个 head 之间有间隔
  - RDMA 描述符: 需要 num_heads 个描述符

HND (Head-Number-Depth): [num_heads, num_tokens, head_dim]
  - 所有 tokens 的同一 head 数据连续
  - 一个 head 对应一个连续内存区域
  - RDMA 描述符: 只需 num_heads 个 (但每个更大)

vLLM NIXL: 强制使用 HND 布局
  - 原因: 更好的 RDMA 描述符效率
  - 代价: 需要在传输前重排 KV Cache
```

## 7. 关键洞察

1. **GPUDirect 消除 CPU 瓶颈**: KV Cache 传输不再经过 CPU 内存, 释放 DDR 带宽给其他任务
2. **NIXL 是 P/D 分离的核心**: 基于 UCX + GPUDirect RDMA, 支持异构 TP, 异步传输
3. **HND 布局优化**: 减少 RDMA 描述符数量, 提高传输效率
4. **Lease 机制**: 30s 默认租约 + 心跳续期, 平衡显存压力和传输延迟
5. **Host Buffer 兜底**: 不支持 GPUDirect 的设备自动降级到 CPU staging
6. **64 tokens 阈值**: 经验值, 平衡传输设置开销和 prefill 重算成本
7. **遥测完善**: 传输时间/字节/失败率都有 Prometheus 指标

## 参考资料

- [NVIDIA GPUDirect RDMA](https://developer.nvidia.com/gpudirect)
- [NVIDIA DOCA GPUNetIO](https://developer.nvidia.com/blog/tag/gpudirect/)
- vLLM 源码: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`
- vLLM NIXL Utils: `vllm/distributed/nixl_utils.py`
- [UCX Framework](https://www.openucx.org/)
