# Disaggregated Serving — Prefill/Decode 分离架构

> LLM 推理架构的前沿演进 — 将计算密集的 prefill 和访存密集的 decode 分离到不同 GPU 实例

## 1. 核心动机

```
传统单体服务 (Monolithic Serving) 的问题:

Prefill 阶段 (处理输入 prompt):
  - 计算密集型 (compute-bound)
  - 需要处理所有 input tokens 的 attention (O(N²d))
  - 大矩阵乘法，充分利用 Tensor Core
  - 关键指标: TTFT (Time To First Token)

Decode 阶段 (逐 token 生成):
  - 访存密集型 (memory-bound)
  - 每个 token 只做一次小矩阵乘 + KV cache 读取
  - 瓶颈是 HBM 带宽 (读权重 + KV cache)
  - 关键指标: ITL (Inter-Token Latency)、吞吐量

矛盾:
  1. Prefill 需要高 FLOPS → 适合 TP 大、高计算 GPU
  2. Decode 需要高带宽 → 适合大显存、高带宽 GPU
  3. 混合部署时两者互相干扰
  4. Prefill 期间 decode 的 ITL 暴增 (tail latency)
```

### 1.1 资源利用率问题

```
单体部署的时间线 (简化):
  ┌──── Prefill ────┐┌─ Decode ─┐┌─ Decode ─┐
  │  GPU 100% 计算   ││ GPU 20%   ││ GPU 20%   │
  │  带宽 60%       ││ 带宽 100% ││ 带宽 100% │
  └─────────────────┘└──────────┘└──────────┘
   ← TTFT 延迟 →    ← ITL 受影响 →

分离部署的时间线:
  P 实例: ┌── Prefill ──┐┌── Prefill ──┐  ← 专注高计算
  D 实例: ┌── Decode ───────────────────┐  ← 专注低延迟
          └── KV transfer ──┘

收益:
  - P 实例利用率: 80-90% (vs 40-60%)
  - D 实例 ITL: 稳定无抖动
  - 独立扩展 P/D 数量
```

## 2. 架构模式

### 2.1 Splitwise (Microsoft Research)

```
核心思想: 将 prefill 和 decode 分配到不同的 GPU pool

架构:
                    ┌─────────────────┐
                    │   Load Balancer  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │ Prefill Pool   │──KV──────▶│ Decode Pool    │
     │                │ Transfer  │                │
     │ GPU Type A     │           │ GPU Type B     │
     │ (高计算)        │           │ (高带宽)        │
     │ TP=8           │           │ TP=4           │
     └────────────────┘           └────────────────┘

关键设计:
  1. 硬件异构: P 实例用计算型 GPU，D 实例用高带宽 GPU
  2. 独立扩展: 根据 prefill/decode 负载分别扩缩容
  3. KV Transfer: 通过高速网络传输 KV cache
  4. 共置优化: P/D 对在同一机架减少网络延迟
```

### 2.2 DistServe

```
DistServe 的架构特点:

1. 前端路由:
   - 请求分析: 根据 prompt 长度路由到合适的 P 实例
   - 负载均衡: 考虑 P/D 实例的当前负载

2. Prefill 集群:
   - 多个 prefill 实例，支持不同 TP 配置
   - 独立 batching 策略 (可大 batch)
   - 计算 KV cache 后发送到 D 实例

3. Decode 集群:
   - 多个 decode 实例
   - Continuous batching (iteration-level scheduling)
   - 接收 KV cache 后开始生成

4. KV Store (共享存储):
   - 分布式 KV cache 存储系统
   - 支持多级缓存 (GPU/CPU/SSD)
```

### 2.3 Mooncake

```
Mooncake: 面向多级缓存的分离式架构

                    ┌─────────────────┐
                    │  Request Router  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │ Prefill Node   │           │ Decode Node    │
     └────────┬───────┘           └────────▲───────┘
              │                             │
              ▼                             │
     ┌──────────────────────────────────────────────┐
     │           KV Cache Pool (分布式)               │
     │  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
     │  │  GPU    │  │  CPU    │  │  SSD    │     │
     │  │ (Hot)   │  │ (Warm)  │  │ (Cold)  │     │
     │  └─────────┘  └─────────┘  └─────────┘     │
     └──────────────────────────────────────────────┘

核心特性:
  1. 前缀复用: 相同 system prompt 的 KV cache 跨请求共享
  2. 多级存储: GPU → CPU → SSD 层级缓存
  3. RDMA 传输: GPU Direct RDMA 零拷贝
  4. 状态代理: 维护多轮对话状态

前缀复用收益:
  - Chat 场景 KV cache 命中率: 40-60%
  - 避免重复计算 system prompt
  - Prefill 延迟降低 30-50%
```

### 2.4 vLLM KV Transfer Framework

```
vLLM 的 KV Transfer 框架 (已在源码中实现):

组件:
  ┌────────────────────────────────────────────┐
  │           KV Connector Framework           │
  │  (抽象接口，支持多种传输后端)                  │
  ├────────────────────────────────────────────┤
  │  NixlConnector │ PyNixlConnector │ ...     │
  │  (NVIDIA NIXL) │ (Python NIXL)   │         │
  ├────────────────────────────────────────────┤
  │           KV Transfer Scheduler            │
  │  (集成到 vLLM V1 Scheduler)                 │
  └────────────────────────────────────────────┘

工作流程:
  1. Prefill 实例完成 KV cache 计算
  2. 通过 ZMQ 握手建立 P-D 连接
  3. NIXL agent 注册 KV block 内存
  4. 异步 RDMA 读取 KV 数据
  5. Decode 实例接收后开始生成

支持的角色:
  - Producer: 只做 prefill，发送 KV
  - Consumer: 只做 decode，接收 KV
  - Both: 同时支持 prefill 和 decode

异构 TP:
  - Prefill TP=8 (高计算)
  - Decode TP=4 (高带宽)
  - KV transfer 自动处理 TP 聚合/拆分
```

## 3. KV Cache 传输

### 3.1 传输方式对比

| 方式 | 带宽 | 延迟 | 复杂度 | 适用场景 |
|------|------|------|--------|---------|
| RDMA (IB/RoCE) | 100+ GB/s | <1ms | 高 | 生产环境 |
| NCCL P2P | 50-100 GB/s | 1-5ms | 中 | 同机房 |
| TCP Socket | 10-25 GB/s | 5-20ms | 低 | 开发测试 |
| NVLink | 300-900 GB/s | <0.1ms | 低 | 同节点 |

### 3.2 NIXL 传输流程

```
NIXL (NVIDIA I/O Transfer Library):

1. 初始化:
   - 创建 NIXL Agent (绑定 GPU device)
   - 注册内存区域 (KV blocks)
   - 建立网络连接 (ZMQ handshake)

2. 传输:
   Producer (P):                    Consumer (D):
   ┌────────────────┐              ┌────────────────┐
   │ KV blocks ready│              │                │
   │ send meta via  │─── ZMQ ────▶│ recv meta      │
   │ ZMQ            │              │ register local │
   │                │              │ buffers        │
   │                │◀── RDMA ────│ RDMA read req  │
   │ handle RDMA    │              │                │
   │ read           │              │ wait for       │
   │                │─── NOTIF ──▶│ completion     │
   │                │              │                │
   └────────────────┘              └────────────────┘

3. 清理:
   - D 确认接收完成
   - P 释放 KV blocks
   - 统计传输 metrics

关键指标:
  - Transfer latency: <1ms (典型)
  - Bandwidth utilization: >90%
  - Per-rank throughput: 各 rank 独立统计
```

### 3.3 内存布局优化

```
KV Cache 的内存布局对传输效率的影响:

HND (Head-NumTokens-Dim) — vLLM 默认:
  k_cache: [num_blocks, num_heads, block_size, head_dim]
  优点: 传输时按 block 整块读取，效率高
  缺点: attention 需要转置

NHD (NumTokens-Head-Dim) — 部分框架使用:
  k_cache: [num_blocks, block_size, num_heads, head_dim]
  优点: attention 直接使用
  缺点: 传输时需额外处理

传输优化:
  - Block 级传输: 按 PagedAttention block 粒度
  - 零拷贝: RDMA 直接读写 GPU 内存
  - 异步: 传输与计算 overlap
```

## 4. 调度策略

### 4.1 请求生命周期

```
用户请求 → [Load Balancer] → Prefill 实例
                                  │
                            计算 KV cache
                                  │
                            KV Transfer
                                  │
                            Decode 实例 → 生成 tokens → 用户

多轮对话:
  第 1 轮: P → D (完整 prefill)
  第 2 轮: D → P (发送已有 KV) → P 只 prefill 新内容 → D
  或: D 本身保留 KV，新轮次继续 decode (prefix caching)
```

### 4.2 负载均衡

```
Prefill 实例选择:
  - Round-robin: 简单，不考虑负载
  - Least-loaded: 选择当前 prefill 数最少的实例
  - Prompt-length-aware: 长 prompt → 高 TP 实例

Decode 实例选择:
  - KV affinity: 尽量路由到已有相关 KV 的实例 (prefix reuse)
  - Load-aware: 考虑当前 batch 大小和 KV cache 使用率
  - Latency-aware: 选择延迟最低的实例

P/D 配比优化:
  - 根据实际 workload 特征调整 P/D 实例比例
  - 长 prompt 多 → 增加 P 实例
  - 长 generation 多 → 增加 D 实例
  - 动态调整: 根据实时负载自动扩缩
```

## 5. 网络要求

```
KV Transfer 的网络需求:

LLaMA-70B 单次请求的 KV Cache 大小:
  - 80 heads × 128 dim × 2 (K+V) × FP16 × 2048 tokens
  = 80 × 128 × 2 × 2 × 2048 = 80 MB

高并发场景 (100 req/s, avg 2048 tokens):
  - 传输带宽需求: 80 MB × 100 = 8 GB/s
  - 需要: 100 Gbps+ InfiniBand

推荐网络配置:
  ┌─────────────┐                   ┌─────────────┐
  │ Prefill Rack │─── 400 Gb/s ──── │ Decode Rack  │
  │ (IB HCA x4) │    InfiniBand     │ (IB HCA x4) │
  └─────────────┘                   └─────────────┘

  最低: 100 Gb/s (开发/测试)
  推荐: 400 Gb/s (生产)
  理想: 800 Gb/s (大规模集群)

延迟要求:
  - KV transfer: < 1ms (同机房)
  - 端到端增加: < 5ms (对 ITL 影响可接受)
```

## 6. 性能收益

### 6.1 吞吐量

```
Splitwise 论文数据 (LLaMA-70B, A100):

配置                     吞吐量 (tok/s)    延迟 P99
─────────────────────────────────────────────────
Monolithic (TP=8)         1,200           85 ms
Disaggregated (P=8,D=4)   3,800           22 ms
提升                       3.2x            3.9x

GPU 利用率:
  Monolithic: 40-60% (混合负载时互相干扰)
  Disaggregated: 80-90% (各实例专注一种任务)
```

### 6.2 延迟

```
ITL (Inter-Token Latency) 分布:

Monolithic:
  P50: 12 ms    P90: 45 ms    P99: 120 ms
  ← Prefill 干扰导致 P99 暴增

Disaggregated:
  P50: 8 ms     P90: 10 ms    P99: 15 ms
  ← 无 prefill 干扰，延迟稳定

TTFT (Time To First Token):
  Monolithic: 略好 (无需网络传输)
  Disaggregated: 增加 1-3ms (KV transfer)
  → 增加可接受，且可通过 P 实例高 TP 补偿
```

### 6.3 成本效率

```
硬件配置灵活性:

Monolithic (LLaMA-70B):
  需要: 2× A100-80GB (TP=2) 或 4× A100-40GB (TP=4)
  每种 GPU 都要满足 prefill + decode 需求

Disaggregated:
  Prefill: 4× A100-40GB (TP=4, 高计算)
  Decode:  2× A100-80GB (TP=2, 高带宽)
  → 可以混合不同 GPU 类型，降低总体成本

独立扩展:
  高 prompt/短 generation 场景 → 增加 P 实例
  长 generation 场景 → 增加 D 实例
  → 比统一扩展更经济
```

## 7. 实际部署考量

### 7.1 挑战

```
1. 系统复杂度增加
   - 多实例管理、网络配置、状态同步
   - 监控和调试更复杂

2. 网络依赖
   - KV transfer 依赖高速网络
   - 网络故障影响更大
   - 跨机房部署延迟高

3. 状态管理
   - 多轮对话的 KV cache 路由
   - 故障恢复需要重建 KV cache
   - 内存管理更复杂

4. 调度复杂性
   - P/D 配比优化
   - 负载均衡策略
   - 异构硬件管理
```

### 7.2 最佳实践

```
1. 从小规模开始
   - 单 P/D 对验证功能
   - 逐步扩展到多实例

2. 监控网络性能
   - KV transfer 带宽利用率
   - 传输延迟 P99
   - 网络错误率

3. 共置 P/D 实例
   - 同一机架部署 P 和 D
   - 减少网络延迟
   - 使用同一 IB switch

4. 预热和缓存
   - 预分配 KV cache 内存
   - 预建网络连接
   - 编译缓存 (torch.compile)

5. 容错设计
   - D 实例故障 → 重路由到其他 D 实例
   - P 实例故障 → 重试其他 P 实例
   - KV cache 损坏 → 重新 prefill
```

## 8. 未来方向

```
1. 多级 KV Cache
   GPU → CPU → SSD → 远端存储
   更大容量的 KV cache 复用

2. 动态 P/D 切换
   根据 workload 实时切换 P/D 角色
   空闲实例动态复用

3. KV Cache 压缩
   量化 KV cache (FP8/INT4) 减少传输量
   Attention sink 保留 + 剪枝

4. 与 Prefix Caching 深度集成
   全局 prefix cache pool
   跨 D 实例共享 system prompt KV

5. 更高速互联
   NVLink 5.0 (1.8 TB/s)
   CXL (Compute Express Link)
   专用 KV transfer ASIC
```

## 模拟验证

- `tools/disaggregated_serving_sim.py` — Disaggregated Serving 模拟器（5 个实验）
  - 实验 1: Monolithic vs Disaggregated (1P+1D, 100 reqs, H100, NVLink → 2.14x 吞吐)
  - 实验 2: KV Transfer 带宽影响 (10GbE 184 tok/s → NVLink 415 tok/s)
  - 实验 3: P/D 实例比例 (1P+2D GPU效率最高 133.1 tok/GPU)
  - 实验 4: Prompt 长度影响 (短 prompt 两者无差异, 超长 prompt 分离优势)
  - 实验 5: 扩展效率 (Monolithic 4-GPU 扩展效率 0.36, Disaggregated 0.40)

关键发现:
- **1P+2D 是最优比例**: GPU 效率 133.1 tok/GPU, Decode 是吞吐瓶颈
- **KV Transfer 开销与带宽线性相关**: NVLink <4ms, 10GbE >2300ms
- **短 prompt 场景无差异**: TTFT 62ms vs 63ms, 分离架构收益可忽略
- **扩展效率都不理想**: 4x GPU 仅 ~0.4 效率, 说明调度和分配是瓶颈

## 参考

- [Splitwise: Efficient generative LLM inference using phase splitting (ISCA 2024)](https://arxiv.org/abs/2311.18677)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (OSDI 2024)](https://arxiv.org/abs/2401.09670)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
- [vLLM KV Transfer Documentation](https://docs.vllm.ai/en/latest/serving/disaggregated_prefill.html)
- [NIXL Library](https://github.com/ai-dynamo/nixl)
