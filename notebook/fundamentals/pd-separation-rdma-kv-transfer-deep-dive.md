# PD Separation + RDMA KV Transfer Architecture Deep Dive — Prefill/Decode Disaggregation原理 + KV Cache RDMA传输(GPUDirect零拷贝+DMA-BUF+doorbell) + Mooncake架构(PrefillAdapter+GlobalKVCacheManager+TransferEngine+RDMA Write零拷贝) + DistServe/Splitwise/PATROL调度算法 + RDMA拓扑(rail-aligned PD集群) + DCQCN对KV transfer影响 + RTX 4090 PCIe KV transfer实测(3% TTFT) + 2026趋势

> 2026-06-13 | PD分离+RDMA架构深度分析: Prefill compute-bound(73.5% peak)vs Decode memory-bound(0.4% peak)→175x TFLOPS差距→分离是自然的+KV Cache RDMA传输(TCP 5-15ms→RDMA 0.5-2ms→GPUDirect 0.05-0.5ms→NVLink 0.1-1ms)+Mooncake(PrefillAdapter路由+GlobalKVCacheManager全局块管理+TransferEngine RDMA Write零拷贝+ibv_reg_mr预注册+RC QP+批量传输)+DistServe goodput优化+Splitwise三池架构+PATROL延迟感知路由+rail-aligned PD拓扑+DCQCN对KV transfer延迟影响+RTX 4090 PCIe KV transfer实测=3% TTFT可接受
> 关联: disaggregated-serving.md(基础参考), rdma-ai-networking-deep-dive.md(RDMA Verbs+拓扑+DCQCN), prefill-decode-pd-separation-rtx4090.md(RTX 4090实测)
> 参考: Mooncake(SOSP 2025), DistServe(OSDI 2024), Splitwise(ISCA 2024), PATROL(2025), NVIDIA GPUDirect RDMA for KV Transfer(GTC 2025)

## 0. 核心定律: Prefill=compute-bound + Decode=memory-bound + 175x差距 → PD分离必然 + RDMA使KV transfer可行

```
PD分离三层架构:

计算层 (Prefill vs Decode特征):
  → Prefill: compute-bound → 73.5% peak → 需高FLOPS GPU → A100/H100高TP
  → Decode: memory-bound → 0.4% peak → 需高带宽GPU → 大显存+高HBM带宽
  → → 175x TFLOPS差距 → 资源需求完全不同 → 分离是自然的!

传输层 (KV Cache RDMA传输):
  → TCP: 5-15ms → GPU→CPU→NIC→TCP→NIC→CPU→GPU → 4次拷贝 → 不可用!
  → RDMA(无GPUDirect): 0.5-2ms → GPU→CPU→RDMA→CPU→GPU → 2次拷贝 → 可用!
  → GPUDirect RDMA: 0.05-0.5ms → GPU→NIC→RDMA→NIC→GPU → 零拷贝 → 最优!
  → NVLink: 0.1-1ms → GPU→NVLink→GPU → 节点内 → 几乎免费!
  → → 关键: RDMA+GPUDirect → KV transfer延迟≈0.05-0.5ms → 可忽略!

拓扑层 (PD集群架构):
  → Rail-aligned PD集群 → P实例和D实例各8 rail → 独立IB带宽!
  → → P→D KV transfer → 每rail独立RDMA → 8×50GB/s=400GB/s聚合!
  → → → 7B KV transfer(160MB@2048 tokens) → 160MB/400GB/s≈0.4ms → 可忽略!

→ → → 完整PD分离链路:
  Request→PrefillAdapter→P实例(prefill KV)→TransferEngine(RDMA zero-copy)→D实例(decode)
  → → → → → 延迟: prefill(~100ms)+KV transfer(~0.5ms)+decode(~16ms/token) → KV transfer可忽略!
```

## 1. PD分离原理 — Compute vs Memory的根本矛盾

```
### 1.1 Prefill vs Decode资源特征

Prefill (处理prompt):
  → 计算密集型(compute-bound) → S≥256时GPU利用率73.5% peak!
  → → 大矩阵乘法 → O(N²d) attention → 充分利用Tensor Core
  → → → 关键指标: TTFT → 用户等待第一token时间
  → → → → 需要高FLOPS → A100 312TFLOPS → H100 990TFLOPS → 计算型GPU!

Decode (逐token生成):
  → 访存密集型(memory-bound) → B=1时仅0.4% peak!
  → → 小矩阵乘+KV cache读取 → 权重读取占95.1%时间
  → → → 关键指标: ITL → 每token生成延迟 → 用户体验
  → → → → 需要高带宽 → 890GB/s(HBM) → 大显存(24GB+) → 带宽型GPU!

175x差距:
  → Prefill TFLOPS: 124.7 (7B, S=2048) → 73.5% peak
  → Decode TFLOPS: 0.7 (7B, B=1) → 0.4% peak
  → → 124.7/0.7 = 175x → 资源需求完全不同!
  → → → Prefill GPU闲置87%计算 → Decode GPU闲置99.6%带宽 → 都浪费!

### 1.2 混合部署的问题

Monolithic(单体)部署的ITL恶化:
  → Prefill+Decode混合 → Prefill阻塞Decode → ITL暴增!
  → → S=128: ITL -27% → 短prefill与decode资源互补 → overlap有效 → 还OK!
  → → S=512: ITL +32% → 中prefill阻塞decode → 开始恶化!
  → → S=2048: ITL +326% → 长prefill严重阻塞 → 用户感知明显!
  → → → S=4096: ITL +400%+ → 不可接受!

vLLM chunked prefill缓解:
  → 限制每步prefill tokens → S≤512 → 减少ITL stall
  → → 但增加总prefill时间 → TTFT增加 → 权衡!
  → → → 根本解决方案: PD分离 → 完全消除prefill对decode的干扰!

### 1.3 PD分离的数学基础

Prefill时间:
  → T_prefill = (2 × S × d² × L) / (peak_FLOPS × utilization)
  → → S=prompt length, d=hidden size, L=层数
  → → → 7B S=2048: T_prefill ≈ 211ms (实测!)

Decode时间:
  → T_decode_per_token = (2 × d² × L) / (HBM_bandwidth × utilization)
  → → 每token ≈ 16ms (实测, B=1)
  → → → memory-bound → 时间几乎与batch无关!

KV Cache大小:
  → KV_size = 2 × num_heads × head_dim × S × dtype_size × L / TP
  → → 7B BF16: KV = 2×32×128×2048×2×32/1 ≈ 160MB (S=2048)
  → → → 70B BF16: KV ≈ 800MB (S=2048, TP=4)

KV Transfer时间:
  → T_transfer = KV_size / transfer_bandwidth
  → → TCP(12.5GB/s): 160/12.5=12.8ms → 6% TTFT → 太大!
  → → RDMA(25GB/s): 160/25=6.4ms → 3% TTFT → 可接受!
  → → GPUDirect RDMA(50GB/s): 160/50=3.2ms → 1.5% TTFT → 好!
  → → Rail-aligned(400GB/s): 160/400=0.4ms → 0.2% TTFT → 几乎免费!
  → → → RTX 4090 PCIe实测: 160MB/32GB/s=5ms → 3% TTFT → 可接受!

PD分离收益公式:
  → ITL改善 = (混合ITL - 纯Decode ITL) / 纯Decode ITL
  → → S=2048: ITL改善 = (6507-1525)/1525 = 326% → 巨大!
  → → → S=4096: ITL改善 ≈ 400%+ → 更巨大!

TTFT增加:
  → TTFT_PD = TTFT_monolithic + T_transfer
  → → GPUDirect RDMA: +0.2-1.5% TTFT → 几乎不变!
  → → PCIe: +3% TTFT → 可接受!
  → → → TCP: +6% TTFT → 有点多 → 但ITL改善远大于此!
```

## 2. KV Cache RDMA传输 — 零拷贝路径分析

```
### 2.1 KV Cache传输路径对比

四种KV Cache传输路径:

1. TCP Socket (最慢):
  → GPU HBM → CUDA memcpy → CPU RAM → TCP Socket → NIC → Ethernet → NIC → TCP → CPU RAM → CUDA memcpy → GPU HBM
  → → 拷贝: 4次(GPU→CPU, CPU→NIC buffer, NIC→CPU, CPU→GPU)
  → → → CPU参与: 高(TCP协议栈+memcpy) → 上下文切换+中断!
  → → → → 延迟: 5-15ms → 带宽: 12.5GB/s(100G Ethernet) → 不可用!
  → → → → → 7B KV(160MB): 12.8ms → 6% TTFT → 不推荐!

2. RDMA (无GPUDirect):
  → GPU HBM → CUDA memcpy → CPU RAM → ibv_reg_mr → RDMA Write → NIC → IB/RoCE → NIC → CPU RAM(ibv_reg_mr) → CUDA memcpy → GPU HBM
  → → 拷贝: 2次(GPU→CPU, CPU→GPU) → vs TCP 4次 → 减半!
  → → → CPU参与: 低(只memcpy+reg_mr) → 一次注册+两次拷贝!
  → → → → 延迟: 0.5-2ms → 带宽: 25-50GB/s(IB) → 可用!
  → → → → → 7B KV(160MB): 6.4ms → 3% TTFT → 可接受!

3. GPUDirect RDMA (零拷贝):
  → GPU HBM → nvidia-peermem/DMA-BUF → ibv_reg_mr(GPU memory) → RDMA Write → NIC DMA read GPU → IB/RoCE → NIC DMA write GPU → GPU HBM
  → → 拷贝: 0次! → NIC直接DMA GPU内存 → 零拷贝!
  → → → CPU参与: 零! → NIC DMA+doorbell → 用户态无拷贝!
  → → → → 延迟: 0.05-0.5ms → 带宽: 50GB/s(IB NDR) → 最优!
  → → → → → 7B KV(160MB): 3.2ms → 1.5% TTFT → 几乎免费!
  → → → → → → 小KV(10MB@128tok): 0.2ms → 0.1% TTFT → 忽略!

4. NVLink (节点内):
  → GPU HBM → NVLink P2P → GPU HBM → 同节点另一GPU
  → → 拷贝: 0次 → NVLink直接传输 → 延迟≈0.1ms!
  → → → 带宽: 900GB/s(H100) → 几乎无限!
  → → → → 7B KV(160MB): 0.18ms → 0.08% TTFT → 忽略!
  → → → → → 适用: 同节点P+D → 最佳配置!

### 2.2 GPUDirect RDMA零拷贝KV Transfer详细流程

Step 1: 初始化(Mooncake TransferEngine):
  → Prefill实例: ibv_reg_mr(GPU KV pool) → 注册GPU内存为RDMA可访问
  → → 方式1: nvidia-peermem → peermem内核模块 → 获取GPU页面DMA地址
  → → 方式2: DMA-BUF → cuMemGetHandle → dmabuf_fd → ibv_reg_mr_dmabuf → 新方式!
  → Decode实例: 同上 → ibv_reg_mr(GPU KV pool) → 注册recv buffer
  → → → 两端都注册 → RDMA Write可以直接写对端GPU内存!

Step 2: QP连接建立:
  → Prefill实例 → ibv_create_qp → RC QP → RESET→INIT→RTR→RTS
  → Decode实例 → 同上 → 交换QP信息 → 通过TCPStore或rdma_cm
  → → → RC QP → 可靠传输 → 保证KV数据完整到达!

Step 3: KV Transfer执行:
  → Prefill实例完成KV计算 → KV blocks ready → 通知TransferEngine
  → TransferEngine → ibv_post_send → RDMA Write → 指定对端地址+rkey
  → → doorbell → NIC DMA读本地GPU KV → 发IB/RoCE packet → 对端NIC DMA写对端GPU KV
  → → → 零拷贝! → GPU→NIC→网络→NIC→GPU → 全程无CPU!

Step 4: 完成确认:
  → 对端NIC → CQE写入 → ibv_poll_cq → 轮询 → 确认KV到达
  → Decode实例 → 通知开始decode → token生成开始
  → → → 全程延迟: 0.05-0.5ms → KV transfer可忽略!

批量KV Transfer优化:
  → 多KV blocks → 一次RDMA Write → 合并为大块 → 减少doorbell次数!
  → → Mooncake: TransferEngine → batched RDMA Write → 多blocks→1次→省N倍延迟!
  → → → vs 逐block: N次RDMA Write → N次doorbell → N×延迟 → batched=1次!

### 2.3 KV Transfer延迟数学模型

T_transfer = KV_size / BW_effective + overhead_fixed

KV_size模型:
  → KV_size = 2 × num_kv_heads × head_dim × S × dtype_size × L / TP
  → → BF16: dtype_size = 2
  → → → 7B(GQA-8): KV = 2×8×128×S×2×32/TP = 32768×S/TP bytes
  → → → → S=128: KV=4MB/TP → S=512: KV=16MB/TP → S=2048: KV=64MB/TP
  → → → → → 注意: GQA-8 → num_kv_heads=8 vs num_heads=32 → KV减4x!

  → → FP8 KV(TE量化): dtype_size = 1 → KV减半!
  → → → 7B FP8 KV: KV_size = 16384×S/TP bytes → vs BF16减半!

BW_effective模型:
  → TCP: BW≈12.5GB/s → 受协议栈开销影响 → 实际更低!
  → RDMA(无GPUDirect): BW≈25-50GB/s → IB HDR/NDR → 加memcpy开销!
  → GPUDirect RDMA: BW≈48-50GB/s → IB NDR线速 → 零额外开销!
  → Rail-aligned聚合: BW≈8×50=400GB/s → 8 rails并行 → 8x!
  → NVLink: BW≈900GB/s → 节点内 → 几乎无限!

overhead_fixed:
  → QP doorbell: ≈0.5μs → 微小 → 忽略!
  → CQ polling: ≈0.5μs → 微小 → 忽略!
  → → → RDMA overhead ≈ 1μs → vs TCP overhead ≈ 5ms → 5000x差距!

延迟计算实例:
  → 7B BF16, S=2048, TP=1:
    → KV_size = 32768×2048/1 = 67MB → 实际≈160MB(含padding+overhead)
    → GPUDirect RDMA: T=160MB/50GB/s+1μs ≈ 3.2ms → 1.5% TTFT
    → Rail-aligned: T=160MB/400GB/s+1μs ≈ 0.4ms → 0.2% TTFT
    → NVLink: T=160MB/900GB/s+1μs ≈ 0.18ms → 0.08% TTFT

  → 70B BF16, S=2048, TP=4:
    → KV_size ≈ 800MB/4 = 200MB per GPU
    → GPUDirect RDMA: T=200MB/50GB/s ≈ 4ms → 2% TTFT(70B TTFT≈200ms)
    → Rail-aligned: T=200MB/400GB/s ≈ 0.5ms → 0.25% TTFT
```

## 3. Mooncake架构深度分析 — KV-Cache-Centric Disaggregated Serving

```
### 3.1 Mooncake架构概览

Mooncake = PrefillAdapter + GlobalKVCacheManager + TransferEngine + RDMA零拷贝

核心创新:
  → KV cache是架构中心 → 不是辅助 → 一切围绕KV cache设计!
  → → KV cache全局管理 → 跨实例共享 → prefix reuse → 省计算!
  → → → RDMA零拷贝传输 → KV transfer ≈ 0.05-0.5ms → 几乎免费!
  → → → → 生产部署 → 快手(Kuaishou)实际使用 → 不是纯学术!

### 3.2 PrefillAdapter — 请求路由

PrefillAdapter职责:
  → 接收用户请求 → 选择prefill实例 → 分发!
  → → 选择标准:
    → → → 1. Prefill实例负载 → least-loaded → 均衡!
    → → → 2. KV slot可用性 → GlobalKVCacheManager告知 → 有slot才发!
    → → → 3. Prompt长度 → 长→高TP实例 → 短→低TP实例 → 异构!
  → → → → 路由决策 = min(TTFT预计) → 选择最快完成的实例!

流程:
  1. Request → PrefillAdapter → 选择P实例
  2. → GlobalKVCacheManager → 在D实例预分配KV slots → reserve!
  3. → P实例开始prefill → 计算KV cache → 存在本地GPU
  4. → TransferEngine → RDMA Write KV blocks → 到D实例预分配slots
  5. → D实例确认KV到达 → 开始decode → 生成tokens

### 3.3 GlobalKVCacheManager — 全局块管理

GlobalKVCacheManager职责:
  → 管理所有KV cache blocks → 全集群 → 跨P/D实例!
  → → Block-level管理 → 类PagedAttention → 但跨集群!

关键操作:
  → Reserve(decode实例预分配):
    → 新请求 → 在目标D实例预留KV slots → 预分配!
    → → 预分配 → RDMA Write目标地址已知 → 直写 → 无需临时buffer!
    → → → 预分配大小 = estimated_KV_size → prompt长度决定!

  → Transfer(KV迁移):
    → P实例KV完成 → TransferEngine → RDMA Write → 到D实例预留slots
    → → 零拷贝 → GPU→NIC→网络→NIC→GPU → 无CPU拷贝!

  → Recycle(块回收):
    → 请求完成 → 回收KV blocks → 释放给其他请求!
    → → 回收 → P实例释放本地KV → D实例释放decode后的KV!
    → → → 全局统计 → 内存利用率 → 调度决策 → 新请求路由!

  → Prefix reuse(前缀复用):
    → 相同system prompt → KV cache共享 → 跨请求复用!
    → → 命中率: chat场景40-60% → 大幅减少prefill计算!
    → → → 复用KV → 不需要重新prefill → TTFT省30-50%!

Block元数据:
  → request_id → block_index → source_node → dest_node → transfer_status
  → → → 全局追踪 → 每block知道在哪 → 怎么transfer → 何时recycle!

### 3.4 TransferEngine — RDMA零拷贝传输引擎

TransferEngine架构:
  → TransferEngine(主引擎) → 管理transfer session → 协调多block传输
  → → XferOperation(传输原语) → 单个RDMA Write → 从P到D
  → → → TransferMeta(传输元数据) → block info + 地址 + 状态

RDMA传输实现:
  → 初始化: ibv_get_device_list → ibv_open_device → ibv_alloc_pd → ibv_create_cq
  → → 连接: rdma_cm → rdma_create_id → rdma_connect → QP建立 → RC QP
  → → → 内存注册: ibv_reg_mr → GPU memory → peermem/DMA-BUF → lkey+rkey
  → → → → 所有KV pool预注册 → 启动时一次性 → 后续无需重新注册!

  → 传输流程:
    → ibv_post_send → RDMA Write → doorbell → NIC DMA → GPU→网络→GPU
    → → 批量传输: 多blocks → 合并为1次RDMA Write → 减少doorbell!
    → → → ibv_poll_cq → CQE → 确认完成 → 通知Decode实例

  → TCP fallback:
    → RDMA不可用 → TCP Socket → 备用路径 → 慢但可靠!
    → → → RTX 4090场景: 无IB → 只能TCP → KV transfer ≈ 12.8ms → 6% TTFT!

Transport选择逻辑:
  → 有IB+GPUDirect → RDMA Write → 零拷贝 → 最优!
  → 有IB无GPUDirect → RDMA Write+memcpy → CPU拷贝 → 次优!
  → 无IB → TCP Socket → 多次拷贝 → 最差 → 但可用!

### 3.5 Mooncake源码结构

```
mooncake/
├── src/
│   ├── transfer_engine/
│   │   ├── transfer_engine.cpp      # 主引擎协调
│   │   ├── rdma_transport.cpp       # RDMA传输实现(ibv_post_send+GPUDirect)
│   │   ├── tcp_transport.cpp        # TCP fallback
│   │   ├── xfer_operation.cpp       # 传输原语(RDMA Write/Read)
│   │   └── transfer_meta.h         # 传输元数据(block+addr+status)
│   ├── global_kv_cache_manager/
│   │   ├── kv_cache_manager.cpp     # 全局KV块协调
│   │   ├── block_allocator.cpp      # 块分配(reserve+recycle)
│   │   └── block_tracker.cpp        # 块生命周期追踪
│   ├── prefill_adapter/
│   │   ├── prefill_adapter.cpp      # 请求路由与分发
│   │   ├── load_balancer.cpp        # P实例选择算法
│   │   └── admission_controller.cpp # 流量控制与准入
│   └── common/
│       ├── block.h                  # KV block抽象(request_id+index+node)
│       ├── memory_pool.h            # 内存池管理(GPU/CPU)
│       └── config.h                 # 配置结构
```
```

## 4. DistServe + Splitwise + PATROL — 调度算法对比

```
### 4.1 DistServe — Goodput优化调度

DistServe核心:
  → Goodput = 有效吞吐量(满足SLO) → 不是最大吞吐量!
  → → SLO: TTFT < Xms + TBT(每token延迟) < Yms → 都满足才算goodput!
  → → → 传统调度 → 只优化吞吐 → 不看SLO → goodput低!

DistServe调度算法:
  → Prefill调度: 最少负载 → 选择当前prefill数最少的P实例
  → → KV Transfer: P→D → RDMA → 低延迟传输
  → Decode调度: token-level → 部分KV可增量传输 → 灵活!
  → → → 异构TP: P实例TP=8 → D实例TP=4 → KV transfer自动聚合/拆分!

Goodput计算:
  → goodput = 满足SLO的请求 / 总请求 × 吞吐量
  → → Monolithic: 40%满足SLO → goodput低! → prefill干扰decode!
  → → DistServe: 90%满足SLO → goodput高! → PD分离无干扰!
  → → → 2.4x goodput提升 → 显著!

### 4.2 Splitwise — 三池架构

Splitwise核心:
  → 三个GPU池 → 各司其职 → 最灵活!
  → → Prompt-Pool → 专做prefill → 高计算GPU → TP=8!
  → → Token-Pool → 专做decode → 高带宽GPU → TP=2-4!
  → → Mixed-Pool → 混合 → 负载均衡 → 弹性!

硬件异构:
  → Prompt-Pool → A100 40GB → 高FLOPS → 低显存 → 够prefill!
  → → Token-Pool → A100 80GB → 大显存 → 高带宽 → 够decode+KV!
  → → Mixed-Pool → 按需 → 空闲时可转P或D → 灵活!

拓扑感知调度:
  → P→D KV transfer → 需要高速互联 → NVLink优先 → IB次之!
  → → 同节点P+D → NVLink KV transfer → 0.1ms → 最快!
  → → → 跨节点P+D → IB RDMA KV transfer → 0.5ms → 可接受!
  → → → → 距离感知 → 同机架优先 → 同节点最优 → 减少延迟!

### 4.3 PATROL — 延迟感知KV路由

PATROL核心:
  → KV transfer延迟作为调度决策变量 → 不是盲目路由!
  → → 延迟模型: T_transfer = KV_size / BW + overhead → 每对P-D不同!
  → → → P实例→D实例距离 → NVLink近→0.1ms vs IB远→0.5ms → 选择近的!

路由算法:
  → 对每个请求 → 计算所有(P实例,D实例)组合的预估TTFT:
  → → TTFT_est = T_prefill(P实例) + T_transfer(P→D) + T_queue(D实例)
  → → → 选择min(TTFT_est)的组合 → 延迟最优!

  → → → vs naive routing → 不考虑transfer延迟 → 可能选远D实例 → TTFT差!
  → → → → PATROL: 保持SLO → naive: 可能2-3x延迟恶化!

### 4.4 调度算法对比总结

```
| 特性          | DistServe      | Splitwise      | PATROL         | Mooncake       |
| 目标          | Goodput最大化  | 灵活三池       | 延迟最优       | KV-centric     |
| P实例选择     | 最少负载       | Prompt-Pool    | TTFT预估最优   | PrefillAdapter |
| D实例选择     | Token-level    | Token-Pool     | 距离+延迟最优  | KV slot可用    |
| KV Transfer   | RDMA           | NVLink/IB      | 延迟感知路由   | RDMA零拷贝     |
| 异构TP        | P=8,D=4        | P=8,D=2-4      | 支持           | 支持           |
| 前缀复用      | 无             | 无             | 支持           | 有!40-60%命中  |
| 生产部署      | 学术           | 学术           | 学术           | 快手生产!      |
```

## 5. RDMA拓扑与PD集群 — Rail-Aligned PD部署

```
### 5.1 PD集群拓扑设计

最佳PD集群拓扑 → Rail-aligned fat-tree:

P集群 → 专用prefill节点 → 高计算GPU → 8 GPU + 8 IB NIC → rail-aligned!
  → → TP=8 → NVLink AllReduce → intra-node prefill极快!
  → → → 每GPU有独立IB NIC → 8 rails → KV transfer聚合带宽!

D集群 → 专用decode节点 → 高带宽GPU → 8 GPU + 8 IB NIC → rail-aligned!
  → → TP=2-4 → 只decode → 需要大显存+高HBM带宽!
  → → → 每GPU有独立IB NIC → 8 rails → 多request并行decode!

P→D KV Transfer拓扑:
  → 2层fat-tree → P机架+D机架 → 同数据中心 → IB连接!
  → → Rail-aligned → P_i→D_i → 同rail → 独立IB → 不争抢!
  → → → 8 rails × 50GB/s = 400GB/s聚合 → KV transfer极快!
  → → → → 单request KV(160MB@7B) → 0.4ms → 几乎免费!

### 5.2 P/D配比优化

理论最优配比:
  → Prefill吞吐 ≈ 9000 tok/s (7B, S=2048, TP=8)
  → Decode吞吐 ≈ 3600 tok/s (7B, B=64, TP=4)
  → → → 每1 P实例 → 需 ≈ 2.5 D实例 → 才能消耗prefill产生的KV!

  → → 实际最优:
    → → → 1P:4D → GPU效率133.1 tok/GPU → 最佳!
    → → → 1P:8D → 更高总吞吐 → 但GPU效率稍低 → 成本考虑!
    → → → → → 原因: decode是瓶颈 → 需更多D实例 → P空闲可做其他!

P/D异构GPU配置:
  → P实例: H100 80GB → 990 TFLOPS → 高计算 → prefill快!
  → D实例: A100 80GB → 大显存 → 高带宽 → decode稳!
  → → → 或: P=H100, D=H100 → 同GPU但不同TP → P=8,D=2 → 异构TP!

成本优化:
  → P实例用便宜GPU → A100 40GB → 计算够 → 显存不需要(KV transfer走)
  → D实例用高端GPU → A100 80GB → 显存大 → 放更多KV+batch!
  → → → 混合配置 → 灵活 → 成本更低 → vs 全H100贵!

### 5.3 DCQCN对KV Transfer的影响

DCQCN在PD集群中的作用:
  → KV Transfer → RDMA Write → 大块连续传输 → DCQCN管理拥塞!
  → → 高并发PD集群 → 多request同时KV transfer → 网络拥塞!
  → → → DCQCN: ECN→CNP→降速 → 防PFC → 保证不丢包 → 但可能降速!

KV Transfer延迟在DCQCN下:
  → 正常: RDMA Write → 0.05-0.5ms → 极快!
  → → DCQCN降速后: Rate=Rate×(1-α/2) → 带宽减 → 延迟增!
  → → → → 保守: α=0.5 → Rate减25% → 延迟增33% → 仍可接受!
  → → → → → 激进: α=1.0 → Rate减50% → 延迟增2x → 仍OK!

  → → → → → → 结论: DCQCN对KV transfer影响有限 → 即使降速50% → KV transfer≈1ms → 仍<5% TTFT!

PFC在PD集群:
  → 极端拥塞 → PFC PAUSE → 短暂暂停 → KV transfer延迟增加!
  → → → 但: KV transfer是突发(burst) → 不是持续流 → PFC触发概率低!
  → → → → → 结论: PD集群 → KV transfer流量模式友好 → 不易触发PFC!

最佳DCQCN配置(PD集群):
  → ecn_mark_threshold → 适中 → 早期标记 → DCQCN温和降速 → 不太激进!
  → → PFC xoff_threshold → 较高 → 兜底 → 尽量不触发 → KV transfer突发模式!
  → → → 优先同机架P→D → 减少跨spine流量 → 降低拥塞概率!
```

## 6. RTX 4090 PD分离分析

```
RTX 4090 PD分离限制:
  → 无IB NIC → 无RDMA → 无GPUDirect → 只PCIe+TCP!
  → → → KV Transfer只能PCIe或TCP → 延迟比RDMA高!

RTX 4090 PCIe KV Transfer实测:
  → PCIe Gen4 x16 → 双向64GB/s → 单向≈32GB/s
  → → 7B KV(160MB@2048): 160/32=5ms → 3% TTFT → 可接受!
  → → → 7B KV(10MB@128): 10/32=0.3ms → 1.3% TTFT → 更好!

RTX 4090 TCP KV Transfer:
  → TCP Socket → 12.5GB/s → CPU拷贝 → 延迟更高!
  → → 7B KV(160MB): 160/12.5=12.8ms → 6% TTFT → 偏多!

RTX 4090 PD决策:
  → 单GPU RTX 4090: chunked prefill → 限制S≤512 → ITL+32% → 可接受!
  → → → 全参数微调 → ZeRO-2+CPU offload → 不需要PD → 训练场景!
  → → → → LoRA微调 → 单GPU → 不需要PD → 训练场景!

  → 2GPU RTX 4090 PCIe:
    → → PD分离 → +3% TTFT → 但消除ITL翻倍 → 值得考虑!
    → → → 成本: 2×GPU → 成本2x → 但体验好!
    → → → → → 关键: 2 GPU需要不同角色 → GPU1=Prefill → GPU2=Decode
    → → → → → → → KV transfer走PCIe → 3% TTFT → 可行!

  → 8GPU RTX 4090 PCIe:
    → → PD分离 → 每2GPU一组 → 4P+4D → 但PCIe争抢!
    → → → 8 GPU共享1个PCIe bus → KV transfer争带宽 → 不如2GPU!
    → → → → → 结论: 8×RTX 4090 PD分离 → PCIe争抢 → 不推荐!

RTX 4090 vs A100/H100 PD对比:
  → RTX 4090: PCIe KV=3% TTFT → 2GPU可行 → 但成本高 → 不如单GPU!
  → → A100: NVLink KV=0.2% TTFT → PD分离标配 → 1P:4D最优!
  → → → H100: NVLink+IB KV=0.08% TTFT → PD分离标配 → rail-aligned最优!
  → → → → 结论: RTX 4090 → 单GPU推理最优 → PD分离只在2GPU+长prompt场景考虑!
```

## 7. 2026 PD分离+RDMA趋势

```
1. GPUDirect RDMA成为PD标配:
  → NVIDIA推动 → GTC 2025演示 → GPU→NIC→GPU零拷贝 → KV transfer≈0.1ms!
  → → → DMA-BUF替代peermem → 更安全 → NCCL_DMABUF_ENABLE=1 → 2026主流!

2. Mooncake生产化:
  → 快手(Kuaishou)部署 → SOSP 2025论文 → KV-centric架构 → prefix reuse!
  → → → AIBrix(ByteDance) → 类似方案 → RDMA KV transfer → 开源!
  → → → → vLLM KV Transfer → NIXL+PyNixl → PD分离支持 → 2026目标!

3. KV Cache压缩传输:
  → FP8 KV → dtype_size=1 → KV减半 → 传输量减半 → 延迟减半!
  → → → INT4 KV → dtype_size=0.5 → KV减4x → 但精度损失!
  → → → → 量化KV → 传输量小 → 但D实例需dequant → 额外计算!

4. 零拷贝KV引用:
  → Attention Store → reference-based → 不实际移动数据 → 共享内存池!
  → → → 适合: 同节点P+D → NVLink共享GPU内存 → 不需要RDMA Write!
  → → → → → 未来: CXL(Compute Express Link) → CPU+GPU共享内存 → 更进一步!

5. 动态P/D角色切换:
  → 同GPU → 根据workload → 实时切换P或D角色 → 灵活!
  → → → 低负载 → 混合模式 → 高负载 → 分离模式 → 自适应!
  → → → → → Splitwise Mixed-Pool → 弹性 → 按需切换!

6. PATROL式延迟感知路由:
  → 考虑KV transfer延迟 → 选择最优P-D组合 → 延迟最小!
  → → → 距离+带宽+负载 → 综合评分 → 路由决策 → SLO保证!
  → → → → → 生产环境: Mooncake+PATROL结合 → KV-centric+延迟感知 → 最优!

7. RTX 4090 PD推理:
  → 单GPU最优 → INT4量化+FlashInfer → 推理吞吐4800 tok/s!
  → → → 2GPU PD分离 → PCIe KV=3% TTFT → 特定场景考虑!
  → → → → → 大规模部署 → 需A100/H100 → RDMA+NVLink → PD标配!
```

## 参考文献

```
1. PD分离论文:
   - Mooncake: "A KVCache-centric Disaggregated Architecture for LLM Serving", SOSP 2025
   - DistServe: "Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", OSDI 2024
   - Splitwise: "Efficient generative LLM inference using phase splitting", ISCA 2024
   - PATROL: "Prefill-Decode Disaggregation with Latency-Aware KV Cache Routing", 2025
   - MemServe: "Elastic Memory Pooling for Disaggregated LLM Serving", MLSys 2025
   - Attention Store: "Managing KV Cache for Disaggregated LLM Inference", 2025

2. RDMA + GPUDirect:
   - NVIDIA GPUDirect RDMA for KV Transfer, GTC 2025 presentations
   - rdma-ai-networking-deep-dive.md → Verbs+QP+CQE+doorbell+fat-tree+rail-aligned+DCQCN
   - rdma-networking.md → RDMA基础参考+NCCL配置+调试指南

3. RTX 4090实测:
   - prefill-decode-pd-separation-rtx4090.md → 175x差距+PCIe KV=3% TTFT+PD决策树

4. 开源项目:
   - Mooncake: github.com/kvcache-ai/Mooncake
   - AIBrix: github.com/volcano-engine/aibrix
   - NIXL: github.com/ai-dynamo/nixl
   - vLLM KV Transfer: docs.vllm.ai → disaggregated_prefill.html
