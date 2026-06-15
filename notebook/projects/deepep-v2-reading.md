# DeepEP V2 — 高效 Expert Parallelism 通信库 (源码级深度分析)

> deepseek-ai/DeepEP | V2 公开发布: 2026-04-26 (commit b306af06) | MIT License | 版本 2.0.0
> 源码路径: rollout-infra/deepep-src/ | GitHub: https://github.com/deepseek-ai/DeepEP
> 作者: Chenggang Zhao, Shangyan Zhou 等 | NCCL Gin backend 重构自 NVSHMEM

## 1. 概述

★★★★★ DeepEP = DeepSeek 开源的高性能 Expert Parallelism 通信库 — 不是通用集合通信(NCCL), 而是针对 MoE dispatch/combine 的 asymmetric 通信优化。

V2 是完全重构版本 (2026-04-26 公开发布), 相比 V1:
- NVSHMEM → NCCL Gin backend (更轻量, header-only, 复用 NCCL communicator)
- 高吞吐+低延迟统一为 ElasticBuffer 接口 (V1 分 Buffer/LowLatencyBuffer)
- 分析式 SM/QP 计算 — 无需 auto-tuning (V1 需要)
- SM 使用从 24 降至 4-6, 峰值 1.3x V1
- EP2048 支持 (V1 最大 EP256)
- JIT 编译 → 无安装时 CUDA 编译
- 新增 Engram (RDMA remote fetch) / PP (pipeline) / CP (copy engine) / AGRS (all-gather/reduce-scatter)

### V2 vs V1 关键对比

| 维度 | V1 (NVSHMEM) | V2 (NCCL Gin) |
|------|-------------|---------------|
| **Backend** | NVSHMEM (需独立安装) | NCCL Gin (header-only, 复用NCCL comm) |
| **API** | Buffer (normal) + LowLatencyBuffer | ElasticBuffer (统一) |
| **SM 计算** | Auto-tuning | Analytical (get_theoretical_num_sms) |
| **SM 使用 (V3训练)** | 24 SM | 4-6 SM (4x节省) |
| **峰值性能** | Baseline | 1.3x V1 |
| **最大 EP** | 256 | 2048 |
| **编译** | 预编译 + JIT | Pure JIT |
| **低延迟 RDMA** | 0 SM (RDMA mode) | 不再支持 0 SM RDMA 低延迟 |
| **Handle 缓存** | 无 | EPHandle 可缓存 → decode 推理复用 |
| **FP8** | 有 | 有 + TMA-aligned col-major SF |
| **Hybrid mode** | 有 | 有 + 更大 scaleup/scaleout 域 |
| **Engram/PP/CP** | 无 | 有 (实验性) |
| **AGRS** | 无 | 有 (实验性) |

## 2. 核心架构

### 2.1 ElasticBuffer (V2 统一接口)

★★★★★ ElasticBuffer = 一次初始化支持训练+推理所有场景:

```python
buffer = ElasticBuffer(
    group,                          # NCCL ProcessGroup
    num_max_tokens_per_rank=8192,   # 每个rank最大token数
    hidden=7168,                    # 隐藏维度 (DeepSeek-V3)
    num_topk=8,                     # Top-K 专家数
    use_fp8_dispatch=False,         # FP8 dispatch
    allow_hybrid_mode=True,         # Hybrid mode (默认开启)
    allow_multiple_reduction=True,  # 多次reduction (默认开启)
    prefer_overlap_with_compute=True, # 优先overlap (默认开启)
)
num_comm_sms = buffer.get_theoretical_num_sms(num_experts, num_topk)
```

关键设计:
- 初始化后自动分析最优 SM/QP 数量 → 无需调参
- NCCL communicator 复用 → 不需要额外 NVSHMEM 安装
- 逻辑域 vs 物理域分离 → scaleout(NVLink域内)/scaleup(RDMA域) 抽象
- CPU buffer 可选 (num_cpu_bytes) → Engram remote memory + 不平衡 EP

### 2.2 EPHandle (路由元数据缓存)

★★★★★ EPHandle = dispatch 返回的路由元数据, 可缓存 → decode 推理复用:

```python
class EPHandle:
    do_expand: bool              # 扩展模式 (one-slot-per-expert-per-token)
    num_experts: int             # 专家总数
    expert_alignment: int        # 每个local expert token数对齐值
    num_max_tokens_per_rank: int # 每个rank最大token数
    num_sms: int                 # dispatch使用的SM数 (combine复用)
    topk_idx: torch.Tensor       # 克隆的 top-k 专家索引 [num_tokens, num_topk]
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor  # scaleup域prefix-sum
    psum_num_recv_tokens_per_expert: torch.Tensor        # per-expert prefix-sum (alignment padded)
    recv_src_metadata: torch.Tensor    # 源token索引和buffer slot索引
    dst_buffer_slot_idx: torch.Tensor  # 目标buffer slot索引
    token_metadata_at_forward: Optional[torch.Tensor]  # hybrid mode转发元数据
    channel_linked_list: Optional[torch.Tensor]         # hybrid mode通道链表
    num_recv_tokens: int  # 接收token总数 (推断值, 无CPU sync时可能不精确)
```

### 2.3 Dispatch/Combine 流程

```
训练/Prefill:
  Dispatch Forward:
    x → buffer.dispatch(x, topk_idx, topk_weights, ...) → recv_x, handle, event

  Expert Compute:
    event.current_stream_wait()     # 等待通信完成
    y = expert(recv_x)              # 专家计算
    # 可以在 event 之前做独立计算 → 通信计算重叠!

  Combine Forward:
    buffer.combine(y, handle=handle, ...) → combined_x, event

  Backward:
    dispatch_backward = combine (反向是combine)
    combine_backward = dispatch (反向是dispatch)
```

### 2.4 通信-计算重叠 (EventOverlap)

```python
recv_x, ..., handle, event = buffer.dispatch(..., async_with_compute_stream=True)
independent_work()  # 不依赖 recv_x 的独立计算
event.current_stream_wait()  # 计算流插入等待
y = expert(recv_x)
```

EventOverlap 管理 compute stream 和 communication stream 的依赖:
- dispatch/combine 在通信流异步执行
- `event.current_stream_wait()` 在计算流插入等待
- 实现零开销的通信计算重叠

## 3. NCCL Gin Backend — 源码级分析

★★★★★ Gin Backend = DeepEP V2 核心通信引擎, 绕过NCCL proxy thread → GPU kernel直接 RDMA put/get/signal。

### 3.1 NCCLGin 结构体 (handle.cuh)

```cuda
struct NCCLGin {
    ncclDevComm_t nccl_dev_comm;
    ncclWindow_t nccl_window;
    ncclGin gin;            // NCCL Gin handle
    ncclTeam team_world;    // 全域 team (所有rank)
    ncclTeam team_lsa;      // NVLink域 team (intra-node)
    ncclTeam team_rail;     // RDMA域 team (inter-node rail)
    uint64_t lsa_base_ptr;  // NVLink域基址

    // 核心操作:
    put(recv_sym_ptr, send_sym_ptr, num_bytes, dst_rank_idx)
    get(src_ptr, dst_ptr, num_bytes, src_rank_idx)
    signal(dst_rank_idx, remote_action)
    red_add_rel(sym_ptr, value, dst_rank_idx)  // NVLink用atomic, RDMA用gin.signal(VA)
    flush() / flush_async(src_rank_idx, request)
    wait(request)
}
```

### 3.2 三种 Team 和资源共享模式

★★★★ NCCL Gin 提供 3 种 team:
- **ncclTeamTagWorld**: 全域 → 所有 rank
- **ncclTeamTagLsa**: NVLink域 → intra-node (scaleup)
- **ncclTeamTagRail**: RDMA域 → inter-node rail (scaleout)

★★★★ 资源共享模式 (ncclGinResourceSharingMode):
- **NCCL_GIN_RESOURCE_SHARING_CTA**: 线程块独占 → 每个CTA有自己的QP
- **NCCL_GIN_RESOURCE_SHARING_GPU**: GPU共享 → 所有CTA共享QP

QP分配策略 (get_qp_mode):
- 1个QP → GPU共享
- Notify warp → 1个QP, CTA独占
- SM <= QP: 每个 SM 独占一个 QP, CTA mode
- SM > QP: 所有SM共享所有QP, GPU mode

### 3.3 对称内存 (symmetric.hpp) — 3种策略

★★★★★ Symmetric Memory = GPU/CPU 对称分配, 所有 rank 同一偏移量可相互访问:

| 策略 | 实现 | 适用场景 |
|------|------|---------|
| **GPUSymmetricMemory** | ncclMemAlloc/ncclMemFree | 纯GPU (当前默认) |
| **ElasticSymmetricMemory** | CUDA VMM (cuMemCreate/cuMemMap) | GPU+CPU混合 (Engram) |
| **HybridElasticSymmetricMemory** | POSIX FD跨进程共享 + CUDA VMM | 多节点CPU段共享 |

★★★★ ElasticSymmetricMemory 内存布局:
- [GPU VRAM (前段)] + [CPU RAM / NUMA-local (后段)]
- 整个连续VA范围兼容 ncclCommWindowRegister

★★★★ HybridElasticSymmetricMemory 内存布局:
- [GPU VRAM (前段)] + [CPU rank0 | CPU rank1 | ... | CPU rank(N-1) (后段)]
- 每个rank创建NUMA-local CPU段 → 导出POSIX FD → 跨进程共享
- 使用 pidfd_open/pidfd_getfd 系统调用导入其他进程的FD

### 3.4 Gin Backend 初始化流程 (nccl.cu)

```
1. ncclCommInitRank → 创建NCCL communicator
2. ncclCommQueryProperties → 查询Gin支持 (ginType/railedGinType)
3. ncclDevCommCreate → 创建device communicator
   - ginContextCount = num_allocated_qps (默认65/129)
   - ginExclusiveContexts = true
   - ginQueueDepth = 1024
   - ginTrafficClass = sl_idx (RDMA SL)
   - ginSignalCount = num_ranks + 4 (barrier信号)
   - ginConnectionType = RAIL (hybrid) / FULL (direct)
4. ncclCommWindowRegister → 注册symmetric memory为NCCL window
5. ncclGetLsaDevicePointer → 获取所有NVLink域peer指针
```

★★★★ QP数量自动计算:
- Hybrid mode: 65 QPs (fast RDMA atomic) / 129 QPs (slow)
- Direct mode: 17 QPs
- 原因: Hybrid mode每个channel需独立QP → 8 channel * 8 SM + 1 notify = 65

### 3.5 为什么 Gin 比传统 NCCL 快?

```
传统NCCL All-to-All:
  → CPU proxy thread mediation → GPU→CPU→GPU round trip
  → 多SM占用 (NCCL channel model)
  → Symmetric假设 → padding浪费带宽
  → 每步CPU-GPU sync → 推理延迟高

DeepEP Gin Backend:
  → GPU kernel直接init RDMA → ncclGin.put/get/signal → 无CPU介入!
  → 4-6 SM → 更多SM留给计算
  → Asymmetric → 原生支持MoE不等量dispatch
  → Handle caching → decode推理跳过CPU sync
```

## 4. MoE Dispatch & Combine — 源码级详解

### 4.1 Dispatch: token路由到expert GPU

★★★★★ Dispatch kernel分两阶段:

```
Phase 1: dispatch_impl
  - Notify warps: 统计rank/expert token数量 → atomicAdd → layout计算
  - Dispatch warps: 写入symmetric send buffers → ncclGin.put → RDMA/NVLink

Phase 2: dispatch_copy_epilogue_impl
  - 从symmetric recv buffers → TMA load → 复制到output tensors
```

★★★★ 两种模式:
- **Direct** (单节点, num_scaleout_ranks==1): NVLink-only → 一组send/recv buffers
- **Hybrid** (多节点, num_scaleout_ranks>1): 分层 — RDMA跨节点 → NVLink域内转发 → channel tracking

### 4.2 Hybrid Dispatch (hybrid_dispatch.cuh, 36674 lines)

★★★★★ Hybrid Dispatch = RDMA + NVLink域内转发 → 三组 warp:

```
Notify warps:  统计scaleout/scaleup域的token数量
  → atomicAdd on shared memory → layout计算

Scaleout warps: RDMA put → 跨节点发送
  → 每个warp = 一个channel → ncclGin.put(rail team)
  → kNumMaxTokensPerChannel = ceil(kNumMaxTokensPerRank / kNumChannels)

Forward warps: NVLink域内转发 → RDMA收到的tokens转发到正确local expert rank
  → ncclGin.put(lsa team) 或 symmetric pointer write
  → token_metadata_at_forward + channel_linked_list → 跟踪转发链
  → TMA store 用于高效数据写入
```

★★★★ Buffer布局 (hybrid mode):
```
scaleup_buffer:    [num_scaleup_ranks][scaleout_ranks * max_tokens_per_rank]
scaleout_send_buf: [1][max_tokens_per_rank]  → 发送到RDMA
scaleout_recv_buf: [num_scaleout_ranks][channels * max_tokens_per_channel] → RDMA接收
```

### 4.3 Combine: expert输出回到原token位置

```
Phase 1: combine_impl
  - 读取expert outputs → 推入symmetric recv buffers → Gin put/signal

Phase 2: combine_reduce_epilogue_impl
  - Reduce接收数据 → weighted sum用top-k weights
  - 支持bias addition (0/1/2个bias tensor)
  - allow_multiple_reduction: scaleup域内先reduce → 减少跨节点流量
  - use_expanded_layout: one-slot-per-expert-per-token → GEMM-friendly
```

### 4.4 Handle Caching — decode推理关键优化

★★★★★ Handle Caching = EPHandle from previous dispatch → 缓存 → reuse:

```python
# 首次 decode: 正常 dispatch
recv_x, ..., handle, event = decode_dispatch(x, topk_idx, topk_weights, ...)

# 后续 decode: 缓存 handle, 跳过布局重算和 CPU 同步
recv_x, ..., handle, event = decode_dispatch(x, cached_handle=handle)
```

★★★★ 源码级机制 (elastic.py dispatch方法):
```
当 handle != None:
  → topk_idx = handle.topk_idx (不再需要新输入)
  → do_cpu_sync = False (跳过CPU-GPU同步 — 最大延迟来源!)
  → cached_num_recv_tokens = handle.num_recv_tokens (已知接收数量)
  → cached_psum/dst_buffer_slot_idx/token_metadata/channel_linked_list = handle的值
  → 跳过notify warps → 无layout重算
  → 跳过prefix-sum → slot indices reuse
  → 限制: num_experts/expert_alignment/num_max_tokens_per_rank 必须与handle一致
```

★★★★ 什么时候handle可以缓存?
- decode阶段: 同一请求的连续decode步 → gating决策稳定 → top-k expert选择不变
- 不能用于prefill/training: 每次输入不同 → routing不同 → handle不能复用

## 5. Asymmetric Routing — 为什么DeepEP比NCCL快4.6x

★★★★★ 核心洞察: MoE dispatch天然asymmetric → 不同expert收到不同数量token

### 5.1 NCCL All-to-All (Symmetric)

```
NCCL all-to-all假设:
  → 每个rank发等量数据给每个其他rank
  → 固定chunk大小 → padding浪费带宽
  → MoE场景: expert0收到1000 tokens, expert1收到50 tokens → padding到1000 → 950浪费!
```

### 5.2 DeepEP Asymmetric Dispatch

```
DeepEP直接解决asymmetric:
  → Notify warps: 统计实际token数量 → 无padding
  → Gin put: 按实际大小发送 → 精确带宽利用
  → Layout计算: prefix-sum per expert → 不需要等量
  → expert_alignment: 可选对齐(如128) → GEMM-friendly但非强制
```

### 5.3 4.6x优势量化

```
| Token Count | DeepEP (GB/s) | NCCL All-to-All (GB/s) | Improvement |
|-------------|---------------|------------------------|-------------|
| 2,048       | ~23.5         | ~5.1                   | 4.6x        |
| 4,096       | ~45.8         | ~9.9                   | 4.6x        |
| 8,192       | ~88.2         | ~19.2                  | 4.6x        |

5个根源:
1. Asymmetric vs Symmetric → padding消除
2. Bypass proxy thread → GPU直接RDMA
3. SM效率: 4-6 SM vs NCCL多SM → 98 SM留给计算
4. FP8 dispatch → 跨节点流量减半
5. Handle caching → decode推理消除CPU sync
```

## 6. FP8 Dispatch — 跨节点流量减半

★★★★ FP8 Dispatch = 激活值cast到float8_e4m3fn + per-token scale factor → RDMA传输减半

### 6.1 FP8 数据格式

```python
# Input: x = (fp8_data, scale_factors)
fp8_data:     [num_tokens, hidden], dtype=float8_e4m3fn  → 1 byte per element
scale_factors: per-token SF → quantization缩放因子

# vs BF16: 2 bytes per element → FP8 = 50% bandwidth!
```

### 6.2 TMA-aligned Column-major SF

★★★★★ use_tma_aligned_col_major_sf = True:
- Scale factors按TMA对齐的column-major布局排列
- 直接作为GEMM input加载 → 与DeepSeek-V3 FP8 tile-wise量化一致
- column-major → warp-level GEMM可以直接TMA load SF

### 6.3 SF Pack格式

```cuda
// csrc中: sf_pack_t = packed scale factor
// kNumSFPacks = ceil_div(hidden, 32)  → 每32个hidden维度1个SF pack
// SF按token行存储 → TMA-aligned
```

## 7. Hybrid Mode — NVLink Forwarding + RDMA Scaleout

★★★★★ Hybrid Mode = 分层通信 → RDMA跨节点 + NVLink域内转发 → 物理topology匹配

### 7.1 逻辑域 vs 物理域

```
物理域:
  num_rdma_ranks: RDMA可达的rank数 (跨节点)
  num_nvl_ranks: NVLink可达的rank数 (节点内)

逻辑域 (allow_hybrid_mode=True):
  num_scaleout_ranks = num_rdma_ranks → 跨节点域
  num_scaleup_ranks = num_nvl_ranks → 节点内域

逻辑域 (allow_hybrid_mode=False):
  num_scaleout_ranks = 1 → 无跨节点
  num_scaleup_ranks = num_ranks → 全部rank在节点内
```

### 7.2 Hybrid Barrier

★★★★★ GPU Barrier = scaleup + scaleout 双层并行 barrier:

```
SM0 → scaleup barrier (NVLink域内)
  → nvlink_barrier: atomic red_add_rel_sys + phase-based(even/odd)
  → 或 gin_barrier (如果非NVLink scaleup)

SM1+ → scaleout barrier (RDMA域)
  → gin_barrier: ncclGin.signal(SignalInc) + shadow counter + GDAKI read
  → rail team → 跨节点同步

grid_sync → 等所有SM完成
```

### 7.3 Forward机制 (RDMA → NVLink转发)

```
1. RDMA收到的tokens → scaleout_recv_buffer
2. Forward warps → 读recv buffer → TMA store到scaleup send buffer
3. ncclGin.put(lsa team) → NVLink域内转发到正确local expert rank
4. token_metadata_at_forward: 每个channel的转发token元数据
5. channel_linked_list: 通道级链表跟踪 → forward完成后通知
```

## 8. 性能 Benchmark

### 8.1 V2 (DeepSeek-V3 style: 8K tokens, 7168 hidden, top-8, FP8 dispatch)

| Architecture | NIC | Topology | Dispatch BW | Combine BW | SMs |
|-------------|-----|----------|-------------|------------|-----|
| SM90 (H100/H800) | CX7 IB | EP 8x2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 IB | EP 8x4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 (B200) | CX7 IB | EP 8x2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | NVLink | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (max) |
| SM100 | NVLink | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (min) |

★★★★ V2 vs V1: 峰值1.3x V1, SM省4x (V1=24SM → V2=4-6SM)

### 8.2 V1 Low-latency (decode推理, H800)

| EP size | Dispatch latency | RDMA BW | Combine latency | RDMA BW |
|---------|------------------|---------|-----------------|---------|
| 8 | 77 us | 98 GB/s | 114 us | 127 GB/s |
| 32 | 155 us | 48 GB/s | 273 us | 53 GB/s |
| 128 | 192 us | 39 GB/s | 369 us | 39 GB/s |

## 9. SM/QP 分析式计算 — 源码级

★★★★★ get_theoretical_num_sms() — 带宽建模估算最优SM数:

```python
# 核心逻辑:
sm_read, sm_write = 0, 0
rdma_traffic, nvlink_traffic = 0, 0

# 期望top-k分布 (balanced gate):
num_expected_topk = num_groups * (1 - comb(n-e, k) / comb(n, k))

# 读token: 1/num_expected_topk
sm_read += 1 / num_expected_topk

# Hybrid mode traffic:
rdma_traffic += scaleout部分
nvlink_traffic += scaleup部分 (1 - 1/scaleup_ranks)

# 取bound: rdma或nvlink哪个是瓶颈
bounded = max(rdma_traffic/rdma_gbs, nvlink_traffic/nvlink_gbs)

# SM数 = bounded_gbs/bounded_traffic * (sm_read/sm_read_gbs 或 sm_write/sm_write_gbs)
num_sms = max(4, ceil(num_sms * 1.25))  # 1.25x safety margin
num_sms = min(num_sms, num_device_sms)
```

★★★★ QP计算:
- Direct mode: min(num_sms, 9) → 减少DB ringing开销
- Hybrid mode: num_sms * 16 + 1 → 每个channel独立QP

★★★★ 默认SM读写带宽假设:
- sm_read_gbs = 200 GB/s (HBM读)
- sm_write_gbs = 50 GB/s (HBM写)

## 10. JIT 编译系统

★★★★ 所有V2 kernel运行时JIT编译:
- NVCC → CUBIN → hash缓存 → 无需预编译
- 缓存目录: $HOME/.deep_ep (可配置EP_JIT_CACHE_DIR)
- 支持PTX/SASS dump (EP_JIT_DUMP_ASM/PTX/SASS)
- 20+ 环境变量控制编译/调试

★★★★ JIT优势:
- 无安装时编译 → pip install即可用
- 适配不同GPU架构 → 同一包支持SM90/SM100
- 缓存 → 第二次运行无需重新编译
- 调试 → PTX/SASS dump方便性能分析

## 11. 实验性分支 — 最新状态 (2026-06-15)

★★★★ **重要修正**: 之前笔记错误记录 "Zero-copy已merged" — 实际PR #453仍然OPEN!

### 11.1 分支状态

| 分支/PR | 状态 | 详情 |
|---------|------|------|
| **Zero-copy** (PR #453) | ★★★ **仍然OPEN** (非merged!) | Tencent → buffer fusion + TMA offloading → 减SM → PyTorch→通信buffer零拷贝 |
| **Eager** (PR #437) | ★★★★ **已merged** (2025-09-30) | eager RDMA → 消除atomic RTT → IBV_WR_RDMA_WRITE single write |
| **Hybrid-EP** (branch) | ★★★★ **活跃开发** → Megatron v0.15已集成 | TMA指令 + IBGDA + PCIe kernel + NVFP4 |
| **AntGroup SMFree** (PR #347) | ★★★ **仍然OPEN** | SMFree RDMA → 解耦comm-kernel与NIC token transfer → SM留给计算 |
| **AntGroup LL-SBO** (PR #483) | ★★★★ **已merged** (2025-11-21) → antgroup-opt branch | Single Batch Overlap: Down GEMM与Combine Send重叠 → 减e2e延迟 |
| **AntGroup LL-Layered** (PR #500) | ★★★★ **已merged** (2025-12-04) → antgroup-opt branch | 分层dispatch → 减少跨节点重复数据传输 → rail优化 |
| **Mori-EP** (branch) | ★★★ 实验/社区 | ROCm/AMD GPU → MORI backend → 低latency mode |
| **tencent-zcopy** (branch) | 存在 | PR #453的源分支 |

### 11.2 社区fork

| Fork | 用途 |
|------|------|
| **uccl/uccl-ep** | 异构GPU(NVIDIA+AMD)+多NIC(EFA/Broadcom/CX7) → 统一EP通信 |
| **Infrawaves/DeepEP_ibrc_dual-ports_multiQP** | 双端口IB+multi-QP → 跨节点带宽聚合 |
| **antgroup/DeepXTrace** | slow-rank诊断 → straggler检测 → 生产运维 |
| **ROCm/mori** | AMD下一代通信库 (Wide EP + KVCache + Collectives) |

### 11.3 近期修复 (2026-05~06)

```
#642: TMA fence fix — fence.proxy.async.shared::cta between mbarrier wait and TMA load
#640: NCCL lib name fix — match "libnccl" instead of "nccl"
#641: Internode dispatch args fix — missing num_worst_tokens arg
#630: V2初始化fix — 纯单节点intra EP初始化问题
#664: 实验性分支intro — 添加分支描述到README
#665: PP multi-QP support (open, 2026-06-15)
```

## 12. Megatron-LM 集成

★★★★★ 两条集成路径:

### Path A: DeepEP Backend (V1 Buffer API)

```python
# fused_a2a.py: FusedDispatch/FusedCombine
# --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep
# _DeepepManager: token_dispatcher.py
# 限制: moe_router_dtype=fp32 (DeepEP只支持float32 probs)
```

### Path B: HybridEP Backend (TMA + IBGDA)

```python
# fused_a2a.py: HybridEPDispatch/HybridEPCombine
# --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
# _HybridEPManager: token_dispatcher.py
# FP8/FP4 quantization alignment via pad_multiple
# expert rank capacity factor → token dropping + overflow detection
# Target: GB200 NVL72, B200, H100
```

### Megatron MoE版本路线

```
MCore v0.12 (2025/05): Added DeepEP support (--moe-enable-deepep)
MCore v0.15 (2025/11): Added HybridEP backend to Flex Dispatcher
MCore dev (2026/01): DeepSeek-V3.2 + pipeline-aware activation offloading

3种dispatcher: alltoall(NCCL) → DeepEP(V1) → HybridEP(TMA+IBGDA)
```

## 13. RTX 4090 限制

★★★★★ RTX 4090完全不适用于DeepEP!

| 限制 | 影响 |
|------|------|
| **SM89 (非SM90)** | V2 kernel需要SM90 → RTX 4090不可用V2 |
| **无NVLink** | Hybrid mode不可用 → scaleup域不存在 |
| **PCIe only** | 无RDMA → 只能用Direct mode(但NVLink-only也不行) |
| **V1 legacy** | V1需要NVSHMEM → RTX 4090不满足 |
| **TMA指令** | HybridEP需TMA → RTX 4090不支持 |

★★★★ RTX 4090 MoE替代方案:
- NCCL all-to-all → 通用但慢 (4.6x差距)
- 单GPU MoE → EP=1 → 无跨GPU通信 → DeepEP无用
- TP=1 → 专家本地计算 → 无分布式 → 性能受限

★★★★★ RTX 5090 (SM120) 前瞻:
- SM120满足DeepEP SM90要求 ✓
- 但NVLink/RDMA仍缺失 → PCIe only → Hybrid mode不可用
- 未来: PCIe kernel (HybridEP分支) → 可能支持非NVLink → 但仍需RDMA

## 14. CUDA PTX 指令使用 (源码级)

★★★★★ DeepEP V2 使用大量SM90+ PTX指令:

| 指令 | 用途 |
|------|------|
| **elect.sync** | 随机选举一个lane → warp-level任务分配 |
| **mbarrier.init/arrive/wait** | TMA同步 → 异步数据传输协调 |
| **cp.async.bulk** (TMA load/store) | 大块数据异步传输 → HBM↔SMEM |
| **red.release.sys.global.add** | 跨GPU atomic add → barrier信号 |
| **ld.acquire.sys/st.release.sys** | 系统级内存一致性 → NVLink/RDMA同步 |
| **fence.proxy.async.shared::cta** | TMA store fence → 防止proxy内存问题 (fix #642) |
| **setmaxnreg.inc/dec.sync** | warp group寄存器分配 → 动态调整 |
| **bar.sync** | named barrier → warp组间同步 |

★★★★ TMA (Tensor Memory Accelerator):
- cp.async.bulk.shared::cluster.global → GPU→SMEM大块异步load
- cp.async.bulk.global.shared::cta → SMEM→GPU大块异步store
- L2 cache hint: kEvictFirst(load) / kEvictNormal(store)
- 32-byte alignment (kNumTMAAlignBytes)

## 15. WorkspaceLayout — 固定布局

★★★★ WorkspaceLayout = 所有设置共享固定布局位置 → buffer可复用:

```
最大配置: 1024 ranks / 2048 experts / 256 experts_per_rank / 1280 channels / 32 AGRS sessions
信号类型: NVLink barrier(16B) + notify reduction + scaleup/scaleout rank+expert counts
          + channel metadata + PP counts + AGRS signals
```

★★★★ BufferLayout = 每个rank一个slot → 固定大小:
```
[rank0 tokens | rank1 tokens | ... | rankN tokens]
每token: hidden_bytes + sf_bytes + metadata_bytes + mbarrier(可选)
```

## 16. Engram / PP / CP / AGRS — 实验性通信原语

### 16.1 Engram (RDMA Remote Memory Fetch)

```
★★★★ 0 SM RDMA远程内存访问:
engram_write: KV cache → CPU NUMA-local segment → barrier for visibility
engram_fetch: RDMA get from remote CPU → ncclGin.get() → returns callable hook
→ PD分离场景: 0 SM occupation via RDMA!
→ Hook模式: fetch返回callable → 调用时等待数据到达 → 异步fetch!
```

### 16.2 PP (Pipeline Parallel Send/Recv)

```
★★★★ Ring-based PP:
pp_send/pp_recv: prev/next rank only → ncclGin put/get + symmetric buffers + inflight slots
→ 实验性, 0 SM with RDMA
→ 最新PR #665: 支持multi-QP for PP
```

### 16.3 CP (Context Parallelism — Copy Engine)

```
★★★★ 0 SM Copy Engine:
→ 实验性 → 详细实现待公开
```

### 16.4 AGRS (All-Gather Reduce-Scatter)

```
★★★★ Session-based:
create_agrs_session → all_gather → destroy_agrs_session
cudaMemcpyBatchAsync for NVLink + CUDA driver batched_write_and_wait
→ 实验性, NVLink-only
→ agrs_get_inplace_tensor: 无拷贝获取tensor → 直接在buffer上操作
```

## 17. 与 DeepSeek-V3 的关系

★★★★★ DeepEP是DeepSeek-V3/R1训练和推理的生产通信库:
- V3: 256 routed experts, 37B active params, top-8 → EP通信是核心瓶颈
- DualPipe (计算-通信overlap) → orchestrate DeepEP的dispatch/combine
- V3论文 (arxiv:2412.19437) → DeepEP提供底层kernel
- FP8 dispatch → 与V3 FP8 tile-wise量化一致

## 18. 网络配置最佳实践

| 配置 | 建议 | 原因 |
|------|------|------|
| InfiniBand | 推荐 | 原生 RDMA 支持 |
| RoCE | 理论兼容 | 未充分测试 |
| Adaptive Routing | 开启 | 多路径均衡, 虽有额外延迟 |
| Congestion Control | 关闭 | 损害最大带宽 |
| Traffic Isolation | 按虚拟通道分离 | EP vs 其他流量隔离 |
| PCI Atomic Mode | 设为 4 | 提升 RDMA atomic 性能 |
| NIC Name | EP_NIC_NAME=mlx5_0 | 默认NIC |

## 19. 环境变量 (重要)

```
EP_BUFFER_DEBUG=1: 打印初始化/SM/backend调试信息
EP_DISABLE_GIN=1: 禁用Gin backend → 回退non-Gin路径
EP_JIT_CACHE_DIR: JIT缓存目录 (默认$HOME/.deep_ep)
EP_JIT_DUMP_ASM=1: dump PTX+SASS → 性能分析
EP_OVERRIDE_RDMA_SL: 覆盖RDMA服务级别 → 流量隔离
EP_NUM_TOPK_IDX_BITS: 覆盖top-k索引编码位数
EP_AVOID_RECORD_STREAM=1: 避免record_stream → 减少内存碎片
EP_NCCL_ROOT_DIR: NCCL安装路径 → 自动检测
EP_NVSHMEM_ROOT_DIR: NVSHMEM安装路径 → 自动检测
```

## 20. 核心学习总结

1. ★★★★★ **EP通信是MoE核心瓶颈**: All-to-All延迟决定EP效率上限
2. ★★★★★ **Asymmetric vs Symmetric**: NCCL padding浪费 → DeepEP原生asymmetric → 4.6x优势 → 架构级优势, NCCL无法通过优化消除!
3. ★★★★ **Gin Backend**: 绕过proxy thread → GPU直接RDMA → header-only → 复用NCCL comm → 3 teams(World/LSA/Rail)
4. ★★★★ **SM资源**: V2从24降至4-6 → analytical计算 → 1.25x safety margin
5. ★★★★★ **Handle Caching**: decode推理跳过CPU sync → 最大延迟优化 → gating决策不变时复用
6. ★★★★ **FP8 Dispatch**: 跨节点流量减半 + TMA-aligned SF → 与V3 FP8量化一致
7. ★★★★ **Hybrid Mode**: NVLink转发+RDMA scaleout → 物理topology匹配 → channel级跟踪
8. ★★★★ **JIT编译**: 无预编译 → pip install即用 → 缓存 → PTX/SASS dump调试
9. ★★★★ **Symmetric Memory**: 3策略(GPU/Elastic/HybridElastic) → POSIX FD跨进程共享
10. ★★★ **实验性分支**: Zero-copy(OPEN), Eager(merged), HybridEP(活跃), AntGroup(3 PR: 1 OPEN + 2 merged), Mori(AMD)
11. ★★★★ **Megatron集成**: FlexTokenDispatcher → 一行配置切换 → DeepEP/HybridEP → v0.12/v0.15
12. ★★★★★ **RTX 4090**: 完全不适用 → SM89非SM90 + 无NVLink + 无RDMA → NCCL all-to-all是唯一选择

## 21. 与7框架的关系

| 框架 | MoE EP通信 | DeepEP关系 |
|------|-----------|------------|
| **Megatron-LM** | 3种dispatcher (AllGather/AlltoAll/Flex) | DeepEP是Flex dispatcher的EP backend |
| **DeepSpeed** | 无MoE层 (AutoEP实验性) | 无直接关系 (但AutoEP可能集成DeepEP) |
| **vLLM** | expert_parallel (NCCL all-to-all) | DeepEP可替代vLLM的EP通信(需SM90) |
| **verl** | via Megatron → NCCL | 可替换Megatron EP backend |
| **rLLM** | via verl → Megatron → NCCL | 同verl |
| **MindIE** | HCCL (昇腾) | 不同硬件 → DeepEP-Ascend (HCCL版) |
| **PyTorch** | c10d_functional.all_to_all | DeepEP底层用NCCL Gin → PyTorch之上 |

## 参考

- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP) — 主仓库, V2公开发布2026-04-26
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 本地源码: rollout-infra/deepep-src/
- 源码关键文件: elastic.py, handle.cuh, comm.cuh, ptx.cuh, layout.cuh, nccl.cu, symmetric.hpp, hybrid_dispatch.cuh
- 相关笔记: deepep-source-reading.md, deepep-megatron-integration-latest.md, deepep-ascend-reading.md
