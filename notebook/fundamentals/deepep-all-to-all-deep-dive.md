# DeepEP V2 All-to-All Deep Dive

> 2026-06-08 | DeepSeek高性能MoE Expert Parallelism通信库源码深读
> 基于: DeepEP V2源码(deepseek-ai/DeepEP), NCCL Gin backend, SM90/Hopper/Blackwell
> 关联: moe-serving-expert-parallelism.md (MoE总览), cuda-kernel-optimization-sm89.md (SM89优化)

## 0. DeepEP是什么?

DeepEP (DeepEveryParallel) 是DeepSeek开源的高性能EP通信库, 专注于MoE的All-to-All dispatch/combine。

**V2核心升级**:
- **NVSHMEM → NCCL Gin**: 更轻量, 可复用NCCL communicator
- **统一ElasticBuffer**: 高吞吐+低延迟合并为单一接口
- **GEMM Layout**: 新的token排列支持直接GEMM
- **SM 4-6**: V1需要24 SMs → V2仅4-6 SMs达同等性能(4x SM节省!)
- **EP2048**: 支持更大scale-up/scale-out域
- **JIT编译**: runtime编译, 安装无需CUDA编译
- **FP8 Dispatch**: BF16 combine + FP8 dispatch → 带宽减半

**性能实测(V3配置: 8K tokens, 7168 hidden, top-8 experts)**:

| Arch | NIC | Topo | Dispatch BW | Combine BW | #SMs |
|------|-----|------|-------------|------------|------|
| SM90 | CX7 | EP 8x2 | 90 GB/s(RDMA) | 81 GB/s(RDMA) | 12 |
| SM100 | CX7 | EP 8x2 | 90 GB/s(RDMA) | 91 GB/s(RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s(NVLink) | 740 GB/s(NVLink) | 64(Max) |
| SM100 | N/A | EP 8 | 643 GB/s(NVLink) | 675 GB/s(NVLink) | 24(Min) |

**V2 vs V1**: 1.3x peak performance + 4x SM节省!

## 1. 源码架构: 三层设计 + JIT

### 1.1 Python → C++ → CUDA调用栈

```
Python API (deep_ep/__init__.py)
  → ElasticBuffer.dispatch() (buffers/elastic.py)
    → self.runtime.dispatch() → pybind11绑定 → C++ _C模块
      → JIT编译: 参数hash → Jinja模板渲染 → .cu → ninja → .so
        → CUDA kernel: dispatch_impl() / hybrid_dispatch_impl() / combine_impl()
```

### 1.2 核心模块划分

| 模块 | 文件 | 功能 |
|------|------|------|
| **Python API** | `deep_ep/__init__.py` | init_jit + 导入 |
| **ElasticBuffer** | `deep_ep/buffers/elastic.py` | V2统一接口 |
| **EPHandle** | `deep_ep/buffers/elastic.py` | 通信handle(路由元数据) |
| **EventOverlap** | `deep_ep/utils/event.py` | compute-comm重叠控制 |
| **C++ Binding** | `csrc/python_api.cpp` | pybind11注册 |
| **JIT Backend** | `csrc/kernels/backend/api.cuh` | NCCL Gin backend |
| **Dispatch Kernel** | `deep_ep/impls/dispatch.cuh` | 单节点dispatch |
| **Hybrid Dispatch** | `deep_ep/impls/hybrid_dispatch.cuh` | 多节点层次dispatch |
| **Combine Kernel** | `deep_ep/impls/combine.cuh` | combine/reduce |
| **Copy Epilogue** | `deep_ep/impls/dispatch_copy_epilogue.cuh` | 数据搬入输出 |
| **Barrier** | `deep_ep/impls/barrier.cuh` | GPU级barrier |
| **Engram** | `deep_ep/impls/engram_fetch.cuh` | RDMA远程KV cache fetch |
| **PP** | `deep_ep/impls/pp_send_recv.cuh` | Pipeline并行send/recv |

### 1.3 JIT编译机制

```
调用ElasticBuffer.dispatch() →
  1. 计算kernel参数: num_experts, num_topk, hidden, num_max_tokens_per_rank, num_sms
  2. hash → 特化C++模板参数 → Jinja渲染dispatch_impl<...>()的.cu代码
  3. ninja编译 .cu → .so
  4. TVM-FFI加载 → 缓存到 ~/.deep_ep/
后续调用 → 直接使用缓存的.so → 无重编译!

关键: 所有kernel参数都是编译时constexpr → 零运行时分支!
- kNumSMs, kNumRanks, kNumHiddenBytes, kNumExperts, kNumTopk
- kNumQPs, kNumNotifyWarps, kNumDispatchWarps
- kDoCPUSync, kReuseSlotIndices, kExpertAlignment
```

## 2. Dispatch Kernel: Token路由核心

### 2.1 dispatch_impl — 单节点(NVLink)dispatch

```cuda
// 源码: deep_ep/impls/dispatch.cuh
template <bool kIsScaleupNVLink,
          bool kDoCPUSync, bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,  // warp分工!
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles>
__global__ void dispatch_impl(x, sf, topk_idx, topk_weights, ...) {
    // 1. GPU Barrier: 确保所有rank的buffer可用
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks, ...>(gin, workspace, ...);

    // 2. Warp角色分工:
    //    warp_idx < kNumNotifyWarps → Notify warps (统计token分布)
    //    warp_idx >= kNumNotifyWarps → Dispatch warps (发送数据)

    if (warp_idx < kNumNotifyWarps) {
        // Notify warp: 统计每个expert收到多少token
        // atomicAdd_block(expert_count + dst_expert_idx, 1)
        // → 前缀求和 → psum_num_recv_tokens_per_expert
        // → 写入workspace让所有rank可见
    } else {
        // Dispatch warp: 发送token到对应rank的buffer
        // NCCL Gin → NVLink直写远程buffer
        // gin.put<team_t>(remote_buffer + slot_offset, local_data, num_bytes, dst_rank)
    }

    // 3. Barrier: 确保所有数据到达
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks, ...>(gin, workspace, ...);
}
```

**关键设计**: **warp分工** — notify warps统计分布, dispatch warps发送数据!

### 2.2 Notify Warp: 统计token分布

```cuda
// Notify warp核心逻辑:
// 1. 清零expert/rank计数 (shared memory)
for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
    rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

// 2. 遍历所有token, atomicAdd统计每个expert选择
for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
    const auto dst_expert_idx = lane_idx < kNumTopk ?
        __ldg(topk_idx + i * kNumTopk + lane_idx) : -1;
    if (dst_expert_idx >= 0)
        atomicAdd_block(expert_count + dst_expert_idx, 1);
}

// 3. 前缀求和 → 建立slot映射
// → 写入workspace让dispatch warps读取
```

**洞察**: notify warps用shared memory做atomic统计 → 1次barrier → dispatch warps读取统计结果 → 知道每个token该发到哪个rank的哪个slot!

### 2.3 Dispatch Warp: 数据传输

```cuda
// Dispatch warp: 通过NCCL Gin发送token
// NVLink: gin.put() → 直写远程GPU buffer
// RDMA: gin.put() → RDMA写远程GPU buffer

// 关键: 每个warp是一个"channel"
// channel → QP分配 → get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>()
// SM少 → 每SM独占QP → CTA sharing
// SM多 → 多SM共享QP → GPU sharing
```

## 3. Hybrid Dispatch: 多节点层次通信

### 3.1 三层Warp分工

```cuda
// 源码: deep_ep/impls/hybrid_dispatch.cuh
template <bool kDoCPUSync, bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps,     // 统计分布
          int kNumScaleoutWarps,   // RDMA发送(scale-out)
          int kNumForwardWarps,    // NVLink转发(scale-up)
          ...>
__global__ void hybrid_dispatch_impl(...) {
    // Warp角色分工:
    // warp_idx < kNumNotifyWarps → Notify (统计)
    // kNumNotifyWarps ≤ < kNumNotifyWarps+kNumScaleoutWarps → Scale-out (RDMA)
    // kNumNotifyWarps+kNumScaleoutWarps ≤ → Forward (NVLink)

    if (warp_idx < kNumNotifyWarps) {
        // 统计token分布 → workspace
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        // Scale-out warp: RDMA发送到其他节点
        // 每个warp是一个RDMA "channel"
        // gin.put() → RDMA写远程GPU buffer
    } else {
        // Forward warp: NVLink转发(本节点内)
        // 收集本节点所有rank的数据 → NVLink直写
        // gin.put<ncclTeamTagLsa>() → NVLink写
    }
}
```

**Hybrid模式**: RDMA + NVLink层次通信
1. **Scale-out**: RDMA发送token到其他节点
2. **Forward**: NVLink转发到同节点其他GPU
3. **优势**: 多plane/多rail网络更友好 → 层次化减少RDMA连接数

### 3.2 Channel + Linked List设计

```cuda
// 每个warp = 1个RDMA channel
// channel_linked_list: per-channel per-scaleup-peer linked list
// → 多channel并行 → RDMA带宽充分利用

// kNumChannels = kNumScaleoutWarps * kNumSMs
// kNumMaxTokensPerChannel = ceil_div(kNumMaxTokensPerRank, kNumChannels)
// → 每个channel处理少量token → 低延迟!
```

## 4. Combine Kernel: Token聚合

### 4.1 combine_impl — 反向路由

```cuda
// 源码: deep_ep/impls/combine.cuh
template <bool kIsScaleupNVLink,
          bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          ...>
__global__ void combine_impl(x, topk_weights, src_metadata, ...) {
    // 1. Barrier: 确保所有expert计算完成
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks, ...>(gin, workspace, ...);

    // 2. TMA写远程buffer: expert结果 → 原始rank
    for (int i = token_start_idx; i < token_end_idx; ++ i) {
        // 读取src_metadata → 知道token原始来自哪个rank
        const int src_rank_idx = src_rank_topk_idx / kNumTopk;

        // NVLink bypass: 直写远程rank的recv buffer
        if (nvlink_bypass) {
            // gin.get_sym_ptr<team_t>() → NVLink直写
        } else {
            // RDMA写远程 → gin.put()
        }
    }

    // 3. Barrier: 确保所有数据到达
    // 4. Reduce epilogue: 多top-k权重加权求和
}
```

**关键**: combine使用TMA(Tensor Memory Accelerator)写远程buffer → SM90/Hopper专用!

### 4.2 Reduce Epilogue: top-k加权求和

```cuda
// 源码: deep_ep/impls/combine_reduce_epilogue.cuh
// 读取多个top-k expert的输出 → topk_weights加权求和 → 最终输出

// kAllowMultipleReduction=true → 多次reduce(精度略低但更快)
// kAllowMultipleReduction=false → 单次reduce(最佳精度)
```

### 4.3 Dispatch ↔ Combine对称性

```
Dispatch forward = Combine backward (对偶!)
Combine forward = Dispatch backward (对偶!)

Dispatch: token → 按expert路由 → 发到对应rank
Combine: expert结果 → 按原始rank路由 → reduce回原rank
```

## 5. NCCL Gin Backend: 通信核心

### 5.1 Gin是什么?

NCCL Gin = NCCL的GPU-initiated通信机制:
- **Header-only**: 轻量级, 嵌入kernel代码
- **复用NCCL communicator**: 不需要额外初始化
- **NVLink + RDMA**: 统一接口
- **对称内存(Symmetric Memory)**: 所有rank共享同一虚拟地址 → 直接远程写!

```cuda
// 源码: deep_ep/common/handle.cuh (推断)
struct NCCLGin {
    ncclDevComm_t nccl_dev_comm;
    ncclWindow_t nccl_window;
    int qp_idx;
    ncclGinResourceSharingMode sharing_mode;

    // NVLink写: gin.put<ncclTeamTagLsa>(dst_ptr, src_ptr, num_bytes, peer)
    // RDMA写: gin.put<ncclTeamTagWorld>(dst_ptr, src_ptr, num_bytes, peer)
    // NVLink读: gin.get_sym_ptr<ncclTeamTagLsa>(ptr, peer) → 返回远程对称地址

    // 关键: get_sym_ptr → 直接获得远程GPU的对应buffer地址!
    // 不需要显式put/get → 可以直接TMA写远程!
};
```

### 5.2 QP分配策略

```cuda
// 源码: deep_ep/common/comm.cuh:56-86
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM>
__device__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx) {
    if (kNumQPs == 1) return {0, kSharingGrid};  // 1QP → 全grid共享

    if (kNumSMs <= kNumAvailableQPs) {
        // SM少 → 每SM独占QP → CTA sharing (更低延迟)
        // SM0: QP0,3,6,9  SM1: QP1,4,7  SM2: QP2,5,8
        return {qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA};
    } else {
        // SM多 → 所有SM共享QP → GPU sharing
        // global_channel_idx % kNumAvailableQPs → round-robin
        return {qp_idx, NCCL_GIN_RESOURCE_SHARING_GPU};
    }
}
```

### 5.3 GPU Barrier机制

```cuda
// 源码: deep_ep/common/comm.cuh
// NVLink barrier: atomic red → 所有rank signal → 等所有signal到达
// RDMA barrier: RDMA atomic → 跨节点signal

// Phase翻转设计: 避免counter重置
const int phase = status & 1, sign = status >> 1;
ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);  // atomic +1/-1

// 超时保护: clock64() - start_clock >= kNumTimeoutCycles → trap()
// kNumTimeoutCycles ≈ 100秒 × 2GHz = 200,000,000,000 cycles
```

## 6. SM优化: 为什么V2只需4-6 SMs?

### 6.1 SM理论计算

```python
# 源码: deep_ep/buffers/elastic.py:582-687
# get_theoretical_num_sms() — 分析式SM计算, 无需auto-tuning!

# 关键: 带宽建模 → SM是HBM读写的瓶颈而非NVLink/RDMA瓶颈
# HBM读带宽: 200 GB/s/SM
# HBM写带宽: 50 GB/s/SM
# NVLink带宽: ~726 GB/s (SM100)
# RDMA带宽: ~90 GB/s (CX7)

# 如果NVLink/RDMA是瓶颈 → 需要足够SM使HBM读写跟上
# 如果HBM读写不是瓶颈 → 少SM即可!

# 计算公式:
num_sms = max(
    bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
    bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
)
num_sms = align(max(4, ceil(num_sms * 1.25)), 2)  // 1.25x余量

# V3配置(E=256, top-8, EP8 NVLink):
# nvlink_traffic ≈ 0.875 (7/8个rank需要NVLink发送)
# nvlink_gbs ≈ 726 GB/s
# sm_read ≈ 1/expected_topk ≈ 0.35
# → num_sms ≈ 726/0.875 * 0.35/200 ≈ 1.45 → align(4, 2) = 4!

# prefer_overlap_with_compute=True → 少SM(留更多SM给计算)
# prefer_overlap_with_compute=False → max(64, num_sms)(全力通信)
```

### 6.2 SM节省的根因

**V1问题**: 24 SMs → 很多SM空闲等待 → 浪费!
**V2解决**: 分析式计算 → 只分配需要的SM → 剩余SM给计算!

```
V1: 24 SMs用于通信 → 计算只有104-24=80 SMs → 计算受限
V2: 4-6 SMs用于通信 → 计算有104-6=98 SMs → 计算不受限!

通信-计算重叠:
  4 SMs做dispatch → 98 SMs做forward GEMM → 完美重叠!
  4 SMs做combine → 98 SMs做backward GEMM → 完美重叠!
```

## 7. FP8 Dispatch: 带宽减半

### 7.1 FP8量化流程

```python
# 源码: deep_ep/buffers/elastic.py: dispatch() API
# FP8模式: x = (fp8_data, scale_factors)
# x[0]: torch.float8_e4m3fn, [num_tokens, hidden]
# x[1]: scale factors (per-block or per-tensor)

# Dispatch:
#   BF16 → FP8(E4M3) → 发FP8数据(带宽省50%) → 接收端FP8 → BF16
# Combine:
#   BF16直接发送(精度更重要!)

# FP8数据大小: 1 byte/element vs BF16 2 bytes → 带宽减半!
# 但scale factors额外开销: ceil(hidden/32) × 4 bytes/entry
# → hidden=7168: sf = 224×4=896 bytes vs data = 7168×1=7168 bytes → 12.5% overhead
# → 总节省: (7168+896)/(7168×2) = 56.25% → 约44%带宽省!
```

### 7.2 Scale Factor Packing

```cuda
// 源码: deep_ep/common/compiled.cuh
using sf_pack_t = uint32_t;  // 32-value block → 1 E8M0 scale

// kNumSFPacks = ceil(hidden / 32) = ceil(7168/32) = 224
// FP8 block scaling: 32 values per block → 1 E8M0 exponent scale
// → MX-style block quantization → 精度好(0.05%相对误差)
```

## 8. EventOverlap: 通信-计算重叠

### 8.1 核心设计

```python
# 源码: deep_ep/utils/event.py
class EventOverlap:
    event: EventHandle  # C++ CUDA event
    extra_tensors: Tuple[torch.Tensor]  # CUDA graph兼容的stream recording

    def current_stream_wait(self):
        """计算流等待通信流完成"""
        self.event.current_stream_wait()

    # with语法支持重叠:
    # with event_overlap:
    #     do_something_on_compute_stream()  # 与通信重叠!
    # # 退出时自动wait
```

### 8.2 重叠模式

```python
# DeepSeek DualPipe重叠模式:
recv_x, handle, event = buffer.dispatch(x, ..., async_with_compute_stream=True)
# ↑ 通信在comm_stream上运行

with event:  # 在此期间计算流可以做其他工作
    expert_compute_on_different_tokens()  # 与dispatch重叠!

event.current_stream_wait()  # 等dispatch完成 → 使用recv_x

combined_x, event2 = buffer.combine(expert_output, handle, async_with_compute_stream=True)
with event2:
    other_backward_compute()  # 与combine重叠!
```

**关键**: `async_with_compute_stream=True` → 通信在专用comm_stream → 计算流不阻塞!

## 9. Engram: 0-SM RDMA远程KV Cache Fetch

### 9.1 设计

```cuda
// 源码: deep_ep/impls/engram_fetch.cuh
// Engram = KV cache存储在CPU内存 → RDMA fetch到GPU

// CPU端: storage = [num_entries, hidden] (BF16)
// GPU端: fetched = [num_tokens, hidden] (接收buffer)

// 0-SM RDMA: 使用NCCL Gin的RDMA get → GPU直接从远程CPU内存读!
// → 不需要GPU SM参与搬运 → 0 SM占用!

// Gin RDMA get:
gin.get<team_t, ncclCoopThread, ncclGin_SegmentMixed>(
    remote_storage + src_byte_offset,
    local_fetched + token_idx * kNumHiddenBytes,
    kNumHiddenBytes, peer_idx);
```

### 9.2 Elastic GPU+CPU Buffer

```python
# 源码: deep_ep/buffers/elastic.py: engram_write/fetch
# CPU buffer: mmap + POSIX FD handle → RDMA可访问
# GPU buffer: 接收RDMA数据

# Elastic = 未来支持GPU+CPU混合物理内存
# → 连续虚拟地址 → 底层混合GPU/CPU → 自动透明!
```

## 10. RTX 4090(SM89)可用性分析

### 10.1 硬件限制

**DeepEP V2要求**:
- SM90(Hopper)或更高 → RTX 4090(SM89) ❌ **不支持V2!**
- CUDA 12.3+ → RTX 4090有CUDA 12.8 ✅
- NCCL 2.30.4+ → 需检查版本
- NVLink → RTX 4090无NVLink ❌ **单卡无法做NVLink EP!**
- RDMA → RTX 4090无RDMA网卡 ❌

**结论**: RTX 4090 **完全无法使用DeepEP**:
1. SM89不支持TMA/mbarrier/elect等SM90 PTX指令
2. 无NVLink → 无法做intra-node EP
3. 无RDMA → 无法做inter-node EP
4. PCIe EP → 带宽太低(~12 GB/s vs NVLink 726 GB/s)

### 10.2 RTX 4090上的MoE EP替代方案

```
RTX 4090 MoE推理 → 单GPU → 不需要EP!
- 7B MoE(如Mixtral): 单GPU推理 → FlashInfer decode + INT4 weight-only
- DeepSeek-V3 671B: 不可能在RTX 4090上运行(需要256+ GPU)

RTX 4090 MoE训练 → 多GPU PCIe → EP不可行
- PCIe带宽~12 GB/s vs NVLink 726 GB/s → 60x差距!
- EP All-to-All延迟: PCIe >10ms vs NVLink <1ms → 完全不可行
- 替代: TP(Tensor Parallelism) → 通信量小(仅AllReduce) → 可行!
```

## 11. 关键源码文件索引

| 文件 | 位置 | 内容 |
|------|------|------|
| `__init__.py` | `deep_ep/` | JIT初始化 + API导入 |
| `elastic.py` | `deep_ep/buffers/` | ElasticBuffer + EPHandle (V2核心) |
| `event.py` | `deep_ep/utils/` | EventOverlap (重叠控制) |
| `envs.py` | `deep_ep/utils/` | NVLink/RDMA检测 + 环境变量 |
| `gate.py` | `deep_ep/utils/` | Top-k门控辅助 |
| `python_api.cpp` | `csrc/` | pybind11注册 |
| `dispatch.cuh` | `deep_ep/impls/` | 单节点dispatch kernel |
| `hybrid_dispatch.cuh` | `deep_ep/impls/` | 多节点层次dispatch |
| `combine.cuh` | `deep_ep/impls/` | combine kernel |
| `dispatch_copy_epilogue.cuh` | `deep_ep/impls/` | 数据搬入输出tensor |
| `combine_reduce_epilogue.cuh` | `deep_ep/impls/` | top-k加权reduce |
| `barrier.cuh` | `deep_ep/impls/` | GPU barrier |
| `engram_fetch.cuh` | `deep_ep/impls/` | RDMA KV fetch |
| `pp_send_recv.cuh` | `deep_ep/impls/` | PP send/recv |
| `comm.cuh` | `deep_ep/common/` | Gin通信 + barrier + QP分配 |
| `ptx.cuh` | `deep_ep/common/` | PTX指令(TMA/mbarrier/elect) |
| `layout.cuh` | `deep_ep/common/` | Buffer/Workspace布局 |
| `handle.cuh` | `deep_ep/common/` | NCCLGin handle |
| `api.cuh` | `csrc/kernels/backend/` | NCCL Gin backend API |
| `intranode.cu` | `csrc/kernels/legacy/` | V1 NVLink kernel |
| `internode.cu` | `csrc/kernels/legacy/` | V1 RDMA kernel |
| `internode_ll.cu` | `csrc/kernels/legacy/` | V1低延迟RDMA kernel |

---

**Sources**:
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [DeepEP README (V2)](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)
- [NCCL Gin Backend](https://github.com/nvidia/nccl)
- [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437)

**Related notes**: moe-serving-expert-parallelism.md (MoE总览), cuda-kernel-optimization-sm89.md (SM89 vs SM90)