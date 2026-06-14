# DeepEP 源码阅读笔记

> DeepEP = DeepSeek-AI 开发的高性能 MoE Expert Parallel 通信库
> 版本: V2 (2.0.0) — 从NVSHMEM重构到NCCL Gin backend
> 作者: Chenggang Zhao, Shangyan Zhou 等 | MIT License
> 源码路径: rollout-infra/deepep-src/

## 核心定位

DeepEP是 **专为MoE Expert Parallel设计的通信库** — 不是通用集合通信(NCCL), 而是针对MoE dispatch/combine的asymmetric通信优化。

**关键区别**: NCCL all-to-all假设对称通信(每个rank发等量数据), MoE dispatch是 **asymmetric**(不同expert收到不同数量token) → DeepEP直接解决这个瓶颈!

---

## 架构概览

```
deepep-src/
├── deep_ep/                    # Python package
│   ├── buffers/
│   │   ├── elastic.py          # ElasticBuffer + EPHandle (V2核心API)
│   │   └── legacy.py           # V1 Buffer API
│   └── include/deep_ep/
│       ├── common/             # 共享头文件(layout/math/PTX/handle/comm)
│       └── impls/              # JIT编译kernel实现(.cuh)
│           ├── dispatch.cuh / combine.cuh
│           ├── hybrid_dispatch.cuh / hybrid_combine.cuh
│           ├── engram_fetch.cuh / engram_fetch_wait.cuh
│           └── pp_send_recv.cuh / barrier.cuh
├── csrc/                       # C++/CUDA源码
│   ├── python_api.cpp          # pybind11 (V1+V2+JIT)
│   ├── elastic/buffer.hpp      # ElasticBuffer C++实现
│   └── kernels/
│       ├── backend/            # 通信backend抽象
│       │   ├── api.cuh         # NCCL + NVSHMEM + CUDA driver
│       │   ├── nccl.cu         # NCCL Gin context初始化
│       │   └── cuda_driver.cu  # Batched write/wait
│       ├── elastic/            # V2 kernel launcher
│       └── jit/                # Just-In-Time编译系统
│           ├── api.hpp / compiler.hpp / cache.hpp / launch_runtime.hpp
└── tests/
    ├── test_ep.py              # Dispatch/combine correctness + benchmark
    ├── test_agrs.py            # All-gather/reduce-scatter
    ├── test_engram.py          # RDMA fetch
    └── test_pp.py              # Pipeline parallel
```

### 6大核心组件

| 组件 | 作用 | 关键特性 |
|------|------|----------|
| **ElasticBuffer** | 统一buffer管理(dispatch/combine/engram/PP/AGRS) | GPU+CPU对称内存+NCCL Gin context+专用通信stream |
| **EPHandle** | 路由元数据(top-k indices+prefix-sum+slot indices) | 可缓存!decode推理复用→消除CPU sync |
| **NCCL Gin Backend** | RDMA通信引擎 | ncclGin(RDMA put/get/signal)+ncclMemAlloc+3种team |
| **JIT Compiler** | 运行时编译 | NVCC→CUBIN→hash缓存→无需预编译 |
| **Symmetric Memory** | GPU/CPU对称分配 | 3策略:GPU(ncclMemAlloc)/Elastic(CUDA VMM)/Hybrid(POSIX FD) |
| **WorkspaceLayout** | 固定layout | 1024 ranks/2048 experts/1280 channels max |

---

## MoE Dispatch & Combine (核心)

### Dispatch: token路由到expert GPU

```
Phase 1: dispatch_impl
  - 读取input tokens → 写入symmetric send buffers
  - NCCL Gin put/signal → 通知远程rank

Phase 2: dispatch_copy_epilogue_impl
  - 从symmetric recv buffers → 复制到output tensors

支持: BF16 + FP8(per-token scale factor)
```

**两种模式**:
- **Direct** (单节点, scaleout_ranks==1): NVLink-only → 一组send/recv buffers
- **Hybrid** (多节点, scaleout_ranks>1): 分层 — NVLink域内转发 + RDMA域跨节点 → channels(8/SM) → linked list tracking

### Combine: expert输出回到原token位置

```
Phase 1: combine_impl
  - 读取expert outputs → 推入symmetric recv buffers → Gin put/signal

Phase 2: combine_reduce_epilogue_impl
  - Reduce接收数据(weighted sum用top-k weights) → 最终output
  - 支持bias addition(0/1/2个bias tensor)
  - allow_multiple_reduction: scaleup域内先reduce → 减少跨节点流量
  - use_expanded_layout: one-slot-per-expert-per-token → GEMM-friendly
```

### Handle Caching (decode推理关键优化)

```
EPHandle from previous dispatch → 缓存 → reuse:
  - 跳过notify warps → 消除CPU sync
  - 跳过layout重计算 → prefix-sum + slot indices reuse
  - decode阶段路由模式稳定 → 大幅减少延迟
```

---

## NCCL Gin Backend vs 传统NCCL

| 维度 | NCCL All-to-All | DeepEP |
|------|-----------------|--------|
| **设计** | 通用对称集合通信 | MoE专用asymmetric dispatch/combine |
| **数据模式** | 对称: 每个rank发等量chunk | 不对称: 不同expert不同token数 |
| **SM占用** | NCCL proxy threads+多SM | V2: 4-6 SM (V1: 24 SM) |
| **计算overlap** | 有限(proxy model) | built-in async_with_compute_stream + dual-stream |
| **FP8** | 无特殊支持 | FP8 dispatch + per-token SF + TMA-aligned col-major |
| **拓扑感知** | 通用topo detect | 分离scaleup(NVLink)/scaleout(RDMA)域 → hybrid forwarding |
| **Backend** | NCCL内部transport | NCCL Gin(lightweight, header-only, reuse现有NCCL comm) |

**为什么DeepEP比NCCL all-to-all快**:
1. MoE dispatch/combine天然不对称 → NCCL symmetric浪费带宽(padding)
2. Hybrid mode(NVLink forwarding + RDMA scaleout) → 物理topology匹配
3. Gin backend绕过NCCL proxy thread → GPU kernel直接RDMA
4. FP8 dispatch → 跨节点流量减半(1 byte vs 2 bytes)
5. Handle caching → decode推理消除CPU sync
6. V2 analytical SM/QP → 无auto-tuning开销

---

## MoE专项优化

### 1. Expert Alignment (`expert_alignment`)

```
对齐每个local expert收到的token数到指定值(如128)
→ GEMM-friendly input shape → grouped GEMM高效
→ prefix-sum psum_num_recv_tokens_per_expert 考虑alignment padding
```

### 2. Expand Mode (`do_expand=True`)

```
one-slot-per-expert-per-token layout:
  num_recv_tokens行 → num_expanded_tokens行
  每个token每个top-k expert有dedicated slot
  → 直接feed grouped GEMM → expert batching完美
  → DeepSeek-V3 style MoE: combine reduction alongside SwiGLU
```

### 3. FP8 Dispatch + TMA-aligned SF

```
Tokens cast to FP8(float8_e4m3fn) + per-token scale factor → RDMA传输减半
Scale factors: TMA-aligned column-major → 直接作为GEMM input加载
```

### 4. Hybrid Domain Forwarding

```
RDMA收到的tokens → NVLink域内转发到正确local expert rank
token_metadata_at_forward + channel_linked_list → 跟踪转发链
```

### 5. Deterministic Mode (训练backward)

```
deterministic prologue kernel → 预计算destination slot indices
→ 多次dispatch一致 → backward正确性
```

---

## 性能 Benchmark

### V2 (DeepSeek-V3 style: 8K tokens, 7168 hidden, top-8, FP8 dispatch)

| Architecture | Topology | Dispatch BW | Combine BW | SMs |
|-------------|----------|-------------|------------|-----|
| SM90 (H800/H100) | EP 8x2 (2 nodes) | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | EP 8x4 (4 nodes) | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 (next-gen) | EP 8 (1 node) | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 |
| SM100 | EP 8 (1 node, min SM) | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 |

**V2 vs V1**: V2 peak 1.3x V1, 但SM省4x (V1=24SM → V2=4-6SM)

### V1 Low-latency (decode推理, H800)

| EP size | Dispatch latency | RDMA BW | Combine latency | RDMA BW |
|---------|------------------|---------|-----------------|---------|
| 8 | 77 us | 98 GB/s | 114 us | 127 GB/s |
| 32 | 155 us | 48 GB/s | 273 us | 53 GB/s |
| 128 | 192 us | 39 GB/s | 369 us | 39 GB/s |
| 256 | 194 us | 39 GB/s | 360 us | 40 GB/s |

### V1 Normal (训练, H800)

| EP size | Intranode Dispatch | Internode Dispatch | Internode Combine |
|---------|--------------------|--------------------|-------------------|
| 8 | 153 GB/s (NVLink) | — | 158 GB/s (NVLink) |
| 16 | — | 43 GB/s (RDMA) | 43 GB/s (RDMA) |
| 64 | — | 51 GB/s (RDMA) | 50 GB/s (RDMA) |

---

## 其他通信原语

### Barrier

- **NVLink barrier**: atomic red_add_rel_sys on symmetric signals + phase-based(even/odd)
- **Gin barrier**: ncclGin.signal(SignalInc) + shadow counter + direct GDAKI read
- **Hybrid barrier**: SM0→scaleup + SM1+→scaleout → grid sync

### Engram (RDMA Remote Memory Fetch)

```
engram_write: KV cache → CPU NUMA-local segment → barrier for visibility
engram_fetch: RDMA get from remote CPU → ncclGin.get() → returns callable hook
→ PD分离场景: 0 SM occupation via RDMA!
```

### Pipeline Parallel Send/Recv

```
pp_send/pp_recv: ring-based, prev/next rank only
NCCL Gin put/get + symmetric buffers + inflight slots
→ 实验性, 0 SM with RDMA
```

### All-Gather Reduce-Scatter (AGRS)

```
Session-based: create_agrs_session → all_gather → destroy_agrs_session
cudaMemcpyBatchAsync for NVLink + CUDA driver batched_write_and_wait
→ 实验性, NVLink-only
```

---

## 与 DeepSeek-V3 的关系

DeepEP是DeepSeek-V3/R1训练和推理的 **生产通信库**:
- V3: 256 routed experts, 37B active params, top-8 → EP通信是核心瓶颈
- DualPipe (计算-通信overlap) → orchestrate DeepEP的dispatch/combine
- V3论文(arxiv:2412.19437)描述MoE架构 → DeepEP提供底层kernel

---

## RTX 4090 限制

| 限制 | 影响 |
|------|------|
| **SM89 (非SM90)** | V2 kernel需要SM90 → RTX 4090 **不可用V2** |
| **无NVLink** | Hybrid mode不可用 → scaleup域不存在 |
| **PCIe only** | 无RDMA → 只能用Direct mode(NVLink-only也不行) |
| **V1 legacy** | V1需要NVSHMEM → RTX 4090不满足 |

**结论**: RTX 4090 **完全不适合DeepEP** → 需H100/H800(SM90+NVLink+RDMA)

### RTX 4090 MoE替代方案

- Megatron AlltoAllTokenDispatcher → NCCL all-to-all → 通用但慢
- vLLM expert_parallel → NCCL → 无asymmetric优化
- 单GPU MoE → EP=1 → 无跨GPU通信 → 本地expert计算 → DeepEP无用

---

## 实验性分支 (活跃开发)

| 分支 | 优化 | 来源 |
|------|------|------|
| **zero-copy** (PR #453) | 消除PyTorch→通信buffer拷贝 → 减SM | Tencent |
| **eager** (PR #437) | 低延迟协议 → 减RDMA RTT | 社区 |
| **hybrid-EP** (branch) | TMA指令+PCIe kernel+NVFP4+细粒度overlap | DeepSeek |
| **AntGroup-Opt** (branch) | SMFree RDMA+LL-SBO(Down GEMM overlap)+LL-Layered(rail优化) |蚂蚁 |
| **Mori-EP** (branch) | ROCm/AMD GPU支持 | Mori |

---

## 与7框架的关系

| 框架 | MoE EP通信 | DeepEP关系 |
|------|-----------|------------|
| **Megatron-LM** | 3种dispatcher(AllGather/AlltoAll/Flex) | DeepEP是Megatron AlltoAll的强化版(asymmetric优化) |
| **DeepSpeed** | 无MoE层 | 无直接关系 |
| **vLLM** | expert_parallel (NCCL all-to-all) | DeepEP可替代vLLM的EP通信(需SM90) |
| **verl** | via Megatron strategy → NCCL | 可替换Megatron EP backend |
| **rLLM** | via verl → Megatron → NCCL | 同verl |
| **MindIE** | HCCL (昇腾) | 不同硬件平台 → 无交集 |
| **PyTorch** | c10d_functional.all_to_all | DeepEP底层用NCCL Gin → PyTorch之上 |

**核心定位**: DeepEP填补了MoE EP通信的空白 — 介于NCCL通用集合通信和MoE-specific asymmetric优化之间 → DeepSeek-V3生产级 → 未来可能成为MoE EP标准

---

## 参考资料

- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 本地源码: rollout-infra/deepep-src/
