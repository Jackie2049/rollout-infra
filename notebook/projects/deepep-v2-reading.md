# DeepEP V2 — 高效 Expert Parallelism 通信库

> deepseek-ai/DeepEP | 2025-06 | MIT License
> 与我们 Expert Parallelism Simulator 的直接关联项目

## 1. 概述

DeepEP 是 DeepSeek 开源的高性能 Expert Parallelism 通信库, 为 MoE 模型 (如 DeepSeek-V3) 提供:
- **All-to-All GPU kernels**: MoE dispatch (分发) + combine (合并)
- **低精度支持**: FP8 dispatch + BF16 combine
- **零 SM 占用**: RDMA/Copy Engine 模式释放 SM 给计算
- **扩展支持**: PP/CP/Engram (远程内存访问)

V2 是完全重构版本, 相比 V1:
- NVSHMEM → NCCL Gin backend (更轻量)
- 高吞吐+低延迟统一为 `ElasticBuffer` 接口
- 分析式 SM/QP 计算 (无需 auto-tuning)
- V3 训练 SM 使用从 24 降至 4-6

## 2. 核心架构

### 2.1 ElasticBuffer

V2 统一接口, 一次初始化支持训练+推理所有场景:

```python
# 初始化 — 自动分析最优 SM 数量
buffer = ElasticBuffer(
    group,                          # NCCL ProcessGroup
    num_max_tokens_per_rank=8192,   # 每个rank最大token数
    hidden=7168,                    # 隐藏维度 (DeepSeek-V3)
    num_topk=8,                     # Top-K 专家数
    use_fp8_dispatch=False,         # FP8 dispatch
)
num_comm_sms = buffer.get_theoretical_num_sms(num_experts, num_topk)
```

### 2.2 Dispatch/Combine 流程

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
    dispatch_backward = combine
    combine_backward = dispatch
```

### 2.3 通信-计算重叠

```python
# 关键: async_with_compute_stream=True
recv_x, ..., handle, event = buffer.dispatch(..., async_with_compute_stream=True)

# 在通信进行时做独立计算
independent_work()  # 不依赖 recv_x

# 等待通信完成
event.current_stream_wait()

# 现在安全使用 recv_x
y = expert(recv_x)
```

`EventOverlap` 管理 compute stream 和 communication stream 的依赖:
- dispatch/combine 在通信流异步执行
- `event.current_stream_wait()` 在计算流插入等待
- 实现零开销的通信计算重叠

## 3. 性能数据

### 3.1 DeepSeek-V3 配置测试

配置: 8K tokens/batch, 7168 hidden, top-8, FP8 dispatch + BF16 combine

| 架构 | 网卡 | 拓扑 | Dispatch BW | Combine BW | SM 数 |
|------|------|------|------------|-----------|-------|
| SM90 (H100) | CX7 (IB) | EP 8×2 | 90 GB/s | 81 GB/s | 12 |
| SM90 | CX7 | EP 8×4 | 61 GB/s | 61 GB/s | 6 |
| SM100 (B200) | CX7 | EP 8×2 | 90 GB/s | 91 GB/s | 12 |
| SM100 | NVLink | EP 8 | **726 GB/s** | **740 GB/s** | 64 (峰值) |
| SM100 | NVLink | EP 8 | 643 GB/s | 675 GB/s | 24 (最小SM) |

### 3.2 与我们 EP Simulator 的对比

我们的模拟器 (NVLink 理论模型):
- NVLink BW: 900 GB/s → 通信开销 <4%
- 8 GPUs EP: 7.74x 加速, 96.8% 效率

DeepEP 实测:
- NVLink EP 8: 726-740 GB/s (理论 900 GB/s → 80-82% 利用率)
- SM 从 24 降至 4-6, 性能不变 → 更多 SM 留给计算

**结论**: 我们的模拟器理论分析方向正确, 实际带宽约理论值的 80%。

## 4. 推理 Decode 优化

### 4.1 Handle Caching

```python
# 首次 decode: 正常 dispatch
recv_x, ..., handle, event = decode_dispatch(x, topk_idx, topk_weights, ...)

# 后续 decode: 缓存 handle, 跳过布局重算和 CPU 同步
recv_x, ..., handle, event = decode_dispatch(x, cached_handle=handle)
```

当 gating 决策不变时 (同一请求的 decode 步骤), 重用 handle 避免了:
- CPU-GPU 同步 (最大延迟来源!)
- 布局重算 (token 路由分配)

### 4.2 与 vLLM/SGLang 的关系

- vLLM EP: 使用自定义 all-to-all (Python 层)
- SGLang EP: 也使用自定义 all-to-all
- DeepEP: 底层 CUDA kernel, 可作为 vLLM/SGLang 的 EP 通信后端

## 5. 网络配置最佳实践

| 配置 | 建议 | 原因 |
|------|------|------|
| InfiniBand | 推荐 | 原生 RDMA 支持 |
| RoCE | 理论兼容 | 未充分测试 |
| Adaptive Routing | 开启 | 多路径均衡, 虽有额外延迟 |
| Congestion Control | 关闭 | 损害最大带宽 |
| Traffic Isolation | 按虚拟通道分离 | EP vs 其他流量隔离 |
| PCI Atomic Mode | 设为 4 | 提升 RDMA atomic 性能 |

## 6. 实验性功能

### 6.1 0 SM Engram (RDMA)
- 远程内存访问, 不占用任何 GPU SM
- 可用于异构 EP (GPU 数量不均衡)

### 6.2 0 SM PP (RDMA)
- Pipeline Parallelism 通信不占 SM
- 与计算完全重叠

### 6.3 0 SM CP (Copy Engine)
- Context Parallelism 通过 Copy Engine
- 解放 SM 资源

## 7. 技术要点

### 7.1 JIT 编译
所有 kernel 运行时 JIT 编译:
- 无需安装时 CUDA 编译
- 缓存在 `$HOME/.deep_ep`
- 支持调试/转储 PTX/SASS

### 7.2 NCCL Gin Backend
- Header-only, 轻量级
- 复用已有 NCCL communicator
- 替代 NVSHMEM (V1 的后端)

### 7.3 FP8 Dispatch
- Dispatch 用 FP8 (通信量减半)
- Combine 用 BF16 (精度要求更高)
- 与 DeepSeek-V3 的 FP8 tile-wise 量化一致

## 8. 核心学习

1. **EP 通信是 MoE 的核心瓶颈**: All-to-All 延迟决定 EP 效率上限
2. **SM 资源竞争**: 通信 kernel 占用 SM → 计算可用 SM 减少 → V2 降至 4-6 SM
3. **通信-计算重叠是关键**: `EventOverlap` 机制让 dispatch/combine 与独立计算并行
4. **Handle Caching 优化 decode**: 推理时避免 CPU-GPU 同步, 对延迟敏感场景重要
5. **JIT 编译的工程价值**: 无需预编译, 灵活适配不同 GPU 架构
6. **NVLink EP 是最优**: 726 GB/s vs RDMA 90 GB/s, 节点内 EP 远优于跨节点

## 与我们项目的联系

- **Expert Parallelism Simulator** (`tools/expert_parallel_sim.py`): 分析模型验证了 DeepEP 的实测趋势
- **MoE Architecture** (`notebook/fundamentals/moe-architecture.md`): EP 是 MoE 推理的关键并行策略
- **DeepSeek-V3 Notes** (`notebook/papers/deepseek-v3.md`): DeepEP 是 DeepSeek-V3 训练的通信核心
- **开源贡献机会**: DeepEP 是活跃的开源项目, 有多个实验性分支 (Zero-copy/Eager/Hybrid-EP)

## 参考

- GitHub: https://github.com/deepseek-ai/DeepEP
- V1 文档: docs/legacy.md (NVSHMEM-based)
- 引用: Chenggang Zhao et al., "DeepEP: an efficient expert-parallel communication library", 2025
