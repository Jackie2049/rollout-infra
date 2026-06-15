# PyTorch 2.12.0 Features 深度阅读

> Released: 2026-05-13 | Reading Date: 2026-06-15
> Source: [Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.12.0), GitHub PRs, Triton 3.7.0 Release Notes
> Previous: pytorch-fsdp2-internals, pytorch-dtensor-autograd, pytorch-compile-e2e, pytorch-dynamo-internals, pytorch-inductor-triton-codegen

---

## 0. Release Overview

PyTorch 2.12.0 是 2026 年最重要的基础设施发布之一，6 个 Highlights:

| Highlight | PR | 影响 |
|-----------|-----|------|
| Batched linalg.eigh on CUDA 100x faster | cuSolver backend | 科学计算 |
| ★ torch.accelerator.Graph | #171285 | 统一 CUDA/XPU graph API |
| torch.export.save supports MX quant | AOTI shim | 极限压缩模型导出 |
| Adagrad fused=True | 单 kernel optimizer | 训练速度 |
| ★ torch.cond + CUDA graphs | #168912 | 数据依赖控制流→GPU |
| ROCm expandable segments | ROCm 生态 | AMD GPU |

**关键基础设施变化**: FSDP2 breaking change(fullgraph), DTensor twice-differentiable, per-parameter meshes, DataParallelMeshDims SPMD, batch_isend_irecv compile, Triton 3.7

---

## 1. ★★★ torch.accelerator.Graph — 统一 CUDA/XPU Graph API

### 1.1 问题背景

之前 `torch.cuda.CUDAGraph` 和 `torch.xpu.XPUGraph` 是两个完全独立的 API，代码重复，用户必须写设备特定代码。随着 XPU(Intel GPU)和 PrivateUse1(第三方加速器)后端增长，统一 API 成为必需。

### 1.2 设计与实现

**PR Stack** (6 PRs, #171285 是主 PR):
- #171270: C++ `at::accelerator::Graph` 后端注册机制
- #171285: ★ Python `torch.accelerator.Graph` 统一前端接口
- #171313: CUDA backend 支持 (`REGISTER_GRAPH_IMPL`)
- #171445: `torch.accelerator.is_graph_available()` 检查可用性
- #171780: 共享 mempool 支持
- #171443: 其他辅助改动

**核心 API**:
```python
# 统一 API — 不再需要 torch.cuda.CUDAGraph 或 torch.xpu.XPUGraph
g = torch.accelerator.Graph()
with torch.Stream(), g:  # ← context manager 模式
    # capture region
    result = model(input)

g.replay()  # replay on whichever accelerator is active
```

**设计关键差异 vs torch.cuda.graph**:
1. **Stream 选择语义**: `torch.cuda.graph` 隐式创建非默认 stream 作为 capture stream；`torch.accelerator.Graph` **总是在当前 stream 上 capture**。在 CUDA 上，如果当前 stream 是默认 stream，会报错"CUDA graphs must be captured on a non-default stream"
2. **Memory pool**: `torch.cuda.graph` 允许指定 graph pool 共享内存；`torch.accelerator.Graph` 在构造时支持指定 pool
3. **不提供 `torch.accelerator.graph`**: 只有 `torch.accelerator.Graph` (大写 G) 作为 context manager，不再有函数式 `graph()` 包装器

**Backend Dispatch 机制**:
- `torch.accelerator.Graph.__init__` → 调用 `torch.accelerator.current_accelerator()` 确定后端
- 通过 `REGISTER_GRAPH_IMPL` 注册后端 → CUDA 重用 `CUDAGraph` 实现，XPU 有自己的实现
- #176421: XPU backend 支持 (merged)
- #171313: CUDA backend 支持 (open，但核心逻辑已在 CUDAGraph 中)

**PrivateUse1/第三方后端**: RFC #158827 提出 Graph 通用化 → 第三方加速器可通过 `REGISTER_GRAPH_IMPL` 注册自己的 graph 实现

### 1.3 对 AI infra 工程师的影响

★★★ **vLLM/Megatron CUDA graph 代码迁移**: 当前这些框架都直接使用 `torch.cuda.CUDAGraph`。迁移到 `torch.accelerator.Graph` 可以:
- 自动适配 XPU/PrivateUse1 后端
- 统一 mempool 管理
- 但需要注意: 必须显式设置非默认 stream（不再自动创建）

★★ **向后兼容**: `torch.cuda.CUDAGraph` 和 `torch.xpu.XPUGraph` 暂不移除 → 旧代码继续工作，但长远应迁移

★ **vLLM CUDAGraphDispatcher**: vLLM V1 的 `CUDAGraphDispatcher` 使用 `torch.cuda.CUDAGraph.capture()` 和 `replay()` → 迁移到 `torch.accelerator.Graph` 后可在 Ascend/XPU 上自动工作，配合 vLLM-Ascend NPU 适配

---

## 2. ★★ DTensor Twice-Differentiable Redistribution

### 2.1 问题背景

DTensor 的 redistribution (改变分片方式, 如 Shard(0)→Replicate via AllGather, Replicate→Shard(0) via ReduceScatter) 之前只支持一次微分。这意味着:
- FSDP2 的 AllGather→forward→ReduceScatter→backward 中的 redistribution 只能通过一次 backward
- 如果需要二阶梯度 (meta-learning, gradient-based hyperparameter optimization, neural network architecture search)，backward 中的 redistribution 本身也需要可微
- 之前 `Redistribute` 和 `NestedRedistribute` 的 backward 函数不支持 `grad(grad(...))`

### 2.2 实现详情

**PRs**:
- #160509: ★ "Add support for twice-differentiable DTensor redistribution" (主 PR)
- #160313: "Make Redistribute autograd function twice-differentiable" (Redistribute 部分)
- #179495: Fix precision loss in NestedRedistribute backward dtype handling
- #182032: Fix redistribute(backward_dtype=...) ignoring the backward dtype

**核心机制**:
- `Redistribute` 和 `NestedRedistribute` 的 backward 函数现在也通过 autograd dispatcher → 二阶 backward 可以 trace 通过 redistribution 逻辑
- 具体实现: backward 函数内部的操作 (如 AllGather/ReduceScatter) 现在也注册了 autograd Function → 梯度的梯度可以正确传播

**数学含义**:
- Shard(0)→Replicate (forward AllGather): backward = ReduceScatter ( Shard(0) )
- 现在 ReduceScatter 的 backward = AllGather ( Shard(0)→Replicate ) → 二阶梯度正确传播
- 这与 DTensor autograd reading 中的 gradient placement 规则一致:
  - Shard→Shard: 无通信
  - Replicate→Replicate: 无通信
  - Partial→Replicate: AllReduce

### 2.3 对 FSDP2 的影响

★★ **FSDP2 + 二阶梯度**: FSDP2 使用 DTensor 作为核心抽象:
- `fully_shard` → 参数变成 `DTensor(Shard(0))`
- forward: AllGather → full params → compute → reshard
- backward: AllGather → full params → compute grad → ReduceScatter
- 之前 backward 中的 AllGather/ReduceScatter 不可微 → 二阶梯度被截断
- 现在 twice-differentiable → FSDP2 训练可以用 `torch.autograd.grad(loss, params, create_graph=True)` → meta-learning within FSDP2

★ **实际影响**: 大多数 LLM 训练不需要二阶梯度 (GRPO/PPO 只需要一阶)。但以下场景受益:
- MAML-style meta-learning with FSDP2
- Gradient-based hyperparameter tuning within distributed training
- Neural ODE / physics-informed neural networks with FSDP2

### 2.4 grad_placements in from_local()

**PR #175867**: Add `grad_placements` parameter to `DTensor.from_local()`

**问题**: 在 MoE (torchtitan #2416) 中发现 `from_local(Replicate)` 在 ColwiseLinear (shared expert) 的 backward 产生了不必要的 all-reduce → 期望的 gradient placement 是 `Partial`，不是 `Replicate`

**解决方案**:
```python
# 之前: gradient placement 由 autograd 自动推断 (可能有误)
dt = DTensor.from_local(local_tensor, mesh, placements=[Replicate()])
# gradient placement = Replicate → backward all-reduce (不必要的!)

# 现在: 显式控制 gradient placement
dt = DTensor.from_local(local_tensor, mesh, placements=[Replicate()],
                          grad_placements=[Partial()])
# backward: Partial→Replicate via all-reduce (正确的!)
```

★★★ **MoE + DTensor 关键修复**: 这解决了 MoE shared expert 在 DTensor 中的 numerics bug → 与 DeepEP/AutoEP 的 shared expert overlap 对齐

---

## 3. ★★ DeviceMesh Compile Compatibility

### 3.1 变化

之前 DeviceMesh 和 placements 在 `torch.compile` 下是 Python 对象 → Dynamo 会尝试 trace 它们的内部属性 → 导致 graph breaks 或错误行为。

**2.12 解决方案** (3 PRs):
- #176661: ★ "Make DeviceMesh opaque" → DeviceMesh 在 Dynamo 中被视为 opaque object → 不 trace 内部属性
- #171482: "Make placements opaque" → Placement (Shard/Replicate/Partial) 在 Dynamo 中被视为 opaque → 不 trace 内部属性
- 两者配合 → `torch.compile` 可以正确处理 DTensor 操作 → FSDP2 + compile 兼容性提升

### 3.2 技术细节

**Opaque 机制**: Dynamo 的 `AllowedPythonAccess` → 当对象被标记为 opaque 时，Dynamo 不会 trace 其属性访问 → 只当作外部常量 → graph 不会被 break

**效果**:
```python
# 之前: DeviceMesh 被 Dynamo trace → graph break
mesh = DeviceMesh("nccl", [0, 1, 2, 3])
@torch.compile
def fn(x):
    return some_dtensor_op(x, mesh)  # mesh 访问 → graph break

# 现在: DeviceMesh opaque → no graph break
mesh = DeviceMesh("nccl", [0, 1, 2, 3])
@torch.compile
def fn(x):
    return some_dtensor_op(x, mesh)  # mesh opaque → no graph break ✓
```

### 3.3 对 FSDP2 + compile 的影响

★★★ **FSDP2 + torch.compile 关键解锁**: FSDP2 内部大量使用 DeviceMesh → 之前 compile 会在 DeviceMesh 访问处 graph break → 现在可以更完整地 compile FSDP2 模型

**注意**: FSDP2 hooks 仍然导致 graph break (见第 8 节)，但 DeviceMesh/placements 本身不再 contribute to graph breaks → 减少了不必要的 break 点

---

## 4. ★★ FSDP2 Per-Parameter Meshes

### 4.1 问题背景

之前 `fully_shard(model, mesh)` 使用**单一 mesh** 分片所有参数 → 所有参数共享同一个 DeviceMesh。但在 MoE 模型中:
- expert 参数需要 EP mesh (expert parallel)
- 非 expert 参数 (attention, MLP) 需要 FSDP mesh (data parallel)
- 单一 mesh 无法同时支持两种分片策略

### 4.2 实现

**PR #173509**: `[FSDP2] support per-param mesh`

**核心设计**:
- 每个 `FSDPParamGroup` 可以有自己的 `mesh` 属性
- 不同参数组可以使用不同的 DeviceMesh → 在同一个 model 中混合 EP 和 FSDP 分片
- `fully_shard` 对 transformer_block 指定不同 mesh for experts vs non-experts

**调度**:
- FSDPModule 调度 2 个 all-gather **顺序执行**: 1st on transformer blocks, 2nd on experts
- 每个 param group 有独立的 `reduce_scatter_state` (PR #177319)

**Overlap PR #177319**: `[FSDP2] overlap reduce-scatter with compute for per-param-mesh`
- 每个 param group 有自己的 RS state (`reduce_scatter_states` list)
- 只有 backward 顺序中最后一个 param group waits on/clears prior module's RS states
- ★ AG on dp communicator (mesh size 4) 可以与 RS on efsdp communicator (mesh size 2) overlap

**验证**: TorchTitan DeepSeek-V3 16B profiler traces 确认 overlap 有效

### 4.3 对 MoE 训练的影响

★★★ **MoE + FSDP2 关键特性**: 之前 MoE 训练需要 Megatron 的 FlexTokenDispatcher + ExpertParallel → 现在可以用 FSDP2 per-param mesh:
- expert 参数: EP mesh → expert parallel sharding
- attention/MLP: FSDP mesh → data parallel sharding
- 同一个 `fully_shard` call → 不同参数组不同 mesh

★★ **与 AutoEP 对齐**: DeepSpeed AutoEP (PR#7938) 提供零代码 EP → PyTorch 的 per-param mesh 是另一种 EP 方式 (更细粒度，需要手动配置 mesh)

### 4.4 与 DataParallelMeshDims 的关系

Per-parameter meshes 定义**哪些参数用哪个 mesh** → DataParallelMeshDims 定义**mesh 的哪个维度是 DP** → 两者配合使用

---

## 5. ★★ DataParallelMeshDims SPMD

### 5.1 问题背景

之前 `fully_shard` 只处理 DP 维度不在 DTensor mesh 中的情况 (即 TP/EP mesh → 然后 FSDP 在 DP 维度上 shard)。但实际需要: 参数已经是 DTensor 分布在**完整 SPMD mesh** 上 (包括 DP 维度) → `fully_shard` 需要知道哪些 mesh 维度是 DP → 将 Replicate 的 DP 维度替换为 Shard

### 5.2 实现

**PR #176334**: `[fully_shard][DTensor] Support fully_shard with DTensor on full SPMD mesh`

**核心概念 — DataParallelMeshDims**:
```python
# 2D mesh: TP × DP
mesh = DeviceMesh("nccl", [[0, 1], [2, 3]], mesh_dim_names=("tp", "dp"))

# 参数在 TP 维度已经 Shard(0) → 在 DP 维度 Replicate
# DataParallelMeshDims 指定 "dp" 是 DP 维度
dp_dims = DataParallelMeshDims("dp")

fully_shard(model, mesh, dp_dims=dp_dims)
# → 将 Replicate on "dp" 维度替换为 Shard(0) → FSDP sharding on DP dimension
```

**关键约束**:
- DP 维度必须在原始 DTensor spec 中 carry `Replicate` placements → `fully_shard` 将它们替换为 `Shard`
- 输入/activation 也是 DTensor 分布在完整 SPMD mesh 上
- 支持 3D mesh: TP × DP × EP → 不同维度不同策略

**验证**: TorchTitan PR #2616 确认正确工作

### 5.3 与现有 DP/TP/PP 的关系

| 概念 | 之前 | 2.12 |
|------|------|------|
| DP | FSDP2 在单一 DP mesh 上 shard | DataParallelMeshDims → DP 维度可以是多维 mesh 的一个维度 |
| TP | DTensor TP 在 TP mesh 上 shard | TP + FSDP 可以共存于同一个 2D/3D mesh |
| PP | Pipeline Parallel 独立 | PP + DTensor metadata foundation (#177727/#177728) |
| EP | Expert Parallel 独立 | per-param mesh → EP+FSDP on same model |

★★ **2D/3D SPMD mesh**: 这是 PyTorch 走向 Megatron-style 5D mixed-radix 的关键一步:
- Megatron: TP×DP×PP×EP×CP → 5D process group
- PyTorch 2.12: TP×DP (2D) / TP×DP×EP (3D) → DataParallelMeshDims 指定 DP 维度 → FSDP2 自动 shard DP 维度

---

## 6. ★ torch.cond + CUDA Graphs

### 6.1 问题背景

CUDA graphs 要求 capture 时控制流确定性 → 数据依赖分支 (如 `if x.mean() > threshold`) 之前只能在 CPU 上执行 → 导致 graph break 或需要 CUDA graph trees (vLLM CUDAGraphDispatcher 的 PIECEWISE 模式)

### 6.2 实现

**PR #168912**: `[CUDA] Add support for torch.cond with cuda graphs`

**核心机制 — CUDA 12.4 Conditional Nodes**:
- CUDA 12.4 引入了 `cudaGraphConditionalNode` → 可以在 graph 中嵌入 IF/THEN/ELSE 节点
- `torch.cond` 在 capture 时将**两个分支都 capture** → 运行时通过 conditional node 选择活跃分支
- 参考: NVIDIA Blog "Dynamic Control Flow in CUDA Graphs with Conditional Nodes"

**API**:
```python
@torch.compile(backend="cudagraphs")
def model(x, threshold):
    def true_branch(x):
        return x * 2
    def false_branch(x):
        return x + 1
    return torch.cond(x.mean() > threshold, true_branch, false_branch, [x])
```

**限制**:
- ★ 需要 CUDA 12.4+ (RTX 4090: CUDA 12.x ✓ → 支持)
- ★ 只支持 `eager` 和 `cudagraphs` backend → **不支持 Inductor!**
- 两个分支的输出 shape/type 必须匹配
- 仍然 experimental → API 可能变化

### 6.3 对推理框架的影响

★★ **vLLM CUDAGraphDispatcher**: vLLM V1 的 PIECEWISE 模式用于 decode batch size 变化 → 需要 graph trees → 条件节点可能简化:
- 当前: 多个 CUDA graph for 不同 batch size → PIECEWISE dispatch
- 潜在: 一个 CUDA graph with conditional nodes → 根据 batch size 选择分支 → 减少 graph 数量

★★ **Megatron InferenceEngine**: Megatron 的 DynamicInferenceEngine + continuous batching + CUDA graph 多 batch 维度 → conditional nodes 可能在 single graph 中处理不同 batch → 但需要 Inductor 支持 (当前不支持)

★ **RTX 4090**: CUDA 12.4+ 支持 → `torch.cond` + CUDA graphs 技术上可行 → 但目前只 eager/cudagraphs backend → 不能 torch.compile(mode="inductor") + torch.cond

---

## 7. batch_isend_irecv Compile Compatibility

### 7.1 变化

**PR #161213**: `[dynamo] Enabling batch_isend_irecv to compile`

**核心设计**:
- 之前 `batch_isend_irecv` 返回 Work 对象 → Dynamo 无法 trace → graph break
- 新实现: 修改 function call args → `batch_isend_irecv` 不返回 Work 对象 → graph 中只包含 tensor → Work objects 注册到 coalesced groups
- `isend` 和 `irecv` 也现在可以 compile

**测试**: NCCL, RCCL, Gloo backend 全部验证通过

### 7.2 影响

★★ **TP + compile**: Pipeline Parallel 和 Tensor Parallel 中 `isend/irecv` 用于 P2P 通信 → 之前必须在 eager mode → 现在可以 compile → TP + torch.compile 更完整

★ **注意**: `torch.distributed.nn.functional` ops 现在在 `torch.compile` 下**报 RuntimeError** (#177342) → 必须迁移到 `_functional_collectives` API:
```python
# 2.11 (报错):
torch.distributed.nn.functional.all_reduce(x, op=ReduceOp.SUM)

# 2.12 (正确):
torch.distributed._functional_collectives.all_reduce(x, reduceOp="sum", group=group)
```

---

## 8. ★★★ Breaking Change: FSDP2 fullgraph=True No Longer Supports Hooks

### 8.1 问题

FSDP2 使用 `torch._dynamo.disable` 包裹 hooks → hooks 不被 Dynamo trace → 但 `fullgraph=True` 要求**零 graph break** → 两者矛盾

### 8.2 变化

**PRs**:
- #174863: ★ `[FSDP2] Remove dynamo tracing support from fully_shard` → 移除所有让 Dynamo trace 进入 FSDP2 hooks 的代码 → hooks unconditionally wrapped with `torch._dynamo.disable`
- #174906: `[FSDP2] remove compiled autograd since we are not tracing into hooks` → 移除 compiled autograd 支持

### 8.3 迁移路径

**Release Notes 给出两种方案**:

```python
# 方案 1: fullgraph=False (允许 graph breaks at hooks)
compiled_model = torch.compile(fsdp_model, fullgraph=False)
loss = compiled_model(input).sum()
loss.backward()

# 方案 2: ★ 先 compile 再 fully_shard (hooks 在 compile 之外)
compiled_model_pre_fsdp = torch.compile(model, fullgraph=True)  # ← compile 内部零 break
compiled_model = fully_shard(compiled_model_pre_fsdp, ...)       # ← FSDP hooks 在 compile 外
loss = compiled_model(input).sum()
loss.backward()
```

★★★ **方案 2 是推荐路径**: 先 compile model → 再 apply FSDP → hooks 在 compile boundary 外 → fullgraph=True 可以用于模型本身 → FSDP hooks 在 compiled function 外执行 → 没有 graph break 矛盾

### 8.4 对 AI infra 工程师的影响

★★★ **所有使用 FSDP2 + torch.compile 的代码必须更新**:
- 如果之前用 `torch.compile(fsdp_model, fullgraph=True)` → **现在会失败**
- 必须改为 `fullgraph=False` 或 `compile → then fully_shard` 顺序

★★ **与 torchtitan 实测一致**: torchtitan H100 的 FSDP2 + compile 流程就是 `compile → fully_shard` → +15-16% / +48-50% Float8 → 这个顺序本来就是推荐做法

★ **ZeRO-3 + compile 不兼容** (已知 → 不变) → FSDP2 + compile 需要方案 2 → 这个 breaking change 只是明确化了

---

## 9. Triton 3.7.0 (PyTorch Pin Update)

### 9.1 版本信息

PyTorch 2.12.0 pins Triton to **3.7.0** (released 2026-05-07):
- PR #174896: Initial Triton 3.7 pin
- PR #178821: Update to top of release/3.7.x branch → version 3.7.0
- PR #179971: Cherry-pick update (AMD fixes + plugin extension enablement)
- PR #179586/#177364/#177723: Additional pin updates

**注意**: Triton 3.7.1 尚未发布 (截至 2026-06-15 最新是 v3.7.0)。MEMORY.md 中提到的 "Triton 3.7.1" 可能是预期中的 patch release，但尚未 tagged。

### 9.2 Triton 3.7.0 关键变化 (对 Inductor 相关)

| 类别 | 变化 | Inductor 影响 |
|------|------|--------------|
| ★ Frontend | `tl.squeeze`/`tl.unsqueeze` | Inductor codegen 可用新 ops |
| ★ Frontend | Scaled BMM 支持 | FP8/MX GEMM 更好 |
| ★ Frontend | FP8 constants | FP8 kernel 更容易写 |
| ★ Frontend | `tl.cat(can_reorder=False)` | 非 reorder cat → fusion 更安全 |
| Frontend | `make_block_ptr` deprecated → tensor descriptors | ★ 迁移路径 |
| ★ Backend | 2CTA/Multicast/TMA support | Hopper/Blackwell GEMM 更强 |
| Backend | Warp Specialization nested loops | 更复杂 WS 内核 |
| ★ Backend | SM89 ptxas workaround reverted (#9756) | ★ RTX 4090 SM89 → 之前有 workaround → 现在已移除 (bug 已在 ptxas 中修复) |
| ★ NVIDIA | tcgen05 MMA on sm110 (Jetson Thor) | 新 GPU 支持 |
| NVIDIA | TMA im2col mode | Conv TMA 加速 |
| ★ AMD | gfx1250 RDNA4 maturation, WS, TDM | AMD GPU 大进步 |
| AMD | AsyncCopy default on gfx950/gfx1250 → reverted | 需要 opt-in |
| ★ Breaking | `triton_kernels` matmul refactor (BC-breaking API) | 下游用户需 review |
| Breaking | Proton `GlobalScratchAllocOp` deprecated | out-of-tree 迁移 |

### 9.3 对 Inductor 最关键的变化

★★ **SM89 ptxas workaround removed**: RTX 4090 是 SM89 (Ada) → 之前 Triton 有 workaround for ptxas bug → 现在移除 (ptxas 已修复) → RTX 4090 Triton kernel 应更稳定

★★ **`make_block_ptr` deprecated**: Inductor 之前用 `make_block_ptr` codegen TMA loads → Triton 3.7 deprecated → 应迁移到 tensor descriptors → 但 Inductor 可能仍用 `make_block_ptr` 直到完全迁移

★★ **2CTA + Multicast + TMA**: Hopper GPU (H100/H800) 的 grouped GEMM 和 persistent kernel → Inductor max_autotune 的 Triton template 可以用 2CTA → 更高吞吐

★ **Plugin Hooks & Out-of-Tree Dialects**: Triton 现在支持第三方 IR passes → vLLM 的 Triton kernels 可以注册自定义 pass → 但需要 Triton ext enabled

---

## 10. PyTorch 2.12 其他重要变化

### 10.1 Inductor 新特性

★★ **User-defined stream support** (#165390 stack, 7 PRs):
- Inductor codegen stream context managers (enter/exit) 和 `record_stream` calls
- User streams 可以 flow through compiled regions → 正确同步
- ★ 对 overlap scheduling 非常重要: compute stream + comm stream → compile 内 overlap

★★★ **Async activation offloading ops** (#177621):
- `ao::offload`, `ao::reload`, `ao::wait` → 3 个 custom ops
- 封装 async CPU offloading stream management → IR 从 7 nodes(offload)/5 nodes(reload) → **2 nodes each**
- 与 c10d functional collectives 相同的 async 2-op pattern
- ★★ ★ 这与 ZeRO-Offload/ZenFlow 的 overlap_grad 思路一致 → 但在 Inductor IR 中更简洁

★★ **User Triton kernel unary epilogue fusion** (#173662):
- 用户 Triton kernel + downstream pointwise epilogue (e.g. `relu()`) → 自动 fusion
- AST 解析 Triton kernel source → inline epilogue into `tl.store` expression
- ★ vLLM Triton kernels + activation function → 自动 fusion → 省一个 kernel launch

★★ **Custom op out-variant discovery** (#175116/#176117):
- Inductor 自动发现 custom op 的 functional 和 `.out` overload → lowering 到 `.out` variant
- `ExternKernelOut` (should_allocate=True) → memory planner buffer reuse
- ★ vLLM 的 scaled_fp4_quant 等 custom ops → functional API 但自动获得 buffer reuse

★★ **max_autotune for combo kernels** (#177715/#178936/#179317):
- 之前 combo kernels 只有 heuristic configs → `max_autotune` 无效
- 现在: 生成 O(N*K) "phase configs" → 每次变一个 sub-kernel 的 block size → benchmark
- ★ 更精确的 autotuning → 更优的 kernel 性能

★ **Non-TMA persistent Triton templates for mm/addmm** (#177781/#179095):
- 之前 persistent kernels 需要 TMA → RTX 4090 无 TMA → 不能用
- 现在有 non-TMA 版本 → RTX 4090 可能用 persistent mm → 但需要 autotune 验证

★ **CUTLASS FP8 e5m2 backend** (#171176): `torch.float8_e5m2` GEMM autotuning → FP8 训练 inference 更完整

### 10.2 Dynamo 新特性

★★ **aot_compile made public** (#179917/#180008):
- `torch._dynamo.aot_compile` → 公开 API → `aot_eager` 和 `inductor` backend support + docs
- ★ 不再需要用 internal API → AOT compilation 更容易

★★ **recompile_limit in torch.compile** (#177936):
- `torch.compile(fn, recompile_limit=N)` → override per-function recompile cap → 不碰全局 config
- ★ 减少 recompile 风暴 → 更可控

★★ **bdb debugger for Dynamo tracing** (#174626/#174746/#175200):
- `pdb`-style debugger → step through nested frames during Dynamo tracing
- 支持 `n`(next)/`u`(up)/`d`(down)/`r`(return)/`bt`(backtrace)
- ★★ Dynamo debugging 不再需要 read bytecode → interactive debugging

★★ **mark_unbacked min/max bounds** (#176313):
- `torch._dynamo.mark_unbacked(symbol, min, max)` → 传递值范围给 symbolic shape system
- ★ 减少 recompilation → 更精确的 shape specialization

★★★ **fullgraph=True empty compile warning → 2.13 will error** (#181940):
- `fullgraph=True` 下如果 Dynamo 被 bypass (如 TorchDispatchMode active) → 2.12 warn → 2.13 error
- ★★ 这是 2.13 breaking change 的预告 → 必须确保 fullgraph=True 下确实有 compiled code 运行

★ **inline_inbuilt_nn_modules deprecated** (#177489/#178205):
- 现在默认 inline → 设置 flag 会发 deprecation warning → 未来移除

### 10.3 Distributed 新特性

★★ **NCCL communicator suspend/resume/memory_stats** (#176300):
- `backend.suspend()`: free communicator memory → sleep/wake pattern
- `backend.resume()`: restore communicator memory
- `backend.memory_stats()`: return memory usage info
- ★★ 与 verl/vLLM 的 sleep/wake 对齐 → FSDP2 sleep 时可以 free NCCL 内存 → 多框架共享 GPU

★★ **SymmMem for reduce-scatter in FSDP** (#177111):
- `SymmMemReduceScatter` → allocate from symmetric memory pool → NCCL 检测 symm mem tensor → 用 optimized symmetric kernel
- ★ 与 Mori symmetric memory / NVLS 对齐 → ReduceScatter 也可以用 symmetric memory 优化

★★ **Store::barrier API** (#174920):
- TCPStore client BARRIER support → 减少 round trips vs ADD+WAIT pattern
- ★ 更快的 initialization synchronization

★ **reduce_scatter_offset in symmetric memory** (#177791):
- Variable-sized block reductions with NVLink multicast or LSA fallback
- ★ 与 DeepEP asymmetric dispatch 对齐

★ **DDP batched_grad_copy** (#176638):
- Reduce per-parameter kernel launches to 2 kernels per bucket
- ★ DDP + compile → 少 kernel launch → 更快

★★ **AsyncCollectiveTensor fix** (#179849):
- AsyncCollectiveTensor inputs leaking into compiled regions → RuntimeError or silent data corruption in TP + compile
- ★★ ★ TP + compile 的 correctness bug → 2.12 fixed

### 10.4 Pipeline Parallel + DTensor

★★ **DTensor metadata foundation for PP** (#177727/#177728):
- PP stage 和 schedule refactoring → DTensor-aware
- ★ 未来 PP + DTensor (跨 stage 分片) → 与 Megatron PP + DTensor pipeline 对齐

★ **PP: Dispatch homogeneous P2P ops individually** (#175712):
- 避免 stream serialization → 更灵活的 P2P overlap

### 10.5 其他 Breaking Changes

★★ **CUDA 12.6 minimum for source build** (#178925):
- 之前 CUDA 12.4 → 现在 CUDA 12.6 → source build 需更新 CUDA
- ★ RTX 4090: CUDA 12.6 driver ✓ → 无问题

★★ **CUDA 12.8 binaries deprecated** (#179072):
- `cu128` index URL 不再提供 → 迁移到 `cu130`(推荐) 或 `cu126`
- ★ RTX 4090: 用 `cu126` 或 `cu130` wheel → OK

★ **C++20 minimum** (#178662): Source build 需要 C++20 compiler → 不影响 wheel users

★ **Profiler metadata_json deprecated → event_metadata** (#179417)

### 10.6 FSDP2 Bug Fixes & Improvements

★ **FSDP2 param FQN lookup quadratic→linear** (#174675): 大模型 FSDP2 init 加速

★ **Cache FSDP2 shard_mesh** (#179655): Avoid repeated submesh creation → 更快

★ **FSDP2 non-float parameters** (#177948): 排除 non-float parameters from reduce-scatter → 仍然 shard 和 all-gather → ★ MoE router 参数 (int) 不需要 reduce-scatter

★ **FSDP fix _unshard() passing CUDA stream where event expected** (#170525): Correctness fix

### 10.7 Performance Highlights

★★ **DCP consolidation I/O optimization** (#175762): from ~2h to 35s for remote mounts
- ★★ 大规模 checkpoint save/load → remote FS (S3/NFS) → 100x faster

★★ **CP head-tail load balancer** (#178199): up to 1555x speedup for 1M sequence length
- ★★ Context Parallel → 1M tokens → 巨大加速

★ **FSDP2 reduce-scatter overlap with compute for per-param meshes** (#177319): 上面已详述

★ **Blackwell cuBLAS 32MiB workspaces** (#175344): 更大 workspace → 更优 GEMM

---

## 11. PyTorch 2.12.1 / v2.13 Outlook

### 11.1 PyTorch 2.12.1 Preparation

截至 2026-06-15:
- v2.12.1 **尚未发布** — 没有 tagged release 或 release branch
- 预期内容: critical bug fixes from 2.12.0, security patches, regression fixes
- 典型 PyTorch patch release 节奏: major release 后 ~4-6 周

### 11.2 v2.13 Branch Status

截至 2026-06-15:
- v2.13 branch **尚未创建** — 没有 `release/2.13` 或 `v2.13.0` tag
- 预期时间线: 2.12.0 → ~3个月后 → 2.13.0 (约 2026-08)

### 11.3 已知的 v2.13 Breaking Changes (从 2.12 Release Notes 预告)

★★★ **`fullgraph=True` will error on empty compile** (#181940):
- 2.12: warning → 2.13: **error**
- 如果 `fullgraph=True` 下 Dynamo 被 bypass (如 TorchDispatchMode) → 将 raise RuntimeError
- ★★ 所有使用 `fullgraph=True` 的代码必须确保 compiled code 实际运行

★ **CMake < 3.10 will be removed** (#166259): 2.12 deprecation warning → 2.13 可能 error

★ **`inline_inbuilt_nn_modules` config removed** (#177489): 2.12 deprecated → 未来 release 移除

### 11.4 预期的 v2.13 主题 (推测)

基于 2.12 release 和 ongoing development:
- ★ FSDP2 compile 进一步优化 (方案 2: compile→then fully_shard 更成熟)
- ★ Inductor + torch.cond 支持 (当前只有 eager/cudagraphs)
- ★ Triton tensor descriptors migration (make_block_ptr deprecated → Inductor codegen 更新)
- ★ PP + DTensor (metadata foundation laid in 2.12 → implementation in 2.13?)
- ★ torch.accelerator.Graph CUDA backend 完善 (#171313 still open)
- ★ Spmd_types (DTensor sharding type checker → 从 grad_placements PR 的 "longer term" 注释推断)

---

## 12. ★★★ RTX 4090 影响分析

### 12.1 直接利好

| 特性 | RTX 4090 影响 | 等级 |
|------|--------------|------|
| torch.accelerator.Graph | ✓ CUDA backend → 统一 API → 但 CUDA-only 不需要迁移 | ★ |
| DTensor twice-differentiable | ✓ 但 GRPO/PPO 不需要二阶梯度 | ★ |
| FSDP2 per-parameter meshes | ✗ 需要 2+ GPU → RTX 4090 单 GPU 无意义 | ✗ |
| DataParallelMeshDims SPMD | ✗ 需要 2+ GPU → 单 GPU 无 DP mesh | ✗ |
| ★ torch.cond + CUDA graphs | ✓ CUDA 12.4+ → RTX 4090 支持 → 但 Inductor 不支持 → 只 eager/cudagraphs | ★★ |
| batch_isend_irecv compile | ✗ 单 GPU 无 P2P | ✗ |
| ★★ SM89 ptxas workaround removed | ✓ RTX 4090 SM89 → Triton kernel 更稳定 | ★★ |
| FSDP2 fullgraph breaking | ✓ 影响 FSDP2 + compile → 需改 compile→then shard 顺序 | ★★ |
| ★★ Inductor non-TMA persistent mm | ✓ RTX 4090 无 TMA → 但有 non-TMA 版本 → 可能用 | ★★ |
| ★ Inductor user Triton epilogue fusion | ✓ vLLM Triton kernels + relu → 自动 fusion | ★★ |
| ★★ FSDP2 non-float params | ✓ MoE router int params → 不 reduce-scatter → 省通信 | ★ |
| CUDA 12.6 minimum source build | ✓ RTX 4090 driver 支持 | ✓ |
| cu128 deprecated → cu126/cu130 | ✓ RTX 4090 用 cu126 wheel | ✓ |

### 12.2 ★★ 关键 RTX 4090 变化

1. **SM89 ptxas fix**: Triton 3.7 移除了 SM89 workaround → ptxas 自身已修复 → RTX 4090 Triton kernels 应更稳定 → 减少编译时 bug
2. **Non-TMA persistent mm**: RTX 4090 (Ada, SM89) 无 TMA → 之前不能用 Triton persistent GEMM → 现在有 non-TMA 版本 → 可能获得 persistent kernel 加速
3. **torch.cond + CUDA graphs**: RTX 4090 CUDA 12.x → 支持 conditional nodes → 但只 eager/cudagraphs → 不能 Inductor compile → 实际用于 vLLM CUDA graph dispatch 可能简化 PIECEWISE 模式
4. **FSDP2 + compile顺序**: 如果用 FSDP2 + compile on RTX 4090 → 必须 `compile(model) → fully_shard(compiled)` → 否则 fullgraph=True 失败
5. **FSDP2 per-param mesh + SPMD**: ✗ RTX 4090 单 GPU → 无 mesh 概念 → 这些特性需要多 GPU

### 12.3 ★★★ RTX 4090 最佳实践不变

| 方案 | 2.12 变化 | 结论 |
|------|----------|------|
| rLLM Tinker + GRPO + LoRA | 无变化 → 仍然最优 | ✓ 不变 |
| FSDP2 + compile | 顺序必须 compile→then shard | ★★ 变化 |
| INT4 + INT8KV inference | Triton SM89 fix → 更稳定 | ★ 微改进 |
| Triton custom kernels | epilogue fusion → 更快 | ★ 微改进 |

---

## 13. 跨框架影响总结

| 框架 | 2.12 影响 | 关键变化 |
|------|----------|---------|
| ★★ vLLM | CUDA Graph API → torch.accelerator.Graph; torch.cond → PIECEWISE 简化; Triton epilogue fusion; SM89 ptxas fix | vLLM CUDAGraphDispatcher 可迁移到统一 API |
| ★★ Megatron | per-param mesh → MoE EP+FSDP; DataParallelMeshDims → TP+DP SPMD; batch_isend_irecv compile → TP+compile | ★★ Megatron-style 5D mesh → PyTorch 走向同一方向 |
| ★★ verl | NCCL suspend/resume → sleep/wake; FSDP2 per-param mesh → MoE GRPO | sleep/wake 对齐 verl 架构 |
| ★ rLLM | Tinker in-process → torch.accelerator.Graph 可用; compile 顺序变化 | 单 GPU 变化不大 |
| ★ DeepSpeed | ZeRO-3 + compile 不兼容 (不变); FSDP2 per-param mesh vs AutoEP | 两条 EP 路径 |

---

## 附录: 关键 PR 列表

| PR# | Title | 类别 |
|-----|-------|------|
| #171285 | ★ torch.accelerator.Graph unified frontend | Python Frontend |
| #171313 | accelerator.Graph on CUDA | Accelerator |
| #176421 | accelerator.Graph on XPU | XPU |
| #158827 | [RFC] Graph generalization | RFC |
| #168912 | ★ torch.cond + CUDA graphs | CUDA |
| #160509 | ★ twice-differentiable DTensor redistribution | DTensor |
| #160313 | Redistribute twice-differentiable autograd | DTensor |
| #179495 | NestedRedistribute backward dtype fix | DTensor |
| #175867 | ★ grad_placements in from_local | DTensor |
| #176661 | ★ DeviceMesh opaque (compile compat) | DTensor |
| #171482 | Placements opaque (compile compat) | DTensor |
| #173509 | ★ FSDP2 per-param mesh | FSDP2 |
| #176334 | ★ DataParallelMeshDims SPMD | FSDP2 |
| #177319 | ★ overlap RS with compute for per-param mesh | FSDP2 |
| #174863 | ★★★ FSDP2 remove dynamo tracing hooks | FSDP2 Breaking |
| #174906 | FSDP2 remove compiled autograd | FSDP2 Breaking |
| #161213 | ★ batch_isend_irecv compile | Distributed |
| #177342 | nn.functional RuntimeError under compile | Distributed Breaking |
| #176300 | NCCL suspend/resume/memory_stats | Distributed |
| #177111 | SymmMem reduce-scatter in FSDP | FSDP |
| #165390 | ★ Inductor user stream support | Inductor |
| #177621 | ★★ ao::offload/reload/wait ops | Inductor |
| #173662 | ★ Triton epilogue fusion | Inductor |
| #175116 | Custom op out-variant lowering | Inductor |
| #177715 | max_autotune for combo kernels | Inductor |
| #177781 | Non-TMA persistent Triton mm | Inductor |
| #171176 | CUTLASS FP8 e5m2 backend | Inductor |
| #181940 | ★★ fullgraph=True warning → 2.13 error | Dynamo |
| #179917 | aot_compile made public | Dynamo |
| #174896 | Triton 3.7 pin | Release Eng |
| #178821 | Triton 3.7.0 update | Release Eng |
