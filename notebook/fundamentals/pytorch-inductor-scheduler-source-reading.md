# PyTorch Inductor Scheduler 源码阅读: Fusion + Memory Planning + Codegen

> 2026-06-15 | 源码级概览 (scheduler.py=2305行, lowering.py=5400行, ir.py=7305行, triton.py=3008行)
> 源码路径: torch/_inductor/scheduler.py, codegen/triton.py
> 衔接: notebook/fundamentals/pytorch-compiler-roadmap-26-28.md (roadmap+Inductor概览)
> 衔接: notebook/fundamentals/torch-compile-deep-dive.md (compile架构)

## 1. Scheduler 核心类

```
scheduler.py 核心类层次:

  BaseSchedulerNode (120行) — 基类
    ├── ExternKernelSchedulerNode (633行) — 外部kernel(如cuBLAS)
    ├── NopKernelSchedulerNode (665行) — 空操作
    ├── SchedulerNode (669行) — 普通计算节点(pointwise/reduction)
    ├── FusedSchedulerNode (797行) — 融合后的节点 ← 关键!
    │     └── ForeachKernelSchedulerNode (947行) — foreach批量操作
    └── Scheduler (1184行) — 主调度器 ← 最核心!

  关键: FusedSchedulerNode = 多个SchedulerNode合并成1个 → 1个Triton kernel!
```

## 2. Scheduler.__init__ 初始化流程

```
Scheduler.__init__(nodes):

  1. create_scheduler_node(n) → 为每个IR节点创建SchedulerNode
  2. prune_deps() → 剪枝无用依赖
  3. compute_dependencies() → 计算节点间数据依赖关系
  4. topological_sort_schedule() → 拓扑排序 → 执行顺序
  5. dead_node_elimination() → 消除死节点(无消费者)
  6. compute_ancestors() → 计算祖先节点(用于cycle检测)
  7. create_foreach_nodes() → 创建foreach批量操作节点
  8. topological_sort_schedule() → 重排序(foreach可能改变顺序)
  9. fuse_nodes() ← 融合! ← 最关键步骤!
  10. compute_last_usage() → 计算buffer最后使用 → 内存释放时机
```

## 3. fuse_nodes() — 融合算法 (核心!)

### 3.1 迭代融合

```
fuse_nodes(): 迭代最多10轮!
  for i in range(10):
    old_len = len(self.nodes)
    self.fuse_nodes_once() ← 每轮尝试所有可能融合
    new_len = len(self.nodes)
    if new_len == old_len or new_len == 1: ← 无变化或只剩1个 → 停止

→ 多轮迭代 → 因为融合可能创造新的融合机会!
  例: [A, B, C] → 第1轮融合(A,B) → [AB, C]
      → 第2轮可能融合(AB, C) → [ABC]
```

### 3.2 fuse_nodes_once() 融合一轮

```
fuse_nodes_once():
  for node1, node2 in self.get_possible_fusions(): ← 候选融合对
    node1 = name_to_fused_node[node1] ← 可能已融合
    node2 = name_to_fused_node[node2]
    if self.can_fuse(node1, node2): ← 融合合法性检查
      if not self.will_fusion_create_cycle(node1, node2): ← cycle检测
        if self.speedup_by_fusion(node1, node2): ← benchmark检查(可选)
          node3 = fuse(node1, node2) ← 执行融合!
          name_to_fused_node更新

→ 三重检查: can_fuse + no_cycle + speedup → 确保融合安全+有益
```

### 3.3 can_fuse() 融合合法性规则

```
can_fuse(node1, node2) → False的条件:

  1. node1==node2 → 不能融合自身
  2. ExternKernel/NopKernel → 外部kernel不融合(除非template)
  3. node2依赖node1的祖先 → 顺序冲突 → cycle
  4. device不同 → 跨设备不融合!
  5. no_shared_data → 不共享数据 → 融合无意义
     (除非aggressive_fusion=True, 且不是reduction)
  6. node2是template → template只融合epilogue
  7. node1是template + node2有aliasing/mutation/reduction → 不融合
  8. max_fusion_size超限 → 太多ops → 1个kernel太复杂
  9. atomic_add mutation → 不融合(会破坏原子操作)

→ 最关键规则: **必须共享数据 + 同设备 + 顺序正确 + 不超大小**
```

### 3.4 score_fusion_memory() — 共享数据评分

```
score_fusion_memory(node1, node2):
  → 计算node1和node2共享的buffer数量/大小
  → 共享越多 → 融合收益越大 → 减少内存读写!
  → score=0 → 不共享 → 融合无意义(除非aggressive)

  LLM例子:
    q_proj + k_proj + v_proj → 共享input → 高score → 应融合!
    layer1.output + layer2.input → 1个buffer → 中score → 可融合
    无关操作 → score=0 → 不融合
```

## 4. Triton Codegen (codegen/triton.py, 3008行)

### 4.1 核心类

```
codegen/triton.py核心:

  TritonKernel:
    - 从FusedSchedulerNode的IR生成Triton kernel源码
    - 生成grid计算: grid = (ceil(numel / BLOCK_SIZE), 1, 1)
    - 生成kernel body: pointwise/reduction操作 → tl.load/store

  TritonCodeGen:
    - 遍历scheduler的fused nodes → emit kernel定义 + C++ launch wrapper
    - 生成C++ wrapper函数 → 调用Triton kernel → 处理参数传递

  关键: **整个fusion group → 1个Triton kernel → 1次GPU launch → 减少overhead!**
```

### 4.2 Pointwise vs Reduction Kernel

```
Pointwise kernel (element-wise):
  @triton.jit
  def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(ptr + offsets, mask=mask)
    result = op(data)  ← 融合多个pointwise ops
    tl.store(out_ptr + offsets, result, mask=mask)

Reduction kernel (两步):
  Step 1: Partial reduce → 中间结果
  Step 2: Finalize → 完整reduce
  → 大维度reduction需要分步 → 防止一个program处理太多数据

→ 类似FlashAttention的split-k思路 → 分步reduce → 适配GPU并行!
```

## 5. Lowering (lowering.py, 5400行)

```
Lowering: FX Graph → Inductor IR

  核心映射:
    torch.add → PointwiseOp(ComputedBuffer, "add")
    torch.sum → ReductionOp(ComputedBuffer, "sum")
    torch.matmul → ExternKernel(cuBLAS GEMM)
    torch.nn.functional.linear → ExternKernel(cuBLAS) + epilogue
    torch.relu → PointwiseOp

  关键: **matmul/linear → ExternKernel → 不融合(只有epilogue)!**
  → GEMM由cuBLAS执行 → Inductor只融合GEMM后的pointwise ops
  → 这就是为什么LLM的GEMM不会被Inductor重新生成Triton kernel!
```

## 6. Memory Planning

```
compute_last_usage():
  → 每个buffer → 计算最后一次使用位置 → 确定释放时机
  → buffer_names_to_free → 代码生成中标记释放

  策略:
  - Extern inputs → 不释放(模型权重)
  - Temporary buffers → 最后一个consumer后释放
  - Fused buffers → 融合内自动复用(无需显式释放)

  → 类似vLLM的KV Cache block管理 → 生命周期驱动释放!
```

## 7. LLM特定影响

```
LLM + Inductor:

  1. GEMM(cuBLAS) → 不融合 → 只融合GEMM后的RMSNorm/SiLU/Residual
  2. Attention → ExternKernel(FlashAttention) → 不融合
  3. LoRA → pointwise ops → 可以融合 → compile加速LoRA训练!
  4. GRPO advantage → reduction → 两步kernel → 但可以与loss融合

  RTX 4090最优:
    - GEMM: cuBLAS最优(不用Inductor Triton)
    - RMSNorm+SiLU+Residual: Inductor Triton融合 → 减少内存读写
    - LoRA: pointwise → 融合 → 加速
    - 不推荐: max-autotune(首次编译太慢) → reduce-overhead更好

  关键结论: Inductor对LLM的加速主要来自**GEMM后epilogue的融合**
  → 不是替换GEMM本身 → GEMM仍由cuBLAS处理!
```

## 8. 跨框架对比

| 维度 | Inductor Scheduler | DeepSpeed ZeRO-3 Coordinator | vLLM Scheduler |
|------|--------------------|------------------------------|----------------|
| Fusion | 贪心迭代(10轮)+共享数据评分 | 无(kernel预定义) | 无(专用kernel) |
| Memory | last_usage→释放时机 | ds_status→gather/partition | block管理→preempt |
| Codegen | Triton自动生成 | 无(使用原始ops) | C++/CUDA手写 |
| 排序 | 拓扑排序+cycle检测 | layer顺序固定 | priority+ preempt |

## 9. 下一步

- [x] Inductor scheduler源码概览 → 本文件
- [x] can_fuse()合法性规则 → Section 3.3
- [x] Triton codegen概览 → Section 4
- [ ] GPU可用时: 实测compile各模式性能(default/reduce-overhead/max-autotune)
- [ ] 深入lowering.py具体op映射规则
- [ ] 实测FSDP2+compile vs ZeRO-3+eager在7B训练

---

Sources:
- `torch/_inductor/scheduler.py` (2305 lines) — Scheduler, can_fuse, fuse_nodes
- `torch/_inductor/codegen/triton.py` (3008 lines) — TritonKernel, TritonCodeGen
- `torch/_inductor/lowering.py` (5400 lines) — FX→IR mapping
- `torch/_inductor/ir.py` (7305 lines) — PointwiseOp, ReductionOp, Buffer
