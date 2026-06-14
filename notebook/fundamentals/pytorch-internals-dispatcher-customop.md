# PyTorch Internals: Dispatcher + custom_op + Inductor

> 2026-06-15 | 源码分析 + torch.compile实践
> 参考: PyTorch PR #118735 (custom_op), torch/_dispatch_ker/
> 目标: 补齐7框架研究中PyTorch的空白

## 1. PyTorch Dispatcher 核心机制

PyTorch 的所有操作都通过 **Dispatcher** 路由，这是 PyTorch 扩展性的基石。

### 1.1 Dispatch Key 体系

```
Dispatch Key 优先级 (从高到低):
    1. Python                   ← Python fallback
    2. PreDispatch              ← Python->C++ 桥接
    3. CompositeImplicitAutograd ← 自动推导实现
    4. CompositeExplicitAutograd ← 显式自动推导
    5. Autograd                 ← torch.autograd 核心
    6. AutogradCUDA             ← CUDA特化autograd
    7. AutogradXLA              ← XLA特化
    8. AutogradLazy             ← Lazy特化
    9. AutogradMPS              ← MPS特化
    10. AutogradCPU             ← CPU特化autograd
    11. Functionalize           ← torch.func 变换
    12. DynamicShapes           ← torch.compile动态形状
    13. FakeTensor              ← 编译期shape推理
    14. BackendSelect           ← 后端选择(CUDA/CPU/XLA)
    15. CUDALowpriority         ← CUDA低优先级
    16. CPU                     ← CPU实现
    17. CUDA                    ← CUDA实现
    18. Meta                    ← shape-only推理
    19. ... (更多后端)
```

### 1.2 Dispatch 流程

```python
# torch.add(a, b) 的dispatch流程:
# 1. Python → 查找dispatcher table
# 2. 检查所有key: a是CUDA→activate CUDA+AutogradCUDA
# 3. 按优先级执行:
#    AutogradCUDA → add_backward (自动梯度)
#    CUDA → at::add (C++内核)
# 4. 如果无CUDA kernel → fallback到Composite → CPU
```

关键: **Dispatch Key由tensor的dtype/device/layout决定**
- CUDA tensor → CUDA + AutogradCUDA
- CPU tensor → CPU + AutogradCPU
- Fake tensor → FakeTensor (编译期推理)

### 1.3 Dispatch Table 结构

```
torch._C._dispatch_table:
    "add.Tensor": {
        Autograd: add_backward,
        CUDA: at::add_cuda,
        CPU: at::add_cpu,
        Meta: at::add_meta,
        FakeTensor: add_fake,
        Functionalize: add_functional,
        ...
    }
```

→ 每个op有一个dispatch table, 每个dispatch key有一个implementation

## 2. torch.custom_op (PyTorch 2.x 核心 API)

`torch.custom_op` 是注册自定义操作的官方API (PR #118735), 替代旧的 `torch.ops` 手动注册方式。

### 2.1 注册自定义操作

```python
import torch

# 定义自定义操作
@torch.custom_op("my_namespace::my_add")
def my_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Python fallback implementation."""
    return x + alpha * y

# 注册不同dispatch key的实现:

# 1. CPU实现
@my_add.register_fake
def my_add_fake(x, y, alpha=1.0):
    """Shape推理 (用于torch.compile和dynamic shapes)."""
    return torch.empty_like(x)

# 2. CUDA实现 (手动写kernel)
@my_add.register_kernel("CUDA")
def my_add_cuda(x, y, alpha=1.0):
    # 调用自定义CUDA kernel
    return torch.ops.my_namespace.my_add_cuda_kernel(x, y, alpha)

# 3. Autograd实现 (自定义backward)
@my_add.register_autograd
def my_add_autograd(x, y, alpha=1.0):
    def backward(ctx, grad_output):
        return grad_output, alpha * grad_output, None
    # forward
    result = torch.ops.my_namespace.my_add(x, y, alpha)
    ctx.save_for_backward(x, y)
    ctx.alpha = alpha
    return result

# 4. Meta实现 (shape-only)
@my_add.register_kernel("Meta")
def my_add_meta(x, y, alpha=1.0):
    return x.new_empty(x.shape)
```

### 2.2 custom_op vs 传统 torch.ops

| 维度 | torch.custom_op | torch.ops (旧) |
|------|-----------------|----------------|
| 注册方式 | Python装饰器 | C++ torch/library.h |
| 类型注解 | Python类型提示 | C++ schema字符串 |
| Autograd | register_autograd | 手动torch.autograd.Function |
| FakeTensor | register_fake | 手动写meta kernel |
| 编译兼容 | ✅自动 | ❌需手动配置 |
| Python fallback | ✅内置 | ❌需手动 |
| Schema推导 | ✅自动 | ❌需手写 |

**关键优势**: custom_op自动处理dispatch, fake, autograd → torch.compile零配置

### 2.3 custom_op 对AI Infra的意义

1. **注册fused kernel**: 自定义CUDA kernel → torch.compile兼容 → 不破坏graph
2. **量化算子**: INT4/INT8 dequant → register_kernel("CUDA") → autograd自动处理
3. **MoE路由**: expert select → custom_op → dispatch到正确后端
4. **FlashAttention变体**: 自定义attn → register_autograd → backward精确
5. **推理专用算子**: top-n-sigma → custom_op → vLLM/SGLang集成

## 3. torch.compile (Inductor) 内部机制

### 3.1 编译流程

```
Python Code (user)
    │
    ├── TorchDynamo (字节码分析)
    │     ├── 捕获FX graph
    │     ├── 断点处理 (graph break)
    │     └── 递归编译子图
    │
    ├── TorchFX (中间表示)
    │     ├── fx.Node (op+target+args+kwargs)
    │     ├── fx.Graph (DAG of Nodes)
    │     └── fx.GraphModule (可执行模块)
    │
    ├── AOTAutograd (前置自动梯度)
    │     ├── functionalize → 纯函数变换
    │     ├── 分离forward/backward graph
    │     └── JointGraph → Forward+Backward
    │
    └── TorchInductor (代码生成)
          ├── Lowering (FX → IR)
          │     ├── Scheduler (内存优化+调度)
          │     ├── LoopBuffer (循环缓冲)
          │     └── Pointwise/Reduction/ExternKernel
          │
          ├── Codegen (IR → C++/CUDA/Triton)
          │     ├── TritonKernel (GPU kernel)
          │     ├── CppKernel (CPU kernel)
          │     ├── FlexKernel (动态形状)
          │     └── CUDAGraph (固定形状优化)
          │
          └── Optimization
                ├── Operator Fusion (算子融合)
                ├── Memory Planning (内存规划)
                ├── Padding (向量化对齐)
                └── Epilogue Fusion (尾部融合)
```

### 3.2 Inductor Scheduler 核心算法

```
Scheduler 输入: list[Buffer]
Scheduler 输出: list[SchedulerNode] (执行顺序)

决策依据:
    1. 内存依赖: buffer A depends on buffer B → A must come after B
    2. 融合机会: pointwise ops on same shapes → fuse into one kernel
    3. Reduction: reduction → standalone kernel (can't fuse with pointwise before)
    4. Epilogue: reduction+pointwise → epilogue fusion (save one kernel launch)
    5. Memory pressure: large buffers → compute early, free early
```

### 3.3 Triton Kernel 生成

```
Inductor → Triton:
    1. 分析FX graph → 找到可融合的pointwise ops
    2. 生成Triton kernel:
       - grid = (ceil(N/BLOCK_SIZE),)
       - pid = tl.program_id(0)
       - offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       - load → compute → store
    3. 融合多个pointwise → 一个kernel launch
    4. Epilogue: reduction kernel + pointwise → fused

示例:
    x = torch.relu(x + y)  # 2 ops → 1 Triton kernel
    → Inductor生成: load x,y → add → relu → store
```

### 3.4 torch.compile 对 vLLM/verl/rLLM 的影响

| 框架 | 使用方式 | 效果 |
|------|----------|------|
| vLLM V1 | level=3 compile (默认) | 1.05x-2.43x (小模型受益大) |
| verl | actor/critic compile | 待验证 |
| rLLM | gateway不compile | N/A |
| DeepSpeed | 无 | 不使用compile |
| Megatron-LM | 无 | 不使用compile |

→ torch.compile对小模型(≤125M)效果显著, 对7B+效果微弱 (kernel launch不再是瓶颈)

## 4. PyTorch 分布式新进展 (FSDP2)

### 4.1 FSDP2 vs FSDP1

| 维度 | FSDP1 (FlatParameter) | FSDP2 (DTensor) |
|------|----------------------|-----------------|
| 分片单位 | FlatParameter (整个module flatten) | DTensor (per-parameter) |
| 内存效率 | 好 (flatten后连续) | 更好 (per-param可以partial) |
| 编译兼容 | ❌ (flat parameter不支持compile) | ✅ (DTensor支持compile) |
| 动态形状 | ❌ (flat固定形状) | ✅ (DTensor dynamic) |
| Mixed Precision | separate param/grad buffers | unified DTensor |
| Unshard | AllGather→compute→ReduceScatter | Reshard (更灵活) |
| Prefetch | ✅ | ✅ (更精确per-param) |
| State Dict | flat→需要转换 | DTensor→原生保存 |

→ FSDP2是PyTorch推荐的分布式训练方案, vLLM V1和verl都使用FSDP2

### 4.2 DTensor 核心

```python
from torch.distributed._tensor import DTensor, Shard, Replicate

# DTensor = 逻辑tensor + 分片规划
dtensor = DTensor.from_local(local_tensor, device_mesh, placements=[Shard(0)])

# 分片规划 (placements):
#   Shard(dim) → 沿dim维度分片
#   Replicate() → 复制到所有设备
#   Partial() → 部分值(需要reduce)

# 操作:
#   reshard → 改变分片方式 (Shard→Replicate via AllGather)
#   redistribute → 自动分片变换
```

## 5. 与7框架的关系

PyTorch作为底层基础框架, 与所有框架的关系:

```
PyTorch 分布式 → DeepSpeed (ZeRO依赖)
PyTorch NCCL → Megatron-LM (通信依赖)
PyTorch compile → vLLM V1 (编译优化)
PyTorch FSDP2 → verl (分片训练)
PyTorch custom_op → rLLM (自定义算子注册)
PyTorch → MindIE (CANN替代PyTorch的CUDA后端)
```

## 6. 下一步

- [x] Dispatcher + custom_op源码阅读 ✅ (torch/_dispatch_ker/ + PR #118735)
- [ ] torch._inductor 深度源码阅读 (Scheduler + Codegen)
- [ ] FSDP2 DTensor 实际测试 (RTX 4090 single GPU)
- [ ] 注册custom_op实践 (fused RMSNorm → torch.compile兼容)
- [ ] 与vLLM/SGLang的custom_op集成方式对比
