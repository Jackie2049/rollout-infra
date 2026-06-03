# PyTorch torch.compile 与 Inductor 后端

> PyTorch 2.0 的 JIT 编译系统 — 自动将 PyTorch 代码编译为高性能 Triton kernel

## 1. 为什么 torch.compile 重要

```
传统 PyTorch Eager 模式的问题:
  1. 每个 op 是独立的 Python 函数调用 → kernel launch overhead
  2. 中间结果必须写回 HBM → 内存带宽浪费
  3. 无法跨 op 优化 → 错失融合机会
  4. Python 解释器开销 → CPU 成为瓶颈

torch.compile 解决方案:
  1. 图捕获: 将 Python 代码转为计算图 (FX Graph)
  2. 自动融合: 多个小 op 编译为一个大 kernel
  3. Triton 代码生成: Inductor 后端自动生成 Triton kernel
  4. CUDA Graphs 集成: 消除 kernel launch overhead

性能收益:
  - 训练: 1.2-2x 加速 (典型 workload)
  - 推理: 1.5-3x 加速 (小 batch 更显著)
  - 开发成本: 零 — 只需加一行代码
```

## 2. 架构总览

```
Python 代码
    │
    ▼
┌─────────────┐
│   Dynamo     │  图捕获引擎
│  (Frontend)  │  将 Python 代码转为 FX Graph
└──────┬──────┘
       │ FX Graph
       ▼
┌─────────────┐
│ AOTAutograd  │  前向图 + 反向图联合优化
│              │  将 eager autograd 提前展开
└──────┬──────┘
       │ Joint Graph
       ▼
┌─────────────┐
│  Inductor    │  代码生成后端
│  (Backend)   │  FX Graph → Triton/CPP 代码
└──────┬──────┘
       │
       ├──→ Triton Kernel (GPU)
       └──→ C++/OpenMP Kernel (CPU)
```

### 2.1 Dynamo — 图捕获

```python
import torch

# Dynamo 的核心工作:
# 1. 拦截 Python 字节码执行
# 2. 将 tensor 操作记录到 FX Graph
# 3. 遇到不支持的 op 时产生 graph break

# 简单示例
@torch.compile
def forward(x, w):
    return torch.matmul(x, w) + torch.relu(torch.matmul(x, w))

# 等价的 FX Graph (简化):
# graph():
#   %x, %w = ...

#   %matmul_1 = torch.matmul(%x, %w)
#   %matmul_2 = torch.matmul(%x, %w)
#   %relu = torch.relu(%matmul_2)
#   %output = %matmul_1 + %relu
#   return %output

# Inductor 优化后 (融合):
# 1. 消除重复 matmul (CSE)
# 2. 将 relu + add 融合为一个 kernel
```

### 2.2 Inductor — 代码生成

```
Inductor 编译流程:
1. 接收 FX Graph
2. 优化 passes:
   - 算子融合 (fusion)
   - 通用子表达式消除 (CSE)
   - 内存布局优化 (memory planning)
   - 形状推导 (shape propagation)
3. 代码生成:
   - GPU → 生成 Triton kernel (Python 代码)
   - CPU → 生成 C++ kernel
4. 编译执行:
   - Triton kernel 被 Triton 编译器编译为 PTX
   - 缓存编译结果，下次直接加载

关键优势:
  tl.dot() 自动利用 Tensor Core
  自动 shared memory 管理
  自动 autotune block size
```

### 2.3 AOTAutograd — 提前构建反向图

```
传统 eager autograd:
  前向执行时，动态构建计算图
  → backward() 时逐个 op 反向
  → 每个 op 独立 kernel launch

AOTAutograd:
  在编译时 (ahead-of-time) 构建完整的前向+反向图
  → Inductor 可以优化整个反向传播
  → 反向传播也被融合为少量 kernel

示例:
  Eager: forward 12 kernels + backward 15 kernels = 27 kernel launches
  Compiled: forward 3 kernels + backward 4 kernels = 7 kernel launches
```

## 3. 使用方式

### 3.1 基本用法

```python
import torch

# 方式 1: 装饰器
@torch.compile
def train_step(model, x, y):
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    return loss

# 方式 2: 包装整个模型
model = torch.compile(model)

# 方式 3: 只编译部分代码
class MyModel(torch.nn.Module):
    @torch.compile
    def forward_attention(self, x):
        # 只编译 attention 部分
        ...

    def forward(self, x):
        x = self.forward_attention(x)
        return self.head(x)
```

### 3.2 编译模式

```python
# 默认模式 — 平衡编译时间和性能
model = torch.compile(model, mode="default")

# 推理优化 — 最小化 Python 开销
model = torch.compile(model, mode="reduce-overhead")

# 最大性能 — 长时间 autotune
model = torch.compile(model, mode="max-autotune")

# 完整图 — 不允许 graph break
model = torch.compile(model, fullgraph=True)

# 动态形状 — 支持可变输入尺寸
model = torch.compile(model, dynamic=True)
```

| 模式 | 编译时间 | 运行性能 | 适用场景 |
|------|---------|---------|---------|
| `default` | 中等 | 好的训练性能 | 通用训练 |
| `reduce-overhead` | 中等 | 最优推理 | 推理 (小 batch) |
| `max-autotune` | 很长 | 最优性能 | 生产部署 (一次性编译) |
| `fullgraph=True` | 取决于模型 | 最好 (如果成功) | 简单模型 |

### 3.3 推理 vs 训练配置

```python
# 推理最佳实践
model.eval()
compiled = torch.compile(
    model,
    mode="reduce-overhead",
    fullgraph=True,  # 推理时通常可以完整捕获
)

# 训练最佳实践
model.train()
compiled = torch.compile(
    model,
    mode="default",
    dynamic=True,  # 训练时 batch/seq 可能变化
)

# 注意: 第一次调用会触发编译 (JIT)
# 后续相同 shape 的调用使用缓存
output = compiled(input_tensor)  # 第一次: 编译 + 执行
output = compiled(input_tensor)  # 第二次: 直接执行 (缓存命中)
```

## 4. Graph Breaks

### 4.1 什么是 Graph Break

```
当 Dynamo 遇到无法追踪的操作时:
  → 中断当前图的构建
  → 回退到 eager 模式执行
  → 之后尝试恢复图捕获

影响:
  1. 性能下降: break 前后的图无法跨 break 优化
  2. 额外开销: eager ↔ compiled 切换有 data movement
  3. 多次编译: 不同的 graph segment 分别编译

示例:
@torch.compile
def forward(x):
    y = x * 2          # Graph 1
    if y.sum() > 0:    # ← Graph Break! (数据依赖控制流)
        z = y + 1      # Graph 2 (可能不执行)
    else:
        z = y - 1      # Graph 2 (另一个版本)
    return z
```

### 4.2 常见 Graph Break 原因

```
1. 数据依赖控制流:
   if tensor.item() > 0:    # ❌ 值在运行时才确定
   while tensor.sum() > 0:  # ❌

2. 不支持的 Python 操作:
   print(tensor)            # ❌ side effect
   tensor.numpy()           # ❌ 跨框架
   id(tensor)               # ❌ Python 内置函数

3. 动态形状操作:
   tensor = tensor[:n]      # ❌ n 是运行时变量
   torch.nonzero()          # ❌ 输出大小不确定

4. 第三方库调用:
   numpy_op(tensor)         # ❌
   scipy_func(tensor)       # ❌

解决方法:
   torch.where(condition, x, y)  # 替代 if/else
   torch.cond(condition, fn1, fn2)  # 条件执行
```

### 4.3 调试工具

```bash
# 查看 graph breaks
TORCH_LOGS=graph_breaks python train.py

# 查看动态形状 guards
TORCH_LOGS=+dynamic python train.py

# 查看生成的 Triton 代码
TORCH_LOGS=output_code python train.py

# 详细编译日志
TORCH_LOGS="+" python train.py

# 使用 torch._dynamo.explain 分析
python -c "
import torch
@torch.compile
def fn(x):
    return x + 1
torch._dynamo.explain(fn)(torch.randn(10))
"
```

## 5. CUDA Graphs 集成

### 5.1 工作原理

```
torch.compile 推理模式的优化链:
  PyTorch Model → Dynamo → FX Graph → Inductor → Triton Kernels
                                                          │
                                                          ▼
                                                    CUDA Graph 录制
                                                          │
                                                          ▼
                                                    CUDA Graph 重放

CUDA Graph 的作用:
  1. 录制所有 kernel launch 到一个 graph
  2. 重放 graph 时跳过 CPU 端的所有开销
  3. kernel launch 9us × 10 kernels → 0.1ms 单次重放
```

### 5.2 vLLM 中的 CUDA Graphs

```
vLLM V1 使用 Piecewise CUDA Graphs:

策略:
  1. 在 attention 操作处将图分段
  2. 每段独立录制 CUDA Graph
  3. 运行时动态组合各段

分段原因:
  - Attention 的 seq_len 和 KV cache 大小动态变化
  - 无法在编译时确定所有形状
  - 分段录制允许每段有固定的输入形状

配置模式:
  FULL_DECODE_ONLY  — 只对 decode 阶段录制
  PIECEWISE         — 分段录制 (更灵活)
  FULL_AND_PIECEWISE — 两者都用 (默认)
```

## 6. 实际应用

### 6.1 vLLM 中的使用

```
vLLM 使用 torch.compile 的场景:
  1. 自定义模型前向传播编译
  2. MoE 模型的 router + expert 计算
  3. Speculative Decoding 的 draft model
  4. CUDA Graph 辅助的推理优化

vLLM 特有优化:
  - TorchCompileWithNoGuardsWrapper: 跳过 shape guards 减少重编译
  - 动态形状模式:
    BACKED: 默认，可能不安全地跳过 guards
    UNBACKED: 最保守，保证正确性
    BACKED_SIZE_OBLIVIOUS: 实验性平衡
  - 编译缓存: ~/.cache/vllm/torch_compile_cache/
```

### 6.2 PyTorch Inductor 生成的 Triton Kernel 示例

```python
# Inductor 自动生成的融合 kernel (简化版)
# 原始 PyTorch 代码:
#   output = torch.relu(x * w + b)

# Inductor 生成的 Triton kernel:
@triton.jit
def kernel_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)  # x
    tmp1 = tl.load(in_ptr1 + x0, xmask)  # w
    tmp2 = tl.load(in_ptr2 + x0, xmask)  # b
    tmp3 = tmp0 * tmp1                     # x * w
    tmp4 = tmp3 + tmp2                     # + b
    tmp5 = 0.0
    tmp6 = tl.where(tmp4 < tmp5, tmp5, tmp4)  # relu
    tl.store(out_ptr0 + x0, tmp6, xmask)       # output
```

## 7. 性能对比

```
典型加速效果 (A100, 训练):

模型                  Eager    Compiled    加速
───────────────────────────────────────────────
ResNet-50            3.1ms     2.2ms      1.4x
BERT-Large           5.8ms     3.9ms      1.5x
GPT-2 Small          4.2ms     2.8ms      1.5x
LLaMA-7B (BF16)      12.3ms    8.7ms      1.4x

推理加速效果 (batch=1, reduce-overhead):

模型                  Eager    Compiled    加速
───────────────────────────────────────────────
GPT-2 decode step    1.2ms     0.5ms      2.4x
BERT forward         2.8ms     1.1ms      2.5x

加速来源:
  1. 算子融合: 减少 kernel launch (27 → 7)
  2. 内存优化: 中间结果留在 SRAM
  3. CUDA Graphs: 消除 CPU 开销
  4. Tensor Core: tl.dot() 自动利用
```

## 8. 常见陷阱

```
1. 编译时间过长
   - 第一次编译可能需要几分钟
   - 解决: 使用编译缓存，生产环境预热

2. Graph break 导致性能退化
   - 不知不觉的 graph break 使 compile 无效
   - 解决: TORCH_LOGS=graph_breaks 检查

3. 动态形状导致反复重编译
   - 每个 unique shape 都触发一次编译
   - 解决: 使用 dynamic=True 或固定 batch size

4. 内存使用增加
   - CUDA Graphs 需要预分配内存
   - 解决: 调整 pool 大小或使用 piecewise

5. 调试困难
   - 编译后的 stack trace 不直观
   - 解决: TORCH_LOGS="+dynamo" 查看
```

## 9. 与其他方案对比

| 方案 | 开发成本 | 性能 | 灵活性 | 适用场景 |
|------|---------|------|--------|---------|
| Eager | 最低 | 基线 | 最高 | 原型开发、调试 |
| torch.compile | 极低 (1 行) | 1.5-2.5x | 高 | 通用优化 |
| torch.jit.script | 中等 | 1.2-1.5x | 中 | 已被 compile 取代 |
| 手写 Triton kernel | 高 | 最优 (2-5x) | 低 | 关键热点优化 |
| 手写 CUDA kernel | 很高 | 最优 (2-5x) | 低 | 极致性能需求 |
| TensorRT | 中等 | 2-4x | 低 | 固定模型部署 |

## 参考

- [PyTorch 2.0 Blog](https://pytorch.org/get-started/pytorch-2.0/)
- [torch.compile 文档](https://pytorch.org/docs/stable/torch.compiler.html)
- [Triton: An Intermediate Language and Compiler for Neural Networks](https://openai.com/index/triton/)
- [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
