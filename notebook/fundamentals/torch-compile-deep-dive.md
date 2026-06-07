# torch.compile Deep Dive — Dynamo + AOTAutograd + Inductor

> 2026-06-07 | PyTorch 2.x编译器三层架构 → 8-10x小模型加速 + 1.5-2x大模型加速

## 架构概览

```
Python Code
  ↓
[Dynamo] — Python字节码级图捕获(PEP 523)
  ↓ FX Graph (Python IR)
[AOTAutograd] — 前向+反向图分离
  ↓ Forward FX Graph + Backward FX Graph
[Inductor] — 图降低 → Triton GPU kernel / C++ CPU kernel
  ↓ Compiled Kernel
GPU Execution

→ 三层编译: 捕获→分离→降低 → 每层独立优化
→ 关键: graph break(不支持的Python代码) → 回退eager模式 → 性能损失
```

## Dynamo — Python字节码级图捕获

```
Dynamo工作原理:
1. 使用PEP 523(frame evaluation)拦截Python函数调用
2. 逐字节码分析 → 识别torch操作 → 构建FX Graph
3. 遇到不支持的Python代码 → **graph break** → 分段编译

Dynamo处理的操作:
  ✓ torch.* 操作 (matmul, softmax, etc.)
  ✓ Python数学运算 (简单的算术)
  ✓ Tensor视图操作 (.view, .reshape, .permute)
  ✓ 条件(基于常量/shape, 不基于data)
  ✗ 数据依赖分支 (if tensor.item() > threshold)
  ✗ Python side effects (print, logging, file I/O)
  ✗ 不支持的Python函数 (numpy, random, etc.)
  ✗ Dynamic shapes in某些op

→ 每个**graph break** = 一次eager回退 → 启动新编译区域
→ 大量graph breaks → 编译收益被回退稀释 → 性能退化

Dynamo调试:
  torch._dynamo.explain(model, *args) → 报告graph break位置+原因
  TORCH_LOGS="+dynamo" → 详细trace日志
  torch._dynamo.allow_in_graph(fn) → 强制函数入图
  torch._dynamo.disable() → 跳过问题区域
```

## AOTAutograd — 前向+反向图分离

```
AOTAutograd工作原理:
1. 使用PyTorch dispatch mechanism → 拦autograd dispatch key
2. 执行前向 → 收集所有操作 → 构建Forward FX Graph
3. 反向传播 → 构建Backward FX Graph
4. **Ahead-of-Time** → 不需要实时grad跟踪 → 更高效

关键优势:
  → 前向+反向都编译到Inductor → GPU kernel融合跨越fwd/bwd边界
  → 传统eager: fwd→CUDA→bwd→CUDA → 两次GPU启动
  → AOT编译: fwd+bwd合并→一次GPU启动 → 消除launch overhead

AOTAutograd特殊处理:
  ✓ View操作 → 保持view关系 → 反向正确传播
  ✓ In-place mutation → 拷贝+替换 → 保证正确性
  ✓ Custom autograd函数 → 需要注册到AOT系统
  ✗ 某些复杂view链 → 可能触发graph break

→ AOTAutograd是torch.compile的核心创新:
  之前: 反向=Python循环+逐op autograd → 慢
  现在: 反向=编译Triton kernel → 快 → 8-10x加速(小模型)
```

## Inductor — Triton/C++ Backend编译器

```
Inductor工作原理:
1. 接收Forward + Backward FX Graphs
2. Scheduling算法: 基于memory+fusion启发式 → 分组操作
3. 降低: FX Graph → Triton IR → Triton GPU kernel (GPU)
          FX Graph → C++ IR → OpenMP CPU kernel (CPU)

关键优化:
  ✓ **Kernel fusion**: 多个op合并到单个Triton kernel → 减少launch
  ✓ **Memory planning**: 分析liveness → 优化中间tensor存储
  ✓ **Loop splitting**: 大计算 → 分tile → GPU并行
  ✓ **Recomputation vs materialization**: 依赖memory pressure → 选择重计算或存储

Triton kernel生成:
  → Inductor为每个融合组生成Triton Python kernel
  → Triton编译到PTX → GPU执行
  → 关键: Triton是Python-level → 调试友好(vs CUDA C++)
  → 但: Triton不是最优 → 手写CUDA仍可更快(1.7x实测)

Inductor调试:
  TORCH_LOGS="+inductor" → 降低+codegen日志
  torch.compiler.config.trace_graph_tile = True → 可视化编译区域
```

## 性能实测

### RTX 4090 已有数据

```
| 模型/场景       | torch.compile | Eager | 加速   | 备注
| OPT-125M B=1   | 2.43x         | 1x    | 2.43x | CUDA Graph主导
| 7B B=1         | 1.05x         | 1x    | 1.05x | compute主导,编译收益小
| RMSNorm fwd    | ~1.0x         | 1x    | ~1.0x | 已是fused kernel
| RMSNorm fwd+bwd| ~1.3x         | 1x    | ~1.3x | 融合fwd+bwd
| BPE tokenizer  | —             | —     | —     | Python-level, compile无效

→ 小模型(125M) → launch占比大 → compile消除84% → 2.43x
→ 大模型(7B) → compute占比大 → compile收益小 → 1.05x
→ 关键: compile收益 ∝ 1 - (compute_time / total_time)
→ 模型越大 → compute占比越高 → compile收益越低 → 但仍减少jitter
```

### MiniGQATransformer RTX 4090 Benchmark (2026-06-07)

```
模型: MiniGQATransformer 2.28M params (H=256, L=4, heads=8, kv=4)
PyTorch: 2.9.0+cu128, CUDA: 12.8, GPU: RTX 4090 (24GB)

实验1: Graph Break Analysis
  torch._dynamo.explain → **0 graph breaks** ← MiniGQA完全Dynamo-friendly!
  原因: 纯torch ops, 无数据依赖分支, 无Python side effects
  Eager: 2.07ms → Compile: 1.49ms → 1.40x

实验2: Compile Modes Comparison (Forward-only)
  Eager baseline: 2.04ms
  | Mode             | Median(ms) | Speedup | 备注
  | default          | 1.51       | 1.35x   | 标准编译+kernel fusion
  | reduce-overhead  | 0.54       | **3.75x** | CUDA Graph模式→消除launch
  | max-autotune     | 0.55       | 3.70x   | 自动调优→与reduce-overhead接近

  → reduce-overhead = CUDA Graph → 将整个计算图录制为单个GPU操作 → 消除所有kernel launch
  → 3.75x vs 1.35x → CUDA Graph贡献2.8x额外加速!
  → max-autotune几乎等于reduce-overhead → 2.28M模型已充分优化

实验3: Training Step (Fwd+Bwd) Compile vs Eager
  Eager train step: 7.85ms
  Compile train step: 5.99ms → **1.31x**
  → 训练加速低于推理(1.31x vs 3.75x)
  → 原因: backward占比更大→compute-bound→compile收益被稀释
  → backward=55% of training time → compute主导→launch占比低

实验4: Batch Size × Compile Speedup
  | Batch | Eager(ms) | Compile(ms) | Speedup
  | 1     | 2.15      | 1.55        | 1.38x
  | 4     | 2.05      | 1.48        | 1.39x
  | 16    | 1.92      | 1.46        | 1.31x
  | 32    | 2.02      | 1.45        | 1.39x
  | 64    | 1.95      | 1.43        | 1.36x

  → Speedup **flat** across batch sizes (1.31-1.39x)!
  → 原因: 2.28M模型太小 → 所有batch都memory-bound → HBM带宽饱和
  → batch增大→eager也很快→compile相对收益不明显

关键发现:
  1. **0 graph breaks** → MiniGQA架构Dynamo友好 → 无eager回退
  2. **reduce-overhead 3.75x** → CUDA Graph对小模型推理巨大收益
  3. **训练1.31x** → backward compute-bound → compile收益被稀释
  4. **cuBLAS beats Triton** → Inductor自动选择cuBLAS作为最优kernel
     → RTX 4090 cuBLAS高度优化(FP16 101% peak TFLOPS)
     → Triton kernel仅在fusion场景有优势(cuBLAS不能融合的op组合)
  5. **flat batch scaling** → 小模型memory-bound → batch不改变瓶颈性质

工具: tools/torch_compile_benchmark_4090.py
```

## FSDP2 + torch.compile 组合

```
2025年最重要的训练优化组合:

FSDP2 + torch.compile:
  1. FSDP2分片 → 每GPU只存自己负责的参数片段
  2. torch.compile → 编译分片计算图 → 减少Python overhead
  3. 组合效果: 内存↓ + 吞吐↑ → 1.2-1.5x over FSDP1

关键交互:
  FSDP2 all-gather → compile将all-gather+compute融合 → 减少通信回合
  FSDP2 reduce-scatter → compile融合反向gradient sync → 隐藏通信
  → FSDP2+compile = 通信-计算重叠 + kernel fusion = 双重优化

Benchmark:
  FSDP1 eager:       100% baseline
  FSDP1 + compile:   ~110-120% (10-20%提升)
  FSDP2 eager:       ~115-125% (15-25%提升,sharding改进)
  FSDP2 + compile:   ~120-150% (20-50%提升, 最佳组合)
  Megatron-LM:       ~130-160% (30-60%, 手动kernel+PP)

→ FSDP2+compile达到85-95% Megatron-LM吞吐 → 差距缩小!
```

## 实操技能清单

```
torch.compile调试技能(P0):

1. torch._dynamo.explain(model, *args)
   → 识别graph break → 理解为什么某些op不能编译
   → 输出: graph break数量 + 位置 + 原因

2. TORCH_LOGS环境变量
   → TORCH_LOGS="+dynamo" → trace过程
   → TORCH_LOGS="+aot" → AOT dispatch
   → TORCH_LOGS="+inductor" → kernel生成

3. torch._dynamo.allow_in_graph(fn)
   → 注册自定义函数 → 不触发graph break
   → 适用于: 简单数学计算+tensor操作

4. torch._dynamo.disable()
   → 标记不支持区域 → 整块跳过编译
   → 适用于: 复杂Python逻辑+side effects

5. torch.compile(model, mode="reduce-overhead")
   → 专用CUDA Graph模式 → 进一步减少launch overhead
   → 适用于: 小batch推理→推理延迟敏感场景

6. torch.compile(model, mode="max-autotune")
   → 最大自动调优 → 多kernel配置benchmark → 选最优
   → 适用于: 训练→吞吐优先场景

→ AI Infra工程师核心竞争力:
  识别graph break → 修复 → 编译 → benchmark → 超越eager性能
```

## 参考资料

- PyTorch torch.compile documentation (pytorch.org)
- PyTorch Blog: "Deep Dive into torch.compile" series (2024-2025)
- torch._dynamo.explain API docs
- FSDP2 RFC & blog posts
- Triton compiler documentation (openai.com/triton)