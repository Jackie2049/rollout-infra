# PyTorch Compiler Roadmap (2.6→2.8+): torch.compile + Inductor + MaxAutotune

> 2026-06-15 | Roadmap分析 + 动态形状专题 (基于官方博客+dev-discuss+wiki)
> 衔接: notebook/fundamentals/torch-compile-deep-dive.md (架构内部)
> 衔接: notebook/fundamentals/pytorch-internals-dispatcher-customop.md (Dispatcher+custom_op)

## 1. 总体目标

PyTorch Compiler团队的终极目标:
- **torch.compile 应该"零配置"工作** — `torch.compile(model)` 自动适配任何形状
- **性能不因动态形状牺牲** — ≤5%开销 vs 静态形状特化
- **编译开销可控** — MaxAutotune结果可持久缓存

```
2025: 稳定+动态形状基础 → 2026: 零配置+生产级 → 未来: 多后端+全自动
```

## 2. 版本路线图

### 2.1 PyTorch 2.6 (2025 Q1)

| 领域 | 重点 | 状态 |
|------|------|------|
| torch.compile | 稳定性提升, 减少graph breaks | ✅ 已发布 |
| Dynamic Shapes | `mode="dynamic"` 稳定化 | ✅ 基础可用 |
| Inductor | Op覆盖扩展, Triton kernel改进 | ✅ |
| MaxAutotune | 缓存基础设施, 减少开销 | ✅ 初版缓存 |
| Flex Attention | 改进+更多pattern支持 | ✅ |

### 2.2 PyTorch 2.7 (2025 Q2-Q3)

| 领域 | 重点 | 目标 |
|------|------|------|
| torch.compile | **编译速度** — 减少重编译次数 | 2x编译加速 |
| Dynamic Shapes | **自动检测** — 无需mark_dynamic | 零标注 |
| Inductor | **更优融合** — 新后端支持(ROCm) | 扩展后端 |
| MaxAutotune | **快速默认模式** — 持久化缓存 | 缓存跨session |
| 调试工具 | 更好的compile错误诊断 | 开发者友好 |

### 2.3 PyTorch 2.8+ (2025 Q4→2026)

| 领域 | 重点 | 目标 |
|------|------|------|
| torch.compile | **"Just works"** — 任何模型任何形状 | 零配置 |
| Dynamic Shapes | **单次编译所有形状** — 无重编译风暴 | ≤5%开销 |
| Inductor | **多后端生产级** — CUDA/ROCm/CPU/XLA | 全后端 |
| MaxAutotune | **默认启用** — 无感知最优kernel选择 | 零开销 |
| custom_op | **compile完全兼容** — 无graph break | 全覆盖 |

## 3. Dynamic Shapes 深度分析

### 3.1 问题: Recompilation Storm

```
传统torch.compile:
  shape [1, 10] → 编译kernel_1
  shape [1, 20] → 编译kernel_2  ← 重编译!
  shape [1, 30] → 编译kernel_3  ← 重编译!
  → N种shape = N次编译 = 编译风暴!

LLM场景: seq_len变化 → 每个新长度重编译 → 服务启动慢+内存爆炸
```

### 3.2 解决方案: Symbolic Shapes

```
Inductor Symbolic Shapes (PyTorch 2.5+):
  shape [1, s0] → 编译1次symbolic kernel
  s0可以是任意值 → 1次编译覆盖所有形状!

实现:
  1. Dynamo捕获 → 将具体shape替换为符号变量(s0, s1, ...)
  2. Inductor lowering → Triton kernel使用符号变量
  3. runtime → 具体shape值代入 → kernel直接执行

→ 1次编译 = 覆盖所有seq_len → 无重编译!
```

### 3.3 动态形状演进路线

| 阶段 | 特性 | 用户操作 | 性能 |
|------|------|----------|------|
| Phase 1 (2.5-2.6) | Symbolic shapes基础 | `torch.compile(mode="dynamic")` | 10-30%开销 |
| Phase 2 (2.7) | 自动检测+推断 | `torch.compile(model)` 自动 | 5-15%开销 |
| Phase 3 (2.8+) | 零配置+全形状 | 无需任何标注 | ≤5%开销 |

### 3.4 Data-Dependent Shapes (最难!)

```
问题: torch.unique(), torch.nonzero() → 输出shape取决于数据值
  → 无法用symbolic shapes → 必须graph break!

当前处理:
  - PyTorch 2.6: 部分支持, graph break仍常见
  - PyTorch 2.7: 更优雅处理, 最少graph break
  - PyTorch 2.8+: 目标无graph break

LLM影响:
  - Sampling: top_k→不同token数 → data-dependent shape!
  - vLLM/SGLang: 已用bucket化规避 → 固定shape不触发此问题
```

## 4. MaxAutotune 深度分析

### 4.1 工作原理

```
MaxAutotune:
  1. 为每个kernel生成多个配置(tile_size, num_warps, ...)
  2. 在GPU上benchmark每个配置 → 选最快
  3. 缓存最优配置 → 下次编译跳过benchmark

问题:
  - 编译时间: MaxAutotune让编译慢10-100x!
  - 缓存局限: 只缓存到本地 → 新session重新benchmark
  - 环境依赖: 不同GPU最优配置不同 → 缓存不通用
```

### 4.2 MaxAutotune演进路线

| 版本 | 缓存 | 开销 | 用户体验 |
|------|------|------|----------|
| 2.5 | 本地文件缓存 | 高(10-100x编译时间) | 需显式启用 |
| 2.6 | 改进缓存基础设施 | 中(5-20x) | 缓存命中率提升 |
| 2.7 | **持久化跨session** + 快速默认模式 | 低(2-5x首次, <1x后续) | 自动缓存 |
| 2.8+ | **默认启用** — 最优kernel自动选择 | 极低(首次稍慢, 后续零开销) | 无感知 |

### 4.3 实际影响: RTX 4090

```
RTX 4090场景:
  - 首次compile: MaxAutotune耗时30-120秒
  - 缓存后: 0.5-2秒
  - GRPO训练: compile只在第1步慢 → 后续加速
  - 推理服务: 启动时compile → MaxAutotune缓存 → 服务无开销

建议:
  - 训练: mode="reduce-overhead" (不MaxAutotune) → 编译快
  - 推理: mode="max-autotune" (首次慢但最优) → 长期最优
  - 调试: mode="default" → 不优化, 最快编译
```

## 5. Inductor Backend演进

### 5.1 当前架构

```
Inductor Codegen:
  GPU kernel → Triton (Python-like, 自动优化)
  CPU kernel → C++ (手动优化模板)

  Triton优势:
  - Python语法 → 开发快
  - 自动memory coalescing → 性能接近手写CUDA
  - RTX 4090 SM89: Triton kernel 5-9x PyTorch eager
```

### 5.2 Backend扩展

| Backend | 当前 | 2.8+目标 | 说明 |
|---------|------|----------|------|
| CUDA (Triton) | ✅ 生产级 | ✅ | 主力后端 |
| ROCm (Triton) | ⚠️ 改进中 | ✅ | AMD GPU |
| CPU (C++) | ✅ | ✅ 改进 | SIMD优化 |
| XLA | 🔄 实验性 | ⚠️ 部分 | TPU/其他 |
| Metal | 🔄 | ⚠️ | Apple GPU |

## 6. 与7框架交叉影响

| 框架 | torch.compile影响 | 关键 |
|------|-------------------|------|
| DeepSpeed | ZeRO-3 graph break(参数动态) → 需FSDP2 | FSDP2 per-param=compile友好 |
| Megatron-LM | CUDA Graph+TP → compile部分兼容 | Megatron已有专用kernel |
| vLLM | 动态batch→compile困难 → 当前不用 | bucket化后可能compile |
| verl | GRPO训练 → compile 1.5-2x加速 | mode="reduce-overhead" |
| MindIE | 不适用(昇腾NPU) | vLLM-Ascend可能用CANN编译 |
| rLLM | verl backend→compile加速训练 | tinker backend无compile |
| PyTorch | compile是PyTorch自身特性 | custom_op=compile兼容 |

### 6.1 关键发现: FSDP2 + compile = 最佳组合

```
为什么FSDP2与compile兼容:
  - FSDP1: FlatParameter → 整个module视为1个大参数 → compile看到动态形状 → graph break
  - FSDP2: DTensor per-param → 每个参数独立 → compile可以静态特化 → 无graph break!
  - FSDP2 unshard/reshard = explicit aten ops (aten.unshard, aten.shard) → compile可trace!

  DeepSpeed ZeRO-3: 类似FSDP1问题 → 参数gather/partition → 动态 → graph break
  → ZeRO-3 + compile = 不兼容
  → FSDP2 + compile = 兼容

RTX 4090建议:
  训练: FSDP2 + compile("reduce-overhead") + BF16 + LoRA → 最优方案
  推理: 不用compile → vLLM/SGLang有专用kernel
```

### 6.2 实测基准数据 (2025 torchtitan benchmarks, H100)

```
Llama 3.1 8B, 8×H100 80GB:

| 配置 | TPS/GPU | MFU | vs eager |
|------|---------|-----|----------|
| FSDP2 eager | 5,762 | 33% | baseline |
| FSDP2 + compile | 6,667 | 39% | +15.7% |
| FSDP2 + compile + Float8 | 8,532 | — | +48.1% |

Llama 3.1 8B, 128×H100:

| 配置 | TPS/GPU | MFU | vs eager |
|------|---------|-----|----------|
| FSDP2 eager | 5,605 | — | baseline |
| FSDP2 + compile | 6,514 | — | +16.2% |
| FSDP2 + compile + Float8 | 8,380 | — | +49.5% |

关键发现:
  → compile alone: +15-16% throughput → kernel fusion是主要收益
  → compile + Float8: +48-50% → FP8 all-gather减少通信2× → 巨大!
  → 128 GPU scaling: ~90% efficiency → ZeRO-3 ~75-80% → FSDP2显著更优

ZeRO-3 vs FSDP2 量化对比:
  → FSDP2比ZeRO-3快10-25% (同等硬件) → compile差距更大
  → 内存: 基本等价(都是全分片) → 但ZeRO-3有padding浪费 → FSDP2无!
  → ZeRO-3优势: NVMe offload → GPU内存不够时唯一选择
  → FSDP2优势: compile兼容 → 15-50%额外加速 → 决定性优势!
```

### 6.3 FSDP2 reshard_after_forward 参数

```
reshard_after_forward=True (默认):
  → Forward后释放full params → 重新shard → 省GPU内存
  → Backward需要重新AllGather → 更多通信(2Ψ per module)
  → 等价FSDP1 FULL_SHARD → 推荐(默认)
  → 适用: 内存受限 → 大模型训练

reshard_after_forward=False:
  → Forward后保留full params → 不释放 → 更多GPU内存占用
  → Backward不需要重新AllGather → 省通信 → 但peak更高!
  → 等价FSDP1 NO_SHARD_GRAD_ONLY → 内存富裕时可选
  → 适用: 小模型或通信瓶颈场景

HSDP (2D DeviceMesh) 场景:
  → reshard只影响跨节点sharding维度 → intra-node params已replicate
  → 可per-module设置: 大module=True(省内存) + 小module=False(省通信)
```

### 6.4 FSDP2 CPU Offload 状态

```
FSDP2 CPU Offload: 实验性/alpha (nightly PyTorch)

API:
  from torch.distributed._composable.fsdp import CPUOffloadPolicy
  model = FSDP(model, offload_policy=CPUOffloadPolicy())

性能(vs FSDP1 CPUOffload):
  → FSDP2 offload: ~20-30% better throughput than FSDP1 offload
  → 原因: 改进prefetch + pinned memory + async transfers
  → 但: ~50-60% slower than pure GPU FSDP2 → offload仍然代价高!

Timeline:
  → Stable CPU offload: late 2025 / early 2026
  → MixedOffloadPolicy (selective): 2026 planned
  → Optimizer state CPU offload: 2026 planned

vs ZeRO-Offload/ZeRO-Infinity:
  → ZeRO更成熟 → NVMe offload → GPU不够时唯一选择
  → FSDP2 offload还不稳定 → 如果需要CPU offload → ZeRO仍是首选!
  → 但长期: FSDP2 offload + compile → 可能超越ZeRO

RTX 4090:
  → ZeRO-2+CPU_Adam+LoRA → 当前唯一可行CPU offload方案
  → FSDP2 CPU offload → 等稳定后再考虑 → 目前不推荐!
```

### 6.5 PyTorch 2.7 DTensor改进

```
PyTorch 2.7 (2025 Q2-Q3):

ProcessGroupCollection:
  → DeviceMesh新增ProcessGroupCollection accessor
  → 每mesh维度自动映射对应ProcessGroup → 无需手动mapping!
  → Shard(0)自动→TP组, Shard(1)→DP组 → 2D Mesh全自动!
  → dist.new_group() 软deprecation → 迁移开始

2D/3D DeviceMesh:
  → FSDP+TP, FSDP+CP 组合并行 → DeviceMesh表达
  → DTensor→local tensor改进 → migration path更顺畅
  → compile兼容分布式 → torch.compile sees fixed shapes even with DTensor!

PyTorch 2.8 目标:
  → ProcessGroupCollection first-class → new_group() hard deprecation
  → "零配置" → 用户写单设备代码 → compiler+DTensor自动推导最优分布式
  → HSDP和nested parallelism完全支持 → 2D+ DeviceMesh

→ 长期: DTensor将成为PyTorch分布式唯一抽象 → ZeRO-3 ds_tensor将被替代!
```

## 7. 编译模式选择指南

```
torch.compile模式选择决策树:

  是否首次运行?
    ├── 是 → mode="default" (快速编译, 验证正确性)
    │        → 确认无错误后切换
    ├── 否 → 什么场景?
              ├── 训练(反复step) → mode="reduce-overhead"
              │     → CUDA Graph+减少launch开销 → 1.5-2x加速
              ├── 推理(长期服务) → mode="max-autotune"
              │     → 最优kernel选择 → 首次慢但长期快
              ├── 调试(快速验证) → mode="default"
              │     → 最少优化 → 最快编译+最可读错误
              └── 动态形状 → mode="reduce-overhead" + dynamic=True
                    → 2026后: 无需指定 → 自动处理
```

## 8. Inductor 内部流水线 (源码级概览)

> 关键源码路径: torch/_inductor/scheduler.py + codegen/triton.py + lowering.py + ir.py

### 8.1 三阶段流水线

```
Python字节码 → Dynamo → FX Graph → AOTAutograd → Inductor:
  1. Lowering (lowering.py):
     FX ops → Inductor IR (Buffer + Operation objects)
     - PointwiseOp: element-wise操作 (add, mul, relu等)
     - ReductionOp: reduce操作 (sum, max, mean等)
     - Buffer: 中间结果存储单元

  2. Scheduling (scheduler.py):
     Scheduler类 → 3个pass:
     a) Fusion pass — create_fused_groups()
        - 贪心融合启发式: 同dtype+同memory pattern+同tile_size → 1个Triton kernel
        - 融合减少kernel launch → 减少内存读写 → 关键优化!
     b) Memory planning pass — V.graph.memory_pool
        - Buffer生命周期分析 → 确定何时分配/释放/复用
        - extern(输入/输出) vs inline(就地计算) vs realized(显式存储)
        - Buffer复用 → 减少峰值内存
     c) Scheduling pass — DAG launch order
        - 拓扑排序 → 依赖约束 → 确定kernel执行顺序

  3. Codegen (codegen/triton.py):
     Scheduled IR → Triton kernel Python源码 + C++ launch wrapper
     - TritonKernel类: 生成kernel定义
     - TritonCodeGen: 遍历scheduled IR → emit kernel + launch
     - Reduction特殊处理: split为partial-reduce + finalize两步
     - Persistent kernels: softmax/layernorm等维持跨launch状态
```

### 8.2 Triton Kernel代码生成细节

```
典型生成的Triton kernel:

@triton.jit
def fused_kernel(X_ptr, Y_ptr, OUT_ptr, N,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N           ← 动态形状用SymInt!
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    out = x + y                  ← 融合: add不再单独kernel
    tl.store(OUT_ptr + offsets, out, mask=mask)

→ Inductor从IR自动生成 → 无需手写Triton kernel!
→ tile策略: numel / tile_size → grid维度自动计算
→ 动态形状: SymInt(s0, s1) → 运行时mask
```

### 8.3 MaxAutotune在Inductor中的实现

```
MaxAutotune流程:
  1. 为每个fused kernel生成多个config:
     - 不同的BLOCK_SIZE (64, 128, 256, 512, 1024)
     - 不同的num_warps (2, 4, 8)
     - 不同的num_stages (1, 2, 3, 4)
  2. Triton @triton.autotune装饰器 → benchmark每个config
  3. 选最快 → 缓存到本地文件
  4. 下次编译 → 读取缓存 → 跳过benchmark

→ 问题: 首次编译10-100x慢 (大量benchmark)
→ 解决(2.7+): 持久化缓存跨session + 快速默认模式
```

### 8.4 Flex Attention Kernel生成

```
PyTorch 2.5+新特性:
  torch.nn.attention.flex_attention → Inductor生成专用Triton kernel

  不再fallback到cuBLAS → 而是生成融合attention kernel:
  - Q×K → 融合softmax → 融合mask → 融合×V
  - 类似FlashAttention但由Inductor自动生成
  - 适配自定义mask函数 → FlashInfer不支持的场景!

  RTX 4090影响:
  - Flex Attention让compile后的attention接近FA-2性能
  - 但仍不如专用FlashInfer → 推理用vLLM/SGLang更好
  - 训练: compile+flex_attention可能有用 → 待实测
```

## 9. 下一步

- [x] 创建PyTorch compiler roadmap笔记 → 本文件
- [ ] 在GPU可用时实测 torch.compile 各模式性能对比 (default/reduce-overhead/max-autotune)
- [ ] 研究 Inductor Triton kernel codegen细节 → 与手写CUDA对比
- [ ] 实测 FSDP2 + compile vs DeepSpeed ZeRO-3 + eager 在7B训练
- [ ] 研究 MaxAutotune持久缓存机制 (本地文件格式+跨session复用)

---

Sources:
- [PyTorch 2.6 Release Blog](https://pytorch.org/blog/pytorch-2-6/)
- [PT2 Compiler Team Roadmap (Wiki)](https://github.com/pytorch/pytorch/wiki/PT2-Compiler-Team-Roadmap)
- [torch.compile Dynamic Shapes Blog](https://pytorch.org/blog/)
- [Symbolic Shapes in PyTorch Inductor](https://pytorch.org/blog/)
- [MaxAutotune Overhead Discussion](https://dev-discuss.pytorch.org/t/max-autotune-overhead/)
- [PyTorch Dev Discussions - Compiler Team](https://dev-discuss.pytorch.org/)
- [PyTorch Compiler Performance Update](https://pytorch.org/blog/compiler-performance-update/)
