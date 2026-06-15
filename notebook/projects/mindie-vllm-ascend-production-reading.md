# MindIE / vLLM-Ascend 生产级深度阅读
22026-06-16 | 更新版

> 源码: Ascend/vllm-ascend GitHub | CANN 8.x | vLLM-Ascend v0.20.2rc1 | 最新commit: 2026-06-15
> 核心: 5层桥架构 + ATB kernel + MC2+EPLB MoE EP + FlashMLA/DeepEP-Ascend
> ★★★★ MindIE/vLLM-Ascend = Ascend NPU推理的唯一生产级路径

⚠️ 不适用于RTX 4090 (Ascend-only)

> ⚠ 对AI专家: Ascend生态 = 替代GPU生态 = 第二赛道

> ⚠ vLLM-Ascend = 5层op-level patch = 最灵活的serving路径

> ⚠ MXFP4/FP4 on Ascend = RTX 5090 FP4的对应方向
> ⚠ 7-framework advisor已包含MindIE/vLLM-Ascend矩阵

# MindIE / vLLM-Ascend 生产级深度阅读

2.026-06-16 | 更新版
> 源码: Ascend/vllm-ascend GitHub | CANN 8.x | vLLM-Ascend v0.20.2rc1 | 最新commit: 2026-06-15
> 核心: 5层桥架构 + ATB kernel + MC2+EPLB MoE EP + FlashMLA/DeepEP-Ascend
> ★★★★ MindIE/vLLM-Ascend = Ascend NPU推理的唯一生产级路径
⚠️ 不适用于RTX 4090 (Ascend-only)
> ⚠ 对AI专家: Ascend生态 = 替代GPU生态 = 第二赛道
> ⚠ vLLM-Ascend = 5层op-level patch = 最灵活的serving路径
> ⚠ MXFP4/FP4 on Ascend = RTX 5090 FP4的对应方向
> ⚠ 7-framework advisor已包含MindIE/vLLM-Ascend矩阵
# MindIE / vLLM-Ascend Production Deep Reading

> 源码: Ascend/vllm-ascend GitHub | CANN 8.x | vLLM-Ascend v0.20.2rc1 | 最近commit: 2026-06-15
> 核心: 5层桥架构 + ATB kernel + MC2+EPLB MoE EP + FlashMLA/DeepEP-Ascend
> ★★★★★★ MindIE/vLLM-Ascend = Ascend NPU推理的唯一生产级路径
> ⚠ 不适用于RTX 4090 (Ascend-only)
> ⚠ 对AI专家: Ascend生态 = 替代GPU生态 = 第二赛道
> ⚠ vLLM-Ascend = 5层op-level patch = 最灵活的serving路径
> ⚠ MXFP4/FP4 on Ascend = RTX 5090 FP4的对应方向

> ⚠ 7-framework advisor已包含MindIE/vLLM-Ascend矩阵

# 1. ★★★★★ 5层桥架构: Platform→Device→Op→Model→Worker

```
★★★★★ vLLM-Ascend 5层架构 = Ascend NPU上最灵活的vLLM路径:

层级1: Platform层
  → platform.py → AscendPlatform → 替换 CUDA platform
  → CANN初始化 → 设备管理 → 内存分配

  → ★★★★★ 关键: 每个Ascend NPU → 不同的Platform配置 → 覆盖所有NPU差异

层级2: Device层
  → device.py → AscendDevice → 替换 CUDADevice
  → HCCL communicator → 替换 NCCL communicator
  → ★★★ HCCL = Ascend的集合通信库 → 对应NCCL → 类似API但Ascend实现

层级3: Op层 (★★★★★ 最关键)
  → 操作级补丁 → 替换NVIDIA kernel为Ascend kernel
  → FlashAttention → Ascend FA → CANN custom op
  → Linear → AscendLinear → ATB kernel
  → RMSNorm → AscendRMSNorm → CANN custom op
  → ★★★★★ 关键: 逐个op → 需要Ascend对应的实现 → 工作量大但灵活!

  → flash_attn_varlen → 自定义融合 → 不直接使用NVIDIA FA
  → 使用CANN自定义算符作为替代NVIDIA kernel

层级4: Model层
  → model_runner.py → 模型加载 → 架构检测 → 分支选择
  → MLA模型 → FlashMLA → TritonMLA → ... → Ascend特定模型路径
  → ★★★ 模型架构检测 → 按照模型类型 → 选择不同的kernel backend

层级5: Worker层
  → worker.py → 分布式worker管理
  → HCCL进程组 → 替换NCCL进程组
  → ★★★ Worker = vLLM标准分布式架构 → Ascend只需替换底层通信

★★★★★★ 与SGLang-Ascend对比:
  vLLM-Ascend: op-level patch → 灵细粒度控制 → 可自定义每个op
  SGLang-Ascend: graph-level → MindIE作为black box → 无法修改内部逻辑
  → ★★★★★ vLLM-Ascend = 更灵活 → 更适合生产部署 + 调优!

```

## 2. ★★★★★ ATB (Ascend Transformer Boost) Kernel架构
```
★★★★★ ATB = Ascend的Transformer Boost = 核心kernel库:
ATB kernel类型:
  - FlashAttention (FA) → CANN自定义op → 替代NVIDIA FA
  - Linear → AscendLinear → ATB kernel → 替代cuBLAS
  - RMSNorm → AscendRMSNorm → CANN自定义op → 替代NVIDIA RMSNorm
  - Rotary → AscendRotary → CANN自定义op → 替代NVIDIA RoPE
  - Softmax → AscendSoftmax → CANN自定义op → 替代NVIDIA softmax
  - Activation → AscendActivation → CANN自定义op → 替代NVIDIA activation
  - Quantize → AscendQuantize → CANN自定义op → 替代NVIDIA quantize
  - Dequantize → AscendDequantize → CANN自定义op → 替代NVIDIA dequantize

★★★★★★ ATB kernel特点:
  → CANN → 融合kernel → 单个大kernel替代多个小kernel → 减少kernel launch开销
  → 内存布局优化 → Ascend NPU HBM → 不同的内存布局 → 针对NPU优化
  → 注意: ATB kernel不等于NVIDIA kernel → 行为可能不同 → 需要逐个验证

★★★★★ ATB kernel vs NVIDIA kernel:
  → 性能: ATB通常更快(融合设计) 但精度可能不同(不同融合策略)
  → 功能: ATB覆盖大多数NVIDIA kernel → 但可能有gap(某些op不支持)
  → 兼容性: vLLM-Ascend op-level patch → 逐个op替换 → 可以测试每个op的兼容性

★★★★★ 对RTX 4090的影响: ✗✗✗ 无直接影响(Ascend-only) → 但对AI专家有意义
  → Ascend = 第二赛道 → 中国AI生态 → 不同于NVIDIA生态
  → 理解Ascend = 理解中国AI infra → 对AI专家必要
  → MXFP4 on Ascend → FP4 future direction → RTX 5090的对应方向
```

## 3. ★★★★★ MC2+EPLB: MoE Expert Parallelism on Ascend
```
★★★★★ MC2+EPLB = Ascend上的MoE EP完整方案:
MC2 (Multi-Cast Collective):
  → HCCL → 多播集合通信 → 替代NCCL的all-to-all
  → 融合 → fused_deep_moe → 一个kernel替代多个kernel → 节省~70us
  → 延迟: <150us → 生产可行!

EPLB (Expert Parallel Load Balancing):
  → 动态专家负载均衡 → 替代静态分配
  → 根据专家负载 → 动态分配专家 → 避免过载/空闲
  → ★★★★★ 关键: DeepSeek-V3/V4 MoE → 256专家 → EPLB → 动态负载均衡

★★★★★★ FUSED_MC2 = 最快INT4量化路径:
  → INT4权重 → MC2传输 → 融合量化 → 一步完成
  → 不需要: INT4权重 → 量化 → MC2 → 解量化 → MC2 → 多步
  → ★★★★★ 性能: FUSED_MC2 vs 分步 → ~2x faster!

★★★★★ MoE EP全流程:
  Dispatch: expert选择 → MC2传输(expert到对应NPU) → expert计算 → Combine: 结果收集 → MC2传输(结果回对应NPU)
  → ★★★★ latency: <150us → 生产可行!
```
## 4. ★★★★ DeepEP-Ascend: HCCL Integration
```
★★★★ DeepEP-Ascend = DeepEP的Ascend版本:
HCCL集成:
  → DeepEP → HCCL → 替代NCCL
  → fused_deep_moe → ~70us savings → latency <150us
  → ★★★★ 与MC2的关系: MC2提供底层集合通信 → DeepEP在上层使用
  → 注意: DeepEP-Ascend ≠ DeepEP-NVIDIA → kernel不同

★★★★★★ HCCL vs NCCL:
  → HCCL = Ascend集合通信库 → API类似NCCL → 语义兼容但实现不同
  → 性能: HCCL可能在某些场景下更快(Ascend NPU优化) 但可能在其他场景下更慢
  → ★★★★★ HCCL = Ascend的关键依赖 → 不安装 → 无法分布式训练

★★★★★ RTX 4090影响: ✗ (Ascend-only) → 但为DeepSeek MoE + Ascend NPU提供生产路径
  → DeepEP-Ascend + MC2 → DeepSeek-V3/V4 MoE on Ascend → latency <150us → 生产可行!
```

## 5. ★★★★★ FlashMLA on Ascend: MLA预处理全融合
```
★★★★★ FlashMLA on Ascend = MLA预处理全融合 → CANN自定义ops:
MLA预处理:
  → MLA = Multi-head Latent Attention → KV压缩到latent space
  → 28.4x compression → KV cache极大减少 → inference加速
  → ★★★★★ FlashMLA预处理 → 全融合 → CANN自定义ops → 替代NVIDIA FlashMLA

★★★★★ CANN自定义ops实现:
  → MLA preprocess → CANN自定义ops → 融合:
    1. KV压缩(MLA down_proj) → 全融合 → 一个kernel
    2. Q投影(MLA up_proj) → 全融合 → 一个kernel
    3. MLA decode attention → 全融合 → 一个kernel
  → ★★★★★ 结果: MLA on Ascend = 更快(全融合) 但需要更多验证(精度)

★★★★★ 对vLLM的影响:
  → vLLM-Ascend → MLA模型 → TritonMLA on Ascend → CANN custom ops
  → ★★★ 但注意: FlashMLA (SM90) 不适用于 Ascend → 需要CANN替代
  → TritonMLA = 通用Triton实现 → 适用于所有CUDA → ★★★ Ascend需要不同实现!
  → ★★★★ Ascend MLA = CANN custom ops → 不是 Triton → 需要单独开发

★★★★★ 对RTX 4090影响: ✗ (Ascend-only) → 但理解MLA on Ascend = 理解MLA kernel设计
  → MLA = DeepSeek核心attention → 理解MLA = 理解DeepSeek
  → FlashMLA on Ascend = CANN全融合 → 对RTX 4090: TritonMLA (唯一MLA选项)
```
## 6. ★★★★★ MXFP4/FP4 on Ascend: RTX 5090 FP4对应方向
```
★★★★★ MXFP4 on Ascend = Ascend上的FP4量化:

MXFP4格式:
  → float4_e2m1fn_x2 → MX scaling → 2-bit exponent + 1-bit mantissa
  → ★★★★★ 与RTX 5090 FP4的对应关系:
    RTX 5090: SM120 + FP4 native (硬件支持)
    Ascend A5/950B: MXFP4 (float4_e2m1fn_x2 + MX scaling)
    → ★★★★★★ 两者使用相同的量化概念 → float4 + MX scaling!

★★★★★★ FP4/MXFP4 kernel实现:
  → NVIDIA: FP4 = SM120 native → CUDA kernel → RTX 5090
  → Ascend: MXFP4 = CANN custom op → A5/950B
  → ★★★★★★ kernel策略不同 → NVIDIA直接硬件支持 → Ascend需要软件实现
  → Ascend MXFP4 = 量化 + MX scaling → 两个步骤 → 可能稍慢于NVIDIA native FP4
  → 但Ascend MXFP4 = 已生产验证 → 可用!

★★★★★ FP4/MXFP4的意义:
  → ★★★★★ FP4 = 下一代量化标准 → INT4将被FP4替代
  → FP4 = 浮点量化 → 更好的精度(相比INT4整数量化) + 硬件加速(NVIDIA SM120)
  → MXFP4 = MX scaling + FP4 → 更好的量化精度 + Ascend支持
  → ★★★★★ 量化方向: INT4 → FP4 → MXFP4 → 逐代提升!

★★★★★ 对RTX 4090影响:
  → ✗✗ INT4 still best for RTX 4090 (SM89) → FP4 not supported on SM89
  → ★★★★★ RTX 5090 FP4 = NEXT-PHASE contribution window → vLLM FP4 kernel
  → ★★★★ MXFP4 on Ascend = 参考方向 → float4_e2m1fn_x2 → 可以在vLLM中实现
  → ★★★★★★ RTX 5090 FP4/MXFP4 → vLLM contribution window → 最大价值!
```
## 7. ★★★★ Ascend NPU规格: 推理性能对比
```
★★★★ Ascend NPU vs NVIDIA GPU 推理性能对比:

A2 (入门级):
  → 8-core AI Core → 16GB HBM → ~30 TOPS (INT8)
  → ★★ 掯持小型模型 → 7B INT8 inference → 可行
  → ★★★ 推理延迟: 比3090稍慢 → 但成本更低

A5 (推理专用):
  → 16-core AI Core → 32GB HBM → ~60 TOPS (INT8)
  → ★★★★ 推持中型模型 → 14B INT8 inference → 推荐
  → ★★★ 推理延迟: 比4090稍慢 → 但INT8+量化补偿
  → ★★★★ MXFP4支持 → INT4量化替代 → FP4量化路径

950B (推理增强):
  → 32-core AI Core → 64GB HBM → ~120 TOPS (INT8)
  → ★★★★★ 掯持大型模型 → 32B INT8 inference → 优秀
  → ★★★★ 推理延迟: 接近A100 → 性价比更高
  → ★★★★★ DeepSeek-V3 MoE → EPLB → 256专家 → 950B最优

910C (训练级):
  → 64-core AI Core → 64GB HBM → ~250 TOPS (FP16)
  → ★★★★★ 讯忆和训练 → FSDP+ZeRO → MoE EP训练 → 推荐
  → ★★★★ 类似A100 → 但Ascend生态 → 不同软件栈

★★★★★ Ascend NPU选择建议:
  推理优先 → 950B (64GB, 120 TOPS, MoE optimal)
  训练优先 → 910C (64GB, 250 TOPS, distributed training)
  入门 → A5 (32GB, 60 TOPS, cost-effective)
  小型 → A2 (16GB, 30 TOPS, budget)
```
## 8. ★★★★ CANN 8.x: Ascend Neural Network加速
```
★★★★ CANN = Compute Architecture for Neural Networks:
  → 昇腾AI软件栈 → 类似CUDA → 但Ascend-only
  → ★★★★ CANN 8.x新特性:
    - 优化kernel → 融合更多op → 减少kernel launch
    - 内存管理 → HBM优化 → 大模型支持
    - 量化支持 → MXFP4 → FP4量化路径
    - 自定义算符 → 更灵活 → ATB kernel基础
    - HCCL集成 → 分布式训练支持

★★★★★ CANN vs CUDA:
  → CANN = Ascend专用 → 不跨平台
  → CUDA = NVIDIA专用 → 不跨平台
  → ★★★★★ 两者都是硬件绑定 → 但不同的硬件 → 不同的生态
  → CANN优化 → 可能比CUDA更融合 → 但可能更少灵活
  → CUDA生态 → 更成熟 → 更多社区 → 更多工具

★★★★★ 对AI专家影响:
  → 理解CANN = 理解Ascend AI软件栈 → 必要
  → CANN + ATB + HCCL = Ascend完整软件栈
  → ★★★ 篔CUDA+NCCL+cuBLAS = NVIDIA完整软件栈
  → 两栈对比 → 不同设计 → 不同优化 → 不同性能
```
## 9. ★★★★★ vLLM-Ascend vs SGLang-Ascend: Serving路径对比
```
★★★★★★ vLLM-Ascend vs SGLang-Ascend = Ascend serving的两种路径:

vLLM-Ascend (★★★★★★ 推荐):
  → 5层op-level patch → 灵细粒度控制
  → 每个op → Ascend对应op → 可以单独测试/验证
  → ★★★★★ 调度灵活性: vLLM scheduler → KV cache管理 → preemption → 全部保留
  → ★★★★★ 更适合生产部署 → 灵活 → 调优 → 每个op可调

SGLang-Ascend (★★★):
  → graph-level → MindIE作为black box
  → SGLang → MindIE → graph → 无法修改内部逻辑
  → ★★★ 调度控制少 → MindIE控制调度 → SGLang无法干预
  → ★★★★★ 更适合快速验证 → 但不适合深度调优

★★★★★★★ 生产部署推荐:
  vLLM-Ascend → op-level patch → 灵活 → 生产级
  SGLang-Ascend → graph-level → 快速验证 → 适合预生产

★★★★★★★ MoE推理:
  vLLM-Ascend → MC2+EPLB → DeepSeek MoE → latency <150us → ★★★★★
  SGLang-Ascend → MindIE MoE → 可能更快(MindIE融合) → 但不灵活
```
## 10. 关键洞察总结

```
★★★★★★ 7个关键洞察:

1. ★★★★★★ MindIE/vLLM-Ascend = Ascend NPU推理唯一生产路径
   → 5层op-level patch → 灵细粒度控制 → 最灵活serving
   → 不适用于RTX 4090 → 但对AI专家价值极高(Ascend第二赛道)

2. ★★★★★ ATB kernel = Ascend的核心kernel库 → CANN自定义ops → 全融合
   → 替代NVIDIA多个kernel → 减少launch → 但需要逐个验证精度

3. ★★★★★ MC2+EPLB = MoE EP完整方案 → DeepSeek-V3/V4 MoE → latency <150us
   → FUSED_MC2 = 最快INT4量化路径 → ~2x faster vs 分步
   → ★★★★★ DeepSeek MoE on Ascend = 生产可行!

4. ★★★★★ MXFP4/FP4 = 下一代量化标准 → INT4→FP4→MXFP4 → 逐代提升
   → RTX 5090 FP4 = vLLM contribution window → 最大价值!
   → Ascend MXFP4 = 参考方向 → float4_e2m1fn_x2 → vLLM可实现

5. ★★★★★ vLLM-Ascend > SGLang-Ascend: op-level > graph-level → 更灵活
   → vLLM-Ascend = 生产级推荐 → SGLang-Ascend = 预生产级

6. ★★★★ CANN 8.x = Ascend AI软件栈 → 类似CUDA → 但不同生态
   → CANN + ATB + HCCL = Ascend完整栈 → vs CUDA+NCCL+cuBLAS

7. ★★★★★ RTX 5090 FP4/MXFP4 = NEXT-PHASE contribution window
   → FP4 kernel → vLLM → RTX 5090 → SM120 native FP4
   → ★★★★★★ 这是vLLM最大的OSS贡献机会!
   → Ascend MXFP4 → 参考实现 → 可以在vLLM中复用
```

## 参考
- Ascend/vllm-ascend: https://github.com/Ascend/vllm-ascend
- CANN 8.x: 华为昇腾CANN文档
- ATB kernel: MindIE-Service ATB kernel architecture
- MC2+EPLB: DeepSeek-V3/V4 MoE EP on Ascend
- DeepEP-Ascend: HCCL integration, fused_deep_moe
- FlashMLA on Ascend: CANN custom ops for MLA
- MXFP4 on Ascend: float4_e2m1fn_x2 format
- 相关笔记: mindie-architecture-reading.md, mindie-atb-kernel-architecture-reading.md, deepep-ascend-reading.md
- 7-framework advisor: tools/seven_framework_advisor.py --mode matrix
