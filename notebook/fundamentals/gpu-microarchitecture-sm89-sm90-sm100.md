# GPU Microarchitecture Deep Dive: SM89/SM90/SM100

> 2026-06-08 | 从warp scheduler到Tensor Core, GPU微架构全面分析
> 基于: NVIDIA Architecture Whitepapers, CUDA Programming Guide, RTX 4090实测
> 关联: cuda-kernel-optimization-sm89.md (SM89 kernel优化), cutlass-3x-gemm-architecture.md (CUTLASS)

## 0. 三代架构总览

| 特性 | Ada SM89 (RTX 4090) | Hopper SM90 (H100) | Blackwell SM100 (B200) |
|------|---------------------|--------------------|-----------------------|
| SM数 | 128 | 132 | 168 |
| CUDA Cores | 16,384 | 16,896 | 21,504 |
| Tensor Core Gen | 4th Gen | 4th Gen(WGMMA) | 5th Gen(Enhanced WGMMA) |
| 精度 | FP16/BF16/FP8/INT8 | FP64/FP32/TF32/FP16/BF16/FP8/INT8 | FP4/FP6/FP8/FP16/BF16/INT4/INT8 |
| Shared Memory | 100 KB/SM | 228 KB/SM | 256 KB/SM |
| L2 Cache | 72 MB | 50 MB | 64 MB |
| HBM | 24 GB GDDR6X(~1 TB/s) | 80 GB HBM3(~3.35 TB/s) | 192 GB HBM3e(~8 TB/s) |
| NVLink | ❌ | NVLink4(900 GB/s) | NVLink5(1800 GB/s) |
| TMA | ❌ | ✅(1st Gen) | ✅(2nd Gen Enhanced) |
| WGMMA | ❌ | ✅(128-thread) | ✅(Enhanced, larger tiles) |
| Process | TSMC 4N | TSMC 4N | TSMC 4NP |

## 1. SM内部结构

### 1.1 SM89(Ada) — Warp-Centric设计

```
SM89内部:
  ┌────────────────────────────────────────────────────┐
  │  4个Warp Scheduler (每个调度2条指令/时钟)          │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
  │  │ WS0      │ │ WS1      │ │ WS2      │ │ WS3      │ │
  │  │32 FP32   │ │32 FP32   │ │32 FP32   │ │32 FP32   │ │
  │  │4 TC      │ │4 TC      │ │4 TC      │ │4 TC      │ │
  │  │1 RT Core │ │          │ │          │ │1 RT Core │ │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
  │                                                      │
  │  Register File: 256 KB (65536×32-bit)                │
  │  Shared Memory: 100 KB (可配)                        │
  │  L1 Cache: 与smem共享空间                             │
  │                                                      │
  │  128 CUDA Cores + 16 Tensor Cores + 2 RT Cores      │
  └────────────────────────────────────────────────────┐
```

**关键特性**:
- **4th Gen Tensor Core**: HMMA.16816(FP16/BF16), HMMA.16832(FP8/INT8), HMMA.SPARSE
- **FP8 E4M3/E5M2**: 新增! 2x吞吐 vs FP16 → 但需要fused kernel(TE)避免Python dequant
- **72MB L2**: 巨大L2 → 很多kernel不再需要高occupancy来隐藏延迟 → 缓存命中替代!
- **100 KB smem**: 比Ampere增加4KB → 更大tile但occupancy风险

### 1.2 SM90(Hopper) — Warp-Group-Centric设计

```
SM90关键变化:
  1. TMA(Tensor Memory Accelerator): 硬件自动搬运数据 → 线程不参与!
     → TMA unit per SM → 异步从HBM到smem → 不消耗线程!
     → 5D tensor descriptor → 硬件计算地址 → 无需CPU参与

  2. WGMMA(Warp-Group MMA): 128线程协作矩阵运算
     → 4个warp组成1个warp group → 一起完成一次大MMA
     → 16×8×16 → 更大tile → 更高吞吐
     → FP8 WGMMA: 16×8×32 → 双倍FP8吞吐!

  3. 228 KB smem: 比SM89大2.28x → 更多stage buffer → 更深pipeline!

  4. Cluster: 多SM共享smem → TMA multicast → 跨SM通信!
```

### 1.3 SM100(Blackwell) — Enhanced WGMMA

```
SM100关键变化:
  1. 5th Gen Tensor Core: FP4/FP6新增!
     → FP4(E2M1): 4-bit → 2x吞吐 vs FP8 → 超低精度推理!
     → FP6(E3M2): 6-bit → 1.5x吞吐 vs FP8 → 更精确
     → Enhanced WGMMA: 更大tile(128×128×128), 更灵活layout

  2. 256 KB smem: 比SM90多28KB → 更double-buffer空间

  3. 2nd Gen TMA: 更大bulk(512 bytes vs 256), 更快setup, 更好overlap

  4. HBM3e: 8 TB/s → 比SM90的3.35 TB/s快2.4x → memory-bound kernel大加速!

  5. NVLink5: 1800 GB/s → 比NVLink4(900 GB/s)快2x → 通信加速!

  6. Decompression Engine: 硬件解压 → compressed data → 无需CPU!
```

## 2. Warp Scheduler机制

### 2.1 Warp调度流程

```
每个Warp Scheduler的工作:
  1. 从active warps中选择eligible warp
     → Eligibility: 源寄存器ready + 目标执行单元free + 无memory依赖

  2. Dual-issue: 每时钟发2条独立指令到不同执行单元
     → 例: 1条FP32 + 1条LD/ST → 同时执行
     → 例: 1条FP32 + 1条INT32 → 同时执行
     → 但不能: 2条FP32(同一pipeline)

  3. Scoreboard: 每个warp有scoreboard追踪寄存器依赖
     → 写后读(RAW) → 必须等
     → 写后写(WAW) → 必须等
     → 读后写(WAR) → 通常不等(register file多端口)
```

### 2.2 Occupancy与延迟隐藏

```
Occupancy = active warps / max warps per SM
  → SM89: max 48 warps/SM (1536 threads / 32)
  → 实际: 取决于register/shared memory用量

  每线程registers → occupancy影响:
    32 regs → 48 warps → 100% occupancy
    40 regs → 38 warps → 79% → 仍OK
    64 regs → 24 warps → 50% → 下降
    128 regs → 12 warps → 25% → 可能不够!

延迟隐藏:
  → 每个warp等待(如memory load)时 → scheduler切换到其他warp
  → 需要足够warps → 否则SM闲置!

  但SM89(L2 72MB)新规律:
  → 大L2 → 很多memory操作直接cache命中 → 延迟短 → 需要更少warps!
  → 之前需要48 warps → 现在可能24 warps就够了
  → → 更少warps → 更多registers/warp → 更大kernel tile!
```

### 2.3 Occupancy对不同kernel的影响

```
Compute-bound kernel: occupancy不是瓶颈 → 4 warps也能充分利用!
  → 因为compute pipeline长 → 只要有4个warp交替计算 → 足够
  → 例: HMMA kernel ~40 regs → 38 warps → 79% → 足够!

Memory-bound kernel: occupancy关键 → 需要很多warps隐藏延迟!
  → HBM load ~300 cycles → 需要足够warps让SM不闲置
  → SM89: L2 hit ~30 cycles → 需要更少warps → 低occupancyOK
  → SM90: TMA搬运 → warp不参与 → occupancy无关!

→ SM89优化: 适度occupancy(24-48 warps) + L2 friendly access pattern
→ SM90优化: 低occupancy(4-16 warps) + TMA pipeline + WGMMA
```

## 3. Tensor Core微架构

### 3.1 HMMA(SM89) — Warp-Level MMA

```
HMMA指令(4th Gen Tensor Core):
  HMMA.16816.F32 BF16 → 16×8×16 BF16 MMA → FP32累加
  HMMA.16816.F32 FP16 → 同上但FP16输入
  HMMA.16832.F32 FP8  → 16×8×32 FP8 MMA → FP32累加 ×2吞吐!
  HMMA.SPARSE.16832   → 2:4稀疏 → 2x吞吐!

执行流程:
  1 warp(32 threads) → 每线程持有矩阵片段 → 协作完成MMA
  → 16×8×16: M=16行, N=8列, K=16列 → 一次乘加128个元素
  → FP8: 16×8×32 → K=32 → 一次乘加256个元素 → 2x吞吐!

  线程分工:
  → 每4线程处理1个8×8 tile → 32线程 × 4 = 128个元素(FP16)
  → ldmatrix指令 → 从smem加载矩阵片段到寄存器
  → HMMA → 寄存器片段 × × 累加 → 结果在寄存器
```

### 3.2 WGMMA(SM90) — Warp-Group MMA

```
WGMMA指令(Hopper 4th Gen Enhanced Tensor Core):
  128线程(4 warps)协作 → 更大tile

  WGMMA指令格式:
  → wgmma.mma_async 64×64×16 BF16 → 64×64 output tile
  → wgmma.mma_async 64×64×32 FP8  → 双倍K维度

  关键区别 vs HMMA:
  → HMMA: 1 warp → 小tile → 需要loop → 很多overhead
  → WGMMA: 4 warps → 大tile → loop少 → 更高吞吐!

  数据来源:
  → A: smem(TMA加载的) → 直接送WGMMA → 无需ldmatrix!
  → B: smem(TMA加载的) → 直接送WGMMA
  → C: 寄存器 → 累加结果

  → TMA + WGMMA pipeline:
     TMA加载A到smem → TMA加载B到smem → WGMMA计算 → 循环
     → 线程不参与加载! → 0线程搬运开销!
```

### 3.3 SM100 Enhanced WGMMA

```
Blackwell 5th Gen Tensor Core新特性:
  1. FP4/FP6: E2M1/E3M2 → 4bit/6bit → 超低精度推理
     → FP4 WGMMA: 更大K维度 → 4x吞吐 vs FP16
     → NVFP4: block scaling(16值/block) → 精度更好

  2. 更大MMA tile: 128×128×128 → 更少loop → 更高吞吐
  3. 更灵活layout: A/B/C layout约束放松 → 更容易编写kernel
  4. 改善累加: FP8可选FP16累加(比FP32快但精度OK)

  RTX 4090限制: SM89不支持FP4/FP6/Enhanced WGMMA!
```

## 4. Memory Hierarchy

### 4.1 RTX 4090实测

```
Memory层次:
  Register: ~0 cycles → 最快但有限(256KB/SM)
  Shared Memory: ~30 cycles → 100KB/SM → 手动管理
  L1 Cache: ~30 cycles → 与smem共享 → 自动
  L2 Cache: ~200 cycles → 72MB → 自动 → 巨大!
  HBM: ~300-400 cycles → 24GB → 876.8 GB/s实测

关键发现: L2 72MB → 很多数据直接L2 hit → 不需要HBM!
  → 权重: 7B模型14GB → 不在L2 → 每次decode都HBM读取
  → KV: B×S×H×2 = 几MB → 可能L2 hit! → KV读取更快
  → Activation: 几KB → L1/L2 hit → 非常快

→ Decode瓶颈是权重HBM读取 → 量化(INT4→3.5GB) → L2可能fit部分!
```

### 4.2 SM90/SM100 Memory变化

```
SM90(Hopper):
  → 228 KB smem → 比SM89大2.28x → 3-stage FP16 pipeline可行!
  → 50 MB L2 → 比SM89小 → 但TMA预取弥补
  → HBM3 3.35 TB/s → 比SM89 GDDR6X 1 TB/s快3.35x

SM100(Blackwell):
  → 256 KB smem → 再增28KB → 4-stage pipeline可行!
  → 64 MB L2 → 比SM90增14MB
  → HBM3e 8 TB/s → 比SM90快2.4x! → memory-bound kernel大加速!

  → NVLink5 1800 GB/s → 通信2x加速 → TP/EP带宽瓶颈减轻!
```

## 5. 关键微架构洞察

### 5.1 SM89 vs SM90设计哲学差异

```
SM89(Ada)设计哲学:
  → Warp-centric → 1 warp做1件事 → 精细控制 → 但需要手动pipeline
  → 无TMA → 线程必须搬运数据 → cp.async是手动pipeline的核心
  → 无WGMMA → HMMA小tile → 需要loop → overhead
  → 大L2 → 缓存友好 → occupancy可以低

SM90(Hopper)设计哲学:
  → Warp-group-centric → 4 warps协作做1件事 → 更高吞吐但控制复杂
  → TMA → 硬件自动搬运 → 线程释放! → 只做计算
  → WGMMA → 大tile → 少loop → 高吞吐
  → Cluster → 多SM共享smem → 跨SM通信
  → 但: 编程更复杂 → CUTLASS 3.x抽象了复杂性

SM100(Blackwell)设计哲学:
  → Enhanced WGMMA → 更大tile更灵活 → 编程更容易
  → Enhanced TMA → 更快更好overlap → pipeline更高效
  → FP4/FP6 → 极低精度推理 → 吞吐再翻倍
  → Decompression → 硬件解压 → 存压缩数据→读取时解压→省带宽!
```

### 5.2 RTX 4090(SM89)优化策略

```
RTX 4090核心优化:
  1. L2-friendly access → sequential → cache line命中率高
  2. cp.async pipeline → 3-stage → 手动但有效(20-30%加速)
  3. 适度occupancy → 24-48 warps → 不追求100% → 释放更多register给tile
  4. FP8 HMMA → 16×8×32 → 2x吞吐 → 需fused kernel(TE)
  5. 2:4 Sparsity → HMMA.SPARSE → 2x计算 → 但精度损失

RTX 4090不可用:
  → TMA → 不支持 → 手动cp.async
  → WGMMA → 不支持 → HMMA loop
  → Cluster → 不支持 → 单SM独立
  → FP4/FP6 → 不支持 → FP8是最低精度
  → NVLink → 不支持 → 单GPU或PCIe DDP/FSDP
```

### 5.3 SM89 → SM90迁移指南

```
SM89 kernel → SM90适配:
  1. HMMA loop → WGMMA: 小tile loop → 大tile → 减少循环次数
  2. cp.async → TMA: 手动pipeline → TMA自动 → 简化但需重设计
  3. 1 warp → warp group: 32线程 → 128线程 → 需要4x线程协调
  4. smem 100KB → 228KB: 更多buffer → 可用4-stage pipeline
  5. register 40 → 48+: WGMMA需要更多register → occupancy下降但吞吐更高

CUTLASS 3.x已抽象了SM90复杂性:
  → CollectiveMainloop → TMA+WGMMA pipeline → 自动
  → Epilogue → 输出处理 → 自动
  → 但SM89 → 需用SM80 path(cp.async+HMMA) → 不用SM90 path!
```

---

**Sources**:
- [NVIDIA Ada Lovelace Architecture Whitepaper](https://resources.nvidia.com/en-us-ada-lovelace-architecture)
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-hopper-architecture)
- [NVIDIA Blackwell Architecture](https://developer.nvidia.com/blog/nvidia-blackwell-architecture-demystified/)
- [CUDA C++ Programming Guide (SM89/SM90)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS 3.x Architecture](https://github.com/NVIDIA/cutlass)

**Related notes**: cuda-kernel-optimization-sm89.md (SM89优化), cutlass-3x-gemm-architecture.md (CUTLASS), flashattention-3-architecture.md (FA-3)