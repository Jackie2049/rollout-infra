# Data Center GPU Architecture Deep Dive — 从Consumer到数据中心GPU的Infra决策

> 2026-06-11 | SM89→SM90→SM100架构差异如何影响训练/推理决策! TMA解放线程! WGMMA 4x吞吐! Cluster跨SM协作! FP8训练成熟! Blackwell FP4+NVLink5+HBM3e!
> 参考: NVIDIA Hopper/Blackwell Whitepapers, CUTLASS 3.x, FlashAttention-3, TransformerEngine
> 关联: gpu-microarchitecture-sm89-sm90-sm100.md, cutlass-gemm-benchmark-rtx4090.md, fp8-gemm-algorithm-analysis-rtx4090.md

## 0. 核心定律: Consumer→数据中心 = 范式转移 → 不是简单升级!

```
RTX 4090 (SM89 Consumer):
  → 890 GB/s HBM → 169.6 TFLOPS FP16 → 24GB GDDR6X → 无NVLink
  → → Decode: memory-bound → weight reads 95% → 投机解码利用闲置
  → → → PCIe scaling灾难 → 单GPU最优 → LoRA训练 → INT4推理

数据中心GPU (SM90/SM100):
  → 3.35-8 TB/s HBM → 1-2 PFLOPS FP8 → 80-192GB HBM3 → NVLink 900-1800GB/s
  → → Decode: 相对compute更多 → 但仍memory-bound → 不同量化策略!
  → → → NVLink scaling可行! → FSDP/DDP多GPU → MoE EP → PD分离!

  关键差异:
    → RTX 4090: **计算是瓶颈的解药** → 投机解码利用计算余量 → INT4省内存→并发↑
    → → H100: **NVLink是瓶颈的解药** → 多GPU通信低延迟 → MoE EP → PD分离
    → → → → B200: **FP4+NVLink5是终极解药** → 量化极致+通信极致+内存极致

  范式转移:
    → Consumer: 单GPU最优 → LoRA+INT4+投机解码 → 一切为了fit 24GB!
    → → 数据中心: 多GPU最优 → FSDP+EP+PD分离 → 一切为了scale out!
    → → → → infra工程师必须理解: Consumer vs 数据中心 → 不同优化范式!
```

## 1. Hopper SM90 (H100/H200) — 4大架构创新

### 1.1 TMA: Tensor Memory Accelerator — 线程解放!

```
传统(SM89 Ada):
  → 线程自己计算地址 + 自己拷贝数据 → global→smem copy = 线程工作!
  → → cp.async: 线程发出异步拷贝 → 但线程仍然要计算地址!
  → → → 线程数 = 地址计算瓶颈 → 小tensor浪费线程 → 低效!

TMA(SM90 Hopper):
  → 硬件单元负责所有地址计算+数据拷贝 → 线程完全解放!
  → → TMA descriptor: 预编码tensor layout(bounds+strides+swizzle)
  → → → 一次创建 → 多次使用 → 线程只issue "load from descriptor" → 硬件完成!

  TMA能力:
    → 最多5维tensor → 直接加载tile到smem → 硬件自动swizzle → bank-conflict-free!
    → → Cluster-wide TMA: 一个TMA transaction可以为cluster内所有thread block加载!
    → → → → 不需要每个block单独加载 → 通信量↓ → 效率↑!

  对AI Infra的影响:
    → GEMM pipeline: TMA→smem→WGMMA → 不经过寄存器 → 更少register pressure!
    → → → → occupancy↑ → tile更大 → 吞吐↑ → CUTLASS 3.x的新pipeline设计!

    → RTX 4090对比:
      → SM89: cp.async → 线程计算地址 → register pressure高 → occupancy受限
      → → SM90: TMA → 硬件计算地址 → register pressure低 → occupancy↑ → 更大tile!

  实测影响(CUTLASS benchmark):
    → SM89 cp.async: B=256接近peak但occupancy低 → 小B利用率差
    → → SM90 TMA: B=128+更高利用率 → pipeline效应更显著 → 接近peak更快!
```

### 1.2 WGMMA: Warp Group MMA — 4x吞吐!

```
传统(SM89 HMMA):
  → 1 warp(32 threads) → MMA 16×8×16(FP16) → 累加器在寄存器
  → → 每条指令: ~256 FLOPs → 32线程共享 → 每线程8 FLOPs → 低!
  → → → 输入必须从寄存器提供 → register pressure高 → occupancy受限!

WGMMA(SM90):
  → 4 warp(128 threads) → MMA 64×64×16(BF16) / 64×64×32(FP8) → 累加器在寄存器
  → → 每条指令: ~262,144 FLOPs(BF16) → 128线程共享 → 每线程2048 FLOPs → 256x↑!
  → → → 输入从shared memory直接读取! → 不经过寄存器 → register↓ → occupancy↑!

  WGMMA关键特性:
    → 输入A: 从shared memory → 需要特定swizzle layout → TMA自动处理!
    → → 输入B: 从shared memory → 同样需要swizzle → TMA自动处理!
    → → → 累加器C: 在寄存器 → 128线程分布式持有 → warp group协作!

  精度支持:
    → BF16: M=64, N=64, K=16 → 每指令4,096乘加 → ~8K FLOPs per thread
    → → FP8(E4M3): M=64, N=64, K=32 → 2x吞吐 → ~16K FLOPs per thread
    → → → FP8(E5M2): backward → 同样64×64×32 → gradient量化

  对AI Infra的影响:
    → FP8训练: WGMMA 64×64×32 → 2x vs BF16 → TransformerEngine利用!
    → → → 实测: TE FP8 1.48-1.59x → 与WGMMA FP8 2x理论一致(但有quantize overhead)

    → RTX 4090对比:
      → SM89 HMMA.16816: 16×8×16 → per-instruction FLOPs低
      → → SM90 WGMMA.64×64×16: 4x per-instruction → 但需要warp group → 更复杂!

  CUTLASS 3.x pipeline:
    → 2/3-stage async pipeline: TMA(load) → smem(buffer) → WGMMA(compute)
    → → Pipeline barrier: bar.arrive/bar.sync → 纯硬件同步 → 比SM89软件同步更快!
    → → → → Pipeline overlap: load next tile while compute current → 延迟隐藏!
```

### 1.3 Thread Block Cluster — 跨SM协作!

```
传统(SM89):
  → Thread block = 1 SM → 只能访问自己的smem → 不能直接与其他SM通信
  → → 跨block通信 → 必须经过global memory → 慢! → 禁止在kernel内跨block!

Cluster(SM90):
  → Cluster = 最多8个thread block → 分布式shared memory(DSMEM)!
  → → → 任何block可以读任何其他block的smem → cluster_map_shared_rank()
  → → → → → 跨SM通信不需要global memory → DSMEM直接! → 快!

  Cluster编程:
    → cudaLaunchKernelEx → 设置cluster维度 → 最多8 block/cluster
    → → cluster_barrier::wait() → cluster级同步 → bar.arrive/bar.sync
    → → → → atomic操作跨DSMEM → 支持!

  对AI Infra的影响:
    → FlashAttention-3: 用cluster → 多SM协作计算attention → 接近peak利用率!
    → → 多block GEMM: 多SM共同计算一个大tile → DSMEM共享中间结果 → 更高效!
    → → → Distributed softmax: cluster内多个SM协作 → 不需要global memory roundtrip!

  RTX 4090:
    → SM89不支持cluster → 只能单block → flash attention必须单SM → 利用率低!
    → → → 这解释了RTX 4090上decode B=1仅1.8% peak → 单block无法充分利用!
```

### 1.4 FP8 Training — Hopper是数据中心FP8平台

```
FP8格式(SM90):
  → E4M3 (4 exponent + 3 mantissa): forward/weight → 动态范围小但精度高
  → → E5M2 (5 exponent + 2 mantissa): backward/gradient → 动态范围大但精度低
  → → → → 两者配合 → forward用E4M3(精度) → backward用E5M2(范围) → 训练不溢出!

  FP8 vs BF16:
    → WGMMA吞吐: FP8 64×64×32 vs BF16 64×64×16 → 2x!
    → → 内存: FP8 1 byte vs BF16 2 bytes → 50%省!
    → → → → 训练: 2x吞吐 + 50%内存省 → 与BF16一致 → 但有量化精度损失!

  TransformerEngine FP8:
    → Delayed Scaling: 用上一步的amax计算当前步的scale → 无额外overhead!
    → → → Per-block quantization: 128元素/block → 每block独立scale → 精度足够!
    → → → → Fused C++ kernel: quantize+GEMM+dequantize一体 → 无Python dequant overhead!

  RTX 4090 FP8实测:
    → SM89 FP8: WGMMA不支持! → 只能用HMMA.16832 → 比SM90慢!
    → → → → 但! SM89也有FP8 Tensor Core → HMMA.16832 → 吞吐比FP16低!
    → → → → → TE实测: 1.48-1.59x(B≥4) → 不是2x! → quantize overhead!

  SM90 FP8优势:
    → WGMMA 64×64×32 → 真正2x吞吐 → + TMA → + pipeline → → 整体更快!
    → → → 量化overhead更小 → fused kernel更好 → 整体接近2x!
```

## 2. Blackwell SM100 (B200/B300) — FP4 + NVLink5 + HBM3e

### 2.1 Blackwell关键创新

```
| 特性 | Hopper SM90 (H100) | Blackwell SM100 (B200) |
|------|--------------------|------------------------|
| Die | 单die | 双die(10TB/s互联→统一GPU!) |
| HBM | 80GB HBM3(3.35TB/s) | 192GB HBM3e(8TB/s) |
| NVLink | NVLink4(900GB/s) | NVLink5(1800GB/s) |
| Tensor Core | 4th Gen(WGMMA) | 5th Gen(Enhanced WGMMA) |
| 新精度 | FP8(E4M3/E5M2) | **FP4/FP6** + FP8 + INT4 |
| smem | 228KB | 256KB |
| L2 | 50MB | 64MB |
| Cluster | 8 block | 更多(可能16+?) |
| TDP | 350W(H100) | 1000W(B200) |
| Process | TSMC 4N | TSMC 4NP(208B transistors!) |

FP4/FP6:
  → FP4: 2 exponent + 2 mantissa → 极低精度 → 4x吞吐 vs FP16!
  → → FP6: 3 exponent + 3 mantissa → 中间精度 → 2.67x吞吐 vs FP16
  → → → → FP4用于推理量化 → 4x权重节省 → 7B → 1.75GB!
  → → → → → 但! FP4训练精度损失严重 → 仅推理用途!

NVLink5:
  → 1800 GB/s双向 → 比NVLink4(900GB/s) 2x → 比PCIe(~32GB/s) 56x!
  → → → 72 GPU全互联 → 130 TB/s总带宽 → 单rack训练万亿参数模型!
  → → → → → MoE EP: expert All-to-All → NVLink5下延迟<1ms → 可行!
  → → → → → → vs PCIe: All-to-All延迟>10ms → RTX 4090灾难!

双Die设计:
  → 2个GPU die → 10TB/s die-to-die互联 → 表现为单个统一GPU!
  → → → 不需要软件知道2个die → CUDA看作1个GPU → 透明!
  → → → → → 192GB = 2 × 96GB → 统一地址空间 → 单GPU 192GB!
```

### 2.2 数据中心GPU对AI Infra的5大影响

```
影响1: 训练 → 从FSDP灾难到NVLink scaling可行!

  RTX 4090 PCIe:
    → FSDP 7B 8GPU = 0.46x → 灾难! → PCIe带宽瓶颈 → 通信占75%!
    → → → 单GPU最优 → LoRA训练 → 不需要多GPU!

  H100 NVLink4:
    → AllReduce带宽: 900GB/s × 8GPU → ~8x vs PCIe → FSDP可行!
    → → → 7B FSDP 8GPU: 估计3-4x scaling → 可接受!
    → → → → → MoE EP: NVLink → All-to-All延迟<1ms → DeepSeek-V3可行!

  B200 NVLink5:
    → 1800GB/s → 2x vs H100 → FSDP scaling更好 → 万亿参数训练!
    → → → → → 72GPU NVL72 → 13.5TB统一内存 → 训练万亿参数!

影响2: 推理 → 从单GPU INT4到PD分离!

  RTX 4090:
    → 单GPU推理 → INT4+INT8KV+FlashInfer → 4,791 tok/s(B=118)
    → → → PD分离需要NVLink → RTX 4090不适合 → 单GPU最优!

  H100 NVLink4:
    → PD分离可行! → prefill GPU+decode GPU → KV transfer via NVLink!
    → → → → → prefill: compute-bound → 大batch → 高FLOPS利用率
    → → → → → → decode: memory-bound → 小batch → 高内存利用率
    → → → → → → → → NVLink KV transfer <1ms → PD分离有效!

  B200 NVLink5:
    → PD分离更高效 → KV transfer更快 → 2x vs H100!
    → → → → → 192GB HBM → KV cache容量更大 → 更多并发 → 更长上下文!

影响3: 量化 → 从INT4 AWQ到FP8训练!

  RTX 4090:
    → INT4 AWQ推理 → 75%权重节省 → cos_sim=0.993 → 但需fused kernel(Marlin)!
    → → → FP8训练: TE实测1.48-1.59x → SM89 FP8不是2x!
    → → → → → Python dequant 20x慢 → fused kernel必需!

  H100:
    → FP8训练: WGMMA 2x → TE更高效 → 整体接近2x加速!
    → → → → → 推理FP8: 不需要INT4 → FP8精度更好(cos_sim=0.999996)!

  B200:
    → FP4推理: 4x吞吐 → 7B→1.75GB → 但精度loss大 → 需要验证!
    → → → → → FP8训练更成熟 → FP4推理是新方向 → 需要fused kernel!

影响4: MoE → 从PCIe灾难到EP可行!

  RTX 4090:
    → MoE serving: PCIe → All-to-All延迟>10ms → 不可行!
    → → → → → DeepSeek-V3 MoE: 256 experts → 需NVLink!

  H100:
    → EP可行! → NVLink All-to-All → 延迟<1ms → DeepEP v2!
    → → → → → DeepSeek-V3: EP+NVLink → 37B active → 671B total → 18x稀疏!

  B200:
    → EP更好 → NVLink5 2x带宽 → 更多expert并行 → 更高吞吐!

影响5: Kernel → 从手动pipeline到硬件辅助pipeline!

  SM89:
    → cp.async → 软件pipeline → 需要手动multi-stage → 程序员负担大!
    → → → → → CUTLASS 2.x: 2-stage pipeline → 手动管理 → 复杂!

  SM90:
    → TMA → 硬件pipeline → 程序员只issue TMA load → 硬件完成!
    → → → → → CUTLASS 3.x: async pipeline → TMA+WGMMA → 更简洁!
    → → → → → → FlashAttention-3: TMA+WGMMA+Cluster → 接近peak!

  SM100:
    → Enhanced TMA+WGMMA → 更大tile → 更多pipeline stages → 更接近peak!
    → → → → → → 新precision: FP4/FP6 → kernel需支持新格式
```

## 3. Consumer vs 数据中心: Infra决策对比

```
| 维度 | RTX 4090 (Consumer) | H100 (数据中心) | B200 (数据中心) |
|------|--------------------|--------------------|-----------------|
| 训练策略 | 单GPU+LoRA | FSDP多GPU | FSDP+PP+EP |
| 推理策略 | INT4+投机解码 | BF16+投机+FlashInfer | FP8+PD分离 |
| MoE策略 | 不推荐(PCIe灾难) | EP+NVLink | EP+NVLink5 |
| PD分离 | 不适合(无NVLink) | 适合(NVLink4) | 适合(NVLink5) |
| FP8训练 | 1.48-1.59x(TE) | ~2x(WGMMA) | >2x(Enhanced WGMMA) |
| 量化推理 | INT4 AWQ | FP8 BF16推理 | FP4推理 |
| KV优化 | INT8 KV+GQA-8 | INT8 KV+GQA | FP8 KV+GQA |
| 投机解码 | n-gram/Eagle | Eagle/MTP | Eagle/MTP+FP4 draft |
| 上下文 | S≤4K default | S≤128K | S≤256K |
| 并发 | ~118(INT4+INT8KV) | ~500+(HBM3) | ~1000+(HBM3e) |
| 成本 | $1,500 GPU | $25,000 GPU | $40,000 GPU |

RTX 4090最优配置(我们实测):
  → 7B INT4+INT8KV+GQA-8+FlashInfer+B=118 → 4,791 tok/s → $1,500

H100最优配置(估计):
  → 7B BF16+INT8KV+GQA-8+FlashInfer+B=500+ → ~15,000 tok/s → $25,000
  → → 3x吞吐 → 但17x成本 → 性价比不如RTX 4090!
  → → → → 但! H100可以多GPU → 总吞吐更高 → 适合大规模生产!

  关键: RTX 4090性价比更高 → H100绝对性能更高 → 选择取决于规模!
  → → 小规模(≤100并发) → RTX 4090最优
  → → → 大规模(>1000并发) → H100/B200必需
```

## 4. CUTLASS Pipeline: SM89 vs SM90对比

```
SM89 Pipeline (CUTLASS 2.x):
  → 2-stage: load_stage + compute_stage
  → → Stage 1: cp.async load → global→smem → 线程计算地址
  → → → Stage 2: HMMA compute → smem→registers→TC→registers
  → → → → → 手动管理: 程序员必须手动管理pipeline state → 复杂!

  Pipeline流程:
    for each tile:
      wait_for(load_stage[0])  // 等待数据到达smem
      compute(HMMA, smem[0])   // 从smem读→寄存器→TC计算→寄存器累加
      issue_load(cp.async, smem[1]) // 异步加载下一个tile
      swap_stages()             // 交换stage指针

SM90 Pipeline (CUTLASS 3.x):
  → 2/3-stage async: TMA_load → smem → WGMMA
  → → Stage 1: TMA load → global→smem → 硬件计算地址(TMA descriptor)
  → → → Stage 2: WGMMA compute → smem→TC→寄存器(累加器)
  → → → → → 硬件管理: TMA descriptor预计算 → bar.arrive/bar.sync → 更简洁!

  Pipeline流程:
    for each tile:
      issue_tma_load(descriptor, smem[i])  // TMA异步加载
      barrier_wait(smem[i])                 // 等TMA完成
      wgmma_issue(smem[i], accumulator)     // WGMMA从smem直接读!
      release_buffer(smem[i-2])             // 释放已消费的buffer
      issue_tma_load(descriptor, smem[i-2]) // 加载下一个tile到释放的buffer

关键差异:
  → SM89: 线程计算地址 + 拷贝 → 寄存器压力高 → occupancy低
  → → SM90: TMA硬件计算地址 + 拷贝 → 寄存器压力低 → occupancy高
  → → → → → WGMMA从smem直接读 → 不经过寄存器 → 更多资源给累加器!

  实测影响:
    → RTX 4090 CUTLASS BF16: decode B=1 → 1.8% peak → 低利用率
    → → → H100 CUTLASS FP8: decode B=1 → 估计~3% peak → 仍然低!
    → → → → → 但! 大batch → H100利用率更高 → WGMMA更大tile → 更接近peak!
    → → → → → → H100优势在大batch训练/prefill → 小batch decode仍然memory-bound!
```

## 5. FlashAttention-3: Hopper如何突破

```
FlashAttention-1(SM80 Ampere):
  → Online softmax + tiling → O(N²/M) IO → 内存省 → 但GPU利用率低
  → → → → → 单block → 每block一个SM → 利用率低

FlashAttention-2(SM89/SM90):
  → 改进tiling → 减少non-matmul FLOPs → 更接近peak → 2x vs FA-1
  → → → → → 仍然单block → SM利用率有限

FlashAttention-3(SM90 Hopper ONLY):
  → 利用TMA+WGMMA+Cluster+FP8 → 接近peak利用率!
  → → → TMA: 异步加载Q/K/V → 不需要线程拷贝 → 更快!
  → → → → → WGMMA: 从smem直接计算QK^T → 更大tile → 更高效!
  → → → → → → Cluster: 多SM协作 → distributed softmax → 更准确!
  → → → → → → → → FP8: Q/K/V用E4M3 → 2x吞吐 → 精度仍可接受!

  FA-3实测(H100):
    → 接近peak HBM bandwidth → ~75-90% of peak → 极高效!
    → → → vs RTX 4090 FA-2: ~3-5% of peak → 差距巨大!

  RTX 4090限制:
    → 无TMA → 无WGMMA → 无Cluster → FA-2是RTX 4090上限!
    → → → → → FlashInfer是RTX 4090最优 → 但FA-3需要Hopper!
```

## 6. 7 Core Laws — Consumer vs 数据中心GPU

```
1. **Consumer-Single-GPU-Optimal**: RTX 4090 → 单GPU最优 → LoRA+INT4+投机解码
   → PCIe scaling灾难 → NVLink不可用 → 一切fit in 24GB

2. **数据中心-Multi-GPU-Optimal**: H100/B200 → 多GPU最优 → FSDP+EP+PD分离
   → NVLink scaling可行 → 多GPU通信低延迟 → 一切scale out!

3. **TMA-Frees-Threads**: SM90 TMA → 线程不再计算地址 → occupancy↑ → tile↑
   → vs SM89 cp.async → 线程计算地址 → register pressure → occupancy受限

4. **WGMMA-4x-Throughput**: SM90 WGMMA → 4x vs HMMA → FP8 2x → 大tile → 高效
   → 但需要warp group + swizzle layout → CUTLASS 3.x自动处理

5. **Cluster-Cross-SM-Cooperation**: SM90 Cluster → DSMEM → 跨SM协作 → FA-3
   → vs SM89 → 单block → 跨SM必须global memory → FA-2上限

6. **NVLink-Unlocks-Multi-GPU**: NVLink4/5 → FSDP/DDP可行 → MoE EP可行 → PD分离可行
   → vs PCIe → FSDP灾难 → MoE不可行 → PD分离不适合

7. **FP4/FP8-Quantization-Evolution**: INT4(推理)→FP8(训练)→FP4(推理next-gen)
   → RTX 4090: INT4 AWQ推理 → 需fused kernel → Python dequant灾难
   → → H100: FP8训练(WGMMA) → 2x → FP8推理(精度更好) → 不需要INT4!
   → → → B200: FP4推理(next-gen) → 4x → 需验证精度 → fused kernel必需
```

## 7. RTX 4090 vs H100 vs B200推理计算

```
7B模型推理对比:

| 配置 | RTX 4090 | H100 | B200 |
|------|----------|------|------|
| BF16权重(GB) | 14 | 14 | 14 |
| INT4权重(GB) | 3.5 | 3.5 | ~1.75(FP4!) |
| FP8权重(GB) | 7(不适合推理) | 7 | 7 |
| HBM(GB) | 24 | 80/141 | 192 |
| INT8KV并发(S=4K) | ~118 | ~1000+ | ~2000+ |
| INT8KV吞吐(tok/s) | 4,791 | ~15,000+ | ~30,000+ |
| FP8KV并发(S=4K) | ~236 | ~2000+ | ~4000+ |
| NVLink PD分离 | ❌ | ✅ | ✅ |
| 投机解码 | n-gram/Eagle | Eagle/MTP | Eagle/MTP |
| GPU价格 | $1,500 | $25,000 | $40,000 |
| 性价比(tok/s/$) | 3.2 | 0.6 | 0.75 |

→ RTX 4090性价比5x! → 但H100/B200绝对性能3-6x更高
→ → → 选择取决于规模: 小规模→RTX 4090 / 大规模→H100/B200
```

## 8. Infra工程师决策树

```
问题: 我要训练/推理什么模型? 多大规模?

模型≤7B, 并发≤100:
  → RTX 4090最优! → INT4+INT8KV+FlashInfer → 4,791 tok/s → 1-2 GPU
  → → → 训练: LoRA BF16 + B=4 accum=4 → 单GPU → 6-8GB → 可行!
  → → → → → 成本: ~$3,000(2 GPU) → 性价比最高!

模型7B-13B, 并发100-1000:
  → H100最优! → BF16+INT8KV+FlashInfer → ~15,000 tok/s → 2-8 GPU
  → → → 训练: FSDP 8GPU → NVLink → scaling 3-4x → 可行!
  → → → → → 成本: ~$200,000(8 GPU) → 大规模必需!

模型≥70B, 并发>1000:
  → B200/GB200最优! → FP8+NVLink5+PD分离 → ~30,000+ tok/s → 72 GPU NVL72
  → → → 训练: FSDP+PP+EP → 万亿参数 → NVL72单rack!
  → → → → → 成本: ~$3,000,000(72 GPU rack) → 超大规模!

MoE模型(DeepSeek-V3级):
  → 必须H100/B200! → EP需要NVLink → PCIe灾难!
  → → → RTX 4090: MoE不可行 → PCIe All-to-All延迟>10ms!
  → → → → → H100: EP可行 → NVLink → <1ms → DeepEP v2!

推理PD分离:
  → 必须H100/B200! → KV transfer需要NVLink!
  → → → RTX 4090: PD分离不适合 → PCIe KV transfer慢!
  → → → → → H100: PD分离可行 → NVLink4 KV transfer → vLLM NIXL!
```

## 参考文献

```
1. NVIDIA Hopper Architecture Whitepaper — SM90, TMA, WGMMA, Cluster
2. NVIDIA Blackwell Architecture Whitepaper — SM100, FP4, NVLink5
3. CUTLASS 3.x Documentation — TMA+WGMMA async pipeline
4. FlashAttention-3: Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision", 2024
5. TransformerEngine — NVIDIA FP8 training library
6. DeepSeek-V3 Technical Report — EP+NVLink production deployment

我们的笔记:
- gpu-microarchitecture-sm89-sm90-sm100.md — 三代SM内部结构
- cutlass-3x-gemm-architecture.md — CUTLASS pipeline
- fp8-gemm-algorithm-analysis-rtx4090.md — FP8 on RTX 4090
- transformer-engine-fp8-rtx4090.md — TE FP8实测
- rtx4090-pcie-decision-guide.md — RTX 4090决策树