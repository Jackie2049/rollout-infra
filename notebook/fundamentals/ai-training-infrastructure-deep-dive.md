# AI Training Infrastructure Deep Dive

> 2026-06-10 | 训练基础设施=AI模型诞生的工厂! 从training loop到DP internals, 从gradient accumulation到mixed precision, 从checkpointing到pipeline orchestration, 训练=系统工程的极致!
> 关联: fsdp2-scaling-benchmark-rtx4090.md, optimization-algorithms-deep-dive.md, distributed-training-simulator.md, verl-rl-infra-reading.md

## 0. 核心定律: 训练 = 计算 + 通信 + 内存 的三方博弈

训练≠推理 → backward + optimizer + checkpoint → 内存=3x推理 → 通信=必需!

关键数据(RTX 4090实测):
- FSDP 8GPU = 0.46x scaling → PCIe灾难 → 训练需NVLink!
- AdamW 7B = 84GB → 单GPU OOM → CPU offload或Lion 42GB!
- FP8 TE训练 = 1.48-1.59x 加速 → B≥4 → fused kernel必需!
- BF16+FSDP1 = 1.51x加速+49%内存省 → 最佳组合!
- Gradient accumulation = 等效大batch → 小GPU→大batch→有效!

## 1. Training Loop Architecture — 训练循环架构

```
标准训练循环 (PyTorch):

for epoch in range(num_epochs):
    for batch in dataloader:
        # Step 1: Forward pass → 计算loss
        output = model(batch.input)
        loss = criterion(output, batch.target)

        # Step 2: Backward pass → 计算梯度
        loss.backward()  # → 梯度存储在param.grad

        # Step 3: Gradient sync (DP/FSDP) → 通信!
        # DDP: AllReduce gradients → 每步同步
        # FSDP: ReduceScatter+AllGather → 每层同步

        # Step 4: Optimizer step → 更新参数
        optimizer.step()  # → AdamW: m, v, param update

        # Step 5: Gradient zero → 清零
        optimizer.zero_grad()  # → 释放梯度内存

训练循环内存分析 (7B BF16):
  → 模型参数: 14GB (BF16)
  → → → AdamW状态: m(14GB) + v(14GB) = 28GB → 2x参数!
  → → → → → 梯度: 14GB → 1x参数!
  → → → → → → → 激活值: ~14GB → 1x参数 (backward需!)
  → → → → → → → → → 总: 14+28+14+14=70GB → 5x参数! → RTX 4090 24GB→OOM!

内存优化路径:
  → FSDP ZeRO-3: 参数分片→每GPU仅14/8=1.75GB → 省内存!
  → → → Gradient accumulation: 不每步sync→累积N步→1次sync → 省通信!
  → → → → → Mixed precision BF16: 参数16bit→激活BF16→梯度BF16→省内存!
  → → → → → → → CPU offload: Adam状态→CPU→省GPU→但慢!
  → → → → → → → → → Checkpointing: 不存所有激活→重新计算→省2x内存!

RTX 4090最优训练循环:
  → 7B BF16: 单GPU→OOM → 必需优化!
  → → → FSDP1 2GPU: 参数分片→每GPU7GB+Adam14+梯度7+激活7=35GB→仍OOM!
  → → → → → FSDP1+BF16+Adam CPU offload→勉强→但CPU慢→通信瓶颈!
  → → → → → → → 小模型(25M-125M)→单GPU→可行→训练OK!
  → → → → → → → → → 7B→多GPU→PCIe→FSDP灾难→RTX 4090不适合7B训练!
```

## 2. Data Parallelism Internals — 数据并行内部机制

```
数据并行3代:

1. DDP (DistributedDataParallel) — PyTorch原生:
   → 每GPU→完整模型→不同数据→forward→backward→AllReduce梯度!
   → → → AllReduce: 所有GPU→求和→分发→每GPU→相同梯度→同步更新!
   → → → → → 内存: 每GPU=完整模型+Adam+梯度+激活 → 5x参数 → OOM!

   → → → → → → → DDP通信: 每步→AllReduce→梯度总量=参数量→通信量大!
   → → → → → → → → → 7B BF16: AllReduce 14GB → PCIe 2.76GB/s → 5.07s → 灾难!
   → → → → → → → → → → → NVLink: 14GB / 300GB/s = 0.047s → 可接受!

2. FSDP (FullyShardedDataParallel) — ZeRO-3实现:
   → 参数分片→每GPU仅1/N → 按需AllGather→计算→ReduceScatter→梯度分片!
   → → → 每层(Layer-by-layer):
     → → → → → Forward: AllGather参数→计算→丢弃(释放)!
     → → → → → → → Backward: AllGather参数→反向→ReduceScatter梯度→丢弃!
   → → → → → → → → → 通信: 2×参数量/N per-GPU → 比DDP少→但每层2次!

   → → → → → → → → → 内存: 每GPU=参数/N+Adam/N+梯度/N+激活 → ZeRO-3省8x!
   → → → → → → → → → → → 7B 8GPU BF16: 14/8=1.75GB+28/8=3.5GB+14/8=1.75+激活 → ~10GB → 可行!

3. Hybrid DP+TP+PP — 大模型训练:
   → DP(数据)+TP(张量)+PP(流水线) → 三维并行 → Megatron!
   → → → TP: 参数→切分→NVLink→单节点 → 通信少!
   → → → → → PP: 层→切分→P2P→跨节点 → 气泡→但省内存!
   → → → → → → → DP: 数据→切分→AllReduce → 多节点 → 简单!

RTX 4090 DP决策:
  → ≤2GPU: FSDP1勉强 → 25M/125M可行 → 7B不行!
  → → → >2GPU: PCIe灾难 → FSDP 8GPU=0.46x → 不推荐!
  → → → → → 7B训练→NVLink必需 → A100/H100 → RTX 4090不适合!
  → → → → → → → 小模型→DDP简单→FSDP内存省→RTX 4090≤2GPU!
```

## 3. Gradient Accumulation — 梯度累积

```
Gradient Accumulation = 小GPU→大batch的桥梁!

原理:
  → 真实batch太大 → GPU内存不够 → 分多步 → 累积梯度 → 1次更新!
  → → → micro_batch=B/accum_steps → 每步forward+backward → 不update!
  → → → → → 累积N步 → 等效batch=N×micro_batch → 1次optimizer.step()!

数学:
  → 等效batch = micro_batch × accum_steps
  → → → B=32 → micro_batch=4 → accum_steps=8 → 等效B=32!
  → → → → → 但: 梯度=累积 → 不是瞬间 → 延迟↑ → 但梯度等价!

内存影响:
  → 不累积: B=32 → 激活值=32×大 → OOM!
  → → → 累积: micro_batch=4 → 激活值=4×小 → 内存可行!
  → → → → → 梯度: 累积→不清零→占用1×参数 → 但不大!

通信影响 (DDP/FSDP):
  → DDP: 每micro_batch→AllReduce→通信↑accum_steps倍→灾难!
  → → → → → 解决: gradient_accumulation→只在最后step→sync→1次AllReduce!
  → → → → → → → PyTorch DDP: no_sync() → 禁用sync → 累积 → 最后sync → 省!

  → FSDP: 每层→仍需AllGather→无法累积→每层每步通信!
  → → → → → 但: micro_batch小→参数AllGather快→单层快!
  → → → → → → → 总通信量=层数×micro_batch×accum_steps → 与不累积相同!
  → → → → → → → → → 但: 每步通信量小→PCIe→overlap → 可行!

RTX 4090梯度累积推荐:
  → 7B BF16: micro_batch=1 → accum=32 → 等效B=32 → 但32步→慢!
  → → → FSDP1 2GPU: micro_batch=2 → accum=16 → 等效B=32 → 通信少!
  → → → → → 25M模型: micro_batch=4 → accum=8 → 等效B=32 → 快速可行!
```

## 4. Mixed Precision Training — 混合精度训练

```
混合精度3代:

1. FP16 AMP (Automatic Mixed Precision):
   → forward: FP16计算 → master params: FP32 → loss scaling!
   → → → 问题: FP16动态范围小→overflow→loss scaling→复杂→RTX 4090不支持!
   → → → → → 实测: FP16 AMP比BF16慢! → 反直觉 → 原因: loss scaling overhead!

2. BF16 (Bfloat16) — 现代训练首选:
   → forward+backward: BF16 → optimizer: FP32(AdamW) → 无需loss scaling!
   → → → BF16动态范围=FP32 → 不overflow → 简单 → RTX 4090支持!
   → → → → → 实测: BF16+FSDP1最佳! → 1.51x加速+49%内存省!
   → → → → → → → BF16训练=RTX 4090最优 → 不需要AMP → 简单!

3. FP8 (TransformerEngine) — 未来训练:
   → forward: FP8(E4M3) → backward: FP8(E5M2) → 省带宽→快!
   → → → 实测: FP8 TE 1.48-1.59x → B≥4 → fused kernel必需!
   → → → → → B=1: FP8慢0.75x → 小GEMM量化开销占优 → 不推荐!
   → → → → → → → FP8训练需要fused kernel → TE → Python dequant=灾难!

混合精度内存分析 (7B):
  → FP32训练: 参数28GB+Adam56GB+梯度28GB+激活28GB=140GB → 灾难!
  → → → BF16训练: 参数14GB+Adam28GB+梯度14GB+激活14GB=70GB → 仍大!
  → → → → → BF16+FSDP ZeRO-3 8GPU: 70/8≈9GB → 每GPU → 可行!
  → → → → → → → FP8训练: 参数7GB+Adam28GB(FP32)+梯度7GB+激活7GB=49GB → 省30%!

RTX 4090混合精度推荐:
  → BF16训练 → 最佳 → RTX 4090原生支持 → 简单!
  → → → FP16 AMP → 不推荐 → 慢+loss scaling → 不需要!
  → → → → → FP8 TE → B≥4可行 → 1.48-1.59x → 但需fused kernel!
```

## 5. Checkpointing Strategies — 检查点策略

```
训练checkpointing=防丢失+省内存+断点恢复!

3种checkpointing策略:

1. 同步barrier checkpointing (传统):
   → 所有GPU→暂停→保存→barrier→继续 → 一致!
   → → → 优点: 一致 → 任意恢复 → 简单!
   → → → → → 缺点: 全部暂停→浪费时间→GPU空闲→效率低!
   → → → → → → → 7B 8GPU: 保存14GB→10-30s→每1000步→浪费1-3%!

2. 异步checkpointing (推荐):
   → GPU→继续训练 → 后台线程→保存 → 无暂停!
   → → → 优点: GPU不暂停→训练继续→效率高!
   → → → → → 缺点: 不一致→最新checkpoint→可能旧1-2步→可接受!
   → → → → → → → verl: 异步+版本管理 → 最优 → 可恢复任意版本!

3. 异步+版本管理 (verl最优):
   → 异步保存 + 版本号 + 保留最近N → 可恢复任意版本!
   → → → verl实现: Ray → checkpoint_actor → 异步 → 不阻塞训练!
   → → → → → 版本: step_1000, step_2000, step_3000 → 最近3 → 保留!

Gradient checkpointing (activation recomputation):
  → 不存所有激活 → backward时重新计算 → 省内存但增加计算!
  → → → 激活值: 14GB → checkpointing → 仅存关键节点 → 重计算 → 省2x!
  → → → → → 但: 计算增加33% → 重新forward部分 → 代价!
  → → → → → → → FSDP+checkpointing → 反而增加内存! → FSDP+no checkpoint=最优(0.478GB)!
  → → → → → → → → → 原因: FSDP→参数分片→释放→激活checkpoint→不释放→矛盾!

RTX 4090 checkpointing推荐:
  → 小模型(25M): 同步→简单→快→10ms保存→无所谓!
  → → → 7B: 异步+版本管理 → verl → 推荐!
  → → → → → Gradient checkpointing: FSDP下不需要 → FSDP+no checkpointing=最优!
```

## 6. Training Pipeline Orchestration — 训练流水线编排

```
训练流水线编排=训练的指挥官!

verl RL训练流水线 (14步):
  → 1. 初始化 → actor+critic+reward+ref → 4模型!
  → → → 2. 生成rollout → vLLM/SGLang → 推理 → 采样!
  → → → → → 3. 计算reward → reward_model → 评分!
  → → → → → → → 4. 计算advantage → GRPO/PPO → 归一化!
  → → → → → → → → → 5. Actor训练 → forward+backward → PPO loss!
  → → → → → → → → → → → 6. Critic训练 → value → 更新!

  → → → → → → → → → → → verl优化:
    → → → → → → → → → → → → → Colocation: actor+critic→同一GPU→sleep/wake→省50%!
    → → → → → → → → → → → → → → Async rollout: vLLM→推理→actor训练→异步→1.29x!

Megatron训练流水线:
  → 1. 数据加载 → DataLoader → batch!
  → → → 2. Forward → TP+PP → 多GPU → 分层!
  → → → → → 3. Backward → TP+PP → 反向 → 分层!
  → → → → → → → 4. AllReduce → DP → 梯度同步!
  → → → → → → → → → 5. Optimizer → AdamW → 更新!
  → → → → → → → → → → → 6. Logging → metrics → 监控!

训练流水线瓶颈分析:
  → Forward: 2-6%时间 → 快 → 不是瓶颈!
  → → → Backward: 50-62%时间 → 主导 → 计算瓶颈!
  → → → → → Communication: FSDP→99.4%(PCIe) → 灾难 / NVLink→3.3%→OK!
  → → → → → → → Optimizer: 5-10% → AdamW → memory-bound → 但快!
  → → → → → → → → → DataLoader: 1-3% → 预取 → 不阻塞 → OK!

RTX 4090训练流水线推荐:
  → 小模型(25M-125M): DDP/FSDP1 ≤2GPU → 简单 → 可行!
  → → → RL训练(verl GRPO): 单GPU→LoRA→INT4→可行 → 不需要多GPU!
  → → → → → 7B训练: 多GPU→PCIe→灾难 → 不推荐 → 需NVLink!
```

## 7. RTX 4090 Training Optimization — RTX 4090训练优化总结

```
RTX 4090训练能力矩阵:

| 任务 | GPU需求 | RTX 4090可行? | 推荐方案 |
|------|---------|-------------|----------|
| 25M训练 | 1-2GPU | 可行! | DDP/FSDP1 |
| 125M训练 | 1-2GPU | 可行 | FSDP1 BF16 |
| 7B训练(全参数) | 4-8GPU NVLink | 不可行! | PCIe灾难→需NVLink |
| 7B LoRA训练 | 1GPU | 可行! | verl GRPO+LoRA+INT4 |
| 7B RL(PPO/GRPO) | 1-2GPU | 可行(LoRA) | verl colocation |
| FP8训练 | 1-2GPU | 可行(B≥4) | TE fused kernel |
| 7B推理 | 1GPU | 最优! | INT4+INT8KV+FlashInfer |

RTX 4090训练优化清单:
  → 1. BF16 → 不FP16 AMP → 简单+快!
  → → → 2. FSDP1 → 不DDP → 内存省50%+1.51x加速!
  → → → → → 3. ≤2GPU → 不>2GPU → PCIe scaling灾难!
  → → → → → → → 4. Gradient accumulation → 小GPU→大batch → micro_batch+accum!
  → → → → → → → → → 5. LoRA → 不全参数 → 参数省99.6% → 单GPU可行!
  → → → → → → → → → → → 6. verl GRPO → 不PPO → outcome-only+2模型 → 简单!
  → → → → → → → → → → → → → 7. Colocation → actor+critic→同一GPU → sleep/wake → 省50%!
```

## 8. Core Laws — 训练基础设施核心定律

1. **Training-Memory-5x Law**: 训练内存=5x参数 → 7B BF16=70GB → RTX 4090 OOM → 需FSDP/LoRA!
   → → → FSDP ZeRO-3→8x省 / LoRA→99.6%参数省 → 单GPU可行!

2. **PCIe-Training-Disaster Law**: RTX 4090 PCIe→FSDP 8GPU=0.46x→比1GPU慢2x→灾难!
   → → → ≤2GPU勉强→>2GPU完全不划算→7B训练需NVLink→A100/H100!

3. **BF16-Best Law**: BF16训练=最优→1.51x加速+49%内存省→RTX 4090原生→不需AMP!
   → → → FP16 AMP反而慢→loss scaling overhead→BF16动态范围=FP32→简单!

4. **Gradient-Accumulation-Bridge Law**: 梯度累积=小GPU→大batch桥梁→micro_batch×accum=等效B!
   → → → DDP→no_sync()→只在最后AllReduce→省通信→FSDP→每层仍需→但量小!

5. **LoRA-Training-Feasibility Law**: LoRA→99.6%参数省→单GPU训练7B可行→verl GRPO+LoRA→RTX 4090推荐!
   → → → r=8 α=16→全target_modules→95%性能→INT4+LoRA→进一步省!

6. **Async-Checkpoint Law**: 异步checkpoint→GPU不暂停→效率高→verl异步+版本管理→最优!
   → → → FSDP+gradient checkpointing→反而增内存→FSDP+no checkpointing=最优!

7. **Training-Decision Law**: 小模型→DDP/FSDP≤2GPU / 7B LoRA→单GPU / 7B全参数→NVLink→RTX 4090不适合!
   → → → RTX 4090最优=推理 / 训练=小模型+LoRA / 大模型训练→NVLink必需!

## 关键参考

- DDP/FSDP: AllReduce/AllGather/ReduceScatter → 通信瓶颈 → PCIe灾难
- BF16训练: RTX 4090原生 → 1.51x → 最佳
- FP8 TE: 1.48-1.59x → B≥4 → fused kernel必需
- Gradient accumulation: micro_batch×accum=等效B → 小GPU桥梁
- LoRA: r=8 → 99.6%参数省 → 单GPU可行
- verl: colocation+async rollout+GRPO → RTX 4090推荐
- FSDP benchmark: results/fsdp2_scaling_benchmark_4090.json → 0.46x灾难