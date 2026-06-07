# LLM Distributed Training Frameworks 2025-2026 — 前沿进展

> 2026-06-07 | TorchPrime统一框架+FSDP2+torch.compile逼近Megatron-LM(85-95%吞吐)

## 1. TorchPrime — PyTorch统一分布式训练框架

```
TorchPrime是PyTorch 2025年推出的统一分布式训练框架:
- 整合 FSDP2 + Tensor Parallelism + Pipeline Parallelism
- 统一API → 不再需要在Megatron-LM/FSDP/DeepSpeed之间选择
- 基于torch.compile → JIT编译分片计算图 → 自动kernel fusion
- 取代torchtitan(早期FSDP1参考项目) → 更先进的架构

关键目标:
  1. 统一并行策略: FSDP(DP) + TP + PP under single API
  2. 自动性能优化: torch.compile减少Python开销 + GPU kernel fusion
  3. 易用性: 简化配置 vs Megatron-LM复杂config
  4. 灵活性: 支持组合不同并行策略(MoE的EP+DP等)
```

## 2. FSDP2 vs FSDP1 — 关键改进

```
FSDP2 (2024-2025) vs FSDP1:

| 特性                | FSDP1          | FSDP2              |
|---------------------|----------------|--------------------|
| 分片API              | 复杂           | 更clean sharding   |
| torch.compile集成   | 部分           | **原生支持**        |
| Checkpoint          | 慢             | **更快**            |
| Mixed parallelism   | DP only        | **DP+TP(实验性PP)**|
| 内存效率             | 好             | **更好**            |
| 吞吐(7B模型)        | baseline       | **1.2-1.5x over FSDP1** |

核心改进:
  1. torch.compile → 自动融合操作 → 减少Python→dispatch桥梁开销
  2. 更好的sharding API → 分片粒度更灵活 → 内存效率更高
  3. TP支持 → 之前只有DP → 现在DP+TP组合 → 接近Megatron-LM能力
  4. Faster checkpoint → 训练中断恢复更快 → 大模型训练必需
```

## 3. FSDP2 + torch.compile vs Megatron-LM 性能对比

```
2025 benchmark数据:

| 场景               | FSDP2+compile | Megatron-LM | 比率    |
|--------------------|---------------|-------------|--------|
| 单节点(8 GPU) 7B   | ~85-95%       | 100%        | 接近持平|
| 单节点(8 GPU) 70B  | ~85-90%       | 100%        | TP=8时接近|
| 多节点(64 GPU)     | ~80-90%       | 100%        | 仍有差距|
| 多节点(256+ GPU)   | ~80-85%       | 100%        | PP优势明显|

→ FSDP2+compile差距从之前的20-40%缩小到5-15%!
→ 单节点场景几乎持平 → 多节点仍有差距(PP不够成熟)

差距来源:
  1. Megatron-LM手动CUDA kernel fusion → FSDP2用torch.compile(自动)
  2. Megatron-LM PP+SP成熟 → FSDP2 PP实验性,SP不完整
  3. Megatron-LM通信-计算重叠更优化 → FSDP2仍在改进
  4. 256+GPU场景 → Megatron-LM的pipeline+overlap更高效

→ 大趋势: PyTorch-native → 逐步接近 → 但极端scale仍需Megatron-LM
```

## 4. DeepSeek训练系统 — 671B @ $5.6M

```
DeepSeek-V3训练系统(2024-2025):

核心创新:
1. **DualPipe**: 新型pipeline parallelism → 更细粒度计算-通信重叠 → 通信延迟≈0
2. **FP8 MoE训练**: MoE用FP8 → 内存↓+速度2x → 精度损失极小
3. **Cross-node EP**: MoE expert跨节点 → 优化All-to-All → GPU空闲时间最小
4. **HAI-LLM**: 自定义训练框架 → DP+TP+SP+PP(DualPipe) → 全组合

关键数据:
  模型: 671B/37B active = 18x稀疏
  GPU: 2,048 H800 (80GB each)
  训练时间: ~2个月(14.8T tokens pretraining)
  **成本: $5,576,000** ← 10x cheaper than Llama 3.1 405B!

为什么这么便宜?
  1. FP8 → 2x计算效率 → 相当于免费翻倍GPU
  2. DualPipe → 通信≈0 → GPU利用率接近100%
  3. MoE → 仅37B active → 计算量远小于671B dense
  4. GRPO → 无critic模型 → RLHF省50%GPU

→ DeepSeek证明: 算法+系统创新 → 成本10x降低 → 不需要更多GPU!
```

## 5. RL训练框架对比 (2025)

```
| 框架    | RL算法         | Rollout后端     | 特点                 |
|---------|---------------|----------------|---------------------|
| verl    | PPO/GRPO等14种 | vLLM/SGLang/TRT| 最灵活,注册表式扩展   |
| OpenRLHF| PPO/DPO       | vLLM           | 简洁但算法少          |
| TRL     | PPO/DPO       | HF generate    | 最简单但吞吐最低      |
| DeepSeek| GRPO          | 自研           | 最高效(无critic+FP8) |

→ 2025趋势:
  1. GRPO取代PPO成为主流(critic省50%GPU)
  2. vLLM/SGLang async rollout成为标配(HF generate太慢)
  3. MoE+EP+FP8 → RL训练成本继续降低
  4. Prefix sharing → rollout throughput 2-6x提升
```

## 6. 对AI Infra工程师的启示

```
技能优先级(2025-2026):

1. **FSDP2 + torch.compile** (P0 — 必须掌握)
   → 2025主流训练框架 → 易用+性能接近Megatron-LM
   → torch.compile graph breaks调试 → 新的核心技能

2. **Megatron-LM TP/PP** (P1 — 大scale仍需要)
   → 256+GPU训练仍然首选 → TP+PP+SP组合
   → 但PyTorch-native趋势 → 理解原理但不必深度使用

3. **MoE Serving + EP** (P1 — DeepSeek推动的主流)
   → 671B/37B架构 → EP+FP8 → 新的生产范式
   → All-to-All优化 → DeepEP → 核心技能

4. **GRPO Training Pipeline** (P0 — RLHF新标准)
   → PPO→GRPO转变 → 理解advantage estimation差异
   → prefix sharing + rollout throughput → 实操技能

5. **GPU集群管理** (P2 — 工程支撑)
   → Slurm/Kubernetes → 多job调度 → GPU利用率优化
   → checkpoint management → failure recovery → 2个月训练必需

→ 核心转变:
  2024: Megatron-LM主导 → 手动kernel → 复杂config
  2025: PyTorch-native → torch.compile → 简化config
  2026: TorchPrime统一 → 自动优化 → 一个API搞定

→ AI Infra工程师价值 = 理解底层原理 + 掌握最新框架 + 实操优化
```

## 参考资料

- PyTorch FSDP2 RFC & Blog (2024-2025)
- DeepSeek-V3 Technical Report (arxiv 2412.19437)
- TorchPrime GitHub discussions
- verl framework (GitHub: volcengine/verl)
- NVIDIA Megatron-LM repository