# DeepSpeed 最新发展 (2026年6月) 深度阅读

# DeepSpeed 最新发展 (2026年6月) 深度阅读

> 2026-06-16 | 源码: deepspeed/deepspeed GitHub | 最新PR: #7771/#7997/#7999/#7979
> 核心: ZenFlow Adam修复 + singleton MoE优化 + SDMA allgather + dynamic offload兼容
> ★★★ DeepSpeed持续活跃开发，但RTX 4090单GPU场景仍不如LoRA+compile

## 1. ★★★★★ 2026年关键合并PR

### 1.1 PR #7771: ZenFlow Adam集成修复 (2026-06-12)

```
★★★★ PR #7771 — Fix: ZenFlow Adam integration for updated PyTorch backward flow

→ 修复ZenFlow与DeepSpeed Adam/AdamW optimizers的集成
→ 跟随PR #7759 — PyTorch backward执行模型改变
→ 快速修复 → 恢复correct behavior under new loss.backward() flow
→ 2026-06-12 merged

★★★★ RTX 4090影响:
→ ZenFlow → 生成式 → 巟作流调度 → RTX 4090单GPU不太用ZenFlow
→ 但修复说明DeepSpeed在积极维护与PyTorch最新版本的兼容性
```

### 1.2 PR #7997: ★★★★★ Singleton MoE Collectives优化 (2026-05-13)
```
★★★★★ PR #7997 — Optimize singleton MoE collectives
→ ep_size=1时 → MoE all-to-all和all_reduce都是identity operations → 不需要!
→ 跳过MOELayer._AllToAll.apply → 跳过top-k capacity all_reduce(MAX)
→ 保持existing collective paths unchanged for non-singleton expert-parallel groups
→ ★★★★★ 性能提升: late-step timing 13s → 0.864s (15x speedup!)

★★★★★ 这就是之前发现的Megatron #5203 singleton PG bug的DeepSpeed解决方案!
→ DeepSpeed → singleton MoE → skip collectives → 极大优化
→ Megatron → singleton PG → dp_cp_params_list=None → TypeError → CRASH!
→ ★★★★★ 对比: DeepSpeed正确处理了singleton → Megatron没有!

★★★★ RTX 4090影响:
→ singleton MoE → DeepSpeed优化后 → 可以用!
→ 但RTX 4090只有1个GPU → 没有多GPU MoE需求 → 这个优化对RTX 4090不太重要
→ ★★★ 但对多GPU MoE训练 → 非常重要 → DeepSpeed在MoE方面优于Megatron!
```

### 1.3 PR #7999: ★★★★ SDMA Allgather via Mori (2026-05-14)
```
★★★★ PR #7999 — zero3: SDMA allgather via mori (sdma_allgather)
→ ZeRO-3 allgather → 路由mori_cpp.AllGatherIntoTensor → SDMA copy (AMD MI300X)
→ transparent fallback到dist.allgather_fn (RCCL/NCCL) on init failure
→ ★★★ 性能提升: 8x MI300X → +10.85% step time (GPT-7B-ish), +9.93% (Qwen3-32B)

★★★★ RTX 4090影响:
→ SDMA = AMD MI300X专用 → RTX 4090用NCCL → 不受影响
→ 但这个PR展示了DeepSpeed在ZeRO-3通信优化方面的持续投入
```

### 1.4 PR #7979: ★★★ Dynamic Offload兼容Static Offload (2026-04-24)

```
★★★ PR #7979 — Dynamic offload compatible with static optimizer offload
→ Fix #6596 → dynamic offload之前与static optimizer offload不兼容
→ 修复: 两者现在可以共存
→ 2026-04-24 merged

★★★ RTX 4090影响:
→ Dynamic offload → NVMe offload → 可以在训练时offload optimizer到NVMe
→ ★★★ 但需要NVMe → RTX 4090如果有NVMe → 可用
→ 但RTX 4090只有24GB → offload主要用于大模型训练 → 对7B INT4不太需要
```

## 2. ★★★★★ AutoEP — 自动Expert Parallelism (merged #7938)

```
★★★★★ AutoEP PR#7938 merged 2026-06-11 → DeepSpeed 2026最重要新feature!

AutoEP = Automatic Expert Parallelism:
  → deepspeed.initialize()自动检测MoE block → 替换为EP-enabled执行路径
  → 用户只需JSON config → 无需修改模型代码!
  → 当前支持: ZeRO-0/1/2 → ZeRO-3 follow-up (#8060)

★★★★★ 5 preset MoE models:
  | Preset | 模型 | has_shared_experts |
  |--------|------|--------------------|
  | Mixtral | mixtral | False |
  | Qwen3-MoE | qwen3_moe/qwen2_moe | True(shared_expert) |
  | Qwen3.5-MoE | qwen3_5_moe_text | True(shared_expert+gate) |
  | DS-V2 | deepseek_v2 | True(shared_experts) |
  | DS-V3 | deepseek_v3 | True(shared_experts) |

★★★★★ AutoEP+AutoTP folding (#8064 OPEN):
  → TP(attention/dense) + EP(expert) → 同一组ranks共存!
  → Dense: tp*dp → Expert: ep*etp*edp → 不变量: tp*dp == ep*etp*edp
  → ★★★ 8GPU → TP=2+EP=4 → dense 2卡TP + expert 4卡EP → 最优利用!

★★★★★ AutoEP+ZeRO-3 (#8060 OPEN) → 解决double-partitioning:
  → expert params → expert replica group分区 → 不是global DP
  → router params → global DP group分区 → 标准ZeRO-3
  → ★★★★★ 这是正确的解决方案 → 与之前workaround完全不同!

★★★★ RTX 4090 AutoEP可行性:
  → EP=1 → 单GPU → 所有experts同GPU → 无AllToAll → 可行!
  → ★★★ Singleton MoE (#7997) → EP=1 → 15x speedup → AutoEP自动走singleton path!
  → Qwen3-MoE(A0.6B+B4B) + LoRA + AutoEP ZeRO-2 → RTX 4090唯一MoE训练方案!
```

## 3. ★★★★★ Singleton MoE Collectives优化 (#7997)

### 2.1 AutoEP状态

```
★★★ AutoEP → 仍然experimental/roadmap:
→ 没有最新合并PR → 仍在规划中
→ ★★★ RTX 4090: AutoEP需要多GPU EP → 单GPU不需要
→ 但AutoEP是DeepSpeed MoE方向的重要feature → 未来可能影响RTX 4090多GPU MoE训练
```

### 2.2 DeepCompile状态
```
★★★ DeepCompile → 仍然experimental:
→ 没有最新合并PR → 仍在实验阶段
→ ★★★ RTX 4090: DeepCompile需要多GPU → 单GPU不太用
→ 但DeepCompile的graph优化 → 未来可能帮助RTX 4090 compile加速
```

### 2.3 ZeRO-3+ backward refactor
```
★★★ ZeRO-3+ backward refactor → 进行中:
→ ZeRO-3 backward → 逐步refactor → 更清晰的代码结构
→ ★★★ 但没有merged PR → 仍在开发
```

### 2.4 Muon optimizer
```
★★★ Muon optimizer → 新优化器:
→ Momentum-based optimizer → 可能改善训练稳定性
→ ★★★ 还在研究阶段 → 没有production support
```

## 3. ★★★★★ DeepSpeed vs FSDP2 vs Megatron — 2026年对比

```
★★★★★★ 2026年分布式训练框架对比:

| 维度 | DeepSpeed ZeRO-3 | PyTorch FSDP2 | Megatron-LM |
|------|------------------|----------------|-------------|
| 内存 | 3Ψ comm overhead | 2Ψ (33% less) | TP+PP (手动) |
| MoE | ★★★★★ singleton MoE优化 | ★★★ partial | ★★★✗✗ CRASH on single GPU |
| 通信 | SDMA (AMD) + NCCL | NCCL only | NCCL only |
| NVMe offload | ★★★★★ NVMe swap | ✗✗ not supported | ✗✗ not supported |
| LoRA | ✗✗ not in core | ★★★ via NeMo2 | ✗✗ not in core |
| RTX 4090 | ★★★ ZeRO-2+offload | ★★★ compile+LoRA | ★★ NOT viable |
| SM89 | ★★★ partial FP8 | ★★★ BF16 only | ★✗ SM90 exclusive |

★★★★★★ RTX 4090推荐:
  Single GPU → PyTorch compile+LoRA → 最优
  Multi GPU → DeepSpeed ZeRO-2 → 最优
  MoE training → DeepSpeed → 唯一可行的 (singleton MoE优化!)
  Megatron → ✗✗✗ 单GPU crash → NOT viable for RTX 4090
```

## 4. ★★★ RTX 4090影响分析

```
★★★★ RTX 4090 (SM89) DeepSpeed影响:

正面:
  → ZeRO-2 → 多GPU训练 → 有效
  → ZeRO-3 NVMe offload → 大模型训练 → 有效 (需要NVMe)
  → ZeRO-3+ → backward refactor → 更清晰的ZeRO-3代码
  → singleton MoE (#7997) → 多GPU MoE → 15x speedup → 非常重要!

负面:
  → ZeRO-3 MoE conflict → 仍然存在 (singleton优化有帮助但不是完全解决)
  → AutoEP → 仍然experimental → 需要多GPU
  → DeepCompile → 仍然experimental → 需要多GPU
  → 单GPU → ZeRO不如LoRA+compile → DeepSpeed在单GPU场景价值有限

★★★★★★ RTX 4090最优策略:
  训练: PyTorch compile+LoRA → single GPU最优
  训练: DeepSpeed ZeRO-2 → multi GPU最优
  推理: vLLM → serving最优
  MoE: DeepSpeed singleton优化 → 多GPU MoE唯一可行方案
```

## 5. 关键洞察总结

```
★★★★★★ 5个关键洞察:

1. ★★★★★ DeepSpeed singleton MoE (#7997) → 15x speedup → vs Megatron crash (#5203)
   → DeepSpeed正确处理singleton → Megatron没有 → 这是框架质量的关键差异!

2. ★★★★★ SDMA allgather (#7999) → +10.85% step time → AMD MI300X专用
   → 展示了DeepSpeed在通信优化方面的持续投入 → 但NVIDIA GPU不受影响

3. ★★★★ DeepSpeed持续维护PyTorch兼容性 (#7771) → 修复backward flow
   → 说明DeepSpeed在积极维护与最新PyTorch的兼容 → 保障训练稳定性

4. ★★★ ZeRO-3 MoE conflict → singleton优化有帮助但不完全解决
   → 需要继续关注 → 未来可能有完整fix

   → ★★★ RTX 4090单GPU → ZeRO价值有限 → LoRA+compile更优

5. ★★★★★ DeepSpeed vs FSDP2 → 2Ψ vs 3Ψ → FSDP2内存效率更高
   → 但DeepSpeed有NVMe offload → FSDP2没有 → 大模型训练时DeepSpeed可能更有优势
```

## 参考

- PR #7771: Fix: ZenFlow Adam integration for updated PyTorch backward flow
- PR #7997: Optimize singleton MoE collectives (15x speedup!)
- PR #7999: zero3: SDMA allgather via mori (+10.85% on AMD MI300X)
- PR #7979: Dynamic offload compatible with static optimizer offload
- Issue #6596: Dynamic offload compatibility
- PR #7141: ep_size=1 MoE all-to-all behavior
- Megatron #5203: Single-GPU LayerWise optimizer CRASH (singleton PG bug)
- 相关笔记: deepspeed-latest-developments-2026-06-reading.md (已有→更新), deepspeed-zero3-data-flow.md
