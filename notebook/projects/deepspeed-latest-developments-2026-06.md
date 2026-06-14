# DeepSpeed 最新进展 (2026年6月)

> 2026-06-15 | 源码: GitHub microsoft/DeepSpeed recent merged PRs
> 核心: AutoEP(自动Expert Parallelism)已merged → ZeRO-0/1/2+MoE支持 → TorchTitan内核 → 4个preset模型

## 1. AutoEP — 自动Expert Parallelism (PR #7938, merged 2026-06-11)

**这是DeepSpeed 2026年最重要的新feature!**

```
AutoEP = Automatic Expert Parallelism
→ deepspeed.initialize()自动检测MoE block → 替换为EP-enabled执行路径
→ 用户只需配置 → 无需修改模型代码!
→ 当前支持: ZeRO-0/1/2 → ZeRO-3作为follow-up

支持的preset模型:
  → Mixtral
  → Qwen3-MoE
  → DeepSeek-V2
  → DeepSeek-V3

自定义模型支持:
  → moe_layer_pattern → 正则匹配MoE层
  → router_pattern → 路由器子模块名
  → expert_pattern → 专家子模块名
  → expert_w1/w2/w3 → 专家权重名/布局
  → num_experts_attr, top_k_attr → 配置属性
  → optional shared-expert → 共享专家
```

**AutoEP执行路径**:
```
1. detect → deepspeed.initialize()扫描模型 → 找到MoE blocks → 替换为AutoEPMoELayer
2. route → EP router执行 → top-k routing → token assignment
3. reorder → token按expert assignment重排 → permutation indices
4. dispatch → all-to-all dispatch across EP group (when autoep_size > 1)
5. compute → local grouped expert compute → grouped-GEMM → 单次kernel launch!
6. combine → all-to-all combine → restore original token order
7. merge → shared experts merge (if model has them) → DeepSeek-V3 shared expert
```

**TorchTitan内核** — AutoEP大量借鉴TorchTitan:
```
ep_router.py ← TorchTitan TokenChoiceTopKRouter
ep_experts.py ← TorchTitan GroupedExperts + grouped-GEMM
ep_kernels.py ← TorchTitan TokenReorderer + generate_permute_indices + Triton fill-indices

→ grouped-GEMM → 单次kernel launch → 比逐expert GEMM快!
→ Triton fill-indices → 高效token permutation → GPU kernel!
→ 与DeepEP的asymmetric不同 → AutoEP用symmetric all-to-all → 但grouped-GEMM弥补!
```

**AutoEP vs Megatron MoE**:
```
AutoEP (DeepSpeed):
  → 自动检测MoE → 无需修改模型代码 → HuggingFace兼容!
  → all-to-all → symmetric → NCCL标准
  → ZeRO-0/1/2 → 参数不分片 → EP只分专家
  → 没有DeepEP integration → 但有grouped-GEMM

Megatron MoE:
  → 手动配置 → 需改模型代码 → FlexTokenDispatcher → DeepEP/HybridEP
  → DeepEP → asymmetric → 4.6x faster than NCCL all-to-all!
  → ZeRO无 → FSDP2/TP → 参数分片
  → TMA+IBGDA → HybridEP → 更少SM → GB200优化

→ AutoEP优势: 零代码修改 → HuggingFace兼容 → 简单配置!
→ Megatron优势: DeepEP → asymmetric → 更快通信 → H100集群专用!
→ RTX 4090: AutoEP ZeRO-2+EP → 可行(但EP>1=跨GPU → PCIe限制!)
→ H100集群: Megatron DeepEP → 更优 → asymmetric+FP8!
```

## 2. 其他近期merged PRs

```
#8063 (6/14): ZeRO-3 grad dtype → register_z3_param → fp32精度
#8056 (6/10): Consistent fp32 grads flow → 梯度精度一致性 → BF16+fp32 optimizer
#8055 (6/10): Transpose kernel bank conflicts fix → shared memory indexing → 性能提升
#8043 (6/10): ZeRO-3 coordinator trace invalidation → hook re-registration → 正确prefetch
#8042 (6/4): ZenFlow ZeRO-3 selective optimizer crash → NVMe offload修复
#8044 (6/9): Remove AutoSP assertion → AutoSP不再限制Transformers版本
#8047 (6/6): Muon lr overrides → 新optimizer支持

版本: 0.19.1 (5/30 release) → AutoEP在0.20.0将包含!
```

## 3. AutoEP对RTX 4090的影响

```
RTX 4090 + AutoEP:
  → EP=1 → 单GPU → experts不分片 → 无分布式通信 → 可行!
  → EP>1 → 需跨GPU → PCIe带宽限制 → 7B 8GPU=0.46x → 不推荐!
  → AutoEP ZeRO-2 → optimizer分片 → 但参数不分片 → 内存减少有限
  → AutoEP ZeRO-0 → 无分片 → 最简单 → 但内存最大!

→ RTX 4090最优: EP=1 + LoRA + ZeRO-2(optimizer分片) → 7B MoE训练可行!
→ 但: 7B MoE = 8 experts × 7B MLP → 专家MLP远小于7B → 实际可能~1-2GB
→ Mixtral 8x7B: ~47GB → RTX 4090不可行 → 但Qwen3-MoE(A0.6B+B4B) → 可行!

→ ★ Qwen3-MoE + LoRA + AutoEP ZeRO-2 → RTX 4090唯一MoE训练可行方案!
```

## 4. 关键设计洞察

```
1. AutoEP = TorchTitan内核 + DeepSpeed集成层 → 借鉴而非重写!
   → TorchTitan: routing+grouped-GEMM+permutation → 已验证的MoE实现
   → DeepSpeed: HF模型检测+自动替换+checkpoint → 零代码修改!
   → 这表明: AI infra框架越来越多互相借鉴 → 不重复造轮子!

2. grouped-GEMM → 单次kernel → 比逐expert GEMM更快!
   → 传统: for expert in experts: GEMM → 8次launch → 8次memory round-trip
   → grouped: 一次launch所有experts → 1次memory → 8x fewer launches!
   → ATB(MindIE)也有grouped_matmul → 跨平台共识!

3. AutoEP ZeRO-3作为follow-up → EP+参数分片 → 最复杂 → 未完成
   → ZeRO-3 + EP → 参数分片+专家分片 → 2层分片 → 通信复杂!
   → 当前只有ZeRO-0/1/2 → 参数不分片 → 通信只有EP all-to-all → 简单!

4. DeepSpeed版本策略: 0.19.x稳定 → 0.20.x新feature(AutoEP)
   → 这表明DeepSpeed仍然活跃开发 → 不是"过时"框架!
   → AutoEP是2026最重要新增 → 使DeepSpeed成为MoE训练选择之一!

5. ZenFlow持续修复 → 仍在alpha → 但方向正确 → overlap grad+optimizer
   → #8042: NVMe offload crash → selective optimizer → 修复
   → #8056: fp32 grads flow → 精度一致性 → 基础稳定性改进
   → ZenFlow + AutoEP → 未来: EP + overlap → 极致性能!
```

---

Sources:
- GitHub microsoft/DeepSpeed PR #7938 (AutoEP, merged 2026-06-11)
- GitHub microsoft/DeepSpeed recent merged PRs (#8063, #8056, #8055, #8043, #8042)
- DeepSpeed 0.19.1 release (2026-05-30)
- notebook/projects/deepep-megatron-integration-latest.md (DeepEP vs Megatron)
- notebook/projects/mindie-atb-kernel-architecture-reading.md (grouped_matmul)
