# verl v0.8.0最新发展 — 2026年6月深度分析

> 2026-06-16 | verl-project/verl | v0.8.0 | 最新PR合并状态
> ★★★★★ 21+个新PR — 多个重大新feature — VeOmni, SGLang PD, ReMax, IcePop, FlowGRPO
> ★★★★★ MRv2 handling = ZERO → 确认 → VLLM_USE_V2_MODEL_RUNNER=0仍然MUST

## 1. ★★★★★ VeOmni Engine (#6072 merged)

```
★★★★★★★ VeOmni = 全新训练backend → on-policy distillation (OPD)!

  → PR #6072 merged → 新engine backend
  → 支持: on-policy distillation → 训练策略模仿teacher模型
  → 多-teacher OPD (#6051 merged) → 多个teacher同时指导
  → Fused top-K distillation kernel (#5997 merged) → VeOmni专用
  → ★★★★★ 值模型(head) → critic-less → 简化训练
  → ★★★ 不同于GRPO → OPD是不同训练范式 → teacher-guided
```

## 2. ★★★★★ SGLang PD Disaggregation (#6117 merged)

```
★★★★★★★ SGLang PD disaggregated rollout → prefill:decode分离!

  → PR #6117 merged → SGLang作为rollout backend → prefill:decode不对称布局
  → Prefill服务器 → 处理长context → 生成KV cache → 传输到decode
  → Decode服务器 → 每步生成1 token → 低延迟优先
  → ★★★★★ SGLang作为vLLM替代rollout → vLLM vs SGLang选择!
  → ★★★★★ PD disaggregation → KV transfer across servers → Ascend已有实践

★★★★★★★ SGLang rollout vs vLLM rollout:
  → vLLM → 现有默认 → V1 engine → continuous batching
  → SGLang → NEW option → RadixAttention → PD disaggregation → 可选
  → → verl现在有3种rollout backend: vLLM, SGLang, HF

★★★★ RTX 4090影响:
  → PD disaggregation → 多GPU → RTX 4090单GPU → 不直接受益
  → 但SGLang rollout → 单GPU也可用 → RadixAttention → prefix caching更好
  → ★★★★★ RTX 4090可尝试SGLang rollout → prefix caching → GRPO training加速
```

## 3. ★★★★★ ReMax Algorithm (#6340 merged)

```
★★★★★★★ ReMax = greedy baseline + TransferQueue sync trainer!

  → PR #6340 merged → 新算法: greedy rollout作为baseline
  → 每个prompt生成1个greedy sample (temperature=0) → 作为baseline
  → vs GRPO: group of n samples → group baseline → variance更高
  → → ReMax: per-sample greedy baseline → variance更低 → 更稳定!
  → ★★★★★ TransferQueue sync trainer → 必须用sync → greedy rollout需要同步
  → ★★★★★ greedy baseline = zero cost → 不需要额外sampling → 只改temperature

★★★★★★★ ReMax vs GRPO vs CPPO:

| 算法 | Baseline | Variance | Trust Region | RTX 4090 |
|------|----------|----------|--------------|----------|
| GRPO | group mean | medium | clip ratio (uniform) | ✓ |
| ReMax | greedy rollout | ★★★★★ low | implicit (greedy anchor) | ★★★★★ ✓ |
| CPPO | rollout μ (bypass) | medium | prefix-weighted | ✓ (with bypass) |
| PPO | value function | low | clip ratio | ✗ (need critic) |

★★★★★★★ RTX 4090推荐:
  → ReMax → 最简单 → greedy baseline → TransferQueue sync → zero extra cost
  → CPPO → best trust region → 但需要bypass_mode → more config
  → → ★★★★★ 简单场景 → ReMax → 长CoT → CPPO → 灵活选择!
```

## 4. ★★★★★ IcePop Rollout Correction (#5722 merged)

```
★★★★★★★ IcePop = Importance Sampling (IS) rollout correction!

  → PR #5722 merged → IS correction for off-policy training
  → 类似bypass_mode → 但更精确 → IS weights → 修正off-policy偏差
  → rollout_is_threshold → IS weight裁剪 → 防止过大IS权重 → 稳定训练
  → ★★★★★ 比PPO KL penalty更灵活 → 可配置IS阈值
  → ★★★★★ 可与GRPO组合 → GRPO + IS correction → 更稳定

★★★★★★★ IcePop vs bypass_mode:
  → bypass_mode → skip ref model → old_log_probs = rollout_log_probs → 简化
  → IcePop → IS correction → 更精确 → 但需要额外计算IS weights
  → → ★★★★★ bypass_mode更简单 → IcePop更精确 → RTX 4090推荐bypass_mode!
```

## 5. ★★★★★ FlowGRPO (#5616 merged)

```
★★★★★★★ FlowGRPO = Diffusion RL + vLLM-Omni rollout!

  → PR #5616 merged → diffusion model GRPO训练
  → vLLM-Omni作为rollout backend → 图像生成模型
  → FlowGRPO → 连续token flow → 不离散 → diffusion trajectory
  → ★★★★★ 新范式 → 不是传统LLM GRPO → 但架构创新 → RL for generation

★★★★ RTX 4090影响:
  → Diffusion RL → 需要图像生成模型 → RTX 4090可能可行 → 但需要diffusion inference
  → ★★★ 实验性 → 不是RTX 4090首要方向 → 但架构值得了解
```

## 6. ★★★★★ ContinuousToken Builder (#6720/#6721 open)

```
★★★★★★★ ContinuousToken = Agent Loop → append-only runtime token stream!

  → PR #6720 (open): ContinuousTokenBuilderCore → 追加token → 不重建整个prompt
  → PR #6721 (open): ContinuousTokenBuilder → agent loop → multi-turn RL
  → ★★★★★ 这是agent RL的基础设施 → 多轮对话 → 持续追加 → 不需要每次重建

★★★★★★★ Agent Loop RL训练:
  → 传统GRPO → 1 prompt → 1 response → 1 reward → 训练
  → Agent Loop → 1 prompt → tool call → observe → next response → ...
  → → ★★★★★ 多轮交互 → 需要append-only token stream → ContinuousToken解决!
  → → verl正在构建agent RL基础设施 → 未来可做tool-use RL

★★★★ RTX 4090影响:
  → Agent RL → 多轮 → KV cache需要持久化 → RTX 4090 memory挑战
  → 但append-only → 不重建 → 省计算 → 内存友好 → ★★★ 中期可能可行
```

## 7. ★★★★★ Prefix-Tree MAGI Attention (#6689 draft)

```
★★★★★★★ Prefix-tree MAGI = shared-prefix KV deduplication for GRPO!

  → PR #6689 (draft) → trie-based shared prefix detection
  → GRPO group_size=n → n个responses共享同一prompt → prefix tokens完全相同
  → → ★★★★★ KV cache deduplication → 共享prefix只存一次 → 省n倍prefix KV memory!
  → → Flat token layout → attention rectangle spec → MAGI calc_attn

★★★★★★★ RTX 4090影响:
  → GRPO group_size=8 → 8个response → 共享prefix → dedup → 省大量KV memory!
  → → ★★★★★★ RTX 4090 GRPO KV memory可能减半! → 最直接RTX 4090收益!
  → → 但还在draft阶段 → 未merged → 中期关注

★★★★★★★ vs vLLM HMA:
  → vLLM HMA → per-attention-type KV grouping → 前缀缓存自动 → 不需要trie
  → MAGI prefix-tree → explicit trie → GRPO-specific → dedup group prefix
  → → ★★★★★ 两者互补 → HMA for inference, MAGI for training!
```

## 8. ★★★★★ Pluggable Router (#6712 open)

```
★★★★★★★ LoadBalancerRegistry → 可插拔负载均衡路由器!

  → PR #6712 (open) → LoadBalancerRegistry → 注册不同路由策略
  → global_sticky_inflight → sticky routing → 同request → 同worker → prefix caching
  → plugin_extension → 可注册自定义路由策略
  → ★★★★★ 与SGLang PD disaggregation结合 → prefill→decode路由

★★★★ RTX 4090影响:
  → 单GPU → 不需要路由 → 但了解路由架构对多GPU理解有价值
  → ★★★ 未来多GPU时 → sticky routing → prefix caching最大化 → throughput优化
```

## 9. ★★★★★ 其他重要PR

```
★★★★ SGLang LoRA support (#5564 merged, #5769 merged):
  → merge=True → merge LoRA adapter into base weights → inference efficiency
  → merge=False → native LoRA → keep adapter separate → training flexibility
  → ★★★★★ RTX 4090: merge=True for inference, merge=False for training

★★★★ vLLM upgraded to 0.20.2 (#6393 merged):
  → 从v0.18.0 → v0.20.2 → 2个版本跳!
  → → ★★★★★ 但verl pinned v0.20.2 → vLLM v0.23.0 → 3版本差!
  → → MRv2 default → verl ZERO handling → VLLM_USE_V2_MODEL_RUNNER=0 STILL MUST!

★★★★ CUDA graph config fix (#6620 merged):
  → cuda_graph_sizes config logic → vLLM >= 0.11.1 compatible
  → ★★★★★ RTX 4090: SM89 batch invariance → enforce_eager=True → graphs禁用

★★★★ ZMQ DP>1 fix (#6091 merged):
  → ZMQ handles for DP>1 → data parallel多进程 → handle正确传递

★★★★ TRT-LLM rollout (#5631 merged, #6730 open):
  → TensorRT-LLM作为rollout backend → inference optimization
  → ★★★ RTX 4090: TRT-LLM INT8 → inference加速 → 但不是训练backend

★★★★ Megatron-Bridge renamed (#6072):
  → mbridge → megatron-compose → naming争议 → 见Megatron Lite reading
```

## 10. ★★★★★ RTX 4090 GRPO综合影响

```
★★★★★★★ RTX 4090 GRPO最新推荐 (2026-06更新):

★★★★★★ 算法选择:
  → 简单场景 → ReMax (#6340) → greedy baseline → 最简单 → zero extra cost
  → 长CoT → CPPO (#6731) → prefix-weighted trust region → best for math
  → 标准场景 → GRPO + bypass_mode → 经典 → 稳定
  → ★★★★★ ReMax > CPPO > GRPO (by simplicity) → CPPO > ReMax > GRPO (by trust region)

★★★★★★ 框架选择:
  → rLLM Tinker #1 → simplest → in-process → bypass default
  → verl ReMax #2 → TransferQueue sync → greedy baseline → simple
  → verl CPPO #2 → best trust region → more config → long CoT
  → verl + Lite #2.5 → mid-term → need small model + LoRA

★★★★★★ MRv2 still MUST disable:
  → verl v0.8.0 pins vLLM v0.20.2 → MRv2 default for dense models
  → ZERO MRv2 handling → VLLM_USE_V2_MODEL_RUNNER=0 → MUST!
  → ★★★★★ 这个问题不会自动解决 → verl需要PR来处理MRv2

★★★★★★★ 新OSS贡献机会:
  → verl MRv2 handling → detect MRv2 → set compilation_config → PR!
  → MAGI prefix-tree → GRPO prefix dedup → RTX 4090 KV省一半 → 关注draft
  → BudgetRefiner → vLLM SLO-aware scheduling → GPU也可用 → upstream贡献
```

## 参考
- verl-project/verl GitHub: https://github.com/verl-project/verl (formerly volcengine/verl)
- verl v0.8.0 release notes
- PR #6072: VeOmni engine
- PR #6117: SGLang PD disaggregation rollout
- PR #6340: ReMax algorithm
- PR #5722: IcePop IS correction
- PR #5616: FlowGRPO diffusion RL
- PR #6720/#6721: ContinuousToken builder
- PR #6689: Prefix-tree MAGI attention (draft)
- PR #6712: Pluggable router
- PR #6717: Tinker Worker Primitives (merged)
- PR #6731: CPPO algorithm (open)
- 相关笔记: verl-grpo-bypass-mode-reading.md, verl-tinker-worker-primitives-reading.md, verl-cppo-algorithm-reading.md
