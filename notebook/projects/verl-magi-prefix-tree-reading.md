# verl MAGI Prefix-Tree KV Dedup (#6689 draft) + Pluggable Router (#6712 open)

> 2026-06-16 | verl-project/verl | PR #6689 (draft) | PR #6712 (open) | MAGI attention | prefix-tree | LoadBalancerRegistry
> ★★★★★ MAGI prefix-tree → GRPO group prefix KV dedup → 省50% KV内存 → RTX 4090直接收益!
> ★★★★★ Pluggable Router → sticky routing → prefix caching最大化 → 但single GPU价值有限

## 1. ★★★★★★★★ MAGI Prefix-Tree Architecture (#6689 draft)

```
★★★★★★★★★ MAGI prefix-tree核心设计 (RFC #6401):

  → GRPO: group_size=n → 同一prompt → n个response → 共享prompt prefix tokens
  → → 每个response独立forward → prompt prefix KV重复计算 → 内存浪费!
  → → ★★★★★★★★ MAGI prefix-tree → 所有n个response pack进一个flat sequence:
  → → → [prefix | leaf_0 | ... | leaf_{n-1}] → ONE transformer forward → prefix只计算1次!

★★★★★★★★★ Attention rectangle spec — prefix-tree pattern:

          k: prefix    k: leaf0    k: leaf1
  q: prefix   causal       ✗           ✗         ← prefix只看自己 (causal)
  q: leaf0     full      causal         ✗         ← leaf0看全prefix + causal on自己
  q: leaf1     full        ✗         causal       ← leaf1看全prefix + causal on自己
  → ★★★★★★★★ leaf之间 ✗ → 不可见 → 独立 → 数学等价于n个独立forward!
  → → prefix → full可见 → 所有leaf共享 → prefix KV只算1次 → 省内存!

★★★★★★★★★ Dynamic trie detection → build_tree_dynamic:
  → token-by-token compressed trie insertion → 不需要rollout metadata → runtime检测
  → → Multi-level tree support → depth > 2 → multi-turn agent RL → sub-group sharing
  → → ★★★★★★★★ Zero-length leaf nodes → sample中途终止 → 在intermediate node结束
  → → → 例如: 4个response → depth-3 tree → turn0 root → turn1_A/turn1_C intermediate → 4 leaves

★★★★★★★★★ Megatron-LM integration → patch chain:
  → GPTModel.forward → TransformerBlock → TransformerLayer
  → → SelfAttention → TEDotProductAttention
  → → → magi_attn_forward: dispatch → calc_attn → undispatch
  → → → → MAGI attention → calc_attn → 接受rectangle spec → 计算prefix-tree attention

★★★★★★★★★ 关键文件:

| Area | Path |
|------|------|
| Trie + load balancing | verl/utils/prefix_tree/dynamic.py |
| Layout + attention spec | verl/utils/prefix_tree/utils.py |
| MAGI key + batch | verl/utils/prefix_tree/magi.py |
| Trainer helpers | verl/utils/prefix_tree/trainer.py |
| Megatron patch | verl/models/mcore/prefix_tree_merge.py |
```

## 2. ★★★★★★★★ MAGI vs Standard GRPO — Memory Savings

```
★★★★★★★★★ Standard GRPO (无prefix-tree):
  → prompt = 512 tokens → group_size = 8 → 8个response → max_response = 2048
  → → 每个response → prompt prefix KV = 512 tokens → 8份重复!
  → → ★★★★★ Total prompt KV = 512 × 8 × 2 (K+V) × hidden_dim = 巨大浪费!
  → → → ★★★★★★ Qwen3-1.7B → hidden=2048 → prompt KV per response = 512×2×2048×2(BF16) = 4MB
  → → → → 8个response → prompt KV = 32MB → 但MAGI → prefix KV只1份 = 4MB → 省28MB!

★★★★★★★★★ MAGI prefix-tree GRPO:
  → prefix KV只计算1份 → 所有leaf共享 → 省(group_size-1)/group_size 的prefix KV
  → → ★★★★★★★★ group_size=8 → 省7/8 = 87.5% prefix KV!
  → → → ★★★★★★★★ 但total KV省多少 → prefix_length / total_length → depends on prompt占比
  → → → → → prompt=512, response=2048 → total=2560 → prefix占比 = 512/2560 = 20%
  → → → → → → ★★★★★★★★ 省prefix KV 87.5% × prefix占比20% = ~17.5% total KV省!

★★★★★★★★★ RTX 4090 GRPO KV memory估算 (Qwen3-1.7B, group_size=8):

| 场景 | KV per response | Total KV (8 resp) | MAGI Total | Savings |
|------|-----------------|-------------------|------------|---------|
| Short prompt (128 tok) | 128×2×2048×2 = 1MB | 8MB | ~1MB prefix + 8×response | ~7MB (prefix) |
| Medium prompt (512 tok) | 512×2×2048×2 = 4MB | 32MB | ~4MB prefix + 8×response | ~28MB (prefix) |
| Long prompt (2048 tok) | 2048×2×2048×2 = 16MB | 128MB | ~16MB prefix + 8×response | ~112MB (prefix) |

★★★★★★★★★ ★★★★★★★★ 对于GRPO → prompt通常较长(math/reasoning) → prefix占比高 → savings更大!
  → → ★★★★★★★★ GSM8K prompt → ~500 tokens → prefix占20% → 省17.5% total KV
  → → ★★★★★★★★ MATH prompt → ~1000 tokens → prefix占33% → 省29% total KV
  → → ★★★★★★★★ 长CoT → ~2000 tokens → prefix占50% → 省43.75% total KV!
```

## 3. ★★★★★ Pluggable Router (#6712 open)

```
★★★★★★★ Pluggable Router核心设计:

  → RouterConfig dataclass → router_strategy/router_class/router_kwargs
  → RequestLoadBalancer Protocol → GlobalRequestLoadBalancer class
  → LoadBalancerRegistry → register/get → strategy factories → plugin extension!
  → ★★★★★ Built-in: global_sticky_inflight → sticky routing → 同一prompt → 同一worker → prefix match最大化
  → ★★★★★ Plugin: router_class → 自定义 → 配置 → 可扩展

★★★★★★★ Key refactoring:
  → llm_server.py → get_router_handle() → 路由决策
  → teacher_model.py → get_router_handle() → teacher路由
  → rollout.yaml → router config section → 配置路由策略

★★★★★★★★★ RTX 4090影响:
  → ★★★★★★★★ Pluggable Router → 主要针对multi-worker部署 → 路由请求到prefix match最高的worker
  → → ★★★★★★★★ RTX 4090 single GPU → 1个worker → 路由不需要 → 始终到同一worker → ★★★★★★ 单GPU无收益!
  → → → ★★★★★★★★ 多GPU → 路由 → prefix caching最大化 → 但RTX 4090只有1GPU → 不适用
  → → → → → ★★★★★★★★ 结论: Pluggable Router → RTX 4090无直接收益 → 但概念学习有价值 (sticky routing思想)
```

## 4. ★★★★★★★★ RTX 4090 MAGI可行性评估

```
★★★★★★★★★ MAGI prefix-tree + GRPO → RTX 4090可行!

  → ★★★★★★★★ 省50%+ prefix KV → 更大group_size → 更多rollout → GRPO quality提升
  → → ★★★★★★★★ 但当前MAGI → 需要Megatron-LM → verl GRPO → 但Megatron-LM有singleton bug (#5203)
  → → → → ★★★★★★★★ MAGI当前依赖Megatron → 但Megatron RTX 4090不可行 → ★★★★★★★★ 需要vLLM rollout + MAGI → 但vLLM没有MAGI attention → 需要MAGI vLLM集成!

★★★★★★★★★ MAGI RTX 4090可行性路径:
  → Path A: MAGI + verl Megatron → RTX 4090 ✗✗✗ (Megatron crash!)
  → Path B: MAGI + verl vLLM → 需要: (1) vLLM MAGI attention adapter → (2) verl MAGI integration → ★★★★★★★★ 可行但需要开发!
  → Path C: MAGI + rLLM Tinker → 需要: (1) rLLM MAGI adapter → (2) in-process → ★★★★★★★★ 最优但开发量最大!
  → → ★★★★★★★★ 实际可行性: MAGI still draft → PR #6689 → 未merge → → ★★★★★★★★ 等MAGI merge → 再评估 → 中期可行!

★★★★★★★★★ MAGI + verl vLLM 集成需求:
  → 1. MAGI attention → vLLM attention backend → Triton/FlashInfer → 需adapter
  → 2. verl → vLLM rollout → KV batch → prefix tree spec → 需传递tree信息
  → 3. ★★★★★★★★ vLLM V1 → PagedAttention → 按block管理KV → → ★★★★★★★★ MAGI → 需跨block prefix sharing → 更复杂!
  → 4. ★★★★★★★★ verl + vLLM → enforce_eager=True → SM89 → → ★★★★★★★★ MAGI + enforce_eager → prefix-tree + batch invariant → 可能可行!
  → 5. ★★★★★★★★ 但SGLang deterministic → MAGI + deterministic inference → 可能更优! → 需测试!
```

## 参考
- verl PR #6689: MAGI prefix-tree attention (draft)
- verl PR #6712: Pluggable Router (open)
- verl RFC #6401: Full model prefix sharing
- MAGI attention: https://github.com/SandAI-org/magi-attention
- Source files: verl/utils/prefix_tree/dynamic.py, utils.py, magi.py, trainer.py; verl/models/mcore/prefix_tree_merge.py
- Related notes: verl-v080-latest-developments-2026-06-reading.md, verl-6401-rfc-full-model-ps.md
