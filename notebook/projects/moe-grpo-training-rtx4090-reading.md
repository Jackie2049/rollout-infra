# MoE + GRPO Training — RTX 4090实战分析

> 2026-06-15 | 综合7框架MoE+GRPO交叉点 → RTX 4090 MoE RL训练唯一可行路径
> ★ ★ ★ 核心: MoE GRPO → LoRA(active params only)+INT4(experts)+rule-based reward → ~9GB可行

## 1. MoE + GRPO的挑战

```
★ ★ ★ MoE模型做GRPO训练 → 3重挑战:

Challenge 1: 内存 → MoE模型参数比Dense大得多!
  → Qwen3-MoE: 15B total → 3B active → inactive experts ~12GB
  → → ★ 所有expert weights必须加载(EP=1 → 同GPU) → 内存爆炸!
  → → → 3B active BF16=6GB + 12B experts=24GB → ✗✗✗ 30GB→超24GB!

Challenge 2: Expert routing → rollout时不同token选择不同expert
  → → ★ rollout_n=8 → 每个prompt→8个response → routing不同 → KV不共享!
  → → → ★ 但: system prompt部分 → 所有response共享 → prefix caching部分有效

Challenge 3: Weight sync → MoE LoRA weights更多 → 更大的sync
  → → ★ LoRA on active+expert → 更多adapter weights → 更多GPU内存
  → → → ★★ 但: LoRA只训active → expert frozen → sync更少 → ★ 还是可行!
```

## 2. 4种MoE GRPO架构

```
★ ★ ★ 4种MoE + GRPO训练架构:

1. ★★★ rLLM Tinker + LoRA(active only):
   → TinkerBackend → in-process → GRPO → LoRA only on active params
   → → ★★★ expert weights frozen → 不参与gradient → 内存最小!
   → → → active LoRA: ~0.3GB(rank=8) → total ~6.3GB active → ★★★ 最省!
   → → → 但: 所有expert weights still need加载(推理用) → ~18GB total → tight

2. ★★ verl HYBRID + freeze_moe_router + LoRA:
   → ★ verl GRPO + freeze_moe_router → router不训练 → stability
   → → ★ router_replay → rollout和训练用相同routing → ★ consistency
   → → → ★★ LoRA on active → expert frozen → same as rLLM approach

3. ★ DeepSpeed AutoEP + ZeRO-2 + LoRA:
   → AutoEP preset → EP=1 → ZeRO-2 + LoRA → MoE specific config
   → → ★ AutoEP零代码 → 但DeepSpeed无GRPO → 需自实现RL loop
   → → → ★★ 可以做static training → 但不能做dynamic RL → 价值有限

4. ★★ Megatron GRPO + MoE(InferenceTopKRouter):
   → ★ DynamicInferenceEngine → InferenceTopKRouter + dense_output=True → FlashInfer兼容
   → → ★ refit weight swap → 同GPU做ref → 但无LoRA → 需手动注入
   → → → ★★★ RTX 4090: overkill → 但MoE inference engine可用
```

## 3. 内存解决方案 — 3种策略

```
★ ★ ★ ★★ MoE on RTX 4090 内存解决方案(24GB):

Strategy 1: ★★★ LoRA(active only) + INT4(experts) + CPU_Adam
  → active weights: 3B BF16 → ~6GB
  → expert weights: 12B INT4 → ~3GB → ★★★ 4x compression!
  → LoRA trainable: ~0.3GB (rank=8 on active params)
  → optimizer: ~0GB (CPU_Adam → offload)
  → activations: ~2GB
  → total: ~11.3GB → ★★★ 12.7GB headroom → ✓✓✓ 可行!

Strategy 2: ★★ Expert offloading (CPU when not active)
  → active weights: 3B BF16 → ~6GB (always on GPU)
  → expert weights: offload to CPU → only load needed experts → ~2GB GPU at any time
  → → ★★ expert routing → 每token只需2-3 experts → load on-demand
  → → → 但: CPU→GPU transfer → latency → ★ decode慢 → 推理不实用!
  → → → ★★ 训练时: expert weights需要但可以batch load → 前向时全部加载 → 还是30GB

Strategy 3: ★★★ Full INT4 → active INT4 + expert INT4
  → active INT4: ~1.5GB
  → expert INT4: ~3GB
  → total: ~4.5GB → ★★★★ 19.5GB headroom → ✓✓✓✓✓ 充裕!
  → → ★ 但: INT4训练精度不对 → BF16是唯一正确训练精度
  → → → ★★★★ 解决: BF16 active(for training) + INT4 experts(for compute only)
  → → → → → ★★★★ 最佳: mixed precision → active BF16 + experts INT4

★ ★ ★ ★ ★ RTX 4090最优MoE GRPO配置:
  → LoRA(active, rank=8, BF16) + INT4(expert weights, compute only)
  → → rLLM Tinker + GRPO + bypass_mode + rule-based reward
  → → total: ~11GB → 13GB headroom → ✓✓✓
  → → ★★★ 推理: merge LoRA → INT4 all → vLLM → 可行!
```

## 4. MoE GRPO rollout特殊性

```
★ ★ ★ MoE GRPO rollout的特殊性:

Rollout generation (MoE推理):
  → 每token → router选择2-3 experts → 不同token不同expert组合
  → → ★ MoE inference比Dense慢 → 但active params更少 → ★ faster per-token!
  → → → Qwen3-MoE 3B active → vs Dense 7B → ★ ★★ 每token更快的compute!

★ ★ Prefix caching + MoE:
  → system prompt → 所有rollout共享 → prefix caching → ★ 有效!
  → → ★ 但: system prompt后 → routing开始分化 → prefix失效
  → → → ★ vLLM: hash不含LoRA ID → prefix caching可能不兼容
  → → → ★ SGLang: radix attention → prefix+LoRA → ★ 更安全

★ ★ Expert routing在GRPO中的角色:
  → 训练: freeze_moe_router → router权重不变 → ★ stability
  → → verl: router_replay → rollout和训练用相同routing → ★ consistency
  → → ★★★ 但: 如果训练LoRA → active params变化 → routing可能变化
  → → → ★★★ 最佳: LoRA只训attention → 不训router → routing稳定

★ ★ MoE + CUDA graph:
  → InferenceTopKRouter → @torch.compile → dense_output=True → FlashInfer grouped GEMM兼容
  → → ★★★ 但: expert selection是dynamic → CUDA graph不能pre-capture所有组合
  → → → ★ Megatron solution: local CG + persist CG → inference/training separate scopes
  → → → ★★★ vLLM MoE: CG for dense ops → expert GEMM dynamic → mixed strategy
```

## 5. MoE INT4 Triton fallback — RTX 4090关键

```
★ ★ ★ ★★ MoE INT4 Triton fallback对RTX 4090极关键:

问题: MoE intermediate_size % 128 != 0 → Marlin不支持 → ✗ crash!

★ ★ ★ PR#43731 TritonW4A16LinearKernel → 安全网:
  → TritonW4A16 → lowest priority → 只在Marlin拒绝时激活
  → → ★★★ DS-V2-Lite (K=704) → 之前crash → 现在loadable on RTX 4090
  → → ★★★ Qwen2-MoE (K=2496) → 之前crash → 现在loadable
  → → Triton ~2-5x slower than Marlin → 但之前完全不可能
  → → ★★ RTX 4090: ~5-15% overall throughput hit → ★ 可接受!

★ ★ ★ Qwen3-MoE alignment检查:
  → 需要检查Qwen3-MoE的expert intermediate_size是否128-aligned
  → → 如果aligned → ★ Marlin kernel → 最快
  → → 如果non-aligned → ★ Triton fallback → slower but working
  → → ★★★ 关键: 只有down_proj可能non-aligned → gate_up_proj通常aligned
  → → → ★ overall impact取决于non-aligned layers比例 → 通常很小

★ ★ MoE INT4 inference config:
  → INT4 experts → Triton fallback for non-aligned → ★ 可行!
  → INT8 KV → 5GB vs 10GB → ★ 多轮可行!
  → prefix caching → system prompt KV → ★ rollout共享!
  → → ★★★ 配置: vLLM INT4 + INT8KV + prefix caching + enforce_eager(for Triton)
```

## 6. verl MoE GRPO 特殊路径

```
★ ★ ★ verl MoE GRPO特殊路径(verl-grpo-source-reading + moe-serving):

1. ★★ freeze_moe_router → router权重freeze → 不训练 → stability
   → ★★★ verl特化 → 其他框架没有 → ★ MoE GRPO stability!

2. ★★ router_replay → rollout和训练用相同routing → consistency
   → → ★ 训练时用rollout时的routing → 不re-compute → ★ faster!
   → → ★★★ 但: 如果LoRA改变active params → routing可能变化 → 需要re-compute

3. ★ RoutedExpertsCapturer → RL+推理桥梁 → slot_mapping索引
   → → ★★★ 但: 与KV connector不兼容 → GRPO不能PD分离!
   → → → ★★★ MoE GRPO: 不能prefill/decode分离 → 只能同GPU → RTX 4090条件!

4. ★★ verl MoE + DeepSpeed backend → AutoEP + ZeRO-2 + LoRA
   → → ★ verl训练loop + DeepSpeed AutoEP → MoE特化
   → → → ★★★ 但: still需要GRPO → verl has GRPO → ★ 可行组合!

★ ★ ★ verl MoE GRPO RTX 4090最优配置:
  → verl HYBRID + GRPO + freeze_moe_router + router_replay
  → → LoRA rank=8 on attention only → expert frozen
  → → ★★★ 但: verl更复杂 → rLLM Tinker更简 → ★ 选更简的!
```

## 7. Megatron MoE GRPO — Inference Engine

```
★ ★ ★ Megatron MoE inference engine for GRPO rollout:

DynamicInferenceEngine:
  → ★ InferenceTopKRouter → @torch.compile → dense_output=True → FlashInfer grouped GEMM兼容
  → → ★ CUDA graph + local CG + persist CG → inference/training separate scopes
  → → ★★ MoE CUDA graph问题 → expert selection dynamic → 不能pre-capture所有组合
  → → → ★ Megatron: local CG → dense ops captured → expert GEMM dynamic → mixed strategy

★ ★ RTX 4090 Megatron MoE GRPO:
  → DynamicInferenceEngine ✓ → NCCL ✓ → FlashInfer ✓
  → ✗ NVLS → ✗ TMA → ✗ FP8 → ✗ DeepEP → SM89不支持
  → → ★★★ inference engine可用 → 但训练侧overkill → rLLM Tinker更简
  → → → ★★★ 最佳: Megatron inference做rollout → rLLM Tinker做训练 → 各最优!
  → → → → 但: 两框架integration复杂 → ★★★ 实际仍用单一框架 → rLLM Tinker
```

## 8. RTX 4090 MoE GRPO最优路径

```
★ ★ ★ ★ ★ RTX 4090 MoE GRPO最优路径:

Phase 1: GRPO训练 (rLLM Tinker, ~11GB)
  ┌──────────────────────────────────────┐
  │ rllm workflow --backend tinker        │
  │ --algorithm grpo                      │
  │ --lora-rank 8                          │  ← LoRA只训active params!
  │ --lora-target attention_only           │  ← 不训router/expert!
  │ --bypass-mode true                     │  ← 省ref forward!
  │ --reward-function my_math_reward.py   │  ← rule-based → CPU
  │ --freeze-moe-router true               │  ← router不训练 → stability
  │ ★ active BF16=6GB + INT4 experts=3GB  │
  │ ★ LoRA=0.3GB + activations=2GB         │
  │ ★ total ~11.3GB → 12.7GB headroom ✓✓✓ │
  └──────────────────────────────────────┘

Phase 2: Merge LoRA → HF format
  → save_weights → LoRA merge into active weights → ★ 等价全参训练

Phase 3: INT4 quantization → ALL weights (active + experts)
  → GPTQ → INT4 → active ~1.5GB + experts ~3GB → ★ total ~4.5GB!

Phase 4: vLLM INT4 MoE serving (~4.5-6GB)
  → INT4 + INT8KV + Triton fallback(non-aligned) + prefix caching
  → ★ enforce_eager for Triton kernel stability → until CG confirmed
  → ★★★ MoE inference on RTX 4090 → ✓✓✓ 4x compression → 内存充裕!

Phase 5: EAGLE → ★★★ 可能不支持MoE → 需要确认 → 如果支持→更快!

★ ★ ★ ★ ★ 关键数字:
  → 训练: ~11GB / 24GB → 12.7GB headroom → ✓✓✓
  → 推理: ~4.5-6GB / 24GB → 18GB headroom → ✓✓✓✓✓
  → ★★★ LoRA(active)+INT4(experts) → mixed precision → 各最优!
  → ★★★ MoE active params更少 → per-token更快 → ★ inference advantage!
```

## 参考资料

- MoE serving patterns: notebook/projects/moe-serving-architecture-patterns-reading.md
- INT4 Triton fallback: notebook/projects/vllm-int4-triton-fallback-reading.md
- verl GRPO source: notebook/projects/verl-grpo-source-reading.md
- DeepSpeed AutoEP: notebook/projects/deepspeed-autoep-rtx4090-moe-practical.md
- Megatron GRPO SM89: notebook/projects/megatron-grpo-sm89-reading.md
- rLLM Tinker: notebook/projects/rllm-tinker-backend-deep-reading.md
- RTX 4090 decision tree: notebook/fundamentals/rtx4090-rl-training-decision-tree.md
- RTX 4090 ADR: notebook/fundamentals/rtx4090-architecture-decision-records.md
