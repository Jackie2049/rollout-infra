# 7框架 RL训练决策树 — RTX 4090实战指南

> 2026-06-15 | 综合57+深度阅读 → 5层决策 → 30秒内选最优方案
> ★ ★ ★ 核心决策: rLLM Tinker+GRPO+LoRA → RTX 4090唯一最优 → 其他都是辅助

## 0. 3秒快速决策

```
Q: RTX 4090上做RL训练?
A: ★★★ rLLM Tinker + GRPO + LoRA-32 + bypass_mode + rule-based reward
   → 17GB / 24GB → 7GB headroom → ✓✓✓
   → 训练后: merge→INT4→vLLM→4,791 tok/s→EAGLE→9,088 tok/s

Q: 为什么要GRPO而不是PPO?
A: PPO需要critic → ~270GB → ✗✗✗ RTX 4090完全不可能
   GRPO不需要critic → 17GB → ✓✓✓ 唯一可行

Q: 为什么要LoRA?
A: 全参数训练 → 7B 14GB weights + 14GB gradients + 14GB optimizer = 42GB → ✗✗✗
   LoRA-32 → 0.8% params → 2.6GB → +0.5GB optim → ✓✓✓ 17GB feasible

Q: 为什么要rule-based reward?
A: RM需要14GB GPU → 24GB只剩9GB → ✗✗✗ 不可行
   rule-based → CPU → 无GPU → ✓✓✓ math/code reward → 确定性

Q: 为什么要Tinker而不是verl?
A: Tinker → in-process → 零拷贝 → 无IPC → 无Ray → 最简最快
   verl → Ray → IPC overhead → 更复杂 → 但多GPU更好
   ★ 单GPU → Tinker最优; 多GPU → verl最优
```

## 1. 5层决策树

```
Layer 1: 选择RL算法
┌─────────────────────────────────────────────┐
│ Q: 选PPO还是GRPO?                            │
│                                               │
│ PPO: critic+GAE+value loss → 270GB → ✗✗✗    │
│ GRPO: group-relative → no critic → 17GB → ✓  │
│                                               │
│ ★★★ Answer: GRPO → 唯一可行                    │
│                                               │
│ GRPO变体选择:                                  │
│ → GRPO_VECTORIZED ★★★(推荐→10-100x快)         │
│ → GRPO (loop → 简单但慢)                       │
│ → GRPO_PASSK (只best response → eval对齐)      │
│ → GDPO (per-dimension normalize → rule dict)  │
│                                               │
│ ★ Dr.GRPO: norm_adv_by_std_in_grpo=False      │
│ → 防止小group σ=0 → 梯度消失                   │
│ ★★★ 推荐: GRPO_VECTORIZED + Dr.GRPO            │
└─────────────────────────────────────────────┘
                    ↓
Layer 2: 选择训练参数策略
┌─────────────────────────────────────────────┐
│ Q: 全参数还是LoRA?                             │
│                                               │
│ 全参数: 42GB → ✗✗✗ 不可能                      │
│ LoRA-32: 17GB → ✓✓✓ 可行                      │
│ LoRA-16: 16GB → ✓✓ 更省但可能不够              │
│ LoRA-64: 18GB → ✓ 但接近极限                    │
│                                               │
│ ★★★ Answer: LoRA-32 → 0.8% params → 最佳平衡  │
│                                               │
│ LoRA配置:                                      │
│ → target: attn+mlp+unembed → auto-init(rLLM) │
│ → rank: 32 → ~2.6GB                           │
│ → merge后推理等价全参数 → ★ 无损失!              │
└─────────────────────────────────────────────┘
                    ↓
Layer 3: 选择Reward策略
┌─────────────────────────────────────────────┐
│ Q: 选什么reward?                               │
│                                               │
│ Reward Model(RM): 14GB GPU → ✗✗✗ 不可行      │
│ Colocate RM(sleep/wake): 内存太紧 → ✗          │
│ Rule-based: CPU → ✓✓✓ 无GPU开销               │
│                                               │
│ ★★★ Answer: Rule-based → 唯一可行              │
│                                               │
│ Rule-based类型:                                │
│ → Math reward(正确/错误 → 0/1)                 │
│ → Code reward(执行测试 → pass/fail)             │
│ → Format reward(格式检查 → 部分分数)             │
│ → ★ Outcome reward=最后token非零 → GRPO对齐    │
│                                               │
│ ★ GDPO: rule-based返回dict → per-dimension    │
│ → 可用! → 防止dominant信号淹没                  │
└─────────────────────────────────────────────┘
                    ↓
Layer 4: 选择框架
┌─────────────────────────────────────────────┐
│ Q: 选哪个框架?                                 │
│                                               │
│ ★★★ rLLM Tinker → ★★★★ 最优!                 │
│   → in-process → 零拷贝 → 最简最快             │
│   → LoRA auto-init → 5 loss → fused           │
│   → bypass_mode → 省forward → ~40% faster     │
│                                               │
│ ★★ verl HYBRID → ★★★ 备选                     │
│   → Ray colocated → naive generator → 可行    │
│   → 更成熟/更大社区 → 但更复杂                   │
│   → GRPO_VECTORIZED → 10-100x faster advantage │
│                                               │
│ ★ DeepSpeed ZeRO-2 → ★ 辅助                   │
│   → 无GRPO → 需自实现 → 太复杂                  │
│   → LoRA+CPU_Adam → 可行但不是RL最优            │
│                                               │
│ ✗ Megatron → 推理引擎可做rollout → 太heavy     │
│ ✗ PyTorch compile → 可overlay → 需自实现       │
│ ✗ MindIE → NPU only → RTX 4090不适用           │
│                                               │
│ ★★★ Answer: rLLM Tinker → 单GPU最优            │
│ ★★ 备选: verl HYBRID → 多GPU时切换             │
└─────────────────────────────────────────────┘
                    ↓
Layer 5: 选择推理部署
┌─────────────────────────────────────────────┐
│ Q: 训练后如何部署推理?                          │
│                                               │
│ ★★★ INT4 + INT8KV → 唯一可行推理精度            │
│   → 3.5GB weights + 5GB KV → 11GB → 13GB head │
│   → 4,791 tok/s baseline                      │
│                                               │
│ ★★★★ INT4 + EAGLE → 极速推理                   │
│   → +0.5GB draft → 11.5GB → 12.5GB headroom  │
│   → 9,088 tok/s → ★★★★ 极速!                  │
│                                               │
│ ✗ BF16推理 → 24GB精确占满 → ✗✗✗               │
│                                               │
│ ★★★ Answer: INT4+INT8KV+EAGLE → 最优推理       │
│                                               │
│ 部署路径:                                      │
│ → rLLM训练 → merge LoRA → save HF             │
│ → GPTQ量化 → INT4 → vLLM部署                   │
│ → EAGLE + INT4 → 9,088 tok/s                   │
│                                               │
│ ★ ★ ★ ★ ★ 完整路径:                            │
│ GRPO Tinker(17GB) → merge → HF → INT4 →      │
│ vLLM(4,791) → EAGLE(9,088) → pass@k(CPU)     │
└─────────────────────────────────────────────┘
```

## 2. 内存预算详细表

```
| 场景 | weights | LoRA | KV | graph | buf | optim | act | total | headroom | 可行? |
|------|---------|------|-----|-------|-----|-------|-----|-------|----------|-------|
| ★★★ GRPO Tinker训练(BF16) | 14 | 2.6 | - | - | 0.5 | 0.5 | 2 | 17.1 | 6.9 | ✓✓✓ |
| ★★ verl HYBRID训练(BF16) | 14 | 2.6 | - | - | 0.5 | 0.5 | 2 | 17.1+1 | 5.9 | ✓✓ |
| PPO训练(BF16) | 14 | 2.6 | - | - | 0.5 | 0.5 | - | 270+ | ✗✗✗ | ✗✗✗ |
| ★★★ INT4推理 | 3.5 | - | 5 | 2 | 0.5 | - | - | 11 | 13 | ✓✓✓ |
| ★★★★ INT4+EAGLE推理 | 3.5 | - | 5 | 2 | 0.5+0.5 | - | - | 11.5 | 12.5 | ✓✓✓ |
| ✗ BF16推理 | 14 | - | 10 | - | - | - | - | 24 | 0 | ✗✗✗ |
| DeepSpeed Z2+LoRA | 14 | 2.6 | - | - | 0.5 | 0(CPU) | 2 | 17.1 | 6.9 | ✓ |
```

## 3. 7框架RL训练对比

```
| 框架 | GRPO支持 | LoRA支持 | 单GPU可行 | Weight Sync | RTX4090最优? |
|------|---------|---------|----------|------------|-------------|
| ★★★ rLLM Tinker | ✓✓✓(5 loss) | ✓✓✓(auto-init) | ✓✓✓(17GB) | zero-copy(0ms) | ★★★★ |
| ★★ verl HYBRID | ✓✓✓(15 advantage) | ✓✓(manual) | ✓✓(17GB+1) | naive(0ms) | ★★★ |
| ★ DeepSpeed Z2 | ✗(需自实现) | ✓✓(ZeRO-2) | ✓(17GB) | checkpoint(慢) | ★ |
| ★ Megatron | ✗(太heavy) | ✓(manual) | 推理✓(INT4) | - | 推理★ |
| ★ PyTorch | ✗(需自实现) | ✓(manual) | ✓(17GB) | - | compile overlay★ |
| ✗ MindIE | ✗(NPU only) | ✗ | ✗✗✗ | - | ✗✗✗ |
```

## 4. Weight Sync详细对比

```
| 方法 | 耗时 | 传输量 | RTX4090 | 适用框架 |
|------|------|--------|---------|---------|
| ★★★ zero-copy(in-process) | ~0ms | 0 bytes | ★★★✓✓✓ | rLLM Tinker |
| ★ GPU-only merge | <1ms | ~2.6GB(LoRA) | ✓✓✓ | rLLM Tinker |
| ★ naive(generator→rollout) | ~0ms | ~0 bytes | ✓✓✓ | verl HYBRID |
| sleep/wake | ~300ms | 0(offload) | ✓ | vLLM colocated |
| CUDA IPC(ZMQ) | ~50ms | ~14GB | ✗(PCIe) | verl COLOCATED |
| NCCL broadcast | ~600ms | ~14GB | ✗✗(PCIe) | verl STANDALONE |
| NIXL RDMA | ~10ms | ~14GB | ✗(需RDMA) | H100集群 |

★ ★ ★ RTX 4090最优: zero-copy(Tinker) → 0ms → 0 bytes → ★★★ 最简最快!
```

## 5. bypass_mode详解

```
★ ★ ★ bypass_mode = 省forward pass → ~40% faster → 极关键!

正常PPO/GRPO training需要2个forward:
  1. pi_old = ref_model(prompt+response) → old_log_probs → 用于PPO ratio
  2. pi_new = actor_model(prompt+response) → new_log_probs → 用于gradient

★ ★ ★ bypass_mode=true → 3行代码:
  1. apply_bypass_mode(old_log_probs=rollout_log_probs)
  2. → pi_old = rollout_log_probs → ★ 已经在rollout时计算过!
  3. → 省掉1个forward pass → ★ ~40% time saved!

rLLM Tinker: bypass_mode=true → ★★★ 默认最优
verl: bypass_mode available → ★ 但需要配置

★ ★ ★ RTX 4090: bypass_mode=true → 必须! → 算力有限→每个forward都很贵→省40%极关键!
```

## 6. 完整实战路径

```
★ ★ ★ ★ ★ RTX 4090 RL训练→推理→评估完整路径:

Phase 1: GRPO Training (rLLM Tinker, ~17GB)
  ┌──────────────────────────────────────┐
  │ rllm workflow --backend tinker        │
  │ --algorithm grpo                      │
  │ --lora-rank 32                        │
  │ --bypass-mode true                    │
  │ --reward-function my_math_reward.py   │
  │ --rollout-n 8                         │
  │ --gen-batch-size 8                    │
  │ ★ ~19,743 tok/s training throughput   │
  │ ★ zero-copy weight sync → 0ms         │
  └──────────────────────────────────────┘

Phase 2: Merge LoRA (GPU, <1ms)
  ┌──────────────────────────────────────┐
  │ save_weights_and_get_sampling_client  │
  │ → GPU-only LoRA merge into base       │
  │ ★ 等价全参数训练 → 无损失              │
  │ ★ new SamplingClient → 继续可用       │
  └──────────────────────────────────────┘

Phase 3: Save HF Format (CPU, <1min)
  ┌──────────────────────────────────────┐
  │ save_pretrained → BF16 weights        │
  │ ★ 直接HF → 最简路径                   │
  └──────────────────────────────────────┘

Phase 4: INT4 Quantization (CPU, ~1min)
  ┌──────────────────────────────────────┐
  │ auto-gptq --bits 4 --group-size 128  │
  │ → INT4 weights 3.5GB                  │
  │ → INT8 KV config → 5GB KV             │
  │ ★ 4x compression → 推理可控           │
  └──────────────────────────────────────┘

Phase 5: vLLM INT4 Inference (~11GB)
  ┌──────────────────────────────────────┐
  │ vllm serve model --quantization gptq  │
  │ --kv-cache-dtype fp8_e4m3             │
  │ --enable-prefix-caching               │
  │ --enable-chunked-prefill              │
  │ ★ 4,791 tok/s baseline                │
  │ ★ EAGLE → 9,088 tok/s → ★★★★ 极速!  │
  └──────────────────────────────────────┘

Phase 6: Evaluation (CPU warm-pool)
  ┌──────────────────────────────────────┐
  │ rllm eval --attempts 8 → pass@k       │
  │ ★ CPU warm-pool → 不占GPU             │
  │ ★ pass@k ↔ GRPO rollout_n → 完美对齐 │
  └──────────────────────────────────────┘

★ ★ ★ 循环优化: 评估→再训练→再推理→再评估 → 推理scaling+GRPO完美结合!
```

## 7. 避坑清单

```
★ ★ ★ RTX 4090 RL训练常见错误 → 绝对避免!

✗✗✗ PPO训练 → 270GB → 完全不可能 → ★ 用GRPO
✗✗✗ 全参数训练 → 42GB → 不可能 → ★ 用LoRA-32
✗✗✗ RM reward → 14GB → 不可能 → ★ 用rule-based
✗✗✗ BF16推理 → 24GB → 无headroom → ★ 用INT4+INT8KV
✗✗✗ 分布式训练(TP/PP) → PCIe灾难 → ★ 单GPU
✗✗✗ FSDP2/ZeRO-3 → PCIe不可用 → ★ LoRA no distributed
✗✗✗ Megatron TP>1 → PCIe不可用 → ★ single GPU only
✗✗✗ MindIE → NPU专用 → ★ 用vLLM GPU版
✗✗✗ FP8训练 → SM89不支持 → ★ BF16训练唯一正确
✗✗✗ FP8 E5M2 → SM89不支持 → ★ INT4推理
✗✗✗ NVLS/TMA → SM89不支持 → ★ NCCL+FlashInfer
✗✗✗ GRPO cudagraph exponential sizing → +13.6%peak → ★ linear sizing
✗✗✗ GRPO small group(n=1) → σ=0 → gradient vanish → ★ Dr.GRPO或rollout_n≥4
✗✗✗ LoRA+prefix caching → vLLM hash不含LoRA ID → ★ 单adapter merge后可用
```

## 参考资料

- rLLM Tinker: notebook/projects/rllm-tinker-backend-deep-reading.md
- verl GRPO: notebook/projects/verl-grpo-training-loop-internals-reading.md
- verl Reward: notebook/projects/verl-reward-integration-reading.md
- RTX 4090 Quick Ref: notebook/fundamentals/rtx4090-rl-training-quick-reference.md
- RTX 4090 Config Card: notebook/fundamentals/rtx4090-grpo-training-config-card.md
- Inference-Training: notebook/projects/inference-training-integration-pipeline-reading.md
- Quantization Crosspoint: notebook/projects/quantization-inference-training-crosspoint-reading.md
- RL Design Patterns: notebook/projects/rl-training-design-patterns-comparison.md
- RTX 4090 Practical Config: notebook/projects/rtx4090-seven-framework-practical-config.md
