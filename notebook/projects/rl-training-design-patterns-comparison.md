# 7框架 RL Training Design Patterns 综合对比

> 2026-06-15 | 综合7框架源码阅读 → RL训练是2025-2026最有价值方向
> ★ ★ ★ 核心洞察: GRPO>PPO(省50%内存+compute) → RTX 4090唯一可行路径 → 5框架有GRPO支持

## 1. RL训练数据流全景 (5框架×3角色)

```
★ ★ ★ RL训练核心问题: 如何高效完成 rollout→reward→advantage→actor_update→weight_sync 循环?

角色分配:
  Actor: 生成responses + 计算logprobs + PPO/GRPO loss → 需要GPU!
  Critic: 计算value → GAE advantage → PPO需要, GRPO不需要! → ★ ★ 省50%!
  Reference: 计算KL penalty → base model → 可LoRA disable→同GPU → ★ 零额外GPU!

数据流(PPO):
  prompt → actor.generate → responses+logprobs → reward → critic.forward → values
  → GAE(A=G_t+γV_{t+1}-V_t) → actor.update(PPO clip+KL) → critic.update(value loss)
  → weight_sync → next iteration

数据流(GRPO):
  prompt → actor.generate_n → responses+logprobs → reward(outcome)
  → group_normalize(A=(r-μ)/σ) → actor.update(PPO clip+KL) → ★ ★ 无critic!
  → weight_sync → next iteration

★ ★ ★ 关键差异:
  PPO: 2模型(actor+critic) → 2×内存 → 2×forward → RTX 4090 48GB ✗✗✗
  GRPO: 1模型(actor only) → 1×内存 → 1×forward → RTX 4090 17GB ✓✓✓
```

## 2. 5框架GRPO实现对比

```
★ ★ ★ 5框架GRPO实现对比:

| 维度 | verl | rLLM | DeepSpeed | Megatron | vLLM-Ascend |
|------|------|------|-----------|----------|-------------|
| 核心代码 | core_algos.py | process_backend_batch | AutoEP+ZeRO | gpt_grpo recipe | PPO adapter |
| Advantage | 3变体(loop/vectorized/passk) | GRPO→PPO自动映射 | group normalize | group normalize | same as verl |
| Rollout引擎 | vLLM/SGLang | Tinker(SamplingClient) | HF generate | Megatron inference | vLLM-Ascend |
| Weight Sync | 4后端(naive/IPC/NCCL/NIXL) | 零拷贝(GPU merge) | ZeRO gather | NCCL broadcast | HCCL |
| LoRA | 手动配置 | ★ auto-init(rank=32) | LoRAOptimizedLinear | 无原生LoRA | 通过vLLM |
| bypass_mode | ✗(V1可能有) | ★ ★★(pi_old=rollout) | ✗ | ✗ | ✗ |
| 多GPU | Ray分布式 | Tinker单GPU/VerlBackend | ZeRO-2/3 | TP+PP+DP | Ascend DP |
| Memory | 17GB(GRPO+LoRA) | 17GB(GRPO+LoRA) | ~20GB(ZeRO-2) | 32GB(min) | ~20GB |
| RTX4090 | HYBRID+naive | ★ ★★ Tinker+in-process | ZeRO-2+LoRA+CPU | ✗(需8GPU+) | ✗(Ascend) |
```

## 3. Advantage计算 — 3种实现详解

```
★ ★ ★ GRPO advantage 3种实现:

1. GRPO (loop版) — compute_grpo_outcome_advantage:
   → 遍历每个group → 计算mean和std → (r-μ)/σ → singleton n=1→advantage=0!
   → ★ 简单但慢 → O(n×group_size)

2. GRPO_VECTORIZED (推荐) — compute_grpo_vectorized:
   → scatter-add by uid → groupwise normalization → ★ ★ O(n) → 10-100x快!
   → 需要: gen_batch.repeat(interleave=True) → 同prompt连续 → uid索引
   → ★ ★ verl默认推荐 → production用!

3. GRPO_PASSK — 只取best response:
   → pass@k estimator → best reward → advantage=best_reward
   → 用于: pass@k evaluation → 与rLLM --attempts完美对齐

★ Dr.GRPO修正:
   → norm_adv_by_std_in_grpo=False → 不除std → 防止σ=0时梯度消失
   → ★ singleton(n=1): μ=0,σ=1→advantage=raw_reward → 有学习信号!
   → ★ ★ 但n≥4更好 → group variance提供更多信息

★ ★ RTX 4090最优: GRPO_VECTORIZED + rollout_n=8 → group有方差 → 稳定advantage
```

## 4. Weight Sync — 7种机制对比

```
★ ★ ★ 7种Weight Sync机制:

| 机制 | 框架 | 原理 | GPU需求 | RTX4090 |
|------|------|------|---------|---------|
| naive(零拷贝) | verl HYBRID | 同进程→save→sampling_client→直接替换 | 1 | ★ ★★★ 最快! |
| GPU-only merge | rLLM Tinker | LoRA merge→new SamplingClient→零拷贝 | 1 | ★ ★★★ 最快! |
| CUDA IPC | verl COLOCATED | ZMQ+torch IPC→跨进程→同PG | 2+ | ✗(PCIe慢) |
| NCCL | verl STANDALONE | 跨GPU→broadcast→慢 | 2+ | ✗✗(PCIE) |
| NIXL RDMA | verl advanced | RDMA P2P ring→双buffer→零拷贝 | H100+ | ✗(需RDMA) |
| ZeRO AllGather | DeepSpeed | 分区→gather→replace→释放 | 2+ | ✗(PCIe灾难) |
| sleep/wake | vLLM | sleep(level=1/2)→release→wake→restore | 1(colocate) | ✓ |

★ ★ ★ RTX 4090最优: naive(verl HYBRID) 或 GPU-only merge(rLLM Tinker)
  → 同进程→零拷贝→无通信→最快 → 单GPU唯一可行路径!
```

## 5. LoRA在RL中的管理 — 3种模式

```
★ ★ ★ LoRA在RL训练中的3种管理模式:

1. Auto-init (rLLM Tinker):
   → create_lora_training_client_async(rank=32, train_attn/mlp/unembed=True)
   → ★ ★ 全自动! → 无手动配置 → TinkerSDK管理 → 最简单!
   → merge → save_weights_and_get_sampling_client_async → 零拷贝

2. Manual (verl):
   → 手动配置LoRA rank+target_modules → lora_config dict
   → disable_adapter() → ref model → 同worker → ★ 零额外GPU!
   → ★ 但配置复杂 → 需要理解每个参数

3. Native (DeepSpeed):
   → LoRAOptimizedLinear → fuse/unfuse per step → ZeRO兼容
   → ★ ZeRO-3+LoRA有bug → ZeRO-2+LoRA可行
   → vs PEFT → DeepSpeed原生 → 不依赖外部库

★ ★ LoRA与RL的关键交互:
  → actor用adapter → rollout用adapter → 推理和训练同adapter!
  → ref用base model → disable_adapter → 同GPU → ★ 不需要额外GPU!
  → ★ ★ 单adapter → merge → INT4 → vLLM → 4,791 tok/s → 最快推理路径!
  → ★ ★ 多adapter → dynamic → Punica SGMV → ~64MB → 可行但overhead ~2-5%
```

## 6. Rollout引擎 — 3种选择

```
★ ★ ★ 3种Rollout引擎:

1. vLLM (verl default):
   → ServerAdapter(Ray client) + vLLMHttpServer(AsyncLLM)
   → ★ 高吞吐 → continuous batching → CUDA graph → INT4 → 4,791 tok/s
   → 但: sleep/wake模式 → 权重sync需要额外步骤
   → ★ ★ 推理吞吐最优 → 但weight sync稍复杂

2. SGLang (verl alternative):
   → SGLang backend → radix attention → prefix reuse → GRPO rollout省7x prefill
   → ★ ★★ GRPO rollout_n=8 → system prompt KV只1次 → 极大省prefill!
   → 但: verl SGLang integration newer → 可能不稳定

3. Tinker (rLLM in-process):
   → SamplingClient → in-process → 无RPC → 最简单!
   → ★ ★★★ 单GPU最快路径 → 无Ray → 无IPC → 直接GPU操作
   → 但: 只能单GPU → 无法scale → 多GPU需要VerlBackend

★ ★ ★ RTX 4090最优: Tinker(训练) + vLLM/SGLang(推理评估)
  → 训练: TinkerBackend → in-process → 零拷贝 → 最快
  → 推理(部署): vLLM INT4 → continuous batching → 高吞吐
  → ★ 两者不是同一阶段 → 可以分别最优!
```

## 7. RTX 4090 RL训练最优路径 — 最终综合

```
★ ★ ★ ★ ★ RTX 4090 RL训练最优路径(2026-06-15 最终版):

训练阶段:
  Framework: rLLM TinkerBackend (in-process, 零拷贝)
  Algorithm: GRPO (group-relative, no critic, 50%省内存+compute)
  Model: 7B BF16 + LoRA-32 (auto-init, train_attn/mlp/unembed)
  Advantage: GRPO_VECTORIZED (10-100x快, rollout_n=8)
  ★ ★★ bypass_mode=true → pi_old=rollout logprobs → 省1个forward pass!
  ★ ★ fused fwd-bwd-optim → asyncio.gather overlap → GPU pipeline
  Memory: ~17GB (7B BF16=14GB + LoRA+activations=3GB) → 24GB ✓✓✓
  Reward: rule-based (math/code correctness) → 无GPU空间给RM
  Weight sync: GPU-only merge → 零拷贝 → new SamplingClient

部署阶段:
  Step 1: save_pretrained → merge LoRA → HF format
  Step 2: HF → INT4 quantization (GPTQ/AWQ)
  Step 3: vLLM INT4 + INT8KV + GQA-8 + prefix caching → 4,791 tok/s
  Step 4: ★ ★ EAGLE + INT4 → 9,088 tok/s → 8.3x加速!

评估阶段:
  rllm eval → pass@k (--attempts N) → warm-pool → CPU沙盒 → 不占GPU
  ★ ★ 训练用GPU+评估用CPU → 最大化GPU利用率!

★ ★ ★ ★ ★ 完整路径:
  rLLM Tinker+GRPO+LoRA-32+bypass_mode → merge → HF → INT4 vLLM → 4,791 tok/s
  → 或 INT4+EAGLE → 9,088 tok/s → ★ ★★ 极简极快!

★ ★ 备选路径:
  verl HYBRID+naive+GRPO+LoRA → 功能等价 → 但LoRA需手动配置
  → ★ ★ Tinker更简单(auto-init) → 但verl更灵活(多rollout引擎)
```

## 8. 关键设计Pattern — 5个跨框架共性

```
★ ★ ★ 5个跨框架RL训练共性Pattern:

1. ★ ★ ★ GRPO > PPO (2026年趋势):
   → 所有5框架都支持GRPO → PPO在24GB GPU上不可能(48GB内存)
   → ★ ★ 这是2025-2026最重要RL训练范式转变!
   → DeepSeekMath paper → group-relative → 无critic → 省50%

2. ★ ★ LoRA + RL = 完美组合:
   → LoRA减少内存(只训~0.8%参数) + RL需要多轮rollout(内存紧张)
   → ★ LoRA auto-init(rLLM) → 无需手动配置 → 最简单
   → ★ LoRA merge后推理同(无overhead) → 最快部署路径

3. ★ ★ Sleep/Wake + RL = GPU共享:
   → vLLM sleep/wake → rollout→sleep→train→wake→rollout
   → ★ 单GPU可以同时做rollout和training → 时间交替 → 不是空间共享!
   → ★ rLLM Tinker → in-process → 无需sleep/wake → 更简单!

4. ★ ★ Rule-based Reward > RM (小GPU):
   → rule-based → 无需GPU → math correctness / code execution
   → RM → 需要14GB GPU → 24GB只剩9GB → 不够!
   → ★ ★ GRPO典型用rule-based → PPO需要RM(value) → 另一个GRPO优势!

5. ★ ★ Bypass Mode = 省forward pass:
   → pi_old从rollout logprobs → 不需要重新forward → 省1个完整forward!
   → ★ ★ ★ rLLM Tinker默认bypass → verl也可以 → RTX 4090极关键!
   → ★ 省forward=省14GB内存peak(参数从offload)或省时间(7B forward ~0.3s)
```

## 参考资料

- verl GRPO: notebook/projects/verl-grpo-data-flow-reading.md + verl-grpo-training-loop-internals-reading.md
- rLLM Tinker: notebook/projects/rllm-tinker-backend-deep-reading.md + rllm-v0.3-terminal-rl-reading.md
- DeepSpeed AutoEP: notebook/projects/deepspeed-latest-developments-2026-06.md + deepspeed-0.19-features-reading.md
- Megatron GRPO: notebook/projects/megatron-v0.17-latest-reading.md
- vLLM inference: notebook/projects/vllm-v0.23-new-features-reading.md + vllm-mrv2-architecture-reading.md
- RTX 4090 config: notebook/projects/rtx4090-seven-framework-practical-config.md
- 相关: [7框架对比](seven-framework-comparison.md), [训练数据流](seven-framework-training-data-flow-comparison.md)
