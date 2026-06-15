# Inference-Training Integration Pipeline — 跨框架综合阅读

> 2026-06-15 | 综合7框架深度研究 → 推理-训练集成是2025-2026最重要架构趋势
> ★ ★ ★ 核心洞察: GRPO=推理scaling+训练 → rollout_n=推理compute → vLLM是最大受益者
> ★ ★ ★ RTX 4090: rLLM Tinker → 最简推理-训练集成 → 单GPU唯一可行

## 1. 推理-训练集成的本质

```
★ ★ ★ 推理-训练集成 = 推理引擎(生成) + 训练引擎(更新) → 同一系统两个面!

传统分离:
  → 训练框架(DeepSpeed/Megatron) → checkpoint → 推理框架(vLLM/SGLang/TRT-LLM)
  → ★ 训练和推理完全分离 → checkpoint转换 → 4步→部署慢!

RL集成:
  → Rollout引擎(vLLM/SGLang) → 生成responses → Reward → Advantage → Actor更新 → Weight Sync → 下一步
  → ★ ★ 推理和训练在同一loop → 无checkpoint转换 → 直接部署!

★ ★ ★ 为什么这是2025-2026最重要趋势?
  → 推理scaling(推理时compute更多→质量更好) → GRPO rollout_n=推理compute
  → 推理成本将超越训练成本 → 推理引擎需要更高效 → vLLM受益最大
  → RL训练需要推理引擎 → 推理-训练自然融合 → 架构必然演化!
```

## 2. 3种推理-训练集成架构

```
★ ★ ★ 3种集成架构:

1. ★ ★★ In-Process (rLLM Tinker):
   → 同进程 → 无IPC → 无RPC → 无Ray → 最简单!
   → ServiceClient/SamplingClient → GPU-only merge → 零拷贝
   → ★ ★ ★ 单GPU最优 → RTX 4090唯一可行路径!

   数据流:
     prompt → SamplingClient.sample_async → vLLM-like推理 → responses+logprobs
     → rule-based reward → GRPO advantage → PPO loss → LoRA update
     → save_weights → GPU merge → new SamplingClient → ★ 零拷贝!

2. ★ ★ Colocated (verl HYBRID/COLOCATED):
   → Ray actor → 同PG → 同GPU → 不同进程
   → vLLM ServerAdapter → AsyncLLM引擎 → HTTP/ZMQ IPC → 体重同步
   → ★ 适合同GPU但需要RPC → 比in-process慢 → 但更灵活

   数据流:
     prompt → vLLM HTTP API → responses → DataProto → Ray actor → FSDP训练
     → weight sync → ZMQ IPC → vLLM update → ★ 体重传输 overhead!

3. ★ Separated (verl STANDALONE):
   → 不同GPU → NCCL/NIXL → 分布式
   → ★ 需要多GPU → PCIe RTX 4090灾难 → ✗✗✗ 不适用!

   数据流:
     prompt → rollout GPU(vLLM) → responses → DataProto → Ray → 训练GPU(FSDP)
     → weight sync → NCCL broadcast → ★ PCIe带宽瓶颈!
```

## 3. Weight Sync — 推理-训练集成的关键瓶颈

```
★ ★ ★ Weight Sync = 推理引擎和训练引擎之间权重同步 → 核心瓶颈!

7种Weight Sync性能(7B模型):

| 方法 | 耗时 | 传输量 | RTX4090可行? | 场景 |
|------|------|--------|-------------|------|
| ★★★ zero-copy(in-process) | ~0ms | 0 bytes | ★★★✓✓✓ | rLLM Tinker |
| ★★ GPU-only merge | <1ms | ~2.6GB(LoRA) | ✓✓✓ | rLLM Tinker |
| ★ naive(generator→rollout) | ~0ms | ~0 bytes | ✓✓✓ | verl HYBRID |
| CUDA IPC(ZMQ) | ~50ms | ~14GB | ✗(PCIe) | verl COLOCATED |
| NCCL broadcast | ~600ms | ~14GB | ✗✗(PCIe) | verl STANDALONE |
| NIXL RDMA | ~10ms | ~14GB | ✗(需RDMA) | H100集群 |
| sleep/wake | ~300ms | 0(offload) | ✓ | vLLM colocated |

★ ★ ★ RTX 4090最优: in-process zero-copy → 无传输 → 无延迟 → 极简!
  → rLLM Tinker → GPU-only LoRA merge → save_weights → new SamplingClient
  → ★ ★★ 这是最简最快的推理-训练集成路径!

★ ★ ★ Weight Sync优化关键:
  1. LoRA → 只sync adapter(~2.6GB) vs 全模型(~14GB) → ★ 5.4x减少!
  2. bypass_mode → pi_old=rollout logprobs → 省forward → ★ 省compute!
  3. in-process → 无IPC → 无拷贝 → ★ 省延迟!
  4. sleep/wake → offload weights → release GPU → wake → restore → ★ GPU共享!
```

## 4. 推理引擎选择 — 3种rollout引擎对比

```
★ ★ ★ 3种rollout引擎在推理-训练集成中的角色:

| 引擎 | 启动延迟 | 推理吞吐 | prefix复用 | Weight Sync | RTX4090 |
|------|---------|---------|-----------|------------|---------|
| ★★★ vLLM | ~5s(冷启动) | 4,791tok/s INT4 | prefix caching | sleep/wake | ★★★ |
| ★★ SGLang | ~3s | similar | ★★ radix tree | sleep/wake | ★★ |
| ★★ Tinker(in-process) | ~0s | similar | 无 | 零拷贝 | ★★★ |

★ ★ ★ 关键差异:

vLLM:
  → continuous batching → CUDA graph → INT4 → ★ 最高吞吐!
  → sleep/wake → GPU共享 → ★ colocated可行
  → ★ 但weight sync需要sleep→offload→wake→restore → 300ms overhead

SGLang:
  → radix attention → ★★ prefix复用 → GRPO rollout省7x prefill
  → FutureMap → GPU-only relay → ★ 无CPU round-trip
  → ★ 但verl SGLang integration较新 → 可能不稳定

Tinker(in-process):
  → SamplingClient → ★★★ 零拷贝 → 最快weight sync
  → in-process → ★★★ 无RPC → 无IPC → 最简单
  → ★★★ RTX 4090唯一最优路径!

★ ★ ★ 最佳策略:
  → 训练阶段: Tinker(in-process) → GRPO+LoRA → ★ 最快
  → 推理部署: vLLM INT4+EAGLE → ★ 最高吞吐
  → ★ 训练和推理不是同一阶段 → 可以分别最优!
```

## 5. 推理Scaling + GRPO → 同一系统两个面

```
★ ★ ★ ★ ★ 推理scaling和GRPO训练是同一系统两个面:

推理scaling(Best-of-N):
  → 给同一prompt生成N个response → 选最好的 → quality↑
  → ★ 推理compute增加 → 质量更好 → 推理成本↑↑

GRPO训练(rollout_n):
  → 给同一prompt生成rollout_n个response → group-relative advantage → 训练
  → ★ ★ rollout_n=N → 训练compute增加 → 模型更好 → 推理成本↓↓

★ ★ ★ 关键洞察:
  → 推理scaling = 不训练 → 直接用 → 推理compute↑↑ → 但模型不变
  → GRPO训练 = 训练 → 模型变好 → 推理compute↓↓ → ★ 更高效!
  → ★ ★ ★ 推理scaling+GRPO训练 → 两者结合 → 推理scaling提供训练数据 → GRPO优化模型 → 下次推理更快!

RTX 4090最优策略:
  → Phase 1: GRPO训练(rollout_n=8) → 模型变好 → ★ 训练17GB ✓
  → Phase 2: INT4推理 → 变好的模型 → 4,791 tok/s → ★ 推理 ✓
  → Phase 3: EAGLE加速 → 9,088 tok/s → ★★ 推理更快!
  → ★ ★ ★ 训练→推理→再训练 → 循环优化 → 推理scaling和GRPO完美结合!
```

## 6. Production Deployment考虑

```
★ ★ ★ Production Deployment关键考虑:

1. Weight Sync一致性:
   → 训练更新 → 推理引擎必须同步 → 否则stale weights → ★ 安全问题!
   → ★ rLLM Tinker: save_weights → new SamplingClient → 保证一致性
   → ★ verl HYBRID: naive generator → 零拷贝 → 保证一致性
   → ✗ sleep/wake: offload→restore → 地址不变 → ★ 但需要level 2 → 释放全部

2. Rollout质量:
   → rollout_n越大 → advantage越稳定 → 但compute越多
   → ★ rollout_n=8 → 7B模型 → 单GPU17GB ✓ → RTX 4090最优
   → ★ bypass_mode → 省forward → effective 7× compute → 极关键!

3. Reward可靠性:
   → rule-based → 确定性 → 无GPU → ★ 小GPU最优
   → RM → 需要14GB GPU → 24GB只剩9GB → ✗✗✗ RTX 4090不可行
   → ★ ★ GRPO典型用rule-based → math/code → 不需要RM!

4. 部署路径:
   → 训练 → merge LoRA → HF format → INT4 → vLLM → ★ 最简
   → ★ ★ rLLM → save_pretrained → 直接HF → ★★★ 最简路径!
   → vs DeepSpeed → 4步转换(ZeRO→FP32→HF→INT4) → ✗ 最复杂

5. 监控和稳定性:
   → advantage分布监控 → 异常值检测 → gradient clipping
   → KL penalty → 防止偏离ref → ★ kl_loss_coef=0.001推荐
   → ★ entropy_coef → 防止collapse → 需要>0
```

## 7. RTX 4090 完整推理-训练集成路径

```
★ ★ ★ ★ ★ RTX 4090 完整推理-训练集成路径(2026-06-15最终版):

┌───────────────────────────────────────────────────────────┐
│  Phase 1: GRPO Training (GPU, ~17GB)                      │
│  rLLM TinkerBackend + GRPO + LoRA-32 + bypass_mode       │
│  → SamplingClient.sample_async → responses+logprobs       │
│  → rule-based reward → GRPO advantage → PPO loss          │
│  → fused fwd-bwd-optim → LoRA update                     │
│  → save_weights → GPU-only merge → new SamplingClient     │
│  → ★★★ 零拷贝weight sync → 无IPC → 最简最快!           │
└───────────────────────────────────────────────────────────┘
                          │
                          ↓ merge LoRA → save_pretrained → HF format
┌───────────────────────────────────────────────────────────┐
│  Phase 2: INT4 Quantization (CPU, ~1min)                  │
│  HF model → GPTQ/AWQ → INT4 weights + INT8 KV config    │
│  → ★ INT4 = 3.5GB vs BF16 = 14GB → 4x省内存!           │
└───────────────────────────────────────────────────────────┘
                          │
                          ↓ deploy to vLLM
┌───────────────────────────────────────────────────────────┐
│  Phase 3: Production Inference (GPU, ~11GB)               │
│  vLLM INT4 + INT8KV + GQA-8 + prefix caching             │
│  → CUDA graph (FULL decode) → FlashInfer → 4,791 tok/s   │
│  → ★ ★ EAGLE + INT4 → 9,088 tok/s → 8.3x加速!          │
└───────────────────────────────────────────────────────────┘
                          │
                          ↓ evaluate → pass@k
┌───────────────────────────────────────────────────────────┐
│  Phase 4: Evaluation (CPU, warm-pool)                     │
│  rllm eval --attempts N → pass@k → Terminus-2 harness    │
│  → ★ 不占GPU → CPU warm-pool → 最大化GPU利用率!         │
└───────────────────────────────────────────────────────────┘

★ ★ ★ ★ ★ 关键数字:
  训练内存: ~17GB / 24GB → 7GB headroom
  推理内存: ~11GB / 24GB → 13GB headroom
  推理吞吐: 4,791 tok/s → EAGLE→9,088 tok/s
  训练吞吐: ~19,743 tok/s
  Weight sync: 零拷贝 → 0ms → ★★★ 最简!
  总路径: Tinker→merge→HF→INT4→vLLM → ★★★ 极简极快!
```

## 参考资料

- rLLM Tinker: notebook/projects/rllm-tinker-backend-deep-reading.md + rllm-v0.3-terminal-rl-reading.md
- verl HYBRID: notebook/projects/verl-worker-lifecycle-ray-weight-sync-reading.md + verl-grpo-training-loop-internals-reading.md
- vLLM inference: notebook/projects/vllm-v1-gpu-model-runner-reading.md + vllm-v0.23-new-features-reading.md
- GRPO data flow: notebook/projects/verl-grpo-data-flow-reading.md
- RL Training Patterns: notebook/projects/rl-training-design-patterns-comparison.md
- RTX 4090 Quick Ref: notebook/fundamentals/rtx4090-rl-training-quick-reference.md
- Inference-Compute Scaling: notebook/fundamentals/inference-time-compute-scaling.md
- DeepSeek-R1 GRPO: https://arxiv.org/abs/2501.12948
- verl framework: https://github.com/volcengine/verl
