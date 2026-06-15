# RTX 4090 RL Training Quick Reference Card

> 2026-06-15 | 7框架知识合成 → RTX 4090 RL训练3秒决策
> ★ ★ ★ 核心结论: GRPO+LoRA+Tinker+bypass_mode → 单GPU唯一可行RL路径

## 决策树 (3秒)

```
RTX 4090 RL训练 →
  ├─ PPO? → ✗✗✗ (48GB → 2x24GB不可行)
  ├─ GRPO? → ✓✓✓ (17GB → 单GPU可行)
  │    ├─ 多GPU? → ✗ (PCIe灾难 → 8GPU=0.46x)
  │    └─ 单GPU? → ✓ (Tinker in-process → 最快)
  │         ├─ 全参数? → ✗ (7B BF16=14GB+梯度→超内存)
  │         └─ LoRA? → ✓ (rank=32 → ~3GB → 17GB total)
  │              ├─ rLLM Tinker? → ★★★ 最优(auto-init+bypass)
  │              ├─ verl HYBRID? → ✓(功能等价,手动LoRA)
  │              └─ DeepSpeed ZeRO-2? → ✓(但不如Tinker)
  ├─ MoE? → ★ AutoEP EP=1+ZeRO-2+LoRA → Qwen3-MoE可行
  └─ 推理部署 → INT4+INT8KV+EAGLE → 9,088 tok/s
```

## 最优配置

```
★ ★ ★ RTX 4090最优RL训练配置:

框架: rLLM TinkerBackend (in-process, 零拷贝)
算法: GRPO (group-relative, no critic)
模型: 7B BF16 + LoRA-32 (auto-init, train_attn/mlp/unembed)
advantage: GRPO_VECTORIZED (rollout_n=8, 10-100x快)
★ ★ bypass_mode=true → pi_old=rollout logprobs → 省forward pass!
★ ★ fused fwd-bwd-optim → asyncio.gather → GPU pipeline overlap
Reward: rule-based → math correctness / code execution → 无GPU给RM
Weight sync: GPU-only merge → save_weights → new SamplingClient → 零拷贝
Memory: ~17GB → 24GB ✓ → 7GB headroom

部署: merge LoRA → HF → INT4 → vLLM → 4,791 tok/s
EAGLE: INT4+EAGLE → 9,088 tok/s → 8.3x!
评估: rllm eval --attempts N → pass@k → CPU warm-pool → 不占GPU
```

## 内存估算

```
| 配置 | 内存 | RTX4090可行? |
|------|------|-------------|
| PPO BF16全参数 | 270GB | ✗✗✗ |
| PPO LoRA | 48GB | ✗✗✗ (2x24GB) |
| GRPO BF16全参数 | 28GB | ✗ (超24GB) |
| GRPO LoRA-16 | 15.62GB | ✓✓✓ |
| GRPO LoRA-32 | 17.02GB | ✓✓✓ (最优) |
| DeepSpeed ZeRO-2+LoRA+CPU | 17.15GB | ✓ |
| FSDP2单GPU | 25.9GB | ✗ |
```

## 7框架RL训练速查

```
| 框架 | RL算法 | 单GPU可行? | 关键优势 | RTX4090评级 |
|------|--------|-----------|---------|-------------|
| rLLM Tinker | GRPO+PPO | ✓✓✓ | auto-init+bypass+zero-copy | ★★★ 最优 |
| verl HYBRID | GRPO+PPO | ✓✓✓ | 多rollout引擎(vLLM/SGLang) | ✓✓ 优秀 |
| DeepSpeed | GRPO(PPO) | ✓ | ZeRO-2+CPU_Adam | ✓ 可行 |
| Megatron | GRPO(recipe) | ✗ | TP+PP+MoE → PCIe不行 | ✗ |
| vLLM | 推理only | ✓ | INT4+9,088tok/s | ★★★ 推理最优 |
| MindIE | Ascend only | ✗ | NPU专用 | ✗ |
| PyTorch | compile+LoRA | ✓ | reduce-overhead+compile | ✓ 可选加速 |
```

## 关键数字

```
★ 模型内存: 7B BF16=14GB / INT4=3.5GB / LoRA-32=0.11GB(trainable)
★ KV cache: INT8KV+GQA-8 → ~5GB / BF16 → ~10GB(不可行)
★ 推理吞吐: INT4 → 4,791 tok/s / INT4+EAGLE → 9,088 tok/s
★ 训练吞吐: Tinker+GRPO+LoRA → ~19,743 tok/s
★ PPO vs GRPO: PPO=48GB ✗ / GRPO=17GB ✓ → ★唯一选择!
★ PCIe scaling: 8GPU=0.46x → ✗ / 单GPU=最优 ✓
★ bypass_mode: 省1个forward → ~0.3s per step → ★★★
★ rollout_n: 最少4(防止singleton n=1→advantage=0) / 推荐8
★ LoRA merge后推理同 → 无overhead → merge→INT4→最快路径
★ CUDA graph decode: ~10-15% throughput improvement
★ FlashInfer sampler: temperature>0+top-k/top-p → O(k) → RTX4090✓
```

## Checkpoint→推理路径

```
rLLM Tinker → save_pretrained → merge LoRA → HF format
  → INT4 quantize → vLLM → 4,791 tok/s → ★最简路径!

verl HYBRID → FSDPModelMerger → HF format
  → INT4 quantize → vLLM → 4,791 tok/s → ★等价但多1步!

DeepSpeed ZeRO-2 → universal checkpoint → FP32 → HF
  → INT4 quantize → vLLM → 4,791 tok/s → ★4步转换!
```

## GPU上线时优先实验

```
1. rLLM Tinker+GRPO+LoRA-32+bypass_mode → 训练7B → 实测吞吐
2. vLLM INT4+INT8KV+EAGLE → 推理 → 实测9,088 tok/s
3. torch.compile(reduce-overhead)+LoRA → 训练加速 → 实测+15-16%
4. GRPO rollout_n sweep(4/8/16) → advantage质量 → 实测
5. pass@k eval → 独立评估 → 不需要GPU → 本地可做
```

## 不可行配置 (避免!)

```
✗ PPO (48GB → 2x24GB → 不可能)
✗ FSDP2单GPU (25.9GB → 超24GB)
✗ TP>1 (PCIe → 0.46x scaling → 灾难)
✗ PP>1 (PCIE P2P → 11ms → 灾难)
✗ MoE EP>1 (DeepEP需SM90 → RTX4090✗)
✗ ZeRO-3 (PCIe AllGather 42GB/step → 灾难)
✗ Reward Model (14GB → 24GB只剩9GB → 不够)
✗ NVLS/TMA/FA3 (SM90 only → RTX4090✗)
```
