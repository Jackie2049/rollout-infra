# 7框架训练数据流全景对比

> 2026-06-15 | 从初始化到optimizer step的完整数据流对比
> 核心问题: 每个框架在训练一个step时,数据是如何流动的?

## 1. 初始化阶段对比

| 框架 | 模型初始化 | 优化器初始化 | 分布式设置 | 特殊机制 |
|------|-----------|-------------|-----------|---------|
| **DeepSpeed ZeRO-2** | `deepspeed.initialize()` | config指定→CPU Adam或GPU Adam | 自动创建DP组 | `ipg_buckets`梯度桶初始化 |
| **DeepSpeed ZeRO-3** | `with zero.Init():` metaclass注入→参数自动分区 | FP32分区动态填充 | 自动创建DP组 | `PartitionedParameterCoordinator` trace初始化 |
| **Megatron-LM** | TP/PP分区→Column/RowParallelLinear | DistributedOptimizer | 5D并行组(tp×cp×ep×dp×pp) | `initialize_model_parallel()`→15+复合组 |
| **vLLM** | PagedAttention初始化 | 不适用(推理) | TP组(推理) | KV cache block池预分配 |
| **verl GRPO** | Actor model(LoRA) | Actor optimizer | FSDP2或单GPU | `need_critic()=False→`跳过critic初始化 |
| **verl PPO** | Actor+Critic两个模型 | 2个Adam优化器 | FSDP2+Ray PG | Critic worker独立进程 |
| **rLLM Tinker** | LoRA auto-init(r=16) | TinkerBackend创建LoRA client | 无(单GPU in-process) | `SamplingClient`自动创建→rollout+train共用 |
| **PyTorch FSDP2** | `torch.distributed.FSDP()` wrap→DTensor分区 | ShardedOptimizer | DeviceMesh→DTensor | `DTensor.from_local()`→每个参数自动分区 |

### 关键差异

```
ZeRO-3: metaclass注入 → Module.__init__结束时自动分区 → 无需手动wrap!
  with deepspeed.zero.Init():
    model = LlamaForCausalLM(config) → 每个参数自动获得ds_tensor属性

FSDP2: 手动wrap → 需要选择wrap策略 → 决定哪些参数一起分区
  model = FSDP(model, auto_wrap_policy=...) → FlatParameter或DTensor

Megatron: 构建时替换 → ColumnParallelLinear/RowParallelLinear
  → 模型代码需要修改! → 不是透明wrap

rLLM Tinker: 无wrap → LoRA自动初始化 → 最简单!
  → 但只支持单GPU → 无分布式能力
```

## 2. Forward阶段对比

| 框架 | 参数获取 | 计算方式 | 激活管理 | 通信 |
|------|---------|---------|---------|------|
| **ZeRO-2** | 本GPU完整参数 | 标准forward | activation checkpointing可选 | 无 |
| **ZeRO-3** | AllGather每层参数→NOT_AVAILABLE | 标准forward(参数在forward后释放!) | release_sub_module→参数释放 | AllGather(Ψ per layer) |
| **Megatron TP** | TP组内AllReduce→完整层 | 标准forward | Sequence Parallel: ReduceScatter→省内存 | AllReduce(2×hidden×seq×batch per layer) |
| **Megatron PP** | 本stage完整参数 | 标准forward | 1F1B调度→deallocate output | P2P send/recv between stages |
| **verl** | FSDP2→AllGather每层 or 单GPU完整 | LoRA forward | 标准 | FSDP2: AllGather; 单GPU: 无 |
| **rLLM Tinker** | 本GPU完整参数(base+LoRA) | LoRA forward | 标准 | 无 |
| **PyTorch FSDP2** | AllGather DTensor→完整参数 | 标准(或compile→fused) | 保留full param用于backward | AllGather(Ψ per layer) |

### ZeRO-3 vs FSDP2 forward关键差异

```
ZeRO-3 forward:
  for each layer in forward_order:
    PartitionedParameterCoordinator.fetch_sub_module(layer)
    → AllGather(layer params) → params状态=AVAILABLE
    compute forward with gathered params
    → release_sub_module(layer) → params状态=NOT_AVAILABLE!
  → forward结束时所有参数都回到分区状态!

FSDP2 forward:
  for each module in FSDP wrap order:
    AllGather(module DTensor params) → DTensor→full tensor
    compute forward
    → keep full params for backward! (不释放)
  → forward结束时参数仍然完整! (直到backward完成)

→ ZeRO-3: forward后内存回到Ψ/N per GPU (低peak!)
→ FSDP2: forward后内存=Ψ per GPU (peak higher but saves backward AllGather!)
```

## 3. Backward阶段对比

| 框架 | 梯度计算 | 梯度聚合 | 参数更新参与 | 通信 |
|------|---------|---------|-------------|------|
| **ZeRO-2** | 标准backward | ipg_buckets→ReduceScatter→本分区梯度 | 只有本分区参数更新 | ReduceScatter(≈AllReduce带宽) |
| **ZeRO-3** | 标准backward(需重新gather参数!) | ReduceScatter→本分区梯度 | 只有本分区参数更新 | AllGather(params)+ReduceScatter(grads)+AllGather(params) = 3Ψ |
| **Megatron TP** | 标准backward | AllReduce梯度→完整grad | 本TP rank更新完整层参数 | AllReduce(2× per layer) |
| **Megatron PP** | 标准backward(1F1B调度) | DP AllReduce(延迟到cooldown) | 本stage参数更新 | P2P backward + DP AllReduce |
| **verl GRPO** | LoRA backward only(base冻结) | FSDP2 ReduceScatter or 单GPU | LoRA params only | FSDP2: ReduceScatter; 单GPU: 无 |
| **verl PPO** | Actor backward + Critic backward | FSDP2 ReduceScatter | LoRA params(actor)+Critic params | FSDP2: ReduceScatter×2(actor+critic) |
| **rLLM Tinker** | LoRA backward only | 无(单GPU) | LoRA params only | 无 |
| **PyTorch FSDP2** | 标准backward(使用保留的full params) | ReduceScatter→DTensor shard | 本shard参数更新 | ReduceScatter(grads) = 1Ψ |

### GRPO vs PPO backward关键差异

```
PPO backward:
  1. Actor forward (generate sequences) → old_log_probs
  2. Critic forward → values tensor [bsz, resp_len]
  3. GAE advantage computation (token-level, recursive)
  4. Actor backward → actor gradients
  5. Critic backward → critic gradients
  → 2× backward! → 2× memory for activation!

GRPO backward:
  1. Actor forward (generate sequences × rollout_n) → old_log_probs
  2. NO Critic forward → SKIPPED!
  3. GRPO advantage computation (response-level, group normalization)
  4. Actor backward → actor gradients only
  5. NO Critic backward → SKIPPED!
  → 1× backward! → 省50% compute+memory!

→ RTX 4090: PPO需要2×模型内存=28GB→✗不fit 24GB!
→ GRPO只需1×模型=14GB+LoRA+activation=17GB→✓fit!
```

## 4. Optimizer Step阶段对比

| 框架 | 优化器类型 | 操作对象 | Step后操作 | 通信 |
|------|-----------|---------|-----------|------|
| **ZeRO-2** | CPU Adam or GPU Adam | FP32分区(1/N参数) | AllGather更新后的fp16参数→重建完整model | AllGather(Ψ) |
| **ZeRO-3** | GPU Adam(分区) | FP32分区(1/N参数) | 不AllGather!下次forward时按需gather | 无(延迟到forward) |
| **Megatron** | DistributedOptimizer | 每TP rank完整层参数 | DP AllReduce梯度→更新完整层 | DP AllReduce(grads) |
| **verl** | Adam | LoRA FP32 params or FSDP2 shard | FSDP2: ReduceScatter(grads); 单GPU: 直接更新 | 视策略而定 |
| **rLLM Tinker** | LoRA Adam(client端) | LoRA FP32 params only | 保存checkpoint→新SamplingClient→rollout使用新权重 | 文件传输(checkpoint) |
| **PyTorch FSDP2** | ShardedOptimizer | FP32 DTensor shard | ReduceScatter(grads)→reshard params→下次forward AllGather | ReduceScatter(grads) |

### ZeRO-2 vs ZeRO-3 optimizer step关键差异

```
ZeRO-2 optimizer step (stage_1_and_2.py:2140-2269):
  1. Free gradients NOT in this partition
  2. Flatten averaged_gradients IN this partition
  3. Assign as fp32_partition.grad
  4. optimizer.step() on single fp32 partition
  5. Copy fp32→fp16 partition
  6. AllGather updated fp16 weights → 重建完整model!
  → Step后: 所有GPU看到一致更新后的完整model

ZeRO-3 optimizer step (stage3.py:2424-2468):
  1. _partition_all_parameters() → 所有参数回分区状态
  2. For each sub_group:
     prepare_fp32_grad → optimizer.step() → copy fp32→fp16_partition
  3. _post_step() → gather persistent params, invalidate secondary tensors
  → Step后: 参数仍然分区! 不AllGather!
  → 下次forward时 PartitionedParameterCoordinator 按需gather

→ ZeRO-2: step后model params一致(每GPU有完整fp16 model) → 直接做下一个forward
→ ZeRO-3: step后model params分区 → 下次forward需要AllGather → 但内存更低peak!
```

## 5. 通信量对比(7B模型, 8GPU, 每1个training step)

| 框架/策略 | Forward通信 | Backward通信 | Optimizer通信 | **总计** | NVLink时间 | PCIe时间 |
|----------|------------|-------------|-------------|---------|-----------|---------|
| DDP | 0 | AllReduce(grads)=Ψ | 0 | **1Ψ=14GB** | 0.14s | 4.56s |
| ZeRO-2 | 0 | ReduceScatter(grads)≈Ψ | AllGather(updated)=Ψ | **2Ψ=28GB** | 0.28s | 9.12s |
| ZeRO-2+CPU+LoRA | 0 | ReduceScatter(LoRA grads)≈0.11GB | AllGather(LoRA)=0.11GB | **0.22GB** | 0.002s | 0.01s |
| ZeRO-3 | AllGather(Ψ) | RS(Ψ)+AG(Ψ) | 0(延迟到fwd) | **3Ψ=42GB** | 0.21s | 1.75s |
| FSDP2 | AllGather(Ψ) | ReduceScatter(Ψ) | 0(延迟到fwd) | **2Ψ=28GB** | 0.14s | 1.17s |
| FSDP2+compile | AllGather(Ψ) | ReduceScatter(Ψ) | 0 | **2Ψ=28GB** | 0.14s+2.5x compute savings | 1.17s |
| Megatron TP=2 | AllReduce×2/layer | AllReduce×2/layer | DP AllReduce | **每层2×hidden×seq×bs** | ~0.01s/layer | ~0.3s/layer |
| Megatron PP=2 | P2P | P2P backward | DP AllReduce | **2×(PP-1)×activation** | ~0.001s/P2P | ~0.05s/P2P |
| rLLM Tinker | 0 | 0 | 0 | **0** | 0 | 0 |
| verl GRPO+Tinker | 0 | 0 | 0 | **0** | 0 | 0 |
| verl GRPO+FSDP2 | AllGather(Ψ) | ReduceScatter(LoRA grads) | 0 | **~Ψ+0.01Ψ** | ~0.14s | ~1.17s |

### RTX 4090关键约束

```
PCIe带宽: ~25GB/s (双向) → 8GPU AllReduce ≈ 4.56ms/GB
NVLink带宽: ~300GB/s → 8GPU AllReduce ≈ 0.55ms/GB

RTX 4090 PCIe上的通信灾难:
  ZeRO-3: 42GB/step × 4.56ms/GB = 1.75s → 计算时间~0.5s → 通信占比>75%!
  FSDP2: 28GB/step × 4.56ms/GB = 1.17s → 同样>60%通信占比!
  DDP+LoRA: 0.22GB/step × 4.56ms/GB = 1ms → <5%通信占比 → 可行!

→ RTX 4090 PCIe唯一可行: DDP+LoRA+CPU_Adam or rLLM Tinker
→ NVLink集群: FSDP2+compile最优(2-3x compute speedup)
```

## 6. 内存峰值对比(7B模型, RTX 4090 24GB)

| 框架/策略 | 模型参数 | 梯度 | 优化器 | 临时 | 激活 | **Peak** | **Fits 24GB** |
|----------|---------|------|-------|------|------|---------|-------------|
| DDP | 14GB | 14GB | 56GB | 0 | 1.4GB | **141GB** | ✗ |
| ZeRO-2(8GPU) | 14+4GB | 14/8=1.75 | 56/8=7 | 0 | 1.4GB | **55GB** | ✗ |
| ZeRO-2+CPU+LoRA | 14 | 0.26 | 0(CPU) | 0 | 1.4GB | **17.15GB** | ✓ |
| ZeRO-3(8GPU) | 14/8=1.75 | 14/8=1.75 | 56/8=7 | 14(AG) | 1.4GB | **32.9GB** | ✗ |
| FSDP2(8GPU) | 14/8=1.75 | 14/8=1.75 | 56/8=7 | 14(AG) | 1.4GB | **32.9GB** | ✗ |
| Megatron TP=1 | 14 | 14 | 56 | 0 | 1.4GB | **141GB** | ✗ |
| rLLM Tinker+GRPO | 14 | 0.26(LoRA) | 0(CPU) | 0 | 2.8(×2) | **17.02GB** | ✓ |
| verl GRPO+LoRA+CPU | 14 | 0.26(LoRA) | 0(CPU) | 0 | 1.4 | **15.62GB** | ✓ |
| PPO | 28(actor+critic) | 28 | 112 | 0 | 2.8 | **270GB** | ✗✗✗ |

**RTX 4090可行方案只有3个**: LoRA+CPU_Adam / GRPO+LoRA+CPU_Adam / rLLM Tinker+GRPO

## 7. 完整训练Loop对比(verl GRPO vs PPO)

```
PPO训练loop (verl ray_trainer.py):
  1. ActorWorker.rollout → generate sequences (vLLM engine)
  2. ActorWorker.recompute_log_prob → old_log_probs
  3. RefPolicyWorker.recompute_log_prob → ref_log_probs (optional)
  4. Reward computation → token_level_scores
  5. ★ CriticWorker.infer → values [bsz, resp_len]
  6. ★ GAE advantage: recursive TD-lambda (token-level)
  7. ActorWorker.update → ppo_loss + dual-clip
  8. ★ CriticWorker.update → value_loss
  → 4个GPU进程: ActorWorker, CriticWorker, RefPolicyWorker, RewardWorker
  → 内存: actor+critic = 2×模型 + 2×optimizer = ~4×单模型

GRPO训练loop (verl ray_trainer.py):
  1. ActorWorker.rollout → generate sequences × rollout_n
  2. ActorWorker.recompute_log_prob → old_log_probs
  3. RefPolicyWorker.recompute_log_prob → ref_log_probs (optional)
  4. Reward computation → token_level_scores
  5. ✗ NO Critic forward → SKIPPED
  6. ★ GRPO advantage: group mean/std (response-level)
  7. ActorWorker.update → ppo_loss (same code, different advantages)
  8. ✗ NO Critic update → SKIPPED
  → 3个GPU进程: ActorWorker, RefPolicyWorker, RewardWorker (no Critic!)
  → 内存: actor only = 1×模型 + 1×optimizer = ~2×单模型(LoRA→更少)
```

## 8. 完整训练Loop对比(rLLM Tinker vs verl)

```
rLLM Tinker GRPO loop (UnifiedTrainer):
  1. generate (SamplingClient → 本GPU inference, in-process)
  2. transform (prefix-merge → mask=[1,0,1...] → action-only)
  3. RS (RejectionSampling → filter low-quality)
  4. to_batch (TrajectoryGroupBuffer → batch formation)
  5. process (TinkerBackend → compute advantage in-process)
  6. advantage (GRPO: group mean/std → broadcast via mask)
  7. update (fused fwd-bwd-optim → LoRA training)
  8. cleanup (save checkpoint → new SamplingClient → weight sync)
  → 全程in-process! 无Ray/HTTP/ZMQ overhead!
  → SyncCoordinator: staleness=2, sync_step=4 → async training
  → 内存peak: 17.02GB → ✓ fit RTX 4090 24GB

verl GRPO FSDP2 loop (ray_trainer.py):
  1. ActorWorker.rollout → vLLM engine (Ray subprocess)
  2. ActorWorker.recompute_log_prob → old_log_probs
  3. compute_reward → token_level_scores (separate worker)
  4. compute_grpo_advantage → group mean/std
  5. ActorWorker.update_policy → ppo_loss + LoRA backward
  → Ray分布式 → 有通信overhead
  → 需多GPU → RTX 4090单GPU不适用(Ray overhead)
  → 多GPU场景下: FSDP2+compile → 2-3x加速
```

## 9. 总结: 选择决策树

```
条件 → 推荐框架+策略:

1. 单GPU RTX 4090 24GB + GRPO训练
   → rLLM Tinker (in-process, 17GB, 无分布式overhead)
   → verl GRPO+LoRA+CPU_Adam (备选, 15.6GB)

2. 8×NVLink A100/H100 80GB + 全参数训练
   → PyTorch FSDP2+compile (2Ψ comm, 2-3x加速)
   → DeepSpeed ZeRO-3 (3Ψ, 但有prefetch overlap)

3. 8×NVLink + GRPO训练
   → verl GRPO+FSDP2+compile (最佳: GRPO省50%内存+FSDP2加速)
   → rLLM VerlBackend+GRPO (上层编排)

4. 4-8×RTX 4090 PCIe + LoRA训练
   → DeepSpeed ZeRO-2+CPU_Adam+LoRA (DP+LoRA+CPU offload)
   → PyTorch DDP+LoRA+CPU_Adam (标准PyTorch)
   → 不推荐任何sharding策略(PCIe瓶颈)

5. 大MoE模型+NVLink集群
   → Megatron TP+EP (expert all-to-all)
   → +DeepEP (asymmetric dispatch, 更高效)

6. 推理
   → vLLM INT4+INT8KV+GQA+FlashInfer (最大吞吐)
   → SGLang Overlap调度 (decode/prefill overlap)

7. 华为昇腾NPU
   → MindIE+ATB+HCCL (原生优化)
   → vLLM-Ascend (开源替代)
```

---

Sources:
- notebook/projects/deepspeed-zero3-data-flow.md
- notebook/projects/deepspeed-distributed-optimizer-source-reading.md
- notebook/projects/deepspeed-comm-overlap-reading.md
- notebook/projects/zero3-vs-fsdp2-system-comparison.md
- notebook/projects/verl-ppo-vs-grpo-training-loop-comparison.md
- notebook/projects/rllm-gateway-backend-trainer-source-reading.md
- notebook/projects/megatron-lm-reading.md
- notebook/projects/seven-framework-comparison.md
- tools/comm_cost_calculator.py + zero_memory_peak_simulator.py + training_speed_estimator.py
