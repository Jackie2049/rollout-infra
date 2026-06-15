# DeepSpeed Training Pipeline深度分析 — RTX 4090视角

> 2026-06-16 | DeepSpeed ZeRO | LoRA | Gradient Accumulation | AutoEP | Singleton MoE
> ★★★★★ ZeRO-2 + CPU optimizer + LoRAOptimizedLinear + coalesce_grad_reduction = RTX 4090最优DeepSpeed配置
> ★★★★★ DeepSpeed唯一优势: MoE AutoEP → 其他框架没有proper MoE EP support!

## 1. ★★★★★ DeepSpeed Trainer API核心循环

```
★★★★★★★ DeepSpeedEngine核心循环:
  forward → loss → backward → (ZeRO gradient reduction) → step → (ZeRO param update)

★★★★★★★ ZeRO stage行为差异:

ZeRO-1 (optimizer partitioning only):
  → forward: all params local → 正常forward
  → backward: gradients reduce-scatter → 每GPU只持optimizer state shard
  → step: optimizer更新shard → all-gather更新后的params
  → ★★★★★ 最小memory节省 → 但通信最少

ZeRO-2 (optimizer + gradient partitioning):
  → forward: all params local → 正常forward
  → backward: gradients reduce-scatter → 每GPU只持gradient shard
  → step: optimizer更新 → all-gather updated params
  → ★★★★★★ RTX 4090最优: 省optimizer memory → 通信适中 → gradient partitioning省2x

ZeRO-3 (optimizer + gradient + parameter partitioning):
  → forward: ★★★★★ all-gather params → forward → discard gathered params → 省内存!
  → backward: all-gather params → backward → reduce-scatter gradients → discard params
  → step: optimizer update shard → all-gather updated params
  → ★★★★★ Single GPU = 无意义 → 没有partitioning → 只增加overhead!

★★★★★★★ RTX 4090推荐: ZeRO-2
  → ZeRO-1 → 太少memory节省 → 不值得overhead
  → ZeRO-2 → 省optimizer state → gradient partitioning → ★★★★★ 最优!
  → ZeRO-3 → 单GPU无partitioning benefit → ★★★✗✗ RTX 4090不推荐!
  → ZeRO-0 → no partitioning → standard DDP → 简单但memory最差
```

## 2. ★★★★★ DeepSpeed LoRA Integration

```
★★★★★★★ DeepSpeed有自己的LoRA → LoRAOptimizedLinear!

  → 不是PEFT/HuggingFace LoRA → DeepSpeed原生实现
  → base_weight_sharding → ZeRO sharding of frozen weights → 省内存
  → offload_ratio → partial CPU offload of base weights → GPU只持LoRA+active shard
  → QuantizedParameter (FP8) → 量化+LoRA共存 → inference+training

★★★★★★★ LoRAOptimizedLinear核心机制:
  → frozen base_weight → partitioned by ZeRO (if stage>=2) → 每GPU只持shard
  → LoRA adapter → full on each GPU → 可训练 → 不partitioned
  → forward: all-gather base_weight shard → compute LoRA → merge output → discard shard
  → ★★★★★★ offload_ratio → 部分base weights on CPU → GPU只持active部分 → RTX 4090内存友好!

★★★★★★★ vs PEFT LoRA:
  → PEFT → HuggingFace → general purpose → 但不integrated with ZeRO
  → DeepSpeed LoRA → ZeRO-integrated → partitioning + offloading → memory更优
  → → ★★★★★ RTX 4090: DeepSpeed LoRAOptimizedLinear + offload_ratio → 最省内存!

★★★★★★★ DeepSpeed-Chat convert_linear_layer_to_lora:
  → 另一种LoRA → simpler → 但不支持ZeRO sharding → base weights全GPU
  → → RTX 4090 → LoRAOptimizedLinear优先 → ZeRO-integrated → memory更优
```

## 3. ★★★★★ Gradient Accumulation for RL Training

```
★★★★★★★ 3种gradient accumulation机制:

Mechanism 1: Standard gas (gradient_accumulation_steps)
  → 累积micro-batches → 每gas步reduce → 简单
  → ★★★★★ 但ZeRO-2/3中 → 每个micro-batch都reduce-scatter → 通信overhead!

Mechanism 2: no_sync()
  → 禁用gradient sync → 累积多个micro-batch → 只在boundary reduce
  → ★★★✗✗✗ 不兼容ZeRO-2/3! → ZeRO-2/3必须reduce every backward!
  → → ZeRO-1 only → 不适用于RTX 4090最优配置

Mechanism 3: ★★★★★★ coalesce_grad_reduction()
  → 累积多个micro-batch gradients → 只在boundary reduce → 但兼容ZeRO!
  → ★★★★★ Works with ZeRO-1/2/3 → RTX 4090最优!
  → set_gradient_accumulation_boundary() → manual override → RL training必须!

★★★★★★★ RTX 4090推荐: coalesce_grad_reduction + ZeRO-2
  → 累积多个micro-batch → 只在boundary reduce → 通信减少 → 内存不变
  → ★★★★★ 这是GRPO训练的正确方式 → standard gas不适合ZeRO-2!
```

## 4. ★★★★★ ZeRO-2 (actor) + ZeRO-3 (critic) Mix

```
★★★★★★★ DeepSpeed-Chat explicitly supports different ZeRO stages per model!

  → Pattern: "1.3b: zero-2 for actor/ref, zero-0 for others"
  → Pattern: "13b+: zero-3 for all"
  → → 每个模型可以不同ZeRO stage → 混合配置

★★★★★★★ DeepSpeedRLHFEngine orchestrates 4 models:
  → actor (policy) → ZeRO-2 → optimize for training throughput
  → critic (value) → ZeRO-3 → optimize for memory (critic大)
  → reward → ZeRO-0 → 最简单 → 不需要partitioning
  → ref → ZeRO-0 → inference only → 不需要optimizer

★★★★★★★ RTX 4090分析:
  → GRPO → no critic → 只需要actor + ref (或bypass_mode → skip ref)
  → → actor: ZeRO-2 → 省optimizer memory → ★★★★★ 最优
  → → ref: bypass_mode → skip → 省14GB → ★★★★★ RTX 4090 MUST
  → → ✗✗✗ PPO需要critic → ZeRO-3 critic → memory太大 → 24GB OOM!

★★★★★★★ Hybrid Engine → inference/training mode切换:
  → LoRA fuse/unfuse → inference时merge LoRA → training时split
  → ZeRO-3 gather → inference时all-gather params → training时partition
  → Overflow alignment → align_overflow → 同步actor/critic overflow
```

## 5. ★★★★★ Optimizer Offloading

```
★★★★★★★ ZeRO-2 + CPU_Adam → optimizer on CPU!

  → Forward: GPU → 正常 → params on GPU
  → Backward: GPU → 正常 → gradients on GPU → reduce-scatter → gradient shard on GPU
  → Optimizer step: CPU → 80% CPU cores → pinned buffers → optimizer state on CPU
  → ★★★★★★ 省optimizer GPU memory → ~2GB for 7B model → RTX 4090关键!

★★★★★★★ LoRAOptimizedLinear offload_ratio:
  → offload_ratio=0.5 → 50% frozen base weights on CPU → 50% on GPU
  → forward时 → swap base weights to GPU → compute → swap back
  → ★★★★★★ 进一步省GPU memory → RTX 4090可能hold 7B model with LoRA!

★★★★★★★ ZenFlow → async optimizer overlapping:
  → CPU optimizer step → overlap with next GPU forward → latency隐藏
  → ★★★★★ 不省内存 → 但省时间 → training throughput优化
```

## 6. ★★★★★ AutoEP (#7938) in Training Loop

```
★★★★★★★ AutoEP → replace HF MoE blocks with AutoEPMoELayer during engine init:

  → Expert params → EDP (Expert Data Parallel) gradient reduction → only over EP group
  → Router params → global DP gradient reduction → over all GPUs
  → → ★★★★★★ 不同的梯度reduce路径 → 不同partitioning → 更精确!

★★★★★★★ ZeRO-3 conflict:
  → Currently only ZeRO-0/1/2 supported → ZeRO-3 asserts "MoE not supported with Stage 3"
  → ★★★★★ AutoEP+ZeRO-3 (#8060) → 解决冲突 → per-parameter partition groups → 但still OPEN!

★★★★★★★ Single GPU (ep_size=1) → AutoEP behavior:
  → All experts local → no AllToAllV → singleton MoE (#7997) → skip identity collectives
  → ★★★★★★ RTX 4090 single GPU → AutoEP degenerates to standard training → 但MoE model太大!
  → → ★★★★★★ DeepSpeed正确处理singleton → vs Megatron crash (#5203) → 框架质量差异!
```

## 7. ★★★★★ Singleton MoE (#7997) Training Impact

```
★★★★★★★ Singleton MoE = ep_size=1 → 15x speedup!

  → Skip both AllToAll calls → no token dispatch → no token collection
  → Skip capacity all_reduce → no capacity statistics → no load balance monitoring
  → → ★★★★★ 13s → 0.864s → 15x speedup → dramatic!

★★★★★★★ RTX 4090 (world_size=1):
  → ALL MoE collectives degenerate to identity operations → no communication needed
  → → ★★★★★ DeepSpeed正确处理 → Megatron crash (#5203) → 关键差异!
  → → → ★★★★★★ DeepSpeed > Megatron for single GPU MoE → confirmed!

★★★★★★★ 但MoE模型太大 → RTX 4090 24GB → 30B MoE → ~60GB → ✗✗✗ not feasible!
  → → 需要INT4/INT8 quantization → 或小MoE → 未来可能
```

## 8. ★★★★★ RTX 4090最优DeepSpeed GRPO配置

```
★★★★★★★ RTX 4090最优DeepSpeed配置:

  ZeRO stage: ZeRO-2 → 省optimizer memory → gradient partitioning → ★★★★★ 最优
  Optimizer: CPU_Adam → optimizer on CPU → 省~2GB GPU memory
  LoRA: LoRAOptimizedLinear → offload_ratio=0.5 → base weights 50% CPU → 省更多GPU
  Gradient accumulation: coalesce_grad_reduction → 累积 → boundary reduce → ZeRO-2 compatible
  Bypass mode: ★★★★★ skip ref model → 省14GB → MUST for RTX 4090
  MoE: AutoEP + Singleton → 如果用MoE → 正确处理singleton → vs Megatron crash

★★★★★★★ 但RTX 4090 GRPO排名:
  → rLLM Tinker #1 → simplest → in-process → bypass default → ★★★★★ 最简单!
  → verl #2 → larger community → CPPO available → but Ray overhead + enforce_eager
  → DeepSpeed #3 → ★★★★★ MoE唯一优势 → 但dense model不如rLLM/verl简单
  → Megatron #4 → crash + no LoRA → ✗✗✗ not viable

★★★★★★★ DeepSpeed unique advantage → MoE training with AutoEP:
  → 其他框架没有proper MoE EP support → DeepSpeed是唯一
  → → 但MoE模型太大 → RTX 4090短期不可行 → 中期 + INT4 quant可能
  → → ★★★★★ DeepSpeed = MoE训练的长期选择 → 但RTX 4090 dense model → rLLM/verl更好

★★★★★★★ DeepSpeed vs verl for RTX 4090:
  → Dense model → verl/rLLM → simpler → bypass_mode → community
  → MoE model → DeepSpeed → AutoEP + Singleton → 正确处理 → 但模型太大
  → ★★★★★ 短期 → dense → rLLM/verl → 中期 → MoE → DeepSpeed (if small MoE available)
```

## 参考
- DeepSpeed repo: https://github.com/microsoft/DeepSpeed
- DeepSpeed-Chat: https://github.com/microsoft/DeepSpeedExamples
- LoRAOptimizedLinear: deepspeed/compression/linear/lora_linear.py
- coalesce_grad_reduction: deepspeed/engine/engine.py
- AutoEP (#7938): deepspeed/moe/auto_ep/
- Singleton MoE (#7997): deepspeed/moe/singleton_moe/
- ZeRO-2+CPU_Adam: deepspeed/ops/cpu_adam/
- 相关笔记: deepspeed-latest-developments-2026-06-reading.md, deepspeed-autoep-zero3-reading.md, deepspeed-inference-reading.md
