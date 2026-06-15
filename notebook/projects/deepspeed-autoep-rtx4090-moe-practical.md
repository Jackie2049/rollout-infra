# DeepSpeed AutoEP + Qwen3-MoE — RTX 4090 单GPU实战配置

> 2026-06-15 | 基于deepspeed-latest-developments-2026-06.md + AutoEP PR#7938 → RTX 4090 EP=1单GPUMoE训练配置
> ★ ★ ★ AutoEP PR#7938 merged 6/11 → 5 presets → 零代码修改 → EP=1+ZeRO-2+LoRA → Qwen3-MoE可行

## 1. AutoEP是什么 — 5秒理解

```
★ ★ ★ AutoEP = Automatic Expert Parallelism → 自动配置MoE的Expert Parallel

传统:
  → 用户手动配置EP degree → 需要知道GPU topology → 复杂!
  → → EP>1 → 需要多GPU → AllToAll通信 → RTX 4090 PCIe灾难!

AutoEP:
  → 一行配置 → auto_ep_preset: "Qwen3-MoE" → 自动决定EP degree
  → → world_size=1 → EP=1 → 所有experts同GPU → ★ 单GPU可行!
  → → ★ ★★ 零代码修改 → 只改JSON config → 最简!
```

## 2. 5种AutoEP Presets

```
★ ★ ★ 5种AutoEP预设(PR#7938):

| Preset | 模型 | Experts | Active | EP=1可行? |
|--------|------|---------|--------|----------|
| ★ Mixtral | Mixtral-8x7B | 8 | 2 | ✓(7B active→17GB with LoRA) |
| ★★ Qwen3-MoE | Qwen3-MoE | ~64 | ~8 | ✓(if active params fit) |
| ★ Qwen3.5-MoE | Qwen3.5-MoE | 更多 | ~8 | ✓(similar to Qwen3) |
| ★ DS-V2 | DeepSeek-V2-Lite | 6+shared | 6+1 | ✓(small MoE) |
| ★ DS-V3 | DeepSeek-V3 | 256+shared | 8+1 | ✗(671B→需要多GPU) |
| ★ LLaMA-4 | Llama-4-Scout | 16 | 4 | ✓(small MoE) |

★ ★ ★ EP=1 on RTX 4090:
  → world_size=1 → AutoEP自动检测 → ep_degree=1 → 所有experts同GPU
  → ★ 无AllToAll通信 → 无PCIe overhead → 单GPU最优!
  → → ★★★ 但: 24GB内存 → 只能fit active params → LoRA必须!
```

## 3. RTX 4090实战配置 — ds_config.json

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "all_contiguous_gradients": true,
    "reduce_bucket_size": 5e8
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "moe": {
    "ep_degree": 1,
    "moe_param_expert_parallelism": false
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 100
    }
  },
  "flops_profiler": {
    "enabled": true,
    "profile_step": 5
  }
}
```

## 4. Qwen3-MoE 内存估算

```
★ ★ ★ Qwen3-MoE 内存估算 (RTX 4090):

假设 Qwen3-MoE ≈ 类似Qwen2.5-MoE:
  → total params: ~15B → active params: ~3B
  → active 3B BF16 → ~6GB weights
  → LoRA rank=8 → ~0.3GB trainable → ~0.15GB optimizer(CPU offload)
  → expert weights(非active): ~12GB → 全部需要加载!

★ ★ ★ 内存预算:
  → active weights BF16: ~6GB
  → inactive expert weights: ~12GB → ★★★ 但activation时只加载2-3个expert!
  → LoRA trainable: ~0.3GB
  → optimizer(offload CPU): ~0.15GB on GPU → ★ CPU_Adam!
  → gradients: ~0.3GB (LoRA only)
  → activations: ~2GB
  → total: ~8.3GB (active only) → ★★★ 但需要所有expert weights!

★ ★ ★ 问题:
  → 所有expert weights ~12GB → 必须全部加载(EP=1 → 同GPU)
  → → 6GB active + 12GB expert = 18GB → ★★★ 太大! → 6GB headroom → tight!

★ ★ ★ ★ 解决方案:
  1. LoRA只训active layers → expert weights frozen → ★ 省gradient!
  2. expert weights offload to CPU → load on-demand → ★ 省GPU内存!
  3. INT4 quantize expert weights → 3x compression → ★ 更省!
  4. ★★★ 最佳: LoRA(active)+INT4(experts)+CPU_Adam → ~9GB → 15GB headroom!
```

## 5. 训练启动命令

```bash
# ★★★ RTX 4090 AutoEP + ZeRO-2 + LoRA 训练命令
deepspeed --num_gpus=1 train.py \
  --model_name_or_path Qwen/Qwen3-MoE \
  --peft_method lora \
  --lora_rank 8 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --deepspeed ds_config_zero2_lora.json \
  --output_dir ./output

# ★ AutoEP preset方式 (PR#7938后可用)
# → 零代码修改 → auto_ep_preset 自动配置
# → EP=1 → world_size=1 → 单GPU

# ★ Muon optimizer (可选 → 45%省optimizer state)
deepspeed --num_gpus=1 train.py \
  --model_name_or_path Qwen/Qwen3-MoE \
  --peft_method lora \
  --lora_rank 16 \
  --deepspeed ds_config_zero2_muon.json \
  --output_dir ./output
```

## 6. AutoEP vs 手动EP vs DeepEP

```
★ ★ ★ 3种EP方式对比:

| 方式 | 代码修改 | RTX 4090 | 灵活性 |
|------|---------|----------|--------|
| ★★★ AutoEP preset | 零 → 只改JSON | EP=1 ✓ | 5 preset → 自动 |
| ★ 手动EP config | 少 → ep_degree=1 | EP=1 ✓ | 自定义 → 需了解模型 |
| ✗✗✗ DeepEP | 大 → 改代码 | ✗(需SM90) | asymmetric → 高性能 |

★ ★ ★ RTX 4090最优: AutoEP preset → 零代码 → EP=1 → ZeRO-2+LoRA → 最简!

★ ★ AutoEP内部机制:
  → TorchTitan内核 → grouped-GEMM + Triton fill-indices → ★ kernel级优化
  → 但EP=1 → grouped-GEMM退化为普通GEMM → 无parallel优势
  → → ★★★ RTX 4090上AutoEP的价值: 自动配置 → 简化 → 不是性能优化
```

## 7. 训练→部署路径

```
★ ★ ★ Qwen3-MoE AutoEP训练→部署:

Step 1: DeepSpeed训练 → LoRA weights → ZeRO-2 → CPU_Adam
Step 2: 合并LoRA → HF format → save_pretrained
Step 3: INT4量化 → GPTQ → expert weights INT4 → ★ 3x compression
Step 4: vLLM INT4 MoE serving → Triton fallback for non-Marlin shapes
  → ★★★ Qwen3-MoE intermediate_size → 检查是否128-aligned
  → → 如果non-aligned → TritonW4A16LinearKernel → ★ PR#43731 → RTX 4090可用!
Step 5: EAGLE speculative → 可能不支持MoE → ★ 需要确认

★ ★ ★ 关键注意:
  → MoE推理 → expert routing → 每token不同expert → ★ CUDA graph不能batch所有expert
  → → ★ Megatron: InferenceTopKRouter + dense_output=True → FlashInfer兼容
  → → ★ vLLM MoE: 8+ AllToAll backends → EP=1 → 无通信 → 同GPU所有expert
  → → ★★★ INT4 MoE inference on RTX 4090 → Triton fallback → 部分layer slower → 但可行!
```

## 8. 与rLLM Tinker对比

```
★ ★ ★ DeepSpeed AutoEP vs rLLM Tinker — RTX 4090 MoE:

| 维度 | DeepSpeed AutoEP | rLLM Tinker |
|------|-----------------|------------|
| MoE支持 | ★★ AutoEP preset | ★★★ 无MoE特化但LoRA可行 |
| EP配置 | 自动(EP=1) | 不需要(单GPU) |
| LoRA | 手动配置 | ★★★ auto-init |
| GRPO | ✗ 需自实现 | ★★★ built-in |
| 优势 | ★★ MoE专用preset | ★★★ 极简+GRPO+bypass |
| 复杂度 | ★★ 中等 | ★★★★ 极简 |

★ ★ ★ 结论:
  → Dense model → rLLM Tinker最优 → GRPO+LoRA+bypass → 最简
  → MoE model → DeepSpeed AutoEP → preset+ZeRO-2+LoRA → MoE优化
  → ★★★ 但: MoE GRPO → rLLM Tinker也可以 → LoRA all layers → 无需DeepSpeed
  → → ★★★★ 两者都可行 → 选更简的 → rLLM Tinker!
```

## 参考资料

- AutoEP PR#7938: notebook/projects/deepspeed-latest-developments-2026-06.md
- DeepSpeed 0.19 features: notebook/projects/deepspeed-0.19-features-reading.md
- INT4 Triton fallback: notebook/projects/vllm-int4-triton-fallback-reading.md
- MoE serving patterns: notebook/projects/moe-serving-architecture-patterns-reading.md
- RTX 4090 config card: notebook/fundamentals/rtx4090-grpo-training-config-card.md
- RTX 4090 decision tree: notebook/fundamentals/rtx4090-rl-training-decision-tree.md
