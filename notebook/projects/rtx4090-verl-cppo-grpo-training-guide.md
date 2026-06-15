# RTX 4090 verl CPPO+GRPO+bypass_mode+LoRA 实用训练指南

> 2026-06-16 | 实战配置 | verl PR #6731 (OPEN) | TransferQueue sync trainer
> ★★★★★ RTX 4090最优trust region训练方案 — 完整YAML配置 + launch command
> ⚠️ PR #6731尚未merged → 需checkout PR branch或等待merge

## 1. ★★★★★ 配置概要

```
★★★★★★ RTX 4090 CPPO GRPO最优组合:
  → algorithm.adv_estimator = grpo (无critic)
  → algorithm.rollout_correction.bypass_mode = True (无ref model → 省14GB)
  → actor.policy_loss.loss_mode = cppo (比GRPO更好的trust region)
  → detach_metrics_per_micro_batch = True (防OOM)
  → enforce_eager = True (SM89必须)
  → VLLM_USE_V2_MODEL_RUNNER = 0 (SM89必须)

★★★★★★ 推荐模型大小:
  → ★★★★★ Qwen3-4B → 10-14GB peak → 10-14GB headroom → 最佳平衡!
  → ★★★★ Qwen2.5-7B → 17-22GB peak → 2-7GB headroom → 可行但紧凑
  → ★★★★★ Qwen2.5-3B → 8-10GB peak → 充裕 → 简单训练
  → ★★★★★ Qwen2.5-1.5B → 4-5GB peak → 非常充裕 → 快速实验
  → ✗✗✗ Qwen3-30B-A3B (MoE) → NOT feasible on single RTX 4090
```

## 2. ★★★★★ RTX 4090 VRAM预算

```
★★★★★★ RTX 4090 (24GB) VRAM预算:

| Component | 4B BF16 | 7B BF16 | 1.5B BF16 |
|-----------|---------|---------|-----------|
| Base model | ~8GB | ~14GB | ~3GB |
| LoRA (r=32) | ~0.3GB | ~0.5GB | ~0.1GB |
| Adam (LoRA FP32) | ~1GB | ~2GB | ~0.5GB |
| KV cache (rollout) | ~3GB | ~4GB | ~1GB |
| Training buffers | ~1.5GB | ~2GB | ~0.5GB |
| Metrics (detach=True) | ~0GB | ~0GB | ~0GB |
| **Total** | **~13.8GB** | **~22.5GB** | **~5.1GB** |
| **Headroom** | **~10.2GB** | **~1.5GB** | **~18.9GB** |

★★★★★★ RTX 4090推荐:
  → 4B模型 → 最优平衡 (13.8GB peak, 10GB headroom)
  → 7B模型 → 可行但紧凑 (22.5GB peak, 1.5GB headroom → 要offload)
  → 1.5B模型 → 快速实验 (5.1GB peak, 19GB headroom)

★★★★★★ 内存节省措施:
  → bypass_mode=True → 省ref model → ~14GB (7B模型关键!)
  → detach_metrics=True → 省autograd graph → ~0.27GiB/micro-batch
  → LoRA-32 → 只训练adapter → 不需要full model optimizer state
  → param_offload → CPU offload unused params → 省GPU内存
  → optimizer_offload → CPU offload optimizer → 省GPU内存
  → gradient_checkpointing → 省activation memory → 但增加计算时间
```

## 3. ★★★★★ 完整YAML配置 (Qwen3-4B)

```yaml
# RTX 4090 CPPO + GRPO + bypass_mode + LoRA 训练配置
# 模型: Qwen3-4B (dense, 4B参数)
# 需要: pip install TransferQueue (sync trainer)
# Trainer: main_ppo_sync (synchronous TransferQueue trainer)

data:
  train_files: "['data/gsm8k/train.parquet']"
  val_files: "['data/gsm8k/test.parquet']"
  train_batch_size: 64
  max_prompt_length: 512
  max_response_length: 2048
  filter_overlong_prompts: true
  truncation: left

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
  norm_adv_by_std_in_grpo: true
  gamma: 1.0
  rollout_correction:
    bypass_mode: true               # ★★★★★ REQUIRED for CPPO!
    loss_type: ppo_clip
    rollout_is: null
    rollout_is_threshold: 2.0
    rollout_rs: null
    rollout_rs_threshold: null
    rollout_is_batch_normalize: false

actor_rollout_ref:
  model:
    path: Qwen/Qwen3-4B
    use_remove_padding: true
    enable_gradient_checkpointing: true  # ★★★ VRAM savings!
    lora_rank: 32
    lora_alpha: 32
    target_modules: all-linear
    lora_adapter_path: null

  actor:
    strategy: fsdp2
    optim:
      lr: 1e-6
      weight_decay: 0.01
      lr_scheduler_type: constant
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 2   # ★★★ small for 24GB VRAM
    ppo_epochs: 1
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 16384

    # ★★★★★ CPPO config
    policy_loss:
      loss_mode: cppo                 # ★★★★★ KEY: CPPO mode!
      cppo_w_min: 0.8                 # position weight floor
      cppo_delta_b: 0.02              # prefix budget floor
      cppo_delta_b_q: 0.9             # P90 quantile calibration
      cppo_delta_b_k: 1.0             # budget calibration scale
    clip_ratio: 0.15                  # ★★★ dense model delta (NOT epsilon!)
    clip_ratio_c: 20.0                # truncated IS upper bound
    loss_agg_mode: seq-mean-token-sum-norm
    use_kl_loss: false                # ★★★★★ CPPO provides trust region, no KL needed!
    entropy_coeff: 0.01               # compensate for KL=0

    # ★★★★★ RTX 4090 critical settings
    fsdp_config:
      param_offload: true             # CPU offload unused params
      optimizer_offload: true         # CPU offload optimizer (saves ~2GB)

  rollout:
    name: vllm
    tensor_model_parallel_size: 1     # ★★★ single GPU: no TP
    gpu_memory_utilization: 0.5       # conservative for RTX 4090
    n: 4                              # GRPO group size
    enforce_eager: true               # ★★★★★ RTX 4090 MUST (SM89 batch invariance!)
    free_cache_engine: true
    calculate_log_probs: true         # ★★★★★ REQUIRED for bypass_mode + CPPO
    log_prob_micro_batch_size_per_gpu: 2
    log_prob_use_dynamic_bsz: true
    log_prob_max_token_len_per_gpu: 16384

  ref:
    # bypass_mode=True: ref model NOT needed!
    log_prob_micro_batch_size_per_gpu: 2
    log_prob_use_dynamic_bsz: true
    log_prob_max_token_len_per_gpu: 16384
    fsdp_config:
      param_offload: true

trainer:
  n_gpus_per_node: 1                  # ★★★★★ single RTX 4090
  nnodes: 1
  critic_warmup: 0                    # no critic (GRPO)
  total_epochs: 15
  save_freq: 20
  test_freq: 5
  balance_batch: true
  logger: '["console","wandb"]'
  project_name: verl_cppo_grpo_4090
  experiment_name: qwen3_4b_cppo_bypass_lora32

model_engine: fsdp2
detach_metrics_per_micro_batch: true  # ★★★★★ RTX 4090 MUST!
```

## 4. ★★★★★ Launch Command

```bash
# ★★★★★ 环境变量 (RTX 4090必须):
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=0    # ★★★★★ 禁用MRv2 (SM89)
export TRANSFER_QUEUE_ENABLE=1       # ★★★★★ CPPO需要sync trainer

# ★★★★★ Install TransferQueue:
pip install TransferQueue -i https://mirrors.aliyun.com/pypi/simple/

# ★★★★★ Launch (override defaults for RTX 4090):
python3 -m verl.trainer.main_ppo_sync \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=cppo \
    actor_rollout_ref.actor.policy_loss.cppo_w_min=0.8 \
    actor_rollout_ref.actor.policy_loss.cppo_delta_b=0.02 \
    actor_rollout_ref.actor.policy_loss.cppo_delta_b_q=0.9 \
    actor_rollout_ref.actor.policy_loss.cppo_delta_b_k=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.15 \
    actor_rollout_ref.actor.clip_ratio_c=20.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    detach_metrics_per_micro_batch=True \
    model_engine=fsdp2 \
    "$@"
```

## 5. ★★★★★ 关键注意事项

```
★★★★★★ 8个关键注意事项:

1. ★★★★★ PR #6731尚未merged → checkout PR branch:
   → git clone https://github.com/chongqichuizi875/verl -b cppo-policy-loss
   → 或等merge后使用verl main branch

2. ★★★★★ 必须用main_ppo_sync trainer → 不是main_ppo:
   → sync trainer → bypass_mode → old_log_probs = rollout_log_probs → CPPO读μ
   → TransferQueue → pip install TransferQueue → 必须!

3. ★★★★★ enforce_eager=True → RTX 4090 MUST:
   → SM89 CUDA graphs → batch invariance bug → enforce_eager=True禁用compile+graphs
   → ★★★ throughput损失10-15% → 但spec decode正确 → correctness优先!

4. ★★★★★ VLLM_USE_V2_MODEL_RUNNER=0 → SM89 MUST:
   → verl无MRv2 handling → 禁用 → 用MRv1 → safe
   → INT4量化 → MRv1 → 不受MRv2影响

5. ★★★★★ detach_metrics_per_micro_batch=True → MUST:
   → .detach().item() → 防止autograd graph累积 → +0.27GiB/micro-batch OOM
   → ★★★ 28→18GiB → critical for RTX 4090!

6. ★★★★★ bypass_mode=True是CPPO的必需条件:
   → CPPO divergence = |π(y_t|s_t) - μ(y_t|s_t)| → μ=rollout policy
   → bypass → old_log_probs=rollout_log_probs → μ正确
   → ★★★★★ 如果不用bypass → pi_old ≠ μ → divergence measurement错误!

7. ★★★★★ clip_ratio在CPPO中语义不同:
   → 不是PPO的ratio clip epsilon → 而是divergence threshold delta
   → Dense模型: 0.15 → MoE模型: 0.20
   → ★★★★★ 不要设太大 → delta太大 → CPPO失去约束力!

8. ★★★★★ CPPO + bypass_mode → KL penalty冗余:
   → CPPO divergence mask → provably tighter trust region (Theorem 1)
   → → use_kl_loss=False → entropy_coeff=0.01 → 补偿无KL → 简化!
```

## 6. ★★★★★ 与rLLM Tinker对比

```
★★★★★★ RTX 4090 GRPO训练路径对比:

| 维度 | verl CPPO+GRPO+bypass | rLLM Tinker GRPO |
|------|------------------------|------------------|
| Trust region | ★★★★★ CPPO (position-weighted) | ★★★ implicit (rollout stability) |
| Ref model | ✗ (bypass_mode) | ✗ (default bypass) |
| Critic | ✗ (GRPO) | ✗ (GRPO) |
| Process model | Ray actors (跨进程) | In-process (同GPU) |
| detach_metrics | MUST (.detach().item()) | NOT needed (in-process) |
| enforce_eager | MUST (SM89) | NOT needed (不用vLLM) |
| MRv2 | MUST禁用 (VLLM_USE_V2=0) | NOT applicable |
| Setup complexity | ★★★ 高 (Ray+vLLM+CPPO+bypass) | ★★★★★ 低 (rllm train → 一行) |
| Long CoT (>4k) | ★★★★★ CPPO essential | ★★★ OK (GRPO heuristic) |
| Short responses | ★★★★★ OK (CPPO overhead zero) | ★★★★★ OK (simpler) |
| Community | ★★★★★ Large | ★★★ Small |

★★★★★★ 选择指南:
  → 简单setup → rLLM Tinker → rllm train → 一行命令 → 最简单!
  → 长CoT数学推理 → verl CPPO → better trust region → 防止cascading drift
  → 多模态 → verl (Gemma4 + Qwen-VL) → 更完整VLM RL
  → 多GPU → verl HYBRID → 最成熟多GPU路径
```

## 参考
- verl PR #6731: https://github.com/volcengine/verl/pull/6731
- CPPO paper: arXiv:2606.10968
- verl rollout correction docs: docs/algo/rollout_corr.md
- verl GRPO examples: examples/grpo_trainer/
- verl bypass mode: verl-grpo-bypass-mode-reading.md
- SM89 batch invariance: vllm-sm89-batch-invariance-bug-reading.md
- rLLM Tinker: rllm-tinker-backend-deep-reading.md
