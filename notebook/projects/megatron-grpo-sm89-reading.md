# Megatron-LM GRPO Training Recipe & SM 8.9 Implications 源码级深度阅读

> 2026-06-15 | 源码: NVIDIA/Megatron-LM, train_rl.py(425行) + rl_utils.py(2137行) + inference/ + arguments.py
> 核心: Megatron GRPO end-to-end pipeline → DynamicInferenceEngine rollout → cudagraph sizing(exp vs linear) → ProcessGroupCollection → single-GPU behavior
> ★ ★ RTX 4090: GRPO理论上可行但overkill → rLLM Tinker更简单 → cudagraph linear sizing必须 / NVLS/TMA/FP8-training全不可用

## 1. GRPO End-to-End Pipeline — 源码级拆解

### train_rl.py — 入口文件 (425行)

```
train_rl.py → 425行 → 极简入口

★ 入口架构:
  → train_valid_test_datasets_provider → MinimalDataset (1 token dummy!) → RL不关心数据集!
  → _gpt_builder → GPTModel → FP8/TE/BF16 + selective recompute
  → pretrain(full_config, None, model_provider, ModelType.encoder_or_decoder, forward_step)
  → ★ forward_step才是核心 — RL训练的每一步

★ forward_step(data_iterator, model, loss_only=False):
  → runtime_state = get_rl_runtime_state()
  → batch_data = next(data_iterator) → 来自rl_utils准备好的迭代器

  → 如果sequence_packing=True (★推荐):
    → bin_tensor → load_packed_data_by_index → 解包packed data
    → tokens, advantages, old_logprobs, loss_mask, position_ids, ref_logprobs
    → ★ inference_logprobs → IS correction → rl_inference_logprobs_is_correction门控
    → packed_seq_params → FlashInfer THD格式

  → 如果sequence_packing=False:
    → 直接解包batch_data → 手动.cuda()
    → inference_logprobs.cuda() → IS correction门控

  → model_to_use = model[0]
  → ★ 关键: model.eval() → inference模式下cudagraph已capture
  → logprobs_or_hidden_states = get_logprobs(model_to_use, tokens, ...)
  → calculate_grpo_loss(...) → ★ 完整GRPO loss
  → output_tensor = loss → partial(loss_func, ...) → 返回
```

### rl_utils.py — 核心RL逻辑 (2137行)

```
rl_utils.py → 2137行 → Megatron GRPO核心逻辑

★ RLRuntimeState (299-337): 不checkpoint的运行时状态
  → packing_context = None → sequence packing上下文
  → last_collection_iteration = 0 → 上次rollout收集的iteration
  → sequences_this_iteration_on_rank = 0 → 本iteration序列计数
  → throughput metrics: tokens_per_sec / compute_tokens_per_sec / actual_tokens_per_sec

★ ★ calculate_grpo_loss (1855-1943): 核心GRPO loss函数
  → 输入: current_logprobs(pi), old_logprobs(pi_old), ref_logprobs(pi_ref), advantages
  → clamp_eps_lower/upper → ★ asymmetric clamping (DAPO-style: 0.2/0.28)
  → kl_beta → KL(pi||pi_ref)权重 → 默认0.0 → ★ DAPO不用KL!
  → entropy_weight → entropy bonus权重 → 默认0.0

  → ratios = (current - old).exp() → pi/pi_old
  → clamped_ratios = ratios.clamp(1-eps_lower, 1+eps_upper)
  → ★ ★ advantages → packed/unpacked双路径
    → packed: packed_advantages[0, start:end] = advantages[seq_idx].item() → 每个token映射
    → unpacked: advantages.view(-1, 1) → [batch,1] broadcast到[batch,seq]

  → ref_diff = ref - current → kl_term = exp(ref_diff) - ref_diff - 1 → ★ KL=0时无惩罚
  → entropy_term = -exp(current) * current → ★ entropy_weight=0时无奖励

  → ★ ★ IS correction (importance sampling):
    → inference_logprobs可选 → rollout engine计算的logprobs
    → is_weights = (old - inference).exp() → pi_old/pi_inference
    → is_truncation_coef → min(is_weights, coef) → ★ 截断IS → 防止过大IS weights

  → ★ ★ loss = -is_weights * min(ratio * adv, clamped_ratio * adv) + kl_beta * kl - entropy * entropy_term
    → ★ ★ PPO-style clipped loss + KL + entropy → 但GRPO无critic → advantage直接来自reward!

★ ★ calculate_grpo_advantages (824-848): GRPO advantage计算
  → rewards = np.array(rewards) → [groups, group_size]
  → num_turns = np.array(num_turns) → 多turn展开
  → group_turns = num_turns.sum(axis=-1) → 每组总turn数
  → reward_means = rewards.mean(axis=1) → 组均值
  → reward_stds = rewards.std(axis=1) → 组标准差
  → rewards.flatten().repeat(num_turns.flatten()) → 多turn展开
  → ★ ★ return (rewards - reward_means) / (1e-4 + reward_stds)
    → ★ ★ 1e-4 → 防止小group std=0时除零 → 等同Dr.GRPO的norm_adv_by_std_in_grpo=False精神
```

### ★ ★ GRPO Training Loop — get_grpo_data_iterator

```
rl_utils.py → get_grpo_data_iterator (1618-1700):

★ Loop逻辑:
  → global_batches_per_collection = (grpo_prompts_per_step * grpo_group_size) // global_batch_size
  → 判断是否需要收集新rollouts:
    → buffered_rollouts is None OR iteration == last_collection + grpo_iterations * global_batches_per_collection
    → ★ grpo_iterations=1 → 每次collect后只训练1轮 → on-policy

  → 如果需要新rollouts:
    → rollouts = get_environment_rollouts(model, inference_model, optimizer, ...)
    → ★ ★ 这就是MegatronLocal → DynamicInferenceEngine → rollout generation

    → buffered_rollouts, group_stats, example_groups = prepare_data_for_update(...)
      → ★ 这是核心数据准备函数 → 包含ref_logprobs计算

    → ★ ★ 如果optimizer_is_on_cpu:
      → restore_grad_buffers → restore optimizer from CPU → inference完成后恢复训练

  → maybe_log_training_metrics(...) → wandb/tensorboard logging
  → return buffered_rollouts → 循环使用的数据迭代器
```

### ★ ★ prepare_data_for_update — 数据准备核心 (1372行)

```
rl_utils.py → prepare_data_for_update (1200-1572):

★ 流程:
1. compute_group_stats → rewards + advantages
   → ★ advantages = global_advantages = torch.tensor(group_stats.advantages).cuda()

2. ★ ★ DP split → rollouts按DP rank划分
   → data_split_range → 每个DP rank处理1/DP_size的数据
   → advantages也需要split → steps_before slicing
   → ★ 不同于verl → Megatron直接在DP rank间切分

3. prepare_trajectories → tokenize + pad
   → trajs, generation_masks, inference_logprobs
   → ★ ★ inference_logprobs → IS correction的关键 → rollout engine计算的logprobs

4. ★ ★ Ref logprobs计算 — 最复杂部分:
   → ref_state_dict → ref model的state dict
   → ★ ★ 不是单独ref model → 而是ref checkpoint的state dict!

   → 如果sequence_packing:
     → pack_all_trajectories → PackingContext
     → compute_logprobs_batch(model, ref_state_dict, ...) → ★ ★ 核心!
       → swap_model_weights(train_model, ref_model) → refit技术 → ★ 权重替换!
       → model.load_state_dict(ref_state_dict) → ★ 加载ref权重到train model
       → compute_logprobs_batch → 用ref权重forward
       → model.load_state_dict(cur_st_dict) → ★ 恢复train权重
       → ★ ★ 不需要单独ref model → 同一个GPU! → ★ RTX 4090友好!

   → ★ ★ 老logprobs也类似 → swap weights → forward → restore
     → old_logprobs = compute_logprobs_batch(model, cur_st_dict, ...)
     → ★ ★ pi_old是训练model用当前权重计算的 → 不是rollout engine的!

   → ★ inference_logprobs = rollout engine的logprobs → IS correction用

5. ★ ★ Sequence packing完成后:
   → pack_inference_logprobs → packed_inference_logprobs
   → compute_packed_inference_logprobs_stats → log IS mismatch
   → update_microbatch_calculator → 重算microbatch数
   → get_microbatch_dataloader → bins batch_size=1 → ★ packed模式下micro_batch_size必须=1!

   → ★ 如果非packing模式:
     → align_unpacked_inference_logprobs → IS alignment
     → reconfigure_num_microbatches_calculator → 多turn需要重算
     → DataLoader(data, batch_size=micro_batch_size)
```

### ★ ★ Ref Logprobs & Weight Swap — 关键优化

```
★ ★ ★ Weight Swap (refit技术) — 不需要单独ref model GPU!

rl_utils.py → compute_logprobs_batch:
  → cur_st_dict = model.state_dict() → 保存当前训练权重
  → model.load_state_dict(ref_state_dict) → ★ 加载ref权重
  → forward_step_func=partial(logprobs_forward_step) → forward计算ref logprobs
  → model.load_state_dict(cur_st_dict) → ★ 恢复训练权重
  → ★ torch.cuda.synchronize() + gc.collect() + empty_cache → 清理

★ ★ ★ 这意味着:
  → 不需要额外GPU给ref model! → RTX 4090上可行!
  → → verl: ref_in_actor=no_lora_adapter → 同GPU → 但Megatron更直接
  → → rLLM: bypass_mode=true → 跳过ref forward → 更省!
  → → ★ ★ Megatron方式: swap weights → 更通用 → 但swap开销不可忽略
  → → ★ ★ rLLM bypass_mode更好: 零forward pass → 用rollout logprobs替代old_logprobs
```

## 2. DynamicInferenceEngine — Rollout引擎

### MegatronLocal — 本地推理接口

```
megatron/rl/inference/megatron.py → MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):

★ ★ 核心: DynamicInferenceEngine + OpenAI API server → 同进程!

launch(model, **kwargs):
  → get_dynamic_inference_engine(model=model) → ★ 创建推理引擎
  → dp_addr = inference_engine.start_listening_to_data_parallel_coordinator() → DP coordinator
  → rank 0 → start_text_gen_server → ★ OpenAI-compatible API server!
  → rank 0 → InferenceClient → coordinator连接
  → 其他rank → client=None → 只做inference step

  → ★ AsyncOpenAI(base_url=...) → HTTP client → local API
  → concurrency_limit = grpo_prompts_per_step * grpo_group_size * rl_parallel_generation_tasks
  → httpx.Limits(max_connections=concurrency_limit)

★ ★ base_generate → rollout generation:
  → client.chat.completions.create(...) → ★ OpenAI API调用!
  → temperature, top_p, n=1, logprobs=True
  → skip_prompt_log_probs=True → ★ 只取generation logprobs
  → add_BOS → tokenizer.bos控制

  → InferenceResponse:
    → token_ids = prompt + generation → ★ 完整token序列
    → logprobs = generation_log_probs → ★ rollout logprobs → IS correction用
    → ★ policy_epoch, kv_cache_epoch, num_evictions → staleness tracking!

★ ★ suspend/resume — sleep/wake pattern:
  → suspend: client.pause_engines → EngineState.PAUSED → client.suspend_engines → SUSPENDED
  → resume: client.resume_engines → RESUMED → client.unpause_engines → RUNNING
  → ★ ★ 类似vLLM sleep/wake → 但更细粒度 → PAUSED vs SUSPENDED

★ ★ kill — 清理:
  → client.pause_engines → PAUSED
  → client.stop_engines → STOPPED
  → client.shutdown_coordinator → stop
  → stop_text_gen_server → 关闭Flask server
```

### DynamicInferenceEngine — 推理核心

```
megatron/core/inference/engines/dynamic_engine.py → DynamicInferenceEngine:

★ ★ EngineState — 7状态状态机:
  → RUNNING → PAUSING → PAUSED → UNPAUSING
  → SUSPENDING → SUSPENDED → RESUMING → RESUMED
  → STOPPING → STOPPED

★ ★ 核心功能:
  → continuous batching → DynamicInferenceContext → KV cache管理
  → CUDA graph → InferenceBatchDimensions → 多batch维度
  → NCCL/NVLS AllGather dispatcher → TP通信
  → InferenceTopKRouter → @torch.compile + dense_output → FlashInfer兼容
  → MoE → expert_padding → decode CG
  → Router Replay → 记录路由决策 → CG replay时复用 → ★ MoE稳定CG!

★ ★ GRPO rollout时的行为:
  → megatron_rl_inference_mode → context manager → 管理inference/training切换
  → model.eval() → lang_module.eval()
  → ★ toggle_cuda_graphs → inference用local CG → training用global CG(可不同!)
  → inference_interface.set_generation_epoch → ★ staleness tracking
  → inference_interface.resume → RUNNNG → 生成rollouts

  → ★ ★ inference完成后:
    → inference_interface.suspend → SUSPENDED → GPU释放
    → toggle_cuda_graphs(lang_module, 'none') → ★ 关闭CG → training重新capture
    → restore cuda_graph_impl → 训练CG scope → MoE partial capture
    → ★ ★ _maybe_prefetch_separate_inference_model_weights(to_cpu=True)
      → UVM: advise_managed_module_parameters_preferred_location(device=-1) → CPU
      → torch_memory_saver: torch_memory_saver.pause("rl_inference_model") → CPU
    → ★ ★ offload optimizer → restore optimizer → inference完成后恢复训练
```

### ★ ★ megatron_rl_inference_mode — 关键Context Manager

```
rl_utils.py → megatron_rl_inference_mode (1955-2070):

★ ★ ★ 这是整个RL训练中最关键的context manager!

流程:
1. ★ 设置inference CG scope:
   → model[0].config.cuda_graph_modules = [] → ★ full-layer capture
   → model[0].config.cuda_graph_impl = "local" → ★ inference用local CG
   → transition_moe_cudagraphs(lang_module, 'full') → MoE full capture

2. model.eval()
3. _maybe_prefetch_separate_inference_model_weights(to_cpu=False) → ★ GPU恢复
4. rotary_pos_emb.cache_clear() → ★ ★ 清除LRU cache → inference mode下的缓存会破坏训练!

5. torch.no_grad():
   → 如果offload_optimizer_during_inference:
     → ★ ★ model.offload_grad_buffers() → ★ 释放gradient buffers到CPU
     → ★ ★ optimizer.offload_to_cpu() → optimizer state移到CPU
     → ★ ★ 这是RTX 4090最关键的功能! → inference时optimizer占GPU → 释放给inference!

   → toggle_cuda_graphs → inference CG
   → inference_interface = get_inference_interface(args, loop, model)
   → inference_interface.set_generation_epoch → staleness
   → loop.run_until_complete(inference_interface.resume())

   → ★ ★ yield inference_interface → rollout generation在此发生

   → inference_interface.suspend() → 暂停推理
   → toggle_cuda_graphs → 关闭inference CG

6. ★ ★ restore training state:
   → set_decode_expert_padding(unwrap_model(model), set_to=False) → 清除inference padding
   → restore cuda_graph_impl → args.cuda_graph_impl
   → restore cuda_graph_modules → ★ 训练CG scope
   → transition_moe_cudagraphs → 'partial' → ★ MoE partial capture for training

   → ★ ★ _maybe_prefetch_separate_inference_model_weights(to_cpu=True) → inference weights回CPU

   → 如果offload_optimizer:
     → restore_grad_buffers → ★ GPU恢复gradient buffers
     → restore_from_cpu → ★ optimizer state回GPU

   → training_lang_module.train() → ★ 恢复training mode
   → rotary_pos_emb.cache_clear() → ★ 再次清除LRU cache
```

## 3. ★ ★ CUDA Graph Sizing — Exponential vs Linear (RTX 4090关键!)

### CudaGraphSizingDistribution Enum

```
megatron/core/inference/config.py → CudaGraphSizingDistribution (120-133):

★ EXPONENTIAL (默认) → log-spaced from max_tokens down to tp_size
  → ~log2(max_tokens) graphs → bounded 2x worst-case padding → ★ 推理最优
  → 9B: 60 graphs → 15GB内存节省 → throughput 137.488 tok/s

★ LINEAR → [1,2,4] + range(8,256,8) + range(256,max+1,16)
  → 更密集的小尺寸graphs → ★ GRPO需要 → rollout shape多变
  → ★ ★ RTX 4090必须用linear → 防止exponential regression
```

### ★ ★ ★ PR #3509 → Exponential Regression Root Cause

```
★ ★ ★ PR #3509 — 从linear→exponential → 独立推理更优但GRPOregression!

原设计(linear):
  → 123 graphs → mem 33.2/40.2 GB
  → throughput 135.500 tok/s

新设计(exponential):
  → 60 graphs → mem 29.8 allocated / 36.3 reserved → ★ 15GB节省!
  → throughput 137.488 tok/s → ★ 更好!
  → ★ ★ 但mixed prefill grid有变化 → geometric {1,2,4,8,...,max_requests}

★ ★ Regression根因:
  → GRPO场景: rollout batch shape变化多端
  → exponential sizing在低端过多graphs → 但GRPO需要稳定的小尺寸匹配
  → mixed prefill grid → 固定P=16→改为geometric → 实际P≠16时 → CG捕获但无法replay
  → ★ ★ peak 69.2 GB vs 60.9 GB → +13.6% 峰值内存 → pool compounding!

★ ★ 理解:
  → 独立推理 → batch稳定 → exponential → 稀疏大graph → 省内存 → ★ 优化
  → GRPO → batch多变 → linear → 密集小graph → 稳定匹配 → ★ 稳定
  → ★ ★ ★ 不同场景需要不同策略! → 不能一刀切!
```

### ★ ★ ★ PR #5280 → Fix: Linear Sizing for GRPO

```
★ ★ ★ PR #5280 — Pin linear sizing for GRPO

Fix:
  → --inference-dynamic-batching-cuda-graph-sizing-distribution: linear
  → gpt_grpo_tp4_pp1_dp2_8b_throughput + cudagraphs → ★ CI pin linear

CI config YAML:
  → MODEL_ARGS:
    → --inference-dynamic-batching-cuda-graph-sizing-distribution: linear → ★ ★ 强制linear!
    → # Pin the pre-#3509 (linear) CUDA-graph sizing distribution.
    → # #3509 switched the default to exponential + a mixed-prefill grid,
    → # which raised peak memory and tripped this test's mem-allocated-bytes
    → guardrail (~69GB vs ~61GB golden, +13.6%).

★ ★ ★ RTX 4090配置:
  → GRPO → ★ ★ ★ 必须用linear sizing → 否则内存+13.6% → 24GB更不够!
  → 推理only → 可以用exponential → 省15GB → 但需要稳定batch
  → ★ ★ 7B GRPO → linear sizing → 最少graphs → 内存可控
```

### CUDAGraphBatchDimensionBuilder — 源码级

```
megatron/core/inference/batch_dimensions_utils.py → CUDAGraphBatchDimensionBuilder:

★ ★ _calculate_cuda_graph_token_counts (exponential):
  → num_cuda_graphs == -1 → auto_size → log2(max_tokens) halvings
  → 从cuda_graph_max_tokens → // 2 → 每次取整到rounder(2) × tp_size
  → 去重 → sort descending → trim from middle if > num_cuda_graphs
  → ★ ★ endpoints: cuda_graph_max_tokens + tp_size → 总包含最大和最小

★ ★ _calculate_token_counts_linear:
  → num_cuda_graphs == -1 → [1,2,4] + range(8,256,8) + range(256,max+1,16)
  → TP-aligned + dedupe → reverse → descending order
  → ★ ★ ★ 小尺寸有1和2 → decode single-request → GRPO rollout常见!

  → explicit N → even stride = round_up(max/N, rounder), TP-aligned
  → ★ ★ N=1 → [cuda_graph_max_tokens] → 单graph → GRPO inference可能用这个!

★ ★ RTX 4090 → tp_size=1 → 所有align无效果 → 直接按rounder=2取整
  → ★ linear: [1,2,4,8,16,32,...,512,528,...,max] → 完整覆盖
  → ★ exponential: [max, max//2, max//4, ...] → 稀疏 → GRPO不稳
```

## 4. ProcessGroupCollection & Single-GPU Behavior

### ProcessGroupCollection — 新架构

```
megatron/core/process_groups_config.py → ProcessGroupCollection:

★ ★ 从parallel_state globals → ProcessGroupCollection dataclass → ★ 渐进迁移!

18+ PG fields:
  → tp, pp, mp, embd, pos_embd, cp, tp_cp, hcp
  → ep, expt_tp, tp_ep, tp_ep_pp
  → dp, dp_cp, dp_cp_ag
  → expt_dp, expt_dp_ag
  → intra_dp_cp, intra_expt_dp, inter_dist_opt, intra_dist_opt

★ ★ __init__(**kwargs) → 只设置kwargs中提供的 → 其他保持init=False
  → ★ ★ 单GPU: 所有PG = None → 不需要任何PG → singleton world!
  → → ★ 不影响功能 → 只是所有通信都是identity → no-op

★ ★ __repr__ → 简洁显示哪些PG存在和size → 调试友好
```

### PR #5260 — MIMO Hetero Topology

```
★ ★ PR #5260 — Add MIMO hetero topology + distributed bootstrap

核心:
  → HyperCommGrid → per-module topology → TP/CP/PP/DP per module
  → MultiModuleProcessGroupCollection → 多模块各自PG
  → examples/mimo/training/ → bootstrap代码

★ ★ ★ 单GPU行为:
  → HyperCommGrid([1,1,1,1], ["tp","cp","dp","pp"]) → world=1
  → → 所有PG = singleton → 没有通信 → identity → ★ 正确退化!
  → → ProcessGroupCollection → 所有PG = dist.new_group([0]) → 单rank组
  → ★ ★ RTX 4090 → 世界=1 → 所有PG=singleton → 无额外开销!

★ ★ 异构GPU场景(MIMO):
  → 不同模块不同TP/EP → 各自HyperCommGrid → 各自PG
  → PP group必须匹配 → decoder_pp_enum == expert_pp_enum → ★ 约束
  → validate → partition world [0, world_size) → no gaps → pairwise-disjoint-or-shared
  → → RTX 4090: 单GPU → 不需要MIMO → 简单退化
```

### parallel_utils.py — Inference PG Builder

```
megatron/rl/parallel_utils.py → build_inference_pg_collection:

★ ★ ★ ★ inference model可以有不同的并行配置!

build_inference_pg_collection(world_size, tp_size, pp_size, cp_size, ep_size, ...):
  → 默认使用training的TP/PP/CP/EP → 但可以覆盖!
  → ★ ★ 例如: training TP=4 → inference TP=2 → ★ 不同并行度!
  → decoder_grid: tp, cp, dp, pp → dense layers
  → expert_grid: tp, ep, dp, pp → MoE layers
  → ★ PP groups必须匹配 → decoder_pp == expert_pp → 约束

★ ★ ★ RTX 4090 → inference PG:
  → tp_size=1, pp_size=1, cp_size=1, ep_size=1 → 所有singleton
  → dp_size = 1 // (1*1*1) = 1 → 单GPU
  → → ProcessGroupCollection → 所有PG = singleton → ★ 正确
  → → ★ inference model可以用不同并行度 → 但单GPU没区别
```

## 5. GRPO Model Configs — 7B规模配置分析

### Qwen3-4B Config (最接近RTX 4090)

```
examples/rl/model_configs/qwen3_4b.sh:

★ 模型参数:
  → 36 layers, hidden=2560, ffn=9728, heads=32, GQA-8
  → vocab=151936, RoPE, swiglu, qk-layernorm
  → TP=1, PP=1 → ★ ★ 单GPU!

★ GRPO参数:
  → GRPO_CLAMP_EPS=0.2/0.2 → symmetric → vanilla GRPO
  → MAX_INFERENCE_BS=32 → rollout batch size
  → GRPO_GROUP_SIZE=16 → 每prompt16个response → ★ 标准配置
  → GRPO_PROMPTS_PER_STEP=64 → 每步64个prompt
  → GRPO_ITERATIONS=1 → on-policy → ★ 每次只训练1轮
  → GRPO_KL_BETA=0.0 → ★ DAPO不用KL → 与verl一致
  → TRAINING_BATCH_SIZE=256 → global batch size
  → MAX_SEQ_LENGTH=32768 → ★ 长序列 → RTX 4090可能需要缩短

★ 训练参数:
  → micro_batch_size=1 → gradient accumulation 256步
  → lr=1e-6 → min_lr=1e-7 → constant decay
  → clip-grad=1.0 → weight-decay=0.01
  → selective recompute → core_attn → ★ ★ 省activation memory
  → adam → beta1=0.9, beta2=0.999 → ★ ★ 不是DAPO的0.95!

★ ★ RTX 4090可行性分析:
  → 4B参数 → BF16 → ~8GB → optimizer Adam → ~16GB → total ~24GB → ★ 紧!
  → selective recompute → 省activation → 但24GB内可能还行
  → ★ ★ 但: inference model同GPU → 需要offload optimizer → rl_offload_optimizer_during_inference
  → → ★ inference时optimizer→CPU → model→GPU inference → ★ 24GB内可行!
  → → ★ ★ 但没有LoRA → 全参数训练 → optimizer state很大 → RTX 4090不一定够
  → → ★ ★ ★ 建议: LoRA rank=16 → optimizer state省97% → 才可行!
```

### Qwen3-8B Config (需要优化才能在RTX 4090跑)

```
examples/rl/model_configs/qwen3_8b.sh:

★ 模型参数:
  → 36 layers, hidden=4096, ffn=12288, heads=32, GQA-8
  → vocab=151936, RoPE, swiglu, qk-layernorm
  → TP=1, PP=1 → ★ 单GPU配置 → 但需要优化!

★ GRPO参数:
  → 同4B → GRPO_CLAMP_EPS=0.2/0.2, GRPO_GROUP_SIZE=16, GRPO_KL_BETA=0.0
  → TRAINING_BATCH_SIZE=256 → ★ 大batch → RTX 4090可能需要减少
  → MAX_SEQ_LENGTH=32768 → ★ 太长! → RTX 4090需要8192或更短

★ ★ RTX 4090可行性:
  → 8B BF16 → ~16GB → Adam optimizer → ~32GB → total ~48GB → ★ ❌ 不可行!
  → → ★ ★ 必须: LoRA + CPU_Adam / 或 INT4 inference + LoRA training
  → → ★ ★ ★ 8B → LoRA rank=16 → trainable params ~0.5% → optimizer ~1GB → 可行!
  → → → 但Megatron没有原生LoRA支持 → ★ ★ 需要修改或用rLLM!
```

### Nemotron6-3B-MoE Config (MoE GRPO)

```
examples/rl/model_configs/nemotron6_3b_moe.sh:

★ ★ MoE GRPO配置 → EP=32 → 需要128 GPU!

  → TP=2, PP=1, EP=32 → ★ ★ 多GPU only!
  → rl-skip-bos-token → MoE tokenizer特殊
  → no-sequence-packing → ★ MoE不用packing → 可能因为EP通信
  → rl-partial-rollouts → ★ ★ 多turn partial rollout
  → rl-inference-logprobs-is-correction → ★ ★ IS correction启用!
  → rl-importance-sampling-truncation-coef=10.0 → ★ ★ IS truncation
  → moe-pad-experts-for-cuda-graph-inference → ★ MoE CG padding
  → decode-only-cuda-graphs → ★ ★ 只capture decode CG → 稳定!
  → moe-router-dtype fp64 → ★ router用FP64 → 精度
  → moe-token-dispatcher-type alltoall → EP AllToAll通信

★ ★ RTX 4090 → EP=32不可行 → EP=1 → 单GPU → 但MoE需要EP通信
  → → ★ ★ 3B MoE EP=1 → 所有expert同GPU → 内存可能够
  → → → 但: MoE inference CG padding → 需要expert alignment → ★ 复杂
  → → → ★ ★ ★ 结论: MoE GRPO → RTX 4090可行但需要EP=1 → 比dense更复杂
```

### Qwen3-30B-A3B MoE Config (需要多GPU)

```
examples/rl/model_configs/qwen3_30b_a3b_moe.sh:

★ ★ 大MoE → TP=4 → 需要4+ GPU
  → 128 experts, topk=8 → ★ 活跃参数多
  → moe-ffn-hidden-size=768 → 小expert → ★ 参数效率高
  → ★ ★ RTX 4090: 完全不可行 → TP=4 → 需要4GPU → PCIe scaling灾难
```

### ★ ★ Common.sh — 共享配置

```
examples/rl/model_configs/common.sh:

★ ★ 核心共享配置:
  → CUDA_DEVICE_MAX_CONNECTIONS=1 → NCCL优化
  → transformer_engine → TE backend → ★ RTX 4090: TE可用
  → bf16 → ★ 正确训练精度
  → inference-dynamic-batching-num-cuda-graphs=1 → ★ 单CG → 简化!
  → inference-dynamic-batching-unified-memory-level=1 → ★ UVM KV cache

  → ★ ★ ★ CUDA graph 配置:
    → ENABLE_CUDA_GRAPH=true → 默认启用
    → --cuda-graph-impl=local → ★ local CG → 不是global
    → --rl-persist-cuda-graphs → ★ ★ ★ 持久化CG → inference/train共享!
    → → ★ ★ RTX 4090: local CG ✓ / persist CG ✓ → 但需要linear sizing!

  → adam-beta1=0.9, adam-beta2=0.95 → ★ ★ 注意: DAPO用0.95!
    → → Nemotron MoE用0.999 → ★ ★ 不同模型不同beta2!
```

## 6. GRPO Loss — 数学分析

### ★ ★ ★ Megatron GRPO Loss公式

```
calculate_grpo_loss → 完整公式:

★ ★ ★ Loss per token:

L_i = -w_IS * min(r_i * A_i, clip(r_i, 1-ε_l, 1+ε_u) * A_i) + β_KL * KL_i - β_H * H_i

其中:
  r_i = exp(log π_i - log π_old_i) → pi/pi_old ratio
  A_i = advantage → group-relative → (reward - mean_group) / (std_group + 1e-4)
  ε_l = clamp_eps_lower (默认0.2)
  ε_u = clamp_eps_upper (默认0.2 → symmetric / DAPO: 0.28 → asymmetric)
  β_KL = kl_beta (默认0.0 → ★ DAPO不KL!)
  β_H = entropy_weight (默认0.0 → ★ DAPO不entropy!)

★ ★ IS correction (importance sampling):
  w_IS = exp(log π_old - log π_inference) → pi_old / pi_inference
  → ★ ★ rollout engine(inference)和训练model的logprobs不同 → IS correction
  → truncation: min(w_IS, is_truncation_coef) → ★ 截断 → 防止过大IS weights

★ ★ ★ 关键对比 → verl vs Megatron vs rLLM:
  → verl: GRPO_VECTORIZED → vectorized advantage → 10-100x快 → 但loss同公式
  → Megatron: 标准PPO-style loss → 但有IS correction → inference_logprobs → ★ 更严谨
  → rLLM: bypass_mode → π_old = rollout logprobs → 零ref forward → ★ 最省!
  → → ★ ★ ★ rLLM bypass_mode等效于IS correction=1 → π_old=π_inference → w_IS=1 → 无IS开销

★ ★ KL term:
  KL_i = exp(ref_diff) - ref_diff - 1 → ★ ★ 不是标准KL → 而是log-ratio近似
  → ref_diff = log π_ref - log π → ref - current
  → 当β_KL=0 → KL term消失 → ★ DAPO: KL在reward端不是loss端 → 但Megatron可选loss端
```

## 7. ★ ★ RTX 4090 SM 8.9 Implications — 完整分析

### SM 8.9 (Ada Lovelace) Capability Matrix

```
★ ★ ★ RTX 4090 SM 8.9 → Capability Matrix:

功能                 SM 8.9支持   GRPO需要   状态
CUDA graph           ✓            ✓          ★ 可用
NCCL AllGather       ✓            ✓(TP>1)    ★ 但TP=1不需要
NVLS AllGather       ✗(SM90)      ✓(TP>1)    ✗ → TP=1 anyway
TMA (Hopper)         ✗(SM90)      inference  ✗ → 但不required
FP8 training         ✗(SM89)      optional   ✗ → BF16是唯一正确精度
FP8 inference        ✗(SM89)      optional   ✗ → INT4替代
FP16/BF16            ✓            ✓          ★ 可用
Flash Attention      ✓            ✓          ★ 可用
FlashInfer           ✓            ✓(decode)  ★ 可用
INT4 quantization    ✓            inference  ★ ★ 可用 → 关键!
INT8 KV cache        ✓            inference  ★ ★ 可用 → 关键!
UVM (unified memory) ✓            inference  ★ 可用 → KV cache
torch_memory_saver   ✓            inference  ★ 可用 → optimizer offload
selective recompute  ✓            training   ★ 可用 → 省activation memory
ProcessGroupCollection ✓          PG管理     ★ 可用 → singleton退化正确
```

### ★ ★ ★ RTX 4090 GRPO可行性 — 逐项分析

```
★ ★ ★ 7B GRPO on RTX 4090 — 完整可行性分析:

内存预算(24GB):
  → 模型权重 BF16: ~14GB (7B × 2 bytes)
  → → ★ INT4推理: ~3.5GB → ★ ★ 省空间 → inference model在GPU上可以共存
  → Optimizer (Adam): ~28GB → ❌ 超出!
  → → ★ ★ CPU_Adam: optimizer在CPU → GPU只存gradients → ★ 可行
  → → ★ ★ LoRA rank=16: trainable ~0.5% → optimizer ~1GB → ★ ★ ★ 可行!
  → Activation memory: ~2-4GB → selective recompute → ★ 可控
  → KV cache: INT8 → ~2GB → ★ 可控
  → CUDA graphs: linear sizing → 稳定 → ★ 内存可控

★ ★ ★ 推荐配置:
  → Model: 7B dense (如Qwen3-8B或Qwen3-4B)
  → Precision: BF16 training + INT4 inference
  → Optimizer: LoRA rank=16 + CPU_Adam → 或 LoRA + fused adam on GPU
  → CUDA graph: local + linear sizing + rl-persist-cuda-graphs
  → Offload: rl-offload-optimizer-during-inference → ★ inference时optimizer→CPU
  → Sequence packing: rl-use-sequence-packing → ★ 提升compute efficiency
  → GRPO: clamp_eps=0.2, kl_beta=0.0, group_size=16, prompts_per_step=64
  → IS correction: rl-inference-logprobs-is-correction → ★ 可选 → 但overhead

★ ★ ★ 但 ★ ★ ★ Megatron没有原生LoRA支持!
  → → 需要修改model provider → 手动注入LoRA → ★ 复杂
  → → → ★ ★ ★ RTX 4090最优方案: rLLM Tinker + GRPO + LoRA → ★ 简单得多!
  → → → → rLLM: LoRA auto-init → zero-copy weight sync → bypass mode → ★ 全自动!
```

### ★ ★ ★ 框架对比 — RTX 4090 GRPO最优选择

```
★ ★ ★ 框架对比 → RTX 4090 GRPO训练:

维度              Megatron-LM         verl               rLLM Tinker
LoRA              ❌ 无原生支持         ✓ LoRA auto-init    ★ ★ ★ LoRA auto-init
Optimizer offload ✓ CPU_Adam          ✓ CPU_Adam          ★ ★ ★ in-process → CPU_Adam可选
IS correction     ✓ inference_logprobs ✓ (bypass_mode)     ★ ★ ★ bypass_mode=true
Sequence packing  ✓ FIFO/other        ✓ (DataProto)       ★ ★ ★ prefix-merge
CUDA graph        ✓ local + linear    ✓ (vLLM CG)         ★ ★ ★ (vLLM CG)
Weight swap       ✓ refit technique   ✓ Ray IPC           ★ ★ ★ zero-copy → SamplingClient
Rollout engine    ✓ DynamicInference  ✓ vLLM/VLLM         ★ ★ ★ ServiceClient
Single GPU        ✓ TP=1 PG=singleton ✓ colocated          ★ ★ ★ in-process → 最优
复杂度            ★★★★★ 极复杂         ★★★ 中等             ★ ★ ★ ★ ★ 极简单
RTX 4090最优      ❌ overkill          ✓ 可行但需配置        ★ ★ ★ ★ ★ ★ 最优!

★ ★ ★ ★ ★ ★ 结论:
  → Megatron GRPO → 功能最完整 → 但对RTX 4090是overkill
  → → 单GPU → 所有并行=singleton → Megatron并行优势完全消失
  → → 没有LoRA → 全参数训练 → RTX 4090内存不够
  → → 需要: 自定义LoRA + CPU_Adam + INT4 inference + offload → ★ 配置复杂
  → → → ★ ★ ★ rLLM Tinker: 一行配置 → 自动LoRA → bypass → zero-copy → 最优!

  → → ★ ★ ★ 但Megatron GRPO代码价值在于:
    → 1. GRPO loss完整实现 → IS correction → ★ 最严谨
    → 2. cudagraph sizing策略 → exponential/linear → ★ 深入理解
    → 3. megatron_rl_inference_mode → context manager → ★ RL inference/training切换范式
    → 4. ProcessGroupCollection → PG管理 → ★ 多GPU架构设计学习
    → 5. MoE GRPO → EP + MoE CG + router replay → ★ ★ MoE RL学习
```

## 8. GRPO Arguments — 完整参数表

```
★ ★ ★ Megatron GRPO参数 → arguments.py关键参数:

★ GRPO核心参数:
  --grpo-group-size: 16 → 每prompt多少response → ★ GRPO group size
  --grpo-prompts-per-step: 64 → 每步多少prompt → ★ effective batch = 64*16=1024
  --grpo-iterations: 1 → reuse data次数 → ★ on-policy=1
  --grpo-clamp-eps-lower: 0.2 → ratio下限 → ★ symmetric=0.2/0.2, DAPO=0.2/0.28
  --grpo-clamp-eps-upper: 0.2 → ratio上限 → ★ ★ asymmetric clamping → DAPO创新
  --grpo-kl-beta: 0.0 → KL权重 → ★ DAPO不用KL → verl也默认0.001
  --grpo-entropy-term-weight: 0.0 → entropy权重 → ★ DAPO不用entropy
  --grpo-samples-per-iteration: → prompts * group_size → ★ 自动计算

★ RL inference参数:
  --rl-persist-cuda-graphs: → ★ ★ 持久化CG → inference/train共享CG
  --rl-training-cuda-graphs: → ★ ★ 训练CG → inference CG不同scope
  --rl-offload-optimizer-during-inference: → ★ ★ ★ inference时optimizer→CPU → RTX 4090关键!
  --rl-offload-inference-model-weights-when-idle: → inference model weights offload
  --rl-inference-model-unified-memory-level: 1 → UVM → inference model KV cache
  --rl-kv-cache-management-mode: persist/offload → KV cache管理
  --rl-use-sequence-packing: → ★ ★ FIFO packing → compute效率
  --rl-sequence-packing-max-sequences-per-bin: → bin容量
  --rl-sequence-packing-algo: fifo → packing算法
  --rl-partial-rollouts: → ★ ★ 多turn partial rollout
  --rl-skip-bos-token: → tokenizer BOS处理
  --rl-inference-logprobs-is-correction: → ★ ★ IS correction → inference_logprobs参与loss
  --rl-importance-sampling-truncation-coef: 10.0 → ★ IS truncation上限
  --rl-default-top-k: -1 → 默认采样top-k
  --rl-default-temperature: 1.0 → 默认采样温度
  --rl-default-top-p: 1.0 → 默认采样top-p
  --rl-prompts-per-eval: 32 → evaluation prompts数
  --rl-num-parallel-generations: → ★ 并行生成 → rollout并行
  --rl-generation-batch-size: → generation batch

★ CUDA graph参数:
  --inference-dynamic-batching-cuda-graph-sizing-distribution: linear → ★ ★ ★ RTX 4090必须!
  --inference-dynamic-batching-num-cuda-graphs: 1 → ★ 最少graph → 简化
  --inference-dynamic-batching-buffer-size-gb: 20 → KV cache buffer
  --inference-dynamic-batching-unified-memory-level: 1 → UVM
  --cuda-graph-impl: local → ★ local CG → 不是global

★ ★ ★ RTX 4090最小配置参数集:
  --tensor-model-parallel-size 1 → TP=1
  --pipeline-model-parallel-size 1 → PP=1
  --bf16 → BF16训练
  --inference-dynamic-batching-cuda-graph-sizing-distribution linear → ★ ★ ★ 必须!
  --inference-dynamic-batching-num-cuda-graphs 1 → 最少CG
  --cuda-graph-impl local → local CG
  --rl-persist-cuda-graphs → 持久化CG
  --rl-offload-optimizer-during-inference → ★ inference时optimizer→CPU
  --grpo-group-size 16 → standard group size
  --grpo-prompts-per-step 64 → standard prompts
  --grpo-kl-beta 0.0 → DAPO-style
  --recompute-granularity selective → 省activation memory
  --recompute-activations → selective recompute
  --recompute-modules core_attn → 只recompute attention
```

## 9. GRPO Training Loop — 完整流程

```
★ ★ ★ Megatron GRPO Training Loop — 9-step pipeline:

1. ★ 初始化:
   → pretrain(full_config, MinimalDataset, model_provider, forward_step)
   → MinimalDataset → RL不关心训练数据 → rollout生成才是数据来源

2. ★ Each iteration → get_grpo_data_iterator:
   → 判断是否需要新rollouts:
     → buffered_rollouts is None → 需要
     → iteration >= last_collection + grpo_iterations * global_batches → 需要
   → ★ grpo_iterations=1 → 每次只训练1轮 → on-policy

3. ★ ★ Rollout generation → megatron_rl_inference_mode:
   → model.eval() → inference CG scope → persist CG
   → optimizer.offload_to_cpu() → ★ GPU释放给inference
   → inference_interface.resume() → DynamicInferenceEngine → RUNNING
   → → OpenAI API → generation → ★ rollout tokens + logprobs
   → → ★ ★ policy_epoch/kv_cache_epoch/num_evictions → staleness tracking
   → inference_interface.suspend() → SUSPENDED

4. ★ ★ Data preparation → prepare_data_for_update:
   → compute_group_stats → rewards + advantages
   → DP split → rollouts按rank划分
   → prepare_trajectories → tokenize + pad + generation_masks

5. ★ ★ ★ Ref logprobs → weight swap (refit):
   → cur_st_dict = model.state_dict() → 保存训练权重
   → model.load_state_dict(ref_state_dict) → 加载ref权重 → ★ 同GPU!
   → forward → ref_logprobs
   → model.load_state_dict(cur_st_dict) → 恢复训练权重

6. ★ ★ ★ Old logprobs → 同样weight swap:
   → model.load_state_dict(cur_st_dict) → ★ ★ 用当前训练权重计算π_old
   → forward → old_logprobs
   → ★ ★ inference_logprobs → rollout engine → IS correction

7. ★ Sequence packing (if enabled):
   → pack_all_trajectories → FIFO algorithm → bins
   → pack inference_logprobs → packed data
   → update_microbatch_calculator → 重算batch
   → DataLoader → batch_size=1 → ★ packed = 1 microbatch per bin

8. ★ ★ Training → forward_step:
   → next(data_iterator) → get bin/batch data
   → tokens, advantages, old_logprobs, ref_logprobs, inference_logprobs
   → model.eval() → ★ ★ inference CG模式 → persist CG → replay!
   → logprobs = get_logprobs(model, tokens, ...) → ★ ★ 当前π
   → calculate_grpo_loss → ★ ★ ★ 完整GRPO loss
   → output_tensor → loss_func → masked loss → DP all-reduce

9. ★ ★ Optimizer step:
   → pretrain内部 → backward → optimizer step → standard Megatron training
   → → ★ ★ 注意: Megatron GRPO用的是标准pretrain loop → 不是自定义RL loop!
   → → → forward_step提供loss → pretrain做backward+optimizer → ★ ★ 极简!
```

### ★ ★ ★ vs verl vs rLLM 流程对比

```
★ ★ ★ 三框架GRPO流程对比:

Step          Megatron                  verl                   rLLM Tinker
数据来源       MinimalDataset(dummy)     custom_reward_function   custom_reward_function
Rollout        DynamicInferenceEngine    vLLM/SGLang HTTP         ServiceClient (in-process)
Weight Swap    refit(cur→ref→cur)       Ray IPC/ZMQ             ★ ★ ★ zero-copy SamplingClient
IS correction  inference_logprobs       bypass_mode(=skip ref)   ★ ★ ★ bypass_mode(true)
Loss           calculate_grpo_loss      GRPO_VECTORIZED          ★ ★ ★ Tinker IS loss
Optimizer      MegatronOptimizer        FSDP2/ZeRO              ★ ★ ★ in-process Adam
Loop           pretrain(标准)            Trainer.train()          ★ ★ ★ UnifiedTrainer 8-stage
Offload        optimizer→CPU during inf  sleep/wake              ★ ★ ★ sleep/wake(vLLM level)

★ ★ ★ Megatron特色:
  → ★ ★ IS correction → inference_logprobs → 最严谨的off-policy修正
  → ★ ★ refit weight swap → 同GPU做ref → 不需要额外GPU → RTX 4090友好
  → ★ ★ ★ 但: 没有bypass_mode → 总需要ref forward → ★ 比rLLM多1个forward!

  → ★ ★ rLLM特色:
  → → ★ ★ ★ bypass_mode → π_old = rollout logprobs → 零ref forward → ★ 最省!
  → → → 等效于Megatron IS correction中 w_IS=1 → π_old=π_inference → 但更简单
  → → → ★ ★ ★ RTX 4090最优 → 少一个ref forward → 省50% compute → LoRA auto → 最优!
```

## 10. ★ ★ ★ RTX 4090 终极结论

```
★ ★ ★ ★ ★ ★ RTX 4090 GRPO训练终极结论:

1. ★ ★ ★ Megatron GRPO → 功能最完整 → 但RTX 4090上是overkill:
   → 所有并行=singleton → 并行优势消失
   → 无LoRA → 全参数训练 → 内存不够 → ★ 需要大量自定义
   → IS correction → 最严谨 → 但overhead大 → RTX 4090不需要(off-policy少)
   → → ★ ★ ★ 结论: Megatron GRPO用于学习 → 不用于RTX 4090实战

2. ★ ★ ★ CUDA graph sizing → linear必须:
   → exponential → +13.6% peak memory → GRPO regression
   → linear → 稳定 → 小尺寸graphs → rollout shape多变 → ★ 正确
   → → ★ ★ ★ RTX 4090 GRPO: --inference-dynamic-batching-cuda-graph-sizing-distribution linear

3. ★ ★ ★ ProcessGroupCollection → singleton退化正确:
   → 单GPU → 所有PG=singleton → 无通信开销 → ★ 正确退化
   → MIMO → 多模块异构 → 单GPU不需要 → 简单退化
   → → ★ 单GPU上Megatron和rLLM同样简单 → 但rLLM有LoRA!

4. ★ ★ ★ DynamicInferenceEngine → RTX 4090可用:
   → NCCL ✓ / NVLS ✗ → TP=1 → 不需要NVLS
   → Flash Attention ✓ / FlashInfer ✓ → decode优化
   → CUDA graph ✓ → local CG + persist → ★ 可用
   → → ★ ★ inference engine本身不是问题 → 但整体配置复杂

5. ★ ★ ★ ★ ★ ★ RTX 4090最优方案:
   → ★ ★ ★ rLLM Tinker + GRPO + LoRA-32 + bypass_mode=true
   → → 1行配置 → 自动LoRA → zero-copy → bypass → ★ ★ ★ 最简!
   → → vs Megatron: 需要手动LoRA + CPU_Adam + offload + INT4 → ★ ★ ★ 复杂得多!

   → → ★ ★ ★ 但学习Megatron GRPO代码 → 深入理解:
     → IS correction → inference_logprobs → ★ off-policy理论
     → cudagraph sizing → exponential vs linear → ★ ★ ★ 场景特定优化
     → megatron_rl_inference_mode → context manager → ★ RL inference/training切换
     → refit weight swap → 同GPU做ref → ★ 单GPU RL设计范式
     → MoE GRPO → EP + router replay → ★ ★ MoE RL前沿
```

## 附录A: GRPO Issues & PRs 完整清单

```
★ ★ ★ Megatron-LM GRPO相关PRs/Issues (2026-06):

★ Merged:
  → #5280 — fix(ci): GRPO cudagraph-memory regression → linear sizing → ★ ★ ★ 关键!
  → #3509 — cudagraph distribution: linear→exponential → ★ ★ ★ 引入regression的PR
  → #5260 — MIMO hetero topology + ProcessGroupCollection → ★ PG新架构
  → #5242 — fix(ci): pin linear sizing for GRPO throughput test → ★ 5280的前置
  → #2403/#2411 — Add grpo loop functional test → ★ 最早的GRPO CI
  → #2472 — Make grpo CI test use read-only data → ★ CI稳定
  → #2587 — Move model configs to github → ★ 配置外部化
  → #2952 — ci: Restore grpo tests → ★ CI恢复
  → #3065 — Harden GRPO functional tests → ★ 测试加固
  → #3323 — Add simple GRPO functional test → ★ ★ 第一个GRPO测试
  → #3348 — re-enable gpt grpo tests → ★ CI恢复
  → #3517 — Implement forced lag in RL → ★ ★ staleness control
  → #3515 — Track per-token off-policy in RL → ★ ★ IS measurement
  → #3580 — Reverse polarity of off-policy measurement → ★ IS修正
  → #3740 — Fix dynamic inference and GRPO functional tests → ★ CI修复
  → #1409 — GRPO rlhf support → ★ ★ ★ 最早的GRPO支持PR!

★ Open:
  → #5306 — RL rollout submission/consumption granularity controls → ★ ★ 新feature
  → #2949 — RL training cudagraphs test → ★ 测试
  → #4168 — Router Replay (R3) for stable RL with MoE → ★ ★ ★ MoE RL关键!
  → #4549 — MTP SFT during RL train → ★ MTP + RL
  → #4256 — Rollout Routing Replay training-side support → ★ ★ R3
  → #4590 — Reduce downstream Megatron patching for RL → ★ ★ 简化RL使用
  → #4125 — Prevent unnecessary inference mode entry → ★ 优化
  → #1691 — backward_step/get_grad_norm time increase → ★ bug

★ ★ ★ DeepSeek-V4-Flash recipe:
  → 不是独立GRPO config → 而是DeepSeek-V4模型配置 + GRPO参数
  → v0.17.1 release notes提到 → 但代码中搜索无独立config
  → ★ ★ DeepSeek-V4需要SM90 → RTX 4090 ✗ → FP8/grouped GEMM/TMA全不可用
  → → ★ ★ ★ 结论: DeepSeek-V4-Flash recipe → H100/H800专属 → RTX 4090无法使用
```

## 附录B: Key File Paths

```
★ ★ ★ Megatron-LM GRPO关键文件:

训练入口:
  → train_rl.py → 425行 → 极简入口 → forward_step + MinimalDataset
  → megatron/rl/rl_utils.py → 2137行 → GRPO核心逻辑 → loss/advantage/inference_mode/data_prep

推理引擎:
  → megatron/rl/inference/megatron.py → MegatronLocal → DynamicInferenceEngine + OpenAI API
  → megatron/core/inference/engines/dynamic_engine.py → DynamicInferenceEngine → 状态机
  → megatron/core/inference/config.py → InferenceConfig + CudaGraphSizingDistribution

PG管理:
  → megatron/core/process_groups_config.py → ProcessGroupCollection → 18+ PG fields
  → megatron/rl/parallel_utils.py → build_inference_pg_collection → inference PG builder
  → megatron/core/hyper_comm_grid.py → HyperCommGrid → per-module topology

CUDA graph:
  → megatron/core/inference/batch_dimensions_utils.py → CUDAGraphBatchDimensionBuilder → sizing策略
  → megatron/core/transformer/cuda_graphs.py → CudaGraphManager → CG capture/replay

模型配置:
  → examples/rl/model_configs/common.sh → 共享配置 → CUDA graph + BF16 + inference
  → examples/rl/model_configs/qwen3_4b.sh → ★ 4B dense → TP=1 → 最接近RTX 4090
  → examples/rl/model_configs/qwen3_8b.sh → 8B dense → TP=1 → 需LoRA
  → examples/rl/model_configs/nemotron6_3b_moe.sh → ★ MoE GRPO → EP=32
  → examples/rl/model_configs/qwen3_30b_a3b_moe.sh → 大MoE → TP=4

环境配置:
  → examples/rl/environment_configs/dapo.yaml → DAPO math environment
  → examples/rl/environment_configs/gsm8k.yaml → GSM8K math
  → examples/rl/environment_configs/math.yaml → Math environment

CI配置:
  → tests/test_utils/recipes/h100/gpt-grpo.yaml → GRPO CI recipe → ★ linear sizing pinned!
  → tests/test_utils/recipes/h100/moe-grpo.yaml → MoE GRPO CI recipe

参数:
  → megatron/training/arguments.py → GRPO参数 → grpo_*/rl_* → ★ 40+RL参数
```
