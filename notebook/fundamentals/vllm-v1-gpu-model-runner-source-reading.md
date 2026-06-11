# vLLM V1 GPU Model Runner Source Reading — Two-Phase Execution + Persistent Batch + CUDA Graph + Spec Decode + Microbatching

> 2026-06-12 | vLLM V1 GPU Model Runner全链路源码分析: execute_model→sample_tokens两阶段执行 + _update_states持久化batch + CUDA Graph决策 + Spec Decode draft proposal + KVConnector + DBO microbatching
> 源码: vllm/v1/worker/gpu_model_runner.py (7516行)
> 关联: vllm-v1-kv-cache-architecture-source-reading.md, vllm-v1-spec-decode-architecture-source-reading.md, vllm-v1-scheduler-architecture-source-reading.md

## 0. 核心定律: Two-Phase Execution + Persistent Batch + CUDA Graph分档

```
GPUModelRunner = LoRAModelRunnerMixin + KVConnectorModelRunnerMixin + ECConnectorModelRunnerMixin

执行模式: 两阶段!
  Phase 1: execute_model(scheduler_output) → forward → 保存execute_model_state → 返回None/IntermediateTensors
  Phase 2: sample_tokens(grammar_output) → sample + draft proposal → 返回ModelRunnerOutput

为什么要两阶段?
  → 1. Grammar bitmask在forward和sample之间应用 → structured output需要!
  → 2. Async scheduling → forward和sampling时间线不同 → 需要分离!
  → 3. Spec decode draft → 在sample后运行draft model → 需要hidden_states!

GPUModelRunner核心组件:
  → model: nn.Module → 实际LLM模型
  → input_batch: InputBatch → 持久化batch数据(避免每次重建!)
  → attn_groups + kv_cache_config → attention backend管理
  → speculator: BaseSpeculator → draft模型(EAGLE/独立draft)
  → sampler: Sampler → token采样器
  → cudagraph_dispatcher → CUDA graph缓存和调度
  → execute_model_state → 两阶段间状态传递

关键优化:
  → Persistent Batch → input_batch跨step保持 → 只增删变化部分 → 避免全重建!
  → CUDA Graph → cudagraph_mode(FULL/PIECEWISE/NONE) → 分档执行 → decode用FULL → prefill用PIECEWISE
  → DBO Microbatching → use_ubatching → 大prefill拆成小ubatch → 不阻塞decode!
```

## 1. execute_model() — Phase 1: Forward Pass

```
execute_model(scheduler_output) 流程(约400行):

Step 1: 状态检查 + 特殊处理
  → ngram_gpu → replace(scheduler_output) → 避免修改scheduler的原始数据
  → has_kv_transfer_group → handle_preemptions → 处理KV connector抢占
  → _update_states(scheduler_output) → 更新持久化batch → deferred_state_corrections_fn

Step 2: 输入准备
  → _prepare_inputs(scheduler_output, num_scheduled_tokens_np)
  → → logits_indices → 哪些token需要计算logits
  → → spec_decode_metadata → 投机解码元数据

Step 3: Cascade Attention
  → _compute_cascade_attn_prefix_lens → 计算公共前缀长度
  → → num_common_prefix_blocks来自scheduler → 共享KV cache前缀!

Step 4: Batch Execution决策
  → _determine_batch_execution_and_padding → 决定CUDA graph模式+padding
  → → cudagraph_mode: FULL(pure decode) / PIECEWISE(mixed) / NONE(prefill/first)
  → → batch_desc: BatchDescriptor(num_tokens, num_reqs) → padding维度
  → → should_ubatch → 是否需要microbatching(DBO)

Step 5: Slot Mapping
  → _get_slot_mappings → 为每个KV cache组计算slot_mapping
  → → pad_attn=FULL → 需要padded维度 → slot_mapping匹配KV tensor大小
  → → has_separate_kv_update → 有些backend(FlashInfer)在forward外单独更新KV

Step 6: Attention Metadata
  → _build_attention_metadata → 为每个attention group构建attn_metadata
  → → 使用FlashInfer/SDPA/Triton backend → 根据配置选择
  → → spec_decode_metadata → draft model的attention元数据

Step 7: Model Forward
  → _preprocess → input_ids + positions + inputs_embeds + intermediate_tensors
  → set_forward_context → 注册全局forward context(attn_metadata+slot_mapping)
  → _model_forward → self.model(input_ids, positions, ...) → 真正的forward!
  → → EAGLE3 → hidden_states, aux_hidden_states → 双输出!

Step 8: 保存状态
  → execute_model_state = ExecuteModelState(...)
  → → scheduler_output + logits + hidden_states + spec_decode_metadata + ...
  → → 两阶段间传递 → sample_tokens()会解包这些
```

## 2. sample_tokens() — Phase 2: Sampling + Draft

```
sample_tokens(grammar_output) 流程(约200行):

Step 1: 解包execute_model_state
  → scheduler_output + logits + hidden_states + spec_decode_metadata + ...

Step 2: Grammar Bitmask (如果有)
  → apply_grammar_bitmask → xgrammar bitmask → logits[invalid_token] = -inf
  → → 结构化输出 → FSM状态驱动 → 确保只输出合法token

Step 3: Sampling
  → _sample(logits, spec_decode_metadata) → Sampler → 生成token ids
  → → greedy/temperature/top-k/top-p/min-p → 全在GPU上执行!

Step 4: Update States
  → _update_states_after_model_execute → 更新input_batch
  → → num_accepted_tokens_cpu += 1 → 标记接受了新token
  → → async scheduling → PP broadcast sampled tokens

Step 5: Spec Decode Draft Proposal
  → propose_draft_token_ids(sampled_token_ids) → 如果spec_config存在
  → → _input_fits_in_drafter → 检查batch大小是否在drafter范围内
  → → → 是: propose_draft_token_ids → 返回draft token ids
  → → → 否: 不运行draft → 零填充 → 返回空draft
  → → _copy_draft_token_ids_to_cpu → 复制draft结果到CPU → scheduler需要!

Step 6: 构造ModelRunnerOutput
  → sampled_token_ids → 采样的token ids
  → draft_token_ids → 投机draft token ids
  → draft_probs → draft概率(如果probabilistic sampling)
  → logits → 原始logits(如果需要logprobs)
  → kv_connector_output → KV connector传输结果
```

## 3. _update_states() — Persistent Batch管理

```
_update_states(scheduler_output) 流程(~370行):

1. 清理finished请求:
  → self.requests.pop(req_id) → 删除request数据
  → self.input_batch.remove_request(req_id) → 从持久化batch移除

2. GPU Block清零:
  → _zero_block_ids(new_block_ids_to_zero) → 新分配的KV blocks → GPU内存清零
  → → 防止stale NaN/data → 腐败attention或SSM计算!

3. 清理unscheduled请求:
  → unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
  → → 被抢占或未在本次step调度的running请求 → 从batch移除但保留cached states

4. 添加new请求:
  → 从scheduler_output.scheduled_new_reqs → 添加到input_batch
  → → request_id, prompt_token_ids, block_ids, sampling_params, lora_request, ...

5. 添加resumed请求:
  → 被抢占后恢复的请求 → 添加到input_batch → resumed_req_ids标记

6. 更新running请求:
  → scheduled_cached_reqs → 已在batch的请求 → 只更新差量!
  → → new_block_ids → 追加或替换block ids
  → → num_computed_tokens → 更新进度
  → → new_token_ids → PP用的新token ids

7. 更新SamplingMetadata:
  → sampling_metadata → temperature, top_p, top_k, etc → CPU→GPU复制
  → → 仅在有new/resumed/paused/finished请求时更新 → 避免不必要的GPU复制!

Persistent Batch设计:
  → 核心: 大多数请求跨step不变 → 只更新变化部分 → 避免全重建!
  → → 类似数据库的"增量更新" → 比SGLang的"全重建"更高效!
  → → 但! 如果request overlap低(交替两组request) → 反而低效!
```

## 4. _determine_batch_execution_and_padding — CUDA Graph决策

```
CUDA Graph三种模式:
  FULL → 所有token的forward captured → decode-only → 最快!
  PIECEWISE → 部分captured → mixed prefill+decode → 次快
  NONE → 不用CUDA graph → prefill或first step → 最慢但最灵活

决策逻辑:
  → uniform decode (所有request相同scheduled_tokens) → FULL
  → → 但! 有spec decode → PIECEWISE (draft model不captured)
  → → → 有new/resumed requests → PIECEWISE (新请求需要不同处理)
  → → → → 有encoder inputs → NONE (编码器需要灵活forward)
  → → → → → has_separate_kv_update → PIECEWISE (KV更新需灵活)

Padding决策:
  → batch_desc → BatchDescriptor(num_tokens_padded, num_reqs_padded)
  → → CUDA graph需要固定维度 → padding到最近的captured尺寸
  → → → 例如: 实际55 tokens → padding到64 tokens → 多9个padding tokens
  → → → → → padding tokens → lm_head logits_indices跳过 → 不浪费compute

DBO Microbatching:
  → use_ubatching → Dynamic Batching Optimization → 大prefill拆成小ubatch
  → → num_ubatches → 拆分数量 → 每个ubatch独立forward
  → → → 目的: 大prefill不阻塞decode → 交错执行 → ITL更稳定!
  → → → → → 但! microbatching增加总forward次数 → overhead!

RTX 4090 CUDA Graph决策:
  → decode B=32: FULL → 最快 → capture一次 → 重复使用 → 零Python overhead
  → → 但! INT4 AWQ → 可能不支持CUDA graph → 需fused kernel
  → → → decode B=1: PIECEWISE → 小batch → graph overhead > compute → 可能NONE更好
  → → → → → 生产最优: B≥32 → FULL CUDA graph → ~10%加速
```

## 5. RTX 4090 Model Runner Implications

```
1. Persistent Batch是RTX 4090关键优化:
  → 避免每次重建batch → GPU→CPU→GPU同步减少 → ~2x加速!
  → → 但需要input_batch维护 → 增删请求 → 状态同步

2. CUDA Graph加速:
  → decode FULL → 预先capture → 每step只需replay → ~10-15%加速
  → → 但! FlashInfer有自己cudagraph → 可能冲突
  → → → PIECEWISE → 只有部分captured → 次优

3. 两阶段执行好处:
  → grammar bitmask → 结构化输出 → FSM → <1% overhead → 几乎free
  → → spec decode → draft proposal → EAGLE hidden_states → 单GPU最优
  → → → async scheduling → PP用 → RTX 4090不适合(单GPU!)

4. DBO Microbatching:
  → RTX 4090 24GB → 大prefill(B=32 S=2048)占满budget → 阻塞decode
  → → → microbatching → 大prefill拆成4个小 → decode穿插 → ITL稳定!
  → → → → → 但! microbatch overhead → 每个ubatch一次forward → 总次数↑

5. Block清零:
  → 新KV blocks → GPU清零 → 防NaN → RTX 4090必须!
  → → → 否则stale data → attention计算错误 → 输出乱码!

6. 生产最优组合:
  → decode B=32 → FULL CUDA graph → Persistent Batch
  → → chunked prefill → PIECEWISE → 2048 chunk
  → → → EAGLE d=5 → draft proposal → INT4+INT8KV+FlashInfer
  → → → → → grammar bitmask → structured output → <1% overhead
  → → → → → → DBO microbatching → 2-4 ubatches → ITL稳定
```

## 参考文献

```
1. vLLM V1 GPU Model Runner源码:
   - vllm/v1/worker/gpu_model_runner.py (7516行) — GPUModelRunner
   - vllm/v1/worker/gpu/input_batch.py — InputBatch持久化batch

2. CUDA Graph: NVIDIA CUDA Graph Documentation
3. Persistent Batch: Orca/continuous batching设计
4. DBO: Dynamic Batching Optimization → micro-batching for LLM serving

我们的笔记:
- vllm-v1-kv-cache-architecture-source-reading.md — KV Cache
- vllm-v1-spec-decode-architecture-source-reading.md — Spec Decode
- vllm-v1-scheduler-architecture-source-reading.md — Scheduler