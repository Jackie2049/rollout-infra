# vLLM V1 GPUModelRunner 源码级深度阅读

> 2026-06-15 | 源码: vllm/v1/worker/gpu_model_runner.py(~7400行) + gpu_input_batch.py + block_table.py + sampler.py
> 核心: GPUModelRunner→3 Mixin(LoRA+KVConnector+ECConnector)→execute_model(update→prepare→forward)→sample_tokens(sample→draft→bookkeep→output)→InputBatch双缓冲→Triton slot_mapping→Sampler 9步→CUDA graph capture+replay

## 1. GPUModelRunner 类架构

```
GPUModelRunner (417) extends 3 Mixin:
  → LoRAModelRunnerMixin → LoRA管理
  → KVConnectorModelRunnerMixin → PD分离/KV Transfer
  → ECConnectorModelRunnerMixin → 多模态编码器

★ 初始化(__init__, 420-893):
  1. 配置存储(425-436): vllm_config所有子配置(model/cache/offload/compilation/lora/...)
  2. Sampler初始化(506): Sampler(logprobs_mode) → 独立创建
  3. Spec Decode Drafter初始化(541-617):
     → method → 选proposer → 创建drafter → RejectionSampler
  4. InputBatch初始化(655-683): 所有per-request CPU/GPU buffers
  5. Persistent CUDA Graph Buffers(714-760): 预分配固定GPU tensors
  6. CudagraphDispatcher(813): 根据compilation config管理CUDA graph keys
  7. ExecuteModelState(891-893): 两阶段执行状态传递
```

### Persistent CUDA Graph Buffers (714-760)

```
★ ★ 预分配固定GPU tensors → 地址不变 → CUDA graph replay直接写入!

self.input_ids: max_num_tokens, int32
self.positions: max_num_tokens, int64
self.query_start_loc: max_num_reqs+1, int32 → cumsum of scheduled tokens
self.seq_lens: max_num_reqs, int32 → sequence lengths
self.num_computed_tokens: max_num_reqs, int32 → computed tokens per request
self.req_indices: max_num_tokens, int64 → request index per token
self.prev_positions: max_num_reqs, int64 → 当前→上一轮batch位置映射
self.num_scheduled_tokens: max_num_reqs, int32
self.discard_request_mask: max_num_reqs, bool
self.num_accepted_tokens: max_num_reqs, int32 → spec decode accepted count

★ 关键设计: 所有buffer预分配 → 固定地址 → CUDA graph replay无需重新分配!
  → 动态写入: 每次replay前CPU计算 → copy_to_gpu() → 到固定地址
  → CUDA graph replay → 直接读取这些buffer的GPU地址 → 极快!
```

### ExecuteModelState (401-414) — 两阶段核心

```
ExecuteModelState(NamedTuple):
  scheduler_output, logits, spec_decode_metadata
  spec_decode_common_attn_metadata
  hidden_states, sample_hidden_states, aux_hidden_states
  ec_connector_output, cudagraph_stats, slot_mappings

★ ★ 两阶段执行机制:
  → execute_model() → 前向传播 → 打包ExecuteModelState → 返回None
  → sample_tokens() → 解包 → 采样/输出 → 返回ModelRunnerOutput
  → 分离forward和sample → 允许forward在GPU继续 → sample在CPU准备下一轮
  → 这是vLLM V1核心设计 → forward→sample分离 → pipeline overlap可能!
```

## 2. execute_model → 完整前向管线 (4002-4363)

```
★ ★ 3阶段管线:

Phase 1: Preprocess (4039-4238, 在synchronize_input_prep() context manager内)
  1. _update_states(scheduler_output) (4044):
     → 移除finished/unscheduled requests
     → 添加new/resumed requests到CachedRequestState和InputBatch
     → 更新num_computed_tokens, block_ids, output_token_ids
     → 处理spec decode corrections(async scheduling)
     → 返回deferred_state_corrections_fn → ★ 延迟到forward后执行!

  2. _prepare_inputs(scheduler_output) (4086):
     → 计算logits_indices和spec_decode_metadata

  3. _compute_cascade_attn_prefix_lens() (4095): 可选cascade attention

  4. ★ ★ _determine_batch_execution_and_padding() (4107):
     → 确定CUDA graph模式(FULL/PIECEWISE/NONE)
     → Uniform decode check: max_scheduled==uniform_query AND num_tokens==max*num_reqs
     → Padding: num_tokens padding到cudagraph capture size

  5. _get_slot_mappings() (4202): 每KV cache group的slot mapping

  6. _build_attention_metadata() (4213): per KV cache group/attn group

  7. _preprocess() (4236): input_ids, positions, inputs_embeds, model_kwargs

Phase 2: Forward (4260-4284)
  → 在set_forward_context()内:
  → model_output = self._model_forward(input_ids, positions, ...)
  → _model_forward() (3715-3745): 简单self.model() → 可子类override

Phase 3: Post-forward (4286-4363)
  → 提取hidden_states (EAGLE3有aux_hidden_states)
  → logits = model.compute_logits(sample_hidden_states) (4313)
  → 打包ExecuteModelState (4344-4355)
  → 执行deferred spec decode corrections (4360-4361)
```

## 3. _prepare_inputs — 核心输入准备 (1868-2187)

```
★ ★ 从scheduler output到GPU input tensors的完整转换!

Position IDs (1891-1904):
  → req_indices = np.repeat(arange[:num_reqs], num_scheduled_tokens)
  → positions = num_computed_tokens[req_indices] + query_offset

Input IDs (1920-1933):
  → token_indices = positions + req_indices * max_model_len
  → torch.index_select(token_ids_cpu_tensor.flatten(), 0, token_indices_tensor)
  → ★ 2D→flatten→index_select → 从CPU tensor按position抽取token id!

★ ★ Slot Mapping (2098-2102):
  → block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)
  → Triton kernel: _compute_slot_mapping_kernel
  → slot = block_table[req_idx, position/block_size] * block_size + position % block_size

logits_indices:
  → 无spec decode: query_start_loc[1:] - 1 → 每request最后一个scheduled token
  → 有spec decode: _calc_spec_decode_metadata → bonus+target verification positions
  → ★ 只计算需要位置的logits → 不计算全部 → 省compute!
```

## 4. InputBatch — CPU/GPU双缓冲

```
gpu_input_batch.py:

CachedRequestState (34-89): per-request状态缓存
  → req_id, prompt_token_ids, mm_features
  → sampling_params, generator, block_ids(per KV group)
  → num_computed_tokens, output_token_ids
  → lora_request, prev_num_draft_len(async spec)

InputBatch (91-): 核心CPU/GPU双缓冲

★ Token IDs:
  → token_ids_cpu_tensor: (max_num_reqs, max_model_len) → 存所有prompt+output tokens
  → num_tokens_no_spec: per-request总token数(不含spec)
  → num_computed_tokens_cpu: per-request已计算token数

★ Block Table:
  → MultiGroupBlockTable → per-KV-cache-group block映射
  → 每group有BlockTable(block_table+slot_mapping CpuGpuBuffer)

★ Sampling Parameters (CPU/GPU双缓冲):
  → temperature/top_p/top_k: CPU numpy + GPU tensor
  → frequency/presence/repetition_penalties → 双缓冲
  → greedy_reqs/random_reqs: set分类 → sampling方式选择
  → generators: dict[int, torch.Generator] → per-request随机种子

★ ★ add_request (335-481):
  → 分配req_index(优先填充空位)
  → 写入prompt_token_ids+output_token_ids到token_ids_cpu
  → 设置sampling params → 添加block_ids → 设置LoRA mapping
```

## 5. _build_attention_metadata (2188-2491)

```
★ CommonAttentionMetadata (2285-2301):
  → query_start_loc, seq_lens, num_reqs, num_actual_tokens
  → max_query_len, max_seq_len
  → block_table_tensor, slot_mapping
  → causal, is_prefilling, positions

★ Per-group Metadata (2333-2451):
  → 每KV cache group有独立CommonAttentionMetadata(shallow copy)
  → 差异: block_table_tensor, slot_mapping, encoder_seq_lens → per-group!

★ 缓存机制 (2329-2391):
  → 相同KVCacheSpec+AttentionMetadataBuilder组合 → cached_attn_metadata
  → supports_update_block_table → 只更新block_table不全重建 → 极快!
  → 减少重复metadata构建开销 → 大batch decode极重要!
```

## 6. Sampler — 9步采样管线

```
sampler.py: Sampler.forward (67-144):

★ ★ 9步采样管线:

1. Logprobs保存 → 计算或clone raw logits(before any modification)
2. Float32转换 → logits.to(torch.float32) → 精度保证
3. ★ apply_logits_processors (93-95):
   → allowed_token_ids whitelist → masked_fill_(-inf)
   → bad_words exclusion → 排除
   → non_argmax_invariant processors → min_tokens/logit_bias
   → penalties → frequency/presence/repetition
   → thinking_budget → state tracking + apply
4. ★ sample (97):
   → all_greedy: argmax → 最快!
   → all_random: temperature + argmax_invariant + topk_topp
   → mixed: torch.where(temperature<eps, greedy, random) → 双路径!
5. Int64→Int32转换 → GPU输出格式
6. SamplerOutput → sampled_token_ids shape [num_requests, 1]

★ SamplingMetadata (metadata.py:14-56):
  → temperature, all_greedy, all_random, top_p, top_k
  → generators, max_num_logprobs, no_penalties
  → prompt_token_ids, penalties, allowed_token_ids_mask
  → logitsprocs, spec_token_ids, thinking_budget_state_holder
```

## 7. sample_tokens — 采样+Draft+Bookkeep (4381-4640)

```
★ ★ 4阶段:

Phase 1: Sample (4416-4417):
  → sampler_output = _sample(logits, spec_decode_metadata)

Phase 2: Speculative Decoding (4439-4531):
  → GPU proposers(EAGLE/Draft/ngram_gpu) → 直接用GPU sampled tokens
  → CPU proposers(ngram/suffix) → 等bookkeeping后用CPU tokens
  → _input_fits_in_drafter() → 检查是否超过drafter max_model_len

Phase 3: ★ Bookkeeping (4532-4547):
  → _bookkeeping_sync():
    → 计算NaN统计
    → 更新output_token_ids → 写入InputBatch CPU tensor + CachedRequestState
    → 计算prompt logprobs(如请求)

Phase 4: Output (4567-4640):
  → ModelRunnerOutput 或 AsyncGPUModelRunnerOutput
  → Async path: 单独stream上异步D2H copy → sampled_token_ids + logprobs
  → ★ Async → 不阻塞主路径 → GPU继续下一forward → 极快!
```

## 8. CUDA Graph Capture + Replay

```
★ ★ CudagraphDispatcher (cudagraph_dispatcher.py:15-):
  → cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]]
  → FULL和PIECEWISE两组keys → 不同模式
  → dispatch(num_tokens, has_lora, uniform_decode, num_active_loras)
  → → runtime选cudagraph mode和BatchDescriptor

★ _determine_batch_execution_and_padding (3768-3880):
  → Uniform decode: max_scheduled==uniform_query AND num_tokens==max*num_reqs
  → Cudagraph dispatch: batch_size+LoRA+uniform_decode → FULL/PIECEWISE/NONE
  → Padding: num_tokens → cudagraph capture size → 支持SP

★ ★ _warmup_and_capture (6584-6617):
  1. Warmup: num_warmups次 _dummy_run(CUDAGraphMode.NONE)
     → PyTorch/cuDNN完成autotune和内存分配 → 稳定后capture!
  2. Capture: _dummy_run(cudagraph_runtime_mode, is_graph_capturing=True)
     → 实际CUDA graph capture → 固定所有操作

★ _dummy_run (5595-5940):
  → 创建虚拟batch → model forward
  → slot_mappings填-1(跳过KV write) → 不污染真实cache!
  → Drafter也有独立dummy_run → EAGLE/DraftModel需要额外capture

★ ★ CUDA Graph中固定vs动态:

固定(persistent buffers → 地址不变 → replay直接写入):
  → input_ids, positions, seq_lens, query_start_loc
  → num_computed_tokens, block_table, slot_mapping → 预分配!

动态(每次replay前CPU计算 → copy_to_gpu() → 到固定地址):
  → positions, input_ids → CPU准备 → GPU copy → replay读取

Padding(for FULL cudagraph):
  → Padding tokens: positions=0, input_ids=0
  → Padding requests: seq_lens=0, block_table=NULL_BLOCK_ID
  → ★ 这些padding值让CUDA graph中固定shape的kernel正确执行 → 不崩溃!
```

## 9. 完整执行流程详解

```
★ ★ Decode Step(每request 1 token, 无spec decode):

1. Scheduler → SchedulerOutput(每running request 1 scheduled token)
2. execute_model():
   → _update_states: 更新num_computed_tokens, 添加新block_ids
   → _prepare_inputs:
     → positions = num_computed_tokens → 已经完成的token数
     → input_ids = token_ids[num_computed_tokens] → 从CPU flatten抽取
     → query_start_loc = cumsum of [1,1,...] = [0,1,2,...]
     → logits_indices = query_start_loc[1:] - 1 → 每request最后token
   → _build_attention_metadata → FlashInfer decode metadata
   → _model_forward → model(input_ids, positions) → hidden_states
   → logits = compute_logits(hidden_states[logits_indices]) → [num_reqs, vocab]
   → 打包ExecuteModelState → 返回None
3. sample_tokens():
   → _sample(logits, None) → sampler → 1 token per request
   → _bookkeeping_sync → 写入InputBatch + CachedRequestState
   → 返回ModelRunnerOutput(sampled_token_ids, logprobs)

★ ★ Prefill Step(新请求, 多prompt tokens):

1. Scheduler → SchedulerOutput(新请求, num_scheduled=prompt_len or chunk)
2. _update_states → 创建CachedRequestState → 添加到InputBatch
3. _prepare_inputs:
   → positions = [0,1,...,prompt_len-1] for each new request
   → input_ids = prompt_token_ids
   → logits_indices = 每请求最后scheduled token
   → ★ discard_request_mask = True for chunked prefill → 中间chunk不采样!
4. Forward → model处理所有prompt tokens
5. Sampling → 只logits_indices位置计算logits → chunked prefill丢弃中间结果
6. KV cache → 新tokens写入(slot_mapping + attention kernel)

★ ★ Spec Decode Step(decode + draft tokens):

1. Scheduler → SchedulerOutput(含scheduled_spec_decode_tokens)
2. _prepare_inputs → spec_decode_metadata(bonus_logits_indices, target_logits_indices)
3. Forward → model处理1+draft tokens per request
4. _sample(logits, spec_decode_metadata):
   → RejectionSampler → bonus + rejection verification
   → 输出: [num_reqs, max_spec_len+1] → accepted+bonus → rejected=PLACEHOLDER
5. propose_draft_token_ids → drafter生成下一轮draft tokens
6. Bookkeeping → parse_output过滤accepted tokens
```

## 10. 关键设计洞察

```
1. 两阶段执行(execute_model→sample_tokens) → 分离forward和sample!
   → forward在GPU → sample准备下一轮 → CPU-GPU pipeline overlap可能!
   → ExecuteModelState打包 → forward→None → sample→Output → 两步分离
   → 这是vLLM V1核心设计 → vs V0 → forward和sample一体 → 无法overlap!

2. Persistent GPU buffers → CUDA graph核心 → 地址固定 → replay安全!
   → 所有关键tensor预分配 → 固定地址 → CUDA graph replay直接写入
   → 动态值: CPU计算 → copy_to_gpu() → 到固定地址 → 极快!
   → Padding: 填0和NULL → 固定shape kernel → 不崩溃 → 设计精妙!

3. Triton slot_mapping kernel → GPU kernel计算 → 不回CPU → 极快!
   → compute_slot_mapping → Triton kernel → block_id*block_size+offset
   → slot_mapping = physical position in KV cache → attention kernel用
   → ★ 全GPU → CPU只传参数 → Triton kernel执行 → 极快!

4. 9步Sampler → modular → 每步独立 → 可选择性执行!
   → greedy: skip most steps → argmax → 最快
   → random: all 9 steps → temperature+topk+topp → 精确但慢
   → mixed: torch.where → 每request选路径 → 同时处理!
   → ★ Modular设计 → 不同request不同路径 → 同batch混合!

5. Deferred corrections → async spec decode → 不阻塞主路径!
   → spec decode rejection → corrections deferred → forward后执行
   → 主路径不等corrections → GPU继续下一forward → 极快!
   → vs 传统: 等rejection完成 → 阻塞 → 慢 → vLLM V1改进!

6. InputBatch双缓冲 → CPU准备+GPU执行 → 无GPU stall!
   → CPU端: token_ids_cpu, sampling_params_cpu, block_table_cpu
   → GPU端: 同名tensor → copy_to_gpu() → 固定地址 → CUDA graph
   → CPU准备时GPU可执行上一轮 → pipeline overlap!
   → ★ 双缓冲 = 生产级推理引擎核心设计!

7. CUDAGraphDispatcher → FULL/PIECEWISE/NONE → runtime智能选择!
   → FULL: 固定shape → 完整CUDA graph → 最快
   → PIECEWISE: 模块级CUDA graph → 更灵活
   → NONE: eager → fallback → 最慢
   → dispatch根据batch_size+LoRA+uniform_decode → 自动选最优!

8. logits_indices → 只计算需要位置 → 不计算全部 → 省compute!
   → decode: 每request最后1 token → [num_reqs, vocab]
   → prefill: 每request最后scheduled token → 不算中间
   → ★ materialize_only_last_token_logits=True → InferenceConfig → decode优化!
```

---

Sources:
- vllm/v1/worker/gpu_model_runner.py (GPUModelRunner ~7400行)
- vllm/v1/worker/gpu_input_batch.py (InputBatch + CachedRequestState)
- vllm/v1/worker/block_table.py (BlockTable + Triton slot_mapping)
- vllm/v1/sample/sampler.py (Sampler 9步管线)
- vllm/v1/sample/metadata.py (SamplingMetadata)
- vllm/v1/sample/rejection_sampler.py (RejectionSampler)
- vllm/v1/cudagraph_dispatcher.py (CudagraphDispatcher)
