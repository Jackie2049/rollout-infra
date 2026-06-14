# Megatron-LM 推理引擎 源码级深度阅读

> 2026-06-15 | 源码: Megatron-LM/megatron/core/inference/ + moe/token_dispatcher_inference.py + export/trtllm/
> 核心: DynamicInferenceEngine→CUDA graph多batch维度→InferenceTopKRouter(@torch.compile)→NCCL/NVLS dispatcher→KV block allocator→TRT-LLM export→RTX 4090: NCCL✓/NVLS✗(SM90)

## 1. DynamicInferenceEngine — 推理主引擎

```
megatron/core/inference/engines/dynamic_engine.py:

DynamicInferenceEngine (153-169): @experimental_api
  → TextGenerationController + DynamicInferenceContext
  → __init__ (184-267): controller, context, num_speculative_tokens
  → reset() (268-313): 清除requests/waiting_request_ids → EngineState.RUNNING

★ EngineState (113-125): 状态机
  → RUNNING → PAUSING → PAUSED → UNPAUSING
  → SUSPENDING → SUSPENDED → RESUMING → RESUMED
  → STOPPING → STOPPED

★ create_cuda_graphs() (328-455): 关键!
  → 遍历context.cuda_graph_batch_dimensions_list
  → 每个batch dim: warmup forward → sample logits → MTP warmup
  → Router replay recording enabled during warmup (line 391)
  → context.reset() → 清除KV cache state → 每次warmup后重置!

★ StaticInferenceEngine (deprecated → wrapper):
  → __init__内部创建DynamicInferenceContext + DynamicInferenceEngine
  → generate() → generate_using_dynamic_engine() → 动态引擎
  → legacy fallback → run_engine() → Scheduler → static batch
```

### InferenceConfig — 推理配置

```
megatron/core/inference/config.py (121-366):
  → block_size_tokens=256 → KV cache block大小
  → buffer_size_gb=20 → GPU内存预留给KV cache
  → max_requests, max_tokens → in-flight batching限制
  → num_cuda_graphs → 最大CUDA graph数量
  → cuda_graph_mixed_prefill_count=16 → mixed prefill graphs
  → enable_chunked_prefill → chunked prefill支持
  → num_speculative_tokens → MTP speculative decode
  → enable_prefix_caching + prefix_caching_eviction_policy
  → sampling_backend: 'torch' or 'flashinfer'
  → unified_memory_level: 0=GPU only, 1=UVM for KV cache
  → materialize_only_last_token_logits=True → 只计算最后token logits → decode优化!
```

## 2. 推理Token Dispatcher — NCCL vs NVLS

```
megatron/core/transformer/moe/token_dispatcher_inference.py:

★ NCCLAllGatherDispatcher (92-283): 两种模式!

CG path (_use_allgather_v=False, decode专用):
  → 所有EP rank贡献等量tokens → decode CUDA graph兼容!
  → 标准NCCL AllGather/ReduceScatter → gather_from_sp / reduce_scatter_to_sp
  → update_metadata() (139-164): _valid_tokens_tensor.fill_(ep_size * local_tokens)
    → ★ 单次in-place .fill_() → CUDA graph capture安全 → 无host sync!
  → token_dispatch() (166-196): 标准AllGather(tp_ep_group)
  → token_combine() (242-262): 标准ReduceScatter

Non-CG path (_use_allgather_v=True, prefill):
  → 各rank可能不同token数量 → pad到max → AllGather → compact去padding
  → update_metadata() (153-160): all-gather counts → .tolist() host sync!
  → ★ 此路径绝不在graph capture下运行!
  → token_combine() (263-283): expand→padded→ReduceScatter→truncate

★ ★ NVLSAllGatherVDispatcher (286-603): Hopper+专用!
  → Requires: Hopper+ GPU + NVLink + symmetric memory (line 293)
  → 所有metadata在GPU上 → Triton kernel → 无host sync → decode极快!

  Class-level symmetric buffers (296-309):
    → _step_metadata[3] int32 → [valid_tokens, rank_token_offset, ep_max_tokens]
    → _symm_agv_hidden/routing/probs → AllGather-V symmetric buffers
    → _symm_rsv → ReduceScatter-V symmetric buffer

  allocate_buffers() (336-419): 模型初始化时一次性分配!
    → SymmetricMemoryManager.get_buffer() → 固定大小 → 不重新分配!
    → 失败 → RuntimeError: "Use inference_moe_token_dispatcher_type='nccl' on non-NVLS systems"

  ★ ★ fused_metadata_update() Triton kernel (metadata.py:46-134):
    → 单个CTA Triton kernel → 替代5个独立kernel!
    → (1) multicast-store local_tokens → (2) barrier → (3) 读所有rank
    → (4) 计算 sum/prefix/max → (5) 写step_metadata[0:3]
    → ★ 一次kernel → 5个操作的融合 → 极快!

  token_dispatch() (496-542):
    → multimem_all_gatherv_3tensor → 同时gather hidden/routing/probs → 3tensor!
    → → 预分配symmetric buffer → 不动态分配!

  token_combine() (554-585):
    → multimem_reduce_scatter_v → variable-count ReduceScatter
    → → 如果expert output直接写入RSV buffer(mcore_fused_moe) → 跳过copy!

  ★ Shared expert overlap (458-468):
    → SharedExpertMLP.stream → 并发AGV+experts+RSV → overlap!
    → → NVLS独有 → NCCL路径不支持overlap!
```

## 3. InferenceTopKRouter — @torch.compile推理路由

```
megatron/core/transformer/moe/router.py (749-848):

InferenceTopKRouter: stripped-down TopKRouter
  → 跳过: z_loss / aux losses / token dropping / expert bias updates
  → 约束: moe_router_num_groups=None + score_function in [sigmoid/softmax]

★ ★ _compiled_topk_routing() (785-812): @staticmethod @torch.compile
  → topk_routing_with_score_function(dense_output=True)
  → dense_output=True → 返回 [num_tokens, topk] dense tensors!
  → → 不是sparse [num_tokens, num_experts] → 大部分零 → 不高效
  → → dense format → FlashInfer grouped GEMM直接按top_indices索引 → 极快!

_forward() (814-830):
  → logits = self.gating(input).squeeze(1) → gate前向
  → _compiled_topk_routing(dense_output=True)
  → 返回 probs.squeeze(1), top_indices.squeeze(1) → [num_tokens, topk]

forward() (832-848):
  → InferenceMode.is_active() → _forward() → 推理路径
  → 否则 → parent TopKRouter.forward() → 训练路径

★ 关键设计: 推理路由 = compile + dense output → 2大优化!
  → @torch.compile → 消除Python overhead → 纯kernel执行
  → dense_output → FlashInfer grouped GEMM兼容 → 无需sparse→dense转换
  → CUDA graph capture友好 → 固定shape → 可graph!
```

## 4. Text Generation Pipeline + KV Cache

```
TextGenerationController (text_generation_controller.py:72-105):
  → AbstractModelInferenceWrapper + tokenizer
  → _init_dynamic_sampling_tensors() (141-192):
    → _all_logits_cuda = [1, max_logits, vocab_size] → 预分配!
    → FlashInferSampling or TorchSampling backend
  → _dynamic_step_forward_logits() (614): forward → logits

★ DynamicInferenceContext KV Cache:

KV Block Format (dynamic_context.py:392-412):
  → Standard: dtype_size × 2(K+V) × num_layers × block_size_tokens × heads × head_dim
  → MLA latent: dtype_size × num_layers × block_size_tokens × kv_reduced_dim
  → Default block_size_tokens = 256

★ KVBlockAllocator (kv_block_allocator.py:13-80):
  → block_bag (stack) → 总block池 → 分配/释放O(1)
  → dummy_block_idx → 初始占位
  → ★ Prefix caching: block_hashes + kv_hash_to_block_id + block_ref_counts + LRU timestamps
  → 2种eviction: REF_ZERO(立即释放) / LRU(保留hash→evict最旧)

★ Generation Loop (Dynamic Engine):
  → 每step: (1) batch active requests (2) prepare input_ids/position_ids
  → (3) forward logits (4) sample tokens (5) update KV cache
  → (6) check completion → decode: 1 token per request
  → MTP: 1+num_speculative_tokens → speculative decode!
```

## 5. CUDA Graph集成 — 推理核心优化

```
★ InferenceCudaGraphScope (enums.py:101-112):
  → none → eager → 无graph
  → layer → per-module graph → 细粒度
  → block → per-TransformerBlock → 粗粒度

★ InferenceBatchDimensions (batch_dimensions_utils.py:20-74):
  → (token_count, prefill_req_count, decode_req_count)
  → is_applicable_for_batch_dim() → 检查graph batch dims是否适配

★ ★ adjust_batch_dims_for_expert_parallelism() (143-206):
  → NCCL dispatcher: all-reduce-max token count across EP
  → 如果任何rank有prefill → 所有rank fallback eager → return None
  → ★ ZMQ CPU-only sync → 避免per-step NCCL AllReduce → 极快!

★ ★ CUDAGraphBatchDimensionBuilder (208-551):
  → token_counts均匀分布 → rounded to CUDA_GRAPH_ROUNDER=8 → aligned to TP
  → decode-only: bounded by max_requests*(num_speculative_tokens+1)
  → mixed prefill+decode: 可扩展到max_tokens → cuda_graph_all_prefills=True
  → match_graph_config(): 找最小applicable graph batch dim → 最优匹配!

★ CUDA Graph Creation (dynamic_engine.py:328-455):
  → inference_cuda_graph_scope==none → skip
  → 遍历batch_dimensions_list → warmup forward → sample → MTP warmup
  → Router replay recording during warmup → capture路由决策
  → context.reset() → 清除每次warmup后的KV state

★ ★ 固定Shape要求 → CUDA graph核心约束:
  1. 预分配所有tensor最大size → _all_logits_cuda[max_logits, vocab_size]
  2. 捕获多个graph → 不同batch dimensions → 最优匹配
  3. Runtime → match real batch to smallest applicable captured graph
  4. EP decode: NCCL所有rank必须等量tokens → 固定shape AllGather
  5. NVLS: variable-count AGV/RSV → 不同token counts per rank → 更灵活!

★ MoE CUDA Graph Partial Capture (moe_layer.py:437-444):
  → @maybe_skip_or_early_return_by_cudagraph("route") → route()装饰器
  → fwd_execution_map = ["route", "expert_compute", "postprocess"]
  → MoECudaGraphPartialCaptureSignal → 只capture部分forward → 其余eager

★ Training CUDA Graphs (full_cuda_graph.py:138-234):
  → FullCudaGraphWrapper → 整个forward-backward iteration graph
  → StaticBufferLoader → copy dataloader output到static CUDA tensor
  → warmup → capture → replay → shared graph pool + capture stream → 避免碎片!
```

## 6. TRT-LLM Export — 训练→推理桥梁

```
★ TRTLLMHelper (trtllm_helper.py:39-108):
  → TransformerConfig + ModelType + conversion dict + position embedding + MoE params
  → _get_trtllm_config() (110-208):
    → 构建TRT-LLM PretrainedConfig (GPT/LLaMA/Mixtral/Gemma/Falcon)
    → MoE config: moe_num_experts + moe_top_k + moe_normalization_mode + moe_tp_mode
  → get_trtllm_pretrained_config_and_model_weights() (264-350):
    → Single-device conversion(CPU/GPU): per-rank weights
    → On-device distributed: each GPU shard → DistributedTRTLLMModelWeightsConverter

★ TRTLLMLayers (trtllm_layers.py:8-54):
  → Enum: Megatron→TRT-LLM layer name mapping
  → position_embedding/vocab_embedding/lm_head/final_layernorm
  → attention_qkv/dense → mlp_fc/projection → mlp_router → expert variants

★ TRTLLMEngineBuilder (engine_builder.py:19-172):
  → build_and_save_engine() → tensorrt_llm build
  → PluginConfig: paged_kv_cache + remove_input_padding + gpt_attention_plugin + gemm_plugin
  → BuildConfig → model class → load weights → build engine → save disk

★ TRT_MODEL_CONFIG (trt_model_config.py:16-25):
  → gpt→GPTConfig / llama→LLaMAConfig / mixtral→LLaMAConfig
  → gemma→GemmaConfig / falcon→FalconConfig / nemotron_nas→DeciConfig

★ Export流程: Megatron训练 → TRT-LLM export → TensorRT推理 → 生产级!
  → 这是Megatron推理的推荐路径 → 不是直接用DynamicInferenceEngine!
  → 但: DynamicInferenceEngine可用于开发/调试/小规模 → TRT-LLM用于生产!
```

## 7. RTX 4090 推理兼容性

```
★ ★ RTX 4090推理完整兼容性表:

| Feature | RTX 4090 Support |
|---------|------------------|
| DynamicInferenceEngine | ✓ YES |
| CUDA graphs (decode-only) | ✓ YES |
| CUDA graphs (mixed prefill) | ✓ YES |
| NCCL token dispatcher | ✓ YES |
| NVLS token dispatcher | ✗ NO (SM90 required) |
| InferenceTopKRouter | ✓ YES |
| FlashInfer grouped GEMM | ✓ YES (SM89) |
| MXFP8 quantization | ✓ YES |
| Prefix caching (REF_ZERO/LRU) | ✓ YES |
| Chunked prefill | ✓ YES |
| MTP speculative decode | ✓ YES |
| TRT-LLM export | ✓ YES (offline) |
| EP>1 CUDA graphs | ✓ YES (NCCL equal tokens) |
| EP>1 variable tokens | ✗ NO (NVLS AGV/RSV only) |
| TE FP8 inference | ✓ YES (SM89 E4M3/E5M2) |
| Shared expert overlap (NVLS) | ✗ NO |
| Shared expert overlap (NCCL) | ✓ YES (non-overlapped) |

★ RTX 4090不可用:
  1. NVLSAllGatherVDispatcher → SM90+NVLink+symmetric memory TMA multicast
  2. multimem_all_gatherv_3tensor / multimem_reduce_scatter_v → TMA SM90-only
  3. fused_multimem_rs_add_norm_ag → NVLS fused kernel SM90-only
  4. Shared expert overlap with NVLS → 需要NVLS dispatcher
  5. Variable token counts across EP ranks in CUDA-graphed decode → NVLS AGV/RSV唯一机制

★ RTX 4090最优推理配置:
  → DynamicInferenceEngine + NCCL dispatcher
  → CUDA graphs (decode-only) + InferenceTopKRouter
  → FlashInfer grouped GEMM + prefix caching (LRU)
  → EP=1 (单GPU) → 无分布式 → 最简单最稳定!
  → TRT-LLM export → 生产级推理 → 最大吞吐!
  → INT4量化 → vLLM/SGLang比Megatron更优 → 4,791 tok/s
```

## 8. 关键设计洞察

```
1. DynamicInferenceEngine vs StaticInferenceEngine → 动态是未来!
   → Static: 固定batch → Scheduler → 简单但低效
   → Dynamic: continuous batching → CUDA graph多维度 → 高效但复杂
   → Dynamic = @experimental_api → 还在迭代 → 但方向正确!

2. CUDA graph多batch维度 → 推理性能关键!
   → 预分配多个graph → 不同token_count/request_count → 最优匹配
   → match_graph_config() → 找最小applicable → 最小overhead
   → 每次warmup后reset → 清除KV → 防止污染 → 设计正确!
   → RTX 4090: CUDA graph完全支持 → decode性能极大提升!

3. InferenceTopKRouter @torch.compile + dense_output → 推理路由最优!
   → compile → 消除Python overhead → 纯kernel → 极快
   → dense_output → [num_tokens, topk] → FlashInfer grouped GEMM直接用
   → vs 训练TopKRouter: sparse [num_tokens, num_experts] → 大部分零 → 浪费!
   → 推理和训练用不同路由 → 推理有专用优化 → 这是关键设计!

4. NVLS vs NCCL → 推理dispatcher的两种范式!
   → NCCL: 等量tokens → 固定shape → CUDA graph友好 → 但prefill限制
   → NVLS: variable tokens → 更灵活 → fused Triton metadata → 但需SM90!
   → Hopper+集群: NVLS → 最优; RTX 4090: NCCL → 次优但可用
   → NVLS是GB200推理的关键 → multimem → 3tensor同时gather → 极快!

5. TRT-LLM Export → Megatron推理的推荐生产路径!
   → Megatron训练 → TRT-LLM export → TensorRT推理 → 最大吞吐!
   → 不是直接用DynamicInferenceEngine → 而是export后用TensorRT
   → PluginConfig: paged_kv_cache + remove_input_padding → 生产级优化
   → 这表明: Megatron定位是训练框架 → 推理靠TRT-LLM!

6. KV block_size_tokens=256 → 与vLLM block_size=16不同!
   → Megatron更大block → 256 tokens/block → 更粗粒度 → 更少管理overhead
   → vLLM更小block → 16 tokens/block → 更精细 → 更少浪费 → 93.75%最低利用率
   → Megatron MLA: kv_reduced_dim = kv_lora_rank + qk_pos_emb_head_dim → 特殊处理!

7. MTP speculative decode → Multi-Token Prediction → 新方向!
   → num_speculative_tokens → 不是draft model → 是target model预测多token!
   → vs EAGLE: draft model → 需要额外model → MTP不需要!
   → vs Medusa: multi-head → 需要额外head → MTP更轻量!
   → Megatron推理内置MTP → 不需要额外组件 → 更简单!
```

---

Sources:
- Megatron-LM/megatron/core/inference/engines/dynamic_engine.py (DynamicInferenceEngine)
- Megatron-LM/megatron/core/inference/config.py (InferenceConfig)
- Megatron-LM/megatron/core/transformer/moe/token_dispatcher_inference.py (NCCL/NVLS dispatcher)
- Megatron-LM/megatron/core/transformer/moe/router.py (InferenceTopKRouter)
- Megatron-LM/megatron/core/inference/text_generation_controllers/text_generation_controller.py
- Megatron-LM/megatron/core/inference/contexts/dynamic_context.py (KV cache format)
- Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py
- Megatron-LM/megatron/core/inference/batch_dimensions_utils.py
- Megatron-LM/megatron/core/export/trtllm/ (TRT-LLM export)
- Megatron-LM/megatron/core/transformer/cuda_graph_config.py
- Megatron-LM/megatron/core/full_cuda_graph.py
- Background agent research (Megatron inference engine)
