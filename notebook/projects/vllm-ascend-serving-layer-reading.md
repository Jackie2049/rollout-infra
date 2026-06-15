# vLLM-Ascend Serving Layer 源码级深度阅读

> 2026-06-15 | 源码: vllm-ascend/vllm_ascend/ (worker + attention + scheduler)
> 核心: NPUModelRunner(继承GPUModelRunner)→AscendAttentionBackend(6种专用)→torch_npu.npu_fused_infer_attention_score→ACL graph(torch.npu.graph_task_*)→NPU; BudgetRefiner SLO动态batch; MultiGroupBlockTable MLA/DSA; block_size=128; PCP+DCP上下文并行; CaMemAllocator sleep/wake via CANN; vs vLLM: 6 attention backends vs 2-3 + BudgetRefiner unique + recompute default + CompressAttention MLA

## 1. NPUModelRunner — GPUModelRunner的Ascend移植

```
★ ★ NPUModelRunner(model_runner_v1.py:255): 继承GPUModelRunner → 最小化修改!

关键差异:
  → NPUWorker(worker.py): 替换GPUWorker → NPUModelRunner
  → NPUInputBatch(npu_input_batch.py): 替换InputBatch → MultiGroupBlockTable
  → AscendConfig(ascend_config.py): Ascend专用配置(attention backend选择/PCP/DCP等)
  → CaMemAllocator(device_allocator/camem.py): sleep/wake via CANN memcpy → vLLM level 1/2映射

★ ★ execute_model (1904-2100+):
  → 和GPUModelRunner几乎相同 → 但PCP(pcp_size>1)需要deepcopy scheduler_output!
  → PCP+Multi-Modal → 特殊deepcopy → temporary workaround → 等vLLM upstream native support
  → _update_states → _prepare_inputs → _determine_batch_execution_and_padding → forward
  → mamba_cache_mode="align" → deferred_state_corrections先执行 → preprocess_mamba
  → cascade_attn_prefix_lens → 禁用DBO(enable_dbo=False → 无cascade)

★ KV cache初始化:
  → initialize_kv_cache (3700): Ascend专用 → 每层绑定hashk_cache(MLA)
  → initialize_kv_cache_tensors (3764): KVCacheConfig → MultiGroup分配
  → bind_hashk_cache(kvcomp_utils.py): hashk_caches绑定到forward_context和runner
```

## 2. AscendAttentionBackend — 6种专用注意力

```
★ ★ ★ AscendAttentionBackend(attention_v1.py:73): Registered as "ASCEND" (或"FLASH_ATTN" for V2 runner)

@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
  accept_output_buffer: bool = True
  → get_name(): "CUSTOM" or "FLASH_ATTN" (V2 runner hack → bypass assertion)
  → get_impl_cls(): AscendAttentionCPImpl(context parallel) / AscendAttentionBackendImpl(normal)
  → get_builder_cls(): AscendAttentionCPMetadataBuilder / AscendAttentionMetadataBuilder
  → get_kv_cache_shape: (2, num_blocks, block_size, num_kv_heads, head_size) → 2=separate K/V!
  → swap_blocks: src→dst indices copy → CPU/GPU KV transfer
  → copy_blocks: src→dst copy → KV cache duplication
  → ★ get_supported_kernel_block_sizes: [128] → ★ vs vLLM 16! → 8x bigger!

★ ★ 6种注意力实现:
  1. FA (FlashAttention): torch_npu.npu_fused_infer_attention_score → Ascend FA
  2. FA3 (FlashAttention v3): npu_fused_infer_attention_score_v2 → Ascend FA3
  3. SFA (Split FlashAttention): 长序列分段 → overlap
  4. DSA (Dynamic Sparse Attention): 动态稀疏 → KVComp + Hamming
  5. MLA (Multi-head Latent Attention): DeepSeek-V2专用 → CompressAttentionManager
  6. KVComp (KV Compression): Hamming sparse → long-context compression

★ ★ AscendAttentionState (143-148):
  PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding
  → vs vLLM: 更细分 → PrefillNoCache vs PrefillCacheHit → 区分cold vs warm prefill!
```

## 3. ACL Graph — Ascend的CUDA Graph等效

```
★ ★ ★ ACL Graph = Ascend Compute Language Graph → CUDA graph的Ascend等价!

关键API:
  torch.npu.graph_task_group_begin(stream) → 开始graph task组
  torch.npu.graph_task_group_end(stream) → 结束 → 返回handle
  torch.npu.graph_task_update_begin(update_stream, handle) → replay开始
  torch.npu.graph_task_update_end(update_stream) → replay结束

★ ★ FIA full_graph_fia (680-826):
  → torch.npu.graph_task_group_begin(stream) → 开始capture
  → torch_npu.npu_fused_infer_attention_score.out(...) → 记录attention op
  → handle = torch.npu.graph_task_group_end(stream) → 结束capture → 返回handle
  → graph_params.handles[num_tokens].append(handle) → 存储handle

★ ★ Replay (full_graph_fia_replay, 560-670):
  → torch.npu.graph_task_update_begin(update_stream, handle) → replay开始
  → 解包attn_params → query/key/value/block_table/attn_mask等
  → torch_npu.npu_fused_infer_attention_score.out(...) → replay中执行!
  → torch.npu.graph_task_update_end(update_stream) → replay结束

★ ★ vs vLLM CUDA graph:
  → vLLM: torch.cuda.CUDAGraph.capture() + replay() → PyTorch原生
  → Ascend: torch.npu.graph_task_* API → CANN原生 → 更低层
  → vLLM: persistent buffer固定地址 → replay写入
  → Ascend: graph_params workspaces + handles → ExternalEvent(event) → 同步机制
  → ★ 两者核心逻辑相同: capture→replay→固定shape→内存不变!
```

## 4. BudgetRefiner — SLO动态Batch

```
★ ★ BudgetRefiner (scheduler_dynamic_batch.py): Ascend独有 → vLLM没有!

核心: 根据SLO(Service Level Objective)限制动态调整batch大小
  → profile_table.csv → 查表获得不同token数量下的延迟预估
  → BudgetRefiner → 调整chunked prefill大小 → 确保decode不被延迟太久

★ SchedulerDynamicBatch:
  → 继承vLLM V1 Scheduler → 但加Ascend专用修改
  → Decode-first chunked prefills → decode优先 → prefill可以被打断
  → Dynamic token budget → 不固定 → 根据SLO调整

★ ★ vs vLLM V1 Scheduler:
  → vLLM: 固定token_budget → 不考虑SLO → decode可能被长prefill阻塞
  → Ascend: BudgetRefiner → SLO-aware → decode优先保证 → prefill动态缩减
  → vLLM: FCFS/PRIORITY → 无SLO概念
  → Ascend: SLO → SLA保证 → 更适合生产环境!
```

## 5. MultiGroupBlockTable — MLA/DSA多组KV

```
★ ★ MultiGroupBlockTable: Ascend独有 → 支持MLA+DSA多组KV cache!

核心: 不同attention type → 不同KV cache组 → 共享block pool
  → MultiGroupBlockTable: 多个BlockTable实例 → 每组独立管理
  → CompressAttentionManager: MLA compress_ratio → KV压缩 → 省内存!

★ ★ vs vLLM HybridKVCacheCoordinator:
  → vLLM: fixed-point算法 → 多attention type共享block pool → 但每组相同block_size
  → Ascend: MultiGroupBlockTable → 每组独立 → compress_ratio不同 → MLA压缩KV
  → vLLM: block_size=16 → 小 → 精细
  → Ascend: block_size=128 → 大 → MLA压缩需要更大block → 效率优先

★ ★ hashk_cache (kvcomp_utils.py):
  → bind_hashk_cache(): hashk_caches → forward_context → 每attention层绑定
  → MLA模型 → hashk_cache = compressed KV → compress_ratio → 独立管理
  → extract_layer_index → 按层序排列 → runner_hashk_caches
```

## 6. PCP + DCP — Ascend上下文并行

```
★ ★ PCP (Prefill Context Parallel) + DCP (Decode Context Parallel):
  → PCP: prefill阶段 → 长prompt → 多GPU并行计算 → 类似SP
  → DCP: decode阶段 → 多GPU → KV cache分布式 → 推理加速
  → pcp_size > 1 → 启用PCP → deepcopy scheduler_output → 防止跨进程修改

★ ★ PCP + Multi-Modal:
  → special deepcopy → temporary workaround → 等vLLM native support
  → PCP MLA mask → get_pcp_mla_mask → 512×512 → pcp_mla_mask

★ AscendForwardContext (ascend_forward_context.py):
  → MoECommType → MoE通信模式 → Ascend专用
  → set_ascend_forward_context → 全局上下文设置 → NPU执行环境
  → _EXTRA_CTX → is_draft_model → spec decode draft model标识
```

## 7. AscendSampler — NPU专用采样

```
★ ★ AscendSampler (sample/sampler.py):
  → async_exponential: 异步指数采样 → separate stream → 不阻塞主stream
  → reduce_sample: TP采样 → 多GPU → reduce到rank0 → 只rank0决定token
  → ★ vs vLLM: greedy/random/mixed三路径 → 但Ascend有async_exponential新选项!

★ ★ c8_quant (INT8 KV cache on Ascend):
  → enable_c8_quant → key_antiquant_scale + value_antiquant_scale
  → antiquant_mode=0 → 反量化模式
  → inner_precise=1 → 内部精确
  → input_layout从TND→BNSD → c8_quant需要BNSD布局
  → ★ Ascend INT8 KV → antiquant_scale + antiquant_mode → vs vLLM INT8 KV(c8_scale)
```

## 8. 关键设计洞察

```
1. NPUModelRunner继承GPUModelRunner → 最小化修改 → Ascend移植最简!
   → 只改关键差异(worker/input_batch/attention/sampler)
   → forward逻辑几乎相同 → PCP特殊case → deepcopy
   → ★ 这是vLLM-Ascend设计的核心策略 → 继承→差异覆盖 → 不重写!

2. 6种Ascend attention → vs vLLM 2-3种 → Ascend更专用!
   → FA/FA3/SFA/DSA/MLA/KVComp → 每种有专用kernel → 不通用但极快
   → vs vLLM: FlashInfer+PagedAttention → 通用但非最优
   → ★ Ascend NPU → 专用kernel → 6种选择 → 性能极致优化!

3. ACL Graph vs CUDA Graph → 核心逻辑相同 → API不同!
   → capture→replay→固定shape → 两者一致
   → torch.npu.graph_task_* → CANN原生 → PyTorch不直接支持
   → ★ ACL Graph = Ascend的CUDA graph → 设计理念完全相同 → 只是实现不同!

4. BudgetRefiner SLO → Ascend独有 → 生产级调度!
   → vLLM无SLO → 固定budget → decode可能被长prefill阻塞
   → Ascend: SLO-aware → decode优先保证 → prefill动态缩减
   → ★ 这是生产级serving的关键差异 → SLO保证 → 更适合云服务!

5. block_size=128 vs 16 → Ascend效率优先 → vLLM精细优先!
   → vLLM: 16 → 93.75%最低利用率 → 精细
   → Ascend: 128 → 更低最低利用率 → 但MLA压缩需要大block → 效率优先
   → ★ MLA compress_ratio → 大block → 压缩比更高 → 省更多内存!

6. CompressAttentionManager MLA → Ascend独有 → DeepSeek-V2/V3 MLA专用!
   → hashk_cache → compressed KV → 独立管理
   → compress_ratio → 1:4或1:8 → KV cache内存大幅减少
   → ★ NVIDIA无等价 → vLLM需要FlashInfer MLA支持 → 但Ascend有原生kernel!

7. RecomputeScheduler → Ascend默认recompute → vs vLLM无swap!
   → vLLM V1: dropped swap → preemption→full reset→从头recompute
   → Ascend: RecomputeScheduler → 专门的recompute路径 → 可能更高效
   → ★ 两者策略相同(无swap,recompute only) → 但Ascend有专门调度器!

8. CaMemAllocator → Ascend CANN内存管理 → sleep/wake映射!
   → vLLM: engine.sleep(level=1 KV/level=2全部) → 释放GPU
   → Ascend: CaMemAllocator → CANN memcpy → 内存池管理 → NPU更细粒度
   → ★ 两者都支持sleep/wake → 但实现不同 → CANN vs PyTorch内存管理!
```

---

Sources:
- vllm-ascend/vllm_ascend/worker/model_runner_v1.py (NPUModelRunner, execute_model, _prepare_inputs)
- vllm-ascend/vllm_ascend/worker/worker.py (NPUWorker)
- vllm-ascend/vllm_ascend/worker/npu_input_batch.py (NPUInputBatch, MultiGroupBlockTable)
- vllm-ascend/vllm_ascend/attention/attention_v1.py (AscendAttentionBackend + 6 impls)
- vllm-ascend/vllm_ascend/attention/attention_mask.py (AttentionMaskBuilder)
- vllm-ascend/vllm_ascend/compilation/acl_graph.py (ACL graph capture/replay)
- vllm-ascend/vllm_ascend/device/device_allocator/camem.py (CaMemAllocator)
- vllm-ascend/vllm_ascend/worker/kvcomp_utils.py (bind_hashk_cache)
- vllm-ascend/vllm_ascend/scheduler/scheduler_dynamic_batch.py (BudgetRefiner)
- vllm-ascend/vllm_ascend/scheduler/recompute_scheduler.py (RecomputeScheduler)
- vllm-ascend/vllm_ascend/sample/sampler.py (AscendSampler)
- vllm-ascend/vllm_ascend/ascend_config.py (AscendConfig)
- vllm-ascend/vllm_ascend/ascend_forward_context.py (AscendForwardContext)
- Background agent research (vLLM-Ascend serving internals)
