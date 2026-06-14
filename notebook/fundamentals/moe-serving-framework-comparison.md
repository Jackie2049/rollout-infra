# MoE Serving 跨框架对比: vLLM vs SGLang vs Megatron vs verl

> 2026-06-15 | 源码级对比: vLLM/SGLang/Megatron-LM/verl MoE推理+训练架构
> 核心: vLLM 7种router+8种AllToAll+EPLB → SGLang DeepEP+6dispatcher+kimi_k2 → Megatron FlexToken+DeepEP/HybridEP+NVLS → verl委托引擎+router replay

## 1. Routing机制对比

### vLLM — 7种Router Factory

```
router_factory.py (39-222): Factory-based路由选择
优先级: RoutingSimulatorRouter → ZeroExpertRouter → GroupedTopKRouter
        → CustomRoutingRouter → FusedTopKBiasRouter → AiterSharedRouted → FusedTopKRouter

FusedTopKRouter (fused_topk_router.py:116-165):
  → scoring_func="softmax"或"sigmoid"
  → renormalize选项
  → CUDA kernels: topk_softmax/topk_sigmoid (69-113)

GroupedTopKRouter (grouped_topk_router.py:247-351):
  → DeepSeek-V2/V3专用!
  → num_expert_group, topk_group, scoring_func
  → routed_scaling_factor, e_score_correction_bias
  → fused_grouped_topk() → 全融合kernel路径(29-72)
  → grouped_topk() → fallback路径(81-162)

RoutingMethodType (config.py:102-126):
  → Default/Renormalize/DeepSeekV3/Llama4/RenormalizeNaive/TopK
  → SigmoidRenorm/MiniMax2/DeepseekV4/Custom/Simulated
  → get_routing_method_type(): 从参数自动检测路由类型

★ vLLM路由最丰富 → 11种routing method → 覆盖几乎所有MoE模型!
```

### SGLang — Multi-Path Routing

```
TopK (topk.py:324-558): MultiPlatformOp
  → forward_cuda/forward_cpu/forward_npu/forward_xpu → 多硬件!
  → output_format依赖moe_runner_backend(STANDARD/TRITON_KERNEL/BYPASSED)

select_experts() (topk.py:1449-1636): 中央路由dispatch
  → use_grouped_topk → torch_native → custom_routing → scoring_func

biased_grouped_topk_gpu() (topk.py:1076-1285): 多路径路由
  → flashinfer fused_topk_deepseek → moe_fused_gate kernel → aiter
  → kimi_k2_moe_fused_gate → jit grouped_topk → fallback

★ kimi_k2专用路由: 384 experts + num_expert_group=1 → 优化超大MoE!
★ _remap_topk_for_deepep(): DeepEP interleaved expert layout → shared experts每rank唯一ID

FusedMoeRouter (router.py:392-429): Triton CUDA kernels
  → cudacore(小batch/expert) → tensorcore(大batch/expert)
```

### Megatron — TopKRouter + InferenceTopKRouter

```
Router (router.py:30-136): 基类 → gating network + weight param
  → moe_router_dtype(fp32/fp64) → 路由精度选择
  → bias support → 路由偏置

TopKRouter (router.py:138-741): 完整路由实现
  → score_function: sigmoid/softmax
  → routing_type: sinkhorn/aux_loss/seq_aux/global_aux → 负载均衡
  → moe_router_num_groups, moe_router_group_topk → grouped top-k
  → expert_bias, z_loss, input jitter → 训练稳定性

★ InferenceTopKRouter (moe_layer.py:749-848):
  → 推理专用 → @torch.compile → dense [num_tokens, topk]
  → CUDA-graphed decode兼容 → 固定shape → 重编译避免!
  → 与训练TopKRouter分离 → 推理路径独立优化
```

### verl — 委托引擎路由

```
verl不自己实现路由 → 委托底层引擎:

Megatron引擎:
  → TopKRouter → moe_router_load_balancing_type, moe_router_topk
  → freeze_moe_router=True(Qwen默认) → RL训练不更新路由 → 稳定!

Automodel引擎:
  → HF路由 → Qwen/Llama原生实现
  → MoEParallelizerConfig → nemo_automodel

★ router_replay (transformer_impl.py:118-124):
  → enable_routing_replay → rollout时存储mini_layer_topk_idx_list
  → 训练时replay → rollout和训练路由一致 → 训练稳定性!
  → mode != "disabled" → 可配置 → 默认disabled

★ freeze_moe_router → RL训练最关键设置!
  → MoE模型RL训练 → 路由不稳定 → 专家漂移 → 训练崩溃
  → freeze + replay → 路由固定 → 稳定RL训练!
```

## 2. Expert Parallelism对比

### vLLM — ExpertMapManager + 8种AllToAll Backend

```
FusedMoE.__init__ (layer.py:102-142):
  → ep_size, dp_size, pcp_size, tp_size → 多维并行

ExpertMapManager (expert_map_manager.py:152-516):
  → global-to-local expert ID映射
  → determine_expert_map() (22-113):
    → "linear"放置 → 专家按rank顺序分配
    → "round_robin"放置 → 交叉分配 → 需DeepEP/NIXL backend!

FusedMoEParallelConfig (config.py:994-1226):
  → use_ep, all2all_backend, enable_eplb
  → use_deepep_ht/ll, use_fi_nvl_one/two_sided, use_mori, use_nixl, use_ag_rs
  → make() (1080-1207): EP enabled → tp_size=ep_size, tp_size=1
    → EP和TP互斥 → EP时不用TP → 专家不分片到TP!

★ 8种AllToAll Backend:
  1. deepep_ht → DeepEP high-throughput → SM90+NVLink
  2. deepep_ll → DeepEP low-latency → decode优化
  3. flashinfer_nvl_one_sided → NVLink单边 → 最简单
  4. flashinfer_nvl_two_sided → NVLink双边 → 更快
  5. mori → AMD MoE → ROCm专用
  6. nixl → NVIDIA统一通信 → 新!
  7. naive_dp_ep → 数据并行+EP → 不用AllToAll
  8. no_dp_ep → 纯EP → 无DP维度

★ RTX 4090: naive_dp_ep(单GPU)或no_dp_ep → 无NVLink → DeepEP不可用!
```

### SGLang — DeepEPMoE + 6种Dispatcher

```
DeepEPMoE (ep_moe/layer.py:47-267): EP实现 → 基于DeepEP
  → 支持FP8/W4AFp8/BF16量化
  → Pipeline: dispatch → run_moe_core → combine
  → get_moe_impl_class() (269-283): deepep/mooncake/nixl/mori → DeepEPMoE, 否则FusedMoE

BaseDispatcher (token_dispatcher/base.py:261-360): 抽象类
  → dispatch/combine方法 + pre/post hooks → 可扩展!

DispatchOutputFormat: STANDARD/DEEPEP_NORMAL/DEEPEP_LL/FLASHINFER
CombineInputFormat: 同上

★ 6种Dispatcher实现:
  1. deepep.py → DeepEP normal/low-latency → SM90必需
  2. flashinfer.py → NVLink单边/双边 → 2种变体
  3. mooncake.py → Mooncake KV transfer → PD分离
  4. moriep.py → AMD MoE → ROCm
  5. nixl.py → NVIDIA统一通信 → 新!
  6. standard.py → 标准AllToAll → 无特殊硬件要求

★ EPLB: expert_location_dispatch_info + topk_ids_logical_to_physical
  → 逻辑expert ID → 物理expert ID映射 → 动态负载均衡!
```

### Megatron — 3种Dispatcher + DeepEP/HybridEP/NVLS

```
BaseMoELayer (moe_layer.py:160-210):
  → ep_group, ep_size, num_local_experts = num_moe_experts // ep_size
  → local_expert_indices → 每rank持有部分专家

Token Dispatcher选择 (moe_layer.py:298-323):
  → "allgather" → MoEAllGatherTokenDispatcher
  → "alltoall" → MoEAlltoAllTokenDispatcher
  → "flex" → MoEFlexTokenDispatcher

★ MoEAllGatherTokenDispatcher (212-351):
  → AllGather across TP×EP → 本地expert selection → 简单但通信量大!

★ MoEAlltoAllTokenDispatcher (354-933): 完整pipeline!
  → preprocess → permute1 → A2A → AG(TP) → sort → experts → unsort → RS(TP) → A2A → unpermute
  → 5步通信 → 2次AllToAll + 1次AllGather + 1次ReduceScatter → TP×EP组合!

★ MoEFlexTokenDispatcher (1395-1618): 最先进!
  → 抽象TP和EP → 单个TP×EP通信组 → 简化!
  → Backend: "deepep" → _DeepepManager 或 "hybridep" → _HybridEPManager
  → HybridEP → fused dispatch/combine → overflow tracking → DeepEP hybrid-ep kernel
  → DeepEP → pad_multiple → 量化alignment → FP8 dispatch兼容

★ NVLS推理Dispatcher (token_dispatcher_inference.py:286-604):
  → NVLS AllGather-V/ReduceScatter-V → multimem kernels
  → Requires Hopper+ + NVLink + symmetric memory → GB200专用!
  → Class-level symmetric buffer handles → _symm_agv_hidden/routing/probs/_symm_rsv
  → 固定buffer → CUDA graph兼容 → decode推理最优!

★ SharedExpert Overlap (shared_experts.py):
  → cached_fc1_input/fc2_input/fc2_output/cached_output → 缓存中间结果
  → Shared expert在dedicated stream → 与AGV dispatch并发 → overlap!
  → DeepSeek-V3: shared experts + routed experts → overlap → 更快!
```

### verl — 委托引擎EP

```
McoreEngineConfig.expert_model_parallel_size (engine.py:186): default 1
  → Megatron mpu.initialize_model_parallel() → EP初始化

AutomodelEngineConfig.ep_size (engine.py:544): default 1
  → create_device_mesh() → DTensor EP mesh

★ MoEParallelizerConfig → nemo_automodel:
  → ep_size > 1 → import MoEParallelizerConfig
  → mp_policy + moe_kwargs → EP配置传递

★ expert_tensor_parallel_size → 专家TP独立 → EP×TP组合!

Torchtitan引擎EP (transformer_impl.py:516-531):
  → EP enabled → per-tensor EP all-gather
  → 否则 → fp32→bf16 → weight sync

★ verl EP = 委托 → 不自己实现 → 随引擎升级自动获得新特性!
```

## 3. MoE + KV Cache交互

```
★ 关键洞察: MoE路由和KV Cache是正交系统!
  → KV Cache: per-head per-token → attention状态
  → MoE Routing: per-token → expert MLP选择
  → 两者不直接交互 → 但RL训练需要桥接!

vLLM RoutedExpertsCapturer (routed_experts_capturer.py:58-220):
  → 捕获per-token routing decisions(topk_ids)
  → indexed by KV cache slot_mapping → block_id * block_size + offset
  → device buffer: (max_num_batched_tokens, num_layers, num_experts_per_tok)
  → RoutedExpertsManager (223-350): scheduler端 → slot-indexed buffer
  → ★ routing数据随KV cache block → preemption后路由数据不丢失!
  → ★ 关键限制: enable_return_routed_experts与KV connector不兼容!
    → PD分离(PD disaggregation)和KV offload不能同时使用!
    → config/vllm.py (883-890) → 硬性限制

SGLang RoutedExpertsCapturer (routed_experts.py:20-126):
  → BaseTopkCapturer device_cache → DP-rank-aware slicing
  → DeepEP a2a backend → all-gather topk_ids across attn-TP → capture时聚合
  → 返回: base64 encoding in meta_info → HTTP API传递
  → OpenAI protocol: return_routed_experts: bool, routed_experts_start_len: int
  → IndexerTopkCapturer → alternative → indexing use cases

Megatron:
  → MoE不直接交互KV Cache → attention独立管理
  → InferenceTopKRouter @torch.compile → dense routing → CUDA graph兼容
  → FP8 KV cache → kv_cache_scaling_factor → TRT-LLM export兼容

verl:
  → 委托底层引擎 → vLLM RoutedExpertsCapturer → 同vLLM

★ RL训练桥接: RoutedExpertsCapturer → GRPO/PPO需要知道哪些expert被选中!
  → slot_mapping索引 → preemption后路由恢复 → prefix reuse of routing!
  → 这是vLLM/SGLang独有的特性 → Megatron/verl(Megatron引擎)没有!
```

## 4. EPLB (Expert Parallel Load Balancing)

```
vLLM:
  → enable_eplb → redundant_experts + zero_expert_type + hash_indices_table
  → 动态expert迁移 → 负载均衡 → 减少straggler
  → set_eplb_state() → 运行时配置 → 不重启!

SGLang:
  → expert_location_dispatch_info → logical→physical mapping
  → topk_ids_logical_to_physical → TopK类内实现
  → 与DeepEP集成 → dispatch时自动映射

Megatron:
  → 无直接EPLB → 用aux_loss/seq_aux负载均衡 → 训练时
  → 推理无动态EPLB → 固定expert分配

verl:
  → 无直接EPLB → 委托引擎 → 随引擎升级获得

★ vLLM EPLB最先进 → redundant_experts → 动态负载 → 生产级!
★ SGLang EPLB → logical→physical → 与DeepEP协同 → 灵活!
```

## 5. FP8/量化支持对比

```
vLLM FusedMoEQuantConfig (config.py:209-584):
  → fp8/int8/int4/nvfp4/mxfp4/mxfp8/ocp_mx/block → 8种量化格式!
  → 12+ expert kernel implementations:
    triton_moe/cutlass_moe/flashinfer_moe/deep_gemm_moe/marlin_moe
    rocm_aiter_moe/nvfp4_emulation_moe/ocp_mx_emulation_moe
    trtllm variants(bf16/fp8/mxfp4/nvfp4/mxint4)

SGLang:
  → 10+ quantization schemes: awq_moe/compressed_tensors(6变体)/gptq_moe/modelslim/quark/mxfp4/int4fp8
  → 10+ runner backends: aiter/deep_gemm/flashinfer(3)/marlin/triton/cudacore/tensorcore
  → MXFP4 Marlin MoE → DeepSeekV4 MXFP4 → Marlin backend → 推理专用

Megatron:
  → MXFP8 inference → fused quant+swizzle kernels → 推理最优
  → ★ 不兼容training/colocated RL! → 训练用BF16
  → flashinfer/torch/vllm grouped_gemm → 3种推理backend

verl:
  → enable_fp8 → nemo_automodel → Automodel引擎
  → vLLM FP8 utils → rollout量化 → moe_config传递
  → ModelOpt QAT → _modelopt_moe_init_marlin → 训练后量化

★ RTX 4090量化选择:
  → INT4+INT8KV → 推理最优 → vLLM marlin_moe → 4,791 tok/s
  → FP8 → RTX 4090不支持SM90 → 不可用DeepEP FP8
  → BF16 → 训练唯一选择 → 不量化训练权重
```

## 6. 跨框架全景对比表

| Aspect | vLLM | SGLang | Megatron | verl |
|--------|------|---------|----------|------|
| **Router类型** | 11种method factory | Multi-path(flashinfer/triton/aiter/kimi_k2) | TopK+InferenceTopK | 委托引擎 |
| **EP Backend** | 8种AllToAll | 6种Dispatcher | 3种+DeepEP/HybridEP/NVLS | 委托(ep_size/EP_MPU) |
| **EPLB** | redundant_experts+动态迁移 | logical→physical+DeepEP | 无(aux_loss训练均衡) | 无 |
| **KV+MoE** | RoutedExpertsCapturer+slot_mapping | RoutedExpertsCapturer+DeepEP AG | 无直接交互 | 委托vLLM |
| **量化** | 8格式+12+kernel | 10+scheme+10+runner | MXFP8(推理only) | FP8(nemo/vLLM) |
| **推理Dispatcher** | naive_dp/no_dp | standard | NCCL/NVLS(Hopper+) | 委托 |
| **训练Dispatcher** | 无(推理only) | 无(推理only) | AllGather/AlltoAll/Flex | 委托Megatron |
| **Shared Expert** | 配置支持 | num_fused_shared_experts | Overlap with routed | 委托 |
| **DeepEP集成** | ht/ll backend | DeepEPMoE层 | _DeepepManager | 委托 |
| **NVLS集成** | 无 | 无 | inference dispatcher(Hopper+) | 委托 |

## 7. 关键设计洞察

```
1. 推理框架(vLLM/SGLang) vs 训练框架(Megatron/verl) → MoE实现截然不同!
   → 推理: 融合kernel+量化+AllToAll backend → 最小延迟
   → 训练: 通信优化+负载均衡+梯度同步 → 最大吞吐
   → verl独特: 委托架构 → 同一框架支持两者 → 但不自己实现!

2. RoutedExpertsCapturer → RL训练+推理的桥梁 → 2025-2026关键创新!
   → GRPO/PPO需要知道expert选择 → 但推理引擎不保存routing
   → slot_mapping索引 → routing数据随KV block → preemption恢复
   → vLLM/SGLang独有 → Megatron训练自含routing → 不需要capture
   → 限制: 与PD分离不兼容 → GRPO训练不能同时用PD分离!

3. DeepEP → 跨框架共识 → 但实现层级不同!
   → vLLM: prepare_finalize dispatcher → 调用层
   → SGLang: DeepEPMoE层 → 模型层 → 更完整
   → Megatron: _DeepepManager → 通信层 → 最底层控制
   → 都需要SM90+NVLink → RTX 4090不可用!

4. EPLB → 生产推理关键 → 但训练不需要!
   → 推理: expert负载不均 → latency spike → EPLB解决
   → 训练: aux_loss → 训练时自然均衡 → 推理时固定分配
   → vLLM最先进 → redundant_experts → 备用expert → 动态迁移

5. NVLS → Megatron独有 → GB200推理优化 → 未来方向!
   → AllGather-V/ReduceScatter-V → variable token count → decode兼容
   → symmetric memory → multimem → NVLink专用 → Hopper+必需
   → vLLM/SGLang无NVLS → 但有FlashInfer NVLink → 类似但不同
   → GB200集群 → Megatron NVLS → 最优推理路径

6. MoE + PD分离 → 矛盾!
   → RoutedExpertsCapturer与KV connector不兼容 → GRPO不能PD分离
   → 解决方案: GRPO用单集群推理 → 不PD分离 → 但吞吐低
   → 未来: 可能需要新的routing capture机制 → 与PD分离兼容

7. verl freeze_moe_router → RL训练MoE最关键配置!
   → MoE RL训练 → 路由不稳定 → expert drift → 训练崩溃
   → freeze → 路由固定 → 只训expert weights → 稳定!
   → router_replay → rollout和训练路由一致 → 进一步稳定
   → Qwen MoE默认freeze → 生产级配置!
```

---

Sources:
- vLLM: fused_moe/router/, fused_moe/layer.py, fused_moe/expert_map_manager.py, fused_moe/config.py, fused_moe/prepare_finalize/, routed_experts_capturer.py
- SGLang: moe/topk.py, moe/router.py, moe/ep_moe/, moe/token_dispatcher/, state_capturer/routed_experts.py
- Megatron-LM: moe/router.py, moe/moe_layer.py, moe/token_dispatcher.py, moe/token_dispatcher_inference.py, moe/shared_experts.py
- verl: workers/config/engine.py, workers/engine/megatron/transformer_impl.py, models/mcore/model_initializer.py, utils/megatron_utils.py
- notebook/projects/deepep-source-reading.md + deepep-v2-reading.md + deepep-megatron-integration-latest.md
- notebook/projects/deepspeed-latest-developments-2026-06.md (AutoEP)
