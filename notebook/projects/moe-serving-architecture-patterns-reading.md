# MoE Serving 架构模式跨框架综合阅读

> 2026-06-15 | 综合: vLLM/SGLang/Megatron/verl/DeepSpeed 源码级对比
> 核心: 6大MoE架构模式 — routing/EP通信/EPLB/RL训练/CUDA Graph/内存估算 — 跨框架深度对比
> 基于: deepep-source-reading.md + deepep-v2-reading.md + deepep-megatron-integration-latest.md + moe-serving-framework-comparison.md + vllm-v0.23-new-features-reading.md + megatron-inference-engine-reading.md + megatron-moe-architecture.md + vllm-moe-serving-reading.md + vllm-cuda-graph-reading.md + verl-grpo-data-flow-reading.md + deepspeed-latest-developments-2026-06.md

---

## 1. Expert Routing 模式

### 1.1 跨框架Router全景

```
5种主流routing范式:
  → softmax top-k (最常见, Mixtral/Qwen2-MoE)
  → sigmoid top-k (DeepSeek-V2/V3/V4)
  → grouped top-k (DeepSeek-V2/V3, num_expert_group+topk_group)
  → kimi_k2专用路由 (384 experts, num_expert_group=1 → 优化超大MoE)
  → shared expert routing (DeepSeek系列, 共享+路由双通道)
```

### 1.2 vLLM — 11种Routing Method Factory

```
router_factory.py 优先级链:
  RoutingSimulatorRouter → ZeroExpertRouter → GroupedTopKRouter
  → CustomRoutingRouter → FusedTopKBiasRouter → AiterSharedRouted → FusedTopKRouter

RoutingMethodType (config.py:102-126):
  Default / Renormalize / DeepSeekV3 / Llama4 / RenormalizeNaive / TopK
  SigmoidRenorm / MiniMax2 / DeepseekV4 / Custom / Simulated
  → ★ 11种routing method → 覆盖几乎所有MoE模型!

GroupedTopKRouter (grouped_topk_router.py:247-351):
  → DeepSeek-V2/V3专用!
  → num_expert_group, topk_group → 先选组,组内选top-k
  → routed_scaling_factor → 路由专家权重缩放
  → e_score_correction_bias → 修正偏差 → DeepSeek-V3原文公式
  → fused_grouped_topk() → 全融合kernel路径
  → ★ vs softmax: sigmoid评分+grouped筛选 → 更精准但更复杂

FusedTopKRouter CUDA kernels:
  → topk_softmax / topk_sigmoid → 融合top-k+评分
  → renormalize选项 → softmax后重归一化 → 某些模型需要
```

### 1.3 SGLang — Multi-Path Routing

```
select_experts() (topk.py:1449-1636): 中央路由dispatch
  → use_grouped_topk → torch_native → custom_routing → scoring_func
  → 多路径fallback → 硬件适配!

biased_grouped_topk_gpu() (topk.py:1076-1285):
  → flashinfer fused_topk_deepseek → moe_fused_gate kernel → aiter
  → ★ kimi_k2_moe_fused_gate → jit grouped_topk → fallback
  → ★ _remap_topk_for_deepep() → DeepEP interleaved expert layout
     → shared experts每rank唯一ID → remap topk_ids → DeepEP兼容!

★ kimi_k2专用路由: 384 experts + num_expert_group=1 → 优化超大MoE!
   → 不需要grouped筛选 → 所有384专家平等竞争 → 极大简化路由!

FusedMoeRouter (router.py:392-429): Triton CUDA kernels
  → cudacore(小batch/expert) → tensorcore(大batch/expert)
  → 自动选择最优kernel → 无需手动配置!
```

### 1.4 Megatron — 训练vs推理双路由

```
训练: TopKRouter (router.py:138-741)
  → score_function: sigmoid/softmax
  → routing_type: sinkhorn/aux_loss/seq_aux/global_aux → 4种负载均衡!
  → moe_router_dtype(fp32/fp64) → 路由精度选择 → fp32更稳定
  → expert_bias, z_loss, input jitter → 训练稳定性3层保障
  → moe_router_num_groups, moe_router_group_topk → grouped top-k

★ ★ 推理: InferenceTopKRouter (router.py:749-848)
  → @torch.compile + dense_output=True → 2大优化!
  → 跳过: z_loss / aux losses / token dropping / expert bias updates
  → dense_output → [num_tokens, topk] → 不是sparse [num_tokens, num_experts]
  → FlashInfer grouped GEMM直接按top_indices索引 → 极快!
  → CUDA graph兼容 → 固定shape → 无Python overhead!

★ 关键设计: 推理和训练用不同路由器!
  → 训练: 负载均衡+z_loss+aux_loss → 保证训练稳定性
  → 推理: compile+dense+no_aux → 保证推理速度
  → InferenceMode.is_active() → 自动切换 → 零配置!
```

### 1.5 verl — 委托引擎 + freeze + replay

```
verl不自己实现路由 → 委托底层引擎:
  Megatron: TopKRouter → moe_router_load_balancing_type, moe_router_topk
  Automodel: HF路由 → Qwen/Llama原生
  TorchTitan: per-tensor EP all-gather → 统一通信

★ ★ freeze_moe_router=True (Qwen默认):
  → RL训练不更新路由权重 → 路由稳定 → expert不漂移!
  → MoE RL训练最大问题: 路由不稳定 → 专家漂移 → 训练崩溃
  → freeze → 只训expert weights → 路由固定 → 稳定!

★ ★ router_replay (transformer_impl.py:118-124):
  → enable_routing_replay → rollout时存储mini_layer_topk_idx_list
  → 训练时replay → rollout和训练路由一致 → 进一步稳定!
  → vs freeze: freeze冻结权重 → replay固定决策 → 双保险!
  → ★ Megatron GRPO cudagraph warmup也用router_replay → 记录路由决策!

★ verl路由 = 委托 + freeze + replay → 3层稳定保障!
```

### 1.6 DeepSpeed — AutoEP + TorchTitan内核

```
AutoEP (PR #7938, merged 2026-06-11):
  → deepspeed.initialize()自动检测MoE → 替换为AutoEPMoELayer
  → EP router → TokenChoiceTopKRouter ← TorchTitan
  → 无freeze_moe_router → 不专门支持RL → 但可以手动freeze权重!

★ vs Megatron: AutoEP零代码修改 → HF兼容 → 但无DeepEP asymmetric
★ vs verl: 无freeze/replay → RL训练需额外配置!
```

### 1.7 Scoring Function对比: Softmax vs Sigmoid

```
Softmax top-k (Mixtral/Qwen2-MoE/Llama4):
  → scores = softmax(logits)[top-k indices]
  → 天然归一化 → scores sum = 1 → 简单稳定
  → 缺点: 大num_experts时softmax计算贵 → 需要O(E)计算!

Sigmoid top-k (DeepSeek-V2/V3/V4):
  → scores = sigmoid(logits)[top-k indices]
  → 不归一化 → 需要手动renormalize → scores sum不一定=1
  → 优点: sigmoid是O(1) → 大num_experts更快!
  → DeepSeek-V3: sigmoid + renormalize + routed_scaling_factor
  → vLLM: SigmoidRenorm routing method → sigmoid后手动normalize

★ DeepSeek-V3 grouped top-k:
  → 先分组选topk_group → 组内选top-k
  → sigmoid评分 → e_score_correction_bias修正
  → num_expert_group=4, topk_group=2 → 先选2组 → 组内选4expert → top-8
  → 为什么grouped? → 256 experts太多 → softmax/sigmoid计算贵 → grouped降维度!

★ RTX 4090 routing: 全在GPU上 → SM89兼容 → 无硬件限制
   → top-k kernel: Triton/CUDA → RTX 4090✓
   → grouped top-k: 更复杂 → 但RTX 4090✓ (纯计算kernel)
```

### 1.8 Shared Expert 设计模式

```
DeepSeek-V2/V3/V4: shared expert + routed expert → 双通道!

架构:
  output = shared_expert(tokens) + routed_scaling_factor * sum(routed_expert_i * weight_i)

  → shared_expert: 所有token都经过 → 不路由 → 稳定baseline
  → routed_expert: top-k token选择 → specialization
  → routed_scaling_factor: 路由权重缩放 → 防止routed淹没shared

★ Megatron SharedExpertMLP (shared_experts.py):
  → cached_fc1_input/fc2_input/fc2_output → 缓存中间结果
  → Shared expert在dedicated stream → 与AGV dispatch并发 → overlap!
  → NVLS独有overlap → NCCL路径不支持overlap!

★ vLLM: num_fused_shared_experts → 配置支持 → 但无overlap
★ SGLang: num_fused_shared_experts → DeepEPMoE层支持

★ RTX 4090: shared expert无overlap → NCCL路径 → sequential
   → 但shared expert只1个 → 计算量小 → sequential影响有限!
```

---

## 2. Expert Parallelism 通信模式

### 2.1 通信模式全景

```
4种EP通信范式:

  A. Symmetric All-to-All (NCCL):
     → 每个rank发等量chunk → padding浪费 → 通用但慢
     → Megatron AlltoAllTokenDispatcher / DeepSpeed AutoEP

  B. Asymmetric Dispatch/Combine (DeepEP):
     → 不同expert不同token数 → 无padding → MoE天然模式!
     → DeepEP V1/V2 / SGLang DeepEPMoE / vLLM DeepEP-HT/LL

  C. AllGather-V/ReduceScatter-V (NVLS):
     → variable-count AllGather → multicast TMA → Hopper+NVLink专用
     → Megatron NVLSAllGatherVDispatcher

  D. NVLink单边/双边 (FlashInfer):
     → 节点内NVLink All-to-All → 不需RDMA → 最简单
     → vLLM/SGLang FlashInfer NVLink backend
```

### 2.2 DeepEP — Asymmetric核心设计

```
Dispatch: token路由到expert GPU
  Phase 1: dispatch_impl → input tokens → symmetric send buffers → NCCL Gin put/signal
  Phase 2: dispatch_copy_epilogue → symmetric recv buffers → output tensors

Combine: expert输出回到原token位置
  Phase 1: combine_impl → expert outputs → symmetric recv buffers → Gin put/signal
  Phase 2: combine_reduce_epilogue → reduce(weighted sum用top-k weights) → 最终output

★ 5个asymmetric优势根源:
  1. 不对称通信 → NCCL padding浪费 → DeepEP原生asymmetric → 4.6x!
  2. Bypass NCCL proxy thread → Gin → GPU kernel直接RDMA
  3. SM效率 → V2: 4-6 SM → 98 SM留给计算!
  4. FP8 dispatch → 1 byte vs BF16 2 bytes → 跨节点流量减半!
  5. Handle caching → decode推理 → reuse routing handle → skip CPU sync!

★ 两种通信模式:
  Direct (单节点): NVLink-only → 一组send/recv buffers → 最简单
  Hybrid (多节点): NVLink域内转发 + RDMA域跨节点 → 分层通信
    → channels(8/SM) → linked list tracking → 跟踪转发链

★ Expert Alignment: 对齐每个expert的token数到128 → GEMM-friendly!
★ Expand Mode: one-slot-per-expert-per-token → 直接feed grouped GEMM → DeepSeek-V3 style!
★ Deterministic Mode: backward用 → 多次dispatch一致 → 正确性!
```

### 2.3 DeepEP V2 — ElasticBuffer统一接口

```
V1 → V2 重构:
  NVSHMEM → NCCL Gin backend → 更轻量
  Buffer + EPHandle → ElasticBuffer → 一次初始化 → 训练+推理统一

ElasticBuffer API:
  buffer = ElasticBuffer(group, num_max_tokens, hidden, num_topk, use_fp8_dispatch)
  recv_x, handle, event = buffer.dispatch(x, topk_idx, topk_weights, ...)
  combined_x, event = buffer.combine(y, handle=handle, ...)

★ 通信-计算overlap关键:
  recv_x, ..., handle, event = buffer.dispatch(..., async_with_compute_stream=True)
  independent_work()  → 不依赖recv_x → 与通信并行!
  event.current_stream_wait()  → 等待通信完成
  y = expert(recv_x)  → 现在安全使用!

★ Handle Caching (decode关键优化):
  decode_dispatch(x, cached_handle=handle) → 跳过layout重算+CPU sync!
  → decode阶段路由模式稳定 → 大幅减少延迟!
```

### 2.4 Megatron FlexTokenDispatcher — 3 Backend

```
★ MoEAlltoAllTokenDispatcher (354-933): 完整pipeline!
  preprocess → permute1 → A2A → AG(TP) → sort → experts
  → unsort → RS(TP) → A2A → unpermute
  → 5步通信 → 2次AllToAll + 1次AllGather + 1次ReduceScatter → TP×EP组合!

★ MoEFlexTokenDispatcher (1395-1618): 最先进!
  → 抽象TP和EP → 单个TP×EP通信组 → 简化!
  → Backend选择: deepep → _DeepepManager / hybridep → _HybridEPManager
  → 一行配置切换: --moe-flex-dispatcher-backend deepep/hybridep!

★ MoEAllGatherTokenDispatcher (212-351): 简单但通信量大!
  → AllGather across TP×EP → 本地expert selection → 所有rank看全部token!

★ ★ 推理Dispatcher (token_dispatcher_inference.py):
  NCCLAllGatherDispatcher:
    CG path (decode): 等量tokens → .fill_() → 无host sync → CUDA graph安全!
    Non-CG path (prefill): pad到max → host sync → 不在graph下运行!

  ★ ★ NVLSAllGatherVDispatcher (Hopper+专用):
    → multimem_all_gatherv_3tensor → 同时gather hidden/routing/probs → 3tensor!
    → fused_metadata_update Triton kernel → 单CTA → 5操作融合!
    → 固定symmetric buffer → 不动态分配 → CUDA graph兼容!
    → Shared expert overlap: dedicated stream → 与AGV并发 → NVLS独有!
```

### 2.5 vLLM — 8种AllToAll Backend

```
FusedMoE模块化架构:
  [Router] → [Prepare] → [Experts] → [Finalize]

  Prepare: 量化+All-to-All Dispatch+Permute → 分发token到专家
  Experts: matmul+activation+matmul → 专家计算
  Finalize: All-to-All Combine+weight+reduce → 收集结果

★ 8种AllToAll Backend (prepare_finalize/):
  1. deepep_ht → DeepEP high-throughput → SM90+NVLink → prefill优化
  2. deepep_ll → DeepEP low-latency → SM90+NVLink → decode优化
  3. flashinfer_nvl_one_sided → NVLink单边 → 最简单 → 节点内
  4. flashinfer_nvl_two_sided → NVLink双边 → 更快 → 节点内
  5. mori → AMD MoE → ROCm专用 → Mori通信库
  6. nixl → NVIDIA统一通信 → 新! → RDMA+NVLink统一
  7. naive_dp_ep → DP+EP → 不用AllToAll → 最简单fallback
  8. no_dp_ep → 纯EP → 无DP维度 → 最基础

★ ExpertMapManager: global-to-local expert ID映射
  → "linear"放置 → 按rank顺序 → 最简单
  → "round_robin"放置 → 交叉分配 → 需DeepEP/NIXL backend!
  → EPLB时动态调整 → 物理ID ≠ 逻辑ID!

★ vLLM EP和TP互斥!
  → EP enabled → tp_size=ep_size, tp_size=1
  → 专家不分片到TP → EP独占通信 → 不TP分片专家!
```

### 2.6 SGLang — DeepEPMoE + 6 Dispatcher

```
DeepEPMoE (ep_moe/layer.py:47-267):
  → dispatch → run_moe_core → combine → 3步pipeline
  → 支持FP8/W4AFp8/BF16量化 → 量化dispatch → 精度combine
  → get_moe_impl_class(): deepep/mooncake/nixl/mori → DeepEPMoE, 否则FusedMoE

★ 6种Dispatcher (token_dispatcher/):
  1. deepep → DeepEP normal/low-latency → SM90必需
  2. flashinfer → NVLink单边/双边 → 2种变体
  3. mooncake → Mooncake KV transfer → PD分离场景
  4. moriep → AMD MoE → ROCm专用
  5. nixl → NVIDIA统一通信 → 新!
  6. standard → 标准AllToAll → 无特殊硬件 → fallback

★ DispatchOutputFormat: STANDARD/DEEPEP_NORMAL/DEEPEP_LL/FLASHINFER
  → 不同dispatcher输出不同格式 → Expert kernel需适配!
```

### 2.7 verl — 委托引擎EP

```
verl EP = 委托 → 不自己实现:
  McoreEngineConfig.expert_model_parallel_size → Megatron mpu.initialize_model_parallel()
  AutomodelEngineConfig.ep_size → DTensor EP mesh
  MoEParallelizerConfig → nemo_automodel → EP配置传递
  ★ expert_tensor_parallel_size → 专家TP独立 → EP×TP组合!

★ verl EP随引擎升级自动获得新特性!
  → Megatron引擎升级 → verl自动获得DeepEP/HybridEP
  → Automodel引擎升级 → verl自动获得DTensor EP
  → 委托架构优势: 无需自己维护 → 随上游升级!
```

### 2.8 DeepSpeed AutoEP — TorchTitan内核

```
AutoEP = TorchTitan内核 + DeepSpeed集成层:
  → TokenChoiceTopKRouter ← TorchTitan
  → GroupedExperts ← TorchTitan → grouped-GEMM → 单次kernel!
  → TokenReorderer + generate_permute_indices + Triton fill-indices ← TorchTitan

★ vs Megatron DeepEP:
  AutoEP: symmetric all-to-all → NCCL标准 → 无asymmetric优化
  Megatron: DeepEP → asymmetric → 4.6x faster → 但需SM90+NVLink
  → AutoEP简单但慢 → Megatron复杂但快 → 取舍!

★ ★ grouped-GEMM → 单次kernel → 比逐expert GEMM更快!
  → 传统: for expert in experts: GEMM → 8次launch → 8次memory
  → grouped: 一次launch → 1次memory → 8x fewer launches!
  → 跨平台共识: Megatron/DeepSpeed/MindIE(ATB)都有grouped matmul!
```

### 2.9 RTX 4090 EP通信选择

```
★ ★ RTX 4090 EP通信限制:

可用:
  ✓ NCCL All-to-All → symmetric → padding浪费但唯一通用选项
  ✓ naive_dp_ep → 无AllToAll → DP+EP → 最简单
  ✓ no_dp_ep → 纯EP → 无DP → 最基础

不可用:
  ✗ DeepEP V1/V2 → SM90必需 → RTX 4090 SM89不满足!
  ✗ HybridEP → TMA指令 → SM90必需 → RTX 4090不支持!
  ✗ FlashInfer NVLink → 需NVLink → RTX 4090只有PCIe!
  ✗ NVLS → Hopper+NVLink+symmetric memory → RTX 4090全不满足!
  ✗ NIXL RDMA → 需RDMA → RTX 4090无RDMA!
  ✗ FP8 dispatch → DeepEP FP8 → SM90必需!

★ 结论: RTX 4090 EP=1 → 单GPU → 无分布式通信 → NCCL All-to-All=0
   → DeepEP无用 → 但理解asymmetric优势 → AI infra思维
   → H100集群 → DeepEP → 4.6x → MoE推理必须!
```

---

## 3. Expert Load Balancing (EPLB)

### 3.1 为什么需要EPLB?

```
MoE负载不均衡的根源:
  → 不同expert收到不同数量token → "hot expert" vs "cold expert"
  → hot expert → 更多token → 更长计算时间 → latency spike!
  → cold expert → 少token → GPU空闲 → 浪费资源!
  → EP场景: hot expert所在rank是straggler → 整个batch延迟!

★ 推理EPLB vs 训练EPLB 完全不同:
  推理: 动态负载均衡 → 运行时迁移expert → 减少straggler
  训练: aux_loss → 训练时自然均衡 → 推理时固定分配
  → 推理需要EPLB → 训练不需要(训练时均衡已在权重中)!
```

### 3.2 vLLM EPLB — redundant_experts 动态迁移

```
★ vLLM EPLB最先进 (PR#43339, DS-V4):
  → enable_eplb → redundant_experts + zero_expert_type + hash_indices_table
  → 动态expert迁移 → 负载均衡 → 减少straggler
  → set_eplb_state() → 运行时配置 → 不重启!

  redundant_experts: 备用expert → 热expert复制到其他rank → 请求分流!
  → 类似: 多副本 → 负载分流 → hot expert不再瓶颈!

  EPLB配置 (8xB200性能数据):
    无EPLB: 1,288 output tok/s → 有EPLB: 1,350 tok/s → +4.8%!
    Mean TTFT: 1,694ms → 1,520ms → -10.3%!
    → ★ TTFT改善最大 → prefill阶段最受益 → hot expert latency spike消除!

★ ExpertMapManager: logical→physical映射
  → enable_eplb → 逻辑expert ID ≠ 物理expert ID → 动态映射!
  → hash_indices_table → 映射表 → lookup极快!
  → round_robin放置 → 交叉分配 → 为redundant_experts做准备!
```

### 3.3 SGLang EPLB — logical→physical + DeepEP协同

```
SGLang EPLB:
  → expert_location_dispatch_info → 逻辑→物理映射表
  → topk_ids_logical_to_physical → TopK类内实现 → 路由时直接映射!
  → 与DeepEP集成 → dispatch时自动映射 → 无额外步骤!

★ ★ _remap_topk_for_deepep(): DeepEP interleaved expert layout
  → shared experts每rank唯一ID → remap → 兼容DeepEP的interleaved分配!
  → ★ 这是SGLang独有的 → vLLM没有这种DeepEP协同remap!

★ ★ kimi_k2专用负载均衡:
  → 384 experts → num_expert_group=1 → 不grouped → 所有专家平等
  → 但384个expert → 负载不均更严重 → 需要EPLB!
```

### 3.4 Megatron — 训练均衡 vs 推理固定

```
训练均衡 (4种):
  1. Aux Loss: Switch Transformer → E * sum(f_i * P_i) → 促均衡路由
  2. Z-Loss: log(sum(exp(z)))^2 → 限制logit magnitude → 稳定
  3. Sinkhorn: 最优传输 → 理论最优均衡 → 但计算贵
  4. Seq Aux / Global Aux: 序列级/全局级 → 不同粒度

★ MoEAuxLossAutoScaler: 自动调节aux_loss权重 → 动态平衡!

推理: 无动态EPLB → 固定expert分配
  → 训练时均衡 → 推理时路由稳定 → 不需要迁移expert!
  → InferenceTopKRouter → 跳过aux_loss → 不做均衡计算 → 更快!

★ ★ NVLS shared expert overlap → 部分均衡效果:
  → shared expert在dedicated stream → 与routed expert并发 →
  → 即使routed不均衡 → shared提供稳定baseline → 间接均衡!
```

### 3.5 verl — 无直接EPLB → 委托引擎

```
verl无自己的EPLB → 委托引擎:
  Megatron引擎 → 无推理EPLB → 训练aux_loss
  vLLM引擎 → vLLM EPLB → 运行时迁移 → 随引擎升级获得!
  → ★ verl用vLLM rollout → 自动获得vLLM EPLB → 无需自己实现!
```

### 3.6 DeepSpeed AutoEP — 无EPLB

```
AutoEP无推理EPLB → 训练aux_loss:
  → AutoEP面向训练 → 训练时均衡 → 推理需额外工具
  → ★ 推理用TRT-LLM或vLLM → 不用DeepSpeed → 无需AutoEP EPLB!
```

### 3.7 EPLB策略对比表

| 框架 | 推理EPLB | 训练均衡 | 动态迁移 | RTX 4090可行性 |
|------|----------|----------|----------|----------------|
| vLLM | redundant_experts+运行时配置 | 无训练(推理only) | yes | EP=1不需要 |
| SGLang | logical→physical+DeepEP协同 | 无训练(推理only) | yes | EP=1不需要 |
| Megatron | 无(aux_loss训练均衡) | aux_loss/z-loss/sinkhorn | no | 训练均衡✓ |
| verl | 委托(vLLM/Megatron) | 委托引擎 | 委托 | EP=1不需要 |
| DeepSpeed | 无(TRT-LLM/vLLM推理) | aux_loss(AutoEP) | no | EP=1不需要 |

```
★ RTX 4090 EP=1 → 无分布式 → 无straggler → EPLB不需要!
   → 但理解EPLB原理 → H100集群推理需要!
```

---

## 4. MoE + RL Training

### 4.1 核心矛盾: 路由不稳定 + expert drift

```
MoE RL训练最大问题:
  → RL更新actor权重 → 路由权重也更新 → 路由决策变化 → expert drift!
  → expert drift: 原来token→expert_A → 更新后→expert_B → 不一致!
  → rollout时路由 → 训练时路由不同 → off-policy问题 → 训练崩溃!

★ 3层稳定保障 (verl设计):
  1. freeze_moe_router=True → 路由权重不更新 → 路由固定!
  2. router_replay → rollout路由决策记录 → 训练时replay → 一致!
  3. GRPO不需要critic → 只有actor → 更简单 → 更稳定!
```

### 4.2 verl freeze_moe_router + router_replay

```
★ ★ freeze_moe_router=True (Qwen MoE默认):
  → 路由权重self.gating.weight → 不参与optimizer → 不更新!
  → 只训expert weights → 路由固定 → 无expert drift → 稳定!
  → ★ 这不是hack → 是verl正式配置 → Qwen MoE推荐!

★ ★ router_replay (transformer_impl.py:118-124):
  → enable_routing_replay → rollout时存储mini_layer_topk_idx_list
  → 训练时replay → rollout和训练路由完全一致!
  → ★ vs freeze: freeze冻结权重 → replay冻结决策 → 双保险!
  → ★ Megatron GRPO cudagraph warmup也用router_replay → 记录路由决策!

★ verl MoE RL训练流程:
  rollout → router决策 → 存topk_idx → GRPO advantage → actor update
  → freeze → router权重不变 → expert权重更新 → 路由固定 → 稳定!
  → replay → rollout路由 → 训练复用 → 完全on-policy → 稳定!
```

### 4.3 RoutedExpertsCapturer — RL+推理桥梁

```
vLLM RoutedExpertsCapturer (routed_experts_capturer.py:58-220):
  → 捕获per-token routing decisions(topk_ids)
  → indexed by KV cache slot_mapping → block_id * block_size + offset
  → device buffer: (max_num_batched_tokens, num_layers, num_experts_per_tok)
  → RoutedExpertsManager (223-350): scheduler端 → slot-indexed buffer
  → ★ routing数据随KV cache block → preemption后路由数据不丢失!

SGLang RoutedExpertsCapturer (routed_experts.py:20-126):
  → BaseTopkCapturer device_cache → DP-rank-aware slicing
  → DeepEP a2a backend → all-gather topk_ids across attn-TP → capture时聚合
  → 返回: base64 encoding in meta_info → HTTP API传递
  → OpenAI protocol: return_routed_experts: bool, routed_experts_start_len: int

★ ★ 关键限制: enable_return_routed_experts与KV connector不兼容!
  → PD分离(PD disaggregation)和KV offload不能同时使用!
  → config/vllm.py (883-890) → 硬性限制

为什么?
  → KV connector → token在不同GPU间移动 → slot_mapping变化!
  → RoutedExpertsCapturer → slot_mapping索引 → slot变化 → 索引失效!
  → PD分离 → prefill GPU → decode GPU → 不同slot → routing数据丢失!

★ ★ GRPO不能同时用PD分离 → 这是架构级限制!
  → 解决方案1: GRPO用单集群推理 → 不PD分离 → 但吞吐低
  → 解决方案2: 未来需要新的routing capture机制 → 与PD分离兼容
  → ★ 这是2026年MoE RL训练的关键矛盾!
```

### 4.4 MoE RL训练数据流

```
GRPO + MoE完整数据流:
  gen_batch → repeat(rollout_n) → vLLM/SGLang rollout
  → RoutedExpertsCapturer捕获topk_ids → 随response返回
  → reward → GRPO advantage → actor update
  → freeze_moe_router → 路径权重不变 → 只训expert weights

★ verl vs rLLM:
  verl: freeze + replay → Megatron/vLLM引擎 → 分布式 → 多GPU
  rLLM TinkerBackend: in-process → 单GPU → 无freeze → 但GRPO足够稳定!

★ RTX 4090 MoE RL训练:
  → 只能rule-based reward → 无GPU空间给RM
  → LoRA + HYBRID + naive → actor+ref+rollout同一进程 → 1GPU
  → EP=1 → 无分布式 → 最简单最稳定!
  → ★ Qwen3-MoE(A0.6B+B4B) + LoRA → RTX 4090唯一MoE RL可行方案!
```

---

## 5. MoE + CUDA Graph 交互

### 5.1 核心矛盾: 动态路由 vs 固定shape

```
CUDA Graph核心约束:
  → 捕获时所有tensor有固定shape → replay时shape必须相同!
  → MoE路由 → top-k indices → 每次不同 → 不同expert不同token数!
  → → EP dispatch → 不同rank不同token数 → shape不固定 → graph不兼容!

★ 这是MoE+CUDA Graph最根本的矛盾!
  → 路由决策 → 数据依赖 → 每次不同 → shape变化 → graph break!
  → 但decode → 1 token per request → 路由决策小 → 可以graph!
```

### 5.2 Megatron 推理 CUDA Graph — 双路径设计

```
★ ★ NCCLAllGatherDispatcher 两种模式:

CG path (decode专用, _use_allgather_v=False):
  → 所有EP rank贡献等量tokens → 标准AllGather → 固定shape!
  → _valid_tokens_tensor.fill_(ep_size * local_tokens) → 单次in-place fill!
  → ★ ★ .fill_() → CUDA graph安全 → 无host sync → 无CPU round-trip!
  → ★ 但限制: 所有rank必须等量tokens → 不均衡时padding浪费!

Non-CG path (prefill, _use_allgather_v=True):
  → 各rank不同token数量 → pad到max → AllGather → compact去padding
  → all-gather counts → .tolist() → host sync → CPU round-trip!
  → ★ 绝不在graph capture下运行 → prefill不能用graph!

★ ★ NVLSAllGatherVDispatcher (Hopper+):
  → variable-count AllGather → 不需等量 → 更灵活!
  → fused_metadata_update Triton kernel → 5操作融合 → 无host sync!
  → multimem_all_gatherv_3tensor → 3tensor同时gather → 极快!
  → ★ decode+prefill都能graph → 不需要CG/Non-CG双路径!

★ ★ adjust_batch_dims_for_expert_parallelism():
  → NCCL dispatcher: all-reduce-max token count across EP
  → 如果任何rank有prefill → 所有rank fallback eager → return None
  → ZMQ CPU-only sync → 避免per-step NCCL AllReduce → 极快!
```

### 5.3 InferenceTopKRouter @torch.compile + CUDA Graph

```
★ ★ 推理路由的2大优化:
  @torch.compile → 消除Python overhead → 纯kernel → 极快
  dense_output → [num_tokens, topk] → FlashInfer grouped GEMM直接用

★ CUDA graph兼容:
  → 固定shape → [num_tokens, topk] → 不变 → graph capture安全!
  → vs 训练TopKRouter: sparse [num_tokens, num_experts] → 大部分零 → 不高效
  → dense format → graph友好 → FlashInfer友好 → 双赢!

★ Megatron GRPO cudagraph memory regression (PR #5280):
  → exponential distribution + mixed-prefill grid → +13.6% peak memory!
  → Fix: linear sizing → [1,2,3,...] vs exponential [1,2,4,8,...]
  → ★ ★ GRPO → linear sizing → 省内存 → RTX 4090适用!
  → ★ Router replay recording during warmup → 记录路由 → graph capture!
```

### 5.4 vLLM CUDA Graph + MoE

```
★ vLLM CUDA Graph模式:
  FULL / PIECEWISE / FULL_DECODE_ONLY / FULL_AND_PIECEWISE(默认)

MoE + CUDA Graph交互:
  → EP enabled → AllToAll → 通信 → graph break?
  → ★ vLLM MoE在FULL_DECODE_ONLY下可能graph → 但EP通信是问题!

★ Breakable CUDA Graph (BCG):
  → @eager_break_during_capture → 打断graph → eager → 继续
  → DS-V4自动启用BCG → 移除torch.compile依赖
  → ★ MoE routing是data-dependent → BCG允许部分eager → 部分graph!

★ RTX 4090 MoE + CUDA Graph:
  → EP=1 → 无AllToAll → 无通信 → MoE全是计算 → graph完全可行!
  → InferenceTopKRouter → dense output → graph友好 → RTX 4090✓
  → ★ RTX 4090: EP=1 → MoE kernel在graph内 → 无通信break → 完美!
```

### 5.5 MoE CUDA Graph交互模式对比

| 框架 | Decode Graph | Prefill Graph | EP通信 | Router在Graph | RTX 4090 |
|------|-------------|---------------|--------|--------------|----------|
| Megatron NCCL | yes(.fill_) | no(host sync) | 等量AllGather | yes(@compile) | ✓ EP=1 |
| Megatron NVLS | yes(AGV) | yes(AGV) | variable AGV | yes(@compile) | ✗ SM90 |
| vLLM FULL | yes | no | 无(EP=1) | yes | ✓ |
| vLLM BCG | yes(break) | yes(break) | break | yes | ✓ |
| SGLang | yes | no | 无(EP=1) | yes | ✓ |

```
★ ★ RTX 4090最优: EP=1 → decode CUDA graph → MoE kernel全在graph内
   → 无EP通信 → 无shape变化 → graph完美 → 性能最优!

★ H100集群最优: NVLS AGV/RSV → decode+prefill都能graph → variable tokens
   → fused Triton metadata → 3tensor → 最快 → 但需SM90!
```

---

## 6. MoE Memory Estimation

### 6.1 MoE vs Dense: KV Cache差异

```
★ ★ MoE KV Cache = Dense KV Cache → MoE路由不影响KV!
  → KV Cache: per-head per-token → attention状态 → 与expert无关
  → MoE Routing: per-token → expert MLP选择 → 与KV无关
  → → KV Cache大小 = Dense模型相同! → MoE不增加KV Cache!

但MoE模型通常有更多KV heads (GQA vs MHA):
  → Dense Llama-3: 32 KV heads (GQA-8) → KV=32*128*2*seq_len
  → MoE DeepSeek-V3: MLA → kv_reduced_dim → KV极小!
  → MoE Qwen2-MoE: GQA → KV类似dense → 但模型更大 → KV更大!

★ ★ MLA (DeepSeek-V2/V3/V4) → KV Cache最小化:
  → kv_lora_rank + qk_pos_emb_head_dim → KV压缩维度!
  → KV Cache: dtype_size × num_layers × block_size × kv_reduced_dim
  → vs Dense: dtype_size × 2(K+V) × num_layers × block_size × heads × head_dim
  → MLA: kv_reduced_dim=512 vs Dense: heads×head_dim=32×128=4096 → 8x reduction!

★ DeepSeek-V4 MLA Sparse:
  → sliding-window sparse attention → 只看部分KV → KV Cache更小!
  → FlashMLA_SPARSE → bf16/fp8_ds_mla → head_dim=576 → 特殊格式!
```

### 6.2 MoE Expert参数内存

```
★ ★ MoE参数 = router权重 + expert权重 + shared expert权重

router权重: [hidden_dim, num_experts] → 很小!
  → 7B MoE: 4096 × 8 = 32KB → 可忽略!

expert权重: num_experts × [hidden_dim, intermediate_size] × 3(w1/w2/w3)
  → Mixtral 8x7B: 8 × 3 × 4096 × 14336 = ~1.4GB per layer
  → DeepSeek-V3: 256 routed × 3 × 7168 × 18432 = ~100GB+ → EP必需!
  → Qwen3-MoE(A0.6B+B4B): 64 routed × 3 × hidden × inter = ~8GB → 单GPU可行!

shared expert权重: 1 × [hidden_dim, intermediate_size] × 3
  → DeepSeek-V3: 1 × 3 × 7168 × 18432 = ~0.5GB → 较小

★ ★ MoE总参数 vs Dense:
  Dense 7B: ~14GB (BF16)
  Mixtral 8x7B: ~47GB (BF16) → 8 experts → 3.4x
  DeepSeek-V3: ~671GB (BF16) → 256+1 experts → 需要EP!
  Qwen3-MoE(A0.6B+B4B): ~8GB → RTX 4090可以!
```

### 6.3 INT4 Quantization对MoE的影响

```
★ ★ INT4量化 → expert权重压缩4x → MoE推理关键优化!

Dense 7B INT4: ~3.5GB → RTX 4090 ✓
MoE Mixtral 8x7B INT4: ~12GB → RTX 4090 可以!(但KV cache大)
MoE Qwen2-MoE INT4: ~2-4GB → RTX 4090 ✓ ✓
MoE DeepSeek-V3 INT4: ~170GB → RTX 4090 ✗ (太大)

★ ★ vLLM INT4 Triton fallback (PR #43731):
  → TritonW4A16LinearKernel → SM89可用 → RTX 4090✓
  → 之前: intermediate_size≠128对齐 → ValueError → 无法加载!
  → 现在: Triton fallback → 可以运行 → 比Marlin慢但可行!
  → ★ ★ 这是RTX 4090最有价值的MoE特性 → 更多INT4 MoE模型可用!

★ INT4 expert kernel选择 (vLLM):
  Marlin → SM80+ → 128对齐 → 最快 → RTX 4090✓
  Triton → SM89+ → 无对齐要求 → fallback → 比Marlin慢2-5x
  CUTLASS → SM90 → Hopper → RTX 4090✗
  DeepGEMM → grouped-GEMM → 新!

★ ★ MoE INT4推理最优路径 (RTX 4090):
  → Marlin-aligned层 → Marlin kernel → 最快
  → Marlin-unaligned层 → Triton fallback → 可运行
  → INT8 KV cache → KV压缩2x → 更多KV blocks
  → → RTX 4090: INT4+INT8KV → 4,791 tok/s (7B) → 已验证!
```

### 6.4 MoE Memory估算公式

```
★ ★ RTX 4090 (24GB) MoE内存估算:

Model weights (BF16):
  router: hidden × num_experts × 2bytes → <1MB → 可忽略
  experts: num_experts × 3 × hidden × intermediate × 2bytes
  shared_expert: 3 × hidden × shared_intermediate × 2bytes

Model weights (INT4):
  experts: num_experts × 3 × hidden × intermediate × 0.5bytes
  → 4x压缩 → INT4 MoE推理首选!

KV Cache:
  Standard: dtype × 2(K+V) × num_layers × block_size × num_kv_heads × head_dim
  MLA: dtype × num_layers × block_size × kv_reduced_dim
  INT8 KV: dtype=1 → 2x压缩

★ ★ 7B MoE (Qwen2-MoE style, 64 experts, ~4B active):
  BF16 weights: ~8GB
  INT4 weights: ~2GB
  KV Cache (INT8, 32 heads, 128 dim, 128 blocks): ~3GB
  CUDA graph pool: ~0.5GB
  Total (INT4+INT8KV): ~5.5GB → 24GB绰绰有余!

★ ★ Mixtral 8x7B INT4+INT8KV:
  INT4 weights: ~12GB
  KV Cache (INT8, 8 KV heads GQA): ~2GB (GQA节省!)
  CUDA graph pool: ~1GB
  Total: ~15GB → 24GB可行! 但注意KV block数量!

★ ★ DeepSeek-V3/V4: 太大 → RTX 4090完全不可能 → 需H100集群!
```

### 6.5 MoE vs Dense 内存对比总结

```
| 模型 | BF16 | INT4 | INT4+INT8KV | RTX 4090 |
  |------|------|------|-------------|----------|
  | Dense 7B | ~14GB | ~3.5GB | ~5GB (weights+KV) | ✓ |
  | Qwen2-MoE(~4B active) | ~8GB | ~2GB | ~5.5GB | ✓ ✓ |
  | Mixtral 8x7B | ~47GB | ~12GB | ~15GB | ✓(但KV紧张) |
  | DeepSeek-V3 | ~671GB | ~170GB | ~175GB | ✗ |
  | DeepSeek-V4-Pro | ~700GB+ | ~175GB+ | ~180GB+ | ✗ |

★ ★ RTX 4090 MoE可行模型:
  → Qwen2-MoE/Qwen3-MoE → ~4-8B active → INT4 → ✓ ✓
  → Mixtral 8x7B → INT4+GQA → ✓ (但KV有限)
  → DeepSeek-V2-Lite → INT4 → ✓ (如果存在lite版本)

★ ★ RTX 4090 MoE不可行模型:
  → DeepSeek-V3/V4 → 太大 → 需H100集群
  → 大MoE(>256 experts) → 参数量太大 → 单GPU不可能
```

---

## 7. 跨框架MoE架构全景对比

### 7.1 核心对比表

| 维度 | vLLM | SGLang | Megatron | verl | DeepSpeed |
|------|------|---------|----------|------|-----------|
| **Router类型** | 11种method factory | Multi-path(flashinfer/triton/aiter/kimi_k2) | TopK+InferenceTopK(双路由) | 委托+freeze+replay | AutoEP(TorchTitan) |
| **EP Backend** | 8种AllToAll | 6种Dispatcher | 3种+DeepEP/HybridEP/NVLS | 委托(ep_size/EP_MPU) | AutoEP(NCCL symmetric) |
| **EPLB** | redundant_experts+运行时 | logical→physical+DeepEP协同 | 无(aux_loss训练均衡) | 委托(vLLM EPLB) | 无(训练aux_loss) |
| **KV+MoE** | RoutedExpertsCapturer+slot_mapping | RoutedExpertsCapturer+DeepEP AG | 无直接交互 | 委托vLLM | 无(训练不需要) |
| **量化** | 8格式+12+kernel | 10+scheme+10+runner | MXFP8(推理only) | FP8(nemo/vLLM) | BF16(训练only) |
| **推理Dispatcher** | naive_dp/no_dp | standard | NCCL/NVLS(Hopper+) | 委托 | 无(TRT-LLM/vLLM) |
| **训练Dispatcher** | 无(推理only) | 无(推理only) | AllGather/AlltoAll/Flex | 委托Megatron | AutoEP(NCCL) |
| **Shared Expert** | 配置支持 | num_fused_shared_experts | Overlap(NVLS独有) | 委托 | AutoEP支持 |
| **DeepEP集成** | ht/ll backend | DeepEPMoE层 | _DeepepManager | 委托 | 无(asymmetric) |
| **NVLS集成** | 无 | 无 | inference dispatcher(Hopper+) | 委托 | 无 |
| **CUDA Graph** | FULL/PIECEWISE/BCG | decode-only | decode(.fill_()/NVLS AGV) | 委托 | 无 |
| **RL训练** | RoutedExpertsCapturer | RoutedExpertsCapturer | router_replay(GRPO) | freeze+replay | 无(RL不支持) |

### 7.2 框架定位差异

```
★ ★ 推理框架(vLLM/SGLang) vs 训练框架(Megatron/verl/DeepSpeed):
  → MoE实现截然不同!

推理框架:
  → 融合kernel+量化+AllToAll backend → 最小延迟
  → EPLB → 动态负载均衡 → 生产推理必须!
  → RoutedExpertsCapturer → RL推理桥梁 → 独有特性!
  → CUDA graph → decode加速 → 通信shape固定

训练框架:
  → 通信优化+负载均衡+梯度同步 → 最大吞吐
  → aux_loss → 训练时均衡 → 推理时不需要!
  → verl freeze+replay → RL训练稳定 → 独有特性!
  → Megatron DeepEP → asymmetric → H100集群必须!

★ ★ verl独特: 委托架构 → 同一框架支持推理+训练 → 但不自己实现!
  → 推理: vLLM/SGLang → RoutedExpertsCapturer → RL推理桥梁
  → 训练: Megatron/Automodel → freeze+replay → RL训练稳定
  → ★ 唯一跨推理+训练的框架 → 但每个方向都不是最优!
```

---

## 8. RTX 4090 MoE 综合分析

### 8.1 RTX 4090 MoE限制汇总

```
★ ★ RTX 4090 MoE限制 (SM 8.9, 24GB, PCIe only):

通信限制:
  ✗ DeepEP → SM90必需 → asymmetric不可用
  ✗ HybridEP → TMA指令 → SM90必需
  ✗ NVLS → Hopper+NVLink+symmetric memory → 全不满足
  ✗ FlashInfer NVLink AllToAll → 需NVLink → 只有PCIe
  ✗ NIXL RDMA → 需RDMA → 无RDMA支持
  ✗ FP8 dispatch → DeepEP FP8 → SM90必需

计算限制:
  ✓ INT4 Marlin → SM80+ → 128对齐 → ✓
  ✓ INT4 Triton fallback → SM89+ → 无对齐 → ✓ (v0.23新!)
  ✓ FlashInfer attention → SM89 → ✓
  ✓ CUDA graph → SM89 → ✓
  ✓ @torch.compile → 纯软件 → ✓
  ✗ FP8 E5M2 → SM89不支持 → E4M3 only
  ✗ NVLS fused kernel → SM90必需

内存限制:
  → 24GB → 小MoE(~4-8B active)可行 → 大MoE(>256 experts)不行
  → INT4量化 → 4x压缩 → 关键优化!
  → INT8 KV → 2x压缩 → 更多KV blocks!
```

### 8.2 RTX 4090 MoE最优配置

```
★ ★ ★ RTX 4090 MoE推理最优配置:

模型选择:
  → Qwen2-MoE/Qwen3-MoE (~4B active, 64 experts) → 最适合!
  → Mixtral 8x7B → INT4勉强可行 → KV cache紧张
  → DeepSeek-V2-Lite → 如果存在 → INT4可行

框架选择:
  → vLLM → INT4+INT8KV → 最成熟 → 4,791 tok/s (dense 7B baseline)
  → SGLang → DeepEPMoE → 但EP=1 → 无DeepEP优势 → 用标准FusedMoE
  → Megatron → DynamicInferenceEngine + NCCL → 但INT4不如vLLM

并行选择:
  → EP=1 → 单GPU → 无分布式 → 最简单最稳定!
  → TP=1 → expert不分片 → 无TP通信 → 最快!
  → ★ EP和TP都=1 → 全local → 无通信 → RTX 4090最优!

量化选择:
  → INT4 weights → Marlin kernel (128对齐层) + Triton fallback (不对齐层)
  → INT8 KV cache → 2x压缩 → 更多KV blocks
  → ★ ★ INT4+INT8KV → RTX 4090 MoE推理唯一出路!

CUDA Graph:
  → EP=1 → 无AllToAll → decode graph完全可行!
  → InferenceTopKRouter → dense output → graph友好
  → ★ ★ RTX 4090 MoE decode → CUDA graph → 最优!

★ ★ ★ RTX 4090 MoE RL训练最优配置:

模型选择:
  → Qwen3-MoE(A0.6B+B4B) → ~8GB BF16 → LoRA → ~9GB → 24GB可行!

框架选择:
  → verl → HYBRID mode → actor+ref+rollout同一进程 → 1GPU
  → rLLM TinkerBackend → in-process → 单GPU最快路径

RL配置:
  → GRPO → 无critic → 省50% compute+memory!
  → LoRA rank=32-64 → ref_in_actor=True → 无需额外ref GPU!
  → freeze_moe_router=True → 路由固定 → 稳定RL训练!
  → rule-based reward → 无需GPU → CPU计算 → 最轻量!
  → naive weight sync → Python generator zero-copy → 不需要NCCL/IPC

★ ★ RTX 4090唯一MoE RL可行方案:
  → Qwen3-MoE + GRPO + LoRA + freeze_moe_router + HYBRID + naive!
```

### 8.3 RTX 4090 vs H100集群对比

```
| 维度 | RTX 4090 (SM89, PCIe, 24GB) | H100集群 (SM90, NVLink+RDMA, 80GB) |
  |------|----------------------------|-------------------------------------|
  | MoE推理 | EP=1, INT4+INT8KV | EP≥8, DeepEP, FP8 dispatch |
  | EP通信 | NCCL(唯一) → padding浪费 | DeepEP → asymmetric → 4.6x |
  | EPLB | 不需要(EP=1) | 必须(redundant_experts) |
  | CUDA Graph | EP=1 → 完全可行 | NVLS AGV → variable tokens |
  | MoE RL训练 | GRPO+LoRA+freeze, 1GPU | GRPO+DeepEP, 多GPU EP |
  | 适用模型 | 小MoE(~4-8B active) | 大MoE(DeepSeek-V3/V4, 256+experts) |
  | 最大瓶颈 | 内存(24GB) | EP通信延迟 |

★ ★ 核心差异:
  → RTX 4090: 受限于内存 → 只能小MoE → EP=1 → 通信=0 → 最简单
  → H100集群: 受限于通信 → EP通信延迟 → DeepEP解决 → 但需要SM90!
  → → 不同GPU → 不同瓶颈 → 不同最优策略!
```

---

## 9. 关键设计洞察

```
1. ★ ★ 推理vs训练 → MoE实现截然不同 → 不可混用!
   → 推理: 融合kernel+量化+EPLB+RoutedExpertsCapturer
   → 训练: aux_loss均衡+DeepEP+梯度同步+freeze/replay
   → verl跨两者 → 但委托 → 不是最优 → 是方便!

2. ★ ★ Asymmetric vs Symmetric → DeepEP vs NCCL → 架构级差异!
   → NCCL: symmetric → padding浪费 → 通用但慢 → 4.6x差距!
   → DeepEP: asymmetric → 无padding → MoE天然模式 → 必需SM90!
   → → 这不是kernel优化 → 是架构级优势 → NCCL无法通过优化消除!
   → RTX 4090: 只能用NCCL → asymmetric优势无法获得 → H100必需!

3. ★ ★ RoutedExpertsCapturer → RL训练+推理桥梁 → 但与PD分离不兼容!
   → GRPO/PPO需要知道expert选择 → capture → slot_mapping索引
   → PD分离 → slot_mapping变化 → 索引失效 → 不兼容!
   → ★ 这是2026年MoE RL训练的关键矛盾 → 需要新机制解决!

4. ★ ★ freeze_moe_router → MoE RL训练最关键配置!
   → 路由不稳定 → expert drift → 训练崩溃 → freeze解决!
   → Qwen MoE默认freeze → 生产级配置!
   → router_replay → 双保险 → rollout和训练路由一致!

5. ★ ★ NVLS → Megatron独有 → GB200推理优化 → 未来方向!
   → AllGather-V → variable token count → decode+prefill都graph
   → 3tensor同时gather → fused Triton metadata → 极快
   → RTX 4090: NVLS✗ → NCCL次优但可用 → H100: NVLS最优!

6. ★ ★ MoE+CUDA Graph → decode可行 → prefill难 → EP通信是关键!
   → decode: 等量tokens → 固定shape → graph可行
   → prefill: 不同token数 → shape变化 → host sync → graph不可行
   → NVLS: variable-count → 无host sync → decode+prefill都graph → SM90必需!
   → RTX 4090: EP=1 → 无通信 → MoE全在graph → 完美!

7. ★ ★ INT4 Triton fallback → RTX 4090最重要MoE特性!
   → 之前: shape不对齐 → ValueError → 无法加载MoE模型!
   → 现在: Triton fallback → 可以运行 → 更多MoE模型可用!
   → 性能: 比Marlin慢 → 但从"完全无法运行"到"可以运行" → 净影响正!

8. ★ ★ MoE Memory → KV Cache不增加 → Expert参数增加 → INT4唯一出路!
   → KV Cache = Dense相同 → MoE路由不影响KV!
   → Expert参数 → num_experts × hidden × intermediate × 3 → 大!
   → INT4 → 4x压缩 → expert参数可接受 → RTX 4090可行!
   → MLA → KV Cache极小 → DeepSeek独有 → 8x KV reduction!

9. ★ ★ verl委托架构 → 灵活但不最优 → RTX 4090方便选择!
   → 推理委托vLLM → 训练委托Megatron → 不自己实现 → 随引擎升级!
   → freeze+replay → verl独有 → RL训练稳定保障!
   → RTX 4090: verl HYBRID → 单GPU → actor+ref+rollout同一进程 → 方便!

10. ★ ★ DeepSpeed AutoEP → 零代码修改 → HF兼容 → 但无asymmetric!
    → TorchTitan内核 → grouped-GEMM → 单次kernel → 简单快速!
    → vs Megatron DeepEP: 0代码 vs 4.6x通信 → 取舍!
    → RTX 4090: AutoEP ZeRO-2+LoRA → 唯一DeepSpeed MoE训练路径!
```

---

## 10. MoE Serving决策树

```
★ ★ ★ MoE Serving决策树 (基于GPU和模型大小):

Q1: 模型大小?
  <10B active → 小MoE → 单GPU推理可行 → 继续
  10-100B active → 中MoE → 需要多GPU → 继续
  >100B active → 大MoE → 需要H100集群 → DeepEP必需!

Q2: GPU类型?
  RTX 4090 (SM89, PCIe) → EP=1 → INT4+INT8KV → vLLM/SGLang
  A100 (SM80, NVLink) → EP≤8 → NCCL → vLLM/SGLang/Megatron
  H100 (SM90, NVLink+RDMA) → DeepEP → Megatron FlexDispatcher → 最优!

Q3: 场景?
  纯推理 → vLLM/SGLang → INT4+INT8KV+EPLB+CUDA graph
  RL训练 → verl/rLLM → GRPO+LoRA+freeze_moe_router
  训练 → Megatron/DeepSpeed → aux_loss+DeepEP/AutoEP

Q4: RTX 4090具体配置:
  推理: vLLM INT4+INT8KV+EP=1+prefix caching+CUDA graph
  RL训练: verl GRPO+LoRA+freeze+HYBRID+naive+rule-based reward
  研究实验: Qwen3-MoE(A0.6B+B4B) → RTX 4090唯一MoE RL可行模型!

★ ★ ★ 最优路径:
  RTX 4090小MoE推理: vLLM INT4 → 4,791 tok/s → 已验证!
  RTX 4090小MoE RL: verl GRPO+LoRA+freeze → 唯一可行!
  H100大MoE推理: Megatron DeepEP+NVLS → 4.6x通信 → 最优!
  H100大MoE RL: verl GRPO+DeepEP+EP≥8 → 多GPU分布式!
```

---

## Sources

- notebook/projects/deepep-source-reading.md — DeepEP V1架构+asymmetric核心+5个优势根源
- notebook/projects/deepep-v2-reading.md — DeepEP V2 ElasticBuffer+Gin backend+Handle Caching+通信overlap
- notebook/projects/deepep-megatron-integration-latest.md — Megatron双路径(DeepEP V1+HybridEP)+4.6x benchmark+社区fork
- notebook/fundamentals/moe-serving-framework-comparison.md — 7框架MoE对比+RoutedExpertsCapturer+KV+MoE交互
- notebook/projects/vllm-v0.23-new-features-reading.md — MRv2 MoE扩展+INT4 Triton fallback+DS-V4 EPLB+BCG
- notebook/projects/megatron-inference-engine-reading.md — DynamicInferenceEngine+NCCL/NVLS dispatcher+InferenceTopKRouter+GRPO cudagraph regression
- notebook/projects/megatron-moe-architecture.md — MoE Layer+Router+Dispatcher+Shared Expert
- notebook/projects/vllm-moe-serving-reading.md — FusedMoE模块化架构+Router→Prepare→Experts→Finalize+8 AllToAll backend
- notebook/projects/vllm-cuda-graph-reading.md — CUDA Graph模式+Breakable CG+GRPO regression+sleep/wake+pool
- notebook/projects/verl-grpo-data-flow-reading.md — GRPO advantage+need_critic+LoRA+reward+RTX 4090可行配置
- notebook/projects/deepspeed-latest-developments-2026-06.md — AutoEP+TorchTitan内核+grouped-GEMM+ZeRO-0/1/2
