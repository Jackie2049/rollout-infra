# 7框架生产集成深度阅读 — RL训练Pipeline跨框架组合

> 2026-06-16 | 综合7框架源码+文档+实测 → 生产RL训练Pipeline如何组合7框架
> ★★★★★ 核心: verl=RL中枢 → 连接vLLM/SGLang/Megatron/FSDP2 → rLLM=RTX 4090最优 → HF=桥梁
> ★★★★★ 7个关键集成点: verl+vLLM / verl+SGLang / verl+Megatron / rLLM Tinker / Megatron→TRT-LLM / DeepSpeed ZeRO / MindIE/vLLM-Ascend

---

## 目录

1. [verl + vLLM: 3种集成模式](#1-verl--vllm-3种集成模式)
2. [verl + SGLang: PD disaggregation](#2-verl--sglang-pd-disaggregation)
3. [verl + Megatron: 训练后端](#3-verl--megatron-训练后端)
4. [rLLM Tinker: In-process零拷贝](#4-rllm-tinker-in-process零拷贝)
5. [Megatron → TensorRT-LLM: 推理部署桥梁](#5-megatron--tensorrt-llm-推理部署桥梁)
6. [DeepSpeed ZeRO: 训练辅助+MoE局限](#6-deepspeed-zero-训练辅助moe局限)
7. [MindIE/vLLM-Ascend: Ascend推理替代](#7-mindievllm-ascend-ascend推理替代)
8. [7框架完整集成拓扑](#8-7框架完整集成拓扑)
9. [RTX 4090生产部署推荐](#9-rtx-4090生产部署推荐)
10. [关键洞察总结](#10-关键洞察总结)

---

## 1. verl + vLLM: 3种集成模式

### 1.1 三种RolloutMode架构

★★★★★ verl + vLLM是当前最成熟的RL训练集成路径。verl定义了3种RolloutMode，控制训练和推理引擎的GPU资源分配方式：

```
★★★★★ verl 3种RolloutMode + vLLM集成:

┌─────────────────────────────────────────────────────────────────┐
│ RolloutMode.HYBRID (★★★★★ RTX 4090最优)                        │
│  → 训练+推理在同一进程(same process) → naive generator            │
│  → ActorRolloutRefWorker: actor+rollout+ref 三合一                │
│  → Weight Sync: naive(零拷贝) → 参数更新立即可见 → 0ms            │
│  → Sleep/Wake: sleep(level=2) → 释放权重+KV → 训练空间 →          │
│                wake_up → 加载新权重+KV → 推理空间                   │
│  → ★★★★★ RTX 4090 GRPO生产路径: HYBRID+naive = 零拷贝!          │
│                                                                   │
│ RolloutMode.COLOCATED (★★★ 多GPU)                                │
│  → 同PlacementGroup但不同进程 → Ray actor隔离                     │
│  → ServerAdapter → vLLMHttpServer → AsyncLLM → HTTP/ZMQ IPC      │
│  → Weight Sync: BucketedWeightSender → ZMQ IPC → ~50-100ms       │
│  → Sleep/Wake: sleep(level=1) → 只释放推理引擎 → wake恢复         │
│  → ★ 适合同GPU但需要进程隔离 → GRM(LLM judge)场景                 │
│                                                                   │
│ RolloutMode.STANDALONE (★★ 大规模集群)                            │
│  → 推理独占GPU → 不同PlacementGroup → NCCL/NIXL通信               │
│  → Weight Sync: NCCL broadcast(~600ms PCIe) / NIXL RDMA(~10ms)   │
│  → ★★★ 无sleep/wake → 推理独立 → off-policy场景                  │
│  → ✗✗✗ RTX 4090: PCIe带宽瓶颈 → NCCL灾难 → 不适用!              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Online Rollout — 实时推理+训练循环

★★★★★ Online Rollout = verl标准RL训练模式 → rollout和训练在同一循环：

```
★★★★★ Online Rollout完整数据流 (verl HYBRID):

  Prompt × rollout_n → vLLM generate → [n responses per prompt]
    → extract_reward() → rule-based / RM → per-response reward
    → compute_old_log_prob() → actor forward → log π_θ(y|x)
    → compute_ref_log_prob() → ref forward → log π_ref(y|x)
      → ★ bypass_mode: pi_old = rollout logprobs → 省1个forward!
    → apply_kl_penalty() → r_i - β * KL
    → compute_advantage(GRPO) → (r_i - μ_group) / σ_group
    → update_actor() → PPO clip loss → LoRA update
    → update_weights() → sync π_θ → rollout engine

  ★★★★★ vLLM解决了Python rollout瓶颈:
    Python rollout: 逐token串行 × max_len → 慢
    vLLM rollout: continuous batching × 一次性 → 快2-5x!
    + prefix caching: 同prompt的n个request → 共享KV → 58%省!

  ★★★★★ Sleep/Wake机制 — GPU时分复用:
    1. rollout.sleep(level=2) → offload权重+KV → GPU空出给FSDP训练
    2. 训练中 → FSDP训练 → 占用全部GPU → rollout睡眠
    3. rollout.wake_up() → 加载新权重+KV → 准备推理
    4. 推理 → vLLM generate → 占用GPU → 训练暂停
    → ★★★ 同GPU → 训练时训练用 → 推理时推理用 → 省50%GPU!
```

### 1.3 Offline Rollout — 预生成+离线训练

★★★ Offline Rollout = 推理和训练解耦 → 先生成后训练：

```
★★★ Offline Rollout模式:

  Phase 1: 预生成 (STANDALONE模式)
    → vLLM独立服务 → 批量生成responses → 存储到磁盘
    → Prompt dataset × rollout_n → 大批量推理 → 巨量数据
    → ★ vLLM可以持续运行 → 不需要sleep/wake → 最大化吞吐!

  Phase 2: 离线训练
    → 加载预生成的数据 → 训练循环 → 无需rollout
    → ★★★ 适合: 大规模数据收集 → 小规模训练 → 训练GPU数量<<推理GPU
    → ★★★ DAPO(Dynamic Advantage Policy Optimization) → prompt过滤 → 利用off-policy数据
    → ✗ 限制: stale policy → off-policy数据 → importance sampling修正

  ★★★ 适用场景:
    → 推理集群>>训练集群 → 独立GPU → 最大化推理吞吐
    → 数据收集+过滤 → DAPO → 先收集后过滤后训练
    → 不需要实时反馈 → batch generation → 更高效
```

### 1.4 PD Disaggregation — Prefill/Decode分离

★★★★ PD Disaggregation = 将prefill(compute-bound)和decode(memory-bound)分离到不同GPU组：

```
★★★★★ vLLM PD Disaggregation架构 (verl集成路径):

  Client → verl Proxy → Prefill Instance (N GPU) ──KV Transfer──→ Decode Instance (M GPU)
                    │                                              │
                    └── 计算 prompt KV cache                      └── 持续生成 tokens
                    └── KV Cache 发送给 Decode 节点              └── 只需 model weights + KV

  ★★★★ KV Connector实现:
    → NixlConnector: NIXL DMA + ZMQ side channel → 高性能RDMA
    → FlexKVConnector: 灵活缓存策略
    → MooncakeConnector: 分布式KV缓存(月之暗面Kimi)
    → LMCacheConnector: 结合prefix caching

  ★★★★ 1P2D配置(推荐):
    → Prefill: 1 GPU(compute-bound → H100/B200最优)
    → Decode: 2 GPU(TP=2, memory-bound → 便宜GPU可用)
    → ★ 独立扩展P和D → 最优资源利用率 → 85-95% GPU利用率
    → ★ 异构TP: P/D可使用不同TP → 灵活适配硬件

  ★★★★ 与verl GRPO训练的关系:
    → GRPO rollout_n=8 → 同prompt的8个request → prefill只算1次 → 7x省!
    → Prefill节点 → compute-heavy → 短时间大批量KV → transfer到decode
    → Decode节点 → 持续低延迟 → 训练期间不需要 → ★ 训练时prefill GPU可释放!

  ★★★ RTX 4090: PD Disaggregation不适用 → 需要多GPU+RDMA → PCIe瓶颈!
```

### 1.5 Weight Sync — 权重同步机制对比

★★★★★ Weight Sync是verl+vLLM集成的核心瓶颈：

```
★★★★★ 7种Weight Sync机制完整对比 (7B模型):

| 机制 | 耗时 | 传输量 | RTX4090? | 适用场景 |
|------|------|--------|----------|---------|
| ★★★★★ naive(HYBRID) | ~0ms | ~0 bytes | ★★★✓✓ | 同进程 → 参数直接可见 |
| ★★★★★ GPU-only merge(rLLM) | <1ms | ~2.6GB(LoRA) | ★★★✓ | LoRA merge → new SamplingClient |
| ★★★ sleep/wake(vLLM) | ~300ms | 0(offload) | ✓✓ | vLLM colocated → 释放→恢复 |
| ★★ CUDA IPC(ZMQ) | ~50ms | ~14GB | ✗(PCIe) | verl COLOCATED → 跨进程 |
| ✗ NCCL broadcast | ~600ms | ~14GB | ✗✗(PCIe) | verl STANDALONE → 跨GPU |
| ✗ NIXL RDMA | ~10ms | ~14GB | ✗(需RDMA) | H100集群 → 双buffer零拷贝 |
| ★ ZeRO AllGather | ~800ms | ~14GB | ✗✗ | DeepSpeed → 分区→gather→replace |

★★★★★ RTX 4090最优: naive(HYBRID) 或 GPU-only merge(rLLM)
  → 同进程→零拷贝→无通信→最快 → 单GPU唯一可行路径!

★★★★ LoRA对Weight Sync的影响:
  → 只sync adapter(~2.6GB) vs 全模型(~14GB) → ★ 5.4x减少!
  → rLLM Tinker: LoRA auto-init → save_weights → 零拷贝 → 最简!
  → verl HYBRID: lora_as_adapter=True → sleep(level=1) → 只释放adapter → faster wake-up!
```

---

## 2. verl + SGLang: PD Disaggregation

### 2.1 SGLang作为Rollout引擎

★★★★ verl v0.8.0新增SGLang rollout engine option → 第二种推理后端：

```
★★★★ verl + SGLang集成架构:

  verl Driver → LlmServerManager → SGLang async server
    → async_sglang_server.py → TokenizerManager → Scheduler → Mixin事件循环
    → ★★★ SGLang RadixAttention → prefix复用 → GRPO rollout省7x prefill!

  ★★★★★ SGLang核心优势 = RadixAttention:
    → 基数树(radix tree) → prefix KV cache自动复用 → 与vLLM BlockHash不同
    → GRPO rollout_n=8 → system prompt KV只1次 → 7x省prefill compute!
    → vs vLLM prefix caching: BlockHash → block-level → 可能碎片化
    → ★★★ SGLang的prefix caching更灵活 → radix tree → 精确匹配 → 更少浪费!

  ★★★ Sleep/Wake (async_sglang_server.py):
    → LoRA: sleep只释放KV cache → 保留base weights → wake只需sync adapter delta
    → Full weights: sleep释放weights+KV → wake需要重新加载 → ~300ms
    → ★★★ LoRA+sleep = 极快wake → 只sync adapter(~2.6GB vs ~14GB) → 5.4x快!
```

### 2.2 PD Disaggregation with SGLang

★★★★ SGLang原生支持PD disaggregation → verl可利用SGLang作为PD分离后端：

```
★★★★★ verl + SGLang PD Disaggregation:

  SGLang Disaggregated Serving模式:
    → Prefill节点: TokenizerManager → Scheduler → compute prompt KV
    → Decode节点: 接收KV → 持续decode → 低延迟
    → ★★★★ SGLang的PD分离更完整 → 内置KV transfer → 不需要额外connector!
    → vs vLLM: 需要NixlConnector/MooncakeConnector → 外部依赖更多

  ★★★★ verl GRPO rollout with SGLang PD:
    → rollout_n=8 → 同prompt → sticky session → 同server → RadixAttention自动生效
    → Prefill节点 → 计算prompt KV → 传给decode → ★ 只1次prefill → 7x省!
    → Decode节点 → 8个response并行decode → ★ RadixAttention+prefix cache → 最优!

  ★★★★ 与vLLM PD对比:
    | 维度 | vLLM PD | SGLang PD |
    |------|---------|-----------|
    | KV Transfer | NixlConnector(RDMA) | 内置Mooncake风格 |
    | 异构TP | ✓(P/D不同TP) | ✗(TP需一致) |
    | MLA支持 | ✓ | ✗ |
    | Prefix Caching | BlockHash | ★★★★★ RadixAttention |
    | Overlap Scheduling | ✗ | ✓(SGLang核心) |
    | RL集成成熟度 | ★★★★★ verl默认 | ★★★ verl v0.8.0新增 |

  ★★★★ RTX 4090:
    → HYBRID模式不适用SGLang → SGLang无HYBRID → COLOCATED only
    → PD Disaggregation → 需多GPU → ✗ 不适用
    → ★★★ RTX 4090: SGLang不如vLLM(HYBRID模式) → 但多GPU集群SGLang prefix caching更优!
```

### 2.3 verl SGLang集成当前状态

★★★ verl+SGLang集成仍在成熟阶段 → 不如vLLM集成成熟：

```
★★★ verl + SGLang集成成熟度:

  ✗ HYBRID模式不支持 → SGLang无法同进程 → 无naive零拷贝
  ✓ COLOCATED模式 → 同PG不同进程 → sleep/wake可用
  ✓ STANDALONE模式 → 独立GPU → PD disaggregation
  ✓ LoRA as adapter → SGLang原生LoRA → sleep(level=1) → fast wake-up
  ✗ 集成较新 → 可能不稳定 → 不如vLLM成熟
  ✓ verl CI: e2e_ppo_trainer_megatron_sglang.yml → Megatron+SGLang E2E测试!

  ★★★ 实战建议:
    → RTX 4090单GPU → vLLM HYBRID → 最成熟最优
    → 多GPU集群+GRPO → SGLang COLOCATED → prefix caching更优
    → 大规模PD分离 → SGLang PD → 内置KV transfer → 更简单
    → ★★★★★ 组合策略: 训练用verl+vLLM → 推理部署用SGLang → 各取所长!
```

---

## 3. verl + Megatron: 训练后端

### 3.1 Megatron作为verl的训练引擎

★★★★ verl支持Megatron作为actor/critic的训练策略后端 → 大规模集群专用：

```
★★★★★ verl + Megatron集成架构:

  verl Driver → RayPPOTrainer → MegatronActorWorker/MegatronCriticWorker
    → actor.strategy = "megatron" → Megatron-FSDP mode
    → ★★★★★ 支持671B MoE (DeepSeek-V3) → TP+EP+DP → Megatron 3D并行!
    → ★★★ verl管理RL loop → Megatron管理并行训练 → 分工明确!

  ★★★★★ Megatron并行策略(verl集成):
    → TP=4 + PP=2 + DP=4 → 32 GPU → 70B模型训练
    → TP=8 + EP=64 + DP → 512 GPU → 671B MoE训练
    → ★★★ Megatron 1F1B pipeline → 通信重叠 → 大规模最优
    → ★★★ DeepEP → asymmetric all-to-all → 4.6x faster than NCCL!

  ★★★★★ verl Megatron配置:
    → ppo_megatron_trainer.yaml → Megatron专有配置
    → megatron_actor.yaml / megatron_critic.yaml → 并行策略
    → megatron.yaml → model_engine配置
    → ★★★ verl/examples/grpo_trainer/ → 多个Megatron GRPO示例脚本!

  ★★★★★ verl Megatron示例脚本 (当前repo):
    → run_qwen3_8b_megatron.sh → 8B GRPO
    → run_qwen3_30b_a3b_megatron.sh → 30B MoE GRPO
    → run_qwen3_235b_a22b_megatron.sh → 235B MoE GRPO
    → run_deepseek_v3_671b_megatron.sh → 671B MoE GRPO ★★★★!
    → run_qwen3_5_35b_megatron.sh → 35B GRPO
    → ★★★★ 完整覆盖: 8B→30B→35B→235B→671B → 从小到大!
```

### 3.2 Megatron推理引擎 + verl Rollout

★★★ verl + Megatron训练 + vLLM/SGLang推理 → 训练推理分离：

```
★★★★★ verl + Megatron训练 + vLLM推理:

  训练: MegatronActorWorker → TP+PP+DP → 大规模3D并行
  推理: vLLM/SGLang → TP inference → continuous batching → 高吞吐
  ★★★★★ 3D-HybridEngine → 零冗余权重重分片:
    → 训练: TP=2, PP=2, DP=4 → Megatron 3D
    → 推理: TP=8 → vLLM张量并行
    → 预计算分片映射 → point-to-point传输 → 无全局all-gather
    → 开销: 权重重分片 < 5% 训练时间 → ★★★ 极低!

  ★★★★★ verl Megatron CI测试:
    → e2e_ppo_trainer_megatron_vllm.yml → Megatron+vLLM E2E
    → e2e_ppo_trainer_megatron_sglang.yml → Megatron+SGLang E2E
    → e2e_ppo_trainer_megatron_sglang_ascend.yml → Megatron+SGLang+Ascend E2E!
    → ★★★★★ 三种推理后端都有Megatron训练E2E → 完整覆盖!

  ★★★ verl Megatron实验性功能:
    → fully_async_policy → fully_async_ppo_megatron_trainer.yaml → 异步策略更新
    → one_step_off_policy → one_step_off_ppo_megatron_trainer.yaml → off-policy一步优化
    → ★★★ 实验性 → 不如FSDP2成熟 → 但671B MoE唯一路径!

  ★★★ Megatron推理引擎(DynamicInferenceEngine) → verl也可使用:
    → 但: TRT-LLM export是推荐生产路径 → 不是直接用DynamicInferenceEngine!
    → DynamicInferenceEngine → 开发/调试 → TRT-LLM → 生产!
```

### 3.3 Megatron-FSDP模式

★★★★ verl支持Megatron-FSDP混合模式 → 灵活组合：

```
★★★★ Megatron-FSDP hybrid mode:

  → verl docs/examples/megatron_fsdp_example.rst → 混合模式示例
  → Megatron训练 → FSDP数据并行 → ★★ 两种并行策略组合
  → ★★★ 小模型(7-35B) → FSDP更简单 → Megatron overkill
  → ★★★★ 大模型(70B+/671B) → Megatron TP+PP+EP → 必须使用!

  ★★★ verl训练策略选择决策树:
    → 7B + 单GPU → FSDP2(or rLLM Tinker) → 最简单
    → 7-35B + 8GPU → FSDP2 → 2Ψ通信 → composable → 推荐
    → 35-70B + 16-32GPU → FSDP2+compile → 推荐
    → 70B+ + 32+GPU → Megatron → TP+PP+DP → 必须用!
    → 671B MoE + 128+GPU → Megatron+DeepEP → EP+TP+DP → 必须用!
    → ★★★★★ 规模决定策略 → 小用FSDP2 → 大用Megatron!
```

---

## 4. rLLM Tinker: In-process零拷贝

### 4.1 TinkerBackend架构

★★★★★ rLLM TinkerBackend = RTX 4090唯一最优RL训练路径：

```
★★★★★ rLLM TinkerBackend完整架构:

  TinkerBackend(tinker_backend.py:41):
    → 继承BackendProtocol[Iterable, list[tinker.Datum]]
    → ServiceClient → Tinker SDK客户端 → 连接Tinker service
    → TinkerPolicyTrainer → 延迟初始化 → LoRA auto-init
    → TinkerEngine → in-process推理引擎 → SamplingClient
    → ★★★★★ 全in-process → 无Ray → 无NCCL → 无HTTP → 最简单!

  ★★★★★ 关键特性:
    → bypass_mode = true(default!) → pi_old = rollout logprobs → 省1个forward!
    → LoRA auto-init → create_lora_training_client_async(rank=32) → 无手动配置!
    → train_unembed=True → RL需要output vocabulary → 自动!
    → ★★★★★ RTX 4090最优: TinkerBackend + GRPO + LoRA-32 + bypass_mode → ~17GB!

  ★★★★★ vs verl HYBRID + vLLM:
    | 维度 | TinkerBackend | verl HYBRID |
    |------|---------------|-------------|
    | 分布式 | 无(single GPU) | Ray(multi GPU) |
    | 推理 | TinkerEngine(in-process) | vLLM async server |
    | LoRA | ★★★ auto-init | 手动配置 |
    | Weight Sync | ★★★★★ 零拷贝(GPU merge) | naive(零拷贝) |
    | bypass_mode | ★★★★★ default true | 需手动配置 |
    | detach_metrics | ★★★★★ auto-safe | 需手动配置 |
    | ★ | Tinker更简更快 | verl更成熟更灵活 |

  ★★★★★ 功能等价但更简单:
    → Tinker: in-process → SamplingClient → 零拷贝 → LoRA auto-init
    → verl: 同进程 → naive generator → 零拷贝 → LoRA手动配置
    → ★★★★★ 两者都是单GPU零拷贝 → 但Tinker更简洁(TinkerSDK管理一切)
```

### 4.2 Weight Sync零拷贝详解

★★★★★ rLLM Tinker weight sync = save_checkpoint → new SamplingClient → 零拷贝：

```
★★★★★ Tinker Weight Sync完整路径:

  1. TinkerBackend.on_policy_updated():
     → do_save = global_step % save_freq == 0
     → sampling_client = policy_trainer.save_checkpoint_and_get_sampling_client(global_step, do_save)

  2. TinkerPolicyTrainer.save_checkpoint_and_get_sampling_client():
     → do_save=False: ★★★★★ save_weights_and_get_sampling_client_async()
       → LoRA weights合并到base → GPU上创建新SamplingClient → 无磁盘IO → 极快!
     → do_save=True: save_state + save_weights + checkpoint.json → 磁盘持久化

  3. rollout_engine.set_sampling_client(sampling_client) → 一行完成!

  ★★★★★ 零拷贝原理:
    → Tinker在GPU上管理LoRA merge/unmerge → SamplingClient指向merged weights
    → 无序列化/反序列化循环 → GPU上直接切换 → 极快!
    → SamplingClient是只读快照 → 推理端用merged weights → 无LoRA overhead!
    → 训练端继续从merged state更新LoRA → 推理端用merged → ★★★ 完美解耦!

  ★★★★★ vs verl Weight Sync:
    → verl HYBRID: naive → 参数更新立即可见 → 0ms → 但需要sleep/wake切换GPU空间
    → rLLM Tinker: save_weights → new SamplingClient → <1ms → 但不需要sleep/wake!
    → ★★★★★ Tinker更简单 → 无sleep/wake → 无GPU空间切换 → 更稳定!

  ★★★★★ LoRA auto-init + zero-copy = 完美组合:
    → auto-init → 无手动配置 → rank=32 → train_attn/mlp/unembed → 全自动
    → zero-copy → save_weights → new SamplingClient → GPU-only → 极快
    → ★★★★★ 这是RTX 4090 GRPO训练的最简最快路径!
```

### 4.3 GRPO→PPO自动映射 + Fused Fwd-Bwd-Optim

★★★★ rLLM Tinker内部的GRPO实现 → 自动映射到PPO loss：

```
★★★★★ GRPO→PPO自动映射 (ADV_TO_LOSS_FN_AUTO_MAP):

  REINFORCE → importance_sampling (无clipping)
  REINFORCE_PLUS_PLUS_BASELINE → importance_sampling
  ★★★★★ GRPO → ppo ← GRPO自动映射到PPO clip loss!
  RLOO → importance_sampling
  OTHER → importance_sampling

  ★★★★★ 为什么GRPO→PPO是正确的:
    → GRPO advantage = (r - μ) / σ → group-relative → 正确!
    → PPO clip(r, 1-ε, 1+ε) × A → 防止过大更新 → 稳定训练 → 正确!
    → ★★★★★ group-relative advantage + PPO clip → 最佳组合!

  ★★★★★ Fused Forward-Backward-Optim:
    → fuse_forward_backward_and_optim_step = true → ★★★ async overlap!
    → 1. transform_trajectory_groups_to_datums → advantage computation
    → 2. fwd_bwd_futures = forward_backward_async → ★ async!
    → 3. optim_step_future = optim_step → ★ async! 与fwd-bwd并行!
    → 4. asyncio.gather(*fwd_bwd_futures) + optim_step_future → ★ 重叠!
    → ★★★★★ GPU pipeline overlap → 更高GPU利用率 → 更快训练!
```

---

## 5. Megatron → TensorRT-LLM: 推理部署桥梁

### 5.1 TRT-LLM Export完整流程

★★★★★ Megatron训练 → TRT-LLM export → TensorRT推理 = 生产级推理推荐路径：

```
★★★★★ Megatron → TRT-LLM Export架构:

  ★★★★★ Export是Megatron推理的推荐路径 → 不是直接用DynamicInferenceEngine!

  TRTLLMHelper (trtllm_helper.py):
    → TransformerConfig + ModelType + conversion dict → TRT-LLM PretrainedConfig
    → _get_trtllm_config(): 构建GPT/LLaMA/Mixtral/Gemma/Falcon配置
    → ★★★ MoE config: moe_num_experts + moe_top_k + moe_normalization_mode + moe_tp_mode

  TRTLLMLayers (trtllm_layers.py):
    → Enum: Megatron→TRT-LLM layer name mapping
    → position_embedding/vocab_embedding/lm_head/final_layernorm
    → attention_qkv/dense → mlp_fc/projection → mlp_router → expert variants

  TRTLLMEngineBuilder (engine_builder.py):
    → build_and_save_engine() → tensorrt_llm build
    → PluginConfig: paged_kv_cache + remove_input_padding + gpt_attention_plugin + gemm_plugin
    → BuildConfig → model class → load weights → build engine → save disk
    → ★★★★★ 编译期优化 → AOT graph optimization → 最大吞吐!

  ★★★★★ Export流程:
    Megatron训练 → distributed checkpoint
      → TRTLLMHelper.get_trtllm_pretrained_config_and_model_weights()
        → Single-device: per-rank weights → CPU/GPU conversion
        → Distributed: each GPU shard → DistributedTRTLLMModelWeightsConverter
      → TRTLLMEngineBuilder.build_and_save_engine()
        → TensorRT engine → paged KV cache + remove padding + GPT attention plugin
      → trtllm-serve → OpenAI兼容API → ★★★★ 生产级推理!
```

### 5.2 TRT-LLM vs vLLM vs SGLang推理部署

★★★★★ 三种推理框架部署对比 — Megatron训练后的选择：

```
★★★★★ 从Megatron训练到推理部署的3种路径:

  ★★★★★ Path A: Megatron → TRT-LLM (NVIDIA推荐):
    → 编译期优化 → XQA decode kernel → FP4/FP8 → 最大吞吐
    → ★★★★★ DeepSeek-R1 on GB200: FP4 + Wide-EP → 显著优于其他框架
    → ★★★★ MoE最优: Wide-EP + EPLB + NVFP4 GEMM → DeepSeek专用最优
    → ✗ 编译时间长(10-60min) → 新模型需等待适配 → NVIDIA only
    → ✗ 调试困难 → 编译后难以debug → 黑盒

  ★★★★★ Path B: Megatron → HF safetensors → vLLM (通用推荐):
    → mbridge → Megatron→HF格式转换 → safetensors
    → vLLM INT4 + INT8KV + prefix caching → 4,791 tok/s (RTX 4090)
    → ★★★★★ 200+架构开箱即用 → 新模型快速支持 → 社区活跃
    → ★★★★★ verl集成 → RL训练天然配合 → 最灵活
    → ★★★★ 多硬件(NVIDIA/AMD/TPU) → 不绑定NVIDIA

  ★★★★ Path C: Megatron → HF → SGLang (prefix最优):
    → SGLang → RadixAttention → prefix复用 → GRPO rollout最优
    → ★★★★★ overlap scheduling → 零开销CPU调度 → 高吞吐
    → ★★★ xAI/Cursor等头部采用 → RL领域主力
    → ✗ 新于vLLM → 不如vLLM成熟

  ★★★★★★ 决策建议:
    → NVIDIA集群+极致性能 → TRT-LLM → 最大吞吐
    → RTX 4090+通用推理 → vLLM INT4 → 最灵活
    → GRPO训练+prefix需求 → SGLang → prefix最优
    → ★★★★★ 实战: 训练用verl+vLLM → 部署看场景 → 通用选vLLM → 极致选TRT-LLM!
```

### 5.3 DynamicInferenceEngine — Megatron内置推理

★★★ Megatron DynamicInferenceEngine → 可用但不推荐作为生产推理：

```
★★★ Megatron DynamicInferenceEngine:

  → DynamicInferenceEngine → @experimental_api → 还在迭代
  → continuous batching → CUDA graph多batch维度 → InferenceTopKRouter(@torch.compile)
  → ★★★ 推理路由 = compile + dense_output → 2大优化
  → NCCL dispatcher → RTX 4090可用 → NVLS dispatcher → SM90 only

  ★★★ RTX 4090推理兼容性:
    ✓ DynamicInferenceEngine → YES
    ✓ CUDA graphs (decode-only) → YES
    ✓ NCCL token dispatcher → YES
    ✓ InferenceTopKRouter → YES
    ✓ FlashInfer grouped GEMM → YES (SM89)
    ✓ Prefix caching → YES
    ✗ NVLS dispatcher → NO (SM90)
    ✗ Variable token EP decode → NO (NVLS AGV/RSV only)

  ★★★ 推荐使用场景:
    → 开发/调试 → DynamicInferenceEngine → 快速验证
    → 小规模内部服务 → 可用但不最优
    → ★★★★ 生产 → TRT-LLM export → 最大吞吐!
```

---

## 6. DeepSpeed ZeRO: 训练辅助+MoE局限

### 6.1 DeepSpeed ZeRO作为训练后端

★★★ DeepSpeed ZeRO → verl训练后端选项 → 但不如FSDP2推荐：

```
★★★ verl + DeepSpeed集成:

  verl --actor.strategy=deepspeed → ZeRO-2/3 backend
    → ★★★ 多GPU场景 → ZeRO-2+LoRA → 3x内存省(optimizer分片)
    → ★★★ ZeRO-Offload → CPU Adam → 内存更省 → 但速度慢

  ★★★★★ DeepSpeed ZeRO核心功能:
    → ZeRO-1: optimizer state分片 → 4x内存省
    → ZeRO-2: optimizer+gradient分片 → 8x内存省 → ★★★ 推荐
    → ZeRO-3: optimizer+gradient+parameter分片 → Nx内存省 → 但通信量大
    → ★★★★★ Universal Checkpoint → ZeRO→HF格式 → 跨框架桥梁

  ★★★ DeepSpeed几乎停转 → ★★★★ 不推荐投入:
    → PR merge极低 → 社区活跃度下降 → 微软重心转移
    → ★★★★ FSDP2替代 → PyTorch原生 → 更简洁 → 未来标准
    → ★★★ AutoEP(ZeRO-0/1/2+MoE)已merged → 但ZeRO-3+MoE不支持!

  ★★★ RTX 4090:
    → 单GPU → ZeRO backend无用 → LoRA+compile更effective
    → ZeRO-2+LoRA+CPU_Adam → ~17GB → ✗ 无GRPO → 需自实现RL
    → ★★★ 辅助角色 → 非RL → 适合static finetuning → 不适合GRPO
```

### 6.2 DeepSpeed ZeRO与MoE的冲突

★★★★★ DeepSpeed ZeRO-3与MoE Expert Parallelism存在根本冲突：

```
★★★★★ ZeRO-3 + MoE EP根本冲突:

  ★★★★★ 核心问题:
    → ZeRO-3分片所有参数(包括expert参数) → 按DP rank分配
    → 但EP需要expert分布在不同GPU → device-local → 不应跨DP分片!
    → ★★★★★ ZeRO-3 uniform sharding ≠ MoE sparse activation → 冲突!
    → 导致: expert参数all-gather通信 → 冗余 → 破坏EP locality benefit

  ★★★ 当前最佳实践:
    → ZeRO-2 + EP → 最稳定组合 → optimizer+gradient分片 → 参数不分片
    → ★★★★ ZeRO-3 + EP → 需custom tuning → 不推荐 → 官方不支持!
    → AutoEP(ZeRO-0/1/2) → ★★★★ 已merged → 自动检测MoE → 无需改模型代码!
    → AutoEP(ZeRO-3) → follow-up → ★★★ 还不支持!

  ★★★★★ vs Megatron MoE:
    | 维度 | DeepSpeed AutoEP | Megatron MoE |
    |------|------------------|-------------|
    | 代码修改 | ★★★★★ 零修改(HF兼容) | 手动配置(需改模型代码) |
    | EP通信 | all-to-all symmetric(NCCL) | ★★★★ DeepEP asymmetric → 4.6x快 |
    | ZeRO集成 | ZeRO-0/1/2 → ★★★ 稳定 | 无ZeRO → FSDP2/TP |
    | 量化 | ✗ 无FP8/W4A8 | ★★★★ FP8/W4A8 grouped-GEMM |
    | 适用 | HuggingFace兼容模型 | Megatron原生模型 |

  ★★★★★ 关键洞察:
    → DeepSpeed定位: HuggingFace兼容 → 零修改 → 适合快速实验
    → Megatron定位: 大规模集群 → DeepEP+FP8 → 适合生产训练
    → ★★★★★ RTX 4090: AutoEP ZeRO-2 → 可行(但EP>1=跨GPU→PCIe限制!)
    → ★★★★★ H100集群: Megatron DeepEP → 更优 → asymmetric+FP8!
```

### 6.3 DeepSpeed Universal Checkpoint → HF桥梁

★★★★ DeepSpeed universal checkpoint → 4步转换 → 最慢但可行：

```
★★★★ DeepSpeed → HF → 推理部署路径:

  ★★★★ 4步转换(最慢):
    1. ZeRO checkpoint → universal checkpoint → 跨rank合并
    2. Universal → FP32 per-parameter → safetensors
    3. FP32 safetensors → HF format → merge LoRA(if applicable)
    4. HF → INT4/FP8 → vLLM/SGLang/TRT-LLM → 推理

  ★★★★★ vs 其他框架:
    → FSDP2 → FSDPModelMerger → HF → ★★★★ 最简洁(1步)
    → Megatron → mbridge → HF → ★★★ 中等复杂度(2步)
    → verl → actor checkpoint = HF → ★★★★ 直接推理(1步)
    → rLLM → save_pretrained → HF → ★★★★★ 最简路径(1步)
    → ★★★★ DeepSpeed → 4步 → 最慢 → 不推荐!

  ★★★★★ HF format = 所有框架的统一桥梁 → AI infra的"HTTP":
    → 训练任何框架 → 推理任何框架 → 通过HF format桥接
    → ★★★★★ verl/rLLM → HF → vLLM = 最简最快完整路径!
```

---

## 7. MindIE/vLLM-Ascend: Ascend推理替代

### 7.1 3种Ascend推理路径

★★★★★ Ascend NPU推理生态系统3条路径 → 中国AI infra战略关键：

```
★★★★★ Ascend推理3路径:

  ★★★★★ Path 1: vLLM-Ascend (推荐★★★★):
    → 继承vLLM scheduling+batching+KV cache管理 → op-level CUDA→NPU替换
    → 5层桥接: Platform→Device→Op→Model→Worker → 系统化替换
    → ★★★★★ 保留vLLM调度逻辑 → 可控 → 生产运维友好
    → ★★★ LoRA支持(bgmv/sgmv Ascend custom ops) → Multi-LoRA
    → ★★★★ PD disaggregation → DeepSeek-V3.1/V3.2 → Production!
    → ★★★ prefix caching → GRPO rollout可控
    → ✗ 性能不如MindIE → op-level vs graph-level

  ★★★ Path 2: MindIE (性能最高★★★):
    → 华为官方推理引擎 → ATB graph-level → 整Transformer=1 op
    → ★★★ npu_dequant_swiglu_quant → 三融合 → NVIDIA无等价!
    → ★★★ MindIE Turbo → DeepSeek专用kernel级优化
    → ✗ 核心不开源 → 黑盒 → 无法自定义 → 无法debug
    → ✗ 版本绑定CANN → 升级困难 → 与vLLM生态脱节

  ★★★ Path 3: SGLang-Ascend (中间★★):
    → SGLang HTTP → MindIE internal → wrapper
    → ★★★ 零代码改推理 → MindIE性能直接获得 → 最快部署
    → ✗★★★ 丢失SGLang调度控制 → 无RadixAttention → 不是真SGLang!
    → ✗ 与SGLang GPU版功能差距大 → prefix scheduling不同

  ★★★★★★ 关键选择:
    → 需要灵活+可控 → vLLM-Ascend → ★★★★★ 推荐
    → 需要最高性能+不想自定义 → MindIE → ★★★
    → 需要快速部署+不关心调度 → SGLang-Ascend → ★★ (不推荐!)
    → ★★★★★ GRPO rollout → prefix caching关键 → vLLM-Ascend更可控!
```

### 7.2 vLLM-Ascend与verl集成

★★★★ verl支持Ascend NPU → vLLM-Ascend作为rollout引擎：

```
★★★★ verl + vLLM-Ascend (Ascend NPU)集成:

  ★★★★★ verl CI测试:
    → e2e_ppo_trainer_megatron_vllm_2_ascend.yml → Megatron+vLLM-Ascend E2E!
    → e2e_ppo_trainer_megatron_sglang_ascend.yml → Megatron+SGLang-Ascend E2E!
    → ★★★★★ verl完整支持Ascend NPU → 多硬件→ NVIDIA+AMD+Ascend!

  ★★★★★ Ascend GRPO训练路径:
    → verl Driver → vLLM-Ascend rollout → HCCL通信 → Ascend 910B/910C
    → Megatron → TP+EP → HCCL → Ascend训练
    → ★★★★★ 中国NPU场景 → vLLM-Ascend + verl → 生产级RL训练!

  ★★★★★ vs GPU场景:
    → Ascend: FP8/W8A8为主 → MXFP4 on A5 → 无INT4 → 不同量化路径
    → NVIDIA: INT4(GPTQ)为主 → FP8(SM90+) → FP4(SM120) → 不同量化路径
    → ★★★★ MXFP4 = 未来统一方向 → Ascend A5 + RTX 5090 → 都支持!

  ★★★★★ RTX 4090: 全部不适用(NPU only) → 但理解对中国AI infra战略关键!
```

### 7.3 MindIE ATB kernel架构

★★★ MindIE = 华为官方推理 → ATB graph-level → 最深融合：

```
★★★ MindIE ATB (Ascend Transformer Boost):

  → ATB = graph-level optimization → 整Transformer=1 op → 最深融合
  → ★★★ npu_dequant_swiglu_quant → dequant+SwiGLU+quant → 三融合 → NVIDIA无等价!
  → ★★★ MindIE Turbo → DeepSeek-V3/R1/Qwen-2 → 额外kernel级优化
  → ★★★ FP8 → npu_quant_matmul → Ascend原生FP8 → 910C硬件加速
  → ★★★★ CANN custom ops = FlashMLA equivalent → MLA preprocess all-fused

  ★★★★★ vs vLLM-Ascend:
    → MindIE: graph-level → 整图优化 → 最快 → 但黑盒
    → vLLM-Ascend: op-level → 逐op替换 → 更慢 → 但透明可控
    → ★★★★★ 生产选vLLM-Ascend → 性能选MindIE → 不可兼得!

  ★★★★★ vs NVIDIA推理:
    → MindIE ATB → 类似TensorRT-LLM → 都是graph-level → 编译式优化
    → vLLM-Ascend → 类似vLLM GPU → 都是op-level → 解释式执行
    → ★★★★★ 设计哲学完全相同 → 只是硬件不同 → CUDA→CANN
```

---

## 8. 7框架完整集成拓扑

### 8.1 3层集成架构

★★★★★ 7框架生产集成 = 3层拓扑 → verl=中枢 → HF=桥梁：

```
★★★★★ 7框架生产集成 = 3层:

Layer 1: ★★★ 训练层 (Training)
  DeepSpeed ZeRO → 通用分布式训练 → ZeRO-2+LoRA → RTX 4090辅助
  PyTorch FSDP2 → 未来标准 → BF16+compile → 多GPU首选
  Megatron-LM → 3D并行 → TP+PP+DP+EP → 大规模集群 → 671B MoE
  ★★★★★ verl → 训练编排 → actor.strategy=fsdp/fsdp2/megatron/deepspeed

Layer 2: ★★★★ RL/训练-推理集成层 (RL Training)
  verl → ★★★★★ 中枢 → 训练+vLLM/SGLang推理 → weight sync
  → RolloutMode: HYBRID(COLOCATED/STANDALONE → 3种GPU分配
  → Rollout引擎: vLLM/SGLang/TRT-LLM/naive → 4种推理后端
  rLLM Tinker → ★★★★★ RTX 4090最优 → in-process → GRPO+LoRA
  Megatron GRPO → ★★★ 大规模 → 但推理引擎需外部

Layer 3: ★★★★ 推理层 (Inference Serving)
  vLLM → ★★★★★ 通用GPU推理 → INT4+INT8KV → 4,791 tok/s
  SGLang → ★★★★ 高吞吐推理 → RadixAttention → GRPO rollout更优
  MindIE → ★★★ Ascend NPU推理 → ATB+FP8 → 中国场景
  vLLM-Ascend → ★★★★ Ascend灵活推理 → op-level → 可控
  TRT-LLM → ★★★★★ NVIDIA极致性能 → 编译式 → FP4+Wide-EP

★★★★★ 桥梁:
  HF format → ★★★★★ 所有训练→推理的桥梁 → AI infra的"HTTP"
  PyTorch → ★★★★ 所有框架的底层 → DTensor+compile+NCCL/HCCL
```

### 8.2 跨框架数据流全景

★★★★★ 生产数据流 = 训练 → 检查点 → 转换 → 量化 → 推理：

```
★★★★★ 生产数据流拓扑 (2026-06-16最终版):

Training → Checkpoint → Conversion → Quantization → Serving
    ↓          ↓           ↓           ↓           ↓
DeepSpeed   universal     FP32→HF    GPTQ INT4   vLLM/SGLang
FSDP2       ModelMerger   per-param  INT8 KV     INT4+EAGLE
Megatron    mbridge       safetensor W4A8 Triton  TRT-LLM
verl        actor ckpt    HF direct  FP8(910C)   MindIE/vLLM-Ascend
rLLM        save_pretrained HF      INT4+INT8KV vLLM GPU

★★★★★ RL Training特殊数据流:
  Prompt → Rollout(vLLM/SGLang/Tinker) → Reward(rule-based) →
  Advantage(GRPO) → Policy Update(LoRA) → Weight Sync →
  New Rollout → 循环!

  ★★★★★ rLLM: 全in-process → 无跨框架传输 → 最简!
  ★★★★★ verl HYBRID: 同进程 → naive零拷贝 → 最简GPU共享!
  ★★★ verl COLOCATED: 跨进程 → ZMQ IPC → 多GPU
  ✗✗✗ verl STANDALONE: 跨GPU → NCCL/NIXL → PCIe灾难(RTX 4090)

★★★★★★ 完整生产路径(从训练到推理):
  rLLM Tinker → save_pretrained → HF format → INT4 → vLLM → 4,791 tok/s
  → 或 INT4+EAGLE → 9,088 tok/s → ★★★★★ 极简极快!
  → ★★★★★ 这就是7框架组合的最终目标 → 从训练到推理的完整闭环!
```

---

## 9. RTX 4090生产部署推荐

### 9.1 RTX 4090最优路径

★★★★★★ RTX 4090生产RL训练最优路径(2026-06-16最终版)：

```
★★★★★★ RTX 4090 最优路径排序:

★★★★★★ Path 1: rLLM Tinker (最简最快):
  → TinkerBackend + GRPO + LoRA-32 + bypass_mode → ~17GB
  → Weight sync: zero-copy → <1ms → GPU-only merge
  → → save_pretrained → HF → INT4 → vLLM → 4,791 tok/s → EAGLE → 9,088 tok/s
  → ★★★★★ 从训练到推理 → 最简最快 → RTX 4090最优!

★★★★ Path 2: verl HYBRID + vLLM (更成熟):
  → verl HYBRID + GRPO + LoRA + bypass_mode + detach_metrics → ~17.6GB
  → Weight sync: naive → 0ms (same process)
  → → merge → HF → INT4 → vLLM → 4,791 tok/s
  → ★★★★ 更成熟 → Ray生态 → GPU集群支持 → 但单GPU不如rLLM

★★★ Path 3: DeepSpeed ZeRO-2 (辅助训练):
  → ZeRO-2 + LoRA + CPU_Adam → ~17GB → ✗ 无GRPO → 需自实现RL
  → → universal ckpt → FP32 → HF → INT4 → vLLM
  → ★★ 辅助 → 非RL → 适合static finetuning → 不适合GRPO

★★ Path 4: Megatron推理引擎 (推理):
  → DynamicInferenceEngine → INT4 → CUDA graph → NCCL → 单GPU
  → ★★ 推理可用 → 但不如vLLM成熟 → SM89缺关键kernel
  → ★★★ 不适合RL训练 → 太重 → 单GPU overkill

✗✗✗ Path 5: MindIE (NPU only → RTX 4090不适用)
  → NPU推理 → 910C → FP8 → ATB → 高性能 → 但GPU不可用
  → ★★★ 中国NPU场景 → 但RTX 4090完全不行
```

### 9.2 多GPU集群路径

★★★★★ 多GPU集群RL训练路径(8+GPU)：

```
★★★★★ 多GPU集群RL训练路径:

★★★★★★ Path A: verl + FSDP2 + vLLM (8-32 GPU推荐):
  → verl HYBRID/COLOCATED → FSDP2训练 → vLLM rollout → DP并行
  → ★★★★★ 2Ψ通信 → composable → compile兼容 → 未来标准
  → ★★★★ TransferQueue → 零拷贝 → 49.1% e2e improvement
  → ★★★★★ DP=8 → 8个rollout replica → prefix caching × 8 → 高吞吐!

★★★★★ Path B: verl + Megatron + vLLM/SGLang (32+ GPU):
  → verl → Megatron训练(TP+PP+DP) → vLLM/SGLang推理(TP)
  → ★★★★★ 3D-HybridEngine → 零冗余权重重分片 → <5%开销
  → ★★★★★ 671B MoE → DeepEP+FP8 → H100集群专用
  → ★★★★★ verl管理RL loop → Megatron管理并行 → 分工明确!

★★★★ Path C: verl + SGLang PD Disaggregation (大规模集群):
  → Prefill GPU → compute KV → Transfer → Decode GPU → 持续decode
  → ★★★★★ 1P:N decode → 独立扩展 → 最优资源利用率
  → ★★★ SGLang RadixAttention → GRPO prefix复用 → 7x省prefill
  → ★★★★★ 适合万级GPU集群 → DeepSeek-R1规模训练

★★★★ Path D: verl + vLLM-Ascend (Ascend NPU集群):
  → verl → vLLM-Ascend rollout → HCCL → Ascend 910B/910C
  → ★★★★★ 中国NPU场景 → vLLM调度 → 可控prefix caching → 生产级
  → ★★★ verl + Megatron-Ascend → TP+EP → HCCL → 大规模MoE训练

★★★ Path E: DeepSpeed ZeRO-2 + AutoEP (HuggingFace兼容):
  → ZeRO-2 + AutoEP → MoE零代码 → HuggingFace兼容 → 快速实验
  → ★★★ 适合实验 → 不适合大规模生产 → DeepSpeed停转
```

### 9.3 关键配置参数

★★★★★ RTX 4090 GRPO训练关键配置：

```
★★★★★ RTX 4090 GRPO训练关键参数:

  ★★★★★ rLLM Tinker配置 (最优):
    backend: tinker
    estimator: GRPO (→ PPO loss auto)
    model.lora_rank: 32 (auto-init)
    model.train_unembed: true (auto)
    model.train_attn: true (auto)
    model.train_mlp: true (auto)
    bypass_mode: true (default!)
    fuse_forward_backward_and_optim_step: true (overlap)
    rollout.n: 8 (group_size=8)
    train_batch_size: 32
    learning_rate: 1e-6
    reward: rule-based (math/code)
    ★★★★★ 总内存: ~17GB → 24GB ✓✓✓ → 7GB headroom!

  ★★★★ verl HYBRID配置 (备选):
    actor.strategy: fsdp2 (or fsdp)
    rollout.name: vllm
    rollout.mode: hybrid (★★★★ RTX 4090)
    rollout.n: 8
    actor.use_kl_loss: false (GRPO bypass)
    actor.ppo_mini_batch_size: 16
    model.enable_gradient_checkpointing: true
    model.lora_rank: 32 (★★ 手动配置)
    algorithm.adv_estimator: grpo
    reward.custom_reward_function: compute_score (rule-based)
    ★★★★ 总内存: ~17.6GB → 24GB ✓✓✓ → 6.4GB headroom!

  ★★★★★ INT4推理配置 (部署阶段):
    quantization: gptq_int4 (or awq)
    kv_cache_dtype: int8 (★★★★★ RTX 4090唯一可行KV路径)
    enable_prefix_caching: true
    gpu_memory_utilization: 0.9
    enforce_eager: true (★★★ SM89 spec decode必须)
    max_model_len: 4096
    ★★★★★ 推理内存: ~11GB → 4,791 tok/s → EAGLE→9,088 tok/s
```

---

## 10. 关键洞察总结

### 10.1 ★★★★★ 顶级洞察

```
★★★★★★ 7框架集成10个关键洞察:

1. ★★★★★★★ verl = RL中枢 → 连接训练+推理:
   → 3种RolloutMode(HYBRID/COLOCATED/STANDALONE) → 4种推理后端(vLLM/SGLang/TRT-LLM/naive)
   → 4种训练策略(fsdp/fsdp2/megatron/deepspeed) → ★★★★★ 最灵活的RL编排!

2. ★★★★★★★ rLLM Tinker = RTX 4090最优:
   → in-process → 零拷贝 → LoRA auto-init → bypass_mode default → ★★★★★ 最简最快!
   → vs verl HYBRID: 功能等价但更简单 → TinkerSDK管理一切 → 无Ray overhead!

3. ★★★★★★★ HF format = AI infra的HTTP:
   → 所有训练→推理的统一桥梁 → 训练任何框架 → 推理任何框架
   → ★★★★★ verl/rLLM → HF → vLLM = 最简最快完整路径!

4. ★★★★★★★ GRPO > PPO (2026年趋势):
   → 所有5框架都支持GRPO → PPO在24GB GPU不可能(48GB内存)
   → ★★★★★ 这是2025-2026最重要RL训练范式转变!

5. ★★★★★★★ Weight Sync = 集成核心瓶颈:
   → naive(HYBRID)/GPU-only merge(rLLM) → 零拷贝 → RTX 4090唯一可行
   → CUDA IPC/NCCL/NIXL → 跨进程 → RTX 4090PCIe灾难
   → ★★★★★ LoRA只sync adapter → 5.4x减少 → 极关键!

6. ★★★★★★★ Megatron → TRT-LLM = 生产推理推荐:
   → 不是直接用DynamicInferenceEngine → 而是export后用TRT-LLM
   → ★★★★★ 编译式优化 → 最大吞吐 → NVIDIA集群最优
   → 但: vLLM更灵活 → SGLang prefix最优 → 不同场景不同选择!

7. ★★★★★★★ DeepSpeed ZeRO-3 + MoE EP根本冲突:
   → ZeRO-3分片所有参数(含expert) → 与EP device-local冲突
   → ★★★★★ AutoEP(ZeRO-0/1/2)解决 → 但ZeRO-3不支持 → 设计局限
   → ★★★★★ vs Megatron: 无ZeRO → FSDP2+TP → EP+TP自然组合

8. ★★★★★★★ vLLM-Ascend = Ascend推荐:
   → 保留vLLM调度 → 可控 → prefix caching可控 → GRPO rollout可行
   → ★★★★★ vs MindIE: 更灵活 → vs SGLang-Ascend: 更真实(不是wrapper)
   → ★★★★★ 中国NPU场景 → verl+vLLM-Ascend → 生产级RL训练!

9. ★★★★★★★ 3层拓扑 = 生产架构:
   → 训练层(DeepSpeed/FSDP2/Megatron) → RL层(verl/rLLM) → 推理层(vLLM/SGLang/TRT-LLM/MindIE)
   → ★★★★★ 每层独立选择 → HF format桥接 → 灵活组合!

10. ★★★★★★★ Scale决定策略 → 小用rLLM → 大用verl+Megatron:
    → 1 GPU → rLLM Tinker → in-process → 最简最快
    → 4-8 GPU → verl+FSDP2+vLLM → 成熟灵活
    → 32+ GPU → verl+Megatron+vLLM/SGLang → 大规模最优
    → 128+ GPU → verl+Megatron+DeepEP+SGLang PD → 671B MoE
    → ★★★★★★★ 规模决定策略 → 不要over-engineering!
```

### 10.2 ★★★★★ 实战部署建议

```
★★★★★★ 实战部署建议(2026-06-16):

★★★★★★ RTX 4090单GPU → rLLM Tinker + GRPO + LoRA-32 + bypass_mode:
  → 训练: TinkerBackend → ~17GB → zero-copy weight sync
  → 推理部署: HF → INT4 → vLLM → 4,791 tok/s → EAGLE → 9,088 tok/s
  → 评估: rllm eval --attempts → pass@k → CPU warm-pool → 不占GPU
  → ★★★★★★★ 这是RTX 4090唯一最优路径 → 不要选其他!

★★★★★★ 8×A100/H100 → verl + FSDP2 + vLLM HYBRID:
  → 训练: FSDP2 + compile → 2Ψ通信 → composable
  → 推理: vLLM HYBRID → naive零拷贝 → prefix caching
  → ★★★★★★★ TransferQueue + KV reuse → 49.1% improvement!

★★★★★★ 32+×H100 → verl + Megatron + vLLM/SGLang:
  → 训练: Megatron TP+PP+DP → 3D并行 → 70B+模型
  → 推理: 3D-HybridEngine → 零冗余重分片 → vLLM/SGLang TP
  → ★★★★★★★ 671B MoE → DeepEP+FP8 → Megatron必须!

★★★★★★ Ascend NPU → verl + vLLM-Ascend:
  → 训练: Megatron-Ascend → HCCL → TP+EP
  → 推理: vLLM-Ascend → prefix caching → GRPO rollout可控
  → ★★★★★★★ 中国NPU生产级 → 不要选SGLang-Ascend(丢失核心优势)!
```

---

## 参考资料

### 项目源码
- [verl](https://github.com/volcengine/verl) — RL训练中枢
- [rLLM](https://github.com/rllm-org/rLLM) — Agent RL编排
- [vLLM](https://github.com/vllm-project/vllm) — 通用推理服务
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) — Ascend推理
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — 3D并行训练
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA推理
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — ZeRO训练
- [MindIE](https://www.hiascend.com/document) — 华为推理引擎

### 关键论文
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256) (EuroSys 2025)
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Nature 2025)
- [RLOO](https://arxiv.org/abs/2402.14740)
- [REINFORCE++](https://arxiv.org/abs/2501.03262)

### 本项目相关笔记
- verl架构: `notebook/projects/verl-architecture.md`
- verl异步rollout: `notebook/projects/verl-async-rollout-architecture-deep-dive.md`
- verl分布式架构: `notebook/projects/distributed-rl-training-verl-architecture.md`
- rLLM Tinker: `notebook/projects/rllm-tinker-backend-deep-reading.md`
- rLLM v0.3: `notebook/projects/rllm-v0.3-terminal-rl-reading.md`
- verl vs rLLM: `notebook/projects/verl-vs-rllm-transform-comparison.md`
- 7框架拓扑: `notebook/projects/seven-framework-integration-topology.md`
- 7框架对比: `notebook/projects/seven-framework-comparison.md`
- RL训练对比: `notebook/projects/rl-training-design-patterns-comparison.md`
- 推理-训练集成: `notebook/projects/inference-training-integration-pipeline-reading.md`
- RLHF infra: `notebook/projects/rlhf-training-infra-2026.md`
- vLLM PD分离: `notebook/projects/vllm-pd-disaggregation-reading.md`
- Megatron推理: `notebook/projects/megatron-inference-engine-reading.md`
- TRT-LLM对比: `notebook/projects/tensorrt-llm-comparison.md`
- NPU生态: `notebook/projects/npu-inference-ecosystem-comparison.md`
- vLLM-Ascend生产: `notebook/projects/vllm-ascend-production-reading.md`
- DeepSpeed最新: `notebook/projects/deepspeed-latest-developments-2026-06.md`
- verl Megatron扩展: `verl/docs/advance/megatron_extension.rst`
