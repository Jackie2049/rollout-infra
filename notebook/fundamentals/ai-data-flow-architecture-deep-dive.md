# AI Data Flow Architecture Deep Dive

> 2026-06-10 | 数据流=AI系统血脉! 从请求到响应、从训练到部署、从checkpoint到推理, 三条核心数据流路径
> 关联: scheduler-architecture-deep-dive.md, kv-cache-management-deep-dive.md

## 0. 核心定律: 数据流 = 系统的本质

AI系统本质是数据转换管道。理解数据流就是理解系统本身。

三条核心数据流:
1. **Inference**: request→token→prefill→decode→response (实时服务)
2. **Training**: data→tokenize→batch→gradient→model→checkpoint (离线训练)
3. **Deployment**: checkpoint→model_registry→load→serve (部署上线)

## 1. Inference数据流 — vLLM V1请求路径 (7步)

```
Step 1: HTTP → API Server
  POST /v1/completions → JSON body (prompt + params)
  OpenAI兼容API → 多模型统一接口

Step 2: API Server → Tokenizer
  prompt(string) → BPE tokenize → prompt_token_ids(list[int])
  LLaMA: "Hello world" → [1, 15496, 995] (3 tokens)
  数据量: 512 prompt → 512 ints → 2KB (几乎零传输成本)

Step 3: Tokenizer → Scheduler
  Request对象 → waiting queue → FCFS/优先级调度
  分配block_ids → PagedAttention → block_size=16 → 按需分配
  SchedulerOutput → 传给ModelRunner

Step 4: Scheduler → ModelRunner (关键数据传输)
  SchedulerOutput包含:
  - scheduled_new_reqs: list[NewRequestData] → 首次全量发送
    req_id + prompt_token_ids + block_ids + sampling_params + mm_features
  - scheduled_cached_reqs: CachedRequestData → 增量发送(减少通信!)
    req_ids + new_token_ids + new_block_ids + num_computed_tokens
  - total_num_scheduled_tokens: GPU budget (e.g. 320 tokens)
  - kv_connector_metadata: PD分离KV传输指令

  双进程架构: EngineCoreProc → ZMQ IPC → serialize→send→receive→deserialize
  IPC开销 ≈ 0.1ms (可忽略)

Step 5: ModelRunner → GPU Forward
  prefill: 所有prompt token → 一次处理 → compute-bound → 高TFLOPS
  decode: 1 token per request → batch → memory-bound → 低TFLOPS
  KV cache写入 → PagedAttention → block→page→slot

Step 6: GPU → Sampling
  logits → temperature → top-p → top-k → min-p → next_token_id
  sampling overhead ≈ 0.06ms (不是瓶颈!)
  structured output → FSM bitmask → xgrammar约束解码

Step 7: Sampling → Scheduler → Response
  next_token_id → 重复Step 3-7 → 直到EOS或max_tokens
  finish → finish_reason → JSON response → HTTP → 用户!
```

**数据量估算 (7B模型)**:
- prompt S=512 → 512 ints → 2KB
- logits → vocab_size × num_requests → 32K × 32 = 1MB
- KV per layer → 512 × 2 × 4096 × 8 = 2MB → 32 layers = 64MB (BF16)
- INT8 KV → 32MB (省一半)

## 2. Training数据流 — verl RL训练 (14步)

```
1. Dataset → DataLoader → batch → tokenize → input_ids
2. input_ids → embedding → attention → MLP → output (forward)
3. output → reward_model → reward_scores → advantage estimation
4. advantage → policy_loss → gradient → backward
5. gradient → optimizer → weight_update → new_model → repeat
6. model → checkpoint → disk/S3 → recovery → load → continue

verl RL特殊数据流:
  rollout: vLLM inference → prompt→response (推理服务)
  reward: response→score (奖励计算)
  训练: score→gradient→update (参数更新)
  循环: 上述3步反复 (RL loop)

  colocation模式: 2模型共享GPU → actor+critic → sleep/wake → 50% GPU省
  async rollout: 推理→CPU队列→训练→异步→吞吐↑
```

**Checkpoint数据流**:
- model.state_dict → serialize → write → 7B BF16 = 14GB (慢!)
- INT4模型 = 3.5GB (快!)
- verl异步checkpoint → 版本管理 → 最新一致版本恢复

## 3. Serialization格式 — 数据编码与压缩

| 格式 | 用途 | 大小 | 速度 | 安全 |
|------|------|------|------|------|
| PyTorch .pt | 模型权重 | 大 | 慢(pickle) | 不安全 |
| SafeTensors | HF标准 | 同 | 快(无pickle) | 安全 |
| GGUF | llama.cpp | 小(量化) | 快 | 安全 |
| JSON | tokenizer | KB级 | 快 | 安全 |
| pickle | checkpoint | 大 | 慢 | 不安全 |

**压缩率对比**:
- BF16: 2 bytes/param (基准)
- FP8: 1 byte/param (50%省, TE用)
- INT4: 0.5 byte/param (75%省, AWQ用, 需fused kernel!)
- INT8 KV: 1 byte/value (50%省)

**传输瓶颈**:
- PCIe: 12GB/s → 模型加载 14GB → 1.2s (慢) / INT4 → 0.3s (快)
- NVLink: 300GB/s → KV transfer瞬间 (RTX 4090无NVLink!)
- Ethernet: 10GB/s → 跨节点需RDMA

## 4. Checkpoint管理 — 状态持久化

Checkpoint内容:
- model.state_dict (权重): 7B BF16=14GB → 最大部分
- optimizer.state_dict (Adam m+v): 2×参数=28GB → 更大!
  - ZeRO-3 → 分片 → 每GPU只需1/N → 训练时才需全量

Checkpoint策略:
1. **同步barrier**: 所有GPU→同一step→一起save → 一致但慢
2. **异步**: GPU独立save→不等barrier→快但需版本管理
3. **verl选择**: 异步+版本追踪 → 最新一致版本≤最新保存版本

Checkpoint vs OOM:
- 7B BF16模型+Adam → 14+28=42GB → 单GPU OOM!
- ZeRO-3 → 分片 → 每GPU只需42/8=5.25GB → 可行!
- INT4模型 → 3.5GB → 单GPU也可行 → 推理用

## 5. Model Registry — 模型版本管理

Model Registry模式:
- **HF Hub**: 模型仓库 → safetensors+config+tokenizer → 版本化
- **MLflow**: 实验追踪 → 模型+metrics+params → 全生命周期
- **TorchServe**: 模型存储 → MAR格式 → 多版本 → 热切换
- **vLLM**: HF download → 本地缓存 → load → serve → 简单!

部署数据流:
  checkpoint → convert → quantize → safetensors → push → HF Hub
  vLLM启动 → HF download → load → allocate KV → serve → done!

RTX 4090部署流程:
  1. HF download → 本地SSD → 7B INT4 = 3.5GB → 30s
  2. load到GPU → allocate KV cache → 剩余20GB → B=118
  3. serve → continuous batching → 4,791 tok/s → 生产!

## 6. Feature Store — 实时特征服务

Feature Store架构:
- **在线store**: Redis → 实时特征 → 低延迟 → 推理用
- **离线store**: HDFS/S3 → 批量特征 → 训练用
- **Feast**: 开源 → online+offline → 统一API → 标准化

AI Serving中的Feature Store:
- RAG推理 → embedding → FAISS → 特征检索 → 47.5ms pipeline
- Agent推理 → feature lookup → Redis → <1ms → 快!
- 训练 → feature batch → offline store → DataLoader → 慢但全量

数据一致性:
- Online(Redis): AP → 快但不一致 → 推理可接受
- Offline(HDFS): CP → 慢但一致 → 训练必须
- Feast: 双写 → online+offline → 一致性挑战!

## 7. Core Laws — 数据流核心定律

1. **Incremental-Transfer Law**: CachedRequestData → 增量传输 → 减少IPC/ZMQ通信 → 关键优化!
   - 首次: 全量 → 之后: 增量 → 通信量↓90%

2. **Serialization-Cost Law**: pickle=慢+不安全 → safetensors=快+安全 → 生产必须!
   - checkpoint用pickle → 但推理模型用safetensors → 格式选择决定加载速度!

3. **Compression-Ratio Law**: BF16→FP8→INT4 → 50%→75%省 → 但需fused kernel!
   - Python dequant=20x慢 → fused kernel消除 → 量化才有效

4. **Checkpoint-Consistency Law**: 同步=一致但慢 → 异步=快但需版本管理!
   - verl选择异步 → 版本追踪 → 最新一致版本恢复 → 生产策略!

5. **Transfer-Bottleneck Law**: PCIe=12GB/s → 模型加载瓶颈 → NVLink=300GB/s → 生产理想!
   - RTX 4090: PCIe only → 模型加载INT4=0.3s → 可接受!

6. **Feature-Store-Dual-Write Law**: online=AP(快) → offline=CP(一致) → 双写挑战!
   - 推理用online → 训练用offline → 一致性是挑战 → Feast解决!

7. **Data-Flow-Debugging Law**: metrics→log→trace→profile → 逐层深入数据流!
   - 数据流问题 → 先查metrics → 再查log → 再trace → 最后profile → 逐步定位!

## 关键参考

- vLLM V1 SchedulerOutput: vllm/v1/core/sched/output.py → NewRequestData + CachedRequestData
- vLLM V1 outputs: vllm/v1/outputs.py → LogprobsLists/Tensors + ModelRunnerOutput
- SafeTensors: HuggingFace标准 → 无pickle → 安全快速加载
- verl checkpoint: 异步 → 版本管理 → 最新一致版本恢复
- Feast: Feature Store → online(Redis)+offline(HDFS) → 统一API
- MLflow: Model Registry → 实验追踪 → 全生命周期管理
- PagedAttention: block_size=16 → 按需分配 → KV cache零碎片