# verl Async Rollout Architecture Deep Dive — vLLM/SGLang Integration

> 2026-06-07 | verl v0.9异步rollout架构: Ray actor + load balancer + sticky session + sleep/wake权重同步

## 架构概览

```
verl异步rollout架构(2025-2026最新):

┌───────────────────────────────────────────────────────────────────────────┐
│  RayPPOTrainer (orchestrator)                                             │
│    ↓ 14步 pipeline (data→rollout→reward→logprob→adv→actor→...)           │
│                                                                           │
│  ┌─ AgentLoopWorker ──────────────────────────────────────────────┐       │
│  │  LLMServerClient → GlobalLoadBalancer → vLLMReplica           │       │
│  │    ↓ request_id + prompt_ids                                    │       │
│  │  vLLMHttpServer.generate() → AsyncLLM → token_ids + logprobs  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  权重同步: ServerAdapter.update_weights() → BucketedWeightSender          │
│    → ZMQ IPC / SHM → vLLM worker → update_weights_from_ipc               │
│                                                                           │
│  Sleep/Wake: sleep(level=2) → offload权重+KV → 训练空间 →                │
│              wake_up(tags=['weights','kv_cache']) → 恢复推理空间          │
└───────────────────────────────────────────────────────────────────────────┘

→ 3种RolloutMode: HYBRID(同进程)+COLOCATED(同GPU不同进程)+STANDALONE(独立GPU)
→ HYBRID: 训练+推理在同一Ray placement group → sleep/wake切换GPU内存
→ COLOCATED: 训练+推理不同进程但共享GPU → 适用于GRM(LLM judge)
→ STANDALONE: 推理独立GPU → off-policy场景 → 无sleep/wake开销

→ **关键设计**: SPMD模式已废弃(PR #4411) → 全部使用异步server接口
→ ServerAdapter不支持同步generate_sequences() → 必须用LLMServerClient
```

## LLMServerClient — 负载均衡 + Sticky Session

```python
# verl/workers/rollout/llm_server.py

class LLMServerClient:
    """管理多个OpenAI兼容LLM server → 负载均衡+sticky session"""

    async def generate(self, request_id, prompt_ids, sampling_params, ...):
        # 1. Acquire server via global load balancer (sticky session + least-loaded)
        server_id, server = await self._load_balancer.acquire_server.remote(request_id=request_id)

        # 2. Generate via Ray remote call to vLLMHttpServer
        output = await server.generate.remote(
            request_id=uuid4().hex,  # 新request_id(每轮新)
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

        # 3. Release server (fire-and-forget → counter decrement)
        self._load_balancer.release_server.remote(server_id=server_id)

        return output  # TokenOutput(token_ids, log_probs, stop_reason)

→ Sticky session: LRU cache map request_id → server_id
  → 多轮对话路由到同一server → prefix caching自动生效!
  → 如果server被移除 → cache自动失效 → 重选最小负载server

→ Least-loaded selection: min(inflight_requests) → 自动负载均衡
→ GlobalRequestLoadBalancer: Ray actor → 全局协调 → 跨Worker共享

→ **与GRPO训练的关系**:
  → GRPO n=8采样 → 同prompt的8个request → sticky session → 同server
  → prefix caching自动生效 → 8个request共享prompt KV → 58%节省!
  → 这就是为什么vLLM async rollout比Python rollout快2-5x!
```

## vLLMHttpServer — 推理引擎

```python
# verl/workers/rollout/vllm_rollout/vllm_async_server.py

class vLLMHttpServer:
    """vLLM HTTP server → 相当于 `vllm serve` 命令"""

    async def launch_server(self):
        # 构建vLLM CLI args → 包括TP/DP/PP/LoRA/量化/cuda_graph等
        engine_args = AsyncEngineArgs.from_cli_args(server_args)
        vllm_config = engine_args.create_engine_config()

        # 创建AsyncLLM引擎(V1引擎)
        engine_client = AsyncLLM.from_vllm_config(vllm_config)

        # Monkey-patch vocab size(适配tokenizer)
        await engine_client.collective_rpc(
            method="monkey_patch_model",
            kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )

        # 启动HTTP server(uvicorn) → OpenAI兼容API
        self._server_port, self._server_task = await run_uvicorn(app)

    async def generate(self, prompt_ids, sampling_params, request_id):
        # 1. 构建TokensPrompt → token_ids + multi_modal_data
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)

        # 2. 创建SamplingParams → max_tokens, temperature, top_k, top_p
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

        # 3. 调用AsyncLLM.generate() → 流式输出
        generator = self.engine.generate(prompt, sampling_params, request_id)

        # 4. 收集最终输出 → TokenOutput
        final_res = None
        async for output in generator:
            final_res = output

        # 5. 提取token_ids + logprobs + routed_experts
        token_ids = final_res.outputs[0].token_ids
        log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(...)]

        return TokenOutput(token_ids, log_probs, stop_reason="completed")

→ **关键**: vLLM V1引擎处理整个rollout → 消除Python-level for循环!
  → Python rollout: for step in range(max_len): model.forward → softmax → multinomial → cat → loop
  → vLLM rollout: engine.generate(prompt_ids) → AsyncLLM处理所有 → 一次性返回所有token_ids

→ **这就是为什么vLLM rollout比Python快2-5x**:
  → vLLM内部: continuous batching + paged attention + CUDA Graph + prefix caching
  → Python rollout: 逐token串行 → kernel launch开销 × max_len
  → vLLM: 批量处理 → kernel launch开销 × 1(整个sequence)

→ LoRA支持: lora_as_adapter=True → LoRARequest(lora_name, lora_int_id, lora_path)
  → LoRA不merge → vLLM直接使用LoRA adapter推理 → 无merge开销
  → sleep(level=1)只释放adapter → faster wake-up
```

## Sleep/Wake — GPU内存时分复用

```python
# Sleep: 释放推理引擎GPU内存 → 给训练空间
async def sleep(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        # LoRA: sleep level=1 (只释放adapter)
        # Full weights: sleep level=2 (释放权重+KV+全部GPU内存)
        if self.lora_as_adapter:
            sleep_level = 1
        else:
            sleep_level = 2  # ← GRPO标准模式
        await self.engine.sleep(level=sleep_level)

    elif self.rolloutMode == RolloutMode.COLOCATED:
        await self.engine.sleep(level=1)  # 只释放推理引擎

    elif self.rollout_mode == RolloutMode.STANDALONE:
        # 不sleep → 推理独立GPU → 不需要释放

# Wake: 恢复推理引擎GPU内存 → 推理空间
async def wake_up(self, tags=['kv_cache', 'weights']):
    if self.rollout_mode == RolloutMode.HYBRID:
        await self.engine.wake_up(tags=tags)
        await self.engine.reset_prefix_cache()  # 清除旧KV → 防止数据泄漏

    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.wake_up(tags=tags)
        await self.engine.reset_prefix_cache()

→ Sleep/Wake时序:
  1. 训练前: rollout.sleep(level=2) → offload权重+KV → GPU空出给FSDP训练
  2. 训练中: FSDP训练 → 占用全部GPU内存 → rollout进程睡眠
  3. 训练后: rollout.wake_up() → 加载新权重+KV → 准备推理
  4. 推理中: vLLM generate → 占用GPU → 训练进程暂停

→ **内存时分复用**: 同GPU → 训练时训练用 → 推理时推理用 → 省50%GPU!
  → GRPO只需2模型(actor+ref) → 同GPU → 省50% vs PPO需要4模型

→ Sleep level:
  → Level 0: 暂停调度 → 不释放GPU内存(最快恢复)
  → Level 1: offload权重 → GPU空出大部分 → KV仍保留
  → Level 2: 丢弃全部GPU内存 → 完全释放 → 训练获得最大空间

→ **Prefix cache必须在wake_up后reset**:
  → 新权重 → 旧KV不匹配 → 继续使用会产出错误token → 必须reset!
  → reset_connector=True → 同时断开MooncakeStoreConnector(外部KV存储)
```

## 权重同步 — BucketedWeightSender (ZMQ IPC)

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py

class ServerAdapter(BaseRollout):
    async def update_weights(self, weights, global_steps):
        # 1. 通过Ray collective_rpc告诉vLLM server准备接收
        future = await self._execute_method("update_weights_from_ipc", non_block=True)

        # 2. 创建BucketedWeightSender → ZMQ IPC通道
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,  # ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{rank}-rank-{local_rank}.sock
            bucket_size_mb=config.checkpoint_engine.update_weights_bucket_megabytes,
            use_shm=self.use_shm,
        )

        # 3. 异步发送权重 → 分桶传输 → 减少IPC开销
        await sender.async_send_weights(weights)

        # 4. 等待vLLM server确认接收
        await future

        # 5. 清除KV cache + 设置global_steps
        await self.server_handle.clear_kv_cache.remote()
        await self.server_handle.set_global_steps.remote(global_steps)

→ 权重同步数据流:
  FSDP训练 → model.state_dict() → (name, tensor) generator
    → BucketedWeightSender → ZMQ IPC → vLLM worker
    → vLLM worker.update_weights_from_ipc() → 更新模型参数
    → 清除KV cache → 设置global_steps

→ **IPC vs SHM**:
  → CUDA IPC: 直接GPU内存传输 → 零拷贝 → 最快 → 需要P2P支持
  → Shared Memory: CPU staging → 2次拷贝 → 较慢 → 兼容性好
  → ROCm/Ascend: IPC可能不支持 → fallback SHM

→ **分桶传输**: bucket_size_mb配置 → 大权重分桶 → 流式传输 → 减少内存峰值

→ zmq_handle: ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{rank}-rank-{local_rank}.sock
  → 包含Ray job_id → 防止两个verl作业在同一节点碰撞
  → 包含replica_rank + local_rank → 防止多replica冲突
```

## 数据流 — 从Prompt到Training

```
GRPO异步rollout完整数据流(最新架构):

1. **Data Loading**: DataLoader → prompts (text/ids)
   → 每个prompt = (input_ids_tensor, prompt_length)

2. **Rollout Request**:
   AgentLoopWorker → LLMServerClient.generate()
     → request_id = unique_id (sticky session)
     → prompt_ids = [token_ids_list] (每个prompt)
     → sampling_params = {temperature, top_k, top_p, max_tokens}

   → GlobalLoadBalancer → acquire_server(request_id)
     → sticky session → 同prompt路由到同server → prefix caching

   → vLLMHttpServer.generate()
     → AsyncLLM V1引擎处理 → continuous batching
     → 返回 TokenOutput(token_ids, log_probs, stop_reason)

   → **n=8 GRPO**: 同prompt的8个request → 同server → prefix caching生效!

3. **Reward Computation**:
   → RewardManager → rule-based或model-based
   → 每个response → reward (float)
   → 组归一化: mean_r, std_r, advantage = (r - mean_r) / std_r

4. **Logprob Computation**:
   → 训练模型(full_ids) → log_softmax → gather → per-token log_probs
   → **这里才是torch.compile的收益点**: logprob forward → 1.3-1.4x加速
   → 但logprob只占19% → 整体收益有限

5. **Training Update**:
   → GRPO loss = -advantage × Σ(log_probs)
   → FSDP backward → clip_grad → optimizer.step
   → 新权重 → ServerAdapter.update_weights() → vLLM server

→ **总时间分解(7B模型, 8×A100)**:
   Rollout: ~2-5s (vLLM async, n=64)
   Reward: ~0.5s (rule-based)
   Logprob: ~1s (training model forward)
   Loss+Bwd: ~2s (FSDP backward)
   Total: ~5-8s per GRPO step

→ **vLLM解决了Python rollout瓶颈(74%)!**:
   Python rollout: 逐token串行 × n × max_len → 慢
   vLLM rollout: continuous batching × 一次 → 快2-5x!
```

## 与torch.compile Benchmark结果的综合

```
之前RTX 4090实测发现:
  GRPO Python rollout占step 74% → 瓶颈
  torch.compile只能优化40%(GPU部分) → E2E 1.39x
  Triton fused advantage 40x → E2E 0.99x(advantage占比<3%)

verl架构如何解决:
  → vLLM async rollout → 消除Python rollout瓶颈 → 2-5x
  → torch.compile → 优化logprob/loss/bwd → 1.2-1.4x (这部分占比19+5=24%)
  → 组合: vLLM(74%→快2-5x) + compile(24%→快1.3x) = E2E 1.5-2.5x!

→ **生产GRPO训练加速路线**:
  1. 用vLLM/SGLang async rollout → 解决74%瓶颈 → 2-5x
  2. 用torch.compile优化训练部分 → 1.3x → 总1.5-2.5x
  3. Prefix Sharing → n=8时58%计算+88%KV节省 → 再加速1.5-2x
  4. FSDP2 + compile → 通信-计算重叠 → 1.2-1.5x

→ 组合效果: 2-5x × 1.3x × 1.5-2x × 1.2x = 3.5-15x E2E!
  → 生产GRPO训练: verl+vLLM+compile+PS+FSDP2 → 3.5-15x加速
  → 这比任何单一优化都更有效 → 组合优化是关键!
```

## 关键文件映射

```
verl v0.9异步rollout架构:

| 文件                                  | 功能                     | 行数 |
| verl/workers/rollout/llm_server.py    | LB+Client+Manager        | 374  |
| verl/workers/rollout/vllm_rollout/    | vLLM rollout实现          |      |
|   vllm_async_server.py                | HTTP server+generate      | 1130 |
|   vllm_rollout.py                     | ServerAdapter(weight sync)| 223  |
|   bucketed_weight_transfer.py         | ZMQ IPC weight sender     | ~200 |
|   utils.py                            | ColocateWorkerExtension   | ~150 |
| verl/workers/rollout/replica.py        | RolloutReplica+TokenOutput| ~200 |
| verl/workers/rollout/base.py           | BaseRollout abstract      | ~50  |

→ SPMD模式已废弃(PR #4411) → 全部使用异步server接口
→ ServerAdapter不支持generate_sequences() → raise NotImplementedError
→ 只能通过LLMServerClient + vLLMHttpServer.generate() 异步推理
```

## 参考资料

- verl源码: verl/workers/rollout/vllm_rollout/
- RTX 4090 GRPO benchmark: notebook/fundamentals/torch-compile-grpo-training-rtx4090.md
- Triton fused kernel: notebook/fundamentals/triton-fused-grpo-kernel-rtx4090.md
- verl分布式架构: notebook/projects/distributed-rl-training-verl-architecture.md