# verl x vLLM MRv2 Interaction 深度阅读

> 2026-06-16 | 源码: verl v0.8.0 (2026-06-01), vLLM v0.23.0 (2026-06-12)
> 核心: verl 对 MRv2 完全零感知 → vLLM oracle 自动为 eligible dense model 启用 MRv2 → 升级时静默切换!
> ★★★★★ verl 无任何 VLLM_USE_V2_MODEL_RUNNER 引用 → BF16 Qwen3/Llama/Mistral 升级到 v0.23 自动 MRv2 → 未测试!
> ★★★★★ RTX 4090 INT4 仍走 MRv1 → 不受影响 → 但 BF16 小模型 GRPO 有风险!

## 1. Oracle: vLLM 如何决定 MRv1 vs MRv2

```
★★★★★ Oracle = vLLM 内部的3层自动决策系统!

源码: vllm/config/vllm.py L519-558

Layer 1: VLLM_USE_V2_MODEL_RUNNER 环境变量
  → envs.VLLM_USE_V2_MODEL_RUNNER (vllm/envs.py L267)
  → 如果设置(True/False): 强制覆盖一切 → 不再检查!
  → 如果 None(未设置): 继续到 Layer 2

Layer 2: _is_default_v2_model_runner_model() → ★★★ 关键函数!
  → model_config.runner_type != "generate" → False (pooling model排除)
  → architecture NOT in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES → False
  → model_config.is_moe → False (MoE排除)
  → model_config.is_quantized → False (INT4/GPTQ/FP8排除)
  → 全通过 → True → 进入 Layer 3

Layer 3: _get_v2_model_runner_unsupported_features() → 完整检查
  → 检查 Triton 可用性 (HAS_TRITON) → 无 Triton → False
  → 检查10+个不支持特性 → 详见第8节
  → ANY不支持 → log warning → MRv1 fallback
  → ALL支持 → True → MRv2 自动激活!

★★★★★ DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES (v0.23.0 源码确认):
  源码: vllm/config/vllm.py L69-75

  DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({
      "LlamaForCausalLM",
      "MistralForCausalLM",
      "Qwen3ForCausalLM",
  })

  → ★★★★★ 只有这3个架构! 不是5个!
  → ★★★ DeepseekV2ForCausalLM → NOT in DEFAULT_V2 → MRv1
  → ★★★ Qwen2MoeForCausalLM → NOT in DEFAULT_V2 → MRv1
  → ★★★ Qwen2ForCausalLM → NOT in DEFAULT_V2 → MRv1
  → 只有 Llama, Mistral, Qwen3 dense BF16 自动 MRv2!

★★★★ Oracle 是 vLLM 内部机制 → verl 看不到这个决策过程!
★★★★ 升级 vLLM → eligible model 静默切换 → verl 无感知!
```

## 2. verl 对 MRv2 的完全零感知

```
★★★★★ KEY FINDING: verl 有 ZERO MRv2 知觉!

源码搜索证据:
  1. grep VLLM_USE_V2_MODEL_RUNNER → verl 全树 → 0 matches
  2. grep MRv2/ModelRunnerV2/model_runner_v2 → verl → 0 matches in RL code
  3. verl GitHub issues/PRs 搜索 MRv2 → 无专项issue
  4. verl v0.8.0 release notes → 无 MRv2 相关变更
  5. verl/utils/vllm/ → 无 MRv2 补丁/适配层

verl 对 model_runner 的唯一引用 (verl/workers/rollout/vllm_rollout/utils.py):
  → self.model_runner.model → 获取底层 nn.Module
  → self.model_runner.vllm_config → 获取配置
  → self.model_runner.drafter → speculative decode model
  → ★★★ self.model_runner 在 MRv1 和 MRv2 下都是同一属性名!
  → ★★★ MRv1 GPUModelRunner.model → nn.Module ✓
  → ★★★ MRv2 GPUModelRunnerV2.model → nn.Module ✓ (同名属性)
  → → verl 的 model_runner.model 引用 → 两种 runner 都可用!

★★★ 但: MRv2 的 model_runner 没有 drafter 属性?
  → MRv2 用 BaseSpeculator hierarchy → drafter 可能在不同位置
  → verl: getattr(self.model_runner, "drafter", None) → None fallback → 安全!
  → ★★★ MTP drafter weight sync → verl 有 _use_mtp_drafter_weight_sync() → MRv2 兼容

★★★★★ 核心问题不是属性名 → 而是执行行为差异!
  → verl 不检查 vLLM 到底用了哪个 runner
  → 无日志, 无 assertion, 无配置选项
  → 升级 vLLM 后 → eligible model 静默切换 → 未测试配置!
```

## 3. verl 启动 vLLM: Oracle 决策在何处发生

```
★★★★ verl → vLLM async server 启动路径:

Step 1: verl RolloutConfig → build_cli_args_from_config()
  → utils.py L321-359: 将 config dict → CLI args list
  → 传入: dtype, load_format, enable_sleep_mode, enforce_eager,
    compilation_config (cudagraph_mode), quantization, logprobs_mode,
    tensor_parallel_size, max_model_len
  → ★★★ 不传入 VLLM_USE_V2_MODEL_RUNNER → 让 oracle 自行决定!

Step 2: AsyncEngineArgs.from_cli_args(args) → create_engine_config()
  → vLLM 内部创建 VllmConfig
  → ★★★ Oracle 在这里运行! use_v2_model_runner property computed!
  → 3层决策 → Qwen3 BF16 dense → MRv2 = True!
  → INT4/GPTQ → MRv2 = False (is_quantized blocks)

Step 3: AsyncLLM.from_vllm_config(vllm_config)
  → EngineCore 创建 → GPUWorker 初始化
  → GPUWorker.__init__(vllm_config) → self.use_v2_model_runner = vllm_config.use_v2_model_runner

Step 4: GPUWorker.init_worker()
  → if self.use_v2_model_runner: import GPUModelRunnerV2 → MRv2 path
  → else: import GPUModelRunner → MRv1 path (已有, 已测试)
  → ★★★ GPUWorker 用同一个类名 GPUModelRunner → 但实际是不同实现!
  → L332-333: "HACK(woosuk): temporary fix to avoid type errors"
  → self.model_runner: GPUModelRunner = GPUModelRunnerV2(...) # type: ignore

★★★★ Oracle 在 vLLM config 创建内部 → verl 看不到!
★★★★ verl 无法得知 vLLM 选择了哪个 runner!
★★★★ 无 logging, 无 assertion, 无 config option 控制此决策!
```

## 4. RTX 4090 Model-by-Model MRv2 影响矩阵

```
★★★★★ 升级到 vLLM v0.23.0 的具体影响:

| Model                    | Architecture         | Quantized | MoE | MRv2? | Runner | verl Impact       |
|--------------------------|----------------------|-----------|-----|-------|---------|-------------------|
| Qwen3-1.7B BF16         | Qwen3ForCausalLM     | No        | No  | YES★★ | MRv2    | ★★★ AUTO-SWITCH!  |
| Qwen3-4B BF16           | Qwen3ForCausalLM     | No        | No  | YES★★ | MRv2    | ★★★ AUTO-SWITCH!  |
| Qwen3-8B BF16           | Qwen3ForCausalLM     | No        | No  | YES★★ | MRv2    | ★★★ AUTO-SWITCH!  |
| Llama-3.1-8B BF16       | LlamaForCausalLM     | No        | No  | YES★★ | MRv2    | ★★★ AUTO-SWITCH!  |
| Mistral-7B-v0.3 BF16    | MistralForCausalLM   | No        | No  | YES★★ | MRv2    | ★★★ AUTO-SWITCH!  |
| Qwen3-8B INT4/GPTQ      | Qwen3ForCausalLM     | Yes★★     | No  | NO★★★ | MRv1    | ★ SAME AS BEFORE  |
| Qwen2.5-7B BF16         | Qwen2ForCausalLM     | No        | No  | NO★★★ | MRv1    | ★ SAME AS BEFORE  |
| Qwen3-30B-A3B MoE BF16  | Qwen3MoeForCausalLM  | No        | Yes | NO★★★ | MRv1    | ★ SAME AS BEFORE  |
| DeepSeek-V2 BF16        | DeepseekV2ForCausalLM| No        | Yes | NO★★★ | MRv1    | ★ SAME AS BEFORE  |
| Llama-3.1-8B FP8        | LlamaForCausalLM     | Yes★★     | No  | NO★★★ | MRv1    | ★ SAME AS BEFORE  |

★★★★★ 关键结论:
  → ★★★★★ 只有 Qwen3, Llama, Mistral BF16 dense 自动切换 MRv2!
  → ★★★★★ INT4/GPTQ → is_quantized=True → MRv1 强制 → 不受影响!
  → ★★★★★ MoE → is_moe=True → MRv1 强制 → 不受影响!
  → ★★★★★ Qwen2/Qwen2.5 → 不在 DEFAULT_V2 → MRv1 → 不受影响!
  → ★★★★★ DeepseekV2 → 不在 DEFAULT_V2 → MRv1 → 不受影响!
```

## 5. Sleep/Wake Cycle: MRv1 vs MRv2 Worker 层兼容性

```
★★★★★ Sleep/wake 在 GPUWorker 层 → 不是 model_runner 层!

源码: vllm/v1/worker/gpu_worker.py L165-200

GPUWorker.sleep(level):
  → MRv1 和 MRv2 共用 SAME sleep() method
  → level=1: allocator.sleep(offload_tags=("weights",)) → 只释放权重
  → level=2: allocator.sleep(offload_tags=()) → 释放权重+KV+所有buffer
  → level=2: 保存 model.named_buffers() 到 CPU → 恢复时用
  → ★★★ 两种 runner 共用 → COMPATIBLE!

GPUWorker.wake_up(tags):
  → allocator.wake_up(tags) → 恢复指定 tag 的 GPU 内存
  → 恢复 model buffers from CPU (level=2 sleep 后)
  → 然后调用 model_runner.post_kv_cache_wake_up() → runner-specific hook

post_kv_cache_wake_up() 差异:
  MRv1 (gpu_model_runner.py):
    → self.init_fp8_kv_scales() → 重置 FP8 KV scale 为 1.0
    → ★ RTX 4090: FP8 KV scale reset → 但 FP8 KV 在 SM89 本身有问题

  MRv2 (gpu/model_runner.py):
    → self.block_tables.init_block_table_layout_tensors() → 重新初始化 block table
    → ★ 不同 hook → 但相同目的: wake 后恢复状态

★★★★ 两个 hook 都从同一 GPUWorker.wake_up() 调用 → verl 兼容!
★★★★ Sleep/wake = worker 层机制 → MRv1/MRv2 都支持!

verl HYBRID sleep/wake flow:
  1. engine.sleep(level=1/2) → 释放 GPU 给 FSDP 训练
  2. FSDP 训练 → actor 更新权重
  3. engine.wake_up(tags=["weights"]) → 恢复权重到 GPU
  4. update_weights_from_ipc() → ZMQ bucketed 传输新权重
  5. engine.wake_up(tags=["kv_cache"]) → 恢复 KV cache 内存
  6. Generation → vLLM 产出 rollout tokens
  7. 回到 Step 1

★★★ 此流程 MRv1/MRv2 兼容:
  → sleep/wake = worker 层 → 不是 runner 层
  → post_kv_cache_wake_up() = hook → 两种 runner 都实现
  → verl 无 runner-specific sleep/wake logic
```

## 6. Weight Sync: verl Bucketed IPC Transfer + MRv2

```
★★★★ verl weight update 路径 (naive/async):

源码: verl/workers/rollout/vllm_rollout/utils.py L106-289

vLLMColocateWorkerExtension.update_weights_from_ipc():
  → BucketedWeightReceiver 通过 ZMQ IPC 接收权重
  → 每个 bucket 调用 _update_weights()
  → _update_weights() → model.load_weights(weights) → 标准 BF16 权重
  → 或 load_quanted_weights() → FP8 权重转换
  → model = self.model_runner.model → ★★★ nn.Module → runner无关!

★★★★ Weight sync COMPATIBLE with MRv2 因为:
  1. model.load_weights() 操作 nn.Module → 不是 runner
  2. MRv1 和 MRv2 包装同一 PyTorch model
  3. ZMQ IPC bucketed transfer = worker extension 层 → runner无关
  4. verl 无 runner-specific weight loading logic

★★★★★ Potential concern: CUDA graph invalidation!
  → MRv2 使用 Breakable CUDA Graph (BCG) → warmup 时 capture
  → weight update 后 → BCG segments 包含旧权重引用
  → vLLM 处理此情况: graphs invalidated → 下次 warmup re-capture
  → verl flow: sleep → wake → update_weights → generation
  → generation 时 → vLLM re-capture CUDA graphs if weights changed
  → ★★★★★ 应该可以工作 → 但在 verl RL training context 未测试!

★★★★ Additional concern: compilation_config interaction
  → verl 设置 cudagraph_mode = "FULL_AND_PIECEWISE" (default)
  → MRv2 的 BCG 替换 PIECEWISE 编译 → FULL 仍然 capture
  → compilation_config 在 vLLM config 创建时处理
  → MRv2 active 时 → BCG 取代 torch.compile PIECEWISE
  → ★★★ 应该OK → 但交互未明确测试!

★★★★ MTP drafter weight sync:
  → verl 有 _use_mtp_drafter_weight_sync() → self.model_runner.vllm_config.speculative_config
  → self.model_runner.drafter → getattr → None fallback → 安全
  → MRv2 speculator hierarchy: EagleSpeculator → thin subclass
  → ★★★ MTP weight sync 在 MRv2 下 → drafter.model 可能在不同位置
  → 但 getattr(self.model_runner, "drafter", None) → 可拿到 → OK
```

## 7. logprobs_mode: ★★★★★ 最关键兼容性检查

```
★★★★★ logprobs_mode = verl+MRv2 最重要兼容性指标!

verl RolloutConfig default:
  logprobs_mode: Optional[str] = "processed_logprobs"

vLLM v0.23.0 MRv2 unsupported features 检查:
  源码: vllm/config/vllm.py L2045-2050

  if model_config.logprobs_mode in ("raw_logits", "processed_logits"):
      unsupported.append(f"logprobs mode '{model_config.logprobs_mode}'")

  → "raw_logits" → UNSUPPORTED → oracle fallback MRv1
  → "processed_logits" → UNSUPPORTED → oracle fallback MRv1
  → ★★★★★ "processed_logprobs" → NOT in unsupported → MRv2 OK!

★★★★★ verl default "processed_logprobs" IS COMPATIBLE with MRv2!
  → 这是 vLLM 标准/default logprobs mode
  → MRv2 支持 → oracle 不触发 fallback
  → → ★★★★★ GRPO training 的 log_prob 计算不受影响!

★★★★ BUT: 如果用户设置 logprobs_mode 为 "processed_logits" 或 "raw_logits":
  → oracle 检测 → unsupported feature → silent fallback MRv1
  → 用户可能不知道自己实际在 MRv1 → 尽管 model eligible for MRv2
  → verl 无 validation 或 warning!

RTX 4090 GRPO training 需要的 logprobs:
  → log_prob computation in actor's compute_log_prob()
  → PPO/GRPO advantage calculation → 需要 per-token logprobs
  → "processed_logprobs" = 标准 mode → MRv2 compatible → ★★★★★ OK!
```

## 8. MRv2 Unsupported Features 完整列表

```
★★★★★ 源码: vllm/config/vllm.py L1982-2060

_get_v2_model_runner_unsupported_features() 完整检查:

| Feature                              | Unsupported | verl Default | verl Risk |
|--------------------------------------|-------------|--------------|-----------|
| processed_logprobs                   | NO ✓        | YES (default)| ★★★★★ OK |
| processed_logits                     | YES         | NO           | Safe      |
| raw_logits                           | YES         | NO           | Safe      |
| hybrid/mamba align cache mode        | YES         | NO           | Safe      |
| prefill context parallel >1          | YES         | NO           | Safe      |
| stock torch.compile                  | YES         | NO           | Safe      |
| SP with TP>1 (enable_sp)             | YES         | TP=2 but ★  | ★★★★     |
| ngram/ngram_gpu spec decode          | YES         | NO           | Safe      |
| non-eagle/mtp spec methods           | YES         | eagle only ★ | ★★★      |
| parallel drafting (PEagle)           | YES         | NO           | Safe      |
| EAGLE3 + PP>1                        | YES         | PP=1 RTX 4090| Safe      |
| dual batch overlap (dbo)             | YES         | NO           | Safe      |
| routed experts capture               | YES         | NO           | Safe      |
| custom logits processors             | YES         | NO           | Safe      |
| prompt_embeds                        | YES         | NO           | Safe      |
| KV sharing fast prefill              | YES         | NO           | Safe      |
| EC transfer (PD disaggregation)      | YES         | NO           | Safe      |
| mamba_cache_mode='align'             | YES         | NO           | Safe      |
| LoRA                                 | NO ✓★★      | Optional     | ★★★★ OK  |

★★★★★ TP>1 + SP: verl default tensor_parallel_size=2
  → 如果用户 enable SP + TP>1 → MRv2 blocks → silent MRv1 fallback
  → SP NOT default → Safe for default config
  → ★★★★★ 如果 enable SP + TP>1 → 用户不知道自己实际 MRv1!

★★★★ LoRA: MRv2 继承 LoRAModelRunnerMixin → SUPPORTED ✓
  → verl LoRA integration: TensorLoRARequest → add_lora/remove_lora
  → LoRA weight sync → runner-independent → ★★★★★ MRv2 compatible!

★★★★★ thinking_token_budget:
  → vllm/v1/engine/input_processor.py L112-116
  → MRv2 不支持 thinking_token_budget → 报 ValueError
  → "Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 to use thinking_token_budget"
  → ★★★ verl GRPO 不用 thinking_token_budget → Safe
  → ★★★★★ 但: Qwen3 thinking mode + MRv2 → 可能冲突!
```

## 9. Breakable CUDA Graph (BCG) + verl 交互

```
★★★★★ BCG = MRv2 最具影响力的变更 → 对 verl!

BCG (v0.23.0):
  → 替换 torch.compile + Inductor CUDA graph capture
  → 单 capture context → 遇到 @eager_break_during_capture → break
  → 生成 segments: graph.replay + eager_fn → 顺序 replay
  → ★★★★★ 无 torch.compile 依赖 → SM89 OK! → SM无关!

BCG vs verl compilation_config:
  → verl 设置: cudagraph_mode = "FULL_AND_PIECEWISE"
  → MRv2 BCG 替换 PIECEWISE torch.compile 路径
  → ★★★ MRv2 active → BCG 接管 CUDA graph 管理
  → FULL mode 仍然 capture per decode batch size → same as MRv1

★★★★★ BCG + verl weight update cycle:
  → update_weights_from_ipc() → model weights 改变
  → BCG segments 引用旧 weight tensor addresses
  → vLLM: graphs invalidated → next iteration re-capture
  → ★★★★★ weight update → graph invalidation → recapture → 应该OK
  → ★★★★★ 但: 此 cycle 在 verl RL training context 未测试!

★★★★★ BCG + verl sleep/wake cycle:
  → sleep(level=2) → weights offload → wake → weights 可能不同地址
  → CUDA graphs 引用特定 tensor addresses → wake 后需要 recapture
  → MRv2 处理: post_kv_cache_wake_up → 重新初始化 state
  → ★★★★★ FULL recapture after wake → MRv2 标准机制 → 应该OK
  → ★★★★★ 但: verl 的 HYBRID flow → 未验证!

★★★★ Risk Assessment:
  → BCG + weight update: SHOULD work (vLLM 设计为此)
  → BCG + sleep/wake: SHOULD work (标准 GPUWorker 机制)
  → BCG + verl specific HYBRID training flow: ★★★★★ UNTESTED!
  → ★★★★★ 建议: BF16 Qwen3 GRPO with MRv2 → 先测试再生产!
```

## 10. MRv2 Scheduler 交互差异

```
★★★★ Scheduler 对 MRv2 的处理不同!

源码: vllm/v1/core/sched/scheduler.py L894-903

MRv2 scheduler output 差异:
  → if self.use_v2_model_runner:
      scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
      scheduled_resumed_reqs = []  → ★★★ MRv2 合并 new + resumed!
      new_reqs_data = [NewRequestData.from_request(...)] → ★ 专用格式!

  → MRv1:
      scheduled_new_reqs 和 scheduled_resumed_reqs 分开传递
      → SchedulerOutput 含两个独立列表

★★★★ MRv2 scheduler 用 NewRequestData:
  → from_request(req, block_ids, all_token_ids)
  → ★★★ 包含 block_ids 和 all_token_ids → MRv1 不用此格式
  → → scheduler_output 结构不同 → 但 verl 不直接处理 scheduler_output!

★★★★ verl 只通过 AsyncLLM API 交互 → 不看 scheduler internals
  → verl → engine.generate(prompts, sampling_params) → AsyncLLM 处理
  → scheduler → model_runner → execute_model → 对 verl 透明
  → ★★★★★ Scheduler 差异 → 对 verl 无直接影响!

★★★★ BUT: generation timing 可能不同:
  → MRv2 2-phase forward/sample → execute_model → forward only
  → → sample_tokens() 单独执行 → timing 差异
  → → verl 的 generation timeout 可能需要调整
  → ★★★ 但: AsyncLLM 封装了此差异 → verl 不应受影响
```

## 11. verl v0.8.0 → v0.23.0 升级: MRv2 专项分析

```
★★★★ verl v0.8.0 (2026-06-01) vLLM 版本状态:

  → Stable vLLM 版本: v0.20.2 (NOT v0.23.0!)
  → ★★★★★ MRv2 在 v0.23.0 引入 → verl v0.8.0 predates MRv2!
  → verl v0.8.0 release notes → 无 MRv2 相关变更
  → verl v0.8.0 代码 → 无 MRv2 适配层

★★★★ verl v0.8.0 对 vLLM 的关键改动 (relevant to future MRv2):
  → #6456: engine.sleep() 替代 collective_rpc("sleep")
    → ★★★★★ 确保 sleep 传播到 ALL DP workers
    → ★★★★★ 对 MRv2 重要 → MRv2 DP handling 与 MRv1 不同
    → ★★★★★ 无此 fix → sleep 不会到达 MRv2 workers
    → ★★★ v0.23.0 已包含此 fix → 升级路径 sleep/wake OK

  → #6091: NCCL/NIXL checkpoint engine → 大权重分 chunk
    → Weight transfer optimization → runner-independent → OK

  → #6373: MooncakeStoreConnector → weight update hard-reset
    → KV store invalidation → runner-independent → OK

★★★★★ 升级路径:
  → v0.20.2 → v0.23.0 → MRv2 引入 → eligible model 静默切换
  → Sleep/wake: #6456 fix → v0.23.0 包含 → 升级 OK
  → Weight sync: model.load_weights() → runner-independent → 升级 OK
  → ★★★★★ MRv2 本身 → 新功能 → 需要 explicit testing!

★★★★★ 升级风险矩阵:
  → INT4/GPTQ 模型 → MRv1 → ★★★★★ ZERO risk → 升级安全
  → MoE 模型 → MRv1 → ★★★★★ ZERO risk → 升级安全
  → Qwen2/Qwen2.5 → MRv1 → ★★★★★ ZERO risk → 升级安全
  → Qwen3/Llama/Mistral BF16 → MRv2 auto → ★★★★ MEDIUM risk → 需测试
```

## 12. verl → vLLM MRv2 E2E Interaction Flow

```
★★★★★ 完整执行路径:

Step 1: verl config → vLLM CLI args
  RolloutConfig → build_cli_args_from_config() → CLI args
  → 传入: dtype, load_format, enable_sleep_mode, enforce_eager,
    compilation_config, quantization, logprobs_mode, TP size, max_model_len
  → ★★★★★ 不传入 VLLM_USE_V2_MODEL_RUNNER → oracle 自行决定!

Step 2: vLLM config creation → oracle runs
  AsyncEngineArgs.from_cli_args() → create_engine_config()
  → model_config created → architecture, is_quantized, is_moe
  → use_v2_model_runner property → 3层决策
  → ★★★ Qwen3 BF16 dense + no unsupported → MRv2 = True
  → ★★★ INT4/GPTQ → MRv2 = False (is_quantized)
  → ★★★ Llama BF16 → MRv2 = True
  → ★★★ Mistral BF16 → MRv2 = True

Step 3: Engine creation → worker gets runner type
  AsyncLLM.from_vllm_config(vllm_config) → EngineCore
  → GPUWorker.init_worker()
  → self.use_v2_model_runner = vllm_config.use_v2_model_runner
  → ★★★ If True: import GPUModelRunnerV2 → MRv2 path
  → ★★★ If False: import GPUModelRunner → MRv1 path (tested)
  → ★★★★★ L332-333: HACK comment → type: ignore → 临时修复!

Step 4: Model loading → same for both runners
  → Both runners call model.load_weights() → same nn.Module
  → verl: vLLMColocateWorkerExtension handles → runner-independent
  → ★★★★★ COMPATIBLE

Step 5: Warmup → ★★★★ DIFFERENT for MRv2!
  MRv1: capture FULL CUDA graphs per batch size + torch.compile PIECEWISE
  MRv2: capture FULL CUDA graphs per batch size + BCG (no torch.compile)
  → ★★★ BCG warmup timing 可能不同 → latency 差异
  → ★★★★★ 但: 对 verl 无直接影响 → warmup 是内部步骤

Step 6: Generation → ★★★★ DIFFERENT execution!
  MRv1: execute_model() → forward + sample → 一体 → monolithic
  MRv2: execute_model() → forward only → sample_tokens() 分离
  → ★★★★★ MRv2 2-phase → PP overlap (RTX 4090 TP=1 无影响)
  → ★★★★★ FlashInfer sampler → temperature>0 → GRPO 直接受益!
  → ★★★★★ sampling 质量 → rejection-based → 更快更准

Step 7: Sleep/wake → SAME worker mechanism
  → Both runners share GPUWorker.sleep()/wake_up()
  → Different post_kv_cache_wake_up() hooks → both called from same place
  → ★★★★★ COMPATIBLE

Step 8: Weight sync → SAME mechanism
  → vLLMColocateWorkerExtension.update_weights_from_ipc()
  → model.load_weights() → runner-independent
  → ★★★★★ After weight update → CUDA graphs invalidated → recapture
```

## 13. MRv2 对 GRPO/PPO 训练 Rollout 的影响

```
★★★★★ GRPO/PPO 特定影响分析:

★★★★ 正面影响 (MRv2 potentially beneficial):
  1. FlashInfer sampler → temperature>0 decode → GRPO rollout更快
     → rejection-based O(k) vs O(vocab_size) sort → ★★★★★ 采样更快
     → MRv1 用 native gumbel sampling → MRv2 用 FlashInfer → 更优
  2. BCG → 无 torch.compile → warmup 更快 → SM89 无限制
     → ★★★★★ 消除 Inductor 编译开销 → startup 更快
  3. Modular architecture → 1.5K行 vs 7.5K行 → 维护更易
     → ★★★ CPU-side path 更快 → 更少 overhead
  4. Shared graph pool → Eagle + main model 省 ~0.27 GiB
     → ★★★ 更多 KV blocks → GRPO 更多并发 rollout

★★★★ 风险 (MRv2 untested with verl RL):
  1. ★★★★★ BCG + weight update cycle → 每步训练 → weight sync
     → graph invalidation → recapture → 可能 overhead!
     → MRv1 FULL CUDA graph → weight update 也 invalidate → 但更简单
     → MRv2 BCG segments → recapture 更复杂 → untested!
  2. ★★★★★ BCG + sleep/wake cycle → sleep(2) → wake → recapture
     → 理论上 OK → 但 verl HYBRID flow → 100+ steps → 压力测试!
  3. ★★★ 2-phase forward/sample → timing 差异
     → verl generation timeout → 可能需要调整
  4. ★★★ compilation_config interaction → BCG 替换 PIECEWISE
     → verl 设置 FULL_AND_PIECEWISE → MRv2 下 BCG 替换 PIECEWISE 部分
     → 可能导致不同 warmup behavior
  5. ★★★★★ DP>1 → MRv2 scheduler NewRequestData 格式不同
     → verl 不直接处理 → AsyncLLM 封装 → 但 DP>1 scenario 未测试!

★★★★★★ 关键发现: GRPO training 的 rollout 需要什么?
  → Generation: vLLM → prompt → tokens → ★★★ MRv1/MRv2 都能做
  → logprobs: processed_logprobs → ★★★★★ MRv2 支持 → OK!
  → Weight sync: model.load_weights() → ★★★★★ runner-independent → OK!
  → Sleep/wake: GPUWorker → ★★★★★ runner-independent → OK!

★★★★★★ 结论: MRv2 在 GRPO training 应该工作 → 但需要测试!
```

## 14. RTX 4090 GRPO Training: ★★★★★ 推荐配置

```
★★★★★ Configuration Matrix:

Scenario A: ★★★★★ Qwen3-1.7B BF16 + GRPO (最常见小模型)
  → Architecture: Qwen3ForCausalLM → in DEFAULT_V2 → ★★★ MRv2 AUTO-ACTIVATE!
  → ★★★★★ 升级 v0.23.0 → 静默切换 MRv2 → UNTESTED!
  → Risk: MEDIUM → MRv2 未与 verl RL training flow 测试
  → Benefit: FlashInfer sampler → temperature>0 → 更快 sampling
  → Benefit: BCG → 无 torch.compile → 更简单 warmup
  → ★★★★★ Recommendation: TEST EXPLICITLY → 再生产部署
  → ★★★★★ Fallback: VLLM_USE_V2_MODEL_RUNNER=0 → 强制 MRv1

Scenario B: ★★★★★ Qwen3-8B INT4/GPTQ + GRPO (RTX 4090 最常用)
  → is_quantized=True → ★★★★★ MRv1 FORCED → 升级无影响!
  → ★★★★★ SAFE → 同 runner → 无交互 concern
  → ★★★★★★★★ RTX 4090 INT4 最优配置不变: V1 INT4 + INT8KV + EAGLE
  → ★★★★★ MRv2 INT4 support → FUTURE → track vLLM progress!

Scenario C: ★★★★ Llama-3.1-8B BF16 + GRPO
  → Architecture: LlamaForCausalLM → in DEFAULT_V2 → ★★★ MRv2 AUTO-ACTIVATE!
  → 同 Scenario A → 需要 explicit testing

Scenario D: ★★★★★ Qwen2.5-7B BF16 + GRPO
  → Architecture: Qwen2ForCausalLM → NOT in DEFAULT_V2 → ★★★★★ MRv1 ALWAYS
  → ★★★★★ SAFE → 同 runner → 无交互 concern

★★★★★★★★★ Action Items for RTX 4090 GRPO:

★★★★★ IMMEDIATE (升级前):
  → 在 verl env_vars 中设置 VLLM_USE_V2_MODEL_RUNNER=0 → 安全保险!
  → → 这强制 MRv1 → 直到 MRv2 明确验证!
  → → Location: vLLMReplica.launch_servers() → env_vars dict → add key
  → → 或: 在 shell 中 export VLLM_USE_V2_MODEL_RUNNER=0

★★★★★ TESTING (升级 v0.23.0 后):
  → Run Qwen3-1.7B BF16 GRPO with MRv2 enabled (无 env override)
  → Verify 5 points:
    1. generation 产出有效 tokens
    2. log_prob computation correct (与 MRv1 比较)
    3. sleep/wake cycle 稳定 → 100+ training steps
    4. weight sync 正确 → IPC transfer 后 model 正确
    5. MRv2 vs MRv1 throughput 对比 → 同 GRPO workload
  → ★★★★★ 如果 MRv2 通过所有测试 → 移除 VLLM_USE_V2_MODEL_RUNNER=0!
  → ★★★★★ 如果 MRv2 有问题 → 保持 MRv1 → 等 vLLM fix!

★★★★★ MONITORING:
  → Watch vLLM GitHub issues → MRv2 + RL training 相关
  → Track when MRv2 supports INT4 quantization (future PR)
  → Track when verl adds explicit MRv2 configuration options
  → Track DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES 扩展 → 更多 model 可能 auto-switch

★★★★★ LONG-TERM:
  → ★★★★★★★★ MRv2 INT4 support → RTX 4090 INT4 + BCG = 最佳组合!
  → ★★★★★ FlashInfer sampler → GRPO temperature>0 → 显著 decode speedup
  → ★★★★★ BCG → SM89 无 torch.compile → 更快 startup
  → ★★★★★ 当 MRv2 INT4 可用 → 升级 → 无需 MRv1 fallback!
```

## 15. ★★★★★ Summary Risk Assessment

```
★★★★★★★★★ Overall: BF16 dense MEDIUM RISK, INT4/MoE/Qwen2 ZERO RISK

★★★★★★ SAFE scenarios (无需 action):
  → INT4/GPTQ 量化模型 → MRv1 forced → 升级前后一样
  → MoE 模型 → MRv1 forced → 升级前后一样
  → Qwen2/Qwen2.5 → NOT in DEFAULT_V2 → MRv1 always
  → NPU/Ascend → MRv1 forced → 升级前后一样
  → FP8 量化模型 → is_quantized → MRv1 forced

★★★★ RISK scenarios (需要 testing):
  → ★★★★★ Qwen3 BF16 dense → MRv2 auto → UNTESTED with verl!
  → ★★★★ Llama BF16 dense → MRv2 auto → UNTESTED with verl!
  → ★★★★ Mistral BF16 dense → MRv2 auto → UNTESTED with verl!

★★★★★ Specific concerns for risky scenarios:
  1. ★★★★★ BCG + verl weight update → 每步训练 → graph recapture → untested
  2. ★★★★ MRv2 2-phase forward/sample → timing 差异 → 可能影响 generation timeout
  3. ★★★★★ FlashInfer sampler → temperature>0 GRPO → ★★★ BENEFICIAL but new path!
  4. ★★★ compilation_config interaction → BCG 替换 PIECEWISE → 不同 warmup
  5. ★★★★ DP>1 with MRv2 → #6456 fix → v0.23.0 已包含 → 但未在 verl context 测试

★★★★★★★★★ FINAL RECOMMENDATION:
  ★★★★★★★★ Set VLLM_USE_V2_MODEL_RUNNER=0 until explicit testing confirms
  MRv2 compatibility with verl's RL training flow.

  ★★★★★★★★ This applies to ALL BF16 Qwen3/Llama/Mistral models on RTX 4090
  until a full GRPO training loop test passes with MRv2.
```

## 16. 关键源码文件

```
★★★★★ 核心参考文件:

| Component               | Path (verl)                                              | Path (vLLM)                                  |
|-------------------------|----------------------------------------------------------|----------------------------------------------|
| MRv2 oracle             | N/A (verl 无代码)                                        | vllm/config/vllm.py:519-558                  |
| VLLM_USE_V2 env var     | N/A (verl 无引用)                                        | vllm/envs.py:267                             |
| DEFAULT_V2 architectures| N/A                                                      | vllm/config/vllm.py:69-75 ★★★ 只有3个!     |
| Unsupported features    | N/A                                                      | vllm/config/vllm.py:1982-2060                |
| MRv2 thinking check     | N/A                                                      | vllm/v1/engine/input_processor.py:112-116    |
| GPU worker runner select| N/A                                                      | vllm/v1/worker/gpu_worker.py:161,302,327     |
| MRv2 model runner       | N/A                                                      | vllm/v1/worker/gpu/model_runner.py (1569行)  |
| MRv1 model runner       | N/A                                                      | vllm/v1/worker/gpu_model_runner.py (7576行)  |
| MRv2 scheduler diff     | N/A                                                      | vllm/v1/core/sched/scheduler.py:894-903      |
| Sleep/wake (worker)     | N/A                                                      | vllm/v1/worker/gpu_worker.py:165-200         |
| verl async server       | verl/workers/rollout/vllm_rollout/vllm_async_server.py  | N/A                                          |
| verl rollout config     | verl/workers/config/rollout.py                           | N/A                                          |
| verl engine workers     | verl/workers/engine_workers.py                           | N/A                                          |
| verl weight sync        | verl/workers/rollout/vllm_rollout/utils.py               | N/A                                          |
| verl model_runner refs  | verl/workers/rollout/vllm_rollout/utils.py:159-184      | N/A                                          |
| verl FP8 utils          | verl/utils/vllm/vllm_fp8_utils.py                       | N/A                                          |
| verl MRv2 search        | N/A (★★★★★★★ 0 results)                                | N/A                                          |

★★★★★ Local repo paths:
  verl: /Users/jackiemac/workspace/rollout-infra/verl/
  vLLM: /Users/jackiemac/workspace/rollout-infra/vllm-latest/
```

## 参考

- [verl GitHub](https://github.com/volcengine/verl)
- [vLLM GitHub - MRv2 config](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py)
- [vLLM GitHub - GPUWorker](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_worker.py)
- [verl HybridFlow 论文](https://arxiv.org/abs/2409.19256)
- [vLLM v0.23.0 Release](https://blog.vllm.ai/2025/v0.23-release)
