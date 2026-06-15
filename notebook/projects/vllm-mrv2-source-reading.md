# vLLM Model Runner V2 (MRv2) — Source-Level Architecture Analysis

> 2026-06-16 | vllm-project/vllm | Issue #41286 (open) | MRv2 migration | DBO ubatching | RTX 4090 default impact
> ★★★★★ MRv2 = vLLM V1架构重构 → execute_model + sample_tokens 两步分离 → AsyncOutput → 异步调度
> ★★★★★ Qwen3/Llama/Mistral DEFAULT启用MRv2 → RTX 4090 GRPO直接受影响!
> ★★★★★ verl有ZERO MRv2处理代码 → VLLM_USE_V2_MODEL_RUNNER=0 是RTX 4090安全兜底

## 1. ★★★★★ MRv2 Architecture — Two-Step Split Execution

```
★★★★★★★ 核心架构差异 — MRv1 vs MRv2:

| Feature | MRv1 (gpu_model_runner.py) | MRv2 (gpu/model_runner.py) |
|---------|---------------------------|---------------------------|
| File size | 335KB, ~9000+ lines | 1574 lines — clean modular |
| Execution | execute_model + sample 一步 | execute_model → None, then sample_tokens() 两步 |
| Output | direct ModelRunnerOutput | AsyncOutput (async D2H copy overlaps postprocess) |
| Modular | monolithic single file | split: block_table, input_batch, attn_utils, buffer_utils, cudagraph_utils, dp_utils |
| Async scheduling | ✗ (single step) | ✓ (two-step enables async scheduling) |
| DBO (ubatching) | UBatchWrapper in same file | separate layer — not in runner directly |
| PP handling | basic | pp_handler with broadcast/receive |
| LoRA | inlined | LoRAModelRunnerMixin |
| Spec decode | basic | speculator.propose + draft_tokens_handler |

★★★★★★★ MRv2两步分离 — 关键代码流程:

Step 1: execute_model(scheduler_output) → returns None (or IntermediateTensors for PP)
  → GPUModelRunner.execute_model() → line 1102
  → update requests → prepare inputs → forward model → hidden_states
  → Store ExecuteModelState → self.execute_model_state = ExecuteModelState(...)
  → return None ← 关键: 不直接sample!

Step 2: sample_tokens(grammar_output) → returns AsyncOutput | ModelRunnerOutput
  → GPUModelRunner.sample_tokens() → line 1323
  → Retrieve ExecuteModelState → self.execute_model_state
  → sample(hidden_states) → sampler_output
  → postprocess → AsyncOutput (async D2H copy)
  → spec decode → speculator.propose → draft_tokens
  → return async_output ← 关键: 异步!

★★★★★★★ 为什么两步分离重要:
  → 异步调度 → execute_model在GPU → sample_tokens可以CPU并行 → overlap
  → DP同步 → dispatch_cg_and_sync_dp → 多DP rank协调
  → PP pipeline → execute_model + sample_tokens → 可以overlap PP stages
  → DBO → execute_model → microbatch → 不同ubatch线程并行 → 通信overlap
```

## 2. ★★★★★ DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES

```
★★★★★★★ vllm/config/vllm.py line 68:

DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({
    "Qwen3ForCausalLM",
    "DeepseekV2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
})

★★★★★★★ use_v2_model_runner property (line 528):

def use_v2_model_runner(self) -> bool:
    use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:    # ← env var override!
        return use_v2_model_runner          # ← 用户可以强制True/False

    if self.model_config.is_diffusion:
        return True                          # ← diffusion models always MRv2

    if not self._is_default_v2_model_runner_model():
        return False                         # ← 不在default列表 → MRv1

    if not HAS_TRITON:
        return False                         # ← Triton不可用 → fallback MRv1

    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        return False                         # ← 有不支持feature → fallback MRv1

    return True                              # ← 默认启用MRv2!

★★★★★★★ 关键: 不在default列表的模型 → MRv1!
  → DeepSeek-V3/V4 → DeepseekV3ForCausalLM → NOT in default → MRv1!
  → Qwen2.5 → Qwen2ForCausalLM → NOT in default → MRv1!
  → Qwen3 → Qwen3ForCausalLM → IN default → MRv2! ← RTX 4090 GRPO常用!
  → Llama → LlamaForCausalLM → IN default → MRv2!
  → Mistral → MistralForCausalLM → IN default → MRv2!

★★★★★★★ RTX 4090影响:
  → ★★★★★ Qwen3-1.7B/8B → MRv2 default → GRPO rollout → verl需要适配!
  → ★★★★★ Llama-3.1-8B → MRv2 default → GRPO rollout → verl需要适配!
  → ★★★★★ Mistral-7B → MRv2 default → GRPO rollout → verl需要适配!
  → ★★★ Qwen2.5-7B → MRv1 default → GRPO rollout → verl不需要适配!
```

## 3. ★★★★★ MRv2 Key Source Code Paths

```
★★★★★★★ MRv2 worker initialization (gpu_worker.py line 329):

if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import (
        GPUModelRunner as GPUModelRunnerV2,    # ← 新路径! gpu/model_runner.py
    )
    self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
else:
    from vllm.v1.worker.gpu_model_runner import (
        GPUModelRunner as GPUModelRunnerV1,    # ← 旧路径! gpu_model_runner.py
    )
    self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

★★★★★★★ MRv2 modular directory structure (vllm/v1/worker/gpu/):

model_runner.py       — 1574 lines — clean, common-only code
block_table.py        — BlockTables class
input_batch.py        — InputBatch + InputBuffers + combine/expand helpers
attn_utils.py         — build_slot_mappings_by_layer + get_kv_cache_spec + init_attn_backend
buffer_utils.py       — async_copy_to_gpu + set_default_max_concurrency
cudagraph_utils.py    — BatchExecutionDescriptor + ModelCudaGraphManager
dp_utils.py           — dispatch_cg_and_sync_dp (DP coordination)
eplb_utils.py         — EPLBController
async_utils.py        — AsyncOutput + AsyncPoolingOutput
cp_utils.py           — prepare_dcp_local_seq_lens (context parallel)
lora_utils.py         — get_num_active_loras_for_dispatch + set_active_mm_loras
prompt_logprobs.py    — prompt_logprobs worker
spec_utils.py         — speculator handling

★★★★★★★ MRv2 coding style guide (model_runner.py header comment):

"NOTE: This model runner is shared by all models: text and multimodal,
 generative and embedding, public and private. As a result, this file
 must only contain code that is common to every model."

→ ★★★★★★★★ 严格common-only → 模型特定逻辑 → 在model-specific files → 不在runner!
→ → ★★★★★★★★ 比MRv1更干净 → 更易维护 → 更易扩展!
```

## 4. ★★★★★ DBO (Disjoint Batch Overlap) / Ubatching Architecture

```
★★★★★★★ DBO = MRv1特有的通信overlap机制 (不是MRv2的!):

UBatchWrapper (gpu_ubatch_wrapper.py, 527 lines):
  → 初始化 → ready_barrier = threading.Barrier(num_ubatches + 1)
  → comm_stream + compute_stream → 双流分离 → overlap通信和计算
  → SMControlContextManager → comm_sms + compute_sms → SM分区 → DeepEP专用!
  → cudagraph_wrapper → FULL graph → replay → PIECEWISE → execute
  → __call__ → ubatch_slices → slice inputs → run ubatches in threads

UBatchContext (ubatching.py):
  → compute_stream + comm_stream → 双CUDA流
  → ready_barrier → threading.Barrier → 多线程同步
  → cpu_wait_event + cpu_signal_event → CPU线程协调
  → gpu_comm_done_event + gpu_compute_done_event → GPU事件同步
  → switch_to_comm / switch_to_compute → 流切换
  → yield_ → yield_and_switch → 协调式线程yield

ubatch_utils.py:
  → UBatchSlice → request_slice + token_slice → 微批切片
  → check_ubatch_thresholds → dbo_decode_token_threshold=32 / dbo_prefill_token_threshold=512
  → maybe_create_ubatch_slices → 按num_ubatches切分 → 均匀分配tokens

★★★★★★★ DBO vs MRv2:
  → ★★★★★ DBO = MRv1的ubatching → multi-thread → DP>1时自动启用 → SM分区
  → ★★★★★ MRv2 → 不在runner内处理DBO → 由调度层处理 → 更干净
  → ★★★★★★★★ RTX 4090: DP=1 → DBO不需要 → 单GPU → no DP overlap → DBO=False
  → → ★★★★★★★★ 但MRv2默认启用 → 不影响DBO → MRv2只是runner重构 → 不改变调度逻辑!
```

## 5. ★★★★★★★ verl MRv2 Handling — ZERO Code!

```
★★★★★★★★★ 关键发现: verl对MRv2完全没有处理!

搜索 verl-project/verl:
  → VLLM_USE_V2_MODEL_RUNNER → 0 matches → ZERO!
  → model_runner_v2 → 0 matches → ZERO!
  → USE_V2_MODEL → 0 matches → ZERO!

★★★★★★★★★ verl vLLM rollout worker → 如何调用vLLM:
  → verl/workers/rollout/vllm_rollout/ → vLLM rollout worker
  → 初始化 → vllm.LLM(..., model_runner_cls=???)
  → → ★★★★★★★ 如果vLLM默认MRv2 → 但verl不知道MRv2 → execute_model + sample_tokens两步
  → → → ★★★★★★★★ verl可能只调用一次 → 得到None → 不知道需要再调sample_tokens()!

★★★★★★★★★ RTX 4090安全兜底:
  → ★★★★★★★★ VLLM_USE_V2_MODEL_RUNNER=0 → 强制MRv1 → verl安全!
  → → 这是之前MEMORY中记录的建议 → 现在有源码级证据!
  → → ★★★★★★★★ 所有verl+vLLM GRPO实验 → 必须设置此环境变量!

★★★★★★★★★ verl MRv2 PR机会:
  → verl需要PR → 检测vLLM是否使用MRv2 → 正确处理两步分离
  → → ★★★★★★★★ 但这是Tier 3贡献 → 优先级低于BudgetRefiner/Inductor Fusion Guard
  → → → ★★★★★★★★★ 短期推荐: 环境变量兜底 → 不需要PR
```

## 6. ★★★★★ MRv2 Migration Roadmap (Issue #41286)

```
★★★★★★★ Issue #41286 tracking — open, 6 phases:

Phase 1 (merged): #39337 → 基础infra → initial MRv2 GPUModelRunner
  → #39353, #39937, #40559, #40648, #41285 → all merged

Phase 2 (merged): #43458 → dense model support + spec decode + LoRA
  → #42673, #42676, #42778, #42783, #43160, #43139, #43233, #43719 → merged

Phase 3 (open): #42667 → MoE support + EPLB
  → #43915 (ElasticEPScalingExecutor) → open
  → MoE MRv2 support → still in progress

Phase 4 (open): #44443 → Enable ALL dense models for MRv2
  → #44450, #44568 → merged (partial)
  → #45467 → open (bugfix)

Phase 5 (open): #44446 → Quantized model support

Phase 6 (open): #45461 → GraniteMOE for MRv2

★★★★★★★ 当前状态:
  → Dense models (Qwen3, Llama, Mistral) → MRv2 DEFAULT ✓ → merged
  → Quantized models → NOT default → Phase 5 open → quantized fallback MRv1
  → MoE models → NOT default → Phase 3 partial → MoE still MRv1
  → 最终目标 → "Switch to model runner v2 by default" → 所有模型 → 未完成

★★★★★★★ RTX 4090影响:
  → ★★★★★ Qwen3-1.7B (dense, non-quantized) → MRv2 default → verl需兜底!
  → ★★★★★ Llama-3.1-8B (dense, non-quantized) → MRv2 default → verl需兜底!
  → ★★★★★ INT4/INT8 quantized → MRv1 fallback → 安全!
  → ★★★★★ MoE models → MRv1 → 安全 (但RTX 4090不能跑MoE anyway)
```

## 参考
- vLLM Issue #41286: MRv2 migration tracking (open)
- vLLM vllm/config/vllm.py: DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES, use_v2_model_runner property
- vLLM vllm/v1/worker/gpu/model_runner.py: MRv2 GPUModelRunner (1574 lines)
- vLLM vllm/v1/worker/gpu_model_runner.py: MRv1 GPUModelRunner (335KB monolithic)
- vLLM vllm/v1/worker/gpu_ubatch_wrapper.py: UBatchWrapper (DBO)
- vLLM vllm/v1/worker/ubatching.py: UBatchContext (threading-based microbatching)
- vLLM vllm/v1/worker/ubatch_utils.py: UBatchSlice + threshold checking
- vLLM vllm/envs.py: VLLM_USE_V2_MODEL_RUNNER env var (line 257)
- verl: ZERO MRv2 handling code → VLLM_USE_V2_MODEL_RUNNER=0 required for safety
- Related notes: vllm-v0.23-release-reading.md, vllm-v1-scheduler-deep-reading.md
