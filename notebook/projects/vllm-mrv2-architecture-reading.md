# vLLM Model Runner V2 (MRv2) 源码级架构阅读

> 2026-06-15 | 源码: vllm-project/vllm v0.23.0 (2026-06-12), vllm/v1/worker/gpu/
> 核心: MRv2 = next-gen model runner → 1569行(vs V1 7576行) → modular → forward/sampling分离
> 关键: Breakable CUDA Graph + FlashInfer sampler + PP bubble elimination + Gemma4 MTP
> ★ ★ ★ RTX 4090 INT4推理仍用V1 runner → MRv2暂不支持量化模型!

## 1. MRv2是什么? vs V1

```
★ ★ ★ MRv2 = vLLM V1的下一代model runner

V1 GPUModelRunner:
  → 单文件 gpu_model_runner.py = 7,576行 → monolithic!
  → 继承3 Mixin(LoRA+KVConnector+ECConnector)
  → execute_model() = ~400行 → forward+sample一体 → 无法pipeline overlap

MRv2 GPUModelRunner:
  → 目录 vllm/v1/worker/gpu/ → 25+文件 → modular!
  → model_runner.py = 1,569行 → 只继承LoRAModelRunnerMixin
  → execute_model()只forward → sample_tokens()分离 → ★ pipeline overlap!
  → README: "This file must only contain code common to every model"

★ ★ ★ V1→MRv2的关键变化:
  1. 文件结构: monolithic→modular (7.5K行→1.5K行核心+25+专用文件)
  2. 执行模式: forward+sample一体→两阶段分离
  3. Mixin: 3→1(只有LoRA; KV/EC分离)
  4. 状态管理: ExecuteModelState 10字段→6字段
  5. 异步输出: AsyncOutput(CUDA event+D2H非阻塞)

默认范围 (v0.23.0):
  DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = {
    LlamaForCausalLM, MistralForCausalLM,
    Qwen3ForCausalLM, DeepseekV2ForCausalLM, Qwen2MoeForCausalLM
  }
  → ★ 只覆盖dense, unquantized, generative models!
  → ★ ★ INT4模型→MRv1! → RTX 4090 INT4推理仍用V1 runner!
```

## 2. Forward/Sampling分离 → Pipeline Overlap

```
★ ★ ★ 两阶段执行 → 这是MRv2的核心设计!

V1:
  execute_model() → forward+sample一体 → scheduler必须等整个完成
  → 10字段ExecuteModelState → logits+scheduler_output+spec_decode+...
  → ★ 无法overlap → decode+prefill顺序执行!

MRv2:
  execute_model() → 只forward → 返回IntermediateTensors(非最后PP rank)或None
  → ExecuteModelState只有6字段: input_batch+attn_metadata+slot_mappings+hidden_states+aux+finished
  → ★ 无logits! 无scheduler_output! → 最简状态!

  sample_tokens() → 单独sampling → consume execute_model_state
  → logits→process→sample→draft→output → 分离forward和sample

★ ★ 为什么重要?
  → Pipeline Overlap: 最后PP rank在sample step T → 第一PP rank可forward step T+1!
  → V1无法overlap → scheduler等整个execute_model+sample → 每步都stall!
  → MRv2可以overlap → GPU利用率更高 → ★ PP bubble消除!

AsyncOutput:
  → 专用output_copy_stream → CUDA event → 非阻塞D2H
  → get_output() → sync event → 返回ModelRunnerOutput
  → ★ GPU compute继续 → CPU output处理异步 → 不阻塞!
```

## 3. Breakable CUDA Graph (BCG)

```
★ ★ ★ Breakable CUDA Graph = v0.23最重要创新!

传统PIECEWISE模式:
  → torch.compile把模型分成FX segments(attn边界)
  → 每段编译 → Inductor Triton kernel → 编译慢!
  → prefill和decode用不同artifact → 复杂!

BCG创新:
  → 单个capture context → 遇到@eager_break_during_capture → break!
  → 流程: capture_begin → ... → 遇到attention op → break → eager执行 → 新capture → ...
  → 结果: 一系列segments(graph.replay + eager_fn) → 按序replay!

源码: vllm/compilation/breakable_cudagraph.py
  BreakableCUDAGraphCapture:
    → _begin_segment() → torch.cuda.CUDAGraph().capture_begin(pool=...)
    → add_eager(fn) → 结束当前segment → 运行fn eagerly → 记录fn → 新segment
    → _end_segment() → capture_end() → append graph.replay到segments list
    → Replay: for r in self.segments: r() → 逐段replay!

@eager_break_during_capture decorator:
  → 标记attention/KV-cache ops → break点
  → ★ 必须是outermost decorator!
  → ★ 必须写进caller-provided output tensor(in-place) → 不能返回新tensor!

★ ★ BCG vs PIECEWISE:
  1. 无torch.compile → 无Inductor编译开销 → 简单!
  2. 同一artifact for prefill+decode → 无mode-specific dispatch
  3. 消除torch.compile依赖 → 任何SM架构都能用!
  4. 共享graph pool(Eagle+main model) → 省~0.27 GiB (Llama-8B+EAGLE)

Benchmark (GB200):
  Qwen3-30B-A3B: 71.04 vs 70.46 req/s
  Qwen3-235B FP8 TP=4: 30.29 vs 28.17 req/s (+7.5% throughput!)
  Online serve: TTFT 369ms vs 425ms (-13%!)

★ RTX 4090: BCG可用(SM无关) → 但INT4模型→MRv1 → BCG暂时不适用!
★ ★ ★ 当MRv2支持量化后 → RTX 4090 INT4+BCG = 最佳组合!
```

## 4. FlashInfer Sampler Integration

```
★ FlashInfer sampler → MRv2的sampling优化!

源码: vllm/v1/worker/gpu/sample/sampler.py + vllm/v1/sample/ops/topk_topp_sampler.py

Sampler.__init__ → self.use_flashinfer = flashinfer_sampler_supported()
  → 检查CUDA + SM capability + env var
  → ★ RTX 4090 SM 8.9 → FlashInfer支持! → SM 8.x + 9.x → ✓!

sample() → 条件性使用FlashInfer:
  → 必须有top-k或top-p filtering
  → ★ 不能是greedy(any_greedy=False)
  → ★ 不能有explicit seed(any_explicit_seed=False)
  → ★ 不能有processed_logprobs mode
  → 全满足 → flashinfer_sample() → rejection-based → O(k) vs O(vocab_size) sort!
  → 否则 → native apply_top_k_top_p + gumbel_sample

★ ★ FlashInfer sampling:
  top_p_sampling_from_probs → rejection sampling → 无CPU sync
  top_k_sampling_from_logits → rejection → 确定性(deterministic=True)
  top_k_top_p_sampling → combined → 最高效

★ RTX 4090: FlashInfer sampler✓ → temperature>0+top-k/top-p → FlashInfer更快!
  → greedy(temperature=0) → native → FlashInfer不适用
  → ★ 实际GRPO rollout → temperature>0 → FlashInfer可用!
```

## 5. PP Bubble Elimination

```
★ ★ ★ PP bubble消除 → MRv2多GPU最大改进!

PR #42187 (njhill, merged 6/2)

V1 PP问题:
  → 每PP stage等整个pipeline完成 → 才能开始下一步
  → 非最后rank等sampled tokens → 才能update state → 巨大bubble!

MRv2解决方案: PPHandler + async scheduling

1. Decoupled broadcast:
   → 专用NCCL communicator(broadcast_group) + side stream(broadcast_stream)
   → ★ broadcast和inter-stage p2p hidden-state并行!

2. Deferred consumption via FIFO:
   → deque[PendingRecv | None] → 预填充pp_size个None
   → Step T receive → push PendingRecv → pp_size步后consume
   → ★ ★ 不需要等Step T broadcast → 就能开始Step T+1 forward!

3. Request-index generation counter:
   → req_idx_gen_np → track request frees → 消费时filter freed indices(-1)

4. Selective broadcast:
   → compute_need_sampled_mask() → 只broadcast真正需要的requests
   → finishing/非final prefill → exclude → 减少broadcast数据量!

Benchmark (MiniMax-M2.7 FP8 GB200 PP=4):
  128k/1: 3.17x throughput → ★ ★ ★ 极大improvement!
  80k/1k: 2.28x throughput
  1k/1k: 1.24x throughput

★ RTX 4090: PP>1 → PCIe灾难 → 不可行 → 但架构知识为多GPU准备!
```

## 6. Gemma4 MTP Support

```
★ Gemma4 MTP → MRv2 speculator hierarchy重构!

PR #43241 (TheEpicDolphin, merged 6/4)

旧: EagleSpeculator monolithic
新: 分层hierarchy:
  BaseSpeculator(ABC) → init_cudagraph_manager/capture/propose → 最小接口
  DraftModelSpeculator → shared state + buffer allocation + DP config + load_model
  AutoRegressiveSpeculator → autoregressive loop + 可override hooks:
    advance_draft_positions → True(Eagle/MTP) / False(Gemma4 Q-only)
    model_returns_tuple → True(Eagle/Gemma4) / False(MTP)
    load_draft_model → 主要扩展点
    sample_draft → hidden states→draft tokens
  EagleSpeculator → thin subclass, only override load_draft_model
  MTPSpeculator → override model_returns_tuple=False
  Gemma4Speculator → override advance_draft_positions=False

★ ★ Gemma4 MTP特殊性:
  → Q-only attention → 读target的K/V → 不写新KV → positions保持不变!
  → Cross-model KV sharing → draft层映射到target最后非KV-shared层
  → kv_sharing_target_layer_name → 每层映射 → 共享KV cache!
  → Heterogeneous head dimensions → sliding=256 / global=512 → TRITON_ATTN必须
  → Embedding sharing → draft share embed_tokens with target → 省内存!

Benchmark (Gemma4-E2B MTP temperature=1.0):
  concurrency=1: 2.40 vs 2.25 req/s
  concurrency=32: 61.04 vs 60.09 req/s
  concurrency=64: 106.20 vs 104.86 req/s
  → ★ MRv2 consistently outperforms V1 in TPOT!
```

## 7. RTX 4090 Impact Summary

```
★ ★ ★ MRv2对RTX 4090的影响:

直接可用:
  ✓ FlashInfer sampler → SM 8.9支持 → temperature>0 decode更快
  ✓ Modular architecture → 1.5K行 → CPU-side path更快
  ✓ Shared graph pool → Eagle+main省~0.27 GiB → 更多KV blocks!
  ✓ Breakable CUDA Graph → SM无关 → 消除compile依赖

✗ ✗ ✗ 关键限制: INT4模型→MRv1!
  → MRv2 DEFAULT_V2只覆盖dense unquantized models
  → model_config.is_quantized → True → MRv1
  → ★ ★ RTX 4090最优配置(INT4+INT8KV+EAGLE) → 仍用V1 runner!
  → ★ ★ ★ MRv2 INT4支持 → 何时? → 需要跟踪!

不适用:
  ✗ PP bubble elimination → PP>1 on PCIe → 灾难级慢 → RTX 4090不可行
  ✗ NVLS/TMA → SM 8.9 not SM 9.0 → Hopper only

MRv2不支持的特征 (relevant to RTX 4090):
  ✗ Custom logits processors (包括PR #7 top-n-sigma)
  ✗ prompt_embeds
  ✗ raw_logits/processed_logprobs
  ✗ KV sharing fast prefill
  ✗ EC transfer (PD disaggregation)
  ✗ Ngram/ngram_gpu spec decode
  ✗ Dynamic speculative decoding
  ✗ Sequence parallelism with TP>1
  ✗ Elastic expert parallelism

★ ★ ★ 总结:
  MRv2是架构改进 → dense BF16推理更快 → 但RTX 4090 INT4最优配置仍用V1
  ★ ★ 等MRv2支持量化 → INT4+BCG+FlashInfer → RTX 4090推理会更快!
  ★ ★ ★ 当前最优: V1 INT4+INT8KV+EAGLE → 4,791→9,088 tok/s (不变!)
```

## 8. 关键PR和源码文件

```
| 项目 | PR/路径 |
|------|---------|
| MRv2 model runner | vllm/v1/worker/gpu/model_runner.py (1569行) |
| V1 model runner | vllm/v1/worker/gpu_model_runner.py (7576行) |
| BCG | vllm/compilation/breakable_cudagraph.py |
| MRv2 CUDA graph | vllm/v1/worker/gpu/cudagraph_utils.py (519行) |
| MRv2 sampler | vllm/v1/worker/gpu/sample/sampler.py |
| FlashInfer ops | vllm/v1/sample/ops/topk_topp_sampler.py |
| MRv2 PP handler | vllm/v1/worker/gpu/pp_utils.py |
| MRv2 async output | vllm/v1/worker/gpu/async_utils.py |
| MRv2 speculator | vllm/v1/worker/gpu/spec_decode/speculator.py |
| MRv2 warmup | vllm/v1/worker/gpu/warmup.py |
| MRv2 default config | vllm/config/vllm.py (DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES) |
| BCG for MRv2 | PR #44050 (WoosukKwon) |
| FlashInfer sampler | PR #42472 (njhill) |
| PP bubble elimination | PR #42187 (njhill) |
| Llama/Mistral MRv2 | PR #43458 (njhill + yewentao256) |
| Gemma4 MTP | PR #43241 (TheEpicDolphin) |
| Eagle shared pool | PR #44078 (LucasWilkinson) |
| Experimental BCG | PR #42304 (ZJY0516) |
```
