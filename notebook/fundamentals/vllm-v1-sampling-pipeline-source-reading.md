# vLLM V1 Sampling Pipeline Source Reading — Sampler + Logits Processors + TopK/TopP + Penalties + Min-P + FlashInfer Sampler

> 2026-06-12 | vLLM V1采样管线全链路源码分析: Sampler→logits_processors(allowed/bad_words/min_tokens/logit_bias)→penalties(repetition/frequency/presence)→temperature→min-p→top-k/top-p→greedy/random→logprobs→rejection_sampler
> 源码: vllm/v1/sample/sampler.py (439行), metadata.py (55行), ops/penalties.py (57行), ops/topk_topp_sampler.py (515行), rejection_sampler.py (937行), logits_processor/interface.py (106行), logits_processor/builtin.py (332行)
> 关联: vllm-v1-gpu-model-runner-source-reading.md, vllm-v1-spec-decode-architecture-source-reading.md

## 0. 核心定律: 9步采样管线 + argmax_invariant分离 + FlashInfer sampler

```
vLLM V1 Sampler执行顺序(严格!):

Step 1: Logprobs → raw_logprobs/raw_logits → 用于后续gather
Step 2: logits→float32 → 精度保证
Step 3: allowed_token_ids → masked_fill(-inf) → 白名单过滤
Step 4: bad_words → apply_bad_words → 禁止词排除
Step 5: non_argmax_invariant logits_processors → min_tokens + logit_bias → 影响贪心!
Step 6: penalties → repetition/frequency/presence → 影响贪心!
Step 7: greedy_sample → argmax → 贪心路径(如果all_greedy直接返回)
Step 8: temperature → logits/temp → 温度缩放(如果all_random则T≥eps)
  → argmax_invariant logits_processors → min_p → 不影响贪心!
  → top_k/top_p → 截断 → FlashInfer或Triton
  → random_sample → multinomial → 随机采样
  → 如果T<eps → 用greedy(不是random!) → 混合batch!
Step 9: logprobs gather → topk+sampled → 返回SamplerOutput

关键设计:
  → argmax_invariant分离: min_p不影响argmax → 在temperature后应用
  → → 非argmax_invariant(min_tokens/logit_bias) → 在temperature前应用 → 影响贪心!
  → → → 为什么? temperature后min_p截断不影响argmax → 但logit_bias改变最大值→影响!
  → all_greedy快速路径: 全贪心→argmax→直接返回 → 无temperature/sampling开销!
  → all_random快速路径: 全随机→无需greedy_sample → 跳过argmax
  → mixed path: greedy+random → torch.where(T<eps, greedy, random) → 混合选择
  → FlashInfer sampler: VLLM_USE_FLASHINFER_SAMPLER → GPU原生top-k/top-p → 最快!
```

## 1. Sampler.forward() — 采样入口

```
Sampler.forward(logits, sampling_metadata)流程(~70行):

Step 1: Raw Logprobs
  → if num_logprobs or logprob_token_ids:
    → raw_logprobs = compute_logprobs(logits) → log_softmax → FP32
    → 或 raw_logits = logits.clone() → raw_logits模式(保留原始值)

Step 2: Float32转换
  → logits = logits.to(torch.float32) → 统一精度

Step 3: Logits Processors (apply_logits_processors)
  → allowed_token_ids_mask → masked_fill(-inf) → 白名单
  → bad_words_token_ids → apply_bad_words → 禁止词
  → non_argmax_invariant processors → min_tokens + logit_bias → 影响贪心
  → penalties → repetition/frequency/presence
  → thinking_budget → holder.apply_to_logits → 推理深度控制

Step 4: Sample
  → sampled, processed_logprobs = self.sample(logits, sampling_metadata)
  → → greedy或random → 混合选择

Step 5: Logprobs Gather
  → logprob_token_ids → gather_specific_token_logprobs → 更高效(只取指定token)
  → 或 num_logprobs → gather_logprobs → topk+sampled token
  → → logprob_token_ids优先 → 更精确且更高效

Step 6: Return SamplerOutput
  → sampled_token_ids → int32 → [num_reqs, 1] → unsqueeze(-1)
  → logprobs_tensors → LogprobsTensors → indices+logprobs+ranks
```

## 2. sample() — 核心采样逻辑

```
sample(logits, sampling_metadata)流程(~60行):

Fast Path 1: all_greedy
  → greedy_sampled = argmax(logits) → dim=-1
  → all_greedy=True → 直接返回greedy_sampled → 无temperature开销!
  → → 如果需要logprobs → processed_logprobs = compute_logprobs(logits)

Fast Path 2: all_random
  → 不执行greedy_sample → greedy_sampled=None
  → → 温度>0 → 直接走random path → 无argmax开销

Mixed Path: some greedy + some random
  → greedy_sampled = argmax(logits) → 先算贪心(所有请求)
  → apply_temperature → logits / temp(T<eps→1.0) → 温度缩放
  → argmax_invariant processors → min_p → 不影响argmax
  → topk_topp_sampler → top-k/top-p截断 + random sampling
  → torch.where(T<eps, greedy, random) → T<eps用贪心/T≥eps用随机 → 混合!

关键优化:
  → all_greedy → argmax一次 → 不做temperature/top-k/top-p → ~0.06ms vs random ~0.12ms
  → → 生产大部分贪心(T=0) → all_greedy快速路径 → 几乎零开销
  → in-place操作 → logits.div_(temp) → 不创建新tensor → 省内存+省带宽
  → torch.where → reuse greedy_sampled tensor → out=greedy_sampled → 省内存
```

## 3. apply_temperature() — 温度缩放

```
apply_temperature(logits, temp, all_random):

  → if not all_random → temp<eps → temp=1.0 → 防除零! → 贪心请求T=0→T=1→不变
  → if all_random → 不防除零 → 所有T≥eps → 直接除
  → logits.div_(temp.unsqueeze(1)) → in-place! → 逐请求不同温度

Temperature数学:
  → logits/T → softmax(logits/T) → Boltzmann分布
  → T→0: softmax→one-hot → 确定性 → greedy
  → T→1: softmax(logits) → 原始分布 → 标准
  → T→∞: softmax→uniform → 完全随机
  → T<eps: 等效于greedy → torch.where(T<eps, greedy, random)
```

## 4. TopKTopPSampler — top-k/top-p截断+采样

```
TopKTopPSampler实现选择:

1. FlashInfer Sampler (最快!):
  → flashinfer_sampler_supported() → CUDA+SM≥8.0+VLLM_USE_FLASHINFER_SAMPLER
  → → RTX 4090(SM89) → 支持!
  → → FlashInfer sampling_ops.top_k_top_p_sampling → GPU原生
  → → → 但! FlashInfer不暴露post-top-k logits → 不能用于logprobs

2. Triton Sampler (次快):
  → apply_top_k_top_p_triton → Triton kernel → 自定义top-k/top-p
  → → 需要HAS_TRITON=True

3. PyTorch Sampler (最慢但兼容):
  → torch.topk → masked_fill → multinomial → 标准PyTorch操作

决策逻辑:
  → FlashInfer可用 + 不需要processed_logprobs → FlashInfer (最快!)
  → → FlashInfer不可用或需要processed_logprobs → Triton或PyTorch
  → → ROCm → rocm_aiter_ops → AMD专用

FlashInfer sampler优势:
  → GPU原生 → 无Python overhead → 比PyTorch快2-3x
  → → 但! 不支持logprobs_mode="processed_logits" → 需要原始logprobs→用PyTorch/Triton
  → → RTX 4090推荐: FlashInfer sampler(T=0/greedy) → 生产大部分T=0 → FlashInfer可用!
```

## 5. SamplingMetadata — 采样参数数据结构

```
@dataclass SamplingMetadata:
  → temperature: Tensor | None → 温度(all_greedy→None→省GPU复制!)
  → all_greedy: bool → 全贪心 → 快速路径
  → all_random: bool → 全随机 → 快速路径
  → top_p: Tensor | None → top-p截断(no_top_p→None→省复制!)
  → top_k: Tensor | None → top-k截断(no_top_k→None→省复制!)
  → generators: dict[int, torch.Generator] → 随机种子
  → max_num_logprobs: int | None → logprobs数量
  → logprob_token_ids: dict[int, list[int]] → 指定token logprobs(更高效!)
  → no_penalties: bool → 无penalty → 省GPU复制!
  → prompt_token_ids: Tensor | None → prompt ids(penalty用)
  → frequency/presence/repetition_penalties: Tensor → 三种penalty
  → output_token_ids: list[list[int]] → 已输出token ids(penalty用)
  → allowed_token_ids_mask: Tensor | None → 白名单mask
  → bad_words_token_ids: dict[int, list[list[int]]] → 禁止词
  → logitsprocs: LogitsProcessors → 自定义处理器
  → spec_token_ids: list[list[int]] | None → 投机解码tokens
  → thinking_budget_state_holder → 推理深度控制

差量更新优化(来自InputBatch.refresh_metadata):
  → all_greedy → temperature=None → 不复制GPU tensor → 省带宽!
  → no_top_p → top_p=None → 不复制
  → no_top_k → top_k=None → 不复制
  → no_penalties → penalties=None → 不复制 → prompt_token_ids也跳过
  → → 生产: B=118, ~80%贪心 → 大部分metadata保持不变 → 省GPU复制!
```

## 6. LogitsProcessors — 自定义logits处理器

```
LogitsProcessor ABC:
  → apply(logits) → 应用处理器 → in-place或返回新tensor
  → is_argmax_invariant() → True=不影响贪心 → False=影响贪心
  → update_state(batch_update) → 差量更新状态 → Persistent Batch驱动

两类处理器:
  → non_argmax_invariant → 在temperature前应用 → 影响贪心
    → MinTokensLogitsProcessor → EOS token惩罚 → 最低长度保证
    → LogitBiasLogitsProcessor → 自定义logit偏置 → 指定token偏置
  → argmax_invariant → 在temperature后应用 → 不影响贪心
    → MinPLogitsProcessor → min-p截断 → 2025新方法 → 只影响随机采样

MinPLogitsProcessor (332行):
  → min_p_cpu_tensor → CPU numpy → 差量更新
  → min_p_device → GPU tensor → 非阻塞复制
  → min_p_count → 有min_p的请求数 → 决定是否需要GPU复制
  → update_state → 处理added/removed/moved → 差量更新
  → → 如果min_p_count==0 → 不更新GPU → 省带宽!

Min-P算法:
  → softmax(logits) → 找max_prob
  → threshold = max_prob * min_p → 最低概率阈值
  → logits[prob < threshold] = -inf → 截断 → 只保留高于阈值的
  → → 为什么argmax_invariant? min_p截断不影响argmax → argmax仍然是max!
  → → 但影响随机采样 → 改变概率分布 → 更集中或更均匀

BatchUpdate → LogitsProcessors差量更新桥梁:
  → removed/added/moved → 增删移动 → InputBatch.condense()产生
  → → LogitsProcessor.update_state → 根据BatchUpdate更新内部状态
  → → → 无BatchUpdate → 不更新 → 节省!
```

## 7. Penalties — 重复惩罚

```
apply_all_penalties(logits, prompt_token_ids, presence/frequency/repetition, output_token_ids):

三种Penalty:
  1. Repetition Penalty → 重复token logits除以penalty(如果>1→降低/如果<1→升高)
     → token出现过 → logits[token] /= penalty → 减少重复
     → → 但! 改变正负号 → 可能翻转偏好 → 不理想 → presence/frequency更好

  2. Frequency Penalty → token出现次数 → logits[token] -= count * freq_penalty
     → 出现越多 → logits越低 → 惩罚频率 → 更公平
     → → 只降低logits → 不改变符号 → 更稳定

  3. Presence Penalty → token是否出现 → logits[token] -= presence_penalty
     → 出现就惩罚 → 不关心次数 → 防止重复出现

  → prompt_token_ids → prompt中的token也计算 → 防止重复prompt内容
  → output_token_ids → 生成的token → 异步调度时placeholder=-1 → masked_fill(vocab_size)

RTX 4090 Penalties:
  → 大部分请求no_penalties → 跳过GPU复制+计算 → 省带宽+时间
  → → 只有指定penalty的请求才计算 → 差量处理
  → → Penalties overhead ~0.02ms → sampling ~0.06ms → negligible
```

## 8. Rejection Sampler — 投机解码拒绝采样

```
文件: vllm/v1/sample/rejection_sampler.py (937行)

RejectionSampler:
  → 6 Triton kernels (已在spec decode笔记详细分析)
  → → 改进版: 支持bonus token + probabilistic path + FP64 gumbel
  → → 与vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py共享kernel代码

关键区别:
  → v1/sample/rejection_sampler.py → 完整RejectionSampler类 → 调度+执行
  → v1/worker/gpu/spec_decode/rejection_sampler_utils.py → 6 Triton kernels → 底层

RejectionSampler流程:
  → target_logits → draft_probs → rejection_kernel → 拒绝/接受
  → → 贪心: exact match → 极快
  → → 概率: log_p > log(u) + log_q → 精确分布保证
  → → resample → residual distribution → Gumbel-max trick → 高效
```

## 9. RTX 4090 Sampling Pipeline Implications

```
1. all_greedy快速路径是RTX 4090最优:
  → 生产大部分greedy(T=0) → argmax一次 → ~0.06ms
  → → temperature=None → 省GPU复制 → 省带宽
  → → no_penalties → 省penalty计算 → 省带宽
  → → → 大部分step: argmax → 返回 → 几乎零采样开销

2. FlashInfer sampler:
  → RTX 4090(SM89) → FlashInfer支持 → GPU原生top-k/top-p
  → → 但greedy(T=0)不需要top-k/top-p → FlashInfer sampler不加速greedy
  → → → 随机采样(T>0) → FlashInfer 2-3x faster → 但RTX 4090生产很少T>0
  → → → → RTX 4090: greedy为主 → FlashInfer sampler几乎无关

3. Sampling overhead vs Attention:
  → Sampling: ~0.06ms (greedy) → vs Attention: ~0.22ms (FlashInfer B=32)
  → → Sampling不是瓶颈 → Attention才是 → FlashInfer更重要!
  → → → 但! penalty + spec decode → overhead叠加 → 可能~0.3ms → 仍不是瓶颈

4. Logits Processors对RTX 4090:
  → min_tokens → 生产几乎不用 → overhead negligible
  → logit_bias → 生产几乎不用 → overhead negligible
  → min-p → argmax_invariant → greedy不受影响 → 只T>0时生效
  → → → RTX 4090: 大部分处理器不激活 → overhead ≈0

5. Grammar Bitmask:
  → 两阶段设计 → execute_model + sample_tokens分离 → grammar在中间应用
  → → xgrammar → C++ FSM → bitmask → logits[invalid] = -inf → <1% overhead
  → → → RTX 4090: grammar bitmask几乎free → 生产可启用structured output

6. 生产最优采样配置:
  → T=0 (greedy) → all_greedy快速路径 → argmax → ~0.06ms → 最快
  → → FlashInfer attention (not sampler) → attention是瓶颈
  → → INT4+INT8KV+EAGLE → 推理最快 → 采样开销negligible
  → → → 结构化输出 → grammar bitmask → <1% → 生产标配

7. Sampling性能瓶颈分析:
  → Greedy: argmax → 0.06ms → 不瓶颈
  → Random: temperature+top-k+top-p+multinomial → 0.12ms → 2x greedy → 仍不瓶颈
  → Penalty: scatter+apply → 0.02ms → negligible
  → Spec decode: 6 Triton kernels → 0.5ms → 可能是瓶颈(但只在spec decode时)
  → → → 真正瓶颈: Attention (0.22ms) + Weight Reads (95.1% decode) → Sampling不瓶颈!
```

## 参考文献

```
1. vLLM V1 Sampling Pipeline源码:
   - vllm/v1/sample/sampler.py (439行) — Sampler核心类
   - vllm/v1/sample/metadata.py (55行) — SamplingMetadata
   - vllm/v1/sample/ops/penalties.py (57行) — Penalties
   - vllm/v1/sample/ops/topk_topp_sampler.py (515行) — TopKTopPSampler
   - vllm/v1/sample/ops/topk_topp_triton.py — Triton top-k/top-p
   - vllm/v1/sample/rejection_sampler.py (937行) — RejectionSampler
   - vllm/v1/sample/logits_processor/builtin.py (332行) — MinPLogitsProcessor

2. FlashInfer: sampling_ops.top_k_top_p_sampling — GPU原生采样
3. Min-P: Wang et al., "Min-P Sampling: Scaling Dynamic Truncation to Enhance LLM Generation", 2025

我们的笔记:
- vllm-v1-spec-decode-architecture-source-reading.md — Rejection Sampler详细
- vllm-v1-gpu-model-runner-source-reading.md — Two-phase执行(grammar bitmask)
- inference-sampling-deep-dive.md — 采样理论