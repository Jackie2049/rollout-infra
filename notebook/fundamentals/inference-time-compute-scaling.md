# Inference-Time Compute Scaling: AI Infra的新范式

> 2026-06-15 | 基于最新研究+行业趋势+7框架视角
> 核心: 推理时投入更多计算(思考更长)→推理成本将超越训练成本→推理infra重要性急剧上升

## 1. 范式转变: 从训练scaling到推理scaling

```
2020-2024: Training-time scaling → 大模型+大数据+大算力
  → GPT-3(175B) → GPT-4(~1.8T) → 参数和数据scaling
  → 成本: 训练$100M+ → GPU集群数千卡

2025-2026+: Inference-time scaling → 推理时思考更长→质量更高
  → o1/o3/Claude extended thinking → chain-of-thought reasoning
  → 成本: 推理$0.01-0.1/query → 但reasoning query 5-10x更贵!
  → 推理总成本将超越训练总成本(量×价效应)
```

## 2. 推理时计算方法

| 方法 | 描述 | 计算倍增 | 适用场景 |
|------|------|---------|---------|
| **Extended thinking** | 内部推理链→更多token→更高质量 | 5-50x (token增加) | 数学/编程/科学推理 |
| **Best-of-N sampling** | 生成N个候选→选最优 | N×(N=4-64) | 开放性任务 |
| **Self-verification** | 模型检查自己答案→修正 | 2-3x (验证pass) | 可验证任务(数学/编程) |
| **Tree search (MCTS)** | 多分支推理→选最优路径 | 10-100x (分支×深度) | 复杂规划/推理 |
| **Iterative refinement** | 多次迭代→逐步改进 | 3-10x (迭代次数) | 写作/代码 |

### Extended Thinking详解

```
传统推理: prompt → 1次forward → response → 完成(~500 tokens)
Extended Thinking: prompt → 内部reasoning(~5000 tokens) → response → 完成(~5500 tokens)

计算量: 7B模型 decode 5500 tokens
  → 5500 × 14GB(weight reads per token) × 169.6 TFLOPS throughput
  → ~5500/4791 tok/s ≈ 1.15秒 (INT4+INT8KV)
  → vs 传统: 500/4791 ≈ 0.10秒 → 11.5x更慢!

  但: 输出质量大幅提升 → 数学准确率从60%→90%!
  → 用户愿意付10x价钱获得3x更好的答案
```

### Best-of-N Sampling详解

```
生成N=8个候选 → reward model评分 → 选最高分

计算量: N× forward
  → 8 × 500 tokens × 14GB × 169.6 TFLOPS
  → 约8×0.10秒 = 0.8秒 (可并行!)

  并行加速: 8个候选同时在8GPU上生成
  → 单GPU: 0.8秒
  → 8GPU并行: 0.10秒(与单次相同!) → 最适合多GPU推理!

  GRPO训练: Best-of-N是GRPO的rollout_n!
  → rollout_n=8 → 8个候选 → group mean/std normalization
  → 推理时也用best-of-N → 训练和推理对齐!
```

## 3. 对7框架的影响

| 框架 | 推理scaling影响 | 关键变化 |
|------|----------------|---------|
| **DeepSpeed** | 推理权重增加 → ZeRO-3推理? → 但ZeRO推理无优势 → vLLM替代 | 训练仍重要但占比下降 |
| **Megatron-LM** | TP推理加速 → 但MoE推理更复杂 → EP推理需求上升 | MoE推理需要All-to-All |
| **vLLM** | **最大受益者!** → 推理需求5-10x增长 → 每query更多token → KV cache压力增大 | 需要更大KV cache+更长序列支持 |
| **verl** | GRPO rollout_n=推理scaling → 训练和推理一体化 | rollout engine(vLLM)需求增加 |
| **rLLM** | @rllm.rollout=推理+训练 → 推理scaling直接嵌入训练loop | SamplingClient需要更长推理时间 |
| **MindIE** | Ascend推理优化 → 华为NPU推理需求增加 | ATB kernel优化更重要 |
| **PyTorch** | compile推理加速 → max-autotune推理优化 → 推理infra新需求 | inference mode需要重新优化 |

### vLLM具体影响

```
推理scaling → vLLM关键挑战:

1. KV cache容量: longer reasoning → 更多tokens → KV cache更大!
   → 7B BF16: 5500 tokens × 2×hidden×2bytes ≈ 22MB → 单request可!
   → 但batch推理: 8×5500 = 44K tokens → KV cache 176MB → 需要更大cache!
   → INT8 KV: 减半 → 88MB → RTX 4090 24GB够(但batch受限)

2. 序列长度限制: 5500+tokens → 超标准2048/4096
   → 需要long context优化: Ring Attention/Sequence Parallel推理
   → vLLM chunked prefill → 需要更大chunk size!

3. Decode throughput: longer reasoning → decode time更长
   → memory-bound → INT4+INT8KV唯一出路
   → RTX 4090: 4791 tok/s → 5500 tok/1.15s → 单request延迟1.15s → 可接受!
   → 但batch: 8×5500→44K tokens→8×1.15s→需要speculative decoding加速

4. PD分离: reasoning模型更需要prefill/decode分离
   → prefill短(prompt~500tok) → decode长(reasoning~5500tok)
   → prefill GPU利用率低 → 专门prefill GPU更高效!
   → SGLang Overlap调度: prefill+decode overlap → 更适合reasoning模型
```

## 4. 推理vs训练成本趋势

```
2024: 训练成本 >> 推理成本
  → GPT-4训练: $100M → 推理: $0.03/query → 总推理成本<训练(量不够大)

2025+: 推理成本 >> 训练成本
  → 原因1: 推理scaling → 每query 5-10x更贵
  → 原因2: 用户量增长 → 日推理量10B+ queries
  → 原因3: reasoning模型 → 每response 5500 tokens vs 500 tokens
  → 估算: 10B queries × 5500 tokens × $0.0001/token = $550M/月 推理成本!
  → vs 训练: $100M一次性 → 推理成本远超训练成本(月度!)

结论: 推理infra的重要性将超越训练infra!
  → vLLM/SGLang inference optimization → 最有价值的技能!
  → 但训练infra仍重要(GRPO训练是reasoning模型的基础)
  → 最佳路径: 训练infra(基础)+推理infra(差异化)
```

## 5. 推理scaling与GRPO训练的关系

```
推理scaling方法 → GRPO训练的rollout机制:

Best-of-N推理 → GRPO rollout_n
  → 推理时: 生成8个候选→选最优
  → 训练时: GRPO rollout_n=8→8个候选→group normalization
  → 完美对齐! 推理方法=训练方法!

Extended Thinking推理 → verl/rLLM rollout engine
  → 推理时: longer reasoning chain
  → 训练时: rollout engine生成reasoning chain→reward model评分
  → 需要更长序列推理 → vLLM需要支持!

Self-verification → GRPO reward function
  → 推理时: 模型自检→修正
  → 训练时: reward function包含verification accuracy
  → reward shaping更精细!

结论: 推理scaling和GRPO训练是同一系统的两个面!
  → 推infra = 推训练infra + 推推理infra
  → verl/rLLM: 训练+推理一体化 → 最适合reasoning模型开发!
```

## 6. AI Infra工程师的新优先级

```
2024优先级:
  1. 训练infra(ZeRO-3, FSDP2, Megatron) → 40%
  2. 推理infra(vLLM, SGLang) → 30%
  3. RL训练(verl, rLLM) → 20%
  4. 系统优化(CUDA, Triton) → 10%

2025-2026+优先级(预测):
  1. 推理infra(vLLM, SGLang, INT4, PD分离) → 40% ↑↑
  2. RL训练(verl, rLLM, GRPO) → 25% ↑↑
  3. 训练infra(FSDP2+compile) → 20% ↓↓
  4. 系统优化(CUDA, Triton, compile) → 15% ↑

关键变化:
  → 推理重要性从30%→40%(reasoning模型需求)
  → RL训练从20%→25%(GRPO是reasoning训练的核心)
  → 传统训练从40%→20%(全参数训练需求减少)
  → 系统优化从10%→15%(compile+Triton+INT4)

→ 作为AI infra工程师: 推理+RL是最有价值的方向!
→ 但训练infra是基础 → 不应忽视 → 只是需要调整权重!
```

## 7. RTX 4090推理scaling实战

```
RTX 4090 24GB (7B模型):

Scenario 1: Extended Thinking推理
  INT4+INT8KV+GQA-8: 4791 tok/s → 5500 tok/1.15s
  BF16推理: ~500 tok/s → 5500 tok/11s → 太慢!
  → INT4推理唯一出路!
  → 但INT4精度→reasoning质量可能下降 → 需验证!

Scenario 2: Best-of-N=8 并行推理
  单GPU: 8×500tok × INT4 → 0.83s → 可行
  但: 8个候选KV cache → 8×22MB=176MB → 加INT4 KV=88MB → 加模型4GB → 4.08GB → fit!
  → RTX 4090可以同时推理8个候选(batch=8)!

Scenario 3: Speculative Decoding加速reasoning
  EAGLE → 9,088 tok/s → 5500 tok/0.6s → 加速2x!
  → 但EAGLE需要draft model内存 → 7B+2B draft=4+1=5GB → INT4 KV=0.88GB → fit!
  → 最优推理方案: INT4+INT8KV+GQA+EAGLE → 5500 tok/0.6s

Scenario 4: Long context reasoning
  5500+tokens → KV cache:
  BF16: 5500×2×4096×2 = 88MB → 加模型14GB → 14.088GB → barely fit!
  INT8 KV: 44MB → 加模型INT4 4GB → 4.044GB → 很好!

  但batch推理同时8个5500-token request:
  INT8 KV: 8×44MB=352MB → 加INT4模型4GB → 4.352GB → fit!
  → RTX 4090可以batch推理reasoning模型!
```

## 8. 对AI专家长远目标的启示

```
成为资深AI专家 → 需要理解推理scaling趋势:

1. 推理scaling是2025-2026最重要的AI范式转变
   → 不只是"训练大模型" → 而是"推理时思考更长"
   → 推理infra技术将是最稀缺的技能

2. GRPO训练是reasoning模型的训练方法
   → 推理scaling方法(best-of-N, verification)→ GRPO rollout
   → 推理=训练的上层应用 → 两者不可分割

3. 推理成本将超越训练成本
   → 推理优化(INT4/PD分离/KV压缩) → 经济价值巨大!
   → vLLM/SGLang推理优化 → 最实用的技能

4. RTX 4090推理完全可行
   → INT4+INT8KV+GQA → 4791 tok/s → reason~5500 tok/1.15s
   → batch推理8个候选 → 4GB → fit!
   → 推理是RTX 4090的最大优势(vs训练内存受限)

5. 技能组合建议
   → 核心: 推理infra(vLLM/SGLang优化)
   → 基础: 训练infra(FSDP2+compile/GRPO)
   → 加分: 系统优化(CUDA/Triton/compile/量化)
   → 战略: 推理scaling理解(best-of-N/MCTS/verification)
```

---

Sources:
- "Scaling Laws for Inference-Time Compute" research paper (2024)
- OpenAI o1 model strategy
- Epoch AI analysis on inference scaling
- notebook/projects/rtx4090-seven-framework-practical-config.md
- memory/benchmark-results.md
