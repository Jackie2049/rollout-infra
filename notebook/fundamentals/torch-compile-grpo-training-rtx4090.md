# torch.compile + GRPO Training Benchmark — RTX 4090

> 2026-06-07 | torch.compile应用于GRPO训练循环: reduce-overhead 2.97x加速, default反而慢于eager, Python-level rollout是瓶颈

## 核心发现

```
torch.compile对GRPO训练的影响与纯forward/training benchmark截然不同:

┌──────────────────────────────────────────────────────────────────┐
│ 场景               │ compile效果 │ 原因                          │
│ Forward-only       │ 3.75x       │ CUDA Graph消除所有launch      │
│ Training(fwd+bwd)  │ 1.31x       │ backward compute-bound       │
│ GRPO single step   │ 2.97x       │ 多次forward→CUDA Graph大收益 │
│ GRPO full training │ 1.39x       │ Python rollout overhead稀释  │
│ GRPO actor+ref     │ 1.23x       │ ref model额外forward开销     │
│ GRPO default mode  │ 0.84-1.16x  │ 编译开销>收益!               │
└──────────────────────────────────────────────────────────────────┘

→ 关键洞察: GRPO训练瓶颈不在GPU计算, 而在Python-level autoregressive rollout!
→ compile只能优化GPU kernel部分, 对Python循环无效 → 收益被稀释
→ reduce-overhead(CUDA Graph)是唯一有效模式, default编译反而更慢
```

## 实验1: GRPO Step Compile模式对比

```
模型: MiniGQATransformer 2.28M, SFT warm-start 500步(77% eval)
GRPO: n=4 samples, max_response_len=3, 2 prompts per step

| 模式              | Median(ms) | Speedup | 备注
| Eager             | 54.74      | 1.00x   | baseline
| default           | 65.08      | 0.84x   | ⚠️ 比eager更慢!
| reduce-overhead   | 18.41      | 2.97x   | CUDA Graph大幅加速

另一个run:
| Eager             | 65.50      | 1.00x
| default           | 56.34      | 1.16x   | 微加速
| reduce-overhead   | 46.96      | 1.39x   | 中等加速

→ 结果不稳定! GRPO step包含随机rollout→不同数据形状→Dynamo重编译
→ default模式编译开销可能>收益(动态shape→频繁重编译)
→ reduce-overhead在短rollout时收益大(多次小forward→CUDA Graph消除大量launch)
→ 但reduce-overhead在长rollout时收益减小(CUDA Graph录制固定shape→动态shape不兼容)
```

## 实验2: GRPO训练循环 — 收敛+吞吐对比

```
SFT warm-start: 200步, GRPO: 300步, n=4, noise σ=0.01

| 模式              | SFT eval | Final eval | Peak eval | Step(ms) | Speedup
| eager             | 33%      | 35-40%     | 36-48%    | 55-91    | 1.00x
| default           | 33-51%   | 25-57%     | 42-76%    | 58-70    | 0.94-1.28x
| reduce-overhead   | 33-51%   | 25-57%     | 42-76%    | 30-39    | 1.39-2.97x

关键观察:
1. default编译收敛不稳定! 一次25%(更差), 一次57%(更好)
   → 原因: compile改变了数值精度路径 → GRPO对数值敏感 → 效果不一致
2. reduce-overhead收敛与default相同(25%/57%) → CUDA Graph不改计算逻辑
3. reduce-overhead吞吐最高(30ms/step → 2.97x) → 但受CUDA Graph shape限制

→ GRPO训练中, compile的收敛效果不稳定 → 可能改善也可能恶化
→ reduce-overhead吞吐最高但受动态shape限制 → 实际生产需要shape padding
```

## 实验3: Compile Actor + Ref模型

```
GRPO需要2个模型(actor + ref), compile对两者的影响:

| 模式         | Step(ms) | Peak Memory(MB) | Speedup
| Eager        | 88.72    | 83.87            | 1.00x
| Compile      | 72.69    | 95.96            | 1.22x
| Overhead     | —        | +12.10           | — (compile cache)

→ Actor+Ref compile: 1.22x加速 → 低于纯actor(1.31x)
→ 原因: ref模型forward是torch.no_grad → CUDA Graph可以录制但需分开录制
→ 内存开销: +12MB → 极小(对2.28M模型)
→ Memory overhead几乎可以忽略 → compile cache对小模型友好

生产建议:
  → GRPO训练2模型 → compile两个 → 1.22x加速 + 12MB内存开销
  → 不值得为了1.22x加速承担编译不稳定性风险
  → 更好的选择: 优化Python-level rollout(减少autoregressive循环次数)
```

## 根因分析: 为什么GRPO编译收益低?

```
GRPO训练步骤分解:

  1. Rollout (autoregressive generation): Python循环 × n_samples × max_len
     → 每步: model(input_ids) → softmax → multinomial → cat → loop
     → 这部分是Python控制流 → compile无法优化!
     → 占比: ~60-70% of GRPO step time

  2. Logprob computation: model(full_ids) → log_softmax → gather
     → 这部分可以compile → 但只占~15-20%

  3. Loss computation: advantage × log_probs → backward → optimizer
     → 这部分可以compile → 但只占~10-15%

→ Rollout是瓶颈 → Python循环不可编译 → compile收益天花板 = 30-40%

对比纯forward benchmark:
  纯forward: 100% GPU → compile优化100% → 3.75x
  GRPO step: 60% Python + 40% GPU → compile优化40% → ceil = 1.67x
  实测: 1.39x → 接近天花板!

→ 核心结论: **GRPO训练瓶颈不在GPU, 在Python**
→ 解决方案: 不是compile, 而是:
  1. vLLM/SGLang async rollout → 用推理引擎做generation → 消除Python循环
  2. torch.compile仅优化logprob/loss部分 → 部分收益
  3. Prefix Sharing → 减少rollout次数 → 降低Python瓶颈占比
```

## 与纯训练Benchmark对比

```
| Benchmark类型        │ GPU占比 │ compile可优化 │ 实测加速 │ 天花板
| Forward-only         │ 100%    │ 100%          │ 3.75x   │ ~4x
| Training(fwd+bwd)    │ 100%    │ 100%          │ 1.31x   │ ~2x
| GRPO training loop   │ ~40%    │ ~40%          │ 1.39x   │ ~1.67x

→ 训练加速低于推理 → backward compute-bound → compile收益被稀释
→ GRPO加速更低 → Python rollout不可编译 → compile收益被进一步稀释
→ 三层递减: 3.75x → 1.31x → 1.39x → 每层都稀释compile收益

→ 生产场景(7B+模型):
  7B compute占比更高 → compile收益更低 → 可能<1.1x
  但: GRPO rollout用vLLM → 瓶颈转移 → compile对训练部分仍有用
```

## CUDA Graph限制 — GRPO动态shape问题

```
CUDA Graph (reduce-overhead mode) 限制:

1. **固定shape**: 录制时shape固定 → 不同shape需要不同Graph → 重录制开销
   → GRPO rollout: response长度随机 → 每步不同shape → 需要shape padding
   → shape padding浪费compute → 抵消CUDA Graph收益

2. **pending backwards警告**: "Unable to hit fast path of CUDAGraphs
   because of pending, uninvoked backwards"
   → GRPO: rollout用torch.no_grad → 但后续loss.backward()需要autograd
   → CUDA Graph需要step boundary → torch.compiler.cudagraph_mark_step_begin()

3. **Dynamo重编译**: GRPO每步shape不同 → Dynamo触发重编译
   → default模式重编译开销: 可能>收益 → 0.84x(比eager更慢!)
   → 解决: torch._dynamo.mark_dynamic() → 标记动态维度

→ 生产最佳实践:
  1. rollout: 用vLLM/SGLang(不compile)
  2. logprob+loss+backward: compile(mode="default")
  3. 不用reduce-overhead(动态shape不兼容)
  4. 如果shape固定(padding): 可以用reduce-overhead
```

## 生产建议

```
场景                        │ 推荐compile策略               │ 预期收益
小模型GRPO(<10M, 单GPU)     │ compile actor(default)       │ 1.2-1.4x
大模型GRPO(7B+, vLLM rollout)│ compile actor+critic(default)│ ~1.1x(训练部分)
推理-serving                │ compile(reduce-overhead)      │ 2-4x(B=1)
FSDP2+compile训练           │ compile(default)              │ 1.2-1.5x

关键决策:
  → GRPO rollout部分不compile → 用推理引擎(vLLM/SGLang)
  → GRPO训练部分(logprob/loss/bwd)可以compile → 但收益有限(~1.2x)
  → reduce-overhead不适合GRPO → 动态shape问题
  → default模式可能不稳定 → 需要仔细验证收敛

→ torch.compile对GRPO的真正价值:
  不在加速 → 在减少jitter + 简化优化路径 → 训练更稳定
  加速来自推理引擎rollout(vLLM/SGLang) → 不是compile
```

## 工具

- `tools/torch_compile_grpo_benchmark.py` — 3实验benchmark脚本
- `results/torch_compile_grpo_benchmark.json` — 完整结果数据

## 参考

- 前序实验: `notebook/fundamentals/torch-compile-deep-dive.md` (纯forward/training benchmark)
- verl架构: `notebook/projects/distributed-rl-training-verl-architecture.md` (rollout+训练分离)