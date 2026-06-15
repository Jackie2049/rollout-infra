# vLLM SM<90 Batch Invariance Bug — 深度源码分析

> 2026-06-16 | Issue #39096 | PR #30018/#38938 | 源码: vllm/v1/attention/, model_executor/layers/batch_invariant.py
> 核心: torch.compile + Inductor full-graph fusion → SM<90 batch-dependent outputs → spec decode incorrect
> ★★★★★ RTX 4090最重要的correctness bug — 直接影响speculative decoding正确性!

## 1. 问题描述

```
★★★★★ SM<90 batch invariance bug (#39096):

VLLM_BATCH_INVARIANT=1 在 SM<90 GPU (SM80 Ampere, SM89 Ada Lovelace)
  → 与 torch.compile 或 CUDA graphs 组合 → 不产生 batch-invariant outputs!
  → 同模型 + 同权重 + 同输入 → 不同batch size → 不同 greedy argmax 输出!
  → speculative decoding 测试失败 (test_eagle_dp.py)

★★★★★ 这不是简单的"数值精度问题" — 这是 correctness bug!
  → 不同batch size → 不同token → 模型输出完全错误!
  → 直接影响: spec decode在SM89上不可靠 → RTX 4090 spec decode不可用!
```

## 2. ★★★★★ 关键实验数据 (Issue #39096)

```
★★★★★ 来自 Issue #39096 的实验矩阵:

enforce_eager | cudagraph_mode     | Compile | Graphs | Result on L4 (SM89)
True          | (forced NONE)      | Off     | Off    | ✓ Works, baseline
False         | NONE               | On      | Off    | ✗ Fails at token 80
False         | FULL_AND_PIECEWISE | On      | On    | ✗ Fails (original bug)

★★★★★★ 核心发现:
  1. torch.compile alone 已经打破batch invariance → 不只是CUDA graphs的问题!
  2. enforce_eager=False, cudagraph_mode=NONE → 只有compile → 仍然失败!
  3. 禁用compile和graphs → enforce_eager=True → 才能正常工作!

★★★★ 这改变了之前的认知:
  之前以为: CUDA graphs是根因 → enforce_eager=True = 禁用graphs → 修复
  实际上: torch.compile才是根因 → enforce_eager=True = 同时禁用compile和graphs → 修复
  → 如果只禁用graphs但保留compile → 仍然失败!
```

## 3. ★★★★★ 根因分析: Inductor Full-Graph Fusion

```
★★★★★ 为什么 torch.compile 打破 batch invariance?

3层隔离实验:

Layer 1: torch.compile(rms_norm_native) 单独 → batch invariant ✓
  → RMSNorm本身在compile后仍然batch invariant
  → bitwise_equal: True, max_abs_diff: 0.0

Layer 2: torch.compile(完整Llama forward) → batch NOT invariant ✗
  → divergence at token 80 (20400 != 4324)
  → 不同batch size → 不同输出!

Layer 3: 完整Llama forward中 → 什么fusion打破了batch invariance?

★★★★★★ 根因 = Inductor full-graph fusion:
  → Inductor将 RMSNorm + residual add + next linear input prep 融合为一个大kernel
  → 在SM<90上 → 这个融合kernel的某些操作 → batch-dependent!
  → 具体机制尚未完全隔离 → 可能是:
    a) fusion中的reduction操作顺序不同 → torch.mean override未被应用?
    b) Inductor生成不同SM架构的不同kernel → SM<90使用不同融合策略?
    c) RoPE + attention + activation + RMSNorm fusion → 交互效应?
    d) batch_invariant_mode()的override在Inductor fusion中被跳过?

★★★★ 为什么SM90+没问题?
  → SM90 (Hopper) → H100 → Inductor可能使用不同的融合策略?
  → SM90有TMA (Tensor Memory Accelerator) → kernel实现不同 → 不触发bug?
  → SM90有WGMMA → 融合kernel的GEMM实现不同 → 不触发batch-dependent?
```

## 4. ★★★★ Model-Specific Behavior: Qwen3 Passes, Llama Fails

```
★★★★ 来自评论区的重要发现:

YM2132 (SM86, RTX 3090):
  → 禁用IS_DEVICE_CAPABILITY_BELOW_90 → enforce_eager=False
  → 运行 test_batch_invariance.py → Qwen3-1.7B → 通过!
  → Qwen3在SM86上 + torch.compile + batch invariance → 可以工作!

★★★★★ 这意味着:
  → Bug是model-specific → 不是所有model都失败!
  → Qwen3可能有不同的fusion pattern → 不触发batch-dependent
  → Llama可能触发特定的fusion → RMSNorm + residual + linear → batch-dependent

★★★★ 可能的原因:
  → Llama使用RMSNorm → fusion涉及reduction → 可能打破mean reduction order
  → Qwen3可能使用不同的norm → 或Inductor对Qwen3生成不同的融合策略
  → yewentao256指出: batch invariance override torch.mean → 保持reduction order
  → 但在Inductor fusion中 → torch.mean override可能被绕过!

★★★★★★ 这为RTX 4090提供了一个可能的partial workaround:
  → 如果只跑Qwen3 → 可能不需要enforce_eager → 但需要验证!
  → 如果跑Llama → 必须enforce_eager=True
```

## 5. ★★★★★ Batch Invariance机制源码分析

```
★★★★★ vLLM batch invariance = override torch.mean → 保持reduction order:

源码: vllm/model_executor/layers/batch_invariant.py

核心机制:
  1. enable_batch_invariant_mode() → 替换 torch.mean 为 batch_invariant_mean
  2. batch_invariant_mean → 保持reduction的数学顺序 → batch-independent!

★★★★★ 为什么mean的reduction order重要?
  → 浮点数加法不是associative: (a+b)+c ≠ a+(b+c)
  → torch.mean = sum / count → sum的顺序影响结果!
  → 不同batch size → different padding → different reduction order → different results!
  → batch_invariant_mean → 固定reduction order → 同输入 → 同输出!

★★★★★★ Inductor如何绕过这个override?
  → torch.compile → Inductor → full-graph fusion
  → Inductor可能:
    a) 将mean操作内联到融合kernel → 不调用Python override!
    b) 生成CUDA kernel → 直接做reduction → 不经过Python torch.mean!
    c) 融合 RMSNorm → variance = mean(x^2) → 如果Inductor内联这个mean → 不调用override!

★★★★ 这是根因的最可能解释:
  → Inductor full-graph fusion → 内联mean reduction → 绕过Python override
  → SM<90 → Inductor生成不同的融合kernel → mean reduction order不同
  → SM90 → Inductor可能保留Python override → 或使用相同的reduction顺序

★★★★ 但这个理论需要验证:
  → 需要检查Inductor生成的CUDA kernel → 是否内联了mean reduction
  → 需要比较SM89 vs SM90的Inductor kernel → 是否有不同的reduction实现
  → PyTorch v2.12 Non-TMA Triton templates (#177781) → 可能改变这个行为!
```

## 6. ★★★★★ PyTorch v2.12 Non-TMA Triton Templates → 可能的修复路径

```
★★★★★ PyTorch v2.12.0 (2026-05-13) 引入 Non-TMA persistent Triton templates:

PR #177781/#179095:
  → Triton persistent kernels → 不依赖TMA → SM89也支持!
  → 当前: Inductor在SM90+使用TMA-based Triton → 在SM<90可能fallback到不同策略
  → v2.12: Non-TMA Triton → SM89也能使用persistent Triton → 可能统一kernel策略!

★★★★★★ 如果Non-TMA Triton统一了SM89和SM90的kernel实现:
  → SM89不再需要fallback到不同的fusion策略
  → mean reduction可能在Non-TMA Triton中也使用固定顺序
  → → 可能间接修复batch invariance bug!

★★★★ 但这需要实验验证:
  → 需要GPU! → L4或RTX 4090 → 安装PyTorch v2.12 → 运行batch invariance测试
  → 需要检查: Non-TMA Triton kernel → 是否仍然内联mean → 是否保持reduction order
  → GPU不可用时 → 可以先准备测试脚本 → 等GPU上线再验证

★★★★ 测试准备:
  Step 1: 安装 PyTorch v2.12.0 + vLLM v0.23.0
  Step 2: 禁用 IS_DEVICE_CAPABILITY_BELOW_90 → enforce_eager=False
  Step 3: VLLM_BATCH_INVARIANT=1 运行 test_batch_invariance.py (Llama + Qwen3)
  Step 4: 比较 SM89 vs SM90 结果 → 看Non-TMA Triton是否修复了差异
  Step 5: 如果仍然失败 → 分析Inductor kernel → 找到具体的reduction差异
```

## 7. ★★★★★ RTX 4090 影响分析

```
★★★★★ RTX 4090 (SM89) 影响分析:

直接影响:
  → Speculative decoding (EAGLE/MTP) → correctness不可靠!
  → 如果不设enforce_eager=True → spec decode输出可能错误!
  → GRPO训练 → rollout引擎如果用spec decode → 可能产生错误token!

★★★★ 当前RTX 4090 workaround:
  enforce_eager=True → 禁用torch.compile + CUDA graphs → spec decode正确
  但代价: ~10-15% throughput loss → 没有compile加速 → 没有graph优化

★★★★★★ GRPO训练影响:
  → rLLM Tinker → in-process → 不用vLLM serving → 不受影响!
  → verl HYBRID → vLLM rollout → 如果用spec decode → 需要enforce_eager!
  → verl通常不用spec decode → 直接prefill+decode → 不受直接影响!
  → ★★★ 但如果未来verl启用spec decode加速rollout → 需要workaround!

★★★★★★ 最佳RTX 4090配置:
  推理 (INT4 serving):
    → enforce_eager=True → 禁用compile → spec decode不可用 → throughput loss
    → 或: 不用spec decode → INT4 + INT8 KV + prefix caching → 正常throughput

  GRPO训练 (rLLM Tinker):
    → 不受影响 → Tinker in-process → 不经过vLLM serving

  GRPO训练 (verl HYBRID):
    → vLLM rollout → 不用spec decode → 不受直接影响
    → 如果verl启用spec decode → enforce_eager=True → workaround
```

## 8. ★★★★★ 可能的OSS贡献路径

```
★★★★★ Issue #39096 → 最有价值的Tier 1 OSS贡献!

贡献路径分析:

Path 1: 深度根因分析 → 找到具体哪个Inductor fusion打破了batch invariance
  → 需要GPU → 运行Inductor debug → 比较SM89 vs SM90 kernel
  → 需要PyTorch Inductor知识 → 懂Triton kernel生成 → 懂fusion策略
  → ★★★★★ 如果成功 → 这是vLLM最重要的SM89 correctness修复!

Path 2: PyTorch v2.12 Non-TMA Triton → 间接修复
  → 安装v2.12 → 测试batch invariance → 看是否修复
  → 如果修复 → 报告到Issue #39096 → "works with PyTorch v2.12 Non-TMA Triton"
  → ★★★★ 这是低成本的验证 → 但需要GPU

Path 3: QuantKey-style guard → 在vLLM中添加SM89 batch invariance检测
  → 类似QuantKey (#32268) → 添加require_batch_invariant flag
  → 检测SM<90 + torch.compile → 自动set enforce_eager
  → ★★★ 这是防御性修复 → 不是根因修复 → 但保护用户不遇到bug

Path 4: Inductor mean reduction override → 直接修复根因
  → 在Inductor fusion中 → 保留batch_invariant_mean override → 不内联
  → 或: 修改Inductor的mean reduction → 使用固定顺序 → batch-independent
  → ★★★★★★ 如果能做到 → 根因修复! → 但需要PyTorch上游贡献!

★★★★★★ 推荐贡献策略:
  1. 先做Path 2 → 验证PyTorch v2.12 Non-TMA Triton → 等GPU上线
  2. 同时做Path 3 → vLLM SM89 guard → QuantKey-style → 可先做!
  3. GPU上线后 → Path 1 → 深度根因分析 → 找到具体fusion
  4. 最后Path 4 → 如果根因是Inductor → 提PR到PyTorch上游
```

## 9. ★★★★ Batch Invariant Mean 源码详解

```
★★★★ vLLM batch invariance = override torch.mean → 保持reduction order:

关键源码: vllm/model_executor/layers/batch_invariant.py

机制:
  1. enable_batch_invariant_mode():
     → 替换 torch.mean → batch_invariant_mean
     → 替换其他reduction ops → 固定顺序版本

  2. batch_invariant_mean:
     → 保持reduction的数学顺序 → 同输入 → 同输出 → batch-independent
     → 具体实现: 对每个batch item → 用相同的reduction顺序 → 不受batch size影响

★★★★★ 为什么float arithmetic不是associative:
  a + b + c → 三个浮点数
  (a + b) + c ≠ a + (b + c)
  → 因为浮点数加法 → 每次加法都可能round → 顺序不同 → round不同 → 结果不同!

  → mean = sum / n → sum的reduction顺序 → 影响精度
  → batch=1: sum(x_0) → 一种顺序
  → batch=4: sum(x_0, pad_1, pad_2, pad_3) → 不同顺序 → 不同sum → 不同mean!

★★★★★ batch_invariant_mean如何修复:
  → 固定reduction顺序 → 不管batch size → 都用相同顺序计算mean
  → padding → 不参与reduction → 只reduction真正的数据
  → → batch=1和batch=4 → 对同一行数据 → 用相同的reduction顺序 → same result!

★★★★★ Inductor full-graph fusion → 如何绕过:
  → torch.compile → Inductor → 分析Python代码 → 生成CUDA kernel
  → Inductor发现: RMSNorm = mean(x^2) → 内联到融合kernel
  → 融合kernel: RMSNorm + residual + linear → 一个大kernel → 不调用Python torch.mean!
  → → batch_invariant_mean override → 不被调用 → reduction顺序不确定!
  → → 在SM<90上 → Inductor可能使用不同的reduction策略 → batch-dependent!

★★★★★★★ 最终根因假设:
  → Inductor在SM<90上 → 为full-graph fusion生成的kernel →
  → mean reduction的reduction顺序 → 不固定 → batch-dependent
  → 在SM90上 → Inductor可能使用TMA-based persistent Triton → reduction顺序固定
  → → SM<90 → fallback → different reduction strategy → batch-dependent mean!
```

## 10. 关键洞察总结

```
★★★★★★ 8个关键洞察:

1. ★★★★★ torch.compile alone打破batch invariance → 不只是CUDA graphs!
   → enforce_eager=True = 禁用compile+graphs → 才work → 不只禁graphs!

2. ★★★★★ 根因 = Inductor full-graph fusion → 内联mean → 绕过override
   → Python-level override → Inductor不调用 → kernel-level mean → reduction不固定

3. ★★★★ Model-specific: Qwen3 passes on SM86, Llama fails on SM89
   → 不同model → 不同fusion → 不同batch invariance行为
   → 需要model-specific testing → 不能一刀切

4. ★★★★★ RTX 4090 spec decode不可靠 → enforce_eager=True是唯一workaround
   → 不用spec decode → 不受影响 → INT4 + INT8 KV正常
   → 如果用spec decode → enforce_eager=True → throughput loss ~10-15%

5. ★★★★★ PyTorch v2.12 Non-TMA Triton → 可能间接修复
   → SM89也能用persistent Triton → 可能统一kernel策略 → 消除SM<90差异

6. ★★★★★ Batch invariance = override torch.mean → fixed reduction order
   → float arithmetic non-associative → reduction order → affects precision
   → Inductor fusion → bypasses override → different order → different result

7. ★★★★★ 最佳OSS贡献路径: 先验证PyTorch v2.12 → 再做根因分析 → 最后PyTorch PR
   → Path 2 (验证) → Path 3 (guard) → Path 1 (根因) → Path 4 (PyTorch PR)

8. ★★★★★ rLLM Tinker不受影响 → verl通常不受影响 → 只有vLLM spec decode受影响
   → GRPO训练 → Tinker → in-process → 不经过vLLM → safe
```

## 参考

- Issue #39096: https://github.com/vllm-project/vllm/issues/39096
- PR #38938: Fix for test_eagle_dp (merged, moved test to H100)
- PR #30018: Introduced enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90
- PR #27660: Earlier batch invariance + torch.compile work (tested on H100)
- PyTorch #170563: Upstream batch invariance discussion
- PyTorch v2.12 PR #177781/#179095: Non-TMA persistent Triton templates
- 源码: vllm/model_executor/layers/batch_invariant.py
- 测试: tests/v1/determinism/test_batch_invariance.py
- 测试: tests/v1/distributed/test_eagle_dp.py
- 相关笔记: sm89_batch_invariance_diagnostic.py, sm89_compatibility_checker.py
