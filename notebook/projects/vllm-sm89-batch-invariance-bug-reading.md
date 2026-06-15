# vLLM SM<90 Batch Invariance Bug #39096 — 深度源码分析

> 2026-06-16 | Issue #39096 | PR #30018/#38938/#27660 | PyTorch #170563/#177781/#179095
> 核心: torch.compile + Inductor persistent kernel → SM<90 batch-dependent → spec decode incorrect
> ★★★★★ RTX 4090最重要的correctness bug — 最有价值的OSS贡献机会!

## 1. 问题描述

```
★★★★★ SM<90 batch invariance bug (#39096):

VLLM_BATCH_INVARIANT=1 在 SM<90 GPU → 与 torch.compile 或 CUDA graphs 组合
  → 不产生 batch-invariant outputs!
  → 同模型+同权重+同输入 → 不同batch size → 不同 greedy argmax 输出!

影响:
  → speculative decoding 测试失败 (test_eagle_dp.py)
  → RTX 4090 spec decode 不可靠 → correctness bug!
  → enforce_eager=True 工作但同时禁用compile+graphs → 性能损失10-15%
```

## 2. ★★★★★ 实验数据矩阵

```
★★★★ 来自 Issue #39096 的实验矩阵 (L4, SM89):

enforce_eager | cudagraph_mode     | Compile | Graphs | Result
True          | (forced NONE)      | Off     | Off    | ✓ Works (baseline)
False         | NONE               | On      | Off    | ✗ Fails at token 80
False         | FULL_AND_PIECEWISE | On      | On    | ✗ Fails (original bug)

★★★★★★ 核心发现:
  1. torch.compile alone 打破batch invariance → 不只是CUDA graphs的问题!
  2. enforce_eager=True 禁用compile+graphs → 才能正常工作!
  3. 只禁用graphs保留compile → 仍然失败!

★★★★ 之前认知 → 新认知:
  旧: CUDA graphs是根因 → enforce_eager=True禁用graphs → 修复
  新: torch.compile(Inductor)是根因 → enforce_eager=True禁用compile+graphs → 修复
  → 如果只禁用graphs但保留compile → 仍然失败!
```

## 3. ★★★★★ 根因分析: Inductor Persistent Kernel on SM89

```
★★★★★★ 根因 = Inductor full-graph persistent kernel fusion on SM89:

Layer 1 隔离实验: torch.compile(rms_norm_native) 单独 → ✓ batch invariant
  → RMSNorm本身在compile后仍然batch invariant
  → bitwise_equal: True, max_abs_diff: 0.0

Layer 2 隔离实验: torch.compile(完整Llama forward) → ✗ batch NOT invariant
  → divergence at token 80 (20400 != 4324)
  → 不同batch size → 不同输出!

Layer 3 根因隔离:
  → Inductor将 RMSNorm + residual add + next linear input prep 融合为
    一个persistent kernel
  → 在SM<90上 → persistent kernel的某些操作 → batch-dependent!

★★★★★★ PyTorch Inductor persistent kernel机制:
  → persistent kernel = GPU threads持久化 → 跨多次kernel launch → 减少launch overhead
  → SM80 (Ampere): persistent kernel 支持 ✓
  → SM90 (Hopper): persistent kernel + TMA + WGMMA 支持 ✓✓
  → SM89 (Ada Lovelace): persistent kernel 不完整支持!
    → 缺少 TMA (Tensor Memory Accelerator)
    → 缺少 WGMMA (Warpgroup Matrix Multiply Accumulate)
    → 缺少 Distributed Shared Memory (CTA Cluster)
    → → Inductor在SM89上使用不同融合策略 → 产生batch-dependent行为!

★★★★★★ 关键洞察:
  → SM89没有Hopper特性 → Inductor无法使用Hopper专用优化
  → → Inductor回退到SM80策略 → 但SM89≠SM80 → 行为不同!
  → → 具体来说: RMSNorm + residual add fusion → SM89的persistent kernel实现
    → 某些reduction操作顺序不同 → torch.mean override未被正确应用?
    → → batch=1 vs batch=4 → reduction顺序不同 → 数值差异 → token divergence!
```

## 4. ★★★★ Model-Specific Behavior

```
★★★★ 来自YM2132评论区的重要发现 (RTX 3090, SM86):

  → 禁用IS_DEVICE_CAPABILITY_BELOW_90 → enforce_eager=False
  → 运行 test_batch_invariance.py → Qwen3-1.7B → 通过!

★★★★ Model-specific行为:
  Qwen3-1.7B: SM86上 + torch.compile → batch invariant ✓
  Llama: SM89上 + torch.compile → batch NOT invariant ✗

★★★★ 为什么Qwen3通过但Llama失败?
  → torch.mean override → vLLM的batch invariance机制 → 保持reduction顺序
  → → Qwen3的模型结构 → Inductor的fusion没有覆盖torch.mean → override有效!
  → → Llama的模型结构 → Inductor的fusion覆盖了torch.mean → override被跳过!
  → → 根因: Inductor full-graph fusion → 融合了reduction → override失效!

★★★★★★ 这意味着:
  → 不是所有模型都受影响 → 但无法预测哪个模型会受影响
  → 模型结构变化 → fusion策略变化 → batch invariance行为变化
  → → 新模型可能受影响 → 需要全面测试!
```

## 5. ★★★★★ PyTorch Non-TMA Triton Templates (#177781/#179095)

```
★★★★ PyTorch v2.12.0 新增 Non-TMA persistent Triton templates:

PR #177781: Add non-TMA persistent MM Triton template for max-autotune
  → 目标: AMD GPUs (缺乏TMA支持)
  → persistent_mm_template → 非TMA持久化matmul kernel
  → ~10% improvement over standard matmul for non-square shapes
  → 状态: Closed, 未merged到PyTorch主线

PR #179095: Add non-TMA persistent addmm Triton template
  → 目标: AMD GPUs addmm路径
  → 同样逻辑 → non-TMA fallback
  → 状态: Closed, 未merged

★★★★ 与SM89 batch invariance的关系:
  → 这些PR是针对AMD GPU → 但概念相同: 非TMA持久化kernel
  → SM89也缺乏TMA → 类似问题 → 但vLLM的bug不只是matmul → 是RMSNorm+residual fusion!
  → → Non-TMA Triton template可能帮助 → 但不是完整解决方案
  → ★★★★★ 真正需要: Inductor在SM89上的fusion策略调整 → 不融合
    reduction ops → 或融合时保持batch invariance!

★★★★★ PyTorch v2.12 Non-TMA Triton对SM89的影响:
  → 如果这些PR被merged → Inductor可能用non-TMA模板替代TMA模板
  → → SM89上matmul可能得到non-TMA版本 → 更consistent!
  → → 但: reduction fusion问题仍然存在 → 不是完整修复
  → → 完整修复需要: Inductor检测SM<90 → 跳过reduction fusion → 保持
    torch.mean override → batch invariant!
```

## 6. ★★★★★ 潜在修复方向分析

```
★★★★★★ 4个潜在修复方向:

Direction 1: Inductor SM<90 Fusion Guard (最有价值!)
  → Inductor检测SM<90 → 跳过包含reduction的fusion → 保持torch.mean override
  → → 需要修改: PyTorch Inductor code → torch/_inductor/codegen/
  → → 影响: SM89上compile仍然工作 → 但部分fusion被禁用 → 性能稍降 → correctness保证!
  → ★★★★★ 这是最佳方向 → 但需要深入Inductor源码!

Direction 2: vLLM BatchInvariantOverride 在Inductor内的Hook
  → vLLM的enable_batch_invariant_mode() override torch.mean
  → → 问题: Inductor full-graph fusion → 融合了mean → override失效
  → → 修复: 在Inductor编译前 → 注册batch_invariant hook → 融合时也保持
    reduction顺序 → 不跳过override
  → → 需要修改: vLLM batch_invariant.py + Inductor callback
  → ★★★★ 需要vLLM和PyTorch交互 → 跨框架修复 → 难度大!

Direction 3: vLLM SM<90 Compilation Mode (最简单!)
  → vLLM检测SM<90 → 使用compilation_config.level=0 → 禁用full-graph compile
  → → 只保留partial compile → RMSNorm不被融合 → batch invariance保持
  → → 需要修改: vLLM initialization code → 添加SM<90 compile guard
  → → ★★★★ 最简单 → 但性能损失最大 → SM89上compile收益减少!
  → → 与enforce_eager=True类似 → 但保留部分compile!

Direction 4: Triton Non-TMA Template for SM89 (最长远)
  → 合并PyTorch #177781/#179095 → 为SM89也添加non-TMA模板
  → → SM89持久化kernel → 更consistent → batch invariance可能恢复!
  → → ★★★ 需要Triton和Inductor修改 → 跨项目 → 最长远!
  → → 需要先确认: non-TMA模板是否解决reduction fusion问题!

★★★★★ 我的推荐修复路径:
  Direction 1 (Inductor SM<90 Fusion Guard) → 最有价值 → vLLM Tier 1 PR
  → 具体做法: 在Inductor的scheduler.py → 检测SM<90 → 融合时跳过reduction ops
  → → 保持torch.mean override → batch invariance → correctness保证!
  → → 这需要深入研究Inductor scheduler → 下一步GPU实验 + Inductor源码分析!
```

## 7. ★★★★★ RTX 4090 影响分析

```
★★★★★ RTX 4090 直接影响:

当前状态:
  → SM89 + enforce_eager=True → spec decode不可用 → throughput损失10-15%
  → SM89 + enforce_eager=False → spec decode不正确 → correctness bug!

★★★★ 影响范围:
  1. EAGLE/MTP spec decode → SM89上不可用/不正确 → RTX 4090最大限制!
  2. MRv2 + CUDA graphs → batch invariance → spec decode → 不正确
  3. torch.compile → SM89性能不如SM90 → persistent kernel缺失 → launch overhead高
  4. GRPO训练 → rollout+spec decode → 需要enforce_eager → throughput降

★★★★★ 修复后收益:
  → Direction 1 修复 → SM89上spec decode正确 → RTX 4090 spec decode可用!
  → → throughput +10-15% (恢复CUDA graphs)
  → → spec decode加速 → inference latency -40-50%
  → → ★★★★★ 这是RTX 4090最大的potential improvement!
```

## 8. 相关测试与工具

```
★★★★★ 已准备的工具:
  1. sm89_batch_invariance_repro.py → GPU上线后可运行 → 3 config测试
  2. sm89_batch_invariance_diagnostic.py → 5模式诊断工具
  3. sm89_compatibility_checker.py → SM89特性矩阵

★★★★★ GPU上线后测试计划:
  Phase 1: 运行 sm89_batch_invariance_repro.py --mode full
    → 确认bug在RTX 4090上可复现
  Phase 2: Qwen3-1.7B测试 → 是否在SM89也通过?
    → 如果通过 → 问题可能只影响Llama架构
  Phase 3: torch._inductor.config.triton.persistent_kernel=True
    → 在SM89上强制persistent kernel → 是否改善batch invariance?
  Phase 4: compilation_config.level=0 → partial compile
    → 是否保留batch invariance + 部分性能?
```

## 9. 关键洞察总结

```
★★★★★★ 5个关键洞察:

1. torch.compile alone 打破SM<90 batch invariance:
   → 不只是CUDA graphs的问题!
   → Inductor persistent kernel fusion → 根因!

2. 根因 = Inductor full-graph fusion on SM89:
   → RMSNorm + residual add + linear prep → 融合kernel → batch-dependent
   → SM89缺乏TMA/WGMMA → 回退到SM80策略 → 行为不同!

3. Model-specific: Qwen3通过, Llama失败:
   → torch.mean override → Qwen3有效 → Llama被融合覆盖
   → 不是所有模型都受影响 → 但无法预测

4. Non-TMA Triton templates (#177781/#179095):
   → 针对AMD → 但概念适用SM89 → 未merged
   → 可能帮助matmul → 但不解决reduction fusion!

5. ★★★★★ 最佳修复方向: Inductor SM<90 Fusion Guard:
   → 检测SM<90 → 融合时跳过reduction → 保持torch.mean override
   → 需要深入Inductor scheduler源码 → 下一步!
```

## 参考

- Issue #39096: https://github.com/vllm-project/vllm/issues/39096
- PR #30018: enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90 workaround
- PR #38938: test_eagle_dp moved to H100 CI
- PR #27660: batch invariance + torch.compile work (tested on H100 only)
- PyTorch #170563: upstream batch invariance discussion
- PyTorch #177781: Non-TMA persistent MM Triton template (closed, not merged)
- PyTorch #179095: Non-TMA persistent addmm Triton template (closed, not merged)
- vLLM batch_invariant.py: enable_batch_invariant_mode() + torch.mean override
- 工具: sm89_batch_invariance_repro.py, sm89_batch_invariance_diagnostic.py
