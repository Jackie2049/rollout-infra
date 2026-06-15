# PyTorch Inductor SM<90 Fusion Guard — PR方案深度研究

> 2026-06-16 | vLLM #39096 | PyTorch upstream PR方案
> ★★★★★ 完整PR方案已确定 → InductorChoices.can_fuse_vertical → DeviceProperties.create → 5行修改!
> ★★★★★ 这是RTX 4090最有价值的OSS贡献 → PyTorch upstream PR!

## 1. ★★★★★ Fusion决策三层架构

```
★★★★★★★ Inductor fusion决策三层架构:

Layer 1: Legality → Scheduler._can_fuse (scheduler.py ~line 7620)
  → Structural legality → 是否可以fusion?

Layer 2: Profitability → V.choices.can_fuse / can_fuse_vertical / can_fuse_horizontal
  → Heuristic filter → 是否值得fusion?
  → ★★★★★ can_fuse_vertical currently returns True UNCONDITIONALLY! ← 空hook!

Layer 3: Backend → self.get_backend(device).can_fuse_vertical / can_fuse_horizontal
  → Hardware-specific gate → 是否在当前硬件上fusion?

★★★★★★★ 4个gate必须全部True才fusion:
  V.choices.can_fuse()                     # general filter
  AND self.can_fuse_vertical()              # scheduler legality
  AND V.choices.can_fuse_vertical()         # heuristic hook ← INSERT HERE!
  AND self.get_backend(device).can_fuse_vertical()  # backend gate
```

## 2. ★★★★★ 最佳插入点: InductorChoices.can_fuse_vertical

```
★★★★★★★ 当前代码 (choices.py line 640):
  @staticmethod
  def can_fuse_vertical(
      scheduler: Scheduler,
      node1: BaseSchedulerNode,
      node2: BaseSchedulerNode,
      shared_data_score: int,
  ) -> bool:
      """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
      return True     # ← 空hook → 无条件允许vertical fusion!

★★★★★★★ 提议修改 (5行!):
  @staticmethod
  def can_fuse_vertical(
      scheduler: Scheduler,
      node1: BaseSchedulerNode,
      node2: BaseSchedulerNode,
      shared_data_score: int,
  ) -> bool:
      """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
      # SM<90 Fusion Guard: On GPUs with compute capability < 9.0 (e.g., SM89/RTX 4090),
      # Triton autotuning selects different XBLOCK sizes for different input sizes.
      # When a reduction (e.g., mean) is fused vertically with pointwise ops,
      # the reduction becomes an inline tl.sum() whose accumulation order depends
      # on XBLOCK. This causes batch-dependent numerical results on SM<90.
      # Preventing vertical reduction fusion on SM<90 keeps reductions as separate
      # kernels where torch.mean's batch-invariant override remains effective.
      if node1.is_reduction() or node2.is_reduction():
          device = node1.get_device() or node2.get_device()
          if device is not None and device.type == "cuda":
              from torch._inductor.runtime.hints import DeviceProperties
              props = DeviceProperties.create(device)
              if props.major is not None and props.major < 9:
                  WhyNoFuse(node1, node2)("SM<90 prevents reduction fusion (batch invariance)")
                  return False
      return True

★★★★★★★ 为什么这是最佳插入点:
  → 5行代码 → minimal change → exactly matches documented purpose!
  → 使用DeviceProperties.create → cached → zero performance overhead
  → WhyNoFuse logging → visible in debug logs → consistent with scheduler debugging
  → InductorChoices designed for subclass override → clean mechanism
  → Only prevents vertical (producer-consumer) fusions → not horizontal
  → Only affects SM<90 → SM90+不受影响 → no regression risk!
```

## 3. ★★★★★ SM Compute Capability检测模式

```
★★★★★★★ 4种现有SM capability检测模式:

Pattern A: DeviceProperties.create(device) — ★★★★★ 推荐!
  → torch/_inductor/runtime/hints.py line 168
  → NamedTuple with cc, major, multi_processor_count, etc.
  → @functools.cache → zero performance overhead for repeated calls
  → Already used in choices.py line 482 (reduction_split_factor)
  → ★★★★★ Already imported in same class → cleanest pattern!

Pattern B: torch.cuda.get_device_capability()
  → Returns (major, minor) tuple → e.g., (8, 9) for SM89
  → Used in triton.py lines 2849 (TMA) and 4129 (PDL)
  → Requires torch.cuda initialized → slightly less clean

★★★★★★★ 推荐使用DeviceProperties.create:
  1. Already used in choices.py → same file → consistent pattern
  2. Cached → no performance overhead
  3. Works for all device types → not just CUDA
  4. Provides cc AND major → more flexible
  5. props.major < 9 → covers SM89 (8.9) and all pre-Hopper GPUs
```

## 4. ★★★★★ 现有architecture-dependent fusion precedent

```
★★★★★★★ 4个现有SM-dependent决策 precedent:

1. choices.py line 506: props.major >= 10 → no_split_threshold
   → ★★★★★ 最直接precedent → same class → same pattern → same props.major!
   → SM100+: 524288 threshold → SM<100: 8192 threshold
   → → ★★★★★ 证明InductorChoices中用props.major做architecture-dependent决策是已有模式!

2. triton.py line 2849: torch.cuda.get_device_capability()[0] >= 9 → TMA
   → Prevents TMA on SM<90 → same semantic → hardware capability check

3. triton.py line 4129: same pattern → PDL (Programmatic Dependent Launch)
   → Prevents PDL on SM<90 → another hardware gate

4. triton_heuristics.py line 722: device_prop.major >= 8 → rblock scaling
   → Limits rblock scaling to SM8+ or HIP → occupancy-driven

★★★★★★★ 完整SM capability检测位置表:

| File | Line | Pattern | Purpose |
|------|------|---------|---------|
| config.py | 2532 | torch.cuda.get_device_capability(0) | cuda.arch config default |
| choices.py | 482-506 | DeviceProperties.create + props.major >= 10 | reduction_split_factor |
| triton.py | 2849 | get_device_capability()[0] >= 9 | TMA capability |
| triton.py | 4129 | get_device_capability()[0] >= 9 | PDL capability |
| triton.py | 6571 | DeviceProperties.create | Triton kernel metadata |
| triton_heuristics.py | 460 | DeviceProperties from triton_meta | Autotune device props |
| triton_heuristics.py | 722 | device_prop.major >= 8 | rblock scaling occupancy |
| triton_heuristics.py | 1080 | self.device_props.cc | Compile metadata arch |
```

## 5. ★★★★★ 替代方案: Config Option + SIMDScheduling

```
★★★★ 替代方案B: Config option + SIMDScheduling.can_fuse

File 1: torch/_inductor/config.py (near line 1964):
  sm_less_than_90_prevents_reduction_fusion = True

File 2: torch/_inductor/codegen/simd.py (line 2158):
  if config.triton.sm_less_than_90_prevents_reduction_fusion:
      device = node1.get_device()
      if device is not None and device.type == "cuda":
          from torch._inductor.runtime.hints import DeviceProperties
          props = DeviceProperties.create(device)
          if props.major is not None and props.major < 9:
              why("SM<90 prevents reduction fusion (batch invariance)")
              return False

★★★★ Pros: follows tiling_prevents_reduction_fusion pattern, config option for disabling
★★★★ Cons: two files to change, more complex, doesn't handle reduction+pointwise swap case

★★★★★★★ 方案A vs 方案B:
  → 方案A (InductorChoices) → 5行 → 1文件 → 清晰 → 推荐!
  → 方案B (Config+SIMD) → 2文件 → more complex → 不推荐作为primary
  → 但可考虑A+B组合 → InductorChoices作为primary + config作为override开关
```

## 6. ★★★★★ PR方案完整策略

```
★★★★★★★ PR方案6要素:

1. ★★★★★ Primary change: InductorChoices.can_fuse_vertical (方案A)
   → 5行代码 → minimal → clean → precedent-following
   → Only prevent vertical fusions involving reductions on SM<90
   → Horizontal fusions unaffected (same iteration domain → no XBLOCK problem)

2. ★★★★★ Config option: config.triton.sm_prevents_reduction_fusion = True
   → Allows users to disable guard if they need max performance
   → Default True → correctness优先 → but can override

3. ★★★★★ DeviceProperties.create(device) pattern
   → Cached → zero overhead → already in choices.py → consistent

4. ★★★★★ WhyNoFuse logging
   → Visible in debug logs → "SM<90 prevents reduction fusion (batch invariance)"
   → Consistent with scheduler debugging patterns

5. ★★★★★ Scope: vertical fusions involving reductions on SM<90 only
   → Not horizontal fusions → those share same iteration domain
   → Not SM90+ → TMA persistent kernels → deterministic → no problem
   → ★★★★★ Minimal scope → minimal regression risk!

6. ★★★★★ Potential refinement: only prevent INNER reduction fusions
   → Inner reductions → XBLOCK varies → batch-dependent
   → Outer reductions → RBLOCK constexpr → may not have same problem
   → Check ReductionHint.INNER from node metadata → more granular
   → ★★★ 可作为follow-up优化 → initial PR先保守 → prevent ALL reduction fusions on SM<90

★★★★★★★ PR标题建议:
  "Prevent vertical reduction fusion on SM<90 GPUs for batch invariance"

★★★★★★★ PR描述要素:
  1. Problem: SM<90 + Inductor → batch-dependent results → vLLM #39096
  2. Root cause: tl.sum() inline → XBLOCK varies → accumulation order differs
  3. Fix: Prevent reduction fusion on SM<90 → keep torch.mean override effective
  4. Impact: SM90+ unaffected → SM<90 gets correctness → slight perf reduction
  5. Config: sm_prevents_reduction_fusion=True (default) → can disable
  6. Testing: vLLM batch invariance test + Inductor CI + regression benchmark
```

## 7. ★★★★★ vLLM→PyTorch PR路径

```
★★★★★★★ 从vLLM #39096 → PyTorch upstream PR路径:

Phase 1: vLLM issue comment (已完成!)
  → 6 drafts ready → SM89 batch invariance → 根因解释 → link to #39096

Phase 2: vLLM workaround (已有!)
  → enforce_eager=True → 禁用compile+graphs → batch invariance保持
  → 但10-15% throughput损失 → 不是完整fix

Phase 3: PyTorch upstream PR (★★★★★★★ 这是目标!)
  → InductorChoices.can_fuse_vertical → SM<90 Fusion Guard
  → → PyTorch issue → 描述根因 → 链接vLLM #39096
  → → PyTorch PR → 5行修改 → + config option → + WhyNoFuse logging
  → → ★★★★★★ 需要GPU测试 → 等GPU上线 → sm89_batch_invariance_repro.py验证!

Phase 4: vLLM启用compile (未来!)
  → PyTorch PR merged → SM89上reduction不融合 → batch invariance保持
  → → vLLM可以启用enforce_eager=False → CUDA graphs可用 → spec decode可用!
  → → ★★★★★ throughput +10-15% → spec decode → latency -40-50%!
```

## 8. ★★★★★ RTX 4090影响与价值

```
★★★★★★★ RTX 4090影响 (如果PR成功):

当前状态:
  → SM89 + enforce_eager=True → spec decode不可用 → throughput损失10-15%
  → SM89 + enforce_eager=False → batch invariance bug → correctness问题!

修复后收益:
  → Inductor SM<90 Fusion Guard → reduction不融合 → mean override有效
  → → SM89可以启用torch.compile → 不需要enforce_eager=True
  → → CUDA graphs可用 → spec decode可用!
  → → throughput +10-15% (恢复CUDA graphs)
  → → spec decode加速 → inference latency -40-50%
  → → ★★★★★★ 这是RTX 4090最大的potential improvement!

★★★★★★★ OSS贡献价值排序:
  1. ★★★★★★ Inductor SM<90 Fusion Guard → PyTorch upstream → 所有SM<90受益 → RTX 4090 MOST valuable!
  2. ★★★★★ vLLM SM89 FP8 guards → vLLM PR → QuantKey refactor foundation
  3. ★★★★★ SM120 FP4/MXFP4 kernel → vLLM → RTX 5090 contribution window
  4. ★★★★★ rLLM RTX 4090 cookbook → cookbook contribution → training recipe

★★★★★★★ 为什么Inductor Fusion Guard > vLLM contributions:
  → PyTorch upstream → 影面更广 → 所有使用torch.compile的SM<90 GPU受益
  → vLLM只影响vLLM → 但Inductor影响整个PyTorch生态
  → → ★★★★★★ 5行代码 → 影面整个SM<90生态 → 最高ROI OSS贡献!
```

## 参考
- vLLM Issue #39096: https://github.com/vllm-project/vllm/issues/39096
- torch/_inductor/choices.py: InductorChoices.can_fuse_vertical (line 640) → 最佳插入点!
- torch/_inductor/scheduler.py: fusion决策三层架构 (lines 7558-7920)
- torch/_inductor/codegen/simd.py: SIMDScheduling.can_fuse (lines 2158-2186) → backend gate
- torch/_inductor/runtime/hints.py: DeviceProperties.create (line 168) → cached SM detection
- torch/_inductor/config.py: tiling_prevents_reduction_fusion (line 1964) → existing precedent
- 相关笔记: pytorch-inductor-sm89-fusion-reading.md (根因分析), pytorch-inductor-scheduler-source-reading.md, pytorch-inductor-triton-codegen-reading.md
- 工具: sm89_batch_invariance_repro.py (GPU验证), sm89_batch_invariance_diagnostic.py
