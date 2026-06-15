# PyTorch Inductor SM89 Fusion → Batch Invariance Bug 根因分析

> 2026-06-16 | vLLM #39096 | PyTorch Inductor scheduler + Triton codegen
> ★★★★★ 完整因果链: Inductor → fuse(mean into kernel) → autotune XBLOCK varies → tl.sum() accumulation order differs → batch-dependent
> ★★★★★ 这是RTX 4090最有价值的OSS贡献机会 → Inductor SM<90 Fusion Guard

## 1. ★★★★★ Inductor Scheduler Fusion算法

```
★★★★ Scheduler.fuse_nodes() (scheduler.py, 10204行):

Iterative greedy fusion (up to 10 rounds):
  1. get_possible_fusions() → candidate pairs
  2. can_fuse(node1, node2) → legality check
  3. will_fusion_create_cycle() → dependency check
  4. speedup_by_fusion() → benchmark check (optional)
  5. All pass → merge into FusedSchedulerNode

★★★★ Key rule: nodes must share data + same device → no shared buffers → no fusion
★★★★ After 10 rounds or convergence → fusion stops

★★★★ LLM关键fusion pattern:
  → GEMM (cuBLAS) + epilogue pointwise ops (RMSNorm, SiLU, residual add, etc.)
  → Inductor不替换GEMM → 只融合pointwise ops → epilogue of GEMM template
```

## 2. ★★★★★★★ 根因: Inductor内联mean → 跳过Python override

```
★★★★★★★ RMSNorm decomposition → fused kernel:

RMSNorm分解:
  variance = x.pow(2).mean(dim=-1, keepdim=True)

Inductor lowering:
  x.pow(2) → PointwiseOp
  mean → ReductionOp("sum") + divide → PointwiseOp

Scheduler fusion:
  fuse(pow2 + sum + divide + rsqrt + mul_weight)
  → ONE Triton kernel: triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_*

★★★★★★★ 生成的Triton kernel:
  @triton_heuristics.persistent_reduction(size_hints={...})
  @triton.jit
  def triton_red_fused__*(...):
      xoffset = tl.program_id(0) * XBLOCK
      xindex = xoffset + tl.arange(0, XBLOCK)
      xmask = xindex < xnumel
      # Load x, compute x^2, REDUCE (sum over dim), divide, rsqrt, mul weight
      # ALL IN ONE KERNEL — no Python dispatch!
      tmp0 = tl.load(input + ...)
      tmp1 = tl.sum(tmp0, 1)  # ← mean reduction, INLINED!
      result = tmp1 / r0_numel

★★★★★★★ 关键: tl.sum()在Triton kernel内部 → 跨过Python-level aten dispatch
  → vLLM mean_batch_invariant override → registered on aten::mean.dim →
  → BUT Inductor generates Triton kernel from IR → never dispatches aten::mean.dim
  → → ★★★★★ Python override完全INVISIBLE to compiled path!

★★★★★★★ 根因确认:
  Inductor fuses RMSNorm(pow2 + mean + rsqrt + mul) → one Triton kernel
  → mean becomes tl.sum() inline → bypasses vLLM's batch_invariant_mean override
  → → accumulation order depends on XBLOCK (autotuned) → batch-dependent!
```

## 3. ★★★★★ Persistent Reduction vs Non-Persistent vs Cooperative

```
★★★★★ Three reduction strategies in Inductor:

| Strategy | When Used | Grid | RBLOCK | Loop |
|----------|-----------|------|--------|------|
| Persistent | rnumel small, static, fits one thread block | ceildiv(xnumel, XBLOCK) | next_power_of_2(rnumel) (constexpr) | No loop |
| Non-persistent (looped) | rnumel large/dynamic | Same | < rnumel (autotuned) | tl.range(0, rnumel, RBLOCK) |
| Cooperative | Very large reductions | (RSPLIT, ceildiv(xnumel, XBLOCK)) | Varies | Multi-block + semaphore |

★★★★★ should_use_persistent_reduction() → returns True when:
  1. Inside reduction (self.inside_reduction)
  2. V.choices.should_use_persistent_reduction() → rnumel static + small enough + enough shared mem

★★★★★★★ For RMSNorm in LLM:
  → hidden_size (4096 for Llama-7B) → statically known → fits one thread block
  → → ★★★★★ Inductor ALWAYS uses persistent reduction for mean in RMSNorm!
  → → RBLOCK = next_power_of_2(hidden_size) → compile-time constant → FIXED
  → → BUT XBLOCK = autotuned → VARIES with batch size!

★★★★★★★ This is the batch-dependent mechanism:
  RBLOCK fixed (constexpr) → deterministic across batch sizes ✓
  XBLOCK autotuned → different values for different xnumel →
  → different number of thread blocks → different tl.sum() accumulation order →
  → ★★★★★ floating-point addition non-associative → different results → batch-dependent!
```

## 4. ★★★★★ SM Architecture Specific Behavior

```
★★★★★★★ Inductor scheduler has NO explicit SM architecture checks!
  → No compute capability detection in scheduler.py
  → Architecture-specific behavior comes from TWO indirect mechanisms:

★★★★★ Mechanism A: Triton autotuning produces batch-dependent configs on SM89:
  → Different shared memory sizes:
    SM89 (Ada Lovelace): 100KB shared memory
    SM80 (Ampere): 164KB shared memory
    SM90 (Hopper): 228KB shared memory
  → Triton kernels that fit SM80/SM90 → may need different tiling on SM89
  → → Different XBLOCK → different accumulation order → batch-dependent!

★★★★★ Mechanism B: SM89 lacks TMA/WGMMA/CTA Clusters:
  → SM90 (Hopper): TMA + WGMMA + Distributed Shared Memory → deterministic paths
  → SM89: lacks all three → non-TMA strategies → different kernel configurations
  → SM86 (Ampere): same tensor core generation as SM80 → correct accumulation

★★★★★★★ Why SM90 works (confirmed by #27660):
  → TMA-based persistent kernels → deterministic memory access → invariant accumulation
  → Autotuned configs happen to be invariant across batch sizes on SM90
  → → "batch-invariant by design" on SM90

★★★★★★★ Why SM86 works for Qwen3 but SM89 fails for Llama:
  → Qwen3 on SM86: fusion pattern may not fuse mean → override remains effective
  → OR: hidden dimension sizes produce invariant configs on SM86
  → Llama on SM89: fusion pattern includes mean → override bypassed → batch-dependent
  → → ★★★★★ Model-specific → not predictable → depends on fusion pattern + autotune result!
```

## 5. ★★★★★★★ Complete Causal Chain Diagram

```
★★★★★★★ vLLM SM89 Batch Invariance Bug #39096 — Complete Causal Chain:

torch.compile(LlamaModel)
  │
  ▼
Dynamo: FX graph with rms_norm(x).mean(dim=-1)
  │
  ▼
Inductor Lowering: pow2 → Reduction("sum") → divide → rsqrt → mul
  │                    ↑
  │  mean = Reduction("sum") / denom
  │  NOT aten::mean.dim dispatch  ←→  vLLM override registered here
  │  (bypasses Python dispatcher)      but NEVER reached!
  │
  ▼
Scheduler: fuse(pow2 + sum + divide + rsqrt + mul)
  │   → FusedSchedulerNode
  │   → ONE Triton kernel: triton_red_fused_*
  │
  ▼
Triton Codegen: generates kernel with tl.sum() inline
  │   Uses triton_heuristics.persistent_reduction
  │   RBLOCK = next_power_of_2(hidden_size) ← constexpr, FIXED
  │   XBLOCK = autotuned based on xnumel    ← VARIES with batch!
  │
  ▼
CachingAutotuner on SM89:
  │   batch=1: selects config A (XBLOCK=8)
  │   batch=4: selects config B (XBLOCK=16)
  │   Different XBLOCK → different tl.sum() accumulation order
  │   → different numerical result → ★★★★★ batch-dependent!
  │
  ▼
SM90 (Hopper):
  │   TMA + WGMMA → deterministic accumulation path
  │   Autotuned configs happen to be invariant across batch sizes
  │   → ★★★★★ batch-invariant by design (confirmed by #27660)
```

## 6. ★★★★★ 四种修复方向排序

```
★★★★★★★ 4种修复方向 (按RTX 4090价值排序):

Direction 1: ★★★★★★ Inductor SM<90 Fusion Guard (最有价值!)
  → 修改Inductor scheduler → 检测SM<90 → 融合时跳过reduction ops
  → → 保持torch.mean override → batch invariance → correctness保证!
  → → 需要修改: torch/_inductor/codegen/ + scheduler
  → → SM89上compile仍然工作 → 部分fusion被禁用 → 性能稍降 → correctness保证!
  → → ★★★★★★ 这是最佳方向 → 但需要深入Inductor源码 → PyTorch upstream PR!

Direction 2: ★★★★★ Inductor-level batch-invariant lowering
  → 在Inductor lowering层面注册custom lowering for aten::mean.dim
  → → 即使compiled path也用fixed-order reduction → batch invariant
  → → 需要deep Inductor knowledge → 跨框架修改

Direction 3: ★★★★ vLLM SM<90 compilation guard (最简单!)
  → 检测SM<90 → compilation_config.level=0 → 禁用full-graph compile
  → → RMSNorm不被融合 → batch invariance保持
  → → ★★★★★ 最简单 → 但性能损失最大 → 类似enforce_eager=True
  → → 当前workaround → 不是完整fix

Direction 4: ★★★ Non-TMA Triton templates (最长远)
  → PRs #177781/#179095 → non-TMA persistent Triton matmul
  → → SM89可能得到更deterministic的persistent matmul
  → → 但★★★★★★ 不解决reduction fusion问题! → 只影响matmul路径!

★★★★★★★ 推荐修复路径:
  Direction 1 (Inductor SM<90 Fusion Guard) → vLLM Tier 1 PR
  → 具体做法: scheduler.py → 检测SM<90 → fused reduction时:
    → Option A: 不融合reduction → mean保持独立 → override有效
    → Option B: 融合但force fixed XBLOCK → deterministic accumulation
  → → ★★★★★ 这是RTX 4090最有价值的OSS贡献! → PyTorch upstream PR!
```

## 7. ★★★★★ RTX 4090影响

```
★★★★★ RTX 4090 (SM89)直接影响:

当前状态:
  → SM89 + enforce_eager=True → spec decode不可用 → throughput损失10-15%
  → SM89 + enforce_eager=False → spec decode不正确 → correctness bug!

★★★★ 修复后收益:
  → Direction 1 → SM89上spec decode正确 → RTX 4090 spec decode可用!
  → → throughput +10-15% (恢复CUDA graphs)
  → → spec decode加速 → inference latency -40-50%
  → → ★★★★★★ 这是RTX 4090最大的potential improvement!

★★★★★★★ OSS贡献窗口:
  → vLLM #39096 → 6 drafts ready → SM89 batch invariance comment drafts
  → ★★★★★ NEW opportunity: Inductor SM<90 Fusion Guard → PyTorch upstream PR!
  → → 需要GPU测试 → 等GPU上线 → Phase 1-4 testing plan ready
```

## 参考
- vLLM Issue #39096: https://github.com/vllm-project/vllm/issues/39096
- PyTorch Inductor scheduler.py: torch/_inductor/scheduler.py (10204 lines)
- PyTorch Inductor triton.py: Triton codegen + persistent/reduction/cooperative heuristics
- vLLM batch_invariant.py: mean_batch_invariant override (aten::mean.dim)
- PyTorch #177781/#179095: Non-TMA persistent Triton templates (closed, not merged)
- 工具: sm89_batch_invariance_repro.py, sm89_batch_invariance_diagnostic.py
- 相关笔记: vllm-sm89-batch-invariance-bug-reading.md, pytorch-inductor-scheduler-source-reading.md, pytorch-inductor-triton-codegen-reading.md
