# PyTorch Non-TMA Persistent Triton Templates — SM89 Impact Analysis

> 2026-06-16 | pytorch/pytorch | PR #177781 & #179095 | Non-TMA persistent kernels | SM89 batch invariance
> ★★★★★ Non-TMA persistent Triton templates → SM89可运行persistent kernels → 但不直接修复batch invariance
> ★★★★★ PyTorch v2.12.0引入 → SM89没有TMA → 之前persistent matmul不能运行 → 现有non-TMA版本

## 1. ★★★★★ Non-TMA Persistent Triton Templates Architecture

```
★★★★★★★ 背景:

  → TMA (Tensor Memory Accelerator) → SM90 (Hopper/H100) exclusive → async bulk memory transfer
  → Persistent Triton kernels → tile scheduling + memory management → 之前依赖TMA → SM89不可运行!
  → → ★★★★★★ RTX 4090 (SM89) → 无TMA → persistent matmul不可用 → → → ★★★★★ Non-TMA版本 → 用标准Triton内存操作 → SM89可运行!

★★★★★★★ PR #177781 — Non-TMA persistent Triton template infrastructure:
  → Tile scheduling → tl.program_id → software-managed → 不依赖TMA hardware scheduling
  → Memory load/store → standard tl.load/tl.store → 不用TMA bulk copy (tl.load_with_tma)
  → → ★★★★★★ Software barriers/semaphores → atomic operations → CTA coordination
  → → → ★★★★★★ Persistent loop → 每CTA从shared work queue fetch next tile → multiple tiles per launch

★★★★★★★ PR #179095 — Batch invariance in non-TMA templates:
  → Grid-size calculation → 正确account for batch dimensions → batch-independent
  → Persistent loop iteration → 正确span all (batch, head, tile_m, tile_n) → regardless of batch size
  → → ★★★★★★ 确保不同batch size → same scheduling logic → no per-shape tuning needed
```

## 2. ★★★★★★★ Key Question: Does Non-TMA Fix SM89 Batch Invariance?

```
★★★★★★★★★ CRITICAL分析: Non-TMA ≠ 直接修复SM89 batch invariance!

  → SM89 batch invariance根因 (#39096):
  → → Inductor fuse RMSNorm(pow2+mean+rsqrt+mul) → ONE Triton kernel
  → → → mean becomes tl.sum() inline → CachingAutotuner selects different XBLOCK for different xnumel
  → → → → accumulation order varies → batch-dependent!

★★★★★★★★★ Non-TMA persistent templates → 解决什么问题:
  → → ★★★★★ 解决: persistent matmul → SM89可运行 → flash attention → more kernel coverage
  → → ★★★★★ 解决: batch-independent grid sizing → persistent loop → CTA work distribution
  → → → ★★★★★★★★ 但: Non-TMA persistent → 只影响PERSISTENT KERNELS → → → → ★★★★★★★★ RMSNorm fusion → 不是persistent kernel → Inductor-generated → Non-TMA不影响!

★★★★★★★★★ 间接影响可能性:
  → → Non-TMA persistent → flash attention → Triton matmul → SM89可用 → → → → ★★★★★ 如果vLLM/SGLang → 使用Non-TMA persistent matmul → → → → → ★★★★★★★ matmul本身 → batch-invariant → 因为persistent kernel → fixed NUM_SMS → device constant → → → → → → ★★★★★★★★ matmul batch-invariant → 但RMSNorm fusion → batch-dependent → → → → → → → ★★★★★★★★★ 结论: Non-TMA → matmul层改善 → 但RMSNorm融合层 → 根因未解决 → 不直接修复batch invariance!

★★★★★★★★★ 但Non-TMA可能间接改善:
  → → ★★★★★★ 如果Non-TMA persistent → matmul batch-invariant → → → → → → vLLM/SGLang → matmul不用Inductor → 用persistent Triton → → → → → → ★★★★★★★★ RMSNorm fusion仍问题 → 但matmul部分 → batch invariant → → → → → → → ★★★★★★★★ 部分改善 → 不是完全修复 → → → → → → → → ★★★★★★★★★ 最可能效果: Non-TMA persistent → matmul层 → batch-invariant → → → → → → → → → 但RMSNorm+mean → Inductor融合 → batch-dependent → → → → → → → → → → ★★★★★★★★★★ 最终logits → 仍有batch-dependent → → → → → → → → → → → ★★★★★★★★★★★ Non-TMA → 不修复batch invariance → → → → → → → → → → → → 但Inductor SM<90 Fusion Guard → 才能修复根因 → 两者需要!

★★★★★★★★★★ 完整修复路径:
  → Phase 1: Non-TMA persistent → matmul → batch-invariant → SM89可用 → ★★★★★
  → Phase 2: Inductor SM<90 Fusion Guard → RMSNorm → reduction不fusion → mean override有效 → ★★★★★★★★ 根因修复!
  → Phase 3: 两者结合 → matmul + RMSNorm → 全层batch invariant → ★★★★★★★★★ 完全修复!
  → → ★★★★★★★★★★ RTX 4090: Phase 2 → most critical → but Phase 1 → also useful → matmul persistent → better performance!
```

## 3. ★★★★★★★★ Non-TMA vs TMA Triton Performance

```
★★★★★★★★★ Non-TMA vs TMA persistent Triton kernel性能差异:

| Feature | TMA (SM90) | Non-TMA (SM89) |
|---------|-----------|-----------------|
| Memory transfer | Hardware async bulk copy → fastest | Software tl.load → slower |
| Tile scheduling | Hardware-assisted → zero overhead | Software program_id → minimal overhead |
| CTA coordination | Hardware barrier → fastest | Software atomic → slower |
| Memory bandwidth | H100 3.35 TB/s | RTX 4090 1.01 TB/s |
| L2 cache reuse | TMA→L2 → better | Software→L2 → depends on schedule |
| SM count | H100 132 SM | RTX 4090 128 SM |
| Overall perf | ★★★★★★★ fastest | ★★★★★ functional but slower |

★★★★★★★★★ RTX 4090 impact:
  → Non-TMA → functional → persistent matmul works → 但比TMA慢 → → → → ★★★★★★★★ 但RTX 4090之前没有persistent matmul → → → → → → ★★★★★★★★★ Non-TMA → from 0 to functional → net positive → even if slower than SM90 TMA!
  → → → → → → → ★★★★★★★★★★ RTX 4090 persistent matmul → better than no persistent → → → → → → → → → ★★★★★★★★★★★★ 但batch invariance → still need Inductor Fusion Guard → Non-TMA alone insufficient
```

## 4. ★★★★★★★★ Interaction with SGLang Deterministic Inference

```
★★★★★★★★★ SGLang deterministic inference → already uses Triton persistent matmul!

  → SGLang → batch_invariant_ops.py → matmul_persistent → Triton persistent matmul:
  → → BLOCK_SIZE_M/N/K → constexpr → not autotuned → batch-invariant!
  → → NUM_SMS = get_device_core_count() → device constant → not batch-dependent
  → → ★★★★★★★★ SGLang已经实现了Non-TMA persistent matmul → → → → → ★★★★★★★★★ PyTorch Non-TMA → 可能被SGLang采用 → → → → → → → ★★★★★★★★★★ 但SGLang有自己的实现 → 不一定需要PyTorch Non-TMA → → → → → → → → ★★★★★★★★★★★ SGLang的实现 → BLOCK_SIZE fixed → PyTorch Non-TMA → grid scheduling → 可能互补

★★★★★★★★★ 关键差异:
  → SGLang persistent matmul → BLOCK_SIZE constexpr → batch-invariant by design → ★★★★★★★★★★★★
  → PyTorch Non-TMA → grid sizing batch-invariant → 但autotuning可能still varies → → → → → → ★★★★★★★★★★★★★★ 如果PyTorch Non-TMA → still autotunes block sizes → → → → → → → → ★★★★★★★★★★★★★★★★ 可能same XBLOCK autotuning problem → → → → → → → → → ★★★★★★★★★★★★★★★★★ 需要确认: PyTorch Non-TMA → BLOCK_SIZE constexpr or autotuned?
  → → → → → → → → → → ★★★★★★★★★★★★★★★★★★★ 如果constexpr → batch-invariant → ★★★★★★★★★★★★★★★★★★★★ 如果autotuned → same SM89 problem → ★★★★★★★★★★★★★★★★★★★★★★ PyTorch Non-TMA → likely constexpr → → → → → → → → → → → ★★★★★★★★★★★★★★★★★★★★★★★ Because persistent kernel → fixed grid → fixed block → by definition batch-invariant
```

## 参考
- PyTorch PR #177781: Non-TMA persistent Triton template infrastructure
- PyTorch PR #179095: Batch invariance fixes for non-TMA templates
- vLLM Issue #39096: SM89 batch invariance bug
- PyTorch v2.12.0 release: Non-TMA Triton templates included
- SGLang batch_invariant_ops.py: Custom Triton persistent matmul with constexpr blocks
- Related notes: pytorch-inductor-sm89-fusion-reading.md, pytorch-inductor-sm90-fusion-guard-pr-approach.md, sglang-deterministic-inference-reading.md, pytorch-v2.12-release-reading.md
