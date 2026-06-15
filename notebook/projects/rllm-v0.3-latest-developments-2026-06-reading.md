# rLLM v0.3 Latest Developments (June 2026) — 3 Training Backends + Sandbox Architecture

> 2026-06-16 | rllm-org/rllm | PRs #579/#576/#577/#557 | Fireworks backend | Backend-agnostic step merge | Harbor 0.5
> ★★★★★ rLLM现在支持3个训练backend: Tinker (in-process) + verl (Ray) + Fireworks (cloud API)
> ★★★★★ Backend-agnostic step merge → shared MergedSegment → 跨backend代码复用!
> ★★★★★ RTX 4090: Tinker仍然#1 → 但verl backend也now natively supported → 灵活度提升

## 1. ★★★★★ 3 Training Backends Architecture

```
★★★★★★★ rLLM 3 training backends:

| Backend | Description | RTX 4090 | Notes |
|---------|-------------|----------|-------|
| Tinker | In-process, bypass default, auto LoRA, zero-copy | ★★★★★★★★ #1 | 最简单 → 最内存友好 |
| verl | Ray-based, vLLM/SGLang rollout, CPPO/ReMax | ★★★★★★ #2 | 大community → 多算法 |
| Fireworks | Cloud API, Firetitan policy update, deployment sampling | ★★★ ✗✗ | 需要cloud → RTX 4090不适用 |

★★★★★★★ 配置: backend="tinker"/"verl"/"fireworks" → AgentTrainer/UnifiedTrainer

★★★★★★★ Fireworks backend (PR #579 open, experimental):
  → rllm.experimental.fireworks → launcher + policy trainer
  → FireworksEngine → rollout support → deployment-based sampling
  → checkpoint resume/save → optimizer steps → deployment weight sync
  → → ★★★★★ Cloud-only → 不适用于RTX 4090 → 但展示了rLLM的backend可扩展性!
```

## 2. ★★★★★ Backend-Agnostic Step Merge (PR #576 open)

```
★★★★★★★ Unified step merge → rllm/experimental/common/step_merge.py:
  → MergedSegment → post-merge intermediate → prompt_ids + response_ids + response_mask + extras
  → merge_trajectory_steps() → walks Trajectory.steps → emits MergedSegment per cumulative-prefix run
  → TokenOps Protocol → typed against TokenInput → DefaultTokenOps → covers list[int] (verl) + Tinker adapters
  → ★★★★★★★★ Tinker和verl之前各自实现step merge → 现在统一 → 跨backend代码复用!

★★★★★★★ RTX 4090影响:
  → ★★★★★★★ Backend-agnostic → Tinker和verl共享merge逻辑 → bug修复一次 → 两个backend受益
  → → ★★★★★★★★★★★★★★★ 如果Tinker merge有bug → 修一次 → verl也修复 → 维护成本降低!
```

## 3. ★★★★★ Sandbox + Harness Architecture (Harbor 0.5)

```
★★★★★★★ rLLM sandbox architecture → Harbor-compatible:
  → PR #557 → Harbor 0.5 + --runtime flag → opencode/mini-swe-agent/oracle harnesses
  → sandbox backends → docker → snapshots → warm-queue → lifetimes → production-grade
  → ★★★★★ SWE-bench Pro + DeepSWE datasets (PR #651 merged) → sandbox builder → Harbor format
  → ★★★★★ 60+ benchmarks → via rllm dataset pull → registry

★★★★★★★ RTX 4090影响:
  → Sandbox → docker → CPU → 不需要GPU → RTX 4090可以run agent RL sandbox evaluation!
  → → ★★★★★★★★★★★★★★★★★★★ 但training → still需要GPU → GRPO → Tinker → GPU needed for rollout + training
  → → → → ★★★★★★★★★★★★★★★★★★★★★★★★★★★ sandbox evaluation → CPU → 可run → 但training → GPU → still offline!
```

## 参考
- rLLM PR #579: Fireworks backend (open, experimental)
- rLLM PR #576: Backend-agnostic step merge (open)
- rLLM PR #577: verl + Qwen3-Coder-30B MigrationBench (open)
- rLLM PR #557: Harbor 0.5 + opencode harnesses (open)
- rLLM PR #651: SWE-bench Pro + DeepSWE datasets (merged)
- rLLM PR #653: swe-rl cookbook (open)
- rLLM GitHub: https://github.com/rllm-org/rllm
- Related notes: rllm-architecture-reading.md, rllm-tinker-backend-deep-reading.md, rllm-v0.3-terminal-rl-reading.md
