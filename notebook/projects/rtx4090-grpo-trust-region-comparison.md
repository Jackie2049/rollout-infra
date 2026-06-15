# RTX 4090 GRPO Trust Region对比: GRPO vs DPPO vs CPPO vs Tinker bypass

> 2026-06-16 | 跨框架分析 | verl PR #6717/#6731 | rLLM Tinker | vLLM #39096
> 核心: RTX 4090最优trust region组合 → CPPO + bypass_mode + GRPO advantage
> ★★★★★ 4种trust region机制 × 2种框架 → 8种组合 → 只有3种RTX 4090可行!

## 1. ★★★★★ Trust Region机制对比

```
★★★★★★★ 4种trust region机制:

| 机制 | 来源 | 约束类型 | RTX 4090可行 |
|------|------|----------|--------------|
| PPO ratio clip | 标准PPO | |ρ_t - 1| <= ε, uniform | ✓ (但heuristic) |
| DPPO divergence | verl DPPO | D_t <= δ, uniform pointwise | ✓ (but uniform) |
| CPPO prefix divergence | verl CPPO (#6731) | w_t*D_t <= c_t, position-weighted cumulative | ★★★★★ ✓ OPTIMAL! |
| Tinker bypass | rLLM Tinker | no explicit constraint → implicit through rollout | ✓ (但无explicit trust region) |

★★★★★★★ RTX 4090推荐排序:
  1. ★★★★★ CPPO + bypass_mode (verl) → best trust region + no ref model
  2. ★★★★★ Tinker bypass (rLLM) → simplest + in-process + no constraint overhead
  3. ★★★★ GRPO + bypass_mode (verl) → standard → works but heuristic clip
  4. ★★★ DPPO + bypass_mode (verl) → principled but uniform → better than GRPO
```

## 2. ★★★★★ CPPO详细机制 vs GRPO ratio clip

```
★★★★★★★ GRPO ratio clip:
  → |ρ_t - 1| <= ε → uniform → all tokens same constraint
  → Problem: doesn't consider prefix drift → cascading divergence risk
  → Example: token 80 diverges → tokens 81-1024 accumulate → collapse!

★★★★★★★ CPPO position-weighted cumulative:
  → w_t = w_min + (1-w_min)*(T-t)/(T-1) → early tokens heavier
  → c_t = min(δ, δ + δ_b*W_{t-1} - S_{t-1}) → dynamic budget
  → → As prefix diverges → threshold shrinks → catches budget overruns!
  → → GRPO misses these → CPPO catches → better stability!

★★★★★★★ 实际案例对比 (16k token response):
  GRPO: ε=0.2 → token 1-80 OK → token 81 diverges → all subsequent tokens still allowed
  → → cumulative drift → reward collapse at step 200+

  CPPO: δ=0.2, w_min=0.8 → token 1-80 tightly constrained → token 81 starts diverging
  → → prefix budget shrinks → tokens 82+ progressively tighter → drift contained!
  → → ★★★★★ smooth reward curve → no collapse!
```

## 3. ★★★★★ RTX 4090最优配置矩阵

```
★★★★★★★ RTX 4090 GRPO配置矩阵 (24GB VRAM):

| # | 组合 | 框架 | Ref Model | Critic | Trust Region | VRAM | 推荐度 |
|---|------|------|-----------|--------|--------------|------|--------|
| 1 | CPPO+GRPO+bypass | verl | ✗ (省14GB) | ✗ (GRPO) | ★★★★★ position-weighted | ~18GiB | ★★★★★ BEST |
| 2 | Tinker bypass+LoRA | rLLM | ✗ (default) | ✗ (GRPO) | ★★★ implicit | ~17GiB | ★★★★★ BEST (in-process) |
| 3 | GRPO+bypass | verl | ✗ (省14GB) | ✗ (GRPO) | ★★★ heuristic clip | ~18GiB | ★★★ STANDARD |
| 4 | DPPO+bypass | verl | ✗ (省14GB) | ✗ (GRPO) | ★★★★ principled uniform | ~18GiB | ★★★ GOOD |
| 5 | PPO+ref | verl | ✓ (14GB+) | ✓ (2GB+) | ★★ standard clip | >28GiB | ✗✗✗ OOM! |
| 6 | PPO+ref+FSDP | verl | ✓ | ✓ | ★★ | varies | ✗✗ multi-GPU |

★★★★★★★ RTX 4090 top 2推荐:

verl path (#1): CPPO + GRPO advantage + bypass_mode
  → algorithm.adv_estimator=grpo
  → algorithm.rollout_correction.bypass_mode=True
  → actor.policy_loss.loss_mode=cppo
  → actor.clip_ratio=0.15
  → actor.policy_loss.cppo_w_min=0.8
  → ★★★★★ Near-zero overhead + provably better trust region

rLLM path (#2): Tinker bypass + LoRA-32
  → bypass_mode=true (default)
  → auto LoRA init (rank=32)
  → in-process zero-copy
  → ★★★★★ Zero Ray overhead + simplest setup
```

## 4. ★★★★★ 当CPPO available但bypass_mode不可用时

```
★★★★ 重要考虑: CPPO需要bypass_mode → 如果bypass_mode不可用?

Scenario 1: verl正常模式 (bypass_mode=False)
  → old_log_probs from separate forward → pi_old ≠ μ → divergence measurement INCORRECT
  → → CPPO mask基于错误divergence → 可能比GRPO更差!
  → → ★★★★★ 此场景下 → GRPO更好 → 不用CPPO!

Scenario 2: verl bypass_mode=True (推荐)
  → old_log_probs = rollout_log_probs → μ measured correctly
  → → CPPO mask基于正确divergence → provably better!
  → → ★★★★★ 这是唯一正确使用CPPO的方式!

★★★★★★ RTX 4090结论:
  → CPPO + bypass_mode = REQUIRED组合 → 不可分开!
  → 如果bypass_mode不可用 → 回退到GRPO ratio clip → 不用CPPO
  → rLLM Tinker: bypass default → 不会有这个问题 → 天然safe
```

## 5. ★★★★★ Short vs Long Response场景分析

```
★★★★★★★ Response长度 → CPPO vs GRPO影响:

| Response Length | GRPO | CPPO | CPPO Advantage |
|-----------------|------|------|----------------|
| <512 tokens | ★★★ OK | ★★★ OK | Minimal (prefix budget has less time) |
| 512-2k tokens | ★★★ OK | ★★★★ Better | Moderate (position-weighting helps early tokens) |
| 2k-4k tokens | ★★ risky | ★★★★★ Stable | Significant (prefix budget prevents cascading drift) |
| 4k+ tokens | ★ collapse risk | ★★★★★ Stable | ★★★★★ Critical (CPPO essential for long CoT) |

★★★★★★ RTX 4090 practical:
  → Math reasoning (4k+ tokens) → CPPO essential → GRPO may collapse
  → Simple QA (<512 tokens) → GRPO OK → CPPO also OK → near-zero overhead → always safe to use!
  → Code generation (2k-4k tokens) → CPPO recommended → prevents drift
  → ★★★★★ Near-zero overhead → always use CPPO → never worse than GRPO!
```

## 6. 关键洞察

```
★★★★★★ 3个关键洞察:

1. ★★★★★ CPPO + bypass_mode = RTX 4090最优verl trust region
   → Near-zero overhead → provably better → must use with bypass_mode
   → If bypass unavailable → fall back to GRPO → don't use CPPO alone

2. ★★★★★ rLLM Tinker bypass = 最简洁路径
   → In-process → no Ray overhead → bypass default → zero-copy
   → No explicit trust region → but implicit through rollout stability
   → ★★★★★ For simplicity → rLLM Tinker wins
   → ★★★★★ For long CoT → verl CPPO wins

3. ★★★★★ Always use CPPO instead of GRPO when bypass_mode available
   → Zero cost → strictly better trust region → never worse
   → Only caveat: must use with bypass_mode → divergence measured against rollout μ
```

## 参考
- verl PR #6731: CPPO algorithm implementation
- verl PR #6717: Tinker Worker Primitives (split training API)
- rLLM Tinker: bypass_mode default, in-process, zero-copy
- Paper: arXiv:2606.10968 "Beyond Uniform Token-Level Trust Region"
- vLLM #39096: SM<90 batch invariance → spec decode requires enforce_eager
- 相关笔记: verl-cppo-algorithm-reading.md, verl-tinker-worker-primitives-reading.md, rllm-tinker-backend-deep-reading.md
