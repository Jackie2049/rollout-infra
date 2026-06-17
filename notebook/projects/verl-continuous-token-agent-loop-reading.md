# verl ContinuousToken Agent Loop (#6720/#6721 open) — Multi-Turn RL

> 2026-06-16 | verl-project/verl | PR #6720 + #6721 | Continuous Token Builder | AgentLoop | Multi-turn RL
> ★★★★★ ContinuousToken → append-only token stream → multi-turn agent RL → 不重复构造previous assistant text
> ★★★★★ Disabled by default → multi_turn.continuous_token.enable=True → 现有rollout行为不受影响
> ★★★ RTX 4090: short-term不需要 (GRPO single-turn) → medium-term可能有价值 (agent RL)

## 1. ★★★★★ ContinuousToken Architecture

```
★★★★★★★ PR #6720 — Continuous Token Builder Core:
  → Runtime token stream → append-only → across turns → 不重复构造previous assistant text
  → ContinuousToken disabled by default → multi_turn.continuous_token.enable=True
  → AgentLoop代码 → preserve generated assistant tokens → append tool/user/system → through CT builder
  → ★★★★★ Append-only → 不重建 → 省memory + 省compute → KV reuse → prefix cache最大化!

★★★★★★★ PR #6721 — Agent Loop Integration:
  → CT wired into existing AgentLoop → text-only → multimodal still legacy
  → AgentLoop → builds initial prompt with CT builder → appends generated assistant tokens
  → merges tool/user/system messages through CT → keeps response_mask + rollout logprobs aligned
  → ★★★★★ Multimodal processor → continues legacy → not yet CT-enabled

★★★★★★★ Key design decisions:
  → Append-only token stream → 上一turn的tokens → 保留 → append新tokens → KV reuse
  → Multi-turn → agent loop → tool calls → response → tool response → next turn
  → ContinuousToken → 保留整个trajectory → append-only → 不truncate
  → response_mask → alignment → 每turn的response tokens → 正确标记 → training正确
```

## 2. ★★★ RTX 4090 Impact

```
★★★★★★★ RTX 4090 ContinuousToken可行性:

  → ★★★★★ Append-only → 不重建 → KV reuse → 省内存 → RTX 4090 24GB → 内存友好
  → ★★★ 但: multi-turn agent RL → 长trajectory → KV累积 → 内存可能反而增大
  → → 需要prefix-tree dedup (MAGI) → 才能真正省内存 → CT + MAGI = 中期组合

★★★★★★★ 短期: GRPO (single-turn) → rLLM/verl → 不需要CT → single-turn不需要append-only
★★★★★★★ 中期: agent RL → multi-turn → tool calls → CT有价值 → 但需要environment + tool
★★★★★★★ 长期: CT + MAGI → prefix dedup → memory savings → RTX 4090 agent RL可能可行

★★★★★★★ 实际影响排序:
  → 短期 (GRPO) → 无影响 → 不需要CT → single-turn sufficient
  → 中期 (agent RL) → 小影响 → CT enables multi-turn → 但model太小 → 需要LoRA
  → 长期 (CT + MAGI + LoRA) → 大影响 → memory savings → multi-turn RL → RTX 4090可行
```

## 参考
- verl PR #6720: Continuous Token Builder Core (OPEN, +737 lines)
- verl PR #6721: Agent Loop Integration (OPEN, +1270/-39 lines)
- verl AgentLoop: multi-turn execution paths
- verl multi_turn.continuous_token.enable: config flag
- Related notes: verl-v080-latest-developments-2026-06-reading.md, verl-magi-prefix-tree-reading.md

### ★★★★★★★★ response_mask alignment with GRPO/PPO/CPPO (source-level):

ContinuousToken's alignment mechanism preserves `response_mask` correctly after token merges:
  → Inserted boundary tokens (e.g., `<|im_end|>` newlines) → mask=0 → NOT trainable
  → Assistant tokens → mask=1 → trainable → action tokens
  → Removed prefix tokens → reflected in mask → correct loss computation

  → ★★★★★★★★ This means ZERO changes needed to GRPO/PPO/CPPO training algorithms!
  → → GRPO: compute_grpo_outcome_advantage uses response_mask → mask=0 tokens get zero advantage → no loss contribution
  → → PPO/GAE: compute_gae_advantage_return uses response_mask → observation tokens (mask=0) transparent to advantage computation
  → → Policy loss: response_mask → bool → loss_mask → only action tokens participate
  → → KL penalty: kld * response_mask → KL only on action tokens

★★★★★ Key insight: mask=[1,0,1,0,...] = standard multi-step RL design. Policy learns "what to do" (action tokens, mask=1), NOT "what to see" (observation tokens, mask=0). ContinuousToken preserves this design while fixing token inconsistency.
