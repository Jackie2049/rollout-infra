# TRL Contribution Strategy Plan

## Strategic Context

**Decision rationale**: vLLM/SGLang PR response too slow → pivot to faster-review communities
**Target**: HuggingFace TRL — merges daily (6353+ PRs), GRPO focus, agent/tool support
**User directive**: ALL PRs on fork repos only (`jackie2049/trl`), user reviews then manually submits

## TRL Strengths for Contribution

1. **Fast PR review**: daily merges, responsive maintainers
2. **GRPO is core**: directly aligned with AI self-training
3. **Tool/environment system**: enables agent self-evolution
4. **Research-friendly**: 8 loss types, multi-reward, async tools
5. **Growing community**: 6353+ PRs, active issue tracker

## Contribution Tiers

### Tier 1 — Unique, High-Impact (P9)

#### P9-1: Explicit bypass_mode flag (~20 LOC)
**File**: `trl/trainer/grpo_trainer.py`
**Target method**: `_generate_and_score_completions` + `GRPOConfig`

**Current behavior**: TRL infers bypass opportunity by checking if `gradient_accumulation_steps % generate_every != 0`. This heuristic works for standard cases but can fail with non-standard schedules (e.g., uneven gradient accumulation, dynamic batch sizing).

**Improvement**: Add explicit `bypass_mode` config flag + `skip_old_logps` attribute:
```python
# In _generate_and_score_completions, lines 2417-2432:
if not self.bypass_mode and (
    self.args.gradient_accumulation_steps % generate_every != 0 or (...)
):
    old_per_token_logps = compute_old_logps(...)
else:
    old_per_token_logps = None  # will be .detach()ed in _compute_loss
```

**Rationale**: Explicit bypass mode is cleaner and allows external schedulers to control the optimization. Matches verl's `bypass_mode=True` behavior. Especially important for RTX 4090 where every saved forward pass = significant memory reduction.

**Connection to user's work**: Direct parallel to verl bypass expertise. Can reference as "widely adopted in verl community."

#### P9-2: Per-unit LoRA summon for FSDP (~250 LOC)
**File**: `trl/trainer/grpo_trainer.py` + new module `trl/trainer/fsdp_unit.py`

**Current behavior**: TRL uses standard HF Trainer with FSDP via Accelerate. When computing ref logprobs with PEFT, the full model (including all LoRA adapters) is loaded in memory.

**Improvement**: Port per-unit FSDP parameter discovery from verl #6512:
```python
# During FSDP init, dynamically discover units from LoRA target_modules
# Instead of 8 hard-coded prefixes, inspect PEFT config
units = discover_fsdp_units(model, lora_target_modules)
```

**Memory impact**: ~10x reduction (60→6 GiB) for typical GRPO training with LoRA.

**Key insight**: TRL already has `is_peft_model()` checks — extend to per-unit FSDP.

**Dependencies**: Requires understanding of `accelerate.FSDPPlugin` integration. May need coordination with HF Accelerate team.

### Tier 2 — Important, Moderate Impact (P8)

#### P8-1: GRPO group_size=1 detection warning (~10 LOC)
**File**: `trl/trainer/grpo_trainer.py` in `__init__`

**Current behavior**: When `num_generations=1`, GRPO degenerates to REINFORCE (advantage = 0). This is a well-known degenerate case across all frameworks (verl, rLLM, TRL).

**Improvement**: Add detection + warning + auto-config:
```python
if self.num_generations == 1:
    logger.warning(
        "num_generations=1 degenerates GRPO to REINFORCE (advantage=0). "
        "Setting scale_rewards='none' to avoid division by zero."
    )
    if self.scale_rewards == "group":
        self.scale_rewards = "none"
```

**Impact**: Prevents silent training degradation. 0 reviewers would oppose this — pure safety improvement.

#### P8-2: Off-policy mask for Liger kernel (~80 LOC)
**File**: `trl/trainer/grpo_trainer.py` + Liger kernel adaptation

**Current behavior**: Liger fused GRPO loss does not support `off_policy_mask_threshold`. When `use_liger_kernel=True` and `off_policy_mask_threshold` is set, trainer raises ValueError.

**Improvement**: Implement off-policy masking within the Liger fused op:
```python
# In liger_loss computation, apply mask to per_token_loss
per_token_loss = liger_fused_loss(...)
if off_policy_mask is not None:
    per_token_loss = per_token_loss * off_policy_mask
```

**Impact**: Unlocks off-policy correction for Liger users. Moderate complexity.

### Tier 3 — Specialized, Niche Impact (P7)

#### P7-1: Delta weight sync for vLLM (~150 LOC)
**File**: `trl/trainer/vllm_engine.py` (new module or extend existing)

**Current behavior**: Full weight sync on every generation step when using vLLM colocate mode. For LoRA training, only adapter weights changed → full sync wastes bandwidth.

**Improvement**: Compute delta since last sync, transfer only changed weights:
```python
def sync_weights(self):
    delta = {}
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            change = param.data - self._last_synced[name]
            if change.abs().sum() > 0:
                delta[name] = change
    self.transfer_to_vllm(delta)
    self._last_synced = {n: p.data.clone() for n, p in ...}
```

**Impact**: ~100x payload reduction for LoRA, faster generation cycles.

**Connection to user's work**: Direct port from verl #6794 expertise.

#### P7-2: Top-n-sigma reward clipping (~30 LOC)
**File**: `trl/trainer/grpo_trainer.py` in advantage computation

**Current behavior**: No reward outlier clipping. Extreme reward values can destabilize training.

**Improvement**: Add configurable top-n-sigma clipping:
```python
# After advantage computation
if self.top_n_sigma > 0:
    mean_adv = advantages.mean()
    std_adv = advantages.std()
    upper = mean_adv + self.top_n_sigma * std_adv
    lower = mean_adv - self.top_n_sigma * std_adv
    advantages = advantages.clamp(lower, upper)
```

**Impact**: Stabilizes training with noisy reward functions. Common in vLLM PPO already.

## Implementation Roadmap

### Phase 1 (Immediate, 1-2 days each)
1. **P9-1**: Explicit bypass_mode flag (~20 LOC)
2. **P8-1**: Group_size=1 detection (~10 LOC)

### Phase 2 (Short-term, 3-5 days each)
3. **P8-2**: Off-policy mask for Liger (~80 LOC)
4. **P7-2**: Top-n-sigma reward clipping (~30 LOC)

### Phase 3 (Medium-term, 1-2 weeks each)
5. **P7-1**: Delta weight sync for vLLM (~150 LOC)
6. **P9-2**: Per-unit LoRA summon (~250 LOC)

## Fork Workflow

```
jackie2049/trl
  ├── fix/bypass-mode          (P9-1, ~20 LOC)
  ├── fix/grpo-group-size      (P8-1, ~10 LOC)
  ├── feat/liger-off-policy     (P8-2, ~80 LOC)
  ├── feat/top-n-sigma          (P7-2, ~30 LOC)
  ├── feat/delta-sync-vllm      (P7-1, ~150 LOC)
  └── feat/per-unit-lora-fsdp   (P9-2, ~250 LOC)
```

Each branch: single-purpose, rebased on upstream/main, user reviews then manually submits to huggingface/trl.

## Risk Assessment

| PR | Risk | Mitigation |
|----|------|------------|
| P9-1 bypass_mode | Low | Additive, non-breaking |
| P8-1 group_size | Very low | Warning only |
| P8-2 Liger mask | Medium | Needs Liger compatibility |
| P7-2 top-n-sigma | Low | Optional, default=0 |
| P7-1 delta sync | Medium | vLLM version dependent |
| P9-2 per-unit FSDP | High | FSDP integration complexity |

## Cross-Framework Synergies

The contributions to TRL build on user's existing expertise:
- **verl bypass_mode** → P9-1 explicit bypass for TRL
- **verl delta sync (#6794)** → P7-1 delta weight sync
- **verl per-unit LoRA (#6512)** → P9-2 per-unit FSDP
- **vLLM top-n-sigma (PR #7 fork)** → P7-2 reward clipping
- **cross-framework GRPO degeneration analysis** → P8-1 group_size detection

## Agent Self-Evolution Connection

TRL's tool/environment system (lines 556-672) directly enables agent self-evolution research:
- `environment_factory`: per-rollout environment instances with `reset`/`get_reward`
- Multi-turn tool calling with response parsing
- Async coroutine support for LLM judge rewards
- `log_extra`/`log_metric` for reward function observability

**Contribution angle**: Improve the environment system for self-evolving agents — e.g., add support for environment-defined curriculum learning, adaptive difficulty scaling based on agent performance, or multi-agent competitive environments.

This directly aligns with user's stated interest in "agent自演进" (agent self-evolution).
