# verl vs TRL Advantage Estimator + Policy Loss Comparison (Updated 2026-07-14)

## verl Advantage Estimators (14 registered)

| # | Name | Enum | Registration | Key Feature |
|---|------|------|-------------|-------------|
| 1 | GAE | `AdvantageEstimator.GAE` | Built-in | Lambda-return, requires value function |
| 2 | GRPO | `AdvantageEstimator.GRPO` | Built-in | Group relative, mean=0 std=1 normalization |
| 3 | REINFORCE++ | `reinforce_plus_plus` | Built-in | Token-level baseline subtraction |
| 4 | REINFORCE++Baseline | `reinforce_plus_plus_baseline` | Built-in | With learned baseline |
| 5 | ReMax | `remax` | Built-in | Reward maximization |
| 6 | RLOO | `rloo` | Built-in | Leave-one-out baseline |
| 7 | OPO | `opo` | Built-in | Offline policy optimization |
| 8 | GRPO pass@k | `grpo_passk` | Built-in | Pass@k advantage |
| 9 | GPG | `gpg` | Built-in | Group policy gradient |
| 10 | RLOO Vectorized | `rloo_vectorized` | Built-in | Vectorized leave-one-out |
| 11 | GRPO Vectorized | `grpo_vectorized` | Built-in | Vectorized GRPO |
| 12 | Optimal Token Baseline | `optimal_token_baseline` | Built-in | Per-token optimal baseline |
| 13 | TIR Optimal Token Baseline | `tir_optimal_token_baseline` | Built-in | TIR variant |
| 14 | GDPO | `gdpo` | Built-in | Group DPO |

**Registration mechanism**: `@register_adv_est` decorator + `ADV_ESTIMATOR_REGISTRY` dict + `get_adv_estimator_fn()` lookup. Extensible via string name (no need to add to Enum).

**NaN protection**: NONE across all 14 estimators — our PR #6 fills this gap.

## verl Policy Loss Modes (10+)

| # | Name | Key Feature |
|---|------|-------------|
| 1 | vanilla (PPO-clip) | Standard clip(r, 1-ε, 1+ε) * A |
| 2 | dppo_tv | DPPO total variation |
| 3 | dppo_kl | DPPO KL divergence |
| 4 | gspo | Group sequence policy optimization |
| 5 | sapo | Soft adaptive policy optimization |
| 6 | gpg | Group policy gradient |
| 7 | clip_cov | Clip coverage ratio |
| 8 | kl_cov | KL coverage |
| 9 | geo_mean | Geometric mean |
| 10 | bypass_mode | Self-anchored ratio (skip old_log_prob forward) |

**bypass_mode**: `r = exp(logπ - logπ.detach())` — structurally half of UP-GRPO. Does NOT remove positive clip.

## TRL Loss Types (9)

| # | Name | Key Feature | Normalization |
|---|------|-------------|---------------|
| 1 | grpo | Standard PPO-clip | Sequence-length (biased) |
| 2 | dapo | Standard PPO-clip + global token norm | Global token count |
| 3 | bnpo | Standard PPO-clip | Local batch |
| 4 | dr_grpo | Standard PPO-clip + const norm | Global constant |
| 5 | **up** | **UP-GRPO asymmetric** (NEW, our PR #6) | **Global token count** |
| 6 | cispo | Clip importance weights | Global token count |
| 7 | sapo | Soft temperature gate | Sequence-length |
| 8 | luspo | Length-unbiased sequence | Sequence-level |
| 9 | vespo | Variational gamma weights | Global token count |

**Key difference**: TRL combines advantage computation + loss formulation into `loss_type`, while verl separates them into `adv_estimator` + `policy_loss_mode`. This gives verl more flexibility (14×10 = 140 combos vs 9).

## Cross-Framework Gap Analysis

| Feature | verl | TRL | Gap |
|---------|------|-----|-----|
| NaN guard | NONE (PR #6 fixes) | `nan_to_num` builtin | verl MISSING → our PR |
| UP-GRPO loss | bypass_mode (partial) | loss_type="up" (full, our PR) | verl missing remove-positive-clip |
| Self-anchored ratio | bypass_mode | implicit in "up" | verl has it but clips both sides |
| Advantage/Loss separation | Separate | Combined | Design difference |
| Singleton degeneration | mean=0 std=1 | mean=0 std=1 | Same bug, cross-framework |
| Token-level importance | importance_sampling_level | Not configurable | TRL missing this option |

## Potential Future Contributions

1. **UP-GRPO for verl**: Add `policy_loss_mode="up"` that combines bypass_mode's self-anchor with remove-positive-clip for A>0. ~15 LOC, leverages existing bypass_mode infrastructure.
2. **Cross-framework NaN guard standardization**: TRL has nan_to_num, verl now has it (our PR #6), rLLM and SGLang don't. Push for consistent NaN handling across RL training frameworks.
3. **verl advantage estimator + UP-GRPO combo**: `adv_estimator=GRPO + policy_loss_mode=up` would be the strongest training config for RTX 4090.
