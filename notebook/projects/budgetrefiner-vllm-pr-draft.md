# BudgetRefiner SLO → vLLM Upstream PR Draft

> ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
> RTX 4090 UNIQUE CONTRIBUTION — NO OTHER CONTRIBUTOR HAS THIS PROFILE DATA!
> ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
> Target: vllm-project/vllm → vllm/v1/core/scheduler.py
> Origin: vllm-Ascend BudgetRefiner (95%+ GPU-generic)
> Status: DRAFT — needs BudgetRefiner source-level analysis + RTX 4090 profiling

## PR Title

`[Scheduler] Add SLO-aware dynamic token budget refinement (BudgetRefiner)`

## Problem

vLLM V1 scheduler has no SLO-aware scheduling mechanism. It uses fixed max_num_seqs and max_num_tokens limits that don't adapt to actual SLO requirements. This leads to:
1. Over-provisioning → wasted GPU resources when SLO allows fewer tokens
2. Under-provisioning → SLO violations when load exceeds static limits
3. No decode-first priority → prefill tokens compete with decode tokens → high decode latency
4. No dynamic budget adjustment → scheduler cannot respond to changing load patterns

MindIE/vLLM-Ascend has solved this with BudgetRefiner SLO, which:
- Dynamically adjusts token budgets per scheduling iteration based on SLO targets
- Prioritizes decode tokens over prefill tokens (decode-first reordering)
- Uses profile_table.csv + quadratic predictor to estimate compute time
- Achieves consistent SLO compliance with minimal GPU resource waste

## Solution

Port BudgetRefiner from vLLM-Ascend to standard vLLM V1. The core logic is 95%+ GPU-generic — only the profile_table.csv needs GPU-specific profiling data.

### Key Components to Port

1. **BudgetRefiner class** (105 lines) — Dynamic token budget calculator
   - `refine_budget()` — Main entry: compute optimal budget given SLO constraints
   - `_read_lookup_table()` — Load profile_table.csv with precomputed timing data
   - `_align_key()` — Match runtime config to profile table entries
   - `_get_max_budget()` — Calculate maximum tokens that fit within SLO time

2. **Decode-first reordering** (20 lines) — Prioritize decode requests over prefill
   - `d_lst + p_lst` pattern from vLLM-Ascend scheduler
   - Decode requests get priority → lower latency for ongoing conversations

3. **SLO config** (15 lines) — New SchedulerConfig fields
   - `slo_limits` — SLO target latency (e.g., 100ms per decode step)
   - `budget_refiner_enabled` — Toggle BudgetRefiner on/off

4. **profile_table.csv** — GPU-specific profiling data
   - Columns: model, quantization, num_layers, hidden_dim, seq_len, batch_size, chunk_time
   - Pre-populated with RTX 4090 data (our unique contribution!)
   - Community can add H100/A100/other GPU profiles

### Integration Points

```python
# vllm/v1/core/scheduler.py — Integration sketch

class Scheduler:
    def __init__(self, ...):
        if scheduler_config.budget_refiner_enabled:
            self.budget_refiner = BudgetRefiner(
                slo_limits=scheduler_config.slo_limits,
                profile_table=scheduler_config.profile_table_path,
            )

    def _schedule(self, ...):
        # Step 1: Get candidate requests
        running = self._get_running_requests()
        waiting = self._get_waiting_requests()

        # Step 2: Budget refinement (NEW!)
        if self.budget_refiner:
            max_budget = self.budget_refiner.refine_budget(
                num_running_seqs=len(running),
                num_running_tokens=sum(r.num_computed_tokens for r in running),
            )
        else:
            max_budget = self.max_num_tokens  # Default static limit

        # Step 3: Decode-first reordering (NEW!)
        d_lst, p_lst = self._reorder_decode_first(running, waiting)

        # Step 4: Schedule within budget
        scheduler_output = self._schedule_within_budget(d_lst, p_lst, max_budget)
        return scheduler_output
```

### RTX 4090 Profile Data (Unique Contribution)

This is our unique value — no other vLLM contributor has RTX 4090 profiling data. We will collect:

| Model | Quant | Layers | Hidden | Seq | Batch | ChunkTime(ms) |
|-------|-------|--------|--------|-----|-------|---------------|
| Qwen3-1.7B | BF16 | 28 | 2048 | 4K | 1-64 | TBD (GPU needed) |
| Qwen3-1.7B | INT4 | 28 | 2048 | 4K | 1-64 | TBD |
| Qwen3-8B | BF16 | 36 | 4096 | 4K | 1-32 | TBD |
| Qwen3-8B | INT4 | 36 | 4096 | 4K | 1-32 | TBD |
| Llama-3.1-8B | BF16 | 32 | 4096 | 4K | 1-32 | TBD |
| Llama-3.1-8B | INT4 | 32 | 4096 | 4K | 1-32 | TBD |

Collection script:
```bash
# When GPU is available:
for model in Qwen3-1.7B Qwen3-8B Llama-3.1-8B; do
  for batch in 1 2 4 8 16 32 64; do
    python3 tools/profile_vllm_budget.py \
      --model $model --batch $batch --seq-len 4096
  done
done
```

## Precedent Analysis

BudgetRefiner is conceptually similar to:
1. **vLLM watermark** (#44594) — Dynamic memory limits → BudgetRefiner extends to SLO-aware
2. **vLLM admission control** — Static limits → BudgetRefiner makes dynamic
3. **SLO-aware serving** (TensorRT-LLM, SGLang Model Gateway) — Production requirement

But BudgetRefiner is UNIQUE in:
- Being fully open-source (MindIE/vLLM-Ascend)
- Having a portable design (95%+ GPU-generic)
- Using profile_table + quadratic predictor (more robust than heuristics)

## Phase Plan

### Phase 0: Pre-Work (comments on existing issues)
- Comment on #44594 watermark → suggest SLO-aware extension
- Comment on #39096 batch invariance → establish SM89 expertise
- Comment on #44879/#45038 FP8 crash → demonstrate SM89 knowledge

### Phase 1: RTX 4090 Profile Data Collection (needs GPU)
- Write `tools/profile_vllm_budget.py` profiling script
- Collect profile_table.csv for Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B on RTX 4090
- Validate BudgetRefiner logic with collected data

### Phase 2: RFC → vLLM community discussion
- Open RFC issue on vllm-project/vllm
- Present BudgetRefiner concept, scope, porting plan
- Get community feedback on approach and integration points

### Phase 3: PR submission
- Fork vllm-project/vllm
- Implement BudgetRefiner + decode-first + SLO config
- Include RTX 4090 profile_table.csv
- Run vLLM CI tests
- Submit PR with full documentation

## Checklist Before Submission

- [ ] Collect RTX 4090 profile_table.csv (GPU needed)
- [ ] BudgetRefiner source-level analysis complete (background agent running)
- [ ] Write profile collection script (tools/profile_vllm_budget.py)
- [ ] Draft RFC issue → get community feedback
- [ ] Validate BudgetRefiner logic on RTX 4090
- [ ] Implement BudgetRefiner class in vLLM V1 scheduler
- [ ] Add SLO config to SchedulerConfig
- [ ] Add decode-first reordering
- [ ] Run vLLM CI tests
- [ ] Submit PR

## 参考
- Contribution plan: notebook/projects/budgetrefiner-vllm-contribution-plan.md
- MindIE BudgetRefiner: vllm-Ascend BudgetRefiner class (source-level agent analyzing)
- vLLM watermark #44594: preemptions -82%, ITL p99 -56%
- vLLM V1 scheduler: notebook/projects/vllm-v1-scheduler-deep-reading.md
