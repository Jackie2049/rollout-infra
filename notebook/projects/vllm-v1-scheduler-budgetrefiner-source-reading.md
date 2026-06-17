# vLLM V1 Scheduler Source Reading — BudgetRefiner Integration Points

> 2026-06-18 | Source-level analysis of vLLM V1 scheduler for BudgetRefiner P10 contribution
> ★★★★★★★★ 3 integration points verified at current checkout line numbers
> ★★★★★★★★ Full scheduler flow documented for contribution quality

---

## Scheduler Architecture Overview

```
★★★★★★★★★ vLLM V1 Scheduler (vllm/v1/core/sched/scheduler.py):

  Core scheduling method: _schedule()
  → Called every step → decides which requests to process
  → Token budget system → limits total tokens per step
  → FCFS (First Come First Serve) → priority-based preemption
  → RUNNING → decode requests → high priority
  → WAITING → prefill requests → lower priority
  → SKIPPED_WAITING → requests that couldn't be scheduled → retry next step

  Key data structures:
    → self.running: list of currently executing requests
    → self.waiting: queue of pending prefill requests
    → self.skipped_waiting: requests skipped due to budget/LoRA constraints
    → self.max_num_scheduled_tokens: total token budget per step
    → self.max_num_running_reqs: max concurrent request count
    → token_budget: remaining tokens for current step (starts from max)
```

---

## 1. ★★★★★★★★ Integration Point 1: token_budget (Line 407)

```python
# Line 407 — CURRENT (verified in latest checkout):
token_budget = self.max_num_scheduled_tokens
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
```

```
★★★★★★★★★ BudgetRefiner integration at line 407:

  CURRENT behavior:
    → token_budget = max_num_scheduled_tokens (static config value)
    → Fixed per step → no adaptation to current decode load

  BudgetRefiner change:
    → token_budget = BudgetRefiner.adjust_prefill_budget(
        max_budget=self.max_num_scheduled_tokens,
        num_decode_reqs=len(decode_reqs),
        num_prefill_reqs=len(prefill_reqs),
        ctx_len=current_context_length,
        chunk_size=current_chunk_size,
      )
    → Dynamic budget → adapts to decode pressure
    → When decode requests active → reduce prefill budget → protect decode latency
    → When no decode requests → full budget for prefill → no throttling

★★★★★★★★★ KEY DESIGN DECISION:
    → BudgetRefiner ONLY throttles prefill when ACTIVE decode requests exist
    → Pure-prefill phases → zero impact → full throughput
    → This is why BudgetRefiner is complementary to Watermark → different pressure type
```

---

## 2. ★★★★★★★★ Integration Point 2: decode-first reorder (Before Line 430)

```python
# Lines 425-428 — Current:
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)

# Line 430 — Current:
req_index = 0
while req_index < len(self.running) and token_budget > 0:
```

```
★★★★★★★★★ BudgetRefiner decode-first reorder (4 lines):

  CURRENT behavior:
    → self.running = mixed order (prefill chunks + decode requests)
    → FCFS ordering → prefill may consume budget before decode

  BudgetRefiner change (before line 430 RUNNING loop):
    → Split self.running into decode and prefill lists:
        decode_reqs = [r for r in self.running if not r.is_prefill_chunk]
        prefill_reqs = [r for r in self.running if r.is_prefill_chunk]
    → Reorder: decode first, then prefill:
        self.running = decode_reqs + prefill_reqs
    → 4 lines total → minimal change → preserves FCFS within each category

★★★★★★★★★ WHY decode-first is important:
    → Decode requests have latency SLO → must complete within time budget
    → Prefill requests are throughput-oriented → can be deferred
    → BudgetRefiner ensures decode requests always get budget first
    → Then remaining budget goes to prefill chunks
    → This is the same principle as MindIE's BudgetRefiner SLO
```

---

## 3. ★★★★★★★★ Integration Point 3: dynamic max_seqs (Line 629)

```python
# Line 629 — Current:
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break
```

```
★★★★★★★★★ BudgetRefiner dynamic max_seqs at line 629:

  CURRENT behavior:
    → max_num_running_reqs = static config value
    → Hard limit → new requests cannot start if limit reached
    → Even if budget has room → blocked by count limit

  BudgetRefiner change:
    → dynamic_max = BudgetRefiner.adjust_max_running(
        max_running=self.max_num_running_reqs,
        num_decode=len(decode_reqs),
        num_prefill=len(prefill_reqs),
        available_budget=token_budget,
      )
    → When many decodes running → reduce max → protect decode throughput
    → When few decodes → increase max → allow more prefills → higher throughput

★★★★★★★★★ This is the least critical integration point:
    → Current behavior is reasonable → just needs dynamic adaptation
    → BudgetRefiner makes it adaptive based on decode/prefill ratio
```

---

## 4. Complete Scheduler Flow (11 Steps)

```
★★★★★★★★★ vLLM V1 Scheduler _schedule() flow:

  Step 1: Initialize budget (line 407)
    → token_budget = max_num_scheduled_tokens
    → BudgetRefiner: adjust_prefill_budget() here

  Step 2: KV cache manager new step (line 421)
    → self.kv_cache_manager.new_step_starts()

  Step 3: Determine prefill deferral (lines 425-428)
    → defer_prefills = throttle_prefills and not prefill_capacity_bound
    → And: any running decode requests

  Step 4: Decode-first reorder (before line 430)
    → BudgetRefiner: reorder self.running → decode first
    → 4 lines: decode_reqs + prefill_reqs

  Step 5: Schedule RUNNING decode requests (lines 430-598)
    → Loop: while req_index < len(self.running) and token_budget > 0
    → Each request: compute num_new_tokens → allocate KV blocks → schedule
    → Budget consumed: token_budget -= num_new_tokens
    → If no KV blocks → preempt lowest-priority request

  Step 6: Record scheduled LoRAs (lines 614-622)
    → Track which LoRAs are active → enforce max_loras constraint

  Step 7: Schedule WAITING prefill requests (lines 625-830+)
    → Loop: while waiting and token_budget > 0
    → BudgetRefiner: dynamic max_seqs check (line 629)
    → Compute num_new_tokens for each prefill
    → Allocate KV blocks → schedule → consume budget
    → If LoRA conflict → skip to step_skipped_waiting

  Step 8: Schedule RESUMED requests
    → Requests that were preempted and then resumed
    → Similar to prefill scheduling

  Step 9: Build output (SchedulerOutput)
    → scheduled_new_reqs + scheduled_resumed_reqs + scheduled_running_reqs
    → preempted_reqs → notification to workers
    → num_scheduled_tokens → per-request token counts

  Step 10: Update request states
    → Mark scheduled → update num_computed_tokens
    → Update waiting/skipped_waiting queues

  Step 11: Return SchedulerOutput
    → Sent to worker processes for execution
```

---

## 5. BudgetRefiner Profile Table Requirements

```
★★★★★★★★★ BudgetRefiner profile_table.csv columns (verified):

  Required columns only:
    → chunk_size: prefill chunk size (e.g., 512, 1024, 2048, 4096)
    → d_num: number of concurrent decode requests (e.g., 1, 2, 4, ... 17)
    → ctx_len: context length of decode requests (e.g., 512, 1024, 2048, 4096, 8192)
    → cost: estimated compute cost (ms) → from profiling

  RTX 4090 profile_table dimensions:
    → 5 ctx_len values: 512, 1024, 2048, 4096, 8192
    → 17 d_num values: 1, 2, 3, ..., 17 (up to max_num_running_reqs)
    → 4 chunk_size values: 512, 1024, 2048, 4096
    → Total: 5 × 17 × 4 = 340 rows
    → vs Ascend's 10,875 rows → RTX 4090 table is much smaller

★★★★★★★★★ GPU needed for profile_table collection:
    → Must run on actual RTX 4090 → cannot simulate
    → profile_vllm_budget.py tool ready → collect/validate/estimate modes
    → When GPU available → highest priority experiment
```

---

## 6. BudgetRefiner SLO Code Structure (58 Lines GPU-generic)

```
★★★★★★★★★ BudgetRefiner SLO code architecture:

  budget_refiner.py (NEW, ~105 lines):
    → class BudgetRefiner:
    → adjust_prefill_budget(): dynamic budget based on decode load
    → adjust_max_running(): dynamic max_seqs based on decode/prefill ratio
    → load_profile_table(): load RTX 4090 profile data
    → estimate_cost(): estimate compute cost for given parameters

  scheduler.py modifications (~20 lines):
    → Line 407: BudgetRefiner.adjust_prefill_budget() call
    → Before 430: decode-first reorder (4 lines)
    → Line 629: BudgetRefiner.adjust_max_running() call

  output.py modifications (~20 lines):
    → BudgetRefiner metrics in SchedulerOutput

  config additions (~15 lines):
    → budget_refiner_enabled config option
    → budget_refiner_profile_table path config

  arg_utils additions (~10 lines):
    → CLI arguments for BudgetRefiner configuration

  metrics additions (~15 lines):
    → BudgetRefiner SLO metrics tracking

  profile_tables/ directory:
    → profile_table.csv per GPU type
    → rtx4090.csv, a100.csv, h100.csv, etc.

★★★★★★★★★ Total: 7 files, ~300 LOC
    → 95%+ GPU-generic → Ascend's approach already validated on their GPU
    → RTX 4090 contribution: profile_table.csv data → unique → no other contributor has this
```

---

## Key Findings Summary

★★★★★★★★★ Line 407: token_budget = max_num_scheduled_tokens → BudgetRefiner adjusts dynamically
★★★★★★★★★ Before line 430: decode-first reorder → 4 lines → decode_reqs + prefill_reqs
★★★★★★★★★ Line 629: dynamic max_seqs → BudgetRefiner adjusts based on decode/prefill ratio
★★★★★★★★★ Full scheduler flow: 11 steps → RUNNING first → WAITING next → budget consumed
★★★★★★★★★ BudgetRefiner ONLY throttles prefill when ACTIVE decode → zero pure-prefill impact
★★★★★★★★★ profile_table.csv: 340 rows (5×17×4) → unique RTX 4090 data → GPU needed

---

## References

- BudgetRefiner SLO: notebook/fundamentals/watermark-budgetrefiner-complementary-synthesis.md
- BudgetRefiner source: notebook/projects/budgetrefiner-slo-source-reading.md
- BudgetRefiner PR draft: notebook/projects/budgetrefiner-vllm-pr-draft.md
- Profile tool: tools/profile_vllm_budget.py
- vLLM latest developments: notebook/projects/vllm-latest-developments-2026-06-reading.md
- OSS contribution: notebook/fundamentals/rtx4090-oss-contribution-opportunity-analysis.md
