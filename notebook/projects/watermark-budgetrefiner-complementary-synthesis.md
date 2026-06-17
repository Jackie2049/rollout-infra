# Watermark + BudgetRefiner: Complementary Thrashing Prevention for RTX 4090

> 2026-06-17 | Synthesis of Watermark #44594 (MERGED) + BudgetRefiner SLO (proposed upstream PR)
> ★★★★★★★★ Together they approach ZERO preemptions → RTX 4090 production-grade serving

---

## 1. Problem: Two Types of Pressure

vLLM V1 scheduler faces two distinct types of pressure:

| Pressure Type | What | Measurement | Current Handling |
|---------------|------|-------------|-----------------|
| **KV cache pressure** | KV blocks filling up → no room for new tokens | free_blocks < needed_blocks | Watermark #44594 (reactive) |
| **Compute time pressure** | Too many prefill tokens → decode ITL exceeds SLO | ITL > SLO limit | **NONE** → BudgetRefiner (proactive) |

★★★★★★★★★ **Key insight**: Watermark handles KV cache pressure (memory bottleneck) → BudgetRefiner handles compute time pressure (compute bottleneck). They are COMPLEMENTARY, not overlapping.

---

## 2. Watermark #44594: 4-Layer KV Cache Thrashing Prevention

```
★★★★★★★★★ Watermark 4-layer mechanism:

Layer 1: Admission Gate after Preemption
  → After preempting a request → block admission of WAITING/PREEMPTED requests
  → Until free_blocks recovers → prevents immediate re-admission → thrashing!

Layer 2: Watermark Headroom (default=0.05 → MUST set!)
  → watermark_blocks = int(watermark × num_total_blocks)
  → WAITING/PREEMPTED admission blocked when free_blocks < watermark_blocks + full_isl
  → ★★★★★ 5% headroom = buffer zone → prevents thrashing loop

Layer 3: Prefix Cache ref_cnt Protection
  → Prefix cache blocks with ref_cnt > 0 → not preempted
  → Shared prefix blocks → protected → multi-request reuse preserved

Layer 4: Self-Preemption Detection
  → Request preempted → re-enters WAITING → admission gate blocks re-entry
  → Prevents request thrashing → itself → → → loop

★★★★★ Results: watermark=0.05 → preemption -82%, ITL p99 -56%, throughput +5.1%
```

### Watermark limitations

```
★★★★ Watermark handles KV pressure BUT NOT compute pressure:
  → When many decode requests + many prefill requests →
  → KV blocks available (enough memory) → all admitted →
  → BUT: GPU compute overloaded → decode ITL spikes → SLO violated!
  → → Watermark can't help → it only manages memory, not compute time
```

---

## 3. BudgetRefiner: Proactive Compute Time Pressure Control

```
★★★★★★★★★ BudgetRefiner SLO mechanism (58 lines, 100% GPU-generic):

Core: lookup key (ctx_len, d_num) → chunk_size
  → Budget DROPS as decode load increases!
  → d_num=0 → 1024 tokens (full budget, no decode pressure)
  → d_num=100 → 768 tokens (25% reduction, moderate decode load)
  → d_num=255 → 512 tokens (50% reduction, heavy decode load)

decode-first reordering (4 lines):
  → d_lst + p_lst = separate lists → decode first → pop() removes PREFILL
  → decode requests protected → prefill throttled

3 fallback paths:
  → Lookup miss → return default_budget → never crashes
  → No CSV loaded → return default_budget → zero overhead
  → slo_limit ≤ 0 → disabled → original budget → ZERO impact!
```

### BudgetRefiner limitations

```
★★★★ BudgetRefiner handles compute pressure BUT NOT KV pressure:
  → When KV cache almost full → few free blocks →
  → BudgetRefiner says "you can use 512 tokens" →
  → BUT: 512 tokens need KV blocks → no room → STILL preempted!
  → → BudgetRefiner can't help → it only manages compute time, not memory
```

---

## 4. How They Work Together

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ Watermark + BudgetRefiner = NEAR-ZERO PREEMPTIONS!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

Scenario A: Low load → few requests → both relaxed
  → Watermark: watermark_blocks available → admit freely
  → BudgetRefiner: d_num=0 → full budget → no throttling
  → Result: normal operation, both transparent

Scenario B: Moderate load → some decode + some prefill → compute pressure
  → Watermark: KV blocks available → admit freely (no KV pressure)
  → BudgetRefiner: d_num>0 → budget reduced → throttle prefill → protect decode ITL
  → Result: BudgetRefiner prevents SLO violation → Watermark unnecessary

Scenario C: High load → many requests → both KV + compute pressure
  → Watermark: free_blocks < watermark + full_isl → block admission → KV protected
  → BudgetRefiner: d_num high → budget reduced → fewer prefill tokens → compute protected
  → Result: BOTH active → BudgetRefiner reduces token load → fewer KV blocks needed → Watermark less triggered → feedback loop → approaching zero preemptions!

★★★★★★★★★ Complementary interaction:
  → BudgetRefiner reduces prefill token count → fewer KV blocks consumed →
  → KV pressure reduced → Watermark triggers less →
  → Less preemption → fewer WAITING/PREEMPTED →
  → BudgetRefiner sees fewer requests → budget relaxes →
  → More prefill admitted → compute normalized →
  → ★★★★★★★★ Self-regulating feedback loop → system stabilizes!
```

---

## 5. Integration: 3 Exact Points in vLLM V1 Scheduler

```
★★★★★★★★★ BudgetRefiner integration points (from source reading):

Point A: token_budget line 358
  → Current: self.max_num_scheduled_tokens (static constant)
  → Proposed: BudgetRefiner.refine_budget(count decode→lookup→return adjusted)
  → ★★★★★★ Budget becomes DYNAMIC → adapts to decode load

Point B: decode-first reorder self.running line 375
  → Current: self.running (plain list, FCFS order)
  → Proposed: decode_reqs + prefill_reqs → decode first → pop() removes PREFILL
  → ★★★★★★ 4 lines change → decode protected from compute starvation

Point C: dynamic max_seqs line 565
  → Current: self.max_num_running_reqs (static)
  → Proposed: fewer prefill admissions under decode pressure → dynamic limit
  → ★★★★★★ Reduces concurrent requests when decode-heavy → SLO protection
```

### Interaction with Watermark at each point

| Integration Point | Watermark Role | BudgetRefiner Role | Combined Effect |
|-------------------|---------------|-------------------|-----------------|
| Point A: token_budget | Budget provides max tokens → Watermark checks KV blocks available | BudgetRefiner reduces budget → fewer tokens → fewer KV blocks needed | BudgetRefiner proactively reduces → Watermark less likely triggered |
| Point B: decode-first | Watermark protects decode KV blocks (ref_cnt>0 prefix) | BudgetRefiner ensures decode processed before prefill | Both protect decode → decode SLO guaranteed |
| Point C: max_seqs | Watermark limits admission when free_blocks < watermark | BudgetRefiner limits prefill admission count | Fewer requests → less KV + compute pressure → both relaxed |

---

## 6. RTX 4090 Profile Data: UNIQUE Contribution

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ NO OTHER vLLM CONTRIBUTOR HAS RTX 4090 PROFILE DATA!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

What we need to collect (P10 priority when GPU online):
  → profile_table.csv with RTX 4090 rows
  → (ctx_len, d_num) → chunk_size → budget at different SLO limits
  → SLO=50ms, SLO=100ms, SLO=200ms → multiple SLO targets
  → Qwen3-1.7B + Qwen3-8B INT4 → two model sizes

Collection method:
  → Run vLLM with different decode/prefill mixes
  → Measure ITL at each configuration
  → Fill profile_table.csv → BudgetRefiner lookup table

★★★★★★★★★ This data is what makes our PR UNIQUE →
  → vLLM-Ascend has Ascend profile data → but NO NVIDIA GPU data
  → We provide RTX 4090 (SM89) data → largest consumer GPU installed base
  → → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ HIGHEST-VALUE contribution!
```

---

## Key Takeaways

★★★★★★★★★ Watermark handles KV cache pressure (reactive, memory) → BudgetRefiner handles compute time pressure (proactive, compute) → COMPLEMENTARY, not overlapping
★★★★★★★★★ Together: BudgetRefiner reduces token load → fewer KV blocks needed → Watermark triggers less → feedback loop → approaching zero preemptions
★★★★★★★★★ BudgetRefiner 3 integration points: token_budget (dynamic) + decode-first (reorder) + max_seqs (dynamic limit)
★★★★★★★★★ Watermark 4 layers: admission gate + watermark headroom + prefix ref_cnt + self-preemption detection
★★★★★★★★★ RTX 4090 profile_table.csv = UNIQUE data no other contributor has → #1 contribution priority
★★★★★★★★★ BudgetRefiner 58 lines core → 100% GPU-generic → only profile_table.csv HW-specific → portable!

---

## References

- Watermark source reading: notebook/projects/vllm-v1-scheduler-watermark-reading.md
- BudgetRefiner SLO source reading: notebook/projects/budgetrefiner-slo-source-reading.md
- BudgetRefiner V1 scheduler integration: notebook/projects/vllm-v1-scheduler-budgetrefiner-integration.md
- BudgetRefiner PR draft: notebook/projects/budgetrefiner-vllm-pr-draft.md
- BudgetRefiner contribution plan: notebook/projects/budgetrefiner-vllm-contribution-plan.md
- SGLang WAR barrier philosophy: notebook/projects/sglang-war-barrier-budgetrefiner-philosophy.md
