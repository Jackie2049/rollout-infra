# vLLM #45964 — MLA DCP Query Replication Reading

> 2026-06-18 | PR #45964 OPEN | +299/-14 | 6 files | by sungsooha
> ★★★★★★★★ Opt-in: replicate MLA query projection → skip query all-gather during decode
> ★★★★★★★★ 2.88-4.51% TPOT reduction, 2.10-4.74% throughput improvement
> ★★★★★★★★ RTX 4090 NOT affected (DCP=1 → pure no-op) → but architectural lesson valuable
> ★★★★★★★★ Accuracy-equivalent (GSM8K within ±0.22 pp)

---

## 1. Problem: Query All-Gather on Decode Critical Path

```
★★★★★★★★★ With Decode Context Parallelism (DCP):

  → KV cache sharded across DCP group → each rank has KV shard
  → Standard MLA decode: all-gather query across DCP group EVERY decode step
  → Each rank needs full head set → attend its KV shard → LSE-reduce partials
  → ★★★★★★★★ Query all-gather = decode critical path → collective every step → latency!

★★★★★★★★★ Why query projection is small:
  → MLA: query projection = W_UK_T → absorbs KV into query → small relative to KV
  → DeepSeek-V2/V3/R1: query heads much smaller than KV cache
  → Replicating query projection = small memory cost → big latency savings
  → ★★★★★★★★ "Replicate what's small, shard what's large" → smart asymmetry!
```

---

## 2. The Fix: Query Replication (Opt-In)

```
★★★★★★★★★ VLLM_DCP_Q_REPLICATE=1 (default OFF → exact current behavior):

Design:
  → At load time: replicate MLA query projection within DCP group
  → Every rank materializes full group-local query head set locally
  → Decode: SKIP query all-gather → group-local W_UK_T_dcp_qrep BMM
  → ★★★★★★★★ Remove collective from decode hot path → latency savings!

★★★★★★★★★ Implementation:
  → Build W_UK_T_dcp_qrep at weight-load → replicated query projection
  → Guarded: only works with FlashInfer-MLA backend + DCP decode-LSE (#44044/#43729)
  → NOT supported: head-padding MLA (CutlassMLA), FP4/FP8 Triton MLA BMM
  → ★★★★★★★★ Backend-agnostic feature → only requires decode-LSE from backend

★★★★★★★★★ Memory cost:
  → Query projection replicated per rank → dcp * num_heads * hidden_dim * 2 bytes
  → vs KV cache: MUCH smaller → negligible memory overhead
  → ★★★★★★★★ "Small projection replicated → large KV sharded" → asymmetric optimization
```

---

## 3. Test Results

```
★★★★★★★★★ Setup: 8× B200, TP8/DCP8, fp8 KV, full CUDA graph, max_num_seqs=32

Decode latency / throughput:

| Model | Regime | qrep P50 TPOT | all-gather P50 TPOT | TPOT Δ | tput Δ |
|-------|--------|-------------:|-------------------:|-------:|-------:|
| DeepSeek-R1-NVFP4 | decode | 16.55ms | 17.04ms | -2.88% | +2.91% |
| DeepSeek-R1-NVFP4 | mixed | 17.57ms | 18.07ms | -2.78% | +2.10% |
| Kimi-K2.6-NVFP4 | decode | 15.32ms | 16.04ms | -4.51% | +4.74% |
| Kimi-K2.6-NVFP4 | mixed | 15.97ms | 16.60ms | -3.75% | +4.05% |

★★★★★★★★★ Accuracy equivalence:

| Model | qrep (flex/strict) | all-gather (flex/strict) | Δ flex (pp) |
|-------|--------------------|-------------------------|------------|
| DeepSeek-R1-NVFP4 | 0.9553/0.9538 | 0.9538/0.9530 | +0.15 |
| Kimi-K2.6-NVFP4 | 0.9371/0.9363 | 0.9393/0.9386 | -0.22 |

★★★★★★★★★ Within ±0.22 percentage points → accuracy-equivalent
```

---

## 4. GPU-generic Lessons

```
★★★★★★★★★ Lesson 1: Asymmetric replication vs sharding
  → "Replicate what's small, shard what's large" → universal principle
  → MLA query = small → replicate → skip collective
  → KV cache = large → shard → save memory
  → ★★★★★★★★ Same principle applies to LoRA adapters → small → replicate → skip all-gather

★★★★★★★★★ Lesson 2: Opt-in feature gating
  → VLLM_DCP_Q_REPLICATE=1 → default OFF → exact baseline
  → Pure no-op at DCP=1 → single GPU unaffected
  → ★★★★★★★★ Good practice: opt-in → backward compatible → no risk

★★★★★★★★★ Lesson 3: DCP becoming standard for long-context
  → vLLM-Ascend #10225: DyCP (Dynamic CP) → same direction
  → #45964: MLA DCP query replication → same CP evolution
  → #45966: Pre-reserve A2A workspace → DCP infrastructure
  → ★★★★★★★★ CP = Context Parallelism → emerging standard for long-context serving

★★★★★★★★★ RTX 4090 assessment:
  → DCP=1 → qrep is pure no-op → NO impact → NO benefit → NO risk
  → Single GPU → no DCP → no all-gather → query projection already local
  → ★★★★★★★★ Not relevant for RTX 4090 → but architectural knowledge useful for multi-GPU future
```

---

## 5. Related Developments

```
★★★★★★★★★ vLLM DCP cluster (June 17-18):
  → #45964: MLA DCP query replication (+299/-14)
  → #45966: Pre-reserve packed A2A workspace (+218/-17)
  → #45971: Parallelize KV load with receive-thread pool (Mooncake)
  → #45969: Compact chunk-hash keys (Mooncake)
  → #45972: Revert DSV4 cudagraph optimization

★★★★★★★★★ vLLM-Ascend #10225: DyCP → same CP direction
  → MLA CP support + Mooncake KV transfer
  → ★★★★★★★★ Cross-platform: DCP evolving on both CUDA and Ascend

★★★★★★★★★ Megatron #5389: GDN THD all-to-all → MERGED June 17
  → Restore fused GDN THD all-to-all on dev
  → ★★★★★★★★ Same class: all-to-all optimization for MoE/MLA attention
```

---

## Key Findings Summary

★★★★★★★★★ #45964: MLA DCP query replication → skip all-gather → 2-5% TPOT improvement
★★★★★★★★★ Opt-in VLLM_DCP_Q_REPLICATE=1 → default OFF → DCP=1 pure no-op
★★★★★★★★★ "Replicate small, shard large" → universal asymmetric optimization principle
★★★★★★★★★ Accuracy-equivalent → ±0.22 pp on GSM8K
★★★★★★★★★ RTX 4090 NOT affected → DCP=1 → no benefit but no risk
★★★★★★★★★ DCP becoming standard for long-context → vLLM + Ascend + Megatron convergence
★★★★★★★★★ #45966 A2A workspace → related DCP infrastructure
★★★★★★★★★ #45972 Revert DSV4 cudagraph optimization → WATCH

---

## References

- vLLM #45964: https://github.com/vllm-project/vllm/pull/45964
- vLLM #45966: https://github.com/vllm-project/vllm/pull/45966
- vLLM #44044/#43729: FlashInfer-MLA DCP decode-LSE
- vLLM-Ascend #10225: DyCP
- Megatron #5389: GDN THD all-to-all
- RFC #34018: DCP query replication (prematurely closed)
