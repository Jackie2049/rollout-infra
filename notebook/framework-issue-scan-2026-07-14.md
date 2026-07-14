# Framework Issue Scan — 2026-07-14

## Methodology
Scanned vLLM, verl, SGLang, Megatron-LM, DeepSpeed for open issues matching our expertise areas: GRPO stability, NaN/FP8, MoE, weight sync, think markers, template bugs.

## Key Findings by Framework

### verl (most actionable)

| # | Title | Relevance | Priority |
|---|-------|-----------|----------|
| #6468 | CPU memory leak during FSDP2 rollout weight sync | **HIGH** — weight sync is our core expertise; DAPO training on Qwen3.5-2B | **A** |
| #7022 | UP-GRPO asymmetric policy loss | **HIGH** — new GRPO variant, relates to our bypass_mode/top_n_sigma work | **B** |
| #6974 | Delta weight sync over NCCL | **HIGH** — weight sync feature, byte-diff approach | B (review) |
| #2911 | loss≈0 early training with clip_cov | **MED** — our GRPO patches could help | C |
| #6856 | vLLM rollout DP>1 fails multi-node | MED — multi-node topology | C |
| #6473 | DeepSeek V4 GRPO support | MED — GRPO + MoE | C |

### vLLM

| # | Title | Relevance | Priority |
|---|-------|-----------|----------|
| #48585 | FP8 NaN/Inf logprob → JSON serialization crash | HIGH — FP8 NaN handling | **A** (already has bugfix PR) |
| #48587 | CUTLASS FP8 block-scaled N not multiple of 128 | HIGH — FP8 + CUTLASS SM90 | A (already has bugfix PR) |
| #48541 | FlashInfer CUTLASS MoE on fp4-less builds | MED — MoE + CUTLASS | B |
| #48574 | Wrong expert count to WNA16 MoE block config | MED — MoE config | B |
| #46042/#46616 | MiniMax-M3 `<mm:think>` reasoning leak | MED — think marker handling | B |
| #48411 | Inline per-token-head scales in offloaded page transfer | MED — KV cache FP8 | B |

### Megatron-LM

| # | Title | Relevance | Priority |
|---|-------|-----------|----------|
| #5798 | MoE aux loss MBS-dependent gradient scaling | **HIGH** — MoE load balancing bug | **A** |
| #5470 | Muon FP8 param gather decoupled layout | HIGH — FP8 + optimizer | A (already open PR) |
| #5735 | NCCL EP zero copy | MED — Expert Parallelism | B |
| #5745 | DeepEP v2 dispatcher | MED — Expert Parallelism | B |
| #4922 | Don't route pad/dummy tokens to experts | MED — MoE routing | B |
| #5276 | Delegate reasoning token retention to chat template | MED — think markers | B |

### SGLang
Few open bug issues. LoRA + tokenizer-worker (#31084) is the most recent.

### DeepSpeed
Repo access issues — need to verify correct org name (may be deepspeed-ai or under HF).

## Top 3 Actionable Contributions

1. **verl #6468 — Weight sync memory leak diagnosis**
   - CPU memory grows linearly during FSDP2 DAPO training
   - Happens after actor param update / rollout weight sync
   - Ray object store near zero, leak is in worker processes
   - We have weight sync + FSDP2 expertise to investigate root cause

2. **Megatron #5798 — MoE aux loss MBS-dependent gradient**
   - seq_aux_loss gradient has spurious 1/MBS factor
   - routing_map reshape uses seq_length not seq_len*MBS
   - We understand MoE load balancing math from our EP work

3. **verl #7022 review — UP-GRPO asymmetric policy loss**
   - Self-anchored ratio with stop-gradient trick for Â>0
   - Standard GRPO clip for Â≤0
   - Our bypass_mode and top_n_sigma patches are related primitives

## Next Steps
- Our verl NaN guard (Jackie2049/verl PR #6) is UNIQUE — no competing PR exists
- Monitor Megatron #5760/#5761 (FP8 autocast fix already in draft PR)
- Monitor SGLang #31118/#31119 (streaming think truncation fix already in PR)
- DeepSpeed AutoEP gaps: ZeRO-3+folding (HIGH), ETP>1 (HIGH), offload+folding (MED-HIGH)

## Cross-Framework Bug Patterns (from background scan)

| Pattern | Repos | Issues |
|---------|-------|--------|
| NaN / loss divergence | Megatron, PyTorch Inductor, SGLang | #5782, #189808/189801/189799, #31133 |
| MoE / EP | DeepSpeed, Megatron, vLLM, SGLang, verl | #8085/8084, #5749/5781/5776, #48541, #31116, #7016 |
| FP8 quantization | vLLM, SGLang, Megatron | #48592/48508, #31103, #5760 |
| Streaming / chat template | SGLang | #31118 |
| GRPO / RL training | verl, rLLM | #7016/7004, #605/717 |

## Top New Findings (with PR status)

1. **Megatron #5760** — `with ctx and fp8_ctx:` → autocast never entered. PR #5761 exists (draft by huthvincent). One-line fix. MONITOR.
2. **SGLang #31118** — streaming truncation of `hés`/` ` tags → infinite reasoning. PR #31119 exists. MONITOR.
3. **verl #7016** — FSDP2 + MoE backward crash. No PR fix yet. WATCH for opportunity.
4. **PyTorch #189799** — Inductor -0.0/subnormal NaN cluster. No fix PR yet. RESEARCH.
5. **DeepSpeed #8085** — MoE ZeRO-3 full-materialization OOM. PR #8103 in progress.
