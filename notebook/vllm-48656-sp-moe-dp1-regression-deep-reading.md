# vLLM #48656: SP-MoE dp=1 Regression + #48657 Override — RTX 4090 CRITICAL

**Date**: 2026-07-15 (Session 10)
**Issue**: vllm-project/vllm #48656
**Fix PR**: #48657 (+41/-0)
**Significance**: ★★★★★★★★ RTX 4090 CRITICAL — SP-MoE auto-enabled on dp=1 causes -19% throughput, -24% KV cache

---

## 1. Problem

#48036 removed `and self.data_parallel_size > 1` from `ParallelConfig.use_sequence_parallel_moe`. This auto-enables sequence-parallel MoE on **every** TP+EP deployment without data parallelism.

### Impact on dp=1 (RTX 4090 single GPU):

| Metric | SP-MoE OFF (pre-#48036) | SP-MoE ON (current main) | Delta |
|--------|-------------------------|--------------------------|-------|
| GPU KV cache | 1,032,448 tok | 784,300 tok | **-24%** |
| N=1 throughput | 75 tok/s | 61 tok/s | **-19%** |
| N=8 throughput | 407 tok/s | 353 tok/s | **-13%** |
| N=32 throughput | 931 tok/s | 875 tok/s | **-6%** |

### Why it's bad on dp=1:
- Extra `all_gather/reduce_scatter` per MoE layer costs MORE than duplicate expert work it avoids
- Activation memory for SP MoE is wasted on single GPU
- No benefit: SP-MoE avoids duplicate expert computation across DP ranks → but dp=1 means no duplication
- KV cache loss directly cuts how many long sessions fit on GPU → CRITICAL for GRPO rollout

### Why this configuration is common:
- Serving one large MoE model on single node with TP+EP is standard single-replica setup
- Model doesn't fit on one GPU → TP mandatory → EP natural choice for expert layers
- Before #48036: these deployments NEVER took SP-MoE path
- After #48036: ALL silently auto-enable SP-MoE, no opt-out

---

## 2. Fix (#48657)

Adds explicit `--sequence-parallel-moe` / `--no-sequence-parallel-moe` CLI flag:
- Default: `None` → current heuristic keeps deciding (no behavior change)
- `True` → force SP-MoE on regardless of heuristic
- `False` → force SP-MoE off (RTX 4090 SAFE)

```python
# New config option
sequence_parallel_moe: Optional[bool] = None  # None=auto, True=force on, False=force off
```

Lines: +41/-0 (tri-state config, CLI parsing, test coverage)

---

## 3. RTX 4090 Impact

### Single GPU (dp=1, no TP/EP):
- SP-MoE doesn't activate on single GPU (no TP/EP)
- **No impact** for dp=1 single GPU RTX 4090

### Multi-GPU RTX 4090 (TP=2, EP):
- RTX 4090 has NO NVLink → PCIe bottleneck → all_gather extremely slow
- SP-MoE all_gather/reduce_scatter would be DISASTER on PCIe
- **MUST `--no-sequence-parallel-moe`** on any multi-GPU RTX 4090 setup

### Comparison with other platforms:
- H200 (NVLink): SP-MoE regression -19% single, -6% batch → net negative even on NVLink!
- RTX 4090 (PCIe only): regression would be MUCH worse (2.76 GB/s vs 300+ GB/s NVLink)

---

## 4. Cross-Framework Connection

| Bug | Framework | Mechanism | Fix |
|-----|-----------|-----------|-----|
| #48656 | vLLM | SP-MoE auto-enabled on dp=1 | #48657 explicit override |
| #48036 | vLLM | Removed dp>1 guard from SP-MoE heuristic | Root cause PR |
| P9 thesis | PyTorch | SM<90 prologue fusion batch-dependent | choices.py guard |
| #48650 | vLLM | tl.constexpr batch-dependent | Demote to runtime arg |

### Pattern family: Auto-Enable Without Guard
- #48036 removed a critical guard condition (dp>1) → feature auto-enables where it shouldn't
- Similar to P9: Inductor auto-fuses on SM<90 without checking if fusion is appropriate
- Universal lesson: derived properties NEED explicit override mechanisms

---

## 5. GRPO Relevance

For GRPO training on RTX 4090 with MoE models:
- KV cache capacity = number of concurrent GRPO completions possible
- -24% KV cache = fewer concurrent sequences = longer rollout batch time
- MUST use `--no-sequence-parallel-moe` on dp=1 MoE deployments

For verl HYBRID with vLLM rollout:
- vLLM rollout config should explicitly set `sequence_parallel_moe=False`
- This protects KV cache capacity for GRPO concurrent generations

---

## 6. Update Cross-Framework Rules

### New MUST NOT Rule candidate:
| # | Rule | Why | Bug IDs |
|---|------|-----|---------|
| 18 | SP-MoE on dp=1 | -24% KV cache, no benefit | #48656 |

### New MUST DO Rule candidate:
| # | Rule | Why | Bug IDs |
|---|------|-----|---------|
| 20 | --no-sequence-parallel-moe on dp=1 MoE | Preserve KV cache for GRPO | #48656 |

---

## Session Stats
- **Issue research**: #48656 (SP-MoE dp=1 regression) + #48657 fix PR
- **RTX 4090 CRITICAL**: -24% KV cache, -19% throughput on dp=1
- **Fix**: tri-state override (+41/-0), no behavior change by default
- **Cross-framework**: Auto-Enable Without Guard pattern family (P9, #48036, #48656)
