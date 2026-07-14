# Cross-Framework Rule Updates (Session 7+8 Findings)

## New Rules Identified

### MUST DO (2 new candidates)

| # | Rule | Why | Bug IDs | Source |
|---|------|-----|---------|--------|
| 20 | Validate cos/sin dim before fused MLA RoPE | OOB read → nondeterministic NaN at iter 2 | #5317, #5799 | Megatron #5799 |
| 21 | LoRA lora_expand block_n=32 on sm_90 | block_n=128 NaN on Hopper (H100/H20) | #48590 | vLLM #48590 |

### MUST NOT (1 new candidate)

| # | Rule | Why | Bug IDs | Source |
|---|------|-----|---------|--------|
| 18 | GDN batch_invariance flag alone | Both FlashInfer (18) & Triton (22) mismatches at bs=32 | #48613, #45819 | vLLM #48613 |

### WARN (2 new)

| # | Rule | Why | Bug IDs |
|---|------|-----|---------|
| 3 | GDN batch invariance unsupported | Batch-dependent logprobs → inconsistent GRPO advantages | #48613 |
| 4 | LoRA NaN on Hopper with block_n=128 | Only affects sm_90, workaround available | #48590 |

## Validated Config Update

The existing RTX 4090 config (32/34 PASS) already covers most new findings:
- ZeRO-3 (must_not) already covers #8130 frozen params issue
- FP32 gating softmax (must_do #17) already covers MoE NaN
- enforce_eager=True (must_do #5) avoids CUDA graph issues
- FSDP1 (must_do #6) avoids FSDP2 issues

## New Cross-Framework Connections

| Pattern | Frameworks | Bug IDs | Rule Status |
|---------|-----------|---------|-------------|
| **Triton OOB tile read** | Megatron, vLLM | #5799, #48590 | NEW — affects Triton kernels across frameworks |
| **Cross-chunk tag truncation** | SGLang, TRL (HF) | #31118, #6361 | Existing — our fork fixes both |
| **FSDP param lifecycle** | Megatron | #5788, #5789, #5790 | 3 bugs in same FSDP implementation |
