# 7-Framework Critical Developments Report — June 20, 2026

> Created: 2026-06-20 | Source: Background agent 7-framework monitor
> Focus: RTX 4090 GRPO Training implications

## ★★★★★★★★ NEW CRITICAL FINDINGS

### 1. verl #6782 — LoRA GRPO EOS Bug (NEW, CRITICAL)
- **Title**: Qwen3.5-27B LoRA GRPO: vLLM never emits EOS with rank=64/alpha=128
- **Severity**: CRITICAL — directly blocks GRPO training with larger LoRA ranks
- **RTX 4090 Impact**: CRITICAL — LoRA rank=64 + alpha=128 → all responses truncated → no valid completions
- **Workaround**: Use rank=32/alpha=64 (our recommended config avoids this bug!)
- **Status**: OPEN (created 2026-06-17)
- **★★★★★★★★ Our RTX 4090 config uses r=32 → NOT affected by this bug**

### 2. Megatron #5394 — Muon optimizer clipping stalls (NEW, CRITICAL)
- **Title**: ChainedOptimizer applies global grad-norm clipping to Muon → silently stalls training
- **Severity**: CRITICAL for Muon users — training silently stalls without error
- **RTX 4090 Impact**: HIGH (not using Muon on RTX 4090, but pattern relevant)

### 3. SGLang #28771 — EAGLE accept_length degradation (NEW, CRITICAL)
- Already saved in sglang-28771-eagle-accept-length-degradation-reading.md

### 4. SGLang #28752 — DSA indexer OOM (NEW, HIGH)
- **Title**: HiSparse DSA indexer memory budgeting bug → startup OOM with host_to_device_ratio > 1
- **RTX 4090 Impact**: HIGH for MoE models with DSA

### 5. PyTorch #187759 — svdvals NaN swallowing (NEW, relevant)
- **Title**: torch.linalg.svdvals silently swallows NaN, returning finite values
- **Impact**: Muon optimizer uses SVD → NaN silently absorbed → debugging harder

## Previously Tracked Issues — Status Updates

| Framework | Issue | Status Change | Notes |
|-----------|-------|---------------|-------|
| DeepSpeed | #8080 | OPEN (fix PR) | Fix correct, awaiting review |
| DeepSpeed | #8078 | NEW | Avoid CUDA context at import time |
| DeepSpeed | #8072/#8076 | STALLED | 0 maintainer comments for days |
| vLLM | #46125 | OPEN (revert PR) | Revert removes stale cache fix → RLHF risk |
| vLLM | #46203 | NEW | ROCm cumem sleep fix (same pattern as #45552) |
| SGLang | #28754 | NEW | KV-commit bookkeeping unification (related to #28771) |
| SGLang | #28703 | OPEN | DSA LoRA targets (needed for GRPO) |
| verl | #6782 | ★★★★★★★★ NEW CRITICAL | LoRA rank=64 EOS bug |
| verl | #6794 | DRAFT | Delta weight sync (4 sub-issues) |
| verl | #6512 | MERGED | Per-unit LoRA (RTX 4090 WIN) |
| Megatron | #5394 | ★★★★★★★★ NEW CRITICAL | Muon clipping stalls training |
| Megatron | #5395 | CHANGES_REQUESTED | Skip grad-norm clip for Muon |
| Megatron | #5400 | NEW | Route GDN to Adam instead of Muon |
| Megatron | #5401 | NEW | MoE router z-loss + TE CUDA Graph |
| vLLM-Ascend | #10702 | NEW | npugraph_ex crash on PD decode |
| vLLM-Ascend | #10735 | NEW | npugraph_ex override persistence fix |
| PyTorch | #187653 | CI running | NanDetectMode |
| PyTorch | #187749 | NEW | CUDA graph debug flag + capture hooks |

## Cross-Framework Interactions

1. **DeepSpeed #8061 + PyTorch #187653**: Stream race → NaN → NanDetectMode would catch forward-pass NaN
2. **verl #6782 + vLLM #46125**: LoRA EOS bug + stale encoder cache may interact
3. **SGLang #28771 + #28754 + verl #6794**: EAGLE degradation + KV-commit fix + delta sync interact
4. **Megatron #5394 + PyTorch #187759**: Muon clipping + SVD NaN swallowing → double masking

## RTX 4090 GRPO Immediate Action Items

1. ★★★★★★★★ Our config (LoRA r=32) AVOIDS the verl #6782 EOS bug — confirmed safe
2. ★★★★★★★★ Monitor EAGLE accept_length during rollout — restart if < 2.0
3. ★★★★★★★★ Apply SGLang #28752 DSA indexer memory fix if using MoE + DSA
4. ★★★★★★★★ Continue using overlap_comm=False (safe regardless of #8080 status)
5. ★★★★★★★★ Test PyTorch NanDetectMode when it merges (NaN debugging)
6. ★★★★★★★★ DO NOT use Muon optimizer (stalls under global clipping)

## New MUST NOT Rule
- **NOT use LoRA rank >= 64 with vLLM rollout in GRPO** (verl #6782 EOS bug)
  - Our config (r=32) is SAFE
  - Workaround: use r=32/alpha=64 until bug fixed
