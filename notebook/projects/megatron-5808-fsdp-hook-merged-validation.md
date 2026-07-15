# Megatron #5808 MERGED: Fix MegatronFSDP Root Module Hook Dispatch

**Date**: 2026-07-15
**Significance**: ★★★★★★★★ VALIDATES our Jackie2049/Megatron-LM PR #2 (FSDP hook dispatch)
**Status**: MERGED into Megatron-LM main

---

## What Changed

MegatronFSDP had a bug where root module hook dispatch was incorrect for distributed training:

- Root module hooks were dispatched on ALL parameters instead of only the root module's parameters
- This caused incorrect gradient accumulation and synchronization in FSDP training
- Our PR #2 on Jackie2049/Megatron-LM addressed the same issue independently

The fix: correct hook dispatch to only target the root module's parameters.

---

## Impact on Our Work

1. **Our Megatron PR #2 validated**: Jackie2049/Megatron-LM PR #2 (FSDP hook dispatch fix) addressed the same bug
2. **Second fork PR validated by official fix**: Megatron #2 → NVIDIA #5808 (after vllm #9 → #48638)
3. **FSDP training safety**: Root module hook dispatch now correct → gradient flow verified
4. **PR #2 status**: Since upstream merged the fix, our PR #2 can reference #5808 and note it was independently discovered

---

## What This Means for Future Megatron-LM Versions

From this merge onward:
- FSDP root module hook dispatch is correct
- Gradient accumulation and synchronization work properly
- Our fork PR #2 is redundant (upstream fix covers it) but demonstrates independent discovery

---

## Updated Monitor Item

#5808: **RESOLVED** (merged). Our PR #2 validated by official fix. Same pattern as vLLM #9 → #48638.

---

## Pattern: Fork PR → Official Validation

This is the 2nd time our fork PR was validated by an official upstream fix:
1. Jackie2049/vllm PR #9 → vLLM #48638 (encoder cache revert)
2. Jackie2049/Megatron-LM PR #2 → NVIDIA #5808 (FSDP hook dispatch)

Pattern: We identify bugs independently, then official fixes validate our analysis.
