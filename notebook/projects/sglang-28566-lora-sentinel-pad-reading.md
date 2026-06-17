# SGLang #28566 — LoRA Sentinel-Pad for DP-Attention Foreign Tokens Reading

> 2026-06-18 | PR #28566 OPEN | Author: (from #25141 carve-out)
> ★★★★★★★★ ANOTHER source of multi-LoRA divergence under DP-attention!
> ★★★★★★★★ Foreign tokens get uninitialized/out-of-bounds LoRA mapping → random routing
> ★★★★★★★★ Fix: -1 sentinel convention → foreign tokens mapped to "no LoRA"

---

## 1. Bug: Uninitialized LoRA Mapping for Foreign Tokens

```
★★★★★★★★★ Under DP-attention, gathered forward batch has local + foreign tokens:

Problem in _compute_moe_lora_info (backend/base_backend.py):

  → seg_indptr / weight_indices cover ONLY local requests
  → Foreign positions are NOT covered → but LoRA kernel processes ALL positions

Two wrong behaviors at foreign positions:

CUDA-kernel path:
  → token_lora_mapping = torch.empty(...) → uninitialized
  → Foreign-position slots stay uninitialized when kernel only writes within segments
  → ★★★★★★★★ MoE LoRA hook dispatches ON GARBAGE → random LoRA routing!
  → Same manifestation as #27097 Factor 1 → SGMV dynamic routing diverges

Python fallback path:
  → torch.searchsorted(seg_indptr, ..., right=True) - 1 → returns num_segments
  → torch.index_select(weight_indices, 0, req_indices) → indexes ONE PAST END
  → ★★★★★★★★ Out-of-bounds read → wrong LoRA index → wrong routing

★★★★★★★★★ This is ANOTHER source of multi-LoRA divergence:
  → Same class as #27097 Factor 1 (SGMV dynamic routing)
  → Not just dynamic routing → but GARBAGE routing for foreign tokens
  → Under DP-attention + multi-LoRA → foreign tokens → uninitialized → any LoRA slot → divergence
```

---

## 2. Fix: -1 Sentinel Convention

```
★★★★★★★★★ Two-part fix with same -1 sentinel convention:

1. Pre-fill token_lora_mapping with -1 before either path runs:
  → Un-segmented slots stay -1 after kernel returns
  → ★★★★★★★★ -1 = "no LoRA" → kernel skips these tokens → NO garbage routing

2. Python fallback: torch.cat -1 entry onto weight_indices:
  → Use padded tensor for index_select → same -1 at foreign positions
  → Foreign tokens → -1 → "no LoRA" → correct local behavior

★★★★★★★★★ Why -1 sentinel works:
  → Foreign-token LoRA outputs discarded by DP-attention scatter anyway
  → Mapping to -1 ("no LoRA") is the CORRECT local behavior
  → Just stops kernel from acting on uninitialized data / out-of-bounds reads
```

---

## 3. Relationship to #27097

```
★★★★★★★★★ #28566 is a COMPLEMENTARY fix to #27097:

#27097 Factor 1 (SGMV dynamic routing):
  → Same adapter → different slot index depending on batch composition
  → Dynamic tl.load routing → float32 accumulation divergence

#28566 (LoRA sentinel-pad):
  → ★★★★★★★★ SAME class of problem → wrong LoRA routing
  → But different manifestation: foreign tokens → uninitialized → ANY random LoRA slot
  → Under DP-attention → foreign tokens → garbage routing → divergence
  → Under single-GPU DP=1 → no foreign tokens → bug NOT manifest → safe

★★★★★★★★★ Combined effect with #28499 (csgmv CUDA graph fix):
  → #28499: fixes segment skipping under CUDA graph (Factor 2)
  → #28566: fixes garbage routing for foreign tokens (Factor 1 subset)
  → ★★★★★★★★ Still NOT complete fix → dynamic routing for LOCAL tokens still diverges
  → But eliminates TWO major sources → remaining divergence is float32 accumulation order only

★★★★★★★★★ RTX 4090 impact:
  → DP=1 → no foreign tokens → #28566 bug NOT manifest → single-GPU safe
  → Multi-GPU DP-attention → foreign tokens present → bug manifests
  → ★★★★★★★★ RTX 4090 single GPU NOT affected → but multi-GPUs would be
```

---

## Key Findings Summary

★★★★★★★★★ #28566: LoRA sentinel-pad for DP-attention foreign tokens → -1 sentinel convention
★★★★★★★★★ Bug: foreign tokens get uninitialized/out-of-bounds LoRA mapping → garbage routing
★★★★★★★★★ Fix: pre-fill -1 + padded weight_indices → foreign tokens → "no LoRA" → correct
★★★★★★★★★ Same class as #27097 Factor 1 → but subset: foreign tokens only, not local routing
★★★★★★★★★ RTX 4090 single GPU: NOT affected (DP=1 → no foreign tokens)
★★★★★★★★★ Multi-GPU DP-attention: affected → must use -1 sentinel fix
★★★★★★★★★ Combined with #28499 → eliminates 2 major #27097 sources → float32 divergence remains

---

## References

- SGLang #28566: https://github.com/sgl-project/sglang/pull/28566
- SGLang #27097: https://github.com/sgl-project/sglang/issues/27097
- SGLang #28499: https://github.com/sgl-project/sglang/pull/28499 (csgmv CUDA graph fix)
- #27097 reading: notebook/projects/sglang-27097-multi-lora-determinism-bug-reading.md
- #28499 reading: notebook/projects/sglang-28499-csgmv-cuda-graph-fix-reading.md
