# DeepSpeed #8068 MERGED: gradient_clipping Default Changed to 1.0

**Date**: 2026-07-15
**Significance**: ★★★★★★★★ VALIDATES our MUST DO rule #2 (always set gradient_clipping=1.0)
**Status**: MERGED into DeepSpeed main

---

## What Changed

DeepSpeed previously defaulted `gradient_clipping` to `0.0`, which silently DISABLED gradient clipping. This meant:
- No bound on parameter updates → gradient explosion risk
- Especially dangerous for GRPO (group-normalized advantages still can have large gradients)
- Combined with Muon optimizer (#5394/#8068) → clipping is essential to prevent stalling

The fix: default changed from `0.0` to `1.0` in DeepSpeed config.

---

## Impact on Our Work

1. **Our MUST DO rule #2 validated**: We always recommended `gradient_clipping=1.0` — now it's the default!
2. **RTX 4090 config**: Still explicitly set `gradient_clipping=1.0` for clarity (not relying on defaults)
3. **Fork PR impact**: Jackie2049/DeepSpeed PR #1 (overlap_comm stream race) does NOT touch gradient_clipping — still valid independently
4. **Muon interaction**: With default=1.0, Muon optimizer now has proper clipping → #5394/#5395 partially addressed

---

## What This Means for Future DeepSpeed Versions

From v0.19.2+:
- `gradient_clipping=1.0` is now the default → safer for all training
- No need to explicitly set it (but still recommended for clarity)
- Our OSS comment draft for #8068 can reference this merge as validation

---

## Updated Monitor Item

#8068: **RESOLVED** (merged, default=1.0). Remove from critical monitor, add to validated rules.
