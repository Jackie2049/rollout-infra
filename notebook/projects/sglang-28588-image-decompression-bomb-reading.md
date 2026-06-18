# SGLang #28588 — Image Decompression Bomb Guard Deep Reading

> 2026-06-18 | Source-level deep reading of 2nd SGLang security issue this week
> ★★★★★★★★ Unbounded image decompression → OOM before GPU inference → pre-inference DoS!
> ★★★★★★★★ nano_nemotron_vl.py disables PIL's built-in guard (MAX_IMAGE_PIXELS = None)!

---

## 1. Vulnerability Details

```
★★★★★★★★★ _load_image function (common.py:911):
  → Image.open(BytesIO(image_bytes)) → ZERO pixel-count guard before decoding!
  → PIL Image.open is LAZY → reads header only → does NOT decompress pixels
  → But caller downstream triggers full decode via .convert() / .load()
  → By then: 12000x12000 PNG (144M pixels) → ~432 MB RGB buffer ALREADY decompressed!
  → SGLANG_IMAGE_MAX_PIXELS (smart_resize in qwen_vl.py) applied AFTER full decode → useless guard!

★★★★★★★★★ Worse: nano_nemotron_vl.py line 74:
  → Image.MAX_IMAGE_PIXELS = None → COMPLETELY DISABLES PIL's built-in decompression bomb warning!
  → vLLM has SAME pattern in its nano_nemotron_vl.py and nemotron_vl.py
  → ★★★★★★★★ This means ANY VLM request with oversized image → bypasses ALL guards!

★★★★★★★★★ Exploit mechanics:
  → Send valid oversized image via /v1/chat/completions with data:image/png;base64,...
  → Each request: pins CPU core at ~100% for 1m50s+, grows RSS by ~1 GB
  → With --max-running-requests 256 → concurrent oversized → linear memory scaling → OOM!
  → ★★★★★★★★ Pre-inference DoS → server crashes BEFORE GPU inference starts!
```

---

## 2. Fix Pattern

```
★★★★★★★★★ The PR adds _check_image_pixels(width, height) right after Image.open BEFORE pixel decode:

1. Image.open(BytesIO(image_bytes)) → lazy → header only → width/height available
2. _check_image_pixels(image.width, image.height) → guard BEFORE committing to decode
3. Raises ValueError if width * height > SGLANG_IMAGE_MAX_DECODE_PIXELS (default 89,478,485 = PIL default)
4. ★★★★★★★★ CORRECT pattern: check header dimensions before committing to full decode!

★★★★★★★★★ GPU JPEG path (nvJPEG) unaffected → processes on-device → no CPU memory explosion
```

---

## 3. Comparison with #28582 RCE

```
★★★★★★★★★ Same attack surface: SGLang HTTP API endpoints

| Aspect | #28582 RCE (CVSS 9.8) | #28588 DoS |
|--------|----------------------|------------|
| Severity | RCE → arbitrary code execution | DoS → server OOM |
| Auth required | NONE (missing @auth_level) | Legitimate-looking request |
| Attack vector | /load_lora_adapter_from_tensors → snapshot_download → RCE | /v1/chat/completions → oversized image → OOM |
| Exploitability | Requires LoRA endpoint access | ANY VLM user can exploit |
| Fix pattern | Add @auth_level + path validation | Add pre-decode pixel guard |
| Broadly exploitable | Less (LoRA admin ops) | ★★★★★★★★ More (any VLM user!) |

★★★★★★★★★ #28582 worse severity (RCE > DoS) but #28588 more broadly exploitable!
```

---

## 4. RTX 4090 HYBRID Mode Risk

```
★★★★★★★★★ HYBRID localhost deployment (typical RTX 4090 rollout):
  → #28582 RCE: requires local access → reduced risk but NOT eliminated
  → #28588 DoS: MORE relevant for RTX 4090!
    → 24 GiB VRAM + limited system RAM → single oversized image → OOM
    → --max-running-requests high for throughput → concurrent large-image requests → rapid memory exhaustion
    → ★★★★★★★★ MUST apply #28588 guard even in HYBRID localhost mode!

★★★★★★★★★ Cross-framework: vLLM has same nano_nemotron_vl.py MAX_IMAGE_PIXELS=None pattern!
```

---

## Key Findings Summary

★★★★★★★★★ _load_image had zero pixel-count guard before full image decode → pre-inference DoS!
★★★★★★★★★ nano_nemotron_vl.py DISABLES PIL's built-in guard (MAX_IMAGE_PIXELS=None) → bypass ALL guards!
★★★★★★★★★ vLLM has SAME disabled PIL guard pattern → cross-framework vulnerability!
★★★★★★★★★ Fix: check header dimensions before committing to decode → SGLANG_IMAGE_MAX_DECODE_PIXELS
★★★★★★★★★ #28582 RCE worse severity but #28588 DoS more broadly exploitable (any VLM user!)
★★★★★★★★★ RTX 4090 HYBRID: #28588 DoS MORE relevant than #28582 RCE for localhost deployment

---

## References

- SGLang #28588: https://github.com/sgl-project/sglang/pull/28588 (DoS guard)
- SGLang #28582: https://github.com/sgl-project/sglang/pull/28582 (RCE fix)
- Source: sglang/python/sglang/srt/utils/common.py:911 (_load_image)
- Source: sglang/python/sglang/srt/multimodal/processors/nano_nemotron_vl.py:74 (MAX_IMAGE_PIXELS=None)
- Source: sglang/python/sglang/srt/environ.py:572 (SGLANG_IMAGE_MAX_DECODE_PIXELS)
- vLLM same pattern: vllm/vllm/multimodal/processors/nano_nemotron_vl.py
