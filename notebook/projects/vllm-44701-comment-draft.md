# vLLM Issue #44701 — Comment Draft

> Status: DRAFT — User should review before posting
> Issue: https://github.com/vllm-project/vllm/issues/44701
> Title: [Bug]: V1 prefix-cache extra-key domain collision between LoRA name and cache_salt

## Proposed Comment

```
Thanks for flagging this. I've been studying vLLM V1's LoRA serving + prefix caching interaction in depth and can confirm this collision is a real correctness issue, not just a theoretical concern.

**The root problem**: `_gen_lora_extra_hash_keys` and `cache_salt` both produce strings that get concatenated into the same `extra_keys` tuple without domain separation. This means:

1. A LoRA adapter named "cache_salt_xyz" would produce identical hash keys as a `cache_salt` value "xyz" with no LoRA → silent KV cache corruption
2. Two different LoRA adapters whose names happen to collide with different `cache_salt` values could share prefix blocks → incorrect attention outputs

**Impact for GRPO training**: This is particularly relevant for GRPO rollout scenarios where `rollout_n=8` responses share the same system prompt prefix. With the same LoRA adapter, prefix caching should work correctly (same adapter → same hash). But with multi-tenant serving where different adapters process different requests, this collision could silently corrupt KV cache.

**Comparison with SGLang**: SGLang's RadixAttention handles this correctly by maintaining per-adapter KV cache trees. Each adapter gets its own subtree, so prefix blocks never cross adapter boundaries. This is safer but also simpler — no hash collision possible by design.

**Suggested fix direction**: Instead of flat string concatenation, use structured domain separation:
- `extra_keys = ("lora:" + lora_name, "salt:" + cache_salt)` instead of `(lora_name, cache_salt)`
- Or use a hash function that includes domain tags: `hash(domain + ":" + value)`
- This ensures LoRA identity and cache_salt can never collide regardless of their string values

I've documented this interaction in detail at my research notes (https://github.com/Jackie2049/rollout-infra/blob/main/notebook/projects/vllm-lora-serving-reading.md) — the "LoRA + prefix caching不兼容" section covers the hash collision analysis from the source code level.
```

## Key Points for User Review

1. ★★★ This comment references our actual source-code-level analysis → demonstrates deep expertise
2. ★★★ Mentions GRPO training impact → connects to our core research area
3. ★★★ Comparison with SGLang RadixAttention → shows cross-framework knowledge
4. ★★★ Concrete fix direction → not just "this is broken" but "here's how to fix it"
5. ★★★ Link to our public research notes → credibility + depth

## Why This is Tier 1

- ★★★★★ Most unique expertise match: LoRA serving + prefix caching + vLLM V1 source-level knowledge
- ★★★ SM89 (RTX 4090) perspective is scarce in vLLM community
- ★★★ Concrete, actionable contribution — not just observation but proposed fix
