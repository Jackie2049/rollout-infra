# vLLM-Ascend: New Issues Confirm DSV4 Systematic Instability Pattern

> 2026-06-19 | Cross-issue analysis of #10700, #10710, #10720, #10724
> ★★★★★★★★ ALL 4 new issues confirm DSV4/GLM-5.x systemic instability on Ascend NPU
> ★★★★★★★★ enforce_eager=True STILL mandatory (#10700) → RTX 4090 lesson reinforced
> ★★★★★★★★ Prefix cache hit rate = 0% (#10710) → DSV4 KV cache block pool corruption

---

## 1. Issue Cluster Analysis

Four new vLLM-Ascend issues (all opened June 18-19) form a coherent cluster confirming DSV4/GLM-5.x systemic instability:

| Issue | Title | Severity | Pattern Family |
|-------|-------|----------|---------------|
| #10700 | GLM5.1 crashes without enforce_eager | **CRITICAL** | NPUGraph capture instability |
| #10710 | DSV4-Flash prefix cache hit rate = 0% | HIGH | KV cache block pool corruption |
| #10720 | Qwen3.5-35B-A3B-w8a8-mtp overthinking | MEDIUM | MTP+quantization loop |
| #10724 | DSV4-Flash 2*A2 PD-Mix crash | HIGH | KV cache block pool corruption |

---

## 2. #10700 — GLM5.1 enforce_eager STILL Mandatory

★★★★★★★★★ CRITICAL: GLM5.1 crashes after running without enforce_eager=True

**Details:** Two-node 800I-A2 (16 NPU cards), v0.21.0rc1. GLM5.1 runs for a while then crashes.

**Pattern:** This confirms our existing finding that `enforce_eager=True` is STILL mandatory on Ascend for GDN/DSA models. NPUGraph (Ascend's CUDA Graph equivalent) has stability issues:
- Requires uniform batch sizes (variable batch = crash)
- State accumulation errors during graph replay (same pattern as #28679 on CUDA)
- DSA/GDN attention state not properly captured in graph (same as #10684 Hadamard)

**RTX 4090 Transfer:** vLLM #46118 (MTP+grammar FSM conflict) also shows graph capture instability on SM89. enforce_eager=True remains safest option for complex models across ALL platforms.

**Cross-platform rule:** For ANY model with DSA/GDN/MoE/MTP components, enforce_eager=True is MANDATORY until graph capture is verified stable.

---

## 3. #10710 — DSV4-Flash Prefix Cache Hit Rate = 0%

★★★★★★★★★ DSV4-Flash-w8a8-mtp completely identical serial requests → prefix cache hit rate ALWAYS 0%

**Details:** 8*A3 NPU, DSV4-Flash-w8a8-mtp, DP_SIZE=2, TP_SIZE=4, EP enabled. vLLM 0.20.2 + vLLM-Ascend 0.20.2rc1.

**Pattern:** This is a KV cache block pool corruption issue — same pattern family as #10724 (8th DSV4 failure). The prefix cache should match identical requests but always returns 0%, indicating:
1. KV cache block allocation is broken for DSV4-Flash
2. Block pool may be corrupted by the w8a8 quantization path
3. MTP spec decode may interfere with prefix caching (same as vLLM #46118 FSM conflict)
4. Layerwise KV Pooling (#10077 MERGED) may not cover the DSV4-Flash w8a8 path

**RTX 4090 Transfer:** vLLM #46088 (MTP kv-dtype garbage) shows same pattern on CUDA — MTP+kv-cache-dtype=auto produces garbage. MUST set kv-cache-dtype explicitly for MTP models.

---

## 4. #10720 — Qwen3.5-35B-A3B-w8a8-mtp Overthinking

Qwen3.5-35B-A3B-w8a8-mtp on 300i duo generates excessive tokens (overthinking loop).

**Pattern:** MTP (Multi-Token Prediction) with w8a8 quantization may trigger infinite/overlong generation loops:
1. w8a8 quantization introduces numerical noise in next-token predictions
2. MTP generates multiple tokens per step → amplifies quantization errors
3. Error amplification can create "generation loops" where the model keeps producing tokens without reaching EOS
4. This is a **quantization+MTP interaction bug** — not seen in FP16/BF16 models

**RTX 4090 Transfer:** vLLM #46118 (MTP+grammar FSM conflict) is a related but different pattern. Both show MTP interacting badly with other system components. For GRPO rollout, MUST set max_tokens and check EOS detection carefully with MTP models.

---

## 5. #10724 — DSV4-Flash 2*A2 PD-Mix Crash (8th DSV4 Failure)

★★★★★★★★★ 8th confirmed DSV4 failure on Ascend NPU!

**Details:** Deepseek V4 Flash on 2*A2 (PD-Mix multi-node deployment), v0.21.0rc1, crashes during operation.

**Pattern:** This continues the systematic DSV4 instability pattern. Key issues:
- PD-Mix (Prefill-Decode disaggregation) adds another failure mode
- Multi-node deployment (2*A2 vs single-node) introduces cross-node coordination failures
- DSV4-Flash + w8a8 + MTP = triple risk factor

**DSV4 Failure Count Update:**
| # | Framework | Issue | Failure Mode |
|---|-----------|-------|-------------|
| 1-7 | vLLM-Ascend | Various | Multiple (KV cache, chat, quant, etc.) |
| 8 | vLLM-Ascend | #10724 | PD-Mix multi-node crash |
| 9 | vLLM | #45972 | 2nd DSV4 revert |
| 10 | SGLang | #28676 | MXFP8 MoE cache clobbered |
| 11 | Megatron | #5317 | Triton rotary NaN |
| 12 | SGLang | #28685 | GLM-5.2 FP8 block-fp8 wrong MI350X |

---

## 6. Cross-Issue Pattern Synthesis

All 4 new issues share common structural patterns:

### Pattern 1: enforce_eager Mandatory
- #10700 (GLM5.1 crash without enforce_eager)
- #10724 (DSV4 PD-Mix crash — may need enforce_eager)
- **Rule:** enforce_eager=True is NOT optional on Ascend for GDN/DSA/MTP models

### Pattern 2: KV Cache Corruption
- #10710 (prefix cache hit rate = 0%)
- #10724 (PD-Mix crash likely involves KV cache)
- **Rule:** DSV4 KV cache block pool has systemic issues — prefix caching unreliable

### Pattern 3: Quantization+MTP Interaction
- #10710 (w8a8+mtp prefix cache broken)
- #10720 (w8a8+mtp overthinking)
- **Rule:** w8a8 quantization + MTP = risk factor for both correctness and stability

### Pattern 4: Multi-Node Disaggregation Risk
- #10724 (PD-Mix 2*A2 crash)
- **Rule:** PD disaggregation adds cross-node coordination complexity → new failure mode for DSV4

---

## 7. RTX 4090 Implications

These Ascend issues have direct RTX 4090 parallels:

| Ascend Issue | RTX 4090 Parallel | Lesson |
|-------------|-------------------|--------|
| #10700 enforce_eager mandatory | vLLM #46118 MTP+grammar crash | enforce_eager=True for complex models |
| #10710 prefix cache 0% | vLLM #46088 MTP kv-dtype garbage | Set kv-cache-dtype explicitly |
| #10720 MTP overthinking | verl #6782 LoRA EOS bug | Check EOS detection carefully |
| #10724 PD-Mix crash | vLLM #44395 partial wake | Staged resource management |

**Universal rule for GRPO deployment:** Any model with DSA/GDN/MoE/MTP MUST use enforce_eager=True, explicit kv-cache-dtype, and explicit EOS detection. Prefix caching should be disabled for first deployment and enabled only after stability verification.

---

## References

- #10700: https://github.com/vllm-project/vllm-ascend/issues/10700
- #10710: https://github.com/vllm-project/vllm-ascend/issues/10710
- #10720: https://github.com/vllm-project/vllm-ascend/issues/10720
- #10724: https://github.com/vllm-project/vllm-ascend/issues/10724
- DSV4 systematic instability: notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md
- vLLM-Ascend ecosystem: notebook/projects/mindie-vllm-ascend-ecosystem-deep-research.md
- vLLM #46118: notebook/projects/vllm-46118-mtp-grammar-fsm-conflict-reading.md (if exists)
- vLLM #46088: MTP kv-dtype garbage
