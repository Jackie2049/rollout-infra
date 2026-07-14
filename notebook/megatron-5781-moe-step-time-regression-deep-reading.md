# Megatron #5781: MoE Step-Time Regression in core_r0.18.0

## Overview
- **Issue**: NVIDIA/Megatron-LM #5781 by ko3n1g, July 13, 2026
- **Status**: OPEN, labels: `bug`, `module: moe` — 1 contributor response (CodersAcademy006), 0 maintainer
- **Scope**: Two separate regression causes across release recipes

## The Regression

Two MoE release recipes show step-time degradation between `core_v0.18.0` and `core_r0.18.0`:

### Case A: deepseekv3_proxy_flex_* recipes (4 recipes)
- **Cause**: Deliberate switch from TE-fused cross-entropy loss to native CE implementation
- **Why**: Cherry-picks of PRs #5115 and #5162 hard-assert against fused-TE path, flip recipes to `native`
- **Config change**: `--cross-entropy-fusion-impl: te` → `native`
- **Impact**: Known throughput trade for stability. DeepSeekV3 runs in bf16 so NVFP4 doesn't affect it.

### Case B: nemotron3_super_release_gb200* recipes (2 recipes)
- **Cause**: TE pin bump (Transformer Engine `4220403e` → `b9d690e0`) — 9 commits, all on grouped-MLP / grouped-GEMM / NVFP4 / SReLU path
- **Byte-identical config**: recipe config unchanged across refs — regression entirely from TE
- **Prime suspect**: TE PR #2981 (ScaledSReLU recompute) or TE #3048 (NVFP4 fused grouped MLP)

## Analysis (from contributor CodersAcademy006)

Refined the Case B analysis:
1. **ScaledSReLU recompute path (#2981)**: "Shouldn't be active for this recipe" since nemotron3 doesn't set `--recompute-*` flags
2. **More likely**: The **forward fused kernel swap** in #2981 or **NVFP4 fused grouped-MLP path switch** in #3048
3. **Proposed disambiguation**: Memory signatures from existing W&B runs

## GRPO Relevance
- **Indirect**: MoE training on Megatron core is a common foundation for GRPO pipelines
- **TE interaction**: Shows Megatron-TE version coupling is fragile — any GRPO setup must pin TE versions carefully
- **RTX 4090**: Not directly affected (RTX 4090 doesn't use TE for most operations)

## Cross-Framework
- TE integration fragility pattern: same class as vLLM FlashInfer version coupling issues
- Pattern: **P7 (Weight Reload/Cache Staleness)** — version mismatch at framework-library boundary

## Monitoring
- No maintainer response on root cause
- Contributor analysis helps narrow but no fix identified
- Likely resolved by TE version pin or config workaround
