# 7-Framework Developments Tracker — June 19, 2026 (Updated)

**Generated: 2026-06-19 | Cross-framework comprehensive check (updated with 24-48h scan)**

---

## Key Changes Since Previous Check

### IMPROVING
1. **Megatron #5395**: Substantive review response addressing ALL 4 findings. 41 layer-wise tests green. Progressing toward merge.
2. **vLLM #45819**: Extensive local verification by yuvalluria. GDN_ATTN confirmed working, e2e blocked by MoE quant.
3. **vLLM-Ascend #10684**: FIRST maintainer engagement — COLLABORATOR tagged SOMEONEUNSEEN.
4. **SGLang #28695**: ReplaySSM +13.1% throughput, Triton on SM89, ring cursors for RTX 4090.

### WORSENED / NEW CONCERNS
1. **verl #6468**: Multi-user FSDP2 leak scaling confirmed: 0.6 GiB/step (2B) to 6.3 GiB/step (35B). RTX 4090 OOMs in ~40 steps.
2. **DeepSpeed #8061**: Production evidence detailed — multi-stream race condition confirmed with profiler trace.
3. **vLLM #46118**: NEW — MTP+grammar FSM conflict, 58% failure on RTX 4090.
4. **SGLang #28676**: Still 0 maintainer engagement on CRITICAL MoE cache clobber blocker.

### STALLED (NO MOVEMENT)
1. DeepSpeed #8072/#8073/#8068: All 0 reviews
2. vLLM #46088, #45683: 0 comments
3. PyTorch #184119: Stalled at reviewer engagement
4. rLLM #605: 17+ days, ZERO comments

---

## New PRs and Issues (48h)

### DeepSpeed
- #8076 NEW: Independent user confirms #8072 regression (4xH100 + LLaMA Factory + Qwen3.5-9B)
- #8077 NEW: CI diff-driven test selection (infra, not RTX 4090)

### Megatron
- #5400 NEW: Route GDN in_proj to Adam instead of Muon (skip_orthogonalization flag)
- #5401 NEW: Fix MoE router z-loss + TE CUDA graph capture compatibility

### vLLM
- #46118 NEW: MTP+grammar FSM conflict (RTX 4090 reproduction!)

### verl
- #6795 NEW: Remove invalid single_turn_response_length override
- #6796 NEW: Fix fully_async metrics logging alignment
- #6793 NEW: Open-R1 multimodal + TinyLLaVA-Video-R1 dataset support
- #6779 NEW: Continuous Token mechanism for agentic rollout (multi-turn tokenization builder)

### SGLang
- #28695 NEW: ReplaySSM Ring Spec-Verify (+13.1% throughput, -11.8% TPOT)
- #28692 NEW: Fix mamba radix partial page prefix matching
- #28689 NEW: MoE triton_kernels backend quant-arg dedup

### PyTorch
- #187653: CI triggered, 21 new failures (mostly doctest/public API)
- #187636: CI triggered, 3 new failures (inductor_timm, distributed, AOTInductor)

### vLLM-Ascend
- #10684: First maintainer engagement (COLLABORATOR tagged reviewer)

---

*Updated 2026-06-19. Based on 7-framework 24-48h comprehensive scan.*
