# Megatron-LM June 2026 Critical Developments Report

> Created: 2026-06-20 | Priority: HIGH for RTX 4090 GRPO
> Source: Background agent comprehensive monitoring + deep reading

## 1. CRITICAL: #5394 — Muon Optimizer Clipping Stalls Training

**Severity**: CRITICAL | **Status**: OPEN (June 17)

**Root Cause**: `ChainedOptimizer.step()` computes single global `grad_norm` across all sub-optimizers, then applies `clip_grad_by_total_norm_fp32(...)` to ALL sub-optimizer parameters. When `grad_norm` is huge (~5e7 to ~2e11 on GatedDeltaNet), clip coefficient `c = clip_grad / grad_norm ≈ 2e-8`. For Muon, this pushes per-matrix gradients below Newton-Schulz's `F.normalize(eps=1e-7)` floor → orthogonalization silently degenerates → positive-feedback stall.

**Optimizer-agnostic**: AdamW with `clip_grad=1.0` also stalls at loss ~0.5 (eps floor crushed). With `clip_grad=0`, both AdamW (loss 0.029) and Muon (loss 0.019) learn normally.

**RTX 4090 GRPO**: HIGH. Any GRPO using `dist_muon` or `ChainedOptimizer` will silently stall on models with large gradients (GatedDeltaNet, hybrid architectures).

**Cross-framework**: NVIDIA-NeMo/Emerging-Optimizers #229/#230 (Newton-Schulz degeneration). ms-swift + Megatron-Core SFT of Qwen3.5-35B-A3B.

## 2. #5395 — Skip Grad-Norm Clip for Muon (CHANGES_REQUESTED)

**Severity**: HIGH | **Status**: CHANGES_REQUESTED by ShauryaaSharma (June 18)

**Fix approach**: Per-optimizer `skip_grad_norm_clip` attribute. When `True`, `ChainedOptimizer.step()` skips `clip_grad_by_total_norm_fp32` for that sub-optimizer. `grad_norm` still computed for logging.

**4 review findings**: (1) flag propagation bug on layer_wise_optimizer, (2) flag applied to SOAP/Lion incorrectly → now gated on `isinstance(optimizer, OrthogonalizedOptimizer)`, (3) `_get_grad_norm_skip_threshold()` excludes skip-flagged subs from norm threshold, (4) `should_clip` also checks skip flag.

## 3. #5400 — Route GDN to Adam Instead of Muon

**Severity**: MEDIUM | **Status**: Draft

GDN `in_proj` packs q/k/v/conv/gate/beta into one fused matrix → Muon orthogonalizes as single matrix (semantically meaningless for heterogeneous fused weight). Fix: tag `in_proj.weight` with `skip_orthogonalization=True`.

**RTX 4090**: LOW-MEDIUM. Only affects GDN hybrid models with Muon.

## 4. #5401 — MoE Router Z-Loss + TE CUDA Graph

**Severity**: MEDIUM | **Status**: OPEN

`torch.tensor(logits.shape[0], device=logits.device)` creates CPU→CUDA copy during graph capture → rejected by PyTorch. Fix: use Python int instead of device tensor.

**RTX 4090**: LOW. Only affects MoE models with z-loss + CUDA Graph.

## 5. #5387 — FSDP fully_shard Implementation (★★★ HIGH)

**Severity**: HIGH (architectural) | **Status**: Final Review

Experimental per-module `fully_shard(...)` using DBuffer primitives. `FsdpModule`, `FsdpParameterGroup`, `Placements` runtime state. Groups params by dtype/requires_grad, manages sharded buffers, installs forward/backward hooks for unshard/reshard/gradient reduction. 27 tests passed.

**RTX 4090**: ★★★ HIGH. FSDP fully_shard = key path for large models on 24 GiB. GRPO requires policy + reference → FSDP sharding enables fitting larger models.

## 6. #5384/#5386 — DSA/DSv4 Indexer Replay (★★★ VERY HIGH for GRPO)

**Severity**: HIGH | **Status**: OPEN

Rollout inference and training may use different kernels/precision/batching → indexer top-k selections differ → train/rollout logprob mismatch → incorrect reward attribution and policy gradient. Proposed `IndexerReplay` class (RECORD, REPLAY_FORWARD, REPLAY_BACKWARD). Analogue to MoE RouterReplay.

**RTX 4090**: ★★★★★★★★ VERY HIGH. DSA indexer mismatch directly causes incorrect GRPO training signal. verl RL batch layouts specifically mentioned.

## 7. New Issues (June 17-20)

| Issue | Title | Severity | RTX 4090 Impact |
|-------|-------|----------|-----------------|
| #5396 | Fold q/k L2-norm into GDN kernel | LOW | LOW-MEDIUM (memory savings at 128K context) |
| #5391/#5388 | Decoupled compact LayerWise DDP | HIGH | HIGH (reduces Muon memory overhead) |
| #5398 | Fast-cache-load rank sync guard | LOW | NEGLIGIBLE |
| #5403 | Rename CP batch helpers | LOW | NEGLIGIBLE |
| #5405 | Remove ModelOpt loading | LOW | NEGLIGIBLE |
| #5337 | _reduce returns unreduced tensor | LOW | NEGLIGIBLE |

## 8. Release Status

- v0.17.0 (April 16): MTP, MLA pipeline, MoE router z-loss, RouterReplay, Lion optimizer, CUDA graph for Adam, SHA-256 checkpoint integrity
- v0.17.1 (May 28): Backport fixes, TE bump 2.14
- v0.18: Expected late June/early July (4-week cadence)
- No v0.18 milestone created yet

## 9. Architecture Highlights

**Optimizer system** (1784 lines): MegatronOptimizer → FP32Optimizer → MixedPrecisionOptimizer → DistributedOptimizer → LayerWiseDistributedOptimizer → ChainedOptimizer. ChainedOptimizer = where #5394 bug lives.

**Gradient clipping**: `clip_grad_by_total_norm_fp32` computes `clip_coeff = max_norm / (total_norm + 1e-6)`. When total_norm huge → clip_coeff tiny → gradients crushed.

**MoE routing**: RouterReplay singleton-registry (RECORD/REPLAY_FORWARD/REPLAY_BACKWARD) for RL training. DSA IndexerReplay follows same architecture.

## 10. RTX 4090 GRPO Impact Priority Matrix

| Priority | Issue | Impact |
|----------|-------|--------|
| ★★★★★★★★ P0 | #5384/#5386 DSA Indexer Replay | Train/rollout mismatch → incorrect GRPO signal |
| ★★★★★★★★ P0 | #5394/#5395 Muon clipping | Silent training stall |
| ★★★ P1 | #5387 FSDP fully_shard | Enables larger models on 24 GiB |
| ★★★ P1 | #5391 LayerWise DDP | Reduces Muon memory overhead |
| ★★ P2 | #5401 z-loss CUDA Graph | MoE + CUDA Graph |
| ★ P3 | #5400 GDN routing | Hybrid model Muon routing |
