# PyTorch Issue Draft — SM<90 Batch-Dependent Numerical Results from Inductor Reduction Fusion

> ★★★★★★★★ PRE-STEP before submitting PR: file issue first
> ★★★★★★★★ This issue documents the root cause and proposes a 5-line fix
> ★★★★★★★★ Target: pytorch/pytorch GitHub Issues

## Issue Title

`[Inductor] Vertical reduction fusion produces batch-dependent numerical results on SM<90 GPUs`

## Issue Body Draft

### Problem

On GPUs with compute capability < 9.0 (SM89/RTX 4090, SM86/RTX 3090, SM80/A100-40GB, etc.), `torch.compile` models containing RMSNorm (Llama, Mistral, Qwen3, etc.) produce **batch-dependent numerical results** — the same model with the same inputs yields different output values depending on the batch size.

This is the root cause of vLLM issue #39096 (batch invariance bug on consumer GPUs).

### Reproducible Example

```python
import torch

class SimpleRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # rms_norm = x * rsqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_inv_rms = variance.rsqrt()
        return x * self.weight * x_inv_rms

model = SimpleRMSNorm(4096).cuda()
compiled_model = torch.compile(model)

# Single batch
x1 = torch.randn(1, 4096, device='cuda')
result_batch1 = compiled_model(x1)

# Batch of 8
x8 = x1.repeat(8, 1)
result_batch8 = compiled_model(x8)

# Compare first element
diff = (result_batch1[0] - result_batch8[0]).abs().max().item()
print(f"Max difference: {diff}")  # Expected: ~0, Actual: varies (0.01-0.1+)
# On SM90+ GPUs: diff ≈ 0 (batch-invariant)
# On SM89/SM86/SM80: diff > 0 (batch-dependent!)
```

### Root Cause

Triton's `CachingAutotuner` selects different XBLOCK sizes for different input shapes on SM<90 GPUs due to varying shared memory availability (SM89=100KB, SM80=164KB, SM90=228KB).

When Inductor's scheduler fuses a reduction (`mean`) vertically with pointwise operations (`pow2 + sum + divide + rsqrt + mul` for RMSNorm), the reduction becomes an inline `tl.sum()` whose accumulation order depends on the autotuned XBLOCK. Because floating-point addition is non-associative, different XBLOCK → different accumulation order → different numerical results per batch size.

**Three-layer fusion gate architecture** (choices.py):
- Layer 0: `V.choices.can_fuse` (common heuristic) — shared
- Layer 1: `Scheduler.can_fuse_vertical` (structural legality) — hard rules
- **Layer 2: `V.choices.can_fuse_vertical` (profitability heuristic) — CURRENTLY RETURNS TRUE UNCONDITIONALLY!**
- Layer 3: `Backend.can_fuse_vertical` (tiling legality) — Triton checks

The unconditional `True` at Layer 2 means ANY vertical reduction fusion passes, regardless of SM capability or batch invariance implications.

### Additional Evidence (June 2026 Update)

5. **PyTorch #187435**: `no_fuse_region` per-op fusion barrier — `mark_no_fuse_region(graph, [ops])` → `annotations["no_fuse_region"]` → scheduler requires exact region-set equality for fusion. Complementary fine-grained mechanism to P9's global approach. P9 covers ALL ops universally; #187435 gives per-op control. Both needed eventually, but P9 first = simplest, fastest path.

6. **verl #6572** (OPEN, June 2026): 5-layer full determinism for vLLM rollout — PRODUCTION VALIDATES VLLM_BATCH_INVARIANT=1 as the mechanism for batch-invariant inference. Key finding: on SM89, vLLM's batch_invariant.py does NOT override RMSNorm at aten level (only SM80 matmul Triton overrides). verl's determinism layer REQUIRES a working batch-invariant backend → P9 fills the SM89 gap that vLLM leaves.

7. **vLLM batch_invariant.py source analysis** (984 lines): SM89 only gets CUBLASLt workspace config for matmuls — no Triton overrides. RMSNorm `_rms_norm_kernel` (lines 775-881) uses `tl.constexpr` BLOCK_SIZE but is NOT registered as aten override on SM89. This means vLLM's batch_invariant on RTX 4090 still has Inductor's RMSNorm fusion problem → P9 is REQUIRED for full determinism.

8. **SGLang #24459** (MERGED May 6): Added `aten::rms_norm` + `aten::mm.dtype` overrides → KERNEL-level now has MORE overrides than originally counted (7 → 9+). This STRENGTHENS the KERNEL-level position and proves that aten overrides work — P9's guard makes these overrides more effective by preventing Inductor from bypassing them.

### Evidence

1. **vLLM #39096**: Multiple reports of batch-dependent results on SM89/SM86 with `torch.compile`
2. **PyTorch #185814**: XBLOCK derivation for RMSNorm backward — same architectural issue
3. **YM2132 test (2026-04-17)**: Qwen3-1.7B passes on SM86 when reduction stays as separate kernel, but fails when Inductor fuses it → model-size-dependent!
4. **PR #187275** (2026-06-14): "Fix Combo Kernel Crash with Dynamic Persistent Reduction Dimensions" — confirms same root cause class (persistent reduction dimension handling)
5. **SGLang's approach**: 7 aten overrides with `tl.constexpr` BLOCK_SIZE → KERNEL-level batch invariance → bypasses Inductor entirely

### Proposed Solution

Add a 5-line guard in `InductorChoices.can_fuse_vertical` (choices.py line 640-647) that prevents vertical reduction fusion on CUDA devices with compute capability < 9.0:

```python
@staticmethod
def can_fuse_vertical(scheduler, node1, node2, shared_data_score) -> bool:
    """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
    # SM<90 Fusion Guard: prevent batch-dependent results from autotuned XBLOCK
    if node1.is_reduction() or node2.is_reduction():
        device = node1.get_device() or node2.get_device()
        if device is not None and device.type == "cuda":
            props = DeviceProperties.create(device)
            if props.major is not None and props.major < 9:
                WhyNoFuse(node1, node2)(
                    "SM<90 prevents reduction fusion (batch invariance)"
                )
                return False
    return True
```

**Why Layer 2 (not Layer 3)?**
- `can_fuse_vertical` hook is intentionally empty (designed for users to fill)
- Same pattern as `reduction_split_factor` which already uses `props.major >= 10` in the same class
- No additional imports needed — `DeviceProperties` and `WhyNoFuse` already imported

**5 existing precedents** for SM-capability checks in Inductor:
- `choices.py` lines 482-506: `DeviceProperties.create(device)` + `props.major >= 10` (primary precedent)
- `triton.py` line 2849: `torch.cuda.get_device_capability()[0] >= 9`
- `triton.py` line 4129: same pattern
- `triton_heuristics.py` lines 722-723: `device_prop.major >= 8`
- `triton_heuristics.py` line 4223: `device_major >= 10`

### Impact

- **SM<90**: Vertical reduction+pointwise fusions blocked → reductions as separate kernels → `torch.mean` batch-invariant overrides work correctly
- **SM90+**: No change — `props.major >= 9` → pass guard → TMA/WGMMA deterministic
- **XPU/CPU**: No change — `device.type != "cuda"` → skip guard
- **Pure pointwise**: No change — no reduction → skip guard
- **Horizontal fusions**: No change — separate method

### Environment

- PyTorch version: 2.12+
- GPU: RTX 4090 (SM89), RTX 3090 (SM86), A100-40GB (SM80), etc.
- OS: Any
- Python: 3.10+

### Related Issues

- vLLM #39096 — Batch invariance bug on SM89 (this issue addresses root cause)
- PyTorch #185814 — XBLOCK derivation for RMSNorm backward (complementary)
- PyTorch #187275 — Combo kernel crash (same root cause class)
- PyTorch #187435 — `no_fuse_region` per-op fusion barrier (complementary mechanism)
- verl #6572 — 5-layer full determinism validates VLLM_BATCH_INVARIANT=1 production use
- SGLang #24459 — aten::rms_norm + mm.dtype override strengthens KERNEL-level
- vLLM batch_invariant.py — SM89 RMSNorm gap (not registered as aten override)

### Labels

`module: _inductor`, `topic: not user facing`, `bug`, `severity: medium`

## Checklist Before Filing

- [ ] Create GitHub issue with above content
- [ ] Include reproducible example that works on SM89 GPU
- [ ] Wait for PyTorch team feedback before submitting PR
- [ ] Once feedback positive → submit PR with 5-line guard

### Integration Path with Complementary Mechanisms

**P9 is the simplest first step, but two complementary mechanisms strengthen the long-term solution:**

1. **P9 (this issue) — GLOBAL SM<90 policy**: 5 lines, blocks ALL reduction fusions on SM<90 universally. Zero config, zero per-op maintenance. Works immediately with existing vLLM/verl/VLLM_BATCH_INVARIANT. When SM90+ gets deterministic TMA/WGMMA → guard passes → no regression.

2. **PyTorch #187435 — PER-OP `no_fuse_region`**: Fine-grained control, ops annotated individually. Useful for SM90+ where some fusions ARE safe but others aren't. Also useful for expert users who want to selectively block fusions without disabling all reductions. 804 LOC → larger change → can follow P9 as Phase 2.

3. **verl #6572 — DEPLOYMENT layer**: VLLM_BATCH_INVARIANT=1 + deterministic scheduling + RM max_num_seqs=1 serialization. Validates that batch-invariant inference IS production-viable for GRPO. On SM89, requires P9 (or vLLM fixing their RMSNorm gap) to achieve full determinism.

**Complete SM89 deterministic GRPO stack** (all 5 components needed):
- P9 Inductor Fusion Guard (5 LOC) → prevents reduction fusions
- VLLM_BATCH_INVARIANT=1 (env var) → enables vLLM aten overrides
- verl #6572 5-layer determinism → production deployment
- SGLang KERNEL-level overrides → gold standard baseline
- Triton constexpr BLOCK_SIZE → deterministic accumulation order

## 参考

- Full PR draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Root cause analysis: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Triton codegen pipeline: notebook/projects/pytorch-inductor-triton-codegen-pipeline-reading.md
- vLLM issue: github.com/vllm-project/vllm/issues/39096
- PyTorch issue: github.com/pytorch/pytorch/issues/185814
- Diagnostic tool: tools/sm89_batch_invariance_diagnostic.py
- Repro script: tools/sm89_batch_invariance_repro.py
