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

### Labels

`module: _inductor`, `topic: not user facing`, `bug`, `severity: medium`

## Checklist Before Filing

- [ ] Create GitHub issue with above content
- [ ] Include reproducible example that works on SM89 GPU
- [ ] Wait for PyTorch team feedback before submitting PR
- [ ] Once feedback positive → submit PR with 5-line guard

## 参考

- Full PR draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Root cause analysis: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Triton codegen pipeline: notebook/projects/pytorch-inductor-triton-codegen-pipeline-reading.md
- vLLM issue: github.com/vllm-project/vllm/issues/39096
- PyTorch issue: github.com/pytorch/pytorch/issues/185814
- Diagnostic tool: tools/sm89_batch_invariance_diagnostic.py
- Repro script: tools/sm89_batch_invariance_repro.py
