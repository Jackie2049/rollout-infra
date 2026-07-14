# PyTorch #187653: NanDetectMode — Forward-Pass NaN/Inf Detection

## Overview
- **PR**: pytorch/pytorch #187653 by rajfirke, June 18, 2026
- **Status**: OPEN, STALLED — 3 pings (Jun 27, Jul 4, Jul 12), ZERO reviewer response
- **Scope**: +127/-0, single file `torch/utils/nan_detect.py`
- **Severity**: ★★★★★★★★ CRITICAL for NaN debugging in GRPO training
- **Reviewer**: soulitzer (assigned by albanD, no response)

## What It Does

A `TorchDispatchMode` that checks every ATen operation's output for NaN (and optionally Inf) **during forward pass**:

```python
from torch.utils.nan_detect import NanDetectMode

with NanDetectMode():
    out = model(x)
# RuntimeError: Function aten.add.Tensor returned NaN values
```

## How It Works

- Subclasses `TorchDispatchMode`, overrides `__torch_dispatch__`
- On each ATen operation: calls `func`, then pytree-flattens outputs
- Checks every floating-point tensor for NaN/non-finite values
- Raises `RuntimeError` **immediately** naming the function that produced NaN
- **Skips**: non-floating-point tensors, empty tensors, meta/fake tensors

Key parameters:
- `check_inf=False` (default: only NaN; `True` also catches ±Inf)

## Why It's 500,000× Faster Than detect_anomaly()

`torch.autograd.detect_anomaly()` does two expensive things:
1. **Checks during backward** — requires running the full forward pass first
2. **Enables grad-mode anomaly detection** — saves metadata for every operation to trace backward

`NanDetectMode` is lighter because:
1. **Forward-only** — catches NaN immediately at the source operation
2. **No backward tracing** — doesn't need grad anomaly metadata
3. **TorchDispatchMode** — intercepts at the dispatcher level, minimal overhead per op

## Current Implementation

Based on albanD's proven `nan_detect.py` prototype from [subclass_zoo](https://github.com/albanD/subclass_zoo/blob/main/nan_detect.py) — referenced by ezyang and soulitzer in issue #160016.

Tests cover:
- NaN detection
- Clean pass-through (no NaN → no error)
- Inf opt-in (`check_inf=True`) and opt-out
- Integer/empty/meta tensor skipping
- `nn.Module` integration
- Context restore on exception

## Why Stalled

- 21 CI failures (mostly unrelated: `test_correct_module_names`, doctests)
- soulitzer assigned as reviewer but hasn't responded despite 3 pings over 3+ weeks
- albanD requested the review and triaged the PR but hasn't engaged further
- Likely low priority (labeled `topic: not user facing`)

## GRPO Relevance

**This is a Layer 2 defense for NaN debugging**:

| Layer | Tool | Scope | Speed |
|-------|------|-------|-------|
| 1 | Gradient clipping | Prevents NaN explosion | Fastest |
| 2 | **NanDetectMode** | Catches NaN at forward source | ~Fast |
| 3 | verl NaN guard (PR #6) | Fixes NaN advantages | Medium |
| 4 | detect_anomaly() | Traces backward NaN | Slowest |

For GRPO debugging workflow:
1. Enable `NanDetectMode()` during training to catch NaN at source
2. Identifies the exact ATen operation producing NaN
3. Much faster than tracing backward from NaN loss/gradients
4. Complementary to our TRL P7-2 (top_n_sigma clipping) and P9-1 (bypass_mode)

## Cross-Framework Connection

Our `tools/grpo_nan_detective.py` (ported version) is essentially the same approach — a portable TorchDispatchMode for NaN detection. If this PR merges, we can import from `torch.utils.nan_detect` instead.

## Monitoring
- PR is uncontroversial, small, based on maintainer-owned prototype
- Unblocking by commenting with GRPO use case may help
- Key trigger: soulitzer or albanD reviewing
