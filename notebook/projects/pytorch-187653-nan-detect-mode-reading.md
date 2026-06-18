# PyTorch #187653 NanDetectMode — Forward-Pass NaN/Inf Detection

> 2026-06-18 | RTX 4090 GRPO debugging tool | +127 LOC, OPEN
> ★★★★★★★★ Forward-pass NaN detection → complements detect_anomaly (backward-only)
> ★★★★★★★★ GRPO relevance: locate NaN's FIRST appearance in model forward → pinpoint root cause

---

## What NanDetectMode Does

```python
from torch.utils.nan_detect import NanDetectMode

with NanDetectMode():
    out = model(x)
# RuntimeError: Function aten.add.Tensor returned NaN values
```

- `NanDetectMode(TorchDispatchMode)` wraps every torch operation
- After each op: flatten outputs via pytree → check each float tensor for NaN
- `check_inf=True` option → also detects ±Inf (off by default)
- Skips non-float tensors, empty tensors, meta/fake tensors

---

## Why This Matters for RTX 4090 GRPO

```
★★★★★★★★★ GRPO NaN debugging timeline:

Before NanDetectMode:
  → NaN appears in reward/advantage → WHERE did it originate?
  → torch.autograd.detect_anomaly() → only catches NaN in BACKWARD
  → Backward NaN = derivative of forward NaN → but WHICH forward op?
  → Manual search: add torch.isnan() checks throughout model → tedious!

After NanDetectMode:
  → Wrap model forward in NanDetectMode → IMMEDIATE NaN detection
  → Error message: "Function aten.add.Tensor returned NaN values"
  → → Know EXACTLY which op first produced NaN → root cause pinpointed!

★★★★★★★★★ Common RTX 4090 GRPO NaN sources:
  → SM89 batch invariance → different numerical results per batch → accumulation
  → LoRA rank mismatch → wrong weight shapes → NaN in matmul
  → ZeRO-2 overlap_comm → multi-stream race → NaN in gradient
  → FSDP2 CPU leak → stale parameters → NaN in forward
  → DSV4 DSA indexer → wrong positions → NaN in attention
```

---

## Implementation Details

```
★★★★★★★★★ Key design choices:

1. TorchDispatchMode (not anomaly detection):
  → TorchDispatchMode intercepts ALL torch ops → forward pass
  → detect_anomaly intercepts autograd ops → backward pass
  → → Complementary! Use both for full NaN tracing

2. Pytree flattening:
  → Handles complex output types (dicts, lists, nested tuples)
  → Only checks float tensors → skips int/bool/meta

3. check_inf option:
  → Inf detection useful for: overflow detection, MoE router logits overflow
  → Default off → some models intentionally use Inf (e.g., attention masks)
  → Opt-in per user's needs

4. Exception context:
  → RuntimeError with function name → clear error message
  → Context restored on exception → no state corruption
```

---

## References

- PyTorch #187653: NanDetectMode PR
- PyTorch #160016: original issue requesting forward NaN detection
- albanD/subclass_zoo: original nan_detect.py prototype
- verl #6468: FSDP2 CPU memory leak → NaN potential source
- DeepSpeed #8061: overlap_comm NaN → production example
