# Cross-Framework Attention NaN Pattern: Running Max = -inf

## The Pattern

A numerical stability bug in Triton attention kernels: when the running max is
initialized to `-inf` AND the first chunk of a row is fully masked (all padding),
the online softmax produces `nan` permanently.

### Mechanism

```
e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # = -inf
qk    = tl.where(mask, qk, -float("inf"))                      # all -inf if fully masked
n_e_max = tl.maximum(tl.max(qk, 1), e_max)                    # max(-inf, -inf) = -inf
re_scale = tl.exp2(e_max - n_e_max)                           # exp2(-inf - -inf) = exp2(nan) = nan
p = tl.exp2(qk - n_e_max[:, None])                             # nan
acc *= re_scale                                                # nan × 0 = nan ← PERMANENT
```

Once NaN enters the accumulator, no later valid chunk can rescue it (`nan × 0 = nan`).

### The Math Root

The online softmax algorithm (Milakov & Gimelshein 2018) initializes:
- `m0 = -inf` (running max)
- `d0 = 0` (running denominator)

When the first chunk's max is also `-inf`:
- `m1 = max(-inf, -inf) = -inf`
- `d1 = 0 * exp(-inf - (-inf)) + sum(exp(-inf - (-inf))) = 0 * exp(0) + N * exp(0) = N`

Wait — actually the online softmax rescaling factor is `exp(m_old - m_new) = exp(-inf - -inf) = exp(0) = 1`, not `nan`.

The issue is in the **Triton implementation** using `exp2` and `e_max - n_e_max`:
- `e_max - n_e_max = -inf - (-inf) = nan` (infinity arithmetic: same sign infinity subtraction = nan)
- This is different from the mathematical `exp(m_old - m_new)` where `-inf - (-inf)` would be computed differently

So the actual bug class is:
**`inf - inf = nan` in IEEE 754 floating point, specifically `-inf - (-inf) = nan`.**

## Known Instances

### Instance 1: vLLM #48364 — xpu_mla_sparse (2026-07-11)
- **File**: `vllm/v1/attention/ops/xpu_mla_sparse.py`
- **Kernel**: `_bf16_mla_sparse_kernel`
- **Trigger**: First BLOCK_N (=16) topk index entries of a row are all masked (-1 or >= seq_kv)
- **Fix**: Use finite sentinel instead of `-inf`
- **Already fixed in**: Newer kernel from #47629
- **Affects**: XPU_MLA_SPARSE backend forward, DSA prefill, fp8 decode wrapper
- **Test gap**: Test writes valid indices at front, padding at back → never triggers

### Instance 2: vLLM #47629 — New sparse-MLA kernel (2026-06)
- **Fix direction**: Uses finite sentinel instead of infinite running max
- **Predates**: The bug in #48364 (the newer kernel avoids the pattern)

### Instance 3: General flash attention pattern
- **Many implementations** initialize `m = -inf` and process chunks
- **Safe when**: Every row has at least one valid key (common in autoregressive decoding)
- **Vulnerable when**: Full rows of padding exist (sparse attention, variable-length batches with padding)

## Why Tests Don't Catch It

The standard test pattern places valid indices at the front of each row:
```python
indices = torch.full((T, TOPK), -1)  # padding
indices[:, :VALID_K] = valid_k       # valid at front, padding at back
```

This means the first chunk always has valid keys → `e_max` becomes finite → trailing
padding chunks are safe (finite - (-inf) = inf, but exp2 scales to 0).

To reproduce, place **all valid indices after the first chunk boundary**:
```python
indices[0, 0, 16] = 0  # token 0: valid key only after first BLOCK_N=16 chunk
```

## Defense

1. **Initialization**: Use finite sentinel (e.g., `-1e38` or `float("-1e20")`) instead of `-inf`
2. **Chunk-level guard**: Check if current chunk has any valid key before computing stats
3. **Row-level guard**: Pre-scan each row for valid keys, return zeros for empty rows
4. **Output sanitization**: Replace NaN outputs with zeros after kernel (band-aid, not fix)

## References
- vLLM #48364: https://github.com/vllm-project/vllm/issues/48364
- vLLM #47629: Fix with finite sentinel
- Online softmax: Milakov & Gimelshein 2018, "Online normalizer calculation for softmax"
- IEEE 754: `inf - inf = nan`, `(-inf) - (-inf) = nan`

