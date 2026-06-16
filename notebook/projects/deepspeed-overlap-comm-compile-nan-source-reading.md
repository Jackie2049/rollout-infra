# DeepSpeed ZeRO overlap_comm + torch.compile NaN Bug — Source-Level Analysis

> 2026-06-16 | Issue #8061 (CRITICAL) | RTX 4090 impact: HIGH
> Root cause: multi-stream gradient copy_ → average_tensor reads incomplete data → NaN from step 1

---

## 1. Bug Description

ZeRO Stage 1/2 with `overlap_comm=True` and `torch.compile` produces NaN gradients from step 1.

**Affected configs**: ANY ZeRO-1/2 config with `overlap_comm=True` when using `torch.compile` (DeepCompile or `torch.compile(model)`).

**Manifestation**: All gradients become NaN immediately → training unusable.

---

## 2. Root Cause — Source-Level

### Key assumption violated: single CUDA stream

DeepSpeed's overlap_comm design assumes gradient operations happen on a single CUDA stream. The entire pipeline:

1. Backward pass produces gradients on `current_stream`
2. `copy_()` moves gradients into contiguous IPG buffer on `current_stream`
3. `average_tensor()` waits `current_stream` then operates on `reduction_stream`

This works when `current_stream` is consistent — all writes happen on the same stream.

### torch.compile breaks the single-stream assumption

When `torch.compile` is enabled, **compiled autograd** dispatches `copy_()` operations across **multiple CUDA streams**. The compiled autograd engine (engine.py line 2829):

```python
with compiled_autograd(self._is_compiled_autograd_enabled, self._compile_kwargs):
```

This creates a fused backward graph that may schedule gradient copy operations on different streams for different parameters. Now:

- Stream A writes gradient slice A to IPG buffer
- Stream B writes gradient slice B to IPG buffer
- Stream C (current_stream at the time `average_tensor` is called) waits only for itself

### The critical code path (stage_1_and_2.py:1230-1237)

```python
def average_tensor(self, tensor, communication_data_type):
    if self.overlap_comm:
        stream = self.reduction_stream
        if not get_accelerator().resolves_data_dependency():
            stream.wait_stream(get_accelerator().current_stream())      # ← BUG! Only waits current_stream
            get_accelerator().current_stream().wait_stream(stream)      # ← Not ALL producer streams
    else:
        stream = get_accelerator().current_stream()
```

**The bug**: `stream.wait_stream(get_accelerator().current_stream())` only synchronizes with whichever stream happens to be "current" at the moment `average_tensor` is called. It does NOT wait for ALL streams that wrote gradient data into the IPG bucket.

**Result**: `average_tensor()` reads from the IPG buffer before streams A and B have completed their writes → incomplete/corrupted data → NaN.

---

## 3. The Full Pipeline (with overlap_comm=True + torch.compile)

### Normal (no torch.compile) — SAFE

```
Backward on current_stream:
  grad_1 computed → copy_(grad_1, ipg_buffer[0:param_1.numel]) on current_stream
  grad_2 computed → copy_(grad_2, ipg_buffer[param_1.numel:total]) on current_stream

average_tensor:
  reduction_stream.wait_stream(current_stream)   ← waits ALL writes (they're all on current_stream)
  reduce_scatter on reduction_stream              ← reads complete data ✓
```

### With torch.compile — UNSAFE

```
Backward with compiled autograd:
  grad_1 computed on stream_A → copy_() dispatched on stream_A → writes ipg_buffer[0:param_1.numel]
  grad_2 computed on stream_B → copy_() dispatched on stream_B → writes ipg_buffer[param_1.numel:total]

average_tensor:
  current_stream = stream_C (whatever autograd decides)
  reduction_stream.wait_stream(stream_C)         ← only waits stream_C!
  reduce_scatter on reduction_stream              ← reads incomplete data ✗ (stream_A, stream_B may not be done)
  → NaN!
```

---

## 4. IPG Bucket Structure

```python
@dataclass
class IPGBucket:
    buffer: List[torch.Tensor]  # contiguous gradient buffer
    params: List[torch.Tensor]  # which params are in this bucket
    grads: List[torch.Tensor]   # individual gradient tensors
    elements: int = 0           # total elements in buffer
    index: int = 0              # buffer index (0 or 1, swapped for overlap)
    has_moe_params: bool = False
```

When `overlap_comm=True` and `contiguous_gradients=True`, the buffer index is swapped (line 1099-1101):

```python
if self.contiguous_gradients and self.overlap_comm:
    bucket.index = 1 - bucket.index  # double-buffering for overlap
```

This double-buffering allows the next bucket to accumulate while the current bucket is being reduced. But with torch.compile, the writes to the NEW buffer index may happen on different streams than the read.

---

## 5. Reduction Stream Lifecycle

```python
# Initialization (line 516)
self.reduction_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()
```

The `reduction_stream` is a dedicated CUDA stream for gradient reduction operations. It's only used when `overlap_comm=True`. Its purpose is to overlap gradient reduction with backward computation — but on single GPU (dp=1), this overlap provides **zero benefit** because there's no cross-GPU communication to overlap with.

---

## 6. resolves_data_dependency Check

```python
if not get_accelerator().resolves_data_dependency():
    stream.wait_stream(get_accelerator().current_stream())
```

`resolves_data_dependency()` returns `True` for devices that automatically track data dependencies across streams (like some CPU accelerators). For CUDA, it returns `False` → explicit `wait_stream()` is needed. But this `wait_stream()` only synchronizes with `current_stream`, not all streams.

---

## 7. Proposed Fix Direction

The fix needs to track ALL streams that write to IPG buckets:

```python
# BEFORE (buggy):
stream.wait_stream(get_accelerator().current_stream())

# AFTER (proposed fix):
# Record all streams that issued copy_() to IPG bucket
# During copy_grad_in_bucket or reduce_independent_p_g_buckets_and_remove_grads:
#   record_stream = get_accelerator().current_stream() at time of each copy_()
#   self.ipg_producer_streams.append(record_stream)

# During average_tensor:
for producer_stream in self.ipg_producer_streams:
    stream.wait_stream(producer_stream)
self.ipg_producer_streams.clear()
```

This ensures `reduction_stream` waits for ALL producer streams before reading the IPG buffer.

---

## 8. RTX 4090 Impact & Workaround

### Impact: HIGH

RTX 4090 users commonly use:
- ZeRO-2 + CPU_Adam + torch.compile → for training efficiency
- LoRAOptimizedLinear + torch.compile → for LoRA fine-tuning
- Muon optimizer + torch.compile → for experimental training

ALL of these are affected by the #8061 bug if `overlap_comm=True`.

### Workaround: overlap_comm=False on single GPU

On single GPU (dp_world_size=1):

1. **overlap_comm=False has ZERO throughput penalty** because there's no cross-GPU reduction to overlap
2. **overlap_comm=False eliminates the NaN risk entirely**
3. **Both safer AND equally performant** on single GPU

```json
"zero_optimization": {
    "stage": 2,
    "overlap_comm": false,  // MUST be false on single GPU
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "all_contiguous_gradients": true
}
```

### Why overlap_comm is meaningless on single GPU

- `reduce_scatter` with dp=1 is an identity operation (no sharding)
- `all_reduce` with dp=1 is a no-op
- The "overlap" is between gradient computation and reduction of the SAME GPU's gradients
- This overlap provides no benefit when both happen on the same GPU

---

## 9. Related Code Paths

| File | Line | Function | Role |
|------|------|----------|------|
| stage_1_and_2.py | 1230-1237 | `average_tensor()` | Bug location: only waits current_stream |
| stage_1_and_2.py | 516 | `__init__` | Creates `reduction_stream` |
| stage_1_and_2.py | 1091-1102 | `reduce_independent_p_g_buckets_and_remove_grads` | IPG bucket accumulation + buffer swap |
| stage_1_and_2.py | 1551-1577 | `reduce_ipg_grads` | Calls `average_tensor()` on full bucket |
| stage_1_and_2.py | 884-892 | `independent_gradient_partition_epilogue` | Final reduction after backward |
| stage_1_and_2.py | 1186-1192 | `allreduce_and_copy_with_multiple_ranks` | `record_stream(reduction_stream)` for allreduce output |
| engine.py | 460 | `__init__` | `_is_compiled_autograd_enabled = False` |
| engine.py | 2829 | `backward` | `with compiled_autograd(...)` context — multi-stream dispatch |
| engine.py | 4852-4859 | `compile` | Sets `_is_compiled_autograd_enabled` |

---

## 10. Safety Checker Tool

`tools/deepspeed_zero_safety_checker.py` — 7 checks for RTX 4090 ZeRO configs:
1. **overlap_comm + torch.compile NaN (#8061)** — CRITICAL
2. **gradient_clipping default 0→1.0 (#8068)** — STABILITY
3. **ZeRO-3 single GPU overhead** — CONFIG
4. **LoRAOptimizedLinear compatibility** — CONFIG
5. **contiguous_gradients + overlap_comm interaction** — WARNING
6. **bf16/fp16 selection** — CONFIG (SM89)
7. **Muon optimizer experimental** — EXPERIMENTAL

Modes: `check` (validate config), `generate` (produce safe config), `explain` (detailed check info)

Usage:
```bash
python3 tools/deepspeed_zero_safety_checker.py --mode check --config configs/muon_lora_zero2_rtx4090.json
python3 tools/deepspeed_zero_safety_checker.py --mode generate --scenario lora-grpo --model qwen3-1.7b
python3 tools/deepspeed_zero_safety_checker.py --mode explain --check-name overlap_comm_compile_nan
```

---

## Key Findings Summary

★★★★★★★★★ CRITICAL BUG: overlap_comm + torch.compile = NaN from step 1
★★★★★★★★★ Root cause: average_tensor() only waits current_stream, not ALL producer streams
★★★★★★★★★ RTX 4090 workaround: overlap_comm=False (zero throughput penalty on single GPU, safer)
★★★★★★★ Muon config FIXED: overlap_comm changed from true→false in configs/muon_lora_zero2_rtx4090.json
★★★★★★★ New tool: deepspeed_zero_safety_checker.py — 7 RTX 4090 config checks
★★★★★★★★★ Proposed upstream fix: record IPG producer streams per bucket → reduction_stream waits all

---

## References

- DeepSpeed Issue #8061: overlap_comm + torch.compile NaN bug
- DeepSpeed stage_1_and_2.py: ZeRO-1/2 gradient reduction pipeline
- DeepSpeed engine.py: compiled autograd integration
- tools/deepspeed_zero_safety_checker.py: RTX 4090 ZeRO config safety checker
