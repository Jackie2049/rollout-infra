# DeepSpeed ZeRO-3 + PEFT LoRA Regression (#8072/#8073/#8076) — Source Code Deep Reading

> 2026-06-19 | Deep reading: exact bug mechanism, source code walk-through, fix analysis, cross-framework pattern
> Files: partition_parameters.py (3 all-gather variants), engine.py (#8066 change), stage3.py (call site)

---

## 1. Bug Mechanism — Exact Code Path and Dtype Mismatch Chain

### 1.1 Trigger: _post_step Calls all_gather on persistent_parameters

The crash occurs at the end of every optimizer step. In `stage3.py` line 2391-2392:

```python
# deepspeed/runtime/zero/stage3.py, _post_step()
if len(self.persistent_parameters) > 0:
    self.persistent_parameters[0].all_gather(self.persistent_parameters)
```

`persistent_parameters` is a list of ZeRO-3 parameters that remain replicated throughout training (only partitioned before the optimizer step). After the optimizer step completes, they must be all-gathered back so the model can use them for the next forward pass.

### 1.2 Dispatch: _all_gather -> _allgather_params_coalesced

The `all_gather` bound method (defined on each `param` in `partition_parameters.py` line 1231) calls `self._all_gather(param_list)` (line 1606). Inside `_all_gather`:

```python
# partition_parameters.py line 1636
self._allgather_params_coalesced(all_gather_nonquantize_list, hierarchy, quantize=False)
```

The list `all_gather_nonquantize_list` contains ALL non-quantized persistent parameters, which now includes mixed-dtype parameters (bf16 base model + fp32 LoRA).

### 1.3 Bug Site: _allgather_params_coalesced Line 1927

The exact bug is in `partition_parameters.py` at line 1925-1928 (before #8073 fix):

```python
# BUGGY CODE (before #8073):
for psize in partition_sizes:
    tensor_size = psize * self.num_partitions
    flat_tensor = torch.empty(tensor_size, dtype=param_list[0].ds_tensor.dtype,    # <-- BUG
                              device=self.local_device).view(-1)
    flat_tensor.requires_grad = False
    allgather_params.append(flat_tensor)
```

The loop iterates over `partition_sizes` (one per parameter), but allocates ALL output buffers using `param_list[0].ds_tensor.dtype` — the dtype of the FIRST parameter. This assumes uniform dtype across all parameters in the list.

### 1.4 The Dtype Mismatch Chain

When PEFT LoRA is used with `autocast_adapter_dtype=True`:

```
persistent_parameters list:
  [0]: base_model.weight (ds_tensor.dtype = torch.bfloat16)
  [1]: base_model.bias   (ds_tensor.dtype = torch.bfloat16)
  [2]: lora_A.weight     (ds_tensor.dtype = torch.float32)    <-- PEFT autocast
  [3]: lora_B.weight     (ds_tensor.dtype = torch.float32)    <-- PEFT autocast
```

Output buffer allocation:
```
  allgather_params[0]: dtype=bfloat16 (from param_list[0])  -- CORRECT for param[0]
  allgather_params[1]: dtype=bfloat16 (from param_list[0])  -- CORRECT for param[1]
  allgather_params[2]: dtype=bfloat16 (from param_list[0])  -- WRONG for param[2] (fp32)
  allgather_params[3]: dtype=bfloat16 (from param_list[0])  -- WRONG for param[3] (fp32)
```

The all-gather launch (line 1948):
```python
h = dist.all_gather_into_tensor(allgather_params[param_idx],    # output: bf16 buffer
                                input_tensor,                   # input: fp32 tensor
                                group=self.get_partition_dp_group(param),
                                async_op=True)
```

When `param_idx=2` (LoRA param), `allgather_params[2]` is bf16 but `input_tensor` is fp32. PyTorch's `all_gather_into_tensor` (which wraps `group._allgather_base`) requires `output_tensor.dtype == input_tensor.dtype`. The mismatch raises:

```
TypeError: output tensor must have the same type as input tensor
```

This is a HARD CRASH on the FIRST optimizer step — no silent corruption, no partial success.

### 1.5 Quantize Path Has Same Bug

The quantize scale allocation (line 1934-1935) also uses `param_list[0].ds_tensor.ds_quant_scale.dtype` for ALL buffers:

```python
for psize in quantize_scale_sizes:
    tensor_size = psize * self.num_partitions
    flat_tensor = torch.empty(tensor_size,
                              dtype=param_list[0].ds_tensor.ds_quant_scale.dtype,   # <-- SAME BUG pattern
                              device=self.local_device).view(-1)
```

However, #8073 does NOT fix the quantize path. This is a potential incompleteness (see Section 3).

---

## 2. Root Cause — How #8066 Exposed the Latent Bug

### 2.1 Before #8066 (DeepSpeed v0.19.1): The Accidental Normalizer

In `engine.py`, `_configure_distributed_model` (pre-0.19.2):

```python
# BEFORE (v0.19.1):
elif self.bfloat16_enabled():
    if is_zero_init_model:
        self.__check_params(self.module, torch.bfloat16)
    self.module.bfloat16()  # <-- BLANKET CAST: normalizes ALL params to bf16
```

`self.module.bfloat16()` is `nn.Module.bfloat16()` — it recursively casts ALL parameters and buffers to bf16. This means PEFT LoRA parameters (originally fp32 from autocast_adapter_dtype) were accidentally downcast to bf16. The effect:

```
Before module.bfloat16():
  base_model.weight: fp32 (loaded from checkpoint)
  lora_A.weight:     fp32 (PEFT autocast)

After module.bfloat16():
  base_model.weight: bf16 (cast)
  lora_A.weight:     bf16 (cast -- PEFT's fp32 intent was overwritten!)

Then ZeRO-Init partitions:
  base_model.weight: ds_tensor.dtype = bf16
  lora_A.weight:     ds_tensor.dtype = bf16  (UNIFORM!)
```

The uniform dtype meant `param_list[0].ds_tensor.dtype` happened to work for ALL parameters. The bug in `_allgather_params_coalesced` was latent — it always worked because the blanket cast enforced uniformity as a side effect.

### 2.2 After #8066 (DeepSpeed v0.19.2): Per-Policy Dtype Cast

PR #8066 ("Mixed-precision: per-policy param/buffer dtype cast (preserve fp32 buffers)"), merged June 16 by tjruwase (commit b919284a), replaced the blanket cast:

```python
# AFTER (v0.19.2):
if self.fp16_enabled() or self.bfloat16_enabled():
    check_dtype = torch.half if self.fp16_enabled() else torch.bfloat16
    if is_zero_init_model:
        self.__check_params(self.module, check_dtype)
    # Cast params only; preserve fp32 buffers (e.g. rotary inv_freq)
    # unless buffer_dtype is set. Replaces blanket module.half()/bfloat16().
    param_dtype, buffer_dtype = self._mixed_precision_dtypes()
    self._cast_module_mixed_precision(param_dtype, buffer_dtype, is_zero_init_model)
```

The new `_cast_module_mixed_precision` method:

```python
def _cast_module_mixed_precision(self, param_dtype, buffer_dtype, is_zero_init_model):
    # ZeRO-Init params are already at the configured dtype and partitioned, so
    # the per-parameter cast applies only in the non-zero-init path.
    if param_dtype is not None and not is_zero_init_model:   # <-- SKIPPED for ZeRO-Init
        for p in self.module.parameters(recurse=True):
            if p.is_floating_point() and p.dtype != param_dtype:
                p.data = p.data.to(param_dtype)

    # Buffers are never ZeRO-partitioned.
    if buffer_dtype is not None:
        for b in self.module.buffers(recurse=True):
            if b.is_floating_point() and b.dtype != buffer_dtype:
                b.data = b.data.to(buffer_dtype)
```

Two critical effects:

1. **ZeRO-Init models skip the param cast entirely** (`not is_zero_init_model` gate). ZeRO-Init models are partitioned at initialization — their ds_tensor.dtype is already set during Init. PEFT LoRA params that were added AFTER ZeRO-Init retain their original fp32 dtype.

2. **Buffers are preserved by default** (buffer_dtype=None by default). This is the INTENDED improvement — fp32 buffers like `inv_freq` (RoPE frequencies) are no longer accidentally downcast to bf16. This was the legitimate motivation for #8066.

The net effect on PEFT LoRA:
```
After #8066 (no blanket cast):
  base_model.weight: bf16 (from ZeRO-Init partitioning)
  lora_A.weight:     fp32 (from PEFT autocast_adapter_dtype=True, NOT cast)

persistent_parameters = MIXED DTYPES (bf16 + fp32)
param_list[0].ds_tensor.dtype = bf16  (first param)
param_list[2].ds_tensor.dtype = fp32  (LoRA param)
→ param_list[0].ds_tensor.dtype used for ALL buffers → MISMATCH
```

### 2.3 #8066's Intent Was Correct

The change was motivated by a real problem: `module.bfloat16()` was over-casting fp32 buffers. For example, RoPE `inv_freq` buffers (used for rotary position embeddings) are mathematically defined in fp32 and should remain fp32 for numerical accuracy at long context lengths. The blanket cast destroyed this precision. The #8066 fix correctly preserved buffer dtypes.

The issue is that removing the blanket cast revealed a LATENT assumption in `_allgather_params_coalesced` that was never documented or enforced — it simply happened to work because the blanket cast made all dtypes uniform.

---

## 3. Fix Analysis — Is #8073 Correct? Is It Complete?

### 3.1 The #8073 Fix (2-line change)

```python
# BEFORE (buggy):
for psize in partition_sizes:
    tensor_size = psize * self.num_partitions
    flat_tensor = torch.empty(tensor_size, dtype=param_list[0].ds_tensor.dtype, ...)
    allgather_params.append(flat_tensor)

# AFTER (#8073 fix):
for i, psize in enumerate(partition_sizes):
    tensor_size = psize * self.num_partitions
    flat_tensor = torch.empty(tensor_size, dtype=param_list[i].ds_tensor.dtype, ...)
    allgather_params.append(flat_tensor)
```

**Is it correct?** YES. Each parameter's output buffer now uses that parameter's own dtype. The `all_gather_into_tensor` call at line 1948 will have matching output/input dtypes for every parameter.

### 3.2 Incompleteness: The Quantize Path

The quantize scale buffer allocation (line 1934-1935) has the SAME `param_list[0]` pattern:

```python
for psize in quantize_scale_sizes:
    tensor_size = psize * self.num_partitions
    flat_tensor = torch.empty(tensor_size,
                              dtype=param_list[0].ds_tensor.ds_quant_scale.dtype,   # <-- NOT fixed
                              device=self.local_device).view(-1)
```

#8073 does NOT fix this line. If quantized ZeRO-3 parameters also have mixed dtypes, the quantize path would have the same TypeError. However, this is likely a lower-priority incompleteness because:
- Quantized parameters typically use int8 for the tensor and fp32 for the scale — these are separate all_gather calls
- The quantize path is less commonly used than the non-quantize path
- The PEFT LoRA scenario (the actual bug trigger) uses non-quantized parameters

Still, the quantize path should be fixed for completeness. A complete fix would also change `param_list[0]` to `param_list[i]` in the quantize scale allocation.

### 3.3 Comparison With Other All-Gather Variants — Why Only This Method Has the Bug

There are FOUR distinct all-gather implementations in partition_parameters.py, and the bug only exists in ONE of them:

| Method | Location | Dtype Handling | Bug? |
|---|---|---|---|
| `_allgather_params_sequential` | Line 2002 | `param.ds_tensor.dtype` (per-param) | NO — already correct |
| `_allgather_params_coalesced` | Line 1925 | `param_list[0].ds_tensor.dtype` (first-param) | YES — the bug site |
| `_all_gather_sequential` (inner) | Line 1275 | `get_allgather_dtype(param, param_ds_tensor)` (per-param) | NO — already correct |
| `_all_gather_coalesced` (inner) | Line 1348 | Groups by dtype with `dtype_params = defaultdict(list)` then calls `_all_gather_dtype` per group | NO — already correct |

The inner `_all_gather_coalesced` (line 1348) is the MOST sophisticated:

```python
# Inner _all_gather_coalesced — handles mixed dtypes by grouping
if not quantize:
    dtype_params = defaultdict(list)
    for p in params:
        allgather_dtype = get_allgather_dtype(p, p.ds_tensor)
        dtype_params[allgather_dtype].append(p)
    handles = []
    for dtype in sort_dtypes(dtype_params.keys()):
        handles.append(
            _all_gather_dtype(dtype_params[dtype], world_size, rank_in_group, ds_process_group, dtype))
    return MultipleAllGatherHandles(handles)
```

This groups parameters by dtype, then all-gathers each group separately. It even has an assertion in `_all_gather_dtype`:

```python
# _all_gather_dtype (line 1233):
def _all_gather_dtype(params, world_size, rank_in_group, ds_process_group, allgather_dtype):
    dtype = params[0].dtype
    assert all(p.dtype == dtype for p in params), "all params must have the same dtype"
```

The inner `_all_gather_coalesced` also uses `get_only_unique_item` for the all_reduce path (line 1354):

```python
dtype=get_only_unique_item(p.ds_tensor.dtype for p in params)
```

`get_only_unique_item` RAISES RuntimeError if there are multiple unique dtypes — an explicit assertion that prevents the latent assumption.

The `_allgather_params_coalesced` (the buggy class method) is the ONLY variant that uses `param_list[0].ds_tensor.dtype` without any grouping, assertion, or per-param indexing. This is clearly a code quality oversight — the other three variants all handle mixed dtypes properly.

### 3.4 Why the Bug Only Manifests With persistent_parameters

The `_allgather_params_coalesced` method is only called from `_all_gather` (line 1636), which is triggered via `param.all_gather(param_list)`. This is used for:
- `persistent_parameters` all-gather after optimizer step (the crash site)
- Regular parameter all-gather during forward/backward pass hooks

For regular forward/backward hooks, parameters are typically all-gathered one module at a time (via `allgather_before` decorator). Parameters within a single module usually share the same dtype, so `param_list[0].ds_tensor.dtype` happens to work.

For `persistent_parameters`, the list contains ALL persistent parameters across the entire model — including base model params AND PEFT LoRA params. This is where mixed dtypes appear, because LoRA adapters have different dtype from the base model.

### 3.5 Assessment: Is #8073 Complete?

**The fix is correct for the reported bug, but NOT fully complete.**

Missing:
1. Quantize scale buffer allocation (line 1934-1935) — same `param_list[0]` pattern
2. No dtype grouping or assertion (unlike the inner `_all_gather_coalesced` which groups by dtype)

A more robust fix would either:
- (Option A) Use per-param dtype for ALL buffer allocations (including quantize) — minimal, like #8073 but extended
- (Option B) Group parameters by dtype before all-gathering, like the inner `_all_gather_coalesced` — more complex but more robust

For practical purposes, #8073's fix is sufficient because:
- The non-quantize path is the one that crashes with PEFT LoRA
- The quantize path is rarely used with PEFT
- Adding dtype grouping would be a larger change with more review burden

**Recommended approach**: Merge #8073 now (fixes the immediate crash), then follow up with a completeness PR for the quantize path.

---

## 4. RTX 4090 Implications

### 4.1 Direct Impact: ZERO Under Standard Config

ZeRO-2 + CPU_Adam on RTX 4090 (dp=1) is completely unaffected:
- ZeRO-2 does NOT partition model parameters — no `_allgather_params_coalesced` call
- ZeRO-2 does NOT have `persistent_parameters` concept
- The dtype mismatch only arises in ZeRO-3 parameter sharding/reassembly

### 4.2 What Happens If Someone Uses ZeRO-3 on RTX 4090

ZeRO-3 on single GPU (dp=1) was ALREADY not viable for multiple reasons:
- Pure overhead: all-gather/reduce-scatter self-communication with no sharding benefit
- Higher memory: ZeRO-3 partitioning overhead + persistent_parameters replication
- Known regressions: 3 confirmed v0.19.2 bugs (#8072/#8076, #8075, #8068)

With this regression, ZeRO-3 + PEFT LoRA on RTX 4090:
- **Immediate hard crash** on first optimizer step
- Not a silent failure — explicit TypeError, no ambiguity
- Combined with ZeRO-3's inherent single-GPU overhead, this makes ZeRO-3 even more impractical

### 4.3 Workaround Options

| Workaround | Config Change | Effectiveness | Trade-off |
|---|---|---|---|
| Use ZeRO-2 | `zero_stage: 2` | 100% | Already optimal for single GPU |
| autocast_adapter_dtype=False | PEFT config | 100% | LoRA stays in bf16 (slight stability risk) |
| Pin DeepSpeed v0.19.1 | pip install | 100% | Loses #8066 buffer preservation |
| Apply #8073 patch manually | 2-line change | 100% | Requires manual source patching |

**Best option for RTX 4090**: ZeRO-2 + CPU_Adam (already standard config, avoids ALL ZeRO-3 bugs).

---

## 5. Cross-Framework Pattern — Dtype Mismatch in Collective Ops

### 5.1 Pattern Definition

**Collective Dtype Mismatch**: When a distributed collective operation (all-gather, reduce-scatter, all-reduce) assumes uniform dtype across all participants, but one or more participants have a different dtype. The mismatch manifests as:
- Hard crash: TypeError/RuntimeError when output buffer dtype != input tensor dtype
- Silent corruption: If the collective "succeeds" but with wrong dtype reassembly (numerical garbage)
- State lifecycle mismatch: If dtype-dependent state (caches, buffers) is not invalidated at dtype transition boundaries

### 5.2 Pattern Members Across Frameworks

| Framework | Bug | Manifestation | Root Cause | Severity |
|---|---|---|---|---|
| **DeepSpeed** | #8072/#8073 | Hard crash (TypeError) | `param_list[0].ds_tensor.dtype` assumes uniform dtype | CRITICAL (crash) |
| **DeepSpeed** | #8061 | NaN corruption | overlap_comm multi-stream race on mixed-precision buckets | CRITICAL (silent) |
| **DeepSpeed** | #8058 | Silent optimizer update loss | contiguous() bug — non-contiguous tensors in ZenFlow copyback | HIGH (silent) |
| **SGLang** | #28676 | 64x accuracy blowup | MXFP8 MoE shuffle cache not invalidated at weight-reload boundary | CRITICAL (silent) |
| **vLLM-Ascend** | #10684 | ALL-ZERO Hadamard output | DSA Hadamard class variable lost during sleep/wake transfer | CRITICAL (silent) |
| **vLLM** | #46118 | 58% request failure | MTP+grammar FSM conflict — state lifecycle mismatch | CRITICAL (hard error) |
| **vLLM** | #44395/#44483 | Illegal memory access | wake_up(tags=["weights"]) + forward when KV cache still asleep | CRITICAL (crash) |
| **Megatron** | #5317 | NaN at iter 2 | Triton in-place rotary kernel bypasses autograd version counter | CRITICAL (silent) |

### 5.3 Pattern Taxonomy

The dtype mismatch pattern has 3 sub-patterns:

**A. Output Buffer Dtype Assumption** (DeepSpeed #8072)
- Collective op allocates output buffer using one participant's dtype
- Fails when participants have mixed dtypes
- Fix: Use per-participant dtype or group by dtype

**B. State Cache Not Reset at Dtype Boundary** (SGLang #28676, vLLM-Ascend #10684)
- GPU-resident cache assumes a specific dtype/layout
- Weight reload changes dtype/layout but cache is not invalidated
- Fix: Invalidate/reset cache at weight-reload boundary

**C. Mixed-Precision Collective Race** (DeepSpeed #8061)
- Different precision buckets processed on different CUDA streams
- Synchronization gap allows reading before all streams complete
- Fix: Stream synchronization or single-stream reduction

### 5.4 Defense Stack (Updated)

| Layer | Defense | Applicable To |
|---|---|---|
| **L1: Compile-time** | dtype grouping in collective ops (like inner `_all_gather_coalesced`) | A |
| **L2: Runtime assertion** | `assert all(p.dtype == dtype for p in params)` before collective | A, B |
| **L3: NaN/Inf detection** | NanDetectMode (PyTorch #187653) — catches silent corruption AFTER collective | B, C |
| **L4: Boundary reset** | Cache/buffer invalidation at weight-reload, sleep/wake, dtype-change boundaries | B, C |

For the #8072 bug specifically, L1 (dtype grouping) or a simpler L1 variant (per-param dtype indexing) would prevent the crash. #8073 implements the simpler L1 variant.

### 5.5 MUST DO / MUST NOT Rules (Updated)

**MUST DO**:
- MUST use per-param dtype for all-gather output buffer allocation (not param_list[0])
- MUST group parameters by dtype before coalesced all-gather (inner `_all_gather_coalesced` pattern)
- MUST invalidate GPU-resident caches at weight-reload boundaries (SGLang #28676 lesson)
- MUST add dtype assertion in collective ops that assume uniform dtype
- MUST use ZeRO-2 on single GPU (dp=1) RTX 4090 — ZeRO-3 has 3+ confirmed regressions

**MUST NOT**:
- MUST NOT assume all parameters in a coalesced all-gather share the same dtype
- MUST NOT use blanket module.half()/bfloat16() as a substitute for dtype-aware collective ops
- MUST NOT use ZeRO-3 on single GPU RTX 4090 (pure overhead + regression risk)
- MUST NOT mix fp32 LoRA adapters with bf16 base model in ZeRO-3 without per-param dtype fix
- MUST NOT skip dtype assertion in collective ops that are "working" — latent bugs hide under blanket casts

---

## 6. DeepSpeed v0.19.2 Regression Cluster

Three confirmed regressions in v0.19.2, all with trivial fixes, all with ZERO reviews:

| Regression | Issue | Root Cause | Fix | Lines | Reviews |
|---|---|---|---|---|---|
| ZeRO-3+PEFT dtype mismatch | #8072, #8076 | #8066 per-policy dtype | #8073 | +2/-2 | 0 |
| FD leak in async I/O | #8075 | Missing close() call | #8075 | +1/-1 | 0 |
| Gradient clipping default change | #8068 | Default 0->1.0 | Unknown | ? | 0 |

Pattern: v0.19.2 changes (#8066 and others) removed long-standing "accidental normalizers" (blanket casts, implicit defaults). This exposed latent assumptions that were never documented. The fixes are trivial, but the DeepSpeed maintainer pipeline is unable to review them promptly.

---

## 7. PR #8073 Review Status

- **State**: OPEN
- **Author**: albertvillanova (Albert Villanova del Moral) — also the #8072 reporter
- **Reviewers requested**: tjruwase (Olatunji Ruwase), tohtana (Masahiro Tanaka)
- **Actual reviews**: 0 (none completed)
- **Inline comments**: 0
- **Issue comments on #8072**: 0
- **Issue comments on #8076**: 0

The fix has been open for ~2 days with no review engagement. Both requested reviewers have not responded. This matches the pattern of other stalled PRs (#8075, #8068).

---

## 8. Source Code Reference Map

| File | Lines | Content | Bug/Relevance |
|---|---|---|---|
| partition_parameters.py | 1925-1928 | `_allgather_params_coalesced` output buffer allocation | **BUG SITE** — `param_list[0].ds_tensor.dtype` |
| partition_parameters.py | 1934-1935 | Quantize scale buffer allocation | **SAME BUG pattern** — `param_list[0]`, NOT fixed by #8073 |
| partition_parameters.py | 1948 | `dist.all_gather_into_tensor` launch | **CRASH SITE** — dtype mismatch raises TypeError |
| partition_parameters.py | 2002-2010 | `_allgather_params_sequential` | Per-param dtype — already correct |
| partition_parameters.py | 1233-1246 | `_all_gather_dtype` (inner) | Per-group dtype with assertion — already correct |
| partition_parameters.py | 1348-1387 | `_all_gather_coalesced` (inner) | Groups by dtype with defaultdict — already correct |
| partition_parameters.py | 1354 | `get_only_unique_item` for all_reduce path | Raises RuntimeError if multiple dtypes — already correct |
| partition_parameters.py | 53-55 | `get_allgather_dtype` helper | Uses `get_comm_dtype` if available, else `param_ds_tensor.dtype` |
| engine.py | 1569-1580 | `_configure_distributed_model` (v0.19.2) | Per-policy dtype cast — **ROOT CAUSE** (#8066) |
| engine.py | 1322+ | `_cast_module_mixed_precision` | Skips param cast for ZeRO-Init models |
| stage3.py | 2391-2392 | `_post_step` persistent_parameters all_gather | **TRIGGER** — calls buggy method |

---

## 9. Key Takeaways

**Bug mechanism**: `_allgather_params_coalesced` uses `param_list[0].ds_tensor.dtype` for ALL output buffers, but PEFT LoRA parameters have fp32 ds_tensor while base model parameters have bf16. The all-gather output buffer dtype must match the input tensor dtype, and bf16 != fp32 raises TypeError.

**Root cause**: #8066 (per-policy dtype cast, MERGED June 16) correctly stopped the blanket `module.bfloat16()` cast that was over-casting fp32 buffers. This was the RIGHT change. However, it removed an "accidental normalizer" that was making all persistent_parameters share the same dtype, exposing the latent `param_list[0]` assumption in `_allgather_params_coalesced`.

**Fix assessment**: #8073's 2-line change (`param_list[0]` -> `param_list[i]`) is correct for the non-quantize path. The quantize path (line 1934-1935) has the same `param_list[0]` pattern but is NOT fixed — this is an incompleteness. A more robust fix would group parameters by dtype (like the inner `_all_gather_coalesced`), but that's a larger change. Recommend: merge #8073 now, follow up with quantize fix.

**RTX 4090**: ZeRO-2 + CPU_Adam (standard config) completely avoids this bug. ZeRO-3 on single GPU was already not viable — this regression makes it crash instead of just being slow. No change to RTX 4090 recommended config.

**Cross-framework pattern**: "Collective Dtype Mismatch" — 3 sub-patterns (output buffer assumption, cache not reset at boundary, mixed-precision stream race). DeepSpeed #8072 is sub-pattern A. SGLang #28676 and vLLM-Ascend #10684 are sub-pattern B. DeepSpeed #8061 is sub-pattern C. All share the theme: an implicit assumption about dtype/state uniformity that is violated when the system transitions between configurations.

---

*End of deep reading. Generated 2026-06-19.*
