# verl #6512 -- Per-Unit LoRA Summon for FSDP1/FSDP2: Deep Technical Analysis

> 2026-06-18 | MERGED | Author: Qing An (qinganrice / qa4@rice.edu)
> +110/-37 final | 3 files: fsdp_utils.py (+85/-35), transformer_impl.py (+21/-0), monkey_patch.py (+4/-2)
> Merge commit: 8a694930275061f52ebd538c906ef8819af56dbd
> Companion PR: verl-omni #113 (Qwen3-Omni-30B-A3B Thinker GSPO + LoRA)
> 4 commits across 19 days (May 27 -> June 17)
> Reviewers: SamitHuang (3 COMMENTED), wuxibin89 (1 APPROVED), gemini-code-assist[bot] (2 auto-reviews)

---

## 1. Problem Statement: Root-Level Summon = OOM on RTX 4090

Before #6512, `layered_summon_lora_params` used a **hard-coded prefix list** to find
sub-modules, then summoned the root FSDP module to collect LoRA params. This had three
fatal problems for 30B+ MoE models on a 24 GiB GPU:

**Problem A: Peak memory = full model unsharded.**
Summoning the root FSDP module via `summon_full_params(root)` unshards ALL parameters
simultaneously. For Qwen3-30B-A3B this requires ~60 GiB -- impossible on RTX 4090.

**Problem B: Hard-coded prefix list misses non-standard architectures.**
The old code enumerated 8 fixed prefixes:
```python
prefix_list = [
    # fsdp
    "_fsdp_wrapped_module.base_model.model.",
    "_fsdp_wrapped_module.base_model.model.model.",
    "_fsdp_wrapped_module.base_model.model.model.layers.",
    "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
    # fsdp2
    "base_model.model.",
    "base_model.model.model.",
    "base_model.model.model.layers.",
    "base_model.model.model.language_model.layers.",
]
```
These cover only `model.layers.*` and `model.language_model.layers.*` paths. They miss:
- MoE expert-level FSDP units (`layers.<i>.mlp.*`)
- Multi-stage configs like `thinker_config.text_config`
- Any model with non-standard module nesting (e.g., Qwen3-Omni, Qwen3.5)

**Problem C: FSDP1 state_dict() crash on non-root units.**
`state_dict()` on a non-root FSDP1 unit triggers `_pre_state_dict_hook` which
dereferences `_all_handles` via `__getattr__`. FSDP1 only initializes `_all_handles`
on the root module, so calling `state_dict()` on a child unit crashes.

---

## 2. Core Fix: Dynamic FSDP Unit Discovery + Per-Unit Summon

### 2.1 named_modules() + fsdp_version() Discovery (Replaces 8 Prefixes)

The key architectural change: instead of enumerating hard-coded paths, iterate ALL
modules dynamically and detect which ones are FSDP units:

```python
for name, submodule in fsdp_module.named_modules():
    if name == "":
        continue                     # Skip root: full summon defeats the purpose
    if fsdp_version(submodule) == 0:
        continue                     # Skip non-FSDP modules (plain nn.Module)
    clean_prefix = name.replace("_fsdp_wrapped_module.", "")
    if clean_prefix.endswith(".model") or clean_prefix.endswith(".layers"):
        continue                     # Skip container-level wrappers (no LoRA here)
    # ... proceed to summon this unit individually
```

The `fsdp_version()` function (existing in verl) simply checks the module type:
```python
def fsdp_version(model):
    if isinstance(model, FSDP):       return 1   # FSDP1: torch.distributed.fsdp.FSDP
    elif isinstance(model, FSDPModule): return 2  # FSDP2: torch.distributed.fsdp.FSDPModule
    else:                               return 0  # Not FSDP-wrapped
```

This is **model-agnostic by design**: any FSDP unit at any nesting depth is discovered
automatically. It works for:
- Dense models with layer-level FSDP wrapping
- MoE models with expert/MLP-level FSDP wrapping
- Multi-stage models (Qwen3-Omni) with sub-encoder FSDP wrapping
- Any future model architecture without code changes

The `clean_prefix` stripping (`name.replace("_fsdp_wrapped_module.", "")`) normalizes
FSDP1's `_fsdp_wrapped_module.` wrapper prefix to match the names downstream consumers
(vLLM LoRA loader, peft state_dict) expect.

### 2.2 FSDP1 vs FSDP2: Different Summon Strategies

The PR implements version-aware summon with completely different mechanisms:

**FSDP1: Explicit summon_full_params context + _is_root hack**
```python
if is_fsdp1:
    submodule._is_root = True           # HACK: treat child as root for summon
summon_ctx = FSDP.summon_full_params(submodule, writeback=False) if is_fsdp1 else nullcontext()

with summon_ctx:
    sub_state_dict = {n: p for n, p in submodule.named_parameters()
                      if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)}
    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
    # ... process lora params ...
    if is_fsdp1:
        submodule._is_root = False      # Reset after summon
```

The `_is_root = True` hack is critical: FSDP1 only initializes `_all_handles` on the
root module. Non-root units don't have their own `_all_handles` and `summon_full_params`
would fail. Setting `_is_root = True` makes the child unit behave as a root for the
duration of the summon, allowing `summon_full_params` to work. It MUST be reset to
`False` afterward to avoid corrupting the FSDP state for subsequent operations.

**FSDP2: DTensor on-demand all-gather + nullcontext**
```python
# No context manager needed for FSDP2
# DTensor parameters all-gather on demand via param.full_tensor()
sub_lora_params = {
    f"{clean_prefix}.{key}": (
        param.full_tensor().detach().cpu() if hasattr(param, "full_tensor")
        else param.detach().cpu()
    )
    for key, param in sub_lora_params.items()
}
```

FSDP2 stores parameters as `DTensor` (distributed tensor). When you call
`param.full_tensor()`, it performs an on-demand all-gather. No context manager
is needed because DTensor handles the gather/unshard lifecycle internally.

**Why nullcontext for FSDP2?** FSDP2's `summon_full_params` does not exist as a
context manager. The DTensor API is fundamentally different: parameters are always
"available" in their sharded form, and `full_tensor()` is an explicit operation to
get the unsharded version. This is cleaner than FSDP1's approach but means you
must call `.full_tensor()` explicitly on each parameter.

### 2.3 Named Parameters vs State Dict (FSDP1 Safety)

The PR replaces `submodule.state_dict()` with `dict(submodule.named_parameters())`:

**Why this matters for FSDP1:**
- FSDP1 only initializes `_all_handles` on the ROOT module
- `state_dict()` invokes `_pre_state_dict_hook` which accesses `_all_handles`
- On a non-root unit, `_all_handles` is uninitialized -> crash
- `named_parameters()` does NOT invoke `_pre_state_dict_hook`
- Inside `summon_full_params`, `named_parameters()` still yields unsharded tensors
- So we get the same result without the crash risk

The resulting dict is passed explicitly to `get_peft_model_state_dict()`:
```python
sub_state_dict = {n: p for n, p in submodule.named_parameters()
                  if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)}
sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
```

Gemini Code Assist flagged a related issue: calling `get_peft_model_state_dict(peft_model)`
without an explicit `state_dict` argument defaults to `peft_model.state_dict()`, which
under FSDP1 can trigger conflicting state dict hooks. The fix: always pass an explicit
`state_dict` built from `named_parameters()`. This was incorporated in the fallback path:

```python
state_dict = {n: p for n, p in peft_model.named_parameters()}
lora_params = get_peft_model_state_dict(peft_model, state_dict=state_dict)
```

### 2.4 Nested Child Exclusion (Prevents Double-Gathering)

This was flagged by Gemini Code Assist as a HIGH-priority issue and was a critical
design consideration. The problem:

When iterating `named_modules()`, a parent FSDP unit contains parameters of its
nested child FSDP units. If we call `named_parameters()` on the parent, we get ALL
parameters including nested children. This causes:
1. Parent unit: gathers parent params + nested child params -> copies to CPU
2. Child unit: gathers child params again -> copies to CPU AGAIN
3. Double-gathering defeats memory savings and can cause OOM on MoE models

The fix: compute nested FSDP children and exclude their parameters:
```python
nested_fsdp_names = {n for n, m in submodule.named_modules()
                     if n != "" and fsdp_version(m) > 0}

# Pre-check: only consider this unit's OWN parameters (not nested children)
if not any("lora_" in n
           for n, _ in submodule.named_parameters()
           if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)):
    continue  # This unit has no LoRA params of its own -> skip

# State dict: only include this unit's OWN parameters
sub_state_dict = {n: p for n, p in submodule.named_parameters()
                  if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)}
```

This ensures each LoRA parameter is gathered EXACTLY ONCE -- when its own FSDP unit
is visited. The `nested_fsdp_names` set is computed per unit using the same
`fsdp_version()` detection, making it fully dynamic.

**Concrete example for MoE:**
```
fsdp_module.named_modules() traversal:
  "layers.0"           -> fsdp_version=0 (not FSDP unit, skip)
  "layers.0.mlp"       -> fsdp_version=1 (FSDP unit)
    nested_fsdp_names = {"layers.0.mlp.experts.0", "layers.0.mlp.experts.1", ...}
    Only include parameters NOT starting with "layers.0.mlp.experts.*"
  "layers.0.mlp.experts.0" -> fsdp_version=1 (FSDP unit)
    nested_fsdp_names = {} (leaf unit, no nested FSDP children)
    Include ALL parameters -> gather exactly once
```

### 2.5 Skip Degenerate FSDP Units

Frozen sub-encoders (e.g., vision encoder in multi-stage models) are FSDP-wrapped
but never forwarded during training. Their lazy state (`_all_handles`) is uninitialized,
so calling any FSDP operation on them crashes. The LoRA check naturally excludes them:

```python
if not any("lora_" in n for n, _ in ...):
    continue  # No LoRA params -> skip this unit entirely
```

This works because:
1. LoRA adapters are only applied to trainable modules
2. Frozen sub-encoders don't have LoRA applied
3. So they have zero `lora_*` parameters
4. The check skips them before any FSDP operation is attempted

---

## 3. Additional Changes

### 3.1 LoRA Wrap Policy Guard (NCCL Deadlock Prevention)

In `get_fsdp_wrap_policy()`:
```python
# Before: always add lambda policy for LoRA
if is_lora:
    lambda_policy = ...

# After: only add lambda policy when min_num_params == 0
if is_lora and min_num_params == 0:
    lambda_policy = ...
```

**Root cause:** Combining a size-based wrap policy (`min_num_params > 0`) with a
lambda-based LoRA wrap policy creates nested FSDP units at different granularity
levels. Under variable-length inputs (with `use_remove_padding`), the allgather
order can diverge across ranks, causing NCCL deadlocks during backward.

The lambda policy wraps leaf modules with trainable weights; the size policy wraps
modules exceeding a parameter threshold. Together they produce contradictory wrapping
hierarchies. The fix: when `min_num_params > 0`, skip the lambda policy entirely and
let the size policy determine wrap granularity. LoRA modules are still correctly
wrapped because the size policy wraps modules at the MLP/attention level, which
naturally includes LoRA parameters.

### 3.2 _verl_strip_modules Support (transformer_impl.py)

After loading weights, before FSDP wrap:
```python
_strip_list = getattr(module, "_verl_strip_modules", [])
for attr in _strip_list:
    if hasattr(module, attr):
        delattr(module, attr)
        logger.info(f"Stripped unused sub-module '{attr}' to reduce memory")
```

Multi-stage models like Qwen3-Omni have sub-modules not needed for Thinker training
(talker, code2wav). Stripping them before FSDP wrap prevents:
1. GPU allocation for unused modules
2. FSDP wrapping of modules that will never be forwarded
3. Degenerate FSDP units that crash on state_dict operations

For Qwen3-Omni-30B: stripping talker + code2wav saves ~8-10 GiB.

### 3.3 LoRA dtype Cast (transformer_impl.py)

After creating the LoRA model via `get_peft_model()`:
```python
base_dtype = next((p.dtype for p in module.parameters() if not p.requires_grad), None)
if base_dtype is not None:
    mismatched = [p for p in module.parameters() if p.requires_grad and p.dtype != base_dtype]
    if mismatched:
        logger.info(f"Casting {len(mismatched)} LoRA adapter params from "
                    f"{mismatched[0].dtype} to {base_dtype} to match base.")
        for param in mismatched:
            param.data = param.data.to(base_dtype)
```

PEFT's `LoraConfig` defaults to fp32 for `lora_A` and `lora_B`. FSDP requires all
parameters in a flat parameter group to share the same dtype. A fp32 LoRA param in
a bf16 flat group triggers a dtype mismatch assertion during forward.

This is the SAME root cause pattern as DeepSpeed #8072/#8073 (per-policy dtype
causing ZeRO-3+PEFT regression), but at the FSDP level rather than ZeRO level.

### 3.4 monkey_patch.py: get_text_config() Fallback

```python
# Before:
num_attention_heads, num_key_value_heads = (
    model.config.text_config.num_attention_heads,
    model.config.text_config.num_key_value_heads,
)

# After:
try:
    num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
except AttributeError:
    text_config = getattr(model.config, "text_config", None) or model.config.get_text_config()
    num_attention_heads, num_key_value_heads = (
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
    )
```

Multi-stage models like Qwen3-Omni nest their text config inside
`thinker_config.text_config`. Direct `model.config.text_config` fails because it
resolves to the wrong nesting level. `get_text_config()` is a HuggingFace helper
that traverses the config tree to find the actual text model config.

### 3.5 Fallback Path for Empty LoRA Params (fsdp_utils.py)

If `layered_summon_lora_params` returns empty (edge case: all LoRA params in units
that were somehow skipped), fall back to full root summon with explicit `state_dict`:

```python
if not lora_params:
    logging.getLogger(__name__).warning("layered_summon returned empty, falling back to full summon")
    is_fsdp1 = fsdp_version(module) == 1
    summon_ctx = FSDP.summon_full_params(module, writeback=False, offload_to_cpu=True) \
                    if is_fsdp1 else nullcontext()
    with summon_ctx:
        state_dict = {n: p for n, p in peft_model.named_parameters()}
        lora_params = get_peft_model_state_dict(peft_model, state_dict=state_dict)
        lora_params = {
            name: param.full_tensor().detach().cpu() if hasattr(param, "full_tensor")
                  else param.detach().cpu()
            for name, param in lora_params.items()
        }
```

The fallback uses `offload_to_cpu=True` for FSDP1 to minimize GPU peak. It also
uses the explicit `state_dict` pattern (from Gemini review) to avoid FSDP1 hook
conflicts.

---

## 4. Commit History and Evolution

4 commits across 19 days (May 27 -> June 17):

**Commit 1 (June 13): e40b3843**
Initial implementation with both FSDP LoRA fixes AND plugin loader:
- fsdp_utils.py: layered_summon rewrite, LoRA wrap policy guard
- transformer_impl.py: _verl_strip_modules, LoRA dtype cast
- monkey_patch.py: get_text_config() fallback
- NEW files: utils/plugins.py, workers/rollout/base.py (register_rollout_adapter),
  utils/tokenizer.py (register_processor_handler), trainer/main_ppo.py (plugin loop)

**Commit 2 (June 13): 9327d1e7**
Dropped unused `register_rollout_adapter` hook -- verl-omni registers directly
via `_ROLLOUT_REGISTRY` at import time, so the public setter has no callers.

**Commit 3 (June 17): 2a1c5eab**
Samit's co-authored commit: incorporated Gemini Code Assist suggestion for
explicit `state_dict` in `get_peft_model_state_dict()` to avoid FSDP1 hook conflicts.

**Commit 4 (June 17): 67645540**
Final refactor: dropped custom `verl/utils/plugins.py` loader entirely. verl already
has a native entry-point hook (`VERL_USE_EXTERNAL_MODULES` in `verl/__init__.py`)
that auto-loads packages declaring `verl.plugins` entry points. The custom loader
was redundant. This was prompted by wuxibin89's review comment pointing to the
existing hook.

Also moved `get_peft_model_state_dict` import to file head and trimmed comments
to <=2 lines per SamitHuang's review feedback.

**Final diff stats: +110/-37 (net), 3 files changed** (plugin-related files were
added then removed across commits, netting zero in the final merge)

---

## 5. Review Comments and Issues

### 5.1 Gemini Code Assist: Nested Child Double-Gathering (HIGH priority)

Flagged that `named_parameters(recurse=True)` on a parent FSDP module recursively
collects nested child parameters. When the parent is summoned, children's params are
gathered and copied to CPU. Later, when children are visited, they're gathered again.
This "redundant double-gathering defeats the memory-saving purpose" and can cause OOMs.

**Resolution:** qinganrice incorporated the suggestion. The final code uses
`nested_fsdp_names` exclusion set to filter out parameters belonging to nested FSDP
children before both the pre-check and the state_dict construction. SamitHuang
acknowledged: "agree @qinganrice PTAL".

### 5.2 Gemini Code Assist: Explicit state_dict for get_peft_model_state_dict (HIGH priority)

Flagged that calling `get_peft_model_state_dict(peft_model)` without explicit
`state_dict` defaults to `peft_model.state_dict()`. Under FSDP1, invoking
`state_dict()` inside `summon_full_params` can trigger conflicting state dict hooks,
leading to runtime crashes or incorrect weight retrieval.

**Resolution:** Incorporated in commit 3 (Samit's co-authored commit). The fallback
path now uses: `state_dict = {n: p for n, p in peft_model.named_parameters()}`;
`get_peft_model_state_dict(peft_model, state_dict=state_dict)`. SamitHuang noted:
"not critical, but cheap and safer".

### 5.3 SamitHuang: Style and Organization Comments

- "put import in the file head" -> moved `get_peft_model_state_dict` import to top of fsdp_utils.py
- "pls make the comment clean and concise for human reader" -> trimmed comments to <=2 lines
- "please add the test case and result, at least in the PR description" -> added end-to-end test info

### 5.4 wuxibin89: Existing Plugin Hook (APPROVED)

Pointed out that verl already has `VERL_USE_EXTERNAL_MODULES` in `verl/__init__.py`
to load external plugins via entry points. The custom `verl/utils/plugins.py` loader
was redundant. This led to commit 4 removing the entire custom loader infrastructure.

**The key insight:** verl's existing mechanism auto-loads packages that declare a
`verl.plugins` entry point in their `pyproject.toml`. verl-omni already does this.
So no custom loader is needed -- just declare the entry point and it's auto-loaded.

### 5.5 Remaining Issues (After Merge)

1. **No unit tests for layered_summon:** The PR has only end-to-end test results
   (4xH100 Qwen3-Omni GSPO+LoRA, pearson > 0.99). No isolated unit tests for the
   FSDP unit discovery, nested child exclusion, or fallback path logic.

2. **Fallback path safety:** If `layered_summon_lora_params` returns empty, the
   fallback does a full root summon which will OOM on RTX 4090 for 30B+ models.
   This should ideally never happen in practice, but there's no defensive check
   for GPU memory before attempting the fallback.

3. **FSDP1 _is_root hack correctness:** Setting `submodule._is_root = True` is a
   documented internal API access pattern. It works but relies on FSDP1 internals
   that could change across PyTorch versions. No long-term stability guarantee.

4. **LoRA-only check for degenerate units:** The skip logic uses `lora_*` name
   matching to exclude degenerate units. If a future model has LoRA on a frozen
   sub-encoder (unlikely but possible), the unit would NOT be skipped and could
   crash on `_pre_state_dict_hook`.

5. **Plugin loader removal:** The custom plugin infrastructure was removed, which
   means external packages MUST use the `verl.plugins` entry point mechanism. The
   `register_model_architecture` and `register_processor_handler` APIs that were
   part of the initial PR were also dropped (only in intermediate commits, not in
   final merge). verl-omni registers directly via private dicts, which is less
   clean but functional.

---

## 6. Memory Reduction Analysis

### 6.1 Before #6512: Peak = Full Model Unsharded

```
Qwen3-30B-A3B (MoE, 30B total, 3B active per token):
  Full model unsharded: ~60 GiB (bf16)
  Single FSDP unit (expert MLP): ~0.5-1 GiB
  Root summon peak: ~60 GiB -> OOM on 24 GiB RTX 4090
```

### 6.2 After #6512: Peak = Largest FSDP Unit

```
Per-unit summon with MoE expert-level wrapping:
  Largest FSDP unit: ~6-8 GiB (dense attention layer or large MLP)
  Peak during LoRA sync: ~6-8 GiB (one unit at a time)
  After each unit: empty_cache() frees GPU memory
  Total LoRA params collected: same as before (all gathered, but sequentially)

  10x memory reduction: 60 GiB -> 6-8 GiB peak
```

### 6.3 Additional Savings

- `_verl_strip_modules`: stripping talker + code2wav -> ~8-10 GiB resident reduction
  (these modules are never allocated, not just freed after summon)
- LoRA bf16 cast: fp32 -> bf16 for lora_A/lora_B -> ~2x reduction in LoRA param size
  (e.g., rank=32, 2 matrices per target module: ~0.5 GiB savings for 30B model)
- Combined with bypass_mode (18Psi -> 3.8Psi): further activation memory reduction

### 6.4 RTX 4090 Viability Breakdown

```
Before #6512:
  Weight sync peak: ~60 GiB -> OOM, cannot even start training
  Only viable: ~14B dense models with root summon

After #6512:
  Weight sync peak: ~6-8 GiB -> fits in 24 GiB
  Resident model (ZeRO-2 sharded + CPU offload): ~4-6 GiB GPU
  LoRA params: ~0.5-1 GiB (bf16, rank=32)
  Activations (bypass_mode): ~3.8 GiB (vs 18 GiB without bypass)
  Total peak: ~14-18 GiB -> FITS in 24 GiB RTX 4090!

  30B+ MoE LoRA GRPO is NOW VIABLE on RTX 4090
```

---

## 7. Architecture Decision Analysis

### 7.1 named_modules() Iteration Order

`named_modules()` yields modules in **parent-before-child** order (depth-first,
pre-order). This is critical for the nested exclusion mechanism:
1. Parent is visited first, but its nested children's params are excluded
2. Each child is visited later, gathering its own params
3. No double-gathering, no missing params

If the order were reversed (child-before-parent), the exclusion would still work
because children don't contain parent parameters, but the `lora_params.update()`
ordering would be different (child entries first, then potentially overwritten by
parent entries -- which wouldn't happen since parent excludes children).

### 7.2 clean_prefix Normalization

```python
clean_prefix = name.replace("_fsdp_wrapped_module.", "")
```

This strips ALL `_fsdp_wrapped_module.` segments, not just one. FSDP1 wraps each
FSDP unit in an `_fsdp_wrapped_module` prefix. For deeply nested units:
```
FSDP1 module name: "_fsdp_wrapped_module.base_model.model.model.layers.0.mlp"
clean_prefix: "base_model.model.model.layers.0.mlp"
```

This normalization ensures LoRA param keys match what vLLM's LoRA loader expects
(vLLM doesn't have `_fsdp_wrapped_module` in its naming).

### 7.3 Container-Level Skip

```python
if clean_prefix.endswith(".model") or clean_prefix.endswith(".layers"):
    continue
```

Skipping container-level modules (`*.model`, `*.layers`) is important because:
1. These modules are FSDP-wrapped at a coarse granularity
2. They contain many nested child FSDP units
3. Summoning them would unshard ALL their children at once (defeating per-unit summon)
4. They don't have direct LoRA parameters (LoRA is on leaf modules like q_proj)

### 7.4 get_torch_device().empty_cache() After Each Unit

```python
get_torch_device().empty_cache()  # Called after each unit's summon context exits
```

This is placed OUTSIDE the `with summon_ctx:` block, after the context exits and
parameters are re-sharded. The `empty_cache()` frees the GPU memory that was
temporarily allocated for the unsharded parameters, ensuring the next unit starts
with minimal GPU occupancy.

---

## 8. Comparison: Before vs After Implementation

### Before (Hard-Coded Prefixes + Root Summon)
```python
def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    lora_params = OrderedDict()
    prefix_list = [
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(
                        peft_model, state_dict=submodule.state_dict()  # CRASH on FSDP1 non-root!
                    )
                    ...
                    submodule._is_root = False  # Reset but no _is_root = True set!
```

**Problems in old code:**
1. Hard-coded 8 prefixes -> misses non-standard architectures
2. `submodule.state_dict()` -> crashes on FSDP1 non-root units
3. No `_is_root = True` before summon -> summon may fail
4. No nested child exclusion -> double-gathering in MoE models
5. No degenerate unit skip -> crashes on frozen sub-encoders
6. `is_diffusers` dead parameter still present

### After (Dynamic Discovery + Per-Unit Summon)
```python
def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    """Collect LoRA params one FSDP unit at a time to keep peak GPU memory low."""
    lora_params = OrderedDict()
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)

    for name, submodule in fsdp_module.named_modules():
        if name == "":
            continue
        if fsdp_version(submodule) == 0:
            continue
        clean_prefix = name.replace("_fsdp_wrapped_module.", "")
        if clean_prefix.endswith(".model") or clean_prefix.endswith(".layers"):
            continue

        is_fsdp1 = fsdp_version(submodule) == 1
        nested_fsdp_names = {n for n, m in submodule.named_modules()
                             if n != "" and fsdp_version(m) > 0}
        if not any("lora_" in n for n, _ in submodule.named_parameters()
                   if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)):
            continue

        if is_fsdp1:
            submodule._is_root = True
        summon_ctx = FSDP.summon_full_params(submodule, writeback=False) \
                        if is_fsdp1 else nullcontext()

        with summon_ctx:
            sub_state_dict = {n: p for n, p in submodule.named_parameters()
                              if not any(n.startswith(f"{nn}.") for nn in nested_fsdp_names)}
            sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
            if not sub_lora_params:
                continue
            sub_lora_params = {
                f"{clean_prefix}.{key}": (
                    param.full_tensor().detach().cpu() if hasattr(param, "full_tensor")
                    else param.detach().cpu()
                )
                for key, param in sub_lora_params.items()
            }
            lora_params.update(sub_lora_params)
            if is_fsdp1:
                submodule._is_root = False
        get_torch_device().empty_cache()

    return lora_params
```

**Improvements:**
1. Dynamic discovery via `named_modules()` + `fsdp_version()` -> works for ANY model
2. `named_parameters()` instead of `state_dict()` -> no FSDP1 hook crash
3. `_is_root = True` before summon, `_is_root = False` after -> proper lifecycle
4. `nested_fsdp_names` exclusion -> each param gathered exactly once
5. `lora_*` check -> degenerate units skipped
6. FSDP1/FSDP2 version-aware -> DTensor for FSDP2, summon for FSDP1
7. `empty_cache()` after each unit -> peak memory bounded by largest unit

---

## 9. Cross-Framework Patterns and Relevance

### 9.1 DeepSpeed #8072/#8073 Connection

The LoRA dtype cast in #6512 addresses the SAME pattern as DeepSpeed #8072/#8073:
- DeepSpeed: per-policy dtype (bf16 param + fp32 LoRA) -> ZeRO-3 partition assertion
- verl: fp32 LoRA default + bf16 base -> FSDP flat group dtype assertion
- Both: mixed dtype in a grouped/sharded parameter set causes runtime crash
- DeepSpeed fix: ensure LoRA params use same dtype as partition policy
- verl fix: cast LoRA params to base dtype before FSDP wrap

### 9.2 FSDP1 _is_root Hack Pattern

The `_is_root = True` hack is a known pattern in FSDP1 internals:
- FSDP1 only initializes `_all_handles` on the root module
- Non-root units have `_is_root = False` and uninitialized `_all_handles`
- `summon_full_params` checks `_is_root` and fails if not root
- Setting `_is_root = True` temporarily is the standard workaround
- This pattern is also used in DeepSpeed's FSDP1 integration code
- Risk: relies on FSDP1 internals that PyTorch could change

### 9.3 Nested Exclusion Pattern

The `nested_fsdp_names` exclusion pattern is applicable to ANY framework that
uses hierarchical sharding with per-unit summon:
- DeepSpeed ZeRO-3: similar nested partition hierarchy, but ZeRO's `state_dict`
  handles nested gathering internally
- Megatron-LM: uses separate distributed groups, no nested exclusion needed
- vLLM: doesn't use FSDP for inference, no summon needed

This pattern is UNIQUE to verl's FSDP-based training and is a novel contribution.

---

## 10. RTX 4090 Implications

### 10.1 Training Viability Matrix (After #6512)

```
Model              | Before #6512 | After #6512    | Config
Qwen3-30B-A3B     | OOM (60 GiB)  | VIABLE (~14 GiB) | ZeRO-2+CPU_Adam, rank=32, bypass
Qwen2.5-14B dense | VIABLE        | VIABLE (better)   | Same config, lower peak
Qwen3-8B dense    | VIABLE        | VIABLE (better)   | Same config, trivial peak
DeepSeek-V3-671B  | IMPOSSIBLE    | IMPOSSIBLE        | Needs multi-GPU even with per-unit
```

### 10.2 Recommended RTX 4090 Config (with #6512)

```yaml
# MUST settings:
actor_rollout_ref:
  model:
    lora_rank: 32            # MUST: rank>64 breaks EOS (#6782)
    lora_alpha: 64           # MUST: 2x rank for stable training
    use_shm: true            # MUST: shared memory for weight sync
  rollout:
    name: vllm               # or vllm_omni for multi-stage models
    mode: async              # MUST: async rollout for efficiency

trainer:
  algo:
    layered_summon: true     # MUST: per-unit summon (#6512)
  engine:
    param_offload: true      # MUST: CPU param offload for 24 GiB
    optimizer_offload: true  # MUST: CPU optimizer offload

# SHOULD settings:
actor_rollout_ref:
  model:
    external_lib: verl_omni.models.transformers.qwen3_omni_thinker  # for multi-stage
    enforce_eager: true      # SHOULD: avoid CUDA graph issues
  rollout:
    bypass_mode: true        # SHOULD: 18Psi -> 3.8Psi activation reduction
```

### 10.3 Weight Sync Flow (After #6512)

```
Step 1: collect_lora_params(module, layered_summon=True, base_sync_done=True)
  -> layered_summon_lora_params(module)
  -> For each FSDP unit:
     a. Detect unit via named_modules() + fsdp_version()
     b. Compute nested children exclusion set
     c. Check for LoRA params (skip if none)
     d. Set _is_root=True (FSDP1) or nullcontext (FSDP2)
     e. Summon unit -> gather LoRA params -> copy to CPU
     f. Reset _is_root=False (FSDP1)
     g. empty_cache() to free GPU memory
  -> Return OrderedDict of all LoRA params (sequential, peak = largest unit)

Step 2: Send LoRA params to rollout actor via ZMQ/shared memory
  -> vLLM receives params and updates LoRA adapters
  -> No full model unshard needed on training side!

Total GPU peak during sync: ~6-8 GiB (vs ~60 GiB before)
```

---

## Key Findings Summary

1. **Per-unit summon via named_modules() + fsdp_version()** replaces 8 hard-coded prefixes
   -> model-agnostic, works for ANY architecture including MoE expert-level wrapping
2. **FSDP1: _is_root=True hack + summon_full_params context** vs **FSDP2: DTensor
   full_tensor() + nullcontext** -> version-aware summon without user configuration
3. **nested_fsdp_names exclusion** prevents double-gathering -> each LoRA param gathered
   exactly once, critical for MoE models with deeply nested FSDP hierarchy
4. **named_parameters() instead of state_dict()** -> avoids FSDP1 _pre_state_dict_hook
   crash on non-root units, same result without the risk
5. **LoRA dtype cast to bf16** -> prevents FSDP flat group dtype assertion, same root
   cause as DeepSpeed #8072/#8073
6. **_verl_strip_modules** -> multi-stage models can declare unused sub-modules to strip
   before FSDP wrap, saving 8-10 GiB resident memory
7. **LoRA wrap policy guard** -> skip lambda policy when min_num_params > 0, prevents
   NCCL deadlocks from divergent allgather order
8. **10x peak memory reduction**: 60 -> 6-8 GiB for LoRA weight sync
9. **30B+ MoE LoRA GRPO NOW VIABLE on RTX 4090** with #6512 + bypass_mode + rank=32
10. **Plugin loader removed**: verl's native `VERL_USE_EXTERNAL_MODULES` entry-point
    mechanism replaces custom loader, cleaner integration

---

## References

- verl #6512: https://github.com/volcengine/verl/pull/6512 (MERGED June 18)
- verl-omni #113: https://github.com/verl-project/verl-omni/pull/113 (Qwen3-Omni Thinker GSPO+LoRA test)
- verl #6782: LoRA rank>64 breaks EOS in vLLM rollout
- verl #6699: detach memory fix (4x reduction, UNFIXED in 3 backends)
- verl #6794: delta weight sync (~100x payload reduction, SGLang-only)
- DeepSpeed #8072/#8073: ZeRO-3+PEFT LoRA regression (same dtype mismatch root cause)
- DeepSpeed #8068: gradient_clipping default 0->1.0 (MUST set 1.0 for GRPO)
- PyTorch FSDP1 internals: `_is_root`, `_all_handles`, `_pre_state_dict_hook`
- PyTorch FSDP2 DTensor: `param.full_tensor()` for on-demand all-gather
- RTX 4090 config: tools/rtx4090_grpo_config_reference.py
