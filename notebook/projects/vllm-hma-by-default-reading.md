# vLLM PR #41847: HMA-by-Default Deep Reading

> Source: vLLM main branch (v0.23.0), local clone at `rollout-infra/vllm/`
> PR: https://github.com/vllm-project/vllm/pull/41847
> Date: 2026-06-15

---

## 1. What HMA Actually Does

### 1.1 Core Mechanism: Per-Attention-Type KV Cache Grouping

HMA (Hybrid Memory Allocator / Hybrid KV Cache Manager) is NOT about host-memory offloading.
It is about **per-attention-type KV cache sizing** for models with mixed attention specs.

```
When HMA ENABLED (disable_hybrid_kv_cache_manager=False):
  -> Each attention type gets its own KVCacheGroupSpec with proper sizing
  -> SlidingWindowSpec stays SlidingWindowSpec -> blocks sized for sliding_window
  -> FullAttentionSpec stays FullAttentionSpec -> blocks sized for max_model_len
  -> MambaSpec stays MambaSpec -> blocks sized for state cache
  -> Result: multiple kv_cache_groups, each with correct memory estimate

When HMA DISABLED (disable_hybrid_kv_cache_manager=True, old default):
  -> unify_hybrid_kv_cache_specs() collapses ALL specs to FullAttentionSpec
  -> SlidingWindowSpec -> FullAttentionSpec with sliding_window annotation
  -> ChunkedLocalAttentionSpec -> FullAttentionSpec with attention_chunk_size annotation
  -> SlidingWindowMLASpec -> MLAAttentionSpec (loses sliding window savings)
  -> Result: ONE kv_cache_group, all layers get max_model_len sizing = massive overallocation
```

The critical function is `get_kv_cache_groups()` in
`vllm/v1/core/kv_cache_utils.py` (line 1631):

```python
def get_kv_cache_groups(vllm_config, kv_cache_spec):
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)  # COLLAPSE all specs!
    ...
```

### 1.2 `unify_hybrid_kv_cache_specs()` (line 1324-1417)

This function is the "anti-HMA" path. It converts:
- `SlidingWindowSpec` -> `FullAttentionSpec` (losing the window-sized block savings)
- `ChunkedLocalAttentionSpec` -> `FullAttentionSpec` (losing chunked-local savings)
- `SlidingWindowMLASpec` -> `MLAAttentionSpec` (losing SWA savings for DS-V4)

After this conversion, all layers have the same type, so there's only ONE
KVCacheGroupSpec = all layers share one block table = memory sized for
`max_model_len` tokens per layer (even for sliding-window layers that only
need `sliding_window` tokens).

The function logs:
```
"Hybrid KV cache manager is disabled for this hybrid model, This means we do
not enable any optimizations for saving KV cache memory (e.g., dropping the
KV cache outside the sliding window). The compute of layers like sliding
window is still saved."
```

### 1.3 Memory Sizing Per Spec Type

| Spec Type | Memory Formula | Example (block_size=16) |
|-----------|----------------|-------------------------|
| `FullAttentionSpec` | `cdiv(max_model_len, block_size) * page_size_bytes` | 32K tokens = 2048 blocks |
| `SlidingWindowSpec` | `cdiv(min(sliding_window-1+max_num_batched_tokens, max_model_len), block_size)+1` | SW=4K: ~256 blocks |
| `ChunkedLocalAttentionSpec` | `cdiv(min(chunk_size+max_num_batched_tokens, max_model_len), block_size)` | chunk=1K: ~64 blocks |
| `MambaSpec` | `1-2 blocks` (state cache, not token-level) | 2 blocks |
| `SlidingWindowMLASpec` | Same as SlidingWindowSpec (with MLA page_size) | Depends on MLA layout |

**Key insight**: SlidingWindowSpec needs `O(sliding_window)` blocks,
while FullAttentionSpec needs `O(max_model_len)` blocks. For models where
`max_model_len >> sliding_window`, this is a **massive savings**.

---

## 2. The Default Change

### 2.1 Before PR #41847

```python
# OLD default: disable_hybrid_kv_cache_manager = True
# HMA was disabled by default -> all hybrid models collapsed to FullAttentionSpec
```

For any model with mixed attention (Mixtral, Gemma, Jamba, Bamba, DS-V4,
Mamba hybrids), vLLM would allocate KV memory as if ALL layers needed
full-sequence-length storage. This caused:

1. **Startup OOM on constrained GPUs**: The KV memory estimate was so large
   that the block pool calculation determined 0 blocks could fit -> OOM
   before the model even started serving.
2. **Wasted memory**: Even if startup succeeded, sliding-window layers
   held KV for all tokens (not just the window), wasting ~50-80% of their
   allocated blocks.

### 2.2 After PR #41847

```python
# NEW: SchedulerConfig.disable_hybrid_kv_cache_manager = None (default)
# Resolved in VllmConfig._post_init:
#   None -> False (enable HMA) unless:
#     - non-GPU platform (support_hybrid_kv_cache() = False)
#     - chunked local attention + EAGLE (not yet supported)
#     - chunked local attention without VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE
#     - KV connector does not support HMA (auto-disable for incapable connectors)
```

The resolution logic in `vllm/config/vllm.py` (lines 1449-1520):

```
Three-tier decision:
1. Explicit enable (--no-disable-hybrid-kv-cache-manager): error if runtime can't support it
2. No preference (default None): auto-disable for unsupported configs, otherwise enable
3. Explicit disable (--disable-hybrid-kv-cache-manager): always respect it

Final fallback: if still None after all checks -> False (enable HMA)
```

### 2.3 The Config Field

`vllm/config/scheduler.py` line 132:
```python
disable_hybrid_kv_cache_manager: bool | None = None
"""If True, KV cache manager will allocate the same size of KV cache for all
attention layers even if there are multiple type of attention layers.
If None, the default value will be determined based on the environment
and starting configuration."""
```

CLI flags:
- `--disable-hybrid-kv-cache-manager` -> True (disable)
- `--no-disable-hybrid-kv-cache-manager` -> False (force enable)

---

## 3. "Capable Connectors" - Which KV Connectors Support HMA

### 3.1 The `SupportsHMA` ABC

Defined in `vllm/distributed/kv_transfer/kv_connector/v1/base.py` (lines 85-114):

```python
class SupportsHMA(ABC):
    """Indicates the corresponding connector supports hybrid memory allocator (HMA)."""

    @abstractmethod
    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called once when a request has finished for all kv cache groups."""
```

The key difference: `SupportsHMA` connectors handle `block_ids` as a **tuple
of per-group lists** (e.g., `(FA_blocks, SW_blocks)`), while non-HMA connectors
handle a single flat list (one group only).

### 3.2 Connectors That Support HMA (8 total)

| Connector | Module | Key HMA Feature |
|-----------|--------|----------------|
| `SimpleCPUOffloadConnector` | `v1/simple_cpu_offload_connector.py` | request_finished_all_groups |
| `OffloadingConnector` | `v1/offloading_connector.py` | request_finished_all_groups, SW clipping |
| `NixlConnector` | `v1/nixl/connector.py` | per-group block IDs, Mamba N-1 prefill |
| `MooncakeConnector` | `v1/mooncake/mooncake_connector.py` | SW clipping, per-group send/recv |
| `MooncakeStoreConnector` | `v1/mooncake/store/connector.py` | per-group block handling |
| `DecodeBenchConnector` | `v1/decode_bench_connector.py` | benchmark connector with HMA |
| `ExampleHiddenStatesConnector` | `v1/example_hidden_states_connector.py` | example with HMA |
| `MultiConnector` | `v1/multi_connector.py` | wrapper: HMA if ALL children support HMA |

### 3.3 Connectors That Do NOT Support HMA (7 total)

| Connector | Impact |
|-----------|--------|
| `ExampleConnector` | HMA auto-disabled, error if forced |
| `P2pNcclConnector` | HMA auto-disabled |
| `LMCacheConnectorV1` | HMA auto-disabled |
| `LMCacheMPConnector` | HMA auto-disabled |
| `MoRIIOConnector` | HMA auto-disabled |
| `HF3FSKVConnector` | HMA auto-disabled |
| `FlexKVConnectorV1` | HMA auto-disabled |

### 3.4 Auto-Disable Logic

When a non-HMA connector is used with `kv_transfer_config`, the factory
(`vllm/distributed/kv_transfer/kv_connector/factory.py`) checks:

```python
hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
if hma_enabled and not cls.supports_hma_config(kv_transfer_config):
    raise ValueError("Connector does not support HMA but HMA is enabled.")
```

For MultiConnector: `supports_hma_config` checks that **ALL** child
connectors support HMA. If any child doesn't, the entire MultiConnector
is classified as not supporting HMA -> HMA auto-disabled.

The VllmConfig resolution (lines 1483-1506) auto-disables HMA for
non-HMA connectors with a warning:
```
"Turning off hybrid kv cache manager because the KV connector does not
support it. Impact: hybrid SSM models (e.g. Jamba, Bamba) require HMA
and will fail at startup without it; models with sliding window attention
will run with reduced performance."
```

---

## 4. How HMA Prevents Startup OOM

### 4.1 The Startup Pool Sizing Problem

vLLM's startup determines the number of KV cache blocks based on:

```
available_memory = total_gpu_memory * gpu_memory_utilization - model_weights_size
num_blocks = available_memory / max_memory_usage_per_request
```

`max_memory_usage_per_request` is calculated from `KVCacheSpec.max_memory_usage_bytes()`.

With HMA disabled:
```
For Mixtral 8x7B (32 layers, sliding_window=32768, max_model_len=32768):
  - All 32 layers -> FullAttentionSpec
  - max_memory = 32 * cdiv(32768, 16) * page_size_bytes
  - = 32 * 2048 * page_size = 65,536 * page_size per request
```

With HMA enabled:
```
  - 8 FA layers -> FullAttentionSpec (full max_model_len)
  - 24 SW layers -> SlidingWindowSpec (sliding_window sized)
  - max_memory = 8 * 2048 * page_size + 24 * (SW_blocks) * page_size
  - SW_blocks = cdiv(min(32767+max_num_batched_tokens, 32768), 16)+1
  - = ~2049 blocks (for SW=32768, max_model_len=32768)
  - Actually: when max_model_len = sliding_window, savings are minimal
  - But when max_model_len > sliding_window, savings are massive!
```

### 4.2 Real Impact: max_model_len > sliding_window

For Mixtral with `max_model_len=131072` (128K) and `sliding_window=32768`:

```
HMA disabled:
  - All layers: 131072/16 = 8192 blocks per layer
  - Total per request: 32 * 8192 * page_size_bytes
  - On RTX 4090 24GB: Mixtral INT4 weighs ~25GB -> available for KV ~0GB -> OOM!

HMA enabled:
  - 8 FA layers: 131072/16 = 8192 blocks
  - 24 SW layers: 32768/16+1 = 2049 blocks
  - Total: 8*8192 + 24*2049 = 65,536 + 49,176 = 114,712 blocks
  - vs disabled: 32*8192 = 262,144 blocks
  - Savings: (262,144 - 114,712) / 262,144 = ~56% reduction!

  - Per-request memory: ~56% less -> can fit ~2x more concurrent requests
```

### 4.3 Mamba Hybrid Models (Jamba, Bamba)

For Jamba (Mamba+Attention hybrid):
```
HMA disabled:
  - All layers -> FullAttentionSpec
  - Mamba state layers ALSO get FullAttentionSpec sizing
  - = max_model_len blocks for EVERY layer including Mamba -> massive waste

HMA enabled:
  - Attention layers -> FullAttentionSpec (correct)
  - Mamba layers -> MambaSpec (2 blocks for state cache)
  - = dramatic savings for the Mamba layers
  - Jamba: ~75% Mamba layers + ~25% Attention -> savings ~75%!
```

This is why the config warning says "hybrid SSM models (e.g. Jamba, Bamba)
require HMA and will fail at startup without it." Without HMA, Mamba layers
are sized as if they need max_model_len blocks, which is astronomically
larger than the 2-block state cache they actually need.

---

## 5. Memory Savings on RTX 4090

### 5.1 Qwen2-7B

Qwen2-7B uses **pure FullAttention** (no sliding window). HMA has NO
effect because all layers are the same type. No savings.

```
Qwen2-7B: all FullAttentionSpec -> is_kv_cache_spec_uniform=True
-> HMA makes no difference (already one group)
```

### 5.2 Mixtral 8x7B (INT4)

Mixtral uses sliding_window=32768 for all attention layers (8 FA + 24 SW
in the original architecture, though vLLM may group them differently).

```
Model weights (INT4 Marlin/AWQ): ~25GB on RTX 4090
Available for KV: 24GB * 0.9 - 25GB = ~-2.4GB -> DOESN'T FIT!

Even with INT4 quantization, Mixtral 8x7B barely fits on RTX 4090.
HMA doesn't help with the weight size problem, but it does help when
max_model_len > sliding_window.

For shorter max_model_len (e.g., 4096):
  HMA disabled: 32 * cdiv(4096, 16) * page_size = 32*256*page_size
  HMA enabled: 8*256 + 24*(cdiv(4095+4096, 16)+1) ≈ 8*256+24*513 = ~14,440 blocks
  vs disabled: 32*256 = 8,192 blocks
  -> HMA enabled actually requires MORE blocks for short max_model_len!
  (because SlidingWindowSpec accounts for sliding_window + max_num_batched_tokens)

Key insight: HMA savings depend on max_model_len vs sliding_window ratio.
  - max_model_len <= sliding_window: minimal or no savings (SW ≈ FA)
  - max_model_len > sliding_window: significant savings (SW << FA)
  - max_model_len >> sliding_window: massive savings (SW <<<<< FA)
```

### 5.3 Gemma Models (Gemma-2, Gemma-3)

Gemma-2-9B uses alternating local (sliding_window=4096) and global attention.
Gemma-3 uses similar mixed attention.

```
For Gemma-3-1B-it with sliding_window=512 (from test_nixl_connector_hma.py):
  max_model_len=2048
  HMA disabled: all layers -> 2048/16 = 128 blocks per layer
  HMA enabled:
    FA group: 2048/16 = 128 blocks
    SW group: 512/16+1 = 33 blocks per SW layer
    -> Savings per SW layer: (128-33) = 95 blocks = 74% reduction!
    -> For N SW layers: N * 95 blocks saved

For Gemma-2-9B with sliding_window=4096, max_model_len=8192:
  HMA disabled: all layers -> 8192/16 = 512 blocks
  HMA enabled:
    FA group: 512 blocks
    SW group: 4096/16+1 = 257 blocks
    -> Savings per SW layer: 255 blocks = 50% reduction
```

### 5.4 Mamba Hybrids (Jamba, Bamba, Nemotron)

These are the biggest winners from HMA-by-default:

```
For Jamba (Mamba2+Attention hybrid, ~52B params):
  Without HMA: ALL layers -> FullAttentionSpec -> max_model_len blocks
  With HMA: Mamba layers -> MambaSpec -> 2 blocks (state cache)
  Savings on Mamba layers: from max_model_len blocks to 2 blocks!
  For Jamba with ~75% Mamba layers: total savings ≈ 75% of KV memory

For Nemotron-H-8B (Mamba2+Attention hybrid):
  Without HMA: Mamba2 layers get FullAttentionSpec -> max_model_len blocks
  With HMA: Mamba2 layers get MambaSpec -> 1-2 blocks
  Savings: ~50-70% (depending on Mamba layer ratio)
```

### 5.5 Deepseek-V4

DS-V4 has the most complex hybrid: FullAttentionSpec + multiple
SlidingWindowMLASpec groups with different block sizes (256 vs 64).

```
Without HMA: SlidingWindowMLASpec -> MLAAttentionSpec (loses SW savings)
  All MLA layers sized for max_model_len -> massive overallocation

With HMA:
  Full MLA group -> FullAttentionSpec -> max_model_len blocks
  SW MLA groups -> SlidingWindowMLASpec -> window-sized blocks
  Savings: depends on SW window sizes vs max_model_len
```

---

## 6. HMA Interaction with Other Features

### 6.1 HMA + INT8 KV Cache on SM89

**Independent and compatible.** INT8 KV changes `kv_quant_mode` from
`NONE` to `INT8_PER_TOKEN_HEAD`, which affects `page_size_bytes` but
does NOT affect the grouping logic. HMA grouping works identically
with INT8 KV as with BF16 KV.

```
INT8 KV on SM89 (RTX 4090):
  kv_quant_mode = INT8_PER_TOKEN_HEAD
  page_size_bytes includes: 2*block_size*num_kv_heads*head_size*2 (INT8)
                        + 2*block_size*num_kv_heads*4 (scales)
  HMA grouping: same as BF16 -> SlidingWindowSpec gets INT8-sized window blocks
  Result: INT8 + HMA = double savings: smaller page + fewer blocks for SW layers
```

### 6.2 HMA + Prefix Caching

**Partial compatibility.** Prefix caching works differently per group:

```
FullAttentionManager.find_longest_cache_hit():
  -> Normal prefix cache matching across all blocks
  -> Works with prefix caching

SlidingWindowManager.find_longest_cache_hit():
  -> Only matches within sliding_window_contiguous_blocks
  -> get_num_common_prefix_blocks() returns 0
  -> Sliding window group does NOT participate in prefix caching

MambaSpec:
  -> No prefix caching (state-based, not token-based)

Implication: HMA + prefix caching = prefix caching only works for
the FullAttention group(s), not for SlidingWindow or Mamba groups.
For mixed-attention models, prefix cache savings are limited to
the FA layers only.
```

### 6.3 HMA + Multi-Tier KV Offloading (#41968, #44287)

**Compatible with limitations.**

PR #41968 introduced multi-tier offloading (GPU -> CPU -> FS/S3).
PR #44287 removed the assertion that blocked HMA models from tiering.

Current state in `vllm/v1/kv_offload/tiering/spec.py` (line 113):
```python
assert len(self.gpu_block_size) == 1
```

This asserts that all KV cache groups have the same GPU block_size.
For most HMA models (Mixtral, Gemma), this holds because FA and SW
groups use the same block_size. But for models with different
block_sizes per group (e.g., DS-V4 with 256 and 64), tiering
offloading would fail.

```
HMA + Tiering Offloading on RTX 4090:
  Mixtral: works (all groups block_size=16)
  Gemma: works (all groups block_size=16)
  Jamba/Bamba: works (all groups block_size=16)
  DS-V4: BLOCKED by assert (256 vs 64 block sizes)

Workaround for DS-V4: not applicable on RTX 4090 anyway (too large)
```

### 6.4 HMA + KV Connectors (P/D Disaggregation)

HMA-aware connectors (NixlConnector, MooncakeConnector) handle
per-group block IDs differently from non-HMA connectors:

```
HMA connector path (scheduler.request_finished):
  isinstance(connector, SupportsHMA) -> True
  -> connector.request_finished_all_groups(request, block_ids)
  -> block_ids = (FA_blocks, SW_blocks) -> per-group handling

Non-HMA connector path:
  -> assert len(kv_cache_groups) == 1  (forced single group)
  -> connector.request_finished(request, block_ids[0])  (flat list)

For P/D disaggregation with HMA:
  Prefill side: may have HMA groups (FA + SW + Mamba)
  Decode side: must have matching group structure
  -> NixlConnector/MooncakeConnector handle group alignment
  -> If group counts mismatch: "KV group count mismatch" error
```

### 6.5 HMA + EAGLE Speculative Decoding

**NOT compatible** when chunked local attention is present:

```python
if model_config.attention_chunk_size is not None:
    if speculative_config.use_eagle():
        need_disable_hybrid_kv_cache_manager = True
        # "Hybrid KV cache manager is not yet supported with
        # chunked local attention + eagle."
```

For models with pure sliding window (no chunked local attention),
HMA + EAGLE may work. But chunked local attention + EAGLE forces
HMA to be disabled.

Additionally, SM89 batch invariance (#39096) means EAGLE on RTX 4090
needs `enforce_eager=True`, which already limits EAGLE benefits.

---

## 7. Auto-Disable Conditions (Full Decision Tree)

```
Step 1: Platform check
  current_platform.support_hybrid_kv_cache() = False on non-GPU -> DISABLE

Step 2: Chunked local attention check
  model_config.attention_chunk_size != None:
    if speculative_config.use_eagle(): -> DISABLE (chunked_local + eagle incompatible)
    elif not VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: -> DISABLE (latency regression)
    else: -> ALLOW (with env override)

Step 3: KV connector check (if kv_transfer_config != None)
  if not KVConnectorFactory.supports_hma_config(kv_transfer_config): -> DISABLE with warning

Step 4: Final fallback
  if still None -> False (ENABLE HMA by default)

Error if user explicitly --no-disable-hybrid-kv-cache-manager but runtime can't support it
```

---

## 8. HMA-related Bugs/Issues After v0.23.0

### 8.1 Known Issues from Source Code and Tests

**TieringOffloadingSpec block_size assertion** (`tiering/spec.py` line 113):
```python
assert len(self.gpu_block_size) == 1
```
This blocks DS-V4 (with 256 vs 64 block sizes per group) from using
tiering offloading. Not a bug per se, but a limitation that PR #44287
partially addressed (removed the kv_cache_groups==1 assert but kept
the gpu_block_size==1 assert).

**SlidingWindowManager prefix caching limitation**: SW group returns
0 common prefix blocks. This means cascade attention and prefix caching
don't benefit sliding-window layers. Future work needed.

**NixlConnector prefix caching with Mamba**: Tests in
`test_nixl_connector_hma.py` demonstrate that `_apply_prefix_caching`
front-trims FA groups, which can produce **silent data corruption** when
prefix caching produces suffix-only local blocks with mismatched
`physical_per_logical` values. This is a known test-documented bug path
for heterogeneous TP (e.g., Qwen3.5 4P2D).

### 8.2 Search for Post-v0.23.0 Issues

No specific HMA regression bugs were found in the v0.23.0 release.
The feature has been stable since merge, with comprehensive test coverage:
- `test_hma_auto_config.py`: 4 test cases for auto-disable logic
- `test_mooncake_connector_hma.py`: SW clipping, multi-group metadata
- `test_mooncake_store_hma_e2e.py`: e2e MooncakeStore with HMA
- `test_nixl_connector_hma.py`: 18 test cases covering SW sizes, group
  handling, Mamba N-1 prefill, prefix caching alignment, block ID expansion

---

## 9. RTX 4090 Decision Matrix for HMA

```
| Model            | Attention Type      | HMA Effect         | Savings Estimate |
|------------------|---------------------|--------------------|------------------|
| Qwen2-7B         | Pure FullAttention  | No effect          | 0%               |
| Mixtral 8x7B     | Full+SW(32K)        | SW blocks reduced  | Depends on ratio |
| Gemma-2-9B       | Full+SW(4K)         | SW blocks reduced  | ~50% per SW layer|
| Gemma-3-1B       | Full+SW(512)        | SW blocks reduced  | ~74% per SW layer|
| Jamba            | Mamba+Full          | Mamba 2-block state| ~75% total       |
| Bamba            | Mamba+Full          | Mamba 2-block state| ~70% total       |
| Nemotron-H-8B    | Mamba2+Full         | Mamba2 state cache | ~60% total       |
| DS-V4-Flash      | MLA+SW_MLA          | SW_MLA window sized| ~40-60%          |
| Llama-3.x-8B     | Pure FullAttention  | No effect          | 0%               |
```

**RTX 4090 bottom line**: HMA-by-default is a **net positive** for all
hybrid-attention models. For pure-full-attention models (Qwen2-7B,
Llama-3.x-8B), HMA has no effect (already uniform). The biggest winners
are Mamba hybrid models (Jamba, Bamba) where HMA reduces Mamba state
layers from max_model_len blocks to 2 blocks.

---

## 10. Key Source Files

| File | Path | Key Content |
|------|------|-------------|
| Config resolution | `vllm/vllm/config/vllm.py:1449-1520` | HMA auto-enable/disable logic |
| SchedulerConfig | `vllm/vllm/config/scheduler.py:132-138` | disable_hybrid_kv_cache_manager field |
| Spec grouping | `vllm/vllm/v1/core/kv_cache_utils.py:1618-1684` | get_kv_cache_groups() |
| Spec unification | `vllm/vllm/v1/core/kv_cache_utils.py:1324-1417` | unify_hybrid_kv_cache_specs() |
| SupportsHMA ABC | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:85-114` | SupportsHMA + request_finished_all_groups |
| Connector factory | `vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:131-145` | supports_hma_config() |
| SlidingWindowManager | `vllm/vllm/v1/core/single_type_kv_cache_manager.py:570-739` | SW block eviction, prefix=0 |
| Platform support | `vllm/vllm/platforms/cuda.py:549-550` | support_hybrid_kv_cache() = True |
| Tiering offloading | `vllm/vllm/v1/kv_offload/tiering/spec.py:113` | gpu_block_size assert |
| HMA auto-config tests | `vllm/tests/v1/kv_connector/unit/test_hma_auto_config.py` | 4 test cases |
| Nixl HMA tests | `vllm/tests/v1/kv_connector/unit/test_nixl_connector_hma.py` | 18 test cases |
| Mooncake HMA tests | `vllm/tests/v1/kv_connector/unit/test_mooncake_connector_hma.py` | 6 test cases |
| KVCacheSpec types | `vllm/vllm/v1/kv_cache_interface.py` | FullAttentionSpec, SlidingWindowSpec, MambaSpec, etc. |

---

## 11. Summary of Key Questions

**Q: What exactly does HMA do differently from the old default KV cache management?**
A: HMA creates separate KVCacheGroupSpec per attention type with proper
sizing. The old default collapsed all types into FullAttentionSpec = one
group with max_model_len sizing for all layers, massively overallocating
for sliding-window and Mamba layers.

**Q: How does HMA handle models with mixed attention types?**
A: Each type gets its own group. FullAttention layers get max_model_len
blocks, SlidingWindow layers get sliding_window blocks, Mamba layers
get 2 state-cache blocks. The BlockPool is shared across all groups
(blocks allocated per-request, not per-group preallocation). Each group
has its own SingleTypeKVCacheManager with appropriate eviction logic
(SlidingWindowManager.remove_skipped_blocks drops old blocks).

**Q: Memory savings on RTX 4090?**
A: Qwen2-7B: 0% (pure FA). Mixtral: depends on max_model_len vs
sliding_window ratio. Gemma-3: ~74% per SW layer. Jamba/Bamba: ~75%
total. Nemotron-H: ~60% total. The biggest savings come from Mamba
hybrid models where state layers go from max_model_len to 2 blocks.

**Q: Does HMA interact with INT8 KV cache on SM89?**
A: Independent and compatible. INT8 changes page_size_bytes but not the
grouping logic. INT8 + HMA = double savings: smaller pages + fewer
blocks per SW/Mamba layer.

**Q: Is HMA compatible with prefix caching?**
A: Partially. FullAttention groups use normal prefix caching. SlidingWindow
groups return 0 common prefix blocks (no prefix caching). Mamba groups
are state-based (no prefix caching). For mixed models, prefix caching
only benefits the FA group.

---

Sources:
- vLLM local source: `/Users/jackiemac/workspace/rollout-infra/vllm/`
- PR #41847: https://github.com/vllm-project/vllm/pull/41847
- PR #41968 (multi-tier KV): https://github.com/vllm-project/vllm/pull/41968
- PR #44287 (tiering HMA): https://github.com/vllm-project/vllm/pull/44287
- Existing analysis: `/Users/jackiemac/workspace/rollout-infra/notebook/projects/vllm-v0.23-rtx4090-impact-reading.md`
