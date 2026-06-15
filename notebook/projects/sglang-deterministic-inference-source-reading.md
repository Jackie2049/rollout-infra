# SGLang Deterministic Inference Source-Level Reading

**Date**: 2026-06-16
**Repo**: sgl-project/sglang (main branch)
**Focus**: Deterministic inference, RadixAttention, Triton kernels, DeepGEMM SM89 fallback
**Context**: RTX 4090 (SM89) GRPO training infrastructure

---

## 1. SGLang Deterministic Inference Implementation (SOURCE-LEVEL)

### ★★★★★★★★★★★ CRITICAL: SGLang solves batch invariance at KERNEL level (Triton persistent constexpr) vs vLLM at COMPILE level (Inductor fusion) — DIFFERENT architectural layers!

### 1.1 Architecture Overview

SGLang's deterministic inference system is built on two pillars:
1. **batch_invariant_ops** — aten override layer that replaces non-deterministic ops with Triton persistent kernels with constexpr BLOCK_SIZE
2. **Triton attention backend** — fixed split_tile_size replaces dynamic KV splitting

The system was adapted from [Thinking Machines Lab's batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) and significantly extended in SGLang with DeepGEMM integration, bmm override, and rms_norm override.

### 1.2 The `--enable-deterministic-inference` Flag Wiring

**File**: `python/sglang/srt/server_args.py`

```python
enable_deterministic_inference: bool = False  # line 822
rl_on_policy_target: Optional[str] = None     # line 823
```

**CRITICAL**: `rl_on_policy_target` automatically enables deterministic inference:

```python
def _handle_deterministic_inference(self):  # line 4430
    if self.rl_on_policy_target is not None:
        logger.warning("Enable deterministic inference because of rl_on_policy_target.")
        self.enable_deterministic_inference = True
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(True)
```

★★★★★★★ **RTX 4090 GRPO Direct Impact**: When verl/rLLM sets `rl_on_policy_target`, SGLang automatically enables deterministic inference! No manual flag needed!

The `_handle_deterministic_inference` method then:
- Disables `enable_aiter_allreduce_fusion` and `enable_flashinfer_allreduce_fusion`
- Sets `sampling_backend = "pytorch"` (not flashinfer sampling)
- Falls back to `fa3` or `triton` attention backend for SM<100 (RTX 4090 gets `fa3` by default)
- Disables radix cache for FlashInfer backend (but keeps it for FA3/Triton)

**Attention backend fallback logic** (line 4482-4493):
```python
if self.attention_backend is None:
    if is_sm100_supported() or is_sm120_supported():
        # Blackwell and newer architectures
        self.attention_backend = "triton" if is_deepseek_model else "flashinfer"
    else:
        # Hopper (SM90) and older architectures — RTX 4090 falls here!
        self.attention_backend = "fa3"
```

★★★★★★★★★ **RTX 4090 Recommendation**: Use `--attention-backend triton` for deterministic inference + radix cache. FA3 also works but Triton backend gives full compatibility (radix cache + CUDA graphs + chunked prefill + non-greedy sampling).

### 1.3 Aten Override Registration Mechanism

**File**: `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`

The core mechanism uses `torch.library.Library("aten", "IMPL")` to replace aten ops at dispatch level:

```python
def enable_batch_invariant_mode(enable_bmm: bool = True):  # line 980
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_MODE:
        return

    dispatch_key = get_dispatch_device_backend()  # "CUDA" for NVIDIA
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    # Register 7 aten overrides for deterministic inference:
    _batch_invariant_LIB.impl("aten::mm",            mm_batch_invariant,            dispatch_key)  # line 992
    _batch_invariant_LIB.impl("aten::addmm",         addmm_batch_invariant,         dispatch_key)  # line 993
    _batch_invariant_LIB.impl("aten::_log_softmax",  _log_softmax_batch_invariant,  dispatch_key)  # line 994
    _batch_invariant_LIB.impl("aten::mean.dim",      mean_batch_invariant,          dispatch_key)  # line 997
    _batch_invariant_LIB.impl("aten::rms_norm",      _rms_norm_aten_compat,         dispatch_key)  # line 998
    _batch_invariant_LIB.impl("aten::mm.dtype",      _mm_dtype_compat,              dispatch_key)  # line 999
    _batch_invariant_LIB.impl("aten::bmm",           bmm_batch_invariant,           dispatch_key)  # line 1002
```

★★★★★★ **7 aten overrides registered**: mm, addmm, _log_softmax, mean.dim, rms_norm, mm.dtype, bmm. This is MORE than the original Thinking Machines Lab implementation which only had 4 (mm, addmm, _log_softmax, mean.dim). SGLang added rms_norm, mm.dtype, and bmm.

The `bmm` override is also monkeypatched directly on `torch.bmm`:
```python
if enable_bmm:
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, dispatch_key)
    _original_torch_bmm = torch.bmm
    torch.bmm = bmm_batch_invariant  # line 1004-1005
```

### 1.4 Triton Kernel Implementations (SOURCE-LEVEL)

#### 1.4.1 matmul_kernel_persistent — The Core Matmul

**File**: `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, line 79-170

★★★★★★★★★★★ **KEY INSIGHT**: All BLOCK_SIZE parameters are `tl.constexpr` — compile-time constants, NOT autotuned!

```python
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,   # ← COMPILE-TIME CONSTANT
    BLOCK_SIZE_N: tl.constexpr,   # ← COMPILE-TIME CONSTANT
    BLOCK_SIZE_K: tl.constexpr,   # ← COMPILE-TIME CONSTANT
    GROUP_SIZE_M: tl.constexpr,   # ← COMPILE-TIME CONSTANT
    NUM_SMS: tl.constexpr,        # ← COMPILE-TIME CONSTANT
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        # ... standard persistent matmul loop with tl.dot accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = tl.load(a_ptrs, ...)
            b = tl.load(b_ptrs, ...)
            accumulator = tl.dot(a, b, accumulator)  # ← deterministic: fixed block sizes
```

**Fixed configs** (line 199-224):
```python
configs = {
    torch.bfloat16: {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
    torch.float16:  {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
    torch.float32:  {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8},
}
```

★★★★★★★★★★ **Why this is batch-invariant**: Unlike cuBLAS which autotunes tile sizes based on M/N/K dimensions (varying with batch size), this kernel uses FIXED constexpr block sizes. The same (128, 128, 64) configuration is used regardless of batch size. This means:
- The accumulation order is always the same
- `tl.dot` always processes the same number of elements per accumulation step
- No CachingAutotuner selecting different XBLOCK/RBLOCK for different xnumel

**This directly addresses the vLLM root cause**: vLLM's Inductor fuses RMSNorm into a single Triton kernel where `tl.sum()` becomes an inline reduction. CachingAutotuner selects different XBLOCK for different xnumel (batch sizes), causing different accumulation orders. SGLang's approach bypasses this entirely by using constexpr block sizes at the kernel level BEFORE Inductor ever gets a chance to fuse.

#### 1.4.2 _log_softmax_kernel

**File**: line 318-388

```python
@triton.jit
def _log_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride: tl.constexpr,    # ← constexpr!
    output_row_stride: tl.constexpr,   # ← constexpr!
    n_cols: tl.constexpr,              # ← constexpr!
    BLOCK_SIZE: tl.constexpr,          # ← constexpr! = 1024
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # Step 1: Find max — iterate with fixed BLOCK_SIZE
    max_val = tl.max(vals_init)
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))
    # Step 2: Sum of exp — iterate with fixed BLOCK_SIZE
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))
    # Step 3: Output — subtract max and log_sum_exp
    log_sum_exp = tl.log(sum_exp)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        output = vals - max_val - log_sum_exp
```

★★★★★★★ **Key design**: Each row is processed by ONE program (`grid = (n_rows,)`). The max and sum reductions iterate over columns with fixed BLOCK_SIZE=1024. The accumulation order is deterministic because:
- One program per row (not parallelized across rows with different tile counts)
- Fixed iteration pattern (always BLOCK_SIZE chunks)
- No autotuning of block sizes

#### 1.4.3 mean_kernel

**File**: line 434-482

```python
@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    input_stride0, input_stride1, input_stride2,
    output_stride0, output_stride1,
    M, N, K,  # dimensions before/after/being reduced
    BLOCK_SIZE: tl.constexpr,          # ← constexpr! = 1024
):
    pid = tl.program_id(0)
    m_idx = pid // K
    k_idx = pid % K
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)
    mean_val = acc / N
    tl.store(output_ptr + output_idx, mean_val)
```

★★★★★★★★★★★★★★ **This is THE kernel that replaces torch.mean.dim!** In vLLM's bug (Issue #39096), Inductor fuses RMSNorm so that `torch.mean` becomes an inline `tl.sum()` with autotuned XBLOCK. Here, SGLang's mean_kernel uses a FIXED BLOCK_SIZE=1024 constexpr, so the accumulation order NEVER changes with batch size.

**Grid**: `(M * K,)` — one program per output element, not per-batch. Each program independently walks the reduction dimension N with fixed step size. This guarantees deterministic accumulation regardless of how many other programs are running concurrently.

#### 1.4.4 _rms_norm_kernel

**File**: line 822-871

```python
@triton.jit
def _rms_norm_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_row_stride: tl.constexpr,    # ← constexpr!
    output_row_stride: tl.constexpr,   # ← constexpr!
    n_cols: tl.constexpr,              # ← constexpr!
    eps,
    BLOCK_SIZE: tl.constexpr,          # ← constexpr! = 1024
):
    row_idx = tl.program_id(0).to(tl.int64)
    # Step 1: Sum of squares with fixed BLOCK_SIZE
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    # Step 2: Compute RMS
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    # Step 3: Normalize and scale
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        output_f32 = vals_f32 * inv_rms * weight_f32
```

★★★★★★★★★★★★★★★★★★★★★★★★★ **CRITICAL: This is SGLang's solution to vLLM's #39096 root cause!**

vLLM's problem: Inductor fuses RMSNorm's `pow2 + mean + rsqrt + mul` into ONE Triton kernel where `mean(x^2)` becomes inline `tl.sum()` → CachingAutotuner selects different XBLOCK → batch-dependent accumulation.

SGLang's solution: RMSNorm is computed as a **standalone Triton kernel** with `tl.constexpr BLOCK_SIZE=1024`. The sum-of-squares reduction uses `tl.sum(tl.where(mask, sq_vals, 0.0))` with a fixed iteration pattern. No Inductor fusion, no autotuning, no batch-dependent behavior.

**Grid**: `(n_rows,)` — one program per row. Each program independently accumulates sum_sq. The program count changes with batch size, but each program's internal computation is identical.

#### 1.4.5 bmm_kernel_persistent (NEW in SGLang, not in original batch_invariant_ops)

**File**: line 607-723

```python
@triton.jit
def bmm_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak, stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,   # ← constexpr!
    BLOCK_SIZE_N: tl.constexpr,   # ← constexpr!
    BLOCK_SIZE_K: tl.constexpr,   # ← constexpr!
    GROUP_SIZE_M: tl.constexpr,   # ← constexpr!
    NUM_SMS: tl.constexpr,        # ← constexpr!
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    # Process tiles in a deterministic order: batch-major ordering
    for tile_id in tl.range(start_pid, num_tiles_total, NUM_SMS, flatten=True):
        batch_idx = tile_id // num_tiles_per_batch
        tile_in_batch = tile_id % num_tiles_per_batch
```

★★★★ **Design**: This is a batched matmul that processes ALL batch dimensions in a single persistent kernel. The tile ordering is **batch-major** (batch_idx derived from tile_id), ensuring deterministic processing order regardless of batch size.

### 1.5 Matmul Dispatch: DeepGEMM vs Triton

**File**: `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, line 275-315

★★★★★★★★★★★ **CRITICAL RTX 4090 finding**: DeepGEMM is NOT available on SM89!

```python
def matmul_persistent(a, b, bias=None):  # line 275
    MIN_DEEPGEMM_DIM = 16
    if (
        _ENABLE_MM_DEEPGEMM                    # True by default
        and ENABLE_JIT_DEEPGEMM                # ★★★ False on SM89!
        and (a.dtype == torch.bfloat16)
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= MIN_DEEPGEMM_DIM
    ):
        return _matmul_persistent_deepgemm(a, b, bias)  # SM90+ only
    # Fallback path for SM89:
    return _matmul_persistent_triton(a, b, bias)  # ← RTX 4090 uses THIS
```

**DeepGEMM configurer** (`python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`):
```python
def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False  # ← SM89 (RTX 4090) = False!
    # DeepGEMM requires TMEM/tcgen05 (SM100+), not available on SM120
    if sm_version == 120:
        return False
```

★★★★★★★★★★ **RTX 4090 consequence**: On SM89, `ENABLE_JIT_DEEPGEMM = False`, so matmul always falls through to `_matmul_persistent_triton`. This means:
- The Triton persistent matmul kernel with constexpr block sizes is the ONLY matmul path on RTX 4090
- This is GOOD for determinism — no DeepGEMM/TMA autotuning involved
- Performance may be lower than DeepGEMM on SM90+ (no TMA async loads), but determinism is guaranteed

### 1.6 Triton Attention Backend Deterministic Configuration

**File**: `python/sglang/srt/layers/attention/triton_backend.py`, line 90-260

```python
class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner, ...):
        # Decide whether enable deterministic inference
        self.enable_deterministic = model_runner.server_args.enable_deterministic_inference

        # Configure deterministic inference settings
        if self.enable_deterministic:
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256  # ← Fixed 256!
            )
            self.static_kv_splits = False  # ← NOT static, but deterministic via split_tile_size
```

★★★★★★ **KV Split Determinism**: In decode attention, the number of KV splits is computed per-request using a fixed `split_tile_size=256`:

```python
def get_num_kv_splits(self, num_kv_splits, seq_lens):  # line 260
    # deterministic path
    if self.split_tile_size is not None and self.enable_deterministic:
        num_kv_splits[:] = (expanded_seq_lens + self.split_tile_size - 1) // self.split_tile_size
        return
```

★★★★★★★ **Why this matters**: The KV split count determines how many parallel programs process the attention reduction. With a fixed split_tile_size, the split count depends ONLY on seq_len, NOT on batch size. This means the attention reduction order is deterministic.

**Prefill kernel BLOCK sizes** (triton_ops/prefill_attention.py):
```python
BLOCK_M: tl.constexpr = 16  # fixed
BLOCK_N: tl.constexpr = 16  # fixed (same as BLOCK_M)
BLOCK_DMODEL: tl.constexpr   # derived from Lk (head_dim) — model-specific, not batch-dependent
```

**Decode kernel BLOCK sizes** (triton_ops/decode_attention.py):
```python
BLOCK_N: tl.constexpr  # = 32 or 16 depending on head_dim parity
BLOCK_DMODEL: tl.constexpr  # = triton.next_power_of_2(Lk)
MIN_BLOCK_KV: tl.constexpr = 32
```

★★★★★★★★★ **All Triton attention kernels use `tl.constexpr` for BLOCK sizes** — NOT autotuned. This is fundamentally different from FlashInfer which autotunes tile sizes.

---

## 2. SGLang RadixAttention Architecture (SOURCE-LEVEL)

### ★★★★★★★ RadixAttention = nn.Module wrapper; RadixTreeCpp = C++ radix tree implementation with JIT-compiled pybind11 binding

### 2.1 RadixAttention Layer (Python)

**File**: `python/sglang/srt/layers/radix_attention.py`

`RadixAttention` is an `nn.Module` that delegates to the attention backend:

```python
class RadixAttention(nn.Module):
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, layer_id, ...):
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        # For extend mode with piecewise CUDA graph:
        if forward_batch.forward_mode.is_extend() and get_tc_piecewise_forward_context() is not None:
            return unified_attention_with_output(q, k, v, output, ...)
        # Normal path:
        return get_attn_backend().forward(q, k, v, self, forward_batch, save_kv_cache, **kwargs)
```

The actual KV cache management and prefix matching happens in the `tree_cache` (RadixTreeCpp), NOT in the RadixAttention layer itself. RadixAttention just routes to the appropriate backend.

### 2.2 RadixTreeCpp (C++ Implementation with JIT Compilation)

**File**: `python/sglang/srt/mem_cache/cpp_radix_tree/radix_tree.py`

★★★★★★ **The radix tree is JIT-compiled from C++ source at runtime!**

```python
radix_tree_cpp = load(
    name="radix_tree_cpp",
    sources=[
        f"{_abs_path}/tree_v2_binding.cpp",
        f"{_abs_path}/tree_v2_debug.cpp",
        f"{_abs_path}/tree_v2.cpp",
    ],
    extra_cflags=["-O3", "-std=c++20"],
)
```

**Key operations**:
```python
class RadixTreeCpp:
    def __init__(self, disabled, host_size, page_size, write_through_threshold):
        self.tree = radix_tree_cpp.RadixTree(disabled, host_size, page_size, write_through_threshold)

    def match_prefix(self, prefix: List[int]):
        """Returns: (device_indices, host_hit_length, device_node, host_node)"""

    def evict(self, num_tokens: int) -> List[torch.Tensor]:
        """Evicts num_tokens from tree, returns freed indices"""

    def lock_ref(self, handle, lock: bool):
        """Lock/unlock a node — locked nodes can't be evicted"""

    def writing_through(self, key, indices):
        """Insert key-value pair + write-through check (GPU→CPU sync)"""

    def loading_onboard(self, host_node, new_device_indices):
        """Load from CPU→GPU within a range of nodes"""
```

★★★★★★★★★ **Dual-tree architecture**: The RadixTreeCpp maintains TWO trees:
1. **Device tree** — GPU KV cache indices, lives in VRAM
2. **Host tree** — CPU mirror, for write-through/loading_onboard (HiCache)

The `write_through_threshold` parameter controls when GPU→CPU sync happens.

### 2.3 Radix Tree C++ Header

**File**: `python/sglang/srt/mem_cache/cpp_radix_tree/tree_v2.h`

```cpp
struct RadixTree {
 public:
  RadixTree(bool disabled, std::optional<std::size_t> host_size,
            std::size_t page_size, std::size_t threshold);

  // match_prefix: returns (device_indices, host_hit_length, device_node, host_node)
  std::tuple<std::vector<at::Tensor>, std::size_t, NodeHandle, NodeHandle>
  match_prefix(const token_vec_t& key);

  // evict: returns device indices to free
  std::vector<at::Tensor> evict(std::size_t num_tokens);

  // lock_ref: increment/decrement reference count
  void lock_ref(NodeHandle node_id, bool increment);

  // writing_through: insert + GPU→CPU write-through check
  std::tuple<std::vector<std::tuple<IOTicket, at::Tensor, at::Tensor>>, std::size_t>
  writing_through(const token_vec_t& key, at::Tensor value);

  // loading_onboard: CPU→GPU load within node range
  std::tuple<IOTicket, std::vector<at::Tensor>>
  loading_onboard(NodeHandle host_id, at::Tensor indices);

  std::size_t evictable_size() const;  // ref_count=0 tokens
  std::size_t protected_size() const;  // ref_count>0 tokens (locked)
  std::size_t total_size() const;      // all device tokens
};
```

★★★★★★★ **Key design features**:
- `NodeHandle` = opaque integer ID (not pointer — survives tree restructuring)
- `IOTicket` = async IO handle for GPU↔CPU transfers
- `match_prefix` returns BOTH device match (indices + node) AND host match (for HiCache loading)
- `lock_ref` uses boolean `increment` for atomic ref count management
- `evictable_size()` only counts nodes with ref_count=0 (shared prefixes have ref_count>0 → protected from eviction)

### 2.4 BasePrefixCache Architecture

**File**: `python/sglang/srt/mem_cache/base_prefix_cache.py`

★★★★★★★ **Unified dataclass-based API** with typed params/results:

```python
class BasePrefixCache(ABC, PrefixCacheTrait):
    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req, is_insert=True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req, **kwargs):
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass
```

**MatchResult** (NamedTuple):
```python
class MatchResult(NamedTuple):
    device_indices: torch.Tensor       # GPU KV cache indices matched
    last_device_node: Any              # last GPU tree node matched
    last_host_node: Any                # last CPU tree node matched
    best_match_node: Any               # deepest validated node (for load-back)
    host_hit_length: int = 0           # CPU-side KV tokens needing GPU load-back
    swa_host_hit_length: int = 0       # Sliding Window Attention hit
    mamba_host_hit_length: int = 0     # Mamba state hit
```

★★★★★★★★★ **GRPO Prefix Reuse Flow**:
1. GRPO sends `group_size` requests with SAME prompt prefix + different suffixes
2. First request: `match_prefix` finds no match → full prefill → `writing_through` inserts into tree
3. Second request: `match_prefix` finds the prefix node → only suffix prefill needed
4. Subsequent requests: prefix KV reused, only suffix computed
5. `lock_ref(handle, True)` prevents eviction during active group processing
6. After group completes: `lock_ref(handle, False)` → prefix becomes evictable

**Theoretical savings**: With group_size=G and prompt length=P, total prefill = P + G*S instead of G*(P+S) = savings of (G-1)*P tokens. For G=8, P=1024: saves 7168 tokens of prefill compute.

### 2.5 Radix Cache vs vLLM Prefix Cache

| Feature | SGLang RadixAttention | vLLM V1 APC |
|----------|----------------------|-------------|
| **Data structure** | Radix tree (C++ JIT-compiled) | BlockManager (hash-based) |
| **Matching granularity** | Token-level (page_size=1 possible) | Block-level (block_size=16/64) |
| **Tree branching** | YES — natural divergent suffix handling | NO — flat block sharing |
| **HiCache (CPU mirror)** | YES — write_through + loading_onboard | NO — no CPU KV mirror |
| **Ref counting** | Node-level (lock_ref) | Block-level (ref_cnt) |
| **Eviction** | LRU on evictable nodes (ref_count=0) | Watermark-based block eviction |
| **GRPO benefit** | Direct — prefix node shared across group | Indirect — block-level sharing |

★★★★★★★★★★★ **SGLang radix cache is superior for GRPO** because:
1. Token-level matching (not block-level) means NO wasted tokens from block alignment
2. Tree branching naturally handles divergent suffixes (each group member gets its own branch)
3. HiCache enables CPU→GPU load-back for cold prefixes (vLLM has no equivalent)
4. lock_ref guarantees prefix won't be evicted during group processing

---

## 3. SGLang Scheduler Architecture (SOURCE-LEVEL)

### ★★★★★ **SGLang scheduler is a 4145-line monolithic class** with extensive mixin decomposition

### 3.1 Scheduler Class Overview

**File**: `python/sglang/srt/managers/scheduler.py`

```python
class Scheduler(
    SchedulerMlxOverlapMixin,
    # ... numerous mixin classes
):
    def __init__(self, server_args, port_args, model_config, ...):
        # ~300 lines of initialization
        self.tree_cache = ...           # RadixTreeCpp via PrefixCache
        self.token_to_kv_pool_allocator = ...
        self.running_batch = ...
        self.waiting_queue = ...
        self.chunked_req = None
        # ... init all components
```

★★★★ **Key architectural difference from vLLM V1**: SGLang uses a **monolithic scheduler with mixin decomposition** (all scheduling logic in one class, broken into mixins for maintainability). vLLM V1 uses a **unified Scheduler class** that tightly couples scheduling with BlockManager.

### 3.2 Prefill Scheduling — get_new_batch_prefill

**File**: line 2612-2700+

```python
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    prefill_delayer = ...
    ret = self._get_new_batch_prefill_raw(prefill_delayer_single_pass=...)
    return ret

def _get_new_batch_prefill_raw(self, ...):
    # Check if running batch is full
    if self.running_batch.batch_is_full or len(self.waiting_queue) == 0:
        return None

    # Priority scheduling
    self.policy.calc_priority(self.waiting_queue, self.running_batch)

    # Prefill policy — PrefillAdder manages token budget
    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,                # ← RadixAttention prefix matching HERE
        self.token_to_kv_pool_allocator,
        self.running_batch,
        ...
        chunked_prefill_size,
        max_prefill_bs=...,
    )
```

★★★★★★★ **PrefillAdder** is the key scheduling primitive that:
1. Matches prefix for each incoming request against `self.tree_cache`
2. Computes how many NEW tokens need prefill (after prefix reuse)
3. Fits requests into the token budget considering prefix reuse savings
4. Supports chunked prefill for long requests

### 3.3 Decode Scheduling — update_running_batch

**File**: line 2905-2970

```python
def update_running_batch(self, batch: ScheduleBatch):
    batch.filter_batch()  # remove finished requests
    if not batch.check_decode_mem():  # OOM check
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(server_args)
```

★★★★ **Retraction mechanism**: When decode OOM occurs, SGLang retracts (evicts) requests to free KV cache. This is similar to vLLM V1's preemption but operates at the request level (not block level).

### 3.4 Deterministic Inference Config in Scheduler

**File**: line 1305-1321

```python
def init_deterministic_inference_config(self):
    if not self.server_args.enable_deterministic_inference:
        self.truncation_align_size = None
        return

    backend_sizes = {
        "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
        "triton":    ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
    }
    env_var, default_size = backend_sizes.get(self.server_args.attention_backend, (None, None))
    self.truncation_align_size = get_int_env_var(env_var, default_size) if env_var else None
```

★★★★★ **truncation_align_size=4096**: When deterministic inference is enabled, chunked prefill splits are aligned to 4096 tokens. This ensures the prefill computation has consistent token counts per chunk, avoiding batch-dependent behavior in the prefill kernels.

---

## 4. Deterministic Sampling — murmur_hash32 + Gumbel Trick

### ★★★★★★★★★★ SGLang implements per-position deterministic sampling via murmur_hash32 seed → float64 Gumbel noise → reproducible sampling across runs

### 4.1 Sampling Seed Flow

**File**: `python/sglang/srt/sampling/sampling_batch_info.py`

```python
sampling_seed = (
    torch.tensor(
        [r.sampling_params.sampling_seed if r.sampling_params.sampling_seed is not None else 42
         for r in reqs],
        dtype=torch.int64,
        pin_memory=_pin,
    ).to(device, non_blocking=True)
    if enable_deterministic  # ← only created when deterministic inference enabled
    else None
)
```

★★★★★★★★★ **Default seed = 42** when no explicit sampling_seed provided. For GRPO: each group member should specify a different `sampling_seed` (e.g., 42, 43, 44, ...) to get diverse but reproducible rollouts.

### 4.2 murmur_hash32 Triton Kernel

**File**: `python/sglang/srt/layers/utils/hash.py`

★★★★★★★★★★ **Full Triton implementation of MurmurHash3 32-bit!**

```python
@triton.jit
def murmur_hash32_kernel(
    seed_ptr, positions_ptr, col_indices_ptr, output_ptr,
    num_rows, num_cols,
    BLOCK_SIZE: tl.constexpr,          # ← constexpr! = 1024
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_idx = pid_row
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load inputs
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)     # 64-bit sampling_seed
    pos = tl.load(positions_ptr + row_idx).to(tl.uint32)  # token position
    col = tl.load(col_indices_ptr + col_offsets, mask=mask, other=0).to(tl.uint32)  # vocab index

    # Hash: mix seed_low + seed_high + position + col_index
    h: tl.uint32 = 0
    h = murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))   # seed low 32 bits
    h = murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))  # seed high 32 bits
    h = murmur3_mix(h, pos)   # position (ensures per-token uniqueness)
    h = murmur3_mix(h, col)   # vocab index
    h = fmix32(h)              # finalization

    tl.store(output_ptr + row_idx * num_cols + col_offsets, h, mask=mask)
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ **CRITICAL GRPO Insight**: The hash combines seed + position + vocab_index. This means:
- Same seed + same position = same hash = same Gumbel noise = same sampled token
- Different seed (e.g., 42 vs 43) = different hash = different Gumbel noise = diverse rollout
- Per-position hashing ensures each token in the sequence gets a unique random value

### 4.3 multinomial_with_seed — Gumbel Trick Sampling

**File**: `python/sglang/srt/layers/sampler.py`, line 603-660

```python
def multinomial_with_seed(logprobs, seed, positions):
    n, m = logprobs.shape  # batch_size, vocab_size
    seed = seed.to(torch.uint64)
    col_indices = torch.arange(m, device=logprobs.device)
    hashed = murmur_hash32(seed, positions, col_indices)

    # ★★★★★★★★ CRITICAL: float64 for Gumbel noise to avoid numerical instability!
    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max  # uniform [0, 1]

    # Gumbel noise: -log(-log(x)) — all in float64
    x.log_().clamp_(min=torch.finfo(x.dtype).min).neg_()  # -log(x)
    x.log_().neg_()                                        # -log(-log(x)) = Gumbel noise

    x.add_(logprobs.to(torch.float64))                     # Gumbel-max trick

    return torch.argmax(x, dim=1, keepdim=True)             # deterministic argmax
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ **Gumbel-max trick** is the gold standard for deterministic categorical sampling:
1. Hash seed+position → uniform random value
2. Apply Gumbel transform: -log(-log(x))
3. Add Gumbel noise to logprobs
4. argmax = deterministic (no randomness in argmax!)
5. All in float64 to prevent numerical instability

★★★★★★★★★★★ **GRPO Impact**: For GRPO group_size=8, set sampling_seed=[42,43,44,45,46,47,48,49]. Each seed produces a different but reproducible rollout. Across runs, the same seed always produces the same output. This eliminates inference-induced variance in GRPO advantage estimation.

---

## 5. DeepGEMM SM89 Fallback Analysis

### ★★★★★★★★★★ DeepGEMM is NOT available on SM89 (RTX 4090). Triton persistent matmul is the ONLY path.

### 5.1 SM Version Gate

**File**: `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py`

```python
def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False  # ← SM89 = False! RTX 4090 blocked!
    if sm_version == 120:
        return False  # ← RTX 5090 also blocked! (no TMEM/tcgen05)
```

★★★★★★★★★★ **RTX 4090 (SM89) AND RTX 5090 (SM120) both cannot use DeepGEMM**:
- SM89: below SM90 minimum (no TMA hardware)
- SM120: consumer GPU, no TMEM/tcgen05 (datacenter-only feature)

### 5.2 DeepGEMM Legacy Module (SM80/SM89 Triton Kernels)

**File**: `deep_gemm/legacy/` in the DeepGEMM repo

The DeepGEMM repo includes a `legacy` module for A100 (SM80) GPUs:
```python
# deep_gemm/legacy/__init__.py
from .m_grouped_gemm import *
from .a_fused_m_grouped_gemm import *
from .a_fused_k_grouped_gemm import *
from .b_fused_k_grouped_gemm import *
```

★★★★★ **Legacy kernels use `@triton.autotune`** — NOT batch-invariant! They have autotuned configs which would introduce batch-dependent behavior. This is why SGLang's batch_invariant_ops does NOT use DeepGEMM legacy — it uses the custom `matmul_kernel_persistent` with constexpr BLOCK_SIZE instead.

Example from legacy m_grouped_gemm.py:
```python
@triton.autotune(configs=get_m_grouped_gemm_configs(), key=[])  # ← AUTOTUNED!
@triton.jit
def m_grouped_bf16_gemm_contiguous_tl_impl(a_ptr, b_ptr, d_ptr, ...):
```

### 5.3 Triton Persistent Matmul vs DeepGEMM Comparison

| Feature | DeepGEMM (SM90+) | Triton Persistent (SM89) |
|----------|------------------|--------------------------|
| **Data Loading** | TMA async descriptors (`tl.experimental_descriptor_load`) | Standard `tl.load` |
| **BLOCK Sizes** | Autotuned/configured per SM | `tl.constexpr` — FIXED |
| **Persistent Mode** | Full persistent with TMA preload | Persistent with `tl.range` loop |
| **Determinism** | Batch-invariant (TMA + fixed configs) | Batch-invariant (constexpr) |
| **Performance** | ~2.7x over cuBLAS on Hopper | Lower than DeepGEMM but deterministic |
| **SM89 Availability** | **NOT available** | **YES — primary path** |

★★★★★★★★★★ **The Triton persistent matmul in batch_invariant_ops is specifically designed for determinism, not maximum performance.** On SM89, this is acceptable because:
1. RTX 4090 has 128 SMs — fewer than H100 (132) but sufficient for persistent kernels
2. The configs use reasonable block sizes (128x128x64 for BF16)
3. Determinism is more important than raw throughput for GRPO training

---

## 6. RTX 4090 GRPO Training Integration Summary

### ★★★★★★★★★★★★★★★★★★ RTX 4090 Recommended SGLang Configuration for GRPO

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend triton \
    --enable-deterministic-inference \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096
```

**Why Triton backend**:
- Supports radix cache + CUDA graphs + chunked prefill + non-greedy sampling
- FlashInfer deterministic does NOT support radix cache → prefix reuse lost for GRPO!
- FA3 works but Triton is more widely tested for deterministic inference

**GRPO-specific configuration**:
```python
# In verl/rLLM GRPO training loop:
sampling_seeds = [42 + i for i in range(group_size)]  # e.g., [42, 43, 44, ...]
for seed in sampling_seeds:
    response = requests.post("http://localhost:30000/generate", json={
        "text": prompt,
        "sampling_params": {
            "temperature": 0.8,
            "max_new_tokens": 128,
            "sampling_seed": seed,  # ← Reproducible diverse rollouts
        },
    })
```

### 6.1 SGLang vs vLLM Deterministic Inference Comparison

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Architecture level** | KERNEL level (Triton persistent constexpr) | COMPILE level (Inductor fusion guard) |
| **Root cause addressed** | Direct — replaces all non-deterministic ops with fixed-block Triton kernels | Indirect — tries to prevent Inductor from fusing RMSNorm |
| **Ops replaced** | mm, addmm, _log_softmax, mean.dim, rms_norm, mm.dtype, bmm (7 ops) | Only mean (1 op via torch.mean override) |
| **RMSNorm solution** | Standalone Triton kernel with constexpr BLOCK_SIZE | Inductor fusion guard (still OPEN, not merged) |
| **Attention** | Triton/FA3 backend with fixed split_tile_size | FlashInfer with autotuned tile sizes |
| **Sampling** | murmur_hash32 → Gumbel-max (float64) | No deterministic sampling equivalent |
| **Radix cache + deterministic** | YES (Triton/FA3 backend) | NO (FlashInfer incompatible) |
| **GRPO prefix reuse** | Token-level radix tree prefix sharing | Block-level APC prefix sharing |
| **SM89 support** | Full (Triton persistent matmul works on all SM) | Partial (Inductor fusion guard needed, not merged) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ **CONCLUSION**: SGLang's deterministic inference is architecturally superior for GRPO training on RTX 4090 because:
1. It addresses batch invariance at the KERNEL level (7 aten overrides with constexpr BLOCK_SIZE) — more complete than vLLM's compile-level approach (1 op override + pending Inductor fusion guard)
2. It provides deterministic sampling (murmur_hash32 + Gumbel-max) — no equivalent in vLLM
3. It supports radix cache prefix reuse WITH deterministic inference — vLLM's FlashInfer deterministic doesn't support prefix caching
4. The Triton persistent matmul works on ALL SM versions including SM89 — no SM version gate

### 6.2 Strategic Implications

★★★★★★★★★★ ★★ **Tier 1 insight**: SGLang's deterministic inference already solves the problem that our proposed Inductor SM<90 Fusion Guard PR is trying to address. However:
- The Fusion Guard PR is still valuable for vLLM users who DON'T use SGLang
- SGLang's approach is a different architectural layer — it's a runtime aten override, not a compile-time guard
- Both approaches are complementary, not competing

★★★★★★★★★★★ ★★ **RTX 4090 GRPO ranking update**: SGLang Triton deterministic + radix cache + murmur_hash sampling is now a viable alternative to rLLM Tinker for GRPO training. However:
- SGLang is a pure inference engine — no training loop (need verl/rLLM for training)
- rLLM Tinker is still #1 for in-process training efficiency
- SGLang + verl is a valid #2 option with deterministic inference guarantees

---

## Key Source Files Reference

| File | Path | Key Content |
|------|------|-------------|
| batch_invariant_ops.py | `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` | 7 aten overrides, all Triton persistent kernels |
| batch_invariant_ops/__init__.py | `python/sglang/srt/batch_invariant_ops/__init__.py` | Public API exports |
| server_args.py | `python/sglang/srt/server_args.py` | enable_deterministic_inference, rl_on_policy_target, _handle_deterministic_inference |
| scheduler.py | `python/sglang/srt/managers/scheduler.py` | 4145-line monolithic scheduler, init_deterministic_inference_config |
| radix_attention.py | `python/sglang/srt/layers/radix_attention.py` | RadixAttention nn.Module wrapper |
| radix_tree.py | `python/sglang/srt/mem_cache/cpp_radix_tree/radix_tree.py` | JIT-compiled C++ radix tree binding |
| tree_v2.h | `python/sglang/srt/mem_cache/cpp_radix_tree/tree_v2.h` | C++ radix tree API (match_prefix, evict, lock_ref, writing_through) |
| base_prefix_cache.py | `python/sglang/srt/mem_cache/base_prefix_cache.py` | MatchResult, BasePrefixCache ABC |
| sampler.py | `python/sglang/srt/layers/sampler.py` | multinomial_with_seed, Sampler class |
| hash.py | `python/sglang/srt/layers/utils/hash.py` | murmur_hash32 Triton kernel |
| sampling_batch_info.py | `python/sglang/srt/sampling/sampling_batch_info.py` | sampling_seed tensor creation |
| triton_backend.py | `python/sglang/srt/layers/attention/triton_backend.py` | TritonAttnBackend, split_tile_size=256 |
| prefill_attention.py | `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py` | Triton prefill kernel with constexpr BLOCK_M/N |
| decode_attention.py | `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` | Triton decode kernel with constexpr BLOCK_N |
| configurer.py | `python/sglang/srt/layers/deep_gemm_wrapper/configurer.py` | SM89 DeepGEMM gate (returns False) |
| deterministic_inference.md | `docs/advanced_features/deterministic_inference.md` | User-facing documentation |
