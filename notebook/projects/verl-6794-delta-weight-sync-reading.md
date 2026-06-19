# verl #6794 — Delta Weight Sync (Bytewise Diff Encoding)

> 2026-06-19 | OPEN (blocked) | +1110/-7 lines, 9 files | Author: ChangyiYang
> Branch: feat/delta-weight-sync-sglang | Base: main
> ★★★★★★★★ ~100x payload reduction via bytewise diff encoding — dtype-agnostic, lossless, bit-identical
> ★★★★★★★★ 3 encoding formats: indices (int32 absolute), deltas (uint16 gap), deltas_zstd (gap+zstd)
> ★★★★★★★★ RTX 4090 HYBRID mode: limited benefit (in-process IPC) — LoRA delta deferred explicitly
> ★★★★★★★★ SGLang-only; vLLM deferred pending #31848/#39451 sparse weight-update API
> ★★★★★★★★ 4 review issues (2 CRITICAL): missing record_stream, TP>1 disk race+incomplete file list, big_values concat overhead, makedirs race
> ★★★★★★★★ UPDATED June 19: Added RTX 4090 OOM risk analysis for big_values, cross-framework #8061 connection, sleep/wake interaction, LoRA delta future pathway

---

## 1. Full PR Description

Title: `[rollout][sglang] feat: delta weight sync (sparse trainer->rollout updates)`

**Motivation**: In RL post-training at typical learning rates, >99% of BF16 weight bytes are unchanged step-over-step. This is confirmed by PULSE (arXiv:2502.03839), SparrowRL (2602.11456), and Fireworks "Frontier RL Is Cheaper Than You Think". Sending only the changes drops the wire payload by ~100x and makes disaggregated trainer/rollout viable over commodity networks and shared-FS cross-DC setups.

**Design lineage**: Mirrors THUDM/slime's `update_weight_from_distributed_delta.py` design, pairs with SGLang receiver in sgl-project/sglang#26519.

**Key design points (from PR body)**:
- Bytewise diff is dtype-agnostic and arithmetic-free (`current.view(int) != snapshot.view(int)`), so apply is lossless/bit-identical — no per-step drift, no periodic re-syncs needed
- Snapshot is full-coverage, lives in pinned host memory — purely additive resource cost; the trade is host RAM for cheaper per-step communication
- `delta` does NOT replace RDMA — it stacks with it: `delta + nccl` runs over IB/RoCE just like full broadcast, shrinking payload on top; `delta + disk` is cross-DC transport where RDMA isn't available
- CUDA-optional: snapshot and pipelining gracefully degrade to synchronous host copies on CPU, unit tests run without GPU
- Default behavior unchanged: `weight_mode` defaults to `"full"`, PR is no-op for existing configs

**Explicitly deferred (out of scope)**:
- vLLM rollout path (waiting on vllm-project/vllm#31848 / #39451 for native sparse weight-update API)
- TRTLLM rollout path
- LoRA path (`peft_config + base_sync_done`): orthogonal; base sync stays full, adapter sync could layer on later as delta
- Lossy delta (dropping changes below a threshold)
- End-to-end CI exercising SGLang delta path — depends on delta-capable SGLang build being available to CI

**AI assistance disclosure**: Core algorithms ported from THUDM/slime and SGLang #26519, with adaptations to verl's generator-based weight-sync API. Prepared with Claude assistance; human submitter reviewed and accepts accountability.

---

## 2. Architecture: File Layout and Data Flow

### 2.1 New files (+1290 lines)

```
verl/workers/rollout/delta_sync/              # engine-agnostic core
  __init__.py                                  (47 lines)  — exports
  delta_state.py                               (182 lines) — pinned-CPU snapshot + D2H/H2D side streams
  encode.py                                    (384 lines) — bytewise diff + indices/deltas/deltas_zstd + DeltaBucket + decode
  wrapper.py                                   (168 lines) — iter_delta_flushes(): adapts (name,tensor) gen -> DeltaFlush stream

verl/workers/rollout/sglang_rollout/
  delta_dispatch.py                            (164 lines) — SGLang-only; defensive import of DeltaSpec/DeltaParam/DeltaEncoding from sglang io_struct; NCCL + disk dispatchers

verl/workers/config/rollout.py                 (29 lines)  — new CheckpointEngineConfig fields
verl/trainer/config/rollout/rollout.yaml       (25 lines)  — matching defaults

tests/workers/rollout/test_delta_sync.py       (180 lines) — round-trip bit-identity tests for both encodings
```

### 2.2 Modified files (+111/-7 lines)

```
verl/workers/rollout/sglang_rollout/sglang_rollout.py  — update_weights() branches on checkpoint_engine.weight_mode
```

### 2.3 Data flow

```
Trainer worker                           Rollout worker (SGLang)
    │                                         │
    │  Step 1: seed (first call)              │  (no RPC — assumes same init checkpoint)
    │  DeltaState.seed(named_tensors)         │
    │                                         │
    │  Step N: subsequent sync                │
    │  1. prefetch_snapshot (H2D side stream) │
    │  2. compute_diffs (bytewise diff)       │
    │  3. encode_chunk (indices/deltas)       │
    │  4. update_snapshot_async (D2H side)    │
    │  5. bucket into DeltaFlush              │
    │                                         │
    │  ── NCCL broadcast ──>                  │  _apply_delta_from_distributed()
    │     positions + values + DeltaSpec      │  → checksum verify
    │                                         │  → decode positions (indices or gap-deltas)
    │                                         │  → index_copy_ into model params
    │                                         │
    │  ── disk (safetensors) ──>              │  _apply_delta (read from shared FS)
    │     per-rank .safetensors files         │  → decode + apply with chunk cap
```

---

## 3. SGLang-Only: Why Not vLLM?

### 3.1 Direct dependency on SGLang receiver API

The PR explicitly states: "SGLang is the only rollout engine targeted in this PR; vLLM is deferred until vllm-project/vllm#31848 / #39451 lands a native sparse weight-update API."

The reason is architectural, not preference:
- **SGLang** has `update_weights_from_distributed()` with a `load_format` parameter that can accept `"delta"` — this is the receiver API added in #26519
- **vLLM** has no equivalent sparse weight-update API. Its `update_weights` always expects full parameter tensors. vLLM issues #31848 and #39451 are tracked for adding this capability, but they are not merged
- The `delta_dispatch.py` module defensively imports `DeltaSpec`, `DeltaParam`, `DeltaEncoding` from `sglang.srt.managers.io_struct` — these are SGLang-specific data structures that define the on-wire protocol

### 3.2 Why SGLang was chosen first

1. SGLang's `rl_on_policy_target` feature has existing hooks for weight sync in RL training loops
2. SGLang's `update_weights_from_distributed()` already supports NCCL group-based weight broadcast — delta piggybacks on this existing transport
3. The THUDM/slime project (which this PR ports from) already targeted SGLang as the rollout engine
4. SGLang's model_runner.py has a clean `load_weights()` interface that can accept partial tensors with NaN masking (the `_delta_apply_context` patches `copy_` and `fill_` to skip NaN positions)

### 3.3 vLLM pathway (future)

When vLLM #31848/#39451 merge, the engine-agnostic core (`delta_sync/`) can be reused — only a new `vllm_rollout/delta_dispatch.py` would need to be written, analogous to the SGLang one. The encode/decode format is independent of the rollout engine.

---

## 4. Bytewise Diff Encoding: dtype-agnostic, lossless

### 4.1 Core algorithm: `bytewise_diff_mask`

```python
def bytewise_diff_mask(current: torch.Tensor, snapshot: torch.Tensor) -> torch.Tensor:
    """Per-element bool mask: True where current and snapshot differ.
    Dtype-agnostic via view-as-integer; no arithmetic."""
    es = current.element_size()
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(es)
    return current.view(int_dtype) != snapshot.view(int_dtype)
```

**Key properties**:
- **dtype-agnostic**: Works for ANY dtype — uint8, fp16, bf16, fp32, fp64, int8, fp8, MXFP4. The comparison happens on the integer view of the raw bytes, so no floating-point arithmetic is involved
- **Lossless**: Since the comparison is at the byte/bit level, the reconstruction is bit-identical. The receiver gets EXACTLY the same bytes as the trainer. No rounding, no per-step drift
- **No periodic re-sync needed**: Unlike lossy approaches where cumulative drift requires periodic full syncs, bytewise diff guarantees zero drift. This is stated explicitly in the PR
- **Efficient**: View-as-integer comparison is a single GPU operation, no host sync needed for the diff itself. The mask is a bool tensor marking changed positions

### 4.2 Why byte-level (not float-level)?

Float-level diff (`current - snapshot != 0`) has problems:
- bf16 rounding: two tensors with the same bf16 values can differ in their raw bytes due to NaN vs 0, or different bit patterns for identical values
- fp8/MXFP4: element-level comparison requires per-dtype logic; byte-level is universal
- The PR explicitly states: "Dtype-agnostic via view-as-integer; no arithmetic" — this is the design principle

### 4.3 Snapshot lifecycle

```
DeltaState lifecycle:
  1. First sync: seed(snapshot) from current model weights → full D2H copy, no engine RPC
  2. Each subsequent sync:
     a. prefetch_snapshot (H2D side stream, overlapped with previous chunk compute)
     b. compute_diffs (bytewise diff on GPU, after waiting on prefetch event)
     c. encode_chunk (positions + values)
     d. update_snapshot_async (D2H side stream, writes new values back to pinned host buffer)
     e. flush_snapshot (blocks until all D2H copies land) — called at end of iter_delta_flushes
```

The snapshot lives in **pinned host memory** — this is a deliberate trade: host RAM for cheaper per-step communication. On CUDA machines, `pin_memory=True` enables async DMA; on CPU-only (CI/tests), it falls back to blocking copies.

---

## 5. Three Encoding Formats

### 5.1 Wire format: shared layout

All three encodings share one on-wire layout: a `uint8` positions blob (`__positions__`) plus a parameter-dtype values tensor (`__values__`) with a per-parameter manifest (`DeltaParam`). Decoders dispatch on metadata.

The `DeltaParam` dataclass:
```python
@dataclass
class DeltaParam:
    name: str           # HF parameter name
    dtype: str          # e.g. "bfloat16"
    shape: list[int]    # original tensor shape
    pos_start: int      # byte offset into __positions__ blob
    pos_end: int        # byte end offset
    pos_width: int      # 2 (uint16) or 4 (int32/uint32)
    val_start: int      # element offset into __values__ tensor
    val_end: int        # element end offset
```

### 5.2 `indices` format

- **Position encoding**: int32 absolute positions (4 bytes per changed element)
- **Value encoding**: raw dtype values (2 bytes per bf16 element)
- **Wire bytes** (test data): 48 B for 8 changed BF16 elements
- **Pros**: Lowest compute, simplest decode — `idx = positions.view(int32)`, then `flat.index_copy_(0, idx, values)`
- **Cons**: Largest wire size — 4 bytes per position vs 2 bytes for uint16 deltas
- **Best for**: Local IPC / NCCL where bandwidth is abundant and compute savings matter more

### 5.3 `deltas` format (gap encoding)

- **Position encoding**: uint16 gap-deltas with uint32 per-parameter fallback
  - Store `idx[k] - idx[k-1] - 1` with `idx[-1] := -1` so first delta = first index
  - Each parameter downcasts to uint16 if max gap <= 65535, else uint32
  - At ~2% density on bf16 weights, typical max gap is ~300, so uint16 normally suffices
  - Receiver inverts via `idx = cumsum(delta + 1) - 1`
- **Wire bytes** (test data): 32 B for 8 changed BF16 elements (vs 48 B for indices)
- **Pros**: Smaller wire size than indices (usually 2 bytes/position vs 4)
- **Cons**: Slightly more compute (cumsum inversion on receiver)
- **Best for**: Network/disk transport where wire size matters

### 5.4 `deltas_zstd` format

- **Position encoding**: same gap-delta stream as `deltas`, then wrapped in zstd compression at safetensors write time
- **The zstd wrap is applied by the disk transport, NOT in encode.py** — the encode module produces the same uncompressed gap stream for both `deltas` and `deltas_zstd`; zstd is applied at `write_flush_to_disk()` via safetensors' built-in compression
- **On receiver side**: `_maybe_zstd_decompress()` checks for zstd frame magic (0xFD2FB528) and decompresses before parsing
- **Pros**: Smallest wire size — best for cross-DC / cross-region transport over shared FS
- **Cons**: Highest compute (zstd decompress + cumsum inversion); NCCL transport doesn't benefit from zstd since positions are already on GPU
- **Best for**: `weight_transport="disk"` cross-DC scenarios

### 5.5 Encoding format comparison table

| Format       | Pos encoding          | Bytes/pos | Wire (8 nnz bf16) | Compute  | Best transport     |
|--------------|----------------------|-----------|--------------------|----------|--------------------|
| indices      | int32 absolute       | 4         | 48 B               | Lowest   | NCCL local         |
| deltas       | uint16 gap (uint32 fallback) | 2-4 | 32 B           | Medium   | NCCL / disk local  |
| deltas_zstd  | deltas + zstd wrap   | ~1-2*     | <32 B (compressed) | Highest  | disk cross-DC      |

*zstd compression ratio depends on data; gap-encoded uint16 deltas are highly compressible.

### 5.6 Checksum mechanism

Sender computes: `checksum(positions_gpu, values_gpu)` using `torch.hash_tensor` (XOR-reduce over uint64 bitcast). This is a single GPU reduction plus one `.item()` sync. Receiver verifies checksum matches — detects corruption between encode and apply.

---

## 6. ~100x Payload Reduction Claims

### 6.1 Theoretical basis

At typical RL learning rates (1e-5 to 1e-6), weight updates are tiny. For bf16:
- Most weight elements change by <1 ULP (unit in last place) — the bf16 value doesn't change at all
- Only ~0.5-2% of bf16 bytes actually differ between steps
- This is confirmed by PULSE (2502.03839), SparrowRL (2602.11456), and Fireworks production data

### 6.2 Calculation for specific models

**Qwen3-30B-A3B (bf16)**:
- Full weight payload: ~30B params * 2 bytes = ~60 GiB per sync
- Delta at ~1% change: ~600 MiB (positions + values)
- Reduction: 60 GiB / 600 MiB = ~100x

**Qwen2.5-7B (bf16)**:
- Full weight payload: ~7B params * 2 bytes = ~14 GiB
- Delta at ~1% change: ~140 MiB
- Reduction: ~100x

**With `deltas` gap encoding**:
- Positions: ~1% of elements * 2 bytes/pos (uint16) = very small
- Values: ~1% of elements * 2 bytes/val (bf16) = dominant term
- Total delta payload ≈ nnz * (2 + 2) bytes ≈ 4 * nnz bytes
- vs full payload ≈ total * 2 bytes
- Reduction ≈ total/(2*nnz) * (2/4) = approximately (total/nnz) * 0.5 ≈ 50-100x depending on density

### 6.3 Caveats

- 100x is at typical RL learning rates; at higher rates (e.g. pre-training), more bytes change and reduction drops
- First sync is always full (seed) — no delta savings
- LoRA-only training: LoRA parameters are small enough that delta encoding has minimal additional benefit over full LoRA transfer
- The PR description says "~100x" as a typical claim, not a guarantee for all scenarios

---

## 7. RTX 4090 HYBRID Mode Implications

### 7.1 Why limited benefit on RTX 4090 HYBRID

The PR config comment explicitly states:
> "Incompatible with CUDA-IPC colocate transports since IPC only crosses a memory handle — delta would have no bytes to save there."

RTX 4090 HYBRID mode (in-process, single GPU):
- Training and rollout share the same GPU process
- Weight sync uses Python generator `(name, tensor)` — zero-copy, no serialization
- CUDA-IPC (colocate) transport: passes memory handles, not bytes — delta encoding has nothing to compress
- Delta sync adds overhead (snapshot allocation, diff computation, encoding) with minimal payload savings
- The snapshot itself costs host RAM = full model size in bf16 (~14 GiB for 7B) — significant on a 24 GiB GPU machine with only ~32-64 GiB system RAM

### 7.2 Where delta DOES benefit RTX 4090

Theoretical scenario: disaggregated training with separate trainer and rollout machines:
- Trainer on one machine, rollout RTX 4090 on another
- Network IPC between machines
- Delta reduces network payload from ~14 GiB to ~140 MiB → fits in reasonable transfer time
- But this requires multi-machine setup, not HYBRID mode

### 7.3 Practical RTX 4090 assessment

| Scenario          | Delta benefit | Reason                           |
|-------------------|---------------|----------------------------------|
| HYBRID in-process | None/negative | CUDA-IPC = memory handle, not bytes |
| NCCL local        | Low           | Same-machine NCCL is fast anyway |
| Disk cross-machine| High          | Network bandwidth = bottleneck   |
| Multi-node NCCL   | Medium-High   | IB/RoCE bandwidth savings       |

**Bottom line for RTX 4090**: Delta sync is primarily beneficial for disaggregated/multi-node setups. On a single RTX 4090 HYBRID, it adds overhead without payload savings. The config default `"full"` is the correct choice for HYBRID mode.

---

## 8. LoRA Delta Deferred Mechanism

### 8.1 Explicitly deferred in PR scope

From the PR body: "LoRA path (`peft_config + base_sync_done`): orthogonal; can layer on later (base sync stays full, adapter sync could be delta)."

### 8.2 Why LoRA delta is deferred

1. **LoRA params are already small**: LoRA A/B matrices for rank=16 on a 7B model are ~2-3 MiB total. Full transfer is already cheap. Delta encoding saves very little on top of a small payload.
2. **LoRA changes are large per step**: Unlike base model weights (which change ~1% of bytes), LoRA weights update significantly each step — more bytes change, reducing delta's compression ratio.
3. **Orthogonal design**: Base model delta sync and LoRA delta sync are independent features. The PR focuses on the base model case first, which has the largest payload reduction potential.
4. **verl #6512 context**: Per-unit LoRA summon already reduces peak memory 10x (60 -> 6-8 GiB). LoRA delta sync would be an additional optimization on top of this, but the marginal benefit is small on HYBRID mode.

### 8.3 Future LoRA delta pathway

When LoRA delta is eventually implemented:
- Base model: stays `"full"` for first sync, then `"delta"` for subsequent steps
- LoRA adapter: could use `"delta"` as well, but likely `"full"` is sufficient since LoRA payload is small
- MoE LoRA: larger total LoRA params → delta encoding may have more benefit
- This aligns with the AutoEP + LoRA pathway (DeepSpeed #8064)

---

## 9. Four Review Issues (2 CRITICAL)

All 4 review issues are from `gemini-code-assist[bot]` (automated code review). No human review comments yet — the PR is draft/RFC status, 1 general comment, 4 review comments.

### 9.1 CRITICAL Issue 1: Missing `record_stream` on async D2H copies

**Location**: `verl/workers/rollout/delta_sync/delta_state.py`, line 171 (`update_snapshot_async`)

**Problem**: When copying tensors asynchronously on a non-default CUDA stream (`self.d2h_stream`), `tensor.record_stream(self.d2h_stream)` must be called. Without it, PyTorch's caching allocator may reclaim or reuse the tensor's memory before the copy operation completes, causing **silent data corruption** in the snapshot. Since the generator may yield new tensors or FSDP may free/reuse gathered parameter buffers in subsequent steps, omitting `record_stream` is dangerous.

**Severity**: CRITICAL — silent data corruption is the worst kind of bug. The snapshot would contain wrong values, and subsequent delta diffs would be computed against corrupted data, propagating errors through the entire weight sync pipeline.

**Fix**: Add `tensor.record_stream(self.d2h_stream)` after each `copy_()` call:
```python
with torch.cuda.stream(self.d2h_stream):
    self.d2h_stream.wait_event(event)
    for name, tensor in named_tensors:
        self._allocate(name, tensor)
        self.snapshot[name].copy_(tensor.detach(), non_blocking=True)
        tensor.record_stream(self.d2h_stream)  # <-- MISSING, must add
```

### 9.2 CRITICAL Issue 2: TP>1 disk transport race condition + incomplete file list

**Location**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`, line 452 (`_update_weights_delta` disk path)

**Problem**: Two sub-issues:
1. **Race condition**: The leader rank might call `dispatch_disk_files()` and delete `version_dir` before other ranks finish writing their files, or before SGLang finishes reading them. A `torch.distributed.barrier()` is required to synchronize all ranks.
2. **Incomplete file list**: `pending_files` is local to each rank, so the leader only dispatches its own files. SGLang ranks >0 will fail to load their corresponding delta files. The leader should reconstruct the complete deterministic list of files for all ranks based on TP size and `flushes_emitted`.

**Severity**: CRITICAL — in any TP>1 deployment, this causes either data loss (deleted before read) or missing files (other ranks' files never dispatched). TP>1 is the norm for multi-GPU inference.

**Fix**: Add barrier before dispatch, and reconstruct complete file list:
```python
if transport == "disk":
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # <-- wait for all ranks to finish writing
    if self._is_server_tp_leader() and flushes_emitted > 0:
        tp_size = (self.device_mesh["infer_tp"].size()
                   if "infer_tp" in self.device_mesh.mesh_dim_names else 1)
        all_files = [f"rank{r:04d}_flush{f:06d}.safetensors"
                     for r in range(tp_size) for f in range(flushes_emitted)]
        await dispatch_disk_files(self._engine, out_dir=version_dir,
                                  files=all_files, weight_version=self._delta_version)
        if not getattr(ce, "delta_keep_files", False):
            import shutil
            shutil.rmtree(version_dir, ignore_errors=True)
```

### 9.3 HIGH Issue 3: `big_values` concat memory overhead

**Location**: `verl/workers/rollout/delta_sync/encode.py`, line 255 (`_sparse_boundaries`)

**Problem**: `_sparse_boundaries()` concatenates all parameter values via `torch.cat([d.values.contiguous().view(-1) for d in diffs])` into a massive temporary tensor `big_values`. This allocates a tensor containing ALL parameter elements (including unchanged ones) for every chunk. For large models, this is significant GPU memory overhead.

**Severity**: HIGH — not correctness-breaking but significant memory overhead. For a 30B model, `big_values` would temporarily hold all 30B elements in a flat tensor on GPU, even though only ~1% are needed.

**Suggested fix**: Instead of concatenating all values and then indexing, index into each `d.values` individually using `local_idx` on the GPU, avoiding the `big_values` allocation entirely. The reviewer provided a full rewrite of `_encode_indices` and `_encode_deltas` that avoids `big_values`.

**Assessment**: This is a valid optimization concern. The current implementation works correctly but has unnecessary memory overhead. The suggested fix is more memory-efficient but adds complexity. This should be addressed before merge for production use on large models.

### 9.4 HIGH Issue 4: `makedirs` race in `write_flush_to_disk`

**Location**: `verl/workers/rollout/sglang_rollout/delta_dispatch.py`, line 130

**Problem**: In distributed environments, non-leader ranks might attempt to write flush files to `out_dir` before the leader rank has finished creating it, leading to `FileNotFoundError`.

**Severity**: HIGH — not silent corruption (it would crash), but a reliability issue in TP>1 disk transport.

**Fix**: Add `os.makedirs(out_dir, exist_ok=True)` inside `write_flush_to_disk()` before writing:
```python
from safetensors.torch import save as st_save_bytes
os.makedirs(out_dir, exist_ok=True)  # <-- add this
metadata = {"encoding": flush.encoding, ...}
```

### 9.5 Summary of review issues

| #  | Severity | File                    | Issue                              | Impact                    |
|----|----------|------------------------|------------------------------------|---------------------------|
| 1  | CRITICAL | delta_state.py:171     | Missing `record_stream`            | Silent data corruption    |
| 2  | CRITICAL | sglang_rollout.py:452  | TP>1 disk race + incomplete files  | Data loss, missing files  |
| 3  | HIGH     | encode.py:255          | `big_values` concat overhead        | GPU memory waste          |
| 4  | HIGH     | delta_dispatch.py:130  | `makedirs` race                    | FileNotFoundError crash   |

**Status**: All 4 issues are from automated review (gemini-code-assist). No human reviews yet. PR is draft/RFC. Issues 1 and 2 must be fixed before any production use; issues 3 and 4 are quality improvements that should be addressed before merge.

---

## 10. SGLang #26519 Dependency (Receiver Side)

### 10.1 PR details

- **Repository**: sgl-project/sglang
- **Title**: "Add delta weight update receiver"
- **Author**: nanjiangwill
- **State**: OPEN
- **Branch**: feat/delta-weight-sync-receiver
- **Companion**: radixark/miles#1235
- **Size**: +339/-4 lines, 8 files
- **Created**: 2026-05-28

### 10.2 Files changed

```
python/sglang/srt/entrypoints/engine.py           (+4)   — expose delta update API
python/sglang/srt/managers/io_struct.py           (+35)  — DeltaSpec, DeltaParam, DeltaEncoding
python/sglang/srt/managers/tokenizer_manager.py   (+2/-2) — delta request routing
python/sglang/srt/managers/tp_worker.py           (+2)   — delta worker dispatch
python/sglang/srt/model_executor/model_runner.py  (+280/-2) — _apply_delta, _apply_delta_from_distributed, _decode_delta_one_param, _delta_apply_context, _delta_checksum, _maybe_zstd_decompress
python/sglang/srt/server_args.py                  (+14)  — delta chunk bytes + read workers config
python/sglang/srt/speculative/eagle_worker_v2.py  (+1)
python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py (+1)
```

### 10.3 Key receiver-side data structures

```python
class DeltaEncoding(str, Enum):
    INDICES = "indices"
    DELTAS = "deltas"
    DELTAS_ZSTD = "deltas_zstd"

@dataclass
class DeltaParam:
    name: str; dtype: str; shape: List[int]
    pos_start: int; pos_end: int; pos_width: int
    val_start: int; val_end: int

@dataclass
class DeltaSpec:
    encoding: DeltaEncoding
    params: List[DeltaParam]
    checksum: int = 0
```

These mirror verl's `DeltaParam` exactly — the on-wire protocol is defined symmetrically.

### 10.4 Receiver decode logic: `_decode_delta_one_param`

```python
def _decode_delta_one_param(self, encoding, positions, values, param):
    numel = math.prod(param.shape)
    flat = torch.full((numel,), float("nan"), dtype=param_dtype, device=self.device)
    val_slice = values[param.val_start : param.val_end]
    pos_bytes = positions[param.pos_start : param.pos_end]

    # Decode positions based on encoding
    if encoding is DeltaEncoding.INDICES:
        width = 4  # int32 absolute positions
    elif encoding in (DeltaEncoding.DELTAS, DeltaEncoding.DELTAS_ZSTD):
        width = param.pos_width  # 2 (uint16) or 4 (uint32) gap-deltas

    # Unpack byte-packed positions into int64 indices
    byte_view = pos_bytes.view(n_elems, width).to(torch.int64)
    if width == 2:
        unpacked = byte_view[:, 0] | (byte_view[:, 1] << 8)
    else:  # width == 4
        unpacked = byte_view[:, 0] | (byte_view[:, 1] << 8) | (byte_view[:, 2] << 16) | (byte_view[:, 3] << 24)

    # Invert gap-delta encoding: idx = cumsum(delta + 1) - 1
    if encoding is DeltaEncoding.INDICES:
        idx = unpacked  # already absolute positions
    else:
        idx = (unpacked + 1).cumsum(dim=0) - 1

    # Scatter values into full-shape tensor (NaN = unchanged)
    flat.index_copy_(0, idx, val_slice.to(param_dtype))
    return flat.view(tuple(param.shape))
```

### 10.5 `_delta_apply_context`: NaN-safe weight patching

The receiver applies delta weights via `model.load_weights(chunk)` where each tensor has NaN at unchanged positions. To prevent NaN from overwriting good values, `_delta_apply_context` patches `torch.Tensor.copy_` and `torch.Tensor.fill_`:

```python
def patched_copy_(self, src, *args, **kwargs):
    if is_param_target(self) is not None:
        mask = ~torch.isnan(src_aligned)
        self[mask] = src_aligned[mask]  # only overwrite changed positions
        return self
    return original_copy_(self, src, *args, **kwargs)

def patched_fill_(self, value):
    if is_param_target(self) is not None:
        if math.isnan(value):
            return self  # skip NaN fill — leave param unchanged
    return original_fill_(self, value)
```

This is an elegant approach: rather than modifying `load_weights` to understand deltas, it monkey-patches the fundamental tensor operations for the duration of the apply, making NaN values transparently skipped. The patching scope is limited to model parameters only (via `_param_storage_index`).

### 10.6 Two transport paths on receiver

**NCCL path**: `_apply_delta_from_distributed()` — receives positions and values via NCCL broadcast, applies via `_apply_delta_payload()`. Positions and values are already on GPU after broadcast.

**Disk path**: `_apply_delta()` — reads safetensors files from shared FS, decompresses zstd if needed, decodes and applies. Uses concurrent file reads with configurable `update_weight_delta_read_workers` (default 4).

### 10.7 Checksum verification

Receiver verifies checksum via `_delta_checksum()` (same algorithm as sender):
```python
def _delta_checksum(positions, values):
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)
```

Mismatch raises `RuntimeError` — detects corruption between sender encode and receiver apply.

### 10.8 Blocking dependency

Until #26519 merges into SGLang upstream:
- verl #6794 requires a custom SGLang build vendoring #26519
- The `delta_dispatch.py` uses defensive import: `try: from sglang.srt.managers.io_struct import DeltaSpec... except ImportError: _DELTA_RECEIVER_AVAILABLE = False`
- If delta mode is requested without the receiver, a clear `RuntimeError` is raised with instructions
- End-to-end CI testing is deferred until a delta-capable SGLang build is available in CI

---

## 11. Integration with verl's Existing Architecture

### 11.1 Config integration

New fields on `CheckpointEngineConfig`:
```python
weight_mode: str = "full"            # "full" (default) or "delta"
weight_transport: str = "nccl"       # "nccl" or "disk"
weight_encoding: str = "indices"     # "indices", "deltas", "deltas_zstd"
delta_chunk_megabytes: int = 512     # receiver-side chunk cap
delta_disk_dir: Optional[str] = None # shared FS dir for disk transport
delta_keep_files: bool = False       # keep delta files after apply
```

Matching YAML defaults in `rollout.yaml`.

### 11.2 `update_weights()` branching in sglang_rollout.py

```python
async def update_weights(self, weights, ...):
    ce = self.config.checkpoint_engine
    weight_mode = getattr(ce, "weight_mode", "full")
    if weight_mode == "delta":
        await self._update_weights_delta(weights, update_weights_bucket_bytes, global_steps)
    else:
        # existing full broadcast path
        async for params_batch in get_named_tensor_buckets(weights, ...):
            await sgl_update_weights(...)
```

### 11.3 `_update_weights_delta()` method

Creates `DeltaState` lazily (persists across calls), then iterates through `iter_delta_flushes()`:
- NCCL path: each flush broadcasted via `dispatch_flush_nccl()`
- Disk path: each flush written via `write_flush_to_disk()`, then `dispatch_disk_files()` tells engine to read and apply

---

## 12. Testing

### 12.1 Unit tests (CPU-only, no GPU required)

```
test_first_call_seeds_no_flushes[indices]  PASSED
test_first_call_seeds_no_flushes[deltas]   PASSED
test_round_trip_bit_identical[indices]     PASSED
test_round_trip_bit_identical[deltas]      PASSED
test_no_change_emits_no_flush              PASSED
test_dtype_agnostic_diff_fp32              PASSED
```

These verify:
- First call seeds snapshot, emits no flushes (no engine RPC)
- Both encodings reconstruct changed values to bit-identity
- No-change cycles emit no flushes
- dtype-agnostic diff works for fp32 (not just bf16)

Wire bytes example: 48 B / 32 B for 8 changed BF16 elements (indices / deltas respectively).

### 12.2 Missing: end-to-end SGLang integration test

The PR explicitly acknowledges this gap: "An end-to-end smoke against a real SGLang server with PR #26519 vendored in is not included here — that's the natural next step once the receiver lands upstream and we have a CI-able SGLang build for it."

---

## 13. Status as of June 18, 2026

- **State**: OPEN, draft/RFC
- **Mergeable**: Yes (no conflicts), but blocked (draft status)
- **Reviews**: 1 general comment (from gemini-code-assist summarizing all 4 issues), 4 inline review comments (all from gemini-code-assist, no human reviews yet)
- **Human engagement**: Zero human review comments at this point. The PR was opened on June 18, 2026 and is explicitly marked as draft/RFC to "claim the slot and discuss the design"
- **CI**: No CI runs yet (draft status, no tests triggered)
- **Blocking**: 2 CRITICAL issues must be fixed before merge; SGLang #26519 must merge upstream for complete functionality

---

## 14. Technical Innovation Assessment

### 14.1 Bytewise diff as core innovation

The key innovation is **bytewise diff** rather than float-level diff. This is:
- **dtype-agnostic**: single algorithm works for any tensor dtype
- **lossless**: bit-identical reconstruction, zero drift
- **simple**: `current.view(int) != snapshot.view(int)` — one line, no per-dtype branching
- **practical**: exploits the empirical observation that >99% of bf16 bytes are unchanged per RL step

This approach is borrowed from THUDM/slime but adapted to verl's generator-based API. The port is clean and the adaptation is well-designed.

### 14.2 Encoding format design

The three formats are well-designed for their respective use cases:
- `indices`: simple, fast, for local transport
- `deltas`: compact, for network transport
- `deltas_zstd`: maximum compression, for cross-DC disk transport

The uint16/uint32 adaptive width in `deltas` encoding is clever — at ~2% density, uint16 normally suffices (max gap ~300), with uint32 fallback for pathological inputs. This avoids a separate "width negotiation" protocol.

### 14.3 NaN-masking approach for delta apply

The `_delta_apply_context` monkey-patching approach in SGLang is elegant: rather than modifying `load_weights` to understand deltas, it patches `copy_` and `fill_` to skip NaN values. This makes delta weights transparently compatible with SGLang's existing `load_weights()` path. The scope is limited to model parameters only.

### 14.4 Snapshot + pipelining design

The three-stream pipelining (H2D prefetch, default-stream compute, D2H snapshot update) is well-designed for GPU efficiency. The prefetch overlaps with previous chunk's compute, hiding H2D latency. The CPU-only fallback makes testing practical.

---

## 15. RTX 4090 Practical Summary

| Aspect                  | Impact                          | Recommendation                     |
|-------------------------|---------------------------------|------------------------------------|
| HYBRID in-process       | Delta = overhead, no savings    | Keep weight_mode="full"            |
| Snapshot host RAM cost  | ~14 GiB for 7B bf16 model      | Can't afford on 32 GiB system      |
| LoRA delta              | Deferred, minimal benefit       | Full LoRA transfer is sufficient   |
| Multi-machine NCCL      | High benefit                    | Use deltas/deltas_zstd             |
| Cross-DC disk           | Highest benefit                 | Use deltas_zstd                    |
| per-unit summon (#6512) | 10x memory, independent         | Combined: delta for base + full LoRA |

**RTX 4090 HYBRID verdict**: Delta weight sync is primarily designed for disaggregated multi-node training. On a single RTX 4090 HYBRID, the config should remain `weight_mode="full"` with zero-copy generator — delta adds snapshot overhead without payload savings. The technology is impressive and will be valuable when multi-node setups become available.

---

## 16. RTX 4090 OOM Risk: `big_values` Concatenation (NEW June 19)

★★★★★★★★★ **The `_sparse_boundaries()` function allocates a massive temporary GPU tensor that can OOM on RTX 4090!**

```python
big_values = torch.cat([d.values.contiguous().view(-1) for d in diffs], dim=0)
```

For an 8B model in BF16:
- `d.values` for all parameters = ~16 GiB (the entire model weights)
- `big_values` concatenates ALL values into one flat tensor → another ~16 GiB temporary allocation
- On RTX 4090 (24 GiB), base model weights already consume ~16 GiB
- Adding `big_values` temporary → ~32 GiB total → **OOM!**

★★★★★★★★★ **This is a MUST FIX for RTX 4090 deployment**. The fix (per-parameter indexing) avoids the 16 GiB temporary entirely:

```python
# Instead of big_values = torch.cat([...]):
# Index into each d.values individually:
for i, d in enumerate(diffs):
    b = bounds[i]
    nnz = b - prev_b
    if nnz > 0:
        local_idx = big_idx[prev_b:b] - prev_param_start
        val_pieces.append(d.values.contiguous().view(-1)[local_idx.to(d.values.device)])
```

### RTX 4090 Memory Budget with Delta Sync (8B BF16)

| Component | GPU Memory | Host Memory |
|-----------|-----------|-------------|
| Base model weights | ~16 GiB | — |
| LoRA adapter (rank=32) | ~4 MiB | — |
| Optimizer (CPU_Adam offloaded) | — | ~3.8 GiB |
| Delta snapshot (pinned) | — | ~16 GiB |
| `big_values` temporary (BUG) | ~16 GiB! | — |
| **Total GPU (with bug)** | **~32 GiB → OOM!** | — |
| **Total GPU (fixed)** | **~16-18 GiB ✓** | — |
| **Total Host** | — | ~36-48 GiB |

★★★★★★★★★ Without the `big_values` fix, delta sync is NOT viable on RTX 4090 for any model >4B parameters. The fix MUST be applied before testing.

---

## 17. Cross-Framework Connection: `record_stream` = DeepSpeed #8061 Pattern (NEW June 19)

★★★★★★★★★ **The `record_stream` bug in `update_snapshot_async()` is the SAME root cause as DeepSpeed #8061 (overlap_comm NaN)!**

| Issue | Framework | Root Cause | Symptom |
|-------|-----------|-----------|---------|
| #6794 review CRITICAL-1 | verl | Missing `tensor.record_stream(side_stream)` on async D2H copy | Silent data corruption in snapshot |
| #8061 | DeepSpeed | Multi-stream gradient copy without stream ordering | NaN from race condition |

Both issues share the same **CUDA stream safety** pattern:
1. Async operations on a non-default CUDA stream
2. Missing `record_stream()` or stream synchronization
3. PyTorch caching allocator assumes default stream for lifetime tracking
4. Tensor memory can be reclaimed before async operation completes

★★★★★★★★★ **Lesson**: ANY multi-stream code path in CUDA must include `record_stream()` calls. This applies to:
- verl delta sync D2H/H2D side streams
- DeepSpeed overlap_comm gradient bucket streams
- Megatron ChainedOptimizer side streams
- Any `torch.cuda.Stream()` usage in weight sync, gradient computation, or optimizer step

★★★★★★★★★ **4-layer defense update**: Layer 1 (Framework Safety) should include "verify record_stream on ALL multi-stream code paths". This is a systematic concern across 3 frameworks.

---

## 18. Sleep/Wake Interaction Analysis (NEW June 19)

★★★★★★★★★ **How does delta sync interact with SGLang sleep/wake for GRPO?**

### Current sleep/wake flow (sleep_level=1, LoRA adapter path):

```
Step 1: Trainer trains on batch → LoRA adapter updated
Step 2: Trainer: update_weights() → SGLang wake → receive LoRA adapter tags
Step 3: Rollout generates responses
Step 4: Trainer: SGLang sleep → LoRA adapter unloaded → KV cache preserved
Step 5: Repeat
```

### Delta sync in this flow:

The `DeltaState.snapshot` is seeded from initial model weights and persists across calls. After LoRA update:
- Snapshot stores LAST broadcast state (updated via `update_snapshot_async`)
- Each call computes diff from snapshot → captures ALL accumulated changes
- Even after SLEEP → WAKE → UPDATE_WEIGHTS cycle, the diff should be correct

★★★★★★★★★ **But**: Current delta sync does NOT handle LoRA-only updates. The PR explicitly says: "LoRA path (`peft_config + base_sync_done`): orthogonal — base sync stays full, adapter sync could be delta later."

For RTX 4090 sleep_level=1 (LoRA adapter path):
- **Base model sync**: already one-time, stays resident → delta sync irrelevant
- **LoRA adapter sync**: ~4 MiB per step → delta sync would compress to ~80 KiB at 2% sparsity → minimal additional savings
- **Current delta sync targets FULL parameter updates, NOT LoRA-only updates**

★★★★★★★★★ **RTX 4090 verdict**: Delta sync is NOT beneficial for the optimal sleep_level=1 LoRA adapter path. The LoRA adapter payload (~4 MiB) is already 80x smaller than full model (~16 GiB). Delta compression of LoRA deltas would save ~3.2 MiB per step — negligible compared to the 80x already achieved by sleep_level=1.

### Future LoRA Delta Pathway

When LoRA delta is eventually implemented:
- Delta of LoRA adapter → from ~4 MiB to ~80 KiB (at 2% LoRA sparsity)
- Combined with sleep_level=1 → LoRA adapter tags + delta encoding
- Further reduces per-step payload by ~50x
- But this requires: (1) LoRA delta support in #6794, (2) SGLang delta receiver for LoRA

---

## 19. Updated RTX 4090 Decision Matrix (NEW June 19)

| Config | Payload (8B) | Per-Step | RTX 4090 Rank |
|--------|-------------|----------|---------------|
| sleep_level=1 LoRA full | ~4 MiB | NCCL tags | ★★★★★★★★ #1 BEST |
| sleep_level=1 LoRA delta (future) | ~80 KiB | NCCL tags + delta | ★★★★★★★★ #1+ (future) |
| Full broadcast | ~16 GiB | NCCL full | ★★ (avoid) |
| Full broadcast delta (after fixes) | ~0.16-0.32 GiB | NCCL delta | ★★★★ #3 (dp>1) |

★★★★★★★★★ **Migration path**: sleep_level=1 LoRA full → sleep_level=1 LoRA delta (after #6794 adds LoRA delta support) → full delta for dp>1 setups.

---

## References

- verl #6794: https://github.com/verl-project/verl/pull/6794 (delta weight sync sender)
- SGLang #26519: https://github.com/sgl-project/sglang/pull/26519 (delta weight update receiver)
- THUDM/slime: https://github.com/THUDM/slime (original bytewise diff design)
- PULSE: arXiv:2502.03839 (>99% bytes unchanged at RL learning rates)
- SparrowRL: arXiv:2602.11456 (sparse weight update for RL)
- vLLM #31848/#39451: sparse weight-update API (dependency for future vLLM delta path)
- verl #6512: per-unit LoRA summon (10x memory reduction, complementary)
- verl #6699: detach memory fix (4x reduction, complementary)
- radixark/miles#1235: companion PR to SGLang #26519
- DeepSpeed #8061: overlap_comm NaN — SAME root cause as record_stream bug (multi-stream CUDA safety)
- Cross-framework GRPO training stack final reference: notebook/synthesis/cross-framework-grpo-training-stack-rtx4090-final-reference.md
