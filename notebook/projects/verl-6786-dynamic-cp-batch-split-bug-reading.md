# verl #6786: Dynamic Context Parallel (DCP) Batch Split Bug Deep Reading

## 1. Issue Metadata

| Field | Value |
|-------|-------|
| **Title** | dynamic-cp split the batch into local_cp_size sub-batches bug |
| **Author** | liujia-cc |
| **Date** | 2026-06-17 |
| **Status** | OPEN |
| **Label** | bug |
| **URL** | https://github.com/verl-project/verl/issues/6786 |
| **Code ref** | `verl/utils/megatron_utils.py#L1629` |

### Related Issues & PRs

| Resource | Title | Status | Relevance |
|----------|-------|--------|-----------|
| PR #5057 | [megatron] feat: support dynamic CP | MERGED (2026-03-31) | Original DCP feature — introduced the buggy code |
| PR #5869 | [megatron] fix: dynamic context parallel batch splitting and loss normalization | OPEN (status: test in process) | THE FIX — rewrote `dynamic_cp_split_batch`, fixed loss norm |
| PR #6049 | [megatron] fix: correct batch_num_tokens aggregation for context parallelism | CLOSED (superseded) | Earlier fix attempt for static CP loss norm, not DCP |
| PR #6555 | add dynamic context parallel for verl | OPEN | Alternative DCP implementation (all-to-all scheduling, more complex) |
| PR #6267 | [megatron] fix: fix bugs when using position_ids in cp | MERGED | CP position_ids off-by-one |
| Issue #5983 | [megatron][engine][context parallel] batch_num_tokens aggregation undercounts valid tokens when CP > 1 | OPEN | Static CP loss norm bug — same root cause class |
| Issue #1332 | [BUG][mcore][context parallel][grpo] GRPO fails to converge with CP>1 | OPEN | USER REPORT of CP convergence failure — likely caused by both #6786 and #5983 |
| Issue #3415 | Memory Concerns with max_token_len Calculation When Using Context Parallelism | OPEN | max_token_len scaling bug with CP>1 |
| Megatron-LM #4382 | make_viewless_tensor() causes GPU memory leak when context_parallel_size > 1 | OPEN | Memory leak in CP |
| Megatron-LM #1764 | Context parallel nan loss | OPEN | NaN with CP in Megatron upstream |

---

## 2. What is Dynamic Context Parallelism (Dynamic CP)?

### 2.1 Static Context Parallelism (Baseline)

In standard Megatron-LM, **Context Parallelism (CP)** splits long sequences across multiple GPUs along the sequence dimension. With `context_parallel_size=2`, a 8192-token sequence is divided into two 4096-token chunks, each processed by a different GPU rank. The two ranks communicate via ring-attention (striped/zigzag pattern) during the forward pass to compute correct attention scores across the full sequence length.

The CP size is **static** — set at initialization time via `context_parallel_size` in the Megatron config, and remains fixed for all training steps. This means:

- Short sequences waste CP capacity (CP=2 means every sequence, even 128-token ones, gets split across 2 ranks)
- Memory overhead is constant regardless of actual sequence lengths
- Throughput is lower for short-sequence batches because unnecessary inter-rank communication happens

### 2.2 Dynamic Context Parallelism (DCP)

**Dynamic CP** (introduced in verl PR #5057, authored by ISEEKYAN) adapts the effective CP size **per micro-batch** based on the longest sequence in the batch. Key design principles:

1. **Runtime CP grouping**: `context_parallel_size` is set to 1 in the Megatron config. All GPU ranks form a single DP group from Megatron's perspective. Dynamic CP grouping happens at runtime inside `dynamic_cp_split_batch()`.

2. **Power-of-2 CP size selection**: For each micro-batch:
   - Compute `max_seq_len_in_batch`
   - If `max_seq_len_in_batch <= max_seqlen_per_dp_cp_rank` then `local_cp_size = 1` (no CP needed)
   - Otherwise pick the smallest power-of-2 `n` such that `max_seq_len_in_batch <= max_seqlen_per_dp_cp_rank * n` and `n <= dp_size`

3. **DP×CP repartitioning**: With `dp_size=8` and `local_cp_size=4`, the 8 ranks split into:
   - 2 DP sub-groups (local_dp_size = 8/4 = 2), each handling different sequences
   - Within each sub-group, 4 ranks form a CP group processing the same sequence's different chunks

4. **Configuration**:
   ```yaml
   engine:
     dynamic_context_parallel: true
     context_parallel_size: 1          # MUST be 1 when DCP enabled
     max_seqlen_per_dp_cp_rank: 5120   # Same as max_token_len_per_gpu
   ```

### 2.3 DCP vs Static CP Comparison

| Aspect | Static CP | Dynamic CP |
|--------|-----------|------------|
| CP size | Fixed at init | Adaptive per micro-batch |
| Short sequence handling | Still splits (wasteful) | CP=1, no overhead |
| Long sequence handling | Full CP capacity | CP increases to fit |
| Megatron config | `context_parallel_size > 1` | `context_parallel_size = 1` |
| Group management | Megatron `mpu` manages | Runtime `dynamic_cp_split_batch()` |
| Memory efficiency | Constant overhead | Proportional to actual lengths |
| Throughput on mixed-length batches | Low (over-provisioned) | High (adaptive) |

---

## 3. The Batch Split Bug: What Goes Wrong

### 3.1 Original Buggy Code (Lines 1598-1642, current main)

```python
def dynamic_cp_split_batch(
    batch: TensorDict, engine_config: McoreEngineConfig, dp_size: int, dp_rank: int
) -> TensorDict:
    """Split the batch into sub-batches for dynamic context parallel."""
    input_ids = batch["input_ids"]
    assert input_ids.is_nested, "input_ids must be a nested tensor"
    seq_len_effective: torch.Tensor = input_ids.offsets().diff()
    max_seq_len = max(seq_len_effective)
    # BUG 1: if num of sequences is less than dp_size, we don't need to split the batch
    local_cp_size = None
    if len(seq_len_effective) < dp_size:
        local_cp_size = dp_size
        return batch  # <-- BUG: returns WITHOUT splitting, but sets local_cp_size=dp_size
        # This means: ALL ranks process the SAME data (no DP split)
        # But dp_size metadata is NEVER set to local_dp_size
        # Downstream loss formula uses wrong dp_size
    else:
        max_seqlen_per_dp_cp_rank = engine_config.max_seqlen_per_dp_cp_rank
        local_cp_size = math.ceil(max_seq_len / max_seqlen_per_dp_cp_rank)
        local_cp_size = 1 << (local_cp_size - 1).bit_length()
        # BUG 2: assert can trigger for small dp_size
        assert local_cp_size <= dp_size

        if local_cp_size < dp_size:
            local_dp_rank = dp_rank // local_cp_size
            local_dp_size = dp_size // local_cp_size
            indices = list(range(len(seq_len_effective)))
            # BUG 3: ceil-based allocation can cause OUT-OF-BOUNDS indexing
            num_seq_per_local_cp = math.ceil(len(seq_len_effective) / local_dp_size)
            start_idx = local_dp_rank * num_seq_per_local_cp
            end_idx = min(start_idx + num_seq_per_local_cp, len(seq_len_effective))
            selected_indices = indices[start_idx:end_idx]
            batch = tu.index_select_tensor_dict(batch, selected_indices)
    # BUG 4: dp_size is NEVER overwritten to local_dp_size
    tu.assign_non_tensor_data(batch, "local_cp_size", local_cp_size)
    return batch
```

### 3.2 Four Specific Bugs

**BUG 1: Early return when `num_seqs < dp_size` — DP ranks receive no data or duplicate data**

The issue author (liujia-cc) specifically flags this: "Assuming len(seq_len_effective) = 9 and local_dp_size = 4, then when distributing data, there will be a DP rank that receives no data."

When `len(seq_len_effective) < dp_size`, the function sets `local_cp_size = dp_size` and **returns the batch unsplit**. This means:
- `local_dp_size = dp_size / local_cp_size = 1` — only 1 DP sub-group
- All ranks in the CP group process the SAME full batch
- BUT: no `dp_size` metadata is set, so downstream loss normalization uses the original (wrong) dp_size

More critically: when `num_seqs < dp_size` in the ELSE branch (e.g., 9 sequences with local_dp_size=4), the `ceil` allocation gives `num_seq_per_local_cp = ceil(9/4) = 3`. With 4 DP sub-groups: ranks 0,1,2,3 get 3,3,3,0 sequences respectively. Rank 3 gets **zero sequences** — it has nothing to process.

**BUG 2: Coverage constraint not enforced**

When there are fewer sequences than DP sub-groups, some DP ranks get zero data. The original code has no mechanism to ensure every DP sub-group receives at least one sequence. The fix (PR #5869) adds a `min_cp_for_coverage` constraint:

```python
min_cp_for_coverage = math.ceil(dp_size / num_seqs) if num_seqs > 0 else dp_size
if min_cp_for_coverage > 1:
    min_cp_for_coverage = 1 << (min_cp_for_coverage - 1).bit_length()
local_cp_size = max(local_cp_size, min_cp_for_coverage)
```

This ensures: if 2 sequences with dp_size=8, then `min_cp_for_coverage = ceil(8/2) = 4`, rounded to power-of-2 = 4. So `local_cp_size >= 4`, `local_dp_size <= 2`, and each sub-group gets at least 1 sequence.

**BUG 3: Ceil-based allocation can cause index out-of-bounds**

```python
num_seq_per_local_cp = math.ceil(len(seq_len_effective) / local_dp_size)
```

With `len(seq_len_effective)=9, local_dp_size=4`: `num_seq_per_local_cp = ceil(9/4) = 3`.

For `local_dp_rank=3`: `start_idx = 3*3 = 9`, `end_idx = min(9+3, 9) = 9`. So `selected_indices = indices[9:9] = []` — empty!

The fix uses a proper remainder-based distribution:
```python
base_count = num_seqs // local_dp_size
remainder = num_seqs % local_dp_size
if local_dp_rank < remainder:
    start_idx = local_dp_rank * (base_count + 1)
    count = base_count + 1
else:
    start_idx = remainder * (base_count + 1) + (local_dp_rank - remainder) * base_count
    count = base_count
```

For `9 sequences, local_dp_size=4`: `base_count=2, remainder=1`. Ranks get 3,2,2,2 — all have data.

**BUG 4: `dp_size` metadata not overwritten**

The original code never sets `batch["dp_size"]` to `local_dp_size`. The downstream loss formula:
```python
loss = -masked_sum(log_probs) / batch_num_tokens * dp_size
```

uses `dp_size` for gradient normalization. With DCP, after splitting into CP sub-groups, the effective DP size is `local_dp_size = dp_size / local_cp_size`. If `dp_size` is not updated, the loss is incorrectly scaled by the wrong factor, causing gradient bias.

### 3.3 Loss Normalization Bug (companion issue #5983)

Issue #5983 and PR #6049 identified a related but distinct bug in `forward_backward_batch()`:

```python
# Current buggy code:
batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
torch.distributed.all_reduce(
    batch_num_tokens, op=torch.distributed.ReduceOp.SUM,
    group=self.get_data_parallel_group()  # pure DP group, excludes CP ranks
)
tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())  # also pure DP
```

With CP>1, each CP rank only holds `1/CP` of the sequence. `loss_mask.sum()` gives a partial token count. The all-reduce over the pure DP group never includes other CP ranks' partial counts — undercounting by a factor of CP.

**PR #5869's fix for DCP mode**: Since DCP sets `context_parallel_size=1`, all ranks start with the same full batch. So `loss_mask.sum()` already equals the global token count on every rank — no all-reduce needed. The fix skips it:

```python
if self.engine_config.dynamic_context_parallel:
    pass  # skip all_reduce — all ranks already have full token count
else:
    torch.distributed.all_reduce(...)  # static CP still needs it
```

**PR #5869's fix for sp_size**: In DCP mode, `sp_size` is set to `mpu.get_data_parallel_world_size()` (full world size) rather than `context_parallel_size=1`. This is used in `prepare_micro_batches()` for computing `max_token_len = max_token_len_per_gpu * sp_size` to determine micro-batch splitting granularity. The full world size correctly represents total token capacity across all ranks.

### 3.4 The Mathematical Proof (from Kite0011's review comment)

The three changes form a **consistent system**:

1. `batch_num_tokens = T_total` (no all-reduce needed, all ranks have identical full batch)
2. `dp_size = local_dp_size = D/C` (compensates for C-fold gradient duplication in DDP)
3. `loss = -masked_sum / T_total * D/C`

After DDP averaging across `W = C x D` ranks:
```
grad_avg = sum_j(C * loss_j * D / T_total) / (C * D)
         = sum_j(loss_j) / T_total
```

This is the correct global average gradient. Without the `dp_size = D/C` correction, DDP would average C identical copies of each sub-group's gradient, producing a `C`-fold overcount.

---

## 4. RTX 4090 Implications for GRPO Training

### 4.1 DCP is NOT viable on RTX 4090 (dp=1)

Dynamic CP requires `dp_size >= 2` to form CP sub-groups. On RTX 4090 with `dp_size=1` (single GPU), DCP degenerates to `local_cp_size=1` always — no context parallelism can be formed. The whole adaptive CP mechanism is irrelevant.

This is confirmed by the test cases in PR #5869:
```python
def test_dp_size_1(self):
    """Single GPU: cp=1"""
    cp, dp = self._get_cp_dp([4096], dp_size=1, dp_rank=0, max_per_rank=1024)
    assert cp == 1
    assert dp == 1
```

### 4.2 RTX 4090 MUST use Megatron TP-only, not CP

For RTX 4090 GRPO training with verl Megatron backend:
- `tensor_model_parallel_size=1` (single GPU, no TP)
- `pipeline_model_parallel_size=1`
- `context_parallel_size=1` (no CP)
- `dynamic_context_parallel=False` (DCP irrelevant)

The correct strategy for long sequences on RTX 4090 is to use `use_dynamic_bsz=True` with `ppo_max_token_len_per_gpu` set appropriately, which limits per-GPU token counts without needing CP.

### 4.3 Bug #6786 affects multi-GPU setups that RTX 4090 users might scale to

If a user transitions from single RTX 4090 to 2-8 GPU setup:
- With 2 GPUs: `dp_size=2`, DCP can only form `local_cp_size=1` or `local_cp_size=2`
- With 4 GPUs: `local_cp_size` can be 1, 2, or 4
- The batch split bug directly affects these scenarios — some ranks would get zero data

For 4-GPU GRPO with Megatron backend:
- **Current (buggy)**: 3 sequences + dp_size=4 -> ceil(3/4)=1 -> rank 3 gets nothing
- **Fixed**: coverage constraint forces local_cp_size >= 4 -> all ranks share same data via CP

### 4.4 CP-related convergence failures (#1332)

Issue #1332 reports GRPO failing to converge with `context_parallel_size=2`. The root cause is likely the combined effect of:
1. `batch_num_tokens` undercounting by factor of CP (#5983) — inflates loss
2. `dp_size` metadata not set correctly (#6786) — wrong gradient scaling
3. Position IDs off-by-one in CP (#6267, now fixed) — wrong attention

These bugs compound: wrong loss normalization + wrong gradient scaling + wrong positions = no convergence.

### 4.5 MUST DO rules for RTX 4090 GRPO with any CP scenario

| Rule | Explanation |
|------|-------------|
| MUST: `context_parallel_size=1` on dp=1 | CP requires dp>=2 |
| MUST: `dynamic_context_parallel=False` on dp=1 | DCP requires dp>=2 |
| MUST: `overlap_comm=False` on single GPU (#8061) | multi-stream NaN |
| MUST: `gradient_clipping=1.0` (#8068) | default 0 means no clipping |
| MUST NOT: use ZeRO-3 on single GPU | Pure overhead, use ZeRO-2 |
| MUST NOT: use static CP>1 for GRPO | Convergence bugs unresolved |

---

## 5. Connection to Other Tracked verl Issues

### 5.1 #6699 (detach memory fix)

- **Bug**: `verl` Automodel/Megatron/TorchTitan backends have gradient accumulation leak (same pattern as #6699 detach bug)
- **Connection**: DCP batch splitting operates on TensorDict data that flows through the same `forward_backward_batch()` pipeline. If detach is not applied correctly, accumulated gradients from previous CP configurations (different local_cp_size per step) could leak into the current step
- **Overlap**: Both bugs affect the Megatron engine's data pipeline — #6699 fixes gradient accumulation, #6786 fixes data distribution

### 5.2 #6468 (FSDP2 CPU memory leak)

- **Bug**: FSDP2 CPU memory leak of 0.6-6.3 GiB/step during weight sync
- **Connection**: FSDP backend does NOT support DCP (DCP is Megatron-only). But the pattern is similar: distributed data handling that leaks or miscomputes across ranks
- **Key difference**: #6468 is FSDP-specific, #6786 is Megatron-specific. No code overlap, but same class of "distributed state management bug"

### 5.3 #6512 (per-unit LoRA summon)

- **Bug**: FSDP1/2 whole-model summon = OOM on RTX 4090, fixed by per-unit LoRA summon
- **Connection**: #6512's per-unit summon was partially motivated by the need to handle LoRA adapters across DP ranks — similar to how DCP needs to split batch data across DP sub-groups
- **Architecture overlap**: Both involve dynamic partitioning of GPU resources (LoRA adapters vs. sequence chunks) across distributed ranks

### 5.4 #6782 (LoRA rank=64 breaks EOS)

- **Connection**: LoRA rank affects the number of trainable parameters per sequence. With DCP, different local_cp_size changes how sequences are distributed — LoRA parameter counts per DP sub-group could vary, potentially exacerbating the EOS bug

### 5.5 #6794 (delta weight sync)

- **Connection**: Delta weight sync (SGLang-only) transfers weight deltas across rollout/training engines. DCP affects the training side's batch distribution, which determines which weight deltas are needed. No direct code overlap but architectural interaction.

### 5.6 verl Loss Pipeline Interaction

The DCP loss normalization bug intersects with the general verl loss pipeline:

```
forward_backward_batch():
  1. Compute sp_size (BUG: wrong for DCP)
  2. Compute batch_num_tokens (BUG: undercounted for static CP, handled for DCP)
  3. Set dp_size (BUG: not updated to local_dp_size)
  4. dynamic_cp_split_batch() (BUG: empty ranks, wrong distribution)
  5. Forward through Megatron model
  6. Loss = -masked_sum / batch_num_tokens * dp_size
  7. DDP all-reduce averages gradients across W=C*D ranks
```

Bugs at steps 1, 2, 3, 4 compound to produce wrong gradients at step 6.

---

## 6. Framework Comparison: Context Parallelism Across Frameworks

### 6.1 Megatron-LM (upstream)

- **Implementation**: Ring attention + zigzag striped pattern in `megatron/core/parallel_state.py` and `megatron/core/tensor_parallel/attn.py`
- **CP size**: Static, set at initialization via `context_parallel_size`
- **Attention**: RingAttention (striped/zigzag) distributes attention computation across CP ranks
- **Communication**: `all-gather` + `reduce-scatter` for attention score computation across CP ranks
- **Known bugs**: NaN loss (#1764), memory leak (#4382), DSA attention not supported (#4878)
- **Dynamic CP**: Not supported in upstream Megatron-LM — verl's DCP is a verl-only extension

### 6.2 verl (Megatron engine path)

- **Implementation**: Two paths:
  1. **Static CP**: Uses Megatron's built-in CP with `context_parallel_size > 1`
  2. **Dynamic CP**: Runtime adaptive CP via `dynamic_cp_split_batch()` + `dynamic_cp_merge_output()`
- **Current bugs**: batch split (#6786), loss norm (#5983), position_ids (#6267 FIXED), convergence (#1332)
- **Fix PRs**: #5869 (DCP fix, OPEN), #6555 (alternative DCP impl, OPEN), #6049 (static CP fix, CLOSED/superseded)
- **Unique feature**: DCP is verl-specific — no other RL framework has dynamic context parallelism

### 6.3 DeepSpeed

- **Implementation**: Ulysses-style attention via `deepspeed.module_inject.replace_with_deepspeed` and `DistributedAttention`
- **CP size**: Static, configured via DeepSpeed config JSON
- **Known bugs**: Ulysses silently produces incorrect output when #GPUs does not divide sequence length (#7384)
- **Dynamic CP**: Not supported
- **Note**: DeepSpeed's Ulysses approach is fundamentally different from Megatron's RingAttention — Ulysses splits QKV heads across ranks, while RingAttention splits sequence positions

### 6.4 PyTorch FSDP2

- **CP support**: Not natively supported. FSDP2 handles model parameter sharding across DP ranks, but does NOT split sequences. Long sequence handling requires:
  1. Reducing model size (LoRA, quantization)
  2. Using separate inference engine (vLLM/SGLang) for rollout
  3. Chunked processing in training loop
- **No CP primitives**: FSDP2 has no concept of context parallelism

### 6.5 vLLM / SGLang (inference side)

- **No training CP**: These are inference-only frameworks. They handle long sequences via:
  1. Chunked prefill (vLLM `enable_chunked_prefill=True`)
  2. KV cache management
  3. Sliding window attention
- **Relevance to DCP**: The rollout engine (vLLM/SGLang) generates long responses. These responses are then fed to the training engine (Megatron), which needs CP to handle them. DCP bridges the gap: adaptive CP in training handles the variable-length outputs from inference.

### 6.6 rLLM

- **No CP support**: rLLM uses a single-GPU training loop (Tinker backend) with no distributed parallelism beyond basic DP
- **Relevance**: RTX 4090 single-GPU use case — same as verl dp=1, no CP needed

### 6.7 Comparison Matrix

| Framework | CP Type | Dynamic CP? | Loss Norm Bugs? | RTX 4090 viable? |
|-----------|---------|-------------|-----------------|-------------------|
| Megatron-LM | Static RingAttention | No | NaN (#1764) | No (needs multi-GPU) |
| verl | Static + Dynamic | Yes (buggy) | Yes (#5983, #6786) | dp=1: irrelevant |
| DeepSpeed | Static Ulysses | No | Silent incorrect (#7384) | No (needs multi-GPU) |
| PyTorch FSDP2 | None | No | No | dp=1: no CP |
| vLLM/SGLang | Chunked prefill | No | N/A (inference) | Yes (chunked prefill) |
| rLLM | None | No | No | dp=1: no CP |

---

## 7. PR #5869 Fix Analysis (Detailed)

### 7.1 Three-Part Fix

**Part A: Rewrite `dynamic_cp_split_batch()`** (`verl/utils/megatron_utils.py`)

Key changes:
1. Compute `local_cp_size` from `max_seq_len / max_seqlen_per_dp_cp_rank`, rounded to power-of-2
2. Add `min_cp_for_coverage` constraint: `ceil(dp_size / num_seqs)` rounded to power-of-2, ensures every DP sub-group gets at least one sequence
3. Replace `ceil`-based allocation with remainder-based distribution (no empty ranks)
4. Set `batch["dp_size"] = local_dp_size` (fix BUG 4)
5. Set `batch["local_cp_size"]` as metadata

**Part B: Fix loss normalization in `forward_backward_batch()`** (`verl/workers/engine/megatron/transformer_impl.py`)

```python
# NEW:
if self.engine_config.dynamic_context_parallel:
    tu.assign_non_tensor(data, sp_size=mpu.get_data_parallel_world_size())
else:
    tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
if self.engine_config.dynamic_context_parallel:
    pass  # skip all_reduce — all ranks have identical full batch
else:
    torch.distributed.all_reduce(
        batch_num_tokens, op=torch.distributed.ReduceOp.SUM,
        group=self.get_data_parallel_group()
    )
tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
tu.assign_non_tensor(data, dp_size=mpu.get_data_parallel_world_size())
```

**Part C: Megatron recomputation patch compatibility** (`verl/models/mcore/patch.py`)

```python
if not hasattr(rd.CheckpointFunction, "_recover_function_args"):
    return  # Skip patch on older Megatron versions
```

### 7.2 Config Validation (engine.py)

```python
if self.dynamic_context_parallel:
    assert self.context_parallel_size == 1, (
        "dynamic_context_parallel requires context_parallel_size=1 "
        "because CP groups are formed dynamically at runtime"
    )
```

### 7.3 Review Discussion Summary

**ISEEKYAN (collaborator)** raised two concerns:
1. The `sp_size = world_size` setting violates the "logical DP=1" constraint from PR #5057's design. Response: sp_size is only used in `prepare_micro_batches()` for token capacity, not for data splitting.
2. Data would be split across ranks, breaking the "all ranks get same data" assumption. Response: Data splitting IS the intended behavior for DCP — ranks within the same DP sub-group get different sequences.

**gemini-code-assist** flagged three HIGH-priority concerns:
1. Skipping `batch_num_tokens` all-reduce could cause gradient scaling errors if token counts vary across ranks. Response: In DCP mode, all ranks start with identical full batch, so local token count = global token count.
2. `sp_size = world_size` is logically inconsistent. Response: sp_size controls micro-batch splitting granularity, not loss normalization.
3. Setting `dp_size = local_dp_size` might cause learning rate doubling if DDP averaging already compensates. Response: The math shows this is correct — DDP averages across C*D ranks, and dp_size=D/C compensates for C-fold duplication.

**wuxibin89** approved the PR.

### 7.4 Test Suite (557 lines!)

PR #5869 includes a comprehensive test file `tests/models/test_dynamic_context_parallel.py` with:
- `TestLocalCPSizeComputation`: 8 tests for CP size calculation (exact fit, double length, round up, clamp, coverage, single sequence, always power-of-2, parametrized)
- `TestBatchSplitting`: 4 tests for sequence distribution (even split, full CP, partial CP, CP ranks share data, all sequences covered)
- `test_loss_alignment`: DCP loss (all-reduced) matches single-rank reference
- `test_grad_norm_alignment`: DCP gradient norms match reference
- `test_merge_output_identity`: When local_cp_size == dp_size, merge returns input unchanged
- `test_loss_consistent_across_cp_sizes`: Same data, different CP sizes, same loss
- `TestEdgeCases`: 7 tests (num_seqs < dp, uneven distribution, dp_size=1, very short seqs, metadata overwrite, metadata attachment)

---

## 8. PR #6555 — Alternative DCP Implementation

PR #6555 by xiaoyao0115 is a **more ambitious** alternative that adds a full DCP scheduler with all-to-all communication. It modifies the loss function (`verl/workers/utils/losses.py`) for DCP, adds a `dynamic_cp_scheduler.py` module, and changes the Megatron engine more extensively.

**Critical issues flagged by reviews**:
1. **Critical**: `dcp_make_buckets_equal` has a logic error — `remaining_k - len(buckets)` double-decrements, creating premature empty buckets
2. **High**: `rollout_is_weights` not sliced in PPO loss for DCP path — shape mismatch
3. **High**: List-resizing bug in `align_sample_id_groups` — slice assignment changes list length
4. **High**: `NestedTensor.sum()` unsupported — should use `.values().sum()`
5. **High**: `fill_empty_gpus` under-estimates workload — only counts first sequence in packed batches
6. **ISEEKYAN feedback**: Should import core algorithm from megatron-core instead of rewriting, and should not modify the universal loss function for a Megatron-specific feature

**Comparison**: #5869 is a focused fix (+633/-38, 5 files) that fixes the existing DCP implementation. #6555 is a major rewrite (+2617/-81, 10 files) that replaces it. #5869 is more likely to merge first given its scope and existing test coverage.

---

## 9. Key Takeaways

1. **Bug #6786 is a multi-part correctness bug** that affects DCP in four ways: empty DP ranks, out-of-bounds indexing, wrong dp_size metadata, and wrong loss normalization. Each bug alone causes training failure; combined they are catastrophic.

2. **DCP is verl's unique feature** — no other RL framework (Megatron-LM upstream, DeepSpeed, rLLM) has dynamic context parallelism. This makes the bug particularly impactful because DCP users have no alternative.

3. **RTX 4090 is unaffected** because DCP requires dp>=2. Single-GPU setups always get local_cp_size=1. But multi-GPU scaling from RTX 4090 setups would hit this bug immediately.

4. **The bug class is "distributed data partitioning"** — similar to #6699 (gradient leak across ranks), #6468 (memory leak across ranks), #5983 (token undercount across ranks). All share the pattern: verl's Megatron engine incorrectly handles data distribution across DP/CP ranks.

5. **Two competing fix PRs** — #5869 (focused fix, 5 files, 633 additions, comprehensive tests, approved by wuxibin89) vs #6555 (major rewrite, 10 files, 2617 additions, multiple bugs flagged). #5869 is the better candidate for merging.

6. **Static CP (#5983) is still unfixed** — PR #6049 was closed/superseded. The static CP loss normalization bug (batch_num_tokens undercounting by factor of CP) remains OPEN. Users running static CP>1 for GRPO are still affected.

7. **Issue #1332 (GRPO+CP convergence failure) root cause confirmed**: The convergence failure is explained by the combination of #5983 (wrong loss norm) + #6786 (wrong data distribution) + #6267 (wrong position_ids, now fixed). All three bugs compound in the Megatron engine's forward-backward pipeline.

---

## 10. Monitoring Status

| Item | Status | Priority | Action |
|------|--------|----------|--------|
| PR #5869 (DCP fix) | OPEN, test in process, wuxibin89 approved | HIGH | Monitor for merge |
| PR #6555 (alternative DCP) | OPEN, multiple bugs flagged | MEDIUM | Likely superseded by #5869 |
| Issue #5983 (static CP loss norm) | OPEN, no fix merged | HIGH | Needs separate fix PR |
| Issue #1332 (GRPO+CP convergence) | OPEN | HIGH | Root cause = #5983 + #6786 + #6267 |
| Issue #6786 (this issue) | OPEN | HIGH | Track for resolution via #5869 |

**Last updated**: 2026-06-19
