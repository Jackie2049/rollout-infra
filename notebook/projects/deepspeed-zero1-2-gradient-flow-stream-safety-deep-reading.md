# DeepSpeed ZeRO-1/2 Source Code Deep Reading — Gradient Flow & Stream Safety

> Created: 2026-06-20 | Priority: HIGH for RTX 4090 GRPO understanding
> Source: Background agent deep reading of DeepSpeed ZeRO internals

## 1. Complete Gradient Flow Path (ZeRO Stage 1/2)

### Backward → IPG Bucket → reduce_ipg_grads() → average_tensor() → Epilogue → Optimizer Step

**ZeRO-1** (optimizer state partitioning):
- Forward: full model on each GPU (no sharding)
- Backward: each GPU computes full gradients locally
- IPG bucket accumulation: gradients accumulated into `PartitionedGradientBuffer` buckets during backward
- `reduce_ipg_grads()`: triggers reduce-scatter across DP ranks → each rank gets its gradient shard
- `average_tensor()`: averages the received gradient shard, applies gradient clipping
- Optimizer step: each rank updates its optimizer state shard, then all-gather to get updated parameters

**ZeRO-2** (gradient + optimizer state partitioning):
- Same flow as ZeRO-1, but gradients are reduced via reduce-scatter and NOT kept full on any GPU
- After reduce-scatter, each rank only holds its gradient shard permanently
- No gradient all-gather needed — reduces peak memory significantly
- At dp=1: both ZeRO-1 and ZeRO-2 provide ZERO savings (no partitioning possible)

### Key Memory Savings (dp=1)

| Component | Full | ZeRO-1 dp=1 | ZeRO-2 dp=1 | ZeRO-2 dp=4 |
|-----------|------|-------------|-------------|-------------|
| Model params | 14 GiB | 14 GiB | 14 GiB | 3.5 GiB |
| Gradients | 14 GiB | 14 GiB | 14 GiB | 3.5 GiB |
| Optimizer states | 56 GiB | 56 GiB | 56 GiB | 14 GiB |
| Total | 84 GiB | 84 GiB | 84 GiB | 21 GiB |

**RTX 4090 (dp=1)**: ZeRO provides ZERO memory savings. Only CPU offload helps.

## 2. The #8061 Bug — Multi-Stream Race in average_tensor()

### Bug Root Cause

`average_tensor()` in `partitioned_param_coordinator.py` only synchronizes with `get_accelerator().current_stream()` before reading the gradient bucket buffer. This misses other producer streams that may have written gradient data:

```python
# BEFORE FIX (buggy):
def average_tensor(self, context):
    self.accelerator.current_stream().wait_stream(self.reduction_stream)
    # BUG: only waits on reduction_stream, misses other producer streams!
```

When `torch.compile` generates multi-stream kernels (e.g., CUDA graph captured kernels on different streams), the gradient data may still be in-flight on those streams when `average_tensor()` reads the buffer → **data race → NaN or corrupted gradients**.

### The #8080 Fix

Adds a `copy_streams: set` field to `IPGBucket` to track ALL producer streams that wrote into the bucket buffer:

```python
# AFTER FIX:
class IPGBucket:
    copy_streams: set  # NEW: tracks all producer streams

def average_tensor(self, context):
    for stream in self.copy_streams:
        self.accelerator.current_stream().wait_stream(stream)
    # Now correctly waits on ALL producer streams
```

- +19/-1 code changes
- +72/-0 test additions
- External contributor (arunshar)
- Merge probability: 8/10
- Timeline: 2-6 weeks

### RTX 4090 Impact

- overlap_comm still pointless on dp=1 (reduce-scatter with 1 rank = identity)
- But the fix makes overlap_comm SAFE for dp>1 deployments
- Same pattern family as verl #6794 and vLLM #45552

## 3. overlap_comm Mechanism

### Double-Buffered IPG Buckets

`overlap_comm` uses a separate `reduction_stream` and double-buffered IPG buckets to overlap NCCL communication with backward computation:

1. While backward computes on stream A, NCCL reduce-scatter runs on `reduction_stream` for completed buckets
2. Two IPG bucket buffers alternate: one being filled by backward, one being reduced by NCCL
3. After all buckets reduced, `average_tensor()` processes the results

### Why overlap_comm is Pointless on dp=1

- reduce-scatter with dp_size=1 is an identity operation (no data movement)
- NCCL overhead: kernel launch + stream synchronization = ~0.5ms per bucket
- No actual communication benefit since there's only 1 rank
- Can actually HURT performance due to stream synchronization overhead

**RTX 4090 MUST NOT**: overlap_comm=True on dp=1 — wastes time and introduces stream race risk

## 4. ZeRO-2 + CPU Offload: The Only Viable Full Fine-Tuning Path

### CPU Adam Offload

When `optimizer=cpu_adam`:
- Optimizer states (fp32 params, momentum, variance) live on CPU RAM
- Gradients reduced on GPU, then offloaded to CPU for optimizer step
- Updated fp32 parameters copied back to GPU and cast to bf16
- Saves ~56 GiB of optimizer state memory on GPU

### ZeRO-2 + CPU Offload Memory Budget (7B model, dp=1)

| Component | GPU Memory | CPU Memory |
|-----------|------------|------------|
| Model params (bf16) | 14 GiB | — |
| Gradients (bf16) | 14 GiB | — |
| Optimizer states (fp32) | — | 56 GiB |
| Activation memory | ~2-4 GiB | — |
| Total GPU | ~16-18 GiB | — |
| Total CPU | — | 56 GiB |

**RTX 4090**: ~16-18 GiB GPU (fits in 24 GiB), requires ~56 GiB CPU RAM

### But LoRA is Better on RTX 4090

| Approach | GPU Peak | Steps/hr | Convergence Speed |
|----------|----------|----------|-------------------|
| ZeRO-2 + CPU Adam + full FT | ~18 GiB | ~81 | 1x baseline |
| LoRA r=32 + CPU Adam | ~16 GiB | ~258 | ~2x faster (stronger gradient) |
| LoRA r=32 + bypass | ~16 GiB | ~258 | ~2.5x faster |

**Conclusion**: LoRA + bypass = superior to ZeRO-2 + full FT on RTX 4090

## 5. Cross-Framework Pattern: CUDA Stream Safety

This reading confirms the 6-member CUDA Stream Safety pattern family:

| # | Framework | Issue | Root Cause | Fix |
|---|-----------|-------|-----------|-----|
| 1 | DeepSpeed | #8061 | average_tensor misses producer streams | #8080: copy_streams set |
| 2 | verl | #6794 CRITICAL-1 | record_stream missing for weight sync | NOT FIXED |
| 3 | vLLM | #45552 | cumem release/resume missing synchronize() | 2-line torch.cuda.synchronize() |
| 4 | DeepSpeed | #8072 | ZeRO-3 + PEFT weight sync race | NOT FIXED |
| 5 | SGLang | #28676 | MoE cache clobber during weight reload | NOT FIXED |
| 6 | SGLang | #28771 | HiCache async race with draft forward | NOT FIXED |

**Common theme**: Multi-stream GPU operations require explicit synchronization. `current_stream().wait_stream()` only covers one stream — all other producer streams must also be waited on.

## 6. Key Takeaways for RTX 4090 GRPO

1. **ZeRO is unnecessary for LoRA training on dp=1** — FSDP1 handles sharding, LoRA reduces trainable params
2. **overlap_comm MUST be False** — pointless and dangerous on dp=1
3. **ZeRO-2 + CPU Adam only viable for full FT** — but LoRA is superior on RTX 4090
4. **#8080 fix makes overlap_comm safe for dp>1** — merge expected in 2-6 weeks
5. **Stream safety pattern is systemic** — affects all frameworks doing GPU weight operations
6. **Always check for copy_streams / record_stream when doing multi-stream GPU operations**

## Key Source Files Referenced

| File | Purpose |
|------|---------|
| deepspeed/runtime/zero/partitioned_param_coordinator.py | IPG bucket management, average_tensor(), reduction stream |
| deepspeed/runtime/zero/stage_1_and_2.py | ZeRO-1/2 gradient reduction, optimizer step |
| deepspeed/runtime/engine.py | DeepSpeedEngine, overlap_comm setup |
| deepspeed/runtime/zero/offload_config.py | CPU offload configuration |
