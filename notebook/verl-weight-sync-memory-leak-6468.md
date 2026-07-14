# verl Weight Sync Memory Leak Analysis (#6468)

## Bug Summary

CPU memory grows monotonically during FSDP2 DAPO training on Qwen3.5-2B.
Leak rate: 0.6–6.3 GiB/step. Happens after actor param update / rollout weight sync.
Ray object store is near zero — leak is in Python process RSS.

## Weight Sync Code Flow

### Stage A: Actor Parameter Extraction
`ActorRolloutRefWorker.update_weights()` → `FSDPEngine.get_per_tensor_param()`

1. `self.module.state_dict()` returns DTensor references
2. Generator materializes each DTensor:
   ```python
   param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
   ```
3. For old code (pre-PR#7005): staging round trip `load_fsdp_model_to_gpu()` → `offload_fsdp_model_to_cpu()`

### Stage B: Weight Transfer
- Naive/colocated: `BucketedWeightSender` over ZMQ+IPC
- NCCL: `NCCLCheckpointEngine.send_weights()` broadcast via cupy
- Gloo: `_dtensor_full_tensor_gloo()` CPU all_gather

### Stage C: Rollout Weight Loading
`ServerAdapter.update_weights()` → vLLM `update_weights_from_ipc`

## 5 Leak Points

### LEAK 1 — DTensor.full_tensor() CPU staging buffers (PRIMARY)
Each `full_tensor()` creates a full-sized CPU staging buffer. For Qwen3.5-2B ~340 params:
- ~340 separate CPU staging buffers per sync step
- Pageable (not pinned) CPU tensors → PyTorch allocator may not release to OS
- `.to(torch.bfloat16, non_blocking=True)` creates another intermediate tensor
- Generator pattern → tensors from consecutive steps may overlap in memory

### LEAK 2 — Old staging round trip (pre-PR#7005)
`load_fsdp_model_to_gpu()` / `offload_fsdp_model_to_cpu()` round trip:
- Hundreds of small blocking H2D + D2H copies
- Fragmented CPU allocator → increased RSS
- Fixed by PR #7005 (merged Jul 10, 2026)

### LEAK 3 — LoRA collect_lora_params CPU accumulation
```python
lora_params = {name: param.full_tensor().detach().cpu() ...}
```
Creates CPU tensors for every LoRA parameter each step.
`empty_cache()` only clears GPU cache — CPU tensors remain.

### LEAK 4 — backup_base_model_weights clone leakage
```python
backup[name] = param.data.clone().cpu()
```
Clones all parameters to CPU simultaneously. PyTorch CPU allocator may not release pages.

### LEAK 5 — Gloo all_gather_object staging buffers
`torch.distributed.all_gather_object()` serializes Python objects through Gloo,
creating temporary buffers never explicitly freed.

## Root Cause Classification

NOT Ray object store (confirmed by reporter).
It's **Python process RSS** — specifically PyTorch CPU memory allocator:
- CPU caching allocator retains freed blocks for reuse
- For monotonic growth: **live Python references** preventing deallocation
- Generator closure holds params dict → intermediate tensors kept alive
- `non_blocking=True` CUDA ops → buffers pile up asynchronously

## Fix Approaches

### Fix 1 (Partially done — PR #7005): Skip staging round trip
Eliminates hundreds of H2D+D2H blocking copies.
Does NOT fix DTensor materialization leak itself.

### Fix 2: Explicit CPU cleanup after weight sync
```python
del per_tensor_param
gc.collect()
torch.cuda.empty_cache()
```
`aggressive_empty_cache(force_sync=True)` already called but targets GPU only.
CPU-side PyTorch allocator memory is NOT freed by `empty_cache()`.

### Fix 3: Sharded weight sync (avoid full_tensor())
Use `fsdp2_sharded_save_to_cpu()` pattern — transfer only local shards.
No all-gather needed on actor side. Rollout side handles reconstruction.

### Fix 4: Pin and reuse CPU buffers
- `pin_memory=True` for staging buffers → better tracking + release
- Buffer pool reused across steps
- Explicit pool free + `gc.collect()` after each sync

### Fix 5: gc.collect() after NCCLCheckpointEngine cycles
Currently `finalize()` calls `torch.cuda.empty_cache()` for GPU only.
Need CPU cleanup too.

## Related PRs & Issues

- **PR #7005** (merged Jul 10): Skip FSDP2 staging round trip
- **PR #6699** (merged Jun 12): Fix GPU-side autograd graph retention in forward_step
- **PR #6974**: Delta weight sync over NCCL — alternative path avoiding full materialization
- **Issue #7015** (follow-up to #7005): Batch via `FSDPModule.unshard()` instead of per-param gathering
- **vLLM #48312** (RFC): Weight reload correctness — 7-category taxonomy

## Assessment for Our Contribution

This is a complex, multi-layered issue. The most tractable entry point would be:
- **Fix 2** (gc.collect + CPU cleanup): Small, focused, low-risk
- Could fork verl, add explicit gc.collect() calls after weight sync paths
- Test on our H20-3e server with a simple DAPO training run

However, PR #7005 is already merged and PR #6974 (delta weight sync) is in progress.
We should monitor these before duplicating effort.
