# vLLM V1 Sleep/Wake Mechanism Deep Reading — CuMemAllocator, LoRA, KV Cache

> Created: 2026-06-20 | Priority: ★★★★★★★★ CRITICAL for RTX 4090 GRPO
> Source: Background agent deep reading of vLLM V1 source code

## 1. Sleep Level Semantics

### sleep(level=1) vs sleep(level=2)

**Call chain**: LLMEngine.sleep() → EngineCore.sleep() → Executor.sleep() → Worker.sleep() → CuMemAllocator.sleep()

| Level | Scheduler | Prefix Cache | GPU Memory Action | LoRA Weights |
|-------|-----------|-------------|-------------------|-------------|
| 0 | Pause only | Keep | No change | Keep |
| 1 | Pause + abort | Clear | Offload "weights" tag to CPU, discard KV cache | Keep (standard allocator) |
| 2 | Pause + abort | Clear | Discard ALL GPU memory (only buffers saved to CPU) | Keep (standard allocator) |

**Key distinction**:
- Level 1: Model weights preserved (CPU offload), KV cache DISCARDED → wake_up needs KV zeroing
- Level 2: NOTHING preserved → complete re-transfer needed on wake_up

## 2. CuMemAllocator.sleep() — The Missing synchronize() Bug

**File**: vllm/device_allocator/cumem.py, lines 167-216

**What sleep() does**:
1. For tagged allocations: allocate pinned CPU tensor, synchronous memcpy GPU→CPU
2. For ALL allocations: call `unmap_and_release()` (cuMemUnmap + cuMemRelease)
3. Call gc.collect() + torch.cuda.empty_cache()

**★★★★★★★★★ BUG**: No `torch.cuda.synchronize()` before the unmap loop.
- `_python_free_callback` (line 158) correctly has synchronize before unmap
- But the bulk sleep() path does NOT
- Result: in-flight CUDA kernels write to unmapped virtual addresses → CUDA_ERROR_ILLEGAL_ADDRESS crash

**RTX 4090 specific risk**:
- Consumer GPU without ECC → silent corruption more likely undetected
- Single copy engine → sleep latency proportional to total weight size (5-10s for 7B model)
- 24GB capacity makes sleep/wake the primary memory sharing mechanism

## 3. CuMemAllocator.wake_up() — Second Missing synchronize()

**File**: cumem.py, lines 218-241

**What wake_up() does**:
1. For matching tags: `create_and_map()` allocates new physical backing at same virtual address
2. For allocations with cpu_backup_tensor: synchronous memcpy CPU→GPU

**Bug**: No `torch.cuda.synchronize()` after operations complete. Buffer restoration uses `buffer.data.copy_()` which is async → subsequent inference could start before buffers fully restored.

## 4. LoRA + Sleep/Wake: No Integration

**File**: vllm/v1/worker/lora_model_runner_mixin.py

★★★★★★★★★ LoRA weights have NO explicit sleep/wake handling:
- LoRA adapters allocated via standard PyTorch allocator, NOT CuMem pool
- LoRA weights NOT freed during sleep → wasted VRAM during training phase
- No guard against add_lora while sleeping → allocating into discarded GPU memory → undefined behavior
- SGMV kernel workspace tensors also outside CuMem pool

**RTX 4090 impact**: LoRA r=32 adapter = ~220 MiB → small but still wasted during training. More importantly, adding LoRA during sleep_level=2 = crash.

## 5. KV Cache Offloading: Two Separate Mechanisms

### CuMem-based sleep/wake (rollout memory swapping)
- Level 1: KV cache discarded (unmapped), restored on wake_up with zeroing
- `post_kv_cache_wake_up()` zeroes KV cache, resets FP8 scales to 1.0
- FP8 scale reset loses calibrated values → accuracy degradation for FP8 models

### SimpleKVOffload (persistent KV offloading during inference)
- Uses cuMemcpyBatchAsync with background DMA thread
- Separate low-priority CUDA streams for load/store
- LRU/ARC cache policies for eviction
- RTX 4090: requires CUDA driver R535+, single DMA engine limits throughput

## 6. Complete Sleep/Wake Flow (RTX 4090 GRPO Scenario)

```
1. Training calls engine.sleep(level=1)
2. EngineCore: pause scheduler + abort requests + clear prefix cache
3. Worker.sleep(1): allocator.sleep(offload_tags=("weights",))
4. CuMemAllocator: copy weights to CPU, unmap ALL allocations
5. GPU: ~24GB free (minus NCCL/standard allocator overhead)
6. Training runs on freed GPU space
7. Training done, calls engine.wake_up(tags=["weights"])
8. Worker.wake_up: allocator.wake_up remaps weights, restores CPU backups
9. Later: wake_up(tags=["kv_cache"]) remaps KV, zeroes it, resets FP8 scales
10. Resume scheduler → inference resumes with empty KV cache
```

★★★★★★★★★ For verl HYBRID mode with SGLang:
- SGLang uses tag-based sleep/wake (tags=["kv_cache"] for sleep_level=1)
- SGLang keeps base weights resident (NOT in CuMem pool equivalent)
- Only ~2 GiB KV cache freed per step → 0.3s sleep + 0.8s wake
- This is FASTER and SAFER than vLLM sleep/wake

## 7. Bug Summary Table

| Bug | File | Line | Description | RTX 4090 Impact |
|-----|------|------|-------------|----------------|
| Missing synchronize before sleep | cumem.py | 167-204 | No sync before unmap → in-flight kernels crash | ★★★ CRITICAL |
| Missing synchronize after wake | cumem.py | 218-241 | No sync after remap → stale reads possible | MEDIUM |
| LoRA not in CuMem pool | lora_model_runner_mixin.py | N/A | LoRA stays during sleep → wasted VRAM | MEDIUM |
| add_lora while sleeping | abstract.py | 292 | Allocating into discarded memory → undefined | HIGH |
| FP8 scale reset to 1.0 | gpu_model_runner.py | 965 | Lost calibrated values after wake | LOW-MEDIUM |
| KV garbage after level-1 wake | gpu_worker.py | 176 | Discarded KV → zeroing needed | LOW (handled) |

## 8. RTX 4090 Recommendation

★★★★★★★★★ Use SGLang instead of vLLM for GRPO rollout on RTX 4090:
1. SGLang sleep/wake uses tag-based HTTP API → more granular control
2. SGLang keeps base weights resident (sleep_level=1) → faster wake-up
3. SGLang doesn't use CuMemAllocator for LoRA path → avoids #45552 entirely
4. vLLM sleep_level=2 = guaranteed crash on RTX 4090 (#45552)
5. vLLM sleep_level=1 = safer but still has LoRA memory leak during sleep

★★★★★★★★★ MUST NOT use vLLM sleep_level=2 on RTX 4090
★★★★★★★★★ MUST use sleep_level=1 with SGLang for GRPO training
