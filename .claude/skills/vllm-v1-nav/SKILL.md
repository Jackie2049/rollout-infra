---
name: vllm-v1-nav
description: Navigate vLLM V1 source code. Quick reference for subsystem locations, key classes, and data flow.
triggers:
  - vllm v1 source
  - vllm architecture
  - vllm code location
  - vllm scheduler
  - vllm model runner
  - vllm attention
  - vllm kv cache
  - vllm weight loading
  - vllm engine core
  - vllm sampling
  - vllm spec decode
  - vllm ubatch
  - vllm DBO
---

# vLLM V1 Source Code Navigator

## Architecture: 4-Layer

```
Executor → Worker → GPUModelRunner → Model
```

## Subsystem Index

### 1. Scheduler (~2400 lines)
- **File**: `vllm/v1/core/sched/scheduler.py`
- **Notes**: `notebook/vllm/vllm-v1-scheduler.md` + `notebook/projects/vllm-v1-scheduler-reading.md`
- **Key**: FCFS + priority preemption, unified prefill/decode via `num_computed_tokens`

### 2. GPUModelRunner (~7550 lines)
- **File**: `vllm/v1/worker/gpu_model_runner.py`
- **Notes**: `notebook/vllm/vllm-v1-model-runner.md` + `notebook/projects/vllm-v1-executor-reading.md`
- **Key**: Mixin architecture, 12-step `execute_model` pipeline

### 3. Attention Backend
- **Files**: `vllm/v1/attention/backends/` (10+ backends)
- **Notes**: `notebook/vllm/vllm-v1-attention-system.md` + `notebook/projects/vllm-mla-backend-reading.md`
- **Key**: FlashInfer (primary), FlashMLA (SM90), Triton (fallback), platform-aware selection
- **MLA backends**: FlashMLA, FlashInfer (SM100), Triton, Cutlass (SM100)

### 4. KV Cache Manager
- **Files**: `vllm/v1/core/kv_cache_manager.py` + 5 supporting files (~11,700 lines total)
- **Notes**: `notebook/vllm/vllm-v1-kv-cache-manager.md` + `notebook/projects/vllm-v1-kv-cache-manager-reading.md`
- **Key**: BlockHashToBlockMap (1:N mapping), FreeKVCacheBlockQueue (O(1) doubly-linked list)
- **Flow**: Scheduler → Manager → Runner → Backend

### 5. Weight Loading
- **Files**: `vllm/v1/worker/gpu_model_runner.py` (loading section)
- **Notes**: `notebook/vllm/vllm-v1-weight-loading.md` + `notebook/projects/vllm-weight-loading-reading.md`
- **Key**: 5-stage pipeline (Build→Discover→Iterate→Map→Postprocess), EP pre-disk filter

### 6. Engine Core
- **File**: `vllm/v1/engine/engine_core.py`
- **Notes**: `notebook/projects/vllm-engine-core-reading.md`
- **Key**: `step()` loop (schedule→execute→sample→update), EngineCoreProc busy loop

### 7. Sampling Pipeline
- **File**: `vllm/v1/sample/sampler.py`
- **Notes**: `notebook/vllm/vllm-v1-sampling-pipeline.md` + `notebook/projects/vllm-sampling-pipeline-reading.md`
- **Key**: Sequential whitelist→penalties→temperature→min_p→top_k/p→sample, FlashInfer > Triton > PyTorch

### 8. Speculative Decoding
- **Files**: `vllm/v1/spec_decode/`
- **Notes**: `notebook/vllm/vllm-v1-spec-decode.md` + `notebook/projects/vllm-spec-decode-reading.md`
- **Key**: Proposer-Scorer dual layer, 8+ proposers (Eagle recommended, N-gram zero overhead)

### 9. UBatch / DBO (Comm-Compute Overlap)
- **Files**: `vllm/v1/worker/gpu_model_runner.py` (ubatch section)
- **Notes**: `notebook/vllm/vllm-v1-ubatch-mechanism.md`
- **Key**: 2 microbatches default, compute_stream+comm_stream, CUDA Event sync, SMControlContextManager
- **Limitation**: Designed for A100/H100 only (108+ SM + NVLink), NOT for A16

### 10. AsyncLLM Frontend
- **Files**: `vllm/v1/engine/async_llm.py`
- **Notes**: `notebook/projects/vllm-async-engine-frontend-reading.md`
- **Key**: Dual-process (asyncio+ZMQ), output_handler background Task, per-request RequestOutputCollector

### 11. DP Serving
- **Files**: `vllm/v1/engine/`
- **Notes**: `notebook/projects/vllm-data-parallel-reading.md`
- **Key**: DPEngineCoreProc (MoE only), Wave coordination, every 32 steps AllReduce

### 12. Quantization Pipeline
- **Files**: `vllm/quantization/`
- **Notes**: `notebook/projects/vllm-quantization-pipeline-reading.md`
- **Key**: 3-layer (Config→Method→Kernel), QuantKey dispatch, 30+ methods

### 13. CUDA Graph
- **Files**: `vllm/v1/worker/`
- **Notes**: `notebook/projects/vllm-cuda-graph-reading.md`
- **Key**: 5 modes, CudagraphDispatcher, Breakable Graph

### 14. LoRA Serving
- **Files**: `vllm/lora/`
- **Notes**: `notebook/projects/vllm-lora-serving-reading.md`
- **Key**: Punica Multi-LoRA, lora_shrink/expand Triton kernel, LRU eviction

### 15. P/D Disaggregation
- **Files**: `vllm/v1/spec_decode/` + `vllm/distributed/kv_transfer/`
- **Notes**: `notebook/projects/vllm-pd-disaggregation-reading.md` + `notebook/projects/vllm-kv-transfer-connector.md`
- **Key**: NIXL connector, async DMA+ZMQ, heterogeneous TP, KV Lease delayed release

## Key Data Flow

```
User Request
  → AsyncLLM (asyncio+ZMQ)
  → EngineCore.step()
    → Scheduler.schedule() → SchedulerOutput
    → GPUModelRunner.execute_model() → ModelOutput
    → Sampler.sample() → SamplerOutput
  → RequestOutput → User
```

## Critical Paths

- **Prefill**: hidden_states → RadixAttention → FlashInfer backend → KV write
- **Decode**: hidden_states → RadixAttention → FlashInfer paged decode → sampling
- **KV Cache**: BlockPool → FreeQueue → BlockHashToBlockMap → LRU eviction
- **Weight**: Config → Method → Kernel → TP narrow + copy_ → EP pre-disk filter
