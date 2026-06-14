# SGLang Serving Architecture Deep Reading

> Date: 2026-06-14 | Context: GPU offline,6th session | Author: Jackie2049

## 1. Request Lifecycle

### Full Request Flow

```
HTTP Client → FastAPI (http_server.py) → Engine → TokenizerManager → ZMQ IPC → Scheduler (subprocess) → ModelRunner → GPU Forward → Result → DetokenizerManager → ZMQ → TokenizerManager → HTTP Response
```

**3-process Architecture** (vs vLLM V1 2-process):
1. **TokenizerManager** (主进程): tokenizes input, routes requests, handles response
2. **Scheduler** (子进程, GPU): scheduling + batching + forward + KV management
3. **DetokenizerManager** (子进程): decodes output tokens → text

IPC via **ZMQ** (每个进程不同端口).

### ForwardMode 7种

| Mode | Description | Use Case |
|------|-------------|----------|
| EXTEND | Prefill (extend KV cache) | New request processing |
| DECODE | Single token generation | Ongoing inference |
| MIXED | EXTEND + DECODE mixed | Chunked prefill |
| IDLE | No sequences | DP attention idle worker |
| TARGET_VERIFY | Verify speculative drafts | Spec decoding verify |
| DRAFT_EXTEND_V2 | Draft model extend | Spec decoding draft |
| PREBUILT | KV cache ready (disagg) | PD separation decode start |

## 2. Scheduler Architecture

### Class: Scheduler (4145 lines!)

**Mixin继承体系** (vs vLLM 单一EngineCore):
```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,    # PD分离decode
    SchedulerDisaggregationPrefillMixin,   # PD分离prefill
    SchedulerMultiplexMixin,               # PD多路复用
    SchedulerPPMixin,                      # Pipeline Parallel
    SchedulerDllmMixin,                    # DLLM (diffusion LLM)
    SchedulerMlxOverlapMixin,              # MLX overlap
):
```

**Event Loop选择** — `dispatch_event_loop()` 根据3种条件选择:
1. `enable_pdmux` → `event_loop_pdmux()` — PD多路复用
2. `pp_size > 1` → `event_loop_pp()` — Pipeline Parallel
3. `enable_overlap_mlx` → `event_loop_overlap_mlx()` — MLX overlap
4. `enable_overlap` → `event_loop_overlap()` — **标准overlap** (最核心!)
5. else → `event_loop_normal()` — 无overlap

### Event Loop: Normal vs Overlap

**Normal** (`event_loop_normal`):
```
while True:
    recv_reqs → process_input_requests
    batch = get_next_batch_to_run()
    if batch:
        result = run_batch(batch)
        process_batch_result(batch, result)
    else:
        on_idle()
    last_batch = batch
```

**Overlap** (`event_loop_overlap`) — **SGLang核心创新**:
```
while True:
    recv_reqs → process_input_requests
    batch = get_next_batch_to_run()
    if overlap disabled for batch:
        pop_and_process()  # 立即处理上一批结果
    # Launch GPU forward
    if batch:
        batch_result = run_batch(batch)
        result_queue.append((batch.copy(), batch_result))
    # Process LAST batch result (CPU处理与GPU计算重叠)
    if last_batch and not overlap disabled:
        pop_and_process()
    # Launch sample for current batch
    launch_batch_sample_if_needed(batch_result)
    last_batch = batch
```

**关键设计**:
- GPU计算(forward)与CPU处理(process_batch_result) **并行**
- War Barrier: schedule_stream.wait_stream(forward_stream) — 防止schedule写入与forward读取冲突
- **FutureMap** — pool-indexed relay, 跨迭代传递output_tokens/new_seq_lens
- 连续prefill时禁用overlap → 改善TTFT

## 3. KV Cache Architecture

### Radix Cache (vs vLLM PagedAttention)

**UnifiedRadixCache** — SGLang改进版:
- `radix_cache.py`: 树结构, prefix sharing
- `unified_radix_cache.py`: 统一接口, 支持 HiCache

**HiCache** — 3层缓存(GPU KV + CPU DRAM + SSD):
```
GPU KV Cache (fast, expensive) → CPU DRAM (medium) → SSD/NVMe (slow, cheap)
```

**HiRadixCache** — SGLang特有:
- decode时查询HiCache → CPU DRAM命中 → 减少GPU内存占用
- 支持async background offload → decode不停顿

**HiSparse** — 稀疏注意力协调器:
- 长上下prefill → 稀疏KV → decode时继续计算
- 不阻塞running decode → 颋新的推理方式

### KV Allocator

**ReqToTokenPool** → `req_to_token` mapping (request → token slots)
**TokenToKVPoolAllocator** → page-based KV allocation with:
- `allocator.alloc(num_tokens)` → allocate new pages
- `allocator.free(sorted_free_indices)` → evict + reuse
- Eviction: LRU + prefix-aware (优先evict无prefix共享的)

## 4. Speculative Decoding

### 6种SpecAlgorithm (vs vLLM EAGLE)

| Algorithm | Worker | Draft Method | Key Feature |
|-----------|--------|--------------|-------------|
| EAGLE/EAGLE3 | EAGLEWorkerV2 | Hidden state → draft tree | Top-k tree, verify+draft extend |
| DFlash | DFlashWorkerV2 | Draft via attention pattern | Draft-Flash attention |
| FROZEN_KV_MTP | FrozenKVMTPWorkerV2 | Frozen KV + multi-token prediction | MTP draft |
| STANDALONE | StandaloneWorkerV2 | Vanilla draft | Simple standalone |
| NGRAM | NGRAMWorker | N-gram matching | Tree-based verify |

**SpecInput** → abstract class with SpecInputType:
- EAGLE_DRAFT / EAGLE_DRAFT_EXTEND / EAGLE_VERIFY
- FROZEN_KV_MTP_DRAFT / FROZEN_KV_MTP_VERIFY
- DFLASH_DRAFT / DFLASH_VERIFY
- NGRAM_VERIFY

**SpecInputType → Worker dispatch**:
```python
spec_algorithm.create_worker(server_args) → WorkerClass
```

**Plugin registration**: `SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=...)`

## 5. Disaggregation (PD Separation)

### Architecture

```
Prefill Worker (专做prefill) → KV Transfer → Decode Worker (专做decode)
```

**3种Transfer Backend**:
1. **NIXL** (NVIDIA) → RDMA-based, GPU→GPU direct
2. **Mooncake** → RDMA + TCP, TransferEngine zero-copy
3. **Fake** → in-process, 测试用

**Prefill Flow**:
```python
SchedulerDisaggregationPrefillMixin:
    event_loop_overlap_disagg_prefill():
        recv_reqs → process_input_requests (tokenize + bootstrap)
        batch = get_next_batch_to_run() → prefill forward
        send KV via KVSender (RDMA/NIXL/Mooncake)
        → Decode worker接收KV → BootstrapReady → start decode
```

**Decode Flow**:
```python
SchedulerDisaggregationDecodeMixin:
    event_loop_overlap_disagg_decode():
        recv KV from prefill worker (KVReceiver)
        bootstrap → mark ready
        batch = get_next_batch_to_run() → prebuilt forward (no prefill!)
        normal decode loop after bootstrap
```

**KV Transfer路径对比**:
| Path | Latency | Use Case |
|------|---------|----------|
| TCP | 5-15ms | 测试/简单部署 |
| RDMA | 0.5-2ms | 生产部署 |
| GPUDirect | 0.05-0.5ms | NVLink同节点 |
| NVLink | 0.1ms | 最优(需rail-aligned拓扑) |

## 6. SGLang vs vLLM V1 — Key Differences

| Feature | SGLang | vLLM V1 |
|---------|--------|---------|
| **Process** | 3-process (Tokenizer+Scheduler+Detokenizer) | 2-process (API+EngineCore) |
| **Overlap** | FutureMap+WAR barrier+stream-based | No overlap (serial) |
| **Spec Decode** | 6 algorithms + plugin system | EAGLE only |
| **KV Cache** | Radix+HiCache(3层)+HiSparse | PagedAttention only |
| **PD Separation** | Full(Prefill+Decode+KV transfer+bootstrap) | Experimental |
| **Env Var** | EnvField API (1061行environ.py) | Simple os.getenv |
| **Radix Sharing** | Prefix sharing + automatic eviction | Manual prefix |
| **Grammar** | GrammarManager + constrained decoding | Limited support |
| **LoRA** | LoRAOverlapLoader + LoRADrainer | Basic LoRA |

### SGLang's Key Innovations

1. **Overlap Scheduling**: GPU-CPU parallel → 20-40% throughput improvement
   - FutureMap: pool-indexed relay, zero-copy cross-iter data passing
   - WAR Barrier: schedule_stream.wait_stream(forward_stream) → safe concurrent access

2. **Radix Cache + HiCache**: 3层缓存 → decode时offload KV to CPU/SSD
   - decode不停顿(async background offload)
   - GPU内存释放 → 支持更长上下文

3. **Plugin-based Speculative Decoding**: 6种算法 + extensible registry
   - vs vLLM: 只有EAGLE

4. **EnvField API**: 严格环境变量管理
   - .get()/.set()/.override() context manager
   - temp_set_env() 拒绝SGLANG_* 直接修改
   - vs vLLM: 直接os.getenv()

5. **Mixin Architecture**: 功能组合 vs 单一大类
   - Disaggregation/PP/DLLM/MLX → 独立mixin
   - vs vLLM: EngineCore单类包含所有逻辑

## 7. RTX 4090 Deployment

### Optimal Configuration
```
7B INT4 + INT8 KV + GQA-8 + FlashInfer B=118 → 4,791 tok/s
Overlap scheduling → 20-40% throughput improvement
Radix sharing → prefix reuse (system prompt等)
HiCache → decode KV offload to CPU → 50% GPU memory saving
```

### Limitations
- No NVLink/RDMA → PD separation only via TCP (5-15ms)
- No Hopper → CUTLASS SM80 (cp_async+HMMA), not SM90 TMA+WGMMA
- 24GB → 7B INT4 optimal; larger models need INT4+CPU offload
