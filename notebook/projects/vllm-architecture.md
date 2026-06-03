# vLLM 架构深度阅读笔记

> 基于 vLLM V1 源码的架构分析

## 1. 整体架构

vLLM V1 采用**多进程架构**：

```
[API Server Process]
  └─ AsyncLLM
       ├─ InputProcessor (分词、多模态特征提取)
       ├─ OutputProcessor (去分词、logprob)
       └─ EngineCoreClient (ZMQ 通信)
            ↕ ZMQ
[EngineCore Process]                    ← 核心引擎进程
  ├─ Scheduler                          ← 调度器
  │    ├─ KVCacheManager                ← KV 缓存管理
  │    │    └─ BlockPool                 ← 物理块池
  │    └─ RequestQueue                   ← 请求队列
  └─ ModelExecutor                       ← 模型执行器
       ↕ multiprocessing / shared memory
  [Worker 0] [Worker 1] ... [Worker N]  ← 每个 GPU 一个 Worker
    └─ GPUModelRunner                    ← 模型执行
         ├─ Model (PyTorch)              ← 模型权重
         └─ KV Caches (Paged)            ← 分页 KV 缓存
```

## 2. 请求生命周期

```
1. HTTP Request → API Server (FastAPI)
2. AsyncLLM.generate() → InputProcessor → EngineCoreRequest
3. ZMQ → EngineCore.add_request() → Scheduler 等待队列
4. EngineCore.step() 循环：
   a. Scheduler.schedule() → SchedulerOutput (token IDs + block tables)
   b. Executor.execute_model() → Worker → GPUModelRunner → 前向 + 采样
   c. Scheduler.update_from_output() → EngineCoreOutputs
5. ZMQ → AsyncLLM → OutputProcessor → RequestOutput → HTTP Response
```

## 3. 核心组件详解

### 3.1 EngineCore (`v1/engine/core.py`)

vLLM 的内循环，运行在独立进程中。

**step() 方法（核心迭代）：**
```python
def step(self):
    scheduler_output = self.scheduler.schedule()
    model_output = self.model_executor.execute_model(scheduler_output)
    outputs = self.scheduler.update_from_output(scheduler_output, model_output)
    return outputs
```

### 3.2 Scheduler (`v1/core/sched/scheduler.py`)

决定每个 iteration 处理哪些请求、分配多少 token。

关键职责：
- 管理 RequestQueue（支持 FCFS 和优先级策略）
- KV 缓存块分配
- 前缀缓存命中查找
- 抢占（内存紧张时逐出请求）
- 分块预填充 (chunked prefill)

### 3.3 Executor 层

| 类型 | 用途 |
|------|------|
| UniProcExecutor | 单 GPU 调试 |
| MultiprocExecutor | 多 GPU (TP/PP) |
| RayDistributedExecutor | Ray 分布式 |

### 3.4 Worker (`v1/worker/gpu_worker.py`)

每个 GPU 一个 Worker 进程。关键方法：
- `init_device()` — 初始化 CUDA
- `load_model()` — 加载权重
- `determine_available_memory()` — 显存分析
- `execute_model()` — 前向 + 采样

### 3.5 GPUModelRunner (`v1/worker/gpu_model_runner.py`)

Worker 内部的模型执行引擎：
- 将 SchedulerOutput 转为模型输入张量
- CUDA graph 捕获优化
- LoRA、多模态、投机解码支持
- 采样 (sample_tokens)

## 4. PagedAttention 机制

### 4.1 核心思想

借鉴 OS 虚拟内存分页：
- KV Cache 被划分为**固定大小的 block**
- 每个 block 存储 `block_size` 个 token 的 K/V（如 16/32/64 个 token）
- 按需分配，不需要预分配最大长度

### 4.2 块管理架构

```
Scheduler 侧 (CPU):
  KVCacheManager
    ├─ KVCacheCoordinator
    │    ├─ NoPrefixCache / Unitary / Hybrid
    │    └─ SingleTypeKVCacheManager
    │         ├─ FullAttentionManager
    │         ├─ SlidingWindowManager
    │         └─ MambaManager
    └─ BlockPool
         ├─ FreeKVCacheBlockQueue (双向链表)
         └─ BlockHashToBlockMap (哈希 → 块缓存)

Worker 侧 (GPU):
  BlockTable [num_reqs, max_blocks]
    └─ slot_id = block_number * block_size + offset
```

### 4.3 块分配流程

1. `Scheduler` 调用 `KVCacheManager.allocate_slots()`
2. 计算需要的块数：`ceil(num_tokens / block_size)`
3. `BlockPool.get_new_blocks(n)` 从空闲队列弹出
4. 每个 block 的 `ref_cnt` +1

### 4.4 前缀缓存

**Merkle 风格哈希链：**
```
block_hash = hash(parent_block_hash, token_ids_in_block, extra_keys)
```

- 请求到达时，`find_longest_cache_hit()` 查找最长匹配前缀
- 匹配的 block 直接复用（`ref_cnt` +1），零额外开销
- 支持多模态、LoRA 等场景的额外 key

### 4.5 驱逐与释放

- **驱逐**：空闲列表空时，LRU 驱逐 `ref_cnt == 0` 的缓存块
- **释放**：请求完成时逆序释放（尾部优先，保持 LRU 语义）
- 滑动窗口管理器会动态释放超出窗口的 block

### 4.6 传统 vs PagedAttention

| 指标 | 传统 | PagedAttention |
|------|------|----------------|
| KV Cache 浪费 | 60-80% | < 4% |
| 分配方式 | 静态预分配 | 动态按需分页 |
| 前缀共享 | 不支持 | 零开销共享 |

## 5. 支持的注意力类型

| 类型 | 管理器 | 块策略 |
|------|--------|--------|
| FullAttention | FullAttentionManager | 保留至请求完成 |
| SlidingWindow | SlidingWindowManager | 超出窗口的释放 |
| ChunkedLocal | ChunkedLocalAttentionManager | 块边界回收 |
| Mamba | MambaManager | 仅保留最新状态 |
| CrossAttention | CrossAttentionManager | 不共享 |
| SinkFull | SinkFullAttentionManager | 保留初始 sink 块 |

## 6. 关键文件索引

| 组件 | 路径 |
|------|------|
| EngineCore | `vllm/v1/engine/core.py` |
| AsyncLLM | `vllm/v1/engine/async_llm.py` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| KVCacheManager | `vllm/v1/core/kv_cache_manager.py` |
| BlockPool | `vllm/v1/core/block_pool.py` |
| GPUWorker | `vllm/v1/worker/gpu_worker.py` |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` |
| BlockTable (GPU) | `vllm/v1/worker/gpu/block_table.py` |
| VllmConfig | `vllm/config/vllm.py` |

## 7. 学习要点

- vLLM 的核心创新是 **PagedAttention**：用 OS 分页思想管理 KV Cache
- **V1 架构**采用集中式调度 + 分布式 Worker，比 V0 更清晰
- 前缀缓存的 Merkle 哈希链设计非常巧妙
- BlockPool 是共享的，不同注意力类型的 Manager 从同一个池分配
- 未来趋势：分离式预填充/解码 (disaggregated prefill/decode)
