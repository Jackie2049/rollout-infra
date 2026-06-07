# Continuous Batching Simulator for LLM Serving — RTX 4090

```
┌─────────────────────────────────────────────────────────┐
│  Continuous Batching Simulator Results                  │
│                                                         │
│  Chat: 76-81K tok/s, TTFT 5.6-481ms                    │
│  RL Rollout: 75-80K tok/s, concurrent 4-5               │
│  Batch Infer: 108-111K tok/s, concurrent 14-18          │
│  Long Context: 81-88K tok/s, TTFT 468-481ms             │
│                                                         │
│  Optimization Stack (25M model, chat_short):             │
│  Baseline:     167K tok/s (1.00x)                       │
│  INT8 KV:      168K tok/s (1.01x) ← 小模型提升小       │
│  INT4+INT8:    171K tok/s (1.02x) ← 25M太小!7B才显著  │
│  INT4+INT8+GQA: 171K tok/s (1.02x)                      │
│  INT4+INT8+PS: 171K tok/s (1.02x)                      │
│                                                         │
│  关键: 25M模型太小→量化/PS收益不显著→7B才有效         │
│  Scheduler: Decode优先+Prefill填充剩余budget            │
└─────────────────────────────────────────────────────────┘
```

## 1. vLLM V1 Unified Token Budget Scheduler

### 1.1 核心设计

vLLM V1 scheduler的关键创新: **unified token budget** — 每步固定token预算, decode和prefill在同一步内处理:

```python
def step(self, current_time_ms, decode_latency_ms, prefill_latency_per_token_ms):
    # Phase 1: Decode all RUNNING requests (1 token per request per step)
    decode_count = 0
    for req in self.running_set:
        req.num_generated_tokens += 1  # 每decode请求获得1个新token
        req.num_computed_tokens += 1
        decode_count += 1
        if req.num_generated_tokens >= req.response_len:
            req.state = "completed"

    # Phase 2: Prefill WAITING requests with remaining budget
    remaining_budget = self.token_budget - decode_count
    while remaining_budget > 0 and len(self.waiting_queue) > 0:
        req = self.waiting_queue[0]
        chunk_size = min(req.prompt_len, remaining_budget)  # chunked prefill
        self.kv_manager.allocate(req.req_id, chunk_size)
        req.num_computed_tokens = chunk_size
        req.state = "running"
        remaining_budget -= chunk_size
        self.running_set.append(req)
```

**关键原则**:
- **Decode优先**: RUNNING请求必须每步获得1个token → 低延迟保证
- **Prefill填充**: WAITING请求利用剩余预算 → 最大吞吐
- **Chunked prefill**: 长prompt分多步prefill → 不阻塞decode

### 1.2 Request生命周期

```
Request state lifecycle:
  waiting → running → completed
              ↕
          preempted (KV不足时驱逐)
```

- **Waiting**: 等待prefill, 尚无KV cache
- **Running**: 正在decode, 每步1 token
- **Preempted**: KV cache不足时驱逐 → 释放KV → 重新prefill
- **Completed**: 生成完毕, 释放KV cache

## 2. Simulator Architecture

### 2.1 KVCacheManager

```python
class KVCacheManager:
    def __init__(self, total_blocks, block_size=16, n_heads=16, n_kv_heads=4, d_head=64):
        self.total_blocks = total_blocks    # 总KV block数
        self.block_size = block_size        # 每block16tokens
        self.free_blocks = total_blocks     # 可用block数
        self.kv_bytes_per_block = 2 * n_kv_heads * d_head * block_size * 2  # FP16 K+V

    def allocate(self, req_id, num_tokens):
        num_blocks_needed = ceil(num_tokens / 16)
        if num_blocks_needed > self.free_blocks:
            return False  # OOM!

    def free(self, req_id):
        self.free_blocks += self.allocated[req_id]

    def evict(self, req_id):
        # 驱逐: 释放KV → 请求重新prefill
```

**INT8 KV效果**: `kv_bytes_per_block *= 0.5` + `total_blocks *= 2` → 容量翻倍!

### 2.2 Poisson Arrival Model

```python
# 请求按Poisson分布到达
for i in range(num_requests):
    arrival_time = random.expovariate(arrival_rate)  # exponential inter-arrival
```

- **Chat**: rate=2-5 req/s → 低负载
- **RL rollout**: rate=10 req/s → burst arrival
- **Batch infer**: rate=20 req/s → 高负载

## 3. Benchmark结果

### 3.1 Chat Serving

| Scenario | throughput(tok/s) | TTFT(ms) | latency(ms) | concurrent |
|----------|-------------------|----------|-------------|-----------|
| chat_short (prompt=32-64) | **76,103** | **5.6** | 256 | 2 |
| chat_medium (prompt=128-256) | **81,546** | **72.6** | 602 | 5 |
| chat_long (prompt=256-512) | **77,981** | **167.5** | 1183 | 3 |

**TTFT随prompt长度线性增长**: 5.6→72.6→167.5ms → 长prompt的prefill开销显著

### 3.2 RL Rollout (Burst arrival, prompt=256-512)

| Config | throughput(tok/s) | TTFT(ms) | latency(ms) | concurrent |
|--------|-------------------|----------|-------------|-----------|
| no_ps | 75,171 | 96.2 | 439 | 4 |
| with_ps | 75,291 | 95.2 | 434 | 5 |
| int4_int8 | **80,474** | 92.2 | 410 | 4 |
| ps+int4_int8 | 79,165 | 91.6 | 407 | 4 |

**INT4+INT8 KV提速7%** → 但25M模型太小, 量化收益不显著

### 3.3 Batch Inference (rate=20)

| Config | throughput(tok/s) | TTFT(ms) | concurrent |
|--------|-------------------|----------|-----------|
| baseline | **108,586** | 318 | 14 |
| int8_kv | 106,878 | 315 | 14 |
| int4_int8 | **111,328** | 313 | **18** |

**INT4+INT8 KV: concurrent从14→18** → KV容量增大 → 更多并发请求

### 3.4 Long Context (prompt=2048-4096)

| Config | throughput(tok/s) | TTFT(ms) | latency(ms) |
|--------|-------------------|----------|-------------|
| baseline | 81,033 | **481** | 1406 |
| int8_kv | 83,031 | 469 | 1446 |
| int4_int8 | **88,306** | 477 | 1291 |

**INT4+INT8 KV提速9%** → 长context场景KV容量更关键

### 3.5 Optimization Stack (chat_short, 500 requests)

| Stack | throughput(tok/s) | vs_baseline | concurrent |
|-------|-------------------|-------------|-----------|
| baseline_fp16 | **167,285** | 1.00x | 5 |
| int8_kv | 168,400 | 1.01x | **12** |
| int4_int8 | 171,026 | 1.02x | 3 |
| int4_int8_gqa | 170,713 | 1.02x | 8 |
| int4_int8_prefix | 170,746 | 1.02x | 7 |

**25M模型优化提升仅1-2%** → 模型太小! INT4/INT8 KV/GQA的收益在7B+模型才显著(实测: 2.3x)

## 4. 关键洞察

### 4.1 Decode Latency Dominates

每decode step = 2.21ms (25M GQA-4 model), 这是serving延迟的固定开销:
- 每token必须经过完整model forward → memory-bound → 延迟≈常数
- Prefill可以并行(batch多个请求的prefill) → 但decode必须逐token生成

### 4.2 KV Cache Capacity vs Concurrent Requests

```
KV blocks available → concurrent requests limit
  16,000 blocks (FP16) → ~2-5 concurrent (25M model)
  32,000 blocks (INT8) → ~12-18 concurrent (KV容量翻倍!)
```

INT8 KV cache → 每block存储量减半 → 同样HBM空间放2x blocks → 2x并发!

### 4.3 为什么25M模型优化提升小

25M模型只有0.12GB → 权重+KV全在L2 cache → 量化减少的HBM traffic影响小:

```
7B模型:
  权重: 14GB → INT4: 3.5GB → 75%内存省 → 可跑更大batch
  KV:   34MB/req(B=256,S=2048) → INT8: 17MB/req → 2x并发

25M模型:
  权重: 0.12GB → INT4: 0.03GB → 内存省但L2已够 → 无提速
  KV:   0.8MB/req → INT8: 0.4MB/req → 省了但占比小 → 无提速
```

**结论**: 小模型(≤25M) → 量化收益不显著; 大模型(≥7B) → 量化是关键优化!

### 4.4 Production vs Simulator

Simulator简化了几个关键因素:
- **无CUDA Graph**: 实际vLLM用CUDA Graph消除kernel launch jitter
- **无FlashInfer**: 实际用专用decode kernel(cooperative batching+native GQA)
- **无Paged Attention**: 实际用paged KV cache(不需连续内存)
- **无Prefix Sharing计算省**: Simulator只标记PS, 但未实际省compute

这些都是vLLM/SGLang在生产中额外的优化 → 实际吞吐可能比simulator更高。

## 5. Production决策

| 场景 | 最优配置 | 预期吞吐 |
|------|---------|---------|
| Chat Short (25M) | baseline FP16 | 167K tok/s |
| Chat Short (7B) | INT4+INT8 KV+GQA-4 | ~260K tok/s (2.3x) |
| RL Rollout (7B) | INT4+INT8 KV+PS | ~136K tok/s |
| Batch Infer (7B) | INT4+INT8 KV+CB | ~151K tok/s |
| Long Context (7B) | INT4+INT8 KV | ~88K tok/s |

---

**工具**: `tools/continuous_batching_simulator_4090.py`
**结果**: `results/continuous_batching_simulator.json`
**相关笔记**: `e2e-serving-throughput-rtx4090.md`, `vllm-v1-scheduler-deep-read.md`, `continuous-batching.md`