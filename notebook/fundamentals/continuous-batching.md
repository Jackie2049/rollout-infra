# Continuous Batching 深度解析

> LLM 推理服务的核心调度技术：从 static batching 到 continuous batching，为什么它让吞吐量翻倍

## 1. 背景：推理服务的批处理问题

### 1.1 LLM 推理的两个阶段

```
Prefill（预填充）:
  输入完整 prompt → 一次性计算所有 token 的 KV Cache
  特点: 计算密集（矩阵乘法），并行处理所有 prompt tokens
  时间: 与 prompt 长度成正比

Decode（解码）:
  每次 generate 一个 token → 用 KV Cache 做 attention → 输出下一个 token
  特点: 显存带宽瓶颈（每次读整个 KV Cache，只算一个 token）
  时间: 每个 token 一次 forward pass
```

### 1.2 关键特征：变长序列

```
请求 A: prompt=50 tokens,  生成 100 tokens  → 总长 150
请求 B: prompt=500 tokens, 生成 20 tokens   → 总长 520
请求 C: prompt=200 tokens, 生成 300 tokens  → 总长 500

问题:
  1. 各请求 prefill 时间差异大（50 vs 500 tokens）
  2. 各请求生成长度差异大（20 vs 300 tokens）
  3. 无法预知生成长度（需要生成才知道何时结束）
```

## 2. Static Batching（朴素方法）

### 2.1 做法

```
1. 等待凑齐一批请求（比如 batch_size=4）
2. Pad 所有请求到相同长度（padding 到最长的那个）
3. 一起跑 prefill
4. 一起跑 decode，每步所有请求都生成一个 token
5. 等所有请求都生成结束 → 返回结果 → 接收下一批
```

### 2.2 问题

```
Timeline (static batching, 3 requests):

Request A: [prefill][decode 20 tokens][=== idle ===]  ← 等 B, C 完成
Request B: [prefill ][decode 100 tokens]              ← 最长的
Request C: [prefill ][decode 50 tokens][== idle ==]   ← 等 B 完成

                    ↑ A 完成     ↑ C 完成    ↑ B 完成
                    |            |            |
                    |<-- 空等 -->|<-- 空等 -->|

问题 1: Padding 浪费
  - A 的 prompt 只有 50 tokens，pad 到 500
  - 浪费了计算和显存

问题 2: 空闲等待
  - A 生成 20 tokens 就结束了，但要等 B 生成 100 tokens
  - GPU 空闲时间占 50-70%

问题 3: 延迟高
  - 新请求必须等当前 batch 全部完成才能开始
  - 尾部延迟 = 最慢的请求
```

### 2.3 吞吐量分析

```
假设:
  batch_size = 4
  平均生成长度 = 100 tokens
  最长生成长度 = 300 tokens

Static batching 吞吐:
  实际处理时间 = 300 steps（等最慢的）
  有效 tokens = 4 × 100 = 400
  利用率 = 400 / (4 × 300) = 33%

浪费了 67% 的算力！
```

## 3. Continuous Batching（连续批处理）

### 3.1 核心思想

```
关键洞察: decode 阶段，每个请求的 token 生成是独立的
  → 不需要所有请求在同一个 step 都参与
  → 完成的请求可以立即退出，新请求可以立即加入

Continuous batching:
  每个 iteration:
    1. 检查哪些请求已经生成完毕（EOS）→ 从 batch 中移除
    2. 从等待队列中取新请求加入 batch
    3. 对新加入的请求做 prefill
    4. 对所有活跃请求做 decode（包括刚 prefill 完的）
    5. 生成的 token append 到各请求的输出
```

### 3.2 Timeline 对比

```
Static batching:
  A: [prefill][decode....................][idle][idle][idle]
  B: [prefill ][decode................................]
  C: [prefill ][decode..............][idle][idle][idle]
  D:                                        [等待整批完成...]

Continuous batching:
  A: [prefill][decode..]  ← 20 steps 后退出
  B: [prefill ][decode..........................]
  C: [prefill ][decode............]  ← 50 steps 后退出
  D:              [prefill][decode......]  ← A 退出后立即加入
  E:                      [prefill][decode...]  ← C 退出后加入

GPU 始终满载，没有空闲！
```

### 3.3 实现：Iteration-level Scheduling

```python
# 伪代码：continuous batching 核心循环

class ContinuousBatcher:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.max_batch_size = max_batch_size
        self.running = []       # 正在运行的请求
        self.waiting = Queue()  # 等待中的请求

    def step(self):
        # 1. 移除已完成的请求
        self.running = [r for r in self.running if not r.finished]

        # 2. 从等待队列补充新请求
        while len(self.running) < self.max_batch_size and not self.waiting.empty():
            req = self.waiting.get()
            self.running.append(req)

        if not self.running:
            return  # 没有请求

        # 3. 分离 prefill 和 decode 请求
        prefill_reqs = [r for r in self.running if r.num_tokens == 0]
        decode_reqs = [r for r in self.running if r.num_tokens > 0]

        # 4. Prefill: 处理新请求的 prompt
        if prefill_reqs:
            self.model.prefill(prefill_reqs)  # 填充 KV Cache

        # 5. Decode: 所有请求各生成一个 token
        new_tokens = self.model.decode(self.running)  # 一次 forward

        # 6. 更新状态
        for req, token in zip(self.running, new_tokens):
            req.append_token(token)
            if token == EOS:
                req.finished = True
```

## 4. 技术挑战与解决方案

### 4.1 变长 Attention

```
问题: continuous batching 中，各请求的序列长度不同
  请求 A: seq_len = 150
  请求 B: seq_len = 520
  请求 C: seq_len = 500

传统方法: padding 到最长 → 浪费计算
```

**解决方案：PagedAttention + Block Tables**

```
每个请求有自己的 block table:
  请求 A: [block_3, block_7]           → 2 blocks (32 tokens)
  请求 B: [block_1, block_5, block_8]  → 3 blocks (48 tokens)
  请求 C: [block_2, block_4, block_6]  → 3 blocks (48 tokens)

Attention kernel 根据 block table 读取各自的 KV blocks
→ 不需要 padding
→ 显存利用率接近 100%
```

### 4.2 Prefill 与 Decode 的混合调度

```
挑战:
  - Prefill 是 compute-bound（大量矩阵乘法）
  - Decode 是 memory-bound（读 KV Cache，算一个 token）
  - 两者同时运行时，如何分配 GPU 资源？

方案 1: 分时复用
  Odd iterations: 只做 prefill（可能有多个新请求）
  Even iterations: 只做 decode
  简单但延迟波动大

方案 2: 混合 batch（vLLM 的做法）
  同一个 iteration 中：
    - 部分请求做 prefill
    - 部分请求做 decode
  - Attention kernel 需要处理混合模式
  - 好处：延迟更平滑

方案 3: Chunked Prefill（vLLM V1 默认）
  将长 prompt 分成 chunks，每个 iteration 处理一个 chunk
  每个 iteration = 一个 chunk prefill + 所有 decode
  好处：
    - 避免长 prompt 占满 GPU 导致 decode 停滞
    - decode 延迟更稳定
    - 更好的 GPU 利用率
```

### 4.3 Chunked Prefill 详解

```
长 prompt: 4096 tokens, chunk_size = 512

Iteration 1: Prefill chunk 0 (tokens 0-511)  + Decode step for others
Iteration 2: Prefill chunk 1 (tokens 512-1023) + Decode step for all
...
Iteration 8: Prefill chunk 7 (tokens 3584-4095) + Decode step for all
Iteration 9: 开始 decode 该请求

好处:
  - 每个 iteration 的 prefill 计算量恒定
  - decode 不会被长 prompt 饿死
  - 可以同时 prefill 多个请求的不同 chunks

挑战:
  - 需要跨 chunk 维护 KV Cache（PagedAttention 天然支持）
  - prefix caching 可以跳过已计算的 chunks
```

### 4.4 抢占与 Swap

```
问题: GPU 显存有限，当 KV Cache 占满时怎么办？

方案 1: Preemption（抢占）
  - 暂停低优先级请求（如最早到达的）
  - 释放其 KV Cache blocks
  - 被抢占的请求放回等待队列
  - 后续需要重新计算 KV Cache（recomputation）

方案 2: Swapping（换出到 CPU）
  - 将低优先级请求的 KV Cache 拷贝到 CPU 内存
  - 释放 GPU blocks 给新请求
  - 当 GPU 有空余时，从 CPU 拷贝回来继续

方案 3: vLLM V1 策略 — 纯重计算（Pure Recomputation）
  - 只用 preemption + recomputation，不用 swap
  - 被抢占请求: 释放所有 KV blocks，num_computed_tokens 重置为 0
  - 放回等待队列 (优先队列前端)
  - 如果有 prefix caching，prefix 部分的 KV 可以跳过重计算
  - 优势: 实现简单，无 CPU-GPU 数据传输开销
  - 劣势: 重计算有额外开销，但通常比 swap 更快（PCIe 带宽限制）
```

## 5. Scheduling 策略

### 5.1 FCFS (First Come First Served)

```
简单但可能不最优:
  - 长请求先到 → 占用大量 blocks
  - 短请求排队等待

适合: 交互式聊天（公平性优先）
```

### 5.2 SJF (Shortest Job First)

```
优先处理短请求:
  - 吞吐量更高（短请求快速释放 blocks）
  - 但需要预估生成长度（不总是可靠）

适合: 批量推理（吞吐优先）
```

### 5.3 LPFS (Longest Prefix First)

```
优先调度与正在运行的请求有共享前缀的新请求:
  - 命中 prefix caching → 省 KV Cache 计算和显存
  - 适合 RLHF 场景（大量共享 prompt）

vLLM / SGLang 都支持这种调度策略
```

## 6. 性能数据

### 6.1 Static vs Continuous Batching

```
vLLM 论文数据 (LLaMA-13B, A100):

                    Static     Continuous    提升
Throughput          1000       2100          2.1x
(tok/s)
P99 Latency         12.3       5.8           2.1x 降低
(s)

NVIDIA TensorRT-LLM 数据 (LLaMA-70B, 4×H100):

                    Static     Continuous    提升
Throughput          800        2400          3.0x
(tok/s)
```

### 6.2 GPU 利用率

```
Static batching:
  GPU 利用率: 30-50%（大量时间在等待或 padding）
  显存利用率: 60-80%（碎片化 + padding 浪费）

Continuous batching + PagedAttention:
  GPU 利用率: 80-95%（几乎无空闲）
  显存利用率: ~100%（分页消除碎片，无 padding）
```

## 7. 各框架的 Continuous Batching 实现

### 7.1 vLLM

```python
# vLLM 自动使用 continuous batching
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
params = SamplingParams(max_tokens=100)

# 多个请求自动 continuous batching
outputs = llm.generate(["prompt 1", "prompt 2", "prompt 3"], params)
```

```
vLLM V1 调度器架构 (schedule() 方法):

1. new_step_starts() → KVCacheManager 准备新步
2. 处理 RUNNING 请求:
   - 计算 num_new_tokens (decode=1, prefill=min(remaining, threshold))
   - long_prefill_token_threshold 限制每步 prefill tokens
   - 分配 KV cache slots → 失败则触发抢占循环
3. 处理 WAITING 请求 (仅当没有抢占时):
   - 检查 prefix cache 命中
   - 计算剩余 tokens
   - 分配 slots → 提升到 RUNNING
4. 返回 SchedulerOutput

调度策略:
  FCFS: deque 等待队列, 抢占最后到达的请求
  PRIORITY: min-heap 按 (priority, arrival_time) 排序

关键参数:
  --max-num-seqs 256              # 最大并发序列数
  --max-num-batched-tokens 8192   # 每 iteration 最大 token 数
  --gpu-memory-utilization 0.9    # GPU 显存使用比例
```

### 7.2 SGLang

```
SGLang 的 continuous batching:
  - RadixAttention: 用 Radix Tree 管理前缀 → 更快的 prefix matching
  - Chunked Prefill: 默认开启
  - 更激进的调度策略 → 更高吞吐

性能: 在多个 benchmark 上吞吐比 vLLM 高 20-50%
```

### 7.3 TensorRT-LLM

```
NVIDIA 的优化:
  - Inflight Batching: NVIDIA 的 continuous batching 名称
  - CUDA Graphs 优化: 录制 decode kernel → 减少 CPU 开销
  - Tensor Core 优化: 针对 NVIDIA GPU 极致优化
```

## 8. 关键要点总结

1. **Continuous batching 是推理服务的基石** — 通过 iteration-level scheduling，GPU 利用率从 30-50% 提升到 80-95%
2. **与 PagedAttention 天然配合** — 分页管理解决变长序列和碎片化，continuous batching 解决请求级调度
3. **Chunked Prefill 是最新实践** — 将长 prompt 分块处理，避免饿死 decode 请求
4. **Prefix-aware 调度可以进一步提升** — 优先调度有共享前缀的请求，命中 prefix caching
5. **Preemption + Swap 保证稳定性** — 显存不足时优雅降级而非 OOM

## 模拟验证

- `tools/continuous_batching_sim.py` — Continuous Batching 模拟器（5 个实验）
  - 实验 1: Static vs Continuous 对比 (到达率 5-50 req/s)
  - 实验 2: 调度策略对比 (FCFS/SJF/Priority)
  - 实验 3: KV Cache 大小影响 (8K-128K tokens)
  - 实验 4: Batch Size 扩展效率 (4-64)
  - 实验 5: Chunked Prefill 阈值影响

关键发现:
- **TTFT**: Continuous 比 Static 好 10-100x (32-58ms vs 1231-3324ms)
- **KV Cache 压力**: 8K tokens → 66 次抢占/TTFT 4492ms; 32K+ → 0 抢占/TTFT 1162ms
- **Batch Size**: Continuous 吞吐一致 (~4450 tok/s, batch 8-64); Static 在 batch>32 后下降
- **高到达率 (50 req/s)**: Continuous 吞吐 4579 vs Static 3816 tok/s (+20%)

## 参考

- 论文: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM, SOSP 2023)
- 论文: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022, continuous batching 首次提出)
- 博客: [How continuous batching enables efficient LLM serving](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- 博客: [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
- SGLang: https://github.com/sgl-project/sglang
