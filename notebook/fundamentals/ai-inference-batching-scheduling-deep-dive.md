# AI Inference Batching & Scheduling Deep Dive

> 2026-06-10 | 批处理+调度=推理吞吐的灵魂! 从no batching到continuous batching, 从FCFS到priority hybrid, 从token budget到SLO-aware, 调度=吞吐×公平!
> 关联: scheduler-architecture-deep-dive.md, continuous-batching-simulator.md, long-context-serving-deep-dive.md, kv-cache-management-deep-dive.md

## 0. 核心定律: 调度 = 吞吐 × 公平 的平衡艺术

单请求推理不需要调度 → 但生产级serving → 调度=决定因素!

关键数据(RTX 4090实测+模拟):
- B=1 → 79 tok/s → B=32 → 2,468 tok/s → **31.4x吞吐提升!**
- B=118 → 4,791 tok/s → INT4+INT8KV → **60x吞吐提升!**
- Continuous batching → 76K-171K tok/s → 2-18 concurrent → 吞吐↑
- Prefill S=2048 → 326% ITL增加 → 调度需隔离prefill → chunked prefill!
- S↑ → KV↑ → 并发↓ → 吞吐↓ → **线性反比!** → 长上下文=零和博弈!

## 1. Batching Strategies — 批处理策略演进

```
批处理4代演进:

1. No Batching (最原始):
   → 1请求→1forward→返回 → B=1 → 吞吐最低!
   → → → GPU利用率0.5% → 资源浪费 → 不可生产!
   → → → → → 延迟=最低 → 单请求最优 → 但吞吐灾难!

2. Static Batching (传统):
   → 等凑够B=N → 一次性forward → 返回 → 下一batch!
   → → → 问题: 请求长度不同 → 等最长的完成 → padding浪费!
   → → → → → 等凑够时间→增加延迟 → 等最长→浪费计算!
   → → → → → → → 小batch→等凑→延迟↑ / 大batch→等完成→浪费↑!

3. Continuous Batching / Iteration-level (vLLM/SGLang):
   → 每步(iteration)动态组batch → 完成的请求→退出 → 新请求→加入!
   → → → 不等凑够 → 不等完成 → 每步动态 → 吞吐最优!
   → → → → → vLLM V1: unified token budget → decode优先+prefill填充!
   → → → → → → → SGLang: PrefillAdder → conservative调度 → ITL更稳定!

4. Chunked Continuous Batching (vLLM V1):
   → prefill→chunk→多步完成 → decode→每步 → 混合!
   → → → 长prefill→chunk S=512 → 4步 → 不阻塞decode → 公平!
   → → → → → 但: chunk→ITL增加32% → 限制chunk≤512 → 可接受!

RTX 4090 batching关键数据:
  → B=1 BF16: 79 tok/s → 0.5% GPU利用 → 灾难!
  → → → B=32 BF16: 2,468 tok/s → 37.2% GPU利用 → 生产级!
  → → → → → B=118 INT4+INT8KV: 4,791 tok/s → 最高吞吐!
  → → → → → → → 吞吐∝B → 线性增长 → memory-bound → batch=带宽利用!
  → → → → → → → → → 最优: B=32-118 → 吞吐×延迟平衡 → RTX 4090甜蜜点!
```

## 2. Token Budget — vLLM V1调度核心

```
vLLM V1 unified token budget模型:

核心思想:
  → 每步(iteration) → 固定token budget → 分配给请求!
  → → → budget = forward_pass_capacity → GPU每步能处理的token数!

分配策略:
  → Step 1: 优先decode → 每decode=1 token → 先分配!
  → → → decode预算 = min(budget, num_decode_tokens)!
  → → → → → 为什么decode优先? → decode延迟敏感 → ITL SLO → 必须保证!

  → Step 2: 剩余budget → 分配prefill → 每prefill=S tokens!
  → → → 剩余 = budget - decode_tokens → 分给prefill!
  → → → → → chunked prefill: S=2048 → chunk=512 → 4步 → 不一次性!

Budget数学:
  → budget ≈ 128 tokens (理论)
  → → → decode: 118 tokens → 剩余=10 → prefill仅10tok → 太少!
  → → → → → 如果保留chunk=512 → decode≤128-512 → 不够!

  → → → → → → → RTX 4090最优: decode≤55 → 剩余73 → chunk=512 → 每7步完成1个prefill!

Budget vs SLO:
  → ITL SLO < 100ms → decode必须每步完成 → budget≥decode_tokens!
  → → → TTFT SLO < 500ms → prefill可以多步 → 但不能太多!
  → → → → → 优先级: decode ITL > prefill TTFT → decode优先分配!
```

## 3. Scheduling Algorithms — 调度算法

```
AI serving调度算法对比:

1. FCFS (First Come First Serve) — vLLM V1默认:
   → 按到达时间排序 → 先来先服务 → 简单!
   → → → 优点: 简单实现 → 无排序开销 → 公平(时间)!
   → → → → → 缺点: 短请求等长请求 → TTFT不均 → 长请求阻塞!

2. SJF (Shortest Job First):
   → 预估长度 → 短请求优先 → 平均TTFT最低!
   → → → 优点: 平均延迟最低 → 吞吐最高(完成快→释放slot)!
   → → → → → 缺点: 长请求饥饿 → 需要预估长度 → 不精确!

3. Priority-based Scheduling:
   → 请求→优先级 → 高优先级→先调度 → VIP/SLA分级!
   → → → 优点: SLA分级 → VIP优先 → 商业可行!
   → → → → → 缺点: 低优先级饥饿 → 需最小保障 → 复杂!

4. Preemptive Scheduling — vLLM V1 eviction:
   → 新请求→无slot → evict旧请求 → 释放KV → 新请求加入!
   → → → eviction策略: recompute(默认) / swap(快4-44x但PCIe慢)!
   → → → → → vLLM: FreeKVCacheBlockQueue(LRU) → 最早未使用→evict!

5. Hybrid Priority+FCFS — 生产推荐:
   → priority→分级 → FCFS→同级排序 → 平衡公平+SLA!

调度决策树:
  → 单用户 → FCFS → 简单!
  → → → 多用户+SLA分级 → Priority+FCFS → 生产最优!
  → → → → → GPU不足 → Preemptive+LRU → eviction管理!
  → → → → → → → RTX 4090: FCFS+preemption(recompute)+chunk=512 → 推荐!
```

## 4. Prefill Scheduling — 预填充调度

```
Prefill=推理第一步 → TTFT直接相关 → 调度关键!

Prefill调度挑战:
  → prefill计算量大 → O(N^1.5) → 占GPU大量时间!
  → → → S=2048 → 124.7 TFLOPS → 73.5% peak → compute-bound!
  → → → → → 占GPU→decode→等→ITL↑ → 影响在运行请求!

Chunked Prefill (vLLM V1):
  → 将长prefill → chunk → 多步完成 → 不阻塞decode!
  → → → chunk=512 → 每步512 tokens → ITL↑32% → 可接受!
  → → → → → chunk=2048 → 每步2048 → ITL↑326% → 不可接受!

Prefill隔离策略:
  → 策略1: 优先decode → prefill用剩余budget → RTX 4090推荐!
  → → → 策略3: PD分离 → prefill→专用GPU → decode→专用GPU → 最优但需多GPU!

RTX 4090最优prefill调度:
  → 优先decode → chunked prefill chunk=512-1024 → ITL可控!
  → → → ≤2 concurrent prefill → 不太多 → decode优先!
```

## 5. Decode Scheduling — 解码调度

```
Decode=推理主要阶段 → ITL直接相关 → 调度核心!

Batch size vs ITL vs 吞吐:
  → B↑ → 吞吐↑(线性) → ITL↑(weight sharing → 每请求分得少)!
  → → → B=32 → 2,468 tok/s → ITL≈12.9ms → 可接受!
  → → → → → B=55 → 4,190 tok/s → ITL≈13ms → 仍可接受!
  → → → → → → → B=118 → 4,791 tok/s → ITL≈24.4ms → SLO边缘!

Decode eviction:
  → KV cache满 → 新请求无slot → eviction → 释放KV!
  → → → vLLM: FreeKVCacheBlockQueue(LRU) → 最早未使用→evict!
  → → → → → evict→recompute(默认) → rejoin→重新prefill → 延迟↑!

StreamingLLM decode:
  → 固定KV=sink(4)+window(W) → 不增长 → 无eviction → 最简单!
  → → → W=4096 → 固定168MB → 不变 → 无内存压力 → 无OOM!
  → → → → → B=57 → 2,311 tok/s → 无限对话 → RTX 4090最优!

RTX 4090最优decode调度:
  → B=32-55 → ITL<15ms → SLO达标 → 推荐日常!
  → → → B=118 → 吞吐最高 → ITL≈24ms → 可接受 → 高负载时!
  → → → → → StreamingLLM → 固定KV → 无eviction → 无限对话 → 最优!
```

## 6. KV Cache Pressure Management — KV缓存压力管理

```
KV压力=调度最大挑战!

KV压力来源:
  → 并发请求 → 每请求KV cache → 总KV → 超GPU → OOM!
  → → → 7B GQA-5 BF16: KV/tok=81.92KB → S=2048 → 163.8MB/请求 → 并发=2→灾难!

KV压力缓解策略:

1. 量化KV → INT8 KV → 50%省 → 并发×2!
2. GQA → KV/tok↓ → 并发↑! GQA-8→FlashInfer支持!
3. Prefix Sharing → 共享KV → 84%省 → 并发↑5x!
4. Sliding Window → KV不随S增长 → 固定预算!
5. Eviction → LRU释放 → recompute(17ms)/swap(381us)!

KV压力数学:
  → 并发 = (GPU_memory - model_size) / (KV_per_tok × S)
  → → → RTX 4090 INT4+INT8KV+GQA-8: 并发=118 → 最高!

RTX 4090最优KV管理:
  → INT4+INT8KV+GQA-8+FlashInfer → B=118 → 最高并发!
  → → → Prefix sharing → RAG/Agent天然适合 → 省84%KV!
  → → → → → StreamingLLM → 固定168MB → 无eviction → 无限对话!
```

## 7. SLO-Aware Scheduling — SLO驱动调度

```
SLO=调度最终约束 → SLO违规=生产事故!

4核心SLI:
  → TTFT < 500ms P99 → prefill调度!
  → → → ITL < 100ms P99 → decode调度!
  → → → → → throughput > 1K tok/s → batch管理!
  → → → → → → → availability 99.9% → 稳定性!

SLO-Aware调度策略:

1. ITL-first (推荐):
   → decode优先 → 保证ITL → prefill用剩余 → TTFT可等!

2. Balanced (生产推荐):
   → ITL-first + throughput limit → B≤55 → ITL≤15ms → 4,190 tok/s!

3. SLO-violation detection:
   → ITL P99 > 100ms → alert → 减少batch → 降吞吐保SLO!

Error Budget管理:
  → 99.9% availability → error budget=43.2min/月
  → → → budget耗尽→降级→减少并发→保证SLO→但吞吐↓!

RTX 4090 SLO推荐:
  → 日常: B≤55 → ITL≤15ms → SLO舒适 → 4,190 tok/s!
  → → → 高负载: B≤118 → ITL≤25ms → SLO边缘 → 4,791 tok/s!
  → → → → → StreamingLLM → availability 99.99% → 最稳!
```

## 8. Core Laws — 批处理调度核心定律

1. **Batch-Throughput Law**: B↑→吞吐↑(线性)→60x! 但B↑→ITL↑→SLO约束→最优B=32-118
2. **Decode-First Law**: decode优先→ITL保证→prefill用剩余→SLO驱动→RTX 4090推荐!
3. **Chunk-Prefill Law**: chunk=512→ITL↑32%可接受 / chunk=2048→ITL↑326%不可!
4. **KV-Pressure Law**: 量化KV→并发↑6x→INT4+INT8KV=必需!
5. **Long-Context-Zero-Sum Law**: S↑→并发↓→线性反比→StreamingLLM打破!
6. **SLO-Balance Law**: B≤55→ITL≤15ms→4,190 tok/s→全部SLO合规→生产最优!
7. **Prefix-Sharing-Batching Law**: RAG/Agent→84%KV省→并发↑5x→天然适合!

## 关键参考

- vLLM V1: unified token budget → decode优先+prefill填充 → FCFS
- SGLang: PrefillAdder → conservative → ITL更稳定
- Chunked prefill: chunk=512 → ITL↑32% → RTX 4090推荐
- StreamingLLM: 固定168MB → sink+window → 无限对话
- RTX 4090 INT4+INT8KV: B=118 → 4,791 tok/s → 最高吞吐
- KV eviction: recompute(17ms) vs swap(381us) → vLLM默认recompute