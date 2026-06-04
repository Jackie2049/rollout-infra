# vLLM V1 可观测性与指标体系 源码阅读

> vLLM V1 的 metrics/stats/logging 架构，生产监控基础

## 1. 指标体系概览

```
┌─────────────────────────────────────────────────────┐
│                    vLLM V1 Metrics                    │
├───────────────┬──────────────┬───────────────────────┤
│ Stats 采集层   │ Logger 层    │ Export 层             │
│ (stats.py)    │ (loggers.py) │ (prometheus.py)       │
├───────────────┼──────────────┼───────────────────────┤
│ SchedulerStats│ LoggingStat  │ Prometheus /metrics   │
│ IterationStats│ Prometheus   │ Grafana Dashboard     │
│ RequestStats  │ Aggregated   │ Alert Rules           │
│ CacheStats    │ Plugins      │                       │
│ PerfStats     │              │                       │
│ SpecDecode    │              │                       │
│ KVConnector   │              │                       │
└───────────────┴──────────────┴───────────────────────┘
```

## 2. Stats 采集层 (`vllm/v1/metrics/stats.py`)

### 2.1 SchedulerStats — 调度器级统计

```python
@dataclass
class SchedulerStats:
    num_running_reqs: int = 0          # 正在运行的请求数
    num_waiting_reqs: int = 0          # 等待队列长度
    num_skipped_waiting_reqs: int = 0  # 跳过等待的请求数
    kv_cache_usage: float = 0.0        # KV Cache 使用率 (0-1)

    # Prefix Caching
    prefix_cache_stats: PrefixCacheStats
    connector_prefix_cache_stats: PrefixCacheStats | None  # 外部 KV

    # 高级特性
    spec_decoding_stats: SpecDecodingStats | None  # 投机解码
    kv_connector_stats: dict[str, Any] | None      # KV 连接器
    cudagraph_stats: CUDAGraphStat | None           # CUDA Graph
    perf_stats: PerfStats | None                    # 性能 (MFU)

    # LoRA
    waiting_lora_adapters: dict[str, int]  # 等待中的 LoRA 请求数
    running_lora_adapters: dict[str, int]  # 运行中的 LoRA 请求数
```

### 2.2 IterationStats — 迭代级统计

```python
class IterationStats:
    iteration_timestamp: float              # 迭代时间戳
    num_generation_tokens: int              # 本迭代生成 token 数
    prompt_token_stats: PromptTokenStats    # Prompt token 来源统计
    num_preempted_reqs: int                 # 抢占数
    finished_requests: list[FinishedRequestStats]  # 完成的请求
    time_to_first_tokens_iter: list[float]  # TTFT 列表
    inter_token_latencies_iter: list[float] # ITL 列表
    num_corrupted_reqs: int                 # NaN 请求数
```

### 2.3 PromptTokenStats — Token 来源统计

```python
@dataclass
class PromptTokenStats:
    ALL_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
    computed: int = 0              # 本地计算 (无缓存)
    local_cache_hit: int = 0       # Prefix Cache 命中
    external_kv_transfer: int = 0  # 外部 KV Transfer
    cached_tokens: int = 0         # 总缓存 token
    total: int = 0                 # 总 prompt token
```

不变量: `computed + local_cache_hit + external_kv_transfer = total`

### 2.4 FinishedRequestStats — 完成请求统计

```python
@dataclass
class FinishedRequestStats:
    finish_reason: FinishReason
    e2e_latency: float = 0.0             # 端到端延迟
    num_prompt_tokens: int = 0           # Prompt token 数
    num_generation_tokens: int = 0       # 生成 token 数
    queued_time: float = 0.0             # 排队时间
    prefill_time: float = 0.0            # Prefill 时间
    inference_time: float = 0.0          # 推理时间
    decode_time: float = 0.0             # Decode 时间
    mean_time_per_output_token: float    # 平均每 token 时间
    num_cached_tokens: int = 0           # 缓存 token 数
    is_corrupted: bool = False           # 是否 NaN
```

**时间分解**:
```
arrival_time                    → E2E Latency
    │
    ├── queued_ts ─── scheduled_ts → Queued Time
    │                   │
    │                   ├── first_token_ts → Prefill Time
    │                   │       │
    │                   │       └── last_token_ts → Decode Time
    │                   │
    │                   └── last_token_ts → Inference Time
```

### 2.5 CachingMetrics — 滑动窗口命中率

```python
class CachingMetrics:
    """最近 N 个请求的缓存命中率 (默认 N=1000)"""
    max_recent_requests: int = 1000
    query_queue: deque[tuple[int, int, int]]  # (requests, queries, hits)

    @property
    def hit_rate(self) -> float:
        return self.aggregated_query_hit / self.aggregated_query_total
```

三种缓存追踪:
- `prefix_caching_metrics`: 本地 Prefix Cache
- `connector_prefix_caching_metrics`: 外部 KV Transfer
- `mm_caching_metrics`: 多模态缓存

## 3. Logger 层 (`vllm/v1/metrics/loggers.py`)

### 3.1 Logger 层次结构

```
StatLoggerBase (ABC)
├── LoggingStatLogger           — stdout 文本日志 (per engine)
│   └── AggregatedLoggingStatLogger — 跨 DP engine 聚合
└── PrometheusStatLogger        — Prometheus /metrics endpoint
    └── AggregateStatLoggerBase — 跨 engine 聚合
        └── PerEngineStatLoggerAdapter — per-engine 包装
```

### 3.2 日志输出格式

```
Engine 000: Avg prompt throughput: 1234.5 tokens/s,
            Avg generation throughput: 567.8 tokens/s,
            Running: 42 reqs, Waiting: 10 reqs,
            GPU KV cache usage: 67.3%,
            Prefix cache hit rate: 45.2%,
            External prefix cache hit rate: 12.1%
```

空闲时降级为 `debug` 级别，避免日志噪音。

### 3.3 AggregatedLoggingStatLogger (DP 场景)

```python
class AggregatedLoggingStatLogger(LoggingStatLogger):
    """跨多个 DP engine 的聚合日志"""
    engine_indexes: list[int]

    def aggregate_scheduler_stats(self):
        # 累加 running/waiting/skipped
        # KV cache usage 取平均
        self.last_scheduler_stats.kv_cache_usage /= len(...)
```

## 4. Prometheus 指标 (`vllm/v1/metrics/prometheus.py`)

### 4.1 核心 Gauge 指标

| 指标名 | 类型 | 含义 |
|--------|------|------|
| `vllm:num_requests_running` | Gauge | 运行中的请求数 |
| `vllm:num_requests_waiting` | Gauge | 等待中的请求数 |
| `vllm:num_requests_waiting_by_reason` | Gauge | 等待原因 (capacity/deferred) |
| `vllm:kv_cache_usage_perc` | Gauge | KV Cache 使用率 (%) |
| `vllm:gpu_cache_usage_perc` | Gauge | GPU Cache 使用率 |
| `vllm:engine_sleep_state` | Gauge | 引擎休眠状态 |

### 4.2 Counter 指标

| 指标名 | 含义 |
|--------|------|
| `vllm:prompt_tokens` | Prefill token 数 |
| `vllm:prompt_tokens_by_source` | Token 来源 (local_compute/cache_hit/kv_transfer) |
| `vllm:generation_tokens` | 生成 token 数 |
| `vllm:num_preemptions` | 累计抢占次数 |
| `vllm:prefix_cache_queries/hits` | Prefix Cache 查询/命中 |
| `vllm:external_prefix_cache_queries/hits` | 外部 KV 查询/命中 |
| `vllm:mm_cache_queries/hits` | 多模态缓存查询/命中 |
| `vllm:corrupted_requests` | NaN 请求数 (需启用) |

### 4.3 Histogram 指标

| 指标名 | 含义 |
|--------|------|
| `vllm:e2e_request_latency_seconds` | 端到端延迟 |
| `vllm:request_queue_time_seconds` | 排队延迟 |
| `vllm:request_inference_time_seconds` | 推理延迟 |
| `vllm:request_prefill_time_seconds` | Prefill 延迟 |
| `vllm:request_decode_time_seconds` | Decode 延迟 |
| `vllm:time_to_first_token_seconds` | TTFT |
| `vllm:time_per_output_token_seconds` | TPOT |
| `vllm:inter_token_latency_seconds` | Token 间延迟 |

### 4.4 Label 维度

所有指标带 `model_name` 和 `engine` 两个 label:
```
vllm:num_requests_running{model_name="deepseek-v3", engine="0"} 42
```

## 5. PerfStats — 性能指标 (`vllm/v1/metrics/perf.py`)

```python
class PerfStats:
    """跟踪 MFU (Model FLOPS Utilization)"""
    # 需要 --observability-config enable-mfu-metrics=true
```

MFU 计算:
```
MFU = 实际 FLOPS / 硬件峰值 FLOPS
实际 FLOPS = C × num_tokens (C = 6 × params_per_token)
```

## 6. SpecDecoding Metrics

```python
class SpecDecodingStats:
    num_drafts: int = 0          # Draft token 数
    num_accepted: int = 0        # 接受的 token 数
    acceptance_rate: float       # 接受率
```

## 7. KV Connector Metrics (`distributed/kv_transfer/kv_connector/v1/metrics.py`)

| 指标 | 含义 |
|------|------|
| KV Transfer 发送/接收字节 | 跨实例 KV 传输量 |
| KV Transfer 延迟 | 传输耗时 |
| KV Load 失败次数 | 加载失败计数 |

## 8. 插件系统

```python
# 自定义 Stat Logger 插件
class StatLoggerBase(ABC):
    @abstractmethod
    def record(self, scheduler_stats, iteration_stats, ...): ...
    @abstractmethod
    def log_engine_initialized(self): ...

# 通过 plugin 注册
STAT_LOGGER_PLUGINS_GROUP = "vllm.stat_logger_plugins"
```

## 9. 生产监控告警建议

### 9.1 关键告警规则

| 指标 | 阈值 | 级别 | 说明 |
|------|------|------|------|
| `kv_cache_usage_perc` | > 90% | Warning | KV Cache 接近满 |
| `kv_cache_usage_perc` | > 95% | Critical | 即将 OOM |
| `num_requests_waiting` | > 100 | Warning | 请求堆积 |
| `e2e_request_latency_seconds` P99 | > 5s | Warning | 延迟过高 |
| `time_to_first_token_seconds` P99 | > 2s | Warning | TTFT 过高 |
| `num_preemptions` rate | > 10/min | Warning | 频繁抢占 |
| `inter_token_latency_seconds` P99 | > 200ms | Warning | Decode 变慢 |

### 9.2 关键仪表盘面板

1. **吞吐**: Prompt throughput + Generation throughput (tokens/s)
2. **延迟**: TTFT P50/P95/P99, ITL P50/P95/P99, E2E P50/P95/P99
3. **容量**: Running/Waiting 请求数, KV Cache 使用率
4. **缓存**: Prefix Cache hit rate, External Cache hit rate
5. **Token 来源**: 本地计算 vs Cache 命中 vs KV Transfer
6. **高级**: Spec Decoding 接受率, MFU, LoRA 请求数

## 10. 关键洞察

1. **Stats 采集在 EngineCore**: 每次 step 更新 IterationStats，请求完成时更新 FinishedRequestStats
2. **Logging 异步**: Stats 由 EngineCore 进程采集，通过 output_queue 传给前端 Logger
3. **滑动窗口缓存**: Prefix Cache 命中率基于最近 1000 个请求，避免老数据干扰
4. **Token 来源三分类**: local_compute / local_cache_hit / external_kv_transfer
5. **DP 聚合**: AggregatedLoggingStatLogger 跨 engine 累加 running/waiting，KV usage 取平均
6. **空闲时静默**: 无流量时日志降级为 debug，减少生产噪音
7. **Prometheus 多进程**: 使用 `multiprocess_mode="mostrecent"` 支持 DP 多进程
8. **LoRA 追踪**: per-LoRA 的 waiting/running 请求数，用于多 LoRA 监控
9. **NaN 检测**: 可选检测 logits 中的 NaN，标记 corrupted 请求
10. **MFU 可选**: 需要 `--observability-config enable-mfu-metrics=true` 开启

## 参考资料

- `vllm/v1/metrics/stats.py` — Stats 数据结构定义
- `vllm/v1/metrics/loggers.py` — Logger 层 (Logging/Prometheus/Aggregated)
- `vllm/v1/metrics/prometheus.py` — Prometheus 指标注册
- `vllm/v1/metrics/perf.py` — MFU 性能指标
- `vllm/v1/metrics/reader.py` — Metrics 读取器
- `vllm/v1/spec_decode/metrics.py` — Spec Decoding 指标
- `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py` — KV Connector 指标
- 相关: [V1 Architecture Map](vllm-v1-architecture-map.md)
