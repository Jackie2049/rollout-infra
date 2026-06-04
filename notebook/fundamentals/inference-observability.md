# GPU 推理服务监控与可观测性

> 目标：掌握生产环境 LLM 推理服务的监控、告警、追踪体系

## 1. 监控架构总览

```
┌────────────────────────────────────────────────────────────────┐
│                     Grafana Dashboard                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Latency  │ │Throughput│ │KV Cache  │ │GPU Hardware      │ │
│  │ P50/P99  │ │ tok/sec  │ │ Usage %  │ │ Util/Temp/Power  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │
│       ▲            ▲            ▲               ▲              │
│       └────────────┴────────────┴───────────────┘              │
│                           Prometheus                            │
│                    (scrape every 5s)                            │
│                    ┌──────┴──────┐                              │
│                    │             │                              │
│              vLLM :8000   DCGM Exporter :9400                  │
│              (app metrics)    (GPU hardware)                    │
└────────────────────────────────────────────────────────────────┘
```

## 2. vLLM 核心指标

### 延迟指标 (Histogram)

| 指标名 | 含义 | SLO 目标 |
|--------|------|---------|
| `vllm:time_to_first_token_seconds` | TTFT | P50<200ms, P99<2s |
| `vllm:time_per_output_token_seconds` | TPOT (inter-token) | P50<30ms, P95<100ms |
| `vllm:e2e_request_latency_seconds` | 端到端延迟 | P99<30s |
| `vllm:request_queue_time_seconds` | 排队等待时间 | P99<1s |
| `vllm:request_prefill_time_seconds` | Prefill 阶段时间 | — |
| `vllm:request_decode_time_seconds` | Decode 阶段时间 | — |

### 调度器状态 (Gauge)

| 指标名 | 含义 |
|--------|------|
| `vllm:num_requests_running` | 当前运行的请求数 |
| `vllm:num_requests_waiting` | 排队等待的请求数 |
| `vllm:num_requests_swapped` | 被换出到 CPU 的请求数 |

### 缓存利用率 (Gauge)

| 指标名 | 含义 | 告警阈值 |
|--------|------|---------|
| `vllm:gpu_cache_usage_perc` | GPU KV Cache 使用率 | >90% 持续 2min |
| `vllm:cpu_cache_usage_perc` | CPU KV Cache 使用率 | >80% |
| `vllm:gpu_prefix_cache_hit_rate` | 前缀缓存命中率 | — |

### 吞吐量 (Counter)

| 指标名 | 含义 |
|--------|------|
| `vllm:prompt_tokens_total` | 累计 prefill tokens |
| `vllm:generation_tokens_total` | 累计 generation tokens |
| `vllm:request_success_total` | 成功请求数 (by finish_reason) |

### 推测解码

| 指标名 | 含义 |
|--------|------|
| `vllm:spec_decode_draft_acceptance_rate` | Draft token 接受率 |
| `vllm:spec_decode_efficiency` | 整体 spec decode 效率 |

### 抢占与驱逐

| 指标名 | 含义 |
|--------|------|
| `vllm:num_preemptions_total` | 累计抢占次数 |
| `vllm:iteration_tokens_total` | 每步处理 token 数分布 |

## 3. 关键 PromQL 查询

```promql
# P99 TTFT
histogram_quantile(0.99, sum by(le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))

# P95 TPOT
histogram_quantile(0.95, sum by(le) (rate(vllm:time_per_output_token_seconds_bucket[5m])))

# Token 吞吐量 (tokens/sec)
rate(vllm:generation_tokens_total[5m])

# 平均 E2E 延迟
rate(vllm:e2e_request_latency_seconds_sum[5m])
/
rate(vllm:e2e_request_latency_seconds_count[5m])

# 抢占率
rate(vllm:num_preemptions_total[5m])

# 排队深度
vllm:num_requests_waiting
```

## 4. SLO 定义

| SLO | 目标 | 告警条件 |
|-----|------|---------|
| P50 TTFT | < 200ms | — |
| P95 TTFT | < 1s | — |
| P99 TTFT | < 2s | 持续 5min 触发 critical |
| P50 TPOT | < 30ms | — |
| P95 TPOT | < 100ms | 持续 5min 触发 critical |
| P99 E2E | < 30s | — |
| Error Rate | < 0.1% | — |
| KV Cache | < 80% 持续 | >90% 持续 2min 触发 warning |

## 5. 告警规则

```yaml
groups:
  - name: vllm_inference
    rules:
      # KV Cache 接近耗尽 → 将导致抢占
      - alert: VLLMHighGPUKVCacheUsage
        expr: vllm:gpu_cache_usage_perc > 0.9
        for: 2m
        labels:
          severity: warning

      # 队列堆积 → 吞吐量不足
      - alert: VLLMHighQueueDepth
        expr: vllm:num_requests_waiting > 50
        for: 1m
        labels:
          severity: warning

      # P99 TTFT 违反 SLO
      - alert: VLLMHighTTFT
        expr: >
          histogram_quantile(0.99,
            sum by(le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
          ) > 2.0
        for: 5m
        labels:
          severity: critical

      # P95 TPOT 违反 SLO
      - alert: VLLMHighTPOT
        expr: >
          histogram_quantile(0.95,
            sum by(le) (rate(vllm:time_per_output_token_seconds_bucket[5m]))
          ) > 0.1
        for: 5m
        labels:
          severity: critical

      # 抢占风暴
      - alert: VLLMPreemptionRate
        expr: rate(vllm:num_preemptions_total[5m]) > 10
        for: 2m
        labels:
          severity: warning

      # Swap 发生 → GPU 内存不足
      - alert: VLLMSwappedRequests
        expr: vllm:num_requests_swapped > 0
        for: 1m
        labels:
          severity: warning

      # 零吞吐 → 服务可能宕机
      - alert: VLLMZeroThroughput
        expr: rate(vllm:tokens_total[5m]) == 0
        for: 5m
        labels:
          severity: critical
```

## 6. 瓶颈定位

通过指标组合快速定位问题：

| 症状 | 组合查询 | 诊断 | 解决方案 |
|------|---------|------|---------|
| 高 TTFT | queue_time 低, prefill_time 高 | Prefill 是瓶颈 | 减小 max_model_len 或增加 GPU |
| 高 TTFT | queue_time 高 | 调度瓶颈 | 增加 replicas 或降低并发 |
| 高 TPOT | gpu_cache_usage 高 | KV Cache 压力 | 增加 gpu_memory_utilization |
| 高 TPOT | preemption_rate 高 | 频繁抢占 | 减小 max_num_seqs |
| 低吞吐 | GPU_UTIL 低 | GPU 利用不足 | 增加 batch size / 并发 |
| 低吞吐 | DRAM_ACTIVE 高, SM_ACTIVE 低 | Memory-bound (正常) | Speculative Decoding |

## 7. 分布式追踪 (OpenTelemetry)

```bash
# 启动 Jaeger
docker run --rm --name jaeger \
    -p 16686:16686 -p 4317:4317 -p 4318:4318 \
    jaegertracing/all-in-one:1.57

# 启动 vLLM with tracing
export OTEL_SERVICE_NAME="vllm-server"
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
vllm serve <model> --otlp-traces-endpoint="grpc://<jaeger-ip>:4317"
```

追踪覆盖: Client span → vLLM server span → FastAPI spans → 模型执行

## 8. GPU 硬件监控 (DCGM)

### 安装

```bash
sudo apt-get install datacenter-gpu-manager
sudo systemctl enable --now nvidia-dcgm
```

### 关键 DCGM 指标

| Field ID | 指标 | 含义 |
|----------|------|------|
| 1005 | DEV_GPU_UTIL | GPU 利用率 % |
| 1006 | DEV_MEM_COPY_UTIL | 显存带宽利用率 % |
| 140 | DEV_GPU_TEMP | GPU 温度 |
| 150 | DEV_POWER_USAGE | 功耗 W |
| 252 | DEV_FB_USED | 显存已用 MB |
| 253 | DEV_FB_FREE | 显存空闲 MB |
| 1001 | PROF_SM_ACTIVE | SM 活跃度 |
| 1003 | PROF_PIPE_TENSOR_ACTIVE | Tensor Core 利用率 |
| 1004 | PROF_DRAM_ACTIVE | DRAM 利用率 |

### DCGM Exporter (Prometheus)

```bash
docker run --rm -d --gpus all -p 9400:9400 \
    nvcr.io/nvidia/k8s/dcgm-exporter:latest
```

### LLM 推理功耗优化

```bash
# 设置 LLM 推理功耗配置文件
dcgmi config --set-workload-power-profile --profile LLM_INFERENCE
```

### nvidia-smi 快速检查

```bash
# 一次性查询
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv

# 持续监控 (5秒间隔)
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > gpu_memory_log.csv
```

## 9. 关键洞察

1. **TTFT + TPOT 是核心 SLO**: 分别衡量 prefill 和 decode 性能
2. **KV Cache 使用率 >90% = 危险**: 即将触发抢占风暴
3. **排队深度是最直接的扩缩信号**: 等于 GPU 吞吐量不足
4. **DRAM 高 + SM 低 = memory-bound**: Decode 阶段正常，无法优化
5. **vLLM 暴露 30+ 指标**: 全部在 /metrics 端点，5s 抓取间隔
6. **DCGM 提供硬件级指标**: 温度、功耗、XID 错误、SM 活跃度
7. **Grafana 组合面板**: 应用指标 + 硬件指标，跨层关联分析

## 参考资料

- [vLLM Serving Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html)
- [vLLM Prometheus + Grafana Setup](https://docs.vllm.ai/en/latest/getting_started/examples/prometheus_grafana.html)
- [vLLM OpenTelemetry Integration](https://docs.vllm.ai/en/latest/getting_started/examples/opentelemetry.html)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html)
