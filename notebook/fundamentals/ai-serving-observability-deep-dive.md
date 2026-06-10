# AI Serving Observability & Operations Deep Dive

> 2026-06-10 | 可观测性=AI Infra的眼睛! 从metrics→logging→tracing→SLO→autoscaling→incident response, 6层运维体系
> 关联: scheduler-architecture-deep-dive.md, flashinfer-attention-deep-dive.md, continuous-batching-simulator.md

## 0. 核心定律: 可观测性 = 在生产中理解系统行为

```
可观测性三大支柱:
  → Metrics: 定量指标 → Prometheus → 时间序列 → 全局视角!
  → Logging: 定性描述 → 结构化日志 → 事件记录 → 局部详情!
  → Tracing: 请求追踪 → 分布式trace → 完整路径 → 调试利器!

与AI Infra联系:
  → vLLM生产部署 → 必须监控GPU利用率/KV cache命中率/延迟/吞吐!
  → → → 无监控=盲飞 → GPU空闲浪费钱 → 延迟超标影响用户体验!
  → → → → → 可观测性=生产必备 → 开发→生产的关键差距!

vLLM V1 Metrics系统:
  → LoggingStatLogger: 日志输出 → interval统计 → throughput+latency+cache
  → → → PrometheusStatLogger: Prometheus集成 → Counter/Gauge/Histogram → Grafana可视化
  → → → → → StatLoggerBase: 插件架构 → 用户自定义logger → 可扩展!
  → → → → → → → Pipeline: scheduler_stats→record()→_update_stats()→log() → 定期输出!

  核心metrics:
    → prompt_throughput: tokens/s → prefill速度 → TTFT相关!
    → → generation_throughput: tokens/s → decode速度 → ITL相关!
    → → → num_requests_running: 当前并发 → 调度器状态 → 资源利用率!
    → → → → gpu_cache_usage_perc: KV cache占用率 → 内存瓶颈 → OOM预警!
    → → → → → prefix_cache_hit_rate: prefix sharing命中率 → 缓存效率 → 节省KV!
    → → → → → → num_preemptions: 请求抢占 → 资源紧张 → 用户体验下降!
    → → → → → → → KV connector stats: 传输延迟+吞吐+成功数 → PD分离健康度!

RTX 4090可观测性要点:
  → 单GPU推理 → gpu_cache_usage_perc=关键 → >80%→OOM风险!
  → → → 7B INT4+INT8KV → 4,791 tok/s → generation_throughput监控!
  → → → → → FlashInfer 1.06-3.20x加速 → 监控attention占比 → 调优依据!
  → → → → → → 无分布式 → 不需要trace → 单进程可观测性足够!
```

## 1. Metrics系统 — Prometheus + vLLM

```
vLLM V1 Prometheus metrics分类:

Schedule相关:
  → vllm:num_requests_running(Gauge) → 当前运行请求数 → 负载指标!
  → → vllm:num_requests_waiting(Gauge) → 等待请求数 → backlog指标!
  → → → vllm:num_requests_finished(Counter) → 完成请求数 → 吞吐指标!
  → → → → vllm:num_preemptions(Counter) → 抢占数 → 资源紧张信号!
  → → → → → vllm:gpu_cache_usage_perc(Gauge) → KV cache占用% → 内存指标!

Throughput相关:
  → vllm:prompt_tokens_total(Counter) → prefill tokens → 输入吞吐!
  → → vllm:generation_tokens_total(Counter) → decode tokens → 输出吞吐!
  → → → vllm:iteration_tokens_total(Counter) → 每iteration总tokens → 调度器效率!

Latency相关:
  → vllm:e2e_request_latency_seconds(Histogram) → 请求总延迟 → P50/P90/P99!
  → → vllm:request_prefill_latency_seconds(Histogram) → TTFT → 首token延迟!
  → → → vllm:request_decode_latency_seconds(Histogram) → decode延迟 → ITL!

KV Cache相关:
  → vllm:prefix_cache_hit_rate(Gauge) → prefix共享命中率 → 缓存效率!
  → → vllm:cpu_cache_usage_perc(Gauge) → CPU swap占用 → swap监控!
  → → → KV connector metrics(新增PR#45148) → 传输延迟+吞吐+成功数!

Speculative Decoding:
  → vllm:spec_decode_num_drafts(Counter) → draft token数 → 投机活跃度!
  → → vllm:spec_decode_num_accepted(Counter) → 接受token数 → 接受率!

GPU相关:
  → vllm:gpu_utilization_perc(Gauge) → GPU利用率 → 计算资源监控!
  → → DCGM metrics → 温度+功耗+ECC错误 → 硬件健康!

生产阈值建议:
  → gpu_cache_usage > 80% → 预警 → 可能OOM!
  → → e2e_request_latency P99 > 5s → 预警 → 用户体验差!
  → → → prompt_throughput < 100 tok/s → 低吞吐 → 可能GPU空闲!
  → → → → num_preemptions > 10/min → 预警 → 资源不足→需要扩容!
  → → → → → prefix_cache_hit_rate < 20% → 缓存效率低 → 可能需要优化prefix!

Prometheus架构:
  → vLLM → /metrics endpoint → Prometheus pull → Grafana dashboard → Alert!
  → → → Prometheus server → scrape_interval=5s → 推理足够!
  → → → → → Grafana → 实时可视化 → dashboard模板 → 分享!
  → → → → → → → Alertmanager → 规则→通知 → PagerDuty/Slack → 响应!
```

## 2. Logging — 结构化日志

```
vLLM日志系统:
  → vllm.logger(init_logger) → Python logging → 标准库!
  → → → 日志级别: DEBUG/INFO/WARNING/ERROR/CRITICAL → 分级!
  → → → → → 生产=INFO级 → 调试=DEBUG级 → 关键事件=WARNING!

关键日志事件:
  → Engine初始化: "Using model..." → 模型+设备+配置 → 启动确认!
  → → KV cache分配: "Allocating GPU KV cache..." → 内存大小 → 资源确认!
  → → → 延迟统计: "Avg prompt throughput..." → 定期输出 → 性能监控!
  → → → → 抢占事件: "Preempting..." → 请求抢占 → 资源紧张信号!
  → → → → → KV transfer: "KV Transfer metrics..." → connector stats → PD分离健康!

结构化日志最佳实践:
  → JSON格式 → 结构化 → 机器可读 → 便于搜索!
  → → → 关键字段: timestamp/request_id/model/latency/tokens → 标准化!
  → → → → → 关联metrics → 同一request_id → trace+metric+log → 三支柱关联!

vLLM当前问题:
  → 日志不是JSON → 文本格式 → 不便于机器解析 → 生产改进空间!
  → → → request_id → 请求级追踪 → 但不跨分布式 → 需要tracing!
  → → → → → 无OpenTelemetry集成 → vLLM缺少分布式tracing → 未来需求!
```

## 3. Distributed Tracing — 请求级追踪

```
分布式追踪(OpenTelemetry):
  → Trace: 一次请求的完整路径 → 多个Span → 服务→服务→服务!
  → → → Span: 单个操作 → 名称+开始时间+持续时间+属性!
  → → → → → Context propagation: trace_id跨服务传递 → W3C TraceContext!

AI Serving tracing场景:
  → 用户请求 → API gateway → vLLM scheduler → prefill → decode → response!
  → → → 每步是Span → trace_id串联 → 完整路径 → 调试慢请求!

  → PD分离: 用户请求 → prefill GPU → KV transfer → decode GPU → response!
  → → → → → 3个关键Span: prefill/KV_transfer/decode → 定位瓶颈!

  → verl RL: rollout request → vLLM inference → reward → PPO update → model save!
  → → → → → → → 4个Span: inference/reward/PPO_update/save → 定位慢步骤!

OpenTelemetry集成(未来):
  → vLLM → opentelemetry-sdk → Span创建 → trace_id → 分布式!
  → → → HTTP header → traceparent → 传递 → 服务链 → 完整trace!
  → → → → → → → Jaeger/Zipkin → trace可视化 → UI → 时间线 → 定位!

当前状态:
  → vLLM无OpenTelemetry → 需要手动追踪 → request_id是唯一线索!
  → → → Ray Dashboard → actor级监控 → 但不是request级trace!
  → → → → → 未来: vLLM+OTel → 生产可观测性 → 关键改进!
```

## 4. SLO/SLA管理 — AI Serving服务质量

```
AI Serving SLI(Service Level Indicator)定义:

  → TTFT(Time To First Token):
    → → 定义: 从请求发送到首token返回的时间 → prefill+调度延迟!
    → → → SLO: P99 TTFT < 500ms(对话)/< 2s(批量) → 用户感知阈值!
    → → → → → 计算: histogram → P50/P90/P99 → vllm:request_prefill_latency_seconds!

  → ITL(Inter-Token Latency):
    → → 定义: 连续两个token间的延迟 → decode速度 → 流式体验!
    → → → SLO: P99 ITL < 100ms(对话)/P50 ITL < 50ms → 流式阈值!
    → → → → → 计算: request_decode_latency / num_tokens → 平均ITL!

  → Throughput:
    → → 定义: tokens/s → 总吞吐 → 资源利用率指标!
    → → → SLO: generation_throughput > 1000 tok/s → 资源利用阈值!
    → → → → → 计算: generation_tokens_total / time → Prometheus!

  → Availability:
    → → 定义: 成功请求比例 → 服务稳定性 → 非OOM/非超时!
    → → → SLO: 99.9% availability → 876min/year downtime → 严格!
    → → → → → 计算: success_requests / total_requests → Prometheus!

  → KV Cache Hit Rate(Prefix Sharing):
    → → 定义: prefix匹配比例 → 缓存效率 → 节省计算!
    → → → SLO: hit_rate > 30%(RAG场景)/> 50%(Agent场景) → 缓存阈值!
    → → → → → 计算: prefix_cache_hits / prefix_cache_queries → Prometheus!

SLI → SLO → SLA映射:
  → SLI: 具体测量值 → 如P99 TTFT=320ms
  → → → SLO: 目标值 → 如P99 TTFT<500ms → 320ms<500ms → 达标!
  → → → → → SLA: 商业承诺 → 如99.9%availability → 否则赔偿!
  → → → → → → → Error Budget: SLI - SLO → 如availability 99.95% → budget=0.05%!

RTX 4090 SLO估算:
  → TTFT: 7B INT4 S=512 → TTFT≈100ms → P99<200ms → SLO达标!
  → → ITL: 7B INT4 B=55 → 4,190 tok/s → ITL≈0.24ms → P99<10ms → SLO达标!
  → → → Throughput: 4,190 tok/s → >1,000 → SLO达标!
  → → → → → Availability: GPU故障→手动恢复 → 99.9%需要冗余 → RTX 4090不能保证!
  → → → → → → → → → RTX 4090最优SLO: TTFT<200ms, ITL<10ms, Throughput>1K → 但availability需冗余!
```

## 5. Autoscaling — 自动扩缩容

```
Kubernetes Autoscaling策略:

HPA(Horizontal Pod Autoscaler):
  → 基于metrics扩缩 → CPU/GPU利用率 → 自定义metrics!
  → → → vLLM: generation_throughput → 如果<阈值 → 增加pod!
  → → → → → Prometheus Adapter → custom metrics → HPA → 自动!
  → → → → → → → minReplicas=1 → maxReplicas=10 → 冷启动问题!

VPA(Vertical Pod Autoscaler):
  → 基于资源调优 → 内存/GPU → 调整pod资源请求!
  → → → 不适合AI serving → GPU资源不可动态调整 → 固定!

Custom HPA for AI Serving:
  → 指标选择:
    → → 并发请求数(num_requests_running) → 最直接 → 请求→GPU映射!
    → → → KV cache利用率(gpu_cache_usage_perc) → 内存指标 → >80%→扩容!
    → → → → → 延迟(TTFT P99) → 用户体验 → >阈值→扩容!
    → → → → → → → 吞吐(generation_throughput) → 资源利用率 → <阈值→扩容!

  → 冷启动问题:
    → → 新pod → 模型加载+编译 → 7B INT4 ≈ 30s → 冷启动!
    → → → → → 解决: Pre-warm → 保持1个空闲pod → 请求来→立刻服务!
    → → → → → → → → → vLLM sleep/wake → 模型常驻GPU → sleep模式→零计算→wake→秒级恢复!
    → → → → → → → → → → → verl用 → colocation → 2模型共享GPU → sleep/wake!

  → 缩容策略:
    → → pod空闲→延迟缩容 → 冷启动太贵 → 保持5-10min空闲 → 突发流量缓冲!
    → → → → → vLLM sleep模式 → GPU闲置→sleep→0功耗 → wake→秒级 → 缩容替代!

RTX 4090 autoscaling:
  → 单GPU推理 → 不需要autoscaling → 固定pod → 单GPU足够!
  → → → 如果需要更多吞吐 → 增加GPU pod → 独立推理 → 不需要分布式!
  → → → → → RTX 4090最优=固定1pod+7B INT4 → 简单 → 不需要autoscaling复杂度!
```

## 6. Production Debugging — 常见问题诊断

```
常见生产问题+诊断:

1. OOM(Out of Memory):
   → 原因: KV cache超限 → 并发太高 → gpu_cache_usage > 100%!
   → → → 诊断: gpu_cache_usage_perc → 持续>90% → OOM前兆!
   → → → → → 解决: 减少并发(INT4量化+INT8 KV) → 增加GPU → streaming mode!

2. 高延迟(Slow Inference):
   → 原因: GPU利用率低 → memory-bound → prefill阻塞decode!
   → → → 诊断: e2e_request_latency P99 → 找P99 → 定位瓶颈!
   → → → → → → TTFT高 → prefill慢 → 增加GPU/chunked prefill!
   → → → → → → → ITL高 → decode慢 → 检查batch size/量化!
   → → → → → → → → → 混合: prefill阻塞decode → PD分离/chunked prefill!

3. 低吞吐(Low Throughput):
   → 原因: batch太小 → 并发低 → GPU未充分利用!
   → → → 诊断: generation_throughput → <100 tok/s → 低利用率!
   → → → → → 解决: 增加并发(max_num_seqs) → continuous batching → 填充GPU!

4. Request Drops:
   → 原因: 抢占(preemption) → 资源紧张 → 请求被踢出!
   → → → 诊断: num_preemptions → >0 → 抢占发生 → 为什么?
   → → → → → 解决: 增加KV cache → 减少并发 → 更好调度策略!

5. GPU故障:
   → 原因: ECC错误 → 过热 → 硬件故障 → DCGM检测!
   → → → 诊断: DCGM → ECC errors/XID errors → 硬件级!
   → → → → → 解决: 替换GPU → 重启服务 → checkpoint恢复!

vLLM调试流程:
  → 1. 查metrics → Prometheus → 定位问题类型(延迟/吞吐/OOM)!
  → → 2. 查日志 → vllm.log → 找具体错误信息!
  → → → 3. 查trace → request_id → 请求级路径 → 定位慢步骤!
  → → → → 4. GPU profiling → Nsight Systems → kernel级 → 定位最慢kernel!
  → → → → → 5. 修复 → 配置调整/代码修改/硬件替换 → 解决!

RTX 4090调试要点:
  → 单GPU → 无分布式 → 调试简单 → 本地足够!
  → → → 内存限制 → 24GB → OOM是首要关注 → 监控gpu_cache_usage!
  → → → → → 温度 → RTX 4090消费级 → 过热降频 → 监控DCGM温度!
```

## 7. Core Laws — 可观测性核心定律

```
1. Observability Law: Metrics+Logging+Tracing → 三支柱 → 不可缺一!
   → → → Metrics=全局 → Logging=局部 → Tracing=请求 → 三维视角!

2. SLO-Driven Law: 生产监控=SLO驱动 → 不是CPU利用率 → 是用户体验!
   → → → TTFT+ITL+Throughput+Availability → 4个核心SLI → 用户感知!
   → → → → → vLLM已有3个(TTFT/ITL/Throughput) → Availability需冗余!

3. Cache-Utilization Law: gpu_cache_usage → 内存瓶颈 → AI serving核心指标!
   → → → >80%→预警 → >90%→OOM风险 → 生产必须监控!

4. Cold-Start Law: autoscaling冷启动≈30s → 预热/sleep/wake → 避免冷启动!
   → → → → → vLLM sleep模式→秒级恢复 → 比冷启动快30x!

5. Debugging-Hierarchy Law: metrics→log→trace→profile → 逐层深入!
   → → → → → → → 先全局(metrics)→局部(log)→请求(trace)→kernel(profile)→逐步定位!

6. Error-Budget Law: availability=99.9% → error budget=0.1% → 允许的故障时间!
   → → → → → budget内→继续创新 → budget外→停止→专注稳定性 → 平衡!

7. Prefix-Sharing-Visibility Law: prefix_cache_hit_rate → 缓存效率 → RAG/Agent关键!
   → → → → → 低命中率 → 缓存未利用 → 需优化prefix → RAG天然适合prefix sharing!
```

## 关键参考

```
- vLLM V1 metrics: vllm/v1/metrics/loggers.py → LoggingStatLogger + PrometheusStatLogger
- vLLM observability_config: vllm/config/observability.py → enable_mfu_metrics/cudagraph_metrics
- Prometheus: prometheus_client Python → Counter/Gauge/Histogram → /metrics endpoint
- Grafana: dashboard模板 → vLLM社区 → 实时可视化
- OpenTelemetry: distributed tracing → 未来vLLM集成 → 请求级追踪
- DCGM: NVIDIA GPU monitoring → ECC errors/temperature/utilization → 硬件健康
- Nsight Systems: GPU profiling → kernel级 → 生产调试深度工具
- SRE Book (Google): SLO/SLA/SLI → error budget → incident response → 生产标准
- vLLM sleep/wake: 冬眠模式 → 秒级恢复 → autoscaling替代 → verl用!
- Kubernetes HPA: custom metrics → vLLM throughput → 自动扩缩容