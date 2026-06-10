# AI Request Routing & Endpoint Optimization Deep Dive

> 2026-06-10 | 请求路由=AI serving的入口守门人! 从负载均衡到SLA路由, 从API设计到circuit breaker, 从graceful degradation到健康检查, 路由=用户→GPU的桥梁!
> 关联: ai-serving-observability-deep-dive.md, ai-inference-batching-scheduling-deep-dive.md, ai-serving-cost-optimization-deep-dive.md

## 0. 核心定律: 路由 = 用户→GPU的最优桥梁

AI serving不是单GPU→单用户 → 而是 N用户→M GPU → 路由=决定谁去哪!

关键数据:
- B=55 → 4,190 tok/s → 单GPU吞吐 → 但N用户→1 GPU → 需路由!
- 99.9% SLO → error budget 43min/月 → 路由错误→SLO违规→预算消耗!
- Sleep/wake 1s → HPA 30s → 路由需感知GPU状态→sleep→不路由!
- RTX 4090 $0.03/1M tok → 但路由效率低→浪费→成本↑!

## 1. Load Balancing — 负载均衡

```
AI serving负载均衡5种策略:

1. Round Robin (最简单):
   → 请求→轮流→GPU → 均匀分配 → 简单!
   → → → 问题: 请求长度不同→GPU负载不均→长请求→GPU慢→短请求→等!

2. Least Connections (推荐):
   → 请求→最少连接GPU → 动态→负载均衡!
   → → → 连接=活跃请求数 → 长请求→连接多→新请求→其他GPU!
   → → → → → 但: 连接数≠实际负载→KV不同→GPU压力不同!

3. Least Load (AI最优):
   → 请求→最少负载GPU → 负载=KV使用率+并发数+ITL!
   → → → vLLM metrics: gpu_cache_usage_pct → 实时 → 最优!
   → → → → → 但: 需GPU实时metrics → latency→过时→需容错!

4. Hash-based (Prefix Routing):
   → 请求→hash(prefix)→特定GPU → prefix→同一GPU→prefix sharing!
   → → → RAG/Agent→system prompt→hash→同一GPU→84%KV省!
   → → → → → 但: hash→不均匀→GPU负载不均→需动态调整!

5. Hybrid (生产推荐):
   → prefix hash + least load fallback → prefix→hash→命中率→miss→least load!
   → → → prefix命中→KV省84% → miss→least load→负载均衡 → 双优!

RTX 4090负载均衡:
  → 单GPU→不需要负载均衡 → 但: sleep/wake→多GPU→需路由!
  → → → 多RTX 4090→prefix hash→RAG天然适合→84%KV省!
  → → → → → 但: RTX 4090≥2→PCIe→FSDP灾难→独立GPU→独立路由→各自服务!
```

## 2. SLA-Aware Routing — SLA驱动路由

```
SLA路由=VIP优先 → SLA分级 → 商业化关键!

SLA路由策略:
  → Tier 1 (Premium): ITL<50ms P99 → 专用GPU → 低负载 → VIP体验!
  → → → Tier 2 (Standard): ITL<100ms P99 → 共享GPU → 高负载 → 正常体验!
  → → → → → Tier 3 (Free): ITL<500ms P99 → 共享GPU → 最低负载 → 可等!

路由决策:
  → Premium→专用GPU→B≤20→ITL<50ms→保证!
  → → → Standard→共享GPU→B≤55→ITL<15ms→超额保证!
  → → → → → Free→共享GPU→B=118→ITL≈24ms→基本保证!

Error Budget分级:
  → Premium: 99.99% → 4.32min/月 → 严格 → 1GPU专用!
  → → → Standard: 99.9% → 43.2min/月 → 正常 → 共享GPU!
  → → → → → Free: 99.5% → 216min/月 → 宽松 → 共享GPU高负载!

SLA路由实现:
  → API Gateway→priority header→路由→Tier GPU!
  → → → Nginx/Envoy→upstream→weight→优先级→分级!
  → → → → → vLLM: priority参数→0(默认)/1-3(VIP)→调度器→优先级队列!

RTX 4090 SLA路由:
  → 单GPU→3 Tier→batch管理→Premium少→Standard多→Free挤!
  → → → 多GPU→1 GPU→Premium专用→其余→Standard+Free共享!
  → → → → → sleep/wake→Free tier→sleep→Standard→wake→弹性→成本省!
```

## 3. API Endpoint Design — API端点设计

```
AI serving API设计对比:

1. REST HTTP (最常见):
   → POST /v1/completions → JSON → 简单 → OpenAI兼容!
   → → → 问题: HTTP/1.1→1连接→串行→吞吐低!
   → → → → → HTTP/2→多路复用→改善→但仍是request-response→每次新连接开销!
   → → → → → → → 适合: 一次性请求→聊天→简单→兼容→广泛!

2. gRPC (高性能):
   → protobuf → 二进制 → 小 → 快 → 流式!
   → → → 优点: 二进制→JSON省50%+ → 流式→实时 → 连接复用!
   → → → → → 但: 非HTTP标准→浏览器不支持→需gRPC-web→复杂!
   → → → → → → → 适合: 内部服务→微服务→低延迟→生产内部!

3. WebSocket (流式输出):
   → 双向→持久→流→token逐步→实时!
   → → → 优点: 1连接→多次→token→实时→chat→好体验!
   → → → → → 但: 连接管理→超时→断连→重连→复杂!
   → → → → → → → vLLM: streaming=True→Server-Sent Events(SSE)→类似WS但简单!

4. Server-Sent Events (SSE) — vLLM默认流式:
   → HTTP→单向→server→client→token→text/event-stream!
   → → → 优点: HTTP标准→浏览器原生→简单→不需WS!
   → → → → → 但: 单向→server→client→不能client→server→chat够!
   → → → → → → → vLLM: stream=True→SSE→每个token→data:→发送!

5. HTTP/3 (QUIC) — 未来:
   → UDP→0-RTT→多路复用→低延迟→未来标准!
   → → → 优点: UDP→不拥塞→快→0-RTT→首次连接快!
   → → → → → 但: 服务器支持少→2025仍不广泛→未来!

RTX 4090 API推荐:
  → 单GPU→REST+SSE→简单→OpenAI兼容→生产够!
  → → → 多GPU→内部gRPC→外部REST+SSE→两层→最优!
  → → → → → WebSocket→chat→实时→但管理复杂→SSE更简单→推荐!
```

## 4. Rate Limiting — 速率限制

```
Rate limiting=保护GPU→防过载→防DDoS→成本控制!

Rate limiting层级:

1. Global Rate Limit:
   → 总RPM(Requests Per Minute) → GPU容量限制 → 不超过!
   → → → RTX 4090 B=55 → 55 RPM → 但: 每请求不等→用token budget!
   → → → → → Token Rate Limit: max_tokens/min → 更精确 → 推荐!

2. Per-User Rate Limit:
   → 用户→RPM → 防DDoS → 公平 → 商业!
   → → → Free: 10 RPM → Standard: 100 RPM → Premium: 1000 RPM!
   → → → → → 但: 用户→多短请求→RPM低但token多→需token rate!

3. Per-Endpoint Rate Limit:
   → /v1/completions → 100 RPM → /v1/embeddings → 500 RPM → 分级!
   → → → 不同endpoint→不同计算量→不同限制 → 合理!

4. Adaptive Rate Limit:
   → GPU负载→动态调整 → 高负载→降RPM→保SLO → 低负载→升RPM→增吞吐!
   → → → gpu_cache_usage_pct>80% → 降50% → <50% → 升100% → 自适应!

Rate limiting算法:

1. Fixed Window:
   → 每分钟→固定→简单 → 但: 窗口边界→突发→2x!

2. Sliding Window:
   → 滑动→平滑→无突发 → 推荐 → Redis→sorted set!

3. Token Bucket:
   → token→桶→填充→消费→突发允许→平滑→最优!
   → → → burst=桶容量→允许突发→但长期=填充速率→限制!
   → → → → → 生产推荐: 令牌桶→突发+长期→平衡!

4. Leaky Bucket:
   → 漏→恒速→无突发→严格 → 但: 太严格→浪费!

RTX 4090 rate limiting推荐:
  → Token bucket → burst=10→refill=55/min → 突发+长期平衡!
  → → → GPU-aware: gpu_cache_usage→动态→高负载降→保SLO!
  → → → → → Per-user: Free 10RPM / Standard 100 / Premium 1000 → SLA分级!
```

## 5. Circuit Breaker — 熔断器

```
Circuit breaker=保护系统→防级联故障→故障隔离!

熔断器3态:

1. Closed (正常):
   → 请求→正常→GPU→响应 → 全部OK!
   → → → 监控: error_rate < 5% → 正常 → 请求放行!

2. Open (熔断):
   → 请求→拒绝→不转发→保护→快速失败!
   → → → 触发: error_rate > 50% → 或连续5次失败 → 立即熔断!
   → → → → → 拒绝: 503 Service Unavailable → 用户→稍后重试!
   → → → → → → → 冷却时间: 30s → 之后→half-open → 尝试恢复!

3. Half-Open (半开):
   → 少量请求→试探→成功→closed / 失败→open!
   → → → 试探: 1请求→GPU→成功→closed → 恢复!
   → → → → → 但: 试探失败→open → 继续冷却 → 保护!

AI serving熔断场景:
  → GPU OOM → 请求→拒绝→熔断→保护→恢复!
  → → → ITL P99>100ms → SLO violation→降级→熔断→保护其他!
  → → → → → GPU故障 → 无响应 → 熔断 → 路由→其他GPU → 故障转移!
  → → → → → → → 网络分区 → GPU→不可达 → 熔断 → 防级联!

熔断+路由:
  → GPU1→熔断→请求→GPU2→故障转移 → 高可用!
  → → → 所有GPU→熔断→503→降级→用户→稍后重试 → 诚实!

RTX 4090熔断:
  → 单GPU→熔断→503→无故障转移→简单→但: 单点故障!
  → → → 多GPU→熔断→路由→其他→高可用→但: 多GPU成本↑!
  → → → → → sleep/wake→GPU→sleep→熔断→wake→恢复→动态!
```

## 6. Graceful Degradation — 优雅降级

```
Graceful degradation=SLO违规时→降级→保核心→弃非核心!

降级策略层级:

Level 1: 减少并发 (最轻):
  → B=118→B=55 → 吞吐↓ → 但ITL恢复 → SLO合规!
  → → → 减少新请求接收 → 限流 → 等GPU恢复 → 最简单!

Level 2: 减少功能 (中等):
  → 禁用logprobs/top_logprobs → 减少output → 减少GPU工作!
  → → → 禁用streaming → batch→一次性 → 减少连接管理!
  → → → → → 禁用长上下文 → max_tokens限制→512 → 减少KV压力!

Level 3: 模型降级 (严重):
  → 7B→1.4B(SFT蒸馏) → 推理快5x → 但质量↓5-10%!
  → → → INT4→FP16降级 → 推理慢但更安全 → 防INT4 bug!

Level 4: 拒绝服务 (最严重):
  → 503 Service Unavailable → 拒绝所有请求 → 等恢复!
  → → → 熔断器→open → 拒绝 → 冷却 → half-open → 恢复!

降级触发条件:
  → ITL P99 > 100ms → Level 1 → 减少并发!
  → → → gpu_cache_usage > 80% → Level 2 → 减少功能!
  → → → → → error_rate > 10% → Level 3 → 模型降级!
  → → → → → → → error_rate > 50% → Level 4 → 拒绝!

降级恢复:
  → ITL恢复 → Level 4→3→2→1→0 → 逐步 → 不跳!
  → → → 每级→30s稳定→升级 → 防反复→降级→升级→降级→震荡!

RTX 4090降级策略:
  → 单GPU→Level 1(限流)+Level 2(限制功能) → 推荐!
  → → → 多GPU→故障转移→Level 3(降级模型) → 备用!
  → → → → → StreamingLLM→永不OOM→Level 1最多→最稳!
```

## 7. Timeout & Retry — 超时与重试

```
Timeout=请求最长等待 → Retry=失败后重试 → 两者配合→可靠性!

Timeout策略:
  → Connect timeout: 5s → GPU连接超时 → 网络问题!
  → → → Read timeout: 30s → GPU响应超时 → 计算慢/OOM!
  → → → → → Total timeout: 60s → 整体超时 → 用户等待极限!

  → → → → → → → vLLM建议: TTFT timeout=500ms → ITL关注!
  → → → → → → → → → Streaming timeout=30s per request → 流式持续!

Retry策略:
  → Retry count: 2-3 → 不太多 → 防DDoS放大!
  → → → Retry on: 5xx(server error) → 不retry: 4xx(client error)!
  → → → → → Retry delay: exponential backoff → 1s→2s→4s → 不立即!
  → → → → → → → Jitter: +random(0,1s) → 防所有客户端同时retry→thundering herd!

  → → → → → → → → → Retry different GPU → 故障转移 → 但: KV cache不同→recompute!

Idempotency:
  → completions → 非幂等 → 随机输出 → retry→不同结果!
  → → → 但: seed参数 → 幂等 → retry→相同 → 推荐!
  → → → → → embedding → 幂等 → retry→相同 → 安全!

RTX 4090 timeout/retry:
  → TTFT timeout=500ms → ITL monitor→30s → Total=60s → 推荐!
  → → → Retry=2 → backoff=1-2-4 → jitter → 不同GPU → 故障转移!
  → → → → → 但: 单GPU→retry=同GPU→无转移→简单→但: 可能OOM→retry也OOM!
```

## 8. Health Check — 健康检查

```
Health check=监控GPU→路由决策→熔断触发→基础!

健康检查层级:

1. Liveness (存活):
   → /health → GPU进程→alive? → 简单 → 进程存在→OK!
   → → → k8s livenessProbe → 重启→进程死→自动恢复!
   → → → → → 但: 进程alive≠GPU可用→OOM→进程活但服务不了!

2. Readiness (就绪):
   → /health/ready → GPU→ready? → 能服务请求?
   → → → vLLM: model loaded → KV cache allocated → CUDA graph captured → ready!
   → → → → → k8s readinessProbe → 不路由→未ready → 保护!
   → → → → → → → warmup: 30s(AOT) → ready → 路由 → 服务!

3. Startup (启动):
   → /health/startup → 初始化完成? → 长启动→startupProbe→不kill!
   → → → vLLM: model download+load+compile → startup → 2-5min → 长!
   → → → → → k8s startupProbe → 失败→不kill→等→初始化→长启动保护!

GPU健康指标:
  → GPU temperature < 85°C → 过热→降频→慢→预警!
  → → → GPU memory usage < 90% → OOM风险→限流→预警!
  → → → → → GPU utilization < 50% → 闲置→可接收请求!
  → → → → → → → Error rate < 1% → 正常 → >5% → 预警 → >50% → 熔断!

vLLM健康检查:
  → /health → liveness → 进程存活!
  → → → /v1/models → readiness → 模型加载!
  → → → → → Prometheus metrics → gpu_cache_usage → 动态 → 路由!

RTX 4090健康检查:
  → Liveness: 进程→alive → 简单!
  → → → Readiness: INT4+INT8KV→loaded→FlashInfer→ready → 2-5min!
  → → → → → GPU temp: RTX 4090→消费级→无ECC→需额外监控→输出验证!
```

## 9. Core Laws — 请求路由核心定律

1. **Least-Load Routing Law**: least load→GPU→负载均衡最优→需实时metrics→vLLM gpu_cache_usage!
   → → → Prefix hash→84%KV省→miss→least load→fallback→Hybrid最优!

2. **SLA-Tier Routing Law**: Premium→专用GPU→99.99%→Standard→共享→99.9%→Free→高负载→99.5%!
   → → → RTX 4090: 单GPU→batch管理→3 Tier→多GPU→1专用+2共享!

3. **Token-Rate Limiting Law**: token bucket→突发+长期→平衡→RPM不精确→token/min更准确!
   → → → GPU-aware: 动态→高负载降→保SLO→低负载升→增吞吐!

4. **Circuit-Breaker Protection Law**: 熔断→防级联→5xx→open→30s冷却→half-open→closed→恢复!
   → → → 单GPU→503→多GPU→故障转移→RTX 4090→sleep/wake→动态!

5. **Graceful-Degradation Law**: 4级降级→限流→减功能→降模型→拒绝→逐步→不跳→30s稳定→升级!
   → → → ITL>100ms→限流 / cache>80%→减功能 / error>10%→降模型 / error>50%→拒绝!

6. **Timeout-Backoff Law**: timeout=分级(5s/30s/60s) → retry=2-3 → exponential backoff → jitter!
   → → → 幂等: seed参数→completions幂等→retry→相同→推荐!

7. **Health-Readiness Law**: liveness≠readiness→进程活≠能服务→startupProbe→长启动保护→30s warmup!
   → → → RTX 4090: 2-5min启动→readiness→model loaded+KV ready→路由!

## 关键参考

- vLLM V1: /health + /v1/models + priority + gpu_cache_usage metrics
- SGLang: similar health + RadixAttention prefix routing
- OpenAI API: /v1/completions + SSE streaming + rate limiting
- k8s probes: liveness/readiness/startup → 3级健康检查
- Envoy/Nginx: least connections + circuit breaker + retry
- RTX 4090: sleep/wake 1s → dynamic routing → cost optimization