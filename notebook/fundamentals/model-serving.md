# 模型服务架构

> 将大模型部署为可用的服务 — 从单卡到千卡推理集群

## 1. 服务架构概览

```
用户请求 → API Gateway → 负载均衡 → 推理实例 × N
              ↓
          请求排队 → 调度器 → GPU 执行 → 响应

核心组件:
  1. API 层 — 接收请求、解析参数、格式化响应
  2. 调度层 — 请求路由、负载均衡、优先级管理
  3. 推理层 — 模型加载、batch 管理、KV Cache 管理
  4. 基础设施层 — 容器编排、自动扩缩、监控告警
```

## 2. API 设计

### 2.1 OpenAI 兼容接口 (事实标准)

```python
# POST /v1/chat/completions
# 请求:
{
    "model": "llama-3-70b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true,           # 流式返回
    "top_p": 0.9,
    "frequency_penalty": 0.0
}

# 响应 (非流式):
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "llama-3-70b",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help you?"},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28
    }
}

# 响应 (流式 SSE):
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"},"index":0}]}
data: [DONE]
```

### 2.2 关键 API 设计决策

```
1. 同步 vs 异步:
   - 同步: 请求阻塞等待完成 (简单，适合短请求)
   - 异步: 返回 task_id，轮询结果 (适合长请求)
   - 流式 SSE: 逐步返回 token (ChatGPT 标准)

2. 请求参数:
   - model: 模型名称 (支持路由到不同模型)
   - messages: 对话上下文
   - max_tokens: 最大生成长度
   - temperature / top_p / top_k: 采样参数
   - stop: 停止词
   - stream: 是否流式返回

3. 错误处理:
   - 400 Bad Request: 参数错误
   - 429 Too Many Requests: 限流
   - 500 Internal Server Error: 模型错误
   - 503 Service Unavailable: 模型加载中
```

## 3. 负载均衡

### 3.1 路由策略

```
策略 1: Round Robin
  请求均匀分配到每个实例
  简单但不考虑实例状态
  适合: 请求大小均匀的场景

策略 2: Least Connections
  选择当前连接数最少的实例
  考虑实例负载
  适合: 请求大小不均匀

策略 3: Prefix-Aware Routing (vLLM / SGLang)
  将相似前缀的请求路由到同一实例
  利用 KV Cache 的 prefix caching
  显著降低首 token 延迟
  适合: 多用户共享 system prompt

策略 4: 基于队列长度
  选择排队请求最少的实例
  减少平均等待时间
```

### 3.2 请求调度

```
Continuous Batching (vLLM/SGLang):
  不等 batch 填满，而是:
  - 有请求就处理
  - 生成的请求完成一个就插入一个新的
  - iteration-level scheduling

调度决策:
  1. 新请求到达 → 放入 waiting queue
  2. 每个 iteration:
     a. 检查 running requests 中完成的 → 移出
     b. 检查 waiting queue → 按 priority 和 KV Cache 可用性插入
  3. 约束: 总 token 数 < max_num_tokens (防 OOM)

优先级策略:
  - 长上下文请求: 低优先级 (占用资源多)
  - 短对话请求: 高优先级 (快速响应)
  - 超时请求: 最低优先级
```

## 4. 多副本部署

### 4.1 副本管理

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 4  # 4 个推理副本
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 2  # 每副本 2 GPU (TP=2)
        command:
          - python
          - -m
          - vllm.entrypoints.openai.api_server
          - --model
          - meta-llama/Llama-3-70B
          - --tensor-parallel-size
          - "2"
          - --max-model-len
          - "8192"
          - --gpu-memory-utilization
          - "0.9"
```

### 4.2 自动扩缩容 (HPA + Custom Metrics)

```yaml
# 基于自定义指标的 HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  minReplicas: 2
  maxReplicas: 16
  metrics:
  - type: Pods
    pods:
      metric:
        name: num_requests_waiting  # 自定义指标
      target:
        type: AverageValue
        averageValue: "10"  # 平均等待请求 > 10 就扩容
  - type: Resource
    resource:
      name: nvidia.com/gpu_utilization
      target:
        type: Utilization
        averageUtilization: 80  # GPU 利用率 > 80% 就扩容

扩缩策略:
  Scale Up:   等待 60s，每次 +4 副本 (快速响应)
  Scale Down: 等待 300s，每次 -1 副本 (保守缩容)
```

### 4.3 模型加载策略

```
1. 预加载 (Preload):
   Pod 启动时加载模型
   首次请求零延迟
   Pod 启动慢 (70B 模型加载 ~2-5 分钟)

2. 懒加载 (Lazy Load):
   首次请求时加载
   节省资源但首次延迟大
   不推荐生产使用

3. 模型热交换:
   不重启 Pod 切换模型
   vLLM 不原生支持
   需要自行实现 (SGLang 部分支持)

4. Speculative Model Loading:
   预测即将需要的模型
   提前加载到 GPU
   适合多模型服务
```

## 5. 健康检查与容错

### 5.1 健康检查

```python
# /health 端点
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_used": torch.cuda.memory_allocated(),
        "gpu_memory_total": torch.cuda.get_device_properties(0).total_mem,
        "num_running_requests": len(running_requests),
    }

# /ready 端点 (K8s readiness probe)
@app.get("/ready")
async def ready():
    if model is None:
        return Response(status_code=503)
    if torch.cuda.memory_allocated() > 0.95 * total_gpu_memory:
        return Response(status_code=503)  # OOM risk
    return {"status": "ready"}
```

### 5.2 容错机制

```
1. 请求超时:
   - 设置 max_wait_time (如 30s)
   - 超时返回 504 Gateway Timeout
   - 避免请求无限等待

2. 请求重试:
   - 503 → 重试其他副本 (最多 3 次)
   - 500 → 不重试 (应用错误)
   - 使用指数退避

3. 熔断:
   - 副本连续 N 次失败 → 标记为 unhealthy
   - 路由跳过 unhealthy 副本
   - 定期尝试恢复

4. 优雅关闭:
   - SIGTERM → 停止接收新请求
   - 等待 running requests 完成 (最多 60s)
   - 强制关闭残留请求
```

## 6. 滚动更新

```
策略: 先扩后缩 (Surge Rolling Update)

旧版本: [v1] [v1] [v1] [v1]
Step 1: [v1] [v1] [v1] [v1] [v2] ← 新增 v2 副本
Step 2: [v1] [v1] [v1] [v2] [v2] ← 新增, 旧副本 drain
Step 3: [v1] [v1] [v2] [v2] [v2] ← 继续
Step 4: [v1] [v2] [v2] [v2]     ← 缩减旧副本
Step 5: [v2] [v2] [v2] [v2]     ← 完成

关键点:
  - 新副本 ready 后再缩减旧副本 (零停机)
  - LLM 场景: 新副本模型加载期间旧副本继续服务
  - 总资源峰值 = 旧副本 + 新副本 (需要额外 GPU)
  - 模型加载慢 → 整个更新过程 ~10-20 分钟
```

## 7. 推理性能指标

```
关键指标:

1. Time to First Token (TTFT):
   从请求到第一个 token 的时间
   受 prefill 速度影响
   目标: < 500ms (短上下文), < 2s (长上下文)

2. Inter-Token Latency (ITL):
   生成每个 token 的平均时间
   受 decode 速度影响
   目标: < 50ms/token

3. Throughput:
   tokens/second (总吞吐量)
   requests/second (请求吞吐量)
   受 batch size 和模型利用率影响

4. End-to-End Latency (E2E):
   从请求到完整响应的时间
   = TTFT + num_tokens × ITL

5. GPU Utilization:
   目标: > 80% (高负载)
   < 50% → 需要 larger batch 或更多请求

监控栈:
  DCGM Exporter → Prometheus → Grafana
  vLLM metrics → Prometheus → Grafana
```

## 8. 推理引擎对比

```
| 特性            | vLLM          | TensorRT-LLM   | SGLang       |
|----------------|---------------|----------------|--------------|
| 开发语言        | Python        | C++/Python     | Python       |
| 调度策略        | Continuous    | Continuous     | Continuous   |
| KV Cache       | PagedAttention| TensorRT MM    | RadixAttention|
| Prefix Caching | 支持 (hash)   | 支持           | 支持 (Radix) |
| TP/PP          | 支持          | 支持           | 支持         |
| Speculative    | 支持          | 支持           | 支持         |
| 量化            | GPTQ/AWQ/FP8  | INT8/FP8       | GPTQ/AWQ/FP8 |
| 社区活跃度      | 高            | 中 (NVIDIA)    | 高           |
| 部署难度        | 低 (pip)      | 高 (编译)      | 低 (pip)     |

选择建议:
  快速上手: vLLM (最活跃的开源生态)
  最高性能: TensorRT-LLM (NVIDIA 深度优化)
  前缀复用: SGLang (RadixAttention 最优)
```

## 9. 学习要点

1. **OpenAI API 是事实标准** — 所有框架都兼容这个接口
2. **Continuous Batching 是核心** — 区别于传统 static batching，显著提高吞吐
3. **Prefix-Aware Routing 很重要** — 相似前缀路由到同一实例，利用 KV Cache 复用
4. **模型加载是部署瓶颈** — 70B 模型加载需要数分钟，滚动更新需考虑
5. **监控 TTFT 和 ITL** — 推理服务最重要的两个延迟指标
6. **HPA 基于等待队列** — GPU 利用率不适合直接做扩缩容指标（满载不一定需要扩容）

## 参考

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [vLLM Serving Documentation](https://docs.vllm.ai/en/latest/serving/index.html)
- [SGLang Architecture](https://github.com/sgl-project/sglang)
- [Triton Inference Server](https://github.com/triton-inference-server)
- [KServe](https://github.com/kserve/kserve)
