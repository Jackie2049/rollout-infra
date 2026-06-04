# LLM 推理服务框架

> 目标：理解生产环境 LLM 推理服务架构，掌握 vLLM/TGI/Triton/SGLang 选型

## 1. 推理服务架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    生产环境 LLM 推理服务                          │
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────────┐  │
│  │  Client   │───▶│  API 网关  │───▶│  Load Balancer (K8s LB)  │  │
│  │ (HTTP/gRPC)   │ (Nginx/   │    │  轮询/最少连接/Hash 路由   │  │
│  └──────────┘    │  Envoy)   │    └──────────────────────────┘  │
│                  └───────────┘              │                    │
│                                    ┌────────┼────────┐           │
│                                    ▼        ▼        ▼           │
│                              ┌────────┐┌────────┐┌────────┐     │
│                              │Replica1││Replica2││Replica3│     │
│                              │ vLLM/  ││ vLLM/  ││ vLLM/  │     │
│                              │SGLang  ││SGLang  ││SGLang  │     │
│                              │TP=4    ││TP=4    ││TP=4    │     │
│                              │[GPU×4] ││[GPU×4] ││[GPU×4] │     │
│                              └────────┘└────────┘└────────┘     │
│                                                                  │
│  监控: Prometheus + Grafana + OpenTelemetry                      │
│  扩缩: HPA (GPU利用率/请求队列/Running请求数)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 主要框架对比

| 维度 | vLLM | SGLang | TGI | Triton+TRT-LLM | llama.cpp |
|------|------|--------|-----|-----------------|-----------|
| 语言 | Python | Python | Rust+Python | C++ | C/C++ |
| 状态 | 活跃 | 活跃 (快速成长) | **维护模式** | 活跃 | 活跃 |
| Continuous Batching | V1 Scheduler | 零开销 CPU 调度 | 有 | Iterative Sequences | 无 |
| Attention Backend | 20+ (Flash/MLA...) | RadixAttention | Flash+Paged | TensorRT 优化 | 无 |
| Prefix Caching | Hash-based | RadixTree (更优) | 有 | 有 | 有 |
| Speculative Decoding | 8+ proposer | 有 | 基本 | 有 | 有 |
| 量化 | 30+ 方法 | FP4/FP8/INT4 | 7+ 方法 | FP8/INT4/INT8 | GGUF 1.5-8bit |
| Multi-LoRA | Punica (完整) | 有限 | 有限 | 无 | 无 |
| 多模态 | 完整 | 有 | 有限 | 有 | 无 |
| 硬件 | NVIDIA/AMD/Intel/TPU | NVIDIA/AMD/Intel/TPU | NVIDIA/AMD | NVIDIA only | 全平台 |
| 适用场景 | 通用生产服务 | 高吞吐/前缀密集 | HF 生态 (已过时) | 纯 NVIDIA 企业级 | 边缘/本地 |

## 3. TGI (Text Generation Inference)

### 架构

```
Client (HTTP/gRPC)
       │
       ▼
[Rust 前端] ← 高性能 HTTP/gRPC 处理
       │ gRPC
       ▼
[Python 后端] ← 模型加载、推理、解码
       │
       ▼
[CUDA/PyTorch] ← FlashAttention, PagedAttention
```

### 关键特性
- Continuous Batching (iteration-level)
- Flash Attention + Paged Attention
- Tensor Parallelism (多 GPU)
- Speculative Decoding (~2x 加速)
- SSE Token Streaming

### 现状 (2025)
**已进入维护模式**。HuggingFace 推荐使用 vLLM/SGLang/llama.cpp 替代。

## 4. Triton Inference Server

### 架构

```
[Model Repository] ← 模型注册中心 (目录结构)
       │
[Request Scheduler] ← 每模型调度、分批、队列
       │
[Backend System] ← 可插拔执行引擎
  ├── TensorRT-LLM  (LLM 最高吞吐)
  ├── vLLM Backend  (直接嵌入 vLLM)
  ├── Python Backend (自定义逻辑)
  ├── PyTorch Backend
  ├── ONNX Runtime
  └── TensorRT      (非 LLM GPU 优化)
       │
[GPU/CPU Hardware]
```

### 调度模式

**Dynamic Batcher**: 自动将请求组成 batch
```protobuf
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 1000
  preserve_ordering: true
}
```

**Iterative Sequences**: LLM 的 continuous batching 实现
- 每步从所有活跃序列组成 batch
- 序列完成 (EOS) → 立即释放 slot 给新请求
- 等价于 vLLM 的 continuous batching

### 优势场景
- 多模型共存服务 (LLM + 视觉 + 分类)
- 企业级需要 NVIDIA 支持合同
- 精细化批处理控制 (优先级、队列策略)
- 同一基础设施服务多种模型类型

## 5. SGLang vs vLLM 深度对比

### SGLang 的优势

**RadixAttention**: 树结构前缀缓存
```
请求 A: [system][query_A]
请求 B: [system][query_B]  → 共享 system 前缀节点
请求 C: [system][query_A][response][followup]  → 复用 A 的完整路径

Radix Tree:
        [system]
        /       \
   [query_A]   [query_B]
      |
  [response]
      |
  [followup]
```
- 自动引用计数
- O(1) 前缀查找 (Radix Tree vs vLLM 的 Hash 链式查找)
- 零开销 CPU 调度器

**大规模生产验证**:
- xAI, NVIDIA, LinkedIn, Cursor 等采用
- 支撑 400,000+ GPU 规模部署

### vLLM 的优势

**更广泛的生态**:
- 30+ 量化方法 (SGLang ~5)
- 8+ Speculative Decoding proposer (SGLang ~3)
- 完整 Multi-LoRA (Punica)
- 20+ Attention Backend
- 多模态完整支持

**更成熟的可观测性**:
- 30+ Prometheus 指标
- OpenTelemetry 集成
- Grafana 仪表板

**更灵活的集成**:
- KServe, KubeRay, AIBrix, KAITO
- P/D 分离 (NIXL/FlexKV/Mooncake/LMCache)

### 选型建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 通用生产服务 | vLLM | 生态最全，模型支持最广 |
| RL 训练 rollout | SGLang | RadixAttention 前缀复用，零开销调度 |
| 高吞吐前缀密集 | SGLang | RadixAttention 优势明显 |
| 多模型混合服务 | Triton | 唯一支持 LLM+非LLM 共存 |
| 纯 NVIDIA 最高性能 | Triton+TRT-LLM | NVIDIA kernel 级优化 |
| 本地/边缘推理 | llama.cpp | 全平台，零依赖 |
| 快速原型验证 | vLLM | pip install 即用 |

## 6. Kubernetes 生产部署

### vLLM K8s 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 2  # TP=2
        args:
          - --model
          - meta-llama/Llama-3-70B
          - --tensor-parallel-size
          - "2"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_num_requests_running
      target:
        type: Utilization
        averageUtilization: 70
```

### 自动扩缩策略

**关键指标**:
- `vllm_num_requests_running`: 活跃请求数
- `vllm_num_requests_waiting`: 排队请求数
- GPU 利用率
- KV Cache 使用率

**注意事项**:
- **冷启动**: 加载 70B 模型需 30-120s，不要 scale-to-zero
- **GPU 亲和**: Pod 必须调度到 GPU 节点
- **TP 约束**: TP=4 需 4 GPU 同节点，扩缩以 TP 组为单位
- **KV Cache 预热**: 新副本初始 KV 为空，前缀缓存可缓解

### 负载均衡策略

**无状态** (标准):
- 轮询 / 最少连接
- 适用于独立请求

**有状态** (前缀缓存优化):
- Hash 路由: 相似前缀的请求路由到同一副本
- 最大化 KV Cache 命中
- SGLang RadixAttention: Tree 结构天然支持
- vLLM: 需要外部路由代理实现

### A/B 测试与金丝雀部署

**KServe**: 内置金丝雀发布
```yaml
# 10% 流量到 v2, 90% 到 v1
canaryTrafficPercent: 10
```

**Istio/Envoy**: 服务网格级加权路由

**Shadow Deployment**: 新模型并行运行，比较输出但不服务用户

## 7. 关键洞察

1. **TGI 已过时**: 维护模式，新项目不应选用
2. **vLLM vs SGLang**: vLLM 生态更全，SGLang 吞吐更高 (RadixAttention)
3. **Triton 定位**: 多模型混合服务的企业级平台
4. **生产核心问题**: 冷启动、GPU 调度、TP 约束、KV 预热
5. **P/D 分离**: 新趋势，预计算和推理分离到不同硬件池
6. **K8s 生态**: KServe + KubeRay + KAITO 是主流部署方式
7. **量化是标配**: FP8 首选 (<0.5% 损失), INT4 AWQ 次选

## 参考资料

- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [NVIDIA Triton Documentation](https://github.com/triton-inference-server/server)
- [KServe Documentation](https://kserve.github.io/)
- [vLLM Production Setup](https://docs.vllm.ai/en/latest/serving/deploying.html)
