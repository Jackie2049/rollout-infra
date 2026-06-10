# AI Model Deployment & Production Deep Dive

> 2026-06-10 | 模型部署=AI从实验到生产的最后一步! 从safetensors到容器化, 从CI/CD到版本管理, 从rolling deploy到canary, 部署=工程化的极致!
> 关联: ai-serving-cost-optimization-deep-dive.md, ai-serving-observability-deep-dive.md, ai-request-routing-endpoint-optimization-deep-dive.md

## 0. 核心定律: 部署 = 模型 → 用户 的安全桥梁

训练好的模型≠生产 → 需要: 打包→容器→部署→监控→回滚 → 安全桥梁!

关键数据:
- Safetensors load: 1.2s (7B) → vs pickle: 2.5s + 不安全 → 2x快+安全!
- Docker GPU: nvidia-docker → CUDA→容器→隔离 → 生产必需!
- Cold start: 30s (HPA) → vs 1s (sleep/wake) → 30x差距!
- Rolling deploy: 0 downtime → vs blue-green: 双资源 → vs canary: 渐进 → 分场景!

## 1. Model Packaging — 模型打包

```
模型打包格式对比:

1. PyTorch Native (.pt/.pth) → pickle:
   → 优点: PyTorch原生 → 简单 → 任何对象可保存!
   → → → 缺点: 不安全! → pickle→执行任意代码→恶意模型→攻击!
   → → → → → 大: 完整state_dict → 未压缩 → 磁盘大!
   → → → → → → → 速度: load 7B ≈ 2.5s → 慢 → pickle overhead!

2. SafeTensors (.safetensors) — HuggingFace推荐:
   → 优点: 安全! → 零代码执行 → 仅张量数据 → 无pickle!
   → → → → → 快: load 7B ≈ 1.2s → mmap → 2x faster → 零拷贝!
   → → → → → → → 小: 无pickle元数据 → 紧凑 → 省5-10%!
   → → → → → → → → → mmap: memory-mapped → 不全读→按需→省内存!

3. GGUF (.gguf) — llama.cpp/本地推理:
   → 优点: 量化内置 → INT4/INT8/Q8_0 → 推理直接用!
   → → → → → 小: INT4 → 3.5GB(7B) → vs BF16 14GB → 4x省!
   → → → → → → → CPU推理: llama.cpp → 无GPU → 本地!
   → → → → → → → → → 缺点: 不通用 → 仅llama.cpp → vLLM不支持!

4. ONNX (.onnx) — 跨框架:
   → 优点: 跨框架 → PyTorch→ONNX→TensorRT→推理!
   → → → → → TensorRT: 优化推理 → NVIDIA专用 → 2-5x加速!
   → → → → → → → 缺点: 转换不完美 → 动态shape→难 → 复杂!
   → → → → → → → → → vLLM: 不用ONNX → FlashInfer+CUDA → 更快!

RTX 4090最优打包:
  → 生产推理: SafeTensors → vLLM → mmap → 1.2s → 推荐!
  → → → 本地推理: GGUF INT4 → llama.cpp → 3.5GB → 无GPU → 便携!
  → → → → → TensorRT: ONNX→TRT→2x加速→但vLLM FlashInfer→更快→不需要!
  → → → → → → → 绝对不用: pickle → 不安全+慢 → 已淘汰!
```

## 2. Containerization — 容器化

```
AI serving容器化关键:

1. Docker基础镜像:
   → nvidia/cuda:12.8.0-devel-ubuntu22.04 → CUDA开发 → 编译kernel!
   → → → nvidia/cuda:12.8.0-runtime-ubuntu22.04 → CUDA运行 → 小 → 推理!
   → → → → → vLLM官方镜像: vllm/vllm-openai:latest → 一切内置 → 简单!
   → → → → → → → 但: 镜像大→5-10GB→CUDA+Python+vLLM→推慢→需优化!

2. nvidia-container-toolkit:
   → Docker→GPU → nvidia-container-toolkit → GPU直通 → 生产必需!
   → → → docker run --gpus all → 所有GPU → 或 --gpus 0,1 → 指定GPU!
   → → → → → RTX 4090: --gpus 0 → 单GPU推理 → 不需要多GPU!
   → → → → → → → 多GPU: --gpus all → 但PCIe→FSDP灾难→单GPU最优!

3. 多阶段构建 (Multi-stage build):
   → Stage 1: 编译 → CUDA devel → 编译kernel → FlashInfer → CUTLASS!
   → → → Stage 2: 运行 → CUDA runtime → 仅二进制 → 小镜像!
   → → → → → 优点: 最终镜像小 → 2-3GB → vs 单阶段 10GB → 3x省!
   → → → → → → → 缺点: 编译慢 → 30-60min → 但仅一次 → CI/CD!

4. 镜像优化策略:
   → .dockerignore → 排除测试/文档/缓存 → 精简!
   → → → apt最小化 → 不装vim/tmux → 仅必需 → 瘦身!
   → → → → → pip缓存 → 不保留 → --no-cache-dir → 省1-2GB!
   → → → → → → → squash → Docker --squash → 合层 → 省重复!

5. Kubernetes部署:
   → Pod→1容器→1vLLM→1GPU → 简单 → 单GPU推理!
   → → → Deployment→副本→N Pod → 多GPU→但各自独立→不互联!
   → → → → → Service→ClusterIP→负载均衡 → 请求→分发→各Pod!
   → → → → → → → HPA→autoscale→CPU/GPU metrics→扩缩 → 但cold start 30s!
   → → → → → → → → → sleep/wake→vLLM→1s→推荐→30x快→RTX 4090最优!

RTX 4090容器化推荐:
  → 单GPU→1 Pod→1 vLLM→INT4+INT8KV → 简单 → 最优!
  → → → Docker: nvidia/cuda runtime → 小镜像 → 2-3GB!
  → → → → → K8s: Deployment+Service → sleep/wake autoscale → 1s → 推荐!
  → → → → → → → 不需要HPA→cold start 30s→太慢→sleep/wake更好!
```

## 3. CI/CD Pipeline — CI/CD流水线

```
AI serving CI/CD关键步骤:

1. Model Validation (模型验证):
   → 下载模型 → 加载 → 推理测试 → 输出检查 → 安全!
   → → → vLLM: python -m vllm.entrypoints.openai.api_server → 启动!
   → → → → → 基准测试: 7B INT4 → latency/throughput → vs baseline → 检查!
   → → → → → → → 安全: safetensors → 无pickle → hash验证 → 信任!

2. Container Build (容器构建):
   → Docker build → 多阶段 → FlashInfer+cuBLAS+CUDA Graph → 镜像!
   → → → 优化: --squash + .dockerignore + --no-cache-dir → 小!
   → → → → → 测试: docker run → 简单推理 → 输出正确 → OK!
   → → → → → → → 推送: registry → Harbor/ECR/GCR → 存储!

3. Integration Test (集成测试):
   → K8s部署 → 完整serving → 多请求 → SLO检查 → 生产级!
   → → → TTFT<500ms / ITL<100ms / throughput>1K → SLO合规!
   → → → → → Prefix sharing → RAG请求 → 验证 → 功能完整!
   → → → → → → → 错误处理 → OOM → 限流 → 熔断 → 验证!

4. Deployment (部署):
   → Rolling → 渐进 → 0 downtime → 推荐!
   → → → Canary → 5%流量 → 观察 → 100% → 渐进 → 更安全!
   → → → → → Blue-Green → 双环境 → 切换 → 快但贵 → 双资源!

5. Monitoring (监控):
   → Prometheus → vLLM metrics → ITL/TTFT/cache → SLO!
   → → → Grafana → dashboard → 可视化 → 人类理解!
   → → → → → Alert → SLO违规 → PagerDuty → 响应!

RTX 4090 CI/CD推荐:
  → 简单CI: git push→Docker build→测试→推送 → 5-10min!
  → → → 简单CD: rolling update→1 Pod→验证→完成 → 2-3min!
  → → → → → 不需要复杂canary→单GPU→简单→rolling足够!
  → → → → → → → 监控: Prometheus+Grafana → vLLM metrics → SLO!
```

## 4. Model Versioning & Registry — 模型版本管理

```
模型版本管理=生产的回滚能力!

版本管理策略:

1. HuggingFace Hub — 开源模型registry:
   → model_id → 版本 → 分支 → tag → 下载!
   → → → vLLM: model_id="Qwen/Qwen2.5-7B-Instruct" → 自动下载!
   → → → → → 但: 大模型→下载慢→10GB→HF网络→中国慢→需镜像!
   → → → → → → → HF-Mirror: hf-mirror.com → 中国镜像 → RTX 4090推荐!

2. Local Model Store — 本地存储:
   → /models/v1.0/ → /models/v1.1/ → 版本 → 目录!
   → → → vLLM: model_path="/models/Qwen2.5-7B-v1" → 本地加载 → 快!
   → → → → → 但: 无远程→无分享→手动→小团队可用!

3. MLflow Model Registry — 生产registry:
   → 实验→注册→版本→阶段(Staging→Production→Archived)!
   → → → MLflow: model.log()→注册→版本化→生产管理!
   → → → → → 优点: 完整生命周期 → 实验→staging→production → 管理!

4. vLLM模型管理:
   → model_config.json → 参数 → 版本 → 配置!
   → → → HF download → safetensors → mmap → 加载 → 1.2s!
   → → → → → 热更新: 不需要→vLLM不支持→需要重启→2-5min!

版本回滚:
  → v1.0 → production → OK → v1.1 → canary → error → rollback → v1.0!
  → → → 回滚策略: 保留最近3版本 → 随时回滚 → 安全!
  → → → → → K8s: Deployment→rollback→1 command → 简单!

RTX 4090版本管理推荐:
  → HF-Mirror → 中国镜像 → 下载快 → 推荐!
  → → → 本地存储 → 下载后保存 → 不每次下载 → 省!
  → → → → → 保留3版本 → 随时回滚 → 安全 → K8s rollback!
```

## 5. Deployment Strategies — 部署策略

```
3种部署策略对比:

1. Rolling Update (滚动更新) — K8s默认:
   → 逐步替换 → 新Pod→启动→旧Pod→终止 → 0 downtime!
   → → → 速度: 1-5min → 新Pod启动→验证→替换→完成!
   → → → → → 成本: 1x资源 → 不需要额外 → 最省!
   → → → → → → → 风险: 中 → 如果新版本有问题→部分用户→受影响!
   → → → → → → → → → 回滚: 快 → K8s rollback → 1 command!

2. Blue-Green Deployment:
   → 两套环境 → blue(v1)→running + green(v2)→standby → 切换!
   → → → 切换: Service→selector→blue→green → 立即 → 0影响!
   → → → → → 成本: 2x资源 → 双GPU → RTX 4090→2台→贵!
   → → → → → → → 风险: 低 → green→验证→OK→切换 → 最安全!
   → → → → → → → → → 回滚: 最快 → 切回blue → 1s → 最快!

3. Canary Deployment (金丝雀):
   → 5%流量→新版本 → 观察 → 逐步→100% → 渐进!
   → → → 观察: ITL/TTFT/error_rate → SLO → OK→增加→10%→25%→50%→100%!
   → → → → → 成本: 1x+5% → 少量额外 → 省钱!
   → → → → → → → 飀险: 最低 → 渐进 → 问题→停止 → 影响小!
   → → → → → → → → → 回滚: 停止新流量 → 切回旧 → 简单!

部署策略决策树:
  → 单GPU/简单 → Rolling → 简单→0 downtime→推荐!
  → → → 双GPU/高SLO → Blue-Green → 最安全→双资源→贵!
  → → → → → 大规模/高风险 → Canary → 渐进→最安全→推荐!
  → → → → → → → RTX 4090: Rolling → 简单→单GPU→0 downtime→推荐!

热更新挑战:
  → vLLM不支持热更新 → 需重启 → 2-5min → 新Pod启动!
  → → → 解决: Rolling → 逐Pod → 0 downtime → 但: 旧Pod→旧模型→混合!
  → → → → → 模型一致性: 旧请求→旧模型→完成→新Pod→新模型→替换→渐进!
```

## 6. A/B Testing & Model Evaluation — A/B测试与模型评估

```
A/B测试=生产环境模型对比 → 真实用户→真实数据→真实结果!

A/B测试架构:
  → 用户→路由→模型A(50%) / 模型B(50%) → 比较!
  → → → 分流: hash(user_id) → A/B → 同一用户→同一模型→稳定!
  → → → → → Metrics: ITL/TTFT/满意度/正确率 → 对比!

A/B测试指标:
  → Latency: ITL_A=13ms vs ITL_B=15ms → B慢2ms → 不显著!
  → → → Quality: accuracy_A=93% vs accuracy_B=95% → B好2% → 可能显著!
  → → → → → Cost: cost_A=$0.03/1M vs cost_B=$0.05/1M → A便宜 → trade-off!

  → → → → → → → 统计显著性: t-test → p<0.05 → 显著 → 确认!
  → → → → → → → → → 需要样本量: 1000+请求 → 才能判断 → 不能太少!

模型评估生产级:
  → Offline eval → benchmark → MMLU/GSM8K → 快 → 但不真实!
  → → → Online eval → A/B test → 真实用户 → 慢 → 但真实!
  → → → → → Shadow mode → 新模型→不服务→仅推理→对比→最安全!
  → → → → → → → → → 用户→旧模型→服务 + 新模型→shadow→不服务→离线对比!

RTX 4090 A/B测试:
  → 双GPU → A+独立B → 分流 → 最简单!
  → → → 单GPU → 交替 → A→5min→B→5min → 不公平→不推荐!
  → → → → → Shadow → 1 GPU → 旧模型服务 + 新模型→离线→对比→最安全!
```

## 7. Production Lifecycle — 生产生命周期

```
AI serving生产生命周期6阶段:

1. Development (开发):
   → 实验→训练→评估→选择模型 → 本地 → 快速迭代!

2. Staging (预发布):
   → 容器化→配置→测试→验证 → 非生产 → 安全!

3. Canary (金丝雀):
   → 5%流量→新模型→观察→验证 → 渐进 → 低风险!

4. Production (生产):
   → 100%流量→模型→服务 → 监控→SLO→运维!

5. Monitoring (监控):
   → Prometheus→metrics→dashboard→alert → 实时 → SLO!

6. Retirement (退役):
   → 旧版本→归档→删除 → 节省资源 → 版本管理!

关键运维操作:
  → 扩缩容: sleep/wake→1s→比HPA 30s→30x快→RTX 4090推荐!
  → → → 回滚: K8s rollback→1命令→2-3min→安全!
  → → → → → 升级: rolling update→0 downtime→渐进!
  → → → → → → → 故障: circuit breaker→熔断→恢复→保护!

RTX 4090生产生命周期推荐:
  → Dev→本地训练+推理 → Staging→Docker→测试 → Prod→K8s→rolling!
  → → → 监控→Prometheus+Grafana → 扩缩→sleep/wake → 回滚→K8s!
```

## 8. Core Laws — 部署生产核心定律

1. **SafeTensors-Over-Pickle Law**: safetensors→安全+2x快+小 → pickle→不安全+慢 → 生产必须safetensors!
   → → → mmap→零拷贝→1.2s vs pickle→2.5s → RTX 4090推荐!

2. **Single-GPU-Container Law**: RTX 4090→1 Pod→1 vLLM→1 GPU → 独立→不互联→PCIe不需要!
   → → → sleep/wake autoscale→1s→比HPA 30s→30x快→RTX 4090最优!

3. **Multi-Stage-Build Law**: 多阶段构建→编译+运行→最终镜像2-3GB vs 单阶段10GB→3x省!
   → → → CI/CD→一次编译→多次部署→AOT→生产!

4. **Rolling-Deploy-Simple Law**: Rolling→0 downtime→1x资源→简单→RTX 4090推荐!
   → → → Blue-Green→2x资源→最安全但贵→Canary→渐进→大规模用!

5. **Model-Version-Rollback Law**: 保留最近3版本→随时回滚→K8s 1命令→安全!
   → → → HF-Mirror→中国镜像→下载快→RTX 4090推荐!

6. **Shadow-Eval-First Law**: Shadow mode→新模型→不服务→离线对比→最安全→生产评估第一步!
   → → → A/B→5%流量→渐进→确认→Shadow→先→A/B→后→安全!

7. **Production-6-Phase Law**: Dev→Staging→Canary→Prod→Monitor→Retire → 6阶段→生命周期→管理!
   → → → 每阶段→验证→下一阶段→不跳→安全→RTX 4090→简单流程!

## 关键参考

- SafeTensors: HuggingFace → mmap → 安全 → 生产首选
- nvidia-container-toolkit: Docker→GPU → K8s → nvidia-docker
- vLLM镜像: vllm/vllm-openai → 一切内置 → 简单
- K8s Deployment: rolling → 0 downtime → rollback → 1 command
- sleep/wake: vLLM V1 → 1s → 比 HPA 30s → 30x快
- HF-Mirror: hf-mirror.com → 中国镜像 → RTX 4090推荐