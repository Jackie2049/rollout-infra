# vLLM 开源贡献准备 — 源码定位

> 针对已识别的 good first issues，深入定位相关代码

## Issue #41230: NIXL KV Connector Metrics 文档

### 相关文件

| 文件 | 作用 |
|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py` | NIXL 指标定义 |
| `docs/features/nixl_connector_usage.md` | 已有使用文档（缺少 metrics 部分）|

### 指标列表

```
NixlPromMetrics (继承自 KVConnectorPromMetrics):
  - vllm:nixl_xfer_time_seconds     — 传输耗时直方图 (1ms-5s)
  - vllm:nixl_post_time_seconds     — post 耗时直方图
  - vllm:nixl_bytes_transferred     — 传输字节数直方图 (2KB-16GB)
  - vllm:nixl_num_descriptors       — 描述符数量直方图 (10-50K)
  - vllm:nixl_num_failed_transfers  — 失败传输计数器
  - vllm:nixl_num_failed_notifications — 失败通知计数器
  - vllm:nixl_num_kv_expired_reqs   — 过期 KV 请求计数器
```

### 贡献计划

```
1. ✅ 已阅读 stats.py 完整代码 (NixlKVConnectorStats + NixlPromMetrics)
2. ✅ 已阅读 nixl_connector_usage.md (现有文档缺少 metrics 章节)
3. 在 nixl_connector_usage.md 末尾添加 Metrics 章节:
   - 每个指标的 Prometheus 名称、类型、含义
   - 典型值范围和查询示例
   - CLI 日志输出说明 (reduce() 方法的输出)
4. 提交 [Doc] PR

代码分析:
  - NixlKVConnectorStats: 内部数据收集器
    - record_transfer(): 记录成功的传输 (duration, bytes, descriptors)
    - record_failed_transfer(): 记录失败传输
    - record_failed_notification(): 记录失败通知
    - record_kv_expired_req(): 记录过期 KV 请求
    - reduce(): 聚合统计 (avg, p90, throughput)

  - NixlPromMetrics: Prometheus 指标暴露器
    - 4 个 Histogram: xfer_time, post_time, bytes_transferred, num_descriptors
    - 3 个 Counter: failed_transfers, failed_notifications, kv_expired_reqs
    - observe(): 从 stats 数据推送到 Prometheus

  - 注意: stats 是在 worker 进程收集, 然后序列化传到 logger 进程
    (KVConnectorStats 必须可序列化)
```

## Issue #31414: 统一 FlashInfer 工具模块

### 当前状况

```
FlashInfer 在 194 个文件中被引用:
  - 直接导入: from flashinfer import ...
  - 包装导入: from vllm.utils.flashinfer import ...
  - 后端实现: vllm/v1/attention/backends/flashinfer.py

问题:
  - 导入风格不统一
  - FlashInfer API 变更时需要修改多处
  - 缺少统一的兼容层
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `vllm/utils/flashinfer.py` | 兼容性包装器 |
| `vllm/v1/attention/backends/flashinfer.py` | V1 注意力后端 |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_*.py` | MoE 专家层 |

### 贡献计划

```
1. 梳理所有 flashinfer 导入点
2. 统一到 vllm/utils/flashinfer.py 兼容层
3. 确保所有模块通过统一入口导入
4. 提交 [Refactor] PR
```

### 深度源码分析 (Session 3 新增)

```
FlashInfer 后端架构 (vllm/v1/attention/backends/flashinfer.py):

三层类结构:
  FlashInferBackend (L326-437)
    → 静态配置: GPU 能力检测 (SM 7.5~12.1), KV Cache shape 计算
    → 工厂方法: get_impl_cls(), get_metadata_builder_cls()

  FlashInferMetadataBuilder (L539-1265)
    → 动态 metadata 构建, prefill/decode 分路
    → CUDA graph 捕获支持
    → 关键方法: build(), _build_native_prefill(), _build_native_decode()

  FlashInferImpl (L1267-1864)
    → 注意力计算实现
    → forward(): 主入口, 调度到 prefill/decode/append

操作分派机制:
  Prefill:
    Native FI: BatchPrefillWithPagedKVCacheWrapper (L1135-1196)
    TRTLLM:    trtllm_batch_context_with_kv_cache (L1102-1134)
    决策:      use_trtllm_attention() 根据 SM/head 数/数据类型判断

  Decode:
    Native FI: BatchDecodeWithPagedKVCacheWrapper (L1210-1254)
    TRTLLM:    trtllm_batch_decode_with_kv_cache (L1200-1209)
    支持 CUDA graph 固定 batch size

  Cascade Attention:
    MultiLevelCascadeAttentionWrapper (L1045-1090)
    处理共享前缀的请求

兼容性包装器 (vllm/utils/flashinfer.py):
  - _lazy_import_wrapper(): 延迟导入 + 特性检测
  - Feature Detection: 检查 FlashInfer 包、nvcc、cubin
  - TRTLLM 支持: SM100 检测、head count 约束
  - MoE 后端检测: 多种 MoE 实现

重构建议 (针对 #31414):
  1. 创建 FlashInferOperations 类集中操作
  2. 提取 AttentionStrategy 策略模式
  3. 创建 FlashInferQuantizationManager 统一量化
  4. 创建 FlashInferWrapperFactory 统一 wrapper 生命周期
  5. 创建 FlashInferConfig 集中配置管理
```

### 后端注册机制

```
vllm/v1/attention/backends/registry.py:
  FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
  支持运行时覆盖: register_backend() 函数

选择逻辑:
  1. 用户指定 --attention-backend flashinfer
  2. 或自动选择 (基于 GPU capability)
  3. FlashInfer 支持: SM 7.5 ~ 12.1
```

## Issue #38734/#38425/#38389: Transformers v5 兼容

### 模型位置

| 模型 | 注册名 | 文件 |
|------|--------|------|
| SarvamMLA | "SarvamMLAForCausalLM" | `models/sarvam.py` |
| InternVL2 | "InternVLChatModel" | `models/internvl.py` |
| Isaac | "IsaacForConditionalGeneration" | `models/isaac.py` |

### 版本处理

```
版本检查: vllm/transformers_utils/config.py (lines 70-75)
自定义配置: vllm/transformers_utils/configs/isaac.py 等

问题:
  - 模型使用 transformers v4 API
  - v5 中部分 API 变更
  - 需要逐个适配
```

### 贡献计划

```
1. 阅读各模型的 transformers 依赖点
2. 对比 transformers v4/v5 API 差异
3. 添加版本判断兼容代码
4. 提交 [Model] PR
```

## 下一步

- [ ] GPU 到位后拉取最新 vLLM 代码
- [ ] 先做 #41230 (最简单，纯文档)
- [ ] 然后做 #31414 (重构，需要运行测试)
- [ ] 最后做 Transformers v5 兼容 (需要 GPU 测试)
