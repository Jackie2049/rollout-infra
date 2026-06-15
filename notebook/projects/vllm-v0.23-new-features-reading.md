# vLLM v0.23.0 新特性源码级阅读

> 日期: 2026-06-15 | 版本: v0.23.0 (开发中, 最新release为v0.22.1 2026-06-05)
> 聚焦: MRv2默认化 / INT4 Triton fallback / DeepSeek-V4 MLA+EPLB / Rust前端 / 多层KV卸载 / Breakable CUDA Graph

---

## 1. MRv2 默认化: Llama / Mistral / Qwen3 (及 MoE 扩展)

### 1.1 Oracle 系统架构

MRv2 的默认启用采用 **oracle 机制** — 不是一刀切, 而是根据模型架构和特性逐个判定:

```python
# vllm/config/vllm.py
DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen3ForCausalLM",
    "DeepseekV2ForCausalLM",   # PR#42667 新增
    "Qwen2MoeForCausalLM",     # PR#42667 新增
})
```

**三态环境变量** `VLLM_USE_V2_MODEL_RUNNER`:
- `None` (未设置): 使用 oracle 自动判定
- `"1"`: 强制 MRv2
- `"0"`: 强制 MRv1

### 1.2 Oracle 判定逻辑 (源码路径: `vllm/config/vllm.py`)

```python
@property
def use_v2_model_runner(self) -> bool:
    # 1. 环境变量优先
    use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:
        return use_v2_model_runner

    # 2. Oracle 判定: 架构+特性
    if not self._is_default_v2_model_runner_model():
        return False

    # 3. Triton 必须可用 (MRv2依赖Triton kernel)
    if not HAS_TRITON:
        logger.warning_once("MRv2 requires Triton; using v1 runner instead.")
        return False

    # 4. 检查不支持特性列表
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        logger.warning_once("MRv2 does not yet support %s", ", ".join(unsupported))
        return False

    return True
```

`_is_default_v2_model_runner_model()` 判定规则(经 PR#42667 修改):
- `runner_type != "generate"` → False (pooling/encoder模型不走MRv2)
- `is_quantized == True` → False (**量化模型仍用v1**)
- `architectures` 必须包含 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` 中的至少一个
- **MoE 不再是排除条件** (PR#42667移除了`is_moe`检查)

### 1.3 不支持特性黑名单 (`_get_v2_model_runner_unsupported_features`)

- KV connector (NIXL/Mooncake等)
- Dual batch overlap (dbo)
- Elastic expert parallelism (elastic_ep)
- Routed experts capture
- Pipeline parallelism > 1 (部分)
- LoRA + CUDA graph (部分场景)

### 1.4 PR 演进时间线

| PR | 合并日期 | 内容 |
|---|---|---|
| #39337 | 2026-05-14 | Oracle 1/N: Qwen3ForCausalLM dense only, is_moe=False + is_quantized=False |
| #43458 | 2026-06-02 | 扩展到 LlamaForCausalLM + MistralForCausalLM |
| #42667 | 2026-06-12 | 扩展到 MoE: DeepseekV2ForCausalLM + Qwen2MoeForCausalLM, 移除is_moe限制 |
| #43233 | 2026-05-23 | CI测试强制v1 runner (避免MRv2干扰测试) |
| #45461 (open) | 2026-06-12 | GraniteMoeForCausalLM 加入默认列表 (进行中) |

### 1.5 迁移路径

- Dense BF16 模型 (Llama-3.x, Mistral-7B, Qwen3): **自动启用MRv2**
- MoE dense 模型 (DS-V2-Lite, Qwen2-MoE): **自动启用MRv2**
- **量化模型** (INT4, FP8, GPTQ, AWQ): **仍用MRv1** — oracle排除
- VLLM_USE_V2_MODEL_RUNNER=0 可回退到v1

### 1.6 RTX 4090 影响

- **INT4模型→MRv1** (oracle判定is_quantized=True→False)
- **BF16 dense模型→MRv2** (如Qwen3-1.7B BF16)
- MRv2依赖Triton → RTX 4090 Triton可用(SM89)
- 关键: MRv2 **不改变推理kernel**, 只是model runner架构更模块化 → 推理性能本身应该相同或略好

---

## 2. INT4 Triton Fallback (PR #43731)

### 2.1 问题背景

某些 W4A16 compressed-tensors 模型的线性层维度不能被 Marlin 的 `GPTQ_MARLIN_MIN_THREAD_K=128` 整除, 例如:
- `intermediate_size=2112` (2112 % 128 ≠ 0)
- `moe_intermediate_size=704` (704 % 128 ≠ 0)

在 Ampere (SM80) 上, 所有可用 W4A16 kernel 都拒绝这些shape:
- CutlassW4A8/Machete: 需要 SM90 (Hopper)
- Marlin: 需要 `input_size_per_partition % 128 == 0`
- AllSpark: 只支持 `group_size=-1`
- Conch: 只支持 `group_size ∈ [-1, 128]`
- Exllama: 只支持 float16 (不支持 bf16)

之前的结果: `ValueError: Failed to find a kernel that can implement the WNA16 linear layer`

### 2.2 解决方案 (源码: `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`)

```python
# 修改前:
def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
    if not current_platform.is_rocm():
        return False, "TritonW4A16LinearKernel only targets ROCm"

# 修改后:
def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
    if not (current_platform.is_rocm() or current_platform.is_cuda()):
        return False, "TritonW4A16LinearKernel requires CUDA or ROCm"
```

### 2.3 Kernel 优先级列表 (源码: `vllm/model_executor/kernels/linear/__init__.py`)

CUDA W4A16 kernel 优先级 (从高到低):
1. `MacheteLinearKernel` (SM90 Hopper)
2. `CutlassW4A8LinearKernel` (SM90)
3. `MarlinLinearKernel` (SM80+, 需要128对齐)
4. `ConchLinearKernel`
5. `ExllamaLinearKernel`
6. **`TritonW4A16LinearKernel`** ← 新增, 最低优先级 fallback

关键: **零影响已存在模型** — 高优先级kernel仍先被选择, Triton只在所有其他kernel拒绝shape时激活。

### 2.4 TritonW4A16LinearKernel 特性

- 纯 Triton 实现, 无平台特定操作
- 只要求 `N % 8 == 0` (宽松shape限制)
- JIT编译 → 比Marlin慢(无预编译优化), 但是fallback而非首选
- 支持 bf16 激活值 (Exllama不支持)

### 2.5 RTX 4090 影响 ★★★

**这是RTX 4090最有价值的v0.23特性之一:**

| 场景 | v0.22 | v0.23 |
|---|---|---|
| W4A16 + intermediate_size=128对齐 | Marlin kernel ✅ | Marlin kernel ✅ (无变化) |
| W4A16 + intermediate_size≠128对齐 | ValueError ❌ | Triton fallback ✅ |
| INT4 MoE模型 + 非对齐维度 | 无法加载 ❌ | Triton fallback ✅ 可运行 |

- **RTX 4090 SM89 = Ampere+** → TritonW4A16 kernel 可用
- **更多INT4模型类型可用** — 之前因shape不对齐而无法加载的模型现在可以运行
- 性能影响: Triton kernel 比Marlin慢(约2-5x), 但:
  - 只用于不对齐的层 → 整体模型大部分层仍用Marlin
  - 从"完全无法运行"→"可以运行但略慢" → 净影响正

### 2.6 性能参考

- 31B W4A16 dense 模型测试: 29/29 测试通过, Marlin-aligned层仍用MarlinLinearKernel
- Triton fallback 层延迟: 比Marlin慢但可接受 (JIT编译 overhead ~1-2s 首次)

---

## 3. DeepSeek-V4 MLA + EPLB

### 3.1 DeepSeek-V4 集成概览

DeepSeek-V4 是一个巨大的模型支持工程, 涉及30+个PR, 3个硬件平台(NVIDIA/AMD/XPU):

| PR | 合并日期 | 内容 |
|---|---|---|
| #40860 | 2026-04-27 | DeepSeek-V4 初始集成 (rebased) |
| #43039 | 2026-05-19 | 模型重构: DS-V4 layers → models/deepseek_v4/ [2/N] |
| #43073 | 2026-05-19 | 模型重构: DS-V4 ops → models/deepseek_v4/ [3/N] |
| #43149 | 2026-05-22 | 提取 sparse MLA impl 到 model folder |
| #43339 | 2026-06-02 | **EPLB for DeepSeek-V4 Mega MoE** |
| #43746 | 2026-05-28 | 移除torch compile依赖 → BCG替代 |
| #44914 | 2026-06-09 | Fix DS-V4 OOM issue |
| #44821 | 2026-06-10 | Fix DS-V4 MTP projections |
| #43447 | 2026-06-04 | Prefix caching for DS-V4 sliding-window KV |
| #44454 | 2026-06-07 | KV-Cache Layout Refactor for DS-V4 |

### 3.2 MLA (Multi-head Latent Attention) 在 vLLM 的实现

#### 目录结构 (源码路径: `vllm/models/deepseek_v4/`)

```
deepseek_v4/
  ├── attention.py           # DeepseekV4MLAAttention 主类
  ├── sparse_mla.py          # Sparse MLA 模块
  ├── compressor.py          # KV压缩器
  ├── quant_config.py        # 量化配置
  ├── common/ops/            # 共享fused ops
  │   ├── fused_compress_quant_cache.py
  │   ├── fused_indexer_q.py
  │   ├── fused_inv_rope_fp8_quant.py   # inverse RoPE + FP8量化融合
  │   ├── fused_qk_rmsnorm.py           # Q/K RMSNorm融合
  │   ├── fused_mtp_input_rmsnorm.py    # MTP专用融合
  │   └── save_partial_states.py
  ├── nvidia/
  │   ├── model.py           # NVIDIA主模型
  │   ├── mtp.py             # NVIDIA MTP
  │   ├── flashmla.py        # FlashMLA sparse backend
  │   ├── flashinfer_sparse.py  # FlashInfer sparse backend
  │   └── ops/
  │       ├── prepare_megamoe.py      # Mega MoE input preparation
  │       ├── sparse_attn_compress_cutedsl.py  # CuteDSL sparse attn
  │       ├── fused_indexer_q_cutedsl.py       # CuteDSL indexer
  │       ├── dequant_gather_k_cutedsl.py      # CuteDSL dequant+gather
  │       └── o_proj.py                       # Output projection
  ├── amd/
  │   ├── model.py           # AMD ROCm模型
  │   ├── mtp.py             # AMD MTP
  │   ├── rocm.py            # ROCm sparse impl
  │   └── ...
  ├── xpu/
  │   ├── model.py           # Intel XPU模型
  │   ├── xpu_sparse.py      # XPU sparse impl
  │   └── ...
```

#### MLA 平台选择机制

```python
# vllm/models/deepseek_v4/attention.py
class DeepseekV4MLAAttention(nn.Module):
    def __init__(self, ...):
        # 平台判定 → 选CUDA或ROCm impl
        self._select_v4_sparse_impl()

    def _select_v4_sparse_impl(self):
        # CUDA → FlashMLASparse 或 FlashInferMLASparse
        # ROCm → ROCmAiterMLASparse
        # forward() 和 get_attn_backend() 平台无关
```

#### MLA Sparse Decode Backend 选项

| Backend | KV dtype | head_dim | 平台 | 特性 |
|---|---|---|---|---|
| FLASHMLA_SPARSE | bf16, fp8_ds_mla | 576 | NVIDIA SM80+ | FlashMLA optimized |
| FLASHINFER_MLA_SPARSE | bf16, fp8 | 576 | FlashInfer 10.x | 通用 |
| ROCM_AITER_MLA_SPARSE | bf16, fp8 | 1, 64 | AMD ROCm | AMD专用 |

### 3.3 EPLB (Expert Parallel Load Balancing)

#### 启用方式

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --enable-eplb \
  --eplb-config '{"use_async": true, "num_redundant_experts": 8}'
```

#### 源码路径: `vllm/distributed/eplb/eplb_utils.py` + `vllm/models/deepseek_v4/nvidia/model.py`

EPLB 核心变更 (PR#43339):
- `model.py` 增加 211 行, 减少 38 行
- DeepseekV4MoE 层新增 EPLB 路由逻辑
- `deep_gemm.py` 新增 mega_moe backend 支持
- `gpu_worker.py` 增加 EPLB 初始化

#### 性能数据 (8xB200)

| 指标 | 无EPLB | 有EPLB | 改善 |
|---|---|---|---|
| Output tok/s | 1,288 | 1,350 | +4.8% |
| Total tok/s | 11,631 | 12,195 | +4.9% |
| Mean TTFT (ms) | 1,694 | 1,520 | -10.3% |
| Mean TPOT (ms) | 96.1 | 91.7 | -4.6% |

GSM8K accuracy: 95.68% (5-shot), GPQA: 90.40% pass@1

### 3.4 DS-V4 关键架构特性

- **Mega MoE**: 256+ routed experts → deep_gemm_mega_moe backend
- **Sparse MLA**: sliding-window sparse attention for decode
- **MTP (Multi-Token Prediction)**: shared head + fused ops
- **Breakable CUDA Graph**: DS-V4 自动启用 BCG (替代torch compile)
- **FP8 KV cache**: kv-cache-dtype fp8 + fp8_ds_mla for MLA

### 3.5 RTX 4090 影响

- DS-V4-Pro 模型太大 (700B+), RTX 4090 24GB 无法运行
- DS-V4-Lite 可能可行 (如果有类似V2-Lite的小版本)
- FlashMLA: SM80+ → RTX 4090 SM89 ✅
- Sparse MLA: 需要 FlashMLA 或 FlashInfer → RTX 4090 ✅
- EPLB: 需要多GPU EP → RTX 4090 PCIe scaling不佳
- **实用性**: RTX 4090不太可能直接serving DS-V4-Pro, 但 MLA kernel和sparse attention架构值得关注(未来小模型可能采用MLA)

---

## 4. Rust Frontend (vllm-rs)

### 4.1 整体架构

```
rust/
  ├── Cargo.toml              # Rust workspace root
  ├── src/
  │   ├── vllm_rs/            # 主binary (axum HTTP server)
  │   │   ├── app.rs          # HTTP路由注册
  │   │   ├── engine.rs       # Engine-core IPC (gRPC/ZMQ)
  │   │   ├── routes.rs       # /v1/completions, /v1/chat/completions
  │   │   ├── state.rs        # AppState (model registry, LoRA tracking)
  │   │   ├── lora.rs         # Dynamic LoRA管理
  │   │   └── ...
  │   ├── chat/               # Chat template处理
  │   │   ├── src/
  │   │   │   ├── renderer/   # Jinja2模板渲染 (deepseek_v4/v32/hf)
  │   │   │   ├── parser/     # Reasoning + Tool parser
  │   │   │   ├── output/     # Structured/reasoning/tool output
  │   │   │   ├── backend/    # HF tokenizer backend
  │   │   │   └── stream.rs   # SSE streaming
  │   │   └── ...
  │   └── bridge/             # PyO3 Python bridge
  │       └── tool_parser.rs  # Rust→Python tool parser bridge
  ├── proto/
  │   └── vllm_grpc.proto     # gRPC protocol定义
  └── ...
```

### 4.2 关键PR时间线

| PR | 合并日期 | 内容 |
|---|---|---|
| #40848 | 2026-05-21 | **RFC + 初始集成** (VLLM_USE_RUST_FRONTEND=1) |
| #43283 | 2026-05-22 | 从 Inferact/vllm-frontend-rs → rust/ 目录 |
| #43469 | 2026-05-28 | Mock engine 基准测试 |
| #43778 | 2026-06-03 | Dynamic LoRA endpoints (/v1/load_lora_adapter) |
| #43779 | 2026-06-01 | Streaming generate endpoint |
| #43942 | 2026-06-03 | /server_info endpoint |
| #44981 | 2026-06-10 | setuptools-rust 统一构建 |
| #44624 | 2026-06-11 | PyO3 Python bridge for tool parsers |
| #45216 | 2026-06-12 | granite4 standalone tool parser |
| #44222 | 2026-06-09 | /tokenize + /detokenize endpoints |
| #44499 | 2026-06-08 | /pause + /resume + /is_paused endpoints |
| #44321 | 2026-06-09 | API key authentication |
| #44391 | 2026-06-05 | include_reasoning=false 支持 |
| #45030 | 2026-06-11 | vllm:lora_requests_info metrics export |
| #44887 | 2026-06-11 | cached_token_count in responses |
| #44856 | 2026-06-08 | Utility call interfaces refactor |

### 4.3 性能数据 (Mock Engine 基准, PR#43469, GB200 DP=4)

| 配置 | Rust前端+mock | Rust前端+真实模型 | Python前端+mock | Python前端+真实模型 |
|---|---|---|---|---|
| (32, 512) | 4,015 req/s, 2.06M tok/s | 663 req/s, 340K tok/s | 807 req/s, 413K tok/s | 525 req/s, 269K tok/s |
| (16, 1024) | 2,079 req/s, 2.13M tok/s | 274 req/s, 280K tok/s | 463 req/s, 474K tok/s | 239 req/s, 245K tok/s |

**关键结论:**
- Rust前端+mock → **2.1M output tok/s** (mimalloc allocator)
- Python前端+mock → **0.4-0.5M output tok/s** (4x差距!)
- 说明前端层开销是瓶颈时, Rust前端比Python快~4x
- 真实模型场景: Rust 340K vs Python 269K → ~26%吞吐提升 (GPU计算瓶颈时差距缩小)

### 4.4 Rust vs Python 前端技术对比

| 方面 | Python FastAPI | Rust vllm-rs |
|---|---|---|
| HTTP框架 | uvicorn + FastAPI | axum + tokio |
| JSON解析 | Pydantic (Python) | serde (Rust, zero-copy) |
| 并发模型 | asyncio (单线程) | tokio (多线程M:N调度) |
| GIL | 有 (Python限制) | 无 |
| 内存分配 | Python allocator | mimalloc (高性能) |
| Chat模板 | Jinja2 (Python) | Jinja2 (Rust minijinja) |
| Detokenization | HuggingFace tokenizer | HuggingFace tokenizer (Rust binding) |
| 构建方式 | pip install | setuptools-rust + cargo |

### 4.5 Rust Frontend 当前状态

- **Experimental** → 需 `VLLM_USE_RUST_FRONTEND=1` 显式启用
- **不是默认前端** → Python FastAPI仍是默认
- **External engine path only** → Rust前端通过gRPC连接engine-core
- **功能覆盖**: completions/chat/models/LoRA/tokenize/pause/resume/streaming ✅
- **缺失**: structured output (部分), batch inference, 某些advanced参数
- **未来计划**: 成为默认前端 (Python前端逐步退役), 更多tool parser → Rust native

### 4.6 RTX 4090 影响

- Rust前端**不依赖GPU特性** → RTX 4090完全可用
- 单请求场景: Rust vs Python差距小 (GPU计算主导)
- 多并发场景: Rust前端有明显优势 (尤其TTFT)
- **实际建议**: RTX 4090推理场景, 如果并发>10, Rust前端值得启用

---

## 5. Multi-Tier KV Offloading (多层KV卸载)

### 5.1 之前: 单层CPU卸载

vLLM V1 已有 CPU offloading (GPU→CPU单一层级):
- `CPUOffloadingManager`: GPU KV blocks → CPU SharedOffloadRegion
- 限制: 只有CPU一级, 无磁盘/网络层级

### 5.2 新架构: Multi-Tier (PR#40020)

#### 类层次 (源码路径: `vllm/v1/kv_offload/tiering/`)

```
SecondaryTierManager (ABC)           # base.py — 二级存储抽象接口
    ├── ExampleSecondaryTier         # example/ — 内存mock (测试用)
    ├── FSSecondaryTierManager       # fs/ — 本地文件系统
    ├── ObjectStoreSecondaryTier     # obj/ — S3对象存储 (NIXL)

CPUPrimaryTierOffloadingManager      # manager.py — 包装CPUOffloadingManager
    → 提供 secondary-facing API:
      prepare_read/complete_read
      prepare_write/complete_write
      → 方向性明确(GPU→CPU→secondary)

TieringOffloadingManager             # manager.py — 多层编排器
    ├── primary_tier (CPU)
    ├── secondary_tiers (FS/ObjStore/Dummy...)
    └── Cascade + Promotion 逻辑

TieringOffloadingSpec                # spec.py — 入口点
    → CPUOffloadingSpec 子类
    → 读取 kv_connector_extra_config["secondary_tiers"]
    → 组装 TieringOffloadingManager
```

#### 数据流

```
Store (写入):
  GPU block → CPU primary (prepare_write + complete_write)
              → Cascade fan-out → all secondary tiers (async)

Load (读取):
  lookup(key) → hit in primary? → return True
              → hit in secondary? → promotion: secondary → primary → GPU
              → in-flight promotion → return None ("retry later")
              → miss everywhere → return False

Eviction:
  primary LRU eviction → secondary tiers independent eviction
  touch() propagates to ALL tiers (keep hot blocks everywhere)
```

#### ref_cnt 保护机制

- `prepare_read()` 增加 ref_cnt → 防止block在async传输期间被evict
- promotion完成后释放 ref_cnt

### 5.3 Secondary Tier 实现

#### FileSystem Tier (PR#41735, `tiering/fs/`)

- 本地磁盘作为二级存储
- `io.py`: 文件读写操作
- `thread_pool.py`: 异步I/O线程池
- 适合: 本地NVMe SSD → 大容量低成本卸载

#### ObjectStore Tier (PR#41968, `tiering/obj/`)

- S3兼容对象存储作为二级存储
- 使用 **NIXL** (NVIDIA Inference Exchange Layer) 进行数据传输
- 配置示例:
```json
{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": "5GB",
        "secondary_tiers": [{
            "type": "obj",
            "bucket": "my-kv-cache-bucket",
            "endpoint_override": "s3.amazonaws.com",
            "access_key": "...",
            "secret_key": "..."
        }]
    }
}
```

### 5.4 Async Batched Lookup (PR#44193)

- `AsyncLookupManager` → 批量异步查询二级存储
- 大幅提升二级存储性能
- 单key → `lookup()` → None (首次, 排队)
- 批量 → `flush()` → 统一提交 → batch_lookup() → 批量返回
- cleanup() 按request_id清理, 共享key保留

### 5.5 启用方式

```bash
# 单层 CPU offloading (已有)
vllm serve model --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": "5GB"}}'

# 多层 offloading (新增)
vllm serve model --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": "5GB",
        "secondary_tiers": [{"type": "fs", "path": "/mnt/nvme/kv_cache"}]
    }
}'
```

### 5.6 RTX 4090 影响

- **FileSystem tier**: RTX 4090有本地NVMe → ✅ 可用
- **ObjectStore tier**: 需NIXL+S3 → ✅ 可用 (无需RDMA)
- **实用性**: RTX 4090 24GB → long context需要CPU/disk offloading → 多层卸载对长上下文场景有价值
- **性能**: disk latency >> GPU latency → 只适合offload不常访问的KV blocks

---

## 6. Breakable CUDA Graph (BCG) 状态

### 6.1 PR 演进

| PR | 合并日期 | 内容 |
|---|---|---|
| #42304 | 2026-05-16 | **Experimental BCG 初始实现** |
| #44050 | 2026-05-30 | **MRv2 BCG 支持** |
| #43746 | 2026-05-28 | DS-V4 移除torch compile → BCG替代 (auto-enable) |

### 6.2 BCG 核心概念 (源码: `vllm/compilation/breakable_cudagraph.py`)

**传统 CUDA Graph**: 整个decode iteration捕获为单一graph → 无法灵活处理条件操作

**Breakable CUDA Graph**: graph可被"打断" → 部分ops退出graph执行eager → 然后继续graph

```python
@eager_break_during_capture  # 装饰器标记: capture期间这些op退出graph
def some_dynamic_op(x):
    ...
```

**关键机制**:
- `BreakableCUDAGraphCapture` — TLS(thread-local state)管理capture上下文
- `eager_break_during_capture` — 装饰器: capture外→正常执行; capture内→打断graph
- `BreakableCUDAGraphCapture._tls.active` — 当前活跃的capture对象

### 6.3 BCG vs torch.compile 关系

**BCG和torch.compile fullgraph互斥!**

PR#43746 的关键修改:
```python
# vllm/config/vllm.py
if envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
    logger.warning_once(
        "VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's "
        "torch.compile based piecewise cudagraph."
    )
    # BCG替代piecewise CG → 不需要torch.compile
```

DS-V4 自动启用BCG:
```python
# vllm/config/vllm.py
if "VLLM_USE_BREAKABLE_CUDAGRAPH" not in os.environ \
    and any(a in ("DeepseekV4ForCausalLM", "DeepSeekV4MTPModel")
            for a in self.model_config.architectures):
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    logger.info_once("Auto-enabling BCG for DeepSeek V4")
```

### 6.4 BCG 目前覆盖范围

- **DS-V4**: 自动启用BCG (移除@support_torch_compile装饰器)
- **DS-V2-Lite**: 可手动启用 (VLLM_USE_BREAKABLE_CUDAGRAPH=1)
- **其他模型**: 需手动启用, **BCG不默认开启**
- @eager_break_during_capture 标记的ops:
  - `mla_attention.py` (MLA层的某些操作)
  - `deepseek_v4_attention.py` (DS-V4 attention)
  - `sparse_attn_indexer.py` (sparse attention indexer)

### 6.5 性能数据

| 模型 | 配置 | 无BCG | 有BCG | 改善 |
|---|---|---|---|---|
| DS-V2-Lite | B300 offline | 51.33 req/s, 59,133 tok/s | 52.54 req/s, 60,529 tok/s | +2.3% |
| Qwen3-30B-A3B | GB200 offline | 70.46 req/s | 71.04 req/s | +0.8% |
| Qwen3-235B | GB200 online, TP=4 | 28.17 req/s, 3,606 tok/s | 30.29 req/s, 3,877 tok/s | **+7.5%** |
| DS-V4 MTP gsm8k | 8xB200 | — | 95.68% accuracy ✅ | — |

### 6.6 BCG + FlashInfer 交互

**FlashInfer + CUDA Graph 模式**:

| 模式 | FlashInfer支持 | BCG状态 |
|---|---|---|
| FULL | ✅ (decode only) | BCG不涉及(无piecewise需求) |
| FULL_DECODE_ONLY | ✅ | BCG不涉及 |
| FULL_AND_PIECEWISE | ✅ (piecewise需要compile或BCG) | **BCG替代compile实现piecewise** |
| NONE | ✅ | BCG不需要 |

**BCG对FlashInfer的意义**:
- 之前: piecewise CG需要torch.compile → compile慢+不稳定
- 现在: BCG允许piecewise CG **不依赖torch.compile** → 更稳定
- FlashInfer FULL_DECODE_ONLY → 不需要piecewise → BCG无关
- FlashInfer FULL_AND_PIECEWISE → BCG提供piecewise路径 **无需compile**

### 6.7 RTX 4090 影响

- BCG: SM89 ✅ (CUDA graph基础功能)
- DS-V4: 太大, RTX 4090无法运行
- DS-V2-Lite: 可手动启用BCG → RTX 4090可以测试
- **FlashInfer on RTX 4090**: FULL_DECODE_ONLY ✅ → BCG不改变这个
- **INT4+BCG**: 量化模型用MRv1 → v1的piecewise CG仍用torch.compile → BCG不直接影响INT4推理
- **实际建议**: RTX 4090上BCG意义有限, 因为主要受益者是DS-V4等大模型

---

## 7. 其他v0.23重要变更

### 7.1 vLLM版本追踪 (2026-06 第3周)

| 版本 | 日期 | 关键变化 |
|---|---|---|
| v0.22.0 | 2026-05-29 | — |
| v0.22.1 | 2026-06-05 | bugfix |
| v0.23.0 | 开发中 | MRv2 default + INT4 Triton + DS-V4 + Rust + Multi-tier KV + BCG |

### 7.2 重要PR补充

- PR #43447: DS-V4 prefix caching for sliding-window KV
- PR #44454: KV-Cache Layout Refactor for DS-V4 (contiguous per-block allocations)
- PR #44914: DS-V4 OOM fix
- PR #44821: DS-V4 MTP projections fix
- PR #42209: NVFP4 MoE support for DS-V4
- PR #45216: granite4 standalone Rust tool parser
- PR #44981: setuptools-rust 统一构建 (Rust artifact管理)

### 7.3 MRv2 架构演进路线图 (Issue #41286)

```
[1/N] PR#39337: Qwen3 dense oracle ✅
[2/N] PR#43458: Llama + Mistral dense ✅
[3/N] PR#42667: MoE models (DS-V2, Qwen2-MoE) ✅
[4/N] PR#45461: GraniteMoE (open) 🔄
[...] 更多架构逐步加入
最终目标: 所有generate模型都默认MRv2
```

---

## 8. RTX 4090 综合影响总结

| 特性 | RTX 4090可行性 | 实际影响 |
|---|---|---|
| **MRv2 default** | ✅ (dense BF16模型) | Llama/Qwen3 dense自动MRv2; INT4仍MRv1 |
| **INT4 Triton fallback** | ✅✅ **最重要** | 更多INT4模型类型可运行! 之前shape不对齐→ValueError, 现在→Triton fallback |
| **DS-V4 MLA+EPLB** | ❌ (模型太大) | MLA kernel架构值得关注, 但无法直接serve DS-V4-Pro |
| **Rust frontend** | ✅ (纯软件) | 高并发场景~4x前端吞吐提升; 单请求差距小 |
| **Multi-tier KV offloading** | ✅ (FS tier) | 长上下文offload到本地磁盘有价值 |
| **Breakable CUDA graph** | ✅ (SM89) | DS-V2-Lite可测试; 对RTX 4090 INT4推理影响有限 |

**★★★ RTX 4090 最有价值特性**: INT4 Triton fallback → 从"无法加载"到"可以运行" → 开启更多INT4 MoE模型可能性

---

## 关键源码路径索引

| 特性 | 源码路径 |
|---|---|
| MRv2 Oracle | `vllm/config/vllm.py` (DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES + use_v2_model_runner) |
| MRv2 Env | `vllm/envs.py` (VLLM_USE_V2_MODEL_RUNNER) |
| INT4 Triton | `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py` + `__init__.py` |
| DS-V4 MLA | `vllm/models/deepseek_v4/attention.py` + `sparse_mla.py` + `compressor.py` |
| DS-V4 NVIDIA | `vllm/models/deepseek_v4/nvidia/model.py` + `flashmla.py` + `flashinfer_sparse.py` |
| DS-V4 Common ops | `vllm/models/deepseek_v4/common/ops/` (fused ops) |
| EPLB | `vllm/distributed/eplb/eplb_utils.py` + `vllm/models/deepseek_v4/nvidia/model.py` |
| Rust Frontend | `rust/src/` (axum HTTP server + chat + bridge) |
| Multi-tier KV | `vllm/v1/kv_offload/tiering/` (manager.py + spec.py + base.py + fs/ + obj/) |
| Async Lookup | `vllm/v1/kv_offload/tiering/async_lookup.py` |
| BCG | `vllm/compilation/breakable_cudagraph.py` |
| BCG Env | `vllm/envs.py` (VLLM_USE_BREAKABLE_CUDAGRAPH) |
| BCG DS-V4 auto | `vllm/config/vllm.py` (DS-V4 architecture check → auto-set env) |
