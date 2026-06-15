# DeepSpeed 0.19.x 六大新特性深度解读

> 2026-06-15 | 源码级深度分析 | 基于 GitHub deepspeedai/DeepSpeed PR源码+release notes
> 版本: v0.19.0 (2026-05-06) + v0.19.1 (2026-05-27)

---

## 目录

1. [AutoEP — 自动Expert Parallelism (PR#7938)](#1-autoep)
2. [ZenFlow — CPU optimizer overlap pipeline](#2-zenflow)
3. [DeepCompile — torch.compile + ZeRO-3](#3-deepcompile)
4. [Muon Optimizer — Newton-Schulz orthogonalization](#4-muon)
5. [AutoSP — 自动Sequence Parallelism (ViT+LLM)](#5-autosp)
6. [ZeRO-3 Defragment — 内存碎片整理](#6-defragment)
7. [补充: SDMA AllGather via Mori + Dynamic/Static Offload混用](#7-supplementary)
8. [RTX 4090 实战分析](#8-rtx4090)

---

## 1. AutoEP — 自动Expert Parallelism <a id="1-autoep"></a>

**PR#7938, merged 2026-06-11 | 作者: @tohtana (DeepSpeed team) + @delock + @PKUWZP**

### 1.1 核心思想

AutoEP = **Automatic Expert Parallelism** — 在 `deepspeed.initialize()` 时自动检测HuggingFace MoE模型的MoE block, 建立EP/EDP process group, 并用EP-enabled的执行路径替换原始MoE block. **用户只需JSON config → 无需修改模型代码!**

### 1.2 五个 Preset 模型 (v0.19.0 = 4个, 后续+LLaMA-4 = 5个)

PR#7938初始包含4个preset, 评论中@tohtana确认第5个LLaMA-4 preset也已加入:

| Preset | expert_storage | score_func | score_apply | route_norm | has_shared_experts | group routing | HF model_type |
|--------|---------------|------------|-------------|------------|--------------------|---------------|---------------|
| **Mixtral** | fused_3d | softmax | post | True | False | N/A | `mixtral` |
| **Qwen3-MoE** | fused_3d | softmax | post | True | True (shared_expert) | N/A | `qwen3_moe`, `qwen2_moe` |
| **Qwen3.5-MoE** | fused_3d | softmax | post | True | True (shared_expert+gate) | N/A | `qwen3_5_moe_text` |
| **DeepSeek-V2** | fused_3d | softmax | post | False | True (shared_experts) | group_limited_greedy → n_group/topk_group | `deepseek_v2` |
| **DeepSeek-V3** | fused_3d/module_list(自适应) | sigmoid | post | False | True (shared_experts) | n_group/topk_group/routed_scaling_factor | `deepseek_v3` |

**Preset定义关键字段** (源码: `deepspeed/module_inject/auto_ep_presets/base.py` → `MoEModelPreset` dataclass):

```python
@dataclass
class MoEModelPreset:
    moe_layer_pattern: str          # 正则匹配MoE层 (e.g. r"model\.layers\.\d+\.mlp")
    router_pattern: str             # 路由器子模块名 (e.g. "gate")
    experts_pattern: str            # 专家子模块名 (e.g. "experts")
    expert_storage: str             # "fused_3d" 或 "module_list"
    expert_w1: str                  # gate+up projection名 (e.g. "gate_up_proj")
    expert_w2: str                  # down projection名 (e.g. "down_proj")
    expert_w3: str | None           # 独立up projection名 (DeepSeek-V3=分离→"up_proj")
    num_experts_attr: str           # config属性名 (e.g. "n_routed_experts")
    top_k_attr: str                 # config属性名 (e.g. "num_experts_per_tok")
    score_func: str                 # "softmax" 或 "sigmoid"
    score_apply: str                # "pre" 或 "post"
    route_norm: bool                # 是否normalize routing score
    has_shared_experts: bool        # 是否有shared expert
    hf_model_types: tuple[str,...]  # HF model_type匹配
```

**DeepSeek-V3 preset的特殊处理**:
- `expert_storage`自适应: 默认`fused_3d`, 但`DeepSeekV3PresetAdapter.resolve_expert_layout()`会检测HF模型是否有分离的`gate_proj/up_proj/down_proj`, 若有则切换为`module_list`并设`expert_w1="gate_proj", expert_w2="down_proj", expert_w3="up_proj"`
- `score_func="sigmoid"` → DeepSeek-V3/V4用sigmoid routing
- `routed_scaling_factor` → 从HF config的`routed_scaling_factor`属性读取
- `load_balance_coeff=None` → 不支持auxiliary-loss-free load balancing (当前版本)

### 1.3 运行时执行路径 (7步)

源码: `deepspeed/module_inject/auto_ep_layer.py` → `AutoEPMoELayer`

```
1. detect & replace → deepspeed.initialize()扫描模型 → 正则匹配MoE层 → 构建MoELayerSpec → 替换为AutoEPMoELayer
2. route → EP router (ep_router.py ← TorchTitan TokenChoiceTopKRouter) → top-k routing → token assignment
3. reorder → TokenReorderer (ep_kernels.py ← TorchTitan) → Triton fill-indices → permutation indices
4. dispatch → all-to-all dispatch across EP group (when autoep_size > 1) → token发到对应expert所在的GPU
5. compute → local grouped expert compute → grouped-GEMM → 单次kernel launch!
6. combine → all-to-all combine → restore original token order
7. merge → shared experts merge (if model has them) → DeepSeek-V3/V2 shared expert + output
```

### 1.4 grouped-GEMM + Triton fill-indices → 零代码修改的秘诀

**grouped-GEMM** (源码: `deepspeed/moe/ep_experts.py` ← TorchTitan `GroupedExperts`):
- 传统MoE执行: 逐expert分别launch GEMM → N次kernel launch → 大量GPU空闲
- grouped-GEMM: 将所有local expert的GEMM合并为**一次grouped GEMM kernel launch** → 大幅减少kernel launch overhead + 更好的GPU利用率
- 需要token按expert分组并alignment → `TOKEN_GROUP_ALIGN_SIZE_M = 8` (bf16)

**Triton fill-indices** (源码: `deepspeed/moe/ep_kernels.py` ← TorchTitan):
- 核心kernel: `_fill_indices_kernel` → Triton JIT
- 输入: `tokens_per_expert_group`, `start_index_values`, `write_offsets`
- 输出: permutation indices → 指示每个token应该去哪个expert位置
- 支持`experts_per_rank`和`num_ranks`作为constexpr → Triton编译优化
- **纯PyTorch fallback**: 若Triton不可用 (如RTX 4090某些配置), 有CPU fallback实现
- token group alignment: `_round_up(x, TOKEN_GROUP_ALIGN_SIZE_M)` → 保证grouped-GEMM的M维度alignment

**为什么能零代码修改**:
- AutoEP在`deepspeed.initialize()`阶段hook → 扫描模型结构 → 自动匹配preset → 用`AutoEPMoELayer`替换原始HF MoE block
- Weight repacking (源码: `deepspeed/moe/ep_repack.py`) → 将HF expert权重布局转换为grouped expert tensor布局 → 一次性完成
- 所有这些发生在initialize → 训练开始前 → 用户模型代码不需要任何修改!

### 1.5 AutoEP架构分层

源码文件路径:

| 层 | 源码路径 | 作用 |
|----|---------|------|
| Config解析 | `deepspeed/module_inject/auto_ep_config.py` | 解析ds_config JSON → AutoEPConfig |
| Preset注册 | `deepspeed/module_inject/auto_ep_presets/registry.py` | PRESET_MODELS字典 + adapter注册 |
| Preset定义 | `deepspeed/module_inject/auto_ep_presets/{mixtral,qwen3_moe,qwen3_5_moe,deepseek_v2,deepseek_v3}.py` | 5个preset的MoEModelPreset |
| 基础dataclass | `deepspeed/module_inject/auto_ep_presets/base.py` | MoEModelPreset/MoELayerSpec/AutoEPConfig/AutoEPPresetAdapter |
| MoE检测+替换 | `deepspeed/module_inject/auto_ep.py` | 扫描模型 → 检测MoE block → 验证 → 构建MoELayerSpec → 替换为AutoEPMoELayer |
| 执行wrapper | `deepspeed/module_inject/auto_ep_layer.py` | AutoEPMoELayer: route→reorder→dispatch→compute→combine→merge |
| EP Router | `deepspeed/moe/ep_router.py` ← TorchTitan | TokenChoiceTopKRouter: top-k routing |
| Grouped Experts | `deepspeed/moe/ep_experts.py` ← TorchTitan | GroupedExperts: grouped-GEMM执行 |
| EP Kernels | `deepspeed/moe/ep_kernels.py` ← TorchTitan | TokenReorderer + Triton fill-indices + alignment |
| Weight Repack | `deepspeed/moe/ep_repack.py` | HF expert weights → grouped expert tensor layout |
| Engine集成 | `deepspeed/runtime/engine.py` | deepspeed.initialize() → AutoEP hook |
| Checkpoint | `deepspeed/checkpoint/autoep_universal.py` | universal checkpoint save/load |
| Process Group | `deepspeed/utils/groups.py` | EP/EDP process group setup |

### 1.6 Benchmark数据 (PR#7938评论, @tohtana发布)

单节点测试 (8 GPU, 小模型减层):

| Model | ZeRO-3 leaf | AutoEP (+ZeRO-1) | throughput | memory |
|-------|------------|-----------------|------------|---------|
| Qwen3.5 MoE (8 layers) | 42,128 tok/s, 34.99 GB | 87,540 tok/s, 25.58 GB | **2.08x** | **0.73x** |
| Llama4 (7 layers) | 19,144 tok/s, 56.95 GB | 60,178 tok/s, 60.08 GB | **3.14x** | 1.06x |
| Mixtral 8x7B (8 layers) | 32,622 tok/s, 50.47 GB | 69,052 tok/s, 35.03 GB | **2.12x** | **0.69x** |

**AutoEP vs ZeRO-3 leaf**: 2-3x throughput提升! 原因: ZeRO-3 leaf模式每层都要AllGather全部参数 → MoE模型参数量巨大 → 通信开销巨大; AutoEP用EP让每个GPU只持有一部分expert → 大幅减少通信

### 1.7 AutoEP限制

- **不支持AutoTP**: `tp_size > 1` → ValueError → AutoEP+AutoTP是follow-up
- **ZeRO-3扩展是follow-up**: 当前只支持ZeRO-0/1/2 → PR#7928 (ZeRO-3)需先merge
- **load_balance_coeff不支持**: auxiliary-loss-free load balancing暂不可用
- **DeepSeek-V3 regex需检查**: @nathon-lee测试发现DeepSeek-V3-Base → "no MoE layers detected" → regex可能需调整
- **requires_grad同步**: @delock发现freezed expert在AutoEP替换后不再freeze → @tohtana已修复(复制requires_grad/dtype/device)
- **Muon兼容性**: @delock的`gma/autoep-muon-fixes`分支已merge → 5个commit修复noaux_tc/Muon兼容

---

## 2. ZenFlow — CPU optimizer overlap pipeline <a id="2-zenflow"></a>

**首次引入: PR#7759 → 修复: PR#7771, #8042 → 进化: PR#8058 (OPEN, native CPU optimizer process)**

### 2.1 核心思想

ZenFlow = **选择性optimizer + overlap pipeline** → 不是每步都更新所有参数 → 只更新"重要"参数 → 其余参数延迟更新 → 减少CPU-GPU通信 + overlap CPU optimizer step with GPU forward/backward

### 2.2 ZenFlow配置 (源码: `deepspeed/runtime/zenflow/zenflow_config.py`)

```python
class ZenFlowConfig:
    topk_ratio: float = 0.1            # 保留top-10%重要gradient列
    select_strategy: str = "auto"       # "auto"/"step"/"epoch" → 选择重要参数的策略
    select_interval: str|int = "auto"   # 重选重要参数的间隔
    update_interval: str|int = "auto"   # 延迟更新(非重要参数)的间隔
    overlap_step: bool = False          # ★ 是否overlap CPU optimizer step with GPU computation
    offload: bool = False               # 是否offload selective optimizer states到CPU
    auto_ratio: float = 0.99            # auto策略的threshold
    full_warm_up_rounds: int = 0        # 前N轮全部更新(不做选择)
    pt_reserved_cores_perc: float = 0.5 # PyTorch线程预留核心百分比 → 剩余给ZenFlow optimizer worker
```

### 2.3 4th Layer Overlap — 双buffer机制

**不是"第4层"overlap → 而是chunk granularity overlap**:

ZenFlow的overlap不是per-layer (太细 → overhead大), 也不是per-entire-backward (太粗 → 无overlap), 而是**以chunk为粒度pipeline overlap**:

```
GPU backward chunk N (e.g. layers 1-4) ← → CPU optimizer update chunk N-1
                                    ↓
双buffer交替:
  Buffer A: 当前accumulating GPU gradients (layers 1-4)
  Buffer B: 正在进行CPU optimizer update (previous chunk)
                                    ↓
当chunk N完成 → swap:
  Buffer A → 发到CPU做optimizer step
  Buffer B → 开始accumulate下一个GPU chunk的gradients
```

**overlap_step=True时** → `ZenFlowZeroOptimizerParallel`:
- GPU backward computation → 产生gradient → 双buffer交替 → 一半做CPU optimizer, 一半accumulate新gradient
- CPU optimizer运行在**独立进程/线程** → NUMA-local pinned core → 不干扰GPU computation

### 2.4 选择性更新机制 (Selective Optimizer)

源码: `deepspeed/ops/adam/ZenFlowSelectiveAdamW`

- **topk_ratio=0.1** → 每次只选最重要的10%参数列进行更新
- 重要性判断: gradient magnitude → top-k selection
- 非重要参数 → 累积gradient → 等update_interval步后一次性更新
- 梯度累积 = gradient_accumulation_steps (ZenFlow重用这个概念)

### 2.5 ZenFlow架构分层

源码路径:

| 层 | 源码路径 | 作用 |
|----|---------|------|
| Config | `deepspeed/runtime/zenflow/zenflow_config.py` | ZenFlowConfig pydantic model |
| Z1/Z2 Engine | `deepspeed/runtime/zenflow/zenflow_stage_1_and_2.py` | ZenFlowZeroOptimizer → Sequential/Parallel |
| Z3 Engine | `deepspeed/runtime/zenflow/engine_stage3.py` | ZeRO-3 ZenFlow extension |
| Utils | `deepspeed/runtime/zenflow/zenflow_utils.py` | start_optimizer_process等 |
| Selective Adam | `deepspeed/ops/adam/ZenFlowSelectiveAdamW` | 选择性AdamW optimizer |
| Engine Config | `deepspeed/runtime/zenflow/engine.py` | configure_zenflow() + zenflow_step() |
| Native CPU Adam (OPEN) | `deepspeed/ops/adam/zenflow_cpu_adam.py` (PR#8058) | C++ native CPU optimizer process |

### 2.6 PR#8058 — native CPU optimizer process (当前OPEN)

这是ZenFlow的**重大进化**:

- **之前**: Python `multiprocessing` subprocess + pickling Pipe → 每参数Python↔C++循环 + Python-side `clone()`
- **之后**: native CPU optimizer process in `cpu_adam` op → shared-memory POSIX-semaphore控制 → NUMA-local pinned thread pool
- **Fused multi-tensor CPU Adam** (`adam_update_multi`): C++一次性驱动整个flattened partition → 无per-parameter loop
- **Chunked copyback**: streaming fp32→bf16 → GPU spike从2944 MiB降到256 MiB (0.75B-param partition) → 之前`fp32.to(device)`一次性materialize整个fp32 partition
- **`ZenFlowCPUAdam`**: recognized ZeRO optimizer → 不再需要`zero_allow_untested_optimizer`

---

## 3. DeepCompile — torch.compile + ZeRO-3 <a id="3-deepcompile"></a>

**核心PR: #7951 (Fix DeepCompile+Z3 on PT v2.9/2.10) → 后续: #8024, #8032, #8033, #8036, #8038, #8059**

### 3.1 DeepCompile ≠ ZeRO-3+compile incompatibility

**关键区别**: ZeRO-3 + `torch.compile()` 之所以不兼容, 是因为:
- ZeRO-3在运行时动态AllGather/Release参数 → Dynamo无法捕获静态图 → graph break
- ZeRO-3的`param.data`在released状态是`empty(0)` → Dynamo guard检查shape → "expected 4096, actual 0" → 失败

**DeepCompile如何解决**:
- DeepCompile是DeepSpeed的**专用编译层** → 不是简单wrap `torch.compile()`
- 在Dynamo tracing之前, 先**unset ZeRO-3 hooks** → 去掉runtime AllGather/Release → 让Dynamo看到完整参数
- 用**full-shape dummy tensors**做symbolic tracing → 不触发实际AllGather
- **Override guard metadata** → 对ZeRO-3参数, 将size/stride guard覆盖为**stable released representation** → 不是transient gathered sizes
- 然后在FX graph上运行**DeepSpeed自定义passes** → 插入优化后的gather/release操作

### 3.2 DeepCompile架构

源码: `deepspeed/compile/` 目录

| 文件 | 作用 |
|------|------|
| `config.py` | CompileConfig → deepcompile/free_activation/offload_activation/double_buffer/symmetric_memory等 |
| `init_z3.py` | ZeRO-3 compile初始化 → unset hooks → register_z3_param → launch compile passes |
| `init_z1.py` | ZeRO-1 compile初始化 |
| `init_sp.py` | AutoSP compile初始化 (sequence parallelism) |
| `fx.py` | FX tracing + patch |
| `backend.py` | make_backend + launch_compile_passes + init_schedule |
| `graph_param.py` | DSGraphParamManager → 管理编译后的参数生命周期 |
| `partitioner.py` | 图分区 |
| `list_schedule.py` | 列调度 |
| `input_storage.py` | 编译输入存储 |
| `patch_fake_tensor.py` | FakeTensor patching → 解决ZeRO-3参数shape问题 |
| `patch_compiled_func.py` | 编译函数patching |
| `util.py` | get_deepcompile_handle + add_pre_backward_hook |
| `inductor.py` | Inductor backend集成 |
| `csrc/compile/` | C++扩展 → deepcompile.cpp, z3.h → 底层guard override |

### 3.3 Compiler Passes (源码: `deepspeed/compile/passes/`)

| Pass | 文件 | 作用 |
|------|------|------|
| **zero3_compile** | `zero3_compile.py` | ZeRO-3核心compile pass → add_z3_gather_release → 在FX graph中插入优化的gather/release |
| **selective_gather** | `selective_gather.py` | ★ 选择性gather → 只gather真正需要的参数 → 计算persistence budget → 减少通信 |
| **zero1_compile** | `zero1_compile.py` | ZeRO-1 compile pass |
| **sp_compile** | `sp_compile.py` | AutoSP compile pass → sequence parallelism |
| **prefetch** | `prefetch.py` | 参数prefetch → overlap通信和计算 |
| **offload_parameters** | `offload_parameters.py` | 参数offload → CPU/NVMe offload在编译图中的实现 |
| **offload_activation** | `offload_activation.py` | 激活offload |
| **offload_adam_states** | `offload_adam_states.py` | Optimizer states offload |
| **long_context_checkpointing** | `long_context_checkpointing.py` | 长上下文gradient checkpointing |

### 3.4 selective_gather pass详解

源码: `deepspeed/compile/passes/selective_gather.py`

**核心思想**: 在ZeRO-3中, 每次forward/backward访问参数都要AllGather → 但编译图可以分析哪些参数可以**持久化(persist)**在GPU上, 避免反复gather/release

**算法**:
1. Profile每个FX graph的内存使用 → 记录每个op的resident allocation和transient peak
2. 计算`persistence_budget` = `usable_mem - peak_resident_alloc` → 可以持久化多少参数
3. 对每个ZeRO-3参数, 计算其gather时间(cost) vs 持久化内存(benefit)
4. 选择性地让某些参数persist → 减少该参数的AllGather次数
5. 只在backward的最后一个graph上运行 → 因为forward结束后参数才release

**persistence budget计算**:
```python
_compute_persistence_budget(all_graph_mem_records, total_mem, mem_margin):
    usable_mem = total_mem * (1 - MEM_MARGIN)  # MEM_MARGIN = 0.1
    peak_resident_alloc = max(所有op的resident allocation)
    available_mem = usable_mem - peak_resident_alloc
    # → 这个available_mem就是可以用来persist参数的空间
```

### 3.5 PR#7951修复详解

PyTorch v2.9/2.10的TorchDynamo开始严格执行parameter tensor-match guards → 在DeepCompile tracing时, ZeRO-3参数暂时all-gathered → Dynamo记录完整size (如4096) → 但guard evaluation时, DeepSpeed已release → `param.data = empty(0)` → guard失败 "expected 4096, actual 0"

修复:
1. **Keep full-shape dummy tensors for symbolic tracing** → 不用真实gathered参数
2. **Override guard size/stride metadata** → 对ZeRO-3参数, guard用stable released representation
3. 修复backward graph hoist `end_backward`的问题 (v2.7/2.8)
4. 修复selective_gather persistence memory budget overcount

**当前兼容性**:
- PT v2.9/2.10: 已修复 (PR#7951)
- PT v2.11: PR#8024修复AOT kwargs patching → 但仍有frame-level问题 (PR#8059 OPEN)
- PT v2.7/2.8: backward hoist问题已修复

### 3.6 CompileConfig关键字段

```python
class CompileConfig:
    deepcompile: bool = False      # 开启DeepCompile
    free_activation: bool = False  # 激活释放
    offload_activation: bool = False
    double_buffer: bool = True     # ★ 双buffer → overlap activation
    symmetric_memory: bool = False # Mori symmetric memory (AMD MI300)
    offload_parameters: bool = False
    passes: List[PassName] = None  # 组合passes: "z1"/"z3"/"autosp"
```

---

## 4. Muon Optimizer — Newton-Schulz orthogonalization <a id="4-muon"></a>

**PR#7953 (Gram NS, merged 2026-04-30) + PR#7919 (ZeRO-3 extension, merged 2026-03-26) + PR#7962 (Blog, merged 2026-05-21)**

### 4.1 Muon核心算法

源码: `deepspeed/runtime/zero/muon/original_muon.py`

**Muon = Momentum + Orthogonalization via Newton-Schulz**:
- 对2D权重矩阵: gradient → momentum → Newton-Schulz orthogonalization → 用orthogonalized matrix更新权重
- 对非2D参数 (embedding/LN/bias/lm_head): **回退到AdamW**
- Muon是**混合optimizer**: 一个momentum buffer vs Adam的两个 (m+v) → 省~45% optimizer state memory (对~90%的2D参数)

### 4.2 Newton-Schulz迭代 (标准版)

源码: `zeropower_via_newtonschulz5()` → `@compiler.compile()`装饰

```python
# Quintic iteration (5阶), coefficients maximize slope at zero:
a, b, c = (3.4445, -4.7750, 2.0315)

# X = G (gradient matrix) → normalize → iterate:
X = G.to(bf16)  # 或fp32
X = X / (X.norm() + 1e-7)  # spectral norm ≤ 1
for _ in range(5):
    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X
# → X ≈ U·S'·V^T where S' ≈ Uniform(0.5, 1.5) → 不完全正交但足够好!
```

**数学原理**: Newton-Schulz迭代是计算矩阵polar decomposition的迭代方法 → 收敛到正交矩阵 → 不需要SVD (GPU-friendly, 只有矩阵乘法)

### 4.3 Gram Newton-Schulz (PR#7953新引入, 默认方法)

源码: `zeropower_via_gram_newtonschulz()` → **由@delock和@PKUWZP贡献**

**核心改进**: 标准NS迭代在full rectangular matrix X (n × m)上操作 → 但transformer权重矩阵典型aspect ratio α ≈ 5 (m >> n) → **Gram NS迭代在更小的Gram矩阵 R = X @ X^T (n × n)上操作 → FLOPs大幅减少!**

```python
# Gram Newton-Schulz:
X = G.to(fp16)  # ★ fp16而非bf16 → 更好精度
n, m = X.size(-2), X.size(-1)

# 先normalize
X = X / (X.norm() + 1e-7)

# 对方阵(n == m) → 无FLOP优势 → fallback到标准NS
if m <= n:
    # 标准NS迭代
    ...

# 对m >> n → Gram NS:
R = X @ X.mT  # n×n (而非n×m) → 小得多!
Q = None
restart_at = 2  # ★ 在iteration 2重启 → half-precision稳定性

for i in range(steps):
    if i == restart_at and i != 0:
        X = Q @ X       # 重启: 用累积的Q更新X
        R = X @ X.mT    # 重新计算Gram matrix

    Z = b * R + c * R @ R

    if Q is None:
        Q = Z.clone()
        Q.diagonal().add_(a)    # Q = a*I + b*R + c*R^2
    else:
        Q = a * Q + Z @ Q       # Q = a*Q + Z*Q

    if i < steps-1 and (i+1) != restart_at:
        RZ = a * R + Z @ R      # 更新R
        R = a * RZ + Z @ RZ

# 最终:
X = Q @ X  # 正交化结果
```

**性能提升**: 参考Tri Dao blog (https://tridao.me/blog/2026/gram-newton-schulz/) → Gram NS比标准NS快得多, 特别是m >> n时

### 4.4 ns_method配置

```json
"optimizer": {
    "type": "Muon",
    "params": {
        "ns_method": "gram"     // 默认, 推荐!
        // "ns_method": "standard"  // fallback, 用于debug convergence
    }
}
```

- `ns_method="gram"` → 默认 → Gram Newton-Schulz → 方阵自动fallback到standard
- `ns_method="standard"` → 标准NS → 用于调试收敛问题
- Muon更新: `muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, ns_method="gram", is_expert_group=False)`
- `is_expert_group=True` → MoE expert参数 → AutoEP+Muon兼容 (PR#7938 + gma/autoep-muon-fixes)

### 4.5 ZeRO-3 Muon扩展 (PR#7919)

- ZeRO-3为Muon创建**dedicated momentum buffer** → partitioned like `fp32_partitioned_groups_flat`
- 支持GPU/CPU/NVMe三种optimizer state dispatch → NVMe时momentum buffer随其他optimizer states一起swap in/out
- DeepSpeed config支持`muon_lr`和`adam_lr`分离 → Muon参数和Adam参数不同学习率

### 4.6 Benchmark数据 (Blog PR#7962)

Moonlight-16B-A3B (MoE) finetune, ZeRO-2, bf16, autoep_size=4, 4 GPUs:

| Optimizer | MBPP | MBPP+ | MMLU | GSM8K |
|-----------|------|-------|------|-------|
| baseline | 0.495 | 0.431 | 0.401 | 0.526 |
| AdamW | 0.661 | 0.534 | 0.660 | 0.805 |
| Muon | 0.646 | **0.548** | **0.678** | **0.810** |

Muon在3/4 metrics上优于AdamW → 特别是MBPP+ (generalization)和MMLU

**Memory**: Qwen2.5-3B finetune (8xA100, ZeRO-2):
- AdamW: 34.5 GiB → Muon: 31.4 GiB → **省9%** (3 GiB)

### 4.7 生产级采用

- **Kimi-K2** (1T参数, Moonshot AI) → MuonClip变体
- **GLM-5** (744B参数, Zhipu AI) → Muon + "Muon Split" (MLA up-projection按attention head独立正交化)
- **DeepSeek-V4** (1.6T参数) → Muon optimizer → faster convergence + greater training stability
- **Moonlight-16B-A3B** → Muon pretrained → DeepSpeed finetune验证

---

## 5. AutoSP — 自动Sequence Parallelism (ViT+LLM) <a id="5-autosp"></a>

**PR#7860 (AutoSP base, merged 2026-03-30) + PR#7984 (Multimodal ViT+LLM, merged 2026-05-01)**

### 5.1 AutoSP核心思想

AutoSP = **Compiler-based Sequence Parallelism** → 自动将input沿sequence维度切分 → Ulysses-style SP → 防止`torch.compile()`时的graph break

**不是手动标注哪些op需要SP → 而是编译器pass自动插入通信操作**

### 5.2 AutoSP架构 (PR#7860)

源码: `deepspeed/compile/` 相关文件

| 文件 | 作用 |
|------|------|
| `deepspeed/compile/init_sp.py` | AutoSP compile初始化 |
| `deepspeed/compile/passes/sp_compile.py` | SP compile pass → 在FX graph上自动插入AllGather/ReduceScatter |
| `deepspeed/compile/custom_ops/all_to_all.py` | AllToAll custom op → SP通信 |
| `deepspeed/compile/custom_ops/sp_compat.py` | SP兼容性 |
| `deepspeed/compile/custom_ops/sp_dp_registry.py` | SP/DP registry |

**用户入口**: `prepare_autosp_inputs()` → 显式调用, 准备SP输入

### 5.3 Multimodal ViT+LLM (PR#7984) — 三层设计

源码路径:

| 层 | 源码路径 | 作用 |
|----|---------|------|
| AutoSP Detector | `deepspeed/sequence/autosp_detector.py` | 自动检测ViT encoder和LLM decoder |
| ViT SP | `deepspeed/sequence/autosp_vit.py` | UlyssesSPViTAttention → Gather-Compute-Scatter |
| Cross-Modal Fusion | `deepspeed/sequence/autosp_fusion.py` | ViT→LLM boundary scatter/gather |
| AutoSP Scaffold | `deepspeed/sequence/auto_sp.py` | auto_wrap_model_for_sp() → 自动wrap |

#### Layer 1: AutoSP Scaffolding & Detector

`auto_wrap_model_for_sp()`:
- 扫描multimodal模型 → 自动检测ViT encoder和LLM decoder
- 对LLM decoder attention层 → 自动wrap `DistributedAttention` (DeepSpeed已有)

#### Layer 2: ViT Sequence Parallelism

`UlyssesSPViTAttention` → **Gather-Compute-Scatter** for non-causal ViT attention:
- **Gather**: 每个rank只持有1/N的sequence → AllGather到full sequence
- **Compute**: 在full sequence上计算ViT attention (non-causal → 无mask限制)
- **Scatter**: ReduceScatter回1/N的sequence → 释放显存

**效果**: ViT FFN和LayerNorm的activation memory减少1/N → 但attention仍然计算full matrix (每个rank)

#### Layer 3: Cross-Modal Fusion Adapters

Vision-Language boundary → 复杂的sequence scatter/gather → 确保LLM decoder收到uniformly sharded fused sequences

支持3种架构:
- **LLaVA** (`LlavaFusionAdapter`): visual token splice → 替换image placeholder
- **InternVL** (`InternVLFusionAdapter`): `IMG_CONTEXT` token splice
- **Qwen2-VL** (`Qwen2VLFusionAdapter`): vision_start/end bounded splice

### 5.4 AutoSP SP deny list (PR#7887, @kashif)

之前是allow list → 只标注哪些op需要SP → 现在改为deny list → **默认所有op都SP → 只标注哪些不需要** → 更自动化!

### 5.5 限制

- **Fusion layer手动wrap**: ViT和LLM attention自动wrap → 但vision projection仍需手动`ModalityFusionSPAdapter`
- **ViT SP trade-off**: UlyssesSPViTAttention用Gather-Compute-Scatter → FFN memory省1/N → 但attention仍计算full matrix → 真正的O(N²) reduction需要All-to-All sequence-to-head transposition (未来工作)
- **Padding**: fused_len % world_size != 0 → zero-padding → attention_mask需手动patch

---

## 6. ZeRO-3 Defragment — 内存碎片整理 <a id="6-defragment"></a>

**PR#7940, merged 2026-03-31 | 作者: @nathon-lee**

### 6.1 问题背景

ZeRO-3维护一个flat `ds_tensor` → 统一GPU内存buffer → 存放partitioned optimizer states, gradients, parameters

当参数在forward/backward中被gather (AllGather) → 使用 → release → **在ds_tensor中留下hole/gap** → 内存碎片化

碎片化的后果:
- 后续allocation请求无法找到足够大的连续free chunk → 即使总free memory足够
- Peak memory升高 → 可训练的模型变小

### 6.2 defragment实现

源码变更:
- 从 `deepspeed/runtime/zero/stage3.py` → 提取到 `deepspeed/runtime/zero/utils.py`
- **Before**: `defragment`是`DeepSpeedZeroOptimizer_Stage3`的方法 → 和optimizer耦合
- **After**: `defragment`是独立utility function → 可复用

**defragment操作步骤**:
1. 扫描`ds_tensor`中的所有occupied和free segments
2. 将occupied blocks向buffer前端compact → 消除中间的gaps
3. 更新每个partitioned tensor的offset table → 指向compact后的新位置
4. 结果: **一个连续的free region在buffer尾部** → 未来allocation可以高效使用

### 6.3 适用场景

- **长时间训练**: 大量forward/backward → gather/release → 碎片化累积 → defragment清理
- **大模型**: ds_tensor更大 → 碎片化影响更严重 → defragment收益更大
- **CPU/NVMe offload**: offload场景下GPU memory更紧张 → 碎片化更致命 → defragment更关键
- **DeepCompile场景**: 编译图可能改变参数生命周期 → 需要更灵活的内存管理 → defragment作为utility可被多个pass调用

### 6.4 源码变更统计

- `deepspeed/runtime/zero/stage3.py`: +4/-36 (提取方法)
- `deepspeed/runtime/zero/utils.py`: +33 (新增defragment utility)

**33行的defragment utility** → 精简但完整 → scan→compact→update offsets

---

## 7. 补充特性 <a id="7-supplementary"></a>

### 7.1 SDMA AllGather via Mori (PR#7999, merged 2026-05-14)

**源码: `deepspeed/comm/mori.py` + `examples/sdma_allgather/`**

- **Mori**: AMD MI300的**intra-node SDMA (Symmetric Direct Memory Access)** → 在同一node内, GPU间直接内存拷贝 → 不经过RCCL
- **mori_cpp.AllGatherIntoTensor**: C++ extension → 替代NCCL/RCCL AllGather → 用于ZeRO-3参数prefetch
- **fallback**: Mori init失败 → 自动回退到`dist.allgather_fn` (RCCL/NCCL)

**配置**: 环境变量 (非ds_config):
```bash
DS_SDMA_ALLGATHER=1                    # opt-in (必须显式开启)
DS_SDMA_ALLGATHER_MAX_NUMEL=N          # transit buffer size (default 64M = 256 MiB per-rank input)
```

**Benchmark (8x MI300X)**:
| | GPT-7B-ish | Qwen3-32B |
|---|---|---|
| SDMA off | 697.7 ms/step | 1402.5 ms/step |
| SDMA on | 622.0 ms/step | 1263.2 ms/step |
| **gain** | **+10.85%** | **+9.93%** |

Loss curves完全一致 → 功能正确性确认

**RTX 4090**: ❌ → Mori需要AMD MI300 + ROCm → NVIDIA GPU不受益

### 7.2 Dynamic offload compatible with static optimizer offload (PR#7979, merged 2026-04-24)

Fix #6596 → 允许ZeRO-3的**动态offload和静态optimizer offload混用**

之前: offload_optimizer (static, CPU/NVMe) 和 dynamic offload (按需offload) 不兼容 → 现在可以同时启用 → 更灵活的内存管理策略

### 7.3 BF16 optimizer states with CPU offload (PR#8010, v0.19.1)

源码: @lucaspirola → CPU offload的optimizer states现在支持BF16 → 之前只有FP32 → 节省CPU memory

### 7.4 Singleton MoE collectives optimization (PR#7997, v0.19.1)

当EP size = 1 → 单GPU → all-to-all是identity → 优化: 跳过collective → 直接local compute → 减少通信开销

### 7.5 v0.19.1其他修复

- Fix ZenFlow ZeRO-3 selective optimizer crash with NVMe offload (PR#8042)
- Fix DeepCompile AOT kwargs patching for PT >= v2.11 (PR#8024)
- Fix DeepCompile ZeRO-3 release parameter lifetime (PR#8032)
- Fix DeepCompile all-gather scheduler candidate selection (PR#8033)
- Fix DeepCompile ZeRO-1 grad target lifetime (PR#8036)
- Normalize DeepCompile ZeRO-3 grad dtype before reduction (PR#8038)

---

## 8. RTX 4090 实战分析 <a id="8-rtx4090"></a>

### 8.1 AutoEP → EP=1场景

**RTX 4090 = 单GPU → autoep_size=1 → EP=1 → singleton MoE collectives (PR#7997优化)**

EP=1时的AutoEP行为:
- all-to-all dispatch/combine → **identity operation** → PR#7997已优化为skip collective → 直接local compute
- grouped-GEMM → **仍然有用!** → 即使EP=1, grouped-GEMM把所有expert的GEMM合并为一次kernel launch → 比逐expert GEMM更快
- Triton fill-indices → **RTX 4090支持Triton** → 但若Triton不可用 → 有纯PyTorch CPU fallback

**AutoEP + LoRA + ZeRO-2**: RTX 4090可行组合!
- ZeRO-2: 分区optimizer states → GPU memory可控
- LoRA: 只训练少量参数 → 可训练MoE的router + LoRA adapter → expert参数freeze (注意: AutoEP已修复requires_grad同步bug)
- AutoEP EP=1: grouped-GEMM优化 → 无通信开销 → 但仍有kernel launch合并优势
- Muon optimizer: 可搭配 → 但需注意is_expert_group=True → Gram NS对expert权重矩形的aspect ratio优势大

**但**: RTX 4090 PCIe → 多GPU场景 → EP>1 → all-to-all → PCIe带宽瓶颈 → 性能灾难 (和DeepEP结论一致)

### 8.2 ZenFlow → RTX 4090

- **overlap_step=True**: CPU optimizer process overlap → RTX 4090的CPU强大 → 可行!
- **但**: ZenFlow要求`cpu_offload=True` → GPU optimizer states全部offload到CPU → GPU memory大幅释放 → 但CPU→GPU copyback有延迟
- **selective optimizer**: 只更新10%参数 → 减少CPU-GPU通信 → 合理
- **PR#8058 native CPU Adam**: chunked copyback → GPU spike从2944 MiB降到256 MiB → RTX 4090 24GB → spike大幅减少 → 可行!
- **pt_reserved_cores_perc=0.5**: RTX 4090 host CPU → 需预留一半给PyTorch → 剩余给ZenFlow → 合理

**RTX 4090最优ZenFlow配置**:
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "zenflow": {
      "topk_ratio": 0.1,
      "overlap_step": true,
      "offload": true,
      "pt_reserved_cores_perc": 0.5
    }
  }
}
```

### 8.3 DeepCompile → RTX 4090

- **ZeRO-3 + DeepCompile**: RTX 4090 → ZeRO-3的3Ψ AllGather通信量 → PCIe灾难 → 即便DeepCompile优化了gather/release, 通信瓶颈不变
- **ZeRO-1/2 + DeepCompile**: 可行 → selective_gather pass对Z1/Z2也有优化 → 减少gradient reduction通信
- **PT v2.9/2.10**: RTX 4090 → DeepCompile已兼容 → 但PT v2.11仍有问题
- **symmetric_memory**: ❌ → Mori需AMD MI300

**RTX 4090**: DeepCompile本身是纯软件层 → ✅ → 但搭配ZeRO-3 → ❌ (通信瓶颈) → 搭配ZeRO-2 + selective optimizer → ✅

### 8.4 Muon → RTX 4090

- **Gram Newton-Schulz**: ✅ → 纯GPU矩阵运算 → RTX 4090完全支持 → 且m >> n时FLOPs减少 → 更快
- **fp16 compute**: ✅ → RTX 4090支持fp16 → Gram NS用fp16 → 更好精度
- **Memory savings**: 9% → modest → 但对24GB GPU → 每GB都有价值 → 3 GiB省 ≈ 多fit一些batch/sequence
- **ZeRO-2 Muon**: ✅ → partitioned momentum buffer → 可行
- **ZeRO-3 Muon**: ❌ → ZeRO-3本身在RTX 4090不可行 (3Ψ通信)

**RTX 4090最优Muon配置**:
```json
{
  "optimizer": {
    "type": "Muon",
    "params": {
      "ns_method": "gram",
      "muon_lr": 1e-4,
      "adam_lr": 2e-6
    }
  },
  "zero_optimization": {"stage": 2}
}
```

### 8.5 AutoSP → RTX 4090

- **单GPU SP**: ❌ → SP需要多GPU → sequence维度切分 → 每个GPU持1/N的sequence
- **ViT+LLM SP**: ❌ → 需多GPU → RTX 4090 PCIe多GPU → 通信瓶颈
- **但**: AutoSP的compile pass → 可用于ZeRO-2 compile优化 → 即使不用SP本身 → 编译优化仍有价值

### 8.6 Defragment → RTX 4090

- **✅**: RTX 4090 24GB → 内存紧张 → 碎片化影响大 → defragment收益显著
- **ZeRO-3 defragment**: ZeRO-3本身不可行 → 但defragment utility也可用于其他场景
- **DeepCompile+defragment**: 编译图改变参数生命周期 → 可能需要defragment → utility化使其可被多个系统调用

### 8.7 RTX 4090综合最优组合 (0.19.x)

```
★ ★ ★ 最佳方案: ZeRO-2 + LoRA + AutoEP(EP=1) + Muon(Gram NS) + ZenFlow(overlap) + grouped-GEMM

训练7B MoE (如Qwen3-MoE-A2.7B):
  ZeRO-2 → optimizer states分区 → 12GB GPU → 比ZeRO-3少通信
  LoRA rank=32 → 只训练router+LoRA → expert freeze → AutoEP需requires_grad同步 ✓
  AutoEP EP=1 → grouped-GEMM → 单GPU无通信 → 但kernel合并优化 ✓
  Muon Gram NS → 省momentum buffer → 9% memory saving → 更快收敛 ✓
  ZenFlow overlap → CPU optimizer → GPU memory释放 → selective update ✓

vs 之前的最佳方案 (ZeRO-2 + LoRA + CPU_Adam):
  Muon → 省momentum buffer → vs Adam的m+v → 更省内存
  AutoEP grouped-GEMM → expert compute更高效 → vs naive逐expert
  ZenFlow selective update → 只更新10%参数 → vs 每步全更新
  → 综合预计: 更好throughput + 更省memory + 更快收敛
```

---

## 版本追踪

| 特性 | 版本 | PR | 状态 |
|------|------|-----|------|
| AutoEP | v0.19.1+ | #7938 | MERGED (2026-06-11) |
| ZenFlow base | 早期版本 | #7759 | MERGED |
| ZenFlow native CPU Adam | v0.19.x+ | #8058 | OPEN |
| DeepCompile+Z3 fix (PT 2.9/2.10) | v0.19.0 | #7951 | MERGED |
| DeepCompile AOT kwargs (PT 2.11) | v0.19.1+ | #8024 | MERGED |
| DeepCompile Dynamo frame skip (PT 2.11+) | v0.19.x | #8059 | OPEN |
| Muon ZeRO-2 | 早期版本 | initial | MERGED |
| Muon ZeRO-3 | v0.18.9+ | #7919 | MERGED |
| Gram Newton-Schulz | v0.19.0 | #7953 | MERGED |
| Muon Blog | v0.19.1 | #7962 | MERGED |
| AutoSP base | v0.18.9 | #7860 | MERGED |
| AutoSP Multimodal ViT+LLM | v0.19.0 | #7984 | MERGED |
| ZeRO-3 Defragment utility | v0.19.0 | #7940 | MERGED |
| SDMA AllGather via Mori | v0.19.1 | #7999 | MERGED |
| Dynamic+Static offload | v0.19.0 | #7979 | MERGED |
| BF16 CPU optimizer states | v0.19.1 | #8010 | MERGED |
| Singleton MoE collectives | v0.19.1 | #7997 | MERGED |

---

## 关键源码路径索引

```
# AutoEP
deepspeed/module_inject/auto_ep_config.py         # config解析
deepspeed/module_inject/auto_ep.py                # MoE检测+替换
deepspeed/module_inject/auto_ep_layer.py          # AutoEPMoELayer执行wrapper
deepspeed/module_inject/auto_ep_presets/           # 5个preset定义
deepspeed/moe/ep_router.py                         # EP router (← TorchTitan)
deepspeed/moe/ep_experts.py                        # GroupedExperts (← TorchTitan)
deepspeed/moe/ep_kernels.py                        # Triton fill-indices (← TorchTitan)
deepspeed/moe/ep_repack.py                         # weight repack (HF→grouped)
deepspeed/checkpoint/autoep_universal.py           # checkpoint save/load

# ZenFlow
deepspeed/runtime/zenflow/engine.py                # configure_zenflow + zenflow_step
deepspeed/runtime/zenflow/zenflow_config.py        # ZenFlowConfig
deepspeed/runtime/zenflow/zenflow_stage_1_and_2.py  # Z1/Z2 ZenFlowZeroOptimizer
deepspeed/runtime/zenflow/engine_stage3.py         # Z3 ZenFlow extension

# DeepCompile
deepspeed/compile/config.py                        # CompileConfig
deepspeed/compile/init_z3.py                       # Z3 compile初始化
deepspeed/compile/passes/selective_gather.py       # 选择性gather pass
deepspeed/compile/passes/zero3_compile.py          # Z3 compile pass
deepspeed/compile/passes/sp_compile.py             # AutoSP compile pass
csrc/compile/deepcompile.cpp                       # C++ guard override

# Muon
deepspeed/runtime/zero/muon/original_muon.py       # Newton-Schulz + Gram NS + muon_update

# AutoSP
deepspeed/sequence/autosp_detector.py              # ViT/LLM detector
deepspeed/sequence/autosp_vit.py                   # UlyssesSPViTAttention
deepspeed/sequence/autosp_fusion.py                # Cross-modal fusion adapters
deepspeed/compile/passes/sp_compile.py             # SP compile pass

# ZeRO-3 Defragment
deepspeed/runtime/zero/utils.py                    # defragment utility function

# Mori SDMA
deepspeed/comm/mori.py                             # Mori SDMA AllGather backend
```
