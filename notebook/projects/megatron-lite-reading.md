# Megatron Lite 深度阅读笔记

> 来源: NVIDIA/Megatron-LM PR #4885 (merged 2026-06-11 into dev branch)
> 作者: ISEEKYAN (NVIDIA contributor)
> 代码: 71 commits, +29,072 lines, 178 files
> 评审: sbhavani (NVIDIA) — 要求改名 lite→megatron-compose, 但已按原名合入 dev
> 社区: mindlab-bot (Mind Lab) — 强力支持, 关注 LoRA + PEFT 集成

---

## 一、Megatron Lite 是什么?

★★★★★ **定位**: 实验性、agentic-native 训练运行时 + 原生模型实现层

Megatron Lite 不是"轻量版 Megatron-LM",而是**在 Megatron-Core 分布式原语之上的窄合约运行时**。核心设计哲学:

- **合约驱动**: Runtime→Model→Primitive 三层分离, 每层通过协议(protocol)交互
- **Agentic-native**: 代码结构设计为 agent 可理解的组合边界, skills 系统提供操作合约
- **原生实现**: `backend="mlite"` 构建原生 `megatron.lite` 模型代码, 不是包装(wrapper)
- **验证优先**: 每个 change 都有 correctness pair + SHA256 fingerprint 签off

**关键区别**: 传统 Megatron-LM 是 recipe-level 代码 (全局初始化 + 脚本驱动); Lite 是 `RuntimeConfig → create_runtime → build_model` 的**窄合约 API**。

### 初始 PR 包含内容

| 模块 | 内容 |
|------|------|
| runtime | API + 后端注册 (mlite/mbridge/bridge) |
| model | Qwen3 MoE + Qwen3.5 MoE 原生实现 |
| primitive | parallel/ckpt/modules/ops/optimizers |
| examples/bench | 正确性+性能 benchmark |
| examples/verl | SFT + GRPO 集成脚本 |
| skills | Agent 操作合约 (6 namespace, 30+ skill) |
| docs | 4 design docs (architecture/runtime/models/porting) |

### 初始 PR 刻意排除

| 排除项 | 状态 |
|--------|------|
| Hybrid model implementations | 未实现 |
| Bridge model/runtime implementations | 仅引用后端 |
| FSDP2 optimizer primitives | **已包含** (与排除列表矛盾 — 实际已实现) |
| Benchmark scripts | **已包含** (同上) |
| Dense Qwen3 support | 不包含, 仅 MoE |
| LoRA as first-class feature | 已有基础设施, 但非正式特性 |

---

## 二、架构: Lite vs Full Megatron-LM

★★★★★ **核心架构差异**

### 三层结构

```
experimental/lite/
  megatron/lite/
    runtime/        ← 第1层: 生命周期+训练步编排
      __init__.py       (公开 API)
      backends/         (mlite/mbridge/bridge 实现)
      contracts/        (PackedBatch, Handle, Config)
      megatron_utils.py (Core 适配)
    model/          ← 第2层: 模型注册+协议
      registry.py       (model_name → protocol 映射)
      qwen3_moe/        (原生 Qwen3 MoE)
      qwen3_5/          (原生 Qwen3.5 MoE)
    primitive/      ← 第3层: 可复用底层原语
      parallel/         (TP/EP/PP/CP/SP state + linear)
      ckpt/             (checkpoint save/load)
      modules/          (attention/gqa/moe/lora/mlp/router/mrope/mtp)
      ops/              (算子级原语)
      optimizers/       (distopt wrap + FSDP2)
      kernels/          (内核级原语)
      protocols.py      (通用协议定义)
      train_step.py     (前向/后向编排)
```

### Runtime vs Model 边界

★★★★ **这是 Lite 最核心的设计决策**

Runtime 负责:
- 分布式初始化
- 模型协议加载 (不直接知道 Qwen 实现细节!)
- checkpoint save/load dispatch
- forward/backward microbatch 编排
- optimizer + LR scheduler stepping
- offload hooks (可选)

Model Protocol 负责:
- 架构 config 创建
- Model chunk 构建
- 模型特定 recompute/offload wiring
- 模型特定 optimizer 构建
- HF checkpoint load/export mapping

**设计含义**: Runtime 层可以完全不修改地支持新模型 — 只需注册新 protocol。

### API Tier

★★★★ 公开运行时入口:

```python
from megatron.lite.runtime import MegatronLiteConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="mlite",
    hf_path="/path/to/hf-model",
    backend_cfg=MegatronLiteConfig(
        model_name="qwen3_moe",
        impl="lite",
        parallel=ParallelConfig(tp=1, pp=1, cp=1, ep=1),
    ),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

运行时 API 层级:

| Tier | 方法 | 说明 |
|------|------|------|
| Pretraining | build_model | 构建分布式模型 |
| | save_checkpoint / load_checkpoint | checkpoint 管理 |
| | train_mode / eval_mode | 模式切换 |
| | forward_backward | 前向后向步 |
| | zero_grad / optimizer_step / lr_scheduler_step | 训练步 |
| Extended | export_weights | HF 权重导出 (lite 独有) |
| | to | 设备迁移 (lite 独有) |

### Backend Registry

★★★ 内置后端键: `mlite`, `mbridge`, `bridge`

| Backend | 用途 |
|---------|------|
| `mlite` | 原生 Lite 运行时 — 生产路径 |
| `mbridge` | 旧 mbridge 包引用 — correctness/性能对比 |
| `bridge` | 真 Megatron-Bridge 运行时 — 需 import megatron.bridge |

自定义后端可注册:
```python
from megatron.lite.runtime import register_runtime
register_runtime("my_backend", "my_package.my_runtime")
```

---

## 三、LoRA: 排除但已有完整基础设施

★★★★★ **这是 RTX 4090 最关键的发现!**

### Mind Lab 的评论 (PR #4885 评论 #2, 19 reactions)

Mind Lab 明确表示 LoRA 是 Lite 的核心用例:

> "LoRA illustrates the problem concretely. While it is possible to implement LoRA
> on top of Megatron, making it reusable across models and protocols is difficult
> because adapter injection, frozen base weights, adapter-only optimizer groups,
> and distributed post-training workflows span multiple layers of the stack."

> "A narrower RuntimeConfig → create_runtime → build_model/optimizer_step contract
> makes Megatron much easier to integrate into external training and post-training
> systems... For LoRA specifically, Lite could provide well-defined injection points
> and a cleaner path for adapter checkpointing and HF/PEFT interoperability."

Mind Lab 表示愿意在 LoRA、Qwen/MoE、post-training 集成方面帮助验证。

### 实际已实现的 LoRA 基础设施

★★★★★ 虽然 LoRA 被列为"排除", 但代码中已有**完整的 LoRA 原语**:

**1. `primitive/modules/lora.py` — 核心 LoRA 原语模块**

| 类/函数 | 功能 | TP 感知 |
|---------|------|---------|
| `LoraConfig` | LoRA 配置 dataclass (rank/alpha/dropout/targets) | - |
| `LinearLoRA` | TP 分片线性层 LoRA delta | 完全支持 SP/CP/row/col parallel |
| `GroupedLinearLoRA` | MoE 专家级 per-expert LoRA | 支持 num_local_experts |
| `SharedGroupedLinearLoRA` | 所有专家共享 LoRA delta | 跨专家共享 |
| `freeze_non_lora_params` | 冻结 base 参数, 仅留 adapter 可训练 | - |
| `normalize_lora_config` | 多种配置格式统一为 LoraConfig | - |

★★★★★ `LinearLoRA` 的 TP 感知设计极其精细:

- `sequence_parallel_input`: 支持 SP 输入 all-gather
- `row_parallel_output`: 支持 row-parallel reduce-scatter
- `rank_partitioned_a`: LoRA rank 维度跨 TP 分片
- `output_partitioned_b`: LoRA 输出跨 TP 分片
- `input_parallel_reduce`: 输入 all-reduce 路径
- `_SequenceParallelRankPartitionedLoRA`: 自定义 autograd Function, 在 backward 中重算而非存储大激活 → **内存节省**

★★★★ `GroupedLinearLoRA` 专门为 MoE 专家设计:
- per-expert LoRA A/B 权重: `(num_local_experts, rank, in_features)`
- 按专家 splits 切分输入, 分别计算 LoRA delta
- 这直接解决了 MoE LoRA 的核心难题

**2. `model/qwen3_moe/lite/lora_adapter.py` — Qwen3 MoE PEFT 适配器**

- PEFT/Mint-compatible adapter import/export
- `_iter_qwen_chunks`: 解包 Megatron 分布式模型 chunk
- `_all_gather_cat`: 跨 TP rank gather LoRA 权重
- `_select_tp_replicated`: 选择 TP 复制部分
- HF safetensors LoRA 权重加载/导出
- 支持 TP 分片权重正确拼接导出

**3. README 明确承认 Mind Lab 的贡献**:

> "The Qwen3 MoE LoRA adapter support follows Mind-Lab's PEFT/Mint-compatible
> adapter work. Thanks to Mind-Lab for the reference implementation and guidance."

### 为什么 LoRA 被列为"排除"?

★★★ ISEEKYAN 和 sbhavani 的讨论揭示了原因:

1. **PR 太大** (29,072 lines, 178 files) — sbhavani 要求拆分
2. **命名争议** — sbhavani 要求改名 lite→megatron-compose, 移到 examples/
3. **需要分步合入 main** — skeleton→contracts→validation→skills→模型实现→checkpoint→LoRA→FSDP2→verl
4. **LoRA 跨越多层**: adapter injection + frozen base + optimizer groups + distributed + checkpoint → 太复杂, 不适合首批合入

**结论**: LoRA 不是"不打算做", 而是"太复杂, 需要单独 PR 分步合入"。Mind Lab 的评论表明社区已经在推动 LoRA 成为 Lite 的核心特性。

---

## 四、性能 Benchmark

★★★★ **8x H100 Qwen3.5-35B-A3B 验证结果**

| 运行时 | Impl | Optimizer | Step ms | Tok/s | Tok/s/GPU | Peak Mem GB | TFLOPs/GPU |
|--------|------|-----------|---------|-------|-----------|-------------|------------|
| **mlite** | lite | distopt | **309.433** | **105,896.935** | **13,237.117** | **14.324** | **80.444** |
| mbridge | bridge | distopt | 332.201 | 98,639.089 | 12,329.886 | 17.987 | 74.931 |
| bridge | bridge | distopt | 334.936 | 97,833.496 | 12,229.187 | 16.403 | 74.319 |

★★★★★ **MLite vs mbridge (Core/distopt 引用) 关键指标**:

| 指标 | MLite 优势 | 量化 |
|------|-----------|------|
| Step 速度 | MLite 更快 | -7.0% step ms |
| Token throughput | MLite 更高 | +7.3% tok/s |
| GPU throughput | MLite 更高 | +7.6% tok/s/GPU |
| 内存 | MLite 更少 | **-20.3% peak memory** |
| TFLOPs | MLite 更高 | +7.3% TFLOPs/GPU |

★★★★★ **内存优势 (-20.3%) 是最重要的发现**: 17.987GB → 14.324GB, 节省 3.663GB。这意味着 Lite 运行时比传统 Megatron-Core 路径更内存高效。

### 确定性正确性验证

★★★★★ **1x GPU, seed=42, Qwen3.5 MoE, 2 steps**

| 指标 | 结果 |
|------|------|
| max_loss_abs | **0.0** (bitwise 相同!) |
| max_grad_norm_abs | **0.0** |
| mismatches | [] |
| post-step weight SHA256 | 完全匹配 |
| eval logits SHA256 | 完全匹配 |

**step 0**: loss=13.027, grad_norm=120.755
**step 1**: loss=14.699, grad_norm=96.157

★★★★★ **bitwise 正确性** — MLite 与 Megatron-Core/distopt 在数学上完全一致。这证明了 Lite 的原生实现没有牺牲精度。

### 8x H100 运行配置

- Slurm job 12624917, exit code 0:0
- torch==2.10.0+cu129
- 8 GPU, STEPS=15, WARMUP=5, SEQ_LEN=1024
- NUM_MICROBATCHES=4, TRUNCATE_LAYERS=8
- KEEP_EXPERTS=8, SAME_DATA_ACROSS_DP=1

---

## 五、verl 集成: GRPO 训练路径已实现!

★★★★★ **这是 RTX 4090 GRPO 的第二条可用路径!**

### verl_mlite 包结构

```
experimental/lite/examples/verl/
  verl_mlite/
    engine/
      mlite_engine.py    ← VERL BaseEngine 实现, backed by megatron.lite.runtime
      config.py           ← MegatronLiteEngineConfig
    config/
      engine/mlite.yaml   ← Hydra engine config: engine=mlite
      actor/mlite_actor.yaml ← actor config
    compat.py             ← VERL API 版本兼容层
  scripts/
    run_qwen3moe_sft.sh       ← SFT launcher
    run_qwen3moe_gsm8k_sft.sh ← GSM8K SFT wrapper
    run_qwen3moe_gsm8k_grpo.sh ← **GRPO launcher!**
```

### GRPO 启动脚本关键配置

★★★★★ `run_qwen3moe_gsm8k_grpo.sh` 默认设置:

| 配置 | 值 | RTX 4090 意义 |
|------|-----|---------------|
| actor backend | `mlite_actor` | Megatron Lite 训练引擎 |
| rollout backend | `vllm` | vLLM 推理引擎 |
| KL in reward | `False` | **类似 bypass_mode!** |
| KL loss | `False` | **类似 bypass_mode!** |
| reference model | disabled | **省 14GB!** |
| policy loss | `vanilla` | 纯 GRPO, 无 KL penalty |
| loss agg | `seq-mean-token-sum-norm` | 标准 GRPO aggregation |
| critic | `enable=False` | 不用 critic 网络 |
| legacy worker | `disable` | 新 engine worker path |
| adv_estimator | `grpo` | GRPO 优势估计 |

★★★★ **Offload 支持**:

| 配置 | 说明 |
|------|------|
| PARAM_OFFLOAD | CPU↔GPU 参数迁移 (train/eval 上下文切换) |
| OPTIMIZER_OFFLOAD | FSDP2 optimizer state 保持 CPU, offload_fraction=1.0 |
| GRAD_OFFLOAD | 梯度 offload |

★★★ **FSDP2 optimizer offload**: `OPTIMIZER_OFFLOAD=True` 设置 `offload_fraction=1.0`, forward/backward 期间 optimizer update state 保持 CPU → 减少 GPU 内存压力。

### mlite_engine.py 核心架构

★★★★ VERL BaseEngine 实现:
- 继承 `BaseEngine`, 注册为 `mlite` backend
- 通过 `create_runtime(cfg)` 构建模型
- 支持 checkpoint save/load (verl → Lite ckpt primitive)
- 支持 param/optimizer/grad offload (上下文切换时)
- `_isolate_compile_cache_per_rank()` — 避免 Inductor/Triton cache 竞争
- 支持 `train_mode`/`eval_mode` 上下文切换 (verl rollout→training 切换)

---

## 六、Skills 系统: Agentic 操作合约

★★★ **Skills 是 Lite 的独特设计 — 为 AI agent 定义操作合约**

### Skill 结构

每个 skill 文件有 3 个区域:
1. Schema 块 (检索+路由) — `MLITE_SKILL_SCHEMA_BEGIN/END` 分隔
2. Body — Python-like pseudocode, 不是自由散文
3. 必须有 explicit exit: `done/blocked/out_of_scope`

### Skill Namespace

| Namespace | Skills | 用途 |
|-----------|--------|------|
| basic | constitution, find_reference, align_precision, align_e2e_precision, lint_skill, construct_proxy_task, review_threshold | 基础验证 |
| primitive | contract, principle, select_for_compose, design, validate, fuse, process_group, parallel.tp/ep/pp/cp, optimizer.fsdp/distopt, module.moe/gqa/thd | 原语操作 |
| model_compose | config_mapping, weight_mapping, build_model, qwen | 模型组合 |
| application | runtime, verl, bench | 应用集成 |
| perf | measure, memory, fusion, optimize | 性能优化 |
| insight | progressive_model_support, progressive_primitive_design, scope_control | 进度洞察 |

★★★ **Skill 与 docs 的边界**: Skills 描述 agent 如何工作; docs 描述 Megatron Lite 给用户/评审者。Skills 不替代测试, 不存储任务历史。

---

## 七、命名争议与合入路径

★★★★ **sbhavani vs ISEEKYAN 的分歧**

### sbhavani 的意见 (NVIDIA 内部评审):

1. **改名 lite→megatron-compose** — 认为 "lite" 不符合目标定位
2. **移到 examples/ 而非 experimental/** — 不需要新目录
3. **先合 skeleton**: 仅 API contracts + docs + skills + toy example, 不含真实模型
4. **分步合入 main**: contracts→validation→skills→model→ckpt→LoRA→FSDP2→verl

### ISEEKYAN 的回应:

1. **Skills 本身不是最重要的** — 关键是 skills + modular runnable reference code 的组合
2. **Toy example 不够** — 需要真实模型实现来锻炼 hard integration surfaces (MoE, ckpt, optimizer, HF export)
3. **先合入 dev, 再拆分合入 main** — 让社区用户先试用完整 OOTB VERL 路径
4. **对改名开放** — 但 "lite" 强调更小运行时表面

### 最终结果:

★★★ PR 按 ISEEKYAN 的版本**合入 dev 分支** (2026-06-11), 原名 "lite" 保留。但 sbhavani 的意见意味着后续 main 分支合入可能改名。这表示:
- 当前 dev 分支有完整的 Lite 可用
- main 分支合入形态可能变化
- 命名可能最终变为 `megatron-compose` 或类似

---

## 八、RTX 4090 可行性分析

★★★★ **核心问题: Megatron Lite 能否使 RTX 4090 单 GPU 训练可行?**

### 正面因素

★★★★★ **确定性正确性验证在 1x GPU 上完成**!

- Slurm job 12630675: 1 GPU, seed=42, Qwen3.5 MoE
- max_loss_abs=0.0, max_grad_norm_abs=0.0, bitwise 完全一致
- **Lite 在单 GPU 上数学正确性已验证**

★★★★ **Offload 支持**:

- PARAM_OFFLOAD: CPU↔GPU 参数迁移
- OPTIMIZER_OFFLOAD: FSDP2 optimizer state CPU 常驻 (offload_fraction=1.0)
- GRAD_OFFLOAD: 梯度 offload
- **这些机制在 verl GRPO 脚本中已默认可用**

★★★★ **LoRA 原语完整**:

- LinearLoRA 支持 TP-aware sharding
- GroupedLinearLoRA 支持 MoE per-expert LoRA
- freeze_non_lora_params 冻结 base, 仅训练 adapter
- **LoRA + offload 组合可能是 RTX 4090 关键路径**

★★★ **窄合约 API**: RuntimeConfig→create_runtime→build_model — 简洁, 不需要全局初始化

### 负面因素

★★★★★ **模型尺寸问题**:

- 当前仅支持 Qwen3 MoE 和 Qwen3.5 MoE — 都是 35B+ 级别模型
- Dense Qwen3 不包含 → 小模型训练路径缺失
- Qwen3.5-35B-A3B 即使只有 3B active params, 仍需加载全部专家权重
- **RTX 4090 24GB VRAM 无法容纳 35B 模型, 即使 LoRA + offload**

★★★★ **FSDP2 在单 GPU 无用**:

- FSDP2 per-parameter sharding 需多 GPU
- 单 GPU = 无分片收益, 仅 offload 功能有用
- offload_fraction=1.0 = optimizer 全在 CPU → 训练速度极慢
- **与之前分析一致: FSDP2 对单 GPU 无实质帮助**

★★★ **MoE 特殊挑战**:

- GroupedLinearLoRA 需要 num_local_experts — 单 GPU = 所有专家 local
- 内存压力: 专家权重 + LoRA adapter + optimizer state + activation
- **RTX 4090 上 MoE LoRA 训练需要 extreme offload**

★★★ **缺少小模型支持**:

- 没有 1-7B dense 模型实现
- 没有 LoRA + small model 的验证路径
- **需要社区添加小模型 protocol**

### RTX 4090 可行性评估

★★★ **短期 (v0.1)**: 不可行 — 模型太大, 缺小模型, LoRA 未正式发布
★★★★ **中期 (v0.2-0.3)**: **可能可行** — 条件:
1. LoRA 成为正式特性 (Mind Lab 已推动)
2. 小模型 protocol 添加 (如 Qwen3-1.7B dense)
3. LoRA + offload 组合验证通过
4. 单 GPU correctness benchmark 发布

★★★★★ **长期**: **非常可行** — Lite 的窄合约 + LoRA 原语 + offload 是正确的技术方向:
- Megatron-Core distopt 优化器在单 GPU 上正确性已验证
- LoRA freeze + offload 可大幅减少训练内存
- verl GRPO 集成已 disable KL (类似 bypass_mode)
- Skills 系统让 agent 可以自动适配模型

### RTX 4090 GRPO 排名更新

★★★★★ 考虑 Megatron Lite 后的 GRPO 框架排名:

| 排名 | 框架 | 说明 | RTX 4090 可行性 |
|------|------|------|-----------------|
| #1 | rLLM Tinker | in-process, bypass default, auto LoRA | 立即可用 |
| #2 | verl + vLLM | 外部引擎, bypass_mode, INT8 KV | 立即可用 |
| **#2.5** | **verl + Megatron Lite** | 外部引擎, KL disabled, offload, LoRA 域语 | **中期可用** |
| #3 | Megatron-LM (传统) | singleton PG bug (#5203), 无 LoRA | 不可行 |

★★★★ **verl + Megatron Lite vs verl + vLLM**:

| 维度 | verl + vLLM | verl + Megatron Lite |
|------|-------------|---------------------|
| 训练引擎 | PyTorch native | Megatron-Core distopt |
| rollout 引擎 | vLLM | vLLM (相同) |
| LoRA | PEFT/transformers | Megatron-style sharded LoRA |
| 分布式 | DDP | TP/EP/PP/CP (但单GPU无用) |
| 内存优化 | bypass_mode | offload + freeze_non_lora |
| KL penalty | bypass_mode=True | KL disabled (类似) |
| 数学正确性 | 未验证 bitwise | **bitwise 验证** |
| 模型支持 | 广泛 | 仅 Qwen3/3.5 MoE |
| 成熟度 | 生产可用 | 实验性 |

---

## 九、与 verl/rLLM 的 GRPO 训练对比

★★★★★ **三框架 GRPO 训练路径对比**

### verl + Megatron Lite (新路径)

```
用户数据 → verl GRPO trainer
  → actor: Megatron Lite runtime (mlite_engine.py)
    → build_model → RuntimeConfig → create_runtime
    → forward_backward → Lite train_step
    → optimizer_step → distopt/FSDP2
    → offload hooks → CPU↔GPU migration
  → rollout: vLLM inference engine (同 verl 标准)
  → reward: external reward function
  → no reference model (KL disabled)
```

★★★★ **优势**: Megatron-Core distopt bitwise 正确性 + 精细 TP-aware LoRA + 完整 offload 栈
★★★ **劣势**: 仅 MoE 模型 + 实验性 + 单 GPU 无分布式收益

### rLLM Tinker (当前最优)

```
用户数据 → rLLM Fully Async Trainer
  → Tinker backend: in-process model
    → auto LoRA init + zero-copy weight sync
    → bypass_mode default true (no ref model)
    → detach metrics → OOM prevention
  → rollout: 同进程推理 (无 vLLM 依赖)
  → reward: built-in verifiers
```

★★★★★ **优势**: in-process 最高效 + bypass default + zero-copy + 最简架构
★★★ **劣势**: 灵活性较低 + 不用 Megatron-Core + 小社区

### verl + vLLM (成熟路径)

```
用户数据 → verl GRPO trainer
  → actor: PyTorch native (DDP/FSDP)
    → PEFT LoRA (transformers-style)
    → bypass_mode=True → skip ref model
  → rollout: vLLM V1 engine
    → INT8 KV (FlashInfer backend)
    → MRv2 → 需 VLLM_USE_V2_MODEL_RUNNER=0
  → reward: external reward function
```

★★★★ **优势**: 最成熟 + 模型支持最广 + 生产验证
★★★★★ **劣势**: MRv2 交互问题 + SM89 batch invariance + bitwise 正确性未验证

---

## 十、Primitive 模块详解

★★★★ **Lite 原语层是技术深度最高的部分**

### Parallel Primitives

| 模块 | 功能 |
|------|------|
| state.py | ParallelState 管理 (TP/EP/PP/CP/SP/DP groups) |
| linear.py | TP 分片线性层原语 |
| pipeline.py | Pipeline parallel 编排 |
| pp.py | PP stage 管理 |
| cp.py | Context parallel |
| sp.py | Sequence parallel |
| mhc.py | Multi-head cross attention |
| thd.py | THD attention format |

### Module Primitives

★★★★★ **modules/ 是最丰富的原语集合**:

| 模块 | 功能 | RTX 4090 意义 |
|------|------|---------------|
| **lora.py** | LinearLoRA/GroupedLinearLoRA | ★★★★★ 核心 |
| attention/ | 注意力实现 | FP16/BF16 可用 |
| gqa.py | Grouped-query attention | 支持 |
| moe.py | MoE 专家模块 | ★★★ 需要 |
| mlp.py | MLP 模块 | 标准 |
| router.py | MoE router | 标准 |
| mrope.py | Multimodal rotary PE | VLM 支持 |
| mtp.py | Multi-token prediction | 推理加速 |
| gated_delta_net.py | Delta net attention | 实验性 |
| dispatcher.py | MoE token dispatch | ★★★ 需要 |

### Optimizer Primitives

★★★★ **distopt + FSDP2 双路径**:

```
primitive/optimizers/
  megatron_wrap.py     ← Megatron-Core distopt 包装
  fsdp2/
    adamw.py           ← FSDP2 AdamW 实现
    grad_clip.py       ← FSDP2 梯度裁剪 (multi-rank offload fix!)
    optimizer.py       ← FSDP2 optimizer 主逻辑
    state.py           ← FSDP2 optimizer state 管理
    wrap.py            ← FSDP2 optimizer 包装
```

★★★★ **重要**: `grad_clip.py` 修复了 FSDP2 offload 在 multi-rank 下的梯度裁剪问题 (commit cabdee8)。这表明 Lite 在 FSDP2 offload 上已经做了实际 bugfix。

### Checkpoint Primitives

```
primitive/ckpt/
  load/save training checkpoint
  HF safetensors 格式支持
```

---

## 十一、LoRA Adapter 实现细节

★★★★★ **Qwen3 MoE LoRA — 目前最精细的 Megatron-style LoRA 实现**

### lora_adapter.py 核心功能

| 函数 | 功能 |
|------|------|
| `_unwrap_model` | 解包分布式模型 chunk (DDP/FSDP wrapper) |
| `_iter_qwen_chunks` | 遍历所有 pipeline stage chunk |
| `_all_gather_cat` | 跨 TP rank gather LoRA 权重用于导出 |
| `_select_tp_replicated` | 选择 TP 复制部分 (非分片) |
| PEFT import | 加载 HF PEFT safetensors LoRA 权重到 Megatron 分片模型 |
| PEFT export | 导出 Megatron 分片 LoRA 权重为 HF PEFT safetensors |

★★★★ **PEFT/Mint-compatible**: 遵循 Mind Lab 的 PEFT/Mint 格式, 确保与 HF PEFT 库互操作。

### LinearLoRA 的 TP-aware 内存优化

★★★★★ `_SequenceParallelRankPartitionedLoRA` — 自定义 autograd Function:

- **问题**: 普通 all-gather + matmul 组合在 backward 中保存全序列 parallel gathered input → 巨大内存
- **解决**: 仅保存 local input + LoRA 权重, backward 时重算 gather/matmul
- **效果**: QKV LoRA 的 gathered input 远大于 low-rank hidden → 重算比存储更高效
- **条件**: 仅在 SP + rank_partitioned_a + 无 dropout + 非 row_parallel_output 时触发

---

## 十二、设计哲学与未来方向

★★★★ **Lite 的核心设计哲学**

### 1. 窄合约 (Narrow Contract)

★★★★★ 传统 Megatron-LM: recipe-level 代码, 跨越初始化/分布式/模型/数据/优化器
★★★★★ Lite: `RuntimeConfig → create_runtime → build_model` — 3 个合约边界

**意义**: 外部系统 (verl/rLLM/custom) 只需对接 3 个接口, 不需穿越 Megatron 内部 5+ 层。

### 2. Agentic-native

★★★ Skills 系统不是为特定 agent (Codex/Claude) 设计, 而是 agent-agnostic 操作合约。

**Schema 模型**: 每个 skill = function signature + imports + calls + inputs + outputs + exits

**意义**: AI agent 可以通过 skill schema 路由到正确的操作, 无需阅读完整设计文档。

### 3. Validation-first

★★★★★ 每个 change 需要:
- Correctness pair (deterministic bitwise 验证)
- SHA256 fingerprint (post-step weight + eval logits)
- Performance pair (paired benchmark, identical input stream)

**意义**: 这是可复现验证的最高标准 — 比 vLLM/verl 的 CI 测试更严格。

### 未来路线 (基于 PR 评论推断)

★★★★ **分步合入 main 的可能顺序**:

1. Skeleton: docs + API contracts + skills + toy model
2. Qwen3 MoE lite model implementation
3. Qwen3.5 MoE lite model implementation
4. Checkpoint save/load/export
5. **LoRA as first-class feature** (Mind Lab 验证)
6. FSDP2 optimizer primitives (已实现, 等合入)
7. verl integration (已实现, 等合入)
8. Hybrid model implementations
9. Bridge model/runtime implementations
10. 更多模型: dense Qwen3, Llama, etc.

---

## 十三、关键发现总结

★★★★★ **对 RTX 4090 最关键的 5 个发现**:

### 1. ★★★★★ LoRA 基础设施已完整

- LinearLoRA: TP-aware sharded LoRA (SP/CP/row/col parallel 全支持)
- GroupedLinearLoRA: MoE per-expert LoRA
- freeze_non_lora_params: base frozen, adapter trainable
- Qwen3 MoE PEFT import/export: Mind Lab 兘容
- **缺失**: 仅未作为正式特性发布, 需分步合入

### 2. ★★★★★ verl GRPO 集成已实现

- mlite_engine.py: 完整 BaseEngine 实现
- GRPO 脚本: KL disabled + offload + vanilla policy loss
- SFT 脚本: 参数/优化器 offload 支持
- **RTX 4090 路径**: verl + mlite actor + vLLM rollout

### 3. ★★★★★ 单 GPU 正确性已验证

- 1x GPU deterministic run: bitwise 完全一致
- max_loss_abs=0.0, max_grad_norm_abs=0.0
- **但**: 验证在 H100 上, 非在 RTX 4090 上

### 4. ★★★★ 内存优势 -20.3%

- mlite 14.324 GB vs mbridge 17.987 GB
- 节省 3.663 GB on 8x H100
- **RTX 4090 意义**: 如果 8-GPU 有 -20.3% 内存, 单 GPU 的内存效率可能类似

### 5. ★★★★★ FSDP2 offload 已 bugfix

- grad_clip multi-rank offload fix (commit cabdee8)
- offload_fraction=1.0 optimizer CPU 常驻
- **RTX 4090 意义**: offload 栈已验证, 但速度极慢

---

## 十四、RTX 4090 行动建议

★★★★ **基于 Megatron Lite 发现的 RTX 4090 行动**:

### 立即可做 (0-2 周)

1. **Clone dev 分支, 试运行 1-GPU correctness**
   - `PYTHONPATH=$PWD/experimental/lite:$PYTHONPATH`
   - 在 RTX 4090 上验证 deterministic correctness pair
   - 观察是否 SM89 有问题

2. **评估 LoRA 原语独立可用性**
   - `primitive/modules/lora.py` 是否可独立导入
   - 是否可在非 Lite 运行时中使用 LinearLoRA

3. **评估 verl_mlite 引擎在 RTX 4090 的可行性**
   - dry-run GRPO 脚本
   - 确认 vLLM rollout + mlite actor 组合

### 中期 (2-8 周)

4. **添加小模型 protocol**
   - Qwen3-1.7B dense 或 Llama-3.2-1B/3B
   - 验证 LoRA + offload 在 24GB VRAM 上的可行性

5. **LoRA + offload 组合 benchmark**
   - LoRA rank=8/16 + optimizer offload
   - 测量 RTX 4090 可训练模型尺寸

6. **跟踪 Mind Lab LoRA 验证进展**
   - Mind Lab 承诺验证 LoRA on Lite
   - 关注他们的 PR/issue

### 长期 (8+ 周)

7. **贡献小模型 protocol 到 Megatron Lite**
   - 1-7B dense model implementations
   - 这是 RTX 4090 最需要的补充

8. **RTX 4090 单 GPU LoRA + offload 完整 benchmark**
   - 端到端 GRPO 训练验证
   - 与 rLLM Tinker 对比

---

## 十五、与已有笔记的关联

★★★★ **更新 7-framework 分析**:

- **Megatron-LM**: 从 9 notes → 需增加 Lite note, 重评排名 #3 → #2.5 (Lite 改善前景)
- **verl**: 需增加 mlite_engine 交互分析 (verl GRPO + mlite actor)
- **rLLM**: Tinker 仍是 #1, 但 verl+Lite 是有力的 #2.5
- **PyTorch FSDP2**: Lite 已有 FSDP2 primitive + offload fix → 交叉验证

### 关闭的 coverage gap

★★★★★ 之前笔记仅提到 "Megatron Lite (#4885): lightweight runtime → LoRA excluded but planned → potential RTX 4090 future path"。现在有了完整的 200+ 行深度分析, 覆盖了:
- 架构 (3-layer separation)
- LoRA 实际基础设施 (远超"排除" — 代码完整)
- verl 集成 (已实现 GRPO 脚本)
- 性能 benchmark (8x H100 结果)
- Skills 系统 (agentic-native)
- 命名争议 (lite vs megatron-compose)
- RTX 4090 可行性评估 (短期不可行, 中期可能, 长期非常可行)

---

## 参考

- PR: https://github.com/NVIDIA/Megatron-LM/pull/4885
- 作者 fork: https://github.com/ISEEKYAN/mlite
- Mind Lab 评论: PR #4885 comment #2 (19 reactions, LoRA focus)
- sbhavani 评审: PR #4885 comments #3/#5/#7 (命名/拆分争议)
- Benchmark README: experimental/lite/examples/bench/README.md
- verl 集成: experimental/lite/examples/verl/README.md
- LoRA 原语: experimental/lite/megatron/lite/primitive/modules/lora.py
- LoRA adapter: experimental/lite/megatron/lite/model/qwen3_moe/lite/lora_adapter.py
- FSDP2 optimizer: experimental/lite/megatron/lite/primitive/optimizers/fsdp2/
- Design docs: experimental/lite/docs/ (architecture/runtime/models/porting)
- Skills: experimental/lite/skills/ (6 namespace, 30+ skill)

---

*笔记日期: 2026-06-16 | 来源: NVIDIA/Megatron-LM PR #4885 + ISEEKYAN/mlite fork*
