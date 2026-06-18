# verl PR #6791 — DSv4/GLM5/KimiK2.5 via Megatron Lite 深度阅读

> 2026-06-18 | volcengine/verl PR #6791 | MERGED June 18, 00:12 UTC
> 作者: ISEEKYAN (NVIDIA) | +864/-9 | 9 files | merge_commit: 7f16719
> 评审: 1 review comment (gemini-code-assist naming check), 0 issue comments

---

## 一、PR 概要: 文档 + 启动脚本, 不含运行时代码

★★★★★ **核心定位**: 纯文档 + 示例脚本 PR, **零运行时代码变更**

PR #6791 不修改 verl 的训练/推理引擎代码。它将已有的 mlite 运行时集成 (来自 Megatron-LM PR #4885, 已合入 dev 分支) 以**用户文档和 launch 脚本的形式暴露给 verl 社区**。

### 变更文件清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `docs/advance/megatron_lite_backend.rst` | NEW (81行) | Megatron Lite 后端文档 |
| `docs/index.rst` | MOD (+1行) | 添加 mlite 文档链接 |
| `examples/README.md` | MOD (+3/-3行) | 添加 `megatron_lite` 为合法 train-backend |
| `examples/grpo_trainer/run_deepseek_v4_megatron_lite.sh` | NEW (191行) | DSv4 GRPO launcher |
| `examples/grpo_trainer/run_kimi_k2_6_glm5_1_megatron_lite.sh` | NEW (209行) | Kimi K2.6 / GLM 5.1 GRPO launcher |
| `examples/grpo_trainer/run_qwen3_5_35b_megatron_lite.sh` | NEW (192行) | Qwen3.5-35B-A3B GRPO launcher |
| `examples/sft/gsm8k/run_deepseek_v4_megatron_lite.sh` | NEW (170行) | DSv4 SFT launcher |
| `tests/special_sanity/check_example_naming.py` | MOD (+13/-6行) | 添加 `megatron_lite` 到 ALLOWED_BACKENDS |
| `tests/special_sanity/test_check_example_naming.py` | MOD (+4行) | 测试覆盖 mlite 命名 |

★★★★★ **命名约定更新**: `megatron_lite` 正式加入 verl 的合法训练后端列表 (`ALLOWED_BACKENDS`), 与 `fsdp`, `fsdp2`, `megatron`, `mindspeed`, `automodel`, `veomni` 并列。这标志着 mlite 从"实验性"变为"verl 官方认可的后端"。

---

## 二、Megatron Lite 是什么 (从 PR #6791 的文档视角)

★★★★★ **文档定义** (megatron_lite_backend.rst):

> "Megatron Lite (`mlite`) is Megatron's experimental, agent-friendly training
> path for work that needs to move quickly. It is optimized for fast iteration,
> small reviewable changes, and agentic development."

★★★★★ **关键架构原则** (文档明确):

1. **运行时代码在 verl 树之外** — mlite 通过 `MLITE_ROOT` / `PYTHONPATH` 加载
2. **verl 只做编排** — 不含 mlite 运行时实现代码
3. **自定义扩展放在用户代码路径** — 通过 `MLITE_ROOT` 或 `PYTHONPATH` 添加
4. **上游路径**: NVIDIA/Megatron-LM dev 分支的 `experimental/lite`

★★★★★ **dist_opt 正确性声明** (文档核心段落):

> "In deterministic runs, the `mlite` path has been validated against the
> Megatron-Core distributed optimizer path with **bitwise-aligned loss and
> gradient norms**, and its step time / throughput are also aligned with the
> Core path."

★★★★★ 这与 Megatron-LM PR #4885 的 benchmark 数据一致: max_loss_abs=0.0, max_grad_norm_abs=0.0。

---

## 三、DSv4 (DeepSeek-V4) 支持路径

★★★★★★★ **这是 PR #6791 的旗舰功能** — 首次将 DSv4 模型引入 verl GRPO 训练生态

### DSv4 架构约束

★★★★★★★ **DSv4 固定 TP=1/ETP=1**:

> "DS4 is fixed to TP1/ETP1. The architecture does not support TP/ETP
> sharding, and there is no plan to support it."

★★★★★ 这意味着:
- DSv4 **不支持 tensor parallel** — 不能跨 GPU 切分单个层
- 只能通过 **PP (pipeline parallel) + EP (expert parallel) + CP (context parallel)** 分布
- DSv4 是 "MoE-only-parallelism" 架构 — 与传统 TP-shardable 模型不同

### DSv4 两个变体

| 变体 | 节点数 | GPU数 | PP | EP | CP | Mesh公式 |
|------|--------|-------|----|----|-----|----------|
| **flash** | 16 | 128 | 4 | 8 | 4 | 128/4=32=TP1*EP8*DP4 |
| **pro** | 64 | 512 | 8 | 16 | 4 | 512/8=64=TP1*EP16*DP4 |

★★★★ Mesh 公式遵循 Megatron Lite 的 `ngpu / pp = tp * ep * dp = etp * ep * edp`

### DSv4 DSA Kernel 依赖

★★★★★ **文档明确指出 DSA kernel 需求**:

> "DeepSeek-V4 uses fused DSA kernels on Hopper and Blackwell GPUs. The
> critical DSA-only dependencies are `nvidia-cutlass-dsl==4.5.2` and
> `nvidia-cudnn-frontend`."

| GPU | cudnn-frontend 需求 |
|-----|---------------------|
| **Blackwell** | Release 1.24.1 **足够** |
| **Hopper (H100)** | 需要 **develop 分支 build** + `IndexerForwardSm90` 支持 |

★★★★★★★ **RTX 4090 影响**: DSA kernels 需要 SM90+ (Hopper) 或 SM100+ (Blackwell)。RTX 4090 是 SM89 (Ada Lovelace), **不支持 DSA kernels**。这意味着 DSv4 on RTX 4090 需要:
- 不使用 DSA kernel → 回退到标准 attention
- 或者使用 Qwen3.5-35B-A3B (不需要 DSA) 代替 DSv4

### DSv4 GRPO 脚本关键配置

```bash
# 核心配置参数
OPTIMIZER=fsdp2              # FSDP2 optimizer wrapper (默认, 内存更低)
ALL_OFFLOAD=True             # 全部 offload: param + optimizer + grad
ACTOR_LR=1e-6                # GRPO 标准 LR
CLIP_RATIO_LOW=0.2           # CPPO-style clip range
CLIP_RATIO_HIGH=0.28         # 上界 clip
CLIP_RATIO_C=10.0            # DAPO/CPPO clip C
ROLLOUT_TP=2                 # vLLM rollout 使用 TP=2
ROLLOUT_N=16                 # 每个 prompt 生成 16 个 response
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
algorithm.kl_ctrl.kl_coef=0.0
critic.enable=False           # 纯 GRPO, 无 critic
```

★★★★★★★ **ALL_OFFLOAD=True** 是所有 launcher 的默认值:
- `param_offload=True` → 训练时 params 在 GPU, idle 时移回 CPU
- `optimizer_offload=True` → optimizer state 全在 CPU (offload_fraction=1.0)
- `grad_offload=True` → 梯度 offload 到 CPU

★★★★★★★ 这是 RTX 4090 最关键的配置模式! 全 offload + fsdp2 = 最低 GPU 内存压力

---

## 四、Kimi K2.6 / GLM 5.1 支持路径

★★★★ **256-GPU GRPO 默认配置**

| 模型 | 节点数 | GPU数 | PP | EP | CP | TP | ETP | DP | EDP |
|------|--------|-------|----|----|-----|----|-----|----|----|
| kimi_k2_6 | 32 | 256 | 8 | 8 | 8 | 1 | 1 | 4 | 4 |
| glm5_1 | 32 | 256 | 8 | 8 | 8 | 1 | 1 | 4 | 4 |

★★★★ **GLM 5.1 DSA kernel 需求**: 与 DSv4 相同 — 需要 `nvidia-cutlass-dsl==4.5.2` 和 `nvidia-cudnn-frontend`。Hopper 需要 develop 分支 build, Blackwell 使用 1.24.1 release。

★★★★★★★ **RTX 4090 影响**: Kimi K2.6 和 GLM 5.1 都是超大规模 MoE 模型, **完全不可能在 RTX 4090 单 GPU 上训练**。256 GPU 是最小推荐规模。

---

## 五、Qwen3.5-35B-A3B 支持路径 — ★★★★★ RTX 4090 最相关的发现

★★★★★★★ **这是 PR #6791 对 RTX 4090 最重要的部分!**

### Qwen3.5 Launcher 默认: 单节点 8 GPU!

```bash
NNODES=1                     # ★★★★★ 单节点!
NDEVICES_PER_NODE=8          # 8 GPU (单机)
TP=1
PP=1                         # ★★★★★ 无 pipeline parallel!
EP=8                         # 8 专家并行
CP=8                         # 8 context parallel
ETP=1
OPTIMIZER=fsdp2              # FSDP2 wrapper
ALL_OFFLOAD=True             # 全部 offload
```

★★★★★★★ **Mesh accounting**:
```
ngpu / pp = 8 / 1 = 8 = tp * ep * dp = 1 * 8 * 1 → DP=1
                  = etp * ep * edp = 1 * 8 * 1 → EDP=1
```

★★★★★★★ **DP=1 + EDP=1** → 无数据并行, 每个处理组只有 1 个 GPU 做数据并行副本 → 这是**最小可行配置**!

### Qwen3.5 Allgather CP Path + FLA 5.0 依赖

★★★★★ **脚本注释明确说明**:

> "Qwen3.5 uses the Megatron Lite allgather CP path. CP is intentionally kept
> as an explicit mlite knob below; the default single-node run uses CP8. This
> path depends on FLA (flash-linear-attention) 5.0 in the runtime environment."

★★★★★★★ **FLA 5.0** = `flash-linear-attention` — https://github.com/fla-org/flash-linear-attention

★★★★ **Allgather CP** vs **Zigzag CP**:
- Zigzag CP: ring attention style, 跨 CP rank 的序列交错切分 → 需要通信 overlap
- Allgather CP: 先 allgather 输入, 然后本地计算 → 更简单但内存更高
- Qwen3.5 使用 allgather → **单节点更友好** (通信模式更简单)

### Qwen3.5 GRPO 关键配置

```bash
TRAIN_BATCH_SIZE=32           # 比 DSv4 (128) 小 — 适配单节点
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=1
PPO_MAX_TOKEN_LEN_PER_GPU=4096  # ★★★★★ 动态 batch size 限制
ROLLOUT_TP=8                 # 8 GPU 全用于 rollout TP
ROLLOUT_GPU_MEM_UTIL=0.6     # 60% GPU memory for rollout
ROLLOUT_N=5                  # 每个 prompt 5 个 response
ACTOR_LR=1e-6
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False  # 无 KL penalty
critic.enable=False               # 纯 GRPO
```

★★★★★★★ **ROLLOUT_TP=8**: 推理时所有 8 GPU 做 tensor parallel → 模型分片到 8 GPU 上推理 → 训练时用 EP=8 专家并行 + CP=8 context 并行 → **不同并行策略用于训练和推理**

---

## 六、DSv4 支持路径分析 — 从训练到推理的完整流程

★★★★★★★ **verl + mlite 的 DSv4 GRPO 数据流**:

```
用户数据 → verl main_ppo trainer (orchestrator)
  ├─ actor: mlite_actor (Megatron Lite engine)
  │   ├─ initialize():
  │   │   → create_runtime(RuntimeConfig(backend="mlite", hf_path=DSv4))
  │   │   → runtime.build_model() → handle (model + optimizer + parallel state)
  │   │   → self.to(device="cpu") → 全 offload → params/optimizer/grad to CPU
  │   ├─ train_mode():
  │   │   → runtime.train_mode(handle) → CPU→GPU migration
  │   ├─ forward_backward_batch():
  │   │   → prepare_micro_batches (verl) → micro batch 分割
  │   │   → runtime.forward_backward(handle, loss_fn=...)
  │   │   → PackedBatch → THD/no-padding format
  │   │   → Megatron Lite native forward + loss + backward
  │   │   → MTP loss as extra metric (if enabled)
  │   ├─ optimizer_step():
  │   │   → runtime.optimizer_step(handle) → FSDP2/dist_opt optimizer
  │   │   → offload hooks → GPU→CPU migration
  │   └─ get_per_tensor_param():
  │       → runtime.export_weights(handle) → vLLM-compatible format
  │       → Qwen3.5: target="vllm" → vLLM specific export
  ├─ rollout: vLLM async engine
  │   → tensor_model_parallel_size=2 (DSv4) or 8 (Qwen3.5)
  │   → weight sync: mlite export → vLLM load
  ├─ reward: external reward function
  └─ no reference model (KL disabled, critic disabled)
```

★★★★★★★ **关键数据流节点**:
1. **初始化**: MLite 运行时构建 → 立即全 offload 到 CPU
2. **训练步**: CPU→GPU (train_mode) → forward/backward → optimizer step → GPU→CPU
3. **权重同步**: mlite export → vLLM format → vLLM load → rollout inference
4. **GRPO**: 无 KL, 无 critic, 无 ref model → 最大内存节省

---

## 七、Megatron Lite vs Full Megatron-LM — 从 #6791 的视角

★★★★★ **文档明确对比** (megatron_lite_backend.rst):

### 架构差异

| 维度 | Full Megatron-LM | Megatron Lite (mlite) |
|------|------------------|---------------------|
| 入口 | 全局脚本 + recipe | `RuntimeConfig → create_runtime → build_model` |
| 运行时 | 跨越 5+ 层的全局初始化 | 窄合约 3 层 (runtime → model → primitive) |
| 模型代码 | Megatron-Bridge wrapper | **原生 lite 实现**, 不是 wrapper |
| 代码位置 | verl 树内 (`verl/workers/engine/megatron/`) | **verl 树外** (`MLITE_ROOT/experimental/lite`) |
| 正确性 | tolerance-based CI | **bitwise exact** (loss=0.0, grad=0.0) |
| 优化器 | Megatron-Core distopt only | **distopt + FSDP2** 双路径 |
| Agentic | 无 | skills 系统 (6 namespace, 30+ skill) |
| 扩展 | 修改 verl + Megatron 源码 | 用户代码路径, 不改 verl/Megatron |

★★★★★★★ **最重要的差异: 代码位置**

> "The verl integration intentionally keeps the backend glue outside this
> repository. The `mlite` checkout provides `megatron.lite` and the
> `verl_mlite` launcher/config package used by the example scripts here."

这意味着:
- verl 主仓库不包含 mlite 运行时代码
- mlite 通过 `PYTHONPATH` 加载 → 版本独立管理
- 用户可以升级 mlite 而不升级 verl → **解耦部署**

### 安装方式

★★★★ 两种安装路径:

**路径 A (pip install)**:
```bash
git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
pip install -e Megatron-LM/experimental/lite/examples/verl
```

**路径 B (PYTHONPATH)**:
```bash
MLITE_ROOT=/path/to/mlite  # 或 ISEEKYAN/mlite fork
# Scripts 自动添加:
#   $MLITE_ROOT/experimental/lite
#   $MLITE_ROOT/experimental/lite/examples/verl
# 到 PYTHONPATH
```

★★★★★ **ISEEKYAN/mlite fork**: PR 注释说明当前 launcher 跟踪 ISEEKYAN 的活跃分支, 直到最新 mlite 变更合入上游。这暗示 mlite 仍在快速迭代。

### verl_mlite 包结构 (外部包)

```
verl_mlite/
  __init__.py             ← 空初始化
  compat.py               ← verl API 版本兼容层 + runtime patches
  launch.py               ← torchrun 入口 (apply patches + import engine)
  config/
    __init__.py
    actor/
      mlite_actor.yaml    ← Hydra actor config: strategy=mlite
    engine/
      mlite.yaml          ← Hydra engine config: engine=mlite
  engine/
    __init__.py
    config.py             ← MegatronLiteEngineConfig dataclass
    mlite_engine.py       ← ★★★★★ 842行核心引擎 (BaseEngine subclass)
```

★★★★★ **mlite_engine.py 是实际运行时引擎** — 842行, 注册为 `@EngineRegistry.register(model_type="language_model", backend="mlite", device="cuda")`

---

## 八、mlite_engine.py 核心架构更新 (vs 之前 #4885 版本)

★★★★★ **PR #6791 没有修改 mlite_engine.py**, 但我们可以从当前版本分析新特性

### 关键变更 (vs 2026-06-16 首次阅读的 779行版本)

★★★★ 当前版本 842行 (增加 63行) — 主要新增:

1. **_forward_backward_batch_with_runtime() 方法** — PP>1 路径重构:
   - 之前: PP>1 用手动 micro-batch loop
   - 现在: `self.runtime.forward_backward(handle, iter(runtime_batches), loss_fn=..., num_microbatches=...)` — 统一委托给运行时
   - 支持 `forward_only` 参数 → rollout 推理路径

2. **_make_runtime_batch() → PackedBatch** — 新的 THD 数据格式:
   - `PackedBatch(input_ids, labels, loss_mask, seq_lens)` — Megatron Lite 原生数据格式
   - `input_ids.values().contiguous()` — jagged NestedTensor → flat packed
   - `seq_lens = input_ids.offsets().diff()` — 序列长度从 offsets 计算

3. **_make_runtime_loss_context() → LossContext** — 新的 loss 上下文:
   - `LossContext(temperature, calculate_entropy, return_log_probs, loss_scale, source_batch)`
   - `loss_scale = dp_size * num_micro_batches / batch_num_tokens` — 动态 loss scaling

4. **_build_verl_model_output() → unpack log_probs** — THD 输出解包:
   - `unpack(self.module, runtime_batch, log_probs)` — 反向 THD pack
   - 需要 model protocol 的 `unpack_forward_output` 方法

5. **MTP loss support** → `mtp_loss` 作为额外 metric 处理

★★★★★★★ **这些变更表明 mlite_engine.py 已从 "PP=1 only" 进化到 "PP>=1 general"** — 支持完整的 Megatron pipeline parallel!

### Offload 默认行为

★★★★★★★ **所有 PR #6791 的 launcher 都默认 ALL_OFFLOAD=True**:

```python
# mlite_engine.py initialize() 末尾:
self.to(
    device="cpu",
    model=self.is_param_offload_enabled,     # True → params to CPU
    optimizer=self.is_optimizer_offload_enabled,  # True → optimizer to CPU
    grad=self.is_param_offload_enabled,      # True → grad to CPU
)
```

★★★★★★★ **offload_fraction=1.0 自动设置**:
```python
# _build_mlite_optimizer_config():
if offload_fraction is None and self.is_optimizer_offload_enabled:
    offload_fraction = 1.0  # 100% optimizer state on CPU
```

---

## 九、RTX 4090 深度可行性分析

★★★★★★★ **PR #6791 对 RTX 4090 的影响分级评估**

### Qwen3.5-35B-A3B — ★★★★★ 最相关但不直接可行

★★★★★ **正面因素**:
- Launcher 默认 **NNODES=1, PP=1** → 单节点设计意图明确
- ALL_OFFLOAD=True → 最大内存节省
- FSDP2 optimizer → offload_fraction=1.0 → optimizer 全在 CPU
- 无 KL / 无 critic / 无 ref model → 三重内存节省
- Allgather CP (而非 zigzag) → 单节点通信更简单
- Bitwise 正确性已验证 → 结果可信

★★★★★★★ **负面因素**:
- **Qwen3.5-35B-A3B 是 35B 总参数, 3B active** — 即使只有 3B active, **全部 35B 专家权重必须加载**
- EP=8 需要 8 GPU → RTX 4090 只有 1 GPU → **EP=1 = 所有专家在单 GPU**
- 单 GPU EP=1: 全部 128 个专家权重 ~30GB → **超过 24GB VRAM**
- FLA 5.0 依赖 — flash-linear-attention 是否支持 SM89 未验证
- No LoRA in mlite → 全参数训练 → optimizer state 巨大
- 单 GPU CP=1 (无法做 context parallel) → 全序列在单 GPU

★★★★★ **Qwen3.5-35B-A3B on RTX 4090 内存估算**:
```
模型权重:    ~35B * 2 bytes (bf16) = ~70 GB → 远超 24GB
即使 LoRA:
  base frozen:  ~70 GB → 仍然远超 24GB

只有 EP=1 + LoRA + extreme offload:
  GPU 训练时:  活跃专家参数 + LoRA adapter + activation ~8-12GB?
  CPU 常驻:    全部 base weights + optimizer state

→ ★★★★★ EP=1 + LoRA + offload 组合下 GPU 可能够用,
  但 CPU RAM 需要 >70GB → 32GB 系统内存不够!
```

★★★★★★★ **结论**: Qwen3.5-35B-A3B on RTX 4090 **目前不可行** — 需要更多系统 RAM (64GB+) 和 LoRA 集成才能尝试。

### DSv4 / GLM 5.1 / Kimi K2.6 — ★★★★★ 完全不可行

★★★★★★★ 这些模型都需要:
- 128+ GPU (flash) 或 256+ GPU (pro/kimi/glm5)
- DSA kernels (SM90+, RTX 4090 SM89 不支持)
- 超大规模 MoE → 单 GPU 绝对 OOM

### 间接正面影响

★★★★★★★ **PR #6791 为 RTX 4090 开启了 mlite 训练路径**:

1. **官方认可**: `megatron_lite` 加入 ALLOWED_BACKENDS → verl 官方后端
2. **文档完善**: megatron_lite_backend.rst → 用户可以参考配置
3. **单节点设计**: Qwen3.5 launcher NNODES=1 → 设计意图包含小规模部署
4. **全 offload 默认**: ALL_OFFLOAD=True → 面向内存受限环境的设计
5. **fsdp2 默认**: OPTIMIZER=fsdp2 → 比 dist_opt 内存更低

★★★★★★★ **RTX 4090 的 mlite 路径需要 3 个前提** (与之前分析一致):
1. **LoRA 集成** → mlite_engine.py 需要添加 LoRA adapter injection + freeze + export
2. **小模型 protocol** → 需要 <8B dense 模型 (非 35B MoE)
3. **FLA SM89 验证** → flash-linear-attention 是否在 Ada Lovelace 上可用

---

## 十、DSv4 DSA Kernel 对 SM89 的影响

★★★★★★★ **DSA (DeepSeek Attention) = fused kernel on SM90+**

PR 文档明确说明:

> "The `nvidia-cudnn-frontend` 1.24.1 release is sufficient for Blackwell,
> while Hopper still needs a develop-branch build with `IndexerForwardSm90`
> support."

★★★★★ **SM89 (RTX 4090 Ada Lovelace) 的 DSA 状态**:
- DSA kernels 需要 SM90 (Hopper) 或 SM100 (Blackwell) 的硬件特性
- SM89 **不支持** DSA fused kernels
- DSv4 和 GLM 5.1 **不能在 RTX 4090 上使用 DSA**
- ★★★★★ 但 mlite 可以配置 `attention_backend_override=flash` → 回退到 FlashAttention-2 (SM89 支持!)

★★★★★ **所有 launcher 都设置**: `actor_rollout_ref.actor.engine.attention_backend_override=flash`

这意味着 mlite actor 使用 FlashAttention-2 而非 DSA → **RTX 4090 推理路径可用** (只是 slower than DSA on H100)

★★★★★★★ **但 DSv4/GLM5 模型本身太大 → 即使 attention 回退到 Flash, 模型参数仍超 24GB**

---

## 十一、OPTIMIZER 双路径详解

★★★★★ **所有 launcher 提供 fsdp2 和 dist_opt 两种优化器路径**

### FSDP2 Optimizer (默认)

★★★★★★★ **选择理由**: "lower memory pressure"

```bash
OPTIMIZER=fsdp2
+actor_rollout_ref.actor.engine.impl_cfg.optimizer=fsdp2
```

★★★★ FSDP2 optimizer 在 mlite 中的实现:
- `primitive/optimizers/fsdp2/` — 完整 FSDP2 AdamW + grad_clip + state + wrap
- offload_fraction=1.0 → optimizer update state 全在 CPU
- 比 dist_opt 内存更低 (FSDP2 sharding + CPU offload)
- ★★★★★ 单 GPU 无 sharding 收益 → 仅 offload 功能有用

### dist_opt (Megatron-Core 分布式优化器)

★★★★ **选择条件**: "When using dist_opt, prefer a larger PP*EP mesh to reduce per-rank model and optimizer memory pressure and avoid OOM."

```bash
OPTIMIZER=dist_opt
+actor_rollout_ref.actor.engine.impl_cfg.optimizer=dist_opt
```

★★★★★ dist_opt 特点:
- Megatron-Core 的标准分布式优化器
- **bitwise 正确性已验证** → 与 Core 路径完全一致
- **不支持 CPU offload** → 更高 GPU 内存压力
- ★★★★★ 适合多 GPU (大 PP*EP mesh → 分散 optimizer state)

★★★★★★★ **RTX 4090 必须使用 fsdp2** → dist_opt 无 offload → 单 GPU OOM

---

## 十二、verl_mlite Engine Registration 分析

★★★★★ **mlite_actor.yaml 的 Hydra 配置**:

```yaml
defaults:
  - /optim@optim: megatron
  - /engine@engine: mlite
  - actor
  - _self_

_target_: verl.workers.config.ActorConfig

strategy: mlite
```

★★★★★ **MegatronLiteEngineConfig dataclass** (config.py):

```python
@dataclass
class MegatronLiteEngineConfig(EngineConfig):
    strategy: str = "mlite"
    custom_backend_module: str | None = "verl_mlite.engine.mlite_engine"
    model_name: str = "auto"    # 自动从 HF config 推断
    impl: str = "lite"

    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1

    attention_backend_override: str | None = "flash"
    export_dtype: str | None = "bfloat16"
    load_hf_weights: bool = True
    impl_cfg: dict[str, Any] = field(default_factory=dict)
```

★★★★★★★ **model_name="auto"**: 自动推断 → `resolve_model_type_from_hf(hf_config)` → 读 HF config.json 的 model_type → 映射到 mlite 注册名

★★★★ **当前注册模型** (registry.py):
- `qwen3` → qwen3_moe (HF: qwen3_moe, qwen2_moe)
- `qwen3_moe` → 同上
- `qwen3_5` → qwen3_5 (HF: qwen3_5_moe)

★★★★★★★ **DSv4, GLM 5.1, Kimi K2.6 未在 registry 中注册!**

这意味着 PR #6791 的 launcher 脚本引用了尚未注册的模型 — 它们需要 mlite 运行时添加对应的 model protocol (在 ISEEKYAN/mlite 的活跃分支中, 尚未合入 NVIDIA/Megatron-LM dev)

★★★★★ **这解释了为什么 PR 注释说**: "This launcher currently tracks the submitter's active branch until the latest mlite changes merge upstream."

---

## 十三、compat.py — 运行时兼容层

★★★★★ **compat.py 的关键功能**:

1. **`load_verl_engine_api()`** — 动态加载 verl BaseEngine API:
   - 首选: `from verl.workers.engine.base import ...` — 正常包导入
   - 备选: `importlib.util.spec_from_file_location` — 文件级导入 (verl 不可安装时)
   - ★★★★★ **必须用包导入**: 文件级导入创建 duplicate EngineRegistry → "Unknown backend: mlite" 错误!

2. **`apply_runtime_patches()`** — transformers rope ignore_keys 兼容:
   - Patch `_check_received_keys` → 将 `ignore_keys` list → set (HF transformers bug)
   - 防止 HF safetensors 加载时类型错误

★★★★★ **这些 patches 说明 mlite 需要与 HF transformers 的特定版本兼容** — verl v0.8.0 是 REQUIRED_VERL.txt 的要求版本

---

## 十四、Launch Entry Point 差异

★★★★★ **SFT vs GRPO 的不同入口**:

### SFT (torchrun 直接启动)

```bash
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NDEVICES_PER_NODE}" \
    ...
    -m verl_mlite.launch verl.trainer.sft_trainer \
    "${DATA[@]}" "${MODEL[@]}" "${OPTIM[@]}" "${ENGINE[@]}" "${TRAINER[@]}"
```

★★★★★ **`verl_mlite.launch`** — 通过 `python -m verl_mlite.launch <module>`:
1. `apply_runtime_patches()` — 应用兼容性修补
2. `import verl_mlite.engine` — 触发 `@EngineRegistry.register` → 注册 mlite backend
3. `runpy.run_module(module)` — 启动目标训练模块

### GRPO (verl main_ppo 启动)

```bash
python3 -m verl.trainer.main_ppo \
    "${EXTRA[@]}" "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" \
    "${ACTOR[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}"
```

★★★★★ **EXTRA 配置**:
```bash
EXTRA=(
    hydra.searchpath=[pkg://verl_mlite.config]
)
```

★★★★★★★ **Hydra searchpath 扩展**: 将 `verl_mlite.config` 加入 Hydra 的配置搜索路径 → 使 `actor@actor_rollout_ref.actor=mlite_actor` 和 `engine=mlite` 可以被 Hydra 解析

★★★★★★★ **GRPO 入口通过 `actor@actor_rollout_ref.actor=mlite_actor`** → Hydra 在 `verl_mlite.config/actor/mlite_actor.yaml` 查找配置 → 该配置设置 `strategy: mlite` → verl trainer 在 EngineRegistry 中查找 `backend="mlite"` → 找到 MegatronLiteEngine → 使用 mlite 运行时

---

## 十五、关键发现总结

★★★★★★★ **PR #6791 的 7 个关键发现**

### 1. ★★★★★★★★ 纯文档/脚本 PR — 零运行时代码

- 不修改 verl 训练/推理引擎代码
- 仅添加文档 + 4 个 launcher + 命名约定更新
- mlite 运行时通过 MLITE_ROOT 外部加载

### 2. ★★★★★★★★ DSv4 固定 TP=1/ETP=1 — 无 tensor parallel 支持

- DSv4 不支持 TP/ETP sharding, 无计划支持
- 分布策略: PP + EP + CP only
- DSA kernels 需要 SM90+ → RTX 4090 不支持 DSA
- 但 mlite actor 用 `attention_backend_override=flash` → FlashAttention-2 回退可用

### 3. ★★★★★★★★ Qwen3.5-35B-A3B 默认 NNODES=1, PP=1 — 单节点设计意图

- 8 GPU, EP=8, CP=8, DP=1 — 最小可行配置
- Allgather CP path + FLA 5.0 依赖
- ALL_OFFLOAD=True + fsdp2 → 最大内存节省配置
- ★★★★★ 单 GPU RTX 4090 仍不可行 (35B 总参数 > 24GB)

### 4. ★★★★★★★★ ALL_OFFLOAD=True 是所有 launcher 的默认值

- param_offload=True → 训练时 GPU, idle 时 CPU
- optimizer_offload=True → optimizer state 100% CPU (offload_fraction=1.0)
- grad_offload=True → 梯度 CPU 常驻
- ★★★★★ 这是内存受限环境的标准配置模式

### 5. ★★★★★★★★ fsdp2 optimizer 是默认路径 — 比 dist_opt 内存更低

- OPTIMIZER=fsdp2 (默认) — FSDP2 AdamW wrapper + CPU offload
- OPTIMIZER=dist_opt (可选) — Megatron-Core distopt, bitwise 正确, 无 offload
- ★★★★★ RTX 4090 必须使用 fsdp2 + offload → dist_opt 单 GPU OOM

### 6. ★★★★★★★★ DSv4/GLM5/KimiK2.6 模型未在 mlite registry 中注册

- registry.py 只有 qwen3_moe 和 qwen3_5
- DSv4/GLM5/KimiK2.6 在 ISEEKYAN/mlite 活跃分支中, 尚未合入上游
- PR launcher 跟踪 ISEEKYAN 的活跃分支 → 需要等待上游合入

### 7. ★★★★★★★★ `megatron_lite` 加入 verl 官方 ALLOWED_BACKENDS

- 与 fsdp/fsdp2/megatron/mindspeed/automodel/veomni 并列
- 文档链接加入 docs/index.rst
- ★★★★★ mlite 从"实验性"变为"verl 官方认可的后端"

---

## 十六、RTX 4090 GRPO 排名更新 (含 #6791 影响)

★★★★★★★ **考虑 PR #6791 后的 RTX 4090 GRPO 框架排名**:

| 排名 | 框架 | 说明 | RTX 4090 可行性 |
|------|------|------|-----------------|
| #1 | verl CPPO+bypass | FSDP, external engine, INT8 KV, bypass_mode | 立即可用 |
| #1.5 | verl Tinker primitives | in-process-like, split training | 立即可用 |
| #2 | verl GRPO+bypass | FSDP, vLLM rollout, bypass_mode | 立即可用 |
| #2.5 | DeepSpeed ZeRO-2 | ZeRO-2+CPU_Adam, overlap_comm=False | 立即可用 |
| **#3** | **verl + Megatron Lite** | mlite actor, ALL_OFFLOAD, fsdp2, GRPO only | **中期 (需 LoRA + 小模型)** |
| #4 BLOCKED | rLLM Tinker | #605 GRPO grouping bug | BLOCKED |
| #5 | Megatron-LM core | singleton PG bug, no LoRA, MoE crash | 不可行 |

★★★★★★★ **#6791 对排名的影响**: mlite 从 #2.5 → #3 — 因为:
1. 排名下降: CPPO+bypass (#1) 已立即可用 → mlite 需要等待 LoRA
2. 正面: verl 官方认可 → 文档完善 → 路径清晰
3. 仍需: LoRA 集成 + 小模型 protocol + FLA SM89 验证

---

## 十七、与已有笔记的关联更新

★★★★ **需更新的已有笔记**:

1. **megatron-lite-reading.md** (797行): 增加 DSv4/GLM5/KimiK2.5 模型信息, DSA kernel 依赖, ALL_OFFLOAD 默认, fsdp2 optimizer 默认
2. **verl-mlite-integration-reading.md** (375行): 增加新版 mlite_engine.py (842行 vs 779行) 的 PP>1 支持, PackedBatch/LossContext 新数据格式, verl_mlite.launch 入口
3. **rtx4090-grpo-training-runbook.md**: 增加 mlite 路径章节 (中期可行, 需 LoRA)
4. **7-framework-developments-tracker**: verl 增加 #6791 MERGED, mlite 官方后端

---

## 参考

- verl PR #6791: https://github.com/volcengine/verl/pull/6791
- merge_commit: 7f167196ccdfdcc769631597326307ee144845f9
- 文档: verl/docs/advance/megatron_lite_backend.rst (81行)
- DSv4 GRPO: verl/examples/grpo_trainer/run_deepseek_v4_megatron_lite.sh (191行)
- Kimi/GLM5 GRPO: verl/examples/grpo_trainer/run_kimi_k2_6_glm5_1_megatron_lite.sh (209行)
- Qwen3.5 GRPO: verl/examples/grpo_trainer/run_qwen3_5_35b_megatron_lite.sh (192行)
- DSv4 SFT: verl/examples/sft/gsm8k/run_deepseek_v4_megatron_lite.sh (170行)
- mlite_engine.py: ISEEKYAN/mlite/experimental/lite/examples/verl/verl_mlite/engine/mlite_engine.py (842行)
- MegatronLiteEngineConfig: ISEEKYAN/mlite/experimental/lite/examples/verl/verl_mlite/engine/config.py
- mlite_actor.yaml: ISEEKYAN/mlite/experimental/lite/examples/verl/verl_mlite/config/actor/mlite_actor.yaml
- compat.py: ISEEKYAN/mlite/experimental/lite/examples/verl/verl_mlite/compat.py
- launch.py: ISEEKYAN/mlite/experimental/lite/examples/verl/verl_mlite/launch.py
- REQUIRED_VERL.txt: verl v0.8.0 requirement
- Megatron-LM dev branch: NVIDIA/Megatron-LM/experimental/lite/
- ISEEKYAN/mlite fork: https://github.com/ISEEKYAN/mlite (10 stars, public)
- Megatron Lite README: NVIDIA/Megatron-LM dev/experimental/lite/README.md
- Model registry: NVIDIA/Megatron-LM dev/experimental/lite/megatron/lite/model/registry.py
- FLA 5.0: https://github.com/fla-org/flash-linear-attention
- ISEEKYAN blog: https://iseekyan.github.io/posts/qwen35-long-sequence-moe-rl/ (not fetchable)
- 相关笔记: megatron-lite-reading.md, verl-mlite-integration-reading.md, rtx4090-grpo-training-runbook.md

---

*笔记日期: 2026-06-18 | 来源: volcengine/verl PR #6791 + ISEEKYAN/mlite fork + NVIDIA/Megatron-LM dev branch*
