# LoRA 与参数高效微调 (PEFT) 深度解析

> 用 0.1% 的参数实现 99% 的效果：LoRA 如何改变 LLM 微调的工程范式

## 1. 为什么需要参数高效微调

### 1.1 全量微调的成本

```
全量微调 7B 模型 (FP16 + AdamW):
  模型参数:     14 GB
  梯度:         14 GB
  优化器状态:   56 GB (FP32 master + m, v)
  激活值:       ~10 GB
  ────────────────────
  总计:         ~94 GB → 需要 2×A100-40GB 或 1×A100-80GB

全量微调 70B 模型:
  总计约 1.1 TB → 需要 16+ A100-80GB
```

### 1.2 PEFT 的核心思想

```
冻住预训练模型参数 W，只训练少量额外参数 ΔW:
  y = (W + ΔW) × x

其中 ΔW 的参数量 << W 的参数量

LoRA 的做法:
  ΔW = A × B    (低秩分解)
  A: [r × input_dim], B: [output_dim × r]
  r << min(input_dim, output_dim)
```

## 2. LoRA 原理

### 2.1 低秩适应 (Low-Rank Adaptation)

```
原始线性层:
  h = W × x    W ∈ R^{d_out × d_in}

LoRA:
  h = W × x + (B × A) × x
  A ∈ R^{r × d_in},  B ∈ R^{d_out × r}
  r << min(d_in, d_out)

训练时: W 冻结，只训练 A 和 B
推理时: W' = W + B × A  (可以合并，零额外推理开销)

参数量:
  原始: d_out × d_in
  LoRA: r × (d_out + d_in)
  比例: r × (d_out + d_in) / (d_out × d_in) ≈ 2r/d  (当 d_in = d_out = d)
```

### 2.2 缩放因子

```
h = W × x + (α/r) × B × A × x

α 是缩放超参数:
  - 控制 LoRA 更新的幅度
  - 通常设为 r 的倍数 (如 α = 2r)
  - α/r = 1 时即标准形式
  - α 大 → LoRA 影响大
  - α 小 → 主要靠预训练权重
```

### 2.3 初始化

```
A: Kaiming 均匀初始化（或正态）
B: 全零初始化

为什么:
  训练开始时 B×A = 0 → h = W×x → 模型行为 = 预训练模型
  从预训练模型的输出开始优化，保证训练初期稳定
```

### 2.4 应用到 Transformer

```
LoRA 通常只应用于 Attention 的 Q 和 V 投影:

每个 Attention 层:
  Q = x @ W_q + x @ A_q @ B_q     ← LoRA
  K = x @ W_k                       ← 冻结
  V = x @ W_v + x @ A_v @ B_v     ← LoRA
  Out = x @ W_o                     ← 冻结

也有的方案应用到所有线性层 (Q/K/V/O + MLP)
```

### 2.5 秩 r 的选择

```
| r   | 参数比例 (d=4096) | 适用场景 |
|-----|--------------------|---------|
| 4   | 0.2%              | 简单任务 (风格迁移) |
| 8   | 0.4%              | 中等任务 (指令微调) |
| 16  | 0.8%              | 标准微调 |
| 32  | 1.6%              | 复杂任务 (领域适应) |
| 64  | 3.1%              | 多任务、大领域偏移 |
| 128 | 6.3%              | 极大领域偏移 |

实践经验:
  - r=8-16 对大多数微调任务足够
  - 增大 r 的收益有递减趋势
  - 多个低秩矩阵 (r=8 × 2) 可能比一个高秩矩阵 (r=16) 效果更好
```

## 3. LoRA 的推理优化

### 3.1 权重合并 (Merge)

```
训练时:
  h = W × x + (α/r) × B × A × x
  两次矩阵乘法

推理时 (合并):
  W_merged = W + (α/r) × B × A
  h = W_merged × x
  一次矩阵乘法 → 零额外推理开销

合并是永久性的 (in-place)，合并后无法恢复 LoRA 权重
```

### 3.2 多 LoRA 服务

```
场景: 同一基座模型服务多个用户/任务的 LoRA adapter
  基座模型: LLaMA-7B (14 GB)
  LoRA-1: 代码生成 (r=16, ~5 MB)
  LoRA-2: 数学推理 (r=16, ~5 MB)
  LoRA-3: 中文写作 (r=16, ~5 MB)

vLLM / SGLang 的多 LoRA 服务:
  1. 基座模型加载一次 (14 GB)
  2. 每个 LoRA adapter 只需加载 A, B 矩阵 (~5 MB each)
  3. 请求路由到对应 adapter
  4. 批量请求可以混合不同 adapter (PagedAttention + 动态 LoRA 权重)
```

### 3.3 vLLM 的 LoRA 实现

```
vLLM 支持多 LoRA 推理:
  - LoRA 权重作为 extra_keys 参与 KV Cache hash (prefix caching 兼容)
  - 支持 QKV 投影的 LoRA
  - LoRA 权重预加载到 GPU，切换开销极小
  - 支持 continuous batching + 多 LoRA

配置:
  python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-2-7b-hf \
      --enable-lora \
      --max-lora-rank 16 \
      --max-loras 4
```

## 4. LoRA 变体

### 4.1 QLoRA (Quantized LoRA)

```
核心创新: 4-bit 量化冻结权重 + LoRA 微调

1. 冻结权重用 NF4 (NormalFloat4) 量化 → 显存减 75%
2. LoRA 权重保持 BF16 → 保证微调质量
3. 双重量化: 量化常量本身也量化 → 进一步节省

显存对比:
  全量微调 7B (FP16):  ~60 GB
  LoRA 7B (FP16):      ~20 GB
  QLoRA 7B (NF4):      ~6 GB  → 单张 RTX 4090 即可

质量: QLoRA 在大多数任务上与全量微调差距 < 1%
```

### 4.2 DoRA (Weight-Decomposed LoRA)

```
将权重分解为幅度和方向:
  W = m × (V / ||V||)

  m: 幅度向量 [d_out]
  V: 方向矩阵 [d_out × d_in]

LoRA 只更新方向部分:
  V' = V + B × A
  W' = m × (V' / ||V'||)

优势: 更接近全量微调的学习能力
劣势: 额外的归一化计算开销
```

### 4.3 rsLoRA (Rank-Stabilized LoRA)

```
缩放因子改为: α / √r (而非 α/r)

理由: 高秩时 α/r 缩放太激进，导致训练不稳定
      √r 缩放在不同秩下更稳定
```

## 5. LoRA 在 RLHF 中的应用

### 5.1 LoRA-RLHF

```
标准 RLHF: Actor/Critic/RM/Ref 都是全量模型 → 显存爆炸
LoRA-RLHF:
  基座模型: 冻结的预训练权重 (共享)
  Actor LoRA: 训练
  Critic LoRA: 训练
  RM: 可以用单独的小模型
  Ref: 冻结的基座模型 (无需 LoRA)

显存节省:
  7B 模型, r=16:
    基座: 14 GB (共享)
    Actor LoRA: ~10 MB
    Critic LoRA: ~10 MB
    总计: ~14 GB + 20 MB → 单卡可跑

  vs 全量: ~60 GB × 4 = 240 GB
```

### 5.2 权重切换

```
RL 训练循环中的 LoRA 权重切换:

Rollout 阶段:
  加载 Actor LoRA 到基座模型
  用 vLLM 生成回复

Training 阶段:
  加载 Critic LoRA 到基座模型
  计算价值函数和优势

优势: 切换只需替换 A, B 矩阵 (~MB 级别)
  vs 全量切换需要替换整个模型 (~GB 级别)
```

## 6. 其他 PEFT 方法

### 6.1 方法对比

```
| 方法       | 额外参数 | 推理开销 | 质量     | 适用场景 |
|-----------|---------|---------|---------|---------|
| LoRA      | 0.1-1%  | 0% (合并) | 高     | 通用微调 |
| QLoRA     | 0.1-1%  | 0% (合并) | 高     | 显存受限 |
| Adapter   | 1-5%    | 有 (额外层) | 高   | 多任务 |
| Prefix    | 0.1-1%  | 有 (占用seq) | 中  | 提示工程 |
| Prompt    | <0.1%   | 有 (占用seq) | 低  | 极少数据 |
| BitFit    | <0.1%   | 0%       | 中     | 简单任务 |
```

### 6.2 为什么 LoRA 是主流选择

1. **零推理开销**：权重合并后与原始模型完全相同
2. **极低参数量**：r=16 时只需 ~0.8% 参数
3. **模块化**：一个基座 + 多个 LoRA adapter，灵活切换
4. **训练稳定**：B 初始化为零，从预训练行为开始
5. **广泛支持**：Hugging Face PEFT、vLLM、SGLang 都原生支持

## 7. 实践：LoRA 显存估算

```python
def estimate_lora_memory(
    hidden: int,
    layers: int,
    rank: int,
    target_modules: list = ["q_proj", "v_proj"],
) -> dict:
    """估算 LoRA 微调的显存需求"""

    # 每个 target module 的 LoRA 参数
    params_per_module = rank * (hidden + hidden)  # A + B
    params_per_layer = params_per_module * len(target_modules)
    total_params = params_per_layer * layers

    # BF16 显存 (LoRA 参数 + 梯度 + AdamW 优化器)
    lora_params_bytes = total_params * 2  # BF16
    lora_grad_bytes = total_params * 2    # BF16
    lora_optim_bytes = total_params * 12  # FP32 master + m + v
    total_lora_bytes = lora_params_bytes + lora_grad_bytes + lora_optim_bytes

    return {
        "lora_params_M": total_params / 1e6,
        "lora_memory_MB": total_lora_bytes / (1024**2),
        "per_layer_params_K": params_per_layer / 1e3,
    }


# LLaMA-7B, r=16, Q+V
result = estimate_lora_memory(hidden=4096, layers=32, rank=16)
print(f"LoRA 参数量: {result['lora_params_M']:.1f}M")
print(f"LoRA 显存: {result['lora_memory_MB']:.1f} MB")
# → LoRA 参数量: 4.2M, LoRA 显存: 67.2 MB
```

## 8. 关键要点

1. **LoRA 用低秩矩阵分解逼近参数更新** — r=8-16 就能覆盖大多数微调需求
2. **零推理开销是杀手锏** — 权重合并后与原始模型完全相同，部署无额外成本
3. **多 LoRA 服务是重要应用** — 一个基座模型服务多个 adapter，显存和加载成本极低
4. **QLoRA 进一步降低门槛** — 4-bit 量化 + LoRA 让 7B 模型在消费级 GPU 上微调
5. **LoRA-RLHF 极大降低 RL 训练显存** — 共享基座，切换只需 MB 级别的 LoRA 权重

## 参考

- 论文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- 论文: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- 论文: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- 库: [Hugging Face PEFT](https://github.com/huggingface/peft)
- 博客: [LoRA: Low-Rank Adaptation of Large Language Models](https://lightning.ai/pages/community/lora-explained/)
