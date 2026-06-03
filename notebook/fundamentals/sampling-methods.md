# LLM 采样方法

> 目标：理解 LLM 推理中从 logits 到 token 的采样过程，掌握各种采样策略的原理和适用场景

## 1. 采样管线概览

```
模型输出 logits (vocab_size,)
    ↓ Temperature 缩放
    ↓ 重复惩罚 (Repetition/Frequency/Presence Penalty)
    ↓ Softmax → 概率分布
    ↓ Top-K / Top-P / Min-P 过滤
    ↓ 采样 / 贪心 / Beam Search
    ↓ token_id
```

## 2. Temperature

Temperature 控制概率分布的"锐度"：

```python
probs = softmax(logits / temperature)
```

| Temperature | 效果 | Entropy | 适用场景 |
|------------|------|---------|---------|
| T→0 | 趋近 one-hot (贪心) | →0 | 代码生成、事实问答 |
| T=0.3 | 较集中 | 低 | 翻译、摘要 |
| T=0.7 | 适中 | 中 | 对话、写作 |
| T=1.0 | 原始分布 | 基准 | 默认 |
| T=1.5 | 较分散 | 高 | 创意写作 |
| T→∞ | 趋近均匀 | →log(V) | 随机实验 |

### 实测数据

```
T=0.1: Top-1 = 99.3%, Entropy = 0.058 (几乎确定性)
T=0.7: Top-1 = 54.6%, Entropy = 1.805 (适中)
T=1.0: Top-1 = 42.5%, Entropy = 2.305 (原始分布)
T=2.0: Top-1 = 25.4%, Entropy = 2.982 (很分散)
```

## 3. Top-K 采样

只保留概率最高的 K 个 token，其余设为 -inf 后重新归一化。

```python
top_k_indices = np.argsort(probs)[-K:]
filtered = np.where(np.isin(range(V), top_k_indices), probs, 0)
filtered /= filtered.sum()
```

**问题**：
- 分布集中时（Top-1 = 90%），K=50 浪费 49 个低概率位置
- 分布分散时（Top-1 = 10%），K=5 可能丢弃有用的 token

**常用值**: K=40-100（ChatGPT 用 K=0 即不限制，依赖 Top-P）

## 4. Top-P (Nucleus) 采样

按概率从大到小排序，累积概率达到 P 时截断。

```python
sorted_probs = sort(probs, descending=True)
cumsum = cumulative_sum(sorted_probs)
cutoff = searchsorted(cumsum, P)
# 保留 sorted_probs[:cutoff+1]，丢弃其余
```

**优势**：自适应——概率集中时自动少保留，分散时多保留。

| Top-P | 效果 |
|-------|------|
| P=0.5 | 只保留概率最大的 1-2 个 token |
| P=0.9 | 保留 90% 概率质量的 token |
| P=0.95 | 保留更多 token，更随机 |
| P=1.0 | 不做过滤 |

**推荐**: P=0.9-0.95

### Top-K vs Top-P 对比

| 特性 | Top-K | Top-P |
|------|-------|-------|
| 参数 | 固定数量 K | 累积概率 P |
| 自适应 | 否 | 是 |
| 集中分布 | 保留过多无用 token | 自动减少 |
| 分散分布 | 可能丢弃有用 token | 自动保留更多 |
| 推荐 | K=40-100 | P=0.9 |

## 5. Min-P 采样

动态阈值：只保留概率 ≥ max_prob × min_p 的 token。

```python
threshold = max(probs) * min_p
filtered = np.where(probs >= threshold, probs, 0)
filtered /= filtered.sum()
```

**Min-P 的自适应优势**：
- 集中分布 (Top-1=50%): Min-P=0.1 → 阈值=5%，保留 5 个 token
- 分散分布 (Top-1=15%): Min-P=0.1 → 阈值=1.5%，保留 10 个 token

**推荐**: min_p=0.05-0.1

## 6. 重复惩罚

### 6.1 Repetition Penalty

对已出现过的 token 缩放 logits：

```python
if logit > 0:
    logit /= penalty   # 缩小正 logit
else:
    logit *= penalty   # 放大负 logit
```

penalty > 1 惩罚，= 1 不惩罚。

### 6.2 Frequency Penalty

logit 减去 `frequency_weight × 出现次数`：

```python
logit -= frequency_penalty * count(token)
```

出现越多，惩罚越重。

### 6.3 Presence Penalty

logit 减去固定量（只要出现过）：

```python
logit -= presence_penalty * (count(token) > 0)
```

### 对比

| 方法 | 公式 | 特点 |
|------|------|------|
| Repetition | logit /= or *= penalty | 乘法，非线性 |
| Frequency | logit -= freq × count | 线性，与次数成正比 |
| Presence | logit -= pres × (count > 0) | 固定惩罚 |

**推荐**: RP=1.1-1.3 或 freq=0.2-0.6 + pres=0.2-0.6

## 7. Beam Search

### 原理

每步保留 B 条候选序列（beams），选择累积 log 概率最高的 B 条继续扩展。

```
Step 1: [Start] → B 个最佳 token
Step 2: 每条 beam 各扩展 V 个候选 → 从 B×V 中选 Top-B
...
最终: 选累积得分最高的序列
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 考虑全局最优 | 多样性不足 |
| 适合确定性任务 | 生成速度慢 B×V |
| 输出稳定 | 倾向"安全"但无趣的回答 |

### 适用场景

- **适合**: 翻译、摘要、代码补全、形式化任务
- **不适合**: 对话、创意写作、开放式生成

## 8. 常用组合策略

| 场景 | Temperature | Top-P | Min-P | 其他 |
|------|------------|-------|-------|------|
| Chat (通用) | 0.7 | 0.9 | - | RP=1.1 |
| Code | 0.2 | 0.95 | - | - |
| Creative | 1.0 | 0.9 | 0.05 | - |
| Fact Q&A | 0.1 | - | - | Greedy |
| Translation | - | - | - | Beam=4-5 |
|RL 训练 | 0.6-1.0 | - | - | GRPO/PPO |

## 9. vLLM 中的采样实现

vLLM 的采样流程在 `vllm/model_executor/layers/sampler.py`：

```python
class Sampler:
    def forward(self, logits, sampling_metadata):
        # 1. 应用 penalties (repetition, frequency, presence)
        logits = self._apply_penalties(logits, sampling_metadata)

        # 2. Temperature 缩放
        logits = logits / temperature

        # 3. Top-K + Top-P 过滤
        logits = self._apply_top_k_top_p(logits, top_k, top_p)

        # 4. Softmax → 概率
        probs = softmax(logits)

        # 5. 采样
        if greedy:
            token_ids = argmax(probs)
        else:
            token_ids = multinomial(probs)
```

### Triton Kernel 优化

vLLM 使用 Triton kernel 加速采样操作：
- `top_k_kernel`: GPU 并行的 Top-K 过滤
- `top_p_kernel`: GPU 并行的 Top-P 过滤
- `sampling_kernel`: 并行 multinomial 采样

## 10. RL 训练中的采样

在 GRPO/PPO 中，采样用于生成 rollout：

```python
# verl: 生成时使用 temperature sampling
with torch.no_grad():
    responses = model.generate(
        prompts,
        temperature=0.6-1.0,  # 较高温度增加探索
        top_p=0.9,
        do_sample=True,
    )
```

关键：RL 训练需要一定的随机性来探索不同的回答，温度通常比推理时更高。

## 实验验证

- `tools/sampling_simulator.py` — 采样方法模拟器（6 个实验）
  - 实验 1: Temperature 缩放效果 (entropy: 0.058 → 3.263)
  - 实验 2: Top-K vs Top-P 过滤对比
  - 实验 3: Min-P 动态阈值
  - 实验 4: 重复惩罚机制对比
  - 实验 5: Greedy vs Sampling vs Beam Search
  - 实验 6: 采样分布统计 (10000 次)

关键发现：
- T=0.7 时 Top-1 概率 54.6%，entropy 1.805 — 适合对话
- Top-P=0.9 通常比 Top-K=5 效果好（自适应阈值）
- Min-P 在分散分布时保留更多 token，集中时保留更少
- Beam Search 与 Greedy 输出相似，但理论上全局更优

## 参考资料

- [The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)](https://arxiv.org/abs/1904.09751) — Nucleus Sampling 论文
- [vLLM Sampler](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/sampler.py)
- [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Min-P Sampling](https://arxiv.org/abs/2407.01082)
