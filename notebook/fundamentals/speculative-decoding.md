# Speculative Decoding 推理加速深度解析

> 用小模型猜测、大模型验证 — 在不损失精度的前提下加速自回归生成

## 1. 核心问题：自回归生成的瓶颈

自回归 (AR) 生成一次只产出一个 token：

```
输入: "The capital of France is"
Step 1: "Paris"  ← 一次 forward pass 只得到 1 个 token
Step 2: " and"
Step 3: " it"
...
```

**瓶颈**：每个 token 需要一次完整的模型 forward pass。
- 7B 模型，batch=1：~10ms/token
- 70B 模型，batch=1：~50ms/token
- **推理延迟与生成长度线性增长**

关键观察：**推理时 GPU 利用率极低**（通常 < 10%），大量算力被浪费。

## 2. Speculative Decoding 基本原理

### 2.1 核心思想

```
┌──────────────────────────────────────────────────┐
│ Draft Model (小，快)                              │
│   快速生成 K 个候选 token: [t1, t2, ..., tK]    │
│                                                   │
│ Target Model (大，准)                             │
│   一次 forward pass 验证所有 K 个 token           │
│   接受正确的 prefix，拒绝错误的 token             │
└──────────────────────────────────────────────────┘
```

### 2.2 Draft-then-Verify 流程

```
Step 1: Draft
  小模型自回归生成 K 个 token:
  x1 = draft_model(input)      # 快速
  x2 = draft_model(input, x1)  # 快速
  ...
  xK = draft_model(input, x1..xK-1)

Step 2: Verify
  大模型一次 forward pass:
  P_target(xi | input, x1..xi-1)  for i = 1..K
  同时得到所有 K 个位置的 logits

Step 3: Accept/Reject
  对每个位置 i，比较:
  P_draft(xi) vs P_target(xi)

  如果接受: 继续检查下一个位置
  如果拒绝: 从 target 分布重新采样该位置，丢弃后续

  最坏情况: 拒绝第1个 → 只得到1个token (同标准AR)
  最好情况: 全部接受 → 得到 K+1 个token!
```

### 2.3 接受/拒绝采样标准

使用**拒绝采样**保证输出分布与 target model 完全一致：

```python
# 对每个候选 token x_i
def accept_reject(x_i, p_target, p_draft, epsilon=0):
    """保证输出分布 = target 分布"""
    r = random.uniform(0, 1)

    if r < min(1, p_target(x_i) / (p_draft(x_i) + epsilon)):
        # 接受：这个 token 保持不变
        return True, x_i
    else:
        # 拒绝：从修正分布中采样
        # 修正分布 = max(0, p_target(x) - p_draft(x)) 归一化
        corrected_dist = normalize(max(0, p_target - p_draft))
        new_token = sample(corrected_dist)
        return False, new_token
```

**关键性质**：接受/拒绝采样保证最终输出分布与直接使用 target model 完全一致 — **零精度损失**。

## 3. 性能分析

### 3.1 加速比

```
理论加速比 = E[接受的 token 数] / 标准 AR 的时间

假设:
  t_target = target model 单步时间
  t_draft  = draft model 单步时间
  α = 平均接受率 (acceptance rate)
  K = draft 长度

接受的 token 数期望:
  E[accepted] = (1 - α^(K+1)) / (1 - α)

加速比:
  Speedup = E[accepted] / (1 + K * t_draft/t_target)

  当 t_draft << t_target (小模型很快):
  Speedup ≈ E[accepted]
```

### 3.2 接受率的影响

```
α = 0.8, K = 5:
  E[accepted] = (1 - 0.8^6) / (1 - 0.8) = 0.738/0.2 = 3.69
  Speedup ≈ 3.69x

α = 0.9, K = 5:
  E[accepted] = (1 - 0.9^6) / (1 - 0.9) = 0.469/0.1 = 4.69
  Speedup ≈ 4.69x

α = 0.6, K = 5:
  E[accepted] = (1 - 0.6^6) / (1 - 0.6) = 0.954/0.4 = 2.39
  Speedup ≈ 2.39x
```

### 3.3 实际加速效果

| Target | Draft | K | 加速比 |
|--------|-------|---|--------|
| LLaMA-70B | LLaMA-7B | 5 | 2-3x |
| LLaMA-13B | LLaMA-1.4B | 5 | 2-2.5x |
| LLaMA-7B | LLaMA-350M | 5 | 1.5-2x |

**关键发现**：
- Draft model 越接近 target model → 接受率越高
- 但 draft model 太大 → draft 时间太长
- **最优选择**：draft model 大小约为 target 的 1/10

## 4. 变体与优化

### 4.1 Self-Speculative Decoding

```
不需要单独的 draft model！
用同一个模型的 earlier layers 作为 draft：
  - 浅层 (1-N/2) → 快速 draft
  - 完整模型 → 验证

优点: 不需要额外模型
缺点: draft 质量可能不如专用小模型
```

### 4.2 Medusa (多头预测)

```
给 LM 添加额外的预测头:
  - 主 head: 预测 next token
  - Head 1: 预测 next-next token
  - Head 2: 预测 next-next-next token
  ...

一次 forward pass → 多个位置的候选
用 tree-based verification 高效验证

优点: 不需要 draft model，单模型即可
实现: vLLM 支持 Medusa
```

### 4.3 Eagle

```
基于特征级的 speculative decoding:
  - 用 target model 的中间特征训练轻量 draft model
  - Draft 在特征空间而非 token 空间操作
  - 接受率更高
```

### 4.4 Staged Speculative Decoding

```
多级 speculative:
  Draft Level 1 (超小模型) → Draft Level 2 (小模型) → Target Model

  层层筛选，提高最终接受率
```

## 5. vLLM 中的 Speculative Decoding

### 5.1 配置

```python
from vllm import LLM, SamplingParams

# 使用 speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",  # target model
    speculative_model="meta-llama/Llama-2-7b-hf",  # draft model
    num_speculative_tokens=5,  # K
)

sampling_params = SamplingParams(
    temperature=0,  # greedy 时 speculative decoding 最有效
    max_tokens=256,
)

output = llm.generate(prompts, sampling_params)
```

### 5.2 vLLM 的实现细节

```
1. Draft 阶段:
   - Draft model 独立运行，生成 K 个 token
   - 利用 vLLM 的 PagedAttention 管理 draft KV cache

2. Verify 阶段:
   - Target model 一次 forward pass 验证 K 个 token
   - 利用 vLLM 的 chunked prefill 优化

3. 接受后:
   - 复用已验证 token 的 KV cache
   - 丢弃被拒绝 token 的 KV cache

4. GPU 利用率:
   - 标准 AR: GPU 利用率 ~5-10%
   - Speculative: GPU 利用率提升至 20-40%
```

### 5.3 支持的后端

| 方法 | 是否需要额外模型 | vLLM 支持 |
|------|----------------|----------|
| Speculative (draft model) | 是 | Yes |
| Medusa | 否 (多头) | Yes |
| Eagle | 否 (轻量 draft) | Yes |
| Self-speculative | 否 | 实验性 |

## 6. 适用场景分析

### 适合

```
✅ 低 batch size 推理 (GPU 利用率低时)
✅ 延迟敏感的应用 (聊天、交互式)
✅ Target model 远大于 draft model (t_draft << t_target)
✅ Greedy 或低 temperature 采样 (确定性高)
✅ 有现成的同架构小模型
```

### 不适合

```
❌ 高 batch size 推理 (GPU 已充分利用)
❌ 高 temperature 采样 (接受率低)
❌ Draft model 太大 (draft 时间抵消收益)
❌ Target 和 draft 架构差异大 (接受率低)
❌ 对吞吐量而非延迟敏感的场景
```

## 7. 学习要点

1. **核心洞察**：推理时 GPU 大量闲置，可以用算力换延迟
2. **拒绝采样**：保证输出分布与 target 完全一致 — 零精度损失
3. **加速比取决于接受率** — draft model 越接近 target，加速越高
4. **最优 draft 大小约为 target 的 1/10**
5. **Medusa/Eagle 等变体**不需要单独的 draft model
6. **vLLM 原生支持**多种 speculative decoding 方法

## 模拟验证

- `tools/speculative_decoding_sim.py` — Speculative Decoding 模拟器（5 个实验）
  - 实验 1: Standard AR vs Speculative 对比 (70B+7B, K=5, quality=0.8 → 1.48x 加速, target forward 节省 62%)
  - 实验 2: Draft 质量影响 (quality=0.5→0.83x减速, 0.8→1.40x, 0.99→2.14x; 分界线在 0.6-0.8)
  - 实验 3: 投机长度 K 最优化 (quality=0.7→最优K=2, quality=0.8→最优K=2, quality=0.9→最优K=3)
  - 实验 4: 不同模型规模 (7B无收益, 13B→1.60x, 70B→1.48x, 405B→1.88x)
  - 实验 5: Batch 扩展 (batch=1→1.54x, batch=32→1.46x; 大 batch 下收益递减)

关键发现:
- **最优 K 比预期小**: quality=0.8 时 K=2-3 最优，而非常见的 K=5。原因是位置衰减使远端 token 接受率低
- **Draft 质量分界线**: quality<0.6 时反而减速（draft 延迟 > 接受收益）
- **大模型收益最大**: 405B 模型加速 1.88x，因为 target 延迟远大于 draft 延迟
- **Batch 影响有限**: 1.42-1.54x，说明 speculative 主要适合在线推理而非批量场景

## 参考

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2302.01318) (Leviathan et al., 2023)
- [Speculative Decoding: Exploiting Speculative Execution for Faster AI](https://arxiv.org/abs/2305.09781) (Xia et al., 2023)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/models/speculative_decoding.html)
