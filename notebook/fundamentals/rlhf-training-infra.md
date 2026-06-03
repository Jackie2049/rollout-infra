# RLHF/GRPO 训练基础设施

> 从 PPO 四模型到 GRPO 无 Critic，RL 训练如何重塑 AI Infra 系统

## 1. RLHF 训练管线总览

### 1.1 三阶段流程

```
Stage 1: SFT（监督微调）
  预训练基座 → 高质量指令数据微调 → 学会遵循指令

Stage 2: Reward Model 训练
  人类偏好数据（同一 prompt 多个回复排序）→ Bradley-Terry loss → RM

Stage 3: RL 优化（PPO/GRPO）
  SFT 模型作为初始策略 → RL 算法优化 → 输出更符合人类偏好
```

### 1.2 四模型架构

PPO 训练需要同时管理四个模型：

| 模型 | 参数量 | 作用 | 训练/推理 |
|------|--------|------|-----------|
| Actor (Policy) | 与 SFT 相同 | 生成回复，策略优化 | 训练 + 推理 |
| Critic (Value) | 与 Actor 相当 | 估计状态价值 V(s) | 训练 + 推理 |
| Reward Model | 通常较小 | 对完整回复打分 | 仅推理 |
| Reference Model | 与 Actor 相同 | 提供 KL 约束的参考分布 | 仅推理 |

**显存挑战**：7B 模型 FP16 下，四模型参数 = 56 GB，加上优化器状态/梯度/激活值轻松超过 200 GB。

### 1.3 RLHF 训练循环

```
┌──────────────────────────────────────────────────┐
│               RLHF Rollout Loop                   │
│                                                   │
│  1. Prompt Batch → Actor 生成 N 个回复 (Rollout)   │
│  2. Reference Model → 计算 log π_ref (KL 约束)    │
│  3. Reward Model → 对每个回复打分 r(x, y)         │
│  4. Critic → 估计 V(s_t) 每个位置的 value         │
│  5. 计算优势函数 A_t (GAE)                        │
│  6. 策略梯度更新 Actor + Value 更新 Critic         │
│  └────────── 回到步骤 1 ──────────┘               │
└──────────────────────────────────────────────────┘
```

**Rollout 瓶颈**：自回归生成占总训练时间 ~80%，因为 token-by-token 串行生成无法充分并行。这是 RLHF 基础设施优化的核心焦点。

## 2. PPO for LLM Training

### 2.1 Clipped Surrogate Objective

```
L^CLIP(θ) = E_t [ min(
    r_t(θ) * A_t,
    clip(r_t(θ), 1-ε, 1+ε) * A_t
)]

其中:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  — 新旧策略概率比
  A_t    — 优势函数估计
  ε      — 裁剪范围（通常 0.2）
```

裁剪限制策略更新幅度，避免单步更新过大导致训练不稳定。

### 2.2 KL 散度惩罚

```
R(x, y) = r(x, y) - β * KL[π_θ || π_ref]

其中:
  r(x, y) — Reward Model 给出的奖励
  β       — KL 惩罚系数（0.01-0.1）
  KL      — 按 token 级别计算: log(π_θ) - log(π_ref)
```

**为什么需要 KL 约束**：无约束时策略可能坍缩到高奖励但无意义的模式（reward hacking），或退化成极度简短的回复。

### 2.3 GAE (Generalized Advantage Estimation)

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

A^GAE_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

γ = 1.0 (LLM 场景通常不折扣)
λ = 0.95 (控制偏差-方差权衡)
  λ=0: 高偏差低方差（单步 TD）
  λ=1: 低偏差高方差（Monte Carlo）
```

### 2.4 关键超参数

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| ε (clip) | 0.2 | 策略裁剪范围 |
| β (KL) | 0.01-0.1 | KL 散度惩罚系数 |
| γ (discount) | 1.0 | 折扣因子 |
| λ (GAE) | 0.95 | GAE 参数 |
| learning rate | 1e-6 ~ 5e-7 | 比 SFT 小 10x |
| PPO epochs | 1-4 | 每批数据更新轮数 |
| response length | 512-2048 | 最大生成 token 数 |

## 3. GRPO (Group Relative Policy Optimization)

### 3.1 核心创新：无 Critic

GRPO 来自 DeepSeekMath (arxiv 2402.03300)，关键创新是**完全移除 Critic 模型**。

```
对每个 prompt x：
1. 从旧策略 π_θ_old 采样 N 个回复 {y_1, ..., y_N}
2. 用 RM 对每个回复打分：{r_1, ..., r_N}
3. 组内标准化优势：
   Â_i = (r_i - mean(r)) / std(r)
4. 优化 clipped objective（同 PPO 但用组内优势）
```

### 3.2 PPO vs GRPO 对比

| 方面 | PPO | GRPO |
|------|-----|------|
| Critic 模型 | 需要 | **不需要** |
| GPU 显存 | 4 个模型 | **3 个模型** |
| 优势估计 | GAE + Value Function | 组内相对排名 |
| 超参数敏感度 | 较高 | 相对较低 |
| 适合场景 | 通用 RLHF | 推理任务（有明确对错标准） |

### 3.3 GRPO 的局限

1. **组大小权衡**：N 越大估计越准，但计算成本线性增长
2. **有偏估计**：REINFORCE++ 论文指出组内标准化在理论上是有偏的
3. **不适合单样本**：N=1 时退化为无基线 REINFORCE，方差极高

### 3.4 DeepSeek-R1 的纯 RL 路线

DeepSeek-R1 展示了无需 SFT 推理数据即可激发推理能力：

```
传统: 预训练 → SFT(指令+推理数据) → RLHF
R1:   预训练 → SFT(仅指令) → GRPO(可验证奖励)
```

关键发现：模型自主发展出 self-reflection、verification、backtracking 等推理行为——"aha moment"。

## 4. 基础设施挑战

### 4.1 显存困境

```
7B 模型 RLHF (FP16):
  Actor 参数:     14 GB
  Critic 参数:    14 GB
  RM 参数:        14 GB
  Ref 参数:       14 GB
  ─────────────────────
  仅参数:         56 GB
  + 优化器+梯度:  ~200 GB

70B 模型: 仅参数就 560 GB → 必须多 GPU 分布式
```

### 4.2 共置 vs 分离部署

**共置模式 (Colocate)**：
```
同一组 GPU 时分复用:
  Rollout 阶段: 只加载 Actor
  训练阶段: 加载 Actor + Critic
  优点: GPU 利用率高，无网络开销
  缺点: 频繁权重加载/卸载，显存管理复杂
  适合: 模型可装入单机
```

**分离模式 (Disaggregated)**：
```
不同模型 → 不同 GPU 组:
  Actor GPU 组: 8 GPU (推理优化)
  Critic GPU 组: 4 GPU
  RM + Ref GPU 组: 4 GPU
  优点: 各模型独立管理
  缺点: 部分时间某些 GPU 空闲
  适合: 大规模分布式训练
```

### 4.3 vLLM 集成挑战

将推理引擎集成到 RL 训练循环的技术难点：

1. **权重同步**：Actor 训练后需同步到 vLLM 推理引擎
2. **模型重分片**：训练用 TP+PP，推理用不同并行策略 → 需动态重排权重
3. **显存协调**：vLLM 的 PagedAttention 与训练框架的显存管理策略不同
4. **批处理对齐**：训练 batch 与推理 continuous batching 需要协调

## 5. verl 架构 (ByteDance, EuroSys 2025)

### 5.1 核心设计

```
┌──────────────────────────────────────────────┐
│              verl 架构                        │
│                                              │
│  Hybrid Controller                           │
│  ├── Single Ctrl (调度编排)                   │
│  └── Multi Controller (分布式计算执行)         │
│                                              │
│  Actor Worker ─── 3D-HybridEngine            │
│  Critic Worker ── Megatron / FSDP            │
│  Reward Worker ── vLLM                       │
│  Ref Worker ───── vLLM                       │
└──────────────────────────────────────────────┘
```

### 5.2 3D-HybridEngine

verl 最关键的基础设施创新：

```
问题: 训练和推理对模型分布有不同最优配置
  训练: TP=2, PP=2, DP=4 (Megatron 3D 并行)
  推理: TP=8 (vLLM 张量并行)

解决: 零冗余权重重分片
  预计算训练→推理的分片映射
  直接在 GPU 间传输对应分片
  无需全局 all-gather
  零额外显存开销

开销: 权重重分片 < 5% 训练时间
```

### 5.3 三级前缀缓存

```
系统级: 跨 episode 的 prompt 缓存（长期共享）
进程级: 同一 worker 内的 KV Cache 复用
请求级: 同 batch 内相同 prompt 的分组复用

配合 vLLM APC 或 SGLang RadixAttention
```

### 5.4 性能数据

```
HybridFlow 论文:
  vs 基线: 1.53x ~ 20.57x 吞吐提升
  可扩展到 671B 模型
```

## 6. OpenRLHF 架构

### 6.1 设计

```
Ray Cluster:
  vLLM Engine (Actor Rollout) ── 推理优化的生成
  DeepSpeed Critic Worker ────── 训练
  DeepSpeed RM Worker ────────── 仅推理
  vLLM Ref Worker ────────────── 仅推理

PPO Trainer (协调器):
  调度 rollout → reward → ref → advantage → update
```

### 6.2 部署模式

- **Colocate-All**: 所有模型在同一组 GPU，DeepSpeed ZeRO-3 + 权重卸载，适合 7B-13B
- **Separate**: 模型分布在不同 GPU 组，Actor 用 vLLM，Critic+RM 用 DeepSpeed，适合 70B+

### 6.3 支持的算法

PPO, REINFORCE++, GRPO, RLOO, DAPO, DPO, SimPO, KTO, Online DPO 等。

## 7. Prefix Sharing in RL Training

### 7.1 冗余分析

```
Prompt 长度 L_p, Response 长度 L_r, 组大小 N

不共享: 总计算 = N × (L_p + L_r)
共享:   总计算 = L_p + N × L_r

节省比例 = (N-1) × L_p / (N × (L_p + L_r))

示例:
  L_p = L_r, N=8  → 节省 87.5%
  L_p = 2×L_r, N=4 → 节省 60%
```

### 7.2 实现方式

```
verl PrefixGrouper:
  1. 将相同 prompt 的 N 个生成请求归为一组
  2. 只计算一次 prompt 的 KV Cache
  3. 同组请求调度到同一 GPU，利用 vLLM APC

vLLM Automatic Prefix Caching:
  Block 级 hash 去重 → 新请求命中已缓存 prefix → 直接复用

SGLang RadixAttention:
  基数树管理 token 序列 → KV Cache 映射 → 前缀匹配自动复用
```

### 7.3 对训练吞吐的影响

```
Rollout 占训练 ~80% 时间
Prefix Sharing 减少 Rollout 计算 40-80%
→ 整体训练吞吐提升 2-5x
```

## 8. 最新趋势 (2025-2026)

### 8.1 简化 RL 算法

```
演进路径:
  PPO (4模型) → GRPO (3模型) → 可验证奖励 (2模型)
                                      ↑ 无需 RM

RLOO: REINFORCE + Leave-One-Out 基线，无需 Critic
REINFORCE++: 全局标准化（修正 GRPO 的有偏估计）
DAPO: 动态优势策略 + 可验证奖励
```

### 8.2 可验证奖励

```
数学: 答案精确匹配
代码: 测试用例验证
优势: 零 RM 开销，无 reward hacking，信号更准确
代表: DeepSeek-R1, DAPO
```

### 8.3 Multi-Turn RL

```
新方向: 训练多轮交互能力（工具使用、代码调试、对话）
挑战:
  - 变长序列管理
  - 环境交互延迟（代码沙箱、搜索 API）
  - 多轮奖励累积
  - 安全沙箱管理
```

### 8.4 趋势总结

| 趋势 | 方向 | Infra 影响 |
|------|------|-----------|
| Critic-free | 移除 Critic | 显存减 25%，管线简化 |
| 可验证奖励 | 移除 RM | 显存再减少，需环境 |
| 在线 DPO/IPO | 简化 RL 循环 | 无需 GAE/Critic |
| 混合引擎 | 训练推理统一 | 3D-HybridEngine |
| Prefix Sharing | 减少 rollout 计算 | KV Cache 管理 |
| Multi-turn RL | 更复杂交互 | 环境管理 |

## 关键要点

1. **Rollout 是瓶颈**：80% 时间在生成，优化推理引擎集成和 prefix sharing 是 ROI 最高的方向
2. **模型数在减少**：PPO 4 模型 → GRPO 3 模型 → 可验证奖励 2 模型，基础设施在简化
3. **混合引擎是核心**：verl 3D-HybridEngine 和 OpenRLHF 解决训练-推理权重同步的根本问题
4. **Prefix Sharing 是免费午餐**：RL 场景天然大量共享前缀，利用 prefix caching 可大幅减少 rollout 计算

## 参考

- 论文: [HybridFlow (verl)](https://arxiv.org/abs/2409.19256) (EuroSys 2025)
- 论文: [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- 论文: [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- 论文: [RLOO](https://arxiv.org/abs/2402.14740)
- 论文: [REINFORCE++](https://arxiv.org/abs/2501.03262)
- 项目: [verl](https://github.com/volcengine/verl)
- 项目: [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
