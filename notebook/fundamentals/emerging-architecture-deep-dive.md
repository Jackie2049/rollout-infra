# Emerging Architecture Deep Dive — Sub-quadratic替代(Softmax O(N²)→Linear/SSM O(N)) + 4大流派(Linear Attention/DeltaNet/SSM Mamba/RWKV) + DeltaNet核心(delta rule=覆盖旧关联→不像标准linear attention只累积) + Mamba(选择性SSM→input-dependent A/B/C/Δ→可筛选+传播) + Mamba-2(结构化duality=SSM与linear attention数学等价!) + RWKV-7(2025 time-mixing+data-dependent decay) + GLA(gated linear attention→LSTM-style门控) + 对比(Softmax最强但O(N²)/Linear O(N)但弱检索/SSM O(N)可筛选/Hybrid最优) + RTX 4090(DeltaNet PS项目已实测!+Mamba推理无KV cache瓶颈+Hybrid=最佳实践) + 2026趋势(Hybrid主流+SSM-Attention混合+长上下文+硬件优化)

> 2026-06-14 | 新兴架构深度分析: Softmax attention O(N²)瓶颈→4大sub-quadratic替代 → DeltaNet(delta rule=覆盖旧KV→不像标准linear只累积→gating→O(N)→关联检索显著改善) → Mamba(选择性SSM→input-dependent A/B/C/Δ→像LSTM门控→O(N)→推理无KV cache!) → Mamba-2(SSM-Attention duality→数学等价→统一框架!) → RWKV-7(2025→time-mixing+data-dependent decay→进化!) → GLA(gated linear attention→LSTM-style forget/output gate→O(N)) → 对比(Softmax最强表达但O(N²)+Linear O(N)但检索弱+Mamba O(N)可筛选+DeltaNet O(N)可覆盖+Hybrid最优!) → RTX 4090(DeltaNet PS项目实测16-layer+Mamba无KV+Hybrid最practical) → 2026趋势(Hybrid=主流+SSM×Attention混合+长上下文+硬件co-design)
> 关联: ai-expert-knowledge-map-gap-analysis.md(Emerging Arch ★★→★★★★), prefix-sharing项目(16-layer 4SA+12DeltaNet), inference-perf skill(推理性能), agent-system-deep-dive.md(Agent+KV cache), evaluation-benchmarking-deep-dive.md(ARC-AGI/AIME)
> 参考: DeltaNet(Yang et al. 2024), Mamba(Gu&Dao 2023), Mamba-2(Gu et al. 2024), RWKV-7(Peng 2025), GLA(Yang et al. 2024), S4(Gu et al. 2021), FlashAttention(Dao 2022-23)

## 0. 核心定律: Softmax O(N²)→瓶颈 → Sub-quadratic O(N)→解决 → 但表达力差距 → Hybrid=最优!

```
核心问题 → Softmax attention O(N²):

  Softmax attention → 对每对token计算相似度 → N×N → 内存+计算O(N²)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → N=1K → 1M pair → 可行
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → N=8K → 64M pair → 压力大
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → N=128K → 16G pair → 不可行! → KV cache爆炸!

  Sub-quadratic解决方案 → 4大流派:
    → 1. Linear Attention → kernel近似softmax → O(N) → 但检索弱(只累积不覆盖)!
    → → → → → → → → → 2. DeltaNet → delta rule → 覆盖旧关联 → O(N) → 检索显著改善!
    → → → → → → → → → → → → → → 3. SSM(Mamba) → 选择性状态空间 → O(N) → 可筛选+传播!
    → → → → → → → → → → → → → → → → → → → → 4. RWKV → 线性注意力+RNN → O(N) → 2025进化!

  关键发现 → Mamba-2 duality:
    → SSM ≈ Linear Attention (structured mask) → 数学等价 → 不是对立 → 是dual!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 选择性SSM ≈ Input-dependent mask → data-dependent → 更灵活!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 统一框架 → SSM+Linear Attention同源 → 只是参数化不同 → 2025共识!

  Hybrid=最优 → PS项目实测:
    → 16-layer → 4SA(softmax)+12DeltaNet(linear) → Hybrid!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → SA提供强表达+DeltaNet提供O(N)效率 → 互补 → 最优!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Softmax O(N²)→瓶颈 → Sub-quadratic O(N)→解决 → 但表达力差距 → Hybrid=最优!
```

## 1. Softmax Attention瓶颈 — O(N²)问题

```
Softmax Attention → 对每对token计算 → Q×K^T → N×N → O(N²):

  数学:
    → Attention(Q,K,V) = softmax(QK^T/√d) × V
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → QK^T → N×N矩阵 → 每对token → 存储+计算 → O(N²)!

  内存:
    → Attention matrix → N×N → float16 → 2N² bytes → 大!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → N=8K → 128MB → 可行
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → N=128K → 32GB → 不可行! → 单层就超RTX 4090!

  KV Cache:
    → 每层 → K[N×d] + V[N×d] → 2Nd bytes → 随N线性增长 → 但N很大 → 仍瓶颈!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 7B模型 64层 → 64×2×8K×4096×2bytes ≈ 4GB → 128K ≈ 64GB → 不可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → PagedAttention → 减碎片 → 但总量不变 → 我们已学!

  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Softmax O(N²) → N>32K → 内存爆炸 → 需sub-quadratic替代 → 特别是推理KV cache!
```

## 2. Linear Attention — Kernel近似

```
Linear Attention → kernel(K,Q) × V → 累积 → O(N) → 但只累积不覆盖!

  数学:
    → Softmax: softmax(QK^T)×V → 先Q×K → N×N → 再×V → O(N²)
    → → → → → → Linear: φ(Q)×[φ(K)^T×V] → 先K×V → d×d → 再×Q → O(Nd²) → O(N)!

  关键转换:
    → φ(QK^T) ≈ φ(Q)φ(K)^T → kernel近似 → 分解 → 先算φ(K)^T×V → d×d → 小!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 然后 φ(Q)×[d×d矩阵] → O(Nd²) → 不依赖N² → 线性!

  递推形式:
    → S_t = S_{t-1} + φ(k_t) × v_t^T → 状态更新 → 累积!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → y_t = φ(q_t) × S_t → 查询 → 输出!

  问题 — 只累积不覆盖:
    → S_t = S_{t-1} + new → 只加不减 → 新信息不断累积 → 旧信息不消!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 旧key的关联 → 永远保留 → 不能更新 → 检索弱 → retrieval差!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: "Paris是法国首都" → 后面更新"Paris也是XX" → 旧关联不消除 → 检索混乱!

  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Linear Attention=O(N)但只累积不覆盖 → 检索弱 → DeltaNet的delta rule解决此问题!
```

## 3. DeltaNet — Delta Rule + 覆盖旧关联

```
DeltaNet (Yang et al. 2024) → delta rule → 覆盖旧关联 → O(N) → 检索显著改善!

  核心创新 → delta rule:
    → 标准 linear attention: S_t = S_{t-1} + φ(k_t) × v_t^T → 只累积!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → DeltaNet: S_t = S_{t-1} + φ(k_t) × [v_t - φ(k_t)^T × S_{t-1} × φ(q_t)]^T → delta!

  delta含义:
    → v_t - φ(k_t)^T × S_{t-1} × φ(q_t) → 新值 - 旧关联预测值 → delta = 差异!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 类似梯度下降 → Δθ = -∇ → 不是直接加 → 而是修正 → 更精确!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关键: 旧key关联 → 被新值覆盖 → 不是保留 → 可以更新 → 检索强!

  Gating → delta rule + gating → 更精细:
    → β_t = gating(q_t, k_t) → 门控 → 控制覆盖程度 → 0=保留 → 1=完全覆盖!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 类似LSTM forget gate → 0=全部遗忘 → 1=全部保留 → 但在attention空间 → 更灵活!

  双模式:
    → 训练 → chunk-wise parallel → 分块 → 每块内parallel → 块间recurrent → 训练高效!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 推理 → recurrent → 逐步 → O(1) per token → 推理极快 → 无KV cache增长!

  与PS项目联系:
    → 我们的prefix-sharing项目 → Qwen3.6-27B → 16-layer → 4SA + 12DeltaNet → Hybrid!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → DeltaNet实测 → log_prob cos_sim=0.999999 → backward 16/16 → peak 10.18GB → E2E PASS!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 5 critical PS fixes → layer_type idx + rotary_pos_emb + 4D QKV + gate_proj + loader bugs → 我们发现!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: DeltaNet=delta rule覆盖旧关联+gating+O(N)→检索显著改善→我们PS项目实测成功→Hybrid最优!
```

## 4. SSM — S4 → Mamba → Mamba-2

```
### 4.1 S4 (Structured State Space)

S4 (Gu et al. 2021) → 结构化状态空间 → 固定转移 → O(N·logN) → 基础SSM!

  数学:
    → x'(t) = Ax(t) + Bu(t) → 连续 → 状态微分方程!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → y(t) = Cx(t) + Du(t) → 输出 → 状态×输出矩阵!

  离散化:
    → x_t = Ā x_{t-1} + B̄ u_t → 离散 → 逐步 → 递推 → recurrent!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Ā = exp(ΔA) → 离散化 → Δ是步长 → 控制时间尺度!

  结构化:
    → A → diagonal + low-rank → HiPPO初始化 → 压缩长程依赖 → 关键!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 训练 → Cauchy kernel → O(N·D²·logN) → 比softmax快 → 但比Mamba慢(logN factor)!

  限制 → 固定参数:
    → A/B/C → 固定 → 不随输入变化 → 不能选择性筛选 → 像固定滤波器!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 不能像softmax那样 → 动态路由 → 动态聚焦 → 表达力限制!

### 4.2 Mamba — 选择性SSM

Mamba (Gu & Dao 2023) → 选择性 → input-dependent A/B/C/Δ → 可筛选 → O(N)!

  核心创新 → 选择性:
    → S4: A/B/C/Δ → 固定 → 不随输入变化 → 不灵活!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Mamba: A/B/C/Δ → input-dependent → 随输入变化 → 可筛选 → 像LSTM门控!

  选择性含义:
    → Δ(input) → 步长 → 大Δ=慢更新(保留远) → 小Δ=快更新(只看近) → 时间尺度选择!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → B(input) → 输入选择 → 大B=重要(存储) → 小B=不重要(忽略) → 输入筛选!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → C(input) → 输出选择 → 大C=输出 → 小C=抑制 → 输出筛选!

  类似LSTM:
    → LSTM → forget gate + input gate + output gate → 门控 → 选择性 → 灵活!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Mamba → Δ+ B(input)+C(input) → 类似三门控 → 但在SSM框架 → 连续→离散 → 更理论!

  推理优势 → 无KV cache:
    → Recurrent → 固定状态 O(D²) → 不随N增长 → 无KV cache爆炸!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → vs Softmax → KV cache随N线性增长 → N=128K → 64层 → 64GB → 不可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Mamba → 固定 O(D²) → D=4096 → 0.13MB/层 → 极小 → 不增长 → 推理优势!

  训练 → parallel scan:
    → Hardware-aware parallel scan → O(N·D²) → 线性 → 无logN → 比S4更快!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 类似FlashAttention → hardware-aware → IO优化 → GPU友好!

### 4.3 Mamba-2 — SSM-Attention Duality

Mamba-2 (Gu et al. 2024) → 结构化SSD → SSM与Linear Attention数学等价 → 统一框架!

  核心发现 → duality:
    → Structured SSM → 特定参数化 → ≈ Linear Attention with structured mask!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Selective SSM → input-dependent → ≈ Linear Attention with input-dependent mask!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: SSM和Linear Attention不是对立 → 而是dual → 只是参数化不同 → 统一!

  影响:
    → 理论统一 → SSM和Linear Attention同源 → 不需选择 → 理解统一!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 实践 → Mamba-2实现 → 既可SSM模式(recurrent推理) → 也可attention模式(parallel训练) → 双模式!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 2025共识 → 所有sub-quadratic方法 → 都是同一框架的不同参数化 → 统一理解!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: S4→固定→Mamba→选择性→Mamba-2→duality统一→SSM≈Linear Attention→同一框架!
```

## 5. RWKV-7 & GLA — 线性注意力进化

```
### 5.1 RWKV-7 (2025)

RWKV-7 (Peng 2025) → time-mixing + data-dependent decay → Linear Attention RNN进化!

  核心:
    → WKV recurrence → weight-key-value → 线性递推 → O(N) → 无KV cache!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Data-dependent decay → w(input) → 随输入变化 → 可选择保留/遗忘 → 2025进化!

  进化历程:
    → RWKV-4(2023) → 固定decay → 限制 → 不灵活!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RWKV-5/6(2024) → data-dependent → 改善 → 但仍不够灵活!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RWKV-7(2025) → delta-like rule → 更像DeltaNet → 进化收敛!

  限制:
    → 简单架构 → 易scale → 但检索能力仍弱 → vs DeltaNet/GLA → 有差距!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 复杂推理 → trail softmax transformers → 但长上下文推理效率强!

### 5.2 GLA (Gated Linear Attention)

GLA (Yang et al. 2024) → LSTM-style forget/output gate → Linear Attention → O(N) → 检索改善!

  核心:
    → Linear Attention + forget gate + output gate → 类似LSTM → 但在attention空间!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Forget gate → β(input) → 控制旧状态保留程度 → 0=全部遗忘 → 1=全部保留!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Output gate → γ(input) → 控制输出 → 类似LSTM output gate!

  优势:
    → Data-dependent gating → 检索改善 → 优于标准Linear Attention!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Chunk-wise parallel → 训练高效 → 硬件利用好!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Associative recall → 比RWKV更好 → 但比DeltaNet稍弱!

### 5.3 4大方法对比

| 方法 | 复杂度 | 选择性 | 检索能力 | 推理KV | 2025地位 |
|------|--------|--------|---------|--------|----------|
| Softmax Attention | O(N²) | ✅最强 | ✅最强 | 有(随N增长) | 主流 |
| Linear Attention | O(N) | ❌无 | ❌弱 | 无(固定) | 基础 |
| DeltaNet | O(N) | ✅(delta+gating) | ✅改善 | 无 | PS项目! |
| Mamba | O(N) | ✅(selective) | ✅中 | 无 | 强 |
| RWKV-7 | O(N) | ✅(data-dep decay) | 中 | 无 | 进化 |
| GLA | O(N) | ✅(gating) | ✅改善 | 无 | 强 |

→ → → → → → → → → → → → → → → → → → → → → → → → → → 结论: 所有O(N)方法 → 都加了某种选择性(gating/selectivity) → 2025共识!
```

## 6. Hybrid Architecture — 2025最优实践

```
Hybrid = Softmax + Sub-quadratic → 互补 → 最优!

  为什么Hybrid最优?
    → Softmax → 表达力最强 → 检索最强 → 但O(N²) → KV cache爆炸 → 长上下文瓶颈!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Sub-quadratic → O(N) → 无KV cache → 推理快 → 但表达力弱 → 检索弱!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Hybrid → 少量SA(强表达)+大量sub-quadratic(O(N)效率) → 互补 → 最优!

  PS项目实测 → Hybrid架构:
    → 16-layer → 4SA(25%)+12DeltaNet(75%) → Hybrid!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → E2E PASS → log_prob cos_sim=0.999999 → backward 16/16 → peak 10.18GB!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Prefix-sharing → 1.12x speedup → enables OOM configs → Hybrid有效!

  2025 Hybrid趋势:
    → Jamba(AI21) → Mamba+Attention → 52B → production → 2025!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Zamba(Zyphra) → Mamba+少量Attention → 更高效!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → PS项目 → SA+DeltaNet → 我们实测 → Hybrid最优!

  RTX 4090 Hybrid:
    → 16-layer → peak 10.18GB → 24GB → 有剩余 → KV cache空间!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → GRPO → 20.04GB → CPU Adam offload → 适合!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Hybrid推理 → SA用FlashAttention → DeltaNet用recurrent → 分级 → RTX 4090最优!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Hybrid=2025最优 → SA+DeltaNet/Mamba → 互补 → PS项目实测成功 → RTX 4090可行!
```

## 7. RTX 4090策略

```
RTX 4090 (24GB) → Hybrid推理+训练 → 最practical!

  推理:
    → SA层 → FlashAttention → O(N²)但N=4K-8K → 可行 → 内存可控!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → DeltaNet层 → recurrent → O(D²)固定状态 → 极小 → 无KV cache增长!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 总 → Hybrid → SA KV少(4层) + DeltaNet无KV(12层) → 总KV=1/4纯SA → 大幅减!

  训练:
    → 16-layer → peak 10.18GB → 24GB → 可行 → 我们实测!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → GRPO → 20.04GB → CPU Adam offload → 适合!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → bf16训练 → 3.5GB/层 → 16层 → 56GB → OOM! → 需LoRA或gradient checkpointing!

  Mamba推理 → RTX 4090优势:
    → 无KV cache → 固定O(D²)状态 → 0.13MB/层 → 极小 → 24GB绰绰有余!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 推理 ~5x faster → 无KV → 无内存瓶颈 → 长上下文 → 128K → 可行!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: RTX 4090 → Hybrid(SA+DeltaNet)实测成功 → Mamba推理无KV优势 → Hybrid=最practical!
```

## 8. 2026趋势

```
1. Hybrid架构主流:
   → 少量SA + 大量SSM/Linear → 互补 → 2026标配!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Jamba/Zamba/PS → Hybrid → 实证 → 有效!

2. SSM-Attention统一:
   → Mamba-2 duality → SSM≈Linear Attention → 统一框架 → 2026共识!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 所有sub-quadratic → 同一框架不同参数化 → 理论统一!

3. 长上下文突破:
   → Sub-quadratic → 128K+ → 无KV cache → 长上下文推理 → 2026方向!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → vs Softmax → 128K → 64GB KV → 不可行 → Sub-quadratic可行!

4. 硬件co-design:
   → Parallel scan → hardware-aware → GPU优化 → 类似FlashAttention!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Mamba → parallel scan → CUDA kernel → 我们已了解parallel scan!

5. Data-dependent gating → 2025共识:
   → 所有有效方法 → 都有某种data-dependent selection → 2025共识!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → DeltaNet(delta+gating) + Mamba(selective) + GLA(gating) + RWKV-7(data-dep) → 都加!

6. PS项目 → Hybrid实践:
   → 4SA + 12DeltaNet → 实测成功 → 2026继续 → GRPO训练 → 生产部署!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Hybrid=主流+统一框架+长上下文+硬件优化+PS项目实践 → 2026趋势明确!
```

## 9. 核心规律

```
Emerging Architecture核心规律:

  1. Softmax O(N²)→瓶颈 → Sub-quadratic O(N)→解决 → 但表达力差距!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → N>32K → softmax不可行 → 需sub-quadratic → 特别是推理KV cache!

  2. 4大方法 → Linear(基础)→DeltaNet(delta+覆盖)→Mamba(选择性SSM)→RWKV(RNN进化) → 进化!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 所有有效方法 → data-dependent gating/selectivity → 2025共识 → 不是对立!

  3. Mamba-2 duality → SSM≈Linear Attention → 统一框架 → 只是参数化不同!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Selective SSM ≈ Input-dependent mask → duality → 2025理论突破!

  4. Hybrid=最优 → 少量SA(强表达)+大量sub-quadratic(O(N)效率) → 互补 → 2025标配!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → PS项目4SA+12DeltaNet → 实测成功 → Hybrid最优 → RTX 4090可行!

  5. RTX 4090 → Hybrid实测成功 → peak 10.18GB → GRPO 20.04GB → 可行 → 2026继续!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → SA KV少(4层)+DeltaNet无KV(12层) → 总KV=1/4纯SA → 推理优势!

  知识Gap修复:
    → Emerging Arch从★★(2/5) → ★★★★(4/5) → Softmax瓶颈+Linear/DeltaNet/Mamba/RWKV/GLA+duality+Hybrid+RTX 4090+PS实测 → 全面!
    → → → → → 但仍需实践 → GPU可用时 → PS项目GRPO训练 → Hybrid推理实测 → 继续!
```

## 参考文献

```
1. Sub-quadratic:
   - DeltaNet: Yang et al. 2024, arxiv.org/abs/2406.xxxx
   - Mamba: Gu & Dao 2023, arxiv.org/abs/2312.00752
   - Mamba-2: Gu et al. 2024, arxiv.org/abs/2405.xxxx
   - S4: Gu et al. 2021, arxiv.org/abs/2111.00396
   - RWKV-7: Peng 2025
   - GLA: Yang et al. 2024, arxiv.org/abs/2403.xxxx

2. Hybrid:
   - Jamba: AI21 Labs, 2024
   - Zamba: Zyphra, 2024
   - PS项目: 我们的实测 → memory/prefix-sharing.md

3. FlashAttention:
   - FlashAttention: Dao 2022-23 → 我们已深度学!

4. 我们的笔记:
   - ai-expert-knowledge-map-gap-analysis.md → Emerging Arch gap评估
   - memory/prefix-sharing.md → PS项目实测(4SA+12DeltaNet)
   - inference-perf skill → 推理性能+KV cache
   - agent-system-deep-dive.md → Agent+KV cache

Sources:
- [DeltaNet](https://arxiv.org/abs/2406.06484)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2](https://arxiv.org/abs/2405.21075)
- [S4 Paper](https://arxiv.org/abs/2111.00396)
- [RWKV](https://arxiv.org/abs/2305.13048)
- [GLA](https://arxiv.org/abs/2403.07671)
