# LoRA/PEFT Deep Dive: 从低秩分解到Multi-LoRA Serving

> 2026-06-08 | PEFT是训练+serving双优化, DoRA=方向+幅度分解, PiSSA=SVD初始化, Multi-LoRA=SegMM+Paging
> 基于: LoRA(Hu 2021), DoRA(NVIDIA 2024), PiSSA(2024), rsLoRA(2024), QoRA(2024), S-LoRA/Punica(2024), vLLM V1(2025)
> 关联: kv-cache-management-deep-dive.md, inference-cost-analysis.md, rl-alignment-unified-comparison.md

## 0. 核心定律: LoRA = 低秩更新 ≈ 主成分微调 ≈ 参数省99%但效果接近全量微调

```
LoRA核心思想:
  → 全量微调: W_new = W_pretrained + ΔW (ΔW是d×d, 参数量=d²)
  → LoRA: ΔW = A × B (A是d×r, B是r×d, 参数量=2dr)
  → → 参数省: d² → 2dr → 省d/(2r)倍! (d=4096, r=8 → 省256倍=99.6%!)

为什么低秩足够?
  → 预训练权重W已经包含大部分"知识"
  → 微调只需要"调整方向" → 调整是低秩的! (Aghajanyan 2020: ΔW intrinsic rank << d)
  → → 任务的适应信息集中在少数方向 → 低秩矩阵足以表达!

直觉:
  → W_pretrained = "通用知识" (高秩, 信息丰富)
  → ΔW = "任务特化" (低秩, 少数方向调整)
  → → 微调=在大知识矩阵上做小调整 → 低秩自然合适!

RTX 4090实际影响:
  → 7B模型: 全量微调28GB参数 → LoRA r=8仅0.5MB → 单卡可训练!
  → 70B模型: 全量微调280GB → LoRA r=8仅5MB → 需TP但不需ZeRO!
  → → LoRA让RTX 4090可以微调7B模型 → 之前不可能!
```

## 1. LoRA数学推导

```
标准LoRA (Hu et al. 2021):

原始权重: W ∈ ℝ^(d_out × d_in)
LoRA更新: h = Wx + BAx → 其中:
  → B ∈ ℝ^(d_out × r): "up-projection" (r → d_out)
  → A ∈ ℝ^(r × d_in): "down-projection" (d_in → r)
  → → ΔW = BA ∈ ℝ^(d_out × d_in) → 低秩(r << d_in, d_out)

缩放:
  → h = Wx + (α/r) × BAx → α/r是缩放因子
  → → α控制更新强度, r控制秩
  → → 当α=r → 缩放=1 → ΔW直接贡献
  → → 当α>r → 放大更新 → 更强适应但可能过拟合

初始化:
  → A: 随机Gaussian初始化 → ΔW初始≈0 → 开始时LoRA贡献≈0
  → B: 零初始化 → ΔW初始=0 → 保护预训练权重!
  → → 这是关键设计: 初始ΔW=0 → 模型开始=预训练模型 → 逐步适应

合并 (推理时):
  → W_merged = W + (α/r) × BA → 推理无额外开销!
  → → 合并后权重=标准权重 → 推理速度不变
  → → 这是LoRA vs其他PEFT的关键优势: 推理零开销!

取消 (回到原始模型):
  → W_original = W_merged - (α/r) × BA → 简单减法!
  → → Multi-LoRA serving: 可以动态切换adapter
  → → 合并/取消是O(dr) → 很快!

量化LoRA:
  → W可以是INT4量化 → BA是BF16 → 合并时需要dequant+add
  → → AWQ/LoRA: W用AWQ量化, BA用BF16 → 合并需要fused kernel
  → → QoRA: 量化感知LoRA → BA也是低精度 → 合并更友好
```

## 2. LoRA改进方法对比

```
2024-2025 PEFT方法全景:

| 方法 | 核心创新 | vs LoRA | 适用场景 |
|------|---------|---------|---------|
| **LoRA** | 低秩分解BA | baseline | 通用微调 |
| **DoRA** | 幅度+方向分解 | 更接近全量微调 | 高精度微调 |
| **PiSSA** | SVD初始化(主成分) | 收敛更快 | 快速微调 |
| **rsLoRA** | α/√r缩放 | 高秩更稳定 | 大秩微调 |
| **LoRA+** | A/B不同lr | 收敛更快 | 通用微调 |
| **QoRA** | 量化+LoRA | 推理更省 | 量化部署 |
| **AdaLoRA** | 动态rank分配 | 重要层更多rank | 自动调优 |
| **OLoRA** | 正交初始化 | 更好初始 | 初始化敏感任务 |

### DoRA: 幅度+方向分解 (NVIDIA 2024)

核心数学:
  → 权重分解: W = m × (V / ||V||) → m是幅度向量, V/||V||是方向矩阵
  → → m ∈ ℝ^(d_out) → 每行一个幅度值
  → → V ∈ ℝ^(d_out × d_in) → 方向矩阵
  → LoRA只更新方向: V_new = V + BA → 方向调整
  → 幅度单独更新: m_new = m + Δm → 幅度调整(1D向量, 极少参数)

为什么DoRA更好:
  → 全量微调 = 同时调整幅度和方向 → DoRA也同时调整两者!
  → LoRA只更新方向 → 丢失幅度调整 → DoRA补回幅度 → 更接近全量微调!
  → → 实测: DoRA在相同参数量下优于LoRA 1-3%

### PiSSA: 主成分初始化 (2024)

核心思想:
  → 不是随机初始化A/B → 用W的主成分(SVD)初始化!
  → W = UΣV^T → 取前r个奇异值:
    → A_init = Σ_r V_r^T → 包含W的主成分
    → B_init = U_r → 包含W的主要方向
  → → ΔW_init = B_init × A_init = U_r Σ_r V_r^T → 包含W最重要的r个成分!
  → → 残差W_residual = W - ΔW_init → 包含次要成分 → 训练时不动!

为什么PiSSA更好:
  → LoRA: ΔW初始=0 → 训练从头学习 → 需要更多步
  → PiSSA: ΔW初始=W的主成分 → 训练只需微调主成分 → 更快收敛!
  → → 实测: PiSSA在相同步数下优于LoRA → 尤其小rank时差距大!

### rsLoRA: 修正缩放 (2024)

核心修正:
  → LoRA缩放: ΔW_scaled = (α/r) × BA → α/r随r增大→缩小
  → → 问题: r增大 → 更大矩阵 → 但缩放减小 → 效果抵消 → 大rank反而不稳定!
  → rsLoRA: ΔW_scaled = (α/√r) × BA → α/√r → 缩放减小的速度更慢
  → → 好处: 大rank时更新不会太小 → 更稳定 → 高秩微调更有效

### LoRA+: 不对称学习率 (2024)

核心发现:
  → LoRA的A和B用相同lr → 但A和B的角色不同:
    → A: down-projection → 压缩 → 需要更精确定方向 → 需要更高lr!
    → B: up-projection → 恢复 → 需要更精确恢复 → 也需要调整
  → → 理论证明: A的lr应该比B高约16倍! (lr_A / lr_B ≈ 16)
  → → 实测: 不对称lr → 收敛更快 + 最终性能更好
```

## 3. 量化与LoRA交互

```
核心问题: 量化权重 + LoRA → 合并时怎么办?

量化LoRA推理路径:
  → 方案1: 合并后量化 → W_merged = (W_int4 + dequant(BA)) → 再量化 → 精度损失!
  → 方案2: 不合并 → W_int4 + BA分开推理 → BA每步计算 → 推理额外开销!
  → 方案3: QoRA → 量化感知训练 → 合并精度更好 → 但训练复杂!

QoRA/QA-LoRA关键创新:
  → QA-LoRA: 按列分组量化scale → 确保量化一致性
  → → W_quantized = W / scale_per_column × round → per-column scale
  → → 合并: (W + BA) / scale × round → 同一scale → 精度保持!
  → → vs普通量化: per-tensor scale → 合合后重新量化 → scale变了 → 精度差!

RTX 4090部署路径:
  → 推理最优: AWQ量化(base) + LoRA BF16 → 不合并 → vLLM支持!
  → → vLLM: base_model量化(INT4/AWQ) + LoRA adapter(BF16) → runtime应用
  → → 合并路径: AWQ → dequant → add LoRA → requant → 精度损失 → 不推荐!
  → → 不合并路径: base INT4推理 + LoRA额外计算 → 小overhead → 推荐!
  → → Multi-LoRA: 多个adapter → SegMM → batch不同adapter → 高效!
```

## 4. Multi-LoRA Serving架构

```
问题: 一个base模型 + 多个LoRA adapter → 如何高效serving?

传统方案 (每个adapter单独推理):
  → 请求1(adapter_A): base + LoRA_A → forward → 0.3ms
  → 请求2(adapter_B): base + LoRA_B → forward → 0.3ms
  → → 2次forward → 总时间0.6ms → 不能batch!

Multi-LoRA方案 (Punica/S-LoRA/vLLM V1):
  → 所有请求共享base → LoRA部分用Segmented MatMul(SegMM)
  → → base部分: 一次forward(B=2) → 共享计算!
  → → LoRA部分: SegMM → 不同adapter不同BA → 但合并为一个kernel!

### Punica: Segmented Matrix Multiplication

SegMM核心思想:
  → 标准GEMM: Y = XW → 所有token用同一个W
  → SegMM: Y[i] = X[i] × W_adapter[i] → 每组token用不同adapter!
  → → CUDA kernel实现: 一个kernel内 → 读取不同adapter的BA → 分段计算
  → → → 类似Grouped GEMM → 但按adapter分组而不是按expert分组!

SegMM实现:
  → 输入: X ∈ ℝ^(B × d_in), 多个adapter的BA ∈ ℝ^(r × d_in) 和 ℝ^(d_out × r)
  → → kernel: 对每个segment(token组) → load对应BA → compute BAx → add到base output
  → → → 一个kernel → 无需per-adapter forward → 效率极高!

### S-LoRA: Unified Paging

S-LoRA扩展Punica → 加入Paging机制:
  → LoRA adapter weights用Paged管理 → 类似KV cache的PagedAttention!
  → → adapter_weights按block分配 → 按需加载 → 不用的adapter可以swap到CPU
  → → → 1000个adapter → 24GB只装常用的 → swap不常用的 → 高效!
  → → TP策略: LoRA weights也沿TP切分 → 但需要AllReduce → 与base模型不同!

### vLLM V1: Integrated Multi-LoRA

vLLM V1整合了Punica/S-LoRA → 核心改进:
  → V0: LoRA是中间件 → scheduler不知道adapter → 低效batching
  → V1: LoRA集成到scheduler → scheduler按adapter_id分组请求 → 高效SegMM
  → → ModelExecutor: forward时 → base部分标准forward → LoRA部分SegMM
  → → → 一个step内 → 不同adapter的请求共享base → LoRA部分SegMM → 高效!

RTX 4090 Multi-LoRA:
  → 7B INT4 + 10个LoRA adapter(BF16 r=8) → 每个adapter0.5MB
  → → base: 1.75GB(INT4) → 10个adapter: 5MB → 总内存<2GB → 单卡可行!
  → → SegMM: 不同adapter共享base → B=32 → SegMM额外<0.1ms → 几乎免费!
  → → → Multi-LoRA是RTX 4090推理的重要方向 → 低成本多租户serving!
```

## 5. LoRA in RL Training (verl GRPO)

```
verl GRPO + LoRA架构:

  Actor(base_model + LoRA_actor): 生成rollout → 用LoRA微调策略
  Critic(base_model + LoRA_critic): 估计value → 用LoRA微调价值
  Reference(base_model): 计算KL → 不微调 → 固定!

  关键设计:
    → Actor和Critic可以共享base → 只需2个LoRA adapter → 内存省50%!
    → → 7B base + 2×LoRA(r=8) = 28GB + 1MB → vs 全量2×7B = 56GB → 省一半!
    → → RTX 4090: 7B BF16需要28GB → 不够! → INT4 base + LoRA → 可行!

  训练流程:
    → 1. Rollout: actor生成G个response → LoRA merged → 推理
    → 2. Reward: 计算reward → 无模型
    → 3. Advantage: GRPO计算advantage → reference计算KL → LoRA unmerged
    → 4. Update: actor LoRA更新 → critic LoRA更新 → FSDP

  Merge/Demerge关键:
    → 推理时: merge LoRA → W_merged = W + BA → 推理快
    → 训练时: demerge LoRA → W_original → 然后梯度只更新A和B → 内存省
    → → merge/demerge是O(dr) → 很快 → 但需要fused kernel避免Python overhead!

  verl代码结构:
    → LoRAConfig: rank, alpha, target_modules(qkv/out/mlp)
    → LoRAModel: wrap base_model + lora_A + lora_B → enable/disable
    → Merge前: model.forward(x) → Wx + BAx → 两步计算
    → Merge后: model.forward(x) → (W+BA)x → 一步计算 → 快!
```

## 6. LoRA vs 全量微调: RTX 4090成本对比

```
7B模型 RTX 4090 (24GB):

  全量微调:
    → 参数: 7B × 4bytes(BF16) = 28GB → 不够! 需要ZeRO-3或多卡
    → RTX 4090: 需要FSDP2 2GPU → 通信12GB/s → 1.12x(25M实测) → 勉强
    → 成本: 2×$0.35/hr = $0.70/hr → 2GPU

  LoRA微调 (r=8):
    → 参数: 7B × 2bytes(base frozen) + 2×7B×8/4096×4bytes ≈ 0.5MB
    → → 内存: 14GB(base) + 0.5MB(LoRA) + 4GB(optimizer+grad) = ~18GB → 单卡可行!
    → → RTX 4090: 单卡 → 无通信 → 全速!
    → → 成本: 1×$0.35/hr = $0.35/hr → 1GPU → 省50%!

  LoRA + INT4 base:
    → 参数: 7B × 0.5bytes(INT4) + 0.5MB(LoRA) = 3.5GB + 0.5MB
    → → 内存: 3.5GB(base) + 0.5MB(LoRA) + 4GB = ~8GB → 单卡轻松!
    → → 成本: 1×$0.35/hr → 但INT4训练需要fused kernel(AWQ/Marlin)

  全量微调 vs LoRA性能:
    → LoRA r=8: 95-98%全量微调性能(大多数任务)
    → LoRA r=16: 98-100%全量微调性能
    → DoRA r=8: 97-100% → 比LoRA好1-3%
    → → RTX 4090: LoRA是性价比最优! 单卡训练+95%性能!

  GRPO LoRA vs 全量:
    → GRPO LoRA: 1GPU → $0.35/hr → 95%效果
    → GRPO 全量: 2GPU → $0.70/hr → 100%效果 → 但PCIe scaling差
    → → RTX 4090推荐GRPO+LoRA → 单卡训练 → 性价比最优!
```

## 7. LoRA实用指南

```
LoRA配置决策树:

  rank选择:
    → r=4: 最小 → 简单任务(风格转换) → 参数最少
    → r=8: 默认 → 通用微调 → 95%性能 → 推荐!
    → r=16: 高秩 → 复杂任务(代码/数学) → 98%性能
    → r=32+: 通常不必要 → rsLoRA可以帮助稳定
    → → 规律: 任务越复杂→需要更高rank → 但r>16通常收益很小

  target_modules:
    → 最小: q_proj, v_proj → 仅attention → 参数最少 → 80%性能
    → 推荐: q_proj, k_proj, v_proj, o_proj → 全attention → 90%性能
    → 全量: + gate_proj, up_proj, down_proj → +MLP → 95%性能 → 推荐!
    → → MLP LoRA很重要 → 尤其数学/代码任务(知识存储在MLP)

  alpha选择:
    → α=16 (r=8): α/r=2 → 中等强度 → 推荐!
    → α=32 (r=8): α/r=4 → 强适应 → 快收敛但可能过拟合
    → α=r: α/r=1 → 最弱 → 需要更多步 → 但最稳定
    → → 规律: α/r ≈ 2-4 → 多数任务最优

  学习率:
    → LoRA: lr=1e-4 → 默认
    → LoRA+: lr_A=1e-3, lr_B=6e-5 → A比B高16x → 收敛更快
    → → 实测: LoRA+在相同步数下优于LoRA → 但需要更仔细调参

  合并时机:
    → 训练完 → 合并LoRA → 推理零开销 → 推荐!
    → 量化模型 → 不合并 → runtime应用 → vLLM支持
    → Multi-LoRA → 不合并 → SegMM → vLLM V1支持
    → → 单adapter推理: 合并最优 → 多adapter推理: 不合并+SegMM最优

  RTX 4090最优LoRA配置:
    → 7B BF16: r=8, α=16, 全target_modules → 单卡训练 → 推荐!
    → 7B INT4+LoRA: r=8, α=16 → 单卡轻松 → INT4训练需fused kernel
    → GRPO LoRA: r=8, actor+critic各一个adapter → 单卡训练 → 推荐!
    → Multi-LoRA serving: 10个adapter r=8 → INT4 base + SegMM → 推荐!
```

---

**Sources**:
- [LoRA Paper (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)
- [DoRA Paper (NVIDIA 2024)](https://arxiv.org/abs/2402.09197)
- [PiSSA (2024)](https://arxiv.org/abs/2404.02948)
- [rsLoRA (2024)](https://arxiv.org/abs/2312.03732)
- [Punica (2024)](https://arxiv.org/abs/2312.03732)
- [S-LoRA (2024)](https://arxiv.org/abs/2311.03285)
- [vLLM Multi-LoRA Blog](https://blog.vllm.ai)
- [verl GitHub](https://github.com/volcengine/verl)

**Related notes**: kv-cache-management-deep-dive.md, inference-cost-analysis.md, rl-alignment-unified-comparison.md