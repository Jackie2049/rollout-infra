# Diffusion Models Deep Dive — 从DDPM到Flow Matching的范式革命

> 2026-06-10 | 生成式AI的第二次范式革命: DDPM→DDIM→Flow Matching→Rectified Flow→Consistency Models, U-Net→DiT→MMDiT, RTX 4090推理分析
> 关联: vlm-inference-benchmark-rtx4090.md, weight-quantization-deep-dive.md, gpu-microarchitecture-sm89-sm90-sm100.md

## 0. 扩散模型范式演进: 4代迭代

```
扩散模型4代演进:
  → 1代 DDPM(2020 Ho et al.): 噪声→去噪→随机采样→1000步→质量好但太慢!
  → → 2代 DDIM(2021 Song et al.): 确定性ODE→50-100步→快但质量略差→加速推理!
  → → → 3代 Flow Matching(2022 Lipman): 向量场匹配→ODE路径→更简单训练→更直路径!
  → → → → 4代 Rectified Flow+Consistency(2023→2025): 直线ODE→1-4步→实时生成!

与Transformer对比:
  → Transformer: 自回归→顺序生成token→1步/1token→确定性→文本/代码
  → → Diffusion: 噪声→去噪→并行生成→N步/整张图→概率性→图像/视频
  → → → → → Diffusion和Transformer不是竞争→是互补→文本用Transformer/图像用Diffusion!
  → → → → → → 2025融合: DiT(Transformer架构+Diffusion训练)→两者结合→SD3/Sora!

RTX 4090影响:
  → Diffusion推理: VAE encode→DiT N步→VAE decode → 每步独立→batch友好!
  → → 但! N步=50→总推理50x→慢! → 需要加速(Flow Matching→1-4步→12-50x加速!)
  → → → VAE decode=memory-bound→RTX 4090瓶颈→768×768→0.3s→但1024→1.5s→更大更慢!
```

## 1. DDPM — 噪声去噪扩散

```
DDPM核心机制:
  → 前向过程: x_0 → x_1 → ... → x_T → 加噪声→逐步变噪!
  → → 数学: x_t = √(1-β_t)·x_{t-1} + √β_t·ε → 马尔可夫链→每步加噪声!
  → → → → 一步公式: x_t = √α_t·x_0 + √(1-α_t)·ε → 直接从x_0到x_t→不需要逐步!

  反向过程(去噪):
    → 学习p(x_{t-1} | x_t) → 每步去一点噪声 → T步恢复!
    → → 关键: 学习score function ∇log p(x_t) → 噪声的梯度方向 → 去噪方向!
    → → → DDPM训练loss: L = E[||ε - ε_θ(x_t, t)||²] → 学习预测噪声ε!

  数学本质:
    → Score-based generative model → 学习数据分布的梯度
    → → Langevin dynamics: x_{t+1} = x_t + σ²/2 · ∇log p(x_t) + σ·ε → 采样!
    → → → Score = ∇log p(x) → 分布的梯度 → 指向高密度区域 → 采样正确!

  DDPM采样:
    → 从x_T(纯噪声) → 逐步去噪 → T=1000步 → 每步1次模型推理 → 1000×慢!
    → → → → SDPM问题: 1000步推理→太慢→每步小改进→需要很多步→不可实时!
```

## 2. DDIM — 确定性加速采样

```
DDIM核心(2021 Song et al.):
  → 关键: DDPM随机→DDIM确定性→非随机ODE→加速!
  → → 数学: DDPM→随机SDE→DDIM→确定性ODE→去掉随机噪声→纯ODE路径!
  → → → → DDIM = DDPM的确定性ODE解释 → 同一训练→不同采样→10-50步!

  DDIM加速:
    → 步数可减少: T=1000→S=50→20x加速! → 但S太少→质量下降!
    → → → S=50: 质量≈DDPM → S=20: 轻微质量损失 → S=10: 明显质量下降!
    → → → → → → DDIM=生产默认 → S=20-50步 → 质量可接受 → 20-50x加速!

  DDIM公式:
    → x_{t-1} = √α_{t-1}·(x_t - √(1-α_t)·ε_θ)/√α_t) + √(1-α_{t-1})·ε_θ
    → → → → 确定性→无随机噪声→每步精确→比DDPM更稳定!

  与DDPM对比:
    → DDPM: 随机SDE → 每步加小噪声 → 采样有随机性 → T=1000 → 慢!
    → → DDIM: 确定性ODE → 每步精确去噪 → 采样确定性 → S=20-50 → 快!
    → → → → → 同一模型! → DDPM训练 → DDIM采样 → 不需要重新训练!

  DPM-Solver++(2022):
    → 高阶ODE solver → 二阶/三阶 → 更少步达到相同质量!
    → → S=10-20步 → 质量≈DDIM S=50 → 5x加速! → 生产推荐!
```

## 3. Flow Matching — 向量场学习

```
Flow Matching核心(2022 Lipman et al.):
  → 范式变化: 不学score(∇log p) → 学velocity field(v) → 更简单!
  → → 数学: dx/dt = v_t(x) → ODE → 从noise到data的路径 → 速度场!
  → → → → v_t(x) = 从x_t到x_0的方向 → 学这个方向 → 然后ODE积分 → 采样!

  训练目标:
    → Conditional Flow Matching: L = E[||v_t(x) - u_t(x_0, x_t)||²]
    → → u_t = x_0 - x_t → 理想速度(指向data) → 学习接近理想速度!
    → → → → → 比DDPM简单! → 不需要score function → 直接学velocity → 更stable!

  与DDPM对比:
    → DDPM: 学score ∇log p(x_t) → 需要估计噪声分布的梯度 → 复杂!
    → → Flow Matching: 学velocity v_t(x) → 指向data方向 → 简单直观!
    → → → → → → 训练更稳定 → 收敛更快 → 采样更确定性 → 2025趋势!

  OT-CFM(Optimal Transport Conditional Flow Matching):
    → 核心: noise-data配对 → 不是随机配对 → 最优传输配对 → 最短路径!
    → → → → → 最短路径=直线 → ODE轨迹更直 → 更少步 → 更快采样!
    → → → → → → SD3使用OT-CFM → 最短路径 → 2-4步高质量 → 生产级!

  数学直觉:
    → DDPM/DDIM: 路径弯曲 → 需要很多步精确跟随弯路 → 慢!
    → → Flow Matching: 路径更直 → 更少步足够 → 快!
    → → → → → OT-CFM: 路径近乎直线 → 1-4步! → 实时生成!
```

## 4. Rectified Flow — 直线ODE路径

```
Rectified Flow核心(2022→2023 Liu et al.):
  → 目标: 让ODE路径变直 → 直线→1步积分足够 → 实时生成!
  → → 方法: Reflow → 重复reflow → 每次轨迹更直 → 逼近直线!
  → → → → → → 类似: gradient descent → 每步更接近最小 → reflow每步更直!

  Reflow过程:
    → Step 0: 训练初始RF → ODE路径弯曲 → 需20-50步
    → → Step 1: 用ODE生成(noise→data) → 记录轨迹 → 用轨迹重新训练 → 路径更直
    → → → Step 2: 再reflow → 更直 → 需要2-3次reflow → 路径接近直线
    → → → → → → → Reflow后: 1-2步高质量 → 实时生成!

  数学:
    → Reflow = 最优传输 → 每次reflow → 路径变直 → 最小化transport cost
    → → → → → → → 直线 = 最短路径 = 最优传输 → 从noise到data最短距离!

  SD3/Flux使用:
    → SD3: Rectified Flow + OT配对 → 训练 → 生成4-8步高质量
    → → SD3.5 Turbo: Reflow后 → 4步 → 质量≈50步DDIM → 12.5x加速!
    → → → Flux(Black Forest Labs): Rectified Flow → 生产级 → 高质量
    → → → → → → → 2025: Rectified Flow = 生产首选 → SD3/Flux都用!

  与DDIM对比:
    → DDIM: 弯曲ODE → 20-50步 → 需要精确积分 → 步数不能太少
    → → Rectified Flow: 直线ODE → 1-4步 → 简单积分 → 实时可行!
    → → → → → → → 关键区别: DDIM路径弯曲(随机noise-data配对) / RF路径直线(OT配对)!
```

## 5. Consistency Models — 一步生成

```
Consistency Models核心(2023 Song et al.):
  → 目标: 一步生成 → 从任何noise level→一步到clean image → 极快!
  → → 数学: f(x_t) = x_0 → 一步映射 → 不需要ODE积分 → 一步推理!
  → → → → → 自一致性约束: f(x_t) = f(x_{t+ε}) → 任何t的输出相同 → 连续性!

  Consistency Training (CT):
    → 从零训练 → 不需要预训练模型 → 直接学consistency function
    → → → 但! CT质量上限较低 → 需要很多数据 → 生产不太用!

  Consistency Distillation (CD):
    → 用预训练DDPM/DDIM模型 → 蒸馏成consistency模型 → 更高质量!
    → → → CD = 蒸馏 → 从50步模型→1步模型 → 质量≈50步的90-95% → 快50x!
    → → → → → → 生产: 先训练DDPM(慢但高质量) → 蒸馏成CM(快但略低质量)

  2025 Consistency Trajectory Models (CTM):
    → 多步consistency → 不是一步 → 1-4步 → 质量更好!
    → → → → → → CTM = Consistency + ODE trajectory → 融合两者优点!

  LCM(Latent Consistency Models):
    → 在latent space做consistency → VAE空间→更快→SDXL-Turbo用!
    → → → → → → LCM = 4-8步 → 质量可接受 → 实时推理 → 生产可用!

  与Rectified Flow对比:
    → CM: 蒸馏 → 需要预训练模型 → 一步→极快 → 但质量天花板较低
    → → RF: 独立训练 → 不需要蒸馏 → 2-4步→快 → 质量天花板更高
    → → → → → → → 生产: RF(独立训练+4步) vs CM(蒸馏+1步) → RF更灵活!
```

## 6. Architecture Evolution — U-Net → DiT → MMDiT

```
架构演进:
  → U-Net(2020 DDPM): 卷积+下采样+上采样 → skip connection → 图像标准!
  → → → 问题: 不可scaling → 参数固定 → 计算不灵活 → 不能像Transformer scaling!

  DiT(2023 Peebles & Xie):
    → Diffusion Transformer → patch→token → Transformer blocks → 可scaling!
    → → → → → 类似ViT → 但训练用diffusion loss → 不是分类loss!
    → → → → → → 关键: DiT scaling law → 参数↑ → FID↓ → 类似LLM scaling law!
    → → → → → → → → DiT-XL/2: 675M → FID=2.27 → 比U-Net更好!

  MMDiT(2024 SD3):
    → Multi-Modal DiT → 文本+图像 → 双流Transformer → 融合注意力!
    → → → → → 文本流: CLIP-L + OpenCLIP-G + T5-XXL → 3个编码器
    → → → → → → 图像流: VAE latent → patch → DiT blocks
    → → → → → → → 融合: cross-attention → 文本条件 → 图像生成
    → → → → → → → → QK-norm(query-key归一化) → 训练稳定 → SD3关键!
    → → → → → → → → → SwiGLU + RMSNorm + RoPE → 与LLaMA类似!

  Video DiT(2024→2025):
    → Sora: 3D patch → video token → DiT + temporal attention → 视频!
    → → → CogVideoX: 3D RoPE + DiT + Flow Matching → 视频生成
    → → → → → → → 3D patch = 空间+时间 → Transformer处理时空 → 新维度!

  RTX 4090架构影响:
    → DiT推理: 每步=matmul密集 → 类似prefill → compute-bound → GPU友好!
    → → → → → 但! N步 → 总计算=N×单步 → N=4→4×计算 → N=50→50×计算!
    → → → → → → → Rectified Flow N=4 → 4×单步 → 比50×DDIM快12.5x!
    → → → → → → → → → DiT单步 ≈ Prefill ≈ compute-bound → RTX 4090高效!
    → → → → → → → → → → → 但VAE decode = memory-bound → 独立瓶颈!
```

## 7. Latent Diffusion — VAE压缩空间

```
Latent Diffusion核心(2022 Rombach et al. → SD1.5/SDXL/SD3):
  → 关键创新: 不在pixel space做diffusion → 在VAE latent space → 8x压缩!
  → → → → → pixel 512×512×3=786K → latent 64×64×4=16K → 48x压缩!
  → → → → → → → → → → latent space做diffusion → 快48x → 然后VAE decode回pixel!

  VAE架构:
    → Encoder: pixel→latent → 下采样8x → 4通道 → KL-regularized
    → → → Decoder: latent→pixel → 上采样8x → 3通道 → RGB
    → → → → → → → → → → → SD3 VAE更大 → 但RTX 4090可以处理!

  Latent Diffusion Pipeline:
    → Step 1: text→CLIP/T5→text embedding → 条件编码
    → → Step 2: noise→latent→N步DiT→clean latent → diffusion采样
    → → → Step 3: clean latent→VAE decode→pixel image → 解码
    → → → → → → → → → → → → → → 关键: DiT步数决定速度 → VAE decode固定!

  RTX 4090 Latent Diffusion推理:
    → VAE encode: 0.1ms → 很快(只1次) → 可忽略
    → → → → DiT每步: 512×512→768×768约0.05s(compute-bound) → N=4→0.2s → N=50→2.5s
    → → → → → VAE decode: 768×768约0.3s(memory-bound) → 固定开销 → 不可省!
    → → → → → → → → → → Total RF(4步): 0.2s + 0.3s = 0.5s → 实时!
    → → → → → → → → → → → Total DDIM(50步): 2.5s + 0.3s = 2.8s → 慢!
    → → → → → → → → → → → → → → → RTX 4090最优 = Rectified Flow 4步 → 0.5s → 实时生成!
```

## 8. RTX 4090 Diffusion推理优化

```
RTX 4090 Diffusion推理分析:
  → DiT推理: compute-bound → matmul密集 → 类似LLM prefill → 高GPU利用率!
  → → → 768×768 DiT: 每步约0.05s → FP16 TFLOPS ~170 → 93%peak → 好利用!
  → → → → → → → 但! 1024×1024 → 每步约0.12s → 仍然compute-bound → 可行!

  VAE decode瓶颈:
    → 768×768: 约0.3s → memory-bound → HBM 890GB/s → bandwidth限制!
    → → → 1024×1024: 约1.5s → memory-bound更严重 → 大图VAE是瓶颈!
    → → → → → → → 优化: tiled decode → 分块解码 → 但需要kernel优化!
    → → → → → → → → → → → → → → → → → → → → → → → VAE FP16→INT8量化→50%带宽省→可能0.15s!

  整体优化策略:
    → 1. Rectified Flow 4步 → 从50步降到4步 → 12.5x加速 → 最大优化!
    → → → 2. DiT INT4量化 → 权重INT4 → 75%带宽省 → 但质量损失需评估!
    → → → → → 3. VAE tiled decode → 分块 → 减少瞬时内存 → 可行!
    → → → → → → → 4. Batch多张图 → weight reads共享 → 吞吐↑ → 与LLM一样!
    → → → → → → → → → → 5. KV-like conditioning cache → text embedding reuse → 省计算!

  RTX 4090最优配置:
    → SD3-like: Rectified Flow 4步 + DiT 7B FP16 + VAE FP16
    → → → → → → → → → → → 768×768: ~0.5s → 实时 → 推荐!
    → → → → → → → → → → → → → → → 1024×1024: ~1.7s → 接近实时 → 可用!
    → → → → → → → → → → → → → → → → → → 512×512: ~0.3s → 极快 → 适合并发!

  与LLM推理对比:
    → LLM: decode=memory-bound → 权重读主导 → INT4量化是唯一出路
    → → Diffusion: DiT=compute-bound → matmul密集 → 量化收益小(compute足够!)
    → → → → → → → → → → → → → → → → → → LLM瓶颈=weight reads → Diffusion瓶颈=步数!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → LLM优化=量化 → Diffusion优化=少步数!
```

## 9. Core Laws — 扩散模型核心定律

```
1. Step-Quality Law: 图像质量 ∝ 步数 → 但步数↑→推理↑→线性成本!
   → → DDPM 1000步: 质量最高 → DDIM 50步: 质量≈ → DDIM 10步: 质量↓
   → → → RF 4步: 质量≈DDIM 50 → RF 2步: 质量≈DDIM 20 → RF 1步: 质量↓
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 最优: RF 4步 → 质量+速度平衡!

2. Latent-Space Law: latent diffusion比pixel快48x → 空间压缩是关键!
   → → Pixel 512² → 786K → Latent 64² → 16K → 48x压缩 → 48x加速!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 所有生产模型都用latent → 不做pixel diffusion!

3. Trajectory-Straightness Law: ODE路径越直 → 需要步数越少 → RF直→4步/DDIM弯→50步!
   → → DDPM/DDIM: 弯曲 → 需精确跟随 → 步数多 → 慢!
   → → → Flow Matching: 更直 → 步数少 → 快 → 但不是完全直线!
   → → → → → → → → → → → → → → → → → → Rectified Flow: 最直 → 1-4步 → 最快!
   → → → → → → → → → → → → → → → → → → → → → → → → → → Reflow=不断变直 → 逼近直线!

4. Architecture-Scaling Law: DiT scaling law类似LLM → 参数↑→FID↓→可预测!
   → → DiT-S: 33M → DiT-B: 130M → DiT-L: 458M → DiT-XL: 675M → FID持续下降!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Transformer scaling = diffsuion scaling → 同一族!

5. Compute-Bound Law: DiT推理=compute-bound(matmul密集) → 与LLM decode不同!
   → → LLM decode: memory-bound → 权重读主导 → INT4量化是唯一出路
   → → → DiT推理: compute-bound → matmul密集 → 量化收益小 → 步数是瓶颈!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → LLM优化策略≠Diffusion优化策略!

6. VAE-Bottleneck Law: VAE decode=memory-bound → 固定开销 → 大图瓶颈!
   → → 768² VAE: 0.3s → 1024² VAE: 1.5s → 5x慢 → memory-bound!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 优化: tiled decode / INT8量化 / 分块处理
```

## 关键论文与参考

```
- DDPM (Ho et al., 2020): 噪声去噪→1000步→质量好但慢→开创性!
- DDIM (Song et al., 2021): 确定性ODE→20-50步→加速→DDPM的ODE解释
- Flow Matching (Lipman et al., 2022): 向量场匹配→更简单训练→更直路径→2025趋势!
- Rectified Flow (Liu et al., 2022→2023): 直线ODE→reflow→1-4步→实时生成→SD3/Flux用!
- Consistency Models (Song et al., 2023): 一步生成→自一致性→1-4步→最快但质量天花板低!
- DiT (Peebles & Xie, 2023): Diffusion Transformer→scaling law→替代U-Net→SD3/Sora用!
- SD3 (Esser et al., 2024): MMDiT+Rectified Flow+OT-CFM→8B→双流Transformer→2025生产级!
- LCM (Luo et al., 2023): Latent Consistency→latent空间蒸馏→4-8步→SDXL-Turbo→实时!
- Sora (OpenAI, 2024): Video DiT→3D patch→temporal attention→视频生成→新维度!
- CogVideoX (2024): 3D RoPE+DiT+Flow Matching→视频→中国团队!

Sources:
- [DDPM](https://arxiv.org/abs/2006.11239)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [DiT](https://arxiv.org/abs/2212.09748)
- [SD3/MMDiT](https://arxiv.org/abs/2403.03206)
- [LCM](https://arxiv.org/abs/2310.04460)
- [Sora](https://openai.com/index/sora/)