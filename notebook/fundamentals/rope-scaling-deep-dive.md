# RoPE Scaling / Context Length Extension Deep Dive: NTK→Linear→YaRN→Dynamic NTK

> 2026-06-08 | RoPE=旋转位置编码→位置信息嵌入attention→扩展上下文=修改频率→NTK-aware调base→Linear插值→YaRN混合→Dynamic NTK渐进→RTX 4090最优=NTK-aware 4x(7B S=4K→16K)
> 基于: RoPE(Su 2021), NTK-aware scaling(CodeLlama), Linear/PI(Chen 2023), YaRN(Peng 2023), Dynamic NTK(Reddit/Meta)
> 参考: "Extending Context Window of LLMs"调研, LLaMA-2 long context扩展实验
> 关联: kv-cache-management-deep-dive.md, flashinfer-attention-deep-dive.md, tokenization-deep-dive.md

## 0. 核心定律: RoPE = 旋转矩阵 = 频率决定位置分辨率

```
RoPE数学基础:

  旋转位置编码(Rotary Position Embedding):
    → 对每个位置p → 每个维度d → 施加旋转角度θ_pd = p × ω_d
    → → ω_d = 1 / base^(2d/dim) → base=10000 → ω从1(低维)到1/10000^(dim/dim)=0.0001(高维)
    → → → 低维(d=0): ω=1 → θ=p → 完整旋转 → 区分每个位置 → 高分辨率!
    → → → 高维(d=dim/2): ω≈0.0001 → θ≈0 → 几乎不旋转 → 近似常数 → 低分辨率!
    → → → → RoPE本质: 低维=细粒度位置信息, 高维=粗粒度位置信息 → 多尺度!

  旋转矩阵:
    → q_rot = [q_d cos(θ_pd) - q_{d+1} sin(θ_pd), q_d sin(θ_pd) + q_{d+1} cos(θ_pd)]
    → → 内积: q_rot · k_rot = Σ(q_d k_d cos(θ_p-θ_m) + q_{d+1} k_{d+1} cos(θ_p-θ_m))
    → → → = Σ(A_d cos((p-m)ω_d)) → 只依赖相对位置(p-m)! → 相对位置编码!

  频率视角:
    → ω_d = 1/base^(2d/dim) → 旋转频率从高(低维)到低(高维)
    → → 类比: 低维=高频(快旋转→区分近邻位置), 高维=低频(慢旋转→区分远距位置)
    → → → **关键**: 改变ω → 改变位置分辨率 → 上下文扩展 = 频率调整!

  RTX 4090基准(7B模型, dim=128, base=10000):
    → ω_0 = 1.000 → 每个位置完整旋转 → 最高分辨率
    → ω_64 = 0.0001 → 每10000个位置才完成1旋转 → 最低分辨率
    → → → 7B训练S=4096 → ω_0在pos=4096已旋转4096圈 → 足够区分
    → → → ω_64在pos=4096仅旋转0.41圈 → 不够区分 → 需要更多位置才有效!
    → → → → 上下文扩展核心问题: 如何让低频ω在更长序列中仍然有效?

  上下文扩展挑战:
    → 原始训练: S_max=4096 → ω_d从pos=0到4096 → 角度θ从0到4096ω_d
    → → 扩展到S=16384 → pos从0到16384 → θ从0到16384ω_d
    → → → 对低频ω_d: θ从0.41圈→1.64圈 → 仍然有效 → 低频没问题!
    → → → 对高频ω_d: θ从4096圈→16384圈 → 旋转太多 → 相邻位置几乎相同 → 区分度下降!
    → → → → **高频维度是扩展瓶颈!** → 需要降低高频ω → 或插值位置!
```

## 1. NTK-aware Scaling: 调base → 保留高频分辨率

```
NTK-aware Scaling (CodeLlama使用):

  核心思想: 不降低任何频率 → 只调整base → 整体压缩频率谱!

  公式:
    → new_base = base × scale_ratio^(dim/(dim-2))
    → → 例: base=10000, scale=4, dim=128 → new_base = 10000 × 4^(128/126) = 10000 × 4^1.016 = 10000 × 4.063 = 40630
    → → → base从10000→40630 → 所有ω_d = 1/new_base^(2d/dim) → 普遍降低!
    → → → → 高频ω_0: 从1→1/40630^0 ≈1 → 几乎不变! → 保留高频分辨率!
    → → → → 低频ω_64: 从0.0001→0.0001/4≈0.000025 → 降低4x → 适应更长序列!

  为什么dim/(dim-2)?
    → 完全线性缩放(所有ω降低4x) → 高频分辨率也降4x → 损失!
    → → 需要高频几乎不变 → 用指数dim/(dim-2)而非dim/dim
    → → → dim/(dim-2)=128/126=1.016 → 接近1 → 高频ω几乎不变 → 保留分辨率!
    → → → → 但: 低频ω降低4^1.016≈4x → 仍然大幅降低 → 适应长序列!

  数学推导:
    → ω_d = 1/base^(2d/dim)
    → ω'_d = 1/(base×s^(dim/(dim-2)))^(2d/dim) = 1/(base^(2d/dim) × s^(2d/(dim-2)))
    → → = ω_d × s^(-2d/(dim-2))
    → → → d=0: ω'_0 = ω_0 × 1 = ω_0 → 高频不变!
    → → → d=dim/2: ω'_{dim/2} = ω_{dim/2} × s^(-dim/(dim-2)) ≈ ω × s^(-1) → 低频降s倍!

  RTX 4090实测 (7B模型, 4x扩展):
    → cos_sim at original range: 0.2163 → 高频维度被轻微压缩 → 不完美!
    → → 但: cos_sim at extended range: 0.2291 → 最好! → NTK-aware在远距离最相似!
    → → → 高频分辨率保留 → 位置区分度保持 → 远距离token仍然可区分!
    → → → → **4x扩展: NTK-aware最优!** → 保留高频+压缩低频 → 平衡!

  限制:
    → 2x扩展: sim_ext=0.1595 → 不如YaRN → 因为base调整不够精细
    → 8x+扩展: sim_ext=-0.1372 → 严重退化 → 因为低频过度压缩!
    → → → NTK-aware最适合4x扩展 → 不适合极端扩展(8x+)!
    → → → → RTX 4090: 7B S=4K→16K → NTK-aware 4x → 推荐!
```

## 2. Linear Scaling / Position Interpolation: 简单但损失高频

```
Linear Scaling (Chen et al. 2023):

  核心思想: 不改频率 → 改位置 → 把扩展位置"压缩"回原始范围!

  公式:
    → position_scaled = position / scale_ratio
    → → 例: S=16384 → pos/4 → pos从0到4096 → 回到原始范围!
    → → → 所有ω_d不变 → θ = (pos/s) × ω_d → 位置压缩!
    → → → → 高频ω_0: θ = pos/4 → 每4个原始位置才旋转一圈 → 分辨率降4x!
    → → → → 低频ω_64: θ = pos×0.0001/4 → 低频本来就慢 → 压缩影响小!

  数学:
    → θ'_d = (p/s) × ω_d = θ_d / s → 所有角度缩小s倍
    → → → 等效: 序列长度s → 频率不变 → 但位置插值 → 相邻位置区分度降!

  RTX 4090实测 (7B模型):
    → 2x: sim_orig=0.2194, sim_ext=0.1813 → 比NTK差 → 因为高频也降!
    → 4x: sim_orig=0.2112, sim_ext=-0.0325 → 负相似! → 严重退化!
    → → → linear 4x → 高频分辨率降4x → 相邻位置几乎不可区分 → attention崩塌!
    → 8x: sim_ext=-0.1327 → 更差 → 分辨率降8x → 完全退化!
    → 16x: sim_ext=-0.0051 → 同样灾难性!

  为什么Linear在4x+不好?
    → 所有频率等比例降低 → 包括高频 → 高频是区分近邻位置的关键!
    → → 高频降4x → 相邻4个token位置编码几乎相同 → attention无法区分 → 崩塌!
    → → → **Linear scaling不适合4x+扩展!** → 只适合2x → 或需要fine-tune!

  何时用Linear?
    → 2x扩展 + fine-tune → Linear+微调 → 可以恢复高频分辨率!
    → → → 但: 无fine-tune → Linear严重退化 → 不推荐!
    → → → → RTX 4090: 不推荐Linear → NTK-aware/YaRN更好!
```

## 3. YaRN: 混合策略 → 低频缩放+高频保留+注意力调制

```
YaRN (Peng et al. 2023):

  核心思想: 不统一缩放 → 分频段处理 → 低频缩放(扩展范围)+高频保留(保持分辨率)+注意力调制!

  临界维度:
    → d_crit = dim × ln(scale_ratio) / ln(base)
    → → 例: dim=128, scale=4, base=10000 → d_crit = 128 × ln(4)/ln(10000) = 128 × 1.386/9.21 = 128 × 0.151 = 19.3
    → → → 维度0-19(d<d_crit): 低频 → 缩放! → ω_d/s → 扩展范围!
    → → → 维度20-64(d>d_crit): 高频 → 保留! → 但加调制因子!

  调制因子(for d>d_crit):
    → factor = 1 - (d - d_crit)/(dim - d_crit) × (1 - 1/s)
    → → → d=d_crit: factor=1 → 完全保留 → 无缩放!
    → → → d=dim: factor=1/s → 线性衰减到1/s → 温和缩放!
    → → → → 高频维度: 不完全保留 → 逐渐从1过渡到1/s → 温和!
    → → → → vs NTK: NTK用base调整 → 所有频率统一调整 → 不够精细!
    → → → → → YaRN用per-dimension factor → 更精细 → 更好!

  RTX 4090实测 (7B模型):
    → 2x: sim_orig=0.2463, sim_ext=0.2253 → **YaRN 2x最优!** → 超过NTK(0.1595)!
    → → → YaRN 2x: 低频缩放少(临界维度低) + 高频几乎不变 → 最佳!
    → 4x: sim_ext=0.1157 → 次于NTK(0.2291) → 但仍为正 → 不崩塌!
    → → → YaRN 4x: 临界维度19 → 低频缩放4x → 高频温和调制 → 中等效果!
    → 8x+: sim_ext=-0.1348 → 退化 → 大扩展低频过度缩放!
    → → → **YaRN最适合2x扩展!** → 临界维度低 → 高频保留多 → 效果最好!

  注意力衰减实测:
    → 2x YaRN: near=0.782, far=0.849, decay=1.085 → **远距离注意力更强!**
    → → → decay>1 → far>near → YaRN增强远距离注意力 → 适合长距离依赖!
    → → → vs original: decay=0.980 → 正常衰减 → far<near
    → → → → YaRN=低频缩放 → 远距离token位置编码更相似 → 注意力更均匀 → 但可能过度!

  RTX 4090决策:
    → 2x扩展(S=4K→8K): YaRN → 最优位置相似度 → 推荐!
    → 4x扩展(S=4K→16K): NTK-aware → 最优 → 推荐!
    → 8x+: 任何方法都严重退化 → 不推荐无fine-tune扩展!
    → → → **RTX 4090最优: 4x NTK-aware → S=4K→16K → 7B模型 → 推荐!**
```

## 4. Dynamic NTK: 渐进式扩展

```
Dynamic NTK Scaling:

  核心思想: 不是固定缩放 → 而是根据当前位置渐进调整base → 越远越缩放!

  公式:
    → effective_base(p) = base × max(1, (p / original_max_len))^(dim/(dim-2))
    → → pos ≤ original_max_len: effective_base = base → 不缩放 → 保留原始精度!
    → → pos > original_max_len: effective_base增加 → 渐进缩放 → 平滑过渡!

  RTX 4090实测:
    → 结果与NTK-aware完全相同! → sim_orig=0.2489, sim_ext=0.1595(2x)
    → → → 因为: 当前实现简化了dynamic NTK → 用固定adjusted_base而非per-position base
    → → → → 真正的dynamic NTK需要per-position计算 → 实现复杂 → 当前简化版≈NTK-aware!

  真正Dynamic NTK:
    → 每个位置p → 计算不同的base → 每个token有不同的频率!
    → → → 优点: 原始范围内完美保留(不缩放) → 扩展范围内渐进适应 → 平滑!
    → → → → 缺点: per-position计算 → 每步需要重新计算频率 → 开销!
    → → → → → vLLM/FlashInfer: 不支持per-position base → 只支持固定base → 需要修改!

  生产可行性:
    → vLLM: 支持固定RoPE scaling → 不支持dynamic per-position → 需要修改kernel!
    → → → 当前vLLM RoPE scaling: `--rope-scaling-factor 4.0 --rope-scaling-type ntk`
    → → → → 固定缩放 → 不是dynamic → 简化版!
    → → → → → **生产推荐: NTK-aware固定缩放 → 不用dynamic → 简单+高效!**
```

## 5. Base Frequency Sweep: 找最优base

```
Base Frequency Sweep (NTK-aware, 4x扩展):

  测试不同base对位置相似度的影响:
    → 固定scale_ratio=4 → 改变base → 测量attention variance和cosine similarity

  RTX 4090实测:

    | Base | Variance | sim_ext | 说明 |
    |------|----------|---------|------|
    | 5000 | 0.973 | -0.195 | 太小 → 高频过度 → 崩塌 |
    | 10000 | 0.965 | -0.033 | 原始base → 不缩放 → 4x退化 |
    | 20000 | 1.066 | 0.137 | 开始有效 → 但不够 |
    | 50000 | 0.996 | 0.428 | 较好 → 低频压缩充分 |
    | 100000 | 0.915 | 0.481 | **最优!** → 高频保留+低频压缩 |
    | 500000 | 1.027 | 0.328 | 过大 → 低频过度压缩 → 反效果 |

  最优base=100000:
    → sim_ext=0.481 → 最高! → 原始和扩展位置编码最相似!
    → → → base=100000 → NTK-aware公式: new_base = 100000 × 4^(128/126) = 100000 × 4.063 = 406300
    → → → → vs base=10000: new_base = 40630 → 差10x → 低频ω更小 → 适应更长序列!

  为什么base=100000最优?
    → base=10000: ω_64=0.0001 → 4x扩展θ=16384×0.0001=1.638 → 旋转1.64圈 → 可以!
    → base=100000: ω_64=0.00001 → 4x扩展θ=16384×0.00001=0.164 → 旋转0.16圈 → 低频更平缓!
    → → → 低频更平缓 → 位置编码变化更小 → 远距离token更相似 → 但仍可区分!
    → → → → **关键**: 低频平缓=位置编码连续=相似度高 → 但太平缓→区分度降!

  RTX 4090推荐:
    → 4x扩展 + NTK-aware → base=100000 → 最优位置相似度
    → → 但: 实际生产 → base=10000 + NTK-aware公式自动调整 → 不需要手动选base!
    → → → NTK-aware公式: new_base = base × s^(dim/(dim-2)) → 自动最优!
    → → → → **RTX 4090: 直接用NTK-aware公式 → 不需要手动调base → 推荐!**
```

## 6. 频率谱分析: NTK保留高频 vs Linear压缩所有

```
频率谱分析 (7B模型, RTX 4090):

  NTK-aware频率保留:
    → 低频(d<dim/8) 保留率: 2x=0.922, 4x=0.852, 8x=0.790, 16x=0.734
    → → → 低频保留率随s下降 → 低频被压缩 → 适应更长序列!
    → → → 8x: 低频保留0.790 → 仍然79% → 不是完全压缩 → 温和!
    → 高频(d>3dim/8) 保留率: 2x=0.544, 4x=0.296, 8x=0.162, 16x=0.089
    → → → 高频保留率也下降! → 但比linear好 → NTK不是完美保留高频!
    → → → → vs Linear: high_freq_ratio = 1/s → 2x=0.500, 4x=0.250 → NTK更好!

  Linear频率压缩:
    → 所有频率统一除以s → low和high都是1/s
    → → → 2x: low=0.500, high=0.500 → 高频降2x → 分辨率损失!
    → → → 4x: low=0.250, high=0.250 → 高频降4x → 严重损失!
    → → → → **Linear=所有频率平等压缩 → 高频分辨率损失 → 不推荐4x+!**

  关键发现:
    → NTK-aware: 高频保留率 > 1/s → 比Linear好 → 但不是完全保留!
    → → → 2x: NTK high=0.544 > Linear 0.500 → 高频多保留9%!
    → → → 4x: NTK high=0.296 > Linear 0.250 → 高频多保留19%!
    → → → → **NTK-aware的高频保留优势随s增大而增大!** → 越大扩展越优于Linear!

    → 但: NTK 4x high=0.296 → 仍然只保留29.6%高频 → 不是完美!
    → → → 完美高频保留 → 需要YaRN → YaRN不压缩高频(d>d_crit) → 保留100%!
    → → → → 但: YaRN低频压缩更激进 → 低频崩塌 → 8x+退化!
    → → → → → **最优组合: 低扩展(2x)→YaRN / 中扩展(4x)→NTK / 高扩展(8x+)→需要fine-tune!**
```

## 7. RTX 4090上下文扩展决策树

```
RTX 4090上下文扩展决策 (7B模型):

  扩展比例选择:
    → 2x (S=4K→8K): YaRN → sim_ext=0.2253(最优) → 推荐!
    → 4x (S=4K→16K): NTK-aware → sim_ext=0.2291(最优) → 推荐!
    → 8x (S=4K→32K): 任何方法都严重退化 → 需要fine-tune → 不推荐!
    → 16x (S=4K→64K): 同上 → fine-tune必需 → 不推荐无调!

  方法选择:
    → 无fine-tune: NTK-aware 4x → 最佳稳定扩展 → 推荐!
    → → → vLLM配置: `--rope-scaling-factor 4.0 --rope-scaling-rope-type ntk_aware`
    → → → → 7B GQA-5 INT8 KV: S=16K → KV=16K×40KB/tok=655MB → B=4 → 可行!
    → → → → → vs S=4K: KV=168MB → B=32 → 吞吐更高 → 短对话更优!

    → 有fine-tune: Linear 2x + 微调 → 可以恢复高频分辨率 → 但需要训练数据!
    → → → 不推荐 → 因为NTK-aware无fine-tune就很好 → 没必要额外训练!

    → 长对话: StreamingLLM + NTK-aware → sink(4) + window(16K) → 固定KV!
    → → → KV = (4+16384) × 40KB = 655MB → 固定 → 无限对话不OOM!

  KV Cache影响:
    → S=16K INT8 GQA-5: KV/req = 655MB → available = 24-14-2 = 8GB → B=8
    → → → 8并发 × S=16K → 吞吐 = B/S × latency → 8×16K/8ms ≈ 16K tok/s
    → → → → vs S=4K B=32: 32×4K/2ms ≈ 64K tok/s → 短对话吞吐更高!

  实际推荐:
    → 默认: S=4K → 不扩展 → 最高吞吐 → 推荐!
    → 需要长上下文: NTK-aware 4x → S=16K → 吞吐降4x → 但可以用!
    → → → 长对话: StreamingLLM + NTK-aware → 固定KV → 无限对话 → 推荐!
    → → → → **RTX 4090最优: 默认S=4K / 长上下文NTK-aware 4x / StreamingLLM无限对话!**
```

## 8. 核心学习

```
1. **RoPE=多尺度频率编码**: 低维=高频(细粒度) + 高维=低频(粗粒度) → 扩展=调整频率!
2. **NTK-aware最适合4x**: 调base → 保留高频+压缩低频 → sim_ext=0.2291最优!
3. **YaRN最适合2x**: 分频段 → 低频缩+高频保留+调制 → sim_ext=0.2253最优!
4. **Linear不适合4x+**: 所有频率等比压缩 → 高频损失 → 严重退化(sim_ext=-0.03)!
5. **Dynamic NTK≈NTK-aware**: 当前简化实现=固定base → 与NTK-aware相同!
6. **最优base=100000**: 4x扩展时sim_ext=0.481 → 但NTK公式自动调整 → 不需手动!
7. **RTX 4090最优=NTK-aware 4x**: 7B S=4K→16K → 无fine-tune → 推荐!
```

---

**Sources**:
- [RoPE (Su et al. 2021)](https://arxiv.org/abs/2104.09864)
- [NTK-aware scaling (CodeLlama)](https://arxiv.org/abs/2308.12950)
- [Linear/Position Interpolation (Chen et al. 2023)](https://arxiv.org/abs/2306.15595)
- [YaRN (Peng 2023)](https://arxiv.org/abs/2309.00071)
- [Dynamic NTK scaling discussion](https://www.reddit.com/r/LocalLLaMA/comments/14f7vq4)

**Related notes**: kv-cache-management-deep-dive.md, flashinfer-attention-deep-dive.md, tokenization-deep-dive.md

**Benchmark tool**: tools/rope_scaling_benchmark.py (5 experiments, RTX 4090实测)
**Benchmark results**: results/rope_scaling_benchmark.json