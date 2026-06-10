# Neural Architecture Innovation Deep Dive — Beyond Transformer

> 2026-06-10 | Transformer之后的5代架构演进! SSM(Mamba)→Linear Attention(RWKV)→Delta Rule(DeltaNet)→Hybrid(Jamba)→Next(GPU co-design), 每代解决前代问题
> 关联: flashattention-attention-deep-dive.md, gpu-microarchitecture-sm89-sm90-sm100.md, mla-architecture-deep-dive.md

## 0. 核心定律: Transformer的O(N²)瓶颈 → 线性复杂度是出路

```
Transformer根本问题:
  → Self-Attention: O(N²) 计算复杂度 → N↑4x → compute↑16x!
  → → KV Cache: O(N) per token → N↑4x → memory↑4x → 并发↓4x!
  → → → 长上下文: N=128K → attention=128K²=16.4G → 内存灾难!
  → → → → → Transformer = 优秀架构但不是最终架构 → 需要更高效替代!

线性注意力/SSM出路:
  → 核心思想: O(N) → 线性复杂度 → N↑4x → compute↑4x(不是16x!)
  → → 但! 线性注意力 ≈ 丢失softmax归一化 → 质量损失!
  → → → → 5代演进: 每代解决"如何保持质量+降低复杂度"的矛盾!

与RTX 4090联系:
  → 7B decode: weight reads 95% → attention仅3.3% → attention不是瓶颈!
  → → → 但长上下文 S=128K → KV=6.4GB → 内存溢出 → SSM有用!
  → → → → → RTX 4090最优 = Transformer(短上下文) + SSM思路(长上下文节省)
```

## 1. Mamba — 选择性状态空间模型 (SSM)

```
Mamba (Gu & Dao, 2023):
  → 核心机制: 选择性状态空间 → input-dependent dynamics → 动态选择!
  → → SSM基础: h(t) = A·h(t-1) + B·x(t), y(t) = C·h(t) → 线性递推!
  → → → Mamba创新: A,B,C = 选择性(不是固定!) → input-dependent → 类似attention的动态权重!
  → → → → → 选择性 = "模型自己决定记住什么、忘记什么" → 类似softmax的soft selection!

  关键数学:
    → 固定SSM: h_new = A·h_old + B·x → A,B,C是常数 → 不能选择 → 类似固定attention pattern
    → → 选择性SSM: A(x), B(x), C(x) → 依赖输入 → 能选择 → 类似softmax attention!
    → → → → → 这就是为什么Mamba效果好 → 选择性 = attention的"选择能力" → 但O(N)!

  Mamba-2 (SSD框架, 2024):
    → Structured State Space Duality → SSM ≈ 半结构化attention!
    → → 数学证明: SSM = 特殊形式的attention → 矩阵结构对比!
    → → → → → SSM和attention不是对立 → 是duality! → hybrid自然!
    → → → → → → 2-8× faster training → hardware-aware → 更好GPU利用!

  复杂度对比:
    → Transformer: O(N²) compute + O(N) KV per token → 内存灾难!
    → → Mamba: O(N) compute + O(D) state per token → D=hidden_dim → 固定状态!
    → → → → → KV: Transformer=O(N)增长 → Mamba=O(D)固定 → 128K上下文可行!

  优势:
    → 长上下文: 128K tokens → KV不增长 → 内存O(1) per token → vs Transformer O(N)!
    → → 推理速度: O(N) vs O(N²) → linear → 128K = 128K操作(不是128K²!)
    → → → 内存效率: 固定state size → 不像KV cache → 内存占用恒定!

  局限:
    → 短上下文: Transformer KV小 → Mamba state≈Transformer KV → 无优势!
    → → → 7B RTX 4090: S=4K → KV=6.4MB → 很小 → Mamba无优势!
    → → → → → Mamba优势仅在N>16K → RTX 4090最优=S=4K → Mamba不是最优!
    → → → → → → 但StreamingLLM=无限上下文 → Mamba思路=无限上下文!

  RTX 4090影响:
    → Mamba kernel: selective scan → 需要CUDA实现 → 但RTX 4090(SM89)无TMA!
    → → → SM89 scan = cp.async → 比SM90 TMA慢 → 但仍可行
    → → → → → 7B decode: weight reads主导 → SSM scan不是瓶颈
    → → → → → → RTX 4090: Mamba可行但优势有限(短上下文场景)
    → → → → → → → 128K上下文 → Mamba优势明显 → 但需更大模型
```

## 2. RWKV — 线性注意力RNN

```
RWKV (Peng et al., 2023→2025→RWKV-7):
  → 核心机制: WKV (Weight-Key-Value) → 线性注意力递推!
  → → WKV公式: wkv_t = (a·wkv_{t-1} + b·k_t·v_t) / (a·wkv_{t-1} + b·k_t)
  → → → ≈ softmax attention的线性化 → 但没有exp → O(N)!
  → → → → a=decay → 控制信息保留 → 类似Mamba的选择性!
  → → → → → b=bonus → 控制新信息权重 → 类似attention的query-key匹配!

  RWKV演进:
    → RWKV-4: 固定decay → 限制 → 不能根据输入选择保留/遗忘
    → → RWKV-5: data-dependent decay → 输入依赖 → 但公式较简单
    → → → RWKV-6(Finch): 更复杂gate → 更好选择 → 但仍有局限性
    → → → → RWKV-7(Goose): 多参数递推 → 最expressive → 竞争Transformer!

  RWKV-7关键:
    → 多参数递推 → learned coefficients → 模型自己学最优递推公式!
    → → → 更expressive → 比Mamba的固定A结构灵活 → 比GLA更参数化!
    → → → → → token-shift(time-mixing) → 当前+前一个token → 简单但有效!
    → → → → → → data-dependent decay → 输入控制遗忘 → 类似Mamba选择性!

  与Transformer对比:
    → Training: parallel → 类似Transformer → 可用GPU并行训练!
    → → Inference: RNN → O(1) per token → 递推 → 不需要KV cache!
    → → → → → 训练=Transformer模式 → 推理=RNN模式 → dual模式!

  复杂度:
    → Training: O(N·D²) → 与Transformer类似 → GPU矩阵乘 → 高效!
    → → Inference: O(D²) per token → 固定 → 不增长 → vs Transformer O(N·D)!
    → → → → → RWKV推理 = 固定cost → 不像Transformer KV=O(N)增长!

  RTX 4090影响:
    → RWKV推理: O(D²) per token → 与Transformer decode相似!
    → → → 但! 无KV cache → 内存更省 → 并发更多 → 适合长上下文!
    → → → → → 7B RWKV: state=4KB vs KV=6.4MB(S=4K) → 1600x内存省!
    → → → → → → 但state小 → 并发极高 → 但信息量有限 → 短上下文质量差!
    → → → → → → → RWKV = 长上下文推理好 → 短上下文不如Transformer!
```

## 3. DeltaNet — 线性注意力 + Delta Rule

```
DeltaNet (Shi et al., 2025):
  → 核心创新: Delta Rule → 不仅add → 还能subtract+replace → overwrite!
  → → 传统线性注意力: S_new = S_old + new_kv → 只add → 不能erase!
  → → → Delta Rule: S_new = S_old + Δ → Δ = target - S_old[query_slot]
  → → → → → → 类似gradient descent → 计算error(delta) → 修正方向!

  数学直觉:
    → Additive(SSM/RWKV): S += kv → 只累积 → 不能修改已有信息 → 信息堆积!
    → → Delta Rule: S -= old_kv_at_slot + new_kv → 先删除 → 再写入 → overwrite!
    → → → → → → 类似memory的"read-write" → 不是"read-only" → 更灵活!
    → → → → → → → 类似gradient descent → error修正 → 而不是盲目累积!

  与Mamba对比:
    → Mamba: 选择性decay → 部分遗忘 → 但不能精确overwrite特定位置!
    → → DeltaNet: delta rule → 精确overwrite → 更细粒度控制!
    → → → → → → DeltaNet = 更精确的memory管理 → 但计算更复杂!

  复杂度:
    → Training: O(N·D²) → chunkwise parallel → 类似Transformer!
    → → Inference: O(D²) per token → 递推 → 固定cost → 不增长!
    → → → → → → 与RWKV/Mamba相同 → 线性复杂度!

  Gate机制:
    → gate控制delta更新量 → 保留多少 vs overwrite多少 → 平衡!
    → → → → → 类似Mamba的selective → 但更精细 → query-addressed!

  RTX 4090影响:
    → DeltaNet kernel: 需要额外subtract操作 → 比Mamba略慢!
    → → → 但overwrite能力 → 更好长上下文 → 精确修改 → 不堆积!
    → → → → → → RTX 4090: DeltaNet可行 → 但kernel优化不如Mamba成熟!
```

## 4. Hybrid Architecture — SSM + Attention 混合

```
Jamba (AI21 Labs, 2024→2025):
  → 核心设计: Mamba blocks + Attention blocks + MoE → 三种机制混合!
  → → 原因: SSM擅长local+sequential → Attention擅长global+parallel → 互补!
  → → → → → hybrid = local用SSM(O(N)) + global用attention(O(N²)但少) → 最优!

  架构比例:
    → Jamba-1.5: ~50% Mamba + ~25% Attention + MoE → 52B参数!
    → → 最佳比例: 每隔3-4个SSM层放1个attention → 好balance!
    → → → → → → 不是random → 研究表明: 底层用SSM(local) → 高层用attention(global)

  GPU硬件挑战:
    → SSM(scan sequential) + attention(matmul parallel) → 异构计算!
    → → → GPU优化: 一种模式(scan) → 另一种模式(matmul) → 切换开销!
    → → → → → Triton kernel fusion → 减少切换 → 但仍不如纯attention优化成熟!
    → → → → → → → SSD框架(Mamba-2) → 统一数学 → 更好GPU映射!

  内存优势:
    → Mamba层: O(D)固定state → 不增长 → 省KV!
    → → Attention层: O(N) KV → 但只25%层 → KV总量减少75%!
    → → → → → → Jamba KV ≈ 25% of pure Transformer → 4x并发提升!

  RTX 4090影响:
    → Hybrid推理: 75% SSM + 25% attention → KV减少75% → 并发4x!
    → → → → 但SSM kernel不够成熟 → 需要优化 → 比纯FlashInfer慢!
    → → → → → → RTX 4090: Hybrid可行 → 但kernel优化是瓶颈 → 需要Triton!
    → → → → → → → 128K上下文 → Hybrid优势明显 → 25% KV = 4x内存省!
    → → → → → → → → prefix-0501项目: Qwen3-27B = 16 attn + 48 DeltaNet → hybrid!
```

## 5. FlashAttention-3 — SM90异步Attention (RTX 4090不支持!)

```
FlashAttention-3 (Dao, 2024):
  → SM90(Hopper)专用 → 3个核心技术 → 1.5-2× over FA-2!
  → → 1. Producer-Consumer Async: GEMM和softmax异步重叠 → pipeline!
  → → → 2. Warpgroup GEMM: 128线程(4 warp) → WGMMA → 4×吞吐!
  → → → → 3. In-place Softmax: 低精度寄存器累积 → 减少register pressure!
  → → → → → → 75% tensor core利用率 → vs FA-2 50% → 1.5x!

  核心范式转变:
    → FA-2: 同步 → compute→softmax→compute → sequential → 等待!
    → → FA-3: 异步 → compute+softmax同时 → pipeline → overlap → 1.5-2×!
    → → → → → → SM89→SM90 = 同步→异步 → 根本范式变化!

  3-stage异步pipeline:
    → Stage 0: TMA加载下一块 → 异步 → 不占线程 → GPU→smem!
    → → Stage 1: WGMMA计算当前块 → 128线程 → tensor core!
    → → → Stage 2: Softmax处理上一块 → 在WGMMA等待窗口 → overlap!
    → → → → → → 3 stage → 每个都忙 → 不idle → 高吞吐!

  FP8 attention (2025方向):
    → softmax FP32累积 + WGMMA FP8输入 → 75% peak → 2× over FP16!
    → → → → → → DeepSeek MLA variant → SM90优化 → custom FlashAttention!

  RTX 4090关键:
    → SM89不支持FA-3 → 无TMA/WGMMA/Cluster → 只能用FA-2!
    → → → → → RTX 4090 = FlashInfer(SM89 SIMT路径) → 不是FA-3!
    → → → → → → FA-3 = Hopper/S100专属 → 未来硬件才可用!
    → → → → → → → 但! FA-3的pipeline思想 → 可用于SM89 cp.async 2-stage!
```

## 6. GPU硬件-架构协同设计 — 2025-2026前沿

```
硬件-架构co-design趋势:
  → 当前: Transformer→GPU → matmul密集 → GPU优化好 → 但长上下文内存差!
  → → Mamba→GPU → scan密集 → GPU不太优化 → 但内存好!
  → → → → Hybrid→GPU → matmul+scan → 异构 → 需要co-design!

  SM90→SM100→未来:
    → SM90(Hopper): TMA+WGMMA → FA-3 → 3-stage异步 → attention优化
    → → SM100(Blackwell): Enhanced WGMMA+FP4+HBM3e → 更强matmul → 但scan仍弱
    → → → → → → 需要: scan native support → 未来GPU可能加入scan加速单元!

  推测2026+:
    → 未来GPU可能: native SSM scan → scan+matmul共存 → hybrid天然高效!
    → → → → → 类似GPU从"纯matmul"到"matmul+scan" → 硬件适应新架构!
    → → → → → → → 这将是GPU架构的第二次范式变化(第一次=GPGPU→tensor core)!
```

## 7. Core Laws — 神经架构创新核心定律

```
1. Complexity-Quality Law: 模型质量 ∝ 复杂度 → 但O(N²)不是唯一选择!
   → → Transformer O(N²) → 高质量 → 但长上下文灾难!
   → → → SSM/RWKV O(N) → 可接受质量 → 但短上下文不如Transformer!
   → → → → Hybrid = O(N) + limited O(N²) → 最佳平衡!

2. Memory-Context Law: 内存 ∝ context长度 → KV=O(N) → 灾难!
   → → Transformer KV=O(N) → S↑4x → KV↑4x → 并发↓4x → 线性反比!
   → → → SSM/RWKV state=O(D) → 固定 → 不增长 → 内存恒定!
   → → → → → 128K上下文 → SSM state=4KB vs KV=6.4GB → 1600x省!

3. Selectivity Law: 模型能力 ∝ 选择性 → 选择性越强 → 质量越好!
   → → 固定SSM: 无选择性 → 类似固定attention → 低质量
   → → → Mamba: input-dependent A → 选择性 → 高质量 → 类似softmax!
   → → → → DeltaNet: delta rule → overwrite → 更细选择性 → 更高质量!

4. Hardware-Architecture Law: 架构效率 ∝ 硬件匹配度!
   → → Transformer: matmul密集 → GPU TC完美 → 最高效率!
   → → → SSM: scan密集 → GPU不太优化 → 需要co-design!
   → → → → Hybrid: matmul+scan → 异构 → 需要fusion → 中间效率!
   → → → → → → RTX 4090(SM89): Transformer最优 → SSM/Hybrid可行但非最优!

5. Dual-Mode Law: 训练=并行/推理=递推 → dual mode = 最practical!
   → → RWKV/Mamba/DeltaNet: training=parallel(GPU) / inference=RNN(O(1))
   → → → → → → dual mode = 训练快+推理省 → 生产最practical!
   → → → → → → → RTX 4090: dual mode推理 = 固定cost → 无KV增长 → 长上下文好!
```

## 关键论文与参考

```
- Mamba (Gu & Dao, 2023): 选择性SSM → input-dependent → O(N) → 长上下文革命!
- Mamba-2 SSD (Dao & Gu, 2024): SSM-Attention duality → 统一框架 → 2-8× faster!
- RWKV (Peng, 2023): WKV线性注意力 → dual mode → RNN推理 → 开源!
- RWKV-7 (Peng, 2025): 多参数递推 → most expressive → competitive with Transformer!
- DeltaNet (Shi et al., 2025): Delta Rule → overwrite → 精确memory → linear+expressive!
- Jamba (AI21, 2024): Hybrid SSM+Attention+MoE → 52B → 25% KV → 4x内存省!
- FlashAttention-3 (Dao, 2024): SM90 async → 3-stage → 75% TC → 1.5-2× → Hopper only!
- Hybrid scaling (2025): ~3-4 SSM : 1 attention → optimal ratio → CMU/Stanford research
- DeepSeek MLA variant (2025): SM90 custom FlashAttention → MLA optimization
- Vision Mamba (2024): SSM for vision → patch→scan → linear complexity → promising!
- Caduceus (2025): Mamba for DNA → long sequence → outperform Transformer → biology!

Sources:
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 SSD](https://arxiv.org/abs/2405.21075)
- [RWKV GitHub](https://github.com/BlinkDL/RWKV-LM)
- [FlashAttention-3](https://arxiv.org/abs/2407.04624)
- [Jamba](https://arxiv.org/abs/2403.19887)
- [DeltaNet](https://arxiv.org/abs/2406.06484)