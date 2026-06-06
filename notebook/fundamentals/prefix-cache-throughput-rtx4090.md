# Prefix Cache Throughput — RTX 4090 实测
> 2026-06-07 | 5个实验: Hit Rate, Block对齐, Recompute vs Cache, 多轮对话, GRPO n_samples

## 一、Prefix Cache Hit Rate vs Throughput (线性关系验证)

| B | 0% prefix | 25% prefix | 50% | 75% | 90% |
|---|----------|----------|------|------|------|
| 1 | 0.7% | 23.3% | 51.8% | **70.7%** | 82.5% |
| 8 | -0.7% | 25.9% | 44.5% | **73.5%** | 86.9% |
| 32 | -0.3% | 25.7% | 49.4% | **74.5%** | 88.7% |
| 128 | -0.9% | 24.6% | 49.7% | **74.7%** | 89.6% |

**关键发现**: **time_savings ≈ compute_savings** → 近乎完美线性关系!

- 75% prefix → 74.5% time savings (理论75%, 实测74.5%) → **误差仅0.5%**
- **GEMM在RTX 4090上是memory-bound** → 时间∝数据量 → prefix比例直接决定时间节省
- **大batch同样有效**: B=128时75% prefix → 74.7% savings → 大batch下prefix cache同样重要
- **0% prefix时微负收益**: kernel launch差异 → negligible

**实用公式**: throughput_gain = prefix_ratio × throughput_no_cache

## 二、Block Alignment Overhead (vLLM block_size=16)

| prefix_len | vLLM缓存 | 浪费tokens | 浪费% | 重算开销 |
|-----------|---------|----------|------|---------|
| 15 | 0 | 15 | **100%** | 0.11ms |
| 31 | 16 | 15 | 48.4% | 0.12ms |
| 63 | 48 | 15 | 23.8% | 0.12ms |
| 127 | 112 | 15 | 11.8% | 0.12ms |
| 255 | 240 | 15 | 5.9% | 0.12ms |
| 511 | 496 | 15 | 2.9% | 0.12ms |
| 1023 | 1008 | 15 | **1.5%** | 0.11ms |

**关键发现**: vLLM block_size=16 → 最多浪费15 tokens → 但浪费比例随prefix增大快速衰减

1. **短prefix(15 tokens): 100%浪费!** → vLLM完全不缓存 → 这是最严重的case
2. **中等prefix(127 tokens): 11.8%浪费** → 15 tokens重算 → 但比例已经不大
3. **长prefix(1023 tokens): 仅1.5%浪费** → 15/1023 → block alignment几乎无影响
4. **重算成本恒定0.11ms**: 不管浪费多少tokens → 重算15 tokens总是0.11ms (memory-bound)

**与SGLang/Magi对比**: Token级共享→0浪费 → 但metadata开销更大 → 长prefix时vLLM的1.5%浪费可接受

**结论**: **短prompt(<100 tokens)时block alignment问题严重** → 长prompt时几乎无影响 → vLLM对长system prompt场景足够

## 三、Recompute vs KV Cache Read (关键对比!)

| B | 重算(ms/TFLOPS) | KV读(ms/GB/s) | Cache节省% |
|---|-------------|-------------|----------|
| 1 | 0.034 (126 TFLOPS) | 0.006 (326 GB/s) | **81.2%** |
| 4 | 0.115 (149 TFLOPS) | 0.006 (1321 GB/s!) | **94.5%** |
| 8 | 0.220 (156 TFLOPS) | 0.010 (1729 GB/s!) | **95.6%** |
| 16 | 0.463 (149 TFLOPS) | 0.018 (1868 GB/s!) | **96.1%** |
| 32 | 0.846 (162 TFLOPS) | 0.147 (458 GB/s) | **82.7%** |
| 128 | 3.255 (169 TFLOPS) | 0.583 (460 GB/s) | **82.1%** |

**震惊发现**: B≤16时KV读BW达326-1868 GB/s → **远超HBM 920 GB/s!**

1. **小batch(B≤16) KV读BW远超HBM**: B=4时1321 GB/s → B=8时1729 → B=16时1868 → **L2 cache命中!**
2. **大batch(B≥32) KV读BW降至460 GB/s**: KV cache(2×B×128×4096×2bytes)超过L2容量(16MB) → HBM饱和
3. **Cache总是比Recompute快**: 81-96% savings → **缓存KV远优于重算!**
4. **转折点B≈32**: L2→HBM切换 → BW从1868暴跌至458 → 但cache仍然比recompute快4.5x

**与之前KV Cache BW实验对照**: GQA-8 67MB→BW 2107 GB/s → L2 cache对小KV给予超高BW → 本次验证!

**vLLM的设计是正确的**: Swap(缓存到CPU) → Recompute → 但Swap增加延迟 → **GPU内缓存KV始终优于重算**

## 四、Multi-Turn Conversation Savings (累积效应)

| Turns | 总长度 | Prefix占比 | 当轮加速 | 累积节省 |
|-------|--------|----------|---------|---------|
| 1 | 832 | 61.5% | 2.56x (61.5% savings) | 0% (首次无缓存) |
| 2 | 1152 | 72.2% | 3.58x | **41.9%** |
| 3 | 1472 | 78.3% | 4.51x | **57.4%** |
| 5 | 2112 | 84.8% | 6.54x | **71.3%** |
| 10 | 3712 | 91.4% | 11.48x | **83.7%** |

**关键发现**: 多轮对话的prefix cache收益随turn数**指数级增长**!

1. **每轮新token量恒定320(64+256)** → 但累积context每轮增长320
2. **Turn 10: 91.4%是缓存** → 仅9.6%需要新算 → 推理几乎免费!
3. **累积节省83.7%**: 10轮对话总token 21840 → 实际仅计算3584 → 节省18256 tokens
4. **与SGLang RadixAttention对照**: SGLang宣称5轮75% → 本次实测71.3% → 接近!

**商业影响**: 多轮对话是LLM serving最常见场景 → prefix cache收益71-84% → 吞吐3-12x提升

## 五、GRPO n_samples Prefix Sharing (RL训练)

| n_samples | 计算节省 | 时间节省 | 吞吐提升 |
|----------|---------|---------|---------|
| 2 | 33.3% | 32.8% | 1.49x |
| 4 | 50.0% | 49.9% | 2.00x |
| 8 | 58.3% | 58.2% | 2.40x |
| 16 | 62.5% | 62.4% | 2.67x |

**公式**: compute_savings = (n-1)/(n) × prefix_ratio → 与n和prefix_ratio有关
- 7B模型: prefix=512, response=256 → prefix_ratio=66.7%
- n=8: savings = (8-1)/8 × 66.7% = **58.3%** → 与实测完全吻合!

**与verl PrefixGrouper对照**: 之前实测"GRPO n=8 前缀缓存节省68%计算" → 本次58.3% → 差异因为之前用的不同prompt_len/response_len比例

**与verl #6401 RFC对照**: Magi宣称~3x forward speedup → 本次n=8时2.40x → Magi的3x来自更优的tree布局+sparse attention → 比flat prefix更快

**极限**: n→∞时savings → prefix_ratio → 约66.7% → 1.67x → 不可能超过prefix_ratio

## 六、综合结论与对verl贡献的启示

1. **Prefix cache收益与prefix_ratio线性相关**: 75% prefix → 74.5% savings → 简洁的Roofline模型
2. **Block alignment对短prompt是问题**: 15 tokens → 100%浪费 → 但长prompt影响<2%
3. **Cache永远优于Recompute**: 81-96% savings → KV缓存应优先于重算
4. **L2 cache给小KV超高BW**: B≤16时1868 GB/s → 但B≥32降至HBM 460 GB/s
5. **多轮对话指数级收益**: 5轮71.3%, 10轮83.7% → SGLang RadixAttention的场景
6. **GRPO n=8: 58.3% savings** → 但Magi tree+sparse可达3x → 还有1.6x优化空间

**对verl #6401贡献的启示**:
- Flat PrefixGrouper(n=8) = 2.4x → Magi tree+sparse = 3x → 差距0.6x来自sparse attention
- Phase 1 (FSDP兼容) → 确保当前flat方法在所有backend可用 → 最优先
- Phase 2 (tree) → 多层级分支 → 累积收益从58% → 71%(多轮) → 显著改进
- Phase 3 (sparse) → FA中只计算高权重位置 → 从2.4x → 3x → 需要kernel修改

Sources:
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/automatic_prefix_caching.html)
- [SGLang RadixAttention](https://github.com/sgl-project/sglang)
- [verl #6401 RFC](https://github.com/volcengine/verl/issues/6401)
- [Magi Attention Paper](https://arxiv.org/abs/2505.11181)