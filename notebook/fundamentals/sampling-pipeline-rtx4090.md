# Sampling Pipeline RTX 4090 实测: 开销极小, decode瓶颈不在sampling

> 2026-06-07 | 8× RTX 4090 sampling pipeline micro-benchmark

## 核心数据

**7B模型 (vocab=32K) 各步骤耗时**:

| Step | B=1 | B=32 | B=128 | B=512 | 说明 |
|------|-----|------|-------|-------|------|
| FP16→FP32 | 0.009ms | 0.009ms | 0.011ms | 0.110ms | 精度转换 |
| Softmax | 0.011ms | 0.010ms | 0.015ms | 0.145ms | 32K vocab |
| Argmax (greedy) | 0.011ms | 0.009ms | 0.011ms | 0.023ms | 几乎免费 |
| Multinomial | **1.087ms** | 0.113ms | 0.127ms | 0.402ms | B=1时最贵! |
| Top-K=100 | 0.238ms | 0.230ms | 0.236ms | 0.474ms | 部分排序 |
| Top-P=0.9 | ~0.25ms | ~0.25ms | ~0.25ms | ~0.50ms | softmax+sort+cumsum |
| Full pipeline (top100) | 0.242ms | 0.233ms | 0.241ms | 0.492ms | fp32+topk+softmax+sample |
| Greedy pipeline | 0.019ms | 0.019ms | 0.020ms | 0.127ms | fp32+argmax |

## 关键发现

### 1. Sampling开销 vs Decode计算

7B模型 decode (memory-bound, B=1):
- **模型计算**: ~4ms (KV load + GEMM + attention)
- **Sampling**: 0.24ms (top100 pipeline) / 0.02ms (greedy)
- **Sampling占比**: 6% (top100) / 0.5% (greedy)

→ Sampling不是decode瓶颈! 模型计算主导。

### 2. Multinomial B=1异常

B=1 multinomial = **1.087ms**, 但 B=8 = 0.111ms → 10x下降!

**原因**: `torch.multinomial` 单token采样有额外的kernel launch和CPU同步开销。
多batch时GPU并行度增加, 每token开销降低。

**实际影响**: vLLM的Gumbel-Max trick (`probs/exp().argmax()` 或 `torch.exp(logits).argmax()`)
可以替代multinomial → 避免这个1ms开销。

### 3. 温度缩放几乎免费

| Temperature | 7B B=128 | 70B B=128 |
|-------------|----------|-----------|
| T=1.0 | 0.019ms | 0.019ms |
| T=0.5 | 0.019ms | 0.019ms |
| T=0.1 | 0.019ms | 0.019ms |
| T=0.01 | 0.019ms | 0.019ms |

温度缩放 = logits / T → 单次FP32除法, 开销与T值无关。

### 4. Top-K开销与K值关系

| K | 7B B=128 | 说明 |
|---|----------|------|
| 1 | 0.230ms | top-1 = argmax |
| 10 | 0.231ms | |
| 50 | 0.233ms | |
| 100 | 0.236ms | |
| 500 | 0.248ms | |
| 1000 | 0.266ms | |

Top-K开销几乎不随K增长! → `torch.topk` 在32K vocab上部分排序很高效。
原因: GPU并行排序32K元素, top-1000只是多取前1000个, 排序本身是瓶颈而非取出数量。

### 5. Batch scaling

**Full pipeline (top100)**:
- B≤256: ~0.23-0.24ms (几乎不变 → kernel launch主导)
- B=512: 0.49ms (翻倍 → 开始compute-bound)

**Greedy**:
- B≤256: ~0.02ms (几乎不变 → argmax极快)
- B=512: 0.13ms (6x增长 → softmax+argmax组合开始compute-bound)

## 与Decode Roofline对比

7B decode B=256 (from earlier roofline benchmark):
- **模型计算**: ~1.7ms/layer × 32 layers = ~54ms total (KV load + GEMM + attention)
- **Sampling**: 0.24ms
- **Sampling占比**: 0.24/54 = **0.4%**

→ 在memory-bound decode中, sampling几乎不贡献延迟。
decode优化应聚焦于: 1) KV cache读取带宽, 2) GEMM效率, 3) kernel fusion

## 与A16对比

| | RTX 4090 | A16 | 4090优势 |
|---|----------|-----|----------|
| Greedy B=1 | 0.02ms | ~0.05ms | 2.5x |
| Top100 B=128 | 0.24ms | ~0.5ms | 2x |
| Multinomial B=1 | 1.09ms | ~2ms | 1.8x |
| Softmax B=512 | 0.15ms | ~0.3ms | 2x |

RTX 4090在sampling上也更快, 但差距不大(2x而非compute的10x+) → sampling不是GPU性能差异的关键因素。

## 实用结论

1. **Greedy (temperature=0)** 是最快的sampling方式: 0.02ms vs 0.24ms → **12x faster**
2. **Top-K=100 pipeline** 是最常用的非greedy方式: 0.24ms, 开销可控
3. **Multinomial B=1** 有异常高延迟(1ms) → 应优先用Gumbel-Max替代
4. **Sampling不是decode瓶颈** → 优化重点应在模型计算和KV cache
5. **B≤256时sampling几乎flat** → kernel launch主导, batch增加不增加sampling时间