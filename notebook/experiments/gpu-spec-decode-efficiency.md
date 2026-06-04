# Speculative Decoding 效率实测

> 文件: `tools/gpu_spec_decode_efficiency.py`
> 日期: 2026-06-04
> GPU: A16 15GB

## 实验 1: Temperature vs 接受率

| Correlation (draft vs target) | T=0.3 | T=0.5 | T=0.7 | T=1.0 | T=2.0 |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Identical** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| High (0.01 noise) | 0.977 | 0.977 | 0.978 | 0.977 | 0.976 |
| Medium (0.1 noise) | 0.782 | 0.788 | 0.785 | 0.786 | 0.790 |
| Low (0.5 noise) | 0.267 | 0.254 | 0.263 | 0.261 | 0.270 |
| Random (1.0 noise) | 0.064 | 0.059 | 0.062 | 0.064 | 0.056 |

**结论**: Temperature 对 argmax-based 接受率影响不大，因为 argmax 不受 scaling 影响。真正的接受率取决于 draft/target probability ratio，而非 temperature。

---

## 实验 2: 最优 Draft Length (K)

Draft cost = 0.15x target:

| Acceptance p | K* | E[tokens] | Speedup |
|:---:|:---:|:---:|:---:|
| 0.5 | 2 | 1.75 | 1.35x |
| 0.6 | 2 | 1.96 | 1.51x |
| **0.7** | **3** | **2.53** | **1.75x** |
| 0.8 | 5 | 3.69 | 2.11x |
| 0.9 | 8 | 6.13 | 2.78x |

**公式**: `E[tokens] = (1-p^(K+1))/(1-p)`
**结论**: 接受率 0.7 时 K*=3-4 最优，与论文一致。

---

## 实验 3: Draft model 成本约束

Target forward = 4.2ms, Draft forward = 4.4ms (ratio = 1.06x):

| Draft/Target | K=2 | K=3 | K=5 | K=8 |
|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 2.15x | 2.46x | 2.80x | 2.96x |
| 0.10 | 1.82x | 1.95x | 1.96x | 1.78x |
| **0.15** | **1.68x** | **1.75x** | **1.68x** | 1.45x |
| 0.25 | 1.46x | 1.45x | 1.31x | 1.07x |
| 0.50 | 1.09x | 1.01x | 0.84x | **0.64x** |

**关键**: draft cost > 0.5x 时大 K 反而减速！这就是为何：
- **N-gram**: 零开销，即使低 acc 也有 net gain
- **Eagle**: 小模型 (0.1x cost)，高 acc → 最大加速
- **完整 model 作为 draft**: 无 net gain（ratio=1.06x）

---

## 关键洞察

1. **argmax 接受率对 T 不敏感**: 真正的 acceptance 用 probability ratio，temperature 影响更大
2. **K 越大不一定越好**: p=0.7 时 K>5 收益递减
3. **Draft cost 是最关键约束**: cost>0.5x target 时 spec decode 有害
4. **N-gram 因为零开销而有效**: 即使接受率低，没有 draft cost 所以 net speedup 总为正
5. **A16 限制**: draft 和 target linear layer 成本接近 (ratio=1.06x)，需要更小的 draft model
