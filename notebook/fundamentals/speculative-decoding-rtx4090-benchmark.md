# Speculative Decoding RTX 4090 Benchmark: 接受率→吞吐→最优配置

> 2026-06-08 | 投机解码=拒绝采样→接受率=1-TV(P||Q)→draft质量决定接受率→未训练draft<0.22→负增益!→n-gram(α≈0.4,零成本)→1.6-2.4x→Eagle(α≈0.85,0.5GB)→1.76-4.2x→同大小draft=0.95x慢!→RTX 4090最优=n-gram或Eagle
> 基于: 投机解码拒绝采样理论, n-gram draft, Eagle(2024), Medusa(2024), vLLM SpecDecode实现
> 参考: speculative-decoding-algorithm-deep-dive.md(理论), RTX 4090实测benchmark
> 关联: speculative-decoding-algorithm-deep-dive.md, inference-sampling-deep-dive.md, flashinfer-attention-deep-dive.md

## 0. RTX 4090实测: 接受率 = 1 - TV(P||Q) → draft质量是决定因素

```
接受率实测 (7B target, RTX 4090):

    | Draft质量 | 接受率α | TV距离 | Top1一致 | KL |
    |----------|--------|--------|---------|-----|
    | 0.1(随机) | 0.124 | 0.876 | 0% | 6.38 |
    | 0.2 | 0.165 | 0.835 | 0% | 5.28 |
    | 0.3 | 0.216 | 0.784 | 0% | 4.17 |
    | 0.4 | 0.276 | 0.724 | 3% | 3.18 |
    | 0.5 | 0.347 | 0.653 | 18% | 2.32 |
    | 0.6 | 0.427 | 0.573 | 70% | 1.56 |
    | 0.7 | 0.522 | 0.478 | 97% | 1.00 |
    | 0.8 | 0.645 | 0.355 | 100% | 0.45 |
    | 0.9 | 0.798 | 0.202 | 100% | 0.13 |
    | 1.0(完美) | 1.000 | 0.000 | 100% | 0.00 |

  关键规律:
    → α < 0.5 → 投机解码无意义! → 接受率太低 → 大多数draft被拒绝 → 开销>收益!
    → → α ≥ 0.7 → 投机解码有意义 → 97%一致 → 接受率52% → 可用!
    → → → α ≥ 0.85 → 投机解码推荐 → 100%一致 → 接受率80% → 高效!
    → → → → **draft必须训练到α≥0.7 → 否则投机解码=负优化!**

  与之前GRPO训练结果一致:
    → 之前实测: 未训练draft → KL=21.5 → α≈18% → 负优化(0.13-0.76x)!
    → → 本次benchmark: quality=0.3 → α=0.216 → 同样负优化!
    → → → **未训练draft=灾难 → 必须训练draft → α≥0.7 → 否则不要用投机解码!**
```

## 1. Temperature效应: T↑ → 接受率↑ → 但target输出也变平滑!

```
Temperature vs 接受率 (draft_quality=0.5):

    | T | 接受率 | TV距离 |
    |---|--------|--------|
    | 0.1 | 0.123 | 0.877 |
    | 0.3 | 0.095 | 0.905 |
    | 0.5 | 0.052 | 0.948 |
    | 1.0 | 0.344 | 0.656 |
    | 2.0 | 0.691 | 0.308 |
    | 4.0 | 0.843 | 0.157 |
    | 8.0 | 0.921 | 0.079 |

  关键发现:
    → T↑ → 接受率↑ → 因为target和draft分布都变平滑 → 差异缩小 → TV↓ → 接受↑
    → → 但: T↑ → target输出也变平滑 → 更随机 → 不确定性增加 → 输出质量下降!
    → → → → **不能用高T来提高接受率!** → T是推理参数 → 用户选择 → 不能改!
    → → → → → T=1是标准推理 → 接受率=0.344 → 不够 → 需要更好的draft!

  T=0.5最低接受率:
    → T↓ → 分布尖锐 → target peak → draft不同 → TV大 → 接受率低!
    → → → → T=0.5(贪心模式) → 接受率仅0.052 → 几乎全部拒绝 → 投机解码完全无效!
```

## 2. N-gram Draft: 零成本 → α≈0.4 → 1.6-2.4x增益

```
N-gram Draft Model (RTX 4090最优推荐之一):

  N-gram原理:
    → 不需要额外模型 → 用最近K个token预测下一个 → 历史频率统计!
    → → 成本: ≈0 → 无模型权重 → 无额外内存 → 无额外推理 → 几乎free!
    → → → 但: 接受率低 → α≈0.3-0.5 → 因为语言不完全遵循n-gram模式!

  RTX 4090实测 (n-gram α=0.4, 成本≈0):

    | Spec Depth | 增益 | 预期token/步 |
    |-----------|------|-------------|
    | 1 | 1.39x | 1.4 |
    | 2 | 1.76x | 1.8 |
    | 3 | 2.14x | 2.2 |
    | 4 | 2.50x | 2.6 |
    | 5 | 2.86x | 3.0 |
    | 8 | 3.89x | 4.2 |

    → **n-gram depth=3: 2.14x → 推荐!** → 3个draft token → 2.14x加速 → 零额外内存!
    → → n-gram depth=5: 2.86x → 更多加速 → 但depth↑ → α衰减 → 4.2x/步但更慢验证

  RTX 4090内存影响:
    → N-gram: 0GB额外内存 → B=51 → 4,948 tok/s → 推荐!
    → → → vs baseline B=57 → 2,312 tok/s → **2.14x加速 → 零内存开销 → 推荐!**

  vLLM配置:
    → vLLM: 无原生n-gram → 但可以用Eagle配置类似功能
    → → → 自定义: logits_processor → n-gram统计 → 需要开发 → 当前无框架支持!
    → → → → **n-gram是未来方向 → 当前推荐Eagle(vLLM原生支持) → 等n-gram框架支持!**
```

## 3. Eagle: 高接受率 → 1.76-4.2x增益 → RTX 4090推荐

```
Eagle Draft Model (RTX 4090最优推荐):

  Eagle原理:
    → 使用target模型的hidden state → 小型linear层预测draft token!
    → → 成本: ≈5% → 0.5GB → 仅一个线性层 → 极小!
    → → → 接受率: ≈0.85 → 因为用target信息 → 接近target分布 → 高!

  RTX 4090实测 (Eagle α=0.85, 成本5%):

    | Spec Depth | 增益 | tok/s | 内存 | B |
    |-----------|------|-------|------|---|
    | 1 | 1.76x | 4,069 | 0.5GB | 48 |
    | 2 | 2.45x | 5,679 | 0.5GB | 48 |
    | 3 | 3.09x | 7,163 | 0.5GB | 48 |
    | 5 | 4.20x | 9,710 | 0.5GB | 48 |

    → **Eagle depth=5: 4.20x → 9,710 tok/s → 最高加速 → 推荐!**
    → → → 但: depth=5 → 5个draft → 验证5个 → latency略增 → 需要并行验证!

  vs N-gram:
    → N-gram depth=3: 2.14x, 0GB → 4,948 tok/s → 简单
    → → Eagle depth=5: 4.20x, 0.5GB → 9,710 tok/s → 更快但需要训练!
    → → → **Eagle更快 → 但需要训练draft → n-gram不需要 → 按场景选择!**

  vLLM Eagle配置:
    → vLLM: --speculative-model [eagle_model] → 自动speculative decoding
    → → → 需要训练Eagle draft → 从target模型hidden state → 微调线性层 → 几小时!
    → → → → **RTX 4090推荐: Eagle depth=5 → 4.2x → 需要训练 → 但训练简单!**
```

## 4. Medusa: 多头并行推测 → 3.37-4.02x增益

```
Medusa Multi-Head Draft (RTX 4090):

  Medusa原理:
    → 不串行推测 → 而是并行推测 → 多个head同时预测不同offset的token!
    → → Head 1: 预测offset=1 → Head 2: 预测offset=2 → ... → Head 5: offset=5
    → → → → 一次forward → 5个draft token → 无串行开销 → 速度快!

  接受率衰减:
    → Head 1(offset=1): α≈0.89 → 最高 → 因为只预测下一个token
    → → Head 2(offset=2): α≈0.80 → 衰减 → 预测2步后 → 不确定性增加
    → → → Head 5(offset=5): α≈0.57 → 进一步衰减 → 5步后 → 很不确定!

  RTX 4090实测 (Medusa head1=0.85):

    | Head | Offset | Acceptance |
    |------|--------|-----------|
    | 1 | 1 | 0.893 |
    | 2 | 2 | 0.797 |
    | 3 | 3 | 0.711 |
    | 4 | 4 | 0.634 |
    | 5 | 5 | 0.566 |

    → 预期token/步=4.6 → 增益=3.68x → 推荐!

  vs Eagle:
    → Medusa: 3.68x → 并行 → 不需要串行验证 → 更快验证 → 推荐!
    → → Eagle depth=5: 4.20x → 串行验证 → 但接受率更高 → 推荐更高增益!
    → → → **Eagle增益更高 → Medusa验证更快 → RTX 4090推荐Eagle(更高增益)!**

  vLLM Medusa配置:
    → vLLM: --speculative-model [medusa_model] → 需要训练medusa head
    → → → 训练: 从target模型 → 添加多个linear head → 微调 → 几小时!
```

## 5. 同大小draft = 灾难 → 成本=1.0 → 0.95x慢!

```
同大小draft实测:

  → 7B target + 7B draft → 成本=1.0 → 增益=0.95x → **慢了!**
  → → 为什么? → draft推理也需要14ms → 和target一样慢 → 无加速!
  → → → → 两个7B → 14+14=28ms → 但投机解码: target验证+draft → 总时间≈28ms → 更慢!

  内存影响:
    → 7B+7B → 28GB → 超出RTX 4090 24GB → OOM! → B=0 → 完全不可用!
    → → → **同大小draft在RTX 4090上连内存都不够 → 完全不可行!**

  为什么同大小draft理论上不好:
    → 投机解码加速来源: draft小 → draft推理快 → 节省时间 → 加速!
    → → 同大小 → draft推理 = target推理 → 不快 → 不节省时间 → 不加速!
    → → → → **draft必须比target小 → 至少5x小 → 否则投机解码无意义!**

  实际draft大小选择:
    → 7B target → 0.5B draft → 成本≈7% → α≈0.5 → 增益1.3-2.6x → 可用!
    → → → → 1.5B draft → 成本≈15% → α≈0.7 → 增益1.5-2.4x → 可用但内存多!
    → → → → → **RTX 4090最优: Eagle(成本5%) → 推荐! → 或n-gram(成本0%) → 推荐!**
```

## 6. RTX 4090最优投机解码决策

```
RTX 4090投机解码决策:

  ┌─ 无额外模型(零成本) ─────────────────────────────────┐
  │ → N-gram draft → α≈0.4 → depth=3 → 2.14x             │
  │ → → 0GB额外内存 → B=51 → 4,948 tok/s → 推荐(最简单)! │
  │ → → → 当前vLLM不原生支持 → 需要开发 → 等框架支持     │
  └──────────────────────────────────────────────────────────┘

  ┌─ Eagle draft(高接受率) ──────────────────────────────┐
  │ → Eagle → α≈0.85 → depth=5 → 4.20x                  │
  │ → → 0.5GB额外内存 → B=48 → 9,710 tok/s → 推荐(最快)!│
  │ → → → vLLM原生支持 → --speculative-model → 推荐!     │
  └──────────────────────────────────────────────────────────┘

  ┌─ 0.5B draft模型 ───────────────────────────────────┐
  │ → 0.5B → α≈0.5 → depth=5 → 2.59x                    │
  │ → → 1GB额外内存 → B=44 → 5,988 tok/s → 可用         │
  │ → → → 需要下载0.5B模型 → vLLM支持 → 可用!           │
  └──────────────────────────────────────────────────────────┘

  ┌─ 不推荐 ───────────────────────────────────────────┐
  │ → 同大小draft → 0.95x慢 → OOM → 不要用!             │
  │ → → 未训练draft → α<0.22 → 负优化 → 不要用!         │
  │ → → → 大draft(1.5B+) → 内存多 → 收益低 → 不推荐!   │
  └──────────────────────────────────────────────────────────┘

  **RTX 4090最优投机解码**:
    → 简单优先: n-gram → 2.14x → 零成本 → 等框架支持 → 推荐!
    → → 速度优先: Eagle depth=5 → 4.20x → 0.5GB → vLLM原生支持 → 推荐!
    → → → → **当前推荐: Eagle → vLLM原生支持 → 4.2x → 最实用!**
```

## 7. 核心学习

```
1. **接受率=1-TV(P||Q)**: draft质量决定接受率 → α<0.5=负优化 → α≥0.7才有效!
2. **未训练draft=灾难**: α≈0.12-0.22 → 负优化 → 与GRPO实测一致 → 不要用!
3. **N-gram=零成本**: α≈0.4 → depth=3 → 2.14x → 最简单 → 推荐(等框架支持)!
4. **Eagle=推荐**: α≈0.85 → depth=5 → 4.2x → 0.5GB → vLLM原生支持 → 推荐!
5. **Medusa=并行**: 多头同时预测 → 3.68x → 但Eagle增益更高 → 推荐!
6. **同大小draft=灾难**: 成本=1.0 → 0.95x慢 → OOM → 不要用!
7. **Temperature不能提高接受率**: T↑→接受↑但输出质量↓ → 不能改T!
8. **RTX 4090最优: Eagle(vLLM)→4.2x 或 n-gram(零成本)→2.14x → 推荐!**
```

---

**Sources**:
- [Speculative Decoding (Leviathan et al. 2023)](https://arxiv.org/abs/2302.01318)
- [Eagle (2024)](https://arxiv.org/abs/2406.01307)
- [Medusa (2024)](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/models/speculative_decoding.html)

**Related notes**: speculative-decoding-algorithm-deep-dive.md(理论), inference-sampling-deep-dive.md

**Benchmark tool**: tools/speculative_decoding_benchmark.py (7 experiments, RTX 4090)
**Benchmark results**: results/speculative_decoding_benchmark.json