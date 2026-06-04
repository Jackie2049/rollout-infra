# Gradient AllReduce + Optimizer Step 实验

> 文件: `tools/gpu_grad_reduce_step.py`
> 日期: 2026-06-04
> GPU: A16 15GB

## 实验 1: AllReduce 开销 vs 模型大小

| Model | Bucket MB | ~AR ms | % of step |
|:---:|:---:|:---:|:---:|
| 10B | 20 | 0.44 | 0.4% |
| 70B | 140 | 2.99 | 2.9% |
| 175B | 350 | 7.46 | 6.9% |
| 405B | 810 | 17.3 | 14.7% |
| 1T | 2000 | 42.6 | 29.9% |

**结论**: AllReduce 占比随模型增大而增加。405B 时 ~15%，需要 bucket-level overlap 隐藏。

---

## 实验 2: FP16 vs BF16 Optimizer Step

| 方式 | 耗时 |
|------|:---:|
| FP16 (scaler + master copy) | 13.1ms |
| BF16 (no scaler) | 13.6ms |
| **Ratio** | **0.97x** |

BF16 optimizer step cost 与 FP16 几乎相同——收益不在速度，在**简化** (无需 GradScaler，不会 loss 溢出)。

---

## 实验 3: Gradient Accumulation 效率

| GA steps | Total ms | vs GA=1 |
|:---:|:---:|:---:|
| 1 | 36.7 | baseline |
| 4 | 25.1 | **31% 更快** |
| 8 | 23.2 | **37% 更快** |
| 16 | 22.2 | **39% 更快** |

**结论**: GA=16 比 GA=1 快 39%——通过减少 AllReduce 频率提升效率。但 GA 越大 → batch 越大 → 剩余显存越少。

---

## 实验 4: Adam 内存构成

| Component | MB | % |
|------|:---:|:---:|
| Params (FP16) | 33.6 | 12.5% |
| Gradients (FP16) | 33.6 | 12.5% |
| Adam m (FP32) | 67.1 | 25.0% |
| Adam v (FP32) | 67.1 | 25.0% |
| Master w (FP32) | 67.1 | 25.0% |
| **TOTAL** | **268.4** | **100%** |

**Adam 优化器 = 75% 训练内存！** ZeRO-1 (DP=8) 将此部分 ÷8 → 总内存节省 58%。

---

## 关键洞察

1. **DP AllReduce = 瓶颈**: 405B 时占 14.7%，必须 bucket-overlap 隐藏
2. **BF16 > FP16**: optimizer step 速度相同但更安全 (无溢出)
3. **Gradient Accumulation = 免费加速**: GA=16 → 39% 更快 (减少同步)
4. **Adam = 内存杀手**: 75% 是优化器，ZeRO-1 是必须的
5. **176B+ 模型必须 ZeRO-2/3**: ZeRO-1 不够，需要先分片 grads + params
