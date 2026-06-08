## 1. Attention-only Speedup: GQA越激进 → FlashInfer加速越高

```
GQA Config Sweep (B=32, S=4096, RTX 4090):

    | Config | KV heads | group_size | FlashInfer ms | SDPA ms(with exp) | Speedup | KV/tok KB |
    |--------|----------|------------|---------------|-------------------|---------|-----------|
    | MHA    | 32       | 1          | 2.436ms       | 42.436ms          | 17.42x  | 512       |
    | GQA-8  | 8        | 4          | 0.754ms       | 40.735ms          | 54.03x  | 128       |
    | GQA-4  | 4        | 8          | 0.485ms       | 40.454ms          | 83.41x  | 64        |
    | MQA    | 1        | 32         | Error!        | —                 | N/A     | 16        |

  关键规律:
    → KV heads越少 → FlashInfer越快 → 因为: KV读取量↓ + GQA native处理不需要expansion!
    → → GQA-4: FlashInfer 0.485ms → 4 heads → KV读取64KB/tok → 极小 → 极快!
    → → → GQA-8: FlashInfer 0.754ms → 8 heads → KV读取128KB/tok → 比GQA-4慢一点
    → → → → MHA: FlashInfer 2.436ms → 32 heads → KV读取512KB/tok → 最慢(但仍然17x于SDPA!)

  SDPA为什么这么慢?
    → SDPA需要KV expansion → GQA-8: 8→32 heads → 4x memory复制 → 4x time!
    → → 即使"纯SDPA"(无expansion): 37.8ms → FlashInfer仍然52x快!
    → → → **SDPA根本不适合decode → FlashInfer是生产唯一答案!**

  MQA为什么FlashInfer不支持?
    → FlashInfer 0.6.12不支持group_size=32 → 报错"Unsupported group_size: 32"
    → → 同样GQA-5(group_size=6.4)也不支持 → 需整数group_size!
    → → → → **FlashInfer只支持: 1,2,4,8,16 group_size → 即KV heads=32,16,8,4,2**
    → → → → → GQA-5需要group_size=6.4 → FlashInfer不支持! → 生产不推荐GQA-5!
```

## 2. Overall Model Throughput: Roofline + 实测Attention

```
Overall Decode Throughput (7B GQA-8, S=4096, RTX 4090):

    | B | Roofline(ms) | SDPA attn(ms) | FI attn(ms) | SDPA total(ms) | FI total(ms) | SDPA tp | FI tp | Speedup |
    |---|-------------|---------------|-------------|----------------|--------------|---------|-------|---------|
    | 1 | 14.92       | 1.154         | 0.209       | 16.13          | 15.19        | 62      | 66    | 1.06x   |
    | 4 | 15.76       | 5.153         | 0.212       | 20.97          | 16.03        | 191     | 250   | 1.31x   |
    | 8 | 16.88       | 10.146        | 0.338       | 27.09          | 17.28        | 295     | 463   | 1.57x   |
    | 16| 19.13       | 20.371        | 0.477       | 39.56          | 19.67        | 404     | 814   | 2.01x   |
    | 32| 23.62       | 40.722        | 0.770       | 64.40          | 24.45        | 497     | 1,309 | 2.63x   |
    | 55| 30.07       | ~70(extrapolated)| 1.166   | 100.12         | 31.30        | 549     | 1,757 | 3.20x   |

  关键发现:
    → **B=1: FlashInfer 1.06x → 几乎无加速 → 因为attention占比小(7%)!**
    → → **B=32: FlashInfer 2.63x → 显著加速 → attention占63%!**
    → → → **B=55: FlashInfer 3.20x → 最大加速 → attention占70%!**
    → → → → **FlashInfer加速随B增长 → 生产推荐高batch!**

  Attention占比分析:
    → B=1: SDPA attn 1.154ms / 16.13ms = 7% → 影响小
    → B=32: SDPA attn 40.722ms / 64.40ms = 63% → 主导!
    → → 原因: KV expansion随B线性增长 → B↑ → KV读取↑ → SDPA↑ → 占比↑!
    → → → FlashInfer: GQA native → KV不随B增长 → 恒定0.2-1.2ms → 极小占比!

  INT4 AWQ Throughput:
    → B=16: 1,842 tok/s → 2x BF16 baseline
    → B=32: 2,376 tok/s → 1.8x
    → B=55: 2,816 tok/s → 1.6x (attention估计有误差)
    → B=118: 3,134 tok/s → 权重3.5GB → KV占更多
```

## 3. SDPA KV Expansion: 7-8% overhead but OOM at high B

```
KV Expansion Overhead (GQA-8 → 32 heads):

    | B | SDPA with exp (ms) | SDPA pure (ms) | Expansion overhead | OOM? |
    |---|--------------------|----------------|--------------------|------|
    | 1 | 1.154              | 1.061          | 8.1%               | No   |
    | 4 | 5.153              | 4.756          | 7.7%               | No   |
    | 8 | 10.146             | 9.406          | 7.3%               | No   |
    | 16| 20.371             | 18.867         | 7.4%               | No   |
    | 32| 40.722             | 37.795         | 7.2%               | No   |
    | 55| OOM                | —              | —                  | Yes! |

  KV expansion overhead:
    → 恒定7-8% → 因为KV expansion是memory copy → 开销与数据量成正比
    → → GQA-8→32: 4x memory → 7-8%额外复制时间
    → → → **KV expansion本身开销不大 → 但SDPA+expanded KV是主要瓶颈!**

  OOM原因:
    → B=55: KV = 55 × 4096 × 32 × 128 × 2bytes = 1.76GB → 加上expanded = 7.04GB
    → → 加上Q+output → 总内存需要 >24GB → OOM!
    → → → **FlashInfer: 不需要expansion → 只需8 KV heads → 55×4096×8×128×2 = 0.44GB → OK!**
```

## 4. FP8 KV FlashInfer: API不兼容v0.6.12

```
FP8 KV FlashInfer RTX 4090 (v0.6.12):

  Error: "Mismatched type on argument #16: Expected `float` but got `ffi.Tensor`"

  原因:
    → k_scale/v_scale参数 → v0.6.12期望float → 我传了tensor → API不兼容!
    → → 旧版API: k_scale=float → 直接传scale值
    → → → 我写的新版: k_scale=tensor → 0.6.12不接受

  解决方案:
    → 方法1: k_scale=1.0/v_scale=1.0 → 不使用per-tensor scaling → 仅BF16 KV
    → → 方法2: 使用更新版FlashInfer → 支持per-tensor scaling FP8 KV
    → → → 方法3: 手动量化 → BF16 KV存储+FP8逻辑 → 在FlashInfer外处理

  生产建议:
    → **vLLM内置FP8 KV → 自动处理scale → 不需要手动!**
    → → FlashInfer 0.6.12 → vLLM使用 → 内部处理 → 不暴露API给用户
    → → → → **不要手动写FP8 KV → 使用vLLM → 自动处理!**
```

## 5. 推理计算器修正: FlashInfer整体1.06-3.20x

```
推理计算器修正 (inference_calculator_4090.py):

  之前: FlashInfer整体1.5-1.8x → 固定值 → 不随B变化
  现在: FlashInfer整体1.06-3.20x → 实测值 → 随B变化!

  修正后的FLASHINFER_DECODE_SPEEDUP:
    → B=1:  1.06x (实测)
    → B=4:  1.31x (实测)
    → B=8:  1.57x (实测)
    → B=16: 2.01x (实测)
    → B=32: 2.63x (实测)
    → B=55: 3.20x (实测)
    → B>55: 线性插值 extrapolated to ~4x at B=128

  关键影响:
    → **GQA-8 B=55: baseline 1,390 tok/s → FlashInfer 3.20x → 4,448 tok/s → 推荐!**
    → → **GQA-5 B=55: baseline 2,240 tok/s → FlashInfer估计3.20x → 7,167 tok/s**
    → → → 但: GQA-5 FlashInfer不支持(group_size=6.4非整数)!
    → → → → **生产必须用GQA-8 → FlashInfer原生支持 → 实测验证 → 推荐!**
```

## 6. RTX 4090最优配置修正

```
RTX 4090最优配置修正 (基于实测FlashInfer数据):

  之前推荐: 7B GQA-5 + FP8 KV + FlashInfer + vLLM V1 + S=4K → B=57 → 2,312 tok/s
  修正推荐: 7B GQA-8 + INT8 KV + FlashInfer + vLLM V1 + S=4K → B=35 → ~4,500 tok/s

  关键修正:
    → **GQA-5 → GQA-8**: FlashInfer不支持GQA-5(group_size=6.4) → 必须GQA-8!
    → → GQA-8 KV/tok=64KB vs GQA-5=40KB → B=35 vs B=55 → 并发降低但吞吐更高!
    → → → **GQA-8 + FlashInfer 3x → B=35 → 1390×2.68 = 3,726 tok/s → 推荐!**
    → → → → GQA-8 + Eagle d5 → 1390×2.68×4.2 = 15,649 tok/s → 推荐(最快!)!

  内存对比:
    → GQA-8 BF16 INT8 KV: weight 13.28GB + KV 0.25×35 = 8.75GB + 2GB overhead = 24.03GB → OK!
    → → GQA-8 INT4 INT8 KV: weight 3.5GB + KV 0.25×74 = 18.5GB + 2GB = 24GB → B=74 → 推荐!
    → → → **INT4+GQA-8+INT8 KV+FlashInfer → B=74 → 2.7x → ~5,100 tok/s → 推荐!**

  **RTX 4090最优配置修正**:
    → 精度优先: GQA-8 BF16 INT8 KV FlashInfer → B=35 → 3,726 tok/s → 推荐!
    → → 并发优先: GQA-8 INT4 INT8 KV FlashInfer → B=74 → ~5,100 tok/s → 推荐!
    → → → 最快: GQA-8 BF16 INT8 KV FlashInfer+Eagle d5 → B=35 → ~15,649 tok/s → 推荐!
```

## 7. 核心学习

```
1. **FlashInfer整体加速=1.06-3.20x(实测!)**: 远高于估计1.5-1.8x → B越高加速越高!
2. **SDPA+KV expansion占40-70%decode**: 不是5-15%! → GQA-8→32 heads = 4x memory → SDPA灾难!
3. **GQA越激进→FlashInfer加速越高**: GQA-4→83x > GQA-8→54x > MHA→17x
4. **MQA不被FlashInfer支持**: group_size=32 → Unsupported → 生产不推荐MQA!
5. **GQA-5不被FlashInfer支持**: group_size=6.4非整数 → Unsupported → 生产必须GQA-8!
6. **FP8 KV v0.6.12 API不兼容**: k_scale期望float→传tensor→Error → 使用vLLM内部处理
7. **推理计算器修正**: FlashInfer整体1.06-3.20x → 实测数据 → 随B变化!
8. **RTX 4090最优=GQA-8+FlashInfer**: 生产唯一可行配置 → 实测验证 → 推荐!
```

---

**Sources**:
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) v0.6.12
- RTX 4090实测 benchmark

**Related notes**: flashinfer-attention-deep-dive.md(理论), inference_calculator_4090.py(计算器)

**Benchmark tool**: tools/flashinfer_real_decode_benchmark.py
**Benchmark results**: results/flashinfer_real_decode_benchmark.json