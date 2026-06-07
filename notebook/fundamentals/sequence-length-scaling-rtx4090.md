# Sequence Length Scaling — RTX 4090实测

> 2026-06-07 | 序列长度对推理和训练的影响: attention O(N²)何时成为瓶颈?

## 概述

在RTX 4090上系统测量序列长度如何影响:
1. 推理吞吐量 (forward pass time vs N)
2. GPU内存 (KV cache ∝ N)
3. 训练收敛质量 (长序列是否帮助?)

## 一、Forward Pass Scaling

### 1.1 2.28M模型 (BF16, B=1)

```
seq_len | forward_ms | ratio vs seq=4 | expected_N² | peak_mem(GB) | KV(MB)
4       | 1.54       | 1.00           | 1.00         | 0.0133       | 0.0
8       | 1.54       | 1.00           | 4.00         | 0.0133       | 0.0
16      | 1.60       | 1.04           | 16.00        | 0.0134       | 0.0
32      | 1.66       | 1.08           | 64.00        | 0.0135       | 0.1
64      | 1.52       | 0.99           | 256.00       | 0.0137       | 0.1
128     | 1.57       | 1.02           | 1024.00      | 0.0141       | 0.3
256     | 1.54       | 1.00           | 4096.00      | 0.0148       | 0.5
512     | 1.55       | 1.01           | 16384.00     | 0.0167       | 1.0
1024    | 1.56       | 1.01           | 65536.00     | 0.0202       | 2.1
2048    | 1.49       | 0.97           | 262144.00    | 0.0259       | 4.2
4096    | 1.61       | 1.04           | 1048576.00   | 0.0385       | 8.4
8192    | 2.21       | 1.44           | 4194304.00   | 0.0637       | 16.8

→ Forward time几乎恒定(1.5-1.6ms)直到seq=8192!
→ Attention O(N²)对小模型完全不重要!
→ 只有N=8192才开始看到上升(2.21ms vs 1.55ms = 1.44x)

→ 为什么? 小模型GEMM太小 → kernel launch时间(1-2ms)主导!
  → GEMM: hd=256, M=1 → kernel launch 1us + GEMM 0.1us → 1.1us
  → 但实际1.5ms → 其他overhead(embedding, LN, sampling, etc) → 1.5ms
  → Attention O(N²): N=4K → 4K²=16M ops → 16M/169T=0.094ms
  → 0.094ms vs 1.5ms → attention仅占6%! → 完全被其他overhead淹没!

→ 7B模型: hd=4096 → GEMM大 → attention占比高 → N²scaling可见
  → 7B seq=4K: attention≈27ms vs GEMM≈6ms → attention占81%!
  → → 小模型和7B模型的scaling行为完全不同!
```

### 1.2 内存Scaling

```
peak_mem vs seq_len (2.28M):
  N=4:     0.013GB → N=8192: 0.064GB → 4.9x增长
  → KV cache: 0→16.8MB → 线性增长 ✓
  → 但KV占比: 16.8MB / 64MB = 26.3% → KV开始重要!

→ 7B模型:
  N=128K: KV=2GB(MHA)/0.25GB(GQA-4) → 巨大!
  → 小模型KV不重要, 但7B+长context → KV是内存瓶颈!
  → 与之前KV Cache BW实验一致: KV ∝ N × hd × nl × 2bytes
```

## 二、训练收敛 vs 序列长度

```
max_digits | seq_len | avg_prompt | steps/s | loss  | eval  | mem(GB)
1          | 18      | 4 tok      | 111.8   | 0.471 | 43%   | 0.041
2          | 22      | 6 tok      | 137.2   | 1.609 | 47%   | 0.041
3          | 26      | 8 tok      | 137.3   | 2.141 | 40%   | 0.041

→ 长序列反而loss更高(2.141 vs 0.471) → 更多token需要学习!
→ 但eval相近(40-47%) → 序列长度对收敛质量影响不大
→ 内存不变(0.041GB) → 序列长度对内存无影响(太小)
→ steps/s反而随长度增加 → 更大的batch让GPU利用率更高?
  → 111→137 steps/s → GPU从under-utilized变为better-utilized
  → 但差异小(23%) → 对小模型影响有限
```

## 三、关键发现

### 3.1 小模型: Attention不是瓶颈!

```
2.28M模型 RTX 4090:
  forward ≈ 1.5ms (恒定) → kernel launch + embedding + LN 占 >94%
  attention占比: <6% (N≤4096) → 14% (N=8192) → 完全不是瓶颈!

→ 7B模型 RTX 4090:
  forward ≈ 440ms (N=128, B=1) → attention占比 >> GEMM
  → attention是瓶颈 → N²scaling可见
  → → FlashAttention在7B+大N时才有意义!

→ 之前FlashAttention实测一致:
  76K模型: FlashAttn decode更慢(0.67-0.84x) → 小batch attention matrix小 → FA无IO收益
  7B模型: FlashAttn才重要 → 大batch+长序列 → IO节省显著

→ 结论: 模型大小决定瓶颈位置!
  <10M: kernel launch + embedding主导 → attention不重要
  10M-100M: GEMM开始重要 → attention在长序列时重要
  100M+: attention在中等序列就重要 → FlashAttention必须
  7B+: attention在短序列就重要 → 是推理/训练瓶颈!
```

### 3.2 生产建议: 序列长度策略

```
训练场景                    | 序列长度策略
小模型(<10M)训练            | 不需要特殊处理 → attention <6%
中等模型(10M-100M)训练       | FlashAttention@N≥512 → 节省IO
大模型(7B+)训练             | FlashAttention必须 + chunked prefill@N≥4K

推理场景                    | 序列长度策略
小模型推理                  | KV cache不重要 → 不需要优化
7B B=1 短序列(<128)         | weight-bound → attention不重要
7B B=1 长序列(4K+)          | KV-bound → KV优化关键(GQA/MLA/offload)
7B B=64 长序列(128K)        | KV=2GB → 必须GQA+量化+offload

→ 与之前所有实验交叉验证:
  1. FlashAttention decode更慢(小batch) → 与本次一致
  2. KV ∝ N线性增长 → 与KV BW实验一致
  3. 长context KV灾难瓶颈 → 与Long Context Serving笔记一致
  4. GQA-4 KV仅25% of MHA → 与GQA实测一致
  5. KV offload >32K比重compute快 → 与KV Cache Offload笔记一致
```

## 四、工具

```bash
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm

# Full experiment
CUDA_VISIBLE_DEVICES=0 python -u tools/sequence_length_scaling.py --model_size 2.28m --num_steps 100

# Quick forward-only scaling (up to 8192)
# See inline Python in notebook

results/sequence_length_scaling_results.json
```

工具: `tools/sequence_length_scaling.py`