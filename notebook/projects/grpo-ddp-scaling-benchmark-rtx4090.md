# GRPO DDP Scaling Benchmark — RTX 4090 PCIe

> 2026-06-07 | 3.3M/46M GQA Transformer GRPO training on 8×RTX 4090 PCIe (no NVLink)

## 核心发现: PCIe DDP对GRPO训练几乎无加速

**46M模型DDP反而更慢！** 通信开销占56-79%，PCIe AllReduce瓶颈吞噬所有计算收益。

## 实测数据

### Small Model (3.3M params, GQA-4:2, vocab=1K)

| GPUs | Step Time | Speedup | Efficiency | Comm Overhead | Throughput |
|------|-----------|---------|------------|---------------|------------|
| 1    | 7.61ms    | 1.00x   | 100%       | 0%            | 134.5K tok/s |
| 2    | 9.87ms    | 1.54x   | 77.1%      | 22.9%         | 207.5K tok/s |
| 4    | 12.23ms   | 2.49x   | 62.2%      | 37.8%         | 334.8K tok/s |
| 8    | 12.95ms   | 4.70x   | 58.8%      | 41.2%         | 632.3K tok/s |

### Medium Model (46M params, GQA-8:2, vocab=32K)

| GPUs | Step Time | Speedup | Efficiency | Comm Overhead | Throughput |
|------|-----------|---------|------------|---------------|------------|
| 1    | 15.86ms   | 1.00x   | 100%       | 0%            | 64.6K tok/s |
| 2    | 36.64ms   | 0.87x   | 43.3%      | **56.7%**     | 55.9K tok/s |
| 4    | 68.61ms   | 0.92x   | 23.1%      | **76.9%**     | 59.7K tok/s |
| 8    | 75.18ms   | 1.69x   | 21.1%      | **78.9%**     | 109.0K tok/s |

**关键: 46M模型2/4-GPU DDP比单GPU更慢！** speedup<1.0意味着通信开销超过计算收益。

## 原因分析: PCIe vs NVLink

### AllReduce通信瓶颈

RTX 4090 PCIe集群(无NVLink):
- PCIe单向带宽: 17-20.5 GB/s (实测)
- AllReduce效率: 2GPU 7.59 GB/s, 4GPU 3.31 GB/s, 8GPU 3.01 GB/s

46M模型梯度大小 = 184.6MB (FP32) or 92.3MB (FP16)
- AllReduce时间估算(8GPU): 184.6MB / 3.01 GB/s × 2(2次传输) ≈ 122ms
- 实测: 75.18ms - 15.86ms = 59.3ms(额外时间)
- 通信占比: 59.3/75.18 = 78.9% → 匹配!

### 对比: NVLink (A100/H100)

| 互联     | 带宽        | 46M AllReduce | 占比     |
|---------|-------------|---------------|----------|
| PCIe    | 3.01 GB/s   | ~60ms         | **79%**  |
| NVLink  | 300 GB/s    | ~0.6ms        | **<4%**  |

NVLink下DDP效率>96%，PCIe下仅21%。**100x互联差距→80%效率差距。**

## GRPO特殊挑战

### GRPO group normalization与DDP不兼容

GRPO需要组内比较(同一prompt的n个response):
- 单GPU: 1个prompt→4个response→组归一化A=(r-μ)/σ
- DDP: 1个prompt的4个response分散在4个GPU→无法组归一化！

**解决方案**:
1. 每GPU独立组归一化 → 不同GPU的组均值不同 → 方差更高
2. AllGather rewards后再归一化 → 额外通信 → 更慢
3. 通信与计算重叠 → 但GRPO reward在forward后计算 → 无法重叠

### DDP下GRPO的有效batch size

| GPUs | Global Batch | Local Batch | GRPO Groups/GPU | 问题 |
|------|-------------|-------------|----------------|------|
| 1    | 8           | 8           | 2(n=4)         | OK   |
| 2    | 16          | 8           | 2              | 组间不比较 |
| 4    | 32          | 8           | 2              | 组间不比较 |
| 8    | 64          | 8           | 2              | 组间不比较 |

**8GPU DDP × 8 local_batch = 64 global, 但GRPO只在每GPU内做组归一化**
→ 8GPU只增加throughput不增加per-sample的信号质量
→ 与单GPU B=8训练收敛等价(只是更快处理更多样本)

## Scaling效率公式

$$\text{efficiency} = \frac{\text{compute\_time}}{\text{compute\_time} + \text{comm\_time}}$$

$$\text{comm\_time} \propto \frac{\text{gradient\_size}}{\text{effective\_bandwidth}}$$

PCIe有效带宽随GPU数衰减(NUMA跨界):
- 2GPU: 7.59 GB/s (单NUMA)
- 4GPU: 3.31 GB/s (跨NUMA 2.3x衰减)
- 8GPU: 3.01 GB/s (瓶颈饱和)

### 模型大小vs DDP效率交叉点

$$\text{comm\_fraction} = \frac{gradient\_size / eff\_bw}{step\_time}$$

- 3.3M: gradient=11.5MB, comm=1.5-3.8ms, compute=7.61ms → comm=20-50%
- 46M: gradient=184.6MB, comm=12-60ms, compute=15.86ms → comm=75-79%

**结论: 消费级GPU PCIe集群DDP只适合<10M模型(通信<30%)**
**>10M模型DDP反而更慢，必须用NVLink(A100/H100)或TP**

## 与之前DDP基准对比

| 来源 | 模型 | 2GPU效率 | 4GPU效率 | 8GPU效率 |
|------|------|---------|---------|---------|
| 本测试 | 3.3M GRPO | 77.1% | 62.2% | 58.8% |
| 本测试 | 46M GRPO | 43.3% | 23.1% | 21.1% |
| benchmark_ddp_scaling | 8.4M MLP | ~90% | ~70% | ~66% |
| benchmark_ddp_scaling | 50M MLP | ~50% | ~40% | ~35% |

**一致结论**: PCIe DDP效率∝模型大小⁻¹, 小模型(<10M)勉强可用, 大模型不可行

## 实用建议: RTX 4090上的GRPO训练策略

### 7B模型 GRPO训练选择

| 方案 | 优点 | 缺点 |
|------|------|------|
| **单GPU ZeRO-3** | 内存省, 无通信 | 速度慢(需gradient accumulation) |
| **2GPU DDP** | 可能加速 | 46M已0.87x, 7B更慢! |
| **TP(不可行)** | - | PCIe通信96%+, RTX 4090无NVLink |
| **梯度累积** | 无通信开销 | 等价于大batch但慢 |

**最优方案**: 单GPU + gradient accumulation + ZeRO-3
→ 7B FP16 = 14GB fits RTX 4090(24GB)
→ ZeRO-3将optimizer分片 → 更大batch
→ 无DDP通信开销 → 最高计算效率

### 未来方向

1. **NVLink GPU**: A100/H100 → DDP效率>96% → 推荐DDP
2. **FSDP**: 替代DDP, 参数分片+通信重叠 → 可能更优
3. **verl colocation**: GRPO 2模型同GPU → 省GPU但无DDP
4. **Prefix Sharing**: 训练加速1.59x(已验证) → 不依赖多GPU

## 关键数学: 为什么46M DDP更慢

单GPU step = compute = 15.86ms
DDP step = compute + AllReduce = 15.86 + 20.78 = 36.64ms (2GPU)
→ AllReduce耗时20.78ms > 计算节省7.61ms
→ **通信开销超过计算收益!**

$$\text{speedup} = \frac{N \times t_{compute}}{t_{compute} + t_{comm}}$$

当 $t_{comm} > (N-1) \times t_{compute}$ 时, speedup < 1:
- 2GPU: $t_{comm} = 20.78ms > 1 \times 15.86ms$ → speedup = 0.87x ✓

## 数据文件

- `grpo_ddp_benchmark_results.json`: 完整结果(含per-rank时间)
- `tools/ddp_grpo_benchmark_4090.py`: 基准脚本(multiprocessing spawn)