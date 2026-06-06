# 8xRTX 4090 PCIe 互连实测: 消费级GPU多卡通信瓶颈
> 2026-06-07 | 8× RTX 4090 PCIe 集群实测

## 核心数据

| 指标 | 值 | 对比A100 NVLink |
|------|---|----------------|
| P2P access | **全部禁用** (56/56) | NVLink: 全部启用 |
| PCIe unidir BW | **17-20.5 GB/s** | NVLink: 300 GB/s (单向) |
| HBM BW | **877.8 GB/s** | A100: 2039 GB/s |
| PCIe/HBM ratio | **2.1%** | NVLink/HBM ≈ 15% |
| GPU间慢于本地 | **48x** | NVLink: 约1.5x |

## PCIe 拓扑分析 (nvidia-smi topo)

```
2个NUMA节点:
  NUMA 0 (CPU 0-27): GPU 0,1,2,3
  NUMA 1 (CPU 28-55): GPU 4,5,6,7

连接类型:
  PIX = 单个PCIe bridge (最近, 同CPU但不同PCIe switch)
  PXB = 多个PCIe bridge (次近, 同NUMA但跨PCIe switch)
  SYS = 跨NUMA (经过SMP互连如UPI/QPI + PCIe)
```

**意外发现**: SYS连接(跨NUMA)带宽比PXB更高!

| 连接 | 类型 | 带宽 |
|------|------|------|
| GPU0→1 | PXB | 16.88 GB/s |
| GPU0→3 | PXB | 16.89 GB/s |
| GPU0→4 | SYS | 20.50 GB/s |
| GPU0→7 | SYS | 20.47 GB/s |
| GPU3→4 | SYS | 20.53 GB/s |

原因: 跨NUMA时，UPI互连带宽很高(~10.4 GT/s per link × 3 links)，
CPU内部路由比多个PCIe switch更快。同NUMA内的PXB需要经过
多个PCIe switch (PLX chip) → 增加延迟+降低带宽。

## 对推理/训练的影响

### Tensor Parallelism (TP)

TP每层需要AllReduce:
```
7B model TP=2:
  每层参数: 4096² × 4 bytes = 68MB
  AllReduce数据量: 68MB × 2(gradient+param) = 136MB
  AllReduce时间: 136MB / (17 GB/s × 2) ≈ 4ms

  对比NVLink:
  136MB / (300 GB/s × 2) ≈ 0.23ms → 17x更快!

  每层总时间: compute 0.17ms + AllReduce 4ms = 4.17ms
  AllReduce占比: 96%! → 几乎全部时间都在等通信!
```

**结论**: PCIe RTX 4090上TP几乎无用 → 通信占比96%+
**对比**: A100 NVLink上TP=8通信仅11.5%

### Data Parallelism (DP/DDP)

DP每步需要gradient AllReduce:
```
7B model DP=2:
  Gradient大小: 7B × 4 bytes = 28GB
  AllReduce时间: 28GB / 17 GB/s ≈ 1.65秒

  但gradient可以与计算overlap → 通信占比取决于overlap效率
  PCIe上overlap效果差(带宽太低) → 实际占比>25%
```

之前DDP实测:
- 2GPU: 1.81x / 90.4%效率
- 4GPU: 3.23x / 80.7%效率
- 8GPU: 预计6-7x / 75-87.5%效率

### Pipeline Parallelism (PP)

PP需要P2P通信(layer间):
```
PP=2: 每步传activation+gradient ≈ 几MB → 几十us
  通信占比很小 → PP在PCIe上可用!

但PP有气泡 → 每步效率 ≈ (P-1)/(M+P-1)
需要大batch(M>>P)来降低气泡比例
```

## 消费级GPU vs 数据中心GPU 通信对比

| GPU | 互连 | 通信BW | vs HBM | TP可行? |
|-----|------|---------|---------|---------|
| RTX 4090 (PCIe) | PCIe 4.0 | 17-20 GB/s | 2.1% | ❌ 不可行 |
| RTX 4090 (理论NVLink) | NVLink | 300 GB/s | 15% | ✅ 可行(但没有!) |
| A100 SXM | NVLink 3 | 300 GB/s | 15% | ✅ 推荐 |
| H100 SXM | NVLink 4 | 900 GB/s | 26% | ✅ 最优 |
| RTX A6000 (PCIe) | PCIe 4.0 | ~25 GB/s | ~3% | ❌ 不可行 |

## 实用建议

| 场景 | RTX 4090推荐 | 原因 |
|------|-------------|------|
| 7B推理单卡 | ✅ 最优 | 14GB fits, 无通信开销 |
| 13B推理 | ❌ 不适合TP | 通信太慢, 用量化+offload |
| 70B推理 | ❌ TP无意义 | 通信占比>96%, 几乎零效率 |
| 训练(小模型) | ✅ DP可行 | 通信占比适中(25%) |
| 训练(7B+) | ⚠️ DP+ZeRO | 需ZeRO-3+大batch降低通信占比 |
| 训练+PP | ✅ PP可行 | 通信量小, 气泡需要大batch弥补 |

**核心结论**: RTX 4090 PCIe集群适合DP训练和小模型单卡推理, **绝对不适合TP推理**。
NVLink是TP的前提条件 — A100/H100才有。