# GPU 通信-计算重叠实验

> 文件: `tools/gpu_comm_compute_overlap.py`
> 日期: 2026-06-04
> GPU: A16 15GB (SM 8.6, 10 SM, PCIe, 无 NVLink)

## 动机

验证 vLLM UBatch DBO 机制中双 CUDA Stream 通信-计算重叠的实际效果。

---

## 实验 1: 双 Stream 重叠基础验证

| 方式 | 耗时 |
|------|------|
| Serial (单stream) | 0.114ms |
| Overlapped (双stream) | 0.152ms |
| **Speedup** | **0.75x (反而更慢!)** |

**结论**: 小数据量下 stream 切换开销 > 重叠收益。

---

## 实验 2: Event vs CPU Sync

| 方式 | 耗时 |
|------|------|
| cuda.Event (GPU-only) | 0.119ms |
| torch.cuda.synchronize (CPU) | 0.125ms |
| **Event 快** | **1.04x** |

CPU 同步差不大，因为在 A16 (10 SM) 上没有足够并行度。

---

## 实验 3: 数据大小对重叠效率的影响

| Compute Batch | Comm MB | Speedup |
|:---:|:---:|:---:|
| 32 | 1 | 0.69x |
| 32 | 64 | 0.96x |
| 64 | 64 | **1.17x** |
| 128 | 64 | 0.96x |
| 512 | 64 | 0.99x |

**仅 compute=64/comm=64MB 时 speedup >1.0**。

大 tensor 时重叠开始抵消开销，但收益仍然微薄 (<20%)。

---

## 实验 4: Microbatch 流水线

| 方式 | 耗时 |
|------|------|
| Serial 2 batches | 0.131ms |
| Pipelined 2 batches | 0.211ms |
| **Speedup** | **0.62x** |

**结论**: A16 上微批次流水线没有收益，开销反而更大。

---

## 实验 5: Stream 切换开销

| 方式 | 耗时 |
|------|------|
| Single stream (10 matmuls) | 0.205ms |
| Multi stream switch (交替) | 0.916ms |
| **Overhead** | **346%** |

**结论**: Stream 切换开销在 A16 上是主导因素。

---

## 核心教训

### 为什么 A16 上重叠没有效果？

```
A100 (108 SM):  [================ compute SM 0-53 ======][========== comm SM 54-107 ==========]
                  ↑ 真正的硬件并行，NVLink 带宽独立

A16 (10 SM):    [== compute SM 0-4 ==][-- stream switch overhead --][== comm SM 0-4 ==]
                  ↑ 只有 10 SM，两个 stream 互相抢占同一组 SM
                  ↑ PCIe：compute 和 comm 争抢同一内存带宽
```

### 重叠有收益的必要条件

1. **足够 SM**: ≥ 40 SM 才可能有效并发（通过 `SMControlContextManager` 预留 SM 给通信）
2. **独立带宽**: NVLink/NVSwitch 提供 compute/comm 独立带宽
   - A16 用 PCIe，compute 和 comm 共享同一 HBM 带宽
3. **大数据量**: tensor 足够大才能抵消 stream 切换开销

### 对 vLLM UBatch (DBO) 的适用性

UBatch DBO 的 `compute_stream + comm_stream` 双流重叠设计是为 **A100/H100** 这类大 GPU 设计的：
- A100: 108 SM，NVLink 600GB/s
- H100: 132 SM，NVLink 900GB/s

在 A16 (10 SM, PCIe 16GB/s) 上，这个优化**反而会有负面影响**。

---

## 代码修复

实验4的流水线版本实际上没有实现真正的重叠（所有 compute 在同一 stream 上顺序执行）。需要修复为：ubatch 0 compute → ubatch 1 compute（填补 ubatch 0 comm 的空隙）。但这需要更精细的 stream 编排。

待解决的脚本改进：
- 用 `torch.cuda.Event` 的 fine-grained 依赖替代全局 synchronize
- 增大 compute 量以模拟真实 model layer

---

## 结果文件

`/root/comm_compute_overlap_results.json`
