# CUDA Graph 实测: RTX 4090 vs A16 对比分析

> 2026-06-07 | RTX 4090 (SM 8.9, 128 MPs, 24GB) vs A16 (SM 8.6, 10 MPs, 15GB)

## 核心发现

**CUDA Graph 的收益取决于 kernel launch overhead 占比 → 小GPU收益大, 大GPU收益小**

| GPU | Launch Overhead | CUDA Graph Speedup (OPT-125M) | CUDA Graph Speedup (7B) |
|-----|----------------|-------------------------------|------------------------|
| A16 (10 MPs) | ~34us | 5.4x | N/A |
| RTX 4090 (128 MPs) | ~8.0us | 2.43x | 1.05x |
| A100 (108 MPs) | ~5-6us* | ~1.5-2x | ~1.1x |

*A100估算, 基于 NVIDIA 文献

## 为什么 A16 比 RTX 4090 的 CUDA Graph 收益大?

```
推理时间 = compute_time + launch_overhead × num_kernels

A16:
  compute = 0.87ms / 5.4 ≈ 0.16ms (每个kernel很小)
  launch = 34us × 48 ops = 1.63ms (launch占总时间的95%!)
  → 消除launch → 0.16ms → 10x+ 潜在加速 (实测5.4x, 因为有其他开销)

RTX 4090:
  compute = 0.87ms / 2.43 ≈ 0.36ms (每个kernel更快)
  launch = 8us × 48 ops = 0.38ms (launch占总时间的44%)
  → 消除launch → 0.36ms → 2.4x 加速

7B on RTX 4090:
  compute = 5.6ms (32层 × 0.17ms/层)
  launch = 8us × 160 ops = 1.28ms (占总时间23%)
  → 消除launch → 5.4ms → 1.05x (compute占77%, launch消除收益有限)
```

**关键**: RTX 4090 的 compute 比 A16 快 (128 MPs vs 10 MPs) → compute 占比更大 → launch 占比更小 → 消除 launch 的收益更小

## RTX 4090 详细数据

### Kernel Launch Overhead
```
size=1: 7.9us    size=16: 7.7us    size=256: 8.2us
size=1024: 8.2us  size=4096: 8.1us

→ Launch overhead ≈ 8us, 与 kernel 大小无关 (纯CPU→GPU dispatch成本)
```

### Single Matmul (decode step)
```
B=1  H=4K→4K:  1.02x (太小, launch不显著)
B=32 H=4K→4K:  1.05x
B=128 H=4K→4K: 0.98x (反而慢! 可能graph replay overhead > launch saving)
B=1  H=4K→32K: 1.00x (compute-dominated, launch无关)
```

### Multi-Op (Transformer layer)
```
1层 (5 ops):  1.04x  |  overhead_saved=7.2us/层
4层 (20 ops): 1.04x  |  overhead_saved=7.4us/层
32层 (160 ops):1.05x |  overhead_saved=7.7us/层

→ 层数增加, speedup微增 (更多launch被消除)
→ 但每层本身compute=0.17ms, launch=8us × 5=40us → 仅消除40us, 收益有限
```

### Decode Step Simulation (vLLM-style)
```
OPT-125M (12L, B=32): 2.43x ← 显著! 小模型 + 大batch
OPT-1.3B  (24L, B=8):  1.52x ← 中等
LLaMA-7B  (32L, B=4):  1.05x ← 微增

→ 模型越大, CUDA Graph收益越小 (compute占比越大)
```

### Memory Pool
```
Weights: 2672MB (32层 × H² × 2bytes ≈ 2.1GB)
Graph pool: 1-2MB (仅存intermediate activation addresses)
→ Graph pool ≈ 0.02% of weights → 几乎不增加内存!

Per-token: 0.1MB/tok → 极低开销
```

## vLLM CUDA Graph 实际策略

vLLM V1 的 CUDA Graph 使用:
- **FULL_AND_PIECEWISE 模式**: 默认, 对所有batch size捕获graph
- **5种模式**: NONE, FULL, PIECEWISE, FULL_AND_PIECEWISE, BREAKABLE
- **CudagraphDispatcher**: 根据 batch size 选择对应graph

**为什么 vLLM 仍然用 CUDA Graph?**
1. 即使1.05x加速, 在高QPS推理中每秒节省5% → 长时间累积显著
2. CUDA Graph提供**确定性延迟** → 无CPU→GPU sync jitter
3. 消除Python overhead → 从CPU侧的角度, 不是GPU侧
4. 与CUDA Stream overlap配合 → 通信-计算重叠更可控

## 实用建议

| 场景 | CUDA Graph推荐 | 原因 |
|------|---------------|------|
| 小模型 (<1B) inference | **强烈推荐** | 2-5x加速, launch是主要瓶颈 |
| 中模型 (1-7B) inference | **推荐** | 1.05-1.5x, 但消除jitter有价值 |
| 大模型 (70B+) inference | **可选** | <1.1x, compute-dominated |
| 小GPU (A16/T4) | **必须** | launch占比>50% → 显著加速 |
| 大GPU (A100/H100) | **推荐** | 消除Python jitter + 微加速 |
| 训练 | **不推荐** | 动态batch+gradient → graph无法捕获 |

## 与之前A16数据的对比

```
A16 (10 SM, PCIe):
  - Launch: 34us (4.3x slower than 4090)
  - CUDA Graph: 5.4x speedup (100 op simulation)
  - Multi-stream: 10.48x (4 GEMM overlap)
  - Stream切换: 346% overhead → 重叠负优化

RTX 4090 (128 SM, PCIe):
  - Launch: 8.0us (30x better than A16 estimate, 4.3x faster)
  - CUDA Graph: 1.02-2.43x (取决于模型大小)
  - Multi-stream: 无收益 (大GEMM饱和GPU, 已实测0.93x)
  - Stream切换: 低开销 (GPU足够快)
```

## Key Takeaways

1. **CUDA Graph收益 ∝ launch_overhead / total_time**: A16 launch=50%, 4090 launch=23-44%
2. **RTX 4090 launch ≈ 8us**: 比A16快4.3x → CUDA Graph收益相应降低
3. **小模型显著加速**: OPT-125M 2.43x → 生产部署应使用CUDA Graph
4. **大模型微加速**: 7B 1.05x → 主要价值是消除jitter而非速度
5. **Memory pool极小**: 0.1MB/tok → 不影响GPU内存预算
6. **确定性延迟**: CUDA Graph最实用的价值可能是消除Python jitter