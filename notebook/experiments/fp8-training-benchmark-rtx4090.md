# FP8 Training Acceleration Benchmark — RTX 4090 (SM89)

> 2026-06-08 | TransformerEngine 2.17.0.dev0 | 实测数据
> 硬件: RTX 4090 24GB SM89, CUDA 12.8, PyTorch 2.9.0+cu128
> 模型: 7M (H=2560, heads=20, kv_heads=5, SwiGLU MLP)

## 1. 核心结果

### 1.1 Training Throughput (tokens/s)

| Batch | BF16 | FP8 DelayedScaling | FP8 CurrentScaling | DS/BF16 | CS/BF16 |
|-------|------|---------------------|--------------------|---------|---------|
| 1 | 178K | 134K | 129K | 0.75x | 0.72x |
| 4 | 229K | 340K | 330K | **1.48x** | **1.44x** |
| 8 | 232K | 368K | 358K | **1.59x** | **1.55x** |
| 16 | 232K | 370K | 359K | **1.59x** | **1.55x** |
| 32 | 233K | 367K | 353K | **1.57x** | **1.52x** |

### 1.2 FP8 Accuracy (cos_sim)

| Batch | DelayedScaling cos_sim | CurrentScaling cos_sim | DS max_diff | CS max_diff |
|-------|----------------------|----------------------|------------|------------|
| 1 | **1.000** | **1.000** | 0.063 | 0.080 |
| 4 | **1.000** | **1.000** | 0.078 | 0.070 |
| 8 | 0.996 | 0.996 | 0.078 | 0.078 |
| 16 | **1.000** | **1.000** | 0.082 | 0.082 |
| 32 | 0.996 | 0.996 | 0.086 | 0.089 |

**精度总结**: cos_sim 最低 0.996, 最高 1.000 → FP8训练精度极好!
max_diff仅0.06-0.09 (相对于norm≈1152-6528) → 相对误差<0.01%

### 1.3 Peak Memory

| Batch | BF16 (GB) | FP8 DS (GB) | FP8 CS (GB) | DS/BF16 ratio |
|-------|-----------|------------|------------|--------------|
| 1 | 0.50 | 1.04 | 1.04 | 2.09x |
| 4 | 0.74 | 1.28 | 1.28 | 1.73x |
| 8 | 1.08 | 1.59 | 1.59 | 1.47x |
| 16 | 1.80 | 2.33 | 2.33 | 1.30x |
| 32 | 3.22 | 3.79 | 3.79 | 1.18x |

## 2. 分析

### 2.1 为什么B=1时FP8反而慢0.75x?

**BF16 B=1**: 2.88ms → 训练吞吐 178K tok/s
**FP8 B=1**: 3.82ms → 训练吞吐 134K tok/s

原因分析:
- **量化开销占主导**: B=1时GEMM很小(M=512, N=2560→K=2560/3840), FP8 quantize/dequantize的固定开销(amax扫描、scale计算、kernel launch)比GEMM本身还大
- **GEMM太小不够加速**: BF16 GEMM在SM89上已经够快→FP8节省的computation不足以覆盖量化overhead
- **kernel launch overhead**: FP8需要额外launch quantize kernel + dequantize kernel → 2次额外kernel launch
- **规律**: GEMM越大→FP8加速比例越高; GEMM太小→FP8反而变慢

这与之前的Quantization Inference benchmark一致: INT4 weight-only对小GEMM也没加速(0.87x)

### 2.2 为什么B≥4时FP8加速1.48-1.59x?

**核心**: TE的fused kernel避免了Python dequant overhead!

对比我们之前的发现:
- **之前(Python dequant)**: FP8 dequant慢1.5-2.5x → FP8比BF16更慢!
- **现在(TE fused kernel)**: FP8加速1.48-1.59x → FP8比BF16更快!

**为什么fused kernel有效?**
1. quantize+GEMM+dequantize在一个CUDA kernel完成 → 无Python-level数据搬运
2. FP8数据在GPU上就地量化→就地GEMM→就地dequant → 无CPU↔GPU往返
3. FP8 GEMM利用SM89的HMMA tensor core → FP8 E4M3输入→FP32累加→BF16输出
4. FP8权重8bit vs BF16权重16bit → weight加载带宽减半 → memory-bound场景受益更大

**理论加速极限**:
- FP8权重8bit→2x bandwidth节省 → memory-bound场景理论加速2x
- FP8 GEMM比BF16 GEMM吞吐量2x(compute-bound) → compute-bound场景理论加速2x
- 实测1.48-1.59x → 不完全达到理论2x → 因为量化开销、kernel launch、dequantize等仍有额外cost

### 2.3 为什么DelayedScaling比CurrentScaling略快?

| Batch | DS tok/s | CS tok/s | DS/CS ratio |
|-------|---------|---------|------------|
| 4 | 340K | 330K | 1.03x |
| 8 | 368K | 358K | 1.03x |
| 16 | 370K | 359K | 1.03x |
| 32 | 367K | 353K | 1.04x |

DelayedScaling快2-3%! 原因:
- **DelayedScaling**: scale从上一步的amax_history计算 → 无需当前tensor扫描amax → 量化kernel更简单
- **CurrentScaling**: 需要先扫描当前tensor的amax → 计算scale → 然后量化 → 多一步amax扫描
- **额外扫描**: CurrentScaling每步多一个amax-reduction kernel → 少量额外时间

但注意: CurrentScaling理论上精度更好(无delay) → 实测cos_sim几乎相同

### 2.4 为什么FP8内存反而更多?

BF16: 0.50-3.22GB → FP8: 1.04-3.79GB → FP8多2x(B=1) → 1.18x(B=32)

原因:
- **Quantizer metadata**: FP8需要存储scale tensor + amax tensor → 每层额外内存
- **rowwise+columnwise双存储**: TE为backward同时存rowwise和columnwise量化数据 → 2x FP8 tensor存储
- **BF16权重仍需保留**: FP8不是weight-only → 训练时BF16权重仍需保留(用于gradient更新)
- **scale+amax tensors**: 每层6个quantizer → 每个quantizer有scale和amax → 额外metadata内存

当B增大→GEMM activation内存占主导→quantizer metadata占比下降→ratio从2x降到1.18x

## 3. 与之前实验对比

| 实验 | 方法 | 场景 | 加速 | 精度 |
|------|------|------|------|------|
| Quant Inference | FP8 Python dequant | 推理 | **0.4-0.67x** ❌ | 好 |
| Quant Inference | INT4 weight-only | 推理 | 0.87-1.08x | 好(75%内存省) |
| Quant Inference | INT8 KV cache | 推理 | **1.00x** | 完美(50%KV省) |
| TE Benchmark | FP8 DelayedScaling | 训练 B≥4 | **1.48-1.59x** ✅ | 0.996-1.0 |
| TE Benchmark | FP8 CurrentScaling | 训练 B≥4 | **1.44-1.55x** ✅ | 0.996-1.0 |

**关键对比**: FP8 Python dequant慢→fused kernel快! 这是TE的核心价值:
- Python-level FP8: 量化→Python搬运→dequant→Python搬运→GEMM → 2x overhead
- TE fused kernel: 量化→GEMM→dequantize (1个CUDA kernel) → 0 overhead

## 4. Roofline分析

### 4.1 BF16 Training Roofline

模型: 7M (95M params), H=2560, MLP=4H

BF16训练每步内存流量:
- 权重: 95M × 2B = 190MB (加载+写入gradient)
- Activation: B × 512 × 2560 × 2B ≈ 2.6MB (B=1)
- Gradient: 同activation大小

RTX 4090 HBM: 890.8 GB/s

**BF16 B=1 roofline**:
- 权重加载: 190MB × 2(read+write) / 890.8GB/s = 0.43ms
- 实测: 2.88ms → 6.7x overhead → 大部分是compute

**FP8 B=1**:
- FP8量化+GEMM+反量化: 3.82ms → 比BF16多1ms → 量化开销
- FP8权重: 95M × 1B = 95MB → 权重加载快2x → 但量化开销抵消

**FP8 B=32**:
- BF16: 70.24ms → FP8: 44.63ms → 加速1.57x
- 此时GEMM变大→compute bound→FP8 GEMM更快→量化开销占比小

### 4.2 加速随B的变化趋势

```
B=1:  0.75x  (量化overhead > GEMM加速)
B=4:  1.48x  (GEMM加速开始超过量化overhead)
B=8:  1.59x  (峰值加速)
B=16: 1.59x  (持续峰值)
B=32: 1.57x  (略微下降→可能因为内存压力)
```

趋势: 从B=4开始FP8有加速, 到B=8-16达到峰值1.59x, B=32略微下降

## 5. 生产建议 (RTX 4090)

| 场景 | 建议 | 原因 |
|------|------|------|
| Training B≥4 | **FP8 DelayedScaling** | 1.48-1.59x加速, 精度好 |
| Training B=1 | **BF16** | FP8反而慢0.75x |
| Training 精度优先 | **FP8 CurrentScaling** | 理论上更精确(实测差不多) |
| Inference | **INT4/INT8 KV** | TE不是推理最优选择 |
| Communication | **FP8 quantize** | 带宽减半(之前的结论) |

**重要**: FP8训练加速需要TE的fused kernel → Python-level FP8没有加速效果!

## 6. TE安装经验 (Section 10补充)

见 `transformer-engine-fp8-rtx4090.md` Section 10

关键构建参数:
```bash
# C++ shared library
CUDACXX=/usr/local/cuda-12.8/bin/nvcc pip install --no-build-isolation --no-deps -e .

# PyTorch extension
CPLUS_INCLUDE_PATH=...nvidia/nccl/include:...nvidia/cudnn/include \
  CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
  pip install --no-build-isolation --no-deps -e .
```

---

**数据**: `results/te_fp8_training_benchmark.json`
**脚本**: `tools/te_fp8_training_benchmark_4090.py`
**相关笔记**: `transformer-engine-fp8-rtx4090.md`, `quantization-inference-rtx4090.md`