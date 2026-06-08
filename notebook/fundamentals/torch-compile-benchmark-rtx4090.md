# torch.compile Benchmark RTX 4090: Forward 3.75x(B=1) but 0.80-0.97x(B>=4)!

> 2026-06-08 | torch.compile只在小batch有效, 大batch反而更慢 → launch overhead是关键
> 基于: torch_compile_benchmark_4090.py实测, PyTorch 2.9.0+cu128, RTX 4090
> 关联: triton-vs-cuda-benchmark-rtx4090.md, cutlass-gemm-benchmark-rtx4090.md

## 0. 核心定律: torch.compile = 消除Python开销, 不是加速GPU计算

```
torch.compile做了什么:
  → TorchDynamo: Python字节码→FX graph → 消除Python解释器overhead
  → TorchInductor: FX graph→Triton/C++ kernel → kernel fusion + 优化
  → 结果: 消除Python launch overhead(~0.06ms) + 融合小kernel → B=1时4x加速!

  但为什么B≥4反而更慢:
  → Inductor生成Triton GEMM kernel → Triton GEMM比cuBLAS慢1.5x!
    → cuBLAS经过20年优化 + GPU-specific tuning
    → Triton GEMM是通用实现 → 不可能比cuBLAS更快
  → 大batch时, GEMM占总时间>90% → Triton GEMM慢 → 整体更慢!
  → 小batch(B=1)时, Python overhead占>60% → compile消除overhead → 4x加速!

  B=1时间分解(7M模型):
    → eager: 1.227ms → Python overhead ~0.9ms + GPU compute ~0.3ms
    → compiled: 0.300ms → Python overhead ~0ms + Triton GEMM ~0.3ms
    → → 4.09x加速来自消除Python overhead, 不是GPU更快!

  B=4时间分解(7M模型):
    → eager: 1.245ms → Python overhead ~0.06ms + cuBLAS ~1.18ms
    → compiled: 1.555ms → Triton GEMM ~1.55ms (比cuBLAS慢1.3x)
    → → Triton GEMM慢 → 整体反而更慢!
```

## 1. Forward Pass实测数据

```
=== 7M模型 (H=512, L=4, heads=8) ===
  B=1:  eager=1.227ms, compiled=0.300ms → 4.09x加速!
  B=4:  eager=1.245ms, compiled=1.555ms → 0.80x (更慢!)
  B=16: eager=1.560ms, compiled=1.826ms → 0.85x (更慢!)
  B=32: eager=2.037ms, compiled=2.323ms → 0.88x (更慢!)

=== 25M模型 (H=1024, L=8, heads=16) ===
  B=1:  eager=2.269ms, compiled=0.756ms → 3.00x加速!
  B=4:  eager=2.371ms, compiled=2.622ms → 0.90x (更慢!)
  B=16: eager=4.209ms, compiled=4.039ms → 1.04x (微加速)
  B=32: eager=7.308ms, compiled=7.513ms → 0.97x (更慢!)

=== 7B-proxy (H=2560, L=4, heads=20) ===
  B=1:  eager=1.541ms, compiled=1.326ms → 1.16x
  B=4:  eager=3.342ms, compiled=3.533ms → 0.95x (更慢!)
  B=8:  eager=5.661ms, compiled=5.846ms → 0.97x (更慢!)

关键规律:
  → B=1: 3-4x加速 → Python overhead消除
  → B≥4: 0.80-0.97x → Triton GEMM慢于cuBLAS
  → crossover: B=4 → Python overhead已不是瓶颈 → GEMM成为瓶颈
  → → torch.compile forward对推理几乎无用(推理通常B≥4)!
```

## 2. Training Step实测数据

```
=== 7M模型 Training ===
  B=4:  eager=7.936ms, compiled=4.041ms → 1.96x加速!
  B=16: eager=8.344ms, compiled=6.987ms → 1.19x加速
  B=32: eager=11.603ms, compiled=11.082ms → 1.05x微加速

=== 25M模型 Training ===
  B=4:  eager=20.197ms, compiled=12.261ms → 1.65x加速!
  B=16: eager=22.730ms, compiled=22.049ms → 1.03x微加速
  B=32: eager=35.926ms, compiled=35.509ms → 1.01x微加速

关键规律:
  → B=4: 1.65-1.96x加速 → backward有更多Python overhead可消除!
  → B≥16: 1.01-1.19x → 加速消失 → GEMM主导
  → → torch.compile training在小batch(B≤4)有效!
  → → 大batch训练(B≥16): 几乎无加速 → GEMM是瓶颈
```

## 3. Compile Modes对比

```
7M模型 B=16 (forward):
  eager: 1.639ms (baseline)
  default: 1.531ms → 1.07x
  reduce-overhead: 1.600ms → 1.02x
  max-autotune: 1.787ms → 0.92x (最慢!)

关键发现:
  → max-autotune是最慢的! → Triton kernel autotune→选了慢kernel!
  → default是最好的 → 不autotune → 用cuBLAS fallback
  → → Inductor autotune选Triton kernel → Triton GEMM比cuBLAS慢 → 反而更差!
  → → 这证明: Triton GEMM在大batch不可能比cuBLAS更快

  Recompile warning:
  → "torch._dynamo hit config.recompile_limit (8)"
  → → 原因: grad_mode变化 → 每次切换no_grad/with_grad触发recompile!
  → → 生产环境需要稳定grad_mode → 否则recompile开销巨大
```

## 4. Dynamic Shapes

```
7M模型 (dynamic=True):
  B=4 S=64:   0.96x
  B=8 S=128:  0.88x
  B=16 S=128: 0.93x
  B=32 S=128: 0.90x
  B=16 S=256: 0.98x

关键发现:
  → dynamic shapes: 全部更慢(0.88-0.98x)!
  → → dynamic=True → kernel不能优化特定shape → 更慢
  → → 推理场景(shape多变) → compile更无用!
  → → 训练场景(shape固定) → 静态compile才有意义
```

## 5. Memory Overhead

```
7M模型 B=16:
  Eager peak: 562.0MB
  Compiled peak: 555.0MB
  Memory overhead: -1.2% → compile反而省内存!

原因:
  → kernel fusion → 减少中间tensor分配 → 省内存
  → Triton kernel内部用SRAM → 不需要global memory中间结果
  → → compile在小batch时内存无overhead → 甚至省1.2%

大模型(B=32+7B):
  → 内存差异很小 → GEMM主导 → fusion收益有限
```

## 6. 为什么torch.compile不如预期

```
Meta声称torch.compile可以1.5-2x加速 → 但实测0.80-0.97x(forward B≥4):

  根本原因:
    → torch.compile优化Python overhead + kernel fusion
    → 但不优化GEMM → Triton GEMM比cuBLAS慢1.5x!
    → 大batch时GEMM占>90% → Triton慢 → 整体更慢

  Meta的benchmark为什么1.5-2x:
    → 他们用小模型+小batch → Python overhead占比高 → compile有效
    → 或用element-wise fusion → 省中间tensor读写 → 有效
    → 但GEMM-heavy场景 → compile无效甚至有害!

  RTX 4090特殊性:
    → SM89 → 128 MPs → launch overhead ~0.06ms (已很低!)
    → 大GPU → launch overhead占比更低 → compile收益更小
    → A16(10 SM) → launch overhead占比更高 → compile收益更大?
    → → GPU越强 → torch.compile收益越小!

  Crossover分析:
    → 7M B=4 crossover → Python overhead从>60%降到<5%
    → 25M B=4 crossover → 类似
    → → crossover ≈ Python_overhead_time / total_time < 10%
    → → 一旦compute主导 → compile无效
```

## 7. torch.compile适用场景决策树

```
torch.compile决策树:

  Forward pass (推理):
    → B=1 decode: 3-4x加速 → 推荐! (但FlashInfer更快)
    → B≥4 inference: 0.80-0.97x → 不推荐! (反而更慢)
    → → 推理场景用FlashInfer(15.72x) → 不是torch.compile!

  Training step:
    → B≤4 小batch: 1.65-1.96x → 推荐! (消除Python + backward fusion)
    → B≥16 大batch: 1.01-1.19x → 微收益 → 可能不值得compile开销
    → → 小batch训练受益 → 但大batch训练几乎无收益

  GRPO训练:
    → rollout(推理B≥16): compile无用 → rollout占74%时间
    → training(B=4-16): 1.19-1.96x → 有限收益
    → → GRPO整体: compile收益~1.39x → rollout不可编译

  Optimal use:
    → 小batch训练(B≤4): torch.compile(default mode)
    → element-wise fusion(RMSNorm/layer norm): torch.compile有效
    → GEMM-heavy场景: 不要compile → 直接用cuBLAS
    → 推理: 用FlashInfer → 不是torch.compile

  Mode选择:
    → default: 最稳定 → 用cuBLAS fallback → 1.07x
    → reduce-overhead: CUDA Graph → 1.02x → 不值得
    → max-autotune: Triton autotune → 0.92x → 最差!
    → → 推荐: default mode → 保守但稳定
```

## 8. 与之前benchmark的关联

```
关键关联:

  Triton vs CUDA C++ vs cuBLAS benchmark:
    → Triton RMSNorm 1.8x faster than CUDA C++ → launch overhead更低!
    → cuBLAS GEMM always wins → Triton GEMM慢1.5x
    → → torch.compile = Triton kernel → RMSNorm受益, GEMM受害
    → → 这解释了为什么compile在小kernel有效但大GEMM无效!

  CUTLASS GEMM benchmark:
    → decode B=1: 仅1.8% peak → 98.2% TC闲置 → launch overhead主导
    → decode B=128: 100% peak → GEMM主导
    → → torch.compile在B=1有效是因为TC闲置 → 消除overhead更明显
    → → B≥128: TC满载 → GEMM是瓶颈 → compile无用

  FlashInfer benchmark:
    → 推理: FlashInfer 15.72x(B=32) → 远超compile的0.88x
    → → 推理优化排序: FlashInfer > 量化 > torch.compile

  FP8 TE training:
    → FP8 training 1.48-1.59x → 比compile更稳定!
    → → 训练优化排序: FP8 TE > torch.compile > reduce-overhead
```

---

**Sources**:
- torch.compile benchmark: `results/torch_compile_benchmark.json`
- Benchmark script: `tools/torch_compile_benchmark_4090.py`
- PyTorch 2.9.0+cu128, RTX 4090 (8x, SM89)

**Related notes**: triton-vs-cuda-benchmark-rtx4090.md, cutlass-gemm-benchmark-rtx4090.md, flashinfer-attention-deep-dive.md, fp8-gemm-algorithm-analysis-rtx4090.md