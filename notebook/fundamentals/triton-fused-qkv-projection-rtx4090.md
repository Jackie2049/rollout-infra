# Triton Fused QKV Projection — RTX 4090

> 2026-06-07 | Triton fused QKV kernel **比PyTorch更慢(0.19-0.57x)**, stacked QKV无明显加速(0.94-1.21x), cuBLAS已近最优, kernel launch主导小GEMM

## 核心发现

```
Triton fused QKV kernel: 正确性完美(cos_sim=1.0), 但比PyTorch慢2-5x!
Stacked QKV (单matmul): 仅MHA时1.21x, GQA时0.97x → 无实际加速

┌──────────────────────────────────────────────────────────────┐
│ QKV投影RTX 4090实测:                                        │
│ Sequential (3 matmul): ~0.08-0.09ms ← kernel launch主导    │
│ Stacked (1 matmul):    ~0.07-0.09ms (0.94-1.21x) ← 无意义 │
│ Triton fused:          ~0.15-0.44ms (0.19-0.57x) ← 更慢!  │
│                                                              │
│ 原因: cuBLAS FP16已达101% peak! 简单Triton无法超越           │
│       kernel launch ~8μs × 3次 ≈ 操作本身的计算时间           │
│       Triton naive for-loop matmul ≈ cuBLAS tiled GEMM 1/5  │
└──────────────────────────────────────────────────────────────┘

→ **教训再次确认**: 不要盲目写自定义kernel, 先benchmark PyTorch!
→ cuBLAS高度优化(101% peak FP16) → 手写kernel需要tl.dot()才能接近
→ 小操作(B×S≤16K) → kernel launch主导 → CUDA Graph才是正确优化路径
```

## 实验1: 正确性验证

```
4种配置 × Q/K/V = 全部cos_sim=1.0 (max_diff < 6.5e-5)

| 配置        │ B×S  │ D   │ H_Q │ H_KV │ Q cos │ K cos │ V cos │ Q diff │
| MHA_256     │ 512  │ 256 │ 8   │ 4    │ 1.000 │ 1.000 │ 1.000 │ 1.9e-5 |
| GQA_equal   │ 512  │ 256 │ 8   │ 8    │ 1.000 │ 1.000 │ 1.000 │ 1.5e-5 |
| large_GQA   │ 8K   │ 512 │ 16  │ 4    │ 1.000 │ 1.000 │ 1.000 │ 4.9e-5 |
| small       │ 64   │ 256 │ 8   │ 2    │ 1.000 │ 1.000 │ 1.000 │ 9.5e-6 |

→ Triton kernel计算正确, 但精度不如FP32(max_diff在FP16误差范围内)
→ large_GQA diff更大(4.9-6.5e-5) → 更多数据→浮点累积误差
→ Triton实现数值等价PyTorch sequential, 只是效率不同
```

## 实验2: 性能对比 — Triton更慢!

```
3种方法: Sequential(3 matmul) / Stacked(1 matmul) / Triton(3 kernel launch)

| 配置         │ Seq(ms) │ Stacked(ms) │ Stack↑ │ Triton(ms) │ Tri↑    │
| B1_GQA4     │ 0.088   │ 0.086       │ 1.02x  │ 0.155      │ 0.57x ⚠ │
| B4_GQA4     │ 0.091   │ 0.090       │ 1.02x  │ 0.156      │ 0.58x ⚠ │
| B16_GQA4    │ 0.090   │ 0.087       │ 1.04x  │ 0.163      │ 0.55x ⚠ │
| B32_GQA4    │ 0.077   │ 0.079       │ 0.97x  │ 0.254      │ 0.30x ⚠ │
| B64_GQA4    │ 0.081   │ 0.084       │ 0.96x  │ 0.436      │ 0.19x ⚠ │
| D512_GQA4   │ 0.084   │ 0.090       │ 0.94x  │ 0.155      │ 0.54x ⚠ │
| D1024_GQA8  │ 0.090   │ 0.081       │ 1.11x  │ 0.388      │ 0.23x ⚠ │

→ Triton全部比PyTorch慢! 原因:
  1. Triton kernel用for-loop逐元素累加 → cuBLAS用tiled GEMM → 5x差距
  2. Triton 3次kernel launch × 自定义kernel → 更慢
  3. cuBLAS FP16达101% peak → 手写很难超越

→ Stacked方法无加速:
  - 小batch: 3个小matmul vs 1个稍大matmul → 时间几乎相同
  - 原因: kernel launch ~8μs主导 → 计算本身<20μs → 合并3×8=24μs vs 1×8=8μs
  - 但cuBLAS对小GEMM有专门优化 → launch不是唯一开销 → stacked收益微弱

→ D1024_GQA8 stacked 1.11x → 大权重时stacked稍有收益(M=512→更大GEMM更好利用GPU)
```

## 实验3: GQA头数 × 融合收益

```
固定H_Q=8, D=256, B=16, S=128 → 变化H_KV

| HKV │ Seq(ms) │ Stacked(ms) │ Stack↑ │ Q(ms) │ KV(ms) │ KV占QKV% │
| 8   │ 0.084   │ 0.069       │ 1.21x  │ 0.034 │ 0.050  │ 67%      │
| 4   │ 0.076   │ 0.078       │ 0.97x  │ 0.036 │ 0.052  │ 50%      │
| 2   │ 0.080   │ 0.081       │ 0.99x  │ 0.034 │ 0.052  │ 33%      │
| 1   │ 0.079   │ 0.076       │ 1.04x  │ 0.033 │ 0.052  │ 20%      │

→ **反直觉**: MHA(HKV=8)时stacked最快(1.21x), GQA时反而无加速!

原因分析:
  → MHA: Q/K/V权重相同大小 → stack=[3×256, 256] → cuBLAS单个大GEMM更好
  → GQA-4: Q权重256但K/V权重128 → stack=[256+128+128, 256]=[512,256]
  → 但512×256 vs 256×256 + 128×256×2 → cuBLAS对小GEMM同样高效
  → KV总占比: MHA 67% → GQA-4 50% → GQA-2 33% → MQA 20%
  → GQA减少KV维度 → 但cuBLAS已足够快 → stacking收益被稀释

→ **关键**: Q-only 0.034ms vs KV-only 0.052ms → KV稍慢是因为2次matmul
  → 3次matmul vs 1次stacked → 节省2次kernel launch ~16μs
  → 但QKV总时间0.08ms → 16μs节省约20% → 仅MHA时可见
```

## 实验4: Decode vs Prefill

```
Decode(B=1,S=1-8) vs Prefill(B=4-16,S=128-512)

| 场景            │ B×S │ Seq(ms) │ Stacked(ms) │ Stack↑ │ Decode │
| B1_S1           │ 1   │ 0.084   │ 0.081       │ 1.04x  │ ✓      │
| B1_S8           │ 8   │ 0.089   │ 0.089       │ 1.00x  │ ✓      │
| B4_S128         │ 512 │ 0.086   │ 0.088       │ 0.98x  │ ✗      │
| B16_S512        │ 8K  │ 0.080   │ 0.077       │ 1.04x  │ ✗      │

→ **所有场景时间几乎相同(~0.08ms)** — kernel launch主导!
  → B×S=1 → 0.084ms → 几乎全是kernel launch时间
  → B×S=8K → 0.080ms → 计算仍然极快(RTX 4090 HBM 920GB/s)
  → Stacked收益在所有场景≤1.04x → 无实际意义

→ Decode时QKV投影每层0.08ms × 32层 ≈ 2.56ms → 占decode总时间?
  → 7B每层0.40ms(B=32) → QKV 0.08ms占20% → 还是有意义的
  → 但优化路径不是自定义kernel → 而是CUDA Graph消除launch!
```

## 为什么Triton QKV更慢 — 深层分析

```
Triton kernel设计问题:

1. **Naive for-loop matmul** (当前实现):
   → for d_start in range(0, N_OUT, BLOCK_D):
   →     w_col = tl.load(w[d_start:, :], ...)  # 逐块加载权重列
   →     result = tl.sum(x_row * w_col, axis=1)  # 逐元素dot product
   → 问题: 每个output维度单独计算 → 没有利用GPU并行度
   → cuBLAS: tiled GEMM → 分块并行 → SM全部利用

2. **缺少tl.dot()**:
   → Triton生产kernel用tl.dot()做tiled matmul → 类似cuBLAS的分块策略
   → 我们用tl.sum(x*w) → 逐元素乘+求和 → 每个program只算1行
   → tl.dot()可以一次计算BLOCK_M×BLOCK_N输出块 → 并行度高

3. **3次独立kernel launch**:
   → Q, K, V各1次kernel launch → 3×8μs = 24μs overhead
   → 真正的fused kernel应该是单次launch → 在1个kernel内计算Q+K+V
   → 我们的fused_qkv_row_kernel不是真正的fused → 只是对3个投影分别调用

4. **BLOCK_D太小(64)**:
   → N_OUT=256(H_Q×D_HEAD) → 分4次迭代 → 4次HBM load × 每行
   → cuBLAS一次加载整行 → 更少的HBM访问

→ **正确做法**: 参考vLLM/SGLang的QKV Triton kernel:
  → 使用tl.dot()做tiled matmul → BLOCK_M×BLOCK_K×BLOCK_N
  → 单次kernel launch计算Q+K+V → stack权重在kernel内
  → 可以额外fuse bias add → 节省1次kernel launch
```

## QKV投影的正确优化路径

```
RTX 4090实测结论:

❌ 自定义Triton kernel: 0.19-0.57x (更慢!)
❌ Stacked matmul: 0.94-1.21x (无意义)
✓ cuBLAS FP16: 101% peak (已最优!)

正确优化路径:

1. **CUDA Graph** (消除kernel launch):
   → QKV每层3次matmul → 3×8μs = 24μs launch overhead
   → CUDA Graph: 整个layer capture → 1次graph launch ~0.034ms
   → 32层 × 24μs = 768μs → Graph消除 ~0.7ms → 但我们实测仅1.05x
   → 原因: 7B每层0.4ms → 24μs launch占比<6% → 收益有限

2. **torch.compile** (kernel fusion):
   → Inductor自动分析QKV → 可能fuse成更少kernel
   → 我们实测reduce-overhead 3.75x → 主要来自CUDA Graph
   → 但这是整个forward的收益 → QKV单独收益难以分离

3. **Production Triton kernel** (vLLM/SGLang方式):
   → 使用tl.dot() + 单次launch + 可选bias fusion
   → vLLM QKVLinearKernel: tl.dot() + bias + activation fusion
   → 主要收益不在加速 → 而在节省内存+减少launch次数
   → 在大模型(7B+)+大batch(B≥64)时才有收益

→ **RTX 4090小模型结论**: QKV投影无需自定义kernel!
  → cuBLAS已最优 → CUDA Graph消除launch → torch.compile自动fusion
  → 自定义kernel只在大模型+大batch+特定场景(bias/activation fusion)才有收益
```

## 与之前实验的串联

```
这次实验再次确认了我们之前的核心发现:

1. **cuBLAS FP16达101% peak** (torch.compile benchmark)
   → 任何手写kernel要超越cuBLAS → 必须用tl.dot() tiled GEMM
   → naive for-loop matmul → cuBLAS的1/5 → 更慢

2. **Kernel launch ~8μs on RTX 4090** (CUDA Graph benchmark)
   → 小操作: 3×8μs = 24μs ≈ 操作本身 → stacking节省launch有限
   → 大操作: 0.4ms/层 → 24μs占比<6% → launch不是瓶颈

3. **不要盲目写自定义kernel** (Triton Workshop)
   → 先benchmark PyTorch → 发现cuBLAS已最优 → 再决定是否写kernel
   → Triton softmax 0.87x(cuDNN更优) → Triton LayerNorm 1.35x(有收益)
   → GEMM → cuBLAS最优 → 只有非GEMM操作(归一化/activation)Triton才可能快

4. **优化铁律再次验证** (Triton fused GRPO advantage):
   → QKV投影每层0.08ms → 32层=2.56ms → 占7B推理0.4ms×32=12.8ms的20%
   → 不是主要瓶颈(MLP占68%) → 即使优化QKV到0也不改变大局
   → 真正瓶颈: decode memory-bound → 优化HBM利用率才是关键

→ **完整优化优先级(RTX 4090)**:
  P0: vLLM async rollout → 解决74%瓶颈 → 2-5x E2E
  P1: torch.compile → 自动fuse+CUDA Graph → 1.3x
  P2: Prefix Sharing → n=8时58% → 1.5-2x
  P3: FlashAttention → 省内存(85-97%)而非加速
  P4: 自定义kernel → 仅在非GEMM操作有收益(归一化/activation)
```

## 工具

- `tools/triton_fused_qkv_projection.py` — Triton fused QKV kernel + 4实验benchmark
- `results/triton_fused_qkv_projection.json` — 完整结果数据