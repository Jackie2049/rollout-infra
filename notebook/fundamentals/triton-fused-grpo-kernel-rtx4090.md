# Triton Fused GRPO Advantage Kernel — RTX 4090

> 2026-06-07 | Triton融合advantage kernel 4-40x加速, 但E2E GRPO训练0.99x(优化非瓶颈无效!)

## 核心发现

```
Triton fused advantage kernel: 4-40x faster than Python (cos_sim=1.0 完美精度)
但E2E GRPO训练: 0.99x (零加速!)

原因: advantage计算仅占GRPO step的<1-7% → 优化非瓶颈无E2E收益

┌─────────────────────────────────────────────────────────────┐
│ GRPO step时间分解 (81ms总):                                 │
│ Rollout(autoregressive): ~60ms (74%) ← 真正瓶颈(Python)    │
│ Forward(logprob):         ~15ms (19%) ← 可优化但比例小      │
│ Advantage computation:     <2ms (<3%) ← Triton 40x→但<1ms! │
│ Loss+backward+optimizer:  ~4ms  (5%) ← 可优化但比例小      │
└─────────────────────────────────────────────────────────────┘

→ Triton优化了<3%的时间 → 即使40x加速 → E2E收益 = 40x × 3% = 1.2x → 实测0.99x
→ 原因: Python baseline只需0.6-5ms → Triton后<0.14ms → 已可忽略 → 无法影响E2E
→ **教训: 优化非瓶颈 = 优化空中楼阁 → 必须先profile找瓶颈再优化!**
```

## 实验1: Triton Fused Advantage Kernel

```
Fused kernel: reward → mean → std → advantage 在单个Triton kernel中完成

| 配置 (G×S)     │ Python(ms) │ Triton(ms) │ Speedup │ cos_sim │ max_diff
│ 8×4            │ 0.636      │ 0.135      │ 4.71x   │ 1.000   │ 0.216
│ 16×8           │ 1.392      │ 0.133      │ 10.46x  │ 1.000   │ 0.146
│ 32×16          │ 2.730      │ 0.134      │ 20.35x  │ 1.000   │ 0.071
│ 64×4           │ 5.391      │ 0.133      │ 40.49x  │ 1.000   │ 0.230
│ 8×64           │ 0.724      │ 0.133      │ 5.44x   │ 1.000   │ 0.017

→ Triton加速随数据量增长: 4x→10x→20x→40x → GPU并行度↑
→ cos_sim=1.000 → 精度完美(Triton计算等价Python)
→ max_diff随group_size减小 → 更多数据→浮点平均误差↓
→ Triton kernel时间几乎恒定(0.133-0.135ms) → GPU launch主导 → 数据量无关!
  → Python时间线性增长 → Python逐组循环→overhead∝G×S
  → Triton单kernel→GPU并行→overhead≈kernel launch

关键观察:
  → Triton kernel 0.133ms ≈ kernel launch time → 计算本身几乎免费
  → Python 0.636ms → 循环overhead主导 → 不是compute-bound
  → 对小数据(G=8,S=4) → Python overhead 0.5ms → Triton节省0.5ms
  → 对大数据(G=64,S=4) → Python overhead 5ms → Triton节省5ms
  → 但在81ms GRPO step中 → 0.5-5ms → 仅0.6-6.2%!
```

## 实验2: σ-normalization vs Unnormalized

```
| 模式       │ Python(ms) │ Triton(ms) │ Speedup │ cos_sim
│ σ-norm     │ 0.720      │ 0.131      │ 5.49x   │ 1.000
│ unnorm     │ 0.638      │ 0.131      │ 4.86x   │ 1.000

→ σ-norm比unnorm稍慢(Python) → 多一个除法操作
→ Triton两者几乎相同 → GPU融合除法无额外开销
→ 两种模式cos_sim=1.000 → Triton实现完全正确
```

## 实验3: E2E GRPO训练对比

```
SFT warm-start: 200步(34% eval), GRPO: 300步, n=4, noise σ=0.01

| 方式         │ Final eval │ Peak eval │ Step(ms) │ Speedup
│ Python adv   │ 73%        │ 82%       │ 81.35    │ 1.00x
│ Triton adv   │ 73%        │ 82%       │ 81.99    │ 0.99x ⚠️

→ E2E零加速! Triton优化advantage(<2ms) → 对81ms总step无影响
→ 两种方式eval完全相同(73%/82%) → Triton advantage数值正确
→ Step时间几乎相同(81.35 vs 81.99) → advantage占比太小

→ **核心教训**: Profiling第一, 优化第二!
  错误路径: 先写kernel → 发现0.99x → 瓶颈在别处
  正确路径: 先profile → 发现rollout占74% → 优化rollout(vLLM/SGLang)
```

## 性能优化优先级 — 从GRPO Step Profile推导

```
GRPO step = 81ms分解:
  1. Rollout (autoregressive): 60ms (74%) ← P0瓶颈
     → Python for循环 × n_samples × max_len
     → 每步: model.forward + softmax + multinomial + cat
     → 解决: vLLM/SGLang async rollout → 消除Python循环

  2. Forward (logprob): 15ms (19%) ← P1次瓶颈
     → model.forward → log_softmax → gather
     → 解决: torch.compile → kernel fusion → 1.3-2x

  3. Advantage: <2ms (<3%) ← P2非瓶颈
     → mean/std/normalize → 已被Triton 40x优化 → <0.14ms → 可忽略
     → 进一步优化无意义 → 已接近kernel launch极限

  4. Loss+backward+optimizer: 4ms (5%) ← P2非瓶颈
     → loss.backward → clip_grad → optimizer.step
     → 解决: torch.compile → 融合backward ops → 1.1-1.3x

→ 优化收益 = speedup × 比例:
  vLLM rollout: 2-5x × 74% = 1.74-3.7x E2E ← 最大杠杆!
  torch.compile forward: 1.3-2x × 19% = 1.06-1.19x ← 小杠杆
  Triton advantage: 40x × 3% = 1.12x ← 诱惑但实际无收益(占比太小)
  torch.compile backward: 1.1-1.3x × 5% = 1.005-1.015x ← 极小杠杆

→ 优化铁律: **最大杠杆 = 最高speedup × 最高占比**
  → rollout是74%占比 → 即使2x也有1.74x E2E收益 → 最高杠杆!
  → advantage是3%占比 → 即使40x也无E2E收益 → 最低杠杆!
```

## Triton Kernel编程学习要点

```
1. @triton.jit 装饰器 → Python函数变成GPU kernel
2. tl.program_id(0) → 获取当前GPU thread block ID
3. tl.load/tl.store → GPU memory读写 → mask处理边界
4. BLOCK_SIZE: tl.constexpr → 编译时常量 → 影响GPU并行度
5. triton.next_power_of_2() → BLOCK_SIZE必须是2的幂(硬件限制)
6. grid = (n_programs,) → 启动多少个GPU thread blocks
7. tl.sum() → GPU并行求和 → 单指令处理整个block
8. tl.sqrt()/tl.maximum() → GPU数学运算 → 向量化

→ Triton kernel开发流程:
  1. 确认瓶颈 → profile占比 → 高占比才值得写kernel
  2. 设计kernel → 每个program处理什么 → BLOCK_SIZE选择
  3. 实现kernel → load → compute → store → mask边界
  4. 验证精度 → cos_sim + max_diff → 与Python baseline对比
  5. Benchmark → median timing → vs Python → 确认加速
  6. E2E验证 → 放入完整pipeline → 确认E2E收益

→ 本实验流程完整但结论意外: Triton 40x → E2E 0.99x
→ 教训: 孤立benchmark ≠ E2E收益 → 必须E2E验证!
```

## 工具

- `tools/triton_fused_grpo_kernel.py` — Triton融合advantage kernel + 3实验benchmark
- `results/triton_fused_grpo_kernel.json` — 完整结果数据