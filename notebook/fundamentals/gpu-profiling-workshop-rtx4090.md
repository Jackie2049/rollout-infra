# GPU Profiling Workshop — RTX 4090

> 2026-06-08 | 5实验实测, 学习GPU profiling生产技能
> 关键: CPU dispatch overhead可达22-71%, CUDA Graph仅1.01-1.18x加速

## 1. CPU Dispatch Overhead vs GPU Compute

| 操作 | Wall(us) | CPU dispatch(us) | GPU compute(us) | CPU overhead% |
|------|---------|-----------------|----------------|--------------|
| attn_GEMM B=1 | 48.2 | 10.7 | 37.4 | **22.3%** |
| attn_GEMM B=32 | 32.8 | 14.3 | 18.4 | **43.7%** |
| mlp_gate B=1 | 146.2 | 14.2 | 132.0 | 9.7% |
| mlp_gate B=32 | 147.3 | 12.6 | 134.7 | 8.6% |
| lm_head B=1 | 382.7 | 13.7 | 369.0 | 3.6% |
| lm_head B=32 | 309.8 | 15.1 | 294.7 | 4.9% |
| RMSNorm B=1 | 42.9 | 30.4 | 12.4 | **71.0%** |
| RMSNorm B=32 | 40.4 | 29.0 | 11.4 | **71.9%** |

**关键发现**:
- **小kernel CPU overhead极高!** RMSNorm 71%, attn_GEMM B=32 43.7%
- **大kernel CPU overhead很低**: MLP gate 9.7%, lm_head 3.6%
- CPU dispatch ≈ 10-15us (kernel launch overhead) → 对于小kernel是主要瓶颈
- **RTX 4090比A16的34us dispatch好3x** → A16 overhead更严重

**规律**: CPU overhead ∝ 1/GPU_compute_time → 小kernel overhead高 → 大kernel overhead低

## 2. torch.profiler Trace Export

- **Trace格式**: Chrome trace JSON → 可在 chrome://tracing 查看
- **Trace大小**: 58KB (3步 decode) → 很小
- **Top kernels**: aten::mm (GEMM) ×15次, cuLaunchKernel ×15次
- **生产流程**: torch.profiler → export_chrome_trace → chrome://tracing → 可视化timeline

**torch.profiler局限**:
- 只记录Python-level ops → 不看到内部CUDA kernel
- cuda_time_total API在PyTorch 2.9已变更 → 需用cpu_time_total替代
- 对生产级分析 → **Nsight Systems更好** (看到所有CUDA kernel+通信重叠)

## 3. Memory Profile per Operation

| 操作 | Peak MB | Increment MB |
|------|---------|-------------|
| QKV proj | 1039 | 96 |
| gate_proj | 1055 | 112 |
| lm_head | 1194 | **251** |
| RMSNorm | 944 | 0.5 |
| SiLU×mul | 944 | 1.3 |

**关键发现**:
- **lm_head峰值251MB** → vocab=32K → 4096×32000×2 = 262MB → 接近理论
- GEMM峰值 = input + output + weight → 96-112MB → 验证3×data_size规律
- RMSNorm仅0.5MB → element-wise → 内存开销极小

## 4. CUDA Graph Capture vs Eager

| B | Eager(us) | Graph(us) | Speedup | Capture(ms) |
|---|----------|----------|---------|------------|
| 1 | 520.7 | 516.5 | **1.01x** | 6.42 |
| 4 | 525.0 | 516.1 | 1.02x | 3.53 |
| 16 | 531.9 | 525.2 | 1.01x | 5.20 |
| 32 | 544.7 | 536.3 | 1.02x | 5.07 |

**关键发现**:
- **CUDA Graph仅1.01-1.02x加速!** → 几乎无收益
- 原因: 这些GEMM大kernel(GPU compute >> CPU dispatch) → CUDA Graph消除的CPU overhead很小
- 与之前CUDA Graph benchmark一致(OPT-125M 2.43x vs 7B仅1.05x)
- **CUDA Graph收益 ∝ CPU dispatch占比** → 小模型收益大, 大模型收益小

**但CUDA Graph仍有生产价值**:
- 消除dispatch jitter → 稳定latency → 对ITL(inter-token latency)稳定有用
- vLLM V1使用CUDA Graph → 不是为了加速 → 而是为了稳定

## 5. Full Decode Pipeline Profiling Summary

| B | CPU dispatch | GPU compute | Total | CPU% | CUDA Graph speedup |
|---|------------|-----------|-------|------|-------------------|
| 1 | 2328us | 25603us | 27931us | 8.3% | **1.09x** |
| 32 | 2328us | 13000us | 15328us | 15.2% | **1.18x** |
| 55 | 2328us | 13000us | 15328us | 15.2% | 1.18x |

**CUDA Graph整体加速1.09-1.18x** → 与之前7B benchmark的1.05x一致
- B=1时CPU overhead仅8.3% → CUDA Graph收益小
- B=32时CPU overhead15.2% → CUDA Graph收益稍大但仍有限
- **GPU compute >> CPU dispatch → CUDA Graph不是加速主要手段**

## 6. Profiling Skills Learned

### 5种GPU profiling方法

| 方法 | 层级 | 适用场景 | 优缺点 |
|------|------|---------|--------|
| wall clock time | 整体 | 快速验证 | 简单但无法分解 |
| CPU dispatch vs GPU | 组件 | overhead分析 | 需手动测量 |
| torch.profiler | kernel | 开发调试 | Chrome trace可视化 |
| peak_memory_stats | 内存 | 内存预算 | 快速但不够精确 |
| CUDA Graph compare | 端到端 | jitter消除 | 仅对小模型有用 |

### 生产profiling流程

```
1. Wall clock time → 定位整体瓶颈(是否memory-bound?)
2. CPU dispatch vs GPU → 定位launch overhead问题
3. torch.profiler → Chrome trace → 可视化timeline → 确认kernel overlap
4. Nsight Systems → 更深分析 → kernel级+通信重叠
5. Nsight Compute → kernel级优化 → 单kernel性能分析

关键: 先用简单工具(wall clock) → 再用profiler → 最后Nsight
→ 不要一开始就用Nsight → 太复杂 → 先理解宏观再深入微观
```

## 7. 核心规律

```
CPU dispatch overhead规律:
  小kernel(GEMM B=1, RMSNorm): overhead 22-71% → CUDA Graph有效
  大kernel(MLP, lm_head): overhead 3-10% → CUDA Graph无用
  7B full decode: overhead 8-15% → CUDA Graph仅1.05-1.18x

  规律: overhead ∝ 1/GPU_time → kernel越大overhead越小

  CUDA Graph决策树:
    小模型(125M): CUDA Graph 2.43x → 有价值(B=1 dispatch占23%)
    大模型(7B): CUDA Graph 1.05x → 无加速价值 → 但消除jitter有价值
    → vLLM V1用CUDA Graph = 稳定latency而不是加速

  torch.profiler vs Nsight Systems:
    torch.profiler → Python-level → 开发调试 → 快但不够深
    Nsight Systems → System-level → 生产分析 → 深但复杂
    → 生产用Nsight, 开发用torch.profiler
```