# FlashAttention-3 Architecture Analysis

> 2026-06-07 | 基于 FlashAttention-3 论文 (Dao & Shah, 2024) + CUTLASS架构理解

## 一、FlashAttention三代演进

| 版本 | 年份 | 核心优化 | 架构目标 | 性能 |
|------|------|---------|---------|------|
| FA-1 | 2022 | Online softmax + tiling + SRAM | 通用GPU (Ampere+) | IO优化 2-4x |
| FA-2 | 2023 | 非split-K优化 + seq_parallel | Ampere/Ada (SM80+) | 2x over FA-1 |
| FA-3 | 2024 | **TMA + WGMMA + warpgroup softmax** | **Hopper (SM90)** | **1.5-2x over FA-2** |

**关键递进**: FA-1解决IO → FA-2优化计算 → FA-3利用新硬件(TMA/WGMMA)

## 二、SM90 三大新特性 (FA-3利用的全部)

### 1. TMA (Tensor Memory Accelerator)

```
传统 (SM80/89): 全局→共享内存
  多线程手动搬运: for(i; i<tile; i+=threads) smem[i] = gmem[offset+i]
  所有线程都参与搬运 → 计算暂停

TMA (SM90): 全局→共享内存
  1条指令: tma_copy(descriptor, smem_ptr, tile_coord)
  硬件自动搬运 → 只需1个线程发起 → 其他127线程继续计算!
  支持5D tensor descriptor → 完美适配attention tiling
```

**FA-3利用**: TMA异步加载Q/K/V tile → 释放warp group做softmax

### 2. WGMMA (Warpgroup Matrix Multiply-Accumulate)

```
传统 (SM80/89): HMMA — warp-level MMA (32 threads)
  32 threads协作 → 小tile → 低compute密度
  同步: 所有32 threads必须同时执行MMA

WGMMA (SM90): warpgroup MMA (128 threads = 4 warps)
  128 threads协作 → 大tile(256×128×64 FP16) → 高compute密度
  **异步**: 发出WGMMA指令 → 返回barrier → 等待时做其他工作!
  2x throughput over HMMA on H100 for FP16
```

**FA-3利用**: WGMMA计算Q×K^T → 等WGMMA完成时做softmax rescaling

### 3. Cluster (跨block同步+共享smem)

SM90允许同一Cluster内的多个CTA:
- 共享smem (跨block数据共享)
- 同步 (cluster_arrive + cluster_wait)
- TMA multicast (1次TMA广播到多个CTA)

**FA-3**: 不直接使用Cluster, 但与CUTLASS cooperative模式结合时可以利用

## 三、FA-3 三大核心技术

### 技术1: Asynchrony (异步重叠)

```
FA-2的pipeline (同步):
  Stage 1: cp.async load Q,K,V → smem   (异步加载)
  Stage 2: HMMA compute Q×K^T            (同步计算, 所有线程参与)
  Stage 3: softmax rescaling              (同步, 所有线程参与)
  → Stage 2和3不能与Stage 1重叠 (HMMA需要所有线程)

FA-3的pipeline (异步重叠):
  TMA load Q,K,V → smem     (异步, 1线程发起, 127线程自由)
  WGMMA compute Q×K^T        (异步, 128线程发起, 等待时做softmax)
  Softmax correction         (在WGMMA等待期间执行)
  → 三级pipeline完全重叠!
```

**关键**: WGMMA的异步性 → 发出指令后warpgroup可以做其他工作 → softmax被"藏"在WGMMA等待窗口内

### 技术2: Cooperative Interleaving (协作交织)

```
传统softmax: 循环结束后统一rescaling
  for k_tile in K_tiles:
      S = Q × K^T  (WGMMA)
  # 循环结束
  softmax_rescale_all(S)  ← 单独步骤, 阻塞WGMMA

FA-3交织: 每个WGMMA完成后立即rescaling
  for k_tile in K_tiles:
      # 等前一个WGMMA完成
      warpgroup_wait()  ← WGMMA barrier
      # 立即做softmax correction on registers
      rescale_accumulator(old_max, new_max)  ← 在WGMMA等待窗口内
      # 发起下一个WGMMA
      WGMMA(Q_tile_i, K_tile_j)  ← 异步
  → softmax correction被"交织"进WGMMA循环
```

**数学**: Online softmax rescaling公式:
- `O_new = O_old * exp(old_max - new_max) + P * exp(cur_max - new_max)`
- 这个rescaling只需要乘法和加法 → 很快 → 可以在WGMMA等待时完成

### 技术3: Warpgroup-level Cooperative Softmax (协作softmax)

```
FA-2: 每个线程独立计算自己的softmax部分
  每线程维护自己的row_max, row_sum → 寄存器压力高

FA-3: warpgroup协作计算softmax
  128 threads共享partial max/sum → 通过smem通信
  每warp计算局部max → smem写入 → warpgroup广播全局max
  register压力降低 (per-thread → per-warpgroup)
```

**优势**: 128 threads共享 → 每线程寄存器需求降低 → occupancy提高

## 四、FA-3 Pipeline 数据流

```
Thread Block结构 (H100):
  6 warpgroups per block (6×128 = 768 threads)
  - 2 WG: Q/K加载+softmax (Producer)
  - 2 WG: WGMMA Q×K^T (Consumer)
  - 2 WG: V加载+O存储

FA-3核心Pipeline (decode简化版):
┌─────────────────────────────────────────────────────────┐
│ Time ─────────────────────────────────────────────────→ │
│                                                         │
│ TMA:  [Load Q₀K₀] [Load Q₀K₁] [Load Q₀K₂] ...       │
│       ────────── ────────── ──────────                  │
│            ↓           ↓           ↓                    │
│ WGMMA:    [Q₀×K₀]   [Q₀×K₁]   [Q₀×K₂] ...           │
│          ──────── ──────── ────────                     │
│              ↓         ↓         ↓                      │
│ Softmax:  [rescale₀] [rescale₁] [rescale₂] ...        │
│          ──────── ──────── ────────                     │
│                                                         │
│ 三级重叠: TMA/WGMMA/Softmax同时活跃                      │
└─────────────────────────────────────────────────────────┘
```

## 五、FP8 Attention

FA-3还实现了FP8 attention (Hopper FP8 Tensor Core):

```
FP8 Attention流程:
  Q, K: FP8 (E4M3FN) → WGMMA FP8 input
  S = Q×K^T: FP8 accumulate → **问题: FP8精度不够做softmax!**

  解决方案: Online softmax in higher precision
  1. WGMMA: FP8 Q×FP8 K → FP32 accumulator (硬件级)
  2. Softmax: FP32 → 精确的max/exp/sum
  3. P = softmax(S): FP32
  4. WGMMA: FP32 P×FP8 V → FP32 O (最终输出)

  关键: softmax在FP32计算 → 防止FP8精度灾难性损失
  结果: ~1.2 TFLOPS (接近H100 FP8峰值)
```

**我的RTX 4090实测对照**: RTX 4090有基础FP8 Tensor Core但没有TMA/WGMMA → FP8 attention只能用HMMA → 吞吐远低于H100

## 六、性能对比

| 配置 | FA-2 (H100) | FA-3 (H100) | 加速 | H100峰值利用率 |
|------|------------|------------|------|--------------|
| FP16 seq=2K | ~190 TFLOPS | ~320 TFLOPS | **1.7x** | ~51% → ~51% |
| FP16 seq=8K | ~260 TFLOPS | ~490 TFLOPS | **1.9x** | ~42% → ~79% |
| FP8 seq=8K | — | ~580 TFLOPS | **2.2x** (vs FA2 FP16) | ~94% |

**FA-3达到75%理论峰值FP16 / 94%理论峰值FP8** → 这接近GEMM级别的效率!

## 七、与CUTLASS 3.x的关系

FA-3和CUTLASS 3.x共享同一套SM90基础设施:

| 共享特性 | CUTLASS GEMM | FlashAttention-3 |
|---------|-------------|-----------------|
| TMA异步加载 | PipelineTmaAsync | TMA load Q/K/V |
| WGMMA | cute::gemm(tiled_mma, A, B, C) | Q×K^T matmul |
| Pipeline | N-stage buffering | 3-stage overlapping |
| Warpgroup | 1 Load WG + 1 MMA WG | 2 Producer + 2 MMA WG |
| ClusterBarrier | ClusterTransactionBarrier | TMA barrier |
| Smem Reuse | Mainloop/Epilogue union | QKV/O smem reuse |

**FA-3 = CUTLASS GEMM pipeline思想 + attention特化(softmax交织)**

## 八、RTX 4090 (SM89) 能用什么?

RTX 4090不支持TMA/WGMMA → 只能用FA-2级别的优化:

- cp.async (SM80+) → 异步全局→共享内存 (但没有TMA的零线程开销)
- HMMA (warp-level MMA) → 32线程协作 (不如WGMMA 128线程)
- 没有Cluster → 不能跨block共享smem
- 有FP8 Tensor Core (基础) → 但没有WGMMA的异步FP8路径

**实测验证**: 我的RTX 4090实测发现FA在decode上比naive更慢(0.67-0.84x) → 因为Q=1时attention matrix太小, Tiling无IO收益 → FA-3的大tile WGMMA也不帮助小batch decode

## 九、实用结论

1. **FA-3是Hopper专用**: SM90 TMA+WGMMA是核心 → RTX 4090无法使用
2. **softmax交织是关键创新**: 将softmax藏在WGMMA等待窗口 → 达到75%峰值
3. **与CUTLASS共享基础设施**: PipelineTmaAsync + WGMMA + Smem Reuse
4. **FP8 attention可行**: softmax在FP32 → WGMMA在FP8 → 近峰值吞吐
5. **decode仍然memory-bound**: Q=1时FA-3加速有限(同FA-2) → 主要收益在prefill
6. **架构趋势**: GPU越来越异步 → kernel设计必须拥抱异步pipeline
7. **我的kernel发展路线**: SM89用warp shuffle+HMMA → 未来SM90用TMA+WGMMA

Sources:
- [FlashAttention-3: Fast and Accurate Attention with New Hardware Features (arXiv:2407.0868)](https://arxiv.org/abs/2407.0868)
- [FlashAttention GitHub Repository](https://github.com/Dao-AILab/flash-attention)
- [CUTLASS 3.x Repository](https://github.com/NVIDIA/cutlass)